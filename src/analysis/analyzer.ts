import { Network } from '../nn/network';
import type { FromAnalyzer, Job, JobContext, Progress, ToAnalyzer } from './protocol';

interface Running {
  id: number;
  gen: Generator<Progress, unknown, void>;
  lastPost: number;
}

type Pending = Extract<ToAnalyzer, { type: 'run' }>;

/**
 * Runs analysis jobs in time slices. One job per channel: a new request on a channel replaces the
 * one still running there. Jobs on different channels take turns. Each job gets its own network
 * copy, so a job started at training step N keeps using step N's weights to the end.
 */
export class Analyzer {
  private data: Extract<ToAnalyzer, { type: 'data' }> | null = null;
  private active = new Map<string, Running>();
  private waiting = new Map<string, Pending>();
  private timer: ReturnType<typeof setTimeout> | null = null;

  constructor(
    private emit: (msg: FromAnalyzer) => void,
    private registry: Record<string, Job>,
    private sliceMs = 30,
  ) {}

  handle(msg: ToAnalyzer): void {
    switch (msg.type) {
      case 'data':
        this.data = msg;
        for (const p of this.waiting.values()) this.start(p);
        this.waiting.clear();
        break;
      case 'run':
        this.active.delete(msg.channel);
        if (!this.data) this.waiting.set(msg.channel, msg);
        else this.start(msg);
        break;
      case 'cancel':
        this.active.delete(msg.channel);
        this.waiting.delete(msg.channel);
        break;
    }
  }

  private start(m: Pending): void {
    const job = this.registry[m.kind];
    if (!job) {
      this.emit({ type: 'error', id: m.id, message: `Unknown analysis "${m.kind}"` });
      return;
    }
    const d = this.data!;
    if (m.arch.input.c * m.arch.input.h * m.arch.input.w !== d.inputSize || m.arch.classes !== d.classes) {
      this.emit({ type: 'error', id: m.id, message: 'The network does not match the loaded dataset.' });
      return;
    }
    const net = new Network(m.arch, 0);
    net.setWeights(m.weights);
    const { testX, testY, inputSize, scale, classes } = d;
    const ctx: JobContext = {
      net,
      arch: m.arch,
      spec: m.arch.layers,
      testX,
      testY,
      inputSize,
      scale,
      classes,
      image(i, out = new Float32Array(inputSize)) {
        const off = i * inputSize;
        for (let j = 0; j < inputSize; j++) out[j] = testX[off + j] * scale;
        return out;
      },
    };
    try {
      this.active.set(m.channel, { id: m.id, gen: job(ctx, m.params), lastPost: 0 });
    } catch (e) {
      this.emit({ type: 'error', id: m.id, message: e instanceof Error ? e.message : String(e) });
    }
    this.schedule();
  }

  private schedule(): void {
    if (this.timer === null && this.active.size) this.timer = setTimeout(this.loop, 0);
  }

  private loop = (): void => {
    this.timer = null;
    const t0 = performance.now();
    while (this.active.size && performance.now() - t0 < this.sliceMs) {
      for (const [channel, job] of [...this.active]) {
        if (this.active.get(channel) !== job) continue; // replaced or cancelled meanwhile
        let r: IteratorResult<Progress, unknown>;
        try {
          r = job.gen.next();
        } catch (e) {
          this.active.delete(channel);
          this.emit({ type: 'error', id: job.id, message: e instanceof Error ? e.message : String(e) });
          continue;
        }
        if (r.done) {
          this.active.delete(channel);
          this.emit({ type: 'result', id: job.id, result: r.value });
        } else {
          const now = performance.now();
          if (now - job.lastPost > 100) {
            job.lastPost = now;
            this.emit({ type: 'progress', id: job.id, progress: r.value });
          }
        }
        if (performance.now() - t0 >= this.sliceMs) break;
      }
    }
    this.schedule();
  };
}
