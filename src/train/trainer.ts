import { Network, argmax } from '../nn/network';
import { Optimizer } from '../nn/optim';
import { Rng } from '../nn/rng';
import type { Hyper } from '../nn/types';
import {
  CUSTOM_REPEAT,
  EVALS_PER_EPOCH,
  POINTS_PER_EPOCH,
  type CustomSample,
  type DataPayload,
  type EvalPoint,
  type FromTrainer,
  type ToTrainer,
  type TrainPoint,
} from './protocol';

interface EvalJob {
  i: number;
  loss: number;
  correct: number;
  confusion: number[];
  epoch: number;
  step: number;
}

/**
 * Owns the training copy of the network. Runs in a Web Worker normally; the same class runs
 * on the main thread (with shorter time slices) when workers are unavailable.
 */
export class Trainer {
  private data: DataPayload | null = null;
  private net: Network | null = null;
  private opt: Optimizer | null = null;
  private hyper: Hyper = { lr: 0.001, batchSize: 32, optimizer: 'adam' };
  private version = 0;
  private custom: CustomSample[] = [];
  private rng = new Rng(12345);
  private order = new Int32Array(0);
  private cursor = 0;
  private epoch = 0;
  private step = 0;
  private seen = 0;
  private running = false;
  private stopAtEpochEnd = false;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private x = new Float32Array(0);
  private frozen: boolean[] = [];

  private winLoss = 0;
  private winAcc = 0;
  private winN = 0;
  private nextPointStep = 0;
  private nextEvalEpoch = 0;
  private lastPointAt = -Infinity;
  private lastEvalAt = -Infinity;
  private evalJob: EvalJob | null = null;
  private points: TrainPoint[] = [];
  private evals: EvalPoint[] = [];

  private lastStatus = 0;
  private lastWeights = 0;
  private rate = 0;
  /** Speed cap (samples per second) and the samples it currently allows (a token bucket). */
  private maxRate: number | null = null;
  private allowance = 0;
  private lastTick = 0;
  private capSeen = 0;
  private capFrom = 0;

  /**
   * `pace` sets wall-clock floors between recorded curve points and between test-set evaluations.
   * Tiny datasets (a few hundred points) finish an epoch in milliseconds; without the floors the
   * curves would get thousands of points per second. Tests use the default 0 for exact counts.
   */
  constructor(
    private emit: (msg: FromTrainer, transfer?: Transferable[]) => void,
    private sliceMs = 40,
    private pace: { pointMs: number; evalMs: number } = { pointMs: 0, evalMs: 0 },
  ) {}

  handle(msg: ToTrainer): void {
    switch (msg.type) {
      case 'data':
        this.data = msg.data;
        this.x = new Float32Array(msg.data.inputSize);
        this.rebuildOrder();
        this.startEval();
        this.schedule();
        break;
      case 'model':
        this.version = msg.version;
        this.net = new Network(msg.arch, 0);
        this.net.setWeights(msg.weights);
        this.frozen = msg.frozen.slice();
        this.hyper = msg.hyper;
        this.opt = new Optimizer(this.net, msg.hyper.optimizer, msg.hyper.lr);
        this.running = this.stopAtEpochEnd = false;
        this.epoch = this.step = this.seen = this.cursor = 0;
        this.winLoss = this.winAcc = this.winN = 0;
        this.points = [];
        this.evals = [];
        this.nextPointStep = 0;
        this.nextEvalEpoch = 0;
        this.evalJob = null;
        this.rebuildOrder();
        this.startEval();
        this.sendStatus(true);
        this.schedule();
        break;
      case 'hyper': {
        const changedOpt = msg.hyper.optimizer !== this.hyper.optimizer;
        this.hyper = msg.hyper;
        if (this.net && (changedOpt || !this.opt)) this.opt = new Optimizer(this.net, msg.hyper.optimizer, msg.hyper.lr);
        else if (this.opt) this.opt.lr = msg.hyper.lr;
        break;
      }
      case 'weights':
        this.net?.setWeights(msg.weights);
        this.startEval();
        this.schedule();
        break;
      case 'frozen':
        this.frozen = msg.frozen.slice();
        break;
      case 'speed':
        this.maxRate = msg.samplesPerSec;
        this.allowance = 0;
        this.lastTick = this.capFrom = performance.now();
        this.capSeen = 0;
        this.rate = 0;
        break;
      case 'custom':
        this.custom = msg.samples;
        this.rebuildOrder(true);
        break;
      case 'play':
        this.running = true;
        this.stopAtEpochEnd = false;
        this.allowance = 0;
        this.lastTick = this.capFrom = performance.now();
        this.capSeen = 0;
        this.sendStatus(true);
        this.schedule();
        break;
      case 'pause':
        this.running = false;
        this.stopAtEpochEnd = false;
        this.flush();
        break;
      case 'step':
        if (!this.ready()) return;
        this.running = false;
        this.trainBatch();
        this.flush();
        this.schedule();
        break;
      case 'epoch':
        this.running = true;
        this.stopAtEpochEnd = true;
        this.sendStatus(true);
        this.schedule();
        break;
    }
  }

  /** Data and network agree on input size and classes (they arrive separately on a dataset switch). */
  private ready(): boolean {
    const d = this.data;
    const net = this.net;
    return !!(d && net && this.opt && net.inputSize === d.inputSize && net.classes === d.classes);
  }

  private get poolSize(): number {
    return (this.data?.trainY.length ?? 0) + this.custom.length * CUSTOM_REPEAT;
  }

  private rebuildOrder(keepCursor = false): void {
    const n = this.poolSize;
    this.order = new Int32Array(n);
    for (let i = 0; i < n; i++) this.order[i] = i;
    this.rng.shuffle(this.order);
    if (!keepCursor || this.cursor >= n) this.cursor = 0;
  }

  private stepsPerEpoch(): number {
    return Math.max(1, Math.ceil(this.poolSize / this.hyper.batchSize));
  }

  /** Writes sample `idx` of the training pool into this.x and returns its label. */
  private load(idx: number): number {
    const d = this.data!;
    const n = d.trainY.length;
    let src: Uint8Array | Float32Array;
    let off: number;
    let y: number;
    if (idx < n || this.custom.length === 0) {
      const i = idx % n;
      src = d.trainX;
      off = i * d.inputSize;
      y = d.trainY[i];
    } else {
      const s = this.custom[(idx - n) % this.custom.length];
      src = s.x;
      off = 0;
      y = s.y;
    }
    const x = this.x;
    const len = d.inputSize;
    const scale = d.scale;
    for (let j = 0; j < len; j++) x[j] = src[off + j] * scale;
    return y;
  }

  private trainBatch(): void {
    const net = this.net!;
    const B = this.hyper.batchSize;
    net.zeroGrad();
    let loss = 0;
    let correct = 0;
    for (let k = 0; k < B; k++) {
      if (this.cursor >= this.order.length) this.endEpoch();
      const y = this.load(this.order[this.cursor++]);
      const p = net.forward(this.x);
      if (argmax(p) === y) correct++;
      loss += net.backward(y, false, this.frozen);
    }
    this.opt!.step(1 / B, this.frozen);
    if (this.cursor >= this.order.length) this.endEpoch();
    this.step++;
    this.seen += B;
    this.winLoss += loss;
    this.winAcc += correct;
    this.winN += B;
    const now = this.pace.pointMs || this.pace.evalMs ? performance.now() : 0;
    if (this.step >= this.nextPointStep && now - this.lastPointAt >= this.pace.pointMs) {
      this.lastPointAt = now;
      this.points.push({ epoch: this.epochFraction(), step: this.step, loss: this.winLoss / this.winN, acc: this.winAcc / this.winN });
      this.winLoss = this.winAcc = this.winN = 0;
      this.nextPointStep = this.step + Math.max(1, Math.round(this.stepsPerEpoch() / POINTS_PER_EPOCH));
    }
    if (this.epochFraction() >= this.nextEvalEpoch && !this.evalJob && now - this.lastEvalAt >= this.pace.evalMs) {
      this.lastEvalAt = now;
      this.startEval();
    }
  }

  private endEpoch(): void {
    this.epoch++;
    this.cursor = 0;
    this.rebuildOrder();
    if (this.stopAtEpochEnd) {
      this.running = false;
      this.stopAtEpochEnd = false;
    }
  }

  private epochFraction(): number {
    return this.epoch + this.cursor / Math.max(1, this.order.length);
  }

  private startEval(): void {
    if (!this.ready()) return;
    const ef = this.epochFraction();
    this.evalJob = { i: 0, loss: 0, correct: 0, confusion: new Array(this.data!.classes * this.data!.classes).fill(0), epoch: ef, step: this.step };
    this.nextEvalEpoch = (Math.floor(ef * EVALS_PER_EPOCH + 1e-9) + 1) / EVALS_PER_EPOCH;
  }

  /** Evaluates up to `budget` test samples of the pending evaluation. */
  private evalChunk(budget: number): void {
    const job = this.evalJob!;
    const d = this.data!;
    const net = this.net!;
    const n = d.testY.length;
    const end = Math.min(n, job.i + budget);
    for (; job.i < end; job.i++) {
      const off = job.i * d.inputSize;
      for (let j = 0; j < d.inputSize; j++) this.x[j] = d.testX[off + j] * d.scale;
      const y = d.testY[job.i];
      const p = net.forward(this.x);
      const pred = argmax(p);
      job.loss += -Math.log(Math.max(p[y], 1e-12));
      if (pred === y) job.correct++;
      job.confusion[y * d.classes + pred]++;
    }
    if (job.i >= n) {
      this.evals.push({ epoch: job.epoch, step: job.step, loss: job.loss / n, acc: job.correct / n, confusion: job.confusion });
      this.evalJob = null;
    }
  }

  private schedule(delay = 0): void {
    if (this.timer !== null) return;
    if (!this.ready() || (!this.running && !this.evalJob)) return;
    this.timer = setTimeout(this.loop, delay);
  }

  /** With a speed cap: milliseconds until the cap allows the next batch (0 = now). */
  private capWait(): number {
    if (this.maxRate === null || !this.running || this.evalJob) return 0;
    const need = this.hyper.batchSize - this.allowance;
    return need <= 0 ? 0 : Math.min(100, Math.max(4, Math.ceil((need / this.maxRate) * 1000)));
  }

  private loop = (): void => {
    this.timer = null;
    if (!this.ready()) return;
    const t0 = performance.now();
    const seen0 = this.seen;
    if (this.maxRate !== null) {
      // Refill the bucket; never bank more than a tenth of a second (or one batch) of samples.
      this.allowance = Math.min(this.allowance + ((t0 - this.lastTick) / 1000) * this.maxRate, Math.max(this.hyper.batchSize, this.maxRate / 10));
      this.lastTick = t0;
    }
    while (performance.now() - t0 < this.sliceMs) {
      if (this.evalJob) this.evalChunk(40);
      else if (this.running) {
        if (this.maxRate !== null && this.allowance < this.hyper.batchSize) break;
        const before = this.seen;
        this.trainBatch();
        if (this.maxRate !== null) this.allowance -= this.seen - before;
      } else break;
    }
    const dt = performance.now() - t0;
    if (this.maxRate !== null) {
      // Capped: measure over wall-clock time, idle waits included.
      this.capSeen += this.seen - seen0;
      const wall = t0 + dt - this.capFrom;
      if (wall >= 500) {
        const r = (this.capSeen * 1000) / wall;
        this.rate = this.rate ? 0.5 * this.rate + 0.5 * r : r;
        this.capSeen = 0;
        this.capFrom = t0 + dt;
      }
    } else if (this.seen > seen0 && dt > 0) {
      const r = ((this.seen - seen0) * 1000) / dt;
      this.rate = this.rate ? 0.8 * this.rate + 0.2 * r : r;
    }
    const now = performance.now();
    if (!this.running && !this.evalJob) {
      this.flush();
      return;
    }
    if (now - this.lastStatus > 120) this.sendStatus(false);
    // Small networks (point datasets) are cheap to copy: send them often so the boundary animates.
    const every = this.net && this.net.paramCount < 20_000 ? 100 : 300;
    if (now - this.lastWeights > every && this.running) this.sendWeights();
    this.schedule(this.capWait());
  };

  private flush(): void {
    this.sendStatus(true);
    this.sendWeights();
  }

  private sendStatus(force: boolean): void {
    const now = performance.now();
    if (!force && now - this.lastStatus < 120) return;
    this.lastStatus = now;
    this.emit({
      type: 'status',
      status: {
        version: this.version,
        running: this.running,
        epoch: this.epoch,
        step: this.step,
        seen: this.seen,
        epochFraction: this.epochFraction(),
        samplesPerSec: this.running ? this.rate : 0,
      },
    });
    if (this.points.length || this.evals.length) {
      this.emit({ type: 'metrics', version: this.version, points: this.points, evals: this.evals });
      this.points = [];
      this.evals = [];
    }
  }

  private sendWeights(): void {
    if (!this.net) return;
    this.lastWeights = performance.now();
    const weights = this.net.getWeights();
    this.emit({ type: 'weights', version: this.version, step: this.step, weights }, weights.map((w) => w.buffer));
  }
}
