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
  private x = new Float32Array(784);

  private winLoss = 0;
  private winAcc = 0;
  private winN = 0;
  private nextPointStep = 0;
  private nextEvalEpoch = 0;
  private evalJob: EvalJob | null = null;
  private points: TrainPoint[] = [];
  private evals: EvalPoint[] = [];

  private lastStatus = 0;
  private lastWeights = 0;
  private rate = 0;

  constructor(
    private emit: (msg: FromTrainer, transfer?: Transferable[]) => void,
    private sliceMs = 40,
  ) {}

  handle(msg: ToTrainer): void {
    switch (msg.type) {
      case 'data':
        this.data = msg.data;
        this.rebuildOrder();
        this.startEval();
        this.schedule();
        break;
      case 'model':
        this.version = msg.version;
        this.net = new Network(msg.spec, 0);
        this.net.setWeights(msg.weights);
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
      case 'custom':
        this.custom = msg.samples;
        this.rebuildOrder(true);
        break;
      case 'play':
        this.running = true;
        this.stopAtEpochEnd = false;
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

  private ready(): boolean {
    return !!(this.data && this.net && this.opt);
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
    let src: Uint8Array;
    let off: number;
    let y: number;
    if (idx < n || this.custom.length === 0) {
      const i = idx % n;
      src = d.trainX;
      off = i * 784;
      y = d.trainY[i];
    } else {
      const s = this.custom[(idx - n) % this.custom.length];
      src = s.x;
      off = 0;
      y = s.y;
    }
    const x = this.x;
    for (let j = 0; j < 784; j++) x[j] = src[off + j] / 255;
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
      loss += net.backward(y);
    }
    this.opt!.step(1 / B);
    if (this.cursor >= this.order.length) this.endEpoch();
    this.step++;
    this.seen += B;
    this.winLoss += loss;
    this.winAcc += correct;
    this.winN += B;
    if (this.step >= this.nextPointStep) {
      this.points.push({ epoch: this.epochFraction(), step: this.step, loss: this.winLoss / this.winN, acc: this.winAcc / this.winN });
      this.winLoss = this.winAcc = this.winN = 0;
      this.nextPointStep = this.step + Math.max(1, Math.round(this.stepsPerEpoch() / POINTS_PER_EPOCH));
    }
    if (this.epochFraction() >= this.nextEvalEpoch) this.startEval();
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
    this.evalJob = { i: 0, loss: 0, correct: 0, confusion: new Array(100).fill(0), epoch: ef, step: this.step };
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
      const off = job.i * 784;
      for (let j = 0; j < 784; j++) this.x[j] = d.testX[off + j] / 255;
      const y = d.testY[job.i];
      const p = net.forward(this.x);
      const pred = argmax(p);
      job.loss += -Math.log(Math.max(p[y], 1e-12));
      if (pred === y) job.correct++;
      job.confusion[y * 10 + pred]++;
    }
    if (job.i >= n) {
      this.evals.push({ epoch: job.epoch, step: job.step, loss: job.loss / n, acc: job.correct / n, confusion: job.confusion });
      this.evalJob = null;
    }
  }

  private schedule(): void {
    if (this.timer !== null) return;
    if (!this.ready() || (!this.running && !this.evalJob)) return;
    this.timer = setTimeout(this.loop, 0);
  }

  private loop = (): void => {
    this.timer = null;
    if (!this.ready()) return;
    const t0 = performance.now();
    const seen0 = this.seen;
    while (performance.now() - t0 < this.sliceMs) {
      if (this.evalJob) this.evalChunk(40);
      else if (this.running) this.trainBatch();
      else break;
    }
    const dt = performance.now() - t0;
    if (this.seen > seen0 && dt > 0) {
      const r = ((this.seen - seen0) * 1000) / dt;
      this.rate = this.rate ? 0.8 * this.rate + 0.2 * r : r;
    }
    const now = performance.now();
    if (!this.running && !this.evalJob) {
      this.flush();
      return;
    }
    if (now - this.lastStatus > 120) this.sendStatus(false);
    if (now - this.lastWeights > 300 && this.running) this.sendWeights();
    this.schedule();
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
