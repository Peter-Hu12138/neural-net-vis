import { store } from '../store';
import { Analyzer } from './analyzer';
import type { FromAnalyzer, Progress, ToAnalyzer } from './protocol';
import { registry } from './registry';

/** Rejection reason when a newer request on the same channel replaced this one. */
export class Superseded extends Error {
  constructor() {
    super('superseded');
  }
}

export const isSuperseded = (e: unknown): e is Superseded => e instanceof Superseded;

interface Waiter {
  channel: string;
  resolve: (r: unknown) => void;
  reject: (e: unknown) => void;
  onProgress?: (p: Progress) => void;
}

/**
 * Page-side handle on the analysis worker. If the worker cannot start, the same Analyzer runs on
 * the main thread in short slices.
 */
export class AnalysisClient {
  private worker: Worker | null = null;
  private local: Analyzer | null = null;
  private ready = false;
  private queue: ToAnalyzer[] = [];
  private waiters = new Map<number, Waiter>();
  private latest = new Map<string, number>();
  private nextId = 1;
  mode: 'worker' | 'main-thread' | 'starting' = 'starting';

  constructor() {
    try {
      this.worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
      this.worker.onmessage = (e: MessageEvent<FromAnalyzer>) => {
        if (e.data.type === 'ready') {
          this.ready = true;
          this.mode = 'worker';
          for (const m of this.queue) this.worker!.postMessage(m);
          this.queue = [];
          return;
        }
        this.receive(e.data);
      };
      this.worker.onerror = () => {
        if (!this.ready) this.fallback();
      };
      setTimeout(() => {
        if (!this.ready) this.fallback();
      }, 4000);
    } catch {
      this.fallback();
    }
  }

  private fallback(): void {
    if (this.local) return;
    this.worker?.terminate();
    this.worker = null;
    this.local = new Analyzer((m) => setTimeout(() => this.receive(m), 0), registry, 10);
    this.ready = true;
    this.mode = 'main-thread';
    for (const m of this.queue) this.local.handle(m);
    this.queue = [];
  }

  private post(msg: ToAnalyzer): void {
    if (!this.ready) this.queue.push(msg);
    else if (this.worker) this.worker.postMessage(msg);
    else this.local!.handle(msg);
  }

  private receive(m: FromAnalyzer): void {
    if (m.type === 'ready') return;
    const w = this.waiters.get(m.id);
    if (!w) return;
    if (m.type === 'progress') {
      w.onProgress?.(m.progress);
      return;
    }
    this.waiters.delete(m.id);
    if (this.latest.get(w.channel) === m.id) this.latest.delete(w.channel);
    if (m.type === 'result') w.resolve(m.result);
    else w.reject(new Error(m.message));
  }

  setData(d: { testX: Uint8Array | Float32Array; testY: Uint8Array; inputSize: number; scale: number; classes: number }): void {
    this.post({ type: 'data', testX: d.testX, testY: d.testY, inputSize: d.inputSize, scale: d.scale, classes: d.classes });
  }

  /**
   * Runs analysis `kind` with the network's current architecture and weights. A newer request on
   * the same channel cancels this one, which then rejects with `Superseded`.
   */
  run<R>(channel: string, kind: string, params: unknown, onProgress?: (p: Progress) => void): Promise<R> {
    this.cancel(channel);
    const id = this.nextId++;
    this.latest.set(channel, id);
    const promise = new Promise<R>((resolve, reject) => {
      this.waiters.set(id, { channel, resolve: resolve as (r: unknown) => void, reject, onProgress });
    });
    this.post({ type: 'run', id, channel, kind, params, arch: store.arch, weights: store.net.getWeights() });
    return promise;
  }

  cancel(channel: string): void {
    const prev = this.latest.get(channel);
    if (prev === undefined) return;
    this.latest.delete(channel);
    const w = this.waiters.get(prev);
    this.waiters.delete(prev);
    w?.reject(new Superseded());
    this.post({ type: 'cancel', channel });
  }
}

export const analysis = new AnalysisClient();
