import type { FromTrainer, ToTrainer } from './protocol';
import { Trainer } from './trainer';

/**
 * Talks to the training worker. If the worker cannot start (blocked by a sandbox or CSP),
 * it falls back to running the same Trainer on the main thread in short time slices.
 */
export class TrainerClient {
  private worker: Worker | null = null;
  private local: Trainer | null = null;
  private ready = false;
  private queue: ToTrainer[] = [];
  mode: 'worker' | 'main-thread' | 'starting' = 'starting';

  constructor(private onMessage: (m: FromTrainer) => void) {
    try {
      this.worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
      this.worker.onmessage = (e: MessageEvent<FromTrainer>) => {
        if (e.data.type === 'ready') {
          this.ready = true;
          this.mode = 'worker';
          for (const m of this.queue) this.worker!.postMessage(m);
          this.queue = [];
          return;
        }
        this.onMessage(e.data);
      };
      this.worker.onerror = () => {
        if (!this.ready) this.fallback();
      };
      // A worker that never reports in is treated as blocked.
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
    this.local = new Trainer((m) => setTimeout(() => this.onMessage(m), 0), 12);
    this.ready = true;
    this.mode = 'main-thread';
    for (const m of this.queue) this.local.handle(m);
    this.queue = [];
  }

  post(msg: ToTrainer): void {
    if (!this.ready) this.queue.push(msg);
    else if (this.worker) this.worker.postMessage(msg);
    else this.local!.handle(msg);
  }
}
