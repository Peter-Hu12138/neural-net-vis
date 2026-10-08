import type { Network } from '../nn/network';
import type { LayerSpec } from '../nn/types';

/**
 * Analysis jobs run in their own worker, beside the trainer. A job is a generator: it does a small
 * piece of work per `next()` call and yields its progress, so long analyses can be time-sliced,
 * interleaved and cancelled. Its return value is the result.
 */

export interface Progress {
  done: number;
  total: number;
  /** Optional intermediate result (for example t-SNE coordinates mid-optimisation). */
  partial?: unknown;
}

/** What every job receives: a private network copy holding the requested weights, plus the test set. */
export interface JobContext {
  net: Network;
  spec: LayerSpec[];
  testX: Uint8Array;
  testY: Uint8Array;
  /** Test image `i` as floats in [0, 1]. Writes into `out` when given. */
  image(i: number, out?: Float32Array): Float32Array;
}

export type Job<P = any, R = any> = (ctx: JobContext, params: P) => Generator<Progress, R, void>;

export type ToAnalyzer =
  | { type: 'data'; testX: Uint8Array; testY: Uint8Array }
  | { type: 'run'; id: number; channel: string; kind: string; params: unknown; spec: LayerSpec[]; weights: Float32Array[] }
  | { type: 'cancel'; channel: string };

export type FromAnalyzer =
  | { type: 'ready' }
  | { type: 'progress'; id: number; progress: Progress }
  | { type: 'result'; id: number; result: unknown }
  | { type: 'error'; id: number; message: string };
