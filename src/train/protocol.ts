import type { Hyper, LayerSpec } from '../nn/types';

export interface DataPayload {
  trainX: Uint8Array;
  trainY: Uint8Array;
  testX: Uint8Array;
  testY: Uint8Array;
}

export interface CustomSample {
  id: number;
  x: Uint8Array;
  y: number;
}

export interface TrainPoint {
  epoch: number;
  step: number;
  loss: number;
  acc: number;
}

export interface EvalPoint {
  epoch: number;
  step: number;
  loss: number;
  acc: number;
  /** 10×10 counts, row = true label, column = prediction. */
  confusion: number[];
}

export interface Status {
  version: number;
  running: boolean;
  epoch: number;
  step: number;
  seen: number;
  epochFraction: number;
  samplesPerSec: number;
}

export type ToTrainer =
  | { type: 'data'; data: DataPayload }
  | { type: 'model'; version: number; spec: LayerSpec[]; weights: Float32Array[]; hyper: Hyper }
  | { type: 'hyper'; hyper: Hyper }
  | { type: 'weights'; weights: Float32Array[] }
  | { type: 'custom'; samples: CustomSample[] }
  | { type: 'play' }
  | { type: 'pause' }
  | { type: 'step' }
  | { type: 'epoch' };

export type FromTrainer =
  | { type: 'ready' }
  | { type: 'status'; status: Status }
  | { type: 'metrics'; version: number; points: TrainPoint[]; evals: EvalPoint[] }
  | { type: 'weights'; version: number; step: number; weights: Float32Array[] };

/** Each custom image is shown this many times per epoch so a handful of them still matter. */
export const CUSTOM_REPEAT = 10;
export const EVALS_PER_EPOCH = 5;
export const POINTS_PER_EPOCH = 60;
