import type { Arch, Hyper } from '../nn/types';

/** Training and test sets as flat arrays: sample i occupies [i·inputSize, (i+1)·inputSize). */
export interface DataPayload {
  inputSize: number;
  /** Stored value × scale = network input (1/255 for images, 1 for point features). */
  scale: number;
  classes: number;
  trainX: Uint8Array | Float32Array;
  trainY: Uint8Array;
  testX: Uint8Array | Float32Array;
  testY: Uint8Array;
}

/** A user-supplied sample, stored the same way as the dataset (same scale). */
export interface CustomSample {
  id: number;
  x: Uint8Array | Float32Array;
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
  /** classes × classes counts, row = true label, column = prediction. */
  confusion: number[];
  /**
   * Which weights were evaluated: the `weightsId` of the 'weights' message that carries them.
   * Unlike `step`, it also changes when weights are replaced without training (the backprop view).
   */
  weightsId: number;
  /** The `rev` of the data it was measured on (see the 'data' message). */
  dataRev: number;
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
  /**
   * New training and test data, numbered by the page (`rev`, 0 when left out) so every evaluation
   * says which data it measured. `data: null` takes the data away while the page switches to
   * other data: until new data arrives the trainer neither trains nor evaluates, so the next
   * network is never measured on the data it replaces.
   */
  | { type: 'data'; data: DataPayload | null; rev?: number }
  | { type: 'model'; version: number; arch: Arch; weights: Float32Array[]; hyper: Hyper; frozen: boolean[] }
  | { type: 'frozen'; frozen: boolean[] }
  | { type: 'hyper'; hyper: Hyper }
  /** Caps training at this many samples per second (null = as fast as possible). */
  | { type: 'speed'; samplesPerSec: number | null }
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
  /** `weightsId` changes with every change to the trainer's weights (see EvalPoint.weightsId). */
  | { type: 'weights'; version: number; step: number; weightsId: number; weights: Float32Array[] };

/** Each custom image is shown this many times per epoch so a handful of them still matter. */
export const CUSTOM_REPEAT = 10;
export const EVALS_PER_EPOCH = 5;
export const POINTS_PER_EPOCH = 60;

/** Training speeds offered on the page, in samples per second (null = as fast as possible). */
export const SPEEDS = { slow: 300, normal: 3000, max: null } as const;
export type Speed = keyof typeof SPEEDS;
