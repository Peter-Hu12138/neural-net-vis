export type Act = 'relu' | 'leaky' | 'tanh' | 'sigmoid' | 'linear';

export const ACTIVATIONS: { id: Act; label: string; formula: string }[] = [
  { id: 'relu', label: 'ReLU', formula: 'max(0, z)' },
  { id: 'leaky', label: 'Leaky ReLU', formula: 'max(0.1z, z)' },
  { id: 'tanh', label: 'Tanh', formula: 'tanh(z)' },
  { id: 'sigmoid', label: 'Sigmoid', formula: '1 / (1 + e^−z)' },
  { id: 'linear', label: 'Linear', formula: 'z' },
];

export interface ConvSpec {
  kind: 'conv';
  filters: number;
  kernel: 3 | 5;
  act: Act;
  pool: boolean;
}

export interface DenseSpec {
  kind: 'dense';
  units: number;
  act: Act;
}

export type LayerSpec = ConvSpec | DenseSpec;

/** Channels × height × width. Dense vectors use h = w = 1. */
export interface Shape {
  c: number;
  h: number;
  w: number;
}

/** A whole model: what goes in, the hidden layers, and how many classes come out. */
export interface Arch {
  input: Shape;
  layers: LayerSpec[];
  classes: number;
}

/** MNIST's input: 28×28 greyscale. */
export const MNIST_INPUT: Shape = { c: 1, h: 28, w: 28 };

/** An MNIST-shaped model (28×28×1 in, 10 classes out) with the given hidden layers. */
export const mnistArch = (layers: LayerSpec[]): Arch => ({ input: MNIST_INPUT, layers, classes: 10 });

/** True when the input is an image (convolutions apply), false for a plain feature vector. */
export const isImage = (s: Shape) => s.h > 1 || s.w > 1;

export const size = (s: Shape) => s.c * s.h * s.w;
export const fmtShape = (s: Shape) => (s.h === 1 && s.w === 1 ? `${s.c}` : `${s.h}×${s.w}×${s.c}`);

export type OptimizerName = 'sgd' | 'momentum' | 'adam';

export interface Hyper {
  lr: number;
  batchSize: number;
  optimizer: OptimizerName;
}
