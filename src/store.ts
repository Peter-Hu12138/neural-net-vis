import { Network, describe } from './nn/network';
import type { Mnist } from './data/mnist';
import { DEFAULT_LR } from './nn/optim';
import type { Hyper, LayerSpec } from './nn/types';
import type { EvalPoint, Status, TrainPoint } from './train/protocol';

export type WeightMode = 'heat' | 'hinton' | 'numbers' | 'hist';

export interface Probe {
  x: Float32Array;
  label: number | null;
  caption: string;
  key: string;
}

export interface CustomEntry {
  id: number;
  x: Float32Array;
  y: number;
  origin: 'upload' | 'drawing';
  name: string;
}

/** Highlight requested by the backprop view: which block, and forward or backward. */
export interface Highlight {
  block: number; // -1 = input, blocks.length = loss
  dir: 'fwd' | 'back' | 'update';
}

type EventName = 'model' | 'weights' | 'status' | 'metrics' | 'probe' | 'select' | 'mode' | 'custom' | 'data' | 'highlight' | 'hyper';

export const PRESETS: { name: string; spec: LayerSpec[] }[] = [
  { name: 'Softmax', spec: [] },
  { name: 'MLP', spec: [{ kind: 'dense', units: 64, act: 'relu' }] },
  {
    name: 'Small CNN',
    spec: [
      { kind: 'conv', filters: 8, kernel: 3, act: 'relu', pool: true },
      { kind: 'conv', filters: 16, kernel: 3, act: 'relu', pool: true },
      { kind: 'dense', units: 32, act: 'relu' },
    ],
  },
  {
    name: 'LeNet-ish',
    spec: [
      { kind: 'conv', filters: 6, kernel: 5, act: 'tanh', pool: true },
      { kind: 'conv', filters: 16, kernel: 5, act: 'tanh', pool: true },
      { kind: 'dense', units: 64, act: 'tanh' },
    ],
  },
];

export class Store {
  spec: LayerSpec[] = structuredClone(PRESETS[2].spec);
  hyper: Hyper = { lr: DEFAULT_LR.adam, batchSize: 32, optimizer: 'adam' };
  seed = 1;
  version = 0;
  net: Network = new Network(this.spec, this.seed);
  weightsStep = 0;
  data: Mnist | null = null;
  status: Status | null = null;
  points: TrainPoint[] = [];
  evals: EvalPoint[] = [];
  probe: Probe | null = null;
  selected = 0; // block index shown in the inspector
  selectedUnit: number | null = null;
  mode: WeightMode = 'heat';
  custom: CustomEntry[] = [];
  highlight: Highlight | null = null;

  private handlers = new Map<EventName, Set<() => void>>();

  on(ev: EventName, fn: () => void): void {
    if (!this.handlers.has(ev)) this.handlers.set(ev, new Set());
    this.handlers.get(ev)!.add(fn);
  }

  emit(ev: EventName): void {
    this.handlers.get(ev)?.forEach((fn) => fn());
  }

  get valid(): boolean {
    return !describe(this.spec).some((l) => l.error);
  }

  get running(): boolean {
    return !!this.status?.running;
  }
}

export const store = new Store();
