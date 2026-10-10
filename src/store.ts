import { datasetInfo, featureShape, type Data, type DatasetId, type DatasetInfo } from './data/datasets';
import { defaultFeatures, type FeatureId } from './data/features';
import { DEFAULT_SYNTHETIC, type SyntheticConfig } from './data/synthetic';
import { Network, describe } from './nn/network';
import { DEFAULT_LR } from './nn/optim';
import type { Arch, Hyper, LayerSpec, Shape } from './nn/types';
import type { EvalPoint, Speed, Status, TrainPoint } from './train/protocol';

export type WeightMode = 'heat' | 'hinton' | 'numbers' | 'hist' | 'qq';

export interface Probe {
  x: Float32Array;
  label: number | null;
  caption: string;
  key: string;
  /** Point datasets: the raw coordinates the input was computed from. */
  coords?: Float32Array;
}

export interface CustomEntry {
  id: number;
  /** Network input (already scaled). */
  x: Float32Array;
  y: number;
  origin: 'upload' | 'drawing' | 'point';
  name: string;
  /** Point datasets: raw coordinates. */
  coords?: Float32Array;
}

/** Highlight requested by the backprop view: which block, and forward or backward. */
export interface Highlight {
  block: number; // -1 = input, blocks.length = loss
  dir: 'fwd' | 'back' | 'update';
}

type EventName =
  | 'model'
  | 'weights'
  | 'status'
  | 'metrics'
  | 'probe'
  | 'select'
  | 'mode'
  | 'custom'
  | 'data'
  | 'highlight'
  | 'hyper'
  | 'dataset'
  | 'frozen';

export interface Preset {
  name: string;
  spec: LayerSpec[];
}

export const IMAGE_PRESETS: Preset[] = [
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

export const POINT_PRESETS: Preset[] = [
  { name: 'Linear', spec: [] },
  { name: 'One layer', spec: [{ kind: 'dense', units: 6, act: 'tanh' }] },
  {
    name: 'Two layers',
    spec: [
      { kind: 'dense', units: 8, act: 'tanh' },
      { kind: 'dense', units: 8, act: 'tanh' },
    ],
  },
  {
    name: 'Deep',
    spec: [
      { kind: 'dense', units: 8, act: 'relu' },
      { kind: 'dense', units: 8, act: 'relu' },
      { kind: 'dense', units: 8, act: 'relu' },
    ],
  },
];

/** @deprecated image presets; use presetsFor(). */
export const PRESETS = IMAGE_PRESETS;

export const presetsFor = (info: DatasetInfo): Preset[] => (info.kind === 'image' ? IMAGE_PRESETS : POINT_PRESETS);

/** Default architecture and training settings when switching to a dataset of this kind. */
export function defaultsFor(info: DatasetInfo): { spec: LayerSpec[]; hyper: Hyper; speed: Speed } {
  // Point datasets train an epoch in a millisecond or two; cap the speed so the boundary can be watched forming.
  if (info.kind === 'points') return { spec: structuredClone(POINT_PRESETS[2].spec), hyper: { lr: 0.03, batchSize: 10, optimizer: 'adam' }, speed: 'normal' };
  const spec = structuredClone(IMAGE_PRESETS[2].spec);
  // Colour photos: the same Small CNN with a Leaky ReLU dense layer. With ReLU, 18–28 of its 32
  // dense units died within two epochs of CIFAR-10 and test accuracy swung between 33% and 48%
  // with the seed; Leaky ReLU kept them alive (1–2 dead) at 48.5–49.2% (docs/TESTING.md, section 7).
  if (info.image!.shape.c === 3) spec[spec.length - 1] = { kind: 'dense', units: 32, act: 'leaky' };
  return { spec, hyper: { lr: DEFAULT_LR.adam, batchSize: 32, optimizer: 'adam' }, speed: 'max' };
}

export class Store {
  dataset: DatasetId = 'mnist';
  /** Point datasets: generator settings and input features. */
  pointsConfig: Omit<SyntheticConfig, 'id'> = { ...DEFAULT_SYNTHETIC };
  features: FeatureId[] = defaultFeatures(2);
  /** Train on only the first N training samples (null = all). */
  trainLimit: number | null = null;
  spec: LayerSpec[] = structuredClone(IMAGE_PRESETS[2].spec);
  hyper: Hyper = { lr: DEFAULT_LR.adam, batchSize: 32, optimizer: 'adam' };
  /** Training speed cap (see SPEEDS). */
  speed: Speed = 'max';
  /** Per block (hidden layers then output): weights held fixed during training. */
  frozen: boolean[] = [];
  /** When the architecture is edited, keep the weights of layers that did not change. */
  keepWeights = false;
  /** Name of the pretrained model the weights came from, if any. */
  modelName: string | null = null;
  seed = 1;
  version = 0;
  net: Network;
  weightsStep = 0;
  /** Bumped on every change to the page's weights (training snapshots, manual updates, rebuilds). */
  weightsRev = 0;
  /** The loaded dataset (null while loading). */
  data: Data | null = null;
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

  constructor() {
    this.net = new Network(this.arch, this.seed);
  }

  on(ev: EventName, fn: () => void): void {
    if (!this.handlers.has(ev)) this.handlers.set(ev, new Set());
    this.handlers.get(ev)!.add(fn);
  }

  /** Calls every listener; one that throws is reported and does not stop the others. */
  emit(ev: EventName): void {
    this.handlers.get(ev)?.forEach((fn) => {
      try {
        fn();
      } catch (e) {
        console.error(`Error in a '${ev}' listener:`, e);
      }
    });
  }

  get info(): DatasetInfo {
    return datasetInfo(this.dataset);
  }

  /** The network's input shape for the current dataset (and, for points, the chosen features). */
  get input(): Shape {
    const info = this.info;
    return info.kind === 'image' ? info.image!.shape : featureShape(this.features);
  }

  get classes(): number {
    return this.info.classes.length;
  }

  get arch(): Arch {
    return { input: this.input, layers: this.spec, classes: this.classes };
  }

  get valid(): boolean {
    return !describe(this.arch).some((l) => l.error);
  }

  get running(): boolean {
    return !!this.status?.running;
  }

  isFrozen(block: number): boolean {
    return !!this.frozen[block];
  }
}

export const store = new Store();
