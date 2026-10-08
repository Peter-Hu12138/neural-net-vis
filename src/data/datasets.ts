import { size, type Shape } from '../nn/types';
import { defaultFeatures, featurize, type FeatureId } from './features';
import { DEFAULT_SYNTHETIC, SYNTHETIC, generate, type SyntheticConfig, type SyntheticId } from './synthetic';

/**
 * Every dataset the app can train on. Image datasets ship as sprite sheets in public/data; point
 * datasets are generated in the browser. All of them become the same `Data` shape: flat arrays of
 * network inputs (stored values × `scale`) with one label per sample.
 */

export type ImageDatasetId = 'mnist' | 'fashion' | 'cifar10';
export type DatasetId = ImageDatasetId | SyntheticId;

export interface ImageSource {
  shape: Shape;
  ext: 'png' | 'jpg';
  train: number;
  test: number;
  /** Images per sheet. */
  chunk: number;
}

export interface DatasetInfo {
  id: DatasetId;
  name: string;
  group: 'Images' | 'Points in 2D' | 'Points in 3D';
  kind: 'image' | 'points';
  /** Class names, in label order. */
  classes: string[];
  /** One- or two-character marks for each class, for plots where points are drawn as text. */
  glyphs: string[];
  description: string;
  image?: ImageSource;
  dims?: 2 | 3;
  credit?: string;
}

const digits = Array.from({ length: 10 }, (_, k) => String(k));

export const DATASETS: DatasetInfo[] = [
  {
    id: 'mnist',
    name: 'MNIST digits',
    group: 'Images',
    kind: 'image',
    classes: digits,
    glyphs: digits,
    description: 'Handwritten digits, 28×28 grey. The classic first dataset.',
    image: { shape: { c: 1, h: 28, w: 28 }, ext: 'png', train: 20_000, test: 2_000, chunk: 5_000 },
    credit: 'MNIST by Yann LeCun, Corinna Cortes and Christopher J.C. Burges (CC BY-SA 3.0)',
  },
  {
    id: 'fashion',
    name: 'Fashion-MNIST',
    group: 'Images',
    kind: 'image',
    classes: ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'],
    glyphs: digits,
    description: 'Photos of clothing, 28×28 grey. Same format as MNIST, harder to classify.',
    image: { shape: { c: 1, h: 28, w: 28 }, ext: 'png', train: 10_000, test: 2_000, chunk: 5_000 },
    credit: 'Fashion-MNIST by Zalando Research (MIT licence)',
  },
  {
    id: 'cifar10',
    name: 'CIFAR-10',
    group: 'Images',
    kind: 'image',
    classes: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
    glyphs: digits,
    description: 'Small colour photos, 32×32×3, of ten kinds of object. Much harder than MNIST.',
    image: { shape: { c: 3, h: 32, w: 32 }, ext: 'jpg', train: 10_000, test: 2_000, chunk: 5_000 },
    credit: 'CIFAR-10 by Alex Krizhevsky, Vinod Nair and Geoffrey Hinton',
  },
  ...SYNTHETIC.map((s): DatasetInfo => {
    const names = Array.from({ length: s.classes }, (_, k) => `Class ${k}`);
    return {
      id: s.id,
      name: s.name,
      group: s.dims === 2 ? 'Points in 2D' : 'Points in 3D',
      kind: 'points',
      classes: names,
      glyphs: names.map((_, k) => String(k)),
      description: s.description,
      dims: s.dims,
    };
  }),
];

export const datasetInfo = (id: DatasetId): DatasetInfo => {
  const d = DATASETS.find((x) => x.id === id);
  if (!d) throw new Error(`Unknown dataset ${id}`);
  return d;
};

export interface PointsMeta {
  dims: 2 | 3;
  features: FeatureId[];
  config: SyntheticConfig;
  /** Raw coordinates (n × dims) for plotting; the network sees `trainX` / `testX` (features). */
  trainCoords: Float32Array;
  testCoords: Float32Array;
}

export interface Data {
  info: DatasetInfo;
  /** Network input shape. */
  input: Shape;
  inputSize: number;
  /** Stored value × scale = network input (1/255 for images, 1 for point features). */
  scale: number;
  trainX: Uint8Array | Float32Array;
  trainY: Uint8Array;
  testX: Uint8Array | Float32Array;
  testY: Uint8Array;
  points?: PointsMeta;
}

/** Sample `i` of a split as network input. */
export function sampleInput(data: Data, split: 'train' | 'test', i: number, out?: Float32Array): Float32Array {
  const src = split === 'train' ? data.trainX : data.testX;
  const n = data.inputSize;
  const x = out ?? new Float32Array(n);
  const off = i * n;
  const s = data.scale;
  for (let j = 0; j < n; j++) x[j] = src[off + j] * s;
  return x;
}

/** Input shape of a point dataset with the given features: a flat vector. */
export const featureShape = (features: FeatureId[]): Shape => ({ c: features.length, h: 1, w: 1 });

/** Builds point data from a synthetic configuration and feature choice. */
export function pointsData(config: SyntheticConfig, features?: FeatureId[]): Data {
  const syn = generate(config);
  const info = datasetInfo(config.id);
  const dims = syn.info.dims;
  const feats = features ?? defaultFeatures(dims);
  const input = featureShape(feats);
  return {
    info,
    input,
    inputSize: size(input),
    scale: 1,
    trainX: featurize(syn.train.coords, dims, feats),
    trainY: syn.train.labels,
    testX: featurize(syn.test.coords, dims, feats),
    testY: syn.test.labels,
    points: { dims, features: feats, config, trainCoords: syn.train.coords, testCoords: syn.test.coords },
  };
}

export const defaultPointsConfig = (id: SyntheticId): SyntheticConfig => ({ id, ...DEFAULT_SYNTHETIC });

/** Keeps the first `limit` training samples (for "what if we only had N examples?"). */
export function limitTraining(data: Data, limit: number | null): Data {
  if (limit === null || limit >= data.trainY.length) return data;
  const n = data.inputSize;
  return {
    ...data,
    trainX: data.trainX.slice(0, limit * n),
    trainY: data.trainY.slice(0, limit),
    points: data.points ? { ...data.points, trainCoords: data.points.trainCoords.slice(0, limit * data.points.dims) } : undefined,
  };
}

/** Caption for a sample, e.g. "Test digit #12 · label 7" or "Test image #3 · cat". */
export function sampleCaption(info: DatasetInfo, split: 'train' | 'test', i: number, y: number): string {
  const which = split === 'test' ? 'Test' : 'Training';
  if (info.id === 'mnist') return `${which} digit #${i} · label ${y}`;
  if (info.kind === 'points') return `${which} point #${i} · ${info.classes[y]}`;
  return `${which} image #${i} · ${info.classes[y]}`;
}
