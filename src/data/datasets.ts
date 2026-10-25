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
  group: 'Images' | 'Points in 2-D' | 'Points in 3-D';
  kind: 'image' | 'points';
  /** Class names, in label order. */
  classes: string[];
  /** One- or two-character marks for each class, for plots where points are drawn as text. */
  glyphs: string[];
  description: string;
  image?: ImageSource;
  /**
   * Image datasets: the first test image of classes 0–3 as a 2 × 2 mosaic at full resolution (a
   * data URI of a few kilobytes), so the dataset strip shows real samples before the dataset loads.
   * Grey sets are lossless PNGs; CIFAR-10's is a JPEG (quality 90, no chroma subsampling).
   */
  preview?: string;
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
    preview: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADgAAAA4CAAAAACN7WTCAAAC/0lEQVR42u2VX0jTURTHj9NZ6UxbCq4HNxC1EEXIEAWVCkqxKYVkReRLBiEWQn/tRR8KBLVUsIdAC0OTUKSokNSSQs0ymBGauQyxmZZkaG55/P7sYW5uv92f0ltg9+nc872f37m/c889l2idDf8a9OlX0SOlArEQwYz8VcCcxUNOW+XiD7m9xo7ifrWI3Gc6mRlNBSlKXMxcjdAPZmYwj+xUALOlVJH78RIATH0CoAD2jfoLvKlmMHO1MaWY+bSQM0hDIu8Eg0dK/Yj0Fp4tVAuW5EovFE6iPZiIiAoYHC5YUiYZxeCrMLup7xWCidP9G12mznNUqRLG7JaXSqUq8QT3aodson0wO0yFrd6XDopS9sEBhqROMCxhHgtCvw7SquANZjYney64JNW5zX3kZRBFRIOCvOvphzDiMJCePg4sQaFyxqU9QrCQ2VGrXC3Qk1kB1FuWQUtHuJ9AL5f6vcUVnFJhB8UX2W9Quqx4TdNauHl/WphQU3e3+tH/8e8OrxXz3KbYbLrZU/+3n2gCAGDYowI0O6qq4lbn3le0AkVyrgTAQr1WgYtfwIBBQ75vUSZTrtp38mWfGDTygI6IimzYLVPysFh5Kn8a1mJxteq1REQmeICP0EhEyd+AerXij563olv+3SVEExEldQF3fRS4A1ZMeDxIT2EgIqKE78BhEj6sFO9LTV1ycJCITnbm5BgaiSLEAVvnUafx8J5A5nabPbMYzQ4UcLopTAo6+ObjmvAZOMbcUc/cdgPlCj+fceehEzVFy8RMGzo0ivn2DgqKigwKqpwF2tzrb2uvckAKTnJYSSbgiZt2DWhWCmg0W7McdsBHzKS5ijZApxTwmBWLiY7JLsYz2aOjZSL6yepA2lJIhIvzDqFhW6mX88BjvWhAFhEAcO96w7J1xeVI2mGujSQiOvtuHqh07QAtWSvrFiV68IZe9rhcSpNugyQt7/B1xrRr67igJorOIar9TC2Cxzf3SIw9B91ttybde85aIzQg73n8cP/Y73XTkP8A6gmDjKRtxVgAAAAASUVORK5CYII=',
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
    preview: 'data:image/png;base64,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',
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
    preview: 'data:image/jpeg;base64,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',
    credit: 'CIFAR-10 by Alex Krizhevsky, Vinod Nair and Geoffrey Hinton',
  },
  ...SYNTHETIC.map((s): DatasetInfo => {
    const names = Array.from({ length: s.classes }, (_, k) => `Class ${k}`);
    return {
      id: s.id,
      name: s.name,
      group: s.dims === 2 ? 'Points in 2-D' : 'Points in 3-D',
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

/** What one sample is called in prose: "digit", "image" or "point" (plural with n ≠ 1). */
export function noun(info: DatasetInfo, n = 1): string {
  const one = info.id === 'mnist' ? 'digit' : info.kind === 'points' ? 'point' : 'image';
  return n === 1 ? one : `${one}s`;
}

/** Caption for a sample, e.g. "Test digit #12 · label 7" or "Test image #3 · cat". */
export function sampleCaption(info: DatasetInfo, split: 'train' | 'test', i: number, y: number): string {
  const which = split === 'test' ? 'Test' : 'Training';
  if (info.id === 'mnist') return `${which} digit #${i} · label ${y}`;
  if (info.kind === 'points') return `${which} point #${i} · ${info.classes[y]}`;
  return `${which} image #${i} · ${info.classes[y]}`;
}
