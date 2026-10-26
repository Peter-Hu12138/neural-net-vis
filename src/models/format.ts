import { DATASETS, type DatasetId, type DatasetInfo } from '../data/datasets';
import { featureCatalog, type FeatureId } from '../data/features';
import { Network, describe } from '../nn/network';
import { isImage, type Arch, type LayerSpec, type Shape } from '../nn/types';

/**
 * A saved model: architecture, weights and where they came from. JSON with base64-encoded
 * little-endian float32 weights, small enough to ship in the repository and to download.
 */
export interface ModelFile {
  format: 'raster-model';
  version: 1;
  name: string;
  /** Dataset the weights were trained on (decides the class names and, for points, the features). */
  dataset: DatasetId;
  /** Point datasets only: which input features the first layer expects. */
  features?: FeatureId[];
  arch: Arch;
  /** [W0, b0, W1, b1, …] as base64 float32. */
  weights: string[];
  /** Per block: frozen when saved (optional). */
  frozen?: boolean[];
  meta: {
    description?: string;
    trainedOn?: string;
    samples?: number;
    epochs?: number;
    testAccuracy?: number;
    created?: string;
  };
}

function toBase64(a: Float32Array): string {
  const bytes = new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
  let s = '';
  const CH = 0x8000;
  for (let i = 0; i < bytes.length; i += CH) s += String.fromCharCode(...bytes.subarray(i, i + CH));
  return btoa(s);
}

/** Decodes weight array `n` (1-based, for messages) of a model file. */
function fromBase64(s: unknown, n: number): Float32Array {
  let bin: string;
  try {
    if (typeof s !== 'string') throw new Error();
    bin = atob(s);
  } catch {
    throw new Error(`Weight array ${n} of the model file is damaged: it is not base64 text.`);
  }
  if (bin.length % 4) throw new Error(`Weight array ${n} of the model file is damaged: it is not a whole number of float32 values.`);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

export function encodeModel(net: Network, info: Omit<ModelFile, 'format' | 'version' | 'arch' | 'weights'>): ModelFile {
  return {
    format: 'raster-model',
    version: 1,
    ...info,
    arch: structuredClone(net.arch),
    weights: net.getWeights().map(toBase64),
  };
}

const sameShape = (a: Shape, b: Shape) => a.c === b.c && a.h === b.h && a.w === b.w;

const isShape = (s: unknown): s is Shape =>
  !!s && typeof s === 'object' && [(s as Shape).c, (s as Shape).h, (s as Shape).w].every((v) => Number.isInteger(v) && v > 0);

const isLayer = (l: unknown): l is LayerSpec => {
  const x = l as LayerSpec;
  if (!x || typeof x !== 'object') return false;
  const actOk = ['relu', 'leaky', 'tanh', 'sigmoid', 'linear'].includes(x.act);
  if (x.kind === 'conv') return actOk && Number.isInteger(x.filters) && x.filters > 0 && (x.kernel === 3 || x.kernel === 5) && typeof x.pool === 'boolean';
  if (x.kind === 'dense') return actOk && Number.isInteger(x.units) && x.units > 0;
  return false;
};

/** Text from a file, shortened for a message. */
const clip = (s: string) => (s.length > 40 ? `${s.slice(0, 40)}…` : s);

const isObject = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v);

/** "28×28 grey inputs", "32×32 colour inputs" or "2 input features", for messages. */
function inputWords(s: Shape): string {
  if (!isImage(s)) return `${s.c} input feature${s.c === 1 ? '' : 's'}`;
  return `${s.h}×${s.w} ${s.c === 1 ? 'grey' : s.c === 3 ? 'colour' : `${s.c}-channel`} inputs`;
}

/** What a dataset feeds the network, for messages: "32×32 colour images" or "2-D points". */
function datasetWords(info: DatasetInfo): string {
  if (info.kind === 'points') return `${info.dims}-D points`;
  const s = info.image!.shape;
  return `${s.h}×${s.w} ${s.c === 1 ? 'grey' : 'colour'} images`;
}

/** Checks the optional meta data: numbers where numbers belong, text where text belongs. */
function checkMeta(meta: unknown): void {
  if (meta === undefined) return;
  if (!isObject(meta)) throw new Error('The model file’s meta data is damaged: “meta” should be a group of named values.');
  const acc = meta.testAccuracy;
  if (acc !== undefined && !(typeof acc === 'number' && acc >= 0 && acc <= 1)) throw new Error('The model file’s test accuracy should be a number between 0 and 1.');
  for (const k of ['samples', 'epochs']) {
    const v = meta[k];
    if (v !== undefined && !(typeof v === 'number' && Number.isFinite(v) && v >= 0)) throw new Error(`The model file’s meta data is damaged: “${k}” should be a number.`);
  }
  for (const k of ['description', 'trainedOn', 'created']) {
    if (meta[k] !== undefined && typeof meta[k] !== 'string') throw new Error(`The model file’s meta data is damaged: “${k}” should be text.`);
  }
}

/**
 * Point models: the feature list must name known features, once each, one per input of the first
 * layer. Image models have no feature list.
 */
function checkFeatures(f: ModelFile, info: DatasetInfo): void {
  const list = (f as { features?: unknown }).features;
  if (info.kind === 'image') {
    if (list != null) throw new Error(`The model file lists input features, but ${info.name} is an image dataset.`);
    return;
  }
  if (list == null) throw new Error('The model file does not say which input features its first layer expects.');
  if (!Array.isArray(list) || !list.length || !list.every((id) => typeof id === 'string')) throw new Error('The model file’s list of input features is damaged.');
  const ids = list as string[];
  const dup = ids.find((id, i) => ids.indexOf(id) !== i);
  if (dup !== undefined) throw new Error(`The model file lists the feature ${clip(dup)} twice.`);
  const catalog = featureCatalog(info.dims!);
  const unknown = ids.filter((id) => !catalog.some((c) => c.id === id));
  if (unknown.length) throw new Error(`The model file asks for input features this page does not have for ${info.dims}-D points: ${unknown.slice(0, 5).map(clip).join(', ')}${unknown.length > 5 ? ', …' : ''}.`);
  const input = f.arch.input;
  if (!isImage(input) && input.c !== ids.length) {
    const labels = ids.map((id) => catalog.find((c) => c.id === id)!.label).join(', ');
    throw new Error(`The model takes ${inputWords(input)}, but the file lists ${ids.length} (${labels}).`);
  }
}

/**
 * Parses and checks a model file. Throws an Error whose message says what is wrong, in words a
 * reader can act on. Returns the file and a network holding its weights. Every field the page
 * reads is checked here, before anything changes: a file that passes can be loaded or transferred
 * without leaving the page half switched.
 */
export function decodeModel(json: unknown): { file: ModelFile; net: Network } {
  const f = json as ModelFile;
  if (!isObject(f) || f.format !== 'raster-model') throw new Error('This is not a Raster model file.');
  if (f.version !== 1) throw new Error(`Model file version ${String(f.version)} is not supported.`);
  if (typeof f.name !== 'string' || !f.name.trim()) throw new Error('The model file has no name.');
  if (typeof f.dataset !== 'string' || !f.dataset) throw new Error('The model file does not say which dataset it was trained on.');
  const info = DATASETS.find((d) => d.id === f.dataset);
  if (!info) {
    const images = DATASETS.filter((d) => d.kind === 'image').map((d) => d.name);
    const points = DATASETS.length - images.length;
    throw new Error(`The model file names a dataset "${clip(f.dataset)}" that this page does not have. This page has ${images.join(', ')} and ${points} point datasets.`);
  }
  checkMeta(f.meta);
  const a = f.arch;
  if (!isObject(a) || !isShape(a.input) || !Array.isArray(a.layers) || !a.layers.every(isLayer) || !Number.isInteger(a.classes) || a.classes < 2) {
    throw new Error('The model file has an invalid architecture.');
  }
  const layers = describe(a);
  const errors = layers.filter((l) => l.error);
  if (errors.length) throw new Error(`The model's architecture is not valid: ${errors[0].error}`);
  // Sizes come from the architecture alone, so a damaged file is refused before any memory is set aside for it.
  if (!Array.isArray(f.weights) || f.weights.length !== layers.length * 2) {
    throw new Error(`Expected ${layers.length * 2} weight arrays, found ${Array.isArray(f.weights) ? f.weights.length : 0}.`);
  }
  const ws = f.weights.map((w, i) => fromBase64(w, i + 1));
  layers.forEach((l, i) => {
    const nb = l.outShape.c;
    const nw = l.params - nb;
    if (ws[2 * i].length !== nw) throw new Error(`Layer ${i + 1} has ${ws[2 * i].length} weights; the architecture needs ${nw}.`);
    if (ws[2 * i + 1].length !== nb) throw new Error(`Layer ${i + 1} has ${ws[2 * i + 1].length} biases; the architecture needs ${nb}.`);
  });
  if (ws.some((w) => w.some((v) => !Number.isFinite(v)))) throw new Error('The model file contains weights that are not finite numbers.');
  // The architecture must fit the dataset it names: Load switches to that dataset.
  const want = info.kind === 'image' ? info.image!.shape : null;
  if (want ? !sameShape(a.input, want) : isImage(a.input)) throw new Error(`The model takes ${inputWords(a.input)}, but ${info.name} has ${datasetWords(info)}.`);
  checkFeatures(f, info);
  if (a.classes !== info.classes.length) throw new Error(`The model has ${a.classes} output classes, but ${info.name} has ${info.classes.length}.`);
  const frozen = (f as { frozen?: unknown }).frozen;
  if (frozen != null && !(Array.isArray(frozen) && frozen.length === layers.length && frozen.every((v) => typeof v === 'boolean'))) {
    throw new Error(`The model file’s frozen list should hold one true or false per layer (${layers.length} layers).`);
  }
  const net = new Network(a, 0);
  net.setWeights(ws);
  return { file: f, net };
}
