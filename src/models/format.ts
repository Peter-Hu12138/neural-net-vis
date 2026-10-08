import type { FeatureId } from '../data/features';
import type { DatasetId } from '../data/datasets';
import { Network, describe } from '../nn/network';
import type { Arch, LayerSpec, Shape } from '../nn/types';

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

function fromBase64(s: string): Float32Array {
  const bin = atob(s);
  if (bin.length % 4) throw new Error('weights are not a whole number of float32 values');
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

/**
 * Parses and checks a model file. Throws an Error whose message says what is wrong, in words a
 * reader can act on. Returns the file and a network holding its weights.
 */
export function decodeModel(json: unknown): { file: ModelFile; net: Network } {
  const f = json as ModelFile;
  if (!f || typeof f !== 'object' || f.format !== 'raster-model') throw new Error('This is not a Raster model file.');
  if (f.version !== 1) throw new Error(`Model file version ${String(f.version)} is not supported.`);
  const a = f.arch;
  if (!a || !isShape(a.input) || !Array.isArray(a.layers) || !a.layers.every(isLayer) || !Number.isInteger(a.classes) || a.classes < 2) {
    throw new Error('The model file has an invalid architecture.');
  }
  const errors = describe(a).filter((l) => l.error);
  if (errors.length) throw new Error(`The model's architecture is not valid: ${errors[0].error}`);
  const net = new Network(a, 0);
  if (!Array.isArray(f.weights) || f.weights.length !== net.blocks.length * 2) {
    throw new Error(`Expected ${net.blocks.length * 2} weight arrays, found ${Array.isArray(f.weights) ? f.weights.length : 0}.`);
  }
  const ws = f.weights.map(fromBase64);
  net.blocks.forEach((b, i) => {
    if (ws[2 * i].length !== b.W.length || ws[2 * i + 1].length !== b.b.length) {
      throw new Error(`Layer ${i + 1} has ${ws[2 * i].length} weights; the architecture needs ${b.W.length}.`);
    }
  });
  if (ws.some((w) => w.some((v) => !Number.isFinite(v)))) throw new Error('The model file contains weights that are not finite numbers.');
  net.setWeights(ws);
  return { file: f, net };
}
