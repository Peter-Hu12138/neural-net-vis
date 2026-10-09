import type { DatasetId } from '../data/datasets';
import { describe } from '../nn/network';
import { ACTIVATIONS, isImage, type Arch, type Shape } from '../nn/types';

/**
 * The pretrained model zoo: public/models/index.json lists the models that scripts/pretrain.ts
 * trained with this app's own engine; each model is a 'raster-model' file next to it.
 * public/models/transfer.json holds the measured transfer-learning results quoted on the page.
 */

/** One pretrained model, as listed in public/models/index.json. */
export interface ZooEntry {
  id: string;
  name: string;
  /** Dataset the model was trained on (Load switches to it). */
  dataset: DatasetId;
  description: string;
  /** Hidden layers in words, e.g. "conv 5×5×6 tanh, pool · dense 64 tanh". */
  layers: string;
  arch: Arch;
  params: number;
  /** Accuracy on the official test set named in testSet. */
  testAccuracy: number;
  testSet: string;
  /** File name inside public/models/. */
  file: string;
  /** Size of that file in bytes. */
  bytes: number;
}

/** One condition of the transfer-learning experiment (public/models/transfer.json). */
export interface TransferCell {
  /** Test accuracy per seed. */
  runs: number[];
  mean: number;
}

export interface TransferRow {
  /** Source model id in the zoo, e.g. 'mnist-cnn'. */
  from: string;
  /** Dataset trained and tested on. */
  to: DatasetId;
  /** Training images used (the first N of the official training set). */
  train: number;
  epochs: number;
  scratch: TransferCell;
  /** Copied hidden layers all frozen; only the new output layer trains (what Transfer does). */
  frozen: TransferCell;
  /** Copied conv layers frozen; the dense layers (hidden and output) train. */
  convFrozen: TransferCell;
  /** Half the epochs with the copied layers frozen, then everything unfrozen at a lower rate. */
  fineTune: TransferCell;
}

export interface TransferReport {
  created: string;
  seeds: number;
  testSet: Record<string, string>;
  settings: string;
  rows: TransferRow[];
}

const actLabel = (id: string) => ACTIVATIONS.find((a) => a.id === id)?.label ?? id;

/** The hidden layers of an architecture in words (the output layer is implied by the classes). */
export function layersSummary(arch: Arch): string {
  if (!arch.layers.length) return 'no hidden layers';
  return arch.layers
    .map((l) => (l.kind === 'conv' ? `conv ${l.kernel}×${l.kernel}×${l.filters} ${actLabel(l.act)}${l.pool ? ', pool' : ''}` : `dense ${l.units} ${actLabel(l.act)}`))
    .join(' · ');
}

/** Total parameter count of an architecture, output layer included. */
export const archParams = (arch: Arch): number => describe(arch).reduce((s, l) => s + l.params, 0);

/** An input shape in words: "28×28 grey", "32×32 colour" or "5 features". */
export function shapeWords(s: Shape): string {
  if (!isImage(s)) return `${s.c} feature${s.c === 1 ? '' : 's'}`;
  return `${s.h}×${s.w} ${s.c === 1 ? 'grey' : s.c === 3 ? 'colour' : `${s.c}-channel`}`;
}

export const sameShape = (a: Shape, b: Shape): boolean => a.c === b.c && a.h === b.h && a.w === b.w;

/** File size for people: "142 kB", "1.2 MB". */
export function fmtBytes(n: number): string {
  if (n < 1000) return `${n} B`;
  if (n < 1_000_000) return `${Math.round(n / 1000)} kB`;
  return `${(n / 1_000_000).toFixed(1)} MB`;
}

const isObj = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object';

/** Checks the zoo index and returns its entries; throws with a readable message when malformed. */
export function parseIndex(json: unknown): ZooEntry[] {
  if (!Array.isArray(json)) throw new Error('The model list is not a list.');
  return json.map((e, i) => {
    if (!isObj(e)) throw new Error(`Entry ${i + 1} of the model list is not an object.`);
    for (const k of ['id', 'name', 'dataset', 'description', 'layers', 'testSet', 'file']) {
      if (typeof e[k] !== 'string' || !(e[k] as string).length) throw new Error(`Entry ${i + 1} of the model list has no ${k}.`);
    }
    for (const k of ['params', 'testAccuracy', 'bytes']) {
      if (typeof e[k] !== 'number' || !Number.isFinite(e[k])) throw new Error(`Entry ${i + 1} of the model list has no ${k}.`);
    }
    const a = e.arch as Arch | undefined;
    if (!isObj(a) || !isObj(a.input) || !Array.isArray(a.layers) || typeof a.classes !== 'number') {
      throw new Error(`Entry ${i + 1} of the model list has no architecture.`);
    }
    if (!/^[a-z0-9-]+\.json$/.test(e.file as string)) throw new Error(`Entry ${i + 1} of the model list names an unexpected file.`);
    return e as unknown as ZooEntry;
  });
}

/** URL of a file in public/models/, relative to wherever the page is served from. */
export const modelsUrl = (file: string): string => `${import.meta.env.BASE_URL}models/${file}`;

async function getJson(url: string, what: string): Promise<unknown> {
  let res: Response;
  try {
    res = await fetch(url);
  } catch {
    throw new Error(`Could not download ${what}: the server did not answer. Check the connection and try again.`);
  }
  if (!res.ok) throw new Error(`Could not download ${what}: the server answered ${res.status}${res.status === 404 ? ' (file not found)' : ''}.`);
  try {
    return await res.json();
  } catch {
    throw new Error(`${cap(what)} could not be read: the file is not valid JSON.`);
  }
}

const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

/** The zoo index. */
export async function fetchIndex(): Promise<ZooEntry[]> {
  return parseIndex(await getJson(modelsUrl('index.json'), 'the list of pretrained models'));
}

/** The measured transfer-learning results, or null when they are missing. */
export async function fetchTransfer(): Promise<TransferReport | null> {
  try {
    const r = (await getJson(modelsUrl('transfer.json'), 'the transfer results')) as TransferReport;
    return isObj(r) && Array.isArray(r.rows) ? r : null;
  } catch {
    return null;
  }
}

/**
 * Downloads one model file, reporting progress as a fraction (0–1). The index's byte count stands
 * in when the server sends no length (or a compressed one).
 */
export async function fetchModel(entry: ZooEntry, onProgress: (fraction: number) => void = () => {}): Promise<unknown> {
  const what = entry.name;
  let res: Response;
  try {
    res = await fetch(modelsUrl(entry.file));
  } catch {
    throw new Error(`Could not download ${what}: the server did not answer. Check the connection and try again.`);
  }
  if (!res.ok) throw new Error(`Could not download ${what}: the server answered ${res.status}${res.status === 404 ? ' (file not found)' : ''}.`);
  const total = Number(res.headers.get('content-length')) || entry.bytes;
  let text: string;
  if (res.body && typeof res.body.getReader === 'function') {
    const reader = res.body.getReader();
    const parts: Uint8Array[] = [];
    let loaded = 0;
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        parts.push(value);
        loaded += value.length;
        onProgress(Math.min(1, loaded / Math.max(total, loaded)));
      }
    } catch {
      throw new Error(`The download of ${what} was interrupted. Try again.`);
    }
    const all = new Uint8Array(loaded);
    let o = 0;
    for (const p of parts) {
      all.set(p, o);
      o += p.length;
    }
    text = new TextDecoder().decode(all);
  } else {
    text = await res.text();
  }
  onProgress(1);
  try {
    return JSON.parse(text);
  } catch {
    throw new Error(`${what} could not be read: the file is not valid JSON.`);
  }
}
