import { Network } from './nn/network';
import { DEFAULT_LR } from './nn/optim';
import { analysis } from './analysis/client';
import { datasetInfo, limitTraining, pointsData, sampleCaption, sampleInput, type Data, type DatasetId } from './data/datasets';
import { defaultFeatures, type FeatureId } from './data/features';
import { loadImageDataset } from './data/images';
import type { SyntheticConfig } from './data/synthetic';
import { decodeModel, encodeModel, type ModelFile } from './models/format';
import type { Hyper, LayerSpec, Shape } from './nn/types';
import { defaultsFor, store, type CustomEntry, type Probe, type WeightMode } from './store';
import { TrainerClient } from './train/client';
import type { FromTrainer } from './train/protocol';

let weightsQueued = false;

function sameWeights(net: Network, ws: Float32Array[]): boolean {
  return net.blocks.every((b, i) => equal(b.W, ws[2 * i]) && equal(b.b, ws[2 * i + 1]));
}

function equal(a: Float32Array, b: Float32Array): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

function onMessage(m: FromTrainer): void {
  switch (m.type) {
    case 'status':
      if (m.status.version !== store.version) return;
      store.status = m.status;
      store.emit('status');
      break;
    case 'metrics':
      if (m.version !== store.version) return;
      store.points.push(...m.points);
      store.evals.push(...m.evals);
      store.emit('metrics');
      break;
    case 'weights':
      if (m.version !== store.version) return;
      // The trainer resends unchanged weights after pauses and evaluations; only real changes count.
      if (m.step !== store.weightsStep || !sameWeights(store.net, m.weights)) store.weightsRev++;
      store.net.setWeights(m.weights);
      store.weightsStep = m.step;
      if (!weightsQueued) {
        weightsQueued = true;
        requestAnimationFrame(() => {
          weightsQueued = false;
          store.emit('weights');
        });
      }
      break;
  }
}

export const client = new TrainerClient(onMessage);

/** Initial weights per block, kept so the histogram can show how far training has moved them. */
export let initialWeights: Float32Array[] = store.net.getWeights();

export interface RebuildOptions {
  /** Start from these exact weights (a loaded model). */
  weights?: Float32Array[];
  /** Copy the weights of every block whose layout matches this network's (transfer, kept layers). */
  from?: Network | null;
  /** With `from`: also copy the output layer when it matches (false gives a fresh head). */
  copyOutput?: boolean;
  /** Frozen flags per block; by default flags survive only on blocks whose weights were kept. */
  frozen?: boolean[];
}

/** Builds a network for the current dataset and spec, sends it to the trainer, resets the history. */
export function rebuild(newSeed = false, opts: RebuildOptions = {}): void {
  if (!store.valid) return;
  if (newSeed) {
    store.seed = (store.seed * 7919 + 17) % 100003;
    store.modelName = null;
  }
  const oldFrozen = store.frozen;
  store.version++;
  store.net = new Network(store.arch, store.seed);
  let kept = store.net.blocks.map(() => false);
  if (opts.weights) {
    store.net.setWeights(opts.weights);
    kept = kept.map(() => true);
  } else if (opts.from) {
    const out = store.net.output;
    const fresh = opts.copyOutput === false ? { W: out.W.slice(), b: out.b.slice() } : null;
    kept = store.net.copyCompatible(opts.from);
    if (fresh) {
      out.W.set(fresh.W);
      out.b.set(fresh.b);
      kept[kept.length - 1] = false;
    }
  }
  store.frozen = opts.frozen ? opts.frozen.slice(0, kept.length) : kept.map((k, i) => k && !newSeed && !!oldFrozen[i]);
  while (store.frozen.length < kept.length) store.frozen.push(false);
  initialWeights = store.net.getWeights();
  store.weightsStep = 0;
  store.weightsRev++;
  store.points = [];
  store.evals = [];
  store.status = null;
  if (store.selected >= store.net.blocks.length) store.selected = store.net.blocks.length - 1;
  store.selectedUnit = null;
  client.post({ type: 'model', version: store.version, arch: store.arch, weights: store.net.getWeights(), hyper: store.hyper, frozen: store.frozen });
  store.emit('model');
  store.emit('frozen');
  store.emit('weights');
  store.emit('metrics');
  store.emit('status');
}

export function setSpec(spec: LayerSpec[]): void {
  store.spec = spec;
  if (store.valid) rebuild(false, { from: store.keepWeights ? store.net : null });
  else store.emit('model');
}

/** Freezes or unfreezes one block (its weights stop or resume changing during training). */
export function setFrozen(block: number, frozen: boolean): void {
  while (store.frozen.length < store.net.blocks.length) store.frozen.push(false);
  store.frozen[block] = frozen;
  client.post({ type: 'frozen', frozen: store.frozen.slice() });
  store.emit('frozen');
}

export function setKeepWeights(keep: boolean): void {
  store.keepWeights = keep;
  store.emit('model');
}

// ── Datasets ──────────────────────────────────────────────────────────────

/** The full dataset as loaded; store.data may be a training subset of it. */
let fullData: Data | null = null;
let datasetToken = 0;

/** Loading progress of an image dataset (null when idle). */
export let loading: { id: DatasetId; done: number; total: number } | null = null;

function payload(d: Data) {
  return { inputSize: d.inputSize, scale: d.scale, classes: d.info.classes.length, trainX: d.trainX, trainY: d.trainY, testX: d.testX, testY: d.testY };
}

function applyData(data: Data): void {
  fullData = data;
  const d = limitTraining(data, store.trainLimit);
  store.data = d;
  client.post({ type: 'data', data: payload(d) });
  analysis.setData(payload(d));
  if (d.testY.length) setProbe(testProbe(d, 0));
  store.emit('data');
}

/** A probe for test sample `i` of the loaded dataset. */
export function testProbe(d: Data, i: number): Probe {
  return {
    x: sampleInput(d, 'test', i),
    label: d.testY[i],
    caption: sampleCaption(d.info, 'test', i, d.testY[i]),
    key: `test:${i}`,
    coords: d.points ? d.points.testCoords.slice(i * d.points.dims, (i + 1) * d.points.dims) : undefined,
  };
}

/**
 * Switches dataset. When the kind of data changes (images ↔ points) the architecture and training
 * settings go back to that kind's defaults; otherwise the current architecture is kept if it still
 * fits (for example MNIST → Fashion-MNIST), with its weights only when "keep weights" is on.
 */
export async function setDataset(id: DatasetId): Promise<void> {
  const token = ++datasetToken;
  const prev = store.info;
  const prevNet = store.net;
  store.dataset = id;
  const info = datasetInfo(id);
  store.data = null;
  fullData = null;
  store.custom = [];
  if (info.kind === 'points' && (prev.kind !== 'points' || prev.dims !== info.dims)) store.features = defaultFeatures(info.dims!);
  if (prev.kind !== info.kind || !store.valid) {
    const d = defaultsFor(info);
    store.spec = d.spec;
    setHyper(d.hyper);
  }
  store.emit('dataset');
  rebuild(false, { from: store.keepWeights ? prevNet : null });
  syncCustom();
  let data: Data;
  try {
    if (info.kind === 'image') {
      loading = { id, done: 0, total: 1 };
      store.emit('dataset');
      data = await loadImageDataset(id as 'mnist' | 'fashion' | 'cifar10', (done, total) => {
        if (token !== datasetToken) return;
        loading = { id, done, total };
        store.emit('dataset');
      });
    } else {
      data = pointsData({ id, ...store.pointsConfig } as SyntheticConfig, store.features);
    }
  } finally {
    if (token === datasetToken) loading = null;
  }
  if (token !== datasetToken) return; // a newer switch won
  applyData(data);
}

/** Point datasets: regenerate with new settings (noise, size, split, seed). Weights are kept. */
export function setPointsConfig(patch: Partial<Omit<SyntheticConfig, 'id'>>): void {
  store.pointsConfig = { ...store.pointsConfig, ...patch };
  if (store.info.kind !== 'points') return;
  store.custom = [];
  syncCustom();
  rebuild(false, { from: store.net });
  applyData(pointsData({ id: store.dataset, ...store.pointsConfig } as SyntheticConfig, store.features));
}

/** Point datasets: choose the input features. The first layer changes size, so it starts fresh. */
export function setFeatures(features: FeatureId[]): void {
  if (store.info.kind !== 'points' || !features.length) return;
  store.features = features;
  store.custom = [];
  syncCustom();
  rebuild(false, { from: store.keepWeights ? store.net : null });
  applyData(pointsData({ id: store.dataset, ...store.pointsConfig } as SyntheticConfig, features));
}

/** Train on only the first `limit` training samples (null = all of them). */
export function setTrainLimit(limit: number | null): void {
  store.trainLimit = limit;
  if (fullData) applyData(fullData);
  else store.emit('data');
}

// ── Models ────────────────────────────────────────────────────────────────

const sameShape = (a: Shape, b: Shape) => a.c === b.c && a.h === b.h && a.w === b.w;

/** Parses a model file (from the zoo, a file or browser storage). Throws with a readable message. */
export const readModel = (json: unknown) => decodeModel(json);

/** Loads a model as it is: switches to its dataset and uses its architecture and weights. */
export async function loadModel(json: unknown): Promise<void> {
  const { file, net } = decodeModel(json);
  if (file.dataset !== store.dataset || (file.features && file.features.join() !== store.features.join())) {
    if (file.features) store.features = file.features.slice();
    store.spec = structuredClone(file.arch.layers);
    await setDataset(file.dataset);
    if (file.features && store.info.kind === 'points') setFeatures(file.features.slice());
  }
  if (!sameShape(store.input, file.arch.input) || store.classes !== file.arch.classes) {
    throw new Error(`${file.name} expects ${fmt(file.arch.input)} inputs and ${file.arch.classes} classes; ${store.info.name} has ${fmt(store.input)} and ${store.classes}.`);
  }
  store.spec = structuredClone(file.arch.layers);
  store.keepWeights = true;
  rebuild(false, { weights: net.getWeights(), frozen: file.frozen ?? [] });
  store.modelName = file.name;
  store.emit('model');
}

/**
 * Transfer learning: takes a model's layers and weights into the current dataset, with a new
 * output layer sized for this dataset's classes. The copied layers start frozen, so training
 * first fits only the new head; unfreeze them later to fine-tune.
 */
export function transferModel(json: unknown): void {
  const { file, net } = decodeModel(json);
  if (!sameShape(store.input, file.arch.input)) {
    throw new Error(`${file.name} expects ${fmt(file.arch.input)} inputs, but ${store.info.name} has ${fmt(store.input)}.`);
  }
  store.spec = structuredClone(file.arch.layers);
  store.keepWeights = true;
  const n = file.arch.layers.length + 1;
  rebuild(false, { from: net, copyOutput: false, frozen: Array.from({ length: n }, (_, i) => i < n - 1) });
  store.modelName = `${file.name} → ${store.info.name}`;
  store.emit('model');
}

/** The current model as a file. */
export function exportModel(name: string): ModelFile {
  const last = store.evals[store.evals.length - 1];
  return encodeModel(store.net, {
    name,
    dataset: store.dataset,
    features: store.info.kind === 'points' ? store.features.slice() : undefined,
    frozen: store.frozen.slice(),
    meta: {
      trainedOn: store.info.name,
      samples: store.status?.seen,
      epochs: store.status ? Math.round(store.status.epochFraction * 100) / 100 : 0,
      testAccuracy: last?.acc,
      created: new Date().toISOString(),
    },
  });
}

const fmt = (s: Shape) => (s.h === 1 && s.w === 1 ? `${s.c}-feature` : `${s.h}×${s.w}×${s.c}`);

export function setHyper(patch: Partial<Hyper>): void {
  const next = { ...store.hyper, ...patch };
  if (patch.optimizer && patch.optimizer !== store.hyper.optimizer && patch.lr === undefined) next.lr = DEFAULT_LR[patch.optimizer];
  store.hyper = next;
  client.post({ type: 'hyper', hyper: next });
  store.emit('hyper');
}

export const play = () => client.post({ type: 'play' });
export const pause = () => client.post({ type: 'pause' });
export const stepOnce = () => client.post({ type: 'step' });
export const runEpoch = () => client.post({ type: 'epoch' });

export function setProbe(p: Probe): void {
  store.probe = p;
  store.emit('probe');
}

export function select(block: number, unit: number | null = null): void {
  store.selected = block;
  store.selectedUnit = unit;
  store.emit('select');
}

export function setMode(mode: WeightMode): void {
  store.mode = mode;
  store.emit('mode');
}

let nextCustomId = 1;

function syncCustom(): void {
  // Custom entries hold network inputs; the trainer wants them stored like the dataset.
  const scale = store.data?.scale ?? (store.info.kind === 'image' ? 1 / 255 : 1);
  client.post({
    type: 'custom',
    samples: store.custom.map((c) => ({
      id: c.id,
      y: c.y,
      x: scale === 1 ? Float32Array.from(c.x) : Uint8Array.from(c.x, (v) => Math.max(0, Math.min(255, Math.round(v / scale)))),
    })),
  });
  store.emit('custom');
}

export function addCustom(entry: Omit<CustomEntry, 'id'>): number {
  const id = nextCustomId++;
  store.custom.push({ ...entry, id });
  syncCustom();
  return id;
}

export function removeCustom(id: number): void {
  store.custom = store.custom.filter((c) => c.id !== id);
  syncCustom();
}

/** Replaces the live weights (used when the backprop view applies its update). */
export function applyWeights(weights: Float32Array[]): void {
  store.net.setWeights(weights);
  store.weightsRev++;
  client.post({ type: 'weights', weights: weights.map((w) => w.slice()) });
  store.emit('weights');
}

export function setHighlight(h: typeof store.highlight): void {
  store.highlight = h;
  store.emit('highlight');
}
