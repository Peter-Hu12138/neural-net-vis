import { Network } from './nn/network';
import { DEFAULT_LR } from './nn/optim';
import { analysis } from './analysis/client';
import { datasetInfo, featureShape, limitTraining, pointsData, sampleCaption, sampleInput, type Data, type DatasetId } from './data/datasets';
import { defaultFeatures, featureCatalog, featurize, type FeatureId } from './data/features';
import { loadImageDataset } from './data/images';
import type { SyntheticConfig } from './data/synthetic';
import { decodeModel, encodeModel, type ModelFile } from './models/format';
import type { Hyper, LayerSpec, Shape } from './nn/types';
import { defaultsFor, store, type CustomEntry, type Probe, type WeightMode } from './store';
import { TrainerClient } from './train/client';
import { SPEEDS, type FromTrainer, type Speed } from './train/protocol';

let weightsQueued = false;
/**
 * The trainer's id for the weights store.net holds (see EvalPoint.weightsId), or null right after
 * a change on the page (a rebuild, a manual update) that the trainer has not sent back yet.
 */
let heldWeightsId: number | null = null;

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
      // An evaluation finished just before the data changed measured the data that is gone: it
      // must not stand as the current data's test accuracy, even for a moment.
      store.evals.push(...m.evals.filter((e) => e.dataRev === store.dataRev));
      store.emit('metrics');
      break;
    case 'weights':
      if (m.version !== store.version) return;
      // The trainer resends unchanged weights after pauses and evaluations; only real changes count.
      if (m.step !== store.weightsStep || !sameWeights(store.net, m.weights)) store.weightsRev++;
      store.net.setWeights(m.weights);
      store.weightsStep = m.step;
      heldWeightsId = m.weightsId;
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
  /**
   * Start the experiment again on the same network: back to the weights it started from (its
   * random initialisation, or the weights as loaded or transferred), keeping its frozen flags.
   */
  restart?: boolean;
}

/** Builds a network for the current dataset and spec, sends it to the trainer, resets the history. */
export function rebuild(newSeed = false, opts: RebuildOptions = {}): void {
  if (!store.valid) return;
  if (newSeed) {
    store.seed = (store.seed * 7919 + 17) % 100003;
    store.modelName = null;
  }
  const oldFrozen = store.frozen;
  // A restart keeps the network object, so everything that tells networks apart (the model
  // panel's account of where the weights came from) sees the same network starting over.
  const restart = !!opts.restart && !newSeed && JSON.stringify(store.net.arch) === JSON.stringify(store.arch);
  store.version++;
  heldWeightsId = null;
  if (!restart) store.net = new Network(store.arch, store.seed);
  let kept = store.net.blocks.map(() => false);
  if (restart) {
    store.net.setWeights(initialWeights);
    kept = kept.map(() => true);
  } else if (opts.weights) {
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
  if (!restart) store.selectedUnit = null;
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

/**
 * How the data changed: another dataset, another training subset of the same one, other input
 * features for the same points, or regenerated points (noise, count, share, seed).
 */
type DataChange = 'dataset' | 'subset' | 'features' | 'points';

function applyData(data: Data, change: DataChange = 'dataset'): void {
  fullData = data;
  const d = limitTraining(data, store.trainLimit);
  store.data = d;
  client.post({ type: 'data', data: payload(d), rev: ++store.dataRev });
  analysis.setData(payload(d));
  const kept = change === 'dataset' ? null : carriedProbe(store.probe, d, change);
  if (kept) setProbe(kept);
  else if (d.testY.length) setProbe(testProbe(d, 0));
  store.emit('data');
}

/**
 * Takes the data away from the trainer before a change that also brings a new network (another
 * dataset, other features, a restart). Without this, the trainer measured the new network on the
 * old data while the new data was still loading or on its way, and the page recorded that figure
 * as the new data's first test accuracy.
 */
function dropTrainerData(): void {
  client.post({ type: 'data', data: null, rev: ++store.dataRev });
}

/**
 * The reader's current input carried over to changed data, or null when it no longer applies.
 * A new training subset leaves every input valid except a training sample that left the subset.
 * New features recompute a point's inputs from its coordinates. Regenerated points keep inputs
 * placed by coordinates (your points, a spot on the plane) but not dataset samples: sample #i is
 * another point now.
 */
function carriedProbe(p: Probe | null, d: Data, change: Exclude<DataChange, 'dataset'>): Probe | null {
  if (!p) return null;
  const m = /^(train|test):(\d+)$/.exec(p.key);
  if (m) {
    const split = m[1] as 'train' | 'test';
    const i = Number(m[2]);
    if (change === 'points' || i >= (split === 'test' ? d.testY : d.trainY).length) return null;
    return change === 'features' ? sampleProbe(d, split, i) : p;
  }
  if (!d.points) return p.x.length === d.inputSize ? p : null;
  if (!p.coords) return null;
  return change === 'features' ? { ...p, x: featurize(p.coords, d.points.dims, d.points.features) } : p;
}

/** A probe for test sample `i` of the loaded dataset. */
export function testProbe(d: Data, i: number): Probe {
  return sampleProbe(d, 'test', i);
}

/** A probe for sample `i` of a split of the loaded dataset. */
export function sampleProbe(d: Data, split: 'train' | 'test', i: number): Probe {
  const y = (split === 'test' ? d.testY : d.trainY)[i];
  const p = d.points;
  return {
    x: sampleInput(d, split, i),
    label: y,
    caption: sampleCaption(d.info, split, i, y),
    key: `${split}:${i}`,
    coords: p ? (split === 'test' ? p.testCoords : p.trainCoords).slice(i * p.dims, (i + 1) * p.dims) : undefined,
  };
}

/**
 * Switches dataset. When the input changes shape (images ↔ points, grey ↔ colour) the architecture
 * and training settings go back to that data's defaults; otherwise the current architecture is kept
 * if it still fits (for example MNIST → Fashion-MNIST), with its weights only when "keep weights" is on.
 */
export async function setDataset(id: DatasetId): Promise<void> {
  const token = ++datasetToken;
  const prev = store.info;
  const prevNet = store.net;
  store.dataset = id;
  const info = datasetInfo(id);
  store.data = null;
  fullData = null;
  dropTrainerData();
  // The current input belongs to the old dataset: its caption, its label (02 would mark that class
  // index among the new classes) and maybe its shape. Until the new data is in (test sample #0
  // then), there is no current input, as when the page first loads.
  setProbe(null);
  store.custom = [];
  if (info.kind === 'points' && (prev.kind !== 'points' || prev.dims !== info.dims)) store.features = defaultFeatures(info.dims!);
  // A new input shape (images ↔ points, grey ↔ colour) gets that data's default network.
  const sameInput = prev.kind === info.kind && (info.kind === 'points' || JSON.stringify(prev.image!.shape) === JSON.stringify(info.image!.shape));
  if (!sameInput || !store.valid) {
    const d = defaultsFor(info);
    store.spec = d.spec;
    setHyper(d.hyper);
    if (prev.kind !== info.kind) setSpeed(d.speed);
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

/**
 * Point datasets: regenerate with new settings (noise, size, split, seed). Your points stay: they
 * live in the same coordinate space as the generated ones.
 *
 * The network's weights and their history go together. New points for a training set of the same
 * size (noise, Regenerate) continue the run: the same network, step, epochs and curves, first
 * evaluated on the new points. A training set of another size (count, share) is another
 * experiment, as with "Train on": it restarts from the weights it started from, with a fresh
 * history on an epoch axis of one unit. Before, every change kept the trained weights but erased
 * their history, so the page called a trained network a random start, before any training.
 */
export function setPointsConfig(patch: Partial<Omit<SyntheticConfig, 'id'>>): void {
  store.pointsConfig = { ...store.pointsConfig, ...patch };
  if (store.info.kind !== 'points') return;
  const data = pointsData({ id: store.dataset, ...store.pointsConfig } as SyntheticConfig, store.features);
  if (!store.data || limitTraining(data, store.trainLimit).trainY.length !== store.data.trainY.length) {
    dropTrainerData();
    rebuild(false, { restart: true });
  }
  applyData(data, 'points');
}

/**
 * Point datasets: choose the input features. The first layer changes size, so it starts fresh.
 * Your points keep their coordinates; their inputs are recomputed with the new features.
 */
export function setFeatures(features: FeatureId[]): void {
  const dims = store.info.dims;
  if (store.info.kind !== 'points' || !dims || !features.length) return;
  store.features = features;
  store.custom = store.custom.map((c) => (c.coords ? { ...c, x: featurize(c.coords, dims, features) } : c));
  dropTrainerData(); // with as many features as before, the new network would fit the old inputs
  rebuild(false, { from: store.keepWeights ? store.net : null });
  applyData(pointsData({ id: store.dataset, ...store.pointsConfig } as SyntheticConfig, features), 'features');
  // After the new model and data: the trainer must never pair inputs of two different sizes.
  syncCustom();
}

/**
 * Train on only the first `limit` training samples (null = all of them). Another training set is
 * another experiment: the network starts again from its starting weights (as initialised, loaded
 * or transferred) with fresh counters and curves. Carrying on from the trained weights would show
 * accuracy learned from the other set as this one's, on an epoch axis that silently changed unit.
 */
export function setTrainLimit(limit: number | null): void {
  store.trainLimit = limit;
  if (!fullData) return store.emit('data');
  dropTrainerData();
  rebuild(false, { restart: true });
  applyData(fullData, 'subset');
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
 * Why a model cannot be transferred to the current data, or null when it can. A point model's
 * first layer was trained on particular input features (x₁², x₂², …); copied and frozen, it only
 * means something when fed those same features, so they come with it (transferModel switches to
 * them) and its input is checked against them, not against the features in use now.
 */
export function transferProblem(file: Pick<ModelFile, 'name' | 'arch' | 'features'>): string | null {
  const info = store.info;
  const features = info.kind === 'points' && file.features ? file.features : null;
  if (features) {
    const have = new Set(featureCatalog(info.dims!).map((f) => f.id));
    const missing = features.filter((f) => !have.has(f));
    if (missing.length) {
      const label = (id: FeatureId) => featureCatalog(3).find((f) => f.id === id)?.label ?? id;
      return `${file.name} was trained on ${missing.map(label).join(', ')}, which ${info.name} does not have: its points have ${info.dims} coordinates.`;
    }
  }
  const input = features ? featureShape(features) : store.input;
  return sameShape(input, file.arch.input) ? null : `${file.name} expects ${fmt(file.arch.input)} inputs, but ${info.name} has ${fmt(input)}.`;
}

/**
 * Transfer learning: takes a model's layers and weights into the current dataset, with a new
 * output layer sized for this dataset's classes. The copied layers start frozen, so training
 * first fits only the new head; unfreeze them later to fine-tune.
 */
export function transferModel(json: unknown): void {
  const { file, net } = decodeModel(json);
  const problem = transferProblem(file);
  if (problem) throw new Error(problem);
  const features = store.info.kind === 'points' && file.features ? file.features : null;
  if (features && features.join() !== store.features.join()) setFeatures(features.slice());
  store.spec = structuredClone(file.arch.layers);
  store.keepWeights = true;
  const n = file.arch.layers.length + 1;
  rebuild(false, { from: net, copyOutput: false, frozen: Array.from({ length: n }, (_, i) => i < n - 1) });
  store.modelName = `${file.name} → ${store.info.name}`;
  store.emit('model');
}

/**
 * The current model as a file. Its test accuracy is included only when the last evaluation was of
 * exactly these weights on the current data; an older one (training ran on, the weights were
 * edited, or the points were redrawn since) would be saved as a figure for weights or data it
 * never saw.
 */
export function exportModel(name: string): ModelFile {
  const last = store.evals[store.evals.length - 1];
  const measured = !!last && heldWeightsId !== null && last.weightsId === heldWeightsId && last.dataRev === store.dataRev;
  return encodeModel(store.net, {
    name,
    dataset: store.dataset,
    features: store.info.kind === 'points' ? store.features.slice() : undefined,
    frozen: store.frozen.slice(),
    meta: {
      trainedOn: store.info.name,
      samples: store.status?.seen,
      epochs: store.status ? Math.round(store.status.epochFraction * 100) / 100 : 0,
      testAccuracy: measured ? last.acc : undefined,
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

/** Caps training speed ('slow', 'normal') or lifts the cap ('max'). */
export function setSpeed(speed: Speed): void {
  store.speed = speed;
  client.post({ type: 'speed', samplesPerSec: SPEEDS[speed] });
  store.emit('hyper');
}

export const play = () => client.post({ type: 'play' });
export const pause = () => client.post({ type: 'pause' });
export const stepOnce = () => client.post({ type: 'step' });
export const runEpoch = () => client.post({ type: 'epoch' });

export function setProbe(p: Probe | null): void {
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
  heldWeightsId = null; // until the trainer, which evaluates them, sends them back
  client.post({ type: 'weights', weights: weights.map((w) => w.slice()) });
  store.emit('weights');
}

export function setHighlight(h: typeof store.highlight): void {
  store.highlight = h;
  store.emit('highlight');
}
