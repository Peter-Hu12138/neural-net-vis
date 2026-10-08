import { Network } from './nn/network';
import { DEFAULT_LR } from './nn/optim';
import type { Hyper, LayerSpec } from './nn/types';
import { store, type CustomEntry, type Probe, type WeightMode } from './store';
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

/** Builds a fresh network from the current spec and sends it to the trainer. */
export function rebuild(newSeed = false): void {
  if (!store.valid) return;
  if (newSeed) store.seed = (store.seed * 7919 + 17) % 100003;
  store.version++;
  store.net = new Network(store.spec, store.seed);
  initialWeights = store.net.getWeights();
  store.weightsStep = 0;
  store.weightsRev++;
  store.points = [];
  store.evals = [];
  store.status = null;
  if (store.selected >= store.net.blocks.length) store.selected = store.net.blocks.length - 1;
  store.selectedUnit = null;
  client.post({ type: 'model', version: store.version, spec: store.spec, weights: store.net.getWeights(), hyper: store.hyper });
  store.emit('model');
  store.emit('weights');
  store.emit('metrics');
  store.emit('status');
}

export function setSpec(spec: LayerSpec[]): void {
  store.spec = spec;
  if (store.valid) rebuild();
  else store.emit('model');
}

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
  client.post({
    type: 'custom',
    samples: store.custom.map((c) => ({ id: c.id, y: c.y, x: Uint8Array.from(c.x, (v) => Math.round(v * 255)) })),
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
