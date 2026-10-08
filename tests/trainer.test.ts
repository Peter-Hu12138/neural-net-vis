import { describe, expect, it } from 'vitest';
import { mnistArch } from '../src/nn/types';
import { Network } from '../src/nn/network';
import type { Hyper, LayerSpec } from '../src/nn/types';
import { CUSTOM_REPEAT, EVALS_PER_EPOCH, type DataPayload, type EvalPoint, type FromTrainer, type Status, type TrainPoint } from '../src/train/protocol';
import { Trainer } from '../src/train/trainer';

/** A learnable toy set: digit k is a bright 6×6 block at a position that depends on k. */
function toyData(nTrain: number, nTest: number): DataPayload {
  const make = (n: number, seed: number) => {
    const x = new Uint8Array(n * 784);
    const y = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      const k = (i * 7 + seed) % 10;
      y[i] = k;
      const oy = 2 + Math.floor(k / 5) * 12 + ((i + seed) % 3);
      const ox = 2 + (k % 5) * 5;
      for (let r = 0; r < 6; r++) for (let c = 0; c < 6; c++) x[i * 784 + (oy + r) * 28 + ox + c] = 255;
    }
    return { x, y };
  };
  const tr = make(nTrain, 1);
  const te = make(nTest, 2);
  return { inputSize: 784, scale: 1 / 255, classes: 10, trainX: tr.x, trainY: tr.y, testX: te.x, testY: te.y };
}

function harness(spec: LayerSpec[] = [], hyper: Hyper = { lr: 0.01, batchSize: 10, optimizer: 'adam' }) {
  const log: FromTrainer[] = [];
  const t = new Trainer((m) => log.push(m), 5);
  const net = new Network(mnistArch(spec), 1);
  t.handle({ type: 'model', version: 3, arch: mnistArch(spec), weights: net.getWeights(), hyper, frozen: [] });
  const statuses = () => log.filter((m): m is { type: 'status'; status: Status } => m.type === 'status').map((m) => m.status);
  const metrics = () => log.filter((m): m is { type: 'metrics'; version: number; points: TrainPoint[]; evals: EvalPoint[] } => m.type === 'metrics');
  const evals = () => metrics().flatMap((m) => m.evals);
  const points = () => metrics().flatMap((m) => m.points);
  const last = () => statuses().at(-1)!;
  const weights = () => log.filter((m): m is { type: 'weights'; version: number; step: number; weights: Float32Array[] } => m.type === 'weights');
  return { t, log, net, statuses, evals, points, last, weights };
}

async function until(pred: () => boolean, ms = 10_000) {
  const t0 = Date.now();
  while (!pred()) {
    if (Date.now() - t0 > ms) throw new Error('timed out');
    await new Promise((r) => setTimeout(r, 5));
  }
}

describe('Trainer', () => {
  it('evaluates the untrained model as soon as data arrives', async () => {
    const h = harness();
    h.t.handle({ type: 'data', data: toyData(100, 40) });
    await until(() => h.evals().length > 0);
    const e = h.evals()[0];
    expect(e.step).toBe(0);
    expect(e.epoch).toBe(0);
    expect(e.confusion.reduce((a, b) => a + b, 0)).toBe(40);
    expect(h.log.every((m) => !('version' in m) || m.version === 3)).toBe(true);
  });

  it('trains exactly one batch per step and reports new weights', async () => {
    const h = harness();
    h.t.handle({ type: 'data', data: toyData(100, 20) });
    await until(() => h.evals().length > 0);
    const before = h.weights().length;
    h.t.handle({ type: 'step' });
    expect(h.last().step).toBe(1);
    expect(h.last().seen).toBe(10);
    expect(h.last().running).toBe(false);
    expect(h.weights().length).toBe(before + 1);
    const w = h.weights().at(-1)!.weights;
    expect(w[0]).not.toEqual(h.net.getWeights()[0]);
  });

  it('runs one epoch, evaluates five times along the way, then pauses and learns the task', async () => {
    const h = harness();
    h.t.handle({ type: 'data', data: toyData(200, 50) });
    h.t.handle({ type: 'epoch' });
    await until(() => h.last().epoch === 1 && !h.last().running && h.evals().length >= EVALS_PER_EPOCH + 1);
    expect(h.last().step).toBe(20);
    const ev = h.evals();
    expect(ev.map((e) => e.epoch.toFixed(1))).toEqual(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0']);
    expect(ev.at(-1)!.acc).toBeGreaterThan(ev[0].acc);
    expect(h.points().length).toBeGreaterThan(5);
    // A few more epochs nail the toy task.
    for (let e = 2; e <= 6; e++) {
      h.t.handle({ type: 'epoch' });
      await until(() => h.last().epoch === e && !h.last().running);
    }
    await until(() => h.evals().at(-1)!.epoch >= 6 - 1e-6);
    expect(h.evals().at(-1)!.acc).toBeGreaterThan(0.95);
  });

  it('play keeps training until pause', async () => {
    const h = harness();
    h.t.handle({ type: 'data', data: toyData(500, 20) });
    h.t.handle({ type: 'play' });
    await until(() => h.last().step > 30);
    h.t.handle({ type: 'pause' });
    const stopped = h.last().step;
    expect(h.last().running).toBe(false);
    await new Promise((r) => setTimeout(r, 60));
    expect(h.last().step).toBe(stopped);
  });

  it('mixes custom samples into the epoch, repeated CUSTOM_REPEAT times', async () => {
    const h = harness([], { lr: 0.01, batchSize: 1, optimizer: 'sgd' });
    h.t.handle({ type: 'data', data: toyData(50, 10) });
    const x = new Uint8Array(784).fill(128);
    h.t.handle({ type: 'custom', samples: [{ id: 1, x, y: 3 }, { id: 2, x, y: 4 }] });
    h.t.handle({ type: 'epoch' });
    await until(() => h.last().epoch === 1 && !h.last().running);
    expect(h.last().seen).toBe(50 + 2 * CUSTOM_REPEAT);
  });

  it('switching optimizer or learning rate keeps the weights', async () => {
    const h = harness();
    h.t.handle({ type: 'data', data: toyData(100, 20) });
    h.t.handle({ type: 'step' });
    const w1 = h.weights().at(-1)!.weights[0].slice();
    h.t.handle({ type: 'hyper', hyper: { lr: 0.1, batchSize: 20, optimizer: 'momentum' } });
    h.t.handle({ type: 'step' });
    expect(h.last().seen).toBe(30);
    const w2 = h.weights().at(-1)!.weights[0];
    let diff = 0;
    for (let i = 0; i < w1.length; i++) diff += Math.abs(w2[i] - w1[i]);
    expect(diff).toBeGreaterThan(0);
    expect(diff / w1.length).toBeLessThan(1);
  });

  it('a new model resets counters, history and pauses training', async () => {
    const h = harness();
    h.t.handle({ type: 'data', data: toyData(300, 20) });
    h.t.handle({ type: 'play' });
    await until(() => h.last().step > 5);
    const spec: LayerSpec[] = [{ kind: 'dense', units: 8, act: 'tanh' }];
    h.t.handle({ type: 'model', version: 4, arch: mnistArch(spec), weights: new Network(mnistArch(spec), 2).getWeights(), hyper: { lr: 0.001, batchSize: 32, optimizer: 'adam' }, frozen: [] });
    const s = h.last();
    expect(s.version).toBe(4);
    expect(s.step).toBe(0);
    expect(s.running).toBe(false);
  });

  it('accepts weights pushed from the backprop view and re-evaluates', async () => {
    const h = harness();
    h.t.handle({ type: 'data', data: toyData(100, 20) });
    await until(() => h.evals().length === 1);
    const ws = h.net.getWeights().map((w) => w.map(() => 0));
    h.t.handle({ type: 'weights', weights: ws });
    await until(() => h.evals().length === 2);
    // All-zero weights give uniform probabilities: loss = ln 10.
    expect(h.evals()[1].loss).toBeCloseTo(Math.log(10), 4);
  });
});
