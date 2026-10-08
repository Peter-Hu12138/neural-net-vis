import { describe as suite, expect, it } from 'vitest';
import { datasetInfo, limitTraining, pointsData, sampleCaption, sampleInput, DATASETS } from '../src/data/datasets';
import { decodeModel, encodeModel } from '../src/models/format';
import { Network, argmax, describe } from '../src/nn/network';
import { Optimizer } from '../src/nn/optim';
import { Rng } from '../src/nn/rng';
import { mnistArch, type Arch, type LayerSpec } from '../src/nn/types';
import type { FromTrainer, Status } from '../src/train/protocol';
import { Trainer } from '../src/train/trainer';

const CIFAR: Arch = {
  input: { c: 3, h: 32, w: 32 },
  layers: [
    { kind: 'conv', filters: 4, kernel: 3, act: 'tanh', pool: true },
    { kind: 'conv', filters: 4, kernel: 3, act: 'tanh', pool: true },
    { kind: 'dense', units: 8, act: 'tanh' },
  ],
  classes: 10,
};

const randomInput = (n: number, seed: number) => {
  const r = new Rng(seed);
  return Float32Array.from({ length: n }, () => r.next());
};

suite('architectures beyond MNIST', () => {
  it('colour images: shapes and gradients through a 3-channel input (finite differences)', () => {
    const info = describe(CIFAR);
    expect(info.map((l) => [l.outShape.h, l.outShape.w, l.outShape.c])).toEqual([
      [16, 16, 4],
      [8, 8, 4],
      [1, 1, 8],
      [1, 1, 10],
    ]);
    expect(info[0].params).toBe(4 * 3 * 9 + 4);
    // Gradients are checked without pooling: max-pool winners flip under finite differences,
    // which would blur an exact comparison (pooling itself is covered by tests/nn.test.ts).
    const net = new Network({ ...CIFAR, layers: CIFAR.layers.map((l) => (l.kind === 'conv' ? { ...l, pool: false } : l)) }, 3);
    const x = randomInput(3072, 1);
    net.zeroGrad();
    net.forward(x);
    net.backward(6, true);
    const g = net.blocks[0].dX.slice();
    const gW = net.blocks[0].gW.slice();
    const eps = 3e-3;
    const loss = () => (net.forward(x), net.loss(6));
    const r = new Rng(9);
    let num = 0;
    let den = 0;
    for (let t = 0; t < 30; t++) {
      const i = r.int(3072);
      const o = x[i];
      x[i] = o + eps;
      const p = loss();
      x[i] = o - eps;
      const m = loss();
      x[i] = o;
      const fd = (p - m) / (2 * eps);
      num += (fd - g[i]) ** 2;
      den += fd ** 2 + g[i] ** 2;
    }
    for (let t = 0; t < 20; t++) {
      const i = r.int(gW.length);
      const w = net.blocks[0].W;
      const o = w[i];
      w[i] = o + eps;
      const p = loss();
      w[i] = o - eps;
      const m = loss();
      w[i] = o;
      const fd = (p - m) / (2 * eps);
      num += (fd - gW[i]) ** 2;
      den += fd ** 2 + gW[i] ** 2;
    }
    expect(Math.sqrt(num / den)).toBeLessThan(2e-3);
  });

  it('feature vectors: dense layers only; convolutions are flagged', () => {
    const arch: Arch = { input: { c: 2, h: 1, w: 1 }, layers: [{ kind: 'dense', units: 4, act: 'tanh' }], classes: 3 };
    expect(describe(arch).every((l) => !l.error)).toBe(true);
    const net = new Network(arch, 1);
    expect(net.forward(new Float32Array([0.3, -0.2]))).toHaveLength(3);
    const bad = describe({ ...arch, layers: [{ kind: 'conv', filters: 2, kernel: 3, act: 'relu', pool: false }] });
    expect(bad[0].error).toMatch(/image input/);
  });
});

suite('freezing and transfer', () => {
  const spec: LayerSpec[] = [
    { kind: 'conv', filters: 4, kernel: 3, act: 'relu', pool: true },
    { kind: 'dense', units: 16, act: 'relu' },
  ];

  it('frozen blocks keep their weights; trainable ones change', () => {
    const net = new Network(mnistArch(spec), 2);
    const frozen = [true, true, false];
    const before = net.getWeights();
    const opt = new Optimizer(net, 'adam', 0.01);
    for (let s = 0; s < 5; s++) {
      net.zeroGrad();
      for (let k = 0; k < 4; k++) {
        net.forward(randomInput(784, s * 10 + k));
        net.backward(k, false, frozen);
      }
      opt.step(1 / 4, frozen);
    }
    const after = net.getWeights();
    expect(after[0]).toEqual(before[0]);
    expect(after[1]).toEqual(before[1]);
    expect(after[2]).toEqual(before[2]);
    expect(after[3]).toEqual(before[3]);
    expect(after[4]).not.toEqual(before[4]);
    // Frozen blocks get no gradient at all: the pass stops above them.
    expect(net.blocks[0].gW.every((v) => v === 0)).toBe(true);
  });

  it('training only the head never runs backprop through the frozen layers below it', () => {
    const net = new Network(mnistArch([{ kind: 'conv', filters: 4, kernel: 3, act: 'relu', pool: true }, { kind: 'dense', units: 8, act: 'relu' }]), 1);
    const x = randomInput(784, 4);
    const run = (frozen: boolean[]) => {
      for (const b of net.blocks) b.dZ.fill(NaN);
      net.forward(x);
      net.backward(1, false, frozen);
      return net.blocks.map((b) => !Number.isNaN(b.dZ[0]));
    };
    // Which blocks the backward pass touched (computed their δ):
    expect(run([true, true, false])).toEqual([false, false, true]);
    expect(run([true, false, false])).toEqual([false, true, true]);
    expect(run([false, false, false])).toEqual([true, true, true]);
    // A frozen block above a trainable one still passes the gradient down.
    expect(run([false, true, false])).toEqual([true, true, true]);
  });

  it('copyCompatible keeps matching layers, and a changed layer starts fresh', () => {
    const a = new Network(mnistArch(spec), 1);
    const b = new Network({ ...mnistArch(spec), classes: 5 }, 9);
    const copied = b.copyCompatible(a);
    expect(copied).toEqual([true, true, false]);
    expect(b.blocks[0].W).toEqual(a.blocks[0].W);
    expect(b.blocks[1].W).toEqual(a.blocks[1].W);
    const c = new Network(mnistArch([spec[0], { kind: 'dense', units: 8, act: 'relu' }]), 9);
    expect(c.copyCompatible(a)).toEqual([true, false, false]);
  });
});

suite('model files', () => {
  it('round-trip architecture, weights and metadata', () => {
    const net = new Network(CIFAR, 7);
    const file = encodeModel(net, { name: 'Test', dataset: 'cifar10', frozen: [true, false, false, false], meta: { testAccuracy: 0.5 } });
    const json = JSON.parse(JSON.stringify(file));
    const { file: f, net: back } = decodeModel(json);
    expect(f.name).toBe('Test');
    expect(back.arch).toEqual(CIFAR);
    expect(back.getWeights()).toEqual(net.getWeights());
    const x = randomInput(3072, 2);
    expect(Array.from(back.forward(x))).toEqual(Array.from(net.forward(x)));
  });

  it('reject broken files with a message a reader can act on', () => {
    const good = JSON.parse(JSON.stringify(encodeModel(new Network(mnistArch([]), 1), { name: 'x', dataset: 'mnist', meta: {} })));
    expect(() => decodeModel({ hello: 1 })).toThrow(/not a Raster model/);
    expect(() => decodeModel({ ...good, version: 2 })).toThrow(/version 2/);
    expect(() => decodeModel({ ...good, weights: good.weights.slice(1) })).toThrow(/Expected 2 weight arrays/);
    expect(() => decodeModel({ ...good, arch: { ...good.arch, classes: 5 } })).toThrow(/Layer 1 has 7840 weights; the architecture needs 3920/);
    expect(() => decodeModel({ ...good, arch: { ...good.arch, layers: [{ kind: 'pool' }] } })).toThrow(/invalid architecture/);
    const nan = new Float32Array(10).fill(NaN);
    const bytes = new Uint8Array(nan.buffer);
    expect(() => decodeModel({ ...good, weights: [good.weights[0], btoa(String.fromCharCode(...bytes))] })).toThrow(/not finite/);
  });
});

suite('datasets', () => {
  it('registry: every dataset has class names, glyphs and a shape or dimensionality', () => {
    for (const d of DATASETS) {
      expect(d.classes.length).toBe(d.glyphs.length);
      expect(d.classes.length).toBeGreaterThanOrEqual(2);
      if (d.kind === 'image') expect(d.image!.shape.h).toBeGreaterThan(1);
      else expect([2, 3]).toContain(d.dims);
    }
    expect(datasetInfo('cifar10').image!.shape).toEqual({ c: 3, h: 32, w: 32 });
  });

  it('point data: features as inputs, coordinates kept for plotting, captions and subsets', () => {
    const d = pointsData({ id: 'blobs', count: 300, noise: 0.1, trainRatio: 0.5, seed: 1 }, ['x1', 'x2', 'x1*x2']);
    expect(d.inputSize).toBe(3);
    expect(d.scale).toBe(1);
    expect(d.trainX.length).toBe(150 * 3);
    expect(d.points!.trainCoords.length).toBe(150 * 2);
    const x = sampleInput(d, 'train', 4);
    const p = d.points!.trainCoords;
    expect(x[2]).toBeCloseTo(p[8] * p[9], 6);
    expect(sampleCaption(d.info, 'test', 3, 2)).toBe('Test point #3 · Class 2');
    expect(sampleCaption(datasetInfo('mnist'), 'test', 0, 7)).toBe('Test digit #0 · label 7');
    expect(sampleCaption(datasetInfo('cifar10'), 'test', 5, 3)).toBe('Test image #5 · cat');
    const small = limitTraining(d, 20);
    expect(small.trainY.length).toBe(20);
    expect(small.trainX.length).toBe(60);
    expect(small.points!.trainCoords.length).toBe(40);
    expect(small.testY).toBe(d.testY);
  });

  it('the trainer learns a point dataset end to end (float inputs, 3 classes)', async () => {
    const d = pointsData({ id: 'blobs', count: 600, noise: 0.05, trainRatio: 0.5, seed: 2 });
    const arch: Arch = { input: d.input, layers: [{ kind: 'dense', units: 8, act: 'tanh' }], classes: 3 };
    const log: FromTrainer[] = [];
    const t = new Trainer((m) => log.push(m), 5);
    t.handle({ type: 'model', version: 1, arch, weights: new Network(arch, 1).getWeights(), hyper: { lr: 0.03, batchSize: 10, optimizer: 'adam' }, frozen: [] });
    t.handle({ type: 'data', data: { inputSize: d.inputSize, scale: 1, classes: 3, trainX: d.trainX, trainY: d.trainY, testX: d.testX, testY: d.testY } });
    const last = () => (log.filter((m) => m.type === 'status').at(-1) as { status: Status }).status;
    for (let e = 1; e <= 10; e++) {
      t.handle({ type: 'epoch' });
      for (let i = 0; i < 2000 && !(last().epoch === e && !last().running); i++) await new Promise((r) => setTimeout(r, 2));
    }
    for (let i = 0; i < 2000; i++) {
      const evals = log.flatMap((m) => (m.type === 'metrics' ? m.evals : []));
      if (evals.at(-1) && evals.at(-1)!.epoch >= 10 - 1e-9) break;
      await new Promise((r) => setTimeout(r, 2));
    }
    const evals = log.flatMap((m) => (m.type === 'metrics' ? m.evals : []));
    expect(evals.at(-1)!.confusion).toHaveLength(9);
    expect(evals.at(-1)!.acc).toBeGreaterThan(0.95);
  });

  it('the trainer leaves frozen layers alone and waits for matching data', async () => {
    const spec: LayerSpec[] = [{ kind: 'dense', units: 6, act: 'relu' }];
    const arch = mnistArch(spec);
    const net = new Network(arch, 1);
    const log: FromTrainer[] = [];
    const t = new Trainer((m) => log.push(m), 5);
    t.handle({ type: 'model', version: 1, arch, weights: net.getWeights(), hyper: { lr: 0.01, batchSize: 4, optimizer: 'sgd' }, frozen: [true, false] });
    // Data of the wrong size (a point dataset): nothing may train.
    t.handle({ type: 'data', data: { inputSize: 2, scale: 1, classes: 2, trainX: new Float32Array(20), trainY: new Uint8Array(10), testX: new Float32Array(4), testY: new Uint8Array(2) } });
    t.handle({ type: 'step' });
    expect(log.some((m) => m.type === 'weights')).toBe(false);
    const x = new Uint8Array(784 * 8).map((_, i) => (i * 37) % 256);
    t.handle({ type: 'data', data: { inputSize: 784, scale: 1 / 255, classes: 10, trainX: x, trainY: new Uint8Array(8).map((_, i) => i), testX: x, testY: new Uint8Array(8) } });
    t.handle({ type: 'step' });
    const w = (log.filter((m) => m.type === 'weights').at(-1) as { weights: Float32Array[] }).weights;
    expect(w[0]).toEqual(net.blocks[0].W);
    expect(w[2]).not.toEqual(net.blocks[1].W);
    t.handle({ type: 'frozen', frozen: [false, false] });
    t.handle({ type: 'step' });
    const w2 = (log.filter((m) => m.type === 'weights').at(-1) as { weights: Float32Array[] }).weights;
    expect(w2[0]).not.toEqual(net.blocks[0].W);
    expect(argmax(new Float32Array([0, 3, 1]))).toBe(1);
  });
});
