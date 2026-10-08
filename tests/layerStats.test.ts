import { readFileSync } from 'node:fs';
import { inflateSync } from 'node:zlib';
import { describe, expect, it } from 'vitest';
import { layerStats, type LayerStatsResult } from '../src/analysis/layerStats';
import type { JobContext, Progress } from '../src/analysis/protocol';
import { registry } from '../src/analysis/registry';
import { Network } from '../src/nn/network';
import type { LayerSpec } from '../src/nn/types';

/** The bundled MNIST test split (2,000 digits), decoded from its PNG sprite sheet. */
function loadTestSet() {
  const buf = readFileSync('public/data/mnist-test.png');
  let off = 8;
  let width = 0;
  let height = 0;
  const idat: Buffer[] = [];
  while (off < buf.length) {
    const len = buf.readUInt32BE(off);
    const type = buf.toString('ascii', off + 4, off + 8);
    const data = buf.subarray(off + 8, off + 8 + len);
    if (type === 'IHDR') {
      width = data.readUInt32BE(0);
      height = data.readUInt32BE(4);
    }
    if (type === 'IDAT') idat.push(data);
    off += 12 + len;
  }
  const raw = inflateSync(Buffer.concat(idat));
  const px = (x: number, y: number) => raw[y * (width + 1) + 1 + x];
  const n = (height / 28) * (width / 28);
  const testX = new Uint8Array(n * 784);
  for (let i = 0; i < n; i++) {
    const ox = (i % 100) * 28;
    const oy = Math.floor(i / 100) * 28;
    for (let r = 0; r < 28; r++) for (let c = 0; c < 28; c++) testX[i * 784 + r * 28 + c] = px(ox + c, oy + r);
  }
  const labels = readFileSync('public/data/mnist-labels.txt', 'utf8').trim();
  const testY = Uint8Array.from(labels.slice(20_000), (ch) => ch.charCodeAt(0) - 48);
  return { testX, testY };
}

const mnist = loadTestSet();

function context(spec: LayerSpec[], seed = 1, data = mnist): JobContext {
  const net = new Network(spec, seed);
  const { testX, testY } = data;
  return {
    net,
    spec,
    testX,
    testY,
    image(i, out = new Float32Array(784)) {
      for (let j = 0; j < 784; j++) out[j] = testX[i * 784 + j] / 255;
      return out;
    },
  };
}

function drain<R>(gen: Generator<Progress, R, void>) {
  const progress: Progress[] = [];
  for (;;) {
    const r = gen.next();
    if (r.done) return { result: r.value, progress };
    progress.push(r.value);
  }
}

const SMALL_CNN: LayerSpec[] = [
  { kind: 'conv', filters: 8, kernel: 3, act: 'relu', pool: true },
  { kind: 'conv', filters: 16, kernel: 3, act: 'relu', pool: true },
  { kind: 'dense', units: 32, act: 'relu' },
];

describe('layerStats job', () => {
  it('is registered under "layerStats"', () => {
    expect(registry.layerStats).toBe(layerStats);
  });

  it('returns sampled z/a, activity and gradients with the right shapes', () => {
    const ctx = context(SMALL_CNN);
    const { result, progress } = drain(layerStats(ctx, { samples: 40, maxValues: 5000 }));
    expect(progress).toHaveLength(40); // one yield per image
    expect(progress.at(-1)).toEqual({ done: 40, total: 40 });
    expect(result.samples).toBe(40);
    expect(result.layers.map((l) => l.block)).toEqual([0, 1, 2, 3]);
    const blocks = ctx.net.blocks;
    result.layers.forEach((l, i) => {
      const b = blocks[i];
      const total = 40 * b.z.length;
      expect(l.seen).toBe(total);
      expect(l.z.length).toBe(Math.min(5000, total));
      expect(l.a.length).toBe(l.z.length);
      expect(l.gW.length).toBe(b.W.length);
      expect(l.gb.length).toBe(b.b.length);
      expect(l.activeFraction.length).toBe(b.kind === 'conv' ? b.spec.filters : b.spec.units);
      for (const f of l.activeFraction) {
        expect(f).toBeGreaterThanOrEqual(0);
        expect(f).toBeLessThanOrEqual(1);
      }
    });
    // Conv 1 sees 40 × 8 × 28 × 28 values: the reservoir caps it; the output layer keeps all 400.
    expect(result.layers[0].z.length).toBe(5000);
    expect(result.layers[3].z.length).toBe(400);
    // Same slots for z and a: a = relu(z) for the hidden layers, a = z (logits) for the output.
    for (const l of result.layers.slice(0, 3)) for (let i = 0; i < l.z.length; i++) expect(l.a[i]).toBe(Math.max(0, l.z[i]));
    expect(Array.from(result.layers[3].a)).toEqual(Array.from(result.layers[3].z));
    expect(result.layers.map((l) => l.dead === null)).toEqual([false, false, false, true]);
    expect(result.layers[3].act).toBe('linear');
  });

  it('keeps every value of the output layer and reports the logits of the chosen images', () => {
    const ctx = context([], 3);
    const { result } = drain(layerStats(ctx, { samples: 4 }));
    // Images at evenly spaced indices 0, 500, 1000, 1500; nothing is dropped below maxValues.
    const check = new Network([], 3);
    const expected: number[] = [];
    for (const idx of [0, 500, 1000, 1500]) {
      check.forward(ctx.image(idx));
      expected.push(...check.output.z);
    }
    const got = result.layers[0].z;
    expect(got.length).toBe(40);
    for (let i = 0; i < 40; i++) expect(got[i]).toBeCloseTo(expected[i], 5);
  });

  it('the active fraction counts image × position entries with a > 0', () => {
    const ctx = context(SMALL_CNN, 2);
    const { result } = drain(layerStats(ctx, { samples: 3, maxValues: 1e9 }));
    // Brute force over every value of Conv 1.
    const check = new Network(SMALL_CNN, 2);
    const counts = new Array(8).fill(0);
    for (const idx of [0, 666, 1333]) {
      check.forward(ctx.image(idx));
      const a = check.blocks[0].a;
      for (let f = 0; f < 8; f++) for (let i = 0; i < 784; i++) if (a[f * 784 + i] > 0) counts[f]++;
    }
    result.layers[0].activeFraction.forEach((frac, f) => expect(frac).toBeCloseTo(counts[f] / (3 * 784), 6));
    // With room for everything, the sample is every value in order.
    expect(result.layers[0].z.length).toBe(3 * 8 * 784);
  });

  it('gW and gb equal the batch mean of a direct forward/backward over the same images', () => {
    const spec: LayerSpec[] = [
      { kind: 'conv', filters: 4, kernel: 3, act: 'tanh', pool: true },
      { kind: 'dense', units: 12, act: 'relu' },
    ];
    const ctx = context(spec, 5);
    const { result } = drain(layerStats(ctx, { samples: 2 }));
    const ref = new Network(spec, 5);
    ref.zeroGrad();
    for (const idx of [0, 1000]) {
      ref.forward(ctx.image(idx));
      ref.backward(ctx.testY[idx]);
    }
    result.layers.forEach((l, i) => {
      const b = ref.blocks[i];
      let worst = 0;
      for (let j = 0; j < b.gW.length; j++) worst = Math.max(worst, Math.abs(l.gW[j] - b.gW[j] / 2));
      expect(worst).toBeLessThan(1e-7);
      for (let j = 0; j < b.gb.length; j++) expect(l.gb[j]).toBeCloseTo(b.gb[j] / 2, 7);
    });
    expect(result.layers[0].dead).toBeNull(); // tanh
    expect(result.layers[1].dead).not.toBeNull(); // ReLU
  });

  it('counts a ReLU unit that can never fire as dead', () => {
    const spec: LayerSpec[] = [{ kind: 'dense', units: 4, act: 'relu' }];
    const ctx = context(spec, 1);
    const b = ctx.net.blocks[0];
    const n = 784;
    // Pixels are ≥ 0, so negative weights and a negative bias keep unit 2 below zero for every image.
    for (let i = 0; i < n; i++) {
      b.W[0 * n + i] = 0.01;
      b.W[1 * n + i] = 0.01;
      b.W[2 * n + i] = -0.01;
      b.W[3 * n + i] = 0.01;
    }
    b.b.set([0.1, 0.1, -0.1, 0.1]);
    const { result } = drain(layerStats(ctx, { samples: 50 }));
    const l = result.layers[0];
    expect(Array.from(l.activeFraction)).toEqual([1, 1, 0, 1]);
    expect(l.dead).toBe(1);
    // A dead unit gets no gradient at all.
    for (let i = 0; i < n; i++) expect(l.gW[2 * n + i]).toBe(0);
    expect(l.gb[2]).toBe(0);
  });

  it('counts a dead conv channel too', () => {
    const spec: LayerSpec[] = [{ kind: 'conv', filters: 3, kernel: 3, act: 'relu', pool: false }];
    const ctx = context(spec, 4);
    const b = ctx.net.blocks[0];
    for (let i = 9; i < 18; i++) b.W[i] = -Math.abs(b.W[i]) - 0.01; // filter 2
    b.b[1] = -0.05;
    const { result } = drain(layerStats(ctx, { samples: 20 }));
    expect(result.layers[0].activeFraction[1]).toBe(0);
    expect(result.layers[0].dead).toBe(1);
  });

  it('is deterministic', () => {
    const a = drain(layerStats(context(SMALL_CNN, 9), { samples: 16, maxValues: 2000 })).result;
    const b = drain(layerStats(context(SMALL_CNN, 9), { samples: 16, maxValues: 2000 })).result;
    for (let i = 0; i < a.layers.length; i++) {
      expect(a.layers[i].z).toEqual(b.layers[i].z);
      expect(a.layers[i].a).toEqual(b.layers[i].a);
    }
  });

  it('runs the default settings on the largest preset in a few seconds, at a few ms per yield', () => {
    const lenet: LayerSpec[] = [
      { kind: 'conv', filters: 6, kernel: 5, act: 'tanh', pool: true },
      { kind: 'conv', filters: 16, kernel: 5, act: 'tanh', pool: true },
      { kind: 'dense', units: 64, act: 'tanh' },
    ];
    const timings: string[] = [];
    for (const [name, spec] of [['Small CNN', SMALL_CNN], ['LeNet-ish', lenet], ['MLP', [{ kind: 'dense', units: 64, act: 'relu' }]]] as const) {
      const gen = layerStats(context(spec as LayerSpec[], 1), {});
      let worst = 0;
      const t0 = performance.now();
      let r: IteratorResult<Progress, LayerStatsResult>;
      for (;;) {
        const s = performance.now();
        r = gen.next();
        worst = Math.max(worst, performance.now() - s);
        if (r.done) break;
      }
      const total = performance.now() - t0;
      timings.push(`${name}: ${total.toFixed(0)} ms total, worst yield ${worst.toFixed(1)} ms`);
      expect(r.value.samples).toBe(256);
      for (const l of r.value.layers) expect(l.z.length).toBeLessThanOrEqual(20000);
      expect(total).toBeLessThan(10_000);
    }
    console.log(`layerStats timings (256 images, 20,000 values):\n  ${timings.join('\n  ')}`);
  });
});
