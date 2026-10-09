import { readFileSync } from 'node:fs';
import { inflateSync } from 'node:zlib';
import { describe as suite, expect, it } from 'vitest';
import { mnistArch, type Arch } from '../src/nn/types';
import { defaultPointsConfig, pointsData } from '../src/data/datasets';
import { featurize } from '../src/data/features';
import { gridCoords, PointEvaluator } from '../src/data/grid';
import type { JobContext, Progress } from '../src/analysis/protocol';
import { centreFieldSize, cropBox, receptiveBox, receptiveSize, wholeImage, type Box } from '../src/analysis/receptive';
import {
  actmax,
  allOff,
  coveragePhrase,
  forwardTo,
  HIST_BINS,
  isDead,
  jobs,
  labelSummary,
  mapScale,
  rankIn,
  rankPhrase,
  sharePct,
  synthBase,
  topk,
  unitColumn,
  unitResponse,
  type ActmaxPartial,
  type ActmaxResult,
  type TopkResult,
} from '../src/analysis/units';
import { Network } from '../src/nn/network';
import { Optimizer } from '../src/nn/optim';
import { Rng } from '../src/nn/rng';
import type { LayerSpec } from '../src/nn/types';
import { PRESETS } from '../src/store';

const preset = (name: string) => structuredClone(PRESETS.find((p) => p.name === name)!.spec);
const SMALL_CNN = preset('Small CNN');
const LENET = preset('LeNet-ish');

/** The bundled 2,000 MNIST test digits, decoded from the PNG sprite (8-bit grey, filter 0). */
function loadTest(): { testX: Uint8Array; testY: Uint8Array } {
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
  const testX = new Uint8Array(2000 * 784);
  for (let i = 0; i < 2000; i++) {
    const ox = (i % 100) * 28;
    const oy = Math.floor(i / 100) * 28;
    for (let r = 0; r < 28; r++) for (let c = 0; c < 28; c++) testX[i * 784 + r * 28 + c] = raw[(oy + r) * (width + 1) + 1 + ox + c];
  }
  expect(height).toBe(20 * 28);
  const labels = readFileSync('public/data/mnist-labels.txt', 'utf8').trim();
  const testY = Uint8Array.from(labels.slice(20_000), (ch) => ch.charCodeAt(0) - 48);
  return { testX, testY };
}

const data = loadTest();

function context(net: Network, d = data): JobContext {
  return {
    net,
    arch: net.arch,
    spec: net.spec,
    inputSize: 784,
    scale: 1 / 255,
    classes: 10,
    testX: d.testX,
    testY: d.testY,
    image(i, out = new Float32Array(784)) {
      for (let j = 0; j < 784; j++) out[j] = d.testX[i * 784 + j] / 255;
      return out;
    },
  };
}

/** Runs a job generator to completion, as the analyzer would. */
function drain<R>(gen: Generator<Progress, R, void>): { result: R; reports: Progress[]; ms: number } {
  const reports: Progress[] = [];
  const t0 = performance.now();
  for (;;) {
    const r = gen.next();
    if (r.done) return { result: r.value, reports, ms: performance.now() - t0 };
    reports.push(r.value);
  }
}

const box = (y0: number, y1: number, x0: number, x1: number): Box => ({ y0, y1, x0, x1 });

suite('receptiveBox', () => {
  it('a first 3×3 conv sees the 3×3 neighbourhood', () => {
    const spec: LayerSpec[] = [{ kind: 'conv', filters: 4, kernel: 3, act: 'relu', pool: false }];
    expect(receptiveBox(mnistArch(spec), 0, 5, 5, 'z')).toEqual(box(4, 6, 4, 6));
    expect(receptiveBox(mnistArch(spec), 0, 5, 9, 'z')).toEqual(box(4, 6, 8, 10));
    expect(receptiveSize(mnistArch(spec), 0)).toBe(3);
  });

  it('Small CNN: conv 2 at (5, 5) of its pre-pool map sees rows and columns 7–14', () => {
    expect(receptiveBox(mnistArch(SMALL_CNN), 1, 5, 5, 'z')).toEqual(box(7, 14, 7, 14));
    expect(receptiveSize(mnistArch(SMALL_CNN), 1)).toBe(8);
    expect(receptiveSize(mnistArch(SMALL_CNN), 0)).toBe(3);
    // Pooled output of conv 1 at (5, 5) covers z rows 10–11, so pixels 9–12.
    expect(receptiveBox(mnistArch(SMALL_CNN), 0, 5, 5, 'out')).toEqual(box(9, 12, 9, 12));
    // Pooled output of conv 2 at (3, 3): z rows 6–7 → conv-1 out 5–8 → z 10–17 → pixels 9–18.
    expect(receptiveBox(mnistArch(SMALL_CNN), 1, 3, 3, 'out')).toEqual(box(9, 18, 9, 18));
    expect(receptiveSize(mnistArch(SMALL_CNN), 1, 'out')).toBe(10);
  });

  it('LeNet-ish 5×5 kernels give a 14-pixel field for conv 2', () => {
    expect(receptiveSize(mnistArch(LENET), 0)).toBe(5);
    expect(receptiveSize(mnistArch(LENET), 1)).toBe(14);
    expect(receptiveBox(mnistArch(LENET), 1, 7, 7, 'z')).toEqual(box(8, 21, 8, 21));
  });

  it('clips at the corners', () => {
    expect(receptiveBox(mnistArch(SMALL_CNN), 0, 0, 0, 'z')).toEqual(box(0, 1, 0, 1));
    expect(receptiveBox(mnistArch(SMALL_CNN), 0, 27, 27, 'z')).toEqual(box(26, 27, 26, 27));
    expect(receptiveBox(mnistArch(SMALL_CNN), 1, 0, 0, 'z')).toEqual(box(0, 4, 0, 4));
    expect(receptiveBox(mnistArch(SMALL_CNN), 1, 13, 13, 'z')).toEqual(box(23, 27, 23, 27));
    expect(receptiveBox(mnistArch(SMALL_CNN), 1, 0, 13, 'z')).toEqual(box(0, 4, 23, 27));
    // Unclipped, the nominal field keeps its full size.
    expect(receptiveBox(mnistArch(SMALL_CNN), 1, 0, 0, 'z', false)).toEqual(box(-3, 4, -3, 4));
  });

  it('works through three pools (28 → 14 → 7 → 3)', () => {
    const spec: LayerSpec[] = [
      { kind: 'conv', filters: 2, kernel: 3, act: 'linear', pool: true },
      { kind: 'conv', filters: 2, kernel: 3, act: 'linear', pool: true },
      { kind: 'conv', filters: 2, kernel: 3, act: 'linear', pool: true },
    ];
    // out (0, 0) → z 0–1 → 0–2 → pool 0–5 → 0–6 → pool 0–13 → 0–14
    expect(receptiveBox(mnistArch(spec), 2, 0, 0, 'out')).toEqual(box(0, 14, 0, 14));
    // out (2, 2) → z 4–5 → 3–6 → pool 6–13 → 5–14 → pool 10–29 → 9–30, clipped to 27
    expect(receptiveBox(mnistArch(spec), 2, 2, 2, 'out')).toEqual(box(9, 27, 9, 27));
    expect(receptiveSize(mnistArch(spec), 2, 'out')).toBe(22);
  });

  it('dense and output blocks see the whole image', () => {
    expect(receptiveBox(mnistArch(SMALL_CNN), 2, 0, 0)).toBeNull();
    expect(receptiveBox(mnistArch(SMALL_CNN), 3, 0, 0)).toBeNull();
    expect(receptiveSize(mnistArch(SMALL_CNN), 2)).toBe(28);
    expect(receptiveSize(mnistArch(SMALL_CNN), 3)).toBe(28);
    expect(centreFieldSize(mnistArch(SMALL_CNN), 2)).toBe(28);
    expect(cropBox(mnistArch(SMALL_CNN), 2, 0, 0)).toBeNull();
  });

  it('deep stacks: the field shown and cropped never exceeds the 28×28 image (U6)', () => {
    const conv = (kernel: 3 | 5, pool: boolean): LayerSpec => ({ kind: 'conv', filters: 2, kernel, act: 'relu', pool });
    const deepA = [conv(3, true), conv(5, true), conv(3, true), conv(5, false)];
    const deepB = [conv(3, true), conv(3, true), conv(3, true), conv(3, true)];
    // The nominal field outgrows the image: 58 and 38 pixels.
    expect(receptiveSize(mnistArch(deepA), 3)).toBe(58);
    expect(receptiveSize(mnistArch(deepB), 3)).toBe(38);
    for (const spec of [deepA, deepB]) {
      // What the centre unit really sees is the whole image, and crops are the image itself.
      expect(centreFieldSize(mnistArch(spec), 3)).toBe(28);
      const map = 3; // 28 → 14 → 7 → 3
      for (let y = 0; y < map; y++) for (let x = 0; x < map; x++) expect(cropBox(mnistArch(spec), 3, y, x)).toEqual(wholeImage(mnistArch(spec)));
    }
    // Shallower layers keep their nominal, same-size crops (blank past the edge).
    expect(cropBox(mnistArch(SMALL_CNN), 1, 0, 0)).toEqual(box(-3, 4, -3, 4));
    expect(cropBox(mnistArch(SMALL_CNN), 1, 5, 5)).toEqual(box(7, 14, 7, 14));
    expect(centreFieldSize(mnistArch(SMALL_CNN), 1)).toBe(8);
    expect(centreFieldSize(mnistArch(LENET), 1)).toBe(14);
    // Every crop of every buildable layer is at most 28 pixels wide.
    for (const spec of [SMALL_CNN, LENET, deepA, deepB]) {
      spec.forEach((l, block) => {
        if (l.kind !== 'conv') return;
        const crop = cropBox(mnistArch(spec), block, 0, 0)!;
        expect(crop.y1 - crop.y0 + 1).toBeLessThanOrEqual(28);
        expect(centreFieldSize(mnistArch(spec), block)).toBeLessThanOrEqual(28);
      });
    }
  });

  /**
   * Perturbs each input pixel by ±δ and records which positions of block `block` (level z, or
   * the pooled output) change. Returns the bounding box of the pixels that moved each position.
   */
  function bruteForce(spec: LayerSpec[], block: number, level: 'z' | 'out', positions: [number, number][], deltas: number[], seed: number) {
    const net = new Network(mnistArch(spec), seed);
    const rng = new Rng(seed + 50);
    const x = new Float32Array(784);
    for (let i = 0; i < 784; i++) x[i] = rng.next();
    const b = net.blocks[block];
    if (b.kind !== 'conv') throw new Error('conv only');
    const read = () => (level === 'z' ? b.z : b.out).slice();
    const W = level === 'z' ? b.zShape.w : b.outShape.w;
    forwardTo(net, x, block);
    const base = read();
    const seen = positions.map(() => ({ y0: 99, y1: -1, x0: 99, x1: -1, any: false }));
    for (let p = 0; p < 784; p++) {
      const orig = x[p];
      for (const d of deltas) {
        x[p] = orig + d;
        forwardTo(net, x, block);
        const now = level === 'z' ? b.z : b.out;
        positions.forEach(([py, px], k) => {
          let moved = false;
          for (let f = 0; f < b.spec.filters && !moved; f++) {
            const i = f * (base.length / b.spec.filters) + py * W + px;
            if (now[i] !== base[i]) moved = true;
          }
          if (!moved) return;
          const s = seen[k];
          const r = Math.floor(p / 28);
          const c = p % 28;
          s.any = true;
          s.y0 = Math.min(s.y0, r);
          s.y1 = Math.max(s.y1, r);
          s.x0 = Math.min(s.x0, c);
          s.x1 = Math.max(s.x1, c);
        });
      }
      x[p] = orig;
    }
    return seen.map((s) => (s.any ? box(s.y0, s.y1, s.x0, s.x1) : null));
  }

  it('agrees with brute-force pixel perturbation on a random network without pooling', () => {
    const spec: LayerSpec[] = [
      { kind: 'conv', filters: 3, kernel: 3, act: 'tanh', pool: false },
      { kind: 'conv', filters: 3, kernel: 5, act: 'leaky', pool: false },
      { kind: 'conv', filters: 2, kernel: 3, act: 'linear', pool: false },
    ];
    const positions: [number, number][] = [[14, 14], [0, 0], [27, 27], [3, 20], [26, 1], [9, 4]];
    const found = bruteForce(spec, 2, 'z', positions, [0.5], 3);
    positions.forEach(([y, x], k) => expect(found[k], `position ${y},${x}`).toEqual(receptiveBox(mnistArch(spec), 2, y, x, 'z')));
    expect(found[0]).toEqual(box(10, 18, 10, 18)); // 3 + 5 + 3 kernels: 9 pixels wide
  });

  it('agrees with brute-force pixel perturbation through a max-pool', () => {
    // Linear convs, so the pool is the only non-linearity; ±δ makes the perturbed position win its
    // pool window for one of the two signs.
    const spec: LayerSpec[] = [
      { kind: 'conv', filters: 2, kernel: 3, act: 'linear', pool: true },
      { kind: 'conv', filters: 2, kernel: 5, act: 'linear', pool: false },
    ];
    const positions: [number, number][] = [[7, 7], [0, 0], [13, 13], [2, 11], [12, 5]];
    const found = bruteForce(spec, 1, 'z', positions, [25, -25], 5);
    positions.forEach(([y, x], k) => expect(found[k], `position ${y},${x}`).toEqual(receptiveBox(mnistArch(spec), 1, y, x, 'z')));
    // The pooled output of block 0 itself, at a few positions.
    const outPos: [number, number][] = [[5, 5], [0, 13], [13, 0]];
    const foundOut = bruteForce(spec, 0, 'out', outPos, [25, -25], 7);
    outPos.forEach(([y, x], k) => expect(foundOut[k], `out ${y},${x}`).toEqual(receptiveBox(mnistArch(spec), 0, y, x, 'out')));
  });

  it('never misses a pixel in a deeper pooled network (affected pixels lie inside the box)', () => {
    const spec: LayerSpec[] = [
      { kind: 'conv', filters: 2, kernel: 3, act: 'linear', pool: true },
      { kind: 'conv', filters: 2, kernel: 3, act: 'linear', pool: true },
      { kind: 'conv', filters: 2, kernel: 3, act: 'linear', pool: false },
    ];
    const positions: [number, number][] = [[3, 3], [0, 6], [6, 6]];
    const found = bruteForce(spec, 2, 'z', positions, [25, -25], 11);
    positions.forEach(([y, x], k) => {
      const want = receptiveBox(mnistArch(spec), 2, y, x, 'z')!;
      const got = found[k]!;
      expect(got).not.toBeNull();
      expect(got.y0).toBeGreaterThanOrEqual(want.y0);
      expect(got.y1).toBeLessThanOrEqual(want.y1);
      expect(got.x0).toBeGreaterThanOrEqual(want.x0);
      expect(got.x1).toBeLessThanOrEqual(want.x1);
    });
  });
});

/**
 * Every unit's response to every test digit, computed the slow, obvious way, plus how often each
 * unit fires: conv → mean share of positions with z > 0; dense → share of digits with z > 0;
 * output → share of digits predicted as this digit.
 */
function bruteResponses(net: Network, block: number, n: number) {
  const b = net.blocks[block];
  const out = block === net.blocks.length - 1;
  const U = b.kind === 'conv' ? b.spec.filters : b.spec.units;
  const r: number[][] = Array.from({ length: U }, () => []);
  const coverage = new Array<number>(U).fill(0);
  const x = new Float32Array(784);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < 784; j++) x[j] = data.testX[i * 784 + j] / 255;
    const pred = net.predict(x);
    for (let u = 0; u < U; u++) {
      if (b.kind === 'conv') {
        const HW = b.zShape.h * b.zShape.w;
        let m = -Infinity;
        let on = 0;
        for (let q = 0; q < HW; q++) {
          m = Math.max(m, b.a[u * HW + q]);
          if (b.z[u * HW + q] > 0) on++;
        }
        r[u].push(m);
        coverage[u] += on / HW / n;
      } else {
        r[u].push(out ? b.z[u] : b.a[u]);
        coverage[u] += (out ? pred === u : b.z[u] > 0) ? 1 / n : 0;
      }
    }
  }
  return Object.assign(r, { coverage });
}

suite('topk job', () => {
  it('is registered with actmax', () => {
    expect(Object.keys(jobs).sort()).toEqual(['actmax', 'topk']);
  });

  for (const block of [0, 1, 2, 3]) {
    it(`matches a brute-force scan on Small CNN block ${block}`, () => {
      const net = new Network(mnistArch(SMALL_CNN), 4);
      const n = 300;
      const { result, reports } = drain(topk(context(net), { block, k: 9, count: n }));
      const brute = bruteResponses(new Network(mnistArch(SMALL_CNN), 4), block, n);
      expect(result.kind).toBe(['conv', 'conv', 'dense', 'output'][block]);
      expect(result.count).toBe(n);
      expect(result.units).toHaveLength(brute.length);
      expect(reports.length).toBeGreaterThan(1);
      expect(reports.at(-1)!.done).toBe(reports.at(-1)!.total);
      for (const u of result.units) {
        const r = brute[u.unit];
        const sorted = [...r].sort((a, b) => b - a);
        expect(u.top).toHaveLength(9);
        expect(u.bottom).toHaveLength(9);
        u.top.forEach((hit, i) => {
          expect(hit.value).toBeCloseTo(sorted[i], 5);
          expect(hit.value).toBeCloseTo(r[hit.index], 5);
        });
        u.bottom.forEach((hit, i) => expect(hit.value).toBeCloseTo(sorted[n - 1 - i], 5));
        expect(u.mean).toBeCloseTo(r.reduce((a, b) => a + b, 0) / n, 4);
        expect(u.coverage).toBeCloseTo(brute.coverage[u.unit], 6);
        expect(u.sorted).toBeInstanceOf(Float32Array);
        expect(Array.from(u.sorted)).toEqual([...r].map(Math.fround).sort((a, b) => a - b));
        expect(u.hist.counts).toHaveLength(HIST_BINS);
        expect(u.hist.counts.reduce((a, b) => a + b, 0)).toBe(n);
        expect(u.hist.lo).toBeLessThanOrEqual(sorted[n - 1]);
        expect(u.hist.hi).toBeGreaterThanOrEqual(sorted[0]);
        expect(u.labelCounts.reduce((a, b) => a + b, 0)).toBe(50);
        if (sorted[49] > sorted[50]) {
          // No tie at the cut: the top 50 are exactly the digits at or above the 50th value.
          const want = new Array(10).fill(0);
          r.forEach((v, i) => {
            if (v >= sorted[49]) want[data.testY[i]]++;
          });
          expect(u.labelCounts).toEqual(want);
        }
        const counts = brute[u.unit].map((v) => Math.min(HIST_BINS - 1, Math.max(0, Math.floor(((v - u.hist.lo) / (u.hist.hi - u.hist.lo)) * HIST_BINS))));
        const histWant = new Array(HIST_BINS).fill(0);
        for (const c of counts) histWant[c]++;
        expect(u.hist.counts.reduce((a, c, i) => a + Math.abs(c - histWant[i]), 0)).toBeLessThanOrEqual(2); // float32 vs float64 at bin edges
      }
    });
  }

  it('records where a conv filter fired and the pixels behind that position', () => {
    const net = new Network(mnistArch(SMALL_CNN), 9);
    const { result } = drain(topk(context(net), { block: 1, k: 4, count: 120 }));
    const probe = new Network(mnistArch(SMALL_CNN), 9);
    for (const u of result.units.slice(0, 5)) {
      for (const hit of u.top) {
        expect(hit.box).toEqual(receptiveBox(mnistArch(SMALL_CNN), 1, hit.y, hit.x, 'z'));
        const x = context(probe).image(hit.index);
        const r = unitResponse(probe, 1, u.unit, x);
        expect(hit.z).toBeCloseTo(probe.blocks[1].z[u.unit * 196 + hit.y * 14 + hit.x], 6);
        expect(r.value).toBeCloseTo(hit.value, 6);
        expect([r.y, r.x]).toEqual([hit.y, hit.x]);
        // The value at that position is the filter's activation there.
        const b = probe.blocks[1];
        expect(b.a[u.unit * 196 + hit.y * 14 + hit.x]).toBeCloseTo(hit.value, 6);
      }
    }
  });

  it('dense and output hits carry no box', () => {
    const net = new Network(mnistArch(SMALL_CNN), 2);
    const { result } = drain(topk(context(net), { block: 3, k: 3, count: 50 }));
    expect(result.units).toHaveLength(10);
    expect(result.units[0].top[0]).toMatchObject({ box: null, y: -1, x: -1 });
  });

  it('labels of the top 50 follow the digit a trained output unit stands for', () => {
    // Train a softmax (no hidden layers) for two quick SGD passes over the first 1,000 digits.
    const net = new Network(mnistArch([]), 1);
    const x = new Float32Array(784);
    const out = net.output;
    for (let epoch = 0; epoch < 2; epoch++) {
      for (let i = 0; i < 1000; i++) {
        for (let j = 0; j < 784; j++) x[j] = data.testX[i * 784 + j] / 255;
        net.zeroGrad();
        net.forward(x);
        net.backward(data.testY[i]);
        for (let w = 0; w < out.W.length; w++) out.W[w] -= 0.05 * out.gW[w];
        for (let w = 0; w < out.b.length; w++) out.b[w] -= 0.05 * out.gb[w];
      }
    }
    const { result } = drain(topk(context(net), { block: 0, k: 16 }));
    expect(result.count).toBe(2000);
    for (const u of result.units) {
      const best = u.labelCounts.indexOf(Math.max(...u.labelCounts));
      expect(best, `digit ${u.unit} top-50 labels ${u.labelCounts}`).toBe(u.unit);
      expect(u.labelCounts[u.unit]).toBeGreaterThan(40);
    }
  });

  it('scans all 2,000 digits for every Small CNN layer within the time budget', () => {
    const net = new Network(mnistArch(SMALL_CNN), 1);
    const times: Record<string, number> = {};
    for (const block of [0, 1, 2, 3]) {
      const { result, ms } = drain(topk(context(net), { block, k: 16 }));
      expect(result.count).toBe(2000);
      times[`block ${block}`] = Math.round(ms);
    }
    console.log('topk, Small CNN, 2,000 digits (ms):', times);
    expect(Math.max(...Object.values(times))).toBeLessThan(10_000);
  }, 120_000);
});

suite('actmax job', () => {
  it('inputGradient with a one-hot seed at the centre is the gradient of that z (finite differences)', () => {
    const net = new Network(mnistArch(SMALL_CNN), 3);
    const rng = new Rng(8);
    const x = new Float32Array(784);
    for (let i = 0; i < 784; i++) x[i] = 0.3 * rng.next();
    const b = net.blocks[1];
    const zi = 5 * 196 + 7 * 14 + 7;
    const seed = new Float32Array(b.z.length);
    seed[zi] = 1;
    forwardTo(net, x, 1);
    const g = net.inputGradient(1, seed).slice();
    const eps = 1e-2;
    let num = 0;
    let den = 0;
    for (const p of [13 * 28 + 13, 14 * 28 + 15, 12 * 28 + 16, 17 * 28 + 11, 10 * 28 + 14]) {
      const o = x[p];
      x[p] = o + eps;
      forwardTo(net, x, 1);
      const zp = b.z[zi];
      x[p] = o - eps;
      forwardTo(net, x, 1);
      const zm = b.z[zi];
      x[p] = o;
      const fd = (zp - zm) / (2 * eps);
      num += (fd - g[p]) ** 2;
      den += fd ** 2 + g[p] ** 2;
    }
    expect(den).toBeGreaterThan(0);
    expect(Math.sqrt(num / den)).toBeLessThan(0.05);
    // Pixels outside the receptive field get no gradient.
    expect(g[0]).toBe(0);
    expect(g[27 * 28 + 27]).toBe(0);
  });

  for (const [name, spec, block] of [
    ['Small CNN conv 1', SMALL_CNN, 0],
    ['Small CNN conv 2', SMALL_CNN, 1],
    ['Small CNN dense', SMALL_CNN, 2],
    ['Small CNN output', SMALL_CNN, 3],
    ['MLP hidden', preset('MLP'), 0],
    ['LeNet-ish conv 2', LENET, 1],
  ] as const) {
    it(`raises the objective above its blank start: ${name}`, () => {
      const net = new Network(mnistArch(spec as LayerSpec[]), 6);
      const units = [0, 1, 2];
      const { result, reports } = drain(actmax(context(net), { block, units, steps: 160 }));
      expect(result.units.map((u) => u.unit)).toEqual(units);
      for (const u of result.units) {
        expect(u.final, `unit ${u.unit}: ${u.start} → ${u.final}`).toBeGreaterThan(u.start);
        expect(Math.min(...u.x)).toBeGreaterThanOrEqual(0);
        expect(Math.max(...u.x)).toBeLessThanOrEqual(1);
        if (u.box) {
          for (let p = 0; p < 784; p++) {
            const r = Math.floor(p / 28);
            const c = p % 28;
            const inside = r >= u.box.y0 && r <= u.box.y1 && c >= u.box.x0 && c <= u.box.x1;
            if (!inside) expect(u.x[p]).toBe(0);
          }
        }
        // `final` is the objective of the returned image.
        forwardTo(net, u.x, block);
        const b = net.blocks[block];
        const zi = b.kind === 'conv' ? u.unit * b.zShape.h * b.zShape.w + (b.zShape.h >> 1) * b.zShape.w + (b.zShape.w >> 1) : u.unit;
        expect(b.z[zi]).toBeCloseTo(u.final, 5);
      }
      expect(reports).toHaveLength(units.length * 160);
      const partials = reports.map((r) => r.partial as ActmaxPartial);
      expect(partials.every((p) => p && p.x.length === 784 && units.includes(p.unit))).toBe(true);
      expect(new Set(partials.map((p) => p.step)).size).toBeGreaterThan(10);
    });
  }

  it('is deterministic and seeds each unit differently', () => {
    const net = new Network(mnistArch(SMALL_CNN), 2);
    const a = drain(actmax(context(net), { block: 2, units: [4, 5], steps: 20 })).result;
    const b = drain(actmax(context(new Network(mnistArch(SMALL_CNN), 2)), { block: 2, units: [4], steps: 20 })).result;
    expect(Array.from(b.units[0].x)).toEqual(Array.from(a.units[0].x));
    expect(Array.from(a.units[1].x)).not.toEqual(Array.from(a.units[0].x));
  });

  it('conv 1 synthesises the pattern its kernel weights describe', () => {
    // With a single linear 3×3 conv, z = b + Σ w·x over the patch: the optimum puts ink (1) where
    // the weight is positive and leaves 0 where it is negative.
    const spec: LayerSpec[] = [{ kind: 'conv', filters: 4, kernel: 3, act: 'linear', pool: false }];
    const net = new Network(mnistArch(spec), 12);
    const { result } = drain(actmax(context(net), { block: 0, steps: 160 }));
    const W = net.blocks[0].W;
    for (const u of result.units) {
      expect(u.box).toEqual(box(13, 15, 13, 15));
      let ss = 0;
      for (let i = 0; i < 9; i++) ss += W[u.unit * 9 + i] ** 2;
      const rms = Math.sqrt(ss / 9);
      for (let ky = 0; ky < 3; ky++) {
        for (let kx = 0; kx < 3; kx++) {
          const w = W[u.unit * 9 + ky * 3 + kx];
          const v = u.x[(13 + ky) * 28 + 13 + kx];
          if (w > 0.5 * rms) expect(v, `filter ${u.unit} w=${w.toFixed(3)}`).toBeGreaterThan(0.6);
          if (w < -0.5 * rms) expect(v, `filter ${u.unit} w=${w.toFixed(3)}`).toBeLessThan(0.4);
        }
      }
    }
  });

  it('synthesises every unit of each Small CNN layer within the time budget', () => {
    const times: Record<string, number> = {};
    for (const block of [0, 1, 2, 3]) {
      const net = new Network(mnistArch(SMALL_CNN), 1);
      const { result, ms } = drain(actmax(context(net), { block }));
      expect(result.steps).toBe(160);
      times[`block ${block} (${result.units.length} units)`] = Math.round(ms);
    }
    console.log('actmax, Small CNN, 160 steps per unit (ms):', times);
    expect(Math.max(...Object.values(times))).toBeLessThan(10_000);
  }, 120_000);
});

// Result types are what the page receives; keep them structured-cloneable.
it('results survive structured cloning', () => {
  const net = new Network(mnistArch(SMALL_CNN), 1);
  const t = drain(topk(context(net), { block: 1, k: 2, count: 20 })).result;
  const a = drain(actmax(context(net), { block: 1, units: [0], steps: 4 })).result;
  expect(structuredClone(t) as TopkResult).toEqual(t);
  expect((structuredClone(a) as ActmaxResult).units[0].x).toBeInstanceOf(Float32Array);
});

/** A few Adam steps on the first test digits: enough to move every weight. */
function train(net: Network, steps: number, from = 0) {
  const opt = new Optimizer(net, 'adam', 0.003);
  const x = new Float32Array(784);
  for (let s = 0; s < steps; s++) {
    net.zeroGrad();
    for (let b = 0; b < 16; b++) {
      const i = (from + s * 16 + b) % 2000;
      for (let j = 0; j < 784; j++) x[j] = data.testX[i * 784 + j] / 255;
      net.forward(x);
      net.backward(data.testY[i]);
    }
    opt.step(1 / 16);
  }
}

suite('ranks against the scan (U1, U2)', () => {
  it('rankIn counts lower, equal and higher responses exactly', () => {
    const sorted = Float32Array.from([0, 0, 0, 0, 0.5, 1, 1, 2]);
    expect(rankIn(sorted, 0)).toEqual({ below: 0, tied: 4, above: 4, n: 8 });
    expect(rankIn(sorted, 0.25)).toEqual({ below: 4, tied: 0, above: 4, n: 8 });
    expect(rankIn(sorted, 1)).toEqual({ below: 5, tied: 2, above: 1, n: 8 });
    expect(rankIn(sorted, 3)).toEqual({ below: 8, tied: 0, above: 0, n: 8 });
    expect(rankIn(sorted, -1)).toEqual({ below: 0, tied: 0, above: 8, n: 8 });
    // Values are compared as float32, the way the scan stores them.
    const f = Float32Array.from([0.1, 0.2, 0.3]);
    expect(rankIn(f, 0.2)).toEqual({ below: 1, tied: 1, above: 1, n: 3 });
    expect(rankIn(f, f[1])).toEqual({ below: 1, tied: 1, above: 1, n: 3 });
  });

  it('rankPhrase is exact and calls out ties instead of interpolating', () => {
    const r = (below: number, tied: number, above: number) => rankPhrase({ below, tied, above, n: below + tied + above });
    // 1,266 of 2,000 digits are exactly 0 (a ReLU unit) and the input is 0 too.
    expect(r(0, 1266, 734)).toBe('tied with 63.3% of the 2,000 test digits at the lowest response');
    // A small positive response sits above all those zeros (the old text said "about 2%").
    expect(r(1474, 1, 525)).toBe('higher than 73.7% of the 2,000 test digits');
    expect(r(1474, 0, 526)).toBe('higher than 73.7% of the 2,000 test digits');
    expect(r(100, 300, 1600)).toBe('higher than 5.0% of the 2,000 test digits and tied with another 15.0%');
    expect(r(1200, 800, 0)).toBe('tied with 40.0% of the 2,000 test digits at the highest response');
    // A dead unit: every digit gives the same value.
    expect(r(0, 2000, 0)).toBe('the same as all 2,000 test digits');
    // The extremes, and digits near them, by count rather than a rounded share.
    expect(r(1999, 1, 0)).toBe('as high as the strongest of the 2,000 test digits');
    expect(r(2000, 0, 0)).toBe('higher than all 2,000 test digits');
    expect(r(0, 1, 1999)).toBe('as low as the weakest of the 2,000 test digits');
    expect(r(0, 0, 2000)).toBe('lower than all 2,000 test digits');
    expect(r(1996, 1, 3)).toBe('only 3 of the 2,000 test digits respond more strongly');
    expect(r(1998, 1, 1)).toBe('only 1 of the 2,000 test digits responds more strongly');
    expect(r(2, 1, 1997)).toBe('only 2 of the 2,000 test digits respond more weakly');
    expect(r(1989, 1, 10)).toBe('higher than 99.5% of the 2,000 test digits');
  });

  it('sharePct never rounds to 0% or 100% unless exact', () => {
    expect(sharePct(0)).toBe('0%');
    expect(sharePct(1)).toBe('100%');
    expect(sharePct(0.9996)).toBe('>99%');
    expect(sharePct(0.004)).toBe('<1%');
    expect(sharePct(0.274)).toBe('27%');
    expect(sharePct(0.9996, 1)).toBe('>99.9%');
    expect(sharePct(0.1036, 1)).toBe('10.4%');
  });

  it('a dense ReLU unit: ranks from the sorted responses count the exact zeros', () => {
    const net = new Network(mnistArch(SMALL_CNN), 3);
    const { result } = drain(topk(context(net), { block: 2, k: 4 }));
    const copy = new Network(mnistArch(SMALL_CNN), 0);
    copy.setWeights(net.getWeights());
    let checked = 0;
    for (const u of result.units) {
      const zeros = u.sorted.filter((v) => v === 0).length;
      if (zeros < 200 || zeros > 1800) continue;
      // The digit just above the zeros: everything at 0 is below it.
      const above = u.sorted[zeros];
      expect(above).toBeGreaterThan(0);
      expect(rankIn(u.sorted, above).below).toBe(zeros);
      // A digit whose response is 0 ties with every zero.
      const zeroHit = u.bottom[0];
      expect(zeroHit.value).toBe(0);
      const v = unitResponse(copy, 2, u.unit, context(copy).image(zeroHit.index)).value;
      expect(rankIn(u.sorted, v)).toEqual({ below: 0, tied: zeros, above: 2000 - zeros, n: 2000 });
      expect(rankPhrase(rankIn(u.sorted, v))).toMatch(/^tied with \d+\.\d% of the 2,000 test digits at the lowest response$/);
      checked++;
    }
    expect(checked).toBeGreaterThan(3);
  });

  it('the current input measured with a copy of the scan weights ranks consistently, even after training moves on', () => {
    const live = new Network(mnistArch(SMALL_CNN), 1);
    const { result } = drain(topk(context(live), { block: 1, k: 16 }));
    // What the page keeps beside the scan: a network holding the weights the scan used.
    const scanNet = new Network(mnistArch(SMALL_CNN), 0);
    scanNet.setWeights(live.getWeights());
    train(live, 25);
    const x0 = context(live).image(0);
    let liveClaims = 0;
    for (const u of result.units) {
      // The scan's own top digit ranks first, and digit #0 ranks where the scan puts it.
      const best = u.top[0];
      const vb = unitResponse(scanNet, 1, u.unit, context(scanNet).image(best.index)).value;
      expect(vb).toBe(best.value);
      expect(rankIn(u.sorted, vb)).toMatchObject({ above: 0 });
      const v0 = unitResponse(scanNet, 1, u.unit, x0).value;
      const r0 = rankIn(u.sorted, v0);
      expect(r0.tied).toBeGreaterThanOrEqual(1); // digit #0 is one of the scanned digits
      const inTop = u.top.some((h) => h.index === 0);
      if (!inTop) expect(r0.above).toBeGreaterThanOrEqual(16);
      // The live weights, by contrast, put digit #0 off the scan's scale for several filters.
      const vLive = unitResponse(live, 1, u.unit, x0).value;
      if (rankIn(u.sorted, vLive).above === 0 && !inTop) liveClaims++;
    }
    expect(liveClaims, 'the bug this guards against: live weights against a stale scan').toBeGreaterThan(0);
  });
});

suite('coverage (U4)', () => {
  it('conv filters report the share of positions that fire, not "some position fired"', () => {
    const net = new Network(mnistArch(SMALL_CNN), 1);
    train(net, 40);
    const { result } = drain(topk(context(net), { block: 1, k: 2, count: 300 }));
    const brute = bruteResponses(net, 1, 300);
    for (const u of result.units) {
      expect(u.coverage).toBeCloseTo(brute.coverage[u.unit], 6);
      // The old "active" share (strongest response above 0) is 1 for almost every filter.
      expect(u.coverage).toBeLessThan(1);
    }
    expect(new Set(result.units.map((u) => u.coverage.toFixed(3))).size).toBeGreaterThan(8);
    expect(coveragePhrase('conv', 0.2741)).toBe('fires at 27% of positions');
  });

  it('sigmoid units are not "active" on every digit: firing means z > 0 (a > 0.5)', () => {
    const spec: LayerSpec[] = [{ kind: 'dense', units: 16, act: 'sigmoid' }];
    const net = new Network(mnistArch(spec), 2);
    const { result } = drain(topk(context(net), { block: 0, k: 2, count: 400 }));
    const brute = bruteResponses(net, 0, 400);
    for (const u of result.units) {
      expect(u.coverage).toBeCloseTo(brute.coverage[u.unit], 6);
      expect(Math.min(...u.sorted)).toBeGreaterThan(0); // every activation is above 0…
    }
    expect(result.units.some((u) => u.coverage < 0.9)).toBe(true); // …but they do not all fire
    expect(coveragePhrase('dense', 0.37)).toBe('fires on 37% of digits');
  });

  it('output units report how often each digit is predicted; the shares sum to 1', () => {
    const net = new Network(mnistArch(SMALL_CNN), 5);
    const { result } = drain(topk(context(net), { block: 3, k: 2, count: 500 }));
    const brute = bruteResponses(net, 3, 500);
    result.units.forEach((u) => expect(u.coverage).toBeCloseTo(brute.coverage[u.unit], 9));
    expect(result.units.reduce((a, u) => a + u.coverage, 0)).toBeCloseTo(1, 9);
    expect(coveragePhrase('output', 0.104, 1)).toBe('predicted for 10.4% of digits');
  });
});

suite('weakest digits heading (U7)', () => {
  it('conv filters still fire on their weakest digits; dense ReLU units can be switched off', () => {
    const net = new Network(mnistArch(SMALL_CNN), 1);
    const conv = drain(topk(context(net), { block: 1, k: 8, count: 500 })).result;
    for (const u of conv.units) {
      expect(u.bottom[0].value).toBeGreaterThan(0);
      expect(allOff(u.bottom)).toBe(false);
    }
    const dense = drain(topk(context(net), { block: 2, k: 8, count: 500 })).result;
    const off = dense.units.filter((u) => allOff(u.bottom));
    expect(off.length).toBeGreaterThan(0);
    for (const u of off) for (const hit of u.bottom) expect(hit.value).toBe(0); // ReLU: z ≤ 0 means a = 0
    expect(allOff([])).toBe(false);
  });
});

suite('labelSummary', () => {
  const counts = (m: Record<number, number>) => Array.from({ length: 10 }, (_, d) => m[d] ?? 0);
  it('names the leading labels in plain words', () => {
    expect(labelSummary(counts({ 0: 38, 6: 7, 9: 3, 1: 2 }))).toBe('Mostly 0s (38), 6s (7), 9s (3) and 1 other digit.');
    expect(labelSummary(counts({ 0: 40, 6: 10 }))).toBe('Mostly 0s (40) and 6s (10).');
    expect(labelSummary(counts({ 7: 50 }))).toBe('All 50 are 7s.');
    expect(labelSummary(counts({ 3: 12, 5: 11, 8: 9, 1: 8, 2: 6, 0: 4 }))).toBe('Mixed: 3s (12), 5s (11), 8s (9) and 3 other digits.');
    expect(labelSummary(counts({}))).toBe('No digits.');
  });
});

suite('labelSummary: ties and class names (NEW-6)', () => {
  const counts = (m: Record<number, number>, k = 10) => Array.from({ length: k }, (_, d) => m[d] ?? 0);
  it('names every label tied with the third, with one shared count', () => {
    // The review case: the bars show 0, 3, 4 and 9 at 7 each; the old text named only 0, 3 and 4.
    expect(labelSummary(counts({ 0: 7, 1: 1, 2: 5, 3: 7, 4: 7, 5: 2, 6: 6, 7: 4, 8: 4, 9: 7 }))).toBe('Mixed: 0s, 3s, 4s and 9s (7 each) and 6 other digits.');
    expect(labelSummary(counts({ 3: 12, 5: 7, 8: 7, 1: 7, 2: 6, 0: 4 }))).toBe('Mixed: 3s (12), 1s, 5s and 8s (7 each) and 2 other digits.');
    // A tie for second that ends the list: one "and" inside the group, none before it.
    expect(labelSummary(counts({ 1: 30, 0: 10, 3: 10 }))).toBe('Mostly 1s (30), 0s and 3s (10 each).');
    expect(labelSummary(counts({ 0: 25, 1: 25 }))).toBe('Mostly 0s and 1s (25 each).');
    // No tie: unchanged.
    expect(labelSummary(counts({ 3: 12, 5: 11, 8: 9, 1: 8 }))).toBe('Mixed: 3s (12), 5s (11), 8s (9) and 1 other digit.');
  });

  it('uses class names as they are, for clothes, objects and point classes', () => {
    const cifar = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'];
    expect(labelSummary(counts({ 0: 18, 2: 8, 1: 5, 3: 3, 4: 2, 5: 3, 6: 1, 7: 2, 8: 4, 9: 4 }), cifar)).toBe('Mixed: airplane (18), bird (8), automobile (5) and 7 other classes.');
    expect(labelSummary(counts({ 3: 30, 5: 20 }), cifar)).toBe('Mostly cat (30) and dog (20).');
    const fashion = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'];
    expect(labelSummary(counts({ 9: 40, 7: 6, 5: 4 }), fashion)).toBe('Mostly ankle boot (40), sneaker (6) and sandal (4).');
    expect(labelSummary(counts({ 0: 20, 6: 20, 2: 10 }), fashion)).toBe('Mixed: t-shirt/top and shirt (20 each) and pullover (10).');
    const points = ['Class 0', 'Class 1'];
    expect(labelSummary(counts({ 1: 50 }, 2), points)).toBe('All 50 are labelled class 1.');
    expect(labelSummary(counts({ 1: 33, 0: 17 }, 2), points)).toBe('Mostly class 1 (33) and class 0 (17).');
    expect(labelSummary(counts({}, 2), points)).toBe('No labels.');
  });
});

suite('wording for every dataset', () => {
  it('ranks and coverage name the samples: digits, images or points', () => {
    const r = (below: number, tied: number, above: number) => ({ below, tied, above, n: below + tied + above });
    expect(rankPhrase(r(1474, 1, 525), 'image')).toBe('higher than 73.7% of the 2,000 test images');
    expect(rankPhrase(r(0, 300, 0), 'point')).toBe('the same as all 300 test points');
    expect(rankPhrase(r(298, 1, 1), 'point')).toBe('only 1 of the 300 test points responds more strongly');
    expect(rankPhrase({ below: 0, tied: 0, above: 0, n: 0 }, 'image')).toBe('with no test images to compare');
    // The default stays MNIST's.
    expect(rankPhrase(r(0, 0, 2000))).toBe('lower than all 2,000 test digits');
    expect(coveragePhrase('dense', 0.37, 0, 'point')).toBe('fires on 37% of points');
    expect(coveragePhrase('output', 0.104, 1, 'image')).toBe('predicted for 10.4% of images');
    expect(coveragePhrase('conv', 0.2741, 0, 'image')).toBe('fires at 27% of positions');
  });

  it('isDead: a hidden unit with one response for every sample, never an output unit (NEW-2)', () => {
    const net = new Network(mnistArch(SMALL_CNN), 1);
    // Push one dense unit's bias far below anything its inputs can reach: it never fires.
    net.blocks[2].b[0] = -1000;
    const dense = drain(topk(context(net), { block: 2, k: 4, count: 200 })).result;
    expect(isDead('dense', dense.units[0])).toBe(true);
    expect(dense.units[0].coverage).toBe(0);
    // Its "strongest" digits are ordered by pre-activation: the closest to firing come first.
    const z = dense.units[0].top.map((h) => h.z);
    for (let i = 1; i < z.length; i++) expect(z[i]).toBeLessThanOrEqual(z[i - 1]);
    // Units that respond differently to different digits are not dead.
    expect(dense.units.some((u) => !isDead('dense', u) && u.sorted[0] !== u.sorted[u.sorted.length - 1])).toBe(true);
    expect(isDead('output', { sorted: Float32Array.from([1, 1, 1]) })).toBe(false);
    expect(isDead('conv', { sorted: Float32Array.from([0, 0, 0]) })).toBe(true);
    expect(isDead('dense', { sorted: new Float32Array(0) })).toBe(false);
  });
});

/** A job context for any network and stored test set (stored value × scale = input). */
function anyContext(net: Network, testX: Uint8Array | Float32Array, testY: Uint8Array, scale: number): JobContext {
  const n = net.inputSize;
  return {
    net,
    arch: net.arch,
    spec: net.spec,
    inputSize: n,
    scale,
    classes: net.classes,
    testX,
    testY,
    image(i, out = new Float32Array(n)) {
      for (let j = 0; j < n; j++) out[j] = testX[i * n + j] * scale;
      return out;
    },
  };
}

const CIFAR_SHAPE = { c: 3, h: 32, w: 32 };
const colourArch = (layers: LayerSpec[]): Arch => ({ input: CIFAR_SHAPE, layers, classes: 10 });

/** Smooth random colour images: a blob of colour on a coloured ground, like a small photo. */
function colourImages(n: number, seed: number): { testX: Uint8Array; testY: Uint8Array } {
  const rng = new Rng(seed);
  const testX = new Uint8Array(n * 3072);
  const testY = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    testY[i] = i % 10;
    const cy = 6 + 20 * rng.next();
    const cx = 6 + 20 * rng.next();
    const rad = 3 + 8 * rng.next();
    for (let ch = 0; ch < 3; ch++) {
      const ground = 255 * rng.next();
      const blob = 255 * rng.next();
      for (let y = 0; y < 32; y++) {
        for (let x = 0; x < 32; x++) {
          const t = Math.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * rad * rad));
          testX[i * 3072 + ch * 1024 + y * 32 + x] = Math.round(ground + (blob - ground) * t);
        }
      }
    }
  }
  return { testX, testY };
}

suite('colour images (CIFAR-10 shape)', () => {
  it('actmax raises the objective of every layer kind in all three channels, starting from plain grey', () => {
    const net = new Network(colourArch(SMALL_CNN), 4);
    const { testX, testY } = colourImages(4, 1);
    const ctx = anyContext(net, testX, testY, 1 / 255);
    expect(synthBase(3)).toBe(0.5);
    expect(synthBase(1)).toBe(0);
    for (const block of [0, 1, 2, 3]) {
      const { result, reports } = drain(actmax(ctx, { block, units: [0, 1], steps: 60 }));
      for (const u of result.units) {
        expect(u.x).toHaveLength(3072);
        expect(u.final, `block ${block} unit ${u.unit}: ${u.start} → ${u.final}`).toBeGreaterThan(u.start);
        expect(Math.min(...u.x)).toBeGreaterThanOrEqual(0);
        expect(Math.max(...u.x)).toBeLessThanOrEqual(1);
        let colourful = 0;
        for (let p = 0; p < 1024; p++) {
          const r = Math.floor(p / 32);
          const c = p % 32;
          const inside = !u.box || (r >= u.box.y0 && r <= u.box.y1 && c >= u.box.x0 && c <= u.box.x1);
          if (!inside) for (let ch = 0; ch < 3; ch++) expect(u.x[ch * 1024 + p]).toBe(0.5);
          else if (Math.abs(u.x[p] - u.x[1024 + p]) > 0.05 || Math.abs(u.x[p] - u.x[2048 + p]) > 0.05) colourful++;
        }
        // The channels move apart: the result is a colour image, not grey.
        expect(colourful, `block ${block} unit ${u.unit}`).toBeGreaterThan(0);
        forwardTo(net, u.x, block);
        const b = net.blocks[block];
        const zi = b.kind === 'conv' ? u.unit * b.zShape.h * b.zShape.w + (b.zShape.h >> 1) * b.zShape.w + (b.zShape.w >> 1) : u.unit;
        expect(b.z[zi]).toBeCloseTo(u.final, 4);
      }
      expect((reports[0].partial as ActmaxPartial).x).toHaveLength(3072);
    }
  }, 60_000);

  it('a linear conv 1 synthesises the colour pattern its kernel weights describe', () => {
    // z = b + Σ w·x over the 3×3×3 patch: the optimum pushes each channel's pixel to 1 where its
    // weight is positive and to 0 where it is negative.
    const spec: LayerSpec[] = [{ kind: 'conv', filters: 4, kernel: 3, act: 'linear', pool: false }];
    const net = new Network(colourArch(spec), 12);
    const { testX, testY } = colourImages(2, 2);
    const { result } = drain(actmax(anyContext(net, testX, testY, 1 / 255), { block: 0, steps: 160 }));
    const W = net.blocks[0].W;
    for (const u of result.units) {
      expect(u.box).toEqual(box(15, 17, 15, 17));
      let ss = 0;
      for (let i = 0; i < 27; i++) ss += W[u.unit * 27 + i] ** 2;
      const rms = Math.sqrt(ss / 27);
      for (let ch = 0; ch < 3; ch++) {
        for (let ky = 0; ky < 3; ky++) {
          for (let kx = 0; kx < 3; kx++) {
            const w = W[u.unit * 27 + ch * 9 + ky * 3 + kx];
            const v = u.x[ch * 1024 + (15 + ky) * 32 + 15 + kx];
            if (w > 0.5 * rms) expect(v, `filter ${u.unit} channel ${ch} w=${w.toFixed(3)}`).toBeGreaterThan(0.6);
            if (w < -0.5 * rms) expect(v, `filter ${u.unit} channel ${ch} w=${w.toFixed(3)}`).toBeLessThan(0.4);
          }
        }
      }
    }
  });

  it('topk scans colour images: conv hits carry boxes inside the 32×32 image, dense hits none', () => {
    const net = new Network(colourArch(SMALL_CNN), 2);
    const { testX, testY } = colourImages(40, 3);
    const ctx = anyContext(net, testX, testY, 1 / 255);
    const conv = drain(topk(ctx, { block: 1, k: 4 })).result;
    expect(conv.count).toBe(40);
    expect(conv.units).toHaveLength(16);
    for (const u of conv.units) {
      for (const hit of u.top) {
        expect(hit.box).not.toBeNull();
        expect(hit.box!.y0).toBeGreaterThanOrEqual(0);
        expect(hit.box!.y1).toBeLessThanOrEqual(31);
        expect(hit.box!.x1).toBeLessThanOrEqual(31);
        expect(hit.y).toBeLessThan(16);
      }
      expect(u.labelCounts).toHaveLength(10);
      expect(u.labelCounts.reduce((a, b) => a + b, 0)).toBe(40);
    }
    const out = drain(topk(ctx, { block: 3, k: 4 })).result;
    expect(out.kind).toBe('output');
    for (const u of out.units) for (const hit of u.top) expect(hit.box).toBeNull();
    expect(out.units.reduce((a, u) => a + u.coverage, 0)).toBeCloseTo(1, 9);
  });
});

suite('point datasets', () => {
  const tanh2: LayerSpec[] = [
    { kind: 'dense', units: 8, act: 'tanh' },
    { kind: 'dense', units: 8, act: 'tanh' },
  ];

  it('topk ranks the test points of every layer; the output shares sum to 1', () => {
    const d = pointsData(defaultPointsConfig('circle'));
    const net = new Network({ input: d.input, layers: tanh2, classes: 2 }, 3);
    const ctx = anyContext(net, d.testX, d.testY, 1);
    for (const block of [0, 1, 2]) {
      const r = drain(topk(ctx, { block, k: 9 })).result;
      expect(r.count).toBe(d.testY.length);
      for (const u of r.units) {
        for (let i = 1; i < u.top.length; i++) expect(u.top[i].value).toBeLessThanOrEqual(u.top[i - 1].value);
        for (const hit of u.top) expect(hit.box).toBeNull();
        expect(u.labelCounts).toHaveLength(2);
        expect(coveragePhrase(r.kind, u.coverage, 0, 'point')).toMatch(/ of points$/);
      }
      if (r.kind === 'output') expect(r.units.reduce((a, u) => a + u.coverage, 0)).toBeCloseTo(1, 9);
    }
  });

  it('response maps from PointEvaluator activations equal a direct forward pass at every grid point', () => {
    for (const id of ['circle', 'shells'] as const) {
      const d = pointsData(defaultPointsConfig(id));
      const dims = d.points!.dims;
      const feats = d.points!.features;
      const net = new Network({ input: d.input, layers: tanh2, classes: 2 }, 5);
      const ev = new PointEvaluator();
      ev.sync(net);
      const res = 9;
      const fixedAt: number[] = [];
      if (dims === 3) fixedAt[2] = 0.5;
      const coords = gridCoords(dims, res, 1.25, [0, 1], fixedAt);
      const { acts } = ev.evaluate(coords, dims, feats, { activations: true });
      const x = featurize(coords, dims, feats);
      for (const block of [0, 1, 2]) {
        const U = net.blocks[block].out.length;
        for (const j of [0, U - 1]) {
          const col = unitColumn(acts![block], U, j);
          expect(col).toHaveLength(res * res);
          for (const q of [0, 40, res * res - 1]) {
            if (dims === 3) expect(coords[q * 3 + 2]).toBeCloseTo(0.5, 6);
            const v = unitResponse(net, block, j, x.slice(q * feats.length, (q + 1) * feats.length)).value;
            expect(col[q]).toBeCloseTo(v, 5);
          }
        }
      }
    }
  });

  it('mapScale: diverging when any value is negative, sequential from 0 otherwise', () => {
    const signed = mapScale(Float32Array.from([-0.5, 0.2, 0.9]));
    expect(signed.signed).toBe(true);
    expect(signed.max).toBeCloseTo(0.9, 6);
    expect(signed.lo).toBeCloseTo(-0.5, 6);
    const relu = mapScale(Float32Array.from([0, 0, 2.5]));
    expect(relu.signed).toBe(false);
    expect(relu.max).toBe(2.5);
    // A map of zeros (a dead unit) still has a usable scale.
    expect(mapScale(new Float32Array(4)).max).toBe(1);
    expect(mapScale(new Float32Array(0))).toEqual({ signed: false, max: 1, lo: 0, hi: 0 });
    expect(Array.from(unitColumn(Float32Array.from([1, 2, 3, 4, 5, 6]), 3, 1))).toEqual([2, 5]);
  });
});
