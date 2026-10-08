import { readFileSync } from 'node:fs';
import { inflateSync } from 'node:zlib';
import { describe as suite, expect, it } from 'vitest';
import type { JobContext, Progress } from '../src/analysis/protocol';
import { receptiveBox, receptiveSize, type Box } from '../src/analysis/receptive';
import { actmax, forwardTo, HIST_BINS, jobs, labelSummary, topk, unitResponse, type ActmaxPartial, type ActmaxResult, type TopkResult } from '../src/analysis/units';
import { Network } from '../src/nn/network';
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
    spec: net.spec,
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
    expect(receptiveBox(spec, 0, 5, 5, 'z')).toEqual(box(4, 6, 4, 6));
    expect(receptiveBox(spec, 0, 5, 9, 'z')).toEqual(box(4, 6, 8, 10));
    expect(receptiveSize(spec, 0)).toBe(3);
  });

  it('Small CNN: conv 2 at (5, 5) of its pre-pool map sees rows and columns 7–14', () => {
    expect(receptiveBox(SMALL_CNN, 1, 5, 5, 'z')).toEqual(box(7, 14, 7, 14));
    expect(receptiveSize(SMALL_CNN, 1)).toBe(8);
    expect(receptiveSize(SMALL_CNN, 0)).toBe(3);
    // Pooled output of conv 1 at (5, 5) covers z rows 10–11, so pixels 9–12.
    expect(receptiveBox(SMALL_CNN, 0, 5, 5, 'out')).toEqual(box(9, 12, 9, 12));
    // Pooled output of conv 2 at (3, 3): z rows 6–7 → conv-1 out 5–8 → z 10–17 → pixels 9–18.
    expect(receptiveBox(SMALL_CNN, 1, 3, 3, 'out')).toEqual(box(9, 18, 9, 18));
    expect(receptiveSize(SMALL_CNN, 1, 'out')).toBe(10);
  });

  it('LeNet-ish 5×5 kernels give a 14-pixel field for conv 2', () => {
    expect(receptiveSize(LENET, 0)).toBe(5);
    expect(receptiveSize(LENET, 1)).toBe(14);
    expect(receptiveBox(LENET, 1, 7, 7, 'z')).toEqual(box(8, 21, 8, 21));
  });

  it('clips at the corners', () => {
    expect(receptiveBox(SMALL_CNN, 0, 0, 0, 'z')).toEqual(box(0, 1, 0, 1));
    expect(receptiveBox(SMALL_CNN, 0, 27, 27, 'z')).toEqual(box(26, 27, 26, 27));
    expect(receptiveBox(SMALL_CNN, 1, 0, 0, 'z')).toEqual(box(0, 4, 0, 4));
    expect(receptiveBox(SMALL_CNN, 1, 13, 13, 'z')).toEqual(box(23, 27, 23, 27));
    expect(receptiveBox(SMALL_CNN, 1, 0, 13, 'z')).toEqual(box(0, 4, 23, 27));
    // Unclipped, the nominal field keeps its full size.
    expect(receptiveBox(SMALL_CNN, 1, 0, 0, 'z', false)).toEqual(box(-3, 4, -3, 4));
  });

  it('works through three pools (28 → 14 → 7 → 3)', () => {
    const spec: LayerSpec[] = [
      { kind: 'conv', filters: 2, kernel: 3, act: 'linear', pool: true },
      { kind: 'conv', filters: 2, kernel: 3, act: 'linear', pool: true },
      { kind: 'conv', filters: 2, kernel: 3, act: 'linear', pool: true },
    ];
    // out (0, 0) → z 0–1 → 0–2 → pool 0–5 → 0–6 → pool 0–13 → 0–14
    expect(receptiveBox(spec, 2, 0, 0, 'out')).toEqual(box(0, 14, 0, 14));
    // out (2, 2) → z 4–5 → 3–6 → pool 6–13 → 5–14 → pool 10–29 → 9–30, clipped to 27
    expect(receptiveBox(spec, 2, 2, 2, 'out')).toEqual(box(9, 27, 9, 27));
    expect(receptiveSize(spec, 2, 'out')).toBe(22);
  });

  it('dense and output blocks see the whole image', () => {
    expect(receptiveBox(SMALL_CNN, 2, 0, 0)).toBeNull();
    expect(receptiveBox(SMALL_CNN, 3, 0, 0)).toBeNull();
    expect(receptiveSize(SMALL_CNN, 2)).toBe(28);
    expect(receptiveSize(SMALL_CNN, 3)).toBe(28);
  });

  /**
   * Perturbs each input pixel by ±δ and records which positions of block `block` (level z, or
   * the pooled output) change. Returns the bounding box of the pixels that moved each position.
   */
  function bruteForce(spec: LayerSpec[], block: number, level: 'z' | 'out', positions: [number, number][], deltas: number[], seed: number) {
    const net = new Network(spec, seed);
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
    positions.forEach(([y, x], k) => expect(found[k], `position ${y},${x}`).toEqual(receptiveBox(spec, 2, y, x, 'z')));
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
    positions.forEach(([y, x], k) => expect(found[k], `position ${y},${x}`).toEqual(receptiveBox(spec, 1, y, x, 'z')));
    // The pooled output of block 0 itself, at a few positions.
    const outPos: [number, number][] = [[5, 5], [0, 13], [13, 0]];
    const foundOut = bruteForce(spec, 0, 'out', outPos, [25, -25], 7);
    outPos.forEach(([y, x], k) => expect(foundOut[k], `out ${y},${x}`).toEqual(receptiveBox(spec, 0, y, x, 'out')));
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
      const want = receptiveBox(spec, 2, y, x, 'z')!;
      const got = found[k]!;
      expect(got).not.toBeNull();
      expect(got.y0).toBeGreaterThanOrEqual(want.y0);
      expect(got.y1).toBeLessThanOrEqual(want.y1);
      expect(got.x0).toBeGreaterThanOrEqual(want.x0);
      expect(got.x1).toBeLessThanOrEqual(want.x1);
    });
  });
});

/** Every unit's response to every test digit, computed the slow, obvious way. */
function bruteResponses(net: Network, block: number, n: number) {
  const b = net.blocks[block];
  const out = block === net.blocks.length - 1;
  const U = b.kind === 'conv' ? b.spec.filters : b.spec.units;
  const r: number[][] = Array.from({ length: U }, () => []);
  const x = new Float32Array(784);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < 784; j++) x[j] = data.testX[i * 784 + j] / 255;
    net.forward(x);
    for (let u = 0; u < U; u++) {
      if (b.kind === 'conv') {
        const HW = b.zShape.h * b.zShape.w;
        let m = -Infinity;
        for (let q = 0; q < HW; q++) m = Math.max(m, b.a[u * HW + q]);
        r[u].push(m);
      } else r[u].push(out ? b.z[u] : b.a[u]);
    }
  }
  return r;
}

suite('topk job', () => {
  it('is registered with actmax', () => {
    expect(Object.keys(jobs).sort()).toEqual(['actmax', 'topk']);
  });

  for (const block of [0, 1, 2, 3]) {
    it(`matches a brute-force scan on Small CNN block ${block}`, () => {
      const net = new Network(SMALL_CNN, 4);
      const n = 300;
      const { result, reports } = drain(topk(context(net), { block, k: 9, count: n }));
      const brute = bruteResponses(new Network(SMALL_CNN, 4), block, n);
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
        expect(u.activeFraction).toBeCloseTo(r.filter((v) => v > 0).length / n, 6);
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
    const net = new Network(SMALL_CNN, 9);
    const { result } = drain(topk(context(net), { block: 1, k: 4, count: 120 }));
    const probe = new Network(SMALL_CNN, 9);
    for (const u of result.units.slice(0, 5)) {
      for (const hit of u.top) {
        expect(hit.box).toEqual(receptiveBox(SMALL_CNN, 1, hit.y, hit.x, 'z'));
        const x = context(probe).image(hit.index);
        const r = unitResponse(probe, 1, u.unit, x);
        expect(r.value).toBeCloseTo(hit.value, 6);
        expect([r.y, r.x]).toEqual([hit.y, hit.x]);
        // The value at that position is the filter's activation there.
        const b = probe.blocks[1];
        expect(b.a[u.unit * 196 + hit.y * 14 + hit.x]).toBeCloseTo(hit.value, 6);
      }
    }
  });

  it('dense and output hits carry no box', () => {
    const net = new Network(SMALL_CNN, 2);
    const { result } = drain(topk(context(net), { block: 3, k: 3, count: 50 }));
    expect(result.units).toHaveLength(10);
    expect(result.units[0].top[0]).toMatchObject({ box: null, y: -1, x: -1 });
  });

  it('labels of the top 50 follow the digit a trained output unit stands for', () => {
    // Train a softmax (no hidden layers) for two quick SGD passes over the first 1,000 digits.
    const net = new Network([], 1);
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
    const net = new Network(SMALL_CNN, 1);
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
    const net = new Network(SMALL_CNN, 3);
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
      const net = new Network(spec as LayerSpec[], 6);
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
    const net = new Network(SMALL_CNN, 2);
    const a = drain(actmax(context(net), { block: 2, units: [4, 5], steps: 20 })).result;
    const b = drain(actmax(context(new Network(SMALL_CNN, 2)), { block: 2, units: [4], steps: 20 })).result;
    expect(Array.from(b.units[0].x)).toEqual(Array.from(a.units[0].x));
    expect(Array.from(a.units[1].x)).not.toEqual(Array.from(a.units[0].x));
  });

  it('conv 1 synthesises the pattern its kernel weights describe', () => {
    // With a single linear 3×3 conv, z = b + Σ w·x over the patch: the optimum puts ink (1) where
    // the weight is positive and leaves 0 where it is negative.
    const spec: LayerSpec[] = [{ kind: 'conv', filters: 4, kernel: 3, act: 'linear', pool: false }];
    const net = new Network(spec, 12);
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
      const net = new Network(SMALL_CNN, 1);
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
  const net = new Network(SMALL_CNN, 1);
  const t = drain(topk(context(net), { block: 1, k: 2, count: 20 })).result;
  const a = drain(actmax(context(net), { block: 1, units: [0], steps: 4 })).result;
  expect(structuredClone(t) as TopkResult).toEqual(t);
  expect((structuredClone(a) as ActmaxResult).units[0].x).toBeInstanceOf(Float32Array);
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
