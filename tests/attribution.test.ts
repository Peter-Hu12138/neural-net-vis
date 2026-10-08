import { readFileSync } from 'node:fs';
import { inflateSync } from 'node:zlib';
import { describe, expect, it } from 'vitest';
import {
  attribution,
  computeAttribution,
  gradTimesInput,
  igAlphas,
  jobs,
  logitGradient,
  occlusionMap,
  oneHot,
  patchGrid,
  patchIsBlank,
  patchOrigins,
  relativeGap,
  saliencyOf,
  targetLogit,
  type AttributionResult,
} from '../src/analysis/attribution';
import { Analyzer } from '../src/analysis/analyzer';
import type { FromAnalyzer } from '../src/analysis/protocol';
import { Network } from '../src/nn/network';
import { Optimizer } from '../src/nn/optim';
import { Rng } from '../src/nn/rng';
import type { LayerSpec } from '../src/nn/types';

// ── Real digits from the bundled sprites (8-bit greyscale PNG, filter 0 on every row) ──

function readPng(path: string) {
  const buf = readFileSync(path);
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
  const px = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) px.set(raw.subarray(y * (width + 1) + 1, (y + 1) * (width + 1)), y * width);
  return { width, px };
}

const labels = Uint8Array.from(readFileSync('public/data/mnist-labels.txt', 'utf8').trim(), (ch) => ch.charCodeAt(0) - 48);
const testSprite = readPng('public/data/mnist-test.png');
const trainSprite = readPng('public/data/mnist-train-0.png');

function digitFrom(sprite: { width: number; px: Uint8Array }, i: number): Float32Array {
  const out = new Float32Array(784);
  const ox = (i % 100) * 28;
  const oy = Math.floor(i / 100) * 28;
  for (let r = 0; r < 28; r++) for (let c = 0; c < 28; c++) out[r * 28 + c] = sprite.px[(oy + r) * sprite.width + ox + c] / 255;
  return out;
}
const testDigit = (i: number) => digitFrom(testSprite, i);
const testLabel = (i: number) => labels[20_000 + i];

// ── Architectures (the app's presets) ──

const SOFTMAX: LayerSpec[] = [];
const MLP: LayerSpec[] = [{ kind: 'dense', units: 64, act: 'relu' }];
const SIGMOID_MLP: LayerSpec[] = [{ kind: 'dense', units: 32, act: 'sigmoid' }];
const SMALL_CNN: LayerSpec[] = [
  { kind: 'conv', filters: 8, kernel: 3, act: 'relu', pool: true },
  { kind: 'conv', filters: 16, kernel: 3, act: 'relu', pool: true },
  { kind: 'dense', units: 32, act: 'relu' },
];
const LENET: LayerSpec[] = [
  { kind: 'conv', filters: 6, kernel: 5, act: 'tanh', pool: true },
  { kind: 'conv', filters: 16, kernel: 5, act: 'tanh', pool: true },
  { kind: 'dense', units: 64, act: 'tanh' },
];

/** A few hundred Adam steps on real training digits: enough for a confident, structured network. */
const trainedCache = new Map<string, Network>();
function trained(spec: LayerSpec[], samples: number, seed = 3): Network {
  const key = JSON.stringify([spec, samples, seed]);
  const hit = trainedCache.get(key);
  if (hit) return hit;
  const net = new Network(spec, seed);
  trainedCache.set(key, net);
  const opt = new Optimizer(net, 'adam', 0.003);
  const batch = 16;
  for (let s = 0; s < samples; s += batch) {
    net.zeroGrad();
    for (let j = 0; j < batch; j++) {
      const i = (s + j) % 5000;
      net.forward(digitFrom(trainSprite, i));
      net.backward(labels[i]);
    }
    opt.step(1 / batch);
  }
  return net;
}

const run = (net: Network, x: Float32Array, target: number, igSteps = 32, occlusion = { size: 6, stride: 2 }) =>
  computeAttribution(net, { x, target, igSteps, occlusion });

describe('attribution helpers', () => {
  it('one-hot seeds and midpoint alphas', () => {
    expect(Array.from(oneHot(3))).toEqual([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]);
    expect(igAlphas(4)).toEqual([0.125, 0.375, 0.625, 0.875]);
    const a = igAlphas(32);
    expect(a.reduce((s, v) => s + v, 0) / 32).toBeCloseTo(0.5, 12);
  });

  it('patch origins step by the stride and always reach the far edge', () => {
    expect(patchOrigins(28, 6, 2)).toEqual([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]);
    expect(patchOrigins(28, 5, 2)).toEqual([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 23]);
    expect(patchOrigins(28, 28, 3)).toEqual([0]);
    expect(patchGrid(28, 6, 2)).toHaveLength(144);
    // every pixel is covered by at least one patch
    const covered = occlusionMap(patchGrid(28, 5, 3), new Float64Array(patchGrid(28, 5, 3).length).fill(1), 5);
    expect(covered.every((v) => v === 1)).toBe(true);
  });

  it('occlusion map is the per-pixel mean over covering patches (brute force)', () => {
    const side = 10;
    const size = 4;
    const patches = patchGrid(side, size, 3);
    const drops = patches.map((_, k) => Math.sin(k * 1.7) * 0.3);
    const map = occlusionMap(patches, drops, size, side);
    for (let r = 0; r < side; r++) {
      for (let c = 0; c < side; c++) {
        const cover = patches.map((p, k) => [p, k] as const).filter(([p]) => r >= p.y && r < p.y + size && c >= p.x && c < p.x + size);
        const mean = cover.length ? cover.reduce((s, [, k]) => s + drops[k], 0) / cover.length : 0;
        expect(map[r * side + c]).toBeCloseTo(mean, 6);
      }
    }
  });

  it('blank-patch test and relative gap', () => {
    const img = new Float32Array(784);
    img[3 * 28 + 4] = 0.5;
    expect(patchIsBlank(img, { y: 0, x: 0 }, 6)).toBe(false);
    expect(patchIsBlank(img, { y: 4, x: 0 }, 6)).toBe(true);
    expect(patchIsBlank(img, { y: 0, x: 5 }, 6)).toBe(true);
    expect(relativeGap(4.21, 4.3)).toBeCloseTo(0.09 / 4.3, 12);
    expect(relativeGap(0, 0)).toBe(0);
    expect(relativeGap(1, 0)).toBe(Infinity);
  });
});

describe('gradient of the target logit', () => {
  /** Picks the 12 largest-gradient pixels plus 40 spread over the image. */
  const pixels = (grad: Float32Array) => {
    const order = Array.from(grad.keys()).sort((a, b) => Math.abs(grad[b]) - Math.abs(grad[a]));
    return [...order.slice(0, 12), ...Array.from({ length: 40 }, (_, k) => (k * 331 + 17) % 784)];
  };
  const centralDiff = (net: Network, x: Float32Array, target: number, i: number, eps: number) => {
    const xp = x.slice();
    const xm = x.slice();
    xp[i] += eps;
    xm[i] -= eps;
    return (targetLogit(net, xp, target) - targetLogit(net, xm, target)) / (2 * eps);
  };

  it('matches central finite differences pixel by pixel on smooth networks (tanh CNN, sigmoid MLP), real digits', () => {
    const conv: LayerSpec[] = [
      { kind: 'conv', filters: 4, kernel: 5, act: 'tanh', pool: false },
      { kind: 'dense', units: 32, act: 'tanh' },
    ];
    const cases: [Network, number, number][] = [
      [new Network(conv, 5), 0, 7],
      [new Network(conv, 6), 1, 3],
      [new Network(SIGMOID_MLP, 2), 1, 2],
      [new Network(LENET.map((l) => (l.kind === 'conv' ? { ...l, pool: false } : l)), 3), 4, 4],
    ];
    for (const [net, i, target] of cases) {
      const x = testDigit(i);
      const { grad, logit } = logitGradient(net, x, target);
      expect(logit).toBeCloseTo(targetLogit(net, x, target), 6);
      const scale = Math.max(...Array.from(grad, Math.abs));
      for (const p of pixels(grad)) {
        const fd = centralDiff(net, x, target, p, 1e-2);
        expect(Math.abs(fd - grad[p]), `pixel ${p}`).toBeLessThan(2e-3 * scale + 2e-2 * Math.abs(grad[p]));
      }
    }
  });

  it('matches finite differences away from kinks on pooled networks (LeNet-ish, Small CNN)', () => {
    // Max-pool and ReLU are only piecewise smooth. Where a pooling window holds near-equal values
    // (blank areas of a digit hold exactly equal ones), ±ε can change the winner and a central
    // difference averages two slopes. So: random inputs, as in the engine's own gradient checks,
    // and at least 90% of the sampled pixels must agree. Completeness (below) checks the same
    // gradients globally, as a line integral.
    const cases: [LayerSpec[], number, number][] = [
      [LENET, 5, 7],
      [LENET, 8, 1],
      [SMALL_CNN, 4, 2],
    ];
    for (const [spec, seed, target] of cases) {
      const net = new Network(spec, seed);
      const rng = new Rng(seed);
      const x = Float32Array.from({ length: 784 }, () => rng.next());
      const { grad } = logitGradient(net, x, target);
      const scale = Math.max(...Array.from(grad, Math.abs));
      let ok = 0;
      const n = 60;
      for (let k = 0; k < n; k++) {
        const p = (k * 331 + 17) % 784;
        const fd = centralDiff(net, x, target, p, 1e-3);
        if (Math.abs(fd - grad[p]) <= 1e-3 * scale + 1e-2 * Math.abs(grad[p])) ok++;
      }
      expect(ok / n).toBeGreaterThanOrEqual(0.9);
    }
  });

  it('saliency is |g| and gradient × input is x ⊙ g', () => {
    const net = new Network(SMALL_CNN, 1);
    const x = testDigit(4);
    const r = run(net, x, 4, 4);
    const { grad, logit, prob } = logitGradient(net, x, 4);
    expect(r.logit).toBeCloseTo(logit, 6);
    expect(r.prob).toBeCloseTo(prob, 7);
    for (let i = 0; i < 784; i++) {
      expect(r.saliency[i]).toBeCloseTo(Math.abs(grad[i]), 7);
      expect(r.gradInput[i]).toBeCloseTo(x[i] * grad[i], 7);
    }
    expect(Array.from(saliencyOf([-2, 3]))).toEqual([2, 3]);
    expect(Array.from(gradTimesInput([0.5, 0], [4, 9]))).toEqual([2, 0]);
  });
});

describe('integrated gradients', () => {
  it('equals x ⊙ W_t exactly for the Softmax preset (no hidden layer), as does gradient × input', () => {
    const net = new Network(SOFTMAX, 7);
    const W = net.output.W; // 10 × 784
    const x = testDigit(2);
    for (const t of [0, 1, 9]) {
      const r = run(net, x, t, 32);
      for (let i = 0; i < 784; i++) {
        const want = x[i] * W[t * 784 + i];
        expect(r.integrated[i]).toBeCloseTo(want, 6);
        expect(r.gradInput[i]).toBeCloseTo(want, 6);
        expect(r.saliency[i]).toBeCloseTo(Math.abs(W[t * 784 + i]), 6);
      }
      // completeness holds exactly for a linear score
      expect(Math.abs(r.igSum - r.igExpected)).toBeLessThan(1e-4 * Math.max(1, Math.abs(r.igExpected)));
      expect(r.baseLogit).toBeCloseTo(net.output.b[t], 6);
    }
  });

  it('completeness: Σ IG is within 5% of z(x) − z(blank) at 32 steps, random and trained networks', () => {
    const nets: [string, Network][] = [
      ['random Small CNN', new Network(SMALL_CNN, 11)],
      ['random LeNet-ish', new Network(LENET, 12)],
      ['random MLP', new Network(MLP, 13)],
      ['trained MLP', trained(MLP, 1600)],
      ['trained Small CNN', trained(SMALL_CNN, 1200)],
    ];
    for (const [name, net] of nets) {
      for (const i of [0, 3, 8]) {
        const x = testDigit(i);
        const r = run(net, x, testLabel(i), 32, { size: 28, stride: 28 });
        expect(Math.abs(r.igExpected), `${name}: score change should be measurable`).toBeGreaterThan(0.01);
        expect(relativeGap(r.igSum, r.igExpected), `${name}, digit ${i}`).toBeLessThan(0.05);
      }
    }
  }, 30_000);

  it('the completeness gap shrinks with more steps (trained networks)', () => {
    for (const net of [trained(MLP, 1600), trained(SMALL_CNN, 1200)]) {
      const gaps = [2, 32, 256].map((m) => {
        let sum = 0;
        for (const i of [0, 3, 5, 7, 8]) {
          const r = run(net, testDigit(i), testLabel(i), m, { size: 28, stride: 28 });
          sum += relativeGap(r.igSum, r.igExpected);
        }
        return sum / 5;
      });
      expect(gaps[1]).toBeLessThan(gaps[0]);
      expect(gaps[2]).toBeLessThan(gaps[1]);
      expect(gaps[2]).toBeLessThan(0.005);
    }
  }, 30_000);

  it('is exact at any step count for a bias-free ReLU network, where it equals gradient × input', () => {
    // ReLU, max-pool and bias-free layers are positively homogeneous: f(αx) = α·f(x), so the
    // gradient is the same all along the path and Σ x ⊙ g = f(x) − f(0) (Euler's theorem).
    for (const spec of [MLP, SMALL_CNN]) {
      const net = new Network(spec, 31); // biases start at 0
      const x = testDigit(9);
      for (const m of [1, 5]) {
        const r = run(net, x, 4, m, { size: 28, stride: 28 });
        expect(r.baseLogit).toBe(0);
        for (let i = 0; i < 784; i++) expect(r.integrated[i]).toBeCloseTo(r.gradInput[i], 6);
        expect(r.igSum).toBeCloseTo(r.igExpected, 5);
      }
    }
  });
});

describe('occlusion', () => {
  it('gives exactly 0 wherever every covering patch is already blank', () => {
    const net = new Network(SMALL_CNN, 3);
    // ink only in the top-left 10×10 corner: a crop of a real digit
    const d = testDigit(0);
    const x = new Float32Array(784);
    for (let r = 0; r < 10; r++) for (let c = 0; c < 10; c++) x[r * 28 + c] = d[(r + 8) * 28 + c + 8];
    const r = run(net, x, 7, 4);
    expect(r.patchesEvaluated).toBeLessThan(r.patches);
    for (let row = 0; row < 28; row++) {
      for (let col = 0; col < 28; col++) {
        // a 6×6 patch at stride 2 that reaches (row, col) starts at ≥ row − 5; beyond row/col 15 none touch the ink
        if (row >= 16 || col >= 16) expect(r.occlusion[row * 28 + col], `(${row}, ${col})`).toBe(0);
      }
    }
    // a completely blank input changes nothing anywhere
    const blank = run(net, new Float32Array(784), 3, 4);
    expect(blank.patchesEvaluated).toBe(0);
    expect(blank.occlusion.every((v) => v === 0)).toBe(true);
    expect(blank.gradInput.every((v) => v === 0)).toBe(true);
    expect(blank.integrated.every((v) => v === 0)).toBe(true);
    expect(blank.igSum).toBe(0);
    expect(blank.igExpected).toBe(0);
  });

  it('matches a brute-force sweep that erases every patch', () => {
    const net = new Network(LENET, 9);
    const x = testDigit(6);
    const t = testLabel(6);
    const size = 7;
    const stride = 4;
    const r = run(net, x, t, 2, { size, stride });
    const p0 = net.forward(x)[t];
    const patches = patchGrid(28, size, stride);
    const drops = patches.map((p) => {
      const e = x.slice();
      for (let rr = p.y; rr < Math.min(28, p.y + size); rr++) for (let cc = p.x; cc < Math.min(28, p.x + size); cc++) e[rr * 28 + cc] = 0;
      return p0 - net.forward(e)[t];
    });
    const want = occlusionMap(patches, drops, size);
    for (let i = 0; i < 784; i++) expect(r.occlusion[i]).toBeCloseTo(want[i], 6);
    expect(r.occlusionSize).toBe(size);
    expect(r.occlusionStride).toBe(stride);
  });

  it('erasing ink of a confidently classified digit lowers its probability somewhere', () => {
    const net = trained(SMALL_CNN, 1200);
    const i = 0;
    const r = run(net, testDigit(i), testLabel(i));
    expect(r.pred).toBe(testLabel(i));
    expect(Math.max(...r.occlusion)).toBeGreaterThan(0.01);
  }, 30_000);
});

describe('attribution job', () => {
  it('validates its parameters', () => {
    const net = new Network(SOFTMAX, 1);
    expect(() => computeAttribution(net, { x: new Float32Array(10), target: 1 })).toThrow(/28×28/);
    expect(() => computeAttribution(net, { x: new Float32Array(784), target: 10 })).toThrow(/digit/);
    const r = computeAttribution(net, { x: testDigit(0), target: 7, igSteps: 0, occlusion: { size: 99, stride: 0 } });
    expect(r.igSteps).toBe(1);
    expect(r.occlusionSize).toBe(28);
    expect(r.occlusionStride).toBe(1);
  });

  it('yields once per forward pass and reports exact progress', () => {
    const net = new Network(SMALL_CNN, 1);
    const x = testDigit(0);
    const gen = attribution(net, { x, target: 7, igSteps: 8 });
    const seen: number[] = [];
    let total = 0;
    let res: AttributionResult | undefined;
    for (;;) {
      const s = gen.next();
      if (s.done) {
        res = s.value;
        break;
      }
      seen.push(s.value.done);
      total = s.value.total;
    }
    expect(res!.patches).toBe(144);
    expect(total).toBe(2 + 8 + res!.patchesEvaluated);
    expect(seen).toEqual(Array.from({ length: total }, (_, k) => k + 1));
  });

  it('runs end to end through the Analyzer on a real CNN in under 3 s', async () => {
    const spec = SMALL_CNN;
    const source = trained(spec, 320);
    const testX = new Uint8Array(2000 * 784);
    for (let i = 0; i < 2000; i++) {
      const d = testDigit(i);
      for (let j = 0; j < 784; j++) testX[i * 784 + j] = Math.round(d[j] * 255);
    }
    const testY = labels.slice(20_000, 22_000);
    const log: FromAnalyzer[] = [];
    const a = new Analyzer((m) => log.push(m), jobs, 30);
    a.handle({ type: 'data', testX, testY });
    const x = testDigit(0);
    const t0 = performance.now();
    a.handle({ type: 'run', id: 1, channel: 'attribution', kind: 'attribution', params: { x, target: 7, igSteps: 32, occlusion: { size: 6, stride: 2 } }, spec, weights: source.getWeights() });
    while (!log.some((m) => m.type === 'result' || m.type === 'error') && performance.now() - t0 < 10_000) await new Promise((r) => setTimeout(r, 2));
    const ms = performance.now() - t0;
    const msg = log.find((m) => m.type === 'result' || m.type === 'error')!;
    expect(msg.type).toBe('result');
    const r = structuredClone((msg as { result: AttributionResult }).result);
    console.log(`attribution job (Small CNN, 32 IG steps, 6×6 / 2 occlusion, ${r.patchesEvaluated}/${r.patches} patches): ${ms.toFixed(0)} ms`);
    expect(ms).toBeLessThan(3000);
    expect(r.target).toBe(7);
    expect(r.saliency).toBeInstanceOf(Float32Array);
    expect(r.integrated).toHaveLength(784);
    expect(r.occlusion).toHaveLength(784);
    expect(relativeGap(r.igSum, r.igExpected)).toBeLessThan(0.05);
    // the job saw the same weights: its logit matches the source network's
    expect(r.logit).toBeCloseTo(targetLogit(source, x, 7), 5);
  }, 20_000);
});
