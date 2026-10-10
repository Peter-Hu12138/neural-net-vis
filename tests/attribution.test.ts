import { readFileSync } from 'node:fs';
import { inflateSync } from 'node:zlib';
import { describe, expect, it } from 'vitest';
import { mnistArch } from '../src/nn/types';
import {
  attribution,
  completenessGap,
  completenessText,
  computeAttribution,
  countKinks,
  exact,
  gradTimesInput,
  igAlphas,
  jobs,
  coordinateGradient,
  featureSentence,
  kinkText,
  logitGradient,
  meanColour,
  occlusionMap,
  patchIsFilled,
  sumChannels,
  oneHot,
  patchGrid,
  patchIsBlank,
  patchOrigins,
  percentText,
  relativeGap,
  saliencyOf,
  sig,
  targetLogit,
  type AttributionResult,
} from '../src/analysis/attribution';
import { Analyzer } from '../src/analysis/analyzer';
import type { FromAnalyzer } from '../src/analysis/protocol';
import { Network } from '../src/nn/network';
import { Optimizer } from '../src/nn/optim';
import { Rng } from '../src/nn/rng';
import type { Arch, LayerSpec } from '../src/nn/types';
import { featureCatalog, featurize } from '../src/data/features';

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
  const net = new Network(mnistArch(spec), seed);
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
      [new Network(mnistArch(conv), 5), 0, 7],
      [new Network(mnistArch(conv), 6), 1, 3],
      [new Network(mnistArch(SIGMOID_MLP), 2), 1, 2],
      [new Network(mnistArch(LENET.map((l) => (l.kind === 'conv' ? { ...l, pool: false } : l))), 3), 4, 4],
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
      const net = new Network(mnistArch(spec), seed);
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

  /** One-sided derivatives (+ε, −ε) of z_t at every blank pixel of x, and the gradients there. */
  function blankPixels(net: Network, x: Float32Array, target: number, eps = 1e-3) {
    const last = net.blocks.length - 1;
    const sym = logitGradient(net, x, target).grad; // the attribution job's gradient (symmetric)
    net.forward(x);
    const train = net.inputGradient(last, oneHot(target), false).slice(); // what training uses
    const z0 = targetLogit(net, x, target);
    const out: { i: number; sym: number; train: number; right: number; left: number }[] = [];
    for (let i = 0; i < 784; i++) {
      if (x[i] !== 0) continue;
      const xp = x.slice();
      const xm = x.slice();
      xp[i] += eps;
      xm[i] -= eps;
      out.push({ i, sym: sym[i], train: train[i], right: (targetLogit(net, xp, target) - z0) / eps, left: (z0 - targetLogit(net, xm, target)) / eps });
    }
    return { pixels: out, max: Math.max(...Array.from(sym, Math.abs)) };
  }

  it('on blank pixels of a real digit, ReLU kinks no longer zero out the saliency (ATTR-2)', () => {
    // The page's first state: Small CNN, seed 1, untrained. Biases start at 0, so over the blank
    // background every conv unit sits exactly at the ReLU kink z = 0. The training convention
    // (slope 0 there) blanks out pixels whose score does move when they are brightened.
    const net = new Network(mnistArch(SMALL_CNN), 1);
    for (const [d, target] of [[0, 7], [1, 2]]) {
      const x = testDigit(d);
      net.forward(x);
      const kinks = countKinks(net);
      expect(kinks.relu, `digit ${d}`).toBeGreaterThan(1000);
      const { pixels, max } = blankPixels(net, x, target);
      const tol = 0.02 * max;
      const moves = (p: (typeof pixels)[number]) => Math.max(Math.abs(p.right), Math.abs(p.left)) > tol;
      const zeroTrain = pixels.filter((p) => p.train === 0 && moves(p)).length;
      const zeroSym = pixels.filter((p) => p.sym === 0 && moves(p)).length;
      expect(zeroTrain, `digit ${d}: the artefact is there with the training convention`).toBeGreaterThan(300);
      expect(zeroSym, `digit ${d}: and gone with the symmetric one`).toBe(0);
      // The symmetric slope lies between the two one-sided slopes on the vast majority of pixels.
      const between = (g: number, p: (typeof pixels)[number]) => g >= Math.min(p.left, p.right) - tol && g <= Math.max(p.left, p.right) + tol;
      expect(pixels.filter((p) => between(p.sym, p)).length / pixels.length, `digit ${d}`).toBeGreaterThan(0.8);
    }
  }, 30_000);

  it('on blank pixels of a real digit, tied max-pool windows leave no lattice (ATTR-2)', () => {
    // Trained LeNet-ish (tanh, so the only kinks are max-pool ties). Over the background the four
    // values of a pooling window are exactly equal; sending the whole gradient to the top-left one
    // made a period-2 lattice: (odd row, odd column) pixels got much less. Finite differences show
    // no such pattern, and neither does the symmetric gradient.
    const net = trained(LENET, 1200);
    for (const d of [1, 2]) {
      const x = testDigit(d);
      net.forward(x);
      expect(countKinks(net).pool, `digit ${d}`).toBeGreaterThan(100);
      const { pixels, max } = blankPixels(net, x, testLabel(d));
      const oddOdd = (i: number) => Math.floor(i / 28) % 2 === 1 && (i % 28) % 2 === 1;
      const ratio = (f: (p: (typeof pixels)[number]) => number) => {
        const sum = [0, 0];
        const n = [0, 0];
        for (const p of pixels) {
          const k = oddOdd(p.i) ? 0 : 1;
          sum[k] += Math.abs(f(p));
          n[k]++;
        }
        return sum[0] / n[0] / (sum[1] / n[1]);
      };
      const fd = ratio((p) => (p.left + p.right) / 2);
      expect(fd, `digit ${d}: finite differences are flat`).toBeGreaterThan(0.9);
      expect(fd).toBeLessThan(1.1);
      expect(ratio((p) => p.train), `digit ${d}: lattice with the training convention`).toBeLessThan(0.75);
      expect(Math.abs(ratio((p) => p.sym) - fd), `digit ${d}: symmetric gradient as flat as the finite differences`).toBeLessThan(0.1);
      // ...and closer to the finite differences pixel by pixel.
      const err = (f: (p: (typeof pixels)[number]) => number) => pixels.reduce((s, p) => s + Math.abs(f(p) - (p.left + p.right) / 2), 0) / pixels.length;
      expect(err((p) => p.sym)).toBeLessThan(0.6 * err((p) => p.train));
      const tol = 0.02 * max;
      const between = pixels.filter((p) => p.sym >= Math.min(p.left, p.right) - tol && p.sym <= Math.max(p.left, p.right) + tol).length;
      expect(between / pixels.length, `digit ${d}`).toBeGreaterThan(0.95);
    }
  }, 30_000);

  it('counts kinks only where they exist, and the Saliency hint mentions them only then', () => {
    const smooth = new Network(mnistArch(SIGMOID_MLP), 2);
    smooth.forward(testDigit(0));
    expect(countKinks(smooth)).toEqual({ relu: 0, pool: 0 });
    const linear = new Network(mnistArch(SOFTMAX), 1);
    linear.forward(testDigit(0));
    expect(countKinks(linear)).toEqual({ relu: 0, pool: 0 });
    expect(kinkText({ relu: 0, pool: 0 })).toBe('');
    expect(kinkText({ relu: 3, pool: 0 })).toContain('ReLUs at exactly 0');
    expect(kinkText({ relu: 3, pool: 0 })).not.toContain('max-pool');
    expect(kinkText({ relu: 0, pool: 5 })).toBe(
      'Where the input leaves tied max-pool windows (mostly the blank background), brightening and darkening a pixel differ; the map shows the average slope.',
    );
    expect(kinkText({ relu: 2, pool: 5 })).toContain('ReLUs at exactly 0 and tied max-pool windows');
    // A ReLU tie among dead units (all z < 0) is flat on both sides: not a kink.
    const net = new Network(mnistArch([{ kind: 'conv', filters: 1, kernel: 3, act: 'relu', pool: true }]), 1);
    net.blocks[0].W.fill(0);
    net.blocks[0].b.fill(-0.5);
    net.forward(new Float32Array(784));
    expect(countKinks(net)).toEqual({ relu: 0, pool: 0 });
    net.blocks[0].b.fill(0.5);
    net.forward(new Float32Array(784));
    expect(countKinks(net)).toEqual({ relu: 0, pool: 196 });
    // The job reports them.
    expect(run(new Network(mnistArch(SMALL_CNN), 1), testDigit(0), 7, 2).kinks.relu).toBeGreaterThan(1000);
  });

  it('saliency is |g| and gradient × input is x ⊙ g', () => {
    const net = new Network(mnistArch(SMALL_CNN), 1);
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
    const net = new Network(mnistArch(SOFTMAX), 7);
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
      ['random Small CNN', new Network(mnistArch(SMALL_CNN), 11)],
      ['random LeNet-ish', new Network(mnistArch(LENET), 12)],
      ['random MLP', new Network(mnistArch(MLP), 13)],
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
      const net = new Network(mnistArch(spec), 31); // biases start at 0
      const x = testDigit(9);
      for (const m of [1, 5]) {
        const r = run(net, x, 4, m, { size: 28, stride: 28 });
        expect(r.baseLogit).toBe(0);
        for (let i = 0; i < 784; i++) expect(r.integrated[i]).toBeCloseTo(r.gradInput[i], 6);
        expect(r.igSum).toBeCloseTo(r.igExpected, 5);
      }
    }
  });

  it('the completeness gap is measured against Σ|IG|, so it cannot blow up when z(x) − z(blank) ≈ 0 (ATTR-1)', () => {
    // Trained MLP, test digit 1 (a 2), target 8: the network neither likes nor rejects an 8 here,
    // so positive and negative attributions cancel and z(x) − z(blank) is close to 0. Relative to
    // that difference, the gap read "78% apart"; relative to the attributions it is tiny.
    const net = trained(MLP, 1600);
    const r = run(net, testDigit(1), 8, 32, { size: 28, stride: 28 });
    expect(Math.abs(r.igExpected)).toBeLessThan(0.01);
    expect(relativeGap(r.igSum, r.igExpected)).toBeGreaterThan(0.5); // the old measure
    let abs = 0;
    for (const v of r.integrated) abs += Math.abs(v);
    expect(r.igAbsSum).toBeCloseTo(abs, 4);
    expect(r.igAbsSum).toBeGreaterThan(1);
    const g = completenessGap(r.igSum, r.igExpected, r.igAbsSum);
    expect(g.ref).toBe('ig');
    expect(g.diff).toBeCloseTo(Math.abs(r.igSum - r.igExpected), 12);
    expect(g.rel).toBeLessThan(0.005);
    // The line shows both numbers with three significant digits (not "0.00" twice) and says what
    // the percentage is of.
    const c = completenessText(r);
    expect(c.sum).toBe(`Σ IG = ${sig(r.igSum)};`);
    expect(c.expected).toBe(`z(x) − z(blank) = ${sig(r.igExpected)}`);
    expect(c.sum).not.toMatch(/= −?0(\.0+)?;$/);
    expect(c.expected).not.toMatch(/= −?0(\.0+)?$/);
    expect(c.sum).not.toBe(c.expected.replace('z(x) − z(blank)', 'Σ IG') + ';');
    expect(c.gap).toMatch(/^\(off by [\d.e−]+, (<0\.1|0\.\d)% of Σ\|IG\|\)$/);
    expect(c.title).toContain(`Σ|IG| = ${exact(r.igAbsSum)}`);
  }, 30_000);

  it('the completeness gap stays under 1% of Σ|IG| for every target, trained MLP and Small CNN (ATTR-1)', () => {
    for (const [name, net, digits] of [
      ['trained MLP', trained(MLP, 1600), 30],
      ['trained Small CNN', trained(SMALL_CNN, 1200), 20],
    ] as const) {
      let nearZero = 0;
      for (let d = 0; d < digits; d++) {
        for (let t = 0; t < 10; t++) {
          const r = run(net, testDigit(d), t, 32, { size: 28, stride: 28 });
          if (Math.abs(r.igExpected) < 0.1) nearZero++;
          const g = completenessGap(r.igSum, r.igExpected, r.igAbsSum);
          expect(g.rel, `${name}, digit ${d}, target ${t}`).toBeLessThan(0.01);
          expect(Number.isFinite(g.rel)).toBe(true);
        }
      }
      expect(nearZero, `${name}: the scan reaches the near-zero case`).toBeGreaterThan(0);
    }
  }, 60_000);

  it('completeness text: a match, the score as reference, and number formatting', () => {
    expect(completenessText({ igSum: 0, igExpected: 0, igAbsSum: 0 })).toMatchObject({ sum: 'Σ IG = 0;', expected: 'z(x) − z(blank) = 0', gap: '(a match)' });
    // IG missing most of a real score change: the gap is measured against the score instead.
    const g = completenessGap(0.1, 2, 0.1);
    expect(g).toEqual({ diff: 1.9, rel: 0.95, ref: 'score' });
    expect(completenessText({ igSum: 0.1, igExpected: 2, igAbsSum: 0.1 }).gap).toBe('(off by 1.9, 95% of |z(x) − z(blank)|)');
    expect(completenessText({ igSum: 8.3912, igExpected: 8.4301, igAbsSum: 12.07 }).gap).toBe('(off by 0.0389, 0.3% of Σ|IG|)');
    expect(completenessGap(0, 0, 0).rel).toBe(0);
    expect(sig(-0.000173)).toBe('−1.7e−4');
    expect(sig(0.00528)).toBe('0.00528');
    expect(sig(-8.3912)).toBe('−8.39');
    expect(sig(0.31, true)).toBe('+0.31');
    expect(sig(-0)).toBe('0');
    expect(exact(-0.00528312)).toBe('−0.00528312');
    expect(percentText(0.0004)).toBe('<0.1%');
    expect(percentText(0.0312)).toBe('3.1%');
    expect(percentText(0.123)).toBe('12%');
  });
});

describe('occlusion', () => {
  it('gives exactly 0 wherever every covering patch is already blank', () => {
    const net = new Network(mnistArch(SMALL_CNN), 3);
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

  it('matches a brute-force sweep that erases every patch (logit map and probability map)', () => {
    const net = new Network(mnistArch(LENET), 9);
    const x = testDigit(6);
    const t = testLabel(6);
    const size = 7;
    const stride = 4;
    const r = run(net, x, t, 2, { size, stride });
    const p0 = net.forward(x)[t];
    const z0 = net.blocks[net.blocks.length - 1].z[t];
    const patches = patchGrid(28, size, stride);
    const zDrops: number[] = [];
    const pDrops: number[] = [];
    for (const p of patches) {
      const e = x.slice();
      for (let rr = p.y; rr < Math.min(28, p.y + size); rr++) for (let cc = p.x; cc < Math.min(28, p.x + size); cc++) e[rr * 28 + cc] = 0;
      pDrops.push(p0 - net.forward(e)[t]);
      zDrops.push(z0 - net.blocks[net.blocks.length - 1].z[t]);
    }
    const want = occlusionMap(patches, zDrops, size);
    const wantP = occlusionMap(patches, pDrops, size);
    for (let i = 0; i < 784; i++) {
      expect(r.occlusion[i]).toBeCloseTo(want[i], 5);
      expect(r.occlusionProb[i]).toBeCloseTo(wantP[i], 6);
    }
    expect(r.occlusionSize).toBe(size);
    expect(r.occlusionStride).toBe(stride);
  });

  it('erasing ink of a confidently classified digit lowers its score somewhere', () => {
    const net = trained(SMALL_CNN, 1200);
    const i = 0;
    const r = run(net, testDigit(i), testLabel(i));
    expect(r.pred).toBe(testLabel(i));
    expect(Math.max(...r.occlusion)).toBeGreaterThan(0.1);
  }, 30_000);

  it('the logit map keeps its detail where the probability saturates (ATTR-5)', () => {
    // A confident digit (p ≈ 1) and a rejected one (p ≈ 0): erasing a patch barely moves p, so a
    // probability map would be noise stretched to full contrast. The logit still moves clearly.
    const net = trained(SMALL_CNN, 1200);
    const x = testDigit(0);
    const label = testLabel(0);
    const confident = run(net, x, label);
    expect(confident.prob).toBeGreaterThan(0.98);
    const rejectedTarget = Array.from(confident.probs.keys()).sort((a, b) => confident.probs[a] - confident.probs[b])[0];
    const rejected = run(net, x, rejectedTarget);
    expect(rejected.prob).toBeLessThan(1e-3);
    for (const r of [confident, rejected]) {
      const zMax = Math.max(...Array.from(r.occlusion, Math.abs));
      const pMax = Math.max(...Array.from(r.occlusionProb, Math.abs));
      expect(zMax, `target ${r.target}: erasing a patch moves the logit by a sizeable amount`).toBeGreaterThan(0.3);
      expect(pMax, `target ${r.target}: but the probability barely`).toBeLessThan(0.05 * zMax);
    }
  }, 30_000);
});

describe('attribution job', () => {
  it('validates its parameters', () => {
    const net = new Network(mnistArch(SOFTMAX), 1);
    expect(() => computeAttribution(net, { x: new Float32Array(10), target: 1 })).toThrow(/28×28/);
    expect(() => computeAttribution(net, { x: new Float32Array(784), target: 10 })).toThrow(/class/);
    const r = computeAttribution(net, { x: testDigit(0), target: 7, igSteps: 0, occlusion: { size: 99, stride: 0 } });
    expect(r.igSteps).toBe(1);
    expect(r.occlusionSize).toBe(28);
    expect(r.occlusionStride).toBe(1);
  });

  it('yields once per forward pass and reports exact progress', () => {
    const net = new Network(mnistArch(SMALL_CNN), 1);
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
    a.handle({ type: 'data', testX, testY, inputSize: 784, scale: 1 / 255, classes: 10 });
    const x = testDigit(0);
    const t0 = performance.now();
    a.handle({ type: 'run', id: 1, channel: 'attribution', kind: 'attribution', params: { x, target: 7, igSteps: 32, occlusion: { size: 6, stride: 2 } }, arch: mnistArch(spec), weights: source.getWeights() });
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

// ── Colour images (CIFAR-10 shape) ──────────────────────────────────────

const COLOUR: Arch['input'] = { c: 3, h: 32, w: 32 };
const colourArch = (layers: LayerSpec[]): Arch => ({ input: COLOUR, layers, classes: 10 });

/** A smooth random "photo": channel-major 3×32×32 values in [0, 1]; with `fill`, a 10×10 corner block of that colour. */
function photo(seed: number, fill: number[] = []): Float32Array {
  const rng = new Rng(seed);
  const x = new Float32Array(3 * 1024);
  const f = [rng.next() * 6, rng.next() * 6, rng.next() * 6];
  for (let c = 0; c < 3; c++) {
    for (let y = 0; y < 32; y++) {
      for (let k = 0; k < 32; k++) x[c * 1024 + y * 32 + k] = Math.min(1, Math.max(0, 0.5 + 0.35 * Math.sin((y + f[c]) / 4) * Math.cos((k - f[c]) / 5) + 0.08 * rng.normal()));
    }
    if (fill.length) for (let y = 22; y < 32; y++) for (let k = 22; k < 32; k++) x[c * 1024 + y * 32 + k] = fill[c];
  }
  return x;
}

describe('colour images', () => {
  const fill = [Math.fround(0.47), Math.fround(0.46), Math.fround(0.42)];

  it('adds the channels up: saliency as Σ|g_c|, gradient × input and integrated gradients as signed sums', () => {
    const net = new Network(colourArch(SMALL_CNN), 4);
    const x = photo(1);
    const r = computeAttribution(net, { x, target: 3, igSteps: 8, occlusion: { size: 6, stride: 2 }, fill });
    const { grad } = logitGradient(net, x, 3);
    expect(r.kind).toBe('image');
    expect(r.shape).toEqual(COLOUR);
    expect(r.saliency).toHaveLength(1024);
    expect(r.gradInput).toHaveLength(1024);
    expect(r.integrated).toHaveLength(1024);
    expect(r.occlusion).toHaveLength(1024);
    expect(r.channels!.integrated).toHaveLength(3072);
    expect(r.gradient).toHaveLength(0);
    for (const p of [0, 77, 500, 1023]) {
      let s = 0;
      let gx = 0;
      let ig = 0;
      for (let c = 0; c < 3; c++) {
        s += Math.abs(grad[c * 1024 + p]);
        gx += x[c * 1024 + p] * grad[c * 1024 + p];
        ig += r.channels!.integrated[c * 1024 + p];
        expect(r.channels!.saliency[c * 1024 + p]).toBeCloseTo(Math.abs(grad[c * 1024 + p]), 7);
      }
      expect(r.saliency[p]).toBeCloseTo(s, 6);
      expect(r.gradInput[p]).toBeCloseTo(gx, 6);
      expect(r.integrated[p]).toBeCloseTo(ig, 6);
    }
    // channel-major: R = [1, −2], G = [3, 4], B = [−5, 6]
    expect(Array.from(sumChannels([1, -2, 3, 4, -5, 6], 3))).toEqual([-1, 8]);
    expect(Array.from(sumChannels([1, -2, 3, 4, -5, 6], 3, true))).toEqual([9, 12]);
  });

  it('completeness holds over all three channels (Σ IG ≈ z(x) − z(black))', () => {
    const net = new Network(colourArch(LENET), 6);
    const x = photo(2);
    const r = computeAttribution(net, { x, target: 5, igSteps: 64, occlusion: { size: 8, stride: 8 }, fill });
    let sum = 0;
    for (const v of r.integrated) sum += v;
    expect(sum).toBeCloseTo(r.igSum, 4);
    expect(r.baseLogit).toBeCloseTo(targetLogit(net, new Float32Array(3072), 5), 5);
    expect(completenessGap(r.igSum, r.igExpected, r.igAbsSum).rel).toBeLessThan(0.01);
    expect(completenessText(r, 'black').expected).toMatch(/^z\(x\) − z\(black\) = /);
    expect(completenessText(r, 'black').title).toContain('from a black image to this one');
  });

  it('occlusion paints all three channels with the fill and matches a brute-force sweep', () => {
    const net = new Network(colourArch(SMALL_CNN), 8);
    const x = photo(3, fill);
    const size = 6;
    const stride = 4;
    const t = 2;
    const r = computeAttribution(net, { x, target: t, igSteps: 2, occlusion: { size, stride }, fill });
    expect(r.fill).toEqual(fill);
    const z0 = targetLogit(net, x, t);
    const patches = patchGrid(32, size, stride);
    const drops = patches.map((p) => {
      const e = x.slice();
      for (let c = 0; c < 3; c++) for (let y = p.y; y < Math.min(32, p.y + size); y++) for (let k = p.x; k < Math.min(32, p.x + size); k++) e[c * 1024 + y * 32 + k] = fill[c];
      return z0 - targetLogit(net, e, t);
    });
    const want = occlusionMap(patches, drops, size, 32, 32);
    for (let i = 0; i < 1024; i++) expect(r.occlusion[i]).toBeCloseTo(want[i], 4);
    // Patches inside the block that already has the fill colour change nothing: no forward pass.
    const filled = patches.filter((p) => patchIsFilled(x, p, size, COLOUR, fill)).length;
    expect(filled).toBeGreaterThan(0);
    expect(r.patchesEvaluated).toBe(patches.length - filled);
    expect(patchIsFilled(x, { y: 22, x: 22 }, 6, COLOUR, fill)).toBe(true);
    expect(patchIsFilled(x, { y: 22, x: 22 }, 6, COLOUR, [fill[0], fill[1], 0])).toBe(false);
  });

  it('the mean colour is the per-channel average over every stored image', () => {
    const shape = { c: 3, h: 2, w: 2 };
    // Two images, stored 0–255 as the dataset stores them.
    const X = Uint8Array.from([0, 0, 0, 0, 255, 255, 255, 255, 10, 20, 30, 40, 255, 255, 255, 255, 0, 0, 0, 0, 50, 50, 50, 50]);
    const m = meanColour(X, 2, shape, 1 / 255);
    expect(m[0]).toBeCloseTo((0 + 255 * 4) / 8 / 255, 9);
    expect(m[1]).toBeCloseTo((255 * 4 + 0) / 8 / 255, 9);
    expect(m[2]).toBeCloseTo((100 + 200) / 8 / 255, 9);
    expect(meanColour(X, 0, shape)).toEqual([0, 0, 0]);
  });

  it('the job paints occluded patches with the test set’s mean colour unless told otherwise', async () => {
    const net = new Network(colourArch(SMALL_CNN), 2);
    const n = 40;
    const testX = new Uint8Array(n * 3072);
    const rng = new Rng(5);
    for (let i = 0; i < testX.length; i++) testX[i] = Math.floor(rng.next() * 256);
    const testY = Uint8Array.from({ length: n }, (_, i) => i % 10);
    const log: FromAnalyzer[] = [];
    const a = new Analyzer((m) => log.push(m), jobs, 30);
    a.handle({ type: 'data', testX, testY, inputSize: 3072, scale: 1 / 255, classes: 10 });
    a.handle({ type: 'run', id: 1, channel: 'attribution', kind: 'attribution', params: { x: photo(4), target: 1, igSteps: 4 }, arch: colourArch(SMALL_CNN), weights: net.getWeights() });
    const t0 = performance.now();
    while (!log.some((m) => m.type === 'result' || m.type === 'error') && performance.now() - t0 < 20_000) await new Promise((r) => setTimeout(r, 2));
    const msg = log.find((m) => m.type === 'result' || m.type === 'error')!;
    expect(msg.type).toBe('result');
    const r = (msg as { result: AttributionResult }).result;
    const want = meanColour(testX, n, COLOUR, 1 / 255);
    for (let c = 0; c < 3; c++) expect(r.fill[c]).toBeCloseTo(want[c], 6);
    expect(r.occlusionSize).toBe(6);
    expect(r.patches).toBe(14 * 14);
  }, 30_000);

  it('the kink note names the input', () => {
    expect(kinkText({ relu: 4, pool: 0 }, 'image')).toBe('Where the input leaves ReLUs at exactly 0, brightening and darkening a pixel differ; the map shows the average slope.');
    expect(kinkText({ relu: 4, pool: 0 }, 'features')).toBe('Where the input leaves ReLUs at exactly 0, raising and lowering a feature differ; the slopes shown are averages.');
  });
});

// ── Point data: one value per input feature ────────────────────────────

describe('point features', () => {
  const ids2 = ['x1', 'x2', 'x1^2', 'x1*x2', 'sin x2'];
  const pointArch = (F: number, layers: LayerSpec[], classes = 2): Arch => ({ input: { c: F, h: 1, w: 1 }, layers, classes });
  const TANH: LayerSpec[] = [
    { kind: 'dense', units: 8, act: 'tanh' },
    { kind: 'dense', units: 8, act: 'tanh' },
  ];
  const logitAt = (net: Network, coords: number[], dims: 2 | 3, ids: string[], t: number) => targetLogit(net, featurize(Float32Array.from(coords), dims, ids), t);

  it('gives one value per feature, the signed gradient, and no occlusion', () => {
    const net = new Network(pointArch(5, TANH), 3);
    const coords = [0.3, -0.6];
    const x = featurize(Float32Array.from(coords), 2, ids2);
    const r = computeAttribution(net, { x, target: 1, igSteps: 32, point: { coords, dims: 2, features: ids2 } });
    expect(r.kind).toBe('features');
    expect(r.shape).toEqual({ c: 5, h: 1, w: 1 });
    for (const v of [r.saliency, r.gradInput, r.integrated, r.gradient]) expect(v).toHaveLength(5);
    expect(r.occlusion).toHaveLength(0);
    expect(r.patches).toBe(0);
    expect(r.channels).toBeUndefined();
    // the gradient matches central differences of the logit, feature by feature
    const h = 1e-3;
    for (let f = 0; f < 5; f++) {
      const up = x.slice();
      const down = x.slice();
      up[f] += h;
      down[f] -= h;
      const fd = (targetLogit(net, up, 1) - targetLogit(net, down, 1)) / (2 * h);
      expect(r.gradient[f]).toBeCloseTo(fd, 3);
      expect(r.saliency[f]).toBeCloseTo(Math.abs(r.gradient[f]), 7);
      expect(r.gradInput[f]).toBeCloseTo(x[f] * r.gradient[f], 7);
    }
    expect(() => computeAttribution(net, { x: new Float32Array(2), target: 0 })).toThrow('Attribution needs 5 feature values, got 2 values');
  });

  it('integrated gradients add up to z(x) − z(origin): every feature is 0 at the origin', () => {
    for (const ids of [ids2, ['x1', 'x2'], featureCatalog(2).map((f) => f.id)]) {
      const net = new Network(pointArch(ids.length, TANH), 7);
      const coords = [-0.8, 0.45];
      const x = featurize(Float32Array.from(coords), 2, ids);
      const r = computeAttribution(net, { x, target: 0, igSteps: 32 });
      expect(r.baseLogit).toBeCloseTo(logitAt(net, [0, 0], 2, ids, 0), 6);
      expect(Math.abs(r.igSum - r.igExpected), ids.join()).toBeLessThan(0.002 * Math.max(1, r.igAbsSum));
    }
    const text = completenessText({ igSum: 0.5, igExpected: 0.5, igAbsSum: 0.9 }, 'origin').title;
    expect(text).toContain('from the point at the origin, where every feature is 0, to this one');
    expect(text).toContain('all the feature attributions');
  });

  it('the gradient with respect to the coordinates follows the chain rule through the features (2-D and 3-D)', () => {
    const cases: [number[], 2 | 3, string[]][] = [
      [[0.3, -0.6], 2, ids2],
      [[-0.2, 0.7, 0.4], 3, featureCatalog(3).map((f) => f.id)],
      [[0.5, 0.1, -0.9], 3, ['x1', 'x3', 'x2*x3', 'sin x1']],
    ];
    for (const [coords, dims, ids] of cases) {
      const net = new Network(pointArch(ids.length, TANH, 3), 11);
      const x = featurize(Float32Array.from(coords), dims, ids);
      const r = computeAttribution(net, { x, target: 2, igSteps: 4, point: { coords, dims, features: ids } });
      expect(r.coordGrad).toHaveLength(dims);
      const h = 1e-3;
      for (let i = 0; i < dims; i++) {
        const up = coords.slice();
        const down = coords.slice();
        up[i] += h;
        down[i] -= h;
        const fd = (logitAt(net, up, dims, ids, 2) - logitAt(net, down, dims, ids, 2)) / (2 * h);
        expect(r.coordGrad![i], `${ids.join()} coordinate ${i + 1}`).toBeCloseTo(fd, 3);
      }
    }
    // With only the raw coordinates as features, it is the feature gradient itself.
    const g = coordinateGradient([2, -3], { coords: [0.1, 0.2], dims: 2, features: ['x1', 'x2'] });
    expect(g[0]).toBeCloseTo(2, 6);
    expect(g[1]).toBeCloseTo(-3, 6);
  });

  it('says in words which feature pushes toward the class and which pushes away', () => {
    const labels = ['x₁', 'x₂', 'x₁²'];
    expect(featureSentence(labels, [0.1, -0.31, 0.82], 'Class 1')).toBe('x₁² pushes toward Class 1 the most (+0.82). x₂ pushes away from it (−0.31).');
    expect(featureSentence(labels, [0.4, 0.0001, 0], 'Class 0')).toBe('x₁ pushes toward Class 0 the most (+0.4).');
    expect(featureSentence(labels, [-0.2, -0.5, 0], 'Class 2')).toBe('Every feature that matters pushes away from Class 2 here; x₂ the most (−0.5).');
    expect(featureSentence(labels, [0, 0, 0], 'Class 1')).toBe('No feature moves the score for Class 1 much at this point.');
    // a feature with under 5% of the total does not count
    expect(featureSentence(labels, [1, -0.04, 0], 'Class 1')).toBe('x₁ pushes toward Class 1 the most (+1).');
    // Measured from the origin: a predicted class can still score lower here than there.
    expect(featureSentence(labels, [-1.24, -3.3, 0], 'Class 0', -4.05)).toBe(
      'From the origin to this point, the score for Class 0 falls by 4.05. Every feature that matters pushes away from Class 0 here; x₂ the most (−3.3).',
    );
    expect(featureSentence(labels, [0.5, 0, 0], 'Class 1', 0.5)).toBe('From the origin to this point, the score for Class 1 rises by 0.5. x₁ pushes toward Class 1 the most (+0.5).');
    expect(featureSentence(labels, [0, 0, 0], 'Class 1', 0)).toBe('From the origin to this point, the score for Class 1 stays about the same. No feature moves the score for Class 1 much at this point.');
  });
});
