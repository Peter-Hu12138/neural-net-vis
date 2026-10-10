import { readFileSync } from 'node:fs';
import { inflateSync } from 'node:zlib';
import { describe as suite, expect, it } from 'vitest';
import { mnistArch } from '../src/nn/types';
import {
  alignPca,
  balancedIndices,
  calibrateRow,
  DEFAULT_N,
  embedIndices,
  gaussianProjection,
  isFrame,
  REDUCING,
  isFlat,
  jobs,
  layerDim,
  layerFeatures,
  niceTicks,
  pcaFit,
  projectPca,
  projectRow,
  randomizedPca,
  jacobiEigen as eigenSmall,
  signedValue,
  sqDistances,
  symmetrise,
  tickLabel,
  TSNE_INIT_STD,
  TSNE_PCA_DIM,
  KEEP_LIMIT,
  tsneGradient,
  tsneKL,
  tsneRun,
  type EmbedResult,
  type TsnePartial,
} from '../src/analysis/embed';
import type { JobContext, Progress } from '../src/analysis/protocol';
import { Network } from '../src/nn/network';
import { Optimizer } from '../src/nn/optim';
import { Rng } from '../src/nn/rng';
import { PRESETS } from '../src/store';

const preset = (name: string) => structuredClone(PRESETS.find((p) => p.name === name)!.spec);

/** Digits from one bundled PNG sprite (8-bit grey, filter 0, 100 per row). */
function decodeSprite(file: string, count: number): Uint8Array {
  const buf = readFileSync(file);
  let off = 8;
  let width = 0;
  const idat: Buffer[] = [];
  while (off < buf.length) {
    const len = buf.readUInt32BE(off);
    const type = buf.toString('ascii', off + 4, off + 8);
    const data = buf.subarray(off + 8, off + 8 + len);
    if (type === 'IHDR') width = data.readUInt32BE(0);
    if (type === 'IDAT') idat.push(data);
    off += 12 + len;
  }
  const raw = inflateSync(Buffer.concat(idat));
  const X = new Uint8Array(count * 784);
  for (let i = 0; i < count; i++) {
    const ox = (i % 100) * 28;
    const oy = Math.floor(i / 100) * 28;
    for (let r = 0; r < 28; r++) for (let c = 0; c < 28; c++) X[i * 784 + r * 28 + c] = raw[(oy + r) * (width + 1) + 1 + ox + c];
  }
  return X;
}

const labelText = readFileSync('public/data/mnist-labels.txt', 'utf8').trim();

/** The bundled 2,000 MNIST test digits. */
function loadTest(): { testX: Uint8Array; testY: Uint8Array } {
  return { testX: decodeSprite('public/data/mnist-test.png', 2000), testY: Uint8Array.from(labelText.slice(20_000), (ch) => ch.charCodeAt(0) - 48) };
}

/** A network trained with Adam (lr 0.003, batches of 32) on the first 5,000 training digits. */
function trainer(name: string, seed: number) {
  const trainX = decodeSprite('public/data/mnist-train-0.png', 5000);
  const trainY = Uint8Array.from(labelText.slice(0, 5000), (ch) => ch.charCodeAt(0) - 48);
  const net = new Network(mnistArch(preset(name)), seed);
  const opt = new Optimizer(net, 'adam', 0.003);
  const x = new Float32Array(784);
  let pos = 0;
  return {
    net,
    train(batches: number) {
      for (let b = 0; b < batches; b++) {
        net.zeroGrad();
        for (let k = 0; k < 32; k++) {
          const i = pos++ % 5000;
          for (let j = 0; j < 784; j++) x[j] = trainX[i * 784 + j] / 255;
          net.forward(x);
          net.backward(trainY[i]);
        }
        opt.step(1 / 32);
      }
    },
  };
}

const data = loadTest();

function context(net: Network): JobContext {
  return {
    net,
    arch: net.arch,
    spec: net.spec,
    inputSize: 784,
    scale: 1 / 255,
    classes: 10,
    testX: data.testX,
    testY: data.testY,
    image(i, out = new Float32Array(784)) {
      for (let j = 0; j < 784; j++) out[j] = data.testX[i * 784 + j] / 255;
      return out;
    },
  };
}

/** Runs a generator to completion, as the analyzer would. */
function drain<Y, R>(gen: Generator<Y, R, void>): { result: R; reports: Y[]; ms: number } {
  const reports: Y[] = [];
  const t0 = performance.now();
  for (;;) {
    const r = gen.next();
    if (r.done) return { result: r.value, reports, ms: performance.now() - t0 };
    reports.push(r.value);
  }
}

/** Brute-force symmetric eigen-decomposition (cyclic Jacobi). Returns eigenpairs, largest first. */
function jacobiEigen(A0: number[][]): { values: number[]; vectors: number[][] } {
  const n = A0.length;
  const A = A0.map((r) => r.slice());
  const V: number[][] = A.map((_, i) => A.map((__, j) => (i === j ? 1 : 0)));
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p][q] ** 2;
    if (off < 1e-24) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(A[p][q]) < 1e-300) continue;
        const theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = A[k][p];
          const akq = A[k][q];
          A[k][p] = c * akp - s * akq;
          A[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p][k];
          const aqk = A[q][k];
          A[p][k] = c * apk - s * aqk;
          A[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k][p];
          const vkq = V[k][q];
          V[k][p] = c * vkp - s * vkq;
          V[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const order = A.map((r, i) => ({ v: r[i], i })).sort((a, b) => b.v - a.v);
  return { values: order.map((o) => o.v), vectors: order.map((o) => V.map((row) => row[o.i])) };
}

const cosine = (a: ArrayLike<number>, b: ArrayLike<number>, bOff = 0, d = a.length) => {
  let ab = 0;
  let aa = 0;
  let bb = 0;
  for (let j = 0; j < d; j++) {
    ab += a[j] * b[bOff + j];
    aa += a[j] * a[j];
    bb += b[bOff + j] * b[bOff + j];
  }
  return ab / Math.sqrt(aa * bb);
};

/** Share of points whose k nearest neighbours in the 2-D layout carry the same label (majority vote). */
function knnPurity(coords: ArrayLike<number>, labels: ArrayLike<number>, k = 5): number {
  const n = labels.length;
  let good = 0;
  for (let i = 0; i < n; i++) {
    const d: { j: number; d: number }[] = [];
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      d.push({ j, d: (coords[2 * i] - coords[2 * j]) ** 2 + (coords[2 * i + 1] - coords[2 * j + 1]) ** 2 });
    }
    d.sort((a, b) => a.d - b.d);
    const votes = new Array(10).fill(0);
    for (const e of d.slice(0, k)) votes[labels[e.j]]++;
    if (votes.indexOf(Math.max(...votes)) === labels[i]) good++;
  }
  return good / n;
}

/** The same purity measure in d dimensions. */
function knnPurityND(X: ArrayLike<number>, d: number, labels: ArrayLike<number>, k = 5): number {
  const n = labels.length;
  let good = 0;
  for (let i = 0; i < n; i++) {
    const ds: { j: number; d: number }[] = [];
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      let s = 0;
      for (let c = 0; c < d; c++) s += (X[i * d + c] - X[j * d + c]) ** 2;
      ds.push({ j, d: s });
    }
    ds.sort((a, b) => a.d - b.d);
    const votes = new Array(10).fill(0);
    for (const e of ds.slice(0, k)) votes[labels[e.j]]++;
    if (votes.indexOf(Math.max(...votes)) === labels[i]) good++;
  }
  return good / n;
}

suite('balancedIndices', () => {
  it('takes the first n/10 digits of each class, in index order', () => {
    const idx = balancedIndices(data.testY, 1000);
    expect(idx.length).toBe(1000);
    const counts = new Array(10).fill(0);
    for (const i of idx) counts[data.testY[i]]++;
    expect(counts).toEqual(new Array(10).fill(100));
    for (let s = 1; s < idx.length; s++) expect(idx[s]).toBeGreaterThan(idx[s - 1]);
    // Every skipped index belongs to a class that was already full.
    const seen = new Array(10).fill(0);
    for (let i = 0; i <= idx[idx.length - 1]; i++) {
      const y = data.testY[i];
      if (idx.includes(i)) seen[y]++;
      else expect(seen[y]).toBe(100);
    }
  });

  it('is deterministic', () => {
    expect(Array.from(balancedIndices(data.testY, 200))).toEqual(Array.from(balancedIndices(data.testY, 200)));
  });
});

suite('PCA by power iteration', () => {
  it('recovers a known dominant direction', () => {
    const n = 600;
    const d = 40;
    const rng = new Rng(3);
    const u = Float64Array.from({ length: d }, () => rng.normal());
    const norm = Math.hypot(...u);
    for (let j = 0; j < d; j++) u[j] /= norm;
    const X = new Float32Array(n * d);
    for (let i = 0; i < n; i++) {
      const a = 4 * rng.normal();
      for (let j = 0; j < d; j++) X[i * d + j] = 5 + a * u[j] + 0.5 * rng.normal();
    }
    const fit = drain(pcaFit(X, n, d)).result;
    expect(Math.abs(cosine(u, fit.components, 0, d))).toBeGreaterThan(0.99);
    // Variance along u is 16 + 0.25, total is 16 + 40·0.25 = 26 (up to sampling noise).
    expect(fit.explained[0]).toBeGreaterThan(0.5);
    expect(fit.explained[0]).toBeLessThan(0.75);
    // Components are unit length and orthogonal.
    expect(cosine(fit.components.subarray(0, d), fit.components, d, d)).toBeCloseTo(0, 5);
    let s = 0;
    for (let j = 0; j < d; j++) s += fit.components[j] ** 2;
    expect(s).toBeCloseTo(1, 5);
  });

  it('matches a brute-force eigen-decomposition of the covariance', () => {
    const n = 400;
    const d = 6;
    const rng = new Rng(11);
    // Correlated data with well-separated variances.
    const mix = Array.from({ length: d }, () => Array.from({ length: d }, () => rng.normal()));
    const scales = [3, 2, 1.2, 0.7, 0.4, 0.2];
    const X = new Float32Array(n * d);
    for (let i = 0; i < n; i++) {
      const z = scales.map((s) => s * rng.normal());
      for (let j = 0; j < d; j++) {
        let v = j - 2;
        for (let k = 0; k < d; k++) v += mix[j][k] * z[k];
        X[i * d + j] = v;
      }
    }
    const mean = new Array(d).fill(0);
    for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) mean[j] += X[i * d + j] / n;
    const C = Array.from({ length: d }, () => new Array(d).fill(0));
    for (let i = 0; i < n; i++) for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) C[a][b] += ((X[i * d + a] - mean[a]) * (X[i * d + b] - mean[b])) / n;
    const eig = jacobiEigen(C);
    const trace = eig.values.reduce((a, b) => a + b, 0);

    const fit = drain(pcaFit(X.slice(), n, d)).result;
    expect(fit.explained[0]).toBeCloseTo(eig.values[0] / trace, 5);
    expect(fit.explained[1]).toBeCloseTo(eig.values[1] / trace, 5);
    expect(fit.total).toBeCloseTo(trace, 4);
    expect(Math.abs(cosine(eig.vectors[0], fit.components, 0, d))).toBeGreaterThan(0.9999);
    expect(Math.abs(cosine(eig.vectors[1], fit.components, d, d))).toBeGreaterThan(0.9999);
    for (let j = 0; j < d; j++) expect(fit.mean[j]).toBeCloseTo(mean[j], 4);
    // projectPca reproduces the stored coordinates from the raw rows.
    for (const i of [0, 17, 399]) {
      const [a, b] = projectPca(X.subarray(i * d, (i + 1) * d), fit);
      expect(a).toBeCloseTo(fit.coords[2 * i], 4);
      expect(b).toBeCloseTo(fit.coords[2 * i + 1], 4);
    }
    expect(fit.iterations[0]).toBeLessThan(300);
  });
});

suite('t-SNE pieces', () => {
  it('calibrates every row to the target perplexity', () => {
    const n = 150;
    const k = 8;
    const rng = new Rng(5);
    const X = Float64Array.from({ length: n * k }, (_, i) => rng.normal() * (1 + (i % 3)) + (i % 7 === 0 ? 4 : 0));
    const D = new Float64Array(n * n);
    sqDistances(X, n, k, D);
    // Spot-check the distance matrix.
    let s = 0;
    for (let c = 0; c < k; c++) s += (X[3 * k + c] - X[40 * k + c]) ** 2;
    expect(D[3 * n + 40]).toBeCloseTo(s, 10);
    expect(D[40 * n + 3]).toBe(D[3 * n + 40]);
    for (const perp of [5, 30]) {
      const P = new Float64Array(n * n);
      for (let i = 0; i < n; i++) {
        const fit = calibrateRow(D, n, i, perp, P);
        expect(Math.abs(fit.entropy - Math.log(perp))).toBeLessThan(1e-4);
        // The row is a distribution with that entropy.
        let sum = 0;
        let H = 0;
        for (let j = 0; j < n; j++) {
          const p = P[i * n + j];
          sum += p;
          if (p > 0) H -= p * Math.log(p);
        }
        expect(P[i * n + i]).toBe(0);
        expect(sum).toBeCloseTo(1, 10);
        expect(Math.abs(H - Math.log(perp))).toBeLessThan(1e-4);
      }
      symmetrise(P, n);
      let total = 0;
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) total += P[i * n + j];
      expect(total).toBeCloseTo(1, 10);
      expect(P[7 * n + 90]).toBe(P[90 * n + 7]);
    }
  });

  it('gradient matches finite differences of KL(P‖Q)', () => {
    const n = 8;
    const rng = new Rng(9);
    const P = new Float64Array(n * n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (i !== j) P[i * n + j] = rng.next() + 0.05;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) P[i * n + j] = P[j * n + i] = (P[i * n + j] + P[j * n + i]) / 2;
    const sum = P.reduce((a, b) => a + b, 0);
    for (let i = 0; i < P.length; i++) P[i] /= sum;
    const Y = Float64Array.from({ length: 2 * n }, () => rng.normal() * 2);
    const grad = new Float64Array(2 * n);
    tsneGradient(P, Y, n, 1, grad, new Float64Array(n * n));
    const h = 1e-5;
    let maxErr = 0;
    let maxG = 0;
    for (let p = 0; p < 2 * n; p++) {
      const y0 = Y[p];
      Y[p] = y0 + h;
      const up = tsneKL(P, Y, n);
      Y[p] = y0 - h;
      const dn = tsneKL(P, Y, n);
      Y[p] = y0;
      const fd = (up - dn) / (2 * h);
      maxErr = Math.max(maxErr, Math.abs(fd - grad[p]));
      maxG = Math.max(maxG, Math.abs(grad[p]));
    }
    expect(maxG).toBeGreaterThan(1e-3);
    expect(maxErr / maxG).toBeLessThan(1e-6);
    expect(tsneKL(P, Y, n)).toBeGreaterThan(0);
  });

  it('separates three well-separated Gaussian clusters', () => {
    const per = 30;
    const n = 3 * per;
    const k = 10;
    const rng = new Rng(21);
    const X = new Float32Array(n * k);
    const labels = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      const c = Math.floor(i / per);
      labels[i] = c;
      for (let j = 0; j < k; j++) X[i * k + j] = (j === c ? 12 : 0) + rng.normal();
    }
    const run = drain(tsneRun(X, n, k, { perplexity: 10, iterations: 400 }));
    const fit = run.result;
    expect(fit.coords.length).toBe(2 * n);
    expect(fit.coords.every(Number.isFinite)).toBe(true);
    expect(knnPurity(fit.coords, labels)).toBeGreaterThan(0.95);
    // Frames arrive every 10 steps; KL falls once exaggeration ends.
    const frames = run.reports.map((r) => r.partial).filter((p): p is TsnePartial => !!p);
    const iters = [...new Set(frames.map((f) => f.iteration))];
    expect(iters.slice(0, 4)).toEqual([0, 10, 20, 30]);
    const at = (it: number) => frames.find((f) => f.iteration === it)!.kl;
    expect(fit.kl).toBeLessThan(at(100));
    expect(fit.kl).toBeLessThan(at(0));
    // Recentred every step.
    let mx = 0;
    for (let i = 0; i < n; i++) mx += fit.coords[2 * i];
    expect(Math.abs(mx / n)).toBeLessThan(1e-3);
  });

  it('random projection keeps pairwise distances on average', () => {
    const d = 784;
    const k = 64;
    const R = gaussianProjection(d, k);
    expect(Array.from(gaussianProjection(d, k).subarray(0, 5))).toEqual(Array.from(R.subarray(0, 5)));
    const imgs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map((i) => Float32Array.from(data.testX.subarray(i * 784, (i + 1) * 784), (v) => v / 255));
    const proj = new Float32Array(10 * k);
    imgs.forEach((x, i) => projectRow(x, R, d, k, proj, i * k));
    let ratio = 0;
    let m = 0;
    for (let a = 0; a < 10; a++) {
      for (let b = a + 1; b < 10; b++) {
        let full = 0;
        let low = 0;
        for (let j = 0; j < d; j++) full += (imgs[a][j] - imgs[b][j]) ** 2;
        for (let c = 0; c < k; c++) low += (proj[a * k + c] - proj[b * k + c]) ** 2;
        ratio += low / full;
        m++;
      }
    }
    expect(ratio / m).toBeGreaterThan(0.75);
    expect(ratio / m).toBeLessThan(1.25);
  });
});

suite('embed job on real digits', () => {
  it('PCA of the last hidden layer: balanced points, predictions and a projection that matches', () => {
    const net = new Network(mnistArch(preset('Small CNN')), 4);
    const layer = net.blocks.length - 2;
    const { result, reports } = drain(jobs.embed(context(net), { layer, method: 'pca', n: 300 }) as Generator<Progress, EmbedResult, void>);
    expect(result.method).toBe('pca');
    expect(result.layer).toBe(layer);
    expect(result.dim).toBe(32);
    expect(result.indices.length).toBe(300);
    expect(result.coords.length).toBe(600);
    expect(reports.length).toBeGreaterThan(0);
    expect(reports.every((r) => r.done <= r.total)).toBe(true);
    const counts = new Array(10).fill(0);
    for (const y of result.labels) counts[y]++;
    expect(counts).toEqual(new Array(10).fill(30));
    const pca = result.pca!;
    expect(pca.explained[0]).toBeGreaterThanOrEqual(pca.explained[1]);
    expect(pca.explained[0] + pca.explained[1]).toBeLessThanOrEqual(1 + 1e-6);
    // Re-project a point on a fresh copy of the network, as the page does.
    for (const s of [0, 123, 299]) {
      const i = result.indices[s];
      expect(result.labels[s]).toBe(data.testY[i]);
      const x = context(net).image(i);
      expect(result.preds[s]).toBe(net.predict(x));
      const [a, b] = projectPca(layerFeatures(net, layer), pca);
      expect(a).toBeCloseTo(result.coords[2 * s], 3);
      expect(b).toBeCloseTo(result.coords[2 * s + 1], 3);
    }
  });

  it('PCA of raw pixels finds the familiar first components', () => {
    const net = new Network(mnistArch([]), 1);
    const { result, ms } = drain(jobs.embed(context(net), { layer: -1, method: 'pca' }) as Generator<Progress, EmbedResult, void>);
    console.log(`PCA, pixels, n = ${result.indices.length}: ${ms.toFixed(0)} ms, ${result.iterations} power iterations`);
    expect(result.dim).toBe(784);
    expect(result.indices.length).toBe(DEFAULT_N);
    // On MNIST, PC1 and PC2 each explain roughly 6–11% of pixel variance.
    expect(result.pca!.explained[0]).toBeGreaterThan(0.06);
    expect(result.pca!.explained[0]).toBeLessThan(0.14);
    expect(result.pca!.explained[1]).toBeGreaterThan(0.04);
    expect(ms).toBeLessThan(10_000);
  });

  it('t-SNE of raw pixels keeps neighbours and streams frames', () => {
    const net = new Network(mnistArch([]), 1);
    const n = 400;
    const run = drain(jobs.embed(context(net), { layer: -1, method: 'tsne', n, iterations: 400 }) as Generator<Progress, EmbedResult, void>);
    const r = run.result;
    expect(r.method).toBe('tsne');
    expect(r.dim).toBe(784);
    expect(r.inputDim).toBe(50);
    expect(r.reduced).toBe('pca');
    expect(r.iterations).toBe(400);
    expect(r.perplexity).toBe(30);
    expect(r.coords.every(Number.isFinite)).toBe(true);
    expect(r.kl).toBeGreaterThan(0);
    const frames = run.reports.map((p) => p.partial).filter(isFrame);
    // Before the frames, the reduction to 50 directions says what it is doing.
    expect(run.reports.some((p) => p.partial === REDUCING)).toBe(true);
    const iters = [...new Set(frames.map((f) => f.iteration))];
    expect(iters.length).toBe(40);
    expect(iters[iters.length - 1]).toBe(390);
    expect(frames[0].coords.length).toBe(2 * n);
    expect(r.kl).toBeLessThan(frames.find((f) => f.iteration === 100)!.kl);
    // Progress is monotone and stays within the total.
    for (let i = 1; i < run.reports.length; i++) expect(run.reports[i].done).toBeGreaterThanOrEqual(run.reports[i - 1].done);
    expect(run.reports[run.reports.length - 1].done).toBe(run.reports[0].total);
    // 5-NN label purity: the 50 PCA directions t-SNE saw keep neighbours better than the 64-d
    // sketch they were found from, and the 2-D map keeps most of what the full 784-d space has.
    const full = new Float32Array(n * 784);
    r.indices.forEach((i, s) => context(net).image(i, full.subarray(s * 784, (s + 1) * 784)));
    const R = gaussianProjection(784, 64);
    const sketch = new Float32Array(n * 64);
    for (let s = 0; s < n; s++) projectRow(full.subarray(s * 784, (s + 1) * 784), R, 784, 64, sketch, s * 64);
    const reduced = drain(randomizedPca(sketch, n, 64, 784, 50, (s) => full.subarray(s * 784, (s + 1) * 784))).result;
    const pFull = knnPurityND(full, 784, r.labels);
    const pSketch = knnPurityND(sketch, 64, r.labels);
    const pReduced = knnPurityND(reduced, 50, r.labels);
    const map = knnPurity(r.coords, r.labels);
    console.log(`5-NN purity, ${n} pixel digits: 784-d ${pFull.toFixed(3)}, 64-d sketch ${pSketch.toFixed(3)}, 50 PCs ${pReduced.toFixed(3)}, t-SNE map ${map.toFixed(3)}`);
    expect(pFull).toBeGreaterThan(0.7);
    expect(pReduced).toBeGreaterThan(pSketch + 0.01);
    expect(map).toBeGreaterThan(0.9 * pFull);
  });

  it('runs the default t-SNE (n = 1000, 500 steps) on a conv layer in time', () => {
    const net = new Network(mnistArch(preset('Small CNN')), 2);
    const ctx = context(net);
    expect(layerDim(net, 0)).toBe(14 * 14 * 8);
    const run = drain(jobs.embed(ctx, { layer: 0, method: 'tsne' }) as Generator<Progress, EmbedResult, void>);
    const r = run.result;
    let longest = 0;
    // Time per yield, on a second run through the reading, PCA and preparation phases.
    const gen = jobs.embed(ctx, { layer: 0, method: 'tsne' });
    for (;;) {
      const t0 = performance.now();
      const step = gen.next();
      longest = Math.max(longest, performance.now() - t0);
      if (step.done || (step.value as Progress).partial) break;
    }
    console.log(`t-SNE, Conv 1 (1,568 values), n = ${r.indices.length}, ${r.iterations} steps: ${run.ms.toFixed(0)} ms; longest slice before the first frame ${longest.toFixed(1)} ms; KL ${r.kl!.toFixed(3)}`);
    expect(r.indices.length).toBe(1000);
    expect(r.inputDim).toBe(50);
    expect(r.coords.every(Number.isFinite)).toBe(true);
    expect(run.ms).toBeLessThan(20_000);
    expect(longest).toBeLessThan(60);
  }, 60_000);
});

suite('PCA orientation (alignPca)', () => {
  /** A fitted PCA of correlated random data, for the synthetic cases. */
  const sample = () => {
    const n = 200;
    const d = 5;
    const rng = new Rng(17);
    const X = new Float32Array(n * d);
    for (let i = 0; i < n; i++) {
      const a = 3 * rng.normal();
      const b = 1.5 * rng.normal();
      for (let j = 0; j < d; j++) X[i * d + j] = a * (j - 2) + b * (j % 2 ? 1 : -1) + 0.1 * rng.normal();
    }
    return { fit: drain(pcaFit(X, n, d)).result, d };
  };

  it('negates a component (and its coordinates) that came out mirrored', () => {
    const { fit, d } = sample();
    const prev = fit.components.slice();
    const coords = fit.coords.slice();
    // The same map, but PC1 mirrored: what a sign-rule flip after a training step looks like.
    for (let j = 0; j < d; j++) fit.components[j] = -fit.components[j];
    for (let i = 0; i < coords.length; i += 2) fit.coords[i] = -fit.coords[i];
    expect(alignPca(fit, fit.coords, prev)).toEqual([true, false]);
    expect(Array.from(fit.components)).toEqual(Array.from(prev));
    expect(Array.from(fit.coords)).toEqual(Array.from(coords));
    // Already aligned: nothing changes. No previous map: nothing changes either.
    expect(alignPca(fit, fit.coords, prev)).toEqual([false, false]);
    expect(alignPca(fit, fit.coords, null)).toEqual([false, false]);
    expect(alignPca(fit, fit.coords, new Float32Array(4))).toEqual([false, false]);
  });

  it('follows a component that swapped places with the other one', () => {
    const { fit, d } = sample();
    const prev = fit.components.slice();
    // New PC1 = old PC2, new PC2 = −old PC1 (the eigenvalues crossed and the sign rule flipped one).
    const c = fit.components;
    c.set(prev.subarray(d, 2 * d), 0);
    for (let j = 0; j < d; j++) c[d + j] = -prev[j];
    expect(alignPca(fit, fit.coords, prev)).toEqual([false, true]);
    for (let j = 0; j < d; j++) expect(c[d + j]).toBe(prev[j]);
  });

  it('keeps a recomputed map facing the same way through training steps', () => {
    // Small CNN, logits layer: softmax ignores the all-ones direction, so the components' entry
    // sums (pcaFit's cold-start sign rule) are ~0 and a single training step can tip them over.
    const t = trainer('Small CNN', 1);
    t.train(100);
    const layer = t.net.blocks.length - 1;
    const dot = (a: Float32Array, b: Float32Array, k: number, m: number, d: number) => {
      let s = 0;
      for (let j = 0; j < d; j++) s += a[k * d + j] * b[m * d + j];
      return s;
    };
    let rawPrev: Float32Array | null = null;
    let alignedPrev: Float32Array | null = null;
    let rawFlips = 0;
    let alignedFlips = 0;
    let c = 0;
    for (; c < 39 && rawFlips < 1; c++) {
      if (c > 0) t.train(2); // 64 more digits, as after two clicks on Step
      const r = drain(jobs.embed(context(t.net), { layer, method: 'pca' }) as Generator<Progress, EmbedResult, void>).result;
      const d = r.dim;
      const raw = r.pca!.components.slice();
      alignPca(r.pca!, r.coords, alignedPrev);
      for (let k = 0; k < 2; k++) {
        if (rawPrev && dot(raw, rawPrev, k, k, d) < -0.5) rawFlips++;
        if (alignedPrev && dot(r.pca!.components, alignedPrev, k, k, d) < -0.5) alignedFlips++;
      }
      // The coordinates still are the digits' projections on the (re-oriented) components.
      const s = 123;
      const net = t.net;
      net.forward(context(net).image(r.indices[s]));
      const [a, b] = projectPca(layerFeatures(net, layer), r.pca!);
      expect(a).toBeCloseTo(r.coords[2 * s], 3);
      expect(b).toBeCloseTo(r.coords[2 * s + 1], 3);
      rawPrev = raw;
      alignedPrev = r.pca!.components.slice();
    }
    console.log(`PCA of the logits, ${c} recomputes 64 digits apart: ${rawFlips} sign flips from pcaFit alone, ${alignedFlips} after alignPca`);
    expect(rawFlips).toBeGreaterThan(0); // the problem reproduces…
    expect(alignedFlips).toBe(0); // …and alignment removes it
  }, 120_000);
});

suite('flat layers', () => {
  it('a layer whose units are all inactive is reported as flat, not drawn as noise', () => {
    const net = new Network(mnistArch(preset('Small CNN')), 4);
    net.blocks[2].b.fill(-100); // Dense 3: every ReLU is off for every digit
    const pca = drain(jobs.embed(context(net), { layer: 2, method: 'pca', n: 200 }) as Generator<Progress, EmbedResult, void>).result;
    expect(pca.flat).toBe(true);
    expect(pca.pca!.explained).toEqual([0, 0]);
    expect(pca.coords.every((v) => v === 0)).toBe(true);
    const run = drain(jobs.embed(context(net), { layer: 2, method: 'tsne', n: 200 }) as Generator<Progress, EmbedResult, void>);
    expect(run.result.flat).toBe(true);
    expect(run.result.iterations).toBe(0);
    expect(run.result.coords.every((v) => v === 0)).toBe(true);
    expect(run.reports.some((r) => r.partial)).toBe(false);
    // The layer before it is alive.
    const live = drain(jobs.embed(context(net), { layer: 1, method: 'pca', n: 200 }) as Generator<Progress, EmbedResult, void>).result;
    expect(live.flat).toBe(false);
    expect(live.pca!.explained[0]).toBeGreaterThan(0);
  });

  it('isFlat: zero or rounding-level variance only', () => {
    expect(isFlat(0, 0)).toBe(true);
    expect(isFlat(0, 4)).toBe(true);
    expect(isFlat(1e-20, 1)).toBe(true);
    expect(isFlat(NaN, 1)).toBe(true);
    expect(isFlat(1e-6, 1)).toBe(false);
    expect(isFlat(1e-30, 1e-31)).toBe(false); // tiny but relatively large: a real spread
  });
});

suite('axis and tooltip numbers', () => {
  it('tick labels: true minus, plain zero, never −0.0000, distinct even on tiny spans', () => {
    expect(tickLabel(-10, 5)).toBe('−10');
    expect(tickLabel(0, 5)).toBe('0');
    expect(tickLabel(0.25, 0.05)).toBe('0.25');
    expect(tickLabel(-0.0002, 0.0001)).toBe('−0.0002');
    expect(tickLabel(-4e-5, 2e-5)).toBe('−4e−5');
    expect(tickLabel(1.2e-4, 2e-5)).toBe('1.2e−4');
    for (const span of [1e-13, 3e-9, 2e-5, 4e-4, 0.03, 7, 1234]) {
      for (const lo of [-span, -span / 3, 0]) {
        const { ticks, step } = niceTicks(lo, lo + 2 * span, 5);
        expect(ticks.length).toBeGreaterThan(2);
        const labels = ticks.map((t) => tickLabel(t, step));
        expect(new Set(labels).size, labels.join(' ')).toBe(labels.length);
        for (const l of labels) {
          expect(l).not.toMatch(/^[−-]0(\.0*)?$/);
          expect(l).not.toContain('-');
          expect(l.length).toBeLessThanOrEqual(8);
        }
      }
    }
  });

  it('tooltip coordinates use a true minus and never print −0.00', () => {
    expect(signedValue(-5.444)).toBe('−5.44');
    expect(signedValue(10.43)).toBe('10.43');
    expect(signedValue(-0)).toBe('0.00');
    expect(signedValue(-0.0049)).toBe('−4.9e−3');
    expect(signedValue(0)).toBe('0.00');
    expect(signedValue(Infinity)).toBe('—');
  });
});

suite('t-SNE start', () => {
  it('starts from the PCA layout scaled to TSNE_INIT_STD; random start still available', () => {
    const net = new Network(mnistArch([]), 1);
    const n = 300;
    const idx = balancedIndices(data.testY, n);
    const x = new Float32Array(784);
    const X = new Float32Array(n * 64);
    const R = gaussianProjection(784, 64);
    idx.forEach((i, s) => projectRow(context(net).image(i, x), R, 784, 64, X, s * 64));
    const frame0 = (init: 'pca' | 'random') =>
      drain(tsneRun(X, n, 64, { iterations: 1, init })).reports.map((r) => r.partial).find((p): p is TsnePartial => !!p && p.iteration === 0)!.coords;
    const pca = drain(pcaFit(X.slice(), n, 64)).result.coords;
    const corr = (a: Float32Array, b: Float32Array, k: number) => {
      const u = Float64Array.from({ length: n }, (_, i) => a[2 * i + k]);
      const v = Float64Array.from({ length: n }, (_, i) => b[2 * i + k]);
      return cosine(u, v);
    };
    const start = frame0('pca');
    for (const k of [0, 1]) expect(Math.abs(corr(start, pca, k))).toBeGreaterThan(0.999);
    let ss = 0;
    for (let i = 0; i < n; i++) ss += start[2 * i] ** 2;
    expect(Math.sqrt(ss / n)).toBeCloseTo(TSNE_INIT_STD, 6);
    const noise = frame0('random');
    expect(Math.abs(corr(noise, pca, 0))).toBeLessThan(0.2);
    // The PCA start already sorts the digits a little; noise does not.
    expect(knnPurity(start, Uint8Array.from(idx, (i) => data.testY[i]))).toBeGreaterThan(0.2);
    expect(knnPurity(noise, Uint8Array.from(idx, (i) => data.testY[i]))).toBeLessThan(0.2);
  });

  it('on a trained conv layer, clusters form during early exaggeration (purity after 100 steps)', () => {
    const t = trainer('Small CNN', 1);
    t.train(100);
    const n = 1000;
    const idx = balancedIndices(data.testY, n);
    const labels = Uint8Array.from(idx, (i) => data.testY[i]);
    const d = layerDim(t.net, 0);
    const R = gaussianProjection(d, 64);
    const X = new Float32Array(n * 64);
    idx.forEach((i, s) => {
      t.net.forward(context(t.net).image(i));
      projectRow(layerFeatures(t.net, 0), R, d, 64, X, s * 64);
    });
    const at100 = (init: 'pca' | 'random', initStd?: number) => {
      const g = tsneRun(X, n, 64, { init, initStd, iterations: 101 });
      for (let r = g.next(); !r.done; r = g.next()) if (r.value.partial?.iteration === 100) return knnPurity(r.value.partial.coords, labels);
      return 0;
    };
    const fromPca = at100('pca');
    const fromNoise = at100('random', 1e-4); // the previous start
    console.log(`t-SNE, Conv 1, 5-NN purity after 100 steps: PCA start ${fromPca.toFixed(3)}, random start (std 1e-4) ${fromNoise.toFixed(3)}`);
    expect(fromPca).toBeGreaterThan(0.45);
    expect(fromPca).toBeGreaterThan(fromNoise + 0.08);
  }, 120_000);
});

suite('randomized PCA (t-SNE reduction)', () => {
  it('jacobiEigen: A·v = λ·v, values in decreasing order', () => {
    const l = 12;
    const rng = new Rng(8);
    const A = new Float64Array(l * l);
    for (let a = 0; a < l; a++) for (let b = a; b < l; b++) A[a * l + b] = A[b * l + a] = rng.normal();
    const { values, vectors } = drain(eigenSmall(A, l)).result;
    for (let c = 1; c < l; c++) expect(values[c]).toBeLessThanOrEqual(values[c - 1]);
    for (let c = 0; c < l; c++) {
      for (let r = 0; r < l; r++) {
        let av = 0;
        for (let k = 0; k < l; k++) av += A[r * l + k] * vectors[k * l + c];
        expect(av).toBeCloseTo(values[c] * vectors[r * l + c], 9);
      }
    }
  });

  it('with a sketch as wide as the data it equals exact PCA', () => {
    const n = 300;
    const d = 24;
    const rng = new Rng(31);
    const mix = Array.from({ length: d }, () => Array.from({ length: d }, () => rng.normal()));
    const X = new Float32Array(n * d);
    for (let i = 0; i < n; i++) {
      const z = Array.from({ length: d }, (_, k) => rng.normal() * 3 * 0.75 ** k);
      for (let j = 0; j < d; j++) {
        let v = 1 + j;
        for (let k = 0; k < d; k++) v += mix[j][k] * z[k];
        X[i * d + j] = v;
      }
    }
    const R = gaussianProjection(d, d);
    const Y = new Float32Array(n * d);
    for (let s = 0; s < n; s++) projectRow(X.subarray(s * d, (s + 1) * d), R, d, d, Y, s * d);
    const F = drain(randomizedPca(Y, n, d, d, 5, (s) => X.subarray(s * d, (s + 1) * d))).result;
    expect(F.length).toBe(n * 5);
    const fit = drain(pcaFit(X.slice(), n, d)).result;
    for (const k of [0, 1]) {
      const a = Float64Array.from({ length: n }, (_, i) => F[i * 5 + k]);
      const b = Float64Array.from({ length: n }, (_, i) => fit.coords[2 * i + k]);
      expect(Math.abs(cosine(a, b))).toBeGreaterThan(0.9999);
      // Same scale: the column's variance is that component's eigenvalue.
      let v = 0;
      for (let i = 0; i < n; i++) v += a[i] * a[i];
      expect(v / n / fit.variance[k]).toBeCloseTo(1, 4);
    }
  });

  it('the job reads very wide layers a second time and gets the same result as from memory', () => {
    // Unpooled conv, 16 filters: 12,544 values per digit; 400 digits exceed KEEP_LIMIT, so the
    // job runs the network again for the PCA step instead of holding 20 MB of rows.
    const net = new Network(mnistArch([{ kind: 'conv', filters: 16, kernel: 3, act: 'relu', pool: false }]), 3);
    const n = 400;
    const d = layerDim(net, 0);
    expect(n * d).toBeGreaterThan(KEEP_LIMIT);
    const run = drain(jobs.embed(context(net), { layer: 0, method: 'tsne', n, iterations: 1 }) as Generator<Progress, EmbedResult, void>);
    expect(run.result.inputDim).toBe(TSNE_PCA_DIM);
    const start = run.reports.map((p) => p.partial as TsnePartial | undefined).find((p) => p?.iteration === 0)!.coords;
    // The same reduction from rows held in memory, then the same start layout.
    const idx = balancedIndices(data.testY, n);
    const rows = new Float32Array(n * d);
    const R = gaussianProjection(d, 64);
    const sketch = new Float32Array(n * 64);
    idx.forEach((i, s) => {
      net.forward(context(net).image(i));
      rows.set(layerFeatures(net, 0), s * d);
      projectRow(rows.subarray(s * d, (s + 1) * d), R, d, 64, sketch, s * 64);
    });
    const reduced = drain(randomizedPca(sketch, n, 64, d, TSNE_PCA_DIM, (s) => rows.subarray(s * d, (s + 1) * d))).result;
    const mine = drain(tsneRun(reduced, n, TSNE_PCA_DIM, { iterations: 1 })).reports.map((r) => r.partial).find((p) => p?.iteration === 0)!.coords;
    let err = 0;
    let scale = 0;
    for (let p = 0; p < 2 * n; p++) {
      err = Math.max(err, Math.abs(start[p] - mine[p]));
      scale = Math.max(scale, Math.abs(mine[p]));
    }
    expect(err / scale).toBeLessThan(1e-5);
  }, 60_000);

  it('one power iteration keeps nearest neighbours almost as well as exact PCA on a slowly decaying spectrum (MATH-1)', () => {
    // 600 rows of width 240 with a ReLU-like spectrum: the k-th direction has standard deviation
    // k^(−1/2), so the 50 kept directions hold under half of the variance and the 64-wide sketch
    // cannot tell them from the next ones without a power iteration.
    const n = 600;
    const d = 240;
    const m = TSNE_PCA_DIM;
    const rng = new Rng(77);
    const basis = Array.from({ length: d }, () => {
      const v = Float64Array.from({ length: d }, () => rng.normal());
      const s = Math.hypot(...v);
      return v.map((x) => x / s);
    });
    const X = new Float32Array(n * d);
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < d; k++) {
        const z = rng.normal() / Math.sqrt(k + 1);
        const v = basis[k];
        for (let j = 0; j < d; j++) X[i * d + j] += z * v[j];
      }
      for (let j = 0; j < d; j++) X[i * d + j] += 0.3; // an offset, so centring matters
    }
    const rowOf = (s: number) => X.subarray(s * d, (s + 1) * d);

    /** For every row, its k nearest neighbours (squared Euclidean) among the n rows of width w. */
    const neighbours = (A: ArrayLike<number>, w: number, k = 30) => {
      const D = new Float64Array(n * n);
      sqDistances(A, n, w, D);
      return Array.from({ length: n }, (_, i) => {
        const order = Array.from({ length: n }, (__, j) => j).filter((j) => j !== i);
        order.sort((a, b) => D[i * n + a] - D[i * n + b]);
        return new Set(order.slice(0, k));
      });
    };
    const truth = neighbours(X, d);
    const recall = (A: ArrayLike<number>, w: number) => {
      const nb = neighbours(A, w);
      let hit = 0;
      nb.forEach((set, i) => set.forEach((j) => (hit += truth[i].has(j) ? 1 : 0)));
      return hit / (30 * n);
    };

    // Exact PCA-m: eigenvectors of the covariance, then the centred rows' coordinates along them.
    const mean = new Float64Array(d);
    for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) mean[j] += X[i * d + j] / n;
    const C = new Float64Array(d * d);
    for (let i = 0; i < n; i++) {
      for (let a = 0; a < d; a++) {
        const va = X[i * d + a] - mean[a];
        for (let b = a; b < d; b++) C[a * d + b] += (va * (X[i * d + b] - mean[b])) / n;
      }
    }
    for (let a = 0; a < d; a++) for (let b = 0; b < a; b++) C[a * d + b] = C[b * d + a];
    const { vectors } = drain(eigenSmall(C, d)).result;
    const exact = new Float32Array(n * m);
    for (let i = 0; i < n; i++) {
      for (let c = 0; c < m; c++) {
        let s = 0;
        for (let j = 0; j < d; j++) s += (X[i * d + j] - mean[j]) * vectors[j * d + c];
        exact[i * m + c] = s;
      }
    }

    const R = gaussianProjection(d, 64);
    const Y = new Float32Array(n * 64);
    for (let s = 0; s < n; s++) projectRow(rowOf(s), R, d, 64, Y, s * 64);
    let calls = 0;
    const counted = (s: number) => {
      calls++;
      return rowOf(s);
    };
    const q0 = drain(randomizedPca(Y, n, 64, d, m, counted, 0));
    expect(calls).toBe(n);
    calls = 0;
    const q1 = drain(randomizedPca(Y, n, 64, d, m, counted));
    expect(calls).toBe(3 * n);
    // Progress only moves forward and ends at 1 or just under it.
    for (let k = 1; k < q1.reports.length; k++) expect(q1.reports[k]).toBeGreaterThanOrEqual(q1.reports[k - 1]);

    const rExact = recall(exact, m);
    const r0 = recall(q0.result, m);
    const r1 = recall(q1.result, m);
    console.log(`30-NN recall, slowly decaying spectrum: sketch only ${r0.toFixed(3)}, one power iteration ${r1.toFixed(3)}, exact PCA-${m} ${rExact.toFixed(3)}`);
    expect(r1).toBeGreaterThan(rExact - 0.03);
    expect(r1).toBeGreaterThan(r0 + 0.03);
  }, 60_000);
});

suite('which test samples a map shows (embedIndices)', () => {
  it('takes every test sample when there are no more than n, else a balanced set', () => {
    const small = Uint8Array.from({ length: 300 }, (_, i) => (i * 7) % 3);
    expect(Array.from(embedIndices(small, DEFAULT_N, 3))).toEqual(Array.from({ length: 300 }, (_, i) => i));
    // 301 samples of 3 unequal classes: balancedIndices alone would drop some.
    const uneven = Uint8Array.from({ length: 301 }, (_, i) => (i < 101 ? 0 : i < 201 ? 1 : 2));
    expect(embedIndices(uneven, DEFAULT_N, 3)).toHaveLength(301);
    expect(balancedIndices(uneven, 301, 3)).toHaveLength(300);
    expect(Array.from(embedIndices(data.testY, DEFAULT_N))).toEqual(Array.from(balancedIndices(data.testY, DEFAULT_N)));
  });

  it('the job maps all 300 test points of a two-class point dataset', () => {
    const arch = { input: { c: 2, h: 1, w: 1 }, layers: [{ kind: 'dense' as const, units: 8, act: 'tanh' as const }], classes: 2 };
    const net = new Network(arch, 4);
    const rng = new Rng(9);
    const testX = Float32Array.from({ length: 600 }, () => rng.next() * 2 - 1);
    const testY = Uint8Array.from({ length: 300 }, (_, i) => i % 2);
    const ctx: JobContext = {
      net,
      arch,
      spec: arch.layers,
      inputSize: 2,
      scale: 1,
      classes: 2,
      testX,
      testY,
      image(i, out = new Float32Array(2)) {
        out[0] = testX[2 * i];
        out[1] = testX[2 * i + 1];
        return out;
      },
    };
    for (const layer of [-1, 0, 1]) {
      const r = drain(jobs.embed(ctx, { layer, method: 'pca', n: DEFAULT_N }) as Generator<Progress, EmbedResult, void>).result;
      expect(r.indices).toHaveLength(300);
      expect(r.dim).toBe(layer < 0 ? 2 : layer === 0 ? 8 : 2);
      expect(r.flat).toBe(false);
    }
    const t = drain(jobs.embed(ctx, { layer: 0, method: 'tsne', n: DEFAULT_N, iterations: 20 }) as Generator<Progress, EmbedResult, void>).result;
    expect(t.indices).toHaveLength(300);
    expect(t.coords).toHaveLength(600);
    expect(t.inputDim).toBe(8);
  });
});
