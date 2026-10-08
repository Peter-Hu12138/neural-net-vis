import { readFileSync } from 'node:fs';
import { inflateSync } from 'node:zlib';
import { describe as suite, expect, it } from 'vitest';
import {
  balancedIndices,
  calibrateRow,
  DEFAULT_N,
  gaussianProjection,
  jobs,
  layerDim,
  layerFeatures,
  pcaFit,
  projectPca,
  projectRow,
  sqDistances,
  symmetrise,
  tsneGradient,
  tsneKL,
  tsneRun,
  type EmbedResult,
  type TsnePartial,
} from '../src/analysis/embed';
import type { JobContext, Progress } from '../src/analysis/protocol';
import { Network } from '../src/nn/network';
import { Rng } from '../src/nn/rng';
import { PRESETS } from '../src/store';

const preset = (name: string) => structuredClone(PRESETS.find((p) => p.name === name)!.spec);

/** The bundled 2,000 MNIST test digits, decoded from the PNG sprite (8-bit grey, filter 0). */
function loadTest(): { testX: Uint8Array; testY: Uint8Array } {
  const buf = readFileSync('public/data/mnist-test.png');
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
  const testX = new Uint8Array(2000 * 784);
  for (let i = 0; i < 2000; i++) {
    const ox = (i % 100) * 28;
    const oy = Math.floor(i / 100) * 28;
    for (let r = 0; r < 28; r++) for (let c = 0; c < 28; c++) testX[i * 784 + r * 28 + c] = raw[(oy + r) * (width + 1) + 1 + ox + c];
  }
  const labels = readFileSync('public/data/mnist-labels.txt', 'utf8').trim();
  const testY = Uint8Array.from(labels.slice(20_000), (ch) => ch.charCodeAt(0) - 48);
  return { testX, testY };
}

const data = loadTest();

function context(net: Network): JobContext {
  return {
    net,
    spec: net.spec,
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
    const net = new Network(preset('Small CNN'), 4);
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
    const net = new Network([], 1);
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
    const net = new Network([], 1);
    const n = 400;
    const run = drain(jobs.embed(context(net), { layer: -1, method: 'tsne', n, iterations: 400 }) as Generator<Progress, EmbedResult, void>);
    const r = run.result;
    expect(r.method).toBe('tsne');
    expect(r.dim).toBe(784);
    expect(r.inputDim).toBe(64);
    expect(r.iterations).toBe(400);
    expect(r.perplexity).toBe(30);
    expect(r.coords.every(Number.isFinite)).toBe(true);
    expect(r.kl).toBeGreaterThan(0);
    const frames = run.reports.map((p) => p.partial as TsnePartial | undefined).filter((p): p is TsnePartial => !!p);
    const iters = [...new Set(frames.map((f) => f.iteration))];
    expect(iters.length).toBe(40);
    expect(iters[iters.length - 1]).toBe(390);
    expect(frames[0].coords.length).toBe(2 * n);
    expect(r.kl).toBeLessThan(frames.find((f) => f.iteration === 100)!.kl);
    // Progress is monotone and stays within the total.
    for (let i = 1; i < run.reports.length; i++) expect(run.reports[i].done).toBeGreaterThanOrEqual(run.reports[i - 1].done);
    expect(run.reports[run.reports.length - 1].done).toBe(run.reports[0].total);
    // The map keeps neighbours: 5-NN label purity in 2-D is close to that of the 64-d input t-SNE saw.
    const R = gaussianProjection(784, 64);
    const P = new Float32Array(n * 64);
    const x = new Float32Array(784);
    r.indices.forEach((i, s) => projectRow(context(net).image(i, x), R, 784, 64, P, s * 64));
    const input = knnPurityND(P, 64, r.labels);
    const map = knnPurity(r.coords, r.labels);
    expect(input).toBeGreaterThan(0.7);
    expect(map).toBeGreaterThan(0.9 * input);
  });

  it('runs the default t-SNE (n = 1000, 500 steps) on a conv layer in time', () => {
    const net = new Network(preset('Small CNN'), 2);
    const ctx = context(net);
    expect(layerDim(net, 0)).toBe(14 * 14 * 8);
    const run = drain(jobs.embed(ctx, { layer: 0, method: 'tsne' }) as Generator<Progress, EmbedResult, void>);
    const r = run.result;
    let longest = 0;
    // Time per yield, on a second pass over the first part of the job.
    const gen = jobs.embed(ctx, { layer: 0, method: 'tsne' });
    for (let i = 0; i < 400; i++) {
      const t0 = performance.now();
      if (gen.next().done) break;
      longest = Math.max(longest, performance.now() - t0);
    }
    console.log(`t-SNE, Conv 1 (1,568 values), n = ${r.indices.length}, ${r.iterations} steps: ${run.ms.toFixed(0)} ms; longest slice ${longest.toFixed(1)} ms; KL ${r.kl!.toFixed(3)}`);
    expect(r.indices.length).toBe(1000);
    expect(r.inputDim).toBe(64);
    expect(r.coords.every(Number.isFinite)).toBe(true);
    expect(run.ms).toBeLessThan(20_000);
  }, 60_000);
});
