import type { Network } from '../nn/network';
import { Rng } from '../nn/rng';
import type { Job, JobContext, Progress } from './protocol';

/**
 * Section 11: flattens what one layer "sees" for a balanced set of test digits to two dimensions,
 * with PCA (linear, axes in real units) or exact t-SNE (non-linear, keeps neighbours).
 *
 * Everything here is plain maths on typed arrays, so the tests can check it against brute force,
 * finite differences and known answers. The job at the bottom strings the pieces together and
 * yields after every few milliseconds of work.
 */

export type EmbedMethod = 'pca' | 'tsne';

export interface EmbedParams {
  /** −1 = raw pixels; otherwise a block index (its `out`: pooled maps, or activations). */
  layer: number;
  method: EmbedMethod;
  /** Number of test digits, split evenly over the ten classes. */
  n?: number;
  perplexity?: number;
  iterations?: number;
}

export interface PcaInfo {
  /** Mean feature vector (length d). */
  mean: Float32Array;
  /** The two principal directions, row-major 2 × d, unit length. */
  components: Float32Array;
  /** Share of the total variance along each direction. */
  explained: [number, number];
}

/** A t-SNE frame mid-optimisation. */
export interface TsnePartial {
  coords: Float32Array;
  /** Gradient steps completed. */
  iteration: number;
  /** KL(P‖Q) at that point (always against the true P, also during early exaggeration). */
  kl: number;
}

export interface EmbedResult {
  layer: number;
  method: EmbedMethod;
  /** Test-set indices of the points, ascending. */
  indices: Int32Array;
  labels: Uint8Array;
  /** The network's prediction for each point. */
  preds: Uint8Array;
  /** x0, y0, x1, y1, … */
  coords: Float32Array;
  pca?: PcaInfo;
  kl?: number;
  /** PCA: power iterations (both components); t-SNE: gradient steps. */
  iterations: number;
  /** Values per digit at the layer. */
  dim: number;
  /** t-SNE only: dimensions after the random projection (equals `dim` when none was needed). */
  inputDim?: number;
  perplexity?: number;
}

export const DEFAULT_N = 1000;
export const DEFAULT_PERPLEXITY = 30;
export const DEFAULT_ITERATIONS = 500;
/** t-SNE inputs wider than this are randomly projected down to it first. */
export const TSNE_MAX_DIM = 64;
export const EXAGGERATION = 12;
export const EXAGGERATION_ITERS = 100;
export const MOMENTUM_SWITCH = 250;
export const PCA_MAX_ITER = 300;
export const PARTIAL_EVERY = 10;

/** How the job's progress (done / total) splits into phases, so the page can name the phase. */
export const PHASES = {
  pca: { collect: 0.6 },
  tsne: { collect: 0.06, prepare: 0.16 },
} as const;
const TOTAL = 1000;

/** Work per yield, in milliseconds. The analyzer slices at ~30 ms, so this keeps it responsive. */
const SLICE_MS = 3;
const now = () => performance.now();

// ── Point selection and features ───────────────────────────────────────

/**
 * The first ⌊n/10⌋ test digits of each class, in ascending index order. Deterministic, balanced,
 * and the order interleaves classes, so no digit is always drawn on top of another.
 */
export function balancedIndices(labels: ArrayLike<number>, n: number): Int32Array {
  const per = Math.max(1, Math.floor(n / 10));
  const seen = new Array<number>(10).fill(0);
  const out: number[] = [];
  for (let i = 0; i < labels.length; i++) {
    const y = labels[i];
    if (seen[y] < per) {
      seen[y]++;
      out.push(i);
    }
  }
  return Int32Array.from(out);
}

/** Number of values per digit at `layer` (−1 = the 784 input pixels). */
export function layerDim(net: Network, layer: number): number {
  return layer < 0 ? 784 : net.blocks[layer].out.length;
}

/** The layer's values for the last forward pass (a live buffer: copy it). */
export function layerFeatures(net: Network, layer: number): Float32Array {
  return layer < 0 ? net.input : net.blocks[layer].out;
}

/** Projects one feature vector onto the two principal directions: ((f − mean)·c₁, (f − mean)·c₂). */
export function projectPca(f: ArrayLike<number>, pca: PcaInfo): [number, number] {
  const { mean, components } = pca;
  const d = mean.length;
  let a = 0;
  let b = 0;
  for (let j = 0; j < d; j++) {
    const v = f[j] - mean[j];
    a += v * components[j];
    b += v * components[d + j];
  }
  return [a, b];
}

/**
 * A d × k Gaussian random projection, entries N(0, 1/k), seeded.
 *
 * Why t-SNE reduces wide layers to 64 dimensions first: exact t-SNE needs all n² pairwise
 * distances, which costs O(n²·d) and, for an unpooled conv layer (up to 12,544 values per digit),
 * many seconds and 50 MB just to hold the features. A random projection is the cheapest reduction
 * that keeps distances: each projected squared distance is an unbiased estimate of the original,
 * within roughly ±25% for most pairs at k = 64 (Johnson–Lindenstrauss), and features are projected
 * as they are collected, so no n × d matrix is ever stored. It does blur the nearest neighbours a
 * little: on 1,000 raw-pixel digits, the share whose 5 nearest neighbours in the final map are
 * mostly the same digit drops from 0.86 (exact distances) to 0.81. The seed keeps maps repeatable.
 */
export function gaussianProjection(d: number, k: number, seed = 0x7a5e): Float32Array {
  const g = projectionChunks(d, k, seed);
  let r = g.next();
  while (!r.done) r = g.next();
  return r.value;
}

/** gaussianProjection, generated a slice at a time (wide layers need up to 800k samples). */
function* projectionChunks(d: number, k: number, seed = 0x7a5e): Generator<void, Float32Array, void> {
  const rng = new Rng(seed);
  const R = new Float32Array(d * k);
  const s = 1 / Math.sqrt(k);
  let t0 = now();
  for (let j = 0; j < d; j++) {
    for (let c = j * k; c < (j + 1) * k; c++) R[c] = rng.normal() * s;
    if (now() - t0 > SLICE_MS) {
      yield;
      t0 = now();
    }
  }
  return R;
}

/** out[o + c] = Σ_j f[j]·R[j, c]. Skips zero features (pixels and ReLU outputs are mostly zero). */
export function projectRow(f: ArrayLike<number>, R: Float32Array, d: number, k: number, out: Float32Array, o: number, acc = new Float64Array(k)): void {
  acc.fill(0);
  for (let j = 0; j < d; j++) {
    const v = f[j];
    if (v === 0) continue;
    const r = j * k;
    for (let c = 0; c < k; c++) acc[c] += v * R[r + c];
  }
  for (let c = 0; c < k; c++) out[o + c] = acc[c];
}

// ── PCA ────────────────────────────────────────────────────────────────

export interface PcaFit extends PcaInfo {
  /** Variance along each direction (eigenvalues of the covariance). */
  variance: [number, number];
  /** Total variance (trace of the covariance). */
  total: number;
  /** Power iterations used per component. */
  iterations: [number, number];
  coords: Float32Array;
}

/**
 * Top two principal components of the n × d rows of X by power iteration with deflation. The
 * covariance C = XcᵀXc / n is never formed: C·v = Σᵢ (xᵢ·v) xᵢ / n is one pass over the centred
 * rows. The second component iterates on C deflated by the first (v ← v − (u₁·v) u₁ each step,
 * which equals Hotelling's C − λ₁u₁u₁ᵀ restricted to u₁'s complement). Start vectors are seeded,
 * so results are reproducible; signs are fixed so each component's entries sum to ≥ 0.
 *
 * Centres X in place. Yields its progress (0–1) after every few milliseconds of work.
 */
export function* pcaFit(X: Float32Array, n: number, d: number, opts: { maxIter?: number; tol?: number } = {}): Generator<number, PcaFit, void> {
  const maxIter = opts.maxIter ?? PCA_MAX_ITER;
  const tol = opts.tol ?? 1e-10;
  const mean64 = new Float64Array(d);
  let t0 = now();
  for (let i = 0; i < n; i++) {
    const o = i * d;
    for (let j = 0; j < d; j++) mean64[j] += X[o + j];
    if (now() - t0 > SLICE_MS) {
      yield 0;
      t0 = now();
    }
  }
  for (let j = 0; j < d; j++) mean64[j] /= n;
  let total = 0;
  for (let i = 0; i < n; i++) {
    const o = i * d;
    for (let j = 0; j < d; j++) {
      const v = X[o + j] - mean64[j];
      X[o + j] = v;
      total += v * v;
    }
    if (now() - t0 > SLICE_MS) {
      yield 0;
      t0 = now();
    }
  }
  total /= n;

  const comps: Float64Array[] = [];
  const iters: [number, number] = [0, 0];
  const w = new Float64Array(d);
  const orthogonalise = (v: Float64Array) => {
    for (const u of comps) {
      let dot = 0;
      for (let j = 0; j < d; j++) dot += u[j] * v[j];
      for (let j = 0; j < d; j++) v[j] -= dot * u[j];
    }
  };
  const normalise = (v: Float64Array) => {
    let s = 0;
    for (let j = 0; j < d; j++) s += v[j] * v[j];
    const norm = Math.sqrt(s);
    if (norm > 0) for (let j = 0; j < d; j++) v[j] /= norm;
    return norm;
  };

  for (let k = 0; k < 2; k++) {
    const rng = new Rng(0x9ca0 + k);
    const v = new Float64Array(d);
    for (let j = 0; j < d; j++) v[j] = rng.normal();
    orthogonalise(v);
    normalise(v);
    let it = 0;
    for (; it < maxIter; it++) {
      // w = C·v, one pass over the rows: Σᵢ (xᵢ·v) xᵢ / n
      w.fill(0);
      for (let i = 0; i < n; i++) {
        const o = i * d;
        let s = 0;
        for (let j = 0; j < d; j++) s += X[o + j] * v[j];
        if (s !== 0) for (let j = 0; j < d; j++) w[j] += s * X[o + j];
        if (now() - t0 > SLICE_MS) {
          yield (k + (it + i / n) / maxIter) / 2;
          t0 = now();
        }
      }
      orthogonalise(w); // deflation: remove the directions already found
      if (normalise(w) === 0) break; // no variance left in this subspace
      let cos = 0;
      for (let j = 0; j < d; j++) cos += w[j] * v[j];
      v.set(w);
      if (1 - Math.abs(cos) < tol) {
        it++;
        break;
      }
    }
    iters[k] = it;
    let sum = 0;
    let big = 0;
    for (let j = 0; j < d; j++) {
      sum += v[j];
      if (Math.abs(v[j]) > Math.abs(big)) big = v[j];
    }
    if (sum < -1e-9 || (Math.abs(sum) <= 1e-9 && big < 0)) for (let j = 0; j < d; j++) v[j] = -v[j];
    comps.push(v);
  }

  const coords = new Float32Array(2 * n);
  const variance: [number, number] = [0, 0];
  for (let i = 0; i < n; i++) {
    const o = i * d;
    for (let k = 0; k < 2; k++) {
      const u = comps[k];
      let s = 0;
      for (let j = 0; j < d; j++) s += X[o + j] * u[j];
      coords[2 * i + k] = s;
      variance[k] += s * s;
    }
    if (now() - t0 > SLICE_MS) {
      yield 1;
      t0 = now();
    }
  }
  variance[0] /= n;
  variance[1] /= n;
  const components = new Float32Array(2 * d);
  components.set(comps[0], 0);
  components.set(comps[1], d);
  const explained: [number, number] = total > 0 ? [variance[0] / total, variance[1] / total] : [0, 0];
  return { mean: Float32Array.from(mean64), components, explained, variance, total, iterations: iters, coords };
}

// ── t-SNE (exact) ──────────────────────────────────────────────────────

/** Pairwise squared Euclidean distances of the n × k rows of X into the n × n matrix D. */
export function sqDistances(X: ArrayLike<number>, n: number, k: number, D: Float64Array, from = 0, to = n): void {
  for (let i = from; i < to; i++) {
    D[i * n + i] = 0;
    const oi = i * k;
    for (let j = i + 1; j < n; j++) {
      const oj = j * k;
      let s = 0;
      for (let c = 0; c < k; c++) {
        const t = X[oi + c] - X[oj + c];
        s += t * t;
      }
      D[i * n + j] = s;
      D[j * n + i] = s;
    }
  }
}

export interface RowFit {
  /** Gaussian precision β = 1 / (2σ²). */
  beta: number;
  /** Entropy of p(·|i) in nats; the target is log(perplexity). */
  entropy: number;
  iterations: number;
}

/**
 * Conditional probabilities p(j|i) ∝ exp(−β·D[i,j]) for row i, with β found by bisection so the
 * entropy matches log(perplexity) (within `tol`). Distances are shifted by the row minimum first,
 * which leaves p unchanged but keeps exp() from underflowing. Writes row i of P.
 */
export function calibrateRow(D: Float64Array, n: number, i: number, perplexity: number, P: Float64Array, tol = 1e-5, maxIter = 200): RowFit {
  const target = Math.log(perplexity);
  const off = i * n;
  let dmin = Infinity;
  let dsum = 0;
  for (let j = 0; j < n; j++) {
    if (j === i) continue;
    const v = D[off + j];
    if (v < dmin) dmin = v;
    dsum += v;
  }
  const spread = dsum / (n - 1) - dmin;
  let beta = spread > 0 ? 1 / spread : 1;
  let lo = 0;
  let hi = Infinity;
  let H = 0;
  let sP = 0;
  let it = 0;
  for (; it < maxIter; it++) {
    sP = 0;
    let sDP = 0;
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      const dd = D[off + j] - dmin;
      const p = Math.exp(-beta * dd);
      P[off + j] = p;
      sP += p;
      sDP += dd * p;
    }
    H = Math.log(sP) + (beta * sDP) / sP;
    const diff = H - target;
    if (Math.abs(diff) < tol) break;
    if (diff > 0) {
      // too flat: sharpen
      lo = beta;
      beta = hi === Infinity ? beta * 2 : (beta + hi) / 2;
    } else {
      hi = beta;
      beta = (beta + lo) / 2;
    }
  }
  P[off + i] = 0;
  for (let j = 0; j < n; j++) P[off + j] /= sP;
  return { beta, entropy: H, iterations: it };
}

/** P ← (P + Pᵀ) / 2n: joint probabilities that sum to 1. */
export function symmetrise(P: Float64Array, n: number): void {
  const s = 1 / (2 * n);
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const v = (P[i * n + j] + P[j * n + i]) * s;
      P[i * n + j] = v;
      P[j * n + i] = v;
    }
    P[i * n + i] = 0;
  }
}

/**
 * Student-t kernel (1 + |yᵢ − yⱼ|²)⁻¹ for every pair i < j, into the upper triangle of `num`.
 * Returns Z = Σ_{i≠j} kernel, so qᵢⱼ = kernel / Z.
 */
export function tsneKernel(Y: Float64Array, n: number, num: Float64Array): number {
  let Z = 0;
  for (let i = 0; i < n; i++) {
    const xi = Y[2 * i];
    const yi = Y[2 * i + 1];
    const row = i * n;
    for (let j = i + 1; j < n; j++) {
      const dx = xi - Y[2 * j];
      const dy = yi - Y[2 * j + 1];
      const q = 1 / (1 + dx * dx + dy * dy);
      num[row + j] = q;
      Z += q;
    }
  }
  return 2 * Z;
}

/**
 * Gradient of KL(αP‖Q) from the kernel: ∂C/∂yᵢ = 4 Σⱼ (α·pᵢⱼ − qᵢⱼ)(1 + |yᵢ − yⱼ|²)⁻¹(yᵢ − yⱼ).
 * α is the early-exaggeration factor (1 = the plain gradient).
 */
export function tsneGradFromKernel(P: Float64Array, Y: Float64Array, n: number, exaggeration: number, num: Float64Array, Z: number, grad: Float64Array): void {
  const invZ = 1 / Z;
  grad.fill(0);
  for (let i = 0; i < n; i++) {
    const xi = Y[2 * i];
    const yi = Y[2 * i + 1];
    const row = i * n;
    let gx = 0;
    let gy = 0;
    for (let j = i + 1; j < n; j++) {
      const q = num[row + j];
      const m = (exaggeration * P[row + j] - q * invZ) * q;
      const dx = m * (xi - Y[2 * j]);
      const dy = m * (yi - Y[2 * j + 1]);
      gx += dx;
      gy += dy;
      grad[2 * j] -= dx;
      grad[2 * j + 1] -= dy;
    }
    grad[2 * i] += gx;
    grad[2 * i + 1] += gy;
  }
  for (let p = 0; p < 2 * n; p++) grad[p] *= 4;
}

/** Kernel plus gradient in one call. Fills `num` and `grad`; returns Z. */
export function tsneGradient(P: Float64Array, Y: Float64Array, n: number, exaggeration: number, grad: Float64Array, num: Float64Array): number {
  const Z = tsneKernel(Y, n, num);
  tsneGradFromKernel(P, Y, n, exaggeration, num, Z, grad);
  return Z;
}

/**
 * KL(P‖Q) = Σ_{i≠j} pᵢⱼ log(pᵢⱼ / qᵢⱼ), from the kernel and Z left by tsneKernel. With `from`/`to`
 * it sums only the pairs whose smaller index lies in [from, to), so the pass can be split.
 */
export function klFromKernel(P: Float64Array, num: Float64Array, Z: number, n: number, from = 0, to = n): number {
  const logZ = Math.log(Z);
  let kl = 0;
  for (let i = from; i < to; i++) {
    const row = i * n;
    for (let j = i + 1; j < n; j++) {
      const p = P[row + j];
      if (p > 0) kl += p * (Math.log(p / num[row + j]) + logZ);
    }
  }
  return 2 * kl;
}

/** KL(P‖Q) for the layout Y (allocates; for tests and one-off use). */
export function tsneKL(P: Float64Array, Y: Float64Array, n: number): number {
  const num = new Float64Array(n * n);
  return klFromKernel(P, num, tsneKernel(Y, n, num), n);
}

export interface TsneOpts {
  perplexity?: number;
  iterations?: number;
  seed?: number;
}

export interface TsneStep {
  /** Progress within t-SNE, 0–1. */
  frac: number;
  /** True while distances and neighbour probabilities are computed, before the first step. */
  preparing: boolean;
  /** Latest frame; refreshed every PARTIAL_EVERY steps and carried by every later yield. */
  partial?: TsnePartial;
}

export interface TsneFit {
  coords: Float32Array;
  kl: number;
  iterations: number;
  perplexity: number;
}

/**
 * Exact O(n²) t-SNE of the n × k rows of X (van der Maaten & Hinton, 2008): Gaussian neighbour
 * probabilities matched to the perplexity, symmetrised; Student-t similarities in 2-D; early
 * exaggeration ×12 for 100 steps; momentum 0.5 then 0.8 from step 250; learning rate
 * max(n/12, 50) with per-coordinate gains (+0.2 when the gradient flips sign against the last
 * step, ×0.8 otherwise, at least 0.01); the layout is recentred every step.
 */
export function* tsneRun(X: ArrayLike<number>, n: number, k: number, opts: TsneOpts = {}): Generator<TsneStep, TsneFit, void> {
  const iterations = Math.max(1, Math.round(opts.iterations ?? DEFAULT_ITERATIONS));
  const perplexity = Math.max(2, Math.min(opts.perplexity ?? DEFAULT_PERPLEXITY, (n - 1) / 3));
  const prepShare = PHASES.tsne.prepare / (1 - PHASES.tsne.collect);

  // Distances, then calibrated conditional probabilities (the same buffer is reused as the kernel).
  const D = new Float64Array(n * n);
  const P = new Float64Array(n * n);
  let t0 = now();
  for (let i = 0; i < n; i++) {
    sqDistances(X, n, k, D, i, i + 1);
    if (now() - t0 > SLICE_MS) {
      yield { frac: (0.3 * prepShare * (i + 1)) / n, preparing: true };
      t0 = now();
    }
  }
  for (let i = 0; i < n; i++) {
    calibrateRow(D, n, i, perplexity, P);
    if (now() - t0 > SLICE_MS) {
      yield { frac: prepShare * (0.3 + (0.7 * (i + 1)) / n), preparing: true };
      t0 = now();
    }
  }
  symmetrise(P, n);
  const num = D; // distances are no longer needed

  const rng = new Rng(opts.seed ?? 0x75e1);
  const Y = new Float64Array(2 * n);
  for (let p = 0; p < 2 * n; p++) Y[p] = rng.normal() * 1e-4;
  const grad = new Float64Array(2 * n);
  const update = new Float64Array(2 * n);
  const gains = new Float64Array(2 * n).fill(1);
  const lr = Math.max(n / 12, 50);
  const half = Math.round(n * (1 - Math.SQRT1_2));
  let partial: TsnePartial | undefined;

  // Each step yields after the kernel pass and after the gradient pass, plus twice more when it
  // computes a frame, so no slice runs longer than one O(n²) pass.
  for (let it = 0; it < iterations; it++) {
    const exaggeration = it < EXAGGERATION_ITERS ? EXAGGERATION : 1;
    const momentum = it < MOMENTUM_SWITCH ? 0.5 : 0.8;
    const at = (f: number) => prepShare + ((1 - prepShare) * (it + f)) / iterations;
    const Z = tsneKernel(Y, n, num);
    yield { frac: at(0.4), preparing: false, partial };
    if (it % PARTIAL_EVERY === 0) {
      // A frame of the layout after `it` steps, with its KL (the kernel is already computed).
      // The KL pass is split where the upper triangle has half its pairs.
      let kl = klFromKernel(P, num, Z, n, 0, half);
      yield { frac: at(0.45), preparing: false, partial };
      kl += klFromKernel(P, num, Z, n, half, n);
      partial = { coords: Float32Array.from(Y), iteration: it, kl };
      yield { frac: at(0.5), preparing: false, partial };
    }
    tsneGradFromKernel(P, Y, n, exaggeration, num, Z, grad);
    for (let p = 0; p < 2 * n; p++) {
      const g = grad[p];
      const u = update[p];
      let gain = u * g < 0 ? gains[p] + 0.2 : gains[p] * 0.8;
      if (gain < 0.01) gain = 0.01;
      gains[p] = gain;
      const nu = momentum * u - lr * gain * g;
      update[p] = nu;
      Y[p] += nu;
    }
    let mx = 0;
    let my = 0;
    for (let i = 0; i < n; i++) {
      mx += Y[2 * i];
      my += Y[2 * i + 1];
    }
    mx /= n;
    my /= n;
    for (let i = 0; i < n; i++) {
      Y[2 * i] -= mx;
      Y[2 * i + 1] -= my;
    }
    yield { frac: at(1), preparing: false, partial };
  }

  const Z = tsneKernel(Y, n, num);
  return { coords: Float32Array.from(Y), kl: klFromKernel(P, num, Z, n), iterations, perplexity };
}

// ── The job ────────────────────────────────────────────────────────────

const argmax = (a: ArrayLike<number>) => {
  let best = 0;
  for (let i = 1; i < a.length; i++) if (a[i] > a[best]) best = i;
  return best;
};

/**
 * Collects the layer's features for the balanced test digits (projected to `k` dims when R is
 * given), plus labels and predictions. Yields progress in 0–1.
 */
function* collect(ctx: JobContext, layer: number, indices: Int32Array, d: number, k: number, R: Float32Array | null) {
  const n = indices.length;
  const X = new Float32Array(n * k);
  const labels = new Uint8Array(n);
  const preds = new Uint8Array(n);
  const img = new Float32Array(784);
  const acc = new Float64Array(k);
  let t0 = now();
  for (let s = 0; s < n; s++) {
    const i = indices[s];
    ctx.image(i, img);
    preds[s] = argmax(ctx.net.forward(img));
    labels[s] = ctx.testY[i];
    const f = layerFeatures(ctx.net, layer);
    if (R) projectRow(f, R, d, k, X, s * k, acc);
    else X.set(f, s * k);
    if (now() - t0 > SLICE_MS) {
      yield (s + 1) / n;
      t0 = now();
    }
  }
  return { X, labels, preds };
}

const embed: Job<EmbedParams, EmbedResult> = function* (ctx, params): Generator<Progress, EmbedResult, void> {
  const net = ctx.net;
  const layer = Math.max(-1, Math.min(net.blocks.length - 1, Math.round(params.layer ?? -1)));
  const method: EmbedMethod = params.method === 'tsne' ? 'tsne' : 'pca';
  const want = Math.max(10, Math.min(ctx.testY.length, Math.round(params.n ?? DEFAULT_N)));
  const indices = balancedIndices(ctx.testY, want);
  const n = indices.length;
  const d = layerDim(net, layer);
  const k = method === 'tsne' && d > TSNE_MAX_DIM ? TSNE_MAX_DIM : d;
  const share = method === 'pca' ? PHASES.pca.collect : PHASES.tsne.collect;
  const report = (f: number, partial?: unknown): Progress => ({ done: Math.round(Math.min(1, f) * TOTAL), total: TOTAL, partial });
  let R: Float32Array | null = null;
  if (k < d) {
    const gp = projectionChunks(d, k);
    let rp = gp.next();
    while (!rp.done) {
      yield report(0);
      rp = gp.next();
    }
    R = rp.value;
  }

  const gc = collect(ctx, layer, indices, d, k, R);
  let c = gc.next();
  while (!c.done) {
    yield report(c.value * share);
    c = gc.next();
  }
  const { X, labels, preds } = c.value;

  if (method === 'pca') {
    const g = pcaFit(X, n, d);
    let r = g.next();
    while (!r.done) {
      yield report(share + (1 - share) * r.value);
      r = g.next();
    }
    const fit = r.value;
    return {
      layer,
      method,
      indices,
      labels,
      preds,
      coords: fit.coords,
      pca: { mean: fit.mean, components: fit.components, explained: fit.explained },
      iterations: fit.iterations[0] + fit.iterations[1],
      dim: d,
    };
  }

  const g = tsneRun(X, n, k, { perplexity: params.perplexity, iterations: params.iterations });
  let r = g.next();
  while (!r.done) {
    yield report(share + (1 - share) * r.value.frac, r.value.partial);
    r = g.next();
  }
  const fit = r.value;
  return { layer, method, indices, labels, preds, coords: fit.coords, kl: fit.kl, iterations: fit.iterations, dim: d, inputDim: k, perplexity: fit.perplexity };
};

export const jobs: Record<string, Job> = { embed };
