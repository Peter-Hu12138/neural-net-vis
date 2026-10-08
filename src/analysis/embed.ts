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
  /**
   * t-SNE on a layer wider than TSNE_MAX_DIM: 'pca' (default) keeps its TSNE_PCA_DIM main
   * directions; 'projection' uses the random sketch alone (faster, blurs neighbours more).
   */
  reduce?: 'pca' | 'projection';
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
  /** t-SNE only: dimensions it worked in (equals `dim` when no reduction was needed). */
  inputDim?: number;
  /** t-SNE only: how a wide layer was reduced to `inputDim` dimensions. */
  reduced?: 'pca' | 'projection';
  perplexity?: number;
  /**
   * Every digit gives (practically) the same values at this layer, so there is nothing to spread
   * out: the coordinates are all 0 and t-SNE is skipped. See isFlat.
   */
  flat?: boolean;
}

export const DEFAULT_N = 1000;
export const DEFAULT_PERPLEXITY = 30;
export const DEFAULT_ITERATIONS = 500;
/** t-SNE inputs wider than this are reduced first (see randomizedPca): a random sketch this wide… */
export const TSNE_MAX_DIM = 64;
/** …from which this many principal directions are kept. */
export const TSNE_PCA_DIM = 50;
/** Up to this many feature values in all (n × d), the collected rows stay in memory for the PCA
 * step; wider layers (unpooled conv maps) are read a second time instead of holding up to 50 MB. */
export const KEEP_LIMIT = 4_000_000;
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
 * Why t-SNE reduces wide layers first: exact t-SNE needs all n² pairwise distances, which costs
 * O(n²·d) and, for an unpooled conv layer (up to 12,544 values per digit), many seconds. A random
 * projection is the cheapest reduction that keeps distances: each projected squared distance is an
 * unbiased estimate of the original, within roughly ±25% for most pairs at k = 64
 * (Johnson–Lindenstrauss), and features are projected as they are collected. On its own it blurs
 * the nearest neighbours, so it serves as the sketch for randomizedPca, which keeps the main
 * directions instead. The seed keeps maps repeatable.
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
 * That sign rule is only a cold start: any rule flips when its input crosses 0, and the page keeps
 * a recomputed map facing the way the previous one did (alignPca). Among the rules tried, it flips
 * least often between recomputes 64 training digits apart (Small CNN, MLP and LeNet-ish, their
 * last two or three layers, 39 recomputes each): entry sum 16 flips, largest loading 32, sum of cubed
 * loadings 25, skewness of the projected digits 37, mean·component 23.
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

/**
 * A principal direction has no sign of its own: v and −v explain the same variance. pcaFit picks
 * one by a fixed rule, which a tiny weight change can tip over (on the logits layer the rule's
 * input is ~0 by construction, since softmax ignores the all-ones direction). Recomputing the map
 * after a training step would then mirror it. This orients each new component like the previous
 * component it matches (|cos| ≥ 0.5, so a PC1/PC2 swap keeps its orientation too), negating the
 * component and its coordinates in place. Components without a clear match keep pcaFit's sign.
 *
 * Returns, per component, whether it was flipped.
 */
export function alignPca(pca: PcaInfo, coords: Float32Array, prev: Float32Array | null): [boolean, boolean] {
  const flipped: [boolean, boolean] = [false, false];
  const c = pca.components;
  const d = c.length / 2;
  if (!prev || prev.length !== c.length) return flipped;
  for (let k = 0; k < 2; k++) {
    let best = 0;
    for (let m = 0; m < 2; m++) {
      let dot = 0;
      for (let j = 0; j < d; j++) dot += c[k * d + j] * prev[m * d + j];
      if (Math.abs(dot) > Math.abs(best)) best = dot;
    }
    if (best > -0.5) continue;
    flipped[k] = true;
    for (let j = 0; j < d; j++) c[k * d + j] = -c[k * d + j];
    for (let i = k; i < coords.length; i += 2) coords[i] = -coords[i];
  }
  return flipped;
}

/**
 * True when the rows hardly differ: the total variance is below 1e-12 of the mean squared value
 * (or both are 0), i.e. the differences are at the level of float32 rounding. This happens when
 * every unit of a layer is inactive for every digit (dead ReLUs, saturated tanh or sigmoid), and
 * then no projection has anything to show.
 */
export function isFlat(total: number, meanSquare: number): boolean {
  return !(total > 0 && total > 1e-12 * meanSquare);
}

/** Total variance and mean squared value of the n × d rows of X. */
export function spread(X: ArrayLike<number>, n: number, d: number): { total: number; meanSquare: number } {
  let total = 0;
  let sq = 0;
  for (let j = 0; j < d; j++) {
    let m = 0;
    for (let i = 0; i < n; i++) m += X[i * d + j];
    m /= n;
    let v = 0;
    for (let i = 0; i < n; i++) v += (X[i * d + j] - m) ** 2;
    total += v / n;
    sq += m * m;
  }
  return { total, meanSquare: sq + total };
}

/**
 * Eigen-decomposition of a small symmetric matrix (row-major l × l, copied) by cyclic Jacobi
 * rotations. Returns the eigenvalues in decreasing order and the eigenvectors as the columns of
 * `vectors` (row-major l × l, column c belongs to values[c]). Yields after each sweep.
 */
export function* jacobiEigen(A0: Float64Array, l: number): Generator<void, { values: Float64Array; vectors: Float64Array }, void> {
  const A = A0.slice();
  const V = new Float64Array(l * l);
  for (let i = 0; i < l; i++) V[i * l + i] = 1;
  let scale = 0;
  for (let i = 0; i < l * l; i++) scale += A[i] * A[i];
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let p = 0; p < l; p++) for (let q = p + 1; q < l; q++) off += A[p * l + q] ** 2;
    if (!(off > 1e-26 * scale)) break;
    for (let p = 0; p < l; p++) {
      for (let q = p + 1; q < l; q++) {
        const apq = A[p * l + q];
        if (apq === 0) continue;
        const theta = (A[q * l + q] - A[p * l + p]) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < l; k++) {
          const akp = A[k * l + p];
          const akq = A[k * l + q];
          A[k * l + p] = c * akp - s * akq;
          A[k * l + q] = s * akp + c * akq;
        }
        for (let k = 0; k < l; k++) {
          const apk = A[p * l + k];
          const aqk = A[q * l + k];
          A[p * l + k] = c * apk - s * aqk;
          A[q * l + k] = s * apk + c * aqk;
        }
        for (let k = 0; k < l; k++) {
          const vkp = V[k * l + p];
          const vkq = V[k * l + q];
          V[k * l + p] = c * vkp - s * vkq;
          V[k * l + q] = s * vkp + c * vkq;
        }
      }
    }
    yield;
  }
  const order = Array.from({ length: l }, (_, i) => i).sort((a, b) => A[b * l + b] - A[a * l + a]);
  const values = Float64Array.from(order, (i) => A[i * l + i]);
  const vectors = new Float64Array(l * l);
  for (let r = 0; r < l; r++) for (let c = 0; c < l; c++) vectors[r * l + c] = V[r * l + order[c]];
  return { values, vectors };
}

/**
 * Randomized PCA (Halko, Martinsson & Tropp, 2011; one sketch, no power iterations) of n rows of
 * width d, from their sketch Y = X·R (n × l, as collected with projectRow):
 *   Q = orth(Y − column means)   (n × l; its columns are centred, so QᵀX = QᵀXc)
 *   B = QᵀX                      (l × d, one more pass over the rows, which `row(s)` supplies)
 *   B·Bᵀ = Û·S²·Ûᵀ
 * and the result is the n × m matrix Q·Û[:, :m]·S[:m]: each row's coordinates along (an estimate
 * of) the top m principal directions. Unlike the sketch itself, this keeps the directions with
 * the most variance and drops the rest, so nearest neighbours survive much better: on 1,000 raw
 * pixel digits 5-NN label purity is 0.862 in 784-d, 0.795 after the 64-d sketch and 0.840 after
 * this (Conv 1 of a trained Small CNN: 0.900, 0.856, 0.874).
 *
 * `row(s)` is called once per row, in order. Yields progress 0–1.
 */
export function* randomizedPca(Y: ArrayLike<number>, n: number, l: number, d: number, m: number, row: (s: number) => ArrayLike<number>): Generator<number, Float32Array<ArrayBuffer>, void> {
  let t0 = now();
  const Q = Float64Array.from({ length: n * l }, (_, i) => Y[i]);
  for (let c = 0; c < l; c++) {
    let mean = 0;
    for (let i = 0; i < n; i++) mean += Q[i * l + c];
    mean /= n;
    for (let i = 0; i < n; i++) Q[i * l + c] -= mean;
  }
  // Modified Gram–Schmidt, twice for orthogonality; columns with nothing left become 0.
  let first = 0;
  for (let c = 0; c < l; c++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += Q[i * l + c] ** 2;
    first = Math.max(first, Math.sqrt(s));
  }
  for (let c = 0; c < l; c++) {
    for (let pass = 0; pass < 2; pass++) {
      for (let p = 0; p < c; p++) {
        let dot = 0;
        for (let i = 0; i < n; i++) dot += Q[i * l + c] * Q[i * l + p];
        for (let i = 0; i < n; i++) Q[i * l + c] -= dot * Q[i * l + p];
      }
    }
    let s = 0;
    for (let i = 0; i < n; i++) s += Q[i * l + c] ** 2;
    const norm = Math.sqrt(s);
    const inv = norm > 1e-9 * first ? 1 / norm : 0;
    for (let i = 0; i < n; i++) Q[i * l + c] *= inv;
    if (now() - t0 > SLICE_MS) {
      yield (0.1 * (c + 1)) / l;
      t0 = now();
    }
  }
  // Bᵀ (d × l), accumulated row by row: Bᵀ[j] += x_s[j] · Q[s].
  const Bt = new Float64Array(d * l);
  for (let s = 0; s < n; s++) {
    const f = row(s);
    const o = s * l;
    for (let j = 0; j < d; j++) {
      const v = f[j];
      if (v === 0) continue;
      const r = j * l;
      for (let c = 0; c < l; c++) Bt[r + c] += v * Q[o + c];
    }
    if (now() - t0 > SLICE_MS) {
      yield 0.1 + (0.75 * (s + 1)) / n;
      t0 = now();
    }
  }
  // G = B·Bᵀ (l × l).
  const G = new Float64Array(l * l);
  for (let j = 0; j < d; j++) {
    const r = j * l;
    for (let a = 0; a < l; a++) {
      const ba = Bt[r + a];
      if (ba === 0) continue;
      for (let b = a; b < l; b++) G[a * l + b] += ba * Bt[r + b];
    }
    if (now() - t0 > SLICE_MS) {
      yield 0.85 + (0.1 * (j + 1)) / d;
      t0 = now();
    }
  }
  for (let a = 0; a < l; a++) for (let b = 0; b < a; b++) G[a * l + b] = G[b * l + a];
  const ge = jacobiEigen(G, l);
  let e = ge.next();
  while (!e.done) {
    if (now() - t0 > SLICE_MS) {
      yield 0.95;
      t0 = now();
    }
    e = ge.next();
  }
  const { values, vectors } = e.value;
  const keep = Math.min(m, l);
  const out = new Float32Array(n * keep);
  for (let c = 0; c < keep; c++) {
    const sv = Math.sqrt(Math.max(0, values[c]));
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let a = 0; a < l; a++) s += Q[i * l + a] * vectors[a * l + c];
      out[i * keep + c] = s * sv;
    }
    if (now() - t0 > SLICE_MS) {
      yield 0.95 + (0.05 * (c + 1)) / keep;
      t0 = now();
    }
  }
  return out;
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
  /** Seed of the random start (used when `init` is 'random', or when the data has no spread). */
  seed?: number;
  /** Start layout: the data's first two principal directions (default) or Gaussian noise. */
  init?: 'pca' | 'random';
  /** Standard deviation of the start layout's first coordinate (default TSNE_INIT_STD). */
  initStd?: number;
}

/**
 * Spread of the start layout: its first coordinate has this standard deviation. Small enough that
 * every pair starts in the kernel's flat middle (distances ≪ 1), as with the usual 1e-4, but on
 * the Small CNN's Conv 1 (1,568 values, trained for 3,200 digits, n = 1,000) it gives the
 * exaggerated steps more to work with: 5-NN label purity after 100 steps 0.51 (two networks),
 * against 0.41/0.35 at 1e-4 and 0.36/0.42 from a random start; final KL 1.00/0.94 against
 * 1.01/1.05 from a random start.
 */
export const TSNE_INIT_STD = 1e-2;

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
 * step, ×0.8 otherwise, at least 0.01); the layout is recentred every step. It starts from the
 * first two principal components scaled to TSNE_INIT_STD (or Gaussian noise, see TsneOpts).
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

  // Start from the data's own two main directions, shrunk to a speck (as scikit-learn and openTSNE
  // do). A random speck carries no structure, and on wide inputs early exaggeration barely finds
  // any from it, so the first frames were a jittering cloud; from PCA they show the linear map
  // being refined. While the layout is this small every q is about 1/(n(n−1)), so the KL of the
  // frames stays near KL(P‖uniform) until exaggeration ends, whatever the arrangement.
  const Y = new Float64Array(2 * n);
  const initStd = opts.initStd ?? TSNE_INIT_STD;
  let started = false;
  if ((opts.init ?? 'pca') === 'pca') {
    const Xc = Float32Array.from({ length: n * k }, (_, i) => X[i]);
    const g = pcaFit(Xc, n, k, { maxIter: 100, tol: 1e-7 });
    let r = g.next();
    while (!r.done) {
      if (now() - t0 > SLICE_MS) {
        yield { frac: prepShare, preparing: true };
        t0 = now();
      }
      r = g.next();
    }
    const sd = Math.sqrt(r.value.variance[0]);
    if (sd > 0 && Number.isFinite(sd)) {
      for (let p = 0; p < 2 * n; p++) Y[p] = (r.value.coords[p] / sd) * initStd;
      started = true;
    }
  }
  if (!started) {
    const rng = new Rng(opts.seed ?? 0x75e1);
    for (let p = 0; p < 2 * n; p++) Y[p] = rng.normal() * initStd;
  }
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

// ── Axis labels ────────────────────────────────────────────────────────

/** Tick values with a 1–2–5 step that fall inside [lo, hi]. */
export function niceTicks(lo: number, hi: number, count: number): { ticks: number[]; step: number } {
  const span = hi - lo;
  if (!(span > 0) || !Number.isFinite(span)) return { ticks: [], step: 1 };
  const raw = span / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const e = raw / mag;
  const step = (e >= 7.5 ? 10 : e >= 3.5 ? 5 : e >= 1.5 ? 2 : 1) * mag;
  const ticks: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) ticks.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return { ticks, step };
}

/** A true minus sign, and never a negative zero ("-0.00" becomes "0.00"). */
const minus = (s: string): string => (/^-[0.]*$/.test(s) ? s.slice(1) : s.replace(/-/g, '−'));

/**
 * Axis label for tick `v` with spacing `step`: just enough decimals to tell neighbours apart (at
 * most 4), scientific notation for smaller steps (rather than printing every tick of a tiny span
 * as "0.0000"), and a plain "0" at zero.
 */
export function tickLabel(v: number, step: number): string {
  if (v === 0 || Math.abs(v) < Math.abs(step) * 1e-6) return '0';
  if (step >= 1) return minus(v.toFixed(0));
  if (step >= 1e-4) return minus(v.toFixed(Math.ceil(-Math.log10(step) - 1e-9)));
  const digits = Math.floor(Math.log10(Math.abs(v)) + 1e-9) - Math.floor(Math.log10(step) + 1e-9);
  return minus(v.toExponential(Math.max(0, Math.min(6, digits))).replace('e+', 'e'));
}

/**
 * A coordinate for tooltips: `d` decimals (scientific below 10^−d), a true minus sign, never a
 * negative zero.
 */
export function signedValue(v: number, d = 2): string {
  if (!Number.isFinite(v)) return '—';
  if (v !== 0 && Math.abs(v) < 10 ** -d) return minus(v.toExponential(1));
  return minus(v.toFixed(d));
}

// ── The job ────────────────────────────────────────────────────────────

const argmax = (a: ArrayLike<number>) => {
  let best = 0;
  for (let i = 1; i < a.length; i++) if (a[i] > a[best]) best = i;
  return best;
};

/**
 * Collects the layer's features for the balanced test digits (projected to `k` dims when R is
 * given; the full rows also go to `keep` when it is given), plus labels and predictions. Yields
 * progress in 0–1.
 */
function* collect(ctx: JobContext, layer: number, indices: Int32Array, d: number, k: number, R: Float32Array | null, keep: Float32Array | null = null) {
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
    if (keep) keep.set(f, s * d);
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
  const wide = method === 'tsne' && d > TSNE_MAX_DIM;
  const sketch = wide ? TSNE_MAX_DIM : d;
  const reduce = !wide ? null : params.reduce === 'projection' ? 'projection' : 'pca';
  const keep = reduce === 'pca' && n * d <= KEEP_LIMIT ? new Float32Array(n * d) : null;
  const share = method === 'pca' ? PHASES.pca.collect : PHASES.tsne.collect;
  // A wide t-SNE layer spends part of its share on the PCA step: little when the rows are kept,
  // half when they are read a second time.
  const collectShare = reduce !== 'pca' ? share : keep ? 0.85 * share : 0.5 * share;
  const report = (f: number, partial?: unknown): Progress => ({ done: Math.round(Math.min(1, f) * TOTAL), total: TOTAL, partial });
  let R: Float32Array | null = null;
  if (wide) {
    const gp = projectionChunks(d, sketch);
    let rp = gp.next();
    while (!rp.done) {
      yield report(0);
      rp = gp.next();
    }
    R = rp.value;
  }

  const gc = collect(ctx, layer, indices, d, sketch, R, keep);
  let c = gc.next();
  while (!c.done) {
    yield report(c.value * collectShare);
    c = gc.next();
  }
  const { labels, preds } = c.value;
  let X: Float32Array = c.value.X;

  if (method === 'pca') {
    const g = pcaFit(X, n, d);
    let r = g.next();
    while (!r.done) {
      yield report(share + (1 - share) * r.value);
      r = g.next();
    }
    const fit = r.value;
    let meanSquare = fit.total;
    for (const m of fit.mean) meanSquare += m * m;
    const flat = isFlat(fit.total, meanSquare);
    if (flat) fit.coords.fill(0);
    return {
      layer,
      method,
      indices,
      labels,
      preds,
      coords: fit.coords,
      pca: { mean: fit.mean, components: fit.components, explained: flat ? [0, 0] : fit.explained },
      iterations: fit.iterations[0] + fit.iterations[1],
      dim: d,
      flat,
    };
  }

  let k = sketch;
  const sp = spread(X, n, k);
  if (isFlat(sp.total, sp.meanSquare)) {
    const perplexity = Math.max(2, Math.min(params.perplexity ?? DEFAULT_PERPLEXITY, (n - 1) / 3));
    return { layer, method, indices, labels, preds, coords: new Float32Array(2 * n), kl: 0, iterations: 0, dim: d, inputDim: k, perplexity, flat: true };
  }

  if (reduce === 'pca') {
    const img = new Float32Array(784);
    const row = keep
      ? (s: number) => keep.subarray(s * d, (s + 1) * d)
      : (s: number) => {
          ctx.net.forward(ctx.image(indices[s], img));
          return layerFeatures(ctx.net, layer);
        };
    const gr = randomizedPca(X, n, sketch, d, TSNE_PCA_DIM, row);
    let rr = gr.next();
    while (!rr.done) {
      yield report(collectShare + (share - collectShare) * rr.value);
      rr = gr.next();
    }
    X = rr.value;
    k = Math.min(TSNE_PCA_DIM, sketch);
  }

  const g = tsneRun(X, n, k, { perplexity: params.perplexity, iterations: params.iterations });
  let r = g.next();
  while (!r.done) {
    yield report(share + (1 - share) * r.value.frac, r.value.partial);
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
    kl: fit.kl,
    iterations: fit.iterations,
    dim: d,
    inputDim: k,
    reduced: reduce ?? undefined,
    perplexity: fit.perplexity,
    flat: false,
  };
};

export const jobs: Record<string, Job> = { embed };
