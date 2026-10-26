import { featureDefs } from '../data/features';
import type { Block, Network } from '../nn/network';
import { fmtShape, isImage, type Shape } from '../nn/types';
import type { Job, Progress } from './protocol';

/**
 * "Why this prediction": which parts of one input drive one class's score.
 *
 * With z_t the target's logit and g(x) = ∂z_t/∂x:
 * - saliency      |g(x)|
 * - gradInput     x ⊙ g(x)
 * - integrated    x ⊙ mean_k g(α_k·x), α_k = (k + ½)/m: integrated gradients from an all-zero
 *                 input (a blank or black image; for point data, every feature 0), midpoint Riemann
 *                 sum. Its total should match z_t(x) − z_t(0) (completeness). With `igTolerance`,
 *                 the steps triple (reusing every gradient already taken) until it does, to within
 *                 that share of z_t(x) − z_t(0), or until IG_MAX_STEPS.
 * - occlusion     images only: paint a size×size patch with a plain fill, record the drop in z_t;
 *                 each pixel gets the mean drop over the patches that cover it. The logit, like the
 *                 other three maps, so all four share units; the probability saturates near 0 and 1,
 *                 where erasing anything barely moves it. The drop in p_t is kept too, for tooltips.
 *
 * Colour images have three values per pixel. The maps add the channels up: gradient × input and
 * integrated gradients as the signed sum Σ_c, so a pixel's value is its share of the score change
 * and the whole map still adds up to z_t(x) − z_t(0); saliency as Σ_c |g_c|, the most the score can
 * change when each of the pixel's channels moves by one small step. The per-channel values are kept
 * for tooltips. Occlusion paints a patch in all three channels at once.
 *
 * Point data feeds the network a short feature vector (x₁, x₂, x₁², …): the same quantities come
 * out per feature, plus the signed gradient, and, given the raw coordinates, the gradient with
 * respect to them (which way to move the point to raise the score). Given the coordinates,
 * integrated gradients also follow the point itself along the straight line from the origin
 * (γ(α) = features(α·coords)), not the straight line in feature space, which leaves the plane as
 * soon as a feature is not a coordinate (halfway there, x₁² is a quarter of its value, not half):
 * feature f gets Σ_k g_f(γ(α_k)) · (γ_f(b_k+1) − γ_f(b_k)) over the steps [b_k, b_k+1], its share of
 * the change in the score along that path. With the raw coordinates only, the two paths coincide.
 *
 * The gradients come from Network.inputGradient in its symmetric mode: blank MNIST pixels leave
 * many units exactly on a kink (a ReLU at z = 0, a max-pool window of equal values), where the
 * slope differs on either side; there it takes the average. `kinks` counts those units.
 */

const SIDE = 28;

export interface OcclusionParams {
  size: number;
  stride: number;
}

export interface PointParams {
  /** Raw coordinates of the point (2 or 3 values). */
  coords: ArrayLike<number>;
  dims: 2 | 3;
  /** Feature ids the network input was computed from (see data/features). */
  features: string[];
}

export interface AttributionParams {
  x: Float32Array;
  target: number;
  /** Integrated gradients: the number of steps to start with. */
  igSteps?: number;
  /**
   * Integrated gradients: triple the steps (up to IG_MAX_STEPS) while Σ IG is further than this
   * share of |z(x) − z(0)| from it. Off when left out: exactly `igSteps` steps.
   */
  igTolerance?: number;
  occlusion?: OcclusionParams;
  /**
   * Images: the value an occluded patch is painted with, per channel. Defaults to 0 (blank) for
   * grey images and, in the analysis worker, to the test set's mean colour for colour images.
   */
  fill?: ArrayLike<number>;
  /** Point data: the point behind the features, for the gradient with respect to its coordinates. */
  point?: PointParams;
}

export interface AttributionResult {
  /** 'image': maps of H×W pixels; 'features': one value per input feature. */
  kind: 'image' | 'features';
  /** The network's input shape. */
  shape: Shape;
  target: number;
  /** Predicted class for x, and the full softmax output. */
  pred: number;
  probs: Float32Array;
  /** z_t(x) and p_t(x). */
  logit: number;
  prob: number;
  /** Per pixel (H·W, channels added up) or per feature. */
  saliency: Float32Array;
  gradInput: Float32Array;
  integrated: Float32Array;
  /** Colour images: the same three quantities per channel (C·H·W, channel-major). */
  channels?: { saliency: Float32Array; gradInput: Float32Array; integrated: Float32Array };
  /** Point data: the signed gradient ∂z_t/∂x per feature (empty for images). */
  gradient: Float32Array;
  /** Point data with `point` given: ∂z_t/∂coordinates. */
  coordGrad?: Float32Array;
  /**
   * The path integrated gradients followed: 'straight', every input scaled together from 0
   * (α·x); 'plane', point data with its coordinates, the point moving from the origin (features
   * of α·coords). The two are the same when the features are the raw coordinates.
   */
  igPath: 'straight' | 'plane';
  /** Σ integrated, and what it should equal: z_t(x) − z_t(0). */
  igSum: number;
  igExpected: number;
  /** Σ |integrated|: the total size of the attributions, the scale for the completeness gap. */
  igAbsSum: number;
  /** z_t of the all-zero input. */
  baseLogit: number;
  /** Images: mean drop in z_t when a covering patch is painted over (empty for features). */
  occlusion: Float32Array;
  /** The same for p_t. */
  occlusionProb: Float32Array;
  /** The value occluded patches were painted with, per channel. */
  fill: number[];
  /** Units sitting exactly on a kink for this input (see above). */
  kinks: Kinks;
  /** The settings actually used, after clamping (and, for igSteps, refinement). */
  igSteps: number;
  occlusionSize: number;
  occlusionStride: number;
  /** Number of patches tried, and how many needed a forward pass (the rest already had the fill). */
  patches: number;
  patchesEvaluated: number;
}

export interface Kinks {
  /** ReLU / leaky ReLU units with z exactly 0. */
  relu: number;
  /** Max-pool windows whose maximum is shared by two or more live entries. */
  pool: number;
}

export const DEFAULT_IG_STEPS = 32;
/** The most steps refinement goes to (32 tripled twice). */
export const IG_MAX_STEPS = 288;
/** The view's tolerance: refine until Σ IG is within 2% of the score change. */
export const IG_TOLERANCE = 0.02;
export const DEFAULT_OCCLUSION: OcclusionParams = { size: 6, stride: 2 };

export function oneHot(k: number, n = 10): Float32Array {
  const v = new Float32Array(n);
  v[k] = 1;
  return v;
}

/** Midpoints of m equal steps along [0, 1]. */
export function igAlphas(m: number): number[] {
  return Array.from({ length: m }, (_, k) => (k + 0.5) / m);
}

/** out = α·x */
export function scaleInto(x: ArrayLike<number>, alpha: number, out: Float32Array): Float32Array {
  for (let i = 0; i < out.length; i++) out[i] = alpha * x[i];
  return out;
}

/**
 * Runs x forward and returns the target's logit, its probability, all probabilities and a copy
 * of ∂z_t/∂x. Leaves the network holding the activations of x.
 */
export function logitGradient(net: Network, x: Float32Array, target: number): { logit: number; prob: number; probs: Float32Array; grad: Float32Array } {
  const probs = net.forward(x).slice();
  const last = net.blocks.length - 1;
  const logit = net.blocks[last].z[target];
  const grad = net.inputGradient(last, oneHot(target, net.classes)).slice();
  return { logit, prob: probs[target], probs, grad };
}

/** The target's logit for x (one forward pass). */
export function targetLogit(net: Network, x: Float32Array, target: number): number {
  net.forward(x);
  return net.blocks[net.blocks.length - 1].z[target];
}

export const saliencyOf = (g: ArrayLike<number>): Float32Array => Float32Array.from(g, Math.abs);

export function gradTimesInput(x: ArrayLike<number>, g: ArrayLike<number>): Float32Array {
  const out = new Float32Array(g.length);
  for (let i = 0; i < out.length; i++) out[i] = x[i] * g[i];
  return out;
}

/**
 * Adds the C channels of a channel-major C×P array up per position (out[p] = Σ_c v[c·P + p]),
 * or their absolute values with `abs`. With C = 1 it is a copy.
 */
export function sumChannels(v: ArrayLike<number>, C: number, abs = false): Float32Array {
  const P = v.length / C;
  const out = new Float32Array(P);
  for (let c = 0; c < C; c++) {
    const o = c * P;
    for (let p = 0; p < P; p++) out[p] += abs ? Math.abs(v[o + p]) : v[o + p];
  }
  return out;
}

/**
 * Patch origins along one axis of length n: every `stride` from 0, plus a last one flush with the
 * far edge so the border is covered too.
 */
export function patchOrigins(n: number, size: number, stride: number): number[] {
  const last = n - size;
  if (last <= 0) return [0];
  const out: number[] = [];
  for (let o = 0; o <= last; o += stride) out.push(o);
  if (out[out.length - 1] !== last) out.push(last);
  return out;
}

export interface Patch {
  y: number;
  x: number;
}

/** All size×size patches of an h×w image (square when w is left out), row by row. */
export function patchGrid(h: number, size: number, stride: number, w = h): Patch[] {
  const oy = patchOrigins(h, size, stride);
  const ox = patchOrigins(w, size, stride);
  const out: Patch[] = [];
  for (const y of oy) for (const x of ox) out.push({ y, x });
  return out;
}

/** True when every pixel of the patch is already 0, so erasing it changes nothing. */
export function patchIsBlank(img: ArrayLike<number>, p: Patch, size: number, side = SIDE): boolean {
  return patchIsFilled(img, p, size, { c: 1, h: side, w: side }, [0]);
}

/** True when every value of the patch, in every channel, already equals that channel's fill. */
export function patchIsFilled(img: ArrayLike<number>, p: Patch, size: number, shape: Shape, fill: ArrayLike<number>): boolean {
  const { c: C, h: H, w: W } = shape;
  const h = Math.min(size, H - p.y);
  const w = Math.min(size, W - p.x);
  for (let c = 0; c < C; c++) {
    const base = c * H * W;
    const f = fill[c] ?? 0;
    for (let r = 0; r < h; r++) {
      const o = base + (p.y + r) * W + p.x;
      for (let k = 0; k < w; k++) if (img[o + k] !== f) return false;
    }
  }
  return true;
}

/**
 * Per pixel of an h×w image (square when w is left out), the mean of `drops` over the patches
 * covering it; 0 where no patch reaches.
 */
export function occlusionMap(patches: Patch[], drops: ArrayLike<number>, size: number, h = SIDE, w = h): Float32Array {
  const sum = new Float64Array(h * w);
  const count = new Uint16Array(h * w);
  patches.forEach((p, k) => {
    const ph = Math.min(size, h - p.y);
    const pw = Math.min(size, w - p.x);
    for (let r = 0; r < ph; r++) {
      const o = (p.y + r) * w + p.x;
      for (let c = 0; c < pw; c++) {
        sum[o + c] += drops[k];
        count[o + c]++;
      }
    }
  });
  const out = new Float32Array(h * w);
  for (let i = 0; i < out.length; i++) out[i] = count[i] ? sum[i] / count[i] : 0;
  return out;
}

/**
 * Mean value of each of the C channels over n channel-major samples stored as `X` (values ×
 * `scale` = network input): the "average colour" occlusion paints with on colour images.
 */
export function meanColour(X: ArrayLike<number>, n: number, shape: Shape, scale = 1): number[] {
  const P = shape.h * shape.w;
  const size = shape.c * P;
  const out = new Array<number>(shape.c).fill(0);
  if (!n) return out;
  for (let c = 0; c < shape.c; c++) {
    let s = 0;
    for (let i = 0; i < n; i++) {
      const o = i * size + c * P;
      for (let p = 0; p < P; p++) s += X[o + p];
    }
    out[c] = (s * scale) / (n * P);
  }
  return out;
}

/**
 * ∂z/∂coordinates from ∂z/∂features by the chain rule: Σ_f g_f · ∂feature_f/∂coord_i. The feature
 * derivatives are central differences with a tiny step (exact to ~1e−9 for these smooth features).
 */
export function coordinateGradient(g: ArrayLike<number>, point: PointParams): Float32Array {
  const defs = featureDefs(point.dims, point.features);
  const out = new Float32Array(point.dims);
  const h = 1e-5;
  const c = Float64Array.from(point.coords);
  for (let i = 0; i < point.dims; i++) {
    const v = c[i];
    let s = 0;
    for (let f = 0; f < defs.length; f++) {
      c[i] = v + h;
      const up = defs[f].fn(c, 0);
      c[i] = v - h;
      const down = defs[f].fn(c, 0);
      s += g[f] * ((up - down) / (2 * h));
    }
    c[i] = v;
    out[i] = s;
  }
  return out;
}

/** γ(α): the network input at step α of a path from the all-zero input (α = 0) to x (α = 1). */
export type IgPath = (alpha: number, out: Float64Array) => Float64Array;

/**
 * Point data: the features along the straight line from the origin to the point, features(α·coords).
 * Null when that line does not run from the all-zero input to x (unknown features, or coordinates
 * that do not match x), so the caller falls back to the straight line in feature space.
 */
export function planePath(point: PointParams, x: ArrayLike<number>): IgPath | null {
  let defs: ReturnType<typeof featureDefs>;
  try {
    defs = featureDefs(point.dims, point.features);
  } catch {
    return null;
  }
  if (defs.length !== x.length || point.coords.length !== point.dims) return null;
  const coords = Float64Array.from(point.coords);
  const c = new Float64Array(point.dims);
  const path: IgPath = (alpha, out) => {
    for (let i = 0; i < c.length; i++) c[i] = alpha * coords[i];
    for (let f = 0; f < defs.length; f++) out[f] = defs[f].fn(c, 0);
    return out;
  };
  const start = path(0, new Float64Array(defs.length));
  const end = path(1, new Float64Array(defs.length));
  for (let f = 0; f < defs.length; f++) {
    if (start[f] !== 0) return null;
    if (Math.abs(end[f] - x[f]) > 1e-5 * Math.max(1, Math.abs(x[f]))) return null;
  }
  return path;
}

/** |a − b| relative to |b|; 0 when both are (near) zero. Blows up as b → 0: see completenessGap. */
export function relativeGap(a: number, b: number): number {
  const d = Math.abs(a - b);
  const s = Math.abs(b);
  if (s < 1e-9) return d < 1e-9 ? 0 : Infinity;
  return d / s;
}

export interface CompletenessGap {
  /** |Σ IG − (z(x) − z(blank))| */
  diff: number;
  /** diff relative to the score change |z(x) − z(blank)|; Infinity when the score does not change but the sum is off. */
  ofScore: number;
  /** diff relative to Σ|IG|, the total size of the attributions (0 when there are none). */
  ofIG: number;
  /**
   * Σ|IG| is at least CANCEL_RATIO × the score change: positive and negative attributions largely
   * cancel, so even a sum that is accurate next to the attributions can miss the small change by a
   * large share. The text then gives both shares.
   */
  cancels: boolean;
  /** Off by more than LARGE_GAP of the score change: the bars do not add up to it. */
  large: boolean;
}

export const CANCEL_RATIO = 10;
export const LARGE_GAP = 0.05;

/**
 * How far integrated gradients are from adding up to the score change, measured against that
 * change, which is what completeness promises, so a large miss always shows. On colour photos the
 * attributions cancel heavily (Σ|IG| is often 30 times the change); measured against Σ|IG|, a sum
 * off by half the change read "0.1%". Against Σ|IG| too, for context, when they cancel.
 */
export function completenessGap(igSum: number, igExpected: number, igAbsSum: number): CompletenessGap {
  const diff = Math.abs(igSum - igExpected);
  const score = Math.abs(igExpected);
  const ofScore = diff === 0 ? 0 : score > 0 ? diff / score : Infinity;
  const ofIG = igAbsSum > 0 ? diff / igAbsSum : 0;
  return { diff, ofScore, ofIG, cancels: igAbsSum >= CANCEL_RATIO * score, large: ofScore > LARGE_GAP };
}

/** Counts the units of the last forward pass that sit exactly on a kink (hidden blocks only). */
export function countKinks(net: Network): Kinks {
  let relu = 0;
  let pool = 0;
  for (let k = 0; k < net.blocks.length - 1; k++) {
    const b: Block = net.blocks[k];
    const act = b.spec.act;
    if (act === 'relu' || act === 'leaky') for (let i = 0; i < b.z.length; i++) if (b.z[i] === 0) relu++;
    if (b.kind !== 'conv' || !b.spec.pool) continue;
    const { c: F, h: H, w: W } = b.zShape;
    const { h: PH, w: PW } = b.outShape;
    const a = b.a;
    for (let f = 0; f < F; f++) {
      for (let py = 0; py < PH; py++) {
        for (let px = 0; px < PW; px++) {
          const i0 = f * H * W + 2 * py * W + 2 * px;
          const idx = [i0, i0 + 1, i0 + W, i0 + W + 1];
          let m = -Infinity;
          for (const j of idx) if (a[j] > m) m = a[j];
          let ties = 0;
          let live = false;
          for (const j of idx) {
            if (a[j] !== m) continue;
            ties++;
            // A ReLU tie of dead units (z < 0) has slope 0 on both sides: no kink.
            if (act !== 'relu' || b.z[j] >= 0) live = true;
          }
          if (ties > 1 && live) pool++;
        }
      }
    }
  }
  return { relu, pool };
}

const clampInt = (v: number | undefined, lo: number, hi: number, dflt: number) =>
  Number.isFinite(v) ? Math.max(lo, Math.min(hi, Math.round(v as number))) : dflt;

/** The attribution job. Yields after every forward pass (with or without a backward pass). */
export function* attribution(net: Network, params: AttributionParams): Generator<Progress, AttributionResult, void> {
  const x = params.x;
  const shape = net.arch.input;
  const image = isImage(shape);
  const n = net.inputSize;
  if (!x || x.length !== n) {
    const what = image ? `a ${fmtShape(shape)} image` : `${n} feature value${n === 1 ? '' : 's'}`;
    throw new Error(`Attribution needs ${what}, got ${x ? x.length : 0} values`);
  }
  const target = params.target;
  if (!Number.isInteger(target) || target < 0 || target >= net.classes) throw new Error(`Target must be a class 0–${net.classes - 1}, got ${target}`);
  const C = image ? shape.c : 1;
  const H = image ? shape.h : 1;
  const W = image ? shape.w : n;
  const P = H * W;
  const m = clampInt(params.igSteps, 1, 1024, DEFAULT_IG_STEPS);
  const size = image ? clampInt(params.occlusion?.size, 1, Math.max(H, W), DEFAULT_OCCLUSION.size) : 0;
  const stride = image ? clampInt(params.occlusion?.stride, 1, Math.max(H, W), DEFAULT_OCCLUSION.stride) : 0;
  const fill = Array.from({ length: C }, (_, c) => {
    const v = Number(params.fill?.[c] ?? 0);
    return image && Number.isFinite(v) ? Math.fround(v) : 0;
  });

  const patches = image ? patchGrid(H, size, stride, W) : [];
  const live = patches.map((p) => !patchIsFilled(x, p, size, { c: C, h: H, w: W }, fill));
  const evaluated = live.filter(Boolean).length;
  // Grows when integrated gradients need more steps (see below).
  let total = 2 + m + evaluated;
  let done = 0;
  const last = net.blocks.length - 1;
  const seed = oneHot(target, net.classes);

  // 1. The gradient at x.
  const at = logitGradient(net, x, target);
  const kinks = countKinks(net); // the network still holds the activations of x
  let pred = 0;
  for (let j = 1; j < at.probs.length; j++) if (at.probs[j] > at.probs[pred]) pred = j;
  const salC = saliencyOf(at.grad);
  const gxC = gradTimesInput(x, at.grad);
  yield { done: ++done, total };

  // 2. The all-zero baseline.
  const baseLogit = targetLogit(net, new Float32Array(n), target);
  yield { done: ++done, total };

  // 3. Integrated gradients from 0 to x: along α·x, or, for a point, along the line from the origin.
  const path = !image && params.point ? planePath(params.point, x) : null;
  const tol = Number(params.igTolerance) > 0 ? Number(params.igTolerance) : 0;
  const igExpected = at.logit - baseLogit;
  const xs = new Float32Array(n);
  const gp = new Float64Array(n);
  /** The gradient at step α of the path (a live buffer of the network). */
  const gradAt = (alpha: number): Float32Array => {
    if (path) {
      path(alpha, gp);
      for (let i = 0; i < n; i++) xs[i] = gp[i];
    } else scaleInto(x, alpha, xs);
    net.forward(xs);
    return net.inputGradient(last, seed);
  };
  // Straight path: every step weighs the same, so the sum of the gradients is enough. Plane path:
  // each feature changes by a different amount per step, so every step's gradient is kept.
  const acc = new Float64Array(n);
  let grads: Float32Array[] = [];
  const take = (g: Float32Array): Float32Array => {
    if (!path) {
      for (let i = 0; i < n; i++) acc[i] += g[i];
      return g;
    }
    return g.slice();
  };
  let M = m;
  for (const alpha of igAlphas(M)) {
    const g = take(gradAt(alpha));
    if (path) grads.push(g);
    yield { done: ++done, total };
  }
  /** IG for the M steps taken: x ⊙ mean g (straight), or Σ_k g_k ⊙ (γ(b_k+1) − γ(b_k)) (plane). */
  const integrate = (): Float64Array => {
    const ig = new Float64Array(n);
    if (!path) {
      for (let i = 0; i < n; i++) ig[i] = (x[i] * acc[i]) / M;
      return ig;
    }
    let lo = new Float64Array(n); // γ(0) = 0
    let hi = new Float64Array(n);
    for (let k = 0; k < M; k++) {
      if (k + 1 < M) path((k + 1) / M, hi);
      else for (let i = 0; i < n; i++) hi[i] = x[i]; // end exactly at x
      const g = grads[k];
      for (let i = 0; i < n; i++) ig[i] += g[i] * (hi[i] - lo[i]);
      [lo, hi] = [hi, lo];
    }
    return ig;
  };
  let ig = integrate();
  // Refinement: the midpoints of 3M steps include those of M, so tripling reuses every gradient.
  while (tol > 0 && 3 * M <= IG_MAX_STEPS) {
    let sum = 0;
    let abs = 0;
    for (let i = 0; i < n; i++) {
      sum += ig[i];
      abs += Math.abs(ig[i]);
    }
    // The floor keeps float32 rounding in the network from asking for steps that cannot help.
    if (Math.abs(sum - igExpected) <= Math.max(tol * Math.abs(igExpected), 1e-5 * (1 + abs))) break;
    total += 2 * M;
    const next: Float32Array[] = [];
    for (let k = 0; k < M; k++) {
      for (const j of [3 * k, 3 * k + 2]) {
        const g = take(gradAt((j + 0.5) / (3 * M)));
        if (path) next[j] = g;
        yield { done: ++done, total };
      }
      if (path) next[3 * k + 1] = grads[k];
    }
    grads = next;
    M *= 3;
    ig = integrate();
  }
  const igC = new Float32Array(n);
  let igSum = 0;
  for (let i = 0; i < n; i++) {
    igC[i] = ig[i];
    igSum += ig[i];
  }
  const integrated = sumChannels(igC, C);
  // Σ|IG| over what is shown: pixels (channels added up) or features.
  let igAbsSum = 0;
  for (let p = 0; p < integrated.length; p++) igAbsSum += Math.abs(integrated[p]);

  // 4. Occlusion (images): paint one patch at a time with the fill.
  const drops = new Float64Array(patches.length);
  const probDrops = new Float64Array(patches.length);
  if (image) {
    const work = x.slice();
    const saved = new Float32Array(C * size * size);
    for (let k = 0; k < patches.length; k++) {
      if (!live[k]) continue; // the patch already has the fill: the input, and so the score, is unchanged
      const p = patches[k];
      const h = Math.min(size, H - p.y);
      const w = Math.min(size, W - p.x);
      for (let c = 0; c < C; c++) {
        for (let r = 0; r < h; r++) {
          const o = c * P + (p.y + r) * W + p.x;
          const so = (c * size + r) * size;
          for (let q = 0; q < w; q++) {
            saved[so + q] = work[o + q];
            work[o + q] = fill[c];
          }
        }
      }
      probDrops[k] = at.prob - net.forward(work)[target];
      drops[k] = at.logit - net.blocks[last].z[target];
      for (let c = 0; c < C; c++) {
        for (let r = 0; r < h; r++) {
          const o = c * P + (p.y + r) * W + p.x;
          const so = (c * size + r) * size;
          for (let q = 0; q < w; q++) work[o + q] = saved[so + q];
        }
      }
      yield { done: ++done, total };
    }
  }
  const occlusion = image ? occlusionMap(patches, drops, size, H, W) : new Float32Array(0);
  const occlusionProb = image ? occlusionMap(patches, probDrops, size, H, W) : new Float32Array(0);

  return {
    kind: image ? 'image' : 'features',
    shape: { ...shape },
    target,
    pred,
    probs: at.probs,
    logit: at.logit,
    prob: at.prob,
    saliency: C > 1 ? sumChannels(salC, C, true) : salC,
    gradInput: C > 1 ? sumChannels(gxC, C) : gxC,
    integrated,
    channels: C > 1 ? { saliency: salC, gradInput: gxC, integrated: igC } : undefined,
    gradient: image ? new Float32Array(0) : at.grad,
    coordGrad: !image && params.point ? coordinateGradient(at.grad, params.point) : undefined,
    igPath: path ? 'plane' : 'straight',
    igSum,
    igExpected,
    igAbsSum,
    baseLogit,
    occlusion,
    occlusionProb,
    fill,
    kinks,
    igSteps: M,
    occlusionSize: size,
    occlusionStride: stride,
    patches: patches.length,
    patchesEvaluated: evaluated,
  };
}

/** Runs the whole job synchronously (tests, scripts). */
export function computeAttribution(net: Network, params: AttributionParams): AttributionResult {
  const gen = attribution(net, params);
  for (;;) {
    const r = gen.next();
    if (r.done) return r.value;
  }
}

// ── Text for the view (pure, so the unit tests can read it) ──

const MINUS = '−';

/**
 * Three significant digits, trailing zeros kept ("−3.30", so a column of values lines up and a
 * value reads the same in the table and in the sentence above it), a true minus sign (exponents
 * too), '+' on request; never "−0".
 */
export function sig(v: number, plus = false): string {
  if (!Number.isFinite(v)) return '–';
  if (v === 0) return '0';
  const a = Math.abs(v);
  const body = a >= 999.5 ? a.toFixed(0) : a >= 0.001 ? a.toPrecision(3) : a.toExponential(1).replace('e-', `e${MINUS}`);
  return (v < 0 ? MINUS : plus ? '+' : '') + body;
}

/** Six significant digits for tooltips, with a true minus sign. */
export function exact(v: number): string {
  if (!Number.isFinite(v)) return '–';
  if (v === 0) return '0';
  return String(Number(v.toPrecision(6))).replace(/-/g, MINUS);
}

/** "0.3%", "<0.1%", "12%" */
export function percentText(fraction: number): string {
  const p = fraction * 100;
  return `${p < 0.05 ? '<0.1' : p < 10 ? p.toFixed(1) : p.toFixed(0)}%`;
}

/**
 * The completeness line under the integrated-gradients map, in three clauses the view may wrap
 * between: "Σ IG = 8.39;" "z(x) − z(blank) = 8.43" "(off by 0.0389, 0.5% of the score change)".
 * The gap is always given as a share of the score change z(x) − z(blank), which is what the
 * attributions should add up to; when they cancel heavily, as a share of Σ|IG| too ("(off by
 * 0.106: 53% of the score change, 0.1% of Σ|IG|)"). `base` names the all-zero input: "blank" for
 * grey images, "black" for colour ones, "origin" for point features. `large` is true when the sum
 * misses by more than LARGE_GAP of the change.
 */
export function completenessText(
  r: Pick<AttributionResult, 'igSum' | 'igExpected' | 'igAbsSum'> & { igSteps?: number },
  base = 'blank',
): { sum: string; expected: string; gap: string; title: string; large: boolean } {
  const g = completenessGap(r.igSum, r.igExpected, r.igAbsSum);
  const what = base === 'origin' ? 'the point at the origin, where every feature is 0,' : `a ${base} image`;
  const parts = base === 'origin' ? 'feature' : 'pixel';
  const change = 'the score change';
  let gap: string;
  if (g.diff === 0) gap = '(a match)';
  else if (!Number.isFinite(g.ofScore)) gap = `(off by ${sig(g.diff)}, ${percentText(g.ofIG)} of Σ|IG|; the score does not change)`;
  else if (g.cancels) gap = `(off by ${sig(g.diff)}: ${percentText(g.ofScore)} of ${change}, ${percentText(g.ofIG)} of Σ|IG|)`;
  else gap = `(off by ${sig(g.diff)}, ${percentText(g.ofScore)} of ${change})`;
  const steps = r.igSteps ? ` from ${r.igSteps} steps` : '';
  const why = g.cancels
    ? ` Here the positive and negative attributions largely cancel: Σ|IG| is ${sig(r.igAbsSum / Math.max(Math.abs(r.igExpected), 1e-12))} times the score change, so even a sum that is accurate next to the attributions (${percentText(g.ofIG)} of Σ|IG|) can miss the small change by a larger share.`
    : '';
  return {
    sum: `Σ IG = ${sig(r.igSum)};`,
    expected: `z(x) ${MINUS} z(${base}) = ${sig(r.igExpected)}`,
    gap,
    large: g.large,
    title:
      `Completeness: integrated gradients should add up to the change in the score from ${what} to this one. ` +
      `Σ IG = ${exact(r.igSum)}${steps}, z(x) ${MINUS} z(${base}) = ${exact(r.igExpected)}, Σ|IG| = ${exact(r.igAbsSum)} (the total size of all the ${parts} attributions). ` +
      `The gap is measured against the score change.${why}`,
  };
}

/**
 * Saliency hint addition when the input leaves units on a kink; '' when there are none. `where`
 * says what the input is: a grey image on a blank background (the hint then says that is where
 * the kinks mostly are), a colour image, or a point's features.
 */
export function kinkText(k: Kinks, where: 'blank' | 'image' | 'features' = 'blank'): string {
  const kinds = [k.relu ? 'ReLUs at exactly 0' : '', k.pool ? 'tied max-pool windows' : ''].filter(Boolean);
  if (!kinds.length) return '';
  const bg = where === 'blank' ? ' (mostly the blank background)' : '';
  const move = where === 'features' ? 'raising and lowering a feature' : 'brightening and darkening a pixel';
  return `Where the input leaves ${kinds.join(' and ')}${bg}, ${move} differ; ${where === 'features' ? 'the slopes shown are averages' : 'the map shows the average slope'}.`;
}

/**
 * The integrated gradients of a point's features in words: which feature pushes the score for
 * `target` up the most, and which pushes it down. A feature counts when its share of Σ|IG| is at
 * least 5%. Values carry their sign (+0.82, −0.31).
 *
 * Integrated gradients measure from the origin, so with `change` (z(x) − z(origin)) the sentence
 * starts there: a point deep inside a class can still score lower than the origin does, and then
 * every feature lowers the score on the way although the class is predicted. With `predicted`
 * (the target is the prediction here) the sentence says so, so it does not read as the features
 * arguing against the prediction.
 */
export function featureSentence(labels: string[], ig: ArrayLike<number>, target: string, change?: number, predicted = false): string {
  let total = 0;
  for (let f = 0; f < labels.length; f++) total += Math.abs(ig[f]);
  const order = labels.map((_, f) => f).sort((a, b) => Math.abs(ig[b]) - Math.abs(ig[a]));
  const counts = (f: number) => total > 1e-6 && Math.abs(ig[f]) >= 0.05 * total && Math.abs(ig[f]) >= 1e-4;
  const up = order.find((f) => ig[f] > 0 && counts(f));
  const down = order.find((f) => ig[f] < 0 && counts(f));
  const measured = change !== undefined && Number.isFinite(change);
  const from = !measured
    ? ''
    : Math.abs(change) < 1e-3
      ? `From the origin to this point, the score for ${target} stays about the same. `
      : `From the origin to this point, the score for ${target} ${change > 0 ? 'rises' : 'falls'} by ${sig(Math.abs(change))}. ` +
        (change < 0 && predicted ? `${target} is still the prediction here; the origin just scores even higher for it. ` : '');
  if (up === undefined && down === undefined) return `${from}No feature moves the score for ${target} much at this point.`;
  const v = (f: number) => sig(ig[f], true);
  if (up === undefined) {
    return measured
      ? `${from}Every feature that matters lowers the score on the way; ${labels[down!]} the most (${v(down!)}).`
      : `${from}Every feature that matters pushes away from ${target} here; ${labels[down!]} the most (${v(down!)}).`;
  }
  const first = `${from}${labels[up]} pushes toward ${target} the most (${v(up)}).`;
  if (down === undefined) return first;
  return `${first} ${labels[down]} pushes away from it (${v(down)}).`;
}

const attributionJob: Job<AttributionParams, AttributionResult> = (ctx, params) => {
  const shape = ctx.arch.input;
  let p = params;
  // Colour images: occlusion paints with the average colour of the test images unless told otherwise.
  if (!p.fill && isImage(shape) && shape.c > 1) p = { ...p, fill: cachedMean(ctx.testX, ctx.testY.length, shape, ctx.scale) };
  return attribution(ctx.net, p);
};

const means = new WeakMap<object, number[]>();
function cachedMean(X: Uint8Array | Float32Array, n: number, shape: Shape, scale: number): number[] {
  let m = means.get(X);
  if (!m || m.length !== shape.c) {
    m = meanColour(X, n, shape, scale);
    means.set(X, m);
  }
  return m;
}

export const jobs: Record<string, Job> = { attribution: attributionJob };
