import type { Network } from '../nn/network';
import { CLASSES } from '../nn/types';
import type { Job, Progress } from './protocol';

/**
 * "Why this prediction": four ways to say which pixels drive one digit's score for one input.
 *
 * With z_t the target's logit and g(x) = ∂z_t/∂x:
 * - saliency      |g(x)|
 * - gradInput     x ⊙ g(x)
 * - integrated    x ⊙ mean_k g(α_k·x), α_k = (k + ½)/m: integrated gradients from a blank image,
 *                 midpoint Riemann sum. Its total should match z_t(x) − z_t(0) (completeness).
 * - occlusion     erase a size×size patch (set it to 0), record the drop in p_t; each pixel gets
 *                 the mean drop over the patches that cover it.
 */

const SIDE = 28;
const PIXELS = SIDE * SIDE;

export interface OcclusionParams {
  size: number;
  stride: number;
}

export interface AttributionParams {
  x: Float32Array;
  target: number;
  igSteps?: number;
  occlusion?: OcclusionParams;
}

export interface AttributionResult {
  target: number;
  /** Predicted digit for x, and the full softmax output. */
  pred: number;
  probs: Float32Array;
  /** z_t(x) and p_t(x). */
  logit: number;
  prob: number;
  saliency: Float32Array;
  gradInput: Float32Array;
  integrated: Float32Array;
  /** Σ integrated, and what it should equal: z_t(x) − z_t(0). */
  igSum: number;
  igExpected: number;
  /** z_t of the blank image. */
  baseLogit: number;
  occlusion: Float32Array;
  /** The settings actually used, after clamping. */
  igSteps: number;
  occlusionSize: number;
  occlusionStride: number;
  /** Number of patches tried, and how many needed a forward pass (the rest were already blank). */
  patches: number;
  patchesEvaluated: number;
}

export const DEFAULT_IG_STEPS = 32;
export const DEFAULT_OCCLUSION: OcclusionParams = { size: 6, stride: 2 };

export function oneHot(k: number, n = CLASSES): Float32Array {
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
  const grad = net.inputGradient(last, oneHot(target)).slice();
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

/** All size×size patches of a side×side image, row by row. */
export function patchGrid(side: number, size: number, stride: number): Patch[] {
  const o = patchOrigins(side, size, stride);
  const out: Patch[] = [];
  for (const y of o) for (const x of o) out.push({ y, x });
  return out;
}

/** True when every pixel of the patch is already 0, so erasing it changes nothing. */
export function patchIsBlank(img: ArrayLike<number>, p: Patch, size: number, side = SIDE): boolean {
  const h = Math.min(size, side - p.y);
  const w = Math.min(size, side - p.x);
  for (let r = 0; r < h; r++) {
    const o = (p.y + r) * side + p.x;
    for (let c = 0; c < w; c++) if (img[o + c] !== 0) return false;
  }
  return true;
}

/** Per pixel, the mean of `drops` over the patches covering it; 0 where no patch reaches. */
export function occlusionMap(patches: Patch[], drops: ArrayLike<number>, size: number, side = SIDE): Float32Array {
  const sum = new Float64Array(side * side);
  const count = new Uint16Array(side * side);
  patches.forEach((p, k) => {
    const h = Math.min(size, side - p.y);
    const w = Math.min(size, side - p.x);
    for (let r = 0; r < h; r++) {
      const o = (p.y + r) * side + p.x;
      for (let c = 0; c < w; c++) {
        sum[o + c] += drops[k];
        count[o + c]++;
      }
    }
  });
  const out = new Float32Array(side * side);
  for (let i = 0; i < out.length; i++) out[i] = count[i] ? sum[i] / count[i] : 0;
  return out;
}

/** |a − b| relative to |b|; 0 when both are (near) zero. */
export function relativeGap(a: number, b: number): number {
  const d = Math.abs(a - b);
  const s = Math.abs(b);
  if (s < 1e-9) return d < 1e-9 ? 0 : Infinity;
  return d / s;
}

const clampInt = (v: number | undefined, lo: number, hi: number, dflt: number) =>
  Number.isFinite(v) ? Math.max(lo, Math.min(hi, Math.round(v as number))) : dflt;

/** The attribution job. Yields after every forward pass (with or without a backward pass). */
export function* attribution(net: Network, params: AttributionParams): Generator<Progress, AttributionResult, void> {
  const x = params.x;
  if (!x || x.length !== PIXELS) throw new Error(`Attribution needs a 28×28 image, got ${x ? x.length : 0} values`);
  const target = params.target;
  if (!Number.isInteger(target) || target < 0 || target >= CLASSES) throw new Error(`Target must be a digit 0–9, got ${target}`);
  const m = clampInt(params.igSteps, 1, 1024, DEFAULT_IG_STEPS);
  const size = clampInt(params.occlusion?.size, 1, SIDE, DEFAULT_OCCLUSION.size);
  const stride = clampInt(params.occlusion?.stride, 1, SIDE, DEFAULT_OCCLUSION.stride);

  const patches = patchGrid(SIDE, size, stride);
  const live = patches.map((p) => !patchIsBlank(x, p, size));
  const evaluated = live.filter(Boolean).length;
  const total = 2 + m + evaluated;
  let done = 0;
  const last = net.blocks.length - 1;
  const seed = oneHot(target);

  // 1. The gradient at x.
  const at = logitGradient(net, x, target);
  let pred = 0;
  for (let j = 1; j < CLASSES; j++) if (at.probs[j] > at.probs[pred]) pred = j;
  const saliency = saliencyOf(at.grad);
  const gradInput = gradTimesInput(x, at.grad);
  yield { done: ++done, total };

  // 2. The blank baseline.
  const baseLogit = targetLogit(net, new Float32Array(PIXELS), target);
  yield { done: ++done, total };

  // 3. Integrated gradients along the straight path from blank to x.
  const xs = new Float32Array(PIXELS);
  const acc = new Float64Array(PIXELS);
  for (const alpha of igAlphas(m)) {
    net.forward(scaleInto(x, alpha, xs));
    const g = net.inputGradient(last, seed);
    for (let i = 0; i < PIXELS; i++) acc[i] += g[i];
    yield { done: ++done, total };
  }
  const integrated = new Float32Array(PIXELS);
  let igSum = 0;
  for (let i = 0; i < PIXELS; i++) {
    const v = (x[i] * acc[i]) / m;
    integrated[i] = v;
    igSum += v;
  }

  // 4. Occlusion: erase one patch at a time.
  const drops = new Float64Array(patches.length);
  const work = x.slice();
  const saved = new Float32Array(size * size);
  for (let k = 0; k < patches.length; k++) {
    if (!live[k]) continue; // erasing blank pixels leaves the input, and so the score, unchanged
    const p = patches[k];
    const h = Math.min(size, SIDE - p.y);
    const w = Math.min(size, SIDE - p.x);
    for (let r = 0; r < h; r++) {
      const o = (p.y + r) * SIDE + p.x;
      for (let c = 0; c < w; c++) {
        saved[r * size + c] = work[o + c];
        work[o + c] = 0;
      }
    }
    drops[k] = at.prob - net.forward(work)[target];
    for (let r = 0; r < h; r++) {
      const o = (p.y + r) * SIDE + p.x;
      for (let c = 0; c < w; c++) work[o + c] = saved[r * size + c];
    }
    yield { done: ++done, total };
  }
  const occlusion = occlusionMap(patches, drops, size);

  return {
    target,
    pred,
    probs: at.probs,
    logit: at.logit,
    prob: at.prob,
    saliency,
    gradInput,
    integrated,
    igSum,
    igExpected: at.logit - baseLogit,
    baseLogit,
    occlusion,
    igSteps: m,
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

const attributionJob: Job<AttributionParams, AttributionResult> = (ctx, params) => attribution(ctx.net, params);

export const jobs: Record<string, Job> = { attribution: attributionJob };
