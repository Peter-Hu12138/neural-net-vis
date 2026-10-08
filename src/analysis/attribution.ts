import type { Block, Network } from '../nn/network';
import type { Job, Progress } from './protocol';

/**
 * "Why this prediction": four ways to say which pixels drive one digit's score for one input.
 *
 * With z_t the target's logit and g(x) = ∂z_t/∂x:
 * - saliency      |g(x)|
 * - gradInput     x ⊙ g(x)
 * - integrated    x ⊙ mean_k g(α_k·x), α_k = (k + ½)/m: integrated gradients from a blank image,
 *                 midpoint Riemann sum. Its total should match z_t(x) − z_t(0) (completeness).
 * - occlusion     erase a size×size patch (set it to 0), record the drop in z_t; each pixel gets
 *                 the mean drop over the patches that cover it. The logit, like the other three
 *                 maps, so all four share units; the probability saturates near 0 and 1, where
 *                 erasing anything barely moves it. The drop in p_t is kept too, for tooltips.
 *
 * The gradients come from Network.inputGradient in its symmetric mode: blank MNIST pixels leave
 * many units exactly on a kink (a ReLU at z = 0, a max-pool window of equal values), where the
 * slope differs on either side; there it takes the average. `kinks` counts those units.
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
  /** Σ |integrated|: the total size of the attributions, the scale for the completeness gap. */
  igAbsSum: number;
  /** z_t of the blank image. */
  baseLogit: number;
  /** Mean drop in z_t when a covering patch is erased. */
  occlusion: Float32Array;
  /** The same for p_t. */
  occlusionProb: Float32Array;
  /** Units sitting exactly on a kink for this input (see above). */
  kinks: Kinks;
  /** The settings actually used, after clamping. */
  igSteps: number;
  occlusionSize: number;
  occlusionStride: number;
  /** Number of patches tried, and how many needed a forward pass (the rest were already blank). */
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
  /** diff relative to `ref` (never infinite). */
  rel: number;
  /** 'ig': Σ|IG|, the total size of the attributions; 'score': |z(x) − z(blank)|, the larger. */
  ref: 'ig' | 'score';
}

/**
 * How far integrated gradients are from adding up to the score change. Measured against Σ|IG|
 * (or |z(x) − z(blank)| if larger), not the score change alone: for a digit the network neither
 * likes nor rejects, large positive and negative attributions cancel, z(x) − z(blank) is close to
 * 0 and a relative gap would read hundreds of percent although the sum is accurate.
 */
export function completenessGap(igSum: number, igExpected: number, igAbsSum: number): CompletenessGap {
  const diff = Math.abs(igSum - igExpected);
  const score = Math.abs(igExpected);
  const ref = igAbsSum >= score ? 'ig' : 'score';
  const s = Math.max(igAbsSum, score);
  // s = 0 means no attribution and no score change, so diff = 0 too.
  return { diff, rel: s > 0 ? diff / s : 0, ref };
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
  if (!x || x.length !== PIXELS) throw new Error(`Attribution needs a 28×28 image, got ${x ? x.length : 0} values`);
  const target = params.target;
  if (!Number.isInteger(target) || target < 0 || target >= net.classes) throw new Error(`Target must be a class 0–${net.classes - 1}, got ${target}`);
  const m = clampInt(params.igSteps, 1, 1024, DEFAULT_IG_STEPS);
  const size = clampInt(params.occlusion?.size, 1, SIDE, DEFAULT_OCCLUSION.size);
  const stride = clampInt(params.occlusion?.stride, 1, SIDE, DEFAULT_OCCLUSION.stride);

  const patches = patchGrid(SIDE, size, stride);
  const live = patches.map((p) => !patchIsBlank(x, p, size));
  const evaluated = live.filter(Boolean).length;
  const total = 2 + m + evaluated;
  let done = 0;
  const last = net.blocks.length - 1;
  const seed = oneHot(target, net.classes);

  // 1. The gradient at x.
  const at = logitGradient(net, x, target);
  const kinks = countKinks(net); // the network still holds the activations of x
  let pred = 0;
  for (let j = 1; j < at.probs.length; j++) if (at.probs[j] > at.probs[pred]) pred = j;
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
  let igAbsSum = 0;
  for (let i = 0; i < PIXELS; i++) {
    const v = (x[i] * acc[i]) / m;
    integrated[i] = v;
    igSum += v;
    igAbsSum += Math.abs(v);
  }

  // 4. Occlusion: erase one patch at a time.
  const drops = new Float64Array(patches.length);
  const probDrops = new Float64Array(patches.length);
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
    probDrops[k] = at.prob - net.forward(work)[target];
    drops[k] = at.logit - net.blocks[last].z[target];
    for (let r = 0; r < h; r++) {
      const o = (p.y + r) * SIDE + p.x;
      for (let c = 0; c < w; c++) work[o + c] = saved[r * size + c];
    }
    yield { done: ++done, total };
  }
  const occlusion = occlusionMap(patches, drops, size);
  const occlusionProb = occlusionMap(patches, probDrops, size);

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
    igAbsSum,
    baseLogit,
    occlusion,
    occlusionProb,
    kinks,
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

// ── Text for the view (pure, so the unit tests can read it) ──

const MINUS = '−';

/** Three significant digits, a true minus sign (exponents too), '+' on request; never "−0". */
export function sig(v: number, plus = false): string {
  if (!Number.isFinite(v)) return '–';
  if (v === 0) return '0';
  const a = Math.abs(v);
  const body = a >= 1000 ? a.toFixed(0) : a >= 0.001 ? String(Number(a.toPrecision(3))) : a.toExponential(1).replace('e-', `e${MINUS}`);
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
 * between: "Σ IG = 8.39;" "z(x) − z(blank) = 8.43" "(off by 0.0389, 0.3% of Σ|IG|)".
 */
export function completenessText(r: Pick<AttributionResult, 'igSum' | 'igExpected' | 'igAbsSum'>): { sum: string; expected: string; gap: string; title: string } {
  const g = completenessGap(r.igSum, r.igExpected, r.igAbsSum);
  const ref = g.ref === 'ig' ? 'Σ|IG|' : `|z(x) ${MINUS} z(blank)|`;
  return {
    sum: `Σ IG = ${sig(r.igSum)};`,
    expected: `z(x) ${MINUS} z(blank) = ${sig(r.igExpected)}`,
    gap: g.diff === 0 ? '(a match)' : `(off by ${sig(g.diff)}, ${percentText(g.rel)} of ${ref})`,
    title:
      `Completeness: integrated gradients should add up to the change in the score from a blank image to this one. ` +
      `Σ IG = ${exact(r.igSum)}, z(x) ${MINUS} z(blank) = ${exact(r.igExpected)}, Σ|IG| = ${exact(r.igAbsSum)}. ` +
      `The gap is measured against Σ|IG|, the total size of all the pixel attributions, so it stays meaningful when positive and negative attributions cancel out.`,
  };
}

/** Saliency hint addition when the input leaves units on a kink; '' when there are none. */
export function kinkText(k: Kinks): string {
  const kinds = [k.relu ? 'ReLUs at exactly 0' : '', k.pool ? 'tied max-pool windows' : ''].filter(Boolean);
  if (!kinds.length) return '';
  return `Where the input leaves ${kinds.join(' and ')} (mostly the blank background), brightening and darkening a pixel differ; the map shows the average slope.`;
}

const attributionJob: Job<AttributionParams, AttributionResult> = (ctx, params) => attribution(ctx.net, params);

export const jobs: Record<string, Job> = { attribution: attributionJob };
