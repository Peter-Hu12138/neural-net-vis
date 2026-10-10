import type { Network } from '../nn/network';
import { Rng } from '../nn/rng';
import type { Job } from './protocol';
import { receptiveBox, type Box } from './receptive';

/**
 * What does each unit detect? Two analyses, for any dataset (grey or colour images, point features):
 * - `topk` scans the test set and keeps, per unit, the samples that excite it most and least, with
 *   a histogram of its responses and the labels of its favourite samples.
 * - `actmax` synthesises an image that excites one unit (activation maximisation): gradient ascent
 *   on the pixels, with a little decay, blur and clamping so the result stays image-like. Point
 *   datasets show each unit's response over the input plane instead (see `unitColumn`).
 */

export type UnitKind = 'conv' | 'dense' | 'output';

/** One test sample and how strongly it drove the unit. */
export interface Hit {
  /** Test-set index. */
  index: number;
  /** The unit's response (see `responses`). */
  value: number;
  /** Conv filters: the input pixels behind the strongest position, else null (the whole input). */
  box: Box | null;
  /** Conv filters: row and column of the strongest position in the pre-pool map; −1 otherwise. */
  y: number;
  x: number;
  /** The pre-activation behind `value` (the unit fires when it is above 0). */
  z: number;
}

export interface UnitSummary {
  unit: number;
  /** Strongest responses first. */
  top: Hit[];
  /** Weakest responses first. */
  bottom: Hit[];
  mean: number;
  /**
   * How often the unit fires, where firing means a pre-activation z above 0 (a > 0 for ReLU,
   * leaky ReLU, tanh and linear units; a > 0.5 for sigmoid):
   * conv filter → the share of map positions that fire, averaged over the test samples;
   * dense unit → the share of test samples that make it fire;
   * output unit → the share of test samples predicted as this class (its logit is the largest).
   */
  coverage: number;
  /** Every scanned sample's response, ascending, for exact ranks (see `rankIn`). */
  sorted: Float32Array;
  /** HIST_BINS equal-width bins over [lo, hi]. */
  hist: { lo: number; hi: number; counts: number[] };
  /** How many of the TOP_LABELS strongest samples carry each class label. */
  labelCounts: number[];
}

export interface TopkParams {
  block: number;
  k?: number;
  /** Number of test samples to scan; all of them by default. */
  count?: number;
}

export interface TopkResult {
  block: number;
  kind: UnitKind;
  /** Test samples scanned. */
  count: number;
  k: number;
  units: UnitSummary[];
}

export interface ActmaxParams {
  block: number;
  /** Units to synthesise; every unit of the block by default. */
  units?: number[];
  steps?: number;
}

export interface UnitSynth {
  unit: number;
  /** The synthesised input, shaped and stored like the network input (channel-major), values in [0, 1]. */
  x: Float32Array;
  /** Objective (the unit's pre-activation) at the blank start and after the last step. */
  start: number;
  final: number;
  /** Conv filters: the pixels the objective can see (everything else stays 0). */
  box: Box | null;
}

export interface ActmaxResult {
  block: number;
  kind: UnitKind;
  steps: number;
  units: UnitSynth[];
}

/** Sent with every actmax progress report so the page can animate the current unit. */
export interface ActmaxPartial {
  unit: number;
  x: Float32Array;
  step: number;
  value: number;
}

export const HIST_BINS = 24;
export const TOP_LABELS = 50;
export const ACTMAX_STEPS = 160;

export function unitKind(net: Network, block: number): UnitKind {
  if (block === net.blocks.length - 1) return 'output';
  return net.blocks[block].kind;
}

export function unitCount(net: Network, block: number): number {
  const b = net.blocks[block];
  return b.kind === 'conv' ? b.spec.filters : b.spec.units;
}

/** Runs blocks 0…`block` only; their tensors then hold this input's values. */
export function forwardTo(net: Network, x: Float32Array, block: number): void {
  let h = x;
  for (let i = 0; i <= block; i++) h = net.blocks[i].forward(h);
}

/**
 * Reads every unit's response from the block's tensors after a forward pass:
 * conv filter → its strongest post-activation value anywhere in the pre-pool map (with that
 * position in `pos`); dense hidden unit → its activation a_j; output unit → its logit z_k.
 * `pre` receives the matching pre-activation, used to break ties (several ReLU zeros).
 * Activations never decrease with z, so the arg-max of z is also an arg-max of a.
 * `cover`, when given, receives how much of the unit fires for this input (see
 * `UnitSummary.coverage`): the share of positions with z > 0 for a conv filter, 1 or 0 (z > 0)
 * for a dense unit, and 1 for the output unit with the largest logit (the prediction), else 0.
 */
export function responses(
  net: Network,
  block: number,
  values: Float32Array,
  pre: Float32Array,
  pos: Int32Array | null,
  cover: Float32Array | null = null,
): void {
  const b = net.blocks[block];
  if (b.kind === 'conv') {
    const F = b.spec.filters;
    const HW = b.zShape.h * b.zShape.w;
    const z = b.z;
    for (let f = 0; f < F; f++) {
      const o = f * HW;
      let best = o;
      let on = z[o] > 0 ? 1 : 0;
      for (let i = o + 1; i < o + HW; i++) {
        if (z[i] > z[best]) best = i;
        if (z[i] > 0) on++;
      }
      values[f] = b.a[best];
      pre[f] = z[best];
      if (pos) pos[f] = best - o;
      if (cover) cover[f] = on / HW;
    }
    return;
  }
  const out = block === net.blocks.length - 1;
  const n = b.spec.units;
  for (let j = 0; j < n; j++) {
    values[j] = out ? b.z[j] : b.a[j];
    pre[j] = b.z[j];
  }
  if (!cover) return;
  if (out) {
    let best = 0;
    for (let j = 1; j < n; j++) if (b.z[j] > b.z[best]) best = j;
    for (let j = 0; j < n; j++) cover[j] = j === best ? 1 : 0;
  } else for (let j = 0; j < n; j++) cover[j] = b.z[j] > 0 ? 1 : 0;
}

/** The response of one unit of `block` to input `x` (full forward pass; for the page's network). */
export function unitResponse(net: Network, block: number, unit: number, x: Float32Array): { value: number; y: number; x: number } {
  net.forward(x);
  const n = unitCount(net, block);
  const values = new Float32Array(n);
  const pre = new Float32Array(n);
  const b = net.blocks[block];
  const pos = b.kind === 'conv' ? new Int32Array(n) : null;
  responses(net, block, values, pre, pos);
  if (!pos || b.kind !== 'conv') return { value: values[unit], y: -1, x: -1 };
  const W = b.zShape.w;
  return { value: values[unit], y: Math.floor(pos[unit] / W), x: pos[unit] % W };
}

const now = () => performance.now();

function checkBlock(net: Network, block: number): void {
  if (!Number.isInteger(block) || block < 0 || block >= net.blocks.length) throw new Error(`No layer ${block}`);
}

/** Per unit: strongest and weakest samples, mean, coverage, sorted responses, histogram and top-50 labels. */
export const topk: Job<TopkParams, TopkResult> = function* (ctx, p) {
  const { net, arch } = ctx;
  const block = p.block;
  checkBlock(net, block);
  const k = Math.max(1, Math.floor(p.k ?? 9));
  const N = Math.min(ctx.testY.length, p.count ?? ctx.testY.length);
  const U = unitCount(net, block);
  const b = net.blocks[block];
  const conv = b.kind === 'conv';
  const mapW = conv ? b.zShape.w : 1;

  const resp = new Float32Array(N * U);
  const preAll = new Float32Array(N * U);
  const posAll = conv ? new Int32Array(N * U) : null;
  const x = new Float32Array(ctx.inputSize);
  const values = new Float32Array(U);
  const pre = new Float32Array(U);
  const pos = conv ? new Int32Array(U) : null;
  const cover = new Float32Array(U);
  const coverSum = new Float64Array(U);

  let last = now();
  for (let i = 0; i < N; i++) {
    ctx.image(i, x);
    forwardTo(net, x, block);
    responses(net, block, values, pre, pos, cover);
    for (let u = 0; u < U; u++) coverSum[u] += cover[u];
    resp.set(values, i * U);
    preAll.set(pre, i * U);
    if (posAll && pos) posAll.set(pos, i * U);
    if (now() - last > 3 || i === N - 1) {
      last = now();
      yield { done: i + 1, total: N + U };
    }
  }

  const hit = (i: number, u: number): Hit => {
    const value = resp[i * U + u];
    const z = preAll[i * U + u];
    if (!posAll) return { index: i, value, box: null, y: -1, x: -1, z };
    const at = posAll[i * U + u];
    const y = Math.floor(at / mapW);
    const xx = at % mapW;
    return { index: i, value, box: receptiveBox(arch, block, y, xx, 'z'), y, x: xx, z };
  };

  const order = new Int32Array(N);
  const units: UnitSummary[] = [];
  for (let u = 0; u < U; u++) {
    let lo = Infinity;
    let hi = -Infinity;
    let sum = 0;
    const sorted = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      const v = resp[i * U + u];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
      sum += v;
      sorted[i] = v;
    }
    sorted.sort();
    for (let i = 0; i < N; i++) order[i] = i;
    // Strongest first; ties (for example ReLU zeros) by pre-activation, then by index.
    order.sort((a, c) => resp[c * U + u] - resp[a * U + u] || preAll[c * U + u] - preAll[a * U + u] || a - c);
    const top: Hit[] = [];
    for (let r = 0; r < Math.min(k, N); r++) top.push(hit(order[r], u));
    const bottom: Hit[] = [];
    for (let r = 0; r < Math.min(k, N); r++) bottom.push(hit(order[N - 1 - r], u));
    const labelCounts = new Array<number>(ctx.classes).fill(0);
    for (let r = 0; r < Math.min(TOP_LABELS, N); r++) labelCounts[ctx.testY[order[r]]]++;

    if (!(hi - lo > 1e-9)) {
      // A unit that never varies (for example a dead ReLU): centre a unit-wide range on it.
      lo -= 0.5;
      hi += 0.5;
    }
    const counts = new Array<number>(HIST_BINS).fill(0);
    const scale = HIST_BINS / (hi - lo);
    for (let i = 0; i < N; i++) counts[Math.min(HIST_BINS - 1, Math.max(0, Math.floor((resp[i * U + u] - lo) * scale)))]++;

    units.push({ unit: u, top, bottom, mean: N ? sum / N : 0, coverage: N ? coverSum[u] / N : 0, sorted, hist: { lo, hi, counts }, labelCounts });
    yield { done: N + u + 1, total: N + U };
  }
  return { block, kind: unitKind(net, block), count: N, k, units };
};

/**
 * 3×3 binomial blur ([1 2 1] ⊗ [1 2 1] / 16) of each channel of a C × H × W image, mixed 30 % into
 * the image. Only pixels inside the spatial `mask` (H × W) are blurred, and only from neighbours
 * inside it, so a small receptive field is not dragged towards the background around it.
 */
function blur(x: Float32Array, mask: Uint8Array, tmp: Float32Array, C: number, H: number, W: number): void {
  tmp.set(x);
  const HW = H * W;
  for (let ch = 0; ch < C; ch++) {
    const o = ch * HW;
    for (let r = 0; r < H; r++) {
      for (let c = 0; c < W; c++) {
        const i = r * W + c;
        if (!mask[i]) continue;
        let s = 0;
        let wsum = 0;
        for (let dr = -1; dr <= 1; dr++) {
          const rr = r + dr;
          if (rr < 0 || rr >= H) continue;
          for (let dc = -1; dc <= 1; dc++) {
            const cc = c + dc;
            if (cc < 0 || cc >= W || !mask[rr * W + cc]) continue;
            const w = (2 - Math.abs(dr)) * (2 - Math.abs(dc));
            s += w * tmp[o + rr * W + cc];
            wsum += w;
          }
        }
        x[o + i] = 0.7 * tmp[o + i] + 0.3 * (s / wsum);
      }
    }
  }
}

/**
 * The blank image activation maximisation starts from and decays towards: black (0) for grey
 * images, whose background is 0 (MNIST, Fashion-MNIST), plain mid-grey (0.5) for colour photos,
 * so every channel has room to move both ways before it is clamped.
 */
export const synthBase = (channels: number): number => (channels === 1 ? 0 : 0.5);

/**
 * Activation maximisation. Objective: a conv filter's pre-activation at the centre of its map,
 * or a dense / output unit's pre-activation z_j. Gradient ascent from a nearly blank image (see
 * `synthBase`) with an RMS-normalised step, L2 decay towards the blank, a light blur of each
 * channel every 4 steps (left out for the last fifth, so the image sharpens) and clamping to
 * [0, 1]. Pixels a conv filter cannot see are held at the blank value. Works for any image shape
 * (28×28 grey, 32×32 colour); colour images are optimised in all three channels at once.
 */
export const actmax: Job<ActmaxParams, ActmaxResult> = function* (ctx, p) {
  const { net, arch } = ctx;
  const block = p.block;
  checkBlock(net, block);
  const steps = Math.max(1, Math.floor(p.steps ?? ACTMAX_STEPS));
  const U = unitCount(net, block);
  const list = (p.units ?? Array.from({ length: U }, (_, u) => u)).filter((u) => Number.isInteger(u) && u >= 0 && u < U);
  const b = net.blocks[block];
  const { c: C, h: H, w: W } = arch.input;
  const HW = H * W;
  const n = C * HW;
  const base = synthBase(C);

  let centre = 0;
  let mapHW = 1;
  let box: Box | null = null;
  const mask = new Uint8Array(HW);
  if (b.kind === 'conv') {
    const cy = b.zShape.h >> 1;
    const cx = b.zShape.w >> 1;
    mapHW = b.zShape.h * b.zShape.w;
    centre = cy * b.zShape.w + cx;
    box = receptiveBox(arch, block, cy, cx, 'z');
  }
  if (box) {
    for (let r = box.y0; r <= box.y1; r++) for (let c = box.x0; c <= box.x1; c++) mask[r * W + c] = 1;
  } else mask.fill(1);
  let inside = 0;
  for (const m of mask) inside += m;
  inside *= C;

  const seed = new Float32Array(b.z.length);
  const tmp = new Float32Array(n);
  const total = list.length * steps;
  const out: UnitSynth[] = [];

  for (let k = 0; k < list.length; k++) {
    const u = list[k];
    const zi = b.kind === 'conv' ? u * mapHW + centre : u;
    const rng = new Rng(0x5eed + 7919 * (u + 1) + 104729 * block);
    const x = new Float32Array(n);
    for (let ch = 0; ch < C; ch++) {
      for (let i = 0; i < HW; i++) x[ch * HW + i] = mask[i] ? base + 0.02 + 0.004 * (rng.next() - 0.5) : base;
    }
    seed.fill(0);
    seed[zi] = 1;
    let start = NaN;
    let shown: ActmaxPartial = { unit: u, x: x.slice(), step: 0, value: NaN };

    for (let t = 0; t < steps; t++) {
      forwardTo(net, x, block);
      const value = b.z[zi];
      if (t === 0) {
        start = value;
        shown.value = value;
      }
      const g = net.inputGradient(block, seed);
      let ss = 0;
      for (let i = 0; i < n; i++) if (mask[i % HW]) ss += g[i] * g[i];
      const rms = Math.sqrt(ss / inside);
      const lr = 0.05 * (1 - (0.75 * t) / steps);
      if (rms > 0) {
        const step = lr / (rms + 1e-8);
        for (let i = 0; i < n; i++) if (mask[i % HW]) x[i] += step * g[i];
      }
      for (let i = 0; i < n; i++) x[i] = base + (x[i] - base) * (1 - 0.003);
      if ((t + 1) % 4 === 0 && t < 0.8 * steps) blur(x, mask, tmp, C, H, W);
      for (let i = 0; i < n; i++) x[i] = mask[i % HW] ? Math.min(1, Math.max(0, x[i])) : base;
      // A fresh snapshot every 10 steps; every report carries the latest one, because the
      // analyzer posts only some reports.
      if ((t + 1) % 10 === 0 || t === steps - 1) shown = { unit: u, x: x.slice(), step: t + 1, value };
      yield { done: k * steps + t + 1, total, partial: shown };
    }
    forwardTo(net, x, block);
    out.push({ unit: u, x, start, final: b.z[zi], box });
  }
  return { block, kind: unitKind(net, block), steps, units: out };
};

/** Where a value falls among the scanned responses. */
export interface Rank {
  /** Responses strictly below, equal to, and strictly above the value. */
  below: number;
  tied: number;
  above: number;
  n: number;
}

/**
 * Exact rank of `v` among the ascending responses `sorted` (float32 values; `v` is compared as a
 * float32 too, so a test sample's own response ties with itself).
 */
export function rankIn(sorted: ArrayLike<number>, v: number): Rank {
  const n = sorted.length;
  if (Number.isNaN(v)) return { below: 0, tied: 0, above: 0, n };
  const t = Math.fround(v);
  // First index with sorted[i] >= t, then first with sorted[i] > t.
  const bound = (strict: boolean) => {
    let lo = 0;
    let hi = n;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (strict ? sorted[mid] <= t : sorted[mid] < t) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  };
  const lo = bound(false);
  const hi = bound(true);
  return { below: lo, tied: hi - lo, above: n - hi, n };
}

/** A share as a percentage that only reads 0% or 100% when it is exactly that. */
export function sharePct(f: number, decimals = 0): string {
  if (!(f > 0)) return '0%';
  if (f >= 1) return '100%';
  const step = 10 ** -decimals;
  const p = f * 100;
  if (p < step) return `<${step.toFixed(decimals)}%`;
  if (p > 100 - step) return `>${(100 - step).toFixed(decimals)}%`;
  return `${p.toFixed(decimals)}%`;
}

/**
 * Plain-language rank against the test samples, for "This input: 0.42, <phrase>." `one` names a
 * sample ("digit", "image", "point"). Ties count when they cover at least 1% of the samples (for
 * example a ReLU unit that is 0 for most of them).
 */
export function rankPhrase(r: Rank, one = 'digit'): string {
  const { below, tied, above, n } = r;
  if (!n) return `with no test ${one}s to compare`;
  if (below + tied + above !== n) return `not comparable with the test ${one}s`;
  const all = `${n.toLocaleString('en-US')} test ${one}${n === 1 ? '' : 's'}`;
  const share = (k: number) => sharePct(k / n, 1);
  if (tied === n) return `the same as all ${all}`;
  if (tied > 1 && tied >= 0.01 * n) {
    if (below === 0) return `tied with ${share(tied)} of the ${all} at the lowest response`;
    if (above === 0) return `tied with ${share(tied)} of the ${all} at the highest response`;
    return `higher than ${share(below)} of the ${all} and tied with another ${share(tied)}`;
  }
  const s = (k: number) => (k === 1 ? 's' : '');
  if (above === 0) return tied ? `as high as the strongest of the ${all}` : `higher than all ${all}`;
  if (below === 0) return tied ? `as low as the weakest of the ${all}` : `lower than all ${all}`;
  if (above < 10) return `only ${above} of the ${all} respond${s(above)} more strongly`;
  if (below < 10) return `only ${below} of the ${all} respond${s(below)} more weakly`;
  return `higher than ${share(below)} of the ${all}`;
}

/**
 * "fires at 23% of positions" (conv), "fires on 37% of digits" (dense), "predicted for 10% of
 * digits" (output). `one` names a sample: "digit", "image", "point".
 */
export function coveragePhrase(kind: UnitKind, f: number, decimals = 0, one = 'digit'): string {
  const p = sharePct(f, decimals);
  if (kind === 'conv') return `fires at ${p} of positions`;
  if (kind === 'dense') return `fires on ${p} of ${one}s`;
  return `predicted for ${p} of ${one}s`;
}

/** True when every one of `hits` leaves the unit off (pre-activation at or below 0). */
export const allOff = (hits: Hit[]): boolean => hits.length > 0 && hits.every((h) => h.z <= 0);

/**
 * A hidden unit that gives the same response to every scanned sample: a dead ReLU unit (always 0)
 * or filter. Its "strongest" samples are then only the ones closest to firing (highest
 * pre-activation), which the view has to say. Output units are never called dead.
 */
export function isDead(kind: UnitKind, s: Pick<UnitSummary, 'sorted'>): boolean {
  const n = s.sorted.length;
  return kind !== 'output' && n > 0 && s.sorted[0] === s.sorted[n - 1];
}

/** "Class 0" → "class 0", "Ankle boot" → "ankle boot"; leaves acronyms ("CNN") alone. */
const lowerFirst = (t: string) => (/^[A-Z][A-Z]/.test(t) ? t : t.charAt(0).toLowerCase() + t.slice(1));

/**
 * Plain-language summary of a label breakdown: "Mostly 0s (38), 6s (7) and 2 other digits." Every
 * label tied with the third is named too ("Mixed: 0s, 3s, 4s and 9s (7 each) and 6 other
 * digits."), so the text never implies a lead that the bars do not show. Without `names` the
 * labels are digits ("7s"); with them, class names as they are ("cat (12)", "class 1 (30)").
 */
export function labelSummary(counts: number[], names?: string[]): string {
  const total = counts.reduce((a, b) => a + b, 0);
  const word = names ? 'class' : 'digit';
  const words = names ? 'classes' : 'digits';
  const label = (d: number) => (names ? lowerFirst(names[d] ?? `class ${d}`) : `${d}s`);
  const order = counts
    .map((n, d) => ({ n, d }))
    .filter((e) => e.n > 0)
    .sort((a, b) => b.n - a.n || a.d - b.d);
  if (!order.length) return names ? 'No labels.' : 'No digits.';
  if (order[0].n === total) return names ? `All ${total} are labelled ${label(order[0].d)}.` : `All ${total} are ${label(order[0].d)}.`;
  const lead = order[0].n * 2 >= total ? 'Mostly' : 'Mixed:';
  const third = order[Math.min(2, order.length - 1)].n;
  const named = order.filter((e) => e.n >= third);
  // Runs of equal counts share one number: "0s, 3s and 4s (7 each)".
  const groups: { n: number; labels: string[] }[] = [];
  for (const e of named) {
    const g = groups[groups.length - 1];
    if (g && g.n === e.n) g.labels.push(label(e.d));
    else groups.push({ n: e.n, labels: [label(e.d)] });
  }
  const list = (xs: string[]) => (xs.length < 2 ? xs.join('') : `${xs.slice(0, -1).join(', ')} and ${xs[xs.length - 1]}`);
  const parts = groups.map((g) => (g.labels.length === 1 ? `${g.labels[0]} (${g.n})` : `${list(g.labels)} (${g.n} each)`));
  const rest = order.length - named.length;
  if (rest > 0) return `${lead} ${parts.join(', ')} and ${rest} other ${rest > 1 ? words : word}.`;
  // A final group that is itself a list joins with a comma, so there is only one "and" in it.
  if (parts.length > 1 && groups[groups.length - 1].labels.length > 1) return `${lead} ${parts.join(', ')}.`;
  return `${lead} ${list(parts)}.`;
}

/**
 * Column `j` of an n × U matrix, row by row: one unit's response over a grid of points, from the
 * per-block activations a PointEvaluator returns.
 */
export function unitColumn(acts: ArrayLike<number>, U: number, j: number, out?: Float32Array): Float32Array {
  const n = Math.floor(acts.length / U);
  const col = out ?? new Float32Array(n);
  for (let q = 0; q < n; q++) col[q] = acts[q * U + j];
  return col;
}

/**
 * How to colour a response map: diverging around 0 (red positive, blue negative) when any value is
 * negative, else sequential from 0; `max` is the largest |value| (1 when the map is all zeros).
 */
export function mapScale(values: ArrayLike<number>): { signed: boolean; max: number; lo: number; hi: number } {
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (!Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (!(lo <= hi)) return { signed: false, max: 1, lo: 0, hi: 0 };
  const signed = lo < -1e-9;
  const max = Math.max(Math.abs(lo), Math.abs(hi));
  return { signed, max: max > 1e-12 ? max : 1, lo, hi };
}

export const jobs: Record<string, Job> = { topk, actmax };
