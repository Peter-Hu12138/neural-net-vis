import type { Network } from '../nn/network';
import { Rng } from '../nn/rng';
import type { Job } from './protocol';
import { receptiveBox, type Box } from './receptive';

/**
 * What does each unit detect? Two analyses:
 * - `topk` scans the test set and keeps, per unit, the digits that excite it most and least, with
 *   a histogram of its responses and the labels of its favourite digits.
 * - `actmax` synthesises an input that excites one unit (activation maximisation): gradient ascent
 *   on the pixels, with a little decay, blur and clamping so the result stays image-like.
 */

export type UnitKind = 'conv' | 'dense' | 'output';

/** One test digit and how strongly it drove the unit. */
export interface Hit {
  /** Test-set index. */
  index: number;
  /** The unit's response (see `responses`). */
  value: number;
  /** Conv filters: the input pixels behind the strongest position, else null (whole image). */
  box: Box | null;
  /** Conv filters: row and column of the strongest position in the pre-pool map; −1 otherwise. */
  y: number;
  x: number;
}

export interface UnitSummary {
  unit: number;
  /** Strongest responses first. */
  top: Hit[];
  /** Weakest responses first. */
  bottom: Hit[];
  mean: number;
  /** Share of test digits with a response above 0. */
  activeFraction: number;
  /** HIST_BINS equal-width bins over [lo, hi]. */
  hist: { lo: number; hi: number; counts: number[] };
  /** How many of the TOP_LABELS strongest digits carry each label 0–9. */
  labelCounts: number[];
}

export interface TopkParams {
  block: number;
  k?: number;
  /** Number of test digits to scan; all of them by default. */
  count?: number;
}

export interface TopkResult {
  block: number;
  kind: UnitKind;
  /** Test digits scanned. */
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
  /** The synthesised 28×28 input, values in [0, 1]. */
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
 */
export function responses(net: Network, block: number, values: Float32Array, pre: Float32Array, pos: Int32Array | null): void {
  const b = net.blocks[block];
  if (b.kind === 'conv') {
    const F = b.spec.filters;
    const HW = b.zShape.h * b.zShape.w;
    const z = b.z;
    for (let f = 0; f < F; f++) {
      const o = f * HW;
      let best = o;
      for (let i = o + 1; i < o + HW; i++) if (z[i] > z[best]) best = i;
      values[f] = b.a[best];
      pre[f] = z[best];
      if (pos) pos[f] = best - o;
    }
    return;
  }
  const out = block === net.blocks.length - 1;
  for (let j = 0; j < b.spec.units; j++) {
    values[j] = out ? b.z[j] : b.a[j];
    pre[j] = b.z[j];
  }
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

/** Per unit: strongest and weakest digits, mean, active share, histogram and top-50 labels. */
export const topk: Job<TopkParams, TopkResult> = function* (ctx, p) {
  const { net, spec } = ctx;
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
  const x = new Float32Array(784);
  const values = new Float32Array(U);
  const pre = new Float32Array(U);
  const pos = conv ? new Int32Array(U) : null;

  let last = now();
  for (let i = 0; i < N; i++) {
    ctx.image(i, x);
    forwardTo(net, x, block);
    responses(net, block, values, pre, pos);
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
    if (!posAll) return { index: i, value, box: null, y: -1, x: -1 };
    const at = posAll[i * U + u];
    const y = Math.floor(at / mapW);
    const xx = at % mapW;
    return { index: i, value, box: receptiveBox(spec, block, y, xx, 'z'), y, x: xx };
  };

  const order = new Int32Array(N);
  const units: UnitSummary[] = [];
  for (let u = 0; u < U; u++) {
    let lo = Infinity;
    let hi = -Infinity;
    let sum = 0;
    let active = 0;
    for (let i = 0; i < N; i++) {
      const v = resp[i * U + u];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
      sum += v;
      if (v > 0) active++;
    }
    for (let i = 0; i < N; i++) order[i] = i;
    // Strongest first; ties (for example ReLU zeros) by pre-activation, then by index.
    order.sort((a, c) => resp[c * U + u] - resp[a * U + u] || preAll[c * U + u] - preAll[a * U + u] || a - c);
    const top: Hit[] = [];
    for (let r = 0; r < Math.min(k, N); r++) top.push(hit(order[r], u));
    const bottom: Hit[] = [];
    for (let r = 0; r < Math.min(k, N); r++) bottom.push(hit(order[N - 1 - r], u));
    const labelCounts = new Array<number>(10).fill(0);
    for (let r = 0; r < Math.min(TOP_LABELS, N); r++) labelCounts[ctx.testY[order[r]]]++;

    if (!(hi - lo > 1e-9)) {
      // A unit that never varies (for example a dead ReLU): centre a unit-wide range on it.
      lo -= 0.5;
      hi += 0.5;
    }
    const counts = new Array<number>(HIST_BINS).fill(0);
    const scale = HIST_BINS / (hi - lo);
    for (let i = 0; i < N; i++) counts[Math.min(HIST_BINS - 1, Math.max(0, Math.floor((resp[i * U + u] - lo) * scale)))]++;

    units.push({ unit: u, top, bottom, mean: N ? sum / N : 0, activeFraction: N ? active / N : 0, hist: { lo, hi, counts }, labelCounts });
    yield { done: N + u + 1, total: N + U };
  }
  return { block, kind: unitKind(net, block), count: N, k, units };
};

/**
 * 3×3 binomial blur ([1 2 1] ⊗ [1 2 1] / 16), mixed 30 % into the image. Only pixels
 * inside `mask` are blurred, and only from neighbours inside it, so a small receptive field is
 * not dragged towards the zeros around it.
 */
function blur(x: Float32Array, mask: Uint8Array, tmp: Float32Array): void {
  tmp.set(x);
  for (let r = 0; r < 28; r++) {
    for (let c = 0; c < 28; c++) {
      const i = r * 28 + c;
      if (!mask[i]) continue;
      let s = 0;
      let wsum = 0;
      for (let dr = -1; dr <= 1; dr++) {
        const rr = r + dr;
        if (rr < 0 || rr > 27) continue;
        for (let dc = -1; dc <= 1; dc++) {
          const cc = c + dc;
          if (cc < 0 || cc > 27 || !mask[rr * 28 + cc]) continue;
          const w = (2 - Math.abs(dr)) * (2 - Math.abs(dc));
          s += w * tmp[rr * 28 + cc];
          wsum += w;
        }
      }
      x[i] = 0.7 * tmp[i] + 0.3 * (s / wsum);
    }
  }
}

/**
 * Activation maximisation. Objective: a conv filter's pre-activation at the centre of its map,
 * or a dense / output unit's pre-activation z_j. Gradient ascent from a nearly blank image with
 * an RMS-normalised step, L2 decay, a light blur every 4 steps (left out for the last fifth, so
 * the image sharpens) and clamping to [0, 1]. Pixels a conv filter cannot see are held at 0.
 */
export const actmax: Job<ActmaxParams, ActmaxResult> = function* (ctx, p) {
  const { net, spec } = ctx;
  const block = p.block;
  checkBlock(net, block);
  const steps = Math.max(1, Math.floor(p.steps ?? ACTMAX_STEPS));
  const U = unitCount(net, block);
  const list = (p.units ?? Array.from({ length: U }, (_, u) => u)).filter((u) => Number.isInteger(u) && u >= 0 && u < U);
  const b = net.blocks[block];

  let centre = 0;
  let HW = 1;
  let box: Box | null = null;
  const mask = new Uint8Array(784);
  if (b.kind === 'conv') {
    const cy = b.zShape.h >> 1;
    const cx = b.zShape.w >> 1;
    HW = b.zShape.h * b.zShape.w;
    centre = cy * b.zShape.w + cx;
    box = receptiveBox(spec, block, cy, cx, 'z');
  }
  if (box) {
    for (let r = box.y0; r <= box.y1; r++) for (let c = box.x0; c <= box.x1; c++) mask[r * 28 + c] = 1;
  } else mask.fill(1);
  let inside = 0;
  for (const m of mask) inside += m;

  const seed = new Float32Array(b.z.length);
  const tmp = new Float32Array(784);
  const total = list.length * steps;
  const out: UnitSynth[] = [];

  for (let n = 0; n < list.length; n++) {
    const u = list[n];
    const zi = b.kind === 'conv' ? u * HW + centre : u;
    const rng = new Rng(0x5eed + 7919 * (u + 1) + 104729 * block);
    const x = new Float32Array(784);
    for (let i = 0; i < 784; i++) x[i] = mask[i] ? 0.02 + 0.004 * (rng.next() - 0.5) : 0;
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
      for (let i = 0; i < 784; i++) if (mask[i]) ss += g[i] * g[i];
      const rms = Math.sqrt(ss / inside);
      const lr = 0.05 * (1 - (0.75 * t) / steps);
      if (rms > 0) {
        const step = lr / (rms + 1e-8);
        for (let i = 0; i < 784; i++) if (mask[i]) x[i] += step * g[i];
      }
      for (let i = 0; i < 784; i++) x[i] *= 1 - 0.003;
      if ((t + 1) % 4 === 0 && t < 0.8 * steps) blur(x, mask, tmp);
      for (let i = 0; i < 784; i++) x[i] = mask[i] ? Math.min(1, Math.max(0, x[i])) : 0;
      // A fresh snapshot every 10 steps; every report carries the latest one, because the
      // analyzer posts only some reports.
      if ((t + 1) % 10 === 0 || t === steps - 1) shown = { unit: u, x: x.slice(), step: t + 1, value };
      yield { done: n * steps + t + 1, total, partial: shown };
    }
    forwardTo(net, x, block);
    out.push({ unit: u, x, start, final: b.z[zi], box });
  }
  return { block, kind: unitKind(net, block), steps, units: out };
};

const plural = (d: number) => `${d}s`;

/** Plain-language summary of a label breakdown: "Mostly 0s (38), 6s (7) and 2 other digits." */
export function labelSummary(counts: number[]): string {
  const total = counts.reduce((a, b) => a + b, 0);
  const order = counts
    .map((n, d) => ({ n, d }))
    .filter((e) => e.n > 0)
    .sort((a, b) => b.n - a.n || a.d - b.d);
  if (!order.length) return 'No digits.';
  if (order[0].n === total) return `All ${total} are ${plural(order[0].d)}.`;
  const lead = order[0].n * 2 >= total ? 'Mostly' : 'Mixed:';
  const named = order.slice(0, 3).map((e) => `${plural(e.d)} (${e.n})`);
  const rest = order.length - named.length;
  if (rest > 0) return `${lead} ${named.join(', ')} and ${rest} other digit${rest > 1 ? 's' : ''}.`;
  const last = named.pop()!;
  return `${lead} ${named.length ? `${named.join(', ')} and ${last}` : last}.`;
}

export const jobs: Record<string, Job> = { topk, actmax };
