/**
 * Distribution statistics for the Q–Q plots and histograms. Pure functions, no DOM, so they run in
 * the analysis worker, on the page and in the unit tests alike. Non-finite values (a diverged
 * network can produce NaN weights) are skipped everywhere.
 */

// Acklam's rational approximation of the inverse normal CDF (relative error < 1.15e-9).
const A = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.38357751867269e2, -3.066479806614716e1, 2.506628277459239];
const B = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
const C = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
const D = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
const P_LOW = 0.02425;
const SQRT_2PI = Math.sqrt(2 * Math.PI);

/** Acklam's approximation for p ∈ (0, 0.5]. */
function acklam(p: number): number {
  if (p < P_LOW) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5]) / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1);
  }
  const q = p - 0.5;
  const r = q * q;
  return ((((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q) / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1);
}

/**
 * Standard normal CDF Φ(x), to about 1e-14 relative error in both tails: Hart's 1968 rational
 * approximation for |x| < 3 (as given by G. West, "Better approximations to cumulative normal
 * functions", 2005), Laplace's continued fraction beyond.
 */
export function normalCdf(x: number): number {
  if (Number.isNaN(x)) return NaN;
  const ax = Math.abs(x);
  const e = Math.exp((-ax * ax) / 2);
  let c: number;
  if (ax < 3) {
    let n = 3.52624965998911e-2 * ax + 0.700383064443688;
    n = n * ax + 6.37396220353165;
    n = n * ax + 33.912866078383;
    n = n * ax + 112.079291497871;
    n = n * ax + 221.213596169931;
    n = n * ax + 220.206867912376;
    let d = 8.83883476483184e-2 * ax + 1.75566716318264;
    d = d * ax + 16.064177579207;
    d = d * ax + 86.7807322029461;
    d = d * ax + 296.564248779674;
    d = d * ax + 637.333633378831;
    d = d * ax + 793.826512519948;
    d = d * ax + 440.413735824752;
    c = (e * n) / d;
  } else {
    // Φ(−x) = φ(x) / (x + 1/(x + 2/(x + 3/(x + …)))), evaluated bottom-up.
    let t = ax;
    for (let k = 40; k >= 1; k--) t = ax + k / t;
    c = e / SQRT_2PI / t;
  }
  return x > 0 ? 1 - c : c;
}

/**
 * Inverse standard normal CDF Φ⁻¹(p): Acklam's algorithm, polished with one Halley step against
 * normalCdf. Returns −∞ at 0, +∞ at 1 and NaN outside [0, 1]. Exactly antisymmetric about 0.5.
 */
export function normalQuantile(p: number): number {
  if (!(p >= 0 && p <= 1)) return NaN;
  if (p === 0) return -Infinity;
  if (p === 1) return Infinity;
  // 1 − p is exact for p ∈ [0.5, 1], so the upper half reuses the more precise lower tail.
  if (p > 0.5) return -normalQuantile(1 - p);
  let x = acklam(p);
  if (x > -37) {
    const e = normalCdf(x) - p;
    const u = e * SQRT_2PI * Math.exp((x * x) / 2);
    const next = x - u / (1 + (x * u) / 2);
    if (Number.isFinite(next)) x = next;
  }
  return x;
}

const Q1_NORMAL = -0.6744897501960817; // Φ⁻¹(0.25)
const Q3_NORMAL = 0.6744897501960817; // Φ⁻¹(0.75)

/** Sample quantile by linear interpolation between order statistics (R's default, type 7). NaN when empty. */
export function quantileSorted(sorted: ArrayLike<number>, p: number): number {
  const n = sorted.length;
  if (n === 0) return NaN;
  if (n === 1) return sorted[0];
  const h = (n - 1) * Math.min(1, Math.max(0, p));
  const lo = Math.floor(h);
  if (lo >= n - 1) return sorted[n - 1];
  return sorted[lo] + (h - lo) * (sorted[lo + 1] - sorted[lo]);
}

/**
 * Ascending sort of a Float32Array by LSD radix sort on the IEEE bit patterns (three passes of
 * 11, 11 and 10 bits). Several times faster than the built-in sort for layer-sized arrays, which
 * matters because the weight views re-sort on every training update.
 */
export function radixSortFloat32(values: Float32Array): Float32Array {
  const n = values.length;
  const src = new Uint32Array(values.buffer, values.byteOffset, n);
  let a = new Uint32Array(n);
  let b = new Uint32Array(n);
  // Map bits so unsigned order is float order: negatives flip all bits, positives flip the sign.
  for (let i = 0; i < n; i++) {
    const x = src[i];
    a[i] = x & 0x80000000 ? ~x >>> 0 : (x | 0x80000000) >>> 0;
  }
  for (const [shift, bits] of [
    [0, 11],
    [11, 11],
    [22, 10],
  ]) {
    const size = 1 << bits;
    const mask = size - 1;
    const count = new Uint32Array(size + 1);
    for (let i = 0; i < n; i++) count[((a[i] >>> shift) & mask) + 1]++;
    for (let k = 0; k < size; k++) count[k + 1] += count[k];
    for (let i = 0; i < n; i++) b[count[(a[i] >>> shift) & mask]++] = a[i];
    const t = a;
    a = b;
    b = t;
  }
  const out = new Float32Array(n);
  const ov = new Uint32Array(out.buffer);
  for (let i = 0; i < n; i++) {
    const k = a[i];
    ov[i] = k & 0x80000000 ? (k ^ 0x80000000) >>> 0 : ~k >>> 0;
  }
  return out;
}

/** Ascending copy of the finite values. */
export function sortedFinite(values: ArrayLike<number>): Float64Array {
  const f32 = values instanceof Float32Array && values.length > 256;
  const buf = f32 ? new Float32Array(values.length) : new Float64Array(values.length);
  let k = 0;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (Number.isFinite(v)) buf[k++] = v;
  }
  const kept = k === buf.length ? buf : buf.slice(0, k);
  if (kept instanceof Float32Array) return Float64Array.from(radixSortFloat32(kept));
  return kept.sort();
}

/** Blom plotting positions (i − 3/8)/(m + 1/4), i = 1…m. */
export function blomProbs(m: number): Float64Array {
  const p = new Float64Array(Math.max(0, m));
  for (let i = 0; i < m; i++) p[i] = (i + 1 - 0.375) / (m + 0.25);
  return p;
}

const blomCache = new Map<number, Float64Array>();

/** Φ⁻¹ of the Blom positions for m points, memoised (the same few sizes recur on every redraw). */
export function blomQuantiles(m: number): Float64Array {
  let q = blomCache.get(m);
  if (!q) {
    q = blomProbs(m).map(normalQuantile);
    if (blomCache.size >= 16) blomCache.delete(blomCache.keys().next().value!);
    blomCache.set(m, q);
  }
  return q;
}

/**
 * Sample quantiles at `probs`. When every value is plotted (m = n) they are the order statistics
 * themselves, as in a textbook Q–Q plot; otherwise R type-7 interpolation.
 */
function sampleQuantiles(sorted: Float64Array, probs: Float64Array): Float64Array {
  if (sorted.length === probs.length) return sorted.slice();
  const out = new Float64Array(probs.length);
  for (let i = 0; i < probs.length; i++) out[i] = quantileSorted(sorted, probs[i]);
  return out;
}

export interface Line {
  slope: number;
  intercept: number;
}

export interface QQNormal {
  /** Number of finite values the plot summarises. */
  n: number;
  probs: Float64Array;
  /** Φ⁻¹(p_i): the x coordinates. */
  theoretical: Float64Array;
  /** Sample quantiles at p_i: the y coordinates. */
  sample: Float64Array;
  /** R's qqline: the normal distribution through the sample's quartiles. */
  line: Line;
}

/** qqline for a sorted sample: slope = IQR / normal IQR, through the first quartile. */
export function qqLineSorted(sorted: ArrayLike<number>): Line {
  if (sorted.length === 0) return { slope: 0, intercept: 0 };
  const q1 = quantileSorted(sorted, 0.25);
  const q3 = quantileSorted(sorted, 0.75);
  const slope = (q3 - q1) / (Q3_NORMAL - Q1_NORMAL);
  return { slope, intercept: q1 - slope * Q1_NORMAL };
}

export function qqNormalSorted(sorted: Float64Array, maxPoints = 256): QQNormal {
  const n = sorted.length;
  const m = Math.min(n, Math.max(1, Math.floor(maxPoints)));
  const probs = blomProbs(m);
  return { n, probs, theoretical: blomQuantiles(m).slice(), sample: sampleQuantiles(sorted, probs), line: qqLineSorted(sorted) };
}

/**
 * Normal Q–Q plot data: at most `maxPoints` Blom plotting positions, their normal quantiles and the
 * sample's quantiles, plus R's qqline. Empty input gives empty arrays and a flat line at 0.
 */
export function qqNormal(values: ArrayLike<number>, maxPoints = 256): QQNormal {
  return qqNormalSorted(sortedFinite(values), maxPoints);
}

export interface QQTwoSample {
  probs: Float64Array;
  /** Quantiles of `b` (x axis). */
  x: Float64Array;
  /** Quantiles of `a` (y axis). */
  y: Float64Array;
  na: number;
  nb: number;
}

/** Two-sample Q–Q plot: quantiles of `a` (y) against quantiles of `b` (x) at shared Blom positions. */
export function qqTwoSample(a: ArrayLike<number>, b: ArrayLike<number>, maxPoints = 256): QQTwoSample {
  const sa = sortedFinite(a);
  const sb = sortedFinite(b);
  const m = Math.min(sa.length, sb.length, Math.max(1, Math.floor(maxPoints)));
  const probs = blomProbs(m);
  return { probs, x: sampleQuantiles(sb, probs), y: sampleQuantiles(sa, probs), na: sa.length, nb: sb.length };
}

export interface Moments {
  n: number;
  mean: number;
  std: number;
  skew: number;
  excessKurtosis: number;
  min: number;
  max: number;
}

/** Population moments (divide by n). Skew and kurtosis are 0 when the spread is 0; all 0 when empty. */
export function moments(values: ArrayLike<number>): Moments {
  let n = 0;
  let s = 0;
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (!Number.isFinite(v)) continue;
    n++;
    s += v;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (n === 0) return { n: 0, mean: 0, std: 0, skew: 0, excessKurtosis: 0, min: 0, max: 0 };
  const mean = s / n;
  let m2 = 0;
  let m3 = 0;
  let m4 = 0;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (!Number.isFinite(v)) continue;
    const d = v - mean;
    const d2 = d * d;
    m2 += d2;
    m3 += d2 * d;
    m4 += d2 * d2;
  }
  m2 /= n;
  m3 /= n;
  m4 /= n;
  // Spread below float noise counts as none, so constant data reads as skew 0 rather than garbage.
  const flat = m2 <= 1e-24 * Math.max(1, mean * mean);
  return {
    n,
    mean,
    std: Math.sqrt(m2),
    skew: flat ? 0 : m3 / Math.pow(m2, 1.5),
    excessKurtosis: flat ? 0 : m4 / (m2 * m2) - 3,
    min,
    max,
  };
}

/** Probability-plot correlation for an ascending sample; NaN when it is undefined (n < 2 or no spread). */
export function ppccSorted(sorted: Float64Array): number {
  const n = sorted.length;
  if (n < 2) return NaN;
  const q = blomQuantiles(n);
  let sx = 0;
  let sy = 0;
  for (let i = 0; i < n; i++) {
    sx += q[i];
    sy += sorted[i];
  }
  const mx = sx / n;
  const my = sy / n;
  let sxy = 0;
  let sxx = 0;
  let syy = 0;
  for (let i = 0; i < n; i++) {
    const dx = q[i] - mx;
    const dy = sorted[i] - my;
    sxy += dx * dy;
    sxx += dx * dx;
    syy += dy * dy;
  }
  if (!(syy > 0) || !(sxx > 0)) return NaN;
  return sxy / Math.sqrt(sxx * syy);
}

/**
 * Probability-plot correlation coefficient: Pearson's r between the sorted sample and the Blom
 * normal quantiles. 1 means a perfectly normal shape; NaN when undefined.
 */
export function ppcc(values: ArrayLike<number>): number {
  return ppccSorted(sortedFinite(values));
}

/** Share of the finite values within `eps` of zero. */
export function fractionAtZero(values: ArrayLike<number>, eps = 1e-9): number {
  let n = 0;
  let z = 0;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (!Number.isFinite(v)) continue;
    n++;
    if (Math.abs(v) <= eps) z++;
  }
  return n ? z / n : 0;
}

/** Counts per equal-width bin over [lo, hi]; values outside are clamped into the end bins. */
export function histogram(values: ArrayLike<number>, bins: number, lo: number, hi: number): Int32Array {
  const counts = new Int32Array(Math.max(1, bins));
  const span = hi - lo;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (!Number.isFinite(v)) continue;
    const k = span > 0 ? Math.floor(((v - lo) / span) * counts.length) : 0;
    counts[Math.min(counts.length - 1, Math.max(0, k))]++;
  }
  return counts;
}

/**
 * "Nice" axis ticks (1, 2 or 5 × 10^k apart) that lie inside [lo, hi], at most about `max` of them.
 * Returns the ticks and the step.
 */
export function niceTicks(lo: number, hi: number, max = 5): { ticks: number[]; step: number } {
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return { ticks: [], step: 1 };
  if (hi < lo) [lo, hi] = [hi, lo];
  if (hi - lo <= 0) return { ticks: [lo], step: Math.abs(lo) || 1 };
  const limit = Math.max(2, Math.floor(max));
  const count = (st: number) => Math.floor(hi / st + 1e-9) - Math.ceil(lo / st - 1e-9) + 1;
  // Smallest 1-2-5 step that yields at most `limit` ticks.
  let mag = Math.pow(10, Math.floor(Math.log10((hi - lo) / limit)) - 1);
  let step = mag;
  for (let guard = 0; guard < 40; guard++) {
    const found = [1, 2, 5].map((m) => m * mag).find((st) => count(st) <= limit);
    if (found !== undefined) {
      step = found;
      break;
    }
    mag *= 10;
  }
  const ticks: number[] = [];
  const first = Math.ceil(lo / step - 1e-9);
  const last = Math.floor(hi / step + 1e-9);
  for (let k = first; k <= last; k++) {
    const t = k * step;
    // Round away float noise (0.30000000000000004) and negative zero.
    const v = Number(t.toPrecision(12)) + 0;
    ticks.push(v);
  }
  return { ticks, step };
}

/** Everything a distribution panel shows, from one sort. */
export interface Summary {
  moments: Moments;
  qq: QQNormal;
  ppcc: number;
  zero: number;
}

export function summarize(values: ArrayLike<number>, maxPoints = 256): Summary {
  const sorted = sortedFinite(values);
  return { moments: moments(sorted), qq: qqNormalSorted(sorted, maxPoints), ppcc: ppccSorted(sorted), zero: fractionAtZero(sorted) };
}

const frozen = new WeakMap<object, Map<number, Summary>>();

/**
 * summarize() for arrays that never change after creation (the initial weights): the result is
 * cached per array, so redrawing every frame does not re-sort them.
 */
export function summarizeFrozen(values: Float32Array, maxPoints = 256): Summary {
  let byPoints = frozen.get(values);
  if (!byPoints) frozen.set(values, (byPoints = new Map()));
  let s = byPoints.get(maxPoints);
  if (!s) byPoints.set(maxPoints, (s = summarize(values, maxPoints)));
  return s;
}
