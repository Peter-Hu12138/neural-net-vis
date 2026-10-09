/**
 * Distribution statistics for the Q–Q plots and histograms, and the number formatting of their
 * labels. Pure functions, no DOM, so they run in the analysis worker, on the page and in the unit
 * tests alike. Non-finite values (a diverged network can produce NaN weights) are skipped everywhere.
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
  if (kept instanceof Float32Array) return new Float64Array(radixSortFloat32(kept));
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

/** Order statistics always plotted at each end when a Q–Q plot is thinned. */
export const QQ_TAIL = 16;

interface QQGrid {
  /** 1-based ranks of the plotted order statistics, ascending. */
  ranks: Int32Array;
  /** Their Blom plotting positions (r − 3/8)/(n + 1/4) among all n values. */
  probs: Float64Array;
  /** Φ⁻¹ of those positions. */
  theoretical: Float64Array;
}

const gridCache = new Map<string, QQGrid>();

/**
 * Which order statistics a Q–Q plot of `n` values draws when it can show at most `maxPoints`.
 * Every one when n ≤ maxPoints. Otherwise the smallest and largest `QQ_TAIL` values (so the
 * minimum, the maximum and the extreme tail are always there), and between them half the points at
 * equal steps of rank and half at equal steps along the normal axis, so the tails are not thinned
 * to a handful of points. Where those steps fall closer than one rank apart, consecutive ranks are
 * taken. Every plotted point is one of the n points of the full Q–Q plot.
 */
export function qqRanks(n: number, maxPoints: number): Int32Array {
  return qqGrid(n, maxPoints).ranks.slice();
}

function qqGrid(n: number, maxPoints: number): QQGrid {
  n = Math.max(0, Math.floor(n));
  const m = Math.min(n, Math.max(1, Math.floor(maxPoints)));
  const key = `${n}:${m}`;
  const hit = gridCache.get(key);
  if (hit) return hit;
  const ranks = new Int32Array(m);
  if (m === n) for (let j = 0; j < m; j++) ranks[j] = j + 1;
  else if (m === 1) ranks[0] = Math.round((n + 1) / 2);
  else {
    const tail = Math.min(QQ_TAIL, Math.floor(m / 8));
    for (let j = 0; j < tail; j++) {
      ranks[j] = j + 1;
      ranks[m - 1 - j] = n - j;
    }
    // The other points cover ranks lo…hi. G(r) ∈ [0, 1] is the share of them at or below rank r:
    // half spread evenly over the ranks, half evenly along the normal axis.
    const inner = m - 2 * tail;
    const lo = tail + 1;
    const hi = n - tail;
    if (inner === 1) ranks[tail] = Math.round((lo + hi) / 2);
    else {
      const xr = (r: number) => normalQuantile((r - 0.375) / (n + 0.25));
      const xlo = xr(lo);
      const xspan = xr(hi) - xlo;
      const G = (r: number) => 0.5 * ((r - lo) / (hi - lo)) + 0.5 * ((xr(r) - xlo) / xspan);
      for (let j = 0; j < inner; j++) {
        const g = j / (inner - 1) - 1e-12;
        let a = lo;
        let b = hi;
        while (a < b) {
          const mid = (a + b) >>> 1;
          if (G(mid) >= g) b = mid;
          else a = mid + 1;
        }
        ranks[tail + j] = a;
      }
    }
    // Distinct ranks: push up where the grid is denser than the data, then pull back from n.
    for (let j = 1; j < m; j++) ranks[j] = Math.max(ranks[j], ranks[j - 1] + 1);
    ranks[m - 1] = n;
    for (let j = m - 2; j >= 0; j--) ranks[j] = Math.min(ranks[j], ranks[j + 1] - 1);
  }
  const probs = new Float64Array(m);
  const theoretical = new Float64Array(m);
  for (let j = 0; j < m; j++) {
    const r = ranks[j];
    probs[j] = (r - 0.375) / (n + 0.25);
    // The upper half mirrors the lower one exactly, so the x axis is symmetric.
    theoretical[j] = 2 * r > n + 1 ? -normalQuantile((n + 1 - r - 0.375) / (n + 0.25)) : normalQuantile(probs[j]);
  }
  const grid = { ranks, probs, theoretical };
  if (gridCache.size >= 32) gridCache.delete(gridCache.keys().next().value!);
  gridCache.set(key, grid);
  return grid;
}

export interface Line {
  slope: number;
  intercept: number;
}

/** The dashed reference line of a normal Q–Q plot and where it comes from. */
export interface Reference {
  line: Line;
  /**
   * 'quartiles': R's qqline, the normal through the first and third quartiles.
   * 'moments': the normal with the sample's mean and standard deviation, used when the quartiles
   * (nearly) coincide, as when most values are ReLU zeros, where qqline would be flat.
   */
  from: 'quartiles' | 'moments';
  /** The sample's first and third quartiles (type 7). */
  q1: number;
  q3: number;
}

/**
 * qqline gives way to the mean/std line when its slope, IQR / 1.349 (a robust estimate of σ), is
 * below this share of the standard deviation. For normal data the two agree (ratio ≈ 1); for a
 * Laplace distribution the ratio is 0.73, for ReLU of a normal 0.86. It falls below 0.1 only when
 * the middle half of the values sits at (nearly) one value.
 */
export const FLAT_QUARTILES = 0.1;

export interface QQNormal {
  /** Number of finite values the plot summarises. */
  n: number;
  /** 1-based ranks of the plotted order statistics among the n values. */
  ranks: Int32Array;
  /** Blom plotting positions of those ranks among all n values. */
  probs: Float64Array;
  /** Φ⁻¹(p_i): the x coordinates. */
  theoretical: Float64Array;
  /** The order statistics at those ranks: the y coordinates. */
  sample: Float64Array;
  /** R's qqline: the normal distribution through the sample's quartiles. */
  line: Line;
  /** The line to draw: qqline, or the mean/std line when the quartiles coincide; null when there is no spread at all. */
  reference: Reference | null;
}

/** qqline for a sorted sample: slope = IQR / normal IQR, through the first quartile. */
export function qqLineSorted(sorted: ArrayLike<number>): Line {
  if (sorted.length === 0) return { slope: 0, intercept: 0 };
  const q1 = quantileSorted(sorted, 0.25);
  const q3 = quantileSorted(sorted, 0.75);
  const slope = (q3 - q1) / (Q3_NORMAL - Q1_NORMAL);
  return { slope, intercept: q1 - slope * Q1_NORMAL };
}

/**
 * The reference line for a sorted sample: qqline, unless the quartiles (nearly) coincide (see
 * FLAT_QUARTILES). Then a line through them is flat and says nothing about normality, so it is the
 * normal with the same mean and standard deviation (intercept = mean, slope = std). Null when every
 * value is the same (or there are fewer than two). `mo` saves a pass when the moments are known.
 */
export function referenceLineSorted(sorted: ArrayLike<number>, mo: Moments = moments(sorted)): Reference | null {
  const n = sorted.length;
  if (n < 2 || !(sorted[n - 1] > sorted[0]) || !(mo.std > 0)) return null;
  const line = qqLineSorted(sorted);
  const q1 = quantileSorted(sorted, 0.25);
  const q3 = quantileSorted(sorted, 0.75);
  if (line.slope >= FLAT_QUARTILES * mo.std) return { line, from: 'quartiles', q1, q3 };
  return { line: { slope: mo.std, intercept: mo.mean }, from: 'moments', q1, q3 };
}

export function qqNormalSorted(sorted: Float64Array, maxPoints = 256, mo?: Moments): QQNormal {
  const n = sorted.length;
  const grid = qqGrid(n, maxPoints);
  const sample = new Float64Array(grid.ranks.length);
  for (let j = 0; j < sample.length; j++) sample[j] = sorted[grid.ranks[j] - 1];
  return {
    n,
    ranks: grid.ranks.slice(),
    probs: grid.probs.slice(),
    theoretical: grid.theoretical.slice(),
    sample,
    line: qqLineSorted(sorted),
    reference: referenceLineSorted(sorted, mo),
  };
}

/**
 * Normal Q–Q plot data: the sorted values against normal quantiles at their Blom plotting
 * positions, thinned to at most `maxPoints` order statistics by qqRanks (the extremes are always
 * kept), plus R's qqline and the reference line to draw. Empty input gives empty arrays.
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

/**
 * Two-sample Q–Q plot from sorted samples: quantiles of `a` (y) against quantiles of `b` (x).
 * Points follow the smaller sample's order statistics (thinned by qqRanks, so both minima and both
 * maxima are always paired). A sample of that same size contributes its order statistics directly;
 * a larger one is interpolated at the same relative rank (r − 1)/(n − 1), as R's qqplot does, so its
 * minimum and maximum land on the ends too. `probs` are the Blom positions of the smaller sample.
 */
export function qqTwoSampleSorted(sa: ArrayLike<number>, sb: ArrayLike<number>, maxPoints = 256): QQTwoSample {
  const ns = Math.min(sa.length, sb.length);
  const grid = qqGrid(ns, maxPoints);
  const at = (s: ArrayLike<number>, r: number) => (s.length === ns ? s[r - 1] : quantileSorted(s, ns > 1 ? (r - 1) / (ns - 1) : 0.5));
  const x = new Float64Array(grid.ranks.length);
  const y = new Float64Array(grid.ranks.length);
  for (let j = 0; j < x.length; j++) {
    x[j] = at(sb, grid.ranks[j]);
    y[j] = at(sa, grid.ranks[j]);
  }
  return { probs: grid.probs.slice(), x, y, na: sa.length, nb: sb.length };
}

/** Two-sample Q–Q plot: quantiles of `a` (y) against quantiles of `b` (x). See qqTwoSampleSorted. */
export function qqTwoSample(a: ArrayLike<number>, b: ArrayLike<number>, maxPoints = 256): QQTwoSample {
  return qqTwoSampleSorted(sortedFinite(a), sortedFinite(b), maxPoints);
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

/** Moments of finite values sorted ascending (sortedFinite's output): same as moments(), in tight loops. */
export function momentsSorted(sorted: Float64Array): Moments {
  const n = sorted.length;
  if (n === 0) return { n: 0, mean: 0, std: 0, skew: 0, excessKurtosis: 0, min: 0, max: 0 };
  let s = 0;
  for (let i = 0; i < n; i++) s += sorted[i];
  const mean = s / n;
  let m2 = 0;
  let m3 = 0;
  let m4 = 0;
  for (let i = 0; i < n; i++) {
    const d = sorted[i] - mean;
    const d2 = d * d;
    m2 += d2;
    m3 += d2 * d;
    m4 += d2 * d2;
  }
  m2 /= n;
  m3 /= n;
  m4 /= n;
  const flat = m2 <= 1e-24 * Math.max(1, mean * mean);
  return {
    n,
    mean,
    std: Math.sqrt(m2),
    skew: flat ? 0 : m3 / Math.pow(m2, 1.5),
    excessKurtosis: flat ? 0 : m4 / (m2 * m2) - 3,
    min: sorted[0],
    max: sorted[n - 1],
  };
}

/** First index in [0, n] whose value satisfies `pred`, for a predicate that is monotone along the array. */
function firstIndex(n: number, pred: (i: number) => boolean): number {
  let lo = 0;
  let hi = n;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (pred(mid)) hi = mid;
    else lo = mid + 1;
  }
  return lo;
}

/** fractionAtZero() for finite values sorted ascending, by binary search. */
export function fractionAtZeroSorted(sorted: Float64Array, eps = 1e-9): number {
  const n = sorted.length;
  if (!n) return 0;
  return (firstIndex(n, (i) => sorted[i] > eps) - firstIndex(n, (i) => sorted[i] >= -eps)) / n;
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
 * histogram() for finite values sorted ascending: the bin index never decreases along the array,
 * so each bin's boundary is found by binary search (bins × log n instead of n). Same counts.
 */
export function histogramSorted(sorted: Float64Array, bins: number, lo: number, hi: number): Int32Array {
  const counts = new Int32Array(Math.max(1, bins));
  const B = counts.length;
  const span = hi - lo;
  const bin = (v: number) => Math.min(B - 1, Math.max(0, span > 0 ? Math.floor(((v - lo) / span) * B) : 0));
  let start = 0;
  for (let k = 0; k < B; k++) {
    const end = firstIndex(sorted.length, (i) => bin(sorted[i]) > k);
    counts[k] = end - start;
    start = end;
  }
  return counts;
}

/**
 * "Nice" axis ticks (1, 2 or 5 × 10^k apart) that lie inside [lo, hi], at most about `max` of them,
 * and at least two whenever hi > lo (then the step may be 2.5 × 10^k and the count max + 1).
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
  if (count(step) < 2) {
    // A range just inside two multiples of the step (±4.8 at step 5) would get a single tick and
    // no readable scale: take the largest smaller step, with 2.5 added to the ladder, that gives two.
    const e = Math.floor(Math.log10(step) + 1e-9);
    search: for (let k = e; k > e - 3; k--) {
      for (const f of [5, 2.5, 2, 1]) {
        const st = Number((f * Math.pow(10, k)).toPrecision(12));
        if (st < step && count(st) >= 2) {
          step = st;
          break search;
        }
      }
    }
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

/** summarize() for values already sorted ascending and finite (sortedFinite's output). */
export function summarizeSorted(sorted: Float64Array, maxPoints = 256): Summary {
  const mo = momentsSorted(sorted);
  return { moments: mo, qq: qqNormalSorted(sorted, maxPoints, mo), ppcc: ppccSorted(sorted), zero: fractionAtZeroSorted(sorted) };
}

export function summarize(values: ArrayLike<number>, maxPoints = 256): Summary {
  return summarizeSorted(sortedFinite(values), maxPoints);
}

const frozenSorted = new WeakMap<object, Float64Array>();
const frozen = new WeakMap<object, Map<number, Summary>>();

/**
 * sortedFinite() for arrays that never change after creation (the initial weights): sorted once
 * per array, then served from a cache. The result must not be modified.
 */
export function sortedFrozen(values: Float32Array): Float64Array {
  let s = frozenSorted.get(values);
  if (!s) frozenSorted.set(values, (s = sortedFinite(values)));
  return s;
}

/**
 * summarize() for arrays that never change after creation (the initial weights): the result is
 * cached per array, so redrawing every frame does not re-sort them.
 */
export function summarizeFrozen(values: Float32Array, maxPoints = 256): Summary {
  let byPoints = frozen.get(values);
  if (!byPoints) frozen.set(values, (byPoints = new Map()));
  let s = byPoints.get(maxPoints);
  if (!s) byPoints.set(maxPoints, (s = summarizeSorted(sortedFrozen(values), maxPoints)));
  return s;
}

// ── Number formatting for the plots and their stats ─────────────────────────

const SUP: Record<string, string> = { '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };
const minus = (s: string) => s.replace(/^-/, '−').replace(/e-/, 'e−');

/** A value for tooltips and stats: three significant digits, exponent form when tiny or huge. */
export function num(v: number, digits = 3): string {
  if (!Number.isFinite(v)) return '—';
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1e-3 && a < 1e5) return minus(String(Number(v.toPrecision(digits))));
  return minus(v.toExponential(digits - 1).replace('e+', 'e'));
}

/** `v` to `d` decimals with a true minus sign; values that round to zero print as 0 (never "−0.00"). */
export function fixed(v: number, d: number): string {
  if (!Number.isFinite(v)) return '—';
  const s = v.toFixed(d);
  return /^-0\.?0*$/.test(s) ? s.slice(1) : minus(s);
}

/**
 * `v` to `digits` significant digits, keeping trailing zeros ("0.450"), so values in one column
 * line up. Plain decimals down to 1e-4 ("−0.000420"); exponent form when tinier or huge.
 */
export function sig(v: number, digits = 3): string {
  if (!Number.isFinite(v)) return '—';
  const a = Math.abs(v);
  if (a === 0 || (a >= 1e-4 && a < Math.pow(10, digits))) {
    const s = v.toPrecision(digits);
    if (!s.includes('e')) return /^-0\.?0*$/.test(s) ? s.slice(1) : minus(s);
  }
  return minus(v.toExponential(digits - 1).replace('e+', 'e'));
}

/** Percentage of `part` in `total` to one decimal; a non-empty share below 0.05% reads "<0.1%". */
export function share(part: number, total: number): string {
  if (!(total > 0)) return '—';
  const p = (100 * part) / total;
  if (part > 0 && p < 0.05) return '<0.1%';
  if (part < total && p >= 99.95) return '>99.9%';
  return `${p.toFixed(1)}%`;
}

/** Tick label formatter for an axis with ticks `step` apart; tiny or huge ranges get a ×10ⁿ factor. */
export function axisFormat(step: number, maxAbs: number): { fmt: (v: number) => string; suffix: string } {
  const e = maxAbs > 0 ? Math.floor(Math.log10(maxAbs)) : 0;
  const scale = e <= -3 || e >= 5 ? Math.pow(10, e) : 1;
  const unit = step / scale;
  let d = Math.max(0, Math.ceil(-Math.log10(unit) - 1e-9));
  // A 2.5 × 10^k step needs one decimal more than its magnitude suggests.
  while (d < 12 && Math.abs(unit * 10 ** d - Math.round(unit * 10 ** d)) > 1e-6 * unit * 10 ** d) d++;
  const fmt = (v: number) => minus((v / scale).toFixed(d)).replace(/^−(0\.?0*)$/, '$1');
  const suffix = scale === 1 ? '' : ` ×10${String(e).split('').map((c) => SUP[c]).join('')}`;
  return { fmt, suffix };
}
