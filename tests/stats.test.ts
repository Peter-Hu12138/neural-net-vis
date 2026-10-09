import { describe, expect, it } from 'vitest';
import {
  blomProbs,
  fractionAtZero,
  histogram,
  moments,
  niceTicks,
  normalCdf,
  normalQuantile,
  ppcc,
  QQ_TAIL,
  qqNormal,
  qqRanks,
  qqTwoSample,
  qqTwoSampleSorted,
  quantileSorted,
  radixSortFloat32,
  referenceLineSorted,
  sortedFinite,
  sortedFrozen,
  summarize,
  summarizeFrozen,
  axisFormat,
  fixed,
  fractionAtZeroSorted,
  histogramSorted,
  momentsSorted,
  share,
  sig,
} from '../src/analysis/stats';
import { Rng } from '../src/nn/rng';

const normalSample = (n: number, mu: number, sigma: number, seed = 7) => {
  const rng = new Rng(seed);
  return Float64Array.from({ length: n }, () => mu + sigma * rng.normal());
};

describe('normalQuantile / normalCdf', () => {
  it('matches reference values', () => {
    expect(normalQuantile(0.5)).toBe(0);
    expect(Math.abs(normalQuantile(0.975) - 1.959963984540054)).toBeLessThan(1.2e-9 * 1.96);
    // Tighter than Acklam alone, thanks to the Halley step.
    expect(normalQuantile(0.975)).toBeCloseTo(1.959963984540054, 13);
    expect(normalQuantile(0.75)).toBeCloseTo(0.6744897501960817, 13);
    expect(normalQuantile(0.841344746068543)).toBeCloseTo(1, 12);
    expect(normalQuantile(1e-10)).toBeCloseTo(-6.361340902404056, 9);
    expect(normalQuantile(0.001)).toBeCloseTo(-3.090232306167813, 12);
    expect(normalCdf(0)).toBe(0.5);
    expect(normalCdf(1.959963984540054)).toBeCloseTo(0.975, 14);
    expect(normalCdf(-1)).toBeCloseTo(0.15865525393145707, 14);
    // Tails, relative error (reference values from R's pnorm).
    const rel = (x: number, ref: number) => Math.abs(normalCdf(x) - ref) / ref;
    expect(rel(-3, 0.0013498980316300946)).toBeLessThan(1e-13);
    expect(rel(-5, 2.866515718791939e-7)).toBeLessThan(1e-13);
    expect(rel(-7, 1.279812543885835e-12)).toBeLessThan(1e-13);
    expect(rel(-8, 6.220960574271785e-16)).toBeLessThan(1e-13);
    expect(rel(-10, 7.619853024160527e-24)).toBeLessThan(1e-13);
    expect(normalCdf(3)).toBeCloseTo(1 - 0.0013498980316300946, 15);
  });

  it('is antisymmetric, monotone and handles the ends', () => {
    for (const p of [0.001, 0.02, 0.1, 0.3, 0.49]) {
      expect(normalQuantile(1 - p)).toBeCloseTo(-normalQuantile(p), 12);
    }
    for (const p of [0.6, 0.9, 0.975, 0.999]) expect(normalQuantile(1 - p)).toBe(-normalQuantile(p));
    let prev = -Infinity;
    for (let p = 0.0005; p < 1; p += 0.0005) {
      const q = normalQuantile(p);
      expect(q).toBeGreaterThan(prev);
      prev = q;
    }
    expect(normalQuantile(0)).toBe(-Infinity);
    expect(normalQuantile(1)).toBe(Infinity);
    expect(normalQuantile(-0.1)).toBeNaN();
    expect(normalQuantile(1.1)).toBeNaN();
    expect(Number.isFinite(normalQuantile(1e-300))).toBe(true);
  });

  it('round-trips with normalCdf across the whole range', () => {
    let worst = 0;
    for (const p of [1e-12, 1e-8, 1e-5, 0.001, 0.0242, 0.0243, 0.1, 0.25, 0.5, 0.7, 0.9, 0.9757, 0.99, 0.999999]) {
      const rel = Math.abs(normalCdf(normalQuantile(p)) - p) / Math.min(p, 1 - p);
      worst = Math.max(worst, rel);
    }
    expect(worst).toBeLessThan(1e-9);
    // Above x ≈ 4, 1 − Φ(x) itself loses digits in double precision, so stop there.
    for (let x = -8; x <= 4; x += 0.25) expect(normalQuantile(normalCdf(x))).toBeCloseTo(x, 10);
  });
});

describe('quantileSorted (R type 7)', () => {
  it('interpolates between order statistics', () => {
    const s = [1, 2, 3, 4, 10];
    expect(quantileSorted(s, 0)).toBe(1);
    expect(quantileSorted(s, 1)).toBe(10);
    expect(quantileSorted(s, 0.5)).toBe(3);
    expect(quantileSorted(s, 0.25)).toBe(2);
    expect(quantileSorted(s, 0.9)).toBeCloseTo(4 + 0.6 * 6, 12); // h = 3.6
    expect(quantileSorted([5], 0.3)).toBe(5);
    expect(quantileSorted([], 0.3)).toBeNaN();
    // R: quantile(c(0, 10), 0.37) = 3.7
    expect(quantileSorted([0, 10], 0.37)).toBeCloseTo(3.7, 12);
  });
});

describe('qqNormal', () => {
  it('uses Blom positions and normal quantiles', () => {
    const q = qqNormal([3, 1, 2]);
    expect(Array.from(q.probs)).toEqual([0.625 / 3.25, 1.625 / 3.25, 2.625 / 3.25]);
    expect(q.theoretical[1]).toBe(0);
    expect(q.theoretical[0]).toBeCloseTo(-normalQuantile(2.625 / 3.25), 12);
    expect(Array.from(q.sample)).toEqual([1, 2, 3]); // every value plotted: the order statistics
    expect(q.n).toBe(3);
  });

  it('thins to maxPoints by plotting order statistics, always including the minimum and maximum', () => {
    const v = Float64Array.from({ length: 1001 }, (_, i) => i);
    const q = qqNormal(v, 11);
    expect(q.sample.length).toBe(11);
    expect(q.sample[0]).toBe(0); // the minimum, at its own Blom position among all 1001 values
    expect(q.sample[10]).toBe(1000); // the maximum
    expect(q.probs[0]).toBeCloseTo(0.625 / 1001.25, 15);
    expect(q.theoretical[0]).toBeCloseTo(normalQuantile(0.625 / 1001.25), 12);
    expect(q.theoretical[10]).toBe(-q.theoretical[0]);
    expect(q.sample[5]).toBe(500); // median
    // Every point is a point of the full Q–Q plot: the r-th order statistic at Blom position r.
    for (let j = 0; j < 11; j++) {
      const r = q.ranks[j];
      expect(q.sample[j]).toBe(r - 1);
      expect(q.probs[j]).toBeCloseTo((r - 0.375) / 1001.25, 15);
    }
  });

  it('keeps outliers and the whole extreme tail when it thins (review case: 10 of 25,088 weights at 1.0)', () => {
    const rng = new Rng(1);
    const v = Float32Array.from({ length: 25_088 }, () => 0.05 * rng.normal());
    for (let i = 0; i < 10; i++) v[i * 1000] = 1;
    const sorted = sortedFinite(v);
    for (const m of [256, 400]) {
      const q = qqNormal(v, m);
      expect(q.sample.length).toBe(m);
      expect(q.sample[0]).toBe(sorted[0]);
      expect(q.sample[m - 1]).toBe(1);
      // All ten outliers are drawn, and the x axis reaches the real extreme positions (≈ ±4.06).
      expect(Array.from(q.sample).filter((y) => y === 1)).toHaveLength(10);
      expect(q.theoretical[m - 1]).toBeCloseTo(normalQuantile((25_088 - 0.375) / 25_088.25), 12);
      expect(q.theoretical[m - 1]).toBeGreaterThan(4);
      // The QQ_TAIL most extreme order statistics on each side, every one of them.
      for (let j = 0; j < QQ_TAIL; j++) {
        expect(q.sample[j]).toBe(sorted[j]);
        expect(q.sample[m - 1 - j]).toBe(sorted[sorted.length - 1 - j]);
      }
    }
  });

  it('qqRanks: distinct ascending ranks from 1 to n, the extremes kept, no wide gaps along the normal axis', () => {
    const rng = new Rng(5);
    for (let t = 0; t < 300; t++) {
      const m = 2 + Math.floor(rng.next() * 400);
      const n = m + 1 + Math.floor(rng.next() ** 3 * 500_000);
      const r = qqRanks(n, m);
      expect(r.length).toBe(m);
      expect(r[0]).toBe(1);
      expect(r[m - 1]).toBe(n);
      for (let j = 1; j < m; j++) expect(r[j]).toBeGreaterThan(r[j - 1]);
      const tail = Math.min(QQ_TAIL, Math.floor(m / 8));
      for (let j = 0; j < tail; j++) {
        expect(r[j]).toBe(j + 1);
        expect(r[m - 1 - j]).toBe(n - j);
      }
      if (m >= 128) {
        // Half the points go at equal steps along the normal axis, so no gap that skips ranks spans
        // more than a few percent of it (neighbouring order statistics can be further apart).
        const x = Array.from(r, (k) => normalQuantile((k - 0.375) / (n + 0.25)));
        let gap = 0;
        for (let j = 1; j < m; j++) if (r[j] > r[j - 1] + 1) gap = Math.max(gap, x[j] - x[j - 1]);
        expect(gap / (x[m - 1] - x[0])).toBeLessThan(0.05);
      }
    }
    expect(Array.from(qqRanks(5, 256))).toEqual([1, 2, 3, 4, 5]);
    expect(Array.from(qqRanks(9, 1))).toEqual([5]);
    expect(Array.from(qqRanks(9, 2))).toEqual([1, 9]);
    expect(qqRanks(0, 10).length).toBe(0);
  });

  it('a large normal sample lies on its qqline with slope ≈ σ and intercept ≈ μ', () => {
    const v = normalSample(50_000, 1.5, 0.3);
    const q = qqNormal(v);
    expect(q.line.slope).toBeCloseTo(0.3, 2);
    expect(q.line.intercept).toBeCloseTo(1.5, 2);
    expect(q.reference).toMatchObject({ line: q.line, from: 'quartiles' });
    expect(ppcc(v)).toBeGreaterThan(0.999);
    // Every plotted point within |x| < 3 sits close to the line; the extreme order statistics are
    // noisier (the maximum of 50,000 normals has a standard deviation of about 0.28σ).
    let worst = 0;
    let worstTail = 0;
    for (let i = 0; i < q.sample.length; i++) {
      const d = Math.abs(q.sample[i] - (q.line.intercept + q.line.slope * q.theoretical[i]));
      if (Math.abs(q.theoretical[i]) < 3) worst = Math.max(worst, d);
      else worstTail = Math.max(worstTail, d);
    }
    expect(worst).toBeLessThan(0.05);
    expect(worstTail).toBeLessThan(0.3 * 0.3 * 4);
  });

  it('a uniform sample bends into an S and has a lower PPCC', () => {
    const rng = new Rng(3);
    const u = Float64Array.from({ length: 20_000 }, () => rng.next());
    const r = ppcc(u);
    expect(r).toBeLessThan(0.99);
    expect(r).toBeGreaterThan(0.95); // known asymptote ≈ 0.977
    const q = qqNormal(u);
    const fit = (i: number) => q.line.intercept + q.line.slope * q.theoretical[i];
    // Light tails: the ends fall inside the line (below at the top, above at the bottom).
    expect(q.sample[0]).toBeGreaterThan(fit(0));
    expect(q.sample[q.sample.length - 1]).toBeLessThan(fit(q.sample.length - 1));
    expect(q.sample[0]).toBeGreaterThanOrEqual(0);
    expect(q.sample[q.sample.length - 1]).toBeLessThanOrEqual(1);
  });

  it('heavy tails bend away from the line', () => {
    const rng = new Rng(11);
    // Laplace: excess kurtosis 3.
    const v = Float64Array.from({ length: 40_000 }, () => {
      const u = rng.next() - 0.5;
      return -Math.sign(u) * Math.log(1 - 2 * Math.abs(u));
    });
    const q = qqNormal(v);
    const last = q.sample.length - 1;
    expect(q.sample[last]).toBeGreaterThan(q.line.intercept + q.line.slope * q.theoretical[last]);
    expect(q.sample[0]).toBeLessThan(q.line.intercept + q.line.slope * q.theoretical[0]);
    expect(moments(v).excessKurtosis).toBeCloseTo(3, 0);
  });

  it('handles empty, single and constant data without NaN', () => {
    const e = qqNormal([]);
    expect(e.sample.length).toBe(0);
    expect(e.line).toEqual({ slope: 0, intercept: 0 });
    const one = qqNormal([4]);
    expect(Array.from(one.theoretical)).toEqual([0]);
    expect(Array.from(one.sample)).toEqual([4]);
    expect(one.line).toEqual({ slope: 0, intercept: 4 });
    const flat = qqNormal(new Float32Array(500).fill(2));
    expect(flat.line.slope).toBe(0);
    expect(flat.line.intercept).toBe(2);
    for (const a of [flat.theoretical, flat.sample, flat.probs]) for (const v of a) expect(Number.isFinite(v)).toBe(true);
    // Non-finite values are skipped.
    expect(qqNormal([1, NaN, 2, Infinity, 3]).n).toBe(3);
  });
});

describe('qqTwoSample', () => {
  it('pairs quantiles of a (y) with quantiles of b (x)', () => {
    const a = [10, 30, 20];
    const b = [3, 1, 2];
    const q = qqTwoSample(a, b);
    expect(Array.from(q.x)).toEqual([1, 2, 3]);
    expect(Array.from(q.y)).toEqual([10, 20, 30]);
    expect(Array.from(q.probs)).toEqual(Array.from(blomProbs(3)));
  });

  it('a scaled copy lies on y = s·x; same distributions lie on y = x', () => {
    const b = normalSample(4000, 0, 1, 5);
    const a = b.map((v) => 2 * v);
    const q = qqTwoSample(a, b, 64);
    expect(q.x.length).toBe(64);
    for (let i = 0; i < q.x.length; i++) expect(q.y[i]).toBeCloseTo(2 * q.x[i], 9);
    const q2 = qqTwoSample(normalSample(30_000, 0, 1, 8), normalSample(30_000, 0, 1, 9));
    // The bulk agrees closely; the extreme order statistics are noisy by nature.
    for (let i = 0; i < q2.x.length; i++) if (Math.abs(q2.x[i]) < 2.5) expect(Math.abs(q2.y[i] - q2.x[i])).toBeLessThan(0.08);
  });

  it('uses the smaller sample size and pairs both minima and both maxima', () => {
    const q = qqTwoSample([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4]);
    expect(q.x.length).toBe(4);
    expect(q.na).toBe(10);
    expect(q.nb).toBe(4);
    expect(Array.from(q.x)).toEqual([1, 2, 3, 4]);
    // The larger sample at the same relative ranks 0, 1/3, 2/3, 1 (R's qqplot): 1 and 10 included.
    expect(Array.from(q.y)).toEqual([1, 4, 7, 10]);
    expect(Array.from(q.probs)).toEqual(Array.from(blomProbs(4)));
  });

  it('thins both samples at the same ranks and keeps their extremes', () => {
    const rng = new Rng(2);
    const b = Float32Array.from({ length: 25_088 }, () => 0.05 * rng.normal());
    const a = b.map((w, i) => (i % 2500 === 0 ? 1 : 1.5 * w));
    const q = qqTwoSample(a, b);
    const sa = sortedFinite(a);
    const sb = sortedFinite(b);
    expect(q.x.length).toBe(256);
    expect(q.x[0]).toBe(sb[0]);
    expect(q.x[255]).toBe(sb[sb.length - 1]);
    expect(q.y[0]).toBe(sa[0]);
    expect(q.y[255]).toBe(1);
    const ranks = qqRanks(25_088, 256);
    for (let j = 0; j < 256; j++) {
      expect(q.x[j]).toBe(sb[ranks[j] - 1]);
      expect(q.y[j]).toBe(sa[ranks[j] - 1]);
    }
    // The pre-sorted variant (used with the cached initial weights) gives the same plot.
    const q2 = qqTwoSampleSorted(sa, sortedFrozen(b));
    expect(Array.from(q2.x)).toEqual(Array.from(q.x));
    expect(Array.from(q2.y)).toEqual(Array.from(q.y));
  });
});

describe('reference line', () => {
  it('is qqline when the quartiles differ', () => {
    const s = sortedFinite(normalSample(2000, 0, 1, 3));
    const ref = referenceLineSorted(s)!;
    expect(ref.from).toBe('quartiles');
    expect(ref.line).toEqual(qqNormal(s).line);
    expect(ref.q1).toBe(quantileSorted(s, 0.25));
    // Heavy tails keep qqline too (Laplace: IQR/1.349 ≈ 0.73 σ).
    const rng = new Rng(8);
    const lap = sortedFinite(Float64Array.from({ length: 20_000 }, () => {
      const u = rng.next() - 0.5;
      return -Math.sign(u) * Math.log(1 - 2 * Math.abs(u));
    }));
    expect(referenceLineSorted(lap)!.from).toBe('quartiles');
  });

  it('falls back to mean and std when half the values or more are equal (ReLU zeros), and is absent with no spread', () => {
    // 81% exact zeros, as in Conv 1's activations: Q1 = Q3 = 0, so qqline would be flat at 0.
    const rng = new Rng(4);
    const v = Float64Array.from({ length: 10_000 }, (_, i) => (i % 100 < 81 ? 0 : Math.abs(rng.normal())));
    const q = qqNormal(v);
    expect(q.line.slope).toBe(0);
    const mo = moments(v);
    expect(q.reference?.from).toBe('moments');
    expect(q.reference!.line.slope).toBeCloseTo(mo.std, 12);
    expect(q.reference!.line.intercept).toBeCloseTo(mo.mean, 12);
    // 72% zeros with Q3 well inside the positive values: qqline stays.
    const w = Float64Array.from({ length: 10_000 }, (_, i) => (i % 100 < 72 ? 0 : 1 + rng.next()));
    expect(qqNormal(w).reference?.from).toBe('quartiles');
    // Conv 1 after some training: 19% zeros, 60% at eight tiny bias values, the rest spread up to
    // ~20. Q1 and Q3 differ (0.002 vs 0.014) but qqline is flat against that spread.
    const conv = Float64Array.from({ length: 20_000 }, (_, i) => {
      const k = i % 100;
      if (k < 19) return 0;
      if (k < 79) return 0.0004 + 0.0018 * (k % 8);
      return 4 * Math.abs(rng.normal());
    });
    const cq = qqNormal(conv);
    expect(cq.reference!.q3).toBeGreaterThan(cq.reference!.q1);
    expect(cq.line.slope).toBeLessThan(0.1 * moments(conv).std);
    expect(cq.reference!.from).toBe('moments');
    expect(qqNormal(new Float32Array(500).fill(2)).reference).toBeNull();
    expect(qqNormal([4]).reference).toBeNull();
    expect(qqNormal([]).reference).toBeNull();
  });
});

describe('moments, ppcc, fractionAtZero', () => {
  it('computes population moments of known arrays', () => {
    const m = moments([2, 4, 4, 4, 5, 5, 7, 9]);
    expect(m.n).toBe(8);
    expect(m.mean).toBe(5);
    expect(m.std).toBe(2);
    expect(m.min).toBe(2);
    expect(m.max).toBe(9);
    // Σd³ = −27 −1 −1 −1 +0 +0 +8 +64 = 42 → m3 = 5.25; skew = 5.25 / 8 = 0.65625
    expect(m.skew).toBeCloseTo(0.65625, 12);
    // Σd⁴ = 81+1+1+1+0+0+16+256 = 356 → m4 = 44.5; kurtosis = 44.5/16 − 3
    expect(m.excessKurtosis).toBeCloseTo(44.5 / 16 - 3, 12);
    const sym = moments([-1, 0, 1]);
    expect(sym.skew).toBe(0);
    expect(sym.excessKurtosis).toBeCloseTo(1 / (2 / 3) ** 2 * (2 / 3) - 3, 12); // m4 = 2/3, m2 = 2/3
    expect(moments([])).toEqual({ n: 0, mean: 0, std: 0, skew: 0, excessKurtosis: 0, min: 0, max: 0 });
    const c = moments([3, 3, 3]);
    expect(c).toMatchObject({ n: 3, mean: 3, std: 0, skew: 0, excessKurtosis: 0 });
    const big = moments(normalSample(100_000, -2, 0.5, 21));
    expect(big.mean).toBeCloseTo(-2, 2);
    expect(big.std).toBeCloseTo(0.5, 2);
    expect(Math.abs(big.skew)).toBeLessThan(0.03);
    expect(Math.abs(big.excessKurtosis)).toBeLessThan(0.06);
  });

  it('ppcc is 1 for exact normal quantiles, undefined for no spread', () => {
    const q = Array.from(blomProbs(200), normalQuantile);
    expect(ppcc(q)).toBeCloseTo(1, 14);
    expect(ppcc(q.map((v) => 3 * v - 1).reverse())).toBeCloseTo(1, 14); // order and scale don't matter
    expect(ppcc([1, 1, 1])).toBeNaN();
    expect(ppcc([5])).toBeNaN();
    // Brute force against a direct Pearson computation.
    const v = [0.3, -1.2, 2.2, 0.1, 0.05, -0.4, 1.7];
    const s = [...v].sort((a, b) => a - b);
    const t = Array.from(blomProbs(v.length), normalQuantile);
    const mean = (a: number[]) => a.reduce((x, y) => x + y, 0) / a.length;
    const ms = mean(s);
    const mt = mean(t);
    let sxy = 0;
    let sxx = 0;
    let syy = 0;
    for (let i = 0; i < v.length; i++) {
      sxy += (s[i] - ms) * (t[i] - mt);
      sxx += (t[i] - mt) ** 2;
      syy += (s[i] - ms) ** 2;
    }
    expect(ppcc(v)).toBeCloseTo(sxy / Math.sqrt(sxx * syy), 12);
  });

  it('fractionAtZero counts exact (and near-exact) zeros', () => {
    expect(fractionAtZero([0, 0, 1, -2])).toBe(0.5);
    expect(fractionAtZero([1e-12, 1e-3], 1e-9)).toBe(0.5);
    expect(fractionAtZero([])).toBe(0);
  });
});

describe('histogram, niceTicks, summarize', () => {
  it('bins values and clamps the ends', () => {
    expect(Array.from(histogram([0, 0.1, 0.5, 0.99, 1, 2, -1], 4, 0, 1))).toEqual([3, 0, 1, 3]);
    expect(Array.from(histogram([5, 5], 3, 5, 5))).toEqual([2, 0, 0]);
  });

  it('niceTicks lie inside the range at 1-2-5 steps', () => {
    const t = niceTicks(-2.86, 2.86, 5);
    expect(t.step).toBe(1);
    expect(t.ticks).toEqual([-2, -1, 0, 1, 2]);
    const u = niceTicks(-0.0123, 0.031, 5);
    expect(u.step).toBe(0.01);
    expect(u.ticks).toEqual([-0.01, 0, 0.01, 0.02, 0.03]);
    expect(niceTicks(-0.0123, 0.031, 4).step).toBe(0.02);
    for (const [lo, hi] of [[0.13, 0.91], [-1e-5, 3e-5], [100, 1234]]) {
      const r = niceTicks(lo, hi, 5);
      expect(r.ticks.length).toBeGreaterThanOrEqual(1);
      for (const v of r.ticks) {
        expect(v).toBeGreaterThanOrEqual(lo);
        expect(v).toBeLessThanOrEqual(hi);
      }
    }
    expect(Object.is(niceTicks(-1, 1, 3).ticks[1], 0)).toBe(true); // never −0
    expect(niceTicks(-1, 1, 2).ticks).toEqual([-1, 0, 1]); // step 2 would leave only 0
    expect(Object.is(niceTicks(-1, 1, 2).ticks[1], 0)).toBe(true);
  });

  it('niceTicks never returns a single tick for a real range (review: ±4.8 and ±0.47 with max 4)', () => {
    expect(niceTicks(-4.8, 4.8, 4)).toEqual({ ticks: [-2.5, 0, 2.5], step: 2.5 });
    expect(niceTicks(-0.47, 0.47, 4)).toEqual({ ticks: [-0.25, 0, 0.25], step: 0.25 });
    const rng = new Rng(9);
    let fallbacks = 0;
    const bad: string[] = [];
    const SUPS = '⁻⁰¹²³⁴⁵⁶⁷⁸⁹';
    for (let t = 0; t < 100_000; t++) {
      const mag = Math.pow(10, Math.floor(rng.next() * 10) - 5);
      const a = (rng.next() * 20 - 10) * mag;
      const b = a + rng.next() * 10 * mag + 1e-12;
      const max = 3 + Math.floor(rng.next() * 4);
      const { ticks, step } = niceTicks(a, b, max);
      const lead = step / Math.pow(10, Math.floor(Math.log10(step) + 1e-9));
      if (Math.abs(lead - 2.5) < 1e-9) fallbacks++;
      // Every label reads back as its tick value, 2.5 steps included.
      const ax = axisFormat(step, Math.max(Math.abs(a), Math.abs(b)));
      const e = ax.suffix ? Number(Array.from(ax.suffix.slice(4), (c) => '-0123456789'[SUPS.indexOf(c)]).join('')) : 0;
      const labels = ticks.map(ax.fmt);
      const ok =
        ticks.length >= 2 &&
        ticks.length <= max + 1 &&
        ticks.every((v) => v >= a - 1e-9 * mag && v <= b + 1e-9 * mag) &&
        [1, 2, 2.5, 5].some((f) => Math.abs(lead - f) < 1e-9) &&
        new Set(labels).size === labels.length &&
        ticks.every((v, i) => Math.abs(Number(labels[i].replace('−', '-')) * 10 ** e - v) <= 1e-9 * Math.max(Math.abs(v), step));
      if (!ok && bad.length < 5) bad.push(`[${a}, ${b}] max ${max}: step ${step} ticks ${ticks} labels ${labels}${ax.suffix}`);
    }
    expect(bad).toEqual([]);
    expect(fallbacks).toBeGreaterThan(0);
  });

  it('the sorted fast paths (moments, zeros, histogram) match the general functions exactly', () => {
    const rng = new Rng(12);
    const v = Float32Array.from({ length: 30_000 }, (_, i) => (i % 5 === 0 ? 0 : i % 7 === 0 ? 1e-10 : rng.normal() ** 3));
    v[17] = NaN;
    v[18] = Infinity;
    const s = sortedFinite(v);
    expect(momentsSorted(s)).toEqual(moments(s));
    expect(fractionAtZeroSorted(s)).toBe(fractionAtZero(v));
    expect(fractionAtZeroSorted(s, 1e-12)).toBe(fractionAtZero(v, 1e-12));
    for (const [bins, lo, hi] of [[40, s[0], s[s.length - 1]], [7, -1, 1], [64, 0, 0], [1, -3, 3], [33, 0.5, 0.2]] as const) {
      expect(Array.from(histogramSorted(s, bins, lo, hi))).toEqual(Array.from(histogram(v, bins, lo, hi)));
    }
    expect(momentsSorted(new Float64Array(0)).n).toBe(0);
    expect(fractionAtZeroSorted(new Float64Array(0))).toBe(0);
  });

  it('summarize matches the separate functions', () => {
    const v = normalSample(3000, 0.2, 1.1, 13);
    const s = summarize(v, 100);
    const m = moments(v);
    for (const k of Object.keys(m) as (keyof typeof m)[]) expect(s.moments[k]).toBeCloseTo(m[k], 12);
    expect(s.ppcc).toBe(ppcc(v));
    expect(Array.from(s.qq.sample)).toEqual(Array.from(qqNormal(v, 100).sample));
    expect(s.zero).toBe(0);
  });
});

describe('sorting', () => {
  it('radix sort matches the built-in sort, including signs, zeros and extremes', () => {
    const rng = new Rng(17);
    const v = Float32Array.from({ length: 20_000 }, (_, i) => {
      if (i % 997 === 0) return 0;
      if (i % 991 === 0) return -0;
      if (i % 983 === 0) return 3.4e38;
      if (i % 977 === 0) return -1e-40; // subnormal
      return (rng.next() - 0.5) * 10 ** Math.floor(rng.next() * 8 - 4);
    });
    const ref = v.slice().sort();
    const got = radixSortFloat32(v);
    expect(got.length).toBe(ref.length);
    for (let i = 0; i < ref.length; i++) expect(got[i] === ref[i] || (got[i] === 0 && ref[i] === 0)).toBe(true);
    expect(Array.from(radixSortFloat32(new Float32Array([2, -1, 0.5, -3])))).toEqual([-3, -1, 0.5, 2]);
  });

  it('sortedFinite drops NaN and infinities on both paths', () => {
    const big = new Float32Array(1000).map((_, i) => (i % 10 === 0 ? NaN : i % 13 === 0 ? -Infinity : 500 - i));
    const s = sortedFinite(big);
    expect(s.length).toBe(Array.from(big).filter(Number.isFinite).length);
    expect(s.length).toBe(831); // 100 NaN; 77 multiples of 13, 8 of them also multiples of 10
    for (let i = 1; i < s.length; i++) expect(s[i]).toBeGreaterThanOrEqual(s[i - 1]);
    expect(Array.from(sortedFinite([3, NaN, -1, Infinity]))).toEqual([-1, 3]);
  });

  it('summarizeFrozen and sortedFrozen cache by array', () => {
    const v = Float32Array.from({ length: 500 }, (_, i) => Math.sin(i));
    const a = summarizeFrozen(v, 100);
    expect(summarizeFrozen(v, 100)).toBe(a);
    expect(summarizeFrozen(v, 50)).not.toBe(a);
    expect(a.ppcc).toBe(summarize(v, 100).ppcc);
    expect(sortedFrozen(v)).toBe(sortedFrozen(v)); // sorted once
    expect(Array.from(sortedFrozen(v))).toEqual(Array.from(sortedFinite(v)));
    expect(sortedFrozen(v.slice())).not.toBe(sortedFrozen(v));
  });
});

describe('number formatting', () => {
  it('fixed never prints −0.00 (review: skew −2.94e−3, kurtosis −3.59e−3)', () => {
    expect(fixed(-2.94e-3, 2)).toBe('0.00');
    expect(fixed(-3.59e-3, 2)).toBe('0.00');
    expect(fixed(-9.41e-4, 2)).toBe('0.00');
    expect(fixed(-0, 2)).toBe('0.00');
    expect(fixed(-0.006, 2)).toBe('−0.01');
    expect(fixed(-0.46, 2)).toBe('−0.46');
    expect(fixed(0.99977, 4)).toBe('0.9998');
    expect(fixed(NaN, 2)).toBe('—');
  });

  it('sig keeps trailing zeros so a column lines up (review: 0.45 next to 0.442)', () => {
    expect(sig(0.45)).toBe('0.450');
    expect(sig(0.4503)).toBe('0.450');
    expect(sig(0.442)).toBe('0.442');
    expect(sig(0.0565)).toBe('0.0565');
    expect(sig(-1.5)).toBe('−1.50');
    expect(sig(999.7)).toBe('1.00e3');
    expect(sig(2.5e-5)).toBe('2.50e−5');
    expect(sig(0)).toBe('0.00');
  });

  it('sig stays in plain decimals down to 1e-4, so a column of means does not switch to exponents (review NEW-5)', () => {
    expect(sig(-4.2e-4)).toBe('−0.000420');
    expect(sig(0.0907)).toBe('0.0907');
    expect(sig(-0.00319)).toBe('−0.00319');
    expect(sig(1.7)).toBe('1.70');
    expect(sig(0.47)).toBe('0.470');
    expect(sig(1e-4)).toBe('0.000100');
    expect(sig(9.99e-5)).toBe('9.99e−5');
    expect(sig(-0.00004)).toBe('−4.00e−5');
    // Never a "−0.00…" for a value that rounds to nothing.
    expect(sig(-0)).toBe('0.00');
  });

  it('share shows a non-empty sliver as <0.1% instead of 0.0% (review: 2 of 8,192)', () => {
    expect(share(2, 8192)).toBe('<0.1%');
    expect(share(0, 8192)).toBe('0.0%');
    expect(share(8190, 8192)).toBe('>99.9%');
    expect(share(8192, 8192)).toBe('100.0%');
    expect(share(1540, 8192)).toBe('18.8%');
  });
});
