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
  qqNormal,
  qqTwoSample,
  quantileSorted,
  radixSortFloat32,
  sortedFinite,
  summarize,
  summarizeFrozen,
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

  it('down-samples to maxPoints with type-7 quantiles', () => {
    const v = Float64Array.from({ length: 1001 }, (_, i) => i);
    const q = qqNormal(v, 11);
    expect(q.sample.length).toBe(11);
    expect(q.probs[0]).toBeCloseTo(0.625 / 11.25, 15);
    expect(q.sample[5]).toBeCloseTo(500, 9); // median
    expect(q.sample[0]).toBeCloseTo(1000 * q.probs[0], 9);
  });

  it('a large normal sample lies on its qqline with slope ≈ σ and intercept ≈ μ', () => {
    const v = normalSample(50_000, 1.5, 0.3);
    const q = qqNormal(v);
    expect(q.line.slope).toBeCloseTo(0.3, 2);
    expect(q.line.intercept).toBeCloseTo(1.5, 2);
    expect(ppcc(v)).toBeGreaterThan(0.999);
    // Every plotted point sits close to the line.
    let worst = 0;
    for (let i = 0; i < q.sample.length; i++) worst = Math.max(worst, Math.abs(q.sample[i] - (q.line.intercept + q.line.slope * q.theoretical[i])));
    expect(worst).toBeLessThan(0.05);
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
    for (let i = 10; i < q2.x.length - 10; i++) expect(Math.abs(q2.y[i] - q2.x[i])).toBeLessThan(0.08);
  });

  it('uses the smaller sample size', () => {
    const q = qqTwoSample([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4]);
    expect(q.x.length).toBe(4);
    expect(q.na).toBe(10);
    expect(q.nb).toBe(4);
    expect(Array.from(q.x)).toEqual([1, 2, 3, 4]);
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
    expect(Object.is(niceTicks(-1, 1, 2).ticks[0], 0)).toBe(true);
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

  it('summarizeFrozen caches by array', () => {
    const v = Float32Array.from({ length: 500 }, (_, i) => Math.sin(i));
    const a = summarizeFrozen(v, 100);
    expect(summarizeFrozen(v, 100)).toBe(a);
    expect(summarizeFrozen(v, 50)).not.toBe(a);
    expect(a.ppcc).toBe(summarize(v, 100).ppcc);
  });
});
