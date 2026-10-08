import { describe, expect, it } from 'vitest';
import { defaultFeatures, featureCatalog, featurize } from '../src/data/features';
import { SYNTHETIC, generate, type SyntheticId } from '../src/data/synthetic';
import { Network, argmax } from '../src/nn/network';
import { Optimizer } from '../src/nn/optim';

const clean = (id: SyntheticId, count = 400) => generate({ id, count, noise: 0, trainRatio: 0.5, seed: 3 });

describe('synthetic datasets', () => {
  it('are deterministic per seed and differ across seeds', () => {
    const a = generate({ id: 'spiral', count: 200, noise: 0.2, trainRatio: 0.5, seed: 1 });
    const b = generate({ id: 'spiral', count: 200, noise: 0.2, trainRatio: 0.5, seed: 1 });
    const c = generate({ id: 'spiral', count: 200, noise: 0.2, trainRatio: 0.5, seed: 2 });
    expect(a.train.coords).toEqual(b.train.coords);
    expect(a.train.coords).not.toEqual(c.train.coords);
  });

  for (const info of SYNTHETIC) {
    it(`${info.id}: sizes, split, balanced classes, coordinates in range`, () => {
      const d = clean(info.id, 600);
      expect(d.train.labels.length + d.test.labels.length).toBe(600);
      expect(d.train.labels.length).toBe(300);
      expect(d.train.coords.length).toBe(300 * info.dims);
      const counts = new Array(info.classes).fill(0);
      for (const l of [...d.train.labels, ...d.test.labels]) counts[l]++;
      for (const c of counts) expect(Math.abs(c - 600 / info.classes)).toBeLessThanOrEqual(1);
      for (const v of d.train.coords) expect(Math.abs(v)).toBeLessThanOrEqual(1.25);
    });
  }

  const rule = (id: SyntheticId, p: Float32Array, k: number, dims: number): number => {
    const x = p[k * dims];
    const y = p[k * dims + 1];
    const z = dims === 3 ? p[k * dims + 2] : 0;
    switch (id) {
      case 'circle':
        return Math.hypot(x, y) < 0.6 ? 0 : 1;
      case 'xor':
        return x * y > 0 ? 0 : 1;
      case 'checker':
        return (Math.floor((x + 1) * 2) + Math.floor((y + 1) * 2)) & 1;
      case 'shells':
        return Math.hypot(x, y, z) < 0.6 ? 0 : 1;
      case 'xor3':
        return x * y * z > 0 ? 0 : 1;
      default:
        return -1;
    }
  };
  for (const id of ['circle', 'xor', 'checker', 'shells', 'xor3'] as SyntheticId[]) {
    it(`${id}: without noise every label follows the dataset's rule`, () => {
      const d = clean(id);
      const dims = d.info.dims;
      for (let k = 0; k < d.train.labels.length; k++) expect(rule(id, d.train.coords, k, dims)).toBe(d.train.labels[k]);
    });
  }

  it('noise spreads the points', () => {
    const spread = (noise: number) => {
      const d = generate({ id: 'circle', count: 2000, noise, trainRatio: 0.5, seed: 1 });
      let m = 0;
      for (let k = 0; k < d.train.labels.length; k++) if (d.train.labels[k] === 0) m = Math.max(m, Math.hypot(d.train.coords[2 * k], d.train.coords[2 * k + 1]));
      return m;
    };
    expect(spread(0)).toBeLessThanOrEqual(0.5);
    expect(spread(0.4)).toBeGreaterThan(0.6);
  });
});

describe('features', () => {
  it('lists linear, square, product and sine features', () => {
    expect(featureCatalog(2).map((f) => f.id)).toEqual(['x1', 'x2', 'x1^2', 'x2^2', 'x1*x2', 'sin x1', 'sin x2']);
    expect(featureCatalog(3)).toHaveLength(12);
    expect(defaultFeatures(3)).toEqual(['x1', 'x2', 'x3']);
  });

  it('computes each feature', () => {
    const p = new Float32Array([0.5, -0.25]);
    const x = featurize(p, 2, ['x1', 'x2', 'x1^2', 'x2^2', 'x1*x2', 'sin x1', 'sin x2']);
    const want = [0.5, -0.25, 0.25, 0.0625, -0.125, Math.sin(Math.PI * 0.5), Math.sin(-Math.PI * 0.25)];
    want.forEach((w, i) => expect(x[i]).toBeCloseTo(w, 6));
    expect(() => featurize(p, 2, ['x3'])).toThrow();
  });

  /** A network with no hidden layer is a linear classifier on whatever features it gets. */
  const fitLinear = (features: string[]) => {
    const d = generate({ id: 'circle', count: 600, noise: 0.05, trainRatio: 0.5, seed: 1 });
    const X = featurize(d.train.coords, 2, features);
    const T = featurize(d.test.coords, 2, features);
    const F = features.length;
    const net = new Network({ input: { c: F, h: 1, w: 1 }, layers: [], classes: 2 }, 1);
    const opt = new Optimizer(net, 'adam', 0.05);
    for (let epoch = 0; epoch < 60; epoch++) {
      net.zeroGrad();
      for (let k = 0; k < d.train.labels.length; k++) {
        net.forward(X.subarray(k * F, (k + 1) * F));
        net.backward(d.train.labels[k]);
      }
      opt.step(1 / d.train.labels.length);
    }
    let ok = 0;
    for (let k = 0; k < d.test.labels.length; k++) if (argmax(net.forward(T.subarray(k * F, (k + 1) * F))) === d.test.labels[k]) ok++;
    return ok / d.test.labels.length;
  };

  it('circle: a linear model fails on raw coordinates but succeeds once squares are added', () => {
    expect(fitLinear(['x1', 'x2'])).toBeLessThan(0.75);
    expect(fitLinear(['x1', 'x2', 'x1^2', 'x2^2'])).toBeGreaterThan(0.95);
  });
});
