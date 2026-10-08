import { describe as suite, expect, it } from 'vitest';
import { Network, describe } from '../src/nn/network';
import { Optimizer } from '../src/nn/optim';
import { Rng } from '../src/nn/rng';
import type { Act, LayerSpec } from '../src/nn/types';

function randomInput(rng: Rng): Float32Array {
  const x = new Float32Array(784);
  for (let i = 0; i < x.length; i++) x[i] = rng.next();
  return x;
}

/** Relative error between analytic and central-difference gradients over sampled parameters. */
function gradCheck(spec: LayerSpec[], seed = 1): { params: number; input: number } {
  const rng = new Rng(seed + 100);
  const net = new Network(spec, seed);
  const x = randomInput(rng);
  const label = 3;
  net.zeroGrad();
  net.forward(x);
  net.backward(label, true);
  const inputGrad = net.blocks[0].dX.slice();
  const eps = 1e-2;
  const lossAt = () => {
    net.forward(x);
    return net.loss(label);
  };

  let num = 0;
  let den = 0;
  for (const b of net.blocks) {
    for (const [w, g] of [
      [b.W, b.gW],
      [b.b, b.gb],
    ] as const) {
      for (let s = 0; s < 12; s++) {
        const i = rng.int(w.length);
        const orig = w[i];
        w[i] = orig + eps;
        const lp = lossAt();
        w[i] = orig - eps;
        const lm = lossAt();
        w[i] = orig;
        const numeric = (lp - lm) / (2 * eps);
        num += (numeric - g[i]) ** 2;
        den += numeric ** 2 + g[i] ** 2;
      }
    }
  }
  let inum = 0;
  let iden = 0;
  for (let s = 0; s < 30; s++) {
    const i = rng.int(784);
    const orig = x[i];
    x[i] = orig + eps;
    const lp = lossAt();
    x[i] = orig - eps;
    const lm = lossAt();
    x[i] = orig;
    const numeric = (lp - lm) / (2 * eps);
    inum += (numeric - inputGrad[i]) ** 2;
    iden += numeric ** 2 + inputGrad[i] ** 2;
  }
  return { params: Math.sqrt(num / (den + 1e-20)), input: Math.sqrt(inum / (iden + 1e-20)) };
}

suite('gradients', () => {
  const smooth: Act[] = ['tanh', 'sigmoid', 'linear'];
  for (const act of smooth) {
    it(`conv + pool + dense with ${act} match finite differences`, () => {
      const r = gradCheck([
        { kind: 'conv', filters: 3, kernel: 3, act, pool: true },
        { kind: 'conv', filters: 2, kernel: 5, act, pool: false },
        { kind: 'dense', units: 6, act },
      ]);
      expect(r.params).toBeLessThan(2e-2);
      expect(r.input).toBeLessThan(2e-2);
    });
  }

  for (const act of ['relu', 'leaky'] as Act[]) {
    it(`${act} network matches finite differences`, () => {
      const r = gradCheck([
        { kind: 'conv', filters: 4, kernel: 3, act, pool: true },
        { kind: 'dense', units: 8, act },
      ], 7);
      expect(r.params).toBeLessThan(5e-2);
      expect(r.input).toBeLessThan(5e-2);
    });
  }

  it('dense-only network matches finite differences', () => {
    const r = gradCheck([
      { kind: 'dense', units: 16, act: 'tanh' },
      { kind: 'dense', units: 8, act: 'sigmoid' },
    ]);
    expect(r.params).toBeLessThan(2e-2);
  });
});

suite('shapes', () => {
  it('propagates shapes through pooling and flattening', () => {
    const info = describe([
      { kind: 'conv', filters: 8, kernel: 3, act: 'relu', pool: true },
      { kind: 'conv', filters: 16, kernel: 3, act: 'relu', pool: true },
      { kind: 'dense', units: 32, act: 'relu' },
    ]);
    expect(info.map((l) => [l.outShape.h, l.outShape.w, l.outShape.c])).toEqual([
      [14, 14, 8],
      [7, 7, 16],
      [1, 1, 32],
      [1, 1, 10],
    ]);
    expect(info[2].params).toBe(7 * 7 * 16 * 32 + 32);
  });

  it('flags pooling a 1×1 map', () => {
    const spec: LayerSpec[] = Array.from({ length: 6 }, () => ({ kind: 'conv', filters: 1, kernel: 3, act: 'relu', pool: true }));
    const info = describe(spec);
    expect(info.some((l) => l.error)).toBe(true);
  });
});

suite('training', () => {
  for (const opt of ['sgd', 'momentum', 'adam'] as const) {
    it(`${opt} lowers the loss on a toy problem`, () => {
      const rng = new Rng(5);
      const xs = Array.from({ length: 20 }, () => randomInput(rng));
      const ys = xs.map((_, i) => i % 10);
      const net = new Network([{ kind: 'conv', filters: 2, kernel: 3, act: 'relu', pool: true }, { kind: 'dense', units: 16, act: 'relu' }], 3);
      const o = new Optimizer(net, opt, { sgd: 0.1, momentum: 0.02, adam: 0.003 }[opt]);
      const total = () => xs.reduce((s, x, i) => (net.forward(x), s + net.loss(ys[i])), 0);
      const before = total();
      for (let e = 0; e < 30; e++) {
        net.zeroGrad();
        xs.forEach((x, i) => {
          net.forward(x);
          net.backward(ys[i]);
        });
        o.step(1 / xs.length);
      }
      expect(total()).toBeLessThan(before * 0.5);
    });
  }
});

suite('weights', () => {
  const spec: LayerSpec[] = [{ kind: 'conv', filters: 4, kernel: 5, act: 'relu', pool: true }, { kind: 'dense', units: 8, act: 'sigmoid' }];

  it('the same seed builds the same network, a different seed does not', () => {
    const a = new Network(spec, 9).getWeights();
    const b = new Network(spec, 9).getWeights();
    const c = new Network(spec, 10).getWeights();
    expect(a).toEqual(b);
    expect(a[0]).not.toEqual(c[0]);
  });

  it('round-trips weights between two copies (worker ↔ page)', () => {
    const src = new Network(spec, 1);
    const dst = new Network(spec, 2);
    dst.setWeights(src.getWeights());
    const x = randomInput(new Rng(3));
    expect(Array.from(dst.forward(x))).toEqual(Array.from(src.forward(x)));
  });

  it('outputs a probability distribution', () => {
    const net = new Network(spec, 4);
    const p = net.forward(randomInput(new Rng(8)));
    expect(p.reduce((s, v) => s + v, 0)).toBeCloseTo(1, 5);
    expect(Math.min(...p)).toBeGreaterThan(0);
    expect(net.paramCount).toBe(describe(spec).reduce((s, l) => s + l.params, 0));
  });
});

suite('input gradients', () => {
  const spec: LayerSpec[] = [
    { kind: 'conv', filters: 3, kernel: 3, act: 'tanh', pool: true },
    { kind: 'conv', filters: 4, kernel: 5, act: 'sigmoid', pool: false },
    { kind: 'dense', units: 6, act: 'tanh' },
  ];

  /** Finite-difference check of ∂(seed·z_block)/∂x for random seeds at every block. */
  for (const block of [0, 1, 2, 3]) {
    it(`matches finite differences for a seed at block ${block}`, () => {
      const net = new Network(spec, 21);
      const rng = new Rng(block + 1);
      const x = randomInput(rng);
      const zLen = net.blocks[block].z.length;
      const seed = Float32Array.from({ length: zLen }, () => rng.normal());
      const objective = () => {
        net.forward(x);
        const z = net.blocks[block].z;
        let s = 0;
        for (let i = 0; i < zLen; i++) s += seed[i] * z[i];
        return s;
      };
      objective();
      const g = net.inputGradient(block, seed).slice();
      const before = net.blocks.map((b) => b.gW.slice());
      let num = 0;
      let den = 0;
      for (let t = 0; t < 40; t++) {
        const i = rng.int(784);
        const orig = x[i];
        // ε = 0.003: large enough for float32, small enough not to flip max-pool winners.
        x[i] = orig + 3e-3;
        const p = objective();
        x[i] = orig - 3e-3;
        const m = objective();
        x[i] = orig;
        const numeric = (p - m) / 6e-3;
        num += (numeric - g[i]) ** 2;
        den += numeric ** 2 + g[i] ** 2;
      }
      expect(Math.sqrt(num / den)).toBeLessThan(1e-2);
      // Parameter gradients are left alone.
      net.blocks.forEach((b, j) => expect(b.gW).toEqual(before[j]));
    });
  }

  it('agrees with the loss gradient at the logits', () => {
    const net = new Network(spec, 5);
    const x = randomInput(new Rng(9));
    net.zeroGrad();
    const p = net.forward(x).slice();
    net.backward(4, true);
    const viaLoss = net.blocks[0].dX.slice();
    net.forward(x);
    const seed = Float32Array.from(p, (v, k) => v - (k === 4 ? 1 : 0));
    const viaSeed = net.inputGradient(net.blocks.length - 1, seed);
    for (let i = 0; i < 784; i += 37) expect(viaSeed[i]).toBeCloseTo(viaLoss[i], 5);
  });
});
