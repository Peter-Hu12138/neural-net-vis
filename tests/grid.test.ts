import { describe, expect, it } from 'vitest';
import { defaultPointsConfig, pointsData } from '../src/data/datasets';
import { DOMAIN, PointEvaluator, argmaxRows, gridCoords, pointDomain } from '../src/data/grid';
import { featurize } from '../src/data/features';
import { Network } from '../src/nn/network';

describe('grid helpers for point datasets', () => {
  it('gridCoords: cell centres, row 0 at the top, other axes fixed', () => {
    const g = gridCoords(2, 4, 1);
    // 4 × 4 cells over [−1, 1]: centres at ±0.25, ±0.75.
    expect(Array.from(g.subarray(0, 2))).toEqual([-0.75, 0.75]); // top-left
    expect(Array.from(g.subarray(3 * 2, 4 * 2))).toEqual([0.75, 0.75]); // top-right
    expect(Array.from(g.subarray(15 * 2, 16 * 2))).toEqual([0.75, -0.75]); // bottom-right
    const g3 = gridCoords(3, 2, 1, [0, 2], [0, 0.4, 0]);
    expect(g3[0]).toBe(-0.5);
    expect(g3[1]).toBeCloseTo(0.4, 6);
    expect(g3[2]).toBe(0.5);
  });

  it('pointDomain covers every point and is at least DOMAIN', () => {
    const d = pointsData({ ...defaultPointsConfig('circle'), noise: 0.5 });
    const r = pointDomain(d);
    expect(r).toBeGreaterThanOrEqual(DOMAIN);
    expect(r).toBeLessThanOrEqual(2);
    for (const c of [d.points!.trainCoords, d.points!.testCoords]) for (const v of c) expect(Math.abs(v)).toBeLessThanOrEqual(r === 2 ? Infinity : r);
    expect(pointDomain(pointsData({ ...defaultPointsConfig('circle'), noise: 0 }))).toBe(DOMAIN);
  });

  it('PointEvaluator matches the network and leaves the page network untouched', () => {
    const d = pointsData(defaultPointsConfig('blobs'), ['x1', 'x2', 'x1^2']);
    const net = new Network({ input: d.input, layers: [{ kind: 'dense', units: 5, act: 'tanh' }], classes: 3 }, 4);
    const probe = featurize(new Float32Array([0.1, -0.2]), 2, ['x1', 'x2', 'x1^2']);
    net.forward(probe);
    const before = Float32Array.from(net.blocks[0].out);
    const ev = new PointEvaluator();
    ev.sync(net);
    const coords = gridCoords(2, 8, 1.25);
    const { probs, acts } = ev.evaluate(coords, 2, ['x1', 'x2', 'x1^2'], { activations: true });
    expect(Array.from(net.blocks[0].out)).toEqual(Array.from(before)); // page network not disturbed
    const x = featurize(coords.subarray(2 * 10, 2 * 11), 2, ['x1', 'x2', 'x1^2']);
    const p = net.forward(x);
    for (let k = 0; k < 3; k++) expect(probs[10 * 3 + k]).toBeCloseTo(p[k], 6);
    expect(acts![0][10 * 5 + 2]).toBeCloseTo(net.blocks[0].out[2], 6);
    expect(acts![1].length).toBe(64 * 3);
    const cls = argmaxRows(probs, 3);
    expect(cls.length).toBe(64);
    expect(cls[10]).toBe(p.indexOf(Math.max(...p)));
  });

  it('PointEvaluator follows weight and architecture changes', () => {
    const arch = { input: { c: 2, h: 1, w: 1 }, layers: [], classes: 2 };
    const net = new Network(arch, 1);
    const ev = new PointEvaluator();
    ev.sync(net);
    const pt = new Float32Array([0.5, 0.5]);
    const a = ev.evaluate(pt, 2, ['x1', 'x2']).probs[0];
    net.output.W.fill(0);
    net.output.b.set([3, 0]);
    ev.sync(net);
    expect(ev.evaluate(pt, 2, ['x1', 'x2']).probs[0]).toBeCloseTo(1 / (1 + Math.exp(-3)), 6);
    expect(a).not.toBeCloseTo(1 / (1 + Math.exp(-3)), 3);
    const wide = new Network({ ...arch, layers: [{ kind: 'dense', units: 3, act: 'relu' }] }, 2);
    ev.sync(wide);
    expect(ev.evaluate(pt, 2, ['x1', 'x2'], { activations: true }).acts!.length).toBe(2);
    expect(() => ev.evaluate(pt, 2, ['x1'])).toThrow(/2 inputs/);
  });
});
