import { Network } from '../nn/network';
import type { Data } from './datasets';
import { featurize, type FeatureId } from './features';

/**
 * Evaluating a point-dataset network over a regular grid of input coordinates: the decision
 * boundary, the playground-style heatmap of every hidden unit, and the response maps in the
 * analysis views all come from here.
 */

/** Half-width of the square (cube) drawn around point data. Coordinates live in about [−1, 1]. */
export const DOMAIN = 1.25;

/**
 * The plotting domain of a point dataset: the symmetric box [−r, r] in every dimension, at least
 * DOMAIN, grown in steps of 0.25 to cover noisy points (up to 2).
 */
export function pointDomain(data: Data): number {
  const p = data.points;
  if (!p) return DOMAIN;
  let m = 0;
  for (const c of [p.trainCoords, p.testCoords]) for (let i = 0; i < c.length; i++) m = Math.max(m, Math.abs(c[i]));
  return Math.min(2, Math.max(DOMAIN, Math.ceil(m / 0.25) * 0.25));
}

/**
 * Raw coordinates of a res × res grid over two axes of a `dims`-dimensional domain [−r, r]:
 * `axes[0]` runs left to right, `axes[1]` bottom to top, and row 0 is the top row (as on a canvas).
 * Other axes (3-D data) are fixed at `fixed[axis]` (default 0). Cell centres, not edges.
 */
export function gridCoords(dims: 2 | 3, res: number, r: number, axes: [number, number] = [0, 1], fixed: number[] = []): Float32Array {
  const out = new Float32Array(res * res * dims);
  for (let row = 0; row < res; row++) {
    const v = r - ((row + 0.5) / res) * 2 * r;
    for (let col = 0; col < res; col++) {
      const u = -r + ((col + 0.5) / res) * 2 * r;
      const o = (row * res + col) * dims;
      for (let d = 0; d < dims; d++) out[o + d] = fixed[d] ?? 0;
      out[o + axes[0]] = u;
      out[o + axes[1]] = v;
    }
  }
  return out;
}

export interface GridResult {
  /** n × classes softmax probabilities. */
  probs: Float32Array;
  /** Per block (hidden layers, then the output logits): n × units activations. With `activations` only. */
  acts?: Float32Array[];
}

/**
 * Runs a private copy of a network over many points, so the page's network (whose intermediates
 * the views read for the current input) is never disturbed. Call `sync` whenever the weights or
 * architecture may have changed; it is cheap when nothing did.
 */
export class PointEvaluator {
  private net: Network | null = null;
  private key = '';

  sync(net: Network): void {
    const key = JSON.stringify(net.arch);
    if (!this.net || key !== this.key) {
      this.net = new Network(net.arch, 0);
      this.key = key;
    }
    const ws = net.getWeights();
    this.net.setWeights(ws);
  }

  /** Evaluates `coords` (n × dims raw coordinates), featurised with `features`. */
  evaluate(coords: Float32Array, dims: 2 | 3, features: FeatureId[], opts: { activations?: boolean } = {}): GridResult {
    const net = this.net;
    if (!net) throw new Error('PointEvaluator.sync() was not called');
    const n = coords.length / dims;
    const F = features.length;
    if (F !== net.inputSize) throw new Error(`The network takes ${net.inputSize} inputs; ${F} features were given`);
    const xs = featurize(coords, dims, features);
    const K = net.classes;
    const probs = new Float32Array(n * K);
    const acts = opts.activations ? net.blocks.map((b) => new Float32Array(n * b.out.length)) : undefined;
    const x = new Float32Array(F);
    for (let i = 0; i < n; i++) {
      x.set(xs.subarray(i * F, (i + 1) * F));
      const p = net.forward(x);
      probs.set(p, i * K);
      if (acts) net.blocks.forEach((b, j) => acts[j].set(b.out, i * b.out.length));
    }
    return { probs, acts };
  }
}

/** Index of the most likely class at every grid cell. */
export function argmaxRows(probs: Float32Array, classes: number): Uint8Array {
  const n = probs.length / classes;
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    let best = 0;
    for (let k = 1; k < classes; k++) if (probs[i * classes + k] > probs[i * classes + best]) best = k;
    out[i] = best;
  }
  return out;
}
