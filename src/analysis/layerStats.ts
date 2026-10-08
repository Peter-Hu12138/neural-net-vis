import type { Act } from '../nn/types';
import { Rng } from '../nn/rng';
import type { Job } from './protocol';

/**
 * "layerStats": the raw material for the distribution plots. Runs `samples` test images through the
 * network and keeps, per layer, a fixed-size random sample of the pre-activations z and activations
 * a (pre-pool for conv layers), how often each unit fires, and the mean weight gradient of the
 * cross-entropy loss over those images.
 */

export interface LayerStatsParams {
  /** Test images to use, at evenly spaced indices. */
  samples?: number;
  /** Most values kept per layer and quantity (a uniform random subset when there are more). */
  maxValues?: number;
}

export interface LayerStatsLayer {
  block: number;
  act: Act;
  /** Pre-activations, sampled. Same positions as `a`, so a[i] = f(z[i]). */
  z: Float32Array;
  /** Activations (before max-pooling), sampled. */
  a: Float32Array;
  /** How many values the samples were drawn from: samples × units × positions. */
  seen: number;
  /** Mean ∂L/∂W and ∂L/∂b over the images. */
  gW: Float32Array;
  gb: Float32Array;
  /** Per unit (channel for conv): share of image × position entries with a > 0. */
  activeFraction: Float32Array;
  /** Units that never fired, for ReLU layers; null for other activations. */
  dead: number | null;
}

export interface LayerStatsResult {
  samples: number;
  layers: LayerStatsLayer[];
}

/** Fixed seed: the same weights always give the same sample. */
const SEED = 0x5eed;

export const layerStats: Job<LayerStatsParams, LayerStatsResult> = function* (ctx, params) {
  const { net, testY } = ctx;
  const N = testY.length;
  const samples = Math.max(1, Math.min(N, Math.floor(params?.samples ?? 256)));
  const maxValues = Math.max(1, Math.floor(params?.maxValues ?? 20000));
  const blocks = net.blocks;
  const rng = new Rng(SEED);

  // One reservoir per layer over the stream of (image, position) entries. z and a share slots.
  const res = blocks.map((b) => {
    const cap = Math.min(maxValues, samples * b.z.length);
    return { z: new Float32Array(cap), a: new Float32Array(cap), seen: 0 };
  });
  const active = blocks.map((b) => new Float64Array(b.kind === 'conv' ? b.spec.filters : b.spec.units));

  net.zeroGrad();
  const x = new Float32Array(784);
  for (let k = 0; k < samples; k++) {
    const idx = Math.floor((k * N) / samples);
    ctx.image(idx, x);
    net.forward(x);
    for (let bi = 0; bi < blocks.length; bi++) {
      const b = blocks[bi];
      const r = res[bi];
      const cap = r.z.length;
      const { z, a } = b;
      const L = z.length;
      for (let i = 0; i < L; i++) {
        const s = r.seen++;
        let slot = s;
        if (s >= cap) {
          slot = Math.floor(rng.next() * (s + 1));
          if (slot >= cap) continue;
        }
        r.z[slot] = z[i];
        r.a[slot] = a[i];
      }
      const act = active[bi];
      const per = L / act.length; // positions per unit: H×W for conv, 1 for dense
      for (let u = 0; u < act.length; u++) {
        let c = 0;
        for (let i = u * per; i < (u + 1) * per; i++) if (a[i] > 0) c++;
        act[u] += c;
      }
    }
    net.backward(testY[idx]);
    yield { done: k + 1, total: samples };
  }

  const layers: LayerStatsLayer[] = blocks.map((b, bi) => {
    const perUnit = (samples * b.z.length) / active[bi].length;
    const activeFraction = Float32Array.from(active[bi], (c) => c / perUnit);
    let dead: number | null = null;
    if (b.spec.act === 'relu') {
      dead = 0;
      for (const f of activeFraction) if (f === 0) dead++;
    }
    return {
      block: bi,
      act: b.spec.act,
      z: res[bi].z,
      a: res[bi].a,
      seen: res[bi].seen,
      gW: Float32Array.from(b.gW, (g) => g / samples),
      gb: Float32Array.from(b.gb, (g) => g / samples),
      activeFraction,
      dead,
    };
  });
  return { samples, layers };
};

export const jobs: Record<string, Job> = { layerStats };
