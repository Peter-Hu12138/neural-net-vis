import type { Act } from '../nn/types';
import { Rng } from '../nn/rng';
import type { Job } from './protocol';

/**
 * "layerStats": the raw material for the distribution plots. Runs `samples` test images through the
 * network and keeps, per layer, a fixed-size random sample of the pre-activations z and activations
 * a (pre-pool for conv layers) and the mean weight gradient of the cross-entropy loss over those
 * images. Then, if some ReLU unit has not fired on those images, it runs the rest of the test set
 * forward only, so "dead" means silent on every test digit rather than on a sample of them (a unit
 * that is silent on 256 digits may fire on a few of the 2,000). It stops as soon as every such unit
 * has fired. While that scan runs, progress updates carry the finished distributions as `partial`
 * (with `complete: false`), so they can be shown before the dead-unit count is final.
 */

export interface LayerStatsParams {
  /** Test images to use, at evenly spaced indices. */
  samples?: number;
  /** Most values kept per layer and quantity (a uniform random subset when there are more). */
  maxValues?: number;
  /**
   * Test images, at evenly spaced indices, that the scan for dead units may cover (the `samples`
   * images are always among them). Default: the whole test set.
   */
  activityImages?: number;
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
  /**
   * Share of those values whose whole input is zero (a blank patch of background for a conv layer),
   * so z equals the unit's bias exactly. On MNIST this is most of the first conv layer.
   */
  blank: number;
  /** Mean ∂L/∂W and ∂L/∂b over the images. */
  gW: Float32Array;
  gb: Float32Array;
  /** Per unit (channel for conv): share of image × position entries with a > 0, over `activityImages` images. */
  activeFraction: Float32Array;
  /** Units that never fired on any of the `activityImages` images, for ReLU layers; null for other activations. */
  dead: number | null;
}

export interface LayerStatsResult {
  samples: number;
  /** Test images the firing counts (activeFraction, dead) cover. */
  activityImages: number;
  /** False only in the `partial` sent while the dead-unit scan is still running. */
  complete: boolean;
  layers: LayerStatsLayer[];
}

/** Fixed seed: the same weights always give the same sample. */
const SEED = 0x5eed;

/** Progress weight of an image that also runs backward, relative to a forward-only one. */
const BACKWARD_COST = 3;

/** How long the partial result rides along on progress updates (the analyzer posts every ~100 ms). */
const PARTIAL_MS = 250;

export const layerStats: Job<LayerStatsParams, LayerStatsResult> = function* (ctx, params) {
  const { net, testY } = ctx;
  const N = testY.length;
  const samples = Math.max(1, Math.min(N, Math.floor(params?.samples ?? 256)));
  const maxValues = Math.max(1, Math.floor(params?.maxValues ?? 20000));
  const activityN = Math.max(1, Math.min(N, Math.floor(params?.activityImages ?? N)));
  const blocks = net.blocks;
  const rng = new Rng(SEED);

  const sampleIdx = Array.from({ length: samples }, (_, k) => Math.floor((k * N) / samples));
  // The scan covers evenly spaced images (all of them by default) that were not sampled.
  const inSample = new Uint8Array(N);
  for (const i of sampleIdx) inSample[i] = 1;
  const inScan = new Uint8Array(N);
  for (let k = 0; k < activityN; k++) inScan[Math.floor((k * N) / activityN)] = 1;
  const extra: number[] = [];
  for (let i = 0; i < N; i++) if (inScan[i] && !inSample[i]) extra.push(i);
  const total = samples * BACKWARD_COST + extra.length;

  // One reservoir per layer over the stream of (image, position) entries. z and a share slots.
  const res = blocks.map((b) => {
    const cap = Math.min(maxValues, samples * b.z.length);
    return { z: new Float32Array(cap), a: new Float32Array(cap), seen: 0, blank: 0 };
  });
  const active = blocks.map((b) => new Float64Array(b.kind === 'conv' ? b.spec.filters : b.spec.units));

  /** Adds the last forward pass's firing counts. */
  const countActive = () => {
    for (let bi = 0; bi < blocks.length; bi++) {
      const a = blocks[bi].a;
      const act = active[bi];
      const per = a.length / act.length; // positions per unit: H×W for conv, 1 for dense
      for (let u = 0; u < act.length; u++) {
        let c = 0;
        for (let i = u * per; i < (u + 1) * per; i++) if (a[i] > 0) c++;
        act[u] += c;
      }
    }
  };
  /** ReLU units that have not fired yet. */
  const silent = () => {
    let n = 0;
    blocks.forEach((b, bi) => {
      if (b.spec.act === 'relu') for (const c of active[bi]) if (c === 0) n++;
    });
    return n;
  };

  net.zeroGrad();
  const x = new Float32Array(ctx.inputSize);
  for (let k = 0; k < samples; k++) {
    const idx = sampleIdx[k];
    ctx.image(idx, x);
    net.forward(x);
    for (let bi = 0; bi < blocks.length; bi++) {
      const b = blocks[bi];
      const r = res[bi];
      const cap = r.z.length;
      const { z, a } = b;
      const L = z.length;
      const per = L / b.b.length;
      for (let i = 0; i < L; i++) {
        // z starts at the bias and adds w·x, so with an all-zero input it is exactly the bias.
        if (z[i] === b.b[Math.floor(i / per)]) r.blank++;
        const s = r.seen++;
        let slot = s;
        if (s >= cap) {
          slot = Math.floor(rng.next() * (s + 1));
          if (slot >= cap) continue;
        }
        r.z[slot] = z[i];
        r.a[slot] = a[i];
      }
    }
    countActive();
    net.backward(testY[idx]);
    yield { done: (k + 1) * BACKWARD_COST, total };
  }

  const gW = blocks.map((b) => Float32Array.from(b.gW, (g) => g / samples));
  const gb = blocks.map((b) => Float32Array.from(b.gb, (g) => g / samples));
  const build = (images: number, complete: boolean): LayerStatsResult => ({
    samples,
    activityImages: images,
    complete,
    layers: blocks.map((b, bi) => {
      const perUnit = (images * b.z.length) / active[bi].length;
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
        blank: res[bi].seen ? res[bi].blank / res[bi].seen : 0,
        gW: gW[bi],
        gb: gb[bi],
        activeFraction,
        dead,
      };
    }),
  });

  // Forward-only scan of the other test images, while some ReLU unit is still silent.
  let scanned = 0;
  if (extra.length && silent() > 0) {
    const partial = build(samples, false);
    const until = performance.now() + PARTIAL_MS;
    while (scanned < extra.length) {
      net.forward(ctx.image(extra[scanned], x));
      countActive();
      scanned++;
      if (silent() === 0) break;
      const progress = { done: samples * BACKWARD_COST + scanned, total };
      yield performance.now() < until ? { ...progress, partial } : progress;
    }
  }
  return build(samples + scanned, true);
};

export const jobs: Record<string, Job> = { layerStats };
