import type { Network } from './network';
import type { OptimizerName } from './types';

interface Slot {
  w: Float32Array;
  g: Float32Array;
  m: Float32Array | null;
  v: Float32Array | null;
  /** Updates applied to this slot so far: Adam's bias correction counts these (see `step`). */
  t: number;
}

export const DEFAULT_LR: Record<OptimizerName, number> = { sgd: 0.03, momentum: 0.01, adam: 0.003 };

export class Optimizer {
  private slots: Slot[] = [];

  constructor(
    net: Network,
    public name: OptimizerName,
    public lr: number,
    public momentum = 0.9,
    public beta1 = 0.9,
    public beta2 = 0.999,
    public eps = 1e-8,
  ) {
    for (const b of net.blocks) {
      this.slots.push(this.slot(b.W, b.gW), this.slot(b.b, b.gb));
    }
  }

  private slot(w: Float32Array, g: Float32Array): Slot {
    const needM = this.name !== 'sgd';
    const needV = this.name === 'adam';
    return { w, g, m: needM ? new Float32Array(w.length) : null, v: needV ? new Float32Array(w.length) : null, t: 0 };
  }

  /**
   * Applies one update with gradients scaled by `scale` (1 / batch size). Blocks flagged in
   * `frozen` (by block index) keep their weights.
   *
   * Adam divides its moment estimates by 1 − β^t, where t must count the updates those moments
   * have actually seen. The count is therefore kept per slot, not per call: a block frozen for
   * 300 steps still has zero moments, and dividing them by 1 − β^300 ≈ 1 instead of 1 − β made
   * its first updates after unfreezing 1.6 times the learning rate (up to 7 times after longer
   * frozen phases), so fine-tuning at a "lower" rate was not lower at all.
   */
  step(scale: number, frozen?: ArrayLike<boolean>): void {
    const lr = this.lr;
    const slots = frozen ? this.slots.filter((_, i) => !frozen[i >> 1]) : this.slots;
    for (const s of slots) s.t++;
    if (this.name === 'sgd') {
      for (const { w, g } of slots) for (let i = 0; i < w.length; i++) w[i] -= lr * g[i] * scale;
    } else if (this.name === 'momentum') {
      const mu = this.momentum;
      for (const { w, g, m } of slots) {
        for (let i = 0; i < w.length; i++) {
          m![i] = mu * m![i] + g[i] * scale;
          w[i] -= lr * m![i];
        }
      }
    } else {
      const { beta1: b1, beta2: b2, eps } = this;
      for (const { w, g, m, v, t } of slots) {
        const a = (lr * Math.sqrt(1 - Math.pow(b2, t))) / (1 - Math.pow(b1, t));
        for (let i = 0; i < w.length; i++) {
          const gi = g[i] * scale;
          m![i] = b1 * m![i] + (1 - b1) * gi;
          v![i] = b2 * v![i] + (1 - b2) * gi * gi;
          w[i] -= (a * m![i]) / (Math.sqrt(v![i]) + eps);
        }
      }
    }
  }

  /**
   * Forgets what the optimiser remembers about one block (Momentum's velocity; Adam's moments and
   * update count), so the block's next update is the one a fresh optimiser would make. Used when a
   * frozen block starts learning again: moments from before the freeze describe other weights and
   * gradients, and Momentum would otherwise replay its old velocity even on a zero gradient.
   */
  reset(block: number): void {
    for (const s of this.slots.slice(2 * block, 2 * block + 2)) {
      s.m?.fill(0);
      s.v?.fill(0);
      s.t = 0;
    }
  }
}
