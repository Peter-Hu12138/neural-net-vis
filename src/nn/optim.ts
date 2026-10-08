import type { Network } from './network';
import type { OptimizerName } from './types';

interface Slot {
  w: Float32Array;
  g: Float32Array;
  m: Float32Array | null;
  v: Float32Array | null;
}

export const DEFAULT_LR: Record<OptimizerName, number> = { sgd: 0.03, momentum: 0.01, adam: 0.003 };

export class Optimizer {
  private slots: Slot[] = [];
  private t = 0;

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
    return { w, g, m: needM ? new Float32Array(w.length) : null, v: needV ? new Float32Array(w.length) : null };
  }

  /**
   * Applies one update with gradients scaled by `scale` (1 / batch size). Blocks flagged in
   * `frozen` (by block index) keep their weights.
   */
  step(scale: number, frozen?: ArrayLike<boolean>): void {
    this.t++;
    const lr = this.lr;
    const slots = frozen ? this.slots.filter((_, i) => !frozen[i >> 1]) : this.slots;
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
      const c1 = 1 - Math.pow(b1, this.t);
      const c2 = 1 - Math.pow(b2, this.t);
      const a = (lr * Math.sqrt(c2)) / c1;
      for (const { w, g, m, v } of slots) {
        for (let i = 0; i < w.length; i++) {
          const gi = g[i] * scale;
          m![i] = b1 * m![i] + (1 - b1) * gi;
          v![i] = b2 * v![i] + (1 - b2) * gi * gi;
          w[i] -= (a * m![i]) / (Math.sqrt(v![i]) + eps);
        }
      }
    }
  }
}
