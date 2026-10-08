import type { Act } from './types';

export const LEAKY_ALPHA = 0.1;

/** a = f(z), elementwise. */
export function activate(act: Act, z: Float32Array, a: Float32Array): void {
  const n = z.length;
  switch (act) {
    case 'relu':
      for (let i = 0; i < n; i++) a[i] = z[i] > 0 ? z[i] : 0;
      break;
    case 'leaky':
      for (let i = 0; i < n; i++) a[i] = z[i] > 0 ? z[i] : LEAKY_ALPHA * z[i];
      break;
    case 'tanh':
      for (let i = 0; i < n; i++) a[i] = Math.tanh(z[i]);
      break;
    case 'sigmoid':
      for (let i = 0; i < n; i++) a[i] = 1 / (1 + Math.exp(-z[i]));
      break;
    case 'linear':
      a.set(z);
      break;
  }
}

/**
 * dZ = dA ⊙ f'(z). Uses the stored activation `a` where that is cheaper.
 * At the ReLU kink (z exactly 0) training uses the left slope; `symmetric` uses the midpoint of the
 * two one-sided slopes instead, which analyses prefer (blank MNIST pixels sit exactly on the kink).
 */
export function activateBackward(act: Act, z: Float32Array, a: Float32Array, dA: Float32Array, dZ: Float32Array, symmetric = false): void {
  const n = z.length;
  switch (act) {
    case 'relu':
      if (symmetric) for (let i = 0; i < n; i++) dZ[i] = z[i] > 0 ? dA[i] : z[i] < 0 ? 0 : 0.5 * dA[i];
      else for (let i = 0; i < n; i++) dZ[i] = z[i] > 0 ? dA[i] : 0;
      break;
    case 'leaky':
      if (symmetric) for (let i = 0; i < n; i++) dZ[i] = z[i] > 0 ? dA[i] : z[i] < 0 ? LEAKY_ALPHA * dA[i] : 0.5 * (1 + LEAKY_ALPHA) * dA[i];
      else for (let i = 0; i < n; i++) dZ[i] = z[i] > 0 ? dA[i] : LEAKY_ALPHA * dA[i];
      break;
    case 'tanh':
      for (let i = 0; i < n; i++) dZ[i] = dA[i] * (1 - a[i] * a[i]);
      break;
    case 'sigmoid':
      for (let i = 0; i < n; i++) dZ[i] = dA[i] * a[i] * (1 - a[i]);
      break;
    case 'linear':
      dZ.set(dA);
      break;
  }
}

/** f'(z) for a single value, for worked examples in the backprop view. */
export function derivative(act: Act, z: number): number {
  switch (act) {
    case 'relu':
      return z > 0 ? 1 : 0;
    case 'leaky':
      return z > 0 ? 1 : LEAKY_ALPHA;
    case 'tanh': {
      const t = Math.tanh(z);
      return 1 - t * t;
    }
    case 'sigmoid': {
      const s = 1 / (1 + Math.exp(-z));
      return s * (1 - s);
    }
    case 'linear':
      return 1;
  }
}

export function derivativeFormula(act: Act): string {
  switch (act) {
    case 'relu':
      return "f′(z) = 1 if z > 0, else 0";
    case 'leaky':
      return `f′(z) = 1 if z > 0, else ${LEAKY_ALPHA}`;
    case 'tanh':
      return "f′(z) = 1 − tanh²(z) = 1 − a²";
    case 'sigmoid':
      return "f′(z) = σ(z)(1 − σ(z)) = a(1 − a)";
    case 'linear':
      return "f′(z) = 1";
  }
}
