import { Rng } from '../nn/rng';

/**
 * Small synthetic classification datasets in the spirit of the TensorFlow Playground: points in
 * [−1, 1]² or [−1, 1]³ whose classes need curved or folded decision boundaries. Every generator
 * is seeded, so the same settings always give the same points.
 */

export type SyntheticId =
  | 'circle'
  | 'xor'
  | 'gauss'
  | 'spiral'
  | 'moons'
  | 'blobs'
  | 'checker'
  | 'shells'
  | 'helix'
  | 'xor3'
  | 'blobs3';

export interface SyntheticInfo {
  id: SyntheticId;
  name: string;
  dims: 2 | 3;
  classes: number;
  description: string;
}

export const SYNTHETIC: SyntheticInfo[] = [
  { id: 'circle', name: 'Circle', dims: 2, classes: 2, description: 'A disc inside a ring. No straight line separates them.' },
  { id: 'xor', name: 'XOR', dims: 2, classes: 2, description: 'Opposite quadrants share a class: the classic problem a single layer cannot solve.' },
  { id: 'gauss', name: 'Two blobs', dims: 2, classes: 2, description: 'Two Gaussian clouds. A straight line is enough.' },
  { id: 'spiral', name: 'Spiral', dims: 2, classes: 2, description: 'Two interleaved spirals: hard, needs depth or good features.' },
  { id: 'moons', name: 'Moons', dims: 2, classes: 2, description: 'Two interlocking half-moons.' },
  { id: 'blobs', name: 'Three blobs', dims: 2, classes: 3, description: 'Three Gaussian clouds, three classes.' },
  { id: 'checker', name: 'Checkerboard', dims: 2, classes: 2, description: 'A 4×4 checkerboard: many small regions.' },
  { id: 'shells', name: 'Shells', dims: 3, classes: 2, description: 'A ball inside a spherical shell.' },
  { id: 'helix', name: 'Double helix', dims: 3, classes: 2, description: 'Two intertwined helices along the vertical axis.' },
  { id: 'xor3', name: 'XOR cube', dims: 3, classes: 2, description: 'The class is the parity of the three signs: XOR in three dimensions.' },
  { id: 'blobs3', name: 'Four blobs', dims: 3, classes: 4, description: 'Four Gaussian clouds at the corners of a tetrahedron.' },
];

export interface SyntheticConfig {
  id: SyntheticId;
  /** Total number of points (train + test). */
  count: number;
  /** 0 = clean; 0.5 = very noisy. Scales the Gaussian jitter added to every point. */
  noise: number;
  /** Share of points used for training (the rest are test points). */
  trainRatio: number;
  seed: number;
}

export interface PointSet {
  dims: 2 | 3;
  classes: number;
  /** n × dims raw coordinates. */
  coords: Float32Array;
  labels: Uint8Array;
}

export interface SyntheticData {
  info: SyntheticInfo;
  train: PointSet;
  test: PointSet;
}

export const DEFAULT_SYNTHETIC: Omit<SyntheticConfig, 'id'> = { count: 600, noise: 0.1, trainRatio: 0.5, seed: 1 };

export const syntheticInfo = (id: SyntheticId): SyntheticInfo => SYNTHETIC.find((s) => s.id === id)!;

const TAU = 2 * Math.PI;

/** One clean point of class `cls` for dataset `id`, before noise. `i`/`n` index the class's points. */
function point(id: SyntheticId, cls: number, i: number, n: number, rng: Rng, out: number[]): void {
  const u = () => rng.next() * 2 - 1;
  switch (id) {
    case 'circle': {
      const r = cls === 0 ? rng.next() * 0.5 : 0.7 + rng.next() * 0.3;
      const a = rng.next() * TAU;
      out[0] = r * Math.cos(a);
      out[1] = r * Math.sin(a);
      return;
    }
    case 'xor': {
      // Keep a small margin around the axes so the four quadrants are unambiguous.
      let x = 0;
      let y = 0;
      do {
        x = u();
        y = u();
      } while (Math.abs(x) < 0.05 || Math.abs(y) < 0.05 || (x * y > 0 ? 0 : 1) !== cls);
      out[0] = x;
      out[1] = y;
      return;
    }
    case 'gauss': {
      const c = cls === 0 ? 0.45 : -0.45;
      out[0] = c + rng.normal() * 0.22;
      out[1] = c + rng.normal() * 0.22;
      return;
    }
    case 'spiral': {
      const t = (i + rng.next()) / n;
      const r = 0.05 + 0.9 * t;
      const a = 1.75 * TAU * t + (cls === 0 ? 0 : Math.PI);
      out[0] = r * Math.sin(a);
      out[1] = r * Math.cos(a);
      return;
    }
    case 'moons': {
      const a = Math.PI * rng.next();
      if (cls === 0) {
        out[0] = Math.cos(a) * 0.6 - 0.3;
        out[1] = Math.sin(a) * 0.6 - 0.15;
      } else {
        out[0] = 0.3 - Math.cos(a) * 0.6;
        out[1] = 0.15 - Math.sin(a) * 0.6;
      }
      return;
    }
    case 'blobs': {
      const a = (cls * TAU) / 3 + Math.PI / 2;
      out[0] = 0.55 * Math.cos(a) + rng.normal() * 0.18;
      out[1] = 0.55 * Math.sin(a) + rng.normal() * 0.18;
      return;
    }
    case 'checker': {
      let x = 0;
      let y = 0;
      do {
        x = u();
        y = u();
      } while (((Math.floor((x + 1) * 2) + Math.floor((y + 1) * 2)) & 1) !== cls);
      out[0] = x;
      out[1] = y;
      return;
    }
    case 'shells': {
      const r = cls === 0 ? Math.cbrt(rng.next()) * 0.5 : 0.7 + rng.next() * 0.3;
      const z = u();
      const a = rng.next() * TAU;
      const s = Math.sqrt(1 - z * z);
      out[0] = r * s * Math.cos(a);
      out[1] = r * s * Math.sin(a);
      out[2] = r * z;
      return;
    }
    case 'helix': {
      const t = (i + rng.next()) / n;
      const a = 2 * TAU * t + (cls === 0 ? 0 : Math.PI);
      out[0] = 0.6 * Math.cos(a);
      out[1] = 0.6 * Math.sin(a);
      out[2] = 2 * t - 1;
      return;
    }
    case 'xor3': {
      let x = 0;
      let y = 0;
      let z = 0;
      do {
        x = u();
        y = u();
        z = u();
      } while (Math.min(Math.abs(x), Math.abs(y), Math.abs(z)) < 0.05 || (x * y * z > 0 ? 0 : 1) !== cls);
      out[0] = x;
      out[1] = y;
      out[2] = z;
      return;
    }
    case 'blobs3': {
      // Tetrahedron corners, alternating cube vertices.
      const corners = [
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
      ][cls];
      for (let d = 0; d < 3; d++) out[d] = 0.45 * corners[d] + rng.normal() * 0.18;
      return;
    }
  }
}

/** Generates the dataset and splits it into train and test points (classes balanced, order shuffled). */
export function generate(config: SyntheticConfig): SyntheticData {
  const info = syntheticInfo(config.id);
  const rng = new Rng(config.seed * 7919 + info.id.length * 104729);
  const { dims, classes } = info;
  const n = Math.max(classes * 2, Math.round(config.count));
  const coords = new Float32Array(n * dims);
  const labels = new Uint8Array(n);
  const p = [0, 0, 0];
  const sigma = config.noise * 0.3;
  for (let k = 0; k < n; k++) {
    const cls = k % classes;
    const perClass = Math.ceil(n / classes);
    point(config.id, cls, Math.floor(k / classes), perClass, rng, p);
    for (let d = 0; d < dims; d++) coords[k * dims + d] = p[d] + (sigma > 0 ? rng.normal() * sigma : 0);
    labels[k] = cls;
  }
  // Shuffle, then split.
  const order = new Int32Array(n);
  for (let i = 0; i < n; i++) order[i] = i;
  rng.shuffle(order);
  const nTrain = Math.min(n - 1, Math.max(1, Math.round(n * config.trainRatio)));
  const take = (from: number, to: number): PointSet => {
    const m = to - from;
    const c = new Float32Array(m * dims);
    const l = new Uint8Array(m);
    for (let j = 0; j < m; j++) {
      const src = order[from + j];
      for (let d = 0; d < dims; d++) c[j * dims + d] = coords[src * dims + d];
      l[j] = labels[src];
    }
    return { dims, classes, coords: c, labels: l };
  };
  return { info, train: take(0, nTrain), test: take(nTrain, n) };
}
