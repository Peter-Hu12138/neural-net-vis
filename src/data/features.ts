/**
 * Input features for point datasets, as in the TensorFlow Playground: the network can be fed the
 * raw coordinates and, optionally, squares, products and sines of them. Coordinates live in
 * [−1, 1], so every feature stays within [−1, 1] too.
 */

export type FeatureId = string; // 'x1', 'x2', 'x3', 'x1^2', 'x1*x2', 'sin x1', …

export interface FeatureDef {
  id: FeatureId;
  /** Display label with sub/superscripts, e.g. "x₁²". */
  label: string;
  /** Plain-language description for tooltips. */
  title: string;
  fn: (p: ArrayLike<number>, off: number) => number;
}

const SUB = ['₁', '₂', '₃'];

/** Every feature available for `dims`-dimensional points, in display order. */
export function featureCatalog(dims: 2 | 3): FeatureDef[] {
  const out: FeatureDef[] = [];
  for (let i = 0; i < dims; i++) {
    out.push({ id: `x${i + 1}`, label: `x${SUB[i]}`, title: `Coordinate ${i + 1}`, fn: (p, o) => p[o + i] });
  }
  for (let i = 0; i < dims; i++) {
    out.push({ id: `x${i + 1}^2`, label: `x${SUB[i]}²`, title: `Coordinate ${i + 1} squared`, fn: (p, o) => p[o + i] * p[o + i] });
  }
  for (let i = 0; i < dims; i++) {
    for (let j = i + 1; j < dims; j++) {
      out.push({ id: `x${i + 1}*x${j + 1}`, label: `x${SUB[i]}x${SUB[j]}`, title: `Coordinates ${i + 1} and ${j + 1} multiplied`, fn: (p, o) => p[o + i] * p[o + j] });
    }
  }
  for (let i = 0; i < dims; i++) {
    out.push({ id: `sin x${i + 1}`, label: `sin x${SUB[i]}`, title: `sin(π · coordinate ${i + 1})`, fn: (p, o) => Math.sin(Math.PI * p[o + i]) });
  }
  return out;
}

/** The raw coordinates only: the default, as in the Playground. */
export const defaultFeatures = (dims: 2 | 3): FeatureId[] => featureCatalog(dims).slice(0, dims).map((f) => f.id);

export function featureDefs(dims: 2 | 3, ids: FeatureId[]): FeatureDef[] {
  const cat = featureCatalog(dims);
  const defs = ids.map((id) => cat.find((f) => f.id === id));
  if (defs.some((d) => !d)) throw new Error(`Unknown feature in ${ids.join(', ')}`);
  return defs as FeatureDef[];
}

/** Maps n points (n × dims coordinates) to network inputs (n × features). */
export function featurize(coords: Float32Array, dims: 2 | 3, ids: FeatureId[], out?: Float32Array): Float32Array {
  const defs = featureDefs(dims, ids);
  const n = coords.length / dims;
  const F = defs.length;
  const x = out ?? new Float32Array(n * F);
  for (let k = 0; k < n; k++) {
    for (let f = 0; f < F; f++) x[k * F + f] = defs[f].fn(coords, k * dims);
  }
  return x;
}
