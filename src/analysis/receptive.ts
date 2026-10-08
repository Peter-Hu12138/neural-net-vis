import { describe } from '../nn/network';
import { INPUT_SHAPE, type LayerSpec } from '../nn/types';

/** Inclusive input-pixel bounds: rows y0…y1, columns x0…x1. */
export interface Box {
  y0: number;
  y1: number;
  x0: number;
  x1: number;
}

/**
 * Back-projects the index range [lo, hi] on one axis of block `block` to input pixels.
 * Convs are stride 1 with same padding, so a conv widens the range by its padding on each side;
 * a 2×2 stride-2 pool maps pooled index i to the pre-pool indices 2i and 2i + 1. When `clip` is
 * set, the range is clipped to each map's real size on the way, so padding positions (and the
 * last row of an odd-sized map, which pooling drops) never count as seen pixels.
 */
function project(spec: LayerSpec[], sizes: number[], block: number, v: number, level: 'z' | 'out', clip: boolean): [number, number] {
  let lo = v;
  let hi = v;
  const clamp = (n: number) => {
    if (!clip) return;
    lo = Math.max(0, lo);
    hi = Math.min(n - 1, hi);
  };
  for (let i = block; i >= 0; i--) {
    const l = spec[i];
    if (l.kind !== 'conv') throw new Error('receptive fields need conv layers');
    // At block i's output level: undo the pool (except at the starting block when asked for z).
    if (l.pool && (i < block || level === 'out')) {
      lo = 2 * lo;
      hi = 2 * hi + 1;
      clamp(sizes[i]);
    }
    const pad = (l.kernel - 1) >> 1;
    lo -= pad;
    hi += pad;
    clamp(sizes[i]);
  }
  return [lo, hi];
}

/** True when block `block` is a conv block whose input path is all convs (a valid spec). */
function isConvPath(spec: LayerSpec[], block: number): boolean {
  if (block < 0 || block >= spec.length) return false;
  for (let i = 0; i <= block; i++) if (spec[i].kind !== 'conv') return false;
  return true;
}

/**
 * The input pixels that position (y, x) of conv block `block` can see. `level` 'z' is a position
 * in the block's pre-pool map, 'out' one in its pooled output. Bounds are inclusive and clipped
 * to [0, 27]; pass `clip = false` for the nominal, unclipped field (useful for fixed-size crops).
 * Dense and output blocks see the whole image and return null.
 */
export function receptiveBox(spec: LayerSpec[], block: number, y: number, x: number, level: 'z' | 'out' = 'z', clip = true): Box | null {
  if (!isConvPath(spec, block)) return null;
  const info = describe(spec);
  // A conv keeps its input's size, so block i's pre-pool map is as large as its input.
  const rows = info.map((l) => l.inShape.h);
  const cols = info.map((l) => l.inShape.w);
  const [y0, y1] = project(spec, rows, block, y, level, clip);
  const [x0, x1] = project(spec, cols, block, x, level, clip);
  return { y0, y1, x0, x1 };
}

/**
 * Side length in input pixels of the field one unit of `block` sees, away from the borders
 * (for example 3 for a first 3×3 conv, 8 for the second conv of the Small CNN preset).
 * Dense and output blocks see the whole 28-pixel image.
 */
export function receptiveSize(spec: LayerSpec[], block: number, level: 'z' | 'out' = 'z'): number {
  const b = receptiveBox(spec, block, 0, 0, level, false);
  return b ? b.y1 - b.y0 + 1 : INPUT_SHAPE.h;
}
