import { describe } from '../nn/network';
import type { Arch } from '../nn/types';

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
function project(arch: Arch, sizes: number[], block: number, v: number, level: 'z' | 'out', clip: boolean): [number, number] {
  let lo = v;
  let hi = v;
  const clamp = (n: number) => {
    if (!clip) return;
    lo = Math.max(0, lo);
    hi = Math.min(n - 1, hi);
  };
  for (let i = block; i >= 0; i--) {
    const l = arch.layers[i];
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
function isConvPath(arch: Arch, block: number): boolean {
  if (block < 0 || block >= arch.layers.length) return false;
  for (let i = 0; i <= block; i++) if (arch.layers[i].kind !== 'conv') return false;
  return true;
}

/**
 * The input pixels that position (y, x) of conv block `block` can see. `level` 'z' is a position
 * in the block's pre-pool map, 'out' one in its pooled output. Bounds are inclusive and clipped
 * to the image; pass `clip = false` for the nominal, unclipped field (useful for fixed-size crops).
 * Dense and output blocks see the whole image and return null.
 */
export function receptiveBox(arch: Arch, block: number, y: number, x: number, level: 'z' | 'out' = 'z', clip = true): Box | null {
  if (!isConvPath(arch, block)) return null;
  const info = describe(arch);
  // A conv keeps its input's size, so block i's pre-pool map is as large as its input.
  const rows = info.map((l) => l.inShape.h);
  const cols = info.map((l) => l.inShape.w);
  const [y0, y1] = project(arch, rows, block, y, level, clip);
  const [x0, x1] = project(arch, cols, block, x, level, clip);
  return { y0, y1, x0, x1 };
}

/**
 * Side length in input pixels of the field one unit of `block` sees, away from the borders
 * (for example 3 for a first 3×3 conv, 8 for the second conv of the Small CNN preset), on any
 * input size. Dense and output blocks see the whole image (28 pixels for MNIST, 32 for CIFAR-10).
 */
export function receptiveSize(arch: Arch, block: number, level: 'z' | 'out' = 'z'): number {
  const b = receptiveBox(arch, block, 0, 0, level, false);
  return b ? b.y1 - b.y0 + 1 : arch.input.h;
}

/** The whole input image. */
export const wholeImage = (arch: Arch): Box => ({ y0: 0, y1: arch.input.h - 1, x0: 0, x1: arch.input.w - 1 });

const side = (b: Box) => Math.max(b.y1 - b.y0 + 1, b.x1 - b.x0 + 1);

/**
 * The square crop that shows what position (y, x) of conv block `block`'s pre-pool map sees.
 * While the nominal field is smaller than the image it is that field, unclipped, so every crop
 * of a layer has the same size and pixels past the image edge show as blank. Once the field is as
 * large as the image (deep stacks reach 38 or 58 pixels) it is the whole image, never more.
 * Boxes are spatial: a colour image's three channels share them. Dense and output blocks return null.
 */
export function cropBox(arch: Arch, block: number, y: number, x: number): Box | null {
  const b = receptiveBox(arch, block, y, x, 'z', false);
  if (!b) return null;
  return side(b) >= arch.input.h ? wholeImage(arch) : b;
}

/** Position at the centre of conv block `block`'s pre-pool map (the one activation maximisation uses). */
export function centrePosition(arch: Arch, block: number): { y: number; x: number } {
  const l = describe(arch)[block];
  return { y: l.inShape.h >> 1, x: l.inShape.w >> 1 };
}

/**
 * Side length in input pixels of what the unit at the centre of `block`'s map really sees,
 * clipped to the image, so at most the image's side (which means the whole image). Dense and
 * output blocks: the image's side.
 */
export function centreFieldSize(arch: Arch, block: number): number {
  if (!isConvPath(arch, block)) return arch.input.h;
  const { y, x } = centrePosition(arch, block);
  return side(receptiveBox(arch, block, y, x, 'z', true)!);
}
