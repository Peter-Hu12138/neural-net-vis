import { describe, expect, it } from 'vitest';
import { centreFieldSize, centrePosition, cropBox, receptiveBox, receptiveSize, wholeImage, type Box } from '../src/analysis/receptive';
import { forwardTo } from '../src/analysis/units';
import { Network } from '../src/nn/network';
import { Rng } from '../src/nn/rng';
import type { Arch, LayerSpec } from '../src/nn/types';
import { IMAGE_PRESETS } from '../src/store';

/**
 * Receptive fields on CIFAR-10's 32×32×3 input: the same convs and pools as on MNIST, but a larger
 * image to clip to, and three channels that share every spatial box.
 */

const CIFAR = { c: 3, h: 32, w: 32 };
const cifar = (layers: LayerSpec[]): Arch => ({ input: CIFAR, layers, classes: 10 });
const SMALL_CNN = structuredClone(IMAGE_PRESETS.find((p) => p.name === 'Small CNN')!.spec);
const LENET = structuredClone(IMAGE_PRESETS.find((p) => p.name === 'LeNet-ish')!.spec);
const box = (y0: number, y1: number, x0: number, x1: number): Box => ({ y0, y1, x0, x1 });

describe('receptive fields on 32×32 colour images', () => {
  it('Small CNN: the same field sizes as on MNIST, centred on the larger map', () => {
    const arch = cifar(SMALL_CNN);
    expect(receptiveSize(arch, 0)).toBe(3);
    expect(receptiveSize(arch, 1)).toBe(8);
    // Conv 2's pre-pool map is 16×16; its centre (8, 8) sees rows and columns 13–20.
    expect(centrePosition(arch, 1)).toEqual({ y: 8, x: 8 });
    expect(receptiveBox(arch, 1, 8, 8, 'z')).toEqual(box(13, 20, 13, 20));
    expect(cropBox(arch, 1, 8, 8)).toEqual(box(13, 20, 13, 20));
    expect(centreFieldSize(arch, 1)).toBe(8);
    expect(centreFieldSize(arch, 0)).toBe(3);
  });

  it('clips at the far corner of the 32×32 image, not at 28', () => {
    const arch = cifar(SMALL_CNN);
    expect(receptiveBox(arch, 0, 31, 31, 'z')).toEqual(box(30, 31, 30, 31));
    expect(receptiveBox(arch, 1, 15, 15, 'z')).toEqual(box(27, 31, 27, 31));
    // Unclipped, crops keep their nominal size so every crop of a layer is the same size.
    expect(cropBox(arch, 1, 15, 15)).toEqual(box(27, 34, 27, 34));
    expect(receptiveSize(cifar(LENET), 1)).toBe(14);
  });

  it('deep stacks: a field larger than the image becomes the whole 32×32 image', () => {
    const conv = (kernel: 3 | 5, pool: boolean): LayerSpec => ({ kind: 'conv', filters: 2, kernel, act: 'relu', pool });
    const deep = [conv(3, true), conv(3, true), conv(3, true), conv(3, true)];
    const arch = cifar(deep);
    expect(receptiveSize(arch, 3)).toBe(38);
    // The centre of an even 4×4 map is (2, 2), a little off the image's centre: it sees rows and
    // columns 1–31, all but the first, and never more than the image.
    expect(centreFieldSize(arch, 3)).toBe(31);
    expect(receptiveBox(arch, 3, 2, 2, 'z')).toEqual(box(1, 31, 1, 31));
    // 32 → 16 → 8 → 4: every position of conv 4's 4×4 map crops to the whole image.
    for (let y = 0; y < 4; y++) for (let x = 0; x < 4; x++) expect(cropBox(arch, 3, y, x)).toEqual(wholeImage(arch));
    expect(wholeImage(arch)).toEqual(box(0, 31, 0, 31));
  });

  it('dense and output blocks see the whole image', () => {
    const arch = cifar(SMALL_CNN);
    expect(receptiveBox(arch, 2, 0, 0)).toBeNull();
    expect(receptiveSize(arch, 2)).toBe(32);
    expect(centreFieldSize(arch, 3)).toBe(32);
    expect(cropBox(arch, 3, 0, 0)).toBeNull();
  });

  it('one box covers all three channels: a pixel outside it, in any channel, never moves the unit', () => {
    const arch = cifar(SMALL_CNN);
    const net = new Network(arch, 5);
    const rng = new Rng(3);
    const x = Float32Array.from({ length: 3 * 32 * 32 }, () => rng.next());
    const HW = 32 * 32;
    for (const [y, xx] of [
      [8, 8],
      [0, 15],
      [15, 3],
    ] as const) {
      const b = receptiveBox(arch, 1, y, xx, 'z')!;
      const zi = (f: number) => f * 16 * 16 + y * 16 + xx;
      forwardTo(net, x, 1);
      const base = Array.from({ length: 16 }, (_, f) => net.blocks[1].z[zi(f)]);
      let inside = 0;
      for (let ch = 0; ch < 3; ch++) {
        for (let r = 0; r < 32; r++) {
          for (let c = 0; c < 32; c++) {
            const p = ch * HW + r * 32 + c;
            const o = x[p];
            x[p] = o + 0.5;
            forwardTo(net, x, 1);
            const moved = base.some((v, f) => Math.abs(net.blocks[1].z[zi(f)] - v) > 1e-6);
            x[p] = o;
            const isIn = r >= b.y0 && r <= b.y1 && c >= b.x0 && c <= b.x1;
            if (!isIn) expect(moved, `channel ${ch} pixel (${r}, ${c}) outside ${JSON.stringify(b)}`).toBe(false);
            else if (moved) inside++;
          }
        }
      }
      // Every channel inside the box can move it (max-pool may hide a few pixels per window).
      expect(inside).toBeGreaterThan(0.5 * 3 * (b.y1 - b.y0 + 1) * (b.x1 - b.x0 + 1));
    }
  }, 60_000);
});
