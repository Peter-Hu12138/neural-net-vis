import { describe, expect, it } from 'vitest';
import {
  adjustImage,
  centreSquare,
  inkFromRGBA,
  inkToFashion,
  isAdjusted,
  maskBox,
  NO_ADJUSTMENT,
  resample,
  toColour,
  toFashion,
  toMnist,
} from '../src/data/preprocess';

function rgba(w: number, h: number, paint: (x: number, y: number) => number): Uint8ClampedArray {
  const d = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = paint(x, y);
      d.set([v, v, v, 255], 4 * (y * w + x));
    }
  }
  return d;
}

function bbox(img: Float32Array) {
  let x0 = 28, y0 = 28, x1 = -1, y1 = -1;
  img.forEach((v, i) => {
    if (v > 0.3) {
      const x = i % 28, y = Math.floor(i / 28);
      x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
    }
  });
  return { x0, y0, x1, y1 };
}

describe('preprocess', () => {
  it('inverts dark ink on white paper', () => {
    const d = rgba(50, 50, (x, y) => (x > 20 && x < 30 && y > 10 && y < 40 ? 0 : 255));
    const ink = inkFromRGBA(d, 50, 50);
    expect(ink[25 * 50 + 25]).toBeCloseTo(1);
    expect(ink[0]).toBe(0);
  });

  it('keeps light ink on a dark background', () => {
    const d = rgba(50, 50, (x, y) => (x > 20 && x < 30 && y > 10 && y < 40 ? 230 : 10));
    const ink = inkFromRGBA(d, 50, 50);
    expect(ink[25 * 50 + 25]).toBeCloseTo(1);
    expect(ink[0]).toBe(0);
  });

  it('scales the longest side to 20 px and centres the mass', () => {
    // A tall bar in the top-left corner of a large image.
    const d = rgba(200, 200, (x, y) => (x >= 5 && x < 25 && y >= 5 && y < 105 ? 0 : 255));
    const img = toMnist(inkFromRGBA(d, 200, 200), 200, 200)!;
    const b = bbox(img);
    expect(b.y1 - b.y0 + 1).toBe(20);
    expect(b.x1 - b.x0 + 1).toBe(4);
    expect((b.y0 + b.y1 + 1) / 2).toBeCloseTo(14, 0);
    expect((b.x0 + b.x1 + 1) / 2).toBeCloseTo(14, 0);
  });

  it('returns null for a blank image', () => {
    const d = rgba(40, 40, () => 255);
    expect(toMnist(inkFromRGBA(d, 40, 40), 40, 40)).toBeNull();
  });

  it('area-averages when shrinking', () => {
    const src = new Float32Array([1, 0, 1, 0, 1, 0, 1, 0]);
    const out = resample(src, 8, 1, 4, 1);
    expect(Array.from(out)).toEqual([0.5, 0.5, 0.5, 0.5]);
  });
});

/** RGBA pixels from a per-pixel [r, g, b, a] painter. */
function rgbaImage(w: number, h: number, paint: (x: number, y: number) => [number, number, number, number?]) {
  const data = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const [r, g, b, a = 255] = paint(x, y);
      data.set([r, g, b, a], 4 * (y * w + x));
    }
  }
  return { data, width: w, height: h };
}

/** Bounding box of values above `t` in a side × side single-channel image. */
function box28(img: Float32Array, t = 0.2, side = 28) {
  let x0 = side, y0 = side, x1 = -1, y1 = -1;
  img.forEach((v, i) => {
    if (v > t) {
      const x = i % side, y = Math.floor(i / side);
      x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
    }
  });
  return { x0, y0, x1, y1, w: x1 - x0 + 1, h: y1 - y0 + 1 };
}

describe('colour preprocessing (CIFAR-10)', () => {
  it('is channel-major RGB in [0, 1]', () => {
    const x = toColour(rgbaImage(64, 64, () => [255, 128, 0]));
    expect(x.length).toBe(3 * 32 * 32);
    expect(x[0]).toBeCloseTo(1);
    expect(x[1023]).toBeCloseTo(1);
    expect(x[1024]).toBeCloseTo(128 / 255);
    expect(x[2048]).toBeCloseTo(0);
    expect(Math.min(...x)).toBeGreaterThanOrEqual(0);
    expect(Math.max(...x)).toBeLessThanOrEqual(1);
  });

  it('crops the centred square of a wide photo', () => {
    // 96 × 32: a red left third, a green middle third, a blue right third. Only green survives.
    const x = toColour(rgbaImage(96, 32, (px) => (px < 32 ? [255, 0, 0] : px < 64 ? [0, 255, 0] : [0, 0, 255])));
    const mean = (ch: number) => x.slice(ch * 1024, (ch + 1) * 1024).reduce((s, v) => s + v, 0) / 1024;
    expect(mean(0)).toBeCloseTo(0);
    expect(mean(1)).toBeCloseTo(1);
    expect(mean(2)).toBeCloseTo(0);
    expect(centreSquare(96, 32)).toEqual({ x0: 32, y0: 0, x1: 64, y1: 32 });
    expect(centreSquare(30, 50)).toEqual({ x0: 0, y0: 10, x1: 30, y1: 40 });
  });

  it('area-averages when shrinking, so fine detail turns into its mean', () => {
    const x = toColour(rgbaImage(64, 64, (px, py) => ((px + py) % 2 ? [255, 255, 255] : [0, 0, 0])));
    for (const v of x) expect(v).toBeCloseTo(0.5, 5);
  });

  it('keeps the top-left of the crop at the top-left of the input', () => {
    // 128 × 128 with a black 64 × 64 top-left quadrant on white: one 16 × 16 block of the input.
    const x = toColour(rgbaImage(128, 128, (px, py) => (px < 64 && py < 64 ? [0, 0, 0] : [255, 255, 255])));
    expect(x[0]).toBeCloseTo(0);
    expect(x[15 * 32 + 15]).toBeCloseTo(0);
    expect(x[16]).toBeCloseTo(1);
    expect(x[16 * 32]).toBeCloseTo(1);
  });

  it('composites transparent pixels on white and enlarges small images', () => {
    const x = toColour(rgbaImage(8, 8, () => [0, 0, 0, 0]));
    expect(x.length).toBe(3072);
    for (const v of x) expect(v).toBeCloseTo(1);
  });
});

describe('Fashion-MNIST preprocessing', () => {
  it('turns a dark item on a white backdrop into a light item on black, filling 28 px', () => {
    // A 40 × 120 dark-grey "trouser" in a 300 × 200 white photo, off centre.
    const img = rgbaImage(300, 200, (px, py) => (px >= 30 && px < 70 && py >= 40 && py < 160 ? [40, 40, 50] : [250, 250, 250]));
    const { x, box } = toFashion(img);
    expect(box).toEqual({ x0: 30, y0: 40, x1: 70, y1: 160 });
    const b = box28(x!);
    expect(b.h).toBe(28); // the longest edge fills the frame
    expect(b.w).toBeGreaterThanOrEqual(9);
    expect(b.w).toBeLessThanOrEqual(10);
    expect(Math.abs((b.x0 + b.x1) / 2 - 13.5)).toBeLessThanOrEqual(0.5); // centred
    expect(x![0]).toBe(0); // black background, as in the dataset
    expect(x![14 * 28 + 14]).toBeGreaterThan(0.9); // the item is light
  });

  it('keeps the polarity for a light item on a dark backdrop', () => {
    const img = rgbaImage(100, 100, (px, py) => (Math.hypot(px - 50, py - 50) < 30 ? [230, 220, 210] : [15, 15, 20]));
    const { x } = toFashion(img);
    expect(x![0]).toBe(0);
    expect(x![14 * 28 + 14]).toBeGreaterThan(0.9);
    const b = box28(x!);
    expect(Math.max(b.w, b.h)).toBe(28);
  });

  it('ignores a few stray pixels when finding the item', () => {
    const img = rgbaImage(200, 200, (px, py) => {
      if (px === 3 && py === 190) return [0, 0, 0]; // a speck of dust
      return px >= 80 && px < 120 && py >= 50 && py < 150 ? [0, 0, 0] : [255, 255, 255];
    });
    expect(toFashion(img).box).toEqual({ x0: 80, y0: 50, x1: 120, y1: 150 });
  });

  it('returns null for an empty photo', () => {
    expect(toFashion(rgbaImage(50, 50, () => [200, 200, 200])).x).toBeNull();
  });

  it('frames a drawing the same way', () => {
    const ink = new Float32Array(100 * 100);
    for (let y = 10; y < 30; y++) for (let x = 40; x < 90; x++) ink[y * 100 + x] = 1; // wide and short
    const x = inkToFashion(ink, 100, 100)!;
    const b = box28(x);
    expect(b.w).toBe(28);
    expect(b.h).toBeGreaterThanOrEqual(11);
    expect(b.h).toBeLessThanOrEqual(12);
    expect(Math.abs((b.y0 + b.y1) / 2 - 13.5)).toBeLessThanOrEqual(0.5);
    expect(inkToFashion(new Float32Array(100), 10, 10)).toBeNull();
  });

  it('finds the box of a mask', () => {
    const m = new Uint8Array(10 * 10);
    m[3 * 10 + 2] = 1;
    m[7 * 10 + 6] = 1;
    expect(maskBox(m, 10, 10, 0)).toEqual({ x0: 2, y0: 3, x1: 7, y1: 8 });
    expect(maskBox(new Uint8Array(100), 10, 10)).toBeNull();
  });
});

describe('adjusting an image input', () => {
  const img = Float32Array.from({ length: 2 * 2 * 3 }, (_, i) => i / 12); // 2 channels × 2 × 3

  it('is the identity without nudges', () => {
    expect(Array.from(adjustImage(img, 2, 2, 3, NO_ADJUSTMENT))).toEqual(Array.from(img));
    expect(isAdjusted(NO_ADJUSTMENT)).toBe(false);
  });

  it('flips every channel left to right', () => {
    const f = adjustImage(img, 2, 2, 3, { ...NO_ADJUSTMENT, flip: true });
    for (let ch = 0; ch < 2; ch++) {
      for (let y = 0; y < 2; y++) {
        for (let x = 0; x < 3; x++) expect(f[(ch * 2 + y) * 3 + x]).toBeCloseTo(img[(ch * 2 + y) * 3 + (2 - x)]);
      }
    }
    expect(isAdjusted({ ...NO_ADJUSTMENT, flip: true })).toBe(true);
  });

  it('applies contrast about mid-grey, then brightness, and clamps to [0, 1]', () => {
    const x = new Float32Array([0, 0.25, 0.5, 0.75, 1]);
    const b = adjustImage(x, 1, 1, 5, { flip: false, brightness: 0.1, contrast: 1 });
    [0.1, 0.35, 0.6, 0.85, 1].forEach((v, i) => expect(b[i]).toBeCloseTo(v));
    expect(Array.from(adjustImage(x, 1, 1, 5, { flip: false, brightness: 0, contrast: 2 }))).toEqual([0, 0, 0.5, 1, 1]);
    const lo = adjustImage(x, 1, 1, 5, { flip: false, brightness: -0.2, contrast: 0.5 });
    expect(lo[2]).toBeCloseTo(0.3);
    expect(lo[0]).toBeCloseTo(0.05);
  });
});
