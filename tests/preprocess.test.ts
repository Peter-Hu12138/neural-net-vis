import { describe, expect, it } from 'vitest';
import { inkFromRGBA, resample, toMnist } from '../src/data/preprocess';

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
