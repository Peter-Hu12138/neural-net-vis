/**
 * Turns arbitrary images into MNIST-style inputs, following the original recipe:
 * crop to the ink, scale the longest side to 20 px (keeping aspect ratio, anti-aliased),
 * then paste into a 28×28 field so the centre of mass sits at the centre.
 */

export const SIDE = 28;
const BOX = 20;

/**
 * Ink intensity in [0, 1] (1 = full ink) from RGBA pixels. The background colour is taken
 * from the border, so dark-on-light photos and light-on-dark images both work.
 */
export function inkFromRGBA(data: Uint8ClampedArray, w: number, h: number): Float32Array {
  const lum = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const a = data[4 * i + 3] / 255;
    const l = (0.299 * data[4 * i] + 0.587 * data[4 * i + 1] + 0.114 * data[4 * i + 2]) / 255;
    lum[i] = l * a + (1 - a); // transparent pixels read as white paper
  }
  const border: number[] = [];
  for (let x = 0; x < w; x++) border.push(lum[x], lum[(h - 1) * w + x]);
  for (let y = 0; y < h; y++) border.push(lum[y * w], lum[y * w + w - 1]);
  border.sort((p, q) => p - q);
  const bg = border[border.length >> 1];
  const darkInk = bg > 0.5;
  const ink = new Float32Array(w * h);
  for (let i = 0; i < ink.length; i++) ink[i] = Math.max(0, darkInk ? bg - lum[i] : lum[i] - bg);

  // Contrast stretch between a noise floor and a robust maximum.
  const sorted = Float32Array.from(ink).sort();
  const hi = sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.995))] || sorted[sorted.length - 1];
  if (hi <= 0.02) return new Float32Array(w * h);
  const lo = 0.2 * hi;
  for (let i = 0; i < ink.length; i++) ink[i] = Math.min(1, Math.max(0, (ink[i] - lo) / (hi - lo)));
  return ink;
}

/** Separable resample: box-filter average when shrinking, linear interpolation when enlarging. */
export function resample(src: Float32Array, sw: number, sh: number, dw: number, dh: number): Float32Array {
  const rows = new Float32Array(dw * sh);
  for (let y = 0; y < sh; y++) resample1d(src, y * sw, 1, sw, rows, y * dw, 1, dw);
  const out = new Float32Array(dw * dh);
  for (let x = 0; x < dw; x++) resample1d(rows, x, dw, sh, out, x, dw, dh);
  return out;
}

function resample1d(
  src: Float32Array, so: number, ss: number, sn: number,
  dst: Float32Array, d0: number, ds: number, dn: number,
): void {
  const scale = sn / dn;
  for (let d = 0; d < dn; d++) {
    let v = 0;
    if (scale >= 1) {
      const a = d * scale;
      const b = a + scale;
      for (let s = Math.floor(a); s < Math.ceil(b) && s < sn; s++) {
        const cover = Math.min(b, s + 1) - Math.max(a, s);
        if (cover > 0) v += src[so + s * ss] * cover;
      }
      v /= scale;
    } else {
      const t = Math.min(sn - 1, Math.max(0, (d + 0.5) * scale - 0.5));
      const s0 = Math.floor(t);
      const s1 = Math.min(sn - 1, s0 + 1);
      const f = t - s0;
      v = src[so + s0 * ss] * (1 - f) + src[so + s1 * ss] * f;
    }
    dst[d0 + d * ds] = v;
  }
}

/** Returns the 28×28 MNIST-style digit, or null when the image has no ink. */
export function toMnist(ink: Float32Array, w: number, h: number): Float32Array | null {
  let max = 0;
  for (let i = 0; i < ink.length; i++) if (ink[i] > max) max = ink[i];
  if (max < 0.05) return null;
  const t = 0.1 * max;
  let x0 = w, y0 = h, x1 = -1, y1 = -1;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (ink[y * w + x] > t) {
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
    }
  }
  const cw = x1 - x0 + 1;
  const ch = y1 - y0 + 1;
  const crop = new Float32Array(cw * ch);
  for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) crop[y * cw + x] = ink[(y0 + y) * w + x0 + x];

  const s = BOX / Math.max(cw, ch);
  const tw = Math.max(1, Math.round(cw * s));
  const th = Math.max(1, Math.round(ch * s));
  const small = resample(crop, cw, ch, tw, th);

  let m = 0, mx = 0, my = 0;
  for (let y = 0; y < th; y++) {
    for (let x = 0; x < tw; x++) {
      const v = small[y * tw + x];
      m += v;
      mx += v * (x + 0.5);
      my += v * (y + 0.5);
    }
  }
  const ox = Math.round(SIDE / 2 - mx / m);
  const oy = Math.round(SIDE / 2 - my / m);
  const out = new Float32Array(SIDE * SIDE);
  for (let y = 0; y < th; y++) {
    const ty = y + oy;
    if (ty < 0 || ty >= SIDE) continue;
    for (let x = 0; x < tw; x++) {
      const tx = x + ox;
      if (tx < 0 || tx >= SIDE) continue;
      out[ty * SIDE + tx] = Math.min(1, small[y * tw + x]);
    }
  }
  return out;
}

export function rgbaToMnist(data: Uint8ClampedArray, w: number, h: number): Float32Array | null {
  return toMnist(inkFromRGBA(data, w, h), w, h);
}
