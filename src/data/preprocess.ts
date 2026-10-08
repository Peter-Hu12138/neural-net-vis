/**
 * Turns arbitrary images (photos, scans, drawings) into inputs that look like each dataset's own
 * samples. Pure functions on pixel arrays, so they run in tests without a browser.
 *
 * - MNIST (the original recipe): crop to the ink, scale the longest side to 20 px (keeping the
 *   aspect ratio, anti-aliased), then paste into a 28×28 field so the centre of mass sits at the
 *   centre. Ink is light on black.
 * - Fashion-MNIST (Zalando's recipe): crop to the item, scale its longest edge to 28 px and centre
 *   it; the item is light on black.
 * - CIFAR-10: the centred square of the photo, area-averaged to 32×32, in colour.
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

// ── Fashion-MNIST and CIFAR-10 ─────────────────────────────────────────────

/** Anything shaped like ImageData: RGBA bytes, row by row. */
export interface ImageLike {
  data: ArrayLike<number>;
  width: number;
  height: number;
}

/** Pixel rectangle [x0, x1) × [y0, y1) in source coordinates. */
export interface Box {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

/** Red, green and blue of pixel `i` in [0, 1], composited on white paper when transparent. */
function rgbAt(img: ImageLike, i: number, out: number[]): number[] {
  const d = img.data;
  const a = d[4 * i + 3] / 255;
  for (let ch = 0; ch < 3; ch++) out[ch] = (d[4 * i + ch] / 255) * a + (1 - a);
  return out;
}

/** The largest centred square of a w × h image. */
export function centreSquare(w: number, h: number): Box {
  const s = Math.min(w, h);
  const x0 = Math.floor((w - s) / 2);
  const y0 = Math.floor((h - s) / 2);
  return { x0, y0, x1: x0 + s, y1: y0 + s };
}

/**
 * CIFAR-10 style input: the centred square of the image, area-averaged down to side × side
 * (enlarged by linear interpolation when smaller), as channel-major RGB in [0, 1] (all red
 * values, then green, then blue), the layout the network reads.
 */
export function toColour(img: ImageLike, side = 32): Float32Array {
  const box = centreSquare(img.width, img.height);
  const s = box.x1 - box.x0;
  const planes = [new Float32Array(s * s), new Float32Array(s * s), new Float32Array(s * s)];
  const px = [0, 0, 0];
  for (let y = 0; y < s; y++) {
    for (let x = 0; x < s; x++) {
      rgbAt(img, (box.y0 + y) * img.width + box.x0 + x, px);
      for (let ch = 0; ch < 3; ch++) planes[ch][y * s + x] = px[ch];
    }
  }
  const HW = side * side;
  const out = new Float32Array(3 * HW);
  for (let ch = 0; ch < 3; ch++) out.set(resample(planes[ch], s, s, side, side), ch * HW);
  return out;
}

/**
 * Bounding box of the cells of a w × h mask that are set, ignoring stray pixels: a row or column
 * counts only when at least `minShare` of it (and at least one pixel) is set. Null when empty.
 */
export function maskBox(mask: Uint8Array, w: number, h: number, minShare = 0.01): Box | null {
  const rows = new Int32Array(h);
  const cols = new Int32Array(w);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (mask[y * w + x]) {
        rows[y]++;
        cols[x]++;
      }
    }
  }
  const rowMin = Math.max(1, Math.round(minShare * w));
  const colMin = Math.max(1, Math.round(minShare * h));
  let y0 = 0;
  while (y0 < h && rows[y0] < rowMin) y0++;
  let y1 = h;
  while (y1 > y0 && rows[y1 - 1] < rowMin) y1--;
  let x0 = 0;
  while (x0 < w && cols[x0] < colMin) x0++;
  let x1 = w;
  while (x1 > x0 && cols[x1 - 1] < colMin) x1--;
  return x0 < x1 && y0 < y1 ? { x0, y0, x1, y1 } : null;
}

/**
 * Fashion-MNIST framing (Zalando's recipe): crop to the object's box, scale its longest edge to
 * 28 px (area-averaged), and centre the shorter edge on the 28 × 28 canvas.
 */
export function frameFashion(values: Float32Array, w: number, box: Box): Float32Array {
  const cw = box.x1 - box.x0;
  const ch = box.y1 - box.y0;
  const crop = new Float32Array(cw * ch);
  for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) crop[y * cw + x] = values[(box.y0 + y) * w + box.x0 + x];
  const s = SIDE / Math.max(cw, ch);
  const tw = Math.max(1, Math.min(SIDE, Math.round(cw * s)));
  const th = Math.max(1, Math.min(SIDE, Math.round(ch * s)));
  const small = resample(crop, cw, ch, tw, th);
  const ox = Math.floor((SIDE - tw) / 2);
  const oy = Math.floor((SIDE - th) / 2);
  const out = new Float32Array(SIDE * SIDE);
  for (let y = 0; y < th; y++) for (let x = 0; x < tw; x++) out[(oy + y) * SIDE + ox + x] = Math.min(1, Math.max(0, small[y * tw + x]));
  return out;
}

/** Where the object sits in a photo, and its Fashion-MNIST-style input (null when no object is found). */
export interface FashionResult {
  x: Float32Array | null;
  box: Box | null;
}

/**
 * Fashion-MNIST style input from a photo of one item of clothing. The background colour is read
 * from the border; a pixel's value is how far it is from that colour, so the item comes out light
 * on a black background whatever the backdrop (Fashion-MNIST negates studio photos taken on
 * white). The item's box is cropped, its longest edge scaled to 28 px and the result centred.
 */
export function toFashion(img: ImageLike): FashionResult {
  const { width: w, height: h } = img;
  const n = w * h;
  const r = new Float32Array(n);
  const g = new Float32Array(n);
  const b = new Float32Array(n);
  const px = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    rgbAt(img, i, px);
    r[i] = px[0];
    g[i] = px[1];
    b[i] = px[2];
  }
  const median = (a: Float32Array) => {
    const border: number[] = [];
    for (let x = 0; x < w; x++) border.push(a[x], a[(h - 1) * w + x]);
    for (let y = 0; y < h; y++) border.push(a[y * w], a[y * w + w - 1]);
    border.sort((p, q) => p - q);
    return border[border.length >> 1];
  };
  const bg = [median(r), median(g), median(b)];
  const bgLum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2];
  const range = Math.max(bgLum, 1 - bgLum, 0.25);
  const mask = new Uint8Array(n);
  const values = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const dist = Math.max(Math.abs(r[i] - bg[0]), Math.abs(g[i] - bg[1]), Math.abs(b[i] - bg[2]));
    if (dist > 0.1) mask[i] = 1;
    const lum = 0.299 * r[i] + 0.587 * g[i] + 0.114 * b[i];
    values[i] = Math.abs(lum - bgLum) / range;
  }
  const box = maskBox(mask, w, h);
  if (!box) return { x: null, box: null };
  // Contrast: lift the object to the full range (as the dataset's sharpened, negated photos are),
  // but leave pale items pale; a small floor turns compression noise in the backdrop into black.
  const inside: number[] = [];
  for (let y = box.y0; y < box.y1; y++) for (let x = box.x0; x < box.x1; x++) inside.push(values[y * w + x]);
  inside.sort((p, q) => p - q);
  const hi = Math.max(0.35, inside[Math.min(inside.length - 1, Math.floor(inside.length * 0.99))]);
  const lo = 0.04;
  for (let i = 0; i < n; i++) values[i] = Math.min(1, Math.max(0, (values[i] - lo) / (hi - lo)));
  return { x: frameFashion(values, w, box), box };
}

/** Fashion-MNIST style input from a drawing (ink in [0, 1]): crop to the ink, fill the 28 × 28 frame. */
export function inkToFashion(ink: Float32Array, w: number, h: number): Float32Array | null {
  let max = 0;
  for (let i = 0; i < ink.length; i++) if (ink[i] > max) max = ink[i];
  if (max < 0.05) return null;
  const mask = new Uint8Array(ink.length);
  for (let i = 0; i < ink.length; i++) mask[i] = ink[i] > 0.1 * max ? 1 : 0;
  const box = maskBox(mask, w, h, 0);
  return box ? frameFashion(ink, w, box) : null;
}

/** Reader's nudges to an image input. */
export interface Adjustment {
  /** Mirror left to right. */
  flip: boolean;
  /** Added to every value, −1 … 1. */
  brightness: number;
  /** Multiplies the distance from mid-grey (0.5); 1 = unchanged. */
  contrast: number;
}

export const NO_ADJUSTMENT: Adjustment = { flip: false, brightness: 0, contrast: 1 };

export const isAdjusted = (a: Adjustment) => a.flip || a.brightness !== 0 || a.contrast !== 1;

/**
 * Applies flip, brightness and contrast to a channel-major image in [0, 1]:
 * v′ = clamp((v − 0.5) · contrast + 0.5 + brightness).
 */
export function adjustImage(x: ArrayLike<number>, c: number, h: number, w: number, a: Adjustment): Float32Array {
  const out = new Float32Array(c * h * w);
  for (let ch = 0; ch < c; ch++) {
    for (let y = 0; y < h; y++) {
      for (let col = 0; col < w; col++) {
        const src = a.flip ? w - 1 - col : col;
        const v = (x[(ch * h + y) * w + src] - 0.5) * a.contrast + 0.5 + a.brightness;
        out[(ch * h + y) * w + col] = Math.min(1, Math.max(0, v));
      }
    }
  }
  return out;
}
