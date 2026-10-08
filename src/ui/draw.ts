import { formatCell } from './format';
import { css, diverging, palette, sequential, type RGB } from './theme';

export type MatrixMode = 'heat' | 'hinton' | 'numbers';

/** Sizes a canvas for crisp drawing at the device pixel ratio and returns a context in CSS pixels. */
export function fitCanvas(canvas: HTMLCanvasElement, w: number, h: number): CanvasRenderingContext2D {
  const dpr = Math.min(3, window.devicePixelRatio || 1);
  const W = Math.max(1, Math.round(w * dpr));
  const H = Math.max(1, Math.round(h * dpr));
  if (canvas.width !== W) canvas.width = W;
  if (canvas.height !== H) canvas.height = H;
  canvas.style.width = `${w}px`;
  canvas.style.height = `${h}px`;
  const ctx = canvas.getContext('2d')!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return ctx;
}

let scratch: HTMLCanvasElement | null = null;
const tmp: RGB = [0, 0, 0];

export function maxAbs(a: ArrayLike<number>, off = 0, n = a.length - off): number {
  let m = 0;
  for (let i = off; i < off + n; i++) {
    const v = Math.abs(a[i]);
    if (v > m) m = v;
  }
  return m;
}

export function hasNegative(a: ArrayLike<number>, off = 0, n = a.length - off): boolean {
  for (let i = off; i < off + n; i++) if (a[i] < -1e-9) return true;
  return false;
}

/**
 * Paints an H×W block of `data` (starting at `off`) as pixels into the rectangle (x, y, dw, dh).
 * Signed data uses the diverging map scaled by ±max; non-negative data uses paper→ink.
 */
export function drawMap(
  ctx: CanvasRenderingContext2D,
  data: ArrayLike<number>,
  off: number,
  H: number,
  W: number,
  x: number,
  y: number,
  dw: number,
  dh: number,
  signed: boolean,
  max: number,
): void {
  if (!scratch) scratch = document.createElement('canvas');
  if (scratch.width < W) scratch.width = W;
  if (scratch.height < H) scratch.height = H;
  const sctx = scratch.getContext('2d')!;
  const img = sctx.createImageData(W, H);
  const d = img.data;
  const inv = max > 0 ? 1 / max : 0;
  for (let i = 0; i < H * W; i++) {
    const v = data[off + i] * inv;
    const c = signed ? diverging(v, tmp) : sequential(v, tmp);
    d[4 * i] = c[0];
    d[4 * i + 1] = c[1];
    d[4 * i + 2] = c[2];
    d[4 * i + 3] = 255;
  }
  sctx.putImageData(img, 0, 0);
  const smooth = ctx.imageSmoothingEnabled;
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(scratch, 0, 0, W, H, x, y, dw, dh);
  ctx.imageSmoothingEnabled = smooth;
}

/**
 * Draws a rows×cols matrix as a heat map, a Hinton diagram or annotated numbers.
 * Cells are cw×ch CSS pixels; values are scaled by `max` (absolute).
 */
export function drawMatrix(
  ctx: CanvasRenderingContext2D,
  data: ArrayLike<number>,
  off: number,
  rows: number,
  cols: number,
  x: number,
  y: number,
  cw: number,
  ch: number,
  mode: MatrixMode,
  max: number,
  signed = true,
): void {
  const p = palette();
  const inv = max > 0 ? 1 / max : 0;
  if (mode === 'heat') {
    if (rows * cols <= 64) {
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const v = data[off + r * cols + c] * inv;
          ctx.fillStyle = css(signed ? diverging(v, tmp) : sequential(v, tmp));
          ctx.fillRect(x + c * cw, y + r * ch, cw, ch);
        }
      }
    } else {
      drawMap(ctx, data, off, rows, cols, x, y, cols * cw, rows * ch, signed, max);
    }
    return;
  }
  ctx.fillStyle = p.surface;
  ctx.fillRect(x, y, cols * cw, rows * ch);
  if (mode === 'hinton') {
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const v = data[off + r * cols + c] * inv;
        const s = Math.sqrt(Math.min(1, Math.abs(v)));
        const side = Math.min(cw, ch) * 0.92 * s;
        if (side < 0.3) continue;
        ctx.fillStyle = v >= 0 ? p.accent : p.neg;
        ctx.fillRect(x + c * cw + (cw - side) / 2, y + r * ch + (ch - side) / 2, side, side);
      }
    }
    return;
  }
  // numbers: tinted cell + value
  const fs = Math.max(7, Math.min(12, ch * 0.5, cw * 0.27));
  ctx.font = `500 ${fs}px "IBM Plex Mono", ui-monospace, monospace`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const raw = data[off + r * cols + c];
      const v = raw * inv;
      const col = signed ? diverging(v * 0.55, tmp) : sequential(v * 0.45, tmp);
      ctx.fillStyle = css(col);
      ctx.fillRect(x + c * cw, y + r * ch, cw, ch);
      ctx.fillStyle = p.ink;
      ctx.fillText(formatCell(raw), x + c * cw + cw / 2, y + r * ch + ch / 2 + 0.5);
    }
  }
  ctx.strokeStyle = p.hair;
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let r = 0; r <= rows; r++) {
    ctx.moveTo(x, y + r * ch + 0.5);
    ctx.lineTo(x + cols * cw, y + r * ch + 0.5);
  }
  for (let c = 0; c <= cols; c++) {
    ctx.moveTo(x + c * cw + 0.5, y);
    ctx.lineTo(x + c * cw + 0.5, y + rows * ch);
  }
  ctx.stroke();
}

/** Small helper for a 28×28 (or any) single-channel thumbnail canvas element. */
export function thumbCanvas(data: ArrayLike<number>, H = 28, W = 28, px = 28): HTMLCanvasElement {
  const c = document.createElement('canvas');
  paintThumb(c, data, H, W, px);
  return c;
}

export function paintThumb(c: HTMLCanvasElement, data: ArrayLike<number>, H = 28, W = 28, px = 28): void {
  const ctx = fitCanvas(c, px, px * (H / W));
  drawMap(ctx, data, 0, H, W, 0, 0, px, px * (H / W), false, 1);
}

export function frame(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, color: string, lw = 1): void {
  ctx.strokeStyle = color;
  ctx.lineWidth = lw;
  ctx.strokeRect(x + lw / 2, y + lw / 2, w - lw, h - lw);
}
