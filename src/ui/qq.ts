import { histogram, niceTicks, type Line } from '../analysis/stats';
import { palette } from './theme';

/**
 * Canvas helpers for Q–Q plots and compact histograms, shared by the Distributions section and the
 * weight inspector. Callers size the canvas (fitCanvas) and pass the rectangle to draw into; every
 * colour is read from the theme at draw time, so redrawing after a theme change is enough.
 */

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface QQSeries {
  x: ArrayLike<number>;
  y: ArrayLike<number>;
  probs: ArrayLike<number>;
  /** 'filled': ink squares (the main series). 'hollow': muted outlines (a comparison). */
  style: 'filled' | 'hollow';
  /** Names the y value in tooltips: "sample", "now", "at init". */
  name: string;
}

export interface QQOptions {
  /** Axis titles, e.g. "Normal quantile" and "Sample quantile". */
  xTitle: string;
  yTitle: string;
  /** Names the x value in tooltips: "normal", "initial". */
  xName: string;
  /** Reference line: a fitted line (qqline), y = x for two-sample plots, or none. */
  line?: Line | 'identity' | null;
  /** Same range on both axes (two-sample plots, so y = x runs corner to corner). */
  equalAxes?: boolean;
  /** Shown instead of points when there are none. */
  empty?: string;
  /** A point to mark (the one under the pointer): accent square with a background ring. */
  highlight?: { series: number; index: number } | null;
}

export interface QQHit {
  series: number;
  index: number;
  text: string;
}

export interface QQPlot {
  /** The plotting area inside the rectangle, in canvas CSS pixels. */
  plot: Rect;
  /** Nearest point to (px, py), within `maxDist` pixels when given. */
  nearest(px: number, py: number, maxDist?: number): QQHit | null;
  /** Every series at the plotting position nearest to column `px` (a vertical crosshair). */
  column(px: number): string | null;
}

export interface HistOptions {
  bins?: number;
  /** A second distribution drawn as a muted outline on the same bins (e.g. initial weights). */
  compare?: ArrayLike<number> | null;
  compareName?: string;
  /** What one value is called in the tooltip ("weights", "values"). */
  noun?: string;
}

export interface HistPlot {
  plot: Rect;
  /** Tooltip text for the bin under column `px`, or null outside the bars. */
  at(px: number): string | null;
}

export const MONO = '"IBM Plex Mono", ui-monospace, monospace';
export const SANS = 'Archivo, "Helvetica Neue", Arial, sans-serif';

const SUP: Record<string, string> = { '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };
const minus = (s: string) => s.replace(/^-/, '−').replace(/e-/, 'e−');

/** A value for tooltips and stats: three significant digits, exponent form when tiny or huge. */
export function num(v: number, digits = 3): string {
  if (!Number.isFinite(v)) return '—';
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1e-3 && a < 1e5) return minus(String(Number(v.toPrecision(digits))));
  return minus(v.toExponential(digits - 1).replace('e+', 'e'));
}

/** Tick label formatter for an axis with ticks `step` apart; tiny or huge ranges get a ×10ⁿ factor. */
function axisFormat(step: number, maxAbs: number): { fmt: (v: number) => string; suffix: string } {
  const e = maxAbs > 0 ? Math.floor(Math.log10(maxAbs)) : 0;
  const scale = e <= -3 || e >= 5 ? Math.pow(10, e) : 1;
  const d = Math.max(0, Math.ceil(-Math.log10(step / scale) - 1e-9));
  const fmt = (v: number) => minus((v / scale).toFixed(d)).replace(/^−(0\.?0*)$/, '$1');
  const suffix = scale === 1 ? '' : ` ×10${String(e).split('').map((c) => SUP[c]).join('')}`;
  return { fmt, suffix };
}

/** Smallest and largest finite value across `arrays`, or null when there are none. */
function rawExtent(arrays: ArrayLike<number>[]): [number, number] | null {
  let lo = Infinity;
  let hi = -Infinity;
  for (const a of arrays) {
    for (let i = 0; i < a.length; i++) {
      const v = a[i];
      if (!Number.isFinite(v)) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  return lo <= hi ? [lo, hi] : null;
}

/** Axis domain: the data range plus 5% each side (a unit range around constant data). */
function extent(arrays: ArrayLike<number>[]): [number, number] {
  const r = rawExtent(arrays);
  if (!r) return [-1, 1];
  const [lo, hi] = r;
  if (lo === hi) {
    const pad = Math.abs(lo) * 0.1 || 1;
    return [lo - pad, hi + pad];
  }
  const pad = (hi - lo) * 0.05;
  return [lo - pad, hi + pad];
}

/** Draws a Q–Q plot into `rect` and returns hit-testing helpers for tooltips. */
export function drawQQ(ctx: CanvasRenderingContext2D, rect: Rect, series: QQSeries[], opts: QQOptions): QQPlot {
  const p = palette();
  if (!series.some((s) => Math.min(s.x.length, s.y.length) > 0)) {
    // Nothing to plot: a frame and a message, no ticks.
    const plot: Rect = { x: rect.x + 1, y: rect.y + 26, w: rect.w - 2, h: Math.max(10, rect.h - 26 - 36) };
    ctx.strokeStyle = p.hair;
    ctx.lineWidth = 1;
    ctx.strokeRect(plot.x + 0.5, plot.y + 0.5, plot.w - 1, plot.h - 1);
    ctx.fillStyle = p.muted;
    ctx.font = `500 12px ${SANS}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(opts.empty ?? 'No values', plot.x + plot.w / 2, plot.y + plot.h / 2);
    return { plot, nearest: () => null, column: () => null };
  }
  let [x0, x1] = extent(series.map((s) => s.x));
  let [y0, y1] = extent(series.map((s) => s.y));
  if (opts.equalAxes) {
    x0 = y0 = Math.min(x0, y0);
    x1 = y1 = Math.max(x1, y1);
  }

  const T = 26;
  const B = 36;
  const R = 6;
  const yTicks = niceTicks(y0, y1, Math.max(3, Math.min(6, Math.floor((rect.h - T - B) / 34))));
  const yAxis = axisFormat(yTicks.step, Math.max(Math.abs(y0), Math.abs(y1)));
  ctx.font = `400 10px ${MONO}`;
  let labelW = 0;
  for (const t of yTicks.ticks) labelW = Math.max(labelW, ctx.measureText(yAxis.fmt(t)).width);
  const L = Math.ceil(labelW) + 10;
  const plot: Rect = { x: rect.x + L, y: rect.y + T, w: Math.max(10, rect.w - L - R), h: Math.max(10, rect.h - T - B) };
  const xTicks = niceTicks(x0, x1, Math.max(3, Math.min(6, Math.floor(plot.w / 46))));
  const xAxis = axisFormat(xTicks.step, Math.max(Math.abs(x0), Math.abs(x1)));
  const tx = (v: number) => plot.x + ((v - x0) / (x1 - x0)) * plot.w;
  const ty = (v: number) => plot.y + plot.h - ((v - y0) / (y1 - y0)) * plot.h;

  // Grid, ticks and titles.
  ctx.lineWidth = 1;
  ctx.fillStyle = p.hair;
  for (const t of yTicks.ticks) ctx.fillRect(plot.x, Math.round(ty(t)), plot.w, 1);
  for (const t of xTicks.ticks) ctx.fillRect(Math.round(tx(t)), plot.y, 1, plot.h);
  ctx.fillStyle = p.ink;
  ctx.fillRect(plot.x, plot.y + plot.h, plot.w, 1);
  ctx.fillRect(plot.x - 1, plot.y, 1, plot.h + 1);

  ctx.fillStyle = p.muted;
  ctx.font = `400 10px ${MONO}`;
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (const t of yTicks.ticks) ctx.fillText(yAxis.fmt(t), plot.x - 6, ty(t));
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (const t of xTicks.ticks) ctx.fillText(xAxis.fmt(t), tx(t), plot.y + plot.h + 5);
  ctx.font = `500 11px ${SANS}`;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'alphabetic';
  ctx.fillText(opts.yTitle + yAxis.suffix, rect.x, rect.y + 11);
  ctx.textAlign = 'right';
  ctx.fillText(opts.xTitle + xAxis.suffix, plot.x + plot.w, plot.y + plot.h + 31);

  const screen = series.map((s) => {
    const n = Math.min(s.x.length, s.y.length);
    const sx = new Float32Array(n);
    const sy = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      sx[i] = tx(s.x[i]);
      sy[i] = ty(s.y[i]);
    }
    return { sx, sy };
  });

  // Reference line, clipped to the plot.
  const line = opts.line === 'identity' ? { slope: 1, intercept: 0 } : opts.line;
  if (line && Number.isFinite(line.slope) && Number.isFinite(line.intercept)) {
    ctx.save();
    ctx.beginPath();
    ctx.rect(plot.x, plot.y, plot.w, plot.h);
    ctx.clip();
    ctx.strokeStyle = p.accent;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(tx(x0), ty(line.intercept + line.slope * x0));
    ctx.lineTo(tx(x1), ty(line.intercept + line.slope * x1));
    ctx.stroke();
    ctx.restore();
  }
  // Comparison series first, so the main series sits on top.
  const order = series.map((_, i) => i).sort((a, b) => (series[a].style === 'hollow' ? 0 : 1) - (series[b].style === 'hollow' ? 0 : 1));
  for (const si of order) {
    const { sx, sy } = screen[si];
    if (series[si].style === 'hollow') {
      ctx.strokeStyle = p.muted;
      ctx.lineWidth = 1;
      for (let i = 0; i < sx.length; i++) ctx.strokeRect(sx[i] - 1.75, sy[i] - 1.75, 3.5, 3.5);
    } else {
      ctx.fillStyle = p.ink;
      for (let i = 0; i < sx.length; i++) ctx.fillRect(sx[i] - 1.5, sy[i] - 1.5, 3, 3);
    }
  }
  const hl = opts.highlight;
  if (hl && screen[hl.series] && hl.index < screen[hl.series].sx.length) {
    const hx = screen[hl.series].sx[hl.index];
    const hy = screen[hl.series].sy[hl.index];
    ctx.fillStyle = p.bg;
    ctx.fillRect(hx - 5.5, hy - 5.5, 11, 11);
    ctx.fillStyle = p.accent;
    ctx.fillRect(hx - 3.5, hy - 3.5, 7, 7);
  }

  const describe = (si: number, i: number) => {
    const s = series[si];
    return `p ${num(s.probs[i])} · ${opts.xName} ${num(s.x[i])} · ${s.name} ${num(s.y[i])}`;
  };

  return {
    plot,
    nearest(px, py, maxDist = Infinity) {
      let bs = -1;
      let bi = -1;
      let bestD = maxDist * maxDist;
      for (let si = 0; si < screen.length; si++) {
        const { sx, sy } = screen[si];
        for (let i = 0; i < sx.length; i++) {
          const d = (sx[i] - px) ** 2 + (sy[i] - py) ** 2;
          if (d < bestD) {
            bestD = d;
            bs = si;
            bi = i;
          }
        }
      }
      return bs < 0 ? null : { series: bs, index: bi, text: describe(bs, bi) };
    },
    column(px) {
      const lines: string[] = [];
      screen.forEach(({ sx }, si) => {
        if (!sx.length) return;
        let bi = 0;
        for (let i = 1; i < sx.length; i++) if (Math.abs(sx[i] - px) < Math.abs(sx[bi] - px)) bi = i;
        const s = series[si];
        if (!lines.length) lines.push(`p ${num(s.probs[bi])} · ${opts.xName} ${num(s.x[bi])}`);
        lines.push(`${s.name} ${num(s.y[bi])}`);
      });
      return lines.length ? lines.join('\n') : null;
    },
  };
}

/** A compact histogram strip: ink bars, a zero marker, the range at the ends. */
export function drawHistogram(ctx: CanvasRenderingContext2D, rect: Rect, values: ArrayLike<number>, opts: HistOptions = {}): HistPlot {
  const p = palette();
  const compare = opts.compare ?? null;
  const labelH = 15;
  const plot: Rect = { x: rect.x, y: rect.y, w: rect.w, h: Math.max(8, rect.h - labelH) };
  // The exact range, so the end labels name the real minimum and maximum.
  let [lo, hi] = rawExtent(compare ? [values, compare] : [values]) ?? [-1, 1];
  if (!(hi > lo)) {
    lo -= 0.5;
    hi += 0.5;
  }
  // Rice rule (2·n^⅓ bins), so a 72-weight layer gets ~8 bins and 25,000 weights get ~58; at
  // least 5 px per bar.
  let n = 0;
  for (let i = 0; i < values.length; i++) if (Number.isFinite(values[i])) n++;
  const bins = opts.bins ?? Math.max(6, Math.min(Math.floor(rect.w / 5), 64, Math.round(2 * Math.cbrt(n))));
  const counts = histogram(values, bins, lo, hi);
  const cmp = compare ? histogram(compare, bins, lo, hi) : null;
  let total = 0;
  for (const c of counts) total += c;

  // A single towering bin (exact zeros after ReLU) would flatten everything else: clip it.
  const sorted = Array.from(counts).concat(cmp ? Array.from(cmp) : []).sort((a, b) => b - a);
  const top = sorted[0] ?? 0;
  const second = sorted.find((c) => c < top) ?? 0;
  const cap = top > 4 * second && second > 0 ? second * 1.6 : top || 1;
  const bw = plot.w / bins;
  const base = plot.y + plot.h;
  const hOf = (c: number) => (Math.min(c, cap) / cap) * (plot.h - 2);

  ctx.fillStyle = p.ink;
  const gap = bw >= 3 ? 1 : 0;
  for (let k = 0; k < bins; k++) {
    if (!counts[k]) continue;
    const hh = Math.max(1, hOf(counts[k]));
    ctx.fillRect(plot.x + k * bw, base - hh, bw - gap, hh);
  }
  // Clipped bars: a gap near the top marks the break.
  for (let k = 0; k < bins; k++) {
    if (counts[k] > cap) ctx.clearRect(plot.x + k * bw - 1, plot.y + 5, bw + 1, 2);
  }
  if (cmp) {
    ctx.strokeStyle = p.muted;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let k = 0; k < bins; k++) {
      const y = base - hOf(cmp[k]) + 0.5;
      if (k === 0) ctx.moveTo(plot.x, y);
      else ctx.lineTo(plot.x + k * bw, y);
      ctx.lineTo(plot.x + (k + 1) * bw, y);
    }
    ctx.stroke();
  }
  ctx.fillStyle = p.ink;
  ctx.fillRect(plot.x, base, plot.w, 1);

  // Range labels and the zero marker.
  ctx.font = `400 10px ${MONO}`;
  ctx.textBaseline = 'top';
  ctx.fillStyle = p.muted;
  ctx.textAlign = 'left';
  const loText = num(lo);
  const hiText = num(hi);
  ctx.fillText(loText, plot.x, base + 4);
  ctx.textAlign = 'right';
  ctx.fillText(hiText, plot.x + plot.w, base + 4);
  if (lo <= 0 && hi >= 0) {
    const zx = Math.round(plot.x + ((0 - lo) / (hi - lo)) * plot.w);
    ctx.fillStyle = p.accent;
    ctx.fillRect(zx, plot.y, 1, plot.h + 4);
    const loW = ctx.measureText(loText).width;
    const hiW = ctx.measureText(hiText).width;
    if (zx - plot.x > loW + 12 && plot.x + plot.w - zx > hiW + 12) {
      ctx.fillStyle = p.muted;
      ctx.textAlign = 'center';
      ctx.fillText('0', zx, base + 4);
    }
  }

  const noun = opts.noun ?? 'values';
  return {
    plot,
    at(px) {
      if (px < plot.x || px >= plot.x + plot.w) return null;
      const k = Math.min(bins - 1, Math.max(0, Math.floor((px - plot.x) / bw)));
      const a = lo + (k / bins) * (hi - lo);
      const b = lo + ((k + 1) / bins) * (hi - lo);
      const share = total ? ` (${((100 * counts[k]) / total).toFixed(1)}%)` : '';
      let text = `${num(a)} to ${num(b)}\n${counts[k].toLocaleString('en-US')} ${noun}${share}`;
      if (cmp) text += `\n${opts.compareName ?? 'compare'} ${cmp[k].toLocaleString('en-US')}`;
      return text;
    },
  };
}
