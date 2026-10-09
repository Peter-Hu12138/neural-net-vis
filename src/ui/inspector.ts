import { initialWeights, select, setMode } from '../actions';
import { summarize, summarizeFrozen } from '../analysis/stats';
import { featureDefs } from '../data/features';
import type { ConvBlock, DenseBlock } from '../nn/network';
import { store, type WeightMode } from '../store';
import { layerDetail, layerName } from './builder';
import { namesAreGlyphs, shortNames } from './charts';
import { $, clear, h, int, segmented, selectField } from './dom';
import { drawMatrix, fitCanvas, maxAbs, type MatrixMode } from './draw';
import { drawQQ, fixed, SANS as SANS_FONT, sig, type QQSeries } from './qq';
import { css, diverging, onThemeChange, palette, sequential, type RGB } from './theme';
import { hideTip, showTip } from './tip';
import './inspector.css';

interface Region {
  x: number;
  y: number;
  w: number;
  h: number;
  rows: number;
  cols: number;
  unit?: number;
  label: (r: number, c: number) => string;
}

const MODES: { value: WeightMode; label: string }[] = [
  { value: 'heat', label: 'Heatmap' },
  { value: 'hinton', label: 'Hinton' },
  { value: 'numbers', label: 'Numbers' },
  { value: 'hist', label: 'Histogram' },
  { value: 'qq', label: 'Q–Q' },
];

const NOTES: Record<WeightMode, string> = {
  heat: 'Each cell is one weight: red is positive, blue is negative, paper-white is near zero. The colour scale is shared across the layer.',
  hinton: 'Hinton diagram: the area of each square is the weight’s magnitude, the colour its sign. Small weights almost vanish.',
  numbers: 'Raw weight values, tinted by sign. Choose which filter or unit to read.',
  hist: 'Distribution of this layer’s weights now (solid) against the values they started from (outline).',
  qq: 'Q–Q plot: each weight quantile against the same quantile of a normal distribution. A straight line means a normal shape; bent ends mean heavier or lighter tails than normal.',
};

const CHANNELS = ['red', 'green', 'blue'];
const CH = ['R', 'G', 'B'];

function stats(a: Float32Array) {
  let mn = Infinity;
  let mx = -Infinity;
  let s = 0;
  let s2 = 0;
  for (const v of a) {
    if (v < mn) mn = v;
    if (v > mx) mx = v;
    s += v;
    s2 += v * v;
  }
  const mean = s / a.length;
  return { mn, mx, mean, std: Math.sqrt(Math.max(0, s2 / a.length - mean * mean)), norm: Math.sqrt(s2) };
}

const MONO = '"IBM Plex Mono", ui-monospace, monospace';
/** Weight values in tooltips and labels: four decimals, true minus sign, never "−0.0000". */
const wv = (v: number) => fixed(v, 4);

/**
 * A matrix cell's printed value: two decimals without the leading zero (".21", "−.07"), a true
 * minus sign, "0" for values that round to zero, one decimal from 10 up. One precision per grid, no
 * exponent forms, so columns read evenly.
 */
export function cellText(v: number): string {
  if (!Number.isFinite(v)) return '—';
  if (Math.abs(v) >= 9.995) return fixed(v, Math.abs(v) >= 99.95 ? 0 : 1);
  const s = fixed(v, 2);
  if (s === '0.00') return '0';
  return s.replace(/^(−?)0\./, '$1.');
}

const tint: RGB = [0, 0, 0];

/** drawMatrix's "numbers" mode with cellText (true minus sign): a tinted cell, its value, hairlines. */
export function drawNumbers(ctx: CanvasRenderingContext2D, data: ArrayLike<number>, off: number, rows: number, cols: number, x: number, y: number, cw: number, ch: number, max: number, signed = true): void {
  const p = palette();
  const inv = max > 0 ? 1 / max : 0;
  const fs = Math.max(7, Math.min(12, ch * 0.5, cw * 0.27));
  ctx.font = `500 ${fs}px ${MONO}`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const raw = data[off + r * cols + c];
      const v = raw * inv;
      ctx.fillStyle = css(signed ? diverging(v * 0.55, tint) : sequential(v * 0.45, tint));
      ctx.fillRect(x + c * cw, y + r * ch, cw, ch);
      ctx.fillStyle = p.ink;
      ctx.fillText(cellText(raw), x + c * cw + cw / 2, y + r * ch + ch / 2 + 0.5);
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

/** drawMatrix, with numbers printed by drawNumbers. */
export function matrix(ctx: CanvasRenderingContext2D, data: ArrayLike<number>, off: number, rows: number, cols: number, x: number, y: number, cw: number, ch: number, mode: MatrixMode, max: number, signed = true): void {
  if (mode === 'numbers') drawNumbers(ctx, data, off, rows, cols, x, y, cw, ch, max, signed);
  else drawMatrix(ctx, data, off, rows, cols, x, y, cw, ch, mode, max, signed);
}

let colourScratch: HTMLCanvasElement | null = null;

/**
 * Draws three channel planes of `W` (channel c starts at off + c·stride, each H×Wd) as one colour
 * image, scaled around mid grey by the largest magnitude among them: zero is grey, a positive red
 * weight adds red, a negative one takes it away. This is the input pattern the unit responds to most.
 */
function drawColourWeights(ctx: CanvasRenderingContext2D, W: Float32Array, off: number, stride: number, H: number, Wd: number, x: number, y: number, dw: number, dh: number): void {
  const m = maxAbs(W, off, 2 * stride + H * Wd) || 1;
  if (!colourScratch) colourScratch = document.createElement('canvas');
  if (colourScratch.width < Wd) colourScratch.width = Wd;
  if (colourScratch.height < H) colourScratch.height = H;
  const sctx = colourScratch.getContext('2d')!;
  const img = sctx.createImageData(Wd, H);
  const d = img.data;
  const n = H * Wd;
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) d[4 * i + c] = Math.round(127.5 + 127.5 * (W[off + c * stride + i] / m));
    d[4 * i + 3] = 255;
  }
  sctx.putImageData(img, 0, 0);
  const smooth = ctx.imageSmoothingEnabled;
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(colourScratch, 0, 0, Wd, H, x, y, dw, dh);
  ctx.imageSmoothingEnabled = smooth;
}

export function mountInspector(): void {
  const root = $('inspector');
  const head = h('div', { class: 'insp-head' });
  const frozenEl = h('p', { class: 'insp-frozen', hidden: true });
  const note = h('p', { class: 'hint', style: { marginBottom: '12px' } });
  const canvas = h('canvas', { role: 'img', 'aria-label': 'Weights of the selected layer' }) as HTMLCanvasElement;
  const box = h('div', { class: 'insp-canvas' }, canvas);
  const statsEl = h('div', { class: 'insp-stats' });
  root.append(head, frozenEl, note, box, statsEl);

  let regions: Region[] = [];

  const isOutput = (i: number) => i === store.net.blocks.length - 1;
  /** Block i reads the input image itself (so its channels are colours or grey). */
  const readsImage = (i: number) => i === 0 && store.info.kind === 'image';
  const colourIn = (i: number) => readsImage(i) && store.net.blocks[0].inShape.c === 3;
  /** Class names for the output layer: digits stay digits, other datasets use their names. */
  const classLabel = (k: number) => (namesAreGlyphs(store.info) ? store.info.glyphs[k] : store.info.classes[k]);

  const unitCount = (i: number) => {
    const b = store.net.blocks[i];
    return b.kind === 'conv' ? b.spec.filters : b.spec.units;
  };
  const unitName = (i: number, u: number) => {
    const b = store.net.blocks[i];
    if (isOutput(i)) return namesAreGlyphs(store.info) ? `Digit ${u}` : store.info.classes[u] ?? `Class ${u}`;
    return b.kind === 'conv' ? `Filter ${u + 1}` : `Unit ${u + 1}`;
  };
  /** Short label of unit j in block i, for tiles and matrix rows. */
  const unitLabel = (i: number, j: number) => (isOutput(i) ? classLabel(j) : `u${j + 1}`);
  /** Labels of block i's inputs when it reads a flat vector: features, or the previous layer's units. */
  const inputLabels = (i: number): string[] | null => {
    const b = store.net.blocks[i];
    if (b.kind !== 'dense' || b.inShape.h > 1 || b.inShape.w > 1) return null;
    const N = b.inSize;
    if (i === 0) {
      const info = store.info;
      if (info.kind !== 'points' || store.features.length !== N) return null;
      try {
        return featureDefs(info.dims!, store.features).map((f) => f.label);
      } catch {
        return null;
      }
    }
    const prev = store.net.blocks[i - 1];
    return Array.from({ length: N }, (_, k) => (prev.kind === 'conv' ? `c${k + 1}` : `u${k + 1}`));
  };
  const layerTitle = (i: number) => (isOutput(i) ? 'Output' : layerName(store.spec[i] ?? null, i));

  const renderHead = () => {
    clear(head);
    const blocks = store.net.blocks;
    const sel = Math.min(store.selected, blocks.length - 1);
    head.append(
      selectField(
        'insp-layer',
        'Layer',
        blocks.map((_, i) => {
          const spec = isOutput(i) ? null : store.spec[i];
          const detail = spec ? layerDetail(spec) : `${store.classes} · softmax`;
          return { value: i, label: `${layerName(spec, i)} · ${detail}${store.isFrozen(i) ? ' · frozen' : ''}` };
        }),
        sel,
        (v) => select(v, null),
      ),
      h('div', { class: 'field' }, h('span', { class: 'label' }, 'View'), segmented(MODES, store.mode, setMode, 'Weight view')),
    );
    renderNotes();
    // Histogram and Q–Q show the whole layer, so a unit choice would do nothing there.
    if (store.mode === 'hist' || store.mode === 'qq') return;
    const n = unitCount(sel);
    const unit = store.selectedUnit ?? 0;
    head.append(
      selectField(
        'insp-unit',
        isOutput(sel) ? (namesAreGlyphs(store.info) ? 'Digit' : 'Class') : blocks[sel].kind === 'conv' ? 'Filter' : 'Unit',
        Array.from({ length: n }, (_, u) => ({ value: u, label: unitName(sel, u) })),
        Math.min(unit, n - 1),
        (u) => select(sel, u),
      ),
    );
  };

  /** The mode's note plus a sentence for what this layer's picture means. */
  const renderNotes = () => {
    const blocks = store.net.blocks;
    const i = Math.min(store.selected, blocks.length - 1);
    const b = blocks[i];
    const mode = store.mode;
    let extra = '';
    if (mode !== 'hist' && mode !== 'qq') {
      if (b.kind === 'conv' && colourIn(i)) {
        extra = ` The square patch shows each filter’s whole ${b.k}×${b.k}×3 kernel as a colour, scaled per filter around mid grey: the colour pattern that excites the filter most. R, G and B beside it are the same weights, one channel at a time.`;
      } else if (b.kind === 'dense' && colourIn(i) && mode === 'heat') {
        extra = ` Here each ${isOutput(i) ? 'class' : 'unit'}’s ${int(b.inSize)} weights are drawn as a ${b.inShape.h}×${b.inShape.w} colour image, scaled around mid grey: the picture that excites it most.`;
      } else if (b.kind === 'dense' && i === 0 && store.info.kind === 'points') {
        extra = ' Columns are the input features, rows the units of this layer.';
      }
    }
    note.textContent = NOTES[mode] + extra;
    const frozen = store.isFrozen(i);
    frozenEl.hidden = !frozen;
    clear(frozenEl);
    if (frozen) {
      frozenEl.append(
        h('span', { class: 'tag' }, 'Frozen'),
        ' ',
        h('b', null, 'Frozen: training leaves these weights alone.'),
        ' The optimizer skips its weights and bias; the error signal still flows through it to any trainable layer below.',
      );
    }
  };

  const render = () => {
    const blocks = store.net.blocks;
    const i = Math.min(store.selected, blocks.length - 1);
    const b = blocks[i];
    const availW = Math.max(280, box.clientWidth - 2);
    regions = [];
    const max = maxAbs(b.W) || 1;
    const unit = Math.min(store.selectedUnit ?? 0, unitCount(i) - 1);
    const mode = store.mode;
    const name = layerTitle(i);

    if (mode === 'hist') {
      renderHist(b.W, initialWeights[2 * i] ?? null, availW);
      describeCanvas('hist', `Histogram of the ${int(b.W.length)} weights of ${name} now, with the distribution they started from as an outline`);
    } else if (mode === 'qq') {
      renderQQMode(b.W, initialWeights[2 * i] ?? null, availW);
      describeCanvas('qq', `Q–Q plot of the ${int(b.W.length)} weights of ${name} against a normal distribution, now and at initialisation`);
    } else if (b.kind === 'conv') renderConv(b, i, mode, max, unit, availW);
    else renderDense(b, i, mode, max, unit, availW);

    const s = stats(b.W);
    const sb = stats(b.b);
    const shape = b.kind === 'conv' ? `${b.spec.filters}×${b.inShape.c}×${b.k}×${b.k}` : `${b.spec.units}×${b.inSize}`;
    clear(statsEl);
    const kv = (k: string, v: string) => h('span', null, `${k} `, h('b', null, v));
    statsEl.append(
      kv('W', shape),
      kv('mean', fixed(s.mean, 4)),
      kv('std', fixed(s.std, 4)),
      kv('min', fixed(s.mn, 3)),
      kv('max', fixed(s.mx, 3)),
      kv('‖W‖', fixed(s.norm, 2)),
      kv('bias mean', fixed(sb.mean, 4)),
      kv('step', int(store.weightsStep)),
    );
    if (store.isFrozen(i)) statsEl.append(h('span', { class: 'insp-frozen-kv' }, 'frozen'));
  };

  const describeCanvas = (view: string, label: string) => {
    canvas.dataset.view = view;
    canvas.setAttribute('aria-label', label);
  };

  const label = (ctx: CanvasRenderingContext2D, text: string, x: number, y: number, align: CanvasTextAlign = 'left', color?: string) => {
    ctx.font = `500 10px ${MONO}`;
    ctx.textAlign = align;
    ctx.textBaseline = 'middle';
    ctx.fillStyle = color ?? palette().muted;
    ctx.fillText(text, x, y);
  };

  const measure = (texts: string[], font = `500 10px ${MONO}`) => {
    const ctx = canvas.getContext('2d')!;
    ctx.font = font;
    return Math.ceil(texts.reduce((m, t) => Math.max(m, ctx.measureText(t).width), 0));
  };

  const ring = (ctx: CanvasRenderingContext2D, x: number, y: number, w: number, hh: number) => {
    ctx.strokeStyle = palette().accent;
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, hh);
  };

  function renderConv(b: ConvBlock, i: number, mode: Exclude<WeightMode, 'hist' | 'qq'>, max: number, unit: number, availW: number) {
    const p = palette();
    const F = b.spec.filters;
    const C = b.inShape.c;
    const k = b.k;
    const kk = k * k;
    const name = layerTitle(i);
    const colour = colourIn(i);
    const chName = (c: number) => (colour ? CHANNELS[c] : `ch ${c + 1}`);
    const modeName = mode === 'heat' ? 'heatmaps' : mode === 'hinton' ? 'Hinton diagrams' : 'numbers';
    if (mode === 'numbers') {
      const cw = 42;
      const ch = 22;
      const kw = k * cw;
      const gap = 18;
      // A colour filter gets its patch as the first slot, then one kernel per channel.
      const slots = C + (colour ? 1 : 0);
      const perRow = Math.max(1, Math.floor((availW - 10 + gap) / (kw + gap)));
      const rows = Math.ceil(slots / perRow);
      const W = Math.max(availW, Math.min(slots, perRow) * (kw + gap));
      const H = 30 + rows * (k * ch + 34) + 30;
      const ctx = fitCanvas(canvas, W, H);
      label(ctx, colour ? `Filter ${unit + 1}: its ${k}×${k}×3 kernel as a colour patch, then one ${k}×${k} kernel per colour channel` : `Filter ${unit + 1}: one ${k}×${k} kernel per input channel`, 0, 12, 'left', p.ink);
      for (let s = 0; s < slots; s++) {
        const x = (s % perRow) * (kw + gap);
        const y = 30 + Math.floor(s / perRow) * (k * ch + 34);
        if (colour && s === 0) {
          const side = Math.min(kw, k * ch);
          label(ctx, 'colour patch', x, y + 8);
          drawColourWeights(ctx, b.W, unit * C * kk, kk, k, k, x, y + 18, side, side);
          ctx.strokeStyle = p.hair;
          ctx.lineWidth = 1;
          ctx.strokeRect(x - 0.5, y + 17.5, side + 1, side + 1);
          regions.push({ x, y: y + 18, w: side, h: side, rows: k, cols: k, unit, label: (r, cc) => colourTip(b, unit, r, cc) });
          continue;
        }
        const c = s - (colour ? 1 : 0);
        label(ctx, colour ? `${CHANNELS[c]} channel` : `in-channel ${c + 1}`, x, y + 8);
        matrix(ctx, b.W, (unit * C + c) * kk, k, k, x, y + 18, cw, ch, 'numbers', max);
        regions.push({ x, y: y + 18, w: kw, h: k * ch, rows: k, cols: k, unit, label: (r, cc) => `W[filter ${unit + 1}, ${chName(c)}, ${r}, ${cc}] = ${wv(b.W[(unit * C + c) * kk + r * k + cc])}` });
      }
      label(ctx, `bias b = ${fixed(b.b[unit], 5)}`, 0, H - 12, 'left', p.ink);
      describeCanvas(
        colour ? 'conv-colour' : 'conv',
        colour
          ? `Filter ${unit + 1} of ${name}: its ${k}×${k}×3 kernel as a colour patch, and the values of its red, green and blue ${k}×${k} kernels`
          : `Filter ${unit + 1} of ${name}: the values of its ${C} ${k}×${k} kernel${C > 1 ? 's' : ''}, one per input channel`,
      );
      return;
    }
    const mm: MatrixMode = mode;
    if (colour) {
      // One card per filter: the colour patch, then the red, green and blue kernels.
      const cell = k === 3 ? 12 : 8;
      const kw = k * cell;
      const P = kw + 8;
      const gap = 4;
      const cardW = P + 10 + 3 * kw + 2 * gap;
      const gapX = 22;
      const cardH = 14 + P + 16;
      const perRow = Math.max(1, Math.min(F, Math.floor((availW + gapX) / (cardW + gapX))));
      const rows = Math.ceil(F / perRow);
      const ctx = fitCanvas(canvas, Math.max(availW, perRow * (cardW + gapX) - gapX + 4), rows * (cardH + 12) + 4);
      for (let f = 0; f < F; f++) {
        const x = 2 + (f % perRow) * (cardW + gapX);
        const y = 2 + Math.floor(f / perRow) * (cardH + 12);
        label(ctx, `f${f + 1}`, x, y + 6, 'left', f === unit ? p.accent : undefined);
        const py = y + 14;
        drawColourWeights(ctx, b.W, f * C * kk, kk, k, k, x, py, P, P);
        ctx.strokeStyle = p.hair;
        ctx.lineWidth = 1;
        ctx.strokeRect(x - 0.5, py - 0.5, P + 1, P + 1);
        regions.push({ x, y: py, w: P, h: P, rows: k, cols: k, unit: f, label: (r, c) => colourTip(b, f, r, c) });
        for (let c = 0; c < 3; c++) {
          const kx = x + P + 10 + c * (kw + gap);
          const ky = py + (P - kw) / 2;
          matrix(ctx, b.W, (f * C + c) * kk, k, k, kx, ky, cell, cell, mm, max);
          label(ctx, CH[c], kx + kw / 2, py + P + 8, 'center');
          regions.push({ x: kx, y: ky, w: kw, h: kw, rows: k, cols: k, unit: f, label: (r, cc) => `W[filter ${f + 1}, ${CHANNELS[c]}, ${r}, ${cc}] = ${wv(b.W[(f * C + c) * kk + r * k + cc])}\nbias ${wv(b.b[f])}` });
        }
        if (f === unit) ring(ctx, x - 4, py - 4, cardW + 8, P + 22);
      }
      describeCanvas(
        'conv-colour',
        `${name} weights: ${F} filters, each shown as a ${k}×${k} colour patch (its ${k}×${k}×3 kernel, mid grey is zero) and as red, green and blue ${modeName}`,
      );
      return;
    }
    if (C === 1) {
      // First conv layer on grey images: one kernel per filter, laid out as a grid.
      const cell = Math.max(6, Math.min(16, Math.floor(72 / k)));
      const kw = k * cell;
      const gap = 16;
      const perRow = Math.max(1, Math.min(F, Math.floor((availW + gap) / (kw + gap))));
      const rows = Math.ceil(F / perRow);
      const H = rows * (kw + 26) + 8;
      const ctx = fitCanvas(canvas, availW, H);
      for (let f = 0; f < F; f++) {
        const x = (f % perRow) * (kw + gap);
        const y = Math.floor(f / perRow) * (kw + 26);
        label(ctx, `f${f + 1}`, x, y + 8, 'left', f === unit ? p.accent : undefined);
        matrix(ctx, b.W, f * kk, k, k, x, y + 16, cell, cell, mm, max);
        if (f === unit) ring(ctx, x - 2, y + 14, kw + 4, kw + 4);
        regions.push({ x, y: y + 16, w: kw, h: kw, rows: k, cols: k, unit: f, label: (r, c) => `W[filter ${f + 1}, ${r}, ${c}] = ${wv(b.W[f * kk + r * k + c])}\nbias ${wv(b.b[f])}` });
      }
      describeCanvas('conv', `${name} weights: ${F} filters of ${k}×${k} as ${modeName}`);
      return;
    }
    // Filters × input channels grid of kernels.
    const left = 34;
    const topH = 20;
    const gap = 5;
    const cell = Math.max(2, Math.min(12, Math.floor((availW - left - 50 - (C - 1) * gap) / (C * k))));
    const kw = k * cell;
    const W = Math.max(availW, left + C * (kw + gap) + 50);
    const H = topH + F * (kw + gap) + 4;
    const ctx = fitCanvas(canvas, W, H);
    for (let c = 0; c < C; c++) label(ctx, `c${c + 1}`, left + c * (kw + gap) + kw / 2, 8, 'center');
    label(ctx, 'bias', left + C * (kw + gap) + 14, 8, 'center');
    const bmax = maxAbs(b.b) || 1;
    for (let f = 0; f < F; f++) {
      const y = topH + f * (kw + gap);
      label(ctx, `f${f + 1}`, 2, y + kw / 2, 'left', f === unit ? p.accent : undefined);
      for (let c = 0; c < C; c++) {
        const x = left + c * (kw + gap);
        matrix(ctx, b.W, (f * C + c) * kk, k, k, x, y, cell, cell, mm, max);
        regions.push({ x, y, w: kw, h: kw, rows: k, cols: k, unit: f, label: (r, cc) => `W[filter ${f + 1}, ch ${c + 1}, ${r}, ${cc}] = ${wv(b.W[(f * C + c) * kk + r * k + cc])}` });
      }
      matrix(ctx, b.b, f, 1, 1, left + C * (kw + gap) + 14 - kw / 2, y, kw, kw, mm, bmax);
      if (f === unit) ring(ctx, left - 3, y - 3, C * (kw + gap) + 2, kw + 6);
    }
    describeCanvas('conv-grid', `${name} weights: ${F} filters by ${C} input channels, ${k}×${k} kernels as ${modeName}, with each filter’s bias`);
  }

  /** Tooltip for one pixel of a filter's colour patch: the three channel weights there. */
  const colourTip = (b: ConvBlock, f: number, r: number, c: number) => {
    const kk = b.k * b.k;
    const at = (ch: number) => wv(b.W[(f * 3 + ch) * kk + r * b.k + c]);
    return `Filter ${f + 1} at (${r}, ${c})\nR ${at(0)} · G ${at(1)} · B ${at(2)}`;
  };

  function renderDense(b: DenseBlock, i: number, mode: Exclude<WeightMode, 'hist' | 'qq'>, max: number, unit: number, availW: number) {
    const p = palette();
    const isOut = isOutput(i);
    const M = b.spec.units;
    const N = b.inSize;
    const spatial = b.inShape.h > 1;
    const name = layerTitle(i);
    const colour = colourIn(i);
    const ul = (j: number) => unitLabel(i, j);
    const who = isOut ? (namesAreGlyphs(store.info) ? 'digit' : 'class') : 'unit';
    if (spatial) {
      const { c: C, h: Hh, w: Ww } = b.inShape;
      const HW = Hh * Ww;
      const chName = (c: number) => (colour ? CHANNELS[c] : `ch ${c + 1}`);
      if (mode === 'numbers' || mode === 'hinton') {
        // One unit, every incoming weight at its pixel position.
        const cw = mode === 'numbers' ? 36 : Math.max(6, Math.min(14, Math.floor((availW - 20) / Math.min(C * Ww, 56))));
        const ch = mode === 'numbers' ? 18 : cw;
        const tileW = Ww * cw;
        const gap = 18;
        const perRow = Math.max(1, Math.floor((Math.max(availW, tileW) + gap) / (tileW + gap)));
        const rows = Math.ceil(C / perRow);
        const W = Math.max(availW, Math.min(C, perRow) * (tileW + gap));
        const H = 26 + rows * (Hh * ch + 28) + 6;
        const ctx = fitCanvas(canvas, W, H);
        label(ctx, `${unitName(i, unit)}: weights from each input position · bias ${wv(b.b[unit])}`, 0, 10, 'left', p.ink);
        for (let c = 0; c < C; c++) {
          const x = (c % perRow) * (tileW + gap);
          const y = 26 + Math.floor(c / perRow) * (Hh * ch + 28);
          if (C > 1) label(ctx, colour ? `from the ${CHANNELS[c]} channel` : `from channel ${c + 1}`, x, y + 6);
          matrix(ctx, b.W, unit * N + c * HW, Hh, Ww, x, y + 16, cw, ch, mode, max);
          regions.push({ x, y: y + 16, w: tileW, h: Hh * ch, rows: Hh, cols: Ww, unit, label: (r, cc) => `W[${ul(unit)}, ${chName(c)}, ${r}, ${cc}] = ${wv(b.W[unit * N + c * HW + r * Ww + cc])}` });
        }
        describeCanvas('dense-unit', `${unitName(i, unit)} of ${name}: its ${int(N)} incoming weights at their input positions${C > 1 ? `, one ${Hh}×${Ww} map per ${colour ? 'colour ' : ''}channel` : ''}, as ${mode === 'numbers' ? 'numbers' : 'a Hinton diagram'}`);
        return;
      }
      if (colour) {
        // Templates in colour: every unit's 32×32×3 weights as one image, scaled per unit.
        const names = isOut ? Array.from({ length: M }, (_, j) => ul(j)) : [];
        const s = Math.max(2, Math.min(3, Math.floor((availW + 12) / Math.min(M, 8) / (Ww + 6))));
        const tile = Ww * s;
        const gap = 14;
        const perRow = Math.max(1, Math.floor((availW + gap) / (tile + gap)));
        const rows = Math.ceil(M / perRow);
        const H = rows * (tile + 22) + 4;
        const ctx = fitCanvas(canvas, availW, H);
        const short = isOut && measure(names) > tile ? shortNames(store.info) : null;
        for (let j = 0; j < M; j++) {
          const x = 2 + (j % perRow) * (tile + gap);
          const y = Math.floor(j / perRow) * (tile + 22);
          label(ctx, short ? short[j] : ul(j), x, y + 7, 'left', j === unit ? p.accent : undefined);
          drawColourWeights(ctx, b.W, j * N, HW, Hh, Ww, x, y + 14, tile, tile);
          if (j === unit) ring(ctx, x - 2, y + 12, tile + 4, tile + 4);
          regions.push({
            x,
            y: y + 14,
            w: tile,
            h: tile,
            rows: Hh,
            cols: Ww,
            unit: j,
            label: (r, c) => `${unitName(i, j)} at (${r}, ${c})\nR ${wv(b.W[j * N + r * Ww + c])} · G ${wv(b.W[j * N + HW + r * Ww + c])} · B ${wv(b.W[j * N + 2 * HW + r * Ww + c])}`,
          });
        }
        describeCanvas('dense-templates-colour', `${name} weights: each of the ${M} ${who === 'unit' ? 'units' : 'classes'}’ ${int(N)} weights as a ${Hh}×${Ww} colour image, mid grey is zero`);
        return;
      }
      // Heat map: every unit's weights reshaped to the input's layout ("templates").
      if (C === 1) {
        const short = isOut && !namesAreGlyphs(store.info) ? shortNames(store.info) : null;
        const s = Math.max(2, Math.min(4, Math.floor((availW + 10) / Math.min(M, 10) / (Ww + 4))));
        const tile = Ww * s;
        const gap = 12;
        const perRow = Math.max(1, Math.floor((availW + gap) / (tile + gap)));
        const rows = Math.ceil(M / perRow);
        const H = rows * (tile + 22) + 4;
        const ctx = fitCanvas(canvas, availW, H);
        const fits = !short || measure(Array.from({ length: M }, (_, j) => ul(j))) <= tile + gap - 2;
        for (let j = 0; j < M; j++) {
          const x = (j % perRow) * (tile + gap);
          const y = Math.floor(j / perRow) * (tile + 22);
          label(ctx, fits ? ul(j) : short![j], x, y + 7, 'left', j === unit ? p.accent : undefined);
          matrix(ctx, b.W, j * N, Hh, Ww, x, y + 14, s, s, 'heat', max);
          if (j === unit) ring(ctx, x - 2, y + 12, tile + 4, tile + 4);
          regions.push({ x, y: y + 14, w: tile, h: tile, rows: Hh, cols: Ww, unit: j, label: (r, c) => `W[${ul(j)}, ${r}, ${c}] = ${wv(b.W[j * N + r * Ww + c])}` });
        }
        describeCanvas('dense-templates', `${name} weights: each of the ${M} ${who === 'unit' ? 'units' : isOut && who === 'digit' ? 'digits' : 'classes'}’ ${int(N)} weights as a ${Hh}×${Ww} heatmap`);
        return;
      }
      const rowNames = Array.from({ length: M }, (_, j) => ul(j));
      const left = Math.max(34, measure(rowNames) + 8);
      const gap = 4;
      const s = Math.max(1, Math.min(4, Math.floor((availW - left - (C - 1) * gap) / (C * Ww))));
      const tile = Ww * s;
      const W = Math.max(availW, left + C * (tile + gap));
      const H = 18 + M * (Hh * s + gap) + 4;
      const ctx = fitCanvas(canvas, W, H);
      for (let c = 0; c < C; c++) label(ctx, `c${c + 1}`, left + c * (tile + gap) + tile / 2, 8, 'center');
      for (let j = 0; j < M; j++) {
        const y = 18 + j * (Hh * s + gap);
        label(ctx, rowNames[j], 2, y + (Hh * s) / 2, 'left', j === unit ? p.accent : undefined);
        for (let c = 0; c < C; c++) {
          const x = left + c * (tile + gap);
          matrix(ctx, b.W, j * N + c * HW, Hh, Ww, x, y, s, s, 'heat', max);
          regions.push({ x, y, w: tile, h: Hh * s, rows: Hh, cols: Ww, unit: j, label: (r, cc) => `W[${ul(j)}, ch ${c + 1}, ${r}, ${cc}] = ${wv(b.W[j * N + c * HW + r * Ww + cc])}` });
        }
        if (j === unit) ring(ctx, left - 3, y - 2, C * (tile + gap) + 2, Hh * s + 4);
      }
      describeCanvas('dense-channels', `${name} weights: one row per ${who}, one ${Hh}×${Ww} heatmap per input channel`);
      return;
    }

    // Plain matrix: rows = units of this layer, columns = its inputs (features or the previous
    // layer's units), plus the bias.
    const numbers = mode === 'numbers';
    const cols = inputLabels(i);
    if (numbers && M * N > 4096) {
      const cw = 44;
      const ch = 22;
      const perRow = Math.max(1, Math.floor(availW / cw));
      const rows = Math.ceil(N / perRow);
      const ctx = fitCanvas(canvas, availW, 26 + rows * ch + 4);
      label(ctx, `${unitName(i, unit)}: ${N} incoming weights · bias ${wv(b.b[unit])}`, 0, 10, 'left', p.ink);
      matrix(ctx, b.W, unit * N, rows, perRow, 0, 26, cw, ch, 'numbers', max);
      describeCanvas('dense-unit', `${unitName(i, unit)} of ${name}: its ${N} incoming weights as numbers`);
      return;
    }
    const rowNames = Array.from({ length: M }, (_, j) => ul(j));
    const left = Math.max(30, measure(rowNames) + 10);
    const top = 18;
    const colW = cols ? measure(cols, `500 12px ${MONO}`) + 8 : 0;
    // Small matrices (point networks) get roomy cells so every column can carry its label.
    const roomy = N <= 12;
    let cw = numbers ? 44 : Math.max(3, Math.min(roomy ? 44 : 22, Math.floor((availW - left - 40) / N)));
    if (cols && roomy && !numbers) cw = Math.max(cw, Math.min(56, colW));
    const ch = numbers ? 22 : Math.max(Math.min(cw, roomy ? 26 : 22), Math.min(22, cw));
    const W = Math.max(availW, left + (N + 1) * cw + 20);
    const H = top + M * ch + 6;
    const ctx = fitCanvas(canvas, W, H);
    const labelEvery = cols && cw >= colW ? 1 : Math.max(1, Math.ceil(16 / cw));
    for (let c = 0; c < N; c += labelEvery) label(ctx, cols ? cols[c] : String(c + 1), left + c * cw + cw / 2, 8, 'center');
    if (cols && i === 0 && store.info.kind === 'points') {
      // Feature labels (x₁², sin x₂) carry sub- and superscripts: redraw them larger.
      ctx.clearRect(left, 0, N * cw, top - 2);
      ctx.font = `500 12px ${MONO}`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = p.ink2;
      for (let c = 0; c < N; c++) ctx.fillText(cols[c], left + c * cw + cw / 2, 8);
    }
    label(ctx, 'b', left + N * cw + 8 + cw / 2, 8, 'center');
    for (let j = 0; j < M; j++) label(ctx, rowNames[j], 2, top + j * ch + ch / 2, 'left', j === unit ? p.accent : undefined);
    matrix(ctx, b.W, 0, M, N, left, top, cw, ch, mode, max);
    matrix(ctx, b.b, 0, M, 1, left + N * cw + 8, top, cw, ch, mode, maxAbs(b.b) || 1);
    const inName = (c: number) => (cols ? cols[c] : `in ${c + 1}`);
    regions.push({ x: left, y: top, w: N * cw, h: M * ch, rows: M, cols: N, label: (r, c) => `W[${rowNames[r]} ← ${inName(c)}] = ${wv(b.W[r * N + c])}` });
    regions.push({ x: left + N * cw + 8, y: top, w: cw, h: M * ch, rows: M, cols: 1, label: (r) => `bias of ${rowNames[r]} = ${wv(b.b[r])}` });
    ring(ctx, left - 1, top + unit * ch - 1, N * cw + 2, ch + 2);
    const what = i === 0 && store.info.kind === 'points' && cols ? `the input features (${cols.join(', ')})` : `${N} inputs`;
    describeCanvas(
      'dense-matrix',
      `${name} weights as ${mode === 'heat' ? 'a heatmap' : mode === 'hinton' ? 'a Hinton diagram' : 'numbers'}: ${M} rows, one per ${who}${isOut && !namesAreGlyphs(store.info) ? ` (${rowNames.join(', ')})` : ''}, by ${what}, plus a bias column`,
    );
  }

  /**
   * Q–Q plot of the layer's weights now and at initialisation, both against normal quantiles, with
   * the normal line through the current weights' quartiles. Legend and a small stats table sit
   * beside the plot when there is room, below it otherwise.
   */
  function renderQQMode(w: Float32Array, init: Float32Array | null, availW: number) {
    const p = palette();
    const now = summarize(w, 400);
    const start = init && init.length ? summarizeFrozen(init, 400) : null; // initial weights never change
    const qqH = 300;
    const sideW = 280;
    // Beside the plot when the plot can stay at least 420 px wide; stacked below it otherwise.
    const beside = availW >= 420 + sideW + 24;
    const plotW = beside ? Math.min(560, availW - sideW - 24) : Math.min(availW, 560);
    const sideX = beside ? plotW + 24 : 0;
    const ref = now.qq.reference;
    const ITEMS: { text: string; mark: 'filled' | 'hollow' | 'line' }[] = [
      { text: 'Now', mark: 'filled' },
      ...(start ? [{ text: 'At initialisation', mark: 'hollow' as const }] : []),
      // Through the quartiles; with the mean and std when the quartiles coincide (that line would be flat).
      ...(ref ? [{ text: ref.from === 'quartiles' ? 'Normal line through the quartiles of now' : 'Normal line with the mean and std of now', mark: 'line' as const }] : []),
    ];

    // Legend entries, wrapped to the available width (measured before the canvas is sized).
    const ctx0 = canvas.getContext('2d')!;
    ctx0.font = `500 12px ${SANS_FONT}`;
    const legendW = beside ? sideW : availW;
    const placed: { x: number; y: number; item: (typeof ITEMS)[number] }[] = [];
    let lx = 0;
    let ly = 0;
    for (const item of ITEMS) {
      const wItem = (item.mark === 'line' ? 22 : 14) + ctx0.measureText(item.text).width + 22;
      if (beside) {
        placed.push({ x: 0, y: ly, item });
        ly += 20;
        continue;
      }
      if (lx > 0 && lx + wItem > legendW) {
        lx = 0;
        ly += 20;
      }
      placed.push({ x: lx, y: ly, item });
      lx += wItem;
    }
    const legendH = ly + (beside ? 0 : 20);
    const tableH = 5 * 18 + 6;
    const plotY = beside ? 8 : legendH + 8;
    const tableY = beside ? legendH + 18 : plotY + qqH + 16;
    const H = Math.max(plotY + qqH, tableY + tableH) + 4;
    const ctx = fitCanvas(canvas, availW, H);

    const legendX = beside ? sideX : 0;
    const legendY = beside ? 6 : 0;
    ctx.font = `500 12px ${SANS_FONT}`;
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'left';
    for (const { x, y, item } of placed) {
      const cx = legendX + x;
      const cy = legendY + y + 9;
      if (item.mark === 'filled') {
        ctx.fillStyle = p.ink;
        ctx.fillRect(cx, cy - 3.5, 7, 7);
      } else if (item.mark === 'hollow') {
        ctx.strokeStyle = p.muted;
        ctx.lineWidth = 1;
        ctx.strokeRect(cx + 0.5, cy - 3, 6, 6);
      } else {
        ctx.strokeStyle = p.accent;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([6, 4]);
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(cx + 16, cy);
        ctx.stroke();
        ctx.setLineDash([]);
      }
      ctx.fillStyle = p.ink2;
      ctx.fillText(item.text, cx + (item.mark === 'line' ? 22 : 14), cy);
    }

    // drawQQ paints hollow series first, so "now" sits on top; the tooltip lists it first too.
    const series: QQSeries[] = [{ x: now.qq.theoretical, y: now.qq.sample, probs: now.qq.probs, style: 'filled', name: 'now' }];
    if (start) series.push({ x: start.qq.theoretical, y: start.qq.sample, probs: start.qq.probs, style: 'hollow', name: 'at init' });
    const plot = drawQQ(ctx, { x: 0, y: plotY, w: plotW, h: qqH }, series, {
      xTitle: 'Normal quantile',
      yTitle: 'Weight quantile',
      xName: 'normal',
      line: ref?.line ?? null,
    });

    // Stats table: one column per series, values right-aligned under their headings.
    const tx = beside ? sideX : 0;
    const nowX = tx + 150;
    const initX = tx + 280;
    const row = (k: number) => tableY + 9 + k * 18;
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'right';
    ctx.font = `600 10px ${SANS_FONT}`;
    ctx.fillStyle = p.muted;
    ctx.fillText('NOW', nowX, row(0));
    if (start) ctx.fillText('AT INITIALISATION', initX, row(0));
    ctx.fillStyle = p.hair;
    ctx.fillRect(tx, row(0) + 9, (start ? initX : nowX) - tx, 1);
    // Fixed decimals (never "−0.00") and three significant digits with trailing zeros, so the two
    // columns line up digit for digit.
    const rows: [string, (s: typeof now) => string][] = [
      ['Skew', (s) => fixed(s.moments.skew, 2)],
      ['Excess kurtosis', (s) => fixed(s.moments.excessKurtosis, 2)],
      ['PPCC r', (s) => fixed(s.ppcc, 4)],
      ['Std', (s) => sig(s.moments.std, 3)],
    ];
    rows.forEach(([nm, get], k) => {
      const y = row(k + 1);
      ctx.font = `500 12px ${SANS_FONT}`;
      ctx.textAlign = 'left';
      ctx.fillStyle = p.ink2;
      ctx.fillText(nm, tx, y);
      ctx.font = `500 12px ${MONO}`;
      ctx.textAlign = 'right';
      ctx.fillStyle = p.ink;
      ctx.fillText(get(now), nowX, y);
      if (start) ctx.fillText(get(start), initX, y);
    });

    regions.push({
      x: plot.plot.x,
      y: plot.plot.y,
      w: plot.plot.w,
      h: plot.plot.h,
      rows: 1,
      cols: Math.max(1, Math.round(plot.plot.w)),
      label: (_r, c) => plot.column(plot.plot.x + c + 0.5) ?? '',
    });
  }

  function renderHist(w: Float32Array, init: Float32Array | null, availW: number) {
    const p = palette();
    const H = 240;
    const ctx = fitCanvas(canvas, availW, H);
    const m = Math.max(maxAbs(w), init ? maxAbs(init) : 0) || 1;
    const bins = 41;
    const count = (a: Float32Array) => {
      const c = new Array(bins).fill(0);
      for (const v of a) c[Math.min(bins - 1, Math.max(0, Math.floor(((v + m) / (2 * m)) * bins)))]++;
      return c;
    };
    const cur = count(w);
    const ini = init ? count(init) : null;
    const top = Math.max(...cur, ...(ini ?? [0]));
    const L = 44;
    const B = 26;
    const pw = availW - L - 10;
    const ph = H - B - 10;
    const bw = pw / bins;
    ctx.font = `400 10px ${MONO}`;
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'right';
    for (let t = 0; t <= 4; t++) {
      const y = 10 + ph - (ph * t) / 4;
      ctx.fillStyle = p.hair;
      ctx.fillRect(L, Math.round(y), pw, 1);
      ctx.fillStyle = p.muted;
      ctx.fillText(int((top * t) / 4), L - 6, y);
    }
    for (let k = 0; k < bins; k++) {
      const v = cur[k] / top;
      const centre = -m + ((k + 0.5) / bins) * 2 * m;
      ctx.fillStyle = centre >= 0 ? p.accent : p.neg;
      ctx.fillRect(L + k * bw + 1, 10 + ph * (1 - v), bw - 2, ph * v);
    }
    if (ini) {
      ctx.strokeStyle = p.ink;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      for (let k = 0; k < bins; k++) {
        const y = 10 + ph * (1 - ini[k] / top);
        if (k === 0) ctx.moveTo(L, y);
        else ctx.lineTo(L + k * bw, y);
        ctx.lineTo(L + (k + 1) * bw, y);
      }
      ctx.stroke();
    }
    ctx.fillStyle = p.ink;
    ctx.fillRect(L, 10 + ph, pw, 1.5);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillStyle = p.muted;
    const d = m >= 1 ? 2 : m >= 0.1 ? 2 : 3;
    for (const t of [-1, -0.5, 0, 0.5, 1]) ctx.fillText(fixed(t * m, d), L + ((t + 1) / 2) * pw, 10 + ph + 6);
    regions.push({
      x: L,
      y: 10,
      w: pw,
      h: ph,
      rows: 1,
      cols: bins,
      label: (_r, c) => {
        const lo = -m + (c / bins) * 2 * m;
        const hi = lo + (2 * m) / bins;
        return `${fixed(lo, 3)} … ${fixed(hi, 3)}\nnow ${int(cur[c])} · at start ${ini ? int(ini[c]) : '–'}`;
      },
    });
  }

  const regionAt = (e: MouseEvent) => {
    const r = canvas.getBoundingClientRect();
    const x = e.clientX - r.left;
    const y = e.clientY - r.top;
    for (const g of regions) {
      if (x >= g.x && x < g.x + g.w && y >= g.y && y < g.y + g.h) {
        const row = Math.min(g.rows - 1, Math.floor(((y - g.y) / g.h) * g.rows));
        const col = Math.min(g.cols - 1, Math.floor(((x - g.x) / g.w) * g.cols));
        return { g, row, col };
      }
    }
    return null;
  };
  canvas.addEventListener('mousemove', (e) => {
    const hit = regionAt(e);
    if (hit) showTip(hit.g.label(hit.row, hit.col), e.clientX, e.clientY);
    else hideTip();
  });
  canvas.addEventListener('mouseleave', hideTip);
  canvas.addEventListener('click', (e) => {
    const hit = regionAt(e);
    if (!hit) return;
    const unit = hit.g.unit ?? (hit.g.rows > 1 && store.mode !== 'hist' ? hit.row : undefined);
    if (unit !== undefined) select(store.selected, unit);
  });

  // Redraws: on the next frame, never for an off-screen section (it catches up on the way in).
  let inView = true;
  let behind = false;
  let queued = false;
  let qqCost = 0;
  let lastDraw = -Infinity;
  const draw = () => {
    const t0 = performance.now();
    render();
    lastDraw = performance.now();
    if (store.mode === 'qq') qqCost = lastDraw - t0;
  };
  const schedule = () => {
    if (!inView) {
      behind = true;
      return;
    }
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => {
      queued = false;
      if (inView) draw();
      else behind = true;
    });
  };
  /**
   * New weights: at most ~10 redraws a second. The Q–Q view sorts the whole layer (400k weights
   * for a wide dense layer), so it waits at least 250 ms and twice its own cost between redraws.
   */
  let timer: ReturnType<typeof setTimeout> | null = null;
  const onWeights = () => {
    if (!inView) {
      behind = true;
      return;
    }
    if (timer) return;
    const gap = store.mode === 'qq' ? Math.max(250, 2 * qqCost) : 100;
    const wait = gap - (performance.now() - lastDraw);
    if (wait <= 0) schedule();
    else
      timer = setTimeout(() => {
        timer = null;
        schedule();
      }, wait);
  };
  new IntersectionObserver(
    (entries) => {
      inView = entries.some((e) => e.isIntersecting);
      // Drawn at once (not on the next frame), so the section never appears with stale weights.
      if (inView && behind) {
        behind = false;
        draw();
      }
    },
    { rootMargin: '200px 0px' },
  ).observe(box);
  const full = () => {
    renderHead();
    schedule();
  };
  store.on('model', full);
  store.on('select', full);
  store.on('mode', full);
  store.on('frozen', full);
  store.on('dataset', full);
  store.on('weights', onWeights);
  onThemeChange(schedule);
  new ResizeObserver(schedule).observe(box);
  full();
}
