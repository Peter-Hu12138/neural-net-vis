import { initialWeights, select, setMode } from '../actions';
import { summarize, summarizeFrozen } from '../analysis/stats';
import type { ConvBlock, DenseBlock } from '../nn/network';
import { store, type WeightMode } from '../store';
import { layerDetail, layerName } from './builder';
import { $, clear, h, segmented, selectField } from './dom';
import { drawMatrix, fitCanvas, maxAbs, type MatrixMode } from './draw';
import { drawQQ, num, SANS as SANS_FONT, type QQSeries } from './qq';
import { onThemeChange, palette } from './theme';
import { hideTip, showTip } from './tip';

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

export function mountInspector(): void {
  const root = $('inspector');
  const head = h('div', { class: 'insp-head' });
  const note = h('p', { class: 'hint', style: { marginBottom: '12px' } });
  const canvas = h('canvas', { role: 'img', 'aria-label': 'Weights of the selected layer' }) as HTMLCanvasElement;
  const box = h('div', { class: 'insp-canvas' }, canvas);
  const statsEl = h('div', { class: 'insp-stats' });
  root.append(head, note, box, statsEl);

  let regions: Region[] = [];

  const unitCount = (i: number) => {
    const b = store.net.blocks[i];
    return b.kind === 'conv' ? b.spec.filters : b.spec.units;
  };
  const unitName = (i: number, u: number) => {
    const b = store.net.blocks[i];
    if (i === store.net.blocks.length - 1) return `Digit ${u}`;
    return b.kind === 'conv' ? `Filter ${u + 1}` : `Unit ${u + 1}`;
  };

  const renderHead = () => {
    clear(head);
    const blocks = store.net.blocks;
    const sel = Math.min(store.selected, blocks.length - 1);
    head.append(
      selectField(
        'insp-layer',
        'Layer',
        blocks.map((_, i) => {
          const spec = i === blocks.length - 1 ? null : store.spec[i];
          return { value: i, label: `${layerName(spec, i)} · ${layerDetail(spec)}` };
        }),
        sel,
        (v) => select(v, null),
      ),
      h('div', { class: 'field' }, h('span', { class: 'label' }, 'View'), segmented(MODES, store.mode, setMode, 'Weight view')),
    );
    const n = unitCount(sel);
    const unit = store.selectedUnit ?? 0;
    head.append(
      selectField(
        'insp-unit',
        sel === blocks.length - 1 ? 'Digit' : blocks[sel].kind === 'conv' ? 'Filter' : 'Unit',
        Array.from({ length: n }, (_, u) => ({ value: u, label: unitName(sel, u) })),
        Math.min(unit, n - 1),
        (u) => select(sel, u),
      ),
    );
    note.textContent = NOTES[store.mode];
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

    if (mode === 'hist') renderHist(b.W, initialWeights[2 * i] ?? null, availW);
    else if (mode === 'qq') renderQQMode(b.W, initialWeights[2 * i] ?? null, availW);
    else if (b.kind === 'conv') renderConv(b, mode, max, unit, availW);
    else renderDense(b, i === blocks.length - 1, mode, max, unit, availW);

    const s = stats(b.W);
    const sb = stats(b.b);
    const shape = b.kind === 'conv' ? `${b.spec.filters}×${b.inShape.c}×${b.k}×${b.k}` : `${b.spec.units}×${b.inSize}`;
    clear(statsEl);
    const kv = (k: string, v: string) => h('span', null, `${k} `, h('b', null, v));
    statsEl.append(
      kv('W', shape),
      kv('mean', s.mean.toFixed(4)),
      kv('std', s.std.toFixed(4)),
      kv('min', s.mn.toFixed(3)),
      kv('max', s.mx.toFixed(3)),
      kv('‖W‖', s.norm.toFixed(2)),
      kv('bias mean', sb.mean.toFixed(4)),
      kv('step', store.weightsStep.toLocaleString('en-US')),
    );
  };

  const label = (ctx: CanvasRenderingContext2D, text: string, x: number, y: number, align: CanvasTextAlign = 'left', color?: string) => {
    ctx.font = `500 10px ${MONO}`;
    ctx.textAlign = align;
    ctx.textBaseline = 'middle';
    ctx.fillStyle = color ?? palette().muted;
    ctx.fillText(text, x, y);
  };

  function renderConv(b: ConvBlock, mode: Exclude<WeightMode, 'hist' | 'qq'>, max: number, unit: number, availW: number) {
    const p = palette();
    const F = b.spec.filters;
    const C = b.inShape.c;
    const k = b.k;
    const kk = k * k;
    if (mode === 'numbers') {
      const cw = 42;
      const ch = 22;
      const kw = k * cw;
      const gap = 18;
      const perRow = Math.max(1, Math.floor((availW - 10 + gap) / (kw + gap)));
      const rows = Math.ceil(C / perRow);
      const W = Math.max(availW, perRow * (kw + gap));
      const H = 30 + rows * (k * ch + 34) + 30;
      const ctx = fitCanvas(canvas, W, H);
      label(ctx, `Filter ${unit + 1}: one ${k}×${k} kernel per input channel`, 0, 12, 'left', p.ink);
      for (let c = 0; c < C; c++) {
        const x = (c % perRow) * (kw + gap);
        const y = 30 + Math.floor(c / perRow) * (k * ch + 34);
        label(ctx, `in-channel ${c + 1}`, x, y + 8);
        drawMatrix(ctx, b.W, (unit * C + c) * kk, k, k, x, y + 18, cw, ch, 'numbers', max);
        regions.push({ x, y: y + 18, w: kw, h: k * ch, rows: k, cols: k, unit, label: (r, cc) => `W[filter ${unit + 1}, ch ${c + 1}, ${r}, ${cc}] = ${b.W[(unit * C + c) * kk + r * k + cc].toFixed(5)}` });
      }
      label(ctx, `bias b = ${b.b[unit].toFixed(5)}`, 0, H - 12, 'left', p.ink);
      return;
    }
    const mm: MatrixMode = mode;
    if (C === 1) {
      // First conv layer: one kernel per filter, laid out as a grid.
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
        drawMatrix(ctx, b.W, f * kk, k, k, x, y + 16, cell, cell, mm, max);
        if (f === unit) {
          ctx.strokeStyle = p.accent;
          ctx.lineWidth = 2;
          ctx.strokeRect(x - 2, y + 14, kw + 4, kw + 4);
        }
        regions.push({ x, y: y + 16, w: kw, h: kw, rows: k, cols: k, unit: f, label: (r, c) => `W[filter ${f + 1}, ${r}, ${c}] = ${b.W[f * kk + r * k + c].toFixed(5)}\nbias ${b.b[f].toFixed(4)}` });
      }
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
        drawMatrix(ctx, b.W, (f * C + c) * kk, k, k, x, y, cell, cell, mm, max);
        regions.push({ x, y, w: kw, h: kw, rows: k, cols: k, unit: f, label: (r, cc) => `W[filter ${f + 1}, ch ${c + 1}, ${r}, ${cc}] = ${b.W[(f * C + c) * kk + r * k + cc].toFixed(5)}` });
      }
      drawMatrix(ctx, b.b, f, 1, 1, left + C * (kw + gap) + 14 - kw / 2, y, kw, kw, mm, bmax);
      if (f === unit) {
        ctx.strokeStyle = p.accent;
        ctx.lineWidth = 2;
        ctx.strokeRect(left - 3, y - 3, C * (kw + gap) + 2, kw + 6);
      }
    }
  }

  function renderDense(b: DenseBlock, isOut: boolean, mode: Exclude<WeightMode, 'hist' | 'qq'>, max: number, unit: number, availW: number) {
    const p = palette();
    const M = b.spec.units;
    const N = b.inSize;
    const spatial = b.inShape.h > 1;
    const unitLabel = (j: number) => (isOut ? `${j}` : `u${j + 1}`);
    if (spatial) {
      const { c: C, h: Hh, w: Ww } = b.inShape;
      const HW = Hh * Ww;
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
        label(ctx, `${isOut ? 'Digit' : 'Unit'} ${isOut ? unit : unit + 1}: weights from each input position · bias ${b.b[unit].toFixed(4)}`, 0, 10, 'left', p.ink);
        for (let c = 0; c < C; c++) {
          const x = (c % perRow) * (tileW + gap);
          const y = 26 + Math.floor(c / perRow) * (Hh * ch + 28);
          if (C > 1) label(ctx, `from channel ${c + 1}`, x, y + 6);
          drawMatrix(ctx, b.W, unit * N + c * HW, Hh, Ww, x, y + 16, cw, ch, mode, max);
          regions.push({ x, y: y + 16, w: tileW, h: Hh * ch, rows: Hh, cols: Ww, unit, label: (r, cc) => `W[${unitLabel(unit)}, ch ${c + 1}, ${r}, ${cc}] = ${b.W[unit * N + c * HW + r * Ww + cc].toFixed(5)}` });
        }
        return;
      }
      // Heat map: every unit's weights reshaped to the input's layout ("templates").
      if (C === 1) {
        const s = Math.max(2, Math.min(4, Math.floor((availW + 10) / Math.min(M, 10) / (Ww + 4))));
        const tile = Ww * s;
        const gap = 12;
        const perRow = Math.max(1, Math.floor((availW + gap) / (tile + gap)));
        const rows = Math.ceil(M / perRow);
        const H = rows * (tile + 22) + 4;
        const ctx = fitCanvas(canvas, availW, H);
        for (let j = 0; j < M; j++) {
          const x = (j % perRow) * (tile + gap);
          const y = Math.floor(j / perRow) * (tile + 22);
          label(ctx, unitLabel(j), x, y + 7, 'left', j === unit ? p.accent : undefined);
          drawMatrix(ctx, b.W, j * N, Hh, Ww, x, y + 14, s, s, 'heat', max);
          if (j === unit) {
            ctx.strokeStyle = p.accent;
            ctx.lineWidth = 2;
            ctx.strokeRect(x - 2, y + 12, tile + 4, tile + 4);
          }
          regions.push({ x, y: y + 14, w: tile, h: tile, rows: Hh, cols: Ww, unit: j, label: (r, c) => `W[${unitLabel(j)}, ${r}, ${c}] = ${b.W[j * N + r * Ww + c].toFixed(5)}` });
        }
        return;
      }
      const left = 34;
      const gap = 4;
      const s = Math.max(1, Math.min(4, Math.floor((availW - left - (C - 1) * gap) / (C * Ww))));
      const tile = Ww * s;
      const W = Math.max(availW, left + C * (tile + gap));
      const H = 18 + M * (Hh * s + gap) + 4;
      const ctx = fitCanvas(canvas, W, H);
      for (let c = 0; c < C; c++) label(ctx, `c${c + 1}`, left + c * (tile + gap) + tile / 2, 8, 'center');
      for (let j = 0; j < M; j++) {
        const y = 18 + j * (Hh * s + gap);
        label(ctx, unitLabel(j), 2, y + (Hh * s) / 2, 'left', j === unit ? p.accent : undefined);
        for (let c = 0; c < C; c++) {
          const x = left + c * (tile + gap);
          drawMatrix(ctx, b.W, j * N + c * HW, Hh, Ww, x, y, s, s, 'heat', max);
          regions.push({ x, y, w: tile, h: Hh * s, rows: Hh, cols: Ww, unit: j, label: (r, cc) => `W[${unitLabel(j)}, ch ${c + 1}, ${r}, ${cc}] = ${b.W[j * N + c * HW + r * Ww + cc].toFixed(5)}` });
        }
        if (j === unit) {
          ctx.strokeStyle = p.accent;
          ctx.lineWidth = 2;
          ctx.strokeRect(left - 3, y - 2, C * (tile + gap) + 2, Hh * s + 4);
        }
      }
      return;
    }

    // Plain matrix: rows = units of this layer, columns = units of the previous layer, plus bias.
    const numbers = mode === 'numbers';
    if (numbers && M * N > 4096) {
      const cw = 44;
      const ch = 22;
      const perRow = Math.max(1, Math.floor(availW / cw));
      const rows = Math.ceil(N / perRow);
      const ctx = fitCanvas(canvas, availW, 26 + rows * ch + 4);
      label(ctx, `${isOut ? 'Digit' : 'Unit'} ${isOut ? unit : unit + 1}: ${N} incoming weights · bias ${b.b[unit].toFixed(4)}`, 0, 10, 'left', p.ink);
      drawMatrix(ctx, b.W, unit * N, rows, perRow, 0, 26, cw, ch, 'numbers', max);
      return;
    }
    const left = 30;
    const top = 18;
    const cw = numbers ? 44 : Math.max(3, Math.min(22, Math.floor((availW - left - 40) / N)));
    const ch = numbers ? 22 : Math.max(cw, Math.min(22, cw));
    const W = Math.max(availW, left + (N + 1) * cw + 20);
    const H = top + M * ch + 6;
    const ctx = fitCanvas(canvas, W, H);
    const step = Math.max(1, Math.ceil(16 / cw));
    for (let c = 0; c < N; c += step) label(ctx, String(c + 1), left + c * cw + cw / 2, 8, 'center');
    label(ctx, 'b', left + N * cw + 8 + cw / 2, 8, 'center');
    for (let j = 0; j < M; j++) label(ctx, unitLabel(j), 2, top + j * ch + ch / 2, 'left', j === unit ? p.accent : undefined);
    drawMatrix(ctx, b.W, 0, M, N, left, top, cw, ch, mode, max);
    drawMatrix(ctx, b.b, 0, M, 1, left + N * cw + 8, top, cw, ch, mode, maxAbs(b.b) || 1);
    regions.push({ x: left, y: top, w: N * cw, h: M * ch, rows: M, cols: N, label: (r, c) => `W[${unitLabel(r)} ← in ${c + 1}] = ${b.W[r * N + c].toFixed(5)}` });
    ctx.strokeStyle = p.accent;
    ctx.lineWidth = 2;
    ctx.strokeRect(left - 1, top + unit * ch - 1, N * cw + 2, ch + 2);
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
    const ITEMS: { text: string; mark: 'filled' | 'hollow' | 'line' }[] = [
      { text: 'Now', mark: 'filled' },
      ...(start ? [{ text: 'At initialisation', mark: 'hollow' as const }] : []),
      { text: 'Normal line through the quartiles of now', mark: 'line' },
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
      line: now.qq.line,
    });

    // Stats table: one column per series, values right-aligned under their headings.
    const tx = beside ? sideX : 0;
    const nowX = tx + 150;
    const initX = tx + 280;
    const row = (i: number) => tableY + 9 + i * 18;
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'right';
    ctx.font = `600 10px ${SANS_FONT}`;
    ctx.fillStyle = p.muted;
    ctx.fillText('NOW', nowX, row(0));
    if (start) ctx.fillText('AT INITIALISATION', initX, row(0));
    ctx.fillStyle = p.hair;
    ctx.fillRect(tx, row(0) + 9, (start ? initX : nowX) - tx, 1);
    const f2 = (v: number) => (Number.isFinite(v) ? v.toFixed(2).replace(/^-/, '−') : '—');
    const f4 = (v: number) => (Number.isFinite(v) ? v.toFixed(4) : '—');
    const rows: [string, (s: typeof now) => string][] = [
      ['Skew', (s) => f2(s.moments.skew)],
      ['Excess kurtosis', (s) => f2(s.moments.excessKurtosis)],
      ['PPCC r', (s) => f4(s.ppcc)],
      ['Std', (s) => num(s.moments.std)],
    ];
    rows.forEach(([name, get], i) => {
      const y = row(i + 1);
      ctx.font = `500 12px ${SANS_FONT}`;
      ctx.textAlign = 'left';
      ctx.fillStyle = p.ink2;
      ctx.fillText(name, tx, y);
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
    const L = 40;
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
      ctx.fillText(String(Math.round((top * t) / 4)), L - 6, y);
    }
    for (let i = 0; i < bins; i++) {
      const v = cur[i] / top;
      const centre = -m + ((i + 0.5) / bins) * 2 * m;
      ctx.fillStyle = centre >= 0 ? p.accent : p.neg;
      ctx.fillRect(L + i * bw + 1, 10 + ph * (1 - v), bw - 2, ph * v);
    }
    if (ini) {
      ctx.strokeStyle = p.ink;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      for (let i = 0; i < bins; i++) {
        const y = 10 + ph * (1 - ini[i] / top);
        if (i === 0) ctx.moveTo(L, y);
        else ctx.lineTo(L + i * bw, y);
        ctx.lineTo(L + (i + 1) * bw, y);
      }
      ctx.stroke();
    }
    ctx.fillStyle = p.ink;
    ctx.fillRect(L, 10 + ph, pw, 1.5);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillStyle = p.muted;
    for (const t of [-1, -0.5, 0, 0.5, 1]) ctx.fillText((t * m).toFixed(2), L + ((t + 1) / 2) * pw, 10 + ph + 6);
    regions.push({ x: L, y: 10, w: pw, h: ph, rows: 1, cols: bins, label: (_r, c) => {
      const lo = -m + (c / bins) * 2 * m;
      const hi = lo + (2 * m) / bins;
      return `${lo.toFixed(3)} … ${hi.toFixed(3)}\nnow ${cur[c]} · at start ${ini ? ini[c] : '–'}`;
    } });
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

  let queued = false;
  const schedule = () => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => {
      queued = false;
      render();
    });
  };
  const full = () => {
    renderHead();
    schedule();
  };
  store.on('model', full);
  store.on('select', full);
  store.on('mode', full);
  store.on('weights', schedule);
  onThemeChange(schedule);
  new ResizeObserver(schedule).observe(box);
  full();
}
