import './networkView.css';
import { select, setProbe, testProbe } from '../actions';
import { fixed } from '../analysis/stats';
import { noun, type Data, type DatasetInfo } from '../data/datasets';
import { featureDefs, featurize } from '../data/features';
import { gridCoords, pointDomain } from '../data/grid';
import type { Block, Network } from '../nn/network';
import { store } from '../store';
import { layerName } from './builder';
import { ACTIVATIONS, type LayerSpec } from '../nn/types';
import { axisName, featureScale, robustScale } from './boundaryMath';
import { Cost, Evaluator, GridJob, Live, netFits, probeCrosses, sliceMs, type Budget } from './boundary2d';
import { onCurrentSlice, onSlice, slice, sliceAxes } from './boundary3d';
import { $, clear, h } from './dom';
import { drawMap, drawSample, fitCanvas, hasNegative, maxAbs, paintSample } from './draw';
import { drawMark } from './marks';
import { css, diverging, onThemeChange, palette, sequential, type RGB } from './theme';
import { hideTip, showTip } from './tip';

/**
 * Section 02: the network for one input. Image datasets show every feature map and unit as
 * activations for the current input; point datasets show, as in the TensorFlow Playground, what
 * every feature and unit computes over the whole input plane.
 */

interface Item {
  x: number;
  y: number;
  s: number;
  /** Height of the value label under a tile (0 = none). */
  lab?: number;
}

interface Hit {
  x: number;
  y: number;
  w: number;
  h: number;
  block: number; // -1 = input
  unit: number | null;
  text: string;
}

/** Two short caption lines for a column header; a frozen layer says so on the second. */
function headerLines(spec: LayerSpec | null, info: DatasetInfo, frozen = false): [string, string] {
  const tag = frozen ? ' · frozen' : '';
  if (!spec) return [info.id === 'mnist' ? '10 digits' : `${info.classes.length} classes`, `softmax${tag}`];
  const act = ACTIVATIONS.find((a) => a.id === spec.act)!.label;
  if (spec.kind === 'conv') return [`${spec.filters} × ${spec.kernel}×${spec.kernel}`, `${act}${spec.pool ? ' · pool' : ''}${tag}`];
  return [`${spec.units} units`, `${act}${tag}`];
}

/** Fits n square items into a box, trying 1–6 sub-columns, and returns their positions. */
function fitItems(n: number, cx: number, top: number, availW: number, availH: number, maxS: number, gap: number, minCols = 1): Item[] {
  let best = { k: 1, s: 0 };
  for (let k = minCols; k <= 6; k++) {
    const rows = Math.ceil(n / k);
    const s = Math.min(maxS, (availW - (k - 1) * gap) / k, (availH - (rows - 1) * gap) / rows);
    if (s > best.s + 0.5) best = { k, s };
  }
  const { k } = best;
  const s = Math.max(2, Math.floor(best.s));
  const rows = Math.ceil(n / k);
  const totalH = rows * s + (rows - 1) * gap;
  const totalW = k * s + (k - 1) * gap;
  const y0 = top + (availH - totalH) / 2;
  const x0 = cx - totalW / 2;
  const items: Item[] = [];
  for (let i = 0; i < n; i++) {
    const c = Math.floor(i / rows);
    const r = i % rows;
    items.push({ x: x0 + c * (s + gap), y: y0 + r * (s + gap), s });
  }
  return items;
}

/** Signed connection strengths between the units drawn in two consecutive columns, [target][source]. */
function strengths(b: Block): Float32Array[] {
  const out: Float32Array[] = [];
  if (b.kind === 'conv') {
    const C = b.inShape.c;
    const kk = b.k * b.k;
    for (let f = 0; f < b.spec.filters; f++) {
      const row = new Float32Array(C);
      for (let c = 0; c < C; c++) {
        let s = 0;
        const o = (f * C + c) * kk;
        for (let i = 0; i < kk; i++) s += b.W[o + i];
        row[c] = s;
      }
      out.push(row);
    }
  } else {
    const spatial = b.inShape.h > 1;
    const C = spatial ? b.inShape.c : b.inSize;
    const HW = spatial ? b.inShape.h * b.inShape.w : 1;
    for (let j = 0; j < b.spec.units; j++) {
      const row = new Float32Array(C);
      const o = j * b.inSize;
      for (let c = 0; c < C; c++) {
        let s = 0;
        for (let p = 0; p < HW; p++) s += b.W[o + c * HW + p];
        row[c] = s;
      }
      out.push(row);
    }
  }
  return out;
}

const SANS = 'Archivo, "Helvetica Neue", Arial, sans-serif';
const MONO = '"IBM Plex Mono", ui-monospace, monospace';

/** Sets `ctx.font` to the largest size from `size` down to `min` at which `text` fits in `maxW`. */
function fitFont(ctx: CanvasRenderingContext2D, text: string, weight: number, size: number, min: number, family: string, maxW: number): void {
  ctx.font = `${weight} ${size}px ${family}`;
  const w = ctx.measureText(text).width;
  if (w <= maxW) return;
  ctx.font = `${weight} ${Math.max(min, Math.floor((size * maxW * 2) / w) / 2)}px ${family}`;
}

/**
 * Draws a column header: layer name and two caption lines, underlined when selected. Text shrinks
 * a little to stay within `maxW` (narrow columns on phones), so neighbouring headers never touch.
 */
function drawHeader(ctx: CanvasRenderingContext2D, cx: number, name: string, lines: [string, string], selected: boolean, maxW = Infinity): void {
  const p = palette();
  ctx.textAlign = 'center';
  ctx.textBaseline = 'alphabetic';
  ctx.fillStyle = selected ? p.accent : p.ink;
  fitFont(ctx, name, 800, 13, 10.5, SANS, maxW);
  ctx.fillText(name, cx, 16);
  ctx.fillStyle = p.muted;
  fitFont(ctx, lines[0], 400, 10.5, 8.5, MONO, maxW);
  ctx.fillText(lines[0], cx, 31);
  fitFont(ctx, lines[1], 400, 10.5, 8.5, MONO, maxW);
  ctx.fillText(lines[1], cx, 44);
  if (selected) {
    ctx.fillStyle = p.accent;
    ctx.fillRect(cx - 18, 51, 36, 3);
  }
}

/** Hatches a frozen layer's column, as 01 hatches its card: its weights stay as they are. */
function hatch(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, hgt: number): void {
  const p = palette();
  ctx.save();
  ctx.beginPath();
  ctx.rect(x, y, w, hgt);
  ctx.clip();
  ctx.beginPath();
  for (let t = -hgt; t < w; t += 9) {
    ctx.moveTo(x + t, y + hgt);
    ctx.lineTo(x + t + hgt, y);
  }
  ctx.strokeStyle = p.hair;
  ctx.lineWidth = 1.5;
  ctx.stroke();
  ctx.restore();
}

const frozenNames = (n: number) => {
  const out: string[] = [];
  for (let i = 0; i < n; i++) if (store.isFrozen(i)) out.push(layerName(i === n - 1 ? null : store.spec[i], i));
  return out;
};

/**
 * Grid sides of the point-data unit maps. While training: a pass of about 8 ms. Paused: 32². A
 * slow network gets coarser maps rather than a long pass (see Budget).
 */
export const BUDGET_TILES: { running: Budget; paused: Budget } = {
  running: { budget: 8, min: 16, max: 32, cap: 30, floor: 8 },
  paused: { budget: 40, min: 32, max: 32, cap: 150, floor: 12 },
};

interface Tiles {
  res: number;
  r: number;
  dims: 2 | 3;
  axes: [number, number];
  /** Slice position (3-D data). */
  pos: number;
  sliceAxis: number;
  /** Feature values over the plane, n × F. */
  feats: Float32Array;
  /** Per block, n × units (hidden activations, then output logits). */
  acts: Float32Array[];
  probs: Float32Array;
  /** The page network these maps were computed for. */
  net: Network;
  features: string;
  rev: number;
}

/** Colour look-up tables of the current theme: diverging over [−1, 1], sequential over [0, 1]. */
function luts(): { div: Uint8ClampedArray; seq: Uint8ClampedArray } {
  const div = new Uint8ClampedArray(511 * 3);
  const seq = new Uint8ClampedArray(256 * 3);
  const c: RGB = [0, 0, 0];
  for (let i = 0; i < 511; i++) {
    diverging(i / 255 - 1, c);
    div.set(c, 3 * i);
  }
  for (let i = 0; i < 256; i++) {
    sequential(i / 255, c);
    seq.set(c, 3 * i);
  }
  return { div, seq };
}

export function mountNetworkView(): void {
  const root = $('netview');
  const strip = h('div', { class: 'probe-strip', role: 'group', 'aria-label': 'Choose the input digit' });
  const caption = h('span', { class: 'probe-caption' });
  const randomBtn = h('button', { type: 'button', class: 'btn btn-sm', disabled: true }, 'Random test digit');
  const canvas = h('canvas', { role: 'img', 'aria-label': 'Network diagram with activations' }) as HTMLCanvasElement;
  const box = h('div', { class: 'canvas-box' }, canvas);
  const ramp = (signed: boolean) => {
    const c = h('canvas', { class: 'ramp' }) as HTMLCanvasElement;
    const paint = () => {
      const ctx = fitCanvas(c, 96, 10);
      for (let i = 0; i < 96; i++) {
        const t = i / 95;
        ctx.fillStyle = css(signed ? diverging(t * 2 - 1) : sequential(t));
        ctx.fillRect(i, 0, 1, 10);
      }
    };
    paint();
    onThemeChange(paint);
    return c;
  };
  const legendNote = h('span', { class: 'legend-item' }, 'Lines show summed weights between units, maps and channels.');
  const wLabel = h('span', null, 'Weight');
  const aLabel = h('span', null, 'Activation');
  const frozenKey = h('span', { class: 'legend-item nv-frozen-key', hidden: true }, h('i', { class: 'nv-hatch', 'aria-hidden': 'true' }), 'Frozen layer');
  const legend = h(
    'div',
    { class: 'legend' },
    h('span', { class: 'legend-item' }, wLabel, h('span', { class: 'mono' }, '−'), ramp(true), h('span', { class: 'mono' }, '+')),
    h('span', { class: 'legend-item' }, aLabel, h('span', { class: 'mono' }, '0'), ramp(false), h('span', { class: 'mono' }, 'max')),
    frozenKey,
    legendNote,
  );
  const planeNote = h('p', { class: 'hint nv-plane-note', hidden: true });
  const resNote = h('p', { class: 'hint nv-res-note', hidden: true });
  root.append(h('div', { class: 'probe-row' }, h('span', { class: 'label' }, 'Input'), strip, randomBtn, caption), planeNote, box, legend, resNote);

  // ── Input picker ──
  const thumbs = new Map<string, HTMLButtonElement>();
  const buildStrip = () => {
    clear(strip);
    thumbs.clear();
    const d = store.data;
    const info = store.info;
    strip.setAttribute('aria-label', `Choose the input ${noun(info)}`);
    randomBtn.textContent = `Random test ${noun(info)}`;
    if (!d) {
      randomBtn.disabled = true;
      return;
    }
    for (let k = 0; k < info.classes.length; k++) {
      const i = d.testY.indexOf(k);
      if (i < 0) continue;
      const name = info.classes[k];
      const c = document.createElement('canvas');
      let title: string;
      if (info.kind === 'points') {
        const ctx = fitCanvas(c, 34, 34);
        const p = palette();
        ctx.fillStyle = p.surface;
        ctx.fillRect(0, 0, 34, 34);
        // The class's own mark (shape and colour), as in 03 and 07.
        drawMark(ctx, k, 'train', 17, 15, 6.5, { color: p.cat[k % 10], ink: p.ink, surface: p.surface });
        title = `Test point #${i} (${name})`;
      } else {
        const x = new Float32Array(d.inputSize);
        for (let j = 0; j < d.inputSize; j++) x[j] = d.testX[i * d.inputSize + j] * d.scale;
        paintSample(c, x, d.input, 34);
        title = info.id === 'mnist' ? `Test digit #${i} (a ${k})` : `Test image #${i} (${name})`;
      }
      const b = h('button', { type: 'button', class: 'thumb', title, 'aria-label': title, 'aria-pressed': 'false' }, c, h('span', { class: 'thumb-label' }, info.glyphs[k])) as HTMLButtonElement;
      b.addEventListener('click', () => setProbe(testProbe(d, i)));
      thumbs.set(`test:${i}`, b);
      strip.append(b);
    }
    randomBtn.disabled = false;
    syncStrip();
  };
  randomBtn.addEventListener('click', () => {
    const d = store.data;
    if (!d) return;
    setProbe(testProbe(d, Math.floor(Math.random() * d.testY.length)));
  });
  const syncStrip = () => {
    const key = store.probe?.key;
    for (const [k, b] of thumbs) b.setAttribute('aria-pressed', String(k === key));
    caption.textContent = store.probe?.caption ?? '';
  };

  let hits: Hit[] = [];
  /** Point data: the network has more weights than LINES, and only the strongest are drawn. */
  let linesCapped = false;
  const LINES = 1200;

  /** Frozen layers' tooltips and labels. */
  const FROZEN_TIP = '\nFrozen: training leaves this layer’s weights alone.';
  const syncFrozenKey = (n: number) => {
    frozenKey.hidden = !frozenNames(n).length;
  };

  // ── Images ──
  const renderImages = () => {
    const p = palette();
    const info = store.info;
    const net = store.net;
    const blocks = net.blocks;
    const probe = store.probe;
    const shape = net.arch.input;
    const K = net.classes;
    const fits = !!probe && probe.x.length === net.inputSize;
    if (fits) net.forward(probe.x);
    const named = info.id !== 'mnist';
    const cols = blocks.length + 1;
    const avail = box.clientWidth || 600;
    const extra = named ? 100 : 56;
    const W = Math.max(avail, cols * 88 + extra);
    const H = Math.round(Math.min(680, Math.max(420, W * 0.72)));
    const ctx = fitCanvas(canvas, W, H);
    hits = [];
    canvas.dataset.view = 'image';
    const frozen = frozenNames(blocks.length);
    canvas.dataset.frozen = frozen.join(',');
    syncFrozenKey(blocks.length);
    canvas.setAttribute(
      'aria-label',
      `Network diagram: the ${shape.h}×${shape.w} ${shape.c === 3 ? 'colour' : 'grey'} input, the activations of every layer for it, and the output probabilities for ${K} classes.${frozen.length ? ` Frozen: ${frozen.join(', ')}.` : ''}`,
    );

    const colW = (W - extra) / cols;
    const headerH = 60;
    const top = headerH + 10;
    const bottom = H - 6;
    const availH = bottom - top;
    const centers = Array.from({ length: cols }, (_, i) => colW * (i + 0.5));
    centers[cols - 1] -= 6;

    const layouts: Item[][] = [];
    layouts.push(fitItems(1, centers[0], top, colW * 0.78, availH, 120, 0));
    blocks.forEach((b, i) => {
      const cx = centers[i + 1];
      if (b.kind === 'conv') layouts.push(fitItems(b.spec.filters, cx, top, colW * 0.74, availH, 76, 6));
      else if (i === blocks.length - 1) layouts.push(fitItems(K, cx - 10, top, 26, availH, 26, 6));
      else layouts.push(fitItems(b.spec.units, cx, top, colW * 0.5, availH, 22, 4));
    });

    const hl = store.highlight;
    if (hl) {
      const col = Math.min(cols - 1, hl.block + 1);
      ctx.fillStyle = p.surface;
      ctx.fillRect(centers[col] - colW / 2 + 2, 0, colW - 4 + (col === cols - 1 ? extra : 0), H);
      ctx.fillStyle = hl.dir === 'fwd' ? p.ink : p.accent;
      ctx.fillRect(centers[col] - colW / 2 + 2, 0, colW - 4 + (col === cols - 1 ? extra : 0), 3);
    }
    blocks.forEach((_b, i) => {
      if (store.isFrozen(i)) hatch(ctx, centers[i + 1] - colW / 2 + 2, 58, colW - 4 + (i === blocks.length - 1 ? extra - 10 : 0), H - 58);
    });

    // Connections run between column edges so they never cross a neighbouring map.
    const right = layouts.map((items) => Math.max(...items.map((it) => it.x + it.s)));
    const left = layouts.map((items) => Math.min(...items.map((it) => it.x)));
    const maxLines = 3500;
    for (let i = 0; i < blocks.length; i++) {
      const b = blocks[i];
      const src = layouts[i];
      const dst = layouts[i + 1];
      const st = strengths(b);
      let m = 0;
      const all: number[] = [];
      for (const row of st) for (const v of row) {
        const a = Math.abs(v);
        if (a > m) m = a;
        all.push(a);
      }
      if (m === 0) continue;
      let thresh = 0;
      if (all.length > maxLines) {
        all.sort((a, b2) => b2 - a);
        thresh = all[maxLines];
      }
      const buckets = 6;
      for (const sign of [1, -1]) {
        for (let q = 0; q < buckets; q++) {
          ctx.beginPath();
          let any = false;
          for (let j = 0; j < st.length; j++) {
            const row = st[j];
            const t = dst[j];
            if (!t) continue;
            for (let c = 0; c < row.length; c++) {
              const v = row[c];
              if (sign > 0 ? v <= 0 : v >= 0) continue;
              const a = Math.abs(v);
              if (a < thresh) continue;
              const bucket = Math.min(buckets - 1, Math.floor((a / m) * buckets));
              if (bucket !== q) continue;
              const s = src[Math.min(c, src.length - 1)];
              ctx.moveTo(right[i] + 1, s.y + s.s / 2);
              ctx.lineTo(left[i + 1] - 1, t.y + t.s / 2);
              any = true;
            }
          }
          if (!any) continue;
          ctx.strokeStyle = sign > 0 ? p.accent : p.neg;
          ctx.globalAlpha = 0.05 + 0.6 * Math.pow((q + 1) / buckets, 1.6);
          ctx.lineWidth = q >= buckets - 2 ? 1.4 : 1;
          ctx.stroke();
        }
      }
      ctx.globalAlpha = 1;
    }

    const headerX = (col: number) => (col === cols - 1 ? centers[col] + 12 : centers[col]);
    drawHeader(ctx, headerX(0), 'Input', [`${shape.h}×${shape.w}`, shape.c === 3 ? 'colour' : 'grey'], false, colW - 8);
    blocks.forEach((_b, i) => {
      const spec = i === blocks.length - 1 ? null : store.spec[i];
      drawHeader(ctx, headerX(i + 1), layerName(spec, i), headerLines(spec, info, store.isFrozen(i)), store.selected === i, colW - 8);
    });

    // Input
    const inItem = layouts[0][0];
    if (fits) drawSample(ctx, probe!.x, shape, inItem.x, inItem.y, inItem.s, inItem.s);
    ctx.strokeStyle = p.ink;
    ctx.lineWidth = 1;
    ctx.strokeRect(inItem.x - 0.5, inItem.y - 0.5, inItem.s + 1, inItem.s + 1);
    hits.push({ x: inItem.x, y: inItem.y, w: inItem.s, h: inItem.s, block: -1, unit: null, text: `Input · ${shape.h}×${shape.w}${shape.c === 3 ? ' colour' : ''} pixels\n${probe?.caption ?? ''}` });

    // Signed activations (tanh, linear, Leaky ReLU layers) use the red/blue ramp: the legend says so.
    let anySigned = false;
    blocks.forEach((b, i) => {
      const items = layouts[i + 1];
      const out = b.out;
      const signed = hasNegative(out);
      const max = maxAbs(out) || 1;
      const isOut = i === blocks.length - 1;
      if (!isOut && signed) anySigned = true;
      const name = layerName(isOut ? null : store.spec[i], i);
      const fz = store.isFrozen(i) ? FROZEN_TIP : '';
      if (b.kind === 'conv') {
        const { h: Hh, w: Ww } = b.outShape;
        const HW = Hh * Ww;
        items.forEach((it, f) => {
          drawMap(ctx, out, f * HW, Hh, Ww, it.x, it.y, it.s, it.s, signed, max);
          ctx.strokeStyle = store.selected === i && store.selectedUnit === f ? p.accent : p.hair;
          ctx.lineWidth = store.selected === i && store.selectedUnit === f ? 2 : 1;
          ctx.strokeRect(it.x - 0.5, it.y - 0.5, it.s + 1, it.s + 1);
          let mx = -Infinity;
          let mean = 0;
          for (let k = 0; k < HW; k++) {
            const v = out[f * HW + k];
            mean += v;
            if (v > mx) mx = v;
          }
          hits.push({
            x: it.x, y: it.y, w: it.s, h: it.s, block: i, unit: f,
            text: `${name} · filter ${f + 1}\n${Hh}×${Ww} map · max ${fixed(mx, 3)} · mean ${fixed(mean / HW, 3)}\nColour scale: ${signed ? '±' : '0 … '}${fixed(max, 3)}, shared by this layer${fz}`,
          });
        });
      } else if (isOut) {
        const probs = net.probs;
        let best = 0;
        for (let k = 1; k < K; k++) if (probs[k] > probs[best]) best = k;
        const barX = items[0].x + items[0].s + 8;
        const barW = Math.max(20, W - barX - 34);
        items.forEach((it, k) => {
          const pk = fits ? probs[k] : 0;
          const top = k === best && fits;
          ctx.fillStyle = css(sequential(pk));
          ctx.fillRect(it.x, it.y, it.s, it.s);
          ctx.strokeStyle = top ? p.accent : p.hair;
          ctx.lineWidth = top ? 2 : 1;
          ctx.strokeRect(it.x - 0.5, it.y - 0.5, it.s + 1, it.s + 1);
          ctx.fillStyle = top ? p.accent : p.ink;
          ctx.font = `${top ? 800 : 500} 12px ${MONO}`;
          ctx.textAlign = 'right';
          ctx.textBaseline = 'middle';
          ctx.fillText(info.glyphs[k], it.x - 6, it.y + it.s / 2);
          const by = named ? it.y + it.s - 6 : it.y + it.s / 2 - 3;
          if (named) {
            ctx.font = `${top ? 700 : 400} 10.5px ${SANS}`;
            ctx.textAlign = 'left';
            ctx.textBaseline = 'alphabetic';
            ctx.fillStyle = top ? p.accent : p.ink2;
            ctx.fillText(info.classes[k], barX, it.y + 10, barW + 26);
          }
          ctx.fillStyle = p.hair;
          ctx.fillRect(barX, by, barW, named ? 5 : 6);
          ctx.fillStyle = top ? p.accent : p.ink;
          ctx.fillRect(barX, by, barW * pk, named ? 5 : 6);
          if (probe?.label === k) {
            const my = by + (named ? 2.5 : 3);
            ctx.fillStyle = p.ink;
            ctx.beginPath();
            ctx.moveTo(barX + barW + 4, my);
            ctx.lineTo(barX + barW + 10, my - 4);
            ctx.lineTo(barX + barW + 10, my + 4);
            ctx.fill();
          }
          const what = info.id === 'mnist' ? `digit ${k}` : info.classes[k];
          hits.push({
            x: it.x - 16, y: it.y, w: barX + barW - it.x + 26, h: it.s, block: i, unit: k,
            text: `Output · ${what}${probe?.label === k ? ' (true label)' : ''}\nlogit ${fits ? fixed(b.z[k], 3) : '—'} · p = ${fits ? (probs[k] * 100).toFixed(1) : '—'}%${fz}`,
          });
        });
      } else {
        const z = b.z;
        items.forEach((it, j) => {
          const v = out[j] / max;
          ctx.fillStyle = css(signed ? diverging(v) : sequential(v));
          ctx.fillRect(it.x, it.y, it.s, it.s);
          const sel = store.selected === i && store.selectedUnit === j;
          ctx.strokeStyle = sel ? p.accent : p.hair;
          ctx.lineWidth = sel ? 2 : 1;
          ctx.strokeRect(it.x - 0.5, it.y - 0.5, it.s + 1, it.s + 1);
          hits.push({ x: it.x, y: it.y, w: it.s, h: it.s, block: i, unit: j, text: `${name} · unit ${j + 1}\nz = ${fixed(z[j], 3)} · a = ${fixed(out[j], 3)}\nColour scale: ${signed ? '±' : '0 … '}${fixed(max, 3)}, shared by this layer${fz}` });
        });
      }
    });
    wLabel.textContent = anySigned ? 'Weight or activation' : 'Weight';
    aLabel.textContent = anySigned ? 'Activation ≥ 0' : 'Activation';
    legend.dataset.signed = String(anySigned);
  };

  // ── Points: every unit over the plane ──
  const ev = new Evaluator();
  const cost = new Cost();
  let tiles: Tiles | null = null;
  let tileRev = 0;
  let job: { job: GridJob; coords: Float32Array; res: number; r: number; dims: 2 | 3; axes: [number, number]; pos: number; sliceAxis: number } | null = null;
  /** Main-thread time of the pass under way (evaluation, painting the maps, drawing). */
  let passMs = 0;

  /** The maps of the current network, or null while they are being computed for a new one. */
  const currentTiles = () => (tiles && tiles.net === store.net && tiles.features === store.features.join() ? tiles : null);

  const startTiles = (d: Data) => {
    const dims = d.points!.dims;
    const r = pointDomain(d);
    const axes: [number, number] = dims === 2 ? [0, 1] : sliceAxes(slice.axis);
    const fixedAt = [0, 0, 0];
    if (dims === 3) fixedAt[slice.axis] = slice.pos;
    // A network that takes long per point gets coarser maps.
    const res = cost.res(store.running ? BUDGET_TILES.running : BUDGET_TILES.paused, 2);
    ev.sync();
    const coords = gridCoords(dims, res, r, axes, fixedAt);
    job = { job: new GridJob(ev.ev, coords, dims, true), coords, res, r, dims, axes, pos: slice.pos, sliceAxis: slice.axis };
  };

  /** Runs one slice of the tile pass; true when the maps are done. */
  const stepTiles = (): boolean => {
    if (!job) return false;
    if (!job.job.step(sliceMs(), cost.perPoint)) return false;
    const J = job;
    job = null;
    cost.add(J.job.busy, J.res * J.res);
    tiles = {
      res: J.res, r: J.r, dims: J.dims, axes: J.axes, pos: J.pos, sliceAxis: J.sliceAxis,
      feats: featurize(J.coords, J.dims, store.features),
      acts: J.job.acts!,
      probs: J.job.probs!,
      net: store.net,
      features: store.features.join(),
      rev: ++tileRev,
    };
    canvas.dataset.weights = String(store.weightsRev);
    canvas.dataset.res = String(J.res);
    return true;
  };

  // ── One picture holding every map (an atlas), repainted only when the maps change ──
  const atlas = document.createElement('canvas');
  let atlasKey = '';
  let atlasCols = 1;
  /** Colour scale of each map, for the tooltips: [feature, hidden layer 0, …]. */
  let scales: { max: number; signed: boolean }[][] = [];
  const paintAtlas = (T: Tiles) => {
    const p = palette();
    const key = `${T.rev}:${p.surface}:${p.ink}:${p.accent}:${p.neg}:${p.cat.join()}`;
    if (key === atlasKey) return;
    atlasKey = key;
    const net = T.net;
    const blocks = net.blocks;
    const F = store.features.length;
    const K = net.classes;
    const hidden = blocks.slice(0, -1).map((b) => b.out.length);
    const N = F + hidden.reduce((a, b) => a + b, 0) + K;
    const R = T.res;
    const S = R + 2; // one pixel of padding around each map (edge copied), so smoothing never bleeds
    atlasCols = Math.max(1, Math.ceil(Math.sqrt(N)));
    const rows = Math.ceil(N / atlasCols);
    atlas.width = atlasCols * S;
    atlas.height = rows * S;
    const actx = atlas.getContext('2d')!;
    const img = actx.createImageData(atlas.width, atlas.height);
    const px = img.data;
    const AW = atlas.width;
    const { div, seq } = luts();
    let t = 0;
    const put = (colour: (q: number, out: Uint8ClampedArray, o: number) => void) => {
      const ox = (t % atlasCols) * S;
      const oy = Math.floor(t / atlasCols) * S;
      t++;
      for (let yy = 0; yy < S; yy++) {
        const sy = Math.min(R - 1, Math.max(0, yy - 1));
        for (let xx = 0; xx < S; xx++) {
          const sx = Math.min(R - 1, Math.max(0, xx - 1));
          const o = 4 * ((oy + yy) * AW + ox + xx);
          colour(sy * R + sx, px, o);
          px[o + 3] = 255;
        }
      }
    };
    const ramp = (v: number, signed: boolean, out: Uint8ClampedArray, o: number) => {
      if (signed) {
        const i = 3 * Math.round(255 * (Math.max(-1, Math.min(1, v)) + 1));
        out[o] = div[i];
        out[o + 1] = div[i + 1];
        out[o + 2] = div[i + 2];
      } else {
        const i = 3 * Math.round(255 * Math.max(0, Math.min(1, v)));
        out[o] = seq[i];
        out[o + 1] = seq[i + 1];
        out[o + 2] = seq[i + 2];
      }
    };
    scales = [];
    // Features: each on its own range over the whole domain (a constant slice shows its value).
    const fsc = store.features.map((id) => featureScale(id, T.r));
    scales.push(fsc);
    for (let f = 0; f < F; f++) {
      const { max, signed } = fsc[f];
      const inv = 1 / (max || 1);
      put((q, out, o) => ramp(T.feats[q * F + f] * inv, signed, out, o));
    }
    // Hidden units: one scale per layer.
    hidden.forEach((units, i) => {
      const A = T.acts[i];
      const signed = hasNegative(A);
      const max = robustScale(A);
      scales.push([{ max, signed }]);
      const inv = 1 / max;
      for (let j = 0; j < units; j++) put((q, out, o) => ramp(A[q * units + j] * inv, signed, out, o));
    });
    // Classes: probability, from the surface toward the class colour.
    const rgb = p.rgb;
    for (let k = 0; k < K; k++) {
      const c = rgb.cat[k % 10];
      put((q, out, o) => {
        const a = 0.85 * T.probs[q * K + k];
        out[o] = rgb.surface[0] + (c[0] - rgb.surface[0]) * a;
        out[o + 1] = rgb.surface[1] + (c[1] - rgb.surface[1]) * a;
        out[o + 2] = rgb.surface[2] + (c[2] - rgb.surface[2]) * a;
      });
    }
    actx.putImageData(img, 0, 0);
  };
  /** Draws map number `t` of the atlas into `it`. */
  const blit = (ctx: CanvasRenderingContext2D, T: Tiles, t: number, it: Item) => {
    const S = T.res + 2;
    ctx.drawImage(atlas, (t % atlasCols) * S + 1, Math.floor(t / atlasCols) * S + 1, T.res, T.res, it.x, it.y, it.s, it.s);
  };

  /**
   * Lays out one column of n tiles, top-aligned in a box wAvail × hAvail: as large as possible up
   * to s0, in sub-columns when a tall stack would get too small (not for the features, whose
   * labels sit to the left). Value labels go under tiles that are big enough to carry them.
   */
  const tileColumn = (n: number, cx: number, top: number, wAvail: number, hAvail: number, s0: number, labelH: number, single: boolean): Item[] => {
    let best = { k: 1, s: 0, lab: 0, g: 6 };
    for (let k = 1; k <= (single ? 1 : 12); k++) {
      const rows = Math.ceil(n / k);
      const g = k > 2 ? 3 : 6;
      const sW = (wAvail - (k - 1) * g) / k;
      let lab = labelH;
      let s = Math.min(s0, sW, (hAvail - rows * (lab + g)) / rows);
      if (s < 26) {
        lab = 0;
        s = Math.min(s0, sW, (hAvail - rows * g) / rows);
      }
      if (s > best.s + 0.5) best = { k, s, lab, g };
    }
    const { k, lab, g } = best;
    const s = Math.max(4, Math.floor(best.s));
    const rows = Math.ceil(n / k);
    const x0 = cx - (k * s + (k - 1) * g) / 2;
    const items: Item[] = [];
    for (let i = 0; i < n; i++) {
      const c = Math.floor(i / rows);
      const r = i % rows;
      items.push({ x: x0 + c * (s + g), y: top + r * (s + lab + g), s, lab });
    }
    return items;
  };

  const renderPoints = () => {
    const d = store.data;
    const p = palette();
    const info = store.info;
    const net = store.net;
    const blocks = net.blocks;
    const K = net.classes;
    const probe = store.probe;
    const fits = netFits() && !!d?.points;
    const pfits = fits && !!probe && probe.x.length === net.inputSize;
    if (pfits) net.forward(probe!.x);
    hits = [];
    canvas.dataset.view = 'points';
    const T = fits ? currentTiles() : null;
    if (!T) {
      fitCanvas(canvas, box.clientWidth || 400, 200);
      return;
    }
    paintAtlas(T);
    const F = store.features.length;
    const defs = featureDefs(T.dims, store.features);
    const counts = [F, ...blocks.map((b) => (b.kind === 'dense' ? b.spec.units : 0))];
    const cols = counts.length;
    const avail = box.clientWidth || 480;
    const leftPad = 46;
    const rightPad = avail < 420 ? 78 : 92;
    const W = Math.max(avail, leftPad + rightPad + cols * 54);
    const colW = (W - leftPad - rightPad) / cols;
    const centers = Array.from({ length: cols }, (_, i) => leftPad + colW * (i + 0.5));
    // Tiles as large as fits, up to 8 in a column; larger layers fold into sub-columns.
    const maxN = Math.min(8, Math.max(...counts));
    const s0 = Math.max(18, Math.min(56, Math.floor(colW * 0.62), Math.floor(460 / maxN) - 20));
    const labelH = 13;
    const hAvail = 8 * (s0 + labelH + 6);
    const top = 70;
    const layouts = counts.map((n, i) => tileColumn(n, centers[i], top, colW * 0.84, hAvail, s0, labelH, i === 0));
    const bottomY = Math.max(...layouts.flat().map((it) => it.y + it.s + (it.lab ?? 0)));
    const H = Math.max(300, Math.ceil(bottomY + 14));
    const ctx = fitCanvas(canvas, W, H);
    const nTiles = counts.reduce((a, b) => a + b, 0);
    canvas.dataset.tiles = String(nTiles);
    const sliceName = `${axisName(T.sliceAxis)} = ${fixed(T.pos, 2)}`;
    const planeName = T.dims === 2 ? 'the input plane' : `the slice ${sliceName}`;
    const frozen = frozenNames(blocks.length);
    canvas.dataset.frozen = frozen.join(',');
    syncFrozenKey(blocks.length);
    canvas.setAttribute(
      'aria-label',
      `Network diagram: ${nTiles} small maps, one per input feature, hidden unit and class, each showing its value over ${planeName}; lines show the weights between them.${frozen.length ? ` Frozen: ${frozen.join(', ')}.` : ''}`,
    );

    const hl = store.highlight;
    if (hl) {
      const col = Math.min(cols - 1, hl.block + 1);
      const x0 = centers[col] - colW / 2 + 2;
      const w = colW - 4 + (col === cols - 1 ? rightPad - 6 : 0);
      ctx.fillStyle = p.surface;
      ctx.fillRect(x0, 0, w, H);
      ctx.fillStyle = hl.dir === 'fwd' ? p.ink : p.accent;
      ctx.fillRect(x0, 0, w, 3);
    }
    blocks.forEach((_b, i) => {
      if (store.isFrozen(i)) hatch(ctx, centers[i + 1] - colW / 2 + 2, 58, colW - 4 + (i === blocks.length - 1 ? rightPad - 6 : 0), H - 58);
    });

    // Connections: one curve per weight, width and colour by sign and size, stroked in a few
    // batches of similar weights. Drawing them is most of the cost of a redraw (rasterising
    // thousands of curves), so a large network shows only its strongest LINES weights, shared
    // among the layers by size.
    const BUCKETS = 8;
    const totalW = blocks.reduce((a, b) => a + (b.kind === 'dense' ? b.W.length : 0), 0);
    linesCapped = totalW > LINES;
    for (let i = 0; i < blocks.length; i++) {
      const b = blocks[i];
      if (b.kind !== 'dense') continue;
      const src = layouts[i];
      const dst = layouts[i + 1];
      const n = b.inSize;
      const absW = new Float32Array(b.W.length);
      for (let q = 0; q < b.W.length; q++) absW[q] = Math.abs(b.W[q]);
      let m = 0;
      for (let q = 0; q < absW.length; q++) if (absW[q] > m) m = absW[q];
      if (m === 0) continue;
      let thresh = 0;
      const keep = linesCapped ? Math.max(48, Math.floor((LINES * absW.length) / totalW)) : absW.length;
      if (absW.length > keep) thresh = absW.slice().sort()[absW.length - keep];
      const paths: Path2D[][] = [[], []];
      for (let s = 0; s < 2; s++) for (let q = 0; q < BUCKETS; q++) paths[s].push(new Path2D());
      const used = new Uint8Array(2 * BUCKETS);
      for (let q = 0; q < b.W.length; q++) {
        if (absW[q] < thresh || absW[q] === 0) continue;
        const j = Math.floor(q / n);
        const c = q % n;
        const s = src[c];
        const t = dst[j];
        if (!s || !t) continue;
        const sign = b.W[q] >= 0 ? 0 : 1;
        const bucket = Math.min(BUCKETS - 1, Math.floor((absW[q] / m) * BUCKETS));
        const x1 = s.x + s.s + 1;
        const y1 = s.y + s.s / 2;
        const x2 = t.x - 1;
        const y2 = t.y + t.s / 2;
        const mx = (x1 + x2) / 2;
        const path = paths[sign][bucket];
        path.moveTo(x1, y1);
        path.bezierCurveTo(mx, y1, mx, y2, x2, y2);
        used[sign * BUCKETS + bucket] = 1;
      }
      for (let q = 0; q < BUCKETS; q++) {
        const a = (q + 0.5) / BUCKETS;
        for (let s = 0; s < 2; s++) {
          if (!used[s * BUCKETS + q]) continue;
          ctx.strokeStyle = s === 0 ? p.accent : p.neg;
          ctx.globalAlpha = 0.18 + 0.7 * a;
          ctx.lineWidth = 0.4 + 3.4 * a;
          ctx.stroke(paths[s][q]);
        }
      }
      ctx.globalAlpha = 1;
    }

    // Headers
    const headW = colW - 8;
    drawHeader(ctx, centers[0], 'Features', [`${F} input${F === 1 ? '' : 's'}`, T.dims === 2 ? `over ${axisName(0)}, ${axisName(1)}` : sliceName], false, headW);
    blocks.forEach((_b, i) => {
      const isOut = i === blocks.length - 1;
      const spec = isOut ? null : store.spec[i];
      drawHeader(ctx, centers[i + 1] + (isOut ? 20 : 0), layerName(spec, i), headerLines(spec, info, store.isFrozen(i)), store.selected === i, isOut ? headW + 40 : headW);
    });

    const r = T.r;
    const pc = probe?.coords;
    // The current input: an accent cross on a surface halo (it reads by its shape on any colour).
    const crosses = new Map<string, number[]>();
    const marker = (it: Item) => {
      if (!pc || pc.length !== T.dims || it.s < 18) return;
      const u = pc[T.axes[0]];
      const v = pc[T.axes[1]];
      const on = T.dims === 2 || onCurrentSlice(pc, r);
      const x = it.x + ((u + r) / (2 * r)) * it.s;
      const y = it.y + ((r - v) / (2 * r)) * it.s;
      if (x < it.x || x > it.x + it.s || y < it.y || y > it.y + it.s) return;
      const size = Math.round(2 * Math.max(3, Math.min(5, it.s / 12))) / 2;
      const key = `${size}:${on ? 1 : 0}`;
      let xy = crosses.get(key);
      if (!xy) crosses.set(key, (xy = []));
      xy.push(x, y);
    };
    // Frames: one path for all, the selected one on top.
    const framePath = new Path2D();
    let selFrame: Item | null = null;
    const frame = (it: Item, sel: boolean) => {
      if (sel) selFrame = it;
      else if (it.s >= 8) framePath.rect(it.x - 0.5, it.y - 0.5, it.s + 1, it.s + 1);
    };
    const labels: { it: Item; text: string; strong: boolean }[] = [];
    const value = (it: Item, text: string, strong = false) => {
      if (it.lab && text) labels.push({ it, text, strong });
    };
    const planeTip = T.dims === 2 ? 'Map: its value over the plane' : `Map: its value over the slice ${sliceName}`;
    const scaleText = (sc: { max: number; signed: boolean }) => `${sc.signed ? '±' : '0 … '}${fixed(sc.max, 2)}`;

    // Feature tiles
    let t = 0;
    ctx.imageSmoothingEnabled = false;
    layouts[0].forEach((it, f) => {
      blit(ctx, T, t++, it);
      frame(it, false);
      marker(it);
      const xv = pfits ? probe!.x[f] : NaN;
      value(it, fixed(xv, 2));
      hits.push({
        x: it.x, y: it.y, w: it.s, h: it.s, block: -1, unit: f,
        text: `Feature ${defs[f].label}: ${defs[f].title}\nFor the current input: ${fixed(xv, 3)}\n${planeTip}\nColour scale: ${scaleText(scales[0][f])}, the feature’s range over the whole ${T.dims === 2 ? 'plane' : 'cube'}`,
      });
    });
    ctx.font = `600 11.5px ${SANS}`;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = p.ink;
    layouts[0].forEach((it, f) => ctx.fillText(defs[f].label, it.x - 6, it.y + it.s / 2));

    // Hidden units and output
    const outLabels: (() => void)[] = [];
    blocks.forEach((b, i) => {
      const items = layouts[i + 1];
      const isOut = i === blocks.length - 1;
      const name = layerName(isOut ? null : store.spec[i], i);
      const fz = store.isFrozen(i) ? FROZEN_TIP : '';
      if (!isOut) {
        const sc = scales[1 + i][0];
        items.forEach((it, j) => {
          blit(ctx, T, t++, it);
          frame(it, store.selected === i && store.selectedUnit === j);
          marker(it);
          value(it, pfits ? fixed(b.out[j], 2) : '');
          hits.push({
            x: it.x, y: it.y, w: it.s, h: it.s, block: i, unit: j,
            text: `${name} · unit ${j + 1}\nFor the current input: z = ${pfits ? fixed(b.z[j], 3) : '—'} · a = ${pfits ? fixed(b.out[j], 3) : '—'}\n${planeTip}\nColour scale: ${scaleText(sc)}, shared by this layer${fz}`,
          });
        });
        return;
      }
      const probs = net.probs;
      let best = 0;
      for (let k = 1; k < K; k++) if (probs[k] > probs[best]) best = k;
      ctx.imageSmoothingEnabled = true;
      items.forEach((it, k) => {
        const tt = { x: it.x + 20, y: it.y, s: it.s, lab: it.lab };
        blit(ctx, T, t++, tt);
        const top = pfits && k === best;
        frame(tt, store.selected === i && store.selectedUnit === k);
        marker(tt);
        value(tt, pfits ? `${(100 * probs[k]).toFixed(0)}%` : '', top);
        outLabels.push(() => {
          const bx = tt.x + tt.s + 7;
          const bw = Math.max(24, W - bx - 10);
          ctx.font = `${top ? 700 : 500} 11px ${SANS}`;
          ctx.textAlign = 'left';
          ctx.textBaseline = 'alphabetic';
          ctx.fillStyle = top ? p.accent : p.ink;
          ctx.fillText(info.classes[k], bx, tt.y + Math.min(tt.s / 2, 14), bw);
          const by = tt.y + Math.min(tt.s / 2, 14) + 5;
          ctx.fillStyle = p.hair;
          ctx.fillRect(bx, by, bw, 5);
          ctx.fillStyle = top ? p.accent : p.cat[k % 10];
          ctx.fillRect(bx, by, bw * (pfits ? probs[k] : 0), 5);
          if (probe?.label === k) {
            ctx.fillStyle = p.ink;
            ctx.font = `400 10px ${MONO}`;
            ctx.fillText('true class', bx, by + 16, bw);
          }
        });
        hits.push({
          x: tt.x, y: tt.y, w: W - tt.x, h: tt.s, block: i, unit: k,
          text: `Output · ${info.classes[k]}${probe?.label === k ? ' (true class)' : ''}\nFor the current input: logit ${pfits ? fixed(b.z[k], 3) : '—'} · p = ${pfits ? (100 * probs[k]).toFixed(1) : '—'}%\nMap: probability of ${info.classes[k]} over ${T.dims === 2 ? 'the plane' : 'the slice'}, 0 to 100%${fz}`,
        });
      });
    });
    ctx.imageSmoothingEnabled = true;
    ctx.strokeStyle = p.hair;
    ctx.lineWidth = 1;
    ctx.stroke(framePath);
    if (selFrame) {
      const it = selFrame as Item;
      ctx.strokeStyle = p.accent;
      ctx.lineWidth = 2;
      ctx.strokeRect(it.x - 0.5, it.y - 0.5, it.s + 1, it.s + 1);
    }
    for (const [key, xy] of crosses) {
      const [size, on] = key.split(':').map(Number);
      probeCrosses(ctx, xy, size, p.accent, p.surface, !on);
    }
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (const l of labels) {
      ctx.font = `${l.strong ? 600 : 400} 10px ${MONO}`;
      ctx.fillStyle = l.strong ? p.accent : p.muted;
      ctx.fillText(l.text, l.it.x + l.it.s / 2, l.it.y + l.it.s + 2);
    }
    for (const fn of outLabels) fn();
    resNote.hidden = T.res >= 16;
    if (!resNote.hidden) resNote.textContent = `This network is large, so the maps are computed on a coarse ${T.res} × ${T.res} grid${store.running ? ' while it trains' : ''}, a few frames at a time, to keep the page responsive.`;
  };

  const setText = (el: HTMLElement, t: string) => {
    if (el.textContent !== t) el.textContent = t;
  };
  const syncNotes = () => {
    const info = store.info;
    const pts = info.kind === 'points';
    planeNote.hidden = !pts;
    if (pts) {
      const where =
        info.dims === 3
          ? `over the slice ${axisName(slice.axis)} = ${fixed(slice.pos, 2)} of the cube (move it in section 03)`
          : `over the whole plane (${axisName(0)} across, ${axisName(1)} up)`;
      setText(planeNote, `Each tile maps one feature, unit or class ${where}. Feature maps use each feature’s full range, a layer’s unit maps share one scale, and class maps show the probability; hover a tile for its scale. The red cross marks the current input; the number under a tile is its value there.`);
      setText(legendNote, linesCapped ? `Lines are the strongest ${LINES.toLocaleString('en-US')} weights: thicker means larger.` : 'Lines are weights: thicker means larger.');
      setText(wLabel, 'Weight or value');
      setText(aLabel, 'Value ≥ 0');
    } else {
      setText(legendNote, 'Lines show summed weights between units, maps and channels.');
      setText(wLabel, 'Weight');
      setText(aLabel, 'Activation');
      resNote.hidden = true;
    }
  };

  const render = (evaluate: boolean, draw: boolean): boolean => {
    if (store.info.kind !== 'points') {
      job = null;
      renderImages();
      return false;
    }
    const d = store.data;
    if (!netFits() || !d?.points) {
      job = null;
      renderPoints();
      return false;
    }
    const t0 = performance.now();
    if (job && !ev.current) job = null; // made for a network that is gone
    if (!job && (evaluate || !currentTiles())) {
      startTiles(d);
      passMs = 0;
    }
    const inPass = !!job;
    const fresh = stepTiles();
    if (fresh || draw) {
      renderPoints();
      syncNotes();
      live.noteDraw(t0, performance.now() - t0);
    }
    // Every millisecond of a pass counts against the budget: evaluation, painting, drawing, and
    // the browser rasterising the drawing afterwards (Live.drawCost).
    if (inPass) passMs += performance.now() - t0;
    if (fresh) {
      live.gap = Math.max(100, 4 * (passMs + live.drawCost));
      canvas.dataset.ms = passMs.toFixed(1);
    }
    return job !== null;
  };
  const live = new Live(box, render);

  const hitAt = (e: MouseEvent) => {
    const r = canvas.getBoundingClientRect();
    const x = e.clientX - r.left;
    const y = e.clientY - r.top;
    return hits.find((t) => x >= t.x - 2 && x <= t.x + t.w + 2 && y >= t.y - 2 && y <= t.y + t.h + 2) ?? null;
  };
  canvas.addEventListener('mousemove', (e) => {
    const t = hitAt(e);
    canvas.style.cursor = t && t.block >= 0 ? 'pointer' : 'default';
    if (t) showTip(t.text, e.clientX, e.clientY);
    else hideTip();
  });
  canvas.addEventListener('mouseleave', hideTip);
  canvas.addEventListener('click', (e) => {
    const t = hitAt(e);
    if (t && t.block >= 0) select(t.block, t.unit);
  });

  const again = () => live.request(true);
  const redraw = () => live.request(false);
  new ResizeObserver(redraw).observe(box);
  onThemeChange(() => {
    buildStrip();
    redraw();
  });
  store.on('weights', again);
  store.on('model', () => {
    job = null;
    live.restart();
  });
  store.on('data', () => {
    tiles = null;
    job = null;
    buildStrip();
    live.restart();
  });
  for (const ev2 of ['probe', 'select', 'highlight', 'frozen'] as const) store.on(ev2, redraw);
  store.on('probe', syncStrip);
  store.on('dataset', () => {
    tiles = null;
    job = null;
    syncNotes();
    buildStrip();
    redraw();
  });
  let wasRunning = false;
  store.on('status', () => {
    // Full-size maps again once training pauses.
    if (wasRunning && !store.running && store.info.kind === 'points') again();
    wasRunning = store.running;
  });
  onSlice(() => {
    syncNotes();
    if (store.info.kind === 'points' && store.info.dims === 3) again();
  });
  syncNotes();
  again();
}
