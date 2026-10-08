import './networkView.css';
import { select, setProbe, testProbe } from '../actions';
import { fixed } from '../analysis/stats';
import { noun, type Data, type DatasetInfo } from '../data/datasets';
import { featureDefs, featurize } from '../data/features';
import { gridCoords, pointDomain, PointEvaluator } from '../data/grid';
import type { Block } from '../nn/network';
import { store } from '../store';
import { layerName } from './builder';
import { ACTIVATIONS, type LayerSpec } from '../nn/types';
import { axisName } from './boundaryMath';
import { Live, netFits } from './boundary2d';
import { onSlice, slice, sliceAxes } from './boundary3d';
import { $, clear, h } from './dom';
import { drawMap, drawSample, fitCanvas, hasNegative, maxAbs, paintSample } from './draw';
import { css, diverging, onThemeChange, palette, sequential } from './theme';
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

/** Two short caption lines for a column header. */
function headerLines(spec: LayerSpec | null, info: DatasetInfo): [string, string] {
  if (!spec) return [info.id === 'mnist' ? '10 digits' : `${info.classes.length} classes`, 'softmax'];
  const act = ACTIVATIONS.find((a) => a.id === spec.act)!.label;
  if (spec.kind === 'conv') return [`${spec.filters} × ${spec.kernel}×${spec.kernel}`, `${act}${spec.pool ? ' · pool' : ''}`];
  return [`${spec.units} units`, act];
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

/** Draws a column header: layer name and two caption lines, underlined when selected. */
function drawHeader(ctx: CanvasRenderingContext2D, cx: number, name: string, lines: [string, string], selected: boolean): void {
  const p = palette();
  ctx.textAlign = 'center';
  ctx.textBaseline = 'alphabetic';
  ctx.fillStyle = selected ? p.accent : p.ink;
  ctx.font = `800 13px ${SANS}`;
  ctx.fillText(name, cx, 16);
  ctx.fillStyle = p.muted;
  ctx.font = `400 10.5px ${MONO}`;
  ctx.fillText(lines[0], cx, 31);
  ctx.fillText(lines[1], cx, 44);
  if (selected) {
    ctx.fillStyle = p.accent;
    ctx.fillRect(cx - 18, 51, 36, 3);
  }
}

/** Pixel resolution of each unit's map over the input plane (point datasets); half for heavy networks while training. */
const TILE_RES = 32;
const TILE_RES_FAST = 16;

interface Tiles {
  res: number;
  r: number;
  dims: 2 | 3;
  axes: [number, number];
  /** Feature values over the plane, n × F. */
  feats: Float32Array;
  /** Per block, n × units (hidden activations, then output logits). */
  acts: Float32Array[];
  probs: Float32Array;
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
  const legend = h(
    'div',
    { class: 'legend' },
    h('span', { class: 'legend-item' }, wLabel, h('span', { class: 'mono' }, '−'), ramp(true), h('span', { class: 'mono' }, '+')),
    h('span', { class: 'legend-item' }, aLabel, h('span', { class: 'mono' }, '0'), ramp(false), h('span', { class: 'mono' }, 'max')),
    legendNote,
  );
  const planeNote = h('p', { class: 'hint nv-plane-note', hidden: true });
  root.append(h('div', { class: 'probe-row' }, h('span', { class: 'label' }, 'Input'), strip, randomBtn, caption), planeNote, box, legend);

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
        ctx.beginPath();
        ctx.arc(17, 15, 7, 0, 2 * Math.PI);
        ctx.fillStyle = p.cat[k % 10];
        ctx.fill();
        ctx.lineWidth = 1;
        ctx.strokeStyle = p.ink;
        ctx.stroke();
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
    canvas.setAttribute('aria-label', `Network diagram: the ${shape.h}×${shape.w} ${shape.c === 3 ? 'colour' : 'grey'} input, the activations of every layer for it, and the output probabilities for ${K} classes.`);

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
    drawHeader(ctx, headerX(0), 'Input', [`${shape.h}×${shape.w}`, shape.c === 3 ? 'colour' : 'grey'], false);
    blocks.forEach((_b, i) => {
      const spec = i === blocks.length - 1 ? null : store.spec[i];
      drawHeader(ctx, headerX(i + 1), layerName(spec, i), headerLines(spec, info), store.selected === i);
    });

    // Input
    const inItem = layouts[0][0];
    if (fits) drawSample(ctx, probe!.x, shape, inItem.x, inItem.y, inItem.s, inItem.s);
    ctx.strokeStyle = p.ink;
    ctx.lineWidth = 1;
    ctx.strokeRect(inItem.x - 0.5, inItem.y - 0.5, inItem.s + 1, inItem.s + 1);
    hits.push({ x: inItem.x, y: inItem.y, w: inItem.s, h: inItem.s, block: -1, unit: null, text: `Input · ${shape.h}×${shape.w}${shape.c === 3 ? ' colour' : ''} pixels\n${probe?.caption ?? ''}` });

    blocks.forEach((b, i) => {
      const items = layouts[i + 1];
      const out = b.out;
      const signed = hasNegative(out);
      const max = maxAbs(out) || 1;
      const isOut = i === blocks.length - 1;
      const name = layerName(isOut ? null : store.spec[i], i);
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
            text: `${name} · filter ${f + 1}\n${Hh}×${Ww} map · max ${mx.toFixed(3)} · mean ${(mean / HW).toFixed(3)}`,
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
            text: `Output · ${what}${probe?.label === k ? ' (true label)' : ''}\nlogit ${fits ? b.z[k].toFixed(3) : '—'} · p = ${fits ? (probs[k] * 100).toFixed(1) : '—'}%`,
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
          hits.push({ x: it.x, y: it.y, w: it.s, h: it.s, block: i, unit: j, text: `${name} · unit ${j + 1}\nz = ${z[j].toFixed(3)} · a = ${out[j].toFixed(3)}` });
        });
      }
    });
  };

  // ── Points: every unit over the plane ──
  const ev = new PointEvaluator();
  let tiles: Tiles | null = null;
  /** Cost of a full-resolution tile pass, in ms. */
  let tileMs = 0;

  const evaluateTiles = (d: Data) => {
    const dims = d.points!.dims;
    const r = pointDomain(d);
    const axes: [number, number] = dims === 2 ? [0, 1] : sliceAxes(slice.axis);
    const fixedAt = [0, 0, 0];
    if (dims === 3) fixedAt[slice.axis] = slice.pos;
    const t0 = performance.now();
    // A network that takes long per point gets coarser maps while it trains.
    const res = store.running && tileMs > 12 ? TILE_RES_FAST : TILE_RES;
    const coords = gridCoords(dims, res, r, axes, fixedAt);
    ev.sync(store.net);
    const out = ev.evaluate(coords, dims, store.features, { activations: true });
    tiles = { res, r, dims, axes, feats: featurize(coords, dims, store.features), acts: out.acts!, probs: out.probs };
    const ms = performance.now() - t0;
    tileMs = (ms * TILE_RES * TILE_RES) / (res * res);
    live.gap = Math.max(100, 4 * ms);
    canvas.dataset.weights = String(store.weightsRev);
  };

  let tileImg: ImageData | null = null;
  let tileCanvas: HTMLCanvasElement | null = null;
  /** Paints a probability map from the surface toward one class colour. */
  const drawProbTile = (ctx: CanvasRenderingContext2D, probs: Float32Array, K: number, k: number, it: Item, R: number) => {
    const p = palette().rgb;
    const n = R * R;
    tileCanvas ??= document.createElement('canvas');
    tileCanvas.width = R;
    tileCanvas.height = R;
    const tctx = tileCanvas.getContext('2d')!;
    if (!tileImg || tileImg.width !== R) tileImg = tctx.createImageData(R, R);
    const img = tileImg.data;
    const c = p.cat[k % 10];
    for (let i = 0; i < n; i++) {
      const t = 0.85 * probs[i * K + k];
      img[4 * i] = p.surface[0] + (c[0] - p.surface[0]) * t;
      img[4 * i + 1] = p.surface[1] + (c[1] - p.surface[1]) * t;
      img[4 * i + 2] = p.surface[2] + (c[2] - p.surface[2]) * t;
      img[4 * i + 3] = 255;
    }
    tctx.putImageData(tileImg, 0, 0);
    ctx.save();
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(tileCanvas, 0, 0, R, R, it.x, it.y, it.s, it.s);
    ctx.restore();
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
    if (!fits || !tiles) {
      fitCanvas(canvas, box.clientWidth || 400, 200);
      return;
    }
    const T = tiles;
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
    const planeName = T.dims === 2 ? 'the input plane' : `the slice ${axisName(slice.axis)} = ${fixed(slice.pos, 2)}`;
    canvas.setAttribute('aria-label', `Network diagram: ${nTiles} small maps, one per input feature, hidden unit and class, each showing its value over ${planeName}; lines show the weights between them.`);

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

    // Connections: one curve per weight, width and colour by sign and size.
    for (let i = 0; i < blocks.length; i++) {
      const b = blocks[i];
      if (b.kind !== 'dense') continue;
      const src = layouts[i];
      const dst = layouts[i + 1];
      const n = b.inSize;
      let m = 0;
      for (let q = 0; q < b.W.length; q++) m = Math.max(m, Math.abs(b.W[q]));
      if (m === 0) continue;
      const idx: number[] = [];
      for (let q = 0; q < b.W.length; q++) idx.push(q);
      idx.sort((a, c) => Math.abs(b.W[a]) - Math.abs(b.W[c]));
      const keep = idx.slice(Math.max(0, idx.length - 2500));
      for (const q of keep) {
        const j = Math.floor(q / n);
        const c = q % n;
        const s = src[c];
        const t = dst[j];
        if (!s || !t) continue;
        const w = b.W[q];
        const a = Math.abs(w) / m;
        const x1 = s.x + s.s + 1;
        const y1 = s.y + s.s / 2;
        const x2 = t.x - 1;
        const y2 = t.y + t.s / 2;
        const mx = (x1 + x2) / 2;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.bezierCurveTo(mx, y1, mx, y2, x2, y2);
        ctx.strokeStyle = w >= 0 ? p.accent : p.neg;
        ctx.globalAlpha = 0.18 + 0.7 * a;
        ctx.lineWidth = 0.4 + 3.4 * a;
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }

    // Headers
    drawHeader(ctx, centers[0], 'Features', [`${F} input${F === 1 ? '' : 's'}`, T.dims === 2 ? `over ${axisName(0)}, ${axisName(1)}` : `${axisName(slice.axis)} = ${fixed(slice.pos, 2)}`], false);
    blocks.forEach((_b, i) => {
      const isOut = i === blocks.length - 1;
      const spec = isOut ? null : store.spec[i];
      drawHeader(ctx, centers[i + 1] + (isOut ? 20 : 0), layerName(spec, i), headerLines(spec, info), store.selected === i);
    });

    const R = T.res;
    const n = R * R;
    const r = T.r;
    const pc = probe?.coords;
    const marker = (it: Item) => {
      if (!pc || pc.length !== T.dims || it.s < 18) return;
      const u = pc[T.axes[0]];
      const v = pc[T.axes[1]];
      const on = T.dims === 2 || Math.abs(pc[slice.axis] - slice.pos) <= 0.2 * r;
      const x = it.x + ((u + r) / (2 * r)) * it.s;
      const y = it.y + ((r - v) / (2 * r)) * it.s;
      if (x < it.x || x > it.x + it.s || y < it.y || y > it.y + it.s) return;
      const rad = Math.max(1.8, it.s / 16);
      ctx.beginPath();
      ctx.arc(x, y, rad, 0, 2 * Math.PI);
      ctx.lineWidth = 3;
      ctx.strokeStyle = p.surface;
      ctx.stroke();
      ctx.lineWidth = 1.6;
      ctx.strokeStyle = p.accent;
      ctx.stroke();
      if (on) {
        ctx.fillStyle = p.accent;
        ctx.fill();
      }
    };
    const frame = (it: Item, sel: boolean) => {
      ctx.strokeStyle = sel ? p.accent : p.hair;
      ctx.lineWidth = sel ? 2 : 1;
      ctx.strokeRect(it.x - 0.5, it.y - 0.5, it.s + 1, it.s + 1);
    };
    const value = (it: Item, text: string, strong = false) => {
      if (!it.lab) return;
      ctx.font = `${strong ? 600 : 400} 10px ${MONO}`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillStyle = strong ? p.accent : p.muted;
      ctx.fillText(text, it.x + it.s / 2, it.y + it.s + 2);
    };
    const planeTip = T.dims === 2 ? 'Map: its value over the plane' : `Map: its value over the slice ${axisName(slice.axis)} = ${fixed(slice.pos, 2)}`;

    // Feature tiles
    layouts[0].forEach((it, f) => {
      const vals = new Float32Array(n);
      for (let q = 0; q < n; q++) vals[q] = T.feats[q * F + f];
      const signed = hasNegative(vals);
      drawMap(ctx, vals, 0, R, R, it.x, it.y, it.s, it.s, signed, maxAbs(vals) || 1);
      frame(it, false);
      marker(it);
      ctx.font = `600 11.5px ${SANS}`;
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = p.ink;
      ctx.fillText(defs[f].label, it.x - 6, it.y + it.s / 2);
      const xv = pfits ? probe!.x[f] : NaN;
      value(it, fixed(xv, 2));
      hits.push({ x: it.x, y: it.y, w: it.s, h: it.s, block: -1, unit: f, text: `Feature ${defs[f].label}: ${defs[f].title}\nFor the current input: ${fixed(xv, 3)}\n${planeTip}` });
    });

    // Hidden units and output
    blocks.forEach((b, i) => {
      const items = layouts[i + 1];
      const isOut = i === blocks.length - 1;
      const name = layerName(isOut ? null : store.spec[i], i);
      const A = T.acts[i];
      const units = b.out.length;
      if (!isOut) {
        const signed = hasNegative(A);
        const max = maxAbs(A) || 1;
        const vals = new Float32Array(n);
        items.forEach((it, j) => {
          for (let q = 0; q < n; q++) vals[q] = A[q * units + j];
          drawMap(ctx, vals, 0, R, R, it.x, it.y, it.s, it.s, signed, max);
          const sel = store.selected === i && store.selectedUnit === j;
          frame(it, sel);
          marker(it);
          value(it, pfits ? fixed(b.out[j], 2) : '');
          hits.push({ x: it.x, y: it.y, w: it.s, h: it.s, block: i, unit: j, text: `${name} · unit ${j + 1}\nFor the current input: z = ${pfits ? fixed(b.z[j], 3) : '—'} · a = ${pfits ? fixed(b.out[j], 3) : '—'}\n${planeTip}` });
        });
        return;
      }
      const probs = net.probs;
      let best = 0;
      for (let k = 1; k < K; k++) if (probs[k] > probs[best]) best = k;
      items.forEach((it, k) => {
        const shift = 20;
        const t = { x: it.x + shift, y: it.y, s: it.s, lab: it.lab };
        drawProbTile(ctx, T.probs, K, k, t, R);
        const top = pfits && k === best;
        frame(t, store.selected === i && store.selectedUnit === k);
        marker(t);
        value(t, pfits ? `${(100 * probs[k]).toFixed(0)}%` : '', top);
        const bx = t.x + t.s + 7;
        const bw = Math.max(24, W - bx - 10);
        ctx.font = `${top ? 700 : 500} 11px ${SANS}`;
        ctx.textAlign = 'left';
        ctx.textBaseline = 'alphabetic';
        ctx.fillStyle = top ? p.accent : p.ink;
        ctx.fillText(info.classes[k], bx, t.y + Math.min(t.s / 2, 14), bw);
        const by = t.y + Math.min(t.s / 2, 14) + 5;
        ctx.fillStyle = p.hair;
        ctx.fillRect(bx, by, bw, 5);
        ctx.fillStyle = top ? p.accent : p.cat[k % 10];
        ctx.fillRect(bx, by, bw * (pfits ? probs[k] : 0), 5);
        if (probe?.label === k) {
          ctx.fillStyle = p.ink;
          ctx.font = `400 10px ${MONO}`;
          ctx.fillText('true class', bx, by + 16, bw);
        }
        hits.push({
          x: t.x, y: t.y, w: W - t.x, h: t.s, block: i, unit: k,
          text: `Output · ${info.classes[k]}${probe?.label === k ? ' (true class)' : ''}\nFor the current input: logit ${pfits ? fixed(b.z[k], 3) : '—'} · p = ${pfits ? (100 * probs[k]).toFixed(1) : '—'}%\nMap: probability of ${info.classes[k]} over ${T.dims === 2 ? 'the plane' : 'the slice'}`,
        });
      });
    });
  };

  const syncNotes = () => {
    const info = store.info;
    const pts = info.kind === 'points';
    planeNote.hidden = !pts;
    if (pts) {
      planeNote.textContent =
        info.dims === 3
          ? `Each tile maps one feature, unit or class over the slice ${axisName(slice.axis)} = ${fixed(slice.pos, 2)} of the cube (move it in section 03). The red dot is the current input; the number under a tile is its value there.`
          : `Each tile maps one feature, unit or class over the whole plane (${axisName(0)} across, ${axisName(1)} up). The red dot is the current input; the number under a tile is its value there.`;
      legendNote.textContent = 'Lines are weights: thicker means larger.';
      wLabel.textContent = 'Weight or value';
      aLabel.textContent = 'Value ≥ 0';
    } else {
      legendNote.textContent = 'Lines show summed weights between units, maps and channels.';
      wLabel.textContent = 'Weight';
      aLabel.textContent = 'Activation';
    }
  };

  const render = (evaluate: boolean) => {
    if (store.info.kind === 'points') {
      const d = store.data;
      if (netFits() && d?.points && (evaluate || !tiles)) evaluateTiles(d);
      renderPoints();
    } else renderImages();
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
  for (const ev2 of ['model', 'weights'] as const) store.on(ev2, again);
  store.on('data', () => {
    tiles = null;
    buildStrip();
    again();
  });
  for (const ev2 of ['probe', 'select', 'highlight'] as const) store.on(ev2, redraw);
  store.on('probe', syncStrip);
  store.on('dataset', () => {
    tiles = null;
    syncNotes();
    buildStrip();
    redraw();
  });
  onSlice(() => {
    syncNotes();
    if (store.info.kind === 'points' && store.info.dims === 3) again();
  });
  syncNotes();
  again();
}
