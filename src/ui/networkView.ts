import { select, setProbe, testProbe } from '../actions';
import { sampleInput } from '../data/datasets';
import type { Block } from '../nn/network';
import { store } from '../store';
import { layerName } from './builder';
import { ACTIVATIONS, type LayerSpec } from '../nn/types';
import { $, clear, h } from './dom';
import { drawMap, fitCanvas, hasNegative, maxAbs, paintThumb } from './draw';
import { css, diverging, onThemeChange, palette, sequential } from './theme';
import { hideTip, showTip } from './tip';

interface Item {
  x: number;
  y: number;
  s: number;
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
function headerLines(spec: LayerSpec | null): [string, string] {
  if (!spec) return ['10 digits', 'softmax'];
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
  const legend = h(
    'div',
    { class: 'legend' },
    h('span', { class: 'legend-item' }, 'Weight', h('span', { class: 'mono' }, '−'), ramp(true), h('span', { class: 'mono' }, '+')),
    h('span', { class: 'legend-item' }, 'Activation', h('span', { class: 'mono' }, '0'), ramp(false), h('span', { class: 'mono' }, 'max')),
    h('span', { class: 'legend-item' }, 'Lines show summed weights between units, maps and channels.'),
  );
  root.append(h('div', { class: 'probe-row' }, h('span', { class: 'label' }, 'Input'), strip, randomBtn, caption), box, legend);

  const thumbs = new Map<string, HTMLButtonElement>();
  const addThumb = (key: string, x: Float32Array, label: string, title: string, onPick: () => void) => {
    const c = document.createElement('canvas');
    paintThumb(c, x, 28, 28, 34);
    const b = h('button', { type: 'button', class: 'thumb', title, 'aria-pressed': 'false' }, c, h('span', { class: 'thumb-label' }, label)) as HTMLButtonElement;
    b.addEventListener('click', onPick);
    thumbs.set(key, b);
    strip.append(b);
  };

  const buildStrip = () => {
    clear(strip);
    thumbs.clear();
    const d = store.data;
    if (!d) return;
    for (let digit = 0; digit < 10; digit++) {
      const i = d.testY.indexOf(digit);
      const x = sampleInput(d, 'test', i);
      addThumb(`test:${i}`, x, String(digit), `Test digit #${i} (a ${digit})`, () =>
        setProbe({ x, label: digit, caption: `Test digit #${i} · label ${digit}`, key: `test:${i}` }),
      );
    }
    randomBtn.disabled = false;
    syncStrip();
  };
  randomBtn.addEventListener('click', () => {
    const d = store.data!;
    const i = Math.floor(Math.random() * d.testY.length);
    setProbe(testProbe(d, i));
  });

  const syncStrip = () => {
    const key = store.probe?.key;
    for (const [k, b] of thumbs) b.setAttribute('aria-pressed', String(k === key));
    caption.textContent = store.probe?.caption ?? '';
  };

  let hits: Hit[] = [];
  let queued = false;
  const schedule = () => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => {
      queued = false;
      render();
    });
  };

  const render = () => {
    const p = palette();
    const net = store.net;
    const blocks = net.blocks;
    const probe = store.probe;
    if (probe) net.forward(probe.x);
    const cols = blocks.length + 1;
    const avail = box.clientWidth || 600;
    const extra = 56;
    const W = Math.max(avail, cols * 88 + extra);
    const H = Math.round(Math.min(680, Math.max(420, W * 0.72)));
    const ctx = fitCanvas(canvas, W, H);
    hits = [];

    const colW = (W - extra) / cols;
    const headerH = 60;
    const top = headerH + 10;
    const bottom = H - 6;
    const availH = bottom - top;
    const centers = Array.from({ length: cols }, (_, i) => colW * (i + 0.5));
    centers[cols - 1] -= 6;

    // Layout per column
    const layouts: Item[][] = [];
    layouts.push(fitItems(1, centers[0], top, colW * 0.78, availH, 120, 0));
    blocks.forEach((b, i) => {
      const cx = centers[i + 1];
      if (b.kind === 'conv') layouts.push(fitItems(b.spec.filters, cx, top, colW * 0.74, availH, 76, 6));
      else if (i === blocks.length - 1) layouts.push(fitItems(10, cx - 10, top, 26, availH, 26, 6));
      else layouts.push(fitItems(b.spec.units, cx, top, colW * 0.5, availH, 22, 4));
    });

    // Highlight band (backprop walkthrough) and selection
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

    // Headers
    const header = (col: number, name: string, lines: [string, string], selected: boolean) => {
      const cx = col === cols - 1 ? centers[col] + 12 : centers[col];
      ctx.textAlign = 'center';
      ctx.textBaseline = 'alphabetic';
      ctx.fillStyle = selected ? p.accent : p.ink;
      ctx.font = '800 13px Archivo, "Helvetica Neue", Arial, sans-serif';
      ctx.fillText(name, cx, 16);
      ctx.fillStyle = p.muted;
      ctx.font = '400 10.5px "IBM Plex Mono", ui-monospace, monospace';
      ctx.fillText(lines[0], cx, 31);
      ctx.fillText(lines[1], cx, 44);
      if (selected) {
        ctx.fillStyle = p.accent;
        ctx.fillRect(cx - 18, 51, 36, 3);
      }
    };
    header(0, 'Input', ['28×28', 'grey'], false);
    blocks.forEach((_b, i) => {
      const spec = i === blocks.length - 1 ? null : store.spec[i];
      header(i + 1, layerName(spec, i), headerLines(spec), store.selected === i);
    });

    // Input
    const inItem = layouts[0][0];
    if (probe) drawMap(ctx, probe.x, 0, 28, 28, inItem.x, inItem.y, inItem.s, inItem.s, false, 1);
    ctx.strokeStyle = p.ink;
    ctx.lineWidth = 1;
    ctx.strokeRect(inItem.x - 0.5, inItem.y - 0.5, inItem.s + 1, inItem.s + 1);
    hits.push({ x: inItem.x, y: inItem.y, w: inItem.s, h: inItem.s, block: -1, unit: null, text: `Input · 28×28 pixels\n${probe?.caption ?? ''}` });

    // Blocks
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
        for (let k = 1; k < 10; k++) if (probs[k] > probs[best]) best = k;
        const barX = items[0].x + items[0].s + 8;
        const barW = Math.max(20, W - barX - 34);
        items.forEach((it, k) => {
          ctx.fillStyle = css(sequential(probs[k]));
          ctx.fillRect(it.x, it.y, it.s, it.s);
          ctx.strokeStyle = k === best ? p.accent : p.hair;
          ctx.lineWidth = k === best ? 2 : 1;
          ctx.strokeRect(it.x - 0.5, it.y - 0.5, it.s + 1, it.s + 1);
          ctx.fillStyle = k === best ? p.accent : p.ink;
          ctx.font = `${k === best ? 800 : 500} 12px "IBM Plex Mono", ui-monospace, monospace`;
          ctx.textAlign = 'right';
          ctx.textBaseline = 'middle';
          ctx.fillText(String(k), it.x - 6, it.y + it.s / 2);
          ctx.fillStyle = p.hair;
          ctx.fillRect(barX, it.y + it.s / 2 - 3, barW, 6);
          ctx.fillStyle = k === best ? p.accent : p.ink;
          ctx.fillRect(barX, it.y + it.s / 2 - 3, barW * probs[k], 6);
          if (probe?.label === k) {
            ctx.fillStyle = p.ink;
            ctx.beginPath();
            ctx.moveTo(barX + barW + 4, it.y + it.s / 2);
            ctx.lineTo(barX + barW + 10, it.y + it.s / 2 - 4);
            ctx.lineTo(barX + barW + 10, it.y + it.s / 2 + 4);
            ctx.fill();
          }
          hits.push({
            x: it.x - 16, y: it.y, w: barX + barW - it.x + 26, h: it.s, block: i, unit: k,
            text: `Output · digit ${k}${probe?.label === k ? ' (true label)' : ''}\nlogit ${b.z[k].toFixed(3)} · p = ${(probs[k] * 100).toFixed(1)}%`,
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

  new ResizeObserver(schedule).observe(box);
  onThemeChange(schedule);
  for (const ev of ['model', 'weights', 'probe', 'select', 'highlight'] as const) store.on(ev, schedule);
  store.on('probe', syncStrip);
  store.on('data', buildStrip);
  schedule();
}
