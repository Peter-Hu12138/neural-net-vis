import './attributionView.css';
import { DEFAULT_IG_STEPS, DEFAULT_OCCLUSION, relativeGap, type AttributionParams, type AttributionResult } from '../analysis/attribution';
import { analysis, isSuperseded } from '../analysis/client';
import { argmax } from '../nn/network';
import { store, type Probe } from '../store';
import { $, h } from './dom';
import { fitCanvas, maxAbs } from './draw';
import { syncedSection } from './snapshot';
import { css, diverging, onThemeChange, palette, sequential, type RGB } from './theme';
import { hideTip, showTip } from './tip';

/**
 * Section 10, "why this prediction": for the current input and one target digit, four maps of
 * which pixels drive that digit's score, beside the input itself.
 */

type PanelKey = 'input' | 'saliency' | 'gradInput' | 'integrated' | 'occlusion';
type Ramp = 'unit' | 'sequential' | 'diverging';

interface PanelDef {
  key: PanelKey;
  title: string;
  /** Short symbol after the title. */
  symbol?: string;
  ramp: Ramp;
  /** Draw the digit faintly under the map. */
  underlay: boolean;
  hint: (t: number, r: AttributionResult | null) => string;
  /** Tooltip line for one pixel's value. */
  value: (t: number, v: number) => string;
  aria: (t: number, max: number) => string;
}

const SIDE = 28;
const PIXELS = SIDE * SIDE;
const DEBOUNCE_MS = 150;
/** Live drawing still refreshes this often, even while the pen keeps moving. */
const MAX_WAIT_MS = 600;
const UNDERLAY = 0.16;
const MINUS = '−';

/** Three significant digits; a true minus sign; '+' on request. */
function sig(v: number, plus = false): string {
  if (!Number.isFinite(v)) return '–';
  if (v === 0) return '0';
  const a = Math.abs(v);
  const body = a >= 1000 ? a.toFixed(0) : a >= 0.001 ? String(Number(a.toPrecision(3))) : a.toExponential(1);
  return (v < 0 ? MINUS : plus ? '+' : '') + body;
}

const SUB = '₀₁₂₃₄₅₆₇₈₉';

const fixed2 = (v: number) => (Math.abs(v) < 0.005 ? '0.00' : (v < 0 ? MINUS : '') + Math.abs(v).toFixed(2));

/** "4% apart", "0.3% apart", "a match" */
function gapText(igSum: number, igExpected: number): string {
  const rel = relativeGap(igSum, igExpected);
  if (rel === 0) return 'a match';
  if (!Number.isFinite(rel)) return 'not comparable';
  const p = rel * 100;
  return `${p < 0.05 ? '<0.1' : p < 10 ? p.toFixed(1) : p.toFixed(0)}% apart`;
}

const PANELS: PanelDef[] = [
  {
    key: 'input',
    title: 'Input',
    ramp: 'unit',
    underlay: false,
    hint: () => 'The 28×28 image the network sees, from 0 (blank) to 1 (full ink).',
    value: (_t, v) => `Pixel value ${v.toFixed(2)}`,
    aria: () => 'The input image, 28 by 28 pixels',
  },
  {
    key: 'saliency',
    title: 'Saliency',
    symbol: '|∂z/∂x|',
    ramp: 'sequential',
    underlay: false,
    hint: (t) => `How strongly the score for ${t} reacts to a small change in each pixel, in either direction.`,
    value: (t, v) => `Saliency |∂z${SUB[t]}/∂x| ${sig(v)}`,
    aria: (t, m) => `Saliency for digit ${t}: darker pixels change its score most. Largest value ${sig(m)}.`,
  },
  {
    key: 'gradInput',
    title: 'Gradient × input',
    ramp: 'diverging',
    underlay: true,
    hint: (t) => `The gradient times each pixel's ink: red ink raises the score for ${t}, blue ink lowers it.`,
    value: (_t, v) => `Gradient × input ${sig(v, true)}`,
    aria: (t, m) => `Gradient times input for digit ${t}: red pixels raise its score, blue lower it. Values up to ±${sig(m)}.`,
  },
  {
    key: 'integrated',
    title: 'Integrated gradients',
    ramp: 'diverging',
    underlay: true,
    hint: (t, r) =>
      `Gradient × input averaged while the image fades in from blank (${r?.igSteps ?? DEFAULT_IG_STEPS} steps), so the pixels share out the whole change in the score for ${t}.`,
    value: (_t, v) => `Integrated gradient ${sig(v, true)}`,
    aria: (t, m) => `Integrated gradients for digit ${t}: red pixels raise its score, blue lower it. Values up to ±${sig(m)}.`,
  },
  {
    key: 'occlusion',
    title: 'Occlusion',
    ramp: 'diverging',
    underlay: true,
    hint: (t, r) => {
      const s = r?.occlusionSize ?? DEFAULT_OCCLUSION.size;
      return `Erases a ${s}×${s} patch at a time: red where that lowers the probability of ${t}, blue where it raises it.`;
    },
    value: (t, v) => `Mean drop in p(${t}) ${sig(v, true)}`,
    aria: (t, m) => `Occlusion for digit ${t}: red areas lower its probability when erased, blue raise it. Values up to ±${sig(m)}.`,
  },
];

interface Panel {
  def: PanelDef;
  fig: HTMLElement;
  canvas: HTMLCanvasElement;
  hint: HTMLElement;
  lo: HTMLElement;
  hi: HTMLElement;
  ramp: HTMLCanvasElement;
  check?: HTMLElement;
}

interface Request {
  probe: Probe;
  x: Float32Array;
  probs: Float32Array;
  pred: number;
  target: number;
}

interface Shown {
  r: AttributionResult;
  x: Float32Array;
  probe: Probe;
}

const rgb: RGB = [0, 0, 0];
let scratch: HTMLCanvasElement | null = null;

export function mountAttribution(): void {
  const root = $('attr-root');
  root.classList.add('attr');
  root.dataset.state = 'idle';

  // ── State ──
  /** The digit the learner chose; null follows the prediction. Cleared when the input changes. */
  let pinned: number | null = null;
  let current: Request | null = null;
  let shown: Shown | null = null;
  let error: string | null = null;
  let hover: number | null = null;
  let runId = 0;
  let busy = false;
  let queued = false;
  let dirty = false;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let waitingSince = 0;
  let px = 140;

  // ── Elements ──
  const fill = h('span', { style: { width: '0%' } });
  const bar = h('div', { class: 'progress', role: 'progressbar', 'aria-label': 'Attribution progress', 'aria-valuemin': '0', 'aria-valuemax': '100', 'aria-valuenow': '0' }, fill);
  const busyLine = h('span', { class: 'attr-busy', 'aria-hidden': 'true' }, bar, h('span', { class: 'hint' }, 'Computing…'));

  const chipBtns: HTMLButtonElement[] = [];
  const marks: HTMLElement[] = [];
  const chipRow = h('div', { class: 'attr-chips', role: 'group', 'aria-labelledby': 'attr-target-label' });
  for (let d = 0; d < 10; d++) {
    const b = h('button', { type: 'button', class: 'chip', id: `attr-target-${d}`, 'aria-pressed': 'false' }, String(d)) as HTMLButtonElement;
    b.addEventListener('click', () => pick(d));
    const m = h('span', { class: 'attr-mark', 'aria-hidden': 'true' });
    chipBtns.push(b);
    marks.push(m);
    chipRow.append(h('div', { class: 'attr-chip-col' }, b, m));
  }
  const predLine = h('p', { class: 'attr-pred', id: 'attr-pred', 'aria-live': 'polite' }, 'Waiting for an input…');
  const inputCaption = h('p', { class: 'attr-input-caption' });
  const notice = h('p', { class: 'notice', id: 'attr-notice', hidden: true });

  const side = h(
    'div',
    { class: 'attr-side' },
    h('div', { class: 'attr-target' }, h('span', { class: 'label', id: 'attr-target-label' }, 'Explain digit'), chipRow),
    predLine,
    inputCaption,
    notice,
  );
  const notes = h(
    'div',
    { class: 'attr-notes' },
    h('p', { class: 'hint' }, 'Pick a digit to see which pixels argue for it. Your pick holds until the input changes; change the input in 02 Network, 03 Draw or 07 Data.'),
    h('p', { class: 'hint' }, 'The gradient maps explain the digit’s score before softmax, its logit z. Occlusion explains its probability p. Signed maps show the digit faintly underneath.'),
  );

  const panels: Panel[] = PANELS.map((def) => {
    const canvas = h('canvas', { role: 'img', id: `attr-map-${def.key}`, 'aria-label': def.aria(0, 0) }) as HTMLCanvasElement;
    const lo = h('span');
    const hi = h('span');
    const ramp = h('canvas', { 'aria-hidden': 'true' }) as HTMLCanvasElement;
    const hint = h('p', { class: 'hint' });
    const check = def.key === 'integrated' ? h('p', { class: 'attr-check', id: 'attr-check', title: 'Completeness: integrated gradients add up to the change in the logit from a blank image to this one.' }) : undefined;
    const fig = h(
      'figure',
      { class: 'attr-panel', 'data-panel': def.key },
      h('h3', { class: 'attr-title' }, def.title, def.symbol ? ' ' : null, def.symbol ? h('span', { class: 'mono' }, def.symbol) : null),
      h('div', { class: 'attr-map' }, canvas),
      h('div', { class: 'attr-scale' }, lo, ramp, hi),
      h('div', { class: 'attr-text' }, hint, check),
    );
    const panel: Panel = { def, fig, canvas, hint, lo, hi, ramp, check };
    canvas.addEventListener('pointermove', (e) => onHover(panel, e));
    canvas.addEventListener('pointerdown', (e) => onHover(panel, e));
    canvas.addEventListener('pointerleave', () => {
      hideTip();
      setHover(null);
    });
    return panel;
  });
  const grid = h('div', { class: 'attr-panels' }, ...panels.map((p) => p.fig));

  const sync = syncedSection(root, () => start());
  sync.status.append(busyLine);
  root.append(sync.status, h('div', { class: 'attr-layout' }, side, grid, notes));

  // ── Computation ──

  /** Probe changes (live drawing included) wait a moment, and never cut off a run in progress. */
  function soon(): void {
    if (!sync.visible) {
      dirty = true;
      return;
    }
    const now = performance.now();
    if (timer) clearTimeout(timer);
    else waitingSince = now;
    timer = setTimeout(
      () => {
        timer = null;
        if (busy) queued = true;
        else start();
      },
      Math.max(0, Math.min(DEBOUNCE_MS, waitingSince + MAX_WAIT_MS - now)),
    );
  }

  function start(): void {
    if (timer) clearTimeout(timer);
    timer = null;
    dirty = false;
    queued = false;
    if (!store.data) return;
    if (!store.valid) {
      runId++;
      analysis.cancel('attribution');
      setBusy(false);
      error = null;
      renderNotice('Fix the architecture in 01 to see attributions.');
      return;
    }
    const probe = store.probe;
    if (!probe) return;
    const x = probe.x.slice();
    const probs = store.net.forward(x).slice();
    const pred = argmax(probs);
    const target = pinned ?? pred;
    current = { probe, x, probs, pred, target };
    error = null;
    renderControls();
    sync.markComputed();
    const id = ++runId;
    setBusy(true);
    setProgress(0);
    const params: AttributionParams = { x, target, igSteps: DEFAULT_IG_STEPS, occlusion: DEFAULT_OCCLUSION };
    analysis
      .run<AttributionResult>('attribution', 'attribution', params, (p) => {
        if (id === runId) setProgress(p.done / p.total);
      })
      .then((r) => {
        if (id !== runId) return;
        shown = { r, x, probe };
        renderPanels();
        renderNotice();
      })
      .catch((e: unknown) => {
        if (isSuperseded(e) || id !== runId) return;
        error = e instanceof Error ? e.message : String(e);
        renderNotice();
      })
      .finally(() => {
        if (id !== runId) return;
        setBusy(false);
        if (queued) start();
      });
  }

  function pick(d: number): void {
    pinned = d;
    if (current) {
      current.target = d;
      renderControls();
    }
    start();
  }

  function setBusy(on: boolean): void {
    busy = on;
    busyLine.classList.toggle('is-on', on);
    busyLine.setAttribute('aria-hidden', String(!on));
    root.dataset.state = on ? 'busy' : error ? 'error' : shown ? 'done' : 'idle';
  }

  function setProgress(f: number): void {
    const v = Math.max(0, Math.min(1, f));
    fill.style.width = `${(v * 100).toFixed(1)}%`;
    bar.setAttribute('aria-valuenow', String(Math.round(v * 100)));
  }

  // ── Rendering ──

  function renderControls(): void {
    const c = current;
    const label = c?.probe.label ?? null;
    for (let d = 0; d < 10; d++) {
      const b = chipBtns[d];
      const isPred = !!c && c.pred === d;
      const isTrue = label === d;
      b.setAttribute('aria-pressed', String(!!c && c.target === d));
      const notes = [isPred ? 'predicted' : '', isTrue ? 'true label' : ''].filter(Boolean).join(', ');
      b.setAttribute('aria-label', `Explain digit ${d}${notes ? ` (${notes})` : ''}`);
      const m = marks[d];
      m.replaceChildren();
      if (isPred) m.append(h('span', { class: 'attr-tag is-pred' }, 'pred'));
      if (isTrue) m.append(h('span', { class: 'attr-tag is-true' }, 'true'));
    }
    if (!c) {
      predLine.textContent = 'Waiting for an input…';
      inputCaption.textContent = '';
      return;
    }
    const pTarget = c.probs[c.target];
    predLine.replaceChildren(
      'Predicted ',
      h('b', null, String(c.pred)),
      ' at ',
      h('b', null, pctText(c.probs[c.pred])),
      '; explaining digit ',
      h('b', null, String(c.target)),
      c.target !== c.pred ? ` (${pctText(pTarget)})` : '',
    );
    inputCaption.textContent = `Input: ${c.probe.caption}`;
    root.dataset.target = String(c.target);
    root.dataset.pred = String(c.pred);
  }

  function renderNotice(text?: string): void {
    let msg = text ?? '';
    if (!msg && error) msg = `Attribution failed: ${error}`;
    if (!msg && shown && !hasInk(shown.x))
      msg = 'The input is blank, so gradient × input, integrated gradients and occlusion are all zero. Draw a digit or pick one in 07 Data.';
    notice.textContent = msg;
    notice.hidden = !msg;
    if (!busy) root.dataset.state = error ? 'error' : shown ? 'done' : 'idle';
  }

  const values = (key: PanelKey, s: Shown): Float32Array => (key === 'input' ? s.x : s.r[key]);

  function renderPanels(): void {
    const s = shown;
    const t = s?.r.target ?? current?.target ?? 0;
    for (const p of panels) {
      const def = p.def;
      p.hint.textContent = def.hint(t, s?.r ?? null);
      const data = s ? values(def.key, s) : null;
      const max = data ? (def.ramp === 'unit' ? 1 : maxAbs(data)) : 0;
      paintMap(p, data, max, s?.x ?? null);
      if (def.ramp === 'unit') {
        p.lo.textContent = '0';
        p.hi.textContent = '1';
      } else if (def.ramp === 'sequential') {
        p.lo.textContent = '0';
        p.hi.textContent = data ? sig(max) : '–';
      } else {
        p.lo.textContent = data ? (max > 0 ? sig(-max) : '0') : '–';
        p.hi.textContent = data ? (max > 0 ? sig(max, true) : '0') : '–';
      }
      p.canvas.setAttribute('aria-label', s ? def.aria(t, max) : `${def.title}: not computed yet`);
      if (p.check) {
        if (s) {
          const r = s.r;
          // Breaks only between clauses when the column is narrow.
          p.check.replaceChildren(
            h('span', null, `Σ IG = ${fixed2(r.igSum)};`),
            ' ',
            h('span', null, `z(x) ${MINUS} z(blank) = ${fixed2(r.igExpected)}`),
            ' ',
            h('span', { class: 'attr-gap' }, `(${gapText(r.igSum, r.igExpected)})`),
          );
        } else p.check.textContent = `Σ IG = –; z(x) ${MINUS} z(blank) = –`;
      }
    }
    root.dataset.resultKey = s ? `${s.probe.key}|${s.r.target}` : '';
  }

  function paintMap(p: Panel, data: Float32Array | null, max: number, x: Float32Array | null): void {
    const pal = palette();
    const ctx = fitCanvas(p.canvas, px, px);
    scratch ??= document.createElement('canvas');
    scratch.width = SIDE;
    scratch.height = SIDE;
    const sctx = scratch.getContext('2d');
    if (!sctx) return;
    const img = sctx.createImageData(SIDE, SIDE);
    const d = img.data;
    const ink = pal.rgb.ink;
    const inv = max > 0 ? 1 / max : 0;
    for (let i = 0; i < PIXELS; i++) {
      if (!data) {
        rgb[0] = pal.rgb.surface[0];
        rgb[1] = pal.rgb.surface[1];
        rgb[2] = pal.rgb.surface[2];
      } else if (p.def.ramp === 'diverging') diverging(data[i] * inv, rgb);
      else sequential(data[i] * inv, rgb);
      if (p.def.underlay && x) {
        const a = UNDERLAY * Math.max(0, Math.min(1, x[i]));
        rgb[0] += (ink[0] - rgb[0]) * a;
        rgb[1] += (ink[1] - rgb[1]) * a;
        rgb[2] += (ink[2] - rgb[2]) * a;
      }
      d[4 * i] = rgb[0];
      d[4 * i + 1] = rgb[1];
      d[4 * i + 2] = rgb[2];
      d[4 * i + 3] = 255;
    }
    sctx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(scratch, 0, 0, SIDE, SIDE, 0, 0, px, px);
    ctx.strokeStyle = pal.hair;
    ctx.lineWidth = 1;
    ctx.strokeRect(0.5, 0.5, px - 1, px - 1);
    if (hover !== null && data) {
      const cell = px / SIDE;
      const cx = (hover % SIDE) * cell;
      const cy = Math.floor(hover / SIDE) * cell;
      ctx.strokeStyle = pal.ink;
      ctx.lineWidth = 2;
      ctx.strokeRect(cx - 1, cy - 1, cell + 2, cell + 2);
      ctx.strokeStyle = pal.surface;
      ctx.lineWidth = 1;
      ctx.strokeRect(cx + 0.5, cy + 0.5, cell - 1, cell - 1);
    }
  }

  function paintRamp(c: HTMLCanvasElement, ramp: Ramp): void {
    const w = 56;
    const hgt = 8;
    const ctx = fitCanvas(c, w, hgt);
    for (let i = 0; i < w; i++) {
      const t = i / (w - 1);
      ctx.fillStyle = css(ramp === 'diverging' ? diverging(t * 2 - 1) : sequential(t));
      ctx.fillRect(i, 0, 1, hgt);
    }
  }

  // ── Hover ──

  function onHover(p: Panel, e: PointerEvent): void {
    const s = shown;
    if (!s) return;
    const r = p.canvas.getBoundingClientRect();
    const col = Math.floor(((e.clientX - r.left) / r.width) * SIDE);
    const row = Math.floor(((e.clientY - r.top) / r.height) * SIDE);
    if (col < 0 || col >= SIDE || row < 0 || row >= SIDE) {
      hideTip();
      setHover(null);
      return;
    }
    const i = row * SIDE + col;
    const t = s.r.target;
    const lines = [`Row ${row}, column ${col}`, `Pixel value ${s.x[i].toFixed(2)}`];
    if (p.def.key !== 'input') lines.push(p.def.value(t, values(p.def.key, s)[i]));
    showTip(lines.join('\n'), e.clientX, e.clientY);
    setHover(i);
  }

  function setHover(i: number | null): void {
    if (hover === i) return;
    hover = i;
    const s = shown;
    if (!s) return;
    for (const p of panels) {
      const data = values(p.def.key, s);
      paintMap(p, data, p.def.ramp === 'unit' ? 1 : maxAbs(data), s.x);
    }
  }

  // ── Layout ──

  /** Largest whole number of screen pixels per image pixel that fits a panel, 3–7. */
  function measure(): boolean {
    const w = panels[0].fig.clientWidth;
    if (!w) return false;
    const cell = Math.max(3, Math.min(7, Math.floor(w / SIDE)));
    const next = cell * SIDE;
    if (next === px) return false;
    px = next;
    return true;
  }

  let queuedPaint = false;
  const schedule = () => {
    if (queuedPaint) return;
    queuedPaint = true;
    requestAnimationFrame(() => {
      queuedPaint = false;
      measure();
      renderPanels();
      for (const p of panels) paintRamp(p.ramp, p.def.ramp);
    });
  };
  new ResizeObserver(() => {
    if (measure()) schedule();
  }).observe(grid);

  // ── Events ──

  store.on('probe', () => {
    pinned = null;
    soon();
  });
  store.on('model', () => {
    if (!store.valid) start();
  });
  new IntersectionObserver(
    (entries) => {
      if (entries.some((e) => e.isIntersecting) && dirty) soon();
    },
    { rootMargin: '200px 0px' },
  ).observe(root);
  onThemeChange(schedule);

  measure();
  renderControls();
  renderPanels();
  for (const p of panels) paintRamp(p.ramp, p.def.ramp);
}

function pctText(p: number): string {
  const v = p * 100;
  return `${v >= 99.95 && v < 100 ? '99.9' : v.toFixed(1)}%`;
}

function hasInk(x: Float32Array): boolean {
  for (let i = 0; i < x.length; i++) if (x[i] > 0) return true;
  return false;
}
