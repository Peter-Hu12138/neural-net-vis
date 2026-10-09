import { applyWeights, setHighlight } from '../actions';
import { fixed } from '../analysis/stats';
import { noun, type DatasetInfo } from '../data/datasets';
import { featureDefs } from '../data/features';
import { derivative, derivativeFormula } from '../nn/activations';
import { Network, argmax, type Block } from '../nn/network';
import { ACTIVATIONS, type Act, type Arch, type Shape } from '../nn/types';
import { store } from '../store';
import { layerName } from './builder';
import { classWord, namesAreGlyphs, shortNames } from './charts';
import { $, clear, digitChips, h, int, selectField } from './dom';
import { drawSample, fitCanvas, hasNegative, maxAbs, type MatrixMode } from './draw';
import { matrix } from './inspector';
import { classColor, onThemeChange, palette } from './theme';
import './backprop.css';

/** A frozen copy of one block's forward and backward tensors for a single example. */
interface BT {
  index: number;
  kind: 'conv' | 'dense';
  isOut: boolean;
  name: string;
  act: Act;
  inShape: Shape;
  zShape: Shape;
  outShape: Shape;
  k: number;
  pad: number;
  pool: boolean;
  W: Float32Array;
  b: Float32Array;
  x: Float32Array;
  z: Float32Array;
  a: Float32Array;
  out: Float32Array;
  argmax: Int32Array | null;
  dOut: Float32Array;
  dA: Float32Array;
  dZ: Float32Array;
  dX: Float32Array;
  gW: Float32Array;
  gb: Float32Array;
}

/** What the dataset calls things, captured with the trace so every step speaks its language. */
interface Ctx {
  info: DatasetInfo;
  /** Number of classes. */
  K: number;
  /** Full class names, and the short ones for vector labels. */
  names: string[];
  short: string[];
  /** The class names are MNIST's digits. */
  digits: boolean;
  /** "digit" or "class". */
  what: string;
  input: 'grey' | 'colour' | 'points';
  /** Point datasets: feature ids and labels (x₁, x₁², …) and the raw coordinates. */
  featIds: string[];
  feats: string[];
  coords: Float32Array | null;
}

interface Trace {
  x: Float32Array;
  label: number;
  probs: Float32Array;
  loss: number;
  blocks: BT[];
  step: number;
  version: number;
  /** The architecture the trace was made with (the page may have moved on to another). */
  arch: Arch;
  ctx: Ctx;
}

interface Step {
  phase: 'Forward pass' | 'Loss' | 'Backward pass' | 'Update';
  title: string;
  block: number;
  dir: 'fwd' | 'back' | 'update';
  render: (el: HTMLElement) => void;
}

const MONO = '"IBM Plex Mono", ui-monospace, monospace';
/** Feature labels carry sub- and superscripts (x₁²), which need a larger size to stay legible. */
const labelFont = (labels?: string[]) => `500 ${labels?.some((l) => /[₁₂₃²]/.test(l)) ? 11.5 : 9.5}px ${MONO}`;
const actLabel = (a: Act) => ACTIVATIONS.find((x) => x.id === a)!.label;
const actFormula = (a: Act) => ACTIVATIONS.find((x) => x.id === a)!.formula;
const CHANNELS = ['red', 'green', 'blue'];
const RGB_LABELS = ['R', 'G', 'B'];
const SUB = ['₁', '₂', '₃'];

/** Number formatting for the worked examples: fixed decimals with a real minus sign, never "−0.000". */
const n = (v: number, d = 3) => fixed(v, d);
const ns = (v: number, d = 3, w = 7) => n(v, d).padStart(w);

function makeCtx(coords: Float32Array | undefined): Ctx {
  const info = store.info;
  const s = store.input;
  const pts = info.kind === 'points';
  let feats: string[] = [];
  if (pts) {
    try {
      feats = featureDefs(info.dims!, store.features).map((f) => f.label);
    } catch {
      feats = store.features.slice();
    }
  }
  return {
    info,
    K: info.classes.length,
    names: info.classes,
    short: shortNames(info),
    digits: namesAreGlyphs(info),
    what: classWord(info),
    input: pts ? 'points' : s.c === 3 ? 'colour' : 'grey',
    featIds: pts ? store.features.slice() : [],
    feats,
    coords: coords ? coords.slice() : null,
  };
}

function makeTrace(net: Network, x: Float32Array, label: number, coords?: Float32Array): Trace {
  net.zeroGrad();
  net.forward(x);
  const loss = net.backward(label, true);
  const blocks = net.blocks.map((b: Block, i): BT => {
    const isOut = i === net.blocks.length - 1;
    const conv = b.kind === 'conv';
    return {
      index: i,
      kind: b.kind,
      isOut,
      name: layerName(isOut ? null : store.spec[i], i),
      act: b.spec.act,
      inShape: b.inShape,
      zShape: conv ? b.zShape : b.outShape,
      outShape: b.outShape,
      k: conv ? b.k : 0,
      pad: conv ? b.pad : 0,
      pool: conv ? b.spec.pool : false,
      W: b.W.slice(),
      b: b.b.slice(),
      x: b.x.slice(),
      z: b.z.slice(),
      a: b.a.slice(),
      out: b.out.slice(),
      argmax: conv && b.argmax ? b.argmax.slice() : null,
      dOut: b.dOut.slice(),
      dA: b.dA.slice(),
      dZ: b.dZ.slice(),
      dX: b.dX.slice(),
      gW: b.gW.slice(),
      gb: b.gb.slice(),
    };
  });
  return { x: x.slice(), label, probs: net.probs.slice(), loss, blocks, step: store.weightsStep, version: store.version, arch: structuredClone(store.arch), ctx: makeCtx(coords) };
}

// ── Labels ────────────────────────────────────────────────────────────────

/** Class k in a vector label or table: the digit, or a short name. */
const classLab = (c: Ctx, k: number) => (c.digits ? String(k) : c.short[k]);
/** Class k in prose: "7", or "cat". */
const className = (c: Ctx, k: number) => (c.digits ? String(k) : c.names[k]);
/** The class name says more than its number ("cat", not "Class 3"). */
const hasName = (c: Ctx, k: number) => !c.digits && c.names[k].toLowerCase() !== `class ${k}`;
/** "the digit 7", "class 3, cat", "class 1". */
const classPhrase = (c: Ctx, k: number) => (c.digits ? `the digit ${k}` : hasName(c, k) ? `class ${k}, ${c.names[k]}` : `class ${k}`);
const classLabels = (c: Ctx) => Array.from({ length: c.K }, (_, k) => classLab(c, k));
/** Block 0 of a point network reads features; their labels replace x[i]. */
const readsFeatures = (c: Ctx, bt: BT) => bt.index === 0 && c.input === 'points' && c.feats.length === bt.x.length;
/** Block 0 of a colour network reads the red, green and blue planes. */
const readsColour = (c: Ctx, bt: BT) => bt.index === 0 && c.input === 'colour';

// ── Figures ───────────────────────────────────────────────────────────────

const figMode = (): MatrixMode => (store.mode === 'hinton' ? 'hinton' : store.mode === 'numbers' ? 'numbers' : 'heat');

function figure(cap: string, canvas: HTMLCanvasElement, note?: string): HTMLElement {
  const [bold, ...rest] = cap.split(' · ');
  return h(
    'figure',
    { class: 'tensor', style: { margin: '0' } },
    h('figcaption', { class: 'tensor-cap' }, h('b', null, bold), rest.length ? ` · ${rest.join(' · ')}` : '', note ? h('div', null, note) : null),
    h('div', { class: 'tensor-body' }, canvas),
  );
}

interface Mark {
  /** Channel to mark; −1 marks the same window in every channel (a colour patch). */
  c: number;
  y: number;
  x: number;
  h?: number;
  w?: number;
}

/** C maps of H×W. Numbers only when the maps are small enough to read. */
function maps(cap: string, data: Float32Array, s: Shape, o: { signed?: boolean; max?: number; mark?: Mark; px?: number; maxRow?: number; chans?: string[]; label?: string } = {}): HTMLElement {
  const { c: C, h: H, w: W } = s;
  const signed = o.signed ?? hasNegative(data);
  const max = o.max ?? (maxAbs(data) || 1);
  const m = figMode();
  const numbers = m === 'numbers' && H * W <= 64 && C <= 16;
  const mode: MatrixMode = numbers ? 'numbers' : m === 'hinton' && H * W <= 196 ? 'hinton' : 'heat';
  const cw = numbers ? 40 : Math.max(2, Math.min(14, Math.round((o.px ?? 84) / H)));
  const ch = numbers ? 20 : cw;
  const mw = W * cw;
  const mh = H * ch;
  const gap = 10;
  const lab = C > 1 ? 13 : 0;
  const perRow = Math.max(1, Math.min(C, o.maxRow ?? 8, Math.floor((640 + gap) / (mw + gap))));
  const rows = Math.ceil(C / perRow);
  const canvas = document.createElement('canvas');
  canvas.setAttribute('role', 'img');
  canvas.setAttribute('aria-label', o.label ?? `${cap.split(' · ')[0]}: ${C > 1 ? `${C} maps of ` : ''}${H}×${W} values`);
  const ctx = fitCanvas(canvas, perRow * (mw + gap) - gap + 2, rows * (mh + gap + lab) - gap + 2);
  const p = palette();
  for (let c = 0; c < C; c++) {
    const x = 1 + (c % perRow) * (mw + gap);
    const y = 1 + Math.floor(c / perRow) * (mh + gap + lab);
    const marked = !!o.mark && (o.mark.c === c || o.mark.c === -1);
    if (lab) {
      ctx.font = `500 9.5px ${MONO}`;
      ctx.fillStyle = marked ? p.accent : p.muted;
      ctx.textBaseline = 'top';
      ctx.textAlign = 'left';
      ctx.fillText(o.chans?.[c] ?? String(c + 1), x, y);
    }
    matrix(ctx, data, c * H * W, H, W, x, y + lab, cw, ch, mode, max, signed);
    ctx.strokeStyle = p.hair;
    ctx.lineWidth = 1;
    ctx.strokeRect(x - 0.5, y + lab - 0.5, mw + 1, mh + 1);
    if (o.mark && marked) {
      const mk = o.mark;
      const y0 = Math.max(0, mk.y);
      const x0 = Math.max(0, mk.x);
      const y1 = Math.min(H, mk.y + (mk.h ?? 1));
      const x1 = Math.min(W, mk.x + (mk.w ?? 1));
      ctx.strokeStyle = p.accent;
      ctx.lineWidth = 2;
      ctx.strokeRect(x + x0 * cw - 1, y + lab + y0 * ch - 1, (x1 - x0) * cw + 2, (y1 - y0) * ch + 2);
    }
  }
  const note = !numbers && m === 'numbers' ? 'Too many values to print; shown as a heat map.' : undefined;
  return figure(cap, canvas, note);
}

/** The input image itself, in its own colours (a CIFAR photo) or as ink on paper (grey). */
function picture(cap: string, x: Float32Array, s: Shape, px: number): HTMLElement {
  const canvas = document.createElement('canvas');
  canvas.setAttribute('role', 'img');
  canvas.setAttribute('aria-label', `${cap.split(' · ')[0]}: the input image, ${s.h}×${s.w} pixels${s.c === 3 ? ' in colour' : ''}`);
  const ctx = fitCanvas(canvas, px + 2, Math.round((px * s.h) / s.w) + 2);
  drawSample(ctx, x, s, 1, 1, px, Math.round((px * s.h) / s.w));
  ctx.strokeStyle = palette().hair;
  ctx.lineWidth = 1;
  ctx.strokeRect(0.5, 0.5, px + 1, Math.round((px * s.h) / s.w) + 1);
  canvas.classList.add('bp-pixelated');
  return figure(cap, canvas);
}

/** A vector. Short vectors (up to ten values) are always printed; longer ones follow the view mode. */
function vec(cap: string, data: Float32Array, o: { labels?: string[]; signed?: boolean; max?: number; mark?: number; mark2?: number } = {}): HTMLElement {
  const len = data.length;
  const signed = o.signed ?? hasNegative(data);
  const max = o.max ?? (maxAbs(data) || 1);
  const m = figMode();
  const numbers = len <= 10 || (m === 'numbers' && len <= 128);
  const mode: MatrixMode = numbers ? 'numbers' : m === 'hinton' ? 'hinton' : 'heat';
  const perRow = Math.min(len, numbers ? 10 : 16);
  const rows = Math.ceil(len / perRow);
  const cw = numbers ? 50 : 18;
  const ch = numbers ? 22 : 18;
  const labH = 14;
  const canvas = document.createElement('canvas');
  canvas.setAttribute('role', 'img');
  canvas.setAttribute('aria-label', `${cap.split(' · ')[0]}: ${len} values${o.labels ? ` for ${o.labels.slice(0, 12).join(', ')}${len > 12 ? ', …' : ''}` : ''}`);
  const ctx = fitCanvas(canvas, perRow * cw + 2, rows * (ch + labH + 4) + 2);
  const p = palette();
  for (let r = 0; r < rows; r++) {
    const cnt = Math.min(perRow, len - r * perRow);
    const y = 1 + r * (ch + labH + 4);
    matrix(ctx, data, r * perRow, 1, cnt, 1, y, cw, ch, mode, max, signed);
    ctx.font = labelFont(o.labels);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (let c = 0; c < cnt; c++) {
      const i = r * perRow + c;
      ctx.fillStyle = i === o.mark || i === o.mark2 ? p.accent : p.muted;
      ctx.fillText(o.labels?.[i] ?? String(i + 1), 1 + c * cw + cw / 2, y + ch + 3, cw - 2);
      if (i === o.mark || i === o.mark2) {
        ctx.strokeStyle = p.accent;
        ctx.lineWidth = 2;
        ctx.strokeRect(1 + c * cw, y, cw, ch);
      }
    }
  }
  return figure(cap, canvas);
}

/** A weight matrix (rows = units of this layer). */
function mat(cap: string, data: Float32Array, rows: number, cols: number, o: { markRow?: number; markCol?: number; rowLabels?: string[]; colLabels?: string[] } = {}): HTMLElement {
  const max = maxAbs(data) || 1;
  const m = figMode();
  const numbers = m === 'numbers' && rows * cols <= 400;
  const p = palette();
  const canvas = document.createElement('canvas');
  const measureCtx = canvas.getContext('2d')!;
  measureCtx.font = `500 9.5px ${MONO}`;
  const widest = (ls?: string[]) => (ls ? Math.ceil(ls.reduce((w, l) => Math.max(w, measureCtx.measureText(l).width), 0)) : 0);
  let cw = numbers ? 44 : Math.max(1, Math.min(18, Math.floor(640 / cols)));
  if (o.colLabels && !numbers) cw = Math.max(cw, Math.min(48, widest(o.colLabels) + 8));
  const ch = numbers ? 20 : Math.max(2, Math.min(18, Math.floor(180 / rows), Math.max(cw, 3)));
  const mode: MatrixMode = numbers ? 'numbers' : m === 'hinton' && cw >= 4 ? 'hinton' : 'heat';
  const left = Math.max(26, widest(o.rowLabels) + 8);
  const top = o.colLabels ? 16 : 0;
  canvas.setAttribute('role', 'img');
  canvas.setAttribute('aria-label', `${cap.split(' · ')[0]}: a ${rows}×${cols} matrix${o.rowLabels ? `, rows ${o.rowLabels.slice(0, 10).join(', ')}` : ''}${o.colLabels ? `, columns ${o.colLabels.join(', ')}` : ''}`);
  const ctx = fitCanvas(canvas, left + cols * cw + 2, top + rows * ch + 2);
  matrix(ctx, data, 0, rows, cols, left, top + 1, cw, ch, mode, max, true);
  ctx.font = `500 9.5px ${MONO}`;
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  const stepR = Math.max(1, Math.ceil(11 / ch));
  for (let r = 0; r < rows; r += stepR) {
    ctx.fillStyle = r === o.markRow ? p.accent : p.muted;
    ctx.fillText(o.rowLabels?.[r] ?? String(r + 1), left - 4, top + 1 + r * ch + ch / 2);
  }
  if (o.colLabels) {
    ctx.textAlign = 'center';
    ctx.font = labelFont(o.colLabels);
    for (let c = 0; c < cols; c++) {
      ctx.fillStyle = c === o.markCol ? p.accent : p.muted;
      ctx.fillText(o.colLabels[c], left + c * cw + cw / 2, 7);
    }
  }
  if (o.markRow !== undefined) {
    ctx.strokeStyle = p.accent;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(left - 0.5, top + 1 + o.markRow * ch - 0.5, cols * cw + 1, ch + 1);
    if (o.markCol !== undefined) {
      ctx.lineWidth = 2;
      ctx.strokeRect(left + o.markCol * cw - 1.5, top + 1 + o.markRow * ch - 1.5, cw + 3, ch + 3);
    }
  }
  const note = !numbers && m === 'numbers' ? 'Too many values to print; shown as a heat map.' : undefined;
  return figure(cap, canvas, note);
}

/** All kernels of a conv layer: filters as rows, input channels as columns. */
function kernels(cap: string, W: Float32Array, F: number, C: number, k: number, markF?: number, markC?: number, chans?: string[]): HTMLElement {
  const max = maxAbs(W) || 1;
  const m = figMode();
  const mode: MatrixMode = m === 'hinton' ? 'hinton' : 'heat';
  const cell = Math.max(2, Math.min(10, Math.floor(460 / (C * k))));
  const kw = k * cell;
  const gap = 4;
  const left = 22;
  const top = chans ? 14 : 0;
  const canvas = document.createElement('canvas');
  canvas.setAttribute('role', 'img');
  canvas.setAttribute('aria-label', `${cap.split(' · ')[0]}: ${F} filters by ${C} input channel${C > 1 ? 's' : ''}${chans ? ` (${chans.join(', ')})` : ''}, ${k}×${k} each`);
  const ctx = fitCanvas(canvas, left + C * (kw + gap), top + F * (kw + gap) + 2);
  const p = palette();
  ctx.font = `500 9.5px ${MONO}`;
  ctx.textBaseline = 'middle';
  if (chans) {
    ctx.textAlign = 'center';
    for (let c = 0; c < C; c++) {
      ctx.fillStyle = c === markC ? p.accent : p.muted;
      ctx.fillText(chans[c], left + c * (kw + gap) + kw / 2, 6);
    }
  }
  ctx.textAlign = 'right';
  for (let f = 0; f < F; f++) {
    const y = top + 1 + f * (kw + gap);
    ctx.fillStyle = f === markF ? p.accent : p.muted;
    ctx.fillText(String(f + 1), left - 5, y + kw / 2);
    for (let c = 0; c < C; c++) {
      const x = left + c * (kw + gap);
      matrix(ctx, W, (f * C + c) * k * k, k, k, x, y, cell, cell, mode, max, true);
      if (f === markF && (markC === undefined || c === markC)) {
        ctx.strokeStyle = p.accent;
        ctx.lineWidth = 2;
        ctx.strokeRect(x - 1, y - 1, kw + 2, kw + 2);
      }
    }
  }
  return figure(cap, canvas, `Rows: filters. Columns: input ${chans ? 'colour ' : ''}channels.`);
}

function work(lines: (string | [string, boolean])[]): HTMLElement {
  const pre = h('pre', { class: 'work' });
  for (const l of lines) {
    if (Array.isArray(l)) pre.append(h('b', null, l[0]), '\n');
    else pre.append(l, '\n');
  }
  return pre;
}

function formula(text: string, back: boolean): HTMLElement {
  return h('div', { class: `formula${back ? ' is-back' : ''}` }, text);
}

function row(...els: HTMLElement[]): HTMLElement {
  return h('div', { class: 'tensors' }, ...els);
}

/** The frozen-layer note for a step: ink rule, not the accent (frozen is a setting, not an alarm). */
function frozenNote(...parts: (string | Node)[]): HTMLElement {
  return h('p', { class: 'bp-frozen' }, h('span', { class: 'tag' }, 'Frozen'), ' ', ...parts);
}

const flatShape = (s: Shape) => s.h > 1;

function argmaxAbs(a: Float32Array): number {
  let best = 0;
  for (let i = 1; i < a.length; i++) if (Math.abs(a[i]) > Math.abs(a[best])) best = i;
  return best;
}

/** Splits a flat index into (channel, row, col). */
const unflat = (i: number, s: Shape) => ({ c: Math.floor(i / (s.h * s.w)), y: Math.floor((i % (s.h * s.w)) / s.w), x: i % s.w });

/** Indices of up to n diverse entries (largest, smallest, around zero) for activation examples. */
function examples(z: Float32Array, count: number): number[] {
  const idx = Array.from(z.keys());
  const byVal = idx.slice().sort((a, b) => z[a] - z[b]);
  const picks = new Set<number>([byVal[byVal.length - 1], byVal[0], byVal[Math.floor(byVal.length / 2)], byVal[Math.floor(byVal.length * 0.75)], byVal[Math.floor(byVal.length * 0.25)]]);
  return [...picks].slice(0, count);
}

/** Where element i of a tensor of shape s sits, in words: "[ch 2, 3, 4]", "[green, 3, 4]", "[7]". */
const locate = (i: number, s: Shape, chans?: string[]) => {
  if (!flatShape(s)) return `[${i + 1}]`;
  const u = unflat(i, s);
  return s.c > 1 ? `[${chans ? chans[u.c] : `ch ${u.c + 1}`}, ${u.y}, ${u.x}]` : `[${u.y}, ${u.x}]`;
};

/** One row of the feature table: the feature, its value, and how it came from the coordinates. */
function featureLine(id: string, label: string, coords: Float32Array | null, value: number): string {
  const c = (k: number) => (coords ? n(coords[k], 3) : '?');
  let how = '';
  let m: RegExpMatchArray | null;
  if ((m = id.match(/^x(\d)$/))) how = `coordinate ${m[1]}`;
  else if ((m = id.match(/^x(\d)\^2$/))) how = `(${c(Number(m[1]) - 1)})²`;
  else if ((m = id.match(/^x(\d)\*x(\d)$/))) how = `${c(Number(m[1]) - 1)} × ${c(Number(m[2]) - 1)}`;
  else if ((m = id.match(/^sin x(\d)$/))) how = `sin(π × ${c(Number(m[1]) - 1)})`;
  return `  ${label.padEnd(9)}${ns(value, 3, 7)}    ${how}`;
}

// ── Steps ─────────────────────────────────────────────────────────────────

function buildSteps(t: Trace, ui: { eta: number; onEta: (v: number) => void; apply: () => void }): Step[] {
  const steps: Step[] = [];
  const L = t.blocks.length;
  const c = t.ctx;

  steps.push({
    phase: 'Forward pass',
    title: c.input === 'points' ? 'Input point' : 'Input image',
    block: -1,
    dir: 'fwd',
    render: (el) => renderInput(el, t),
  });

  for (const bt of t.blocks) {
    if (bt.kind === 'conv') convForwardSteps(steps, bt, c);
    else denseForwardSteps(steps, bt, c);
  }

  const out = t.blocks[L - 1];
  const pred = argmax(t.probs);
  const K = c.K;
  const labels = classLabels(c);
  const nameCol = !c.digits;
  const nameW = nameCol ? Math.max(5, ...c.names.map((s) => s.length)) + 2 : 0;
  steps.push({
    phase: 'Forward pass',
    title: 'Softmax',
    block: L - 1,
    dir: 'fwd',
    render: (el) => {
      const z = out.z;
      let m = -Infinity;
      for (const v of z) if (v > m) m = v;
      let sum = 0;
      const e = Array.from(z, (v) => Math.exp(v - m));
      for (const v of e) sum += v;
      const lines: (string | [string, boolean])[] = [`  k  ${nameCol ? 'class'.padEnd(nameW) : ''}  logit z     e^(z − ${n(m, 2)})    p = e / ${n(sum, 3)}`];
      for (let k = 0; k < K; k++) {
        const s = `  ${k}  ${nameCol ? c.names[k].padEnd(nameW) : ''}${ns(z[k])}     ${ns(e[k], 4, 9)}       ${ns(t.probs[k], 4, 8)}${k === pred ? '   ← prediction' : ''}${k === t.label ? '   ← target' : ''}`;
        lines.push(k === pred || k === t.label ? [s, true] : s);
      }
      el.append(
        h('p', { class: 'bp-text' }, `Softmax turns the ${K === 10 ? 'ten' : K} logits into probabilities, one per ${c.what}: exponentiate each one, then divide by the total. Subtracting the largest logit first keeps the exponentials small and does not change the result.`),
        formula('p[k] = e^z[k] / Σⱼ e^z[j]', false),
        row(vec('z · logits', out.z, { labels, mark: pred }), vec('p · probabilities', t.probs, { labels, mark: pred, signed: false, max: 1 })),
        work(lines),
      );
    },
  });

  steps.push({
    phase: 'Loss',
    title: 'Cross-entropy loss',
    block: L - 1,
    dir: 'fwd',
    render: (el) => {
      const py = t.probs[t.label];
      el.append(
        h('p', { class: 'bp-text' }, `The loss looks only at the probability given to the correct ${c.what}${c.digits ? '' : `, here ${c.names[t.label]}`}. Confident and right costs almost nothing; confident and wrong costs a lot. Training lowers this number.`),
        formula(`L = −log p[y] = −log p[${t.label}]${c.digits ? '' : `   (y = ${t.label}, ${c.names[t.label]})`}`, false),
        work([
          `p[${t.label}] = ${n(py, 4)}`,
          ['L = −log(' + n(py, 4) + ') = ' + n(t.loss, 4), true],
          '',
          pred === t.label
            ? `The network predicts ${className(c, pred)}: correct.`
            : `The network predicts ${className(c, pred)} (p = ${n(t.probs[pred], 3)}): wrong. The target is ${className(c, t.label)}.`,
        ]),
      );
    },
  });

  steps.push({
    phase: 'Backward pass',
    title: 'Gradient at the logits',
    block: L - 1,
    dir: 'back',
    render: (el) => {
      const lines: (string | [string, boolean])[] = [`  k  ${nameCol ? 'class'.padEnd(nameW) : ''}      p    one-hot y        δ = p − y`];
      for (let k = 0; k < K; k++) {
        const s = `  ${k}  ${nameCol ? c.names[k].padEnd(nameW) : ''}${ns(t.probs[k], 4, 8)}       ${k === t.label ? 1 : 0}          ${ns(out.dZ[k], 4, 8)}`;
        lines.push(k === t.label ? [s, true] : s);
      }
      el.append(
        h(
          'p',
          { class: 'bp-text' },
          `Backpropagation starts here. For softmax followed by cross-entropy the gradient of the loss with respect to each logit is simply the predicted probability minus the one-hot target. Only the target ${c.what} (${className(c, t.label)}) gets a negative gradient: raising its logit would lower the loss.`,
        ),
        formula('δ[k] = ∂L/∂z[k] = p[k] − 1[k = y]', true),
        row(vec('δ · ∂L/∂z at the output', out.dZ, { labels, mark: t.label })),
        work(lines),
      );
    },
  });

  for (let i = L - 1; i >= 0; i--) {
    const bt = t.blocks[i];
    if (bt.kind === 'conv') convBackwardSteps(steps, bt, i === 0, c, t.blocks);
    else denseBackwardSteps(steps, bt, i === 0, c, t.blocks);
  }

  steps.push({
    phase: 'Update',
    title: 'Gradient descent step',
    block: L - 1,
    dir: 'update',
    render: (el) => renderUpdate(el, t, ui),
  });
  return steps;
}

function renderInput(el: HTMLElement, t: Trace): void {
  const c = t.ctx;
  const target = classPhrase(c, t.label);
  if (c.input === 'points') {
    const F = t.x.length;
    const dims = c.coords?.length ?? 0;
    const where = c.coords ? `(${Array.from(c.coords, (v) => n(v, 3)).join(', ')})` : '';
    const lines: (string | [string, boolean])[] = [];
    if (c.coords) lines.push(`point ${Array.from({ length: dims }, (_, k) => `x${SUB[k]}`).join(', ')} = ${where}`, '');
    lines.push('  feature    value    from the point');
    for (let i = 0; i < F; i++) lines.push(featureLine(c.featIds[i] ?? '', c.feats[i] ?? `x[${i + 1}]`, c.coords, t.x[i]));
    lines.push('', [`target y = ${t.label}${hasName(c, t.label) ? ` (${c.names[t.label]})` : ''}`, true]);
    const extra = c.feats.length > dims ? ' Squares, products and sines are extra inputs computed from the coordinates; they let even a small network draw curved boundaries.' : '';
    el.append(
      h(
        'p',
        { class: 'bp-text' },
        `The point ${where} enters as ${F} number${F === 1 ? '' : 's'}, one per input feature you chose in 07 Data.${extra} The training target is ${target}.`,
      ),
      formula(`x = (${c.feats.join(', ')})`, false),
      row(vec(`x · ${F} input feature${F === 1 ? '' : 's'}`, t.x, { labels: c.feats, signed: true })),
      work(lines),
    );
    return;
  }
  const shape = t.blocks[0].inShape;
  const { h: H, w: W } = shape;
  if (c.input === 'colour') {
    const HW = H * W;
    const y0 = Math.floor(H / 2) - 2;
    const x0 = Math.floor(W / 2) - 2;
    const lines: (string | [string, boolean])[] = [`Centre 4×4 pixels, rows ${y0}–${y0 + 3}, columns ${x0}–${x0 + 3}:`, `   ${'red'.padEnd(23)}${'green'.padEnd(23)}blue`];
    for (let y = y0; y < y0 + 4; y++) {
      let s = '';
      for (let ch = 0; ch < 3; ch++) {
        for (let x = x0; x < x0 + 4; x++) s += ns(t.x[ch * HW + y * W + x], 2, 5);
        s += '   ';
      }
      lines.push(s);
    }
    const mean = (ch: number) => {
      let s = 0;
      for (let i = 0; i < HW; i++) s += t.x[ch * HW + i];
      return s / HW;
    };
    lines.push('', `Mean of each channel: red ${n(mean(0), 2)} · green ${n(mean(1), 2)} · blue ${n(mean(2), 2)}`, ['target y = ' + t.label + ` (${c.names[t.label]})`, true]);
    el.append(
      h(
        'p',
        { class: 'bp-text' },
        `The photo enters as ${int(t.x.length)} numbers: ${H}×${W} pixels, each with a red, a green and a blue value between 0 and 1. The network sees them as three stacked ${H}×${W} maps, one per colour channel, and the first layer’s kernels reach through all three. The training target is ${target}.`,
      ),
      formula(`x ∈ [0, 1]^(3×${H}×${W})`, false),
      row(picture(`x · ${H}×${W} photo`, t.x, shape, 128), maps('Channels · red, green, blue', t.x, shape, { px: 96, signed: false, max: 1, chans: CHANNELS, mark: { c: -1, y: y0, x: x0, h: 4, w: 4 } })),
      work(lines),
    );
    return;
  }
  // Grey images (MNIST digits, Fashion-MNIST clothes).
  const y0 = Math.floor(H / 2) - 4;
  const x0 = Math.floor(W / 2) - 4;
  const lines: string[] = [`Centre crop, rows ${y0}–${y0 + 7}, columns ${x0}–${x0 + 7}:`];
  for (let y = y0; y < y0 + 8; y++) {
    let s = '';
    for (let x = x0; x < x0 + 8; x++) s += ns(t.x[y * W + x], 2, 6);
    lines.push(s);
  }
  let ink = 0;
  for (const v of t.x) ink += v;
  const isDigit = c.info.id === 'mnist';
  lines.push('', `${isDigit ? 'Total ink' : 'Total brightness'} Σx = ${n(ink, 1)} · target ${c.what} y = ${t.label}${c.digits ? '' : ` (${c.names[t.label]})`}`);
  el.append(
    h(
      'p',
      { class: 'bp-text' },
      isDigit
        ? `The digit enters as ${int(t.x.length)} numbers, one per pixel: 0 is paper, 1 is full ink. The training target is ${target}.`
        : `The ${noun(c.info)} enters as ${int(t.x.length)} numbers, one per pixel: 0 is the black background, 1 the brightest part of the picture. The training target is ${target}.`,
    ),
    formula(`x ∈ [0, 1]^(${H}×${W})`, false),
    row(maps(`x · ${H}×${W} input`, t.x, { c: 1, h: H, w: W }, { px: 196, signed: false, max: 1, mark: { c: 0, y: y0, x: x0, h: 8, w: 8 } })),
    work(lines),
  );
}

function convForwardSteps(steps: Step[], bt: BT, cx: Ctx): void {
  const { c: C, h: H, w: W } = bt.inShape;
  const F = bt.zShape.c;
  const k = bt.k;
  const p = bt.pad;
  const kk = k * k;
  const colour = readsColour(cx, bt);
  const chans = colour ? CHANNELS : undefined;
  // Position of the strongest response, used for the worked example.
  let best = 0;
  for (let i = 1; i < bt.z.length; i++) if (bt.z[i] > bt.z[best]) best = i;
  const f = Math.floor(best / (H * W));
  const y0 = Math.floor((best % (H * W)) / W);
  const x0 = best % W;

  steps.push({
    phase: 'Forward pass',
    title: `${bt.name}: convolution`,
    block: bt.index,
    dir: 'fwd',
    render: (el) => {
      const lines: (string | [string, boolean])[] = [[`Filter ${f + 1} at row ${y0}, column ${x0} (its strongest response):`, true], ''];
      let total = bt.b[f];
      const shown = colour ? 3 : Math.min(C, 2);
      const parts: string[] = [];
      for (let c = 0; c < C; c++) {
        let s = 0;
        const patch: number[] = [];
        for (let i = 0; i < k; i++) {
          for (let j = 0; j < k; j++) {
            const yy = y0 + i - p;
            const xx = x0 + j - p;
            const v = yy >= 0 && yy < H && xx >= 0 && xx < W ? bt.x[c * H * W + yy * W + xx] : 0;
            patch.push(v);
            s += v * bt.W[(f * C + c) * kk + i * k + j];
          }
        }
        total += s;
        parts.push(n(s));
        if (c < shown) {
          const head = colour ? `${CHANNELS[c]}:`.padEnd(12) : `channel ${c + 1}:`.padEnd(12);
          lines.push(`${head}input patch${' '.repeat(Math.max(1, k * 7 - 11))}   kernel W[${f + 1}, ${colour ? CHANNELS[c] : c + 1}]`);
          for (let i = 0; i < k; i++) {
            let a = '';
            let b = '';
            for (let j = 0; j < k; j++) {
              a += ns(patch[i * k + j], 2, 7);
              b += ns(bt.W[(f * C + c) * kk + i * k + j], 3, 7);
            }
            lines.push(`            ${a}   ${b}`);
          }
          lines.push(`            Σ patch × kernel = ${n(s)}`, '');
        }
      }
      if (C > shown) lines.push(`… and ${C - shown} more channel${C - shown > 1 ? 's' : ''}, each summed the same way.`, '');
      lines.push(`bias b[${f + 1}] = ${n(bt.b[f])}`);
      lines.push([`z[${f + 1}, ${y0}, ${x0}] = ${parts.length > 6 ? parts.slice(0, 5).join(' + ') + ' + …' : parts.join(' + ')} + ${n(bt.b[f])} = ${n(total)}`.replace(/\+ −/g, '− '), true]);
      el.append(
        h(
          'p',
          { class: 'bp-text' },
          colour
            ? `Each of the ${F} filters slides a ${k}×${k}×3 kernel over the photo: one ${k}×${k} slice for each colour channel. At every position it multiplies the red, green and blue patches under it by their slices, adds everything up and adds the filter's bias, so a filter can respond to colour as well as to shape. Zero padding of ${p} keeps the output at ${H}×${W}.`
            : `Each of the ${F} filters slides a ${k}×${k} kernel over all ${C} input channel${C > 1 ? 's' : ''}. At every position it multiplies the patch under it by the kernel, adds everything up and adds the filter's bias. Zero padding of ${p} keeps the output at ${H}×${W}.`,
        ),
        formula(`z[f, y, x] = b[f] + Σ_c Σ_i Σ_j W[f, c, i, j] · x[c, y+i−${p}, x+j−${p}]`, false),
        row(
          maps(`x · input ${H}×${W}×${C}`, bt.x, bt.inShape, { mark: { c: colour ? -1 : 0, y: y0 - p, x: x0 - p, h: k, w: k }, chans }),
          kernels(`W · ${F}×${C} kernels of ${k}×${k}`, bt.W, F, C, k, f, undefined, colour ? RGB_LABELS : undefined),
          maps(`z · ${H}×${W}×${F}`, bt.z, bt.zShape, { mark: { c: f, y: y0, x: x0 }, signed: true }),
        ),
        work(lines),
      );
    },
  });

  steps.push(activationForward(bt, bt.zShape));

  if (bt.pool) {
    steps.push({
      phase: 'Forward pass',
      title: `${bt.name}: max-pool 2×2`,
      block: bt.index,
      dir: 'fwd',
      render: (el) => {
        const { h: PH, w: PW } = bt.outShape;
        // Show the window feeding the largest pooled value.
        let o = 0;
        for (let i = 1; i < bt.out.length; i++) if (bt.out[i] > bt.out[o]) o = i;
        const fc = Math.floor(o / (PH * PW));
        const py = Math.floor((o % (PH * PW)) / PW);
        const px = o % PW;
        const base = fc * H * W + 2 * py * W + 2 * px;
        const win = [base, base + 1, base + W, base + W + 1];
        const won = bt.argmax![o];
        el.append(
          h('p', { class: 'bp-text' }, 'Each 2×2 window keeps only its largest value, halving height and width. The layer remembers which position won, because the backward pass needs it.'),
          formula('out[f, y, x] = max( a[f, 2y, 2x], a[f, 2y, 2x+1], a[f, 2y+1, 2x], a[f, 2y+1, 2x+1] )', false),
          row(
            maps(`a · before pooling ${H}×${W}×${F}`, bt.a, bt.zShape, { mark: { c: fc, y: 2 * py, x: 2 * px, h: 2, w: 2 } }),
            maps(`out · after pooling ${PH}×${PW}×${F}`, bt.out, bt.outShape, { mark: { c: fc, y: py, x: px } }),
          ),
          work([
            `Filter ${fc + 1}, window rows ${2 * py}–${2 * py + 1}, columns ${2 * px}–${2 * px + 1}:`,
            `   ${ns(bt.a[win[0]])} ${ns(bt.a[win[1]])}`,
            `   ${ns(bt.a[win[2]])} ${ns(bt.a[win[3]])}`,
            [`out[${fc + 1}, ${py}, ${px}] = ${n(bt.out[o])}  (from row ${Math.floor((won % (H * W)) / W)}, column ${won % W})`, true],
          ]),
        );
      },
    });
  }
}

function activationForward(bt: BT, s: Shape): Step {
  return {
    phase: 'Forward pass',
    title: `${bt.name}: ${actLabel(bt.act)}`,
    block: bt.index,
    dir: 'fwd',
    render: (el) => {
      const idx = examples(bt.z, 5);
      const lines: (string | [string, boolean])[] = [`  position          z   →   a = ${actFormula(bt.act)}`];
      for (const i of idx) lines.push(`  ${locate(i, s).padEnd(14)}${ns(bt.z[i])}   →   ${n(bt.a[i])}`);
      const why: Record<Act, string> = {
        relu: 'ReLU keeps positive values and replaces negative ones with 0, so a unit is either off or passes its signal through unchanged.',
        leaky: 'Leaky ReLU keeps positive values and scales negative ones by 0.1, so a unit never goes completely silent.',
        tanh: 'Tanh squashes every value into the range −1 to 1.',
        sigmoid: 'Sigmoid squashes every value into the range 0 to 1.',
        linear: 'A linear activation leaves the values unchanged. Stacked linear layers collapse into a single linear map.',
      };
      el.append(
        h('p', { class: 'bp-text' }, `The activation function is applied to every value on its own. ${why[bt.act]} Without it, the whole network would be one linear function.`),
        formula(`a = f(z),   f(z) = ${actFormula(bt.act)}`, false),
        row(
          flatShape(s) ? maps(`z · before ${actLabel(bt.act)}`, bt.z, s, { signed: true }) : vec(`z · before ${actLabel(bt.act)}`, bt.z, { signed: true }),
          flatShape(s) ? maps(`a · after ${actLabel(bt.act)}`, bt.a, s) : vec(`a · after ${actLabel(bt.act)}`, bt.a),
        ),
        work(lines),
      );
    },
  };
}

function denseForwardSteps(steps: Step[], bt: BT, c: Ctx): void {
  const N = bt.inShape.c * bt.inShape.h * bt.inShape.w;
  const M = bt.zShape.c;
  const j = bt.isOut ? argmax(bt.z) : argmaxAbs(bt.z);
  const feats = readsFeatures(c, bt);
  const colour = readsColour(c, bt);
  steps.push({
    phase: 'Forward pass',
    title: bt.isOut ? 'Output layer: logits' : `${bt.name}: weighted sum`,
    block: bt.index,
    dir: 'fwd',
    render: (el) => {
      const terms = Array.from({ length: N }, (_, i) => ({ i, v: bt.W[j * N + i] * bt.x[i] })).sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
      const top = terms.slice(0, 6);
      const rest = terms.slice(6).reduce((s, q) => s + q.v, 0);
      const unit = bt.isOut ? `${c.digits ? 'digit' : 'class'} ${className(c, j)}` : `unit ${j + 1}`;
      const lines: (string | [string, boolean])[] = [[`${bt.isOut ? `Logit for ${unit}` : `Unit ${j + 1}`}: ${N > 6 ? `the 6 largest of its ${int(N)} terms` : `its ${N} term${N === 1 ? '' : 's'}`}`, true]];
      const jl = bt.isOut ? String(j) : String(j + 1);
      const wlab = (i: number) => (feats ? `W[${jl}, ${c.feats[i]}] · ${c.feats[i]}` : `W[${jl}, ${i + 1}] · x[${i + 1}]`);
      const wcol = Math.max(16, ...top.map((q) => wlab(q.i).length));
      // Every value ends in the same column: label, " = ", w (7), " × ", x (7), " = ", w·x (7).
      const end = 2 + wcol + 3 + 7 + 3 + 7 + 3 + 7;
      const right = (left: string, value: string) => `${left}${' '.repeat(Math.max(1, end - left.length - value.length))}${value}`;
      for (const q of top) lines.push(`  ${wlab(q.i).padEnd(wcol)} = ${ns(bt.W[j * N + q.i])} × ${ns(bt.x[q.i])} = ${ns(q.v)}`);
      if (N > 6) lines.push(right(`  … ${int(N - 6)} more terms`, `sum ${ns(rest)}`));
      lines.push(right(`  bias b[${jl}]`, ns(bt.b[j])));
      lines.push([`  z = ${n(bt.z[j])}`, true]);
      const flat = flatShape(bt.inShape);
      const outWhat = bt.isOut ? `output units (one per ${c.what})` : 'units';
      el.append(
        h(
          'p',
          { class: 'bp-text' },
          `${flat ? `The ${bt.inShape.h}×${bt.inShape.w}×${bt.inShape.c} input is first flattened into one vector of ${int(N)} numbers. ` : ''}${feats ? `Its inputs are the ${N} features (${c.feats.join(', ')}). ` : ''}Each of the ${M} ${outWhat} multiplies every input by its own weight, adds them up and adds a bias. The highlighted row of W produces ${unit}.`,
        ),
        formula('z[j] = b[j] + Σᵢ W[j, i] · x[i]', false),
        row(
          flat ? maps(`x · input ${bt.inShape.h}×${bt.inShape.w}×${bt.inShape.c}`, bt.x, bt.inShape, { chans: colour ? CHANNELS : undefined }) : vec(feats ? `x · input features` : `x · input (${N})`, bt.x, { labels: feats ? c.feats : undefined }),
          mat(`W · ${M}×${N}`, bt.W, M, N, { markRow: j, rowLabels: bt.isOut ? classLabels(c) : undefined, colLabels: feats ? c.feats : undefined }),
          vec(`z · ${bt.isOut ? 'logits' : `${M} sums`}`, bt.z, { labels: bt.isOut ? classLabels(c) : undefined, mark: j, signed: true }),
        ),
        work(lines),
      );
    },
  });
  if (!bt.isOut) steps.push(activationForward(bt, bt.zShape));
}

/**
 * What freezing means for this layer's backward step. Training skips the backward pass below the
 * lowest trainable layer (nothing there needs a gradient); above it, δ must still pass through.
 */
function frozenBackward(bt: BT, layers: BT[]): HTMLElement | null {
  if (!store.isFrozen(bt.index)) return null;
  const below = layers.filter((b) => b.index < bt.index && !store.isFrozen(b.index));
  if (below.length) {
    return frozenNote(
      h('b', null, 'Frozen: training leaves these weights alone.'),
      ` ∂L/∂W above is what plain training would use, but the update skips it. δ still has to pass through this layer, because ${below.map((b) => b.name).join(' and ')} below ${below.length > 1 ? 'are' : 'is'} trainable.`,
    );
  }
  return frozenNote(
    h('b', null, 'Frozen: training leaves these weights alone.'),
    ' Nothing below this layer is trainable either, so training skips this step entirely: no one needs its gradients. They are worked out here only so you can see them.',
  );
}

function denseBackwardSteps(steps: Step[], bt: BT, first: boolean, c: Ctx, layers: BT[]): void {
  const N = bt.inShape.c * bt.inShape.h * bt.inShape.w;
  const M = bt.zShape.c;
  const feats = readsFeatures(c, bt);
  const colour = readsColour(c, bt);
  if (!bt.isOut) steps.push(activationBackward(bt, bt.zShape));
  steps.push({
    phase: 'Backward pass',
    title: bt.isOut ? 'Output layer: gradients' : `${bt.name}: gradients`,
    block: bt.index,
    dir: 'back',
    render: (el) => {
      const gi = argmaxAbs(bt.gW);
      const gj = Math.floor(gi / N);
      const gc = gi % N;
      const xi = argmaxAbs(bt.dX);
      const terms = Array.from({ length: M }, (_, j) => ({ j, v: bt.W[j * N + xi] * bt.dZ[j] })).sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
      const lab = (j: number) => (bt.isOut ? String(j) : String(j + 1));
      const xin = (i: number) => (feats ? c.feats[i] : `x[${i + 1}]`);
      const wIdx = (j: number, i: number) => `W[${lab(j)}, ${feats ? c.feats[i] : i + 1}]`;
      const where = feats ? ` (the feature ${c.feats[xi]})` : first ? (colour ? ` (a pixel’s ${CHANNELS[unflat(xi, bt.inShape).c]} value)` : ' (a pixel)') : '';
      const lines: (string | [string, boolean])[] = [
        ['Largest weight gradient:', true],
        `  ∂L/∂${wIdx(gj, gc)} = δ[${lab(gj)}] · ${xin(gc)} = ${n(bt.dZ[gj], 4)} × ${n(bt.x[gc], 4)} = ${n(bt.gW[gi], 4)}`,
        `  ∂L/∂b[${lab(gj)}] = δ[${lab(gj)}] = ${n(bt.gb[gj], 4)}`,
        '',
        [`Gradient for input ${feats ? c.feats[xi] : locate(xi, bt.inShape, colour ? CHANNELS : undefined)}${where}:`, true],
      ];
      for (const q of terms.slice(0, 4)) lines.push(`  ${wIdx(q.j, xi)} · δ[${lab(q.j)}] = ${ns(bt.W[q.j * N + xi], 4, 8)} × ${ns(bt.dZ[q.j], 4, 8)} = ${ns(q.v, 4, 8)}`);
      if (M > 4) lines.push(`  … ${M - 4} more terms`);
      lines.push([`  ∂L/∂${xin(xi)} = ${n(bt.dX[xi], 4)}`, true]);
      const flat = flatShape(bt.inShape);
      const tail = first
        ? feats
          ? '. Here that gives the gradient with respect to the input features: how the loss would change if the point’s features moved'
          : '. Here that gives the gradient with respect to the input pixels: a saliency map of which pixels would change the loss most'
        : '';
      el.append(
        h(
          'p',
          { class: 'bp-text' },
          `With δ known for every unit, three gradients follow. The weight gradient is the outer product of δ and the layer's input, so weights from strongly active inputs into units with large error change most. The bias gradient is δ itself. To continue backwards, δ is sent through the same weights, transposed${tail}.`,
        ),
        formula('∂L/∂W[j, i] = δ[j] · x[i]\n∂L/∂b[j] = δ[j]\n∂L/∂x[i] = Σⱼ W[j, i] · δ[j]', true),
        row(
          vec('δ · ∂L/∂z', bt.dZ, { labels: bt.isOut ? classLabels(c) : undefined, mark: gj, signed: true }),
          mat(`∂L/∂W · ${M}×${N}`, bt.gW, M, N, { markRow: gj, markCol: flat ? undefined : gc, rowLabels: bt.isOut ? classLabels(c) : undefined, colLabels: feats ? c.feats : undefined }),
          flat
            ? maps(first ? '∂L/∂x · input gradient (saliency)' : `∂L/∂x · to ${bt.inShape.h}×${bt.inShape.w}×${bt.inShape.c}`, bt.dX, bt.inShape, {
                signed: true,
                mark: { ...unflat(xi, bt.inShape) },
                px: first ? 168 : 84,
                chans: colour ? CHANNELS : undefined,
              })
            : vec(first && feats ? '∂L/∂x · gradient for each feature' : '∂L/∂x · to the previous layer', bt.dX, { signed: true, mark: xi, labels: feats ? c.feats : undefined }),
        ),
        work(lines),
      );
      const fz = frozenBackward(bt, layers);
      if (fz) el.append(fz);
    },
  });
}

function activationBackward(bt: BT, s: Shape): Step {
  return {
    phase: 'Backward pass',
    title: `${bt.name}: back through ${actLabel(bt.act)}`,
    block: bt.index,
    dir: 'back',
    render: (el) => {
      const slope = Float32Array.from(bt.z, (z) => derivative(bt.act, z));
      const idx = examples(bt.dA, 5);
      const lines: (string | [string, boolean])[] = [`  position         ∂L/∂a          z    f′(z)          δ`];
      for (const i of idx) lines.push(`  ${locate(i, s).padEnd(13)}${ns(bt.dA[i], 4, 9)}  ${ns(bt.z[i], 3, 9)}  ${ns(slope[i], 2, 6)}  ${ns(bt.dZ[i], 4, 9)}`);
      const flat = flatShape(s);
      const note = bt.act === 'relu' ? ' Where the unit was off (z ≤ 0) the slope is 0, so no gradient flows through it and its incoming weights do not change for this example.' : bt.act === 'sigmoid' || bt.act === 'tanh' ? ' Where the unit saturated, the slope is close to 0 and the gradient fades.' : '';
      el.append(
        h('p', { class: 'bp-text' }, `The gradient arriving from the layer above is multiplied, element by element, by the slope of the activation at the value it had in the forward pass.${note}`),
        formula(`δ = ∂L/∂a ⊙ f′(z),   ${derivativeFormula(bt.act)}`, true),
        row(
          flat ? maps('∂L/∂a · from above', bt.dA, s, { signed: true }) : vec('∂L/∂a · from above', bt.dA, { signed: true }),
          flat ? maps('f′(z) · slope', slope, s, { signed: false }) : vec('f′(z) · slope', slope, { signed: false }),
          flat ? maps('δ = ∂L/∂z', bt.dZ, s, { signed: true }) : vec('δ = ∂L/∂z', bt.dZ, { signed: true }),
        ),
        work(lines),
      );
    },
  };
}

function convBackwardSteps(steps: Step[], bt: BT, first: boolean, cx: Ctx, layers: BT[]): void {
  const { c: C, h: H, w: W } = bt.inShape;
  const F = bt.zShape.c;
  const k = bt.k;
  const p = bt.pad;
  const kk = k * k;
  const colour = readsColour(cx, bt);
  if (bt.pool) {
    steps.push({
      phase: 'Backward pass',
      title: `${bt.name}: back through max-pool`,
      block: bt.index,
      dir: 'back',
      render: (el) => {
        const o = argmaxAbs(bt.dOut);
        const { h: PH, w: PW } = bt.outShape;
        const fc = Math.floor(o / (PH * PW));
        const py = Math.floor((o % (PH * PW)) / PW);
        const px = o % PW;
        const won = bt.argmax![o];
        const wy = Math.floor((won % (H * W)) / W);
        const wx = won % W;
        el.append(
          h('p', { class: 'bp-text' }, 'Only the value that won each 2×2 window reached the output, so only that position receives the gradient. The other three positions get exactly zero: nudging them would not have changed anything.'),
          formula('∂L/∂a[f, winner of window] = ∂L/∂out[f, y, x],   0 elsewhere', true),
          row(
            maps(`∂L/∂out · ${PH}×${PW}×${F}`, bt.dOut, bt.outShape, { signed: true, mark: { c: fc, y: py, x: px } }),
            maps(`∂L/∂a · ${H}×${W}×${F}`, bt.dA, bt.zShape, { signed: true, mark: { c: fc, y: 2 * py, x: 2 * px, h: 2, w: 2 } }),
          ),
          work([
            `Filter ${fc + 1}, pooled cell (${py}, ${px}) has gradient ${n(bt.dOut[o], 4)}.`,
            `Its window covered rows ${2 * py}–${2 * py + 1}, columns ${2 * px}–${2 * px + 1}; the max was at (${wy}, ${wx}).`,
            [`∂L/∂a[${fc + 1}, ${wy}, ${wx}] = ${n(bt.dA[won], 4)}; the other three get 0.`, true],
          ]),
        );
      },
    });
  }
  steps.push(activationBackward(bt, bt.zShape));
  steps.push({
    phase: 'Backward pass',
    title: `${bt.name}: gradients`,
    block: bt.index,
    dir: 'back',
    render: (el) => {
      const gi = argmaxAbs(bt.gW);
      const f = Math.floor(gi / (C * kk));
      const c = Math.floor((gi % (C * kk)) / kk);
      const ki = Math.floor((gi % kk) / k);
      const kj = gi % k;
      const cName = colour ? CHANNELS[c] : String(c + 1);
      const terms: { y: number; x: number; d: number; v: number }[] = [];
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const yy = y + ki - p;
          const xx = x + kj - p;
          if (yy < 0 || yy >= H || xx < 0 || xx >= W) continue;
          const d = bt.dZ[f * H * W + y * W + x];
          const v = bt.x[c * H * W + yy * W + xx];
          if (d !== 0 && v !== 0) terms.push({ y, x, d, v });
        }
      }
      terms.sort((a, b) => Math.abs(b.d * b.v) - Math.abs(a.d * a.v));
      let sb = 0;
      for (let i = 0; i < H * W; i++) sb += bt.dZ[f * H * W + i];
      const lines: (string | [string, boolean])[] = [
        [`Largest kernel gradient: ∂L/∂W[${f + 1}, ${cName}, ${ki}, ${kj}]`, true],
        `  sums δ × input over all ${H * W} positions; ${terms.length} are non-zero. The largest:`,
      ];
      for (const q of terms.slice(0, 4)) lines.push(`  δ[${f + 1}, ${q.y}, ${q.x}] · x[${cName}, ${q.y + ki - p}, ${q.x + kj - p}] = ${ns(q.d, 4, 8)} × ${n(q.v, 3)} = ${ns(q.d * q.v, 4, 8)}`);
      if (terms.length > 4) lines.push(`  … ${terms.length - 4} more`);
      lines.push([`  ∂L/∂W[${f + 1}, ${cName}, ${ki}, ${kj}] = ${n(bt.gW[gi], 4)}`, true], '', [`∂L/∂b[${f + 1}] = Σ δ[${f + 1}] = ${n(sb, 4)}`, true]);
      const tail = first
        ? colour
          ? '. For this first layer that yields the gradient with respect to the pixels themselves, one map per colour channel: a saliency map'
          : '. For this first layer that yields the gradient with respect to the pixels themselves, a saliency map'
        : '';
      el.append(
        h(
          'p',
          { class: 'bp-text' },
          `The kernel gradient lines the layer's input up with δ: for each kernel tap, add up δ times the input value that tap touched, at every position. The bias gradient adds up δ over its whole map. To continue backwards, δ is convolved with the kernels rotated by 180°${tail}.`,
        ),
        formula(`∂L/∂W[f, c, i, j] = Σ_y Σ_x δ[f, y, x] · x[c, y+i−${p}, x+j−${p}]\n∂L/∂b[f] = Σ_y Σ_x δ[f, y, x]\n∂L/∂x[c, y, x] = Σ_f Σ_i Σ_j W[f, c, i, j] · δ[f, y−i+${p}, x−j+${p}]`, true),
        row(
          maps(`δ · ${H}×${W}×${F}`, bt.dZ, bt.zShape, { signed: true, mark: { c: f, y: 0, x: 0, h: H, w: W } }),
          kernels(`∂L/∂W · ${F}×${C}×${k}×${k}`, bt.gW, F, C, k, f, c, colour ? RGB_LABELS : undefined),
          vec('∂L/∂b', bt.gb, { signed: true, mark: f }),
        ),
        row(maps(first ? '∂L/∂x · input gradient (saliency)' : `∂L/∂x · to ${H}×${W}×${C}`, bt.dX, bt.inShape, { signed: true, px: first ? 168 : 84, chans: colour ? CHANNELS : undefined })),
        work(lines),
      );
      const fz = frozenBackward(bt, layers);
      if (fz) el.append(fz);
    },
  });
}

function renderUpdate(el: HTMLElement, t: Trace, ui: { eta: number; onEta: (v: number) => void; apply: () => void }): void {
  const c = t.ctx;
  const eta = ui.eta;
  const frozen = t.blocks.map((b) => store.isFrozen(b.index));
  const next = nextWeights(t, eta);
  const probe = new Network(t.arch, 0);
  probe.setWeights(next);
  const after = probe.forward(t.x);
  const lossAfter = -Math.log(Math.max(after[t.label], 1e-12));
  // The worked example uses the highest layer that training actually changes.
  const showIdx = t.blocks.map((b) => b.index).filter((i) => !frozen[i]).pop();
  const lines: (string | [string, boolean])[] = [];
  if (showIdx !== undefined) {
    const sb = t.blocks[showIdx];
    const gi = argmaxAbs(sb.gW);
    const idx = sb.kind === 'dense' ? `${Math.floor(gi / sb.x.length) + (sb.isOut ? 0 : 1)}, ${readsFeatures(c, sb) ? c.feats[gi % sb.x.length] : (gi % sb.x.length) + 1}` : `${Math.floor(gi / (sb.inShape.c * sb.k * sb.k)) + 1}, …`;
    lines.push(
      [`One weight in ${sb.isOut ? 'the output layer' : sb.name}:`, true],
      `  W[${idx}] ← ${n(sb.W[gi], 4)} − ${eta} × ${sb.gW[gi] < 0 ? `(${n(sb.gW[gi], 4)})` : n(sb.gW[gi], 4)} = ${n(sb.W[gi] - eta * sb.gW[gi], 4)}`,
      '',
    );
  }
  lines.push(['Every layer:', true], '  layer                 ‖∂L/∂W‖     ‖ΔW‖');
  for (const b of t.blocks) {
    let g = 0;
    for (const v of b.gW) g += v * v;
    const step = frozen[b.index] ? 0 : eta * Math.sqrt(g);
    lines.push(`  ${b.name.padEnd(18)}${ns(Math.sqrt(g), 4, 10)}${ns(step, 4, 10)}${frozen[b.index] ? '   frozen, not updated' : ''}`);
  }
  const before = h('div', { class: 'kpi' }, h('span', { class: 'label' }, 'Loss before'), h('b', null, n(t.loss, 4)), h('span', { class: 'hint' }, `p[${t.label}] = ${n(t.probs[t.label], 3)}`));
  const afterEl = h('div', { class: 'kpi is-accent' }, h('span', { class: 'label' }, 'Loss after'), h('b', null, n(lossAfter, 4)), h('span', { class: 'hint' }, `p[${t.label}] = ${n(after[t.label], 3)}`));
  const allFrozen = frozen.every(Boolean);
  const applyBtn = h('button', { type: 'button', class: 'btn btn-solid', onclick: ui.apply, disabled: allFrozen }, 'Apply to network');
  const frozenNames = t.blocks.filter((b) => frozen[b.index]).map((b) => b.name);
  el.append(
    h(
      'p',
      { class: 'bp-text' },
      'Every weight takes a small step against its gradient, scaled by the step size η. This walkthrough uses plain SGD on this one example, so you can see its effect directly. During training the trainer averages gradients over a batch and uses the optimizer chosen in the bar above.',
    ),
    formula(`W ← W − η · ∂L/∂W        b ← b − η · ∂L/∂b${frozenNames.length ? '\nfrozen layers: W ← W, b ← b' : ''}`, true),
  );
  if (frozenNames.length) {
    el.append(
      frozenNote(
        h('b', null, `Frozen: training leaves ${allFrozen ? 'every layer' : frozenNames.join(', ')} alone.`),
        allFrozen
          ? ' Nothing would change, so there is nothing to apply. Unfreeze a layer in 01 Architecture to train it.'
          : ` ${frozenNames.length > 1 ? 'Their' : 'Its'} gradients were worked out above, but this step leaves ${frozenNames.length > 1 ? 'their' : 'its'} weights as they are: ΔW = 0. That is how transfer learning keeps what a pretrained layer learned while the layers above it adapt to the new ${c.what === 'digit' ? 'data' : 'classes'}.`,
      ),
    );
  }
  el.append(
    h('div', { class: 'tensors', style: { alignItems: 'end' } }, selectField('bp-eta', 'Step size', [0.001, 0.01, 0.03, 0.1, 0.3, 1].map((v) => ({ value: v, label: String(v) })), eta, ui.onEta), h('div', { class: 'loss-compare' }, before, afterEl), applyBtn),
  );
  if (showIdx !== undefined) {
    const sb = t.blocks[showIdx];
    const ni = 2 * t.blocks.indexOf(sb);
    if (sb.kind === 'dense') {
      const M = sb.zShape.c;
      const N = sb.x.length;
      const gi = argmaxAbs(sb.gW);
      const rl = sb.isOut ? classLabels(c) : undefined;
      const cl = readsFeatures(c, sb) ? c.feats : undefined;
      const mc = flatShape(sb.inShape) ? undefined : gi % N;
      el.append(
        row(
          mat(`${sb.isOut ? 'Output' : sb.name} W · before`, sb.W, M, N, { rowLabels: rl, colLabels: cl, markRow: Math.floor(gi / N), markCol: mc }),
          mat(`${sb.isOut ? 'Output' : sb.name} W · after`, next[ni], M, N, { rowLabels: rl, colLabels: cl, markRow: Math.floor(gi / N), markCol: mc }),
        ),
      );
    } else {
      const F = sb.zShape.c;
      const C = sb.inShape.c;
      const chans = readsColour(c, sb) ? RGB_LABELS : undefined;
      el.append(row(kernels(`${sb.name} W · before`, sb.W, F, C, sb.k, undefined, undefined, chans), kernels(`${sb.name} W · after`, next[ni], F, C, sb.k, undefined, undefined, chans)));
    }
  }
  el.append(
    work(lines),
    h(
      'p',
      { class: 'hint' },
      allFrozen
        ? 'With every layer frozen the loss cannot change.'
        : lossAfter < t.loss
          ? `One step lowered the loss on this ${noun(c.info)} by ${n(t.loss - lossAfter, 4)}. Apply it to make the change real; training then continues from the new weights.`
          : 'This step size overshoots: the loss went up. Try a smaller η.',
    ),
  );
}

/** W − η·∂L/∂W for every block that is not frozen; frozen blocks keep their weights. */
function nextWeights(t: Trace, eta: number): Float32Array[] {
  return t.blocks.flatMap((b) => (store.isFrozen(b.index) ? [b.W.slice(), b.b.slice()] : [b.W.map((w, i) => w - eta * b.gW[i]), b.b.map((w, i) => w - eta * b.gb[i])]));
}

// ── Side panel: the example ───────────────────────────────────────────────

/** A point dataset's example: the test points in their class colours, the current point marked. */
function paintPoint(canvas: HTMLCanvasElement, coords: Float32Array | null, px: number): void {
  const p = palette();
  const ctx = fitCanvas(canvas, px, px);
  ctx.fillStyle = p.surface;
  ctx.fillRect(0, 0, px, px);
  const d = store.data?.points;
  const sx = (v: number) => ((v + 1.1) / 2.2) * px;
  const sy = (v: number) => px - ((v + 1.1) / 2.2) * px;
  if (d && store.data) {
    const ys = store.data.testY;
    ctx.globalAlpha = 0.55;
    for (let i = 0; i < ys.length; i++) {
      ctx.fillStyle = classColor(ys[i]);
      ctx.fillRect(sx(d.testCoords[i * d.dims]) - 1, sy(d.testCoords[i * d.dims + 1]) - 1, 2, 2);
    }
    ctx.globalAlpha = 1;
  }
  if (coords) {
    const x = sx(coords[0]);
    const y = sy(coords[1]);
    ctx.strokeStyle = p.surface;
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.stroke();
    ctx.strokeStyle = p.accent;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.strokeStyle = p.hair;
  ctx.lineWidth = 1;
  ctx.strokeRect(0.5, 0.5, px - 1, px - 1);
}

// ── Mount ─────────────────────────────────────────────────────────────────

export function mountBackprop(): void {
  const root = $('bplab');
  const sampleCanvas = h('canvas', { role: 'img', 'aria-label': 'The example' }) as HTMLCanvasElement;
  const sampleCap = h('div', { class: 'hint' });
  const sampleBox = h('div', { class: 'bp-sample' }, sampleCanvas, sampleCap);
  const targetHead = h('p', { class: 'sub' }, 'Target digit y');
  const targetWrap = h('div');
  const snapNote = h('p', { class: 'hint' });
  const refresh = h('button', { type: 'button', class: 'btn btn-sm' }, 'Use current weights');
  const prev = h('button', { type: 'button', class: 'btn' }, '← Prev');
  const next = h('button', { type: 'button', class: 'btn btn-solid' }, 'Next →');
  const restart = h('button', { type: 'button', class: 'btn' }, 'Restart');
  const list = h('ol', { class: 'steps', 'aria-label': 'Steps' });
  const main = h('div', { class: 'bp-main', 'aria-live': 'polite' });

  root.append(
    h(
      'div',
      { class: 'bp' },
      h(
        'div',
        { class: 'bp-side' },
        h('div', null, h('p', { class: 'sub' }, 'Example'), sampleBox),
        h('div', null, targetHead, targetWrap),
        h('div', null, snapNote, refresh),
        h('div', { class: 'bp-nav' }, prev, next, restart),
        list,
      ),
      main,
    ),
  );

  let trace: Trace | null = null;
  let steps: Step[] = [];
  let cur = 0;
  let target: number | null = null;
  let eta = 0.1;
  let visible = false;
  let applied = '';

  const retrace = () => {
    const probe = store.probe;
    // While a new dataset loads, the network can already expect a different input than the probe.
    if (!probe || probe.x.length !== store.net.inputSize) {
      trace = null;
      steps = [];
      renderSide();
      renderStep();
      return;
    }
    const K = store.classes;
    if (target !== null && target >= K) target = null;
    const y = target ?? (probe.label !== null && probe.label < K ? probe.label : argmax(store.net.forward(probe.x)));
    trace = makeTrace(store.net, probe.x, y, probe.coords);
    const ui = {
      get eta() {
        return eta;
      },
      onEta: (v: number) => {
        eta = v;
        applied = '';
        renderStep();
      },
      apply: () => {
        if (!trace) return;
        // Another architecture since this trace was made: its weights no longer fit.
        if (trace.version !== store.version) return retrace();
        const t = trace;
        const nextW = nextWeights(t, eta);
        const before = t.loss;
        applyWeights(nextW);
        retrace();
        applied = `Applied. The loss on this ${noun(t.ctx.info)} went from ${n(before, 4)} to ${n(trace!.loss, 4)}.`;
        renderStep();
      },
    };
    steps = buildSteps(trace, ui);
    cur = Math.min(cur, steps.length - 1);
    renderSide();
    renderStep();
  };

  const renderSide = () => {
    const probe = store.probe;
    const info = store.info;
    const c = trace?.ctx;
    const pts = info.kind === 'points';
    sampleBox.classList.toggle('is-points', pts);
    if (pts) paintPoint(sampleCanvas, trace?.ctx.coords ?? null, 96);
    else {
      const shape = store.input;
      const x = trace && trace.x.length === shape.c * shape.h * shape.w ? trace.x : new Float32Array(shape.c * shape.h * shape.w);
      const ctx = fitCanvas(sampleCanvas, 64, 64);
      drawSample(ctx, x, shape, 0, 0, 64, 64);
    }
    sampleCanvas.setAttribute('aria-label', trace ? `The example: ${probe?.caption ?? noun(info)}` : 'No example yet');
    sampleCap.textContent = probe
      ? probe.caption
      : pts
        ? 'Pick a point in 03 or 07 Data.'
        : `Pick ${info.id === 'mnist' ? 'a digit' : 'an image'} in 02 Network or 07 Data${info.id === 'mnist' ? ', or draw one' : ''}.`;
    const what = classWord(info);
    targetHead.textContent = `Target ${what} y`;
    clear(targetWrap);
    const K = info.classes.length;
    targetWrap.append(
      digitChips(
        trace?.label ?? null,
        (d) => {
          target = d;
          retrace();
        },
        `Target ${what}`,
        K,
        info.classes,
      ),
    );
    if (trace && c && !c.digits) targetWrap.append(h('p', { class: 'hint bp-target-name' }, `y = ${trace.label}: ${c.names[trace.label]}`));
    if (probe && probe.label === null && target === null) {
      const src = probe.key === 'draw' ? 'Your drawing' : probe.key === 'photo' ? 'Your photo' : pts ? 'This point' : 'This input';
      targetWrap.append(h('p', { class: 'hint', style: { marginTop: '6px' } }, `${src} has no label, so the prediction is used. Pick the ${what} you meant.`));
    }
    updateSnap();
    clear(list);
    let phase = '';
    steps.forEach((s, i) => {
      if (s.phase !== phase) {
        phase = s.phase;
        list.append(h('li', { class: 'phase' }, phase));
      }
      const b = h('button', { type: 'button' }, h('span', { class: 'step-n' }, String(i + 1).padStart(2, '0')), h('span', null, s.title));
      b.addEventListener('click', () => go(i));
      list.append(h('li', { class: i === cur ? 'is-current' : i < cur ? 'is-done' : 'is-todo', 'aria-current': i === cur ? 'step' : undefined }, b));
    });
  };

  const updateSnap = () => {
    if (!trace) {
      snapNote.textContent = store.data ? `Waiting for ${noun(store.info) === 'image' ? 'an' : 'a'} ${noun(store.info)}.` : `Loading ${store.info.name}…`;
      refresh.disabled = true;
      return;
    }
    const stale = trace.version !== store.version || trace.step !== store.weightsStep;
    snapNote.textContent = store.running
      ? `Frozen at training step ${int(trace.step)} while training runs.`
      : stale
        ? `Frozen at step ${int(trace.step)}; the network is now at step ${int(store.weightsStep)}.`
        : `Using the network's weights at step ${int(trace.step)}.`;
    refresh.disabled = !stale;
  };

  const renderStep = () => {
    clear(main);
    const s = steps[cur];
    if (!s || !trace) {
      main.append(h('p', { class: 'hint' }, store.data ? 'Pick an example to start.' : `Loading ${store.info.name}…`));
      return;
    }
    const back = s.dir !== 'fwd';
    main.append(
      h(
        'div',
        { class: 'bp-title' },
        h('span', { class: 'mono', style: { color: 'var(--muted)' } }, `${String(cur + 1).padStart(2, '0')} / ${steps.length}`),
        h('h3', null, s.title),
        h('span', { class: `dir${back ? ' is-back' : ''}` }, s.dir === 'fwd' ? 'Forward →' : s.dir === 'back' ? '← Backward' : 'Update'),
      ),
    );
    const body = h('div', { class: 'bp-body' });
    s.render(body);
    main.append(body);
    if (s.dir === 'update' && applied) main.append(h('p', { class: 'notice' }, applied));
    prev.disabled = cur === 0;
    next.disabled = cur === steps.length - 1;
    Array.from(list.querySelectorAll('li:not(.phase)')).forEach((li, i) => {
      li.className = i === cur ? 'is-current' : i < cur ? 'is-done' : 'is-todo';
      if (i === cur) li.setAttribute('aria-current', 'step');
      else li.removeAttribute('aria-current');
    });
    if (visible) setHighlight({ block: s.block, dir: s.dir });
  };

  const go = (i: number) => {
    cur = Math.max(0, Math.min(steps.length - 1, i));
    if (steps[cur]?.dir !== 'update') applied = '';
    renderStep();
    const li = list.querySelectorAll('li:not(.phase)')[cur] as HTMLElement | undefined;
    li?.scrollIntoView({ block: 'nearest' });
  };
  prev.addEventListener('click', () => go(cur - 1));
  next.addEventListener('click', () => go(cur + 1));
  restart.addEventListener('click', () => go(0));
  refresh.addEventListener('click', retrace);
  root.addEventListener('keydown', (e) => {
    const t = e.target as HTMLElement;
    if (t.tagName === 'SELECT' || t.tagName === 'INPUT') return;
    if (e.key === 'ArrowRight') go(cur + 1);
    else if (e.key === 'ArrowLeft') go(cur - 1);
  });

  new IntersectionObserver(
    (entries) => {
      visible = entries.some((e) => e.isIntersecting);
      const s = steps[cur];
      setHighlight(visible && s ? { block: s.block, dir: s.dir } : null);
    },
    { threshold: 0.15 },
  ).observe(root);

  store.on('probe', () => {
    target = null;
    retrace();
  });
  store.on('model', () => {
    cur = 0;
    retrace();
  });
  // A new dataset: the old example and its trace no longer fit the network.
  store.on('dataset', () => {
    target = null;
    trace = null;
    steps = [];
    cur = 0;
    renderSide();
    renderStep();
  });
  store.on('status', updateSnap);
  // Frozen while training runs; follows the network whenever training is paused.
  store.on('weights', () => {
    if (!store.running && trace && trace.version === store.version && trace.step !== store.weightsStep) retrace();
    else updateSnap();
  });
  store.on('frozen', renderStep);
  store.on('mode', renderStep);
  onThemeChange(() => {
    renderSide();
    renderStep();
  });
}
