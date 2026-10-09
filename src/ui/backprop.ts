import { applyWeights, setHighlight } from '../actions';
import { derivative, derivativeFormula } from '../nn/activations';
import { Network, argmax, type Block } from '../nn/network';
import { ACTIVATIONS, type Act, type Shape } from '../nn/types';
import { store } from '../store';
import { layerName } from './builder';
import { $, clear, digitChips, h, selectField } from './dom';
import { drawMatrix, fitCanvas, hasNegative, maxAbs, paintThumb, type MatrixMode } from './draw';
import { onThemeChange, palette } from './theme';

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

interface Trace {
  x: Float32Array;
  label: number;
  probs: Float32Array;
  loss: number;
  blocks: BT[];
  step: number;
  version: number;
}

interface Step {
  phase: 'Forward pass' | 'Loss' | 'Backward pass' | 'Update';
  title: string;
  block: number;
  dir: 'fwd' | 'back' | 'update';
  render: (el: HTMLElement) => void;
}

const MONO = '"IBM Plex Mono", ui-monospace, monospace';
const actLabel = (a: Act) => ACTIVATIONS.find((x) => x.id === a)!.label;
const actFormula = (a: Act) => ACTIVATIONS.find((x) => x.id === a)!.formula;

/** Number formatting for the worked examples: fixed decimals with a real minus sign. */
const n = (v: number, d = 3) => (Object.is(v, -0) ? 0 : v).toFixed(d).replace('-', '−');
const ns = (v: number, d = 3, w = 7) => n(v, d).padStart(w);

function makeTrace(net: Network, x: Float32Array, label: number): Trace {
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
  return { x: x.slice(), label, probs: net.probs.slice(), loss, blocks, step: store.weightsStep, version: store.version };
}

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
  c: number;
  y: number;
  x: number;
  h?: number;
  w?: number;
}

/** C maps of H×W. Numbers only when the maps are small enough to read. */
function maps(cap: string, data: Float32Array, s: Shape, o: { signed?: boolean; max?: number; mark?: Mark; px?: number; maxRow?: number } = {}): HTMLElement {
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
  const ctx = fitCanvas(canvas, perRow * (mw + gap) - gap + 2, rows * (mh + gap + lab) - gap + 2);
  const p = palette();
  for (let c = 0; c < C; c++) {
    const x = 1 + (c % perRow) * (mw + gap);
    const y = 1 + Math.floor(c / perRow) * (mh + gap + lab);
    if (lab) {
      ctx.font = `500 9.5px ${MONO}`;
      ctx.fillStyle = o.mark?.c === c ? p.accent : p.muted;
      ctx.textBaseline = 'top';
      ctx.textAlign = 'left';
      ctx.fillText(String(c + 1), x, y);
    }
    drawMatrix(ctx, data, c * H * W, H, W, x, y + lab, cw, ch, mode, max, signed);
    ctx.strokeStyle = p.hair;
    ctx.lineWidth = 1;
    ctx.strokeRect(x - 0.5, y + lab - 0.5, mw + 1, mh + 1);
    if (o.mark && o.mark.c === c) {
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

/** A vector. Ten-way vectors are always printed; longer ones follow the view mode. */
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
  const ctx = fitCanvas(canvas, perRow * cw + 2, rows * (ch + labH + 4) + 2);
  const p = palette();
  for (let r = 0; r < rows; r++) {
    const cnt = Math.min(perRow, len - r * perRow);
    const y = 1 + r * (ch + labH + 4);
    drawMatrix(ctx, data, r * perRow, 1, cnt, 1, y, cw, ch, mode, max, signed);
    ctx.font = `500 9.5px ${MONO}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (let c = 0; c < cnt; c++) {
      const i = r * perRow + c;
      ctx.fillStyle = i === o.mark || i === o.mark2 ? p.accent : p.muted;
      ctx.fillText(o.labels?.[i] ?? String(i + 1), 1 + c * cw + cw / 2, y + ch + 3);
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
function mat(cap: string, data: Float32Array, rows: number, cols: number, o: { markRow?: number; markCol?: number; rowLabels?: string[] } = {}): HTMLElement {
  const max = maxAbs(data) || 1;
  const m = figMode();
  const numbers = m === 'numbers' && rows * cols <= 400;
  const cw = numbers ? 44 : Math.max(1, Math.min(18, Math.floor(640 / cols)));
  const ch = numbers ? 20 : Math.max(2, Math.min(18, Math.floor(180 / rows), Math.max(cw, 3)));
  const mode: MatrixMode = numbers ? 'numbers' : m === 'hinton' && cw >= 4 ? 'hinton' : 'heat';
  const left = 26;
  const canvas = document.createElement('canvas');
  const ctx = fitCanvas(canvas, left + cols * cw + 2, rows * ch + 2);
  const p = palette();
  drawMatrix(ctx, data, 0, rows, cols, left, 1, cw, ch, mode, max, true);
  ctx.font = `500 9.5px ${MONO}`;
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  const stepR = Math.max(1, Math.ceil(11 / ch));
  for (let r = 0; r < rows; r += stepR) {
    ctx.fillStyle = r === o.markRow ? p.accent : p.muted;
    ctx.fillText(o.rowLabels?.[r] ?? String(r + 1), left - 4, 1 + r * ch + ch / 2);
  }
  if (o.markRow !== undefined) {
    ctx.strokeStyle = p.accent;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(left - 0.5, 1 + o.markRow * ch - 0.5, cols * cw + 1, ch + 1);
    if (o.markCol !== undefined) {
      ctx.lineWidth = 2;
      ctx.strokeRect(left + o.markCol * cw - 1.5, 1 + o.markRow * ch - 1.5, cw + 3, ch + 3);
    }
  }
  const note = !numbers && m === 'numbers' ? 'Too many values to print; shown as a heat map.' : undefined;
  return figure(cap, canvas, note);
}

/** All kernels of a conv layer: filters as rows, input channels as columns. */
function kernels(cap: string, W: Float32Array, F: number, C: number, k: number, markF?: number, markC?: number): HTMLElement {
  const max = maxAbs(W) || 1;
  const m = figMode();
  const mode: MatrixMode = m === 'hinton' ? 'hinton' : 'heat';
  const cell = Math.max(2, Math.min(10, Math.floor(460 / (C * k))));
  const kw = k * cell;
  const gap = 4;
  const left = 22;
  const canvas = document.createElement('canvas');
  const ctx = fitCanvas(canvas, left + C * (kw + gap), F * (kw + gap) + 2);
  const p = palette();
  ctx.font = `500 9.5px ${MONO}`;
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let f = 0; f < F; f++) {
    const y = 1 + f * (kw + gap);
    ctx.fillStyle = f === markF ? p.accent : p.muted;
    ctx.fillText(String(f + 1), left - 5, y + kw / 2);
    for (let c = 0; c < C; c++) {
      const x = left + c * (kw + gap);
      drawMatrix(ctx, W, (f * C + c) * k * k, k, k, x, y, cell, cell, mode, max, true);
      if (f === markF && (markC === undefined || c === markC)) {
        ctx.strokeStyle = p.accent;
        ctx.lineWidth = 2;
        ctx.strokeRect(x - 1, y - 1, kw + 2, kw + 2);
      }
    }
  }
  return figure(cap, canvas, 'Rows: filters. Columns: input channels.');
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

const flatShape = (s: Shape) => s.h > 1;
const DIGITS = Array.from({ length: 10 }, (_, k) => String(k));

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

const locate = (i: number, s: Shape) => {
  if (!flatShape(s)) return `[${i + 1}]`;
  const u = unflat(i, s);
  return s.c > 1 ? `[ch ${u.c + 1}, ${u.y}, ${u.x}]` : `[${u.y}, ${u.x}]`;
};

// ── Steps ─────────────────────────────────────────────────────────────────

function buildSteps(t: Trace, ui: { eta: number; onEta: (v: number) => void; apply: () => void }): Step[] {
  const steps: Step[] = [];
  const L = t.blocks.length;

  steps.push({
    phase: 'Forward pass',
    title: 'Input image',
    block: -1,
    dir: 'fwd',
    render: (el) => {
      const shape = t.blocks[0].inShape;
      if (shape.c !== 1 || shape.h !== 28 || shape.w !== 28) {
        // Other inputs: a feature vector (point datasets) or a colour image. (Placeholder until this
        // step is written out for every dataset.)
        const vec = shape.h === 1 && shape.w === 1;
        const feats = store.features;
        const lines = vec
          ? Array.from(t.x, (v, i) => `  ${(vec && feats[i] ? feats[i] : `x${i + 1}`).padEnd(8)} ${ns(v, 3, 8)}`)
          : [`${shape.h}×${shape.w}×${shape.c} values, ${t.x.length.toLocaleString('en-US')} in all.`];
        el.append(
          h('p', { class: 'bp-text' }, vec ? `The point enters as ${t.x.length} numbers, one per input feature.` : `The image enters as ${t.x.length.toLocaleString('en-US')} numbers, one per pixel and colour channel.`),
          work(lines),
        );
        return;
      }
      const lines: string[] = ['Centre crop, rows 10–17, columns 10–17:'];
      for (let y = 10; y < 18; y++) {
        let s = '';
        for (let x = 10; x < 18; x++) s += ns(t.x[y * 28 + x], 2, 6);
        lines.push(s);
      }
      let ink = 0;
      for (const v of t.x) ink += v;
      lines.push('', `Total ink Σx = ${n(ink, 1)} · target digit y = ${t.label}`);
      el.append(
        h('p', { class: 'bp-text' }, `The digit enters as 784 numbers, one per pixel: 0 is paper, 1 is full ink. The training target is the digit ${t.label}.`),
        formula('x ∈ [0, 1]^(28×28)', false),
        row(maps('x · 28×28 input', t.x, { c: 1, h: 28, w: 28 }, { px: 196, signed: false, max: 1 })),
        work(lines),
      );
    },
  });

  for (const bt of t.blocks) {
    if (bt.kind === 'conv') convForwardSteps(steps, bt);
    else denseForwardSteps(steps, bt);
  }

  const out = t.blocks[L - 1];
  const pred = argmax(t.probs);
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
      const lines: (string | [string, boolean])[] = [`  k    logit z     e^(z − ${n(m, 2)})    p = e / ${n(sum, 3)}`];
      for (let k = 0; k < 10; k++) {
        const s = `  ${k}  ${ns(z[k])}     ${ns(e[k], 4, 9)}       ${ns(t.probs[k], 4, 8)}${k === pred ? '   ← prediction' : ''}${k === t.label ? '   ← target' : ''}`;
        lines.push(k === pred || k === t.label ? [s, true] : s);
      }
      el.append(
        h('p', { class: 'bp-text' }, 'Softmax turns the ten logits into probabilities: exponentiate each one, then divide by the total. Subtracting the largest logit first keeps the exponentials small and does not change the result.'),
        formula('p[k] = e^z[k] / Σⱼ e^z[j]', false),
        row(vec('z · logits', out.z, { labels: DIGITS, mark: pred }), vec('p · probabilities', t.probs, { labels: DIGITS, mark: pred, signed: false, max: 1 })),
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
        h('p', { class: 'bp-text' }, 'The loss looks only at the probability given to the correct digit. Confident and right costs almost nothing; confident and wrong costs a lot. Training lowers this number.'),
        formula(`L = −log p[y] = −log p[${t.label}]`, false),
        work([
          `p[${t.label}] = ${n(py, 4)}`,
          ['L = −log(' + n(py, 4) + ') = ' + n(t.loss, 4), true],
          '',
          pred === t.label ? `The network predicts ${pred}: correct.` : `The network predicts ${pred} (p = ${n(t.probs[pred], 3)}): wrong. The target is ${t.label}.`,
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
      const lines: (string | [string, boolean])[] = ['  k       p    one-hot y        δ = p − y'];
      for (let k = 0; k < 10; k++) {
        const s = `  ${k}  ${ns(t.probs[k], 4, 8)}       ${k === t.label ? 1 : 0}          ${ns(out.dZ[k], 4, 8)}`;
        lines.push(k === t.label ? [s, true] : s);
      }
      el.append(
        h('p', { class: 'bp-text' }, 'Backpropagation starts here. For softmax followed by cross-entropy the gradient of the loss with respect to each logit is simply the predicted probability minus the one-hot target. Only the target digit gets a negative gradient: raising its logit would lower the loss.'),
        formula('δ[k] = ∂L/∂z[k] = p[k] − 1[k = y]', true),
        row(vec('δ · ∂L/∂z at the output', out.dZ, { labels: DIGITS, mark: t.label })),
        work(lines),
      );
    },
  });

  for (let i = L - 1; i >= 0; i--) {
    const bt = t.blocks[i];
    if (bt.kind === 'conv') convBackwardSteps(steps, bt, i === 0);
    else denseBackwardSteps(steps, bt, i === 0);
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

function convForwardSteps(steps: Step[], bt: BT): void {
  const { c: C, h: H, w: W } = bt.inShape;
  const F = bt.zShape.c;
  const k = bt.k;
  const p = bt.pad;
  const kk = k * k;
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
      const shown = Math.min(C, 2);
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
          lines.push(`channel ${c + 1}:   input patch${' '.repeat(Math.max(1, k * 7 - 11))}   kernel W[${f + 1}, ${c + 1}]`);
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
        h('p', { class: 'bp-text' }, `Each of the ${F} filters slides a ${k}×${k} kernel over all ${C} input channel${C > 1 ? 's' : ''}. At every position it multiplies the patch under it by the kernel, adds everything up and adds the filter's bias. Zero padding of ${p} keeps the output at ${H}×${W}.`),
        formula(`z[f, y, x] = b[f] + Σ_c Σ_i Σ_j W[f, c, i, j] · x[c, y+i−${p}, x+j−${p}]`, false),
        row(
          maps(`x · input ${H}×${W}×${C}`, bt.x, bt.inShape, { mark: { c: 0, y: y0 - p, x: x0 - p, h: k, w: k } }),
          kernels(`W · ${F}×${C} kernels of ${k}×${k}`, bt.W, F, C, k, f),
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

function denseForwardSteps(steps: Step[], bt: BT): void {
  const N = bt.inShape.c * bt.inShape.h * bt.inShape.w;
  const M = bt.zShape.c;
  const j = bt.isOut ? argmax(bt.z) : argmaxAbs(bt.z);
  steps.push({
    phase: 'Forward pass',
    title: bt.isOut ? 'Output layer: logits' : `${bt.name}: weighted sum`,
    block: bt.index,
    dir: 'fwd',
    render: (el) => {
      const terms = Array.from({ length: N }, (_, i) => ({ i, v: bt.W[j * N + i] * bt.x[i] })).sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
      const top = terms.slice(0, 6);
      const rest = terms.slice(6).reduce((s, q) => s + q.v, 0);
      const unit = bt.isOut ? `digit ${j}` : `unit ${j + 1}`;
      const lines: (string | [string, boolean])[] = [[`${bt.isOut ? 'Logit for' : 'Unit'} ${bt.isOut ? `digit ${j}` : j + 1}: the ${Math.min(6, N)} largest of its ${N} terms`, true]];
      for (const q of top) lines.push(`  W[${bt.isOut ? j : j + 1}, ${q.i + 1}] · x[${q.i + 1}] = ${ns(bt.W[j * N + q.i])} × ${n(bt.x[q.i])} = ${ns(q.v)}`);
      if (N > 6) lines.push(`  … ${N - 6} more terms                                  sum ${ns(rest)}`);
      lines.push(`  bias b[${bt.isOut ? j : j + 1}]                                       ${ns(bt.b[j])}`);
      lines.push([`  z = ${n(bt.z[j])}`, true]);
      const flat = flatShape(bt.inShape);
      el.append(
        h(
          'p',
          { class: 'bp-text' },
          `${flat ? `The ${bt.inShape.h}×${bt.inShape.w}×${bt.inShape.c} input is first flattened into one vector of ${N} numbers. ` : ''}Each of the ${M} ${bt.isOut ? 'output units (one per digit)' : 'units'} multiplies every input by its own weight, adds them up and adds a bias. The highlighted row of W produces ${unit}.`,
        ),
        formula('z[j] = b[j] + Σᵢ W[j, i] · x[i]', false),
        row(
          flat ? maps(`x · input ${bt.inShape.h}×${bt.inShape.w}×${bt.inShape.c}`, bt.x, bt.inShape) : vec(`x · input (${N})`, bt.x),
          mat(`W · ${M}×${N}`, bt.W, M, N, { markRow: j, rowLabels: bt.isOut ? DIGITS : undefined }),
          vec(`z · ${bt.isOut ? 'logits' : `${M} sums`}`, bt.z, { labels: bt.isOut ? DIGITS : undefined, mark: j, signed: true }),
        ),
        work(lines),
      );
    },
  });
  if (!bt.isOut) steps.push(activationForward(bt, bt.zShape));
}

function denseBackwardSteps(steps: Step[], bt: BT, first: boolean): void {
  const N = bt.inShape.c * bt.inShape.h * bt.inShape.w;
  const M = bt.zShape.c;
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
      const lines: (string | [string, boolean])[] = [
        ['Largest weight gradient:', true],
        `  ∂L/∂W[${lab(gj)}, ${gc + 1}] = δ[${lab(gj)}] · x[${gc + 1}] = ${n(bt.dZ[gj], 4)} × ${n(bt.x[gc], 4)} = ${n(bt.gW[gi], 4)}`,
        `  ∂L/∂b[${lab(gj)}] = δ[${lab(gj)}] = ${n(bt.gb[gj], 4)}`,
        '',
        [`Gradient for input ${locate(xi, bt.inShape)}${first ? ' (a pixel)' : ''}:`, true],
      ];
      for (const q of terms.slice(0, 4)) lines.push(`  W[${lab(q.j)}, ${xi + 1}] · δ[${lab(q.j)}] = ${ns(bt.W[q.j * N + xi], 4, 8)} × ${ns(bt.dZ[q.j], 4, 8)} = ${ns(q.v, 4, 8)}`);
      if (M > 4) lines.push(`  … ${M - 4} more terms`);
      lines.push([`  ∂L/∂x = ${n(bt.dX[xi], 4)}`, true]);
      const flat = flatShape(bt.inShape);
      el.append(
        h('p', { class: 'bp-text' }, `With δ known for every unit, three gradients follow. The weight gradient is the outer product of δ and the layer's input, so weights from strongly active inputs into units with large error change most. The bias gradient is δ itself. To continue backwards, δ is sent through the same weights, transposed${first ? '. Here that gives the gradient with respect to the input pixels: a saliency map of which pixels would change the loss most' : ''}.`),
        formula('∂L/∂W[j, i] = δ[j] · x[i]\n∂L/∂b[j] = δ[j]\n∂L/∂x[i] = Σⱼ W[j, i] · δ[j]', true),
        row(
          vec('δ · ∂L/∂z', bt.dZ, { labels: bt.isOut ? DIGITS : undefined, mark: gj, signed: true }),
          mat(`∂L/∂W · ${M}×${N}`, bt.gW, M, N, { markRow: gj, markCol: flat ? undefined : gc, rowLabels: bt.isOut ? DIGITS : undefined }),
          flat
            ? maps(first ? '∂L/∂x · input gradient (saliency)' : `∂L/∂x · to ${bt.inShape.h}×${bt.inShape.w}×${bt.inShape.c}`, bt.dX, bt.inShape, { signed: true, mark: { ...unflat(xi, bt.inShape) }, px: first ? 168 : 84 })
            : vec('∂L/∂x · to the previous layer', bt.dX, { signed: true, mark: xi }),
        ),
        work(lines),
      );
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
          flat ? maps("f′(z) · slope", slope, s, { signed: false }) : vec("f′(z) · slope", slope, { signed: false }),
          flat ? maps('δ = ∂L/∂z', bt.dZ, s, { signed: true }) : vec('δ = ∂L/∂z', bt.dZ, { signed: true }),
        ),
        work(lines),
      );
    },
  };
}

function convBackwardSteps(steps: Step[], bt: BT, first: boolean): void {
  const { c: C, h: H, w: W } = bt.inShape;
  const F = bt.zShape.c;
  const k = bt.k;
  const p = bt.pad;
  const kk = k * k;
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
        [`Largest kernel gradient: ∂L/∂W[${f + 1}, ${c + 1}, ${ki}, ${kj}]`, true],
        `  sums δ × input over all ${H * W} positions; ${terms.length} are non-zero. The largest:`,
      ];
      for (const q of terms.slice(0, 4)) lines.push(`  δ[${f + 1}, ${q.y}, ${q.x}] · x[${c + 1}, ${q.y + ki - p}, ${q.x + kj - p}] = ${ns(q.d, 4, 8)} × ${n(q.v, 3)} = ${ns(q.d * q.v, 4, 8)}`);
      if (terms.length > 4) lines.push(`  … ${terms.length - 4} more`);
      lines.push([`  ∂L/∂W[${f + 1}, ${c + 1}, ${ki}, ${kj}] = ${n(bt.gW[gi], 4)}`, true], '', [`∂L/∂b[${f + 1}] = Σ δ[${f + 1}] = ${n(sb, 4)}`, true]);
      el.append(
        h('p', { class: 'bp-text' }, `The kernel gradient lines the layer's input up with δ: for each kernel tap, add up δ times the input value that tap touched, at every position. The bias gradient adds up δ over its whole map. To continue backwards, δ is convolved with the kernels rotated by 180°${first ? '. For this first layer that yields the gradient with respect to the pixels themselves, a saliency map' : ''}.`),
        formula(`∂L/∂W[f, c, i, j] = Σ_y Σ_x δ[f, y, x] · x[c, y+i−${p}, x+j−${p}]\n∂L/∂b[f] = Σ_y Σ_x δ[f, y, x]\n∂L/∂x[c, y, x] = Σ_f Σ_i Σ_j W[f, c, i, j] · δ[f, y−i+${p}, x−j+${p}]`, true),
        row(
          maps(`δ · ${H}×${W}×${F}`, bt.dZ, bt.zShape, { signed: true, mark: { c: f, y: 0, x: 0, h: H, w: W } }),
          kernels(`∂L/∂W · ${F}×${C}×${k}×${k}`, bt.gW, F, C, k, f, c),
          vec('∂L/∂b', bt.gb, { signed: true, mark: f }),
        ),
        row(maps(first ? '∂L/∂x · input gradient (saliency)' : `∂L/∂x · to ${H}×${W}×${C}`, bt.dX, bt.inShape, { signed: true, px: first ? 168 : 84 })),
        work(lines),
      );
    },
  });
}

function renderUpdate(el: HTMLElement, t: Trace, ui: { eta: number; onEta: (v: number) => void; apply: () => void }): void {
  const eta = ui.eta;
  const next = t.blocks.flatMap((b) => [b.W.map((w, i) => w - eta * b.gW[i]), b.b.map((w, i) => w - eta * b.gb[i])]);
  const probe = new Network(store.arch, 0);
  probe.setWeights(next);
  const after = probe.forward(t.x);
  const lossAfter = -Math.log(Math.max(after[t.label], 1e-12));
  const out = t.blocks[t.blocks.length - 1];
  const gi = argmaxAbs(out.gW);
  const N = out.inShape.c * out.inShape.h * out.inShape.w;
  const lines: (string | [string, boolean])[] = [
    ['One weight in the output layer:', true],
    `  W[${Math.floor(gi / N)}, ${(gi % N) + 1}] ← ${n(out.W[gi], 4)} − ${eta} × ${out.gW[gi] < 0 ? `(${n(out.gW[gi], 4)})` : n(out.gW[gi], 4)} = ${n(out.W[gi] - eta * out.gW[gi], 4)}`,
    '',
    ['Every layer:', true],
    '  layer                 ‖∂L/∂W‖     ‖ΔW‖',
  ];
  for (const b of t.blocks) {
    let g = 0;
    for (const v of b.gW) g += v * v;
    lines.push(`  ${b.name.padEnd(18)}${ns(Math.sqrt(g), 4, 10)}${ns(eta * Math.sqrt(g), 4, 10)}`);
  }
  const before = h('div', { class: 'kpi' }, h('span', { class: 'label' }, 'Loss before'), h('b', null, n(t.loss, 4)), h('span', { class: 'hint' }, `p[${t.label}] = ${n(t.probs[t.label], 3)}`));
  const afterEl = h('div', { class: 'kpi is-accent' }, h('span', { class: 'label' }, 'Loss after'), h('b', null, n(lossAfter, 4)), h('span', { class: 'hint' }, `p[${t.label}] = ${n(after[t.label], 3)}`));
  const applyBtn = h('button', { type: 'button', class: 'btn btn-solid', onclick: ui.apply }, 'Apply to network');
  el.append(
    h('p', { class: 'bp-text' }, 'Every weight takes a small step against its gradient, scaled by the step size η. This walkthrough uses plain SGD on this one example, so you can see its effect directly. During training the trainer averages gradients over a batch and uses the optimizer chosen in the bar above.'),
    formula('W ← W − η · ∂L/∂W        b ← b − η · ∂L/∂b', true),
    h(
      'div',
      { class: 'tensors', style: { alignItems: 'end' } },
      selectField('bp-eta', 'Step size', [0.001, 0.01, 0.03, 0.1, 0.3, 1].map((v) => ({ value: v, label: String(v) })), eta, ui.onEta),
      h('div', { class: 'loss-compare' }, before, afterEl),
      applyBtn,
    ),
    row(
      mat('Output W · before', out.W, 10, N, { rowLabels: DIGITS, markRow: Math.floor(gi / N), markCol: flatShape(out.inShape) ? undefined : gi % N }),
      mat('Output W · after', next[next.length - 2], 10, N, { rowLabels: DIGITS, markRow: Math.floor(gi / N), markCol: flatShape(out.inShape) ? undefined : gi % N }),
    ),
    work(lines),
    h('p', { class: 'hint' }, lossAfter < t.loss ? `One step lowered the loss on this example by ${n(t.loss - lossAfter, 4)}. Apply it to make the change real; training then continues from the new weights.` : 'This step size overshoots: the loss went up. Try a smaller η.'),
  );
}

// ── Mount ─────────────────────────────────────────────────────────────────

export function mountBackprop(): void {
  const root = $('bplab');
  const sampleCanvas = h('canvas') as HTMLCanvasElement;
  const sampleCap = h('div', { class: 'hint' });
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
        h('div', null, h('p', { class: 'sub' }, 'Example'), h('div', { class: 'bp-sample' }, sampleCanvas, sampleCap)),
        h('div', null, h('p', { class: 'sub' }, 'Target digit y'), targetWrap),
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
    if (!probe || probe.x.length !== store.net.inputSize) return;
    const y = target ?? probe.label ?? argmax(store.net.forward(probe.x));
    trace = makeTrace(store.net, probe.x, y);
    steps = buildSteps(trace, {
      get eta() {
        return eta;
      },
      onEta: (v) => {
        eta = v;
        applied = '';
        renderStep();
      },
      apply: () => {
        if (!trace) return;
        const t = trace;
        const nextW = t.blocks.flatMap((b) => [b.W.map((w, i) => w - eta * b.gW[i]), b.b.map((w, i) => w - eta * b.gb[i])]);
        const before = t.loss;
        applyWeights(nextW);
        retrace();
        applied = `Applied. The loss on this example went from ${n(before, 4)} to ${n(trace!.loss, 4)}.`;
        renderStep();
      },
    });
    cur = Math.min(cur, steps.length - 1);
    renderSide();
    renderStep();
  };

  const renderSide = () => {
    const probe = store.probe;
    paintThumb(sampleCanvas, trace?.x ?? new Float32Array(784), 28, 28, 64);
    sampleCap.textContent = probe ? probe.caption : 'Pick a digit in 02 Network or 07 Data, or draw one.';
    clear(targetWrap);
    targetWrap.append(
      digitChips(trace?.label ?? null, (d) => {
        target = d;
        retrace();
      }, 'Target digit'),
    );
    if (probe && probe.label === null && target === null) targetWrap.append(h('p', { class: 'hint', style: { marginTop: '6px' } }, 'Your drawing has no label, so the prediction is used. Pick the digit you meant.'));
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
      snapNote.textContent = 'Waiting for an example.';
      return;
    }
    const stale = trace.version !== store.version || trace.step !== store.weightsStep;
    snapNote.textContent = store.running
      ? `Frozen at training step ${trace.step.toLocaleString('en-US')} while training runs.`
      : stale
        ? `Frozen at step ${trace.step.toLocaleString('en-US')}; the network is now at step ${store.weightsStep.toLocaleString('en-US')}.`
        : `Using the network's weights at step ${trace.step.toLocaleString('en-US')}.`;
    refresh.disabled = !stale;
  };

  const renderStep = () => {
    clear(main);
    const s = steps[cur];
    if (!s || !trace) {
      main.append(h('p', { class: 'hint' }, 'Loading…'));
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
    const body = h('div', { style: { display: 'grid', gap: '16px' } });
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
  store.on('status', updateSnap);
  // Frozen while training runs; follows the network whenever training is paused.
  store.on('weights', () => {
    if (!store.running && trace && trace.version === store.version && trace.step !== store.weightsStep) retrace();
    else updateSnap();
  });
  store.on('mode', renderStep);
  onThemeChange(() => {
    renderSide();
    renderStep();
  });
}
