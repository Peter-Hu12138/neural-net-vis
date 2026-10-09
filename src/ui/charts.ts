import { loading } from '../actions';
import { fixed, niceTicks, share, sig } from '../analysis/stats';
import { noun, type DatasetInfo } from '../data/datasets';
import { store } from '../store';
import { $, clear, h, int, pct } from './dom';
import { fitCanvas } from './draw';
import { classColor, css, onThemeChange, palette, sequential, type RGB } from './theme';
import { hideTip, showTip } from './tip';
import './charts.css';

const MONO = '"IBM Plex Mono", ui-monospace, monospace';
const SANS = 'Archivo, "Helvetica Neue", Arial, sans-serif';

// ── Class names ─────────────────────────────────────────────────────────────

/** Short class names for tight axes and vector labels; the full names go in tooltips. */
const SHORT: Partial<Record<string, string[]>> = {
  fashion: ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot'],
  cifar10: ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
};

/** At most eight characters per class: known abbreviations, else the name cut short. */
export function shortNames(info: DatasetInfo): string[] {
  return SHORT[info.id] ?? info.classes.map((c) => (c.length <= 8 ? c : `${c.slice(0, 7)}.`));
}

/** True when the class names are just the glyphs (MNIST's digits), so a name adds nothing. */
export const namesAreGlyphs = (info: DatasetInfo): boolean => info.classes.every((c, k) => c === info.glyphs[k]);

/** What a class is called in prose: "digit" for MNIST, "class" otherwise. */
export const classWord = (info: DatasetInfo): string => (info.id === 'mnist' ? 'digit' : 'class');

/** Ink or surface, whichever reads better on a filled cell. */
function textOn(rgb: RGB): string {
  const p = palette();
  const lum = (c: RGB) => {
    const f = (v: number) => {
      const s = v / 255;
      return s <= 0.03928 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4);
    };
    return 0.2126 * f(c[0]) + 0.7152 * f(c[1]) + 0.0722 * f(c[2]);
  };
  const contrast = (a: number, b: number) => (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
  const l = lum(rgb);
  return contrast(l, lum(p.rgb.ink)) >= contrast(l, lum(p.rgb.surface)) ? p.ink : p.surface;
}

// ── Line charts ─────────────────────────────────────────────────────────────

type Pt = [number, number];

interface Series {
  name: string;
  pts: Pt[];
  color: string;
  width: number;
  marks?: boolean;
  fmt: (v: number) => string;
}

interface ChartOpts {
  xMax: number;
  xTicks: number[];
  xFmt: (v: number) => string;
  yMin: number;
  yMax: number;
  log?: boolean;
  yFmt: (v: number) => string;
  yTicks: number[];
  empty?: string;
}

interface Drawn {
  L: number;
  pw: number;
  xMax: number;
  series: Series[];
}

/**
 * Epoch axis: tenths for the first epoch, then whole epochs 1-2-5 × 10ⁿ apart, so a point dataset
 * that has run 2,000 epochs reads "0, 500, 1,000, 1,500, 2,000" rather than odd steps.
 */
export function epochAxis(xEnd: number): { xMax: number; ticks: number[]; fmt: (v: number) => string } {
  if (!(xEnd > 1)) return { xMax: 1, ticks: [0, 0.2, 0.4, 0.6, 0.8, 1], fmt: (v) => fixed(v, 1) };
  const step = Math.max(1, niceTicks(0, xEnd, 5).step);
  const xMax = Math.ceil(xEnd / step - 1e-9) * step;
  const ticks: number[] = [];
  for (let k = 0; k * step <= xMax + 1e-9; k++) ticks.push(k * step);
  return { xMax, ticks, fmt: (v) => int(v) };
}

/** Keeps about `max` points (every k-th, plus the last) so long runs draw as fast as short ones. */
export function thin(pts: Pt[], max: number): Pt[] {
  if (pts.length <= max) return pts;
  const k = Math.ceil(pts.length / max);
  const out: Pt[] = [];
  for (let i = 0; i < pts.length; i += k) out.push(pts[i]);
  if (out[out.length - 1] !== pts[pts.length - 1]) out.push(pts[pts.length - 1]);
  return out;
}

function lineChart(canvas: HTMLCanvasElement, w: number, hgt: number, series: Series[], o: ChartOpts): Drawn {
  const p = palette();
  const ctx = fitCanvas(canvas, w, hgt);
  const L = 44;
  const R = 12;
  const T = 8;
  const B = 34;
  const pw = w - L - R;
  const ph = hgt - T - B;
  const ty = (v: number) => {
    if (o.log) {
      const a = Math.log10(o.yMin);
      const b = Math.log10(o.yMax);
      return T + ph * (1 - (Math.log10(Math.max(o.yMin, v)) - a) / (b - a));
    }
    return T + ph * (1 - (v - o.yMin) / (o.yMax - o.yMin));
  };
  const tx = (v: number) => L + (pw * v) / o.xMax;

  ctx.font = `400 10px ${MONO}`;
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'right';
  for (const t of o.yTicks) {
    const y = Math.round(ty(t)) + 0.5;
    ctx.strokeStyle = p.hair;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(L, y);
    ctx.lineTo(L + pw, y);
    ctx.stroke();
    ctx.fillStyle = p.muted;
    ctx.fillText(o.yFmt(t), L - 6, y);
  }
  ctx.textBaseline = 'top';
  o.xTicks.forEach((v, i) => {
    const x = Math.round(tx(v)) + 0.5;
    ctx.fillStyle = p.muted;
    // The last label is right-aligned so "1,500" never runs past the plot.
    ctx.textAlign = i === o.xTicks.length - 1 && o.xTicks.length > 1 ? 'right' : i === 0 ? 'left' : 'center';
    ctx.fillText(o.xFmt(v), i === o.xTicks.length - 1 ? Math.min(x + 4, L + pw + R) : i === 0 ? x - 2 : x, T + ph + 6);
    ctx.fillStyle = p.ink;
    ctx.fillRect(x - 0.5, T + ph, 1, 4);
  });
  ctx.fillStyle = p.ink;
  ctx.fillRect(L, T + ph, pw, 1.5);
  ctx.textAlign = 'right';
  ctx.fillStyle = p.muted;
  ctx.fillText('epoch', L + pw + R, T + ph + 6 + 12);

  const drawn: Drawn = { L, pw, xMax: o.xMax, series };
  const anyData = series.some((s) => s.pts.length > 0);
  if (!anyData && o.empty) {
    ctx.fillStyle = p.muted;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.font = `500 13px ${SANS}`;
    ctx.fillText(o.empty, L + pw / 2, T + ph / 2);
    return drawn;
  }
  ctx.save();
  ctx.beginPath();
  ctx.rect(L, T - 2, pw + 2, ph + 4);
  ctx.clip();
  for (const s of series) {
    if (!s.pts.length) continue;
    const pts = thin(s.pts, Math.max(200, Math.round(pw * 1.5)));
    ctx.strokeStyle = s.color;
    ctx.lineWidth = s.width;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    pts.forEach(([x, y], i) => (i ? ctx.lineTo(tx(x), ty(y)) : ctx.moveTo(tx(x), ty(y))));
    ctx.stroke();
    // Marks only while they stay apart; a long run would turn them into a smear.
    if (s.marks && pts.length <= pw / 10) {
      ctx.fillStyle = s.color;
      for (const [x, y] of pts) ctx.fillRect(tx(x) - 2.5, ty(y) - 2.5, 5, 5);
    }
  }
  ctx.restore();
  // The latest value of each series.
  for (const s of series) {
    const last = s.pts[s.pts.length - 1];
    if (!last) continue;
    ctx.fillStyle = s.color;
    ctx.beginPath();
    ctx.arc(tx(last[0]), ty(last[1]), 3.5, 0, Math.PI * 2);
    ctx.fill();
  }
  return drawn;
}

/** A loss for the headline figures: three decimals, or two significant digits once it is tiny (a point dataset learned perfectly). */
export const lossText = (v: number): string => (v >= 0.0095 || v === 0 ? fixed(v, 3) : sig(v, 2));

/** EMA smoothing for the noisy per-batch training curve. */
function smooth(pts: Pt[], a = 0.6): Pt[] {
  let m = NaN;
  return pts.map(([x, y]) => {
    m = Number.isNaN(m) ? y : a * m + (1 - a) * y;
    return [x, m];
  });
}

/** Index of the point whose x is closest to `x` (points sorted by x). */
function nearest(pts: Pt[], x: number): number {
  let lo = 0;
  let hi = pts.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (pts[mid][0] < x) lo = mid;
    else hi = mid;
  }
  return Math.abs(pts[lo][0] - x) <= Math.abs(pts[hi][0] - x) ? lo : hi;
}

// ── Mount ───────────────────────────────────────────────────────────────────

export function mountCharts(): void {
  const root = $('curves');
  const kpi = (label: string, accent = false) => {
    const b = h('b', null, '—');
    return { el: h('div', { class: `kpi${accent ? ' is-accent' : ''}` }, h('span', { class: 'label' }, label), b), b };
  };
  const kTrainLoss = kpi('Train loss');
  const kTrainAcc = kpi('Train acc.');
  const kTestLoss = kpi('Test loss', true);
  const kTestAcc = kpi('Test acc.', true);

  const lossCanvas = h('canvas', { role: 'img', 'aria-label': 'Cross-entropy loss per epoch, training and test' }) as HTMLCanvasElement;
  const accCanvas = h('canvas', { role: 'img', 'aria-label': 'Accuracy per epoch, training and test' }) as HTMLCanvasElement;
  const confCanvas = h('canvas', { role: 'img', 'aria-label': 'Confusion matrix on the test set' }) as HTMLCanvasElement;
  const logBox = h('input', { type: 'checkbox', id: 'loss-log' }) as HTMLInputElement;
  const confusedList = h('div', { class: 'hint conf-list' });
  const key = () => h('span', { class: 'key' }, h('span', null, h('i'), 'Train'), h('span', null, h('i', { class: 'is-test' }), 'Test'));

  const lossBox = h('div', { class: 'canvas-box' }, lossCanvas);
  const accBox = h('div', { class: 'canvas-box' }, accCanvas);
  const confBox = h('div', { class: 'canvas-box' }, confCanvas);
  const confWrap = h('div', { class: 'confusion-wrap' }, confBox, confusedList);
  root.append(
    h('div', { class: 'kpis' }, kTrainLoss.el, kTrainAcc.el, kTestLoss.el, kTestAcc.el),
    h(
      'div',
      { class: 'charts' },
      h(
        'div',
        null,
        h('div', { class: 'chart-head' }, h('p', { class: 'sub' }, 'Cross-entropy loss'), h('span', { class: 'key' }, key(), h('label', { class: 'check', for: 'loss-log' }, logBox, 'Log scale'))),
        lossBox,
      ),
      h('div', null, h('div', { class: 'chart-head' }, h('p', { class: 'sub' }, 'Accuracy'), key()), accBox),
      h('div', null, h('p', { class: 'sub' }, 'Confusion matrix · test set'), confWrap),
    ),
  );

  const mixAccent = (t: number): RGB => {
    const p = palette().rgb;
    return [0, 1, 2].map((i) => p.surface[i] + (p.accent[i] - p.surface[i]) * t) as RGB;
  };

  const drawnCharts = new Map<HTMLCanvasElement, Drawn>();
  /** Section on screen (canvases are only drawn then; the text always follows). */
  let inView = true;
  /** Canvases skipped while off screen, drawn on the way back in. */
  let behind = false;

  const render = () => {
    const draw = inView;
    if (!draw) behind = true;
    const p = palette();
    const info = store.info;
    const pts = store.points;
    const evals = store.evals;
    const lastP = pts[pts.length - 1];
    const lastE = evals[evals.length - 1];
    const trainLoss = smooth(pts.map((q) => [q.epoch, q.loss]));
    const trainAcc = smooth(pts.map((q) => [q.epoch, q.acc]));
    kTrainLoss.b.textContent = lastP ? lossText(trainLoss[trainLoss.length - 1][1]) : '—';
    kTrainAcc.b.textContent = lastP ? pct(trainAcc[trainAcc.length - 1][1]) : '—';
    kTestLoss.b.textContent = lastE ? lossText(lastE.loss) : '—';
    kTestAcc.b.textContent = lastE ? pct(lastE.acc) : '—';

    const w = Math.max(260, lossBox.clientWidth);
    const hgt = Math.round(Math.min(200, Math.max(150, w * 0.36)));
    const xEnd = Math.max(lastP?.epoch ?? 0, lastE?.epoch ?? 0);
    const ax = epochAxis(xEnd);
    const empty = !store.data
      ? `Loading ${info.name}…${loading && loading.id === info.id && loading.total > 1 ? ` ${Math.round((100 * loading.done) / loading.total)}%` : ''}`
      : 'Press ▶ in the bar above to start training';

    const testLoss: Pt[] = evals.map((e) => [e.epoch, e.loss]);
    const allLoss = [...trainLoss.map((q) => q[1]), ...testLoss.map((q) => q[1])];
    const log = logBox.checked;
    let yMax = allLoss.reduce((m, v) => Math.max(m, v), 0.5);
    let yMin = 0;
    let ticks: number[];
    if (log) {
      yMin = Math.pow(10, Math.floor(Math.log10(Math.max(1e-3, allLoss.reduce((m, v) => Math.min(m, v), 1)))));
      yMax = Math.pow(10, Math.ceil(Math.log10(yMax)));
      ticks = [];
      for (let t = yMin; t <= yMax * 1.001; t *= 10) ticks.push(t);
    } else {
      yMax = Math.ceil(yMax * 2) / 2;
      const st = yMax > 2 ? 0.5 : 0.25;
      ticks = [];
      for (let t = 0; t <= yMax + 1e-9; t += st) ticks.push(t);
    }
    const lossFmt = lossText;
    if (draw) drawnCharts.set(
      lossCanvas,
      lineChart(lossCanvas, w, hgt, [
        { name: 'train', pts: trainLoss, color: p.ink, width: 1.4, fmt: lossFmt },
        { name: 'test', pts: testLoss, color: p.accent, width: 2, marks: true, fmt: lossFmt },
      ], { xMax: ax.xMax, xTicks: ax.ticks, xFmt: ax.fmt, yMin, yMax, log, yFmt: (v) => (v >= 1 ? v.toFixed(v % 1 ? 1 : 0) : String(+v.toPrecision(2))), yTicks: ticks, empty }),
    );
    const accFmt = (v: number) => pct(v);
    if (draw) drawnCharts.set(
      accCanvas,
      lineChart(accCanvas, w, hgt, [
        { name: 'train', pts: trainAcc, color: p.ink, width: 1.4, fmt: accFmt },
        { name: 'test', pts: evals.map((e) => [e.epoch, e.acc]), color: p.accent, width: 2, marks: true, fmt: accFmt },
      ], { xMax: ax.xMax, xTicks: ax.ticks, xFmt: ax.fmt, yMin: 0, yMax: 1, yFmt: (v) => `${Math.round(v * 100)}%`, yTicks: [0, 0.25, 0.5, 0.75, 1], empty }),
    );
    const where = xEnd > 0 ? `, ${xEnd < 10 ? fixed(xEnd, 2) : int(xEnd)} epochs so far` : '';
    lossCanvas.dataset.xTicks = ax.ticks.join(',');
    lossCanvas.setAttribute('aria-label', `Cross-entropy loss per epoch, training and test${where}${lastE ? `; latest test loss ${lossText(lastE.loss)}` : ''}`);
    accCanvas.setAttribute('aria-label', `Accuracy per epoch, training and test${where}${lastE ? `; latest test accuracy ${pct(lastE.acc)}` : ''}`);

    renderConfusion(lastE?.confusion ?? null, draw);
  };

  // ── Confusion matrix ──
  interface ConfGeom {
    x: number;
    y: number;
    s: number;
    n: number;
    conf: number[] | null;
  }
  let geom: ConfGeom | null = null;

  /** Paints the matrix: digits get one-glyph labels, ten named classes rotated short names, two to four classes full names with their colour. */
  const drawConfusion = (conf: number[] | null) => {
    const p = palette();
    const info = store.info;
    const N = info.classes.length;
    const names = info.classes;
    const short = shortNames(info);
    const glyphs = namesAreGlyphs(info);
    const big = N <= 4;
    const rotated = !glyphs && !big;
    const avail = confBox.clientWidth || 260;
    const ctx0 = confCanvas.getContext('2d')!;
    let left: number;
    let top: number;
    let size: number;
    let swatch = 0;
    if (glyphs) {
      size = Math.min(260, Math.max(200, avail));
      left = top = 22;
    } else if (rotated) {
      ctx0.font = `500 10px ${MONO}`;
      const lw = Math.ceil(Math.max(...short.map((s) => ctx0.measureText(s).width)));
      size = Math.min(340, Math.max(240, avail));
      left = lw + 10;
      top = lw + 10;
    } else {
      ctx0.font = `500 11px ${SANS}`;
      swatch = 12;
      const lw = Math.ceil(Math.max(...names.map((s) => ctx0.measureText(s).width)));
      size = Math.min(320, Math.max(220, avail));
      left = lw + swatch + 10;
      top = 24;
    }
    const s = Math.floor(((size - left - 4) / N) * 2) / 2;
    const ctx = fitCanvas(confCanvas, left + N * s + 4, top + N * s + 4);
    geom = { x: left, y: top, s, n: N, conf };

    // Axis labels.
    ctx.fillStyle = p.muted;
    if (glyphs) {
      ctx.font = `500 10px ${MONO}`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      for (let k = 0; k < N; k++) {
        ctx.fillText(info.glyphs[k], left + s * (k + 0.5), top / 2);
        ctx.fillText(info.glyphs[k], left / 2, top + s * (k + 0.5));
      }
    } else if (rotated) {
      ctx.font = `500 10px ${MONO}`;
      ctx.textBaseline = 'middle';
      for (let k = 0; k < N; k++) {
        ctx.textAlign = 'right';
        ctx.fillText(short[k], left - 6, top + s * (k + 0.5));
        ctx.save();
        ctx.translate(left + s * (k + 0.5), top - 6);
        ctx.rotate(-Math.PI / 2);
        ctx.textAlign = 'left';
        ctx.fillText(short[k], 0, 0);
        ctx.restore();
      }
    } else {
      ctx.font = `500 11px ${SANS}`;
      ctx.textBaseline = 'middle';
      for (let k = 0; k < N; k++) {
        ctx.fillStyle = classColor(k);
        ctx.fillRect(0, top + s * (k + 0.5) - 4, 8, 8);
        ctx.fillStyle = p.ink2;
        ctx.textAlign = 'left';
        ctx.fillText(names[k], swatch, top + s * (k + 0.5));
        const tw = ctx.measureText(names[k]).width;
        const cx = left + s * (k + 0.5);
        ctx.fillStyle = classColor(k);
        ctx.fillRect(cx - (tw + swatch) / 2, top / 2 - 4, 8, 8);
        ctx.fillStyle = p.ink2;
        ctx.fillText(names[k], cx - (tw + swatch) / 2 + swatch, top / 2);
      }
    }

    // Cells: the diagonal in ink (share of the class recognised), mistakes in red.
    for (let r = 0; r < N; r++) {
      let rowTotal = 0;
      if (conf) for (let c = 0; c < N; c++) rowTotal += conf[r * N + c];
      for (let c = 0; c < N; c++) {
        const n = conf ? conf[r * N + c] : 0;
        const frac = rowTotal ? n / rowTotal : 0;
        let rgb: RGB = p.rgb.surface;
        if (conf && r === c) rgb = sequential(frac);
        else if (conf && n) rgb = mixAccent(Math.min(1, frac * 6));
        ctx.fillStyle = css(rgb);
        ctx.fillRect(left + c * s, top + r * s, s - 1, s - 1);
        if (!conf || !n) continue;
        const cx = left + s * (c + 0.5);
        const cy = top + s * (r + 0.5);
        ctx.fillStyle = textOn(rgb);
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        if (big && s >= 48) {
          ctx.font = `600 ${Math.min(18, Math.round(s / 5))}px ${MONO}`;
          ctx.fillText(int(n), cx, cy - 7);
          ctx.font = `500 10px ${MONO}`;
          ctx.fillText(share(n, rowTotal), cx, cy + 10);
        } else if (s >= 18) {
          const digits = String(n).length;
          ctx.font = `500 ${digits >= 3 && s < 26 ? 8.5 : 10}px ${MONO}`;
          ctx.fillText(String(n), cx, cy + 0.5);
        }
      }
    }
  };

  const renderConfusion = (raw: number[] | null, draw: boolean) => {
    const info = store.info;
    const N = info.classes.length;
    const conf = raw && raw.length === N * N ? raw : null;
    const names = info.classes;
    const glyphs = namesAreGlyphs(info);
    confWrap.classList.toggle('is-wide', !glyphs);
    if (draw) drawConfusion(conf);
    const what = classWord(info);
    const correct = conf ? Array.from({ length: N }, (_, k) => conf[k * N + k]).reduce((a, b) => a + b, 0) : 0;
    const total = conf ? conf.reduce((a, b) => a + b, 0) : 0;
    confCanvas.setAttribute(
      'aria-label',
      `Confusion matrix on the test set, ${N} by ${N}${glyphs ? '' : ` (${names.join(', ')})`}: rows are the true ${what}, columns the prediction${conf ? `; ${int(correct)} of ${int(total)} correct` : '; not computed yet'}.`,
    );

    clear(confusedList);
    if (!conf) {
      confusedList.append(`Rows are the true ${what}, columns the prediction. It fills in after the first evaluation.`);
      return;
    }
    const pairs: { t: number; p: number; n: number }[] = [];
    for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) if (r !== c && conf[r * N + c]) pairs.push({ t: r, p: c, n: conf[r * N + c] });
    pairs.sort((a, b) => b.n - a.n);
    const errors = total - correct;
    const label = (k: number) => (glyphs ? info.glyphs[k] : names[k]);
    confusedList.append(
      h('p', { class: 'sub', style: { marginTop: '0' } }, 'Most confused'),
      ...(pairs.length
        ? pairs.slice(0, 6).map((q) => h('div', { class: 'mono conf-pair' }, `${label(q.t)} → ${label(q.p)}  `, h('span', { class: 'conf-n' }, `${int(q.n)}×`)))
        : [h('div', null, 'No mistakes on the test set.')]),
      h('p', { style: { marginTop: '10px' } }, `Rows: true ${what}. Columns: prediction. ${int(errors)} error${errors === 1 ? '' : 's'} out of ${int(total)} test ${noun(info, total)}.`),
    );
  };

  confCanvas.addEventListener('mousemove', (e) => {
    if (!geom || !geom.conf) return hideTip();
    const r = confCanvas.getBoundingClientRect();
    const c = Math.floor((e.clientX - r.left - geom.x) / geom.s);
    const row = Math.floor((e.clientY - r.top - geom.y) / geom.s);
    const N = geom.n;
    if (c < 0 || c >= N || row < 0 || row >= N) return hideTip();
    const info = store.info;
    const n = geom.conf[row * N + c];
    let rowTotal = 0;
    for (let k = 0; k < N; k++) rowTotal += geom.conf[row * N + k];
    const name = (k: number) => (namesAreGlyphs(info) ? info.glyphs[k] : info.classes[k]);
    showTip(`true ${name(row)} → predicted ${name(c)}\n${int(n)} of the ${int(rowTotal)} test ${noun(info, rowTotal)} labelled ${name(row)} (${share(n, rowTotal)})`, e.clientX, e.clientY);
  });
  confCanvas.addEventListener('mouseleave', hideTip);

  // Hovering a curve reads off the nearest value of each series.
  for (const canvas of [lossCanvas, accCanvas]) {
    canvas.addEventListener('mousemove', (e) => {
      const d = drawnCharts.get(canvas);
      if (!d || !d.series.some((s) => s.pts.length)) return hideTip();
      const r = canvas.getBoundingClientRect();
      const x = e.clientX - r.left;
      if (x < d.L - 4 || x > d.L + d.pw + 4) return hideTip();
      const epoch = Math.max(0, ((x - d.L) / d.pw) * d.xMax);
      const lines = [`epoch ${epoch < 10 ? fixed(epoch, 2) : fixed(epoch, 1)}`];
      for (const s of d.series) {
        if (!s.pts.length) continue;
        const q = s.pts[nearest(s.pts, epoch)];
        lines.push(`${s.name} ${s.fmt(q[1])} (epoch ${q[0] < 10 ? fixed(q[0], 2) : fixed(q[0], 1)})`);
      }
      showTip(lines.join('\n'), e.clientX, e.clientY);
    });
    canvas.addEventListener('mouseleave', hideTip);
  }
  logBox.addEventListener('change', render);

  // Live redraws at most ~8 times a second. Off screen only the text follows (cheap DOM updates the
  // tests and screen readers rely on); the canvases catch up as the section scrolls into view.
  let queued = false;
  let lastAt = -Infinity;
  let timer: ReturnType<typeof setTimeout> | null = null;
  const frame = () => {
    queued = false;
    lastAt = performance.now();
    render();
  };
  const schedule = () => {
    if (queued || timer) return;
    const wait = 120 - (performance.now() - lastAt);
    if (wait > 0) {
      timer = setTimeout(() => {
        timer = null;
        queued = true;
        requestAnimationFrame(frame);
      }, wait);
      return;
    }
    queued = true;
    requestAnimationFrame(frame);
  };
  const soon = () => {
    if (!queued) {
      queued = true;
      requestAnimationFrame(frame);
    }
  };
  new IntersectionObserver(
    (entries) => {
      inView = entries.some((en) => en.isIntersecting);
      // Drawn straight away (not on the next frame), so the section never shows stale curves.
      if (inView && behind) {
        behind = false;
        render();
      }
    },
    { rootMargin: '200px 0px' },
  ).observe(root);
  store.on('metrics', schedule);
  store.on('data', soon);
  store.on('dataset', soon);
  onThemeChange(soon);
  new ResizeObserver(schedule).observe(lossBox);
  soon();
}
