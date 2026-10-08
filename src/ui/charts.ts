import { store } from '../store';
import { $, clear, fmt, h, int, pct } from './dom';
import { fitCanvas } from './draw';
import { css, onThemeChange, palette, sequential, type RGB } from './theme';
import { hideTip, showTip } from './tip';

type Pt = [number, number];

interface Series {
  pts: Pt[];
  color: string;
  width: number;
  marks?: boolean;
}

interface ChartOpts {
  xMax: number;
  yMin: number;
  yMax: number;
  log?: boolean;
  yFmt: (v: number) => string;
  yTicks: number[];
  empty?: string;
}

function lineChart(canvas: HTMLCanvasElement, w: number, hgt: number, series: Series[], o: ChartOpts): void {
  const p = palette();
  const ctx = fitCanvas(canvas, w, hgt);
  const L = 44;
  const R = 10;
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

  ctx.font = '400 10px "IBM Plex Mono", ui-monospace, monospace';
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
  // x ticks: epochs
  const step = o.xMax <= 1 ? 0.2 : o.xMax <= 5 ? 1 : Math.ceil(o.xMax / 5);
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (let v = 0; v <= o.xMax + 1e-9; v += step) {
    const x = Math.round(tx(v)) + 0.5;
    ctx.fillStyle = p.muted;
    ctx.fillText(step < 1 ? v.toFixed(1) : String(Math.round(v)), x, T + ph + 6);
    ctx.fillStyle = p.ink;
    ctx.fillRect(x - 0.5, T + ph, 1, 4);
  }
  ctx.fillStyle = p.ink;
  ctx.fillRect(L, T + ph, pw, 1.5);
  ctx.textAlign = 'right';
  ctx.fillStyle = p.muted;
  ctx.fillText('epoch', L + pw, T + ph + 6 + 11);

  const anyData = series.some((s) => s.pts.length > 0);
  if (!anyData && o.empty) {
    ctx.fillStyle = p.muted;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.font = '500 13px Archivo, "Helvetica Neue", Arial, sans-serif';
    ctx.fillText(o.empty, L + pw / 2, T + ph / 2);
    return;
  }
  ctx.save();
  ctx.beginPath();
  ctx.rect(L, T - 2, pw + 2, ph + 4);
  ctx.clip();
  for (const s of series) {
    if (!s.pts.length) continue;
    ctx.strokeStyle = s.color;
    ctx.lineWidth = s.width;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    s.pts.forEach(([x, y], i) => (i ? ctx.lineTo(tx(x), ty(y)) : ctx.moveTo(tx(x), ty(y))));
    ctx.stroke();
    if (s.marks) {
      ctx.fillStyle = s.color;
      for (const [x, y] of s.pts) ctx.fillRect(tx(x) - 2.5, ty(y) - 2.5, 5, 5);
    }
  }
  ctx.restore();
  // emphasise the latest value of each series
  for (const s of series) {
    const last = s.pts[s.pts.length - 1];
    if (!last) continue;
    ctx.fillStyle = s.color;
    ctx.beginPath();
    ctx.arc(tx(last[0]), ty(last[1]), 3.5, 0, Math.PI * 2);
    ctx.fill();
  }
}

/** EMA smoothing for the noisy per-batch training curve. */
function smooth(pts: Pt[], a = 0.6): Pt[] {
  let m = NaN;
  return pts.map(([x, y]) => {
    m = Number.isNaN(m) ? y : a * m + (1 - a) * y;
    return [x, m];
  });
}

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

  const lossCanvas = h('canvas', { role: 'img', 'aria-label': 'Loss per epoch' }) as HTMLCanvasElement;
  const accCanvas = h('canvas', { role: 'img', 'aria-label': 'Accuracy per epoch' }) as HTMLCanvasElement;
  const confCanvas = h('canvas', { role: 'img', 'aria-label': 'Confusion matrix on the test set' }) as HTMLCanvasElement;
  const logBox = h('input', { type: 'checkbox', id: 'loss-log' }) as HTMLInputElement;
  const confusedList = h('div', { class: 'hint' });
  const key = () => h('span', { class: 'key' }, h('span', null, h('i'), 'Train'), h('span', null, h('i', { class: 'is-test' }), 'Test'));

  const lossBox = h('div', { class: 'canvas-box' }, lossCanvas);
  const accBox = h('div', { class: 'canvas-box' }, accCanvas);
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
      h(
        'div',
        null,
        h('p', { class: 'sub' }, 'Confusion matrix · test set'),
        h('div', { class: 'confusion-wrap' }, h('div', { class: 'canvas-box' }, confCanvas), confusedList),
      ),
    ),
  );

  const mixAccent = (t: number) => {
    const p = palette().rgb;
    return [0, 1, 2].map((i) => p.surface[i] + (p.accent[i] - p.surface[i]) * t) as RGB;
  };

  let confCells: { x: number; y: number; s: number } | null = null;
  let lastConfusion: number[] | null = null;

  const render = () => {
    const p = palette();
    const pts = store.points;
    const evals = store.evals;
    const lastP = pts[pts.length - 1];
    const lastE = evals[evals.length - 1];
    kTrainLoss.b.textContent = lastP ? fmt(smooth(pts.map((q) => [q.epoch, q.loss]))[pts.length - 1][1], 3) : '—';
    kTrainAcc.b.textContent = lastP ? pct(smooth(pts.map((q) => [q.epoch, q.acc]))[pts.length - 1][1]) : '—';
    kTestLoss.b.textContent = lastE ? fmt(lastE.loss, 3) : '—';
    kTestAcc.b.textContent = lastE ? pct(lastE.acc) : '—';

    const w = Math.max(260, lossBox.clientWidth);
    const hgt = Math.round(Math.min(200, Math.max(150, w * 0.36)));
    const xEnd = Math.max(lastP?.epoch ?? 0, lastE?.epoch ?? 0);
    const xMax = xEnd <= 1 ? 1 : Math.ceil(xEnd);
    const empty = store.data ? 'Press ▶ in the bar above to start training' : 'Loading MNIST…';

    const trainLoss = smooth(pts.map((q) => [q.epoch, q.loss]));
    const testLoss: Pt[] = evals.map((e) => [e.epoch, e.loss]);
    const allLoss = [...trainLoss.map((q) => q[1]), ...testLoss.map((q) => q[1])];
    const log = logBox.checked;
    let yMax = Math.max(0.5, ...allLoss);
    let yMin = 0;
    let ticks: number[];
    if (log) {
      yMin = Math.pow(10, Math.floor(Math.log10(Math.max(1e-3, Math.min(...allLoss, 1)))));
      yMax = Math.pow(10, Math.ceil(Math.log10(yMax)));
      ticks = [];
      for (let t = yMin; t <= yMax * 1.001; t *= 10) ticks.push(t);
    } else {
      yMax = Math.ceil(yMax * 2) / 2;
      const st = yMax > 2 ? 0.5 : 0.25;
      ticks = [];
      for (let t = 0; t <= yMax + 1e-9; t += st) ticks.push(t);
    }
    lineChart(lossCanvas, w, hgt, [
      { pts: trainLoss, color: p.ink, width: 1.4 },
      { pts: testLoss, color: p.accent, width: 2, marks: true },
    ], { xMax, yMin, yMax, log, yFmt: (v) => (v >= 1 ? v.toFixed(v % 1 ? 1 : 0) : String(+v.toPrecision(2))), yTicks: ticks, empty });

    lineChart(accCanvas, w, hgt, [
      { pts: smooth(pts.map((q) => [q.epoch, q.acc])), color: p.ink, width: 1.4 },
      { pts: evals.map((e) => [e.epoch, e.acc]), color: p.accent, width: 2, marks: true },
    ], { xMax, yMin: 0, yMax: 1, yFmt: (v) => `${Math.round(v * 100)}%`, yTicks: [0, 0.25, 0.5, 0.75, 1], empty });

    renderConfusion(lastE?.confusion ?? null);
  };

  const renderConfusion = (conf: number[] | null) => {
    lastConfusion = conf;
    const p = palette();
    const size = Math.min(260, Math.max(200, (confCanvas.parentElement?.clientWidth ?? 260)));
    const ctx = fitCanvas(confCanvas, size, size);
    const pad = 22;
    const s = (size - pad - 4) / 10;
    confCells = { x: pad, y: pad, s };
    ctx.font = '500 10px "IBM Plex Mono", ui-monospace, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = p.muted;
    for (let k = 0; k < 10; k++) {
      ctx.fillText(String(k), pad + s * (k + 0.5), pad / 2);
      ctx.fillText(String(k), pad / 2, pad + s * (k + 0.5));
    }
    for (let r = 0; r < 10; r++) {
      const rowTotal = conf ? conf.slice(r * 10, r * 10 + 10).reduce((a, b) => a + b, 0) : 0;
      for (let c = 0; c < 10; c++) {
        const n = conf ? conf[r * 10 + c] : 0;
        const frac = rowTotal ? n / rowTotal : 0;
        let color: string;
        if (!conf) color = p.surface;
        else if (r === c) color = css(sequential(frac));
        else color = n ? css(mixAccent(Math.min(1, frac * 6))) : p.surface;
        ctx.fillStyle = color;
        ctx.fillRect(pad + c * s, pad + r * s, s - 1, s - 1);
        if (conf && n && s >= 18) {
          ctx.fillStyle = r === c ? (frac > 0.5 ? p.surface : p.ink) : p.ink;
          ctx.fillText(String(n), pad + s * (c + 0.5), pad + s * (r + 0.5));
        }
      }
    }

    clear(confusedList);
    if (!conf) {
      confusedList.append('Rows are the true digit, columns the prediction. It fills in after the first evaluation.');
      return;
    }
    const pairs: { t: number; p: number; n: number }[] = [];
    for (let r = 0; r < 10; r++) for (let c = 0; c < 10; c++) if (r !== c && conf[r * 10 + c]) pairs.push({ t: r, p: c, n: conf[r * 10 + c] });
    pairs.sort((a, b) => b.n - a.n);
    const total = conf.reduce((a, b) => a + b, 0);
    const errors = pairs.reduce((a, b) => a + b.n, 0);
    confusedList.append(
      h('p', { class: 'sub', style: { marginTop: '0' } }, 'Most confused'),
      ...pairs.slice(0, 6).map((q) => h('div', { class: 'mono', style: { color: 'var(--ink)' } }, `${q.t} → ${q.p}  `, h('span', { style: { color: 'var(--muted)' } }, `${q.n}×`))),
      h('p', { style: { marginTop: '10px' } }, `Rows: true digit. Columns: prediction. ${int(errors)} errors out of ${int(total)} test digits.`),
    );
  };

  confCanvas.addEventListener('mousemove', (e) => {
    if (!confCells || !lastConfusion) return;
    const r = confCanvas.getBoundingClientRect();
    const c = Math.floor((e.clientX - r.left - confCells.x) / confCells.s);
    const row = Math.floor((e.clientY - r.top - confCells.y) / confCells.s);
    if (c < 0 || c > 9 || row < 0 || row > 9) return hideTip();
    const n = lastConfusion[row * 10 + c];
    showTip(`true ${row} → predicted ${c}\n${n} digit${n === 1 ? '' : 's'}`, e.clientX, e.clientY);
  });
  confCanvas.addEventListener('mouseleave', hideTip);
  logBox.addEventListener('change', render);

  let queued = false;
  const schedule = () => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => {
      queued = false;
      render();
    });
  };
  store.on('metrics', schedule);
  store.on('data', schedule);
  onThemeChange(schedule);
  new ResizeObserver(schedule).observe(lossBox);
  schedule();
}
