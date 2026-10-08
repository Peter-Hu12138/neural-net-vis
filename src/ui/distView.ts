import './distView.css';
import { initialWeights } from '../actions';
import { analysis, isSuperseded } from '../analysis/client';
import type { LayerStatsResult } from '../analysis/layerStats';
import { moments, qqTwoSample, summarize, type QQTwoSample, type Summary } from '../analysis/stats';
import type { Block } from '../nn/network';
import { ACTIVATIONS, fmtShape } from '../nn/types';
import { store } from '../store';
import { layerName } from './builder';
import { $, clear, h, int, segmented } from './dom';
import { fitCanvas } from './draw';
import { drawHistogram, drawQQ, num, type HistPlot, type QQPlot } from './qq';
import { syncedSection } from './snapshot';
import { onThemeChange } from './theme';
import { hideTip, showTip } from './tip';

type Quantity = 'weights' | 'z' | 'a' | 'grad';
type Compare = 'normal' | 'init';

const QUANTITIES: { value: Quantity; label: string }[] = [
  { value: 'weights', label: 'Weights' },
  { value: 'z', label: 'Pre-activations' },
  { value: 'a', label: 'Activations' },
  { value: 'grad', label: 'Weight gradients' },
];

const COMPARES: { value: Compare; label: string }[] = [
  { value: 'normal', label: 'Normal' },
  { value: 'init', label: 'Initial weights' },
];

const NOUN: Record<Quantity, string> = { weights: 'weights', z: 'values', a: 'values', grad: 'gradients' };
const QUANTITY_NAME: Record<Quantity, string> = { weights: 'weights', z: 'pre-activations', a: 'activations', grad: 'weight gradients' };

/** Test digits and values per layer for the layerStats job. */
const SAMPLES = 256;
const MAX_VALUES = 20000;

interface Model {
  title: string;
  detail: string;
  values: ArrayLike<number> | null;
  summary: Summary | null;
  /** Two-sample comparison against the initial weights. */
  compare: { values: Float32Array; qq: QQTwoSample } | null;
  /** n shown in the stats line (sampled count, plus the population it came from). */
  n: string;
  extra: { k: string; v: string; title?: string }[];
  empty: string;
}

interface Panel {
  root: HTMLElement;
  title: HTMLElement;
  detail: HTMLElement;
  qq: HTMLCanvasElement;
  hist: HTMLCanvasElement;
  stats: HTMLElement;
  model: Model | null;
  /** The Q–Q point under the pointer. */
  hover: { series: number; index: number } | null;
  /** Hand the latest plots to the hover handlers. */
  setQQ(qq: QQPlot | null): void;
  setHist(hist: HistPlot | null): void;
}

const fixed = (v: number, d: number) => (Number.isFinite(v) ? v.toFixed(d).replace(/^-/, '−') : '—');

/** "8×1×3×3" for a conv kernel tensor, "32×784" for a dense matrix. */
const weightShape = (b: Block) => (b.kind === 'conv' ? `${b.spec.filters}×${b.inShape.c}×${b.k}×${b.k}` : `${b.spec.units}×${b.inSize}`);

/** Shape of z and a (before pooling). */
const zShape = (b: Block) => (b.kind === 'conv' ? b.zShape : b.outShape);

/** 1,605,632 → "1.6M"; small counts stay exact. */
const compact = (n: number) => (n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e4 ? `${Math.round(n / 1e3)}k` : int(n));

export function mountDistributions(): void {
  const root = $('dist-root');
  let quantity: Quantity = 'weights';
  let compare: Compare = 'normal';
  let cache: { version: number; step: number; result: LayerStatsResult } | null = null;
  let inflight: string | null = null;
  let panels: Panel[] = [];

  const controls = h('div', { class: 'dist-controls' });
  const barFill = h('span', { style: { width: '0%' } });
  const progress = h(
    'div',
    { class: 'progress dist-progress', hidden: true, role: 'progressbar', 'aria-label': 'Computing distributions', 'aria-valuemin': '0', 'aria-valuemax': '100', 'aria-valuenow': '0' },
    barFill,
  );
  const setBar = (pct: number) => {
    barFill.style.width = `${pct}%`;
    progress.setAttribute('aria-valuenow', String(pct));
  };
  const error = h('p', { class: 'notice dist-error', hidden: true, role: 'status' });
  const guide = h(
    'div',
    { class: 'dist-guide' },
    h('p', { class: 'sub' }, 'How to read a Q–Q plot'),
    h('p', { class: 'hint' }, h('b', null, 'On the line. '), 'Each point pairs a quantile of the layer’s values with the same quantile of a normal distribution. Points on the dashed line mean a normal, bell-shaped spread.'),
    h('p', { class: 'hint' }, h('b', null, 'Ends bend away. '), 'The low end drops below the line and the high end rises above it: heavy tails, with more extreme values than a normal distribution has.'),
    h('p', { class: 'hint' }, h('b', null, 'Flat run at 0. '), 'Many values are exactly zero, which is what ReLU does to every negative input. The histogram shows the same spike.'),
    h('p', { class: 'hint' }, h('b', null, 'Against initial weights. '), 'Points on y = x mean training has not changed the distribution. A steeper run means the weights spread out; bent ends mean a few weights grew large.'),
  );
  const grid = h('div', { class: 'dist-grid' });

  const isJob = () => quantity !== 'weights';
  const jobKey = () => `${store.version}:${store.weightsStep}`;
  const result = () => (cache && cache.version === store.version ? cache.result : null);

  const synced = syncedSection(root, () => refresh());
  root.append(controls, synced.status, progress, error, guide, grid);

  // ── Controls ─────────────────────────────────────────

  const renderControls = () => {
    clear(controls);
    const q = segmented(QUANTITIES, quantity, (v) => setQuantity(v), 'Quantity');
    q.querySelectorAll('button').forEach((b, i) => (b.id = `dist-q-${QUANTITIES[i].value}`));
    const c = segmented(COMPARES, isJob() ? 'normal' : compare, (v) => setCompare(v), 'Compare with');
    c.querySelectorAll('button').forEach((b, i) => {
      b.id = `dist-cmp-${COMPARES[i].value}`;
      if (isJob() && COMPARES[i].value === 'init') {
        b.disabled = true;
        b.title = 'Only weights have initial values to compare with';
      }
    });
    controls.append(h('div', { class: 'field' }, h('span', { class: 'label' }, 'Quantity'), q), h('div', { class: 'field' }, h('span', { class: 'label' }, 'Compare with'), c));
  };

  const setQuantity = (q: Quantity) => {
    if (q === quantity) return;
    quantity = q;
    renderControls();
    error.hidden = true;
    if (!isJob()) {
      progress.hidden = true;
      refresh();
      return;
    }
    const r = result();
    if (r && cache!.step === store.weightsStep) synced.markComputed();
    else if (store.data && inflight !== jobKey()) runJob();
    showProgress();
    update();
  };

  const setCompare = (c: Compare) => {
    compare = c;
    update();
  };

  // ── Data ─────────────────────────────────────────────

  const showProgress = () => {
    const busy = isJob() && inflight !== null;
    progress.hidden = !busy;
    grid.classList.toggle('is-busy', busy && result() !== null);
  };

  const runJob = () => {
    const key = jobKey();
    if (inflight === key) return;
    inflight = key;
    const version = store.version;
    const step = store.weightsStep;
    synced.markComputed();
    error.hidden = true;
    setBar(0);
    showProgress();
    update();
    analysis
      .run<LayerStatsResult>('distributions', 'layerStats', { samples: SAMPLES, maxValues: MAX_VALUES }, (pr) => {
        setBar(Math.round((100 * pr.done) / Math.max(1, pr.total)));
      })
      .then((r) => {
        if (inflight === key) inflight = null;
        cache = { version, step, result: r };
        showProgress();
        if (isJob()) update();
      })
      .catch((e: unknown) => {
        if (isSuperseded(e)) return;
        if (inflight === key) inflight = null;
        showProgress();
        error.textContent = `Could not compute the distributions: ${e instanceof Error ? e.message : String(e)}`;
        error.hidden = false;
        update();
      });
  };

  /** Called by syncedSection (and the Recompute button) when the section needs fresh numbers. */
  function refresh() {
    if (isJob()) runJob();
    else {
      update();
      synced.markComputed();
    }
  }

  const modelFor = (i: number): Model => {
    const blocks = store.net.blocks;
    const b = blocks[i];
    const isOut = i === blocks.length - 1;
    const title = layerName(isOut ? null : b.spec, i);
    const actLabel = ACTIVATIONS.find((a) => a.id === b.spec.act)?.label ?? b.spec.act;
    const base: Model = { title, detail: '', values: null, summary: null, compare: null, n: '', extra: [], empty: '' };

    if (quantity === 'weights') {
      const values = b.W.slice();
      const summary = summarize(values);
      const init = initialWeights[2 * i];
      const cmp = compare === 'init' && init && init.length === values.length ? { values: init, qq: qqTwoSample(values, init) } : null;
      return {
        ...base,
        detail: `${weightShape(b)} weights`,
        values,
        summary,
        compare: cmp,
        n: int(summary.moments.n),
        extra: cmp ? [{ k: 'std at start', v: num(moments(init).std) }] : [],
      };
    }

    const r = result();
    if (!r) {
      const empty = inflight ? 'Computing…' : store.data ? 'Not computed yet' : 'Waiting for MNIST…';
      return { ...base, detail: quantity === 'grad' ? `${weightShape(b)} gradients` : fmtShape(zShape(b)), empty };
    }
    const l = r.layers[i];
    if (quantity === 'grad') {
      const summary = summarize(l.gW);
      return { ...base, detail: `${weightShape(b)} · mean ∂L/∂W over ${r.samples} digits`, values: l.gW, summary, n: int(summary.moments.n) };
    }
    const values = quantity === 'z' ? l.z : l.a;
    const summary = summarize(values);
    const pooled = b.kind === 'conv' && b.spec.pool;
    const detail = isOut
      ? `${b.z.length} logits · no activation function`
      : `${fmtShape(zShape(b))} · ${actLabel}${quantity === 'a' && pooled ? ' · before pooling' : ''}`;
    const extra: Model['extra'] = [];
    if (quantity === 'a' && !isOut) {
      extra.push({ k: 'exactly zero', v: `${(summary.zero * 100).toFixed(1)}%` });
      if (l.dead !== null) {
        const dead: number[] = [];
        l.activeFraction.forEach((f, u) => f === 0 && dead.push(u + 1));
        extra.push({
          k: 'dead units',
          v: `${l.dead} of ${l.activeFraction.length}`,
          title: dead.length
            ? `${b.kind === 'conv' ? 'Filters' : 'Units'} that never fired on ${r.samples} test digits: ${dead.join(', ')}`
            : `Every ${b.kind === 'conv' ? 'filter' : 'unit'} fired on at least one of ${r.samples} test digits`,
        });
      }
    }
    const n = summary.moments.n < l.seen ? `${int(summary.moments.n)} of ${compact(l.seen)}` : int(summary.moments.n);
    return { ...base, detail, values, summary, n, extra };
  };

  // ── Panels ───────────────────────────────────────────

  const makePanel = (): Panel => {
    const title = h('h3');
    const detail = h('span', { class: 'dist-detail' });
    const qq = h('canvas', { role: 'img' }) as HTMLCanvasElement;
    const hist = h('canvas', { role: 'img' }) as HTMLCanvasElement;
    const stats = h('div', { class: 'dist-stats' });
    const panelRoot = h('article', { class: 'dist-panel' }, h('header', { class: 'dist-head' }, title, detail), qq, hist, stats);
    // Hit-test state lives in closure variables: Rollup's tree-shaking folds reads of a const
    // object literal's properties when the writes happen through another reference.
    let qqPlot: QQPlot | null = null;
    let histPlot: HistPlot | null = null;
    const panel: Panel = {
      root: panelRoot,
      title,
      detail,
      qq,
      hist,
      stats,
      model: null,
      hover: null,
      setQQ(q) {
        qqPlot = q;
      },
      setHist(hp) {
        histPlot = hp;
      },
    };
    const local = (c: HTMLCanvasElement, e: MouseEvent) => {
      const r = c.getBoundingClientRect();
      return { x: e.clientX - r.left, y: e.clientY - r.top };
    };
    qq.addEventListener('mousemove', (e) => {
      const plot = qqPlot;
      const { x, y } = local(qq, e);
      const inside = plot && x >= plot.plot.x - 6 && x <= plot.plot.x + plot.plot.w + 6 && y >= plot.plot.y - 6 && y <= plot.plot.y + plot.plot.h + 6;
      // Nearest point anywhere in the plot (no pixel hunting); it is marked while hovered.
      const hit = inside ? plot.nearest(x, y) : null;
      const next = hit ? { series: hit.series, index: hit.index } : null;
      if (next?.index !== panel.hover?.index || next?.series !== panel.hover?.series) {
        panel.hover = next;
        drawQQPart(panel);
      }
      if (hit) showTip(hit.text, e.clientX, e.clientY);
      else hideTip();
    });
    hist.addEventListener('mousemove', (e) => {
      const { x } = local(hist, e);
      const text = histPlot?.at(x) ?? null;
      if (text) showTip(text, e.clientX, e.clientY);
      else hideTip();
    });
    qq.addEventListener('mouseleave', () => {
      hideTip();
      if (panel.hover) {
        panel.hover = null;
        drawQQPart(panel);
      }
    });
    hist.addEventListener('mouseleave', hideTip);
    return panel;
  };

  const syncPanels = () => {
    const n = store.net.blocks.length;
    if (panels.length === n) return;
    clear(grid);
    panels = Array.from({ length: n }, makePanel);
    for (const p of panels) grid.append(p.root);
  };

  const panelWidth = (p: Panel) => Math.max(200, Math.floor(p.root.clientWidth || 252));

  /** The Q–Q canvas alone (also redrawn when the hovered point changes). */
  function drawQQPart(p: Panel) {
    const m = p.model;
    if (!m) return;
    const W = panelWidth(p);
    const qh = Math.round(Math.min(290, Math.max(230, W * 0.98)));
    const ctx = fitCanvas(p.qq, W, qh);
    const rect = { x: 0, y: 0, w: W, h: qh };
    const s = m.summary;
    if (!s || !m.values) {
      p.setQQ(drawQQ(ctx, rect, [], { xTitle: '', yTitle: '', xName: '', empty: m.empty }));
      return;
    }
    if (m.compare) {
      const q = m.compare.qq;
      p.setQQ(
        drawQQ(ctx, rect, [{ x: q.x, y: q.y, probs: q.probs, style: 'filled', name: 'now' }], {
          xTitle: 'Initial weights',
          yTitle: 'Current weights',
          xName: 'initial',
          line: 'identity',
          equalAxes: true,
          highlight: p.hover,
        }),
      );
    } else {
      p.setQQ(
        drawQQ(ctx, rect, [{ x: s.qq.theoretical, y: s.qq.sample, probs: s.qq.probs, style: 'filled', name: 'sample' }], {
          xTitle: 'Normal quantile',
          yTitle: 'Sample quantile',
          xName: 'normal',
          line: s.qq.line,
          highlight: p.hover,
        }),
      );
    }
  }

  const drawPanel = (p: Panel) => {
    const m = p.model;
    if (!m) return;
    p.title.textContent = m.title;
    p.detail.textContent = m.detail;
    drawQQPart(p);
    const W = panelWidth(p);
    const hctx = fitCanvas(p.hist, W, 52);
    const what = `${m.title} ${QUANTITY_NAME[quantity]}`;
    const s = m.summary;
    if (!s || !m.values) {
      p.setHist(null);
      clear(p.stats);
      p.qq.setAttribute('aria-label', `${what}: ${m.empty}`);
      p.hist.setAttribute('aria-label', `Histogram of ${what}: ${m.empty}`);
      return;
    }
    p.setHist(
      drawHistogram(hctx, { x: 0, y: 2, w: W, h: 50 }, m.values, {
        noun: NOUN[quantity],
        compare: m.compare?.values ?? null,
        compareName: 'at start',
      }),
    );

    const mo = s.moments;
    const items: { k: string; v: string; title?: string; wide?: boolean }[] = [
      { k: 'n', v: m.n, wide: true },
      { k: 'mean', v: num(mo.mean) },
      { k: 'std', v: num(mo.std) },
      { k: 'skew', v: fixed(mo.skew, 2) },
      { k: 'ex. kurtosis', v: fixed(mo.excessKurtosis, 2), title: 'Excess kurtosis: 0 for a normal distribution, above 0 for heavier tails' },
      { k: 'PPCC r', v: fixed(s.ppcc, 4), title: 'Probability-plot correlation: how straight the Q–Q plot is. 1 is a perfectly normal shape.' },
      ...m.extra,
    ];
    clear(p.stats);
    for (const it of items) p.stats.append(h('span', { class: it.wide ? 'is-wide' : undefined, title: it.title }, h('span', { class: 'k' }, it.k), ' ', h('b', null, it.v)));
    const against = m.compare ? 'against the initial weights' : 'against a normal distribution';
    p.qq.setAttribute('aria-label', `Q–Q plot of ${what} ${against}. Skew ${fixed(mo.skew, 2)}, excess kurtosis ${fixed(mo.excessKurtosis, 2)}, PPCC ${fixed(s.ppcc, 4)}.`);
    p.hist.setAttribute('aria-label', `Histogram of ${what}, from ${num(mo.min)} to ${num(mo.max)}.`);
  };

  /** Recomputes every panel's numbers for the current quantity, then draws. */
  function update() {
    syncPanels();
    panels.forEach((p, i) => (p.model = modelFor(i)));
    draw();
  }

  const draw = () => {
    for (const p of panels) drawPanel(p);
  };

  // ── Events ───────────────────────────────────────────

  // Live weights, at most four redraws a second, and only while the section is on screen.
  let lastWeights = 0;
  let weightsTimer: ReturnType<typeof setTimeout> | null = null;
  const weightsNow = () => {
    lastWeights = performance.now();
    update();
    synced.markComputed();
  };
  store.on('weights', () => {
    if (isJob() || !synced.visible) return;
    const wait = 250 - (performance.now() - lastWeights);
    if (wait <= 0) weightsNow();
    else if (!weightsTimer)
      weightsTimer = setTimeout(() => {
        weightsTimer = null;
        if (!isJob()) weightsNow();
      }, wait);
  });

  store.on('model', () => {
    syncPanels();
    if (isJob()) update();
    else weightsNow();
  });

  let queued = false;
  const redraw = () => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => {
      queued = false;
      draw();
    });
  };
  onThemeChange(redraw);
  let lastWidth = 0;
  new ResizeObserver(() => {
    if (grid.clientWidth === lastWidth) return;
    lastWidth = grid.clientWidth;
    redraw();
  }).observe(grid);

  renderControls();
  update();
}
