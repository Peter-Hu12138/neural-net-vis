import './distView.css';
import './dist.css';
import { initialWeights } from '../actions';
import { analysis, isSuperseded } from '../analysis/client';
import type { LayerStatsResult } from '../analysis/layerStats';
import { qqTwoSampleSorted, sortedFinite, sortedFrozen, summarizeFrozen, summarizeSorted, type QQTwoSample, type Summary } from '../analysis/stats';
import { noun } from '../data/datasets';
import type { Block } from '../nn/network';
import { ACTIVATIONS, fmtShape } from '../nn/types';
import { store } from '../store';
import { layerName } from './builder';
import { $, clear, h, int, segmented } from './dom';
import { fitCanvas } from './draw';
import { drawHistogram, drawQQ, fixed, num, share, sig, type HistPlot, type QQPlot } from './qq';
import { isCurrent, syncedSection, type Stamp } from './snapshot';
import { onThemeChange } from './theme';
import { hideTip, showTip } from './tip';

type Quantity = 'weights' | 'z' | 'a' | 'grad';
type Compare = 'normal' | 'init';

/** `short` replaces `label` when the controls fold into a two-column grid on narrow screens. */
const QUANTITIES: { value: Quantity; label: string; short: string }[] = [
  { value: 'weights', label: 'Weights', short: 'Weights' },
  { value: 'z', label: 'Pre-activations', short: 'Pre-act.' },
  { value: 'a', label: 'Activations', short: 'Activations' },
  { value: 'grad', label: 'Weight gradients', short: 'Gradients' },
];

const COMPARES: { value: Compare; label: string; short: string }[] = [
  { value: 'normal', label: 'Normal', short: 'Normal' },
  { value: 'init', label: 'Initial weights', short: 'Initial' },
];

const NOUN: Record<Quantity, string> = { weights: 'weights', z: 'values', a: 'values', grad: 'gradients' };
const QUANTITY_NAME: Record<Quantity, string> = { weights: 'weights', z: 'pre-activations', a: 'activations', grad: 'weight gradients' };

/** Test samples and values per layer for the layerStats job. */
const SAMPLES = 256;
const MAX_VALUES = 20000;
/** Live weight panels redraw at most this often, and never more than one update per 2× its cost. */
const WEIGHTS_MIN_MS = 250;

interface StatItem {
  k: string;
  v: string;
  title?: string;
  wide?: boolean;
}

interface Model {
  title: string;
  detail: string;
  /** The values, finite and sorted ascending (one sort serves the summary and the histogram). */
  values: Float64Array | null;
  summary: Summary | null;
  /** Two-sample comparison against the initial weights (sorted once and cached). */
  compare: { values: Float64Array; qq: QQTwoSample } | null;
  /** n shown in the stats line (sampled count, plus the population it came from). */
  n: string;
  extra: StatItem[];
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
  /** On screen (or within 200 px of it). Off-screen panels are not recomputed or drawn. */
  visible: boolean;
  /** Work owed once the panel is visible: 2 = recompute and draw, 1 = draw. */
  owed: 0 | 1 | 2;
  /** The Q–Q point under the pointer. */
  hover: { series: number; index: number } | null;
  /** Hand the latest plots to the hover handlers. */
  setQQ(qq: QQPlot | null): void;
  setHist(hist: HistPlot | null): void;
}

/** "8×1×3×3" for a conv kernel tensor, "32×784" for a dense matrix. */
const weightShape = (b: Block) => (b.kind === 'conv' ? `${b.spec.filters}×${b.inShape.c}×${b.k}×${b.k}` : `${b.spec.units}×${b.inSize}`);

/** Shape of z and a (before pooling). */
const zShape = (b: Block) => (b.kind === 'conv' ? b.zShape : b.outShape);

/** 1,605,632 → "1.6M"; small counts stay exact. */
const compact = (n: number) => (n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e4 ? `${Math.round(n / 1e3)}k` : int(n));

/** Buttons of a segmented control get a long and a short label; CSS shows one of them. */
function labelButtons(seg: HTMLElement, opts: { value: string; label: string; short: string }[], prefix: string): void {
  seg.querySelectorAll('button').forEach((b, i) => {
    const o = opts[i];
    b.id = `${prefix}-${o.value}`;
    if (o.short === o.label) return;
    clear(b);
    b.setAttribute('aria-label', o.label);
    b.append(h('span', { class: 'seg-long' }, o.label), h('span', { class: 'seg-short' }, o.short));
  });
}

export function mountDistributions(): void {
  const root = $('dist-root');
  let quantity: Quantity = 'weights';
  let compare: Compare = 'normal';
  /** Last layerStats result for this architecture (complete, or the partial sent while dead units are counted). */
  let cache: { stamp: Stamp; result: LayerStatsResult; testX: ArrayLike<number> | null } | null = null;
  /** The layerStats job in flight. */
  let job: { stamp: Stamp; progress: number; testX: ArrayLike<number> | null } | null = null;
  let panels: Panel[] = [];

  const controls = h('div', { class: 'dist-controls' });
  const error = h('p', { class: 'notice dist-error', hidden: true, role: 'status' });
  /** Shown instead of the panels while the architecture in 01 is invalid (the network shown would be the last valid one). */
  const archNote = h('p', { class: 'notice dist-arch', hidden: true, role: 'status' }, 'Fix the architecture in 01 to see its distributions.');
  /** The "flat run" paragraph of the guide depends on the kind of data. */
  const flatRun = h('span');
  const guide = h(
    'div',
    { class: 'dist-guide' },
    h('p', { class: 'sub' }, 'How to read a Q–Q plot'),
    h(
      'p',
      { class: 'hint' },
      h('b', null, 'On the line. '),
      'Each point pairs a quantile of the layer’s values with the same quantile of a normal distribution. Points on the dashed line mean a normal, bell-shaped spread. The line runs through the quartiles, or follows the mean and std when the quartiles almost coincide (most values near 0).',
    ),
    h('p', { class: 'hint' }, h('b', null, 'Ends bend away. '), 'The low end drops below the line and the high end rises above it: heavy tails, with more extreme values than a normal distribution has. The most extreme values are always plotted.'),
    h('p', { class: 'hint' }, h('b', null, 'Flat run. '), flatRun),
    h('p', { class: 'hint' }, h('b', null, 'Against initial weights. '), 'Points on y = x mean training has not changed the distribution. A steeper run means the weights spread out; bent ends mean a few weights grew large.'),
  );
  const grid = h('div', { class: 'dist-grid' });

  const isJob = () => quantity !== 'weights';
  /** The cached result, when it belongs to the current architecture, initialisation and test set. */
  const result = () => (cache && cache.stamp.version === store.version && cache.testX === (store.data?.testX ?? null) ? cache.result : null);
  /** True when the loaded data and the network fit together (a dataset switch updates one, then the other). */
  const ready = () => !!store.data && store.valid && store.data.inputSize === store.net.inputSize && store.data.info.id === store.dataset;
  /** "digits", "images" or "points". */
  const many = () => noun(store.info, 2);

  const sync = syncedSection(root, () => refresh());
  root.append(controls, sync.status, error, archNote, guide, grid);

  const guideText = () => {
    const info = store.info;
    flatRun.textContent =
      info.kind === 'points'
        ? 'Many identical values: ReLU’s exact zeros (“exactly zero” gives the share). Saturated tanh units pile up at −1 and 1 instead, which flattens both ends.'
        : info.image?.shape.c === 1
          ? 'Many identical values: ReLU’s exact zeros, or, in the first conv layer, blank background. Where a whole input patch is 0, z is just the bias. “Exactly zero” and “blank input” give the shares.'
          : 'Many identical values: ReLU’s exact zeros. “Exactly zero” gives their share. Photos have almost no blank patches, so the first conv layer has no flat run of biases.';
  };
  guideText();

  // ── Controls ─────────────────────────────────────────

  const renderControls = () => {
    clear(controls);
    const q = segmented(QUANTITIES, quantity, (v) => setQuantity(v), 'Quantity');
    labelButtons(q, QUANTITIES, 'dist-q');
    const c = segmented(COMPARES, isJob() ? 'normal' : compare, (v) => setCompare(v), 'Compare with');
    labelButtons(c, COMPARES, 'dist-cmp');
    c.querySelectorAll('button').forEach((b, i) => {
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
    if (isJob()) showJob();
    else updateWeights();
  };

  const setCompare = (c: Compare) => {
    compare = c;
    if (!isJob()) updateWeights();
  };

  // ── Data ─────────────────────────────────────────────

  /** Dims the panels while they show an older result than the one being computed. */
  const markBusy = () => {
    grid.classList.toggle('is-busy', isJob() && job !== null && result() !== null && cache!.stamp !== job.stamp);
  };

  /** Puts the status row in step with what job mode shows, and computes when there is nothing to show. */
  const showJob = () => {
    const r = result();
    if (r) sync.done(cache!.stamp);
    if (job) {
      sync.begin();
      sync.setProgress(job.progress);
    } else if (!r) ensureShown();
    invalidate();
  };

  /** Job mode with no result: compute now (the common policy allows a first result even while training). */
  const ensureShown = () => {
    if (isJob() && !result() && !job && ready() && panels.some((p) => p.visible)) runJob();
  };

  const runJob = () => {
    if (job && isCurrent(job.stamp)) return;
    if (!ready()) return;
    const stamp = sync.begin();
    const testX = store.data?.testX ?? null;
    const mine = { stamp, progress: 0, testX };
    job = mine;
    error.hidden = true;
    sync.setProgress(0);
    markBusy();
    invalidate();
    analysis
      .run<LayerStatsResult>('distributions', 'layerStats', { samples: SAMPLES, maxValues: MAX_VALUES }, (pr) => {
        if (job !== mine) return;
        mine.progress = pr.done / Math.max(1, pr.total);
        if (isJob()) sync.setProgress(mine.progress);
        // The distributions are final before the dead-unit scan ends: show them now.
        const partial = pr.partial as LayerStatsResult | undefined;
        if (partial && cache?.stamp !== stamp) {
          cache = { stamp, result: partial, testX };
          markBusy();
          if (isJob()) invalidate();
        }
      })
      .then((r) => {
        if (job !== mine) return;
        job = null;
        cache = { stamp, result: r, testX };
        markBusy();
        if (isJob()) {
          invalidate();
          sync.done(stamp);
        }
      })
      .catch((e: unknown) => {
        if (isSuperseded(e) || job !== mine) return;
        job = null;
        markBusy();
        if (!isJob()) return;
        sync.fail();
        error.textContent = `Could not compute the distributions: ${e instanceof Error ? e.message : String(e)}`;
        error.hidden = false;
        invalidate();
      });
  };

  /** Called by syncedSection (and the Recompute button) when the section needs fresh numbers. */
  function refresh() {
    if (isJob()) runJob();
    else updateWeights();
  }

  // ── Weights (computed on the page, live while training) ─

  let lastAt = -Infinity;
  let lastCost = 0;
  let weightsTimer: ReturnType<typeof setTimeout> | null = null;

  /** Recomputes the visible weight panels from the current weights. */
  function updateWeights() {
    if (weightsTimer) clearTimeout(weightsTimer);
    weightsTimer = null;
    const t0 = performance.now();
    const stamp = sync.begin();
    invalidate();
    sync.done(stamp);
    lastAt = performance.now();
    lastCost = lastAt - t0;
  }

  /**
   * Weights are cheap enough to follow training live, but a layer of 400k weights takes tens of
   * milliseconds to sort: at most one update per WEIGHTS_MIN_MS and per twice the last update's
   * cost, and none while every panel is off screen (they catch up when they scroll in).
   */
  const scheduleWeights = () => {
    if (isJob() || weightsTimer || !panels.some((p) => p.visible)) return;
    const wait = Math.max(WEIGHTS_MIN_MS, 2 * lastCost) - (performance.now() - lastAt);
    if (wait <= 0) updateWeights();
    else
      weightsTimer = setTimeout(() => {
        weightsTimer = null;
        if (!isJob() && !isCurrent(sync.shown)) updateWeights();
      }, wait);
  };

  // ── Models ───────────────────────────────────────────

  const modelFor = (i: number): Model => {
    const blocks = store.net.blocks;
    const b = blocks[i];
    const isOut = i === blocks.length - 1;
    const title = layerName(isOut ? null : b.spec, i);
    const actLabel = ACTIVATIONS.find((a) => a.id === b.spec.act)?.label ?? b.spec.act;
    const base: Model = { title, detail: '', values: null, summary: null, compare: null, n: '', extra: [], empty: '' };

    if (quantity === 'weights') {
      const sorted = sortedFinite(b.W);
      const summary = summarizeSorted(sorted);
      const init = initialWeights[2 * i];
      // The initial weights never change: sorted and summarised once, then cached.
      const initSorted = compare === 'init' && init && init.length === b.W.length ? sortedFrozen(init) : null;
      const cmp = initSorted ? { values: initSorted, qq: qqTwoSampleSorted(sorted, initSorted) } : null;
      return {
        ...base,
        detail: `${weightShape(b)} weights`,
        values: sorted,
        summary,
        compare: cmp,
        n: int(summary.moments.n),
        extra: cmp ? [{ k: 'std at start', v: sig(summarizeFrozen(init).moments.std, 3) }] : [],
      };
    }

    const r = result();
    if (!r) {
      const empty = job ? 'Computing…' : store.data ? 'Not computed yet.' : `Waiting for ${store.info.name} to load…`;
      return { ...base, detail: quantity === 'grad' ? `${weightShape(b)} gradients` : fmtShape(zShape(b)), empty };
    }
    const l = r.layers[i];
    if (quantity === 'grad') {
      const sorted = sortedFinite(l.gW);
      const summary = summarizeSorted(sorted);
      return { ...base, detail: `${weightShape(b)} · mean ∂L/∂W over ${int(r.samples)} ${noun(store.info, r.samples)}`, values: sorted, summary, n: int(summary.moments.n) };
    }
    const values = sortedFinite(quantity === 'z' ? l.z : l.a);
    const summary = summarizeSorted(values);
    const pooled = b.kind === 'conv' && b.spec.pool;
    // The output layer has no activation function: its activations are the logits themselves.
    const detail = isOut ? `${b.z.length} logits` : `${fmtShape(zShape(b))} · ${actLabel}${quantity === 'a' && pooled ? ' · before pooling' : ''}`;
    const extra: StatItem[] = [];
    if (quantity === 'a' && !isOut) extra.push({ k: 'exactly zero', v: share(Math.round(summary.zero * summary.moments.n), summary.moments.n) });
    if (l.blank > 0)
      extra.push({
        k: 'blank input',
        v: share(Math.round(l.blank * l.seen), l.seen),
        title: `Values whose whole input${b.kind === 'conv' ? ' patch' : ''} is 0 (blank background), so z is exactly the bias. They make the flat run near 0.`,
      });
    if (quantity === 'a' && !isOut && l.dead !== null) {
      const unit = b.kind === 'conv' ? 'filter' : 'unit';
      if (!r.complete) {
        extra.push({ k: 'dead units', v: 'counting…', title: `Checking the rest of the ${int(store.data?.testY.length ?? 0)} test ${many()} for ${unit}s that never fire` });
      } else {
        const dead: number[] = [];
        l.activeFraction.forEach((f, u) => f === 0 && dead.push(u + 1));
        extra.push({
          k: 'dead units',
          v: `${l.dead} of ${l.activeFraction.length}`,
          title: dead.length
            ? `${unit === 'filter' ? 'Filters' : 'Units'} that never fired on any of the ${int(r.activityImages)} test ${noun(store.info, r.activityImages)}: ${dead.join(', ')}`
            : `Every ${unit} fired on at least one test ${noun(store.info)}`,
        });
      }
    }
    const n = summary.moments.n < l.seen ? `${int(summary.moments.n)} of ${compact(l.seen)}` : int(summary.moments.n);
    return { ...base, detail, values, summary, n, extra };
  };

  // ── Panels ───────────────────────────────────────────

  const visibility = new IntersectionObserver(
    (entries) => {
      let appeared = false;
      for (const e of entries) {
        const p = panels.find((q) => q.root === e.target);
        if (!p) continue;
        appeared ||= e.isIntersecting && !p.visible;
        p.visible = e.isIntersecting;
      }
      if (!appeared) return;
      // A weight panel that scrolls in after training moved on brings every visible panel up to date.
      if (!isJob() && !isCurrent(sync.shown)) updateWeights();
      else {
        flush();
        ensureShown();
      }
    },
    { rootMargin: '200px 0px' },
  );

  const makePanel = (): Panel => {
    const title = h('h3', { class: 'panel-title' });
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
      visible: false,
      owed: 2,
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
    for (const p of panels) visibility.unobserve(p.root);
    clear(grid);
    panels = Array.from({ length: n }, makePanel);
    for (const p of panels) {
      grid.append(p.root);
      visibility.observe(p.root);
    }
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
          line: s.qq.reference?.line ?? null,
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
        sorted: true,
      }),
    );

    const mo = s.moments;
    const ref = s.qq.reference;
    const items: StatItem[] = [
      { k: 'n', v: m.n, wide: true },
      { k: 'mean', v: sig(mo.mean, 3) },
      { k: 'std', v: sig(mo.std, 3) },
      { k: 'skew', v: fixed(mo.skew, 2) },
      { k: 'ex. kurtosis', v: fixed(mo.excessKurtosis, 2), title: 'Excess kurtosis: 0 for a normal distribution, above 0 for heavier tails' },
      { k: 'PPCC r', v: fixed(s.ppcc, 4), title: 'Probability-plot correlation: how straight the Q–Q plot is. 1 is a perfectly normal shape.' },
      ...m.extra,
    ];
    let lineNote = '';
    if (!m.compare && ref?.from === 'moments') {
      const quartiles = ref.q1 === ref.q3 ? `Q1 and Q3 are both ${num(ref.q1)}` : `Q1 = ${num(ref.q1)} and Q3 = ${num(ref.q3)} almost coincide`;
      items.push({
        k: 'dashed line',
        v: 'mean and std',
        wide: true,
        title: `${quartiles}, so a line through the quartiles would be flat. The dashed line is the normal distribution with the sample’s mean and standard deviation instead.`,
      });
      lineNote = ` ${quartiles}, so the dashed line is the normal with the same mean and standard deviation.`;
    } else if (!m.compare && !ref) {
      items.push({ k: 'dashed line', v: 'none', wide: true, title: 'Every value is the same, so there is no spread to compare with a normal distribution.' });
      lineNote = ' Every value is the same, so there is no reference line.';
    }
    clear(p.stats);
    for (const it of items) p.stats.append(h('span', { class: it.wide ? 'is-wide' : undefined, title: it.title }, h('span', { class: 'k' }, it.k), ' ', h('b', null, it.v)));
    const against = m.compare ? 'against the initial weights' : 'against a normal distribution';
    p.qq.setAttribute('aria-label', `Q–Q plot of ${what} ${against}. Skew ${fixed(mo.skew, 2)}, excess kurtosis ${fixed(mo.excessKurtosis, 2)}, PPCC ${fixed(s.ppcc, 4)}.${lineNote}`);
    p.hist.setAttribute('aria-label', `Histogram of ${what}, from ${num(mo.min)} to ${num(mo.max)}.`);
  };

  /** Does the work owed to every visible panel. */
  function flush() {
    panels.forEach((p, i) => {
      if (!p.visible || !p.owed) return;
      if (p.owed === 2 || !p.model) p.model = modelFor(i);
      drawPanel(p);
      p.owed = 0;
    });
  }

  /** Every panel's numbers are out of date: recompute the visible ones now, the rest when they appear. */
  function invalidate() {
    syncPanels();
    for (const p of panels) p.owed = 2;
    flush();
  }

  // ── Events ───────────────────────────────────────────

  // syncedSection already requests a refresh on these; weights mode also follows training live.
  store.on('weights', scheduleWeights);
  /** While the architecture in 01 is invalid, say so instead of showing the last valid network's panels. */
  function showArch() {
    const bad = !store.valid;
    archNote.hidden = !bad;
    grid.hidden = bad;
  }

  store.on('model', () => {
    showArch();
    // A job for the previous network can only produce a result nobody can show.
    if (job && job.stamp.version !== store.version) {
      analysis.cancel('distributions');
      job = null;
      if (isJob()) sync.fail();
    }
    syncPanels();
    markBusy();
    if (isJob()) invalidate();
    else if (panels.some((p) => p.visible)) updateWeights();
    else for (const p of panels) p.owed = 2;
  });
  store.on('data', () => {
    // A job over another test set (regenerated points, new features) can only produce a result nobody can show.
    if (job && job.testX !== (store.data?.testX ?? null)) {
      analysis.cancel('distributions');
      job = null;
      markBusy();
      if (isJob()) sync.fail();
    }
    if (!isJob()) return;
    invalidate();
    ensureShown();
  });
  store.on('dataset', () => {
    guideText();
    showArch();
  });

  let queued = false;
  const redraw = () => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => {
      queued = false;
      for (const p of panels) p.owed = Math.max(p.owed, 1) as 1 | 2;
      flush();
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
  showArch();
  syncPanels();
  // Nothing is visible before the first intersection callback: draw once so the section is not
  // empty on load; the observer takes over from there.
  for (const p of panels) p.visible = true;
  updateWeights();
  for (const p of panels) p.visible = false;
}
