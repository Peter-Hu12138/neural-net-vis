import './embeddingView.css';
import { setProbe } from '../actions';
import { analysis, isSuperseded } from '../analysis/client';
import { balancedIndices, DEFAULT_ITERATIONS, DEFAULT_N, DEFAULT_PERPLEXITY, layerFeatures, PHASES, projectPca, type EmbedMethod, type EmbedResult, type PcaInfo, type TsnePartial } from '../analysis/embed';
import type { Progress } from '../analysis/protocol';
import { sampleToFloat } from '../data/mnist';
import { Network } from '../nn/network';
import { fmtShape } from '../nn/types';
import { store } from '../store';
import { layerName } from './builder';
import { $, append, clear, fmt, h, int, pct, segmented, selectField } from './dom';
import { fitCanvas, paintThumb } from './draw';
import { syncedSection } from './snapshot';
import { onThemeChange, palette } from './theme';
import { hideTip, showTip } from './tip';

/**
 * Section 11: how one layer arranges a balanced set of 1,000 test digits, flattened to 2-D with
 * PCA (recomputed automatically, cheap and linear) or t-SNE (on request, animated). Every point is
 * drawn as its digit numeral; colour only repeats what the numeral says.
 */

const MONO = '"IBM Plex Mono", ui-monospace, monospace';
const SANS = 'Archivo, "Helvetica Neue", Arial, sans-serif';
const MAX_PLOT = 640;
const FADE = 0.12;
const HIT_RADIUS = 12;
const CHANNEL = 'embedding';
const DIGITS = Array.from({ length: 10 }, (_, d) => String(d));

/** A finished result plus what is needed to place new inputs on it. */
interface Shown {
  res: EmbedResult;
  version: number;
  step: number;
  /** Network copy with the weights the result was computed from (PCA projects the probe with it). */
  net: Network;
}

/** What the scatter currently draws: a finished result or a t-SNE frame mid-run. */
interface View {
  method: EmbedMethod;
  layer: number;
  coords: Float32Array;
  indices: Int32Array;
  labels: Uint8Array;
  preds: Uint8Array | null;
  pca: PcaInfo | null;
}

interface Running {
  id: number;
  method: EmbedMethod;
  layer: number;
  version: number;
  indices: Int32Array;
  labels: Uint8Array;
  frame: TsnePartial | null;
  done: number;
}

/** Plot geometry from the last draw, for hit-testing. */
interface Geo {
  L: number;
  T: number;
  pw: number;
  ph: number;
  /** Screen positions of the points, x0, y0, x1, y1, … */
  xy: Float32Array;
  probe: { x: number; y: number; off: boolean; a: number; b: number } | null;
}

let catCache: string[] | null = null;
/** Digit colours (--cat-0 … --cat-9), read at draw time so they follow the theme. */
function catColours(): string[] {
  if (catCache) return catCache;
  const cs = getComputedStyle(document.documentElement);
  catCache = Array.from({ length: 10 }, (_, d) => cs.getPropertyValue(`--cat-${d}`).trim() || palette().ink);
  return catCache;
}

/** Tick values with a 1–2–5 step that fall inside [lo, hi]. */
function niceTicks(lo: number, hi: number, count: number): { ticks: number[]; step: number } {
  const span = hi - lo;
  if (!(span > 0) || !Number.isFinite(span)) return { ticks: [], step: 1 };
  const raw = span / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const e = raw / mag;
  const step = (e >= 7.5 ? 10 : e >= 3.5 ? 5 : e >= 1.5 ? 2 : 1) * mag;
  const ticks: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) ticks.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return { ticks, step };
}

const tickLabel = (v: number, step: number) => {
  const dec = step >= 1 ? 0 : Math.min(4, Math.ceil(-Math.log10(step) - 1e-9));
  return v.toFixed(dec).replace('-', '−');
};

/** Shared thin progress bar. */
function progressBar(label: string) {
  const fill = h('span', { style: { width: '0%' } });
  const el = h('div', { class: 'progress', role: 'progressbar', 'aria-label': label, 'aria-valuemin': '0', 'aria-valuemax': '100', 'aria-valuenow': '0' }, fill);
  return {
    el,
    set(f: number) {
      const v = Math.max(0, Math.min(1, f));
      fill.style.width = `${(v * 100).toFixed(1)}%`;
      el.setAttribute('aria-valuenow', String(Math.round(v * 100)));
    },
  };
}

/** Keeps "t-SNE" on one line in the section note (it would otherwise break after "t-"). */
function keepTermTogether(note: Element | null, term: string): void {
  if (!note) return;
  for (const node of Array.from(note.childNodes)) {
    const text = node.nodeType === Node.TEXT_NODE ? node.textContent ?? '' : '';
    const at = text.indexOf(term);
    if (at < 0) continue;
    node.replaceWith(text.slice(0, at), h('span', { class: 'embed-nowrap' }, term), text.slice(at + term.length));
    return;
  }
}

export function mountEmbedding(): void {
  const root = $('embed-root');
  root.classList.add('embed');
  keepTermTogether(document.querySelector('#embedding .sec-note'), 't-SNE');

  // ── State ──
  const defaultLayer = () => (store.net.blocks.length >= 2 ? store.net.blocks.length - 2 : -1);
  let layer = defaultLayer();
  let layerVersion = store.version;
  let method: EmbedMethod = 'pca';
  let mistakes = false;
  let focusDigit: number | null = null;
  let hover: number | null = null;
  let hoverProbe = false;
  let running: Running | null = null;
  let error: string | null = null;
  let runId = 0;
  let geo: Geo | null = null;
  const cache: Partial<Record<EmbedMethod, Shown>> = {};
  let probePt: { key: string; a: number; b: number } | null = null;
  /** True while a refresh was asked for by the reader (Recompute, picking t-SNE, a new layer). */
  let manual = false;
  /** Whether the section is on screen (or close), and whether it missed updates while it was not. */
  let onScreen = false;
  let dirty = false;
  let probeDirty = false;

  // ── Elements ──
  const layerSlot = h('div', { class: 'embed-layer' });
  const methodSeg = segmented(
    [
      { value: 'pca', label: 'PCA' },
      { value: 'tsne', label: 't-SNE' },
    ],
    method,
    (v) => setMethod(v),
    'Projection method',
  );
  methodSeg.id = 'embed-method';
  const segButtons = Array.from(methodSeg.querySelectorAll('button'));
  segButtons[0].id = 'embed-method-pca';
  segButtons[1].id = 'embed-method-tsne';
  const mistakeBox = h('input', { type: 'checkbox', id: 'embed-mistakes' }) as HTMLInputElement;
  mistakeBox.addEventListener('change', () => {
    mistakes = mistakeBox.checked;
    renderSide();
    draw();
  });

  const bar = progressBar('Embedding progress');
  const runText = h('span', { class: 'embed-run-text', id: 'embed-run-text', 'aria-live': 'polite' });
  const runLine = h('div', { class: 'embed-run is-idle' }, bar.el, runText);

  const canvas = h('canvas', { id: 'embed-canvas', role: 'img', tabindex: '0', 'aria-label': 'Embedding of test digits' }) as HTMLCanvasElement;
  const plotBox = h('div', { class: 'embed-plot' }, canvas);

  const previewHead = h('p', { class: 'sub' }, 'Current input');
  const previewCanvas = h('canvas', { class: 'embed-preview-img', role: 'img', 'aria-label': 'Preview of the digit' }) as HTMLCanvasElement;
  const previewTitle = h('div', { class: 'embed-preview-title' });
  const previewMeta = h('div', { class: 'embed-preview-meta' });
  const preview = h('div', { class: 'embed-preview', id: 'embed-preview' }, previewCanvas, h('div', { class: 'embed-preview-text' }, previewTitle, previewMeta));

  const chips = h('div', { class: 'chips embed-chips', role: 'group', 'aria-label': 'Highlight a digit' });
  const chipButtons: HTMLButtonElement[] = [];
  for (let d = 0; d < 10; d++) {
    const b = h(
      'button',
      { type: 'button', class: 'chip embed-chip', id: `embed-digit-${d}`, 'aria-pressed': 'false', 'aria-label': `Highlight the ${d}s` },
      h('span', { class: 'embed-swatch', style: { background: `var(--cat-${d})` } }),
      String(d),
    ) as HTMLButtonElement;
    b.addEventListener('click', () => {
      focusDigit = focusDigit === d ? null : d;
      for (const [i, c] of chipButtons.entries()) c.setAttribute('aria-pressed', String(i === focusDigit));
      if (hover !== null && view() && focusDigit !== null && view()!.labels[hover] !== focusDigit) setHover(null);
      draw();
    });
    chipButtons.push(b);
    chips.append(b);
  }
  const keys = h('div', { class: 'embed-keys' });
  const stats = h('div', { class: 'embed-stats', id: 'embed-stats' });
  const methodHint = h('div', { class: 'embed-hints' });

  // ── Sync ──
  const refresh = () => {
    const asked = manual;
    manual = false;
    if (!store.data || !store.valid) return render();
    // t-SNE takes seconds, so it never starts on its own: only on Recompute or a reader's choice.
    if (method === 'tsne' && !asked) return render();
    compute();
  };
  const sync = syncedSection(root, refresh);
  // The Recompute button lives inside the status line: flag its clicks as the reader's request.
  sync.status.addEventListener(
    'click',
    (e) => {
      if (e.target instanceof Element && e.target.closest('button')) manual = true;
    },
    true,
  );
  sync.status.addEventListener('click', () => (manual = false));
  // Progress sits in the status row, so nothing below moves when a run starts or ends.
  sync.status.append(runLine);
  const ask = () => {
    manual = true;
    sync.refreshNow();
    manual = false;
  };

  root.append(
    h(
      'div',
      { class: 'embed-controls' },
      layerSlot,
      h('div', { class: 'field' }, h('span', { class: 'label' }, 'Method'), methodSeg),
      h('label', { class: 'check', for: 'embed-mistakes' }, mistakeBox, 'Mark mistakes'),
    ),
    sync.status,
    h(
      'div',
      { class: 'embed-layout' },
      plotBox,
      h(
        'div',
        { class: 'embed-aside' },
        h('div', null, previewHead, preview),
        h('div', null, h('p', { class: 'sub' }, 'Digits'), chips, h('p', { class: 'hint embed-chip-hint' }, 'Click a digit to highlight it; click again to show all.'), keys),
        stats,
        methodHint,
      ),
    ),
  );

  // ── Helpers ──
  const fresh = (s: Shown | undefined, m: EmbedMethod): s is Shown =>
    !!s && s.res.method === m && s.res.layer === layer && s.version === store.version;
  const current = (): Shown | null => {
    const s = cache[method];
    return fresh(s, method) ? s : null;
  };
  const layerTitle = (l: number) => (l < 0 ? 'the input pixels' : layerName(l < store.net.spec.length ? store.net.spec[l] : null, l));

  /** The points on screen: the live t-SNE frame while it runs, else the finished result. */
  const view = (): View | null => {
    if (running && running.method === method && running.layer === layer && running.version === store.version && running.frame) {
      return { method, layer, coords: running.frame.coords, indices: running.indices, labels: running.labels, preds: null, pca: null };
    }
    const s = current();
    if (!s) return null;
    const r = s.res;
    return { method: r.method, layer: r.layer, coords: r.coords, indices: r.indices, labels: r.labels, preds: r.preds, pca: r.pca ?? null };
  };

  const snapshotNet = () => {
    const net = new Network(store.net.spec, 0);
    net.setWeights(store.net.getWeights());
    return net;
  };

  /** Places the current probe on the PCA map with the result's own weights (main thread, one forward pass). */
  const projectProbe = () => {
    probePt = null;
    const s = cache.pca;
    const p = store.probe;
    if (!s || !s.res.pca || !p || s.version !== store.version) return;
    s.net.forward(p.x);
    const [a, b] = projectPca(layerFeatures(s.net, s.res.layer), s.res.pca);
    probePt = { key: p.key, a, b };
  };

  // ── Computation ──
  const compute = () => {
    const m = method;
    const l = layer;
    const version = store.version;
    const step = store.weightsStep;
    const id = ++runId;
    const indices = balancedIndices(store.data!.testY, DEFAULT_N);
    const labels = Uint8Array.from(indices, (i) => store.data!.testY[i]);
    const net = snapshotNet();
    sync.markComputed();
    running = { id, method: m, layer: l, version, indices, labels, frame: null, done: 0 };
    error = null;
    setHover(null);
    showProgress({ done: 0, total: 1 });
    render();
    analysis
      .run<EmbedResult>(CHANNEL, 'embed', { layer: l, method: m, n: DEFAULT_N, perplexity: DEFAULT_PERPLEXITY, iterations: DEFAULT_ITERATIONS }, (p) => {
        if (running?.id !== id) return;
        const frame = p.partial as TsnePartial | undefined;
        const firstFrame = !!frame && !running.frame;
        const fresher = frame && (!running.frame || frame.iteration !== running.frame.iteration);
        if (frame) running.frame = frame;
        showProgress(p);
        if (fresher && method === m && layer === l) {
          if (hover !== null) setHover(null);
          if (!onScreen) dirty = true;
          else if (firstFrame) render();
          else draw();
        }
      })
      .then((res) => {
        if (running?.id !== id) return;
        running = null;
        hideProgress();
        if (version !== store.version) return render();
        cache[m] = { res, version, step, net };
        if (m === 'pca') projectProbe();
        render();
      })
      .catch((e: unknown) => {
        if (isSuperseded(e)) return;
        if (running?.id === id) running = null;
        hideProgress();
        error = e instanceof Error ? e.message : String(e);
        // Shown in the status row too, since an older map may still be on screen.
        runLine.classList.remove('is-idle');
        bar.el.hidden = true;
        runText.textContent = `The embedding failed: ${error}`;
        render();
      });
  };

  const showProgress = (p: Progress) => {
    const r = running;
    if (!r) return;
    const f = p.total > 0 ? p.done / p.total : 0;
    r.done = f;
    bar.set(f);
    bar.el.hidden = false;
    runLine.classList.remove('is-idle');
    if (r.method === 'tsne' && r.frame) runText.textContent = `Iteration ${r.frame.iteration} / ${DEFAULT_ITERATIONS} · KL ${r.frame.kl.toFixed(2)}`;
    else if (f < (r.method === 'pca' ? PHASES.pca.collect : PHASES.tsne.collect)) runText.textContent = `Reading ${int(r.indices.length)} test digits · ${pct(f, 0)}`;
    else if (r.method === 'pca') runText.textContent = `Finding the two main directions · ${pct(f, 0)}`;
    else runText.textContent = `Measuring each digit’s ${DEFAULT_PERPLEXITY} nearest neighbours · ${pct(f, 0)}`;
  };
  const hideProgress = () => {
    runLine.classList.add('is-idle');
    runText.textContent = '';
    bar.set(0);
  };

  // ── Controls ──
  const buildLayerSelect = () => {
    clear(layerSlot);
    const blocks = store.net.blocks;
    const options = [{ value: -1, label: 'Input pixels · 784' }];
    blocks.forEach((b, i) => {
      const spec = i < store.net.spec.length ? store.net.spec[i] : null;
      options.push({ value: i, label: `${layerName(spec, i)} · ${spec ? fmtShape(b.outShape) : '10 logits'}` });
    });
    layerSlot.append(selectField('embed-layer', 'Layer', options, layer, (v) => setLayer(v)));
  };

  /** Shows a result already computed for these weights, or computes one (the reader asked). */
  const showOrCompute = () => {
    setHover(null);
    const s = current();
    if (s && s.step === store.weightsStep) {
      if (running) {
        analysis.cancel(CHANNEL);
        running = null;
        hideProgress();
      }
      sync.markComputed();
    } else ask();
    render();
  };

  const setLayer = (l: number) => {
    if (l === layer) return;
    layer = l;
    showOrCompute();
  };

  function setMethod(m: EmbedMethod) {
    if (m === method) return;
    method = m;
    showOrCompute();
  }

  // ── Hover, preview and click ──
  const setHover = (i: number | null, probe = false) => {
    hover = i;
    hoverProbe = probe && i === null;
    renderPreview();
  };

  const nearest = (x: number, y: number): { i: number | null; probe: boolean } => {
    const v = view();
    if (!geo || !v) return { i: null, probe: false };
    let best = HIT_RADIUS * HIT_RADIUS;
    let bi: number | null = null;
    const xy = geo.xy;
    for (let s = 0; s < v.labels.length; s++) {
      if (focusDigit !== null && v.labels[s] !== focusDigit) continue;
      const dx = xy[2 * s] - x;
      const dy = xy[2 * s + 1] - y;
      const d2 = dx * dx + dy * dy;
      if (d2 < best) {
        best = d2;
        bi = s;
      }
    }
    if (geo.probe) {
      const dx = geo.probe.x - x;
      const dy = geo.probe.y - y;
      if (dx * dx + dy * dy < Math.min(best, 100)) return { i: null, probe: true };
    }
    return { i: bi, probe: false };
  };

  const pointText = (s: number): string => {
    const v = view()!;
    const i = v.indices[s];
    let t = `Test digit #${i} · label ${v.labels[s]}`;
    if (v.preds) t += ` · predicted ${v.preds[s]}`;
    if (v.method === 'pca') t += `\nPC1 ${fmt(v.coords[2 * s], 2)} · PC2 ${fmt(v.coords[2 * s + 1], 2)}`;
    return t;
  };

  const probeText = () => {
    const p = store.probe;
    if (!p || !geo?.probe) return '';
    return `Current input · ${p.caption}\nPC1 ${fmt(geo.probe.a, 2)} · PC2 ${fmt(geo.probe.b, 2)}${geo.probe.off ? ' (off the chart)' : ''}`;
  };

  const tipAt = (clientX: number, clientY: number) => {
    if (hover !== null) showTip(pointText(hover), clientX, clientY);
    else if (hoverProbe) showTip(probeText(), clientX, clientY);
    else hideTip();
  };

  canvas.addEventListener('pointermove', (e) => {
    const r = canvas.getBoundingClientRect();
    const hit = nearest(e.clientX - r.left, e.clientY - r.top);
    if (hit.i !== hover || hit.probe !== hoverProbe) {
      setHover(hit.i, hit.probe);
      draw();
    }
    tipAt(e.clientX, e.clientY);
  });
  canvas.addEventListener('pointerleave', () => {
    hideTip();
    if (hover !== null || hoverProbe) {
      setHover(null);
      draw();
    }
  });
  const choose = (s: number) => {
    const v = view();
    const d = store.data;
    if (!v || !d) return;
    const i = v.indices[s];
    const label = d.testY[i];
    setProbe({ x: sampleToFloat(d.testX, i), label, caption: `Test digit #${i} · label ${label}`, key: `test:${i}` });
  };
  canvas.addEventListener('click', (e) => {
    const r = canvas.getBoundingClientRect();
    const hit = nearest(e.clientX - r.left, e.clientY - r.top);
    if (hit.i !== null) choose(hit.i);
  });

  // Keyboard: arrows move to the nearest point in that direction, Enter uses it as the input.
  canvas.addEventListener('keydown', (e) => {
    const v = view();
    if (!v || !geo) return;
    const dirs: Record<string, [number, number]> = { ArrowRight: [1, 0], ArrowLeft: [-1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
    const xy = geo.xy;
    const ok = (s: number) => focusDigit === null || v.labels[s] === focusDigit;
    if (e.key in dirs) {
      e.preventDefault();
      let next: number | null = null;
      if (hover === null) {
        // Start from the point nearest the centre.
        let best = Infinity;
        const cx = geo.L + geo.pw / 2;
        const cy = geo.T + geo.ph / 2;
        for (let s = 0; s < v.labels.length; s++) {
          if (!ok(s)) continue;
          const d = (xy[2 * s] - cx) ** 2 + (xy[2 * s + 1] - cy) ** 2;
          if (d < best) {
            best = d;
            next = s;
          }
        }
      } else {
        const [ux, uy] = dirs[e.key];
        const x0 = xy[2 * hover];
        const y0 = xy[2 * hover + 1];
        let best = Infinity;
        for (let s = 0; s < v.labels.length; s++) {
          if (s === hover || !ok(s)) continue;
          const dx = xy[2 * s] - x0;
          const dy = xy[2 * s + 1] - y0;
          const along = dx * ux + dy * uy;
          if (along <= 0.5) continue;
          const score = along + 2 * Math.abs(dx * uy - dy * ux);
          if (score < best) {
            best = score;
            next = s;
          }
        }
      }
      if (next !== null) {
        setHover(next);
        draw();
        const r = canvas.getBoundingClientRect();
        tipAt(r.left + xy[2 * next], r.top + xy[2 * next + 1]);
      }
    } else if ((e.key === 'Enter' || e.key === ' ') && hover !== null) {
      e.preventDefault();
      choose(hover);
    } else if (e.key === 'Escape') {
      setHover(null);
      hideTip();
      draw();
    }
  });
  canvas.addEventListener('blur', () => {
    hideTip();
    if (hover !== null) {
      setHover(null);
      draw();
    }
  });

  const renderPreview = () => {
    const v = view();
    const d = store.data;
    const size = 104;
    if (hover !== null && v && d) {
      const i = v.indices[hover];
      previewHead.textContent = 'Hovered digit';
      paintThumb(previewCanvas, sampleToFloat(d.testX, i), 28, 28, size);
      previewCanvas.hidden = false;
      previewTitle.textContent = `Test digit #${i}`;
      clear(previewMeta);
      const label = v.labels[hover];
      const pred = v.preds ? v.preds[hover] : null;
      append(previewMeta, [
        h('div', null, `Label ${label}`),
        pred !== null ? h('div', null, `Predicted ${pred} `, pred !== label ? h('span', { class: 'tag is-on' }, 'Mistake') : null) : null,
        v.method === 'pca' ? h('div', null, `PC1 ${fmt(v.coords[2 * hover], 2)} · PC2 ${fmt(v.coords[2 * hover + 1], 2)}`) : null,
        h('div', { class: 'embed-preview-act' }, 'Click to use it as the input'),
      ]);
      return;
    }
    const p = store.probe;
    previewHead.textContent = 'Current input';
    clear(previewMeta);
    if (!p) {
      previewCanvas.hidden = true;
      previewTitle.textContent = '';
      previewMeta.append(h('div', null, 'Hover a digit on the map to see it here.'));
      return;
    }
    previewCanvas.hidden = false;
    paintThumb(previewCanvas, p.x, 28, 28, size);
    const parts = p.caption.split(' · ');
    previewTitle.textContent = parts[0];
    const rest = parts.slice(1).join(' · ').replace(/^label /, 'Label ');
    if (rest) previewMeta.append(h('div', null, rest));
    const pcaShown = !!v && v.method === 'pca' && !!probePt && probePt.key === p.key;
    if (pcaShown) previewMeta.append(h('div', null, `PC1 ${fmt(probePt!.a, 2)} · PC2 ${fmt(probePt!.b, 2)}`));
    const marked = pcaShown || (!!v && v.method === 'tsne' && probeIndex(v) !== null);
    previewMeta.append(h('div', { class: 'embed-preview-act' }, marked ? 'Marked on the map with a red cross' : 'Hover a digit on the map to see it here'));
  };

  /** Index of the probe among the plotted points, when it is one of them. */
  const probeIndex = (v: View | null): number | null => {
    const p = store.probe;
    if (!v || !p || !p.key.startsWith('test:')) return null;
    const i = Number(p.key.slice(5));
    let lo = 0;
    let hi = v.indices.length - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (v.indices[mid] === i) return mid;
      if (v.indices[mid] < i) lo = mid + 1;
      else hi = mid - 1;
    }
    return null;
  };
  // ── Side panel ──
  const renderSide = () => {
    const v = view();
    const s = current();
    clear(keys);
    clear(stats);
    clear(methodHint);
    const n = v ? v.labels.length : DEFAULT_N;
    const wrong = v?.preds ? v.preds.reduce((acc, p, i) => acc + (p !== v.labels[i] ? 1 : 0), 0) : null;
    if (mistakes && wrong !== null) keys.append(h('span', { class: 'embed-key' }, h('i', { class: 'embed-ring', 'aria-hidden': 'true' }), `Misclassified · ${int(wrong)} of ${int(n)}`));
    else if (mistakes && running) keys.append(h('span', { class: 'embed-key' }, h('i', { class: 'embed-ring', 'aria-hidden': 'true' }), 'Misclassified · shown when the run ends'));
    if (geo?.probe || (v && probeIndex(v) !== null && v.method === 'tsne')) keys.append(h('span', { class: 'embed-key' }, h('i', { class: 'embed-cross', 'aria-hidden': 'true' }), 'Current input'));
    keys.hidden = !keys.firstChild;

    const stat = (label: string, value: string) => h('div', null, `${label} `, h('b', null, value));
    if (s) {
      const r = s.res;
      stats.append(stat('Digits', `${int(r.indices.length)} · ${int(r.indices.length / 10)} of each`));
      stats.append(stat('Values per digit', `${int(r.dim)} at ${layerTitle(r.layer)}`));
      if (r.method === 'pca' && r.pca) {
        stats.append(stat('Variance shown', `${pct(r.pca.explained[0] + r.pca.explained[1])} (PC1 ${pct(r.pca.explained[0])}, PC2 ${pct(r.pca.explained[1])})`));
      } else if (r.kl !== undefined) {
        stats.append(stat('KL divergence', `${fmt(r.kl, 2)} after ${int(r.iterations)} iterations`));
        stats.append(stat('Perplexity', String(r.perplexity ?? DEFAULT_PERPLEXITY)));
      }
      if (wrong !== null) stats.append(stat('Misclassified', `${int(wrong)} (${pct(wrong / r.indices.length)})`));
    }
    stats.hidden = !stats.firstChild;

    if (method === 'pca') {
      methodHint.append(
        h(
          'p',
          { class: 'hint' },
          'PCA finds the two directions in which this layer’s values vary most and projects every digit onto them. It is a linear map, so distances along the axes are real and any new input can be placed on it: the red cross marks the current input.',
        ),
      );
    } else {
      methodHint.append(
        h(
          'p',
          { class: 'hint' },
          't-SNE moves the digits around until the ones that are neighbours at this layer sit next to each other. It preserves neighbours, not distances: the size of a cluster and the gaps between clusters mean little, so the axes carry no values.',
        ),
      );
      const r = s?.res;
      if (r && r.inputDim !== undefined && r.inputDim < r.dim) {
        methodHint.append(
          h(
            'p',
            { class: 'hint' },
            `This layer has ${int(r.dim)} values per digit. They are first mixed down to ${r.inputDim} random directions, which keeps the distances between digits roughly intact and makes t-SNE many times faster.`,
          ),
        );
      }
    }
    methodHint.append(h('p', { class: 'hint' }, 'Hover a numeral to see the digit; click it to make it the network’s input.'));
  };

  // ── Drawing ──
  const message = (): string | null => {
    if (!store.valid) return 'Fix the architecture above to see its embedding.';
    if (!store.data) return 'Waiting for MNIST to load…';
    if (error) return `The embedding failed: ${error}`;
    if (running && running.method === method && running.layer === layer) return running.method === 'tsne' ? 'Preparing t-SNE…' : 'Computing PCA…';
    if (method === 'tsne') return 't-SNE runs only when you ask. Press Recompute.';
    return 'Not computed yet.';
  };

  function draw(): void {
    const p = palette();
    const cats = catColours();
    const S = Math.max(240, Math.min(MAX_PLOT, plotBox.clientWidth || MAX_PLOT));
    const ctx = fitCanvas(canvas, S, S);
    const v = view();
    const isPca = (v?.method ?? method) === 'pca';
    const L = isPca ? 46 : 8;
    const B = isPca ? 44 : 8;
    const T = 8;
    const R = 8;
    const pw = S - L - R;
    const ph = S - T - B;
    ctx.fillStyle = p.surface;
    ctx.fillRect(L, T, pw, ph);
    ctx.strokeStyle = p.hair;
    ctx.lineWidth = 1;
    ctx.strokeRect(L + 0.5, T + 0.5, pw - 1, ph - 1);

    if (!v) {
      geo = null;
      ctx.fillStyle = p.muted;
      ctx.font = `500 13px ${SANS}`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(message() ?? '', L + pw / 2, T + ph / 2);
      canvas.setAttribute('aria-label', message() ?? 'Embedding');
      return;
    }

    // Equal scale on both axes, centred on the data.
    const n = v.labels.length;
    const c = v.coords;
    let x0 = Infinity;
    let x1 = -Infinity;
    let y0 = Infinity;
    let y1 = -Infinity;
    for (let s = 0; s < n; s++) {
      const x = c[2 * s];
      const y = c[2 * s + 1];
      if (x < x0) x0 = x;
      if (x > x1) x1 = x;
      if (y < y0) y0 = y;
      if (y > y1) y1 = y;
    }
    const pad = 12;
    const scale = Math.min((pw - 2 * pad) / Math.max(x1 - x0, 1e-12), (ph - 2 * pad) / Math.max(y1 - y0, 1e-12));
    const cx = (x0 + x1) / 2;
    const cy = (y0 + y1) / 2;
    const sx = (x: number) => L + pw / 2 + (x - cx) * scale;
    const sy = (y: number) => T + ph / 2 - (y - cy) * scale;

    if (isPca && v.pca) {
      // Ticks on the frame, axis names below and to the left.
      const font = `400 10px ${MONO}`;
      ctx.font = font;
      ctx.fillStyle = p.muted;
      const xs = niceTicks(cx - pw / 2 / scale, cx + pw / 2 / scale, Math.max(3, Math.round(pw / 110)));
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      for (const t of xs.ticks) {
        const x = Math.round(sx(t)) + 0.5;
        if (x < L + 2 || x > L + pw - 2) continue;
        ctx.fillStyle = p.hair;
        ctx.fillRect(x - 0.5, T + ph, 1, 4);
        ctx.fillStyle = p.muted;
        ctx.fillText(tickLabel(t, xs.step), x, T + ph + 6);
      }
      const ys = niceTicks(cy - ph / 2 / scale, cy + ph / 2 / scale, Math.max(3, Math.round(ph / 110)));
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      for (const t of ys.ticks) {
        const y = Math.round(sy(t)) + 0.5;
        if (y < T + 2 || y > T + ph - 2) continue;
        ctx.fillStyle = p.hair;
        ctx.fillRect(L - 4, y - 0.5, 4, 1);
        ctx.fillStyle = p.muted;
        ctx.fillText(tickLabel(t, ys.step), L - 6, y);
      }
      ctx.fillStyle = p.ink2;
      ctx.font = `600 11px ${SANS}`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'alphabetic';
      ctx.fillText(`PC1 · ${pct(v.pca.explained[0])} of variance`, L + pw / 2, S - 6);
      ctx.save();
      ctx.translate(12, T + ph / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textBaseline = 'middle';
      ctx.fillText(`PC2 · ${pct(v.pca.explained[1])}`, 0, 0);
      ctx.restore();
    }

    // Points as numerals. Faded digits first, so the highlighted ones sit on top.
    const xy = new Float32Array(2 * n);
    for (let s = 0; s < n; s++) {
      xy[2 * s] = sx(c[2 * s]);
      xy[2 * s + 1] = sy(c[2 * s + 1]);
    }
    const fs = S < 420 ? 10 : 11;
    ctx.save();
    ctx.beginPath();
    ctx.rect(L, T, pw, ph);
    ctx.clip();
    ctx.font = `500 ${fs}px ${MONO}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const pass = (faded: boolean) => {
      ctx.globalAlpha = faded ? FADE : 1;
      for (let s = 0; s < n; s++) {
        const d = v.labels[s];
        if (focusDigit !== null && (d !== focusDigit) !== faded) continue;
        ctx.fillStyle = cats[d];
        ctx.fillText(DIGITS[d], xy[2 * s], xy[2 * s + 1] + 0.5);
      }
      if (mistakes && v.preds) {
        ctx.strokeStyle = p.accent;
        ctx.lineWidth = 1.25;
        ctx.beginPath();
        for (let s = 0; s < n; s++) {
          if (v.preds[s] === v.labels[s]) continue;
          if (focusDigit !== null && (v.labels[s] !== focusDigit) !== faded) continue;
          ctx.moveTo(xy[2 * s] + 7.5, xy[2 * s + 1]);
          ctx.arc(xy[2 * s], xy[2 * s + 1], 7.5, 0, Math.PI * 2);
        }
        ctx.stroke();
      }
    };
    if (focusDigit !== null) pass(true);
    pass(false);
    ctx.globalAlpha = 1;

    // The current input: projected for PCA; for t-SNE only when it is one of the plotted digits.
    let probe: Geo['probe'] = null;
    if (v.method === 'pca' && probePt && store.probe && probePt.key === store.probe.key) {
      const px = sx(probePt.a);
      const py = sy(probePt.b);
      const off = px < L || px > L + pw || py < T || py > T + ph;
      probe = { x: Math.max(L + 10, Math.min(L + pw - 10, px)), y: Math.max(T + 10, Math.min(T + ph - 10, py)), off, a: probePt.a, b: probePt.b };
    } else if (v.method === 'tsne') {
      const s = probeIndex(v);
      if (s !== null) probe = { x: xy[2 * s], y: xy[2 * s + 1], off: false, a: c[2 * s], b: c[2 * s + 1] };
    }
    if (probe) {
      const { x, y } = probe;
      ctx.strokeStyle = p.surface;
      ctx.lineWidth = 4;
      const arms = () => {
        ctx.beginPath();
        ctx.moveTo(x - 12, y);
        ctx.lineTo(x - 4, y);
        ctx.moveTo(x + 4, y);
        ctx.lineTo(x + 12, y);
        ctx.moveTo(x, y - 12);
        ctx.lineTo(x, y - 4);
        ctx.moveTo(x, y + 4);
        ctx.lineTo(x, y + 12);
        ctx.stroke();
      };
      arms();
      ctx.strokeStyle = p.accent;
      ctx.lineWidth = 2;
      arms();
      const text = probe.off ? 'input, off the chart' : 'input';
      ctx.font = `600 10px ${MONO}`;
      const right = x + 14 + ctx.measureText(text).width < L + pw - 2;
      ctx.textAlign = right ? 'left' : 'right';
      ctx.textBaseline = 'bottom';
      const tx = right ? x + 9 : x - 9;
      const ty = y - 7 < T + 12 ? y + 19 : y - 7;
      ctx.lineWidth = 3;
      ctx.strokeStyle = p.surface;
      ctx.strokeText(text, tx, ty);
      ctx.fillStyle = p.ink;
      ctx.fillText(text, tx, ty);
    }

    // Hovered point: an ink ring and its numeral on top of everything.
    if (hover !== null && hover < n) {
      const x = xy[2 * hover];
      const y = xy[2 * hover + 1];
      ctx.fillStyle = p.surface;
      ctx.beginPath();
      ctx.arc(x, y, 8.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = p.ink;
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.font = `700 ${fs + 1}px ${MONO}`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = cats[v.labels[hover]];
      ctx.fillText(String(v.labels[hover]), x, y + 0.5);
    }
    ctx.restore();

    geo = { L, T, pw, ph, xy, probe };
    const what = `${v.method === 'pca' ? 'PCA' : 't-SNE'} map of ${int(n)} test digits at ${layerTitle(v.layer)}, each drawn as its numeral`;
    canvas.setAttribute(
      'aria-label',
      v.pca ? `${what}. PC1 explains ${pct(v.pca.explained[0])} and PC2 ${pct(v.pca.explained[1])} of the variance.` : `${what}. Use the arrow keys to move between digits and Enter to use one as the input.`,
    );
  }

  function render(): void {
    draw();
    renderSide();
    renderPreview();
    for (const [i, b] of segButtons.entries()) b.setAttribute('aria-pressed', String((i === 0 ? 'pca' : 'tsne') === method));
  }

  // ── Events ──
  store.on('model', () => {
    if (store.version !== layerVersion) {
      layerVersion = store.version;
      layer = defaultLayer();
      if (running) {
        analysis.cancel(CHANNEL);
        running = null;
        hideProgress();
      }
      probePt = null;
      error = null;
    }
    setHover(null);
    buildLayerSelect();
    render();
  });
  // Drawing on the pad changes the probe on every stroke: off screen, catch up later instead.
  store.on('probe', () => {
    if (!onScreen) {
      probeDirty = true;
      dirty = true;
      return;
    }
    projectProbe();
    render();
  });
  new IntersectionObserver(
    (entries) => {
      onScreen = entries.some((e) => e.isIntersecting);
      if (!onScreen || !dirty) return;
      dirty = false;
      if (probeDirty) {
        probeDirty = false;
        projectProbe();
      }
      render();
    },
    { rootMargin: '200px 0px' },
  ).observe(root);
  store.on('data', render);
  onThemeChange(() => {
    catCache = null;
    draw();
    renderPreview();
  });
  let resizeQueued = false;
  new ResizeObserver(() => {
    if (resizeQueued) return;
    resizeQueued = true;
    requestAnimationFrame(() => {
      resizeQueued = false;
      draw();
    });
  }).observe(plotBox);

  buildLayerSelect();
  render();
}
