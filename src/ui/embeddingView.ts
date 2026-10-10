import './embeddingView.css';
import { setProbe, testProbe } from '../actions';
import { analysis, isSuperseded } from '../analysis/client';
import {
  alignPca,
  balancedIndices,
  DEFAULT_ITERATIONS,
  DEFAULT_N,
  DEFAULT_PERPLEXITY,
  layerFeatures,
  niceTicks,
  PHASES,
  projectPca,
  signedValue,
  tickLabel,
  type EmbedMethod,
  type EmbedResult,
  type PcaInfo,
  type TsnePartial,
} from '../analysis/embed';
import type { Progress } from '../analysis/protocol';
import { sampleInput } from '../data/datasets';
import { Network } from '../nn/network';
import { fmtShape } from '../nn/types';
import { store } from '../store';
import { layerName } from './builder';
import { $, append, clear, fmt, h, int, pct, segmented, selectField } from './dom';
import { fitCanvas, paintThumb } from './draw';
import { isCurrent, syncedSection, type Stamp } from './snapshot';
import { onThemeChange, palette, type Palette } from './theme';
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
const FLAT_NOTE = 'Every digit gives the same values at this layer, so there is nothing to spread out. This happens when all of its units are switched off (dead ReLUs) or saturated.';

/** A finished result plus what is needed to place new inputs on it. */
interface Shown {
  res: EmbedResult;
  /** The weights the result was computed from (version, step and revision). */
  stamp: Stamp;
  /** Network copy with those weights (PCA projects the probe with it). */
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
  flat: boolean;
}

interface Running {
  id: number;
  method: EmbedMethod;
  layer: number;
  stamp: Stamp;
  indices: Int32Array;
  labels: Uint8Array;
  frame: TsnePartial | null;
  /** Share of the job done, 0–1. */
  done: number;
}

/** Plot geometry from the last full draw, for hit-testing and for redrawing the marks alone. */
interface Geo {
  L: number;
  T: number;
  pw: number;
  ph: number;
  /** Data → screen: x = L + pw/2 + (a − cx)·scale, y = T + ph/2 − (b − cy)·scale. */
  cx: number;
  cy: number;
  scale: number;
  /** Numeral size in px. */
  fs: number;
  /** Screen positions of the points, x0, y0, x1, y1, … */
  xy: Float32Array;
  probe: { x: number; y: number; off: boolean; a: number; b: number } | null;
}

/** What the cached scatter layer shows; any difference means it must be painted again. */
interface BaseKey {
  coords: Float32Array;
  focus: number | null;
  mistakes: boolean;
  width: number;
  height: number;
  theme: number;
}

/**
 * The current-input cross, shared by 10 and 11: four arms around (x, y), drawn as a 2 px surface
 * halo, then a 1 px ink outline, then the 2 px accent stroke, so it reads by shape on any colour,
 * including class colours close to the accent (UX-5). Arms run from `gap` to `arm` px out.
 */
export function drawInputCross(ctx: CanvasRenderingContext2D, x: number, y: number, p: Palette, arm = 12, gap = 4): void {
  const layer = (colour: string, width: number, extra: number) => {
    ctx.strokeStyle = colour;
    ctx.lineWidth = width;
    ctx.beginPath();
    for (const [dx, dy] of [
      [-1, 0],
      [1, 0],
      [0, -1],
      [0, 1],
    ]) {
      ctx.moveTo(x + dx * (gap - extra), y + dy * (gap - extra));
      ctx.lineTo(x + dx * (arm + extra), y + dy * (arm + extra));
    }
    ctx.stroke();
  };
  ctx.save();
  ctx.lineCap = 'butt';
  layer(p.surface, 8, 3); // the ink outline plus 2 px on every side, ends included
  layer(p.ink, 4, 1); // 1 px around the accent stroke
  layer(p.accent, 2, 0);
  ctx.restore();
}

let catCache: string[] | null = null;
/** Digit colours (--cat-0 … --cat-9), read at draw time so they follow the theme. */
function catColours(): string[] {
  if (catCache) return catCache;
  const cs = getComputedStyle(document.documentElement);
  catCache = Array.from({ length: 10 }, (_, d) => cs.getPropertyValue(`--cat-${d}`).trim() || palette().ink);
  return catCache;
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

/** Splits `text` into lines no wider than `max` in the context's current font. */
function wrapLines(ctx: CanvasRenderingContext2D, text: string, max: number): string[] {
  const lines: string[] = [];
  let line = '';
  for (const word of text.split(' ')) {
    const next = line ? `${line} ${word}` : word;
    if (line && ctx.measureText(next).width > max) {
      lines.push(line);
      line = word;
    } else line = next;
  }
  if (line) lines.push(line);
  return lines;
}

/** What layer index `layer` (−1 = the input) means in `spec`, as a key that compares across edits. */
function layerKey(spec: unknown[], layer: number): string {
  if (layer < 0) return 'input';
  // A hidden layer is the same layer when it and everything before it are unchanged; the output
  // only when the whole network is.
  return layer < spec.length ? JSON.stringify(spec.slice(0, layer + 1)) : `${JSON.stringify(spec)}+output`;
}

export function mountEmbedding(): void {
  const root = $('embed-root');
  root.classList.add('embed');
  keepTermTogether(document.querySelector('#embedding .sec-note'), 't-SNE');

  // ── State ──
  const defaultLayer = () => (store.net.blocks.length >= 2 ? store.net.blocks.length - 2 : -1);
  let layer = defaultLayer();
  /** True once the reader picked a layer; their choice then survives Reset and compatible edits. */
  let layerPicked = false;
  let layerVersion = store.version;
  let layerSpec: unknown[] = structuredClone(store.spec);
  let method: EmbedMethod = 'pca';
  let mistakes = false;
  let focusDigit: number | null = null;
  let hover: number | null = null;
  let hoverProbe = false;
  /** The hover came from the arrow keys (the tooltip then follows the point, not the pointer). */
  let keyed = false;
  /** Last pointer position over the canvas (client coordinates), so hover can follow moving points. */
  let pointer: { x: number; y: number } | null = null;
  /** Whether the shared tooltip currently shows this section's text. */
  let tipOwned = false;
  let running: Running | null = null;
  let error: string | null = null;
  let runId = 0;
  let geo: Geo | null = null;
  const cache: Partial<Record<EmbedMethod, Shown>> = {};
  /** Orientation of the last PCA per layer, so a recomputed map keeps facing the same way. */
  const orientation = new Map<number, Float32Array>();
  let probePt: { key: string; coords: Float32Array; a: number; b: number } | null = null;
  /** True while a refresh was asked for by the reader (Recompute, picking t-SNE, a new layer). */
  let manual = false;
  /** Whether the section is on screen (or close), and whether it missed redraws while it was not. */
  let onScreen = false;
  let dirty = false;
  let probeDirty = false;
  let probeQueued = false;
  let theme = 0;
  /** The scatter without the marks (hover ring, input cross), reused when only a mark moves. */
  const base = document.createElement('canvas');
  let baseKey: BaseKey | null = null;

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

  const canvas = h('canvas', { id: 'embed-canvas', role: 'img', tabindex: '0', 'aria-label': 'Embedding of test digits' }) as HTMLCanvasElement;
  /** Live caption over the map while t-SNE runs (iteration and KL); never moves the layout. */
  const runText = h('div', { class: 'embed-run-text', id: 'embed-run-text', hidden: true });
  const plotBox = h('div', { class: 'embed-plot' }, canvas, runText);

  const previewHead = h('p', { class: 'sub' }, 'Current input');
  const previewCanvas = h('canvas', { class: 'embed-preview-img', role: 'img', 'aria-label': 'Preview of the digit' }) as HTMLCanvasElement;
  const previewTitle = h('div', { class: 'panel-title embed-preview-title' });
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
      draw();
      followPointer();
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
    !!s && s.res.method === m && s.res.layer === layer && s.stamp.version === store.version;
  const current = (): Shown | null => {
    const s = cache[method];
    return fresh(s, method) ? s : null;
  };
  const layerTitle = (l: number) => (l < 0 ? 'the input pixels' : layerName(l < store.net.spec.length ? store.net.spec[l] : null, l));

  /** The points on screen: the live t-SNE frame while it runs, else the finished result. */
  const view = (): View | null => {
    if (running && running.method === method && running.layer === layer && running.stamp.version === store.version && running.frame) {
      return { method, layer, coords: running.frame.coords, indices: running.indices, labels: running.labels, preds: null, pca: null, flat: false };
    }
    const s = current();
    if (!s) return null;
    const r = s.res;
    return { method: r.method, layer: r.layer, coords: r.coords, indices: r.indices, labels: r.labels, preds: r.preds, pca: r.pca ?? null, flat: !!r.flat };
  };

  const snapshotNet = () => {
    const net = new Network(store.net.arch, 0);
    net.setWeights(store.net.getWeights());
    return net;
  };

  /** Places the current probe on the PCA map with the map's own weights (main thread, one forward pass). */
  const projectProbe = () => {
    probePt = null;
    const s = cache.pca;
    const p = store.probe;
    if (!s || !s.res.pca || s.res.flat || !p || s.stamp.version !== store.version) return;
    s.net.forward(p.x);
    const [a, b] = projectPca(layerFeatures(s.net, s.res.layer), s.res.pca);
    probePt = { key: p.key, coords: s.res.coords, a, b };
  };

  const cancelRun = () => {
    if (!running) return;
    analysis.cancel(CHANNEL);
    running = null;
    runText.hidden = true;
    sync.fail();
  };

  // ── Computation ──
  const compute = () => {
    const m = method;
    const l = layer;
    const id = ++runId;
    const indices = balancedIndices(store.data!.testY, DEFAULT_N);
    const labels = Uint8Array.from(indices, (i) => store.data!.testY[i]);
    // The worker gets these same weights: analysis.run copies them synchronously below.
    const net = snapshotNet();
    const stamp = sync.begin();
    running = { id, method: m, layer: l, stamp, indices, labels, frame: null, done: 0 };
    error = null;
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
          if (!onScreen) dirty = true;
          else if (firstFrame) render();
          else {
            draw();
            followPointer();
          }
        }
      })
      .then((res) => {
        if (running?.id !== id) return;
        running = null;
        runText.hidden = true;
        if (stamp.version !== store.version) {
          sync.fail();
          return render();
        }
        if (res.pca && !res.flat) {
          // Keep facing the way the last map of this layer did, so a training step never mirrors it.
          alignPca(res.pca, res.coords, orientation.get(res.layer) ?? null);
          orientation.set(res.layer, res.pca.components.slice());
        }
        cache[m] = { res, stamp, net };
        if (m === 'pca') projectProbe();
        sync.done(stamp);
        render();
      })
      .catch((e: unknown) => {
        // A newer run (or a cached result) replaced this one; it owns the status now.
        if (isSuperseded(e)) return;
        if (running?.id === id) running = null;
        sync.fail();
        error = e instanceof Error ? e.message : String(e);
        render();
      });
  };

  /** The run's phase in words, for the canvas (nothing shown yet) or the caption over an older map. */
  const phaseText = (r: Running): string => {
    if (r.method === 'tsne' && r.frame) return `Iteration ${r.frame.iteration} / ${DEFAULT_ITERATIONS} · KL ${r.frame.kl.toFixed(2)}`;
    if (r.done < (r.method === 'pca' ? PHASES.pca.collect : PHASES.tsne.collect)) return `Reading ${int(r.indices.length)} test digits…`;
    if (r.method === 'pca') return 'Finding the two main directions…';
    return `Measuring each digit’s ${DEFAULT_PERPLEXITY} nearest neighbours…`;
  };

  let lastPhase = '';
  const showProgress = (p: Progress) => {
    const r = running;
    if (!r) return;
    const f = p.total > 0 ? p.done / p.total : 0;
    r.done = f;
    sync.setProgress(f);
    // With nothing on the map yet, the canvas names the phase: repaint it when the phase changes.
    const phase = r.frame ? 'frames' : phaseText(r);
    if (phase !== lastPhase) {
      lastPhase = phase;
      if (!view() && r.method === method && r.layer === layer) {
        if (onScreen) draw();
        else dirty = true;
      }
    }
    updateRunText();
  };

  /** The caption over the map: only for t-SNE (PCA finishes in a moment) and only over a map. */
  const updateRunText = () => {
    const r = running;
    const over = !!r && r.method === 'tsne' && r.method === method && r.layer === layer && !!view();
    if (over) runText.textContent = phaseText(r!);
    else if (error && view()) runText.textContent = `The embedding failed: ${error}`;
    runText.classList.toggle('is-error', !over && !!error);
    runText.hidden = !over && !(error && view());
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
    const s = current();
    if (s && isCurrent(s.stamp)) {
      cancelRun();
      sync.done(s.stamp);
      if (s.res.method === 'pca') projectProbe();
    } else ask();
    render();
  };

  const setLayer = (l: number) => {
    layerPicked = true;
    if (l === layer) return;
    layer = l;
    clearHover();
    showOrCompute();
  };

  function setMethod(m: EmbedMethod) {
    if (m === method) return;
    method = m;
    clearHover();
    showOrCompute();
  }

  // ── Hover, preview and click ──
  const setHover = (i: number | null, probe = false) => {
    hover = i;
    hoverProbe = probe && i === null;
    renderPreview();
  };
  const clearHover = () => {
    if (hover !== null || hoverProbe) setHover(null);
    keyed = false;
    hideOwnTip();
  };

  const nearest = (x: number, y: number): { i: number | null; probe: boolean } => {
    const v = view();
    if (!geo || !v || v.flat) return { i: null, probe: false };
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
    if (v.method === 'pca') t += `\nPC1 ${signedValue(v.coords[2 * s])} · PC2 ${signedValue(v.coords[2 * s + 1])}`;
    return t;
  };

  const probeText = () => {
    const p = store.probe;
    if (!p || !geo?.probe) return '';
    return `Current input · ${p.caption}\nPC1 ${signedValue(geo.probe.a)} · PC2 ${signedValue(geo.probe.b)}${geo.probe.off ? ' (off the chart)' : ''}`;
  };

  const tipAt = (clientX: number, clientY: number) => {
    const v = view();
    if (hover !== null && v && hover < v.labels.length) showTip(pointText(hover), clientX, clientY);
    else if (hoverProbe && geo?.probe) showTip(probeText(), clientX, clientY);
    else return hideOwnTip();
    tipOwned = true;
  };
  /** Hides the shared tooltip only when it shows this section's text (another section may own it). */
  const hideOwnTip = () => {
    if (!tipOwned) return;
    tipOwned = false;
    hideTip();
  };

  /**
   * After the points moved (a t-SNE frame, a new result, a highlight), hover whatever is now under
   * the pointer, or keep a keyboard-chosen digit and move its tooltip with it.
   */
  function followPointer(): void {
    if (pointer) {
      const r = canvas.getBoundingClientRect();
      const x = pointer.x - r.left;
      const y = pointer.y - r.top;
      if (x >= 0 && y >= 0 && x <= r.width && y <= r.height) {
        const hit = nearest(x, y);
        if (hit.i !== hover || hit.probe !== hoverProbe) {
          setHover(hit.i, hit.probe);
          drawMarks();
        }
        tipAt(pointer.x, pointer.y);
        return;
      }
      pointer = null;
    }
    const v = view();
    const ok = hover !== null && keyed && !!v && !!geo && hover < v.labels.length && (focusDigit === null || v.labels[hover] === focusDigit);
    if (ok) {
      const r = canvas.getBoundingClientRect();
      tipAt(r.left + geo!.xy[2 * hover!], r.top + geo!.xy[2 * hover! + 1]);
      return;
    }
    if (hover !== null || hoverProbe) {
      setHover(null);
      drawMarks();
    }
    keyed = false;
    hideOwnTip();
  }

  canvas.addEventListener('pointermove', (e) => {
    pointer = { x: e.clientX, y: e.clientY };
    keyed = false;
    const r = canvas.getBoundingClientRect();
    const hit = nearest(e.clientX - r.left, e.clientY - r.top);
    if (hit.i !== hover || hit.probe !== hoverProbe) {
      setHover(hit.i, hit.probe);
      drawMarks();
    }
    tipAt(e.clientX, e.clientY);
  });
  canvas.addEventListener('pointerleave', () => {
    pointer = null;
    hideOwnTip();
    if (hover !== null || hoverProbe) {
      setHover(null);
      drawMarks();
    }
  });
  const choose = (s: number) => {
    const v = view();
    const d = store.data;
    if (!v || !d) return;
    const i = v.indices[s];
    setProbe(testProbe(d, i));
  };
  canvas.addEventListener('click', (e) => {
    const r = canvas.getBoundingClientRect();
    const hit = nearest(e.clientX - r.left, e.clientY - r.top);
    if (hit.i !== null) choose(hit.i);
  });

  // Keyboard: arrows move to the nearest point in that direction, Enter uses it as the input.
  canvas.addEventListener('keydown', (e) => {
    const v = view();
    if (!v || !geo || v.flat) return;
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
        pointer = null;
        keyed = true;
        setHover(next);
        drawMarks();
        const r = canvas.getBoundingClientRect();
        tipAt(r.left + xy[2 * next], r.top + xy[2 * next + 1]);
      }
    } else if ((e.key === 'Enter' || e.key === ' ') && hover !== null) {
      e.preventDefault();
      choose(hover);
    } else if (e.key === 'Escape') {
      clearHover();
      drawMarks();
    }
  });
  canvas.addEventListener('blur', () => {
    if (!keyed) return;
    clearHover();
    drawMarks();
  });

  const renderPreview = () => {
    const v = view();
    const d = store.data;
    const size = 104;
    if (hover !== null && v && d && hover < v.labels.length) {
      const i = v.indices[hover];
      previewHead.textContent = 'Hovered digit';
      paintThumb(previewCanvas, sampleInput(d, 'test', i), 28, 28, size);
      previewCanvas.hidden = false;
      previewTitle.textContent = `Test digit #${i}`;
      clear(previewMeta);
      const label = v.labels[hover];
      const pred = v.preds ? v.preds[hover] : null;
      append(previewMeta, [
        h('div', null, `Label ${label}`),
        pred !== null ? h('div', null, `Predicted ${pred} `, pred !== label ? h('span', { class: 'tag is-on' }, 'Mistake') : null) : null,
        v.method === 'pca' ? h('div', null, `PC1 ${signedValue(v.coords[2 * hover])} · PC2 ${signedValue(v.coords[2 * hover + 1])}`) : null,
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
    const pcaShown = !!v && v.method === 'pca' && !v.flat && !!probePt && probePt.key === p.key && probePt.coords === v.coords;
    if (pcaShown) previewMeta.append(h('div', null, `PC1 ${signedValue(probePt!.a)} · PC2 ${signedValue(probePt!.b)}`));
    const marked = pcaShown || (!!v && v.method === 'tsne' && !v.flat && probeIndex(v) !== null);
    const act = marked ? 'Marked on the map with a red cross' : v?.flat ? 'Every digit lands on the same point at this layer' : 'Hover a digit on the map to see it here';
    previewMeta.append(h('div', { class: 'embed-preview-act' }, act));
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
  const renderKeys = () => {
    const v = view();
    clear(keys);
    const n = v ? v.labels.length : DEFAULT_N;
    const wrong = v?.preds ? v.preds.reduce((acc, p, i) => acc + (p !== v.labels[i] ? 1 : 0), 0) : null;
    const ring = () => h('i', { class: 'embed-ring', 'aria-hidden': 'true' });
    if (mistakes && wrong !== null) keys.append(h('span', { class: 'embed-key' }, ring(), `Misclassified · ${int(wrong)} of ${int(n)}`));
    else if (mistakes && running) keys.append(h('span', { class: 'embed-key' }, ring(), 'Misclassified · shown when the run ends'));
    if (geo?.probe) keys.append(h('span', { class: 'embed-key' }, h('i', { class: 'embed-cross', 'aria-hidden': 'true' }), 'Current input'));
    keys.hidden = !keys.firstChild;
  };

  const renderSide = () => {
    renderKeys();
    const v = view();
    const s = current();
    clear(stats);
    clear(methodHint);
    const wrong = v?.preds ? v.preds.reduce((acc, p, i) => acc + (p !== v.labels[i] ? 1 : 0), 0) : null;
    const stat = (label: string, value: string) => h('div', null, `${label} `, h('b', null, value));
    if (s) {
      const r = s.res;
      stats.append(stat('Digits', `${int(r.indices.length)} · ${int(r.indices.length / 10)} of each`));
      stats.append(stat('Values per digit', `${int(r.dim)} at ${layerTitle(r.layer)}`));
      if (r.flat) {
        stats.append(stat('Variance', '0 · every digit gives the same values'));
      } else if (r.method === 'pca' && r.pca) {
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
          't-SNE moves the digits around until the ones that are neighbours at this layer sit next to each other. It preserves neighbours, not distances: the size of a cluster and the gaps between clusters mean little, so the axes carry no values. It starts from the PCA map, shrunk to a speck.',
        ),
      );
      const r = s?.res;
      if (r && !r.flat && r.inputDim !== undefined && r.inputDim < r.dim) {
        methodHint.append(
          h(
            'p',
            { class: 'hint' },
            r.reduced === 'projection'
              ? `This layer has ${int(r.dim)} values per digit. They are first mixed down to ${r.inputDim} random directions, which keeps the distances between digits roughly intact and makes t-SNE many times faster.`
              : `This layer has ${int(r.dim)} values per digit. t-SNE works on their ${r.inputDim} main directions (PCA), which keep the distances between neighbouring digits nearly intact and make it many times faster.`,
          ),
        );
      }
    }
    methodHint.append(h('p', { class: 'hint' }, 'Hover a numeral to see the digit; click it to make it the network’s input.'));
  };

  // ── Drawing ──
  const message = (): string => {
    if (!store.valid) return 'Fix the architecture above to see its embedding.';
    if (!store.data) return 'Waiting for MNIST to load…';
    if (error) return `The embedding failed: ${error}`;
    if (running && running.method === method && running.layer === layer) return phaseText(running);
    if (method === 'tsne') return 't-SNE runs only when you ask. Press Recompute.';
    return 'Not computed yet.';
  };

  /** Centred lines of text inside the plot frame. */
  const paintNote = (ctx: CanvasRenderingContext2D, text: string, L: number, T: number, pw: number, ph: number, colour: string) => {
    ctx.fillStyle = colour;
    ctx.font = `500 13px ${SANS}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const lines = wrapLines(ctx, text, Math.min(420, pw - 48));
    const lh = 19;
    lines.forEach((line, i) => ctx.fillText(line, L + pw / 2, T + ph / 2 + (i - (lines.length - 1) / 2) * lh));
  };

  /**
   * Paints everything: frame, axes, the numerals and mistake rings (cached as the scatter layer),
   * then the marks on top. Called when the points, the highlight, the theme or the size change.
   */
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
    runText.style.left = `${L + 8}px`;
    runText.style.top = `${T + 8}px`;
    runText.style.maxWidth = `${pw - 16}px`;
    ctx.fillStyle = p.surface;
    ctx.fillRect(L, T, pw, ph);
    ctx.strokeStyle = p.hair;
    ctx.lineWidth = 1;
    ctx.strokeRect(L + 0.5, T + 0.5, pw - 1, ph - 1);
    baseKey = null;

    if (!v || v.flat) {
      geo = null;
      const text = v ? FLAT_NOTE : message();
      paintNote(ctx, text, L, T, pw, ph, v ? p.ink2 : p.muted);
      canvas.setAttribute('aria-label', v ? `${v.method === 'pca' ? 'PCA' : 't-SNE'} of ${int(v.labels.length)} test digits at ${layerTitle(v.layer)}: ${text}` : text);
      updateRunText();
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
    // An axis without spread (e.g. only one unit varies) takes its scale from the other one.
    const unit = Math.max(x1 - x0, y1 - y0) || 1;
    const scale = Math.min((pw - 2 * pad) / Math.max(x1 - x0, unit * 1e-6), (ph - 2 * pad) / Math.max(y1 - y0, unit * 1e-6));
    const cx = (x0 + x1) / 2;
    const cy = (y0 + y1) / 2;
    const sx = (x: number) => L + pw / 2 + (x - cx) * scale;
    const sy = (y: number) => T + ph / 2 - (y - cy) * scale;

    if (isPca && v.pca) {
      // Ticks on the frame, axis names below and to the left.
      ctx.font = `400 10px ${MONO}`;
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
    ctx.restore();

    // Keep the scatter layer; hover and probe changes only repaint the marks over it.
    if (base.width !== canvas.width) base.width = canvas.width;
    if (base.height !== canvas.height) base.height = canvas.height;
    const bctx = base.getContext('2d')!;
    bctx.clearRect(0, 0, base.width, base.height);
    bctx.drawImage(canvas, 0, 0);
    baseKey = { coords: c, focus: focusDigit, mistakes, width: canvas.width, height: canvas.height, theme };

    geo = { L, T, pw, ph, cx, cy, scale, fs, xy, probe: null };
    paintMarks(ctx, v, geo);
    const what = `${v.method === 'pca' ? 'PCA' : 't-SNE'} map of ${int(n)} test digits at ${layerTitle(v.layer)}, each drawn as its numeral`;
    canvas.setAttribute(
      'aria-label',
      v.pca ? `${what}. PC1 explains ${pct(v.pca.explained[0])} and PC2 ${pct(v.pca.explained[1])} of the variance.` : `${what}. Use the arrow keys to move between digits and Enter to use one as the input.`,
    );
    updateRunText();
  }

  /** Repaints only the marks (input cross, hover ring) over the cached scatter layer. */
  function drawMarks(): void {
    const v = view();
    const k = baseKey;
    if (!v || v.flat || !geo || !k || k.coords !== v.coords || k.focus !== focusDigit || k.mistakes !== mistakes || k.width !== canvas.width || k.height !== canvas.height || k.theme !== theme) {
      draw();
      return;
    }
    const ctx = canvas.getContext('2d')!;
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(base, 0, 0);
    ctx.restore();
    paintMarks(ctx, v, geo);
  }

  /** The input cross and the hovered digit's ring, on top of the scatter. Updates g.probe. */
  function paintMarks(ctx: CanvasRenderingContext2D, v: View, g: Geo): void {
    const p = palette();
    const { L, T, pw, ph, xy, fs } = g;
    const n = v.labels.length;
    const sx = (x: number) => L + pw / 2 + (x - g.cx) * g.scale;
    const sy = (y: number) => T + ph / 2 - (y - g.cy) * g.scale;
    ctx.save();
    ctx.beginPath();
    ctx.rect(L, T, pw, ph);
    ctx.clip();

    // The current input: projected for PCA; for t-SNE only when it is one of the plotted digits.
    let probe: Geo['probe'] = null;
    if (v.method === 'pca' && probePt && store.probe && probePt.key === store.probe.key && probePt.coords === v.coords) {
      const px = sx(probePt.a);
      const py = sy(probePt.b);
      const off = px < L || px > L + pw || py < T || py > T + ph;
      probe = { x: Math.max(L + 10, Math.min(L + pw - 10, px)), y: Math.max(T + 10, Math.min(T + ph - 10, py)), off, a: probePt.a, b: probePt.b };
    } else if (v.method === 'tsne') {
      const s = probeIndex(v);
      if (s !== null) probe = { x: xy[2 * s], y: xy[2 * s + 1], off: false, a: v.coords[2 * s], b: v.coords[2 * s + 1] };
    }
    g.probe = probe;
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
      ctx.fillStyle = catColours()[v.labels[hover]];
      ctx.fillText(String(v.labels[hover]), x, y + 0.5);
    }
    ctx.restore();
  }

  function render(): void {
    draw();
    renderSide();
    renderPreview();
    for (const [i, b] of segButtons.entries()) b.setAttribute('aria-pressed', String((i === 0 ? 'pca' : 'tsne') === method));
    followPointer();
  }

  // ── Events ──
  store.on('model', () => {
    if (store.version !== layerVersion) {
      layerVersion = store.version;
      const spec = structuredClone(store.spec);
      // Reset keeps the reader's layer; a new architecture keeps it only if that layer (and
      // everything before it) is unchanged, otherwise it goes back to the last hidden layer.
      const keep = layerPicked && layer < store.net.blocks.length && layerKey(spec, layer) === layerKey(layerSpec, layer);
      if (!keep) {
        layer = defaultLayer();
        layerPicked = false;
      }
      if (JSON.stringify(spec) !== JSON.stringify(layerSpec)) orientation.clear();
      layerSpec = spec;
      cancelRun();
      probePt = null;
      error = null;
      clearHover();
    }
    buildLayerSelect();
    render();
  });
  // Drawing on the pad changes the probe on every pointer move: coalesce to one update per frame,
  // which repaints only the marks over the cached scatter. Off screen, catch up later instead.
  store.on('probe', () => {
    if (!onScreen) {
      probeDirty = true;
      dirty = true;
      return;
    }
    if (probeQueued) return;
    probeQueued = true;
    requestAnimationFrame(() => {
      probeQueued = false;
      projectProbe();
      drawMarks();
      renderKeys();
      renderPreview();
    });
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
    theme++;
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
