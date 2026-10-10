import './attributionView.css';
import './attribution.css';
import {
  completenessText,
  DEFAULT_IG_STEPS,
  DEFAULT_OCCLUSION,
  featureSentence,
  kinkText,
  sig,
  type AttributionParams,
  type AttributionResult,
} from '../analysis/attribution';
import { analysis, isSuperseded } from '../analysis/client';
import { fixed } from '../analysis/stats';
import { noun } from '../data/datasets';
import { featureDefs } from '../data/features';
import { pointDomain, PointEvaluator } from '../data/grid';
import { argmax, Network } from '../nn/network';
import type { Shape } from '../nn/types';
import { store, type Probe } from '../store';
import { axisName } from './boundaryMath';
import { dotTest, drawAxes, drawField, evaluatePlane, netFits, pickAt, regionCanvas, type PlaneField, type Rect } from './boundary2d';
import { $, h } from './dom';
import { fitCanvas, maxAbs } from './draw';
import { drawInputCross } from './embeddingView';
import { stampNow, syncedSection, type Stamp } from './snapshot';
import { classColor, css, diverging, onThemeChange, palette, sequential, type RGB } from './theme';
import { hideTip, showTip } from './tip';

/**
 * Section 10, "why this prediction": for the current input and one target class, which parts of
 * the input drive that class's score. Images get four maps beside the input itself (grey images
 * as they are; colour photos with the three channels added up per pixel). Point data gets one bar
 * per input feature, plus a small map of the plane with the direction that raises the score.
 */

type Mode = 'grey' | 'colour' | 'points';
type PanelKey = 'input' | 'saliency' | 'gradInput' | 'integrated' | 'occlusion';
type Ramp = 'unit' | 'sequential' | 'diverging';
type Child = Node | string;

interface PanelDef {
  key: PanelKey;
  title: string;
  ramp: Ramp;
  /** Grey images: draw the digit faintly under the map. Colour photos get a grey underlay on every map. */
  underlay: boolean;
}

const DEBOUNCE_MS = 150;
/** Live drawing still refreshes this often, even while the pen keeps moving. */
const MAX_WAIT_MS = 600;
/** Grey images: how much ink the faint digit under a signed map adds. */
const UNDERLAY = 0.16;
/** Colour photos: how far the grey version of the photo shows through under a map. */
const PHOTO_UNDERLAY = 0.42;
const MINUS = '−';
const SUB = '₀₁₂₃₄₅₆₇₈₉';
const ARCH_NOTICE = 'Fix the architecture in 01 to see attributions.';
/** The mini-map of the plane: its grid resolution and CSS size. */
const PLANE_RES = 64;
const PLANE_PAD = { l: 34, r: 8, t: 20, b: 34 };

const PANELS: PanelDef[] = [
  { key: 'input', title: 'Input', ramp: 'unit', underlay: false },
  { key: 'saliency', title: 'Saliency', ramp: 'sequential', underlay: false },
  { key: 'gradInput', title: 'Gradient × input', ramp: 'diverging', underlay: true },
  { key: 'integrated', title: 'Integrated gradients', ramp: 'diverging', underlay: true },
  { key: 'occlusion', title: 'Occlusion', ramp: 'diverging', underlay: true },
];

interface Panel {
  def: PanelDef;
  fig: HTMLElement;
  symbol: HTMLElement;
  canvas: HTMLCanvasElement;
  hint: HTMLElement;
  lo: HTMLElement;
  hi: HTMLElement;
  ramp: HTMLCanvasElement;
  text: HTMLElement;
}

/** What the section is asked to explain: an input, with the prediction for it, and a target. */
interface Request {
  probe: Probe;
  x: Float32Array;
  probs: Float32Array;
  pred: number;
  target: number;
  /** Point data: a copy of the network the run uses, for the map of the plane. */
  net: Network | null;
}

/** A finished result and what it was computed from. */
interface Shown {
  r: AttributionResult;
  x: Float32Array;
  probe: Probe;
  stamp: Stamp;
  net: Network | null;
  /** Point data: the class regions on the plane through the point (computed when first drawn). */
  field?: PlaneField | null;
}

const rgb: RGB = [0, 0, 0];
let scratch: HTMLCanvasElement | null = null;

/** "+0.82", "−0.31", "0": three significant digits, a true minus sign. */
const signed = (v: number) => sig(v, true);

export function mountAttribution(): void {
  const root = $('attr-root');
  root.classList.add('attr');
  root.dataset.state = 'idle';

  // ── State ──
  /** The class the learner chose; null follows the prediction. Cleared when the input changes. */
  let pinned: number | null = null;
  /** The input last announced by the store, to tell a real change from a re-selection. */
  let lastInput: { key: string; x: Float32Array } | null = null;
  let current: Request | null = null;
  let shown: Shown | null = null;
  /** The request and weights of the run in progress. */
  let running: { req: Request; stamp: Stamp } | null = null;
  let error: string | null = null;
  let hover: number | null = null;
  let runId = 0;
  /** A probe change arrived while a run was going; start again when it ends. */
  let queued = false;
  /** The input changed while the section was off screen. */
  let inputDirty = false;
  /** On screen or close to it; tracked here, not read from the status line's own observer, whose
   *  callback may arrive after ours. */
  let onScreen = false;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let waitingSince = 0;
  let px = 140;
  let mode: Mode = modeNow();
  /** Class names the chips were built for. */
  let chipKey = '';

  // ── Words that depend on the dataset ──
  function modeNow(): Mode {
    if (store.info.kind === 'points') return 'points';
    return store.input.c === 3 ? 'colour' : 'grey';
  }
  const isMnist = () => store.info.id === 'mnist';
  const className = (k: number) => store.info.classes[k] ?? `Class ${k}`;
  /** In running text: "digit 7", "cat", "Class 1". */
  const targetWords = (k: number) => (isMnist() ? `digit ${k}` : className(k));
  /** After "the score for": "7", "cat", "Class 1". */
  const scoreName = (k: number) => (isMnist() ? String(k) : className(k));
  /** In "z₇", "z(cat)". */
  const zOf = (k: number) => (isMnist() ? `z${SUB[k]}` : `z(${className(k)})`);
  const pOf = (k: number) => `p(${scoreName(k)})`;
  const one = () => noun(store.info);
  /** The all-zero input, as the completeness line names it. */
  const baseName = () => (mode === 'grey' ? 'blank' : mode === 'colour' ? 'black' : 'origin');
  /** Section 03's heading for this dataset (main.ts sets the same). */
  const drawSection = () => (mode === 'points' ? '03 Decision boundary' : mode === 'colour' ? '03 Try a photo' : '03 Draw');
  const shapeNow = (): Shape => store.net.arch.input;

  // ── Elements: target picker and prediction ──
  let chipBtns: HTMLButtonElement[] = [];
  let marks: HTMLElement[] = [];
  const chipRow = h('div', { class: 'attr-chips', role: 'group', 'aria-labelledby': 'attr-target-label' });
  const targetLabel = h('span', { class: 'label', id: 'attr-target-label' }, 'Explain digit');
  const predLine = h('p', { class: 'attr-pred', id: 'attr-pred', 'aria-live': 'polite' }, 'Waiting for an input…');
  const inputCaption = h('p', { class: 'attr-input-caption' });
  const notice = h('p', { class: 'notice', id: 'attr-notice', hidden: true });

  function buildChips(): void {
    const names = store.info.classes;
    const key = `${store.info.id}|${names.join('|')}`;
    if (key === chipKey) return;
    chipKey = key;
    chipRow.replaceChildren();
    chipBtns = [];
    marks = [];
    const named = !isMnist();
    chipRow.classList.toggle('is-named', named);
    names.forEach((name, d) => {
      const swatch = mode === 'points' ? h('span', { class: 'attr-swatch', 'aria-hidden': 'true', style: { background: `var(--cat-${d % 10})` } }) : null;
      const b = h('button', { type: 'button', class: `chip${named ? ' attr-chip-name' : ''}`, id: `attr-target-${d}`, 'aria-pressed': 'false' }, swatch, named ? name : String(d)) as HTMLButtonElement;
      b.addEventListener('click', () => pick(d));
      const m = h('span', { class: 'attr-mark', 'aria-hidden': 'true' });
      chipBtns.push(b);
      marks.push(m);
      chipRow.append(h('div', { class: 'attr-chip-col' }, b, m));
    });
  }

  const side = h('div', { class: 'attr-side' }, h('div', { class: 'attr-target' }, targetLabel, chipRow), predLine, inputCaption, notice);
  const notes = h('div', { class: 'attr-notes' });

  // ── Elements: image panels ──
  const panels: Panel[] = PANELS.map((def) => {
    const canvas = h('canvas', { role: 'img', id: `attr-map-${def.key}`, 'aria-label': `${def.title}: not computed yet` }) as HTMLCanvasElement;
    const lo = h('span');
    const hi = h('span');
    const ramp = h('canvas', { 'aria-hidden': 'true' }) as HTMLCanvasElement;
    const hint = h('p', { class: 'hint' });
    const symbol = h('span', { class: 'mono' });
    const text = h('div', { class: 'attr-text' }, hint);
    const fig = h(
      'figure',
      { class: 'attr-panel', 'data-panel': def.key },
      h('h3', { class: 'panel-title attr-title' }, def.title, ' ', symbol),
      h('div', { class: 'attr-map' }, canvas),
      h('div', { class: 'attr-scale' }, lo, ramp, hi),
      text,
    );
    const panel: Panel = { def, fig, symbol, canvas, hint, lo, hi, ramp, text };
    canvas.addEventListener('pointermove', (e) => onHover(panel, e));
    canvas.addEventListener('pointerdown', (e) => onHover(panel, e));
    canvas.addEventListener('pointerleave', () => {
      hideTip();
      setHover(null);
    });
    return panel;
  });
  /** The completeness line: under integrated gradients for images, under the bars for points. */
  const check = h('p', { class: 'attr-check', id: 'attr-check' });
  const integratedPanel = panels.find((p) => p.def.key === 'integrated')!;
  integratedPanel.text.append(check);

  // ── Elements: point data ──
  const planeCanvas = h('canvas', { id: 'attr-plane', class: 'attr-plane', role: 'img', tabindex: '0', 'aria-label': 'Map of the input plane: not computed yet' }) as HTMLCanvasElement;
  const planeTitle = h('h3', { class: 'panel-title attr-title' }, 'The point');
  const planeCaption = h('p', { class: 'attr-plane-caption', id: 'attr-plane-caption' });
  const planeHint = h('p', { class: 'hint' });
  const planeFig = h('figure', { class: 'attr-pts-fig attr-pts-plane' }, planeTitle, h('div', { class: 'attr-plane-box' }, planeCanvas), planeCaption, planeHint);
  const sentence = h('p', { class: 'attr-sentence', id: 'attr-sentence' });
  const featHead = h(
    'thead',
    null,
    h(
      'tr',
      null,
      h('th', { scope: 'col' }, 'Feature'),
      h('th', { scope: 'col', class: 'attr-col-slope', title: 'The gradient ∂z/∂f: how fast the score changes as this feature grows' }, 'Slope'),
      h('th', { scope: 'col' }, 'Gradient × input'),
      h('th', { scope: 'col' }, 'Integrated gradients'),
    ),
  );
  const featBody = h('tbody');
  const featTable = h('table', { class: 'attr-feat', id: 'attr-features' }, featHead, featBody);
  const featScale = h('p', { class: 'attr-feat-scale', id: 'attr-feat-scale' });
  const featHint = h('div', { class: 'attr-feat-hints' });
  const barsFig = h(
    'figure',
    { class: 'attr-pts-fig attr-pts-bars' },
    h('h3', { class: 'panel-title attr-title' }, 'Per input feature'),
    sentence,
    h('div', { class: 'attr-feat-wrap' }, featTable),
    featScale,
    featHint,
  );
  const pointsBox = h('div', { class: 'attr-pts', id: 'attr-points', hidden: true }, planeFig, barsFig);

  /** Says what the maps are waiting for while they are out of date (faded) or still blank. */
  const waitLabel = h('p', { class: 'attr-wait', id: 'attr-wait', 'aria-hidden': 'true' });
  const grid = h('div', { class: 'attr-panels' }, ...panels.map((p) => p.fig), pointsBox, waitLabel);

  // The shared policy refreshes when the weights moved on, or on Recompute: both should run.
  const sync = syncedSection(root, () => start(true));
  root.append(sync.status, h('div', { class: 'attr-layout' }, side, grid, notes));

  // ── Computation ──

  const sameX = (a: ArrayLike<number>, b: ArrayLike<number>) => {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
    return true;
  };
  const sameStamp = (a: Stamp, b: Stamp) => a.version === b.version && a.rev === b.rev;
  /** True when `a` and `b` ask for the same maps (same input, target and weights). */
  const sameJob = (a: Request, sa: Stamp, b: Request, sb: Stamp) =>
    a.target === b.target && a.probe.key === b.probe.key && sameStamp(sa, sb) && sameX(a.x, b.x);
  /** True when the result on screen already answers `req` for the current weights. */
  const answered = (req: Request) =>
    !!shown && shown.r.target === req.target && shown.probe.key === req.probe.key && sameStamp(shown.stamp, stampNow()) && sameX(shown.x, req.x);

  /**
   * The prediction for `probe` with the page's current network, and the target to explain. Null
   * while the input does not fit the network (a dataset switch updates the network first).
   */
  function requestFor(probe: Probe | null): Request | null {
    if (!probe || probe.x.length !== store.net.inputSize) return null;
    const x = probe.x.slice();
    const probs = store.net.forward(x).slice();
    const pred = argmax(probs);
    const target = pinned !== null && pinned < probs.length ? pinned : pred;
    let net: Network | null = null;
    if (mode === 'points') {
      net = new Network(store.net.arch, 0);
      net.setWeights(store.net.getWeights());
    }
    return { probe, x, probs, pred, target, net };
  }

  /**
   * Input changes (live drawing included) wait a moment, and never cut off a run in progress.
   * They are the learner's own requests, so unlike weight updates they also run during training.
   */
  function soon(): void {
    if (!onScreen) {
      inputDirty = true;
      return;
    }
    const now = performance.now();
    if (timer) clearTimeout(timer);
    else waitingSince = now;
    timer = setTimeout(
      () => {
        timer = null;
        if (running) queued = true;
        else start();
      },
      Math.max(0, Math.min(DEBOUNCE_MS, waitingSince + MAX_WAIT_MS - now)),
    );
  }

  /** Drops the run in progress, if any, so its result is never shown. */
  function stopRun(): void {
    runId++;
    queued = false;
    if (!running) return;
    running = null;
    analysis.cancel('attribution');
    sync.fail();
    setBusy();
  }

  /** Computes the maps for the current input and target; `force` (Recompute) runs even when they are shown already. */
  function start(force = false): void {
    if (timer) clearTimeout(timer);
    timer = null;
    inputDirty = false;
    queued = false;
    if (!store.data) return;
    if (!store.valid) {
      stopRun();
      current = null;
      error = null;
      renderControls();
      renderNotice(ARCH_NOTICE);
      return;
    }
    const req = requestFor(store.probe);
    if (!req) return;
    // Already computing exactly this (e.g. a probe event and the first refresh together): let it finish.
    if (running && sameJob(running.req, running.stamp, req, stampNow())) {
      current = req;
      renderControls();
      return;
    }
    // Already showing exactly this (the current input picked again, same target and weights): keep it.
    if (!force && !running && answered(req)) {
      current = req;
      renderControls();
      renderNotice();
      return;
    }
    current = req;
    error = null;
    renderControls();
    renderNotice(); // clears "Fix the architecture" as soon as there is something to compute
    const id = ++runId;
    const begun = sync.begin();
    running = { req, stamp: begun };
    setBusy();
    sync.setProgress(0);
    const params: AttributionParams = { x: req.x, target: req.target, igSteps: DEFAULT_IG_STEPS };
    if (mode !== 'points') params.occlusion = DEFAULT_OCCLUSION;
    const coords = req.probe.coords;
    const dims = store.data.points?.dims;
    if (mode === 'points' && coords && dims && coords.length === dims) params.point = { coords: Array.from(coords), dims, features: store.features.slice() };
    analysis
      .run<AttributionResult>('attribution', 'attribution', params, (p) => {
        if (id === runId) sync.setProgress(p.done / p.total);
      })
      .then(
        (r) => {
          if (id !== runId) return;
          running = null;
          shown = { r, x: req.x, probe: req.probe, stamp: begun, net: req.net };
          setBusy();
          renderPanels();
          renderNotice();
          sync.done(begun);
        },
        (e: unknown) => {
          if (id !== runId) return; // replaced by a newer run, which owns the status now
          running = null;
          if (!isSuperseded(e)) error = e instanceof Error ? e.message : String(e);
          setBusy();
          renderNotice();
          sync.fail();
        },
      )
      .then(() => {
        if (id === runId && queued) start();
      });
  }

  function pick(d: number): void {
    pinned = d;
    if (!store.valid) {
      renderControls(); // "Will explain digit d once the architecture is fixed."
      return;
    }
    if (current) current.target = d;
    renderControls();
    start();
  }

  /** Reflects the run state: data-state, aria-busy, and the faded maps with what they wait for. */
  function setBusy(): void {
    const busy = running !== null;
    root.dataset.state = busy ? 'busy' : error ? 'error' : shown ? 'done' : 'idle';
    grid.setAttribute('aria-busy', String(busy));
    const req = running?.req;
    const stale = !!req && !!shown && !(shown.r.target === req.target && shown.probe.key === req.probe.key && sameStamp(shown.stamp, running!.stamp) && sameX(shown.x, req.x));
    grid.classList.toggle('is-stale', stale);
    grid.classList.toggle('is-waiting', !!req && (stale || !shown));
    if (req) waitLabel.textContent = `${shown ? 'Updating' : 'Computing'} for ${targetWords(req.target)}…`;
  }

  // ── Mode: which panels show, and the words that go with the dataset ──

  function applyMode(): void {
    mode = modeNow();
    buildChips();
    const pts = mode === 'points';
    for (const p of panels) p.fig.hidden = pts;
    pointsBox.hidden = !pts;
    grid.classList.toggle('is-points', pts);
    root.dataset.mode = mode;
    (pts ? barsFig : integratedPanel.text).append(check);
    targetLabel.textContent = isMnist() ? 'Explain digit' : 'Explain class';
    for (const p of panels) {
      const sym = p.def.key === 'saliency' ? (mode === 'colour' ? 'Σ|∂z/∂x|' : '|∂z/∂x|') : '';
      p.symbol.textContent = sym;
      p.symbol.hidden = !sym;
    }
    renderNotes();
  }

  function renderNotes(): void {
    const n = one();
    const what = isMnist() ? 'digit' : 'class';
    const paras: string[] = [];
    if (mode === 'points') {
      paras.push(`Pick a class to see which input features argue for it. Your pick holds until the input changes: click the map of the plane, or pick a point in ${drawSection()} or 07 Data.`);
      paras.push(
        'The network sees each point as its input features. Gradient × input and integrated gradients split the class’s logit z, its score before softmax turns the logits into probabilities, among those features: a positive value pushes the point toward the class, a negative one away from it.',
      );
    } else {
      paras.push(`Pick a ${what} to see which pixels argue for it. Your pick holds until the input changes; change the input in 02 Network, ${drawSection()} or 07 Data.`);
      paras.push(
        `All four maps explain the ${what}’s logit z, its score before softmax turns the logits into probabilities. Unlike the probability p, it does not flatten out near 0 or 1, so a confident or rejected ${n} still shows detail.` +
          (mode === 'grey' ? ` Signed maps show the ${n} faintly underneath.` : ''),
      );
      if (mode === 'colour') {
        paras.push(
          'A photo has three values per pixel: red, green and blue. Each map adds them up into one value per pixel. Gradient × input and integrated gradients keep their signs, so the map still adds up to the change in the score; saliency adds sizes, so channels pulling opposite ways do not cancel. The photo shows through in grey, so red and blue only ever mean raise and lower.',
        );
        paras.push(
          'Occlusion paints each patch with the average colour of the test images, not black: a black square is a strong feature in a photo in its own right, so it would measure the reaction to the square as much as what the square hides.',
        );
      }
    }
    notes.replaceChildren(...paras.map((t) => h('p', { class: 'hint' }, t)));
  }

  // ── Rendering ──

  function renderControls(): void {
    const c = current;
    const label = c?.probe.label ?? null;
    const target = c ? c.target : pinned;
    for (let d = 0; d < chipBtns.length; d++) {
      const b = chipBtns[d];
      const isPred = !!c && c.pred === d;
      const isTrue = label === d;
      b.setAttribute('aria-pressed', String(target === d));
      const tags = [isPred ? 'predicted' : '', isTrue ? 'true label' : ''].filter(Boolean).join(', ');
      b.setAttribute('aria-label', `Explain ${targetWords(d)}${tags ? ` (${tags})` : ''}`);
      const m = marks[d];
      m.replaceChildren();
      if (isPred) m.append(h('span', { class: 'attr-tag is-pred' }, 'pred'));
      if (isTrue) m.append(h('span', { class: 'attr-tag is-true' }, 'true'));
    }
    if (!c) {
      if (!store.valid)
        predLine.textContent = pinned !== null ? `Will explain ${targetWords(pinned)} once the architecture is fixed.` : 'No prediction until the architecture is fixed.';
      else predLine.textContent = store.data ? 'Waiting for an input…' : `Waiting for ${store.info.name} to load…`;
      inputCaption.textContent = store.probe && store.data ? `Input: ${store.probe.caption}` : '';
      delete root.dataset.target;
      delete root.dataset.pred;
      return;
    }
    const pTarget = c.probs[c.target];
    predLine.replaceChildren(
      'Predicted ',
      h('b', null, scoreName(c.pred)),
      ' at ',
      h('b', null, pctText(c.probs[c.pred])),
      '; explaining ',
      isMnist() ? 'digit ' : '',
      h('b', null, scoreName(c.target)),
      c.target !== c.pred ? ` (${pctText(pTarget)})` : '',
    );
    inputCaption.textContent = `Input: ${c.probe.caption}`;
    root.dataset.target = String(c.target);
    root.dataset.pred = String(c.pred);
  }

  function renderNotice(text?: string): void {
    let msg = text ?? '';
    if (!msg && !store.valid) msg = ARCH_NOTICE;
    if (!msg && error) msg = `Attribution failed: ${error}`;
    if (!msg && shown && !hasInk(shown.x)) {
      msg =
        mode === 'points'
          ? 'The point is at the origin, where every feature is 0, so gradient × input and integrated gradients are all zero. Click the map to move it.'
          : mode === 'colour'
            ? 'The input is black, so gradient × input and integrated gradients are all zero. Pick an image in 02 Network or 07 Data.'
            : `The input is blank, so gradient × input, integrated gradients and occlusion are all zero. ${isMnist() ? 'Draw a digit' : 'Draw something'} or pick one in 07 Data.`;
    }
    notice.textContent = msg;
    notice.hidden = !msg;
    if (!running) root.dataset.state = error ? 'error' : shown ? 'done' : 'idle';
  }

  /** The result on screen, when it still fits the network (a dataset switch clears it anyway). */
  const result = (): Shown | null => (shown && shown.r.kind === (mode === 'points' ? 'features' : 'image') ? shown : null);

  function renderPanels(): void {
    const s = result();
    root.dataset.resultKey = s ? `${s.probe.key}|${s.r.target}` : '';
    if (mode === 'points') renderPoints(s);
    else renderImages(s);
    renderCheck(s);
  }

  function renderCheck(s: Shown | null): void {
    const base = baseName();
    if (s) {
      const c = completenessText(s.r, base);
      // Unbreakable pieces, so a narrow column wraps after "=" or ",", never inside a term.
      const pieces = (text: string, cls?: string) =>
        text.split(/(?<= =|,) (?=\S)|(?<= of) (?=\|)/).flatMap((t, k) => [k ? ' ' : '', h('span', cls ? { class: cls } : null, t)]);
      check.replaceChildren(...pieces(c.sum), ' ', ...pieces(c.expected), ' ', ...pieces(c.gap, 'attr-gap'));
      check.title = c.title;
    } else {
      check.textContent = `Σ IG = –; z(x) ${MINUS} z(${base}) = –`;
      check.removeAttribute('title');
    }
  }

  // ── Images ──

  const values = (key: PanelKey, s: Shown): Float32Array => (key === 'input' ? s.x : s.r[key]);
  const scaleMax = (def: PanelDef, data: Float32Array) => (def.ramp === 'unit' ? 1 : maxAbs(data));

  function hintFor(key: PanelKey, t: number, r: AttributionResult | null): Child[] {
    const colour = mode === 'colour';
    const shape = shapeNow();
    const sc = scoreName(t);
    switch (key) {
      case 'input':
        return colour
          ? [`The ${shape.h}×${shape.w} photo the network sees: red, green and blue values from 0 to 1 for every pixel.`]
          : [`The ${shape.h}×${shape.w} image the network sees, from 0 (blank) to 1 (full ink).`];
      case 'saliency': {
        const kinks = r ? kinkText(r.kinks, colour ? 'image' : 'blank') : '';
        const base = colour ? `How strongly the score for ${sc} reacts to a small change in each pixel, its three channels added up` : `How strongly the score for ${sc} reacts to a small change in each pixel`;
        return [`${base}${kinks ? `. ${kinks}` : ', in either direction.'}`];
      }
      case 'gradInput':
        return colour
          ? [`The gradient times each pixel’s values, added over its channels: red pixels raise the score for ${sc}, blue ones lower it.`]
          : [`The gradient times each pixel’s ink: red ink raises the score for ${sc}, blue ink lowers it.`];
      case 'integrated':
        return [
          `Gradient × input averaged while the image fades in from ${colour ? 'black' : 'blank'} (${r?.igSteps ?? DEFAULT_IG_STEPS} steps), so the pixels share out the whole change in the score for ${sc}.`,
        ];
      case 'occlusion': {
        const size = r?.occlusionSize ?? DEFAULT_OCCLUSION.size;
        if (!colour) return [`Erases a ${size}×${size} patch at a time: red where that lowers the score for ${sc}, blue where it raises it.`];
        const fill = r?.fill;
        const sw = fill ? h('i', { class: 'attr-fill', title: `Fill: R ${fixed(fill[0], 2)}, G ${fixed(fill[1], 2)}, B ${fixed(fill[2], 2)}`, style: { background: `rgb(${fill.map((v) => Math.round(v * 255)).join(',')})` } }) : null;
        return [
          `Paints a ${size}×${size} patch at a time with the average colour`,
          ...(sw ? [' ', sw] : []),
          ` of the test images: red where that lowers the score for ${sc}, blue where it raises it.`,
        ];
      }
    }
  }

  function ariaFor(key: PanelKey, t: number, max: number): string {
    const tw = targetWords(t);
    const colour = mode === 'colour';
    const shape = shapeNow();
    switch (key) {
      case 'input':
        return colour ? `The input photo, ${shape.h} by ${shape.w} pixels, in colour` : `The input image, ${shape.h} by ${shape.w} pixels`;
      case 'saliency':
        return `Saliency for ${tw}: darker pixels change its score most${colour ? ', the three channels added up' : ''}. Largest value ${sig(max)}.`;
      case 'gradInput':
        return `Gradient times input for ${tw}: red pixels raise its score, blue lower it. Values up to ±${sig(max)}.`;
      case 'integrated':
        return `Integrated gradients for ${tw}: red pixels raise its score, blue lower it. Values up to ±${sig(max)}.`;
      case 'occlusion':
        return `Occlusion for ${tw}: red areas lower its score when ${colour ? 'painted over' : 'erased'}, blue raise it. Values up to ±${sig(max)}.`;
    }
  }

  function renderImages(s: Shown | null): void {
    const t = s?.r.target ?? current?.target ?? 0;
    for (const p of panels) {
      const def = p.def;
      p.hint.replaceChildren(...hintFor(def.key, t, s?.r ?? null));
      const data = s ? values(def.key, s) : null;
      const max = data ? scaleMax(def, data) : 0;
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
      p.canvas.setAttribute('aria-label', s ? ariaFor(def.key, t, max) : `${def.title}: not computed yet`);
    }
  }

  function paintMap(p: Panel, data: Float32Array | null, max: number, x: Float32Array | null): void {
    const pal = palette();
    const shape = shapeNow();
    const W = shape.w;
    const H = shape.h;
    const P = W * H;
    const colour = shape.c === 3 && (!x || x.length === 3 * P);
    const ctx = fitCanvas(p.canvas, px, (px * H) / W);
    scratch ??= document.createElement('canvas');
    scratch.width = W;
    scratch.height = H;
    const sctx = scratch.getContext('2d');
    if (!sctx) return;
    const img = sctx.createImageData(W, H);
    const d = img.data;
    const { ink, surface, accent, neg } = pal.rgb;
    const inv = max > 0 ? 1 / max : 0;
    const usable = !!data && data.length >= (p.def.key === 'input' ? shape.c * P : P);
    for (let i = 0; i < P; i++) {
      if (!usable) {
        rgb[0] = surface[0];
        rgb[1] = surface[1];
        rgb[2] = surface[2];
      } else if (p.def.key === 'input' && colour) {
        rgb[0] = 255 * clamp01(data![i]);
        rgb[1] = 255 * clamp01(data![P + i]);
        rgb[2] = 255 * clamp01(data![2 * P + i]);
      } else if (colour && x) {
        // A faint grey photo underneath, then the attribution laid over it.
        // Saliency runs from the surface to ink, as grey as the photo, so the photo stays fainter there.
        const k = p.def.ramp === 'sequential' ? PHOTO_UNDERLAY / 2 : PHOTO_UNDERLAY;
        const lum = 255 * clamp01(0.299 * x[i] + 0.587 * x[P + i] + 0.114 * x[2 * P + i]);
        const b0 = surface[0] + (lum - surface[0]) * k;
        const b1 = surface[1] + (lum - surface[1]) * k;
        const b2 = surface[2] + (lum - surface[2]) * k;
        const v = data![i] * inv;
        const to = p.def.ramp === 'sequential' ? ink : v >= 0 ? accent : neg;
        const a = p.def.ramp === 'sequential' ? clamp01(v) : Math.pow(Math.min(1, Math.abs(v)), 0.8);
        rgb[0] = b0 + (to[0] - b0) * a;
        rgb[1] = b1 + (to[1] - b1) * a;
        rgb[2] = b2 + (to[2] - b2) * a;
      } else {
        if (p.def.ramp === 'diverging') diverging(data![i] * inv, rgb);
        else sequential(data![i] * inv, rgb);
        if (p.def.underlay && x) {
          const a = UNDERLAY * clamp01(x[i]);
          rgb[0] += (ink[0] - rgb[0]) * a;
          rgb[1] += (ink[1] - rgb[1]) * a;
          rgb[2] += (ink[2] - rgb[2]) * a;
        }
      }
      d[4 * i] = rgb[0];
      d[4 * i + 1] = rgb[1];
      d[4 * i + 2] = rgb[2];
      d[4 * i + 3] = 255;
    }
    sctx.putImageData(img, 0, 0);
    const ph = (px * H) / W;
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(scratch, 0, 0, W, H, 0, 0, px, ph);
    ctx.strokeStyle = pal.hair;
    ctx.lineWidth = 1;
    ctx.strokeRect(0.5, 0.5, px - 1, ph - 1);
    if (hover !== null && usable) {
      const cell = px / W;
      const cx = (hover % W) * cell;
      const cy = Math.floor(hover / W) * cell;
      ctx.strokeStyle = pal.ink;
      ctx.lineWidth = 2;
      ctx.strokeRect(cx - 1, cy - 1, cell + 2, cell + 2);
      ctx.strokeStyle = pal.surface;
      ctx.lineWidth = 1;
      ctx.strokeRect(cx + 0.5, cy + 0.5, cell - 1, cell - 1);
    }
  }

  function paintRamp(p: Panel): void {
    const w = 56;
    const hgt = 8;
    const ctx = fitCanvas(p.ramp, w, hgt);
    const photoInput = mode === 'colour' && p.def.ramp === 'unit';
    for (let i = 0; i < w; i++) {
      const t = i / (w - 1);
      const g = Math.round(255 * t);
      ctx.fillStyle = photoInput ? `rgb(${g},${g},${g})` : css(p.def.ramp === 'diverging' ? diverging(t * 2 - 1) : sequential(t));
      ctx.fillRect(i, 0, 1, hgt);
    }
  }

  // ── Hover (images) ──

  /** Tooltip lines for pixel i of panel `key`. */
  function pixelLines(key: PanelKey, s: Shown, i: number): string[] {
    const t = s.r.target;
    const ch = s.r.channels;
    const P = s.r.saliency.length;
    const split = (v: Float32Array | undefined, plus: boolean) => (v ? ` (R ${sig(v[i], plus)} · G ${sig(v[P + i], plus)} · B ${sig(v[2 * P + i], plus)})` : '');
    switch (key) {
      case 'input':
        return [];
      case 'saliency':
        return [`Saliency ${mode === 'colour' ? 'Σ' : ''}|∂${zOf(t)}/∂x| ${sig(s.r.saliency[i])}${split(ch?.saliency, false)}`];
      case 'gradInput':
        return [`Gradient × input ${signed(s.r.gradInput[i])}${split(ch?.gradInput, true)}`];
      case 'integrated':
        return [`Integrated gradient ${signed(s.r.integrated[i])}${split(ch?.integrated, true)}`];
      case 'occlusion':
        return [`Mean drop in ${zOf(t)} ${signed(s.r.occlusion[i])}`, `Mean drop in ${pOf(t)} ${signed(s.r.occlusionProb[i])}`];
    }
  }

  function onHover(p: Panel, e: PointerEvent): void {
    const s = result();
    if (!s) return;
    const shape = shapeNow();
    const W = shape.w;
    const H = shape.h;
    const r = p.canvas.getBoundingClientRect();
    const col = Math.floor(((e.clientX - r.left) / r.width) * W);
    const row = Math.floor(((e.clientY - r.top) / r.height) * H);
    if (col < 0 || col >= W || row < 0 || row >= H) {
      hideTip();
      setHover(null);
      return;
    }
    const i = row * W + col;
    const P = W * H;
    const value = shape.c === 3 ? `Pixel R ${s.x[i].toFixed(2)} · G ${s.x[P + i].toFixed(2)} · B ${s.x[2 * P + i].toFixed(2)}` : `Pixel value ${s.x[i].toFixed(2)}`;
    const lines = [`Row ${row}, column ${col}`, value, ...pixelLines(p.def.key, s, i)];
    showTip(lines.join('\n'), e.clientX, e.clientY);
    setHover(i);
  }

  function setHover(i: number | null): void {
    if (hover === i) return;
    hover = i;
    const s = result();
    if (!s || mode === 'points') return;
    for (const p of panels) {
      const data = values(p.def.key, s);
      paintMap(p, data, scaleMax(p.def, data), s.x);
    }
  }

  // ── Point data: bars per feature ──

  function renderPoints(s: Shown | null): void {
    const d = store.data;
    const dims = d?.points?.dims ?? store.info.dims ?? 2;
    const t = s?.r.target ?? current?.target ?? 0;
    const sc = scoreName(t);
    let defs: { label: string; title: string }[] = [];
    try {
      defs = featureDefs(dims, store.features);
    } catch {
      defs = store.features.map((f) => ({ label: f, title: f }));
    }
    featBody.replaceChildren();
    if (!s || s.r.integrated.length !== defs.length) {
      sentence.textContent = s ? '' : 'Not computed yet.';
      featScale.textContent = '';
      for (const f of defs) {
        featBody.append(
          h('tr', null, h('th', { scope: 'row', title: f.title }, h('span', { class: 'attr-feat-name' }, f.label)), h('td', { class: 'attr-col-slope' }, '–'), h('td', null, bar(0, 0, '–')), h('td', null, bar(0, 0, '–'))),
        );
      }
      featHint.replaceChildren();
      renderPlane(null);
      return;
    }
    const r = s.r;
    const labels = defs.map((f) => f.label);
    sentence.textContent = featureSentence(labels, r.integrated, sc, r.igExpected);
    const max = Math.max(maxAbs(r.gradInput), maxAbs(r.integrated));
    defs.forEach((f, k) => {
      featBody.append(
        h(
          'tr',
          null,
          h('th', { scope: 'row', title: f.title }, h('span', { class: 'attr-feat-name' }, f.label), h('span', { class: 'attr-feat-value' }, `= ${fixed(s.x[k], 2)}`)),
          h('td', { class: 'attr-col-slope' }, signed(r.gradient[k])),
          h('td', null, bar(r.gradInput[k], max, signed(r.gradInput[k]))),
          h('td', null, bar(r.integrated[k], max, signed(r.integrated[k]))),
        ),
      );
    });
    featScale.textContent = max > 0 ? `Both bar columns share one scale: a full half-bar is ${sig(max)}.` : 'Every value is 0.';
    const kinks = kinkText(r.kinks, 'features');
    // The strongest feature by integrated gradients, to read one value out in words.
    let top = 0;
    for (let k = 1; k < labels.length; k++) if (Math.abs(r.integrated[k]) > Math.abs(r.integrated[top])) top = k;
    const v = r.integrated[top];
    const example =
      Math.abs(v) > 1e-6
        ? `Reading a bar: ${labels[top]} ${signed(v)} means that growing ${labels[top]} from 0 to its value here (${fixed(s.x[top], 2)}), on the straight path from the origin, ${v > 0 ? 'raised' : 'lowered'} the score for ${sc} by ${sig(Math.abs(v))} in all.`
        : '';
    featHint.replaceChildren(
      h('p', { class: 'hint' }, `Slope is the gradient ∂z/∂f: how fast the score for ${sc} changes as one feature grows. Gradient × input multiplies it by the feature’s value. Integrated gradients average the slope along the straight path from the origin, so the bars add up to the whole change in the score.`),
      example ? h('p', { class: 'hint' }, example) : '',
      kinks ? h('p', { class: 'hint' }, kinks) : '',
    );
    renderPlane(s);
  }

  /** A bar from a centre line: right and red for positive, left and blue for negative. */
  function bar(v: number, max: number, text: string): HTMLElement {
    const w = max > 0 ? Math.min(1, Math.abs(v) / max) * 50 : 0;
    const b = h('span', { class: `attr-bar${v < 0 ? ' is-neg' : ''}`, style: { width: `${w}%`, left: v < 0 ? `${50 - w}%` : '50%' } });
    return h('span', { class: 'attr-bar-cell' }, h('span', { class: 'attr-bar-track', 'aria-hidden': 'true' }, b), h('span', { class: 'attr-bar-num' }, text));
  }

  // ── Point data: the map of the plane ──

  const planeEval = new PointEvaluator();
  const planeTex = document.createElement('canvas');
  let planeRect: Rect = { x: 0, y: 0, w: 0, h: 0 };
  let planeR = 1.25;
  let planeSize = 240;

  /** The point's coordinates, when the shown probe has them. */
  const coordsOf = (s: Shown | null): Float32Array | null => {
    const c = s?.probe.coords;
    const dims = store.data?.points?.dims;
    return c && dims && c.length === dims ? c : null;
  };

  function planeField(s: Shown): PlaneField | null {
    if (s.field !== undefined) return s.field;
    const d = store.data;
    const c = coordsOf(s);
    s.field = null;
    if (!d?.points || !c || !s.net || !netFits()) return null;
    planeEval.sync(s.net);
    planeR = pointDomain(d);
    const fixedAt = Array.from(c);
    s.field = evaluatePlane(planeEval, d.points.dims, [0, 1], fixedAt, planeR, PLANE_RES).field;
    return s.field;
  }

  function renderPlane(s: Shown | null): void {
    const pal = palette();
    const d = store.data;
    const dims = d?.points?.dims ?? 2;
    const box = planeCanvas.parentElement!;
    const avail = box.clientWidth || 240;
    planeSize = Math.max(200, Math.min(300, avail));
    const inner = planeSize - Math.max(PLANE_PAD.l + PLANE_PAD.r, PLANE_PAD.t + PLANE_PAD.b);
    planeRect = { x: PLANE_PAD.l, y: PLANE_PAD.t, w: inner, h: inner };
    const W = PLANE_PAD.l + inner + PLANE_PAD.r;
    const H = PLANE_PAD.t + inner + PLANE_PAD.b;
    const ctx = fitCanvas(planeCanvas, W, H);
    ctx.fillStyle = pal.surface;
    ctx.fillRect(planeRect.x, planeRect.y, planeRect.w, planeRect.h);
    const c = coordsOf(s);
    const field = s ? planeField(s) : null;
    planeTitle.textContent = dims === 3 && c ? `The point, on the slice x₃ = ${fixed(c[2], 2)}` : 'The point';
    if (!s || !field || !c || !d?.points) {
      drawAxes(ctx, planeRect, planeR, [0, 1], true);
      planeCaption.textContent = s && !c ? 'This input has no position on the plane.' : '';
      planeHint.textContent = '';
      planeCanvas.setAttribute('aria-label', 'Map of the input plane: not computed yet');
      return;
    }
    planeR = field.r;
    drawField(ctx, field, planeRect, false, regionCanvas(field, false, planeTex));
    drawAxes(ctx, planeRect, planeR, [0, 1], true);
    const sx = (u: number) => planeRect.x + ((u + planeR) / (2 * planeR)) * planeRect.w;
    const sy = (v: number) => planeRect.y + ((planeR - v) / (2 * planeR)) * planeRect.h;
    ctx.save();
    ctx.beginPath();
    ctx.rect(planeRect.x, planeRect.y, planeRect.w, planeRect.h);
    ctx.clip();
    if (dims === 2) {
      // The test points, as hollow rings in their class colours (as in 03).
      const tc = d.points.testCoords;
      for (let i = 0; i < d.testY.length; i++) dotTest(ctx, sx(tc[2 * i]), sy(tc[2 * i + 1]), 2.2, classColor(d.testY[i]), pal.surface);
    }
    const x0 = sx(c[0]);
    const y0 = sy(c[1]);
    // The steepest way up for the target's score, in the plane: an ink arrow with a surface halo.
    const g = s.r.coordGrad;
    const gx = g ? g[0] : 0;
    const gy = g ? g[1] : 0;
    const len = Math.hypot(gx, gy);
    if (len > 1e-9) {
      const L = Math.max(26, 0.22 * planeRect.w);
      const ux = gx / len;
      const uy = -gy / len; // screen y points down
      arrow(ctx, x0, y0, x0 + ux * L, y0 + uy * L, pal.ink, pal.surface);
    }
    drawInputCross(ctx, x0, y0, pal, 10, 4);
    ctx.restore();

    const t = s.r.target;
    const sc = scoreName(t);
    const pos = Array.from(c, (v, i) => `${axisName(i)} ${fixed(v, 2)}`).join(' · ');
    /** "a · b · c" as pieces that never break inside one term. */
    const terms = (parts: string[]) => parts.flatMap((t, k) => [k ? ' · ' : '', h('span', { class: 'attr-nowrap' }, t)]);
    planeCaption.replaceChildren(
      h('span', null, ...terms(Array.from(c, (v, i) => `${axisName(i)} ${fixed(v, 2)}`))),
      g ? h('span', null, ...terms(Array.from(g, (v, i) => `∂z/∂${axisName(i)} ${signed(v)}`))) : '',
    );
    const full = g ? Math.hypot(...Array.from(g)) : 0;
    const move = 'Click the map, or use the arrow keys on it, to move the point.';
    planeHint.textContent =
      len <= 1e-9
        ? `Moving the point a little barely changes the score for ${sc} here. ${move}`
        : dims === 3
          ? `The arrow is the x₁–x₂ part of the way to move the point that raises the score for ${sc} fastest (about ${sig(full)} per unit of distance, counting x₃). ${move}`
          : `The arrow points the way to move the point that raises the score for ${sc} fastest, by about ${sig(full)} per unit of distance. ${move}`;
    planeCanvas.setAttribute(
      'aria-label',
      `Map of the input plane${dims === 3 ? ` at x₃ = ${fixed(c[2], 2)}` : ''}, coloured by predicted class, with the current point at ${pos}${len > 1e-9 ? ` and an arrow toward a higher score for ${sc}` : ''}.`,
    );
  }

  function arrow(ctx: CanvasRenderingContext2D, x0: number, y0: number, x1: number, y1: number, ink: string, surface: string): void {
    const ang = Math.atan2(y1 - y0, x1 - x0);
    const head = 7;
    const path = () => {
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1 - Math.cos(ang) * 2, y1 - Math.sin(ang) * 2);
      ctx.moveTo(x1, y1);
      ctx.lineTo(x1 - head * Math.cos(ang - 0.45), y1 - head * Math.sin(ang - 0.45));
      ctx.moveTo(x1, y1);
      ctx.lineTo(x1 - head * Math.cos(ang + 0.45), y1 - head * Math.sin(ang + 0.45));
    };
    ctx.save();
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    path();
    ctx.strokeStyle = surface;
    ctx.lineWidth = 5;
    ctx.stroke();
    path();
    ctx.strokeStyle = ink;
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.restore();
  }

  /** Data coordinates (all dims) of a position on the plane, or null outside it. */
  function planeAt(clientX: number, clientY: number): Float32Array | null {
    const s = result();
    const c = coordsOf(s);
    if (!c) return null;
    const b = planeCanvas.getBoundingClientRect();
    const x = clientX - b.left;
    const y = clientY - b.top;
    const { x: rx, y: ry, w, h: hh } = planeRect;
    if (x < rx || x > rx + w || y < ry || y > ry + hh) return null;
    const out = Float32Array.from(c);
    out[0] = -planeR + ((x - rx) / w) * 2 * planeR;
    out[1] = planeR - ((y - ry) / hh) * 2 * planeR;
    return out;
  }

  planeCanvas.addEventListener('pointermove', (e) => {
    const at = planeAt(e.clientX, e.clientY);
    if (!at) return hideTip();
    showTip(`${Array.from(at, (v, i) => `${axisName(i)} ${fixed(v, 2)}`).join(' · ')}\nClick to move the point here`, e.clientX, e.clientY);
  });
  planeCanvas.addEventListener('pointerleave', () => hideTip());
  planeCanvas.addEventListener('click', (e) => {
    const at = planeAt(e.clientX, e.clientY);
    if (at) pickAt(at, null, false);
  });
  planeCanvas.addEventListener('keydown', (e) => {
    const dirs: Record<string, [number, number]> = { ArrowRight: [1, 0], ArrowLeft: [-1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1] };
    const dir = dirs[e.key];
    const c = coordsOf(result());
    if (!dir || !c) return;
    e.preventDefault();
    const step = e.shiftKey ? 0.2 : 0.05;
    const next = Float32Array.from(c);
    next[0] = Math.max(-planeR, Math.min(planeR, Math.round((c[0] + dir[0] * step) * 1000) / 1000));
    next[1] = Math.max(-planeR, Math.min(planeR, Math.round((c[1] + dir[1] * step) * 1000) / 1000));
    pickAt(next, null, false);
  });

  // ── Layout ──

  /** Largest whole number of screen pixels per image pixel that fits a panel, 3–7. */
  function measure(): boolean {
    if (mode === 'points') return false;
    const w = panels[0].fig.clientWidth;
    if (!w) return false;
    const side = shapeNow().w;
    const cell = Math.max(3, Math.min(7, Math.floor(w / side)));
    const next = cell * side;
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
      for (const p of panels) paintRamp(p);
    });
  };
  let lastPlaneWidth = 0;
  new ResizeObserver(() => {
    const planeWidth = mode === 'points' ? planeCanvas.parentElement!.clientWidth : 0;
    const planeChanged = planeWidth !== lastPlaneWidth;
    lastPlaneWidth = planeWidth;
    if (measure() || (mode === 'points' && planeChanged)) schedule();
  }).observe(grid);

  // ── Events ──

  store.on('probe', () => {
    const p = store.probe;
    // Re-selecting the same input (same key and values) keeps the pick; anything else resets it.
    const changed = !p || !lastInput || p.key !== lastInput.key || !sameX(p.x, lastInput.x);
    lastInput = p ? { key: p.key, x: p.x.slice() } : null;
    if (changed) pinned = null;
    soon();
  });
  // A new dataset: new classes, and possibly a new kind of input. The network follows at once.
  store.on('dataset', () => {
    if (modeNow() === mode && chipKey === `${store.info.id}|${store.info.classes.join('|')}`) return;
    pinned = null;
    applyMode();
  });
  // A new or re-initialised network: the old maps explain weights that are gone, so clear them
  // now. The status line's refresh fills them in for the new network.
  store.on('model', () => {
    stopRun();
    shown = null;
    hover = null;
    error = null;
    if (modeNow() !== mode || chipKey !== `${store.info.id}|${store.info.classes.join('|')}`) applyMode();
    current = store.valid ? requestFor(store.probe) : null;
    measure();
    renderControls();
    renderPanels();
    renderNotice();
    setBusy();
    for (const p of panels) paintRamp(p);
  });
  store.on('data', () => {
    if (!current && store.valid) current = requestFor(store.probe);
    renderControls();
  });
  // Weight updates and first views go through the shared policy in syncedSection. Only an input
  // that changed while the section was off screen needs its own catch-up when it scrolls in.
  new IntersectionObserver(
    (entries) => {
      onScreen = entries[entries.length - 1].isIntersecting;
      if (!onScreen || !inputDirty) return;
      if (shown || running) soon();
      else sync.request(); // nothing computed yet: the first view follows the shared policy
    },
    { rootMargin: '200px 0px' },
  ).observe(root);
  onThemeChange(schedule);

  applyMode();
  measure();
  renderControls();
  renderPanels();
  for (const p of panels) paintRamp(p);
}

const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);

function pctText(p: number): string {
  const v = p * 100;
  return `${v >= 99.95 && v < 100 ? '99.9' : v.toFixed(1)}%`;
}

function hasInk(x: Float32Array): boolean {
  for (let i = 0; i < x.length; i++) if (x[i] !== 0) return true;
  return false;
}
