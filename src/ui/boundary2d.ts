import { addCustom, setProbe } from '../actions';
import { fixed } from '../analysis/stats';
import { sampleCaption, sampleInput, type Data } from '../data/datasets';
import { featurize, type FeatureId } from '../data/features';
import { gridCoords, pointDomain, PointEvaluator } from '../data/grid';
import type { Network } from '../nn/network';
import { store } from '../store';
import { h } from './dom';
import { fitCanvas } from './draw';
import { axisName, boundarySegments, paintRegions, resolutionFor, ticks, type RGB } from './boundaryMath';
import { drawMarks, type MarkKind } from './marks';
import { onThemeChange, palette } from './theme';
import { hideTip, showTip } from './tip';

/**
 * Section 03 for point datasets, 2-D part: the predicted class over the plane with the decision
 * boundary drawn on top. The plane renderer here (PlaneCanvas) also draws the slice map of the
 * 3-D view, and the sliced evaluation (Live, GridJob) serves sections 02 and 07 too.
 */

// ── Shared view options (toggles above the plot) ─────────────────────────

export const boundaryOpts = { showTest: true, discrete: false, adding: false, addClass: 0 };
const optListeners = new Set<() => void>();
export function setBoundaryOpts(patch: Partial<typeof boundaryOpts>): void {
  Object.assign(boundaryOpts, patch);
  for (const fn of optListeners) fn();
}
export const onBoundaryOpts = (fn: () => void) => optListeners.add(fn);

// ── Throttled, visibility-aware redraws ──────────────────────────────────

/**
 * Runs `render(evaluate, draw)` in animation frames while `el` is on screen (or near it).
 * Evaluations of the network are spaced at least `gap` ms apart; the views set the gap to four
 * times the main-thread time of their last full pass (evaluation and painting), so a slow network
 * is redrawn less often instead of taking over the page. A render that returns true has more work
 * to do (an evaluation sliced across frames) and is called again on the next frame; no new
 * evaluation starts until it is done. Requests that only redraw run on the next frame. Work
 * requested while off screen waits until the element scrolls into view.
 */
export class Live {
  private visible = false;
  private needEval = false;
  private needDraw = false;
  /** A sliced pass is under way. */
  private working = false;
  private raf = 0;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private lastEval = -Infinity;
  active = true;

  constructor(
    el: Element,
    private render: (evaluate: boolean, draw: boolean) => boolean | void,
    /** Minimum ms between evaluations; raised for slow networks so training keeps most of the thread. */
    public gap = 100,
  ) {
    new IntersectionObserver(
      (es) => {
        this.visible = es.some((e) => e.isIntersecting);
        if (this.visible && (this.needEval || this.needDraw || this.working)) this.kick();
      },
      { rootMargin: '240px 0px' },
    ).observe(el);
  }

  /** Asks for a redraw; `evaluate` when the network or the data changed. */
  request(evaluate: boolean): void {
    if (evaluate) this.needEval = true;
    else this.needDraw = true;
    this.kick();
  }

  /**
   * Main-thread time a drawing costs after the script has run: the browser rasterises a canvas
   * when it commits the frame, which no timer around the drawing calls sees. Estimated from how
   * much later than due the next frame starts, smoothed over draws.
   */
  drawCost = 0;
  /** Call after drawing, with the time the frame's work started and how long the script took. */
  noteDraw(t0: number, work: number): void {
    requestAnimationFrame(() => {
      const extra = Math.max(0, performance.now() - t0 - Math.max(1000 / 60, work));
      this.drawCost = 0.6 * this.drawCost + 0.4 * Math.min(extra, 500);
    });
  }

  /** Forgets a pass under way (the view dropped it) and asks for a fresh evaluation. */
  restart(): void {
    this.working = false;
    this.request(true);
  }

  private kick(): void {
    if (!this.visible || !this.active) return;
    const wait = this.needEval && !this.working ? this.lastEval + this.gap - performance.now() : 0;
    if (this.needDraw || this.working || wait <= 0) {
      if (!this.raf) this.raf = requestAnimationFrame(() => this.run());
    }
    if (this.needEval && !this.working && wait > 0 && this.timer === null) {
      this.timer = setTimeout(() => {
        this.timer = null;
        this.kick();
      }, wait);
    }
  }

  private run(): void {
    this.raf = 0;
    if (!this.visible || !this.active) return;
    const due = this.needEval && !this.working && performance.now() >= this.lastEval + this.gap - 1;
    if (!due && !this.needDraw && !this.working) return;
    if (due) {
      this.needEval = false;
      this.lastEval = performance.now();
    }
    const draw = this.needDraw;
    this.needDraw = false;
    this.working = this.render(due, draw) === true;
    if (this.working || this.needEval || this.needDraw) this.kick();
  }
}

// ── Evaluation over many points, sliced across frames ────────────────────

/** Longest slice of evaluation per frame, in ms, while training runs and while it is paused. */
export const SLICE_MS = { running: 6, paused: 10 };
export const sliceMs = (): number => (store.running ? SLICE_MS.running : SLICE_MS.paused);

/**
 * Evaluates the network at many points (n × dims raw coordinates) a slice at a time, so a large
 * network never holds the main thread for long: each step runs for about `ms` and a pass may take
 * several frames. The caller syncs the evaluator once, before the pass, and the features are
 * fixed when the job is made, so a pass stays consistent even if the page's network changes
 * meanwhile (the views drop such passes).
 */
export class GridJob {
  readonly n: number;
  probs: Float32Array | null = null;
  acts: Float32Array[] | null = null;
  /** Main-thread time spent on this pass so far, in ms. */
  busy = 0;
  private i = 0;
  private readonly features: FeatureId[];
  /** Points per ms, measured as the pass goes. */
  private rate = 0;

  constructor(
    private ev: PointEvaluator,
    private coords: Float32Array,
    private dims: 2 | 3,
    private activations = false,
  ) {
    this.n = coords.length / dims;
    this.features = store.features.slice();
  }

  get done(): boolean {
    return this.i >= this.n;
  }

  /** Evaluates points for about `ms` (always a few); true when every point is done. */
  step(ms: number, perPoint: number): boolean {
    const t0 = performance.now();
    let rate = this.rate || 1 / Math.max(1e-4, perPoint);
    while (this.i < this.n) {
      const left = ms - (performance.now() - t0);
      if (left <= 0.25 && this.i > 0) break;
      const a = this.i;
      const b = Math.min(this.n, a + Math.max(4, Math.min(4096, Math.floor(left * rate))));
      const s0 = performance.now();
      const out = this.ev.evaluate(this.coords.subarray(a * this.dims, b * this.dims), this.dims, this.features, { activations: this.activations });
      const K = out.probs.length / (b - a);
      this.probs ??= new Float32Array(this.n * K);
      this.probs.set(out.probs, a * K);
      if (out.acts) {
        const acts = (this.acts ??= out.acts.map((x) => new Float32Array((x.length / (b - a)) * this.n)));
        out.acts.forEach((x, j) => acts[j].set(x, a * (x.length / (b - a))));
      }
      this.i = b;
      const dt = performance.now() - s0;
      if (dt > 0.05) rate = this.rate = this.rate ? 0.5 * this.rate + 0.5 * ((b - a) / dt) : (b - a) / dt;
      if (this.i > 0 && performance.now() - t0 >= ms) break;
    }
    this.busy += performance.now() - t0;
    return this.done;
  }
}

// ── Evaluation over a plane ──────────────────────────────────────────────

export interface PlaneField {
  res: number;
  classes: number;
  /** res² × classes probabilities, row 0 at the top. */
  probs: Float32Array;
  dims: 2 | 3;
  /** Data axes along the map's x (left to right) and y (bottom to top). */
  axes: [number, number];
  /** Coordinates of the axes not on the map (3-D slices). */
  fixedAt: number[];
  r: number;
  /** Decision-boundary segments in grid units (cached). */
  segs: number[];
  /** Bumped per evaluation, so cached pictures know when to repaint. */
  rev: number;
}

let fieldRev = 0;

function planeField(probs: Float32Array, dims: 2 | 3, axes: [number, number], fixedAt: number[], r: number, res: number): PlaneField {
  const classes = probs.length / (res * res);
  const segs = boundarySegments(probs, classes, res, res);
  return { res, classes, probs, dims, axes, fixedAt: fixedAt.slice(), r, segs, rev: ++fieldRev };
}

/** Evaluates the network on a res × res grid over one plane at once; returns the field and its cost. */
export function evaluatePlane(ev: PointEvaluator, dims: 2 | 3, axes: [number, number], fixedAt: number[], r: number, res: number): { field: PlaneField; ms: number } {
  const t0 = performance.now();
  const coords = gridCoords(dims, res, r, axes, fixedAt);
  const { probs } = ev.evaluate(coords, dims, store.features);
  return { field: planeField(probs, dims, axes, fixedAt, r, res), ms: performance.now() - t0 };
}

/** One plane evaluated a slice at a time (see GridJob). */
export class PlaneJob {
  readonly job: GridJob;
  constructor(
    ev: PointEvaluator,
    readonly dims: 2 | 3,
    readonly axes: [number, number],
    readonly fixedAt: number[],
    readonly r: number,
    readonly res: number,
  ) {
    this.job = new GridJob(ev, gridCoords(dims, res, r, axes, fixedAt), dims);
  }
  get busy(): number {
    return this.job.busy;
  }
  /** Runs one slice; returns the finished field (boundary lines included) once every point is done. */
  step(ms: number, perPoint: number): PlaneField | null {
    if (!this.job.step(ms, perPoint)) return null;
    const t0 = performance.now();
    const f = planeField(this.job.probs!, this.dims, this.axes, this.fixedAt, this.r, this.res);
    this.job.busy += performance.now() - t0;
    return f;
  }
}

/** True when the page network takes the current point features (it may lag a dataset switch). */
export function netFits(): boolean {
  return store.info.kind === 'points' && store.net.inputSize === store.features.length && !!store.data?.points;
}

/**
 * A private evaluator that follows the page's network: `ready()` re-syncs it whenever the page
 * network was replaced (a new architecture or new features), so a query between two passes never
 * runs the old network on the new features.
 */
export class Evaluator {
  readonly ev = new PointEvaluator();
  private net: Network | null = null;
  /** Copies the page network's current weights (and architecture) in. */
  sync(): void {
    this.ev.sync(store.net);
    this.net = store.net;
  }
  /** True when the evaluator holds the page's current network (its weights may be a pass behind). */
  get current(): boolean {
    return this.net === store.net;
  }
  /** Class probabilities at one point, or null when the network does not fit the features. */
  probsAt(coords: Float32Array, dims: 2 | 3): Float32Array | null {
    if (!netFits()) return null;
    if (!this.current) this.sync();
    return this.ev.evaluate(coords, dims, store.features).probs;
  }
}

/**
 * Measured cost of evaluating the network at one point, smoothed and kept per weight (plus a
 * fixed overhead), so a switch to a much larger network is anticipated before it is measured.
 */
export class Cost {
  /** ms per point per (parameter + 50), from measurements; a conservative guess at first. */
  private unit = 4e-5;
  private measured = false;
  add(ms: number, n: number): void {
    const u = ms / Math.max(1, n) / (store.net.paramCount + 50);
    this.unit = this.measured ? 0.6 * this.unit + 0.4 * u : u;
    this.measured = true;
  }
  /** Estimated ms per point for the page's current network. */
  get perPoint(): number {
    return this.unit * (store.net.paramCount + 50);
  }
  /** Grid side for a pass of about `b.budget` ms; `b.min` gives way when it would cost more than `b.cap`. */
  res(b: Budget, dims: 2 | 3): number {
    return resolutionFor(this.perPoint, b.budget, dims, b.min, b.max, b.cap, b.floor);
  }
}

/** Main-thread time for one pass of a view (ms) and the grid sides it may use. */
export interface Budget {
  budget: number;
  min: number;
  max: number;
  /** Most time a pass may take before the grid gets coarser than `min`. */
  cap: number;
  /** Coarsest grid side ever used. */
  floor: number;
}

// ── Points and picks ─────────────────────────────────────────────────────

export interface PointRef {
  split: 'train' | 'test' | 'custom';
  i: number;
}

export interface Marker extends PointRef {
  /** Data coordinates on the map's two axes. */
  u: number;
  v: number;
  cls: number;
  alpha: number;
}

const SUBS = ['₁', '₂', '₃'];

/** "Point (0.31, −0.52)". */
export const pointName = (c: ArrayLike<number>) => `Point (${Array.from(c, (v) => fixed(v, 2)).join(', ')})`;

/** Coordinates of a data point. */
export function pointCoords(d: Data, ref: PointRef): Float32Array | null {
  const p = d.points!;
  if (ref.split === 'custom') return store.custom.find((c) => c.id === ref.i)?.coords ?? null;
  const src = ref.split === 'train' ? p.trainCoords : p.testCoords;
  return src.slice(ref.i * p.dims, (ref.i + 1) * p.dims);
}

export function pointLabel(d: Data, ref: PointRef): number {
  if (ref.split === 'custom') return store.custom.find((c) => c.id === ref.i)?.y ?? 0;
  return (ref.split === 'train' ? d.trainY : d.testY)[ref.i];
}

export function describePoint(d: Data, ref: PointRef): string {
  const y = pointLabel(d, ref);
  if (ref.split === 'custom') return `Your point · ${d.info.classes[y]}`;
  return sampleCaption(d.info, ref.split, ref.i, y);
}

/**
 * What a click on the plane does: in "Add points" mode (or with Shift) it adds a labelled point to
 * the training set; on a data point it makes that point the network's input; elsewhere the clicked
 * coordinate becomes the input.
 */
export function pickAt(coords: Float32Array, hit: PointRef | null, add: boolean): void {
  const d = store.data;
  if (!d?.points || !netFits()) return;
  const dims = d.points.dims;
  if (add) {
    const k = Math.min(boundaryOpts.addClass, d.info.classes.length - 1);
    addCustom({ x: featurize(coords, dims, store.features), y: k, origin: 'point', name: pointName(coords), coords: coords.slice() });
    return;
  }
  if (hit && hit.split !== 'custom') {
    const y = pointLabel(d, hit);
    setProbe({ x: sampleInput(d, hit.split, hit.i), label: y, caption: sampleCaption(d.info, hit.split, hit.i, y), key: `${hit.split}:${hit.i}`, coords: pointCoords(d, hit)! });
    return;
  }
  if (hit) {
    const c = store.custom.find((e) => e.id === hit.i);
    if (c?.coords) {
      setProbe({ x: c.x, label: c.y, caption: `Your point · ${d.info.classes[c.y]}`, key: `custom:${c.id}`, coords: c.coords });
      return;
    }
  }
  setProbe({ x: featurize(coords, dims, store.features), label: null, caption: pointName(coords), key: `pt:${Array.from(coords, (v) => v.toFixed(3)).join(',')}`, coords: coords.slice() });
}

/** Tooltip text for a coordinate: where it is and what the network predicts there. */
export function probsText(d: Data, coords: Float32Array, probs: ArrayLike<number> | null, hit: PointRef | null): string {
  const lines: string[] = [];
  if (hit) lines.push(describePoint(d, hit));
  lines.push(Array.from(coords, (v, i) => `x${SUBS[i]} ${fixed(v, 2)}`).join(' · '));
  if (probs) {
    let best = 0;
    for (let k = 1; k < probs.length; k++) if (probs[k] > probs[best]) best = k;
    const w = Math.max(...d.info.classes.map((c) => c.length));
    // No-break spaces: the tooltip collapses ordinary runs of spaces.
    const NB = ' ';
    d.info.classes.forEach((name, k) => {
      const pct = `${(100 * probs[k]).toFixed(1)}%`;
      lines.push(`${k === best ? '▸' : NB}${NB}${name.padEnd(w, NB)}${NB}${pct.padStart(6, NB)}`);
    });
  }
  return lines.join('\n');
}

// ── Drawing helpers ──────────────────────────────────────────────────────

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** Training point: filled disc in the class colour with a thin ink outline (other sections' plots). */
export function dotTrain(ctx: CanvasRenderingContext2D, x: number, y: number, rad: number, fill: string, ink: string): void {
  ctx.beginPath();
  ctx.arc(x, y, rad, 0, 2 * Math.PI);
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.lineWidth = 0.9;
  ctx.strokeStyle = ink;
  ctx.stroke();
}

/** Test point: a hollow ring in the class colour, with a surface halo so it reads on any region. */
export function dotTest(ctx: CanvasRenderingContext2D, x: number, y: number, rad: number, color: string, surface: string): void {
  ctx.beginPath();
  ctx.arc(x, y, rad, 0, 2 * Math.PI);
  ctx.lineWidth = 3.4;
  ctx.strokeStyle = surface;
  ctx.stroke();
  ctx.lineWidth = 1.7;
  ctx.strokeStyle = color;
  ctx.stroke();
}

/** A point you added: a diamond in the class colour with a heavy ink outline. */
export function dotCustom(ctx: CanvasRenderingContext2D, x: number, y: number, rad: number, fill: string, ink: string): void {
  const s = rad * 1.45;
  ctx.beginPath();
  ctx.moveTo(x, y - s);
  ctx.lineTo(x + s, y);
  ctx.lineTo(x, y + s);
  ctx.lineTo(x - s, y);
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.lineWidth = 1.6;
  ctx.strokeStyle = ink;
  ctx.stroke();
}

/** The current input: an accent cross with a surface halo. */
export function probeCross(ctx: CanvasRenderingContext2D, x: number, y: number, size: number, accent: string, surface: string, faint = false): void {
  probeCrosses(ctx, [x, y], size, accent, surface, faint);
}

/** Several current-input crosses (one per map) in one path: a surface halo under an accent cross. */
export function probeCrosses(ctx: CanvasRenderingContext2D, xy: ArrayLike<number>, size: number, accent: string, surface: string, faint = false): void {
  if (xy.length < 2) return;
  ctx.lineCap = 'round';
  ctx.beginPath();
  for (let i = 0; i + 1 < xy.length; i += 2) {
    const x = xy[i];
    const y = xy[i + 1];
    ctx.moveTo(x - size, y - size);
    ctx.lineTo(x + size, y + size);
    ctx.moveTo(x + size, y - size);
    ctx.lineTo(x - size, y + size);
  }
  const s = Math.min(1, size / 6.5);
  ctx.strokeStyle = surface;
  ctx.lineWidth = Math.max(3.8, 5.5 * s);
  ctx.globalAlpha = faint ? 0.6 : 1;
  ctx.stroke();
  ctx.strokeStyle = accent;
  ctx.lineWidth = Math.max(1.8, 2.4 * s);
  ctx.stroke();
  ctx.globalAlpha = 1;
  ctx.lineCap = 'butt';
}

/**
 * Draws point markers in batches, one path per class, kind and (rounded) opacity: test points
 * first, then training points, then yours. Returns nothing; `X`/`Y` map data to canvas pixels.
 */
export function drawMarkers(ctx: CanvasRenderingContext2D, markers: Marker[], X: (u: number) => number, Y: (v: number) => number, rad: number): void {
  const p = palette();
  const groups = new Map<string, { cls: number; kind: MarkKind; alpha: number; xy: number[] }>();
  for (const kind of ['test', 'train', 'custom'] as const) {
    for (const m of markers) {
      if (m.split !== kind) continue;
      const alpha = Math.round(m.alpha * 20) / 20;
      const key = `${kind}:${m.cls}:${alpha}`;
      let g = groups.get(key);
      if (!g) groups.set(key, (g = { cls: m.cls, kind, alpha, xy: [] }));
      g.xy.push(X(m.u), Y(m.v));
    }
  }
  for (const g of groups.values()) {
    ctx.globalAlpha = g.alpha;
    drawMarks(ctx, g.cls, g.kind, g.xy, rad, { color: p.cat[g.cls % 10], ink: p.ink, surface: p.surface });
  }
  ctx.globalAlpha = 1;
}

let regionScratch: HTMLCanvasElement | null = null;

/** Paints a field's class regions into a canvas of its own resolution (for drawImage / textures). */
export function regionCanvas(field: PlaneField, discrete: boolean, target?: HTMLCanvasElement): HTMLCanvasElement {
  const c = target ?? (regionScratch ??= document.createElement('canvas'));
  if (c.width !== field.res) c.width = field.res;
  if (c.height !== field.res) c.height = field.res;
  const cx = c.getContext('2d')!;
  const img = cx.createImageData(field.res, field.res);
  const p = palette().rgb;
  paintRegions(img.data, field.probs, field.classes, p.cat as RGB[], p.surface as RGB, discrete);
  cx.putImageData(img, 0, 0);
  return c;
}

/** Draws the class regions and the boundary line of `field` into `rect`. */
export function drawField(ctx: CanvasRenderingContext2D, field: PlaneField, rect: Rect, discrete: boolean, tex?: HTMLCanvasElement): void {
  const p = palette();
  const src = tex ?? regionCanvas(field, discrete);
  ctx.save();
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(src, 0, 0, field.res, field.res, rect.x, rect.y, rect.w, rect.h);
  ctx.restore();
  const s = field.segs;
  const fx = rect.w / field.res;
  const fy = rect.h / field.res;
  ctx.beginPath();
  for (let i = 0; i < s.length; i += 4) {
    ctx.moveTo(rect.x + (s[i] + 0.5) * fx, rect.y + (s[i + 1] + 0.5) * fy);
    ctx.lineTo(rect.x + (s[i + 2] + 0.5) * fx, rect.y + (s[i + 3] + 0.5) * fy);
  }
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = p.ink;
  ctx.lineWidth = rect.w > 300 ? 2.2 : 1.8;
  ctx.stroke();
  ctx.lineCap = 'butt';
}

/** Ticks and axis names around a square plot of [−r, r]². */
export function drawAxes(ctx: CanvasRenderingContext2D, rect: Rect, r: number, axes: [number, number], small = false): void {
  const p = palette();
  ctx.strokeStyle = p.ink;
  ctx.lineWidth = 1.5;
  ctx.strokeRect(rect.x - 0.75, rect.y - 0.75, rect.w + 1.5, rect.h + 1.5);
  const ts = ticks(r, small || rect.w < 260 ? 1 : 0.5);
  ctx.font = `400 ${small ? 9.5 : 10}px "IBM Plex Mono", ui-monospace, monospace`;
  ctx.fillStyle = p.muted;
  ctx.textBaseline = 'top';
  ctx.textAlign = 'center';
  for (const t of ts) {
    const x = rect.x + ((t + r) / (2 * r)) * rect.w;
    ctx.fillStyle = p.ink;
    ctx.fillRect(Math.round(x) - 0.5, rect.y + rect.h + 1, 1, 4);
    ctx.fillStyle = p.muted;
    ctx.fillText(fixed(t, 1), x, rect.y + rect.h + 7);
  }
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (const t of ts) {
    const y = rect.y + ((r - t) / (2 * r)) * rect.h;
    ctx.fillStyle = p.ink;
    ctx.fillRect(rect.x - 5, Math.round(y) - 0.5, 4, 1);
    ctx.fillStyle = p.muted;
    ctx.fillText(fixed(t, 1), rect.x - 7, y);
  }
  ctx.fillStyle = p.ink;
  ctx.font = `700 ${small ? 11 : 12.5}px Archivo, "Helvetica Neue", Arial, sans-serif`;
  ctx.textAlign = 'right';
  ctx.textBaseline = 'top';
  ctx.fillText(axisName(axes[0]), rect.x + rect.w, rect.y + rect.h + (small ? 18 : 20));
  ctx.textAlign = 'left';
  ctx.textBaseline = 'bottom';
  ctx.fillText(axisName(axes[1]), rect.x - (small ? 26 : 30), rect.y - 4);
}

// ── A square map of one plane, with points, hover and picking ────────────

export interface PlaneHost {
  field(): PlaneField | null;
  markers(): Marker[];
  /** Where the current input sits on the map (u, v), and whether it lies on the plane. */
  probe(): { u: number; v: number; on: boolean } | null;
  /** Full data coordinates of a map position. */
  coordsAt(u: number, v: number): Float32Array;
  probsAt(coords: Float32Array): ArrayLike<number> | null;
}

/** Margins around the plot inside the canvas, for ticks and axis names. */
const PAD = { l: 38, r: 10, t: 22, b: 38 };
const PAD_SMALL = { l: 34, r: 8, t: 20, b: 34 };

let hintId = 0;

/** A visually hidden description, for aria-describedby. */
export function srHint(text: string): HTMLElement {
  return h('p', { class: 'bd-sr', id: `bd-hint-${++hintId}` }, text);
}

export class PlaneCanvas {
  readonly canvas: HTMLCanvasElement;
  /** How to use the plot from the keyboard; the view puts it in the page next to the canvas. */
  readonly keysHint: HTMLElement;
  rect: Rect = { x: 0, y: 0, w: 0, h: 0 };
  private markers: Marker[] = [];
  private tex = document.createElement('canvas');
  private texKey = '';

  constructor(
    private host: PlaneHost,
    private small: boolean,
    label: string,
  ) {
    this.keysHint = srHint(
      'Arrow keys move the network’s input across the plot, 0.05 per press or 0.25 with Shift. Enter or Space makes that spot the input; with Add points on, it adds a training point there. Clicking a point uses it as the input.',
    );
    // An interactive plot, not a picture: it takes the arrow keys (role application).
    this.canvas = h('canvas', { class: 'bd-plane', role: 'application', 'aria-roledescription': 'plot', tabindex: '0', 'aria-label': label, 'aria-describedby': this.keysHint.id }) as HTMLCanvasElement;
    const c = this.canvas;
    c.addEventListener('pointermove', (e) => this.hover(e));
    c.addEventListener('pointerleave', () => hideTip());
    c.addEventListener('click', (e) => {
      const at = this.at(e);
      if (!at) return;
      pickAt(at.coords, at.hit, boundaryOpts.adding || e.shiftKey);
      this.hover(e);
    });
    c.addEventListener('keydown', (e) => this.key(e));
  }

  /** Lays the canvas out `size` CSS pixels wide (square plot) and draws everything. */
  draw(size: number): void {
    const pad = this.small ? PAD_SMALL : PAD;
    const side = Math.max(120, Math.floor(size - pad.l - pad.r));
    const W = side + pad.l + pad.r;
    const H = side + pad.t + pad.b;
    const ctx = fitCanvas(this.canvas, W, H);
    this.rect = { x: pad.l, y: pad.t, w: side, h: side };
    const field = this.host.field();
    const p = palette();
    const rect = this.rect;
    ctx.fillStyle = p.surface;
    ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
    if (field) {
      const key = `${field.rev}:${boundaryOpts.discrete}:${p.surface}:${p.cat.join()}`;
      if (key !== this.texKey) {
        regionCanvas(field, boundaryOpts.discrete, this.tex);
        this.texKey = key;
      }
      drawField(ctx, field, rect, boundaryOpts.discrete, this.tex);
    }
    const r = field?.r ?? 1.25;
    this.canvas.dataset.plot = `${rect.x} ${rect.y} ${rect.w} ${rect.h}`;
    this.canvas.dataset.r = String(r);
    if (field) {
      this.canvas.dataset.res = String(field.res);
      this.canvas.dataset.rev = String(field.rev);
    }
    // Points: a shape per class, filled for training, outlined for test.
    this.markers = this.host.markers();
    const rad = Math.max(2.6, Math.min(4.2, side / 135));
    const X = (u: number) => rect.x + ((u + r) / (2 * r)) * rect.w;
    const Y = (v: number) => rect.y + ((r - v) / (2 * r)) * rect.h;
    ctx.save();
    ctx.beginPath();
    ctx.rect(rect.x - 6, rect.y - 6, rect.w + 12, rect.h + 12);
    ctx.clip();
    drawMarkers(ctx, this.markers, X, Y, rad);
    const pr = this.host.probe();
    if (pr) probeCross(ctx, X(pr.u), Y(pr.v), this.small ? 5 : 6.5, p.accent, p.surface, !pr.on);
    ctx.restore();
    drawAxes(ctx, rect, r, field?.axes ?? [0, 1], this.small);
  }

  private at(e: MouseEvent): { u: number; v: number; coords: Float32Array; hit: PointRef | null } | null {
    const b = this.canvas.getBoundingClientRect();
    const x = e.clientX - b.left;
    const y = e.clientY - b.top;
    const rect = this.rect;
    const r = Number(this.canvas.dataset.r ?? 1.25);
    if (x < rect.x - 4 || y < rect.y - 4 || x > rect.x + rect.w + 4 || y > rect.y + rect.h + 4) return null;
    const u = Math.max(-r, Math.min(r, -r + ((x - rect.x) / rect.w) * 2 * r));
    const v = Math.max(-r, Math.min(r, r - ((y - rect.y) / rect.h) * 2 * r));
    let hit: PointRef | null = null;
    let best = 7 * 7;
    for (const m of this.markers) {
      if (m.alpha < 0.3) continue;
      const dx = rect.x + ((m.u + r) / (2 * r)) * rect.w - x;
      const dy = rect.y + ((r - m.v) / (2 * r)) * rect.h - y;
      const d2 = dx * dx + dy * dy;
      if (d2 <= best) {
        best = d2;
        hit = { split: m.split, i: m.i };
      }
    }
    return { u, v, coords: this.host.coordsAt(u, v), hit };
  }

  private hover(e: MouseEvent): void {
    const d = store.data;
    const at = this.at(e);
    if (!at || !d) {
      hideTip();
      return;
    }
    const coords = at.hit ? (pointCoords(d, at.hit) ?? at.coords) : at.coords;
    let text = probsText(d, coords, this.host.probsAt(coords), at.hit);
    text += boundaryOpts.adding || e.shiftKey ? `\nClick to add a ${d.info.classes[boundaryOpts.addClass] ?? ''} point here` : at.hit ? '\nClick to use this point as the input' : '\nClick to use this spot as the input';
    showTip(text, e.clientX, e.clientY);
    this.canvas.style.cursor = boundaryOpts.adding || e.shiftKey ? 'copy' : 'crosshair';
  }

  /** Arrow keys move the current input across the plane; Enter adds it as a point in "Add points" mode. */
  private key(e: KeyboardEvent): void {
    const field = this.host.field();
    if (!field || !store.data) return;
    const r = field.r;
    const step = e.shiftKey ? 0.25 : 0.05;
    const pr = this.host.probe();
    let u = pr?.u ?? 0;
    let v = pr?.v ?? 0;
    if (e.key === 'ArrowLeft') u -= step;
    else if (e.key === 'ArrowRight') u += step;
    else if (e.key === 'ArrowUp') v += step;
    else if (e.key === 'ArrowDown') v -= step;
    else if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      pickAt(this.host.coordsAt(u, v), null, boundaryOpts.adding);
      return;
    } else return;
    e.preventDefault();
    u = Math.max(-r, Math.min(r, Math.round(u / 0.05) * 0.05));
    v = Math.max(-r, Math.min(r, Math.round(v / 0.05) * 0.05));
    pickAt(this.host.coordsAt(u, v), null, false);
  }
}

/** A plain-language note for a map drawn on a coarser grid than usual, to keep the page responsive. */
export function coarseNote(res: number, what: string, dims: 2 | 3 = 2): string {
  const grid = dims === 3 ? `${res} × ${res} × ${res}` : `${res} × ${res}`;
  return `This network is large, so ${what} is computed on a coarse ${grid} grid${store.running ? ' while it trains' : ''}, a few frames at a time, to keep the page responsive.`;
}

// ── The 2-D view ─────────────────────────────────────────────────────────

/** Markers for every point of a 2-D dataset (and your added points). */
function markers2d(d: Data): Marker[] {
  const p = d.points!;
  const out: Marker[] = [];
  const add = (split: 'train' | 'test', c: Float32Array, y: Uint8Array) => {
    for (let i = 0; i < y.length; i++) out.push({ split, i, u: c[2 * i], v: c[2 * i + 1], cls: y[i], alpha: 1 });
  };
  if (boundaryOpts.showTest) add('test', p.testCoords, d.testY);
  add('train', p.trainCoords, d.trainY);
  for (const c of store.custom) if (c.origin === 'point' && c.coords?.length === 2) out.push({ split: 'custom', i: c.id, u: c.coords[0], v: c.coords[1], cls: c.y, alpha: 1 });
  return out;
}

export interface View {
  el: HTMLElement;
  setActive(on: boolean): void;
}

/**
 * Grid budgets of the 2-D plot. While training: a pass of about 8 ms. Paused: up to 200² (thin
 * slivers of a region stay visible), sliced across frames. A slow network gets a coarser grid
 * rather than a pass of more than the cap.
 */
export const BUDGET_2D: { running: Budget; paused: Budget } = {
  running: { budget: 8, min: 16, max: 100, cap: 40, floor: 8 },
  paused: { budget: 120, min: 32, max: 200, cap: 400, floor: 12 },
};
/** Below this many cells per side, the plot says it is coarse. */
const COARSE_2D = 32;

export function create2D(): View {
  const ev = new Evaluator();
  const cost = new Cost();
  let field: PlaneField | null = null;
  let job: PlaneJob | null = null;
  /** Main-thread time of the pass under way, all frames together. */
  let passMs = 0;
  const host: PlaneHost = {
    field: () => field,
    markers: () => (store.data?.points ? markers2d(store.data) : []),
    probe: () => {
      const c = store.probe?.coords;
      return c && c.length === 2 ? { u: c[0], v: c[1], on: true } : null;
    },
    coordsAt: (u, v) => new Float32Array([u, v]),
    probsAt: (coords) => {
      // A new network (architecture or features) replaces any pass made with the old one.
      if (netFits() && !ev.current && job) {
        job = null;
        live.restart();
      }
      return ev.probsAt(coords, 2);
    },
  };
  const plane = new PlaneCanvas(host, false, 'Decision boundary: the predicted class over the plane, with the training and test points');
  const box = h('div', { class: 'bd-plot' }, plane.canvas);
  const note = h('p', { class: 'hint bd-res-note', hidden: true });
  const el = h('div', { class: 'bd-2d' }, box, note, plane.keysHint);

  const start = (d: Data) => {
    ev.sync();
    const res = cost.res(store.running ? BUDGET_2D.running : BUDGET_2D.paused, 2);
    job = new PlaneJob(ev.ev, 2, [0, 1], [], pointDomain(d), res);
  };

  const render = (evaluate: boolean, draw: boolean): boolean => {
    const d = store.data;
    if (!d?.points || d.points.dims !== 2 || !netFits()) {
      job = null;
      return false;
    }
    const t0 = performance.now();
    if (job && !ev.current) job = null; // made for a network that is gone
    if ((evaluate || (!field && !job)) && !job) {
      start(d);
      passMs = 0;
    }
    let fresh = false;
    const inPass = !!job;
    if (job) {
      const f = job.step(sliceMs(), cost.perPoint);
      if (f) {
        cost.add(job.busy, job.res * job.res);
        field = f;
        fresh = true;
        job = null;
        plane.canvas.dataset.weights = String(store.weightsRev);
        plane.canvas.dataset.ms = (passMs + performance.now() - t0).toFixed(1);
      }
    }
    if (draw || fresh) {
      const w = Math.min(560, box.clientWidth || 560);
      plane.draw(w);
      if (field) {
        plane.canvas.setAttribute(
          'aria-label',
          `Decision boundary of ${d.info.name}: the predicted class over the plane from −${fixed(field.r, 2)} to ${fixed(field.r, 2)} on both axes, shaded by confidence, with ${d.trainY.length} training points${boundaryOpts.showTest ? ` and ${d.testY.length} test points` : ''}, each class with its own shape.`,
        );
        note.hidden = field.res >= COARSE_2D;
        if (!note.hidden) note.textContent = coarseNote(field.res, 'the map');
      }
      live.noteDraw(t0, performance.now() - t0);
    }
    // Every millisecond of a pass counts against the budget: evaluation, painting, drawing, and
    // the browser rasterising the drawing afterwards.
    if (inPass) passMs += performance.now() - t0;
    if (fresh) live.gap = Math.max(100, 4 * (passMs + live.drawCost));
    return job !== null;
  };
  const live = new Live(el, render);
  live.active = false;
  const again = () => live.request(true);
  const redraw = () => live.request(false);
  store.on('weights', again);
  store.on('model', () => {
    job = null;
    live.restart();
  });
  store.on('data', () => {
    field = null;
    job = null;
    live.restart();
  });
  let wasRunning = false;
  store.on('status', () => {
    // Back to full resolution once training pauses.
    if (wasRunning && !store.running) again();
    wasRunning = store.running;
  });
  for (const e of ['probe', 'custom'] as const) store.on(e, redraw);
  onBoundaryOpts(redraw);
  onThemeChange(redraw);
  new ResizeObserver(redraw).observe(box);
  return {
    el,
    setActive(on: boolean) {
      live.active = on;
      if (on) again();
    },
  };
}
