import './unitsView.css';
import './units.css';
import { select, setProbe, testProbe } from '../actions';
import { analysis, isSuperseded } from '../analysis/client';
import { centreFieldSize, centrePosition, cropBox, type Box } from '../analysis/receptive';
import { fixed } from '../analysis/stats';
import {
  ACTMAX_STEPS,
  TOP_LABELS,
  allOff,
  coveragePhrase,
  isDead,
  labelSummary,
  mapScale,
  rankIn,
  rankPhrase,
  sharePct,
  synthBase,
  unitColumn,
  unitResponse,
  type ActmaxPartial,
  type ActmaxResult,
  type Hit,
  type TopkResult,
  type UnitKind,
  type UnitSummary,
} from '../analysis/units';
import { noun, sampleInput, type Data } from '../data/datasets';
import { gridCoords, pointDomain, PointEvaluator } from '../data/grid';
import { Network } from '../nn/network';
import type { Act, Shape } from '../nn/types';
import { store, type Probe } from '../store';
import { dotTrain, pickAt, probeCross } from './boundary2d';
import { onSlice, slabWidth, slice, sliceAxes } from './boundary3d';
import { axisName } from './boundaryMath';
import { layerDetail, layerName } from './builder';
import { $, clear, h, int, selectField } from './dom';
import { fitCanvas } from './draw';
import { isCurrent, stampNow, syncedSection, type Stamp } from './snapshot';
import { classColor, css, diverging, onThemeChange, palette, sequential, type RGB } from './theme';
import { hideTip, showTip } from './tip';

/**
 * Section 09: what each unit responds to. For image datasets: the test images that excite each
 * filter or unit most and least, and an input synthesised to excite it. For point datasets: the
 * test points that excite each unit most, and its response over the input plane (or a slice of
 * the cube), which replaces synthesis since the whole input space fits on one map.
 */

const CARD_LIMIT = 32;
const TOP_K = 16;
const CARD_TOP = 9;
const DETAIL_BOTTOM = 8;
/** Grid cells per side of a point dataset's response maps. */
const MAP_RES = 48;
const MONO = '"IBM Plex Mono", ui-monospace, monospace';

interface SynthUnit {
  x: Float32Array;
  step: number;
  value: number;
  start: number | null;
  final: number | null;
}

interface SynthState {
  block: number;
  /** The weights the synthesis used. */
  stamp: Stamp;
  running: boolean;
  stopped: boolean;
  total: number;
  /** The unit being optimised now. */
  current: number | null;
  units: Map<number, SynthUnit>;
  error: string | null;
}

/** A top-k scan of one layer, with a copy of the network holding the weights it used. */
interface Scan {
  result: TopkResult;
  /** The current input is measured with these weights too, so it ranks against the same scan. */
  net: Network;
  stamp: Stamp;
  /** The data the scan ran on (its indices point into data.testX). */
  data: Data;
  /** Point datasets: the input features the network was built for. */
  features: string[];
}

/** A point dataset's response maps for one scan: every unit of the scanned layer over one plane. */
interface Maps {
  scan: Scan;
  key: string;
  plane: Plane;
  /** Per unit: MAP_RES² responses, row 0 at the top. */
  cols: Float32Array[];
  scales: { signed: boolean; max: number; lo: number; hi: number }[];
}

/** The plane a point dataset is mapped over: the data plane in 2-D, the slice from 03 in 3-D. */
interface Plane {
  dims: 2 | 3;
  r: number;
  /** Data axes along the map's x (left to right) and y (bottom to top). */
  axes: [number, number];
  /** Coordinates of the axis off the map (3-D). */
  fixedAt: number[];
}

const rgb: RGB = [0, 0, 0];
const clamp255 = (v: number) => Math.max(0, Math.min(255, Math.round(v * 255)));

/**
 * Paints an input image (channel-major, values in [0, 1]) at one canvas pixel per input pixel; CSS
 * scales the canvas up. Grey images use the theme's ink-on-surface map, colour images their own
 * colours. NaN marks a pixel outside the image (a crop at the border), painted as the surface.
 */
function paintImage(c: HTMLCanvasElement, data: ArrayLike<number>, s: Shape): void {
  const { h: H, w: W } = s;
  if (c.width !== W) c.width = W;
  if (c.height !== H) c.height = H;
  const ctx = c.getContext('2d');
  if (!ctx) return;
  const img = ctx.createImageData(W, H);
  const d = img.data;
  const HW = H * W;
  const surface = palette().rgb.surface;
  for (let i = 0; i < HW; i++) {
    const v = data[i];
    if (Number.isNaN(v)) {
      d[4 * i] = surface[0];
      d[4 * i + 1] = surface[1];
      d[4 * i + 2] = surface[2];
    } else if (s.c === 3) {
      d[4 * i] = clamp255(v);
      d[4 * i + 1] = clamp255(data[HW + i]);
      d[4 * i + 2] = clamp255(data[2 * HW + i]);
    } else {
      sequential(v, rgb);
      d[4 * i] = rgb[0];
      d[4 * i + 1] = rgb[1];
      d[4 * i + 2] = rgb[2];
    }
    d[4 * i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

/** Every channel of an image (values via `at`, channel-major) inside box `b`; pixels beyond the image are NaN. */
function cropImage(at: (i: number) => number, b: Box, s: Shape): { data: Float32Array; shape: Shape } {
  const H = b.y1 - b.y0 + 1;
  const W = b.x1 - b.x0 + 1;
  const data = new Float32Array(s.c * H * W).fill(NaN);
  for (let ch = 0; ch < s.c; ch++) {
    for (let r = 0; r < H; r++) {
      const y = b.y0 + r;
      if (y < 0 || y >= s.h) continue;
      for (let c = 0; c < W; c++) {
        const x = b.x0 + c;
        if (x >= 0 && x < s.w) data[ch * H * W + r * W + c] = at(ch * s.h * s.w + y * s.w + x);
      }
    }
  }
  return { data, shape: { c: s.c, h: H, w: W } };
}

/** Paints one unit's response map (res² values, row 0 at the top) at one pixel per cell. */
function paintMap(c: HTMLCanvasElement, values: ArrayLike<number>, res: number, signed: boolean, max: number): void {
  if (c.width !== res) c.width = res;
  if (c.height !== res) c.height = res;
  const ctx = c.getContext('2d');
  if (!ctx) return;
  const img = ctx.createImageData(res, res);
  const d = img.data;
  const inv = max > 0 ? 1 / max : 0;
  for (let i = 0; i < res * res; i++) {
    if (signed) diverging(values[i] * inv, rgb);
    else sequential(values[i] * inv, rgb);
    d[4 * i] = rgb[0];
    d[4 * i + 1] = rgb[1];
    d[4 * i + 2] = rgb[2];
    d[4 * i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

/** Point datasets' plotting domain, measured once per data object. */
const domains = new WeakMap<Data, number>();
const domainOf = (d: Data) => {
  let r = domains.get(d);
  if (r === undefined) domains.set(d, (r = pointDomain(d)));
  return r;
};

const planeOf = (d: Data): Plane => {
  const dims = d.points!.dims;
  const fixedAt: number[] = [];
  if (dims === 3) fixedAt[slice.axis] = slice.pos;
  return { dims, r: domainOf(d), axes: dims === 2 ? [0, 1] : sliceAxes(slice.axis), fixedAt };
};

/** "the input plane" or "the slice x₃ = 0.00". */
const planeText = (p: Plane) => (p.dims === 2 ? 'the input plane' : `the slice ${axisName(slice.axis)} = ${fixed(p.fixedAt[slice.axis] ?? 0, 2)}`);

/** Activations whose values can be negative get a diverging map (red positive, blue negative). */
const signedAct = (kind: UnitKind, act: Act) => kind === 'output' || act === 'tanh' || act === 'linear' || act === 'leaky';

/** What one unit of a layer is called: filters in conv layers, outputs in the output layer, units otherwise. */
const nounOf = (kind: UnitKind) => (kind === 'conv' ? 'filter' : kind === 'output' ? 'output' : 'unit');

const cap = (t: string) => t.charAt(0).toUpperCase() + t.slice(1);
const lowerFirst = (t: string) => (/^[A-Z][A-Z]/.test(t) ? t : t.charAt(0).toLowerCase() + t.slice(1));
const article = (w: string) => (/^[aeiou]/i.test(w) ? 'an' : 'a');

/**
 * `d` decimals with a true minus sign. Values too small for `d` decimals keep two significant
 * digits ("0.00022") instead of rounding to a "0.00" or "−0.00" that hides them.
 */
const num = (v: number, d = 3): string => {
  if (!Number.isFinite(v)) return '—';
  const t = v !== 0 && Math.abs(v) < 10 ** -d ? v.toPrecision(2) : (v === 0 ? 0 : v).toFixed(d);
  return t.replace(/^-/, '−').replace('e-', 'e−');
};

/** Two decimals, without a "−0.00". */
const fix2 = (v: number) => (Math.abs(v) < 0.005 ? 0 : v).toFixed(2).replace(/^-/, '−');

/** Thin progress bar for synthesis. It keeps its space while idle, so nothing below it jumps. */
function progressBar(label: string) {
  const fill = h('span', { style: { width: '0%' } });
  const el = h('div', { class: 'progress is-idle', role: 'progressbar', 'aria-label': label, 'aria-valuemin': '0', 'aria-valuemax': '100', 'aria-valuenow': '0' }, fill);
  return {
    el,
    set(f: number) {
      const v = Math.max(0, Math.min(1, f));
      fill.style.width = `${(v * 100).toFixed(1)}%`;
      el.setAttribute('aria-valuenow', String(Math.round(v * 100)));
    },
    show(on: boolean) {
      el.classList.toggle('is-idle', !on);
      el.setAttribute('aria-hidden', String(!on));
    },
  };
}

export function mountUnits(): void {
  const root = $('units-root');
  root.classList.add('units');

  // ── State ──
  let layer = 0;
  /** The scan on screen; it always belongs to `layer`. */
  let scan: Scan | null = null;
  /** The scan being computed, if any, and the test set it runs on. */
  let scanning: { block: number; testX: ArrayLike<number> } | null = null;
  let scanError: string | null = null;
  let showAll = false;
  let visible = false;
  const synths = new Map<number, SynthState>();
  let maps: Maps | null = null;
  const evaluator = new PointEvaluator();

  // ── Elements ──
  const layerSlot = h('div', { class: 'units-layer' });
  const synthBtn = h('button', { type: 'button', id: 'units-synth', class: 'btn btn-sm btn-solid' }, 'Synthesise inputs') as HTMLButtonElement;
  const synthNote = h('span', { class: 'hint units-synth-note', 'aria-live': 'polite' });
  const synthBar = progressBar('Synthesis progress');
  const grid = h('div', { class: 'units-grid' });
  const more = h('button', { type: 'button', id: 'units-show-all', class: 'btn btn-sm' }) as HTMLButtonElement;
  const gridNote = h('p', { class: 'hint units-grid-note' });
  const gridKey = h('p', { class: 'hint units-key' });
  const detail = h('div', { class: 'units-detail', id: 'units-detail', role: 'region', 'aria-label': 'Unit details' });
  const main = h('div', { class: 'units-main' }, gridKey, grid, more);
  const layout = h('div', { class: 'units-layout' }, main, detail);

  // ── Small helpers ──
  const blocks = () => store.net.blocks;
  const clampLayer = (i: number) => Math.max(0, Math.min(blocks().length - 1, i));
  const kindOf = (i: number): UnitKind => (i === blocks().length - 1 ? 'output' : blocks()[i].kind);
  const countOf = (i: number) => {
    const b = blocks()[i];
    return b.kind === 'conv' ? b.spec.filters : b.spec.units;
  };
  const specOf = (i: number) => (i === blocks().length - 1 ? null : store.net.spec[i]);
  const actOf = (i: number): Act => blocks()[i].spec.act;
  const summary = (u: number): UnitSummary | null => (scan && scan.result.block === layer ? scan.result.units[u] ?? null : null);
  const selectedUnit = (): number | null => {
    if (store.selected !== layer || store.selectedUnit === null) return null;
    return store.selectedUnit < countOf(layer) ? store.selectedUnit : null;
  };
  const synth = () => synths.get(layer) ?? null;
  /** Point datasets get maps and point lists instead of pictures and synthesis. */
  const pts = () => store.info.kind === 'points';
  const isMnist = () => store.info.id === 'mnist';
  /** What one sample is called: "digit", "image" or "point". */
  const one = () => noun(store.info);
  const many = () => noun(store.info, 2);
  /** The network's input shape (an image for image datasets). */
  const shape = (): Shape => store.net.arch.input;
  const className = (k: number) => store.info.classes[k] ?? `Class ${k}`;
  /** "Filter 3", "Unit 12", "Digit 7" (MNIST outputs), "Airplane", "Class 1". */
  const unitName = (kind: UnitKind, u: number) => (kind === 'output' ? (isMnist() ? `Digit ${u}` : cap(className(u))) : kind === 'conv' ? `Filter ${u + 1}` : `Unit ${u + 1}`);
  /** The layer in one line: "Conv 2 · 16 × 3×3 · ReLU · pool", "Output · 10 logits". */
  const layerLabel = (i: number) => {
    const spec = specOf(i);
    return spec ? `${layerName(spec, i)} · ${layerDetail(spec)}` : `Output · ${countOf(i)} logits`;
  };
  /** Pixel i (channel-major) of test sample `index` as network input. */
  const sampleAt = (d: Data, index: number) => {
    const n = d.inputSize;
    return (p: number) => d.testX[index * n + p] * d.scale;
  };
  /** What a conv layer's units see, for the copy: the clipped field at the centre of the map. */
  const field = () => {
    const arch = store.net.arch;
    const side = arch.input.h;
    const size = centreFieldSize(arch, layer);
    const c = centrePosition(arch, layer);
    const crop = cropBox(arch, layer, c.y, c.x);
    const cropSide = crop ? crop.y1 - crop.y0 + 1 : side;
    return { size, whole: size >= side, cropSide, crop, side };
  };
  /** The blank image synthesis starts from, in words. */
  const blankWords = () => (synthBase(shape().c) > 0 ? 'plain grey' : 'blank');
  /** True when the loaded data and the network fit together (a dataset switch updates one, then the other). */
  const ready = () => !!store.data && store.valid && store.data.inputSize === store.net.inputSize && store.data.info.id === store.dataset;

  /** Dims the previous scan while a new one for the same layer is computed. */
  const markUpdating = () => layout.classList.toggle('is-updating', !!scanning && !!scan && scanning.block === layer);

  // ── Synced top-k scan ──
  const refresh = () => {
    if (!ready()) return;
    const block = layer;
    const stamp = sync.begin();
    const data = store.data!;
    const features = store.features.slice();
    // The worker gets these same weights (analysis.run copies them synchronously below).
    const net = new Network(store.net.arch, 0);
    net.setWeights(store.net.getWeights());
    scanning = { block, testX: data.testX };
    scanError = null;
    sync.setProgress(0);
    markUpdating();
    if (!scan) {
      renderCards();
      renderDetail();
    } else renderNote();
    analysis
      .run<TopkResult>('units-topk', 'topk', { block, k: TOP_K }, (p) => sync.setProgress(p.done / p.total))
      .then((r) => {
        scanning = null;
        if (r.block !== layer || stamp.version !== store.version || !store.valid || store.data?.testX !== data.testX) {
          sync.fail();
          markUpdating();
          return;
        }
        scan = { result: r, net, stamp, data, features };
        markUpdating();
        renderCards();
        renderDetail();
        renderSynthUI();
        sync.done(stamp);
      })
      .catch((e: unknown) => {
        // A newer scan replaced this one; it owns the status now.
        if (isSuperseded(e)) return;
        scanning = null;
        scanError = e instanceof Error ? e.message : String(e);
        sync.fail();
        markUpdating();
        renderNote();
      });
  };
  const sync = syncedSection(root, refresh);

  /**
   * Scans the current layer when the section is on screen and nothing is shown for this layer
   * yet (first view, another layer picked, a new architecture). Results that merely went stale
   * are left to the shared policy in syncedSection.
   */
  const ensureScan = () => {
    if (!visible || !ready()) return;
    if (scan || scanning?.block === layer) return;
    if (!sync.shown) sync.request();
    else sync.refreshNow(); // the result on record belongs to another layer or another test set
  };
  let planeDirty = false;
  new IntersectionObserver(
    (entries) => {
      visible = entries.some((e) => e.isIntersecting);
      ensureScan();
      if (visible && planeDirty) replane();
      else if (visible && probeDirty) redrawProbe();
    },
    { rootMargin: '200px 0px' },
  ).observe(root);

  // ── Response maps (point datasets) ──
  /** The maps for the scan on screen over the current plane, computed on demand (a few ms). */
  const mapsNow = (): Maps | null => {
    const d = store.data;
    if (!pts() || !scan || !d?.points || scan.data.testX !== d.testX) return null;
    const plane = planeOf(d);
    const key = `${plane.axes.join()}|${plane.fixedAt.join()}|${plane.r}`;
    if (maps && maps.scan === scan && maps.key === key) return maps;
    try {
      evaluator.sync(scan.net);
      const coords = gridCoords(plane.dims, MAP_RES, plane.r, plane.axes, plane.fixedAt);
      const { acts } = evaluator.evaluate(coords, plane.dims, scan.features, { activations: true });
      const block = scan.result.block;
      const U = scan.result.units.length;
      const A = acts![block];
      const cols = Array.from({ length: U }, (_, j) => unitColumn(A, U, j));
      const signed = signedAct(scan.result.kind, scan.net.blocks[block].spec.act);
      const scales = cols.map((c) => ({ ...mapScale(c), signed }));
      maps = { scan, key, plane, cols, scales };
      return maps;
    } catch {
      return null;
    }
  };

  /** Data coordinates of a test point on the map's two axes. */
  const pointUV = (d: Data, i: number, plane: Plane): [number, number] => {
    const p = d.points!;
    return [p.testCoords[i * p.dims + plane.axes[0]], p.testCoords[i * p.dims + plane.axes[1]]];
  };

  /** Mini scatter for a card: every test point faintly, the strongest ones in their class colours. */
  const drawScatter = (c: HTMLCanvasElement, hits: Hit[]) => {
    const d = store.data;
    if (!d?.points) return;
    const S = 96;
    const ctx = fitCanvas(c, S, S);
    c.style.width = '';
    c.style.height = '';
    const p = palette();
    const plane = planeOf(d);
    const r = plane.r;
    const tx = (u: number) => ((u + r) / (2 * r)) * S;
    const ty = (v: number) => ((r - v) / (2 * r)) * S;
    ctx.fillStyle = p.surface;
    ctx.fillRect(0, 0, S, S);
    // Axes through the origin, as a faint cross.
    ctx.fillStyle = p.hair;
    ctx.fillRect(Math.round(tx(0)), 0, 1, S);
    ctx.fillRect(0, Math.round(ty(0)), S, 1);
    ctx.fillStyle = p.muted;
    ctx.globalAlpha = 0.45;
    const n = d.testY.length;
    for (let i = 0; i < n; i++) {
      const [u, v] = pointUV(d, i, plane);
      ctx.fillRect(tx(u) - 0.9, ty(v) - 0.9, 1.8, 1.8);
    }
    ctx.globalAlpha = 1;
    for (const hit of [...hits].reverse()) {
      const [u, v] = pointUV(d, hit.index, plane);
      dotTrain(ctx, tx(u), ty(v), 3.6, classColor(d.testY[hit.index]), p.ink);
    }
  };

  // ── Layer select ──
  const buildLayerSelect = () => {
    clear(layerSlot);
    layerSlot.append(
      selectField(
        'units-layer',
        'Layer',
        blocks().map((_, i) => ({ value: i, label: layerLabel(i) })),
        layer,
        (v) => select(v, null),
      ),
    );
  };

  const setLayer = (b: number) => {
    if (b === layer) return;
    const s = synth();
    if (s?.running) {
      s.running = false;
      s.stopped = true;
      analysis.cancel('units-actmax');
    }
    layer = b;
    showAll = (selectedUnit() ?? 0) >= CARD_LIMIT;
    if (scan && scan.result.block !== layer) scan = null;
    markUpdating();
    const sel = document.getElementById('units-layer') as HTMLSelectElement | null;
    if (sel) sel.value = String(layer);
    renderAll();
    ensureScan();
  };

  // ── Cards ──
  interface CardRefs {
    btn: HTMLButtonElement;
    synth: HTMLCanvasElement;
    empty: HTMLElement;
  }
  const cards = new Map<number, CardRefs>();

  const hitCanvas = (hit: Hit): HTMLCanvasElement => {
    const c = h('canvas', { class: 'units-px', 'aria-hidden': 'true' }) as HTMLCanvasElement;
    const d = store.data!;
    const s = shape();
    const b = hit.y >= 0 ? cropBox(store.net.arch, layer, hit.y, hit.x) : null;
    if (b) {
      const { data, shape: cs } = cropImage(sampleAt(d, hit.index), b, s);
      paintImage(c, data, cs);
    } else {
      paintImage(c, sampleInput(d, 'test', hit.index), s);
    }
    return c;
  };

  const paintSynth = (c: HTMLCanvasElement, x: Float32Array) => {
    const s = shape();
    const b = kindOf(layer) === 'conv' ? field().crop : null;
    if (b) {
      const { data, shape: cs } = cropImage((p) => x[p], b, s);
      paintImage(c, data, cs);
    } else paintImage(c, x, s);
  };

  /** "fires on 37% of points", or "never fires (dead)" for a unit with one response for every sample. */
  const coverText = (kind: UnitKind, s: UnitSummary) => (isDead(kind, s) ? 'never fires (dead)' : coveragePhrase(kind, s.coverage, 0, one()));

  const card = (u: number): HTMLElement => {
    const kind = kindOf(layer);
    const s = summary(u);
    const name = unitName(kind, u);
    const ok = !!s && !!store.data;
    let left: HTMLElement;
    let right: HTMLElement;
    const synthC = h('canvas', { class: 'units-px', 'aria-hidden': 'true' }) as HTMLCanvasElement;
    const empty = h('span', { class: 'units-synth-empty' }, 'Not yet');
    if (pts()) {
      const scatter = h('canvas', { class: 'units-scatter', 'aria-hidden': 'true' }) as HTMLCanvasElement;
      const map = h('canvas', { class: 'units-map', 'aria-hidden': 'true' }) as HTMLCanvasElement;
      const m = ok ? mapsNow() : null;
      if (ok) drawScatter(scatter, s!.top.slice(0, CARD_TOP));
      if (m) {
        const sc = m.scales[u];
        paintMap(map, m.cols[u], MAP_RES, sc.signed, sc.max);
        empty.hidden = true;
      } else {
        map.hidden = true;
        empty.textContent = scanning ? 'Scanning…' : 'Not yet';
      }
      left = h('span', { class: 'units-fig' }, ok ? scatter : h('span', { class: 'units-scatter is-blank' }));
      right = h('span', { class: 'units-fig' }, h('span', { class: 'units-synth' }, map, empty));
    } else {
      const mosaic = h('span', { class: 'units-mosaic' });
      if (ok) for (const hit of s!.top.slice(0, CARD_TOP)) mosaic.append(hitCanvas(hit));
      else for (let i = 0; i < CARD_TOP; i++) mosaic.append(h('span', { class: 'units-px is-blank' }));
      const su = synth()?.units.get(u);
      if (su) {
        paintSynth(synthC, su.x);
        empty.hidden = true;
      } else synthC.hidden = true;
      left = h('span', { class: 'units-fig' }, mosaic);
      right = h('span', { class: 'units-fig' }, h('span', { class: 'units-synth' }, synthC, empty));
    }
    const sel = selectedUnit() === u;
    const cover = s ? coverText(kind, s) : '';
    const dead = !!s && isDead(kind, s);
    const label = s
      ? `${name}: ${dead ? cover : kind === 'conv' ? `${cover} on average` : cover}, mean response ${fix2(s.mean)}. Show details.`
      : `${name}. Show details.`;
    const btn = h(
      'button',
      { type: 'button', class: `units-card${dead ? ' is-dead' : ''}`, 'aria-pressed': String(sel), 'aria-label': label, 'data-unit': String(u) },
      h('span', { class: 'units-card-head' }, h('span', { class: 'units-card-title' }, name), h('span', { class: 'units-card-mean' }, s ? `mean ${fix2(s.mean)}` : '')),
      h('span', { class: 'units-card-body' }, left, right),
      h('span', { class: 'units-card-foot' }, s ? cover : scanning ? 'Scanning…' : 'Not scanned yet'),
    ) as HTMLButtonElement;
    btn.addEventListener('click', () => {
      select(layer, u);
      revealDetail();
    });
    cards.set(u, { btn, synth: synthC, empty });
    return btn;
  };

  /** The line above the cards that says what each card shows. */
  const keyText = () => {
    const kind = kindOf(layer);
    const what = nounOf(kind);
    let t: string;
    if (pts()) {
      const d = store.data;
      const plane = d?.points ? planeOf(d) : null;
      const scale = signedAct(kind, actOf(layer)) ? ' Red is positive, blue negative.' : ' Darker is stronger.';
      t = `Each card: the 9 test points that excite the ${what} most, in their class colours among the rest (left), and its response over ${plane ? planeText(plane) : 'the input plane'} (right).${scale}`;
    } else {
      const f = kind === 'conv' ? field() : null;
      const patch = f && f.cropSide < f.side ? `the ${f.cropSide}×${f.cropSide} patches of ` : '';
      t = `Each card: ${patch}the 9 test ${many()} that excite the ${what} most (left), and an input synthesised from ${blankWords()} to excite it (right).`;
    }
    const res = scan && scan.result.block === layer ? scan.result : null;
    const dead = res ? res.units.filter((s) => isDead(kind, s)).length : 0;
    if (dead) t += ` ${dead === 1 ? `One ${what} never fires` : `${dead} ${what}s never fire`} (dead); ${dead === 1 ? 'its card shows' : 'their cards show'} the ${many()} closest to firing.`;
    return t;
  };

  const renderCards = () => {
    clear(grid);
    cards.clear();
    const n = countOf(layer);
    const shown = showAll ? n : Math.min(n, CARD_LIMIT);
    for (let u = 0; u < shown; u++) grid.append(card(u));
    gridKey.textContent = keyText();
    more.hidden = n <= CARD_LIMIT;
    more.textContent = showAll ? `Show the first ${CARD_LIMIT}` : `Show all ${n}`;
    renderNote();
  };

  const renderNote = () => {
    if (!store.valid) gridNote.textContent = 'Fix the architecture in 01 to see what its units respond to.';
    else if (scanError) gridNote.textContent = `The scan failed: ${scanError}`;
    else if (!store.data) gridNote.textContent = `Waiting for ${store.info.name} to load…`;
    else gridNote.textContent = '';
    gridNote.hidden = !gridNote.textContent;
    layout.hidden = !store.valid;
  };

  const syncPressed = () => {
    const sel = selectedUnit();
    for (const [u, c] of cards) c.btn.setAttribute('aria-pressed', String(u === sel));
  };

  /** On a stacked layout the panel sits below the grid; bring it into view after a click. */
  const revealDetail = () => {
    const r = detail.getBoundingClientRect();
    const bar = document.getElementById('bar')?.getBoundingClientRect().bottom ?? 0;
    if (r.top >= bar && r.top <= window.innerHeight * 0.7) return;
    window.scrollTo({ top: window.scrollY + r.top - bar - 12, behavior: 'smooth' });
  };

  // ── Detail panel ──
  /** Canvas redraws of the detail panel (histogram, response map), run on resize, theme and input changes. */
  let detailDraws: (() => void)[] = [];
  let probeLine: HTMLElement | null = null;
  /** Re-renders the probe line's text only (no forward pass); set while a histogram is shown. */
  let probeText: (() => void) | null = null;
  let probeDirty = false;
  let detailSynth: { unit: number; canvas: HTMLCanvasElement; text: HTMLElement; empty: HTMLElement } | null = null;

  const digitButton = (hit: Hit, showBox: boolean): HTMLElement => {
    const d = store.data!;
    const i = hit.index;
    const y = d.testY[i];
    const s = shape();
    const label = isMnist() ? String(y) : className(y);
    const c = h('canvas', { class: 'units-px', 'aria-hidden': 'true' }) as HTMLCanvasElement;
    paintImage(c, sampleInput(d, 'test', i), s);
    const box = showBox && hit.box
      ? h('span', {
          class: 'units-digit-box',
          style: {
            left: `${(hit.box.x0 / s.w) * 100}%`,
            top: `${(hit.box.y0 / s.h) * 100}%`,
            width: `${((hit.box.x1 - hit.box.x0 + 1) / s.w) * 100}%`,
            height: `${((hit.box.y1 - hit.box.y0 + 1) / s.h) * 100}%`,
          },
        })
      : null;
    const what = isMnist() ? `label ${y}` : className(y);
    const btn = h(
      'button',
      {
        type: 'button',
        class: 'units-digit',
        'data-index': String(i),
        'aria-pressed': String(store.probe?.key === `test:${i}`),
        title: `Test ${one()} #${i} · ${what} · response ${num(hit.value, 3)}`,
        'aria-label': `Use test ${one()} ${i}, ${isMnist() ? `a ${y}` : lowerFirst(label)}, with response ${num(hit.value, 3)}, as the network input`,
      },
      c,
      box,
      h('span', { class: 'units-digit-label' }, isMnist() ? String(y) : store.info.glyphs[y] ?? String(y)),
    );
    btn.addEventListener('click', () => setProbe(testProbe(d, i)));
    return btn;
  };

  /** A test point as a row: class, index, coordinates and the unit's response. Click to make it the input. */
  const pointButton = (hit: Hit): HTMLElement => {
    const d = store.data!;
    const p = d.points!;
    const i = hit.index;
    const y = d.testY[i];
    const coords = Array.from(p.testCoords.subarray(i * p.dims, (i + 1) * p.dims), (v) => fixed(v, 2));
    const btn = h(
      'button',
      {
        type: 'button',
        class: 'units-point',
        'data-index': String(i),
        'aria-pressed': String(store.probe?.key === `test:${i}`),
        title: `Test point #${i} · ${className(y)} · (${coords.join(', ')}) · response ${num(hit.value, 3)}`,
        'aria-label': `Use test point ${i}, ${lowerFirst(className(y))} at ${coords.join(', ')}, with response ${num(hit.value, 3)}, as the network input`,
      },
      h('i', { class: 'units-point-dot', style: { background: classColor(y) } }),
      h('span', { class: 'units-point-cls' }, store.info.glyphs[y] ?? String(y)),
      h('span', { class: 'units-point-id' }, `#${i}`),
      h('span', { class: 'units-point-xy' }, `(${coords.join(', ')})`),
      h('span', { class: 'units-point-v' }, num(hit.value, 3)),
    );
    btn.addEventListener('click', () => setProbe(testProbe(d, i)));
    return btn;
  };

  /**
   * The current input's response, measured with the scan's own weights (not the live ones), so
   * it ranks against the same histogram. Cached per input, scan and unit.
   */
  let probeCache: { probe: Probe; scan: Scan; unit: number; value: number } | null = null;
  const probeResponse = (u: number): number | null => {
    const p = store.probe;
    if (!p || !scan || scan.result.block !== layer || p.x.length !== scan.net.inputSize) return null;
    if (probeCache && probeCache.probe === p && probeCache.scan === scan && probeCache.unit === u) return probeCache.value;
    const value = unitResponse(scan.net, layer, u, p.x).value;
    probeCache = { probe: p, scan, unit: u, value };
    return value;
  };

  /** "Current input, test digit #12 · label 3: 0.419, higher than 61.4% of the 2,000 test digits." */
  const probeSentence = (s: UnitSummary, v: number | null): string => {
    const p = store.probe;
    if (!p || v === null) {
      if (pts()) return 'No input selected. Click a point above or on the map, or pick one in 03 Decision boundary.';
      return isMnist()
        ? 'No input selected. Click a digit above, or pick or draw one in 02 Network or 03 Draw.'
        : `No input selected. Click ${article(one())} ${one()} above, or pick one in 07 Data.`;
    }
    let t = `Current input, ${lowerFirst(p.caption)}: ${num(v, 3)}, ${rankPhrase(rankIn(s.sorted, v), one())}.`;
    if (scan && !isCurrent(scan.stamp)) t += ` Measured with the weights at step ${int(scan.stamp.step)}, as the scan was.`;
    return t;
  };

  const histogram = (s: UnitSummary, u: number, kind: UnitKind): HTMLElement => {
    const count = scan?.result.count ?? 0;
    const canvas = h('canvas', { role: 'img', 'aria-label': `Histogram of ${unitName(kind, u)}'s response over ${int(count)} test ${many()}` }) as HTMLCanvasElement;
    const box = h('div', { class: 'canvas-box units-hist' }, canvas);
    let geo: { L: number; T: number; pw: number; ph: number; probe: number | null; px: number | null } | null = null;
    const text = () => {
      const t = probeSentence(s, probeResponse(u));
      if (probeLine && probeLine.textContent !== t) probeLine.textContent = t;
    };
    const draw = () => {
      if (!box.isConnected) return;
      const p = palette();
      const w = Math.max(240, box.clientWidth);
      const H = 156;
      const ctx = fitCanvas(canvas, w, H);
      const L = 40;
      const R = 12;
      const T = 20;
      const B = 34;
      const pw = w - L - R;
      const ph = H - T - B;
      const { lo, hi, counts } = s.hist;
      const max = Math.max(1, ...counts);
      const yTop = max <= 5 ? max : Math.ceil(max / (max > 100 ? 50 : 10)) * (max > 100 ? 50 : 10);
      const ty = (c: number) => T + ph * (1 - c / yTop);
      const tx = (v: number) => L + ((v - lo) / (hi - lo)) * pw;
      ctx.font = `400 10px ${MONO}`;
      ctx.lineWidth = 1;
      // y grid: 0, middle, top
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      for (const c of [0, yTop / 2, yTop]) {
        const y = Math.round(ty(c)) + 0.5;
        ctx.strokeStyle = p.hair;
        ctx.beginPath();
        ctx.moveTo(L, y);
        ctx.lineTo(L + pw, y);
        ctx.stroke();
        ctx.fillStyle = p.muted;
        ctx.fillText(Number.isInteger(c) ? String(c) : c.toFixed(1), L - 6, y);
      }
      ctx.textAlign = 'left';
      ctx.textBaseline = 'alphabetic';
      ctx.fillText(many(), 0, 10);
      // bars
      const bw = pw / counts.length;
      ctx.fillStyle = p.ink2;
      counts.forEach((c, i) => {
        if (!c) return;
        const y = ty(c);
        ctx.fillRect(L + i * bw + 0.5, y, Math.max(1, bw - 1), T + ph - y);
      });
      ctx.fillStyle = p.ink;
      ctx.fillRect(L, T + ph, pw, 1);
      // x ticks: the range ends, and 0 when it lies inside
      const ticks = [lo, hi];
      if (lo < 0 && hi > 0 && tx(0) - L > 36 && L + pw - tx(0) > 36) ticks.push(0);
      ctx.textBaseline = 'top';
      for (const v of ticks) {
        const x = Math.round(tx(v)) + 0.5;
        ctx.fillRect(x - 0.5, T + ph, 1, 4);
        ctx.fillStyle = p.muted;
        ctx.textAlign = v === lo ? 'left' : v === hi ? 'right' : 'center';
        ctx.fillText(num(v, 2), v === lo ? x - 0.5 : v === hi ? x + 0.5 : x, T + ph + 6);
        ctx.fillStyle = p.ink;
      }
      ctx.fillStyle = p.muted;
      ctx.textAlign = 'center';
      ctx.fillText('response', L + pw / 2, T + ph + 18);
      // current input
      const pv = probeResponse(u);
      let px: number | null = null;
      if (pv !== null) {
        px = Math.max(L, Math.min(L + pw, tx(pv)));
        ctx.fillStyle = p.accent;
        ctx.fillRect(Math.round(px) - 1, T - 6, 2, ph + 6);
        ctx.fillStyle = p.ink;
        ctx.textBaseline = 'alphabetic';
        const label = pv < lo || pv > hi ? `this input ${num(pv, 2)} (off scale)` : 'this input';
        const tw = ctx.measureText(label).width;
        ctx.textAlign = px + 6 + tw > L + pw ? 'right' : 'left';
        ctx.fillText(label, ctx.textAlign === 'right' ? px - 6 : px + 6, T - 2);
      }
      geo = { L, T, pw, ph, probe: pv, px };
      text();
    };
    canvas.addEventListener('mousemove', (e) => {
      if (!geo) return;
      const r = canvas.getBoundingClientRect();
      const mx = e.clientX - r.left;
      const my = e.clientY - r.top;
      const { L, T, pw, ph } = geo;
      if (mx < L || mx > L + pw || my < T - 8 || my > T + ph) return hideTip();
      const { lo, hi, counts } = s.hist;
      const k = Math.min(counts.length - 1, Math.floor(((mx - L) / pw) * counts.length));
      const a = lo + ((hi - lo) * k) / counts.length;
      const b = lo + ((hi - lo) * (k + 1)) / counts.length;
      let tip = `${num(a, 3)} to ${num(b, 3)}\n${int(counts[k])} ${counts[k] === 1 ? one() : many()}`;
      if (geo.px !== null && geo.probe !== null && store.probe && Math.abs(mx - geo.px) < 6) {
        tip += `\nCurrent input · ${store.probe.caption}\n${num(geo.probe, 3)}, ${rankPhrase(rankIn(s.sorted, geo.probe), one())}`;
      }
      showTip(tip, e.clientX, e.clientY);
    });
    canvas.addEventListener('mouseleave', hideTip);
    detailDraws.push(draw);
    probeText = text;
    requestAnimationFrame(draw);
    return box;
  };

  /** MNIST: one column per digit, bars in ink. */
  const labelBars = (counts: number[]): HTMLElement => {
    const max = Math.max(1, ...counts);
    const n = counts.reduce((a, b) => a + b, 0);
    const row = h('div', { class: 'units-labels', role: 'img', 'aria-label': `Labels of the top ${n}: ${counts.map((c, d) => `${c} ${d}s`).join(', ')}` });
    counts.forEach((c, d) => {
      const col = h(
        'div',
        { class: `units-label-col${c === max ? ' is-max' : ''}` },
        h('span', { class: 'units-label-n' }, c ? String(c) : ''),
        h('span', { class: 'units-label-track' }, h('span', { class: 'units-label-bar', style: { height: `${(c / max) * 100}%` } })),
        h('span', { class: 'units-label-d' }, String(d)),
      );
      col.addEventListener('mousemove', (e) => showTip(`${c} of the top ${n} are ${d}s`, e.clientX, e.clientY));
      col.addEventListener('mouseleave', hideTip);
      row.append(col);
    });
    return row;
  };

  /** Named classes (clothes, objects, point classes): one row per class, its colour, name, bar and count. */
  const labelRows = (counts: number[]): HTMLElement => {
    const max = Math.max(1, ...counts);
    const n = counts.reduce((a, b) => a + b, 0);
    const list = h('div', { class: 'units-label-rows', role: 'img', 'aria-label': `Labels of the top ${n}: ${counts.map((c, k) => `${c} ${lowerFirst(className(k))}`).join(', ')}` });
    counts.forEach((c, k) => {
      list.append(
        h(
          'div',
          { class: `units-label-row${c === max ? ' is-max' : ''}`, title: `${c} of the top ${n} are labelled ${lowerFirst(className(k))}` },
          h('i', { class: 'units-label-swatch', style: { background: classColor(k) } }),
          h('span', { class: 'units-label-name' }, className(k)),
          h('span', { class: 'units-label-hbar' }, h('span', { style: { width: `${(c / max) * 100}%` } })),
          h('span', { class: 'units-label-n' }, String(c)),
        ),
      );
    });
    return list;
  };

  const synthText = (s: SynthState | null, su: SynthUnit | undefined, kind: UnitKind): string => {
    const what = kind === 'output' ? 'logit' : 'pre-activation';
    if (!su) return s?.running ? 'Waiting for its turn…' : `Press Synthesise inputs to optimise an input for every ${nounOf(kind)} in this layer.`;
    if (su.final !== null && su.start !== null) return `Its ${what} rose from ${fix2(su.start)} on a ${blankWords()} image to ${fix2(su.final)} after ${ACTMAX_STEPS} steps.`;
    return s?.running ? `Step ${su.step} of ${ACTMAX_STEPS}: ${what} ${fix2(su.value)}.` : `Stopped at step ${su.step} of ${ACTMAX_STEPS}.`;
  };

  /**
   * Point datasets: the unit's response over the plane, with the test points on it (the strongest
   * filled, the rest as faint dots) and the current input. Hover for values, click to pick.
   */
  const planeMap = (s: UnitSummary, u: number, kind: UnitKind): HTMLElement => {
    const d = store.data!;
    const name = unitName(kind, u);
    const canvas = h('canvas', { role: 'img' }) as HTMLCanvasElement;
    const box = h('div', { class: 'units-plane' }, canvas);
    const scaleBar = h('canvas', { class: 'units-scale-bar', 'aria-hidden': 'true' }) as HTMLCanvasElement;
    const lo = h('span', { class: 'units-scale-lo' });
    const hi = h('span', { class: 'units-scale-hi' });
    const legend = h('div', { class: 'units-scale' }, lo, scaleBar, hi);
    const caption = h('p', { class: 'hint units-plane-note' });
    const tex = document.createElement('canvas');
    const strong = new Set(s.top.slice(0, TOP_K).map((hit) => hit.index));
    let geo: { S: number; r: number; plane: Plane } | null = null;

    const draw = () => {
      if (!box.isConnected) return;
      const m = mapsNow();
      if (!m) return;
      const plane = m.plane;
      const sc = m.scales[u];
      const S = Math.max(180, Math.min(340, box.clientWidth || 300));
      const ctx = fitCanvas(canvas, S, S);
      const p = palette();
      const r = plane.r;
      paintMap(tex, m.cols[u], MAP_RES, sc.signed, sc.max);
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(tex, 0, 0, S, S);
      const tx = (v: number) => ((v + r) / (2 * r)) * S;
      const ty = (v: number) => ((r - v) / (2 * r)) * S;
      // Axes through the origin.
      ctx.fillStyle = p.hair;
      ctx.globalAlpha = 0.8;
      ctx.fillRect(Math.round(tx(0)), 0, 1, S);
      ctx.fillRect(0, Math.round(ty(0)), S, 1);
      ctx.globalAlpha = 1;
      // Every test point, faint; points far from a 3-D slice fainter still.
      const slab = slabWidth(r);
      const off = plane.dims === 3 ? 3 - plane.axes[0] - plane.axes[1] : -1;
      const n = d.testY.length;
      const pp = d.points!;
      for (let i = 0; i < n; i++) {
        if (strong.has(i)) continue;
        const [x, y] = pointUV(d, i, plane);
        const near = off < 0 || Math.abs(pp.testCoords[i * pp.dims + off] - (plane.fixedAt[off] ?? 0)) <= slab;
        ctx.globalAlpha = near ? 0.75 : 0.3;
        ctx.fillStyle = classColor(d.testY[i]);
        ctx.beginPath();
        ctx.arc(tx(x), ty(y), 1.8, 0, 2 * Math.PI);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
      for (const hit of [...s.top.slice(0, TOP_K)].reverse()) {
        const [x, y] = pointUV(d, hit.index, plane);
        dotTrain(ctx, tx(x), ty(y), 4.2, classColor(d.testY[hit.index]), p.ink);
      }
      // The current input.
      const c = store.probe?.coords;
      if (c && c.length === plane.dims) {
        const faint = off >= 0 && Math.abs(c[off] - (plane.fixedAt[off] ?? 0)) > slab;
        probeCross(ctx, tx(c[plane.axes[0]]), ty(c[plane.axes[1]]), 5, p.accent, p.surface, faint);
      }
      // Frame and axis names.
      ctx.strokeStyle = p.ink;
      ctx.lineWidth = 1;
      ctx.strokeRect(0.5, 0.5, S - 1, S - 1);
      ctx.font = `500 10px ${MONO}`;
      ctx.fillStyle = p.ink;
      ctx.textBaseline = 'bottom';
      ctx.textAlign = 'right';
      ctx.fillText(`${axisName(plane.axes[0])} →`, S - 4, S - 3);
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.fillText(`↑ ${axisName(plane.axes[1])}`, 4, 4);
      geo = { S, r, plane };

      // Colour bar: −max … max for signed maps, 0 … max otherwise.
      const BW = 120;
      const bctx = fitCanvas(scaleBar, BW, 8);
      for (let k = 0; k < BW; k++) {
        const t = k / (BW - 1);
        bctx.fillStyle = css(sc.signed ? diverging(2 * t - 1, rgb) : sequential(t, rgb));
        bctx.fillRect(k, 0, 1, 8);
      }
      lo.textContent = sc.signed ? num(-sc.max, 2) : '0';
      hi.textContent = num(sc.max, 2);
      const where = planeText(plane);
      caption.textContent =
        plane.dims === 2
          ? `${name}’s response at every point of ${where}, ${axisName(0)} across and ${axisName(1)} up, from the scan’s weights. Filled dots are its ${Math.min(TOP_K, s.top.length)} strongest test points; the cross is the current input.`
          : `${name}’s response over ${where} (move the slice in 03), from the scan’s weights. Test points are projected onto it, faint when far from the slice; filled dots are the ${Math.min(TOP_K, s.top.length)} strongest.`;
      canvas.setAttribute(
        'aria-label',
        `Response map of ${name} over ${where}: values from ${num(sc.lo, 2)} to ${num(sc.hi, 2)}, ${sc.signed ? 'red positive and blue negative' : 'darker for stronger responses'}. The ${Math.min(TOP_K, s.top.length)} strongest test points are drawn filled.`,
      );
    };

    /** Data coordinates under the pointer, the nearest test point within 7 px, and the response there. */
    const at = (e: MouseEvent) => {
      if (!geo || !scan) return null;
      const { S, r, plane } = geo;
      const b = canvas.getBoundingClientRect();
      const mx = e.clientX - b.left;
      const my = e.clientY - b.top;
      if (mx < 0 || my < 0 || mx > S || my > S) return null;
      const uu = (mx / S) * 2 * r - r;
      const vv = r - (my / S) * 2 * r;
      const coords = new Float32Array(plane.dims);
      for (let k = 0; k < plane.dims; k++) coords[k] = plane.fixedAt[k] ?? 0;
      coords[plane.axes[0]] = uu;
      coords[plane.axes[1]] = vv;
      let best = -1;
      let bd = 49;
      for (let i = 0; i < d.testY.length; i++) {
        const [x, y] = pointUV(d, i, plane);
        const dd = (((x + r) / (2 * r)) * S - mx) ** 2 + (((r - y) / (2 * r)) * S - my) ** 2;
        if (dd < bd) {
          bd = dd;
          best = i;
        }
      }
      return { coords, hit: best };
    };
    canvas.addEventListener('mousemove', (e) => {
      const a = at(e);
      if (!a || !scan || !maps) return hideTip();
      const d0 = store.data!;
      let tip: string;
      if (a.hit >= 0) {
        const v = unitResponse(scan.net, layer, u, sampleInput(d0, 'test', a.hit)).value;
        const pc = Array.from(d0.points!.testCoords.subarray(a.hit * d0.points!.dims, (a.hit + 1) * d0.points!.dims), (x) => fixed(x, 2));
        tip = `Test point #${a.hit} · ${className(d0.testY[a.hit])}\n(${pc.join(', ')})\nresponse ${num(v, 3)}${strong.has(a.hit) ? ' · among the strongest' : ''}`;
      } else {
        const plane = geo!.plane;
        const col = Math.min(MAP_RES - 1, Math.max(0, Math.floor(((a.coords[plane.axes[0]] + plane.r) / (2 * plane.r)) * MAP_RES)));
        const row = Math.min(MAP_RES - 1, Math.max(0, Math.floor(((plane.r - a.coords[plane.axes[1]]) / (2 * plane.r)) * MAP_RES)));
        tip = `${Array.from(a.coords, (x, k) => `${axisName(k)} ${fixed(x, 2)}`).join(' · ')}\nresponse ${num(maps.cols[u][row * MAP_RES + col], 3)}`;
      }
      showTip(tip, e.clientX, e.clientY);
    });
    canvas.addEventListener('mouseleave', hideTip);
    canvas.addEventListener('click', (e) => {
      const a = at(e);
      if (!a) return;
      pickAt(a.coords, a.hit >= 0 ? { split: 'test', i: a.hit } : null, false);
    });
    detailDraws.push(draw);
    requestAnimationFrame(draw);
    return h('div', { class: 'units-block units-plane-block' }, h('p', { class: 'sub' }, 'Response map'), box, legend, caption);
  };

  const renderDetail = () => {
    clear(detail);
    detailDraws = [];
    probeText = null;
    probeLine = null;
    detailSynth = null;
    if (scan) detail.dataset.step = String(scan.stamp.step);
    else delete detail.dataset.step;
    const u = selectedUnit();
    const kind = kindOf(layer);
    const what = nounOf(kind);
    if (u === null) {
      detail.append(
        h('p', { class: 'sub' }, 'Details'),
        h(
          'p',
          { class: 'hint' },
          pts()
            ? `Pick ${article(what)} ${what} to see the 16 test points that excite it most, the 8 that excite it least, its response over ${store.data?.points ? planeText(planeOf(store.data)) : 'the input plane'}, and how its response spreads over the test set.`
            : `Pick ${article(what)} ${what} to see the 16 ${many()} that excite it most, the ${many()} that excite it least, how its response spreads over the test set, and its synthesised input.`,
        ),
      );
      return;
    }
    const name = unitName(kind, u);
    const s = summary(u);
    const f = kind === 'conv' ? field() : null;
    const dead = !!s && isDead(kind, s);
    const outName = isMnist() ? `digit ${u}` : lowerFirst(className(u));
    const responseIs =
      kind === 'conv'
        ? `its strongest activation anywhere on the ${one()}. It fires where its pre-activation is above 0`
        : kind === 'output'
          ? `the logit for ${outName}; softmax turns the logits into probabilities. ${cap(article(one()))} ${one()} is predicted as ${isMnist() ? u : outName} when this logit is the largest of the ${countOf(layer)}`
          : 'its activation. It fires when its pre-activation is above 0';

    const head = h('div', { class: 'units-detail-head' }, h('h3', { class: 'panel-title' }, name), h('span', { class: 'units-detail-layer' }, layerLabel(layer)));
    detail.append(head);
    if (!s) {
      detail.append(h('p', { class: 'hint' }, scanning ? `Scanning the test ${many()}…` : 'Not scanned yet.'));
    } else {
      const count = scan!.result.count;
      const kv = (k: string, v: string, after = '') => h('span', null, `${k} `, h('b', null, v), after);
      const [coverLead, coverTail] =
        kind === 'conv' ? ['fires at', ' of positions'] : kind === 'dense' ? ['fires on', ` of ${many()}`] : ['predicted for', ` of ${many()}`];
      detail.append(
        h(
          'div',
          { class: 'units-stats' },
          kv('mean', num(s.mean, 3)),
          kv('max', num(s.top[0]?.value ?? NaN, 3)),
          dead ? h('span', null, h('b', null, 'never fires')) : kv(coverLead, sharePct(s.coverage, 1), coverTail),
          pts() ? null : kv('sees', f && !f.whole ? `${f.size}×${f.size} px` : 'whole image'),
        ),
        h(
          'p',
          { class: 'hint' },
          dead
            ? `Never fires on the ${int(count)} test ${many()}; ${many()} are ranked by pre-activation instead, so the first ones are the closest to firing. Click any ${one()} to make it the network’s input.`
            : `Response is ${responseIs}. Click any ${one()} to make it the network’s input.`,
        ),
      );
      const weak = s.bottom.slice(0, DETAIL_BOTTOM);
      const strongTitle = dead ? `Closest to firing · top ${Math.min(TOP_K, s.top.length)}` : `Strongest responses · top ${Math.min(TOP_K, s.top.length)}`;
      // Conv filters fire somewhere on almost every image, so their weakest images rarely switch them off.
      const weakTitle = dead ? `Furthest from firing · bottom ${weak.length}` : allOff(weak) ? `What switches it off · weakest ${weak.length}` : `Weakest responses · bottom ${weak.length}`;
      if (pts()) {
        detail.append(planeMap(s, u, kind));
        detail.append(
          h(
            'div',
            { class: 'units-pair units-point-pair' },
            h('div', { class: 'units-block' }, h('p', { class: 'sub' }, strongTitle), h('div', { class: 'units-points' }, ...s.top.slice(0, TOP_K).map(pointButton))),
            h('div', { class: 'units-block' }, h('p', { class: 'sub units-weak-title' }, weakTitle), h('div', { class: 'units-points' }, ...weak.map(pointButton))),
          ),
        );
      } else {
        const strongest = h('div', { class: 'units-digits' }, ...s.top.slice(0, TOP_K).map((hit) => digitButton(hit, true)));
        const weakest = h('div', { class: 'units-digits' }, ...weak.map((hit) => digitButton(hit, true)));
        const boxNote = !f
          ? null
          : f.whole
            ? 'The box marks the pixels the filter sees from the position where it fired hardest.'
            : `The box marks the ${f.size}×${f.size} patch where the filter fired hardest.`;
        detail.append(
          h('div', { class: 'units-block' }, h('p', { class: 'sub' }, strongTitle), boxNote ? h('p', { class: 'hint units-box-note' }, h('i', { class: 'units-swatch-box' }), boxNote) : null, strongest),
          h('div', { class: 'units-block' }, h('p', { class: 'sub units-weak-title' }, weakTitle), weakest),
        );
      }
      probeLine = h('p', { class: 'hint units-probe', 'aria-live': 'polite' });
      detail.append(
        h(
          'div',
          { class: 'units-block' },
          h('p', { class: 'sub' }, `Response across ${int(count)} test ${many()}`),
          histogram(s, u, kind),
          h(
            'div',
            { class: 'legend units-legend' },
            h('span', { class: 'legend-item' }, h('i', { class: 'units-swatch-bar' }), `Test ${many()} per bin`),
            h('span', { class: 'legend-item' }, h('i', { class: 'units-swatch-line' }), 'Current input'),
          ),
          probeLine,
        ),
      );
    }

    // Labels and synthesised input side by side when there is room.
    const pair = h('div', { class: 'units-pair' });
    if (s) {
      const n = Math.min(TOP_LABELS, scan!.result.count);
      const names = isMnist() ? undefined : store.info.classes;
      pair.append(
        h(
          'div',
          { class: 'units-block' },
          h('p', { class: 'sub' }, dead ? `Labels of the ${n} closest to firing` : `Labels of the top ${n}`),
          h('p', { class: 'units-label-sum' }, labelSummary(s.labelCounts, names)),
          names ? labelRows(s.labelCounts) : labelBars(s.labelCounts),
        ),
      );
    }
    if (!pts()) {
      const st = synth();
      const su = st?.units.get(u);
      const canvas = h('canvas', { class: 'units-px', role: 'img', 'aria-label': `Input synthesised to excite ${name}` }) as HTMLCanvasElement;
      const empty = h('span', { class: 'units-synth-empty' }, 'Not yet');
      if (su) {
        paintSynth(canvas, su.x);
        empty.hidden = true;
      } else canvas.hidden = true;
      const text = h('p', { class: 'hint' }, synthText(st, su, kind));
      pair.append(
        h(
          'div',
          { class: 'units-block' },
          h('p', { class: 'sub' }, 'Synthesised input'),
          h('div', { class: 'units-synth units-synth-large' }, canvas, empty),
          f
            ? h(
                'p',
                { class: 'hint' },
                f.cropSide < f.side
                  ? `Cropped to the ${f.cropSide}×${f.cropSide} patch the filter sees at the centre of the image.`
                  : f.whole
                    ? 'The filter at the centre of its map sees the whole image.'
                    : `The filter at the centre of its map sees a ${f.size}×${f.size} patch; pixels outside it stay blank.`,
              )
            : null,
          text,
        ),
      );
      detailSynth = { unit: u, canvas, text, empty };
    }
    if (pair.childElementCount) detail.append(pair);
  };

  // ── Synthesis (image datasets) ──
  const renderSynthUI = () => {
    const points = pts();
    synthBtn.hidden = points;
    synthBar.el.hidden = points;
    if (points) {
      const d = store.data;
      const plane = d?.points ? planeOf(d) : null;
      synthNote.textContent = !plane
        ? 'Each card maps the unit’s response over the input plane.'
        : plane.dims === 2
          ? `Each card maps the unit’s response over ${planeText(plane)}, ${axisName(0)} across and ${axisName(1)} up.`
          : `Each card maps the unit’s response over ${planeText(plane)}; move the slice in 03 Decision boundary.`;
      return;
    }
    const s = synth();
    synthBtn.textContent = s?.running ? 'Stop' : 'Synthesise inputs';
    synthBtn.disabled = !ready();
    synthBar.show(!!s?.running);
    const n = countOf(layer);
    const kind = kindOf(layer);
    const what = nounOf(kind);
    const step = s ? s.stamp.step : 0;
    if (!s) synthNote.textContent = `Optimises an input for each of the ${n} ${what}${n === 1 ? '' : 's'}, ${ACTMAX_STEPS} steps each.`;
    else if (s.error) synthNote.textContent = `Synthesis failed: ${s.error}`;
    else if (s.running) {
      synthNote.textContent = s.current === null ? 'Starting…' : `Synthesising ${unitName(kind, s.current)}, ${s.current + 1} of ${s.total}…`;
    } else if (s.stopped) synthNote.textContent = `Stopped after ${s.current ?? 0} of ${s.total} ${what}s.`;
    else if (isCurrent(s.stamp)) synthNote.textContent = `Synthesised from the weights at step ${int(step)}.`;
    else if (step !== store.weightsStep) synthNote.textContent = `Synthesised at step ${int(step)}; the network is now at step ${int(store.weightsStep)}.`;
    else synthNote.textContent = `Synthesised at step ${int(step)}, before the latest manual weight update.`;
  };

  const showSynth = (u: number) => {
    const s = synth();
    const su = s?.units.get(u);
    if (!su) return;
    const c = cards.get(u);
    if (c) {
      paintSynth(c.synth, su.x);
      c.synth.hidden = false;
      c.empty.hidden = true;
    }
    if (detailSynth && detailSynth.unit === u) {
      paintSynth(detailSynth.canvas, su.x);
      detailSynth.canvas.hidden = false;
      detailSynth.empty.hidden = true;
      detailSynth.text.textContent = synthText(s, su, kindOf(layer));
    }
  };

  synthBtn.addEventListener('click', () => {
    const cur = synth();
    if (cur?.running) {
      cur.running = false;
      cur.stopped = true;
      analysis.cancel('units-actmax');
      renderSynthUI();
      if (detailSynth) detailSynth.text.textContent = synthText(cur, cur.units.get(detailSynth.unit), kindOf(layer));
      return;
    }
    if (!ready() || pts()) return;
    const block = layer;
    const version = store.version;
    const st: SynthState = { block, stamp: stampNow(), running: true, stopped: false, total: countOf(block), current: null, units: new Map(), error: null };
    synths.set(block, st);
    synthBar.set(0);
    renderCards();
    renderDetail();
    renderSynthUI();
    const live = () => synths.get(block) === st && version === store.version;
    analysis
      .run<ActmaxResult>('units-actmax', 'actmax', { block, steps: ACTMAX_STEPS }, (p) => {
        if (!live() || !st.running) return;
        synthBar.set(p.done / p.total);
        const part = p.partial as ActmaxPartial | undefined;
        if (part) {
          st.current = part.unit;
          const prev = st.units.get(part.unit);
          if (!prev || prev.final === null) st.units.set(part.unit, { x: part.x, step: part.step, value: part.value, start: null, final: null });
          // Units before this one are finished; their last snapshot stands until the result.
          if (layer === block) showSynth(part.unit);
        }
        renderSynthUI();
      })
      .then((r) => {
        if (!live()) return;
        st.running = false;
        for (const v of r.units) st.units.set(v.unit, { x: v.x, step: r.steps, value: v.final, start: v.start, final: v.final });
        if (layer === block) {
          for (const v of r.units) showSynth(v.unit);
        }
        renderSynthUI();
      })
      .catch((e: unknown) => {
        if (isSuperseded(e) || !live()) return;
        st.running = false;
        st.error = e instanceof Error ? e.message : String(e);
        renderSynthUI();
      });
  });

  more.addEventListener('click', () => {
    const before = more.getBoundingClientRect().top;
    showAll = !showAll;
    renderCards();
    // Collapsing removes cards above the button: keep the button where it was on screen, or the
    // page would be left far below the section.
    const shift = more.getBoundingClientRect().top - before;
    if (shift) window.scrollTo({ top: window.scrollY + shift, behavior: 'instant' });
  });

  // ── Assembly ──
  const renderAll = () => {
    renderCards();
    renderDetail();
    renderSynthUI();
  };

  root.append(
    h('div', { class: 'units-controls' }, layerSlot, h('div', { class: 'units-synth-ctl' }, h('div', { class: 'units-synth-row' }, synthBtn, synthNote), synthBar.el)),
    sync.status,
    gridNote,
    layout,
  );

  // ── Store events ──
  store.on('model', () => {
    for (const s of synths.values()) if (s.running) analysis.cancel('units-actmax');
    synths.clear();
    scan = null;
    maps = null;
    scanning = null; // a scan of the old network is useless; ensureScan below replaces it
    probeCache = null;
    scanError = null;
    showAll = false;
    layer = clampLayer(store.selected);
    markUpdating();
    buildLayerSelect();
    renderAll();
    ensureScan();
  });
  store.on('select', () => {
    const b = clampLayer(store.selected);
    if (b !== layer) {
      setLayer(b);
      return;
    }
    const sel = selectedUnit();
    if (sel !== null && sel >= CARD_LIMIT && !showAll) {
      // The selected unit's card is past the first 32: show them all.
      showAll = true;
      renderCards();
    } else syncPressed();
    renderDetail();
  });

  let queued = false;
  /** Redraws the detail canvases (histogram and map, with the current input) on the next frame, if on screen. */
  function redrawProbe() {
    if (!detailDraws.length) return;
    if (!visible) {
      probeDirty = true;
      return;
    }
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => {
      queued = false;
      probeDirty = false;
      for (const draw of detailDraws) draw();
    });
  }
  store.on('probe', () => {
    for (const b of detail.querySelectorAll<HTMLElement>('.units-digit, .units-point')) b.setAttribute('aria-pressed', String(store.probe?.key === `test:${b.dataset.index}`));
    redrawProbe();
  });
  // New weights do not move the marker (it uses the scan's weights); only the "measured at" note changes.
  store.on('weights', () => {
    probeText?.();
    renderSynthUI();
  });
  store.on('data', () => {
    // A different test set (regenerated points, other features) makes the scan's indices meaningless.
    if (scan && scan.data.testX !== store.data?.testX) {
      scan = null;
      maps = null;
      probeCache = null;
    }
    if (scanning && scanning.testX !== store.data?.testX) {
      analysis.cancel('units-topk');
      scanning = null;
      sync.fail();
    }
    renderAll();
    ensureScan();
  });
  store.on('dataset', renderNote);

  /** A new 3-D slice in 03: recompute the maps (cheap) and repaint, once per frame, when on screen. */
  let replaneQueued = false;
  function replane() {
    if (!pts() || store.data?.points?.dims !== 3) return;
    if (!visible) {
      planeDirty = true;
      return;
    }
    planeDirty = false;
    if (replaneQueued) return;
    replaneQueued = true;
    requestAnimationFrame(() => {
      replaneQueued = false;
      const m = mapsNow();
      gridKey.textContent = keyText();
      renderSynthUI();
      for (const [u, c] of cards) {
        const map = c.btn.querySelector<HTMLCanvasElement>('canvas.units-map');
        const scatter = c.btn.querySelector<HTMLCanvasElement>('canvas.units-scatter');
        const s = summary(u);
        if (m && map) paintMap(map, m.cols[u], MAP_RES, m.scales[u].signed, m.scales[u].max);
        if (s && scatter) drawScatter(scatter, s.top.slice(0, CARD_TOP));
      }
      for (const draw of detailDraws) draw();
    });
  }
  onSlice(replane);

  onThemeChange(() => {
    renderCards();
    renderDetail();
  });
  new ResizeObserver(() => redrawProbe()).observe(detail);

  layer = clampLayer(store.selected);
  buildLayerSelect();
  renderAll();
}
