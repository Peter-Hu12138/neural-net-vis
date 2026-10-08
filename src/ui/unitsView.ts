import './unitsView.css';
import { select, setProbe } from '../actions';
import { analysis, isSuperseded } from '../analysis/client';
import { receptiveBox, receptiveSize, type Box } from '../analysis/receptive';
import { ACTMAX_STEPS, TOP_LABELS, labelSummary, unitResponse, type ActmaxPartial, type ActmaxResult, type Hit, type TopkResult, type UnitKind, type UnitSummary } from '../analysis/units';
import { sampleToFloat } from '../data/mnist';
import { store } from '../store';
import { layerDetail, layerName } from './builder';
import { $, clear, fmt, h, int, pct, selectField } from './dom';
import { fitCanvas } from './draw';
import { syncedSection } from './snapshot';
import { onThemeChange, palette, sequential, type RGB } from './theme';
import { hideTip, showTip } from './tip';

/** Section 09: the test digits that excite each unit most and least, and inputs synthesised to excite it. */

const HINT =
  'Top images are the test digits that excite a unit most; for conv filters only the patch the filter sees is shown. Synthesised inputs are images optimised from blank to excite the unit.';
const CARD_LIMIT = 32;
const TOP_K = 16;
const CARD_TOP = 9;
const DETAIL_BOTTOM = 8;
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
  step: number;
  running: boolean;
  stopped: boolean;
  total: number;
  /** The unit being optimised now. */
  current: number | null;
  units: Map<number, SynthUnit>;
  error: string | null;
}

const rgb: RGB = [0, 0, 0];

/** Paints H×W values in [0, 1] at one canvas pixel each; CSS scales the canvas up, pixelated. */
function paintPixels(c: HTMLCanvasElement, data: ArrayLike<number>, H: number, W: number): void {
  if (c.width !== W) c.width = W;
  if (c.height !== H) c.height = H;
  const ctx = c.getContext('2d');
  if (!ctx) return;
  const img = ctx.createImageData(W, H);
  const d = img.data;
  for (let i = 0; i < H * W; i++) {
    sequential(data[i], rgb);
    d[4 * i] = rgb[0];
    d[4 * i + 1] = rgb[1];
    d[4 * i + 2] = rgb[2];
    d[4 * i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

/** Pixels of a 28×28 image (values via `at`) inside `b`; pixels beyond the image are blank. */
function crop(at: (i: number) => number, b: Box): { data: Float32Array; H: number; W: number } {
  const H = b.y1 - b.y0 + 1;
  const W = b.x1 - b.x0 + 1;
  const data = new Float32Array(H * W);
  for (let r = 0; r < H; r++) {
    const y = b.y0 + r;
    if (y < 0 || y > 27) continue;
    for (let c = 0; c < W; c++) {
      const x = b.x0 + c;
      if (x >= 0 && x <= 27) data[r * W + c] = at(y * 28 + x);
    }
  }
  return { data, H, W };
}

const unitName = (kind: UnitKind, u: number) => (kind === 'output' ? `Digit ${u}` : kind === 'conv' ? `Filter ${u + 1}` : `Unit ${u + 1}`);

const plural = (d: number) => `${d}s`;

/** Two decimals, without a "−0.00". */
const fix2 = (v: number) => (Math.abs(v) < 0.005 ? 0 : v).toFixed(2);

/** Shared thin progress bar. It keeps its space while idle, so nothing below it jumps. */
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
  let top: TopkResult | null = null;
  let scanFor: { block: number; version: number } | null = null;
  let scanning = false;
  let scanError: string | null = null;
  let showAll = false;
  let visible = false;
  const synths = new Map<number, SynthState>();

  // ── Elements ──
  const layerSlot = h('div', { class: 'units-layer' });
  const synthBtn = h('button', { type: 'button', id: 'units-synth', class: 'btn btn-sm btn-solid' }, 'Synthesise inputs') as HTMLButtonElement;
  const synthNote = h('span', { class: 'hint units-synth-note', 'aria-live': 'polite' });
  const synthBar = progressBar('Synthesis progress');
  const scanBar = progressBar('Scan progress');
  const scanLine = h('span', { class: 'units-scan-line' }, scanBar.el, h('span', { class: 'hint' }, 'Scanning the test digits…'));
  const grid = h('div', { class: 'units-grid' });
  const more = h('button', { type: 'button', id: 'units-show-all', class: 'btn btn-sm' }) as HTMLButtonElement;
  const gridNote = h('p', { class: 'hint units-grid-note' });
  const gridKey = h('p', { class: 'hint units-key' });
  const detail = h('div', { class: 'units-detail', id: 'units-detail', role: 'region', 'aria-label': 'Unit details' });

  const blocks = () => store.net.blocks;
  const clampLayer = (i: number) => Math.max(0, Math.min(blocks().length - 1, i));
  const kindOf = (i: number): UnitKind => (i === blocks().length - 1 ? 'output' : blocks()[i].kind);
  const countOf = (i: number) => {
    const b = blocks()[i];
    return b.kind === 'conv' ? b.spec.filters : b.spec.units;
  };
  const specOf = (i: number) => (i === blocks().length - 1 ? null : store.net.spec[i]);
  const summary = (u: number): UnitSummary | null => (top && top.block === layer ? top.units[u] ?? null : null);
  const selectedUnit = (): number | null => {
    if (store.selected !== layer || store.selectedUnit === null) return null;
    return store.selectedUnit < countOf(layer) ? store.selectedUnit : null;
  };
  const synth = () => synths.get(layer) ?? null;
  const digitAt = (i: number) => {
    const d = store.data!;
    return (p: number) => d.testX[i * 784 + p] / 255;
  };
  const centreBox = (raw: boolean): Box | null => {
    const b = blocks()[layer];
    if (b.kind !== 'conv') return null;
    return receptiveBox(store.net.spec, layer, b.zShape.h >> 1, b.zShape.w >> 1, 'z', !raw);
  };

  // ── Synced top-k scan ──
  const refresh = () => {
    if (!store.data || !store.valid) return;
    const block = layer;
    const version = store.version;
    sync.markComputed();
    scanFor = { block, version };
    scanning = true;
    scanError = null;
    scanBar.set(0);
    scanBar.show(true);
    scanLine.classList.add('is-on');
    renderNote();
    analysis
      .run<TopkResult>('units-topk', 'topk', { block, k: TOP_K }, (p) => scanBar.set(p.done / p.total))
      .then((r) => {
        scanning = false;
        scanBar.show(false);
        scanLine.classList.remove('is-on');
        if (r.block !== layer || version !== store.version) return;
        top = r;
        renderCards();
        renderDetail();
      })
      .catch((e: unknown) => {
        if (isSuperseded(e)) return;
        scanning = false;
        scanBar.show(false);
        scanLine.classList.remove('is-on');
        scanError = e instanceof Error ? e.message : String(e);
        renderNote();
      });
  };
  const sync = syncedSection(root, refresh);
  // Scan progress sits in the status row, so the cards below never shift when it ends.
  sync.status.append(scanLine);

  /** Scans the current layer if the section is on screen and no scan covers it yet. */
  const ensureScan = () => {
    if (!visible || !store.data || !store.valid) return;
    if (scanFor && scanFor.block === layer && scanFor.version === store.version) return;
    sync.refreshNow();
  };
  new IntersectionObserver(
    (entries) => {
      visible = entries.some((e) => e.isIntersecting);
      ensureScan();
    },
    { rootMargin: '200px 0px' },
  ).observe(root);

  // ── Layer select ──
  const buildLayerSelect = () => {
    clear(layerSlot);
    layerSlot.append(
      selectField(
        'units-layer',
        'Layer',
        blocks().map((_, i) => ({ value: i, label: `${layerName(specOf(i), i)} · ${layerDetail(specOf(i))}` })),
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
    if (top && top.block !== layer) top = null;
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
    if (hit.y >= 0) {
      const b = receptiveBox(store.net.spec, layer, hit.y, hit.x, 'z', false)!;
      const { data, H, W } = crop(digitAt(hit.index), b);
      paintPixels(c, data, H, W);
    } else {
      paintPixels(c, sampleToFloat(store.data!.testX, hit.index), 28, 28);
    }
    return c;
  };

  const paintSynth = (c: HTMLCanvasElement, x: Float32Array) => {
    const b = centreBox(true);
    if (b) {
      const { data, H, W } = crop((p) => x[p], b);
      paintPixels(c, data, H, W);
    } else paintPixels(c, x, 28, 28);
  };

  const card = (u: number): HTMLElement => {
    const kind = kindOf(layer);
    const s = summary(u);
    const name = unitName(kind, u);
    const mosaic = h('span', { class: 'units-mosaic' });
    if (s && store.data) for (const hit of s.top.slice(0, CARD_TOP)) mosaic.append(hitCanvas(hit));
    else for (let i = 0; i < CARD_TOP; i++) mosaic.append(h('span', { class: 'units-px is-blank' }));
    const synthC = h('canvas', { class: 'units-px', 'aria-hidden': 'true' }) as HTMLCanvasElement;
    const empty = h('span', { class: 'units-synth-empty' }, 'Not yet');
    const su = synth()?.units.get(u);
    if (su) {
      paintSynth(synthC, su.x);
      empty.hidden = true;
    } else synthC.hidden = true;
    const sel = selectedUnit() === u;
    const label = s
      ? `${name}: active on ${pct(s.activeFraction, 0)} of digits, mean response ${fmt(s.mean, 2)}. Show details.`
      : `${name}. Show details.`;
    const btn = h(
      'button',
      { type: 'button', class: 'units-card', 'aria-pressed': String(sel), 'aria-label': label, 'data-unit': String(u) },
      h('span', { class: 'units-card-head' }, h('span', { class: 'units-card-title' }, name), h('span', { class: 'units-card-mean' }, s ? `mean ${fmt(s.mean, 2)}` : '')),
      h(
        'span',
        { class: 'units-card-body' },
        h('span', { class: 'units-fig' }, mosaic),
        h('span', { class: 'units-fig' }, h('span', { class: 'units-synth' }, synthC, empty)),
      ),
      h('span', { class: 'units-card-foot' }, s ? `active on ${pct(s.activeFraction, 0)} of digits` : scanning ? 'Scanning…' : 'Not scanned yet'),
    ) as HTMLButtonElement;
    btn.addEventListener('click', () => {
      select(layer, u);
      revealDetail();
    });
    cards.set(u, { btn, synth: synthC, empty });
    return btn;
  };

  const renderCards = () => {
    clear(grid);
    cards.clear();
    const n = countOf(layer);
    const shown = showAll ? n : Math.min(n, CARD_LIMIT);
    for (let u = 0; u < shown; u++) grid.append(card(u));
    gridKey.textContent =
      kindOf(layer) === 'conv'
        ? 'Each card: the 9 patches that excite the filter most (left) and its synthesised input (right).'
        : 'Each card: the 9 test digits that excite the unit most (left) and its synthesised input (right).';
    more.hidden = n <= CARD_LIMIT;
    more.textContent = showAll ? `Show the first ${CARD_LIMIT}` : `Show all ${n}`;
    renderNote();
  };

  const renderNote = () => {
    if (!store.valid) gridNote.textContent = 'Fix the architecture above to analyse its units.';
    else if (scanError) gridNote.textContent = `The scan failed: ${scanError}`;
    else if (!store.data) gridNote.textContent = 'Waiting for MNIST to load…';
    else gridNote.textContent = '';
    gridNote.hidden = !gridNote.textContent;
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
  let histDraw: (() => void) | null = null;
  let probeLine: HTMLElement | null = null;
  let detailSynth: { unit: number; canvas: HTMLCanvasElement; text: HTMLElement; empty: HTMLElement } | null = null;

  const digitButton = (hit: Hit, showBox: boolean): HTMLElement => {
    const d = store.data!;
    const i = hit.index;
    const label = d.testY[i];
    const c = h('canvas', { class: 'units-px', 'aria-hidden': 'true' }) as HTMLCanvasElement;
    paintPixels(c, sampleToFloat(d.testX, i), 28, 28);
    const box = showBox && hit.box
      ? h('span', {
          class: 'units-digit-box',
          style: {
            left: `${(hit.box.x0 / 28) * 100}%`,
            top: `${(hit.box.y0 / 28) * 100}%`,
            width: `${((hit.box.x1 - hit.box.x0 + 1) / 28) * 100}%`,
            height: `${((hit.box.y1 - hit.box.y0 + 1) / 28) * 100}%`,
          },
        })
      : null;
    const btn = h(
      'button',
      {
        type: 'button',
        class: 'units-digit',
        'data-index': String(i),
        'aria-pressed': String(store.probe?.key === `test:${i}`),
        title: `Test digit #${i} · label ${label} · response ${fmt(hit.value, 3)}`,
        'aria-label': `Use test digit ${i}, a ${label} with response ${fmt(hit.value, 3)}, as the network input`,
      },
      c,
      box,
      h('span', { class: 'units-digit-label' }, String(label)),
    );
    btn.addEventListener('click', () => setProbe({ x: sampleToFloat(d.testX, i), label, caption: `Test digit #${i} · label ${label}`, key: `test:${i}` }));
    return btn;
  };

  const probeResponse = (u: number): number | null => {
    const p = store.probe;
    if (!p || layer >= blocks().length) return null;
    return unitResponse(store.net, layer, u, p.x).value;
  };

  const rankText = (share: number) =>
    share >= 0.995 ? 'as high as the strongest test digits' : share <= 0.005 ? 'as low as the weakest test digits' : `higher than about ${pct(share, 0)} of the test digits`;

  /** Approximate share of test digits below `v`, interpolating inside the histogram bin. */
  const shareBelow = (s: UnitSummary, v: number) => {
    const { lo, hi, counts } = s.hist;
    const n = counts.reduce((a, b) => a + b, 0);
    if (!n) return 0;
    const t = ((v - lo) / (hi - lo)) * counts.length;
    if (t <= 0) return 0;
    if (t >= counts.length) return 1;
    const k = Math.floor(t);
    let below = 0;
    for (let i = 0; i < k; i++) below += counts[i];
    return (below + counts[k] * (t - k)) / n;
  };

  const histogram = (s: UnitSummary, u: number, kind: UnitKind): HTMLElement => {
    const canvas = h('canvas', { role: 'img', 'aria-label': `Histogram of ${unitName(kind, u)}'s response over ${int(top?.count ?? 0)} test digits` }) as HTMLCanvasElement;
    const box = h('div', { class: 'canvas-box units-hist' }, canvas);
    let geo: { L: number; T: number; pw: number; ph: number; probe: number | null; px: number | null } | null = null;
    const draw = () => {
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
      ctx.fillText('digits', 0, 10);
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
        ctx.fillText(fmt(v, 2), v === lo ? x - 0.5 : v === hi ? x + 0.5 : x, T + ph + 6);
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
        const text = pv < lo || pv > hi ? `this input ${fmt(pv, 2)} (off scale)` : 'this input';
        const tw = ctx.measureText(text).width;
        ctx.textAlign = px + 6 + tw > L + pw ? 'right' : 'left';
        ctx.fillText(text, ctx.textAlign === 'right' ? px - 6 : px + 6, T - 2);
      }
      geo = { L, T, pw, ph, probe: pv, px };
      if (probeLine) {
        probeLine.textContent =
          pv === null
            ? 'No input selected.'
            : `This input: ${fmt(pv, 3)}, ${rankText(shareBelow(s, pv))}.`;
      }
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
      let text = `${fmt(a, 3)} to ${fmt(b, 3)}\n${int(counts[k])} digit${counts[k] === 1 ? '' : 's'}`;
      if (geo.px !== null && geo.probe !== null && Math.abs(mx - geo.px) < 6) text += `\nthis input ${fmt(geo.probe, 3)}`;
      showTip(text, e.clientX, e.clientY);
    });
    canvas.addEventListener('mouseleave', hideTip);
    histDraw = draw;
    requestAnimationFrame(draw);
    return box;
  };

  const labelBars = (counts: number[]): HTMLElement => {
    const max = Math.max(1, ...counts);
    const n = counts.reduce((a, b) => a + b, 0);
    const row = h('div', { class: 'units-labels', role: 'img', 'aria-label': `Labels of the top ${n}: ${counts.map((c, d) => `${c} ${plural(d)}`).join(', ')}` });
    counts.forEach((c, d) => {
      const col = h(
        'div',
        { class: `units-label-col${c === max ? ' is-max' : ''}` },
        h('span', { class: 'units-label-n' }, c ? String(c) : ''),
        h('span', { class: 'units-label-track' }, h('span', { class: 'units-label-bar', style: { height: `${(c / max) * 100}%` } })),
        h('span', { class: 'units-label-d' }, String(d)),
      );
      col.addEventListener('mousemove', (e) => showTip(`${c} of the top ${n} are ${plural(d)}`, e.clientX, e.clientY));
      col.addEventListener('mouseleave', hideTip);
      row.append(col);
    });
    return row;
  };

  const synthText = (s: SynthState | null, su: SynthUnit | undefined, kind: UnitKind): string => {
    const what = kind === 'output' ? 'logit' : 'pre-activation';
    if (!su) return s?.running ? 'Waiting for its turn…' : 'Press Synthesise inputs to optimise an input for every unit in this layer.';
    if (su.final !== null && su.start !== null) return `Its ${what} rose from ${fix2(su.start)} on a blank image to ${fix2(su.final)} after ${ACTMAX_STEPS} steps.`;
    return s?.running ? `Step ${su.step} of ${ACTMAX_STEPS}: ${what} ${fix2(su.value)}.` : `Stopped at step ${su.step} of ${ACTMAX_STEPS}.`;
  };

  const renderDetail = () => {
    clear(detail);
    histDraw = null;
    probeLine = null;
    detailSynth = null;
    const u = selectedUnit();
    if (u === null) {
      detail.append(
        h('p', { class: 'sub' }, 'Details'),
        h('p', { class: 'hint' }, 'Pick a unit to see the 16 digits that excite it most, the digits that switch it off, how its response spreads over the test set, and its synthesised input.'),
      );
      return;
    }
    const kind = kindOf(layer);
    const spec = specOf(layer);
    const name = unitName(kind, u);
    const s = summary(u);
    const size = receptiveSize(store.net.spec, layer);
    const responseIs =
      kind === 'conv' ? 'its strongest activation anywhere on the digit' : kind === 'output' ? `the logit for digit ${u}, before softmax` : 'its activation';

    const head = h(
      'div',
      { class: 'units-detail-head' },
      h('h3', null, name),
      h('span', { class: 'units-detail-layer' }, `${layerName(spec, layer)} · ${layerDetail(spec)}`),
    );
    detail.append(head);
    if (s) {
      const kv = (k: string, v: string) => h('span', null, `${k} `, h('b', null, v));
      detail.append(
        h(
          'div',
          { class: 'units-stats' },
          kv('mean', fmt(s.mean, 3)),
          kv('active', pct(s.activeFraction, 0)),
          kv('max', fmt(s.top[0]?.value ?? NaN, 3)),
          kind === 'conv' ? kv('sees', `${size}×${size} px`) : kv('sees', 'whole image'),
        ),
        h('p', { class: 'hint' }, `Response is ${responseIs}. Click any digit to make it the network’s input.`),
      );
      const strongest = h('div', { class: 'units-digits' }, ...s.top.slice(0, TOP_K).map((hit) => digitButton(hit, true)));
      const weakest = h('div', { class: 'units-digits' }, ...s.bottom.slice(0, DETAIL_BOTTOM).map((hit) => digitButton(hit, true)));
      detail.append(
        h(
          'div',
          { class: 'units-block' },
          h('p', { class: 'sub' }, `Strongest responses · top ${Math.min(TOP_K, s.top.length)}`),
          kind === 'conv' ? h('p', { class: 'hint units-box-note' }, h('i', { class: 'units-swatch-box' }), `The box marks the ${size}×${size} patch where the filter fired hardest.`) : null,
          strongest,
        ),
        h('div', { class: 'units-block' }, h('p', { class: 'sub' }, 'What switches it off · weakest 8'), weakest),
      );
      probeLine = h('p', { class: 'hint units-probe' });
      detail.append(
        h(
          'div',
          { class: 'units-block' },
          h('p', { class: 'sub' }, `Response across ${int(top!.count)} test digits`),
          histogram(s, u, kind),
          h(
            'div',
            { class: 'legend units-legend' },
            h('span', { class: 'legend-item' }, h('i', { class: 'units-swatch-bar' }), 'Test digits per bin'),
            h('span', { class: 'legend-item' }, h('i', { class: 'units-swatch-line' }), 'Current input'),
          ),
          probeLine,
        ),
      );
    } else {
      detail.append(h('p', { class: 'hint' }, scanning ? 'Scanning the test digits…' : 'Not scanned yet.'));
    }

    // Labels and synthesised input side by side when there is room.
    const pair = h('div', { class: 'units-pair' });
    if (s) {
      const n = Math.min(TOP_LABELS, top!.count);
      pair.append(h('div', { class: 'units-block' }, h('p', { class: 'sub' }, `Labels of the top ${n}`), h('p', { class: 'units-label-sum' }, labelSummary(s.labelCounts)), labelBars(s.labelCounts)));
    }
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
        kind === 'conv' ? h('p', { class: 'hint' }, `Cropped to the ${size}×${size} patch the filter sees at the centre of the image.`) : null,
        text,
      ),
    );
    detail.append(pair);
    detailSynth = { unit: u, canvas, text, empty };
  };

  // ── Synthesis ──
  const renderSynthUI = () => {
    const s = synth();
    synthBtn.textContent = s?.running ? 'Stop' : 'Synthesise inputs';
    synthBtn.disabled = !store.data || !store.valid;
    synthBar.show(!!s?.running);
    const n = countOf(layer);
    if (!s) synthNote.textContent = `Optimises an input for each of the ${n} unit${n === 1 ? '' : 's'}, ${ACTMAX_STEPS} steps each.`;
    else if (s.error) synthNote.textContent = `Synthesis failed: ${s.error}`;
    else if (s.running) {
      synthNote.textContent = s.current === null ? 'Starting…' : `Synthesising ${unitName(kindOf(layer), s.current)}, ${s.current + 1} of ${s.total}…`;
    } else if (s.stopped) synthNote.textContent = `Stopped after ${s.current ?? 0} of ${s.total} units.`;
    else if (s.step !== store.weightsStep) synthNote.textContent = `Synthesised at step ${int(s.step)}; the network is now at step ${int(store.weightsStep)}.`;
    else synthNote.textContent = `Synthesised from the weights at step ${int(s.step)}.`;
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
    if (!store.data || !store.valid) return;
    const block = layer;
    const version = store.version;
    const st: SynthState = { block, step: store.weightsStep, running: true, stopped: false, total: countOf(block), current: null, units: new Map(), error: null };
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
    showAll = !showAll;
    renderCards();
  });

  // ── Assembly ──
  const renderAll = () => {
    renderCards();
    renderDetail();
    renderSynthUI();
  };

  root.append(
    h('p', { class: 'hint' }, HINT),
    h('div', { class: 'units-controls' }, layerSlot, h('div', { class: 'units-synth-ctl' }, h('div', { class: 'units-synth-row' }, synthBtn, synthNote), synthBar.el)),
    sync.status,
    h('div', { class: 'units-layout' }, h('div', { class: 'units-main' }, gridNote, gridKey, grid, more), detail),
  );

  // ── Store events ──
  store.on('model', () => {
    for (const s of synths.values()) if (s.running) analysis.cancel('units-actmax');
    synths.clear();
    top = null;
    scanFor = null;
    scanError = null;
    showAll = false;
    layer = clampLayer(store.selected);
    buildLayerSelect();
    renderAll();
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
  const redrawProbe = () => {
    if (queued || !histDraw) return;
    queued = true;
    requestAnimationFrame(() => {
      queued = false;
      histDraw?.();
    });
  };
  store.on('probe', () => {
    for (const b of detail.querySelectorAll<HTMLElement>('.units-digit')) b.setAttribute('aria-pressed', String(store.probe?.key === `test:${b.dataset.index}`));
    redrawProbe();
  });
  store.on('weights', () => {
    redrawProbe();
    renderSynthUI();
  });
  store.on('data', () => {
    renderAll();
    ensureScan();
  });
  onThemeChange(() => {
    renderCards();
    renderDetail();
  });
  new ResizeObserver(() => redrawProbe()).observe(detail);

  layer = clampLayer(store.selected);
  buildLayerSelect();
  renderAll();
}
