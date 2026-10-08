import './datasetPicker.css';
import { loading, setDataset, setFeatures, setPointsConfig, setTrainLimit } from '../actions';
import { fixed } from '../analysis/stats';
import { DATASETS, noun, sampleInput, type DatasetId, type DatasetInfo } from '../data/datasets';
import { featureCatalog } from '../data/features';
import { DEFAULT_SYNTHETIC, generate, type SyntheticId } from '../data/synthetic';
import { store } from '../store';
import { $, clear, h, int } from './dom';
import { drawSample, fitCanvas } from './draw';
import { classColor, onThemeChange, palette } from './theme';

/**
 * The dataset strip under the training bar: every dataset as a ruled index (grouped as images,
 * 2-D points and 3-D points) with a small preview of each, the "train on" subset control, and
 * for point datasets the generator settings and input features (as in the TensorFlow Playground).
 */

const PREVIEW = 32;
/** Matches .dsp-index's scroll-padding: where a snapped item's left edge sits. */
const SNAP_PAD = 6;
const GROUPS: DatasetInfo['group'][] = ['Images', 'Points in 2D', 'Points in 3D'];
const IMAGE_LIMITS = [null, 5_000, 1_000, 200];
const POINT_LIMITS = [null, 100, 50, 20];
const POINT_COUNTS = [200, 400, 600, 1_000, 2_000];

/** Training-subset choices for a dataset kind. */
export const limitsFor = (info: DatasetInfo): (number | null)[] => (info.kind === 'image' ? IMAGE_LIMITS : POINT_LIMITS);

/** Number of training points `generate` makes for `count` points and a training share. */
export function pointTrainCount(classes: number, count: number, trainRatio: number): number {
  const n = Math.max(classes * 2, Math.round(count));
  return Math.min(n - 1, Math.max(1, Math.round(n * trainRatio)));
}

/** Full training-set size of a dataset before any "train on" limit. */
function fullTrainCount(info: DatasetInfo): number {
  if (info.kind === 'image') return info.image!.train;
  const c = store.pointsConfig;
  return pointTrainCount(info.classes.length, c.count, c.trainRatio);
}

const SUBS: Record<string, string> = { '₀': '0', '₁': '1', '₂': '2', '₃': '3' };
const SUPS: Record<string, string> = { '²': '2', '³': '3' };

/**
 * A feature label such as "x₁²" or "sin x₂" as text with <sub>/<sup> elements, so indices line up
 * in any font (few fonts carry subscript digits).
 */
export function mathLabel(label: string): DocumentFragment {
  const frag = document.createDocumentFragment();
  let text = '';
  const flush = () => {
    if (text) frag.append(text);
    text = '';
  };
  for (const ch of label) {
    if (SUBS[ch] || SUPS[ch]) {
      flush();
      frag.append(h(SUBS[ch] ? 'sub' : 'sup', null, SUBS[ch] ?? SUPS[ch]));
    } else text += ch;
  }
  flush();
  return frag;
}

/** Screen position (u right, v up) of a 3-D point seen from slightly above and to the side. */
export function project3(x: number, y: number, z: number): [number, number] {
  const az = (35 * Math.PI) / 180;
  const el = (20 * Math.PI) / 180;
  const u = x * Math.cos(az) - y * Math.sin(az);
  const depth = x * Math.sin(az) + y * Math.cos(az);
  return [u, z * Math.cos(el) - depth * Math.sin(el)];
}

/** A dataset name that wraps only at spaces ("Fashion-MNIST" and "CIFAR-10" never split at the hyphen). */
function nameEl(name: string): HTMLElement {
  const el = h('span', { class: 'dsp-name' });
  name.split(' ').forEach((w, i) => {
    if (i) el.append(' ');
    el.append(w.includes('-') ? h('span', { class: 'dsp-word' }, w) : w);
  });
  return el;
}

interface Item {
  info: DatasetInfo;
  btn: HTMLButtonElement;
  canvas: HTMLCanvasElement;
  bar: HTMLElement;
  progress: HTMLElement;
}

export function mountDatasetPicker(): void {
  const root = $('datasets');
  const index = h('div', { class: 'dsp-index', role: 'group', 'aria-label': 'Dataset' });
  const items: Item[] = [];
  /** A few real samples of each image dataset that has been loaded, for its preview. */
  const samples = new Map<DatasetId, Float32Array[]>();
  /** Point previews are generated once (default settings) and redrawn on theme changes. */
  const pointCache = new Map<DatasetId, { coords: Float32Array; labels: Uint8Array; dims: 2 | 3 }>();

  for (const group of GROUPS) {
    const infos = DATASETS.filter((d) => d.group === group);
    const list = h('div', { class: 'dsp-items' });
    for (const info of infos) {
      const canvas = h('canvas', { class: 'dsp-preview', 'aria-hidden': 'true' }) as HTMLCanvasElement;
      const bar = h('span', { style: { width: '0%' } });
      const progress = h('span', { class: 'progress dsp-progress', hidden: true }, bar);
      // The description is read out as the button's description (and shown in the line below).
      const desc = h('span', { id: `dsp-desc-${info.id}`, hidden: true }, info.description);
      const btn = h(
        'button',
        { type: 'button', class: 'dsp-item', 'data-id': info.id, 'aria-pressed': 'false', 'aria-describedby': desc.id, tabindex: '-1' },
        canvas,
        nameEl(info.name),
        progress,
      ) as HTMLButtonElement;
      btn.addEventListener('click', () => choose(info.id));
      btn.addEventListener('mouseenter', () => describe(info));
      btn.addEventListener('mouseleave', () => describe(null));
      btn.addEventListener('focus', () => describe(info));
      btn.addEventListener('blur', () => describe(null));
      items.push({ info, btn, canvas, bar, progress });
      list.append(btn, desc);
    }
    index.append(h('div', { class: 'dsp-group', style: { flexGrow: String(infos.length) } }, h('p', { class: 'dsp-group-label' }, group), list));
  }

  // Arrow keys move along the index (one tab stop for the whole strip); Enter or Space picks.
  index.addEventListener('keydown', (e) => {
    const at = items.findIndex((it) => it.btn === document.activeElement);
    if (at < 0) return;
    let to = at;
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') to = Math.min(items.length - 1, at + 1);
    else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') to = Math.max(0, at - 1);
    else if (e.key === 'Home') to = 0;
    else if (e.key === 'End') to = items.length - 1;
    else return;
    e.preventDefault();
    for (const it of items) it.btn.tabIndex = -1;
    items[to].btn.tabIndex = 0;
    items[to].btn.focus();
  });

  // ── About line and loading state ──
  const about = h('p', { class: 'dsp-about' });
  const status = h('p', { class: 'dsp-status', role: 'status', 'aria-live': 'polite' });
  let shownInfo: DatasetInfo | null = null;
  let error = '';

  // Image descriptions already give the size; the line adds what the reader cannot see.
  const facts = (info: DatasetInfo): string =>
    info.kind === 'image'
      ? `${info.classes.length} classes · ${int(info.image!.train)} training ${noun(info, 2)}`
      : `${info.dims}-D points · ${info.classes.length} classes`;

  function describe(info: DatasetInfo | null): void {
    shownInfo = info;
    const d = info ?? store.info;
    clear(about);
    about.append(h('b', null, d.name), h('span', { class: 'dsp-facts' }, ` · ${facts(d)}. `), d.description);
  }

  // ── Train on ──
  const trainSeg = h('div', { class: 'seg dsp-seg', role: 'group', 'aria-label': 'Train on' });
  const trainUnit = h('span', { class: 'dsp-unit' });
  const trainHint = h('p', { class: 'hint dsp-train-hint' });
  const train = h('div', { class: 'dsp-train' }, h('span', { class: 'label' }, 'Train on'), h('div', { class: 'dsp-train-row' }, trainSeg, trainUnit), trainHint);

  let trainKind = '';
  const trainButtons: { v: number | null; b: HTMLButtonElement }[] = [];
  const renderTrain = () => {
    const info = store.info;
    const full = fullTrainCount(info);
    // Rebuilt only when the choices change (images ↔ points), so a pressed button keeps its focus.
    if (trainKind !== info.kind) {
      trainKind = info.kind;
      clear(trainSeg);
      trainButtons.length = 0;
      for (const v of limitsFor(info)) {
        const b = h('button', { type: 'button', 'aria-pressed': 'false' }, v === null ? 'All' : int(v)) as HTMLButtonElement;
        b.addEventListener('click', () => {
          if (store.trainLimit === v) return;
          setTrainLimit(v);
          renderTrain();
        });
        trainButtons.push({ v, b });
        trainSeg.append(b);
      }
    }
    for (const { v, b } of trainButtons) b.setAttribute('aria-pressed', String(store.trainLimit === v));
    trainUnit.textContent = `of ${int(full)} ${noun(info, full)}`;
    trainHint.textContent =
      info.kind === 'image'
        ? 'Few examples are memorised: training accuracy runs far above test. Transfer learning reuses features learned on more data.'
        : 'Few examples are memorised: training accuracy runs far above test, and the boundary bends around single points.';
  };

  // ── Point-data controls ──
  const points = h('div', { class: 'dsp-points', hidden: true });

  const slider = (id: string, label: string, min: number, max: number, step: number, value: () => number, show: (v: number) => string, commit: (v: number) => void) => {
    const input = h('input', { type: 'range', id, class: 'ds-range', min: String(min), max: String(max), step: String(step) }) as HTMLInputElement;
    const out = h('output', { class: 'dsp-value', for: id });
    let timer: ReturnType<typeof setTimeout> | null = null;
    const sync = () => {
      if (timer) return; // the reader is still dragging
      input.value = String(value());
      out.textContent = show(value());
    };
    input.addEventListener('input', () => {
      const v = Number(input.value);
      out.textContent = show(v);
      if (timer) clearTimeout(timer);
      timer = setTimeout(() => {
        timer = null;
        commit(v);
      }, 250);
    });
    const el = h('div', { class: 'dsp-field' }, h('label', { class: 'label', for: id }, label), h('div', { class: 'dsp-control' }, input, out));
    return { el, sync };
  };

  const noise = slider('dsp-noise', 'Noise', 0, 0.5, 0.05, () => store.pointsConfig.noise, (v) => fixed(v, 2), (v) => setPointsConfig({ noise: v }));
  const share = slider('dsp-share', 'Training share', 0.1, 0.9, 0.1, () => store.pointsConfig.trainRatio, (v) => `${Math.round(v * 100)}%`, (v) => setPointsConfig({ trainRatio: v }));

  const countSeg = h('div', { class: 'seg dsp-seg', role: 'group', 'aria-labelledby': 'dsp-count-label' });
  const countButtons = POINT_COUNTS.map((n) => {
    const b = h('button', { type: 'button', 'aria-pressed': 'false' }, int(n)) as HTMLButtonElement;
    b.addEventListener('click', () => {
      if (store.pointsConfig.count !== n) setPointsConfig({ count: n });
    });
    countSeg.append(b);
    return { n, b };
  });
  const renderCount = () => {
    for (const { n, b } of countButtons) b.setAttribute('aria-pressed', String(store.pointsConfig.count === n));
  };

  const regen = h('button', { type: 'button', class: 'btn btn-sm', id: 'dsp-regenerate', title: 'Draw a new random sample with the same settings' }, 'Regenerate');
  regen.addEventListener('click', () => setPointsConfig({ seed: store.pointsConfig.seed + 1 }));

  const feats = h('div', { class: 'dsp-feats', role: 'group', 'aria-labelledby': 'dsp-feat-label' });
  const featNote = h('p', { class: 'hint dsp-feat-note' });
  const note = (text: string) => {
    clear(featNote);
    featNote.append(mathLabel(text));
  };
  // The Playground's lesson, with an example in the current number of dimensions.
  const playground = () =>
    store.info.dims === 3
      ? 'Hand-made features can make a simple model enough: with x₁², x₂² and x₃², the ball and the shell of Shells are split by a flat plane, so even the Linear preset solves it.'
      : 'Hand-made features can make a simple model enough: with x₁² and x₂², the Circle classes are split by a straight line, so even the Linear preset solves it.';
  let featDims = 0;
  const featButtons: { id: string; title: string; b: HTMLButtonElement }[] = [];
  const renderFeatures = () => {
    const info = store.info;
    if (info.kind !== 'points') return;
    const dims = info.dims!;
    if (featDims !== dims) {
      // Built once per dimension, then updated in place, so a toggled chip keeps its focus.
      featDims = dims;
      clear(feats);
      featButtons.length = 0;
      for (const f of featureCatalog(dims)) {
        const b = h('button', { type: 'button', class: 'ds-feat', 'aria-pressed': 'false', 'aria-label': f.label, 'data-feature': f.id }, mathLabel(f.label)) as HTMLButtonElement;
        b.addEventListener('click', () => {
          const now = new Set(store.features);
          if (now.has(f.id)) {
            if (now.size === 1) {
              note('Keep at least one feature on: the network needs an input.');
              return;
            }
            now.delete(f.id);
          } else now.add(f.id);
          // Keep the catalogue order, so the inputs always appear in the same order.
          setFeatures(featureCatalog(dims).map((d) => d.id).filter((id) => now.has(id)));
          note(playground());
        });
        featButtons.push({ id: f.id, title: f.title, b });
        feats.append(b);
      }
    }
    const on = new Set(store.features);
    for (const { id, title, b } of featButtons) {
      const pressed = on.has(id);
      const last = pressed && on.size === 1;
      b.setAttribute('aria-pressed', String(pressed));
      if (last) b.setAttribute('aria-disabled', 'true');
      else b.removeAttribute('aria-disabled');
      b.title = last ? `${title}. Keep at least one feature on.` : title;
    }
  };

  points.append(
    h(
      'div',
      { class: 'dsp-points-row' },
      noise.el,
      h('div', { class: 'dsp-field' }, h('span', { class: 'label', id: 'dsp-count-label' }, 'Points'), countSeg),
      share.el,
      h('div', { class: 'dsp-field dsp-regen' }, regen),
      h('div', { class: 'dsp-field dsp-feat-field' }, h('span', { class: 'label', id: 'dsp-feat-label' }, 'Input features'), feats),
    ),
    featNote,
  );

  root.append(h('div', { class: 'dsp' }, index, h('div', { class: 'dsp-row' }, h('div', { class: 'dsp-about-col' }, about, status), train), points));

  // ── Previews ──
  const paintPreview = (it: Item) => {
    const ctx = fitCanvas(it.canvas, PREVIEW, PREVIEW);
    const p = palette();
    ctx.fillStyle = p.surface;
    ctx.fillRect(0, 0, PREVIEW, PREVIEW);
    const info = it.info;
    if (info.kind === 'image') {
      const half = PREVIEW / 2;
      const real = samples.get(info.id);
      for (let k = 0; k < 4; k++) {
        const x = (k % 2) * half;
        const y = Math.floor(k / 2) * half;
        if (real?.[k]) drawSample(ctx, real[k], info.image!.shape, x, y, half, half);
        else {
          // Until the dataset is loaded: its first four classes as glyph tiles.
          ctx.fillStyle = classColor(k);
          ctx.font = `600 12px "IBM Plex Mono", ui-monospace, monospace`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(info.glyphs[k], x + half / 2, y + half / 2 + 0.5);
        }
      }
      if (!real) {
        ctx.strokeStyle = p.hair;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(half, 0);
        ctx.lineTo(half, PREVIEW);
        ctx.moveTo(0, half);
        ctx.lineTo(PREVIEW, half);
        ctx.stroke();
      }
      return;
    }
    let pts = pointCache.get(info.id);
    if (!pts) {
      const syn = generate({ id: info.id as SyntheticId, ...DEFAULT_SYNTHETIC, count: 240 });
      const n = syn.train.labels.length + syn.test.labels.length;
      const coords = new Float32Array(n * syn.info.dims);
      coords.set(syn.train.coords);
      coords.set(syn.test.coords, syn.train.coords.length);
      const labels = new Uint8Array(n);
      labels.set(syn.train.labels);
      labels.set(syn.test.labels, syn.train.labels.length);
      pts = { coords, labels, dims: syn.info.dims };
      pointCache.set(info.id, pts);
    }
    const r = 1.3; // half-width of the plotted domain
    const s = (PREVIEW - 4) / (2 * r);
    const c0 = PREVIEW / 2;
    const n = pts.labels.length;
    for (let i = 0; i < n; i++) {
      const o = i * pts.dims;
      const [u, v] = pts.dims === 2 ? [pts.coords[o], pts.coords[o + 1]] : project3(pts.coords[o], pts.coords[o + 1], pts.coords[o + 2]);
      ctx.fillStyle = classColor(pts.labels[i]);
      ctx.fillRect(c0 + u * s - 0.8, c0 - v * s - 0.8, 1.6, 1.6);
    }
  };

  const keepSamples = () => {
    const d = store.data;
    if (!d || d.info.kind !== 'image' || samples.has(d.info.id)) return;
    const out: Float32Array[] = [];
    for (let k = 0; k < 4; k++) {
      const i = d.testY.indexOf(k);
      if (i >= 0) out.push(sampleInput(d, 'test', i));
    }
    samples.set(d.info.id, out);
    const it = items.find((x) => x.info.id === d.info.id);
    if (it) paintPreview(it);
  };

  // ── Sync with the store ──
  let lastId: DatasetId | null = null;

  const syncSelection = () => {
    const id = store.dataset;
    for (const it of items) {
      const on = it.info.id === id;
      it.btn.setAttribute('aria-pressed', String(on));
      if (!items.some((x) => x.btn === document.activeElement)) it.btn.tabIndex = on ? 0 : -1;
      const busy = !!loading && loading.id === it.info.id && on;
      it.progress.hidden = !busy;
      if (busy) it.bar.style.width = `${Math.round((100 * loading!.done) / Math.max(1, loading!.total))}%`;
    }
    const info = store.info;
    if (error) status.textContent = error;
    else if (loading && loading.id === id) {
      status.textContent = loading.done === 0 ? `Loading ${info.name}…` : `Loading ${info.name} · ${loading.done} of ${loading.total} files`;
    } else if (!store.data) status.textContent = `Preparing ${info.name}…`;
    else status.textContent = '';
    status.hidden = !status.textContent;
    if (!shownInfo) describe(null);
  };

  const syncPoints = () => {
    const pts = store.info.kind === 'points';
    points.hidden = !pts;
    if (!pts) return;
    noise.sync();
    share.sync();
    renderCount();
    renderFeatures();
    note(playground());
  };

  const onDataset = () => {
    const id = store.dataset;
    if (id !== lastId) {
      const prev = lastId ? DATASETS.find((d) => d.id === lastId)! : null;
      lastId = id;
      error = '';
      // A subset size only makes sense for its kind of data (5,000 images, 50 points). This runs
      // before the new data is applied, so the change takes effect with it.
      if (prev && prev.kind !== store.info.kind && !limitsFor(store.info).includes(store.trainLimit)) store.trainLimit = null;
      syncPoints();
      renderTrain();
      syncSelection();
      reveal();
      return;
    }
    syncSelection();
  };

  // On a narrow screen the index scrolls sideways: keep the chosen dataset in view (without
  // scrolling the page itself).
  const reveal = () => {
    const btn = items.find((it) => it.info.id === store.dataset)?.btn;
    if (!btn) return;
    const ir = index.getBoundingClientRect();
    const br = btn.getBoundingClientRect();
    // Aligned at its start, which is also where the strip's scroll snapping would put it.
    if (br.left < ir.left || br.right > ir.right) index.scrollLeft += br.left - ir.left - SNAP_PAD;
  };
  let lastWidth = 0;
  if (typeof ResizeObserver !== 'undefined') {
    new ResizeObserver(() => {
      if (index.clientWidth === lastWidth) return;
      lastWidth = index.clientWidth;
      reveal();
    }).observe(index);
  }
  // Leaving the index puts its tab stop back on the chosen dataset.
  index.addEventListener('focusout', (e) => {
    if (index.contains(e.relatedTarget as Node | null)) return;
    for (const it of items) it.btn.tabIndex = it.info.id === store.dataset ? 0 : -1;
  });

  const choose = (id: DatasetId) => {
    for (const it of items) it.btn.tabIndex = it.info.id === id ? 0 : -1;
    if (id === store.dataset && (store.data || loading)) return;
    error = '';
    setDataset(id).catch((err: unknown) => {
      error = `${datasetName(id)} could not be loaded: ${err instanceof Error ? err.message : String(err)}. Pick it again to retry.`;
      syncSelection();
    });
  };
  const datasetName = (id: DatasetId) => DATASETS.find((d) => d.id === id)!.name;

  store.on('dataset', onDataset);
  store.on('data', () => {
    keepSamples();
    syncPoints();
    renderTrain();
    syncSelection();
  });
  onThemeChange(() => items.forEach(paintPreview));
  items.forEach(paintPreview);
  onDataset();
}
