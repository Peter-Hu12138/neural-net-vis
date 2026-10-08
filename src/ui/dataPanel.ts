import './dataPanel.css';
import { addCustom, removeCustom, setProbe, testProbe } from '../actions';
import { fixed } from '../analysis/stats';
import { noun, sampleCaption, sampleInput, type Data, type DatasetId, type DatasetInfo } from '../data/datasets';
import { featureDefs } from '../data/features';
import { PointEvaluator, argmaxRows, pointDomain } from '../data/grid';
import { argmax, type Network } from '../nn/network';
import { size } from '../nn/types';
import { CUSTOM_REPEAT } from '../train/protocol';
import { store, type CustomEntry, type Probe } from '../store';
import { mathLabel } from './datasetPicker';
import { $, clear, digitChips, h, int, selectField } from './dom';
import { fitCanvas, paintSample, sampleCanvas } from './draw';
import { imageInput, privateNetwork, readImageFile } from './drawpad';
import { classColor, onThemeChange, palette } from './theme';
import { hideTip, showTip } from './tip';

/**
 * Section 07, Data. Image datasets: a grid of test images (click one to make it the input),
 * uploads of your own images (converted the way the dataset's own images were made), and the
 * samples you added to the training set. Point datasets: the training and test points as
 * scatter plots, the current input as coordinates and as the features the network actually
 * reads, class counts, and the points you added.
 */

const GRID = 60;

/** True when an image dataset names its classes (Fashion-MNIST, CIFAR-10) rather than numbering them. */
const named = (info: DatasetInfo) => info.classes.some((c, k) => c !== String(k));

/** "7" for MNIST, "Sneaker" for Fashion-MNIST. */
const className = (info: DatasetInfo, k: number) => info.classes[k] ?? String(k);

/** Calls `fn` whenever the element comes into (or near) view, and reports whether it is visible. */
function watchVisibility(el: Element, onShow: () => void): () => boolean {
  let visible = false;
  if (typeof IntersectionObserver === 'undefined') return () => true;
  new IntersectionObserver(
    (entries) => {
      const now = entries.some((e) => e.isIntersecting);
      if (now && !visible) {
        visible = true;
        onShow();
      }
      visible = now;
    },
    { rootMargin: '200px' },
  ).observe(el);
  return () => visible;
}

export function mountDataPanel(): void {
  const root = $('datapanel');
  const images = imagePanel();
  const points = pointPanel();
  root.append(images.el, points.el);
  const sync = () => {
    const pts = store.info.kind === 'points';
    images.el.hidden = pts;
    points.el.hidden = !pts;
  };
  store.on('dataset', sync);
  sync();
}

// ── Images ────────────────────────────────────────────────────────────────

interface Upload {
  id: number;
  dataset: DatasetId;
  name: string;
  url: string;
  x: Float32Array | null;
  label: number | null;
  customId: number | null;
  row?: HTMLElement;
  /** The "Predicts …" line, refreshed with the grid while training runs. */
  pred?: HTMLElement;
}

const DROP: Record<string, { title: string; how: string }> = {
  mnist: {
    title: 'Drop digit images here',
    how: 'or click to choose files. Photos, scans and screenshots all work: each is cropped, scaled to 20 px, centred on a 28×28 grid and inverted when the ink is dark.',
  },
  fashion: {
    title: 'Drop photos of clothing here',
    how: 'or click to choose files. As Fashion-MNIST did with its product photos, each is cropped to the item, scaled to fill 28×28 and made grey, with the backdrop at zero.',
  },
  cifar10: {
    title: 'Drop photos here',
    how: 'or click to choose files. Each photo is cropped to its centre square and averaged down to 32×32 colour pixels, the size of every CIFAR-10 image.',
  },
};

function imagePanel(): { el: HTMLElement } {
  const gridTitle = h('p', { class: 'sub' });
  const grid = h('div', { class: 'sample-grid' });
  const shuffle = h('button', { type: 'button', class: 'btn btn-sm', disabled: true }, 'Shuffle');
  const gridNote = h('p', { class: 'hint', style: { marginTop: '10px' } }, 'Loading…');
  const key = h('dl', { class: 'dp-key', hidden: true });
  const fileInput = h('input', { type: 'file', accept: 'image/*', multiple: true, id: 'upload-input', class: 'dp-file' }) as HTMLInputElement;
  const dropTitle = h('b');
  const dropHow = h('span', { class: 'hint' });
  const drop = h('label', { class: 'drop', for: 'upload-input' }, dropTitle, dropHow);
  const list = h('ul', { class: 'uploads' });
  const customGrid = h('div', { class: 'sample-grid', style: { marginTop: '10px' } });
  const customNote = h('p', { class: 'custom-summary' });
  const yours = h('p', { class: 'sub' }, 'Your images');

  const el = h(
    'div',
    { class: 'data-cols' },
    h('div', null, h('div', { class: 'chart-head' }, gridTitle, shuffle), grid, gridNote, key),
    h('div', null, yours, drop, fileInput, list, h('p', { class: 'sub', style: { marginTop: '22px' } }, 'In the training set'), customGrid, customNote),
  );

  const isImages = () => store.info.kind === 'image';
  const visible = watchVisibility(el, () => {
    lastPredict = 0;
    predict();
  });

  // ── Test grid ──
  let indices: number[] = [];
  const cells: { i: number; btn: HTMLButtonElement; tag: HTMLElement; caption: string }[] = [];
  const pick = () => {
    const d = store.data;
    if (!d || d.info.kind !== 'image') return;
    const n = d.testY.length;
    const set = new Set<number>();
    while (set.size < Math.min(GRID, n)) set.add(Math.floor(Math.random() * n));
    indices = [...set];
    buildGrid();
  };
  const buildGrid = () => {
    clear(grid);
    cells.length = 0;
    const d = store.data;
    if (!d || d.info.kind !== 'image') return;
    for (const i of indices) {
      const x = sampleInput(d, 'test', i);
      const y = d.testY[i];
      const caption = sampleCaption(d.info, 'test', i, y);
      const tag = h('span', { class: 'thumb-label' }, d.info.glyphs[y]);
      const btn = h('button', { type: 'button', class: 'thumb', title: caption, 'aria-label': caption, 'aria-pressed': 'false' }, sampleCanvas(x, d.input, 40), tag) as HTMLButtonElement;
      btn.addEventListener('click', () => setProbe(testProbe(d, i)));
      cells.push({ i, btn, tag, caption });
      grid.append(btn);
    }
    syncPressed();
    gridTitle.textContent = `Test ${noun(d.info, 2)} · ${int(cells.length)} of ${int(d.testY.length)}`;
    gridNote.textContent = `Click any ${noun(d.info)} to feed it through the network.`;
    lastPredict = 0;
    predict();
  };
  const syncPressed = () => {
    for (const c of cells) c.btn.setAttribute('aria-pressed', String(store.probe?.key === `test:${c.i}`));
  };
  shuffle.addEventListener('click', pick);

  // Re-classifying the grid costs about a millisecond per image, so it runs only on screen, in
  // slices of a few milliseconds (the page stays responsive while training), on a private copy of
  // the network (the page's network keeps the current input's activations for the other views),
  // and at most once per ten times as long as it took.
  const syncShadow = privateNetwork();
  let lastPredict = 0;
  let interval = 1200;
  let pending: ReturnType<typeof setTimeout> | null = null;
  let pass = 0; // bumped to abandon a pass that is under way
  const predict = () => {
    const d = store.data;
    if (!d || d.info.kind !== 'image' || !cells.length || !visible()) return;
    if (store.net.inputSize !== d.inputSize) return;
    const now = performance.now();
    if (now - lastPredict < interval) {
      if (!pending) pending = setTimeout(() => ((pending = null), predict()), interval - (now - lastPredict));
      return;
    }
    lastPredict = now;
    const net = syncShadow();
    const info = d.info;
    const todo = cells.slice();
    const id = ++pass;
    const x = new Float32Array(d.inputSize);
    let k = 0;
    let wrong = 0;
    let busy = 0;
    const slice = () => {
      if (id !== pass || store.data !== d) return;
      const t0 = performance.now();
      for (; k < todo.length && performance.now() - t0 < 6; k++) {
        const c = todo[k];
        const y = d.testY[c.i];
        const p = argmax(net.forward(sampleInput(d, 'test', c.i, x)));
        const ok = p === y;
        if (!ok) wrong++;
        c.tag.textContent = ok ? info.glyphs[y] : `${info.glyphs[y]}→${info.glyphs[p]}`;
        c.tag.classList.toggle('is-wrong', !ok);
        c.btn.classList.toggle('is-wrong', !ok);
        c.btn.title = ok ? c.caption : `${c.caption} · predicted ${className(info, p)}`;
        c.btn.setAttribute('aria-label', c.btn.title);
      }
      busy += performance.now() - t0;
      if (k < todo.length) {
        setTimeout(slice, 0);
        return;
      }
      predictUploads(net);
      interval = Math.max(1200, 10 * busy);
      gridNote.textContent = `${todo.length - wrong} of ${todo.length} classified correctly by the current weights. Mistakes are marked true→predicted in red. Click any ${noun(info)} to feed it through the network.`;
    };
    slice();
  };

  const renderKey = () => {
    const info = store.info;
    clear(key);
    key.hidden = !named(info);
    if (key.hidden) return;
    info.classes.forEach((name, k) => key.append(h('div', null, h('dt', { style: { color: classColor(k) } }, info.glyphs[k]), h('dd', null, name))));
  };

  // ── Uploads (kept per dataset; each dataset prepares images its own way) ──
  const uploads = new Map<DatasetId, Upload[]>();
  let nextId = 1;
  const current = () => uploads.get(store.dataset) ?? [];

  const classPicker = (u: Upload): HTMLElement => {
    const info = store.info;
    const pickLabel = (k: number) => {
      u.label = k;
      if (u.customId !== null) {
        removeCustom(u.customId);
        u.customId = addCustom({ x: u.x!, y: k, origin: 'upload', name: u.name });
      }
      renderUpload(u);
    };
    if (!named(info)) return digitChips(u.label, pickLabel, `True label for ${u.name}`, info.classes.length);
    const options = [{ value: -1, label: 'Choose…' }, ...info.classes.map((c, k) => ({ value: k, label: c }))];
    return selectField(`upload-class-${u.id}`, 'True class', options, u.label ?? -1, (k) => {
      if (k >= 0) pickLabel(k);
    });
  };

  const renderUpload = (u: Upload): HTMLElement => {
    const row = u.row ?? h('li', { class: 'upload' });
    u.row = row;
    clear(row);
    const info = store.info;
    const shape = store.input;
    const processed = document.createElement('canvas');
    processed.setAttribute('role', 'img');
    processed.setAttribute('aria-label', `${u.name} as the network sees it`);
    paintSample(processed, u.x ?? new Float32Array(size(shape)), shape, 56);
    const meta = h('div', { class: 'upload-meta' }, h('span', { class: 'upload-name', title: u.name }, u.name));
    if (!u.x) {
      meta.append(h('span', { class: 'layer-error' }, info.id === 'mnist' ? 'No ink found. Try a higher-contrast image.' : 'Nothing stands out from the background. Try a photo with a plain backdrop.'));
    } else {
      u.pred = h('span', { class: 'upload-pred' });
      showUploadPrediction(u, store.net.inputSize === u.x.length ? syncShadow() : null);
      meta.append(u.pred, classPicker(u));
    }
    const buttons = h('div', { class: 'dp-upload-actions' });
    if (u.x) {
      buttons.append(h('button', { type: 'button', class: 'btn btn-sm', onclick: () => setProbe({ x: u.x!, label: u.label, caption: `Upload · ${u.name}`, key: `upload:${u.id}` }) }, 'Use as input'));
      if (u.customId === null) {
        buttons.append(
          h('button', {
            type: 'button',
            class: 'btn btn-sm btn-solid',
            disabled: u.label === null,
            title: u.label === null ? `Pick the true ${named(info) ? 'class' : 'digit'} first` : 'Add to the training set',
            onclick: () => {
              u.customId = addCustom({ x: u.x!, y: u.label!, origin: 'upload', name: u.name });
              renderUpload(u);
            },
          }, 'Train on it'),
        );
      } else {
        buttons.append(h('span', { class: 'tag is-on' }, 'In training set'));
      }
    }
    buttons.append(
      h('button', {
        type: 'button',
        class: 'btn btn-sm',
        onclick: () => {
          if (u.customId !== null) removeCustom(u.customId);
          URL.revokeObjectURL(u.url);
          const all = uploads.get(u.dataset) ?? [];
          all.splice(all.indexOf(u), 1);
          row.remove();
        },
      }, 'Delete'),
    );
    row.append(h('img', { src: u.url, alt: `Uploaded image ${u.name}` }), processed, meta, buttons);
    return row;
  };

  /** Fills an upload's "Predicts …" line using `net` (null: the network does not fit this input). */
  function showUploadPrediction(u: Upload, net: Network | null): void {
    if (!u.pred || !u.x) return;
    const info = store.info;
    clear(u.pred);
    if (!net || net.inputSize !== u.x.length) {
      u.pred.className = 'hint';
      u.pred.textContent = 'Waiting for the network…';
      return;
    }
    const probs = net.forward(u.x);
    const best = argmax(probs);
    u.pred.className = 'upload-pred';
    u.pred.append('Predicts ', h('b', null, className(info, best)), ` · ${fixed(probs[best] * 100, 1)}%`);
  }
  function predictUploads(net: Network): void {
    for (const u of current()) showUploadPrediction(u, net);
  }

  const addFiles = async (files: FileList | File[]) => {
    const info = store.info;
    if (info.kind !== 'image') return;
    for (const f of Array.from(files)) {
      if (!f.type.startsWith('image/')) continue;
      try {
        const { img, url } = await readImageFile(f);
        if (store.dataset !== info.id) {
          URL.revokeObjectURL(url);
          return; // the reader switched dataset meanwhile
        }
        const { x } = imageInput(info, img);
        const u: Upload = { id: nextId++, dataset: info.id, name: f.name, url, x, label: null, customId: null };
        if (!uploads.has(info.id)) uploads.set(info.id, []);
        uploads.get(info.id)!.push(u);
        list.prepend(renderUpload(u));
        if (x) setProbe({ x, label: null, caption: `Upload · ${f.name}`, key: `upload:${u.id}` });
      } catch {
        list.prepend(h('li', { class: 'upload' }, h('span'), h('span'), h('span', { class: 'layer-error' }, `${f.name} could not be read as an image.`), h('span')));
      }
    }
  };
  fileInput.addEventListener('change', () => {
    if (fileInput.files) addFiles(fileInput.files);
    fileInput.value = '';
  });
  drop.addEventListener('dragover', (e) => {
    e.preventDefault();
    drop.classList.add('is-over');
  });
  drop.addEventListener('dragleave', () => drop.classList.remove('is-over'));
  drop.addEventListener('drop', (e) => {
    e.preventDefault();
    drop.classList.remove('is-over');
    if (e.dataTransfer?.files) addFiles(e.dataTransfer.files);
  });
  // Pasting an image anywhere on the page also works (unless section 03 took it as a photo).
  window.addEventListener('paste', (e) => {
    if (e.defaultPrevented || !isImages()) return;
    const files = Array.from(e.clipboardData?.files ?? []).filter((f) => f.type.startsWith('image/'));
    if (files.length) addFiles(files);
  });

  const renderUploads = () => {
    clear(list);
    for (const u of current().slice().reverse()) {
      // Custom samples do not survive a dataset switch; the uploads do.
      if (u.customId !== null && !store.custom.some((c) => c.id === u.customId)) u.customId = null;
      list.append(renderUpload(u));
    }
  };

  const renderCustom = () => {
    clear(customGrid);
    const info = store.info;
    if (info.kind !== 'image') return;
    const shape = store.input;
    const n0 = size(shape);
    for (const c of store.custom) {
      if (c.x.length !== n0) continue;
      const rm = h(
        'button',
        { type: 'button', class: 'thumb', title: `${c.name} · ${named(info) ? className(info, c.y) : `label ${c.y}`}. Click to remove it from the training set.` },
        sampleCanvas(c.x, shape, 40),
        h('span', { class: 'thumb-label' }, info.glyphs[c.y] ?? String(c.y)),
      ) as HTMLButtonElement;
      rm.addEventListener('click', () => {
        removeCustom(c.id);
        const u = current().find((q) => q.customId === c.id);
        if (u) {
          u.customId = null;
          renderUpload(u);
        }
      });
      customGrid.append(rm);
    }
    const n = store.custom.length;
    const draws = info.id !== 'cifar10';
    customNote.textContent =
      n === 0
        ? `Nothing yet. Label an upload${draws ? ' or a drawing' : ''} to add it.`
        : n === 1
          ? `Your image is mixed into training and seen ${CUSTOM_REPEAT}× per epoch. Click it to remove it.`
          : `Your ${n} images are mixed into training, each seen ${CUSTOM_REPEAT}× per epoch. Click one to remove it.`;
  };

  const onDataset = () => {
    const info = store.info;
    if (info.kind !== 'image') return;
    const d = DROP[info.id] ?? DROP.cifar10;
    dropTitle.textContent = d.title;
    dropHow.textContent = d.how;
    gridTitle.textContent = `Test ${noun(info, 2)}`;
    renderKey();
    if (!store.data) {
      clear(grid);
      cells.length = 0;
      shuffle.disabled = true;
      gridNote.textContent = `Loading ${noun(info, 2)}…`;
    }
  };

  let shownDataset: DatasetId | null = null;
  store.on('dataset', () => {
    if (!isImages()) {
      shownDataset = null; // re-render the uploads when an image dataset comes back
      return;
    }
    onDataset();
    if (store.dataset !== shownDataset) {
      shownDataset = store.dataset;
      renderUploads();
      renderCustom();
    }
  });
  store.on('data', () => {
    if (!isImages()) return;
    shuffle.disabled = false;
    pick();
  });
  store.on('weights', predict);
  store.on('model', () => {
    if (!isImages()) return;
    lastPredict = 0;
    predict();
  });
  store.on('probe', syncPressed);
  store.on('custom', renderCustom);
  onThemeChange(() => {
    if (!isImages()) return;
    if (store.data) buildGrid();
    renderKey();
    current().forEach(renderUpload);
    renderCustom();
  });
  onDataset();
  renderCustom();
  return { el };
}

// ── Points ────────────────────────────────────────────────────────────────

const SUB = ['₁', '₂', '₃'];

interface Panel {
  split: 'train' | 'test';
  /** Coordinates plotted across and up (3-D data shows x₁–x₂ and x₁–x₃). */
  axes: [number, number];
  canvas: HTMLCanvasElement;
  title: HTMLElement;
  stat: HTMLElement;
  size: number;
}

function pointPanel(): { el: HTMLElement } {
  const plots = h('div', { class: 'dp-plots' });
  const plotNote = h(
    'p',
    { class: 'hint dp-plot-note' },
    'Click a point, or focus a plot and use the arrow keys, to make it the network’s input. ',
    h('span', { class: 'dp-ring', 'aria-hidden': 'true' }),
    ' Red rings mark points the current weights get wrong.',
  );
  const readout = h('div', { class: 'dp-readout', 'aria-live': 'polite' });
  const counts = h('table', { class: 'dp-counts' });
  const customList = h('ul', { class: 'dp-custom' });
  const customNote = h('p', { class: 'hint' });

  const el = h(
    'div',
    { class: 'data-cols' },
    h('div', null, h('p', { class: 'sub' }, 'Training and test points'), plots, plotNote),
    h(
      'div',
      { class: 'dp-side' },
      h('div', null, h('p', { class: 'sub' }, 'Current input'), readout),
      h('div', null, h('p', { class: 'sub' }, 'Points per class'), counts),
      h('div', null, h('p', { class: 'sub' }, 'Your points'), customList, customNote),
      h('p', { class: 'hint' }, 'Image uploads apply to the image datasets; here you add points by clicking the decision boundary in 03.'),
    ),
  );

  const isPoints = () => store.info.kind === 'points';
  let panels: Panel[] = [];
  const evaluator = new PointEvaluator();
  /** Predicted class of every training and test point (null until evaluated). */
  let pred: { train: Uint8Array; test: Uint8Array } | null = null;
  let predStamp = '';

  const visible = watchVisibility(el, () => schedule(true));

  // ── Layout ──
  const build = () => {
    clear(plots);
    panels = [];
    const d = store.data;
    if (!d?.points) return;
    const dims = d.points.dims;
    const views: [number, number][] = dims === 2 ? [[0, 1]] : [[0, 1], [0, 2]];
    plots.classList.toggle('is-3d', dims === 3);
    for (const split of ['train', 'test'] as const) {
      for (const axes of views) {
        const canvas = h('canvas', { tabindex: '0', class: 'dp-plot' }) as HTMLCanvasElement;
        const title = h('p', { class: 'dp-plot-title' });
        const stat = h('p', { class: 'dp-plot-stat' });
        const panel: Panel = { split, axes, canvas, title, stat, size: 0 };
        const splitName = split === 'train' ? 'Training' : 'Test';
        title.append(h('b', null, splitName), ' · ', mathLabel(`x${SUB[axes[0]]}–x${SUB[axes[1]]}`));
        canvas.setAttribute(
          'aria-label',
          `${splitName} points, x${axes[0] + 1} across, x${axes[1] + 1} up, coloured by class. Click a point, or use the arrow keys, to make it the network's input.`,
        );
        canvas.setAttribute('role', 'img');
        attach(panel);
        panels.push(panel);
        plots.append(h('div', { class: 'dp-panel' }, title, canvas, stat));
      }
    }
    resize();
  };

  const resize = () => {
    if (!panels.length) return;
    const w = plots.clientWidth;
    const s = Math.max(120, Math.min(300, Math.floor((w - 16) / 2)));
    for (const p of panels) p.size = s;
    draw();
  };
  if (typeof ResizeObserver !== 'undefined') new ResizeObserver(() => resize()).observe(plots);

  // ── Geometry ──
  const PAD = 6;
  const toScreen = (p: Panel, r: number, c: ArrayLike<number>, o: number): [number, number] => {
    const u = c[o + p.axes[0]];
    const v = c[o + p.axes[1]];
    const s = (p.size - 2 * PAD) / (2 * r);
    return [PAD + (u + r) * s, PAD + (r - v) * s];
  };

  const coordsOf = (d: Data, split: 'train' | 'test') => (split === 'train' ? d.points!.trainCoords : d.points!.testCoords);
  const labelsOf = (d: Data, split: 'train' | 'test') => (split === 'train' ? d.trainY : d.testY);

  const nearest = (p: Panel, mx: number, my: number, maxDist: number): number => {
    const d = store.data;
    if (!d?.points) return -1;
    const r = pointDomain(d);
    const c = coordsOf(d, p.split);
    const dims = d.points.dims;
    let best = -1;
    let bestD = maxDist * maxDist;
    for (let i = 0; i < c.length / dims; i++) {
      const [x, y] = toScreen(p, r, c, i * dims);
      const dd = (x - mx) ** 2 + (y - my) ** 2;
      if (dd < bestD) {
        bestD = dd;
        best = i;
      }
    }
    return best;
  };

  const probeFor = (d: Data, split: 'train' | 'test', i: number): Probe => {
    if (split === 'test') return testProbe(d, i);
    const dims = d.points!.dims;
    return {
      x: sampleInput(d, 'train', i),
      label: d.trainY[i],
      caption: sampleCaption(d.info, 'train', i, d.trainY[i]),
      key: `train:${i}`,
      coords: d.points!.trainCoords.slice(i * dims, (i + 1) * dims),
    };
  };

  const attach = (p: Panel) => {
    const local = (e: MouseEvent) => {
      const r = p.canvas.getBoundingClientRect();
      return [e.clientX - r.left, e.clientY - r.top] as const;
    };
    p.canvas.addEventListener('mousemove', (e) => {
      const d = store.data;
      const i = nearest(p, ...local(e), 10);
      p.canvas.style.cursor = i >= 0 ? 'pointer' : 'default';
      if (i < 0 || !d?.points) return hideTip();
      const y = labelsOf(d, p.split)[i];
      const dims = d.points.dims;
      const c = coordsOf(d, p.split);
      const at = Array.from(c.slice(i * dims, (i + 1) * dims), (v) => fixed(v, 2)).join(', ');
      const pr = pred?.[p.split][i];
      const verdict = pr === undefined ? '' : pr === y ? '\npredicted correctly' : `\npredicted ${className(d.info, pr)}`;
      showTip(`${sampleCaption(d.info, p.split, i, y)}\n(${at})${verdict}`, e.clientX, e.clientY);
    });
    p.canvas.addEventListener('mouseleave', hideTip);
    p.canvas.addEventListener('click', (e) => {
      const d = store.data;
      const i = nearest(p, ...local(e), 12);
      if (d?.points && i >= 0) setProbe(probeFor(d, p.split, i));
    });
    p.canvas.addEventListener('keydown', (e) => {
      const d = store.data;
      if (!d?.points) return;
      const n = labelsOf(d, p.split).length;
      const step = e.key === 'ArrowRight' || e.key === 'ArrowDown' ? 1 : e.key === 'ArrowLeft' || e.key === 'ArrowUp' ? -1 : 0;
      if (!step || !n) return;
      e.preventDefault();
      const m = store.probe?.key.match(/^(train|test):(\d+)$/);
      const at = m && m[1] === p.split ? Number(m[2]) : step > 0 ? -1 : 0;
      setProbe(probeFor(d, p.split, (at + step + n) % n));
    });
  };

  // ── Drawing ──
  const tickLabel = (ctx: CanvasRenderingContext2D, i: number, x: number, y: number, align: CanvasTextAlign) => {
    ctx.textAlign = align;
    ctx.font = '500 11px "IBM Plex Mono", ui-monospace, monospace';
    ctx.fillText('x', x, y);
    const w = ctx.measureText('x').width;
    ctx.font = '500 8px "IBM Plex Mono", ui-monospace, monospace';
    ctx.textAlign = 'left';
    ctx.fillText(String(i + 1), align === 'right' ? x + 0.5 : x + w + 0.5, y + 3);
  };

  const draw = () => {
    const d = store.data;
    if (!d?.points || !panels.length) return;
    const pal = palette();
    const r = pointDomain(d);
    const dims = d.points.dims;
    const probe = store.probe;
    for (const p of panels) {
      const S = p.size;
      const ctx = fitCanvas(p.canvas, S, S);
      ctx.fillStyle = pal.surface;
      ctx.fillRect(0, 0, S, S);
      // Axes through the origin and the axis names.
      const o = toScreen(p, r, dims === 2 ? [0, 0] : [0, 0, 0], 0);
      ctx.strokeStyle = pal.hair;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(PAD, Math.round(o[1]) + 0.5);
      ctx.lineTo(S - PAD, Math.round(o[1]) + 0.5);
      ctx.moveTo(Math.round(o[0]) + 0.5, PAD);
      ctx.lineTo(Math.round(o[0]) + 0.5, S - PAD);
      ctx.stroke();
      ctx.strokeRect(0.5, 0.5, S - 1, S - 1);
      ctx.fillStyle = pal.muted;
      ctx.textBaseline = 'alphabetic';
      tickLabel(ctx, p.axes[0], S - PAD - 12, Math.round(o[1]) - 5, 'left');
      tickLabel(ctx, p.axes[1], Math.round(o[0]) + 5, PAD + 10, 'left');
      const c = coordsOf(d, p.split);
      const y = labelsOf(d, p.split);
      const n = y.length;
      const wrong = pred?.[p.split];
      const side = n <= 400 && S >= 200 ? 4 : 3;
      const ring = side + 0.5;
      for (let i = 0; i < n; i++) {
        const [x, yy] = toScreen(p, r, c, i * dims);
        ctx.fillStyle = classColor(y[i]);
        ctx.fillRect(x - side / 2, yy - side / 2, side, side);
      }
      // Mistakes: a red ring around every misclassified point.
      if (wrong) {
        // Slightly see-through, so a crowd of mistakes early in training does not hide the classes.
        ctx.save();
        ctx.globalAlpha = 0.75;
        ctx.strokeStyle = pal.accent;
        ctx.lineWidth = 1;
        ctx.beginPath();
        let k = 0;
        for (let i = 0; i < n; i++) {
          if (wrong[i] === y[i]) continue;
          k++;
          const [x, yy] = toScreen(p, r, c, i * dims);
          ctx.moveTo(x + ring, yy);
          ctx.arc(x, yy, ring, 0, 2 * Math.PI);
        }
        ctx.stroke();
        ctx.restore();
        p.stat.textContent = `${int(n)} points · ${int(k)} wrong · ${fixed((100 * (n - k)) / Math.max(1, n), 1)}% right`;
      } else p.stat.textContent = `${int(n)} points`;
      // Your points are training data: larger squares with an ink outline.
      if (p.split === 'train') {
        for (const cu of store.custom) {
          if (!cu.coords || cu.coords.length !== dims) continue;
          const [x, yy] = toScreen(p, r, cu.coords, 0);
          ctx.fillStyle = classColor(cu.y);
          ctx.fillRect(x - 4, yy - 4, 8, 8);
          ctx.strokeStyle = pal.ink;
          ctx.lineWidth = 1.5;
          ctx.strokeRect(x - 4, yy - 4, 8, 8);
        }
      }
      // The current input: a red crosshair, wherever it came from.
      if (probe?.coords && probe.coords.length === dims) {
        const [x, yy] = toScreen(p, r, probe.coords, 0);
        ctx.strokeStyle = pal.accent;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(x, yy, 7, 0, 2 * Math.PI);
        ctx.moveTo(x - 12, yy);
        ctx.lineTo(x - 8, yy);
        ctx.moveTo(x + 8, yy);
        ctx.lineTo(x + 12, yy);
        ctx.moveTo(x, yy - 12);
        ctx.lineTo(x, yy - 8);
        ctx.moveTo(x, yy + 8);
        ctx.lineTo(x, yy + 12);
        ctx.stroke();
      }
    }
  };

  // ── Predictions (a private copy of the network; the page's network is left alone) ──
  const evaluate = () => {
    const d = store.data;
    if (!d?.points) return;
    const feats = d.points.features;
    if (store.net.inputSize !== feats.length || store.net.classes !== d.info.classes.length) return;
    const stamp = `${store.version}:${store.weightsRev}:${d.trainY.length}:${d.testY.length}`;
    if (stamp === predStamp) return;
    predStamp = stamp;
    evaluator.sync(store.net);
    const K = d.info.classes.length;
    const train = argmaxRows(evaluator.evaluate(d.points.trainCoords, d.points.dims, feats).probs, K);
    const test = argmaxRows(evaluator.evaluate(d.points.testCoords, d.points.dims, feats).probs, K);
    pred = { train, test };
  };

  // Redraws are coalesced to one per frame. The points do not move while training, only the
  // mistake rings change, so changed weights are re-evaluated and redrawn at most 4× a second.
  let frame = 0;
  let lastEval = 0;
  let evalTimer: ReturnType<typeof setTimeout> | null = null;
  function schedule(withEval = false): void {
    if (!isPoints() || !visible()) return;
    if (withEval) {
      const wait = 250 - (performance.now() - lastEval);
      if (wait > 0) {
        if (!evalTimer) evalTimer = setTimeout(() => ((evalTimer = null), schedule(true)), wait);
        return;
      }
      lastEval = performance.now();
      evaluate();
    }
    if (frame) return;
    frame = requestAnimationFrame(() => {
      frame = 0;
      draw();
    });
  }

  // ── Side column ──
  const renderReadout = () => {
    clear(readout);
    const d = store.data;
    const probe = store.probe;
    if (!d?.points || !probe) {
      readout.append(h('p', { class: 'hint' }, 'No input chosen.'));
      return;
    }
    const dims = d.points.dims;
    readout.append(h('p', { class: 'dp-caption' }, probe.caption));
    const coords = probe.coords;
    const table = h('table', { class: 'dp-vec' });
    const head = (text: string) => table.append(h('tr', null, h('th', { colspan: '3', scope: 'colgroup' }, text)));
    const row = (label: string, v: number, bar: boolean) => {
      const span = h('span', { class: 'dp-bar' });
      if (bar) {
        const w = Math.min(1, Math.abs(v)) * 50;
        span.append(h('i', { class: v >= 0 ? 'is-pos' : 'is-neg', style: { left: v >= 0 ? '50%' : `${50 - w}%`, width: `${w}%` } }));
      }
      table.append(h('tr', null, h('th', { scope: 'row' }, mathLabel(label)), h('td', { class: 'num' }, fixed(v, 2)), h('td', null, bar ? span : '')));
    };
    head('Point (raw coordinates)');
    if (coords && coords.length === dims) for (let k = 0; k < dims; k++) row(`x${SUB[k]}`, coords[k], false);
    else table.append(h('tr', null, h('td', { colspan: '3', class: 'hint' }, 'Not a point of this dataset.')));
    head('Network input (features)');
    const feats = d.points.features;
    if (probe.x.length === feats.length) featureDefs(dims, feats).forEach((f, k) => row(f.label, probe.x[k], true));
    readout.append(
      table,
      h('p', { class: 'hint' }, 'The network never sees the point itself, only the features computed from it. Toggle features in the strip at the top to change what it gets.'),
    );
  };

  const renderCounts = () => {
    clear(counts);
    const d = store.data;
    if (!d?.points) return;
    const K = d.info.classes.length;
    const tally = (y: Uint8Array) => {
      const n = new Array<number>(K).fill(0);
      for (const v of y) n[v]++;
      return n;
    };
    const tr = tally(d.trainY);
    const te = tally(d.testY);
    counts.append(
      h('thead', null, h('tr', null, h('th', { scope: 'col' }, 'Class'), h('th', { scope: 'col', class: 'num' }, 'Training'), h('th', { scope: 'col', class: 'num' }, 'Test'))),
    );
    const body = h('tbody');
    for (let k = 0; k < K; k++) {
      body.append(
        h(
          'tr',
          null,
          h('th', { scope: 'row' }, h('span', { class: 'dp-swatch', style: { background: classColor(k) } }), d.info.classes[k]),
          h('td', { class: 'num' }, int(tr[k])),
          h('td', { class: 'num' }, int(te[k])),
        ),
      );
    }
    body.append(h('tr', { class: 'dp-total' }, h('th', { scope: 'row' }, 'All'), h('td', { class: 'num' }, int(d.trainY.length)), h('td', { class: 'num' }, int(d.testY.length))));
    counts.append(body);
  };

  const renderCustom = () => {
    clear(customList);
    const d = store.data;
    const mine = store.custom.filter((c): c is CustomEntry & { coords: Float32Array } => !!c.coords);
    if (!d?.points) return;
    for (const c of mine) {
      const at = Array.from(c.coords, (v) => fixed(v, 2)).join(', ');
      const use = h('button', { type: 'button', class: 'btn btn-sm' }, 'Use as input');
      use.addEventListener('click', () => setProbe({ x: c.x, label: c.y, caption: `Your point · ${d.info.classes[c.y]}`, key: `custom:${c.id}`, coords: c.coords }));
      const rm = h('button', { type: 'button', class: 'btn btn-sm', 'aria-label': `Remove your point at (${at})` }, 'Remove');
      rm.addEventListener('click', () => removeCustom(c.id));
      customList.append(
        h(
          'li',
          null,
          h('span', { class: 'dp-swatch', style: { background: classColor(c.y) } }),
          h('span', { class: 'dp-custom-text' }, h('b', null, d.info.classes[c.y]), ' ', h('span', { class: 'mono' }, `(${at})`)),
          use,
          rm,
        ),
      );
    }
    customNote.textContent = mine.length
      ? `${mine.length === 1 ? 'Your point is' : `Your ${mine.length} points are`} mixed into training, each seen ${CUSTOM_REPEAT}× per epoch.`
      : 'None yet. Click the decision boundary in 03 to add points of your own.';
  };

  store.on('data', () => {
    if (!isPoints()) return;
    pred = null;
    predStamp = '';
    build();
    renderCounts();
    renderCustom();
    renderReadout();
    schedule(true);
  });
  store.on('weights', () => schedule(true));
  store.on('model', () => schedule(true));
  store.on('probe', () => {
    if (!isPoints()) return;
    renderReadout();
    schedule();
  });
  store.on('custom', () => {
    if (!isPoints()) return;
    renderCustom();
    schedule();
  });
  onThemeChange(() => {
    if (!isPoints()) return;
    renderCounts();
    renderCustom();
    schedule();
  });
  return { el };
}
