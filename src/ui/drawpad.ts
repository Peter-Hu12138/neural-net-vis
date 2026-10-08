import './datasetPicker.css';
import './drawpad.css';
import { addCustom, setProbe, testProbe } from '../actions';
import { fixed } from '../analysis/stats';
import type { DatasetInfo } from '../data/datasets';
import {
  adjustImage,
  centreSquare,
  inkToFashion,
  isAdjusted,
  NO_ADJUSTMENT,
  rgbaToMnist,
  toColour,
  toFashion,
  toMnist,
  type Adjustment,
  type Box,
} from '../data/preprocess';
import { Network, argmax } from '../nn/network';
import { size } from '../nn/types';
import { store } from '../store';
import { $, clear, digitChips, h, segmented, selectField } from './dom';
import { fitCanvas, paintSample, paintThumb } from './draw';
import { onThemeChange, palette } from './theme';

/**
 * Section 03 for image datasets: try the network on your own input.
 * - Draw (MNIST, Fashion-MNIST): a drawing pad classified live; label a drawing to train on it.
 * - Photo (CIFAR-10, Fashion-MNIST): drop, paste or choose a photo, or start from a test image; see
 *   the photo, the exact input the network gets, and the class probabilities, and nudge the input
 *   (flip, brightness, contrast) to watch the prediction move.
 * Point datasets use #boundary instead (main.ts hides #drawpad).
 */

const SIZE = 280; // drawing coordinates
const SCALE = 2; // backing pixels per drawing unit
const VIEW = 150; // CSS px of the photo and input previews

export const drawState = { x: null as Float32Array | null };

// ── Image files → network inputs (shared with section 07) ──

/** Decodes an image file. The object URL stays valid until it is revoked. */
export async function readImageFile(file: File): Promise<{ img: HTMLImageElement; url: string }> {
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.src = url;
  try {
    await img.decode();
  } catch (e) {
    URL.revokeObjectURL(url);
    throw e;
  }
  return { img, url };
}

/** RGBA pixels of a region of an image, scaled down so its longest side is at most `max` px. */
function imagePixels(img: HTMLImageElement, region: Box, max: number): ImageData {
  const rw = region.x1 - region.x0;
  const rh = region.y1 - region.y0;
  const s = Math.min(1, max / Math.max(rw, rh));
  const w = Math.max(1, Math.round(rw * s));
  const hh = Math.max(1, Math.round(rh * s));
  const c = document.createElement('canvas');
  c.width = w;
  c.height = hh;
  const ctx = c.getContext('2d', { willReadFrequently: true })!;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(img, region.x0, region.y0, rw, rh, 0, 0, w, hh);
  return ctx.getImageData(0, 0, w, hh);
}

/**
 * A photo as network input for an image dataset, prepared the way that dataset's own images were:
 * MNIST-style ink, Fashion-MNIST framing, or a CIFAR-10 centre crop. `box` is the part of the photo
 * that was used (source pixels), when there is one to show.
 */
export function imageInput(info: DatasetInfo, img: HTMLImageElement): { x: Float32Array | null; box: Box | null } {
  const W = img.naturalWidth;
  const H = img.naturalHeight;
  const shape = info.image!.shape;
  if (shape.c === 3) {
    const box = centreSquare(W, H);
    return { x: toColour(imagePixels(img, box, 4 * shape.h), shape.h), box };
  }
  const data = imagePixels(img, { x0: 0, y0: 0, x1: W, y1: H }, 320);
  if (info.id === 'fashion') {
    const r = toFashion(data);
    const s = W / data.width;
    return { x: r.x, box: r.box && { x0: r.box.x0 * s, y0: r.box.y0 * s, x1: r.box.x1 * s, y1: r.box.y1 * s } };
  }
  return { x: rgbaToMnist(data.data, data.width, data.height), box: null };
}

/**
 * A private copy of the page's network, refreshed on each call. Classifying other inputs with it
 * leaves the page's network alone, whose activations the other views read for the current input.
 */
export function privateNetwork(): () => Network {
  let net: Network | null = null;
  let arch = '';
  return () => {
    const a = JSON.stringify(store.net.arch);
    if (!net || a !== arch) {
      net = new Network(store.net.arch, 0);
      arch = a;
    }
    net.copyCompatible(store.net);
    return net;
  };
}

const named = (info: DatasetInfo) => info.classes.some((c, k) => c !== String(k));

type Mode = 'draw' | 'photo';

/** Where the photo panel's input comes from. */
type Source =
  | { kind: 'sample'; x: Float32Array; caption: string; label: number | null; key: string }
  | { kind: 'photo'; img: HTMLImageElement; url: string; name: string; x: Float32Array | null; box: Box | null };

export function mountDrawpad(): void {
  const root = $('drawpad');
  const info0 = () => store.info;

  // ── Draw ──
  const pad = h('canvas', { class: 'pad', width: SIZE * SCALE, height: SIZE * SCALE, 'aria-label': 'Drawing pad: draw a digit with the mouse, a pen or a finger' }) as HTMLCanvasElement;
  const empty = h('div', { class: 'pad-empty' }, 'Draw a digit here');
  const guide = h('div', { class: 'pad-guide', 'aria-hidden': 'true' });
  const ctx = pad.getContext('2d', { willReadFrequently: true })!;
  ctx.scale(SCALE, SCALE);
  const clearBtn = h('button', { type: 'button', class: 'btn btn-sm' }, 'Clear');
  const padHint = h('span', { class: 'hint' }, 'Classified live as you draw.');
  const drawPanel = h('div', { class: 'dp-draw' }, h('div', { class: 'pad-wrap' }, pad, guide, empty), h('div', { class: 'pad-tools' }, clearBtn, padHint));

  // ── Photo ──
  const photoInput = h('input', { type: 'file', accept: 'image/*', id: 'photo-input', class: 'dp-file' }) as HTMLInputElement;
  const photoDrop = h('label', { class: 'drop dp-photo-drop', for: 'photo-input' }, h('b', null, 'Drop, paste or choose a photo'), h('span', { class: 'hint' }, 'It is cropped to its centre square and averaged down to 32×32 pixels.'));
  const randomBtn = h('button', { type: 'button', class: 'btn btn-sm', id: 'photo-random' }, 'Random test image');
  const flipBtn = h('button', { type: 'button', class: 'btn btn-sm', id: 'photo-flip', 'aria-pressed': 'false', title: 'Mirror the input left to right' }, 'Flip ↔');
  const resetBtn = h('button', { type: 'button', class: 'btn btn-sm', id: 'photo-reset', disabled: true, title: 'Undo flip, brightness and contrast' }, 'Undo nudges');
  const origCanvas = h('canvas', { class: 'dp-orig', role: 'img', 'aria-label': 'The original image' }) as HTMLCanvasElement;
  const inputCanvas = h('canvas', { class: 'dp-input', role: 'img', 'aria-label': 'The network input, pixel for pixel' }) as HTMLCanvasElement;
  const origCap = h('figcaption', null, 'Original');
  const inputCap = h('figcaption', null, 'Network input');
  const photoCaption = h('p', { class: 'dp-photo-caption' });
  const range = (id: string, label: string, min: number, max: number, step: number, show: (v: number) => string) => {
    const input = h('input', { type: 'range', id, class: 'ds-range', min: String(min), max: String(max), step: String(step) }) as HTMLInputElement;
    const out = h('output', { class: 'dp-value', for: id });
    const el = h('div', { class: 'dp-field' }, h('label', { class: 'label', for: id }, label), h('div', { class: 'dp-control' }, input, out));
    return { el, input, out, show };
  };
  const signed = (v: number) => (v > 0 ? `+${fixed(v, 2)}` : fixed(v, 2));
  const bright = range('photo-brightness', 'Brightness', -0.5, 0.5, 0.05, signed);
  const contrast = range('photo-contrast', 'Contrast', 0.25, 2, 0.05, (v) => `${fixed(v, 2)}×`);
  const photoPanel = h(
    'div',
    { class: 'dp-photo', hidden: true },
    photoDrop,
    photoInput,
    h('div', { class: 'dp-photo-tools' }, randomBtn, flipBtn, resetBtn),
    h('div', { class: 'dp-figs' }, h('figure', null, origCanvas, origCap), h('span', { class: 'dp-arrow', 'aria-hidden': 'true' }, '→'), h('figure', null, inputCanvas, inputCap)),
    photoCaption,
    h('div', { class: 'dp-nudges' }, bright.el, contrast.el),
  );

  // ── Mode switch (Fashion-MNIST offers both) ──
  const modeWrap = h('div', { class: 'dp-modes', hidden: true });
  let mode: Mode = 'draw';

  // ── Prediction ──
  const digit = h('div', { class: 'pred-digit is-empty', 'aria-live': 'polite' }, '?');
  const conf = h('div', { class: 'pred-conf' }, 'No drawing yet');
  const mini = h('canvas', { class: 'pred-mini', role: 'img', 'aria-label': 'Your drawing as the network sees it', title: 'What the network sees: your drawing scaled to 28×28 and centred' }) as HTMLCanvasElement;
  const bars = h('div', { class: 'bars' });
  const prediction = h('div', { class: 'prediction' }, h('div', { class: 'pred-big' }, h('span', { class: 'label' }, 'Prediction'), digit, conf, mini), bars);
  let rows: { k: number; row: HTMLElement; fill: HTMLElement; pctEl: HTMLElement }[] = [];
  let order = '';
  const buildRows = () => {
    clear(bars);
    order = '';
    const info = info0();
    bars.classList.toggle('is-named', named(info));
    rows = info.classes.map((name, k) => {
      const fill = h('span', { class: 'bar-fill', style: { width: '0%' } });
      const pctEl = h('span', { class: 'bar-pct' }, '–');
      const row = h('div', { class: 'bar-row' }, h('span', { class: 'bar-name', title: name }, named(info) ? name : String(k)), h('span', { class: 'bar-track' }, fill), pctEl);
      bars.append(row);
      return { k, row, fill, pctEl };
    });
  };

  // ── Add to training set (drawings) ──
  const addMsg = h('p', { class: 'hint' }, 'Teach the network your handwriting: label this drawing and it joins the training set.');
  const addPick = h('div', { class: 'dp-add-pick' });
  const addTrain = h('div', { class: 'add-train' }, h('span', { class: 'label' }, 'Add to training set as'), addPick, addMsg);
  const addDrawing = (k: number) => {
    const info = info0();
    if (!drawState.x) {
      addMsg.textContent = `Draw ${info.id === 'mnist' ? 'a digit' : 'something'} first.`;
      return false;
    }
    addCustom({ x: drawState.x.slice(), y: k, origin: 'drawing', name: 'Drawing' });
    const n = store.custom.length;
    const what = named(info) ? info.classes[k] : `a ${k}`;
    addMsg.textContent = `Added as ${what}. ${n === 1 ? 'It is your first image in the training set.' : `You now have ${n} images in the training set.`}`;
    clearPad();
    return true;
  };
  const buildAddPick = () => {
    clear(addPick);
    const info = info0();
    if (!named(info)) {
      const chips = digitChips(null, (k) => {
        if (addDrawing(k)) for (const c of Array.from(chips.children)) c.setAttribute('aria-pressed', 'false');
      }, 'Add drawing to the training set as', info.classes.length);
      addPick.append(chips);
      return;
    }
    let chosen = -1;
    const addBtn = h('button', { type: 'button', class: 'btn btn-sm btn-solid', disabled: true }, 'Add');
    const sel = selectField('draw-class', 'Class', [{ value: -1, label: 'Choose…' }, ...info.classes.map((c, k) => ({ value: k, label: c }))], -1, (k) => {
      chosen = k;
      addBtn.disabled = k < 0;
    });
    addBtn.addEventListener('click', () => {
      if (chosen >= 0) addDrawing(chosen);
    });
    addPick.append(h('div', { class: 'dp-add-row' }, sel, addBtn));
  };

  root.append(h('div', { class: 'draw' }, modeWrap, drawPanel, photoPanel, prediction, addTrain));

  // ── Visibility: no work while the section is off screen ──
  let onScreen = true;
  if (typeof IntersectionObserver !== 'undefined') {
    onScreen = false;
    new IntersectionObserver(
      (entries) => {
        const now = entries.some((e) => e.isIntersecting);
        if (now && !onScreen) {
          onScreen = true;
          refresh();
        }
        onScreen = now;
      },
      { rootMargin: '200px' },
    ).observe(root);
  }
  const active = () => store.info.kind === 'image' && !root.hidden;

  // ── Drawing ──
  let drawing = false;
  let hasInk = false;
  let last: { x: number; y: number } | null = null;
  let lastMid: { x: number; y: number } | null = null;
  const brush = () => (store.dataset === 'fashion' ? 30 : 21);

  const pos = (e: PointerEvent) => {
    const r = pad.getBoundingClientRect();
    return { x: ((e.clientX - r.left) / r.width) * SIZE, y: ((e.clientY - r.top) / r.height) * SIZE };
  };

  const stroke = (from: { x: number; y: number }, ctrl: { x: number; y: number } | null, to: { x: number; y: number }) => {
    ctx.strokeStyle = palette().ink;
    ctx.lineWidth = brush();
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    if (ctrl) ctx.quadraticCurveTo(ctrl.x, ctrl.y, to.x, to.y);
    else ctx.lineTo(to.x + 0.01, to.y);
    ctx.stroke();
  };

  pad.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    pad.setPointerCapture(e.pointerId);
    drawing = true;
    last = pos(e);
    lastMid = last;
    stroke(last, null, last);
    hasInk = true;
    empty.hidden = true;
    queue();
  });
  pad.addEventListener('pointermove', (e) => {
    if (!drawing || !last || !lastMid) return;
    const p = pos(e);
    const mid = { x: (last.x + p.x) / 2, y: (last.y + p.y) / 2 };
    stroke(lastMid, last, mid);
    last = p;
    lastMid = mid;
    queue();
  });
  const end = () => {
    if (drawing && last && lastMid) stroke(lastMid, null, last);
    drawing = false;
    last = lastMid = null;
    queue();
  };
  pad.addEventListener('pointerup', end);
  pad.addEventListener('pointercancel', end);

  const clearPad = () => {
    ctx.clearRect(0, 0, SIZE, SIZE);
    hasInk = false;
    empty.hidden = false;
    drawState.x = null;
    queue();
  };
  clearBtn.addEventListener('click', clearPad);

  // Re-colour existing ink when the theme flips (only alpha is used for classification).
  onThemeChange(() => {
    ctx.save();
    ctx.globalCompositeOperation = 'source-in';
    ctx.fillStyle = palette().ink;
    ctx.fillRect(0, 0, SIZE, SIZE);
    ctx.restore();
  });

  let queued = false;
  function queue() {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => {
      queued = false;
      classifyDrawing(true);
    });
  }

  const blank = () => new Float32Array(size(store.input));
  const copy = privateNetwork();

  /** Shows probabilities (or the empty state) in the prediction block. */
  const showPrediction = (x: Float32Array | null, emptyText: string) => {
    const info = info0();
    const isNamed = named(info);
    digit.classList.toggle('is-word', isNamed);
    prediction.classList.toggle('is-named', isNamed);
    if (!x || store.net.inputSize !== x.length || store.net.classes !== rows.length) {
      if (digit.textContent !== '?') digit.textContent = '?'; // unchanged text is not announced again
      digit.classList.add('is-empty');
      conf.textContent = emptyText;
      for (const r of rows) {
        r.fill.style.width = '0%';
        r.pctEl.textContent = '–';
        r.row.classList.remove('is-top');
      }
      return;
    }
    // The page's network when x is the current input (its activations then match); a copy otherwise.
    const probs = (store.probe?.x === x ? store.net : copy()).forward(x);
    const best = argmax(probs);
    const word = isNamed ? info.classes[best] : String(best);
    if (digit.textContent !== word) digit.textContent = word;
    digit.classList.remove('is-empty');
    conf.textContent = `${fixed(probs[best] * 100, 1)}% confident`;
    for (const r of rows) {
      const p = probs[r.k];
      r.fill.style.width = `${(p * 100).toFixed(1)}%`;
      r.pctEl.textContent = `${fixed(p * 100, 1)}%`;
      r.row.classList.toggle('is-top', r.k === best);
    }
    // Named classes are listed most likely first; digits keep their order.
    if (isNamed) {
      const sorted = rows.slice().sort((a, b) => probs[b.k] - probs[a.k] || a.k - b.k);
      const key = sorted.map((r) => r.k).join();
      if (key !== order) {
        order = key;
        for (const r of sorted) bars.append(r.row);
      }
    }
  };

  function classifyDrawing(fromPad: boolean) {
    if (mode !== 'draw' || store.info.kind !== 'image') return;
    const fashion = store.dataset === 'fashion';
    let x: Float32Array | null = null;
    if (hasInk) {
      const { data, width, height } = ctx.getImageData(0, 0, pad.width, pad.height);
      const ink = new Float32Array(width * height);
      for (let i = 0; i < ink.length; i++) ink[i] = data[4 * i + 3] / 255;
      x = fashion ? inkToFashion(ink, width, height) : toMnist(ink, width, height);
    }
    drawState.x = x;
    showPrediction(x, 'No drawing yet');
    paintThumb(mini, x ?? blank(), 28, 28, 56);
    if (!x) {
      if (fromPad && store.probe?.key === 'draw') setProbe({ x: blank(), label: null, caption: 'Your drawing (empty)', key: 'draw' });
      return;
    }
    if (fromPad && (drawing || store.probe?.key === 'draw' || store.probe === null || hasInk)) {
      setProbe({ x, label: null, caption: 'Your drawing', key: 'draw' });
    }
  }

  // ── Photo ──
  let source: Source | null = null;
  let adj: Adjustment = { ...NO_ADJUSTMENT };
  let photoX: Float32Array | null = null;

  const adjustText = () => {
    const parts: string[] = [];
    if (adj.flip) parts.push('flipped');
    if (adj.brightness !== 0) parts.push(`brightness ${signed(adj.brightness)}`);
    if (adj.contrast !== 1) parts.push(`contrast ${fixed(adj.contrast, 2)}×`);
    return parts.join(', ');
  };

  const syncNudges = () => {
    flipBtn.setAttribute('aria-pressed', String(adj.flip));
    bright.input.value = String(adj.brightness);
    bright.out.textContent = bright.show(adj.brightness);
    contrast.input.value = String(adj.contrast);
    contrast.out.textContent = contrast.show(adj.contrast);
    resetBtn.disabled = !isAdjusted(adj);
  };

  /** The photo panel's input after the reader's nudges. */
  const computePhoto = () => {
    const shape = store.input;
    const base = source?.x ?? null;
    photoX = base && base.length === size(shape) ? (isAdjusted(adj) ? adjustImage(base, shape.c, shape.h, shape.w, adj) : base) : null;
  };

  const paintOriginal = () => {
    const pal = palette();
    const shape = store.input;
    if (!source) {
      const c = fitCanvas(origCanvas, VIEW, VIEW);
      c.fillStyle = pal.surface;
      c.fillRect(0, 0, VIEW, VIEW);
      return;
    }
    if (source.kind === 'sample') {
      paintSample(origCanvas, source.x, shape, VIEW);
      origCanvas.setAttribute('aria-label', `The original: ${source.caption}`);
      return;
    }
    const { img, box } = source;
    // The whole photo, letterboxed into the same square as the input beside it.
    const s = Math.min(VIEW / img.naturalWidth, VIEW / img.naturalHeight);
    const w = Math.max(1, Math.round(img.naturalWidth * s));
    const hh = Math.max(1, Math.round(img.naturalHeight * s));
    const ox = Math.floor((VIEW - w) / 2);
    const oy = Math.floor((VIEW - hh) / 2);
    const c = fitCanvas(origCanvas, VIEW, VIEW);
    c.fillStyle = pal.surface;
    c.fillRect(0, 0, VIEW, VIEW);
    c.imageSmoothingQuality = 'high';
    c.drawImage(img, ox, oy, w, hh);
    if (box) {
      // Dim what the network does not see, and frame what it does.
      const bx = ox + box.x0 * s;
      const by = oy + box.y0 * s;
      const bw = (box.x1 - box.x0) * s;
      const bh = (box.y1 - box.y0) * s;
      c.save();
      c.globalAlpha = 0.62;
      c.fillStyle = pal.surface;
      c.beginPath();
      c.rect(ox, oy, w, hh);
      c.rect(bx, by, bw, bh);
      c.fill('evenodd');
      c.restore();
      c.strokeStyle = pal.ink;
      c.lineWidth = 1.5;
      c.strokeRect(bx + 0.75, by + 0.75, Math.max(0, bw - 1.5), Math.max(0, bh - 1.5));
    }
    origCanvas.setAttribute('aria-label', `Your photo ${source.name}${box ? ', with the part the network sees framed' : ''}`);
  };

  const paintInput = () => {
    const shape = store.input;
    paintSample(inputCanvas, photoX ?? blank(), shape, VIEW);
    inputCanvas.setAttribute('aria-label', `The network input, ${shape.w}×${shape.h} ${shape.c === 3 ? 'colour' : 'grey'} pixels, shown enlarged`);
    inputCap.textContent = `Network input · ${shape.w}×${shape.h}`;
  };

  const photoCaptionText = () => {
    if (!source) return '';
    const base = source.kind === 'sample' ? source.caption : 'Your photo';
    const a = adjustText();
    return a ? `${base} · ${a}` : base;
  };

  const renderPhoto = () => {
    computePhoto();
    paintOriginal();
    paintInput();
    const empty = source?.kind === 'photo' && !source.x;
    photoCaption.textContent = !source
      ? 'Drop or paste a photo, or pick a random test image.'
      : empty
        ? 'Nothing stands out from the background in this photo. Try one with a plain backdrop.'
        : photoCaptionText();
    origCap.textContent = source?.kind === 'photo' ? 'Your photo' : 'Original';
    showPrediction(photoX, source ? 'No input' : 'No photo yet');
  };

  // The probe follows the panel at once after a click, and at most ten times a second while a
  // slider is dragged (`throttle`).
  let lastProbe = 0;
  let probeTimer: ReturnType<typeof setTimeout> | null = null;
  const pushProbe = (throttle = false) => {
    if (!source || !photoX) return;
    const now = performance.now();
    if (throttle && now - lastProbe < 100) {
      if (!probeTimer) probeTimer = setTimeout(() => ((probeTimer = null), pushProbe()), 100 - (now - lastProbe));
      return;
    }
    if (probeTimer) clearTimeout(probeTimer);
    probeTimer = null;
    lastProbe = now;
    if (source.kind === 'sample' && !isAdjusted(adj)) {
      // An untouched sample is just that sample (keeps 07's selection and the label in sync).
      if (store.probe?.key !== source.key) setProbe({ x: source.x, label: source.label, caption: source.caption, key: source.key });
      return;
    }
    setProbe({ x: photoX, label: source.kind === 'sample' ? source.label : null, caption: photoCaptionText(), key: 'photo' });
  };

  const setSource = (s: Source | null, push: boolean) => {
    if (source?.kind === 'photo' && source !== s) URL.revokeObjectURL(source.url);
    source = s;
    renderPhoto();
    if (push) pushProbe();
  };

  const loadPhoto = async (file: File) => {
    const info = store.info;
    if (info.kind !== 'image') return;
    try {
      const { img, url } = await readImageFile(file);
      if (store.dataset !== info.id) {
        URL.revokeObjectURL(url);
        return;
      }
      const { x, box } = imageInput(info, img);
      adj = { ...NO_ADJUSTMENT };
      syncNudges();
      setSource({ kind: 'photo', img, url, name: file.name, x, box }, true);
    } catch {
      photoCaption.textContent = `${file.name} could not be read as an image.`;
    }
  };

  photoInput.addEventListener('change', () => {
    const f = photoInput.files?.[0];
    if (f) loadPhoto(f);
    photoInput.value = '';
  });
  for (const target of [photoDrop, origCanvas] as HTMLElement[]) {
    target.addEventListener('dragover', (e) => {
      e.preventDefault();
      photoDrop.classList.add('is-over');
    });
    target.addEventListener('dragleave', () => photoDrop.classList.remove('is-over'));
    target.addEventListener('drop', (e) => {
      e.preventDefault();
      photoDrop.classList.remove('is-over');
      const f = Array.from(e.dataTransfer?.files ?? []).find((q) => q.type.startsWith('image/'));
      if (f) loadPhoto(f);
    });
  }
  // A pasted image goes to the photo panel while it is on screen (section 07 takes it otherwise).
  window.addEventListener('paste', (e) => {
    if (!active() || mode !== 'photo' || !onScreen) return;
    const f = Array.from(e.clipboardData?.files ?? []).find((q) => q.type.startsWith('image/'));
    if (!f) return;
    e.preventDefault();
    loadPhoto(f);
  });

  randomBtn.addEventListener('click', () => {
    const d = store.data;
    if (!d || d.info.kind !== 'image') return;
    const p = testProbe(d, Math.floor(Math.random() * d.testY.length));
    setSource({ kind: 'sample', x: p.x, caption: p.caption, label: p.label, key: p.key }, true);
  });
  flipBtn.addEventListener('click', () => {
    adj = { ...adj, flip: !adj.flip };
    syncNudges();
    renderPhoto();
    pushProbe();
  });
  resetBtn.addEventListener('click', () => {
    adj = { ...NO_ADJUSTMENT };
    syncNudges();
    renderPhoto();
    pushProbe();
  });
  for (const [r, field] of [
    [bright, 'brightness'],
    [contrast, 'contrast'],
  ] as const) {
    r.input.addEventListener('input', () => {
      adj = { ...adj, [field]: Number(r.input.value) };
      syncNudges();
      renderPhoto();
      pushProbe(true);
    });
  }

  // Inputs chosen elsewhere on the page (07, 02) show up here, so they can be nudged.
  store.on('probe', () => {
    if (!active() || mode !== 'photo') return;
    const p = store.probe;
    if (!p || p.key === 'photo' || p.key === 'draw') return;
    if (source?.kind === 'sample' && source.key === p.key && !isAdjusted(adj)) return;
    if (p.x.length !== size(store.input)) return;
    adj = { ...NO_ADJUSTMENT };
    syncNudges();
    setSource({ kind: 'sample', x: p.x, caption: p.caption, label: p.label, key: p.key }, false);
  });

  // ── Modes and datasets ──
  const setMode = (m: Mode, fromUser = false) => {
    mode = m;
    drawPanel.hidden = m !== 'draw';
    addTrain.hidden = m !== 'draw';
    mini.hidden = m !== 'draw';
    photoPanel.hidden = m !== 'photo';
    if (m === 'photo') {
      // Start from whatever the page is showing.
      const p = store.probe;
      if (!source && p && p.key !== 'draw' && p.x.length === size(store.input)) source = { kind: 'sample', x: p.x, caption: p.caption, label: p.label, key: p.key };
      renderPhoto();
      if (fromUser) pushProbe();
    } else {
      classifyDrawing(fromUser);
    }
  };

  const describePad = (info: DatasetInfo) => {
    const fashion = info.id === 'fashion';
    pad.setAttribute('aria-label', fashion ? 'Drawing pad: draw a piece of clothing with the mouse, a pen or a finger' : 'Drawing pad: draw a digit with the mouse, a pen or a finger');
    empty.textContent = fashion ? 'Draw a piece of clothing here' : 'Draw a digit here';
    guide.hidden = fashion;
    padHint.textContent = fashion ? 'Draw a solid shape, as in the photos: it is scaled to fill 28×28.' : 'Classified live as you draw.';
    mini.title = fashion ? 'What the network sees: your drawing scaled to fill 28×28' : 'What the network sees: your drawing scaled to 28×28 and centred';
    addMsg.textContent = fashion
      ? 'Label this drawing and it joins the training set.'
      : 'Teach the network your handwriting: label this drawing and it joins the training set.';
    const photoHint = photoDrop.querySelector('.hint')!;
    photoHint.textContent = info.image?.shape.c === 3
      ? 'It is cropped to its centre square and averaged down to 32×32 pixels.'
      : 'Like the Fashion-MNIST photos, it is cropped to the item, scaled to fill 28×28 and inverted.';
  };

  let shownDataset = '';
  const onDataset = () => {
    const info = store.info;
    if (info.kind !== 'image') {
      shownDataset = ''; // coming back to an image dataset starts afresh
      return;
    }
    if (info.id === shownDataset) return;
    shownDataset = info.id;
    clear(modeWrap);
    const both = info.id === 'fashion';
    modeWrap.hidden = !both;
    const start: Mode = info.image!.shape.c === 3 ? 'photo' : 'draw';
    if (both) modeWrap.append(segmented([{ value: 'draw', label: 'Draw' }, { value: 'photo', label: 'Photo' }], start, (m) => setMode(m as Mode, true), 'Input'));
    describePad(info);
    buildRows();
    buildAddPick();
    // Inputs of the previous dataset have the wrong size: start afresh.
    ctx.clearRect(0, 0, SIZE, SIZE);
    hasInk = false;
    empty.hidden = false;
    drawState.x = null;
    adj = { ...NO_ADJUSTMENT };
    syncNudges();
    setSource(null, false);
    setMode(start);
  };

  // Live predictions follow training at most ten times a second, and only on screen.
  let lastLive = 0;
  let liveTimer: ReturnType<typeof setTimeout> | null = null;
  function refresh() {
    if (!active() || !onScreen) return;
    const now = performance.now();
    if (now - lastLive < 100) {
      if (!liveTimer) liveTimer = setTimeout(() => ((liveTimer = null), refresh()), 100 - (now - lastLive));
      return;
    }
    lastLive = now;
    if (mode === 'draw') classifyDrawing(false);
    else showPrediction(photoX, source ? 'No input' : 'No photo yet');
  }

  store.on('dataset', onDataset);
  store.on('data', () => {
    if (!active()) return;
    if (mode === 'photo' && !source) setMode('photo');
  });
  store.on('weights', refresh);
  store.on('model', refresh);
  onThemeChange(() => {
    paintThumb(mini, drawState.x ?? blank(), 28, 28, 56);
    if (mode === 'photo') {
      paintOriginal();
      paintInput();
    }
  });
  buildRows();
  buildAddPick();
  syncNudges();
  paintThumb(mini, blank(), 28, 28, 56);
  onDataset();
}
