import { addCustom, setProbe } from '../actions';
import { toMnist } from '../data/preprocess';
import { argmax } from '../nn/network';
import { store } from '../store';
import { $, digitChips, h } from './dom';
import { paintThumb } from './draw';
import { onThemeChange, palette } from './theme';

const SIZE = 280; // drawing coordinates
const SCALE = 2; // backing pixels per drawing unit
const BRUSH = 21;

export const drawState = { x: null as Float32Array | null };

export function mountDrawpad(): void {
  const root = $('drawpad');
  const pad = h('canvas', { class: 'pad', width: SIZE * SCALE, height: SIZE * SCALE, 'aria-label': 'Drawing pad: draw a digit with the mouse, a pen or a finger' }) as HTMLCanvasElement;
  const empty = h('div', { class: 'pad-empty' }, 'Draw a digit here');
  const ctx = pad.getContext('2d', { willReadFrequently: true })!;
  ctx.scale(SCALE, SCALE);
  const clearBtn = h('button', { type: 'button', class: 'btn btn-sm' }, 'Clear');
  const digit = h('div', { class: 'pred-digit is-empty', 'aria-live': 'polite' }, '?');
  const conf = h('div', { class: 'pred-conf' }, 'No drawing yet');
  const mini = h('canvas', { class: 'pred-mini', title: 'What the network sees: your drawing scaled to 28×28 and centred' }) as HTMLCanvasElement;
  const bars = h('div', { class: 'bars' });
  const rows = Array.from({ length: 10 }, (_, k) => {
    const fill = h('span', { class: 'bar-fill', style: { width: '0%' } });
    const pctEl = h('span', { class: 'bar-pct' }, '–');
    const row = h('div', { class: 'bar-row' }, h('span', null, String(k)), h('span', { class: 'bar-track' }, fill), pctEl);
    bars.append(row);
    return { row, fill, pctEl };
  });
  const addMsg = h('p', { class: 'hint' }, 'Teach the network your handwriting: label this drawing and it joins the training set.');
  const chips = digitChips(null, (d) => {
    if (!drawState.x) {
      addMsg.textContent = 'Draw a digit first.';
      return;
    }
    addCustom({ x: drawState.x.slice(), y: d, origin: 'drawing', name: 'Drawing' });
    const n = store.custom.length;
    addMsg.textContent = `Added as a ${d}. ${n === 1 ? 'It is your first image in the training set.' : `You now have ${n} images in the training set.`}`;
    clearPad();
    for (const c of Array.from(chips.children)) c.setAttribute('aria-pressed', 'false');
  }, 'Add drawing to the training set as');

  root.append(
    h(
      'div',
      { class: 'draw' },
      h('div', { class: 'pad-wrap' }, pad, h('div', { class: 'pad-guide', 'aria-hidden': 'true' }), empty),
      h('div', { class: 'pad-tools' }, clearBtn, h('span', { class: 'hint' }, 'Classified live as you draw.')),
      h(
        'div',
        { class: 'prediction' },
        h('div', { class: 'pred-big' }, h('span', { class: 'label' }, 'Prediction'), digit, conf, mini),
        bars,
      ),
      h('div', { class: 'add-train' }, h('span', { class: 'label' }, 'Add to training set as'), chips, addMsg),
    ),
  );

  let drawing = false;
  let hasInk = false;
  let last: { x: number; y: number } | null = null;
  let lastMid: { x: number; y: number } | null = null;

  const pos = (e: PointerEvent) => {
    const r = pad.getBoundingClientRect();
    return { x: ((e.clientX - r.left) / r.width) * SIZE, y: ((e.clientY - r.top) / r.height) * SIZE };
  };

  const stroke = (from: { x: number; y: number }, ctrl: { x: number; y: number } | null, to: { x: number; y: number }) => {
    ctx.strokeStyle = palette().ink;
    ctx.lineWidth = BRUSH;
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
      classify(true);
    });
  }

  function classify(fromPad: boolean) {
    let x: Float32Array | null = null;
    if (hasInk) {
      const { data, width, height } = ctx.getImageData(0, 0, pad.width, pad.height);
      const ink = new Float32Array(width * height);
      for (let i = 0; i < ink.length; i++) ink[i] = data[4 * i + 3] / 255;
      x = toMnist(ink, width, height);
    }
    drawState.x = x;
    if (!x) {
      digit.textContent = '?';
      digit.classList.add('is-empty');
      conf.textContent = 'No drawing yet';
      for (const r of rows) {
        r.fill.style.width = '0%';
        r.pctEl.textContent = '–';
        r.row.classList.remove('is-top');
      }
      paintThumb(mini, new Float32Array(784), 28, 28, 56);
      if (fromPad && store.probe?.key === 'draw') setProbe({ x: new Float32Array(784), label: null, caption: 'Your drawing (empty)', key: 'draw' });
      return;
    }
    const probs = store.net.forward(x);
    const best = argmax(probs);
    digit.textContent = String(best);
    digit.classList.remove('is-empty');
    conf.textContent = `${(probs[best] * 100).toFixed(1)}% confident`;
    rows.forEach((r, k) => {
      r.fill.style.width = `${(probs[k] * 100).toFixed(1)}%`;
      r.pctEl.textContent = `${(probs[k] * 100).toFixed(probs[k] >= 0.995 ? 0 : 1)}%`;
      r.row.classList.toggle('is-top', k === best);
    });
    paintThumb(mini, x, 28, 28, 56);
    if (fromPad && (drawing || store.probe?.key === 'draw' || store.probe === null || hasInk)) {
      setProbe({ x, label: null, caption: 'Your drawing', key: 'draw' });
    }
  }

  store.on('weights', () => classify(false));
  store.on('model', () => classify(false));
  onThemeChange(() => paintThumb(mini, drawState.x ?? new Float32Array(784), 28, 28, 56));
  paintThumb(mini, new Float32Array(784), 28, 28, 56);
}
