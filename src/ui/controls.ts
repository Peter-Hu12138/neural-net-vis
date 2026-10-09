import { pause, play, rebuild, runEpoch, setHyper, setSpeed, stepOnce } from '../actions';
import { noun } from '../data/datasets';
import { store } from '../store';
import { SPEEDS, type Speed } from '../train/protocol';
import { $, h, int, pct, selectField } from './dom';
import './controls.css';

const LRS = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3];
const BATCHES = [1, 8, 16, 32, 64, 128];
const OPTIMIZERS: { value: 'sgd' | 'momentum' | 'adam'; label: string }[] = [
  { value: 'sgd', label: 'SGD' },
  { value: 'momentum', label: 'Momentum' },
  { value: 'adam', label: 'Adam' },
];

const ICON_PLAY = '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M5 3l12 7-12 7z"/></svg>';
const ICON_PAUSE = '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M4 3h4v14H4zM12 3h4v14h-4z"/></svg>';

const SPEED_LABELS: { value: Speed; label: string }[] = [
  { value: 'slow', label: 'Slow' },
  { value: 'normal', label: 'Normal' },
  { value: 'max', label: 'Max' },
];

/** What each speed does, in the dataset's own words ("3,000 points a second"). */
export function speedTitle(speed: Speed, many: string): string {
  const cap = SPEEDS[speed];
  if (speed === 'slow') return `Slow caps training at ${int(cap ?? 0)} ${many} a second, so you can follow the curves and the weights step by step`;
  if (speed === 'normal') return `Normal caps training at ${int(cap ?? 0)} ${many} a second, so you can watch a small network learn`;
  return 'Max trains as fast as this device allows';
}

/** Sets a select's value without rebuilding it (rebuilding would drop keyboard focus). */
function setSelect<T extends string | number>(field: HTMLElement, options: { value: T }[], value: T): boolean {
  const sel = field.querySelector('select')!;
  const i = options.findIndex((o) => o.value === value);
  if (i < 0) return false;
  sel.selectedIndex = i;
  return true;
}

export function mountControls(): void {
  const root = $('bar');
  const playBtn = h('button', { class: 'play', type: 'button', id: 'play', 'aria-label': 'Train', 'aria-pressed': 'false', disabled: true });
  playBtn.innerHTML = ICON_PLAY;
  playBtn.addEventListener('click', () => (store.running ? pause() : play()));
  const stepBtn = h('button', { class: 'btn', type: 'button', id: 'step', title: 'Train on one batch', disabled: true, onclick: stepOnce }, 'Step');
  const epochBtn = h('button', { class: 'btn', type: 'button', id: 'epoch', title: 'Train to the end of this epoch, then pause', disabled: true, onclick: runEpoch }, '+1 Epoch');
  const resetBtn = h('button', { class: 'btn', type: 'button', id: 'reset', title: 'Re-initialise the weights', onclick: () => rebuild(true) }, 'Reset');

  // Speed: a cap on samples per second. Point datasets finish an epoch in a millisecond, so without
  // a cap the decision boundary would jump straight to its final shape.
  const speedSeg = h('div', { class: 'seg', role: 'group', 'aria-labelledby': 'speed-label', id: 'speed' });
  const speedBtns = SPEED_LABELS.map((o) => {
    const b = h('button', { type: 'button', 'data-speed': o.value, 'aria-pressed': String(store.speed === o.value) }, o.label);
    b.addEventListener('click', () => {
      if (store.speed !== o.value) setSpeed(o.value);
    });
    speedSeg.append(b);
    return b;
  });
  const speedNote = h('span', { class: 'speed-note', id: 'speed-note' });
  const speedField = h('div', { class: 'field bar-speed' }, h('span', { class: 'label', id: 'speed-label' }, 'Speed'), speedSeg, speedNote);
  speedSeg.setAttribute('aria-describedby', 'speed-note');

  const stat = (label: string) => {
    const lab = h('span', { class: 'label' }, label);
    const b = h('b', null, '—');
    return { el: h('div', { class: 'stat' }, lab, b), b, lab };
  };
  const sEpoch = stat('Epoch');
  const sStep = stat('Step');
  const sRate = stat('Digits/s');
  const sAcc = stat('Test acc.');

  const lrMenu = () => selectField('lr', 'Learning rate', LRS.map((v) => ({ value: v, label: String(v) })), store.hyper.lr, (lr) => setHyper({ lr }));
  let lrField = lrMenu();
  const batchField = selectField('batch', 'Batch size', BATCHES.map((v) => ({ value: v, label: String(v) })), store.hyper.batchSize, (batchSize) => setHyper({ batchSize }));
  const optField = selectField('optimizer', 'Optimizer', OPTIMIZERS, store.hyper.optimizer, (optimizer) => setHyper({ optimizer }));
  const hyper = h('div', { class: 'bar-hyper' }, lrField, batchField, optField);

  /** Reflects store.hyper and store.speed (setDataset changes both per kind of data). */
  const renderHyper = () => {
    // A learning rate outside the menu (from a loaded model, say) is added to it.
    if (!setSelect(lrField, LRS.map((v) => ({ value: v })), store.hyper.lr)) {
      LRS.push(store.hyper.lr);
      LRS.sort((a, b) => a - b);
      const fresh = lrMenu();
      lrField.replaceWith(fresh);
      lrField = fresh;
    }
    setSelect(batchField, BATCHES.map((v) => ({ value: v })), store.hyper.batchSize);
    setSelect(optField, OPTIMIZERS, store.hyper.optimizer);
    renderSpeed();
  };

  const renderSpeed = () => {
    const many = noun(store.info, 2);
    for (const b of speedBtns) {
      const v = b.dataset.speed as Speed;
      b.setAttribute('aria-pressed', String(store.speed === v));
      b.title = speedTitle(v, many);
    }
    speedNote.textContent = speedTitle(store.speed, many);
    const one = noun(store.info);
    sRate.lab.textContent = `${one[0].toUpperCase()}${one.slice(1)}s/s`;
    sRate.el.title = `Training ${many} per second${SPEEDS[store.speed] ? `, capped at ${int(SPEEDS[store.speed]!)} by the speed setting` : ''}`;
  };
  renderHyper();
  store.on('hyper', renderHyper);
  store.on('dataset', renderSpeed);

  root.append(
    h('div', { class: 'bar-group' }, playBtn, stepBtn, epochBtn, resetBtn),
    speedField,
    h('div', { class: 'bar-stats' }, sEpoch.el, sStep.el, sRate.el, sAcc.el),
    hyper,
  );

  const update = () => {
    const s = store.status;
    const ready = !!store.data && store.valid;
    const running = !!s?.running;
    playBtn.disabled = !ready;
    stepBtn.disabled = !ready || running;
    epochBtn.disabled = !ready || running;
    resetBtn.disabled = !store.valid;
    playBtn.innerHTML = running ? ICON_PAUSE : ICON_PLAY;
    playBtn.setAttribute('aria-pressed', String(running));
    playBtn.setAttribute('aria-label', running ? 'Pause training' : 'Train');
    playBtn.title = running ? 'Pause' : 'Train continuously';
    const ep = s?.epochFraction ?? 0;
    // Point datasets run thousands of epochs; past 100 the decimals only flicker.
    sEpoch.b.textContent = ep < 100 ? ep.toFixed(2) : int(Math.floor(ep));
    sStep.b.textContent = s ? int(s.step) : '0';
    sRate.b.textContent = running && s ? int(s.samplesPerSec) : '—';
    const last = store.evals[store.evals.length - 1];
    sAcc.b.textContent = last ? pct(last.acc) : '—';
  };
  update();
  store.on('status', update);
  store.on('metrics', update);
  store.on('data', update);
  store.on('model', update);
}
