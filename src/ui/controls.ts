import { pause, play, rebuild, runEpoch, setHyper, stepOnce } from '../actions';
import { store } from '../store';
import { $, clear, h, int, pct, selectField } from './dom';

const LRS = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3];
const BATCHES = [1, 8, 16, 32, 64, 128];

const ICON_PLAY = '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M5 3l12 7-12 7z"/></svg>';
const ICON_PAUSE = '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M4 3h4v14H4zM12 3h4v14h-4z"/></svg>';

export function mountControls(): void {
  const root = $('bar');
  const playBtn = h('button', { class: 'play', type: 'button', id: 'play', 'aria-label': 'Train', 'aria-pressed': 'false', disabled: true });
  playBtn.innerHTML = ICON_PLAY;
  playBtn.addEventListener('click', () => (store.running ? pause() : play()));
  const stepBtn = h('button', { class: 'btn', type: 'button', id: 'step', title: 'Train on one batch', disabled: true, onclick: stepOnce }, 'Step');
  const epochBtn = h('button', { class: 'btn', type: 'button', id: 'epoch', title: 'Train to the end of this epoch, then pause', disabled: true, onclick: runEpoch }, '+1 Epoch');
  const resetBtn = h('button', { class: 'btn', type: 'button', id: 'reset', title: 'Re-initialise the weights', onclick: () => rebuild(true) }, 'Reset');

  const stat = (label: string) => {
    const b = h('b', null, '—');
    return { el: h('div', { class: 'stat' }, h('span', { class: 'label' }, label), b), b };
  };
  const sEpoch = stat('Epoch');
  const sStep = stat('Step');
  const sRate = stat('Digits/s');
  const sAcc = stat('Test acc.');

  const hyper = h('div', { class: 'bar-hyper' });
  const renderHyper = () => {
    clear(hyper);
    hyper.append(
      selectField('lr', 'Learning rate', LRS.map((v) => ({ value: v, label: String(v) })), store.hyper.lr, (lr) => setHyper({ lr })),
      selectField('batch', 'Batch size', BATCHES.map((v) => ({ value: v, label: String(v) })), store.hyper.batchSize, (batchSize) => setHyper({ batchSize })),
      selectField(
        'optimizer',
        'Optimizer',
        [
          { value: 'sgd', label: 'SGD' },
          { value: 'momentum', label: 'Momentum' },
          { value: 'adam', label: 'Adam' },
        ],
        store.hyper.optimizer,
        (optimizer) => setHyper({ optimizer }),
      ),
    );
  };
  renderHyper();
  store.on('hyper', renderHyper);

  root.append(
    h('div', { class: 'bar-group' }, playBtn, stepBtn, epochBtn, resetBtn),
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
    sEpoch.b.textContent = (s?.epochFraction ?? 0).toFixed(2);
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
