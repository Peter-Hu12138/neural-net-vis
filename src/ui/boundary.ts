import './boundary.css';
import { removeCustom } from '../actions';
import { fixed } from '../analysis/stats';
import { store } from '../store';
import { $, clear, h } from './dom';
import { boundaryOpts, create2D, onBoundaryOpts, setBoundaryOpts, type View } from './boundary2d';
import { create3D } from './boundary3d';

/**
 * Decision boundary of point datasets (2-D and 3-D), mounted into #boundary in section 03: the
 * shared toggles, the plot for the current dataset, and the playground's loss read-out.
 */
export function mountBoundary(): void {
  const root = $('boundary');

  // ── Toggles ──
  const toggle = (label: string, title: string, get: () => boolean, set: (v: boolean) => void) => {
    const b = h('button', { type: 'button', class: 'bd-toggle', title, 'aria-pressed': String(get()) }, h('i', { 'aria-hidden': 'true' }), label);
    b.addEventListener('click', () => set(!get()));
    onBoundaryOpts(() => b.setAttribute('aria-pressed', String(get())));
    return b;
  };
  const testBtn = toggle('Show test data', 'Draw the test points as rings', () => boundaryOpts.showTest, (v) => setBoundaryOpts({ showTest: v }));
  const discBtn = toggle('Discretise', 'Flat class regions instead of shading by confidence', () => boundaryOpts.discrete, (v) => setBoundaryOpts({ discrete: v }));
  const addBtn = toggle('Add points', 'Click the plot to add training points of the chosen class', () => boundaryOpts.adding, (v) => setBoundaryOpts({ adding: v }));
  const chips = h('div', { class: 'chips bd-classes', role: 'group', 'aria-label': 'Class of the points you add' });
  const removeBtn = h('button', { type: 'button', class: 'btn btn-sm', hidden: true }, 'Remove my points');
  removeBtn.addEventListener('click', () => {
    for (const c of store.custom.filter((e) => e.origin === 'point')) removeCustom(c.id);
  });
  const addNote = h('span', { class: 'hint bd-add-note' });

  const buildChips = () => {
    clear(chips);
    const info = store.info;
    if (info.kind !== 'points') return;
    if (boundaryOpts.addClass >= info.classes.length) boundaryOpts.addClass = 0;
    info.classes.forEach((name, k) => {
      const b = h(
        'button',
        { type: 'button', class: 'chip bd-class', 'aria-pressed': String(k === boundaryOpts.addClass), 'aria-label': `Add points of ${name}`, title: name },
        h('i', { 'aria-hidden': 'true' }),
        info.glyphs[k],
      );
      b.style.setProperty('--c', `var(--cat-${k % 10})`);
      b.addEventListener('click', () => setBoundaryOpts({ addClass: k, adding: true }));
      chips.append(b);
    });
  };
  const syncChips = () => {
    Array.from(chips.children).forEach((b, k) => b.setAttribute('aria-pressed', String(k === boundaryOpts.addClass)));
    const mine = store.custom.filter((c) => c.origin === 'point').length;
    removeBtn.hidden = mine === 0;
    removeBtn.textContent = mine === 1 ? 'Remove my point' : `Remove my ${mine} points`;
    const name = store.info.classes[boundaryOpts.addClass] ?? '';
    addNote.textContent = boundaryOpts.adding ? `Click the ${store.info.dims === 3 ? 'slice map' : 'plot'} to add ${name} points.` : 'Shift-click also adds one.';
  };
  onBoundaryOpts(syncChips);
  store.on('custom', syncChips);

  const lead = h('p', { class: 'hint bd-lead' });
  const key = h('div', { class: 'bd-key', 'aria-label': 'Key' });
  const controls = h(
    'div',
    { class: 'bd-controls' },
    h('div', { class: 'bd-row', role: 'group', 'aria-label': 'Display' }, testBtn, discBtn),
    h('div', { class: 'bd-row', role: 'group', 'aria-label': 'Add training points' }, addBtn, chips, removeBtn, addNote),
  );

  // ── Metrics (as in the playground) ──
  const kpi = (label: string, title: string) => {
    const b = h('b', null, '—');
    return { el: h('div', { class: 'kpi', title }, h('span', { class: 'label' }, label), b), b };
  };
  const mEpoch = kpi('Epoch', 'Passes through the training points so far');
  const mTrainLoss = kpi('Train loss', 'Training loss: cross-entropy on the latest training batches');
  const mTestLoss = kpi('Test loss', 'Test loss: cross-entropy on the test points, at the latest evaluation');
  const mTrainAcc = kpi('Train acc.', 'Training accuracy: share of the latest training batches classified correctly');
  const mTestAcc = kpi('Test acc.', 'Test accuracy: share of test points classified correctly, at the latest evaluation');
  mTestLoss.el.classList.add('is-accent');
  mTestAcc.el.classList.add('is-accent');
  const metrics = h('div', { class: 'bd-metrics', 'aria-live': 'off' }, mEpoch.el, mTrainLoss.el, mTestLoss.el, mTrainAcc.el, mTestAcc.el);
  const pct = (v: number) => `${(100 * v).toFixed(1)}%`;
  let metricsQueued = false;
  const syncMetrics = () => {
    if (metricsQueued) return;
    metricsQueued = true;
    requestAnimationFrame(() => {
      metricsQueued = false;
      const s = store.status;
      const tp = store.points.at(-1);
      const ev = store.evals.at(-1);
      mEpoch.b.textContent = s ? fixed(s.epochFraction, 2) : fixed(0, 2);
      mTrainLoss.b.textContent = tp ? fixed(tp.loss, 3) : '—';
      mTestLoss.b.textContent = ev ? fixed(ev.loss, 3) : '—';
      mTrainAcc.b.textContent = tp ? pct(tp.acc) : '—';
      mTestAcc.b.textContent = ev ? pct(ev.acc) : '—';
    });
  };
  for (const e of ['metrics', 'status', 'model', 'data'] as const) store.on(e, syncMetrics);

  // ── Body: the 2-D or 3-D view ──
  const body = h('div', { class: 'bd-body' });
  root.append(h('div', { class: 'bd' }, lead, controls, key, body, metrics));

  let v2: View | null = null;
  let v3: View | null = null;
  let current: View | null = null;
  const pick = () => {
    const info = store.info;
    let next: View | null = null;
    if (info.kind === 'points' && info.dims === 2) next = v2 ??= create2D();
    else if (info.kind === 'points' && info.dims === 3) next = v3 ??= create3D();
    if (next !== current) {
      current?.setActive(false);
      clear(body);
      if (next) body.append(next.el);
      current = next;
      current?.setActive(true);
    }
    root.dataset.dims = info.kind === 'points' ? String(info.dims) : '';
    if (info.kind !== 'points') return;
    lead.textContent =
      info.dims === 2
        ? 'Colour shows the class the network predicts at every spot of the plane; pale means unsure. The solid line is the decision boundary, where the prediction flips. Click anywhere to make that spot the network’s input.'
        : 'The grey surface is the decision boundary in space: on either side the network predicts a different class. Turn the cube to see its shape, and move the slice to see one flat cross-section.';
    clear(key);
    key.append(
      ...info.classes.map((name, k) => h('span', { class: 'bd-key-item' }, h('i', { class: 'bd-swatch', style: { background: `var(--cat-${k % 10})` } }), name)),
      h('span', { class: 'bd-key-item' }, h('i', { class: 'bd-mark is-train' }), 'Training'),
      h('span', { class: 'bd-key-item' }, h('i', { class: 'bd-mark is-test' }), 'Test'),
      h('span', { class: 'bd-key-item' }, h('i', { class: 'bd-mark is-custom' }), 'Yours'),
      h('span', { class: 'bd-key-item' }, h('i', { class: 'bd-mark is-probe' }), 'Current input'),
    );
    buildChips();
    syncChips();
    syncMetrics();
  };
  store.on('dataset', pick);
  pick();
}
