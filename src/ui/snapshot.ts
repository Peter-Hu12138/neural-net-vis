import { store } from '../store';
import { h } from './dom';

export interface Synced {
  /** One-line status ("Based on the weights at step 1,234.") plus a recompute button; place it in the section. */
  status: HTMLElement;
  /** Call right before starting a computation: records which weights it uses. */
  markComputed(): void;
  /** Recompute now, regardless of visibility or training state. */
  refreshNow(): void;
  /** True while the section is on screen (or close to it). */
  readonly visible: boolean;
}

/**
 * Keeps an analysis section in step with the network without recomputing on every training tick.
 * `refresh` runs when the section first comes into view (once data has loaded), when the
 * architecture changes, and when the weights change while training is paused. While training
 * runs, the status line says which step the result belongs to and offers a recompute button.
 */
export function syncedSection(root: HTMLElement, refresh: () => void, opts: { auto?: boolean } = {}): Synced {
  const auto = opts.auto ?? true;
  const text = h('span', { class: 'hint' }, 'Not computed yet.');
  const btn = h('button', { type: 'button', class: 'btn btn-sm' }, 'Recompute') as HTMLButtonElement;
  const status = h('div', { class: 'synced-status' }, text, btn);
  let computed: { version: number; step: number } | null = null;
  let visible = false;
  let timer: ReturnType<typeof setTimeout> | null = null;

  const stale = () => !computed || computed.version !== store.version || computed.step !== store.weightsStep;

  const update = () => {
    btn.disabled = !store.data;
    if (!computed) {
      text.textContent = store.data ? 'Not computed yet.' : 'Waiting for MNIST to load…';
      return;
    }
    const at = computed.step.toLocaleString('en-US');
    if (computed.version !== store.version) text.textContent = 'The architecture changed since this was computed.';
    else if (!stale()) text.textContent = `Based on the weights at step ${at}.`;
    else text.textContent = `Computed at step ${at}; the network is now at step ${store.weightsStep.toLocaleString('en-US')}.`;
  };

  const run = () => {
    if (!store.data) return;
    refresh();
    update();
  };

  const maybe = () => {
    update();
    if (!auto || !visible || !store.data || !stale()) return;
    if (store.running && computed && computed.version === store.version) return;
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => {
      timer = null;
      if (visible && stale() && !(store.running && computed && computed.version === store.version)) run();
    }, 250);
  };

  btn.addEventListener('click', run);
  new IntersectionObserver(
    (entries) => {
      visible = entries.some((e) => e.isIntersecting);
      maybe();
    },
    { rootMargin: '200px 0px' },
  ).observe(root);
  for (const ev of ['weights', 'model', 'data', 'status'] as const) store.on(ev, maybe);
  update();

  return {
    status,
    markComputed() {
      computed = { version: store.version, step: store.weightsStep };
      update();
    },
    refreshNow: run,
    get visible() {
      return visible;
    },
  };
}
