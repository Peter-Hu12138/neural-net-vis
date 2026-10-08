import { store } from '../store';
import { h } from './dom';

/** Which network state a result belongs to. */
export interface Stamp {
  version: number;
  step: number;
  rev: number;
  spec: string;
}

export const stampNow = (): Stamp => ({
  version: store.version,
  step: store.weightsStep,
  rev: store.weightsRev,
  spec: JSON.stringify(store.spec),
});

/** True when `s` describes the page's current weights. */
export const isCurrent = (s: Stamp | null): boolean => !!s && s.version === store.version && s.rev === store.weightsRev;

export interface Synced {
  /** Status line ("Based on the weights at step 1,234." + Recompute) with a progress bar; place it in the section. */
  status: HTMLElement;
  /** Call when a computation starts. Returns the stamp to hand back to done(). */
  begin(): Stamp;
  /** Call when a result computed from `stamp` is shown. */
  done(stamp: Stamp): void;
  /** Call when a computation ends without a new result (error, superseded). */
  fail(): void;
  /** Progress of the running computation, 0–1, or null to hide the bar. */
  setProgress(fraction: number | null): void;
  /** Ask for a refresh, subject to the section's policy (on screen, data loaded, see below). */
  request(): void;
  /** Recompute now, regardless of visibility or training state. */
  refreshNow(): void;

  /** Stamp of the result on screen, or null. */
  readonly shown: Stamp | null;
  /** True while the section is on screen (or close to it). */
  readonly visible: boolean;
  /** @deprecated use begin()/done(). Marks the current weights as computed. */
  markComputed(): void;
}

/**
 * Keeps an analysis section in step with the network without recomputing on every training tick.
 *
 * Policy (the same for every section): `refresh` runs when the section comes into view with no
 * result yet, when the architecture or weights changed and training is paused, and on Recompute.
 * While training runs a section computes once if it has nothing to show, then holds its result
 * and says which step it belongs to. Requests are throttled to one per 250 ms; a burst of
 * trainer ticks never postpones a pending refresh.
 */
export function syncedSection(root: HTMLElement, refresh: () => void, opts: { auto?: boolean } = {}): Synced {
  const auto = opts.auto ?? true;
  const text = h('span', { class: 'hint' }, 'Not computed yet.');
  const btn = h('button', { type: 'button', class: 'btn btn-sm' }, 'Recompute') as HTMLButtonElement;
  const bar = h('span', { style: { width: '0%' } });
  const progress = h('div', { class: 'progress synced-progress', role: 'progressbar', 'aria-valuemin': '0', 'aria-valuemax': '100', hidden: true }, bar);
  const status = h('div', { class: 'synced-status' }, text, btn, progress);
  let shown: Stamp | null = null;
  let busy: Stamp | null = null;
  // Short computations (live attribution while drawing) should not flash "Computing…".
  let busyVisible = false;
  let busyTimer: ReturnType<typeof setTimeout> | null = null;
  const clearBusy = () => {
    busy = null;
    busyVisible = false;
    if (busyTimer) clearTimeout(busyTimer);
    busyTimer = null;
  };
  let visible = false;
  let timer: ReturnType<typeof setTimeout> | null = null;

  const stale = () => !isCurrent(shown);
  const fmtStep = (n: number) => n.toLocaleString('en-US');

  const update = () => {
    setText();
    text.title = text.textContent ?? '';
  };
  const setText = () => {
    btn.disabled = !store.data;
    if (!store.data) {
      text.textContent = `Waiting for ${store.info.name} to load…`;
      return;
    }
    if (busy && (busyVisible || !shown)) {
      text.textContent = shown && stale() ? `Updating to step ${fmtStep(busy.step)}…` : 'Computing…';
      return;
    }
    if (!shown) {
      text.textContent = 'Not computed yet.';
      return;
    }
    if (shown.spec !== JSON.stringify(store.spec)) text.textContent = 'The architecture changed since this was computed.';
    else if (shown.version !== store.version) text.textContent = 'The weights were re-initialised since this was computed.';
    else if (!stale()) text.textContent = `Based on the weights at step ${fmtStep(shown.step)}.`;
    else if (shown.step !== store.weightsStep)
      text.textContent = `Computed at step ${fmtStep(shown.step)}; the network is now at step ${fmtStep(store.weightsStep)}.`;
    else text.textContent = `Computed at step ${fmtStep(shown.step)}, before the latest manual weight update.`;
  };

  const run = () => {
    if (!store.data) return;
    refresh();
    update();
  };

  /** Whether the policy allows an automatic refresh right now. */
  const wanted = () => {
    if (!auto || !visible || !store.data || !stale()) return false;
    // Already computing exactly this; or training moves the weights on every tick, so let a job for
    // this network finish rather than restart it ~3 times a second and never show anything.
    if (busy && (isCurrent(busy) || (store.running && busy.version === store.version))) return false;
    if (store.running && shown && shown.version === store.version) return false; // hold while training
    return true;
  };

  const request = () => {
    update();
    if (!wanted() || timer) return;
    timer = setTimeout(() => {
      timer = null;
      if (wanted()) run();
    }, 250);
  };

  btn.addEventListener('click', run);
  new IntersectionObserver(
    (entries) => {
      visible = entries.some((e) => e.isIntersecting);
      request();
    },
    { rootMargin: '200px 0px' },
  ).observe(root);
  for (const ev of ['weights', 'model', 'data', 'status'] as const) store.on(ev, request);
  update();

  return {
    status,
    begin() {
      clearBusy();
      busy = stampNow();
      busyTimer = setTimeout(() => {
        busyTimer = null;
        busyVisible = true;
        update();
      }, 300);
      update();
      return busy;
    },
    done(stamp) {
      shown = stamp;
      clearBusy();
      progress.hidden = true;
      update();
      request(); // the network may have moved on while this was computing
    },
    fail() {
      clearBusy();
      progress.hidden = true;
      update();
      request();
    },

    setProgress(f) {
      progress.hidden = f === null;
      if (f !== null) {
        const pct = Math.round(Math.max(0, Math.min(1, f)) * 100);
        bar.style.width = `${pct}%`;
        progress.setAttribute('aria-valuenow', String(pct));
      }
    },
    request,
    refreshNow: run,
    get shown() {
      return shown;
    },
    get visible() {
      return visible;
    },
    markComputed() {
      shown = stampNow();
      update();
    },
  };
}
