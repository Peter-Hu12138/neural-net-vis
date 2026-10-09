import './modelPanel.css';
import { exportModel, loadModel, readModel, setFrozen, transferModel } from '../actions';
import { datasetInfo, noun, type DatasetId } from '../data/datasets';
import type { ModelFile } from '../models/format';
import { controlFinding, fetchIndex, fetchModel, fetchTransfer, fmtBytes, layersSummary, sameShape, shapeWords, transferFinding, type TransferCell, type TransferReport, type ZooEntry } from '../models/zoo';
import { blockSignature, type Network } from '../nn/network';
import type { Shape } from '../nn/types';
import { store } from '../store';
import { $, clear, h, int, pct } from './dom';

/**
 * Pretrained models, saving and loading, mounted into #models at the top of section 01.
 * It also keeps track of where the current weights came from (the provenance line).
 */

type Source = 'zoo' | 'file' | 'browser';

type Origin =
  | { kind: 'random' }
  | { kind: 'pretrained'; name: string; dataset: DatasetId; acc?: number; source: Source; label: string }
  | { kind: 'transfer'; name: string; from: DatasetId; to: string; copied: boolean[]; source: Source; label: string };

/** A line in the status box, optionally with one follow-up action (a button). */
interface Message {
  text: string;
  error: boolean;
  action?: { label: string; run: () => void };
}

interface Provenance {
  origin: Origin;
  /** Blocks of the current network that still hold the origin's weights (after keep-weights edits). */
  kept: number | null;
}

/** A model saved in this browser (the list lives under one key, each file under its own). */
interface SavedEntry {
  id: string;
  name: string;
  dataset: DatasetId;
  created: string;
  bytes: number;
  acc?: number;
  input: Shape;
  layers: string;
}

const SAVED_KEY = 'raster.models';
const savedKey = (id: string) => `raster.model.${id}`;

const isQuota = (e: unknown) =>
  e instanceof DOMException && (e.name === 'QuotaExceededError' || e.name === 'NS_ERROR_DOM_QUOTA_REACHED' || e.code === 22 || e.code === 1014);

function readSaved(): SavedEntry[] {
  try {
    const raw = localStorage.getItem(SAVED_KEY);
    const list = raw ? (JSON.parse(raw) as SavedEntry[]) : [];
    return Array.isArray(list) ? list.filter((e) => e && typeof e.id === 'string') : [];
  } catch {
    return [];
  }
}

function storageError(e: unknown): Error {
  if (isQuota(e)) {
    return new Error('This browser has no room left for saved models (pages get about 5 MB). Delete a saved model, or use Save to file instead.');
  }
  return new Error('This browser does not allow saving here (storage may be turned off, as in some private windows). Use Save to file instead.');
}

const datasetName = (id: DatasetId) => {
  try {
    return datasetInfo(id).name;
  } catch {
    return id;
  }
};

const fmtDate = (iso: string) => {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? '' : d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' });
};

const slug = (s: string) =>
  s
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 60) || 'raster-model';

const errText = (e: unknown) => (e instanceof Error ? e.message : String(e));

/** How many blocks of `next` hold exactly the weights of the same block in `prev`. */
function keptBlocks(prev: Network, next: Network): number {
  let n = 0;
  next.blocks.forEach((b, i) => {
    const p = prev.blocks[i];
    if (!p || blockSignature(p) !== blockSignature(b)) return;
    for (let j = 0; j < b.W.length; j++) if (b.W[j] !== p.W[j]) return;
    for (let j = 0; j < b.b.length; j++) if (b.b[j] !== p.b[j]) return;
    n++;
  });
  return n;
}

/** "Conv layers", "Conv and dense layers" or "Dense layers", for the copied blocks. */
function copiedWords(net: Network, copied: boolean[]): string {
  const kinds = new Set(net.blocks.filter((_, i) => copied[i]).map((b) => b.kind));
  if (kinds.size === 2) return 'Conv and dense layers';
  return kinds.has('conv') ? 'Conv layers' : 'Dense layers';
}

export function mountModelPanel(): void {
  const root = $('models');
  root.classList.add('models');

  let zoo: ZooEntry[] | null = null;
  let zooError: string | null = null;
  let transfer: TransferReport | null = null;
  let busy: string | null = null; // id of the model being fetched or applied
  let progress = 0;
  let message: Message | null = null;
  let opened: { file: ModelFile; json: unknown; fileName: string } | null = null;
  let armed: string | null = null; // saved model whose Delete awaits confirmation
  let armTimer: ReturnType<typeof setTimeout> | null = null;
  let nameEdited = false;
  /** Switches section 01 to its Layers view (set once the tabs exist). */
  let showLayers: () => void = () => {};

  let prov: Provenance = { origin: { kind: 'random' }, kept: null };
  let provVersion = store.version;
  let lastNet = store.net;

  // ── Provenance ──────────────────────────────────────────

  const provLine = h('p', { class: 'model-prov-text', id: 'model-provenance' });
  const evalLine = h('p', { class: 'model-eval', id: 'model-eval' });

  const originText = (o: Origin): string => {
    switch (o.kind) {
      case 'random':
        return `Random start (seed ${store.seed})`;
      case 'pretrained': {
        // After a switch to another dataset (with kept weights) the accuracy belongs to the old one.
        const elsewhere = o.dataset !== store.dataset;
        const acc = o.acc !== undefined ? ` (${pct(o.acc)} test${o.source === 'zoo' ? '' : ' when saved'}${elsewhere ? ' there' : ''})` : '';
        if (o.source === 'zoo') return `${o.name}, pretrained${elsewhere ? ` on ${datasetName(o.dataset)}` : ''}${acc}`;
        if (o.source === 'file') return `${o.name}, from the file ${o.label}${elsewhere ? `, trained on ${datasetName(o.dataset)}` : ''}${acc}`;
        return `${o.name}, saved in this browser${elsewhere ? `, trained on ${datasetName(o.dataset)}` : ''}${acc}`;
      }
      case 'transfer': {
        const copied = o.copied.map((c, i) => c && i < store.net.blocks.length);
        const nFrozen = copied.filter((c, i) => c && store.isFrozen(i)).length;
        const nCopied = copied.filter(Boolean).length;
        const state = nFrozen === nCopied ? 'frozen' : nFrozen === 0 ? 'unfrozen' : 'partly frozen';
        const where = o.source === 'zoo' ? o.name : o.source === 'file' ? `${o.name} (file ${o.label})` : `${o.name} (saved in this browser)`;
        return `${copiedWords(store.net, copied)} transferred from ${where}, ${state}; new output layer for ${o.to}`;
      }
    }
  };

  const renderProv = () => {
    let text = originText(prov.origin);
    const total = store.net.blocks.length;
    if (prov.kept !== null && prov.kept < total && (prov.origin.kind !== 'random' || store.keepWeights)) {
      text = prov.origin.kind === 'random' ? `${text}; ${prov.kept} of ${total} layers kept their weights through an edit` : `${prov.kept} of ${total} layers from: ${text}`;
    }
    const s = store.status;
    if (s && s.seen > 0) text += `, then trained here for ${s.epochFraction.toFixed(2)} epochs`;
    provLine.textContent = text;

    const d = store.data;
    const last = store.evals[store.evals.length - 1];
    if (!store.valid) evalLine.textContent = 'Fix the architecture under Layers to evaluate.';
    else if (!d) evalLine.textContent = `Test accuracy here: waiting for ${store.info.name} to load…`;
    else if (!last) evalLine.textContent = 'Test accuracy here: evaluating…';
    else evalLine.textContent = `Test accuracy here: ${pct(last.acc)} on ${int(d.testY.length)} test ${noun(store.info, d.testY.length)}${last.step === 0 ? ', before any training' : ''}`;
  };

  /** Called after any rebuild the panel did not start: work out what is left of the origin. */
  const onModel = () => {
    if (busy) return;
    if (store.version !== provVersion) {
      const kept = keptBlocks(lastNet, store.net);
      const total = store.net.blocks.length;
      if (kept === 0) prov = { origin: { kind: 'random' }, kept: null };
      else if (kept < total) prov = { origin: prov.origin, kept: Math.min(kept, prov.kept ?? total) };
      provVersion = store.version;
      lastNet = store.net;
      // A rebuild from elsewhere (an edit, a reset, another dataset) makes the last message stale.
      if (message) {
        message = null;
        renderStatus();
      }
    }
    renderProv();
    renderZoo();
    renderSaved();
  };

  const settle = (origin: Origin) => {
    prov = { origin, kept: null };
    provVersion = store.version;
    lastNet = store.net;
  };

  // ── Actions ─────────────────────────────────────────────

  const say = (text: string, error = false, action?: Message['action']) => {
    message = { text, error, action };
    renderStatus();
  };

  /** After a transfer into a network with conv and dense hidden layers: offer to let the dense ones learn. */
  const unfreezeDense = (zooId: string | null): Message['action'] | undefined => {
    const dense = store.net.blocks.map((b, i) => (b.kind === 'dense' && i < store.net.blocks.length - 1 && store.isFrozen(i) ? i : -1)).filter((i) => i >= 0);
    if (!dense.length || !store.net.blocks.some((b, i) => b.kind === 'conv' && store.isFrozen(i))) return undefined;
    const names = dense.map((i) => `Dense ${i + 1}`).join(' and ');
    return {
      label: `Unfreeze ${names}`,
      run: () => {
        for (const i of dense) setFrozen(i, false);
        showLayers();
        document.getElementById(`freeze-${dense[0]}`)?.focus();
        const r = zooId ? measuredRow(zooId, store.dataset) : null;
        say(`${names} will learn too; the conv layers stay frozen.${r ? ` Measured on ${int(r.train)} training ${noun(store.info, r.train)}: ${pct(r.convFrozen.mean)}, against ${pct(r.frozen.mean)} with only the output layer learning.` : ''}`);
      },
    };
  };

  /** The measured transfer result for this source model and target dataset, if there is one. */
  const measuredRow = (zooId: string, to: DatasetId) => transfer?.rows.find((r) => r.from === zooId && r.to === to) ?? null;

  async function apply(id: string, mode: 'load' | 'transfer', get: () => Promise<unknown>, origin: () => { source: Source; label: string }): Promise<void> {
    if (busy) return;
    busy = id;
    progress = 0;
    message = null;
    renderAll();
    try {
      const json = await get();
      const { file } = readModel(json);
      if (mode === 'load') {
        await loadModel(json);
        const o = origin();
        settle({ kind: 'pretrained', name: file.name, dataset: file.dataset, acc: file.meta?.testAccuracy, source: o.source, label: o.label });
        say(`Loaded ${file.name}, architecture and weights, on ${store.info.name}.`);
      } else {
        transferModel(json);
        const o = origin();
        const copied = store.net.blocks.map((_, i) => i < store.net.blocks.length - 1);
        settle({ kind: 'transfer', name: file.name, from: file.dataset, to: store.info.name, copied, source: o.source, label: o.label });
        say(`Transferred ${file.name} to ${store.info.name}. Its hidden layers are frozen, so training fits only the new output layer.`, false, unfreezeDense(o.source === 'zoo' ? id : null));
      }
    } catch (e) {
      say(errText(e), true);
      if (store.version !== provVersion) settle({ kind: 'random' });
    } finally {
      busy = null;
      renderAll();
    }
  }

  const fromZoo = (entry: ZooEntry, mode: 'load' | 'transfer') =>
    apply(
      entry.id,
      mode,
      () =>
        fetchModel(entry, (f) => {
          progress = f;
          renderProgress();
        }),
      () => ({ source: 'zoo', label: entry.id }),
    );

  // ── Pretrained list ─────────────────────────────────────

  const zooList = h('ul', { class: 'zoo', 'aria-label': 'Pretrained models' });
  const zooStatus = h('div', { class: 'zoo-empty' });

  const transferBlock = (input: Shape): string | null =>
    sameShape(input, store.input) ? null : `Needs ${shapeWords(input)} input`;

  /** Re-renders replace the buttons; keep keyboard focus on the one that was in use. */
  const keepFocus = (fn: () => void) => {
    const a = document.activeElement;
    const id = a instanceof HTMLElement && root.contains(a) ? a.id : '';
    fn();
    if (id && document.activeElement !== a) document.getElementById(id)?.focus();
  };

  function renderZoo(): void {
    keepFocus(drawZoo);
  }

  function drawZoo(): void {
    clear(zooList);
    clear(zooStatus);
    if (zooError) {
      zooStatus.append(
        h('p', { class: 'notice' }, zooError),
        h('button', { type: 'button', class: 'btn btn-sm', onclick: () => void loadZoo() }, 'Try again'),
      );
      return;
    }
    if (!zoo) {
      zooStatus.append(h('p', { class: 'hint' }, 'Loading the list of pretrained models…'));
      return;
    }
    for (const e of zoo) {
      const why = transferBlock(e.arch.input);
      const fetching = busy === e.id;
      const load = h(
        'button',
        { type: 'button', id: `zoo-load-${e.id}`, class: 'btn btn-sm', 'aria-label': `Load ${e.name}`, title: `Switch to ${datasetName(e.dataset)} and use this network as it is`, disabled: !!busy, onclick: () => void fromZoo(e, 'load') },
        'Load',
      );
      const tr = h(
        'button',
        {
          type: 'button',
          id: `zoo-transfer-${e.id}`,
          class: 'btn btn-sm',
          'aria-label': `Transfer ${e.name} to ${store.info.name}`,
          title: why ?? `Copy its hidden layers into a network for ${store.info.name}, frozen, with a new output layer`,
          disabled: !!busy || !!why,
          onclick: () => void fromZoo(e, 'transfer'),
        },
        'Transfer',
      );
      zooList.append(
        h(
          'li',
          { class: `zoo-row${fetching ? ' is-busy' : ''}`, 'data-model': e.id },
          h(
            'div',
            { class: 'zoo-head' },
            h('span', { class: 'zoo-name' }, e.name),
            h('span', { class: 'zoo-acc num', title: `Accuracy on the ${e.testSet}` }, `${pct(e.testAccuracy)}`, h('span', { class: 'zoo-acc-label' }, ' test')),
          ),
          h('span', { class: 'zoo-meta num' }, `${datasetName(e.dataset)} · ${int(e.params)} params · ${fmtBytes(e.bytes)}`),
          h('span', { class: 'zoo-layers' }, e.layers),
          h('div', { class: 'zoo-actions' }, load, tr, why ? h('span', { class: 'zoo-why' }, why) : null),
          fetching ? h('div', { class: 'progress zoo-progress', role: 'progressbar', 'aria-label': `Downloading ${e.name}`, 'aria-valuemin': '0', 'aria-valuemax': '100', 'aria-valuenow': String(Math.round(progress * 100)) }, h('span', { style: { width: `${Math.round(progress * 100)}%` } })) : null,
        ),
      );
    }
  }

  function renderProgress(): void {
    const bar = zooList.querySelector<HTMLElement>('.zoo-progress');
    if (!bar) return;
    bar.setAttribute('aria-valuenow', String(Math.round(progress * 100)));
    (bar.firstElementChild as HTMLElement).style.width = `${Math.round(progress * 100)}%`;
    if (busy) statusBox.textContent = `Downloading… ${Math.round(progress * 100)}%`;
  }

  async function loadZoo(): Promise<void> {
    zooError = null;
    zoo = null;
    renderZoo();
    try {
      zoo = await fetchIndex();
    } catch (e) {
      zooError = errText(e);
    }
    renderZoo();
  }

  // ── Status line ─────────────────────────────────────────

  const statusBox = h('div', { class: 'model-status', role: 'status', 'aria-live': 'polite' });
  // The follow-up button sits outside the live region, so it is not read out as part of the message.
  const statusAction = h('div', { class: 'model-status-action', hidden: true });
  function renderStatus(): void {
    keepFocus(() => {
      clear(statusBox);
      clear(statusAction);
      statusBox.classList.toggle('is-error', !busy && !!message?.error);
      if (busy) statusBox.textContent = progress > 0 && progress < 1 ? `Downloading… ${Math.round(progress * 100)}%` : 'Working…';
      else if (message) statusBox.textContent = message.text;
      const act = !busy ? message?.action : undefined;
      statusAction.hidden = !act;
      if (act) statusAction.append(h('button', { type: 'button', class: 'btn btn-sm', id: 'model-status-action', onclick: act.run }, act.label));
    });
  }

  // ── Explanation (measured numbers from transfer.json) ───

  const explain = h('div', { class: 'model-explain' });
  const SHORT: Record<string, string> = { mnist: 'MNIST', fashion: 'Fashion', cifar10: 'CIFAR-10' };
  const short = (id: string) => SHORT[id] ?? datasetName(id as DatasetId);
  const zooName = (id: string) => zoo?.find((z) => z.id === id)?.name ?? id;
  const zooSet = (id: string) => zoo?.find((z) => z.id === id)?.dataset ?? id;
  let detailsOpen = false;

  function renderExplain(): void {
    clear(explain);
    explain.append(
      h(
        'p',
        null,
        'A ',
        h('b', null, 'frozen'),
        ' layer keeps its weights while the rest of the network trains, which is quick: no gradients have to flow into it. Transferred layers start frozen, so at first only the new output layer learns. Unlock a layer with its lock button under Layers to ',
        h('b', null, 'fine-tune'),
        ' it too.',
      ),
    );
    const rows = transfer?.rows ?? [];
    if (!transfer || !rows.length) return;
    // Headline: the measurement closest to what the reader is looking at.
    const r = rows.find((x) => x.to === store.dataset) ?? rows[0];
    const control = controlFinding(r, short(zooSet(r.from)), short(r.to));
    explain.append(h('div', { class: 'model-measured' }, h('p', null, transferFinding(r, zooName(r.from), datasetName(r.to), transfer.checkpoints?.[0])), control ? h('p', null, control) : null));

    const cols: { key: 'scratch' | 'frozen' | 'convFrozen' | 'fineTune'; label: string; title: string }[] = [
      { key: 'scratch', label: 'Scratch', title: 'Random starting weights; every layer learns' },
      { key: 'frozen', label: 'Frozen', title: 'What Transfer does: every copied layer frozen; only the new output layer learns' },
      { key: 'convFrozen', label: 'Conv frozen', title: 'Transfer, then unlock the dense layer: only the conv layers stay frozen' },
      { key: 'fineTune', label: 'Fine-tuned', title: 'Transfer and train the new output layer, then unlock everything and train on at a lower rate' },
    ];
    // One group of rows per pair of datasets ("MNIST → Fashion"), one row per training-set size.
    const pairs = [...new Set(rows.map((x) => `${x.from}>${x.to}`))].map((key) => rows.filter((x) => `${x.from}>${x.to}` === key));
    const table = (caption: string, value: (c: TransferCell) => number | undefined) =>
      h(
        'table',
        { class: 'transfer-table' },
        h('caption', null, caption),
        h('thead', null, h('tr', null, h('th', { scope: 'col' }, 'Images'), ...cols.map((c) => h('th', { scope: 'col', class: 'num', title: c.title }, c.label)))),
        ...pairs.map((group) =>
          h(
            'tbody',
            null,
            h('tr', { class: 'transfer-group' }, h('th', { scope: 'rowgroup', colspan: String(cols.length + 1) }, `${short(zooSet(group[0].from))} → ${short(group[0].to)}`)),
            ...group.map((row) => {
              const vals = cols.map((c) => value(row[c.key]));
              const best = Math.max(...vals.map((v) => v ?? -1));
              return h(
                'tr',
                { class: row === r ? 'is-current' : undefined },
                h('th', { scope: 'row', class: 'num' }, int(row.train)),
                ...vals.map((v) => h('td', { class: `num${v !== undefined && v === best ? ' is-best' : ''}` }, v === undefined ? '—' : pct(v))),
              );
            }),
          ),
        ),
      );
    const seen = rows[0].train * rows[0].epochs;
    const cp = transfer.checkpoints?.[0];
    const details = h(
      'details',
      { class: 'transfer-details', open: detailsOpen },
      h('summary', null, 'All measurements'),
      h(
        'div',
        { class: 'transfer-body' },
        h('div', { class: 'transfer-wrap' }, table(`Test accuracy after ${int(seen)} training images`, (c) => c.mean)),
        cp ? h('div', { class: 'transfer-wrap' }, table(`After only the first ${int(cp)} training images`, (c) => c.early?.[0]?.mean)) : null,
        h(
          'p',
          { class: 'hint' },
          'Frozen: what Transfer does. Conv frozen: Transfer, then unlock the dense layer. Fine-tuned: Transfer, train, then unlock everything at a lower learning rate. ',
          `Images: the first N of the training set, shown again and again until ${int(seen)} have gone by. Means of ${transfer.seeds} runs, each tested on the ${int(10000)} official test images, with Adam at this page's default settings. Both source networks are the small CNN preset, trained on all 60,000 images of their own dataset.`,
        ),
      ),
    );
    details.addEventListener('toggle', () => (detailsOpen = details.open));
    explain.append(details);
  }

  // ── Your model ──────────────────────────────────────────

  const nameInput = h('input', { type: 'text', id: 'model-name', autocomplete: 'off', spellcheck: false, maxlength: '80' }) as HTMLInputElement;
  nameInput.addEventListener('input', () => (nameEdited = true));
  const defaultName = () => `My ${store.info.name} model`;
  const currentName = () => nameInput.value.trim() || defaultName();

  const fileInput = h('input', { type: 'file', id: 'model-file', accept: '.json,application/json', class: 'visually-hidden', tabindex: '-1', 'aria-hidden': 'true' }) as HTMLInputElement;
  fileInput.addEventListener('change', () => {
    const f = fileInput.files?.[0];
    fileInput.value = '';
    if (f) void openFile(f);
  });

  async function openFile(f: File): Promise<void> {
    opened = null;
    message = null;
    try {
      if (f.size > 20_000_000) throw new Error(`${f.name} is ${fmtBytes(f.size)}, too large to be a Raster model.`);
      let json: unknown;
      try {
        json = JSON.parse(await f.text());
      } catch {
        throw new Error(`${f.name} is not a model file: it is not valid JSON.`);
      }
      const { file } = readModel(json);
      opened = { file, json, fileName: f.name };
    } catch (e) {
      message = { text: errText(e), error: true };
    }
    renderAll();
  }

  function saveToFile(): void {
    try {
      const name = currentName();
      const text = JSON.stringify(exportModel(name));
      const url = URL.createObjectURL(new Blob([text], { type: 'application/json' }));
      const a = h('a', { href: url, download: `${slug(name)}.json`, hidden: true });
      document.body.append(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 2000);
      say(`Saved ${name} as ${slug(name)}.json (${fmtBytes(text.length)}).`);
    } catch (e) {
      say(`Could not save the file: ${errText(e)}`, true);
    }
  }

  function saveInBrowser(): void {
    const name = currentName();
    const file = exportModel(name);
    const text = JSON.stringify(file);
    const id = `${Date.now().toString(36)}${Math.floor(Math.random() * 1e6).toString(36)}`;
    const entry: SavedEntry = {
      id,
      name,
      dataset: file.dataset,
      created: file.meta.created ?? new Date().toISOString(),
      bytes: text.length,
      acc: file.meta.testAccuracy,
      input: file.arch.input,
      layers: layersSummary(file.arch),
    };
    try {
      localStorage.setItem(savedKey(id), text);
      try {
        localStorage.setItem(SAVED_KEY, JSON.stringify([...readSaved(), entry]));
      } catch (e) {
        localStorage.removeItem(savedKey(id));
        throw e;
      }
      say(`Saved ${name} in this browser (${fmtBytes(text.length)}).`);
    } catch (e) {
      say(storageError(e).message, true);
    }
    renderSaved();
  }

  function removeSaved(entry: SavedEntry): void {
    try {
      localStorage.removeItem(savedKey(entry.id));
      localStorage.setItem(SAVED_KEY, JSON.stringify(readSaved().filter((e) => e.id !== entry.id)));
      say(`Deleted ${entry.name} from this browser.`);
    } catch (e) {
      say(storageError(e).message, true);
    }
    armed = null;
    renderSaved();
  }

  const getSaved = (entry: SavedEntry) => async () => {
    let raw: string | null;
    try {
      raw = localStorage.getItem(savedKey(entry.id));
    } catch (e) {
      throw storageError(e);
    }
    if (!raw) throw new Error(`${entry.name} is no longer in this browser's storage.`);
    try {
      return JSON.parse(raw) as unknown;
    } catch {
      throw new Error(`${entry.name} is damaged in this browser's storage.`);
    }
  };

  const savedList = h('ul', { class: 'saved', 'aria-label': 'Models saved in this browser' });
  const savedEmpty = h('p', { class: 'hint saved-empty' });
  const choice = h('div', { class: 'model-choice' });

  function renderSaved(): void {
    keepFocus(drawSaved);
  }

  function drawSaved(): void {
    clear(savedList);
    const list = readSaved();
    savedEmpty.textContent = list.length ? '' : 'Nothing saved in this browser yet.';
    savedEmpty.hidden = !!list.length;
    for (const e of list) {
      const why = transferBlock(e.input);
      const isArmed = armed === e.id;
      const del = h(
        'button',
        {
          type: 'button',
          id: `saved-delete-${e.id}`,
          class: `btn btn-sm${isArmed ? ' is-armed' : ''}`,
          'aria-label': isArmed ? `Confirm: delete ${e.name}` : `Delete ${e.name}`,
          disabled: !!busy,
          onclick: () => {
            if (armed === e.id) return removeSaved(e);
            armed = e.id;
            if (armTimer) clearTimeout(armTimer);
            armTimer = setTimeout(() => {
              armed = null;
              renderSaved();
            }, 4000);
            renderSaved();
          },
        },
        isArmed ? 'Sure?' : 'Delete',
      );
      savedList.append(
        h(
          'li',
          { class: 'saved-row', 'data-saved': e.id },
          h('span', { class: 'zoo-name' }, e.name),
          h(
            'span',
            { class: 'zoo-meta num' },
            [datasetName(e.dataset), e.acc !== undefined ? `${pct(e.acc)} test` : null, fmtBytes(e.bytes), fmtDate(e.created)].filter(Boolean).join(' · '),
          ),
          h(
            'div',
            { class: 'zoo-actions' },
            h('button', { type: 'button', id: `saved-load-${e.id}`, class: 'btn btn-sm', 'aria-label': `Load ${e.name}`, disabled: !!busy, onclick: () => void apply(e.id, 'load', getSaved(e), () => ({ source: 'browser', label: e.name })) }, 'Load'),
            h(
              'button',
              { type: 'button', id: `saved-transfer-${e.id}`, class: 'btn btn-sm', 'aria-label': `Transfer ${e.name} to ${store.info.name}`, title: why ?? undefined, disabled: !!busy || !!why, onclick: () => void apply(e.id, 'transfer', getSaved(e), () => ({ source: 'browser', label: e.name })) },
              'Transfer',
            ),
            del,
            why ? h('span', { class: 'zoo-why' }, why) : null,
          ),
        ),
      );
    }

    clear(choice);
    choice.hidden = !opened;
    if (opened) {
      const o = opened;
      const why = transferBlock(o.file.arch.input);
      const acc = o.file.meta?.testAccuracy;
      choice.append(
        h('p', { class: 'model-choice-text' }, h('b', null, o.file.name), ` (${o.fileName}): trained on ${datasetName(o.file.dataset)}, ${shapeWords(o.file.arch.input)} input, ${layersSummary(o.file.arch)}${acc !== undefined ? `, ${pct(acc)} test when saved` : ''}.`),
        h(
          'div',
          { class: 'zoo-actions' },
          h(
            'button',
            {
              type: 'button',
              class: 'btn btn-sm btn-solid',
              disabled: !!busy,
              onclick: () => {
                opened = null;
                void apply('file', 'load', async () => o.json, () => ({ source: 'file', label: o.fileName }));
              },
            },
            'Load as is',
          ),
          h(
            'button',
            {
              type: 'button',
              class: 'btn btn-sm',
              title: why ?? undefined,
              disabled: !!busy || !!why,
              onclick: () => {
                opened = null;
                void apply('file', 'transfer', async () => o.json, () => ({ source: 'file', label: o.fileName }));
              },
            },
            `Transfer to ${store.info.name}`,
          ),
          h(
            'button',
            {
              type: 'button',
              class: 'btn btn-sm',
              onclick: () => {
                opened = null;
                renderSaved();
              },
            },
            'Cancel',
          ),
          why ? h('span', { class: 'zoo-why' }, why) : null,
        ),
      );
    }
  }

  // ── Layout ──────────────────────────────────────────────

  const yours = h(
    'div',
    { class: 'model-yours' },
    h('p', { class: 'hint' }, 'Keep the current network, architecture and weights, as a file or in this browser, and open it again later, here or on another computer.'),
    h('label', { class: 'field model-name', for: 'model-name' }, h('span', { class: 'label' }, 'Name'), nameInput),
    h(
      'div',
      { class: 'zoo-actions' },
      h('button', { type: 'button', class: 'btn btn-sm', id: 'model-save-file', onclick: saveToFile }, 'Save to file'),
      h('button', { type: 'button', class: 'btn btn-sm', id: 'model-save-browser', onclick: saveInBrowser }, 'Save in this browser'),
      h('button', { type: 'button', class: 'btn btn-sm', id: 'model-open-file', onclick: () => fileInput.click() }, 'Open file…'),
      fileInput,
    ),
    choice,
    h('h4', { class: 'model-subsub' }, 'Saved in this browser'),
    savedEmpty,
    savedList,
  );

  // Section 01 shows one view at a time, so it stays about as tall as its neighbours: the layers
  // (the builder, #builder), the pretrained models, or saving and opening.
  type Tab = 'layers' | 'pretrained' | 'files';
  const builderEl = $('builder');
  builderEl.setAttribute('role', 'tabpanel');
  builderEl.setAttribute('aria-labelledby', 'tab-layers');
  const zooHint = h(
    'p',
    { class: 'hint zoo-hint' },
    h('b', null, 'Load'),
    ' uses a model exactly as it was trained, on its own dataset. ',
    h('b', null, 'Transfer'),
    ' keeps its hidden layers, frozen, under a new output layer for the dataset you are on.',
  );
  const panels: Record<Tab, HTMLElement> = {
    layers: builderEl,
    pretrained: h('div', { class: 'model-panel', id: 'models-pretrained', role: 'tabpanel', 'aria-labelledby': 'tab-pretrained' }, zooHint, zooStatus, zooList, explain),
    files: h('div', { class: 'model-panel', id: 'models-files', role: 'tabpanel', 'aria-labelledby': 'tab-files' }, yours),
  };
  const TABS: { id: Tab; label: string }[] = [
    { id: 'layers', label: 'Layers' },
    { id: 'pretrained', label: 'Pretrained' },
    { id: 'files', label: 'Save and open' },
  ];
  let tab: Tab = 'layers';
  const tabButtons = TABS.map((t) =>
    h('button', { type: 'button', role: 'tab', id: `tab-${t.id}`, class: 'model-tab', 'aria-controls': t.id === 'layers' ? 'builder' : `models-${t.id}`, onclick: () => selectTab(t.id) }, t.label),
  );
  const frozenBadge = h('span', { class: 'model-tab-badge' });
  tabButtons[0].append(frozenBadge);
  const tablist = h('div', { class: 'model-tabs', role: 'tablist', 'aria-label': 'Architecture views' }, ...tabButtons);
  tablist.addEventListener('keydown', (e) => {
    const k = (e as KeyboardEvent).key;
    const i = TABS.findIndex((t) => t.id === tab);
    const next = k === 'ArrowRight' ? (i + 1) % TABS.length : k === 'ArrowLeft' ? (i + TABS.length - 1) % TABS.length : k === 'Home' ? 0 : k === 'End' ? TABS.length - 1 : -1;
    if (next < 0) return;
    e.preventDefault();
    selectTab(TABS[next].id, true);
  });

  function selectTab(t: Tab, focus = false): void {
    tab = t;
    TABS.forEach((x, i) => {
      const on = x.id === t;
      tabButtons[i].setAttribute('aria-selected', String(on));
      tabButtons[i].tabIndex = on ? 0 : -1;
      panels[x.id].hidden = !on;
    });
    if (focus) tabButtons[TABS.findIndex((x) => x.id === t)].focus();
  }
  showLayers = () => selectTab('layers');

  /** The Layers tab says how many layers are frozen, so the state shows from the other tabs too. */
  function renderBadge(): void {
    const n = store.net.blocks.reduce((s, _, i) => s + (store.isFrozen(i) ? 1 : 0), 0);
    frozenBadge.textContent = n ? ` · ${n} frozen` : '';
  }

  root.append(
    h('div', { class: 'model-prov' }, h('span', { class: 'label' }, 'Weights'), provLine, evalLine),
    statusBox,
    statusAction,
    tablist,
    panels.pretrained,
    panels.files,
  );
  selectTab('layers');

  function renderAll(): void {
    if (!nameEdited) nameInput.value = defaultName();
    renderProv();
    renderStatus();
    renderZoo();
    renderSaved();
    renderBadge();
  }

  renderAll();
  renderExplain();
  store.on('model', onModel);
  let explained = store.dataset;
  store.on('dataset', () => {
    if (!nameEdited) nameInput.value = defaultName();
    renderZoo();
    renderSaved();
    renderProv();
    // The headline measurement follows the dataset (loading progress also emits 'dataset').
    if (store.dataset !== explained) {
      explained = store.dataset;
      renderExplain();
    }
  });
  store.on('frozen', () => {
    renderProv();
    renderBadge();
  });
  store.on('model', renderBadge);
  store.on('data', renderProv);
  store.on('metrics', renderProv);
  store.on('status', renderProv);
  void loadZoo().then(renderExplain);
  void fetchTransfer().then((t) => {
    transfer = t;
    renderExplain();
  });
}
