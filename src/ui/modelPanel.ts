import './modelPanel.css';
import { exportModel, loadModel, readModel, transferModel } from '../actions';
import { datasetInfo, noun, type DatasetId } from '../data/datasets';
import type { ModelFile } from '../models/format';
import { fetchIndex, fetchModel, fetchTransfer, fmtBytes, layersSummary, sameShape, shapeWords, type TransferReport, type ZooEntry } from '../models/zoo';
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
  | { kind: 'pretrained'; name: string; acc?: number; source: Source; label: string }
  | { kind: 'transfer'; name: string; from: DatasetId; to: string; copied: boolean[]; source: Source; label: string };

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
  let message: { text: string; error: boolean } | null = null;
  let opened: { file: ModelFile; json: unknown; fileName: string } | null = null;
  let armed: string | null = null; // saved model whose Delete awaits confirmation
  let armTimer: ReturnType<typeof setTimeout> | null = null;
  let nameEdited = false;

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
        const acc = o.acc !== undefined ? ` (${pct(o.acc)} test${o.source === 'zoo' ? '' : ' when saved'})` : '';
        if (o.source === 'zoo') return `${o.name}, pretrained${acc}`;
        if (o.source === 'file') return `${o.name}, from the file ${o.label}${acc}`;
        return `${o.name}, saved in this browser${acc}`;
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
    if (!store.valid) evalLine.textContent = 'Fix the architecture below to evaluate.';
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

  const say = (text: string, error = false) => {
    message = { text, error };
    renderStatus();
  };

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
        settle({ kind: 'pretrained', name: file.name, acc: file.meta?.testAccuracy, source: o.source, label: o.label });
        say(`Loaded ${file.name}, architecture and weights, on ${store.info.name}.`);
      } else {
        transferModel(json);
        const o = origin();
        const copied = store.net.blocks.map((_, i) => i < store.net.blocks.length - 1);
        settle({ kind: 'transfer', name: file.name, from: file.dataset, to: store.info.name, copied, source: o.source, label: o.label });
        say(`Transferred ${file.name} to ${store.info.name}. Its hidden layers are frozen; press Train to fit the new output layer.`);
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
  function renderStatus(): void {
    clear(statusBox);
    statusBox.classList.toggle('is-error', !!message?.error);
    if (busy) statusBox.textContent = progress > 0 && progress < 1 ? `Downloading… ${Math.round(progress * 100)}%` : 'Working…';
    else if (message) statusBox.textContent = message.text;
  }

  // ── Explanation (measured numbers from transfer.json) ───

  const explain = h('div', { class: 'model-explain' });
  function renderExplain(): void {
    clear(explain);
    explain.append(
      h(
        'p',
        null,
        h('b', null, 'Transfer'),
        ' copies a trained network’s hidden layers into a network for the current dataset and gives it a new output layer for the new classes. The copied layers start ',
        h('b', null, 'frozen'),
        ': training leaves their weights alone and fits only the new layer, which is quick and needs few examples. Unfreeze them with the lock buttons below to ',
        h('b', null, 'fine-tune'),
        ' every layer once the new one has settled.',
      ),
    );
    const rows = transfer?.rows ?? [];
    if (!rows.length) return;
    const SHORT: Record<string, string> = { mnist: 'MNIST', fashion: 'Fashion', cifar10: 'CIFAR-10' };
    const short = (id: string) => SHORT[id] ?? datasetName(id as DatasetId);
    const sourceSet = (id: string) => zoo?.find((z) => z.id === id)?.dataset ?? id;
    const table = h(
      'table',
      { class: 'transfer-table' },
      h(
        'caption',
        null,
        `Measured test accuracy after training on only the first N images (mean of ${transfer!.seeds} runs, tested on ${int(10000)} official test images). Frozen is what Transfer does; fine-tuned unfreezes every layer halfway through.`,
      ),
      h(
        'thead',
        null,
        h(
          'tr',
          null,
          h('th', { scope: 'col' }, 'From → to'),
          h('th', { scope: 'col', class: 'num' }, 'N'),
          h('th', { scope: 'col', class: 'num' }, 'Scratch'),
          h('th', { scope: 'col', class: 'num', title: 'Copied layers frozen; only the new output layer trains' }, 'Frozen'),
          h('th', { scope: 'col', class: 'num', title: 'Frozen for half the epochs, then every layer trains at a lower rate' }, 'Fine-tuned'),
        ),
      ),
      h(
        'tbody',
        null,
        ...rows.map((r) => {
          const best = Math.max(r.scratch.mean, r.frozen.mean, r.fineTune.mean);
          const cell = (v: number) => h('td', { class: `num${v === best ? ' is-best' : ''}` }, pct(v));
          return h('tr', null, h('th', { scope: 'row' }, `${short(sourceSet(r.from))} → ${short(r.to)}`), h('td', { class: 'num' }, int(r.train)), cell(r.scratch.mean), cell(r.frozen.mean), cell(r.fineTune.mean));
        }),
      ),
    );
    explain.append(h('div', { class: 'transfer-wrap' }, table), h('p', { class: 'hint' }, takeaway(rows)));
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
    h('label', { class: 'field model-name', for: 'model-name' }, h('span', { class: 'label' }, 'Name'), nameInput),
    h(
      'div',
      { class: 'zoo-actions' },
      h('button', { type: 'button', class: 'btn btn-sm', id: 'model-save-file', onclick: saveToFile }, 'Save to file'),
      h('button', { type: 'button', class: 'btn btn-sm', id: 'model-open-file', onclick: () => fileInput.click() }, 'Open file…'),
      h('button', { type: 'button', class: 'btn btn-sm', id: 'model-save-browser', onclick: saveInBrowser }, 'Save in this browser'),
      fileInput,
    ),
    choice,
    h('h4', { class: 'model-subsub' }, 'Saved in this browser'),
    savedEmpty,
    savedList,
  );

  root.append(
    h('div', { class: 'model-prov' }, h('span', { class: 'label' }, 'Weights'), provLine, evalLine),
    statusBox,
    h('h3', { class: 'sub model-h' }, 'Pretrained'),
    zooStatus,
    zooList,
    explain,
    h('h3', { class: 'sub model-h' }, 'Your model'),
    yours,
  );

  function renderAll(): void {
    if (!nameEdited) nameInput.value = defaultName();
    renderProv();
    renderStatus();
    renderZoo();
    renderSaved();
  }

  renderAll();
  renderExplain();
  store.on('model', onModel);
  store.on('dataset', () => {
    if (!nameEdited) nameInput.value = defaultName();
    renderZoo();
    renderSaved();
    renderProv();
  });
  store.on('frozen', renderProv);
  store.on('data', renderProv);
  store.on('metrics', renderProv);
  store.on('status', renderProv);
  void loadZoo().then(renderExplain);
  void fetchTransfer().then((t) => {
    transfer = t;
    renderExplain();
  });
}

/** One plain sentence on what the measured numbers say, worded from the numbers themselves. */
function takeaway(rows: TransferReport['rows']): string {
  const small = rows.filter((r) => r.train === Math.min(...rows.map((x) => x.train)));
  const parts = small.map((r) => {
    const best = Math.max(r.frozen.mean, r.fineTune.mean);
    const gain = (best - r.scratch.mean) * 100;
    const to = datasetName(r.to);
    if (gain >= 1) return `on ${to}, starting from transferred layers beat starting from scratch by ${gain.toFixed(1)} points`;
    if (gain > -1) return `on ${to}, transfer and starting from scratch came out about even`;
    return `on ${to}, transfer did worse than starting from scratch by ${(-gain).toFixed(1)} points`;
  });
  if (!parts.length) return '';
  const n = int(small[0].train);
  return `With only ${n} training images, ${parts.join('; ')}. Features learned on one kind of picture help most when the new pictures look alike and labelled examples are scarce.`;
}
