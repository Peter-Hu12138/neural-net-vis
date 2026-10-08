import { addCustom, removeCustom, setProbe } from '../actions';
import { sampleToFloat } from '../data/mnist';
import { rgbaToMnist } from '../data/preprocess';
import { argmax } from '../nn/network';
import { CUSTOM_REPEAT } from '../train/protocol';
import { store } from '../store';
import { $, clear, digitChips, h } from './dom';
import { paintThumb, thumbCanvas } from './draw';
import { onThemeChange } from './theme';

interface Upload {
  id: number;
  name: string;
  url: string;
  x: Float32Array | null;
  label: number | null;
  customId: number | null;
  row?: HTMLElement;
}

const GRID = 60;

async function imageToMnist(file: File): Promise<{ url: string; x: Float32Array | null }> {
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.src = url;
  await img.decode();
  const s = Math.min(1, 320 / Math.max(img.naturalWidth, img.naturalHeight));
  const w = Math.max(1, Math.round(img.naturalWidth * s));
  const hh = Math.max(1, Math.round(img.naturalHeight * s));
  const c = document.createElement('canvas');
  c.width = w;
  c.height = hh;
  const ctx = c.getContext('2d', { willReadFrequently: true })!;
  ctx.drawImage(img, 0, 0, w, hh);
  const { data } = ctx.getImageData(0, 0, w, hh);
  return { url, x: rgbaToMnist(data, w, hh) };
}

export function mountDataPanel(): void {
  const root = $('datapanel');
  const grid = h('div', { class: 'sample-grid' });
  const shuffle = h('button', { type: 'button', class: 'btn btn-sm', disabled: true }, 'Shuffle');
  const gridNote = h('p', { class: 'hint', style: { marginTop: '10px' } }, 'Loading digits…');
  const fileInput = h('input', { type: 'file', accept: 'image/*', multiple: true, id: 'upload-input', style: { display: 'none' } }) as HTMLInputElement;
  const drop = h(
    'label',
    { class: 'drop', for: 'upload-input' },
    h('b', null, 'Drop digit images here'),
    h('span', { class: 'hint' }, 'or click to choose files. Photos, scans and screenshots all work: each is cropped, scaled to 20 px, centred on a 28×28 grid and inverted when the ink is dark.'),
  );
  const list = h('ul', { class: 'uploads' });
  const customGrid = h('div', { class: 'sample-grid', style: { marginTop: '10px' } });
  const customNote = h('p', { class: 'custom-summary' });

  root.append(
    h(
      'div',
      { class: 'data-cols' },
      h(
        'div',
        null,
        h('div', { class: 'chart-head' }, h('p', { class: 'sub' }, 'MNIST test digits'), shuffle),
        grid,
        gridNote,
      ),
      h(
        'div',
        null,
        h('p', { class: 'sub' }, 'Your images'),
        drop,
        fileInput,
        list,
        h('p', { class: 'sub', style: { marginTop: '22px' } }, 'In the training set'),
        customGrid,
        customNote,
      ),
    ),
  );

  // ── MNIST grid ──
  let indices: number[] = [];
  const cells: { i: number; btn: HTMLButtonElement; tag: HTMLElement }[] = [];
  const pick = () => {
    const n = store.data!.testY.length;
    const set = new Set<number>();
    while (set.size < GRID) set.add(Math.floor(Math.random() * n));
    indices = [...set];
    buildGrid();
  };
  const buildGrid = () => {
    clear(grid);
    cells.length = 0;
    const d = store.data!;
    for (const i of indices) {
      const x = sampleToFloat(d.testX, i);
      const tag = h('span', { class: 'thumb-label' }, String(d.testY[i]));
      const btn = h('button', { type: 'button', class: 'thumb', title: `Test digit #${i} · label ${d.testY[i]}`, 'aria-pressed': 'false' }, thumbCanvas(x, 28, 28, 40), tag) as HTMLButtonElement;
      btn.addEventListener('click', () => setProbe({ x, label: d.testY[i], caption: `Test digit #${i} · label ${d.testY[i]}`, key: `test:${i}` }));
      cells.push({ i, btn, tag });
      grid.append(btn);
    }
    syncPressed();
    lastPredict = 0;
    predict();
  };
  const syncPressed = () => {
    for (const c of cells) c.btn.setAttribute('aria-pressed', String(store.probe?.key === `test:${c.i}`));
  };
  shuffle.addEventListener('click', pick);

  let lastPredict = 0;
  let pending: ReturnType<typeof setTimeout> | null = null;
  const predict = () => {
    const d = store.data;
    if (!d || !cells.length) return;
    const now = performance.now();
    if (now - lastPredict < 1200) {
      if (!pending) pending = setTimeout(() => ((pending = null), predict()), 1200 - (now - lastPredict));
      return;
    }
    lastPredict = now;
    let wrong = 0;
    for (const c of cells) {
      const y = d.testY[c.i];
      const p = argmax(store.net.forward(sampleToFloat(d.testX, c.i)));
      const ok = p === y;
      if (!ok) wrong++;
      c.tag.textContent = ok ? String(y) : `${y}→${p}`;
      c.tag.style.color = ok ? '' : 'var(--accent)';
      c.btn.style.borderColor = ok ? '' : 'var(--accent)';
    }
    gridNote.textContent = `${GRID - wrong} of ${GRID} classified correctly by the current weights. Mistakes are marked true→predicted in red. Click any digit to feed it through the network.`;
  };

  // ── Uploads ──
  const uploads: Upload[] = [];
  let nextId = 1;

  const renderUpload = (u: Upload) => {
    const row = u.row ?? h('li', { class: 'upload' });
    u.row = row;
    clear(row);
    const processed = document.createElement('canvas');
    paintThumb(processed, u.x ?? new Float32Array(784), 28, 28, 56);
    const meta = h('div', { class: 'upload-meta' }, h('span', { class: 'upload-name', title: u.name }, u.name));
    if (!u.x) {
      meta.append(h('span', { class: 'layer-error' }, 'No ink found. Try a higher-contrast image.'));
    } else {
      const probs = store.net.forward(u.x);
      const best = argmax(probs);
      meta.append(
        h('span', { class: 'upload-pred' }, 'Predicts ', h('b', null, String(best)), ` · ${(probs[best] * 100).toFixed(1)}%`),
        digitChips(u.label, (d) => {
          u.label = d;
          if (u.customId !== null) {
            removeCustom(u.customId);
            u.customId = addCustom({ x: u.x!, y: d, origin: 'upload', name: u.name });
          }
          renderUpload(u);
        }, `True label for ${u.name}`),
      );
    }
    const actions = h('div', { style: { display: 'grid', gap: '6px' } });
    if (u.x) {
      actions.append(h('button', { type: 'button', class: 'btn btn-sm', onclick: () => setProbe({ x: u.x!, label: u.label, caption: `Upload · ${u.name}`, key: `upload:${u.id}` }) }, 'Use as input'));
      if (u.customId === null) {
        actions.append(
          h('button', {
            type: 'button',
            class: 'btn btn-sm btn-solid',
            disabled: u.label === null,
            title: u.label === null ? 'Pick the true digit first' : 'Add to the training set',
            onclick: () => {
              u.customId = addCustom({ x: u.x!, y: u.label!, origin: 'upload', name: u.name });
              renderUpload(u);
            },
          }, 'Train on it'),
        );
      } else {
        actions.append(h('span', { class: 'tag is-on' }, 'In training set'));
      }
    }
    actions.append(
      h('button', {
        type: 'button',
        class: 'btn btn-sm',
        onclick: () => {
          if (u.customId !== null) removeCustom(u.customId);
          URL.revokeObjectURL(u.url);
          uploads.splice(uploads.indexOf(u), 1);
          row.remove();
        },
      }, 'Delete'),
    );
    row.append(h('img', { src: u.url, alt: `Uploaded image ${u.name}` }), processed, meta, actions);
    return row;
  };

  const addFiles = async (files: FileList | File[]) => {
    for (const f of Array.from(files)) {
      if (!f.type.startsWith('image/')) continue;
      try {
        const { url, x } = await imageToMnist(f);
        const u: Upload = { id: nextId++, name: f.name, url, x, label: null, customId: null };
        uploads.push(u);
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
  // Pasting an image anywhere on the page also works.
  window.addEventListener('paste', (e) => {
    const files = Array.from(e.clipboardData?.files ?? []).filter((f) => f.type.startsWith('image/'));
    if (files.length) addFiles(files);
  });

  const renderCustom = () => {
    clear(customGrid);
    for (const c of store.custom) {
      const rm = h('button', { type: 'button', class: 'thumb', title: `${c.name} · label ${c.y}. Click to remove from the training set.` }, thumbCanvas(c.x, 28, 28, 40), h('span', { class: 'thumb-label' }, String(c.y))) as HTMLButtonElement;
      rm.addEventListener('click', () => {
        removeCustom(c.id);
        const u = uploads.find((q) => q.customId === c.id);
        if (u) {
          u.customId = null;
          renderUpload(u);
        }
      });
      customGrid.append(rm);
    }
    const n = store.custom.length;
    customNote.textContent =
      n === 0
        ? 'Nothing yet. Label an upload or a drawing to add it.'
        : n === 1
          ? `Your image is mixed into training and seen ${CUSTOM_REPEAT}× per epoch. Click it to remove it.`
          : `Your ${n} images are mixed into training, each seen ${CUSTOM_REPEAT}× per epoch. Click one to remove it.`;
  };

  store.on('data', () => {
    shuffle.disabled = false;
    pick();
  });
  store.on('weights', predict);
  store.on('probe', syncPressed);
  store.on('custom', renderCustom);
  onThemeChange(() => {
    if (store.data) buildGrid();
    uploads.forEach(renderUpload);
    renderCustom();
  });
  renderCustom();
}
