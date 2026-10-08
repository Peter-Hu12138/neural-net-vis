import { size } from '../nn/types';
import { datasetInfo, type Data, type ImageDatasetId } from './datasets';

const COLS = 100;

const url = (name: string) => `${import.meta.env.BASE_URL}data/${name}`;

/**
 * Decodes one sprite sheet into `out`, channel-major per image (all of R, then G, then B), which
 * is the layout the network reads.
 */
async function loadSheet(name: string, count: number, side: number, channels: number, out: Uint8Array, offset: number): Promise<void> {
  const img = new Image();
  img.src = url(name);
  await img.decode();
  const canvas = document.createElement('canvas');
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
  ctx.drawImage(img, 0, 0);
  const { data, width } = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const HW = side * side;
  const per = HW * channels;
  for (let i = 0; i < count; i++) {
    const ox = (i % COLS) * side;
    const oy = Math.floor(i / COLS) * side;
    const base = (offset + i) * per;
    for (let r = 0; r < side; r++) {
      let p = ((oy + r) * width + ox) * 4;
      for (let c = 0; c < side; c++, p += 4) {
        for (let ch = 0; ch < channels; ch++) out[base + ch * HW + r * side + c] = data[p + ch];
      }
    }
  }
}

const cache = new Map<ImageDatasetId, Promise<Data>>();

/** Loads an image dataset once (later calls share the same promise). */
export function loadImageDataset(id: ImageDatasetId, onProgress: (done: number, total: number) => void = () => {}): Promise<Data> {
  const hit = cache.get(id);
  if (hit) return hit;
  const p = load(id, onProgress);
  cache.set(id, p);
  p.catch(() => cache.delete(id));
  return p;
}

async function load(id: ImageDatasetId, onProgress: (done: number, total: number) => void): Promise<Data> {
  const info = datasetInfo(id);
  const src = info.image!;
  const { shape } = src;
  const n = size(shape);
  const trainX = new Uint8Array(src.train * n);
  const testX = new Uint8Array(src.test * n);
  const sheets = Math.ceil(src.train / src.chunk);
  const total = sheets + 2;
  let done = 0;
  const tick = () => onProgress(++done, total);
  const labels = fetch(url(`${id}-labels.txt`))
    .then((r) => {
      if (!r.ok) throw new Error(`labels: HTTP ${r.status}`);
      return r.text();
    })
    .then((t) => {
      tick();
      const s = t.trim();
      if (s.length !== src.train + src.test) throw new Error(`labels: expected ${src.train + src.test}, got ${s.length}`);
      return Uint8Array.from(s, (ch) => ch.charCodeAt(0) - 48);
    });
  const jobs: Promise<void>[] = [];
  for (let c = 0; c < sheets; c++) {
    const count = Math.min(src.chunk, src.train - c * src.chunk);
    jobs.push(loadSheet(`${id}-train-${c}.${src.ext}`, count, shape.h, shape.c, trainX, c * src.chunk).then(tick));
  }
  jobs.push(loadSheet(`${id}-test.${src.ext}`, src.test, shape.h, shape.c, testX, 0).then(tick));
  const [y] = await Promise.all([labels, ...jobs]);
  return {
    info,
    input: shape,
    inputSize: n,
    scale: 1 / 255,
    trainX,
    trainY: y.slice(0, src.train),
    testX,
    testY: y.slice(src.train, src.train + src.test),
  };
}
