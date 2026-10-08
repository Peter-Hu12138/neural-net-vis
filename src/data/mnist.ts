export const TRAIN_COUNT = 20_000;
export const TEST_COUNT = 2_000;
const CHUNK = 5_000;
const COLS = 100;
const SIDE = 28;

export interface Mnist {
  trainX: Uint8Array;
  trainY: Uint8Array;
  testX: Uint8Array;
  testY: Uint8Array;
}

const url = (name: string) => `${import.meta.env.BASE_URL}data/${name}`;

async function loadSprite(name: string, count: number, out: Uint8Array, offset: number): Promise<void> {
  const img = new Image();
  img.src = url(name);
  await img.decode();
  const canvas = document.createElement('canvas');
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
  ctx.drawImage(img, 0, 0);
  const { data, width } = ctx.getImageData(0, 0, canvas.width, canvas.height);
  for (let i = 0; i < count; i++) {
    const ox = (i % COLS) * SIDE;
    const oy = Math.floor(i / COLS) * SIDE;
    const base = (offset + i) * SIDE * SIDE;
    for (let r = 0; r < SIDE; r++) {
      let p = ((oy + r) * width + ox) * 4;
      for (let c = 0; c < SIDE; c++, p += 4) out[base + r * SIDE + c] = data[p];
    }
  }
}

/** Loads the bundled MNIST subset (PNG sprite sheets + a label file). */
export async function loadMnist(onProgress: (done: number, total: number) => void): Promise<Mnist> {
  const trainX = new Uint8Array(TRAIN_COUNT * SIDE * SIDE);
  const testX = new Uint8Array(TEST_COUNT * SIDE * SIDE);
  const chunks = TRAIN_COUNT / CHUNK;
  const total = chunks + 2;
  let done = 0;
  const tick = () => onProgress(++done, total);
  const labels = fetch(url('mnist-labels.bin'))
    .then((r) => {
      if (!r.ok) throw new Error(`labels: HTTP ${r.status}`);
      return r.arrayBuffer();
    })
    .then((b) => {
      tick();
      return new Uint8Array(b);
    });
  const jobs: Promise<void>[] = [];
  for (let c = 0; c < chunks; c++) jobs.push(loadSprite(`mnist-train-${c}.png`, CHUNK, trainX, c * CHUNK).then(tick));
  jobs.push(loadSprite('mnist-test.png', TEST_COUNT, testX, 0).then(tick));
  const [y] = await Promise.all([labels, ...jobs]);
  return { trainX, trainY: y.slice(0, TRAIN_COUNT), testX, testY: y.slice(TRAIN_COUNT, TRAIN_COUNT + TEST_COUNT) };
}

export function sampleToFloat(src: Uint8Array, index: number): Float32Array {
  const x = new Float32Array(784);
  const off = index * 784;
  for (let j = 0; j < 784; j++) x[j] = src[off + j] / 255;
  return x;
}
