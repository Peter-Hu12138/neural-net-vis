/**
 * Pretrains the model zoo in public/models/ and measures transfer learning, with the app's own
 * engine (src/nn Network + Optimizer), so the weights load unchanged in the browser.
 *
 *   npx vite-node scripts/pretrain.ts check                  raw files agree with public/data
 *   npx vite-node scripts/pretrain.ts train [ids…]           train zoo models, write public/models
 *   npx vite-node scripts/pretrain.ts transfer [--seeds 3]   transfer experiments → transfer.json
 *   npx vite-node scripts/pretrain.ts index                  rewrite public/models/index.json
 *
 * Options: --threads N (worker threads, default: CPU count − 1), --raw DIR (holds mnist-raw/,
 * fashion-raw/ and cifar-raw/cifar-10-binary.tar.gz; default $RASTER_RAW or data-raw), or each
 * source on its own: --mnist DIR, --fashion DIR (the four IDX .gz files), --cifar FILE (the
 * CIFAR-10 binary .tar.gz, or the extracted cifar-10-batches-bin/ directory).
 *
 * Sources: MNIST (yann.lecun.com mirror), Fashion-MNIST (github.com/zalandoresearch/fashion-mnist),
 * CIFAR-10 binary version (www.cs.toronto.edu/~kriz/cifar.html). Inputs are scaled exactly as the
 * app scales them: pixel × (1/255), colour images channel-major (all R, then G, then B).
 *
 * Training is data-parallel: each worker thread runs its own copy of the network over a slice of
 * every batch, and the main thread adds up the gradients and takes the optimizer step. The workers
 * run this same file, bundled once with esbuild (which ships with Vite).
 */
import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { cpus, tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads';
import { gunzipSync, inflateSync } from 'node:zlib';
import { decodeModel, encodeModel } from '../src/models/format';
import { archParams, layersSummary, type TransferCell, type TransferReport, type TransferRow, type ZooEntry } from '../src/models/zoo';
import { Network, argmax } from '../src/nn/network';
import { Optimizer } from '../src/nn/optim';
import { Rng } from '../src/nn/rng';
import { size, type Arch, type LayerSpec, type Shape } from '../src/nn/types';

// ── Data ──────────────────────────────────────────────────────────────────

type SetName = 'mnist' | 'fashion' | 'cifar10';

/** Images stored as the app stores them (0–255, channel-major), in shared memory for the workers. */
interface ImageSet {
  shape: Shape;
  n: number;
  x: Uint8Array;
  y: Uint8Array;
}

interface Split {
  train: ImageSet;
  test: ImageSet;
}

const SCALE = 1 / 255; // the app's scale for image data (src/data/images.ts)

function shared(src: Uint8Array): Uint8Array {
  const out = new Uint8Array(new SharedArrayBuffer(src.length));
  out.set(src);
  return out;
}

/** IDX file (MNIST format): magic 0x0000 08 <ndim>, ndim big-endian sizes, then unsigned bytes. */
function readIdx(path: string): { dims: number[]; data: Uint8Array } {
  const buf = gunzipSync(readFileSync(path));
  const magic = buf.readUInt32BE(0);
  if (((magic >>> 8) & 0xff) !== 0x08) throw new Error(`${path}: not an unsigned-byte IDX file`);
  const nd = magic & 0xff;
  const dims = Array.from({ length: nd }, (_, i) => buf.readUInt32BE(4 + 4 * i));
  const off = 4 + 4 * nd;
  const data = new Uint8Array(buf.buffer, buf.byteOffset + off, buf.length - off);
  if (data.length !== dims.reduce((a, b) => a * b, 1)) throw new Error(`${path}: size does not match its header`);
  return { dims, data };
}

function loadIdxDir(dir: string): Split {
  const set = (images: string, labels: string): ImageSet => {
    const im = readIdx(join(dir, images));
    const lb = readIdx(join(dir, labels));
    const [n, h, w] = im.dims;
    if (lb.dims[0] !== n) throw new Error(`${dir}: ${n} images but ${lb.dims[0]} labels`);
    return { shape: { c: 1, h, w }, n, x: shared(im.data), y: shared(lb.data) };
  };
  return { train: set('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz'), test: set('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz') };
}

/** The regular files of a tar archive, by base name. */
function untar(buf: Buffer): Map<string, Buffer> {
  const files = new Map<string, Buffer>();
  let off = 0;
  while (off + 512 <= buf.length) {
    const name = buf.toString('utf8', off, off + 100).replace(/\0[\s\S]*$/, '');
    if (!name) break;
    const sz = parseInt(buf.toString('ascii', off + 124, off + 136).replace(/\0[\s\S]*$/, '').trim() || '0', 8);
    const type = buf[off + 156];
    const start = off + 512;
    if (type === 0 || type === 0x30) files.set(name.split('/').pop()!, buf.subarray(start, start + sz));
    off = start + Math.ceil(sz / 512) * 512;
  }
  return files;
}

/** CIFAR-10 binary: records of 1 label byte + 3,072 pixel bytes (R plane, G plane, B plane). */
function loadCifar(path: string): Split {
  const files = statSync(path).isDirectory()
    ? new Map(readdirSync(path).map((f) => [f, readFileSync(join(path, f))] as const))
    : untar(gunzipSync(readFileSync(path)));
  const set = (names: string[]): ImageSet => {
    const recs = names.map((nm) => {
      const b = files.get(nm);
      if (!b) throw new Error(`${path}: ${nm} is missing`);
      return b;
    });
    const n = recs.reduce((s, b) => s + b.length / 3073, 0);
    const x = new Uint8Array(new SharedArrayBuffer(n * 3072));
    const y = new Uint8Array(new SharedArrayBuffer(n));
    let i = 0;
    for (const b of recs) {
      for (let off = 0; off < b.length; off += 3073, i++) {
        y[i] = b[off];
        x.set(b.subarray(off + 1, off + 3073), i * 3072);
      }
    }
    return { shape: { c: 3, h: 32, w: 32 }, n, x, y };
  };
  return { train: set([1, 2, 3, 4, 5].map((k) => `data_batch_${k}.bin`)), test: set(['test_batch.bin']) };
}

const OFFICIAL_TEST: Record<SetName, string> = {
  mnist: '10,000 official MNIST test images',
  fashion: '10,000 official Fashion-MNIST test images',
  cifar10: '10,000 official CIFAR-10 test images',
};

const NAMES: Record<SetName, string> = { mnist: 'MNIST', fashion: 'Fashion-MNIST', cifar10: 'CIFAR-10' };

/** Writes sample i into x as network input (pixel / 255), mirrored left–right when `flip`. */
function loadSample(set: ImageSet, i: number, x: Float32Array, flip = false): number {
  const { c: C, h: H, w: W } = set.shape;
  const n = C * H * W;
  const off = i * n;
  const src = set.x;
  if (!flip) {
    for (let j = 0; j < n; j++) x[j] = src[off + j] * SCALE;
  } else {
    for (let c = 0; c < C; c++) {
      for (let r = 0; r < H; r++) {
        const row = c * H * W + r * W;
        for (let col = 0; col < W; col++) x[row + col] = src[off + row + W - 1 - col] * SCALE;
      }
    }
  }
  return set.y[i];
}

// ── Sprite cross-check (the app's public/data) ──────────────────────────────

/** 8-bit greyscale PNG reader (the sprites written by scripts/build-datasets.py). */
function readPng(path: string): { width: number; height: number; px: Uint8Array } {
  const buf = readFileSync(path);
  let off = 8;
  let width = 0;
  let height = 0;
  const idat: Buffer[] = [];
  while (off < buf.length) {
    const len = buf.readUInt32BE(off);
    const type = buf.toString('ascii', off + 4, off + 8);
    const data = buf.subarray(off + 8, off + 8 + len);
    if (type === 'IHDR') {
      width = data.readUInt32BE(0);
      height = data.readUInt32BE(4);
      if (data[8] !== 8 || data[9] !== 0) throw new Error(`${path}: expected 8-bit greyscale`);
    }
    if (type === 'IDAT') idat.push(data);
    off += 12 + len;
  }
  const raw = inflateSync(Buffer.concat(idat));
  const px = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) {
    const f = raw[y * (width + 1)];
    const line = raw.subarray(y * (width + 1) + 1, (y + 1) * (width + 1));
    const prev = y ? px.subarray((y - 1) * width, y * width) : new Uint8Array(width);
    const out = px.subarray(y * width, (y + 1) * width);
    for (let i = 0; i < width; i++) {
      const a = i ? out[i - 1] : 0;
      const b = prev[i];
      const c = i ? prev[i - 1] : 0;
      const p = a + b - c;
      const pa = Math.abs(p - a);
      const pb = Math.abs(p - b);
      const pc = Math.abs(p - c);
      const pred = f === 0 ? 0 : f === 1 ? a : f === 2 ? b : f === 3 ? (a + b) >> 1 : pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
      out[i] = (line[i] + pred) & 0xff;
    }
  }
  return { width, height, px };
}

/**
 * Checks that the raw files hold the same images, in the same order and with the same labels, as
 * the subsets the app ships, so a network trained here sees exactly what it will see in the page.
 */
function crossCheck(id: SetName, split: Split): string {
  const labels = readFileSync(`public/data/${id}-labels.txt`, 'utf8').trim();
  const nTrain = id === 'mnist' ? 20_000 : 10_000;
  for (let i = 0; i < 200; i++) {
    if (Number(labels[i]) !== split.train.y[i]) throw new Error(`${id}: training label ${i} differs from public/data`);
    if (Number(labels[nTrain + i]) !== split.test.y[i]) throw new Error(`${id}: test label ${i} differs from public/data`);
  }
  if (id === 'cifar10') return `${id}: the first 200 training and test labels match public/data (sprites are JPEG, so pixels are compared only for the PNG sets)`;
  const sheet = readPng(`public/data/${id}-train-0.png`);
  const test = readPng(`public/data/${id}-test.png`);
  const inputs = (png: { width: number; px: Uint8Array }, i: number) => {
    const out = new Float32Array(784);
    const ox = (i % 100) * 28;
    const oy = Math.floor(i / 100) * 28;
    for (let r = 0; r < 28; r++) for (let c = 0; c < 28; c++) out[r * 28 + c] = png.px[(oy + r) * png.width + ox + c] * SCALE;
    return out;
  };
  const x = new Float32Array(784);
  for (const [png, set, i] of [[sheet, split.train, 0], [sheet, split.train, 4321], [test, split.test, 0], [test, split.test, 1999]] as const) {
    loadSample(set, i, x);
    const app = inputs(png, i);
    for (let j = 0; j < 784; j++) if (app[j] !== x[j]) throw new Error(`${id}: sample ${i} pixel ${j} differs from public/data`);
  }
  return `${id}: labels and four samples (network inputs, bit for bit) match public/data`;
}

// ── Training ──────────────────────────────────────────────────────────────

/** Network weights as one flat array: [W0, b0, W1, b1, …]. */
function flatten(net: Network, into: Float32Array, grads = false): void {
  let o = 0;
  for (const b of net.blocks) {
    const [w, bb] = grads ? [b.gW, b.gb] : [b.W, b.b];
    into.set(w, o);
    o += w.length;
    into.set(bb, o);
    o += bb.length;
  }
}

function unflatten(from: Float32Array, net: Network): void {
  let o = 0;
  for (const b of net.blocks) {
    b.W.set(from.subarray(o, o + b.W.length));
    o += b.W.length;
    b.b.set(from.subarray(o, o + b.b.length));
    o += b.b.length;
  }
}

/** Adds a flat gradient buffer into the network's gradient arrays. */
function addGrads(from: Float32Array, net: Network): void {
  let o = 0;
  for (const b of net.blocks) {
    for (const g of [b.gW, b.gb]) {
      for (let i = 0; i < g.length; i++) g[i] += from[o + i];
      o += g.length;
    }
  }
}

interface Sets {
  [name: string]: Split;
}

type WorkerJob =
  | { type: 'batch'; set: string; idx: Int32Array; flips: Uint8Array | null; frozen: boolean[] | null }
  | { type: 'eval'; set: string; split: 'train' | 'test'; from: number; to: number }
  | { type: 'run'; job: RunJob };

interface Reply {
  loss: number;
  correct: number;
  n: number;
  result?: RunResult;
}

/** A pool of worker threads, each holding a copy of one network, sharing the data sets. */
class Pool {
  private workers: Worker[] = [];
  private waiting: ((r: Reply) => void)[] = [];
  readonly weights: Float32Array;
  readonly grads: Float32Array[];

  constructor(
    bundle: string,
    readonly threads: number,
    sets: Sets,
    arch: Arch | null,
  ) {
    const count = arch ? new Network(arch, 0).paramCount : 1;
    this.weights = new Float32Array(new SharedArrayBuffer(count * 4));
    this.grads = [];
    for (let t = 0; t < threads; t++) {
      const g = new Float32Array(new SharedArrayBuffer(count * 4));
      this.grads.push(g);
      const w = new Worker(bundle, { workerData: { role: 'pretrain-worker', sets, arch, weights: this.weights, grads: g } });
      w.on('message', (r: Reply) => this.waiting[t](r));
      w.on('error', (e) => {
        console.error(`worker ${t} failed:`, e);
        process.exit(1);
      });
      this.workers.push(w);
    }
  }

  call(t: number, job: WorkerJob): Promise<Reply> {
    return new Promise((res) => {
      this.waiting[t] = res;
      this.workers[t].postMessage(job);
    });
  }

  /** One batch spread over the workers; leaves the summed gradients in `net`. */
  async batch(net: Network, set: string, idx: Int32Array, flips: Uint8Array | null, frozen: boolean[] | null = null): Promise<Reply> {
    flatten(net, this.weights);
    const T = Math.min(this.threads, idx.length);
    const per = Math.ceil(idx.length / T);
    const replies = await Promise.all(
      Array.from({ length: T }, (_, t) =>
        this.call(t, { type: 'batch', set, idx: idx.slice(t * per, (t + 1) * per), flips: flips ? flips.slice(t * per, (t + 1) * per) : null, frozen }),
      ),
    );
    net.zeroGrad();
    for (let t = 0; t < T; t++) addGrads(this.grads[t], net);
    return sum(replies);
  }

  async evaluate(net: Network, set: string, split: 'train' | 'test', n: number): Promise<{ acc: number; loss: number }> {
    flatten(net, this.weights);
    const per = Math.ceil(n / this.threads);
    const r = sum(await Promise.all(this.workers.map((_, t) => this.call(t, { type: 'eval', set, split, from: t * per, to: Math.min(n, (t + 1) * per) }))));
    return { acc: r.correct / r.n, loss: r.loss / r.n };
  }

  /** Runs independent jobs, one per free worker, in order of submission. */
  async runAll(jobs: RunJob[], onDone: (i: number, r: RunResult) => void): Promise<RunResult[]> {
    const out: RunResult[] = new Array(jobs.length);
    let next = 0;
    await Promise.all(
      this.workers.map(async (_, t) => {
        while (next < jobs.length) {
          const i = next++;
          const r = await this.call(t, { type: 'run', job: jobs[i] });
          out[i] = r.result!;
          onDone(i, r.result!);
        }
      }),
    );
    return out;
  }

  close(): Promise<number[]> {
    return Promise.all(this.workers.map((w) => w.terminate()));
  }
}

const sum = (rs: Reply[]): Reply => rs.reduce((a, r) => ({ loss: a.loss + r.loss, correct: a.correct + r.correct, n: a.n + r.n }), { loss: 0, correct: 0, n: 0 });

/** Cosine decay from lr to lr / 10 over the run. */
const cosine = (lr: number, t: number) => lr * (0.1 + 0.45 * (1 + Math.cos(Math.PI * Math.min(1, t))));

/** A fresh random order of 0…n−1. */
function permutation(n: number, rng: Rng): Int32Array {
  const a = new Int32Array(n);
  for (let i = 0; i < n; i++) a[i] = i;
  rng.shuffle(a);
  return a;
}

// ── Worker side ───────────────────────────────────────────────────────────

/** One full training run in a single thread (the transfer experiments). */
interface RunJob {
  label: string;
  arch: Arch;
  seed: number;
  set: string;
  /** Train on the first `train` images of the official training set. */
  train: number;
  batch: number;
  /** Phases run back to back with one optimizer, like pressing Play, then unfreezing, in the app. */
  phases: { epochs: number; lr: number; frozen: boolean[] | null }[];
  /** Start from these weights for the blocks marked in `copy` (transfer), else from scratch. */
  source: Float32Array[] | null;
  copy: boolean[] | null;
  /** Also measure test accuracy once this many training images have been seen (the head start). */
  checkpoints: number[];
}

interface RunResult {
  label: string;
  acc: number;
  loss: number;
  /** Time spent training (evaluations excluded), in seconds. */
  seconds: number;
  /** Test accuracy at each checkpoint, in order. */
  early: number[];
}

function runJob(job: RunJob, sets: Sets): RunResult {
  const split = sets[job.set];
  const net = new Network(job.arch, job.seed);
  if (job.source && job.copy) {
    net.blocks.forEach((b, i) => {
      if (!job.copy![i]) return;
      b.W.set(job.source![2 * i]);
      b.b.set(job.source![2 * i + 1]);
    });
  }
  const opt = new Optimizer(net, 'adam', job.phases[0].lr);
  const rng = new Rng(job.seed * 7919 + 3);
  const x = new Float32Array(size(job.arch.input));
  const test = (): { acc: number; loss: number } => {
    let correct = 0;
    let loss = 0;
    const t = split.test;
    for (let i = 0; i < t.n; i++) {
      const y = loadSample(t, i, x);
      const p = net.forward(x);
      if (argmax(p) === y) correct++;
      loss += -Math.log(Math.max(p[y], 1e-12));
    }
    return { acc: correct / t.n, loss: loss / t.n };
  };
  let order = permutation(job.train, rng);
  let cursor = 0;
  let seen = 0;
  let trainMs = 0;
  const early: number[] = [];
  for (const ph of job.phases) {
    opt.lr = ph.lr;
    const steps = Math.ceil((ph.epochs * job.train) / job.batch);
    for (let s = 0; s < steps; s++) {
      const t0 = performance.now();
      net.zeroGrad();
      for (let k = 0; k < job.batch; k++) {
        if (cursor >= job.train) {
          order = permutation(job.train, rng);
          cursor = 0;
        }
        const y = loadSample(split.train, order[cursor++], x);
        net.forward(x);
        net.backward(y, false, ph.frozen ?? undefined);
      }
      opt.step(1 / job.batch, ph.frozen ?? undefined);
      seen += job.batch;
      trainMs += performance.now() - t0;
      while (early.length < job.checkpoints.length && seen >= job.checkpoints[early.length]) early.push(test().acc);
    }
  }
  const final = test();
  return { label: job.label, acc: final.acc, loss: final.loss, seconds: trainMs / 1000, early };
}

function workerMain(): void {
  const { sets, arch, weights, grads } = workerData as { sets: Sets; arch: Arch | null; weights: Float32Array; grads: Float32Array };
  const net = arch ? new Network(arch, 0) : null;
  const x = arch ? new Float32Array(size(arch.input)) : new Float32Array(0);
  parentPort!.on('message', (m: WorkerJob) => {
    if (m.type === 'run') {
      parentPort!.postMessage({ loss: 0, correct: 0, n: 0, result: runJob(m.job, sets) } satisfies Reply);
      return;
    }
    const model = net!;
    unflatten(weights, model);
    let loss = 0;
    let correct = 0;
    if (m.type === 'batch') {
      const set = sets[m.set].train;
      model.zeroGrad();
      for (let k = 0; k < m.idx.length; k++) {
        const y = loadSample(set, m.idx[k], x, !!m.flips?.[k]);
        const p = model.forward(x);
        if (argmax(p) === y) correct++;
        loss += model.backward(y, false, m.frozen ?? undefined);
      }
      flatten(model, grads, true);
      parentPort!.postMessage({ loss, correct, n: m.idx.length } satisfies Reply);
    } else {
      const set = sets[m.set][m.split];
      for (let i = m.from; i < m.to; i++) {
        const y = loadSample(set, i, x);
        const p = model.forward(x);
        if (argmax(p) === y) correct++;
        loss += -Math.log(Math.max(p[y], 1e-12));
      }
      parentPort!.postMessage({ loss, correct, n: Math.max(0, m.to - m.from) } satisfies Reply);
    }
  });
}

// ── The zoo ───────────────────────────────────────────────────────────────

interface ZooSpec {
  id: string;
  name: string;
  set: SetName;
  description: string;
  layers: LayerSpec[];
  epochs: number;
  lr: number;
  batch: number;
  /** Random left–right mirroring of training images. */
  flip: boolean;
  seed: number;
}

const SMALL_CNN: LayerSpec[] = [
  { kind: 'conv', filters: 8, kernel: 3, act: 'relu', pool: true },
  { kind: 'conv', filters: 16, kernel: 3, act: 'relu', pool: true },
  { kind: 'dense', units: 32, act: 'relu' },
];

const ZOO: ZooSpec[] = [
  {
    id: 'mnist-lenet',
    name: 'MNIST LeNet',
    set: 'mnist',
    description: 'LeNet-style: two 5×5 tanh convolutions with max-pooling, then 64 tanh units. The architecture of the LeNet-ish preset.',
    layers: [
      { kind: 'conv', filters: 6, kernel: 5, act: 'tanh', pool: true },
      { kind: 'conv', filters: 16, kernel: 5, act: 'tanh', pool: true },
      { kind: 'dense', units: 64, act: 'tanh' },
    ],
    epochs: 2,
    lr: 0.002,
    batch: 32,
    flip: false,
    seed: 11,
  },
  {
    id: 'mnist-cnn',
    name: 'MNIST small CNN',
    set: 'mnist',
    description: 'The Small CNN preset (the default network): two 3×3 ReLU convolutions with max-pooling, then 32 units.',
    layers: SMALL_CNN,
    epochs: 3,
    lr: 0.002,
    batch: 32,
    flip: false,
    seed: 12,
  },
  {
    id: 'fashion-cnn',
    name: 'Fashion-MNIST small CNN',
    set: 'fashion',
    description: 'The Small CNN preset trained on clothing. Same layers as the MNIST small CNN, so either can be transferred to the other.',
    layers: SMALL_CNN,
    epochs: 4,
    lr: 0.002,
    batch: 32,
    flip: false,
    seed: 13,
  },
  {
    id: 'cifar10-cnn',
    name: 'CIFAR-10 CNN',
    set: 'cifar10',
    description: 'Three 3×3 ReLU convolutions (16, 32, 64 filters), each with max-pooling, then 64 units. Trained with random left–right mirroring.',
    layers: [
      { kind: 'conv', filters: 16, kernel: 3, act: 'relu', pool: true },
      { kind: 'conv', filters: 32, kernel: 3, act: 'relu', pool: true },
      { kind: 'conv', filters: 64, kernel: 3, act: 'relu', pool: true },
      { kind: 'dense', units: 64, act: 'relu' },
    ],
    epochs: 3,
    lr: 0.002,
    batch: 32,
    flip: true,
    seed: 14,
  },
];

const OUT = 'public/models';

async function trainZoo(spec: ZooSpec, split: Split, bundle: string, threads: number): Promise<void> {
  const arch: Arch = { input: split.train.shape, layers: spec.layers, classes: 10 };
  const net = new Network(arch, spec.seed);
  const opt = new Optimizer(net, 'adam', spec.lr);
  const pool = new Pool(bundle, threads, { [spec.set]: split }, arch);
  const rng = new Rng(spec.seed * 31 + 7);
  const n = split.train.n;
  const B = spec.batch;
  const steps = Math.ceil((spec.epochs * n) / B);
  let order = permutation(n, rng);
  let cursor = 0;
  const t0 = performance.now();
  let winLoss = 0;
  let winAcc = 0;
  let winN = 0;
  const report = Math.max(1, Math.round(steps / 40));
  const evalEvery = Math.round(n / 2 / B);
  console.log(`${spec.id}: ${layersSummary(arch)}; ${int(net.paramCount)} parameters; ${spec.epochs} epochs of ${int(n)} images on ${threads} threads`);
  for (let s = 0; s < steps; s++) {
    opt.lr = cosine(spec.lr, s / steps);
    const idx = new Int32Array(B);
    for (let k = 0; k < B; k++) {
      if (cursor >= n) {
        order = permutation(n, rng);
        cursor = 0;
      }
      idx[k] = order[cursor++];
    }
    const flips = spec.flip ? Uint8Array.from({ length: B }, () => (rng.next() < 0.5 ? 1 : 0)) : null;
    const r = await pool.batch(net, spec.set, idx, flips);
    opt.step(1 / B);
    winLoss += r.loss;
    winAcc += r.correct;
    winN += r.n;
    if ((s + 1) % report === 0) {
      const el = (performance.now() - t0) / 1000;
      console.log(`  ${spec.id} epoch ${(((s + 1) * B) / n).toFixed(2)}  loss ${(winLoss / winN).toFixed(3)}  train acc ${(winAcc / winN).toFixed(3)}  lr ${opt.lr.toExponential(1)}  ${el.toFixed(0)} s`);
      winLoss = winAcc = winN = 0;
    }
    if ((s + 1) % evalEvery === 0 && s + 1 < steps) {
      const e = await pool.evaluate(net, spec.set, 'test', 2000);
      console.log(`  ${spec.id} epoch ${(((s + 1) * B) / n).toFixed(2)}  test acc (first 2,000) ${(e.acc * 100).toFixed(2)}%`);
    }
  }
  const final = await pool.evaluate(net, spec.set, 'test', split.test.n);
  const sub = await pool.evaluate(net, spec.set, 'test', 2000);
  await pool.close();
  const seconds = (performance.now() - t0) / 1000;
  console.log(`${spec.id}: test accuracy ${(final.acc * 100).toFixed(2)}% on ${int(split.test.n)} (first 2,000, as in the app: ${(sub.acc * 100).toFixed(2)}%), ${seconds.toFixed(0)} s`);
  const file = encodeModel(net, {
    name: spec.name,
    dataset: spec.set,
    meta: {
      description: spec.description,
      trainedOn: `All ${int(n)} ${NAMES[spec.set]} training images, ${spec.epochs} epochs${spec.flip ? ' with random mirroring' : ''}; Adam, batch ${B}, learning rate ${spec.lr} decaying to ${spec.lr / 10} (cosine)`,
      samples: steps * B,
      epochs: spec.epochs,
      testAccuracy: Math.round(final.acc * 10000) / 10000,
      created: new Date().toISOString(),
    },
  });
  mkdirSync(OUT, { recursive: true });
  writeFileSync(join(OUT, `${spec.id}.json`), JSON.stringify(file) + '\n');
}

/** Rewrites index.json from the model files, in zoo order. */
function writeIndex(): ZooEntry[] {
  const entries: ZooEntry[] = [];
  for (const spec of ZOO) {
    const path = join(OUT, `${spec.id}.json`);
    if (!existsSync(path)) continue;
    const text = readFileSync(path, 'utf8');
    const { file, net } = decodeModel(JSON.parse(text));
    entries.push({
      id: spec.id,
      name: file.name,
      dataset: file.dataset,
      description: file.meta.description ?? '',
      layers: layersSummary(file.arch),
      arch: file.arch,
      params: net.paramCount,
      testAccuracy: file.meta.testAccuracy ?? 0,
      testSet: OFFICIAL_TEST[file.dataset as SetName],
      file: `${spec.id}.json`,
      bytes: Buffer.byteLength(text),
    });
    if (archParams(file.arch) !== net.paramCount) throw new Error(`${spec.id}: parameter count mismatch`);
  }
  writeFileSync(join(OUT, 'index.json'), JSON.stringify(entries, null, 2) + '\n');
  for (const e of entries) console.log(`${e.id.padEnd(12)} ${(e.testAccuracy * 100).toFixed(2)}%  ${int(e.params).padStart(7)} params  ${int(e.bytes).padStart(9)} bytes  ${e.layers}`);
  return entries;
}

// ── Transfer experiments ──────────────────────────────────────────────────

interface TransferOptions {
  seeds: number;
  /** Learning rate for every condition (the page's Adam default is 0.003). */
  lr: number;
  /** Learning rate for the second, everything-unfrozen phase of fine-tuning. */
  fineLr: number;
  batch: number;
  pairs: { from: string; to: SetName }[];
  sizes: { train: number; epochs: number }[];
  conditions: string[];
  /** Test accuracy is also measured after this many training images (the head start). */
  checkpoints: number[];
  out: string;
}

const TRANSFER: TransferOptions = {
  seeds: 3,
  lr: 0.003,
  fineLr: 0.001,
  batch: 32,
  pairs: [
    { from: 'mnist-cnn', to: 'fashion' },
    { from: 'fashion-cnn', to: 'mnist' },
  ],
  // Every condition sees 20,000 training images, whatever N is.
  sizes: [
    { train: 1000, epochs: 20 },
    { train: 200, epochs: 100 },
  ],
  conditions: ['scratch', 'frozen', 'convFrozen', 'fineTune'],
  checkpoints: [1000, 5000],
  out: join(OUT, 'transfer.json'),
};

/** Per block of the source network: which blocks to copy, and which to hold frozen in each phase. */
interface Masks {
  hidden: boolean[];
  convOnly: boolean[];
  none: boolean[];
}

type Phase = RunJob['phases'][number];

interface Condition {
  label: string;
  copy: (m: Masks) => boolean[] | null;
  phases: (m: Masks, epochs: number, o: TransferOptions) => Phase[];
}

/** The conditions compared, each as it would be done on the page. */
const CONDITIONS: Record<string, Condition> = {
  // Random weights, every layer trains.
  scratch: { label: 'scratch', copy: () => null, phases: (_, e, o) => [{ epochs: e, lr: o.lr, frozen: null }] },
  // What Transfer does: every copied hidden layer frozen, only the new output layer trains.
  frozen: { label: 'all copied layers frozen', copy: (m) => m.hidden, phases: (m, e, o) => [{ epochs: e, lr: o.lr, frozen: m.hidden }] },
  // Transfer, then unlock the dense layer: only the conv layers stay frozen.
  convFrozen: { label: 'conv frozen', copy: (m) => m.hidden, phases: (m, e, o) => [{ epochs: e, lr: o.lr, frozen: m.convOnly }] },
  // Transfer and train the new output layer, then unlock everything and go on at a lower rate.
  fineTune: {
    label: 'fine-tune',
    copy: (m) => m.hidden,
    phases: (m, e, o) => [
      { epochs: e / 2, lr: o.lr, frozen: m.hidden },
      { epochs: e / 2, lr: o.fineLr, frozen: m.none },
    ],
  },
};

const mean = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;

async function transfer(sets: Sets, bundle: string, threads: number, o: TransferOptions): Promise<void> {
  const jobs: RunJob[] = [];
  const rows: { pair: TransferOptions['pairs'][number]; size: TransferOptions['sizes'][number]; idx: Record<string, number[]> }[] = [];
  for (const key of o.conditions) if (!CONDITIONS[key]) throw new Error(`Unknown condition ${key}; known: ${Object.keys(CONDITIONS).join(', ')}`);
  for (const pair of o.pairs) {
    const { file, net: src } = decodeModel(JSON.parse(readFileSync(join(OUT, `${pair.from}.json`), 'utf8')));
    const arch: Arch = { ...file.arch, classes: 10 };
    const B = src.blocks.length;
    const masks: Masks = {
      hidden: Array.from({ length: B }, (_, i) => i < B - 1),
      convOnly: src.blocks.map((b) => b.kind === 'conv'),
      none: Array.from({ length: B }, () => false),
    };
    const source = src.getWeights();
    for (const size of o.sizes) {
      const idx: Record<string, number[]> = Object.fromEntries(o.conditions.map((c) => [c, []]));
      for (let s = 0; s < o.seeds; s++) {
        const seed = 101 + s;
        const tag = `${pair.from} → ${pair.to}, ${size.train} images, seed ${seed}`;
        for (const key of o.conditions) {
          const c = CONDITIONS[key];
          const copy = c.copy(masks);
          idx[key].push(jobs.length);
          jobs.push({ arch, seed, set: pair.to, train: size.train, batch: o.batch, label: `${tag}: ${c.label}`, phases: c.phases(masks, size.epochs, o), source: copy ? source : null, copy, checkpoints: o.checkpoints });
        }
      }
      rows.push({ pair, size, idx });
    }
  }
  console.log(`${jobs.length} runs on ${threads} threads`);
  const pool = new Pool(bundle, threads, sets, null);
  const results = await pool.runAll(jobs, (_, r) => console.log(`  ${r.label}: ${(r.acc * 100).toFixed(2)}% (${r.seconds.toFixed(0)} s)`));
  await pool.close();
  const r4 = (v: number) => Math.round(v * 10000) / 10000;
  const cell = (ids: number[]): TransferCell => {
    const runs = ids.map((i) => r4(results[i].acc));
    const early = o.checkpoints.map((seen, k) => {
      const at = ids.map((i) => r4(results[i].early[k]));
      return { seen, runs: at, mean: r4(mean(at)) };
    });
    return { runs, mean: r4(mean(runs)), seconds: Math.round(mean(ids.map((i) => results[i].seconds)) * 10) / 10, early };
  };
  const seen = o.sizes.map((s) => s.train * s.epochs);
  const report: TransferReport = {
    created: new Date().toISOString(),
    seeds: o.seeds,
    checkpoints: o.checkpoints,
    testSet: { mnist: OFFICIAL_TEST.mnist, fashion: OFFICIAL_TEST.fashion },
    settings:
      `Training images: the first N of the official training set, ${seen.every((v) => v === seen[0]) ? `${int(seen[0])} training images seen in every run` : 'a fixed number of epochs per N'}. ` +
      `Adam, batch ${o.batch}, learning rate ${o.lr} (the page's default). Fine-tuning trains only the new output layer for the first half, ` +
      `then every layer at ${o.fineLr}. Seeds change the new layers' starting weights and the shuffling.`,
    rows: rows.map(
      (r) =>
        ({
          from: r.pair.from,
          to: r.pair.to,
          train: r.size.train,
          epochs: r.size.epochs,
          ...Object.fromEntries(Object.entries(r.idx).map(([k, ids]) => [k, cell(ids)])),
        }) as TransferRow,
    ),
  };
  mkdirSync(dirname(o.out), { recursive: true });
  writeFileSync(o.out, JSON.stringify(report, null, 2) + '\n');
  console.log(`wrote ${o.out}`);
  const p = (c?: TransferCell) => (c ? `${(c.mean * 100).toFixed(1)}% (${c.early!.map((e) => (e.mean * 100).toFixed(1)).join(', ')}; ${c.seconds} s)` : '–');
  const keys = o.conditions;
  console.log(`\nFinal test accuracy (after ${o.checkpoints.map(int).join(', ')} images; training seconds per run)`);
  console.log(`| From → to | Training images | ${keys.map((k) => CONDITIONS[k].label).join(' | ')} |`);
  console.log(`| --- | ---: | ${keys.map(() => '---:').join(' | ')} |`);
  for (const r of report.rows) {
    const row = r as unknown as Record<string, TransferCell>;
    console.log(`| ${r.from} → ${NAMES[r.to as SetName]} | ${int(r.train)} | ${keys.map((k) => p(row[k])).join(' | ')} |`);
  }
}

/** Transfer settings from the command line: --seeds, --lr, --fine-lr, --sizes 1000:20,200:100, --pairs, --conditions, --out. */
function transferOptions(flags: Record<string, string>): TransferOptions {
  const o = { ...TRANSFER };
  if (flags.seeds) o.seeds = Math.max(1, Number(flags.seeds));
  if (flags.lr) o.lr = Number(flags.lr);
  if (flags['fine-lr']) o.fineLr = Number(flags['fine-lr']);
  if (flags.sizes) o.sizes = flags.sizes.split(',').map((s) => ({ train: Number(s.split(':')[0]), epochs: Number(s.split(':')[1]) }));
  if (flags.pairs) o.pairs = TRANSFER.pairs.filter((p) => flags.pairs.split(',').includes(p.from));
  if (flags.conditions) o.conditions = flags.conditions.split(',');
  if (flags.checkpoints) o.checkpoints = flags.checkpoints.split(',').map(Number);
  if (flags.out) o.out = flags.out;
  return o;
}

// ── Command line ──────────────────────────────────────────────────────────

const int = (v: number) => Math.round(v).toLocaleString('en-US');

function args() {
  const argv = process.argv.slice(2);
  const flags: Record<string, string> = {};
  const rest: string[] = [];
  for (let i = 0; i < argv.length; i++) {
    if (argv[i].startsWith('--')) flags[argv[i].slice(2)] = argv[++i];
    else rest.push(argv[i]);
  }
  return { cmd: rest[0], rest: rest.slice(1), flags };
}

function loaders(flags: Record<string, string>): Record<SetName, () => Split> {
  const raw = flags.raw ?? process.env.RASTER_RAW ?? 'data-raw';
  return {
    mnist: () => loadIdxDir(flags.mnist ?? join(raw, 'mnist-raw')),
    fashion: () => loadIdxDir(flags.fashion ?? join(raw, 'fashion-raw')),
    cifar10: () => loadCifar(flags.cifar ?? join(raw, 'cifar-raw', 'cifar-10-binary.tar.gz')),
  };
}

/** Bundles this file for the worker threads (Node cannot load the TypeScript sources directly). */
async function bundleSelf(): Promise<string> {
  const { buildSync } = await import('esbuild');
  const self = fileURLToPath(import.meta.url);
  const out = join(mkdtempSync(join(tmpdir(), 'raster-pretrain-')), 'worker.mjs');
  buildSync({ entryPoints: [self], bundle: true, platform: 'node', format: 'esm', outfile: out, external: ['esbuild'], logLevel: 'error' });
  return out;
}

async function main(): Promise<void> {
  const { cmd, rest, flags } = args();
  process.chdir(resolve(dirname(fileURLToPath(import.meta.url)), '..'));
  const threads = Math.max(1, Number(flags.threads ?? cpus().length - 1));
  const load = loaders(flags);
  const cpu0 = process.cpuUsage();
  const t0 = performance.now();
  if (cmd === 'check') {
    for (const id of ['mnist', 'fashion', 'cifar10'] as SetName[]) {
      const s = load[id]();
      console.log(`${id}: ${int(s.train.n)} training and ${int(s.test.n)} test images, ${s.train.shape.h}×${s.train.shape.w}×${s.train.shape.c}`);
      console.log('  ' + crossCheck(id, s));
    }
  } else if (cmd === 'train') {
    const ids = rest.length ? rest : ZOO.map((z) => z.id);
    const bundle = await bundleSelf();
    for (const id of ids) {
      const spec = ZOO.find((z) => z.id === id);
      if (!spec) throw new Error(`Unknown model ${id}; the zoo has ${ZOO.map((z) => z.id).join(', ')}`);
      const split = load[spec.set]();
      console.log(crossCheck(spec.set, split));
      await trainZoo(spec, split, bundle, threads);
    }
    writeIndex();
  } else if (cmd === 'transfer') {
    const sets: Sets = { mnist: load.mnist(), fashion: load.fashion() };
    await transfer(sets, await bundleSelf(), threads, transferOptions(flags));
  } else if (cmd === 'index') {
    writeIndex();
  } else {
    console.log('Usage: npx vite-node scripts/pretrain.ts check | train [ids…] | transfer [--seeds N] | index   [--threads N] [--raw DIR]');
    process.exitCode = 1;
    return;
  }
  const cpu = process.cpuUsage(cpu0);
  console.log(`done in ${((performance.now() - t0) / 1000).toFixed(0)} s wall, ${((cpu.user + cpu.system) / 1e6 / 60).toFixed(1)} CPU-minutes`);
}

if (isMainThread) {
  main().catch((e: unknown) => {
    console.error(e instanceof Error ? e.message : e);
    process.exit(1);
  });
} else if ((workerData as { role?: string } | null)?.role === 'pretrain-worker') {
  workerMain();
}
