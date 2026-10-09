import { existsSync, readFileSync, statSync } from 'node:fs';
import { inflateSync } from 'node:zlib';
import { beforeAll, describe as suite, expect, it, vi } from 'vitest';
import { datasetInfo } from '../src/data/datasets';
import { decodeModel, encodeModel } from '../src/models/format';
import { archParams, controlFinding, layersSummary, parseIndex, shapeWords, transferFinding, type TransferReport, type ZooEntry } from '../src/models/zoo';
import { argmax, blockSignature, Network } from '../src/nn/network';
import type { Arch } from '../src/nn/types';

/**
 * The pretrained model zoo (public/models, written by scripts/pretrain.ts) and the transfer and
 * load actions the model panel calls.
 */

const DIR = 'public/models';
const index = (): ZooEntry[] => parseIndex(JSON.parse(readFileSync(`${DIR}/index.json`, 'utf8')));
const fileOf = (e: ZooEntry) => JSON.parse(readFileSync(`${DIR}/${e.file}`, 'utf8')) as unknown;

/** 8-bit greyscale PNG reader for the app's MNIST sprites (as in tests/data.test.ts). */
function readPng(path: string) {
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
    }
    if (type === 'IDAT') idat.push(data);
    off += 12 + len;
  }
  const raw = inflateSync(Buffer.concat(idat));
  const px = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) {
    expect(raw[y * (width + 1)]).toBe(0); // filter: none
    px.set(raw.subarray(y * (width + 1) + 1, (y + 1) * (width + 1)), y * width);
  }
  return { width, px };
}

suite('pretrained model zoo', () => {
  it('has a well-formed index whose entries match their files', () => {
    const entries = index();
    expect(entries.map((e) => e.id)).toEqual(['mnist-lenet', 'mnist-cnn', 'fashion-cnn', 'cifar10-cnn']);
    expect(new Set(entries.map((e) => e.file)).size).toBe(entries.length);
    for (const e of entries) {
      expect(existsSync(`${DIR}/${e.file}`), e.file).toBe(true);
      expect(statSync(`${DIR}/${e.file}`).size, `${e.id}: bytes`).toBe(e.bytes);
      expect(e.bytes, `${e.id}: at most about 1.5 MB`).toBeLessThanOrEqual(1_600_000);
      expect(e.testAccuracy).toBeGreaterThan(0.5);
      expect(e.testAccuracy).toBeLessThanOrEqual(1);
      expect(e.testSet).toMatch(/^10,000 official .* test images$/);
      expect(e.layers).toBe(layersSummary(e.arch));
      expect(e.params).toBe(archParams(e.arch));
      const info = datasetInfo(e.dataset);
      expect(info.kind).toBe('image');
      expect(e.arch.input).toEqual(info.image!.shape);
      expect(e.arch.classes).toBe(info.classes.length);
    }
  });

  it('every file decodes, matches its index entry and holds finite weights', () => {
    for (const e of index()) {
      const { file, net } = decodeModel(fileOf(e));
      expect(file.name).toBe(e.name);
      expect(file.dataset).toBe(e.dataset);
      expect(file.arch).toEqual(e.arch);
      expect(net.paramCount).toBe(e.params);
      expect(file.meta.testAccuracy).toBe(e.testAccuracy);
      expect(file.meta.trainedOn).toMatch(/training images/);
      for (const w of net.getWeights()) {
        expect(w.every(Number.isFinite)).toBe(true);
        expect(w.some((v) => v !== 0) || w.length < 100).toBe(true);
      }
    }
  });

  it('the documented accuracies are what the zoo says', () => {
    const acc = Object.fromEntries(index().map((e) => [e.id, e.testAccuracy]));
    expect(acc['mnist-lenet']).toBeGreaterThanOrEqual(0.98);
    expect(acc['mnist-cnn']).toBeGreaterThanOrEqual(0.98);
    expect(acc['fashion-cnn']).toBeGreaterThanOrEqual(0.87);
    expect(acc['cifar10-cnn']).toBeGreaterThanOrEqual(0.6);
  });

  it('mnist-cnn classifies the app’s own 2,000 MNIST test digits (decoded from public/data) at ≥ 97%', () => {
    const entry = index().find((e) => e.id === 'mnist-cnn')!;
    const { net } = decodeModel(fileOf(entry));
    const labels = readFileSync('public/data/mnist-labels.txt', 'utf8').trim();
    const png = readPng('public/data/mnist-test.png');
    const x = new Float32Array(784);
    let correct = 0;
    for (let i = 0; i < 2000; i++) {
      const ox = (i % 100) * 28;
      const oy = Math.floor(i / 100) * 28;
      // Same scaling as the app: stored byte × (1/255).
      for (let r = 0; r < 28; r++) for (let c = 0; c < 28; c++) x[r * 28 + c] = png.px[(oy + r) * png.width + ox + c] * (1 / 255);
      if (argmax(net.forward(x)) === Number(labels[20_000 + i])) correct++;
    }
    expect(correct / 2000).toBeGreaterThanOrEqual(0.97);
  });

  it('records the transfer experiment with every condition measured', () => {
    const t = JSON.parse(readFileSync(`${DIR}/transfer.json`, 'utf8')) as TransferReport;
    expect(t.seeds).toBeGreaterThanOrEqual(2);
    expect(t.rows.map((r) => `${r.from}>${r.to}:${r.train}`)).toEqual(['mnist-cnn>fashion:1000', 'mnist-cnn>fashion:200', 'fashion-cnn>mnist:1000', 'fashion-cnn>mnist:200']);
    const avg = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;
    for (const r of t.rows) {
      // Every condition saw the same number of training images.
      expect(r.train * r.epochs).toBe(t.rows[0].train * t.rows[0].epochs);
      for (const c of [r.scratch, r.frozen, r.convFrozen, r.fineTune, r.randomConv!]) {
        expect(c.runs).toHaveLength(t.seeds);
        expect(c.mean).toBeCloseTo(avg(c.runs), 3);
        expect(c.mean).toBeGreaterThan(0.3);
        expect(c.seconds).toBeGreaterThan(0);
        expect(c.early!.map((e) => e.seen)).toEqual(t.checkpoints);
        for (const e of c.early!) {
          expect(e.runs).toHaveLength(t.seeds);
          expect(e.mean).toBeCloseTo(avg(e.runs), 3);
        }
      }
    }
  });

  it('words the measured transfer results from the numbers', () => {
    const t = JSON.parse(readFileSync(`${DIR}/transfer.json`, 'utf8')) as TransferReport;
    const r = t.rows[0];
    const text = transferFinding(r, 'MNIST small CNN', 'Fashion-MNIST', t.checkpoints![0]);
    expect(text).toContain(`with every copied layer frozen, ${(r.frozen.mean * 100).toFixed(1)}%`);
    expect(text).toContain(`with the dense layer unlocked as well, ${(r.convFrozen.mean * 100).toFixed(1)}%`);
    expect(text).toContain(`training from scratch (${(r.scratch.mean * 100).toFixed(1)}%)`);
    expect(text).not.toMatch(/-\d/); // true minus signs only
    expect(controlFinding(r, 'MNIST', 'Fashion')).toContain(`reach ${(r.randomConv!.mean * 100).toFixed(1)}%`);

    // Synthetic rows exercise each wording.
    const cell = (mean: number, seconds = 10, early = mean - 0.2) => ({ runs: [mean], mean, seconds, early: [{ seen: 1000, runs: [early], mean: early }] });
    const row = { from: 'a', to: 'mnist' as const, train: 200, epochs: 100, scratch: cell(0.8, 20, 0.4), frozen: cell(0.6, 8), convFrozen: cell(0.85, 10, 0.7), fineTune: cell(0.84), randomConv: cell(0.75) };
    const s = transferFinding(row, 'A', 'MNIST', 1000);
    expect(s).toContain('5.0 points above training from scratch (80.0%), in about half the training time.');
    expect(s).toContain('The copied dense layer had specialised in the old classes');
    expect(s).toContain('after the first 1,000 images, 70.0% against 40.0% from scratch.');
    expect(transferFinding({ ...row, convFrozen: cell(0.805, 19) }, 'A', 'MNIST')).toContain('level with training from scratch (80.0%).');
    expect(transferFinding({ ...row, convFrozen: cell(0.77, 19) }, 'A', 'MNIST')).toContain('3.0 points below training from scratch');
    expect(controlFinding(row, 'A', 'MNIST')).toContain('worth 10.0 points here');
    expect(controlFinding({ ...row, randomConv: cell(0.845) }, 'A', 'MNIST')).toContain('add little over random ones');
    expect(controlFinding({ ...row, randomConv: undefined }, 'A', 'MNIST')).toBeNull();
  });

  it('summarises layers in sentence case', () => {
    expect(layersSummary({ input: { c: 1, h: 28, w: 28 }, layers: [{ kind: 'conv', filters: 6, kernel: 5, act: 'tanh', pool: true }, { kind: 'dense', units: 64, act: 'leaky' }, { kind: 'dense', units: 8, act: 'relu' }], classes: 10 })).toBe(
      'conv 5×5×6 tanh, pool · dense 64 leaky ReLU · dense 8 ReLU',
    );
  });

  it('describes input shapes in words', () => {
    expect(shapeWords({ c: 1, h: 28, w: 28 })).toBe('28×28 grey');
    expect(shapeWords({ c: 3, h: 32, w: 32 })).toBe('32×32 colour');
    expect(shapeWords({ c: 5, h: 1, w: 1 })).toBe('5 features');
  });

  it('rejects a malformed index with a readable message', () => {
    expect(() => parseIndex({})).toThrow('The model list is not a list.');
    expect(() => parseIndex([{ id: 'x' }])).toThrow(/Entry 1 of the model list has no name/);
    const ok = index()[0];
    expect(() => parseIndex([{ ...ok, file: '../secret.json' }])).toThrow(/unexpected file/);
  });
});

suite('loading and transferring models (the actions the model panel calls)', () => {
  type Actions = typeof import('../src/actions');
  type Store = typeof import('../src/store')['store'];
  let actions: Actions;
  let store: Store;

  beforeAll(async () => {
    // The page's trainer and analysis clients fall back to running in this thread (Node has no
    // Web Worker); weight updates are batched with requestAnimationFrame.
    vi.stubGlobal('requestAnimationFrame', (fn: () => void) => setTimeout(fn, 0));
    actions = await import('../src/actions');
    store = (await import('../src/store')).store;
  });

  const zooFile = (id: string) => fileOf(index().find((e) => e.id === id)!);

  it('transfer copies exactly the compatible hidden layers, freezes them and gives a fresh output layer', () => {
    store.dataset = 'fashion';
    const json = zooFile('mnist-lenet');
    const { net: src, file } = decodeModel(json);
    actions.transferModel(json);
    expect(store.spec).toEqual(file.arch.layers);
    expect(store.frozen).toEqual([true, true, true, false]);
    const net = store.net;
    net.blocks.forEach((b, i) => {
      const s = src.blocks[i];
      expect(blockSignature(b)).toBe(blockSignature(s));
      const same = b.W.every((v, j) => v === s.W[j]) && b.b.every((v, j) => v === s.b[j]);
      expect(same, `block ${i}`).toBe(i < net.blocks.length - 1);
    });
    expect(store.modelName).toContain('MNIST LeNet');
    expect(store.keepWeights).toBe(true);
  });

  it('a frozen layer keeps its weights in training while the new head learns', () => {
    const arch = store.arch;
    const net = new Network(arch, 5);
    net.setWeights(store.net.getWeights());
    const before = net.getWeights();
    const frozen = store.frozen.slice();
    // One Adam step on a random image, as the trainer does it.
    return import('../src/nn/optim').then(({ Optimizer }) => {
      const opt = new Optimizer(net, 'adam', 0.01);
      net.zeroGrad();
      const x = Float32Array.from({ length: 784 }, (_, i) => ((i * 7919) % 255) / 255);
      net.forward(x);
      net.backward(3, false, frozen);
      opt.step(1, frozen);
      const after = net.getWeights();
      net.blocks.forEach((_, i) => {
        const moved = after[2 * i].some((v, j) => v !== before[2 * i][j]);
        expect(moved, `block ${i}`).toBe(!frozen[i]);
      });
    });
  });

  it('transfer refuses an input of another shape, in words', () => {
    store.dataset = 'cifar10';
    expect(() => actions.transferModel(zooFile('mnist-cnn'))).toThrow('MNIST small CNN expects 28×28×1 inputs, but CIFAR-10 has 32×32×3.');
  });

  it('load on the same dataset uses the exact architecture and weights', async () => {
    store.dataset = 'mnist';
    const json = zooFile('mnist-cnn');
    const { net: src } = decodeModel(json);
    await actions.loadModel(json);
    expect(store.net.getWeights()).toEqual(src.getWeights());
    expect(store.frozen.every((f) => !f)).toBe(true);
    expect(store.modelName).toBe('MNIST small CNN');
  });

  it('keep-weights edits copy the unchanged layers only', () => {
    store.dataset = 'mnist';
    const before = store.net.getWeights();
    actions.setKeepWeights(true);
    actions.setSpec(store.spec.map((l) => (l.kind === 'dense' ? { ...l, units: 64 } : l)));
    const after = store.net.getWeights();
    expect(after[0]).toEqual(before[0]); // conv 1
    expect(after[2]).toEqual(before[2]); // conv 2
    expect(after[4].length).not.toBe(before[4].length); // dense 3 resized: fresh
  });

  it('a saved model round-trips through the file format', () => {
    const file = actions.exportModel('Round trip');
    const { net, file: back } = decodeModel(JSON.parse(JSON.stringify(file)));
    expect(back.name).toBe('Round trip');
    expect(net.getWeights()).toEqual(store.net.getWeights());
    const arch: Arch = back.arch;
    expect(arch).toEqual(store.arch);
    expect(encodeModel(net, { name: 'x', dataset: 'mnist', meta: {} }).weights).toEqual(file.weights);
  });
});
