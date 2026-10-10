// Which default network should the page use for CIFAR-10?
// Usage: npx vite-node scripts/cifar-defaults.ts <dir with data_batch_1.bin and test_batch.bin> '<configs JSON>' [samples] [seed]
// Trains on the page's own subset (first 10,000 training images, first 2,000 test images) with the
// page's engine and batch size, and reports test accuracy and dead units in the dense layer.
import { readFileSync } from 'node:fs';
import { Network } from '../src/nn/network';
import { Optimizer } from '../src/nn/optim';
import { Rng } from '../src/nn/rng';
import type { Arch, LayerSpec } from '../src/nn/types';

const dir = process.argv[2];
const N = 3073;
function load(file: string, n: number) {
  const buf = readFileSync(`${dir}/${file}`);
  const x = new Float32Array(n * 3072);
  const y = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    y[i] = buf[i * N];
    for (let j = 0; j < 3072; j++) x[i * 3072 + j] = buf[i * N + 1 + j] / 255;
  }
  return { x, y };
}
const train = load('data_batch_1.bin', 10_000);
const test = load('test_batch.bin', 2_000);

const configs: { name: string; layers: LayerSpec[]; lr: number }[] = JSON.parse(process.argv[3]);
const samples = Number(process.argv[4] ?? 20_000);
const seed = Number(process.argv[5] ?? 1);

for (const c of configs) {
  const arch: Arch = { input: { c: 3, h: 32, w: 32 }, layers: c.layers, classes: 10 };
  const net = new Network(arch, seed);
  const opt = new Optimizer(net, 'adam', c.lr);
  const rng = new Rng(seed + 7);
  const B = 32;
  const order = Int32Array.from({ length: 10_000 }, (_, i) => i);
  let pos = order.length;
  const t0 = performance.now();
  const curve: string[] = [];
  for (let seen = 0; seen < samples; seen += B) {
    net.zeroGrad();
    for (let k = 0; k < B; k++) {
      if (pos >= order.length) {
        for (let i = order.length - 1; i > 0; i--) {
          const j = Math.floor(rng.next() * (i + 1));
          [order[i], order[j]] = [order[j], order[i]];
        }
        pos = 0;
      }
      const i = order[pos++];
      net.forward(train.x.subarray(i * 3072, (i + 1) * 3072));
      net.backward(train.y[i]);
    }
    opt.step(1 / B);
    if ((seen / B) % 200 === 0) curve.push(`${seen}:${evalAcc(net, 500).acc.toFixed(2)}`);
  }
  const { acc, dead, units } = evalAcc(net, 2000);
  console.log(JSON.stringify({ name: c.name, lr: c.lr, params: net.paramCount, testAcc: +acc.toFixed(4), deadDense: `${dead}/${units}`, sec: Math.round((performance.now() - t0) / 1000), curve: curve.join(' ') }));
}

function evalAcc(net: Network, n: number) {
  let correct = 0;
  const dense = net.blocks[net.blocks.length - 2];
  const fired = new Uint8Array(dense.out.length);
  for (let i = 0; i < n; i++) {
    const p = net.forward(test.x.subarray(i * 3072, (i + 1) * 3072));
    let best = 0;
    for (let k = 1; k < 10; k++) if (p[k] > p[best]) best = k;
    if (best === test.y[i]) correct++;
    for (let u = 0; u < fired.length; u++) if (dense.z[u] > 0) fired[u] = 1;
  }
  let dead = 0;
  for (const f of fired) if (!f) dead++;
  return { acc: correct / n, dead, units: fired.length };
}
