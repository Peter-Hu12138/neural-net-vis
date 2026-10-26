import { describe, expect, it } from 'vitest';
import { pointsData } from '../src/data/datasets';
import { Network, argmax } from '../src/nn/network';
import { Optimizer } from '../src/nn/optim';
import { Rng } from '../src/nn/rng';
import type { Arch, Hyper } from '../src/nn/types';
import type { DataPayload, EvalPoint, FromTrainer, Status, TrainPoint } from '../src/train/protocol';
import { Trainer } from '../src/train/trainer';

/**
 * Regression tests for the third review's training findings: what the page reports after training
 * stops (DATA-1, ANA-1), where +1 Epoch stops (DATA-5) and the optimiser state of layers that are
 * unfrozen mid-run (MOD-1).
 */

/** The pace the page's worker uses (src/train/worker.ts): curve points ≥ 50 ms apart, evaluations ≥ 250 ms. */
const WORKER_PACE = { pointMs: 50, evalMs: 250 };

function payload(id: 'blobs' | 'circle' | 'blobs3', count: number, seed = 1): { data: DataPayload; arch: Arch } {
  const d = pointsData({ id, count, noise: 0.1, trainRatio: 0.5, seed });
  const classes = d.info.classes.length;
  return {
    data: { inputSize: d.inputSize, scale: 1, classes, trainX: d.trainX, trainY: d.trainY, testX: d.testX, testY: d.testY },
    arch: { input: d.input, layers: [{ kind: 'dense', units: 8, act: 'tanh' }, { kind: 'dense', units: 8, act: 'tanh' }], classes },
  };
}

function harness(arch: Arch, data: DataPayload, hyper: Hyper, pace = WORKER_PACE, frozen: boolean[] = []) {
  const log: FromTrainer[] = [];
  const t = new Trainer((m) => log.push(m), 5, pace);
  t.handle({ type: 'model', version: 1, arch, weights: new Network(arch, 1).getWeights(), hyper, frozen });
  t.handle({ type: 'data', data });
  const statuses = () => log.flatMap((m) => (m.type === 'status' ? [m.status] : []));
  const last = (): Status => statuses().at(-1)!;
  const evals = (): EvalPoint[] => log.flatMap((m) => (m.type === 'metrics' ? m.evals : []));
  const points = (): TrainPoint[] => log.flatMap((m) => (m.type === 'metrics' ? m.points : []));
  const weights = () => log.flatMap((m) => (m.type === 'weights' ? [m] : []));
  return { t, log, last, evals, points, weights };
}

async function until(pred: () => boolean, ms = 10_000) {
  const t0 = Date.now();
  while (!pred()) {
    if (Date.now() - t0 > ms) throw new Error('timed out');
    await new Promise((r) => setTimeout(r, 2));
  }
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** Test accuracy and confusion matrix of these exact weights, computed here from scratch. */
function evaluate(arch: Arch, weights: Float32Array[], d: DataPayload): { acc: number; confusion: number[] } {
  const net = new Network(arch, 0);
  net.setWeights(weights);
  const K = d.classes;
  const confusion = new Array(K * K).fill(0);
  let correct = 0;
  const x = new Float32Array(d.inputSize);
  for (let i = 0; i < d.testY.length; i++) {
    for (let j = 0; j < d.inputSize; j++) x[j] = d.testX[i * d.inputSize + j] * d.scale;
    const p = argmax(net.forward(x));
    if (p === d.testY[i]) correct++;
    confusion[d.testY[i] * K + p]++;
  }
  return { acc: correct / d.testY.length, confusion };
}

describe('DATA-1 / ANA-1: when training stops, the page gets a curve point and an evaluation of the weights it stopped at', () => {
  it('+1 Epoch on a small point set: the last evaluation and curve point are at the final step and match its weights', async () => {
    const { data, arch } = payload('blobs3', 600);
    const h = harness(arch, data, { lr: 0.03, batchSize: 10, optimizer: 'adam' });
    await until(() => h.evals().length === 1); // the untrained model, at step 0
    h.t.handle({ type: 'epoch' });
    await until(() => h.last().epoch === 1 && !h.last().running);
    const step = h.last().step;
    expect(step).toBe(30); // 300 training points, batches of 10
    // The epoch takes a few milliseconds, far below the 250 ms between periodic evaluations;
    // the closing evaluation must still land, at the step training stopped at.
    await until(() => h.evals().at(-1)!.step === step, 3000);
    const ev = h.evals().at(-1)!;
    expect(ev.epoch).toBe(1);
    expect(h.points().at(-1)!.step).toBe(step);
    expect(h.points().at(-1)!.epoch).toBe(1);
    const w = h.weights().at(-1)!;
    expect(w.step).toBe(step);
    const truth = evaluate(arch, w.weights, data);
    expect(ev.acc).toBe(truth.acc);
    expect(ev.confusion).toEqual(truth.confusion);
  });

  it('Pause during play: the evaluation catches up with the paused weights, and pausing itself is immediate', async () => {
    const { data, arch } = payload('circle', 600);
    const h = harness(arch, data, { lr: 0.03, batchSize: 10, optimizer: 'adam' });
    await until(() => h.evals().length === 1);
    h.t.handle({ type: 'play' });
    await sleep(120);
    const before = h.log.length;
    h.t.handle({ type: 'pause' });
    // The paused status goes out synchronously, before any evaluation work.
    const paused = h.log.slice(before).find((m) => m.type === 'status') as { status: Status } | undefined;
    expect(paused?.status.running).toBe(false);
    const step = paused!.status.step;
    expect(step).toBeGreaterThan(5);
    expect(h.points().at(-1)!.step).toBe(step);
    await until(() => h.evals().at(-1)!.step === step, 3000);
    await sleep(20);
    expect(h.last().step).toBe(step); // the evaluation trains nothing
    const truth = evaluate(arch, h.weights().at(-1)!.weights, data);
    expect(h.evals().at(-1)!.acc).toBe(truth.acc);
    expect(h.evals().at(-1)!.confusion).toEqual(truth.confusion);
  });

  it('Step: every step ends with an evaluation of the new weights, even during a running evaluation', async () => {
    const { data, arch } = payload('blobs', 600);
    const h = harness(arch, data, { lr: 0.03, batchSize: 10, optimizer: 'adam' });
    await until(() => h.evals().length === 1);
    h.t.handle({ type: 'step' });
    h.t.handle({ type: 'step' }); // before the first step's evaluation has run
    await until(() => h.evals().at(-1)!.step === 2, 3000);
    const steps = h.evals().map((e) => e.step);
    expect(steps).not.toContain(1); // the interrupted evaluation of step 1 is dropped, never mixed
    const truth = evaluate(arch, h.weights().at(-1)!.weights, data);
    expect(h.evals().at(-1)!.acc).toBe(truth.acc);
  });

  it('a second Pause without training in between adds nothing', async () => {
    const { data, arch } = payload('blobs', 600);
    const h = harness(arch, data, { lr: 0.03, batchSize: 10, optimizer: 'adam' });
    await until(() => h.evals().length === 1);
    h.t.handle({ type: 'step' });
    await until(() => h.evals().at(-1)!.step === 1);
    const n = [h.evals().length, h.points().length];
    h.t.handle({ type: 'pause' });
    await sleep(40);
    expect([h.evals().length, h.points().length]).toEqual(n);
  });
});

describe('DATA-5: +1 Epoch stops exactly at the epoch boundary', () => {
  it('200 samples in batches of 32: six full batches and one of 8, three times over', async () => {
    const { data, arch } = payload('blobs', 400); // 200 training points
    expect(data.trainY.length).toBe(200);
    const h = harness(arch, data, { lr: 0.01, batchSize: 32, optimizer: 'adam' }, { pointMs: 0, evalMs: 0 });
    for (let e = 1; e <= 3; e++) {
      h.t.handle({ type: 'epoch' });
      await until(() => h.last().epoch === e && !h.last().running);
      expect(h.last().epochFraction).toBe(e);
      expect(h.last().seen).toBe(200 * e);
      expect(h.last().step).toBe(7 * e);
    }
  });

  it('the partial last batch is averaged over its real size', () => {
    // Five identical samples, batches of 4: the second step has one sample. Every sample has the
    // same gradient, so a correctly averaged update equals one from a full batch.
    const arch: Arch = { input: { c: 2, h: 1, w: 1 }, layers: [{ kind: 'dense', units: 3, act: 'tanh' }], classes: 2 };
    const x = new Float32Array([0.3, -0.7]);
    const data: DataPayload = {
      inputSize: 2,
      scale: 1,
      classes: 2,
      trainX: Float32Array.from({ length: 10 }, (_, i) => x[i % 2]),
      trainY: new Uint8Array(5).fill(1),
      testX: x.slice(),
      testY: new Uint8Array([1]),
    };
    const lr = 0.5;
    const h = harness(arch, data, { lr, batchSize: 4, optimizer: 'sgd' }, { pointMs: 0, evalMs: 0 });
    const ref = new Network(arch, 1);
    const opt = new Optimizer(ref, 'sgd', lr);
    for (const n of [4, 1]) {
      h.t.handle({ type: 'step' });
      ref.zeroGrad();
      for (let k = 0; k < n; k++) {
        ref.forward(x);
        ref.backward(1, false);
      }
      opt.step(1 / n);
    }
    expect(h.last().seen).toBe(5);
    expect(h.last().epochFraction).toBe(1);
    const got = h.weights().at(-1)!.weights;
    ref.getWeights().forEach((w, i) => w.forEach((v, j) => expect(got[i][j]).toBeCloseTo(v, 6)));
  });
});

describe('MOD-1: a layer unfrozen mid-training starts with a fresh optimiser state', () => {
  const arch: Arch = { input: { c: 4, h: 1, w: 1 }, layers: [{ kind: 'dense', units: 6, act: 'tanh' }], classes: 3 };
  const rng = new Rng(5);
  const samples = Array.from({ length: 64 }, () => ({ x: Float32Array.from({ length: 4 }, () => rng.next() * 2 - 1), y: rng.int(3) }));

  function train(net: Network, opt: Optimizer, steps: number, frozen?: boolean[], from = 0) {
    for (let s = 0; s < steps; s++) {
      net.zeroGrad();
      for (let k = 0; k < 8; k++) {
        const p = samples[(from + s * 8 + k) % samples.length];
        net.forward(p.x);
        net.backward(p.y, false, frozen);
      }
      opt.step(1 / 8, frozen);
    }
  }

  /** Largest |Δw| / lr of block 0 over one more step. */
  function firstStepRatio(net: Network, opt: Optimizer, lr: number): number {
    const before = net.blocks[0].W.slice();
    train(net, opt, 1, [false, false]);
    let max = 0;
    net.blocks[0].W.forEach((v, i) => (max = Math.max(max, Math.abs(v - before[i]))));
    return max / lr;
  }

  it('Adam: after 300 frozen steps the unlocked layer moves by at most the learning rate, as with a fresh optimiser', () => {
    const lr = 0.001;
    const net = new Network(arch, 2);
    const opt = new Optimizer(net, 'adam', 0.003);
    train(net, opt, 300, [true, false]);
    opt.lr = lr;
    // A fresh Adam's first step moves every weight by lr·|ĝ|/(|ĝ| + ε) ≤ lr.
    expect(firstStepRatio(net, opt, lr)).toBeLessThanOrEqual(1.001); // float32 rounding of the weights
    expect(firstStepRatio(net, opt, lr)).toBeLessThanOrEqual(1.5);
  });

  it('Adam and Momentum: a layer trained, frozen and unfrozen again forgets its old moments', () => {
    for (const name of ['adam', 'momentum'] as const) {
      const lr = 0.01;
      const a = new Network(arch, 3);
      const opt = new Optimizer(a, name, lr);
      train(a, opt, 50, [false, false]); // block 0 builds up moments
      train(a, opt, 200, [true, false], 400); // frozen while the head trains on
      opt.reset(0);
      // Same weights, same next batch, a brand-new optimiser for block 0: identical update.
      const b = new Network(arch, 3);
      b.setWeights(a.getWeights());
      const fresh = new Optimizer(b, name, lr);
      const wa = a.blocks[0].W.slice();
      train(a, opt, 1, [false, true], 7);
      train(b, fresh, 1, [false, true], 7);
      a.blocks[0].W.forEach((v, i) => expect(v - wa[i]).toBeCloseTo(b.blocks[0].W[i] - wa[i], 7));
    }
  });

  it('the trainer resets a block when the page unfreezes it', async () => {
    const { data, arch: pointsArch } = payload('blobs', 600);
    const lr = 0.003;
    const h = harness(pointsArch, data, { lr, batchSize: 10, optimizer: 'adam' }, { pointMs: 0, evalMs: 0 }, [true, true, false]);
    for (let i = 0; i < 300; i++) h.t.handle({ type: 'step' });
    h.t.handle({ type: 'frozen', frozen: [false, false, false] });
    const before = h.weights().at(-1)!.weights[0].slice();
    h.t.handle({ type: 'step' });
    const after = h.weights().at(-1)!.weights[0];
    let max = 0;
    after.forEach((v, i) => (max = Math.max(max, Math.abs(v - before[i]))));
    expect(max).toBeGreaterThan(0);
    expect(max / lr).toBeLessThanOrEqual(1.001); // was 1.61 with one global Adam step count
  });
});

/** The trainer's pending evaluation (private), to act while one is half done. */
const pendingEval = (t: Trainer) => (t as unknown as { evalJob: { i: number } | null }).evalJob;

describe('FLOW-6: Step while a test evaluation is half done', () => {
  it('every recorded evaluation is of the weights at its own step, and the last step gets one', async () => {
    const d = pointsData({ id: 'circle', count: 8000, noise: 0.1, trainRatio: 0.5, seed: 1 });
    const data: DataPayload = { inputSize: d.inputSize, scale: 1, classes: 2, trainX: d.trainX, trainY: d.trainY, testX: d.testX, testY: d.testY };
    expect(data.testY.length).toBe(4000);
    const arch: Arch = { input: d.input, layers: [{ kind: 'dense', units: 8, act: 'tanh' }], classes: 2 };
    const log: FromTrainer[] = [];
    // Very short slices: the 4,000-point evaluation takes many of them, as MNIST's does on the page.
    const t = new Trainer((m) => log.push(m), 0.01, { pointMs: 0, evalMs: 0 });
    const w0 = new Network(arch, 1).getWeights();
    t.handle({ type: 'model', version: 1, arch, weights: w0, hyper: { lr: 0.1, batchSize: 10, optimizer: 'sgd' }, frozen: [] });
    t.handle({ type: 'data', data });
    await until(() => (pendingEval(t)?.i ?? 0) >= 400);
    expect(pendingEval(t)!.i).toBeLessThan(4000); // still under way
    for (let i = 0; i < 10; i++) t.handle({ type: 'step' });
    await until(() => !pendingEval(t), 20_000);
    const evals = log.flatMap((m) => (m.type === 'metrics' ? m.evals : []));
    const at = new Map<number, Float32Array[]>([[0, w0]]);
    for (const m of log) if (m.type === 'weights') at.set(m.step, m.weights);
    expect(evals.length).toBeGreaterThan(0);
    for (const e of evals) expect(e.acc, `evaluation labelled step ${e.step}`).toBe(evaluate(arch, at.get(e.step)!, data).acc);
    expect(evals.at(-1)!.step).toBe(10);
  });
});

describe('FLOW-2: data taken away before a new network arrives', () => {
  it('the new network is measured only on the new data, once, however long that data takes', async () => {
    const a = payload('circle', 600, 1);
    const b = payload('blobs', 600, 2); // same input size and classes: nothing else tells them apart
    const hyper: Hyper = { lr: 0.03, batchSize: 10, optimizer: 'adam' };
    const h = harness(a.arch, a.data, hyper);
    await until(() => h.evals().length === 1);
    h.t.handle({ type: 'play' });
    await sleep(60);
    // What the page does on a dataset switch: data away, the new network, then (later) the new data.
    h.t.handle({ type: 'data', data: null, rev: 1 });
    const w = new Network(b.arch, 7).getWeights();
    h.t.handle({ type: 'model', version: 2, arch: b.arch, weights: w, hyper, frozen: [] });
    await sleep(120); // the new data is still loading: no training, no evaluation
    const v2 = () => h.log.flatMap((m) => (m.type === 'metrics' && m.version === 2 ? m.evals : []));
    expect(v2()).toEqual([]);
    expect(h.last().running).toBe(false);
    h.t.handle({ type: 'data', data: b.data, rev: 2 });
    await until(() => v2().length === 1);
    await sleep(60);
    expect(v2()).toHaveLength(1);
    expect(v2()[0]).toMatchObject({ step: 0, dataRev: 2, acc: evaluate(b.arch, w, b.data).acc });
  });
});

describe('FLOW-3: new points for the same network continue its run', () => {
  it('step, samples, epoch and curves carry on, and the next evaluation is on the new points', async () => {
    const a = payload('circle', 600, 1);
    const b = payload('circle', 600, 2); // Regenerate: a new draw of the same size
    const h = harness(a.arch, a.data, { lr: 0.03, batchSize: 10, optimizer: 'adam' }, { pointMs: 0, evalMs: 0 });
    for (let i = 0; i < 45; i++) h.t.handle({ type: 'step' }); // 1.5 epochs of 300 points
    await until(() => !pendingEval(h.t));
    const before = h.last();
    expect(before.epochFraction).toBeCloseTo(1.5, 9);
    const nPoints = h.points().length;
    h.t.handle({ type: 'data', data: b.data, rev: 1 });
    await until(() => !pendingEval(h.t));
    await sleep(5);
    const ev = h.evals().at(-1)!;
    expect(ev.step).toBe(45);
    expect(ev.epoch).toBeCloseTo(1.5, 9); // not back to 1.0
    expect(ev.dataRev).toBe(1);
    expect(ev.acc).toBe(evaluate(a.arch, h.weights().at(-1)!.weights, b.data).acc);
    expect(h.evals().map((e) => [e.step, e.dataRev])).toEqual([[45, 0], [45, 1]]); // the history stays
    expect(h.points().length).toBe(nPoints);
    // Training goes on from the same step and the same point in the epoch, on the new points.
    h.t.handle({ type: 'epoch' });
    await until(() => h.last().epoch === 2 && !h.last().running);
    expect(h.last().step).toBe(60);
    expect(h.last().seen).toBe(before.seen + 150);
    expect(h.last().epochFraction).toBe(2);
  });

  it('a curve point is never a mix of old and new points: the one in progress closes at the change', async () => {
    const a = payload('circle', 6000, 1); // 300 steps an epoch: a curve point every 5 steps
    const b = payload('circle', 6000, 2);
    const h = harness(a.arch, a.data, { lr: 0.03, batchSize: 10, optimizer: 'adam' }, { pointMs: 0, evalMs: 0 });
    // Two batches as Play trains them (Step and Pause close the curve point themselves).
    const train = () => (h.t as unknown as { trainBatch(): void }).trainBatch();
    train(); // a point at step 1
    train(); // step 2: the next point is due at step 6
    h.t.handle({ type: 'pause' });
    const k = h.points().length;
    train();
    h.t.handle({ type: 'data', data: b.data, rev: 1 });
    h.t.handle({ type: 'pause' });
    expect(h.points().length).toBe(k + 1);
    expect(h.points().at(-1)!.step).toBe(3);
  });
});
