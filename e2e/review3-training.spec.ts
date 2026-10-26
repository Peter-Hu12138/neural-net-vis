import { expect, test, type Page } from '@playwright/test';

/**
 * Third review, training findings: what 05 reports after training stops (DATA-1, ANA-1), where
 * +1 Epoch stops (DATA-5), "Train on" as a new experiment (DATA-2), your points across feature
 * and generator changes (DATA-4), the input while a dataset loads (DATA-6), transferring point
 * models (MOD-5) and the accuracy a saved model reports (MOD-8).
 */

type Probe = { key: string; caption: string; x: Float32Array; label: number | null; coords?: Float32Array };
type Raster = {
  store: {
    dataset: string;
    version: number;
    features: string[];
    trainLimit: number | null;
    status: { step: number; seen: number; epoch: number; epochFraction: number; running: boolean } | null;
    evals: { acc: number; loss: number; epoch: number; step: number; confusion: number[] }[];
    points: { step: number; epoch: number }[];
    custom: { id: number; x: Float32Array; y: number; coords?: Float32Array }[];
    probe: Probe | null;
    frozen: boolean[];
    weightsStep: number;
    data: { inputSize: number; scale: number; testX: ArrayLike<number>; testY: Uint8Array; trainY: Uint8Array } | null;
    net: { inputSize: number; classes: number; forward(x: Float32Array): Float32Array; getWeights(): Float32Array[]; setWeights(w: Float32Array[]): void };
  };
  actions: Record<string, (...a: unknown[]) => unknown>;
};

let errors: string[] = [];

test.beforeEach(async ({ page }) => {
  errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(m.text());
  });
});

test.afterEach(() => {
  expect(errors, 'no console errors or uncaught exceptions').toEqual([]);
});

const R = 'window.raster';
const store = <T>(page: Page, expr: string) => page.evaluate(`(() => { const s = ${R}.store; return ${expr}; })()`) as Promise<T>;

async function open(page: Page) {
  await page.goto('/');
  await expect(page.locator('#fact-data')).toContainText('train', { timeout: 30_000 });
  await expect(page.locator('#play')).toBeEnabled();
}

async function switchTo(page: Page, id: string) {
  await page.evaluate((d) => (window as unknown as { raster: Raster }).raster.actions.setDataset(d), id);
  await page.waitForFunction((d) => {
    const s = (window as unknown as { raster: Raster }).raster.store;
    return s.dataset === d && !!s.data && (s.evals.length > 0 || !!s.status);
  }, id, { timeout: 60_000 });
}

/** Waits until training has stopped and the page holds the trainer's last weights. */
async function stopped(page: Page) {
  await page.waitForFunction(() => {
    const s = (window as unknown as { raster: Raster }).raster.store;
    return !!s.status && !s.status.running && s.weightsStep === s.status.step;
  });
}

/** After stopping: the page's last evaluation must be of the weights training stopped at. */
async function stoppedAndEvaluated(page: Page) {
  await stopped(page);
  await expect
    .poll(() => store<number[]>(page, '[s.evals.at(-1)?.step ?? -1, s.status.step]'), { timeout: 10_000, message: 'step of the last evaluation, and of the weights' })
    .toEqual(await store<number[]>(page, '[s.status.step, s.status.step]'));
}

/** Test accuracy and confusion matrix of the page's current weights, computed from scratch. */
const truth = (page: Page) =>
  page.evaluate(() => {
    const s = (window as unknown as { raster: Raster }).raster.store;
    const d = s.data!;
    const K = s.net.classes;
    const confusion = new Array(K * K).fill(0);
    let correct = 0;
    const x = new Float32Array(d.inputSize);
    for (let i = 0; i < d.testY.length; i++) {
      for (let j = 0; j < d.inputSize; j++) x[j] = d.testX[i * d.inputSize + j] * d.scale;
      const p = s.net.forward(x);
      let best = 0;
      for (let k = 1; k < K; k++) if (p[k] > p[best]) best = k;
      if (best === d.testY[i]) correct++;
      confusion[d.testY[i] * K + best]++;
    }
    return { acc: correct / d.testY.length, confusion };
  });

async function tab(page: Page, name: 'Layers' | 'Pretrained' | 'Save and open') {
  await page.getByRole('tab', { name: new RegExp(`^${name}`) }).click();
  await expect(page.getByRole('tab', { name: new RegExp(`^${name}`) })).toHaveAttribute('aria-selected', 'true');
}

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
const testAccKpi = (page: Page) => page.locator('#training .kpi').filter({ hasText: 'Test acc.' }).locator('b');

test('DATA-1 / ANA-1: after +1 Epoch on Four blobs, 05 describes the weights training stopped at', async ({ page }) => {
  await open(page);
  await switchTo(page, 'blobs3');
  for (let e = 1; e <= 2; e++) {
    await page.click('#epoch');
    await page.waitForFunction((n) => {
      const s = (window as unknown as { raster: Raster }).raster.store.status;
      return !!s && s.epoch === n && !s.running;
    }, e);
    await stoppedAndEvaluated(page);
    const last = await store<{ step: number; epoch: number; evalEpoch: number; pointStep: number; pointEpoch: number; confusion: number[]; acc: number }>(
      page,
      '({ step: s.status.step, epoch: s.status.epochFraction, evalEpoch: s.evals.at(-1).epoch, pointStep: s.points.at(-1).step, pointEpoch: s.points.at(-1).epoch, confusion: s.evals.at(-1).confusion, acc: s.evals.at(-1).acc })',
    );
    expect(last.epoch).toBe(e);
    expect(last.evalEpoch).toBe(e);
    expect(last.pointStep).toBe(last.step); // the train curve reaches the end of the epoch too
    expect(last.pointEpoch).toBe(e);
    const t = await truth(page);
    expect(last.acc).toBe(t.acc);
    expect(last.confusion).toEqual(t.confusion);
    await expect(testAccKpi(page)).toHaveText(pct(t.acc));
  }
});

test('DATA-1 / ANA-1: Pause during Play on Circle: the test accuracy and confusion matrix catch up with the paused weights', async ({ page }) => {
  await open(page);
  await switchTo(page, 'circle');
  for (const ms of [130, 400]) {
    await page.click('#play');
    await page.waitForTimeout(ms);
    await page.click('#play');
    // Pausing is immediate; the evaluation follows.
    await page.waitForFunction(() => !(window as unknown as { raster: Raster }).raster.store.status?.running, null, { timeout: 2000 });
    await stoppedAndEvaluated(page);
    const t = await truth(page);
    expect(await store<number>(page, 's.evals.at(-1).acc')).toBe(t.acc);
    expect(await store<number[]>(page, 's.evals.at(-1).confusion')).toEqual(t.confusion);
    expect(await store<boolean>(page, 's.points.at(-1).step === s.status.step')).toBe(true);
    await expect(testAccKpi(page)).toHaveText(pct(t.acc));
  }
});

test('DATA-5: +1 Epoch on MNIST, Train on 200, batch 32 stops at exactly 1.00, 2.00 and 3.00 epochs', async ({ page }) => {
  await open(page);
  await page.getByRole('group', { name: 'Train on' }).getByRole('button', { name: '200', exact: true }).click();
  await page.waitForFunction(() => (window as unknown as { raster: Raster }).raster.store.data?.trainY.length === 200);
  expect(await store<number>(page, 's.hyper.batchSize')).toBe(32);
  for (let e = 1; e <= 3; e++) {
    await page.click('#epoch');
    await page.waitForFunction((n) => {
      const s = (window as unknown as { raster: Raster }).raster.store.status;
      return !!s && s.epoch === n && !s.running;
    }, e, { timeout: 60_000 });
    const s = (await store<{ step: number; seen: number; epochFraction: number }>(page, 's.status'))!;
    expect(s.epochFraction).toBe(e);
    expect(s.seen).toBe(200 * e);
    expect(s.step).toBe(7 * e); // six batches of 32 and one of 8
    await expect(page.locator('.bar .stat').first().locator('b')).toHaveText(`${e}.00`);
  }
});

test('DATA-2: changing "Train on" starts a new experiment and keeps the current input', async ({ page }) => {
  await open(page);
  await page.evaluate(() => {
    const r = (window as unknown as { raster: Raster; w0: Float32Array[] }).raster;
    (window as unknown as { w0: Float32Array[] }).w0 = r.store.net.getWeights();
    r.actions.setProbe(r.actions.testProbe(r.store.data, 5));
  });
  await page.click('#play');
  await page.waitForFunction(() => ((window as unknown as { raster: Raster }).raster.store.status?.step ?? 0) > 30);
  await page.click('#play');
  await stopped(page);
  const v0 = await store<number>(page, 's.version');

  const sameAsStart = () =>
    page.evaluate(() => {
      const w = (window as unknown as { raster: Raster }).raster.store.net.getWeights();
      const w0 = (window as unknown as { w0: Float32Array[] }).w0;
      return w.every((a, i) => a.length === w0[i].length && a.every((v, j) => v === w0[i][j]));
    });
  for (const limit of ['200', 'All']) {
    await page.getByRole('group', { name: 'Train on' }).getByRole('button', { name: limit, exact: true }).click();
    const s = await store<{ version: number; status: unknown; points: number; key: string; step: number }>(
      page,
      '({ version: s.version, status: s.status, points: s.points.length, key: s.probe.key, step: s.weightsStep })',
    );
    expect(s.version).toBeGreaterThan(v0);
    expect(s.status === null || (s.status as { seen: number }).seen === 0).toBe(true);
    expect(s.points).toBe(0);
    expect(s.step).toBe(0);
    expect(s.key).toBe('test:5'); // still the reader's input
    expect(await sameAsStart()).toBe(true); // the starting weights again, not the trained ones
    await page.waitForFunction(() => {
      const ev = (window as unknown as { raster: Raster }).raster.store.evals;
      return ev.length === 1 && ev[0].step === 0;
    });
    await expect(page.locator('#netview .probe-caption')).toHaveText('Test digit #5 · label 1');
    // Train a little on this set before switching back.
    await page.click('#step');
    await stopped(page);
  }
});

test('DATA-2: after a transfer, "Train on" restarts from the transferred weights, frozen layers and all', async ({ page }) => {
  test.setTimeout(180_000);
  await open(page);
  await switchTo(page, 'fashion');
  await tab(page, 'Pretrained');
  await page.getByRole('button', { name: 'Transfer MNIST small CNN to Fashion-MNIST' }).click();
  await expect(page.locator('#model-provenance')).toContainText('transferred from MNIST small CNN, frozen', { timeout: 60_000 });
  await page.evaluate(() => {
    (window as unknown as { w0: Float32Array[] }).w0 = (window as unknown as { raster: Raster }).raster.store.net.getWeights();
  });
  const frozen = await store<boolean[]>(page, 's.frozen');
  expect(frozen).toEqual([true, true, true, false]);
  for (let i = 0; i < 5; i++) await page.click('#step');
  await page.waitForFunction(() => (window as unknown as { raster: Raster }).raster.store.weightsStep === 5);
  await page.getByRole('group', { name: 'Train on' }).getByRole('button', { name: '1,000', exact: true }).click();
  expect(await store<boolean[]>(page, 's.frozen')).toEqual(frozen);
  expect(
    await page.evaluate(() => {
      const w = (window as unknown as { raster: Raster }).raster.store.net.getWeights();
      const w0 = (window as unknown as { w0: Float32Array[] }).w0;
      return w.every((a, i) => a.every((v, j) => v === w0[i][j]));
    }),
  ).toBe(true);
  await expect(page.locator('#model-provenance')).toHaveText('Conv and dense layers transferred from MNIST small CNN, frozen; new output layer for Fashion-MNIST');
});

/** Pixel position of data coordinates (u, v) on the 03 plane. */
async function at(page: Page, u: number, v: number) {
  const sel = '#boundary .bd-2d canvas.bd-plane';
  await page.locator(sel).scrollIntoViewIfNeeded();
  const box = (await page.locator(sel).boundingBox())!;
  const [x, y, w, h] = (await page.locator(sel).getAttribute('data-plot'))!.split(' ').map(Number);
  const r = Number(await page.locator(sel).getAttribute('data-r'));
  return { x: box.x + x + ((u + r) / (2 * r)) * w, y: box.y + y + ((r - v) / (2 * r)) * h };
}

test('DATA-4: your points survive feature toggles, noise, count, share and regenerate', async ({ page }) => {
  await open(page);
  await switchTo(page, 'circle');
  for (const [u, v] of [[-0.62, 0.52], [0.48, 0.24], [-0.07, -0.88]]) {
    const p = await at(page, u, v);
    await page.keyboard.down('Shift');
    await page.mouse.click(p.x, p.y);
    await page.keyboard.up('Shift');
  }
  await expect.poll(() => store<number>(page, 's.custom.length')).toBe(3);
  // Use one of them as the current input.
  await page.evaluate(() => {
    const r = (window as unknown as { raster: Raster }).raster;
    const c = r.store.custom[1];
    r.actions.setProbe({ x: c.x, label: c.y, caption: 'Your point', key: `custom:${c.id}`, coords: c.coords });
  });
  const train = await store<number>(page, 's.data.trainY.length');

  /** Your points, their inputs, the current input, and how many samples one epoch trains on. */
  const check = async (features: string[]) => {
    const got = await store<{ coords: number[]; x: number[] }[]>(page, 's.custom.map((c) => ({ coords: Array.from(c.coords), x: Array.from(c.x) }))');
    expect(got).toHaveLength(3);
    for (const c of got) {
      const want = features.map((f) => (f === 'x1' ? c.coords[0] : f === 'x2' ? c.coords[1] : f === 'x1^2' ? c.coords[0] ** 2 : NaN));
      c.x.forEach((v, i) => expect(v).toBeCloseTo(want[i], 5));
    }
    const probe = await store<{ key: string; x: number[] }>(page, '({ key: s.probe.key, x: Array.from(s.probe.x) })');
    expect(probe.key).toMatch(/^custom:/);
    expect(probe.x).toHaveLength(features.length);
    await expect(page.getByRole('button', { name: 'Remove my 3 points' })).toBeVisible();
    // The trainer has them: one epoch is the generated points plus 10 repeats of each of yours.
    // (Noise and Regenerate continue the run, so count from where it stands.)
    await page.waitForFunction(() => !(window as unknown as { raster: Raster }).raster.store.status?.running);
    const from = await store<{ epoch: number; seen: number }>(page, '({ epoch: s.status?.epoch ?? 0, seen: s.status?.seen ?? 0 })');
    await page.click('#epoch');
    await page.waitForFunction((e) => {
      const s = (window as unknown as { raster: Raster }).raster.store.status;
      return !!s && s.epoch === e + 1 && !s.running;
    }, from.epoch);
    const n = await store<number>(page, 's.data.trainY.length');
    expect((await store<number>(page, 's.status.seen')) - from.seen).toBe(n + 30);
  };

  await page.locator('button.ds-feat[data-feature="x1^2"]').click();
  await expect.poll(() => store<string[]>(page, 's.features')).toEqual(['x1', 'x2', 'x1^2']);
  await check(['x1', 'x2', 'x1^2']);

  await page.locator('#dsp-noise').evaluate((el: HTMLInputElement) => {
    el.value = '0.2';
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  });
  await expect.poll(() => store<number>(page, 's.pointsConfig.noise')).toBe(0.2);
  await check(['x1', 'x2', 'x1^2']);

  await page.locator('#dsp-regenerate').click();
  await expect.poll(() => store<number>(page, 's.pointsConfig.seed')).toBe(2);
  await check(['x1', 'x2', 'x1^2']);
  expect(await store<number>(page, 's.data.trainY.length')).toBe(train);

  await page.locator('.dsp-seg[aria-labelledby="dsp-count-label"] button', { hasText: /^400$/ }).click();
  await expect.poll(() => store<number>(page, 's.pointsConfig.count')).toBe(400);
  await check(['x1', 'x2', 'x1^2']);

  await page.locator('button.ds-feat[data-feature="x1^2"]').click();
  await check(['x1', 'x2']);
});

test('DATA-6: while CIFAR-10 loads, the MNIST digit is no longer the current input', async ({ page }) => {
  let release: () => void = () => {};
  const gate = new Promise<void>((r) => (release = r));
  await page.route('**/data/cifar10-*', async (route) => {
    await gate;
    await route.continue();
  });
  await open(page);
  await expect(page.locator('#netview .probe-caption')).toHaveText('Test digit #0 · label 7');
  await page.evaluate(() => void (window as unknown as { raster: Raster }).raster.actions.setDataset('cifar10'));
  await page.waitForFunction(() => {
    const s = (window as unknown as { raster: Raster }).raster.store;
    return s.dataset === 'cifar10' && !s.data && s.net.inputSize === 3072;
  });
  await page.waitForTimeout(300);
  expect(await store<unknown>(page, 's.probe')).toBeNull();
  await expect(page.locator('#netview .probe-caption')).toHaveText('');
  await expect(page.locator('#backprop .bp-sample')).not.toContainText('digit');
  release();
  await page.waitForFunction(() => !!(window as unknown as { raster: Raster }).raster.store.data, null, { timeout: 60_000 });
  await expect(page.locator('#netview .probe-caption')).toHaveText(/^Test image #0 · /);
  expect(await store<number>(page, 's.probe.x.length')).toBe(3072);
});

/** Opens a model file through 01's file picker. */
async function openFile(page: Page, json: unknown, name: string) {
  await tab(page, 'Save and open');
  await page.locator('#model-file').setInputFiles({ name, mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(json)) });
  await expect(page.locator('.model-choice')).toBeVisible();
}

const toggle = (page: Page, f: string) => page.locator(`button.ds-feat[data-feature="${f}"]`).click();


test('MOD-5: transferring a point model feeds it the features it was trained on', async ({ page }) => {
  await open(page);
  await switchTo(page, 'circle');
  // Squares only: x₁², x₂².
  await toggle(page, 'x1^2');
  await toggle(page, 'x2^2');
  await toggle(page, 'x1');
  await toggle(page, 'x2');
  await expect.poll(() => store<string[]>(page, 's.features')).toEqual(['x1^2', 'x2^2']);
  for (let e = 1; e <= 3; e++) {
    await page.click('#epoch');
    await page.waitForFunction((n) => (window as unknown as { raster: Raster }).raster.store.status?.epoch === n && !(window as unknown as { raster: Raster }).raster.store.status?.running, e);
  }
  const file = await page.evaluate(() => (window as unknown as { raster: Raster }).raster.actions.exportModel('Circle on squares'));
  // Back to the raw coordinates, then open the file and transfer it.
  await toggle(page, 'x1');
  await toggle(page, 'x2');
  await toggle(page, 'x1^2');
  await toggle(page, 'x2^2');
  await expect.poll(() => store<string[]>(page, 's.features')).toEqual(['x1', 'x2']);
  await page.locator('#models').scrollIntoViewIfNeeded();
  await openFile(page, file, 'circle-on-squares.json');
  await page.getByRole('button', { name: 'Transfer to Circle' }).click();
  await expect(page.locator('#model-provenance')).toContainText('transferred from Circle on squares');
  expect(await store<string[]>(page, 's.features')).toEqual(['x1^2', 'x2^2']);
  await expect(page.locator('button.ds-feat[data-feature="x1^2"]')).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator('button.ds-feat[data-feature="x1"]')).toHaveAttribute('aria-pressed', 'false');
  expect(await store<boolean[]>(page, 's.frozen')).toEqual([true, true, false]);
  // The frozen first layer holds the saved weights and is fed squares again.
  const first = await page.evaluate((f) => {
    const w = (window as unknown as { raster: Raster }).raster.store.net.getWeights()[0];
    const bin = atob((f as { weights: string[] }).weights[0]);
    const saved = new Float32Array(Uint8Array.from(bin, (c) => c.charCodeAt(0)).buffer);
    return w.every((v, i) => v === saved[i]);
  }, file);
  expect(first).toBe(true);
  await expect(page.locator('.model-status')).not.toHaveClass(/is-error/);
});

test('MOD-5: a model trained on a third coordinate cannot be transferred to 2-D points', async ({ page }) => {
  await open(page);
  await switchTo(page, 'blobs3');
  await page.click('#step');
  await stopped(page);
  const file = await page.evaluate(() => (window as unknown as { raster: Raster }).raster.actions.exportModel('Four blobs'));
  await switchTo(page, 'circle');
  await toggle(page, 'x1^2'); // three features, as many inputs as the file's x₁, x₂, x₃
  await expect.poll(() => store<string[]>(page, 's.features')).toEqual(['x1', 'x2', 'x1^2']);
  const msg = await page.evaluate((f) => {
    try {
      (window as unknown as { raster: Raster }).raster.actions.transferModel(f);
      return 'transferred';
    } catch (e) {
      return (e as Error).message;
    }
  }, file);
  expect(msg).toBe('Four blobs was trained on x₃, which Circle does not have: its points have 2 coordinates.');
  expect(await store<string[]>(page, 's.features')).toEqual(['x1', 'x2', 'x1^2']);
});

test('MOD-8: a saved model reports a test accuracy only for the weights it saves', async ({ page }) => {
  await open(page);
  await switchTo(page, 'circle');
  /** Saves through the app, and evaluates the saved weights independently. */
  const save = () =>
    page.evaluate(() => {
      const r = (window as unknown as { raster: Raster }).raster;
      const file = r.actions.exportModel('m') as { weights: string[]; meta: { testAccuracy?: number } };
      const ws = file.weights.map((b) => new Float32Array(Uint8Array.from(atob(b), (c) => c.charCodeAt(0)).buffer));
      const net = r.store.net;
      const keep = net.getWeights();
      net.setWeights(ws);
      const d = r.store.data!;
      let correct = 0;
      const x = new Float32Array(d.inputSize);
      for (let i = 0; i < d.testY.length; i++) {
        for (let j = 0; j < d.inputSize; j++) x[j] = d.testX[i * d.inputSize + j] * d.scale;
        const p = net.forward(x);
        let best = 0;
        for (let k = 1; k < p.length; k++) if (p[k] > p[best]) best = k;
        if (best === d.testY[i]) correct++;
      }
      net.setWeights(keep);
      return { saved: file.meta.testAccuracy ?? null, actual: correct / d.testY.length };
    });

  // While training runs, any figure that is saved must be the saved weights' own.
  await page.click('#play');
  let stale = 0;
  for (let i = 0; i < 25; i++) {
    const s = await save();
    if (s.saved === null) stale++;
    else expect(s.saved).toBe(s.actual);
    await page.waitForTimeout(40);
  }
  expect(stale).toBeGreaterThan(0); // most of the time the last evaluation is behind
  await page.click('#play');
  await stoppedAndEvaluated(page);
  let s = await save();
  expect(s.saved).toBe(s.actual);

  // A manual update (06 Backpropagation's Apply) changes the weights without training: saved
  // straight away, before the trainer has evaluated them, the file has no figure.
  const immediately = await page.evaluate(() => {
    const r = (window as unknown as { raster: Raster }).raster;
    r.actions.applyWeights(r.store.net.getWeights().map((w) => w.map((v) => v * 0.5)));
    return (r.actions.exportModel('m') as { meta: { testAccuracy?: number } }).meta.testAccuracy ?? null;
  });
  expect(immediately).toBeNull();
  await page.waitForFunction(() => {
    const r = (window as unknown as { raster: Raster }).raster;
    const f = r.actions.exportModel('m') as { meta: { testAccuracy?: number } };
    return f.meta.testAccuracy !== undefined;
  });
  s = await save();
  expect(s.saved).toBe(s.actual);

  // Saved in this browser: the list shows that figure.
  await tab(page, 'Save and open');
  await page.locator('#model-save-browser').click();
  await expect(page.locator('.saved-row').last().locator('.zoo-meta')).toContainText(`${pct(s.actual)} test`);
});

// ── Round B: FLOW-2, FLOW-3, FLOW-7 ──────────────────────────────────────

const evalList = (page: Page) => store<string[]>(page, 's.evals.map((e) => `${e.step}:${e.acc}`)');

test('FLOW-2: a switch between point datasets measures the new network on the new points only', async ({ page }) => {
  await open(page);
  await switchTo(page, 'circle');
  for (const id of ['spiral', 'moons', 'xor', 'circle', 'checker']) {
    await page.click('#epoch'); // a trained network, so the old points' figure stands out
    await page.waitForFunction(() => {
      const s = (window as unknown as { raster: Raster }).raster.store.status;
      return !!s && s.epoch === 1 && !s.running;
    });
    await stoppedAndEvaluated(page);
    await switchTo(page, id);
    await page.waitForFunction(() => (window as unknown as { raster: Raster }).raster.store.evals.length > 0);
    await page.waitForTimeout(300);
    const t = await truth(page);
    expect(await evalList(page), `evaluations after the switch to ${id}`).toEqual([`0:${t.acc}`]);
  }
});

test('FLOW-2: while Fashion-MNIST loads, nothing is measured; then one evaluation, on Fashion-MNIST', async ({ page }) => {
  test.setTimeout(180_000);
  let release: () => void = () => {};
  const gate = new Promise<void>((r) => (release = r));
  await page.route('**/data/fashion-*', async (route) => {
    await gate;
    await route.continue();
  });
  await open(page);
  await tab(page, 'Pretrained');
  await page.getByRole('button', { name: 'Load MNIST LeNet' }).click();
  await expect(page.locator('#model-provenance')).toContainText('MNIST LeNet', { timeout: 60_000 });
  await page.waitForFunction(() => (window as unknown as { raster: Raster }).raster.store.evals.length > 0, null, { timeout: 30_000 });
  expect(await store<number>(page, 's.evals.at(-1).acc')).toBeGreaterThan(0.9);
  await page.evaluate(() => void (window as unknown as { raster: Raster }).raster.actions.setDataset('fashion'));
  // The weights are kept (a loaded model) and fit Fashion-MNIST's input: before, the trainer
  // measured them on MNIST while Fashion-MNIST loaded, and 05 showed 98% for Fashion-MNIST.
  await page.waitForTimeout(4000);
  expect(await store<boolean>(page, 's.data === null')).toBe(true);
  expect(await evalList(page)).toEqual([]);
  await expect(page.locator('.bar .stat').filter({ hasText: 'Test acc.' }).locator('b')).toHaveText('—');
  await expect(testAccKpi(page)).toHaveText('—');
  release();
  await page.waitForFunction(() => {
    const s = (window as unknown as { raster: Raster }).raster.store;
    return !!s.data && s.evals.length > 0;
  }, null, { timeout: 60_000 });
  await page.waitForTimeout(1000);
  const t = await truth(page);
  expect(await evalList(page)).toEqual([`0:${t.acc}`]);
  expect(t.acc).toBeLessThan(0.5); // digits' filters on clothes, before any training
  await expect(testAccKpi(page)).toHaveText(pct(t.acc));
});

test('FLOW-3: noise and Regenerate continue the run; point count and share restart it', async ({ page }) => {
  await open(page);
  await switchTo(page, 'circle');
  await page.evaluate(() => {
    (window as unknown as { w0: Float32Array[] }).w0 = (window as unknown as { raster: Raster }).raster.store.net.getWeights();
  });
  const sameWeights = (name: string) =>
    page.evaluate((n) => {
      const w = (window as unknown as { raster: Raster }).raster.store.net.getWeights();
      const ref = (window as unknown as Record<string, Float32Array[]>)[n];
      return w.every((a, i) => a.length === ref[i].length && a.every((v, j) => v === ref[i][j]));
    }, name);
  const keep = () =>
    page.evaluate(() => {
      (window as unknown as { w1: Float32Array[] }).w1 = (window as unknown as { raster: Raster }).raster.store.net.getWeights();
    });
  for (let e = 1; e <= 2; e++) {
    await page.click('#epoch');
    await page.waitForFunction((n) => (window as unknown as { raster: Raster }).raster.store.status?.epoch === n, e);
  }
  await stoppedAndEvaluated(page);
  const prov = page.locator('#model-provenance');
  const evalLine = page.locator('#model-eval');
  await expect(prov).toHaveText('Random start (seed 1), then trained here for 2.00 epochs');

  /** After noise or Regenerate: the same network and run, measured on the new points. */
  const continued = async (change: () => Promise<void>, settled: string) => {
    await keep();
    const before = await store<{ version: number; step: number; points: number; evals: number }>(
      page,
      '({ version: s.version, step: s.status.step, points: s.points.length, evals: s.evals.length })',
    );
    await change();
    await page.waitForFunction(settled);
    await page.waitForFunction(() => {
      const s = (window as unknown as { raster: { store: { dataRev: number; evals: { dataRev: number }[] } } }).raster.store;
      return s.evals.at(-1)?.dataRev === s.dataRev;
    });
    const after = await store<{ version: number; step: number; points: number; evals: number; last: number }>(
      page,
      '({ version: s.version, step: s.status.step, points: s.points.length, evals: s.evals.length, last: s.evals.at(-1).step })',
    );
    expect(after.version).toBe(before.version);
    expect(after.step).toBe(before.step);
    expect(after.last).toBe(before.step);
    expect(after.points).toBeGreaterThanOrEqual(before.points);
    expect(after.evals).toBe(before.evals + 1);
    expect(await sameWeights('w1')).toBe(true);
    const t = await truth(page);
    expect(await store<number>(page, 's.evals.at(-1).acc')).toBe(t.acc);
    await expect(testAccKpi(page)).toHaveText(pct(t.acc));
    await expect(prov).toHaveText('Random start (seed 1), then trained here for 2.00 epochs');
    await expect(evalLine).toHaveText(`Test accuracy here: ${pct(t.acc)} on 300 test points`);
  };
  await continued(
    () =>
      page.locator('#dsp-noise').evaluate((el: HTMLInputElement) => {
        el.value = '0.3';
        el.dispatchEvent(new Event('input', { bubbles: true }));
      }),
    '(window.raster.store.pointsConfig.noise === 0.3)',
  );
  await continued(() => page.locator('#dsp-regenerate').click(), '(window.raster.store.pointsConfig.seed === 2)');

  // While training runs, new points do not stop it.
  await page.click('#play');
  await page.waitForFunction(() => (window as unknown as { raster: Raster }).raster.store.status?.running === true);
  await page.locator('#dsp-regenerate').click();
  const s0 = await store<number>(page, 's.status.step');
  await page.waitForFunction((n) => ((window as unknown as { raster: Raster }).raster.store.status?.step ?? 0) > n + 20, s0);
  expect(await store<boolean>(page, 's.status.running')).toBe(true);
  await page.click('#play');
  await stoppedAndEvaluated(page);

  // Another amount of training data is another experiment: back to the starting weights.
  for (const change of [
    () => page.locator('.dsp-seg[aria-labelledby="dsp-count-label"] button', { hasText: /^400$/ }).click(),
    () =>
      page.locator('#dsp-share').evaluate((el: HTMLInputElement) => {
        el.value = '0.8';
        el.dispatchEvent(new Event('input', { bubbles: true }));
      }),
  ]) {
    const v = await store<number>(page, 's.version');
    await change();
    await page.waitForFunction((n) => (window as unknown as { raster: Raster }).raster.store.version > n, v);
    await page.waitForFunction(() => (window as unknown as { raster: Raster }).raster.store.evals.length > 0);
    await page.waitForTimeout(200);
    expect(await sameWeights('w0')).toBe(true);
    expect(await store<number>(page, 's.status?.seen ?? 0')).toBe(0);
    const t = await truth(page);
    expect(await evalList(page)).toEqual([`0:${t.acc}`]);
    await expect(prov).toHaveText('Random start (seed 1)');
    await expect(evalLine).toContainText(', before any training');
    await page.click('#epoch'); // train a little before the next change
    await page.waitForFunction(() => {
      const s = (window as unknown as { raster: Raster }).raster.store.status;
      return !!s && s.epoch === 1 && !s.running;
    });
    await stoppedAndEvaluated(page);
  }
});

test('FLOW-7: "Train on" keeps a drawing as the input and measures the restarted network once', async ({ page }) => {
  await open(page);
  await page.waitForFunction(() => (window as unknown as { raster: Raster }).raster.store.evals.length === 1, null, { timeout: 30_000 });
  await page.evaluate(() => {
    const r = (window as unknown as { raster: Raster }).raster;
    const x = new Float32Array(784);
    for (let i = 0; i < 28; i++) x[i * 28 + 14] = 1; // a vertical stroke
    r.actions.setProbe({ x, label: null, caption: 'Your drawing', key: 'draw' });
  });
  for (const limit of ['1,000', 'All']) {
    await page.getByRole('group', { name: 'Train on' }).getByRole('button', { name: limit, exact: true }).click();
    expect(await store<string>(page, 's.probe.key')).toBe('draw');
    await page.waitForFunction(() => (window as unknown as { raster: Raster }).raster.store.evals.length > 0, null, { timeout: 30_000 });
    await page.waitForTimeout(1500); // a second evaluation of the same weights would have landed by now
    expect(await evalList(page)).toHaveLength(1);
    expect(await store<string>(page, 's.probe.key')).toBe('draw');
    await expect(page.locator('#netview .probe-caption')).toHaveText('Your drawing');
  }
});
