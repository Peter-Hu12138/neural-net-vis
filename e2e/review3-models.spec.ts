import { readFileSync } from 'node:fs';
import { expect, test, type Page } from '@playwright/test';

/**
 * Third review, section 01 and the data it reads: damaged model files are refused before they
 * touch the page (MOD-2, MOD-4), keep-weights edits follow the layers (MOD-3), the Weights line
 * after such an edit (MOD-6), the early transfer table (MOD-7), the sine features' label (DATA-3),
 * uploads on a transparent background (DATA-7) and the accuracies docs/models.md quotes (DATA-8,
 * ANA-7, MOD-9).
 */

type Raster = {
  store: {
    status: { step: number; running: boolean } | null;
    weightsStep: number;
    evals: { acc: number; step: number }[];
    frozen: boolean[];
    dataset: string;
    data: unknown;
    modelName: string | null;
    info: { id: string; name: string };
    net: { getWeights(): Float32Array[]; blocks: unknown[] };
  };
  actions: { setDataset(id: string): Promise<void>; exportModel(name: string): Record<string, unknown> };
};

const raster = <T>(page: Page, fn: (r: Raster) => T) => page.evaluate(`(${fn.toString()})(window.raster)`) as Promise<T>;

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

async function open(page: Page) {
  await page.goto('/');
  await expect(page.locator('#fact-data')).toContainText('20,000 train', { timeout: 30_000 });
  await expect(page.locator('#play')).toBeEnabled();
  await expect(page.locator('.zoo-row')).toHaveCount(4);
}

async function switchTo(page: Page, id: string) {
  await page.evaluate(`window.raster.actions.setDataset(${JSON.stringify(id)})`);
  await page.waitForFunction((want) => {
    const s = (window as unknown as { raster: Raster }).raster.store;
    return s.dataset === want && !!s.data;
  }, id, { timeout: 60_000 });
}

/** Trains until `pred` holds, pauses, and waits until the page holds the trainer's last weights. */
async function trainUntil(page: Page, pred: string, timeout = 90_000) {
  await page.click('#play');
  await page.waitForFunction(pred, null, { timeout });
  await page.click('#play');
  await page.waitForFunction(() => {
    const s = (window as unknown as { raster: Raster }).raster.store;
    return !s.status?.running && s.weightsStep === s.status?.step;
  });
}

async function tab(page: Page, name: 'Layers' | 'Pretrained' | 'Save and open') {
  await page.getByRole('tab', { name: new RegExp(`^${name}`) }).click();
  await expect(page.getByRole('tab', { name: new RegExp(`^${name}`) })).toHaveAttribute('aria-selected', 'true');
}

/** Opens a model file built from the current network with `damage` applied to its JSON. */
async function openDamaged(page: Page, damage: (f: Record<string, unknown>) => void, name = 'damaged.json') {
  const file = await raster(page, (r) => r.actions.exportModel('Damaged model'));
  damage(file);
  await page.setInputFiles('#model-file', { name, mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(file)) });
}

test('damaged model files are refused at Open, in words, and the page keeps working', async ({ page }) => {
  await open(page);
  await tab(page, 'Save and open');
  const status = page.locator('.model-status');
  const cases: [string, (f: Record<string, unknown>) => void, string | RegExp][] = [
    ['an unknown dataset', (f) => (f.dataset = 'imagenet'), /^The model file names a dataset "imagenet" that this page does not have\. This page has MNIST digits, Fashion-MNIST, CIFAR-10 and \d+ point datasets\.$/],
    ['a frozen list that is not a list', (f) => (f.frozen = 'yes'), 'The model file’s frozen list should hold one true or false per layer (4 layers).'],
    ['weights that are not base64', (f) => ((f.weights as unknown[])[1] = 'not base64!'), 'Weight array 2 of the model file is damaged: it is not base64 text.'],
    ['a number for a weight array', (f) => ((f.weights as unknown[])[3] = 7), 'Weight array 4 of the model file is damaged: it is not base64 text.'],
    ['a number for a name', (f) => (f.name = 42), 'The model file has no name.'],
  ];
  for (const [what, damage, message] of cases) {
    await openDamaged(page, damage);
    await expect(status, what).toHaveText(message);
    await expect(status).toHaveClass(/is-error/);
    await expect(page.locator('.model-choice'), what).toBeHidden();
  }
  expect(await raster(page, (r) => [r.store.dataset, r.store.info.name, r.store.frozen.every((f) => f === false)])).toEqual(['mnist', 'MNIST digits', true]);

  // A point model whose features this page does not have.
  await switchTo(page, 'circle');
  await openDamaged(page, (f) => (f.features = ['x9', 'x10']));
  await expect(status).toHaveText('The model file asks for input features this page does not have for 2-D points: x9, x10.');
  expect(await raster(page, (r) => r.store.dataset)).toBe('circle');

  // Still in working order: a good file opens and loads, and training runs.
  await openDamaged(page, () => {});
  await page.getByRole('button', { name: 'Load as is' }).click();
  await expect(page.locator('#model-provenance')).toHaveText(/^Damaged model, from the file damaged\.json/);
  await trainUntil(page, '(window.raster.store.status?.step ?? 0) >= 3');
});

test('keep trained weights: removing a hidden layer drops its weights, and the output layer keeps its own and its lock', async ({ page }) => {
  await open(page);
  await switchTo(page, 'circle');
  await page.locator('#keep-weights').check();
  await trainUntil(page, '(window.raster.store.status?.step ?? 0) >= 30');
  // Lock the first hidden layer and the output layer; Dense 2 keeps learning.
  await page.click('#freeze-0');
  await page.click('#freeze-2');
  expect(await raster(page, (r) => r.store.frozen)).toEqual([true, false, true]);
  await page.evaluate(() => {
    const w = window as unknown as { raster: Raster; before: Float32Array[] };
    w.before = w.raster.store.net.getWeights();
  });

  // Remove Dense 1. Dense 2 now reads the two coordinates instead of eight units, so it starts
  // fresh; the output layer still reads eight units and keeps its weights and its lock.
  await page.click('#remove-0');
  const after = await page.evaluate(() => {
    const w = window as unknown as { raster: Raster; before: Float32Array[] };
    const now = w.raster.store.net.getWeights();
    const same = (a: Float32Array, b: Float32Array) => a.length === b.length && a.every((v, i) => v === b[i]);
    return {
      blocks: now.length / 2,
      firstIsRemovedLayer: same(now[0], w.before[0]),
      outputKept: same(now[2], w.before[4]) && same(now[3], w.before[5]),
      frozen: w.raster.store.frozen,
    };
  });
  expect(after).toEqual({ blocks: 2, firstIsRemovedLayer: false, outputKept: true, frozen: [false, true] });
  await expect(page.locator('#model-provenance')).toHaveText('Random start (seed 1); 1 of 2 layers kept their weights through an edit');
  await expect(page.locator('.freeze[aria-pressed="true"]')).toHaveCount(1);
  await expect(page.locator('#freeze-1')).toHaveAttribute('aria-pressed', 'true');

  // Adding a dense layer on top: the layer below and the output keep their weights and settings.
  await page.evaluate(() => {
    const w = window as unknown as { raster: Raster; before: Float32Array[] };
    w.before = w.raster.store.net.getWeights();
  });
  await page.getByRole('button', { name: '+ Dense layer' }).click();
  const added = await page.evaluate(() => {
    const w = window as unknown as { raster: Raster; before: Float32Array[] };
    const now = w.raster.store.net.getWeights();
    const same = (a: Float32Array, b: Float32Array) => a.length === b.length && a.every((v, i) => v === b[i]);
    return { first: same(now[0], w.before[0]), output: same(now[4], w.before[2]), frozen: w.raster.store.frozen };
  });
  expect(added).toEqual({ first: true, output: true, frozen: [false, false, true] });
  await expect(page.locator('.builder-keep .hint')).toContainText('as long as its size and the size of its input stay the same');
});

test('the Weights line follows a transferred network through a keep-weights edit', async ({ page }) => {
  await open(page);
  await switchTo(page, 'fashion');
  await tab(page, 'Pretrained');
  await page.click('#zoo-transfer-mnist-cnn');
  const prov = page.locator('#model-provenance');
  await expect(prov).toHaveText('Conv and dense layers transferred from MNIST small CNN, frozen; new output layer for Fashion-MNIST');

  // A wider dense layer starts fresh, and so does the output layer above it.
  await tab(page, 'Layers');
  await page.selectOption('#l2-units', '64');
  await expect(prov).toHaveText('Conv layers transferred from MNIST small CNN, frozen; new dense and output layers for Fashion-MNIST');
  expect(await raster(page, (r) => r.store.frozen)).toEqual([true, true, false, false]);
  await page.click('#freeze-1');
  await expect(prov).toHaveText('Conv layers transferred from MNIST small CNN, partly frozen; new dense and output layers for Fashion-MNIST');

  // Changing the first conv layer's filters leaves no transferred layer: a random start.
  await page.selectOption('#l0-filters', '16');
  await expect(prov).toHaveText(/^Random start \(seed \d+\)/);
});

test('the early transfer table leaves out the Fine-tuned column, which would repeat Frozen, and says why', async ({ page }) => {
  await open(page);
  await tab(page, 'Pretrained');
  await page.locator('.transfer-details summary').click();
  const tables = page.locator('.transfer-table');
  await expect(tables).toHaveCount(2);
  await expect(tables.nth(0).locator('thead th')).toHaveText(['Images', 'Scratch', 'Frozen', 'Conv frozen', 'Fine-tuned']);
  await expect(tables.nth(1).locator('thead th')).toHaveText(['Images', 'Scratch', 'Frozen', 'Conv frozen']);
  await expect(tables.nth(1).locator('tr.transfer-group th').first()).toHaveAttribute('colspan', '4');
  await expect(page.locator('#transfer-early-note')).toHaveText('No Fine-tuned column here: fine-tuning unlocks the copied layers only after 10,000 images, so until then it is the same run as Frozen.');
  // The control is described as it was run: nothing copied.
  await expect(page.locator('.model-measured')).toContainText('Control: with nothing copied (random, untrained conv filters, frozen, and a dense layer that learns from random weights)');
});

test('the sine feature is labelled sin πx₁ wherever it is shown', async ({ page }) => {
  await open(page);
  await switchTo(page, 'circle');
  const chip = page.locator('#datasets .ds-feat[data-feature="sin x1"]');
  await expect(chip).toHaveAttribute('aria-label', 'sin πx₁');
  await expect(chip).toHaveText('sin πx1'); // the index is a <sub>
  await chip.click();
  await page.waitForFunction(() => (window as unknown as { raster: Raster }).raster.store.net.getWeights()[0].length === 8 * 3);
  await expect(page.locator('#builder .builder-input')).toHaveText('3 features: x₁, x₂, sin πx₁');
  // 07's readout of the current input: the label next to the value sin(π·x₁).
  const row = page.locator('#datapanel .dp-vec tr', { hasText: 'sin πx1' });
  await expect(row).toHaveCount(1);
  const [x1, s] = await raster(page, (r) => {
    const p = (r.store as unknown as { probe: { x: Float32Array } }).probe;
    return [p.x[0], p.x[2]];
  });
  expect(s).toBeCloseTo(Math.sin(Math.PI * x1), 5);
  await expect(page.locator('#datasets .ds-feat')).toHaveCount(7);
});

test('the longer sine labels fit at phone width, in the dark theme', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 900 });
  await page.emulateMedia({ colorScheme: 'dark' });
  await open(page);
  await switchTo(page, 'circle');
  for (const id of ['sin x1', 'sin x2']) await page.locator(`#datasets .ds-feat[data-feature="${id}"]`).click();
  await page.waitForFunction(() => (window as unknown as { raster: Raster }).raster.store.net.getWeights()[0].length === 8 * 4);
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow, 'no horizontal scrolling at 390 px').toBeLessThanOrEqual(0);
  for (const chip of await page.locator('#datasets .ds-feat').all()) {
    const box = (await chip.boundingBox())!;
    expect(box.x, 'chips stay on screen').toBeGreaterThanOrEqual(0);
    expect(box.x + box.width, 'chips stay on screen').toBeLessThanOrEqual(390);
  }
  await expect(page.locator('#builder .builder-input')).toHaveText('4 features: x₁, x₂, sin πx₁, sin πx₂');
});

/** A PNG drawn in the page: `paint` gets a transparent 200 × 200 canvas. */
async function png(page: Page, paint: string): Promise<Buffer> {
  const b64 = await page.evaluate((code) => {
    const c = document.createElement('canvas');
    c.width = 200;
    c.height = 200;
    const ctx = c.getContext('2d')!;
    new Function('ctx', code)(ctx);
    return c.toDataURL('image/png').split(',')[1];
  }, paint);
  return Buffer.from(b64, 'base64');
}

test('uploads with a transparent background and light content are read by their alpha', async ({ page }) => {
  await open(page);
  // MNIST: a white 7 on a transparent background (white ink, as on a dark-mode drawing).
  const seven = await png(page, "ctx.fillStyle = '#ffffff'; ctx.font = 'bold 150px Arial'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText('7', 100, 105);");
  await page.setInputFiles('#upload-input', [{ name: 'white-seven.png', mimeType: 'image/png', buffer: seven }]);
  const up = page.locator('.upload', { hasText: 'white-seven.png' });
  await expect(up).toContainText('Predicts');
  await expect(up).not.toContainText('No ink found');
  await up.getByRole('button', { name: 'Use as input' }).click();
  const mass = await raster(page, (r) => {
    const x = (r.store as unknown as { probe: { x: Float32Array } }).probe.x;
    return { mass: x.reduce((s, v) => s + v, 0), corner: x[0] };
  });
  expect(mass.mass).toBeGreaterThan(20);
  expect(mass.corner).toBe(0);

  // Fashion-MNIST: a light grey cut-out on a transparent background.
  await switchTo(page, 'fashion');
  const shirt = await png(page, "ctx.fillStyle = '#ececec'; ctx.fillRect(60, 30, 80, 150); ctx.fillRect(25, 30, 150, 40);");
  await page.setInputFiles('#upload-input', [{ name: 'pale-shirt.png', mimeType: 'image/png', buffer: shirt }]);
  const item = page.locator('.upload', { hasText: 'pale-shirt.png' });
  await expect(item).toContainText('Predicts');
  await expect(item).not.toContainText('Nothing stands out');
});

test('the page measures the test accuracies docs/models.md quotes for it', async ({ page }) => {
  test.setTimeout(240_000);
  const doc = readFileSync('docs/models.md', 'utf8').replace(/\s+/g, ' ');
  await open(page);
  await tab(page, 'Pretrained');
  const models: [string, string, string][] = [
    ['mnist-lenet', 'MNIST LeNet', 'LeNet'],
    ['mnist-cnn', 'MNIST small CNN', 'MNIST small CNN'],
    ['fashion-cnn', 'Fashion-MNIST small CNN', 'Fashion-MNIST small CNN'],
    ['cifar10-cnn', 'CIFAR-10 CNN', 'CIFAR-10 CNN'],
  ];
  for (const [id, name, label] of models) {
    await page.click(`#zoo-load-${id}`);
    await page.waitForFunction(
      (n) => {
        const s = (window as unknown as { raster: Raster }).raster.store;
        return s.modelName === n && s.evals.length > 0;
      },
      name,
      { timeout: 90_000 },
    );
    const acc = await raster(page, (r) => r.store.evals[0].acc);
    const quoted = doc.match(new RegExp(`([\\d.]+)% \\(${label}\\)`))?.[1];
    expect(quoted, `docs/models.md on ${name}`).toBe((acc * 100).toFixed(2));
    await expect(page.locator('#model-eval')).toContainText(`Test accuracy here: ${(acc * 100).toFixed(1)}% on 2,000 test`);
  }
});
