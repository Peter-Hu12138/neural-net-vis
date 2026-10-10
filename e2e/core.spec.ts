import { expect, test, type Page } from '@playwright/test';

/**
 * 04 Weights, 05 Training, 06 Backpropagation and the speed control, for every kind of dataset
 * (grey images, colour images, points). Screenshots land in docs/screenshots/17-core-*.png.
 */

const SHOTS = 'docs/screenshots';
/** Element screenshots without the sticky training bar drawn over their top, or a stray tooltip. */
const SHOT_STYLE = '.bar { visibility: hidden !important; } #tip { display: none !important; }';

type W = {
  raster: {
    store: {
      dataset: string;
      data: unknown;
      speed: string;
      status: { running: boolean; samplesPerSec: number; step: number; epochFraction: number } | null;
      evals: { confusion: number[] }[];
      net: { blocks: unknown[]; getWeights(): Float32Array[] };
      frozen: boolean[];
    };
    actions: {
      setDataset(id: string): Promise<void>;
      setFeatures(ids: string[]): void;
      setFrozen(block: number, frozen: boolean): void;
      select(block: number, unit?: number | null): void;
      setMode(mode: string): void;
      pause(): void;
      applyWeights(w: Float32Array[]): void;
    };
  };
  __texts: string[];
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

async function open(page: Page) {
  await page.goto('/');
  await expect(page.locator('#fact-data')).toContainText('train', { timeout: 30_000 });
  await expect(page.locator('#play')).toBeEnabled();
}

/** Switches dataset through the app's actions and waits until its data is in. */
async function switchTo(page: Page, id: string) {
  await page.evaluate((d) => (window as unknown as W).raster.actions.setDataset(d), id);
  await page.waitForFunction((d) => {
    const s = (window as unknown as W).raster.store;
    return s.dataset === d && !!s.data;
  }, id, { timeout: 60_000 });
  await expect(page.locator('#play')).toBeEnabled();
}

async function trainUntil(page: Page, pred: string, timeout = 60_000) {
  await page.click('#play');
  await page.waitForFunction(pred, null, { timeout });
  await page.click('#play');
  await page.waitForFunction(() => !(window as unknown as W).raster.store.status?.running);
}

/** Records every string drawn on a canvas (fillText), to check labels the canvases print. */
async function recordCanvasText(page: Page) {
  await page.addInitScript(() => {
    const texts: string[] = [];
    (window as unknown as { __texts: string[] }).__texts = texts;
    const orig = CanvasRenderingContext2D.prototype.fillText;
    CanvasRenderingContext2D.prototype.fillText = function (this: CanvasRenderingContext2D, t: string, x: number, y: number, w?: number) {
      texts.push(String(t));
      if (texts.length > 40000) texts.splice(0, 20000);
      return w === undefined ? orig.call(this, t, x, y) : orig.call(this, t, x, y, w);
    };
  });
}
const clearTexts = (page: Page) => page.evaluate(() => ((window as unknown as W).__texts.length = 0));
const texts = (page: Page) => page.evaluate(() => (window as unknown as W).__texts.slice());

const inspCanvas = (page: Page) => page.locator('#inspector canvas');

/** Shows layer `block` in view `mode` in 04 and waits until it is drawn as `view`. */
async function inspect(page: Page, block: number, mode: string, view: string, unit: number | null = null) {
  await page.evaluate(([b, m, u]) => {
    const a = (window as unknown as W).raster.actions;
    a.select(b as number, u as number | null);
    a.setMode(m as string);
  }, [block, mode, unit] as const);
  await page.locator('#weights').scrollIntoViewIfNeeded();
  await expect(inspCanvas(page)).toHaveAttribute('data-view', view);
}

/** Number of distinctly green pixels: they never occur in the red–white–blue weight maps. */
const greenPixels = (page: Page) =>
  inspCanvas(page).evaluate((c: HTMLCanvasElement) => {
    const d = c.getContext('2d')!.getImageData(0, 0, c.width, c.height).data;
    let n = 0;
    for (let i = 0; i < d.length; i += 4) if (d[i + 1] > d[i] + 30 && d[i + 1] > d[i + 2] + 30) n++;
    return n;
  });

test('04 shows a colour network’s first-layer kernels as colour patches', async ({ page }) => {
  await open(page);
  await switchTo(page, 'cifar10');
  await trainUntil(page, '(window.raster.store.status?.step ?? 0) > 60');
  await inspect(page, 0, 'heat', 'conv-colour');
  await expect(inspCanvas(page)).toHaveAttribute(
    'aria-label',
    'Conv 1 weights: 8 filters, each shown as a 3×3 colour patch (its 3×3×3 kernel, mid grey is zero) and as red, green and blue heatmaps',
  );
  expect(await greenPixels(page), 'colour patches are drawn').toBeGreaterThan(20);
  await expect(page.locator('#inspector .hint').first()).toContainText('colour pattern that excites the filter most');

  // A patch pixel's tooltip lists its three channel weights, with true minus signs.
  await inspCanvas(page).hover({ position: { x: 8, y: 22 } });
  await expect(page.locator('#tip')).toBeVisible();
  expect(await page.locator('#tip').textContent()).toMatch(/^Filter 1 at \(0, 0\)\nR −?\d\.\d{4} · G −?\d\.\d{4} · B −?\d\.\d{4}$/);
  await page.mouse.move(0, 0);
  await page.locator('#weights').screenshot({ path: `${SHOTS}/17-core-inspector-cifar-conv.png`, style: SHOT_STYLE });

  await inspect(page, 0, 'hinton', 'conv-colour');
  await expect(inspCanvas(page)).toHaveAttribute('aria-label', /and as red, green and blue Hinton diagrams$/);
  await inspect(page, 0, 'numbers', 'conv-colour', 2);
  await expect(inspCanvas(page)).toHaveAttribute('aria-label', 'Filter 3 of Conv 1: its 3×3×3 kernel as a colour patch, and the values of its red, green and blue 3×3 kernels');

  // Histogram and Q–Q work on every layer, and the stats row uses a true minus sign.
  await inspect(page, 0, 'hist', 'hist');
  await inspect(page, 0, 'qq', 'qq');
  await expect(page.locator('.insp-stats')).toContainText('mean');
  expect(await page.locator('.insp-stats').textContent()).not.toMatch(/-\d/);

  // The output layer's rows are the class names.
  await inspect(page, 3, 'heat', 'dense-matrix');
  await expect(inspCanvas(page)).toHaveAttribute('aria-label', /one per class \(airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck\)/);
  await expect(page.locator('#insp-unit option').first()).toHaveText('airplane');

  // Without convolutions, the first dense layer's "templates" are colour images.
  await page.getByRole('button', { name: 'Softmax', exact: true }).click();
  await page.waitForFunction(() => (window as unknown as W).raster.store.evals.length >= 1);
  await trainUntil(page, '(window.raster.store.status?.step ?? 0) > 300');
  await inspect(page, 0, 'heat', 'dense-templates-colour');
  await expect(inspCanvas(page)).toHaveAttribute('aria-label', 'Output weights: each of the 10 classes’ 3,072 weights as a 32×32 colour image, mid grey is zero');
  expect(await greenPixels(page)).toBeGreaterThan(20);
  await page.locator('#weights').screenshot({ path: `${SHOTS}/17-core-inspector-cifar-templates.png`, style: SHOT_STYLE });

  await page.emulateMedia({ colorScheme: 'dark' });
  await page.getByRole('button', { name: 'Small CNN', exact: true }).click();
  await inspect(page, 0, 'heat', 'conv-colour');
  await page.waitForTimeout(200);
  await page.locator('#weights').screenshot({ path: `${SHOTS}/17-core-inspector-cifar-dark.png`, style: SHOT_STYLE });
});

test('04 labels a point network’s weights with its input features', async ({ page }) => {
  await recordCanvasText(page);
  await open(page);
  await switchTo(page, 'circle');
  await page.evaluate(() => (window as unknown as W).raster.actions.setFeatures(['x1', 'x2', 'x1^2', 'sin x1']));
  await page.waitForFunction(() => (window as unknown as W).raster.store.net.getWeights()[0].length === 8 * 4);
  await clearTexts(page);
  await inspect(page, 0, 'heat', 'dense-matrix');
  await expect(inspCanvas(page)).toHaveAttribute('aria-label', 'Dense 1 weights as a heatmap: 8 rows, one per unit, by the input features (x₁, x₂, x₁², sin x₁), plus a bias column');
  await expect.poll(async () => (await texts(page)).includes('x₁²')).toBe(true);
  const drawn = await texts(page);
  for (const t of ['x₁', 'x₂', 'x₁²', 'sin x₁', 'u1', 'u8', 'b']) expect(drawn, t).toContain(t);

  // A cell's tooltip names the unit and the feature.
  let tip = '';
  for (let x = 30; x < 200 && !tip.startsWith('W['); x += 6) {
    await inspCanvas(page).hover({ position: { x, y: 30 } });
    tip = (await page.locator('#tip').isVisible()) ? ((await page.locator('#tip').textContent()) ?? '') : '';
  }
  expect(tip).toMatch(/^W\[u\d ← (x₁|x₂|x₁²|sin x₁)\] = −?\d\.\d{4}$/);
  await page.mouse.move(0, 0);
  await page.locator('#weights').screenshot({ path: `${SHOTS}/17-core-inspector-points.png`, style: SHOT_STYLE });

  // Numbers, Hinton, histogram and Q–Q all draw; the output rows are the classes.
  await inspect(page, 0, 'numbers', 'dense-matrix');
  await inspect(page, 0, 'hinton', 'dense-matrix');
  await inspect(page, 2, 'heat', 'dense-matrix');
  await expect(inspCanvas(page)).toHaveAttribute('aria-label', /one per class \(Class 0, Class 1\), by 8 inputs/);
  await inspect(page, 2, 'hist', 'hist');
  await inspect(page, 2, 'qq', 'qq');
});

test('05 names the classes of Fashion-MNIST in the confusion matrix', async ({ page }) => {
  await recordCanvasText(page);
  await open(page);
  await page.locator('#training').scrollIntoViewIfNeeded();
  await clearTexts(page);
  const loaded = switchTo(page, 'fashion');
  await expect.poll(async () => (await texts(page)).some((t) => t.startsWith('Loading Fashion-MNIST…')), { timeout: 30_000 }).toBe(true);
  await loaded;
  await page.waitForFunction(() => (window as unknown as W).raster.store.evals.length >= 1);
  await trainUntil(page, '(window.raster.store.evals.length >= 2)');
  await page.locator('#training').scrollIntoViewIfNeeded();
  const conf = page.locator('#curves .confusion-wrap canvas');
  await expect(conf).toHaveAttribute('aria-label', /^Confusion matrix on the test set, 10 by 10 \(T-shirt\/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot\): rows are the true class, columns the prediction; [\d,]+ of 2,000 correct\.$/);
  await expect.poll(async () => (await texts(page)).includes('Sneaker')).toBe(true);
  const drawn = await texts(page);
  for (const t of ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']) expect(drawn, t).toContain(t);
  await expect(page.locator('#curves .confusion-wrap')).toHaveClass(/is-wide/);
  await expect(page.locator('#curves .conf-list')).toContainText('Most confused');
  await expect(page.locator('#curves .conf-list')).toContainText('test images');
  const pair = await page.locator('#curves .conf-pair').first().textContent();
  expect(pair).toMatch(/^(T-shirt\/top|Trouser|Pullover|Dress|Coat|Sandal|Shirt|Sneaker|Bag|Ankle boot) → (T-shirt\/top|Trouser|Pullover|Dress|Coat|Sandal|Shirt|Sneaker|Bag|Ankle boot)\s+\d+×$/);

  // Full names in the tooltip.
  const box = (await conf.boundingBox())!;
  await conf.hover({ position: { x: box.width - 12, y: box.height - 12 } });
  await expect(page.locator('#tip')).toBeVisible();
  expect(await page.locator('#tip').textContent()).toMatch(/^true Ankle boot → predicted Ankle boot\n[\d,]+ of the [\d,]+ test images labelled Ankle boot \([\d.<>]+%\)$/);
  await page.mouse.move(0, 0);
  await page.locator('#training').screenshot({ path: `${SHOTS}/17-core-training-fashion.png`, style: SHOT_STYLE });
});

test('05 on a point dataset: large confusion cells and an epoch axis for long runs', async ({ page }) => {
  await recordCanvasText(page);
  await page.setViewportSize({ width: 390, height: 844 });
  await open(page);
  await switchTo(page, 'circle');
  // On a phone the speed control is a menu in the button row.
  await page.locator('#speed-select').selectOption('max');
  // Long enough for several test evaluations (the trainer runs one at most every 250 ms).
  await trainUntil(page, '(window.raster.store.status?.epochFraction ?? 0) > 60 && window.raster.store.evals.length >= 5', 90_000);
  await page.locator('#training').scrollIntoViewIfNeeded();
  await clearTexts(page);
  // A theme change redraws every canvas; use it to collect one full set of labels.
  await page.evaluate(() => document.documentElement.setAttribute('data-theme', 'light'));
  await expect.poll(async () => (await texts(page)).includes('epoch')).toBe(true);
  const drawn = await texts(page);
  // The epoch axis ticks are round: 0, then multiples of 1, 2 or 5 × 10ⁿ, evenly spaced.
  const ticks = ((await page.locator('#curves canvas').first().getAttribute('data-x-ticks')) ?? '').split(',').map(Number);
  expect(ticks[0]).toBe(0);
  expect(ticks.length).toBeGreaterThanOrEqual(3);
  const step = ticks[1];
  expect([1, 2, 5]).toContain(step / 10 ** Math.floor(Math.log10(step)));
  ticks.forEach((t, i) => expect(t).toBe(i * step));
  expect(ticks.at(-1)!).toBeGreaterThan(60);
  for (const t of ticks.slice(1)) expect(drawn).toContain(t.toLocaleString('en-US'));
  await expect(page.locator('#curves canvas').first()).toHaveAttribute('aria-label', /epochs so far/);
  // Two classes: full names with their colour swatch, and a count and share in every cell.
  expect(drawn).toContain('Class 0');
  expect(drawn.some((t) => /^\d+\.\d%$/.test(t))).toBe(true);
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  expect(overflow).toBeLessThanOrEqual(0);
  await page.locator('#training').screenshot({ path: `${SHOTS}/17-core-training-points-phone.png`, style: SHOT_STYLE });
});

test('06 steps through every stage on CIFAR-10 and on a point dataset', async ({ page }) => {
  await open(page);
  for (const id of ['cifar10', 'circle']) {
    await switchTo(page, id);
    await trainUntil(page, '(window.raster.store.status?.step ?? 0) > 20');
    await page.locator('#backprop').scrollIntoViewIfNeeded();
    const titles = await page.locator('.steps li:not(.phase) button').allTextContents();
    expect(titles[0]).toContain(id === 'cifar10' ? 'Input image' : 'Input point');
    expect(titles.at(-1)).toContain('Gradient descent step');
    expect(titles).toHaveLength(id === 'cifar10' ? 23 : 15);
    await page.locator('.steps li:not(.phase) button').first().click();
    for (let i = 0; i < titles.length; i++) {
      const title = ((await page.locator('.bp-title h3').textContent()) ?? '').trim();
      expect(title).toBe(titles[i].replace(/^\d+/, '').trim());
      await expect(page.locator('.bp-main .formula').first()).toBeVisible();
      const main = page.locator('.bp-main');
      if (i === 0 && id === 'cifar10') {
        await expect(main).toContainText('red, a green and a blue value');
        await expect(main.locator('canvas').first()).toHaveAttribute('aria-label', /the input image, 32×32 pixels in colour/);
        await expect(main.locator('figcaption', { hasText: 'Channels' })).toBeVisible();
        await page.locator('#backprop').screenshot({ path: `${SHOTS}/17-core-backprop-cifar-input.png`, style: SHOT_STYLE });
      }
      if (i === 0 && id === 'circle') {
        await expect(main).toContainText('x₁');
        await expect(main.locator('pre.work')).toContainText('coordinate 1');
        await expect(main.locator('canvas').first()).toHaveAttribute('aria-label', /2 values for x₁, x₂/);
        await page.locator('#backprop').screenshot({ path: `${SHOTS}/17-core-backprop-points-input.png`, style: SHOT_STYLE });
      }
      if (title === 'Conv 1: convolution') {
        await expect(main).toContainText('3×3×3 kernel');
        await expect(main.locator('pre.work')).toContainText('green:');
      }
      if (title === 'Conv 1: gradients') {
        await expect(main).toContainText('one map per colour channel');
        await page.locator('#backprop').screenshot({ path: `${SHOTS}/17-core-backprop-cifar-conv-gradients.png`, style: SHOT_STYLE });
      }
      if (title === 'Softmax' && id === 'cifar10') {
        await expect(main.locator('pre.work')).toContainText('airplane');
        await expect(main.locator('pre.work')).toContainText('automobile');
      }
      if (title === 'Dense 1: weighted sum' && id === 'circle') await expect(main.locator('pre.work')).toContainText(/W\[\d, x[₁₂]\] · x[₁₂]/);
      if (title === 'Cross-entropy loss' && id === 'cifar10') await expect(main).toContainText(/The network predicts (airplane|automobile|bird|cat|deer|dog|frog|horse|ship|truck)/);
      if (i < titles.length - 1) await page.getByRole('button', { name: 'Next →' }).click();
    }
    await expect(page.locator('.dir')).toHaveText('Update');
    // The output layer always learns (its biases at least, even when every input to it is 0).
    const outParams = () => page.evaluate(() => (window as unknown as W).raster.store.net.getWeights().slice(-2).flatMap((w) => Array.from(w)));
    const w0 = await outParams();
    await page.getByRole('button', { name: 'Apply to network' }).click();
    await expect(page.locator('#bplab .notice')).toContainText(`The loss on this ${id === 'cifar10' ? 'image' : 'point'} went from`);
    expect(await outParams()).not.toEqual(w0);
  }
  // The side panel names the classes of the target, and the example is a map of the plane.
  await expect(page.locator('#bplab .bp-side')).toContainText('Target class y');
  await expect(page.locator('#bplab .bp-sample')).toHaveClass(/is-points/);
});

test('06 update step leaves frozen layers alone, and 04 marks them', async ({ page }) => {
  await open(page);
  await trainUntil(page, '(window.raster.store.status?.step ?? 0) > 10');
  await page.evaluate(() => (window as unknown as W).raster.actions.setFrozen(0, true));

  // 04: the marker and the layer menu.
  await inspect(page, 0, 'heat', 'conv');
  await expect(page.locator('#inspector .insp-frozen')).toBeVisible();
  await expect(page.locator('#inspector .insp-frozen')).toContainText('Frozen: training leaves these weights alone.');
  await expect(page.locator('#insp-layer option').first()).toContainText('frozen');
  await inspect(page, 1, 'heat', 'conv-grid');
  await expect(page.locator('#inspector .insp-frozen')).toBeHidden();

  // 06: Conv 1's gradient step explains that training skips it; the update leaves it alone.
  await page.locator('#backprop').scrollIntoViewIfNeeded();
  await page.locator('.steps li:not(.phase) button', { hasText: 'Conv 1: gradients' }).click();
  await expect(page.locator('.bp-main .bp-frozen')).toContainText('Frozen: training leaves these weights alone.');
  await expect(page.locator('.bp-main .bp-frozen')).toContainText('training skips this step entirely');
  await page.locator('.steps li:not(.phase) button', { hasText: 'Conv 2: gradients' }).click();
  await expect(page.locator('.bp-main .bp-frozen')).toHaveCount(0);
  await page.locator('.steps li:not(.phase) button').last().click();
  const main = page.locator('.bp-main');
  await expect(main.locator('.bp-frozen')).toContainText('Frozen: training leaves Conv 1 alone.');
  await expect(main.locator('.bp-frozen')).toContainText('ΔW = 0');
  // Columns: ‖∂L/∂W‖, ‖∂L/∂b‖, then the step ‖ΔW‖, which is 0 for the frozen layer.
  await expect(main.locator('pre.work')).toContainText(/Conv 1\s+[\d.]+\s+[\d.]+\s+0\.0000\s+frozen, not updated/);
  await page.locator('#backprop').screenshot({ path: `${SHOTS}/17-core-backprop-frozen.png`, style: SHOT_STYLE });

  const before = await page.evaluate(() => (window as unknown as W).raster.store.net.getWeights().map((w) => Array.from(w.slice(0, 30))));
  await page.getByRole('button', { name: 'Apply to network' }).click();
  await expect(page.locator('#bplab .notice')).toContainText('Applied');
  const after = await page.evaluate(() => (window as unknown as W).raster.store.net.getWeights().map((w) => Array.from(w.slice(0, 30))));
  expect(after[0], 'frozen Conv 1 kernels unchanged').toEqual(before[0]);
  expect(after[1], 'frozen Conv 1 biases unchanged').toEqual(before[1]);
  expect(after[2], 'Conv 2 trained').not.toEqual(before[2]);

  // Frozen everywhere: nothing to apply.
  await page.evaluate(() => {
    const a = (window as unknown as W).raster.actions;
    for (let i = 0; i < 4; i++) a.setFrozen(i, true);
  });
  await expect(main.locator('.bp-frozen')).toContainText('Frozen: training leaves every layer alone.');
  await expect(page.getByRole('button', { name: 'Apply to network' })).toBeDisabled();
});

test('06 explains a gradient that stops at a layer whose units were all off', async ({ page }) => {
  await open(page);
  // Push every Dense 3 bias far below zero: all its ReLUs are off for any input.
  await page.evaluate(() => {
    const r = (window as unknown as W).raster;
    const w = r.store.net.getWeights();
    w[5] = w[5].map(() => -100);
    r.actions.applyWeights(w);
  });
  await page.locator('#backprop').scrollIntoViewIfNeeded();
  await expect(page.locator('#bplab .bp-side')).toContainText("Using the network's weights at step 0");
  const step = (name: string) => page.locator('.steps li:not(.phase) button', { hasText: name }).click();
  const note = page.locator('.bp-main .bp-dead');

  await step('Dense 3: back through ReLU');
  await expect(note).toContainText('Every unit of Dense 3 was off for this digit.');
  await expect(note).toContainText('the gradient stops here');
  await step('Output layer: gradients');
  await expect(note).toContainText('Every input to Output is 0 for this digit.');
  await expect(note).toContainText('only the biases learn');
  await step('Conv 2: gradients');
  await expect(note).toContainText('No gradient reaches Conv 2 from this digit.');
  await expect(page.locator('.bp-main pre.work')).toContainText('every term is 0');
  await page.locator('.steps li:not(.phase) button').last().click();
  await expect(page.locator('.bp-main pre.work')).toContainText('Every weight gradient in the output layer is 0 for this digit; one bias:');
  await expect(page.locator('.bp-main pre.work')).toContainText(/b\[\d\] ← /);

  // Applying the step moves only the output biases.
  const before = await page.evaluate(() => (window as unknown as W).raster.store.net.getWeights().map((w) => Array.from(w)));
  await page.getByRole('button', { name: 'Apply to network' }).click();
  await expect(page.locator('#bplab .notice')).toContainText('Applied');
  const after = await page.evaluate(() => (window as unknown as W).raster.store.net.getWeights().map((w) => Array.from(w)));
  for (let i = 0; i < 7; i++) expect(after[i], `parameter block ${i} unchanged`).toEqual(before[i]);
  expect(after[7], 'output biases moved').not.toEqual(before[7]);
});

test('the speed control caps the trainer and follows the dataset', async ({ page }) => {
  await open(page);
  const speed = page.locator('#speed');
  await expect(speed.getByRole('button', { name: 'Max' })).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator('.bar .stat .label').nth(2)).toHaveText('Digits/s');
  await switchTo(page, 'circle');
  // Point datasets start at Normal (setDataset changes the speed per kind of data).
  await expect(speed.getByRole('button', { name: 'Normal' })).toHaveAttribute('aria-pressed', 'true');
  await expect(speed.getByRole('button', { name: 'Normal' })).toHaveAttribute('title', 'Normal caps training at 3,000 points a second, so you can watch a small network learn');
  await expect(page.locator('.bar .stat .label').nth(2)).toHaveText('Points/s');

  const rates = async (ms: number) => {
    const out: number[] = [];
    const end = Date.now() + ms;
    while (Date.now() < end) {
      const s = await page.evaluate(() => (window as unknown as W).raster.store.status);
      if (s?.running && s.samplesPerSec > 0) out.push(s.samplesPerSec);
      await page.waitForTimeout(150);
    }
    return out;
  };
  await page.click('#play');
  await page.waitForTimeout(1500);
  const normal = await rates(2000);
  expect(normal.length).toBeGreaterThan(3);
  expect(Math.max(...normal), 'Normal stays near 3,000 a second').toBeLessThanOrEqual(3300);
  expect(Math.max(...normal)).toBeGreaterThan(2000);

  await speed.getByRole('button', { name: 'Max' }).click();
  await expect(speed.getByRole('button', { name: 'Max' })).toHaveAttribute('aria-pressed', 'true');
  expect(await page.evaluate(() => (window as unknown as W).raster.store.speed)).toBe('max');
  await page.waitForTimeout(1500);
  const max = await rates(1500);
  expect(Math.max(...max), 'Max lifts the cap').toBeGreaterThan(5000);

  await speed.getByRole('button', { name: 'Slow' }).click();
  await page.waitForTimeout(2000);
  const slow = await rates(1500);
  expect(Math.max(...slow), 'Slow stays near 300 a second').toBeLessThanOrEqual(400);
  await page.click('#play');
  await page.waitForFunction(() => !(window as unknown as W).raster.store.status?.running);
  await page.addStyleTag({ content: '#tip { display: none !important; }' });
  await page.locator('.bar').screenshot({ path: `${SHOTS}/17-core-bar.png` });

  // Back to images: the speed goes back to Max, and the control shows it.
  await switchTo(page, 'mnist');
  await expect(speed.getByRole('button', { name: 'Max' })).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator('.bar .stat .label').nth(2)).toHaveText('Digits/s');

  // Phone width: the choice moves into the button row as a menu, so the sticky bar does not grow.
  await page.setViewportSize({ width: 390, height: 844 });
  await page.waitForTimeout(300);
  await expect(speed).toBeHidden();
  const menu = page.locator('#speed-select');
  await expect(menu).toBeVisible();
  await expect(menu).toHaveValue('max');
  const [menuTop, playTop] = await Promise.all([menu.evaluate((e) => e.getBoundingClientRect().bottom), page.locator('#play').evaluate((e) => e.getBoundingClientRect().bottom)]);
  expect(Math.abs(menuTop - playTop), 'menu sits in the button row').toBeLessThan(16);
  expect(await page.locator('.bar').evaluate((e) => e.getBoundingClientRect().height)).toBeLessThanOrEqual(180);
  await menu.selectOption('slow');
  expect(await page.evaluate(() => (window as unknown as W).raster.store.speed)).toBe('slow');
  await expect(menu).toHaveAttribute('title', /^Slow caps training at 300 digits a second/);
  await menu.selectOption('max');
  expect(await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)).toBeLessThanOrEqual(0);
  await page.locator('.bar').screenshot({ path: `${SHOTS}/17-core-bar-phone.png` });
});
