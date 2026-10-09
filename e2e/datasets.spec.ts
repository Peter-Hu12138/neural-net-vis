import { expect, test, type Page } from '@playwright/test';

/**
 * The dataset strip (picker, train-on subset, point-data settings), section 07 for every kind of
 * data, and section 03's photo mode. Screenshots land in docs/screenshots/14-datasets-*.png.
 */

const SHOTS = 'docs/screenshots';

type Raster = {
  store: {
    dataset: string;
    features: string[];
    trainLimit: number | null;
    pointsConfig: { count: number; noise: number; trainRatio: number; seed: number };
    data: {
      info: { id: string };
      input: { c: number; h: number; w: number };
      trainY: Uint8Array;
      testY: Uint8Array;
      points?: { trainCoords: Float32Array; testCoords: Float32Array; dims: number };
    } | null;
    probe: { key: string; caption: string; label: number | null; x: Float32Array; coords?: Float32Array } | null;
    custom: { y: number }[];
    net: { inputSize: number };
  };
  actions: { setDataset(id: string): Promise<void> };
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
}

const item = (page: Page, name: string) => page.locator('#datasets').getByRole('button', { name, exact: true });

/** Records every text the loading line shows, so short-lived "Loading …" states can be checked. */
async function watchStatus(page: Page) {
  await page.evaluate(() => {
    const w = window as unknown as { statusTexts: string[] };
    w.statusTexts = [];
    const el = document.querySelector('#datasets .dsp-status')!;
    new MutationObserver(() => w.statusTexts.push(el.textContent ?? '')).observe(el, { childList: true, characterData: true, subtree: true });
  });
}
const statusTexts = (page: Page) => page.evaluate(() => (window as unknown as { statusTexts: string[] }).statusTexts);

/** Picks a dataset in the strip and waits until its data is in. */
async function pick(page: Page, name: string, id: string) {
  await item(page, name).click();
  await page.waitForFunction((want) => (window as unknown as { raster: Raster }).raster.store.data?.info.id === want, id, { timeout: 60_000 });
  await expect(item(page, name)).toHaveAttribute('aria-pressed', 'true');
}

/** Only the chosen dataset is marked. */
async function onlySelected(page: Page, name: string) {
  const pressed = await page.locator('#datasets .dsp-item[aria-pressed="true"]').allTextContents();
  expect(pressed.map((t) => t.trim())).toEqual([name]);
}

const noOverflow = async (page: Page) => expect(await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)).toBeLessThanOrEqual(0);

/** Screenshot without the sticky control bar covering the element. */
async function shot(page: Page, selector: string, name: string) {
  await page.mouse.move(0, 0);
  const style = await page.addStyleTag({ content: '#bar { position: static !important; }' });
  await page.locator(selector).screenshot({ path: `${SHOTS}/${name}` });
  await style.evaluate((el) => (el as Element).remove());
}

/** A colour PNG made in the page: sky, grass and a red box, wider than tall. */
async function colourPng(page: Page): Promise<Buffer> {
  const b64 = await page.evaluate(() => {
    const c = document.createElement('canvas');
    c.width = 480;
    c.height = 320;
    const ctx = c.getContext('2d')!;
    ctx.fillStyle = '#5b9bd5';
    ctx.fillRect(0, 0, 480, 200);
    ctx.fillStyle = '#4f8a3a';
    ctx.fillRect(0, 200, 480, 120);
    ctx.fillStyle = '#c0392b';
    ctx.fillRect(170, 120, 150, 110);
    ctx.fillStyle = '#222';
    ctx.fillRect(190, 230, 30, 30);
    ctx.fillRect(270, 230, 30, 30);
    return c.toDataURL('image/png').split(',')[1];
  });
  return Buffer.from(b64, 'base64');
}

test('the picker switches between image and point datasets, and 07 follows', async ({ page }) => {
  await open(page);
  await expect(item(page, 'MNIST digits')).toHaveAttribute('aria-pressed', 'true');
  await onlySelected(page, 'MNIST digits');
  await expect(page.locator('#datasets .dsp-item')).toHaveCount(14);
  for (const group of ['Images', 'Points in 2D', 'Points in 3D']) await expect(page.locator('#datasets .dsp-group-label', { hasText: group })).toBeVisible();
  // The strip stays compact: index plus the line under it.
  const strip = await page.locator('#datasets .dsp-index').boundingBox();
  const row = await page.locator('#datasets .dsp-row').boundingBox();
  expect(row!.y + row!.height - strip!.y).toBeLessThan(175);
  await expect(page.locator('#datasets .dsp-points')).toBeHidden();
  await expect(page.locator('#datasets .dsp-about')).toContainText('Handwritten digits');
  await shot(page, '#datasets', '14-datasets-picker-light.png');

  // ── Fashion-MNIST ──
  await watchStatus(page);
  await pick(page, 'Fashion-MNIST', 'fashion');
  await onlySelected(page, 'Fashion-MNIST');
  expect((await statusTexts(page)).some((t) => /^Loading Fashion-MNIST/.test(t))).toBe(true);
  await expect(page.locator('#fact-data')).toHaveText('Fashion-MNIST · 10,000 train · 2,000 test');
  await expect(page.locator('#fact-input')).toHaveText('28 × 28 px, grey');
  await expect(page.locator('#datapanel .sample-grid').first().locator('.thumb')).toHaveCount(60);
  await expect(page.locator('#datapanel')).toContainText('Test images · 60 of 2,000');
  await expect(page.locator('#datapanel .dp-key')).toContainText('Ankle boot');
  await expect(page.locator('#datapanel .drop')).toContainText('Drop photos of clothing here');
  // 03 keeps the drawing pad, with a photo option and named classes.
  await expect(page.locator('#h-draw')).toHaveText('Draw');
  await expect(page.locator('.pad')).toBeVisible();
  await expect(page.locator('#drawpad .bar-row')).toHaveCount(10);
  await expect(page.locator('#drawpad .bars')).toContainText('Ankle boot');

  // ── CIFAR-10 ──
  await watchStatus(page);
  await pick(page, 'CIFAR-10', 'cifar10');
  await onlySelected(page, 'CIFAR-10');
  const texts = await statusTexts(page);
  expect(texts.some((t) => /^Loading CIFAR-10/.test(t))).toBe(true);
  expect(texts.some((t) => /^Loading CIFAR-10 · \d of 4 files$/.test(t))).toBe(true);
  await expect(page.locator('#datasets .dsp-status')).toBeHidden();
  await expect(page.locator('#fact-data')).toHaveText('CIFAR-10 · 10,000 train · 2,000 test');
  await expect(page.locator('#fact-input')).toHaveText('32 × 32 px, colour');
  expect(await raster(page, (r) => r.store.data!.input)).toEqual({ c: 3, h: 32, w: 32 });
  const grid = page.locator('#datapanel .sample-grid').first();
  await expect(grid.locator('.thumb')).toHaveCount(60);
  // Thumbnails are in colour: some pixel has clearly different red and blue.
  const colourful = await grid.locator('.thumb canvas').first().evaluate((c) => {
    const cv = c as HTMLCanvasElement;
    const d = cv.getContext('2d')!.getImageData(0, 0, cv.width, cv.height).data;
    let n = 0;
    for (let i = 0; i < d.length; i += 4) if (Math.abs(d[i] - d[i + 2]) > 30) n++;
    return n;
  });
  expect(colourful).toBeGreaterThan(0);
  // Clicking a test image makes it the input.
  await grid.locator('.thumb').nth(3).click();
  expect(await raster(page, (r) => r.store.probe!.key)).toMatch(/^test:\d+$/);
  await expect(grid.locator('.thumb').nth(3)).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator('#h-draw')).toHaveText('Try a photo');
  await expect(page.locator('.pad')).toBeHidden();

  // A colour upload in 07: centre-cropped to 32×32, classified, and it can join the training set.
  await page.setInputFiles('#upload-input', [{ name: 'red-box.png', mimeType: 'image/png', buffer: await colourPng(page) }]);
  const up = page.locator('.upload', { hasText: 'red-box.png' });
  await expect(up).toContainText('Predicts');
  expect(await raster(page, (r) => [r.store.probe!.key.startsWith('upload:'), r.store.probe!.x.length])).toEqual([true, 3072]);
  await expect(up.getByRole('button', { name: 'Train on it' })).toBeDisabled();
  await up.locator('select').selectOption({ label: 'automobile' });
  await up.getByRole('button', { name: 'Train on it' }).click();
  await expect(up.locator('.tag')).toHaveText('In training set');
  expect(await raster(page, (r) => r.store.custom.map((c) => c.y))).toEqual([1]);
  await shot(page, '#data', '14-datasets-data-cifar.png');

  // ── Circle (2-D points) ──
  await watchStatus(page);
  await pick(page, 'Circle', 'circle');
  await onlySelected(page, 'Circle');
  await expect(page.locator('#fact-data')).toHaveText('Circle · 300 train · 300 test');
  await expect(page.locator('#fact-input')).toHaveText('2 features from 2-D points');
  await expect(page.locator('#datapanel .sample-grid').first()).toBeHidden();
  await expect(page.locator('#datapanel .dp-plot')).toHaveCount(2);
  await expect(page.locator('#datapanel .dp-counts')).toContainText('Class 0');
  await expect(page.locator('#datapanel .dp-vec')).toContainText('Network input (features)');
  await expect(page.locator('#datasets .dsp-points')).toBeVisible();
  await expect(page.locator('#datasets .ds-feat')).toHaveCount(7);

  // Keyboard: a focused plot steps through its points.
  await page.locator('#datapanel .dp-plot').nth(1).focus();
  await page.keyboard.press('ArrowRight');
  const key1 = await raster(page, (r) => r.store.probe!.key);
  expect(key1).toMatch(/^test:\d+$/);
  await page.keyboard.press('ArrowRight');
  expect(await raster(page, (r) => r.store.probe!.key)).not.toBe(key1);
  // Clicking a training point makes it the input, with its raw coordinates.
  const target = await raster(page, (r) => {
    const d = r.store.data!;
    const c = d.points!;
    let m = 0;
    for (const a of [c.trainCoords, c.testCoords]) for (const v of a) m = Math.max(m, Math.abs(v));
    const dom = Math.min(2, Math.max(1.25, Math.ceil(m / 0.25) * 0.25));
    return { u: c.trainCoords[0], v: c.trainCoords[1], dom };
  });
  const plot = (await page.locator('#datapanel .dp-plot').first().boundingBox())!;
  const s = (plot.width - 12) / (2 * target.dom);
  await page.mouse.click(plot.x + 6 + (target.u + target.dom) * s, plot.y + 6 + (target.dom - target.v) * s);
  const probe = await raster(page, (r) => ({ key: r.store.probe!.key, coords: Array.from(r.store.probe!.coords ?? []) }));
  expect(probe.key).toMatch(/^train:\d+$/);
  expect(probe.coords).toHaveLength(2);
  await expect(page.locator('#datapanel .dp-caption')).toContainText('Training point #');

  // Noise and Regenerate give new points.
  const before = await raster(page, (r) => Array.from(r.store.data!.points!.trainCoords.slice(0, 6)));
  await page.locator('#dsp-noise').fill('0.3');
  await page.waitForFunction(() => (window as unknown as { raster: Raster }).raster.store.pointsConfig.noise === 0.3);
  await expect(page.locator('#datasets output[for="dsp-noise"]')).toHaveText('0.30');
  const noisy = await raster(page, (r) => Array.from(r.store.data!.points!.trainCoords.slice(0, 6)));
  expect(noisy).not.toEqual(before);
  const seed = await raster(page, (r) => r.store.pointsConfig.seed);
  await page.locator('#dsp-regenerate').click();
  expect(await raster(page, (r) => r.store.pointsConfig.seed)).toBe(seed + 1);
  expect(await raster(page, (r) => Array.from(r.store.data!.points!.trainCoords.slice(0, 6)))).not.toEqual(noisy);

  // Point count and training share.
  await page.locator('#datasets .dsp-points .seg button', { hasText: '1,000' }).click();
  await page.waitForFunction(() => {
    const d = (window as unknown as { raster: Raster }).raster.store.data!;
    return d.trainY.length + d.testY.length === 1000;
  });
  await page.locator('#dsp-share').fill('0.8');
  await page.waitForFunction(() => (window as unknown as { raster: Raster }).raster.store.data!.trainY.length === 800);
  await expect(page.locator('#fact-data')).toHaveText('Circle · 800 train · 200 test');

  // A feature toggle changes what the network reads: x₁² adds an input.
  await page.locator('#datasets .ds-feat[data-feature="x1^2"]').click();
  await expect(page.locator('#datasets .ds-feat[data-feature="x1^2"]')).toHaveAttribute('aria-pressed', 'true');
  expect(await raster(page, (r) => [r.store.net.inputSize, r.store.features.join()])).toEqual([3, 'x1,x2,x1^2']);
  await expect(page.locator('#fact-input')).toHaveText('3 features from 2-D points');
  await expect(page.locator('#datapanel .dp-vec')).toContainText('x12'); // x₁² as x<sub>1</sub><sup>2</sup>
  // Turning features off down to one: the last one stays on.
  await page.locator('#datasets .ds-feat[data-feature="x1"]').click();
  await page.locator('#datasets .ds-feat[data-feature="x2"]').click();
  await expect(page.locator('#datasets .ds-feat[data-feature="x1^2"]')).toHaveAttribute('aria-disabled', 'true');
  await page.locator('#datasets .ds-feat[data-feature="x1^2"]').click({ force: true }); // refused: it is the last one
  expect(await raster(page, (r) => r.store.features)).toEqual(['x1^2']);
  await expect(page.locator('#datasets .dsp-feat-note')).toContainText('Keep at least one feature on');
  await page.locator('#datasets .ds-feat[data-feature="x2^2"]').click();
  expect(await raster(page, (r) => r.store.net.inputSize)).toBe(2);

  // Train on a subset of the points.
  const train = page.locator('#datasets .dsp-train');
  await expect(train).toContainText('of 800 points');
  await train.getByRole('button', { name: '50', exact: true }).click();
  await page.waitForFunction(() => (window as unknown as { raster: Raster }).raster.store.data!.trainY.length === 50);
  await expect(train.getByRole('button', { name: '50', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator('#fact-data')).toHaveText('Circle · 50 train · 200 test');
  await expect(page.locator('#datapanel .dp-counts .dp-total')).toContainText('50');
  await shot(page, '#data', '14-datasets-data-points.png');
  await train.getByRole('button', { name: 'All', exact: true }).click();
  await page.waitForFunction(() => (window as unknown as { raster: Raster }).raster.store.data!.trainY.length === 800);

  // ── Double helix (3-D points) ──
  await pick(page, 'Double helix', 'helix');
  await onlySelected(page, 'Double helix');
  await expect(page.locator('#fact-input')).toHaveText('3 features from 3-D points');
  await expect(page.locator('#datapanel .dp-plot')).toHaveCount(4); // x₁–x₂ and x₁–x₃, training and test
  await expect(page.locator('#datasets .ds-feat')).toHaveCount(12);
  expect(await raster(page, (r) => r.store.probe!.coords!.length)).toBe(3);
  await expect(page.locator('#datapanel .dp-vec')).toContainText('x3');

  // ── Back to MNIST: the image subset sizes come back, the point limit does not carry over ──
  await pick(page, 'MNIST digits', 'mnist');
  await onlySelected(page, 'MNIST digits');
  await expect(page.locator('#fact-data')).toHaveText('MNIST digits · 20,000 train · 2,000 test');
  await expect(page.locator('#fact-input')).toHaveText('28 × 28 px, grey');
  await expect(page.locator('#datasets .dsp-points')).toBeHidden();
  await expect(page.locator('#datapanel .sample-grid').first().locator('.thumb')).toHaveCount(60);
  await expect(page.locator('#datapanel .dp-plot').first()).toBeHidden();
  await expect(page.locator('.pad')).toBeVisible();
  await page.locator('#datasets .dsp-train').getByRole('button', { name: '1,000', exact: true }).click();
  await page.waitForFunction(() => (window as unknown as { raster: Raster }).raster.store.data!.trainY.length === 1000);
  await expect(page.locator('#fact-data')).toHaveText('MNIST digits · 1,000 train · 2,000 test');
  await page.locator('#datasets .dsp-train').getByRole('button', { name: 'All', exact: true }).click();
  await page.waitForFunction(() => (window as unknown as { raster: Raster }).raster.store.data!.trainY.length === 20_000);
  // The real MNIST samples now show in its preview tile; the strip never widens the page.
  await noOverflow(page);
});

test('section 03 tries a photo on CIFAR-10: crop, live probabilities, flip and brightness', async ({ page }) => {
  await open(page);
  await pick(page, 'CIFAR-10', 'cifar10');
  await page.locator('#draw').scrollIntoViewIfNeeded();
  await expect(page.locator('#h-draw')).toHaveText('Try a photo');
  // Before any photo, the panel mirrors the page's input (test image #0).
  await expect(page.locator('#drawpad .dp-photo-caption')).toContainText('Test image #0');
  await expect(page.locator('#drawpad .bar-row')).toHaveCount(10);

  await page.setInputFiles('#photo-input', [{ name: 'photo.png', mimeType: 'image/png', buffer: await colourPng(page) }]);
  await page.waitForFunction(() => (window as unknown as { raster: Raster }).raster.store.probe?.key === 'photo');
  expect(await raster(page, (r) => [r.store.probe!.caption, r.store.probe!.label, r.store.probe!.x.length])).toEqual(['Your photo', null, 3072]);
  // The 32×32 input is the centre square of the 480×320 photo: sky at the top, grass at the bottom.
  const x = await raster(page, (r) => Array.from(r.store.probe!.x));
  const rgb = (row: number, col: number) => [x[row * 32 + col], x[1024 + row * 32 + col], x[2048 + row * 32 + col]];
  expect(rgb(1, 1)[2]).toBeGreaterThan(rgb(1, 1)[0]); // blue sky
  expect(rgb(30, 1)[1]).toBeGreaterThan(rgb(30, 1)[0]); // green grass
  // Live class probabilities, most likely first, with class names.
  const names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'];
  await expect(page.locator('#drawpad .pred-digit')).not.toHaveText('?');
  expect(names).toContain((await page.locator('#drawpad .pred-digit').textContent())!.trim());
  const pcts = (await page.locator('#drawpad .bar-pct').allTextContents()).map((t) => Number(t.replace('%', '')));
  expect(pcts).toEqual([...pcts].sort((a, b) => b - a));
  expect((await page.locator('#drawpad .bar-row').first().locator('.bar-name').textContent())!.trim()).toBe((await page.locator('#drawpad .pred-digit').textContent())!.trim());

  // Nudges: flip mirrors the input; brightness lifts it; both follow through to the page's input.
  await page.locator('#photo-flip').click();
  await expect(page.locator('#photo-flip')).toHaveAttribute('aria-pressed', 'true');
  await page.waitForFunction(() => (window as unknown as { raster: Raster }).raster.store.probe?.caption === 'Your photo · flipped');
  const flipped = await raster(page, (r) => Array.from(r.store.probe!.x));
  expect(flipped[5 * 32 + 0]).toBeCloseTo(x[5 * 32 + 31], 5);
  await page.locator('#photo-brightness').fill('0.3');
  await expect(page.locator('#drawpad output[for="photo-brightness"]')).toHaveText('+0.30');
  await page.waitForFunction(() => (window as unknown as { raster: Raster }).raster.store.probe?.caption === 'Your photo · flipped, brightness +0.30');
  const brighter = await raster(page, (r) => Array.from(r.store.probe!.x));
  const mean = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;
  expect(mean(brighter)).toBeGreaterThan(mean(flipped) + 0.1);
  await shot(page, '#draw', '14-datasets-photo.png');
  await page.locator('#photo-reset').click();
  await page.waitForFunction(() => (window as unknown as { raster: Raster }).raster.store.probe?.caption === 'Your photo');
  // Training on a subset resets the page's input to a test image, but the photo stays in 03.
  await page.locator('#datasets .dsp-train').getByRole('button', { name: '1,000', exact: true }).click();
  await page.waitForFunction(() => (window as unknown as { raster: Raster }).raster.store.data!.trainY.length === 1000);
  await expect(page.locator('#drawpad .dp-photo-caption')).toHaveText('Your photo');
  await expect(page.locator('#drawpad .pred-digit')).not.toHaveText('?');

  // A random test image keeps its label, and a test image picked in 07 shows up here.
  await page.locator('#photo-random').click();
  expect(await raster(page, (r) => r.store.probe!.key)).toMatch(/^test:\d+$/);
  await page.locator('#datapanel .sample-grid').first().locator('.thumb').nth(5).click();
  const key = await raster(page, (r) => r.store.probe!.key);
  await expect(page.locator('#drawpad .dp-photo-caption')).toHaveText((await raster(page, (r) => r.store.probe!.caption)));
  await page.locator('#photo-flip').click();
  await page.waitForFunction(() => (window as unknown as { raster: Raster }).raster.store.probe?.key === 'photo');
  expect(await raster(page, (r) => r.store.probe!.label)).not.toBeNull();
  expect(key).toMatch(/^test:\d+$/);
});

test('Fashion-MNIST: drawings fill the frame, photos become light-on-black items', async ({ page }) => {
  await open(page);
  await pick(page, 'Fashion-MNIST', 'fashion');
  await page.locator('#draw').scrollIntoViewIfNeeded();
  // Draw a wide bar: Fashion framing scales its longest edge to the full 28 px.
  const pad = (await page.locator('.pad').boundingBox())!;
  await page.mouse.move(pad.x + pad.width * 0.3, pad.y + pad.height * 0.5);
  await page.mouse.down();
  await page.mouse.move(pad.x + pad.width * 0.7, pad.y + pad.height * 0.5, { steps: 10 });
  await page.mouse.up();
  await page.waitForFunction(() => (window as unknown as { raster: Raster }).raster.store.probe?.key === 'draw');
  const span = await raster(page, (r) => {
    const x = r.store.probe!.x;
    let x0 = 28, x1 = -1;
    x.forEach((v, i) => {
      if (v > 0.3) {
        x0 = Math.min(x0, i % 28);
        x1 = Math.max(x1, i % 28);
      }
    });
    return x1 - x0 + 1;
  });
  expect(span).toBeGreaterThanOrEqual(27);
  await expect(page.locator('#drawpad .pred-digit')).not.toHaveText('?');
  // Label it by name and add it to the training set.
  await page.locator('#draw-class').selectOption({ label: 'Bag' });
  await page.locator('#drawpad .add-train').getByRole('button', { name: 'Add', exact: true }).click();
  await expect(page.locator('#drawpad .add-train .hint')).toContainText('Added as Bag');
  expect(await raster(page, (r) => r.store.custom.map((c) => c.y))).toEqual([8]);

  // Photo mode: a dark item on a white backdrop becomes a light item on black.
  await page.locator('#drawpad .dp-modes').getByRole('button', { name: 'Photo' }).click();
  await expect(page.locator('#drawpad .dp-photo')).toBeVisible();
  await expect(page.locator('.pad')).toBeHidden();
  const b64 = await page.evaluate(() => {
    const c = document.createElement('canvas');
    c.width = 300;
    c.height = 300;
    const ctx = c.getContext('2d')!;
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, 300, 300);
    ctx.fillStyle = '#202830';
    ctx.fillRect(110, 40, 80, 220); // a trouser-like bar
    return c.toDataURL('image/png').split(',')[1];
  });
  await page.setInputFiles('#photo-input', [{ name: 'item.png', mimeType: 'image/png', buffer: Buffer.from(b64, 'base64') }]);
  await page.waitForFunction(() => (window as unknown as { raster: Raster }).raster.store.probe?.key === 'photo');
  const im = await raster(page, (r) => {
    const x = r.store.probe!.x;
    return { corner: x[0], centre: x[14 * 28 + 14], top: x[0 * 28 + 14], bottom: x[27 * 28 + 14] };
  });
  expect(im.corner).toBe(0);
  expect(im.centre).toBeGreaterThan(0.8);
  expect(im.top).toBeGreaterThan(0.5); // the item fills the frame top to bottom
  expect(im.bottom).toBeGreaterThan(0.5);
});

test('keyboard: one tab stop for the index, arrows move, Enter picks', async ({ page }) => {
  await open(page);
  const tabbable = await page.locator('#datasets .dsp-item[tabindex="0"]').allTextContents();
  expect(tabbable.map((t) => t.trim())).toEqual(['MNIST digits']);
  await item(page, 'MNIST digits').focus();
  await page.keyboard.press('ArrowRight');
  await expect(item(page, 'Fashion-MNIST')).toBeFocused();
  await expect(page.locator('#datasets .dsp-about')).toContainText('Photos of clothing');
  await page.keyboard.press('End');
  await expect(item(page, 'Four blobs')).toBeFocused();
  await page.keyboard.press('ArrowLeft');
  await expect(item(page, 'XOR cube')).toBeFocused();
  await page.keyboard.press('Enter');
  await page.waitForFunction(() => (window as unknown as { raster: Raster }).raster.store.data?.info.id === 'xor3');
  await expect(item(page, 'XOR cube')).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator('#datasets .dsp-item[tabindex="0"]')).toHaveCount(1);

  // Controls keep keyboard focus while the data they change is rebuilt.
  const chip = page.locator('#datasets .ds-feat[data-feature="x1*x2"]');
  await chip.focus();
  await page.keyboard.press('Enter');
  await page.waitForFunction(() => (window as unknown as { raster: Raster }).raster.store.net.inputSize === 4);
  await expect(chip).toHaveAttribute('aria-pressed', 'true');
  await expect(chip).toBeFocused();
  const fifty = page.locator('#datasets .dsp-train').getByRole('button', { name: '50', exact: true });
  await fifty.focus();
  await page.keyboard.press('Space');
  await page.waitForFunction(() => (window as unknown as { raster: Raster }).raster.store.data!.trainY.length === 50);
  await expect(fifty).toHaveAttribute('aria-pressed', 'true');
  await expect(fifty).toBeFocused();
  const count = page.locator('#datasets .dsp-points .seg button', { hasText: '400' });
  await count.focus();
  await page.keyboard.press('Enter');
  await page.waitForFunction(() => (window as unknown as { raster: Raster }).raster.store.pointsConfig.count === 400);
  await expect(count).toBeFocused();
});

test('dark theme and phone width', async ({ page }) => {
  await page.emulateMedia({ colorScheme: 'dark' });
  await open(page);
  await pick(page, 'Spiral', 'spiral');
  await shot(page, '#datasets', '14-datasets-picker-dark.png');

  await page.setViewportSize({ width: 390, height: 844 });
  await page.emulateMedia({ colorScheme: 'light' });
  await noOverflow(page);
  // The index scrolls sideways inside the strip; the chosen dataset is scrolled into view.
  const idx = await page.locator('#datasets .dsp-index').evaluate((el) => ({ sw: el.scrollWidth, cw: el.clientWidth, left: el.scrollLeft }));
  expect(idx.sw).toBeGreaterThan(idx.cw);
  expect(idx.left).toBeGreaterThan(0);
  const box = (await item(page, 'Spiral').boundingBox())!;
  expect(box.x).toBeGreaterThanOrEqual(0);
  expect(box.x + box.width).toBeLessThanOrEqual(390);
  await page.locator('#datasets').scrollIntoViewIfNeeded();
  await shot(page, '#datasets', '14-datasets-picker-phone.png');
  for (const [name, id] of [
    ['Four blobs', 'blobs3'],
    ['CIFAR-10', 'cifar10'],
  ] as const) {
    await pick(page, name, id);
    await noOverflow(page);
  }
  await page.locator('#draw').scrollIntoViewIfNeeded();
  await noOverflow(page);
});
