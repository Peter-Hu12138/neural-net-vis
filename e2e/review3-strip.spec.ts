import { expect, test, type Page } from '@playwright/test';

/**
 * Review 3 (round B) regressions in the dataset strip, the training charts, the drawing pad and
 * the control bar: previews true to each dataset before it loads (design-2), one spelling of
 * "2-D/3-D" (design-8), one loss format (design-9) and the copy fixes of design-14.
 */

type Raster = {
  store: {
    data: { info: { id: string } } | null;
    points: { epoch: number; step: number; loss: number; acc: number }[];
    evals: { epoch: number; step: number; loss: number; acc: number; confusion: number[] }[];
    emit(ev: string): void;
  };
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
  await expect(page.locator('#fact-data')).toContainText('20,000 train', { timeout: 30_000 });
  await expect(page.locator('#play')).toBeEnabled();
}

const item = (page: Page, name: string) => page.locator('#datasets').getByRole('button', { name, exact: true });

/** Picks a dataset in the strip and waits until its data is in. */
async function pick(page: Page, name: string, id: string) {
  await item(page, name).click();
  await page.waitForFunction((want) => (window as unknown as { raster: Raster }).raster.store.data?.info.id === want, id, { timeout: 60_000 });
}

/** The preview tile's pixels (RGBA) of one dataset. */
const previewPixels = (page: Page, id: string) =>
  page.locator(`#datasets .dsp-item[data-id="${id}"] canvas`).evaluate((c: HTMLCanvasElement) => Array.from(c.getContext('2d')!.getImageData(0, 0, c.width, c.height).data));

test('design-2: image previews show real samples before the dataset loads, never digits', async ({ page }) => {
  // Every text drawn into a preview tile is recorded.
  await page.addInitScript(() => {
    const w = window as unknown as { previewTexts: string[] };
    w.previewTexts = [];
    const orig = CanvasRenderingContext2D.prototype.fillText;
    CanvasRenderingContext2D.prototype.fillText = function (this: CanvasRenderingContext2D, t: string, x: number, y: number, mw?: number) {
      if (this.canvas instanceof HTMLCanvasElement && this.canvas.classList.contains('dsp-preview')) w.previewTexts.push(t);
      return mw === undefined ? orig.call(this, t, x, y) : orig.call(this, t, x, y, mw);
    };
  });
  await open(page);
  // No digits stand in for clothing or photos.
  expect(await page.evaluate(() => (window as unknown as { previewTexts: string[] }).previewTexts)).toEqual([]);
  // Fashion-MNIST and CIFAR-10 have not been loaded, yet their tiles show their own images.
  for (const id of ['mnist', 'fashion', 'cifar10']) await expect(page.locator(`#datasets .dsp-item[data-id="${id}"] canvas`)).toHaveAttribute('data-preview', 'samples');

  const before = { fashion: await previewPixels(page, 'fashion'), cifar10: await previewPixels(page, 'cifar10') };
  // The tiles are pictures, not a flat fill: many distinct colours.
  for (const px of Object.values(before)) {
    const colours = new Set<number>();
    for (let i = 0; i < px.length; i += 4) colours.add((px[i] << 16) | (px[i + 1] << 8) | px[i + 2]);
    expect(colours.size).toBeGreaterThan(40);
  }
  // Once loaded, the tile is redrawn from the loaded data: the same four test images (CIFAR-10's
  // shipped preview is a JPEG, so within a few levels).
  for (const [name, id, tol] of [
    ['Fashion-MNIST', 'fashion', 0],
    ['CIFAR-10', 'cifar10', 4],
  ] as const) {
    await pick(page, name, id);
    await expect.poll(async () => {
      const after = await previewPixels(page, id);
      let sum = 0;
      for (let i = 0; i < after.length; i++) sum += Math.abs(after[i] - before[id][i]);
      return sum / after.length;
    }).toBeLessThanOrEqual(tol);
  }
  expect(await page.evaluate(() => (window as unknown as { previewTexts: string[] }).previewTexts)).toEqual([]);
});

test('design-2: the image previews follow the theme', async ({ page }) => {
  await page.emulateMedia({ colorScheme: 'dark' });
  await open(page);
  await expect(page.locator('#datasets .dsp-item[data-id="fashion"] canvas')).toHaveAttribute('data-preview', 'samples');
  const dark = await previewPixels(page, 'fashion');
  await page.emulateMedia({ colorScheme: 'light' });
  await expect.poll(async () => {
    const light = await previewPixels(page, 'fashion');
    return light.some((v, i) => v !== dark[i]);
  }).toBe(true);
});

test('design-8: one spelling of 2-D and 3-D in the strip', async ({ page }) => {
  await open(page);
  const labels = await page.locator('#datasets .dsp-group-label').allTextContents();
  expect(labels).toEqual(['Images', 'Points in 2-D', 'Points in 3-D']);
  await item(page, 'Circle').hover();
  await expect(page.locator('#datasets .dsp-about')).toContainText('Circle · 2-D points · 2 classes');
  expect(await page.locator('#datasets').textContent()).not.toMatch(/\b[23]D\b/);
});

test('design-9: section 05 shows losses to three significant digits', async ({ page }) => {
  await open(page);
  // The untrained network's first evaluation lands first, so it cannot overwrite the values below.
  await page.waitForFunction(() => (window as unknown as { raster: Raster }).raster.store.evals.length > 0);
  const set = async (train: number, test: number) => {
    await page.evaluate(
      ([train, test]) => {
        const s = (window as unknown as { raster: Raster }).raster.store;
        s.points = [{ epoch: 1, step: 100, loss: train, acc: 1 }];
        s.evals = [{ epoch: 1, step: 100, loss: test, acc: 1, confusion: Array(100).fill(0) }];
        s.emit('metrics');
      },
      [train, test],
    );
  };
  const kpis = page.locator('#curves .kpi b');
  await set(0.000912, 0.003124);
  await expect(kpis).toHaveText(['0.000912', '100.0%', '0.00312', '100.0%']);
  await expect(page.locator('#curves canvas').first()).toHaveAttribute('aria-label', /latest test loss 0\.00312$/);
  await set(2.302585, 0.0912);
  await expect(kpis).toHaveText(['2.30', '100.0%', '0.0912', '100.0%']);
  await set(0.234, 0.00004);
  await expect(kpis).toHaveText(['0.234', '100.0%', '<0.0001', '100.0%']);
});

test('design-14: the photo panel and the control bar use plain, British copy', async ({ page }) => {
  await open(page);
  await expect(page.locator('#bar').getByRole('combobox', { name: 'Optimiser', exact: true })).toBeVisible();
  expect(await page.locator('#bar').textContent()).not.toMatch(/optimiz/i);
  await pick(page, 'CIFAR-10', 'cifar10');
  await page.locator('#draw').scrollIntoViewIfNeeded();
  await expect(page.locator('#photo-reset')).toHaveText('Undo changes');
  await expect(page.locator('#photo-reset')).toHaveAttribute('title', 'Undo flip, brightness and contrast');
  expect(await page.locator('#drawpad').textContent()).not.toContain('nudge');
});
