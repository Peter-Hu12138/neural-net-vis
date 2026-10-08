import { expect, test, type Page } from '@playwright/test';

/**
 * Section 11 Embedding: the PCA map (automatic), t-SNE (on request, animated), hover, click,
 * digit highlight, mistakes, dark mode and the phone layout. Screenshots land in
 * docs/screenshots/13-embedding-*.png.
 */

const SHOTS = 'docs/screenshots';

type Raster = {
  store: {
    status: { step: number; running: boolean } | null;
    evals: { acc: number }[];
    probe: { key: string; caption: string } | null;
    weightsStep: number;
  };
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

/** Trains until `pred` holds, then pauses. */
async function trainUntil(page: Page, pred: string, timeout = 90_000) {
  await page.click('#play');
  await page.waitForFunction(pred, null, { timeout });
  await page.click('#play');
  await page.waitForFunction(() => !(window as unknown as { raster: Raster }).raster.store.status?.running);
}

/** Screenshot of the whole section; the sticky control bar would otherwise cover its top. */
async function shot(page: Page, name: string) {
  const style = await page.addStyleTag({ content: '#bar { position: static !important; }' });
  await page.locator('#embedding').screenshot({ path: `${SHOTS}/${name}` });
  await style.evaluate((el) => (el as Element).remove());
}

const status = (page: Page) => page.locator('#embed-root .synced-status');
const runLine = (page: Page) => page.locator('#embed-root .embed-run');
const canvas = (page: Page) => page.locator('#embed-canvas');

/** Waits until a run has finished: status names the step and the progress line is idle. */
async function settled(page: Page) {
  await expect(status(page)).toContainText('Based on the weights at step', { timeout: 60_000 });
  await expect(runLine(page)).toHaveClass(/is-idle/, { timeout: 60_000 });
}

/**
 * CSS-pixel positions (relative to the canvas) of drawn numerals: pixels whose colour is far from
 * both the surface and the ink, sampled on a grid and kept at least 30 px apart.
 */
async function numeralSpots(page: Page): Promise<{ x: number; y: number }[]> {
  return canvas(page).evaluate((c: HTMLCanvasElement) => {
    const ctx = c.getContext('2d')!;
    const { width, height } = c;
    const dpr = width / c.clientWidth;
    const d = ctx.getImageData(0, 0, width, height).data;
    const spots: { x: number; y: number }[] = [];
    for (let y = 60; y < height - 60; y += 3) {
      for (let x = 60; x < width - 60; x += 3) {
        const i = 4 * (y * width + x);
        const r = d[i];
        const g = d[i + 1];
        const b = d[i + 2];
        const sat = Math.max(r, g, b) - Math.min(r, g, b);
        if (d[i + 3] < 250 || sat < 90) continue;
        const p = { x: x / dpr, y: y / dpr };
        if (spots.every((s) => (s.x - p.x) ** 2 + (s.y - p.y) ** 2 > 900)) spots.push(p);
      }
    }
    return spots;
  });
}

/** Moves the mouse over numerals until the tooltip names a test digit; returns that tooltip text. */
async function hoverDigit(page: Page): Promise<string> {
  const box = (await canvas(page).boundingBox())!;
  for (const s of (await numeralSpots(page)).slice(0, 40)) {
    await page.mouse.move(box.x + s.x, box.y + s.y);
    const tip = page.locator('#tip');
    if ((await tip.isVisible()) && /^Test digit #\d+/.test((await tip.textContent()) ?? '')) return (await tip.textContent())!;
  }
  throw new Error('no numeral could be hovered');
}

test('PCA map: automatic, numerals, hover preview, click to probe, highlight and mistakes', async ({ page }) => {
  await open(page);
  await trainUntil(page, '(window.raster.store.evals.length >= 3)');
  await page.locator('#embedding').scrollIntoViewIfNeeded();

  // Small CNN: the default layer is the last hidden one (Dense 3, 32 values).
  await expect(page.locator('#embed-layer')).toHaveValue('2');
  await expect(page.locator('#embed-method-pca')).toHaveAttribute('aria-pressed', 'true');
  await settled(page);
  await expect(canvas(page)).toHaveAttribute('aria-label', /PCA map of 1,000 test digits at Dense 3/);
  await expect(page.locator('#embed-stats')).toContainText('Digits 1,000 · 100 of each');
  await expect(page.locator('#embed-stats')).toContainText('Values per digit 32 at Dense 3');
  await expect(page.locator('#embed-stats')).toContainText(/Variance shown \d+\.\d% \(PC1 \d+\.\d%, PC2 \d+\.\d%\)/);
  await expect(page.locator('#embed-root .embed-hints')).toContainText('PCA finds the two directions');
  const size = await canvas(page).evaluate((c: HTMLCanvasElement) => [c.clientWidth, c.clientHeight]);
  expect(size).toEqual([640, 640]);
  expect((await numeralSpots(page)).length).toBeGreaterThan(60);

  // Hover: tooltip with label and prediction, plus coordinates; the preview shows the digit large.
  const tip = await hoverDigit(page);
  expect(tip).toMatch(/^Test digit #\d+ · label \d · predicted \d\nPC1 −?-?\d/);
  const index = Number(tip.match(/#(\d+)/)![1]);
  await expect(page.locator('#embed-root .embed-aside .sub').first()).toHaveText('Hovered digit');
  await expect(page.locator('#embed-preview')).toContainText(`Test digit #${index}`);
  await expect(page.locator('#embed-preview')).toContainText('Click to use it as the input');

  // Click: the digit becomes the network's input and is marked with the input cross.
  await page.mouse.down();
  await page.mouse.up();
  await expect.poll(() => raster(page, (r) => r.store.probe?.key)).toBe(`test:${index}`);
  await page.mouse.move(5, 5);
  await expect(page.locator('#tip')).toBeHidden();
  await expect(page.locator('#embed-root .embed-aside .sub').first()).toHaveText('Current input');
  await expect(page.locator('#embed-preview')).toContainText(`Test digit #${index}`);
  await expect(page.locator('#embed-preview')).toContainText(/PC1 −?\d.* · PC2 /);
  await expect(page.locator('#embed-preview')).toContainText('Marked on the map with a red cross');
  await expect(page.locator('#embed-root .embed-keys')).toContainText('Current input');

  // Mark mistakes: a ring per misclassified digit, counted in the key.
  await page.locator('#embed-mistakes').check();
  await expect(page.locator('#embed-root .embed-keys')).toContainText(/Misclassified · \d+ of 1,000/);
  await expect(page.locator('#embed-stats')).toContainText(/Misclassified \d+ \(\d+\.\d%\)/);
  await shot(page, '13-embedding-pca.png');

  // Highlight one digit; the others fade. Hovering then only finds that digit.
  await page.click('#embed-digit-7');
  await expect(page.locator('#embed-digit-7')).toHaveAttribute('aria-pressed', 'true');
  const tip7 = await hoverDigit(page);
  expect(tip7).toContain('label 7');
  await page.mouse.move(5, 5);
  await shot(page, '13-embedding-highlight.png');
  await page.click('#embed-digit-7');
  await expect(page.locator('#embed-digit-7')).toHaveAttribute('aria-pressed', 'false');

  // Keyboard: arrows walk the points, Enter picks one.
  await canvas(page).focus();
  await page.keyboard.press('ArrowRight');
  await expect(page.locator('#tip')).toContainText(/Test digit #\d+/);
  await page.keyboard.press('ArrowLeft');
  const kTip = (await page.locator('#tip').textContent())!;
  const kIndex = Number(kTip.match(/#(\d+)/)![1]);
  await page.keyboard.press('Enter');
  await expect.poll(() => raster(page, (r) => r.store.probe?.key)).toBe(`test:${kIndex}`);
  await page.keyboard.press('Escape');

  // Another layer recomputes on its own: raw pixels.
  await page.selectOption('#embed-layer', '-1');
  await settled(page);
  await expect(page.locator('#embed-stats')).toContainText('Values per digit 784 at the input pixels');
  const pc1 = await page.locator('#embed-stats').textContent();
  const share = Number(pc1!.match(/PC1 (\d+\.\d)%/)![1]);
  expect(share).toBeGreaterThan(6);
  expect(share).toBeLessThan(14);
  await expect(canvas(page)).toHaveAttribute('aria-label', /at the input pixels/);
});

test('t-SNE runs only on request, animates, and is kept when switching back', async ({ page }) => {
  await open(page);
  await trainUntil(page, '(window.raster.store.evals.length >= 3)');
  await page.locator('#embedding').scrollIntoViewIfNeeded();
  await settled(page);

  const t0 = Date.now();
  await page.click('#embed-method-tsne');
  await expect(runLine(page)).not.toHaveClass(/is-idle/);
  // Frames arrive while it optimises: the iteration count moves on.
  await expect(page.locator('#embed-run-text')).toHaveText(/^Iteration \d+ \/ 500 · KL \d+\.\d\d$/, { timeout: 30_000 });
  const first = Number((await page.locator('#embed-run-text').textContent())!.match(/Iteration (\d+)/)![1]);
  await shot(page, '13-embedding-tsne-running.png');
  await expect
    .poll(async () => Number(((await page.locator('#embed-run-text').textContent()) ?? '').match(/Iteration (\d+)/)?.[1] ?? 500), { timeout: 30_000 })
    .toBeGreaterThan(first);
  await expect(page.locator('#embed-root .progress')).toHaveAttribute('aria-valuenow', /\d+/);
  await settled(page);
  console.log(`t-SNE in the browser (n = 1,000, 500 iterations, Dense 3): ${((Date.now() - t0) / 1000).toFixed(1)} s`);
  await expect(canvas(page)).toHaveAttribute('aria-label', /t-SNE map of 1,000 test digits at Dense 3/);
  await expect(page.locator('#embed-stats')).toContainText(/KL divergence \d+\.\d\d after 500 iterations/);
  await expect(page.locator('#embed-stats')).toContainText('Perplexity 30');
  await expect(page.locator('#embed-root .embed-hints')).toContainText('It preserves neighbours, not distances');
  const tip = await hoverDigit(page);
  expect(tip).toMatch(/^Test digit #\d+ · label \d · predicted \d$/);
  await page.mouse.move(5, 5);
  await page.locator('#embed-mistakes').check();
  await shot(page, '13-embedding-tsne.png');

  // Both results are kept for these weights: switching back and forth recomputes nothing.
  await page.click('#embed-method-pca');
  await expect(runLine(page)).toHaveClass(/is-idle/);
  await expect(page.locator('#embed-stats')).toContainText('Variance shown');
  await page.click('#embed-method-tsne');
  await expect(runLine(page)).toHaveClass(/is-idle/);
  await expect(page.locator('#embed-stats')).toContainText('KL divergence');

  // New weights do not restart t-SNE on their own; the status says so and Recompute runs it.
  const step = await raster(page, (r) => r.store.weightsStep);
  await page.click('#step');
  await page.waitForFunction((s) => (window as unknown as { raster: Raster }).raster.store.weightsStep > s, step);
  await expect(status(page)).toContainText('the network is now at step');
  await page.waitForTimeout(1500);
  await expect(runLine(page)).toHaveClass(/is-idle/);
  await status(page).getByRole('button', { name: 'Recompute' }).click();
  await expect(runLine(page)).not.toHaveClass(/is-idle/);
  await settled(page);

  // A new architecture clears t-SNE until it is asked for again.
  await page.getByRole('button', { name: 'Softmax', exact: true }).click();
  await expect(page.locator('#embed-layer')).toHaveValue('-1');
  await expect(canvas(page)).toHaveAttribute('aria-label', 't-SNE runs only when you ask. Press Recompute.');
  await page.click('#embed-method-pca');
  await settled(page);
  await expect(page.locator('#embed-stats')).toContainText('Values per digit 784 at the input pixels');
});

test('dark mode follows the theme', async ({ page }) => {
  await page.emulateMedia({ colorScheme: 'dark' });
  await open(page);
  await trainUntil(page, '(window.raster.store.evals.length >= 3)');
  await page.locator('#embedding').scrollIntoViewIfNeeded();
  await settled(page);
  // The plot area is painted with the dark surface, and the numerals are still there.
  const pixel = () => canvas(page).evaluate((c: HTMLCanvasElement) => Array.from(c.getContext('2d')!.getImageData(Math.round(c.width / 2), 20, 1, 1).data.slice(0, 3)));
  expect(Math.max(...(await pixel()))).toBeLessThan(60);
  expect((await numeralSpots(page)).length).toBeGreaterThan(60);
  await page.locator('#embed-mistakes').check();
  await page.click('#embed-digit-3');
  await shot(page, '13-embedding-dark.png');
  await page.click('#embed-digit-3');
  // Switching the theme redraws the canvas.
  await page.emulateMedia({ colorScheme: 'light' });
  await expect.poll(async () => Math.min(...(await pixel()))).toBeGreaterThan(200);
});

test('phone width: no horizontal overflow, panel below the map, tap to pick', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await open(page);
  await trainUntil(page, '(window.raster.store.evals.length >= 3)');
  await page.locator('#embedding').scrollIntoViewIfNeeded();
  await settled(page);
  const width = await canvas(page).evaluate((c: HTMLCanvasElement) => c.clientWidth);
  expect(width).toBeLessThanOrEqual(358);
  expect(width).toBeGreaterThan(300);
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow).toBe(0);
  const plot = (await canvas(page).boundingBox())!;
  const aside = (await page.locator('#embed-root .embed-aside').boundingBox())!;
  expect(aside.y).toBeGreaterThan(plot.y + plot.height - 1);
  expect(aside.x + aside.width).toBeLessThanOrEqual(390);
  // A tap picks a digit as the input.
  const spots = await numeralSpots(page);
  expect(spots.length).toBeGreaterThan(30);
  await page.mouse.click(plot.x + spots[0].x, plot.y + spots[0].y);
  await expect.poll(() => raster(page, (r) => r.store.probe?.key ?? '')).toMatch(/^test:\d+$/);
  await page.mouse.move(1, 1);
  await shot(page, '13-embedding-phone.png');
});
