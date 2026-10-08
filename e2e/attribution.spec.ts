import { expect, test, type Page } from '@playwright/test';

/**
 * Section 10 Attribution: four maps of why the network predicts what it does, for the current
 * input and a chosen digit. Screenshots land in docs/screenshots/12-attribution-*.png.
 */

const SHOTS = 'docs/screenshots';

type Raster = {
  store: {
    status: { step: number; running: boolean } | null;
    evals: { acc: number }[];
    probe: { key: string; caption: string; x: Float32Array; label: number | null } | null;
    weightsStep: number;
    emit(ev: string): void;
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
  await page.locator('#attribution').screenshot({ path: `${SHOTS}/${name}` });
  await style.evaluate((el) => (el as Element).remove());
}

const root = (page: Page) => page.locator('#attr-root');

/** Waits for a finished result whose key (probe key | target) matches. */
async function resultFor(page: Page, key: RegExp) {
  await expect(root(page)).toHaveAttribute('data-result-key', key, { timeout: 30_000 });
  await expect(root(page)).toHaveAttribute('data-state', 'done');
}

/** Distinct colours in a canvas: a blank map has one, a real map many. */
const colours = (page: Page, id: string) =>
  page.locator(`#${id}`).evaluate((c: HTMLCanvasElement) => {
    const d = c.getContext('2d')!.getImageData(0, 0, c.width, c.height).data;
    const set = new Set<number>();
    for (let i = 0; i < d.length; i += 4) set.add((d[i] << 16) | (d[i + 1] << 8) | d[i + 2]);
    return set.size;
  });

/** Parses "Σ IG = 4.21; z(x) − z(blank) = 4.30 (2.1% apart)". */
async function completeness(page: Page) {
  const text = (await page.locator('#attr-check').textContent())!;
  const m = text.match(/Σ IG = (−?[\d.]+); z\(x\) − z\(blank\) = (−?[\d.]+) \((.+)\)/);
  expect(m, text).not.toBeNull();
  const num = (s: string) => Number(s.replace('−', '-'));
  return { sum: num(m![1]), expected: num(m![2]), gap: m![3] };
}

test('explains the prediction four ways, pins a chosen digit and follows the input', async ({ page }) => {
  await open(page);
  await trainUntil(page, '(window.raster.store.evals.length >= 2)');
  await page.locator('#attribution').scrollIntoViewIfNeeded();

  // Test digit #0 is a 7: the default target is the predicted digit.
  await resultFor(page, /^test:0\|\d$/);
  const pred = Number(await root(page).getAttribute('data-pred'));
  await expect(root(page)).toHaveAttribute('data-target', String(pred));
  await expect(page.locator('#attr-pred')).toHaveText(new RegExp(`^Predicted ${pred} at \\d+\\.\\d%; explaining digit ${pred}$`));
  await expect(page.locator(`#attr-target-${pred}`)).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator(`#attr-target-${pred}`)).toHaveAttribute('aria-label', /predicted/);
  await expect(page.locator('#attr-target-7')).toHaveAttribute('aria-label', /true label/);
  await expect(page.locator('#attr-root .attr-tag.is-pred')).toHaveCount(1);
  await expect(page.locator('#attr-root .attr-tag.is-true')).toHaveCount(1);
  await expect(page.locator('#attr-root .attr-input-caption')).toHaveText('Input: Test digit #0 · label 7');
  await expect(page.locator('#attr-root .synced-status')).toContainText('Based on the weights at step');

  // Five panels, all drawn; signed maps carry ± scales with the real maximum.
  for (const key of ['input', 'saliency', 'gradInput', 'integrated', 'occlusion']) {
    expect(await colours(page, `attr-map-${key}`), key).toBeGreaterThan(8);
  }
  for (const key of ['gradInput', 'integrated', 'occlusion']) {
    await expect(page.locator(`[data-panel="${key}"] .attr-scale`)).toHaveText(/^−[\d.e−]+\+[\d.e−]+$/);
  }
  await expect(page.locator('[data-panel="input"] .attr-scale')).toHaveText('01');
  await expect(page.locator('[data-panel="occlusion"] .hint')).toContainText(`Erases a 6×6 patch at a time: red where that lowers the probability of ${pred}`);

  // Completeness: the integrated gradients add up to z(x) − z(blank), within 5%.
  const c = await completeness(page);
  expect(Math.abs(c.expected)).toBeGreaterThan(0.5);
  expect(Math.abs(c.sum - c.expected) / Math.abs(c.expected)).toBeLessThan(0.05);
  expect(c.gap).toMatch(/apart|match/);

  // Hover: row, column, pixel value and the attribution of that pixel.
  const ig = (await page.locator('#attr-map-integrated').boundingBox())!;
  const cell = ig.width / 28;
  await page.mouse.move(ig.x + cell * 14.5, ig.y + cell * 12.5);
  await expect(page.locator('#tip')).toBeVisible();
  await expect(page.locator('#tip')).toContainText('Row 12, column 14');
  await expect(page.locator('#tip')).toContainText(/Pixel value \d\.\d\d/);
  await expect(page.locator('#tip')).toContainText(/Integrated gradient [+−]?[\d.]/);
  const occ = (await page.locator('#attr-map-occlusion').boundingBox())!;
  await page.mouse.move(occ.x + cell * 3.5, occ.y + cell * 20.5);
  await expect(page.locator('#tip')).toContainText('Row 20, column 3');
  await expect(page.locator('#tip')).toContainText(`Mean drop in p(${pred})`);
  await page.mouse.move(ig.x + cell * 14.5, ig.y + cell * 12.5);
  await shot(page, '12-attribution-light.png');
  await page.mouse.move(0, 0);
  await expect(page.locator('#tip')).toBeHidden();

  // Pick another digit: the maps explain it instead, and the line names its probability.
  const other = (pred + 3) % 10;
  await page.click(`#attr-target-${other}`);
  await resultFor(page, new RegExp(`^test:0\\|${other}$`));
  await expect(page.locator('#attr-pred')).toHaveText(new RegExp(`^Predicted ${pred} at [\\d.]+%; explaining digit ${other} \\([\\d.]+%\\)$`));
  await expect(page.locator(`#attr-target-${other}`)).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator(`#attr-target-${pred}`)).toHaveAttribute('aria-pressed', 'false');
  await expect(page.locator('[data-panel="integrated"] .hint')).toContainText(`change in the score for ${other}`);

  // The pick survives new weights (one training step while paused recomputes the maps)...
  const step = await raster(page, (r) => r.store.weightsStep);
  await page.click('#step');
  await page.waitForFunction((s) => (window as unknown as { raster: Raster }).raster.store.weightsStep > s, step);
  await expect(page.locator('#attr-root .synced-status')).toContainText(`Based on the weights at step ${(step + 1).toLocaleString('en-US')}.`);
  await resultFor(page, new RegExp(`^test:0\\|${other}$`));

  // ...but a new input resets it to the prediction. Swap the input while the section is on screen.
  await raster(page, (r) => {
    const x = new Float32Array(784);
    for (let y = 6; y < 22; y++) for (let k = 13; k < 16; k++) x[y * 28 + k] = 1; // a "1"
    (r.store as unknown as { probe: unknown }).probe = { x, label: 1, caption: 'Hand-made 1', key: 'test:hand-1' };
    r.store.emit('probe');
  });
  await resultFor(page, /^test:hand-1\|\d$/);
  const pred2 = await root(page).getAttribute('data-pred');
  await expect(root(page)).toHaveAttribute('data-target', pred2!);
  await expect(page.locator('#attr-root .attr-input-caption')).toHaveText('Input: Hand-made 1');
  // gradient × input is zero off the ink: only the stroke is coloured
  const offInk = await page.locator('#attr-map-gradInput').evaluate((c: HTMLCanvasElement) => {
    const ctx = c.getContext('2d')!;
    const at = (x: number, y: number) => Array.from(ctx.getImageData(Math.round((x * c.width) / 28), Math.round((y * c.height) / 28), 1, 1).data.slice(0, 3));
    return { corner: at(3.5, 3.5), edge: at(25.5, 14.5) };
  });
  expect(offInk.corner).toEqual(offInk.edge);

  // Dark theme: every canvas redraws from the dark tokens.
  await page.emulateMedia({ colorScheme: 'dark' });
  await page.waitForFunction(() => {
    const c = document.getElementById('attr-map-integrated') as HTMLCanvasElement;
    const d = c.getContext('2d')!.getImageData(Math.round(c.width * 0.1), Math.round(c.height * 0.1), 1, 1).data;
    return d[0] + d[1] + d[2] < 150;
  });
  await resultFor(page, /^test:hand-1\|\d$/);
  // Back to the test digit for the dark screenshot.
  await raster(page, (r) => {
    const all = (r as unknown as { store: { data: { testX: Uint8Array; testY: Uint8Array } } }).store.data;
    const x = Float32Array.from(all.testX.subarray(0, 784), (v) => v / 255);
    (r.store as unknown as { probe: unknown }).probe = { x, label: all.testY[0], caption: `Test digit #0 · label ${all.testY[0]}`, key: 'test:0' };
    r.store.emit('probe');
  });
  await resultFor(page, /^test:0\|\d$/);
  await shot(page, '12-attribution-dark.png');
  await page.emulateMedia({ colorScheme: 'light' });
});

test('follows the drawing pad, and says so when the input is blank', async ({ page }) => {
  await open(page);
  await trainUntil(page, '(window.raster.store.evals.length >= 2)');

  // Draw a 7 while section 10 is off screen; it catches up when scrolled into view.
  await page.locator('.pad').scrollIntoViewIfNeeded();
  const pad = (await page.locator('.pad').boundingBox())!;
  const at = (fx: number, fy: number) => [pad.x + pad.width * fx, pad.y + pad.height * fy] as const;
  await page.mouse.move(...at(0.28, 0.24));
  await page.mouse.down();
  for (let t = 0; t <= 1.0001; t += 0.05) await page.mouse.move(...at(0.28 + 0.44 * t, 0.24));
  for (let t = 0; t <= 1.0001; t += 0.05) await page.mouse.move(...at(0.72 - 0.26 * t, 0.24 + 0.56 * t));
  await page.mouse.up();
  await expect.poll(() => raster(page, (r) => r.store.probe?.key)).toBe('draw');

  await page.locator('#attribution').scrollIntoViewIfNeeded();
  await resultFor(page, /^draw\|\d$/);
  await expect(page.locator('#attr-root .attr-input-caption')).toHaveText('Input: Your drawing');
  await expect(page.locator('#attr-notice')).toBeHidden();
  // A drawing has no true label: only the prediction is marked.
  await expect(page.locator('#attr-root .attr-tag.is-true')).toHaveCount(0);
  await expect(page.locator('#attr-root .attr-tag.is-pred')).toHaveCount(1);
  const c = await completeness(page);
  expect(Math.abs(c.sum - c.expected)).toBeLessThan(0.05 * Math.abs(c.expected) + 0.01);

  // Clearing the pad empties the input: every method that multiplies by the ink, or erases it, gives 0.
  await page.locator('#drawpad').getByRole('button', { name: 'Clear' }).click();
  await page.locator('#attribution').scrollIntoViewIfNeeded();
  await expect(page.locator('#attr-notice')).toBeVisible({ timeout: 30_000 });
  await expect(page.locator('#attr-notice')).toContainText('The input is blank');
  await expect(page.locator('#attr-check')).toHaveText('Σ IG = 0.00; z(x) − z(blank) = 0.00 (a match)');
  for (const key of ['gradInput', 'integrated', 'occlusion']) {
    await expect(page.locator(`[data-panel="${key}"] .attr-scale`)).toHaveText('00');
  }
});

test('390 px phone screen: panels wrap two per row, no sideways scrolling', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await open(page);
  await trainUntil(page, '(window.raster.store.evals.length >= 2)');
  await page.locator('#attribution').scrollIntoViewIfNeeded();
  await resultFor(page, /^test:0\|\d$/);

  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow, 'no horizontal page scroll at 390 px').toBeLessThanOrEqual(0);
  const boxes = await page.locator('#attr-root .attr-panel').evaluateAll((els) => els.map((e) => e.getBoundingClientRect().toJSON() as DOMRect));
  expect(boxes).toHaveLength(5);
  expect(Math.abs(boxes[0].top - boxes[1].top)).toBeLessThan(1);
  expect(boxes[2].top).toBeGreaterThan(boxes[0].bottom - 1);
  for (const b of boxes) expect(b.right).toBeLessThanOrEqual(390 - 16 + 0.5);
  // Maps stay a readable size: at least 4 screen pixels per image pixel.
  const w = await page.locator('#attr-map-integrated').evaluate((c) => c.getBoundingClientRect().width);
  expect(w).toBeGreaterThanOrEqual(112);
  await expect(page.locator('#attr-pred')).toBeVisible();
  await shot(page, '12-attribution-phone.png');

  // Tablet and laptop widths: three, then five panels per row, and nothing in the section sticks out.
  // (Measured inside section 10 only: after a resize, 05 Training's loss chart can hold its old width.)
  for (const [width, perRow] of [[700, 3], [1024, 5]] as const) {
    await page.setViewportSize({ width, height: 900 });
    await page.waitForTimeout(300);
    const tops = await page.locator('#attr-root .attr-panel').evaluateAll((els) => els.map((e) => Math.round(e.getBoundingClientRect().top)));
    expect(tops.filter((t) => t === tops[0]), `${width} px`).toHaveLength(perRow);
    const right = await page.locator('#attribution').evaluate((sec) => Math.max(...Array.from(sec.querySelectorAll('*'), (e) => e.getBoundingClientRect().right)));
    expect(right, `section 10 fits at ${width} px`).toBeLessThanOrEqual(width - 16 + 0.5);
  }
});
