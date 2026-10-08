import { expect, test, type Locator, type Page } from '@playwright/test';

/**
 * Section 08 Distributions and the Q–Q view of the weight inspector, in real Chromium against the
 * production build. Screenshots land in docs/screenshots/10-distributions-*.png.
 */

const SHOTS = 'docs/screenshots';

type Layer = { block: number; z: Float32Array; a: Float32Array; gW: Float32Array; activeFraction: Float32Array; dead: number | null };
type Raster = {
  store: {
    status: { step: number; running: boolean } | null;
    weightsStep: number;
    net: { blocks: { W: Float32Array }[] };
  };
  analysis: { run(channel: string, kind: string, params: unknown): Promise<{ samples: number; layers: Layer[] }> };
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
async function trainUntil(page: Page, pred: string, timeout = 60_000) {
  await page.click('#play');
  await page.waitForFunction(pred, null, { timeout });
  await page.click('#play');
  await page.waitForFunction(() => !(window as unknown as { raster: Raster }).raster.store.status?.running);
}

/** Number of painted pixels in a canvas: proves something was drawn. */
const painted = (c: Locator) =>
  c.evaluate((el: HTMLCanvasElement) => {
    const d = el.getContext('2d')!.getImageData(0, 0, el.width, el.height).data;
    let n = 0;
    for (let i = 3; i < d.length; i += 4) if (d[i] > 0) n++;
    return n;
  });

/** Hovers points of a canvas (fractions of its box) until the shared tooltip appears; returns its text. */
async function hoverTip(page: Page, c: Locator, points: [number, number][]) {
  await c.scrollIntoViewIfNeeded();
  const box = (await c.boundingBox())!;
  for (const [fx, fy] of points) {
    await page.mouse.move(box.x + box.width * fx, box.y + box.height * fy);
    if (await page.locator('#tip').isVisible()) return (await page.locator('#tip').textContent()) ?? '';
  }
  return '';
}

const DIAGONAL: [number, number][] = [[0.55, 0.5], [0.5, 0.45], [0.6, 0.4], [0.45, 0.55], [0.7, 0.3], [0.35, 0.65]];

test('weights: one Q–Q panel per layer, live with training, normal and initial-weight comparisons', async ({ page }) => {
  await open(page);
  const sec = page.locator('#distributions');
  await sec.scrollIntoViewIfNeeded();
  const panels = page.locator('#dist-root .dist-panel');
  await expect(panels).toHaveCount(4);
  await expect(panels.locator('h3')).toHaveText(['Conv 1', 'Conv 2', 'Dense 3', 'Output']);
  await expect(panels.first().locator('.dist-detail')).toHaveText('8×1×3×3 weights');
  await expect(page.locator('#dist-q-weights')).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator('#dist-cmp-init')).toBeEnabled();
  await expect(page.locator('#dist-root .synced-status')).toContainText('Based on the weights at step 0');
  for (let i = 0; i < 4; i++) {
    const p = panels.nth(i);
    expect(await painted(p.locator('canvas').first())).toBeGreaterThan(2000);
    expect(await painted(p.locator('canvas').nth(1))).toBeGreaterThan(300);
    await expect(p.locator('.dist-stats')).toContainText('PPCC r');
  }
  // Freshly initialised weights are normal: PPCC very close to 1 for the big dense layer.
  const r0 = Number((await panels.nth(2).locator('.dist-stats > span', { hasText: 'PPCC r' }).locator('b').textContent())!);
  expect(r0).toBeGreaterThan(0.999);
  await expect(panels.first().locator('canvas').first()).toHaveAttribute('aria-label', /Q–Q plot of Conv 1 weights against a normal distribution/);

  // Tooltips on the Q–Q plot and the histogram.
  const tip = await hoverTip(page, panels.nth(2).locator('canvas').first(), DIAGONAL);
  expect(tip).toMatch(/^p [\d.]+ · normal −?[\d.]+ · sample −?[\d.e−]+$/);
  const hist = panels.nth(2).locator('canvas').nth(1);
  const htip = await hoverTip(page, hist, [[0.5, 0.4], [0.45, 0.4], [0.55, 0.4]]);
  expect(htip).toMatch(/ to .*\n[\d,]+ weights \([\d.]+%\)/);
  await page.mouse.move(0, 0);

  // Training moves the weights; the panels follow (throttled) without a recompute.
  const before = await panels.nth(0).locator('.dist-stats').textContent();
  await trainUntil(page, '(window.raster.store.status?.step ?? 0) > 60');
  const step = await raster(page, (r) => r.store.weightsStep);
  await expect(page.locator('#dist-root .synced-status')).toContainText(`Based on the weights at step ${step.toLocaleString('en-US')}`);
  await expect(panels.nth(0).locator('.dist-stats')).not.toHaveText(before!);
  await sec.screenshot({ path: `${SHOTS}/10-distributions-weights.png` });

  // Against the initial weights: two-sample plot with y = x.
  await page.click('#dist-cmp-init');
  await expect(page.locator('#dist-cmp-init')).toHaveAttribute('aria-pressed', 'true');
  await expect(panels.first().locator('.dist-stats')).toContainText('std at start');
  await expect(panels.first().locator('canvas').first()).toHaveAttribute('aria-label', /against the initial weights/);
  const itip = await hoverTip(page, panels.nth(1).locator('canvas').first(), DIAGONAL);
  expect(itip).toMatch(/^p [\d.]+ · initial −?[\d.e−]+ · now −?[\d.e−]+$/);
  const htip2 = await hoverTip(page, panels.nth(1).locator('canvas').nth(1), [[0.5, 0.4], [0.45, 0.4]]);
  expect(htip2).toContain('at start');
  await page.mouse.move(0, 0);
  await sec.screenshot({ path: `${SHOTS}/10-distributions-initial.png` });
});

test('pre-activations, activations and gradients come from the layerStats job', async ({ page }) => {
  await open(page);
  await trainUntil(page, '(window.raster.store.status?.step ?? 0) > 40');
  const sec = page.locator('#distributions');
  await sec.scrollIntoViewIfNeeded();
  const panels = page.locator('#dist-root .dist-panel');

  await page.click('#dist-q-a');
  await expect(page.locator('#dist-cmp-init')).toBeDisabled();
  await expect(page.locator('#dist-cmp-normal')).toHaveAttribute('aria-pressed', 'true');
  await expect(panels.first().locator('.dist-stats')).toContainText('dead units', { timeout: 30_000 });
  await expect(page.locator('#dist-root .progress')).toBeHidden();
  const step = await raster(page, (r) => r.store.weightsStep);
  await expect(page.locator('#dist-root .synced-status')).toContainText(`Based on the weights at step ${step.toLocaleString('en-US')}`);
  for (let i = 0; i < 3; i++) {
    await expect(panels.nth(i).locator('.dist-stats')).toContainText('exactly zero');
    await expect(panels.nth(i).locator('.dist-stats')).toContainText(/dead units \d+ of \d+/);
  }
  await expect(panels.first().locator('.dist-detail')).toHaveText('28×28×8 · ReLU · before pooling');
  await expect(panels.first().locator('.dist-stats')).toContainText('n 20,000 of 1.6M');
  await expect(panels.nth(3).locator('.dist-detail')).toHaveText('10 logits · no activation function');
  await expect(panels.nth(3).locator('.dist-stats')).not.toContainText('dead units');
  // ReLU outputs are never negative and mostly exactly zero.
  const zero = Number((await panels.first().locator('.dist-stats > span', { hasText: 'exactly zero' }).locator('b').textContent())!.replace('%', ''));
  expect(zero).toBeGreaterThan(10);
  const tip = await hoverTip(page, panels.nth(1).locator('canvas').first(), DIAGONAL);
  expect(tip).toMatch(/^p [\d.]+ · normal −?[\d.]+ · sample [\d.e−]+$/);
  await page.mouse.move(0, 0);
  await sec.screenshot({ path: `${SHOTS}/10-distributions-activations.png` });

  // Pre-activations and gradients reuse the same job result.
  await page.click('#dist-q-z');
  await expect(panels.first().locator('.dist-stats')).not.toContainText('dead units');
  await expect(panels.first().locator('.dist-detail')).toHaveText('28×28×8 · ReLU');
  await page.click('#dist-q-grad');
  await expect(panels.first().locator('.dist-detail')).toHaveText('8×1×3×3 · mean ∂L/∂W over 256 digits');
  await expect(panels.nth(2).locator('.dist-stats')).toContainText('n 25,088');
  expect(await painted(panels.nth(2).locator('canvas').first())).toBeGreaterThan(2000);
  await sec.screenshot({ path: `${SHOTS}/10-distributions-gradients.png` });

  // While training runs the result keeps its step; pausing brings it up to date.
  await page.click('#dist-q-a');
  await page.click('#play');
  await page.waitForFunction((s) => ((window as unknown as { raster: Raster }).raster.store.weightsStep ?? 0) > s + 20, step);
  await expect(page.locator('#dist-root .synced-status')).toContainText('the network is now at step');
  await page.click('#play');
  await page.waitForFunction(() => !(window as unknown as { raster: Raster }).raster.store.status?.running);
  const later = await raster(page, (r) => r.store.weightsStep);
  await expect(page.locator('#dist-root .synced-status')).toContainText(`Based on the weights at step ${later.toLocaleString('en-US')}`, { timeout: 30_000 });

  // The job itself, called directly through the page's analysis client.
  const res = await raster(page, async (r) => {
    const out = await r.analysis.run('e2e-dist', 'layerStats', { samples: 8, maxValues: 300 });
    return {
      samples: out.samples,
      z: out.layers.map((l) => l.z.length),
      gW: out.layers.map((l) => l.gW.length),
      W: r.store.net.blocks.map((b) => b.W.length),
      dead: out.layers.map((l) => l.dead),
    };
  });
  expect(res.samples).toBe(8);
  expect(res.z).toEqual([300, 300, 256, 80]);
  expect(res.gW).toEqual(res.W);
  expect(res.dead[3]).toBeNull();
});

test('weight inspector Q–Q view: now against initialisation, with a tooltip', async ({ page }) => {
  await open(page);
  await trainUntil(page, '(window.raster.store.status?.step ?? 0) > 60');
  const insp = page.locator('#inspector');
  await insp.getByRole('button', { name: 'Q–Q' }).click();
  await expect(insp.getByRole('button', { name: 'Q–Q' })).toHaveAttribute('aria-pressed', 'true');
  await page.selectOption('#insp-layer', '2');
  const canvas = insp.locator('canvas');
  await expect.poll(() => canvas.evaluate((c: HTMLCanvasElement) => c.height)).toBeGreaterThan(300);
  expect(await painted(canvas)).toBeGreaterThan(4000);
  const box = (await canvas.boundingBox())!;
  let tip = '';
  for (const fy of [120, 160, 200, 240]) {
    await page.mouse.move(box.x + 220, box.y + fy);
    if (await page.locator('#tip').isVisible()) {
      tip = (await page.locator('#tip').textContent()) ?? '';
      break;
    }
  }
  expect(tip).toMatch(/^p [\d.]+ · normal −?[\d.]+\nnow −?[\d.e−]+\nat init −?[\d.e−]+$/);
  // Clicking the plot does not change the selected unit.
  await page.mouse.click(box.x + 220, box.y + 160);
  await page.mouse.move(0, 0);
  await page.locator('#weights').screenshot({ path: `${SHOTS}/10-distributions-inspector-qq.png` });
});

test('dark theme', async ({ page }) => {
  await page.emulateMedia({ colorScheme: 'dark' });
  await open(page);
  await page.click('#step');
  const sec = page.locator('#distributions');
  await sec.scrollIntoViewIfNeeded();
  const panel = page.locator('#dist-root .dist-panel').first();
  await expect(panel.locator('.dist-stats')).toContainText('PPCC r');
  // Points are drawn in the dark theme's light ink.
  const light = await panel.locator('canvas').first().evaluate((el: HTMLCanvasElement) => {
    const d = el.getContext('2d')!.getImageData(0, 0, el.width, el.height).data;
    let n = 0;
    for (let i = 0; i < d.length; i += 4) if (d[i + 3] > 200 && d[i] > 200 && d[i + 1] > 200 && d[i + 2] > 200) n++;
    return n;
  });
  expect(light).toBeGreaterThan(500);
  await sec.screenshot({ path: `${SHOTS}/10-distributions-dark.png` });

  // Shrinking the window re-lays the panels out within the new width.
  await page.setViewportSize({ width: 390, height: 844 });
  await page.waitForTimeout(400);
  const right = await page.locator('#dist-root *').evaluateAll((els) => Math.max(...els.map((e) => e.getBoundingClientRect().right)));
  expect(right).toBeLessThanOrEqual(390);
});

test('390 px phone screen: no sideways scrolling, panels stack', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await open(page);
  const sec = page.locator('#distributions');
  await sec.scrollIntoViewIfNeeded();
  const panels = page.locator('#dist-root .dist-panel');
  await expect(panels.first().locator('.dist-stats')).toContainText('PPCC r');
  const overflow = () => page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  expect(await overflow()).toBeLessThanOrEqual(0);
  const rights = await page.locator('#dist-root canvas').evaluateAll((cs) => cs.map((c) => c.getBoundingClientRect().right));
  for (const r of rights) expect(r).toBeLessThanOrEqual(390 - 16 + 0.5);
  // One column: every panel starts at the same x.
  const xs = await panels.evaluateAll((ps) => ps.map((p) => Math.round(p.getBoundingClientRect().left)));
  expect(new Set(xs).size).toBe(1);
  await page.click('#dist-q-a');
  await expect(panels.first().locator('.dist-stats')).toContainText('dead units', { timeout: 30_000 });
  expect(await overflow()).toBeLessThanOrEqual(0);
  const tip = await hoverTip(page, panels.first().locator('canvas').first(), DIAGONAL);
  expect(tip).toContain('normal');
  await page.mouse.move(0, 0);
  await panels.first().scrollIntoViewIfNeeded();
  await page.evaluate(() => window.scrollBy(0, -80));
  await page.screenshot({ path: `${SHOTS}/10-distributions-phone.png` });
});
