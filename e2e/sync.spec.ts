import { expect, test, type Page } from '@playwright/test';

/**
 * The shared sync policy of sections 08–11 while training runs. The trainer sends new weights
 * about three times a second; a section with nothing to show must let its one job finish instead
 * of restarting it on every tick (round-2 finding NEW-1).
 */

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

interface Run {
  channel: string;
  outcome: 'running' | 'done' | 'superseded' | 'error';
}

/** Records every analysis run and how it ended. */
async function recordRuns(page: Page) {
  await page.evaluate(() => {
    const w = window as unknown as { raster: { analysis: { run: (...a: unknown[]) => Promise<unknown> } }; runs: Run[] };
    const a = w.raster.analysis;
    const orig = a.run.bind(a);
    w.runs = [];
    a.run = (...args: unknown[]) => {
      const rec: Run = { channel: String(args[0]), outcome: 'running' };
      w.runs.push(rec);
      const p = orig(...args);
      p.then(
        () => (rec.outcome = 'done'),
        (e: unknown) => (rec.outcome = e instanceof Error && e.message === 'superseded' ? 'superseded' : 'error'),
      );
      return p;
    };
  });
}

const runs = (page: Page, channel: string) =>
  page.evaluate((c) => (window as unknown as { runs: Run[] }).runs.filter((r) => r.channel === c), channel);

async function open(page: Page) {
  await page.goto('/');
  await expect(page.locator('#fact-data')).toContainText('train', { timeout: 30_000 });
  await expect(page.locator('#play')).toBeEnabled();
}

const running = (page: Page) => page.evaluate(() => !!(window as unknown as { raster: { store: { status: { running: boolean } | null } } }).raster.store.status?.running);

test('first view during training: one run per section, and it finishes while training continues', async ({ page }) => {
  await open(page);
  await page.click('#play');
  // Past the first evaluation, so the weights are moving on every tick.
  await page.waitForFunction(() => (window as unknown as { raster: { store: { evals: unknown[]; weightsStep: number } } }).raster.store.evals.length >= 1, null, {
    timeout: 60_000,
  });
  await recordRuns(page);
  await page.locator('#units').scrollIntoViewIfNeeded();
  await expect.poll(async () => (await runs(page, 'units-topk')).some((r) => r.outcome === 'done'), { timeout: 20_000 }).toBe(true);
  expect(await running(page), 'still training').toBe(true);
  const topk = await runs(page, 'units-topk');
  expect(topk.filter((r) => r.outcome === 'superseded'), 'no units-topk run was restarted').toEqual([]);
  await expect(page.locator('#units-root .units-card').first()).toBeVisible();

  await page.locator('#embedding').scrollIntoViewIfNeeded();
  await expect.poll(async () => (await runs(page, 'embedding')).some((r) => r.outcome === 'done'), { timeout: 30_000 }).toBe(true);
  expect(await running(page), 'still training').toBe(true);
  expect((await runs(page, 'embedding')).filter((r) => r.outcome === 'superseded'), 'no embedding run was restarted').toEqual([]);
  await page.click('#play');
});

test('Reset then Play with 08–11 all on screen: each section computes once and shows a result', async ({ page }) => {
  // Tall enough that all four analysis sections are visible and share the analysis worker.
  await page.setViewportSize({ width: 1600, height: 9000 });
  await open(page);
  // Activations need a job in the analysis worker (weights are summarised on the page).
  await page.click('#dist-q-a');
  for (const id of ['#dist-root', '#units-root', '#attr-root', '#embed-root']) {
    await expect(page.locator(`${id} .synced-status`)).toContainText('Based on the weights at step 0', { timeout: 30_000 });
  }
  await recordRuns(page);
  await page.click('#reset');
  await page.click('#play');
  for (const channel of ['distributions', 'units-topk', 'attribution', 'embedding']) {
    await expect.poll(async () => (await runs(page, channel)).some((r) => r.outcome === 'done'), { message: `${channel} finished`, timeout: 30_000 }).toBe(true);
    expect((await runs(page, channel)).filter((r) => r.outcome === 'superseded'), `no ${channel} run was restarted`).toEqual([]);
  }
  expect(await running(page), 'still training').toBe(true);
  // While training holds the result, the status says which step it belongs to, on one line.
  const status = page.locator('#dist-root .synced-status');
  await expect(status).toContainText(/step/);
  const height = await status.evaluate((el) => el.getBoundingClientRect().height);
  expect(height).toBeLessThanOrEqual(30);
  await page.click('#play');
});

test('phone width: the status row stays one line in every state', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await open(page);
  await page.locator('#units').scrollIntoViewIfNeeded();
  const status = page.locator('#units-root .synced-status');
  await expect(status).toContainText('Based on the weights at step 0', { timeout: 30_000 });
  const h0 = await status.evaluate((el) => el.getBoundingClientRect().height);
  // Train a little, pause: the long "Computed at step 0; the network is now at step N." text.
  await page.click('#play');
  await page.waitForFunction(() => (window as unknown as { raster: { store: { weightsStep: number } } }).raster.store.weightsStep > 30);
  await page.click('#play');
  await expect(status.locator('.hint')).toHaveAttribute('title', /.+/);
  const h1 = await status.evaluate((el) => el.getBoundingClientRect().height);
  expect(h1).toBe(h0);
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
});
