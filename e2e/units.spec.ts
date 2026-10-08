import { expect, test, type Page } from '@playwright/test';

/**
 * Section 09 Units: top digits per unit, the detail panel, activation maximisation, and the
 * phone layout. Screenshots land in docs/screenshots/11-units-*.png.
 */

const SHOTS = 'docs/screenshots';

type Raster = {
  store: {
    status: { step: number; running: boolean } | null;
    evals: { acc: number }[];
    probe: { key: string; caption: string } | null;
    selected: number;
    selectedUnit: number | null;
    spec: unknown[];
    emit(ev: string): void;
  };
  analysis: { mode: string };
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

const cards = (page: Page) => page.locator('#units-root .units-card');

/** Screenshot of the whole section; the sticky control bar would otherwise cover its top. */
async function shot(page: Page, name: string) {
  const style = await page.addStyleTag({ content: '#bar { position: static !important; }' });
  await page.locator('#units').screenshot({ path: `${SHOTS}/${name}` });
  await style.evaluate((el) => (el as Element).remove());
}

/** Waits until the top-k scan has filled every visible card with its top 9 images. */
async function scanned(page: Page, n: number) {
  await expect(cards(page)).toHaveCount(n);
  await expect(page.locator('#units-root .units-card .units-mosaic canvas')).toHaveCount(9 * n, { timeout: 60_000 });
  await expect(page.locator('#units-root .synced-status')).toContainText('Based on the weights at step');
}

test('top digits, detail panel and synthesised inputs for each layer', async ({ page }) => {
  await open(page);
  await trainUntil(page, '(window.raster.store.evals.length >= 3)');
  await page.locator('#units').scrollIntoViewIfNeeded();

  // Conv 1 (the selected layer) has 8 filters; each card shows 9 receptive-field crops (3×3 px).
  await scanned(page, 8);
  await expect(page.locator('#units-root .hint').first()).toContainText('Top images are the test digits that excite a unit most');
  const crop = await page.locator('#units-root .units-card .units-mosaic canvas').first().evaluate((c: HTMLCanvasElement) => [c.width, c.height]);
  expect(crop).toEqual([3, 3]);
  await expect(cards(page).first()).toContainText(/active on \d+% of digits/);
  await expect(cards(page).first()).toContainText(/mean -?\d/);

  // Switch to conv 2 with the layer select: 16 filters, 8×8 crops; the global selection follows.
  await page.selectOption('#units-layer', '1');
  expect(await raster(page, (r) => r.store.selected)).toBe(1);
  await scanned(page, 16);
  expect(await page.locator('#units-root .units-card .units-mosaic canvas').first().evaluate((c: HTMLCanvasElement) => c.width)).toBe(8);

  // Clicking a card selects that filter (network diagram and inspector follow) and opens details.
  await cards(page).nth(2).click();
  expect(await raster(page, (r) => [r.store.selected, r.store.selectedUnit])).toEqual([1, 2]);
  await expect(cards(page).nth(2)).toHaveAttribute('aria-pressed', 'true');
  await expect(cards(page).nth(0)).toHaveAttribute('aria-pressed', 'false');
  const detail = page.locator('#units-detail');
  await expect(detail.locator('h3')).toHaveText('Filter 3');
  await expect(detail.locator('.units-digit')).toHaveCount(16 + 8);
  await expect(detail.locator('.units-digit-box')).toHaveCount(16 + 8);
  await expect(detail.locator('.units-label-col')).toHaveCount(10);
  await expect(detail.locator('.units-label-sum')).toContainText(/^(Mostly|Mixed:|All)/);
  await expect(detail.locator('.units-probe')).toContainText(/This input: -?\d/);
  const hist = detail.locator('.units-hist canvas');
  await expect(hist).toHaveAttribute('role', 'img');
  await hist.hover();
  await expect(page.locator('#tip')).toBeVisible();
  await expect(page.locator('#tip')).toContainText(/digits?/);
  await page.mouse.move(0, 0);

  // A digit in the panel becomes the network's input.
  const first = detail.locator('.units-digit').first();
  const idx = await first.getAttribute('data-index');
  await first.click();
  expect(await raster(page, (r) => r.store.probe?.key)).toBe(`test:${idx}`);
  await expect(detail.locator('.units-digit').first()).toHaveAttribute('aria-pressed', 'true');

  // Synthesise the whole layer: the button turns into Stop, cards fill in, the result is reported.
  await page.click('#units-synth');
  await expect(page.locator('#units-synth')).toHaveText('Stop');
  await expect(page.locator('#units-root .units-synth-ctl .progress')).toBeVisible();
  await expect(page.locator('#units-root .units-synth-note')).toContainText('Synthesised from the weights at step', { timeout: 90_000 });
  await expect(page.locator('#units-synth')).toHaveText('Synthesise inputs');
  await expect(page.locator('#units-root .units-card .units-synth canvas:visible')).toHaveCount(16);
  await expect(detail.locator('.units-synth-large canvas')).toBeVisible();
  await expect(detail).toContainText(/pre-activation rose from -?\d+\.\d+ on a blank image to -?\d+\.\d+ after 160 steps/);
  const rose = await detail.locator('.units-block').last().innerText();
  const m = rose.match(/from (-?[\d.]+) on a blank image to (-?[\d.]+)/)!;
  expect(Number(m[2])).toBeGreaterThan(Number(m[1]));

  await shot(page, '11-units-light.png');

  // Dark mode: canvases redraw from the theme tokens.
  await page.emulateMedia({ colorScheme: 'dark' });
  await page.waitForTimeout(300);
  const bg = await page.locator('#units-root .units-card .units-mosaic canvas').first().evaluate((c: HTMLCanvasElement) => {
    const d = c.getContext('2d')!.getImageData(0, 0, c.width, c.height).data;
    let min = 255;
    for (let i = 0; i < d.length; i += 4) min = Math.min(min, d[i]);
    return min;
  });
  expect(bg, 'blank pixels use the dark surface').toBeLessThan(60);
  await shot(page, '11-units-dark.png');
  await page.emulateMedia({ colorScheme: 'light' });

  // The dense layer shows full digits; the output layer is named by digit.
  await page.selectOption('#units-layer', '2');
  await scanned(page, 32);
  expect(await page.locator('#units-root .units-card .units-mosaic canvas').first().evaluate((c: HTMLCanvasElement) => c.width)).toBe(28);
  await page.selectOption('#units-layer', '3');
  await scanned(page, 10);
  await expect(cards(page).nth(7).locator('.units-card-title')).toHaveText('Digit 7');
  await cards(page).nth(7).click();
  await expect(detail.locator('.units-digit-box')).toHaveCount(0);
  await expect(detail).toContainText('the logit for digit 7');
  // A trained output unit's favourite digits are mostly its own digit.
  const sevens = await detail.locator('.units-label-col').nth(7).locator('.units-label-n').innerText();
  expect(Number(sevens)).toBeGreaterThan(25);
  await shot(page, '11-units-output.png');

  // Going back to conv 2 keeps its synthesised inputs.
  await page.selectOption('#units-layer', '1');
  await scanned(page, 16);
  await expect(page.locator('#units-root .units-card .units-synth canvas:visible')).toHaveCount(16);
});

test('follows the selection from other views, shows all units on request, works from the keyboard', async ({ page }) => {
  await open(page);
  await page.getByRole('button', { name: 'MLP', exact: true }).click();
  await page.locator('#units').scrollIntoViewIfNeeded();
  // Selecting elsewhere (as the network diagram does) moves this section to that layer and unit.
  await raster(page, (r) => {
    r.store.selected = 0;
    r.store.selectedUnit = 40;
    r.store.emit('select');
  });
  // 64 hidden units: unit 41 is past the first 32, so every card is shown.
  await expect(cards(page)).toHaveCount(64, { timeout: 60_000 });
  await expect(page.locator('#units-show-all')).toHaveText('Show the first 32');
  await expect(cards(page).nth(40)).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator('#units-detail h3')).toHaveText('Unit 41');
  await page.click('#units-show-all');
  await expect(cards(page)).toHaveCount(32);
  await expect(page.locator('#units-show-all')).toHaveText('Show all 64');
  await expect(page.locator('#units-root .units-card .units-mosaic canvas')).toHaveCount(9 * 32, { timeout: 60_000 });

  // Keyboard: Tab to a card and press Enter.
  await cards(page).nth(4).focus();
  await page.keyboard.press('Enter');
  expect(await raster(page, (r) => [r.store.selected, r.store.selectedUnit])).toEqual([0, 4]);
  await expect(page.locator('#units-detail h3')).toHaveText('Unit 5');
  await page.keyboard.press('Tab');
  await page.keyboard.press('Space');
  expect(await raster(page, (r) => r.store.selectedUnit)).toBe(5);

  // Stop halts a synthesis run part-way.
  await page.click('#units-synth');
  await expect(page.locator('#units-synth')).toHaveText('Stop');
  await expect(page.locator('#units-root .units-synth-note')).toContainText('Synthesising Unit');
  await page.click('#units-synth');
  await expect(page.locator('#units-synth')).toHaveText('Synthesise inputs');
  await expect(page.locator('#units-root .units-synth-note')).toContainText(/Stopped after \d+ of 64 units/);
});

test('phone width: no horizontal overflow, detail below the cards', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await open(page);
  await page.locator('#units').scrollIntoViewIfNeeded();
  await scanned(page, 8);
  await page.selectOption('#units-layer', '1');
  await scanned(page, 16);
  await cards(page).nth(5).click();
  await expect(page.locator('#units-detail .units-digit')).toHaveCount(24);
  await page.click('#units-synth');
  await expect(page.locator('#units-root .units-synth-note')).toContainText('Synthesised from the weights at step', { timeout: 90_000 });
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow, 'no horizontal page scroll at 390 px').toBeLessThanOrEqual(0);
  const sec = await page.locator('#units').evaluate((el) => {
    const r = el.getBoundingClientRect();
    const out: string[] = [];
    for (const c of el.querySelectorAll<HTMLElement>('*')) {
      const b = c.getBoundingClientRect();
      if (b.width && (b.right > r.right + 1 || b.left < r.left - 1) && !c.closest('.canvas-box')) out.push(`${c.className} ${Math.round(b.left)}–${Math.round(b.right)}`);
    }
    return out;
  });
  expect(sec, 'nothing pokes out of the section').toEqual([]);
  const [grid, detail] = await Promise.all([page.locator('#units-root .units-grid').boundingBox(), page.locator('#units-detail').boundingBox()]);
  expect(detail!.y).toBeGreaterThan(grid!.y + grid!.height - 1);
  await shot(page, '11-units-phone.png');
});
