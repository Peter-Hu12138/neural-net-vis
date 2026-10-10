import { expect, test, type Page } from '@playwright/test';

/**
 * Section 09 Units: top digits per unit, the detail panel, activation maximisation, and the
 * phone layout. Screenshots land in docs/screenshots/11-units-*.png.
 */

const SHOTS = 'docs/screenshots';

type Raster = {
  store: {
    status: { step: number; running: boolean } | null;
    weightsStep: number;
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
const status = (page: Page) => page.locator('#units-root .synced-status');

/** Screenshot of the whole section; the sticky control bar would otherwise cover its top. */
async function shot(page: Page, name: string) {
  await page.mouse.move(0, 0); // no hover outlines in the documentation shots
  const style = await page.addStyleTag({ content: '#bar { position: static !important; }' });
  await page.locator('#units').screenshot({ path: `${SHOTS}/${name}` });
  await style.evaluate((el) => (el as Element).remove());
}

/** Waits until the top-k scan has filled every visible card with its top 9 images. */
async function scanned(page: Page, n: number) {
  await expect(cards(page)).toHaveCount(n);
  await expect(page.locator('#units-root .units-card .units-mosaic canvas')).toHaveCount(9 * n, { timeout: 60_000 });
  await expect(status(page)).toContainText('Based on the weights at step');
}

/** "Current input, test digit #0 · label 7: 0.901, higher than …" → 0.901 */
const probeValue = (line: string) => Number(line.match(/: (−?[\d.]+(?:e−?\d+)?),/)![1].replace(/−/g, '-'));

test('top digits, detail panel and synthesised inputs for each layer', async ({ page }) => {
  await open(page);
  await trainUntil(page, '(window.raster.store.evals.length >= 3)');
  await page.locator('#units').scrollIntoViewIfNeeded();

  // Conv 1 (the selected layer) has 8 filters; each card shows 9 receptive-field crops (3×3 px).
  await scanned(page, 8);
  await expect(page.locator('#units-root .units-key')).toContainText(
    'Each card: the 3×3 patches of the 9 test digits that excite the filter most (left), and an input synthesised from blank to excite it (right).',
  );
  const crop = await page.locator('#units-root .units-card .units-mosaic canvas').first().evaluate((c: HTMLCanvasElement) => [c.width, c.height]);
  expect(crop).toEqual([3, 3]);
  // How often a filter fires: the share of positions with z > 0, not "fired somewhere" (always 100%).
  await expect(cards(page).first()).toContainText(/fires at [<>]?\d+% of positions/);
  const covers = await page.locator('#units-root .units-card-foot').allInnerTexts();
  expect(covers.filter((t) => t === 'fires at 100% of positions').length, covers.join(' | ')).toBeLessThan(covers.length);
  await expect(cards(page).first()).toHaveAttribute('aria-label', /^Filter 1: fires at [<>]?\d+% of positions on average, mean response −?\d/);
  await expect(cards(page).first()).toContainText(/mean −?\d/);

  // Switch to conv 2 with the layer select: 16 filters, 8×8 crops; the global selection follows.
  await page.selectOption('#units-layer', '1');
  expect(await raster(page, (r) => r.store.selected)).toBe(1);
  await scanned(page, 16);
  expect(await page.locator('#units-root .units-card .units-mosaic canvas').first().evaluate((c: HTMLCanvasElement) => c.width)).toBe(8);
  await expect(page.locator('#units-root .units-synth-note')).toHaveText('Optimises an input for each of the 16 filters, 160 steps each.');

  // Clicking a card selects that filter (network diagram and inspector follow) and opens details.
  await cards(page).nth(2).click();
  expect(await raster(page, (r) => [r.store.selected, r.store.selectedUnit])).toEqual([1, 2]);
  await expect(cards(page).nth(2)).toHaveAttribute('aria-pressed', 'true');
  await expect(cards(page).nth(0)).toHaveAttribute('aria-pressed', 'false');
  const detail = page.locator('#units-detail');
  await expect(detail.locator('h3')).toHaveText('Filter 3');
  await expect(detail.locator('h3')).toHaveClass(/panel-title/);
  await expect(detail.locator('h3')).toHaveCSS('font-size', '16px');
  await expect(detail.locator('.units-digit')).toHaveCount(16 + 8);
  await expect(detail.locator('.units-digit-box')).toHaveCount(16 + 8);
  await expect(detail.locator('.units-label-col')).toHaveCount(10);
  await expect(detail.locator('.units-label-sum')).toContainText(/^(Mostly|Mixed:|All)/);
  await expect(detail.locator('.units-stats')).toContainText(/fires at [<>]?\d+(\.\d)?% of positions/);
  await expect(detail.locator('.units-stats')).toContainText('sees 8×8 px');
  await expect(detail).toContainText('The box marks the 8×8 patch where the filter fired hardest.');
  // A conv filter fires somewhere on every digit, even its weakest: no "switches it off" claim.
  await expect(detail.locator('.units-weak-title')).toHaveText('Weakest responses · bottom 8');
  // The current input is named, with its exact rank among the test digits.
  await expect(detail.locator('.units-probe')).toContainText(/^Current input, test digit #0 · label 7: −?\d/);
  const hist = detail.locator('.units-hist canvas');
  await expect(hist).toHaveAttribute('role', 'img');
  await hist.hover();
  await expect(page.locator('#tip')).toBeVisible();
  await expect(page.locator('#tip')).toContainText(/digits?/);
  await page.mouse.move(0, 0);

  // A digit in the panel becomes the network's input. The strongest digit ranks first.
  const first = detail.locator('.units-digit').first();
  const idx = await first.getAttribute('data-index');
  await first.click();
  expect(await raster(page, (r) => r.store.probe?.key)).toBe(`test:${idx}`);
  await expect(detail.locator('.units-digit').first()).toHaveAttribute('aria-pressed', 'true');
  await expect(detail.locator('.units-probe')).toContainText(new RegExp(`^Current input, test digit #${idx} · label \\d: [\\d.]+, as high as the strongest of the 2,000 test digits\\.$`));
  const max = (await detail.locator('.units-stats').innerText()).match(/max ([\d.]+)/)![1];
  expect(probeValue(await detail.locator('.units-probe').innerText()).toFixed(3)).toBe(max);

  // Synthesise the whole layer: the button turns into Stop, cards fill in, the result is reported.
  await page.click('#units-synth');
  await expect(page.locator('#units-synth')).toHaveText('Stop');
  await expect(page.locator('#units-root .units-synth-ctl .progress')).toBeVisible();
  await expect(page.locator('#units-root .units-synth-note')).toContainText('Synthesised from the weights at step', { timeout: 90_000 });
  await expect(page.locator('#units-synth')).toHaveText('Synthesise inputs');
  await expect(page.locator('#units-root .units-card .units-synth canvas:visible')).toHaveCount(16);
  await expect(detail.locator('.units-synth-large canvas')).toBeVisible();
  await expect(detail).toContainText('Cropped to the 8×8 patch the filter sees at the centre of the image.');
  await expect(detail).toContainText(/pre-activation rose from −?\d+\.\d+ on a blank image to −?\d+\.\d+ after 160 steps/);
  const rose = await detail.locator('.units-block').last().innerText();
  const m = rose.match(/from (−?[\d.]+) on a blank image to (−?[\d.]+)/)!;
  expect(Number(m[2].replace('−', '-'))).toBeGreaterThan(Number(m[1].replace('−', '-')));
  // True minus signs, and never a "−0.00" (tiny values keep their digits, as in "0.00022").
  const text = await page.locator('#units').innerText();
  expect(text).not.toMatch(/(^|[\s(])-\d/m);
  expect(text).not.toMatch(/[−-]0\.0+(?!\d)/);

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

  // The dense layer shows full digits and how often each unit fires; the output layer is named by digit.
  await page.selectOption('#units-layer', '2');
  await scanned(page, 32);
  expect(await page.locator('#units-root .units-card .units-mosaic canvas').first().evaluate((c: HTMLCanvasElement) => c.width)).toBe(28);
  await expect(cards(page).first()).toContainText(/fires on [<>]?\d+% of digits/);
  await page.selectOption('#units-layer', '3');
  await scanned(page, 10);
  await expect(cards(page).nth(7).locator('.units-card-title')).toHaveText('Digit 7');
  // Output units: how often each digit is predicted; the ten shares add up to about 100%.
  const predicted = await page.locator('#units-root .units-card-foot').allInnerTexts();
  expect(predicted.every((t) => /^predicted for [<>]?\d+% of digits$/.test(t)), predicted.join(' | ')).toBe(true);
  const sum = predicted.reduce((a, t) => a + Number(t.match(/(\d+)%/)![1]), 0);
  expect(sum).toBeGreaterThanOrEqual(95);
  expect(sum).toBeLessThanOrEqual(105);
  await cards(page).nth(7).click();
  await expect(detail.locator('.units-digit-box')).toHaveCount(0);
  await expect(detail).toContainText('the logit for digit 7');
  await expect(detail.locator('.units-stats')).toContainText(/predicted for \d+(\.\d)?% of digits/);
  await expect(detail.locator('.units-weak-title')).toHaveText(/^(What switches it off · weakest 8|Weakest responses · bottom 8)$/);
  // A trained output unit's favourite digits are mostly its own digit.
  const sevens = await detail.locator('.units-label-col').nth(7).locator('.units-label-n').innerText();
  expect(Number(sevens)).toBeGreaterThan(25);
  await shot(page, '11-units-output.png');

  // Going back to conv 2 keeps its synthesised inputs.
  await page.selectOption('#units-layer', '1');
  await scanned(page, 16);
  await expect(page.locator('#units-root .units-card .units-synth canvas:visible')).toHaveCount(16);
});

test('while training, the current input is ranked with the scan’s weights; the status matches what is shown', async ({ page }) => {
  await open(page);
  await page.locator('#units').scrollIntoViewIfNeeded();
  await scanned(page, 8);
  await page.selectOption('#units-layer', '1');
  await scanned(page, 16);
  await cards(page).nth(2).click();
  const detail = page.locator('#units-detail');
  const line = detail.locator('.units-probe');
  await expect(line).toContainText(/^Current input, test digit #0 · label 7: /);
  const before = await line.innerText();
  const v0 = probeValue(before);
  const inTop = await detail.locator('.units-digits').first().locator('.units-digit[data-index="0"]').count();
  if (!inTop) expect(before).not.toMatch(/as high as the strongest|higher than all/);
  await expect(detail).toHaveAttribute('data-step', '0');

  // Train with the section on screen. The scan is held (it already shows a result), and the
  // input's value stays the one measured with the scan's weights, with a note saying so.
  await page.click('#play');
  await page.waitForFunction(() => ((window as unknown as { raster: Raster }).raster.store.weightsStep ?? 0) >= 30, null, { timeout: 60_000 });
  await expect(status(page)).toContainText(/Computed at step 0; the network is now at step [\d,]+\./);
  await expect(line).toContainText('Measured with the weights at step 0, as the scan was.');
  expect(probeValue(await line.innerText())).toBe(v0);
  expect((await line.innerText()).split('.')[0]).toBe(before.split('.')[0]);
  await expect(detail).toHaveAttribute('data-step', '0');

  // Pause and watch the rescan: whenever the status says "Based on the weights at step N", the
  // panel shows the scan of step N (the old one is dimmed while the new one computes).
  await page.evaluate(() => {
    const w = window as unknown as { samples: string[][]; sampling: boolean };
    w.samples = [];
    w.sampling = true;
    const tick = () => {
      if (!w.sampling) return;
      const st = document.querySelector('#units-root .synced-status')?.textContent ?? '';
      const d = document.getElementById('units-detail')!;
      const dim = document.querySelector('#units-root .units-layout')!.classList.contains('is-updating');
      w.samples.push([st, d.dataset.step ?? '', String(dim)]);
      requestAnimationFrame(tick);
    };
    tick();
  });
  await page.click('#play');
  await page.waitForFunction(() => !(window as unknown as { raster: Raster }).raster.store.status?.running);
  const step = await raster(page, (r) => r.store.weightsStep);
  const stepText = step.toLocaleString('en-US');
  await expect(status(page)).toContainText(`Based on the weights at step ${stepText}.`, { timeout: 60_000 });
  await expect(detail).toHaveAttribute('data-step', String(step));
  await page.waitForTimeout(300);
  const samples = await page.evaluate(() => {
    const w = window as unknown as { samples: string[][]; sampling: boolean };
    w.sampling = false;
    return w.samples;
  });
  const based = samples.filter(([st]) => /Based on the weights at step/.test(st));
  expect(based.length).toBeGreaterThan(0);
  for (const [st, shown] of based) {
    const n = st.match(/step ([\d,]+)\./)![1].replace(/,/g, '');
    expect(shown, `status "${st}" while the panel shows step ${shown}`).toBe(n);
  }
  expect(samples.some(([st, , dim]) => /Updating to step/.test(st) && dim === 'true'), 'the old scan is dimmed while updating').toBe(true);
  expect(samples.at(-1)![2]).toBe('false');
  await expect(line).not.toContainText('Measured with');
  await expect(line).toContainText(/^Current input, test digit #0 · label 7: /);
});

test('deep conv stacks: fields are clipped to the 28×28 image in text and crops', async ({ page }) => {
  await open(page);
  // Small CNN + 2 conv layers: maps 28 → 14 → 7 → 3 → 1, so conv 4's nominal field is 38×38.
  await page.getByRole('button', { name: '+ Conv layer' }).click();
  await page.getByRole('button', { name: '+ Conv layer' }).click();
  expect(await raster(page, (r) => r.store.spec.length)).toBe(5);
  await page.locator('#units').scrollIntoViewIfNeeded();
  await page.selectOption('#units-layer', '3');
  await scanned(page, 8);
  // Crops are the whole 28×28 digit, never a 38×38 canvas with blank margins.
  const widths = await page.locator('#units-root .units-card .units-mosaic canvas').evaluateAll((cs) => cs.map((c) => (c as HTMLCanvasElement).width));
  expect(new Set(widths)).toEqual(new Set([28]));
  await expect(page.locator('#units-root .units-key')).toContainText(
    'Each card: the 9 test digits that excite the filter most (left), and an input synthesised from blank to excite it (right).',
  );
  await cards(page).first().click();
  const detail = page.locator('#units-detail');
  await expect(detail.locator('.units-stats')).toContainText('sees whole image');
  await expect(detail).toContainText('The box marks the pixels the filter sees from the position where it fired hardest.');
  await expect(detail).toContainText('The filter at the centre of its map sees the whole image.');
  await expect(page.locator('#units')).not.toContainText(/(38|58)×(38|58)/);
  // Boxes on the digits stay inside the image.
  const boxes = await detail.locator('.units-digit-box').evaluateAll((els) =>
    els.map((el) => {
      const s = (el as HTMLElement).style;
      return [s.left, s.top, s.width, s.height].map((v) => parseFloat(v));
    }),
  );
  for (const [l, t, w, h] of boxes) {
    expect(l).toBeGreaterThanOrEqual(0);
    expect(t).toBeGreaterThanOrEqual(0);
    expect(l + w).toBeLessThanOrEqual(100.001);
    expect(t + h).toBeLessThanOrEqual(100.001);
  }
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
  await page.locator('#units-show-all').scrollIntoViewIfNeeded();
  const y0 = (await page.locator('#units-show-all').boundingBox())!.y;
  await page.click('#units-show-all');
  await expect(cards(page)).toHaveCount(32);
  await expect(page.locator('#units-show-all')).toHaveText('Show all 64');
  // Collapsing keeps the button in place on screen instead of leaving the page below the section.
  expect(Math.abs((await page.locator('#units-show-all').boundingBox())!.y - y0)).toBeLessThan(2);
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
  // Record where the cards start while the first scan runs: the shared progress bar must not move them.
  await page.evaluate(() => {
    const w = window as unknown as { tops: number[]; heights: number[] };
    w.tops = [];
    w.heights = [];
    const tick = () => {
      const root = document.getElementById('units-root');
      const layout = document.querySelector('#units-root .units-layout');
      const st = document.querySelector('#units-root .synced-status');
      if (root && layout && st) {
        w.tops.push(Math.round(layout.getBoundingClientRect().top - root.getBoundingClientRect().top));
        w.heights.push(Math.round(st.getBoundingClientRect().height));
      }
      if (w.tops.length < 2000) requestAnimationFrame(tick);
    };
    tick();
  });
  await page.locator('#units').scrollIntoViewIfNeeded();
  await scanned(page, 8);
  const { tops, heights } = await page.evaluate(() => {
    const w = window as unknown as { tops: number[]; heights: number[] };
    return { tops: w.tops.slice(), heights: w.heights.slice() };
  });
  expect(new Set(tops).size, `cards start at ${[...new Set(tops)].join(', ')} px`).toBe(1);
  expect(Math.max(...heights), 'the status row is one 28 px line').toBeLessThanOrEqual(30);
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
  expect((await status(page).boundingBox())!.height).toBeLessThanOrEqual(30);
  await shot(page, '11-units-phone.png');
});

// ── Dead units, other datasets ───────────────────────────────────────────

type Page2 = { raster: { store: { net: { getWeights(): Float32Array[] } }; actions: { applyWeights(w: Float32Array[]): void; setDataset(id: string): Promise<void>; setSpec(spec: unknown[]): void } } };

/** Switches dataset through the page's actions and waits until its data is loaded. */
async function switchTo(page: Page, id: string, fact: string) {
  await page.evaluate((d) => (window as unknown as Page2).raster.actions.setDataset(d), id);
  await expect(page.locator('#fact-data')).toContainText(fact, { timeout: 60_000 });
}

/** How many of the canvases matching `sel` show colour (a pixel whose channels differ), not grey. */
const colourful = (page: Page, sel: string) =>
  page.locator(sel).evaluateAll((cs) =>
    cs.filter((c) => {
      const el = c as HTMLCanvasElement;
      if (!el.width || !el.height) return false;
      const d = el.getContext('2d')!.getImageData(0, 0, el.width, el.height).data;
      for (let i = 0; i < d.length; i += 4) if (Math.abs(d[i] - d[i + 1]) > 12 || Math.abs(d[i] - d[i + 2]) > 12) return true;
      return false;
    }).length,
  );

test('a unit that never fires says so, and shows the digits closest to firing (NEW-2)', async ({ page }) => {
  await open(page);
  await page.locator('#units').scrollIntoViewIfNeeded();
  await page.selectOption('#units-layer', '2');
  await scanned(page, 32);
  // Push Dense 3's first unit's bias far below anything its inputs reach (as "Apply to network" in 06 would change weights).
  await page.evaluate(() => {
    const r = (window as unknown as Page2).raster;
    const ws = r.store.net.getWeights();
    ws[2 * 2 + 1][0] = -1000;
    r.actions.applyWeights(ws);
  });
  const first = cards(page).first();
  await expect(first.locator('.units-card-foot')).toHaveText('never fires (dead)', { timeout: 60_000 });
  await expect(first).toHaveAttribute('aria-label', 'Unit 1: never fires (dead), mean response 0.00. Show details.');
  await expect(page.locator('#units-root .units-key')).toContainText(/(One unit never fires \(dead\); its card shows|\d+ units never fire \(dead\); their cards show) the digits closest to firing\./);
  await first.click();
  const detail = page.locator('#units-detail');
  await expect(detail.locator('.units-stats')).toContainText('never fires');
  await expect(detail.locator('.units-stats')).not.toContainText('fires on');
  await expect(detail).toContainText('Never fires on the 2,000 test digits; digits are ranked by pre-activation instead');
  await expect(detail.locator('.units-block > .sub').first()).toHaveText('Closest to firing · top 16');
  await expect(detail.locator('.units-weak-title')).toHaveText('Furthest from firing · bottom 8');
  await expect(detail).toContainText('Labels of the 50 closest to firing');
  await expect(detail).not.toContainText('What switches it off');
  await expect(detail).not.toContainText('Strongest responses');
  // The current input ties with every digit, and the line says so.
  await expect(detail.locator('.units-probe')).toContainText('0.000, the same as all 2,000 test digits.');

  // 08 counts it among the dead units.
  await page.locator('#distributions').scrollIntoViewIfNeeded();
  await page.click('#dist-q-a');
  const dead = page.locator('#dist-root .dist-panel').nth(2).locator('.dist-stats > span', { hasText: 'dead units' });
  await expect(dead).toContainText(/dead units [1-9]\d* of 32/, { timeout: 30_000 });
  expect((await dead.getAttribute('title'))!.split(': ')[1].split(', ')).toContain('1');
});

test('CIFAR-10: colour crops and thumbnails, class names, a colour synthesis', async ({ page }) => {
  await open(page);
  await switchTo(page, 'cifar10', 'CIFAR-10 · 10,000 train');
  await trainUntil(page, '(window.raster.store.weightsStep ?? 0) > 40');
  await page.locator('#units').scrollIntoViewIfNeeded();
  await scanned(page, 8);
  await expect(page.locator('#units-root .units-key')).toContainText(
    'Each card: the 3×3 patches of the 9 test images that excite the filter most (left), and an input synthesised from plain grey to excite it (right).',
  );
  const crop = await page.locator('#units-root .units-card .units-mosaic canvas').first().evaluate((c: HTMLCanvasElement) => [c.width, c.height]);
  expect(crop).toEqual([3, 3]);
  expect(await colourful(page, '#units-root .units-card .units-mosaic canvas')).toBeGreaterThan(20);
  await expect(cards(page).first().locator('.units-card-foot')).toHaveText(/^(fires at [<>]?\d+% of positions|never fires \(dead\))$/);

  // Synthesise conv 1: colour patches grown from plain grey.
  await page.click('#units-synth');
  await expect(page.locator('#units-root .units-synth-note')).toContainText('Synthesised from the weights at step', { timeout: 90_000 });
  await expect(page.locator('#units-root .units-card .units-synth canvas:visible')).toHaveCount(8);
  expect(await colourful(page, '#units-root .units-card .units-synth canvas')).toBeGreaterThan(4);
  await cards(page).nth(2).click();
  const detail = page.locator('#units-detail');
  await expect(detail.locator('h3')).toHaveText('Filter 3');
  await expect(detail.locator('.units-digit')).toHaveCount(16 + 8);
  await expect(detail.locator('.units-digit-box')).toHaveCount(16 + 8);
  expect(await detail.locator('.units-digit canvas').first().evaluate((c: HTMLCanvasElement) => c.width)).toBe(32);
  await expect(detail).toContainText('Click any image to make it the network’s input.');
  await expect(detail).toContainText(/pre-activation rose from −?\d+\.\d+ on a plain grey image to −?\d+\.\d+ after 160 steps/);
  await expect(detail.locator('.units-probe')).toContainText(/^Current input, test image #0 · [a-z]+: −?\d/);
  await expect(detail.locator('.units-label-row')).toHaveCount(10);
  await shot(page, '11-units-cifar.png');

  // Dense: whole 32×32 colour images.
  await page.selectOption('#units-layer', '2');
  await scanned(page, 32);
  expect(await page.locator('#units-root .units-card .units-mosaic canvas').first().evaluate((c: HTMLCanvasElement) => c.width)).toBe(32);
  const foot = await page.locator('#units-root .units-card-foot').allInnerTexts();
  expect(foot.every((t) => /^(fires on [<>]?\d+% of images|never fires \(dead\))$/.test(t)), foot.join(' | ')).toBe(true);

  // Output: one card per class, named.
  await expect(page.locator('#units-layer option').last()).toHaveText('Output · 10 logits');
  await page.selectOption('#units-layer', '3');
  await scanned(page, 10);
  await expect(page.locator('#units-root .units-card-title')).toHaveText(['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']);
  const predicted = await page.locator('#units-root .units-card-foot').allInnerTexts();
  expect(predicted.every((t) => /^predicted for [<>]?\d+% of images$/.test(t)), predicted.join(' | ')).toBe(true);
  await cards(page).nth(3).click();
  await expect(detail.locator('h3')).toHaveText('Cat');
  await expect(detail).toContainText('the logit for cat; softmax turns the logits into probabilities. An image is predicted as cat when this logit is the largest of the 10');
  await expect(detail.locator('.units-label-row')).toHaveCount(10);
  await expect(detail.locator('.units-label-sum')).toHaveText(/^(Mostly|Mixed:|All)/);
  await expect(detail.locator('.units-label-sum')).not.toContainText(/\ds \(/);
  await expect(page.locator('#units')).not.toContainText(/digit/i, { useInnerText: true });
  await shot(page, '11-units-cifar-output.png');
});

test('point datasets: top points, response maps over the plane or a slice, point lists', async ({ page }) => {
  await open(page);
  await switchTo(page, 'circle', 'Circle · 300 train');
  await trainUntil(page, '(window.raster.store.weightsStep ?? 0) > 200');
  await page.locator('#units').scrollIntoViewIfNeeded();
  await expect(page.locator('#units-layer option')).toHaveText(['Dense 1 · 8 · Tanh', 'Dense 2 · 8 · Tanh', 'Output · 2 logits']);
  await expect(cards(page)).toHaveCount(8);
  const maps = page.locator('#units-root .units-card canvas.units-map:visible');
  await expect(maps).toHaveCount(8, { timeout: 60_000 });
  await expect(page.locator('#units-root .units-card canvas.units-scatter')).toHaveCount(8);
  await expect(status(page)).toContainText('Based on the weights at step');
  // No synthesis: the map already shows the whole input space.
  await expect(page.locator('#units-synth')).toBeHidden();
  await expect(page.locator('#units-root .units-synth-note')).toHaveText('Each card maps the unit’s response over the input plane, x₁ across and x₂ up.');
  await expect(page.locator('#units-root .units-key')).toContainText(
    'Each card: the 9 test points that excite the unit most, in their class colours among the rest (left), and its response over the input plane (right). Red is positive, blue negative.',
  );
  expect(await maps.first().evaluate((c: HTMLCanvasElement) => [c.width, c.height])).toEqual([48, 48]);
  // Tanh units: the maps are diverging, red where positive and blue where negative.
  const hues = await maps.evaluateAll((cs) => {
    let red = 0;
    let blue = 0;
    for (const c of cs) {
      const d = (c as HTMLCanvasElement).getContext('2d')!.getImageData(0, 0, 48, 48).data;
      for (let i = 0; i < d.length; i += 4) {
        if (d[i] > d[i + 2] + 60) red++;
        if (d[i + 2] > d[i] + 60) blue++;
      }
    }
    return { red, blue };
  });
  expect(hues.red).toBeGreaterThan(500);
  expect(hues.blue).toBeGreaterThan(500);
  const foot = await page.locator('#units-root .units-card-foot').allInnerTexts();
  expect(foot.every((t) => /^(fires on [<>]?\d+% of points|never fires \(dead\))$/.test(t)), foot.join(' | ')).toBe(true);

  await cards(page).nth(2).click();
  const detail = page.locator('#units-detail');
  await expect(detail.locator('h3')).toHaveText('Unit 3');
  const plane = detail.locator('.units-plane canvas');
  await expect(plane).toHaveAttribute('aria-label', /^Response map of Unit 3 over the input plane: values from −?[\d.]+ to −?[\d.]+, red positive and blue negative\./);
  await expect(detail.locator('.units-point')).toHaveCount(16 + 8);
  await expect(detail.locator('.units-digit')).toHaveCount(0);
  await expect(detail.locator('.units-label-row')).toHaveCount(2);
  await expect(detail.locator('.units-label-sum')).toHaveText(/^(Mostly|Mixed:|All \d+ are labelled) class [01]/);
  await expect(detail).toContainText('Click any point to make it the network’s input.');
  // A point in the list becomes the network's input.
  const row = detail.locator('.units-point').first();
  const idx = await row.getAttribute('data-index');
  await expect(row).toHaveText(/^[01]#\d+\(−?\d\.\d\d, −?\d\.\d\d\)−?\d\.\d{3}$/);
  await row.click();
  expect(await raster(page, (r) => r.store.probe?.key)).toBe(`test:${idx}`);
  await expect(detail.locator('.units-point').first()).toHaveAttribute('aria-pressed', 'true');
  await expect(detail.locator('.units-probe')).toContainText(new RegExp(`^Current input, test point #${idx} · Class [01]: [\\d.]+, `));
  // The map: hover for the response, click to pick a place in the plane.
  // Centred, so the sticky control bar does not cover it.
  await plane.evaluate((el) => el.scrollIntoView({ block: 'center' }));
  const box = (await plane.boundingBox())!;
  await page.mouse.move(box.x + box.width * 0.13, box.y + box.height * 0.11);
  await expect(page.locator('#tip')).toContainText('response');
  await page.mouse.click(box.x + box.width * 0.13, box.y + box.height * 0.11);
  await expect.poll(() => raster(page, (r) => r.store.probe?.key)).not.toBe(`test:${idx}`);
  await page.mouse.move(0, 0);
  await expect(page.locator('#units')).not.toContainText(/digit|pixel|synthes/i, { useInnerText: true });
  await shot(page, '11-units-points.png');

  // The output layer is named by class.
  await page.selectOption('#units-layer', '2');
  await expect(page.locator('#units-root .units-card-title')).toHaveText(['Class 0', 'Class 1']);
  await expect(page.locator('#units-root .units-card-foot').first()).toHaveText(/^predicted for [<>]?\d+% of points$/, { timeout: 60_000 });

  // 3-D: the maps show the slice set in 03, and follow it.
  await switchTo(page, 'shells', 'Shells · ');
  await page.locator('#units').scrollIntoViewIfNeeded();
  await page.selectOption('#units-layer', '0');
  await expect(maps).toHaveCount(8, { timeout: 60_000 });
  await expect(page.locator('#units-root .units-synth-note')).toHaveText('Each card maps the unit’s response over the slice x₃ = 0.00; move the slice in 03 Decision boundary.');
  await expect(page.locator('#units-root .units-key')).toContainText('its response over the slice x₃ = 0.00 (right)');
  const before = await maps.first().evaluate((c: HTMLCanvasElement) => c.toDataURL());
  await page.evaluate(() => {
    const s = document.getElementById('bd-slice') as HTMLInputElement;
    s.value = '0.6';
    s.dispatchEvent(new Event('input', { bubbles: true }));
  });
  await expect(page.locator('#units-root .units-synth-note')).toContainText('x₃ = 0.60');
  await expect.poll(() => maps.first().evaluate((c: HTMLCanvasElement) => c.toDataURL())).not.toBe(before);
  await cards(page).first().click();
  await expect(detail.locator('.units-plane canvas')).toHaveAttribute('aria-label', /over the slice x₃ = 0\.60/);
  await expect(detail.locator('.units-point').first()).toHaveText(/\(−?\d\.\d\d, −?\d\.\d\d, −?\d\.\d\d\)/);
  await shot(page, '11-units-points-3d.png');

  // A conv layer cannot read a list of features: the section says what to do.
  await page.evaluate(() => (window as unknown as Page2).raster.actions.setSpec([{ kind: 'conv', filters: 4, kernel: 3, act: 'relu', pool: false }]));
  await expect(page.locator('#units-root .units-grid-note')).toHaveText('Fix the architecture in 01 to see what its units respond to.');
  await expect(page.locator('#units-root .units-layout')).toBeHidden();
  await page.evaluate(() => (window as unknown as Page2).raster.actions.setSpec([{ kind: 'dense', units: 4, act: 'relu' }]));
  await expect(page.locator('#units-root .units-grid-note')).toBeHidden();
  await expect(cards(page)).toHaveCount(4);
});

test('point datasets at phone width: no horizontal overflow, the map and lists fit', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await open(page);
  await switchTo(page, 'moons', 'Moons · ');
  await trainUntil(page, '(window.raster.store.weightsStep ?? 0) > 150');
  await page.locator('#units').scrollIntoViewIfNeeded();
  await expect(page.locator('#units-root .units-card canvas.units-map:visible')).toHaveCount(8, { timeout: 60_000 });
  await cards(page).nth(1).click();
  await expect(page.locator('#units-detail .units-point')).toHaveCount(24);
  await expect(page.locator('#units-detail .units-plane canvas')).toBeVisible();
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow, 'no horizontal page scroll at 390 px').toBeLessThanOrEqual(0);
  const out = await page.locator('#units').evaluate((el) => {
    const r = el.getBoundingClientRect();
    const bad: string[] = [];
    for (const c of el.querySelectorAll<HTMLElement>('*')) {
      const b = c.getBoundingClientRect();
      if (b.width && (b.right > r.right + 1 || b.left < r.left - 1) && !c.closest('.canvas-box')) bad.push(`${c.className} ${Math.round(b.left)}–${Math.round(b.right)}`);
    }
    return bad;
  });
  expect(out, 'nothing pokes out of the section').toEqual([]);
  await shot(page, '11-units-points-phone.png');
});
