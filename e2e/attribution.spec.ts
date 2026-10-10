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

/** Parses "Σ IG = 8.39; z(x) − z(blank) = 8.43 (off by 0.0389, 0.3% of Σ|IG|)". */
async function completeness(page: Page) {
  const text = (await page.locator('#attr-check').textContent())!;
  const m = text.match(/^Σ IG = (\S+); z\(x\) − z\(blank\) = (\S+) \((.+)\)$/);
  expect(m, text).not.toBeNull();
  const num = (s: string) => Number(s.replace(/−/g, '-'));
  return { sum: num(m![1]), expected: num(m![2]), gap: m![3] };
}

/**
 * Height of the status row and the top of what sits below it; with `recompute`, measured right
 * after clicking Recompute, in the same task, while the job runs.
 */
const statusLayout = (page: Page, recompute = false) =>
  page.evaluate((click) => {
    const r = document.getElementById('attr-root')!;
    if (click) r.querySelector<HTMLButtonElement>('.synced-status button')!.click();
    return {
      statusHeight: r.querySelector('.synced-status')!.getBoundingClientRect().height,
      layoutTop: r.querySelector('.attr-layout')!.getBoundingClientRect().top,
      state: r.dataset.state,
    };
  }, recompute);

/** Counts the attribution jobs started from now on; returns a reader of the count. */
async function countRuns(page: Page): Promise<() => Promise<number>> {
  await page.evaluate(() => {
    const w = window as unknown as { raster: { analysis: { run: (c: string, ...rest: unknown[]) => unknown } }; e2eAttrRuns?: number; e2eAttrWrapped?: boolean };
    w.e2eAttrRuns = 0;
    if (w.e2eAttrWrapped) return;
    w.e2eAttrWrapped = true;
    const a = w.raster.analysis;
    const run = a.run.bind(a);
    a.run = (c: string, ...rest: unknown[]) => {
      if (c === 'attribution') w.e2eAttrRuns = (w.e2eAttrRuns ?? 0) + 1;
      return run(c, ...rest);
    };
  });
  return () => page.evaluate(() => (window as unknown as { e2eAttrRuns: number }).e2eAttrRuns);
}

/** Switches dataset through the page's actions and waits until its data is loaded. */
async function switchTo(page: Page, id: string, fact: string) {
  await page.evaluate((d) => (window as unknown as { raster: { actions: { setDataset(id: string): Promise<void> } } }).raster.actions.setDataset(d), id);
  await expect(page.locator('#fact-data')).toContainText(fact, { timeout: 60_000 });
}

/** How many pixels of a canvas show colour (channels that differ), not grey. */
const colourPixels = (page: Page, id: string) =>
  page.locator(`#${id}`).evaluate((c: HTMLCanvasElement) => {
    const d = c.getContext('2d')!.getImageData(0, 0, c.width, c.height).data;
    let n = 0;
    for (let i = 0; i < d.length; i += 4) if (Math.abs(d[i] - d[i + 1]) > 12 || Math.abs(d[i] - d[i + 2]) > 12) n++;
    return n;
  });

const num = (s: string) => Number(s.replace(/−/g, '-').replace(/^\+/, ''));

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
  // Occlusion explains the logit like the other three maps, so it does not saturate: erasing part
  // of a confidently classified 7 moves its score by a clear amount.
  await expect(page.locator('[data-panel="occlusion"] .hint')).toContainText(`Erases a 6×6 patch at a time: red where that lowers the score for ${pred}`);
  const occMax = Number((await page.locator('[data-panel="occlusion"] .attr-scale span').last().textContent())!.replace('+', ''));
  expect(occMax).toBeGreaterThan(0.1);
  // Small CNN over a blank background: tied max-pool windows, and the hint says what that means.
  await expect(page.locator('[data-panel="saliency"] .hint')).toContainText('Where the input leaves tied max-pool windows (mostly the blank background)');
  await expect(page.locator('[data-panel="saliency"] .hint')).toContainText('the map shows the average slope.');

  // Completeness: the integrated gradients add up to z(x) − z(blank), within 5%, and the line says
  // how far off they are and relative to what.
  const c = await completeness(page);
  expect(Math.abs(c.expected)).toBeGreaterThan(0.5);
  expect(Math.abs(c.sum - c.expected) / Math.abs(c.expected)).toBeLessThan(0.05);
  expect(c.gap).toMatch(/^off by [\d.e−]+, (<0\.1|\d+(\.\d)?)% of Σ\|IG\|$/);
  await expect(page.locator('#attr-check')).toHaveAttribute('title', /Σ\|IG\| = [\d.]+\. The gap is measured against Σ\|IG\|/);

  // One type scale with the other sections: 16/800 panel titles, 13 px hints, tags ≥ 9 px.
  const type = await page.evaluate(() => {
    const f = (sel: string) => getComputedStyle(document.querySelector(sel)!);
    return { title: [f('#attr-root .attr-title').fontSize, f('#attr-root .attr-title').fontWeight], hint: f('#attr-root .attr-text .hint').fontSize, tag: f('#attr-root .attr-tag').fontSize };
  });
  expect(type).toEqual({ title: ['16px', '800'], hint: '13px', tag: '9px' });

  // The shared status row is one 28 px line, busy or not, and nothing below it moves.
  const idle = await statusLayout(page);
  const busy = await statusLayout(page, true);
  expect(idle.statusHeight).toBe(28);
  expect(busy).toEqual({ state: 'busy', statusHeight: 28, layoutTop: idle.layoutTop });
  await expect(page.locator('#attr-root .synced-progress')).toBeVisible();
  await resultFor(page, /^test:0\|\d$/);
  await expect(page.locator('#attr-root .synced-progress')).toBeHidden();

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
  // The map is in logit units; the tooltip also gives the (saturating) probability change.
  await expect(page.locator('#tip')).toContainText(new RegExp(`Mean drop in z${'₀₁₂₃₄₅₆₇₈₉'[pred]} [+−]?[\\d.]`));
  await expect(page.locator('#tip')).toContainText(`Mean drop in p(${pred})`);
  await page.mouse.move(ig.x + cell * 14.5, ig.y + cell * 12.5);
  await shot(page, '12-attribution-light.png');
  await page.mouse.move(0, 0);
  await expect(page.locator('#tip')).toBeHidden();

  // Pick another digit: the chip and the line switch at once, while the maps, still for the old
  // digit, fade and say what they are waiting for (the fade itself waits 200 ms, against flicker).
  const other = (pred + 3) % 10;
  const during = await page.evaluate((d) => {
    document.getElementById(`attr-target-${d}`)!.click();
    const grid = document.querySelector('#attr-root .attr-panels')!;
    return {
      stale: grid.classList.contains('is-stale'),
      waiting: grid.classList.contains('is-waiting'),
      busy: grid.getAttribute('aria-busy'),
      label: document.getElementById('attr-wait')!.textContent,
      key: document.getElementById('attr-root')!.dataset.resultKey,
      pressed: document.getElementById(`attr-target-${d}`)!.getAttribute('aria-pressed'),
    };
  }, other);
  expect(during).toEqual({ stale: true, waiting: true, busy: 'true', label: `Updating for digit ${other}…`, key: `test:0|${pred}`, pressed: 'true' });
  await resultFor(page, new RegExp(`^test:0\\|${other}$`));
  await expect(page.locator('#attr-root .attr-panels')).not.toHaveClass(/is-stale|is-waiting/);
  await expect(page.locator('#attr-root .attr-panels')).toHaveAttribute('aria-busy', 'false');
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

  // ...and re-selecting the same input (its thumb in 02 Network, already pressed) keeps it, without
  // running the same job again (NEW-2).
  const runs = await countRuns(page);
  await page.locator('#network button.thumb[title="Test digit #0 (a 7)"]').click();
  await page.locator('#attribution').scrollIntoViewIfNeeded();
  await page.waitForTimeout(800);
  await resultFor(page, new RegExp(`^test:0\\|${other}$`));
  await expect(page.locator(`#attr-target-${other}`)).toHaveAttribute('aria-pressed', 'true');
  await expect(root(page)).toHaveAttribute('data-target', String(other));
  expect(await runs(), 'no new attribution job for the input already shown').toBe(0);
  // Recompute still runs it.
  await page.locator('#attr-root .synced-status button').click();
  await expect.poll(runs).toBe(1);
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
  await expect(page.locator('#attr-check')).toHaveText('Σ IG = 0; z(x) − z(blank) = 0 (a match)');
  for (const key of ['gradInput', 'integrated', 'occlusion']) {
    await expect(page.locator(`[data-panel="${key}"] .attr-scale`)).toHaveText('00');
  }
});

test('a new network clears the maps at once; an invalid one waits, keeping the pick', async ({ page }) => {
  await open(page);
  await page.locator('#attribution').scrollIntoViewIfNeeded();
  await resultFor(page, /^test:0\|\d$/);
  await page.click('#attr-target-3');
  await resultFor(page, /^test:0\|3$/);
  expect(await colours(page, 'attr-map-saliency')).toBeGreaterThan(8);

  /** Clicks a preset in 01 without scrolling there, and reads section 10 in the same task. */
  const preset = (name: string) =>
    page.evaluate((n) => {
      const b = Array.from(document.querySelectorAll<HTMLButtonElement>('.presets button')).find((e) => e.textContent === n)!;
      b.click();
      const r = document.getElementById('attr-root')!;
      const c = document.getElementById('attr-map-saliency') as HTMLCanvasElement;
      const d = c.getContext('2d')!.getImageData(0, 0, c.width, c.height).data;
      const set = new Set<number>();
      for (let i = 0; i < d.length; i += 4) set.add((d[i] << 16) | (d[i + 1] << 8) | d[i + 2]);
      return { key: r.dataset.resultKey, colours: set.size, pred: document.getElementById('attr-pred')!.textContent, notice: document.getElementById('attr-notice')!.hidden };
    }, name);

  // MLP: the Small CNN's maps go at once; the line already shows the MLP's prediction; the pick stays.
  const mlp = await preset('MLP');
  expect(mlp.key).toBe('');
  expect(mlp.colours).toBeLessThanOrEqual(3);
  expect(mlp.pred).toMatch(/^Predicted \d at [\d.]+%; explaining digit 3( \([\d.]+%\))?$/);
  await expect(page.locator('#attr-map-saliency')).toHaveAttribute('aria-label', 'Saliency: not computed yet');
  await resultFor(page, /^test:0\|3$/);
  await expect(page.locator('#attr-root .synced-status')).toContainText('Based on the weights at step 0.');
  expect(await colours(page, 'attr-map-saliency')).toBeGreaterThan(8);
  // A plain dense ReLU network on this digit has no kinks to explain.
  await expect(page.locator('[data-panel="saliency"] .hint')).toHaveText('How strongly the score for 3 reacts to a small change in each pixel, in either direction.');

  // An invalid architecture: no maps, a notice, and a pick waits for a fix instead of claiming a result.
  await raster(page, (r) => {
    (r.store as unknown as { spec: unknown }).spec = [
      { kind: 'dense', units: 32, act: 'relu' },
      { kind: 'conv', filters: 4, kernel: 3, act: 'relu', pool: false },
    ];
    r.store.emit('model');
  });
  await expect(page.locator('#attr-notice')).toHaveText('Fix the architecture in 01 to see attributions.');
  await expect(root(page)).toHaveAttribute('data-result-key', '');
  expect(await colours(page, 'attr-map-integrated')).toBeLessThanOrEqual(3);
  await expect(page.locator('#attr-pred')).toHaveText('Will explain digit 3 once the architecture is fixed.');
  await page.click('#attr-target-5');
  await expect(page.locator('#attr-pred')).toHaveText('Will explain digit 5 once the architecture is fixed.');
  await expect(page.locator('#attr-target-5')).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator('#attr-target-3')).toHaveAttribute('aria-pressed', 'false');
  await expect(root(page)).toHaveAttribute('data-state', 'idle');

  // A valid preset: the notice goes at once, and the waiting pick is explained.
  const fixed = await preset('Small CNN');
  expect(fixed.notice).toBe(true);
  expect(fixed.pred).toMatch(/explaining digit 5/);
  await resultFor(page, /^test:0\|5$/);
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
  // The status row stays one 28 px line on a phone too, also while computing.
  const idle = await statusLayout(page);
  const busy = await statusLayout(page, true);
  expect(idle.statusHeight).toBe(28);
  expect(busy).toEqual({ state: 'busy', statusHeight: 28, layoutTop: idle.layoutTop });
  await resultFor(page, /^test:0\|\d$/);
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

// ── Other datasets ───────────────────────────────────────────────────────

/** Waits until the maps on screen belong to the network's current weights. */
async function upToDate(page: Page) {
  const step = await raster(page, (r) => r.store.weightsStep);
  await expect(page.locator('#attr-root .synced-status')).toContainText(`Based on the weights at step ${step.toLocaleString('en-US')}.`, { timeout: 30_000 });
  await expect(root(page)).toHaveAttribute('data-state', 'done');
}

test('CIFAR-10: maps over the photo with the channels added up, mean-colour occlusion, class names', async ({ page }) => {
  await open(page);
  await switchTo(page, 'cifar10', 'CIFAR-10 · 10,000 train');
  await trainUntil(page, '(window.raster.store.weightsStep ?? 0) > 60');
  await page.locator('#attribution').scrollIntoViewIfNeeded();
  await resultFor(page, /^test:0\|\d$/);
  await upToDate(page);
  const names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'];
  const pred = Number(await root(page).getAttribute('data-pred'));
  await expect(page.locator('#attr-target-label')).toHaveText('Explain class');
  await expect(page.locator('#attr-root .attr-chips button')).toHaveText(names);
  await expect(page.locator(`#attr-target-${pred}`)).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator('#attr-target-3')).toHaveAttribute('aria-label', /^Explain cat \(.*true label\)$/);
  await expect(page.locator('#attr-pred')).toHaveText(new RegExp(`^Predicted ${names[pred]} at \\d+\\.\\d%; explaining ${names[pred]}$`));
  await expect(page.locator('#attr-root .attr-input-caption')).toHaveText('Input: Test image #0 · cat');

  // Five 32×32 maps at a whole number of screen pixels per image pixel; the input in colour.
  for (const key of ['input', 'saliency', 'gradInput', 'integrated', 'occlusion']) {
    expect(await colours(page, `attr-map-${key}`), key).toBeGreaterThan(8);
    const w = await page.locator(`#attr-map-${key}`).evaluate((c) => c.getBoundingClientRect().width);
    expect(w % 32, key).toBe(0);
  }
  expect(await colourPixels(page, 'attr-map-input')).toBeGreaterThan(200);
  await expect(page.locator('[data-panel="saliency"] .attr-title')).toHaveText('Saliency Σ|∂z/∂x|');
  await expect(page.locator('[data-panel="input"] .hint')).toContainText('The 32×32 photo the network sees');
  await expect(page.locator('[data-panel="gradInput"] .hint')).toContainText('added over its channels');
  // Occlusion paints with the test images' average colour, shown as a swatch: a muted mid tone.
  await expect(page.locator('[data-panel="occlusion"] .hint')).toContainText('Paints a 6×6 patch at a time with the average colour');
  const fill = await page.locator('[data-panel="occlusion"] .attr-fill').evaluate((e) => getComputedStyle(e).backgroundColor);
  for (const v of fill.match(/\d+/g)!.map(Number).slice(0, 3)) {
    expect(v).toBeGreaterThan(70);
    expect(v).toBeLessThan(180);
  }
  await expect(page.locator('#attr-root .attr-notes')).toContainText('the average colour of the test images, not black');

  // Completeness over all three channels, measured from a black image.
  const text = (await page.locator('#attr-check').textContent())!;
  const m = text.match(/^Σ IG = (\S+); z\(x\) − z\(black\) = (\S+) \((?:off by \S+, ([<\d.]+)% of .+|a match)\)$/);
  expect(m, text).not.toBeNull();
  if (m![3]) expect(Number(m![3].replace('<', ''))).toBeLessThan(5);

  // Hover: the pixel's three values, and the attribution split over its channels.
  const ig = (await page.locator('#attr-map-integrated').boundingBox())!;
  const cell = ig.width / 32;
  await page.mouse.move(ig.x + cell * 14.5, ig.y + cell * 12.5);
  await expect(page.locator('#tip')).toContainText('Row 12, column 14');
  await expect(page.locator('#tip')).toContainText(/Pixel R \d\.\d\d · G \d\.\d\d · B \d\.\d\d/);
  await expect(page.locator('#tip')).toContainText(/Integrated gradient \S+ \(R \S+ · G \S+ · B \S+\)/);
  await shot(page, '12-attribution-cifar.png');
  await page.mouse.move(0, 0);

  // Another class: the line and the hints name it, and nothing in the section says "digit".
  const other = (pred + 4) % 10;
  await page.click(`#attr-target-${other}`);
  await resultFor(page, new RegExp(`^test:0\\|${other}$`));
  await expect(page.locator('#attr-pred')).toHaveText(new RegExp(`explaining ${names[other]} \\([\\d.]+%\\)$`));
  await expect(page.locator('[data-panel="integrated"] .hint')).toContainText(`change in the score for ${names[other]}`);
  await expect(page.locator('#attribution')).not.toContainText(/digit/i, { useInnerText: true });
});

test('point data: a bar per feature, the steepest way up on the plane, click or keys to move the point', async ({ page }) => {
  await open(page);
  await switchTo(page, 'circle', 'Circle · 300 train');
  await trainUntil(page, '(window.raster.store.weightsStep ?? 0) > 300');
  await page.locator('#attribution').scrollIntoViewIfNeeded();
  await resultFor(page, /^test:0\|\d$/);
  await upToDate(page);
  const pred = Number(await root(page).getAttribute('data-pred'));
  await expect(page.locator('#attr-points')).toBeVisible();
  await expect(page.locator('[data-panel="input"]')).toBeHidden();
  await expect(page.locator('#attr-root .attr-chips button')).toHaveText(['Class 0', 'Class 1']);
  await expect(page.locator('#attr-root .attr-chips .attr-swatch')).toHaveCount(2);
  await expect(page.locator('#attr-pred')).toHaveText(new RegExp(`^Predicted Class ${pred} at \\d+\\.\\d%; explaining Class ${pred}$`));
  await expect(page.locator('#attr-root .attr-input-caption')).toHaveText(/^Input: Test point #0 · Class \d$/);

  // One row per input feature (the raw coordinates by default), two bars in each.
  const rows = page.locator('#attr-features tbody tr');
  await expect(rows).toHaveCount(2);
  await expect(rows.locator('.attr-feat-name')).toHaveText(['x₁', 'x₂']);
  await expect(rows.locator('.attr-bar')).toHaveCount(4);
  const cells = await rows.evaluateAll((trs) => trs.map((tr) => Array.from(tr.querySelectorAll('.attr-bar-num'), (e) => e.textContent!)));
  // Completeness: the integrated-gradients column adds up to z(x) − z(origin).
  const check = (await page.locator('#attr-check').textContent())!;
  const m = check.match(/^Σ IG = (\S+); z\(x\) − z\(origin\) = (\S+) /);
  expect(m, check).not.toBeNull();
  const igSum = cells.reduce((s, c) => s + num(c[1]), 0);
  expect(Math.abs(igSum - num(m![2]))).toBeLessThan(0.02 * Math.max(1, Math.abs(num(m![2]))));
  await expect(page.locator('#attr-sentence')).toHaveText(new RegExp(`^From the origin to this point, the score for Class ${pred} (rises|falls) by [\\d.e−]+\\. `));
  await expect(page.locator('#attr-root .attr-feat-hints')).toContainText('Reading a bar:');

  // The arrow is the gradient with respect to the coordinates: checked against central differences
  // of the page's own network (here the features are the coordinates themselves).
  const caption = (await page.locator('#attr-plane-caption').textContent())!;
  const g = caption.match(/∂z\/∂x₁ (\S+) · ∂z\/∂x₂ (\S+)/);
  expect(g, caption).not.toBeNull();
  const fd = await page.evaluate((t) => {
    const r = (window as unknown as { raster: { store: { probe: { coords: Float32Array }; net: { forward(x: Float32Array): Float32Array; blocks: { z: Float32Array }[] } } } }).raster;
    const c = r.store.probe.coords;
    const z = (x1: number, x2: number) => {
      r.store.net.forward(Float32Array.from([x1, x2]));
      return r.store.net.blocks[r.store.net.blocks.length - 1].z[t];
    };
    const h = 1e-3;
    return [(z(c[0] + h, c[1]) - z(c[0] - h, c[1])) / (2 * h), (z(c[0], c[1] + h) - z(c[0], c[1] - h)) / (2 * h)];
  }, pred);
  expect(Math.abs(num(g![1]) - fd[0])).toBeLessThan(0.01 * Math.abs(fd[0]) + 0.01);
  expect(Math.abs(num(g![2]) - fd[1])).toBeLessThan(0.01 * Math.abs(fd[1]) + 0.01);
  expect(await colours(page, 'attr-plane')).toBeGreaterThan(8);
  await expect(page.locator('#attr-plane')).toHaveAttribute('aria-label', new RegExp(`arrow toward a higher score for Class ${pred}`));
  await shot(page, '12-attribution-points.png');

  // Re-selecting the same point runs nothing (NEW-2); picking the other class does.
  const runs = await countRuns(page);
  await raster(page, (r) => {
    const a = (r as unknown as { actions: { setProbe(p: unknown): void; testProbe(d: unknown, i: number): unknown }; store: { data: unknown } }).actions;
    a.setProbe(a.testProbe((r as unknown as { store: { data: unknown } }).store.data, 0));
  });
  await page.waitForTimeout(600);
  expect(await runs()).toBe(0);
  const other = 1 - pred;
  await page.click(`#attr-target-${other}`);
  await resultFor(page, new RegExp(`^test:0\\|${other}$`));
  expect(await runs()).toBe(1);
  await expect(page.locator('#attr-sentence')).toContainText(`the score for Class ${other}`);

  // A click on the plane makes that spot the input, and the pick goes back to the prediction.
  const plane = (await page.locator('#attr-plane').boundingBox())!;
  await page.mouse.click(plane.x + plane.width * 0.62, plane.y + plane.height * 0.32);
  await resultFor(page, /^pt:[-\d.]+,[-\d.]+\|\d$/);
  await expect(page.locator('#attr-root .attr-input-caption')).toHaveText(/^Input: Point \(−?\d\.\d\d, −?\d\.\d\d\)$/);
  await expect(root(page)).toHaveAttribute('data-target', (await root(page).getAttribute('data-pred'))!);
  // The arrow keys on the map nudge it by 0.05.
  const x1 = await raster(page, (r) => (r.store.probe as unknown as { coords: Float32Array }).coords[0]);
  await page.locator('#attr-plane').focus();
  await page.keyboard.press('ArrowRight');
  await expect.poll(() => raster(page, (r) => (r.store.probe as unknown as { coords: Float32Array }).coords[0])).toBeCloseTo(x1 + 0.05, 3);
  await resultFor(page, /^pt:/);

  // More input features: one row each, labelled as in the dataset strip.
  await page.evaluate(() => (window as unknown as { raster: { actions: { setFeatures(f: string[]): void } } }).raster.actions.setFeatures(['x1', 'x2', 'x1^2', 'x2^2', 'x1*x2']));
  await resultFor(page, /^test:0\|\d$/);
  await expect(rows.locator('.attr-feat-name')).toHaveText(['x₁', 'x₂', 'x₁²', 'x₂²', 'x₁x₂']);
  await expect(rows.locator('.attr-bar')).toHaveCount(10);
  await expect(page.locator('#attr-check')).toHaveText(/^Σ IG = \S+; z\(x\) − z\(origin\) = \S+ \(/);
});

test('3-D points at phone width and in the dark: the slice through the point, three slopes', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await open(page);
  await switchTo(page, 'helix', 'Double helix · 300 train');
  await trainUntil(page, '(window.raster.store.weightsStep ?? 0) > 150');
  await page.locator('#attribution').scrollIntoViewIfNeeded();
  await resultFor(page, /^test:0\|\d$/);
  await upToDate(page);
  await expect(page.locator('#attr-points .attr-pts-plane .attr-title')).toHaveText(/^The point, on the slice x₃ = −?\d\.\d\d$/);
  await expect(page.locator('#attr-features tbody tr')).toHaveCount(3);
  await expect(page.locator('#attr-plane-caption')).toContainText('∂z/∂x₃');
  await expect(page.locator('#attr-points .attr-pts-plane .hint')).toContainText('The arrow is the x₁–x₂ part of the way to move the point');
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow, 'no horizontal page scroll at 390 px').toBeLessThanOrEqual(0);
  const right = await page.locator('#attribution').evaluate((sec) => Math.max(...Array.from(sec.querySelectorAll('*'), (e) => e.getBoundingClientRect().right)));
  expect(right).toBeLessThanOrEqual(390 - 16 + 0.5);
  // The status row stays one line here too.
  const idle = await statusLayout(page);
  expect(idle.statusHeight).toBe(28);
  await shot(page, '12-attribution-points-phone.png');

  // Dark theme: the plane redraws from the dark tokens.
  await page.setViewportSize({ width: 1600, height: 1000 });
  await page.emulateMedia({ colorScheme: 'dark' });
  await page.waitForFunction(() => {
    const c = document.getElementById('attr-plane') as HTMLCanvasElement;
    const k = c.width / c.getBoundingClientRect().width;
    const d = c.getContext('2d')!.getImageData(Math.round(40 * k), Math.round(26 * k), 1, 1).data;
    return d[0] + d[1] + d[2] < 300;
  });
  await page.locator('#attribution').scrollIntoViewIfNeeded();
  await shot(page, '12-attribution-points-3d-dark.png');
  await page.emulateMedia({ colorScheme: 'light' });
});
