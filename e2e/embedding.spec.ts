import { expect, test, type Page } from '@playwright/test';

/**
 * Section 11 Embedding: the PCA map (automatic), t-SNE (on request, animated), hover, click,
 * digit highlight, mistakes, orientation across training steps, manual weight updates, Reset,
 * a dead layer, dark mode and the phone layout. Screenshots land in
 * docs/screenshots/13-embedding-*.png.
 */

const SHOTS = 'docs/screenshots';

type Raster = {
  store: {
    status: { step: number; running: boolean } | null;
    evals: { acc: number }[];
    probe: { key: string; caption: string; x: Float32Array; label: number | null } | null;
    weightsStep: number;
    weightsRev: number;
    net: { blocks: { b: Float32Array; W: Float32Array }[] };
    emit(ev: string): void;
  };
  analysis: { run(channel: string, kind: string, params: unknown): Promise<{ pca?: { components: Float32Array } }> };
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
const progress = (page: Page) => page.locator('#embed-root .synced-progress');
const canvas = (page: Page) => page.locator('#embed-canvas');
const fmtStep = (n: number) => n.toLocaleString('en-US');

/** Waits until a run has finished: status names the step and the shared progress bar is hidden. */
async function settled(page: Page) {
  await expect(status(page)).toContainText('Based on the weights at step', { timeout: 60_000 });
  await expect(progress(page)).toBeHidden({ timeout: 60_000 });
}

/** Height of the status row: one line (28 px) at every width, idle or busy (UX-6). */
const statusHeight = (page: Page) => status(page).evaluate((el) => el.getBoundingClientRect().height);
/** Top of the map, relative to the section: must not move when a run starts or ends. */
const plotTop = (page: Page) => page.locator('#embed-root .embed-plot').evaluate((el) => Math.round(el.getBoundingClientRect().top - el.closest('section')!.getBoundingClientRect().top));

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

/** PC1 and PC2 of the current input, as the preview prints them. */
async function probeCoords(page: Page): Promise<[number, number]> {
  const text = (await page.locator('#embed-preview').textContent()) ?? '';
  const m = text.match(/PC1 (−?[\d.]+(?:e−?\d+)?) · PC2 (−?[\d.]+(?:e−?\d+)?)/);
  if (!m) throw new Error(`no coordinates in "${text}"`);
  return [Number(m[1].replace(/−/g, '-')), Number(m[2].replace(/−/g, '-'))];
}

/** WCAG contrast of each digit colour against the surface, and its CIE76 distance from the accent. */
async function digitColours(page: Page) {
  return page.evaluate(() => {
    const cs = getComputedStyle(document.documentElement);
    const hex = (v: string) => [1, 3, 5].map((i) => parseInt(v.trim().slice(i, i + 2), 16));
    const lin = (c: number) => ((c /= 255) <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
    const lum = (v: string) => {
      const [r, g, b] = hex(v).map(lin);
      return 0.2126 * r + 0.7152 * g + 0.0722 * b;
    };
    // OKLab, ×100 so a distance reads like ΔE.
    const lab = (v: string) => {
      const [r, g, b] = hex(v).map(lin);
      const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
      const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
      const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
      return [
        100 * (0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s),
        100 * (1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s),
        100 * (0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s),
      ];
    };
    const dist = (p: number[], q: number[]) => Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
    const surface = cs.getPropertyValue('--surface');
    const accent = lab(cs.getPropertyValue('--accent'));
    const colours = Array.from({ length: 10 }, (_, d) => cs.getPropertyValue(`--cat-${d}`));
    const labs = colours.map(lab);
    return colours.map((c, d) => {
      const a = lum(c);
      const s = lum(surface);
      const nearest = Math.min(...labs.filter((_, k) => k !== d).map((q) => dist(labs[d], q)));
      return { d, colour: c.trim(), contrast: (Math.max(a, s) + 0.05) / (Math.min(a, s) + 0.05), fromAccent: dist(labs[d], accent), nearest };
    });
  });
}

test('PCA map: automatic, numerals, hover preview, click to probe, highlight and mistakes', async ({ page }) => {
  await open(page);
  await trainUntil(page, '(window.raster.store.evals.length >= 3)');
  await page.locator('#embedding').scrollIntoViewIfNeeded();

  // Small CNN: the default layer is the last hidden one (Dense 3, 32 values).
  await expect(page.locator('#embed-layer')).toHaveValue('2');
  await expect(page.locator('#embed-method-pca')).toHaveAttribute('aria-pressed', 'true');
  await settled(page);
  expect(await statusHeight(page)).toBeLessThanOrEqual(30);
  await expect(canvas(page)).toHaveAttribute('aria-label', /PCA map of 1,000 test digits at Dense 3/);
  await expect(page.locator('#embed-stats')).toContainText('Digits 1,000 · 100 of each');
  await expect(page.locator('#embed-stats')).toContainText('Values per digit 32 at Dense 3');
  await expect(page.locator('#embed-stats')).toContainText(/Variance shown \d+\.\d% \(PC1 \d+\.\d%, PC2 \d+\.\d%\)/);
  await expect(page.locator('#embed-root .embed-hints')).toContainText('PCA finds the two directions');
  const size = await canvas(page).evaluate((c: HTMLCanvasElement) => [c.clientWidth, c.clientHeight]);
  expect(size).toEqual([640, 640]);
  expect((await numeralSpots(page)).length).toBeGreaterThan(60);

  // Hover: tooltip with label and prediction, plus coordinates with a true minus sign (EMB-9).
  const tip = await hoverDigit(page);
  expect(tip).toMatch(/^Test digit #\d+ · label \d · predicted \d\nPC1 −?\d+\.\d\d · PC2 −?\d+\.\d\d$/);
  expect(tip).not.toContain('-');
  const index = Number(tip.match(/#(\d+)/)![1]);
  await expect(page.locator('#embed-root .embed-aside .sub').first()).toHaveText('Hovered digit');
  await expect(page.locator('#embed-preview')).toContainText(`Test digit #${index}`);
  await expect(page.locator('#embed-preview')).toContainText('Click to use it as the input');
  expect(await page.locator('#embed-preview').textContent()).not.toContain('-');

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

  // Drawing on the pad fires a probe event per pointer move: they coalesce into one repaint per
  // frame that blits the cached scatter and redraws the cross, not the 1,000 numerals (UX-8).
  const repaint = await page.evaluate(async () => {
    const c = document.getElementById('embed-canvas') as HTMLCanvasElement;
    const proto = CanvasRenderingContext2D.prototype;
    const fillText = proto.fillText;
    const drawImage = proto.drawImage;
    let fills = 0;
    let blits = 0;
    proto.fillText = function (this: CanvasRenderingContext2D, ...a: Parameters<typeof fillText>) {
      if (this.canvas === c) fills++;
      return fillText.apply(this, a);
    };
    proto.drawImage = function (this: CanvasRenderingContext2D, ...a: unknown[]) {
      if (this.canvas === c) blits++;
      return (drawImage as (...x: unknown[]) => void).apply(this, a);
    } as typeof drawImage;
    const r = (window as unknown as { raster: Raster }).raster;
    (window as unknown as { e2eSavedProbe: unknown }).e2eSavedProbe = r.store.probe;
    const x0 = r.store.probe!.x;
    const t0 = performance.now();
    for (let i = 0; i < 30; i++) {
      r.store.probe = { x: x0.map((v) => v * (1 - i / 60)), label: null, caption: 'Your drawing', key: 'e2e-drawing' };
      r.store.emit('probe');
    }
    const sync = performance.now() - t0;
    await new Promise((res) => requestAnimationFrame(() => requestAnimationFrame(res)));
    proto.fillText = fillText;
    proto.drawImage = drawImage;
    const preview = document.getElementById('embed-preview')!.textContent;
    return { fills, blits, sync, preview };
  });
  console.log(`30 probe events: ${repaint.blits} repaint(s), ${repaint.fills} fillText calls on the map, ${repaint.sync.toFixed(1)} ms of synchronous handlers on the page`);
  expect(repaint.blits).toBe(1);
  expect(repaint.fills).toBeLessThan(10);
  expect(repaint.preview).toContain('Your drawing');
  expect(repaint.preview).toContain('Marked on the map with a red cross');
  // Back to the clicked test digit.
  await raster(page, (r) => {
    const w = window as unknown as { e2eSavedProbe: Raster['store']['probe'] };
    r.store.probe = w.e2eSavedProbe;
    r.store.emit('probe');
  });
  await expect(page.locator('#embed-preview')).toContainText(`Test digit #${index}`);

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
  await expect(page.locator('#tip')).toBeHidden();

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
  const top = await plotTop(page);

  const t0 = Date.now();
  await page.click('#embed-method-tsne');
  await expect(progress(page)).toBeVisible();
  // Frames arrive while it optimises: the iteration count moves on, in a caption over the map.
  const runText = page.locator('#embed-run-text');
  await expect(runText).toHaveText(/^Iteration \d+ \/ 500 · KL \d+\.\d\d$/, { timeout: 30_000 });
  const first = Number((await runText.textContent())!.match(/Iteration (\d+)/)![1]);
  // The status row stays one line and the map does not move while it runs (UX-6).
  expect(await statusHeight(page)).toBeLessThanOrEqual(30);
  expect(await plotTop(page)).toBe(top);
  await shot(page, '13-embedding-tsne-running.png');

  // Hold the pointer still over the moving points: the tooltip, the ring and the preview always
  // name the same digit, the one now under the pointer (EMB-5).
  const box = (await canvas(page).boundingBox())!;
  const spots = [
    [0.5, 0.5],
    [0.35, 0.4],
    [0.62, 0.6],
  ];
  let hovered = 0;
  let inconsistent = 0;
  let samples = 0;
  for (let i = 0; i < 30; i++) {
    const [fx, fy] = spots[Math.floor(i / 10)];
    if (i % 10 === 0) await page.mouse.move(box.x + fx * box.width, box.y + fy * box.height);
    await page.waitForTimeout(120);
    const s = await page.evaluate(() => {
      const tip = document.getElementById('tip')!;
      return {
        run: document.getElementById('embed-run-text')!.textContent ?? '',
        tip: tip.hidden ? null : tip.textContent,
        head: document.querySelector('#embed-root .embed-aside .sub')!.textContent,
        title: document.querySelector('#embed-preview .embed-preview-title')!.textContent,
      };
    });
    if (!s.run.startsWith('Iteration')) break;
    samples++;
    if (s.tip) {
      const digit = s.tip.match(/^Test digit #(\d+)/)?.[1];
      if (!digit || s.head !== 'Hovered digit' || s.title !== `Test digit #${digit}`) inconsistent++;
      else hovered++;
    } else if (s.head === 'Hovered digit') inconsistent++;
  }
  console.log(`t-SNE frames, pointer held still: ${samples} samples, ${hovered} hovering a digit, ${inconsistent} inconsistent`);
  expect(samples).toBeGreaterThan(5);
  expect(inconsistent).toBe(0);
  expect(hovered).toBeGreaterThan(0);
  await page.mouse.move(5, 5);
  await expect(page.locator('#tip')).toBeHidden();

  await expect
    .poll(async () => Number(((await runText.textContent()) ?? '').match(/Iteration (\d+)/)?.[1] ?? 500), { timeout: 30_000 })
    .toBeGreaterThan(first);
  await settled(page);
  await expect(runText).toBeHidden();
  expect(await plotTop(page)).toBe(top);
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
  await expect(progress(page)).toBeHidden();
  await expect(page.locator('#embed-stats')).toContainText('Variance shown');
  await page.click('#embed-method-tsne');
  await expect(progress(page)).toBeHidden();
  await expect(page.locator('#embed-stats')).toContainText('KL divergence');

  // New weights do not restart t-SNE on their own; the status says so and Recompute runs it.
  const step = await raster(page, (r) => r.store.weightsStep);
  await page.click('#step');
  await page.waitForFunction((s) => (window as unknown as { raster: Raster }).raster.store.weightsStep > s, step);
  await expect(status(page)).toContainText('the network is now at step');
  await page.waitForTimeout(1500);
  await expect(progress(page)).toBeHidden();
  await status(page).getByRole('button', { name: 'Recompute' }).click();
  await expect(progress(page)).toBeVisible();
  // The old map stays up with the phase named over it until the first frame arrives.
  await expect(runText).toBeVisible();
  await settled(page);

  // A wide layer is reduced to its 50 main directions first; the note says so.
  await page.selectOption('#embed-layer', '0');
  await settled(page);
  await expect(page.locator('#embed-root .embed-hints')).toContainText('This layer has 1,568 values per digit. t-SNE works on their 50 main directions');

  // A new architecture clears t-SNE until it is asked for again.
  await page.getByRole('button', { name: 'Softmax', exact: true }).click();
  await expect(page.locator('#embed-layer')).toHaveValue('-1');
  await expect(canvas(page)).toHaveAttribute('aria-label', 't-SNE runs only when you ask. Press Recompute.');
  await page.click('#embed-method-pca');
  await settled(page);
  await expect(page.locator('#embed-stats')).toContainText('Values per digit 784 at the input pixels');
});

test('orientation across steps, manual weight updates, Reset and a dead layer', async ({ page }) => {
  test.setTimeout(240_000);
  await open(page);
  await trainUntil(page, '(window.raster.store.evals.length >= 3)');
  await page.locator('#embedding').scrollIntoViewIfNeeded();
  await settled(page);

  // EMB-1: the logits layer, where the cold-start sign rule is ~0 by construction. Step training
  // one batch at a time; after each step the map recomputes and must not mirror. The raw PCA
  // (same job, run directly) shows how often a fresh sign would have flipped.
  await page.selectOption('#embed-layer', '3');
  await settled(page);
  await canvas(page).focus();
  await page.keyboard.press('ArrowRight');
  await page.keyboard.press('Enter');
  await page.keyboard.press('Escape');
  await page.mouse.move(5, 5);
  await expect(page.locator('#embed-preview')).toContainText('Marked on the map with a red cross');
  let prev = await probeCoords(page);
  let prevRaw: Float32Array | null = null;
  let rawFlips = 0;
  let mirrored = 0;
  for (let i = 0; i < 20; i++) {
    const step = await raster(page, (r) => r.store.weightsStep);
    await page.click('#step');
    await page.waitForFunction((s) => (window as unknown as { raster: Raster }).raster.store.weightsStep > s, step);
    const now = await raster(page, (r) => r.store.weightsStep);
    await expect(status(page)).toContainText(`Based on the weights at step ${fmtStep(now)}.`, { timeout: 30_000 });
    await expect(progress(page)).toBeHidden();
    const raw = await raster(page, async (r) => Array.from((await r.analysis.run('e2e-raw-pca', 'embed', { layer: 3, method: 'pca', n: 1000 })).pca!.components));
    if (prevRaw) {
      const d = raw.length / 2;
      for (let k = 0; k < 2; k++) {
        let dot = 0;
        for (let j = 0; j < d; j++) dot += raw[k * d + j] * prevRaw[k * d + j];
        if (dot < -0.5) rawFlips++;
      }
    }
    prevRaw = Float32Array.from(raw);
    const cur = await probeCoords(page);
    for (let k = 0; k < 2; k++) {
      // A mirror: a clearly non-zero coordinate changes sign while keeping its size.
      if (Math.abs(prev[k]) > 1 && Math.sign(cur[k]) !== Math.sign(prev[k]) && Math.abs(Math.abs(cur[k]) - Math.abs(prev[k])) < 0.5 * Math.abs(prev[k])) mirrored++;
    }
    prev = cur;
  }
  console.log(`20 single steps on the logits layer: raw PCA sign flips ${rawFlips}, mirrored maps shown ${mirrored}`);
  expect(mirrored).toBe(0);

  // EMB-2 / UX-1: "Apply to network" in 06 changes the weights but not the step. The map is
  // marked out of date and recomputed when it is back on screen; then it is current again.
  const runs: string[] = [];
  await page.exposeFunction('e2eRun', (channel: string) => runs.push(channel));
  await raster(page, (r) => {
    const a = r.analysis as unknown as { run: (c: string, ...rest: unknown[]) => unknown };
    const run = a.run.bind(a);
    a.run = (c: string, ...rest: unknown[]) => {
      (window as unknown as { e2eRun: (c: string) => void }).e2eRun(c);
      return run(c, ...rest);
    };
  });
  const before = await raster(page, (r) => ({ step: r.store.weightsStep, rev: r.store.weightsRev }));
  await page.locator('#backprop').scrollIntoViewIfNeeded();
  await page.locator('.steps li:not(.phase) button').last().click();
  await page.locator('#bp-eta').selectOption('1');
  await page.getByRole('button', { name: 'Apply to network' }).click();
  await expect(page.locator('#bplab .notice')).toContainText('Applied');
  const after = await raster(page, (r) => ({ step: r.store.weightsStep, rev: r.store.weightsRev }));
  expect(after.step).toBe(before.step);
  expect(after.rev).toBeGreaterThan(before.rev);
  await expect(status(page)).toContainText(`Computed at step ${fmtStep(before.step)}, before the latest manual weight update.`);
  runs.length = 0;
  await page.locator('#embedding').scrollIntoViewIfNeeded();
  await expect.poll(() => runs.filter((c) => c === 'embedding').length, { timeout: 10_000 }).toBeGreaterThan(0);
  await settled(page);
  await expect(status(page)).toContainText(`Based on the weights at step ${fmtStep(before.step)}.`);
  await expect(page.locator('#embed-preview')).toContainText('Marked on the map with a red cross');

  // EMB-3 / F5: Reset re-initialises the weights but keeps the layer the reader picked …
  await page.selectOption('#embed-layer', '0');
  await settled(page);
  await page.click('#reset');
  await expect(page.locator('#embed-layer')).toHaveValue('0');
  await settled(page);
  await expect(page.locator('#embed-stats')).toContainText('Values per digit 1,568 at Conv 1');
  await expect(status(page)).toContainText('Based on the weights at step 0.');
  // … and so does an edit after that layer (Dense 3 gets 64 units). A network whose first layer
  // differs (LeNet-ish) falls back to the default, the last hidden layer.
  await page.locator('#l2-units').selectOption('64');
  await expect(status(page)).toContainText('The architecture changed since this was computed.');
  await expect(page.locator('#embed-layer option[value="2"]')).toContainText('Dense 3 · 64');
  await expect(page.locator('#embed-layer')).toHaveValue('0');
  await page.getByRole('button', { name: 'LeNet-ish', exact: true }).click();
  await expect(page.locator('#embed-layer')).toHaveValue('2');
  await page.getByRole('button', { name: 'Small CNN', exact: true }).click();
  await expect(page.locator('#embed-layer')).toHaveValue('2');
  // The presets sit at the top of the page; the map is computed once it is back on screen.
  await page.locator('#embedding').scrollIntoViewIfNeeded();
  await settled(page);

  // EMB-6 / F4: a layer whose units are all switched off for every digit. The map says why it is
  // empty instead of drawing one blob with "−0.0000" ticks.
  await raster(page, (r) => {
    r.store.net.blocks[2].b.fill(-100);
    r.store.weightsRev++;
    r.store.emit('weights');
  });
  await expect(canvas(page)).toHaveAttribute('aria-label', /Every digit gives the same values at this layer/, { timeout: 30_000 });
  await settled(page);
  await expect(page.locator('#embed-stats')).toContainText('Variance 0 · every digit gives the same values');
  await expect(page.locator('#embed-stats')).not.toContainText('Variance shown');
  await expect(page.locator('#embed-preview')).toContainText('Every digit lands on the same point at this layer');
  // Nothing to hover: no tooltip, no ticks.
  const box = (await canvas(page).boundingBox())!;
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await expect(page.locator('#tip')).toBeHidden();
  await page.mouse.move(5, 5);
  await shot(page, '13-embedding-flat.png');
  // t-SNE gets the same explanation, and does not spend seconds optimising identical points.
  await page.click('#embed-method-tsne');
  await settled(page);
  await expect(canvas(page)).toHaveAttribute('aria-label', /t-SNE of 1,000 test digits at Dense 3: Every digit gives the same values/);
  await expect(page.locator('#embed-root .embed-hints')).not.toContainText('main directions');
});

test('dark mode follows the theme; digit colours keep their contrast in both themes', async ({ page }) => {
  await page.emulateMedia({ colorScheme: 'dark' });
  await open(page);
  const dark = await digitColours(page);
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
  await shot(page, '13-embedding-dark-all.png');
  // Switching the theme redraws the canvas.
  await page.emulateMedia({ colorScheme: 'light' });
  await expect.poll(async () => Math.min(...(await pixel()))).toBeGreaterThan(200);
  const light = await digitColours(page);
  // The numerals are the data marks: every digit colour reads at 4.5:1 or more on the surface,
  // and none is mistaken for the red that marks the input and the mistakes (EMB-4, UX-5, UX-7).
  for (const [theme, cols] of [
    ['light', light],
    ['dark', dark],
  ] as const) {
    console.log(`${theme}: ${cols.map((c) => `${c.d} ${c.colour} ${c.contrast.toFixed(2)}:1 ΔE ${c.fromAccent.toFixed(0)} nearest ${c.nearest.toFixed(1)}`).join(' | ')}`);
    for (const c of cols) {
      expect(c.contrast, `${theme} --cat-${c.d}`).toBeGreaterThanOrEqual(4.5);
      // OKLab ΔE×100: clear of the state red (UX-5), and no two digits nearly the same colour (NEW-1).
      expect(c.fromAccent, `${theme} --cat-${c.d} vs --accent`).toBeGreaterThanOrEqual(12);
      expect(c.nearest, `${theme} --cat-${c.d} vs its nearest digit colour`).toBeGreaterThanOrEqual(9);
    }
  }
});

test('phone width: no horizontal overflow, panel below the map, tap to pick', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await open(page);
  await trainUntil(page, '(window.raster.store.evals.length >= 3)');
  await page.locator('#embedding').scrollIntoViewIfNeeded();
  await settled(page);
  expect(await statusHeight(page)).toBeLessThanOrEqual(30);
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
  // While t-SNE runs the status row keeps one line and the caption fits over the map.
  const top = await plotTop(page);
  await page.click('#embed-method-tsne');
  await expect(page.locator('#embed-run-text')).toHaveText(/^Iteration \d+/, { timeout: 30_000 });
  expect(await statusHeight(page)).toBeLessThanOrEqual(30);
  expect(await plotTop(page)).toBe(top);
  const caption = (await page.locator('#embed-run-text').boundingBox())!;
  expect(caption.x + caption.width).toBeLessThanOrEqual(plot.x + plot.width);
  expect(await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)).toBe(0);
  await settled(page);
});

// ── Other datasets ───────────────────────────────────────────────────────

/** Switches dataset through the page's actions and waits until its data is loaded. */
async function switchTo(page: Page, id: string, fact: string) {
  await page.evaluate((d) => (window as unknown as { raster: { actions: { setDataset(id: string): Promise<void> } } }).raster.actions.setDataset(d), id);
  await expect(page.locator('#fact-data')).toContainText(fact, { timeout: 60_000 });
}

/** Moves the mouse over numerals until the tooltip matches `re`; returns the tooltip text. */
async function hoverSample(page: Page, re: RegExp): Promise<string> {
  const box = (await canvas(page).boundingBox())!;
  for (const s of (await numeralSpots(page)).slice(0, 60)) {
    await page.mouse.move(box.x + s.x, box.y + s.y);
    const tip = page.locator('#tip');
    if ((await tip.isVisible()) && re.test((await tip.textContent()) ?? '')) return (await tip.textContent())!;
  }
  throw new Error('no sample could be hovered');
}

test('CIFAR-10: class names as the legend, colour thumbnails, no digits anywhere', async ({ page }) => {
  await open(page);
  await switchTo(page, 'cifar10', 'CIFAR-10 · 10,000 train');
  await trainUntil(page, '(window.raster.store.weightsStep ?? 0) > 40');
  await page.locator('#embedding').scrollIntoViewIfNeeded();
  await settled(page);
  const names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'];
  await expect(page.locator('#note-embed')).toHaveText('How a layer arranges 1,000 test images, flattened to two dimensions with PCA or t-SNE.');
  await expect(page.locator('#embed-layer option').first()).toHaveText('Input pixels · 3,072');
  await expect(page.locator('#embed-layer option').last()).toHaveText('Output · 10 logits');
  // The chips name each class beside the numeral it is drawn as: they are the legend.
  await expect(page.locator('#embed-root .embed-aside .sub').nth(1)).toHaveText('Classes');
  await expect(page.locator('#embed-root .embed-chip')).toHaveText(names.map((n, k) => `${k} ${n}`));
  await expect(page.locator('#embed-digit-3')).toHaveAttribute('aria-label', 'Highlight cat (drawn as 3)');
  await expect(page.locator('#embed-stats')).toContainText('Images 1,000 · 100 of each');
  await expect(page.locator('#embed-stats')).toContainText('Values per image 32 at Dense 3');
  await expect(canvas(page)).toHaveAttribute('aria-label', /^PCA map of 1,000 test images at Dense 3, each drawn as the numeral of its class/);
  expect((await numeralSpots(page)).length).toBeGreaterThan(20);

  // Hover: the image in colour, its class name and the prediction.
  const tip = await hoverSample(page, /^Test image #\d+/);
  expect(tip).toMatch(/^Test image #\d+ · [a-z]+ · predicted [a-z]+\nPC1 −?\d+\.\d\d · PC2 −?\d+\.\d\d$/);
  await expect(page.locator('#embed-root .embed-aside .sub').first()).toHaveText('Hovered image');
  await expect(page.locator('#embed-preview')).toContainText(/Label: [a-z]+/);
  const colour = await page.locator('#embed-preview canvas').evaluate((c: HTMLCanvasElement) => {
    const d = c.getContext('2d')!.getImageData(0, 0, c.width, c.height).data;
    let n = 0;
    for (let i = 0; i < d.length; i += 4) if (Math.abs(d[i] - d[i + 1]) > 12 || Math.abs(d[i] - d[i + 2]) > 12) n++;
    return { n, width: c.getBoundingClientRect().width };
  });
  expect(colour.n).toBeGreaterThan(500);
  expect(colour.width).toBe(104);
  await page.mouse.move(5, 5);
  await page.click('#embed-digit-3');
  await expect(page.locator('#embed-digit-3')).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator('#embed-preview')).toContainText('Test image #0');
  await expect(page.locator('#embed-preview')).toContainText('Label: cat');
  await shot(page, '13-embedding-cifar.png');
  await page.click('#embed-digit-3');
  await expect(page.locator('#embedding')).not.toContainText(/digit/i, { useInnerText: true });

  // The raw photos: 3,072 values each.
  await page.selectOption('#embed-layer', '-1');
  await settled(page);
  await expect(page.locator('#embed-stats')).toContainText('Values per image 3,072 at the input pixels');
});

test('point data: every test point, class chips, where each point lies in the input', async ({ page }) => {
  await open(page);
  await switchTo(page, 'blobs', 'Three blobs · 300 train');
  await trainUntil(page, '(window.raster.store.weightsStep ?? 0) > 200');
  await page.locator('#embedding').scrollIntoViewIfNeeded();
  await settled(page);
  const n = await raster(page, (r) => (r.store as unknown as { data: { testY: Uint8Array } }).data.testY.length);
  expect(n).toBeLessThan(1000);
  const nText = n.toLocaleString('en-US');
  await expect(page.locator('#note-embed')).toHaveText(`How a layer arranges ${nText} test points, flattened to two dimensions with PCA or t-SNE.`);
  await expect(page.locator('#embed-layer option')).toHaveText(['Input features · 2', 'Dense 1 · 8', 'Dense 2 · 8', 'Output · 3 logits']);
  await expect(page.locator('#embed-root .embed-chip')).toHaveText(['Class 0', 'Class 1', 'Class 2']);
  await expect(page.locator('#embed-stats')).toContainText(new RegExp(`Points ${nText} · (\\d+ of each|every test point)`));
  await expect(page.locator('#embed-stats')).toContainText('Values per point 8 at Dense 2');
  await expect(canvas(page)).toHaveAttribute('aria-label', new RegExp(`^PCA map of ${nText} test points at Dense 2`));
  expect((await numeralSpots(page)).length).toBeGreaterThan(20);

  // The legend's input cross is drawn like the one on the map: halo, ink outline, accent (UX-5).
  const strokes = await page.locator('#embed-root .embed-cross-icon path').evaluateAll((ps) => ps.map((p) => getComputedStyle(p).stroke));
  const tokens = await page.evaluate(() => {
    const probe = document.createElement('i');
    document.body.append(probe);
    const out = ['--surface', '--ink', '--accent'].map((t) => {
      probe.style.color = `var(${t})`;
      return getComputedStyle(probe).color;
    });
    probe.remove();
    return out;
  });
  expect(strokes).toEqual(tokens);

  // Hover: the point's class, the prediction and where it lies in the input plane.
  const tip = await hoverSample(page, /^Test point #\d+/);
  expect(tip).toMatch(/^Test point #\d+ · Class \d · predicted Class \d\nPC1 −?\d+\.\d\d · PC2 −?\d+\.\d\d\nAt x₁ −?\d\.\d\d · x₂ −?\d\.\d\d in the input$/);
  await expect(page.locator('#embed-root .embed-aside .sub').first()).toHaveText('Hovered point');
  await expect(page.locator('#embed-preview canvas')).toHaveAttribute('aria-label', /^Where the hovered point lies in the input: x₁ −?\d\.\d\d, x₂ −?\d\.\d\d$/);
  const index = Number(tip.match(/#(\d+)/)![1]);
  await page.mouse.down();
  await page.mouse.up();
  await expect.poll(() => raster(page, (r) => r.store.probe?.key)).toBe(`test:${index}`);
  await page.mouse.move(5, 5);
  await expect(page.locator('#embed-preview')).toContainText(`Test point #${index}`);
  await expect(page.locator('#embed-preview')).toContainText('Marked on the map with a red cross');
  await shot(page, '13-embedding-points.png');

  // The input layer of 2-D points: PCA only turns the plane, so it shows all of the variance.
  await page.selectOption('#embed-layer', '-1');
  await settled(page);
  await expect(page.locator('#embed-stats')).toContainText('Values per point 2 at the input features');
  await expect(page.locator('#embed-stats')).toContainText('Variance shown 100.0%');

  // t-SNE on every test point, in the dark.
  await page.emulateMedia({ colorScheme: 'dark' });
  await page.selectOption('#embed-layer', '1');
  await settled(page);
  await page.click('#embed-method-tsne');
  await expect(page.locator('#embed-run-text')).toHaveText(/^Iteration \d+ \/ 500 · KL \d+\.\d\d$/, { timeout: 30_000 });
  await settled(page);
  await expect(canvas(page)).toHaveAttribute('aria-label', new RegExp(`^t-SNE map of ${nText} test points at Dense 2`));
  await expect(page.locator('#embed-stats')).toContainText('Perplexity 30');
  await expect(page.locator('#embed-root .embed-hints')).toContainText('t-SNE moves the points around');
  await expect(page.locator('#embedding')).not.toContainText(/digit/i, { useInnerText: true });
  await shot(page, '13-embedding-points-dark.png');
  await page.emulateMedia({ colorScheme: 'light' });
});
