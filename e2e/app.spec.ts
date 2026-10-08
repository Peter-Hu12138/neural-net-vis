import { expect, test, type Page } from '@playwright/test';

/**
 * End-to-end checks for every feature, run in real Chromium against the production build.
 * Screenshots land in docs/screenshots/ and are referenced from docs/TESTING.md.
 */

const SHOTS = 'docs/screenshots';

type Raster = {
  store: {
    status: { step: number; seen: number; epoch: number; epochFraction: number; running: boolean } | null;
    evals: { acc: number; loss: number; epoch: number }[];
    points: unknown[];
    spec: { kind: string; act: string; pool?: boolean }[];
    custom: { y: number }[];
    probe: { key: string; caption: string; x: Float32Array } | null;
    selected: number;
    selectedUnit: number | null;
    mode: string;
    highlight: { block: number; dir: string } | null;
    hyper: { batchSize: number; lr: number; optimizer: string };
    net: { getWeights(): Float32Array[] };
  };
  client: { mode: string };
};

/** Runs `fn` in the page with the app's exposed state (window.raster). `fn` must be self-contained. */
const raster = <T>(page: Page, fn: (r: Raster) => T) => page.evaluate(`(${fn.toString()})(window.raster)`) as Promise<T>;
const status = (page: Page) => page.evaluate(() => (window as unknown as { raster: Raster }).raster.store.status);

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

async function drawSeven(page: Page) {
  const pad = (await page.locator('.pad').boundingBox())!;
  const at = (fx: number, fy: number) => [pad.x + pad.width * fx, pad.y + pad.height * fy] as const;
  await page.mouse.move(...at(0.28, 0.24));
  await page.mouse.down();
  for (let t = 0; t <= 1.0001; t += 0.05) await page.mouse.move(...at(0.28 + 0.44 * t, 0.24));
  for (let t = 0; t <= 1.0001; t += 0.05) await page.mouse.move(...at(0.72 - 0.26 * t, 0.24 + 0.56 * t));
  await page.mouse.up();
}

test('loads MNIST, starts the worker and renders all seven sections', async ({ page }) => {
  await open(page);
  await expect(page).toHaveTitle('Raster Net Lab');
  await expect(page.locator('#fact-engine')).toHaveText('In-browser, Web Worker');
  for (const name of ['Architecture', 'Network', 'Draw', 'Weights', 'Training', 'Backpropagation', 'Data']) {
    await expect(page.getByRole('heading', { level: 2, name })).toBeVisible();
  }
  // The untrained network is evaluated once on the test set straight away (≈10% accuracy).
  await page.waitForFunction(() => (window as unknown as { raster: Raster }).raster.store.evals.length === 1);
  const acc = await raster(page, (r) => r.store.evals[0].acc);
  expect(acc).toBeLessThan(0.3);
  await expect(page.locator('.sample-grid .thumb')).toHaveCount(60);
  await page.screenshot({ path: `${SHOTS}/01-initial.png`, fullPage: true });
});

test('play, pause and resume training; curves and confusion matrix fill in', async ({ page }) => {
  await open(page);
  await page.click('#play');
  await expect(page.locator('#play')).toHaveAttribute('aria-pressed', 'true');
  await page.waitForFunction(() => ((window as unknown as { raster: Raster }).raster.store.status?.step ?? 0) > 40);
  await expect(page.locator('.bar .stat').nth(2).locator('b')).not.toHaveText('—'); // digits per second
  await page.click('#play');
  await expect(page.locator('#play')).toHaveAttribute('aria-pressed', 'false');
  const paused = (await status(page))!.step;
  await page.waitForTimeout(600);
  expect((await status(page))!.step, 'stays put while paused').toBe(paused);

  // Resume until the 0.2-epoch evaluation lands.
  await trainUntil(page, '(window.raster.store.evals.length >= 2)');
  expect((await status(page))!.step).toBeGreaterThan(paused);
  const evals = await raster(page, (r) => r.store.evals.map((e) => e.acc));
  expect(evals.at(-1)!).toBeGreaterThan(0.7);
  await expect(page.locator('.kpi').nth(3).locator('b')).toContainText('%');
  await expect(page.locator('#curves')).toContainText('Most confused');
  await page.locator('#training').screenshot({ path: `${SHOTS}/02-training-curves.png` });
});

test('Step trains one batch; +1 Epoch stops exactly at the epoch boundary', async ({ page }) => {
  await open(page);
  await page.getByRole('button', { name: 'Softmax', exact: true }).click();
  await page.selectOption('#batch', '128');
  await page.click('#step');
  await page.waitForFunction(() => (window as unknown as { raster: Raster }).raster.store.status?.step === 1);
  expect((await status(page))!.seen).toBe(128);
  await page.click('#epoch');
  await page.waitForFunction(() => {
    const s = (window as unknown as { raster: Raster }).raster.store.status;
    return !!s && s.epoch === 1 && !s.running;
  }, null, { timeout: 60_000 });
  const s = (await status(page))!;
  expect(s.step).toBe(Math.ceil((20_000 - 128) / 128) + 1);
  // 20,000 is not a multiple of 128, so the last batch straddles the boundary by < 1 batch.
  expect(s.epochFraction).toBeGreaterThanOrEqual(1);
  expect(s.epochFraction).toBeLessThan(1 + 128 / 20_000);
  await expect(page.locator('.bar .stat').first().locator('b')).toHaveText('1.00');
});

test('architecture builder: add, edit, remove layers and presets', async ({ page }) => {
  await open(page);
  await expect(page.locator('.layer[data-block]')).toHaveCount(4); // 3 hidden + output
  await page.getByRole('button', { name: '+ Conv layer' }).click();
  let spec = await raster(page, (r) => r.store.spec.map((l) => l.kind));
  expect(spec).toEqual(['conv', 'conv', 'conv', 'dense']);
  await expect(page.locator('.layer').nth(3)).toContainText('3×3×8'); // 7×7 pooled → 3×3

  await page.selectOption('#l0-act', 'tanh');
  await page.locator('#l0-pool').uncheck();
  await expect(page.locator('.layer').nth(1)).toContainText('28×28×8');
  spec = await raster(page, (r) => r.store.spec.map((l) => `${l.kind}:${l.act}:${l.pool ?? ''}`));
  expect(spec[0]).toBe('conv:tanh:false');

  await page.getByRole('button', { name: 'Remove Conv 1' }).click();
  await page.getByRole('button', { name: '+ Dense layer' }).click();
  await page.selectOption('#l3-units', '64');
  await page.selectOption('#l3-act', 'sigmoid');
  spec = await raster(page, (r) => r.store.spec.map((l) => `${l.kind}:${l.act}`));
  expect(spec).toEqual(['conv:relu', 'conv:relu', 'dense:relu', 'dense:sigmoid']);
  const params = await page.locator('.builder-foot b').textContent();
  expect(Number(params!.replace(/,/g, ''))).toBeGreaterThan(1000);
  await page.locator('#architecture').screenshot({ path: `${SHOTS}/03-builder.png` });

  for (const preset of ['Softmax', 'MLP', 'LeNet-ish', 'Small CNN']) {
    await page.getByRole('button', { name: preset, exact: true }).click();
    await page.waitForFunction(() => (window as unknown as { raster: Raster }).raster.store.evals.length >= 1);
  }
  await expect(page.locator('.layer[data-block]')).toHaveCount(4);
});

test('weights view switches between heatmap, Hinton, numbers and histogram', async ({ page }) => {
  await open(page);
  await trainUntil(page, '(window.raster.store.status?.step ?? 0) > 30');
  const canvas = page.locator('#inspector canvas');
  const shots: Buffer[] = [];
  for (const [label, file] of [['Heatmap', 'heat'], ['Hinton', 'hinton'], ['Numbers', 'numbers'], ['Histogram', 'hist']] as const) {
    await page.locator('#inspector').getByRole('button', { name: label }).click();
    await expect(page.locator('#inspector').getByRole('button', { name: label })).toHaveAttribute('aria-pressed', 'true');
    await page.waitForTimeout(150);
    expect(await raster(page, (r) => r.store.mode)).toBe(file);
    shots.push(await canvas.screenshot());
    await page.locator('#weights').screenshot({ path: `${SHOTS}/04-weights-${file}.png` });
  }
  for (let i = 1; i < shots.length; i++) expect(shots[i].equals(shots[i - 1])).toBe(false);

  // Picking another layer from the menu and a unit from the network diagram.
  await page.locator('#inspector').getByRole('button', { name: 'Heatmap' }).click();
  await page.selectOption('#insp-layer', '2');
  expect(await raster(page, (r) => r.store.selected)).toBe(2);
  await page.locator('#weights').screenshot({ path: `${SHOTS}/04-weights-dense-templates.png` });
  // Hover across the Conv 1 column of the network diagram until a feature map's tooltip shows,
  // then click it: the inspector jumps to that layer and filter.
  await page.locator('#netview .canvas-box canvas').scrollIntoViewIfNeeded();
  const net = (await page.locator('#netview .canvas-box canvas').boundingBox())!;
  const colW = (net.width - 56) / 5;
  let hit = '';
  for (let y = net.y + 70; y < net.y + net.height && !hit; y += 9) {
    for (let x = net.x + colW * 1.1; x < net.x + colW * 1.9 && !hit; x += 9) {
      await page.mouse.move(x, y);
      const tip = page.locator('#tip');
      if (await tip.isVisible()) {
        const text = (await tip.textContent()) ?? '';
        if (text.startsWith('Conv 1 · filter')) {
          hit = text;
          await page.mouse.click(x, y);
        }
      }
    }
  }
  expect(hit).toMatch(/^Conv 1 · filter \d+\n14×14 map/);
  expect(await raster(page, (r) => r.store.selected)).toBe(0);
  expect(await raster(page, (r) => r.store.selectedUnit)).not.toBeNull();
  await expect(page.locator('#insp-layer')).toHaveValue('0');
});

test('drawing pad classifies live while drawing and can add the drawing to training', async ({ page }) => {
  await open(page);
  await page.getByRole('button', { name: 'MLP', exact: true }).click();
  await page.selectOption('#lr', '0.003');
  await trainUntil(page, '(window.raster.store.evals.at(-1)?.acc ?? 0) > 0.9 && window.raster.store.evals.length >= 3');

  await expect(page.locator('.pred-digit')).toHaveText('?');
  const pad = (await page.locator('.pad').boundingBox())!;
  await page.mouse.move(pad.x + pad.width * 0.3, pad.y + pad.height * 0.24);
  await page.mouse.down();
  await page.mouse.move(pad.x + pad.width * 0.7, pad.y + pad.height * 0.24, { steps: 8 });
  // Mid-stroke: a prediction already shows (live classification before the pen lifts).
  await expect(page.locator('.pred-digit')).not.toHaveText('?');
  const mid = await page.locator('.pred-conf').textContent();
  await page.mouse.move(pad.x + pad.width * 0.46, pad.y + pad.height * 0.8, { steps: 12 });
  await page.mouse.up();
  await expect(page.locator('.pred-digit')).toHaveText('7');
  expect(await page.locator('.pred-conf').textContent()).not.toBe(mid);
  expect(await raster(page, (r) => r.store.probe?.key)).toBe('draw');
  await expect(page.locator('.probe-caption')).toHaveText('Your drawing');
  await page.locator('#draw').screenshot({ path: `${SHOTS}/05-draw.png` });

  await page.locator('.add-train .chip', { hasText: '7' }).click();
  await expect(page.locator('.add-train .hint')).toContainText('Added as a 7');
  expect(await raster(page, (r) => r.store.custom.map((c) => c.y))).toEqual([7]);
  await expect(page.locator('.custom-summary')).toContainText('Your image is mixed into training');
  await expect(page.locator('.pred-digit')).toHaveText('?'); // pad cleared for the next one
  await page.click('.pad-tools .btn');
});

test('uploaded images are converted to 28×28, classified and can join the training set', async ({ page }) => {
  await open(page);
  // Two synthetic "photos": dark ink on light paper, and light ink on a dark background.
  const files = await page.evaluate(() => {
    const make = (bg: string, ink: string, text: string) => {
      const c = document.createElement('canvas');
      c.width = 300;
      c.height = 220;
      const ctx = c.getContext('2d')!;
      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, 300, 220);
      ctx.fillStyle = ink;
      ctx.font = 'bold 150px Arial';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(text, 170, 120);
      return c.toDataURL('image/png').split(',')[1];
    };
    return [make('#f4efe6', '#1a1a1a', '3'), make('#101010', '#f0f0f0', '1')];
  });
  await page.setInputFiles('#upload-input', [
    { name: 'paper-three.png', mimeType: 'image/png', buffer: Buffer.from(files[0], 'base64') },
    { name: 'chalk-one.png', mimeType: 'image/png', buffer: Buffer.from(files[1], 'base64') },
  ]);
  await expect(page.locator('.upload')).toHaveCount(2);
  await expect(page.locator('.upload').first()).toContainText('Predicts');
  // Both polarities end up as white-on-black ink, centred like MNIST.
  const centred = await raster(page, (r) => {
    const x = r.store.probe!.x;
    let m = 0, cx = 0, cy = 0;
    x.forEach((v, i) => { m += v; cx += v * (i % 28); cy += v * Math.floor(i / 28); });
    return { mass: m, cx: cx / m, cy: cy / m, corner: x[0] };
  });
  expect(centred.mass).toBeGreaterThan(20);
  expect(Math.abs(centred.cx - 13.5)).toBeLessThan(1.5);
  expect(Math.abs(centred.cy - 13.5)).toBeLessThan(1.5);
  expect(centred.corner).toBe(0);

  const row = page.locator('.upload', { hasText: 'paper-three.png' });
  await expect(row.getByRole('button', { name: 'Train on it' })).toBeDisabled();
  await row.locator('.chip', { hasText: '3' }).click();
  await row.getByRole('button', { name: 'Train on it' }).click();
  await expect(row.locator('.tag')).toHaveText('In training set');
  expect(await raster(page, (r) => r.store.custom.map((c) => c.y))).toEqual([3]);
  await page.locator('#data').screenshot({ path: `${SHOTS}/06-data-uploads.png` });

  // Removing it from the "In the training set" grid updates the upload row too.
  await page.locator('#datapanel .sample-grid').nth(1).locator('.thumb').click();
  expect(await raster(page, (r) => r.store.custom.length)).toBe(0);
  await expect(row.getByRole('button', { name: 'Train on it' })).toBeEnabled();
});

test('backprop walkthrough steps through every stage and applies the update', async ({ page }) => {
  await open(page);
  await trainUntil(page, '(window.raster.store.status?.step ?? 0) > 25');
  await page.locator('#backprop').scrollIntoViewIfNeeded();
  // Paused, so the walkthrough uses the network's latest weights rather than the step-0 snapshot.
  const step = (await status(page))!.step;
  await expect(page.locator('#bplab')).toContainText(`Using the network's weights at step ${step}`);
  const titles = await page.locator('.steps li:not(.phase) button').allTextContents();
  // Small CNN: input, 2×(conv, act, pool), dense, act, logits, softmax, loss, δ, output grads,
  // dense act′, dense grads, 2×(pool′, act′, conv grads), update.
  expect(titles).toHaveLength(23);
  expect(titles[0]).toContain('Input image');
  expect(titles.at(-1)).toContain('Gradient descent step');

  const seen: string[] = [];
  for (let i = 0; i < titles.length; i++) {
    const title = await page.locator('.bp-title h3').textContent();
    seen.push(title!);
    await expect(page.locator('.bp-main .formula').first()).toBeVisible();
    const hl = await raster(page, (r) => r.store.highlight);
    expect(hl, 'network diagram highlights the active layer').not.toBeNull();
    if (title === 'Conv 1: convolution') await page.locator('#backprop').screenshot({ path: `${SHOTS}/07-backprop-conv-forward.png` });
    if (title === 'Softmax') await page.locator('#backprop').screenshot({ path: `${SHOTS}/07-backprop-softmax.png` });
    if (title === 'Gradient at the logits') await page.locator('#backprop').screenshot({ path: `${SHOTS}/07-backprop-output-gradient.png` });
    if (title === 'Conv 2: back through max-pool') await page.locator('#backprop').screenshot({ path: `${SHOTS}/07-backprop-pool-backward.png` });
    if (title === 'Conv 1: gradients') await page.locator('#backprop').screenshot({ path: `${SHOTS}/07-backprop-conv-gradients.png` });
    if (i < titles.length - 1) await page.getByRole('button', { name: 'Next →' }).click();
  }
  expect(seen.map((s) => s.trim())).toEqual(titles.map((t) => t.replace(/^\d+/, '').trim()));
  await expect(page.locator('.dir')).toHaveText('Update');

  const before = Number(await page.locator('.loss-compare .kpi').first().locator('b').textContent());
  const after = Number(await page.locator('.loss-compare .kpi').nth(1).locator('b').textContent());
  expect(after).toBeLessThan(before);
  const w0 = await raster(page, (r) => Array.from(r.store.net.getWeights().at(-2)!.slice(0, 50)));
  await page.getByRole('button', { name: 'Apply to network' }).click();
  await expect(page.locator('.notice')).toContainText('Applied');
  const w1 = await raster(page, (r) => Array.from(r.store.net.getWeights().at(-2)!.slice(0, 50)));
  expect(w1).not.toEqual(w0);
  await page.locator('#backprop').screenshot({ path: `${SHOTS}/07-backprop-update.png` });

  // Arrow keys navigate; numbers mode prints small tensors as values.
  await page.locator('.steps li:not(.phase) button').first().click();
  await page.keyboard.press('ArrowRight');
  await expect(page.locator('.bp-title h3')).toHaveText('Conv 1: convolution');
  await page.locator('#inspector').getByRole('button', { name: 'Numbers' }).click();
  await page.locator('.steps li:not(.phase) button', { hasText: 'Softmax' }).click();
  await page.locator('#backprop').screenshot({ path: `${SHOTS}/07-backprop-softmax-numbers.png` });
});

test('backprop asks for a target when the input is an unlabeled drawing', async ({ page }) => {
  await open(page);
  await drawSeven(page);
  await expect(page.locator('#bplab')).toContainText('Your drawing has no label');
  await page.locator('#bplab .chips .chip', { hasText: '7' }).click();
  await page.locator('.steps li:not(.phase) button', { hasText: 'Cross-entropy loss' }).click();
  await expect(page.locator('.bp-main')).toContainText('−log p[7]');
});

test('dark colour scheme uses the dark palette', async ({ page }) => {
  await page.emulateMedia({ colorScheme: 'dark' });
  await open(page);
  const bg = await page.evaluate(() => getComputedStyle(document.body).backgroundColor);
  expect(bg).toBe('rgb(14, 14, 13)');
  await page.click('#step');
  await page.waitForTimeout(400);
  await page.screenshot({ path: `${SHOTS}/08-dark.png`, fullPage: false });
  // An explicit light theme on the root element wins over the OS setting.
  await page.evaluate(() => document.documentElement.setAttribute('data-theme', 'light'));
  expect(await page.evaluate(() => getComputedStyle(document.body).backgroundColor)).toBe('rgb(245, 245, 242)');
});

test('phone width: single column, no sideways scrolling, drawing still works', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await open(page);
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  expect(overflow).toBeLessThanOrEqual(0);
  await page.locator('#draw').scrollIntoViewIfNeeded();
  await drawSeven(page);
  await expect(page.locator('.pred-digit')).not.toHaveText('?');
  await page.screenshot({ path: `${SHOTS}/09-phone.png`, fullPage: false });
});

test('falls back to main-thread training when Web Workers are unavailable', async ({ page }) => {
  await page.addInitScript(() => {
    (window as unknown as { Worker: unknown }).Worker = function () {
      throw new Error('blocked');
    };
  });
  await open(page);
  await expect(page.locator('#fact-engine')).toHaveText('In-browser, main thread');
  await page.click('#step');
  await page.waitForFunction(() => (window as unknown as { raster: Raster }).raster.store.status?.step === 1);
});
