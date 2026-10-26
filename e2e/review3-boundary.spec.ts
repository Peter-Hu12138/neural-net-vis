import { expect, test, type Page } from '@playwright/test';

/**
 * Regression tests for the third review of sections 02 (network), 03 (decision boundary) and
 * 07's point plots: main-thread time with large point networks, honest colour scales, the
 * current input on 3-D slices, keyboard and screen-reader access, class shapes, frozen layers
 * and number formats.
 */

let errors: string[] = [];
let pointData = false;
/** 06 Backpropagation (owned elsewhere) may still log this on point data; see boundary.spec.ts. */
const FOREIGN_ON_POINTS = /^Error in a '(probe|weights|model|data)' listener: TypeError: Cannot read properties of undefined \(reading 'toFixed'\)/;

test.beforeEach(async ({ page }) => {
  errors = [];
  pointData = false;
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (m) => {
    if (m.type() !== 'error') return;
    if (pointData && FOREIGN_ON_POINTS.test(m.text())) return;
    errors.push(m.text());
  });
});

test.afterEach(() => {
  expect(errors, 'no console errors or uncaught exceptions').toEqual([]);
});


async function open(page: Page) {
  await page.goto('/');
  await expect(page.locator('#fact-data')).toContainText('train', { timeout: 30_000 });
  await expect(page.locator('#play')).toBeEnabled();
}

async function switchTo(page: Page, id: string) {
  pointData = !['mnist', 'fashion', 'cifar10'].includes(id);
  await page.evaluate((d) => (window as any).raster.actions.setDataset(d), id);
  await page.waitForFunction((d) => {
    const s = (window as any).raster.store;
    return s.dataset === d && !!s.data;
  }, id, { timeout: 60_000 });
}

const dense = (units: number, act = 'relu') => ({ kind: 'dense', units, act });

async function setNet(page: Page, features: string[] | null, spec: object[]) {
  await page.evaluate(
    ({ features, spec }) => {
      const a = (window as any).raster.actions;
      if (features) a.setFeatures(features);
      a.setSpec(spec);
    },
    { features, spec },
  );
}

/** Waits until a canvas has been evaluated with the page's current weights. */
async function settled(page: Page, selector: string) {
  await page.locator(selector).scrollIntoViewIfNeeded();
  await page.waitForFunction((sel) => {
    const c = document.querySelector(sel) as HTMLCanvasElement | null;
    return !!c && c.dataset.weights === String((window as any).raster.store.weightsRev);
  }, selector, { timeout: 30_000 });
  await page.waitForTimeout(150);
}

/** Long tasks (over 50 ms) on the main thread while `fn` runs and for `ms` in all. */
async function longTasks(page: Page, fn: () => Promise<void>, ms: number) {
  await page.evaluate(() => {
    (window as any).__lt = [];
    (window as any).__obs?.disconnect();
    const o = new PerformanceObserver((l) => {
      for (const e of l.getEntries()) (window as any).__lt.push(e.duration);
    });
    o.observe({ type: 'longtask' });
    (window as any).__obs = o;
  });
  const t0 = Date.now();
  await fn();
  await page.waitForTimeout(Math.max(0, ms - (Date.now() - t0)));
  const lt: number[] = await page.evaluate(() => (window as any).__lt.slice());
  return { max: Math.max(0, ...lt), total: lt.reduce((a, b) => a + b, 0), n: lt.length };
}

const F2 = ['x1', 'x2', 'x1^2', 'x2^2', 'x1*x2', 'sin x1', 'sin x2'];
const F3 = ['x1', 'x2', 'x3', 'x1^2', 'x2^2', 'x3^2', 'x1*x2', 'x1*x3', 'x2*x3', 'sin x1', 'sin x2', 'sin x3'];

/** Shows 02 and 03 side by side (1600 px). */
const both = (page: Page) => page.evaluate(() => window.scrollTo(0, document.querySelector('#network')!.getBoundingClientRect().top + window.scrollY - 60));

test('BND-1: 3 × 64 units on 12 features trains without taking over the page', async ({ page }) => {
  await open(page);
  await switchTo(page, 'helix');
  await setNet(page, F3, [dense(64), dense(64), dense(64)]);
  await page.evaluate(() => (window as any).raster.actions.setSpeed('max'));
  await both(page);
  await page.waitForTimeout(1500);
  const lt = await longTasks(
    page,
    async () => {
      await page.click('#play');
      await page.waitForTimeout(5000);
      await page.click('#play');
    },
    5300,
  );
  // Before the fix: 47–52 long tasks, 4.5–6.4 s of 6 s, up to 470 ms each.
  expect(lt.total / 5300, `long tasks ${JSON.stringify(lt)}`).toBeLessThan(0.35);
  expect(lt.max).toBeLessThan(250);
  // The maps were redrawn while training, and the redraw rate follows the measured cost.
  await expect(page.locator('#netview .canvas-box canvas')).toHaveAttribute('data-view', 'points');
  expect(Number(await page.locator('#netview .canvas-box canvas').getAttribute('data-ms'))).toBeGreaterThan(0);
});

test('BND-2: a 3 × 256 point network gets coarser maps instead of freezing the page', async ({ page }) => {
  await open(page);
  await switchTo(page, 'spiral');
  await setNet(page, F2, [dense(256), dense(256), dense(256)]);
  await both(page);
  await settled(page, '#boundary .bd-2d canvas.bd-plane');
  await settled(page, '#netview .canvas-box canvas');
  await page.waitForTimeout(1000);
  // One weights change while paused: before the fix one long task of 0.7–1.5 s.
  const lt = await longTasks(page, () => page.evaluate(() => (window as any).raster.actions.applyWeights((window as any).raster.store.net.getWeights())), 3000);
  expect(lt.max, `long tasks ${JSON.stringify(lt)}`).toBeLessThan(400);
  await settled(page, '#boundary .bd-2d canvas.bd-plane');
  // The maps say they are coarse, and are.
  const res2d = Number(await page.locator('#boundary .bd-2d canvas.bd-plane').getAttribute('data-res'));
  expect(res2d).toBeLessThan(32);
  await expect(page.locator('#boundary .bd-res-note')).toBeVisible();
  await expect(page.locator('#boundary .bd-res-note')).toContainText(`${res2d} × ${res2d} grid`);
  expect(Number(await page.locator('#netview .canvas-box canvas').getAttribute('data-res'))).toBeLessThan(32);
  // The default network still gets the full-size picture.
  await setNet(page, ['x1', 'x2'], [dense(8, 'tanh'), dense(8, 'tanh')]);
  await settled(page, '#boundary .bd-2d canvas.bd-plane');
  await page.waitForTimeout(500);
  await expect(page.locator('#boundary .bd-res-note')).toBeHidden();
});

test('BND-3: feature maps on a 3-D slice show the feature’s real value, not full colour', async ({ page }) => {
  await open(page);
  await switchTo(page, 'blobs3');
  await page.locator('#boundary').getByRole('button', { name: 'x₂', exact: true }).click();
  const net = page.locator('#netview .canvas-box canvas');
  const colourAt = async (pos: number) => {
    await page.locator('#bd-slice').evaluate((el: HTMLInputElement, v) => {
      el.value = String(v);
      el.dispatchEvent(new Event('input', { bubbles: true }));
    }, pos);
    await expect(net).toHaveAttribute('data-slice', `1:${pos.toFixed(2)}`);
    await page.waitForTimeout(100);
    return net.evaluate((c: HTMLCanvasElement) => {
      const tiles = c.dataset.featureTiles!.split(';').map((t) => t.split(' ').map(Number));
      const [x, y, s] = tiles[1]; // the x₂ feature
      const dpr = c.width / c.getBoundingClientRect().width;
      const d = c.getContext('2d')!.getImageData(Math.round((x + s / 2) * dpr), Math.round((y + s / 2) * dpr), 1, 1).data;
      const cs = getComputedStyle(document.documentElement);
      const hex = (n: string) => {
        const h = cs.getPropertyValue(n).trim().slice(1);
        return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16));
      };
      const surf = hex('--surface');
      return Math.hypot(d[0] - surf[0], d[1] - surf[1], d[2] - surf[2]);
    });
  };
  await net.scrollIntoViewIfNeeded();
  // Near zero the map is pale on both sides (before: full ink black at +0.05, full blue at −0.05).
  expect(await colourAt(0.05)).toBeLessThan(60);
  expect(await colourAt(-0.05)).toBeLessThan(60);
  // Near the edge of the cube it is strongly coloured.
  expect(await colourAt(1.2)).toBeGreaterThan(120);
  // The tooltip states the scale.
  await expect(page.locator('#netview .nv-plane-note')).toContainText('full range');
});

test('BND-4: hovering right after a feature change does not run the old network', async ({ page }) => {
  await open(page);
  await switchTo(page, 'circle');
  const plane = page.locator('#boundary .bd-2d canvas.bd-plane');
  await settled(page, '#boundary .bd-2d canvas.bd-plane');
  const box = (await plane.boundingBox())!;
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await expect(page.locator('#tip')).toContainText('Class 0');
  for (const f of [['x1', 'x2', 'x1^2'], ['x1', 'x2']]) {
    await page.evaluate((f) => (window as any).raster.actions.setFeatures(f), f);
    await page.mouse.move(box.x + box.width / 2 + 7, box.y + box.height / 2 + 3);
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await expect(page.locator('#tip')).toContainText('Class 1');
  }
  // afterEach: no "The network takes 2 inputs; 3 features were given".
});

test('BND-5: 02 and 03 agree on whether the current input lies on the slice', async ({ page }) => {
  await open(page);
  await switchTo(page, 'blobs3');
  await page.locator('#boundary').getByRole('button', { name: 'x₂', exact: true }).click();
  await page.locator('#bd-slice').evaluate((el: HTMLInputElement) => {
    el.value = '0.35';
    el.dispatchEvent(new Event('input', { bubbles: true }));
  });
  for (const x2 of [0.57, 0.5, 0.2]) {
    await page.evaluate((x2) => {
      const { store, actions } = (window as any).raster;
      const coords = new Float32Array([0.1, x2, -0.1]);
      actions.setProbe({ x: coords.slice(), label: null, caption: 'test', key: `t:${x2}`, coords });
      void store;
    }, x2);
    await page.waitForTimeout(250);
    const a = await page.locator('#netview .canvas-box canvas').getAttribute('data-probe-on');
    const b = await page.locator('#boundary .bd-3d canvas.bd-plane').getAttribute('data-probe-on');
    expect(a, `x₂ = ${x2}`).toBe(b);
  }
  expect(await page.locator('#boundary .bd-3d canvas.bd-plane').getAttribute('data-probe-on')).toBe('1');
});

test('BND-7: a small network on four features is drawn finer than 100² when paused (thin slivers stay visible)', async ({ page }) => {
  await open(page);
  await switchTo(page, 'spiral');
  await setNet(page, ['x1', 'x2', 'sin x1', 'sin x2'], [dense(8, 'tanh'), dense(8, 'tanh')]);
  await settled(page, '#boundary .bd-2d canvas.bd-plane');
  expect(Number(await page.locator('#boundary .bd-2d canvas.bd-plane').getAttribute('data-res'))).toBeGreaterThanOrEqual(120);
});

test('design-1: every class has its own shape in 03 and 07, in the plots and the keys', async ({ page }) => {
  await open(page);
  await switchTo(page, 'blobs3');
  const shapes = await page.locator('#boundary .bd-key svg').evaluateAll((els) => els.slice(0, 4).map((e) => (e as SVGElement).dataset.shape));
  expect(new Set(shapes).size).toBe(4);
  await expect(page.locator('#boundary .bd-class svg')).toHaveCount(4);
  await expect(page.locator('#boundary .bd-key')).toContainText('Test: outline');
  const counts = await page.locator('#datapanel .dp-counts svg').evaluateAll((els) => els.map((e) => (e as SVGElement).dataset.shape));
  expect(counts).toEqual(shapes);
  await page.locator('#datapanel .dp-plot').first().scrollIntoViewIfNeeded();
  await expect(page.locator('#datapanel .dp-plot-note')).toContainText('own shape');
});

test('design-3: 02’s legend names signed activations on image data', async ({ page }) => {
  await open(page);
  await setNet(page, null, [dense(16, 'tanh')]);
  await page.locator('#netview').scrollIntoViewIfNeeded();
  await expect(page.locator('#netview .legend')).toContainText('Weight or activation');
  await expect(page.locator('#netview .legend')).toContainText('Activation ≥ 0');
  await setNet(page, null, [dense(16, 'relu')]);
  await expect(page.locator('#netview .legend')).not.toContainText('Weight or activation');
});

test('design-7, design-13: 02 marks the input with the cross and shows frozen layers', async ({ page }) => {
  await open(page);
  await switchTo(page, 'circle');
  await settled(page, '#netview .canvas-box canvas');
  await expect(page.locator('#netview .nv-plane-note')).toContainText('red cross marks the current input');
  await expect(page.locator('#netview .nv-frozen-key')).toBeHidden();
  await page.evaluate(() => (window as any).raster.actions.setFrozen(0, true));
  await expect(page.locator('#netview .canvas-box canvas')).toHaveAttribute('data-frozen', 'Dense 1');
  await expect(page.locator('#netview .canvas-box canvas')).toHaveAttribute('aria-label', /Frozen: Dense 1\./);
  await expect(page.locator('#netview .nv-frozen-key')).toBeVisible();
  await switchTo(page, 'mnist');
  await page.evaluate(() => (window as any).raster.actions.setFrozen(0, true));
  await page.locator('#netview').scrollIntoViewIfNeeded();
  await expect(page.locator('#netview .canvas-box canvas')).toHaveAttribute('data-frozen', /Dense 1|Conv 1/);
});

test('design-11: the plots are interactive widgets with key hints; the turn buttons have names', async ({ page }) => {
  await open(page);
  await switchTo(page, 'circle');
  const plane = page.locator('#boundary .bd-2d canvas.bd-plane');
  await expect(plane).toHaveAttribute('role', 'application');
  await expect(plane).toHaveAttribute('aria-roledescription', 'plot');
  const hint = await plane.getAttribute('aria-describedby');
  await expect(page.locator(`#${hint}`)).toContainText('Arrow keys');
  await expect(page.locator('#boundary .bd-key')).toHaveAttribute('role', 'list');
  expect(await page.locator('#boundary .bd-key [role="listitem"]').count()).toBeGreaterThanOrEqual(6);
  await switchTo(page, 'shells');
  const orbit = page.locator('#boundary canvas.bd-orbit');
  await expect(orbit).toHaveAttribute('role', 'application');
  await expect(page.locator(`#${await orbit.getAttribute('aria-describedby')}`)).toContainText('turn the cube');
  await expect(page.getByRole('button', { name: 'Turn left 30°' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Turn right 30°' })).toBeVisible();
});

test('design-9: 03 prints the losses as 05 does, and 07 point coordinates to 3 decimals', async ({ page }) => {
  await open(page);
  await switchTo(page, 'circle');
  await page.click('#play');
  await page.waitForFunction(() => (window as any).raster.store.evals.length >= 2, null, { timeout: 60_000 });
  await page.click('#play');
  await page.waitForFunction(() => !(window as any).raster.store.status?.running);
  await page.waitForTimeout(300);
  const kpi = (sel: string, label: string) => page.locator(`${sel} .kpi`, { hasText: label }).locator('b').textContent();
  expect(await kpi('#boundary', 'Train loss')).toBe(await kpi('#training', 'Train loss'));
  expect(await kpi('#boundary', 'Test loss')).toBe(await kpi('#training', 'Test loss'));
  expect(await kpi('#boundary', 'Train acc.')).toBe(await kpi('#training', 'Train acc.'));
  await page.locator('#datapanel .dp-vec').scrollIntoViewIfNeeded();
  await expect(page.locator('#datapanel .dp-vec td.num').first()).toHaveText(/^−?\d\.\d{3}$/);
});
