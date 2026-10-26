import { expect, test, type Page } from '@playwright/test';

/**
 * Review 3 (sections 04, 06 and 08–11): every sentence and percentage is literally true. One test
 * per dataset setup; each checks the findings that show on it.
 */

type Win = {
  raster: {
    store: {
      status: { running: boolean } | null;
      weightsStep: number;
      features: string[];
      net: { forward(x: Float32Array): Float32Array; blocks: { z: Float32Array }[]; getWeights(): Float32Array[]; inputGradient(b: number, seed: Float32Array): Float32Array };
      data: { testY: Uint8Array; points?: { testCoords: Float32Array; dims: number } };
      probe: { x: Float32Array; coords?: Float32Array } | null;
    };
    actions: {
      setDataset(id: string): Promise<void>;
      setFeatures(f: string[]): void;
      setSpec(spec: unknown[]): void;
      applyWeights(w: Float32Array[]): void;
      setProbe(p: unknown): void;
      testProbe(d: unknown, i: number): unknown;
      setPointsConfig(p: Record<string, number>): void;
      loadModel(json: unknown): Promise<void>;
    };
  };
};

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

async function switchTo(page: Page, id: string, fact: string) {
  await page.evaluate((d) => (window as unknown as Win).raster.actions.setDataset(d), id);
  await expect(page.locator('#fact-data')).toContainText(fact, { timeout: 60_000 });
}

async function trainUntil(page: Page, pred: string) {
  await page.click('#play');
  await page.waitForFunction(pred, null, { timeout: 90_000 });
  await page.click('#play');
  await page.waitForFunction(() => !(window as unknown as Win).raster.store.status?.running);
}

const attr = (page: Page) => page.locator('#attr-root');
const num = (s: string) => Number(s.replace(/−/g, '-'));

/** Waits for a finished attribution whose key (probe key | target) matches. */
async function attributed(page: Page, key: RegExp) {
  await expect(attr(page)).toHaveAttribute('data-result-key', key, { timeout: 60_000 });
  await expect(attr(page)).toHaveAttribute('data-state', 'done', { timeout: 60_000 });
}

/** Makes (x₁, x₂) the input, with raw-coordinate features. */
async function pointProbe(page: Page, x1: number, x2: number) {
  await page.evaluate(
    ([a, b]) => {
      const r = (window as unknown as Win).raster;
      const c = Float32Array.from([a, b]);
      r.actions.setProbe({ x: c.slice(), label: null, caption: `Point (${a}, ${b})`, key: `pt:${a.toFixed(3)},${b.toFixed(3)}`, coords: c });
    },
    [x1, x2],
  );
}

test('point data: 06 names classes once, 09 says "a unit", 10 integrates along the point’s own path and explains a falling score', async ({ page }) => {
  await open(page);
  await switchTo(page, 'circle', 'Circle · 300 train');
  await trainUntil(page, '(window.raster.store.weightsStep ?? 0) > 300');

  // 06, output step: "Class 0", never "class Class 0" (ANA-8).
  await page.locator('#backprop').scrollIntoViewIfNeeded();
  await page.locator('.steps li:not(.phase) button', { hasText: 'Output layer: logits' }).click();
  const main = page.locator('.bp-main');
  await expect(main).toContainText(/The highlighted row of W produces the logit for Class \d\./);
  await expect(main).toContainText(/Logit for Class \d:/);
  await expect(main).not.toContainText('class Class');

  // 09: "Pick a unit" (ANA-8, design-5); the note speaks of the plane (design-14a).
  await page.locator('#units').scrollIntoViewIfNeeded();
  await expect(page.locator('#units-detail .hint')).toContainText(/^Pick a unit to see the 16 test points/);
  await expect(page.locator('#note-units')).toHaveText('What each unit responds to: the test points that excite it most, and its response across the plane.');

  // 10 on points near the centre: when the predicted class scores even higher at the origin, the
  // lead sentence says so instead of reading as an argument against the prediction (design-10).
  await page.locator('#attribution').scrollIntoViewIfNeeded();
  let explained = 0;
  for (const [a, b] of [
    [0.35, 0.2],
    [0.45, -0.1],
    [-0.3, 0.4],
    [0.1, -0.5],
  ]) {
    await pointProbe(page, a, b);
    await attributed(page, new RegExp(`^pt:${a.toFixed(3)},${b.toFixed(3)}\\|\\d$`));
    const text = (await page.locator('#attr-sentence').textContent())!;
    const falls = /score for (Class \d) falls by/.exec(text);
    const predicted = (await attr(page).getAttribute('data-pred')) === (await attr(page).getAttribute('data-target'));
    if (falls && predicted) {
      explained++;
      expect(text).toContain(`${falls[1]} is still the prediction here; the origin just scores even higher for it.`);
      expect(text).not.toContain('pushes away from');
    } else expect(text).not.toContain('still the prediction');
  }
  expect(explained, 'some point near the centre scores lower than the origin').toBeGreaterThan(0);

  // Every feature: integrated gradients now follow the point along the line from the origin, and
  // the bars match that line integral, computed here from the page's own network (ANA-2).
  await page.evaluate(() => (window as unknown as Win).raster.actions.setFeatures(['x1', 'x2', 'x1^2', 'x2^2', 'x1*x2', 'sin x1', 'sin x2']));
  // The chosen point stays the input through a feature change (review finding FLOW-7).
  await attributed(page, /^pt:[-\d.]+,[-\d.]+\|\d$/);
  const target = Number(await attr(page).getAttribute('data-target'));
  const shown = await page.locator('#attr-features tbody tr').evaluateAll((trs) => trs.map((tr) => tr.querySelectorAll('.attr-bar-num')[1].textContent!));
  const ref = await page.evaluate((t) => {
    const r = (window as unknown as Win).raster;
    const c = r.store.probe!.coords!;
    const feat = (a: number) => {
      const x = a * c[0];
      const y = a * c[1];
      return Float32Array.from([x, y, x * x, y * y, x * y, Math.sin(Math.PI * x), Math.sin(Math.PI * y)]);
    };
    const net = r.store.net;
    const seed = new Float32Array(net.blocks[net.blocks.length - 1].z.length);
    seed[t] = 1;
    const N = 2000;
    const out = new Array(7).fill(0);
    for (let k = 0; k < N; k++) {
      const a = (k + 0.5) / N;
      net.forward(feat(a));
      const g = net.inputGradient(net.blocks.length - 1, seed);
      const up = feat(a + 1e-3);
      const down = feat(a - 1e-3);
      for (let f = 0; f < 7; f++) out[f] += (g[f] * (up[f] - down[f])) / 2e-3 / N;
    }
    return out;
  }, target);
  const scale = Math.max(...ref.map(Math.abs));
  shown.forEach((s, f) => expect(Math.abs(num(s) - ref[f]), `feature ${f}: ${s} vs ${ref[f]}`).toBeLessThan(0.01 * scale + 0.006));
  const hints = page.locator('#attr-root .attr-feat-hints');
  await expect(hints).toContainText('Integrated gradients move the point along the straight line from the origin to here');
  await expect(hints).toContainText(/halfway there, x₁² is at 25% of its value here\.|halfway there, x₂² is at 25% of its value here\./);
  await expect(hints).toContainText(/Reading a bar: (.+) [+−][\d.]+ is \1’s share of the change in the score for Class \d as the point moves in a straight line from the origin to here, with \1 going from 0 to −?\d\.\d\d\./);
  await expect(hints).not.toContainText('growing');
  // The slope column keeps its trailing zeros (design-9): three significant digits each.
  const slopes = await page.locator('#attr-features tbody td.attr-col-slope').allInnerTexts();
  for (const s of slopes) expect(s).toMatch(/^([+−](\d\.\d\d|\d\d\.\d|\d{3,}|0\.0*[1-9]\d\d)|0|[+−]\d\.\de−\d)$/);
});

test('a ReLU unit dead on every test point but firing on the map says "never fires on a test point" (ANA-3)', async ({ page }) => {
  await open(page);
  await switchTo(page, 'circle', 'Circle · 300 train');
  await page.evaluate(() => (window as unknown as Win).raster.actions.setSpec([{ kind: 'dense', units: 4, act: 'relu' }]));
  // Unit 1 fires only where x₁ + x₂ goes beyond every test point: the top-right corner of the map.
  await page.evaluate(() => {
    const r = (window as unknown as Win).raster;
    const tc = r.store.data.points!.testCoords;
    let edge = -Infinity;
    for (let i = 0; i < r.store.data.testY.length; i++) edge = Math.max(edge, tc[2 * i] + tc[2 * i + 1]);
    const w = r.store.net.getWeights().map((a) => a.slice());
    w[0][0] = 1;
    w[0][1] = 1;
    w[1][0] = -(edge + 0.02);
    r.actions.applyWeights(w);
  });
  await page.locator('#units').scrollIntoViewIfNeeded();
  const first = page.locator('#units-root .units-card').first();
  await expect(first.locator('.units-card-foot')).toHaveText('never fires on a test point', { timeout: 60_000 });
  await expect(page.locator('#units-root .units-key')).toContainText(/never fires? on a test point \(dead on the data\)/);
  await expect(page.locator('#units-root .units-key')).toContainText(/(Its map|maps?) still shows? where (it|they) would fire: (a region|regions) no test point reaches\./);
  await first.click();
  await expect(page.locator('#units-detail .units-stats')).toContainText('never fires on a test point');
  await expect(page.locator('#units-detail')).toContainText('Its map shows where it would fire: a region no test point reaches.');
});

test('11 counts the points its map holds; 09 names the slice on 3-D data (ANA-6, design-14a)', async ({ page }) => {
  await open(page);
  await switchTo(page, 'blobs', 'Three blobs · 300 train');
  // 2,000 points, 10% for training: 1,800 test points, 333 of each class on the map.
  await page.evaluate(() => (window as unknown as Win).raster.actions.setPointsConfig({ count: 2000, trainRatio: 0.1 }));
  await expect(page.locator('#fact-data')).toContainText('1,800 test');
  await expect(page.locator('#note-embed')).toHaveText('How a layer arranges 999 test points, flattened to two dimensions with PCA or t-SNE.');
  await page.locator('#embedding').scrollIntoViewIfNeeded();
  await expect(page.locator('#embed-stats')).toContainText('Points 999 · 333 of each', { timeout: 60_000 });
  // PCA read-outs: plain decimals at one precision for the map, never an exponent (design-9).
  const preview = page.locator('#embed-root').getByText(/^PC1 /).first();
  if (await preview.count()) await expect(preview).toHaveText(/^PC1 −?\d+\.?\d* · PC2 −?\d+\.?\d*$/);

  await switchTo(page, 'helix', 'Double helix · ');
  await expect(page.locator('#note-units')).toHaveText('What each unit responds to: the test points that excite it most, and its response over a slice through the 3-D space.');
});

test('Fashion-MNIST: 10 speaks of brightness, not ink (design-6)', async ({ page }) => {
  await open(page);
  await switchTo(page, 'fashion', 'Fashion-MNIST · ');
  await page.locator('#attribution').scrollIntoViewIfNeeded();
  await attributed(page, /^test:0\|\d$/);
  await expect(page.locator('[data-panel="input"] .hint')).toHaveText('The 28×28 image the network sees, from 0 (the black background) to 1 (the brightest part of the picture).');
  await expect(page.locator('[data-panel="gradInput"] .hint')).toContainText('The gradient times each pixel’s brightness: red pixels raise the score for');
  await expect(page.locator('#attribution')).not.toContainText(/\bink\b/, { useInnerText: true });
});

test('CIFAR-10 photo #1600: the completeness gap is measured against the score change (ANA-4)', async ({ page }) => {
  await open(page);
  await page.evaluate(async () => {
    const json = await (await fetch('models/cifar10-cnn.json')).json();
    await (window as unknown as Win).raster.actions.loadModel(json);
  });
  await expect(page.locator('#fact-data')).toContainText('CIFAR-10 · ', { timeout: 60_000 });
  await page.waitForFunction(() => (window as unknown as Win).raster.store.data?.testY.length > 1600);
  await page.locator('#attribution').scrollIntoViewIfNeeded();
  await page.evaluate(() => {
    const r = (window as unknown as Win).raster;
    r.actions.setProbe(r.actions.testProbe(r.store.data, 1600));
  });
  await attributed(page, /^test:1600\|\d$/);
  // At 32 steps this sum was off by 0.106, 53% of a 0.201 change, and the line read "0.1% of Σ|IG|".
  const text = (await page.locator('#attr-check').textContent())!;
  const m = text.match(/^Σ IG = (\S+); z\(x\) − z\(black\) = (\S+) \(off by (\S+?)[,:] ([<\d.]+)% of the score change(, [<\d.]+% of Σ\|IG\|)?\)$/);
  expect(m, text).not.toBeNull();
  const [sum, change, off, pct] = [num(m![1]), num(m![2]), num(m![3]), Number(m![4].replace('<', ''))];
  // The percentage is the miss over the change, as the numbers beside it say.
  expect(Math.abs(pct - (100 * Math.abs(sum - change)) / Math.abs(change))).toBeLessThan(Math.max(1, pct * 0.1));
  expect(Math.abs(off - Math.abs(sum - change))).toBeLessThan(0.01 * Math.abs(change) + 0.002);
  // More steps bring it within reach of the change.
  await expect(page.locator('[data-panel="integrated"] .hint')).toContainText(/\((96|288) steps\)/);
  expect(pct).toBeLessThan(20);
  const large = (await page.locator('#attr-check').getAttribute('data-large')) === 'true';
  expect(large).toBe(pct > 5);
  if (large) await expect(page.locator('[data-panel="integrated"] .hint')).toContainText('the line below shows how far off they are');
});
