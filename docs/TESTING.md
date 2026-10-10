# Testing and verification

Raster is checked at two levels:

| Level | Tool | Command | What it covers |
| --- | --- | --- | --- |
| Unit | Vitest (Node) | `npm test` | Engine maths, trainer and analysis protocols, statistics, every analysis job, preprocessing, bundled data, formatting |
| Browser | Playwright + Chromium | `npm run test:e2e` | Every user-facing feature, run against the production build, with screenshots |
| Adversarial review | Independent reviewer agents | see section 4 | The analysis views (08–11): maths, behaviour, integration, design and code, each finding backed by a reproduction |

Both suites were last run on the current commit: **176/176 unit tests** and **31/31 browser tests** pass.

```bash
npm install
npm test            # unit tests, ~20 s
npm run test:e2e    # builds, serves dist/ on :4173, runs Chromium, ~7 min
npm run typecheck   # strict TypeScript over src/, tests/ and e2e/
```

On a machine without Playwright's browsers, run `npx playwright install chromium` once first.

---

## 1. Unit tests (`tests/`)

### 1.1 Engine gradients: `tests/nn.test.ts`

Each layer's analytic backward pass is compared with central finite differences
(`(L(w+ε) − L(w−ε)) / 2ε`, ε = 0.01) on sampled weights, biases and input pixels.
The check reports the relative error ‖g_num − g_ana‖ / ‖g_num‖ + ‖g_ana‖.

| Network checked | Tolerance |
| --- | --- |
| conv 3×3 + pool → conv 5×5 → dense, for tanh, sigmoid and linear | < 2 % |
| conv + pool → dense, for ReLU and Leaky ReLU (kinks allowed) | < 5 % |
| dense → dense (tanh, sigmoid) | < 2 % |

The same file also checks:

- Shape propagation through pooling and flattening (`28×28×1 → 14×14×8 → 7×7×16 → 32 → 10`) and parameter counts.
- A too-small map is flagged when pooled.
- SGD, Momentum and Adam each halve the loss on a toy problem.
- The same seed builds identical weights; a different seed does not.
- Weights round-trip between two network copies (this is what the worker and the page do).
- Softmax outputs sum to 1 and the parameter count matches the builder's.

### 1.2 Trainer protocol: `tests/trainer.test.ts`

The `Trainer` class (the Web Worker's body) runs directly in Node on a synthetic,
learnable 10-class dataset:

| Test | Asserts |
| --- | --- |
| Initial evaluation | Loading data immediately evaluates the untrained model at step 0 on every test sample. |
| Step | One `step` message trains exactly one batch (`seen = batch size`) and emits new weights. |
| +1 epoch | Training stops exactly at the epoch boundary, evaluates at 0.0/0.2/0.4/0.6/0.8/1.0, and a few epochs reach > 95 % on the toy task. |
| Play / pause | Training continues until paused and makes no progress afterwards. |
| Custom samples | User images are mixed in and seen `CUSTOM_REPEAT` (10) times per epoch. |
| Hyper-parameters | Switching optimizer or learning rate keeps the weights and uses the new batch size. |
| New model | A new architecture resets counters and history and pauses training. |
| Weights from the page | Weights pushed by the backprop view are used and re-evaluated (all-zero weights give loss ln 10). |

### 1.3 Preprocessing: `tests/preprocess.test.ts`

- Dark ink on white paper is inverted into MNIST polarity (white on black).
- Light ink on a dark background is kept as is.
- An off-centre tall stroke is scaled so its longest side is 20 px and centred by mass in 28×28.
- A blank image returns `null`.
- Shrinking uses area averaging.

### 1.4 Bundled data: `tests/data.test.ts`

Decodes the PNG sprites with a minimal reader, then checks:

- 22,000 labels stored as a line of digits, every digit well represented.
- The training split starts `5 0 4 1 9 2 1 3 1 4` and the test split starts `7 2 1 0 4 1 4 9 5 9`, matching canonical MNIST.
- Sprites are 8-bit greyscale with no gamma or ICC chunks, so browsers decode exact pixel values.
- Sampled digits have ink in the middle and almost none on the outer ring.

### 1.5 Formatting: `tests/format.test.ts`

Matrix-cell labels never exceed five characters; KPI and counter formats are stable.


### 1.6 Analysis scheduler: `tests/analyzer.test.ts` (4 tests)

Jobs wait for the test set, run to completion, and report results. A newer request on a channel
replaces the running one while other channels continue. Cancel stops a job without a result.
Errors and unknown job names are reported.

### 1.7 Statistics and Q–Q plots: `tests/stats.test.ts` (32 tests)

- `normalQuantile` matches reference values, for example Φ⁻¹(0.975) = 1.959963984540054. It is antisymmetric and monotone, and it round-trips with `normalCdf` (relative error under 1e-13 in both tails).
- Type-7 quantiles, Blom plotting positions and the qqline match their definitions.
- A large normal sample lies on its line with slope ≈ σ. A uniform sample bends into an S with a lower PPCC. Heavy tails bend away. Empty, single and constant data give no NaN.
- Thinned Q–Q plots draw real order statistics and always include the minimum, the maximum and the 16 most extreme values per tail. The regression case is the review's: 10 outliers among 25,088 weights must all be drawn. Two-sample plots pair both minima and both maxima.
- The reference line falls back to mean and std when the quartiles coincide, as with ReLU zeros.
- `niceTicks` never returns a single tick (100,000-case property test).
- The sorted fast paths equal the general functions exactly. Formatting never prints "−0.00" and keeps trailing zeros.

### 1.8 Layer statistics: `tests/layerStats.test.ts` (12 tests)

- Sampled z and a values come from the same entries.
- Weight gradients equal the batch mean of a direct forward and backward pass over the same images.
- A ReLU unit that can never fire counts as dead. A unit that is silent on the sampled digits but fires on another test digit does not: the dead-unit scan covers all 2,000 digits and stops early.
- The "blank input" share equals a brute-force count of all-zero input patches.
- Results are deterministic. The default settings run in a few seconds at a few ms per yield.

### 1.9 Units: `tests/units.test.ts` (40 tests)

- **Receptive fields:** known cases (3×3, 5×5, through one to three pools, clipped at corners). Brute-force agreement by perturbing single pixels on random networks with and without pooling. Fields are never larger than the 28×28 image.
- **Top-k:** matches a brute-force scan on every Small CNN layer. Conv hits record where the filter fired and the pixels behind that position. Labels of a trained output unit's top 50 follow its digit. All 2,000 digits are scanned within the time budget.
- **Activation maximisation:** the seeded gradient matches finite differences. The objective rises above its blank start on every layer type (conv, dense, output, MLP, LeNet-ish). Results are deterministic. Conv 1 synthesises the pattern its kernel describes.
- **Ranks and statistics:**
  - Exact ranks over all responses report ties, not interpolation; for example, a dense ReLU unit's exact zeros are counted.
  - The current input is ranked with a copy of the scan's weights.
  - Conv filters report the share of positions that fire, and sigmoid units are not "active on every digit".
  - Output units report how often each digit is predicted.

### 1.10 Attribution: `tests/attribution.test.ts` (24 tests)

- **Gradients:** match central finite differences pixel by pixel on smooth networks and on real digits, and away from kinks on pooled networks.
- **Blank pixels of real digits:** ReLU kinks no longer zero out saliency, and tied max-pool windows leave no lattice pattern.
- **Integrated gradients:**
  - It is exact (equal to x ⊙ W) for the Softmax preset.
  - It satisfies completeness within 5% at 32 steps, and the gap shrinks with more steps.
  - The gap is measured against Σ|IG|, so it cannot blow up when z(x) − z(blank) ≈ 0. It stays under 1% for every target on trained networks.
- **Occlusion:** matches a brute-force sweep, as a logit map and as a probability map. It is exactly 0 where every covering patch is already blank, and keeps detail where the probability saturates.
- **The job:** validates its parameters, reports exact progress, and runs end to end on a real CNN in under 3 s.

### 1.11 Embedding: `tests/embed.test.ts` (24 tests)

- **Sampling:** a balanced, deterministic sample of the first 100 digits of each class.
- **PCA:** recovers a known direction and matches a brute-force covariance eigendecomposition. Mirrored or swapped components are re-oriented to match the previous map, so a recomputed map keeps facing the same way through training steps.
- **t-SNE:**
  - Perplexity calibration hits its target.
  - The gradient matches finite differences of KL(P‖Q).
  - Three Gaussian clusters separate, and raw pixels keep their neighbours.
  - It starts from the PCA layout, so clusters form during early exaggeration.
  - The default run (1,000 points, 500 iterations) on a conv layer finishes in time.
- **Randomized PCA for wide layers:** equals exact PCA when the sketch is as wide as the data. A second pass over very wide layers gives the same result as reading from memory.
- **Flat layers** are reported as flat. Tick and tooltip labels use a true minus and never "−0.0000".

### 1.12 Datasets, freezing and model files: `tests/foundation.test.ts` (12 tests)

- **Colour images:** shapes and gradients through a 32×32×3 input match finite differences.
- **Feature vectors:** dense layers work; a convolution on a feature vector is flagged with a reason.
- **Freezing:** frozen layers keep their weights exactly while the trainable ones change. Training only the head never runs backpropagation through the frozen layers below it; a NaN sentinel in their gradient buffers proves it.
- **Copying layers** (`copyCompatible`): matching layers are copied and a changed layer starts fresh.
- **Model files:** architecture, weights and metadata round-trip. Broken files (wrong format, version, architecture, weight count or non-finite weights) are rejected with a message a reader can act on.
- **Dataset registry:** every dataset has class names, glyphs, and a shape or a dimensionality. Point data keeps its raw coordinates for plotting, with captions and training subsets.
- **Trainer:** learns a 3-class point dataset end to end (> 95%). A speed cap holds the trainer near the set rate, and lifting it trains at full speed again. Frozen layers stay put, and the trainer waits for data that matches the network.

### 1.13 Point datasets and features: `tests/synthetic.test.ts` (21 tests)

- For each of the 11 generators: the right sizes and split, balanced classes and coordinates in range. Every generator is deterministic per seed, differs across seeds, and spreads with noise.
- The feature catalogue lists linear, square, product and sine features, and computes each correctly.
- On the circle, a model with no hidden layer fails on the raw coordinates and succeeds once x₁² and x₂² are added, which is the playground's lesson.

### 1.14 Grid evaluation: `tests/grid.test.ts` (4 tests)

- Grid cells are centred, row 0 is at the top and other axes stay fixed. The plotting domain covers every point.
- `PointEvaluator` matches the network exactly, follows weight and architecture changes, and never disturbs the page's own network (whose intermediates the views read).

### 1.15 Decision boundaries: `tests/boundary.test.ts` (17 tests)

- **Confidence shading:** 1/K maps to pale and certainty to full colour; ties are pale; discrete mode gives flat tints. Margins and argmax agree on the predicted class.
- **Marching squares:** finds a straight boundary in the right place, traces a circle as one closed curve at the right radius, resolves saddles, and draws the edges between three class regions.
- **3-D surfaces:** surface nets put a sphere's vertices at its radius and close the surface; two-class probabilities give the p₀ = p₁ surface; with several classes each pairwise surface appears once.
- **Orbit camera:** a true rotation for any angles, x₁ to the right and x₃ up when level, back-to-front order, and which side of a slice faces the eye.
- **Budgets and labels:** the largest grid that fits the time budget, ticks and axis names, and a robust colour scale for unit maps that one runaway unit cannot wash out.

### 1.16 Dataset strip: `tests/datasetPicker.test.ts` (3 tests)

Training-set sizes the point generator produces, subset options that fit images or points, and 3-D projections with the vertical axis up.

### 1.17 Pretrained models and transfer: `tests/models.test.ts` (15 tests)

- **Zoo:** the index is well formed and matches its files; every file decodes and holds finite weights, and the accuracies in the docs match the index. `mnist-cnn` classifies the page's own 2,000 MNIST test digits, decoded from `public/data`, at 97% or more.
- **Transfer experiment:** every condition is recorded, and the panel's sentences are worded from the measured numbers.
- **Transfer:** copies exactly the compatible hidden layers, freezes them and adds a fresh output layer. A frozen layer keeps its weights while the new head learns, and an input of another shape is refused in words.
- **Load, edit, save:** loading on the same dataset uses the exact architecture and weights; keep-weights edits copy only the unchanged layers; a saved model round-trips.

### 1.18 Receptive fields on colour images: `tests/receptive.test.ts` (5 tests)

Small CNN fields on 32×32 match MNIST's sizes and are centred on the larger map; clipping happens at the edge of the 32×32 image; a field larger than the image becomes the whole image; one box covers all three channels, so a pixel outside it in any channel never moves the unit.

The existing analysis tests (1.7–1.11) also gained colour-image, point-dataset and tie cases.

---

## 2. Browser tests (`e2e/*.spec.ts`)

`playwright.config.ts` serves the production build with `vite preview` on port 4173 and
drives Chromium at 1600×1000. Every test also fails on any console error or uncaught exception.
The app exposes `window.raster = { store, client }` so tests can read training state.

### 2.1 Loads and renders

Opens the page, waits for the MNIST sprites, and checks:

- The title is "Raster Net Lab".
- The engine reports "In-browser, Web Worker".
- All seven sections are present.
- The untrained model's first evaluation is near chance (< 30 %).
- 60 test digits appear in the data grid.

![Initial page](screenshots/01-initial.png)

### 2.2 Play, pause, resume; training curves

Clicks Play and waits for more than 40 steps, then checks that the digits-per-second
counter is live. Pauses, waits 600 ms, and confirms the step count did not move.
Resumes until the 0.2-epoch evaluation lands, then checks:

- Test accuracy is above 70 %.
- The KPIs show percentages.
- The confusion matrix lists its most confused pairs.

![Training curves](screenshots/02-training-curves.png)

### 2.3 Step and +1 Epoch

With the Softmax preset and batch size 128:

- Step trains exactly 128 digits.
- +1 Epoch stops at epoch 1 after `ceil(19,872 / 128) + 1` steps.
- The final batch straddles the boundary by less than one batch.
- The bar shows `1.00`.

### 2.4 Architecture builder

- Adds a third conv layer (7×7 pools to 3×3×8).
- Switches L1 to tanh and turns its pooling off (28×28×8).
- Removes it, adds a 64-unit sigmoid dense layer, and checks the resulting spec.
- Cycles through all four presets, each of which rebuilds and re-evaluates the model.

![Builder](screenshots/03-builder.png)

### 2.5 Weight views: heatmap, Hinton, numbers, histogram

After a few training steps, switches through the four views and checks that each renders
different pixels. It then:

- Picks Dense 3 from the layer menu, which shows the weight templates per conv channel.
- Hovers across the network diagram until a "Conv 1 · filter n" tooltip appears.
- Clicks that filter, which moves the inspector to Conv 1 and selects the filter.

| Heatmap | Hinton | Numbers | Histogram |
| --- | --- | --- | --- |
| ![](screenshots/04-weights-heat.png) | ![](screenshots/04-weights-hinton.png) | ![](screenshots/04-weights-numbers.png) | ![](screenshots/04-weights-hist.png) |

![Dense templates](screenshots/04-weights-dense-templates.png)

### 2.6 Drawing pad with live classification

Trains the MLP preset past 90 % test accuracy, then draws a 7 with the mouse:

- A prediction appears mid-stroke, before the pen lifts, and changes as the stroke continues.
- The finished drawing is classified as **7**.
- The network diagram switches its input to "Your drawing".
- Clicking label chip 7 adds the drawing to the training set, and the pad clears.

![Draw pad](screenshots/05-draw.png)

### 2.7 Uploading images

Generates two PNGs in the page: a dark "3" on cream paper, and a light "1" on a black background.
It uploads them through the file input and checks:

- Each gets a 28×28 preview and a prediction.
- Both end up as white-on-black ink with the centre of mass within 1.5 px of the centre and an empty corner.
- "Train on it" is disabled until a label is picked, then adds the image to the training set.
- Removing it from the "In the training set" grid re-enables "Train on it" on the upload row.

![Uploads](screenshots/06-data-uploads.png)

### 2.8 Backpropagation walkthrough

After 25 training steps, checks the walkthrough follows the paused network ("Using the
network's weights at step n"). The default Small CNN produces 23 steps:

- Forward pass: input, conv/ReLU/pool ×2, dense, ReLU, logits, softmax.
- Loss.
- Backward pass: output δ, output gradients, ReLU′, dense gradients, pool′/ReLU′/conv gradients ×2.
- Update.

The test clicks Next through every step. Each one shows its formula, and the network
diagram highlights the active layer. At the update step it checks:

- The loss after one SGD step is lower than before.
- "Apply to network" changes the live weights and shows a confirmation.
- Arrow keys navigate between steps.
- The Numbers view prints the softmax tensors as values.

| Conv forward | Softmax | Output gradient |
| --- | --- | --- |
| ![](screenshots/07-backprop-conv-forward.png) | ![](screenshots/07-backprop-softmax.png) | ![](screenshots/07-backprop-output-gradient.png) |

| Max-pool backward | Conv gradients + saliency | Update |
| --- | --- | --- |
| ![](screenshots/07-backprop-pool-backward.png) | ![](screenshots/07-backprop-conv-gradients.png) | ![](screenshots/07-backprop-update.png) |

### 2.9 Backprop on an unlabeled drawing

Draws a digit. The walkthrough asks for the intended digit; picking 7 rewrites the loss as `−log p[7]`.

### 2.10 Dark mode

With `prefers-color-scheme: dark`, the page background is `rgb(14, 14, 13)`. Setting
`data-theme="light"` on the root switches it back to light, so an explicit theme wins over
the OS setting.

![Dark](screenshots/08-dark.png)

### 2.11 Phone width

At 390×844 the layout is one column with no horizontal scroll, and drawing still produces
a prediction.

![Phone](screenshots/09-phone.png)

### 2.12 No Web Workers

With `window.Worker` replaced by a constructor that throws, the engine reports
"In-browser, main thread" and training still works.


### 2.13 Distributions and the Q–Q weight view: `e2e/distributions.spec.ts` (5 tests)

- **Weights:** one Q–Q panel per layer follows training live. It shows the comparison with a normal distribution and with the initial weights, and tooltips name exact quantiles.
- **Pre-activations, activations and gradients:**
  - They come from the analysis worker.
  - The top Q–Q point equals the true maximum shown under the histogram.
  - Conv 1 reports its blank-input share.
  - "Dead units" counts over all test digits.
  - After "Apply to network" in 06, the status and numbers update even though the step number is unchanged.
- **Other checks:** the inspector's Q–Q view (now against initialisation, with a tooltip), dark theme, and 390 px with stacked panels and no sideways scroll.

| Activations | Inspector Q–Q | Dark |
| --- | --- | --- |
| ![](screenshots/10-distributions-activations.png) | ![](screenshots/10-distributions-inspector-qq.png) | ![](screenshots/10-distributions-dark.png) |

### 2.14 Units: `e2e/units.spec.ts` (5 tests)

- **Every layer:** top digits, the detail panel, receptive-field boxes, and synthesised inputs.
- **While training:** the current input is ranked with the scan's weights, and the status line matches what is shown.
- **Deep conv stacks:** fields are clipped to the image in text and crops.
- **Other checks:** the selection follows the network diagram and the inspector, "Show all", keyboard use, and phone width.

![Units](screenshots/11-units-light.png)

### 2.15 Attribution: `e2e/attribution.spec.ts` (4 tests)

- **The four maps:** they explain the prediction for the current input. A picked digit stays pinned until the input changes.
- **The drawing pad:** maps follow it live, and a blank input says so.
- **A new network:** clears the old maps at once. An invalid one waits and keeps the pick.
- **Phone width:** two panels per row at 390 px.

![Attribution](screenshots/12-attribution-light.png)

### 2.16 Embedding: `e2e/embedding.spec.ts` (5 tests)

- **PCA:** computes automatically, draws numerals, and has a hover preview, click-to-probe, digit highlight and mistake rings.
- **t-SNE:** runs only on request, animates, and is kept when switching back.
- **Robustness:**
  - The map keeps its orientation across training steps.
  - "Apply to network" makes it stale.
  - Reset keeps the chosen layer.
  - A dead layer is explained rather than drawn as noise.
- **Theme and layout:** in both themes, every digit colour keeps at least 4.5:1 contrast against the surface. Phone width works.

| PCA | t-SNE |
| --- | --- |
| ![](screenshots/13-embedding-pca.png) | ![](screenshots/13-embedding-tsne.png) |

### 2.17 Sync policy while training: `e2e/sync.spec.ts` (3 tests)

- On first view during training, each section runs its job once, and the job finishes while training continues. Before the round-2 fix this failed, with 6 restarted runs.
- After Reset + Play with 08–11 all on screen, every section computes once and shows a result. Before the fix this failed, with no run finishing.
- At phone width the status row stays one line in every state.

### 2.18 Datasets: `e2e/datasets.spec.ts` (5 tests)

- **Picker:** switches between image and point datasets, with the facts line, loading progress, section 07, the point controls and the train-subset control following. The index is one tab stop, and arrow keys and Enter pick from it.
- **CIFAR-10 photo mode:** crop, live class probabilities, flip and brightness.
- **Fashion-MNIST:** drawings fill the frame, and uploaded photos become light-on-black items as in the dataset.
- **Layout:** dark theme, and phone width without horizontal overflow.

| Dataset strip | 07 on point data |
| --- | --- |
| ![](screenshots/14-datasets-picker-light.png) | ![](screenshots/14-datasets-data-points.png) |

### 2.19 Decision boundaries and the network view: `e2e/boundary.spec.ts` (8 tests)

- **Circle:** after training, the colours of 97% or more of the clear sample spots on the canvas match `PointEvaluator`'s predictions.
- **Interaction:** clicks and arrow keys set the input; Add points and shift-click add training points; the toggles redraw. Spiral and XOR follow training live, at about 10 redraws a second.
- **3-D:** shells turn when dragged and with the buttons, and the slice slider moves the slice map. The helix, XOR cube and four blobs render a surface.
- **Network view (02):** unit maps for points, colour images for CIFAR-10, and MNIST unchanged.
- **Layout:** dark theme, and phone width without overflow.

| 2-D boundary | 3-D boundary (shells) |
| --- | --- |
| ![](screenshots/15-boundary-2d-light.png) | ![](screenshots/15-boundary-3d-shells.png) |

### 2.20 Pretrained models, transfer and freezing: `e2e/models.spec.ts` (10 tests)

- **Load:** the pretrained LeNet arrives with its exact architecture and weights and is evaluated on the page.
- **Transfer:** MNIST features move to Fashion-MNIST, and the frozen layers stay put while the new head learns (weights compared through `window.raster`).
- **Save and open:** save to a file and open it again; save in the browser, list, load and delete; a full browser storage is explained.
- **Freezing:** freeze toggles by mouse and keyboard, and keep-weights keeps the first conv layer through a dense-layer edit.
- **Builder:** follows the dataset (features for points, colour for CIFAR-10). A damaged download is explained in words. Phone width and dark theme.

### 2.21 Weights, training and backprop on every dataset: `e2e/core.spec.ts` (8 tests)

- **04 Weights:** colour kernel patches for CIFAR-10's first layer; feature-labelled weights for a point network; frozen layers marked.
- **05 Training:** Fashion-MNIST class names in the confusion matrix; large cells and a long-run epoch axis on a point dataset.
- **06 Backpropagation:**
  - steps through every stage on CIFAR-10 (23 steps) and on circle (15);
  - the update leaves frozen layers alone;
  - a gradient that stops at a layer whose units were all off is explained.
- **Speed control:** on circle, Normal holds the trainer at 3,300 samples a second or fewer, Max goes above 5,000, Slow stays at 400 or fewer. The speed follows the dataset.

| 04 on CIFAR-10 | 06 with a frozen layer |
| --- | --- |
| ![](screenshots/17-core-inspector-cifar-conv.png) | ![](screenshots/17-core-backprop-frozen.png) |

The 08–11 specs (2.13–2.16) gained CIFAR-10 and point-dataset cases.

---

## 3. Design review

The screenshots above were reviewed by eye against the Swiss brief:

- A grotesk type family (Archivo, with IBM Plex Mono for numbers).
- A strict 12-column grid with heavy black rules opening each numbered section.
- Flush-left text.
- Red reserved for state and positive weights, blue for negative weights.
- No shadows, gradients or rounded corners.

The review led to the fixes in section 5.

The digit colours in 11 Embedding were checked with the dataviz palette validator against this
page's own surfaces (`#ffffff` light, `#171716` dark):

- All eight hues pass the normal-vision floor and the contrast check in both themes.
- Every token clears 4.5:1 against the surface, because the numerals are text.
- Colour-blind separation between adjacent hues is 7.7. That is legal only with secondary encoding, which the numeral provides.
- Digits 8 and 9 are deliberate neutrals rather than invented ninth and tenth hues.

## 4. Adversarial review of the analysis views (08–11)

The four analysis sections were built in parallel by separate agents, each owning its own files and
passing its own unit and browser tests. They were then reviewed by **six independent reviewer
agents**, each with one lens:

- the maths of 08 Distributions;
- the maths of 09 Units;
- the maths of 10 Attribution;
- the maths of 11 Embedding;
- whole-app integration, performance and design;
- code robustness.

Reviewers could not edit the project. Every finding had to carry evidence they had observed: a
failing assertion, printed numbers, a screenshot, or a concrete input that breaks the code.

**Round 1** produced 47 findings: 4 high, 18 medium and 25 low. Shared root causes were fixed
centrally (commit `fdc7f05`); each section's findings were fixed by an agent that had to reproduce
the problem first and then prove the fix with a regression test (commit `732f740`). The high and
medium findings:

| Finding | Problem (as reproduced) | Fix |
| --- | --- | --- |
| DIST-1 (high) | Thinned Q–Q plots drew quantiles at 256 Blom positions, so the 40–60 most extreme values per tail were never plotted. 10 outliers at 1.0 among 25,088 weights vanished; the plot looked normal. | Plot real order statistics and always include the 16 most extreme values per tail and the min/max (test: all 10 outliers drawn). |
| U1 (high) | The Units detail ranked the current input with the live weights against a histogram from an older scan. After 25 training steps, 9 of 16 filters claimed "as high as the strongest" for a digit not in their top 16. | The input is measured with a copy of the scan's weights; the text says which step it belongs to. |
| U2 (high) | The rank came from interpolating inside histogram bins. For a ReLU unit where 74% of digits are exactly 0, it said "higher than 2%". | Exact ranks over all 2,000 sorted responses, with ties named ("tied with 63% of digits at 0"). |
| UX-1 (high) | "Apply to network" in 06 changed the weights without changing the step, so 08–11 never refreshed and still claimed to be current. | `store.weightsRev` counts every real weight change; freshness and caches use it. |
| F1 / UX-2 / UX-3 | The refresh debounce was restarted by every trainer status tick (~120 ms), so it never fired while training, and sections followed three different policies. | One shared policy in `syncedSection`, with a throttle that ticks cannot postpone. |
| U3 | The status said "Based on step N" while the previous scan was still shown. | `begin()` / `done(stamp)` record the weights a result was computed from; the status reads "Updating to step N…". |
| U4 | Every conv card read "active on 100% of digits", and every sigmoid unit too. | Conv: share of positions that fire. Dense: share of digits with z > 0. Output: share of digits predicted. |
| U5 | The detail panel never said which input "this input" was. | It names the input from its caption. |
| DIST-2 | Conv 1's flat run at 0 is mostly blank background (z = bias), not ReLU zeros, so the "18.8% exactly zero" figure and the guide contradicted the plot. | A "blank input" share (69%) and a corrected guide. |
| ATTR-1 | The completeness check printed "360% apart" when z(x) − z(blank) ≈ 0, although IG was accurate. | Gap measured against Σ\|IG\|. |
| ATTR-2 | Blank-pixel saliency came from max-pool tie-breaking and ReLU-at-0 conventions, which drew a lattice on LeNet. | Analyses split tied pool windows evenly and use the midpoint slope at kinks (`inputGradient(…, symmetric)`); training is unchanged. |
| EMB-1 | The PCA sign rule flipped the map after a single training step. | Orientation stays continuous with the previous result. |
| EMB-2 / EMB-3 | Same staleness problem as UX-1; Reset discarded the chosen layer. | Fixed through `weightsRev`; Reset keeps the layer when the architecture is unchanged. |
| EMB-4 / UX-5 / UX-7 | Digit colours for 1–4 and 9 fell below 3.5:1 contrast. The colour for 7 was the accent red that marks mistakes and the current input. | New tokens, all ≥ 4.5:1 in both themes. 7 is plum/pink, at least 17 ΔE from every other digit colour and 23 from the accent. |
| UX-4 / F3 | Same as U1, seen from the integration and code reviews. | As U1. |

The 25 low findings were fixed too. They covered tick fallbacks, "−0.00", tooltips during the t-SNE
animation, flat layers, receptive fields larger than the image, segmented controls on phones, the
type scale and wording, the redraw cost of the initial weights, and probe redraws while drawing.

**Round 2** checked the fixes and looked for regressions they introduced. Three reviewers did it: one for the maths, one for the flow between sections, and one for design. They used the same rules as round 1: they could not edit the project, and every claim needed evidence.

- **The fixes held.** The reviewers confirmed 39 of round 1's fixes with their own measurements. For example, a Q–Q property test ran over 25,175 (n, points) cases. The dead-unit counts matched a brute-force count on 13 networks. The IG completeness gap stayed under 0.75% on 400 browser pairs. Neighbour purity for the t-SNE start went from 0.095 to 0.345.
- **One of round 1's fixes caused a new high-severity problem.** The shared sync policy now started a job while training was running. However, every weights tick from the trainer (about 3 per second) made the job stale and restarted it, so a section with nothing to show could restart forever. After Reset + Play with 08–11 on screen, the distributions job ran 32 times in 12 s and finished 0 times.

| Finding | Problem (as reproduced) | Fix |
| --- | --- | --- |
| NEW-1 (high), F1, UX-3 | While training, a section without a result restarted its job on every weights tick: units-topk started 9 times, with 8 superseded. | A job already running for the current network finishes while training; the section then holds its result. `e2e/sync.spec.ts` reproduces the loop on the old build (6 restarts; 0 finished runs after Reset + Play) and passes on the fix. |
| Palette NEW-1 / UX-5 (medium) | Equalising lightness made digits 2 and 5 nearly the same colour (OKLab ΔE 6.8). In dark mode, digit 1 sat 7.8 from the accent red that marks the current input. | Slots 1, 2 and 3 (and dark 4) were moved. Light theme: no two colours closer than 9.9, none closer than 12.5 to the accent. Dark theme: 9.8 and 14.3. The e2e check asserts these floors in OKLab, together with 4.5:1 contrast. |
| UX-6 (low) | On phones, the Recompute button wrapped to a second line, so the content below jumped 27 px at every status change. | The status row stays on one line at every width (full text in the tooltip). The new test checks that its height doesn't change. |
| UX-9 (low) | An odd number of options (the 5 weight views) left "Q–Q" alone on a third row on phones. | An odd last option spans the row, and the frame keeps its 2 px weight. |

Six further findings (four low, two medium) were handed to the dataset work, because the same files were being rewritten for the new datasets:
- MATH-1: the t-SNE reduction for wide layers loses neighbours; one power iteration brings recall within 0.01 of exact PCA.
- NEW-2: re-picking the current input reran an identical attribution job.
- Dead units described with "strongest responses".
- An ASCII minus in the weights stats.
- Leftover unit, filter and logit wording.
- Mixed precision in the 08 stat grid, and a label summary that dropped a tied digit.

Their outcome is recorded in section 7.

## 5. Bugs found during verification

| Found by | Problem | Fix |
| --- | --- | --- |
| Unit test (trainer, +1 epoch) | +1 Epoch trained one extra batch into the next epoch, because the epoch boundary was only detected at the start of the next batch. | The trainer now closes the epoch as soon as the last sample is used. |
| Browser test (backprop) | After pausing, the walkthrough still showed the weights frozen at step 0. | The walkthrough stays frozen while training runs and follows the network while paused. |
| Screenshot review | The epoch counter showed `1.6e-3` after one step. | Fixed two decimals. |
| Screenshot review | Uppercase labels turned "η" into "Η". | The label reads "Step size"; η is in the formula and text. |
| Screenshot review | Conv captions in the network diagram ran into each other. | Two-line captions. |
| Screenshot review | Connection lines crossed through neighbouring feature maps. | Lines now run between column edges. |
| Screenshot review | Thumbnails overflowed their frames by the border width. | Canvas sized to the frame. |
| Screenshot review | "1 of your image mixed into training". | Singular and plural copy. |

## 6. Performance

Measured in the build container, a shared 2.8 GHz Xeon. Expect two to three times faster on a recent laptop.

| Model | Parameters | Training throughput |
| --- | --- | --- |
| Dense 784→32→10 | 25 k | ~6,000 digits/s |
| Small CNN (default) | 27 k | ~400–500 digits/s, so an epoch of 20,000 takes about 40 s |
| Tiny CNN (4 and 8 filters, dense 16) | 7 k | ~1,500 digits/s |

Test accuracy on the 2,000 held-out digits. Each row is one run with seed 1, Adam at learning rate 0.003 (the default) and batch size 32:

| Model | After 0.2 epoch | After 1 epoch |
| --- | --- | --- |
| Small CNN (default) | 90.1 % | 93.3 % |
| MLP | 85.9 % | 91.6 % |

With the old default learning rate of 0.001, the Small CNN reached 92.6 % after one epoch, so the default was raised to 0.003. Accuracy keeps rising with more epochs. The subset is 20,000 digits, so expect a little less than results reported on the full 60,000.

The convolution uses im2col, with the matrix products blocked over four filters. That was about 1.5× faster than direct loops.

Analysis jobs run in their own worker. Times are for the default Small CNN in this container; the page stays interactive because jobs yield every few milliseconds.

| Analysis | Time |
| --- | --- |
| Layer statistics (256 digits, plus the dead-unit scan over all 2,000) | ~0.4 s, up to ~1.6 s when a ReLU unit is silent |
| Top-k scan of one layer (2,000 digits) | 0.3–1.2 s |
| Activation maximisation of a whole layer (160 steps per unit) | 0.3–5 s |
| Attribution (32 IG steps, 6×6 occlusion) | ~0.1 s |
| PCA of a layer (1,000 digits) | 0.3–1.3 s |
| t-SNE (1,000 digits, 500 iterations) | ~5 s in Chromium |

## 7. The datasets release

**How it was built.** The foundation was written first and merged on its own: architectures with any input shape and class count, the dataset registry, the synthetic generators and features, the grid evaluator, the model file format, layer freezing in the engine and trainer, and the training speed cap. Six engineer agents then each took one area, worked in their own git worktree on their own files, and passed the full unit and browser suites before their branch was merged. The six areas were:
- the dataset strip and section 07;
- decision boundaries and section 02;
- the model zoo, transfer and freezing;
- sections 04–06;
- sections 08–09;
- sections 10–11.

Usage limits interrupted several of them. Each saved checkpoint commits, and a fresh agent resumed from the branch.

**Merge seams found by the full suite after merging** (each merge was followed by the full suites):

| Problem | Fix |
| --- | --- |
| With both the dataset strip and the boundary view on the page, three boundary tests clicked buttons that now existed twice ("x₁", "Random test image"). | Locators scoped to their section. |
| Section 06 traced the current example through a network that already expected the next dataset's input while that dataset loaded, and threw. | The lab waits until the example and the network agree. |
| The phone-width check in the dataset spec failed once under full-suite load. It could not be reproduced, even replaying its exact sequence. | The check waits for the layout to settle (polls for up to 5 s). A brief overflow right after a resize may remain and was passed to the review round. |
| The batch-size menu had no 10, the point datasets' default, so it showed 1 while training used 10. | 10 added. |

**The default network for CIFAR-10.** Two engineers reported independently that on CIFAR-10 most of the default Small CNN's 32 dense ReLU units died within the first few dozen steps. `scripts/cifar-defaults.ts` measures this with the page's engine on the page's own subset (the first 10,000 training images, 2,000 test images). It uses batch 32, Adam, two epochs (20,000 images), and counts a unit as dead when it never fires on the 2,000 test images:

| Network | Learning rate | Seed 1: test acc. · dead dense units | Seed 2 |
| --- | ---: | --- | --- |
| Small CNN, ReLU dense layer (old default) | 0.003 | 32.8% · 28 of 32 | 48.3% · 18 of 32 |
| Small CNN, ReLU | 0.001 | 40.7% · 21 of 32 | |
| **Small CNN, Leaky ReLU dense layer (new default)** | 0.003 | **48.5% · 1 of 32** | **49.2% · 2 of 32** |
| Small CNN, Leaky ReLU everywhere | 0.003 | | 47.9% · 0 of 32 |
| 16-32-64 filters and units, ReLU (4× the parameters, 3× slower) | 0.001 | 46.9% · 40 of 64 | |
| 16-32-64, ReLU | 0.003 | 49.4% · 46 of 64 | |

With ReLU, the result depended on how many units happened to die. The Leaky ReLU dense layer costs the same and kept nearly every unit alive at about 49% after two epochs, as good as a network four times larger. Colour datasets now default to it. Switching to a dataset with a different input shape (images and points, or grey and colour) also resets the network to that data's default, because a network built for 28×28 grey input is rarely right for colour photos.
