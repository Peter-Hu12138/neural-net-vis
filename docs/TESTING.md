# Testing and verification

Raster is checked at two levels:

| Level | Tool | Command | What it covers |
| --- | --- | --- | --- |
| Unit | Vitest (Node) | `npm test` | Engine maths, trainer protocol, preprocessing, bundled data, number formatting |
| Browser | Playwright + Chromium | `npm run test:e2e` | Every user-facing feature, run against the production build, with screenshots |

Both suites were last run on the current commit: **33/33 unit tests** and **12/12 browser tests** pass.

```bash
npm install
npm test            # unit tests, ~2 s
npm run test:e2e    # builds, serves dist/ on :4173, runs Chromium, ~70 s
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

- 22,000 labels, every digit well represented.
- The training split starts `5 0 4 1 9 2 1 3 1 4` and the test split starts `7 2 1 0 4 1 4 9 5 9`, matching canonical MNIST.
- Sprites are 8-bit greyscale with no gamma or ICC chunks, so browsers decode exact pixel values.
- Sampled digits have ink in the middle and almost none on the outer ring.

### 1.5 Formatting: `tests/format.test.ts`

Matrix-cell labels never exceed five characters; KPI and counter formats are stable.

---

## 2. Browser tests (`e2e/app.spec.ts`)

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

---

## 3. Design review

The screenshots above were reviewed by eye against the Swiss brief:

- A grotesk type family (Archivo, with IBM Plex Mono for numbers).
- A strict 12-column grid with heavy black rules opening each numbered section.
- Flush-left text.
- Red reserved for state and positive weights, blue for negative weights.
- No shadows, gradients or rounded corners.

The review led to the fixes in section 4.

## 4. Bugs found during verification

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

## 5. Performance

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
