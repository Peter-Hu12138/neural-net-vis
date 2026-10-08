# Raster

An interactive neural-network builder and visualizer for handwritten digits, set in a Swiss
typographic style. You build a small feed-forward network, train it on MNIST in your browser,
and inspect every weight and gradient as it learns. Four interpretability views then show what
the trained network has learned: Q–Q plots of every layer, what each unit detects, which pixels
drive a prediction, and how each layer arranges the digits.

![Raster](docs/screenshots/01-initial.png)

## What it does

| Section | What you can do |
| --- | --- |
| **01 Architecture** | Stack conv and dense layers. Conv layers take 1–16 filters, a 3×3 or 5×5 kernel, an activation (ReLU, Leaky ReLU, Tanh, Sigmoid, Linear) and optional 2×2 max-pooling. Dense layers take 4–128 units and an activation. Output is always 10-way softmax. Four presets: Softmax, MLP, Small CNN, LeNet-ish. Shapes and parameter counts update as you edit. |
| **Control bar** | Play/pause continuous training, **Step** one batch, **+1 Epoch** (trains to the end of the epoch, then pauses), Reset. Learning rate, batch size and optimizer (SGD, Momentum, Adam). Live epoch, step, digits per second and test accuracy. |
| **02 Network** | A live diagram of the network, in the spirit of the TensorFlow Playground. Feature maps for every conv filter, unit activations for dense layers, and output probabilities. Lines show summed weights (red positive, blue negative). Hover for values; click a map or unit to inspect it. |
| **03 Draw** | Draw a digit with a mouse, pen or finger. It is classified live as you draw, with the probability of each digit beside it. The drawing becomes the network's input, and you can add it to the training set with a label. |
| **04 Weights** | Switch between a **heatmap**, a **Hinton diagram**, **raw numbers**, a **histogram** against the initial weights, and a **Q–Q plot** against a normal distribution, for any layer. Dense layers fed by images show each unit's weights as templates. |
| **05 Training** | Loss and accuracy curves, train vs. test, with an optional log scale for loss. Test-set confusion matrix with the most confused pairs. |
| **06 Backpropagation** | A step-by-step walkthrough of one example: forward pass layer by layer, softmax, cross-entropy, then every gradient on the way back. That includes the max-pool routing, activation slopes, kernel gradients and an input saliency map. Each step shows the formula, worked numbers for one unit, and the tensors involved. A final SGD step shows the loss before and after, and can be applied to the real network. |
| **07 Data** | Browse MNIST test digits; mistakes are marked. Upload, drop or paste your own images: each is cropped, scaled to 20 px, centred by mass on 28×28 and inverted if needed, the way MNIST was made. Label them to mix them into training. |
| **08 Distributions** | Q–Q plots for every layer of its weights, pre-activations, activations or weight gradients, against a normal distribution or against the initial weights. The extreme values are always plotted. Each panel also has a histogram and its skew, excess kurtosis and probability-plot correlation. Activations add the share of exact zeros and of blank input, and dead ReLU units are counted over all 2,000 test digits. |
| **09 Units** | What each filter and unit detects. Each card shows the 9 test digits that excite the unit most; conv filters show only the patch the filter sees. It also shows how often the unit fires and an input synthesised by activation maximisation. The detail panel adds the top 16 and bottom 8 digits with their receptive fields, the response histogram with the current input's exact rank, and the labels of the top 50. |
| **10 Attribution** | Which pixels drive the prediction for the current input (including a live drawing) and any target digit: saliency, gradient × input, integrated gradients with a completeness check, and occlusion. |
| **11 Embedding** | How a layer arranges 1,000 test digits in two dimensions, by PCA (with the current input projected live) or t-SNE (animated). Points are drawn as their numerals, with highlights for a digit and for mistakes. Click any point to make it the input. |

Training runs in a Web Worker and the analyses in a second one, so the page stays responsive; if workers are blocked, both fall back to the main thread. Analysis sections compute when they come into view and refresh when the weights change while training is paused. While training runs, a section holds its result and says which step it belongs to.

## Run it

```bash
npm install
npm run dev        # http://localhost:5173
npm run build      # static site in dist/, works from any sub-path
```

## Tests

```bash
npm test           # 176 unit tests: gradient checks, trainer and analysis protocols, statistics, analyses
npm run test:e2e   # 31 Playwright tests in Chromium against the production build
```

[docs/TESTING.md](docs/TESTING.md) records what each test verifies, the screenshots from the
browser run, the two rounds of adversarial review of the analysis views and what they fixed, the
bugs found during verification, and measured speed and accuracy.

## How it is built

- **No ML library.** `src/nn/` is a small engine written for this app: conv (im2col), max-pool, dense, five activations, softmax cross-entropy, and SGD/Momentum/Adam. Each layer keeps its activations and gradients, so the visualizations read them directly. Gradients are checked against finite differences in the tests.
- **Training** (`src/train/`). A `Trainer` class speaks a small message protocol and runs inside a worker. It streams status, metrics and weight snapshots to the page several times per second. Test-set evaluation runs five times per epoch, in slices, so pausing is instant.
- **Analyses** (`src/analysis/`). Each analysis is a generator job: a small piece of work per step, run in time slices by an `Analyzer` in its own worker. Every job gets a private copy of the network with the weights it was asked about. A newer request on the same channel cancels the older one. `Network.inputGradient` backpropagates from any unit to the pixels, which drives activation maximisation and attribution.
- **UI** (`src/ui/`). Plain TypeScript and canvas, with no framework. Colours come from CSS tokens, so every canvas follows the light or dark theme.
- **Data** (`public/data/`). Subsets of MNIST (20,000 / 2,000), Fashion-MNIST (10,000 / 2,000) and CIFAR-10 (10,000 / 2,000), packed as sprite sheets by `scripts/build-datasets.py` from the original files. The synthetic point datasets are generated in the browser (`src/data/synthetic.ts`).

```
src/
  nn/        network.ts (layers, forward/backward), optim.ts, activations.ts, rng.ts, types.ts
  train/     trainer.ts (worker body), worker.ts, client.ts (worker + fallback), protocol.ts
  analysis/  analyzer.ts (job scheduler), client.ts, stats.ts (quantiles, Q–Q), layerStats.ts,
             units.ts + receptive.ts (top images, activation maximisation), attribution.ts, embed.ts (PCA, t-SNE)
  data/      mnist.ts (sprite loader), preprocess.ts (MNIST-style 28×28 conversion)
  ui/        one module per section, plus draw.ts (heatmap/Hinton/number renderers), qq.ts, snapshot.ts and theme.ts
tests/       Vitest unit tests
e2e/         Playwright browser tests
```

## Credits

MNIST handwritten digit database by Yann LeCun, Corinna Cortes and Christopher J.C. Burges,
used under CC BY-SA 3.0. Type: Archivo and IBM Plex Mono, loaded from Google Fonts.
