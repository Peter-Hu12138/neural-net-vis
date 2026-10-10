# Raster

An interactive neural-network builder and visualizer, set in a Swiss typographic style. You
build a small feed-forward network and train it in your browser on handwritten digits, clothing,
colour photos, or 2-D and 3-D point clouds. As it learns you can inspect every weight and gradient,
and on point data the decision boundary itself. You can also load pretrained networks, transfer
their layers to another dataset, and freeze layers while the rest train. Four interpretability
views show what a trained network has learned: Q–Q plots of every layer, what each unit detects,
which inputs drive a prediction, and how each layer arranges the data.

![Raster](docs/screenshots/01-initial.png)

## Datasets

| Dataset | What it is | Shipped with the page |
| --- | --- | --- |
| **MNIST** | Handwritten digits, 28×28 grey | 20,000 training / 2,000 test images |
| **Fashion-MNIST** | Photos of clothing in 10 classes, 28×28 grey. Same format as MNIST, much harder | 10,000 / 2,000 |
| **CIFAR-10** | Colour photos of 10 kinds of object (airplane … truck), 32×32×3 | 10,000 / 2,000 |
| **Points in 2-D** | Circle, XOR, two blobs, spiral, moons, three blobs, checkerboard | generated in the browser |
| **Points in 3-D** | Shells (a ball in a shell), double helix, XOR cube, four blobs | generated in the browser |

The strip under the training bar switches dataset. For point data it also sets the noise, the
number of points, the share used for training and a fresh draw, and it picks the network's input
features. As in the TensorFlow Playground, these can be the raw coordinates or also squares, products
and sines of them (x₁², x₁x₂, sin x₁, …). With the right features even a network with no hidden
layer separates the circle. **Train on** limits any dataset to its first few examples (for MNIST,
200, 1,000 or 5,000), which shows overfitting and why transfer learning helps.

## What it does

| Section | What you can do |
| --- | --- |
| **01 Architecture** | Stack conv and dense layers. Conv layers take 1–64 filters, a 3×3 or 5×5 kernel, an activation (ReLU, Leaky ReLU, Tanh, Sigmoid, Linear) and optional 2×2 max-pooling (images only). Dense layers take 2–256 units. The output layer has one unit per class with softmax. Presets per kind of data. Shapes and parameter counts update as you edit, with a warning when a network gets too big to train comfortably in a tab. **Freeze** any layer (training then leaves its weights alone), and **keep trained weights** while you edit the layers above. |
| **Pretrained models** | Four networks trained in advance on the full original datasets with this app's own engine. **Load** puts one on the page exactly as trained; **Transfer** copies its hidden layers into a network for the current dataset, frozen, under a new output layer. Save your own network to a file or in the browser, and open it again. Measured transfer results are shown in the panel and in [docs/models.md](docs/models.md). |
| **Control bar** | Play/pause, **Step** one batch, **+1 Epoch**, Reset. Learning rate, batch size, optimizer (SGD, Momentum, Adam) and speed: **Normal** caps training at 3,000 samples a second so a small network can be watched learning; **Max** removes the cap. |
| **02 Network** | A live diagram of the network for the current input. For images: feature maps for every conv filter (in colour for CIFAR-10), unit activations and output probabilities by class name. For point data, as in the TensorFlow Playground: every input feature, hidden unit and output drawn as a heatmap over the input plane, with lines whose width and colour show the weights. |
| **03 Draw / Try a photo / Decision boundary** | MNIST and Fashion-MNIST: draw with a mouse, pen or finger and see the classification update live. CIFAR-10: drop, paste or upload a photo; it is cropped and averaged down to 32×32, and you can flip it or change its brightness and contrast to watch the prediction move. 2-D points: the predicted class over the whole plane, shaded by confidence, with the decision boundary as a line, the training and test points, click-to-try and click-to-add points. 3-D points: a cube you can turn, the boundary surface between the classes, and a movable slice through it. |
| **04 Weights** | A **heatmap**, a **Hinton diagram**, **raw numbers**, a **histogram** against the initial weights, or a **Q–Q plot**, for any layer. Colour conv filters show each 3×3×3 kernel as a colour patch beside its R, G and B weights; dense layers fed by images show each unit's weights as a template. Point networks label each input with its feature. Frozen layers are marked. |
| **05 Training** | Loss and accuracy curves, training vs. test. A confusion matrix with class names and the most confused pairs. |
| **06 Backpropagation** | One example, one step at a time: the forward pass layer by layer, softmax, cross-entropy, then every gradient on the way back, including max-pool routing, activation slopes and kernel gradients. Each step shows the formula, worked numbers for one unit and the tensors involved. Colour images and point features have their own input steps. The final SGD step leaves frozen layers alone and can be applied to the real network. Where a gradient stops (every ReLU in a layer off), the lab says why. |
| **07 Data** | Images: browse test images and mark mistakes; upload, drop or paste your own (made MNIST-style, Fashion-style or as a 32×32 colour photo) and mix them into training. Points: the training and test points, the current point's coordinates and the feature values the network actually receives, and the points you added. |
| **08 Distributions** | Q–Q plots of every layer's weights, pre-activations, activations or weight gradients against a normal distribution or against the initial weights, with a histogram, skew, excess kurtosis and dead-unit counts over the whole test set. |
| **09 Units** | What each filter and unit responds to: the test images that excite it most (with the patch a conv filter sees) and an input synthesised to excite it, in colour for CIFAR-10. For point data, each unit's response over the input plane and its strongest test points. Dead units are named as such. |
| **10 Attribution** | Which inputs drive the prediction for the current input and any target class: saliency, gradient × input, integrated gradients with a completeness check, and occlusion. |
| **11 Embedding** | How a layer arranges 1,000 test examples in two dimensions, by PCA (with the current input projected live) or t-SNE (animated), coloured by class. Click any point to make it the input. |

Training runs in a Web Worker and the analyses in a second one, so the page stays responsive; if
workers are blocked, both fall back to the main thread. Analysis sections compute when they come
into view and refresh when the weights change while training is paused. While training runs, a
section holds its result and says which step it belongs to.

## Run it

```bash
npm install
npm run dev        # http://localhost:5173
npm run build      # static site in dist/, works from any sub-path
```

## Tests

```bash
npm test           # Vitest unit tests: gradient checks, data and features, trainer and analysis protocols, statistics, analyses, models
npm run test:e2e   # Playwright browser tests in Chromium against the production build
```

[docs/TESTING.md](docs/TESTING.md) records what each test verifies, the screenshots from the
browser runs, the adversarial reviews and what they fixed, and measured speed and accuracy.
[docs/models.md](docs/models.md) describes how the pretrained models were trained and the transfer
experiment.

## How it is built

- **No ML library.** `src/nn/` is a small engine written for this app: conv (im2col), max-pool, dense, five activations, softmax cross-entropy, and SGD/Momentum/Adam. A network is an architecture (input shape, layers, number of classes), so the same code trains on grey images, colour images and feature vectors. Each layer keeps its activations and gradients, so the visualizations read them directly. Frozen layers get no updates, and backpropagation stops below the lowest trainable layer. Gradients are checked against finite differences in the tests.
- **Training** (`src/train/`). A `Trainer` class speaks a small message protocol and runs inside a worker. It streams status, metrics and weight snapshots to the page several times per second (ten times for small networks, so decision boundaries animate). Test-set evaluation runs in slices, so pausing is instant.
- **Analyses** (`src/analysis/`). Each analysis is a generator job run in time slices by an `Analyzer` in its own worker, on a private copy of the network with the weights it was asked about. A newer request on the same channel cancels the older one; while training, a running job finishes before a section holds its result.
- **Data** (`src/data/`, `public/data/`). Sprite sheets of MNIST (20,000 / 2,000), Fashion-MNIST (10,000 / 2,000) and CIFAR-10 (10,000 / 2,000 as JPEG), packed by `scripts/build-datasets.py` from the original files. The point datasets and their features are generated in the browser (`synthetic.ts`, `features.ts`); `grid.ts` evaluates a network over a grid of points for the boundary and unit maps.
- **Models** (`src/models/`, `public/models/`, `scripts/pretrain.ts`). A small JSON format with base64 float32 weights; the zoo is trained by `scripts/pretrain.ts` with the same engine, in Node.
- **UI** (`src/ui/`). Plain TypeScript and canvas, with no framework. Colours come from CSS tokens, so every canvas follows the light or dark theme.

```
src/
  nn/        network.ts (layers, forward/backward, freezing, copying layers), optim.ts, activations.ts, rng.ts, types.ts
  train/     trainer.ts (worker body), worker.ts, client.ts (worker + fallback), protocol.ts
  analysis/  analyzer.ts (job scheduler), client.ts, stats.ts, layerStats.ts, units.ts, receptive.ts, attribution.ts, embed.ts
  data/      datasets.ts (registry), images.ts (sprite loader), synthetic.ts, features.ts, grid.ts, preprocess.ts
  models/    format.ts (model files), zoo.ts
  ui/        one module per section, plus datasetPicker, modelPanel, boundary2d/3d, draw.ts, qq.ts, snapshot.ts, theme.ts
scripts/     build-datasets.py (sprite sheets), pretrain.ts (model zoo and transfer experiment)
tests/       Vitest unit tests
e2e/         Playwright browser tests
```

## Credits

MNIST by Yann LeCun, Corinna Cortes and Christopher J.C. Burges (CC BY-SA 3.0). Fashion-MNIST by
Zalando Research (MIT licence). CIFAR-10 by Alex Krizhevsky, Vinod Nair and Geoffrey Hinton. The
synthetic point datasets follow the TensorFlow Playground. Type: Archivo and IBM Plex Mono, loaded
from Google Fonts.
