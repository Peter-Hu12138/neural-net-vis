# Pretrained models and transfer learning

Section 01 of the page has three views, chosen with the tabs under the **Weights** line:

- **Layers**: the architecture builder, with a lock button on every layer to freeze it.
- **Pretrained**: a small model zoo of four networks trained in advance with the app's own engine
  on the full original datasets. **Load** puts one on the page exactly as it was trained.
  **Transfer** copies its hidden layers into a network for the current dataset, freezes them, and
  adds a new output layer.
- **Save and open**: your own network to a file or into the browser, and back again.

The **Weights** line above the tabs always says where the current weights came from.

## The zoo

| Model | Trained on | Hidden layers | Parameters | File | Test accuracy |
| --- | --- | --- | ---: | ---: | ---: |
| MNIST LeNet (`mnist-lenet`) | MNIST digits | conv 5×5×6 tanh, pool · conv 5×5×16 tanh, pool · dense 64 tanh | 53,462 | 286 kB | 98.89% |
| MNIST small CNN (`mnist-cnn`) | MNIST digits | conv 3×3×8 ReLU, pool · conv 3×3×16 ReLU, pool · dense 32 ReLU | 26,698 | 143 kB | 98.74% |
| Fashion-MNIST small CNN (`fashion-cnn`) | Fashion-MNIST | conv 3×3×8 ReLU, pool · conv 3×3×16 ReLU, pool · dense 32 ReLU | 26,698 | 143 kB | 90.09% |
| CIFAR-10 CNN (`cifar10-cnn`) | CIFAR-10 | conv 3×3×16 ReLU, pool · conv 3×3×32 ReLU, pool · conv 3×3×64 ReLU, pool · dense 64 ReLU | 89,834 | 480 kB | 70.02% |

Test accuracy is measured on the full official test set (10,000 images for each dataset) with the
weights exactly as stored. The page evaluates on its own test subset, the first 2,000 official test
images, where the numbers come out slightly differently: 98.25% (LeNet), 98.55% (MNIST small CNN),
90.90% (Fashion-MNIST small CNN) and 70.75% (CIFAR-10 CNN).

`mnist-lenet` and `mnist-cnn` have the layers of the builder's LeNet-ish and Small CNN presets, so
after **Load** every builder control shows a familiar network. `mnist-cnn` and `fashion-cnn` share
one architecture, so each can be transferred to the other's dataset.

The four files are `raster-model` JSON (see [File format](#file-format)) and together weigh
1.05 MB; the largest is 480 kB. `public/models/index.json` lists them for the page, and
`public/models/transfer.json` holds the measurements quoted below.

### How they were trained

`scripts/pretrain.ts` trains with `src/nn`, the same `Network` and `Optimizer` classes the page's
training worker uses, so the weights load unchanged in the browser.

- **Data.** The full official training sets: 60,000 MNIST digits, 60,000 Fashion-MNIST images and
  50,000 CIFAR-10 photos, read straight from the original IDX `.gz` files and the CIFAR-10 binary
  `.tar.gz` (parsed in Node with `zlib`; no other dependencies).
- **Scaling.** Exactly as the page scales its data: pixel × (1/255); colour images channel-major
  (all red values, then green, then blue), which is also the CIFAR-10 binary layout. Before
  training, the script checks that the raw files and the page's sprites in `public/data` agree: for
  MNIST and Fashion-MNIST the first 200 training and test labels and four whole images match bit for
  bit as network inputs; for CIFAR-10 (whose sprites are JPEG) the first 200 labels of each split
  match.
- **Optimiser.** Adam, batch 32, learning rate 0.002 decaying to 0.0002 along a cosine curve.
  Weights start from the engine's own initialisation with a fixed seed per model.
- **Augmentation.** CIFAR-10 only: each training photo is mirrored left–right with probability ½.
- **Parallelism.** Each batch is split across three worker threads; each runs its own copy of the
  network over its share, and the main thread adds up the gradients and takes the optimiser step.
  The arithmetic is that of one thread, only faster.

| Model | Epochs | Images seen | Wall time |
| --- | ---: | ---: | ---: |
| `mnist-lenet` | 2 | 120,000 | 202 s |
| `mnist-cnn` | 3 | 180,000 | 137 s |
| `fashion-cnn` | 4 | 240,000 | 182 s |
| `cifar10-cnn` | 3 | 150,016 | 916 s |

All four in one run: 24 minutes of wall time and 53.7 CPU-minutes (three worker threads on a
4-core 2.1 GHz Xeon virtual machine), somewhat over the 40 CPU-minutes planned, almost all of it
CIFAR-10.

### Reproducing

```bash
# The raw files: MNIST and Fashion-MNIST (four IDX .gz files each) and cifar-10-binary.tar.gz.
#   data-raw/mnist-raw/  data-raw/fashion-raw/  data-raw/cifar-raw/cifar-10-binary.tar.gz
npx vite-node scripts/pretrain.ts check              # raw files agree with public/data
npx vite-node scripts/pretrain.ts train              # all four; or name ids, e.g. train mnist-cnn
npx vite-node scripts/pretrain.ts transfer           # the experiment below → transfer.json
npx vite-node scripts/pretrain.ts index              # rewrite index.json from the files
# Options: --raw DIR (or $RASTER_RAW), --mnist DIR, --fashion DIR, --cifar FILE|DIR, --threads N
# transfer also takes --seeds N, --lr, --fine-lr, --sizes 1000:20,200:100, --pairs mnist-cnn,
#   --conditions scratch,frozen,…, --checkpoints 1000,5000 and --out FILE
```

Worker threads run this same file, bundled once with esbuild (it ships with Vite), because Node
cannot load the TypeScript sources directly.

## Transfer learning, measured

**Question.** A network trained on MNIST digits has learned convolution filters for strokes and
edges. Do they help with clothing when only a few labelled clothing images are available? And the
other way round, Fashion-MNIST to MNIST?

**Setup.** `npx vite-node scripts/pretrain.ts transfer` (the defaults: 3 seeds), with the same
engine; 60 training runs, 12 minutes of wall time and 34 CPU-minutes on three threads.

- **Source networks:** `mnist-cnn` and `fashion-cnn` from the zoo. Both are the page's Small CNN
  (conv 3×3×8 → pool → conv 3×3×16 → pool → dense 32 → output), so the comparison is symmetric.
- **Training data:** only the **first 1,000** or the **first 200** images of the target's official
  training set (the same images the page trains on first). **Testing:** the target's **full official
  test set**, 10,000 images.
- **Budget:** every run sees 20,000 training images: 20 passes over 1,000 images or 100 over 200.
- **Settings:** Adam, batch 32, learning rate 0.003, which are the page's defaults.
- **Seeds** 101, 102 and 103 change the new layers' starting weights and the order of the images.
  The tables show the mean of the three runs, with the lowest and highest in brackets; every run is
  in `public/models/transfer.json`.

The five conditions, each as it can be done on the page:

1. **From scratch:** random weights, every layer learns.
2. **Transfer, all frozen:** copy every hidden layer, freeze them, train only a new output layer.
   This is what the Transfer button does.
3. **Transfer, conv frozen:** copy every hidden layer, freeze only the two convolution layers; the
   dense layer and the new output layer learn. On the page: Transfer, then **Unfreeze Dense 3**.
4. **Transfer, then fine-tune:** condition 2 for the first 10,000 images, then every layer unfrozen
   for the other 10,000 at learning rate 0.001. On the page: Transfer, train, unlock every layer,
   lower the learning rate, train on.
5. **Control, random conv frozen:** like condition 3, but the frozen convolution layers keep their
   random starting weights instead of the copied ones. It shows how much the copied filters
   themselves are worth.

### Test accuracy after 20,000 training images

| From → to | Images | From scratch | Transfer, all frozen | Transfer, conv frozen | Transfer, then fine-tune | Control: random conv frozen |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MNIST → Fashion-MNIST | 1,000 | 81.9% (80.9–82.4) | 63.6% (63.0–64.0) | 82.0% (81.5–82.4) | 81.2% (81.1–81.3) | 81.2% (80.1–82.6) |
| MNIST → Fashion-MNIST | 200 | 76.0% (75.2–77.1) | 58.8% (58.7–59.2) | 75.0% (74.5–75.4) | 74.6% (73.7–75.4) | 76.6% (75.4–78.1) |
| Fashion-MNIST → MNIST | 1,000 | 94.0% (93.7–94.2) | 61.7% (60.8–63.0) | 92.8% (92.6–93.0) | 92.3% (92.0–92.6) | 91.8% (90.8–92.5) |
| Fashion-MNIST → MNIST | 200 | 80.0% (79.4–80.5) | 56.1% (54.9–57.6) | 78.9% (77.3–80.2) | 78.6% (77.7–80.0) | 80.2% (79.3–80.8) |

### Early on: after the first 1,000 and 5,000 training images

| From → to | Images | From scratch | Transfer, all frozen | Transfer, conv frozen | Control: random conv frozen |
| --- | ---: | ---: | ---: | ---: | ---: |
| MNIST → Fashion-MNIST | 1,000 | 68.2% → 78.8% | 14.5% → 51.6% | 67.4% → 79.8% | 66.4% → 75.6% |
| MNIST → Fashion-MNIST | 200 | 67.2% → 75.4% | 16.0% → 50.5% | 63.4% → 73.6% | 65.0% → 73.5% |
| Fashion-MNIST → MNIST | 1,000 | 75.3% → 91.4% | 11.2% → 37.8% | 59.3% → 88.6% | 72.4% → 88.7% |
| Fashion-MNIST → MNIST | 200 | 71.4% → 79.7% | 11.5% → 38.5% | 58.6% → 75.9% | 68.9% → 79.5% |

(Fine-tuning is identical to "all frozen" for its first 10,000 images.)

### Training time per run (20,000 images, evaluations excluded)

| From → to | Images | From scratch | Transfer, all frozen | Transfer, conv frozen | Transfer, then fine-tune | Control: random conv frozen |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MNIST → Fashion-MNIST | 1,000 | 28.7 s | 12.9 s | 13.7 s | 21.8 s | 14.2 s |
| MNIST → Fashion-MNIST | 200 | 27.1 s | 11.8 s | 12.0 s | 19.2 s | 12.6 s |
| Fashion-MNIST → MNIST | 1,000 | 28.9 s | 12.1 s | 13.4 s | 21.4 s | 14.2 s |
| Fashion-MNIST → MNIST | 200 | 28.2 s | 12.5 s | 13.2 s | 19.4 s | 12.5 s |

### What the numbers say

1. **Freezing every copied layer costs a lot.** With only the new output layer learning, accuracy
   reaches 56–64%, against 76–94% from scratch. The copied dense layer had learned 32 features for
   telling digits (or clothes) apart, and a single output layer on top of them cannot make up for
   that. It is also slow to get going: only 330 weights learn.
2. **Unfreezing the dense layer brings transfer level with training from scratch** (82.0% against
   81.9% on Fashion-MNIST from 1,000 images; at most 1.2 points behind elsewhere) **in under half the
   training time** (13 s against 28 s per run), because no gradients flow into the frozen
   convolutions.
3. **The copied filters themselves carry little here.** Random, untrained convolution filters,
   frozen, do about as well: within 1.6 points of the copied ones either way. A dense layer that
   learns can do a lot with random features. So the time saved comes from freezing, not from what
   the filters learned.
4. **No head start.** After the first 1,000 training images, the transferred network is level with
   training from scratch on Fashion-MNIST (67.4% against 68.2%) and behind it on MNIST (59.3% against
   75.3%; random filters reach 72.4%): filters learned on clothing slow down the first steps on
   digits.
5. **Fine-tuning everything does not help with this little data**: it ends 0.3–0.8 points below
   keeping the convolutions frozen.

Between two small sets of 28×28 grey pictures, then, transfer mostly saves training time. Transfer
pays off in accuracy when the source network has learned far more than the new examples can teach,
as large networks trained on millions of photos have; a zoo small enough to train in a browser tab
cannot show that. On the page this experiment explains two things: after Transfer the panel offers
to unfreeze the dense layer, and the Pretrained tab quotes the row that matches the current dataset.

An earlier run that was cut short, at learning rate 0.001 and with fine-tuning at 0.0003, gave the
same ordering (scratch, then conv frozen, then fine-tuned, then all frozen), with the frozen
conditions lower still.

## Using the zoo and your own models on the page

- **Load** switches to the model's dataset, rebuilds the exact architecture and copies every weight.
  The trainer evaluates it straight away, so the bar's test accuracy and the line under **Weights**
  show how it does on the page's 2,000 test images before any training. Loading also turns on
  "Keep trained weights when editing", so small edits keep the layers you do not change.
- **Transfer** is offered when the model's input has the same shape as the current dataset's (the
  28×28 grey models fit MNIST and Fashion-MNIST; the CIFAR-10 model needs 32×32 colour). Otherwise
  the button is disabled and the row says what input the model needs. The copied layers are frozen,
  the output layer is new, and the **Layers** tab says how many layers are frozen. When the copied
  layers include a dense layer, the page offers to unfreeze it, because of the measurements above.
- **Freeze** (the lock button on each layer under **Layers**, the output layer included) holds a
  layer's weights fixed while the rest trains. A frozen layer is hatched and its lock is closed. The
  engine skips the gradients of frozen weights, and stops backpropagating below the lowest layer that
  still learns, which is why training a new head is fast.
- **Keep trained weights when editing**: with it on, an edit to the architecture keeps the weights
  (and freeze setting) of every layer whose shape did not change; changed layers start fresh. With
  it off, every edit restarts from random weights.
- The line under **Weights** says where the current weights came from, for example "Random start
  (seed 1)", "MNIST LeNet, pretrained (98.9% test)", "MNIST LeNet, pretrained on MNIST digits (98.9%
  test there)" after a switch to another dataset, "Conv and dense layers transferred from MNIST
  small CNN, frozen; new output layer for Fashion-MNIST", or "3 of 4 layers from: …" after an edit,
  followed by how long they have been trained on the page.
- **Save and open**: Save to file downloads the current network as JSON; Open file reads one back
  and asks whether to load it as it is or transfer it; Save in this browser keeps it in
  `localStorage` (a page gets about 5 MB; when it is full the page says so and suggests deleting a
  model or saving to a file). Delete asks for a second click.

## File format

```jsonc
{
  "format": "raster-model",
  "version": 1,
  "name": "MNIST LeNet",
  "dataset": "mnist",                // decides the class names (and, for points, the features)
  "features": ["x1", "x2"],          // point datasets only
  "arch": { "input": { "c": 1, "h": 28, "w": 28 }, "layers": [/* LayerSpec */], "classes": 10 },
  "weights": ["<base64 float32>", …], // [W0, b0, W1, b1, …], little-endian
  "frozen": [false, false, false, false],
  "meta": { "description": "…", "trainedOn": "…", "samples": 120000, "epochs": 2,
            "testAccuracy": 0.9889, "created": "2026-10-08T20:33:50.412Z" }
}
```

Base64 adds a third to the 4 bytes per weight, so a file takes about 5.3 bytes per parameter. The
page refuses files whose architecture is invalid, whose weight counts do not match the
architecture, or that contain weights that are not finite numbers, and says which.
