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

%TRANSFER%

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
