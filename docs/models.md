# Pretrained models and transfer learning

Section 01 of the page opens with a small model zoo: four networks trained in advance with the
app's own engine on the full original datasets. **Load** puts one on the page exactly as it was
trained. **Transfer** copies its hidden layers into a network for the current dataset, freezes them,
and adds a new output layer. Your own networks can be saved to a file or in the browser and opened
again the same ways.

## The zoo

| Model | Trained on | Hidden layers | Parameters | File | Test accuracy |
| --- | --- | --- | ---: | ---: | ---: |
%ZOO%

Test accuracy is measured on the full official test set (10,000 images for each dataset) with the
weights exactly as stored. The page evaluates on its own test subset (the first 2,000 official test
images), where the numbers come out slightly differently: %SUBSET%.

All four files are `raster-model` JSON (see [File format](#file-format)) and together weigh
%TOTALBYTES%. `public/models/index.json` lists them for the page.

### How they were trained

`scripts/pretrain.ts` trains with `src/nn` (the same `Network` and `Optimizer` classes the page's
training worker uses), so the weights load unchanged in the browser.

- **Data.** The full official training sets: 60,000 MNIST digits, 60,000 Fashion-MNIST images and
  50,000 CIFAR-10 photos, read straight from the original IDX `.gz` files and the CIFAR-10 binary
  `.tar.gz` (parsed in Node with `zlib`; no other dependencies).
- **Scaling.** Exactly as the page scales its data: pixel × (1/255); colour images channel-major
  (all red values, then green, then blue), which is also the CIFAR-10 binary layout. Before training,
  the script checks that the raw files and the page's sprites in `public/data` agree: for MNIST and
  Fashion-MNIST the first 200 training and test labels and four whole images match bit for bit as
  network inputs; for CIFAR-10 (whose sprites are JPEG) the first 200 labels of each split match.
- **Optimiser.** Adam, batch 32, learning rate 0.002 decaying to 0.0002 along a cosine curve.
  Weights start from the engine's own initialisation (He for ReLU, Glorot for tanh) with a fixed seed.
- **Augmentation.** CIFAR-10 only: each training photo is mirrored left–right with probability ½.
- **Parallelism.** Each batch is split across worker threads; each thread runs its own copy of the
  network over its share, and the main thread adds the gradients and takes the optimiser step. This
  is the same arithmetic as one thread, just faster.

| Model | Epochs | Images seen | Wall time | 
| --- | ---: | ---: | ---: |
%TRAIN%

Total: %CPU% on a 4-core 2.8 GHz Xeon shared with other work, run at low priority (`nice -n 10`).

### Reproducing

```bash
# The raw files: MNIST and Fashion-MNIST (four IDX .gz files each) and cifar-10-binary.tar.gz.
#   data-raw/mnist-raw/  data-raw/fashion-raw/  data-raw/cifar-raw/cifar-10-binary.tar.gz
npx vite-node scripts/pretrain.ts check                       # raw files agree with public/data
npx vite-node scripts/pretrain.ts train                       # all four; or name ids, e.g. train mnist-cnn
npx vite-node scripts/pretrain.ts transfer --seeds 3          # the experiment below → transfer.json
npx vite-node scripts/pretrain.ts index                       # rewrite index.json from the files
# Options: --raw DIR (or $RASTER_RAW), --mnist DIR, --fashion DIR, --cifar FILE|DIR, --threads N
```

Worker threads run this same file bundled once with esbuild (it ships with Vite), because Node
cannot load the TypeScript sources directly.

## Transfer learning, measured

**Question.** A network trained on MNIST digits has learned strokes, edges and blobs. Are those
features useful for clothing, when only a few labelled clothing images are available? And the other
way round?

**Setup.** `npx vite-node scripts/pretrain.ts transfer --seeds 3`, same engine.

- Source networks: `mnist-cnn` and `fashion-cnn` from the zoo. Both are the page's Small CNN
  (conv 3×3×8 → pool → conv 3×3×16 → pool → dense 32 → output), so the comparison is symmetric.
- Training data: only the **first 1,000** or **first 200** images of the target's official training
  set (the same images the page uses first). Testing: the **full official test set** of the target
  (10,000 images).
- Every condition sees the same number of training images: 20 epochs for 1,000 images and 60 epochs
  for 200, Adam, batch 32, learning rate 0.001.
- Seeds 101, 102 and 103 change the new layers' starting weights and the shuffling; the table shows
  the mean of the three runs (each run is listed in `public/models/transfer.json`).

The four conditions:

1. **From scratch**: random weights, everything trains.
2. **Transfer, frozen**: copy every hidden layer, freeze them, train only a new output layer. This is
   what the page's Transfer button does.
3. **Transfer, conv frozen**: copy every hidden layer, freeze only the two convolution layers; the
   dense layer and the new output layer train. On the page: Transfer, then unlock Dense 3.
4. **Transfer, then fine-tune**: condition 2 for the first half of the epochs, then unfreeze
   everything and continue at learning rate 0.0003. On the page: Transfer, train, unlock every layer,
   lower the learning rate and train on.

%TRANSFER%

%FINDINGS%

## Using the zoo and your own models on the page

- **Load** switches to the model's dataset, rebuilds the exact architecture and copies every weight.
  The trainer evaluates it straight away, so the bar's test accuracy and the line under "Weights"
  show how it does on the page's 2,000 test images before any training. Loading also turns on
  "Keep trained weights when editing", so small edits keep the layers you do not change.
- **Transfer** is offered when the model's input has the same shape as the current dataset's (all
  28×28 grey models fit MNIST and Fashion-MNIST; the CIFAR-10 model needs 32×32 colour). Otherwise the
  button is disabled and the row says what input the model needs. The copied layers are frozen and
  marked with hatching and a closed lock in the builder; the output layer is new.
- **Freeze** (the lock button on each layer, the output layer included) holds a layer's weights
  fixed while the rest trains. Training below the lowest unfrozen layer is skipped entirely, which is
  why training only a new head is fast.
- **Keep trained weights when editing**: with it on, an edit to the architecture keeps the weights
  (and freeze setting) of every layer whose shape did not change; changed layers start fresh. With
  it off, every edit restarts from random weights.
- The line under **Weights** always says where the current weights came from, for example
  "Random start (seed 1)", "MNIST LeNet, pretrained (98.9% test)", "Conv and dense layers transferred
  from MNIST small CNN, frozen; new output layer for Fashion-MNIST", or "3 of 4 layers from: …" after
  an edit, followed by how long they have been trained on the page.
- **Your model**: Save to file downloads the current network as JSON; Open file reads one back and
  asks whether to load it as it is or transfer it; Save in this browser keeps it in `localStorage`
  (each page gets about 5 MB; when it is full the page says so and suggests deleting a model or saving
  to a file). Delete asks for a second click.

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
            "testAccuracy": 0.989, "created": "2026-10-08T20:30:00.000Z" }
}
```

Base64 adds a third to the 4 bytes per weight, so a file takes about 5.3 bytes per parameter. The
page refuses files whose architecture is invalid, whose weight counts do not match the architecture,
or that contain weights that are not finite numbers, and says which.
