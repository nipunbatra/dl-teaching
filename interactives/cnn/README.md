# Convolutional neural networks — full interactive lecture

One unified **128-frame deck**, built around the ML course CNN lecture and its
original tutorials, with deeper DL worked examples. The first six chapters
follow the ML sequence through the complete MNIST/LeNet walkthrough.

[Present online](https://nipunbatra.github.io/dl-teaching/interactives/cnn/index.html?present) ·
[Read online](https://nipunbatra.github.io/dl-teaching/interactives/cnn/index.html)

## Run locally

From the repository root:

```sh
python3 -m http.server 8792 --bind 127.0.0.1
# http://127.0.0.1:8792/interactives/cnn/index.html?present
```

All lecture assets, model weights, and example images are bundled. No CDN,
API key, or internet connection is needed during local recording. External
notebook/reference links need internet. The photo calculators need HTTP.

## Recording controls

- **Right / Space / N**: reveal the next authored calculation, then advance.
- **Left**: undo a reveal, then return to the previous completed frame.
- **O**: chapter-grouped overview; **S**: teaching notes and source for this frame.
- **P**: enter or leave presentation; **Escape**: close a dialog or read.
- Sliders retain native arrows. **N** advances while a slider/select is focused.
- A link such as `?present#lenet-c2/1` reopens a particular calculation reveal.

The reading page shows every calculation without a reveal step. The 16:9 stage
scales to the window. The MNIST image, channel, and saved epoch are shared across
its walkthrough. The live tiny-model training state survives slide navigation.
Both models reset when the page is reloaded.

## Content

- 1 · Images, data, and locality: 12 frames.
- 2 · Build a local detector: 12 frames.
- 3 · Padding and stride: 10 frames.
- 4 · Pooling, colour, and feature maps: 12 frames.
- 5 · Rebuild the LeNet exercise: 13 frames.
- 6 · Train and open up the MNIST network: 18 frames.
- 7 · Follow the shared gradients: 12 frames.
- 8 · Spatial reasoning and complete accounting: 17 frames.
- 9 · From LeNet to modern CNN blocks: 11 frames.
- 10 · Transfer, representations, and checks: 11 frames.

## Two models, two purposes

**MNIST LeNet:** the same 28×28 architecture and split as the ML teaching
notebook: 5,000 training examples, 1,000 disjoint validation examples, 10,000
held-out test examples, 44,426 parameters, Adam 0.001, batch 64, ten epochs.
The new seeded PyTorch run achieved 9,610/10,000 test predictions correct.
The original notebook also printed 96.1%; these are separate runs, not the
same checkpoint. Saved epochs 0, 1, 3, and 10 are selectable. Browser inference
recomputes all layers from their actual weights. Selecting a checkpoint is
not live training. Four deliberately selected error cases accompany one
example per digit. PCA uses the first 1,000 official test images and labels
only colour the projection.

**Live tiny CNN:** a separately defined 26-parameter model trains on six
synthetic 5×5 bars in JavaScript. Its loss, gradients, probabilities, and
feature inspector share one numerical model. Its reported accuracy is only
training accuracy. It complements, rather than replaces, the MNIST tutorial.

## Reproduce and verify

```sh
node scripts/check_cnn_math.cjs
node scripts/check_cnn_mnist.cjs
node scripts/check_cnn_browser.cjs
uv run --with torch python scripts/build_cnn_mnist_evidence.py
uv run --with torch python interactives/cnn/tiny_cnn.py --check
```

The MNIST builder downloads public MNIST from the torchvision S3 mirror,
verifies the standard archive MD5 values, trains on CPU with seed zero, and
exports evidence. Raw data and the training checkpoint stay in `tmp/cnn-mnist/`.
The JavaScript check compares every selected example's layer outputs against
PyTorch, including logits and predictions. Browser checks inspect every frame
with reveals open, controls, images, errors, overflow, and mobile reading.

## Editable sources

- `index.html`: canonical authored lecture, one entry point.
- `app.js`, `model.js`: presenter, foundational worksheets, and live training.
- `workshops.js`, `lenet.js`: real-image tutorial and MNIST inference inspectors.
- `style.css`, `workshops.css`: reading/presentation layouts.
- `evidence/mnist.json` and `mnist.js`: the same reproducible experiment;
  JSON is for independent checks, JS permits offline loading without fetch.
- `evidence/pet-transfer.json`: existing DL course transfer evidence.
- `coverage.json`: frame-by-frame mapping to the ML and DL source material.
- `sources.html`: exact source links, conventions, and provenance.
- `part1.html`, `part2.html`, `part3.html`: compatibility redirects only.

Images from the ML tutorial and saved notebook plots are copied unchanged.
Pet derivatives retain their source attribution and CC BY-SA 4.0 notice.
See `figures/PROVENANCE.md`. No other local lecture sources were rewritten.
