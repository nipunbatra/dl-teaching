# CNN lecture — recording and source coverage

The lecture is now a complete adaptation of the **ML course CNN lecture**,
not a short independent overview. It has 128 frames in one continuous deck.

[Presentation](https://nipunbatra.github.io/dl-teaching/interactives/cnn/index.html?present)

## Recording route

Chapters 1–6 preserve the ML lecture progression: data and locality, manual
filtering, padding and stride, pooling and RGB, the LeNet shape/parameter
exercise, and training/inspecting the MNIST network. Chapters 7–10 deepen the
same story with the DL course's gradients, geometry, cost accounting, modern
blocks, and transfer-learning evidence. This is a full lecture sequence;
record natural chapter breaks if useful rather than rushing every interaction.

The chapter overview supports jumps. S opens per-frame teaching notes. Right
reveals one calculation at a time before changing frame. Reading mode exposes
all material. Restarting the page resets the shared model selections.

## Coverage of the original ML slides

| Original pages | What is retained and developed |
|---|---|
| 1–14 | Data-first opening, ImageNet/WordNet distinction, classification task, LeNet/AlexNet context. Historical narrative condensed; computational teaching sequence expanded. |
| 15–19 | Camera-resolution dense-layer exercise with a live parameter/memory calculator. Correct float32 bytes and parameter arithmetic. |
| 20–24 | Spatial locality, shifted cat example, reused local detector, hierarchy. Distinguish equivariance from invariant decisions. |
| 25–29 | Exact 6×6 edge notebook, first and second patches, complete output, sign reversal, PyTorch verification, original tutorial photographs. |
| 30–38 | Legal-start derivation, repeated valid 5×5 shrinkage, pixel-use heatmap. Explicitly correct the sequence that never reaches 1×1. |
| 39–45 | Padding, same-padding assumptions, stride starts, floor, shape exercises; add dilation. |
| 46–48 | Max and average pooling calculations, lost location, boundary sensitivity. |
| 49–55 | Original CIFAR RGB planes, full-depth kernels, per-channel contributions, output filter bank, bias and ReLU. |
| 56–73 | Every LeNet exercise: input, inferred kernel, channel count, pooling, second convolution, flatten, dense head, total parameters. Separate 32×32 slide and 28×28 notebook ledgers. |
| 74 | Complete MNIST training loop, curves, held-out predictions and errors, learned filters, exact patch calculation, all activation stages, flatten, hidden features, logits. AlexNet/VGG context follows. |
| 75 | Scratch versus transfer, cached activations, frozen/eval/no_grad contracts, real DL pet evidence. |
| 76 | Raw-pixel versus learned-feature PCA for the same 1,000 MNIST test examples, with explicit interpretation limits. |

## Original tutorials inspected and reused

- `ml-teaching/notebooks/cnn-edge.ipynb`: exact 6×6 input and vertical kernel.
- `convolution-operation.ipynb`: patch movement and arithmetic.
- `convolution-operation-stride.ipynb`: padding/stride animation, MNIST/CIFAR,
  channel planes, and the beach/building filtering experiments.
- `cnn.ipynb`: exact 28×28 LeNet-style architecture, split, training recipe,
  per-layer visualizations, and original saved notebook figures.

## DL extensions

The original L8 supplies the pet crop, exact shared-gradient example,
receptive-field recursion, equivariance conditions, and 5,418-parameter /
1,622,336-MAC classifier. L8B supplies the 28×28 channel-width exercise,
bottlenecks, Inception, residual scalar example, depthwise separation, actual
ResNet activation visualization, and small pet transfer experiment.

Attention I–III guide the presentation mechanics: persistent semantic colours,
one visible computation at a time, predict/reveal, a shared model behind the
whole walkthrough, and teacher notes. The ML teaching order remains the spine.

## Evidence and checks

The recreated MNIST run has 44,426 parameters and 96.10% test accuracy on
10,000 images. This is a separately seeded run, not the old notebook model.
JavaScript/PyTorch maximum checked activation discrepancy is about 2.23e-6
across 14 examples, including four selected errors. The tiny model has an
independent finite-difference gradient check over all 26 parameters.

All input images, weights, and results needed during recording are bundled.
A saved-checkpoint selector is labelled as such; only the tiny model trains
live. Training-only metrics and held-out metrics are kept distinct.

`interactives/cnn/coverage.json` maps every frame to its source; `sources.html`
links the original tutorials and the primary external references.
