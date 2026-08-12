# Lecture 9 real transfer-learning evidence

This directory is the compact, reproducible evidence case behind the public
Lecture 9 transfer-learning section. It uses actual Oxford-IIIT Pet images and
the cached torchvision ResNet-18 `IMAGENET1K_V1` checkpoint; no image in this
experiment is generated.

The experiment is intentionally a **small fixed teaching subset, not a
benchmark**. It compares a linear probe, `layer4` + head fine-tuning, and full
backbone + head fine-tuning under one controlled protocol. Validation selects
both the best epoch in each regime and the final regime. The fixed official
test subset is then evaluated once for the selected checkpoint only.

## Data contract

- Dataset: Oxford-IIIT Pet, CC BY-SA 4.0.
- Classes: six cat breeds (`Abyssinian`, `Bengal`, `Birman`, `Bombay`,
  `British_Shorthair`, and `Egyptian_Mau`).
- Training: 12 images per breed from the official `trainval` split.
- Validation: 6 disjoint images per breed from official `trainval`.
- Sealed test: 6 images per breed from the official `test` split.
- Subselection: SHA-256 rank with seed `20260812`, locked by filename and JPEG
  SHA-256 in `selection-manifest.csv`.

The initial manifest was ranked over official split members whose JPEGs were
available when this teaching case was bootstrapped. This availability filter
is one reason the result must not be presented as a dataset benchmark. Once
written, the manifest—not local archive extraction order—is authoritative.

## Fixed model protocol

- `torchvision.models.ResNet18_Weights.IMAGENET1K_V1`
- checkpoint SHA-256:
  `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`
- official weights transform: resize short side to 256, center crop to 224,
  convert to tensor, then ImageNet mean/std normalization
- no augmentation; 15 epochs; batch size 12; seed `20260812`
- AdamW, weight decay `1e-4`
- head learning rate `3e-3`; `layer4` learning rate `3e-4`; full-backbone
  learning rate `3e-5`
- backbone BatchNorm remains in evaluation mode, so running buffers are fixed
- all regimes share the exact same copied head initialization and minibatch
  order

Within a regime, validation accuracy selects the epoch, followed by lower
validation loss and earlier epoch. Across regimes, the order is validation
accuracy, fewer trainable parameters, lower validation loss, then earlier
epoch. This parsimony tie-break is declared before the sealed test is opened.

## Rebuild

The builder requires Python, PyTorch, torchvision, NumPy, and Pillow. Run from
the repository root, pointing it at an extracted official images directory,
the directory containing Oxford's `trainval.txt` and `test.txt`, and the cached
checkpoint:

```bash
/path/to/python \
  shared/vision-evidence/oxford-iiit-pet/l8b/build_transfer_evidence.py \
  --images-dir /path/to/oxford-images \
  --annotations-dir /path/to/oxford-annotations \
  --weights /path/to/resnet18-f37072fd.pth
```

Normal rebuilds hash-check and reuse `selection-manifest.csv`. Do not pass
`--refresh-selection` unless deliberately defining a new evidence case; that
option changes the sample and invalidates every previous numeric claim.

## Outputs and semantics

- `results.json`: full protocol, histories, selection rule, one sealed-test
  result, confusion matrix, runtime, environment, provenance, and hashes.
- `training-history.csv`: all 15 train/validation epochs for every regime.
- `sealed-test-predictions.csv`: the single selected-model test pass, including
  probabilities. It is an observed model output, not ground truth.
- `pretrained-trainval-features.npz`: 512-D pretrained features for train and
  validation only. It intentionally contains no test feature.
- `preprocessing-contract.png`: observed official transform on the shared
  `Abyssinian_1.jpg` example.
- `resnet18-activations.png`: actual pretrained stage activations on that image;
  the display is explicitly **not saliency**.
- `transfer-curves.svg`: measured vector curves and selected sealed-test result.
- `sealed-test-examples.jpg`: one post-selection test prediction per true breed.

The shared photograph and all subset photographs retain their original
copyright and CC BY-SA 4.0 licensing. The contact sheet is a derived work under
the same license.
