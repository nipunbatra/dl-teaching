# Figure provenance

## ML teaching assets (copied unchanged)

`ml-course/beach.jpg` and `buildings.jpg` are the original assets in
`ml-teaching/neural-networks/assets/cnn/figures/`, used by
`convolution-operation-stride.ipynb`. Their original third-party ownership is
not replaced by this adaptation. The browser resizes them to 192×128 for the
live numeric demonstration; source files remain unchanged.

`notebook-cell-*.png` are decoded, unchanged saved PNG outputs from
`ml-teaching/notebooks/cnn.ipynb`, at the numbered cell indices.
`cifar-43.png` through `cifar-46.png` are unchanged saved CIFAR-10 car/R/G/B
plots in cells 43–46 of `convolution-operation-stride.ipynb`.

## DL teaching assets (copied unchanged)

`head-roi-crop.png`, `real-crop-five-bands.png`, and `dl-course/*` come from
`shared/vision-evidence/oxford-iiit-pet/derived/` or its `l8b/` evidence folder.
They show the existing course's Oxford-IIIT Pet examples, actual pretrained
ResNet-18 activations, preprocessing, and measured transfer curves.
The source records CC BY-SA 4.0 for the pet imagery; copyright remains with
original image owners. See that folder's README, evidence.json, results.json,
and generation scripts for precise crops and model/procedure provenance.

## Newly computed material

MNIST examples, checkpoints, feature maps, curves, and PCA are reproduced by
`scripts/build_cnn_mnist_evidence.py`. They are not invented or generated
illustrations. The browser calculates activations directly from the selected
checkpoint. Diagrams and scalar grids are authored teaching constructions.
