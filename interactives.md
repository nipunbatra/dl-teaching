# Interactives — per-lecture status

Every deck's `[I]` slides point at a browser explainer hosted in the
**[interactive-articles](https://github.com/nipunbatra/interactive-articles)**
repo (`~/git/interactive`), deployed at
`https://nipunbatra.github.io/interactive-articles/<slug>/`.

- **LIVE** — an article already exists and is linked from the deck (`#interbox(link-to: …)`).
- **TO BUILD** — the deck references it as `(to build)`; needs a new article in
  `interactive-articles/src/articles/<slug>/` (self-contained `index.html` + `meta.json`),
  then `node scripts/build.js`, then wire the deck's `#interbox` to `link-to`.

New articles follow the repo house style: single self-contained `index.html`
(warm-paper palette `--bg #fdfcf9`, blue `--accent #2c6fb7`, orange `--warm #d9622b`;
Source Serif 4 / Manrope / IBM Plex Mono; KaTeX from CDN; a canvas/SVG widget with
live controls) + a `meta.json` (slug, title, tagline, summary, difficulty, tags, highlights).

---

## First 10 lectures

| # | Lecture | Interactive | Slug | Status |
|---|---------|-------------|------|--------|
| 1 | Why These Losses? | likelihood → loss (noise density → −log p → fitted line; Gaussian/Laplace/Uniform/Student-t + outlier) | `likelihood-to-loss` | **TO BUILD** |
| 1 | | softmax → cross-entropy (logits, true class, temperature → p → −log p_y → p−y) | `softmax-cross-entropy` | **TO BUILD** |
| 1 | | Bayes classifier (two Gaussians, prior, test point → posterior + boundary) | `bayes-classifier` | **TO BUILD** |
| 1 | | MAP = likelihood × prior (contours, MLE/MAP/0, Gaussian vs Laplace) | `map-regularization` | **TO BUILD** |
| 2 | Linear → MLP | linear-regression & GD, activations, softmax | `optimizer-race`, `softmax-temperature`, `vanishing-gradients` | LIVE |
| 2 | | universal approximation | `universal-approximation` | LIVE |
| 2 | | MLP decision boundary (XOR/circles/spirals playground) | `mlp-decision-boundary` | **TO BUILD** |
| 3A | Calculus Toolkit | gradient field & descent, curvature | `optimizer-race` | LIVE |
| 3A | | tangent line / local linear approx | `derivative-tangent` | **TO BUILD** |
| 3A | | surface slices (∂/∂x, ∂/∂y) | `surface-partials` | **TO BUILD** |
| 3A | | Jacobian as a local warp (grid → ellipse) | `jacobian-local-map` | **TO BUILD** |
| 4 | Backpropagation | computation graph & MLP backprop | `autograd` | LIVE |
| 4 | | gradient flow | `vanishing-gradients` | LIVE |
| 5 | Optimization | GD/SGD/momentum/Adam race, SGD noise | `optimizer-race` | LIVE |
| 5 | | LR schedules | `lr-schedule-visualizer` | LIVE |
| 6 | Trainability | signal/gradient flow, activations | `vanishing-gradients` | LIVE |
| 6 | | BatchNorm vs LayerNorm | `norm-comparison` | LIVE |
| 6 | | residual gradient flow | `resnet` | LIVE |
| 7 | Regularization | capacity / double descent | `double-descent` | LIVE |
| 7 | | dropout | `dropout-playground` | LIVE |
| 7 | | label smoothing / calibration | `calibration` | LIVE |
| 7 | | weight decay (λ sweep → norm & boundary) | `weight-decay` | **TO BUILD** |
| 7 | | augmentation & mixup (2-D boundary) | `augmentation-mixup` | **TO BUILD** |
| 8 | CNNs | convolution playground, padding/stride | `convolution-visualizer` | LIVE |
| 8 | | receptive field | `receptive-field-grower` | LIVE |
| 8 | | pooling (max/avg on a grid) | `pooling-visualizer` | **TO BUILD** |
| 8 | | padding & stride → output size | `padding-stride` | **TO BUILD** |
| 8B | Modern CNNs | receptive field, residual block | `receptive-field-grower`, `resnet` | LIVE |
| 8B | | Inception cost (1×1 bottleneck savings) | `inception-cost` | **TO BUILD** |
| 8B | | transfer-learning (freeze/finetune curves) | `transfer-learning-playground` | **TO BUILD** |
| 9 | Detection | IoU / mAP, NMS | `object-detection` | LIVE |
| 9 | | bbox loss (L1/L2/smooth-L1 gradients) | `bbox-loss` | **TO BUILD** |
| 9 | | anchor assignment (pos/ignore/neg by IoU) | `anchor-assignment` | **TO BUILD** |
| 10 | Segmentation | U-Net, segmentation losses | `unet`, `image-segmentation` | LIVE |

**First-10 to-build backlog (priority order):**
1. **L1 ×4** (`likelihood-to-loss`, `softmax-cross-entropy`, `bayes-classifier`, `map-regularization`) — the opener has *zero* live interactives; highest priority.
2. L3A ×3 (`derivative-tangent`, `surface-partials`, `jacobian-local-map`).
3. L8 ×2 (`pooling-visualizer`, `padding-stride`), L9 ×2 (`bbox-loss`, `anchor-assignment`).
4. L7 ×2 (`weight-decay`, `augmentation-mixup`), L8B ×2 (`inception-cost`, `transfer-learning-playground`), L2 ×1 (`mlp-decision-boundary`).

## Lectures 11–25 (summary)

Mostly **LIVE** via existing articles — `attention`, `positional-encoding`,
`softmax-temperature`, `bpe-merges`, `in-context-learning`, `vision-transformer`,
`clip-zero-shot`, `vlm`, `vision-ssl`, `vae-latent-explorer`, `elbo-decomposition`,
`gan-minimax-dance`, `diffusion-denoise`, `cfg-scale-visualizer`, `text-diffusion`,
`kv-cache`, `quantize-prune`, `mixture-of-experts`, `knowledge-distillation`,
`lora-adapter`, `double-descent`. A handful of `(to build)` widgets remain
(e.g. RNN/BPTT unroll, causal-conv, seq2seq teacher-forcing/beam, tiny-ViT,
decoding-lab) — lower priority than the first-10 backlog above.

## Existing articles worth reusing directly
`bayesian-posterior` and `mle-map-coin` (for L1's Bayes/MAP intuition),
`multivariate-normal` (L1 MVN), `softmax-temperature` (L1 softmax) — these can
back some L1 slides immediately while dedicated widgets are built.
