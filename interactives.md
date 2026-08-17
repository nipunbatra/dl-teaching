# Interactives — per-lecture status

Every deck's `[I]` slides point at a browser explainer hosted in the
**[interactive-articles](https://github.com/nipunbatra/interactive-articles)**
repo (`~/git/interactive`), deployed at
`https://nipunbatra.github.io/interactive-articles/<slug>/`.

- **LIVE** — the article exists and the deck links it via `#interbox(link-to: …)`.
  As of this cycle **every `[I]` slide is LIVE** (see the status section below); the
  `(to build)` backlog is empty.

New articles follow the repo house style: single self-contained `index.html`
(warm-paper palette `--bg #fdfcf9`, blue `--accent #2c6fb7`, orange `--warm #d9622b`;
Source Serif 4 / Manrope / IBM Plex Mono; KaTeX from CDN; a canvas/SVG widget with
live controls) + a `meta.json` (slug, title, tagline, summary, difficulty, tags, highlights).

---

## First 10 lectures

| # | Lecture | Interactive | Slug | Status |
|---|---------|-------------|------|--------|
| 1 | Why These Losses? | likelihood → loss (noise density → −log p → fitted line; Gaussian/Laplace/Uniform/Student-t + outlier) | `likelihood-to-loss` | LIVE |
| 1 | | softmax → cross-entropy (logits, true class, temperature → p → −log p_y → p−y) | `softmax-cross-entropy` | LIVE |
| 1 | | Bayes classifier (two Gaussians, prior, test point → posterior + boundary) | `bayes-classifier` | LIVE |
| 1 | | MAP = likelihood × prior (contours, MLE/MAP/0, Gaussian vs Laplace) | `map-regularization` | LIVE |
| 2 | Linear → MLP | linear-regression & GD, activations, softmax | `optimizer-race`, `softmax-temperature`, `vanishing-gradients` | LIVE |
| 2 | | universal approximation | `universal-approximation` | LIVE |
| 2 | | [ReLU Classification Lab](https://nipunbatra.github.io/interactive-articles/mlp-decision-boundary/) (datasets and custom points; constructed/random/trained models; logit/probability/class; signed contributions; movable 3-D surface) | `mlp-decision-boundary` | LIVE |
| 2 | | [ReLU Approximation Lab](https://nipunbatra.github.io/interactive-articles/relu-function-approximation/) (construction/training lanes; targets and custom drawing; knots and width; residual, grid metrics, and signed terms) | `relu-function-approximation` | LIVE |
| 3A | Calculus Toolkit | gradient field & descent, curvature | `optimizer-race` | LIVE |
| 3A | | tangent line / local linear approx | `derivative-tangent` | LIVE |
| 3A | | surface slices (∂/∂x, ∂/∂y) | `surface-partials` | LIVE |
| 3A | | Jacobian as a local warp (grid → ellipse) | `jacobian-local-map` | LIVE |
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
| 7 | | weight decay (λ sweep → norm & boundary) | `weight-decay` | LIVE |
| 7 | | augmentation & mixup (2-D boundary) | `augmentation-mixup` | LIVE |
| 8 | CNNs | convolution playground, padding/stride | `convolution-visualizer` | LIVE |
| 8 | | receptive field | `receptive-field-grower` | LIVE |
| 8 | | pooling (max/avg on a grid) | `pooling-visualizer` | LIVE |
| 8 | | padding & stride → output size | `padding-stride` | LIVE |
| 8B | Modern CNNs | receptive field, residual block | `receptive-field-grower`, `resnet` | LIVE |
| 8B | | Inception cost (1×1 bottleneck savings) | `inception-cost` | LIVE |
| 8B | | transfer-learning (freeze/finetune curves) | `transfer-learning-playground` | LIVE |
| 9 | Detection | IoU / mAP, NMS | `object-detection` | LIVE |
| 9 | | bbox loss (L1/L2/smooth-L1 gradients) | `bbox-loss` | LIVE |
| 9 | | anchor assignment (pos/ignore/neg by IoU) | `anchor-assignment` | LIVE |
| 10 | Segmentation | U-Net, segmentation losses | `unet`, `image-segmentation` | LIVE |

## Status: every `[I]` slide is now wired to a live widget ✅

The backlog is empty — there are **no `(to build)` interactives left** in any deck.
Built this cycle (all CDP-verified in Brave via `interactive-articles/scripts/verify-interactive.mjs`):

- **L1 ×4** — `likelihood-to-loss`, `softmax-cross-entropy`, `bayes-classifier`, `map-regularization`
- **L2 ×2** — `mlp-decision-boundary` (classification) and
  `relu-function-approximation` (construction versus training)
- **L3A ×3** — `derivative-tangent`, `surface-partials`, `jacobian-local-map`
- **L7 ×2** — `weight-decay`, `augmentation-mixup`
- **L8 ×2** — `pooling-visualizer`, `padding-stride`
- **L8B ×2** — `inception-cost`, `transfer-learning-playground`
- **L9 ×2** — `bbox-loss`, `anchor-assignment`
- **L11 ×2** — `embedding-lookup`, `makemore-forward`
- **L12 ×3** — `rnn-state-stepper`, `bptt-unroll`, `gru-gates`
- **L13 ×2** — `causal-conv`, `gated-residual`
- **L15 ×2** — `transformer-block`, `decoding-lab`
- **L17 ×1** — `shifted-window` · **L19 ×1** — `vq-codebook` · **L21 ×1** — `classifier-guidance`

The rest of L14–L24 stay LIVE via existing articles (`attention`, `positional-encoding`,
`bpe-merges`, `in-context-learning`, `vision-transformer`, `clip-zero-shot`, `vlm`,
`vision-ssl`, `vae-latent-explorer`, `elbo-decomposition`, `gan-minimax-dance`,
`diffusion-denoise`, `cfg-scale-visualizer`, `text-diffusion`, `kv-cache`,
`quantize-prune`, `mixture-of-experts`, `knowledge-distillation`, `lora-adapter`, `double-descent`).

## Verifying interactives (CDP-Brave harness)
`node interactive-articles/scripts/verify-interactive.mjs <slug>` drives Brave via
puppeteer/CDP: it flags JS/KaTeX errors, blank canvases, and unrendered `$…$`, and
exercises every control. Exit 0 = pass. Use this (the Claude-in-Chrome extension is
not connected on this machine).
