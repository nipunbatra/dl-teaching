# Lecture 5 notebooks

The three canonical companions for the current public Lecture 5 are:

1. [`00_full_batch_sgd_minibatch.ipynb`](00_full_batch_sgd_minibatch.ipynb) — **Batch/SGD/Minibatches + Estimators**
2. [`02_ewma_momentum.ipynb`](02_ewma_momentum.ipynb) — **EWMA + Momentum**, including the standardized-but-correlated control
3. [`03_adaptive_optimizers_constraints.ipynb`](03_adaptive_optimizers_constraints.ipynb) — **RMSProp + Adam + Schedules + Constraints**

The public [EWMA + momentum explorer](https://nipunbatra.github.io/dl-teaching/interactives/ewma-explorer.html)
uses the same fixed-seed temperature signal, first-reading initialization, and
computed ravine as Notebook 02. It exposes both the smoother and the optimizer
state one update at a time.

The public [learning-rate schedule explorer](https://nipunbatra.github.io/dl-teaching/interactives/lr-schedule-explorer.html)
uses the same Example A and seed-12 sampled-index stream as Notebook 03. Its two
views separate early travel from late settling without changing the sampled
examples.

Keep these three synchronized with the Typst deck, course page, and teaching
guide. Run each notebook from a fresh kernel and keep its outputs when preparing
the public materials.

## Retained legacy companions

- [`01_gd_quadratic.ipynb`](01_gd_quadratic.ipynb)
- [`06_optimizer_comparison_pytorch.ipynb`](06_optimizer_comparison_pytorch.ipynb)

The legacy notebooks remain at their existing URLs for provenance and old links,
but they are not part of the canonical Lecture 5 route. Do not use them to infer
the current slide order, examples, notation, or recommended classroom timing.
