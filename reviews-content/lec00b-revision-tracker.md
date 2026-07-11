# lec00b Revision Tracker — instructor feedback 2026-06-26

Status: ⬜ todo · ✅ done

## Items
- ✅ 1 · Bayes is now in L00 → make PART 1 here a **refresher** (light), not full intro. Frame conditional-prob/Bayes-formula slides as recap.
- ✅ 2 · Disease-test setup · **explain the terms** (prevalence, sensitivity, specificity) — students don't know them.
- ✅ 3 · Bayesian updating (dynamic view) · explain better + **diagram** (sequential prior→posterior→prior); reference the `bayesian-posterior` interactive.
- ✅ 4 · MAP geometric picture · rework with a **small dataset** · show data objective as **ellipses**, the prior in 2D, the MAP solution, and the effect of **λ** (ridge path). New figure.
- ✅ 5 · Gaussian prior setup · add **drawing**; show what **independence across dims** means (isotropic/diagonal). New figure.
- ✅ 6 · Gaussian prior "stack across all weights" · content cut → **split**.
- ✅ 7 · L2 reading λ / weight-decay · clarify (reduce confusion). Interactive for λ — note as external TODO (interactive-articles repo).
- ✅ 8 · Worked numeric MAP-L2 (loss) · use **actual data** (tiny dataset → real NLL).
- ✅ 9 · Worked numeric MAP-L2 (gradient+update) · explain weight-decay more; **split** (too much on one slide).
- ✅ 10 · Laplace prior setup · **show the distribution/plot first** (Laplace vs Gaussian). New figure.
- ✅ 11 · L1 sparsity geometry · also frame via the **prior's spike at 0** (Laplace peak), not only the ball geometry.
- ✅ 12 · "Every loss is an NLL" master table · make a **poster-worthy image**. New figure.
- ✅ 13 · Clarify the confusing line "log-likelihood works as a model-quality signal · …low per-sample surprise."
- ✅ 14 · Entropy/KL **Huffman** lens · too confusing → remove Huffman slides / references; keep a simpler bits framing.
- ✅ 15 · "Entropy of a Bernoulli · in pictures" · not fitting → split/trim.
- ✅ 16 · KL definition · **intuition first**, then the formula.
- ✅ 17 · Optional content: more intuitive, more examples, break into more slides, slow down, more diagrams. Practice problems · show a **worked subset** on the next slide ("let's do one").

## Figures to generate (diagrams/lec00b_figures.py, same anthropic palette)
- ✅ bayesian_updating.svg (#3)
- ✅ map_geometry_lambda.svg (#4)
- ✅ gaussian_prior_2d.svg (#5)
- ✅ laplace_vs_gaussian.svg (#10)
- ✅ nll_master_poster.svg (#12)
- (existing: beta_binomial_update, map_geometry, l1_l2_geometry, entropy_bernoulli, forward_reverse_kl)

## Verify
- ✅ `make lec00b-html`, build to PDF, spot-check every changed slide for overflow; rsvg-convert to verify figures.
