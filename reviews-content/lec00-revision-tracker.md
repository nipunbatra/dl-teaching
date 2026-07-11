# lec00 Revision Tracker — instructor feedback 2026-06-26

Status legend: ⬜ todo · 🔄 in progress · ✅ done

## Global / factual
- ✅ G1 · Rename previous course **ES 654 → ES 335** everywhere (all decks + course-design.md + lecture-plan*.md + TEACHING_GUIDES.md)
- ⬜ G2 · "You derived BCE from Bernoulli MLE" is wrong — in ES 335 they were **told** the result, never derived it. Reword lec00 bridge row + keypoint: ES 335 *stated* CE; today we *derive* it.

## lec00 slide-by-slide — ALL DONE ✅ (built to PDF, every slide spot-checked for overflow)
- ✅ 1 · Bridge slide — G1 + G2 wording (ES 335; "told, not derived").
- ✅ 2 · Linear regression model — linreg_frequentist.svg added.
- ✅ 3 · Random variable — split: first-year Ω→ℝ definition + figure, then the ~ view.
- ✅ 4 · IID assumption — split: iid_matrix picture/examples first, then formal.
- ✅ 5 · IID when it fails — split: "why it matters" + "when it breaks" with iid_fails.svg + notebook.
- ✅ 6 · Bernoulli examples expanded to a list.
- ✅ 7 · Bernoulli coin — clearer intro.
- ✅ 8 · Categorical — split die / one-hot; one-hot worked with K=4 example.
- ✅ 9 · Categorical worked example — mnist_categorical.svg.
- ✅ 10 · Normal bell curve — normal_bellcurve.svg.
- ✅ 11 · Normal "why everywhere" — kept 1 overview slide; CLT/max-entropy/closed-linear moved to Appendix.
- ✅ 12 · Conditional view — conditional_view.svg.
- ✅ 13 · Reorder — Bayes-in-ML slide before Likelihood; Sampling moved to Part 4 (after MLE).
- ✅ 14 · Two problems w/ likelihood — notebook ref added.
- ✅ 15 · Linreg-MLE picture — overflow fixed (figure shrunk, text trimmed).
- ✅ 16 · Logistic MLE assumption — logistic_mle_picture.svg.
- ✅ 17 · Multiclass — split softmax/assumption; color-coded (slate logits, green probs, rust truth). \textcolor verified to render in MathJax.
- ✅ 18 · P2 → in-class problem + solution slides; practice list renumbered P1–P4.
- ✅ 19 · This tracker file.

## Figures — ALL GENERATED + visually verified via rsvg-convert (diagrams/lec00_figures.py)
- ✅ linreg_frequentist.svg (#2)
- ✅ random_variable_map.svg (#3)
- ✅ iid_matrix.svg (#4)
- ✅ iid_fails.svg (#5)
- ✅ mnist_categorical.svg (#9)
- ✅ normal_bellcurve.svg (#10)
- ✅ conditional_view.svg (#12)
- ✅ logistic_mle_picture.svg (#16)
- (keep existing mle_linear_regression.svg for #15)
- NOTE: qlmanage clips wide SVGs — use `rsvg-convert` to verify figures, not qlmanage.
- NOTE: multi-panel matplotlib figs need `layout="constrained"` + `save(tight=False)` or right axes get clipped.

## Build / validate
- ⬜ `make lec00-html`, zero errors, all image paths exist, visual spot-check of new figures.
