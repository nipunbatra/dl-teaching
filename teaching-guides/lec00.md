# Lecture 0 · A Probabilistic View of ML · Teaching Guide
*First-time-instructor companion. Audience: undergrads, one ML course.*

## The spine (say this in 2 sentences)
Every loss they already use is secretly the same object: the negative log-likelihood of a distribution the model puts over `y`. Pick the distribution to match the output type (Normal/Bernoulli/Categorical) and the loss (MSE/BCE/CE) falls out automatically — no more memorizing a loss per task.

## Where it sits
Builds on … ES 335 (they wrote MSE/BCE, ran SGD, never asked where the loss came from) · first-year probability (random variables, conditional `P(A|B)`).
Sets up … L00B (add a prior → MAP → L1/L2), L00C (the information-theory meaning of `-log p`), and L01 (logistic/softmax classification are just 1-layer nets — same NLL, deeper `p_θ`).

## 80-minute plan
- ⭐ Core (~45 min): the three distributions (Bernoulli/Categorical/Normal) and `~` notation · the compact form `p^y (1-p)^{1-y}` · IID → product → log → sum · likelihood vs probability · the 3-step MLE recipe · MLE for the coin → `p̂ = #H/N` · MLE for linear regression → **MSE** · MLE for logistic → **BCE**. Every student must leave knowing "loss = NLL of a chosen distribution."
- ⭐⭐ Should-cover (~25 min): multiclass softmax + categorical CE · the `ŷ − y` gradient pattern shared by all three · the BCE worked numeric · the pop-quiz "which course is the student from?" (MLE intuition) · Laplace noise → MAE.
- ⭐⭐⭐ Optional / skip-if-tight: the entire Appendix on **why the Normal is everywhere** (CLT, max-entropy, closed-under-linear) — one-line gist if asked: "three independent reasons the Gaussian dominates: sums of many small effects look Gaussian (CLT), it's the least-committal choice given mean+variance (max-entropy), and it stays Gaussian under sums/affine maps (the reason diffusion has a closed form)." Plate notation can be compressed to one slide if time is short.
**Do live on the board:** the **single-feature linear-regression MLE** (slide "In-class problem"): write the Gaussian density → take log → sum over N → drop constants → differentiate → set to 0 → `θ̂ = Σxᵢyᵢ / Σxᵢ²`. It lands because it ends on the exact OLS formula they already know, proving the probabilistic story produces, not replaces, their intro-ML answer.

## Teach it like this
Open with the four mysteries slide (MSE, BCE, L2, L1 — "where do these come from?") and promise that one principle dissolves two of them today. Then do the smallest possible case first — the coin — all the way through to `p̂ = #H/N`, so they see the whole recipe on a problem with no linear algebra. Only then generalize: the *same three steps* on a Gaussian give MSE, on a Bernoulli give BCE. The payoff to say out loud at the end: "you have not learned a new loss today — you have learned the factory that makes every loss."

## Heads-up for YOU (subtle points to get right)  ← most important section
- **Likelihood is NOT a probability over θ.** It is `P(D | θ)` read as a function of θ with the data fixed. Do not call it "the probability of θ." The probability *of* θ is the posterior — that needs a prior, which is next lecture. (A student may push: "but it's `arg max` over θ, isn't that a distribution over θ?" — No: it's not normalized in θ and need not integrate to 1.)
- **Likelihood can exceed 1; log-likelihood can be positive.** For continuous `y`, `p(y|θ)` is a *density*, not a probability — densities can be >1 (a narrow Gaussian peaks above 1). So a positive log-likelihood is not a bug. Only relative values / the `arg max` matter.
- **MSE is NLL of a Gaussian *with σ² assumed constant and known*.** The full per-example NLL is `(y−ŷ)²/(2σ²) + ½log(2πσ²)`. You drop the second term and the `1/(2σ²)` factor *only because σ² is constant in θ*. If a model also predicts variance (heteroscedastic), the `log σ(x)` term comes back and you can't drop it — worth saying so the "we just throw away constants" move doesn't look like hand-waving.
- **The softmax+CE gradient `ŷ − y` is the gradient w.r.t. the LOGIT `z_k`, not the weights.** State it as `∂L/∂z_k = π̂_k − y_k`. The weight gradient is then `(π̂ − y) xᵀ` via the chain rule. If you say "the gradient is ŷ − y" without "of the logit," a sharp student will catch that it can't be — it has the wrong units/shape for a weight.
- **The MLE for variance is biased.** If asked "is MLE always the right estimator?": the Gaussian MLE for σ² divides by `N` and *underestimates* the true variance; the unbiased estimator divides by `N−1` (Bessel). MLE is consistent and asymptotically efficient, but not automatically unbiased. (You don't need to teach this — just don't claim MLE is unbiased.)
- **"Why squared, not absolute or cubed?"** is answered by the *noise model*, not by aesthetics. Gaussian noise → squared error; Laplace noise → absolute error (MAE). Don't answer "because squared is differentiable / penalizes big errors" as the *fundamental* reason — that's the consequence, the noise assumption is the cause.

## Where students stumble (and the fix)
- **Confusing the compact form's exponent trick.** Why `p^y (1−p)^{1−y}`? Walk the two cases live (`y=1 → p`, `y=0 → 1−p`); the exponent that is 0 collapses its factor to 1. Then show the product over N gives `p^{#H}(1−p)^{#T}` — "only counts matter."
- **Thinking "the model outputs a number."** Repeat the reframe: the model outputs the *parameters of a distribution* over `y` (the mean of a Gaussian, the `p` of a Bernoulli, the `π` of a Categorical). The prediction is a by-product (the mean/mode).
- **Mixing up `arg max LL` and `arg min NLL`.** They are the same point — negation flips max↔min, the location of the optimum doesn't move. Say it explicitly; libraries minimize, so we negate.
- **Why take the log at all.** Two reasons, both concrete: products underflow (`0.5^2000 ≈ 10^−602` < smallest float `≈10^−308` → literally 0.0), and log is monotonic so the `arg max` is unchanged. Show the underflow number — it makes it visceral.

## If a student asks…
- "We maximize `P(D|θ)` but we *want* `P(θ|D)` — isn't that backwards?" → Yes, you want the posterior. MLE is the special case of Bayes with a *flat (uniform) prior*, where posterior ∝ likelihood. We add the prior back next lecture; that's MAP.
- "Why assume Gaussian noise for house prices when prices are positive and skewed?" → For a generic continuous target the Gaussian is the least-committal choice given mean+variance (max-entropy). If the data is clearly skewed/heavy-tailed, a Gaussian *is* a bad model — practitioners switch to Laplace (→ MAE, robust to outliers) or model `log(price)`. Loss choice = noise model.
- "What if my output isn't Bernoulli/Categorical/Gaussian?" → Pick the distribution that fits: Poisson for counts, Beta for fractions, mixture-of-Gaussians for multimodal. The recipe (NLL = loss) is universal; only the distribution changes.
- "Is the MLE always the empirical frequency / sample mean?" → For the coin yes (`#H/N`), and for a Gaussian mean it's the sample average — but not in general (e.g. variance MLE is biased; the logistic-regression MLE has no closed form). The recipe is general; the closed form is a luxury of simple models.
- "Why does the logistic-regression gradient look exactly like the linear-regression gradient (`(ŷ−y)x`)?" → Both are *generalized linear models*; the `prediction − truth` form is a structural feature of GLMs trained by MLE, not a coincidence.
- "Is the coin example really independent? Real flips might not be." → IID is an *assumption* that licenses the product factorization. Shuffle a dataset and the summed loss is unchanged — that's independence at work. It breaks for time series / video; the demo notebook shows `sum(log p)` is blind to autocorrelation, which is exactly the failure.

## If you're short on time
Cut: the whole Normal-everywhere Appendix (CLT/max-entropy/closed-under-linear), plate-notation details, and the Laplace→MAE slide. Never cut: the **coin MLE → `#H/N`** and the **linear-regression MLE → MSE** derivations — those two *are* the lecture; everything else is variations on them.

## Closing line
"You didn't learn a new loss today — you learned the one principle that *manufactures* every loss: write down the distribution your model puts over `y`, and minimize its negative log-likelihood."
