# ES335 (ML) → ES667 (DL) — Continuity Brief

Source: deep read of `~/git/ml-teaching` polished decks (linear-regression, logistic-regression, rnn) + `shared/notation/notation.sty` + `theme-nipun.sty`, reconciled with the current DL decks. Goal: make ES667 read as a **seamless sequel** to ES335.

## 1. Voice & rhythm (match this)
**Intuitive problem → visual failure → mathematical fix → explicit derivation.** Never math-first. Motivate every new tool with a *broken* toy example, then fix it.
- Pop quizzes (Q first, answer revealed next click) at the end of sections. DL already has `popquiz` divs — keep them.
- Line-by-line derivation reveal (ES335 uses `\pause`/`<+->`; in Marp, build with successive bullets / split slides).
- Decks **close with "Practice and Review"** (worked problems). DL ends on a one-sentence takeaway — keep, but ensure ≥1 worked problem near the end.

## 2. Notation — reuse EXACTLY (ES335 canonical)
| Thing | Symbol | Notes |
|---|---|---|
| Parameters (generic) | **θ** (bold) | ES335 uses θ everywhere (not w/b). DL already matches (550×). |
| Layer weights (nets) | W_{hh}, W_{xh}, b_h | matrices once we have layers |
| Inputs | **X** (design matrix), **x**_i (one sample) | bold |
| Prediction | **ŷ**, ŷ_i | ES335 uses ŷ heavily; **DL underuses it (1×) → prefer ŷ** |
| True label | **y**, y_i | |
| Dataset | 𝒟 | calligraphic |
| Cost / loss | **J(θ)** | ES335 always J(θ) = NLL; **DL uses 0× → introduce loss as J(θ)**, then may abbreviate |
| Likelihood | P(𝒟\|θ) or 𝓛𝓛(θ) | |
| Sigmoid | σ(z) = 1/(1+e^{−z}) | |
| #samples / #features / #classes | N or n / d (or M) / K | |

**Reconciliation actions:** when a deck first introduces the training objective, call it **J(θ)** and say "the cost, as in ES335." Use **ŷ** for predictions. Keep θ.

## 3. Shared running examples — CALL BACK to these
Anchoring DL concepts to ES335 memories is the single biggest "sequel" win.
- **Oranges vs Tomatoes** — canonical binary classification (by Radius). Reuse for: logistic→MLP decision boundary, CE loss.
- **10 coin flips** `(H,H,T,T,T,H,H,T,T,T)` → p(H)=4/10 — canonical MLE. Reuse in L00 MLE/Bernoulli.
- **[0,1,2,3] toy linear set** X=[0,1,2,3]ᵀ, y=[0,1,2,3]ᵀ, broken by an outlier at (4,0) — canonical matrix-math + robustness. Reuse for loss/optimization.
- **IITGN Water Demand** = f(#occupants, temperature) — multivariable regression.
- **Delhi Pollution** (wind dir N/E/W/S) — categorical / dummy-variable trap.

## 4. Exact handoff points (RECAP in ≤1 slide, then go deeper — do NOT re-teach)
- **MLE / Bernoulli:** they derived θ_MLE = n_h/(n_h+n_t) from the 10 coin flips, and the logistic likelihood ∏ σ(xᵢᵀθ)^{yᵢ}(1−σ)^{1−yᵢ}. → DL generalizes MLE to any model.
- **Cross-entropy / logistic cost:** they have J(θ) = −Σ[yᵢ log ŷᵢ + (1−yᵢ)log(1−ŷᵢ)] and K-class categorical CE with one-hot, and know MSE is non-convex for classification. → DL reuses CE to train deep nets.
- **Gradient descent:** they know ∂J/∂θ_j = Σ(σ_θ(xᵢ)−yᵢ)xᵢʲ, and saw Newton/IRLS. → DL: backprop computes these gradients; SGD/Adam scale them.
- **Linear model ŷ=Xθ + basis expansion φ(x)=[1,x,x²]:** they know "linear in θ, not features," and the projection/geometry. → DL: hidden layers *learn* the basis φ instead of fixing it.
- **RNN/LSTM:** they saw h_t = tanh(W_hh h_{t−1} + W_xh x_t + b_h) and *conceptually* that W_hh>1 explodes / <1 vanishes; 3 LSTM gates by name. They have **not** seen rigorous BPTT or gate math. → DL: derive BPTT, derive gates.

## 5. Ready bridge sentences (open lectures with these)
1. **(L00/00B/00C, L01)** "In ES335 you used cross-entropy J(θ)=−Σ[yᵢ log ŷᵢ+(1−yᵢ)log(1−ŷᵢ)] to classify oranges vs tomatoes; today we generalize that same cost to train deep multi-layer networks."
2. **(L01)** "In ES335 you captured non-linear boundaries with *fixed* basis functions φ(x)=[1,x,x²]; today we show how hidden layers *learn* their own basis automatically."
3. **(L10)** "In ES335 you saw simple RNNs vanish/explode via repeated W_hh; today we unroll the net and derive exactly how BPTT fails — and how LSTM gate math fixes it."

## How to apply (per deck)
1. Opening bridge: one "From ES335 →" line/slide using the matching §5 sentence or §4 handoff.
2. Weave ≥1 §3 running-example callback where natural.
3. Enforce §1 voice on the lecture's headline concept (problem→failure→fix→derivation).
4. Harmonize notation per §2 (ŷ, J(θ), θ) — light touch, don't break math.
5. Build the deck (no overflow), keep one-sentence takeaway + ≥1 worked problem.
