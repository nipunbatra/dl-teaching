# ES 667 Course Design — Making It Land for Post-ML Undergrads

**Audience:** undergrads who just finished ES 654 (ml-teaching). They know: gradient descent from Taylor series, MLPs + autograd, basic CNN/RNN mechanics, MLE → cross-entropy for logistic regression, bias–variance, ridge/lasso.

**Design goal:** amazing + accessible + sustainable. The content is ~80% built (26 decks, 15 notebooks, 30+ interactives). What remains is *calibration*, not creation.

---

## Pillar 1 — Spine vs. Landscape

Split the 25 lectures into two contracts with the students:

| | Lectures | Student contract | Assessment |
|---|---|---|---|
| **Spine** | L00–L14 (probability → transformer → tokenization) | *Mastery*: can derive, compute by hand, implement | ~80% of weight: quizzes 1–3, HW1–HW3 |
| **Landscape** | L15–L24 (LLMs → diffusion → frontier) | *Literacy*: can read a 2026 paper/blog and place it; knows the one key idea + one worked numeric per topic | ~20%: quiz 4 (lighter), project |

**Why:** the content audit says L15–L24 are "survey-like" and should be deepened. For this audience, survey-with-one-worked-numeric is the *right* depth — undergrads who can derive the transformer and explain "why CFG works" in one sentence are ahead of most grad students. Deepening 10 frontier lectures is the wrong place to spend instructor effort. **Resolve the audit by reframing the goal of those lectures, not rewriting them.**

State the split explicitly to students in L01 ("you will *master* the first half and become *fluent* in the second"). It lowers anxiety and tells them where to spend study time.

---

## Pillar 2 — Bridge slides: spend the continuity capital

Students arrive with shared examples and results from ml-teaching. Use them.

Add one **"You already know this"** slide at the top of each early lecture, citing the *exact* ML-course result:

- **L00:** "In ML you derived cross-entropy for logistic regression from Bernoulli MLE. Today that one trick becomes the organizing principle of the whole course."
- **L01:** open with the apples-vs-oranges logistic example they know → show it shattering on raw pixels → representation learning in one sentence.
- **L02:** "You proved gradient descent from Taylor series. Now: what happens to that gradient through 50 layers?"
- **L04–L05:** "You know SGD. Here is the family tree it spawned."
- **L06:** "Ridge = Gaussian prior — you saw this as L2. Now meet the regularizers that have no penalty term."
- **L07:** "You did convolution + pooling mechanics in ML. We start from *why* those choices, not *what* they are."

Reuse ml-teaching's recurring datasets on first contact (apples/oranges, height-weight, iris, MNIST), then graduate to CIFAR-10 / Tiny Shakespeare. Familiar data makes the new *concept* the only new thing on the slide.

---

## Pillar 3 — Example-driven: one worked numeric per lecture, no exceptions

This rule already holds for L00–L14 and L19–L22. Four lectures are missing theirs — these are the **highest-ROI content additions in the whole course** (2–4 slides each):

1. **L15 (LLMs):** compute Chinchilla-optimal tokens for a 1B-param model by hand (N ≈ D/20); compare to what GPT-3 actually used → "that's why Chinchilla mattered."
2. **L16 (Alignment):** DPO loss computed on a toy 2-completion example with made-up logprobs — watch the preferred completion's probability rise.
3. **L18 (VLMs):** CLIP zero-shot classification with 2 images × 3 text classes — actual 3-dim embeddings, dot products, softmax, answer.
4. **L23 (Efficient inference):** KV-cache size for a 7B model at 4k context, by hand (layers × heads × d × 2 × seq × bytes) → "this is why your GPU runs out of memory before compute."

Make the two existing **code threads** visible to students: a "thread map" slide at each module boundary — vision thread (MLP-MNIST → CNN-CIFAR → ResNet-transfer → ViT/CLIP → diffusion) and language thread (micrograd → MLP → transformer → nanoGPT → LoRA) — with a "you are here" marker. Students seeing the same artifact grow across 10 weeks is what makes a course feel designed rather than assembled.

---

## Pillar 4 — Difficulty valves (for the students)

The two cliffs for this audience, and the fix for each:

- **L12–L13 (attention/transformer):** already well-engineered (soft dictionary lookup, by-hand notebook, built-live). No change.
- **L19/L21 (ELBO, diffusion math):** demote the full ELBO derivation and the score-matching/Langevin connection to ⭐⭐⭐ (optional). Keep exactly one derivation per lecture that students must reproduce: reparameterization trick (L19), closed-form q(x_t|x_0) (L21). Teach diffusion primarily through the noise-guessing game + the 2D DDPM notebook.

**General rule: one reproducible derivation per lecture.** Everything else is shown for plausibility ("here's why you'd believe this"), clearly marked. This is mostly a re-starring exercise in TEACHING_GUIDES.md, not a rewrite.

---

## Pillar 5 — Sustainability (for the instructor)

**The only genuinely missing piece is `labs/`.** Do not write new content — convert existing notebooks into 4 assignments + a project:

| HW | After | Built from | Core task |
|---|---|---|---|
| HW1 | L05 | `01-micrograd-mlp`, `05-optimizers` | Extend micrograd, train MLP, run the optimizer race, explain the curves |
| HW2 | L09 | `07-cnn-shape-tour`, `03-debug-ladder` | CNN on CIFAR-10 + transfer learning; includes a *sabotaged* training script students must debug with the ladder |
| HW3 | L14 | `12-attention-by-hand`, `13-nanogpt` | BPE by hand + train nanoGPT on Tiny Shakespeare; sample at 3 temperatures and explain |
| HW4 | L22 | `19-vae-mnist`, `21-ddpm-2d` | VAE on MNIST + DDPM on 2D toy; compare samples, explain the KL term's role |
| Project | L15–L24 window | — | Spine-level rigor, landscape-topic freedom (fine-tune with LoRA, build a tiny CLIP, speculative decoding demo…) |

The HW2 "debug the sabotaged script" idea turns the debug ladder — the course's most practically valuable content — into an assessed skill, and it's nearly free to create (take a working script, break it three ways).

**Other bounded-effort, high-leverage additions:**

- **"What breaks" slide template** for spine lectures: *symptom → suspect → test* (loss flat → LR or init → overfit one batch). One recurring visual format, reused; the audit asked for this and it's cheap.
- **Quizzes:** generate drafts from lecture decks via the make-quiz pipeline; hand-curate. 4 quizzes, best 3, as planned.
- Explicitly **decline** the audit's call to deepen L15–L24, add Bishop-style flows/EBMs, or expand citations beyond hedging the "as of 2026" claims ("as of this writing…"). Out of scope for this audience and this semester.

---

## Pillar 6 — The insight ledger (the "amazing" part)

End every lecture with its **one-sentence takeaway**, and collect all 25 on a single page (site + printable). This becomes the course's identity and the best study artifact:

- L00: *Every loss function is a negative log-likelihood in disguise.*
- L00B: *Every regularizer is a prior in disguise.*
- L01: *Deep learning = learning the representation instead of engineering it.*
- L02: *Depth buys exponential feature reuse — if gradients survive the trip.*
- L03: *If you can't overfit one batch, nothing else matters.*
- L04: *Momentum is a moving average of gradients.*
- L05: *Adam gives every parameter its own learning rate.*
- L06: *The best regularizers don't look like penalties.*
- L07: *A convolution is a linear layer that confesses its assumptions.*
- L08: *Skip connections let layers learn corrections instead of replacements.*
- L12: *Attention is a differentiable dictionary lookup.*
- L13: *The transformer is five ingredients, none of them new.*
- L14: *LLMs see tokens, not letters — half their bugs start there.*
- L15: *Scale changed, the architecture didn't.*
- L21: *Learn to denoise; generate by iterating.*
- L23: *Inference is memory-bound, not compute-bound.*
- … (complete during semester)

---

## Execution order (priority)

1. **Build `labs/`** — HW1–HW4 specs from existing notebooks + project rubric. *(the real gap)*
2. **Bridge slides** in L00–L07. *(one slide each)*
3. **Four worked numerics** — L15, L16, L18, L23. *(2–4 slides each)*
4. **"What breaks" template slide** rolled across spine lectures.
5. **Insight ledger** page + closing slide per deck.
6. **Re-star VAE/diffusion math** to ⭐⭐⭐ in TEACHING_GUIDES.md.

Items 2–6 are each bounded at hours, not days. Item 1 is the only multi-day effort, and it reuses existing notebooks.
