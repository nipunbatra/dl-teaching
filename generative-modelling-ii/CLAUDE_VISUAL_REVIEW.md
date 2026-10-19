# Generative Modelling II — independent visual and teaching review

Reviewer: Claude (Opus 5.5). Date: 19 October 2026. Nothing in the lecture was modified.

## How this review was done

- **Rendered slides, not reports.** I re-captured all **71 scenes / 191 build states** from the live preview (`http://127.0.0.1:8767/generative-modelling-ii/index.html?present`) at 1600 × 954 with my own Puppeteer script ([claude-review-evidence/capture.cjs](claude-review-evidence/capture.cjs)). I did not rely on `qa/`, `VISUAL_REVIEW.md` or `QA_REPORT.md`.
- **Every reveal was measured.** For each of the 120 build-to-build transitions I computed the pixel difference and its bounding box. This shows which reveals change the figure and which only swap the "Talk through" line for the "Takeaway".
- **Suspected defects were zoomed.** Label collisions, overflows, clipping and crop residue were confirmed on full-resolution crops in [claude-review-evidence/crops/](claude-review-evidence/crops/).
- **Every number was recomputed.** I checked the schedule values, the running coordinate, the reverse step, the posterior, the ELBO weight, attention, CFG, the latent ratios, DiT patch counts and the LLaDA loss. I also checked the displayed code against `scripts/train_mnist_ddpm.py`.
- **Two quality bars were applied.**
  1. *Beyond Attention / From Attention to Applications* (rev. 7.25). I studied its PDF directly, in particular pp. 3–6, 37 and 59.
  2. The feedback you accepted in the **Generative Modelling I review** session: one running example, numbers after every equation, measured outputs that support the claims, a return to the opening question, no permanent chrome, caveats in the notes, 2–3 decimals, cues that do not paraphrase the takeaway, and Lil'Log / Jay Alammar-style mechanism figures.

Each scene has a **builds** image: a grid of every build, labelled `sNN/k` at top-left. It is in [claude-review-evidence/builds/](claude-review-evidence/builds/).

**Verdict scale (against Beyond Attention):** **At** = at the reference standard · **Close** = targeted fixes · **Below** = the figure or its reveals need rework · **Far below** = rethink the scene.

I did not re-count the 814,145 U-Net parameters, because no Python with PyTorch was available in this session. I also did not retrain anything. The Codex MCP server failed to connect, so this review has no Codex second opinion.

---

## 1. Verdict in one page

The mathematics is correct. The schedule, posterior and reverse-step numbers all check out, and the deck contains real evidence: the forward strip, the training tuple, the reverse trajectory with x̂₀ estimates, and the guidance gallery. **But as teaching, it is well below Beyond Attention.** Almost every problem the GM-I review identified, and that you asked to be fixed, has recurred:

| GM-I principle you accepted | Status in GM-II | Evidence |
|---|---|---|
| One running example carried through | **Latent but unused.** One consistent coordinate does run through the deck (x₀ = 1, ε = −0.5, ᾱ₂ = 0.64, x₂ = 0.5, ε̂ = −0.4 in s17 → s19 → s23 → s33 → s35 → s37). It is never named or shown as one thread. It stops exactly where it is needed most: s24, s31, s36 and a04. | builds/s19, s33, s35 |
| Every equation followed by its numbers | **No** for s16, s18, s30, s31, s36, s42–s48 and a01–a04 | builds/s18, s31 |
| Measured outputs support the claims | **Partly.** s05, s36 and s53 make claims the figure does not show or contradicts. | §3 |
| Return to the opening examples | **No.** s60 is a 3 × 4 grid of text boxes with no images. s54 revisits 2 of the 8 openers. | builds/s60 |
| No permanent header/footer chrome | **No.** Every slide has an eyebrow, a footer ("Primary source ↗ · Course map · s17 · 3 builds · ES 667"), the toolbar and a two-line Talk-through/Takeaway box taking about 20% of the height. | any builds image |
| Caveats in notes, not on the canvas | **No.** 21 scenes print provenance or caveat sentences inside the figure, and 5 takeaways are hedges ("…does not…"). | §6 |
| 2–3 decimals | **No** in demos: 0.708088, 0.111111, 0.802224, 0.021368, 0.285200 | s20, s22, s37, s49 |
| Maths typeset properly | **No** inside SVGs: εθ(xt, t), ᾱ, εcfg = εu + s(εc − εu), "Q = HWQ", "QKᵀ / √dk" | s21, s28, s48, s52 |
| One idea per reveal | **No.** 5 reveals are pixel-identical to the previous state. 15 more change only the cue box. So 20 of 120 transitions (17%) show nothing new. | §4 |
| Cues that help you explain | **No.** The Talk-through sentence is the same for every non-final build of a scene, by runtime design: `apply()` in `src/lecture.js` picks one `talk_through` per scene. A four-build derivation shows the same sentence under four different figures. | s18, s30, s35, s37 |

**What already works and should be kept:**
- s04 (five RePaint fillings).
- s12 and s39, the real forward and reverse trajectories. The x̂₀ row in s39 is the best figure in the deck.
- s19's cancellation example.
- s17's bar picture of ᾱ.
- s33 and s35's arithmetic chain.
- s22's measured training tuple.
- s44's "64× positions vs 48× entries" ratio.
- s49's attention arithmetic.
- s56's renumbered vocabulary.

**Ratings across the 71 scenes:**

| Rating | Count | Scenes |
|---|---:|---|
| At | 1 | s04 |
| Close | 32 | |
| Below | 33 | |
| Far below | 5 | s24, s31, s36, s58, s60 |

The two weakest areas are the **mathematical heart** (s24 → s31 → s36: why noise MSE, why the network can't know ε, why add noise back) and the **closing** (s54–s60).

---

## 2. The ten highest-impact fixes (ranked)

### 1. Make one pixel of one "7" the spine of the lecture, and show its numbers on every mathematical slide

The deck already has a consistent toy coordinate. Name it ("pixel p of our 7"), put a small persistent "running example" chip beside each derivation, and extend it to the scenes that currently go abstract. Every value below is recomputed and consistent with what the slides already show.

| Scene | What to show | Value | Today |
|---|---|---|---|
| s15 | Stroke pixel of MNIST test 7, uint8 255 → x₀ | **+1** | no value shown |
| s16 | One step, β₁ = 0.2, η₁ = −0.5: x₁ = 0.894·1 + 0.447·(−0.5) | **0.671** | no numbers |
| s17 | ᾱ₁, ᾱ₂ | **0.8, 0.64** | shown ✓ |
| s18 | Variances add: α₂β₁ + β₂ = 0.16 + 0.20 = 1 − ᾱ₂ | **0.36** | no numbers |
| s18 | With η₂ = −0.224 the two-step path lands on the jump value | **x₂ = 0.500** | not shown |
| s19 | 0.8·1 + 0.6·(−0.5) | **0.5** | shown ✓ |
| s23 | Our pixel is the first entry: target −0.5, prediction −0.4 → squared error | **0.01** | shown, never linked |
| s24 | The same x₂ = 0.5 could come from a background pixel x₀ = −1 with ε = +2.17. With equal priors, P(stroke \| x₂) = 0.90, so the MSE-optimal prediction is E[ε \| x₂] | **−0.24** | abstract Gaussians |
| s31 | True posterior mean μ̃ vs model mean μθ: gap = (1/3)(0.1)/0.894. KL = gap²/(2·0.111) = w₂·(ε̂−ε)² with **w₂ = 0.625** | **0.745 vs 0.708; gap 0.037; KL 0.00625** | missing |
| s33 | x̂₀; its error 0.075 = (0.6/0.8)·0.1 is the noise error amplified by √(1−ᾱ)/√ᾱ | **0.925** | value shown; error not |
| s35 | c = β₂/√(1−ᾱ₂); μθ | **1/3; 0.708** | shown ✓ (6 decimals) |
| s36 | β̃₂ = 0.2·0.2/0.36; σ | **0.111; 0.333** | only in s37's readout |
| s37 | z = 0.3 → x₁ | **0.808** | shown ✓ |
| a04 | μ̃ via the x₀ form | **0.745** | formula only |

Pair it with the real schedule wherever MNIST images appear (s12, s20, s26):

| t | 100 | 300 | 350 | 500 | 700 | 920 | 1000 |
|---|---|---|---|---|---|---|---|
| ᾱₜ | 0.897 | 0.396 | 0.285 | 0.079 | 0.0070 | 0.00019 | 0.00004 |

Signal amplitude √ᾱ falls below 0.1 at **t = 674**.

Say once that the toy steps (β = 0.2) are 10× the largest real β, so the arithmetic is visible.

This is exactly what Beyond Attention does with *Raghav goes to school in Delhi*. Every frame, including the NER score table on p37 and the attention weights 0.72 / 0.04 / … on p59, is computed on the same six words.

### 2. Rebuild s31 (and s30) around the one line that turns the bound into noise MSE

This is the lecture's key derivation. Today s31 shows two Gaussians, then three formulas that replace each other. The step that matters is never shown:

μθ = (xₜ − c ε̂)/√αₜ and μ̃ = (xₜ − c ε)/√αₜ, so μ̃ − μθ = c(ε̂ − ε)/√αₜ.

Show it stacked under the KL. Then evaluate it on the running pixel: **0.00625 = 0.625 × 0.1²**. Then plot wₜ = βₜ²/(2σₜ²αₜ(1−ᾱₜ)) against t for the real schedule. It falls from **0.60 at t = 2** to **0.0055 at t = 500**, so L_simple re-weights the terms by roughly 100× and drops the endpoint terms. That plot *is* the "weighting qualification" the brief demands.

s30 promises "The ELBO from Lecture I appears again" but never draws the bridge. Add the side-by-side:

| GM-I VAE | Diffusion |
|---|---|
| z | x₁:T |
| q_φ(z \| x), learned | q(x₁:T \| x₀), fixed, no parameters |
| p_θ(x \| z) | p_θ(xₜ₋₁ \| xₜ) |
| reconstruction + KL | L₀ + Σ Lₜ₋₁ + L_T |

Draw braces from each term to the three boxes. The boxes' current subtext (~11 px) is unreadable.

### 3. Turn s13 → s24 → s06 into one argument: the network can't know the noise, so it predicts an average

- **s24** ("Can the network know the exact sampled noise?") is two unlabelled bell curves with "Schematic uncertainty, not a measured noise distribution". Replace them with the two concrete explanations of x₂ = 0.5 from Fix 1: P = 0.90 / 0.10, giving E[ε] = −0.24. Then show that **neighbouring pixels sharpen this posterior**. That is why the denoiser needs spatial context (a U-Net), and it sets up s40–s41.
- **s13** should show the posterior bars P(7 \| xₜ) and P(2 \| xₜ) under the observed value, and how they flatten as t grows. Its third build currently adds nothing.
- **s06** already contains the visual proof: "Regression" is blurry and the "SR3 sample" is sharp. Call back to it here: "MSE regression returns the average of compatible answers — the blur you saw on s06. Sampling picks one."

### 4. Make every reveal change the figure, and stop deleting derivation lines

20 of 120 transitions change nothing (§4). In s16, s17, s18, s30, s31, s35, s36 and a01–a09, each build *replaces* the equation strip, so the previous line of the derivation disappears. In s18, the variance algebra (b2) is gone by the time the conclusion (b4) appears.

- Keep earlier lines on screen, dimmed, as Beyond Attention does on pp. 3–6.
- Drive the demos from builds, as s49 already does via `goTo`: s14 at t = 0 / 350 / 1000; s20 at three noise levels; s22 with input → target → prediction → error; s37 with mean → noise → sample; s57 with mask → logits → loss.
- Delete builds that exist only to swap the cue.

### 5. Move all chrome and cues off the projected slide, and rewrite the takeaways as claims

This is the GM-I recommendation. It applies again here.

- Remove the eyebrow, the footer metadata and the on-canvas Talk-through/Takeaway box from presentation mode. Keep one short caption with a left rule, as in Beyond Attention (p37, p59).
- Put cues in the Notes, **one per build**.
- Replace the 21 in-figure caveat sentences with a single four-state corner badge: **PUBLISHED** (citation) · **OUR RUN** (checkpoint, steps) · **EXACT TOY** · **SKETCH**. a11 already defines these categories but lives only in the appendix.
- Rewrite the six hedged takeaways (§5). Example for s44: replace "its size ratio is not a runtime measurement" with "Same image, 48× fewer numbers to denoise — then decode once."

### 6. Replace decorative swatches with the real tensors the deck already has

These all draw coloured squares where a real digit or noise field should be:

- s11, s15, s16, s25, s32 (and its duplicate s10), s41, s45, s46.

The assets already exist: `digit.png`, `fresh-gaussian.png`, `batch-7.png` / `batch-350.png` / `batch-920.png`, `new-model-input.png` / `new-model-prediction.png` and `ldm-*.png`.

Two of these mislead, not just decorate:
- **s11** draws the same 3 × 3 "latent" swatch as the start of both the VAE/GAN path and the diffusion path. The crucial difference is that diffusion starts from **image-sized** noise: 784 numbers, latent dimension = data dimension.
- **s25** draws t = 7 and t = 920 with identical-looking icons. One of them is nearly clean and the other is pure noise.

### 7. Answer the question each title asks

| Scene | Title asks | Figure currently shows | Show instead |
|---|---|---|---|
| s21 | Which target should the network predict? | only ε | ε, x₀ and v targets for the same noisy 7, with the conversion (a05) one click away |
| s36 | Why add noise during denoising? | one Gaussian with an arrow | mean-only sampling from the local checkpoint (8 seeds → blurry, near-identical) vs the posterior-variance sampler (8 distinct digits) — affordable with the existing model |
| s43 | Can a transformer predict the noise? | patchify → transformer → unpatchify | add *where t and the condition enter* (adaLN-Zero in DiT); without it, the answer to "how does it know the noise level?" is missing |
| s53 | What changes when we turn up guidance? | rows s = 0/1/3/7, takeaway "larger scale does not guarantee better quality" | the picture says something sharper: at **s = 1 only 2–3 of 8 samples are recognisable sevens**, at s = 3 about 7 of 8, at s = 7 all 8 but thick and alike. Say "fidelity ↑, diversity ↓". That is *why* CFG is used. |
| s56 | Why not add Gaussian noise to token IDs? | two ID numberings | the failed arithmetic. Add the same noise, +1.6, to "dog". Numbering B: 2 → 3.6 → rounds to 4 = "ball". Numbering A: 24 → 25.6 → 26, an unrelated word. The same corruption means different things, and "nearby" IDs are not similar words. |

### 8. Open with the question; close by answering it on the same images

**Opening.** The brief's central question, "How can learning to remove noise let us create an image from noise?", appears in the notes but not on any slide. s01 is a DDPM sample grid with a mis-cropped "Cats" panel (left third is LSUN churches). It is followed by eight two-build application slides (16 states, about 12 min).

Proposed opening:
1. Slide 1 plays the local s39 trajectory: "This digit was pure noise 1,000 steps ago. Only a denoiser touched it. How?"
2. One montage slide of the applications, captioned "By the end you'll be able to explain each of these."

**Closing.** Replace s60's text grid with the opening thumbnails. For each one, show its *one modification* to the sampler learned today:

| Application | Modification |
|---|---|
| RePaint | replace known pixels by forward-noised known pixels at every step |
| SDEdit | start at x_{t₀} = √ᾱ x_strokes + √(1−ᾱ) ε. This is the s18 formula; the t₀ sweep then justifies s05's takeaway, which is currently unsupported. |
| SR3 | concatenate the upsampled low-res image as extra input channels |
| ControlNet | trainable copy fed with edges |
| DreamBooth | fine-tune the denoiser on 3–5 photos |
| GLIGEN | box tokens inside cross-attention |
| LDM | prompt cross-attention |

### 9. Rebuild the language section so masked diffusion looks different from autoregression

- **s58** fills "The dog [?][?][?]" left-to-right, one slot per iteration. That is visually identical to autoregressive decoding, so it undermines the comparison in s59.
  - Show predictions at *all* masked slots each step, with a small vocabulary bar per slot (the Jay Alammar move).
  - Unmask the most confident slots, several at once and out of order.
  - Show one low-confidence re-mask.
- **s59** starts autoregression from "The dog" (2 known tokens) but masked denoising from "? dog ? the ball" (3 known). "The", the prompt in s58, is masked here. Use the same starting state in both columns and count the model calls.
- **s57**: the takeaway says the objective weights by 1/t, but the demo shows "Mean CE = 0.357", which divides by the masked count. The LLaDA loss for this example is (1/0.5)·1.427/5 = **0.571**. Its three builds are visually identical.

### 10. Fix the verified correctness and polish defects

These are objective; the full list is in §3.

- **s38**: `randn()` has no shape and no `torch.` prefix. It should be `torch.randn_like(x)`.
- **s27**: `alpha_bar[t]` with t ∈ [1, T] needs a length-(T+1) table with ᾱ₀ = 1. Show its construction.
- **a03/2**: shows a **false equation** (L_VLB = prior KL − reconstruction, with the local terms missing).
- **Crops**: s01 church crop; s07 paper-layout rules.
- **Label collisions**: s10, s28, s31, s32, s37, s42, s52, a10.
- **Text past border**: s47.
- **Clipped pixel grid**: s46.
- **Caption overlaps images**: s02.
- **Plain-text maths in SVGs**: render with KaTeX.
- **Decimals**: use 2–3.
- **a06**: puts the cosine *schedule* under the title "a faster *sampler*".
- **Training length**: retrain the local DDPM for about 20k steps. The 2k-step run took 98.78 s, so 20k is about 16 min on MPS. The flagship trajectory (s39) and the gallery (s55) are currently malformed.

---

## 3. Verified errors, objectively checkable (separate from design opinion)

| # | Scene/build | Defect | Evidence |
|---|---|---|---|
| E1 | s38/1–2 | Displayed sampler calls `randn()`, with no shape and no `torch.` prefix. As shown, the code raises `NameError`; with `torch.randn()` it would add one scalar to every pixel. Needs `torch.randn_like(x)`. The real script uses `torch.randn(shape, generator=g)`. | builds/s38.jpg; `figures/s38.svg` |
| E2 | a03/2 | "L_VLB = D_KL(q(x_T\|x₀)‖p(x_T)) − E log pθ(x₀\|x₁)" is shown as an equality with the Σ local-KL terms missing. They are added only on a03/3. | builds/a03.jpg |
| E3 | s01/1 | The "Cats" panel's left third is LSUN **church** samples from the adjacent column of the DDPM figure. | crops/s01-cats.jpg |
| E4 | s07/1–2 | Crop residue: grey side bands and black vertical rules from the paper layout. | crops/s07.jpg |
| E5 | s28/1–2 | The red gradient line passes through "we\|ights" and "p\|rediction". The ε̂ hat is displaced to the right. | crops/s28.jpg |
| E6 | s31/1–3 | The curve crosses "Model transition". | crops/s31.jpg |
| E7 | s10, s32 | The feedback line runs into "next st\|ate xₜ₋₁". | crops/s32.jpg, crops/s10.jpg |
| E8 | s37/1–4 | The density curve crosses "sa\|mple". The labels x̂₀, μ and "sample" sit on three near-coincident vertical bars and cannot be read. | crops/s37.jpg |
| E9 | s42/1–2 | The vertical connector strikes "broa\|dcast over H × W". | crops/s42.jpg |
| E10 | s52/2–3 | The dashed line runs through "s=1: εc" and "s=0: εu". | builds/s52.jpg |
| E11 | a10/2–3 | Stacked head outlines cross the "N × L" label. | crops/a10.jpg |
| E12 | s47/1–3 | "Tokeniser + text encoder" overflows its box: T and r reach and pass the border. s08 ("subject-specific weights") and s51 ("Same noisy latent") fill their boxes with no padding. | crops/s47.jpg, s08.jpg, s51.jpg |
| E13 | s46/2–3 | The decoded "Pixels" grid is clipped at the right edge (last column cut). | crops/s46.jpg |
| E14 | s02/1–2 | "Sample N" captions overlap the bottoms of the images; Sample 2's caption sits on the source strip. | crops/s02-imgs.jpg |
| E15 | s05/2 | The takeaway ("More initial noise gives the model more freedom…") is not supported by anything on the slide: no noise-level sweep, and t₀ is undefined at this point. | builds/s05.jpg |
| E16 | s57/1–3 | The takeaway asserts a 1/t-weighted objective. The on-screen loss is the unweighted mean over masked tokens (0.357); the LLaDA-weighted value is 0.571. | builds/s57.jpg |
| E17 | s59/1–3 | The two samplers start from different known-token sets (2 vs 3), and "The", the prompt in s58, is masked. The call-count comparison is therefore unfair. | crops/s59.jpg |
| E18 | s14/1–2, s20, s22, s37, s57 | Builds do not drive the demo state. Static and PDF states show only the initial configuration (s14 only t = 0), so the claimed change is never visible. | §4 |
| E19 | s04 caption | "blue marks the missing region", but the mask is lavender with a green outline. | builds/s04.jpg |
| E20 | Notes, s31 & s36 | "Expected answer" duplicates the narration and does not answer the question. s36 asks "Why add noise during denoising?"; the expected answer begins "Posterior variance is this lecture choice…". | INSTRUCTOR_NOTES.md |

`QA_REPORT.md` reports "zero detected label collisions" in all vector states. E5–E13 show that the check does not test lines or curves through text, or text past a border. A clean QA run should not be read as visual sign-off.

**Recomputed and correct:**
- s17 (0.64)
- s18 (0.36)
- s19 (0.5)
- s20 (β₃₅₀ = 0.007052, ᾱ₃₅₀ = 0.2852, SNR = 0.399)
- s23 (0.0375)
- s33 (0.925)
- s35 (c = 1/3, μθ = 0.70809)
- s36 / s37 (β̃₂ = 0.1111, sample 0.8081)
- s31 (wₜ formula; t > 1 is required because β̃₁ = 0)
- s43 (256 tokens × 16 entries)
- s44 (64×, 48×)
- s49 (A = [0.4011, 0.1978, 0.4011], AV = [0.8022, 0.7967])
- s52 (s = 3 → [1.4, 0.9])
- s57 (−ln 0.7 = 0.3567; × 4 = 1.4267)
- a04 (μ̃, β̃)
- a05 (v ⟂ xₜ in the drawing: 26.6° vs 116.4°)
- a07 (score and ε*)
- a08 (x_τ placed at 40%)
- a09 (LLaDA eq. 3, no 1/L)
- a10 (shapes)

---

## 4. Reveal audit: builds that teach nothing new

**Pixel-identical to the previous build (5):** s20/1→2, s22/1→2, s37/1→2, s37/2→3, s57/1→2.

**Only the cue box changes (15):** s10/1→2, s13/2→3, s14/1→2, s20/2→3, s22/2→3, s30/3→4, s32/1→2, s37/3→4, s44/3→4, s45/2→3, s57/2→3, s59/2→3, a01/2→3, a06/2→3, a09/2→3.

**Builds that replace rather than accumulate a derivation:** s16, s17, s18, s30, s31, s33, s35, s36 and all of a01–a09. Only the latest line is visible, so students lose the chain.

**Reveals that add only a number or a caption:** s44/2–3 (useful), s55/2–3 ("667 sampling seed", "Not computed"), s09/2 (mock grid), s51/3 ("input and target stay fixed"), s42/2 ("encoder · middle · decoder").

Compare Beyond Attention pp. 3–6. Each build adds one question box to the same sentence, the earlier boxes stay, and the final frame holds the whole argument.

---

## 5. Speaking cues and takeaways

**Talk-through lines** average 17 words and are fixed per scene. On a four-build scene (s18, s27, s30, s35, s37, s44, s48, s60) the same sentence sits under four different figures, so it cannot cue the step you are on.

Example rewrite for s18:

| Build | Cue |
|---|---|
| b1 | "Substitute x₁ into x₂: two separate noise draws." |
| b2 | "Their variances add: 0.16 + 0.20 = 0.36." |
| b3 | "So x₂ is a single Gaussian draw around 0.8·x₀." |
| b4 | "Jump straight to any t — this is the line we code." |

**Takeaways** average 14 words. Five are hedges rather than ideas:
- s28 ("…does not run the full generation chain")
- s31 ("…does not include every term")
- s44 ("…not a runtime measurement")
- s53 ("…does not guarantee better quality")
- a07 ("…remains approximate")

s19's takeaway is also phrased as a negative ("…does not identify either contribution"), but it carries a real insight. Keep it.

Several repeat the figure's own words: s08 (70% of its content words are already printed in the figure), s43 (62%), s30 (55%) and s34 (55%).

Memorable alternatives:

| Scene | Rewrite |
|---|---|
| s28 | "Gradients reach θ only through ε̂; the sampled noise is a constant label." |
| s31 | "Each ELBO step is a weighted noise MSE; DDPM keeps the MSE and drops the weight." |
| s36 | "Without fresh noise every seed walks to the same blurry average." |
| s53 | "Guidance buys fidelity with diversity." |
| s58 | "Masked diffusion decides *which* word to fix next, not just *what* word comes next." |

**Good cues worth keeping:**
- s12: "Which features disappear first?"
- s19's cancellation takeaway
- s56: "Would arithmetic on the IDs still mean the same thing?"
- s02: "Would you expect one correct image?"

---

## 6. Published, local, analytical and illustrative: are they distinguished?

**Mostly labelled, but in the wrong way.** Twenty-one figures carry caveat sentences inside the figure, each worded differently. For example:
- "Tensor cells are schematic; the equation defines the actual operation."
- "Hand-designed numerical features, not recovered model attention."
- "Illustrative overlapping densities; the two full image arrays are not asserted to be equal."
- "Proposed task; no local result claimed."

They make the canvas defensive and still leave ambiguity:

- **Mixed provenance inside one figure.** s10 and s32 say "State grid is schematic; the starting Gaussian panel is a recorded draw."
- **A second, incompatible model with no explanation.** s53 is labelled "old checkpoint, seed 1002". It is a different class-conditional model (T = 100, cosine schedule, EMA), unlike the T = 1000 linear unconditional model used everywhere else. Students cannot know this. Either train a class-conditional version of the new model, or say plainly: "a separate class-conditional model (T = 100, cosine schedule)".
- **A mock-up beside real results.** s09/2 puts content-free blue/teal squares labelled "Possible lab task" next to published DiffusionSat images. Either show a real lab dataset (as the CLIP lecture did with brick kilns) or cut the build.
- **Toy vs real schedule.** s17/s19/s33/s35/s37 use β = 0.2; s12/s20/s26 use the real linear schedule. This is never said.

**Recommendation.** Use the four corner badges from Fix 5. Never mix provenance inside one figure. Put the remaining detail in the notes.

---

## 7. Slide-by-slide review

### Applications (s01–s10)

#### s01 · Can we create a new example? — **Below**
Evidence: [builds/s01.jpg](claude-review-evidence/builds/s01.jpg), [crops/s01-cats.jpg](claude-review-evidence/crops/s01-cats.jpg)
- **s01/1 — wrong crop (E3).** Problem: the "Cats" panel includes church towers; an attentive student will ask why. Fix: re-crop to the cat column only, or drop the panel.
- **s01/1 — no hook.** Problem: a published sample grid with the takeaway "A generator learns from examples so that it can produce new samples." This was already established in GM-I, and the lecture's question (noise → image) is absent. Fix: open on the local noise → digit trajectory (s39 data) and ask "How can removing noise create a digit?" (Fix 8).

#### s02 · One description, several valid images — **Close**
Evidence: [builds/s02.jpg](claude-review-evidence/builds/s02.jpg), [crops/s02-imgs.jpg](claude-review-evidence/crops/s02-imgs.jpg)
- **s02/1→2 — good predict-then-reveal.** Keep.
- **s02/1–2 — caption overlap (E14) and low resolution.** Problem: "Sample N" captions sit on the images, and the 240-px LDM crops are upscaled and pixelated. Fix: put captions below with an 8-px gap, show at native size, or use the LDM paper's higher-resolution figure.
- **s02/2 — unremarked artefacts.** Samples 2 and 4 contain stock-photo watermark and caption residue, and several "cherries" look like strawberries. Fix: either crop them out or *use* them. One sentence in the notes ("the model learned watermarks from web data") is a memorable point about training data. Move "the source does not specify matched random seeds" to the notes.

#### s03 · Where should the objects go? — **Close**
Evidence: [builds/s03.jpg](claude-review-evidence/builds/s03.jpg)
- **s03/1 — boxes unlabelled on the canvas.** Problem: red/green identity lives only in the caption, and red vs green is a poor colour-blind pair. Fix: write "planet" and "sun" inside the boxes; use the deck's blue/amber.
- **s03/2 — good.** Three samples obey the layout while varying appearance. Keep.

#### s04 · Fill a missing region — **At**
Evidence: [builds/s04.jpg](claude-review-evidence/builds/s04.jpg)
- Five fillings of one mask is the clearest "conditional distribution" picture in the deck. Only fix: the caption calls the lavender mask "blue" (E19).

#### s05 · Turn rough strokes into a plausible image — **Below**
Evidence: [builds/s05.jpg](claude-review-evidence/builds/s05.jpg)
- **s05/2 — unsupported takeaway (E15).** Problem: it claims a noise-level trade-off that is not shown and cannot yet be understood. Fix: here, ask only "How closely should the result follow the strokes?". In the closing, revisit SDEdit with x_{t₀} = √ᾱ_{t₀}·strokes + √(1−ᾱ_{t₀})·ε and the paper's t₀ sweep (faithful ↔ realistic).
- **s05/1–2 — blurry crops.** Use the SDEdit project page's higher-resolution panels.

#### s06 · What detail is missing at low resolution? — **Close**
Evidence: [builds/s06.jpg](claude-review-evidence/builds/s06.jpg)
- **s06/1 — the low-resolution input is never shown.** The first column is already bicubic-upsampled, although the title is about low resolution. Fix: add a pixelated 16 × 16 input as column 0.
- **s06/1–2 — top row too small.** The comparison lives in the eye zooms. Make those the main row and drop the 115-px faces to thumbnails.
- **Missed link.** "Regression" (blurry average) vs "SR3 sample" (sharp, invented) is precisely the conditional-mean argument of s24. Call it out there (Fix 3).

#### s07 · Draw the structure, generate the appearance — **Close**
Evidence: [builds/s07.jpg](claude-review-evidence/builds/s07.jpg), [crops/s07.jpg](claude-review-evidence/crops/s07.jpg)
- **s07/1–2 — crop residue (E4).** Re-crop.
- **s07/2 — one output, no prompt.** ControlNet takes a prompt too, and one output hides the diversity point from s02/s04. Fix: show the prompt and two or three outputs for the same edge map.

#### s08 · Put a particular subject in a new context — **Close**
Evidence: [builds/s08.jpg](claude-review-evidence/builds/s08.jpg), [crops/s08.jpg](claude-review-evidence/crops/s08.jpg)
- **s08/1 — tight box text (E12), and the paper's serif "Input images" label.** Re-letter in the deck's font.
- **s08/2 — the takeaway repeats the figure** (70% overlap). Make it the training fact instead: "DreamBooth fine-tunes the denoiser on 3–5 photos tied to a rare token."

#### s09 · Could this help answer a lab question? — **Below**
Evidence: [builds/s09.jpg](claude-review-evidence/builds/s09.jpg)
- **s09/2 — content-free mock-up.** Problem: blue/teal squares "Partial observation → Candidate filling" beside real satellite images, with "no local result claimed". Fix: name the lab problem and show one real image (e.g. a cloud-masked Sentinel tile to be filled, or a coarse air-quality field to be downscaled). Otherwise end on the DiffusionSat pair and move the validation point to the notes.

#### s10 · What must a generator learn? — **Below**
Evidence: [builds/s10.jpg](claude-review-evidence/builds/s10.jpg), [crops/s10.jpg](claude-review-evidence/crops/s10.jpg)
- **s10/1 — the full sampler appears before any of its parts is taught.** Problem: "Denoiser · same weights θ · Reverse step · mean + fresh noise · xₜ₋₁ · t ← t−1" in scene 10 is jargon. It also duplicates s32. Fix: replace it with three questions, as on Beyond Attention p.6:
  1. What corrupted input do we train on?
  2. What target do we predict?
  3. How do predictions become a sample?

  Return to them in s21, s32 and s60.
- **s10/1→2 — cue-only reveal; E7 collision; "xT" unsubscripted.**

### Corruption and probability (s11–s20)

#### s11 · From one-step generators to a denoising chain — **Below**
Evidence: [builds/s11.jpg](claude-review-evidence/builds/s11.jpg), [crops/s11.jpg](claude-review-evidence/crops/s11.jpg)
- **s11/2 — misleading input.** Problem: the same 3 × 3 brown "latent" swatch feeds both paths, which hides the key fact that diffusion's starting noise has the image's shape (1 × 28 × 28). Fix: draw a 2-D latent (VAE) vs a 28 × 28 noise field (diffusion), labelled with sizes ("2 numbers" vs "784 numbers").
- **Missed motivation.** GM-I ended on VAEs and GANs. Show *why* we move on, using GM-I's own trained outputs (blurry VAE samples, a GAN collapse). One frame: "one big jump is hard → many small denoising steps."

#### s12 · A digit gradually loses information — **Close**
Evidence: [builds/s12.jpg](claude-review-evidence/builds/s12.jpg)
- **Works:** a real recorded trajectory, a fixed display range and one digit.
- **s12/2 — add the signal level under each panel.** √ᾱ = 0.95, 0.63, 0.28, 0.08, 0.006 (t = 100…1000). Students then *see* that the 7 is readable at 0.63 and gone by 0.08.
- **Missed question.** Ask "Why do t = 500…1000 all look the same?" (√ᾱ < 0.1 from t = 674). This sets up the schedule choice in a06.
- **s12/1→2 — four panels at once.** Reveal t = 300 alone first, ask "Is it still a 7?", then the rest. Replace "recursive trajectory" with "one noise draw per step".

#### s13 · Can we recover the original with certainty? — **Below**
Evidence: [builds/s13.jpg](claude-review-evidence/builds/s13.jpg)
- **s13/1 — abstract axis.** Problem: "One noisy coordinate", but which pixel, and what are its clean values? Fix: pick the running pixel; the 7 has +1 there and the 2 has −1. The two curves are N(√ᾱ·(+1), 1−ᾱ) and N(√ᾱ·(−1), 1−ᾱ).
- **s13/2→3 — cue-only.** Make b3 add posterior bars P(7 \| xₜ) and P(2 \| xₜ) at the observed value, and show the curves widening at a larger t.
- **Naming.** Use "digit 7 / digit 2", not "digit A / B". Move the defensive caption to the notes.

#### s14 · What happens to a whole distribution? — **Below**
Evidence: [builds/s14.jpg](claude-review-evidence/builds/s14.jpg)
- **s14/1–2 — only t = 0 is ever shown (E18).** Problem: the takeaway ("Noise changes a distribution") describes a change no static state shows. Fix: builds at t = 0 / 300 / 1000, as small multiples, with the clusters merging into one blob.
- **s14 — terms before definitions.** The readout uses ᾱ, "score", "εoracle = −√(1−ᾱ) score" and "Fixed ε: direct marginal coupling" before s17 defines ᾱ, and score is never defined in the core. Fix: keep the readout to "t, noise level"; move the oracle to a07.
- **Scale.** The plot uses about 25% of the canvas; make it the figure.

#### s15 · What is a clean image tensor? — **Below**
Evidence: [builds/s15.jpg](claude-review-evidence/builds/s15.jpg)
- **s15/1 — decorative grid.** Problem: a random blue 8 × 8 pattern labelled "28 × 28 positions" does not resemble the 7. Fix: zoom into the real digit's pixel grid and print three actual values (0 → −1, 128 → 0.004, 255 → +1). Circle the running pixel.
- **s15/2 — the takeaway mentions noise leaving [−1, 1]**, which is not shown. Show the running pixel after noise (e.g. 1.3) on the number line.

#### s16 · One small corruption step — **Below**
Evidence: [builds/s16.jpg](claude-review-evidence/builds/s16.jpg)
- **s16/1–3 — schematic swatches and a vanishing equation.** Use the real digit and the real `fresh-gaussian.png`. Keep the three equation lines stacked.
- **s16 — the question is asked but not answered.** "Why are there square roots?" The answer is variance preservation: Var = (1−β)·1 + β·1 = 1. Show it as a stacked bar (signal 0.8 + noise 0.2 = 1) and with the running pixel: x₁ = 0.894·1 + 0.447·(−0.5) = 0.671.

#### s17 · Alpha and alpha-bar mean different things — **Close**
Evidence: [builds/s17.jpg](claude-review-evidence/builds/s17.jpg)
- **Works:** 1 → 0.8 → 0.64 bars ("0.8 × 0.8 — multiply the steps").
- **s17/1–3 — show the complement.** Each bar should be stacked signal ᾱ + noise (1 − ᾱ) = 1. That is the variance-preserving picture, and it makes s18 obvious. State β = 0.2 explicitly. Use the empty right half for a real-schedule ᾱₜ curve with the s12 panel times marked.

#### s18 · Jump directly to a training noise level — **Below**
Evidence: [builds/s18.jpg](claude-review-evidence/builds/s18.jpg)
- **s18/1→4 — the derivation erases itself.** Keep lines 1–4 stacked, with the newest in colour.
- **s18 — no numbers.** Add 0.16 + 0.20 = 0.36 = 1 − 0.64. Add a tiny histogram of 0.4·η₁ + 0.447·η₂ over 10k draws against N(0, 0.36), which takes seconds to compute. Show the recursive path (η₁ = −0.5, η₂ = −0.224) arriving at the same x₂ = 0.5 as the jump.
- **Cue.** Per-build cues (see §5).

#### s19 · Work through one noisy coordinate — **Close**
Evidence: [builds/s19.jpg](claude-review-evidence/builds/s19.jpg)
- **Works:** the best mathematical slide. Real arithmetic (0.8 − 0.3 = 0.5), consistent with s17, and a memorable takeaway (contributions can cancel).
- **s19/1 — weak first reveal** (two numbers only), and the figure uses the left 60%. Fix: use the right side to show the other explanation of the same 0.5 (x₀ = −1, ε = 2.17). This sets up s24 with numbers.
- Replace "Illustrative coordinate" with the running-example badge.

#### s20 · Try changing the noise level — **Below**
Evidence: [builds/s20.jpg](claude-review-evidence/builds/s20.jpg)
- **s20/1→2 identical; s20/2→3 cue only (E18).** Drive the slider from builds: t = 100, 350, 920.
- **s20 — jargon and decimals.** "Shared ε marginal coupling", "Aggregate ε", "β = 0.007052". Use "same noise image ε", "noise", 2–3 decimals.
- **s20 — inspected pixel 400 is background (x₀ = −1.000).** Switch to the running stroke pixel.

### Learning (s21–s31)

#### s21 · Which target should the network predict? — **Below**
Evidence: [builds/s21.jpg](claude-review-evidence/builds/s21.jpg)
- **s21/1–2 — the question is not answered** (Fix 7). Show three candidate targets for the same xₜ: ε (noise image), x₀ (clean 7) and v. Explain that ε has unit variance at every t, which is why DDPM chose it. Link to a05.
- **Plain-text "εθ(xt, t)".** Typeset.

#### s22 · Assemble one training example — **Close**
Evidence: [builds/s22.jpg](claude-review-evidence/builds/s22.jpg)
- **Works:** measured input, target, prediction and error. This is the strongest evidence that the network learns anything.
- **s22/1→2 identical; s22/2→3 cue-only (E18).** Build it: x₀ → xₜ → target ε → predicted ε̂ → error map.
- **Tiles about 105 px.** Make the target/prediction pair large, side by side. Show the readout to 3 decimals (MSE 0.021).

#### s23 · Calculate the loss for four values — **Close**
Evidence: [builds/s23.jpg](claude-review-evidence/builds/s23.jpg)
- **Correct and clear** (0.15 / 4 = 0.0375).
- **Missed link.** Its first entry (−0.5 vs −0.4) *is* the running pixel. Label it, and these four numbers become "four pixels of our 7".

#### s24 · Can the network know the exact sampled noise? — **Far below**
Evidence: [builds/s24.jpg](claude-review-evidence/builds/s24.jpg)
- **s24/1–2 — the central concept is drawn as two unlabelled bell curves**, with an on-canvas apology ("Schematic uncertainty, not a measured noise distribution"). The x-axis is not ε, and nothing is computed. Fix 3: the two explanations of x₂ = 0.5, posterior 0.90 / 0.10, E[ε \| x₂] = −0.24. Then a third build with the neighbouring pixels: "context makes the posterior sharper — that's what the U-Net uses." Callback to s06.

#### s25 · One denoiser, many noise levels — **Below**
Evidence: [builds/s25.jpg](claude-review-evidence/builds/s25.jpg)
- **s25/1–3 — t = 7 and t = 920 drawn as near-identical brown icons.** Use `batch-7.png` and `batch-920.png` (nearly clean vs pure noise), with ᾱ = 0.999 and 0.0002. The point ("one network, very different inputs, so t must be an input") then makes itself.

#### s26 · A batch can contain different timesteps — **Close**
Evidence: [builds/s26.jpg](claude-review-evidence/builds/s26.jpg)
- **Works:** real images at t = 7 / 350 / 920 and the gather → reshape → broadcast shapes.
- **Add the gathered values:** ᾱ[7] = 0.999, ᾱ[350] = 0.285, ᾱ[920] = 0.0002. Draw the [B] → [B,1,1,1] reshape as a tensor picture, as in Beyond Attention p37's "E: 6 × 8" boxes.

#### s27 · Translate the picture into PyTorch — **Close**
Evidence: [builds/s27.jpg](claude-review-evidence/builds/s27.jpg)
- **Works:** code beside its four stage boxes.
- **s27/1–4 — no current-line highlight.** The boxes don't light up with the lines. Build 4 adds five lines at once (model call, loss, three optimiser lines).
- **s27 — hidden indexing assumption.** `alpha_bar[t]` with t = 1…T needs a T + 1 table with ᾱ₀ = 1. Show `alpha_bar = torch.cat([torch.ones(1), torch.cumprod(1 - beta, 0)])`, or students' own code will be off by one.

#### s28 · Where do the gradients go? — **Below**
Evidence: [builds/s28.jpg](claude-review-evidence/builds/s28.jpg), [crops/s28.jpg](claude-review-evidence/crops/s28.jpg)
- **s28/1 — the gradient arrow appears before the loss exists.** Order the builds: loss → gradient → "no gradient into the ε target (a constant label)".
- **E5 collisions; displaced ε̂ hat.**
- **Takeaway is a hedge.** See §5.

#### s29 · Where do the training labels come from? — **Below**
Evidence: [builds/s29.jpg](claude-review-evidence/builds/s29.jpg)
- **s29/2 — text-only reveal.** The point (noise draws don't create new digits) deserves its picture. Example: a model trained only on 7s (or on 0/1 only), with its samples shown, all 7s. That is cheap to train locally.

#### s30 · The ELBO from Lecture I appears again — **Below**
Evidence: [builds/s30.jpg](claude-review-evidence/builds/s30.jpg)
- **s30/1–4 — the promised bridge to Lecture I is absent** (Fix 2).
- **Term boxes' subtext is about 11 px** ("compare q(x_{t−1}\|x_t, x₀) with pθ(x_{t−1}\|x_t)"), and "xT" is unsubscripted.
- **s30/3→4 is cue-only.** The same figure is duplicated in a03.

#### s31 · How does the bound lead to noise MSE? — **Far below**
Evidence: [builds/s31.jpg](claude-review-evidence/builds/s31.jpg), [crops/s31.jpg](claude-review-evidence/crops/s31.jpg)
- **s31/1–3 — the substitution that *is* the derivation is missing.** Each build replaces the previous formula, and wₜ is never plotted (Fix 2). The 'Model transition' collision is E6.
- **Instructor note's "Expected answer" doesn't answer** (E20).

### Generation (s32–s39)

#### s32 · Sampling starts without a clean image — **Below**
Evidence: [builds/s32.jpg](claude-review-evidence/builds/s32.jpg), [crops/s32.jpg](claude-review-evidence/crops/s32.jpg)
- **Duplicate of s10. s32/1→2 is cue-only; E7 collision.** Fix: show the recorded x_T (pure noise) and the question "We have no x₀ — what can the network give us?". Then go straight to s33.

#### s33 · A predicted clean image is useful — **Close**
Evidence: [builds/s33.jpg](claude-review-evidence/builds/s33.jpg)
- **Works:** 0.5 → +0.24 → 0.74 → ÷0.8 → 0.925.
- **Missed punchline.** The true x₀ was 1 (s19), so the error is 0.075 = (0.6/0.8) × 0.1. The noise-prediction error is amplified by √(1−ᾱ)/√ᾱ, which explodes at high t. That is why the x̂₀ row in s39 is blurry early. Write "true value 1.0" in grey beside 0.925.
- **Equations replaced per build; leading-dot numbers** (".5 − .6(−.4)"). Write 0.5.

#### s34 · Which state goes into the next call? — **Close**
Evidence: [builds/s34.jpg](claude-review-evidence/builds/s34.jpg)
- Clean. Takeaway overlaps the figure by 55%; replace it with the misconception it prevents ("x̂₀ is a guess we look at; xₜ₋₁ is the state we keep").

#### s35 · Build one reverse transition — **Close**
Evidence: [builds/s35.jpg](claude-review-evidence/builds/s35.jpg)
- **Numbers correct** (c = 1/3, μθ = 0.708).
- **s35/1–4 — the layout reads as scattered pieces**: a floating red "−", "÷ √αₜ" above μθ, converging arrows. Fix: one left-to-right arithmetic strip, as in s19: 0.5 − (1/3)(−0.4) = 0.633 → ÷ 0.894 → 0.708. Keep the general equation stacked beneath. 3 decimals.
- **Add the true posterior mean 0.745** in grey (from a04). The 0.037 gap is s31's KL.

#### s36 · Why add noise during denoising? — **Far below**
Evidence: [builds/s36.jpg](claude-review-evidence/builds/s36.jpg)
- **The title's question is never answered** (Fix 7). The figure is a Gaussian with an arrow. The notes' expected answer is unrelated (E20).
- **β̃₂ = 0.111 is not computed here**, though the slide defines it. Compute it with the running numbers.

#### s37 · Work through one reverse step (demo) — **Below**
Evidence: [builds/s37.jpg](claude-review-evidence/builds/s37.jpg), [crops/s37.jpg](claude-review-evidence/crops/s37.jpg)
- **s37/1→2→3 identical; 3→4 cue-only (E18).** Drive it: ε̂ → μ → σ·z → sample.
- **E8 — the plot's labels are unreadable.** Separate x̂₀, μ and the sample vertically with leader lines, or show them on a number line from −1 to 1.5 at a larger scale.
- **6-decimal readout and a defensive caption** ("parameters remain internally consistent"). Move the caption to the notes.

#### s38 · The complete sampling loop (code) — **Below**
Evidence: [builds/s38.jpg](claude-review-evidence/builds/s38.jpg)
- **E1: `randn()`.**
- **Highlight lines as the side boxes are revealed.** As in s27, there is no current-line highlight.

#### s39 · What are we watching in a generation video? — **Close**
Evidence: [builds/s39.jpg](claude-review-evidence/builds/s39.jpg), [crops/s39.jpg](claude-review-evidence/crops/s39.jpg)
- **Works:** real states and x̂₀ estimates. This is the best figure in the deck.
- **The teaching moment is unspoken.** The x̂₀ row is a blurry "7/9" at t = 600, becomes a "6/a" at t = 400, and commits by t = 200. That is the conditional mean sharpening (s24) and the error amplification (s33), visible. Annotate it.
- **The flagship sample ends as a malformed "a".** Retrain longer (Fix 10), or choose a seed that ends as a digit and say it was chosen (the gallery stays unselected).

### Architecture (s40–s46)

#### s40 · What should the denoiser output? — **Below**
Evidence: [builds/s40.jpg](claude-review-evidence/builds/s40.jpg)
- **s40/1–2 — input and output both look like noise**, so students can't see what the network did. Add the third panel xₜ − √(1−ᾱ)·ε̂ → x̂₀ (a recognisable 7), or a difference map. Move "Recorded input and prediction from the same t=350 model call" into a badge.

#### s41 · Follow the shapes through a small U-Net — **Below**
Evidence: [builds/s41.jpg](claude-review-evidence/builds/s41.jpg)
- **s41/1–3 — boxes have doubled, offset outlines** that read as a rendering glitch. **All boxes are the same size**, so there is no visual sense of resolution halving and channels doubling. **No input (1 × 28 × 28) and no time embedding.**
- Fix: the standard U-shape, with block width ∝ channels and height ∝ resolution. Use real feature-map thumbnails from the trained model at each level (cheap: register hooks). Add t entering every block (s42). Show "64 + 64 = 128 → 64" at the skips.

#### s42 · Inject the timestep throughout the denoiser — **Close**
Evidence: [builds/s42.jpg](claude-review-evidence/builds/s42.jpg), [crops/s42.jpg](claude-review-evidence/crops/s42.jpg)
- **E9 collision.**
- **s42/2 adds only the words "encoder · middle · decoder".** Instead, show the sinusoidal embedding for t = 7, 350 and 920 as three coloured stripes: different t → different vectors.

#### s43 · Can a transformer predict the noise? — **Below**
Evidence: [builds/s43.jpg](claude-review-evidence/builds/s43.jpg)
- **Numbers correct** (256 × 16).
- **The answer is incomplete.** Time and condition injection (adaLN-Zero) is missing. Add it.
- Takeaway overlaps the figure by 62%. The CLS callback is nice; keep it as one short line.

#### s44 · Why denoise a compressed representation? — **Close**
Evidence: [builds/s44.jpg](claude-review-evidence/builds/s44.jpg)
- **Correct and useful** (64× fewer positions vs 48× fewer entries).
- **s44/3→4 cue-only; the takeaway is a hedge.** Use the `ldm-*.png` decoded examples so "compressed" visibly still means "the same picture".

#### s45 · Train the autoencoder, then the denoiser — **Close**
Evidence: [builds/s45.jpg](claude-review-evidence/builds/s45.jpg)
- Clear two-stage diagram. **s45/2→3 is cue-only.** Put a real image and reconstruction in stage 1; link to GM-I's autoencoder.

#### s46 · Generate in latent space, decode once — **Close**
Evidence: [builds/s46.jpg](claude-review-evidence/builds/s46.jpg), [crops/s46.jpg](claude-review-evidence/crops/s46.jpg)
- **E13 clipping.** The decoder input arrow starts from the feedback loop, which suggests decoding every iteration. Start it from a separate "z₀" node after the loop.

### Conditioning (s47–s54)

#### s47 · What information comes from the prompt? — **Below**
Evidence: [builds/s47.jpg](claude-review-evidence/builds/s47.jpg), [crops/s47.jpg](claude-review-evidence/crops/s47.jpg)
- **E12 overflow.**
- **No tokens are shown.** Split "a | basket | full | of | cherries" into L = 5 (+ padding to 77) rows of C. Say which encoder: students know **CLIP**. Stable Diffusion v1 uses CLIP's text encoder, and that callback is free.
- Merge with s50.

#### s48 · How do image features attend to text? — **Close**
Evidence: [builds/s48.jpg](claude-review-evidence/builds/s48.jpg)
- **A good four-step build.**
- **Typeset and define.** Q = H W_Q, K = C W_K, V = C W_V, with N = H·W image positions and L text tokens. Label the attention matrix's axes with the actual prompt words, as on Beyond Attention p59.

#### s49 · Work through one attention calculation (demo) — **Close**
Evidence: [builds/s49.jpg](claude-review-evidence/builds/s49.jpg)
- **Correct, and builds drive the stage.** Keep.
- **An unrelated "Guidance s" slider is visible in attention view.** Hide it.
- **The query isn't located in the image.** Show which image patch is "query 1", e.g. a patch on the dog.
- **6-decimal AV.** Use 0.80, 0.80.

#### s50 · Use the prompt at every denoising step — **Below**
Evidence: [builds/s50.jpg](claude-review-evidence/builds/s50.jpg)
- Too thin for its own scene: three boxes and one sentence. Merge into s47 as a final build.

#### s51 · Teach one model to work with or without a condition — **Close**
Evidence: [builds/s51.jpg](claude-review-evidence/builds/s51.jpg), [crops/s51.jpg](claude-review-evidence/crops/s51.jpg)
- **E12 tight text.** Add the actual mechanism, `if rand() < 0.1: c = ∅`, and the old model's dropout of 0.15.

#### s52 · Combine predictions with classifier-free guidance — **Close**
Evidence: [builds/s52.jpg](claude-review-evidence/builds/s52.jpg)
- **Vectors correct.**
- **"Axes use different scales"** distorts the very geometry being taught (x is about 3× y). Use equal scales. Fix the E10 collisions. Typeset ε_cfg. Show the s = 7 point, or say why it is off-chart.

#### s53 · What changes when we turn up guidance? — **Below**
Evidence: [builds/s53.jpg](claude-review-evidence/builds/s53.jpg), [crops/s53.jpg](claude-review-evidence/crops/s53.jpg)
- **The takeaway undersells the picture, and the key lesson is missed** (Fix 7).
- **"old checkpoint, seed 1002" is unexplained provenance** (§6).

#### s54 · Revisit completion and super-resolution — **Below**
Evidence: [builds/s54.jpg](claude-review-evidence/builds/s54.jpg)
- **It revisits but does not explain.** The *how* is the payoff: RePaint's replace-known-pixels step, and SR3's concatenated low-res input. Only 2 of the 8 openers are revisited, and the SR3 "low-resolution observation" is displayed at full size and does not look low-resolution. Fold into the redesigned closing (Fix 8).

### Evidence and language (s55–s60)

#### s55 · What do these results establish? — **Below**
Evidence: [builds/s55.jpg](claude-review-evidence/builds/s55.jpg)
- **The reveals add only "667" and "Not computed".** Honest, but deflating, and it belongs in the notes. Better:
  1. The unselected batch (after longer training).
  2. Nearest training neighbour for each sample ("memorised or new?").
  3. One sentence on what FID would measure.

#### s56 · Why not add Gaussian noise to token IDs? — **Close**
Evidence: [builds/s56.jpg](claude-review-evidence/builds/s56.jpg)
- **A good renumbering idea; the failure is never demonstrated** (Fix 7).
- The masking row's colours (purple / teal / brown) have no legend.
- Add one line acknowledging continuous diffusion on *embeddings* (Diffusion-LM), so the claim stays about IDs.

#### s57 · Build a masked-text training example (demo) — **Below**
Evidence: [builds/s57.jpg](claude-review-evidence/builds/s57.jpg)
- **Identical builds; loss/takeaway mismatch (E16).**
- **"Mask probability 0.5" but 4 of 5 tokens masked** at seed 667. That is correct for random masking but confusing on a static slide; use a seed with about half masked.
- "Illustrative p(true token) = .7" for every token: show a small vocabulary bar per masked slot instead.

#### s58 · Fill the response a few tokens at a time — **Far below**
Evidence: [builds/s58.jpg](claude-review-evidence/builds/s58.jpg)
- **It looks exactly like autoregression** (Fix 9). The title says "a few tokens at a time", yet the figure fills one, left to right.

#### s59 · How do the two text samplers differ? — **Below**
Evidence: [builds/s59.jpg](claude-review-evidence/builds/s59.jpg), [crops/s59.jpg](claude-review-evidence/crops/s59.jpg)
- **E17 — unfair start, and "The" masked.**
- In row 2, a generated "The" is coloured as prompt.
- s59/2→3 is cue-only.
- The takeaway is a hedge ("latency depends on the full implementation"). Say instead: "AR: 3 calls for 3 tokens; masked: T calls regardless of length — fewer if several tokens are fixed per call."

#### s60 · Can we now explain the opening examples? — **Far below**
Evidence: [builds/s60.jpg](claude-review-evidence/builds/s60.jpg)
- **No opening example appears.** It is a 3 × 4 grid of text boxes (State × Condition × Target × Sampling rule), and its rows (Pixels / Latents / Tokens) don't map to the openers. Beyond Attention closes on a question that leads into the next lecture. Fix 8.

### Appendix (a01–a11)

#### a01 · Independent Gaussians: means and variances — **Close**
Evidence: [builds/a01.jpg](claude-review-evidence/builds/a01.jpg)
- Correct. Draw bell curves rather than text boxes, and instantiate with s18's numbers (a = 0.4, b = 0.447 → 0.16 + 0.20). a01/2→3 is cue-only, and the first equation disappears at b2.

#### a02 · A lower bound on log evidence — **Close**
Evidence: [builds/a02.jpg](claude-review-evidence/builds/a02.jpg)
- Correct. **Jensen is invoked without its picture.** The concave log curve with a chord was requested in GM-I as well. Give the bars numbers (the GM-I deck used −0.693 = −0.798 + 0.105).

#### a03 · Factor the fixed chain backwards — **Below**
Evidence: [builds/a03.jpg](claude-review-evidence/builds/a03.jpg)
- **E2: false equation at a03/2.**
- Duplicates s30's figure, with unreadable 11-px subtext. Use "= L_T + L₀ + (local terms, next)" at b2.

#### a04 · The tractable training posterior — **Close**
Evidence: [builds/a04.jpg](claude-review-evidence/builds/a04.jpg)
- Correct. **Instantiate:** μ̃₂ = 0.745 for the running pixel vs μθ = 0.708 (s35). This one number connects a04 → s31 → s35.

#### a05 · Alternative prediction targets — **Close**
Evidence: [builds/a05.jpg](claude-review-evidence/builds/a05.jpg)
- Correct, and the geometry is right (v ⟂ xₜ). Label the angle φ with cos φ = √ᾱ, and add a second xₜ at larger t to show the rotation.

#### a06 · A faster sampler needs a different transition — **Below**
Evidence: [builds/a06.jpg](claude-review-evidence/builds/a06.jpg)
- **a06/1 shows the cosine noise schedule** under a title about samplers, conflating two ideas. Split: a schedule slide (the linear vs cosine ᾱₜ curve, with s12's "t = 500…1000 all noise" as motivation) and a DDIM slide (define x̂₀ here; show 50 vs 1000 calls on the local checkpoint if feasible). a06/2→3 is cue-only.

#### a07 · A score oracle for eight Gaussian clusters — **Close**
Evidence: [builds/a07.jpg](claude-review-evidence/builds/a07.jpg)
- Correct. The "sun" glyphs read as icons rather than Gaussian clusters, and τ is undefined on the canvas. `assets/oracle-data.json` already holds trajectories: plot a score field (arrows) and three reverse paths. This is the Lil'Log-style figure this slide wants.

#### a08 · Flow matching: learn a velocity field — **Close**
Evidence: [builds/a08.jpg](claude-review-evidence/builds/a08.jpg)
- Correct (x_τ at 40%). The key sentence ("Learned field averages compatible pair velocities") is text only. Draw two crossing noise–data pairs and the averaged, curved field at the crossing.

#### a09 · Masked language: objective and sampling convention — **Below**
Evidence: [builds/a09.jpg](claude-review-evidence/builds/a09.jpg)
- Objective correct. **The "sampling convention" in the title is never shown** (a09/2→3 is cue-only). Add the low-confidence re-masking schedule here, or retitle.

#### a10 · Cross-attention with heads and spatial layout — **Close**
Evidence: [builds/a10.jpg](claude-review-evidence/builds/a10.jpg), [crops/a10.jpg](claude-review-evidence/crops/a10.jpg)
- Shapes correct. **E11 collision.** Add W_O and the reshape back to [B, d, H, W].

#### a11 · What to report with a generated example — **Below**
Evidence: [builds/a11.jpg](claude-review-evidence/builds/a11.jpg)
- Three text bars. The content is right, but it belongs in the *main deck as the badge system* (Fix 5), not as a closing appendix table.

---

## 8. What Beyond Attention does that this deck does not (with page references)

| Beyond Attention | GM-II today |
|---|---|
| **pp. 3–6.** One sentence ("Raghav goes to school in Delhi.") and three question boxes revealed cumulatively under the governing question "What should the Transformer output?" | The governing question is never on a slide. Builds replace rather than accumulate. |
| **p. 37.** Every row of the NER scores is labelled with the actual word. Shapes sit inside boxes (E: 6 × 8 → W: 8 × 4 → Scores: 6 × 4). One caption line. | Tensor figures use decorative swatches. Shapes are listed but not drawn as tensors. Captions and cue boxes stack. |
| **p. 59.** Weights on the six real words (0.72, 0.04, …), with a one-line caveat: "Illustrative attention weights; their sum is 1." | Caveats are full sentences inside 21 figures. Weights are unlocated. |
| No Talk-through/Takeaway box, no footer metadata. The figure fills the width. | About 20% of each slide is cue box and footer; many figures occupy the upper-left 60%. |
| Section dividers with a mini-map ("Part III · Sequence-to-sequence prediction"). | Section changes only in the eyebrow text. |
| Closes on "Next lecture: what if our tokens are not words?" | Closes on a taxonomy table. |

See [crops/reference-p37-p59.jpg](claude-review-evidence/crops/reference-p37-p59.jpg).

---

## 9. Why this keeps happening (and what to change in the brief)

The GM-I review traced the GM-I deck's problems to a compliance-heavy brief, and the same pattern is visible here:
- `SPECIFICATION.md` is 98 kB.
- `requirements-to-scenes.json` maps 60 scenes / 158 builds.
- The README spends more words on provenance disclaimers than on the running example.

The build hit every count (60 / 158 / 11 / 33) and every "do not claim" clause, and satisfied the automated checks (§3, E5–E13). But nothing in the brief demanded:
- a named running example;
- numbers after each equation;
- that each title's question be answered by its figure;
- that reveals accumulate.

Before another generation pass, rewrite the brief as **figure contracts**. One per scene, for example: "s31 shows μ̃ − μθ = c(ε̂ − ε)/√α stacked under the KL, evaluates to 0.00625 on pixel p, and adds a wₜ-vs-t plot from the real schedule." Then judge each scene against its own contract.

---

## 10. Status after the rebuild (19 October 2026)

The deck was rebuilt from scratch on the Generative Modelling I runtime, not patched. The Codex build is kept as `codex-v1.html`. The new deck has 96 slides and 271 clicks; the export fails if any maths appears outside KaTeX, and every click has a speaker cue.

| Top-ten fix | Where it is now |
|---|---|
| 1 · one pixel as the spine | `#digit` introduces pixel (12, 19) of test 7 (value 255 → +1). The same numbers then run through `#one-step` (0.671), `#jump` (0.16 + 0.20 = 0.36), `#pixel-x2` (0.5, and the paper explanation with ε = 2.17), `#cannot-know` (0.90 / 0.10, E[ε] = −0.24), `#x0hat` (0.925, error 0.075), `#true-step` (0.745), `#plug-guess` / `#mean-step` (0.708), `#noise-step` (0.111, 0.808) and `#kl-step` / `#kl-to-mse` (0.00625 = 0.625 × 0.01) |
| 2 · the KL → MSE substitution and the weight plot | Part 4 (`#two-chains`, `#elbo-bridge`, `#elbo-terms`, `#kl-step`, `#kl-to-mse`, `#weights`). It now comes after sampling, so both reverse means are already known |
| 3 · conditional mean | `#cannot-know` (two explanations, posterior bars), `#guesses` (our model's x̂₀ at five noise levels), `#superres-pub` (regression blurs), `#undo` (P(ink) 0.90 at step 2 vs 0.58 at step 500) |
| 4 · reveals that change the figure | Derivations accumulate (`#jump`, `#kl-to-mse`, appendix). Widgets follow the clicks (`#explorer`, `#noise-step`, `#scrub`) |
| 5 · no chrome; cues in notes; badges | No eyebrow, footer or cue box. One caption line. Cue per click in **S** and in `INSTRUCTOR_NOTES.md`. Four corner badges |
| 6 · real tensors | Real digit, noise and model outputs in every diagram (`one-step`, `tuple`, `gather`, `unet` feature maps, `two-stage` latents, `latent-gen`) |
| 7 · answer each title | `#targets` (ε / x₀ / v), `#why-noise` (our model: without fresh noise all 8 seeds become the same “1”), `#dit` (adaLN conditioning), `#cfg-grid` (95% → 100% sevens, spread 16.4 → 15.7), `#ids` (+1.6 noise turns “dog” into “ball”) |
| 8 · open with the question; close on the openers | `#hook` (our trajectory and the question), `#apps` (8 inputs predicted then revealed), `#revisit` (the same applications, each with its change to the recipe), plus inpainting and SDEdit run on our own model |
| 9 · masked diffusion that does not look autoregressive | `#mask-train` (vocabulary bars, LLaDA 1/t loss = 2.17), `#mask-gen` (most confident slot first, then two at once), `#ar-vs-mask` (same start, same three slots) |
| 10 · verified defects | Every crop redone from the originals (`scripts/crop_published.py`). Code uses `torch.randn_like(x)` and a zero-sentinel β table. No false equation. All labels checked in renders |

The models were retrained: pixel DDPM for 30k steps, plus a class-conditional DDPM, a latent autoencoder with a latent DDPM, and a classifier to measure guidance. Feedback from the Generative Modelling I review is in `FEEDBACK_LOG.md`.
