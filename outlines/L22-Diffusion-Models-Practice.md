---
title: "Lecture 22: Diffusion Models — Practice"
status: "Canonical 24-lecture revision — draft outline"
prerequisites: "L21 forward/reverse diffusion; attention and Transformer basics"
next: "L23 — Efficient Inference"
---

# Lecture 22: Diffusion Models — Practice
## From a denoising objective to controllable text-to-image systems

## Core story

\[
\boxed{\text{Modern diffusion systems combine a denoiser, a compact latent space, and conditioning.}}
\]

The practical question is not “how do we memorize Stable Diffusion?” It is:

> Where does each input—noise level, text, image, mask, or control signal—enter the denoising computation?

**Target:** 80 minutes, approximately 65 slides.  
**One worked mechanism:** classifier-free guidance.  
**One diagram:** latent diffusion end-to-end.  
**Out of scope:** live model fine-tuning, exhaustive sampler families, detailed licensing or product comparisons.

---

## Part I — Turn the theory into a working system

1. **Title**  
   *Diffusion in Practice: Latents, Text Conditioning, and Guidance*

2. **Recall the minimal theory**
   \[
   x_t,\;t \xrightarrow{\epsilon_\theta} \hat\epsilon
   \xrightarrow{\text{reverse update}}x_{t-1}.
   \]
   The remaining design decisions are representation, architecture, condition, and sampling.

3. **System map**
   ~~~text
   prompt → text encoder ─────────────┐
                                      ↓
   noise → denoiser εθ(latent, time, condition) → repeated reverse steps → decoder → image
   ~~~

4. **Why not diffuse raw pixels?**  
   A \(512\times512\times3\) image has 786,432 values. Repeatedly denoising such a large tensor is expensive.

5. **Latent diffusion idea**
   \[
   x\xrightarrow{E}z
   \quad\text{then diffuse in }z\text{-space}\quad
   z_0\xrightarrow{D}\hat x.
   \]
   Reuse the VAE’s compact continuous representation from L19.

6. **What the autoencoder must preserve**  
   The latent should retain perceptual content while reducing spatial cost. It is not merely a JPEG compressor; its decoder defines what the diffusion model can ultimately render.

7. **End-to-end latent-diffusion diagram**
   ~~~text
   training image x → VAE encoder → z0 → add noise → zt
   text prompt c → text encoder ───────────────────────→ denoiser predicts ε
   predicted clean latent ẑ0 → VAE decoder → reconstructed / generated image
   ~~~

8. **Training loss stays familiar**
   \[
   L=\mathbb E\left[\lVert \epsilon-\epsilon_\theta(z_t,t,c)\rVert_2^2\right].
   \]
   L21’s theory transfers unchanged; only the space and conditional inputs change.

---

## Part II — The denoiser architecture

9. **What structure does a denoiser need?**  
   - local detail at many resolutions;
   - a global view of the composition;
   - a route for conditioning information;
   - knowledge of the current noise level.

10. **U-Net at a glance**
    ~~~text
    high-resolution latent → downsample → bottleneck → upsample → predicted noise
                              ↘ skip connections ↗
    ~~~

11. **Why the U shape helps**  
    Downsampling grows receptive field and captures global structure; skip connections preserve fine spatial detail for reconstruction.

12. **Time embedding**  
    A numeric timestep becomes a learned vector injected into residual blocks. The same noisy latent should be denoised differently at \(t=20\) and \(t=900\).

13. **Text is not pasted onto pixels**  
    A text encoder transforms a prompt into a sequence of embeddings:
    \[
    c=(c_1,\ldots,c_L).
    \]
    The denoiser accesses these embeddings through cross-attention.

14. **Cross-attention reminder**
    \[
    \operatorname{Attention}(Q,K,V)=
    \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt d}\right)V.
    \]
    Queries come from image/latent features; keys and values come from text tokens.

15. **Interpret a cross-attention block**  
    At every location, the denoiser asks: “which prompt tokens matter for cleaning up this region now?”

16. **Practical condition types**

    | Condition | Example use |
    |---|---|
    | Text | describe desired semantics/style |
    | Class label | class-conditional generation |
    | Image / low-resolution image | image-to-image, super-resolution |
    | Mask | inpainting |
    | Pose / edges / depth | layout and structure control |

17. **Architecture landscape—one slide**  
    U-Nets remain the intuitive teaching architecture; diffusion Transformers replace much of the convolutional backbone with token processing. The noising objective is the stable idea beneath both.

---

## Part III — Make a condition matter: classifier-free guidance

18. **The conditioning problem**  
    A conditional model can generate plausible images that only weakly follow a prompt. We want stronger prompt adherence without a separate external classifier.

19. **Train with occasional dropped conditions**  
    During training, sometimes replace \(c\) with an empty condition. One network learns:
    \[
    \epsilon_\theta(z_t,t,c)
    \quad\text{and}\quad
    \epsilon_\theta(z_t,t,\varnothing).
    \]

20. **Classifier-free guidance**
    \[
    \hat\epsilon_{\text{guided}}
    =
    \epsilon_\theta(z_t,t,\varnothing)
    +w\Bigl[
    \epsilon_\theta(z_t,t,c)-
    \epsilon_\theta(z_t,t,\varnothing)
    \Bigr].
    \]

21. **Read the formula in words**  
    Start with what the model would create unconditionally; move further in the direction the prompt changes the prediction.

22. **Board example**  
    Let unconditioned prediction be \(2\), conditional be \(1.5\), and \(w=4\). Then guided prediction is \(2+4(1.5-2)=0\). Explain that a bigger \(w\) amplifies prompt influence—it is not a probability.

23. **Guidance-scale trade-off**

    | Low guidance | High guidance |
    |---|---|
    | diverse, weaker prompt adherence | strong prompt adherence, can look oversaturated or repetitive |

24. **Prompt wording is distribution matching**  
    A prompt is a condition, not a spell. Different phrasing changes text embeddings and should be understood as steering the model within its learned training distribution.

---

## Part IV — Sampling, editing, and the engineering trade-off

25. **Sampling is an iterative solver**  
    The reverse update need not use every original training step. Faster samplers choose a shorter path through the same denoising model.

26. **DDPM versus accelerated deterministic-style sampling**

    | | Many stochastic steps | Fewer scheduled steps |
    |---|---|---|
    | Benefit | faithful simple formulation | lower latency |
    | Cost | slow generation | possible quality/diversity trade-off |

    Mention DDIM only as an example; do not derive its update.

27. **Image-to-image**  
    Start from an encoded input image, add noise to an intermediate level, then denoise under a new condition. Noise strength controls faithfulness versus change.

28. **Inpainting**  
    Preserve known regions while repeatedly denoising only a masked region, conditioned on its visible context and prompt.

29. **Control signals**  
    Edges, pose, depth, and segmentation maps constrain *where* content goes. Text often says what; a structural condition says where.

30. **Small reproducible demo**
    - Fix a seed and prompt.
    - Vary guidance scale.
    - Vary number of inference steps.
    - Change one structural condition.

    Ask students to identify which variable changed composition, fidelity, and latency.

31. **Failure modes to discuss honestly**
    - text/image misalignment and unwanted objects;
    - distorted text, hands, or counting;
    - inherited bias and training-data memorization concerns;
    - high compute and energy cost.

32. **The systems bridge**  
    Every extra sampling step calls a large denoiser again. Efficient inference is therefore not a deployment afterthought; it changes what diffusion applications are practical.

33. **Closing line**
    \[
    \text{Diffusion is controllable denoising in a learned representation space.}
    \]
    Next: why memory movement, caching, and compression dominate modern-model inference.

---

---

## Part V — Detailed systems and control slides

## Slide 34 — Freeze or train the autoencoder?

In a common latent-diffusion pipeline:

1. train or obtain an image autoencoder;
2. freeze it;
3. train the denoiser in latent space;
4. decode only final samples.

This separates representation compression from the expensive generative training stage.

---

## Slide 35 — Latent scaling matters

The diffusion noise schedule assumes a predictable latent scale. In practice, latent representations are normalized or scaled so that Gaussian noise and data latents have compatible magnitudes.

The key point: interfaces between learned modules need numerical contracts.

---

## Slide 36 — U-Net resolution hierarchy

At high resolution, the denoiser retains fine texture. It down-samples to lower resolutions to obtain global context, then up-samples while using skip connections to restore detail.

Draw:

\[
64^2\to32^2\to16^2\to\text{bottleneck}\to16^2\to32^2\to64^2.
\]

---

## Slide 37 — Residual blocks and timestep injection

Each block receives spatial features and a timestep embedding:

\[
h'=\operatorname{ResBlock}(h,e_t).
\]

The time embedding modulates the computation so the same visual feature can be handled differently at high and low noise.

---

## Slide 38 — Cross-attention shapes

Let latent spatial features become \(N\) query tokens and text encoder output \(L\) text tokens:

\[
Q\in\mathbb R^{N\times d},
\qquad
K,V\in\mathbb R^{L\times d}.
\]

The attention matrix has shape \(N\times L\): every image position can select relevant prompt tokens.

---

## Slide 39 — Cross-attention interpretation

For a prompt “a red bicycle beside a tree,” a location containing the bicycle can attend more to *bicycle* and *red*, while background locations may attend more to *tree*.

Stress that this is an intuition for information flow, not proof of exact visual grounding.

---

## Slide 40 — Text encoder choices

The denoiser does not consume raw characters. A pretrained text encoder maps tokenized text to contextual embeddings:

\[
\text{prompt}\to\text{tokens}\to(c_1,\ldots,c_L).
\]

This reuses the pretraining and representation-learning ideas from L14–L18.

---

## Slide 41 — CFG training procedure

For each training example:

1. choose condition \(c\);
2. sometimes replace \(c\) by an empty condition;
3. sample \(t\) and \(\epsilon\);
4. train \(\epsilon_\theta(z_t,t,c)\) to predict \(\epsilon\).

One network learns both conditional and unconditional predictions.

---

## Slide 42 — Guidance is vector extrapolation

\[
\epsilon_{\mathrm{cfg}}
=
\epsilon_{\varnothing}
+w(\epsilon_c-\epsilon_{\varnothing}).
\]

For \(w=1\), use the conditional direction. For \(w>1\), extrapolate beyond it. Draw the three vectors in two dimensions.

---

## Slide 43 — CFG misuse check

\[
w\not=\Pr(\text{prompt is correct}).
\]

It is a heuristic scale controlling the strength of one model prediction relative to another. Very high values can make samples less natural or less diverse.

---

## Slide 44 — Seed and reproducibility

A seed fixes the initial latent noise. With the same model, sampler, settings, and condition:

\[
\text{same seed}\Rightarrow\text{same or nearly same trajectory}.
\]

Record seeds, model versions, steps, and guidance when preparing demonstrations.

---

## Slide 45 — Sampling-step budget

| Fewer steps | More steps |
|---|---|
| lower latency | more denoiser calls |
| may miss detail | may improve convergence/quality |
| useful for iteration | useful for final rendering |

Never present a single step count as universally optimal.

---

## Slide 46 — Image-to-image strength

Encode an input to \(z_0\), begin the reverse process from a partially noised \(z_t\), and condition on a new prompt.

\[
\text{more initial noise}\Rightarrow\text{more freedom to change the input}.
\]

---

## Slide 47 — Inpainting mechanics

At each step, combine a denoised prediction for the masked region with a suitably noised version of known pixels outside the mask.

The model must continue to match boundaries and semantics across the visible/edited interface.

---

## Slide 48 — Structural control

| Input condition | What it constrains |
|---|---|
| edges | outline / composition |
| pose | body arrangement |
| depth | geometry |
| segmentation | regions and classes |

The condition gives the denoiser an additional representation of layout; it does not eliminate all generation errors.

---

## Slide 49 — Diffusion Transformers

Patch/token-based diffusion backbones use Transformer blocks rather than a purely convolutional U-Net. The inputs still include noisy latent tokens, timestep information, and condition tokens.

Architecture changes; L21’s denoising loss remains recognizable.

---

## Slide 50 — Fine-tuning a diffusion model, one placement slide

Parameter-efficient adaptation can adjust a small subset of weights to learn a new style or domain. This inherits the LoRA/adaptation trade-offs from the LLM module and raises data-consent questions.

Do not make a first offering depend on a live fine-tuning exercise.

---

## Slide 51 — Where the time goes

Text encoding happens once. The denoiser is called repeatedly for every sampling step. VAE decoding happens once at the end.

This predicts why changing step count affects latency much more than changing prompt length.

---

## Slide 52 — Failure analysis grid

| Symptom | Likely lever to inspect |
|---|---|
| ignores prompt | guidance, text condition, data mismatch |
| wrong composition | structural condition, seed, prompt ambiguity |
| weak detail | step budget, latent/decoder capacity |
| repeated artifacts | data/model limitation, excessive guidance |

Do not promise a deterministic one-to-one diagnosis.

---

## Slide 53 — Provenance and consent

Discuss separately:

- what training data a model may have seen;
- whether output resembles a living artist or a training item;
- how users should label generated content;
- who bears the cost of errors in high-stakes imagery.

Technical control is not the same as permission or accountability.

---

## Slide 54 — Comparison with autoregressive image tokens

| Family | Generate unit | Main inference pattern |
|---|---|---|
| autoregressive visual tokens | one token at a time | causal decoding |
| diffusion | whole latent repeatedly | iterative denoising |

Both can use Transformer representations; their probability factorizations differ.

---

## Slide 55 — Reproducible classroom exercise

Prepare a grid in advance:

1. fixed prompt, vary \(w\);
2. fixed seed, vary steps;
3. fixed settings, vary one word;
4. fixed prompt, add a pose/edge condition.

Students identify which interventions affect meaning, composition, diversity, and latency.

---

## Slide 56 — Retrieval questions

1. Why is a VAE useful even though diffusion provides the generative model?
2. Which tensors supply cross-attention queries, keys, and values?
3. What does classifier-free guidance extrapolate?
4. Why is a seed useful for a scientific demonstration?

---

## Slide 57 — Condition dropout versus prompt dropout

Classifier-free guidance training removes the condition representation, not only a few text words. The model must learn an unconditional denoising estimate suitable for comparison with a fully conditioned estimate.

This is why the empty prompt is a meaningful learned condition.

---

## Slide 58 — Skip connections carry high-frequency detail

The decoder side of a U-Net needs precise location and texture information that may be weakened by a deep bottleneck.

\[
\text{encoder feature at a resolution}
\longrightarrow
\text{matching decoder resolution}.
\]

Skip connections let global reasoning and local detail coexist.

---

## Slide 59 — The sampler is an inference algorithm

The denoiser is trained once; different samplers decide how to traverse its learned reverse field at inference.

This separates:

\[
\text{model parameters}
\qquad\text{from}\qquad
\text{number and type of reverse updates}.
\]

---

## Slide 60 — Batch generation is not free

Generating several images together can improve accelerator utilization, but increases memory use and can affect latency. Latent size, denoiser architecture, and step count all interact.

Connect this directly to the L23 systems discussion.

---

## Slide 61 — Adaptation boundaries

Fine-tuning can change:

- a style;
- a new visual domain;
- a text/image association;
- a safety or refusal behavior.

Students should ask what data licensed the adaptation and whether the base model’s capabilities shift in unintended ways.

---

## Slide 62 — Practical system checklist

For a new diffusion application, document:

1. model and VAE versions;
2. prompt / structural condition;
3. seed;
4. sampler and step budget;
5. guidance scale;
6. post-processing and safety filters.

This turns an impressive image into a reproducible experimental artifact.

---

## Slide 63 — Common misconceptions

| Misconception | Correction |
|---|---|
| “Text is directly painted into an image.” | Text embeddings condition denoising through learned attention. |
| “Higher guidance always improves quality.” | It trades adherence against naturalness/diversity. |
| “A seed is a semantic label.” | It selects initial noise, not a named concept. |
| “Diffusion editing is exact.” | It is probabilistic and may alter unintended regions. |

---

## Slide 64 — Retrieval questions, extended

1. Why run diffusion in a latent space?
2. How does a U-Net combine global context and detail?
3. What does a cross-attention query represent in this setting?
4. Which input controls composition most directly: a prompt or a pose map?

---

## Slide 65 — Next lecture

\[
\text{a denoiser called many times}
\quad\Rightarrow\quad
\text{memory movement, caching, compression, and scheduling matter}.
\]

L23 shifts from model quality to the systems constraints that make these models usable.

---

## Instructor pacing notes

- **Must teach deeply:** latent diffusion, cross-attention as conditioning, and classifier-free guidance.
- **Keep light:** sampler taxonomy, ControlNet internals, DiT details.
- **Preparation-light demo:** run fixed local or hosted generations only if setup is reliable; otherwise use pre-generated grids with seeds/settings recorded.

## Student takeaways

1. Latent diffusion reduces the cost of iterative denoising by operating in a compact VAE representation.
2. Cross-attention lets spatial denoising features retrieve relevant text information.
3. Classifier-free guidance trades diversity for stronger conditional adherence.
4. Editing and control are variations on conditioning the same reverse-denoising process.
