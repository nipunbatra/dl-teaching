---
title: "Lecture 21: Diffusion Models — Theory"
status: "Canonical 24-lecture revision — draft outline"
prerequisites: "Gaussian noise, likelihood intuition, neural-network regression"
next: "L22 — Diffusion Models: Practice"
---

# Lecture 21: Diffusion Models — Theory
## Generation as learning to reverse a known corruption process

## Core story

\[
\boxed{\text{If we know how to add noise, we can train a network to remove it.}}
\]

A diffusion model gradually turns data into Gaussian noise, then learns the reverse sequence one small denoising step at a time.

**Target:** 80–90 minutes, approximately 60 slides.  
**One derivation:** from forward Gaussian noise to the practical \(\epsilon\)-prediction loss.  
**One board example:** construct \(x_t\) from \(x_0\), \(\bar\alpha_t\), and sampled noise.  
**Out of scope:** full SDE formalism, probability-flow ODE proofs, and every sampler variant.

---

## Part I — The generative problem becomes denoising

1. **Title**  
   *Diffusion Models: From Noise Back to Data*

2. **Why a new family?**  
    VAEs organize latents; GANs learn an adversarial critic. Diffusion starts with a corruption process we design and can sample from exactly.

3. **Visual hook: the noising filmstrip**  
    Show \(x_0\) (image) becoming increasingly noisy through \(x_1,\ldots,x_T\), ending at static \(x_T\).

4. **Forward process**
    \[
    q(x_t\mid x_{t-1})=
    \mathcal N\left(\sqrt{1-\beta_t}\,x_{t-1},\beta_tI\right).
    \]
    \(\beta_t\) is a small variance schedule.

5. **Notation simplification**
    \[
    \alpha_t=1-\beta_t,
    \qquad
    \bar\alpha_t=\prod_{s=1}^{t}\alpha_s.
    \]

6. **One-step intuition**  
    Each step shrinks the existing signal slightly and adds a small independent Gaussian perturbation.

7. **The useful closed form**
    \[
    q(x_t\mid x_0)=
    \mathcal N\left(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I\right).
    \]
    We can jump to any noise level without simulating every earlier step.

8. **Sampling a noisy training input**
    \[
    x_t=\sqrt{\bar\alpha_t}x_0+
    \sqrt{1-\bar\alpha_t}\,\epsilon,
    \qquad\epsilon\sim\mathcal N(0,I).
    \]

9. **Board calculation**  
    Let \(x_0=1\), \(\bar\alpha_t=0.64\), and \(\epsilon=-0.5\). Calculate \(x_t=0.8+0.6(-0.5)=0.5\). Identify signal and noise contributions.

10. **At large \(t\)**  
    \(\bar\alpha_t\) is near zero, so \(x_T\) is approximately standard Gaussian noise—easy to sample.

---

## Part II — Learn the reverse process

11. **The impossible-looking direction**  
    From one noisy image, many clean images might be plausible. The reverse is therefore probabilistic.

12. **Parameterized reverse step**
    \[
    p_\theta(x_{t-1}\mid x_t)=
    \mathcal N\bigl(\mu_\theta(x_t,t),\Sigma_\theta(x_t,t)\bigr).
    \]

13. **The prediction task we can supervise**  
    During training we know both \(x_0\) and the sampled noise \(\epsilon\). Teach the network to estimate that noise from \((x_t,t)\).

14. **Noise-prediction network**
    \[
    \epsilon_\theta(x_t,t)\approx\epsilon.
    \]
    It must know the timestep because “remove a little noise” and “remove almost all noise” are different tasks.

15. **Practical DDPM loss**
    \[
    L_{\text{simple}}=
    \mathbb E_{x_0,\epsilon,t}
    \left[\lVert\epsilon-\epsilon_\theta(x_t,t)\rVert_2^2\right].
    \]
    This is supervised regression with self-generated targets.

16. **A useful reframe**  
    Diffusion is self-supervised denoising: every clean example produces unlimited noisy inputs and exact noise labels.

17. **From predicted noise to a clean estimate**
    \[
    \hat x_0=
    \frac{x_t-\sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t,t)}
    {\sqrt{\bar\alpha_t}}.
    \]
    No need to derive the full reverse mean; make the reconstruction intuition visible.

18. **Reverse sampling loop**
    ~~~text
    xT ∼ N(0,I)
    for t = T, …, 1:
        predict εθ(xt, t)
        use it to sample / estimate x(t−1)
    return x0
    ~~~

19. **Why generation is slow**  
    A GAN maps one \(z\) to an image in one forward pass. Diffusion applies a denoiser many times; quality and flexibility trade against latency.

---

## Part III — Interpret the geometry without over-deriving it

20. **Denoising at multiple scales**  
    Early reverse steps recover broad structure from near-pure noise; later steps restore fine texture and detail.

21. **Noise schedule is a curriculum**  
    Training includes easy denoising and hard denoising. A time embedding tells the model which difficulty it faces.

22. **The score connection—one conceptual slide**
    \[
    \nabla_x\log p_t(x)
    \]
    points toward higher-density regions. Predicting Gaussian noise can be transformed into an estimate of this direction. Mention this as a bridge to score-based models; do not begin an SDE derivation.

23. **What the network learns**  
    Not a database of clean images, but local directions from noisy points toward regions the data distribution considers plausible.

24. **Conditional generation preview**  
    Add a class label, a text embedding, a mask, or a low-resolution image as conditioning information so denoising follows a desired trajectory.

25. **Three generative families so far**

    | Family | Training signal | Sampling |
    |---|---|---|
    | VAE | likelihood bound | one latent decode |
    | GAN | adversarial critic | one generator pass |
    | Diffusion | predict known noise | repeated denoising |

26. **What diffusion gains and pays**
    - gains: stable supervised-style training, broad coverage, flexible conditioning;
    - pays: many inference steps and substantial compute.

27. **Knowledge check**  
    If \(t\) is large, should the model rely more on the noisy input or its learned data prior? Explain why.

28. **Closing bridge**  
    Theory says “predict noise repeatedly.” Practice asks: what neural architecture predicts it, how does text steer it, and how can we generate without thousands of steps?

---

---

## Part IV — Detailed diffusion-theory slides

## Slide 29 — The forward chain factorizes

\[
q(x_{1:T}\mid x_0)=\prod_{t=1}^{T}q(x_t\mid x_{t-1}).
\]

Each corruption step depends only on the previous noisy sample. This Markov structure makes the process simple to define even when the data distribution is complicated.

---

## Slide 30 — The reverse model factorizes too

\[
p_\theta(x_{0:T})
=
p(x_T)\prod_{t=T}^{1}p_\theta(x_{t-1}\mid x_t).
\]

At sampling time, start from the easy prior \(p(x_T)=\mathcal N(0,I)\) and apply learned reverse transitions.

---

## Slide 31 — Why the closed form matters computationally

Without the closed form, creating a training pair at timestep \(t\) requires \(t\) sequential noising steps.

With:

\[
x_t=\sqrt{\bar\alpha_t}x_0+
\sqrt{1-\bar\alpha_t}\epsilon,
\]

one random timestep gives one training example in a single operation.

---

## Slide 32 — Training samples all noise difficulties

Training follows five operations:

1. sample a clean \(x_0\);
2. sample a timestep \(t\);
3. sample \(\epsilon\sim\mathcal N(0,I)\);
4. construct \(x_t\) directly;
5. predict \(\epsilon\) from \((x_t,t)\).

The same data point generates a continuum of self-supervised tasks.

---

## Slide 33 — Noise schedule visual

Plot:

\[
\sqrt{\bar\alpha_t}
\quad\text{and}\quad
\sqrt{1-\bar\alpha_t}
\]

against \(t\). The first falls from near one to near zero; the second grows. Interpret the graph as changing signal-to-noise ratio.

---

## Slide 34 — Different timesteps look like different tasks

| Timestep | Input appearance | Main denoising demand |
|---|---|---|
| small \(t\) | mostly clean | remove fine perturbations |
| middle \(t\) | partly structured | recover objects/layout |
| large \(t\) | near Gaussian noise | rely strongly on learned prior |

---

## Slide 35 — Why predict noise rather than the image?

Possible parameterizations include:

\[
\epsilon,\qquad x_0,\qquad v.
\]

Noise prediction gives a simple target with known Gaussian scale across timesteps. Practical systems may use other parameterizations; the diffusion principle is unchanged.

---

## Slide 36 — MSE is a conditional regression loss

\[
\epsilon_\theta(x_t,t)
\]

is trained to approximate the statistically useful estimate of the injected noise. From an ambiguous \(x_t\), the network cannot literally recover one unknowable noise draw independently of the data prior.

---

## Slide 37 — Time must enter the network

The same image-like tensor can arise at multiple noise levels. A model that sees only \(x_t\) cannot know how aggressively to denoise.

Use the processing chain:

> timestep \(t\) → sinusoidal or learned embedding → injected into denoiser blocks.

---

## Slide 38 — Reverse mean intuition

The reverse update combines:

- current noisy state \(x_t\);
- predicted noise \(\epsilon_\theta(x_t,t)\);
- schedule coefficients;
- optional fresh Gaussian noise.

Show arrows rather than the full coefficient-heavy equation in the main deck.

---

## Slide 39 — Stochastic and deterministic sampling paths

Some samplers add randomness at each reverse step; others follow a more deterministic trajectory once the starting noise and condition are fixed.

The distinction affects diversity, reproducibility, and speed—not the basic learned denoiser.

---

## Slide 40 — One sample, many intermediate states

Show a reverse filmstrip:

\[
x_T\to x_{0.8T}\to x_{0.5T}\to x_{0.2T}\to x_0.
\]

Ask students when global layout appears and when texture appears. Treat the answer as an observation, not a universal law.

---

## Slide 41 — Relation to denoising autoencoders

A denoising autoencoder learns:

\[
\text{corrupted input}\to\text{clean input}.
\]

Diffusion extends this across a sequence of carefully calibrated corruption levels and defines a generative sampling process from the reverse chain.

---

## Slide 42 — Variational perspective, one placement slide

The original diffusion objective can be derived from a variational bound on the reverse process. The simple noise-prediction objective is a practical simplification of that likelihood-based view.

Connect to L19’s ELBO without deriving another long bound.

---

## Slide 43 — Score intuition revisited

For a noisy point \(x_t\), the score:

\[
\nabla_{x_t}\log p_t(x_t)
\]

points toward denser regions of the noisy data distribution. Denoising points samples in a related direction. This is geometric intuition, not a vector field students must compute.

---

## Slide 44 — Conditioning changes the reverse distribution

\[
p_\theta(x_{t-1}\mid x_t,c)
\]

uses a condition \(c\) to favor some plausible reverse paths over others. Text, class labels, masks, and low-resolution images all change the same conditional denoising question.

---

## Slide 45 — Classifier guidance preview

An external classifier can supply:

\[
\nabla_{x_t}\log p(c\mid x_t),
\]

nudging a reverse trajectory toward a desired class. This motivates guidance conceptually; L22 teaches the more practical classifier-free version.

---

## Slide 46 — Architectural preview

The loss says *what* to predict. A U-Net or diffusion Transformer says *how* to combine local detail, global context, timestep, and conditioning information.

Do not start an architecture section yet; use this as a forward pointer.

---

## Slide 47 — Why diffusion training is comparatively stable

Each training example has a known target:

\[
\epsilon.
\]

Unlike GAN training, no learned opponent changes the target from batch to batch. This does not make optimization trivial, but it removes the central minimax instability.

---

## Slide 48 — Why diffusion sampling is expensive

| Model family | Typical generation structure |
|---|---|
| VAE | one decode |
| GAN | one generator pass |
| diffusion | repeated denoiser calls |

The model trades inference work for a tractable, stable training signal.

---

## Slide 49 — Common misconceptions

| Misconception | Correction |
|---|---|
| “Diffusion memorizes a reverse noise table.” | A neural network generalizes denoising across inputs and conditions. |
| “The forward process is learned.” | In standard DDPMs it is chosen and fixed. |
| “The network predicts the clean image only.” | Noise, clean image, or velocity can be parameterized. |
| “More steps always mean better image.” | Schedules and samplers create quality/latency trade-offs. |

---

## Slide 50 — Retrieval questions

1. Why can \(x_t\) be sampled directly from \(x_0\)?
2. What target is available during diffusion training without labels?
3. Why must the denoiser know \(t\)?
4. Why is generation slower than GAN sampling?

---

## Slide 51 — The \(\beta_t\) schedule is a design choice

If too little noise is added, \(x_T\) remains structured and is not easy to sample from. If noise is injected too abruptly, intermediate denoising tasks become needlessly hard.

Use a smooth schedule visual and stress that schedules determine the sequence of learning problems.

---

## Slide 52 — Variance in reverse sampling

The learned reverse distribution includes a mean and may use a chosen or learned variance.

\[
p_\theta(x_{t-1}\mid x_t)
=\mathcal N(\mu_\theta,\Sigma_\theta).
\]

Variance controls how much randomness remains in a reverse step; it is not an arbitrary visual-effect knob.

---

## Slide 53 — Loss weighting across timesteps

Not every timestep must contribute equally to a practical objective. Different parameterizations and schedules can change which signal-to-noise regimes dominate learning.

Keep the implementation detail optional; use the slide to reinforce that an “MSE loss” still has modeling choices.

---

## Slide 54 — Data modalities beyond images

Diffusion can model:

- audio waveforms or spectrograms;
- video with temporal structure;
- molecular coordinates;
- trajectories and actions;
- continuous latent representations.

The forward process remains Gaussian corruption, but the denoiser architecture and condition must respect the data’s structure.

---

## Slide 55 — Why direct pixel diffusion is demanding

Noise is added independently to a very high-dimensional tensor. A good denoiser must recover local texture, object identity, global layout, and condition alignment simultaneously.

This motivates L22’s latent space and multi-scale architecture.

---

## Slide 56 — A tiny one-dimensional thought experiment

Suppose data lies near \(-2\) and \(+2\) on a line. At high noise, both modes blend into an approximately Gaussian cloud.

The reverse model must decide which mode a noisy point should approach. This gives a concrete picture of why generation is probabilistic.

---

## Slide 57 — Denoising does not mean nearest training example

For a noisy input between modes, the model uses learned distributional structure to choose a plausible reverse direction. This may resemble interpolation, recombination, or a new sample—not a database lookup.

Separately note that memorization remains an empirical risk to evaluate on narrow datasets.

---

## Slide 58 — Diffusion evaluation lens

Evaluate:

- fidelity;
- coverage/diversity;
- conditioning adherence;
- sampling cost;
- robustness and harmful failure modes.

No single image grid or scalar metric settles all five.

---

## Slide 59 — Theory-to-practice checklist

Before reading an implementation, identify:

1. data space: pixels or latents?
2. denoiser architecture;
3. time/noise parameterization;
4. condition interface;
5. sampler and step budget.

These five questions locate nearly every diffusion system in the course map.

---

## Slide 60 — Common equations, one page

\[
q(x_t\mid x_{t-1})
=
\mathcal N(\sqrt{1-\beta_t}x_{t-1},\beta_tI)
\]

\[
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon
\]

\[
L=\mathbb E\|\epsilon-\epsilon_\theta(x_t,t)\|_2^2.
\]

Use this as a final reference slide, not an additional derivation.

---

## Slide 61 — Retrieval questions, extended

1. Which part of diffusion is fixed and which part is learned?
2. How does the closed form turn one image into many training examples?
3. Why is the reverse direction conditional and probabilistic?
4. What trade-off is introduced by iterative sampling?

---

## Slide 62 — Next lecture

\[
\text{noise-prediction objective}
\quad+\quad
\text{latent space, U-Net/Transformer, and condition}
\quad=\quad
\text{a practical diffusion system}.
\]

L22 builds that system component by component.

---

## Instructor pacing notes

- **Must teach deeply:** forward closed form, \(\epsilon\)-prediction loss, and the reverse loop.
- **Keep light:** variational-bound derivation and score/SDE connection.
- **Preparation-light demo:** a noising slider and a short reverse-denoising animation; show no source code until L22.

## Student takeaways

1. The forward noising distribution is known, so it produces unlimited labeled denoising examples.
2. The network predicts noise conditional on both the corrupted input and its noise level.
3. Sampling reverses the learned process from Gaussian noise to a data sample.
4. Diffusion’s central practical cost is iterative inference.
