---
title: "Lecture 20: Generative Adversarial Networks"
status: "Canonical 24-lecture revision — draft outline"
prerequisites: "Binary cross-entropy, gradient descent, latent-variable intuition from L19"
next: "L21 — Diffusion Models: Theory"
---

# Lecture 20: Generative Adversarial Networks
## Learn to generate by playing a game against a critic

## Core story

\[
\boxed{\text{A generator improves by making a discriminator unable to tell real samples from fake ones.}}
\]

GANs do not assign an explicit likelihood to each image. They turn generation into a two-player optimization problem whose reward is a learned notion of realism.

**Target:** 80 minutes, approximately 60 slides.  
**One worked objective:** binary cross-entropy for the discriminator and non-saturating generator loss.  
**One board sketch:** alternating gradients in a minimax game.  
**Out of scope:** exhaustive GAN variants and detailed FID/Inception-score derivations.

---

## Part I — From a latent noise vector to a fake image

1. **Title**  
   *GANs: The Counterfeiter and the Detective*

2. **Recall the generative task**
   \[
   z\sim p(z)\xrightarrow{G_\theta}\tilde x.
   \]
   We need a training signal that says whether \(\tilde x\) looks like data.

3. **Why not use only pixel reconstruction?**  
   A target image is not available for a newly sampled \(z\). More importantly, pixel distance often rewards averages rather than realism.

4. **The two-player setup**
   ~~~text
   real image x ───────────────→ discriminator Dψ(x) → real / fake score
   noise z → generator Gθ(z) → fake image x̃ ────────→ discriminator Dψ(x̃)
   ~~~

5. **Roles**
   - \(G_\theta\): counterfeiter, maps simple noise to samples.
   - \(D_\psi\): detective, separates real data from generated samples.
   - Neither objective makes sense alone; their competition supplies the signal.

6. **A discriminator is a classifier**
   \[
   D_\psi(x)\in(0,1)\approx P(\text{real}\mid x).
   \]

7. **Discriminator objective**
   \[
   \max_\psi\;
   \mathbb E_{x\sim p_{\mathrm{data}}}\log D_\psi(x)
   +\mathbb E_{z\sim p(z)}\log(1-D_\psi(G_\theta(z))).
   \]
   Label real examples 1 and generated examples 0.

8. **Generator’s original minimax objective**
   \[
   \min_\theta\;
   \mathbb E_{z\sim p(z)}\log(1-D_\psi(G_\theta(z))).
   \]
   It changes generated samples so the detective calls them real.

9. **Minimax, in words**
   \[
   \min_G\max_D V(D,G).
   \]
   The discriminator improves at detection; the generator improves at deception.

10. **Board sketch: alternating steps**
    Draw the same fake samples first moving toward data regions, then the discriminator boundary moving to reject them. Emphasize that both players change.

---

## Part II — How GAN training actually works

11. **One training iteration**
    ~~~text
    1. sample a minibatch of real x and noise z
    2. update D to score real high and G(z) low
    3. sample fresh z
    4. freeze D parameters; update G through D
    ~~~

12. **The gradient path to the generator**
    \[
    z\to G_\theta(z)\to D_\psi(G_\theta(z))\to L_G.
    \]
    Gradients pass *through* the frozen discriminator into the generator.

13. **The saturation problem**  
    When \(D(G(z))\approx0\), the original \(\log(1-D)\) generator objective can provide a weak early gradient.

14. **Non-saturating generator loss**
    \[
    L_G=-\mathbb E_{z\sim p(z)}\log D_\psi(G_\theta(z)).
    \]
    Same intended direction—make fakes look real—but a more useful gradient in practice.

15. **Worked scalar example**  
    If \(D(G(z))=0.01\), compare the intuition of “minimize \(\log(1-0.01)\)” with “maximize \(\log 0.01\).” Do not overdo the calculus; show why the latter pushes harder.

16. **What equilibrium would mean**
    If generated and real distributions matched perfectly, no discriminator could do better than guessing:
    \[
    D(x)=\tfrac12.
    \]

17. **Important correction**  
    \(D(x)=1/2\) at equilibrium does **not** mean every generated image is average-looking; it means the discriminator cannot distinguish distributions.

18. **Latent-space interpolation**  
    Sample \(z_0,z_1\), interpolate, generate. Smooth semantic changes are evidence that \(G\) has organized the noise space.

---

## Part III — Why GANs can be hard to train

19. **Moving target problem**  
    Unlike ordinary supervised learning, the loss changes because the opponent is learning simultaneously.

20. **Failure mode: discriminator too strong**  
    It separates real and fake perfectly, leaving weak or uninformative gradients for the generator.

21. **Failure mode: mode collapse**  
    The generator finds a few outputs that fool the current discriminator and repeats them, ignoring large parts of the data distribution.

22. **Visual: coverage versus realism**  
    Plot a real distribution with several modes. Compare blurry broad coverage, sharp partial coverage, and good coverage.

23. **Failure mode: oscillation**  
    Generator and discriminator can chase each other around rather than converge. Training curves alone can be misleading.

24. **Practical stabilizers—one slide only**
    - convolutional architectural choices (DCGAN);
    - balanced update schedules and regularization;
    - better distances/critics (Wasserstein intuition);
    - monitoring samples as well as losses.

25. **Wasserstein intuition, without the full derivation**  
    A critic that scores *how real* samples look can provide a smoother signal than a saturated binary classifier. Name WGAN as a response to unstable geometry, not a fourth objective to memorize.

26. **Conditional GANs**
    \[
    z,y\xrightarrow{G}\tilde x,
    \qquad
    D(x,y).
    \]
    Supply class, text, pose, or another condition so generation is controllable.

---

## Part IV — Place GANs in the generative-model landscape

27. **VAE versus GAN**

    | | VAE | GAN |
    |---|---|---|
    | Training signal | likelihood bound | learned adversarial critic |
    | Inference encoder | yes | not required |
    | Typical trade-off | coverage, smoother samples | sharpness, instability / collapse |
    | Density estimate | explicit approximate model | implicit sampler |

28. **Where GANs remain useful**  
    Image-to-image translation, super-resolution, domain adaptation, and settings with strong paired/conditional structure.

29. **Why diffusion later displaced many GAN use cases**  
    Diffusion trades one fast adversarial pass for many stable denoising steps, usually improving coverage and training reliability at a sampling-cost trade-off.

30. **Quick conceptual check**  
    “If samples look sharp but all resemble the same face, which failure mode is this?” Then ask whether a discriminator loss alone diagnoses it.

31. **Closing line**
    \[
    \text{GANs learn by fooling a critic; diffusion learns by reversing noise.}
    \]
    Next we derive why a simple corruption process can turn generation into supervised denoising.

---

---

## Part V — Detailed adversarial-training slides

## Slide 32 — Binary cross-entropy from labels

For real data label \(y=1\) and generated data label \(y=0\):

\[
L_D
=
-\bigl[
y\log D(x)+(1-y)\log(1-D(x))
\bigr].
\]

The discriminator objective is ordinary binary classification; the unusual part is that its negative class changes as the generator changes.

---

## Slide 33 — A minibatch discriminator loss

\[
L_D=
-\frac1B\sum_{i=1}^{B}\log D(x_i)
-\frac1B\sum_{i=1}^{B}\log(1-D(G(z_i))).
\]

Use a diagram that labels gradients: when updating \(D\), generated images are detached from the generator’s parameters.

---

## Slide 34 — A minibatch generator loss

\[
L_G=
-\frac1B\sum_{i=1}^{B}\log D(G(z_i)).
\]

When updating \(G\), discriminator weights are frozen, but its computation remains in the graph so the generator receives a direction for changing pixels.

---

## Slide 35 — The generator is not trained on “fake” labels

~~~text
update D: real → 1, generated → 0
update G: generated → make D output 1
~~~

The generator does not become a classifier; it changes its images so the discriminator’s learned features regard them as real.

---

## Slide 36 — The discriminator’s boundary moves

Draw two real Gaussian clusters and a generator cluster. After a discriminator update, a separating boundary changes; after a generator update, fake samples move toward regions judged real.

This visual is more valuable than a formal convergence proof for a first pass.

---

## Slide 37 — Ideal equilibrium, carefully stated

For the original GAN objective, an optimal discriminator for a fixed generator has the form:

\[
D^*(x)=
\frac{p_{\mathrm{data}}(x)}
{p_{\mathrm{data}}(x)+p_G(x)}.
\]

At \(p_G=p_{\mathrm{data}}\), it returns \(1/2\). Show this result as intuition for distribution matching; do not derive the calculus.

---

## Slide 38 — Divergence intuition

Substituting the optimal discriminator links the minimax value to a divergence between real and generated distributions.

The important classroom message:

\[
\text{GAN training tries to match distributions, not pair each fake with one real target.}
\]

---

## Slide 39 — Why early generator gradients can fail

If the discriminator easily assigns:

\[
D(G(z))\approx0,
\]

the original minimax generator loss can flatten. The non-saturating loss changes the gradient behaviour while retaining the same desired direction.

---

## Slide 40 — Alternating optimization pseudocode

~~~text
repeat:
    x ← real minibatch; z ← random noise
    update discriminator using x and G(z).detach()
    z ← fresh random noise
    update generator using -log D(G(z))
~~~

Use different colours for parameters changed in each step.

---

## Slide 41 — Update ratio is a design decision

One discriminator step per generator step is common, not a law. Some critic objectives use multiple discriminator/critic updates to keep the signal informative.

This is a first example of why game optimization is more delicate than minimizing one fixed loss.

---

## Slide 42 — Conditioning input can enter both players

\[
G(z,c)\to\tilde x,
\qquad
D(x,c)\to\text{real/fake}.
\]

For digit generation, \(c\) can be a class. For image translation, \(c\) can be an input image, pose, or segmentation map.

---

## Slide 43 — Conditional generation versus selection

| Mechanism | What it does |
|---|---|
| latent \(z\) | chooses one diverse sample |
| condition \(c\) | specifies desired content |
| discriminator / critic | judges realism under \(c\) |

Students should be able to explain why many \(z\) values can produce different valid samples for the same \(c\).

---

## Slide 44 — Architecture: DCGAN principles

Name the practical pattern:

- convolutional up/downsampling;
- normalization choices;
- avoid large dense hidden layers for images;
- progressively build spatial resolution.

Treat DCGAN as an architectural stabilization milestone, not as a paper to memorize.

---

## Slide 45 — Generator geometry

~~~text
z ∈ Rᵈ → learned upsampling blocks → image
~~~

The generator maps a simple connected latent distribution into a complicated data manifold. Interpolation follows a path on this learned map.

---

## Slide 46 — Mode collapse visual, expanded

Show eight \(z\) values:

- **good generator:** produces varied faces/classes/poses;
- **collapsed generator:** produces near variants of one convincing sample.

The discriminator can be fooled temporarily even though coverage is poor.

---

## Slide 47 — Mode dropping is subtler

The generator may cover several modes but omit rare data regions. This can look impressive in a sample grid while failing fairness, safety, or scientific-use requirements.

Quality and coverage must be evaluated separately.

---

## Slide 48 — Loss curves are not enough

GAN losses may oscillate or change scale as the opponent changes. A lower discriminator loss does not monotonically mean better samples.

Always inspect:

- fixed-noise sample grids over training;
- diversity;
- task-specific downstream measures.

---

## Slide 49 — Precision and recall for generation

Informally:

- generation precision: do samples lie near the data manifold?
- generation recall: does the generator cover the manifold?

Sharp collapsed outputs can have high precision and poor recall.

---

## Slide 50 — FID, one careful slide

Fréchet-style metrics compare statistics of features extracted from real and generated images.

Useful warning:

\[
\text{one scalar metric}\ne\text{complete judgment of quality, diversity, bias, or memorization}.
\]

Do not derive the matrix square root in the first offering.

---

## Slide 51 — Wasserstein intuition

If two distributions have disjoint support early in training, a binary discriminator can become overconfident. A distance-like critic can provide a smoother direction indicating how fake mass should move toward real mass.

Use an “earth mover” visual: piles of probability mass moved across a line.

---

## Slide 52 — The Lipschitz constraint, at a glance

Wasserstein critics require controlled sensitivity:

\[
|f(x)-f(y)|\leq K\|x-y\|.
\]

Mention weight clipping and gradient penalties only as ways to enforce this approximate property. Do not derive WGAN-GP.

---

## Slide 53 — Style-based generation, one placement slide

Separate latent processing can control coarse-to-fine visual attributes at different layers. This is a useful example of architecture shaping the meaning of a latent, not a required implementation section.

---

## Slide 54 — Image-to-image GANs

~~~text
source image + condition → generator → translated image
                                     ↓
                         discriminator judges realism / condition
~~~

Examples: super-resolution, day-to-night, domain adaptation. Contrast with diffusion editing in L22.

---

## Slide 55 — Limited data and memorization

With small or narrow datasets, a generator can memorize or nearly copy examples. A visually persuasive sample grid does not by itself establish generalization.

Include nearest-neighbour checks and data-governance discussion.

---

## Slide 56 — GANs and representations

The discriminator’s intermediate features often become useful representations because distinguishing realism requires semantic and texture-sensitive features.

This ties GANs back to L17: many generative objectives incidentally learn representations, but their evaluation targets differ.

---

## Slide 57 — VAE, GAN, diffusion: first comparison

| Question | VAE | GAN | Diffusion |
|---|---|---|---|
| What trains it? | likelihood bound | adversarial signal | known-noise regression |
| What is difficult? | latent/reconstruction trade-off | game stability | iterative sampling |
| What do we emphasize? | probabilistic latent story | realism and coverage | denoising dynamics |

---

## Slide 58 — Classroom diagnosis

Present four symptoms:

1. fake samples repeat;
2. discriminator accuracy is nearly perfect;
3. samples drift around training;
4. outputs are diverse but implausible.

Students identify a likely issue and one intervention category—without pretending one symptom has a unique fix.

---

## Slide 59 — Common misconceptions

| Misconception | Correction |
|---|---|
| “The discriminator is used at generation time.” | Usually only \(G(z)\) is needed after training. |
| “\(D=1/2\) means bad classifier.” | It is ideal only when distributions match. |
| “Sharp samples imply good generation.” | Mode collapse can be sharp and incomplete. |
| “GANs maximize likelihood.” | Standard GANs are implicit adversarial models. |

---

## Slide 60 — Retrieval questions

1. Why does the generator need gradients through a frozen discriminator?
2. What changes in non-saturating loss?
3. What is the difference between mode collapse and poor visual fidelity?
4. Why do game losses resist simple interpretation?

---

## Slide 61 — Next lecture

\[
\text{GAN: learn a critic to distinguish fake from real}
\qquad\longrightarrow\qquad
\text{diffusion: learn to reverse a known noise process}.
\]

The next objective has explicit supervised targets at every corruption level.

---

## Instructor pacing notes

- **Must teach deeply:** the two roles, the discriminator BCE, non-saturating generator loss, and mode collapse.
- **Keep light:** WGAN mechanics, StyleGAN, quantitative generation metrics.
- **Preparation-light demo:** traverse a pretrained GAN latent space and show a mode-collapse animation or fixed-noise training snapshots.

## Student takeaways

1. A GAN is a two-player optimization problem, not a generator plus a conventional label.
2. The discriminator supplies a learned realism signal, and its gradients train the generator.
3. Sharp samples do not guarantee coverage; mode collapse is the central warning.
4. Diffusion addresses a different training problem: denoising known corruptions rather than fooling a moving critic.
