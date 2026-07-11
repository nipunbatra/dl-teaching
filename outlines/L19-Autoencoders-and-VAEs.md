---
title: "Lecture 19: Autoencoders and Variational Autoencoders"
status: "Canonical 24-lecture revision — draft outline"
prerequisites: "Likelihood, KL divergence, reparameterization; neural-network training"
next: "L20 — Generative Adversarial Networks"
---

# Lecture 19: Autoencoders and Variational Autoencoders
## From compressing inputs to sampling a distribution

## Core story

An autoencoder learns a useful code for reconstructing an input. A VAE turns that code into a *probabilistic latent variable*, so nearby latent points decode to plausible new samples.

\[
\boxed{\text{reconstruction} + \text{regularized latent distribution} = \text{a generative model}}
\]

**Target:** 80 minutes, approximately 65 slides.  
**One derivation:** the ELBO as reconstruction minus KL regularization.  
**One board example:** sample \(z=\mu+\sigma\epsilon\) and decode it.  
**Out of scope:** normalizing flows, detailed importance sampling, and a broad latent-model survey.

---

## Part I — Why reconstruction alone is not generation

1. **Title**  
   *Autoencoders and VAEs: a map of data through latent space*

2. **Generative-modeling question**
   \[
   \text{Learn }p_\theta(x)\text{ well enough to generate plausible }x.
   \]
   Contrast this with a classifier, which only learns \(p(y\mid x)\).

3. **The autoencoder**
   \[
   x\xrightarrow{\text{encoder }f_\phi}z
   \xrightarrow{\text{decoder }g_\theta}\hat x.
   \]

4. **The bottleneck intuition**  
   If \(z\) is constrained, reconstruction requires retaining salient structure rather than copying every input coordinate.

5. **Reconstruction objective**
   \[
   L_{\text{AE}}=\sum_i\lVert x_i-g_\theta(f_\phi(x_i))\rVert_2^2.
   \]
   Connect squared error to a Gaussian-output likelihood.

6. **What an autoencoder can do well**  
   Denoising, compression, visualizing low-dimensional codes, and learning features for another task.

7. **The sampling failure**  
   Encoded training points may occupy disconnected, irregular regions of latent space. Random \(z\) can land between them and decode to nonsense.

8. **Visual: latent-space islands**  
   Show clusters with empty gaps; draw random samples from a standard Gaussian falling into gaps.

9. **Design requirement for a generative latent space**
   \[
   z\sim p(z)\quad\Rightarrow\quad g_\theta(z)\text{ should be plausible.}
   \]

---

## Part II — The VAE probabilistic model

10. **Generative story**
    \[
    z\sim p(z)=\mathcal N(0,I),
    \qquad
    x\sim p_\theta(x\mid z).
    \]
    Sample a simple latent cause, then render an observation from it.

11. **A decoder is a conditional distribution**  
    For images it outputs parameters of a pixel distribution; the usual reconstruction image is the conditional mean.

12. **Why inference is hard**
    \[
    p_\theta(z\mid x)=\frac{p_\theta(x\mid z)p(z)}{p_\theta(x)}.
    \]
    The normalizing evidence \(p_\theta(x)\) requires integrating over all \(z\).

13. **Introduce an approximate posterior**
    \[
    q_\phi(z\mid x)=\mathcal N\bigl(\mu_\phi(x),\operatorname{diag}(\sigma_\phi(x)^2)\bigr).
    \]
    The encoder now predicts a distribution, not one deterministic point.

14. **VAE diagram**
    ~~~text
    x → encoder → (μ(x), σ(x)) → sample z → decoder → distribution over x
          ↑                                            │
          └──────── approximation to latent cause ─────┘
    ~~~

15. **The learning target**
    \[
    \log p_\theta(x)\ge
    \underbrace{\mathbb E_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]}_{\text{reconstruct}}
    -
    \underbrace{D_{\mathrm{KL}}(q_\phi(z\mid x)\Vert p(z))}_{\text{organize latent space}}.
    \]

16. **Name the ELBO**  
    Evidence lower bound: maximize a tractable lower bound on data likelihood. Emphasize what each term *does* before every algebraic detail.

17. **Reconstruction term**  
    It says: sample a plausible latent explanation of \(x\), then require the decoder to explain \(x\).

18. **KL term**  
    It says: the explanation for each example should remain close to the shared prior \(\mathcal N(0,I)\).

19. **Gaussian KL formula—show, do not derive**
    \[
    D_{\mathrm{KL}}(\mathcal N(\mu,\sigma^2)\Vert\mathcal N(0,1))
    =\tfrac12\sum_j(\mu_j^2+\sigma_j^2-\log\sigma_j^2-1).
    \]

20. **The tension**  
    Reconstruction wants every input to have its own code. KL wants all codes to look like one simple cloud. Good VAEs balance both.

---

## Part III — Backpropagation through sampling

21. **The apparent obstacle**  
    Sampling \(z\sim\mathcal N(\mu,\sigma^2)\) seems to block gradients from the decoder back to \(\mu\) and \(\sigma\).

22. **Reparameterization trick**
    \[
    \epsilon\sim\mathcal N(0,I),
    \qquad
    z=\mu+\sigma\odot\epsilon.
    \]
    The noise is external; the neural-network computation is differentiable.

23. **Board calculation**  
    If \(\mu=[1,-1]\), \(\sigma=[0.5,0.2]\), and \(\epsilon=[-0.4,1.5]\), calculate \(z=[0.8,-0.7]\). Ask what changes if \(\sigma\) grows.

24. **Training loop**
    ~~~text
    x → μ, log σ² → ε ∼ N(0,I) → z = μ + σ ⊙ ε → decoder
      → reconstruction loss + KL loss → backpropagate
    ~~~

25. **Generation loop**
    ~~~text
    z ∼ N(0,I) → trained decoder → new x
    ~~~
    Training begins with data; generation does not.

26. **Interpolation as a diagnostic**  
    Interpolate two latent codes and decode the path. A useful latent space yields gradual semantic change, not a series of invalid images.

27. **Conditional VAE**
    \[
    p_\theta(x\mid z,y),\qquad q_\phi(z\mid x,y).
    \]
    Use a class label, text embedding, or other condition to choose what is generated.

28. **\(\beta\)-VAE trade-off**
    \[
    L=-\mathbb E_q[\log p_\theta(x\mid z)]
      +\beta D_{\mathrm{KL}}(q_\phi(z\mid x)\Vert p(z)).
    \]
    Higher \(\beta\) can produce more organized factors but worse reconstruction.

---

## Part IV — Honest limitations and the bridge onward

29. **Blurry samples: why?**  
    Pixelwise likelihoods often average multiple plausible outcomes. Average of several sharp possibilities can look blurry.

30. **Posterior collapse**  
    A very powerful decoder may ignore \(z\), making \(q(z\mid x)\) close to the prior. State the symptom, not a long catalogue of fixes.

31. **VAE versus deterministic autoencoder**

    | | Autoencoder | VAE |
    |---|---|---|
    | Code | one point | distribution |
    | Sampling | unreliable | defined by prior |
    | Objective | reconstruction | likelihood bound |
    | Typical output | useful features | smooth, often softer samples |

32. **VAE versus GAN—preview**  
    VAEs explicitly model a latent probabilistic story; GANs learn by trying to fool a discriminator. Both use a latent \(z\), but their learning signals differ radically.

33. **Connection to diffusion**  
    Latent diffusion models often run the diffusion process in the compact latent space of an autoencoder, then decode once at the end.

34. **Exit ticket**  
    Ask students to label the reconstruction, KL, and sampling steps in a small VAE diagram; then predict what happens if the KL term is removed.

35. **Closing line**
    \[
    \text{VAEs make latent spaces sampleable by regularizing them.}
    \]
    Next, GANs trade explicit likelihoods for an adversarial game that can generate sharper samples.

---

---

## Part V — Detailed latent-variable slides

## Slide 36 — Autoencoders are nonlinear compression

PCA finds a linear low-dimensional reconstruction:

\[
x\longrightarrow W^\top x\longrightarrow WW^\top x.
\]

An autoencoder replaces the two linear maps with neural networks and can therefore learn nonlinear manifolds. This is an intuition link, not a claim that every autoencoder is PCA.

---

## Slide 37 — Shape bookkeeping

For an image tensor:

\[
x\in\mathbb R^{H\times W\times C}
\]

an encoder may produce:

\[
z\in\mathbb R^d.
\]

Draw an example with a \(32\times32\) image compressed to a 16-dimensional latent, then ask what information reconstruction must preserve.

---

## Slide 38 — Undercomplete and overcomplete codes

| Code dimension | Pressure on the model |
|---|---|
| \(d<\dim(x)\) | compress salient structure |
| \(d\geq\dim(x)\) | may learn an identity shortcut |
| overcomplete + regularization | can still learn sparse / robust features |

The bottleneck may be dimensional, stochastic, architectural, or regularization-based.

---

## Slide 39 — Reconstruction likelihood, not just MSE

If:

\[
p_\theta(x\mid z)=\mathcal N(g_\theta(z),\sigma_x^2I),
\]

then negative log-likelihood is proportional to squared reconstruction error.

This grounds the familiar AE loss in the probabilistic language used throughout the course.

---

## Slide 40 — Data type determines decoder output

| Data | Possible decoder distribution | Common reconstruction term |
|---|---|---|
| normalized continuous pixels | Gaussian | squared error |
| binary image | Bernoulli | binary cross-entropy |
| tokens | categorical | cross-entropy |

The decoder predicts distribution parameters, not merely an image file.

---

## Slide 41 — The VAE graphical model

~~~text
     z ∼ N(0,I)
       ↓
 x ∼ pθ(x | z)
~~~

At generation time, the direction is simple. At training time, infer the hidden cause \(z\) that plausibly generated each observed \(x\).

---

## Slide 42 — Marginal likelihood is an integral

\[
p_\theta(x)=\int p_\theta(x\mid z)p(z)\,dz.
\]

For a nonlinear neural decoder and high-dimensional \(z\), this integral cannot usually be evaluated exactly.

---

## Slide 43 — Variational inference inserts a tractable distribution

Choose:

\[
q_\phi(z\mid x)
\]

to imitate the true posterior \(p_\theta(z\mid x)\). The encoder is an *amortized inference network*: one set of parameters handles every input rather than solving a new optimization problem per image.

---

## Slide 44 — The exact decomposition

\[
\log p_\theta(x)
=
\operatorname{ELBO}(x)
+
D_{\mathrm{KL}}\bigl(q_\phi(z\mid x)\Vert p_\theta(z\mid x)\bigr).
\]

Because KL is non-negative, the ELBO is a lower bound. This is the cleanest explanation for its name.

---

## Slide 45 — Expand the ELBO

\[
\operatorname{ELBO}(x)
=
\mathbb E_{q_\phi(z\mid x)}
[\log p_\theta(x,z)-\log q_\phi(z\mid x)].
\]

Then substitute:

\[
p_\theta(x,z)=p_\theta(x\mid z)p(z).
\]

---

## Slide 46 — Rearrange into the two familiar terms

\[
\operatorname{ELBO}
=
\mathbb E_q[\log p_\theta(x\mid z)]
-
D_{\mathrm{KL}}(q_\phi(z\mid x)\Vert p(z)).
\]

Use arrows and colour: the first term explains the data; the second makes latent explanations sampleable from one common prior.

---

## Slide 47 — Which KL is which?

\[
D_{\mathrm{KL}}(q(z\mid x)\Vert p(z))
\]

regularizes each encoded example toward the prior.

\[
D_{\mathrm{KL}}(q(z\mid x)\Vert p(z\mid x))
\]

is the remaining gap between the ELBO and exact log-likelihood. Students commonly conflate these two KL expressions.

---

## Slide 48 — Training loss signs

We maximize the ELBO, equivalently minimize:

\[
L_{\mathrm{VAE}}
=
-\mathbb E_q[\log p_\theta(x\mid z)]
+
D_{\mathrm{KL}}(q_\phi(z\mid x)\Vert p(z)).
\]

State explicitly which term is called “reconstruction loss” in implementation.

---

## Slide 49 — Monte Carlo estimate

The expectation over \(q_\phi(z\mid x)\) is approximated with one or a few sampled latent vectors:

\[
\mathbb E_q[\cdot]\approx \frac1K\sum_{k=1}^{K}[\cdot]_{z^{(k)}}.
\]

One reparameterized sample per input is often enough for the standard training objective.

---

## Slide 50 — Why output log-variance?

Networks commonly predict:

\[
\ell=\log\sigma^2
\]

then use:

\[
\sigma=\exp(\tfrac12\ell).
\]

This keeps variance positive while allowing unconstrained real-valued network outputs.

---

## Slide 51 — Reparameterization computation graph

~~~text
x → encoder → μ, log σ²
                    ↓
ε ∼ N(0,I) → z = μ + σ ⊙ ε → decoder → likelihood
~~~

The randomness is sampled independently; gradients flow through the deterministic transformation from \(\mu,\sigma\) to \(z\).

---

## Slide 52 — Sampling versus reconstruction

| Operation | Latent source | Question answered |
|---|---|---|
| reconstruction | \(z\sim q(z\mid x)\) | can the model explain this input? |
| unconditional sample | \(z\sim p(z)\) | what does the learned distribution generate? |
| interpolation | path between two codes | is latent geometry smooth? |

---

## Slide 53 — A KL trade-off numeric intuition

Imagine two candidate encoders:

| Encoder | Reconstruction term | KL term |
|---|---:|---:|
| highly individualized codes | 1 | 100 |
| codes near prior | 10 | 1 |

Neither is automatically preferable; the objective chooses a balance. This avoids the misconception that lower reconstruction error alone wins.

---

## Slide 54 — Visualize a two-dimensional latent space

Encode a digit or face dataset into \(z_1,z_2\). Place a decoder-generated image at points on a grid.

The visual should distinguish:

- clusters;
- smooth directions;
- empty regions;
- areas where prior samples land.

---

## Slide 55 — Latent traversal

Hold every coordinate fixed except \(z_j\):

\[
z_j\leftarrow z_j+\delta.
\]

Decode several values of \(\delta\). Interpretable changes are interesting evidence about the learned representation, but not proof that factors are perfectly disentangled.

---

## Slide 56 — Conditional VAE, expanded

For a label or text condition \(c\):

\[
q_\phi(z\mid x,c),
\qquad
p_\theta(x\mid z,c).
\]

At sampling time, choose \(c\), sample \(z\sim p(z)\), then decode. Explain what is controlled by \(c\) versus what remains diverse through \(z\).

---

## Slide 57 — \(\beta\)-VAE and disentanglement claims

\[
L=
-\mathbb E_q[\log p_\theta(x\mid z)]
+\beta D_{\mathrm{KL}}(q\Vert p).
\]

Increasing \(\beta\) strengthens pressure toward a simple prior. It can encourage more separated factors on curated data, but “disentangled” is not a guaranteed, universal outcome.

---

## Slide 58 — Posterior collapse, diagnose it

Symptom:

\[
q_\phi(z\mid x)\approx p(z)
\]

for every \(x\), while a powerful decoder predicts well from context alone. The latent has become unused, not magically regularized.

---

## Slide 59 — Posterior collapse, high-level responses

- weaken or schedule the KL pressure early in training;
- constrain decoder capacity;
- force more information through the latent;
- inspect latent-use statistics rather than only reconstructions.

Name responses as design choices, not a new technical module.

---

## Slide 60 — VQ-style discrete latents

Some autoencoders encode into a finite codebook rather than a Gaussian continuous latent. This is a useful contrast:

\[
\text{continuous VAE latent}
\qquad\text{versus}\qquad
\text{discrete codebook token}.
\]

It previews tokenized visual representations without deriving vector quantization.

---

## Slide 61 — Why VAEs matter to modern diffusion

Latent diffusion uses an autoencoder to move:

\[
\text{pixels}\longrightarrow\text{compact continuous latents}.
\]

Diffusion then models the latent distribution. The VAE is therefore not historical background; it is an active systems component.

---

## Slide 62 — How should generative models be evaluated?

Use more than one lens:

- sample quality and diversity;
- reconstruction or likelihood-related metrics where appropriate;
- downstream utility;
- human evaluation with a transparent protocol;
- memorization and data-provenance checks.

---

## Slide 63 — Common misconceptions

| Misconception | Correction |
|---|---|
| “The encoder outputs one latent point.” | It outputs a distribution’s parameters. |
| “KL makes samples better directly.” | It organizes latent codes to match the sampling prior. |
| “Blurry samples prove VAEs are wrong.” | They reflect likelihood/output trade-offs. |
| “Interpolation proves semantic understanding.” | It is a useful diagnostic, not proof. |

---

## Slide 64 — Retrieval questions

1. Why is an ordinary autoencoder not guaranteed to generate from random \(z\)?
2. What does the KL term compare in a VAE?
3. Why does \(z=\mu+\sigma\epsilon\) permit backpropagation?
4. What is posterior collapse?

---

## Slide 65 — Next lecture

\[
\text{VAE: explain data through a latent likelihood}
\qquad\longrightarrow\qquad
\text{GAN: learn realism through an adversarial critic}.
\]

The next model removes the explicit reconstruction target and introduces a game.

---

## Instructor pacing notes

- **Must teach deeply:** latent-space islands; the two ELBO terms; reparameterization; sampling versus reconstruction.
- **Keep light:** the full ELBO derivation, \(\beta\)-VAE variants, posterior-collapse remedies.
- **Preparation-light demo:** encode and interpolate between two examples using a small pretrained VAE; no live training.

## Student takeaways

1. Reconstruction alone does not make random latent samples meaningful.
2. The VAE learns a decoder and an approximate posterior over latent causes.
3. The KL term is what turns individual codes into a shared sampleable latent space.
4. Reparameterization makes stochastic latent-variable learning compatible with backpropagation.
