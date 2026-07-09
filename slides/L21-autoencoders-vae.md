---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# Autoencoders & the Variational Autoencoder

## Lecture 21 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar*

---

# The whole lecture, in one idea

An autoencoder learns to **compress** — squeeze $x$ to a code $z$, then rebuild it. But a plain autoencoder **cannot generate**: its latent space is full of holes, so a random $z$ decodes to garbage.

<div class="keypoint">

**The fix is one change: encode to a *distribution*, not a *point*.**
The encoder outputs a Gaussian $(\mu,\sigma)$; we sample $z$ from it and pull every such cloud toward $\mathcal N(0,I)$. That single change packs the latent space so that **sampling becomes generating**.

</div>

You built the KL divergence in L1. Today it becomes the load-bearing term of a generative model.

$$\underbrace{\text{compress}}_{\text{autoencoder}} \rightarrow \underbrace{\text{encode to } (\mu,\sigma)}_{\text{VAE}} \rightarrow \underbrace{z=\mu+\sigma\odot\epsilon}_{\text{reparam trick}} \rightarrow \underbrace{\text{recon}+\text{KL}}_{\text{ELBO loss}} \rightarrow \underbrace{z\sim\mathcal N(0,I)\to\text{decode}}_{\text{generate}}$$

---

# How today connects to the rest of the course

<div class="columns">
<div>

**Today reuses ideas you already have:**
- **L1** — the **KL divergence** and the Gaussian NLL; today the KL is the regularizer that shapes the latent
- **ES 335** — **PCA** is exactly a *linear* autoencoder
- **Prob. ML** — variational inference: a tractable $q$ approximating an intractable posterior

</div>
<div>

**And it sets up what's next:**
- **L23 — diffusion** is a VAE idea taken to $T$ layers: sample noise, decode iteratively; Stable Diffusion even runs *inside* a VAE's latent
- the reparameterization trick reappears anywhere you must backprop through a sampled quantity

</div>
</div>

<div class="insight">

One genuinely new mechanic today — the **reparameterization trick**. Everything else is the L1 toolkit (NLL, Gaussians, KL) reassembled.

</div>

---

<!-- _class: section-divider -->

## Part 1 · The autoencoder — learned compression

---

# Encoder → bottleneck → decoder

An **autoencoder** is two networks trained together: an **encoder** $f$ squeezes the input to a small code $z$, a **decoder** $g$ rebuilds it.

$$z = f(x) \quad(\text{encode})\qquad \hat x = g(z) \quad(\text{decode})\qquad \mathcal L = \lVert x - \hat x\rVert^2$$

<div class="keypoint">

There are no labels — the target *is the input*. The network is forced to keep only what it needs to reconstruct. The narrow **bottleneck** $\dim(z)\ll\dim(x)$ is what makes the code informative.

</div>

The "postcard" test: if you must describe a painting in 16 numbers so a partner can repaint it, you keep the *essence*, not every brushstroke.

---

# PCA is a linear autoencoder

You already built one in ES 335. **PCA** projects $x$ onto a few directions (encode), then reconstructs (decode), choosing the projection to minimize $\lVert x-\hat x\rVert^2$ — encode → bottleneck → decode.

<div class="math-box">

PCA constrains *both* maps to be **linear**: $z = W^\top x$, $\hat x = Wz$. So it can only fit a flat subspace.

</div>

Swap those two linear maps for **nonlinear layers** (MLP or CNN) and you get a *learned, nonlinear* compressor that can fold curved manifolds a flat subspace never could.

<div class="insight">

On MNIST, a deep AE reaches a given reconstruction quality at noticeably **fewer latent dims** than PCA needs — curved manifold vs linear subspace. Nonlinearity buys you compression.

</div>

---

# The bottleneck is the whole point

<div class="popquiz">

**Warm-up.** If we let the latent be *as wide as the input* ($\dim z = \dim x$), the AE can reach **zero** reconstruction loss. Has it learned anything useful?

</div>

*Answer:* **No.** With no bottleneck the network just copies — $z=x$, $g(z)=z$. Loss is zero, representation is worthless.

<div class="keypoint">

Compression only happens when the bottleneck **forces** it: $\dim z \ll \dim x$ makes the encoder throw away everything redundant and keep the informative structure. That is why AEs learn useful features with **no labels**.

</div>

Modern variants force it differently — add noise (**denoising AE**) or mask patches (**MAE**) instead of a tiny bottleneck. Same idea, different pressure.

---

# Intuition · what survives the bottleneck?

A trained autoencoder is **lossy compression the network designs itself.** JPEG discards high-frequency detail your eye barely notices; an autoencoder learns *which* details to discard — tuned to *this* dataset — with nobody specifying the rule.

<div class="insight">

Train on faces and the code keeps pose, lighting, and identity while dropping sensor noise. Train on MNIST and it keeps stroke shape, not pixel-exact ink. **The bottleneck is a budget, and the encoder spends it on whatever most reduces reconstruction error** — an automatic, data-specific compressor.

</div>

<div class="popquiz">

**Quick check.** Two autoencoders, same architecture: one trained only on cats, one only on street signs. Feed a cat photo through *both*. Which reconstructs it better, and why?

</div>

*Answer:* the **cat** AE — it spent its bottleneck budget learning cat structure. The sign AE kept edges and flat colours useful for signs, so it rebuilds the cat poorly. **A code is only as good as the distribution it was trained to compress.**

---

# Worked numeric · a 4-pixel autoencoder

Tiny grayscale image $x = [0.9,\ 0.2,\ 0.8,\ 0.1]$, latent dim $d=1$.

<div class="math-box">

1. **Encode.** Network outputs $z = 0.7$.
2. **Decode.** $z=0.7 \to \hat x = [0.8,\ 0.3,\ 0.7,\ 0.2]$.
3. **Loss (MSE).**
$$\mathcal L = \tfrac14\big((0.9{-}0.8)^2+(0.2{-}0.3)^2+(0.8{-}0.7)^2+(0.1{-}0.2)^2\big) = \tfrac14(4\times 0.01) = \mathbf{0.01}$$

</div>

Backprop pushes encoder + decoder weights to make this smaller. One scalar $z$ had to summarize four pixels — that is the compression.

---

# What autoencoders are good for

<div class="columns">
<div>

**Uses that don't need generation:**
- **dimensionality reduction** (nonlinear PCA)
- **denoising** — corrupt the input, ask for the clean target
- **pretraining / feature learning** without labels
- **anomaly detection** — high reconstruction error = out of distribution

</div>
<div>

**What it still can't do:**
- **generate** new samples. Ask it to *invent* a digit and you get garbage.

That gap is the rest of the lecture.

</div>
</div>

<div class="notebook">

**📓 Notebook · AE → VAE → β-VAE on MNIST** — trains a plain AE, watches reconstructions sharpen as the latent grows, then upgrades it to a VAE. *(Prob. ML · `notebooks/vae.ipynb`)*

</div>

---

<!-- _class: section-divider -->

## Part 2 · Why a plain autoencoder cannot generate

---

# Generative modeling · the task

<div class="keypoint">

Given $N$ i.i.d. samples $\{x_1,\dots,x_N\}$ from an unknown $p_{\text{data}}$, learn a model we can **sample** from — draw new $x\sim p_{\text{model}}$ with $p_{\text{model}}\approx p_{\text{data}}$.

</div>

The **latent-variable** recipe: posit a hidden $z$ that *generates* $x$.

$$z \sim p(z) \qquad x \sim p(x\mid z)$$

Images live in $\mathbb R^{100{,}000+}$ but on a thin, low-dimensional manifold. Writing $p_{\text{data}}$ down analytically is hopeless; learning a $z\to x$ decoder from samples is the whole game. An AE gives us the decoder — but not a rule for which $z$ to feed it.

---

# The latent space has holes

<div class="popquiz">

**Work it out.** Train a plain AE on MNIST, then sample $z\sim\mathcal N(0,I)$ and decode it. Do you get (a) a clean digit, (b) a noisy near-digit, or (c) garbage?

</div>

*Answer:* **(c) garbage.** The encoder only ever mapped *real training images* to latent points. Everywhere else is a "nothing-mapped-here" hole — the decoder is still *defined* there (it's just a function) but was never trained there, so its output is unreliable.

<div class="warning">

Two failures at once: the trained points are **scattered with gaps**, and there is **no canonical distribution** telling you where to sample. To generate, the latent must be **dense** and **shaped** like a distribution we know. That is exactly what the VAE adds.

</div>

---

# Intuition · the holes, made concrete

Take two training images — a **4** and a **9**. The encoder maps them to two latent points $z_4$ and $z_9$. Now decode the **midpoint** $\tfrac12(z_4+z_9)$.

<div class="columns">
<div>

**What you hope for:** something halfway — a 4 blurring into a 9.

</div>
<div>

**What a plain AE gives:** garbage. The decoder was trained *only at* $z_4$ and $z_9$, never *between* them. The midpoint is a hole.

</div>
</div>

<div class="insight">

The latent is like cities on a map with **no roads between them.** Each training image is a city the decoder knows; the space in between was never visited, so the decoder has no idea what lives there. Generation drops a pin at random — and random pins land on empty countryside. The VAE's whole job is to **pave the roads.**

</div>

---

<!-- _class: section-divider -->

## Part 3 · The VAE — a distribution, not a point

---

# The one change · encode to a Gaussian cloud

A plain AE sends each image to a single **point**. A VAE sends it to a small **Gaussian cloud** — a mean $\mu(x)$ and a spread $\sigma(x)$, both network outputs:

<div class="math-box">

$$q_\phi(z\mid x) = \mathcal N\big(z;\ \mu_\phi(x),\ \sigma^2_\phi(x)\big)$$

</div>

One training step, end to end:

1. Read off $\mu(x),\sigma(x)$ from the encoder.
2. **Sample** $z\sim q_\phi(z\mid x)$ — a point *inside* the cloud.
3. Decode $\hat x = g_\theta(z)$.
4. Loss = reconstruction **+** a pull keeping the cloud near $\mathcal N(0,I)$.

Because we decode a *sampled* $z$, **nearby** $z$'s are also asked to decode sensibly — that is what fills the holes.

---

# AE vs VAE, side by side

![w:900px](figures/lec19/svg/ae_vs_vae.svg)

<div class="notebook">

**🎛 Interactive · VAE latent explorer** — slide the KL weight $\beta$ and watch the latent go from clumpy holes to a smooth Gaussian you can sample. *(Interactive Lab · `interactive/articles/vae-latent-explorer`)*

</div>

---

# Why a distribution packs the latent

Encoding to a *cloud* — and pulling every cloud toward the same prior $p(z)=\mathcal N(0,I)$ — does two jobs at once.

<div class="columns">
<div>

### 1 · Fills the space

Neighbouring images' clouds **overlap**, covering the region around the origin. Any $z$ you draw lands *inside* a cloud the decoder has already seen — no holes.

</div>
<div>

### 2 · Gives a sampling rule

To generate, just draw $z\sim\mathcal N(0,I)$. The prior is the **rule book** for valid $z$'s — without it you would have no idea where to sample.

</div>
</div>

<div class="keypoint">

A VAE is a plain AE **plus a regularizer that makes the latent match a known distribution** $\mathcal N(0,I)$. Everything else — the ELBO, the KL formula — is just making that regularizer principled.

</div>

---

# Intuition · a point has zero width — that's the bug

Why does encoding to a *point* doom generation, but a *cloud* saves it? A point occupies **zero volume**. Stack a million training points in the latent and they still cover **none** of the space between them — measure zero. Sample anywhere and you miss them all.

<div class="insight">

A cloud has **width**. Give each image a Gaussian blob and neighbouring blobs **overlap**, tiling the region with actual trained territory. Pulling every blob toward the *same* $\mathcal N(0,I)$ makes that territory a place you know how to sample from. **Width + a shared home = a latent you can draw from.**

</div>

<div class="notebook">

**🎛 Interactive · VAE latent explorer** — shrink $\sigma$ toward $0$ and watch the clouds collapse back to points as the gaps reopen; raise the KL weight and watch them slide onto the prior. *(Interactive Lab · `interactive/articles/vae-latent-explorer`)*

</div>

---

# The encoder *is* an approximate posterior

In Prob. ML, variational inference approximated an intractable **posterior over parameters** $p(\theta\mid D)$ with a tractable Gaussian $q_\Psi(\theta)$. The VAE is the *same move*, with two renames:

<div class="math-box">

| Variational inference (Prob. ML) | Variational autoencoder |
|:--|:--|
| posterior over params $p(\theta\mid D)$ | posterior over latent $p(z\mid x)$ |
| one global $q_\Psi(\theta)=\mathcal N(\mu,\Sigma)$ | per-input $q_\phi(z\mid x)=\mathcal N(\mu(x),\sigma^2(x))$ |
| fit $\Psi$ by optimization | a network **amortizes** $\Psi$ — one forward pass gives $(\mu,\sigma)$ |

</div>

<div class="insight">

The encoder doesn't just compress — it *predicts the parameters of an approximate posterior* over $z$. "Amortized" = one shared network instead of re-optimizing $q$ for every $x$.

</div>

---

<!-- _class: section-divider -->

## Part 4 · Reparameterization — differentiable sampling

---

# The problem · you can't backprop through a sample

Step 2 was "**sample** $z\sim\mathcal N(\mu,\sigma^2)$." But `sample_gaussian(μ, σ)` is a black box: the draw is random, so there is no functional path from $z$ back to $\mu,\sigma$.

<div class="warning">

No path → **no gradient**. The encoder produces $\mu,\sigma$, but the loss's signal cannot reach them to update them. Training is stuck.

</div>

We need to move the randomness *out of the way* so a deterministic, differentiable path survives.

---

# The trick · $z = \mu + \sigma\odot\epsilon$

![w:820px](figures/lec19/svg/reparam_trick.svg)

Any sample from $\mathcal N(\mu,\sigma^2)$ can be rewritten (Kingma & Welling, 2014):

$$z = \mu + \sigma\odot\epsilon, \qquad \epsilon\sim\mathcal N(0,I)$$

The randomness now lives in $\epsilon$, **outside** $\mu,\sigma$. The path from $z$ back to $\mu,\sigma$ is a plain add + multiply — gradients flow.

---

# Reparameterization · gradient flow

![w:900px](figures/lec19/svg/reparam_gradient_flow.svg)

<div class="insight">

**Conveyor-belt analogy.** A robot that *randomly grabs* a part gives no gradient ("your random grab was wrong" teaches nothing). Instead, load the belt randomly ($\epsilon$) and let the robot make a **deterministic** move $z=\mu+\sigma\epsilon$ — now the motion is trainable.

</div>

---

# Practice problem 1 · does the gradient really flow?

<div class="popquiz">

**Practice problem 1.** The encoder outputs $\mu=2.0,\ \sigma=0.5$. Draw $\epsilon=1.2$, so $z=\mu+\sigma\epsilon$. Suppose the loss sends back $\dfrac{\partial\mathcal L}{\partial z}=4.0$.

Compute $\dfrac{\partial\mathcal L}{\partial\mu}$ and $\dfrac{\partial\mathcal L}{\partial\sigma}$. Then say, in one line, why sampling $z\sim\mathcal N(\mu,\sigma^2)$ **directly** would give neither.

</div>

*Try it before the next slide.*

---

# Solution · practice problem 1

$z = 2.0 + 0.5\cdot 1.2 = 2.6$. The map $z=\mu+\sigma\epsilon$ is differentiable in $\mu,\sigma$:

$$\frac{\partial z}{\partial\mu}=1 \;\Rightarrow\; \frac{\partial\mathcal L}{\partial\mu}=4.0\cdot 1=\mathbf{4.0}
\qquad
\frac{\partial z}{\partial\sigma}=\epsilon=1.2 \;\Rightarrow\; \frac{\partial\mathcal L}{\partial\sigma}=4.0\cdot 1.2=\mathbf{4.8}$$

<div class="keypoint">

Sampling $z\sim\mathcal N(\mu,\sigma^2)$ **directly** has no functional edge from $z$ to $\mu,\sigma$ in the graph — autograd sees a constant, so $\partial z/\partial\mu$ and $\partial z/\partial\sigma$ don't exist. The reparameterization moves the randomness into $\epsilon$, leaving a differentiable path the optimizer can use.

</div>

---

# Worked numeric · the cloud from three draws

Same encoder output $\mu=2.0,\ \sigma=0.5$. Reparameterize $z=\mu+\sigma\epsilon$ with three noise draws:

<div class="math-box">

| $\epsilon\sim\mathcal N(0,1)$ | $z=2.0+0.5\,\epsilon$ |
|:--:|:--:|
| $-1.0$ | $1.5$ |
| $\;\;0.4$ | $2.2$ |
| $\;\;1.2$ | $2.6$ |

</div>

The three $z$'s scatter **around** $\mu=2.0$ with spread set by $\sigma=0.5$ — that is the "cloud." Across a run, a *fresh* $\epsilon$ each step forces the decoder to rebuild the image from *any* point in the cloud, not just its centre. That is literally what fills the holes: the neighbours of $\mu$ are all trained to decode sensibly.

<div class="insight">

Check the statistics: the draws average $\approx 2.1\approx\mu$ and their spread tracks $\sigma$ — reparameterization changed *how* we draw the sample, not *what* distribution it comes from. $z$ is still exactly $\mathcal N(2.0,\,0.5^2)$.

</div>

---

<!-- _class: section-divider -->

## Part 5 · The loss — reconstruction + KL

---

# The whole loss, on one slide

<div class="keypoint">

The VAE loss is just **reconstruction + KL-to-prior**. Minimize it — that is all you need to *use* a VAE.

$$\mathcal L(x) = \underbrace{\big\lVert x - g_\theta(z)\big\rVert^2}_{\text{reconstruction}} \;+\; \underbrace{D_{\text{KL}}\big(q_\phi(z\mid x)\,\Vert\,\mathcal N(0,I)\big)}_{\text{regularizer}}, \qquad z=\mu+\sigma\odot\epsilon$$

</div>

- **Reconstruction** wants each cloud sharp and distinct — push images apart so they rebuild well.
- **KL** wants every cloud centered at $0$ with spread $1$ — pack them onto the prior.

The tension between the two is what produces a latent that is *both* informative *and* samplable.

---

# The recon–KL tug-of-war

![w:900px](figures/lec19/svg/elbo_geometry.svg)

<div class="insight">

Push too hard on reconstruction → clouds drift apart, holes reappear (a plain AE in the limit). Push too hard on KL → every cloud collapses to $\mathcal N(0,I)$ and $z$ forgets which image it came from. The VAE lives at the balance point.

</div>

---

# Intuition · reconstruct well, but stay organized

The loss is two forces in one sentence: **"rebuild the image accurately, but keep the latent tidy."**

<div class="columns">
<div>

**Reconstruction = the messy genius.** Scatter each image's code wherever rebuilding is easiest — every sheet findable, the desk itself chaos.

</div>
<div>

**KL = the librarian.** Shelve every code near the origin in one standard shape $\mathcal N(0,I)$ — tidy and samplable, even at the cost of some detail.

</div>
</div>

The VAE is their **truce**: accurate enough to rebuild, organized enough that a *random* draw lands on a real code. Messy-but-accurate has holes; tidy-but-empty forgets the images.

<div class="notebook">

**🎛 Interactive · information theory by hand** — the VAE's regularizer is the *forward* KL $D_{\text{KL}}(q\Vert p)$; toggle forward vs reverse on a bimodal target to feel **mode-covering vs mode-seeking**, the exact directionality this loss commits to. *(Interactive Lab · `interactive/articles/info-theory`)*

</div>

---

# Where the loss comes from · the ELBO

We want the data likelihood $p(x)=\int p(x\mid z)\,p(z)\,dz$ — intractable (integrate over all $z$).

**Variational trick:** introduce the encoder $q_\phi(z\mid x)$ and lower-bound $\log p(x)$ with **Jensen's inequality** ($\log$ is concave):

<div class="math-box">

$$\log p(x) \;\ge\; \underbrace{\mathbb E_{q_\phi(z\mid x)}\big[\log p_\theta(x\mid z)\big]}_{\text{reconstruction (maximize)}} \;-\; \underbrace{D_{\text{KL}}\big(q_\phi(z\mid x)\,\Vert\,p(z)\big)}_{\text{KL to prior (minimize)}} \;\equiv\; \text{ELBO}$$

</div>

Maximizing the **Evidence Lower BOund** pushes up $\log p(x)$. Negate it and you have the loss from the previous slide. *(Full derivation in the appendix — if you trust the bound, skip it.)*

---

# The KL term · closed form and meaning

For a diagonal Gaussian $q=\mathcal N(\mu,\sigma^2)$ against the standard normal prior $p=\mathcal N(0,1)$, the KL has a closed form (derived in L1, and again in today's appendix):

<div class="math-box">

$$D_{\text{KL}}\big(\mathcal N(\mu,\sigma^2)\,\Vert\,\mathcal N(0,1)\big) = \tfrac12\sum_{i=1}^{d}\big(\mu_i^2 + \sigma_i^2 - 1 - \log\sigma_i^2\big)$$

</div>

Two costs, both zero exactly at the prior:

<div class="columns">
<div>

**Mean cost** $\mu^2$ — zero at $\mu=0$, grows quadratically as the cloud drifts off-center.

</div>
<div>

**Spread cost** $\sigma^2-1-\log\sigma^2$ — zero at $\sigma^2=1$; penalizes clouds that are too tight *or* too wide.

</div>
</div>

The KL keeps pushing $\mu\to0,\ \sigma\to1$ **unless reconstruction pays to keep them elsewhere.**

---

# Practice problem 2 · compute a KL

<div class="popquiz">

**Practice problem 2.** For one image the encoder gives a 1-D posterior $q(z\mid x)=\mathcal N(\mu=1.2,\ \sigma^2=0.25)$. The prior is $\mathcal N(0,1)$.

Compute $D_{\text{KL}}\big(q\,\Vert\,\mathcal N(0,1)\big)$, then say which term dominates and which way training will nudge $\mu$ and $\sigma$.

</div>

*Try it before the next slide.* *(Use $\log 0.25 = -1.386$.)*

---

# Solution · practice problem 2

$$D_{\text{KL}} = \tfrac12\big(\mu^2 + \sigma^2 - 1 - \log\sigma^2\big)
= \tfrac12\big(1.44 + 0.25 - 1 - \log 0.25\big)
= \tfrac12\big(0.69 + 1.386\big) \approx \mathbf{1.04}$$

<div class="insight">

The **mean** term $\mu^2=1.44$ dominates: the cloud sits off-center at $1.2$. The spread $\sigma^2=0.25$ is a tight (confident) encoding, which also costs a little. Training will pull $\mu$ back toward $0$ and $\sigma$ toward $1$ — *unless* reconstruction needs this image encoded far from the others. Every sample negotiates exactly this trade.

</div>

---

# Worked example · the spread cost is asymmetric

Practice problem 2's cloud was off-centre (big $\mu$). Now hold $\mu=0$ and vary only the spread, using $\tfrac12(\mu^2+\sigma^2-1-\log\sigma^2)$:

<div class="math-box">

| posterior | $\sigma^2$ | KL $=\tfrac12(\sigma^2-1-\log\sigma^2)$ |
|:--|:--:|:--:|
| too tight | $0.25$ | $\tfrac12(0.25-1+1.386)=\mathbf{0.32}$ |
| just right | $1.0$ | $\tfrac12(1-1-0)=\mathbf{0}$ |
| too wide | $4.0$ | $\tfrac12(4-1-1.386)=\mathbf{0.81}$ |

</div>

<div class="insight">

The spread cost is **zero only at $\sigma^2=1$** and climbs on *both* sides — a cloud that is too **confident** (tight) is penalized just like one that is too **vague** (wide). That is why a VAE resists collapsing a posterior to a spike: an over-tight $\sigma$ costs KL, so the encoder is nudged to keep a little honest uncertainty. Add the mean cost $\mu^2$ from PP2 and you have the full pressure. *(Uses $\log 0.25=-1.386$, $\log 4=1.386$.)*

</div>

---

# β-VAE · turning the KL knob

Scale the KL by a weight $\beta$ — the ELBO is the special case $\beta=1$:

$$\mathcal L = \text{recon} + \beta\cdot D_{\text{KL}}\big(q_\phi(z\mid x)\,\Vert\,p(z)\big)$$

![w:820px](figures/lec19/svg/vae_loss_balance.svg)

- $\beta=1$ — the exact VAE. $\;\beta>1$ — stronger packing, more disentangled, often **blurrier**. $\;\beta<1$ — favors reconstruction, weaker latent structure.

---

# Disentanglement · what β-VAE can buy

With $\beta>1$ on a faces dataset (Higgins et al., 2017), individual latent coordinates sometimes line up with single semantic factors:

<div class="math-box">

$z_1\to$ face angle · $z_2\to$ lighting · $z_3\to$ smile · $z_4\to$ hair length · …

</div>

<div class="insight">

No supervision — the structure emerges from KL pressure plus reconstruction. It enables **editable generation**: perturb one coordinate → "same face, different smile."

</div>

<div class="warning">

**Caveat.** Disentanglement is *not guaranteed* (Locatello et al., 2019) — which factors separate, and whether they separate at all, varies across runs and datasets. Treat it as a tendency, not a theorem.

</div>

---

<!-- _class: section-divider -->

## Part 6 · Sampling is generating

---

# Now it clicks · the three-line recipe

Everything we built exists so that this works:

<div class="keypoint">

1. Draw $z \sim \mathcal N(0,I)$ — the prior we trained every cloud to match.
2. Run the **decoder** once: $\hat x = g_\theta(z)$.
3. Out comes a brand-new sample the network never saw.

</div>

The encoder isn't even used at generation time — it only *trained* the decoder and *shaped* the latent. The KL term is what makes step 1 land inside trained territory.

---

# Practice problem 3 · why $\mathcal N(0,I)$, and why not a plain AE?

<div class="popquiz">

**Practice problem 3.** Two questions, one idea:

**(a)** Why do we sample $z\sim\mathcal N(0,I)$ at generation time — why *that* distribution?
**(b)** Why does the same recipe (sample $z$, decode) produce garbage for a **plain autoencoder**?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 3

<div class="columns">
<div>

**(a) Why $\mathcal N(0,I)$.** The KL term trained *every* posterior cloud $q(z\mid x)$ to sit near $\mathcal N(0,I)$. So $\mathcal N(0,I)$ is precisely the **map of where the decoder has seen data** — draw from it and you land inside a region the decoder was trained on.

</div>
<div>

**(b) Why the plain AE fails.** It has no KL, so its latent points are **scattered with holes** and there is **no canonical distribution** to sample from. A draw from $\mathcal N(0,I)$ falls into an untrained hole → garbage.

</div>
</div>

<div class="keypoint">

The KL didn't just regularize — it *manufactured the sampling distribution*. That is the entire reason a VAE generates and a plain AE cannot.

</div>

---

# Practice problem 4 · turning the β knob

<div class="popquiz">

**Practice problem 4.** You train two VAEs on the same data, identical except for the KL weight: model **A** with $\beta=0.05$, model **B** with $\beta=10$.

For each, predict (i) which has **sharper** reconstructions, (ii) which latent you can **sample** cleanly, (iii) which risks the latent **forgetting** the image. Then name the failure mode each extreme approaches.

</div>

*Try it before the next slide.*

---

# Solution · practice problem 4

<div class="columns">
<div>

**A · $\beta=0.05$ (KL barely matters)**
- (i) **Sharper** — almost all budget on reconstruction.
- (ii) **No** — clouds drift apart, holes return.
- Approaches a **plain autoencoder**: great recon, can't generate.

</div>
<div>

**B · $\beta=10$ (KL dominates)**
- (i) **Blurrier** — recon is out-voted.
- (ii) **Yes**, cleanly — clouds packed onto $\mathcal N(0,I)$.
- (iii) **Forgets** — every $q(z\mid x)\to\mathcal N(0,I)$: **posterior collapse** (appendix).

</div>
</div>

<div class="keypoint">

$\beta$ slides you along the **recon ↔ KL** trade-off: small $\beta\to$ plain-AE holes, large $\beta\to$ posterior collapse. $\beta=1$ is the exact ELBO; the useful window sits near it, nudged by what you want (fidelity vs disentanglement). *Same shape as L7's regularization strength — too little overfits, too much underfits.*

</div>

---

# Practice problem 5 · the whole loss, one number

<div class="popquiz">

**Practice problem 5.** For one training image a VAE gives reconstruction error $\lVert x-\hat x\rVert^2 = 3.0$ and a 1-D posterior $q(z\mid x)=\mathcal N(\mu=1.2,\ \sigma^2=0.25)$ — the same cloud as PP2, whose KL you found $\approx 1.04$.

Write the total loss $\mathcal L=\text{recon}+\beta\,\text{KL}$ for **$\beta=1$** and **$\beta=4$**. Which term does cranking $\beta$ amplify, and which way will the encoder then move $\mu,\sigma$?

</div>

*Try it before the next slide.* *(Reuse KL $\approx 1.04$ from PP2.)*

---

# Solution · practice problem 5

<div class="math-box">

$$\beta=1:\quad \mathcal L = 3.0 + 1{\cdot}1.04 = \mathbf{4.04}$$
$$\beta=4:\quad \mathcal L = 3.0 + 4{\cdot}1.04 = 3.0 + 4.16 = \mathbf{7.16}$$

</div>

<div class="insight">

At $\beta=4$ the KL contribution ($4.16$) now **outweighs** reconstruction ($3.0$): the gradient cares more about tidying the latent than about pixel fidelity. So the optimizer pulls $\mu$ harder toward $0$ and $\sigma$ toward $1$ — accepting a *worse* recon to shrink that inflated KL. That is how $\beta$ buys packing and disentanglement at the cost of sharpness, made numeric.

</div>

<div class="keypoint">

The VAE objective is always this one sum — **recon $+\ \beta\cdot$KL** — evaluated per image and averaged over the batch. Every design choice ($\beta$, latent width, decoder power) just moves where this sum's minimum sits.

</div>

---

# Latent-space interpolation

![w:900px](figures/lec19/svg/vae_interpolation.svg)

Because the latent is smooth and gap-free, walking a straight line from $z_a=\mu(x_a)$ to $z_b=\mu(x_b)$ and decoding each step produces **valid intermediate images** — a 4 morphing continuously into a 9. In a plain AE that path crosses holes and breaks.

---

<!-- _class: code-heavy -->

# The VAE in PyTorch · the whole thing

```python
class VAE(nn.Module):
    def __init__(self, d_in, d_z):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, 256), nn.ReLU(),
                                 nn.Linear(256, 2 * d_z))   # -> [mu | log_var]
        self.dec = nn.Sequential(nn.Linear(d_z, 256), nn.ReLU(),
                                 nn.Linear(256, d_in))

    def forward(self, x):
        mu, log_var = self.enc(x).chunk(2, dim=-1)
        std = torch.exp(0.5 * log_var)          # sigma from log-variance (stable, positive)
        z   = mu + std * torch.randn_like(std)  # reparam: z = mu + sigma * eps
        recon = self.dec(z)
        kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(-1)  # KL to N(0,I)
        return recon, kl
```

We store $\log\sigma^2$ (not $\sigma$) so it can be any real number and $\sigma=e^{\frac12\log\sigma^2}$ stays positive. The KL line is the closed form from Part 5, verbatim.

---

# Architecture in practice · a ConvVAE for MNIST

<div class="columns">
<div>

**Encoder** — Conv stack shrinks $1{\times}28{\times}28$ to $16{\times}7{\times}7$, flattens, then a linear layer emits $2d$ numbers = $(\mu,\log\sigma^2)$.

**Decoder** — a mirror: linear up to $16{\times}7{\times}7$, then `ConvTranspose2d` back to $1{\times}28{\times}28$, `Sigmoid` for pixels in $[0,1]$.

</div>
<div>

```python
def VAE_loss(x, x_hat, mu, log_var, beta=1):
    recon = F.mse_loss(x_hat, x)
    std   = torch.exp(0.5 * log_var)
    post  = Normal(mu, std)
    prior = Normal(0, 1)
    kl = kl_divergence(post, prior).mean()
    return recon + beta * kl
```

</div>
</div>

<div class="notebook">

**📓 Notebook · ConvVAE on MNIST** — the full AE → VAE → β-VAE build, latent interpolation, and sampling. *(Prob. ML · `notebooks/vae.ipynb` · ES 667 · `notebooks/19-vae-mnist.ipynb`)*

</div>

---

# VAE vs GAN vs diffusion · where it sits

| | Sample quality | Training | Likelihood | Sampling |
|:--|:--|:--|:--|:--|
| **VAE** (today) | often blurry | stable | ✓ ELBO | one pass |
| **GAN** | sharp | brittle | ✗ | one pass |
| **Diffusion** (L23) | best (as of writing) | stable | ≈ | many passes |

<div class="insight">

VAE blur is structural: MSE with overlapping posteriors makes the decoder predict the **average** of the plausible images. **Diffusion (L23)** fixes this by decoding in many small denoising steps — and Stable Diffusion runs that diffusion *inside* a VAE's compressed latent (≈48× fewer numbers). Today's ideas are load-bearing next lecture.

</div>

<div class="notebook">

**🎛 Interactive · meet the neighbours** — *GAN minimax dance* (generator vs discriminator) and *diffusion denoising* (many-step sampling), to contrast with today's one-pass VAE. *(Interactive Lab · `interactive/articles/gan-minimax-dance`, `diffusion-denoise`)*

</div>

---

# See it live

<div class="notebook">

**🎛 Interactive · VAE latent explorer** — sample new digits and slide $\beta$ to watch the latent reshape. *(Interactive Lab · `interactive/articles/vae-latent-explorer`)*

</div>

<div class="notebook">

**🎛 Interactive · ELBO decomposition** — drag the recon/KL balance and see the bound split live. *(Interactive Lab · `interactive/articles/elbo-decomposition`)*

</div>

<div class="notebook">

**🎛 Interactive · Seeing the Multivariate Normal** — the "rotate-then-stretch" picture behind $z=\mu+\sigma\odot\epsilon$. *(Interactive Lab · `interactive/articles/multivariate-normal`)*

</div>

---

<!-- _class: summary-slide -->

# One sentence to remember

<div class="keypoint">

**Encode to a distribution, not a point — sample it with the reparameterization trick, pull it toward $\mathcal N(0,I)$ with a KL term — and sampling becomes generating.**

</div>

- **Autoencoder** — encode → bottleneck → decode; great compressor (nonlinear PCA), **not** generative.
- **VAE** — encoder outputs $(\mu,\sigma)$; the latent has no holes.
- **Reparam** — $z=\mu+\sigma\odot\epsilon$ makes the sample differentiable.
- **Loss** — reconstruction $+$ $D_{\text{KL}}(q\Vert\mathcal N(0,I))$, the ELBO; $\beta$ tunes the trade-off.
- **Generate** — draw $z\sim\mathcal N(0,I)$, decode.

**Next (L23):** turn one decode into $T$ denoising steps — **diffusion**.

---

<!-- _class: compact -->

# Sources & credits

<div class="paper">

This lecture reuses and adapts material from the instructor's own courses (IIT Gandhinagar) and standard references:

- **VAE build (AE → VAE → β-VAE on MNIST), reparameterization, ELBO & Gaussian-KL derivation** — *Probabilistic Machine Learning*, N. Batra · `github.com/nipunbatra/pml-teaching` *(`notebooks/vae.ipynb`, `slides/Variational-Inference.tex`)*
- **KL divergence, Gaussian NLL, information theory** — *Machine Learning / Prob. ML (L1 primer)*, N. Batra
- **ES 667 VAE materials & interactive explainers** — `notebooks/19-vae-mnist.ipynb`, figure library; explainers *vae-latent-explorer, elbo-decomposition, multivariate-normal, info-theory, gan-minimax-dance, diffusion-denoise*
- **Original method** — D. Kingma & M. Welling, *Auto-Encoding Variational Bayes* (2014); β-VAE — Higgins et al. (2017); disentanglement caveat — Locatello et al. (2019).
- **Pedagogical framing** — A. Ng, *Deep Learning Specialization* (representation learning, in-line losses).

All source courses © N. Batra & teaching staff.

</div>

---

<!-- _class: section-divider -->

## ⭐⭐⭐ Appendix · optional depth

*Skip on a first pass — for the curious.*

---

<!-- _class: math-heavy -->

# ⭐⭐⭐ Optional · deriving the ELBO

Start from the log-evidence and insert the encoder $q_\phi(z\mid x)$:

$$\log p(x) = \log\int q_\phi(z\mid x)\,\frac{p(x,z)}{q_\phi(z\mid x)}\,dz = \log\,\mathbb E_{q_\phi}\!\Big[\tfrac{p(x,z)}{q_\phi(z\mid x)}\Big]$$

**Jensen** ($\log$ concave, so $\log\mathbb E[X]\ge\mathbb E[\log X]$) moves $\log$ inside:

$$\log p(x) \ge \mathbb E_{q_\phi}\!\Big[\log\tfrac{p(x,z)}{q_\phi(z\mid x)}\Big]$$

Factor $p(x,z)=p(x\mid z)\,p(z)$ and split the log; the $p(z)/q$ piece is a negative KL:

$$\boxed{\;\log p(x) \ge \mathbb E_{q_\phi(z\mid x)}\big[\log p_\theta(x\mid z)\big] - D_{\text{KL}}\big(q_\phi(z\mid x)\,\Vert\,p(z)\big)\;}$$

The gap between $\log p(x)$ and the ELBO is exactly $D_{\text{KL}}(q_\phi(z\mid x)\Vert p(z\mid x))\ge 0$ — tightening the bound makes $q$ a better posterior approximation. *(Prob. ML · `Variational-Inference.tex`.)*

---

<!-- _class: math-heavy -->

# ⭐⭐⭐ Optional · the Gaussian-KL closed form

For $q=\mathcal N(\mu_q,\sigma_q^2)$, $p=\mathcal N(\mu_p,\sigma_p^2)$, using $\mathbb E_q[\theta]=\mu_q$ and $\mathbb E_q[\theta^2]=\sigma_q^2+\mu_q^2$:

$$D_{\text{KL}}(q\Vert p)=\mathbb E_q\Big[\log\tfrac{q}{p}\Big] = \tfrac12\Big[\log\tfrac{\sigma_p^2}{\sigma_q^2} - 1 + \frac{\sigma_q^2+(\mu_q-\mu_p)^2}{\sigma_p^2}\Big]$$

Set the prior to standard normal, $\mu_p=0,\ \sigma_p^2=1$:

$$D_{\text{KL}}\big(\mathcal N(\mu,\sigma^2)\,\Vert\,\mathcal N(0,1)\big) = \tfrac12\big(\mu^2 + \sigma^2 - 1 - \log\sigma^2\big)$$

— exactly the per-dimension term in the VAE loss, and the `kl = -0.5*(1 + log_var - mu² - log_var.exp())` line of code. *(Full term-by-term expansion: Prob. ML · `Variational-Inference.tex`; also L1 appendix.)*

---

<!-- _class: math-heavy -->

# ⭐⭐⭐ Optional · posterior collapse

![w:820px](figures/lec19/svg/posterior_collapse.svg)

If the decoder is powerful enough to reconstruct without using $z$, the KL term wins and drives $q_\phi(z\mid x)\to\mathcal N(0,I)$ for **every** $x$ — the latent carries no information. Reconstructions look fine (the decoder ignores $z$), but samples are junk. **Fixes:** KL annealing (ramp $\beta$ from $0$), free bits (allow some KL for free), or a weaker decoder.
