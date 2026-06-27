---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# Autoencoders &amp; VAEs

## Lecture 19 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar · Aug 2026*

---

# Learning outcomes

By the end of this lecture you will be able to:

1. State what a **plain autoencoder** is and why it isn't generative.
2. Motivate the need for a **prior distribution** on latent space.
3. Write the **ELBO** from scratch using Jensen's inequality.
4. Derive the **KL term** for Gaussian posterior vs standard normal.
5. Explain and implement the **reparameterization trick**.
6. Train a **β-VAE** and discuss disentanglement.
7. Place VAEs in context · pre-compressor for latent diffusion (Stable Diffusion).

---

# Where we are · PCA was a linear autoencoder

In **ES 335** you used **PCA** to compress data · project $\mathbf{x}$ down to a few coordinates $z$ (encode), then reconstruct $\hat{\mathbf{x}}$ from them (decode) — choosing the projection to minimize $\|\mathbf{x} - \hat{\mathbf{x}}\|^2$. That **is** an autoencoder · encode → bottleneck → decode. PCA's catch · both maps are forced to be **linear**, so it can only fit a flat subspace.

<div class="insight">

So here is the generalization · **replace PCA's two linear maps with nonlinear layers**, and you get a *learned, nonlinear* compressor — an **autoencoder** — that can fold curved data a flat subspace never could. Then one more twist (encode to a *distribution*, not a point) turns it **generative** · the **VAE**. This is also where Module 9 opens · *can we sample brand-new examples that look like the data?*

</div>

<div class="paper">

**Reading & inspiration** ·
- **Kingma & Welling 2013 · *Auto-Encoding Variational Bayes***  — original paper.
- **Lilian Weng · *From Autoencoder to Beta-VAE*** (blog) — best visual walk-through.
- **Doersch · *Tutorial on VAEs***  (arXiv 2016) — clean derivation of ELBO.
- **UDL (Prince) · Ch 17**.

</div>

---

# Four questions

1. What's a plain **autoencoder** and why isn't it generative?
2. What does **VAE** add, and why does it work?
3. What is the **reparameterization trick**?
4. Where do VAEs fit in today's generative landscape?

---

# Pop quiz · why isn't a plain autoencoder generative?

A plain autoencoder squeezes images through a 32-dim bottleneck, then reconstructs.

<div class="popquiz">

Sample a random vector $\mathbf{z} \in \mathbb{R}^{32}$ from $\mathcal{N}(0, I)$ and pass it through the decoder.

(a) You'll get a clean image.
(b) You'll get a noisy *near-image*.
(c) You'll get garbage.

The answer is **(c) garbage** — the decoder only knows the small set of points that real images mapped to. The latent space has *holes*. **Fixing those holes is the entire idea of VAE.**

</div>

---

▶ **Interactives for L19** · [vae-latent-explorer](https://nipunbatra.github.io/interactive-articles/vae-latent-explorer/) (sample new digits) · [multivariate-normal](https://nipunbatra.github.io/interactive-articles/multivariate-normal/) (Gaussian intuition) · [elbo-decomposition](https://nipunbatra.github.io/interactive-articles/elbo-decomposition/) (recon vs KL trade-off live).

---

# Generative modeling · the task

<div class="keypoint">

Given $N$ i.i.d. samples $\{x_1, \ldots, x_N\}$ from an unknown distribution $p_\text{data}$, learn a model from which we can **sample** new $x \sim p_\text{model}$ such that $p_\text{model} \approx p_\text{data}$.

</div>

Two sub-tasks, often pursued together:

1. **Density estimation** · assign a probability $p_\text{model}(x)$ to any candidate sample.
2. **Generation** · draw novel samples from $p_\text{model}$.

Images live in $\mathbb{R}^{100{,}000+}$ on a low-dimensional manifold. Writing down $p_\text{data}$ analytically is hopeless; learning it from samples is the whole game.

---

# Three generative strategies

<div class="columns">
<div>

### Density-based

Write $p_\theta(x)$ explicitly; maximize $\sum_i \log p_\theta(x_i)$.

- Pixel-RNN, PixelCNN, Normalizing flows.
- Clean likelihoods.
- Autoregressive or invertible only.

</div>
<div>

### Latent-based

Hidden variable $z$ generates $x$: $x \sim p(x|z), z \sim p(z)$.

- **VAE**, GAN (implicit).
- Compact latent, rich samples.
- VAE: likelihood via ELBO. GAN: no likelihood at all.

</div>
</div>

Diffusion (L21) is a *layered* latent model with $T$ levels. The recipe of "add structure in latent space" starts here with the VAE.

---

<!-- _class: section-divider -->

### PART 1

# The generative model family tree

Four ways to learn a distribution — where will the VAE sit?

---

# Four families of generative models

| Family | How it samples | Training |
|--------|----------------|----------|
| **VAE** (L19) | sample z ~ p(z), decode | ELBO |
| **GAN** (L20) | sample z ~ p(z), generator | minimax |
| **Normalizing flows** | invertible transforms of p(z) | exact likelihood |
| **Diffusion** (L21-22) | iterative denoising from noise | score matching / denoising |

Today is VAE. Each family has tradeoffs between sample quality, training stability, and tractability.

---

<!-- _class: section-divider -->

### PART 2

# Autoencoder first

What does pure compression buy you — and what does it still miss?

---

# Autoencoder · the postcard analogy

<div class="keypoint">

A perfect forger writes the **most compact possible description** of a painting on a postcard (the latent code · maybe 16 numbers).

They mail it to their partner. The partner must **recreate the original painting** using only the postcard.

</div>

If the postcard is too small, they're forced to learn what's *truly essential* · the "essence" of the painting · not every brushstroke. That's compression. That's what an autoencoder learns.

---

# Postcard analogy · in math terms

<div class="insight">

- **Forger** writes the postcard → **encoder** $f$.
- **Postcard** = compact description → **latent code $z$**.
- **Partner** reconstructs the painting → **decoder** $g$.
- "Goodness" of reconstruction → **MSE loss** $\|x - g(f(x))\|^2$.

</div>

Train encoder + decoder together. The bottleneck $d \ll n$ forces the network to keep only the most informative features.

---

# Worked numeric · 4-pixel autoencoder

Tiny grayscale image $x = [0.9, 0.2, 0.8, 0.1]$. Latent dim $d = 1$.

1. **Encode.** Network outputs $z = 0.7$.
2. **Decode.** Network maps $z = 0.7 \to \hat x = [0.8, 0.3, 0.7, 0.2]$.
3. **Loss (MSE).**
$\mathcal{L} = \tfrac{1}{4}\bigl((0.9-0.8)^2 + (0.2-0.3)^2 + (0.8-0.7)^2 + (0.1-0.2)^2\bigr)$
$= \tfrac{1}{4}(0.01 + 0.01 + 0.01 + 0.01) = \mathbf{0.01}$

Backprop adjusts encoder + decoder weights to push this lower.

**Uses:**
- Denoising · dimensionality reduction · pretraining / feature learning.
- Beats PCA for non-linear data.

---

# Autoencoder vs PCA · what's added

PCA is **the** linear autoencoder with orthogonal weights. What does nonlinearity buy you?

<div class="math-box">

- PCA forces the latent space to be *linear subspace*. Fine for Gaussian-like data; poor for curved manifolds.
- An autoencoder (MLP or CNN) can fold arbitrary manifolds — digits on a swiss-roll latent, faces on a curved surface, etc.

</div>

Roughly · on MNIST a deep AE reaches a given reconstruction quality at noticeably fewer latent dims than PCA needs (curved manifold vs linear subspace). Exact numbers depend on architecture and training — the point is *nonlinearity buys you compression*, not a magic constant.

---

# Bottleneck intuition · why it's crucial

If the latent $d \ge n$, the network can just copy · $z = x$, $g(z) = z$. Loss is zero but nothing learned.

<div class="keypoint">

The bottleneck $d \ll n$ **forces compression** · the network must keep only the most informative features. Anything redundant gets dropped. This is why autoencoders produce useful representations even without labels.

</div>

Modern variants add noise (denoising AE) or masking (MAE, L17) *instead of* a small bottleneck — same idea, different forcing.

---

# A concrete AE · MNIST dimensionality

<div class="math-box">

Input · `28 × 28 = 784` pixels. Encode to latent `z` of size 16. Decode back to 784.

| Layer | Shape | Params |
|:-:|:-:|:-:|
| Input | 784 | — |
| Linear → ReLU | 256 | 200,960 |
| Linear → ReLU | 64 | 16,448 |
| Linear (μ only) | 16 | **bottleneck** · 1,040 |
| Linear → ReLU | 64 | 1,088 |
| Linear → ReLU | 256 | 16,640 |
| Linear → sigmoid | 784 | 201,488 |

</div>

Total · ~440k params. Illustrative reconstruction MSE on MNIST test · ~0.003 for the AE vs ~0.015 for PCA at the same 16 latent dims — a several-fold gain from nonlinearity (exact numbers vary with training).

---

# But autoencoders aren't generative

Suppose you train an AE on MNIST. To *generate* a new digit, you'd:

1. Pick a random z in latent space.
2. Decode it.

**What happens?** Usually garbage. Why?

<div class="warning">

The latent space is **irregular**. The encoder only learned to map *actual training images* to latent points. Random z values likely fall into "nothing-mapped-here" regions — the decoder is still *defined* there (it's just a function), but it was never trained on those points, so its output is unreliable garbage.

</div>

You'd need the latent space to be **dense** and **structured** — that's what VAE adds.

---

<!-- _class: section-divider -->

### PART 3

# The VAE fix

What single change makes the latent space safe to sample from?

---

# AE vs VAE

![w:920px](figures/lec19/svg/ae_vs_vae.svg)

<div class="realworld">

▶ Interactive: slide the KL weight β, watch the latent space go from clumpy to Gaussian — [vae-latent-explorer](https://nipunbatra.github.io/interactive-articles/vae-latent-explorer/).

</div>

---

# Why a prior? · two jobs it does

The prior $p(z) = \mathcal{N}(0, I)$ does two things for us:

<div class="columns">
<div>

### 1. Defines the sampling distribution

At generation time we draw $z \sim p(z)$ and decode. The prior is the *rule book* for producing valid z's.

Without a prior, you wouldn't know how to initialize z for generation.

</div>
<div>

### 2. Regularizes the posterior

The KL term pulls $q(z|x)$ toward $p(z)$ for every training example. Every encoded posterior overlaps in the same region → smooth latent.

Without this, training points occupy disjoint clusters.

</div>
</div>

<div class="keypoint">

A VAE is a plain AE **with a regularizer that makes the latent space match a known distribution**. Everything else follows from making that regularizer principled (the ELBO).

</div>

---

# VAE · the encoder outputs a distribution

The encoder no longer outputs a point $z$. It outputs **parameters of a Gaussian**:

$$q_\phi(z | x) = \mathcal{N}(z; \mu_\phi(x), \sigma_\phi^2(x))$$

Both $\mu$ and $\sigma$ are network outputs. During training:

1. Given $x$, get $\mu(x), \sigma(x)$.
2. **Sample** $z \sim q_\phi(z|x)$.
3. Decode · $\hat{x} = g_\theta(z)$.
4. Loss · reconstruction **+** KL divergence to prior.

---

# ELBO geometry

![w:920px](figures/lec19/svg/elbo_geometry.svg)

---

# The punchline · you can skip the derivation

<div class="keypoint">

The VAE loss is just **reconstruction + KL-to-prior**. Train to minimize it. That's all you need to use a VAE.

$$\mathcal{L} = \underbrace{\|x - \text{decode}(z)\|^2}_\text{reconstruction} + \underbrace{\text{KL}(q(z|x)\,\|\,\mathcal{N}(0,I))}_\text{regularizer}$$

</div>

The next two slides *derive* this from first principles (Jensen's inequality). **If you trust me, you can skip them** · come back to the math later.

---

# ELBO · the hard problem and the easy trick

**Hard problem.** We want $p(x) = \int p(x, z)\,dz$. Intractable for high-dim $z$.

**Easy trick (variational inference).**
1. Define a tractable distribution $q(z|x)$ (our encoder).
2. Derive a **lower bound** on $\log p(x)$ that we *can* compute: the **ELBO**.
3. Maximizing the ELBO pushes up $\log p(x)$ — like getting good practice-exam grades.

---

# ⭐⭐⭐ Optional · Deriving the ELBO · step by step

**Step 1.** Multiply and divide by $q(z|x)$:
$$\log p(x) = \log \int q(z|x) \cdot \frac{p(x, z)}{q(z|x)}\,dz = \log\,\mathbb{E}_{q}\!\left[\frac{p(x, z)}{q(z|x)}\right]$$

**Step 2 · Jensen's inequality.** $\log$ is concave → $\log\mathbb{E}[X] \ge \mathbb{E}[\log X]$. (For two values: $\log\!\frac{a+b}{2} \ge \frac{\log a + \log b}{2}$.) Move log inside expectation:
$$\log p(x) \ge \mathbb{E}_q\!\left[\log\tfrac{p(x, z)}{q(z|x)}\right] \quad \text{(ELBO)}$$

**Step 3 · expand.** Using $p(x, z) = p(x|z)\,p(z)$:
$$\text{ELBO} = \mathbb{E}_q[\log p(x|z)] + \mathbb{E}_q[\log p(z) - \log q(z|x)]$$

The second bracket is exactly $-D_\text{KL}(q(z|x)\,\|\,p(z))$:
$$\boxed{\log p(x) \ge \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_\text{KL}(q(z|x)\,\|\,p(z))}$$

- **First term** · reconstruction (maximize).
- **Second term** · regularizer (minimize).

---

# The ELBO · reconstruction + KL

<div class="math-box">

**Evidence Lower Bound** — a tractable lower bound on $\log p(x)$:

$$\log p(x) \geq \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{reconstruction}} \;-\; \underbrace{D_\text{KL}(q_\phi(z|x) \| p(z))}_{\text{regularization}}$$

- **First term** · data likelihood under the decoder, averaged over encoder samples.
- **Second term** · how far the posterior $q$ is from the prior $p(z) = \mathcal{N}(0, I)$.

</div>

Maximize the ELBO = minimize the negative. This is the VAE loss. Every term is tractable and backpropagatable.

---

# KL · the "cost of being different"

The KL term is the price for $q(z|x)$ deviating from the standard-normal prior. Two parts:

1. **Mean cost** $\mu^2$ · zero when $\mu = 0$, grows quadratically.
2. **Variance cost** $\sigma^2 - \log\sigma^2 - 1$ · zero at $\sigma^2 = 1$, positive everywhere else.

For Gaussian $q$ vs $\mathcal{N}(0, 1)$:
$$D_\text{KL} = \tfrac{1}{2}\sum_{i=1}^d \bigl(\mu_i^2 + \sigma_i^2 - 1 - \log\sigma_i^2\bigr)$$

Quick check on the variance term:
- $\sigma^2 = 1 \Rightarrow 1 - 1 - 0 = 0$ ✓
- $\sigma^2 = 0.5 \Rightarrow 0.5 - 1 - (-0.693) = 0.193$ (penalty)
- $\sigma^2 = 2 \Rightarrow 2 - 1 - 0.693 = 0.307$ (penalty)

The VAE pushes $\mu \to 0$ and $\sigma^2 \to 1$ unless the reconstruction term needs different values.

---

# KL · worked numeric example

Suppose $q(z|x) = \mathcal{N}(\mu = 1.2, \sigma^2 = 0.25)$ for a particular image $x$. Standard normal prior $p(z) = \mathcal{N}(0, 1)$.

<div class="math-box">

$$\text{KL} = \frac{1}{2}(\sigma^2 + \mu^2 - 1 - \log \sigma^2)$$
$$= \frac{1}{2}(0.25 + 1.44 - 1 - \log 0.25) = \frac{1}{2}(0.69 + 1.386) = 1.038$$

</div>

Plugging in: $\sigma^2 = 0.25$ means **narrow** posterior (confident encoding); $\mu = 1.2$ means **off-centre** from the prior. The KL penalty of ~1.0 will pull $\mu$ back toward 0 during training, unless the reconstruction term needs a wide-apart $\mu$ to distinguish this image from others.

This is the trade-off the VAE balances at every sample.

---

# Posterior collapse · the picture

![w:920px](figures/lec19/svg/posterior_collapse.svg)

---

# ⭐⭐⭐ Optional · Posterior collapse · what to watch for

If the decoder is too powerful, the KL term will drive $q(z|x) \to p(z) = \mathcal{N}(0, I)$ · every image encodes to the same latent, $z$ carries no information about $x$.

<div class="warning">

**Posterior collapse** · the VAE becomes an autoencoder where z is just noise. Reconstructions are fine (the decoder ignores z), but samples are junk (there's no latent structure to exploit).

</div>

**Fixes**:
- Reduce decoder capacity (smaller FFN).
- Use β-VAE with $\beta < 1$ (less KL weight).
- KL annealing · start with $\beta = 0$, ramp up over training.
- Free bits · allow some KL "for free" before penalizing.

---

<!-- _class: section-divider -->

### PART 4

# The reparameterization trick

How do you backprop through a random sample?

---

# Reparameterization in one picture

![w:920px](figures/lec19/svg/reparam_trick.svg)

---

# Reparameterization · gradient flow

![w:920px](figures/lec19/svg/reparam_gradient_flow.svg)

---

# Reparameterization · the conveyor-belt analogy

<div class="keypoint">

Imagine training a robot arm that **randomly picks** a part from a bin. You can't train the picking motion · "your random pick was wrong" gives no gradient.

Now change the system · the robot picks a **specific** part from a conveyor belt. The randomness is in *how the belt is loaded*, not in the robot's motion.

</div>

The belt-loading randomness ≡ the noise $\epsilon$. The robot's picking motion ≡ the deterministic map $z = \mu + \sigma \epsilon$. Now we *can* compute gradients through the picking motion · which is exactly what we needed to train $\mu$ and $\sigma$.

---

# Reparameterization trick · making randomness differentiable

**Problem.** We need $z \sim \mathcal{N}(\mu, \sigma^2)$. But `sample_from_gaussian(μ, σ)` is a black box — no gradient through it.

**Trick (Kingma & Welling 2013).** Any sample from $\mathcal{N}(\mu, \sigma^2)$ can be written:
$$z = \mu + \sigma \cdot \epsilon, \qquad \epsilon \sim \mathcal{N}(0, 1)$$

The randomness now sits **outside** $\mu, \sigma$. The path from $z$ back to $\mu, \sigma$ is just deterministic add + multiply → gradients flow.

---

# Worked numeric · gradients through the sample

Encoder outputs $\mu = 2.0$, $\sigma = 0.5$. Sample $\epsilon = 1.2$.
$z = 2.0 + 0.5 \cdot 1.2 = 2.6$.
Suppose $\partial \mathcal{L}/\partial z = 4.0$.

Then:
- $\partial z/\partial \mu = 1$ → $\partial \mathcal{L}/\partial \mu = 4.0 \cdot 1 = \mathbf{4.0}$
- $\partial z/\partial \sigma = \epsilon = 1.2$ → $\partial \mathcal{L}/\partial \sigma = 4.0 \cdot 1.2 = \mathbf{4.8}$

Optimizer can now update $\mu, \sigma$ as usual.

---

<!-- _class: code-heavy -->

# VAE in PyTorch · the whole thing

```python
class VAE(nn.Module):
    def __init__(self, d_in, d_z):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, 256), nn.ReLU(),
                                 nn.Linear(256, 2 * d_z))
        self.dec = nn.Sequential(nn.Linear(d_z, 256), nn.ReLU(),
                                 nn.Linear(256, d_in))

    def forward(self, x):
        h = self.enc(x)
        mu, log_var = h.chunk(2, dim=-1)

        # Reparam: z = mu + sigma · eps
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + std * eps

        recon = self.dec(z)

        # KL against N(0, I)
        kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(-1)

        return recon, kl
```

Entire VAE in 15 lines. The trick is making sure the KL goes in the loss alongside reconstruction.

---

# Training loop

```python
for x in loader:
    recon, kl = model(x)
    recon_loss = F.mse_loss(recon, x, reduction='none').sum(-1)
    loss = (recon_loss + BETA * kl).mean()
    opt.zero_grad(); loss.backward(); opt.step()
```

Tuning `BETA`:
- `BETA = 1` · standard VAE (follows the ELBO derivation).
- `BETA > 1` · β-VAE (Higgins 2017). Stronger regularization → more disentangled, often blurrier.
- `BETA < 1` · more weight to reconstruction, less to latent structure.

---

# β · the seesaw between recon and KL

![w:920px](figures/lec19/svg/vae_loss_balance.svg)

---

# Disentanglement · what β-VAE buys you

With $\beta = 4$ on a faces dataset (Higgins 2017), individual latent dimensions often *start* to line up with single semantic factors (not guaranteed — see caveat below):

<div class="math-box">

- $z_1$ · azimuth (face angle)
- $z_2$ · lighting direction
- $z_3$ · smile / frown
- $z_4$ · hairstyle length
- ...

</div>

<div class="insight">

No supervision — the structure emerges from the KL regularization plus the reconstruction pressure. Disentanglement lets you do **editable generation** · "same face with different smile" by perturbing one z coordinate.

</div>

The trade-off · stronger KL forces shared structure, but loses reconstruction detail. β = 1 recovers the exact ELBO; higher β (4–10 typical) trades reconstruction quality for more interpretable, often-disentangled factors. **Caveat** · β-VAE does not *guarantee* semantic disentanglement (Locatello et al. 2019) — which factors separate, and whether they separate at all, varies across runs and datasets.

---

# ⭐⭐⭐ Optional · Conditional VAE · putting labels into the game

If you have class labels $y$, a **Conditional VAE** extends the game:

<div class="math-box">

- Encoder: $q_\phi(z \mid x, y)$
- Decoder: $p_\theta(x \mid z, y)$
- Prior: $p(z) = \mathcal{N}(0, I)$ unchanged.

</div>

At inference · sample $z \sim \mathcal{N}(0, I)$, fix $y$ to the desired class, decode. Generate class-specific samples without retraining.

<div class="realworld">

CVAE was used for controllable generation before diffusion + CFG took over. Still shipped in some specialized systems (molecule generation, time-series imputation).

</div>

---

<!-- _class: section-divider -->

### PART 5

# Generating with a VAE

The latent space is smooth now — what does that let us do?

---

# Worked example · one VAE forward pass

Suppose · $d_x = 4$ (4-pixel input), $d_z = 2$ (2D latent). Input · $x = [1.0, 0.5, 0.2, 0.8]$.

<div class="math-box">

**Step 1 · encoder.** Suppose it outputs $\mu = [0.4, -0.3]$, $\log \sigma^2 = [-1.4, -2.0]$ → $\sigma = [0.5, 0.37]$.

**Step 2 · sample $\epsilon = [0.6, -0.2]$** (one draw from $\mathcal{N}(0, I)$).
$z = \mu + \sigma \odot \epsilon = [0.4 + 0.5 \cdot 0.6, -0.3 + 0.37 \cdot (-0.2)] = [0.70, -0.37]$

**Step 3 · decode.** Suppose decoder outputs $\hat x = [0.92, 0.45, 0.25, 0.81]$.

**Step 4 · loss.**
- Reconstruction (summed squared error, matching the training loop's `.sum(-1)`) · $\|\hat x - x\|^2 = 0.08^2 + 0.05^2 + 0.05^2 + 0.01^2 \approx 0.012$
- KL · $\frac{1}{2} \sum (\sigma^2 + \mu^2 - 1 - \log \sigma^2)$
  - $z_1$: $0.25 + 0.16 - 1 + 1.4 = 0.81$
  - $z_2$: $0.135 + 0.09 - 1 + 2.0 = 1.23$
  - sum / 2 = **1.02**
- Total loss = 0.012 + 1.02 = **1.03**

</div>

Backprop through this. Update params. Repeat.

---

# Latent-space interpolation · in pictures

![w:920px](figures/lec19/svg/vae_interpolation.svg)

---

# Sampling + interpolation

```python
# Generate new examples
with torch.no_grad():
    z = torch.randn(16, d_z)              # sample from N(0, I)
    samples = model.dec(z)                # decode into image space

# Interpolate between two inputs in latent space
z_a = encoder(x_a)[0]  # mu for image A
z_b = encoder(x_b)[0]  # mu for image B
for alpha in torch.linspace(0, 1, 10):
    z = (1 - alpha) * z_a + alpha * z_b
    morph = model.dec(z)                  # smooth transition
```

The interpolation is the magic · it produces *valid* intermediate images because the latent space is smooth.

---

# Sampling gotchas

<div class="warning">

**Truncated sampling.** Sampling $z \sim \mathcal{N}(0, I)$ occasionally gives large-$\|z\|$ points where training coverage was sparse. Truncate · resample (or scale down) any $z$ that is too far out. The right threshold is *dimension-dependent* — in $d$ dims, $\|z\|$ concentrates near $\sqrt{d}$, so set $\tau$ relative to $\sqrt{d}$ (a flat $\tau = 2.5$ only makes sense for tiny $d$). Truncated samples look cleaner but less diverse.

**Decoder stochasticity.** If decoder outputs a Gaussian $p(x|z) = \mathcal{N}(g(z), \sigma_x^2 I)$, add $\sigma_x \cdot \epsilon_x$ to the mean for a single sample. If you only decode means you get the "mode"; adding variance makes samples diverse.

**VAE blur.** The KL pulls posteriors toward a simple prior · posteriors overlap significantly. Because those posteriors overlap, one latent $z$ can come from several different $x$'s; a Gaussian/MSE decoder must emit a single image, so it predicts their **average** → blurry means. This is the fundamental VAE limitation diffusion (L21) fixes.

</div>

---

# VAE vs GAN vs Diffusion · quality ranking

| | Sample quality | Training stability | Likelihood | Sampling speed |
|--|---------------|---------------------|-----------|----------------|
| **VAE** | ✗ often blurry | ✓ stable | ✓ ELBO | ✓✓ one pass |
| **GAN** | ✓✓ sharp | ✗ brittle | ✗ no | ✓✓ one pass |
| **Diffusion** | ✓✓✓ best (as of writing) | ✓ stable | ≈ | ✗ many passes |

<div class="insight">

VAEs remain useful for **latent-space exploration** and **pre-compression** — Stable Diffusion uses a VAE to compress 512×512×3 images into a 64×64×4 latent (~48× fewer numbers) *before* running diffusion there.

</div>

---

# Common questions · FAQ

**Q. Why is VAE blurrier than GAN?**
A. VAE's loss is MSE on pixels, which is the mean of possible reconstructions. When multiple outputs are possible (e.g., any detailed face), the mean is a smoothed average of those — blurry. GANs don't average; they commit.

**Q. Can I use a perceptual loss (feature-space MSE) instead of pixel MSE?**
A. Yes — produces sharper reconstructions. Stable Diffusion's VAE (an *AutoencoderKL* with a **continuous** latent) is trained with a perceptual + patch-adversarial loss for exactly this reason. VQ-VAE (Van den Oord 2017) is a related but distinct idea: it uses *discrete* latents.

**Q. Is the posterior truly Gaussian?**
A. No — the *true* posterior is arbitrary. The Gaussian parameterization is an **approximation** (the "amortized variational" part). Normalizing flow encoders and hierarchical VAEs address this; vanilla VAE trades approximation quality for simplicity.

---

<!-- _class: summary-slide -->

# Putting it all together · the L19 master sentence

<div class="math-box">

**A VAE replaces "encode to a point" with "encode to a distribution"** · the encoder outputs $\mu(x), \sigma(x)$ · we sample $z \sim \mathcal{N}(\mu, \sigma^2)$ via the **reparameterization trick** ($z = \mu + \sigma\epsilon$, $\epsilon \sim \mathcal{N}(0, I)$) · the decoder reconstructs · we minimize the **ELBO** = reconstruction NLL + $\text{KL}(q_\phi(z\mid x) \,\Vert\, p(z))$. The KL forces the latent to look standard-Normal — so sampling new $z$ produces valid images.

</div>

| Piece | What it does | Where it came from |
|:-:|:-:|:-:|
| Encoder | $x \to (\mu, \sigma)$ | learned parameters of a Gaussian |
| Reparam | $z = \mu + \sigma \epsilon$ | new here · affine push of $\mathcal{N}(0,1)$ noise |
| Decoder | $z \to \hat x$ | NLL of pixel distribution |
| KL term | shape latent | KL from L00; Gaussian closed form derived here |

**Mostly the L00 toolkit** · NLL + Gaussians + KL · the genuinely new pieces are the *reparameterization trick* and the *latent factorization* $p(x,z)=p(x\mid z)p(z)$.

---

# Practice problems

<div class="math-box">

**P1.** Derive the ELBO from $\log p(x) = \log \int p(x, z)\,dz$ via Jensen's inequality. Identify which term is reconstruction and which is the KL.

**P2.** Compute the **closed-form KL** between $\mathcal{N}(\mu, \sigma^2)$ and $\mathcal{N}(0, 1)$. (KL itself is from L00; derive the Gaussian-vs-standard-normal case here and confirm it matches the $\tfrac12(\mu^2+\sigma^2-1-\log\sigma^2)$ used in the slides.)

**P3.** Why does a deterministic encoder **not** get the same job done? Sketch the failure (no continuity, holes in latent space).

**P4.** $\beta$-VAE multiplies the KL term by $\beta > 1$. What happens to (a) reconstruction quality, (b) disentanglement of factors, (c) sample diversity?

**P5.** For images, the decoder typically uses a **per-pixel Bernoulli (BCE)** for binary or **Gaussian (MSE)** for continuous. State why these are NLLs of the chosen distribution and tie back to L00.

**P6.** Stable Diffusion uses a VAE as a **pre-compressor** to a $4 \times 64 \times 64$ latent. Why is this faster than running diffusion in pixel space?

</div>

---

# Lecture 19 — summary

- **Autoencoder** · encode → bottleneck → decode. Great for compression; *not* generative.
- **VAE** · encoder outputs a *distribution* (μ, σ); sample; decode; ELBO loss.
- **ELBO** · reconstruction term + KL divergence to prior.
- **KL term** · closed form for Gaussian; pulls q(z|x) toward N(0, I).
- **Reparameterization trick** · z = μ + σ·ε; differentiable.
- **β-VAE** · tune the KL weight for disentanglement.
- **Role today** · pre-compressor in latent diffusion models (next week).

### Read before Lecture 20

Prince Ch 15 · GANs.

### Next lecture

**GANs** — minimax training, DCGAN, mode collapse, non-saturating loss.

<div class="notebook">

**Notebook 19** · `19-vae-mnist.ipynb` — build and train a VAE on MNIST; visualize 2D latent; interpolate digits; sample from N(0, I).

</div>

---

# The one-sentence takeaway

<div class="insight">

**Encode to a distribution, not a point — then sampling becomes generating.**

*Next · drop the encoder and let two networks fight it out — GANs.*

</div>
