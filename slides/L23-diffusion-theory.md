---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# Diffusion Models I — Theory

## Lecture 23 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar*

---

# The whole lecture, in one idea

To *create* data, first learn to *destroy* it. Slowly turn an image into pure noise (**forward** — no learning), then train **one** network to undo a single step of that noise (**reverse** — the only thing we learn).

<div class="keypoint">

**Diffusion = add noise over $T$ steps, then learn to denoise one step at a time.** The training loss collapses to a single line: the network **predicts the noise $\epsilon$** that was added, and pays plain **MSE** for guessing wrong. No adversary, no equilibrium — just regression.

</div>

You met MSE as "Gaussian maximum likelihood" in L1. Today that same loss powers Stable Diffusion.

$$\text{forward (fixed)} \rightarrow \underbrace{\text{closed form } x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\,\epsilon}_{\text{jump to any }t} \rightarrow \underbrace{\text{predict }\epsilon}_{\text{MSE loss}} \rightarrow \underbrace{\text{reverse}}_{\text{sample}}$$

---

# How today connects to the rest of the course

<div class="columns">
<div>

**Today reuses, almost verbatim:**
- **L1** — a Gaussian output → MSE loss; sum of independent Gaussians stays Gaussian (the closed form)
- **L9** — the **U-Net** (encoder–decoder + skips) is the denoiser
- **L13** — **sinusoidal positional encoding** becomes the *time* embedding
- **L5** — SGD needs only an *unbiased* gradient estimate

</div>
<div>

**Where it sits in the generative arc:**
- **L19 VAE** — sharp structure, blurry samples
- **L20 GAN** — sharp samples, unstable minimax
- **L23 (today)** — sharp *and* stable
- **L24** — the engineering that turns this into **Stable Diffusion**

</div>
</div>

<div class="insight">

This is **theory** — the forward process, the loss, the reverse step. L24 is the practice: latent space, text conditioning, fast samplers.

</div>

---

<!-- _class: section-divider -->

## Part 1 · The idea — destroy to create

---

# The picture: a drop of ink in water

Drop ink into still water. It starts concentrated, blurs, and finally tints the whole glass uniformly — pure structureless noise.

<div class="columns">
<div>

**Forward — easy, and physics does it for free**
Watching the drop spread *is* "noise corrupts the signal." A simple, fixed rule; no learning required.

</div>
<div>

**Reverse — the miracle**
Un-mixing ink back into a drop violates physics (entropy only grows). But with *examples* of clean images, a network can **learn** the reverse direction that first principles deny us.

</div>
</div>

Diffusion learns the un-diffusion that physics won't give you — from data, not from thermodynamics.

---

# The idea in one sentence

<div class="keypoint">

**Gradually turn an image into pure noise, then train a network to reverse that process one tiny step at a time.**

</div>

![w:680px](figures/lec21/svg/forward_reverse.svg)

Top row destroys; bottom row rebuilds. At sampling time we start from random $x_T\sim\mathcal N(0,I)$ and walk *left*, one learned denoising step at a time, until a fresh image appears.

---

# Why not map noise → image in one shot?

<div class="popquiz">

**Think first.** You have a million images and want to draw new ones. (a) Train one huge CNN mapping noise → image in a single pass. (b) Add a *little* noise over $T$ steps, then learn to undo **one** step at a time. Which is easier to learn?

</div>

![w:680px](figures/lec21/svg/diffusion_timeline.svg)

*Answer:* **(b).** Predicting one small denoising step is vastly easier than sculpting an image from pure noise in one leap — like sketching a line at a time instead of the whole drawing in one stroke. Breaking one hard generative problem into many easy denoising problems is *why diffusion won.*

---

# Where diffusion sits among generative models

| | VAE (L19) | GAN (L20) | Diffusion (today) |
|--|-----------|-----------|-------------------|
| Sample quality | blurry | sharp | **SOTA** |
| Training | stable | brittle minimax | **stable** |
| Sampling cost | 1 pass | 1 pass | **$T$ passes (slow)** |
| Mode coverage | strong | collapse risk | **strong** |

<div class="insight">

Diffusion buys sharpness *and* stability at the price of **slow, iterative sampling**. That one weakness — $T$ network calls per sample — is the entire agenda of L24 (latent diffusion, DDIM, few-step samplers).

</div>

---

# One comparison to keep: GAN vs diffusion

<div class="columns">
<div>

**GAN — a moving target**
Two networks in a minimax game; the generator chases the discriminator's shifting boundary. Unstable, mode-collapse-prone, hyperparameter-sensitive.

</div>
<div>

**Diffusion — a fixed target**
**One** network matches the *noise that was added* — a static, known quantity. Every $(x_0,t,\epsilon)$ triple is a clean supervised example.

</div>
</div>

<div class="keypoint">

A static target with a strong gradient everywhere is fundamentally easier to optimize than an adversary's moving one. That is the whole reason diffusion trains where GANs struggle.

</div>

---

# See it move before the algebra

<div class="notebook">

**🎛 Interactive · diffusion-denoise** — slide $t$ and watch a 2D spiral dissolve into noise; press "Reverse animate" to watch it reassemble step by step. Build the forward/reverse intuition in your hands first. *(Interactive Lab · `interactive/src/articles/diffusion-denoise` · [nipunbatra.github.io/interactive-articles/diffusion-denoise](https://nipunbatra.github.io/interactive-articles/diffusion-denoise/))*

</div>

<div class="notebook">

**🎛 Demo · diffusion-interactive** — a runnable app that noises and denoises real 2D datasets end to end, exposing the schedule and the number of steps. *(`~/git/diffusion-interactive` · `app.py`)*

</div>

---

<!-- _class: section-divider -->

## Part 2 · The forward process

---

# One forward step = fade a bit, add faint static

Take $x_{t-1}$, shrink it slightly toward zero, then sprinkle a little Gaussian noise on top:

<div class="math-box">

$$q(x_t \mid x_{t-1}) = \mathcal N\!\big(x_t;\ \sqrt{1-\beta_t}\,x_{t-1},\ \beta_t I\big)
\quad\Longleftrightarrow\quad
x_t = \sqrt{1-\beta_t}\,x_{t-1} + \sqrt{\beta_t}\,\epsilon,\ \ \epsilon\sim\mathcal N(0,I)$$

</div>

The single knob $\beta_t \in (0,1)$ controls **both**: how much the signal fades ($\sqrt{1-\beta_t}$) and how much static comes in ($\sqrt{\beta_t}$). The process is **fixed, Markov, and Gaussian** — nothing here is learned.

---

# One forward step · worked numeric

Let $x_{t-1}=100$ and $\beta_t=0.01$.

- **Fade:** $\sqrt{1-0.01}=\sqrt{0.99}\approx 0.995$, so the mean is $99.5$.
- **Static:** standard deviation $\sqrt{0.01}=0.1$.
- **Update:** $x_t = 0.995\cdot 100 + 0.1\,\epsilon = 99.5 + 0.1\,\epsilon$.

One step barely disturbs the signal. But chain $T=1000$ of them and the image washes out completely into $\mathcal N(0,I)$. The *accumulated* effect — not any single step — turns signal into noise.

---

# Why shrink *and* add noise?

You might ask: why not just keep adding noise? Because then the variance would grow without bound and $x_T$ would look nothing like the tidy $\mathcal N(0,I)$ we want to sample from.

<div class="math-box">

**Variance check.** Suppose $\text{Var}(x_{t-1})=1$. For $x_t=\sqrt{1-\beta_t}\,x_{t-1}+\sqrt{\beta_t}\,\epsilon$ with independent unit-variance $\epsilon$:
$$\text{Var}(x_t) = (1-\beta_t)\cdot 1 + \beta_t \cdot 1 = 1.$$

</div>

The shrink exactly cancels the added variance. Normalize the data to unit variance, and the chain **preserves** it all the way to $x_T$. This is the **variance-preserving** schedule.

---

# The closed form · jump to any $t$ in one shot

Write $\alpha_t \equiv 1-\beta_t$ (per-step *signal retention*) and $\bar\alpha_t \equiv \prod_{s=1}^{t}\alpha_s$ (retention after $t$ steps). Because a sum of independent Gaussians is Gaussian, the whole $t$-step chain collapses to a single draw:

<div class="math-box">

$$\boxed{\,x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon, \qquad \epsilon\sim\mathcal N(0,I)\,}$$

</div>

No iteration. Given $x_0$, a schedule value $\bar\alpha_t$, and one sampled $\epsilon$, you get $x_t$ directly. Note $\sqrt{\bar\alpha_t}$ is the signal fraction and $\sqrt{1-\bar\alpha_t}$ the noise fraction — **their variances sum to 1**, exactly as the variance-preserving design demands.

---

# Signal vs noise, in one chart

![w:560px](figures/lec21/svg/signal_vs_noise.svg)

As $t$ grows, $\sqrt{\bar\alpha_t}$ (signal kept) slides to $0$ and $\sqrt{1-\bar\alpha_t}$ (noise) rises to $1$. Early $t$: almost all signal. Late $t$: almost all noise. The interesting learning happens where the two curves cross.

---

# Signal-to-noise ratio · one number for "how corrupted"

Instead of tracking two curves, collapse them into a single **signal-to-noise ratio** — signal variance over noise variance:

<div class="math-box">

$$\text{SNR}(t) = \frac{\text{signal variance}}{\text{noise variance}} = \frac{\bar\alpha_t}{1-\bar\alpha_t}$$

</div>

With unit-variance data, $x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon$ carries signal power $\bar\alpha_t$ and noise power $1-\bar\alpha_t$. Read it straight off the linear-schedule table:

| $t$ | $\bar\alpha_t$ | $\text{SNR}=\bar\alpha_t/(1-\bar\alpha_t)$ | in dB |
|:-:|:-:|:-:|:-:|
| 250 | 0.52 | $1.08$ | $\approx 0$ dB |
| 500 | 0.08 | $0.087$ | $\approx -11$ dB |
| 750 | 0.003 | $0.003$ | $\approx -25$ dB |

<div class="insight">

$\text{SNR}=1$ (0 dB) is the **crossover**: signal power exactly equals noise power. That is where denoising is hardest *and* most informative — precisely the range the cosine schedule lingers in. Early $t$ is trivially high-SNR, late $t$ trivially low; the model earns its keep in the middle.

</div>

---

# Closed form · worked numeric · straight to $x_{500}$

Take $x_0 = 2.0$; the linear DDPM schedule gives $\bar\alpha_{500}\approx 0.08$.

- Signal scale: $\sqrt{0.08}\approx 0.283$.
- Noise scale: $\sqrt{0.92}\approx 0.959$.
- Sample $\epsilon = -0.8$.

$$x_{500} = 0.283\cdot 2.0 + 0.959\cdot(-0.8) = 0.566 - 0.767 = \mathbf{-0.20}$$

The original signal of $2.0$ has all but vanished — $x_{500}$ is mostly noise. And we reached it in **one** line, never simulating steps 1 through 499.

---

# Why the closed form matters · training speed

<div class="columns">
<div>

**Naive (iterate the chain)**
To get $x_{500}$ you apply 500 Gaussian steps in sequence — 500 passes per example. A batch of 128 over 100k images is ~$10^9$ operations *just to make the targets*. Days.

</div>
<div>

**Closed form (one draw)**
One sample of $\epsilon$, one scaled add — done. Any $t$, any example, in microseconds. **~500× faster** per example. Hours instead of days.

</div>
</div>

<div class="keypoint">

Being able to jump straight to any $t$ is the single biggest practical reason DDPM training is cheap. Without it the loss below would cost 500× more.

</div>

---

# Practice problem 1

<div class="popquiz">

**Practice problem 1.** A data point $x_0 = [\,3.0,\ -1.0\,]$. At timestep $t$ the schedule gives $\bar\alpha_t = 0.36$. You sample noise $\epsilon = [\,0.5,\ 1.0\,]$. Compute the noisy sample $x_t$ using the closed form.

</div>

*Try it before the next slide — you only need $\sqrt{0.36}$ and $\sqrt{0.64}$.*

---

# Solution · practice problem 1

Signal scale $\sqrt{\bar\alpha_t}=\sqrt{0.36}=0.6$; noise scale $\sqrt{1-\bar\alpha_t}=\sqrt{0.64}=0.8$.

$$x_t = 0.6\cdot[\,3.0,\,-1.0\,] + 0.8\cdot[\,0.5,\,1.0\,] = [\,1.8,\,-0.6\,] + [\,0.4,\,0.8\,] = \mathbf{[\,2.2,\ 0.2\,]}$$

<div class="keypoint">

That is *all* the forward process ever does — one scaled add. Each coordinate is handled independently because the noise is isotropic. This exact operation is line 1 of the training loop.

</div>

---

# The noise schedule · one design choice

![w:680px](figures/lec21/svg/alpha_schedule.svg)

**Linear** (original DDPM): $\beta_t$ grows linearly from $10^{-4}$ to $0.02$ over $T=1000$. **Cosine** (Nichol & Dhariwal 2021): $\bar\alpha_t$ follows a cosine curve — noise added gently early, faster late.

---

# Noise schedule · the numbers

| $t$ | linear $\bar\alpha_t$ | cosine $\bar\alpha_t$ | what's left |
|:-:|:-:|:-:|:-:|
| 0 | 1.00 | 1.00 | clean signal |
| 250 | 0.52 | 0.85 | linear: 48% gone · cosine: 15% |
| 500 | 0.08 | 0.49 | linear: already mostly noise |
| 750 | 0.003 | 0.14 | linear: essentially pure noise |
| 1000 | 0.00 | 0.00 | both $\approx \mathcal N(0,I)$ |

<div class="insight">

The linear schedule burns steps near $T$ where everything is already noise. Cosine keeps the *middle* range — where the model actually learns — useful for longer. That is why cosine is the modern default.

</div>

---

<!-- _class: section-divider -->

## Part 3 · The training objective

---

# The reverse process is Gaussian too

We want $p_\theta(x_{t-1}\mid x_t)$ — the learned denoiser. The key fact: for small $\beta_t$ the true reverse step is *also* well-approximated by a Gaussian, so we parameterize

$$p_\theta(x_{t-1}\mid x_t) = \mathcal N\big(x_{t-1};\ \mu_\theta(x_t,t),\ \Sigma_t\big).$$

<div class="insight">

We fix the variance $\Sigma_t$ to the schedule and only learn the **mean**. And rather than predict the mean directly, we make the network predict the **noise $\epsilon$** — a reparameterization that, as Ho et al. showed, trains far better.

</div>

---

# Predict $\epsilon$ — three equivalent targets, one clear winner

From $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\,\epsilon$, three things determine one another once $(x_t,t)$ are known:

<div class="math-box">

- predict $x_0$ — the clean signal directly
- predict $\mu_\theta$ — the reverse-step mean
- **predict $\epsilon$** — the noise that was added

</div>

Ho et al. (2020) found **$\epsilon$-prediction gives the best samples.** Intuition: $\epsilon$ is always unit-variance and dimension-independent, so the network never has to learn the *scale* of the signal — only its direction of corruption.

---

# Training · the noise-guessing game

<div class="keypoint">

Take a clean image. Add a **known** amount of noise $\epsilon$. Show the network only the noisy result and ask: *"what noise did I just add?"* The better it guesses $\epsilon$, the better it denoises — because subtracting the predicted noise moves $x_t$ back toward a clean image.

</div>

Every training example is a fresh supervised pair: input $(x_t,t)$, target $\epsilon$. The label is *exactly* the noise we chose, so there is never any ambiguity about the right answer.

---

# The DDPM loss · the whole payoff, in one line

<div class="math-box">

$$\mathcal L_{\text{DDPM}} = \mathbb E_{\,t,\ x_0,\ \epsilon}\Big[\ \big\lVert\, \epsilon - \epsilon_\theta\big(\underbrace{\sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon}_{x_t},\ t\big)\big\rVert^2\ \Big]$$

</div>

Read it as a recipe: **(1)** sample a clean $x_0$; **(2)** sample a timestep $t$; **(3)** sample noise $\epsilon$; **(4)** form $x_t$ by the closed form; **(5)** predict $\hat\epsilon=\epsilon_\theta(x_t,t)$; **(6)** take MSE. That is the entire objective — simpler than a GAN's minimax or a VAE's ELBO.

---

# The training loop, in one picture

![w:560px](figures/lec21/svg/training_loop.svg)

Clean image → sample $t$ and $\epsilon$ → closed-form $x_t$ → network predicts $\hat\epsilon$ → MSE against the true $\epsilon$ → backprop. One example, one $t$, one gradient step. No second network anywhere.

---

# Worked example · one full training step

<div class="math-box">

Data point $x_0 = [\,2.0,\ 1.0\,]$, timestep $t=500$ with $\bar\alpha_{500}=0.5$, sampled noise $\epsilon=[\,0.3,\ -0.2\,]$.

**1. Form $x_t$:** $\ x_{500}=\sqrt{0.5}\,[\,2.0,1.0\,] + \sqrt{0.5}\,[\,0.3,-0.2\,] = [\,1.414,0.707\,]+[\,0.212,-0.141\,]=[\,1.626,\ 0.566\,]$

**2. Network predicts:** $\ \hat\epsilon = [\,0.25,\ -0.15\,]$

**3. Loss:** $\ \lVert\hat\epsilon-\epsilon\rVert^2 = (0.25-0.3)^2 + (-0.15+0.2)^2 = 0.0025+0.0025 = \mathbf{0.005}$

**4.** Backprop through $\hat\epsilon$; update the weights.

</div>

Sum this over a batch and you have DDPM. No adversary, no cross-entropy, no second model.

---

<!-- _class: code-heavy -->

# The loss in PyTorch · essentially the math, typed out

```python
def ddpm_loss(model, x0, alpha_bar, T=1000):
    B = x0.size(0)
    t     = torch.randint(0, T, (B,), device=x0.device)   # one random t per example
    eps   = torch.randn_like(x0)                           # the target noise
    ab    = alpha_bar[t].view(B, 1, 1, 1)                  # alpha-bar_t per sample

    x_t   = ab.sqrt() * x0 + (1 - ab).sqrt() * eps         # closed-form forward (1 line!)
    eps_hat = model(x_t, t)                                # U-Net predicts the noise
    return F.mse_loss(eps_hat, eps)                        # the entire objective
```

The forward chain is one line because of the closed form; the loss is one line because it is plain MSE. The network `model` is a **U-Net** with the timestep `t` injected into every block (Part 5).

---

# Practice problem 2

<div class="popquiz">

**Practice problem 2.** We said predicting the noise $\epsilon$ is *equivalent* to predicting the clean image $x_0$. Prove it: starting from $x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon$, show that once the network knows $(x_t,t)$, an estimate of $\epsilon$ **determines** an estimate of $x_0$, and vice versa.

</div>

*Try it before the next slide — just solve the forward equation for $x_0$.*

---

# Solution · practice problem 2

Solve the closed form for $x_0$:

$$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon
\quad\Longrightarrow\quad
\hat x_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\hat\epsilon}{\sqrt{\bar\alpha_t}}.$$

<div class="keypoint">

Given $(x_t,t)$ — both known to the network — $\hat x_0$ and $\hat\epsilon$ are related by a fixed **affine** map. Predicting one *is* predicting the other, and MSE on $\epsilon$ is just a $t$-dependent rescaling of MSE on $x_0$. Ho et al. chose $\epsilon$ because its unit variance keeps that rescaling well-behaved across all $t$.

</div>

---

# $\epsilon \to \hat x_0$ · denoise in one shot (worked numeric)

Practice problem 2 said predicting $\epsilon$ *is* predicting $x_0$. Here it is with numbers — reuse the forward example $x_0=[\,3.0,-1.0\,]$, $\bar\alpha_t=0.36$, which gave $x_t=[\,2.2,\ 0.2\,]$ (so $\sqrt{\bar\alpha_t}=0.6$, $\sqrt{1-\bar\alpha_t}=0.8$).

<div class="math-box">

$$\hat x_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\hat\epsilon}{\sqrt{\bar\alpha_t}}$$

**Perfect guess** $\hat\epsilon=[\,0.5,\,1.0\,]$ (the true $\epsilon$): $\ \hat x_0 = \dfrac{[\,2.2,0.2\,]-0.8[\,0.5,1.0\,]}{0.6}=\dfrac{[\,1.8,-0.6\,]}{0.6}=[\,3.0,\,-1.0\,]$ — exact.

**Slightly off** $\hat\epsilon=[\,0.45,\,1.10\,]$: $\ \hat x_0 = \dfrac{[\,2.2,0.2\,]-[\,0.36,0.88\,]}{0.6}=\dfrac{[\,1.84,-0.68\,]}{0.6}=[\,3.07,\,-1.13\,]$.

</div>

<div class="keypoint">

A good noise guess *is* a good clean-image estimate — the error in $\hat\epsilon$ passes through a fixed $\sqrt{1-\bar\alpha_t}/\sqrt{\bar\alpha_t}$ gain to the error in $\hat x_0$. "Predict the noise" and "denoise the image" are the same network, read two ways.

</div>

---

# Practice problem 3

<div class="popquiz">

**Practice problem 3.** The loss $\mathcal L_{\text{DDPM}}$ is an expectation over *all* timesteps $t\in\{1,\dots,T\}$. Yet in code we sample **one** random $t$ per example (see `torch.randint` above) instead of summing over the whole chain. Why is that legitimate?

</div>

*Try it before the next slide — think about what SGD actually requires of a gradient.*

---

# Solution · practice problem 3

Write the objective as an expectation and split off $t$:

$$\mathcal L = \mathbb E_{x_0,\epsilon}\,\mathbb E_{t\sim\text{Unif}\{1,\dots,T\}}\big[\,\ell(x_0,t,\epsilon)\,\big].$$

A single $t$ drawn uniformly gives an **unbiased Monte-Carlo estimate** of the inner expectation, so its gradient is an unbiased estimate of $\nabla\mathcal L$.

<div class="keypoint">

SGD (L5) needs only *unbiased* gradients, not exact ones. The closed form lets us hit any $t$ in one shot, so "pick a random $t$" replaces "simulate the whole chain" — turning a $T$-fold cost into a single cheap sample per example.

</div>

---

# See it end to end in code

<div class="notebook">

**📓 Notebook · `21-ddpm-2d.ipynb`** — implement the full DDPM loop on a 2D toy dataset (Swiss roll): the closed-form forward noising, the $\epsilon$-prediction MSE, and the reverse sampler, with animations of the chain running both directions. *(ES 667 · `notebooks/21-ddpm-2d.ipynb`)*

</div>

<div class="insight">

Everything on the last five slides is ~30 lines of PyTorch. The theory really is this small; the rest of the field is *scaling* and *speeding up* this loop.

</div>

---

# Practice problem 4

<div class="popquiz">

**Practice problem 4.** At some timestep the schedule gives $\bar\alpha_t = 0.2$. Compute the signal-to-noise ratio $\text{SNR}=\bar\alpha_t/(1-\bar\alpha_t)$ (and in dB). Is $x_t$ mostly signal or mostly noise? And at what value of $\bar\alpha_t$ is $\text{SNR}$ exactly $1$?

</div>

*Try it before the next slide — recall the SNR picture from Part 2.*

---

# Solution · practice problem 4

$$\text{SNR}=\frac{\bar\alpha_t}{1-\bar\alpha_t}=\frac{0.2}{0.8}=0.25,\qquad 10\log_{10}(0.25)\approx \mathbf{-6\ \text{dB}}.$$

Noise variance is **4×** the signal variance — $x_t$ is mostly noise. And $\text{SNR}=1 \iff \bar\alpha_t = 1-\bar\alpha_t \iff \bar\alpha_t=\tfrac12$.

<div class="keypoint">

$\bar\alpha_t=0.5$ is the **halfway point of corruption** — signal power equals noise power. Modern schedules are often designed directly in SNR space (drop log-SNR by a fixed amount per step) rather than in $\beta_t$, and the loss can even be reweighted by SNR to balance easy and hard timesteps.

</div>

---

<!-- _class: section-divider -->

## Part 4 · Sampling — running the chain backward

---

# The reverse step · denoise, then re-noise a little

To go from $x_t$ to $x_{t-1}$: predict the noise, subtract a scaled slice of it to get a cleaner mean, then add a touch of fresh noise.

<div class="math-box">

$$x_{t-1} = \underbrace{\frac{1}{\sqrt{\alpha_t}}\Big(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t,t)\Big)}_{\text{denoised mean}} \;+\; \underbrace{\sigma_t\,z}_{\text{small fresh noise}},\qquad z\sim\mathcal N(0,I)$$

</div>

Why re-noise at all? Without the $\sigma_t z$ term every chain from the same $x_T$ collapses to one deterministic path — the fresh noise is what keeps sampling **stochastic**, so you get a *variety* of images.

---

# Reverse sampling, in one picture

![w:600px](figures/lec21/svg/reverse_sampling.svg)

Start at $x_T\sim\mathcal N(0,I)$ and apply the reverse step $T$ times. Each step is a small correction; chained together they carry pure noise back to a sample from the data distribution.

---

# Reverse step · worked numeric

**1D example.** $x_{100}=1.5$; schedule $\alpha_{100}=0.99$, $\bar\alpha_{100}=0.8$; network predicts $\epsilon_\theta=0.6$.

- $1/\sqrt{0.99}\approx 1.005$
- $(1-0.99)/\sqrt{1-0.8} = 0.01/\sqrt{0.2}\approx 0.0224$
- Mean $= 1.005\,(1.5 - 0.0224\cdot 0.6) = 1.005\cdot 1.4866 \approx \mathbf{1.494}$

Add fresh noise: $\sigma_{100}=0.1$, $z=-0.3 \Rightarrow$ noise term $=-0.03$, so $x_{99}=1.494-0.03=\mathbf{1.464}$.

One small step from noisier ($1.5$) to slightly cleaner ($1.464$). At the final step $t=1$ we drop the noise term and output the mean.

---

# Practice problem 5

<div class="popquiz">

**Practice problem 5.** One reverse step in 1D. You have $x_t = 2.0$ with $\alpha_t = 0.8$ and $\bar\alpha_t = 0.36$, and the network predicts $\epsilon_\theta = 1.0$. Compute the **denoised mean** (ignore the fresh-noise term $\sigma_t z$). Did the step move $x_t$ toward zero or away?

</div>

*Try it before the next slide — plug into $\frac{1}{\sqrt{\alpha_t}}\big(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta\big)$.*

---

# Solution · practice problem 5

$$\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}=\frac{0.2}{\sqrt{0.64}}=\frac{0.2}{0.8}=0.25,\qquad x_t-0.25\cdot 1.0 = 1.75.$$

$$\mu = \frac{1}{\sqrt{0.8}}\times 1.75 = 1.118\times 1.75 = \mathbf{1.96}.$$

<div class="keypoint">

We predicted positive noise, so we **subtracted** a slice of it ($2.0\to1.75$); the $1/\sqrt{\alpha_t}$ factor then rescales back toward unit variance. One step barely moves $x_t$ — cleaning is gentle and *iterative*, which is exactly why sampling needs $T$ of these (the cost L24 attacks).

</div>

---

<!-- _class: section-divider -->

## Part 5 · The backbone — U-Net + time conditioning

---

# The denoiser is a U-Net conditioned on $t$

![w:680px](figures/lec21/svg/unet_time_conditioning.svg)

$\epsilon_\theta(x_t,t)$ takes a noisy image *and* a timestep and outputs a noise image of the same shape. Same input/output structure as segmentation (L9) — so we reuse that architecture.

---

# Why a U-Net, and why one network for all $t$

<div class="columns">
<div>

**Image-in, image-out**
The target $\epsilon$ has the same shape as $x_t$. The U-Net's encoder downsamples for context, the decoder upsamples back to full resolution, and **skip connections** (L9) carry fine spatial detail across — exactly what noise prediction needs.

</div>
<div>

**One network, every timestep**
$t$ is just another input, fed in through a **time embedding**. The *same weights* denoise a barely-noised $x_1$ and an almost-pure-noise $x_{999}$ — the model reads $t$ to know how aggressively to denoise.

</div>
</div>

<div class="insight">

Attention layers at the low-resolution stages let distant regions coordinate (global structure); convolutions handle local texture. This is the ancestor of Stable Diffusion's backbone.

</div>

---

# Time conditioning · a positional encoding, reused

The timestep $t$ is an integer. We turn it into a dense vector with the **same sinusoidal encoding** you saw for token positions in L13, then inject it into every block.

```python
def timestep_embedding(t, d):          # identical idea to Transformer positions (L13)
    half  = d // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half) / half)
    args  = t[:, None].float() * freqs[None, :]
    return torch.cat([args.cos(), args.sin()], dim=-1)   # -> add into each U-Net block
```

<div class="keypoint">

Same trick, new meaning: in L13 it encoded *where a token sits in a sentence*; here it encodes *how far along the noising chain we are*. A multi-scale basis the network can read at any resolution of time.

</div>

---

# Explore the backbone yourself

<div class="notebook">

**🎛 Interactive · U-Net** — walk an image down the encoder and back up the decoder, and toggle the **skip connections** to see the fine detail they rescue. The same shape-preserving architecture (L9) that segments images is the one that predicts $\epsilon$. *(Interactive Lab · `interactive/src/articles/unet` · [nipunbatra.github.io/interactive-articles/unet](https://nipunbatra.github.io/interactive-articles/unet/))*

</div>

<div class="notebook">

**🎛 Interactive · positional encoding** — drag the position and watch the sinusoidal basis light up. In L13 the index was a token's position; here it is the timestep $t$ fed to every U-Net block. Same encoding, new meaning. *(Interactive Lab · `interactive/src/articles/positional-encoding` · [nipunbatra.github.io/interactive-articles/positional-encoding](https://nipunbatra.github.io/interactive-articles/positional-encoding/))*

</div>

---

<!-- _class: section-divider -->

## Part 6 · One lens more — noise prediction ≈ score

---

# Predicting the noise *is* estimating the score

The **score** is the gradient of the log-density, $\nabla_{x}\log p(x)$ — an arrow at every point pointing "uphill," toward regions of higher data density.

<div class="columns">
<div>

![w:340px](figures/lec21/svg/score_field.svg)

</div>
<div>

For the Gaussian $q(x_t\mid x_0)$ the score works out to $-\epsilon/\sqrt{1-\bar\alpha_t}$ (Appendix). So a network trained to predict $\epsilon$ is, up to a known scale, a **score model**:
$$s_\theta(x_t,t) = -\frac{\epsilon_\theta(x_t,t)}{\sqrt{1-\bar\alpha_t}}.$$

</div>
</div>

<div class="insight">

**DDPM (Ho 2020)** and **score-based SDEs (Song & Ermon 2019)** are two lenses on the *same* model — "guess the noise" and "follow the uphill arrows" give the same equations. Sampling then looks just like Langevin dynamics: step along the score, add a little jitter.

</div>

---

# Noise → score · worked numeric

The conversion $s_\theta(x_t,t) = -\,\epsilon_\theta(x_t,t)\,/\,\sqrt{1-\bar\alpha_t}$ is just a scaled sign-flip. Put numbers on it: $\bar\alpha_t=0.36$ so $\sqrt{1-\bar\alpha_t}=0.8$, and the network predicts $\epsilon_\theta=[\,0.4,\,-0.2\,]$.

<div class="math-box">

$$s_\theta = -\frac{[\,0.4,\,-0.2\,]}{0.8} = [\,-0.5,\ 0.25\,]$$

</div>

The score points **opposite** the predicted noise: where the model sees positive noise, "uphill toward the data" is negative — i.e. *subtract the noise*, exactly the reverse step. The scale $1/\sqrt{1-\bar\alpha_t}$ grows at small $t$ (little noise, steep density) and shrinks at large $t$.

<div class="insight">

So a Langevin sampler that walks along $+s_\theta$ and a DDPM reverse step that subtracts $\epsilon_\theta$ are **doing the same move**. "Guess the noise" and "follow the score uphill" are one algorithm in two dialects.

</div>

---

<!-- _class: summary-slide -->

# One sentence to remember

<div class="keypoint">

**Noise the data over $T$ steps (fixed, closed-form); train one U-Net to predict the added noise $\epsilon$ with plain MSE; sample by reversing the chain one step at a time.**

</div>

- **Forward** — $x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon$: jump to any $t$ in one shot (sum of Gaussians, L1).
- **Loss** — $\mathbb E\,\lVert\epsilon-\epsilon_\theta(x_t,t)\rVert^2$: the Gaussian NLL / MSE from L1, and predicting $\epsilon$ is the same as predicting $x_0$.
- **Reverse** — subtract predicted noise, add a little fresh noise, repeat.
- **Backbone** — U-Net (L9) + sinusoidal time embedding (L13); the same lens as score matching.

**Next (L24):** the *practice* — do all of this in a compressed **latent** space, steer it with **text**, and cut $T$ from 1000 to ~20. That is **Stable Diffusion**.

---

<!-- _class: compact -->

# Sources & credits

<div class="paper">

This lecture reuses and adapts material from the instructor's own ES 667 diffusion notes and the standard references:

- **Diffusion theory, figures, worked numerics** — *ES 667 Deep Learning* diffusion materials, N. Batra · `github.com/nipunbatra/dl-teaching`
- **DDPM, $\epsilon$-prediction, the training loss** — J. Ho, A. Jain, P. Abbeel, *Denoising Diffusion Probabilistic Models* (NeurIPS 2020)
- **Score matching / Langevin view** — Y. Song & S. Ermon, *Generative Modeling by Estimating Gradients of the Data Distribution* (NeurIPS 2019)
- **The original idea** — J. Sohl-Dickstein et al., *Deep Unsupervised Learning using Nonequilibrium Thermodynamics* (ICML 2015); cosine schedule — Nichol & Dhariwal 2021
- **Pedagogical framing** — A. Ng, *Deep Learning Specialization* (build intuition first, math second); canonical walkthroughs by L. Weng and C. Luo

Interactives: `interactive/src/articles/diffusion-denoise`, `unet`, `positional-encoding` · `~/git/diffusion-interactive` · notebook `notebooks/21-ddpm-2d.ipynb`. Figures from the ES 667 figure library. All source courses © N. Batra & teaching staff.

</div>

---

<!-- _class: section-divider -->

## ⭐⭐⭐ Appendix · optional depth

*Skip on a first pass — for the curious.*

---

<!-- _class: math-heavy -->

# ⭐⭐⭐ Optional · the closed form in three lines

Start from one step and unroll, merging Gaussians as we go:

$$x_t = \sqrt{\alpha_t}\,x_{t-1} + \sqrt{1-\alpha_t}\,\epsilon_t
= \sqrt{\alpha_t}\big(\sqrt{\alpha_{t-1}}\,x_{t-2} + \sqrt{1-\alpha_{t-1}}\,\epsilon_{t-1}\big) + \sqrt{1-\alpha_t}\,\epsilon_t.$$

The two independent Gaussian terms combine (variances add): $\sqrt{\alpha_t(1-\alpha_{t-1})} + \sqrt{1-\alpha_t}$ in quadrature $= \sqrt{1-\alpha_t\alpha_{t-1}}$. Iterating all the way to $x_0$:

$$\boxed{\,x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\bar\epsilon,\quad \bar\alpha_t=\prod_{s=1}^t\alpha_s,\ \ \bar\epsilon\sim\mathcal N(0,I)\,}$$

The whole $t$-step chain of noises pools into one $\bar\epsilon$ — a direct consequence of *Gaussians being closed under convolution* (L1). This is the magic that makes training cheap.

---

<!-- _class: math-heavy -->

# ⭐⭐⭐ Optional · noise prediction = score estimation

The conditional forward density is Gaussian, $q(x_t\mid x_0)=\mathcal N\!\big(\sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t)I\big)$, so up to a constant

$$\log q(x_t\mid x_0) = -\frac{1}{2(1-\bar\alpha_t)}\big\lVert x_t-\sqrt{\bar\alpha_t}\,x_0\big\rVert^2 + C.$$

Differentiate with respect to $x_t$, then substitute $\epsilon=(x_t-\sqrt{\bar\alpha_t}\,x_0)/\sqrt{1-\bar\alpha_t}$ from the forward equation:

$$\nabla_{x_t}\log q(x_t\mid x_0) = -\frac{x_t-\sqrt{\bar\alpha_t}\,x_0}{1-\bar\alpha_t} = -\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}.$$

<div class="keypoint">

The conditional score is exactly the (negated, rescaled) noise. Minimizing the $\epsilon$-MSE drives the network to $\mathbb E[\epsilon\mid x_t]$, which *is* the marginal score $\nabla_{x_t}\log q_t(x_t)$ the sampler needs. DDPM and score-based generation coincide — same model, two derivations.

</div>
