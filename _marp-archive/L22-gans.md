---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# GANs — the Idea that Outlived the Model

## Lecture 22 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar*

---

# The whole lecture, in one idea

When there is no label — because *every* plausible image is "correct" — you can't write down the loss. So you **train a second network to be the loss.**

<div class="keypoint">

A **GAN** is two networks in a minimax game: a generator $G$ turns noise into images, a discriminator $D$ scores "realness," and **$D$'s gradient is the training signal $G$ never had.** The *model* (GANs) lost to diffusion for generation. The *idea* — a neural network as a **learned, adversarial loss** — is everywhere still.

</div>

$$\underbrace{z\sim\mathcal N(0,I)}_{\text{noise}}\xrightarrow{\;G\;}\underbrace{x}_{\text{fake image}}\;\xrightarrow{\;D\;}\underbrace{[0,1]}_{\text{realness}}\qquad\Longrightarrow\qquad \text{Nash at }p_g=p_{\text{data}},\;\; D\equiv\tfrac12$$

---

# How today connects to the rest of the course

<div class="columns">
<div>

**Where GANs sit in the generative arc:**
- **L21** — the VAE: an *explicit* likelihood, blurry samples, a structured latent
- **L22 (today)** — GANs: *no* explicit likelihood, sharp samples, a brittle game
- **L23** — diffusion: the loss that finally won, a stable regression on noise

</div>
<div>

**The durable idea shows up again in:**
- **L16** — RLHF: the reward model is a **critic** training a policy
- actor–critic RL, domain-adversarial transfer, robust training
- diffusion's **adversarial fine-tuning** for few-step samplers

</div>
</div>

<div class="insight">

Read this as **history with a moral**: we study GANs less for the architecture than for the one idea that survived them — *when you can't write the loss, learn it.*

</div>

---

<!-- _class: section-divider -->

## Part 1 · Generation, and the adversarial idea

---

# From *predicting* to *modelling*

Every model since ES 335 was **discriminative** — logistic regression, CNNs, Transformers all learn $p(y\mid x)$, a label given an input. The loss was easy: compare $\hat y$ to the label.

<div class="insight">

**The new question.** What if there is no $y$? We want to *model the data itself* — learn $p(x)$ and **sample brand-new $x$** (faces, cats) that never existed.

</div>

| From ES 335 | Generative modelling |
|---|---|
| Predict $\hat y = p(y\mid x)$ | Generate $x \sim p(x)$ |
| One right answer per input | Every plausible sample is "right" |
| Loss = compare $\hat y$ to label $y$ | Loss = **???** — no label exists |

That missing loss is the entire problem GANs were invented to solve.

---

# Why not just fit a Gaussian? The network *is* the distribution

Tempting: fit $p_{\text{data}}\approx\mathcal N(\mu,\Sigma)$ and sample. But a Gaussian's support is all of $\mathbb R^d$; natural images live on a thin curved manifold. Gaussian samples are pure noise — never a face.

<div class="keypoint">

The deep-generative move: don't write $p_{\text{data}}$ down. Sample simple noise $z\sim\mathcal N(0,I)$, push it through a network $G$, and let the **output distribution** $p_G$ *be* your model. You never evaluate $p_G(x)$ — you just sample it by running $G$ forward.

</div>

The catch: how do you train $G$ to match $p_{\text{data}}$ when you can't even compute the density you're producing?

---

# The 2014 insight: train a classifier to *be* the loss

Goodfellow's idea: even if you can't compute $p_G$, you *can* train a **classifier $D$** to tell real samples from generated ones. As $D$ learns, its gradient tells $G$ which way to nudge fakes so they fool it.

<div class="keypoint">

**GAN = generator $G$ + adversary $D$, trained together.** $D$'s loss is ordinary binary cross-entropy — perfectly well defined. Its gradient carries *implicit* information about $p_{\text{data}}$ that $G$ consumes **without ever touching a density.**

</div>

One paper turned "there is no loss" into "the loss is a network you also train."

---

# Intuition · the counterfeiter and the detective

Goodfellow's own picture: $G$ is a **counterfeiter** printing fake banknotes, $D$ is the **detective** learning to spot them. Neither is handed an answer key — they teach each other by competing.

<div class="columns">
<div>

**The counterfeiter ($G$)** never studies a real note directly. All it ever learns is *the detective's verdict* — "that one got caught" — and adjusts to make the next batch harder to flag.

</div>
<div>

**The detective ($D$)** sees genuine notes *and* the latest fakes, and sharpens its eye. Every time it improves, it forces the counterfeiter to improve in turn.

</div>
</div>

<div class="insight">

The engine is the **arms race**: each side's progress *raises the bar* for the other. Begin with crude fakes and a lazy detective; end with flawless forgeries and a detective reduced to a coin flip. Neither reaches that ceiling alone — **the competition is the curriculum.**

</div>

---

# Meet the two objects

Before any math, meet $G$ and $D$. $G$ turns noise into an image; $D$ looks at an image and returns one number.

![w:680px](figures/lec20/svg/gd_io.svg)

---

# The two objects, in one breath

<div class="keypoint">

**$G$ · noise → image.** Feed it $100$ random numbers $z$; out comes a full picture $G(z)$. That is the *only* thing $G$ does.

**$D$ · image → one number.** Feed it any image; out comes $D(x)\in[0,1]$ = "how real does this look?" ($1$ = surely real, $0$ = surely fake). That is the *only* thing $D$ does.

</div>

<div class="notebook">

**🎛 Interactive · the minimax dance** — scrub through training steps and watch $G$'s distribution slide onto the real one while $D(x)$ flattens to $0.5$. *(Interactive Lab · `interactive/articles/gan-minimax-dance`)*

</div>

---

# The GAN pipeline

Sample noise → generate a fake → mix with a real batch → $D$ scores everything → both networks update from the same scores.

![w:660px](figures/lec20/svg/gan_loop.svg)

Everything else in this lecture is just *how to train these two so $G$'s images fool $D$.*

---

# Intuition · why an adversarial loss gives *sharp* images

The VAE (L21) minimizes pixel MSE, so where several outputs are equally plausible it hedges toward their **average** — and the average of many sharp faces is a **blur**. A GAN never averages.

<div class="keypoint">

$D$ is a **blur detector for free.** A washed-out face is trivially "fake" — no real photograph looks like that — so $D$ hands $G$ a gradient that shoves it *off* the safe average and *onto* one crisp, committed sample. The adversarial loss rewards **picking a specific answer**, not splitting the difference.

</div>

<div class="insight">

This is the flip side of mode collapse (Part 4). The very force that makes GAN samples **sharp** — commit hard to a mode — is what later tempts $G$ to **abandon** the other modes. Sharpness and coverage pull in opposite directions; that tension runs through the whole lecture.

</div>

<div class="notebook">

**🎛 Interactive · VAE latent explorer** — wander the VAE's latent space and watch reconstructions go smooth and blurry; the contrast with GAN sharpness is the point. *(Interactive Lab · `interactive/articles/vae-latent-explorer`)*

</div>

---

<!-- _class: section-divider -->

## Part 2 · The minimax game

---

# The value function — read straight off binary cross-entropy

$D$ is just a binary classifier, and you already know that loss. $D$ wants $D(x)\to1$ on real $x$ and $D(G(z))\to0$ on fakes:

$$\max_D\;\; \mathbb E_{x\sim p_{\text{data}}}\big[\log D(x)\big] \;+\; \mathbb E_{z\sim p(z)}\big[\log(1 - D(G(z)))\big]$$

$G$ wants the opposite — push $D(G(z))\to1$ — i.e. **minimize** the same expression. Stack them:

$$\boxed{\;\min_G\max_D\; V(D,G)\;}$$

![w:640px](figures/lec20/svg/minimax_game.svg)

---

# Intuition · minimax is "$G$ lowers what $D$ raises"

$V(D,G)$ is **one number two players fight over**: $D$ turns the knob *up*, $G$ turns *the same knob* down.

<div class="columns">
<div>

**$D$ raises $V$** by driving $D(x)\to1$ on real and $D(G(z))\to0$ on fakes — pulling the two piles apart.

**$G$ lowers $V$** through the *only* term it touches, $\log(1-D(G(z)))$, by manufacturing fakes $D$ can no longer push down.

</div>
<div>

So there is no "the loss dropped, we're done." Every gain for $D$ is a loss for $G$ and vice-versa; the needle stops **only when neither can move it** — a saddle, not a valley floor.

</div>
</div>

<div class="insight">

Contrast every optimizer you've met: SGD walks *downhill on a fixed surface*. Here the surface $G$ stands on is **reshaped by $D$ at every step** (and $D$'s by $G$). Same symbols as ordinary loss minimization — completely different dynamics, which is why "just make the loss go down" fails for GANs.

</div>

---

# A 1D toy — watch $G$ learn a bimodal target

Target: two Gaussians at $x=-2$ and $x=2$. Watch $p_G$ march onto them.

![w:680px](figures/lec20/svg/gan_1d_toy.svg)

**Step 0** one wide blob at $0$ ($D$ spots fakes easily) · **100** two bumps emerge · **500** nearly on the modes, $D\approx0.5$ · **1000** $p_G = p_{\text{data}}$, $D$ reduced to guessing. That final state is the **Nash equilibrium** — and the math below pins it down exactly.

---

# Worked numeric · a single step

One real $x$, one fake $G(z)$. Early on, $D$ is learning and $G$ is bad:

- $D(x)=0.7$ — $D$ thinks the real one is somewhat real
- $D(G(z))=0.2$ — $D$ thinks the fake is mostly fake

<div class="math-box">

**$D$'s objective** $=\log 0.7 + \log(1-0.2) = -0.36 - 0.22 = \mathbf{-0.58}$. Maximize → push $D(x)\to1$, $D(G(z))\to0$.

**$G$'s objective** — $G$ only sees the second term $\log(1-0.2)=-0.22$. Minimize → push $D(G(z))\to1$.

</div>

Each step both networks adjust, and each move reshapes the other's landscape — an equilibrium *dance*, not a descent.

---

# Practice problem 1

<div class="popquiz">

**Practice problem 1.** Fix $G$. For a given point $x$, the objective contributes $p_{\text{data}}(x)\log D(x) + p_G(x)\log(1-D(x))$. Maximize over $D(x)$ to find the **optimal discriminator** $D^*(x)$. Then say what $D^*$ equals when $p_G = p_{\text{data}}$.

</div>

*Try it before the next slide.*

---

# Solution · practice problem 1

Write $a=p_{\text{data}}(x)$, $b=p_G(x)$ and maximize $a\log D + b\log(1-D)$ pointwise. Set the derivative to zero:

$$\frac{a}{D}-\frac{b}{1-D}=0 \;\Longrightarrow\; a(1-D)=bD \;\Longrightarrow\; \boxed{\,D^*(x)=\dfrac{p_{\text{data}}(x)}{p_{\text{data}}(x)+p_G(x)}\,}$$

<div class="keypoint">

**Sanity check.** Clearly real ($a{=}0.9,b{=}0.1$) → $D^*=0.9$. Clearly fake ($a{=}0.05,b{=}0.95$) → $D^*=0.05$. And at convergence $p_G=p_{\text{data}}$ → $D^*=\tfrac12$ **everywhere**: the best possible detective is reduced to a coin flip. That $D\equiv\tfrac12$ is the signature of a converged GAN.

</div>

---

# Nash equilibrium — what "converged" means here

<div class="columns">
<div>

**A single-objective optimum** is a *bottom of a valley* — move any direction, loss goes up.

**A Nash equilibrium** is a *saddle* — neither player can improve *unilaterally*. $G$ can't fool $D$ better; $D$ can't detect $G$ better.

</div>
<div>

At the equilibrium:
- $p_G = p_{\text{data}}$ — the fakes are drawn from the true distribution
- $D\equiv\tfrac12$ — no detector can do better than guessing
- both losses have *stopped improving*, not *bottomed out*

</div>
</div>

<div class="insight">

This is why the loss curves are useless as a progress bar: at the target, $D$'s loss sits at $\ln 2\approx0.69$, not $0$. "Winning" looks like a **tie.**

</div>

---

# Worked numeric · why a healthy $D$-loss sits at $\ln 2$

The deck keeps asserting "$D$-loss should hover near $\ln 2\approx0.69$." Here is *why*. At the equilibrium $D\equiv\tfrac12$ on **everything**, so plug $D(x)=D(G(z))=0.5$ into $D$'s binary-cross-entropy (averaging the real- and fake-batch terms, the usual reporting convention):

<div class="math-box">

$$L_D = -\tfrac12\log D(x) - \tfrac12\log\big(1-D(G(z))\big) = -\tfrac12\log\tfrac12 - \tfrac12\log\tfrac12 = -\log\tfrac12 = \ln 2 \approx 0.69$$

</div>

<div class="keypoint">

So $\ln 2$ is **not a target you tune toward** — it is the loss of a detective *reduced to guessing*. If $L_D$ slides toward $0$, $D$ is winning too hard and $G$'s gradient is drying up; if it climbs, $G$ is running away with the game. **Healthy GAN training looks boring: $L_D$ parked at $\ln 2$.**

</div>

---

<!-- _class: section-divider -->

## Part 3 · The non-saturating loss

---

# The saturating problem

Early in training, fakes look nothing like real, so $D(G(z))\approx0$. Then the generator's term $\log(1-D(G(z)))\approx\log 1 = 0$ — **flat.** Exactly when $G$ is worst and needs the most guidance, its gradient vanishes.

![w:660px](figures/lec20/svg/saturating_loss_curve.svg)

---

# The non-saturating fix

Flip which quantity $G$ maximizes. Instead of *minimizing* $\log(1-D(G(z)))$, **maximize** $\log D(G(z))$:

<div class="math-box">

$$\max_G\; \mathbb E_z\big[\log D(G(z))\big]$$

Same fixed point ($D(G(z))\to1$), but a large gradient exactly when $D$ is confident the fake is fake.

</div>

Goodfellow noted this in a **footnote** of the 2014 paper. Everyone has used the non-saturating version ever since — a one-line change that made GANs move at all.

---

# Practice problem 2

<div class="popquiz">

**Practice problem 2.** Let $D$'s pre-sigmoid activation be $a$, so $D(G(z))=\sigma(a)$, and suppose $D$ is confident: $\sigma(a)=0.01$. Compute $\partial L/\partial a$ for **both** the saturating loss $L=\log(1-\sigma(a))$ and the non-saturating loss $L=-\log\sigma(a)$. Which gives $G$ a usable gradient?

</div>

*Try it before the next slide. Use $\sigma'=\sigma(1-\sigma)$.*

---

# Solution · practice problem 2

**Saturating** $L=\log(1-\sigma(a))$. Chain rule gives $\dfrac{\partial L}{\partial a}=-\sigma(a)$. At $\sigma=0.01$: gradient $=\mathbf{-0.01}$.

**Non-saturating** $L=-\log\sigma(a)$. Chain rule gives $\dfrac{\partial L}{\partial a}=\sigma(a)-1$. At $\sigma=0.01$: gradient $=0.01-1=\mathbf{-0.99}$.

<div class="keypoint">

The non-saturating gradient is **~100× stronger** in exactly the common early-training regime where $D$ is winning. Same optimum, wildly different learning speed — the difference between a GAN that trains and one that stalls at step zero.

</div>

---

<!-- _class: code-heavy -->

# The training loop — and the one bug everyone hits

```python
for real in loader:
    # 1. Update D — freeze G with .detach()
    fake = G(torch.randn(B, NZ)).detach()          # <- snip graph at G's output
    loss_D = -(D(real).log() + (1 - D(fake)).log()).mean()
    opt_D.zero_grad(); loss_D.backward(); opt_D.step()

    # 2. Update G — non-saturating, no detach
    fake = G(torch.randn(B, NZ))
    loss_G = -(D(fake).log()).mean()               # maximize log D(G(z))
    opt_G.zero_grad(); loss_G.backward(); opt_G.step()
```

<div class="warning">

Forget `.detach()` in the $D$-update and `backward()` flows *through* $G$, quietly updating it on $D$'s loss. It is one of the top-three from-scratch GAN bugs — training "sort of works," then mysteriously collapses.

</div>

---

# Why GAN training is fundamentally hard

Every optimizer you've met — SGD, Adam — minimizes a **single, fixed** objective. You walk a landscape that doesn't move.

<div class="keypoint">

GANs break that assumption: **neither loss is fixed.** When $G$ moves, $D$'s optimal landscape shifts; when $D$ moves, $G$'s shifts. You are searching for a **saddle point** in a $10^9$-dimensional game, with none of the usual guarantees — convergence, stability, a unique optimum.

</div>

Every trick in GAN lore — DCGAN's recipe, spectral norm, WGAN — is fighting this one fact.

---

<!-- _class: section-divider -->

## Part 4 · Mode collapse, and how you see it

---

# Mode collapse, visually

The nightmare failure: $G$ discovers a *few* outputs that reliably fool $D$ and stops exploring. Ten digits go in; three come out, over and over.

![w:660px](figures/lec20/svg/mode_collapse.svg)

---

# Why it happens — $G$ winning a narrow bet

$G$ is never asked to cover $p_{\text{data}}$. It is only asked to **fool $D$ on whatever it currently produces.** Those are different goals.

<div class="keypoint">

Suppose $G$ finds one mode that reliably fools $D$. $D$ can only punish this by *re-learning* that mode; while it does, $G$ hops to another. The two chase through a handful of modes and settle on whichever is easiest — $p_G$ becomes a spike *inside* $p_{\text{data}}$, not a copy of it.

</div>

The image looks perfect and the loss looks fine. Diversity is the casualty, and the loss never mentions it.

---

# Practice problem 3

<div class="popquiz">

**Practice problem 3.** A student says "mode collapse means $G$ is *losing* — its samples are bad." Argue the opposite: explain why mode collapse is $G$ **winning a narrow bet**, and why the minimax objective *permits* it. Then name one diagnostic that catches it when the sample *quality* looks fine.

</div>

*Try it before the next slide.*

---

# Solution · practice problem 3

<div class="columns">
<div>

**It's winning, locally.** Each collapsed sample genuinely fools $D$ — the objective $\max_G \mathbb E[\log D(G(z))]$ is *low-loss*. Nothing in $\min_G\max_D V$ rewards **coverage**; it only rewards fooling $D$ on the samples $G$ chooses to draw. A point mass on one convincing mode is a valid local win.

</div>
<div>

**The diagnostic.** Per-sample quality can be flawless, so eyeball a *grid* and, quantitatively, watch **FID** (next slide): it compares the *whole* fake distribution to the real one, so it spikes when diversity dies even as individual images stay sharp.

</div>
</div>

<div class="insight">

Mode collapse is the clearest lesson of the whole lecture: **a learned loss only optimizes what it measures.** $D$ never measured diversity, so $G$ never delivered it.

</div>

---

# Diagnosing GAN health

<div class="columns">
<div>

**The loss curves lie.** Watch these instead:
- **$D$ loss** should hover near $\ln 2\approx0.69$ ($D$ outputs $\approx0.5$). If $D$ loss $\to0$, $D$ is winning too hard and $G$'s gradient is drying up.
- **$G$ loss** should plateau, not blow up.
- **Sample grid** — same face on repeat? Mode collapse.

</div>
<div>

**Quantitative:**
- **Inception Score** — quality × diversity, from a pretrained classifier.
- **FID** — Fréchet Inception Distance, the field standard. Compares *distributions* of features, so it catches collapse.

</div>
</div>

A converged GAN's curves look boring and flat — remember, "winning" is a tie.

---

# FID in one paragraph

Run real and fake images through a pretrained Inception-V3, take $2048$-dim features, fit a Gaussian to each set, and measure the distance between them:

<div class="math-box">

$$\text{FID} = \lVert\mu_r-\mu_f\rVert^2 + \operatorname{tr}\!\Big(\Sigma_r+\Sigma_f - 2(\Sigma_r\Sigma_f)^{1/2}\Big)$$

— the Wasserstein-2 distance between two Gaussians fit to the real vs fake feature clouds.

</div>

Lower is better: FID $10$–$20$ "looks good," $3$–$5$ "basically indistinguishable." SOTA diffusion on ImageNet sits near $\sim2$ — a number GANs never comfortably reached at web scale.

---

# Practice problem 4

<div class="popquiz">

**Practice problem 4.** Collapse FID to **1D** features, where it reduces to $\text{FID}=(\mu_r-\mu_f)^2+(\sigma_r-\sigma_f)^2$. Real features are $\mathcal N(0,1)$. Compute FID for **(a)** a healthy fake $\mathcal N(0,1)$ and **(b)** a mode-collapsed $G$ that emits a *single point at the real mean*, $\mathcal N(0,0)$. Each collapsed sample sits **exactly on a real value** — so why does FID still flag it?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 4

$$\text{(a)}\;\; \text{FID}=(0-0)^2+(1-1)^2=\mathbf{0}\qquad\qquad \text{(b)}\;\; \text{FID}=(0-0)^2+(1-0)^2=\mathbf{1}$$

A partly-collapsed $G$ with $\sigma_f=0.2$ lands between them: $(1-0.2)^2=\mathbf{0.64}$ — FID climbs monotonically as coverage shrinks $\sigma_f:1\to0.2\to0$, even though the mean stays perfect.

<div class="keypoint">

FID reads the **second moment** — the spread. Collapse zeroes the fake's variance, so the $(\sigma_r-\sigma_f)^2$ term lights up *even when the means match and every individual sample is realistic.* This is exactly the mode-collapse slide's "per-sample quality flawless, FID spikes": **sample-level realism $\ne$ distribution-level coverage**, and FID measures the latter.

</div>

<div class="notebook">

**🎛 Interactive · seeing the multivariate normal** — FID fits one Gaussian to the real feature cloud and one to the fake, then measures their distance; drag $\mu$ and $\Sigma$ to feel what that distance rewards. *(Interactive Lab · `interactive/articles/multivariate-normal`)*

</div>

---

<!-- _class: section-divider -->

## Part 5 · The machinery, lightly

---

# DCGAN — the recipe that made GANs train

Before 2015, GANs barely produced legible MNIST. **DCGAN** (Radford et al.) offered no new loss — just an *architectural cookbook* that stabilized the game.

![w:660px](figures/lec20/svg/dcgan_architecture.svg)

**The five rules:** strided convs instead of pooling · batch-norm in both nets (except $G$'s output & $D$'s input) · no FC hidden layers · **ReLU** in $G$ (Tanh at the output, matching $[-1,1]$ pixels) · **LeakyReLU** in $D$ (keeps gradients alive on obvious fakes). Each is a small wedge against a known failure — not deep theory, but it made the field tractable.

---

# How $G$ upsamples — transposed convolution

$G$ must grow $(\text{batch},\,100)$ into an image $(\text{batch},\,3,\,32,\,32)$ — the opposite of a normal conv. A **transposed convolution** *inflates*: each input pixel is multiplied by the kernel and spread into a larger patch.

<div class="math-box">

**Dimension formula** (inverse of a conv): $\;O=(W-1)\cdot S - 2P + K$.

From a $1{\times}1$ seed with $K{=}4,S{=}1,P{=}0\to4{\times}4$; then stride-$2$ blocks double each time: $4\to8\to16\to32$.

</div>

Cleaner alternative — nearest-neighbour upsample **then** a regular conv — avoids the "checkerboard" artifacts transposed convs are notorious for.

---

# WGAN — a better distance for disjoint supports

The vanilla objective secretly minimizes **Jensen–Shannon divergence**. When $p_G$ and $p_{\text{data}}$ don't overlap — always true early on — JS is *constant* at $\log 2$, and the gradient of a constant is zero. $G$ gets no signal.

![w:660px](figures/lec20/svg/js_vs_wasserstein.svg)

JS jumps to a flat $\log 2$ the instant the supports separate. **Wasserstein** distance grows *smoothly* with the gap — so it always has a slope pointing $G$ toward the data. That single difference is the whole reason WGAN trains more gently.

---

# Earth-mover distance — the picture

Think of two piles of sand (two distributions). **How much work** to reshape one into the other, moving each grain the least distance?

![w:660px](figures/lec20/svg/wasserstein_intuition.svg)

$$W(p,q)=\inf_\gamma\;\mathbb E_{(x,y)\sim\gamma}\big[\lVert x-y\rVert\big]$$

Move the pile $1$ metre → $W=1$; $10$ metres → $W=10$. Unlike JS, $W$ keeps varying **smoothly even when the piles don't overlap** — a live gradient exactly where JS flatlines.

---

# Practice problem 5

<div class="popquiz">

**Practice problem 5.** Let $p$ be a point mass at $x=0$ and $q$ a point mass at $x=d>0$ — **disjoint supports** for every $d>0$. (a) Give the Earth-mover distance $W(p,q)$ as a function of $d$. (b) Give the Jensen–Shannon divergence $\text{JSD}(p\Vert q)$. (c) Differentiate each in $d$. Which one hands $G$ a gradient that *closes the gap*, and which is flat?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 5

<div class="columns">
<div>

**(a)** Move one unit of mass a distance $d$: $\;W(p,q)=d$.

**(b)** The supports never overlap, so with mixture $m=\tfrac12(p+q)$ each KL term is $\log 2$:
$$\text{JSD}=\tfrac12\log2+\tfrac12\log2=\log 2\;(\approx0.69\text{ nats})$$
— a **constant**, independent of $d$.

</div>
<div>

**(c)** $\dfrac{dW}{dd}=1$ — a fixed, nonzero slope pulling $q$ toward $p$.

$\dfrac{d\,\text{JSD}}{dd}=0$ — flat everywhere the supports are disjoint.

</div>
</div>

<div class="keypoint">

This is the entire WGAN argument in two derivatives. Vanilla GANs minimize a JS-like quantity whose gradient is **zero while the fakes are far away** — precisely when $G$ most needs a push. Wasserstein slopes downhill the *whole way in*: **a smooth gradient even when the two distributions don't overlap.**

</div>

---

# What WGAN fixed — and what it didn't

<div class="columns">
<div>

**Fixed**
- smooth gradient even for disjoint supports — $G$ always has a signal
- loss now *correlates* with sample quality — the curve finally means something
- mode collapse becomes rarer
- less hyperparameter-fragile

</div>
<div>

**Didn't fix**
- it's still a two-player game — no convergence guarantee
- WGANs still oscillate and diverge
- the $1$-Lipschitz critic needs enforcing (weight clipping → gradient penalty → spectral norm)

</div>
</div>

<div class="insight">

WGAN fixed the *gradient* problem, not the *game* problem. That distinction is why, even with WGAN, generation eventually moved on.

</div>

---

# StyleGAN — the peak, in one slide

Karras et al. (NVIDIA) pushed the GAN to its ceiling: project $z$ into a **disentangled** latent $w$, then inject it at every resolution.

![w:620px](figures/lec20/svg/stylegan_w_hierarchy.svg)

Coarse blocks control pose and identity; middle blocks, hair and features; fine blocks, lighting and texture. Mix coarse-$w$ from one face with fine-$w$ from another → "style mixing," and $1024{\times}1024$ faces indistinguishable from photographs. This was the high-water mark — *and* the moment diffusion arrived.

---

<!-- _class: section-divider -->

## Part 6 · The honest close

---

# The GAN era · 2014–2021

| Year | Model | What it did |
|------|-------|-------------|
| 2014 | GAN | the original idea — $28{\times}28$ MNIST |
| 2015 | DCGAN | convolutional recipe, stable-ish training |
| 2017 | WGAN-GP | Wasserstein + gradient penalty |
| 2018 | ProGAN | progressive growing → $1024^2$ faces |
| 2019 | StyleGAN | disentangled latent, photoreal faces |
| 2021 | StyleGAN3 | aliasing fixes, temporal consistency |

Seven years from "legible digits" to "indistinguishable faces" — then the ground shifted.

---

# Diffusion won — the honest scorecard

| | GANs | Diffusion |
|---|------|-----------|
| Training | ✗ brittle minimax game | ✓ stable regression on noise |
| Sample quality | ✓ (SOTA 2018–20) | ✓✓ (SOTA 2021+) |
| Mode coverage | ✗ collapse risk | ✓ natural |
| Text conditioning | hacks | ✓ cross-attention each step |
| Inference speed | ✓✓ one pass | ✗ many passes |

<div class="keypoint">

Diffusion's loss (**predict the noise — MSE**) has a *unique global optimum* and no adversary. It beat GANs *at their own strength*, sample quality, while being simultaneously easier to train, cover, and condition. GANs weren't wrong — they were **out-engineered by a loss you can actually minimize.**

</div>

---

# When to still reach for a GAN

Diffusion won *by default*, but GANs keep a few genuine niches:

- **Real-time generation** — one forward pass beats $20$+ denoising steps by orders of magnitude (live avatars, on-device faces).
- **Editing existing content** — GAN inversion into $w$-space, edit, re-decode, is still the cleanest latent-editing pipeline.
- **Small, closed domains** — faces, fashion, a single object class — where StyleGAN still competes on FID.

<div class="insight">

The precedent is RNNs vs Transformers: RNNs didn't vanish, they found niches (streaming, tiny devices). GANs are the same story — outperformed, not erased.

</div>

---

# The idea that outlived the model

GANs lost generation. But the core move — **a neural network can be a learned loss function** — is now everywhere:

- **RLHF (L16)** — a reward model, trained on human preference pairs, is a *critic* shaping a language model.
- **Actor–critic RL** — the value network is a learned reward signal for the policy.
- **Domain-adversarial training** — a domain discriminator forces features to transfer.
- **Diffusion, today** — adversarial *fine-tuning* distills many denoising steps into a few, GAN-style.

<div class="insight">

**When you can't write the loss, learn it.** That framework outlived the original GAN — which is exactly why we still teach it. *Next (L23): drop the adversary entirely — corrupt data with noise, and learn to reverse it.*

</div>

---

# Explore it yourself

<div class="notebook">

**🎛 Interactive · the minimax dance** — scrub training and watch $p_G$ slide onto $p_{\text{data}}$ while $D(x)\to\tfrac12$; toggle saturating vs non-saturating $G$ loss and see the early gradient change. *(Interactive Lab · `interactive/articles/gan-minimax-dance`)*

</div>

<div class="notebook">

**📓 Notebook · DCGAN on MNIST** — build $G$/$D$, train with the non-saturating loss, monitor the $D$-loss $\approx\ln2$ balance, and deliberately *induce* mode collapse to watch FID spike. *(ES 667 · `notebooks/dcgan-mnist.ipynb`)*

</div>

<div class="notebook">

**🎛 Interactive · denoising diffusion** — the model that won; a preview of L23. *(Interactive Lab · `interactive/articles/diffusion-denoise`)*

</div>

---

<!-- _class: summary-slide -->

# One sentence to remember

<div class="keypoint">

**When you can't write the loss, train a network to be the loss** — that adversarial idea outlived the GANs that introduced it.

</div>

- A GAN is a **minimax game**: $G$ turns noise into images, $D$ scores realness, $D$'s gradient trains $G$. Nash at $p_G=p_{\text{data}}$, $D\equiv\tfrac12$.
- **Non-saturating $G$ loss** — maximize $\log D(G(z))$; ~$100\times$ stronger early gradient.
- **Mode collapse** — $G$ wins a narrow bet; the loss never measured diversity, so **FID** must.
- **DCGAN** stabilized the architecture; **WGAN** swapped JS for a smooth Earth-mover distance.
- **Diffusion won generation**, but *learned adversarial loss* lives on in RLHF, actor–critic, and diffusion fine-tuning.

**Next (L23):** the loss that beat GANs — forward noising, and learning to reverse it.

---

<!-- _class: compact -->

# Sources & credits

<div class="paper">

This lecture reuses and adapts material from the instructor's ES 667 GAN materials (IIT Gandhinagar) and standard references:

- **Original GAN, minimax game, non-saturating loss** — I. Goodfellow et al., *Generative Adversarial Nets*, NeurIPS 2014; and Goodfellow, *NIPS 2016 GAN Tutorial*.
- **Architectural recipe** — A. Radford, L. Metz, S. Chintala, *DCGAN*, ICLR 2016.
- **Earth-mover distance, 1-Lipschitz critic** — M. Arjovsky, S. Chintala, L. Bottou, *Wasserstein GAN*, ICML 2017 (gradient penalty: Gulrajani et al. 2017).
- **FID / Fréchet Inception Distance** — M. Heusel et al., *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium*, NeurIPS 2017.
- **Disentangled style latent** — T. Karras et al., *StyleGAN* (1–3), NVIDIA 2019–2021.
- **Pedagogical framing** ("when you can't write the loss, learn it") — A. Ng, *Deep Learning Specialization*; ES 667 figure library. All source material © N. Batra & teaching staff.

</div>

---

<!-- _class: section-divider -->

## ⭐⭐⭐ Appendix · optional depth

*Skip on a first pass — for the curious.*

---

<!-- _class: math-heavy -->

# ⭐⭐⭐ Optional · the GAN objective *is* Jensen–Shannon

Plug the optimal $D^*(x)=\dfrac{p_{\text{data}}}{p_{\text{data}}+p_G}$ back into $V(D^*,G)$:

$$V(D^*,G)=\mathbb E_{p_{\text{data}}}\!\Big[\log\tfrac{p_{\text{data}}}{p_{\text{data}}+p_G}\Big] + \mathbb E_{p_G}\!\Big[\log\tfrac{p_G}{p_{\text{data}}+p_G}\Big]$$

Insert $\log 2 - \log 2$ into each term and regroup around the mixture $m=\tfrac12(p_{\text{data}}+p_G)$:

$$V(D^*,G) = -\log 4 + \underbrace{D_{\text{KL}}(p_{\text{data}}\Vert m) + D_{\text{KL}}(p_G\Vert m)}_{2\,\text{JSD}(p_{\text{data}}\Vert p_G)} = 2\,\text{JSD}(p_{\text{data}}\Vert p_G) - \log 4$$

Since $\text{JSD}\ge0$ with equality iff $p_G=p_{\text{data}}$, the global minimum is $-\log 4$ at $p_G=p_{\text{data}}$ — and there $D^*\equiv\tfrac12$. This is also *why* WGAN exists: JSD is flat on disjoint supports, so its gradient dies exactly when $p_G$ is far from $p_{\text{data}}$.

---

<!-- _class: math-heavy -->

# ⭐⭐⭐ Optional · WGAN duality and the gradient penalty

The Earth-mover distance is an infimum over transport plans — intractable directly. **Kantorovich–Rubinstein duality** turns it into a maximization over $1$-Lipschitz critics:

$$W(p_{\text{data}},p_G)=\sup_{\lVert D\rVert_L\le1}\;\mathbb E_{p_{\text{data}}}[D(x)] - \mathbb E_{p_G}[D(G(z))]$$

No log, no sigmoid — $D$ outputs a raw **score**, maximizing the gap (high on real, low on fake); $G$ pushes its fakes' scores up. Enforcing $\lVert D\rVert_L\le1$ is the whole game. **WGAN-GP** samples random interpolates $\hat x=\epsilon x+(1-\epsilon)G(z)$ and penalizes any slope $\ne 1$:

$$\mathcal L_{\text{GP}}=\lambda\,\mathbb E_{\hat x}\big[(\lVert\nabla_{\hat x}D(\hat x)\rVert_2 - 1)^2\big]$$

A "highway-patrol" trick: you can't check the Lipschitz bound everywhere, so you check random points — and in practice that is enough to keep training stable.
