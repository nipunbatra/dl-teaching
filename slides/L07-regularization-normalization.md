---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# Regularization & Normalization

## Lecture 7 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar*

---

# The whole lecture, in one idea

A network with a billion parameters can **memorize** its training set and learn nothing transferable. Every technique today answers the *same* question: **how do we stop it?**

<div class="keypoint">

**There are exactly three places to intervene.** Penalize the **weights** (classical: L2/L1), reshape the **data** (augmentation, Mixup), or rewire the **architecture** (dropout, normalization). The surprise of deep learning is that the last two — which never add a $\lambda\lVert\theta\rVert$ term — usually win.

</div>

From L1 you already own the deepest one: **a penalty on the weights is a prior.** L2 is a Gaussian prior, L1 a Laplace prior — MAP $=$ MLE $+$ a log-prior. Today we build out from there.

$$\underbrace{\text{L2 / L1}}_{\text{prior on }\theta} \;\rightarrow\; \underbrace{\text{augment · Mixup · label smoothing}}_{\text{prior on the data}} \;\rightarrow\; \underbrace{\text{dropout · BatchNorm · LayerNorm}}_{\text{prior in the wiring}}$$

---

# How today connects to the rest of the course

<div class="columns">
<div>

**You'll reuse today's ideas in:**
- **L8–L10** — BatchNorm lives inside every CNN; weight decay in every training script
- **L12–L15** — LayerNorm / RMSNorm and **pre-norm** are load-bearing for every Transformer
- **L16–L18** — dropout `p=0.1` and label smoothing in LLM pre-training
- **L21** — the VAE's KL term is *itself* a regularizer (a prior on the latent)

</div>
<div>

**Borrowed from your own courses** (we build on, not repeat):
- **ML (ES 335)** — ridge & lasso: Lagrangian, closed form, L1-vs-L2 geometry
- **L1 (this course)** — MAP: L2 $=$ Gaussian prior, L1 $=$ Laplace prior

</div>
</div>

<div class="insight">

We move **fast** where ES 335 already took you (ridge algebra), and **slow** on what is new to deep nets (dropout, normalization, double descent).

</div>

---

# The regularization menu — one spine for the lecture

![w:920px](figures/lec06/svg/reg_menu.svg)

You already own the **classical** family from ES 335 (we skim it). The two families where deep learning actually *wins* — reshape the **data**, rewire the **architecture** — are where we spend most of today.

---

<!-- _class: section-divider -->

## Part 1 · Overfitting — the disease every regularizer treats

---

# Two students, one exam

<div class="keypoint">

**Student A memorizes** the 100 practice problems. Aces the practice test, fails the real exam — the questions changed.

**Student B learns the method.** Does fine on both.

Regularization is every trick we have for forcing the model to be **Student B**.

</div>

A deep network has more than enough capacity to memorize the training set perfectly while learning nothing that transfers. Left alone, it will. Our job is to make transferable patterns the *easier* thing for it to find.

---

# The fingerprint of overfitting: weights explode

Fit a polynomial $f(x)=c_0+c_1x+\dots+c_dx^d$ and push the degree up. As you cross from *fitting* into *memorizing*, one number blows up:

$$\text{degree} \uparrow \;\Longrightarrow\; \max_i |c_i| \uparrow\uparrow \quad(\text{coefficients grow to bend through every point})$$

<div class="insight">

**Large weights are the signature of overfitting.** That is the entire motivation for the classical fix: if big weights mean memorization, *penalize big weights.* Ridge and lasso are two ways to write that penalty down.

</div>

---

# Bias–variance: the classical account

| | High **bias** (underfit) | High **variance** (overfit) |
|---|---|---|
| Model | too simple | too complex / too flexible |
| Train error | high | very low |
| Test error | high | high |
| Weights | small, stable | large, jumpy across resamples |
| Fix | more capacity | **regularize** — add bias to cut variance |

<div class="keypoint">

Regularization deliberately **adds a little bias to remove a lot of variance.** Total error $=\text{bias}^2+\text{variance}+\text{irreducible}$ — and the trade usually lowers the sum. (We'll see in Part 5 that the U-curve isn't the whole story.)

</div>

---

<!-- _class: section-divider -->

## Part 2 · Classical — penalize the weights

---

# Ridge: turn a constraint into a penalty

We *want* small weights. State that as a hard constraint, then use a Lagrange multiplier to make it a penalty:

<div class="math-box">

$$\min_{\theta}\;\lVert y - X\theta\rVert_2^2 \quad\text{s.t.}\quad \lVert\theta\rVert_2^2 \le S \qquad\Longleftrightarrow\qquad \min_{\theta}\;\underbrace{\lVert y - X\theta\rVert_2^2}_{\text{fit the data}} + \underbrace{\lambda\lVert\theta\rVert_2^2}_{\text{keep }\theta\text{ small}}$$

</div>

The multiplier $\lambda\ge 0$ *is* the regularization strength: $\lambda=0$ is plain least squares; $\lambda\to\infty$ crushes every weight to $0$. **Small $S$ (tight leash) $\leftrightarrow$ large $\lambda$.**

---

# Ridge: the closed form in four steps

<div class="math-box">

**1. Objective** $\;J(\theta)=(y-X\theta)^{\!\top}(y-X\theta)+\lambda\,\theta^{\!\top}\theta$

**2. Gradient** $\;\dfrac{\partial J}{\partial\theta} = -2X^{\!\top}y + 2X^{\!\top}X\theta + 2\lambda\theta$

**3. Set to zero** $\;(X^{\!\top}X+\lambda I)\,\theta = X^{\!\top}y$

**4. Solve** $\;\boxed{\;\hat\theta_{\text{ridge}}=(X^{\!\top}X+\lambda I)^{-1}X^{\!\top}y\;}$

</div>

Compare OLS: $\hat\theta_{\text{OLS}}=(X^{\!\top}X)^{-1}X^{\!\top}y$. The only change is $+\lambda I$ inside the inverse — and that tiny change buys a property OLS never had.

---

# Practice problem 1

<div class="popquiz">

**Practice problem 1.** OLS breaks when $X^{\!\top}X$ is singular (collinear features, or more features than data). Ridge inverts $X^{\!\top}X+\lambda I$ instead. Show that for **any** $\lambda>0$ this matrix is invertible — no matter what $X$ is. *(Hint: what are the eigenvalues of $X^{\!\top}X$, and what does adding $\lambda I$ do to them?)*

</div>

*Try it before the next slide.*

---

# Solution · practice problem 1

$X^{\!\top}X$ is **positive semi-definite**: for any $v$, $\;v^{\!\top}(X^{\!\top}X)v=\lVert Xv\rVert^2\ge 0$, so every eigenvalue $\mu_i\ge 0$.

Adding $\lambda I$ shifts **every** eigenvalue up by $\lambda$:

$$X^{\!\top}X = Q\,\text{diag}(\mu_i)\,Q^{\!\top} \;\Longrightarrow\; X^{\!\top}X+\lambda I = Q\,\text{diag}(\mu_i+\lambda)\,Q^{\!\top}$$

<div class="keypoint">

Each eigenvalue is now $\mu_i+\lambda\ge\lambda>0$, so the matrix is **positive definite** — hence full-rank and invertible, always. Ridge doesn't just shrink weights; it **cures the singularity** of OLS (multicollinearity, $p>n$). The penalty and the numerical fix are the *same* $+\lambda I$.

</div>

---

# Ridge in numbers: shrinkage you can watch

Data $(1,1),(2,2),(3,3),(4,0)$, model $y=\theta_0+\theta_1 x$. Then $X^{\!\top}X=\begin{bmatrix}4&10\\10&30\end{bmatrix}$, $X^{\!\top}y=\begin{bmatrix}6\\14\end{bmatrix}$.

<div class="columns">
<div>

**OLS** — invert $X^{\!\top}X$:
$$\hat\theta_{\text{OLS}}=\begin{bmatrix}2.0\\-0.2\end{bmatrix},\quad \lVert\theta\rVert_2^2=4.04$$

</div>
<div>

**Ridge**, $\lambda=2$ — invert $X^{\!\top}X+2I$:
$$\hat\theta_{\text{ridge}}=\begin{bmatrix}0.565\\0.261\end{bmatrix},\quad \lVert\theta\rVert_2^2=0.39$$

</div>
</div>

A **90% cut** in weight magnitude — same data, one added $\lambda I$. Ridge shrinks; it never sets a weight *exactly* to zero. For that we need L1.

---

# L2 = weight decay: read it off the update

SGD on the ridge loss, then group the two $\theta$ terms:

$$\theta_{\text{new}} = \theta - \eta\,\frac{\partial L_{\text{data}}}{\partial\theta} - \eta\lambda\theta \;=\; \underbrace{(1-\eta\lambda)}_{<\,1}\,\theta \;-\; \eta\,\frac{\partial L_{\text{data}}}{\partial\theta}$$

<div class="keypoint">

Every step first **shrinks** the weight by a factor $(1-\eta\lambda)$, then applies the data gradient. That is *literally* why "L2 regularization" and "**weight decay**" are the same thing.

</div>

<div class="realworld">

In PyTorch this is one argument: `AdamW(..., weight_decay=0.05)`. (Why the decoupled `AdamW` variant matters for adaptive optimizers — see L5.)

</div>

---

# Callback to L1: a prior *is* a regularizer

From L1: maximizing the posterior adds $-\log p(\theta)$ to the loss. The **shape of the prior** picks the penalty.

<div class="columns">
<div>

**Gaussian prior** $\;\theta\sim\mathcal N(0,\tau^2)$
$$-\log p(\theta)\propto \lambda\lVert\theta\rVert_2^2$$
$\Rightarrow$ **L2 / ridge / weight decay**
$\;$*"weights are small"*

</div>
<div>

**Laplace prior** $\;\theta\sim\text{Laplace}(0,b)$
$$-\log p(\theta)\propto \lambda\lVert\theta\rVert_1$$
$\Rightarrow$ **L1 / lasso**
$\;$*"weights are small **and many are exactly 0**"*

</div>
</div>

<div class="insight">

Regularization was never a hack bolted onto the loss — it is a **belief about the weights**, written as a prior. Choosing L2 vs L1 is choosing the *shape* of that belief.

</div>

---

# Why L1 gives sparsity — the geometry

The penalty's contour is the whole story: L2 is a **circle** (smooth everywhere), L1 is a **diamond** (corners on the axes). The optimum is where the data's loss contours first touch the constraint region.

![w:900px](figures/lec06/svg/l1_l2_geometry.svg)

The circle is usually touched *off-axis* — every weight nonzero. The diamond's touch point is very often **a corner**, which sits *on an axis* — meaning a weight is exactly $0$. Sharp corners $\Rightarrow$ sparsity.

---

# L1 vs L2 — the gradient tells the same story

Look at the pull each penalty exerts on a single weight as it approaches zero:

<div class="columns">
<div>

**L2**, $\;\tfrac12\theta^2$: gradient $=\theta$
Shrink is **proportional** — vanishes as $\theta\to 0$.
$$5\to2.5\to1.25\to\dots\;(\text{never }0)$$

</div>
<div>

**L1**, $\;|\theta|$: subgradient $=\text{sign}(\theta)$
A **constant** push toward $0$, size $\lambda$, whatever the magnitude.
$$5\to4.5\to4.0\to\dots\to 0\;(\text{finite steps})$$

</div>
</div>

<div class="keypoint">

L2's pull fades as the weight shrinks, so it only *approaches* zero. L1's push is a fixed $\pm\lambda$ that a weak data gradient cannot beat — so small weights get driven to **exactly** zero (soft-thresholding).

</div>

---

# Practice problem 2

<div class="popquiz">

**Practice problem 2.** After training, two weights are tiny: $\theta_a=0.001$, and both an L1 and an L2 penalty are pulling on them. Which penalty will drive such a small weight to **exactly** zero, and which will only shrink it toward zero forever? Explain using the gradient of each penalty at small $\theta$ — and say why DL nonetheless prefers L2.

</div>

*Try it before the next slide.*

---

# Solution · practice problem 2

**L1 zeroes it; L2 does not.** At $\theta=0.001$:

- **L2 gradient** $=\lambda\theta=0.001\lambda$ — *proportional* to the weight, so it shrinks toward but never reaches $0$ (asymptotic).
- **L1 subgradient** $=\lambda\,\text{sign}(\theta)=\lambda$ — a *constant* push. Once $|\rho_j|\le\lambda/2$ (data correlation weaker than the penalty), soft-thresholding snaps $\theta_j$ to **exactly** $0$.

<div class="insight">

**So why does deep learning still reach for L2, not L1?** Features in a neural net are **distributed** across many weights, not localized in a few. L1's zeros would delete pieces of a shared representation. L2 shrinks *everything* gently — the right prior for dense, distributed features. L1 shines when you expect *few relevant inputs* (genomics, sparse signals).

</div>

---

# Early stopping — regularization for free

**Q.** If validation loss starts rising at epoch 30, why bother stopping — the function class hasn't changed?

Because more steps means walking **further** into the parameter landscape: weights keep growing, and the model starts fitting training noise. Stopping early caps the *effective* capacity used — without any $\lambda$ term at all.

<div class="keypoint">

Early stopping is implicit capacity control: **free, and almost always helps.** Every serious training script checkpoints on validation loss.

</div>

<div class="notebook">

**📓 Notebooks · see ridge & lasso in action** — sweep $\lambda$ and watch coefficients shrink (ridge) vs snap to zero (lasso). *(ML ES 335 · `notebooks/ridge.ipynb`, `notebooks/lasso-regression.ipynb`)*
**🎛 Interactive Lab · L1 vs L2 geometry** — drag the OLS point and watch the diamond corner vs the circle decide sparsity. *(Interactive Lab · `~/git/interactive`)*

</div>

---

<!-- _class: section-divider -->

## Part 3 · Regularize the data, not the model

---

# Data augmentation — teach the invariances

<div class="insight">

Classical regularization constrains the **model**. Augmentation constrains the **data** — it tells the model which transformations *should not change the label*. A flipped cat is still a cat; a color-jittered cat is still a cat.

</div>

![w:820px](figures/lec06/svg/aug_examples.svg)

By training on transformed copies, the model **learns** the invariance instead of us hard-coding it — often the single highest-value regularizer for vision.

---

# Which augmentation for which problem

| Domain | Useful | Avoid — it destroys the label |
|--------|--------|-------|
| Natural images | flip · crop · color jitter · rotate | vertical flip (sky↔ground) |
| Medical imaging | small rotations · mild intensity | flips (mirror anatomy) |
| MNIST / digits | small rotations · elastic | flip (a 6 becomes a 9) |
| Satellite | any rotation (no "up") · flip | heavy color jitter (semantic) |
| Text | synonym · back-translation · mask | random char shuffle |

<div class="keypoint">

**The one rule:** augmentation must **preserve the label**. If the transform changes the true answer, you're injecting noise, not signal. *(2026 default for vision: `transforms.RandAugment(num_ops=2, magnitude=9)` — two knobs, +1–3% accuracy.)*

</div>

---

# Mixup & CutMix — augment the *label* too

Standard augmentation reshapes one image, keeping its label. **Mixup** blends *two* images and interpolates their labels to match.

![w:600px](figures/lec06/svg/mixup_cutmix.svg)

<div class="math-box">

Pick $\lambda\sim\text{Beta}(\alpha,\alpha)$, then
$x_{\text{mix}}=\lambda x_1+(1-\lambda)x_2,\qquad y_{\text{mix}}=\lambda y_1+(1-\lambda)y_2$

</div>

Forcing the model to predict a *blend* smooths the decision boundary and calibrates confidence — a free ~1–2% on CIFAR (skip it for boxes/masks, which don't interpolate).

---

# Mixup · the label, worked in numbers

Cat $=$ class 0, dog $=$ class 1, one-hot $y_1=[1,0]$, $y_2=[0,1]$, and $\lambda=0.7$:

$$x_{\text{mix}}=0.7\,x_1+0.3\,x_2 \qquad y_{\text{mix}}=0.7[1,0]+0.3[0,1]=[0.7,\,0.3]$$

Equivalently, mix the **loss** (no need to build a soft label):

$$\text{loss}=\lambda\,\text{CE}(\text{logits},y_1)+(1-\lambda)\,\text{CE}(\text{logits},y_2)$$

<div class="insight">

The model sees a faded cat with a 30%-opacity dog and must *output* $[0.7,0.3]$ to minimize loss. It learns predictions can live **between** classes — calibrated outputs, for free, from inside the training loop.

</div>

---

# Label smoothing — because "1.0 for the true class" is a lie

Hard one-hot targets push the logits toward $\pm\infty$: overconfident, miscalibrated. Bleed a little mass off the true class onto all $K$ classes.

![w:760px](figures/lec06/svg/label_smoothing_bars.svg)

$$y_{\text{smooth}}=(1-\alpha)\,y_{\text{hard}}+\frac{\alpha}{K} \qquad\text{e.g. class 3, }K{=}10,\ \alpha{=}0.1:\ [0.01,\dots,\mathbf{0.91},\dots,0.01]$$

Better calibration, robustness to wrong labels — one flag: `CrossEntropyLoss(label_smoothing=0.1)`. *(Downside: smoothed models make weaker distillation teachers — the inter-class structure flattens.)*

---

<!-- _class: section-divider -->

## Part 4 · Regularize the architecture — dropout & normalization

---

# Dropout — what the layer actually does

<div class="keypoint">

**Effect first.** At **train** time each layer outputs a randomly **masked, rescaled** copy of its activations — some units zeroed, survivors amplified by $1/p$. At **eval** time dropout is off: the layer passes activations through unchanged.

</div>

![w:820px](figures/lec06/svg/dropout_network.svg)

Every forward pass: sample a mask $m\sim\text{Bernoulli}(p)$ per unit, zero the dropped units, rescale the rest. *Why would deliberately deleting neurons help?*

---

# Two intuitions for why dropout helps

<div class="columns">
<div>

### Ensemble view

Each mini-batch trains a **different sub-network** (a random subset of units).

Over many batches you implicitly train an **ensemble of $2^N$ thinned networks** that share weights.

At test time (no mask) you get their average — ensembles generalize.

</div>
<div>

### Co-adaptation view

Without dropout, units **co-adapt** — neuron $j$ leans on neuron $k$ always being alive.

Dropout removes that guarantee, forcing each unit to be useful **on its own**.

The result is a more **distributed**, robust representation.

</div>
</div>

---

# Inverted dropout — why divide by $p$

We want train and eval to agree in expectation. For one unit with keep-probability $p$:

- **Eval:** dropout off, output $=h$. *(this is the target)*
- **Train:** output is random — with prob $p$ the scaled value $h/p$, with prob $1-p$ a zero.

$$\mathbb E[h_{\text{drop}}] = p\cdot\frac{h}{p} + (1-p)\cdot 0 = h \;\checkmark$$

<div class="math-box">

$$\mathbf h_{\text{drop}} = \frac{\mathbf h\odot\mathbf m}{p},\qquad \mathbf m\sim\text{Bernoulli}(p)$$

</div>

Dividing survivors by $p$ **at train time** keeps the expected activation equal to the eval activation — so we can drop the mask at test time and change nothing. This is **inverted dropout**.

---

# Practice problem 3

<div class="popquiz">

**Practice problem 3.** Hidden activations $h=[2.0,\,1.5,\,0.5,\,3.0]$, keep-prob $p=0.5$. A forward pass samples the mask $m=[1,0,1,0]$. **(a)** What does inverted dropout output on this pass? **(b)** Show the *expected* train output over random masks equals the eval output. **(c)** What does the layer output at eval time?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 3

**(a) This pass** — mask, then divide by $p$:
$$h_{\text{drop}}=(h\odot m)/p=[2.0,0,0.5,0]/0.5=[\mathbf{4.0},\,0,\,\mathbf{1.0},\,0]$$
The two survivors are *amplified* $2\times$ to compensate for the dropped half.

**(b) Expected output** (per unit): $\;p\cdot(h/p)+(1-p)\cdot 0 = h = [2.0,1.5,0.5,3.0]$ — same as eval.

**(c) Eval:** no mask, no scaling — $\;h_{\text{eval}}=[2.0,1.5,0.5,3.0]$.

![w:560px](figures/lec06/svg/dropout_effect.svg)

The $1/p$ train-time amplification is exactly what lets us **turn the mask off at test time** without shifting the network's expected input.

---

# Where dropout lives in 2026 — and one nasty trap

| Architecture | Typical dropout |
|--------------|-------------|
| Large MLPs | `p = 0.5` between hidden layers |
| CNNs | usually **none** — BatchNorm + augmentation already regularize |
| Transformers | `p = 0.1` after attention and FFN |
| Fine-tuning an LLM | `0.0`–small — data is scarce |

<div class="warning">

**Two traps.** (1) `nn.Dropout(p)` uses $p$ as the **drop** probability — keep-prob $0.8$ is `nn.Dropout(0.2)`. (2) Dropout only turns off if you call `model.eval()`; forgetting it leaves the mask on at inference — great training, garbage deployment. `assert not model.training` at your inference entry point.

</div>

---

# Normalization — reshape the loss landscape

<div class="keypoint">

**Effect first.** A norm layer outputs a **re-centered, re-scaled** version of each activation (mean $\approx 0$, unit variance), then a learned $\gamma,\beta$ lets the network choose the scale it actually wants back.

</div>

**The hiker-in-a-canyon picture:** an ill-scaled loss is a narrow steep canyon — the optimizer bounces side to side and crawls forward. Normalizing makes each direction's curvature comparable — the canyon becomes a **round bowl**, and the optimizer walks straight down. Same minimum, far easier trajectory. It also fixes **scale drift**: activations that grow or shrink with depth get re-centered at *every* step (He init only fixes it at initialization).

---

# BatchNorm · LayerNorm · RMSNorm — same recipe, three axes

The only question is **which slice you average over.**

![w:860px](figures/lec06/svg/bn_vs_ln_vs_rms.svg)

- **BatchNorm** — normalize each *feature* across the **batch**. Needs big, consistent batches → CNNs.
- **LayerNorm** — normalize each *sample* across its **features**. Batch-independent → sequences/Transformers.
- **RMSNorm** — like LayerNorm but **skip the mean**, divide by RMS only. Cheaper, no worse → modern LLMs.

---

# BatchNorm — the four steps, in numbers

One neuron's activations over a batch: $x=[1,3,5,7]$.

<div class="columns">
<div>

**1. Mean** $\;\mu=\tfrac{1+3+5+7}{4}=4.0$

**2. Variance** $\;\sigma^2=\tfrac{9+1+1+9}{4}=5.0,\ \sigma\approx2.236$

</div>
<div>

**3. Normalize** $\;\hat x=\dfrac{x-\mu}{\sqrt{\sigma^2+\epsilon}}\approx[-1.34,-0.45,0.45,1.34]$

**4. Scale + shift** $(\gamma{=}2,\beta{=}0.5)$: $\;y=\gamma\hat x+\beta\approx[-2.18,-0.39,1.39,3.18]$

</div>
</div>

<div class="insight">

Steps 1–3 force mean 0 / unit variance; step 4's learnable $\gamma,\beta$ hand the scale *back* to the network — maybe unit variance isn't what the next layer wants. The identity init $\gamma{=}1,\beta{=}0$ means BN starts as a no-op and learns from there.

</div>

---

# BatchNorm — train vs eval, and where it fails

![w:760px](figures/lec06/svg/bn_train_eval.svg)

At **train** time BN uses the *batch's* mean/variance; at **eval** it uses the **running averages** collected during training (`model.eval()` flips this). That switch is the source of most BN bugs.

<div class="warning">

**BN struggles** with small batches (`< 32`, noisy stats), variable-length sequences (inconsistent batch statistics), and streaming/online data. Every one of these is why **LayerNorm** was invented.

</div>

---

# LayerNorm & RMSNorm — the sequence-model normalizers

<div class="columns">
<div>

**LayerNorm** — normalize across a sample's own features, so it never touches the batch axis:
$$\hat x_i=\frac{x_i-\mu_{\text{sample}}}{\sqrt{\sigma^2_{\text{sample}}+\epsilon}}$$
Independent of batch size and sequence length.

</div>
<div>

**RMSNorm** — drop the mean-subtraction, keep only the scale:
$$\hat x_i=\frac{x_i}{\sqrt{\text{mean}(x^2)+\epsilon}}\cdot\gamma$$
~15–30% cheaper, no loss of quality.

</div>
</div>

<div class="keypoint">

Every modern Transformer (BERT, GPT, Llama, Claude) uses **LayerNorm** or its cheaper cousin **RMSNorm** — *never* BatchNorm. Llama, Mistral, PaLM standardized on RMSNorm from 2023 on. (`nn.LayerNorm(d)`, `nn.RMSNorm(d)`.)

</div>

---

# Pre-norm vs post-norm — where to place it in a deep net

![w:760px](figures/lec06/svg/pre_vs_post_norm.svg)

<div class="insight">

**Highway analogy.** The residual skip is a multi-lane **highway** carrying the gradient from end to start; the sub-layer is a winding **side road**. **Post-norm** ($\text{LN}(x+\text{Sub}(x))$) puts a toll booth on the highway *after* the merge — all gradient traffic is throttled, so deep nets need warmup. **Pre-norm** ($x+\text{Sub}(\text{LN}(x))$) moves the booth onto the side road — the highway (a clean $+1$ gradient path) stays open, and depth trains stably.

</div>

**Building a Transformer in 2026? Use pre-norm** — the default since GPT-2.

---

# Which norm for which model

| Architecture | Normalization | Why |
|--------------|---------------|-----|
| CNN for images | **BatchNorm** | large batches, fixed shapes |
| Small-batch CNN (`<32`) | **GroupNorm** | batch-independent |
| Transformer | **LayerNorm** (pre-norm) | batch/seq-length independent |
| Modern LLM (Llama, Mistral) | **RMSNorm** (pre-norm) | cheaper, same quality |
| RNN / LSTM | **LayerNorm** across features | same reason as Transformer |

<div class="keypoint">

**Two-sentence rule.** Fixed-size dense data with big batches → **BatchNorm**. Anything else → **LayerNorm** (or **RMSNorm** if you want cheap).

</div>

<div class="notebook">

**🎛 Interactive · dropout & normalization** — [dropout-playground](https://nipunbatra.github.io/interactive-articles/dropout-playground/) (flip train/eval, slide $p$) · [norm-comparison](https://nipunbatra.github.io/interactive-articles/norm-comparison/) (see exactly which slice BN/LN/RMSNorm average over). *(Interactive Lab · `~/git/interactive`)*

</div>

---

<!-- _class: section-divider -->

## Part 5 · Double descent — why huge models still generalize

---

# The U-curve has a second act

**Q.** Modern nets have $10^6$–$10^{12}$ parameters for datasets of $\sim10^6$ examples. The classical U-curve says that's catastrophic-overfit territory. Why aren't they?

![w:820px](figures/lec06/svg/double_descent.svg)

<div class="keypoint">

Test error follows the classic U-curve, **spikes** right where the model can *just barely* interpolate the data (params $\approx$ data), then — surprisingly — **falls again** as you keep growing. A model with far more parameters than data can still generalize.

</div>

---

# Double descent — what to take away

1. **Underparameterized** (params $\ll$ data): the classic U-curve, as ES 335 taught.
2. **Interpolation threshold** (params $\approx$ data): test error **spikes** — the model contorts to fit every point, noise included.
3. **Overparameterized** (params $\gg$ data): test error **drops again** — among the many zero-train-error solutions, SGD's implicit bias finds smooth ones.

<div class="insight">

**This is not "bigger is always better."** The spike is sharpest with label noise and *no* regularization — and explicit regularization (today's whole toolkit) can wash it out. ResNet-50 (25M params) and LLMs ($10^{11}$) both generalize past the threshold — **with** the full regularization stack, never without it.

</div>

<div class="notebook">

**🎛 Interactive · double descent** — sweep model capacity and watch the second descent appear. *([double-descent](https://nipunbatra.github.io/interactive-articles/double-descent/) · Interactive Lab)*

</div>

---

<!-- _class: summary-slide -->

# One sentence to remember

<div class="keypoint">

**Regularization is a prior on the solution — written on the weights (L2/L1), stamped into the data (augment, Mixup, smoothing), or wired into the architecture (dropout, normalization) — and the best ones never look like a penalty.**

</div>

- **Classical** — L2 $=$ Gaussian prior $=$ weight decay; L1 $=$ Laplace prior $=$ sparsity (rare in DL). Ridge's $+\lambda I$ both shrinks weights and cures OLS's singularity.
- **Data** — augmentation teaches invariances; Mixup & label smoothing calibrate.
- **Architecture** — dropout (inverted $1/p$ scaling) trains an implicit ensemble; BatchNorm/LayerNorm/RMSNorm reshape the landscape; **pre-norm** for depth.
- **Double descent** — past the interpolation threshold, more parameters can help — *with* regularization.

**Next (L8):** the architecture that is itself a regularizer — convolution, and why its baked-in assumptions are worth orders of magnitude fewer parameters.

---

<!-- _class: compact -->

# Sources & credits

<div class="paper">

This lecture reuses and adapts material from the instructor's own courses (IIT Gandhinagar) and standard references:

- **Ridge & lasso** — Lagrangian, four-step closed form, L1-vs-L2 geometry, shrinkage, soft-thresholding — *Machine Learning (ES 335)*, N. Batra · `github.com/nipunbatra/ml-teaching` (`notebooks/ridge.ipynb`, `notebooks/lasso-regression.ipynb`)
- **L2 $=$ Gaussian prior, L1 $=$ Laplace prior (MAP)** — *ES 667 · Lecture 1*, this course
- **Dropout, augmentation, Mixup, label smoothing, normalization, double descent** — ES 667 regularization materials; A. Ng, *Deep Learning Specialization* Course 2 (Weeks 1 & 3)
- **BatchNorm** — Ioffe & Szegedy, *Batch Normalization* (2015); the ICS re-analysis — Santurkar et al. (2018)

Figures reused from the ES 667 figure library (`figures/lec06/svg/`). All source courses © N. Batra & teaching staff.

</div>

---

<!-- _class: section-divider -->

## ⭐⭐⭐ Appendix · optional depth

*Skip on a first pass — for the curious.*

---

<!-- _class: math-heavy -->

# ⭐⭐⭐ Optional · L2 is exactly a Gaussian prior (MAP)

Put an independent Gaussian prior on every weight, $\theta_j\sim\mathcal N(0,\sigma_p^2)$, and maximize the posterior $p(\theta\mid\text{data})\propto p(\text{data}\mid\theta)\,p(\theta)$. Take $-\log$:

$$\hat\theta_{\text{MAP}}=\arg\min_\theta\Big[\underbrace{-\log p(\text{data}\mid\theta)}_{\text{data loss}}\;+\;\underbrace{\tfrac{1}{2\sigma_p^2}\lVert\theta\rVert_2^2}_{-\log p(\theta)}\Big]$$

because $-\log p(\theta)=\sum_j\frac{\theta_j^2}{2\sigma_p^2}+\text{const}$. Matching to $\lambda\lVert\theta\rVert_2^2$ gives $\;\boxed{\lambda=\dfrac{1}{2\sigma_p^2}}$: a **tighter prior** (smaller $\sigma_p^2$) $=$ **stronger regularization** (larger $\lambda$). Swap the Gaussian for a Laplace prior and the same computation yields $\lambda\lVert\theta\rVert_1$ — L1. *(Full version: L1 Part 4; ES 335 ridge/lasso.)*

---

<!-- _class: math-heavy -->

# ⭐⭐⭐ Optional · why pre-norm keeps a clean gradient highway

Residual block, two placements of the norm:

$$\text{post-norm: } \text{out}=\text{LN}\big(x+\text{Sub}(x)\big) \qquad \text{pre-norm: } \text{out}=x+\text{Sub}\big(\text{LN}(x)\big)$$

Differentiate the skip path back to $x$:

$$\frac{\partial\,\text{out}_{\text{pre}}}{\partial x}=\underbrace{I}_{\text{clean }+1\text{ highway}}+\frac{\partial\,\text{Sub}(\text{LN}(x))}{\partial x} \qquad\text{vs}\qquad \frac{\partial\,\text{out}_{\text{post}}}{\partial x}=\underbrace{\text{LN}'\cdot\big(I+\tfrac{\partial\text{Sub}}{\partial x}\big)}_{\text{skip routed *through* LN}}$$

In pre-norm the skip contributes an exact $+I$ term at *every* layer, so the gradient reaches layer 1 undiminished through an $L$-deep stack. In post-norm that same gradient is multiplied by $L$ Jacobians of LN — it can vanish or explode with depth, which is why post-norm Transformers need learning-rate warmup and careful init. *(This is the algebra behind the "toll booth on the highway" analogy.)*
