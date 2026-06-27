---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# Information Theory for ML

## Surprise · Entropy · Coding · Cross-Entropy · KL

### Lecture 0C · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar · Aug 2026*

---

<!-- _class: section-divider -->

### READING · Cover & Thomas Ch 2 · Shannon (1948)

# Lecture 0C

A standalone, intuition-first tour of the information theory that every loss in this course is built from. Watch (optional, excellent) · Serrano *"KL divergence"* and Jia-Bin Huang *"Fantastic KL divergence."*

---

# Why a whole lecture on this

You have been minimizing **cross-entropy** since ES 335. But ·

<div class="math-box">

1. What *is* a "bit", really?
2. Why is the classification loss called cross-**entropy**?
3. What does that number — "loss = 2.3" — physically *mean*?
4. Why is **KL divergence** the heartbeat of VAEs, diffusion, RLHF, distillation?

</div>

By the end you will explain entropy, optimal codes, cross-entropy, and KL from **one** picture — surprise — and see that *training a classifier is data compression.*

---

# Bridge from L00 / L00B

| You already saw | Today we open it up |
|---|---|
| L00 · every loss is an **NLL**, $-\log p$ | what $-\log p$ *measures* — surprise, in bits |
| L00B · KL appeared as "extra surprise" | the full story · entropy → cross-entropy → KL |
| CE is "the classification loss" | CE = entropy + KL · and for one-hot labels, **CE = KL = NLL** |

<div class="keypoint">

One idea — **surprise = $-\log p$** — generates entropy, coding, cross-entropy, and KL. Everything else is averaging it.

</div>

---

# Learning outcomes

By the end you will be able to ·

<div class="math-box">

1. Define **information content** (surprise) and explain why it is $-\log p$.
2. Define **entropy** as average surprise / **optimal code length**.
3. Build an **optimal code** and explain "frequent → short."
4. Define **cross-entropy** as coding with the *wrong* distribution.
5. Define **KL divergence** as the **extra bits** wasted, and use it.
6. Show **classification loss = KL to the true label = NLL**, and that **a better model is a better compressor** (perplexity).

</div>

---

# ▶ Interactives for L00C

<div class="paper">

- [info-theory](https://nipunbatra.github.io/interactive-articles/info-theory/) · entropy + KL with sliders.
- [softmax-temperature](https://nipunbatra.github.io/interactive-articles/softmax-temperature/) · watch output entropy change.

</div>

---

<!-- _class: section-divider -->

### PART 1

# Surprise & bits

The atom of information

---

# Information = surprise

<div class="keypoint">

Information is how much a message *surprises* you. A rare event tells you a lot; an expected event tells you almost nothing.

</div>

- "The sun rose today." — expected · ~0 information.
- "It snowed in Gandhinagar." — shocking · huge information.

The **less probable** an outcome, the **more information** observing it carries. We want a number $I(x)$ that grows as $p(x)$ shrinks.

---

# Three requirements pin down the formula

We want a surprise function $I(p)$ with three natural properties ·

<div class="math-box">

1. $I$ depends only on the **probability** $p$ (not what the outcome is).
2. $I$ **decreases** as $p$ grows · certain events ($p=1$) carry $I=0$.
3. Independent events **add** · $I(p_1 p_2) = I(p_1) + I(p_2)$.

</div>

The only function turning **products into sums** while decreasing in $p$ is the logarithm ·

$$\boxed{\,I(x) = -\log p(x)\,}$$

---

# Why the log · additivity

Two independent coin flips · the joint probability **multiplies**, but the surprise should **add** (two flips tell you twice as much).

<div class="math-box">

$$I(p_1 p_2) = -\log(p_1 p_2) = -\log p_1 - \log p_2 = I(p_1) + I(p_2)\;\checkmark$$

</div>

The logarithm is the *unique* tool that converts "multiply probabilities" into "add information." That is the whole reason logs are everywhere in ML losses.

---

# Surprise, in pictures

![w:760px](figures/lec00c/svg/info_content.svg)

Certain ($p\!=\!1$) → 0 bits. Fair coin ($p\!=\!\tfrac12$) → 1 bit. One-in-a-thousand → ~10 bits. The curve is just $-\log_2 p$.

---

# Bits and nats · one yes/no question

The **base of the log** is just the unit ·

<div class="math-box">

- $\log_2$ → **bits** · one bit = the answer to one yes/no question.
- $\log_e$ (natural) → **nats** · what PyTorch's losses use internally.

$$1 \text{ nat} = \tfrac{1}{\ln 2} \approx 1.44 \text{ bits}$$

</div>

A fair coin: $-\log_2 \tfrac12 = 1$ bit — one yes/no question ("heads?") pins it down.

<div class="keypoint">

**3Blue1Brown's picture** · one bit = information that **halves** the possibilities. $N$ options, $I$ bits → $N/2^I$ left. Wordle has ~13 000 words, so winning needs $\log_2 13000 \approx 13.6$ bits; a strong first guess already buys ~5.8 of them.

</div>

---

# Surprise · worked examples

<div class="math-box">

| Event | $p$ | $I = -\log_2 p$ |
|:-:|:-:|:-:|
| Fair coin → heads | $1/2$ | $1$ bit |
| Roll a 6 on a die | $1/6$ | $2.58$ bits |
| 1-in-1024 lottery | $1/1024$ | $10$ bits |
| Sun rises tomorrow | $\approx 1$ | $\approx 0$ bits |
| A specific token from a 50k vocab | $1/50000$ | $15.6$ bits |

</div>

Large $I$ = "I learned a lot." Small $I$ = "I already knew that." This per-event surprise is **exactly the per-sample loss** in disguise.

---

<!-- _class: section-divider -->

### PART 2

# Entropy

Average surprise — and the length of the best possible code

---

# Entropy = expected surprise

The **expectation** $\mathbb{E}_{Y\sim p}[\,\cdot\,]$ means "average the quantity, weighting each outcome $y$ by how often $p$ produces it" ·

<div class="math-box">

$$H(p) = \underbrace{\mathbb{E}_{Y \sim p}}_{\text{average over draws from } p}\big[\,\underbrace{-\log p(Y)}_{\text{surprise}}\,\big] = -\sum_y p(y)\,\log p(y)$$

</div>

The subscript $Y\sim p$ says *which* distribution we average over — it will matter the moment two distributions appear (cross-entropy, soon).

- High entropy · unpredictable draws (fair die) · Low entropy · predictable (loaded die) · Zero · deterministic.

---

# Entropy = the long-run average surprise

An expectation is just the **sample average in the limit** (law of large numbers) · draw many days, average their surprise, and it settles onto $H(P)$ ·

![w:740px](figures/lec00c/svg/entropy_samples.svg)

This is why your **training loss** — the average $-\log p$ over the dataset — *is* an estimate of a (cross-)entropy. The dataset is the samples.

---

# The coding view · send the weather

Why is entropy measured in **bits**? Because it is the length of the best code.

<div class="keypoint">

You must telegraph each day's weather as 0s and 1s, using as few bits as possible on average. Frequent weather should get **short** codes, rare weather **long** codes.

</div>

True weather · Sunny $\tfrac12$ · Cloudy $\tfrac14$ · Rainy $\tfrac14$. What is the best code?

---

# The optimal code · a picture

![w:780px](figures/lec00c/svg/coding_tree.svg)

Give the frequent symbol the 1-bit code, the rare ones 2 bits. The average length is **1.5 bits/day** — and that is exactly $H(P)$.

---

# Optimal code length = −log p

**Why $-\log_2 p$?** A symbol with probability $p$ is one of about $1/p$ equally-likely cases, so it takes $\log_2(1/p) = -\log_2 p$ yes/no questions (halvings) to single out. *Halve a symbol's probability → add exactly one bit to its code.*

<div class="math-box">

| Weather | $p$ | optimal length $-\log_2 p$ | code |
|:-:|:-:|:-:|:-:|
| Sunny | $1/2$ | $1$ bit | `0` |
| Cloudy | $1/4$ | $2$ bits | `10` |
| Rainy | $1/4$ | $2$ bits | `11` |

Average $= \tfrac12(1) + \tfrac14(2) + \tfrac14(2) = 1.5$ bits $= H(P)$.

</div>

**Entropy is the average length of the optimal code** — the fewest bits/symbol achievable for data from $P$.

---

# What if we just used a simple code?

A lazy fixed-length code gives every symbol the same number of bits ·

![w:660px](figures/lec00c/svg/naive_vs_optimal_code.svg)

Ignoring that Sunny is common costs **0.5 bits/day**. Matching code length to probability ($-\log p$) is what reaches the entropy floor — and it is exactly what a good *model* does with its predictions.

---

# Entropy · worked numerics

<div class="math-box">

| Distribution | $H$ (bits) | why |
|:-:|:-:|:-:|
| Fair coin | $1$ | $-2(\tfrac12\log_2\tfrac12)$ |
| Biased coin $p=0.99$ | $0.08$ | almost no surprise |
| Uniform over 8 | $3$ | $\log_2 8$ |
| One-hot (certain) | $0$ | nothing to send |

</div>

Entropy **peaks for the uniform** distribution ($\log_2 K$ for $K$ outcomes) and is **zero** for a deterministic one.

---

# Entropy of a coin · in pictures

![w:600px](figures/lec00/svg/entropy_bernoulli.svg)

$H(p) = -p\log_2 p - (1-p)\log_2(1-p)$ — symmetric, **peaks at 1 bit** for a fair coin, falls to 0 as the coin becomes certain.

---

# Entropy = uncertainty · the ML reading

<div class="keypoint">

A **confident** classifier has *low-entropy* output (mass on one class). A **confused** one has *high-entropy* output (spread out).

</div>

This is exactly what **temperature** controls in an LLM · high $T$ flattens the distribution (raises entropy → more "creative"), low $T$ sharpens it (lowers entropy → more deterministic). Entropy is the dial between certainty and diversity.

---

<!-- _class: section-divider -->

### PART 3

# Cross-entropy

What happens when you use the *wrong* distribution

---

# Coding with the wrong distribution

In ML you **don't know** the true $P$. You have a *model* $Q$ and build your code for $Q$ — then reality sends data from $P$.

<div class="keypoint">

**Cross-entropy** $H(P,Q)$ = the average bits you spend when your code is optimized for $Q$ but the data actually comes from $P$.

$$H(P,Q) = -\sum_y p(y)\,\log q(y)$$

</div>

You pay $-\log q(y)$ per symbol (your code's length), but symbols *arrive* with frequency $p(y)$. Here $p$ = the **true** real-world probability, $q$ = your **model's predicted** probability for the same event.

---

# Cross-entropy · the weather, continued

Your forecast is **wrong** · $Q = (\tfrac14,\tfrac14,\tfrac12)$. You build the code for $Q$, then use it on the *real* weather $P=(\tfrac12,\tfrac14,\tfrac14)$ ·

![w:760px](figures/lec00/svg/kl_weather.svg)

$$H(P,Q) = \tfrac12(2) + \tfrac14(2) + \tfrac14(1) = 1.75 \text{ bits} \;>\; 1.5 = H(P)$$

---

# You can never beat the optimal code

<div class="math-box">

$$H(P,Q) \;\ge\; H(P), \qquad \text{equality iff } Q = P.$$

</div>

Coding for the wrong distribution **always** costs more (or equal) bits. The true distribution's own code is unbeatable — that is why $H(P)$ is the floor and cross-entropy sits above it.

**ML translation** · the lowest achievable loss is $H(P)$, the *irreducible* uncertainty in the data. Your model can only close the gap to it, never go below.

---

# Cross-entropy · worked numeric

True $P = (0.7, 0.3)$, model $Q = (0.5, 0.5)$ ·

<div class="math-box">

$$H(P,Q) = -\big[0.7\log_2 0.5 + 0.3\log_2 0.5\big] = -\log_2 0.5 = 1.0 \text{ bit}$$

Compare the floor · $H(P) = -0.7\log_2 0.7 - 0.3\log_2 0.3 = 0.88$ bits.

</div>

The wrong model costs $1.0$ bit/symbol; the best possible is $0.88$. The **gap is $0.12$ bits** — wasted. That gap has a name.

---

# Cross-entropy · the ML reading (cat vs dog)

In classification the label is **certain** — it really is a cat · $P=(\text{cat}{=}1,\,\text{dog}{=}0)$. The model $Q$ is your classifier's softmax. Cross-entropy = the model's **surprise at the truth** ·

<div class="math-box">

| model $Q(\text{cat})$ | surprise $-\log_2 Q(\text{cat})$ | verdict |
|:-:|:-:|:-:|
| $0.25$ (bad) | $2.0$ bits | very surprised — big loss |
| $0.99$ (good) | $0.014$ bits | barely surprised — tiny loss |

</div>

This **is** the logistic-regression / classification loss from L00 · $-\log Q(\text{true class})$. Training = nudge the weights until the model is *least surprised* by the real labels.

---

<!-- _class: section-divider -->

### PART 4

# KL divergence

The gap · the extra bits you waste

---

# KL = the extra bits

<div class="keypoint">

**KL divergence is the waste** · cross-entropy minus entropy — the extra bits/symbol you pay for using the wrong distribution's code.

$$\text{KL}(P\,\Vert\,Q) = H(P,Q) - H(P)$$

</div>

![w:680px](figures/lec00c/svg/cross_entropy_extrabits.svg)

---

# The dice game · score a model by surprise

Serrano's framing · a sequence of rolls comes from the **true die** $P$. Score each candidate die by the surprise it assigns the data ·

![w:880px](figures/lec00c/svg/dice_scoring.svg)

Best possible score = $H(P)$ — the true die's own surprise; no model can beat it.

**The excess over $H(P)$ is the KL** · $Q_1$ (close) pays little, $Q_2$ (far) pays a lot.

---

# KL · the formula and its properties

Subtract the two averages and the entropy cancels into a single sum ·

<div class="math-box">

$$\text{KL}(P\,\Vert\,Q) = \sum_y p(y)\log\frac{p(y)}{q(y)} \;=\; \mathbb{E}_{P}\!\left[\log\frac{p}{q}\right]$$

</div>

- $\text{KL} \ge 0$ always (Gibbs' inequality), and $=0$ **iff** $P=Q$.
- **Not symmetric** · $\text{KL}(P\Vert Q) \ne \text{KL}(Q\Vert P)$ — it is a directed gap, not a distance.
- Units follow the log base · bits ($\log_2$) or nats ($\ln$).

---

# KL · worked numeric · the two routes agree

$P=(0.7,0.3)$, $Q=(0.5,0.5)$ — compute KL **two ways** ·

<div class="math-box">

**Route 1 · waste = cross-entropy − entropy**
$$\text{KL}(P\Vert Q) = H(P,Q) - H(P) = 1.00 - 0.88 = \mathbf{0.12}\text{ bits}$$

**Route 2 · the direct formula**
$$\text{KL}(P\Vert Q) = 0.7\log_2\tfrac{0.7}{0.5} + 0.3\log_2\tfrac{0.3}{0.5} = 0.7(0.485) + 0.3(-0.737) = \mathbf{0.12}\text{ bits}$$

</div>

Same number. "Extra bits over the optimal code" and "average log-ratio" are *the same quantity* — that is why KL is **the** measure of model error.

---

# KL is asymmetric · two failure modes

Which direction you minimize is a **modelling choice** with different behaviour ·

<div class="keypoint">

- **Forward** $\text{KL}(P\Vert Q)$ — averaged over the truth → *mode-covering* · $Q$ must put mass everywhere $P$ does (blurry, safe). **This is what MLE does.**
- **Reverse** $\text{KL}(Q\Vert P)$ — averaged over the model → *mode-seeking* · $Q$ picks one mode (sharp, can collapse).

</div>

---

# Forward vs reverse · the picture

![w:820px](figures/lec00/svg/forward_reverse_kl.svg)

Same bimodal target. Forward KL spreads across both modes (why **VAE samples blur**); reverse KL locks onto one (why **GANs mode-collapse**). Knowing your loss's KL direction predicts its failure mode.

*These ideas carry over unchanged to the continuous distributions in VAEs/GANs (L19–L20) — sums become integrals, the mode-covering vs mode-seeking behaviour is identical.*

---

<!-- _class: section-divider -->

### PART 5

# The ML connection

Why training a classifier *is* compression

---

# The one-paragraph version

<div class="keypoint">

**Reality** produces data from some true $P$. Your **model** $Q$ assigns a probability to everything it sees. Each example surprises the model by $-\log Q$. Training nudges $Q$ until it is **as unsurprised as possible by the real data** — i.e. until $Q$ matches $P$.

</div>

That single sentence is cross-entropy loss, MLE, KL minimization, and compression — the next four slides are just that idea, made precise.

---

# The master identity

Add and subtract $\log p$ inside the cross-entropy average ·

<div class="math-box">

$$\boxed{\,H(P,Q) = \underbrace{H(P)}_{\text{irreducible}} + \underbrace{\text{KL}(P\,\Vert\,Q)}_{\text{what your model controls}}\,}$$

</div>

Cross-entropy = the data's own uncertainty **plus** how far your model is from the truth. Minimizing cross-entropy over $Q$ minimizes the **only** term you can move · the KL.

---

# Classification · CE = KL = NLL

The true label is **one-hot** · all mass on the correct class $c$. A one-hot has **zero entropy** ·

<div class="math-box">

$$H(P,Q) = \underbrace{H(P)}_{=\,0} + \text{KL}(P\,\Vert\,Q) = \text{KL}(P\,\Vert\,Q) = -\log q_c$$

</div>

So the classification **cross-entropy loss IS** the KL to the true label, which **is** the NLL of the correct class. The three names from L00, L00B and today are *the same number*.

---

# Classification loss · worked over 5 examples

A 3-class problem (cat / dog / fish). Each label is one-hot, so its entropy is 0 and **per-example loss = surprise at the truth = $-\log_2 Q(\text{true})$** ·

<div class="math-box">

| true | model $Q$ (cat, dog, fish) | $-\log_2 Q(\text{true})$ |
|:-:|:-:|:-:|
| cat | $(0.7,\ 0.2,\ 0.1)$ | $0.51$ bits |
| dog | $(0.3,\ 0.4,\ 0.3)$ | $1.32$ bits |
| cat | $(0.6,\ 0.3,\ 0.1)$ | $0.74$ bits |
| fish | $(0.1,\ 0.1,\ 0.8)$ | $0.32$ bits |
| dog | $(0.1,\ 0.8,\ 0.1)$ | $0.32$ bits |

**Training loss = average** $= \dfrac{0.51+1.32+0.74+0.32+0.32}{5} = \mathbf{0.64}$ bits.

</div>

That single number is the cross-entropy loss your optimizer minimizes — and it is the average KL from each one-hot truth to the model. The confident-wrong row (dog at $Q=0.4$) dominates.

---

# MLE = minimizing KL to the data

Treat the training set as the **empirical distribution** $\hat p_{\text{data}}$ (a histogram). Maximum likelihood pulls the model onto it ·

<div class="math-box">

$$\arg\max_\theta \tfrac1N\sum_i \log p_\theta(y_i) = \arg\min_\theta\, \text{KL}\big(\hat p_{\text{data}}\,\Vert\,p_\theta\big)$$

</div>

![w:720px](figures/lec00c/svg/mle_kl_fit.svg)

**Make the model as KL-close as possible to the data.** Every classifier you've trained with cross-entropy has been silently doing this.

---

# A better model is a better compressor

$-\log_2 q(\text{data})$ is the **number of bits** to encode the data with model $Q$. Lower loss = fewer bits = better compression ·

<div class="keypoint">

**3Blue1Brown · "compression is intelligence."** A smart model is rarely surprised — it gives the true next token high probability, so $-\log_2 q \approx 0$, and the data shrinks to a tiny file. Being a good predictor and being a good compressor are *the same skill*.

</div>

So "a good language model = a good compressor of text" is not a metaphor · the cross-entropy loss, in bits, **is** the compressed size per token — which is why frontier reports quote **bits-per-byte**.

---

# Perplexity · the loss you can feel

LLM papers report **perplexity**, not raw loss · the *effective number of equally-likely choices* per token ·

<div class="math-box">

$$\text{perplexity} = e^{\,\text{CE}} \quad(\text{nats}) \;=\; 2^{\,\text{CE}}\;(\text{bits})$$

</div>

- CE $= \ln 10 \approx 2.30$ nats → perplexity $= 10$ · "as unsure as guessing uniformly among 10 words."
- A model improving from perplexity $20 \to 8$ went from "1-in-20 confused" to "1-in-8."

It is just entropy, exponentiated back into "number of choices."

---

# Two ML tricks, read through entropy

<div class="math-box">

- **Label smoothing** · replace the one-hot target $(1,0,\dots)$ with $(0.9, \tfrac{0.1}{K-1}, \dots)$. You raise the target's entropy → the model is penalized for being *over-confident*. CE now has a floor $>0$.
- **Temperature** $T$ · divide logits by $T$ before softmax → directly scales output entropy. $T\!\uparrow$ raises entropy (diverse), $T\!\downarrow$ lowers it (greedy).

</div>

Both are just **moving entropy around** — exactly the quantity this lecture defined.

---

# KL is everywhere from here on

<div class="math-box">

| Model | What it minimizes | Lecture |
|:-:|:-:|:-:|
| Any classifier | $\text{KL}(\hat p_{\text{data}}\Vert p_\theta)$ = CE | L01+ |
| **VAE** | recon NLL $+\ \text{KL}(q_\phi(z\vert x)\,\Vert\,p(z))$ | L19 |
| **Diffusion** | per-step $\text{KL}$ of reverse vs forward | L21 |
| **RLHF / DPO** | reward $-\ \beta\,\text{KL}(\pi_\theta\Vert\pi_{\text{ref}})$ | L16 |
| **Distillation** | $\text{KL}(p_{\text{teacher}}\Vert p_{\text{student}})$ | L23 |

</div>

Every generative or alignment loss ahead is "fit the data **and** don't drift in KL." You now own the whole vocabulary.

---

# Putting L00C together · the one chain

<div class="math-box">

1. **Surprise** of an outcome · $I = -\log p$.
2. **Entropy** · average surprise = optimal code length = $H(P)$.
3. **Cross-entropy** · coding with the wrong $Q$ · $H(P,Q) \ge H(P)$.
4. **KL** · the waste · $H(P,Q) - H(P) \ge 0$.
5. **ML** · classifier loss = CE = KL to one-hot truth = NLL; MLE = min KL to data; loss-in-bits = compressed size.

</div>

One picture — surprise — and the entire loss landscape of deep learning falls out.

---

# Practice problems

<div class="math-box">

**P1.** A 4-symbol source has $P=(\tfrac12,\tfrac14,\tfrac18,\tfrac18)$. Compute $H(P)$ in bits and give an optimal prefix code.

**P2.** True $P=(0.5,0.5)$, model $Q=(0.9,0.1)$. Compute $H(P,Q)$ and $\text{KL}(P\Vert Q)$ in bits.

**P3.** Show that for a one-hot target, cross-entropy reduces to $-\log q_c$.

**P4.** A language model reports CE $=1.5$ nats/token. What is its perplexity? Interpret it.

**P5.** Explain in one line why $\text{KL}(P\Vert Q)\ne\text{KL}(Q\Vert P)$ using the dice-game intuition.

</div>

---

# Let's do one · P2 worked

True $P=(0.5,0.5)$, model $Q=(0.9,0.1)$ ·

<div class="math-box">

**Cross-entropy** · $H(P,Q) = -[0.5\log_2 0.9 + 0.5\log_2 0.1]$
$= -[0.5(-0.152) + 0.5(-3.322)] = 0.076 + 1.661 = \mathbf{1.74}$ bits.

**Entropy** · $H(P) = 1$ bit (fair coin).

**KL** · $\text{KL}(P\Vert Q) = H(P,Q) - H(P) = 1.74 - 1.0 = \mathbf{0.74}$ bits.

</div>

A confidently-wrong model ($Q$ sure of the wrong side) wastes $0.74$ bits/symbol — confident-and-wrong is expensive, exactly as cross-entropy intends.

---

# The one-sentence takeaway

<div class="insight">

**Cross-entropy loss is the number of extra bits your model wastes describing reality — minimize it and you are compressing the data toward the truth.**

</div>

*Next (L01) · we stop reviewing and start building — why deep learning, and what representation learning buys us.*
