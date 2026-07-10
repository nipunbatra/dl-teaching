---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# Expressivity: Universal Approximation & Going Deep

## Lecture 3 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar*

---

# The whole lecture, in one idea

A neural network is a **function builder**. Give one hidden layer enough neurons and it can draw *any* curve you want — but the honest price of "any" is sometimes **exponential** width.

<div class="keypoint">

**One wide layer is universal; depth is what makes it affordable.**
A single nonlinearity is what lets a network bend space at all. Stack those bends and you get *features of features* — the same function a shallow net would need exponentially many neurons to match.

</div>

In L2 you built the neuron: affine map $\to$ nonlinearity. Today we ask what a *pile* of them can represent — and end on the catch that sets up the rest of the course:

$$\underbrace{\text{nonlinearity}}_{\text{why any curve is possible}} \rightarrow \underbrace{\text{UAT}}_{\text{one layer is enough}} \rightarrow \underbrace{\text{width blows up}}_{\text{the catch}} \rightarrow \underbrace{\text{depth}}_{\text{the fix}} \rightarrow \underbrace{\text{but is it trainable?}}_{\text{L4}}$$

---

# How today connects to the rest of the course

<div class="columns">
<div>

**You'll reuse today's ideas in:**
- **L4** — backprop, which turns "weights *exist*" into "weights we can *find*"
- **L5** — vanishing/exploding gradients, the reason deep nets were hard until ~2015
- **L7–L8** — CNNs are the feature-of-features hierarchy made concrete
- **L13** — the Transformer's residual stream is depth made trainable

</div>
<div>

**Borrowed from your own courses** (build on, not repeat):
- basis functions $\phi(x)=[1,x,x^2,\dots]$ you picked *by hand* — **ML (ES 335)**
- linear vs logistic vs neural — **L2**

</div>
</div>

<div class="insight">

UAT itself is **beyond Ng's specialization** — we keep it intuitive (bumps and LEGO), not theorem-heavy. The formal statements wait for the ⭐⭐⭐ appendix.

</div>

---

<!-- _class: section-divider -->

## Part 1 · Why a nonlinearity is non-negotiable

---

# A neuron = an affine map, then a squish

One neuron takes a weighted sum $\mathbf w^\top\mathbf x + b$ (a straight-line score) and passes it through a nonlinearity $\sigma$:

![w:640px](figures/lec01/svg/linear_to_neuron.svg)

$$h = \sigma(\mathbf w^\top\mathbf x + b)$$

The affine part can only draw **hyperplanes**. Everything interesting a network does lives in that $\sigma$. So the first question is: what happens if we *skip* it?

---

# Stack linear on linear, and you still have a line

Two "linear layers" with nothing in between:

<div class="math-box">

$$\mathbf h = W_1\mathbf x + \mathbf b_1,\qquad \mathbf y = W_2\mathbf h + \mathbf b_2 = \underbrace{W_2 W_1}_{W'}\,\mathbf x + \underbrace{(W_2\mathbf b_1+\mathbf b_2)}_{\mathbf b'} = W'\mathbf x + \mathbf b'$$

</div>

![w:560px](figures/lec01/svg/stacked_linear_collapses.svg)

<div class="insight">

**Any depth of purely linear layers collapses to one linear layer.** Ten layers, a hundred layers — still a single hyperplane. Depth buys you *nothing* without a nonlinearity between the maps. That bend is the whole point.

</div>

---

# Two linear layers, in numbers

Pick concrete matrices — no nonlinearity between them:

<div class="math-box">

$$W_1=\begin{bmatrix}1&2\\0&1\end{bmatrix},\quad W_2=\begin{bmatrix}1&-1\\2&0\end{bmatrix} \;\Rightarrow\; W_2W_1=\begin{bmatrix}1&1\\2&4\end{bmatrix}=W'$$

$$\mathbf y = W_2(W_1\mathbf x) = (W_2W_1)\,\mathbf x = W'\mathbf x$$

</div>

<div class="insight">

The two layers multiplied out to **one** $2\times2$ matrix — and that is true for *any* numbers, at *any* depth, because a product of matrices is a matrix. The extra layer added zero expressive power. Drop a $\sigma$ between them and $W_2\,\sigma(W_1\mathbf x)$ can no longer be flattened — depth suddenly means something.

</div>

---

# The data that a straight line can't cut

Real targets aren't linearly separable. No single hyperplane splits the classes on the right:

![w:820px](figures/lec01/svg/linear_vs_nonlinear_data.svg)

A linear model can only ask "which side of the line?" — so it is stuck on the left panel. To carve the right panel we need to **bend** the boundary, and bending needs $\sigma$.

---

# XOR: the smallest function no single neuron can do

Four points, two classes: $(0,0),(1,1)\to 0$ and $(0,1),(1,0)\to 1$. No straight line separates them.

![w:560px](figures/lec01/svg/xor_decision.svg)

<div class="insight">

XOR is the historical embarrassment (Minsky & Papert, 1969) that stalled neural nets for a decade. **One hidden layer with a nonlinearity solves it instantly** — it can fold the plane so the two "1" corners land on the same side. We'll make you prove the linear-layer impossibility in Practice Problem 3.

</div>

---

# One neuron places one kink

A single ReLU neuron bends the input line at exactly one place — where its pre-activation crosses zero:

<div class="math-box">

$$h=\text{relu}(wx+b)=0 \iff x = -\frac{b}{w}.$$

$w$ sets the **slope** after the kink; $b$ **slides** the kink left or right.

</div>

<div class="insight">

So one neuron is **one hinge** you can put anywhere and tilt to any slope. A hidden layer is a *box of hinges* — and Part 2 shows how a handful of them snap together into any bump. The activation zoo below is just different hinge shapes.

</div>

---

# The nonlinearity zoo: sigmoid, tanh, ReLU

![w:820px](figures/lec01/svg/activation_grid.svg)

| Activation | Range | Gradient | Personality |
|---|---|---|---|
| **Sigmoid** $\frac{1}{1+e^{-z}}$ | $(0,1)$ | $\le 0.25$, saturates | historical; squashes to a probability |
| **Tanh** | $(-1,1)$ | $\le 1$, saturates | zero-centered sigmoid |
| **ReLU** $\max(0,z)$ | $[0,\infty)$ | $0$ or $1$ | cheap, no saturation on the active side |

---

# Why ReLU took over

![w:820px](figures/lec02/svg/relu_family.svg)

<div class="keypoint">

ReLU's gradient is **exactly 1** wherever the unit is active — no shrinking factor to multiply down through the layers. That single property is what let networks go deep. Its kinks are also *exactly* the LEGO bricks we're about to build any curve from.

</div>

The saturating gradient of sigmoid/tanh (that $\le 0.25$ ceiling) is a quiet time bomb — we defuse it at the very end of today.

---

<!-- _class: section-divider -->

## Part 2 · Universal approximation — the intuition

---

# One hidden layer can output *any* curve

The punchline, before any proof: make a single hidden layer wide enough and it can match *any* continuous function to *any* accuracy.

![w:900px](figures/lec02/svg/uat_output_first.svg)

More hidden units $\to$ finer fit. It is the **same one layer** throughout — we only made it wider. That's the Universal Approximation Theorem; the rest of Part 2 is *why*, and *at what cost*.

---

# The LEGO-brick idea, in plain English

<div class="keypoint">

Imagine unlimited LEGO bricks. Can you build a sculpture of *anything* — a car, a house, the Eiffel Tower? **Yes**, if the bricks are small enough you can approximate any shape.

UAT says a one-hidden-layer network does the same for **functions**. Its bricks are simple bumps built from a few neurons. Stack enough bricks, weight them, and you trace any curve.

</div>

The theorem is an existence claim about those bricks — it does not hand you a blueprint. Let's read the claim carefully, then build one brick by hand.

---

# UAT, unpacked term by term

<div class="math-box">

**"For any continuous function $f(\mathbf x)$…"** — the true input$\to$output relation, e.g. `price = f(house features)`.

**"…and any error $\epsilon>0$…"** — however close you demand; set $\epsilon=0.01$ or smaller.

**"…there exists a one-hidden-layer network of $N$ neurons…"** — some finite width works; the theorem does not say how large.

**"…within $\epsilon$ of $f$."** $\quad\Big|\,f(\mathbf x)-\sum_{i=1}^{N}\alpha_i\,\sigma(\mathbf w_i^\top\mathbf x+b_i)\,\Big|<\epsilon$

</div>

A **weighted sum of neuron outputs** — each neuron a LEGO brick. The load-bearing word is *exists*: it promises the weights are out there, not that $N$ is small or that SGD will find them.

---

# Build one bump from ReLU kinks

Each ReLU is a one-sided ramp — a hinge. Add three, with the right signs, and you get a **localized bump**:

![w:820px](figures/lec02/svg/uat_bump_construction.svg)

<div class="math-box">

$$\text{bump}(x) = \text{relu}(x-a) - 2\,\text{relu}(x-b) + \text{relu}(x-c)$$

</div>

Weight each bump by $\alpha_i$, slide it to where you need it, and sum: $\;\hat f(x)=\sum_i \alpha_i\,\text{bump}_i(x)$.

---

# Two ReLUs $\to$ a soft step

Before the bump, the simpler brick. A **difference of two ReLUs** turns an endless ramp into a step that levels off:

<div class="math-box">

$$\text{step}(x;a,b) = \text{relu}(x-a) - \text{relu}(x-b),\qquad a<b$$

Zero before $a$ · rises linearly on $[a,b]$ · then **plateaus** at height $b-a$ forever (for $x>b$ the two slopes cancel, but not the accumulated height).

</div>

Shifted, scaled soft steps stack into any **staircase**. But a step never comes back down — for a localized bump we need one more ReLU.

---

# Three ReLUs $\to$ a triangle bump

Add a third ReLU (with weight $-2$ in the middle) so the ramp comes *back to zero*:

<div class="math-box">

$$f(x) = \text{relu}(x-1) - 2\,\text{relu}(x-2) + \text{relu}(x-3)$$

| $x$ | relu$(x{-}1)$ | $-2\,$relu$(x{-}2)$ | relu$(x{-}3)$ | sum |
|:-:|:-:|:-:|:-:|:-:|
| 0   | 0 | 0 | 0 | **0** |
| 2   | 1 | 0 | 0 | **1** (peak) |
| 2.5 | 1.5 | $-1$ | 0 | **0.5** |
| 4   | 3 | $-4$ | 1 | **0** |

</div>

A perfect triangular bump at $x=2$. Tile the axis with these, weight each by the function's value there, and you approximate any continuous function. **This is UAT, constructively.**

---

# From one brick to the whole curve

You have the brick (a triangle bump). Here is the entire assembly line — four moves and you have *any* curve:

<div class="math-box">

1. **Slice** the input axis into $N$ intervals of width $h$.
2. **Place** one bump on each interval — 3 ReLUs apiece.
3. **Scale** bump $i$ by $\alpha_i=f(x_i)$, the target's height there.
4. **Sum:** $\;\hat f(x)=\sum_{i=1}^{N}\alpha_i\,\text{bump}_i(x)$.

</div>

<div class="insight">

Every step lives in **one hidden layer** — the bumps are its neurons, the $\alpha_i$ are the output weights. Finer slices ($h\downarrow$) buy more neurons and a tighter fit. That is the whole trick: **UAT is just "enough LEGO bricks, each scaled to the target."**

</div>

---

# Worked example: approximate $x^2$ with 4 ReLUs

![w:900px](figures/lec02/svg/uat_relu_bumps.svg)

Four one-sided ramps, each turning on a little later, each adding a bit more slope — and the sum hugs the parabola.

---

# Worked example: $x^2$ in numbers

<div class="math-box">

Four ReLU **ramps** at $x=0,0.25,0.5,0.75$ on $[0,1]$, each $\text{relu}(x-b)$ with slope 1:

| ReLU $i$ | $b_i$ | $\alpha_i$ | running slope after $b_i$ |
|:-:|:-:|:-:|:-:|
| 1 | 0.00 | 0.25 | 0.25 |
| 2 | 0.25 | 0.50 | 0.75 |
| 3 | 0.50 | 0.50 | 1.25 |
| 4 | 0.75 | 0.50 | 1.75 |

</div>

The running slope $(0.25,0.75,1.25,1.75)$ is exactly the average slope of $x^2$ on each quarter, so the output **matches $x^2$ at every knot**. Between knots the error is at most $\frac{h^2}{8}\max|f''| = \frac{0.25^2}{8}\cdot 2 = \frac{1}{64}\approx 0.016$ — and shrinks as you add ramps.

---

# Add more bricks, watch the error fall

Triangle-tiling on $[0,1]$ has error $\le \tfrac{h^2}{8}\max|f''|$ with $h=1/N$. For $x^2$ ($\max|f''|=2$) that is $\le h^2/4$:

<div class="math-box">

| ramps $N$ | width $h$ | error bound |
|:-:|:-:|:-:|
| 4  | 0.25   | $1.6\times10^{-2}$ |
| 8  | 0.125  | $3.9\times10^{-3}$ |
| 16 | 0.0625 | $9.8\times10^{-4}$ |
| 32 | 0.031  | $2.4\times10^{-4}$ |

</div>

<div class="insight">

**Double the width, quarter the error** — in 1-D accuracy is *cheap*. But hold onto $N$: to keep this error over $d$ inputs you need a bump per cell of a $d$-dimensional grid — about $N^{d}$ of them. That lone exponent is the whole story of Part 3.

</div>

---

# Any squasher builds bumps — ReLU is just the tidiest

Nothing here is special to ReLU. A steep **sigmoid** is almost a step, so two of them subtracted make a bump — same LEGO game, curvier bricks:

<div class="math-box">

$$\text{bump}(x)\approx \sigma\!\big(s(x-a)\big)-\sigma\!\big(s(x-b)\big),\qquad s \text{ large}.$$

</div>

<div class="insight">

That is why UAT holds for sigmoid, tanh, GELU — **any non-polynomial squash.** ReLU won the day for *training* (its gradient is exactly 1 when active), not for *expressiveness*. Bricks are bricks; only the shape changes.

</div>

---

# Practice problem 1

<div class="popquiz">

**Practice problem 1.** Using only ReLUs, build a **flat-topped (trapezoidal) pulse**: zero for $x<1$, rising to height $1$ at $x=2$, holding a **plateau** on $[2,3]$, falling back to $0$ at $x=4$, and zero after. How many ReLUs do you need, and what are their weights?

</div>

*Try it before the next slide — think of it as two soft steps back to back.*

---

# Solution · practice problem 1

A plateau needs an "on" step *and* an "off" step — four ReLUs:

<div class="math-box">

$$f(x) = \text{relu}(x-1) - \text{relu}(x-2) - \text{relu}(x-3) + \text{relu}(x-4)$$

| region | active terms | slope | value |
|:-:|:-:|:-:|:-:|
| $x<1$ | none | 0 | 0 |
| $1\to2$ | $+(x{-}1)$ | $+1$ | rises to 1 |
| $2\to3$ | $+(x{-}1)-(x{-}2)$ | $0$ | **plateau at 1** |
| $3\to4$ | $\dots-(x{-}3)$ | $-1$ | falls to 0 |
| $x>4$ | all four | 0 | 0 |

</div>

<div class="keypoint">

A **triangle** needs 3 ReLUs, a **trapezoid** needs 4 — every extra corner in your target costs one more ReLU. That linear "corners $\to$ neurons" accounting *is* UAT, and it's exactly why width can explode.

</div>

---

# Build it yourself

<div class="notebook">

**🎛 Interactive · building a curve from ReLU bumps** — drag knots, add ReLUs one at a time, and watch a one-hidden-layer net snap onto a target curve. Bumps light up as you add them. *(Interactive Lab · to be authored at `interactive/articles/relu-bumps`)*

</div>

<div class="notebook">

**🎛 Interactive · universal approximation** — grow a single hidden layer live and see the fit sharpen with width. *(Interactive Articles · `nipunbatra.github.io/interactive-articles/universal-approximation`)*

</div>

---

<!-- _class: section-divider -->

## Part 3 · The catch, and why we go deep

---

# The price of width — the curse of dimensionality

To approximate a generic function to error $\epsilon$ with piecewise-linear pieces, the neuron count can grow **exponentially in the input dimension**:

$$\text{1-D: } N \approx O(1/\sqrt\epsilon)\qquad d\text{-D: } N \approx O(1/\epsilon^{\,d/2})$$

| input dim $d$ | grid cells for $\epsilon=0.01$ |
|:-:|:-:|
| 1 | $\approx 10$ |
| 10 | $10^{10}$ |
| 100 | $10^{100}$ |

<div class="warning">

UAT says good weights **exist**. It does *not* say $N$ is reasonable, that SGD finds them, or that the network generalizes. "Wide enough" can mean wider than the number of atoms in the universe.

</div>

---

# The curse, made tangible

Cover the input space with a grid at resolution $0.1$ — ten bins per axis, one bump per cell:

<div class="math-box">

| inputs $d$ | grid cells $10^{d}$ | vs. a big dataset ($10^7$) |
|:-:|:-:|:-:|
| 2  | $100$        | trivially covered |
| 5  | $100{,}000$  | still fine |
| 10 | $10^{10}$    | **1000× more cells than data** |
| 20 | $10^{20}$    | more cells than seconds since the Big Bang |

</div>

<div class="warning">

In high dimensions **almost every cell is empty** — the data can't even reach most of the bumps a shallow net would need. Distances concentrate, "nearest" neighbors stop being near, and filling space by brute width is hopeless. Depth escapes by *reusing* structure instead of tiling space.

</div>

---

# Three things UAT does *not* promise

<div class="columns">
<div>

**Learnability.**
Weights *exist* $\ne$ SGD *finds* them.

**Width.**
The bound on $N$ can be astronomical.

</div>
<div>

**Generalization.**
A net that memorizes $n$ training points also satisfies UAT — perfect on train, useless on test.

</div>
</div>

<div class="popquiz">

**Warm-up.** True or false: UAT guarantees a wide-enough one-layer net will do well on *unseen* data, given enough neurons and data. *(Which of the three columns bites first?)*

</div>

*Answer:* **False** — all three bite. Existence $\ne$ findability, "enough" can be exponential, and UAT is silent on generalization. The practical escape from exponential width is **depth**.

---

# Shallow enumerates, deep reuses

![w:900px](figures/lec02/svg/depth_compositionality.svg)

A shallow net must draw every bump separately — one unit per feature. A deep net computes simple features once, then **reuses** them to build compound features. Same reach, far fewer neurons.

---

# Depth = features of features

Each layer builds on the last, so representations grow in abstraction with depth:

![w:640px](figures/lec01/svg/feature_hierarchy.svg)

| Domain | Reusable hierarchy |
|---|---|
| images | edges $\to$ textures $\to$ parts $\to$ objects |
| language | characters $\to$ words $\to$ phrases $\to$ meaning |
| audio | samples $\to$ phonemes $\to$ syllables $\to$ words |

This compositional reuse is the inductive bias depth gives you that width alone cannot.

---

# Why reuse wins: one edge detector, every object

<div class="keypoint">

A shallow net that must recognize 1000 objects learns each one *straight from pixels*, separately — no sharing. A deep net learns **edges once**, then every shape reuses those edges and every object reuses those shapes.

</div>

<div class="columns">
<div>

**Shallow — no sharing**
`pixels → object₁`
`pixels → object₂`
… 1000 independent templates

</div>
<div>

**Deep — shared substructure**
`pixels → edges → shapes → objects`
one edge bank serves *all* 1000

</div>
</div>

The same edge that outlines a cat's ear outlines a car's bumper. **Depth's inductive bias is reuse** — pay for a feature once, spend it everywhere. That is why edges→textures→parts→objects beats memorizing whole objects.

---

# Parity — the canonical hard-for-shallow function

![w:820px](figures/lec02/svg/parity_tree.svg)

Parity (is the number of 1-bits odd?) is XOR of pairs, then XOR of those, then XOR again — a **recursive** tree of depth $\log n$.

---

# Why depth cracks parity

Parity's structure *is* a shallow tree of XORs — and depth can mirror it exactly:

- **Deep:** $\log n$ layers, $O(n)$ gates — one layer per level of the tree.
- **Shallow:** no hierarchy to exploit, so it must enumerate all $2^{n}$ input patterns.

<div class="paper">

Formal separation: Telgarsky, *"Benefits of Depth in Neural Networks,"* COLT 2016 — there is a function a depth-$2k$ net computes with $O(k)$ units that any depth-$k$ net needs $\ge 2^{k}$ units to match. We take it on faith today; the sketch is in the appendix.

</div>

---

# Depth intuition: each layer folds the paper

<div class="keypoint">

A nonlinearity **folds** the input space; the next layer draws on the folded sheet. One straight cut through paper folded once becomes **two** cuts unfolded — fold again for **four**, then **eight**.

</div>

Fold $k$ times and a single line becomes $2^{k}$ lines. That is exactly how a depth-$k$ ReLU net makes $2^{k}$ kinks from $O(k)$ neurons — and why a shallow net, with nothing to fold, must draw all $2^{k}$ by hand.

<div class="insight">

Reuse is free; enumeration is exponential. You don't need the depth-separation proof (appendix) to feel it — **the fold is the free lunch depth gives you.**

</div>

---

# Practice problem 2

<div class="popquiz">

**Practice problem 2.** Two MNIST classifiers on the same task:
**(a) wide-shallow** $\;784 \to 2048 \to 10$
**(b) deep-narrow** $\;784 \to 128 \to 128 \to \dots \to 128 \to 10\;$ (8 hidden layers of 128)

Count the parameters of each (weights + biases). Which is bigger — and which would you bet on for natural image data?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 2

<div class="math-box">

**(a) wide-shallow:** $\;(784{\cdot}2048 + 2048) + (2048{\cdot}10 + 10) \approx \mathbf{1.63\,M}$

**(b) deep-narrow:** $\;(784{\cdot}128{+}128) + 7{\cdot}(128{\cdot}128{+}128) + (128{\cdot}10{+}10)$
$\qquad\qquad = 100{,}480 + 115{,}584 + 1{,}290 \approx \mathbf{0.22\,M}$

</div>

<div class="insight">

The wide-shallow net has **~7× more parameters**, yet deep-narrow typically **wins** on images. Parameter count is a crude proxy for capacity — what depth adds, width cannot: **reusable hierarchy.** Bet on (b) for natural data.

</div>

---

# Width vs depth — the cost gap in one picture

For a target with fine, repeated structure, a shallow net's cost **doubles** each time the function gets one notch harder; a deep net reuses each layer, so its cost grows only **linearly**.

![w:820px](figures/lec02/svg/width_vs_depth_cost.svg)

At $k=10$: a shallow net needs ~1000 units to match what a deep net does with ~30. That exponential-vs-linear gap is the entire reason we go deep.

---

# When depth helps — and when it doesn't

<div class="columns">
<div>

**Depth helps when** the target has *reusable substructure*:
- images, language, audio
- anything compositional

</div>
<div>

**Depth helps less when** structure is flat:
- much tabular data
- a deeper model may just be harder to optimize

</div>
</div>

<div class="keypoint">

Depth is **not magic** — it's a strong inductive bias for compositional problems. If the data have no reusable structure, extra layers mostly add optimization pain, not capacity you can use.

</div>

---

# See it yourself: does width really generalize?

<div class="notebook">

**🎛 Interactive · double descent** — UAT promises you can *fit*, never that you *generalize*. Grow the width past the point of zero training error and watch test error fall, spike, then fall again — the over-parameterized regime plain UAT can't explain. *(Interactive Articles · `nipunbatra.github.io/interactive-articles/double-descent`)*

</div>

<div class="notebook">

**📓 Notebook · bricks you can count** — hand-build today's triangle bump, the trapezoidal pulse from Practice Problem 1, and the four-ramp $x^2$ fit as raw ReLUs in NumPy, then check that a one-hidden-layer `Linear → ReLU → Linear` learns the same weights. *(ES 667 · `notebooks/03-relu-bricks.ipynb`)*

</div>

---

<!-- _class: section-divider -->

## Part 4 · Expressive — but is it trainable?

---

# So just stack 500 layers?

We've earned two facts: one wide layer is **universal**, and depth makes hard functions **affordable**. The obvious move is to pile on layers.

<div class="popquiz">

**Think first.** If depth is so good, should we train a 500-layer plain MLP? Give an honest yes/no — then read on.

</div>

The answer is *no, not naively* — and the reason is the quiet time bomb from Part 1: the gradient has to survive a long trip backward, and it usually doesn't.

---

# Backprop is a product of Jacobians

To update an *early* weight, backprop multiplies one factor per layer on the way back. A 4-layer scalar chain $y = w_4 w_3 w_2 w_1 x$:

![w:620px](figures/lec02/svg/chain_rule_product.svg)

<div class="math-box">

$$\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial y}\cdot \underbrace{(w_4\, w_3\, w_2)}_{\text{product of factors}}\cdot x$$

</div>

If those per-layer factors are below 1, their product collapses toward zero — the earliest layers get almost no signal.

---

# Sigmoid's 0.25 ceiling → the gradient evaporates

The sigmoid derivative $\sigma'(z)=\sigma(z)(1-\sigma(z))$ never exceeds **0.25**. With weights $\approx 1$, each backward step multiplies by $\le 0.25$:

![w:760px](figures/lec02/svg/sigmoid_gradient_stack.svg)

| layers back | cumulative factor |
|:-:|:-:|
| 1 | 0.25 |
| 5 | $0.25^5 \approx 10^{-3}$ |
| 10 | $0.25^{10}\approx 10^{-6}$ |

After 10 sigmoid layers the earliest weight's gradient is a **millionth** of its size — learning stalls. This is the **vanishing gradient**, and it's why ReLU, careful initialization, and residual connections (L4–L5) exist.

---

# Practice problem 3

<div class="popquiz">

**Practice problem 3.** Prove that a single linear classifier $\hat y = \text{sign}(w_1 x_1 + w_2 x_2 + b)$ **cannot** separate XOR: $(0,0),(1,1)\mapsto 0$ and $(0,1),(1,0)\mapsto 1$. *(Write the four inequalities the weights would need to satisfy, and find the contradiction.)*

</div>

*Try it before the next slide.*

---

# Solution · practice problem 3

Demand output $>0$ for the two "1" points and $\le 0$ for the two "0" points:

<div class="math-box">

$$(0,0):\; b \le 0 \qquad (1,1):\; w_1+w_2+b \le 0$$
$$(0,1):\; w_2+b > 0 \qquad (1,0):\; w_1+b > 0$$

</div>

Add the two **"1"** inequalities: $\;(w_1+b)+(w_2+b) > 0 \Rightarrow w_1+w_2+2b > 0$.
From the two **"0"** inequalities: $\;b\le 0$ and $w_1+w_2+b\le 0 \Rightarrow w_1+w_2+2b \le 0$.

<div class="insight">

The same quantity is both $>0$ and $\le 0$ — a **contradiction**. No line works. Add one hidden layer with a nonlinearity and the plane folds so both "1" corners land on the same side. **This is exactly why we need nonlinearity, depth, and — next — a way to train them.**

</div>

---

# Practice problem 4

<div class="popquiz">

**Practice problem 4.** The sigmoid derivative is $\sigma'(z)=\sigma(z)\big(1-\sigma(z)\big)$.
**(a)** For which $z$ is $\sigma'$ largest, and what is that maximum?
**(b)** In a 20-layer net with weights $\approx 1$, bound the factor by which an early-layer gradient shrinks — and compare with ReLU.

</div>

*Try it before the next slide.*

---

# Solution · practice problem 4

<div class="math-box">

**(a)** Let $s=\sigma(z)\in(0,1)$. Then $\sigma'=s(1-s)$ — a downward parabola in $s$, peaking at $s=\tfrac12$ (i.e. $z=0$) with value $\tfrac12\cdot\tfrac12=\mathbf{0.25}$. Never larger.

**(b)** Best case (every unit at its peak), 20 layers multiply:
$$0.25^{20}=4^{-20}=2^{-40}\approx 9\times10^{-13}.$$
ReLU on its active side contributes $1$ per layer: $1^{20}=1$.

</div>

<div class="keypoint">

Even in the *best* case, sigmoid crushes the gradient by $\sim10^{-12}$ over 20 layers; ReLU leaves it untouched. That $0.25$ ceiling — not a coding bug — is why deep sigmoid nets wouldn't train, and why **L4–L5** reach for ReLU, careful init, and residual connections.

</div>

---

# Practice problem 5

<div class="popquiz">

**Practice problem 5.** A target on $[0,1]$ oscillates with $2^{12}=4096$ straight-line pieces (a fine sawtooth).
**(a)** At minimum, how many hidden units must a *shallow* ReLU net use?
**(b)** A *deep* net doubles the pieces per layer using $\approx 3$ units per layer — how many layers, and how many units total?
**(c)** State the ratio.

</div>

*Try it before the next slide.*

---

# Solution · practice problem 5

<div class="math-box">

**(a)** Every piece needs its own corner, and each corner is a unit: $\;\ge 2^{12}=\mathbf{4096}$ units.
**(b)** Doubling per layer reaches $2^{12}$ after **12 layers**; at $\approx 3$ units each that is $\;\approx \mathbf{36}$ units.
**(c)** Ratio $\;\dfrac{4096}{36}\approx \mathbf{114\times}$.

</div>

<div class="insight">

Shallow cost grows **linearly with the number of teeth** ($2^{k}$); deep cost grows **linearly with the depth** ($k$). Same function, exponential vs. linear price — the paper-fold picture from Part 3, now in numbers. *(This is the intuition behind Telgarsky's theorem in the appendix.)*

</div>

---

# Where this goes next

<div class="notebook">

**📓 Notebook · shallow-wide vs deep-narrow** — fit a spiral with both; count parameters; plot per-layer gradient norms and watch the early-layer gradients vanish in the plain deep net. *(ES 667 · `notebooks/03-expressivity-depth.ipynb`)*

</div>

<div class="notebook">

**🎛 Interactive · vanishing gradients** — stack sigmoids vs ReLUs and watch the backward signal evaporate with depth. *(Interactive Articles · `nipunbatra.github.io/interactive-articles/vanishing-gradients`)*

</div>

**Next (L4):** *backpropagation* — the algorithm that turns "good weights exist" into "good weights we can actually find," and the initialization/ReLU/residual tricks that keep the gradient alive through depth.

---

<!-- _class: summary-slide -->

# One sentence to remember

<div class="keypoint">

**One nonlinearity makes any curve possible; one wide layer makes it universal; depth makes it affordable — and backprop (next) makes it trainable.**

</div>

- **Nonlinearity is mandatory** — stacked linear layers collapse to one line; XOR needs a fold.
- **UAT**: a wide-enough hidden layer approximates any continuous function — bumps built from ReLUs.
- **The catch**: width can be **exponential** (curse of dimensionality); UAT promises existence, not learnability or generalization.
- **Depth**: compositional reuse (features of features, parity) beats exponential width.

**Next (L4):** backprop, initialization, and beating the vanishing gradient.

---

<!-- _class: compact -->

# Sources & credits

<div class="paper">

This lecture is built on the instructor's ES 667 materials and standard references:

- **UAT bump construction, $x^2$ example, depth-vs-width, parity** — *ES 667 Deep Learning*, "Universal Approximation & Going Deep," N. Batra · `github.com/nipunbatra/dl-teaching`
- **Nonlinearity, XOR, activation figures** — *ES 667 Deep Learning*, "Why Deep Learning," N. Batra
- **Basis functions $\phi(x)$ picked by hand** — *Machine Learning (ES 335)*, N. Batra · `github.com/nipunbatra/ml-teaching`
- **Pedagogical framing** (shallow vs deep, intuition-first) — A. Ng, *Deep Learning Specialization* Course 1, Weeks 3–4.
- **Formal results** — Cybenko (1989), Hornik (1991) — universal approximation; Telgarsky (2016) — depth separation.
- **Visual "build any function from bumps"** (bricks, error table, any-squasher) — M. Nielsen, *Neural Networks and Deep Learning*, Ch. 4.
- **Fold / linear-regions & shallow-vs-deep counting intuition** — Montúfar, Pascanu, Cho & Bengio (2014); Telgarsky (2016).
- **Curse of dimensionality (tangible grid)** — C. Bishop, *PRML* §1.4; R. Bellman (1961).
- **Double-descent explainer** — Belkin et al. (2019); Nakkiran et al. (2019).

Figures adapted from the ES 667 figure library. All source courses © N. Batra & teaching staff.

</div>

---

<!-- _class: section-divider -->

## ⭐⭐⭐ Appendix · optional depth

*Skip on a first pass — for the curious.*

---

<!-- _class: math-heavy -->

# ⭐⭐⭐ Optional · UAT — the formal statement & proof sketch

<div class="math-box">

**Theorem** (Cybenko 1989 · Hornik 1991 · Leshno 1993). For any continuous $f:[0,1]^d\to\mathbb R$ and any $\epsilon>0$, there exist $N$ and weights $\{\mathbf w_i,b_i,\alpha_i\}$ with

$$\Big|\, f(\mathbf x) - \sum_{i=1}^{N}\alpha_i\,\sigma(\mathbf w_i^\top\mathbf x + b_i)\,\Big| < \epsilon,$$

for any non-polynomial activation $\sigma$ (including ReLU). One hidden layer suffices.

</div>

**Proof in three moves.** (1) *Continuous $\to$ step:* $f$ is uniformly continuous on the compact cube (Heine–Cantor), so partition it into cells where $f$ varies by $<\epsilon/2$ and replace $f$ by its per-cell average. (2) *Step $\to$ sum of bumps:* a steep $\sigma$ is nearly a step, and a difference of two is a bump — one bump per cell. (3) *Density:* finite sums $\sum_i\alpha_i\sigma(\mathbf w_i^\top\mathbf x+b_i)$ are **dense** in $C([0,1]^d)$ (Stone–Weierstrass / Hahn–Banach). The catch lives entirely in the word **exist** — nothing bounds $N$ or guarantees SGD finds the weights.

---

<!-- _class: math-heavy -->

# ⭐⭐⭐ Optional · Telgarsky's depth separation

<div class="paper">

**Theorem (Telgarsky 2016, simplified).** There is a function $f:[0,1]\to[0,1]$ representable by a ReLU network of depth $2k$ and width $\le 3$, such that **any** ReLU network of depth $\le k$ that approximates $f$ within constant error needs **at least $2^{k}$** units.

</div>

The witness is the $k$-fold composition of the **sawtooth** $\;T(x)=\begin{cases}2x & x\le \tfrac12\\ 2-2x & x>\tfrac12\end{cases}$.

Each composition **doubles** the number of teeth: depth $k$ gives $2^{k}$ teeth using only $O(k)$ ReLUs, because deep layers *reuse* the fold. A shallow net has no fold to reuse — it must place one unit per tooth, i.e. $2^{k}$ units. Plug in $k=10$: a deep net needs ~30 units where a shallow net needs ~1000. **Depth is exponentially more parameter-efficient than width for compositional structure** — the theorem behind the cost-gap picture in Part 3.
