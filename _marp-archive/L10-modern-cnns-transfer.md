---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# Modern CNN Architectures & Transfer Learning

## Lecture 10 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar*

---

# The whole lecture, in one idea

Depth *should* be free — a deeper network can always copy a shallower one and add nothing. For years it wasn't: stack more layers and the network got **worse**. One change fixed it.

<div class="keypoint">

**Make a block output $x + F(x)$ — a learned *correction*, not a replacement.**
Now "do nothing" is the default ($F\to 0$ gives the identity for free), so depth stops hurting. The very same skip connection also opens a $+1$ **gradient highway** back to layer 1. Networks jumped from 20 layers to 1000 — and the features they learn are so generic you almost never train one from scratch again. That reuse is **transfer learning**, the most-used skill in this course.

</div>

You built the conv machinery in L09. Today answers the question it left open: **how do we actually stack it 100+ layers deep — and then never pay that cost again?**

$$\text{LeNet}\to\text{AlexNet}\to\text{VGG} \rightarrow \underbrace{\text{the wall}}_{\text{deeper got worse}} \rightarrow \underbrace{\text{ResNet}}_{x+F(x)} \rightarrow \underbrace{\text{efficiency}}_{1\times1,\ \text{depthwise}} \rightarrow \underbrace{\text{transfer learning}}_{\text{the payoff}}$$

---

# How today connects to the rest of the course

<div class="columns">
<div>

**You'll reuse today's ideas in:**
- **L13–L17** — the skip connection returns as the **Transformer residual stream**; it is the single most load-bearing idea in modern DL
- **L18** — Vision Transformers *give up* the CNN's inductive bias and pay for it in data
- **every applied project** — a frozen backbone + a fresh head is the default recipe for detection, segmentation, retrieval

</div>
<div>

**Borrowed from your own courses** (build on, not repeat):
- convolution, pooling, receptive field — **L09**
- a network *learns* its own basis $\phi(x)$ — **L01**
- MLE / cross-entropy as the training loss — **L01–L03**

</div>
</div>

<div class="insight">

The reference for today is **A. Ng, Deep Learning Specialization, Course 4, Week 2** — *classic networks & transfer learning*. We move fast on the architecture zoo and slow on the two ideas that actually matter: **the residual block** and **transfer learning**.

</div>

---

<!-- _class: section-divider -->

## Part 1 · The ImageNet lineage — what each network fixed

---

# LeNet → AlexNet → VGG: a decade of "fix one thing"

Every classic network is the previous one with a single bottleneck removed. The template never changes — conv+pool to extract, a head to classify (L09) — only what limited it:

| Year | Network | The problem it fixed |
|---|---|---|
| 1998 | **LeNet-5** | *proved the template* — conv+pool+FC reads digits (60k params) |
| 2012 | **AlexNet** | *scale + optimization* — ReLU, dropout, 2 GPUs; won ImageNet by 10% |
| 2014 | **VGG-16** | *simplicity* — throw away big kernels, stack only $3\times3$ deep |
| 2015 | **ResNet** | *the wall* — depth finally stopped helping; skip connections |

<div class="insight">

The dataset behind all of it is **ImageNet**: 1.2M labelled photos, 1000 classes. "Won ImageNet" means *top-1 accuracy on that benchmark* — the arena where every architecture was forged.

</div>

---

# AlexNet — the network that started the deep-learning era

AlexNet (Krizhevsky et al., 2012) was not a new *idea* so much as LeNet **made to work at scale**. Three fixes, each of which you already know:

<div class="columns">
<div>

- **ReLU** instead of tanh → gradients don't saturate, trains several times faster (L04).
- **Dropout** in the FC head → regularizes 60M parameters (L07).
- **Two GPUs** → the model physically didn't fit on one 2012 card.

</div>
<div>

- **8 layers**, big early kernels ($11\times11$, $5\times5$), $224\times224$ input.
- **Data augmentation** (crops, flips) → cheap extra data.

</div>
</div>

<div class="keypoint">

AlexNet cut ImageNet top-5 error from ~26% to ~16% overnight — the result that convinced the field CNNs were the way. But its big early kernels were wasteful, which VGG fixed next.

</div>

---

# VGG — go deep with only $3\times3$

VGG's one idea is the one you proved in L09: **three stacked $3\times3$ convs reach the same $7\times7$ receptive field as one $7\times7$, for fewer parameters and more non-linearities.**

<div class="math-box">

$$\text{one } 7\times7:\ 49\,C^2 \qquad\text{vs.}\qquad \text{three } 3\times3:\ 27\,C^2 \quad(\text{+ three ReLUs, not one})$$

</div>

So VGG throws away *every* large kernel and stacks $3\times3$, $P=1$ blocks 16–19 layers deep. Uniform, boring, and it works — VGG is still a common feature extractor today.

<div class="insight">

The obvious next move: if depth is this good, **just keep stacking**. VGG-19 → VGG-34 → VGG-56… That is exactly where the lineage hits a wall.

</div>

---

# Under the hood · quantifying the $3\times3$ trick

Before we hit the wall, bank *why* VGG's one move works — the same instinct returns all lecture. Stacking small kernels wins on **both** axes at once:

<div class="columns">
<div>

**Receptive field grows by stacking.** Each extra $3\times3$ adds $2$ to the field: one layer sees $3\times3$, two see $5\times5$, three see $7\times7$. Three $3\times3$s match one $7\times7$'s *view*.

</div>
<div>

**Parameters shrink.** Per channel-pair: $3\times(3^2)=27$ weights vs $7^2=49$ — **~45% fewer** — *and* you get three ReLUs instead of one, so a strictly richer function.

</div>
</div>

<div class="insight">

The lesson generalizes: **replace one wide operation with several narrow ones and you usually pay less for more.** That exact instinct returns as the $1\times1$ bottleneck and the depthwise split in Part 3.

</div>

<div class="notebook">

**🎛 Interactive · receptive-field grower** — stack $3\times3$ layers one at a time and watch the receptive field expand $3\to5\to7\to\dots$ back over the input pixels. *(Interactive Lab · `interactive/articles/receptive-field-grower`)*

</div>

---

# The wall: deeper suddenly made things *worse*

Take a plain CNN that trains fine at 20 layers. Stack the *same* design to 56. Common sense says it should do *at least* as well — the extra 36 layers can always learn the identity and change nothing. Reality (He et al., 2015):

<div class="keypoint">

The 56-layer net had **higher error — on the *training* set.** That rules out overfitting. It is an **optimization** failure: the deep net cannot even *learn to copy* the shallow one. This is the **degradation problem**, and it is the exact wall ResNet was built to break.

</div>

Why is copying so hard? Because a plain stack must learn the identity $h_{\text{out}} = h_{\text{in}}$ *through* its nonlinear weight layers — a surprisingly awkward function to represent. Part 2 makes "do nothing" trivial instead.

---

<!-- _class: section-divider -->

## Part 2 · ResNet — the residual block (the key idea)

---

# Seeing the degradation problem

Deeper is not overfitting — it is worse **training** error. The optimizer, not the model's capacity, is the bottleneck:

![w:840px](figures/lec08/svg/degradation_curve.svg)

<div class="insight">

If depth were merely *unhelpful*, the 56-layer curve would sit *on top of* the 20-layer one (extra layers = identity). Instead it sits **above** — proof the deep net can't find the identity it needs. We must make that identity effortless to represent.

</div>

---

# The fix, output-first: what does a residual block *output*?

Don't ask a block to compute the whole mapping $H(x)$. Ask it to output the **input plus a learned correction** $F(x)$:

![w:900px](figures/lec08/svg/residual_block_output.svg)

<div class="keypoint">

**A residual block outputs $\;y = x + F(x)$.** If the weight layers drive $F(x)\to 0$, the block outputs $x$ **exactly** — a perfect identity, at zero cost. "Do nothing" just became the *easiest* thing the block can do, not the hardest.

</div>

---

# Intuition · learn the *residual* — the change, not the whole map

Why is $x + F(x)$ so much easier to train than a plain block's $H(x)$? Because $F$ only has to learn the **difference** from "leave the input alone."

<div class="columns">
<div>

**Plain block** must synthesize the *entire* target mapping $H(x)$ from scratch — even when the right answer is "barely touch this."

</div>
<div>

**Residual block** learns only $F(x) = H(x) - x$, the **correction**. If the ideal map is near identity ($H(x)\approx x$), then $F\approx 0$ — a tiny, easy thing to fit.

</div>
</div>

<div class="keypoint">

**Editing beats rewriting.** Nudging a good draft ("change these three words") is far easier than writing the essay from a blank page. Deep layers rarely need to *transform* their input — mostly they should *refine* it, and $x+F(x)$ makes refinement the default move.

</div>

<div class="notebook">

**🎛 Interactive · vanishing gradients** — watch a plain deep stack's gradient decay layer by layer toward zero, then switch on skip connections and see the signal survive intact all the way to layer 1. *(Interactive Lab · `interactive/articles/vanishing-gradients`)*

</div>

---

# Why "identity for free" kills degradation

Back to the 56-vs-20-layer puzzle — now with residual blocks:

- The deep net can **reproduce** the shallow net exactly: set the 36 extra blocks to $F=0$.
- So a deeper ResNet can **never be forced below** a shallower one — worst case it copies it.
- Training starts from "pass everything through," then only *adds* corrections that actually lower the loss.

<div class="keypoint">

Plain deep nets struggled because *every* layer was forced to learn something useful and hitting "do nothing" was hard. Skip connections make "do nothing" the **default** — so you can pile on depth and only pay for the layers that earn their keep.

</div>

---

# The receipt · ResNet vs plain, in real numbers

He et al. ran the *same* experiment both ways on ImageNet. Plain nets show the degradation; add the skip and deeper suddenly helps:

| Setup | 18-layer | 34-layer | did depth help? |
|---|:--:|:--:|:--:|
| **Plain** (top-1 error) | 27.9% | 28.5% | **no — got worse** |
| **ResNet** (top-1 error) | 27.9% | 25.0% | **yes — got better** |

<div class="keypoint">

Same depth increase, opposite sign. Plain-34 *regressed* on both training and validation error — the degradation problem in a single row. Bolt on the identity skip and those extra 16 layers finally **earn their keep**. The network didn't become more powerful *in principle* — a plain net can represent the very same function — it became **trainable**.

</div>

---

# The second win: skips are a gradient highway

That was the *forward* story. The identity path pays off again on the **backward** pass. This is the idea to remember from today.

<div class="insight">

**Analogy — telephone down a line of 100 people.** The backward message is a correction: *"person 1, you said 'purple monkey' — it should've been 'purple donkey'."* Relayed person-to-person, it's a meaningless whisper by the time it reaches person 1, who can never learn. That is the **vanishing gradient**.

A skip connection is a **direct, uninterrupted wire** from the end back to the beginning — the correction arrives intact.

</div>

Concretely, the identity path adds a $+1$ to the gradient at *every* block, so the product of many small factors can never collapse to zero. Let's see the arithmetic.

---

# Worked numeric · gradient flow, plain vs residual

Upstream gradient $\partial\mathcal L/\partial h_{\text{out}} = 0.5$. Tiny weights make the block's own derivative small: $F'(h) = 0.01$.

| | Plain: $\;h_{\text{out}}=F(h_{\text{in}})$ | Residual: $\;h_{\text{out}}=h_{\text{in}}+F(h_{\text{in}})$ |
|---|-------|----------|
| one block | $0.5 \cdot 0.01 = 0.005$ | $0.5 \cdot (1+0.01) = 0.505$ |
| 10 blocks | $0.5 \cdot (0.01)^{10} = 5\times10^{-21}$ | $0.5 \cdot (1.01)^{10} \approx 0.55$ |

<div class="keypoint">

Plain → **completely vanished** after 10 blocks. Residual → **still strong**. The identity path stops the gradient from vanishing *at construction time*, not through training luck. This is why ResNet trains 152 layers easily while a plain 34-layer net couldn't even fit its training set.

</div>

---

# Practice problem 1

<div class="popquiz">

**Practice problem 1.** A residual block computes $\;h_{\text{out}} = h_{\text{in}} + F(h_{\text{in}})$.

**(a)** Write $\dfrac{\partial h_{\text{out}}}{\partial h_{\text{in}}}$.
**(b)** For a stack of $L$ such blocks, write $\dfrac{\partial \mathcal L}{\partial h_0}$ as a product over blocks.
**(c)** Use (b) to argue the gradient *cannot* vanish to exactly zero — something a plain stack $h_{\text{out}}=F(h_{\text{in}})$ can't promise.

</div>

*Try it before the next slide.*

---

# Solution · practice problem 1

**(a)** Differentiate through the sum — the identity term contributes $1$:
$$\frac{\partial h_{\text{out}}}{\partial h_{\text{in}}} = \underbrace{\frac{\partial h_{\text{in}}}{\partial h_{\text{in}}}}_{=\,1} + \frac{\partial F}{\partial h_{\text{in}}} = 1 + F'(h_{\text{in}})$$

**(b)** Chain rule across $L$ blocks:
$$\frac{\partial \mathcal L}{\partial h_0} = \frac{\partial \mathcal L}{\partial h_L}\prod_{\ell=1}^{L}\Big(1 + F'_\ell\Big)$$

<div class="keypoint">

**(c)** Every factor is $1 + F'_\ell$, so even when *all* the $F'_\ell \approx 0$ the product is $\approx 1$ — a **clear path** survives. A plain stack multiplies bare $F'_\ell$; if each is $<1$ the product collapses to $0$. **The $+1$ is the whole trick.**

</div>

---

# BatchNorm — the other thing that made deep nets trainable

ResNet's skip is the star, but it rides on a co-star introduced the same era. **Batch Normalization** (Ioffe & Szegedy, 2015) standardizes each layer's pre-activations across the mini-batch:

<div class="math-box">

$$\hat x = \frac{x - \mu_{\text{batch}}}{\sqrt{\sigma^2_{\text{batch}} + \epsilon}}, \qquad y = \gamma\,\hat x + \beta \;\;(\gamma,\beta \text{ learned})$$

</div>

Keeping every layer's inputs at a stable scale lets you use larger learning rates and keeps activations out of saturating regions. Every ResNet block is **conv → BN → ReLU**.

<div class="insight">

A frozen-backbone gotcha for later: BN keeps *running statistics*. If you "freeze" a backbone but leave BN in train mode, those stats drift and quietly wreck your features — call `.eval()` on the frozen part.

</div>

---

# The ResNet block: basic vs bottleneck

Two block shapes. Shallow ResNets use the **basic** block; ResNet-50+ use the **bottleneck**, which squeezes channels with $1\times1$ convs so the expensive $3\times3$ runs on fewer channels (Part 3's idea, previewed):

![w:920px](figures/lec08/svg/resnet_bottleneck.svg)

Same external shape, far fewer parameters in the middle — that $1\times1\to3\times3\to1\times1$ sandwich is why 50-layer ResNets are actually *cheaper* per block than 34-layer ones.

---

# When the shortcut shape doesn't match

The clean skip $y = x + F(x)$ needs $x$ and $F(x)$ to be the **same shape**. But a block often downsamples (stride 2) or changes channel count — then you can't add a $28\times28\times256$ tensor to a $14\times14\times512$ one.

![w:900px](figures/lec08/svg/resnet_skip_closeup.svg)

<div class="math-box">

**Fix — a projection shortcut.** Put a $1\times1$ conv $W_s$ (same stride, right channel count) on the skip:
$$y = F(x) + W_s\,x$$

</div>

$W_s$ is a learnable "adapter plug": it matches spatial size (via stride) and channels (via output count) so the addition is legal. Everywhere else, the skip stays a plain identity.

---

# The bottleneck block, in code

```python
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, c_in, c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c, 1)                       # 1x1 squeeze
        self.conv2 = nn.Conv2d(c, c, 3, stride, padding=1)       # 3x3 spatial
        self.conv3 = nn.Conv2d(c, c * self.expansion, 1)         # 1x1 expand
        self.bn1, self.bn2, self.bn3 = (nn.BatchNorm2d(x) for x in (c, c, c*self.expansion))
        # projection shortcut ONLY when shapes disagree:
        self.shortcut = (nn.Conv2d(c_in, c*self.expansion, 1, stride)
                         if stride != 1 or c_in != c*self.expansion else nn.Identity())
    def forward(self, x):
        out = F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))))
        out = self.bn3(self.conv3(out))
        return F.relu(out + self.shortcut(x))       # <-- the residual add
```

The last line *is* the whole lecture: `out + self.shortcut(x)`. `nn.Identity()` when shapes match, the $1\times1$ projection $W_s$ when they don't.

---

# The ResNet family by depth

One block design, scaled by how many you stack:

| Model | Depth | Params | ImageNet top-1 |
|-------|-------|--------|----------------|
| ResNet-18 | 18 | 12 M | 69.8% |
| ResNet-34 | 34 | 22 M | 73.3% |
| **ResNet-50** | 50 | 26 M | **76.1%** |
| ResNet-101 | 101 | 45 M | 77.4% |
| ResNet-152 | 152 | 60 M | 78.3% |

<div class="notebook">

**🎛 Interactive · ResNet block & skip** — step through a residual block and watch the skip carry $x$ around $F(x)$; toggle the projection shortcut on a shape mismatch. *(Interactive Lab · `interactive/articles/resnet` · [live](https://nipunbatra.github.io/interactive-articles/resnet/))*

</div>

**ResNet-50 is the workhorse** — unless you have a specific reason, start there for any CNN task.

---

<!-- _class: section-divider -->

## Part 3 · Efficiency — $1\times1$, Inception, depthwise-separable

---

# The $1\times1$ convolution — a channel mixer

A $1\times1$ conv looks trivial (one pixel!) but it is the efficiency workhorse. It does *no* spatial mixing — it mixes **channels**: at each position it's a tiny fully-connected layer across the $C_{\text{in}}$ channels.

<div class="math-box">

Cost of a $K\times K$ conv is $(K\cdot K\cdot C_{\text{in}})\cdot C_{\text{out}}$. A $1\times1$ that maps $256\to64$ costs $256\cdot64$ — and it **shrinks the channel count** so the following $3\times3$ runs cheap.

</div>

<div class="insight">

That "squeeze channels → do the expensive conv → expand back" sandwich is the **bottleneck** you just saw in ResNet-50, the reduction inside Inception, and the FFN projections inside every Transformer (L13). Learn it once, see it everywhere.

</div>

---

# Inception — don't pick a kernel size, offer a buffet

VGG fixes one kernel size per layer. But some features are tiny (want $1\times1$) and some are large (want $5\times5$). Inception's answer: **run all of them in parallel** and let SGD weight the branches.

![w:920px](figures/lec08/svg/inception_module.svg)

Each branch's outputs are **concatenated along channels**, so the next layer sees every scale at once. The $1\times1$ convs *inside* each branch keep the channel count — and thus the cost — under control.

---

# Worked numeric · the $1\times1$ makes Inception affordable

A naïve Inception branch runs a $5\times5$ conv straight on a fat input. On a $28\times28\times192$ feature map, producing $32$ output channels:

<div class="math-box">

**Naïve $5\times5$:** output $28\times28\times32$, each value costs $5\cdot5\cdot192$ multiply-adds:
$$(28\cdot28\cdot32)\times(5\cdot5\cdot192) \approx \mathbf{120\ \text{million}}$$

</div>

Now insert a $1\times1$ that first **squeezes** $192\to16$ channels, *then* run the $5\times5$ on only $16$:

<div class="math-box">

$$\underbrace{(28\cdot28\cdot16)(1\cdot1\cdot192)}_{\text{squeeze}\ \approx\,2.4\text{M}} \;+\; \underbrace{(28\cdot28\cdot32)(5\cdot5\cdot16)}_{5\times5\ \text{on 16 ch}\ \approx\,10\text{M}} \approx \mathbf{12.4\ \text{million}}$$

</div>

<div class="keypoint">

**~10× cheaper**, *same* output shape. A $1\times1$ costs almost nothing yet lets the expensive kernel run on a **thin** stack of channels — the identical squeeze that powers ResNet-50's bottleneck and every Transformer's FFN projection (L13). Learn it once, spot it everywhere.

</div>

---

# Depthwise-separable convolution — split spatial from channel

A standard conv does two jobs in one expensive step: mix **space** and mix **channels**. MobileNet (Howard et al., 2017) splits them:

![w:920px](figures/lec08/svg/depthwise_separable.svg)

<div class="insight">

**Smoothie analogy.** Standard conv = one giant blender mixing every fruit spatially *and* combining flavours at once. Depthwise-separable = (1) **depthwise**: a small blender per fruit, blends each *spatially* only; (2) **pointwise** ($1\times1$): the chef takes a spoon of each puree and combines *flavours*. Same smoothie, a fraction of the work.

</div>

---

# Depthwise-separable — the parameter math

Standard $3\times3$ conv, $C_{\text{in}}=C_{\text{out}}=C$: $\;\text{cost} = (D_K\cdot D_K\cdot C)\cdot C = D_K^2\,C^2$.

<div class="columns">
<div>

**Depthwise** — one $D_K\times D_K$ filter *per* input channel (no channel mixing):
$$\text{cost}_1 = D_K^2\cdot C$$

</div>
<div>

**Pointwise** — a $1\times1$ conv mixing $C\to C$ channels:
$$\text{cost}_2 = C^2$$

</div>
</div>

<div class="math-box">

$$\text{ratio} = \frac{D_K^2 C + C^2}{D_K^2 C^2} = \frac{1}{C} + \frac{1}{D_K^2} \;\xrightarrow[\;D_K=3\;]{}\; \frac{1}{C} + \frac{1}{9}$$

</div>

For $C=128$: $147{,}456 \to 17{,}536$ params — **~8.4× cheaper**, ~1% accuracy drop. This split powers every on-device model since 2017 (real-time camera AR, wake-word, live caption all need a ≤30 ms budget).

---

# Practice problem 2

<div class="popquiz">

**Practice problem 2.** Compare a standard $3\times3$ conv with $C_{\text{in}}=C_{\text{out}}=256$ against its depthwise-separable replacement.

**(a)** Parameters in the standard conv?
**(b)** Parameters in the depthwise part, and the pointwise ($1\times1$) part?
**(c)** The savings ratio — is it close to the promised $\sim8\times$?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 2

<div class="columns">
<div>

**(a) Standard $3\times3$**
$$3\cdot3\cdot256\cdot256 = \mathbf{589{,}824}$$

**(b) Depthwise + pointwise**
$$\underbrace{3\cdot3\cdot256}_{\text{depthwise}} = 2{,}304$$
$$\underbrace{1\cdot1\cdot256\cdot256}_{\text{pointwise}} = 65{,}536$$
$$\text{total} = \mathbf{67{,}840}$$

</div>
<div>

**(c) Ratio**
$$\frac{589{,}824}{67{,}840} \approx \mathbf{8.7\times}$$

Check with the formula:
$$\frac{1}{256} + \frac{1}{9} = 0.0039 + 0.111 = 0.115$$
$$\Rightarrow 1/0.115 \approx 8.7\times \checkmark$$

</div>
</div>

<div class="keypoint">

The bigger the channel count $C$, the closer the ratio approaches $D_K^2 = 9\times$ — the $1/C$ term fades. Depthwise-separable convs get *relatively* cheaper exactly where nets are widest.

</div>

---

# EfficientNet — scale depth, width, resolution together

To make a net "bigger" you have three knobs: **depth** (more layers), **width** (more channels), **resolution** (bigger images). The old way turned one all the way up (VGG→depth). EfficientNet (Tan & Le, 2019) grows all three *in balance* with a single knob $\phi$:

| Model | Depth | Width | Res | Params | ImageNet top-1 |
|-------|-------|-------|-----|--------|----------------|
| B0 | 1.0 | 1.0 | 224 | 5.3 M | 77.3% |
| B3 | 1.4 | 1.2 | 300 | 12 M | 81.6% |
| B7 | 2.0 | 2.0 | 600 | 66 M | 84.3% |

<div class="insight">

The *compound-scaling rule* (in the appendix) held the accuracy/param frontier for years. Even after ViTs and ConvNeXt overtook the architecture, the **idea** — scale the knobs jointly — outlived it.

</div>

---

# The whole arc is one principle in different costumes

Everything in Part 3 is the same move: *find a way to add capacity that pays for itself.*

| Architecture | Trick | What it buys |
|:-:|:-:|:-:|
| **Inception** | parallel multi-scale kernels | many receptive fields, cost tamed by $1\times1$ |
| **ResNet** | identity skip-connections | depth without degradation |
| **MobileNet** | depthwise + pointwise split | ~$8$–$10\times$ fewer FLOPs, runs on a phone |
| **EfficientNet** | compound depth×width×res scaling | one knob, balanced growth |

<div class="keypoint">

You do not need to memorize the zoo. You need **the residual block** (depth) and **the $1\times1$/depthwise factorizations** (efficiency) — and then the practical payoff: **you rarely train any of these from scratch.**

</div>

---

<!-- _class: section-divider -->

## Part 4 · Transfer learning — the practical payoff

---

# The premise: 1.2M images already taught a net what an edge is

Someone spent weeks of GPU time teaching ResNet-50 to see edges, textures, fur, and eyes on ImageNet. **Why relearn that** for your 500-image problem?

![w:660px](figures/lec08/svg/transfer_reuse.svg)

<div class="keypoint">

**Transfer learning = reuse the learned features.** L01's point: a network *learns its own basis* $\phi(x)$ — edges → textures → parts — and that $\phi$ is **generic**. So keep a big pretrained net's $\phi$, train only a fresh head. The pretrained net is a **learned prior** — worth ≈ 1–2 orders of magnitude of labelled data.

</div>

---

# Intuition · don't train from scratch — someone already learned edges for you

The first convolutional layers of *every* vision model learn nearly the same thing: oriented edges, colour blobs, simple textures — Gabor-like filters, painstakingly rediscovered from zero every time someone trains from scratch.

<div class="keypoint">

**That work is already done.** A pretrained backbone hands you a feature extractor $\phi$ that cost **weeks of GPU time and 1.2M labels** to build — for free, in one line of code. Refusing it to "train your own" on 500 images is like re-deriving calculus before every homework problem.

</div>

<div class="columns">
<div>

**From scratch** on 500 images: the net burns its scarce data *relearning edges*, and overfits long before it ever reaches useful object parts.

</div>
<div>

**Warm-started** from ImageNet: edges and textures are already there, so the 500 images only need to learn *your* decision boundary on top. That prior is worth **≈ 1–2 orders of magnitude** of labelled data.

</div>
</div>

---

# What the backbone already knows — and what it doesn't

ImageNet pretraining gives you a **generic vision stack**, but not every layer transfers equally:

<div class="columns">
<div>

- **Early layers** — edges, colour blobs, textures. *Universal* — transfer to almost any image domain.
- **Mid layers** — motifs, object parts (an eye, a wheel). *Mostly* transferable.

</div>
<div>

- **Late layers** — whole ImageNet categories (this exact breed of dog). *Domain-specific* — usually **replaced**.

</div>
</div>

<div class="keypoint">

**The transfer rule:** the *more* data you have in your new domain, the *more* layers you unfreeze. Little data → keep the generic early layers frozen, retrain only the top. Lots of data → let the whole thing adapt.

</div>

---

# Three words you'll hear in every transfer conversation

- **Backbone** — the convolutional *body* of the pretrained net. Your "feature extractor" $\phi$.
- **Head** — the final classifier. Discard the original 1000-class ImageNet head; bolt on your own (e.g. 5 plant-disease classes).
- **Freezing** — setting `requires_grad = False`. The optimizer skips that layer, so it acts as a fixed feature extractor.

<div class="insight">

The recipes on the next slides differ in exactly **one** dimension: *how much of the backbone you freeze*. That's the entire design space of transfer learning.

</div>

---

# The three recipes

![w:920px](figures/lec08/svg/transfer_learning.svg)

<div class="columns">
<div>

**1 · Linear probe** — freeze the *whole* backbone, train only a new head. Fast, tiny-data-safe, hard to overfit.

</div>
<div>

**2 · Fine-tune top** — freeze early layers, retrain later blocks + head. The middle ground.

</div>
<div>

**3 · Full fine-tune** — unfreeze everything, low LR. Needs the most data; adapts every layer.

</div>
</div>

---

# Which recipe? Read it off your data size

<div class="math-box">

| Your labelled data | Recipe | What to freeze |
|:-:|:-:|:-:|
| < 100 | Linear probe | everything except the head |
| 100 – 1k | Fine-tune top layers | freeze conv1–3 |
| 1k – 10k | Full fine-tune | nothing (discriminative LR) |
| 10k – 100k | Full fine-tune + LR schedule | nothing, larger LR |
| > 100k | Maybe train from scratch | — (warm start still rarely hurts) |

</div>

<div class="insight">

**Smaller data → freeze more; larger data → train more.** With very little data, an unfrozen 25M-parameter backbone will overfit and *forget* its good features. Freezing protects the prior you paid for.

</div>

---

# Practice problem 3

<div class="popquiz">

**Practice problem 3.** You must build an image classifier twice.

**Dataset A** — **500** labelled images, 5 classes of natural photos.
**Dataset B** — **1,000,000** labelled images in your domain.

For **each**, pick a recipe (linear probe / fine-tune top / full fine-tune / from scratch) and justify it in one line. What single quantity decided your answer?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 3

<div class="columns">
<div>

**Dataset A — 500 images**
→ **Linear probe** (or fine-tune only the top block).
Freeze the backbone: 500 images can't safely train 25M params without overfitting and *erasing* the pretrained features. Reuse the generic $\phi$; learn only a 5-way head — a few thousand params.

</div>
<div>

**Dataset B — 1M images**
→ **Full fine-tune** (from-scratch is now viable too).
With this much in-domain data you *can* adapt every layer; the pretrained weights just give a better-than-random **warm start** that speeds convergence and rarely hurts.

</div>
</div>

<div class="keypoint">

The deciding quantity is **your labelled dataset size relative to the model's capacity**. Little data → lean on the prior (freeze). Lots of data → let the data speak (unfreeze). Everything in transfer learning is this trade-off.

</div>

---

# Practice problem 4

<div class="popquiz">

**Practice problem 4.** You linear-probe a pretrained **ResNet-50** for a **10-class** task. The backbone has $\approx 25.5$M parameters and outputs a $2048$-dim feature vector; you freeze it and train only a fresh linear head.

**(a)** How many parameters does the optimizer actually update (include the bias)?
**(b)** What fraction of the *total* model is that?
**(c)** In one line: why does this make a linear probe so hard to overfit on small data?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 4

<div class="columns">
<div>

**(a) Trainable params — the head only**
$$\underbrace{2048\times 10}_{\text{weights}} + \underbrace{10}_{\text{bias}} = \mathbf{20{,}490}$$

**(b) Fraction of the whole**
$$\frac{20{,}490}{25.5\text{M}+20{,}490} \approx \mathbf{0.08\%}$$

</div>
<div>

**(c) Why it resists overfitting**
You fit **~20k** parameters, not 25M — a plain logistic regression on frozen features. With so few free parameters it simply *can't* memorize a small training set; the 25M-param $\phi$ is fixed, so there is almost nothing to overfit.

</div>
</div>

<div class="keypoint">

"Reuse the features, retrain the head" made literal: **>99.9% of the network never moves.** All the representational power is inherited — you only learn a decision boundary *inside* the pretrained feature space.

</div>

---

# Discriminative (layer-wise) learning rates

When you *do* unfreeze early layers, don't wreck them: they're already good, so they should move **slower** than the fresh head.

```python
# PyTorch — one learning rate per parameter group
params = [
    {"params": model.conv1.parameters(), "lr": 1e-5},   # early  · already great · slow
    {"params": model.layer2.parameters(), "lr": 3e-5},
    {"params": model.layer4.parameters(), "lr": 3e-4},   # late   · needs to adapt · fast
    {"params": model.fc.parameters(),     "lr": 1e-3},   # new head · random init · fastest
]
opt = torch.optim.AdamW(params, weight_decay=0.01)
```

<div class="insight">

A too-high LR on the backbone is the #1 transfer bug: the net does *worse* than a plain linear probe because large steps overwrite the pretrained features on the first batch. Fix: divide the backbone LR by 10, leave the head LR alone.

</div>

---

# Practice problem 5

<div class="popquiz">

**Practice problem 5.** You have **8,000** in-domain images, so you full-fine-tune a pretrained ResNet-50 with a **single** learning rate $10^{-3}$ for *all* parameters. After one epoch your validation accuracy is **worse** than a plain linear probe on the same backbone.

**(a)** What went wrong on that very first batch?
**(b)** Give a two-line fix using discriminative learning rates.
**(c)** Why is the *head's* $10^{-3}$ fine, but the *backbone's* is not?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 5

<div class="columns">
<div>

**(a) The pretrained features got wiped.**
A random-initialized head produces a large, noisy loss → large gradients flow back into the backbone. At $\text{lr}=10^{-3}$ the *first* update already overwrites the carefully learned edges and textures — you've discarded the very prior you came for.

</div>
<div>

**(b) Split the learning rates:**
```python
opt = AdamW([
  {"params": backbone.parameters(), "lr": 1e-5},
  {"params": head.parameters(),     "lr": 1e-3},
])
```
Backbone crawls; head sprints.

</div>
</div>

<div class="keypoint">

**(c)** The head is **random** — it *should* move fast. The backbone is **already excellent** — big steps can only make it worse. Rule of thumb: backbone LR $\approx$ head LR $/\,100$, or warm up the head first (freeze the backbone for an epoch) *then* unfreeze. This is the #1 transfer-learning bug.

</div>

---

# Loading a pretrained backbone — PyTorch

```python
import torch, torch.nn as nn
import torchvision.models as M

model = M.resnet50(weights=M.ResNet50_Weights.IMAGENET1K_V2)   # 1. pretrained weights

for p in model.parameters():                                   # 2. freeze the backbone
    p.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, n_classes)          # 3. new head (trainable by default)

opt = torch.optim.AdamW(model.fc.parameters(), lr=1e-3)        # 4. train ONLY the head

for p in model.layer4.parameters():                            # 5. later: unfreeze the top block
    p.requires_grad = True
```

<div class="warning">

**Preprocessing contract.** Use the *exact* resize + normalization the backbone was pretrained with, or your features are garbage. `ResNet50_Weights.IMAGENET1K_V2.transforms()` hands you the right ones — don't guess the mean/std.

</div>

---

# The `timm` ecosystem — stop hand-rolling architectures

```python
import timm

# 1000+ pretrained vision models, each in one line, head resized for you:
model = timm.create_model('resnet50',            pretrained=True, num_classes=10)
model = timm.create_model('efficientnet_b3',     pretrained=True, num_classes=10)
model = timm.create_model('convnext_base',       pretrained=True, num_classes=10)
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=10)
```

<div class="realworld">

`timm` (Ross Wightman) is the de-facto vision model zoo — every competitive Kaggle vision solution runs on it. `num_classes=10` swaps the head automatically; `timm.data.resolve_data_config` hands you the matching preprocessing.

</div>

---

# Picking a backbone in 2026

| Scenario | Backbone | Why |
|:-:|:-:|:-:|
| Natural photos, plenty of data | ResNet-50 / ConvNeXt | strong, well-supported |
| Edge / real-time | MobileNet-V3 | fits the latency budget |
| Very small labelled set | CLIP / DINO features | cross-domain generalization |
| Arbitrary RGB, unknown domain | **DINOv2 / v3 frozen features** | best general-purpose self-supervised prior |

<div class="realworld">

Today the default is often a **self-supervised** backbone (DINOv2/v3, trained on 100M+ *unlabelled* images) used as a *frozen* feature extractor — frequently beating ImageNet-supervised ResNets downstream. The transfer *recipe* is identical: freeze, add a head, train the head.

</div>

---

# The bitter truth: transfer can *hurt*

<div class="warning">

**Negative transfer.** ImageNet is natural photos. Point that backbone at X-rays, satellite imagery, or microscopy and the mid/late features may simply not apply — sometimes a from-scratch net does *better*.

</div>

Warning signs, some visible before you train:
- **Large domain gap** — grayscale medical, overhead satellite. Early edges transfer; object-level features don't.
- **Very different resolution** — ImageNet is $224\times224$; a $1024^2$ scan may need re-pretraining.
- **Symptom to watch** — validation loss starts *higher* with pretraining than without.

**Fix:** always train a small from-scratch baseline to compare; or pretrain on a closer domain (RadImageNet, SatCLIP); or self-supervise on your own unlabelled data.

---

# Explore transfer, adaptation & efficiency

<div class="notebook">

**🎛 Interactive · domain adaptation** — shift the target distribution away from the source and watch a frozen backbone's accuracy fall, then recover it by adapting — the *negative-transfer* story from the previous slide, made tangible. *(Interactive Lab · `interactive/articles/domain-adaptation`)*

</div>

<div class="notebook">

**🎛 Interactive · LoRA adapters** — instead of moving 25M weights, inject a tiny low-rank $\Delta W = BA$ and train only that. Parameter-efficient fine-tuning: the transfer recipe's modern successor. *(Interactive Lab · `interactive/articles/lora-adapter`)*

</div>

<div class="notebook">

**🎛 Interactive · self-supervised vision features** — see *why* a DINO-style backbone learns transferable structure from **unlabelled** images, the frozen prior behind the 2026 default recipe. *(Interactive Lab · `interactive/articles/vision-ssl`)*

</div>

---

# See it in code

<div class="notebook">

**📓 Notebook · transfer a VGG feature extractor** — freeze a pretrained VGG backbone, train a fresh head on MNIST, and inspect which images it still gets wrong. *(ML ES 335 · `notebooks/vgg-minst.ipynb`)*

</div>

<div class="notebook">

**📓 Notebook · modern frozen-feature transfer (DINOv3)** — pull a `timm` DINOv3 backbone, extract frozen features, and train a linear head on CIFAR-10 — the 2026 default recipe, end to end. *(ML ES 335 · `notebooks/dinov3-classification.ipynb`)*

</div>

<div class="notebook">

**🎛 Interactive · ResNet block & skip** — replay the residual add and the gradient highway one block at a time. *(Interactive Lab · [resnet](https://nipunbatra.github.io/interactive-articles/resnet/); transfer preview: [knowledge-distillation](https://nipunbatra.github.io/interactive-articles/knowledge-distillation/))*

</div>

---

<!-- _class: summary-slide -->

# One sentence to remember

<div class="keypoint">

**Make a block output $x + F(x)$ — depth becomes free and gradients get a $+1$ highway — and once a net has learned generic features, freeze the backbone, swap the head, and fine-tune by data size instead of ever training from scratch.**

</div>

- **The wall** — deeper plain nets got worse on *training* error: an optimization failure, not overfitting.
- **ResNet** — $y=x+F(x)$: identity for free (kills degradation) + a $+1$ gradient path (kills vanishing).
- **Efficiency** — $1\times1$ (channel mix + bottleneck) and depthwise-separable ($\sim8\times$ cheaper) buy capacity that pays for itself.
- **Transfer learning** — reuse the learned $\phi$; freeze more with less data, unfreeze more with more; `timm` + discriminative LRs.

**Next (L11):** the same frozen backbone, now predicting *where* — detection and segmentation.

---

<!-- _class: compact -->

# Sources & credits

<div class="paper">

This lecture reuses and adapts material from the instructor's own courses (IIT Gandhinagar) and standard references:

- **Modern-CNN & transfer framing, figures** — *ES 667 Deep Learning* CNN materials, N. Batra.
- **Learned basis $\phi$, MLE/cross-entropy grounding; VGG / feature-reuse & DINOv3 notebooks** — *Machine Learning (ES 335)*, N. Batra · `github.com/nipunbatra/ml-teaching` (`vgg-minst.ipynb`, `dinov3-classification.ipynb`).
- **Pedagogical structure (classic nets & transfer learning)** — A. Ng, *Deep Learning Specialization*, **Course 4 (CNNs), Week 2**. The $1\times1$-reduction cost example follows GoogLeNet/Inception (Szegedy et al., 2014).
- **Interactive explainers** (ResNet & skip, vanishing gradients, receptive-field grower, domain adaptation, LoRA adapters, self-supervised vision features) — *ES 667 Interactive Lab*, N. Batra.
- **ResNet & the degradation/residual idea** — K. He, X. Zhang, S. Ren, J. Sun, *Deep Residual Learning for Image Recognition*, CVPR 2016 (arXiv 2015).
- **MobileNet / depthwise-separable** — A. Howard et al., *MobileNets: Efficient CNNs for Mobile Vision*, 2017. Also GoogLeNet/Inception (Szegedy et al., 2014), EfficientNet (Tan & Le, 2019), BatchNorm (Ioffe & Szegedy, 2015).

Figures adapted from the ES 667 figure library. All source courses © N. Batra & teaching staff.

</div>

---

<!-- _class: section-divider -->

## ⭐⭐⭐ Appendix · optional depth

*Skip on a first pass — for the curious.*

---

<!-- _class: math-heavy -->

# ⭐⭐⭐ Optional · why $1\times1$ bottlenecks are ~8.5× cheaper

The bottleneck's three-step sandwich versus a direct $3\times3$, both mapping $256\to256$ channels at the same spatial size. Conv cost $=(K_h K_w C_{\text{in}})C_{\text{out}}$.

**Direct $3\times3$:** $\;(3\cdot3\cdot256)\cdot256 = 2304\cdot256 = \mathbf{589{,}824}$.

**$1\times1$ bottleneck (squeeze to 64):**

1. **Squeeze** ($1\times1$, $256\to64$): $(1\cdot1\cdot256)\cdot64 = 16{,}384$
2. **Spatial** ($3\times3$, $64\to64$): $(3\cdot3\cdot64)\cdot64 = 36{,}864$
3. **Expand** ($1\times1$, $64\to256$): $(1\cdot1\cdot64)\cdot256 = 16{,}384$

Total $= 69{,}632$, so $589{,}824 / 69{,}632 \approx \mathbf{8.5\times}$ cheaper for the *same* external shape. The expensive $3\times3$ runs on 64 channels, not 256 — that squeeze is the entire saving, and it's why ResNet-50 is cheaper per block than ResNet-34.

---

<!-- _class: math-heavy -->

# ⭐⭐⭐ Optional · EfficientNet's compound-scaling rule

Instead of tuning depth, width, and resolution separately, use **one** knob $\phi$ with constants $\alpha,\beta,\gamma$ fixed once by a small grid search:

$$\text{depth } d = \alpha^{\phi}, \qquad \text{width } w = \beta^{\phi}, \qquad \text{resolution } r = \gamma^{\phi}$$
$$\text{subject to}\quad \alpha\cdot\beta^2\cdot\gamma^2 \approx 2.$$

**Why $\beta$ and $\gamma$ are squared.** A conv layer's FLOPs scale *linearly* with depth, *quadratically* with width (channels multiply on both input and output), and *quadratically* with resolution (an $r\times r$ feature map). So total compute scales as $d\cdot w^2\cdot r^2 = \alpha^{\phi}\beta^{2\phi}\gamma^{2\phi} = (\alpha\beta^2\gamma^2)^{\phi}$. Pinning $\alpha\beta^2\gamma^2\approx2$ makes each $+1$ in $\phi$ **double** the compute — a clean, predictable budget knob.

For EfficientNet-B0→B7 the paper found $\alpha\approx1.2,\ \beta\approx1.1,\ \gamma\approx1.15$. Scaling B0→B3 ($\phi=2$): depth $1.2^2=1.44$, width $1.1^2=1.21$, resolution $224\cdot1.15^2\approx296$ (rounded to 300) — exactly the B3 row.
