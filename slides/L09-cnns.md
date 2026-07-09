---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# Convolutional Neural Networks

## Lecture 9 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar*

---

# The whole lecture, in one idea

A convolution is not a new kind of layer. It is a **linear layer that confesses its assumptions** about images — and those assumptions buy you a network with **hundreds of thousands of times fewer parameters** that still sees everything.

<div class="keypoint">

**A CNN is an MLP with three constraints wired in: locality, weight sharing, and a receptive field that grows with depth.**
Images obey those exact symmetries, so the constraints cost you nothing and save you almost everything. A single filter, slid across every position, replaces a wall of independent weights.

</div>

You built MLPs in L2–L4 and know they *can* fit an image. Today answers the question those lectures left open: **why does nobody feed a photo to a plain MLP?**

$$\text{flatten an image} \rightarrow \underbrace{\text{param blowup}}_{\text{the problem}} \rightarrow \underbrace{\text{convolution}}_{\text{share a filter}} \rightarrow \underbrace{\text{pooling}}_{\text{invariance}} \rightarrow \underbrace{\text{receptive field}}_{\text{depth = context}} \rightarrow \text{LeNet}$$

---

# How today connects to the rest of the course

<div class="columns">
<div>

**You'll build on today in:**
- **CNN deep-dive** — receptive fields, 1×1 convs, VGG / ResNet, the classic-architecture arms race
- **Transfer learning** — reuse a pretrained CNN backbone instead of training from scratch
- **Detection & segmentation** — the same conv stack, now predicting *where* not just *what*
- **L18 — Vision Transformers** give up this inductive bias, and pay for it in data

</div>
<div>

**Borrowed from your own courses** (we build on, not repeat):
- convolution & pooling *mechanics* — **ML (ES 335)**
- MLPs, ReLU, cross-entropy loss — **L2–L4 of this course**
- the "model outputs $p(y\mid x)$" story — **L1**

</div>
</div>

<div class="insight">

The reference for today is **A. Ng, Deep Learning Specialization, Course 4, Week 1** — the *foundations* of CNNs. We move fast on the mechanics you saw in ES 335 and slow on *why* those choices are the right ones for images.

</div>

---

<!-- _class: section-divider -->

## Part 1 · An MLP throws away everything an image knows

---

# Start honestly: just flatten the image into an MLP

Nothing stops you from feeding pixels to a dense layer. A modest $224\times224$ RGB image is a vector of $224\cdot224\cdot3 = 150{,}528$ numbers. Send it to one hidden layer of 4096 units:

<div class="math-box">

$$\text{params} = 150{,}528 \times 4{,}096 \approx \mathbf{6.2\times10^{8}} \;\text{(616 million)} \quad\text{— for a } single \text{ layer.}$$

</div>

Six hundred million weights before you have detected a single edge. Bigger images and deeper nets make it worse, quadratically. **This is not a tuning problem — it is a modelling problem.**

---

# What the MLP ignored: three facts about images

A dense layer treats the 150,528 pixels as an unordered bag of numbers. But an image is not a bag — it has structure the MLP is blind to:

<div class="columns">
<div>

**1 · Locality**
What makes a pixel an "edge" depends on its *neighbours*, not on a pixel 200 rows away. Useful features are **local**.

**2 · Translation**
A cat in the top-left and the same cat in the centre are the *same* cat. Position should not create a brand-new problem.

</div>
<div>

**3 · Sharing**
An edge detector that is useful at one location is useful at *every* location. The MLP is forced to **relearn** it 150,528 times, once per pixel.

</div>
</div>

<div class="insight">

The MLP's weights are almost all redundant: 99.99% of them *should* be copies of the same small detector applied at different spots. Convolution simply **refuses to store the copies**.

</div>

---

# The fix, in one line

<div class="keypoint">

Take **one small filter** and slide it across every position, reusing the same weights everywhere. That single change delivers all three properties at once:

- a filter only looks at a small window → **locality**
- the same weights run at every position → **weight sharing**
- shift the input, the output shifts with it → **translation equivariance**

</div>

That $3\times3\times3$ filter, with 64 output channels, is $3\cdot3\cdot3\cdot64 + 64 = \mathbf{1{,}792}$ parameters — versus 616 million. Same job, roughly **$350{,}000\times$ fewer weights.** Where did the "lost capacity" go? *Nowhere — it was capacity images never needed.*

---

# Practice problem 1

<div class="popquiz">

**Practice problem 1.** A $32\times32$ RGB image (CIFAR-style) is the input.

**(a)** A fully-connected layer maps it to **100** hidden units. How many weights (ignore biases)?
**(b)** A conv layer uses a **$3\times3$** filter, $3$ input channels, **$16$** output channels. How many weights?
**(c)** What is the ratio?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 1

<div class="columns">
<div>

**(a) Fully connected**
$$32\cdot32\cdot3 = 3{,}072 \text{ features}$$
$$3{,}072 \times 100 = \mathbf{307{,}200}$$

</div>
<div>

**(b) Convolution**
$$(K\cdot K\cdot C_{\text{in}})\cdot C_{\text{out}}$$
$$(3\cdot3\cdot3)\cdot16 = \mathbf{432}$$

</div>
</div>

$$\text{ratio} = \frac{307{,}200}{432} \approx \mathbf{710\times}\ \text{fewer parameters}$$

<div class="keypoint">

And the conv count **does not grow** with image size — a $224\times224$ input uses the *same* 432 weights. The FC count would explode. That decoupling of parameters from resolution is the whole point.

</div>

---

<!-- _class: section-divider -->

## Part 2 · Convolution — filters as feature detectors

---

# A filter is a learned feature detector

<div class="keypoint">

A convolution filter is a small **feature detector**. Slide it across the image and its output map "lights up" wherever its specific feature appears — one map per filter.

</div>

![w:820px](figures/lec07/svg/feature_maps_stack.svg)

Early filters detect edges and colour blobs; deeper ones compose those into textures, parts, and objects. In a CNN we **do not hand-design** these — the network learns whichever detectors lower the loss.

---

# Concretely: a filter that finds vertical edges

Before anything is learned, here is a filter you can read by eye — the classic vertical-edge detector (left bright, right dark):

<div class="math-box">

$$K=\begin{bmatrix} 1 & 0 & -1 \\ 1 & 0 & -1 \\ 1 & 0 & -1 \end{bmatrix} \qquad K^{\top}=\begin{bmatrix} 1 & 1 & 1 \\ 0 & 0 & 0 \\ -1 & -1 & -1 \end{bmatrix}$$

</div>

Where brightness is constant, the $+1$ and $-1$ columns cancel → output $\approx 0$. Where the image jumps from light to dark, they do **not** cancel → a large response. Transpose the filter and you detect **horizontal** edges instead. Training just discovers thousands of such kernels automatically.

---

# One filter, one number: the actual arithmetic

Every cell of the output is **one dot product** of the filter with the patch beneath it — multiply-add, repeated at every position. No calculus, just arithmetic:

![w:940px](figures/lec07/svg/conv_numeric.svg)

<div class="insight">

There is no "convolution magic." Place the filter, multiply overlapping cells, add them up, write the number, slide over one step. The entire feature map is that loop.

</div>

---

# Sliding window, shared weights — with symbols

The same operation, now with the notation we will count with. The filter's weights are **identical** at every stop of the slide:

![w:900px](figures/lec07/svg/convolution_mechanics.svg)

<div class="notebook">

**🎛 Interactive · convolution visualizer** — drag a kernel across an image and watch the feature map fill in cell by cell; flip to an edge kernel and see the edges pop out. *(Interactive Lab · "how convolution sees an edge" · `interactive/articles/convolution-visualizer` · [live](https://nipunbatra.github.io/interactive-articles/convolution-visualizer/))*

</div>

---

# Padding: what to do at the border

Without help, the filter's centre cannot sit on the edge pixels, so the output **shrinks** — and repeated layers erode the image away. Pad the border with zeros to control this:

<div class="columns">
<div>

**"valid" padding** ($P=0$)
No padding. Output is smaller than input. Use it when shrinking is the point.

</div>
<div>

**"same" padding** ($P=\tfrac{K-1}{2}$, $S=1$)
Just enough zeros that **output size = input size**. A $3\times3$ filter uses $P=1$. The default in nearly every modern block.

</div>
</div>

Padding is the "paved sidewalk" that lets the filter start with its centre on the very first pixel.

---

# Stride: how far the filter steps

**Stride $S$** is the step size after each dot product. $S=1$ visits every position; $S=2$ skips every other one, so the output is roughly **half** the size in each dimension.

<div class="insight">

Stride is the cheapest way to **downsample**. A stride-2 conv both extracts features *and* halves resolution in one shot — which is why modern nets increasingly use stride-2 convs in place of a separate pooling layer.

</div>

Together, kernel size sets the *field of view*, padding controls the *border*, and stride controls the *downsampling*. Those three numbers determine the output size exactly.

---

# Channels: a filter spans the full depth

A colour image has **3 channels**; a mid-network tensor might have 256. A single filter is not $3\times3$ — it is $3\times3\times C_{\text{in}}$, covering **all** input channels, and it collapses them to **one** output map.

<div class="math-box">

Stack $C_{\text{out}}$ such filters → $C_{\text{out}}$ output maps. Parameter count of the layer:
$$\boxed{\;\text{params} = K\cdot K\cdot C_{\text{in}}\cdot C_{\text{out}} \;+\; C_{\text{out}}\;}$$

</div>

So "64 filters on an RGB image" means 64 little $3\times3\times3$ cubes. The *spatial* size shrinks per the formula below; the *depth* becomes however many filters you chose.

---

# The output-size formula, derived

A 1-D lawn of width $W$, mower blade of width $K$:

1. **No padding, stride 1.** The blade centre fits at $W-K+1$ positions.
2. **Add padding $P$** on each side → effective width $W+2P$, so $O=(W+2P)-K+1$.
3. **Add stride $S$.** The centre roams a distance $W+2P-K$, taking $S$-sized steps, plus the start:

<div class="math-box">

$$\boxed{\; O = \left\lfloor \dfrac{W + 2P - K}{S} \right\rfloor + 1 \;}$$

</div>

The same rule applies independently to height and width. Now every term has a *reason*, not just a place in a textbook.

---

# Output-size: worked numbers

<div class="columns">
<div>

**"same" block** — $W=224,\,K=3,\,P=1,\,S=1$:
$$O = \left\lfloor\tfrac{224+2-3}{1}\right\rfloor+1 = 224$$
Size unchanged — this is why $3\times3,\,P=1$ is the everyday default.

</div>
<div>

**stride-2 stem** — $W=224,\,K=7,\,P=3,\,S=2$:
$$O = \left\lfloor\tfrac{224+6-7}{2}\right\rfloor+1 = 112$$
Halved — a `Conv2d(3,64,7,stride=2,pad=3)` turns $(3,224,224)$ into $(64,112,112)$.

</div>
</div>

<div class="notebook">

**📓 Notebook · convolution by hand** — run `F.conv2d` with the edge kernels above and with stride/padding, and confirm the output shapes match the formula. *(ML ES 335 · `notebooks/convolution-operation.ipynb`, `convolution-operation-stride.ipynb`, `cnn-edge.ipynb`)*

</div>

---

# Convolution is cheap to *run*, too

Fewer parameters is half the story; convolution also does far fewer **operations**. Take the stride-2 stem ($3\to64$, $K=7$) producing a $112\times112$ map. A conv computes one dot product per output cell:

<div class="columns">
<div>

**Conv**
$$\underbrace{7\cdot7\cdot3}_{\text{filter}}\cdot\underbrace{64}_{\text{maps}}\cdot\underbrace{112^2}_{\text{positions}}\approx 1.2\times10^{8}$$

</div>
<div>

**Equivalent MLP** (same in→out)
$$150{,}528 \times (64\cdot112^2) \approx 1.2\times10^{11}$$

</div>
</div>

<div class="insight">

**~1000× fewer operations** for the same mapping — because the weights are *shared* across all $112^2$ positions and each output only touches a small patch. The parameter saving and the compute saving are the same coin: weight sharing.

</div>

---

# Practice problem 2

<div class="popquiz">

**Practice problem 2.** AlexNet's first layer takes a $227\times227$ image through a filter with $K=11$, stride $S=4$, padding $P=2$.

**(a)** What is the output spatial size?
**(b)** If the layer has $96$ filters, what is the output *tensor* shape?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 2

$$O = \left\lfloor \dfrac{W + 2P - K}{S} \right\rfloor + 1 = \left\lfloor \dfrac{227 + 4 - 11}{4} \right\rfloor + 1 = \left\lfloor \dfrac{220}{4}\right\rfloor + 1 = 55 + 1 = \mathbf{55}$$

<div class="keypoint">

**(a)** $55\times55$.  **(b)** With 96 filters the output tensor is $\mathbf{(96,\,55,\,55)}$ — spatial size set by the formula, depth set by the number of filters. This is exactly AlexNet's real conv1 output.

</div>

The floor matters: had the numbers not divided evenly, we would round *down* — a fractional position has nowhere for the filter to sit.

---

<!-- _class: section-divider -->

## Part 3 · Pooling → translation invariance

---

# Equivariance vs invariance — the two robots

<div class="insight">

**Convolution is translation *equivariant*.** Shift the cat 10 px right → its feature map shifts 10 px right. The output *tracks* the input. Think of a robot that points a laser at the cat wherever it moves.

**A classifier wants translation *invariance*.** "Is there a cat?" should answer the same no matter *where* the cat sits. Think of a robot that just yells **"CAT!"** regardless of position.

</div>

Convolution gives us equivariance for free. **Pooling** is how we start turning that into invariance — a small "I don't care *exactly* where" operation.

---

# Max-pool — the mechanics, and why it gives invariance

Slide a window (usually $2\times2$, stride 2, non-overlapping) and keep only the **strongest** activation in each region:

<div class="math-box">

$$\begin{bmatrix} 1 & 2 & 8 & 3 \\ 4 & 6 & 5 & 1 \\ 9 & 7 & 2 & 3 \\ 5 & 3 & 1 & 0 \end{bmatrix} \xrightarrow{\;2\times2\ \text{max, stride }2\;} \begin{bmatrix} 6 & 8 \\ 9 & 3 \end{bmatrix} \qquad \text{MaxPool}(x)=\max_{i,j\in\text{window}} x_{ij}$$

</div>

Halved the spatial size, kept the peak per region. **The invariance:** moving the strongest feature *within* a window leaves the output unchanged — $\begin{bmatrix}1&2\\4&6\end{bmatrix}\!\to\!6$ and $\begin{bmatrix}6&2\\4&1\end{bmatrix}\!\to\!6$. Small position wobbles no longer change the answer.

---

# Max, average, and global pooling

Three flavours you will meet, all "summarise a window into one number":

| Pooling | Keeps | Used for |
|---|---|---|
| **Max** | the strongest activation | "did this feature fire *anywhere* here?" — the classic default |
| **Average** | the mean activation | smoother summaries; LeNet's original choice |
| **Global average** | one number *per channel* over the whole map | replaces the giant FC head — collapse $(C,H,W)\to(C,1,1)$ |

<div class="insight">

**Global average pooling** is the modern fix for the LeNet problem where the FC head held 98% of the weights: average each feature map down to a single number and feed *those* to the classifier. No flatten, no parameter explosion, and it forces each channel to mean "how much of concept $c$ is present."

</div>

---

# Invariance compounds with depth

One pooling layer only ignores position *within a $2\times2$ window* — tiny. The power comes from **stacking**:

<div class="keypoint">

conv → pool → conv → pool → … Each stage widens the "don't-care" region. By the final layer the network cares **that** a cat is present, not **where** in the frame it sits — local invariance has compounded into near-global invariance.

</div>

This is the **invariance gradient** a classifier builds: early layers are almost equivariant (they tell you *where* an edge is); the last layer is nearly invariant (it tells you *what* the image is). Segmentation, which needs the pixels back, deliberately keeps less of this.

---

<!-- _class: section-divider -->

## Part 4 · The receptive field — depth buys context

---

# What a deep neuron actually sees

<div class="keypoint">

A neuron deep in the network does not look at one input pixel. It looks at a whole **patch** of the original image — its **receptive field**. Stack more layers and that patch grows.

</div>

![w:830px](figures/lec07/svg/receptive_field.svg)

A first-layer $3\times3$ neuron sees a $3\times3$ patch. A neuron two layers up sees a $3\times3$ window *of neurons that each already saw $3\times3$* — so it sees more of the image, without any single filter being large.

---

# How the receptive field grows with depth

For a stack of stride-1 convolutions, each layer adds $K-1$ pixels of reach:

$$\text{RF}_{\ell} = \text{RF}_{\ell-1} + (K-1) \qquad\Rightarrow\qquad \text{after } N \text{ layers of } 3\times3:\quad \text{RF} = 1 + 2N$$

![w:900px](figures/lec07/svg/conv_receptive_field_grow.svg)

RF after conv1 = 3, conv2 = 5, conv3 = 7, … Depth is *how a CNN sees far-away pixels* while every filter stays small and cheap.

---

# 3 stacked 3×3 ≈ one 7×7 — and it's a better deal

Three $3\times3$ convs reach the same $7\times7$ receptive field as one big $7\times7$ conv. But compare the cost ($C\to C$ channels):

<div class="columns">
<div>

**One $7\times7$**
$$7\cdot7\cdot C\cdot C = 49\,C^2$$
one non-linearity

</div>
<div>

**Three $3\times3$**
$$3\cdot(3\cdot3\cdot C\cdot C) = 27\,C^2$$
**three** non-linearities

</div>
</div>

<div class="keypoint">

Same field of view, **~45% fewer parameters**, and *three* ReLUs instead of one — a richer function for less money. This one trade is the entire design philosophy of VGG: never go wide, go **deep with $3\times3$**.

</div>

---

# What a receptive field of a given size can "see"

The RF size is the **scale of feature** a layer can possibly detect — you cannot recognise something larger than your window:

| RF | What fits in this window |
|----|---------------------------|
| $3\times3$ | a corner, a fragment of one edge |
| $5\times5$ | a longer edge or a junction |
| $11\times11$ | on a $28\times28$ digit, the loop of a "6" |
| $\sim49\times49$ | on a $224\times224$ photo, a whole eye or nose |
| $\sim500$ | the entire image (deep final block) |

Edges → textures → parts → objects: the hierarchy is *built by the growing receptive field.* Deeper net = bigger RF = richer features.

---

# Practice problem 3

<div class="popquiz">

**Practice problem 3.** All layers are $3\times3$ convolutions, stride 1, no pooling.

**(a)** What is the receptive field after **4** such layers?
**(b)** How many stacked $3\times3$ layers reach the same receptive field as a single **$9\times9$** convolution?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 3

Recursion $\text{RF}=1+N(K-1)$ with $K=3$, so each layer adds 2:

<div class="columns">
<div>

**(a)** After 4 layers:
$$\text{RF} = 1 + 4\cdot 2 = \mathbf{9}$$
(input 1 → 3 → 5 → 7 → 9)

</div>
<div>

**(b)** Want $\text{RF}=9$:
$$1 + 2N = 9 \;\Rightarrow\; N = \mathbf{4}$$
Four $3\times3$s match one $9\times9$.

</div>
</div>

<div class="keypoint">

And the four-layer stack is far cheaper: $4\cdot9\,C^2 = 36\,C^2$ params vs $81\,C^2$ for the $9\times9$ — **less than half** — with four non-linearities instead of one. Depth wins on every axis.

</div>

---

<!-- _class: section-divider -->

## Part 5 · A full CNN, end to end (LeNet)

---

# LeNet-5 — the first CNN that worked

LeCun et al. (1998) read handwritten digits on real bank cheques. The template every image classifier still follows: **alternate conv + pool to extract features, then a small MLP head to classify.**

<div class="math-box">

$$\text{image} \xrightarrow{\text{conv}} \text{maps} \xrightarrow{\text{pool}} \xrightarrow{\text{conv}} \text{maps} \xrightarrow{\text{pool}} \xrightarrow{\text{flatten}} \underbrace{\text{FC}\to\text{FC}\to\text{FC}}_{\text{classifier head}} \to \text{softmax}$$

</div>

Convolutions and pooling shrink the $28\times28$ image into a compact stack of features; only *then* does a dense layer decide the digit — on a few hundred numbers, not 150,000.

---

# LeNet in PyTorch

```python
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=3), nn.ReLU(), nn.AvgPool2d(2),   # 28→26→13
            nn.Conv2d(6, 16, kernel_size=3), nn.ReLU(), nn.AvgPool2d(2),  # 13→11→5
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                        # 16·5·5 = 400
            nn.Linear(400, 120), nn.ReLU(),
            nn.Linear(120, 84),  nn.ReLU(),
            nn.Linear(84, 10),                   # 10 digit logits
        )
    def forward(self, x):
        return self.classifier(self.features(x))
```

Every classic CNN is this shape: a `features` stack of conv+pool, then a `classifier` head. *(The original LeNet-5 used $5\times5$ kernels on $32\times32$ inputs; the $3\times3$/$28\times28$ variant here matches the ES 335 notebook.)*

---

# Following the shapes — and where the parameters hide

| Stage | Output shape | Params |
|---|---|---|
| input | $(1,28,28)$ | — |
| Conv$(1\to6,3)$ + pool | $(6,13,13)$ | $60$ |
| Conv$(6\to16,3)$ + pool | $(16,5,5)$ | $880$ |
| flatten | $400$ | — |
| FC $400\to120$ | $120$ | $48{,}120$ |
| FC $120\to84$ | $84$ | $10{,}164$ |
| FC $84\to10$ | $10$ | $850$ |

<div class="insight">

The two conv layers do the heavy *seeing* with under **1,000** parameters; the FC head holds **~98%** of the ~60k total. Lesson: **pool down hard before the dense head**, or one flatten will dominate your whole parameter budget.

</div>

---

# What the layers end up learning

Train the stack and its filters organise themselves **bottom-up** — each layer composing the one below:

![w:720px](figures/lec07/svg/feature_hierarchy.svg)

Conv1 learns oriented edges and colour blobs (Gabor-like) → Conv2–3 learn junctions and textures → deeper layers learn object *parts* → the top learns whole objects. Nobody programmed this hierarchy; **the growing receptive field plus the loss produced it.**

---

<!-- _class: section-divider -->

## Part 6 · The three inductive biases — and when CNNs win

---

# The three biases, baked into every convolution

![w:900px](figures/lec07/svg/inductive_bias.svg)

<div class="columns">
<div>

**1 · Locality**
each output depends on a small window → learn local features first

</div>
<div>

**2 · Translation equivariance**
one filter runs everywhere → position doesn't relearn the feature

</div>
<div>

**3 · Hierarchy of scale**
stacking grows the RF → compose edge → part → object

</div>
</div>

These are not tricks bolted on. They are **assumptions about the world** written into the architecture — and images happen to satisfy all three.

---

# An inductive bias is a free data multiplier

The bias is a *prior*: it tells the network where to look before it sees any data, so it needs far fewer examples.

<div class="columns">
<div>

**Small data ($\le 10^4$ images)**
The CNN's prior does the heavy lifting — it wins easily. A plain MLP barely learns; a Transformer, lacking the prior, is worse still.

</div>
<div>

**Huge data ($\ge 10^8$ images)**
With enough examples a Vision Transformer can *learn* the missing structure and match or beat the CNN (L18). The prior stops being necessary.

</div>
</div>

<div class="keypoint">

**Use a CNN when data is limited** (which is most of the time). Its assumptions are worth an enormous multiplier of labelled examples — they don't reduce true capacity, they shrink the *search space* to hypotheses images actually obey.

</div>

---

# See it all in code

<div class="notebook">

**📓 Notebooks · from one filter to a full classifier**
– `convolution-operation.ipynb`, `convolution-operation-stride.ipynb` — the sliding-window arithmetic and the output-size formula, by hand.
– `cnn-edge.ipynb` — apply the $[1,0,-1]$ edge kernels with `F.conv2d` on a real image; watch edges appear.
– `lenet.ipynb` — build and train the full LeNet on MNIST end to end. *(ML ES 335, N. Batra)*

</div>

<div class="notebook">

**🎛 Interactives** — [convolution-visualizer](https://nipunbatra.github.io/interactive-articles/convolution-visualizer/) (drag a kernel, see the edge), [receptive-field-grower](https://nipunbatra.github.io/interactive-articles/receptive-field-grower/) (watch RF grow with depth), and [equivariant-networks](https://nipunbatra.github.io/interactive-articles/equivariant-networks/). *(Interactive Lab · `interactive/articles/…`)*

</div>

---

<!-- _class: summary-slide -->

# One sentence to remember

<div class="keypoint">

**A convolution is a linear layer that shares one small filter across every position — buying locality, translation equivariance, and a depth-grown receptive field, for ~350,000× fewer parameters, because images obey exactly those symmetries.**

</div>

- **An MLP on pixels blows up** — 616M params and no spatial prior; convolution shares a filter instead.
- **Output size** $\;O=\big\lfloor(W+2P-K)/S\big\rfloor+1\;$ — pad to preserve, stride to downsample.
- **Pooling → invariance:** conv is equivariant; stacked pool/conv makes the classifier care *what*, not *where*.
- **Receptive field grows with depth:** 3 stacked $3\times3$ ≈ one $7\times7$, with fewer params and more non-linearity.
- **Three inductive biases** make CNNs the right tool whenever data is limited.

**Next:** the CNN deep-dive — receptive-field math, $1\times1$ convs, and the LeNet → AlexNet → VGG → ResNet arms race.

---

<!-- _class: compact -->

# Sources & credits

<div class="paper">

This lecture reuses and adapts material from the instructor's own courses (IIT Gandhinagar) and standard references:

- **CNN framing, figures, receptive-field & inductive-bias treatment** — *ES 667 Deep Learning* CNN materials, N. Batra.
- **Convolution / pooling / edge-detection / LeNet notebooks** — *Machine Learning (ES 335)*, N. Batra · `github.com/nipunbatra/ml-teaching` (`convolution-operation.ipynb`, `convolution-operation-stride.ipynb`, `cnn-edge.ipynb`, `lenet.ipynb`).
- **Pedagogical structure (foundations of convolution)** — A. Ng, *Deep Learning Specialization*, **Course 4 (CNNs), Week 1**.
- **LeNet-5** — Y. LeCun, L. Bottou, Y. Bengio, P. Haffner, *Gradient-Based Learning Applied to Document Recognition*, Proc. IEEE, 1998.

Figures adapted from the ES 667 figure library. All source courses © N. Batra & teaching staff.

</div>

---

<!-- _class: section-divider -->

## ⭐⭐⭐ Appendix · optional depth

*Skip on a first pass — for the curious.*

---

<!-- _class: math-heavy -->

# ⭐⭐⭐ Optional · receptive field with stride

With pooling or stride-2 layers, each layer's $(K-1)$ contribution is **magnified by the product of all previous strides** — stride is a *multiplier*, which is why deep nets reach RFs in the hundreds:

$$\text{RF}_{\ell} = \text{RF}_{\ell-1} + (K_\ell - 1)\prod_{i<\ell} S_i$$

Two $3\times3$ convs with a stride-2 pool between them: RF $=1 \to 3 \to (3 + 2\cdot(3-1)) = 7$ — the second conv's reach is *doubled* by the pool's stride. A handful of stride-2 stages is enough to make the final neurons see the whole image, even with tiny $3\times3$ kernels throughout. *(Effective RF (Luo et al., 2016) is smaller and Gaussian-shaped — centre pixels dominate.)*

---

<!-- _class: math-heavy -->

# ⭐⭐⭐ Optional · a convolution *is* a sparse, shared linear layer

A convolution can be written as an ordinary matrix multiply $y = Wx$ where $W$ is a **Toeplitz** matrix: every row is the same filter, shifted, with zeros everywhere the window doesn't reach. So a CNN is genuinely an MLP — one with two constraints on $W$:

- **sparsity** (locality): most entries forced to $0$;
- **tied weights** (sharing): the non-zero entries repeat across rows.

Translation equivariance then falls out as an equation — if $T$ shifts the input, $\;\text{conv}(T x) = T\,\text{conv}(x)$. A fully-connected layer, with an unconstrained $W$, satisfies neither property. **The CNN is the MLP you get by imposing exactly the symmetries images have** — and that is why it needs so few parameters to say so much.

---

<!-- _class: math-heavy -->

# ⭐⭐⭐ Optional · "convolution" vs cross-correlation

What deep-learning libraries call *convolution* — and what every slide today showed — is technically **cross-correlation**: slide the filter and take dot products, *without* flipping it. True mathematical convolution flips the kernel first:

$$(f*g)[n] = \sum_m f[m]\,g[n-m] \qquad\text{vs.}\qquad \text{DL "conv"}[n] = \sum_m f[m]\,g[n+m]$$

<div class="insight">

It makes **no practical difference**: since the kernel is *learned*, the network simply learns the flipped version, so the flip is a relabelling. `torch.nn.Conv2d` and `F.conv2d` both implement cross-correlation. Keep the distinction only for reading signal-processing papers — for building CNNs, "convolution" means the sliding dot product you saw all lecture.

</div>
