# Public Lecture 9 · Modern CNN Pipelines & Transfer Learning · Teaching Guide

*First-time-instructor companion for the canonical `L8B` deck. Audience: undergraduates who have completed the convolution, trainability, and generalization lectures.*

## Public identity and the one-sentence spine

This is **public Lecture 9**, titled **Modern CNN Pipelines & Transfer Learning**. Its canonical file family is `L8B`: the source is `lecture8b/L8b-modern-cnns.typ`, the public artifacts are `slides-pdf/L8B.pdf` and `slides-pdf/L8B-presentation.pdf`, and the exact companion is [`notebooks/L08B/01_depthwise_separable_and_transfer_head.ipynb`](https://colab.research.google.com/github/nipunbatra/dl-teaching/blob/master/notebooks/L08B/01_depthwise_separable_and_transfer_head.ipynb). Do not confuse this guide with `teaching-guides/lec09.md`, which belongs to the following detection lecture and an older numbering convention.

The spine is:

> **A modern CNN is not a catalogue of names. It is a sequence of declared shape, context, route, compute, and reuse decisions; real pretrained features make the reuse hypothesis visible, validation chooses how much to adapt, and only then is the sealed test opened once.**

Keep returning to one deployment contract:

\[
28\times28\times64 \longrightarrow 28\times28\times128,
\qquad \text{RF}\ge 3\times3,
\qquad \text{convolution MACs}<10{,}000{,}000.
\]

The opening question asks which design can satisfy it. The final answer is **B, the depthwise-plus-pointwise factorization**: it preserves the required shape and context while reducing the baseline from $57{,}802{,}752$ to $6{,}874{,}112$ MACs. The other candidates are not “wrong”; they optimize different needs and several also satisfy the budget.

## Where this lecture sits

- **L8 provides the operator.** Students already know channels-last board notation, PyTorch NCHW, cross-correlation, padding/stride geometry, receptive fields, parameter counts, MACs, pooling, and shared-weight gradients. Today the question changes from “how does convolution work?” to “where should a CNN spend context, channel mixing, depth, and compute?”
- **L6 provides the optimization bridge.** Retrieve $y=x+F(x)$ and the direct derivative term $1+F'(x)$. Do not reteach all of trainability; use the residual route as one stage-design decision.
- **L7 provides the experimental protocol.** Training fits parameters, validation selects the probe/unfreeze regime, learning rates, stopping epoch, and checkpoint, and the sealed test is evaluated once after the decision rule is frozen.
- **L10 is the handoff.** Classification globally pools spatial evidence into one answer. Detection retains multi-resolution spatial features and adds heads for classes and boxes at many locations.

Say the delta explicitly near the beginning:

> “L8 taught us to count one convolution. Today we keep one shape and budget fixed while changing the block, then decide how much of a learned backbone to reuse.”

## Non-negotiable conventions

Write these beside the board ledger before calculating anything:

- Every held-stage candidate has output height and width $H_{out}=W_{out}=28$.
- Stride is $1$ and padding is the stated same-padding choice.
- Biases are omitted from the block ledger. The transfer head explicitly includes its bias.
- One MAC means one multiply–accumulate per learned weight per output location.
- The ledger counts convolution MACs. It excludes activation cost, pooling comparisons, additions, normalization, memory traffic, and launch overhead.
- Board tensors use $H\times W\times C$; PyTorch convolution inputs use $N\times C\times H\times W$.
- An addition requires identical height, width, and channels. Concatenation requires identical spatial dimensions and adds channel counts.
- Parameter freezing and module mode are different controls: `requires_grad`, optimizer membership, `train()`/`eval()`, and gradient recording each answer a different question.
- Validation may be consulted repeatedly during model selection. The test set may not.
- A real photograph is **observed data**; pretrained weights and checkpoints are **model artifacts**; activations and predictions are **computed model outputs**; a curve is **measured evidence** only when its protocol and split are named; arrows and layouts are **schematic**. Do not slide between these categories.

If a student produces a number without first naming the output shape, receptive field, and counting convention, ask for those three labels before discussing the arithmetic.

## Exact 80-minute route

The route below follows the deck order and totals exactly **55 minutes core + 15 minutes should-cover + 10 minutes optional**.

| Clock | Priority | Deck move | Instructor action |
|---|---:|---|---|
| 0–3 | Core · 3 | L8/L6/L7 bridge, opening commitment, student contract | Read the deployment contract; collect A/B/C commitments without revealing the answer. |
| 3–8 | Core · 5 | Baseline and ledger convention | Derive $73{,}728$ weights and $57{,}802{,}752$ MACs; mark the standard path over budget. |
| 8–15 | Core · 7 | Small kernels, receptive field, VGG-style stage | Derive $1\to3\to5$ and compare two $3\times3$s with one $5\times5$. |
| 15–23 | Core · 8 | $1\times1$ channel mixing and bottleneck | Compute the per-pixel vector example, then the $64\to16\to16\to128$ candidate. |
| 23–29 | Should · 6 | Inception-style parallel scales | Count four branches and enforce spatial agreement before concatenation. |
| 29–37 | Core · 8 | Residual retrieval and projected shortcut | Compute $F(2)$, $y$, and $dy/dx$; repair the held $64\to128$ channel mismatch with $P(x)$. |
| 37–45 | Core · 8 | Depthwise plus pointwise factorization | Separate spatial filtering from channel mixing and derive the exact $8.41\times$ reduction. |
| 45–50 | Optional · 5 | Groups, scaling, ConvNeXt, measured latency | Treat these as recombinations and deployment caveats, not a second architecture survey. |
| 50–54 | Core · 4 | Complete stage ledger and executed backbone hierarchy | Compare all candidates, then read the actual $64\times56\times56\to128\times28\times28\to512\times7\times7$ pretrained activations as computed model outputs, not saliency. |
| 54–58 | Core · 4 | Six-breed commitment, head, probe/late/all | Bind the $72/36/36$ split, count the $3{,}078$-parameter head, and collect a regime prediction before revealing curves. |
| 58–63 | Should · 5 | Official preprocessing and frozen-state contracts | Follow the actual $600\times400\to384\times256\to224\times224$ image path; separate NCHW, normalization, parameters, buffers, and modes. |
| 63–68 | Core · 5 | Measured curves and validation-only selection | Reconstruct each best validation epoch; validation selects `all` at epoch 13 before the sealed test is opened. |
| 68–72 | Should · 4 | Interpretation, sealed examples, notebook | Distinguish GT from predicted label/probability, state the teaching-subset caveat, then replay the real probe from committed train/validation features. |
| 72–77 | Optional · 5 | FPN teaser and primary provenance | Preserve the L10 cliffhanger; use papers for historical attribution. |
| 77–80 | Core · 3 | Opening revisit, checklist, L10 handoff | Require a ledger-based defense of B and close on “what” versus “where.” |

Priority totals: core $3+5+7+8+8+8+4+4+5+3=55$; should $6+5+4=15$; optional $5+5=10$.

## The held arithmetic ledger

Do not change the held case midway through class. The point is that every design is audited against the same output and MAC convention.

| Candidate | Output | Effective context | Weights | Convolution MACs | Budget |
|---|---:|---:|---:|---:|---:|
| standard $3\times3$, $64\to128$ | $28\times28\times128$ | $3\times3$ | $73{,}728$ | $57{,}802{,}752$ | fail |
| two $3\times3$s, $64\to64\to128$ | $28\times28\times128$ | $5\times5$ | $110{,}592$ | $86{,}704{,}128$ | fail |
| one $5\times5$, $64\to128$ | $28\times28\times128$ | $5\times5$ | $204{,}800$ | $160{,}563{,}200$ | fail |
| bottleneck, $64\to16\to16\to128$ | $28\times28\times128$ | $3\times3$ | $5{,}376$ | $4{,}214{,}784$ | pass |
| shaped Inception candidate | $28\times28\times128$ | up to $5\times5$ | $9{,}920$ | $7{,}777{,}280$ | pass |
| projected residual, narrow branch plus projection | $28\times28\times128$ | $3\times3$ | $10{,}304$ | $8{,}078{,}336$ | pass |
| depthwise $3\times3$ plus pointwise $1\times1$ | $28\times28\times128$ | $3\times3$ | $8{,}768$ | $6{,}874{,}112$ | pass |

The load-bearing expansions are:

\[
\begin{aligned}
\text{standard weights}
  &=3\cdot3\cdot64\cdot128=73{,}728,\\
\text{standard MACs}
  &=28^2\cdot73{,}728=57{,}802{,}752.
\end{aligned}
\]

\[
\begin{aligned}
\text{two }3\times3\text{s}
  &=9(64\cdot64+64\cdot128)=110{,}592,\\
\text{one }5\times5
  &=25\cdot64\cdot128=204{,}800.
\end{aligned}
\]

The two paths have the same nominal $5\times5$ receptive-field extent, but they are not the same function class: the stacked path contains two learned transforms and two activations.

\[
\begin{aligned}
\text{bottleneck weights}
 &=64\cdot16+9\cdot16\cdot16+16\cdot128\\
 &=1{,}024+2{,}304+2{,}048=5{,}376,\\
\text{MACs} &=5{,}376\cdot784=4{,}214{,}784.
\end{aligned}
\]

That candidate clears the budget by $10{,}000{,}000-4{,}214{,}784=5{,}785{,}216$ MACs. The saving comes from shrinking channel-pair work before the $3\times3$, not from changing its spatial window.

For the Inception candidate, count every branch separately:

- $1\times1\to32$: $64\cdot32=2{,}048$ weights.
- $1\times1\to8\to3\times3\to48$: $64\cdot8+9\cdot8\cdot48=3{,}968$.
- $1\times1\to4\to5\times5\to16$: $64\cdot4+25\cdot4\cdot16=1{,}856$.
- pool $\to1\times1\to32$: $64\cdot32=2{,}048$ convolution weights.
- Total: $9{,}920$ weights and $9{,}920\cdot784=7{,}777{,}280$ convolution MACs. Pool comparisons are outside this declared ledger.

All four branch outputs are $28\times28$ spatially; concatenation produces $32+48+16+32=128$ channels. Never add these branch outputs: their channel counts differ.

For the projected residual candidate:

\[
\begin{aligned}
\text{residual branch }F &: 64\cdot8+9\cdot8\cdot8+8\cdot128=2{,}112,\\
\text{projection }P &:64\cdot128=8{,}192,\\
\text{total} &=10{,}304,\\
\text{MACs} &=10{,}304\cdot784=8{,}078{,}336.
\end{aligned}
\]

The projection is required because $28\times28\times64$ cannot be added directly to $28\times28\times128$.

For the factorized answer:

\[
\begin{aligned}
\text{depthwise weights} &=3\cdot3\cdot64=576,\\
\text{pointwise weights} &=64\cdot128=8{,}192,\\
\text{total weights} &=8{,}768,\\
\text{total MACs} &=784(576+8{,}192)=6{,}874{,}112,\\
\text{reduction} &=57{,}802{,}752/6{,}874{,}112\approx8.41\times.
\end{aligned}
\]

The baseline and factorized paths match in output shape, not in values or representational freedom. With depth multiplier one, a single depthwise-plus-pointwise block constrains how spatial filters and cross-channel mixing factor; it is not an arbitrary standard $3\times3$ kernel bank.

## Teach the design-move arc

### 1. Open with a budget, not an architecture name

Read the held contract, show A/B/C, and make every student write a choice plus the first quantity they would calculate. Do not accept “MobileNet” or “ResNet” as a reason. A valid defense names shape, context, and cost.

When the standard calculation appears, separate the two facts:

- L8 weight sharing reduced a dense position-specific map to one shared kernel bank.
- The shared bank still performs $73{,}728$ MACs at each of $784$ output locations, so sharing alone does not guarantee the deployment budget.

Use the line: “We are conducting a controlled search over blocks, not touring a museum of names.”

### 2. Context: stack small kernels deliberately

For stride one, retrieve

\[
r_\ell=r_{\ell-1}+(k-1).
\]

Build $1\to3\to5\to7\to9$ one arrow at a time. Then compare two same-padded $3\times3$s with one same-padded $5\times5$. Both obtain $5\times5$ extent, but the stacked path has an intermediate representation and extra activation. It also uses fewer weights than the direct $5\times5$ in the held channel schedule, yet still exceeds the $10$M budget.

The misconception check “ten $1\times1$s” should produce RF $1\times1$: depth composes channel transformations, while kernel size and stride determine spatial reach.

### 3. Channels: make $1\times1$ concrete before calling it a bottleneck

At one pixel use

\[
x=(2,-1,0.5)^\top,\quad
W=\begin{pmatrix}1&-1&0\\0.5&0&2\end{pmatrix},\quad
b=(0.5,-1)^\top.
\]

Then

\[
Wx+b=(3.5,1)^\top.
\]

Stress that the same matrix acts at every spatial coordinate. A $1\times1$ convolution mixes channels but reads no neighbour. Only after this numeric should you introduce the $64\to16\to16\to128$ bottleneck and ask where the expensive $3\times3$ now operates.

### 4. Branches: parallel scale is not sequential depth

Keep this in the should-cover route. Every branch must preserve $28\times28$ before concatenation. Ask students to name the appropriate padding for the $3\times3$, $5\times5$, and $3\times3$ pool: $1$, $2$, and $1$ at stride one.

The decisive distinction is:

- **Concatenation:** same $H,W$, channel counts add.
- **Addition:** same $H,W,C$, values combine elementwise.

Inception widens the alternatives available at one stage. Stacking modules supplies sequential depth.

### 5. Residuals: retrieve the route, then repair its shape

Use the exact scalar from L6:

\[
F(x)=0.1x^2,\quad x=2,\quad F(2)=0.4,\quad y=2.4,
\quad \frac{dy}{dx}=1+0.2x=1.4.
\]

Say “direct derivative term,” not “guaranteed gradient.” The $1$ helps signal flow, but $1+F'(x)$ can still be small through cancellation and an entire network has many interacting routes.

Then return to the held stage: the $64$-channel input and $128$-channel residual output make an identity addition illegal. A $1\times1$, stride-one projection maps the shortcut to $128$ channels. The shortcut is no longer parameter-free, but the addition becomes shape-legal and the candidate stays under budget.

### 6. Efficiency: factor spatial filtering from channel mixing

Write the standard factorization of cost:

\[
k^2C_{in}C_{out}
=\underbrace{k^2}_{\text{spatial positions}}
\times
\underbrace{C_{in}C_{out}}_{\text{channel pairs}}.
\]

Depthwise convolution assigns one $3\times3$ spatial filter to each of $64$ input channels, producing $64$ channels. Pointwise convolution then mixes those $64$ values into $128$ outputs at each location. Students should predict both intermediate shapes before running code.

Keep the systems caveat attached: lower MAC count is a useful algorithmic comparison, but measured latency also depends on tensor layout, memory traffic, batch size, kernel launches, fusion, and device support.

### 7. Move from one stage to a reusable representation

The complete ledger should make the pedagogical claim visible: there is no universally best block. A bottleneck is attractive when channel-pair cost dominates; branching when several scales matter; a residual route when preservation and optimization matter; depthwise separation when mobile-style factorization fits the device.

Repeated stages form a backbone. The held classifier ends with a $7\times7\times512$ feature map, global average pooling to $h\in\mathbb{R}^{512}$, and a task head.

### 8. Transfer: start with the smallest falsifiable hypothesis

For six new classes,

\[
W\in\mathbb{R}^{6\times512},\qquad b\in\mathbb{R}^{6},
\qquad 6\cdot512+6=3{,}078.
\]

Against an $11$M-parameter backbone, the probe trains

\[
\frac{3{,}078}{11{,}000{,}000+3{,}078}\times100\%\approx0.028\%.
\]

A linear probe still computes the backbone forward pass. “Frozen” means its parameters are not updated; it does not mean the backbone is skipped. A probe tests whether **pretrained** features are linearly useful for the new labels. A randomly initialized tiny network can test only mechanics.

## Transfer state and split contracts

### Frozen weights versus frozen behaviour

There are four separate switches:

1. `parameter.requires_grad = False` prevents parameter-gradient storage through that parameter.
2. Excluding frozen parameters from the optimizer makes the update contract explicit.
3. `module.train()` and `module.eval()` control stateful/stochastic layer behaviour.
4. `torch.no_grad()` controls autograd recording; it does not change BatchNorm or Dropout mode.

BatchNorm running means and variances are buffers, not trainable parameters. They can update during a forward pass in training mode even when every backbone parameter has `requires_grad=False`. Dropout likewise remains stochastic in training mode. For a truly fixed probe, keep the backbone in `eval()` while the head trains.

One PyTorch trap deserves a live warning: `model.train()` recursively puts all descendants back into training mode. A probe loop that calls it at the start of each epoch must then call `model.backbone.eval()` again.

Deliberately updating BatchNorm statistics on target data can be an experiment, but it is not the same intervention as a fixed probe. Name it, keep it separate, and let validation decide.

### Validation controls adaptation; test remains sealed

Use the split clock exactly:

- **Train:** fit parameters for one declared regime.
- **Validation:** choose probe versus late/all unfreezing, optimizer groups and rates, augmentation, stopping epoch, and checkpoint.
- **Test:** run once after the entire procedure and checkpoint are frozen.

Repeated test inspection while choosing unfreeze depth converts the test set into validation data. Do not report a “best test epoch.” Report the test metric of the single validation-selected checkpoint.

The fair comparison holds seed, train/validation split, head initialization, augmentation, optimizer family, epoch budget, and selection metric fixed while changing one regime:

| Regime | Trainable modules | Backbone learning rate | Selection evidence |
|---|---|---:|---|
| probe | head only | $0$ | best validation metric and epoch |
| late | head + final stage | small | same validation metric and budget |
| all | entire model | smaller still | same validation metric and budget |

Unfreeze in response to evidence:

- Poor training and validation performance with a small gap suggests the fixed representation may be insufficient; try one late stage.
- If late unfreezing improves both under the same budget, it is evidence for task-specific adaptation.
- High training performance and poor validation performance is not a reason to unfreeze more. First suspect data scarcity, regularization, augmentation, or shift.
- Broader unfreezing is another validation-controlled candidate, never an automatic next step.

Use explicit optimizer groups with $0<\eta_b<\eta_h$: the random head often needs a larger step; reused backbone features should be perturbed more cautiously. The ratio is a hyperparameter, not a law.

## The real six-breed evidence case

The transfer section now uses actual Oxford-IIIT Pet images and the cached torchvision ResNet-18 `IMAGENET1K_V1` checkpoint. It is a **small fixed teaching subset, not a benchmark**. No displayed image is generated. Keep four evidence types verbally distinct:

| Label | What appears in this lecture | What it can support |
|---|---|---|
| **Observed** | real `Abyssinian_1.jpg`; official resize, center crop, and normalization applied to it; dataset labels | what the declared input and ground-truth contracts contain |
| **Model artifact** | exact pretrained and validation-selected weight checkpoints | which fixed learned state produced the displayed outputs |
| **Computed model output** | pretrained layer activations; predicted breed and probability | what a checkpoint computed; activation tiles are not saliency or causal attribution |
| **Measured** | fixed-protocol train/validation histories and one post-selection test pass | comparison under this exact subset, seed, optimizer, budget, and selection rule |
| **Schematic** | pipeline arrows, tensor boxes, regime decision route | conceptual structure only; never an empirical result |

### Data, model, and preprocessing contract

- Six cat breeds: Abyssinian, Bengal, Birman, Bombay, British Shorthair, and Egyptian Mau.
- Seed `20260812`; selection is locked by `SHA256('20260812:<official_split>:<image_id>')`, then bound by filename, dimensions, and JPEG SHA-256 in `selection-manifest.csv`.
- Training: 12 images per breed, $72$ total, from Oxford's official `trainval` split.
- Validation: 6 disjoint images per breed, $36$ total, also from official `trainval`.
- Sealed test: 6 images per breed, $36$ total, from the official `test` split.
- Backbone: torchvision ResNet-18 `IMAGENET1K_V1`; checkpoint SHA-256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
- Official transform: resize short side to 256 using bilinear interpolation, center-crop to $224\times224$, tensorize RGB, then normalize with mean $(0.485,0.456,0.406)$ and standard deviation $(0.229,0.224,0.225)$. No augmentation.

For the shared $600\times400$ photograph, the executed path is

\[
600\times400\ \text{RGB}
\longrightarrow 384\times256
\longrightarrow 224\times224
\longrightarrow 3\times224\times224.
\]

Its observed post-normalization channel ranges are R $[-2.101,1.992]$, G $[-1.966,1.851]$, and B $[-1.804,2.135]$. These are measurements on this photograph, not universal bounds. The actual pretrained activation shapes are

\[
\text{layer1}:64\times56\times56,
\qquad
\text{layer2}:128\times28\times28,
\qquad
\text{layer4}:512\times7\times7.
\]

The displayed eight channels at each stage are ranked by mean absolute activation and each tile is rescaled by its own 1st–99th percentile. Say explicitly: **computed activations, not saliency**. Their spatial shrinkage and channel growth are inspectable; their display intensity cannot be compared across tiles.

### Fixed comparison and exact result

All three regimes use 15 epochs, batch size 12, AdamW with weight decay $10^{-4}$, the same copied head initialization, the same minibatch order, fixed BatchNorm buffers, and the same preprocessing and split. The head learning rate is $3\times10^{-3}$ in every regime; `layer4` uses $3\times10^{-4}$ in `late`, and the full backbone uses $3\times10^{-5}$ in `all`.

Within a regime, choose maximum validation accuracy, then lower validation loss, then earlier epoch. Across regimes, choose maximum validation accuracy, then fewer trainable parameters, lower validation loss, and earlier epoch. This parsimony tie-break is declared before opening test.

| Regime | Trainable parameters | Validation-selected epoch | Best validation | What happened |
|---|---:|---:|---:|---|
| probe | $3{,}078$ | 14 | $33/36=91.7\%$ | pretrained fixed features were already strongly linearly useful |
| late (`layer4` + head) | $8{,}396{,}806$ | 4 | $27/36=75.0\%$ | worse under this exact short, tiny-data protocol |
| all | $11{,}179{,}590$ | 13 | $34/36=94.4\%$ | validation winner; selected before test was opened |

Only the validation-selected `all`, epoch-13 checkpoint was evaluated on the sealed subset: $33/36=91.7\%$, cross-entropy $0.168423$, one evaluation. The six displayed post-selection examples pair **GT labels** with **computed predictions and probabilities** from the selected model artifact.

The defensible conclusion is narrow: pretrained features were useful for these six breeds, and validation selected full adaptation under this exact fixed protocol. Do not claim that full fine-tuning is generally best, that late adaptation is intrinsically weak, or that $91.7\%$ is Oxford-IIIT Pet benchmark performance. The subset is small and its initial selection population was limited to JPEGs available when the manifest was bootstrapped.

### Slide-by-slide visual choreography

Use the physical PDF page numbers below; the printed footer is one lower because the title page is unnumbered. The presentation build reveals commitments and answers across states but preserves this order.

| Handout page | Evidence label | Instructor move |
|---:|---|---|
| 44 | **OBSERVED + COMPUTED** | Start from the real center crop, name the pretrained checkpoint as the model artifact, then trace its computed activations through layer1, layer2, and layer4. Say “not saliency” before interpreting the hierarchy. |
| 45 | **GT + OBSERVED CONTRACT** | Name all six ground-truth breeds and the balanced $72/36/36$ split. Students commit to probe/late/all before seeing any curve. |
| 49 | **OBSERVED** | Walk $600\times400\to384\times256\to224\times224\to(1,3,224,224)$ and identify exactly where normalization enters. |
| 51 | **MEASURED** | Reveal probe $91.7\%@14$, late $75.0\%@4$, and all $94.4\%@13$; apply the stated validation rule, then reveal the single sealed $33/36$ result. |
| 52 | **MEASURED INTERPRETATION** | Ask for one supported claim and one unsupported claim. Protect “small fixed teaching subset, not a benchmark.” |
| 55 | **GT + COMPUTED MODEL OUTPUT** | Compare true breed with predicted breed/probability on six post-selection examples; follow the audit trail to the notebook and committed artifacts. |
| 63 | **PROVENANCE** | Point to Oxford licensing/splits, torchvision weights, checkpoint/manifest/builder/results hashes, and the retained build environment. |

## Exact notebook choreography

Open the direct [`L08B` Colab](https://colab.research.google.com/github/nipunbatra/dl-teaching/blob/master/notebooks/L08B/01_depthwise_separable_and_transfer_head.ipynb). It has two explicit lanes. **Lane A** uses seeded synthetic tensors to isolate mechanics. **Lane B** verifies and inspects the committed real-evidence bundle, then replays only the genuine linear probe from cached train/validation features. In a repository checkout it is fully local. Direct Colab fetches only nine SHA-pinned committed artifacts totaling about $0.91$ MiB; it does not download Oxford archives or ResNet weights and does not invoke the builder.

The executed notebook contains 13 code checkpoints: eight mechanics checkpoints in Lane A and five evidence checkpoints in Lane B. Keep the predict-before-run rhythm:

| Checkpoint | Ask before running | Evidence to inspect |
|---|---|---|
| 1 · setup | Is the depthwise reduction larger than the crude $C_{out}/k^2$ comparison? | Environment and seed are printed. |
| 2 · cost | Compute all four weight counts and both MAC totals. | $73{,}728$, $576$, $8{,}192$, $8{,}768$; $57{,}802{,}752$ versus $6{,}874{,}112$; ratio $8.409\times$. |
| 3 · node paths | What are the depthwise and final NCHW shapes? Must the values match? | Depthwise $(2,64,28,28)$; both final paths $(2,128,28,28)$; independently initialized values need not match. |
| 4 · residual scalar | Predict $F(2)$, $y$, and $dy/dx$. | $0.4$, $2.4$, and $1.4$. |
| 5 · identity | What happens when $\alpha=0$? | Output equals input; gradient is all ones. |
| 6 · held head | Predict feature/logit shapes and trainable count. | Features $(2,512)$; logits $(2,6)$; exactly $3{,}078$ trainable head parameters; held lecture fraction $0.028\%$. |
| 7 · frozen state | Which parameter and buffer changes should be zero? | Frozen BN buffer changes in `train()` ($0.011999$); proper probe gives backbone parameter change $0$, BN change $0$, head change $0.004505$, backbone gradients `None`. |
| 8 · progressive unfreeze | Does the illustrative validation trace trigger adaptation, and which group uses the smaller rate? | Improvement $0.001$, gap $0.029$, target unmet; unfreeze late; rates $0.01$ backbone and $0.10$ head; stem change $0$, late/head changes positive. |

The final validation trace is labeled illustrative. It is a deterministic control input for the decision rule, not evidence generated by the random TinyBackbone. Make students say this distinction aloud.

If live time is short, run checkpoints 2, 3, 6, 7, and 8. Never skip the prediction on the BatchNorm buffer: it exposes the most common silent “frozen backbone” error.

Then use the five executed Lane B checkpoints as a reveal sequence, not a block of output:

| Checkpoint | Ask before running | Executed evidence to inspect |
|---|---|---|
| 9 · provenance gate | What must be verified before trusting a retained result? | 9 artifact hashes, `results.json` SHA `4e22a95b...47e23`, model checkpoint SHA `f37072fd...e07ec`, builder SHA `9fa63046...6da52`. |
| 10 · split audit | Why are there 144 manifest rows but only 108 cached features? | $72/36/36$ balanced split; feature array $(108,512)$ contains train + validation and explicitly no test. |
| 11 · image contract | Predict resized/crop and layer shapes. | $600\times400\to384\times256\to224\times224$; NCHW input $(1,3,224,224)$; activations $(64,56,56)$, $(128,28,28)$, $(512,7,7)$. |
| 12 · validation selection | Which regime wins before test is read? | probe $33/36$ at epoch 14; late $27/36$ at 4; all $34/36$ at 13; validation selects `all`; then inspect one sealed $33/36$ pass. |
| 13 · probe replay | What can cached frozen features reproduce, and what must remain builder evidence? | exact `Linear(512,6)` replay selects epoch 14 and $33/36$ validation; no `late`/`all` replay and no test evaluation. |

For an 80-minute live lecture, show retained outputs for Lane A checkpoints 2, 3, 6, and 7; execute Lane B checkpoints 10–12; assign the probe replay for follow-along. The critical pedagogical rhythm is: **commit to probe/late/all → audit split/preprocessing → reveal validation curves → announce selected regime → open the already-recorded sealed result once**.

## Recurring diagnostics: symptom → suspect → smallest test

| Symptom | Suspect | Smallest discriminating test |
|---|---|---|
| weight count is correct but MACs are too small by $784\times$ | output locations were omitted | print `H_out * W_out` separately, then multiply the weight count |
| residual addition fails | shortcut and residual shapes differ | print both named $(N,C,H,W)$ tuples immediately before `+` |
| Inception concatenation fails | one branch changed height or width | assert every branch's `shape[-2:]` before `cat(dim=1)` |
| depthwise output has the wrong channel count | `groups`, input channels, or depth multiplier is wrong | print `in_channels`, `out_channels`, and `groups`; for multiplier one they are all $C_{in}$ |
| loss remains near random after head training | preprocessing, label, or input-layout contract mismatch | inspect one batch: NCHW shape, dtype, range, channel order, normalization, labels |
| supposedly frozen features drift | BatchNorm buffers or Dropout remain active | diff both parameters and buffers across one forward/step; print module modes |
| probe underfits both train and validation | fixed features may not separate target labels | unfreeze one late block under the same split, seed, budget, and metric |
| validation falls immediately after unfreezing | backbone rate is too large or useful features were disturbed | reduce $\eta_b$ and log layer-wise update norms |
| training rises while validation worsens | generalization gap, not insufficient capacity | inspect gap, augmentation, regularization, and data quality before unfreezing more |
| MACs fall but latency does not | memory traffic, launch overhead, layout, or unsupported kernels | benchmark the actual device and batch size after warm-up |
| repeated test scores influence the next run | test leakage | reconstruct the decision log; move all selection back to validation and reseal test |

Use one diagnostic sentence repeatedly:

> “Name the symptom, list at least two suspects, then run the cheapest test that separates them.”

## Subtleties and likely misconceptions

- **“The factorized path reproduces the standard path.”** No. It reproduces the output *shape* and requested spatial context. Independent weights give different values, and one depthwise-plus-pointwise block is structurally constrained relative to an arbitrary standard kernel bank.
- **“Two $3\times3$s equal one $5\times5$.”** Their receptive-field extents match under the stated stride/padding, but their transformations do not: the stack has an intermediate representation and another activation.
- **“A $1\times1$ convolution sees one channel.”** It sees one spatial location and all input channels. Its matrix has shape $C_{out}\times C_{in}$.
- **“The bottleneck shrinks the $3\times3$ window.”** It shrinks the number of channel pairs processed inside that window.
- **“Branches make a block deeper.”** Parallel branches add alternative transforms at one stage. Sequential modules add depth.
- **“Concatenation and residual addition are interchangeable.”** Concatenation preserves all branch channels; addition requires exactly matching shapes and combines values elementwise.
- **“A shortcut skips learning.”** The learned branch still contributes $F(x)$. The shortcut changes the parameterization by making preservation easy.
- **“Residuals guarantee non-vanishing gradients.”** They introduce a direct derivative term; they do not guarantee that all terms remain large or cannot cancel.
- **“A projection shortcut is an identity.”** It is learned and changes shape. Call it a projection route.
- **“Depthwise means `groups=C_in` for any output count.”** With depth multiplier one, `in_channels=out_channels=groups=C_in`. Other depth multipliers repeat filters per input channel; a pointwise layer normally supplies arbitrary output mixing.
- **“Fewer MACs always means faster.”** MACs omit memory and hardware effects. Benchmark.
- **“`requires_grad=False` freezes BatchNorm.”** It freezes trainable parameters, not running-statistic buffers or training-mode behaviour.
- **“`no_grad()` is the same as `eval()`.”** One disables graph recording; the other changes module behaviour.
- **“The random TinyBackbone proves transfer works.”** It proves mechanics only. Transfer quality needs a pretrained representation and held-out validation evidence.
- **“Bright activation tiles show which pixels caused the prediction.”** No. They are selected feature channels, each rescaled independently for display; the figure is not saliency or attribution.
- **“The ground-truth and predicted labels on the contact sheet are the same evidence type.”** No. GT comes from the dataset; predicted breed and probability are computed outputs from the selected checkpoint artifact.
- **“Because `all` won here, full fine-tuning is generally best.”** No. Validation selected it for one tiny fixed subset and protocol. The result does not rank regimes universally.
- **“The notebook can replay every measured curve from the cached features.”** No. Frozen features reproduce the probe only. Replaying `late` or `all` requires pixels and the checkpoint because the representation changes.
- **“Unfreeze when the probe has trained for N epochs.”** Unfreeze because a predeclared validation rule identifies underfitting under the fixed protocol.
- **“Use the test score to decide how much to unfreeze.”** That leaks test information. Validation selects; test reports once.

## If a student asks

**Why not always use the cheapest block?**

Because the contract includes more than MACs. A task may need wider context, parallel scales, easier optimization, or a kernel supported efficiently by its device. First satisfy shape and representation needs; then compare the resource that actually constrains deployment.

**Why can a bottleneck have fewer weights than a depthwise-separable block here?**

They impose different factorizations. The held bottleneck spends its $3\times3$ at width $16$, while the depthwise path filters all $64$ channels and then mixes $64\to128$. The bottleneck is cheaper in this ledger; the depthwise path is the answer to the opening because option B is the stated spatial/channel factorization and meets every contract. Neither ordering is universal.

**Why put a $1\times1$ after depthwise convolution?**

Depthwise filtering does not mix channels. The pointwise layer lets each output combine the spatially filtered input channels.

**Does global average pooling destroy localization?**

It deliberately discards where each feature fired to produce one vector for classification. That is why L10 returns to spatial feature maps and multi-scale heads for detection.

**If the backbone is frozen, why does the forward pass still cost time?**

Its activations are still computed. Freezing reduces gradient storage and optimizer updates, not inference compute.

**May I update BatchNorm statistics while freezing weights?**

Yes, as a named target-domain adaptation intervention. It is not a fixed linear probe, so validate it as a separate regime.

**Why use a smaller backbone learning rate?**

The new head starts random; the backbone may already encode useful features. Smaller backbone steps test adaptation without immediately overwriting that representation. The ratio is selected on validation.

**When should I unfreeze everything?**

Only when a broader-unfreeze candidate is justified by data volume/domain shift and wins under the same validation protocol. It is not the default final stage of every transfer run.

**Should I cache frozen backbone features?**

It can speed a strict fixed probe when preprocessing is deterministic and the backbone remains in evaluation mode. Do not cache through random augmentation if each epoch is supposed to see new views, and rebuild features after any backbone or preprocessing change.

**Are parameters or MACs the deployment objective?**

They are useful proxies. Deployment may instead be limited by latency, memory, energy, model size, or throughput, so measure the actual constraint on the target system.

## If you are short on time

The **55-minute short route is exactly the deck's core**, not an improvised catalogue:

1. Keep the L8/L6/L7 bridge, opening commitment, and ledger convention.
2. Compute the standard baseline.
3. Show the receptive-field reason for small kernels, but use only the two-$3\times3$ count.
4. Do the $1\times1$ pixel numeric and bottleneck count.
5. Do the residual scalar, shape mismatch, and projection count.
6. Derive depthwise and pointwise weights/MACs and answer the opening.
7. Count the $512\to6$ head; state the probe/validation/sealed-test protocol.
8. Revisit B, use the final checklist, and make the L10 “what versus where” handoff.

Cut the full Inception branch ledger, grouped convolution, scaling, ConvNeXt, FPN mechanics, and provenance slide. Compress the notebook to its printed factorization, held-head, BatchNorm-state, and unfreeze outputs. Even on the short route, do not drop the no-bias/MAC convention, residual shape legality, random-backbone caveat, validation-controlled selection, or one-time test rule.

If only **35 minutes** remain unexpectedly, preserve: opening contract (3), baseline and factorized arithmetic (9), bottleneck (5), residual scalar plus projection legality (5), $3{,}078$ transfer head and state warning (6), validation/test protocol (4), revisit and L10 handoff (3). Announce that context variants and historical architectures are assigned reading rather than pretending they were covered.

## Before-class checklist

- Open the **presentation build**, not the compact handout, so questions precede answers.
- Verify the public Lecture 9 row links `L8B.pdf`, `L8B-presentation.pdf`, and the exact direct Colab.
- Put the held contract and blank ledger on the board before students enter.
- Pre-open the notebook at the cost cell and the frozen-state cell; confirm its outputs are retained.
- Confirm the nine Lane B artifact hashes pass, the manifest reports $72/36/36$, and the feature bundle reports $(108,512)$ with no test rows.
- Rehearse the visual taxonomy: photograph/preprocessing = observed; weights/checkpoints = model artifacts; activations/predictions = computed model outputs; curves = measured; arrows = schematic.
- Decide whether the should-cover Inception and state-diagnostic segments fit this cohort; protect the 55-minute core.
- Keep a separate train/validation/test diagram visible during transfer discussion.
- If demonstrating hardware latency, warm up the device and name batch size, dtype, and target hardware. Do not compare a cold first run with a warmed run.

## Primary sources and exact companion

- Simonyan and Zisserman, [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) — repeated small-kernel stages.
- Szegedy et al., [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842) — multi-branch Inception modules and channel reductions.
- He et al., [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) — residual parameterization and projection shortcuts.
- Howard et al., [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861) — depthwise-separable convolution.
- Tan and Le, [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) — compound scaling.
- Liu et al., [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545) — ConvNeXt's recombination of modern block choices.
- Lin et al., [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144) — the optional L10 bridge.
- PyTorch, [`torch.nn.Conv2d`](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) and [`torch.nn.BatchNorm2d`](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html) — grouped/depthwise shapes and running-statistic behaviour.
- Exact executed [`L08B` Colab notebook](https://colab.research.google.com/github/nipunbatra/dl-teaching/blob/master/notebooks/L08B/01_depthwise_separable_and_transfer_head.ipynb) — deterministic factorization, residual, state, and optimizer-group mechanics.
- Oxford-IIIT Pet, [dataset page](https://www.robots.ox.ac.uk/~vgg/data/pets/) — source, official split boundary, and CC BY-SA 4.0 licensing for the real six-breed evidence case.
- Torchvision, [`ResNet18_Weights.IMAGENET1K_V1`](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) — exact pretrained weights and preprocessing contract; the used checkpoint hash is `f37072fd…e07ec`.
- Committed `shared/vision-evidence/oxford-iiit-pet/l8b/` evidence bundle — manifest, builder, full histories, train/validation-only features, test predictions, retained visuals, hashes, and environment provenance.

Use the papers for historical claims and design provenance. Use the held ledger and Lane A for causal mechanics. Use the manifest-bound Lane B evidence only for its narrow measured claims.

## Closing line

> **“The backbone answers what evidence to preserve; the block ledger tells us what it costs; validation tells us how much to adapt. Next, detection keeps the backbone but refuses to pool away where the evidence occurred.”**
