# Lecture 8 · Convolutional Neural Networks · Teaching Guide
*First-time-instructor companion. Audience: undergraduates who have completed the public Lectures 1–7 and previously encountered basic convolution mechanics in ES 335.*

## The spine (say this in two sentences)

Lecture 7 changed the training recipe while protecting train, validation, and test roles. Lecture 8 keeps that protocol fixed and changes the hypothesis class: local connections decide what can interact, shared weights ask the same learned question everywhere, and depth composes those spatial responses into a prediction.

The line to repeat is:

> **One kernel is one learned local question. A convolution asks it at every legal location.**

## Where it sits

**Public Lecture 7 bridge.** Lecture 7 ended with the claim that architectural inductive bias can make useful rules easier to express. Open Lecture 8 with the exact delta:

- the supervised loss, optimizer, split protocol, and sealed test stay fixed;
- the new lever is the architecture;
- the image assumptions are locality, repeated structure, and compositional hierarchy.

Do not imply that flattening deletes pixels. It preserves every value but presents the following dense layer with no built-in neighbourhood or sharing constraint.

**Public Lecture 9 handoff.** The next public lecture is **Modern CNN Pipelines & Transfer Learning**, represented by the `L8B` deck/PDF family. Lecture 8 builds and audits one small CNN from exact arithmetic. The next lecture asks how VGG, ResNet, efficient blocks, pretrained backbones, and transfer learning scale and reuse that local shared operator.

Close with:

> **Today we learned the local shared operator; next we study the modern pipelines that compose and reuse it.**

## Learning contract

By the end, students should be able to:

1. explain what locality and weight sharing assume, and what arbitrary position-specific connectivity they give up;
2. distinguish library cross-correlation from flipped-kernel mathematical convolution;
3. calculate one feature-map entry and the complete feature map from a displayed image and kernel;
4. name every spatial and channel axis before applying a convolution;
5. derive output size by counting legal starts, including padding, stride, and the floor;
6. calculate convolutional parameter counts without inserting image height or width;
7. distinguish translation equivariance from invariance and state the boundary/stride caveats;
8. track receptive-field width and jump through convolution and pooling;
9. audit the output shapes, parameters, MACs, and receptive fields of one tiny CNN;
10. explain why a shared kernel gradient sums contributions over examples and spatial positions;
11. translate board-order $H\times W\times C$ tensors into PyTorch $N\times C\times H\times W$ tensors;
12. diagnose a CNN failure with a tensor, state, one-batch, or split check before tuning the optimizer.

## The persistent board case

Begin with the real Oxford-IIIT Pet photograph, then state the bridge precisely.
The zero-based photo crop $x\in[328,373)$, $y\in[120,165)$ is $45\times45$
RGB. Convert it to grayscale, split its rows into five consecutive nine-row
bands, and average each band on the raw 0–255 scale:

\[
(113.83,112.37,145.50,145.78,109.32).
\]

Threshold at 128 to obtain $(0,0,1,1,0)$, then repeat each bit across five
columns. This is a **declared pedagogical quantization of an observed crop**,
not a model output or an assertion that the photograph is binary. Keep the
resulting exact image and kernel visible from the first convolution calculation
through geometry, pooling, and receptive field:

\[
X=
\begin{bmatrix}
0&0&0&0&0\\
0&0&0&0&0\\
1&1&1&1&1\\
1&1&1&1&1\\
0&0&0&0&0
\end{bmatrix},
\qquad
K=
\begin{bmatrix}
1&1&1\\
0&0&0\\
-1&-1&-1
\end{bmatrix}.
\]

The kernel compares the row above with the row below. With library-style cross-correlation, no padding, and stride $1$:

\[
Y=
\begin{bmatrix}
-3&-3&-3\\
-3&-3&-3\\
3&3&3
\end{bmatrix}.
\]

The same pair supports three geometry settings:

| padding $p$ | stride $s$ | output shape | exact output |
|---:|---:|---:|---|
| $0$ | $1$ | $3\times3$ | $Y$ above |
| $1$ | $1$ | $5\times5$ | border magnitudes are $0$ or $2$; interior magnitudes are $3$ |
| $1$ | $2$ | $3\times3$ | rows $(0,0,0)$, $(-2,-3,-2)$, $(2,3,2)$ |

Pooling stays on the same trace. A $2\times2$ max pool with stride $1$ maps $Y$ to

\[
P=
\begin{bmatrix}
-3&-3\\
3&3
\end{bmatrix},
\qquad
\operatorname{GAP}(P)=0.
\]

The trace then scales into one tiny classifier:

\[
N\times3\times32\times32
\rightarrow N\times16\times32\times32
\rightarrow N\times16\times16\times16
\rightarrow N\times32\times16\times16
\rightarrow N\times32
\rightarrow N\times10.
\]

Use this as a ledger rather than a sequence of unrelated examples. Every layer must answer four questions: **output shape, learned parameters, stated MAC convention, and receptive field/jump**.

## Exact 80-minute route

### ⭐ Core · exactly 55 minutes

- L7 bridge, opening commitment, locality/sharing, and like-for-like dense comparison: 10 min.
- Real crop → declared $5\times5$ teaching construction, fixed $3\times3$ convolution, first response, full map, and sharing graph: 10 min.
- Multi-channel support, tensor axes, and the $73{,}856$-parameter checkpoint: 6 min.
- Padding, stride, legal starts, and floor: 7 min.
- Exact equivariance proof, computed real Sobel/shift evidence, boundary caveat, and pooling: 7 min.
- Receptive-field recursion and the complete tiny-CNN ledger: 10 min.
- Opening revisit, retrieval, and the Modern-CNN handoff: 5 min.

### ⭐⭐ Should cover · exactly 15 minutes

- Shared-weight forward and backward numeric, including kernel/input/bias accumulation: 9 min.
- HWC→NCHW, standardization/augmentation/Dropout/BatchNorm/autograd state contracts, and one targeted diagnostic: 5 min.
- Prediction-first notebook choreography: 1 min.

### ⭐⭐⭐ Optional · exactly 10 minutes

- Two $3\times3$ layers versus one $5\times5$ layer: 4 min.
- `im2col`, matrix multiplication, and the distinction between duplicated patch entries and shared parameters: 3 min.
- Primary-source provenance or an extended notebook trace: 3 min.

The labels sum to **55 core + 15 should-cover + 10 optional = 80 minutes**. The closing five minutes are core even though they occur after the should-cover section in deck order.

### Suggested clock

| Time | Route | Teaching move |
|---|---|---|
| 0–5 | ⭐ | Reuse the L7 handoff, state what remains fixed, and collect the ten-parameter commitment. |
| 5–12 | ⭐ | Compare the position-specific dense map with the local shared map, including all biases. |
| 12–22 | ⭐ | Show real crop → band means → threshold → $X$; predict $Y[0,0]$, reveal $-3$, and build the exact map. |
| 22–29 | ⭐ | Add channels, name tensor axes, and calculate $73{,}856$ parameters. |
| 29–36 | ⭐ | Count legal starts; separate padding from stride; make the floor unavoidable. |
| 36–43 | ⭐ | Compute the shifted 1-D proof, inspect the real Sobel/+12-pixel/pooling composite, state the boundary caveat, and pool $Y$. |
| 43–50 | ⭐ | Derive $(r,j)$, read theoretical support boxes on the photo, and complete the tiny-CNN ledger. |
| 50–59 | ⭐⭐ | Work the shared-kernel and overlapping-input gradients completely. |
| 59–64 | ⭐⭐ | Translate HWC→NCHW, separate training-state contracts, and run one diagnostic. |
| 64–65 | ⭐⭐ | Launch the single Colab with the six prediction checkpoints visible. |
| 65–70 | ⭐ | Revisit answer B, complete retrieval, and hand off to the `L8B`/Modern-CNN lecture. |
| 70–80 | ⭐⭐⭐ | Compare small/large kernels, show `im2col`, and locate the supporting sources. |

If calculation takes longer, omit the entire optional appendix. Never save time by silently dropping the cross-correlation convention, the floor, the HWC/NCHW translation, or the conditions on equivariance.

## Teach it like this

The lecture is one accumulating audit, not a catalogue of CNN facts:

1. **Name the architectural bet.** Nearby pixels interact, useful detectors repeat across position, and depth composes local evidence.
2. **Separate values from constraints.** Flattening retains values but does not impose image geometry on the next dense layer.
3. **Make the abstraction contract visible.** Derive $X$ from the declared real crop, call it a teaching construction, then use fixed $K$ at all nine legal starts.
4. **Count three different things separately.** Shape counts legal positions; parameters count learned scalars; MACs count repeated work.
5. **Audit movement.** The response moves with the input in the common interior, while finite boundaries and stride limit exact equivariance.
6. **Track context backward.** Receptive field and jump explain what one later unit can depend on.
7. **Scale without changing the arithmetic.** The tiny classifier is just the same operator with channel banks, batches, pooling, GAP, and a linear head.
8. **Differentiate the sharing.** Multiple forward uses of one kernel become a sum of backward contributions.
9. **Protect implementation contracts.** Axis order, preprocessing statistics, module mode, and autograd state are separate decisions.
10. **Return to the commitment.** Ten parameters produce 1,024 responses by reuse, not by representing an arbitrary dense map.

Use **commit → calculate → reveal → explain** throughout:

- “Can ten parameters produce 1,024 values?” before the local/shared comparison.
- “What sign should the first edge response have?” before multiplying.
- “What is the full map?” before revealing all nine entries.
- “Does $H$ or $W$ enter the parameter formula?” before the kernel-bank checkpoint.
- “Which legal starts fit?” before applying the floor formula.
- “Which shifted positions agree exactly?” before naming equivariance.
- “What does pooling guarantee locally—and not globally?” before the answer.
- “What are $(r,j)$ after each layer?” before completing the ledger.
- “Does $\partial\mathcal L/\partial k_0$ use one patch or both?” before summing.
- “What must move when HWC notation enters `Conv2d`?” before running the PyTorch model.

Do not reveal a multiple-choice answer in the same presentation state as its question. The compact handout may collapse builds; the presentation should preserve the commitment.

## Exact mathematics and board work

### 1 · Locality and sharing: compare like with like

A position-specific dense map from $1{,}024$ inputs to $1{,}024$ outputs contains

\[
1{,}024\cdot1{,}024=1{,}048{,}576\text{ weights}
\]

and $1{,}024$ biases, for

\[
1{,}049{,}600\text{ parameters}.
\]

A same-padded $3\times3$ single-channel convolution producing one output channel has nine shared weights and one shared bias:

\[
9+1=10\text{ parameters}.
\]

Thus

\[
\frac{1{,}049{,}600}{10}=104{,}960.
\]

Say **104,960 times fewer parameters**, not weights: both sides include their biases. The convolution cannot reproduce every arbitrary dense map; its savings are exactly the consequence of locality and sharing.

### 2 · Cross-correlation and the fixed response map

For one channel, unit stride, and no padding,

\[
Y[i,j]
=\sum_{u=0}^{k_h-1}\sum_{v=0}^{k_w-1}
K[u,v]X[i+u,j+v].
\]

This is cross-correlation because $K$ is used in its displayed orientation. Mathematical convolution flips the kernel axes first. Deep-learning libraries commonly call cross-correlation “convolution.” Because the kernel is learned, the representable detector orientation is not lost, but the convention matters when reproducing hand calculations.

For the persistent first patch,

\[
X[0\!:\!3,0\!:\!3]
=\begin{bmatrix}0&0&0\\0&0&0\\1&1&1\end{bmatrix},
\]

so

\[
Y[0,0]=0+0+(-1-1-1)=-3.
\]

The next two columns use identical patches and therefore also return $-3$. Moving down changes the local pattern. The final valid map is

\[
Y=\begin{bmatrix}-3&-3&-3\\-3&-3&-3\\3&3&3\end{bmatrix}.
\]

The important sentence is not “the edge detector outputs $-3$.” It is **the same nine learned values were reused to produce all nine responses**.

### 3 · Channels and parameter count

Use the actual photo pixel at zero-based $(x,y)=(373,137)$. Its 8-bit value is
$(131,66,60)$, hence

\[
x=(131,66,60)/255\approx(0.513725,0.258824,0.235294),
\qquad k=(0.5,-1,0.4),
\]

and

\[
y=0.513725(0.5)-0.258824+0.235294(0.4)
=\frac{47}{510}\approx0.092157.
\]

The pixel is observed evidence; $k$ is an illustrative teaching parameter, not
a trained filter.

A $1\times1$ kernel is local in space but still spans all input channels. In board notation,

\[
X\in\mathbb R^{H\times W\times C_{\mathrm{in}}},\quad
K\in\mathbb R^{k_h\times k_w\times C_{\mathrm{in}}\times C_{\mathrm{out}}},\quad
Y\in\mathbb R^{H'\times W'\times C_{\mathrm{out}}}.
\]

With one bias per output channel,

\[
\#\theta=k_hk_wC_{\mathrm{in}}C_{\mathrm{out}}+C_{\mathrm{out}}.
\]

For $3\times3$, $64\to128$ channels:

\[
3\cdot3\cdot64\cdot128+128
=73{,}728+128
=73{,}856.
\]

Output height and width do not enter this parameter formula. They determine how often the parameters are used.

### 4 · Output geometry: count legal starts

For dilation $1$ along one spatial axis,

\[
H_{\mathrm{out}}
=\left\lfloor\frac{H+2p-k}{s}\right\rfloor+1.
\]

Derive it rather than presenting it as a rule to memorize:

- padded length: $H+2p$;
- last legal start: $H+2p-k$;
- legal starts: $0,s,2s,\ldots$ up to that value;
- the $+1$ counts the initial start.

Keep these notebook checks aligned:

| $H$ | $k$ | $p$ | $s$ | legal starts | $H_{\mathrm{out}}$ |
|---:|---:|---:|---:|---|---:|
| 5 | 3 | 0 | 1 | $0,1,2$ | 3 |
| 5 | 3 | 1 | 1 | $0,1,2,3,4$ | 5 |
| 7 | 3 | 1 | 2 | $0,2,4,6$ | 4 |
| 8 | 3 | 0 | 2 | $0,2,4$ | 3 |

For the last row, a start at $6$ would require indices $6,7,8$; index $8$ is outside a length-eight input. The floor discards the incomplete jump.

For odd $k$, unit stride, and symmetric padding,

\[
p=\frac{k-1}{2}\quad\Longrightarrow\quad H_{\mathrm{out}}=H.
\]

Do not generalize that simple formula silently to even kernels or stride greater than one. Framework “same” modes may use asymmetric padding and a stated output convention.

### 5 · Equivariance, boundary evidence, and pooling

Use the exact one-dimensional trace:

\[
x=(0,1,1,0,0,0),\qquad k=(-1,0,1).
\]

Valid cross-correlation gives

\[
f(x)=(1,-1,-1,0).
\]

After a one-position right shift with zero entering from the left,

\[
T_1x=(0,0,1,1,0,0),
\]

so

\[
f(T_1x)=(1,1,-1,-1),\qquad
T_1f(x)=(0,1,-1,-1).
\]

Positions 1–3 agree; position 0 differs because valid cropping removed the output that would have shifted in from the left. This example is valuable precisely because it demonstrates the interior equality and finite-boundary failure together.

Then inspect the secondary real-image evidence. The builder resizes the photo
to $300\times200$, converts it to grayscale, and applies valid
cross-correlation with the declared Sobel kernel

\[
\begin{bmatrix}-1&0&1\\-2&0&2\\-1&0&1\end{bmatrix}.
\]

It recomputes the response after a zero-filled 12-pixel right shift and applies
$2\times2$, stride-2 max pooling to the absolute response. The two signed
response panels use one shared 99th-percentile magnitude scale, so colours are
comparable: blue is negative, white is zero, and orange is positive. The green
panel separately pools the absolute **original** response and uses a clipped
p1–p99 magnitude scale. Treat these as parallel deterministic audits from the
same observed input—not a serial image→response→shift→pool pipeline. Call them
**deterministic operator outputs**, not learned features or predictions.

On an infinite grid, shared stride-one cross-correlation satisfies

\[
f(T_\Delta X)=T_\Delta f(X).
\]

On finite grids, name the conditions:

- crop or padding changes what enters at a boundary;
- stride greater than one samples one phase;
- a stack with total jump $j$ naturally aligns with translations by multiples of $j$;
- pointwise nonlinearities such as ReLU preserve the same interior equivariance.

Pooling makes a different promise. The persistent $2\times2$, stride-one max pool gives

\[
P=\begin{bmatrix}-3&-3\\3&3\end{bmatrix}.
\]

A max is unchanged by rearrangements of the same values inside one fixed window, but crossing a window boundary can change multiple outputs. Pooling discards location; it does not guarantee whole-network translation invariance.

### 6 · Receptive field and jump

Track receptive-field width $r_\ell$ and jump $j_\ell$ from

\[
r_0=1,\qquad j_0=1.
\]

For dilation $1$,

\[
r_\ell=r_{\ell-1}+(k_\ell-1)j_{\ell-1},
\qquad
j_\ell=j_{\ell-1}s_\ell.
\]

The running stack is:

| layer | $k$ | $s$ | $r$ | $j$ |
|---|---:|---:|---:|---:|
| input | 1 | 1 | 1 | 1 |
| conv 1 | 3 | 1 | 3 | 1 |
| max pool | 2 | 2 | 4 | 2 |
| conv 2 | 3 | 1 | 8 | 2 |

The second convolution adds four input pixels, not two, because adjacent inputs at that layer are already two original pixels apart. With dilation $d_\ell$, replace $(k_\ell-1)$ by $(k_\ell-1)d_\ell$. Padding affects boundary alignment, not the theoretical receptive-field size.

The photo overlay draws $r=3,4,8$ with one common display scale of 16 pixels per
input-support unit. These are theoretical supports: they show what *can*
influence a unit, not saliency, attention, or evidence that the trained network
used the cat's face.

### 7 · Tiny-CNN ledger

The exact classifier is:

1. `Conv2d(3, 16, kernel_size=3, padding=1)`;
2. ReLU;
3. `MaxPool2d(2)`;
4. `Conv2d(16, 32, kernel_size=3, padding=1)`;
5. ReLU;
6. adaptive global average pooling;
7. flatten;
8. `Linear(32, 10)`.

Use this ledger:

| layer | output | parameters | MACs under deck convention | $(r,j)$ |
|---|---|---:|---:|---:|
| input | $N\times3\times32\times32$ | 0 | 0 | $(1,1)$ |
| conv 1 | $N\times16\times32\times32$ | 448 | 442,368 | $(3,1)$ |
| max pool | $N\times16\times16\times16$ | 0 | not counted | $(4,2)$ |
| conv 2 | $N\times32\times16\times16$ | 4,640 | 1,179,648 | $(8,2)$ |
| GAP | $N\times32$ | 0 | not counted | global |
| linear | $N\times10$ | 330 | 320 | global |
| **total** |  | **5,418** | **1,622,336** |  |

The MAC convention is explicit: one multiply-accumulate per kernel weight per output position, plus linear weight MACs. It excludes bias adds, activations, pooling, and GAP.

The arithmetic is:

\[
3\cdot3\cdot3\cdot16+16=448,
\]

\[
3\cdot3\cdot16\cdot32+32=4{,}640,
\]

\[
32\cdot10+10=330,
\]

and

\[
448+4{,}640+330=5{,}418.
\]

The MACs are

\[
32\cdot32\cdot16\cdot(3\cdot3\cdot3)=442{,}368,
\]

\[
16\cdot16\cdot32\cdot(3\cdot3\cdot16)=1{,}179{,}648,
\]

and $32\cdot10=320$, giving $1{,}622{,}336$ total.

If height and width both double while all kernels, channels, padding, and stride remain fixed, parameters remain $5{,}418$ while convolution positions and convolution MACs quadruple:

\[
4(442{,}368+1{,}179{,}648)+320=6{,}488{,}384.
\]

The pre-GAP receptive-field width remains $8$; it does not double merely because the input is larger.

### 8 · Shared-weight backpropagation

Use the exact one-dimensional example:

\[
x=(2,1,3),\qquad k=(0.5,-1),\qquad b=0.
\]

The valid forward pass is

\[
y_0=0.5(2)-1(1)=0,
\qquad
y_1=0.5(1)-1(3)=-2.5.
\]

Let the arriving gradient be

\[
g=\frac{\partial\mathcal L}{\partial y}=(1,2).
\]

Because each kernel value is reused, both spatial uses contribute:

\[
\frac{\partial\mathcal L}{\partial k_0}
=g_0x_0+g_1x_1
=1(2)+2(1)=4,
\]

\[
\frac{\partial\mathcal L}{\partial k_1}
=g_0x_1+g_1x_2
=1(1)+2(3)=7.
\]

Thus

\[
\nabla_k\mathcal L=(4,7).
\]

Overlapping patches create a separate accumulation into the input:

\[
\frac{\partial\mathcal L}{\partial x_0}=g_0k_0=0.5,
\]

\[
\frac{\partial\mathcal L}{\partial x_1}=g_0k_1+g_1k_0=-1+1=0,
\]

\[
\frac{\partial\mathcal L}{\partial x_2}=g_1k_1=-2,
\qquad
\frac{\partial\mathcal L}{\partial b}=g_0+g_1=3.
\]

The two-dimensional, batched channels-last form is

\[
\frac{\partial\mathcal L}{\partial K[u,v,c,o]}
=\sum_n\sum_i\sum_j
G[n,i,j,o]
X[n,is+u-p,js+v-p,c].
\]

Teach the meaning before the notation: sum over examples and output locations because all terms update the same shared parameter. Framework autodiff performs this accumulation; the notebook checks it with NumPy, PyTorch `conv1d`, and central finite differences.

### 9 · Board HWC versus PyTorch NCHW

The board uses channels last because it makes the spatial story easy to read:

\[
H\times W\times C.
\]

PyTorch `Conv2d` expects a batch in

\[
N\times C\times H\times W
\]

order, and its weight tensor is

\[
C_{\mathrm{out}}\times C_{\mathrm{in}}\times k_h\times k_w.
\]

Therefore one board image of shape $32\times32\times3$ becomes a PyTorch batch of shape

\[
(1,3,32,32).
\]

Axis order changes; the represented operation does not. Require students to write the axis names, not merely the four integers.

### 10 · Training state is several contracts, not one switch

Keep these mechanisms separate:

| mechanism | training | validation / inference |
|---|---|---|
| input standardization | estimate mean/std on training data | reuse those fixed values |
| augmentation | random, task-valid transforms | deterministic preprocessing |
| Dropout | active | off |
| BatchNorm | batch statistics and running-stat updates | stored running statistics |
| autograd | record graph for updates | usually disable separately |

`model.eval()` changes Dropout and BatchNorm behavior. It does **not** disable gradient recording; use a separate no-gradient context when appropriate. Input standardization is not BatchNorm: dataset statistics are estimated from training data and then fixed.

## Recurring diagnostic: symptom → suspect → test

| observed symptom | first suspect | smallest decisive test | first repair if confirmed |
|---|---|---|---|
| first convolution raises a channel error | HWC/NCHW order or wrong $C_{\mathrm{in}}$ | print named axes and assert `(N,C,H,W)` | transpose/permute once at the data boundary |
| output shape differs by one | padding, floor, or legal-start error | list starts explicitly on one axis | fix $p,s,k$ or the stated framework convention |
| parameter total differs | bias choice, channel count, or groups | calculate kernel volume per output channel | align the declared bias/groups convention |
| map collapses too early | accumulated stride or pooling | extend the shape and $(r,j)$ ledger | remove or delay one downsampling step |
| NumPy and PyTorch signs disagree | correlation versus flipped convolution | run the fixed $X,K$ first response | align the kernel convention |
| same evaluation input changes | Dropout active or BatchNorm state moving | call `eval()` and run two no-grad passes | repair module mode/state handling |
| loss does not fall | data, label, loss, gradient, or update bug | overfit one tiny batch | repair the failing contract before tuning |
| training good, validation poor | generalization, split, augmentation, or mode issue | return to the L7 split/curve audit | change one justified lever using validation |
| memory grows sharply with resolution | activation count, not necessarily parameter count | compare shape and MAC ledgers at two resolutions | reduce batch/resolution or downsample deliberately |

The table is a hypothesis generator, not a lookup oracle. Require one targeted test before changing the optimizer or architecture.

## Notebook choreography: one trace, six predictions

Use the direct public Colab link:

[`01_convolution_shapes_params_from_scratch.ipynb`](https://colab.research.google.com/github/nipunbatra/dl-teaching/blob/master/notebooks/L08/01_convolution_shapes_params_from_scratch.ipynb)

This single notebook remains the minimal sufficient companion. It uses NumPy,
Matplotlib, and PyTorch; trains no model; and retains outputs. It should also
recompute the photo-crop band means, actual RGB pixel, and real Sobel/shift/pool
arrays from the exact SHA-checked source asset. Do not run the raster builder in
the default Colab mainline: the committed composites are presentation assets,
while the notebook's job is to verify their numerical evidence.

### Before running

1. Predict the sign and exact value of $Y[0,0]$, then the full valid $3\times3$ response map.
2. List legal starts for all four geometry cases and predict the three displayed padded/strided array shapes.
3. Predict the crop's five threshold bits and calculate the actual RGB $1\times1$ response, tiny-CNN parameters, MACs, pooling summaries, and receptive-field rows.
4. Predict which shifted output positions should agree and where the valid boundary should fail.
5. Calculate $y$, $\nabla_k\mathcal L$, $\nabla_x\mathcal L$, and $\nabla_b\mathcal L$ before PyTorch.
6. Translate the board input $32\times32\times3$ into PyTorch `(1, 3, 32, 32)` and predict every activation shape.

### Reveal only after commitments

- first response: $Y[0,0]=-3$;
- full valid response: rows $(-3,-3,-3)$, $(-3,-3,-3)$, $(3,3,3)$;
- output lengths for the four geometry cases: $3,5,4,3$;
- photo crop band means $(113.83,112.37,145.50,145.78,109.32)$ and threshold bits $(0,0,1,1,0)$;
- actual RGB pixel $(131,66,60)$, normalized $(.513725,.258824,.235294)$, response $47/510\approx.092157$;
- parameters: conv 1 $448$, conv 2 $4{,}640$, head $330$, total $5{,}418$;
- MACs: $442{,}368$, $1{,}179{,}648$, $320$, total $1{,}622{,}336$;
- shifted trace: the common interior agrees and the first valid-boundary position does not;
- separate $4\times4$ pooling check: max rows $(6,4),(7,8)$; average rows $(3.75,2.25),(2.5,3.75)$; global average $3.0625$;
- receptive-field rows: $(1,1)\to(3,1)\to(4,2)\to(8,2)$;
- manual and PyTorch forward/gradients: $y=(0,-2.5)$, $\nabla_k=(4,7)$, $\nabla_x=(0.5,0,-2)$, $\nabla_b=3$;
- finite-difference kernel gradient: $(4,7)$;
- PyTorch shapes: `(1,3,32,32)` → `(1,16,32,32)` → `(1,16,16,16)` → `(1,32,16,16)` → `(1,32,1,1)` → `(1,32)` → `(1,10)`.

### Explain after running

- The loops make padding, stride, legal starts, and sharing visible.
- Parameters remain fixed when the image grows; repeated spatial uses increase MACs.
- The equivariance assertion intentionally compares the common interior rather than claiming a false boundary equality.
- The pooling array is a separate mechanism check; the deck's persistent pooling calculation stays on $Y$.
- The real Sobel/+12-pixel/pool composite is deterministic evidence; it is not a learned feature map.
- Shared kernel gradients sum across output uses; overlapping input gradients sum across patches.
- The finite-difference check is independent of PyTorch autograd.
- The final model cell trains nothing and performs no additional download; it verifies tensor contracts and counts only.

## Subtleties the instructor must state

- **Flattening is invertible.** The issue is missing architectural structure in the following dense layer, not information deletion by reshape.
- **Library convolution is usually cross-correlation.** Do not flip $K$ in the displayed arithmetic.
- **A kernel is local in space but complete across input channels.** A $1\times1$ RGB kernel has three weights per output channel, not one.
- **Bias convention matters.** The dense/shared ratio includes biases on both sides; other parameter totals must state whether bias is present.
- **Shape, parameters, activations, and MACs are different counts.** Shape, activation, and MAC counts can depend on spatial resolution; the convolutional parameter count does not.
- **The output formula assumes dilation one.** The deck states how dilation modifies receptive field but does not make it core geometry arithmetic.
- **Padding preserves dimensions, not evidence.** Border outputs partly depend on invented values.
- **Equivariance is conditional.** Infinite-grid stride-one theory, finite boundaries, and strided sampling are different cases.
- **Pooling offers local robustness, not a universal invariance theorem.** It also discards the winning location.
- **Receptive field is not kernel size.** It composes through every earlier kernel and stride.
- **Theoretical receptive field is not effective influence.** The deck calculates which inputs can affect a unit, not how strongly trained weights use them.
- **Evidence types are different contracts.** The Oxford box/trimap are ground-truth annotations; $X$ is a teaching construction; the Sobel panels are deterministic computations; no trained prediction is shown in Lecture 8.
- **Global average pooling permits variable spatial maps in this architecture.** It does not mean every CNN with a flattened head accepts arbitrary resolution.
- **Sharing creates a sum in backprop.** Do not average spatial contributions unless another explicit normalization does so.
- **`model.eval()` is not `no_grad()`.** Mode and autograd are separate contracts.
- **Input standardization is not BatchNorm.** Only BatchNorm changes from batch statistics/running updates to stored running statistics.
- **Two $3\times3$ layers are not identical to one $5\times5$ layer.** They match receptive-field width under the stated stride but factorize computation and insert another nonlinearity.

## Where students stumble (and the fix)

- **“Flattening throws away spatial information.”** Reshape the vector back to the image to show that values remain; then point to the unconstrained dense weights.
- **“Ten parameters can produce only ten outputs.”** Separate stored parameters from repeated applications at 1,024 locations.
- **“The convolution has $H'W'$ copies of the kernel.”** Draw one parameter box with arrows to several output equations.
- **“A $1\times1$ convolution sees only one number.”** Rework the RGB calculation: one spatial location, all channels.
- **“Output height belongs in the parameter count.”** Put parameter and MAC formulas in different columns.
- **“Same padding always means $p=(k-1)/2$.”** State odd $k$, stride one, and symmetric padding before using it.
- **“The floor rounds to the nearest integer.”** List legal starts; incomplete windows do not exist.
- **“Shared convolution is exactly translation invariant.”** Use the explicit shifted arrays to distinguish interior equivariance from invariance.
- **“Max pooling makes the entire network invariant.”** Move a value across a pooling-window boundary and compare both outputs.
- **“A second $3\times3$ layer adds only two input pixels after pooling.”** Compute jump first; it adds $(3-1)\cdot2=4$.
- **“Doubling resolution changes the kernels.”** Keep the parameter ledger fixed and quadruple only spatial work.
- **“Each kernel use receives its own gradient.”** The uses are tied to one parameter; write both path contributions in the same derivative.
- **“`(32,32,3)` can go directly into `Conv2d`.”** Add batch, move channels first, and name `(N,C,H,W)` aloud.
- **“`eval()` turns autograd off.”** Demonstrate that a separate no-gradient context is still needed.

## If a student asks…

- **“Why call it convolution if the kernel is not flipped?”** Deep-learning APIs conventionally implement cross-correlation. Since the kernel is learned, either orientation can be learned; hand calculations must still follow one declared convention.
- **“Why should weight sharing improve generalization?”** It restricts the hypothesis class to reuse one detector across positions. That is useful when the task has repeated local structure, but it can be wrong when absolute position requires a different rule.
- **“Can a convolution represent a dense layer?”** A kernel covering the entire input can act like a dense spatial map for one fixed size, but the small shared convolution in this lecture deliberately cannot represent arbitrary position-specific connectivity.
- **“Is convolution always translation equivariant?”** Exactly on the ideal infinite grid at stride one. Finite crop/padding, stride, pooling windows, and later heads determine the practical statement.
- **“Does zero padding bias border predictions?”** It can. Border patches include values not observed in the scene; alternative padding conventions make different assumptions.
- **“Why is conv 2 more expensive after pooling?”** Its map is four times smaller, but it uses 32 output channels and 16 input channels rather than 16 and 3. Kernel volume and output positions both matter.
- **“Why use global average pooling?”** It produces one value per channel, removes a fixed flattened-head size, and encourages channel-level evidence. It deliberately discards spatial location.
- **“Is the receptive field really all the model uses?”** It is the theoretical region that can influence a unit. Learned weights often concentrate effective influence in a smaller part.
- **“Why does the kernel gradient sum over positions?”** The same scalar parameter appears in every local dot product. The multivariable chain rule sums every path from that scalar to the loss.
- **“Why not average the shared gradient over positions?”** The derivative of the stated loss is a sum over parameter uses. A framework may average over the batch because the loss reduction says so; spatial averaging is not implied by sharing itself.
- **“Which tensor order is standard?”** There is no universal notation. This board uses HWC; PyTorch uses NCHW; other ecosystems may default to NHWC. Name axes explicitly.
- **“Where are groups, dilation, and depthwise convolution?”** The general ideas are acknowledged, but modern efficient blocks belong in the next `L8B`/Modern-CNN lecture. Keep standard groups=1 and dilation=1 for this core arithmetic.
- **“Why no dataset training in the notebook?”** The real image already tests operator and annotation contracts without confounding optimization. Empirical transfer evidence begins in the Modern-CNN lecture after these deterministic checks pass.

## If you are short on time

Teach the **55-minute core** and omit the entire should-cover and optional blocks. If less than 55 minutes remain, preserve these in order:

1. L7 bridge, ten-parameter commitment, and the locality/sharing assumption;
2. fixed $X,K$, $Y[0,0]=-3$, and the full $3\times3$ map;
3. output-size derivation with one floor/legal-start example;
4. channel axes, the parameter formula, and one explicit HWC→NCHW translation;
5. interior equivariance plus boundary caveat and one pooling result;
6. $(r,j)$ recursion and the tiny-CNN total $5{,}418$ parameters / $1{,}622{,}336$ MACs;
7. opening revisit and the Modern-CNN handoff.

Assign the shared-gradient numeric and notebook autograd verification as follow-along work. Cut the small-versus-large-kernel comparison, `im2col`, extended state table, provenance, and long Q&A first. Never cut the cross-correlation convention, the distinction between parameter and output counts, or the boundary/stride conditions on equivariance.

## Primary sources and provenance

- PyTorch, [`torch.nn.Conv2d`](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) — operator, NCHW/weight shapes, stride, padding, dilation, and groups.
- Dumoulin and Visin (2016), [*A guide to convolution arithmetic for deep learning*](https://arxiv.org/abs/1603.07285) — convolutional output geometry.
- LeCun et al. (1998), [*Gradient-Based Learning Applied to Document Recognition*](https://ieeexplore.ieee.org/document/726791) — classic shared-weight CNN architecture.
- Parkhi et al. (2012), [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/) — the CC BY-SA real photo, head ROI, and trimap reused through public Lectures 8–11.
- The exact public [L08 Colab notebook](https://colab.research.google.com/github/nipunbatra/dl-teaching/blob/master/notebooks/L08/01_convolution_shapes_params_from_scratch.ipynb) — deterministic slide numerics, from-scratch loops, PyTorch gradients, finite differences, and the NCHW model.

The source assets and derivative contract live in
`shared/vision-evidence/oxford-iiit-pet/`. Run `build_evidence.py` to regenerate
the crop, annotation overlays, Sobel/shift/pool composite, and RF overlay;
`derived/evidence.json` records exact coordinates, values, display scaling, and
statuses. Real-image rasters are deliberate. Equations, grids, plots, and
diagrams remain code-native/vector. None of the Lecture 8 evidence is a
benchmark or trained-model claim.

For this exact sample, the XML head box is VOC-style 1-based inclusive
`(333,72,425,158)` and becomes zero-based half-open `[332,425) × [71,158)`
for course arithmetic. The trimap contract is `1=foreground`, `2=background`,
`3=boundary/ignore`. The Colab needs NumPy, Matplotlib, PyTorch, and Pillow;
stdlib `urllib`, `hashlib`, and XML parsing fetch and verify only these three
small companion assets. It does not run the presentation-raster builder.

## Closing line

> **A convolution is a linear layer that confesses its assumptions: local connections decide what can interact, and shared weights ask the same learned question everywhere. Next, the `L8B`/Modern-CNN lecture asks how deep, efficient, pretrained pipelines compose and reuse that operator.**
