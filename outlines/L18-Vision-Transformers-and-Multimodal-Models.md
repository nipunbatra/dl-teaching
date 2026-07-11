---
title: "Lecture 18: Vision Transformers and Multimodal Models"
source: "https://chatgpt.com/share/6a50728a-9720-83ee-a33d-5f2e3ff744fe"
status: "Imported from the finalized shared-course discussion"
---

# Lecture 18: Vision Transformers and Multimodal Models  
## From Image Patches to CLIP and Vision-Language Models

Core story:

\[
\boxed{
\text{A Transformer can process an image by treating patches as tokens.}
}
\]

Then:

\[
\boxed{
\text{Images and text can be aligned in a shared representation space.}
}
\]

And finally:

\[
\boxed{
\text{Visual features can be connected to a language model for multimodal generation.}
}
\]

The lecture should build in three stages:

\[
\text{ViT}
\rightarrow
\text{vision foundation models}
\rightarrow
\text{vision-language models}
\]

Target: **85–90 minutes, approximately 65 slides**.

---

# Part I — From CNNs to Vision Transformers

## Slide 1 — Title

**Vision Transformers and Multimodal Models**

> From image patches to language-conditioned vision

---

## Slide 2 — Recall: Transformers consume sequences

For text:

\[
x_1,x_2,\ldots,x_T
\]

become token embeddings:

\[
X\in\mathbb R^{T\times d}.
\]

A Transformer block processes these tokens using:

\[
\operatorname{SelfAttention}(X).
\]

Question:

> Can an image also be represented as a sequence?

---

## Slide 3 — Image as a grid

An image:

\[
I\in\mathbb R^{H\times W\times C}.
\]

CNN interpretation:

\[
\text{local grid with spatial structure}.
\]

Transformer interpretation:

\[
\text{sequence of visual tokens}.
\]

---

## Slide 4 — Main Vision Transformer idea

Split image into fixed-size patches:

\[
P\times P.
\]

Flatten each patch, project it into an embedding, and treat it like a token.

The original Vision Transformer showed that a pure Transformer applied to sequences of image patches can perform strongly on image recognition when pretrained at sufficient scale.

---

## Slide 5 — Image-to-token pipeline

```text
Image
  ↓
Split into patches
  ↓
Flatten each patch
  ↓
Linear projection
  ↓
Add position embeddings
  ↓
Transformer encoder
  ↓
Classification head
```

---

# Part II — Patchification

## Slide 6 — Number of patches

For image size:

\[
H\times W
\]

and patch size:

\[
P\times P,
\]

the number of patches is:

\[
\boxed{
N=\frac{H}{P}\frac{W}{P}
}
\]

assuming divisibility.

---

## Slide 7 — Worked patch-count example

Image:

\[
224\times224\times3.
\]

Patch:

\[
16\times16.
\]

Patches along each dimension:

\[
224/16=14.
\]

Total:

\[
N=14\cdot14=196.
\]

So the image becomes a sequence of:

\[
196
\]

visual tokens.

---

## Slide 8 — Patch vector dimension

Each patch has:

\[
P\times P\times C
\]

values.

For:

\[
P=16,\quad C=3,
\]

patch dimension:

\[
16\cdot16\cdot3=768.
\]

So each flattened patch is:

\[
x_i\in\mathbb R^{768}.
\]

---

## Slide 9 — Patch embedding

Project each flattened patch:

\[
z_i=x_iE+b
\]

where:

\[
E\in\mathbb R^{P^2C\times d}.
\]

Thus:

\[
z_i\in\mathbb R^d.
\]

All patches use the same projection matrix.

---

## Slide 10 — Worked patch-embedding shape

Suppose:

\[
P^2C=768,
\qquad
d=768.
\]

Then:

\[
E\in\mathbb R^{768\times768}.
\]

Parameters:

\[
768^2+768
=
590{,}592.
\]

---

## Slide 11 — Patch embedding is a convolution

A convolution with:

\[
\text{kernel}=P,
\qquad
\text{stride}=P,
\qquad
C_{\text{out}}=d
\]

produces the same patch embeddings.

For \(224\times224\), \(P=16\):

\[
224\times224\times3
\rightarrow
14\times14\times d.
\]

Flatten spatial dimensions:

\[
14\cdot14=196.
\]

---

## Slide 12 — Diagram: patchification

Use one image divided into a \(4\times4\) grid.

Highlight:

1. one patch;
2. flattened vector;
3. projected patch token;
4. final patch sequence.

```text
Image grid → [p1, p2, p3, ..., pN] → [z1, z2, ..., zN]
```

---

## Slide 13 — Interactive 1: patch embedding

`vit-patchification.html`

Controls:

- image height and width;
- patch size;
- number of channels;
- embedding dimension.

Show:

- patch grid;
- selected patch;
- flattened patch;
- projected token;
- sequence length;
- parameter count.

---

# Part III — Class token and positional embeddings

## Slide 14 — Classification token

Prepend a learned token:

\[
z_{\text{cls}}\in\mathbb R^d.
\]

Input sequence:

\[
Z_0=
[
z_{\text{cls}};
z_1;
z_2;
\ldots;
z_N
].
\]

Sequence length:

\[
N+1.
\]

---

## Slide 15 — Why a class token?

After Transformer blocks:

\[
Z_L=
[
h_{\text{cls}};
h_1;
\ldots;
h_N
].
\]

Use:

\[
h_{\text{cls}}
\]

as the image representation.

Classification:

\[
p(y\mid I)
=
\operatorname{softmax}
\left(
W_ch_{\text{cls}}+b_c
\right).
\]

---

## Slide 16 — Alternative: global average pooling

Instead of a class token:

\[
h_{\text{image}}
=
\frac1N\sum_{i=1}^{N}h_i.
\]

Then:

\[
z=W_ch_{\text{image}}+b_c.
\]

Both approaches are used in vision architectures.

---

## Slide 17 — Patch order is not automatically known

Self-attention alone is permutation equivariant.

Without position information, the model cannot distinguish:

```text
top-left patch
```

from:

```text
bottom-right patch
```

based solely on sequence position.

---

## Slide 18 — Add position embeddings

Learn:

\[
E_{\text{pos}}
\in
\mathbb R^{(N+1)\times d}.
\]

Input:

\[
\boxed{
Z_0
=
[
z_{\text{cls}};z_1;\ldots;z_N
]
+
E_{\text{pos}}
}
\]

The ViT paper used learned one-dimensional position embeddings; analyses showed that these embeddings learned meaningful two-dimensional spatial structure such as row and column relationships.

---

## Slide 19 — Two-dimensional position intuition

For patch at grid position:

\[
(r,c),
\]

we want the model to know:

- row;
- column;
- relative proximity to other patches.

Possible methods:

- learned flattened position embedding;
- separate row and column embeddings;
- relative positional bias;
- rotary or other relative-position methods.

Keep detailed relative positional methods for a later advanced lecture.

---

## Slide 20 — Changing image resolution

Suppose pretrained position embeddings correspond to:

\[
14\times14
\]

patches.

Finetuning image resolution produces:

\[
24\times24
\]

patches.

Position embeddings can be interpolated from the original spatial grid to the new grid.

Mention as a practical ViT finetuning detail.

---

# Part IV — The ViT encoder

## Slide 21 — ViT block

The patch sequence is processed by Transformer encoder blocks.

Pre-norm version:

\[
Z'_\ell
=
Z_{\ell-1}
+
\operatorname{MHA}
\left(
\operatorname{LN}(Z_{\ell-1})
\right)
\]

\[
Z_\ell
=
Z'_\ell
+
\operatorname{MLP}
\left(
\operatorname{LN}(Z'_\ell)
\right).
\]

---

## Slide 22 — ViT MLP block

For each patch token independently:

\[
\operatorname{MLP}(x)
=
W_2\phi(W_1x+b_1)+b_2.
\]

Typically:

\[
d_{\text{hidden}}>d.
\]

Example:

\[
d=768,\qquad d_{\text{hidden}}=3072.
\]

---

## Slide 23 — Attention between patches

For each patch:

\[
q_i=W_Qh_i
\]

\[
k_j=W_Kh_j
\]

\[
v_j=W_Vh_j.
\]

Attention:

\[
\alpha_{ij}
=
\operatorname{softmax}_j
\left(
\frac{q_i^\top k_j}{\sqrt{d_k}}
\right).
\]

Updated patch:

\[
o_i=\sum_j\alpha_{ij}v_j.
\]

Each patch can directly gather information from every other patch.

---

## Slide 24 — Global context from the first layer

A CNN obtains global context gradually by stacking local convolutions.

Standard ViT self-attention allows a patch to interact with distant image regions in a single block.

The ViT analysis found that some attention heads integrate information over large spatial distances even in relatively early layers.

---

## Slide 25 — ViT architecture diagram

```text
Image
  ↓
Patch embedding
  ↓
[CLS] + patch tokens + positions
  ↓
┌────────────────────────┐
│ LayerNorm              │
│ Multi-head attention   │
│ Residual               │
│ LayerNorm              │
│ MLP                    │
│ Residual               │
└────────────────────────┘ × L
  ↓
CLS representation
  ↓
Linear classifier
```

---

## Slide 26 — Full tensor shapes

Let:

\[
B=\text{batch size},
\quad
N=\text{number of patches},
\quad
d=\text{embedding size}.
\]

Image:

\[
B\times C\times H\times W.
\]

Patch embeddings:

\[
B\times N\times d.
\]

After adding class token:

\[
B\times(N+1)\times d.
\]

Attention matrix per head:

\[
B\times(N+1)\times(N+1).
\]

Output:

\[
B\times(N+1)\times d.
\]

---

# Part V — Fully worked miniature ViT

## Slide 27 — Tiny image

Consider a grayscale image:

\[
4\times4.
\]

Split into:

\[
2\times2
\]

patches.

Number of patches:

\[
N=(4/2)^2=4.
\]

Each patch has dimension:

\[
2\cdot2\cdot1=4.
\]

---

## Slide 28 — Image values

Let:

\[
I=
\begin{bmatrix}
1&1&0&0\\
1&1&0&0\\
0&0&2&2\\
0&0&2&2
\end{bmatrix}.
\]

Patches:

\[
p_1=[1,1,1,1]
\]

\[
p_2=[0,0,0,0]
\]

\[
p_3=[0,0,0,0]
\]

\[
p_4=[2,2,2,2].
\]

---

## Slide 29 — Patch projection

Use a simple projection:

\[
E=
\begin{bmatrix}
1&0\\
0&1\\
1&0\\
0&1
\end{bmatrix}.
\]

Then:

\[
z_i=p_iE.
\]

For \(p_1\):

\[
z_1=[2,2].
\]

For \(p_4\):

\[
z_4=[4,4].
\]

Empty patches produce:

\[
z_2=z_3=[0,0].
\]

---

## Slide 30 — Add class and position tokens

Suppose:

\[
z_{\text{cls}}=[1,1].
\]

Add simple learned position vectors:

\[
r_i=z_i+p_i^{\text{pos}}.
\]

Now tokens contain:

- patch content;
- spatial position.

---

## Slide 31 — One-head attention

Use identity projections:

\[
W_Q=W_K=W_V=I.
\]

Class-token query:

\[
q_{\text{cls}}=[1,1].
\]

Compute scores:

\[
s_j=
\frac{q_{\text{cls}}^\top k_j}{\sqrt2}.
\]

The high-intensity fourth patch receives a larger score.

---

## Slide 32 — Class-token output

Weights:

\[
\alpha_{\text{cls},j}
=
\operatorname{softmax}(s_j).
\]

Output:

\[
o_{\text{cls}}
=
\sum_j\alpha_{\text{cls},j}v_j.
\]

The class token becomes a content-dependent summary of the patches.

---

## Slide 33 — Classification

Feed transformed class token to:

\[
z=W_co_{\text{cls}}+b_c.
\]

Softmax:

\[
p(y\mid I)=\operatorname{softmax}(z).
\]

Cross-entropy:

\[
L=-\log p_y.
\]

Backpropagation updates:

- patch projection;
- position embeddings;
- class token;
- attention parameters;
- MLP parameters;
- classifier.

---

## Slide 34 — Interactive 2: tiny ViT

`tiny-vit.html`

Show:

- \(4\times4\) image;
- four patches;
- patch vectors;
- projected embeddings;
- class token;
- attention scores;
- attention weights;
- class-token output;
- final prediction.

---

# Part VI — ViT versus CNN

## Slide 35 — Inductive bias

CNNs explicitly encode:

- locality;
- translation equivariance;
- hierarchical spatial structure.

Vanilla ViTs encode fewer image-specific assumptions.

Instead, ViTs learn more spatial behaviour from data.

---

## Slide 36 — CNN receptive field versus ViT attention

CNN:

```text
local → local → larger region → global
```

ViT:

```text
all patches can interact at each layer
```

But attention does not automatically mean every interaction is useful.

---

## Slide 37 — Data efficiency

Original ViT results emphasized large-scale pretraining followed by transfer.

DeiT subsequently showed that carefully designed augmentation, regularization and distillation could train competitive convolution-free Transformers using ImageNet alone, reducing the apparent dependence on massive external datasets.

---

## Slide 38 — CNN versus ViT comparison

| Property | CNN | Vanilla ViT |
|---|---|---|
| Core unit | convolution | self-attention |
| Spatial prior | strong | weaker |
| Receptive field | grows with depth | global per block |
| Scaling with resolution | roughly linear spatial cost | quadratic in patch count |
| Data efficiency | often strong | recipe/scale sensitive |
| Interpretability view | feature maps | patch attention/tokens |

---

## Slide 39 — Patch size trade-off

Smaller patch:

\[
P\downarrow
\Rightarrow
N\uparrow.
\]

Advantages:

- finer spatial detail.

Disadvantages:

- longer sequence;
- higher attention cost.

Since:

\[
N=\frac{HW}{P^2},
\]

attention cost scales as:

\[
O(N^2d).
\]

---

## Slide 40 — Worked attention-cost comparison

For \(224\times224\):

### \(P=16\)

\[
N=196
\]

\[
N^2=38{,}416.
\]

### \(P=8\)

\[
N=784
\]

\[
N^2=614{,}656.
\]

Halving patch size increases the attention matrix size by approximately:

\[
16\times.
\]

---

# Part VII — Hierarchical vision Transformers

## Slide 41 — Why vanilla ViT is awkward for dense vision

Classification needs one global representation.

Detection and segmentation need:

- multiple spatial resolutions;
- detailed local features;
- high-resolution maps;
- manageable compute.

Vanilla ViT uses a largely fixed-resolution token sequence.

---

## Slide 42 — Hierarchical representation

CNN-like feature hierarchy:

\[
H/4\times W/4
\rightarrow
H/8\times W/8
\rightarrow
H/16\times W/16
\rightarrow
H/32\times W/32.
\]

Channels increase as spatial resolution decreases.

This supports:

- object detection;
- segmentation;
- feature pyramids.

---

## Slide 43 — Window attention

Instead of attending globally, divide patches into windows.

For window size:

\[
M\times M,
\]

self-attention is computed inside each window.

Cost becomes roughly linear in image size when \(M\) is fixed.

---

## Slide 44 — Shifted windows

Problem:

> Tokens in different windows cannot communicate.

Solution:

- block 1: standard windows;
- block 2: shift window partition;
- neighbouring windows now overlap differently.

Swin Transformer combines hierarchical features with shifted-window attention and was designed as a general-purpose backbone for classification, detection and segmentation.

---

## Slide 45 — Swin diagram

```text
Block A:
| window 1 | window 2 |
| window 3 | window 4 |

Block B:
partition shifted spatially
→ tokens cross previous window boundaries
```

---

## Slide 46 — Patch merging

Combine nearby patch tokens:

\[
2\times2
\]

neighbourhoods are concatenated/projected.

Spatial size:

\[
H\times W
\rightarrow
H/2\times W/2.
\]

Channel dimension increases.

This creates a feature hierarchy.

---

## Slide 47 — ViT versus Swin

| Aspect | ViT | Swin |
|---|---|---|
| Attention | global | local shifted windows |
| Resolution | mostly fixed | hierarchical |
| Cost | quadratic in patch count | approximately linear for fixed window |
| Natural use | classification | general visual backbone |
| Dense prediction | possible | structurally convenient |

---

## Slide 48 — Interactive 3: window attention

`swin-window-attention.html`

Controls:

- patch grid;
- window size;
- shift size;
- selected token.

Show:

- tokens visible in current block;
- shifted-window neighbours;
- number of attention comparisons;
- global versus window attention cost.

---

# Part VIII — Vision pretraining and foundation representations

## Slide 49 — Supervised image pretraining

Standard supervised objective:

\[
I\rightarrow y.
\]

Loss:

\[
L_{\text{CE}}
=
-\log p(y\mid I).
\]

Limitation:

- needs class labels;
- class vocabulary restricts supervision.

---

## Slide 50 — Self-supervised vision

Learn from images without manually supplied class labels.

Possible objectives:

- reconstruct missing patches;
- match augmented views;
- teacher–student consistency;
- contrastive representation learning.

The model learns reusable visual features before downstream finetuning.

---

## Slide 51 — Masked image modelling intuition

Analogous to BERT:

1. split image into patches;
2. mask some patches;
3. predict missing visual content or representations.

\[
\tilde I=C(I)
\]

\[
L=\operatorname{reconstruction}
\left(
f(\tilde I),
I_{\text{masked}}
\right).
\]

Keep this conceptual unless masked autoencoders are taught later.

---

## Slide 52 — Self-distilled visual features

Teacher and student receive different image views.

Encourage consistent representations:

\[
f_{\text{student}}(T_1(I))
\approx
f_{\text{teacher}}(T_2(I)).
\]

DINOv2 demonstrated that scaled self-supervised ViT training on curated image data can yield general-purpose visual features useful across image-level and pixel-level tasks without task-specific labels during pretraining.

---

## Slide 53 — Linear probing and finetuning

Given pretrained visual encoder:

\[
h=f_\theta(I).
\]

### Linear probe

Freeze:

\[
\theta.
\]

Train:

\[
y=Wh+b.
\]

### Finetuning

Update:

\[
\theta,W,b.
\]

Linear probing measures how directly usable the learned representation is.

---

# Part IX — From visual tokens to image-text learning

## Slide 54 — Why connect images and text?

Text provides rich supervision:

```text
a brown dog running through snow
```

One caption contains:

- object;
- attribute;
- action;
- environment;
- relationships.

Compared with a single class label:

```text
dog
```

the supervision is much richer.

---

## Slide 55 — Dual-encoder architecture

Image encoder:

\[
v=f_{\text{img}}(I).
\]

Text encoder:

\[
t=f_{\text{text}}(S).
\]

Project and normalize:

\[
\hat v=\frac{v}{\|v\|}
\]

\[
\hat t=\frac{t}{\|t\|}.
\]

Similarity:

\[
s(I,S)=\hat v^\top\hat t.
\]

---

## Slide 56 — CLIP training idea

Given minibatch of matched pairs:

\[
(I_1,S_1),\ldots,(I_B,S_B),
\]

compute all image-text similarities:

\[
S_{ij}
=
\frac{
\hat v_i^\top\hat t_j
}{\tau}.
\]

The correct matches are on the diagonal:

\[
(i,i).
\]

CLIP trained image and text encoders by predicting which captions corresponded to which images across a large batch, enabling natural-language zero-shot transfer.

---

## Slide 57 — Similarity matrix

\[
S=
\begin{bmatrix}
s(I_1,T_1)&s(I_1,T_2)&\cdots\\
s(I_2,T_1)&s(I_2,T_2)&\cdots\\
\vdots&\vdots&\ddots
\end{bmatrix}.
\]

Correct pairs:

\[
S_{11},S_{22},\ldots,S_{BB}.
\]

Visualize the diagonal as highlighted cells.

---

## Slide 58 — Image-to-text contrastive loss

For image \(i\), matching caption is \(i\):

\[
L_{I\rightarrow T}^{(i)}
=
-\log
\frac{
\exp(S_{ii})
}{
\sum_{j=1}^{B}\exp(S_{ij})
}.
\]

Average:

\[
L_{I\rightarrow T}
=
\frac1B
\sum_i
L_{I\rightarrow T}^{(i)}.
\]

---

## Slide 59 — Text-to-image loss

For caption \(i\):

\[
L_{T\rightarrow I}^{(i)}
=
-\log
\frac{
\exp(S_{ii})
}{
\sum_{j=1}^{B}\exp(S_{ji})
}.
\]

Total symmetric loss:

\[
\boxed{
L=
\frac12
\left(
L_{I\rightarrow T}
+
L_{T\rightarrow I}
\right)
}
\]

---

## Slide 60 — Fully worked contrastive example

Suppose batch size:

\[
B=3.
\]

Similarity logits:

\[
S=
\begin{bmatrix}
3&1&0\\
0&2&1\\
1&0&4
\end{bmatrix}.
\]

For image 1, correct caption is caption 1:

\[
p(T_1\mid I_1)
=
\frac{e^3}{e^3+e^1+e^0}.
\]

Numerically:

\[
\frac{20.09}{20.09+2.72+1}
\approx0.844.
\]

Loss:

\[
-\log(0.844)\approx0.170.
\]

---

## Slide 61 — Negative examples come from the batch

For image \(I_i\):

- positive caption: \(T_i\);
- negative captions: \(T_j,\ j\
eq i\).

Therefore batch size affects the number of available negatives.

A mismatch is not necessarily semantically unrelated, so false negatives can occur.

---

## Slide 62 — Interactive 4: CLIP loss

`clip-contrastive-loss.html`

Controls:

- image embeddings;
- text embeddings;
- temperature;
- batch size;
- matched pair assignment.

Show:

- normalized embeddings;
- similarity matrix;
- row softmax;
- column softmax;
- diagonal targets;
- symmetric loss.

---

# Part X — Zero-shot image classification

## Slide 63 — Classification without a trained class head

Suppose classes:

\[
\mathcal C=
\{\texttt{cat},\texttt{dog},\texttt{car}\}.
\]

Construct prompts:

```text
a photo of a cat
a photo of a dog
a photo of a car
```

Encode each prompt:

\[
t_{\text{cat}},t_{\text{dog}},t_{\text{car}}.
\]

---

## Slide 64 — Compare image to class text

Image embedding:

\[
v=f_{\text{img}}(I).
\]

Class score:

\[
s_c
=
\frac{v^\top t_c}{\tau}.
\]

Probability:

\[
p(c\mid I)
=
\operatorname{softmax}_c(s_c).
\]

No supervised classifier needs to be trained for the new class set.

---

## Slide 65 — Worked zero-shot example

Similarities:

\[
s_{\text{cat}}=4
\]

\[
s_{\text{dog}}=2
\]

\[
s_{\text{car}}=-1.
\]

Then:

\[
p(c\mid I)
=
\operatorname{softmax}([4,2,-1]).
\]

Approximately:

\[
[0.876,0.119,0.006].
\]

Prediction:

\[
\texttt{cat}.
\]

---

## Slide 66 — Prompt ensembling

Instead of one template:

```text
a photo of a cat
```

use several:

```text
a photo of a cat
an image of a cat
a blurry photo of a cat
a close-up of a cat
```

Average text embeddings or predictions.

Prompt wording can materially influence zero-shot results.

---

## Slide 67 — Image-text retrieval

Given text query:

```text
a red car on a snowy road
```

encode query:

\[
t_q.
\]

Rank images by:

\[
\operatorname{cos}(v_i,t_q).
\]

Similarly, retrieve captions given an image.

---

# Part XI — Dual encoders versus fusion models

## Slide 68 — Dual-encoder strength

CLIP-like dual encoder:

```text
image → image embedding
text  → text embedding
          ↓
       similarity
```

Advantages:

- embeddings can be precomputed;
- fast large-scale retrieval;
- zero-shot classification;
- simple contrastive training.

---

## Slide 69 — Dual-encoder limitation

Image and text interact only through final similarity:

\[
v^\top t.
\]

This may be insufficient for detailed compositional reasoning:

- “Is the red cup left of the blue plate?”
- “How many people are holding umbrellas?”
- “What does the sign behind the bus say?”

Need deeper cross-modal interaction.

---

## Slide 70 — Cross-attention fusion

Text queries can attend directly to visual tokens:

\[
Q=\text{text features}
\]

\[
K,V=\text{image features}.
\]

\[
\operatorname{CrossAttention}
(Q_{\text{text}},K_{\text{vision}},V_{\text{vision}}).
\]

This allows question-dependent extraction of image information.

---

# Part XII — Connecting vision encoders to language models

## Slide 71 — Modern vision-language model pattern

```text
Image
  ↓
Vision encoder
  ↓
Visual tokens
  ↓
Connector / projector / query transformer
  ↓
Language-model token space
  ↓
Autoregressive language model
  ↓
Text response
```

---

## Slide 72 — Projection connector

Vision features:

\[
V\in\mathbb R^{N_v\times d_v}.
\]

Map to language dimension:

\[
\tilde V=VW_p+b
\]

where:

\[
W_p\in\mathbb R^{d_v\times d_{\text{LM}}}.
\]

Then treat:

\[
\tilde V
\]

as visual prefix tokens for the language model.

---

## Slide 73 — Language-generation objective

Given image \(I\) and text response:

\[
y_1,\ldots,y_T,
\]

train:

\[
\boxed{
L
=
-\sum_{t=1}^{T}
\log
p(y_t\mid y_{<t},I)
}
\]

The model is still a next-token predictor—but now conditioned on visual tokens.

---

## Slide 74 — BLIP-2 bridge

BLIP-2 keeps a pretrained image encoder and a pretrained language model frozen and trains a lightweight Querying Transformer to bridge the visual and language representation spaces.

Diagram:

```text
Frozen image encoder
        ↓
     Q-Former
        ↓
Frozen language model
```

Teaching point:

\[
\boxed{
\text{Reuse strong unimodal models and learn a small connector.}
}
\]

---

## Slide 75 — Querying Transformer intuition

Use a fixed set of learned queries:

\[
q_1,\ldots,q_M.
\]

Queries attend to image features:

\[
z_m
=
\operatorname{CrossAttention}
(q_m,V,V).
\]

This compresses many image patches into a smaller number of language-relevant visual tokens.

---

## Slide 76 — LLaVA-style architecture

A simple and influential pattern is:

```text
CLIP-like vision encoder
        ↓
linear/MLP projector
        ↓
language model
        ↓
visual instruction response
```

LLaVA showed that connecting a pretrained visual encoder to a language model and performing visual instruction tuning can produce a general-purpose visual conversational model.

---

## Slide 77 — Visual instruction tuning

Training example:

```text
Image: [visual tokens]

User: What is unusual about this image?
Assistant: A cat is sitting inside a refrigerator.
```

Optimize response tokens:

\[
L
=
-\sum_t
\log p(y_t\mid y_{<t},I,\text{instruction}).
\]

Usually user/instruction tokens are masked out of the response loss.

---

## Slide 78 — Frozen versus trained components

Possible strategies:

1. freeze vision encoder and LLM; train connector;
2. freeze vision encoder; finetune connector and LLM;
3. finetune with LoRA/adapters;
4. train most or all components jointly.

Trade-offs:

- compute;
- catastrophic forgetting;
- modality alignment;
- final capability.

---

## Slide 79 — Multimodal task taxonomy

| Task | Input | Output |
|---|---|---|
| image classification | image | class |
| image-text retrieval | image/text | ranked matches |
| captioning | image | description |
| VQA | image + question | answer |
| visual dialogue | image + dialogue | response |
| grounding | text + image | region |
| document understanding | document image + query | structured/text answer |

---

# Part XIII — Limitations and failure modes

## Slide 80 — Spatial reasoning

Patch representations and language supervision may be insufficient for:

- exact counting;
- small objects;
- left/right relationships;
- precise coordinates;
- fine-grained text recognition.

A fluent response is not necessarily visually grounded.

---

## Slide 81 — Hallucination

A generative vision-language model may produce text that is:

- plausible;
- linguistically fluent;
- unsupported by the image.

Cause:

\[
\text{strong language prior}
+
\text{imperfect visual grounding}.
\]

Therefore:

\[
\boxed{
\text{language fluency} \
eq \text{visual correctness}
}
\]

---

## Slide 82 — Data and bias

Image-text web data may contain:

- noisy captions;
- stereotypes;
- duplicated images;
- cultural imbalance;
- missing geographic coverage;
- harmful content.

Model capabilities and failures reflect pretraining data and filtering choices.

---

## Slide 83 — Resolution bottleneck

A high-resolution image may contain millions of pixels.

With patch size \(P\):

\[
N=\frac{HW}{P^2}.
\]

Sending every visual token to an LLM is expensive.

Systems therefore use:

- downsampling;
- pooling;
- query tokens;
- token selection;
- tiling;
- multi-resolution processing.

---

## Slide 84 — Evaluation

Classification:

\[
\text{accuracy}
\]

Retrieval:

\[
\text{Recall@}k
\]

Captioning:

\[
\text{CIDEr, BLEU, SPICE, human evaluation}
\]

VQA:

\[
\text{task accuracy}
\]

Open-ended multimodal reasoning also needs careful human and grounded evaluation.

---

# Part XIV — Integration and summary

## Slide 85 — One pipeline, several models

```text
Image
  ↓
Patch tokens
  ↓
Vision Transformer
  ├→ classifier                     ViT
  ├→ general-purpose representation DINO-style
  ├→ text-aligned embedding          CLIP
  └→ connector → language model      VLM/LMM
```

---

## Slide 86 — Architecture comparison

| Model | Main output | Training signal |
|---|---|---|
| ViT | image class/feature | class labels |
| self-supervised ViT | visual feature | image transformations/masking |
| CLIP | aligned embeddings | image-caption pairs |
| BLIP-2 | visual tokens for LLM | image-text objectives |
| LLaVA-style model | text response | visual instructions |

---

## Slide 87 — Key equations

Patch sequence:

\[
Z_0=
[
z_{\text{cls}};
x_1E;
\ldots;
x_NE
]
+
E_{\text{pos}}.
\]

Self-attention:

\[
A=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt d}
\right).
\]

CLIP similarity:

\[
S_{ij}
=
\frac{
\hat v_i^\top\hat t_j
}{\tau}.
\]

Multimodal LM:

\[
p(y_{1:T}\mid I)
=
\prod_t
p(y_t\mid y_{<t},I).
\]

---

## Slide 88 — Final mental model

\[
\boxed{
\text{ViT turns patches into tokens.}
}
\]

\[
\boxed{
\text{Self-attention lets image regions exchange information.}
}
\]

\[
\boxed{
\text{CLIP aligns images and language in a shared space.}
}
\]

\[
\boxed{
\text{Vision-language models feed visual tokens into language models.}
}
\]

---

## Slide 89 — Retrieval questions

1. Why does halving patch size increase attention cost dramatically?
2. How is patch embedding equivalent to convolution?
3. Why does ViT need positional information?
4. What is the difference between global and window attention?
5. In CLIP, what are the negative examples?
6. Why can CLIP classify unseen class vocabularies?
7. What is the connector’s job in a multimodal LLM?
8. Why can a VLM hallucinate despite seeing the image?

---

## Slide 90 — Next lecture

A natural next lecture is:

# Representation Learning and Autoencoders

Topics:

- bottleneck representations;
- undercomplete autoencoders;
- denoising autoencoders;
- sparse representations;
- latent-variable modelling;
- variational autoencoders;
- reconstruction versus generation.

This then leads cleanly to:

\[
\text{VAEs}
\rightarrow
\text{GANs}
\rightarrow
\text{diffusion models}.
\]

---

# Timing

| Section | Slides | Time |
|---|---:|---:|
| From images to patches | 1–13 | 13 min |
| Position + ViT encoder | 14–26 | 12 min |
| Worked tiny ViT | 27–34 | 9 min |
| CNN vs ViT | 35–40 | 7 min |
| Hierarchical Transformers | 41–48 | 9 min |
| Vision pretraining | 49–53 | 6 min |
| CLIP and contrastive loss | 54–67 | 15 min |
| Fusion and multimodal LMs | 68–79 | 13 min |
| Limitations | 80–84 | 5 min |
| Summary | 85–90 | 4 min |

For a strict 75–80 minute slot, move Slides 49–53 on self-supervised visual pretraining to a supplementary section and keep **ViT → Swin → CLIP → VLM** as the main path.

# Essential diagrams

1. Image split into patches.
2. Patch flattening and linear projection.
3. Patch embedding as strided convolution.
4. Class token aggregation.
5. Complete ViT encoder.
6. CNN receptive-field growth versus global attention.
7. Global versus shifted-window attention.
8. CLIP image-text similarity matrix.
9. Zero-shot classification through text prompts.
10. Dual encoder versus cross-attention fusion.
11. Vision encoder–connector–LLM pipeline.
12. BLIP-2 query-token bridge.
13. Visual instruction-tuning data flow.
14. Hallucination: language prior versus visual evidence.

# Interactives

1. `vit-patchification.html`  
   Patch count, patch vectors, embeddings and sequence length.

2. `tiny-vit.html`  
   Full worked miniature Vision Transformer.

3. `vit-attention-map.html`  
   Select a patch or class token and view attention over image regions.

4. `patch-size-complexity.html`  
   Patch size versus token count and quadratic attention cost.

5. `swin-window-attention.html`  
   Standard and shifted windows.

6. `clip-contrastive-loss.html`  
   Similarity matrix and symmetric contrastive loss.

7. `clip-zero-shot.html`  
   Prompts, image-text similarity and zero-shot classification.

8. `vision-language-connector.html`  
   Vision tokens, projector/query tokens and LLM input sequence.

# Notebooks

1. `01_patch_embedding_from_scratch.ipynb`
2. `02_tiny_vit_forward.ipynb`
3. `03_vit_attention_visualization.ipynb`
4. `04_vit_vs_cnn_transfer_learning.ipynb`
5. `05_window_attention_and_swin.ipynb`
6. `06_clip_loss_from_scratch.ipynb`
7. `07_clip_zero_shot_classification.ipynb`
8. `08_image_text_retrieval.ipynb`
9. `09_simple_vision_to_language_projector.ipynb`
