// Vision Transformers and Multimodal Models
// Lecture 17 · ES 667 Deep Learning
// Compile from the repo root:
//   typst compile --root . lecture17/L17-vit-multimodal.typ /tmp/L17.pdf
//   typst compile --root . --input handout=true lecture17/L17-vit-multimodal.typ /tmp/L17h.pdf
// Theme, palette, helpers and diagram builders live in common/metropolis.typ.
// Schematic figures: python3 lecture17/diagrams/l17_figs.py

#import "../common/metropolis.typ": *
#show: metropolis-deck.with(
  title: [Vision Transformers and Multimodal Models],
  subtitle: [From image patches to CLIP and vision-language models],
)

#let IA = "https://nipunbatra.github.io/interactive-articles/"
#let cream = rgb("#EFEEEB")
#let rect = fletcher.shapes.rect

// ═══════════════════════════ fletcher diagrams (centerpieces) ═══════════════════════════

// ── (A) image -> tokens pipeline: the whole ViT front-end in one row ──
#let imgtoken = align(center, diagram(spacing: (10mm, 9mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let bx(x, lbl, c, f) = node((x, 0), align(center, text(size: 9.5pt, lbl)), shape: rect, corner-radius: 4pt, inset: 5pt, stroke: 0.9pt + c, fill: f)
  node((0, 0), text(size: 10pt)[image $I$], radius: 5.5mm, fill: cream)
  bx(1, [split into \ $P times P$ patches], TEAL, TEAL.lighten(82%))
  bx(2, [flatten $+$ \ project $E$], ACC, ACC.lighten(82%))
  bx(3, [$+$ position \ $+$ [CLS]], BLUE, BLUE.lighten(85%))
  bx(4, [Transformer \ encoder], INK, cream)
  bx(5, [head], GREEN, GREEN.lighten(82%))
  for a in range(5) { edge((a, 0), (a + 1, 0), "-|>", stroke: 0.7pt + MUTED) }
}))

// ── (B) ViT pre-norm encoder block: residual spine + two branch sublayers ──
#let vitblock = align(center, diagram(spacing: (12mm, 10mm), node-stroke: 0.9pt + INK, node-fill: white, {
  // residual spine
  node((0, 0), $Z_(ell-1)$, radius: 6mm, fill: cream, stroke: 1pt + INK)
  node((1.9, 0), text(size: 14pt)[$+$], radius: 5mm, fill: GREEN.lighten(80%), stroke: 1pt + GREEN)
  node((4.7, 0), text(size: 14pt)[$+$], radius: 5mm, fill: GREEN.lighten(80%), stroke: 1pt + GREEN)
  node((5.9, 0), $Z_ell$, radius: 6mm, fill: ACC.lighten(72%), stroke: 1pt + ACC)
  edge((0, 0), (1.9, 0), "-|>", stroke: 1.6pt + INK)
  edge((1.9, 0), (4.7, 0), "-|>", stroke: 1.6pt + INK)
  edge((4.7, 0), (5.9, 0), "-|>", stroke: 1.6pt + INK)
  // branch 1: LN -> MHA
  node((0.7, 1.5), text(size: 11pt)[LN], radius: 5mm, fill: BLUE.lighten(85%), stroke: 0.9pt + BLUE)
  node((1.9, 1.5), align(center, text(size: 9.5pt)[multi-head \ self-attn]), shape: rect, corner-radius: 4pt, inset: 5pt, fill: TEAL.lighten(82%), stroke: 0.9pt + TEAL)
  edge((0, 0), (0.7, 1.5), "-|>", stroke: 0.9pt + TEAL)
  edge((0.7, 1.5), (1.9, 1.5), "-|>", stroke: 0.9pt + TEAL)
  edge((1.9, 1.5), (1.9, 0), "-|>", stroke: 0.9pt + TEAL)
  // branch 2: LN -> MLP
  node((3.5, 1.5), text(size: 11pt)[LN], radius: 5mm, fill: BLUE.lighten(85%), stroke: 0.9pt + BLUE)
  node((4.7, 1.5), text(size: 11pt)[MLP], shape: rect, corner-radius: 4pt, inset: 6pt, fill: ACC.lighten(82%), stroke: 0.9pt + ACC)
  edge((1.9, 0), (3.5, 1.5), "-|>", stroke: 0.9pt + ACC)
  edge((3.5, 1.5), (4.7, 1.5), "-|>", stroke: 0.9pt + ACC)
  edge((4.7, 1.5), (4.7, 0), "-|>", stroke: 0.9pt + ACC)
  node((2.95, -1.0), text(size: 10.5pt, fill: INK)[*pre-norm*: LN sits on the *branch*, the residual spine runs clean], stroke: none, fill: none)
}))

// ── (C) full ViT architecture: patch embed -> [CLS]+pos -> L blocks -> CLS -> classifier ──
#let vitarch = align(center, diagram(spacing: (9mm, 9mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let bx(x, lbl, c, f) = node((x, 0), align(center, text(size: 9.5pt, lbl)), shape: rect, corner-radius: 4pt, inset: 6pt, stroke: 0.9pt + c, fill: f)
  node((0, 0), text(size: 10pt)[image $I$], radius: 5.5mm, fill: cream)
  bx(1, [patch \ embed], TEAL, TEAL.lighten(82%))
  bx(2, [prepend [CLS] \ $+$ positions], BLUE, BLUE.lighten(85%))
  node((3, 0), align(center, text(size: 9.5pt, fill: white)[encoder \ block]), shape: rect, corner-radius: 4pt, inset: 7pt, fill: INK.lighten(4%), stroke: 1pt + INK)
  node((3, 1.2), text(size: 11pt, fill: INK)[$times L$], stroke: none, fill: none)
  bx(4, [take \ $h_"cls"$], ACC, ACC.lighten(82%))
  bx(5, [linear \ classifier], GREEN, GREEN.lighten(82%))
  node((6, 0), text(size: 10pt)[$p(y | I)$], shape: rect, corner-radius: 4pt, inset: 6pt, fill: cream, stroke: 0.9pt + INK)
  for a in range(6) { edge((a, 0), (a + 1, 0), "-|>", stroke: 0.7pt + MUTED) }
}))

// ── (D) CLIP dual encoder: two towers -> normalize -> cosine similarity ──
#let dualenc = align(center, diagram(spacing: (12mm, 8mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let bx(pos, lbl, c, f) = node(pos, align(center, text(size: 9.5pt, lbl)), shape: rect, corner-radius: 4pt, inset: 6pt, stroke: 0.9pt + c, fill: f)
  // image tower (top)
  node((0, 1), text(size: 10pt)[image $I$], radius: 6mm, fill: cream)
  bx((1.5, 1), [image encoder \ (ViT)], TEAL, TEAL.lighten(82%))
  bx((3.0, 1), [normalize], BLUE, BLUE.lighten(85%))
  node((4.2, 1), $hat(v)$, radius: 5mm, fill: TEAL.lighten(70%), stroke: 0.9pt + TEAL)
  edge((0, 1), (1.5, 1), "-|>", stroke: 0.7pt + MUTED)
  edge((1.5, 1), (3.0, 1), "-|>", stroke: 0.7pt + MUTED, label: text(size: 8.5pt, fill: MUTED)[$v$], label-side: left)
  edge((3.0, 1), (4.2, 1), "-|>", stroke: 0.7pt + MUTED)
  // text tower (bottom)
  node((0, -1), text(size: 10pt)[text $S$], radius: 6mm, fill: cream)
  bx((1.5, -1), [text encoder \ (Transformer)], ACC, ACC.lighten(82%))
  bx((3.0, -1), [normalize], BLUE, BLUE.lighten(85%))
  node((4.2, -1), $hat(t)$, radius: 5mm, fill: ACC.lighten(72%), stroke: 0.9pt + ACC)
  edge((0, -1), (1.5, -1), "-|>", stroke: 0.7pt + MUTED)
  edge((1.5, -1), (3.0, -1), "-|>", stroke: 0.7pt + MUTED, label: text(size: 8.5pt, fill: MUTED)[$t$], label-side: left)
  edge((3.0, -1), (4.2, -1), "-|>", stroke: 0.7pt + MUTED)
  // similarity
  node((5.7, 0), align(center, text(size: 10pt)[similarity \ $s = hat(v)^top hat(t)$]), shape: rect, corner-radius: 4pt, inset: 7pt, fill: GREEN.lighten(82%), stroke: 0.9pt + GREEN)
  edge((4.2, 1), (5.7, 0), "-|>", stroke: 0.9pt + TEAL)
  edge((4.2, -1), (5.7, 0), "-|>", stroke: 0.9pt + ACC)
}))

// ── (E) modern VLM: vision encoder -> connector -> visual prefix tokens -> LM -> text ──
#let vlmpipe = align(center, diagram(spacing: (9mm, 8mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let bx(pos, lbl, c, f) = node(pos, align(center, text(size: 9.5pt, lbl)), shape: rect, corner-radius: 4pt, inset: 6pt, stroke: 0.9pt + c, fill: f)
  node((0, 0), text(size: 10pt)[image $I$], radius: 6mm, fill: cream)
  bx((1.35, 0), [vision \ encoder], TEAL, TEAL.lighten(82%))
  bx((2.7, 0), [visual \ tokens $V$], TEAL, TEAL.lighten(70%))
  bx((4.0, 0), [connector \ (projector)], ACC, ACC.lighten(82%))
  bx((5.35, 0), [language \ model], INK, cream)
  node((6.7, 0), align(center, text(size: 9.5pt)[text \ response]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: GREEN.lighten(82%), stroke: 0.9pt + GREEN)
  for (a, b) in ((0, 1.35), (1.35, 2.7), (2.7, 4.0), (4.0, 5.35), (5.35, 6.7)) {
    edge((a, 0), (b, 0), "-|>", stroke: 0.7pt + MUTED)
  }
  // text prompt also feeds the LM
  node((4.0, -1.5), align(center, text(size: 9pt)[text prompt \ (instruction)]), shape: rect, corner-radius: 4pt, inset: 5pt, fill: BLUE.lighten(85%), stroke: 0.9pt + BLUE)
  edge((4.0, -1.5), (5.35, 0), "-|>", stroke: 0.7pt + BLUE)
  node((4.0, 0.95), text(size: 9pt, fill: ACC)[$V W_p -> $ prefix], stroke: none, fill: none)
}))

// ── (F) BLIP-2 bridge: frozen image encoder + tiny Q-Former + frozen LM ──
#let blip2 = align(center, diagram(spacing: (11mm, 8mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 0), text(size: 10pt)[image $I$], radius: 6mm, fill: cream)
  node((1.6, 0), align(center, text(size: 9.5pt)[image encoder]), shape: rect, corner-radius: 4pt, inset: 7pt, fill: TEAL.lighten(82%), stroke: 0.9pt + TEAL)
  node((1.6, -1.15), text(size: 9pt, fill: BLUE)[❄ frozen], stroke: none, fill: none)
  node((3.4, 0), align(center, text(size: 9.5pt)[Q-Former \ (learned queries)]), shape: rect, corner-radius: 4pt, inset: 7pt, fill: ACC.lighten(78%), stroke: 1.2pt + ACC)
  node((3.4, -1.15), text(size: 9pt, fill: ACC)[*trainable* (small)], stroke: none, fill: none)
  node((5.3, 0), align(center, text(size: 9.5pt)[language model]), shape: rect, corner-radius: 4pt, inset: 7pt, fill: INK.lighten(4%), stroke: 0.9pt + INK)
  node((5.3, 0), align(center, text(size: 9.5pt, fill: white)[language model]), shape: rect, corner-radius: 4pt, inset: 7pt, fill: INK.lighten(4%), stroke: 0.9pt + INK)
  node((5.3, -1.15), text(size: 9pt, fill: BLUE)[❄ frozen], stroke: none, fill: none)
  node((6.9, 0), align(center, text(size: 9.5pt)[text]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: GREEN.lighten(82%), stroke: 0.9pt + GREEN)
  for (a, b) in ((0, 1.6), (1.6, 3.4), (3.4, 5.3), (5.3, 6.9)) {
    edge((a, 0), (b, 0), "-|>", stroke: 0.8pt + MUTED)
  }
  node((3.45, 1.35), text(size: 10pt, fill: INK)[reuse strong *frozen* models, learn a small *connector*], stroke: none, fill: none)
}))

// ── (G) cross-attention fusion: text queries attend to visual tokens ──
#let crossfuse = align(center, diagram(spacing: (17mm, 9mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, -0.9), align(center, text(size: 9.5pt)[text tokens \ (query $Q$)]), shape: rect, corner-radius: 4pt, inset: 5pt, fill: ACC.lighten(82%), stroke: 0.9pt + ACC)
  node((0, 0.9), align(center, text(size: 9.5pt)[visual tokens \ ($K, V$)]), shape: rect, corner-radius: 4pt, inset: 5pt, fill: TEAL.lighten(82%), stroke: 0.9pt + TEAL)
  node((2, 0), align(center, text(size: 9.5pt)[cross-attention]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: BLUE.lighten(85%), stroke: 1pt + BLUE)
  node((3.4, 0), align(center, text(size: 9.5pt)[fused \ text rep]), shape: rect, corner-radius: 4pt, inset: 5pt, fill: GREEN.lighten(82%), stroke: 0.9pt + GREEN)
  edge((0, -0.9), (2, 0), "-|>", stroke: 0.9pt + ACC)
  edge((0, 0.9), (2, 0), "-|>", stroke: 0.9pt + TEAL)
  edge((2, 0), (3.4, 0), "-|>", stroke: 0.9pt + INK)
}))

#title-slide()

// ═══════════════════════════ PART I — From CNNs to ViT ═══════════════════════════
= From CNNs to Vision Transformers

== Where we are

We have two big architectures: *CNNs* for images (L8) and *Transformers* for sequences (L15–16).
#pause
$ #text(fill: TEAL)[CNN]: quad I in RR^(H times W times C) --> "conv, pool" quad quad #text(fill: ACC)[Transformer]: quad X in RR^(T times d) --> "SelfAttention"(X) $
#pause
#notebox[The Transformer consumes a *sequence of tokens*. It knows nothing about grids, pixels, or convolutions. This lecture asks a simple question:]
#pause
#result[can we make an *image* look like a *sequence* — and run a plain Transformer on it?]

== An image is a grid; a Transformer wants a sequence #D

The mismatch is structural:
#pause
#two(
  notebox[*CNN view* — an image is a *2-D grid*; a conv kernel mixes each pixel with its *local* neighbours.],
  notebox[*Transformer view* — the input must be a *sequence of vectors* $x_1, dots, x_T$, each a *token*.],
)
#pause
#align(center, text(size: 16pt, fill: MUTED)[so we need to turn the $H times W times C$ grid into a *short list of visual tokens*])

== The ViT main idea #D

Vision Transformer (Dosovitskiy et al. 2021): cut the image into *patches* and treat each patch as a token.
#pause
+ split the image into a grid of $P times P$ patches;
#pause
+ *flatten* each patch and *linearly project* it to an embedding;
#pause
+ feed the resulting sequence of patch tokens to a *pure Transformer encoder*.
#pause
#result[no convolutions — a patch is to an image what a *word* is to a sentence]

== The image-to-token pipeline #V

The whole ViT front-end, end to end:
#pause
#imgtoken
#pause
#align(center, text(size: 14pt, fill: MUTED)[image $->$ patches $->$ flatten $->$ project $->$ add position & [CLS] $->$ Transformer $->$ head])

// ═══════════════════════════ PART II — Patchification ═══════════════════════════
= Patchification

== How many patches? #D

A patch size $P$ tiles the $H times W$ image into a grid of non-overlapping squares:
#pause
$ N = H/P dot W/P quad quad #text(size: 15pt, fill: MUTED)[(patches per column $times$ patches per row)] $
#pause
#align(center, text(size: 16pt, fill: MUTED)[$N$ is the *sequence length* the Transformer sees — usually a few hundred, not thousands])

== Worked: how many patches? #D

The standard ViT setup — a $224 times 224$ image with patch size $P = 16$:
#pause
$ 224 / 16 = 14 quad quad #text(size: 15pt, fill: MUTED)[(patches along each side)] $
#pause
$ N = 14 times 14 = #text(fill: ACC, weight: 600)[196] quad #text(size: 15pt, fill: MUTED)[patch tokens] $
#pause
#align(center, text(size: 16pt, fill: MUTED)[a $224 times 224$ image becomes a sequence of just *196 tokens* — very manageable])

== What is inside one patch? #D

Each $P times P$ patch has $P times P$ pixels, each with $C$ channels — flatten it into one vector:
#pause
$ x_i in RR^(P^2 C) quad quad #text(size: 15pt, fill: MUTED)[(for $P = 16, C = 3$:  $16^2 dot 3 = 768$)] $
#pause
Project it to the model width $d$ with a *shared* embedding matrix $E$:
$ z_i = x_i E + b, quad quad E in RR^(P^2 C times d), quad z_i in RR^d $
#pause
#align(center, text(size: 16pt, fill: MUTED)[the *same* $E$ projects *every* patch — one learned linear map, reused $N$ times])

== Worked: patch-embedding parameters #D

With $P = 16, C = 3$ so $P^2 C = 768$, and model width $d = 768$:
#pause
$ E in RR^(768 times 768) quad quad #text(size: 15pt, fill: MUTED)[(weights)], quad b in RR^768 quad #text(size: 15pt, fill: MUTED)[(bias)] $
#pause
$ 768^2 + 768 = "589,824" + 768 = #text(fill: ACC, weight: 600)[$"590,592"$] quad #text(size: 15pt, fill: MUTED)[parameters] $
#pause
#align(center, text(size: 16pt, fill: MUTED)[the entire patch embedding is *one* small linear layer — cheap compared to the encoder])

== Patch embedding is a strided convolution #V

Flattening every $P times P$ patch and applying a shared $E$ is *exactly* a convolution:
#pause
#fig("/lecture17/figures/patch_as_conv.svg", w: 62%)
#pause
#align(center, text(size: 14pt, fill: MUTED)[kernel $= P times P$, stride $= P$, $C_"out" = d$ — one conv layer turns the image into patch tokens])

== Patchification, drawn #V

One patch: cut it out, flatten its pixels, project to a $d$-dim token — repeat for all $N$:
#pause
#fig("/lecture17/figures/patchify.svg", w: 66%)

== Interactive: patchification #I

#interbox(link-to: IA + "vision-transformer")[
  Drag the patch size $P$ and watch the image re-tile: see $N = (H \/ P)(W \/ P)$ change, one patch flatten into $x_i in RR^(P^2 C)$, and project to a token $z_i$.
]

// ═══════════════════════════ PART III — Class token and positions ═══════════════════════════
= The class token and positions

== A token to summarize the image #D

Prepend one *learned* vector — the class token $z_"cls" in RR^d$ — to the patch sequence:
#pause
$ Z_0 = [thin z_"cls"; thin z_1; thin z_2; thin dots; thin z_N] quad quad #text(size: 15pt, fill: MUTED)[(length $N + 1$)] $
#pause
#notebox[$z_"cls"$ carries no patch content of its own. Its job is to *gather* information from all patches through the attention layers, and end up as the image's representation.]

== Reading out the class token #D

Use the final class-token state $h_"cls"$ as the whole-image vector:
#pause
$ p(y | I) = "softmax"(W_c thin h_"cls" + b_c) $
#pause
Or *global-average-pool* the patch outputs:
$ h_"image" = 1/N sum_(i=1)^N h_i $
#pause
#align(center, text(size: 16pt, fill: MUTED)[[CLS] token *or* mean-pool — both turn $N$ tokens into one image vector])

== Self-attention has no sense of order #D

Attention is a weighted sum over tokens — permute the patches and the outputs just permute too:
#pause
$ "SelfAttention"([z_1; z_2; z_3]) quad #text(fill: RED)[is a permutation of] quad "SelfAttention"([z_3; z_1; z_2]) $
#pause
#alertbox[Self-attention is *permutation-equivariant*: on its own it cannot tell the *top-left* patch from the *bottom-right* one. Spatial layout is invisible.]

== Adding position embeddings #D

Add a learned position vector to every token — patches *and* the class token:
#pause
$ Z_0 = [thin z_"cls"; thin z_1; thin dots; thin z_N] + E_"pos", quad quad E_"pos" in RR^((N+1) times d) $
#pause
#notebox[ViT uses *learned 1-D* position embeddings, one vector per slot in the flattened patch order. They give attention a location signal; at a new resolution, the learned grid must be adapted (usually by interpolation).]

== Position choices and changing resolution #D

Several ways to encode *where* a patch sits:
#pause
- *learned* (ViT default) · *separate row / column* · *relative* (Swin) · *rotary*;
#pause
#align(center, text(size: 16pt, fill: MUTED)[all inject a position signal so attention can use spatial layout])
#pause
#alertbox[*Changing resolution.* Test at a higher resolution $->$ more patches than trained. Fix: *interpolate* the position embeddings from the $14 times 14$ grid up to, say, $24 times 24$.]

// ═══════════════════════════ PART IV — The ViT encoder ═══════════════════════════
= The ViT encoder

== One ViT block (pre-norm) #D

A ViT block is a standard *pre-norm* Transformer block over the patch sequence $Z$:
#pause
$ Z'_ell = Z_(ell-1) + "MHA"("LN"(Z_(ell-1))) $
#pause
$ Z_ell = Z'_ell + "MLP"("LN"(Z'_ell)) $
#pause
#align(center, text(size: 16pt, fill: MUTED)[identical to the language Transformer — only the *tokens* changed, from words to patches])

== The per-token MLP #D

Inside each block, an MLP is applied *independently* to every token:
#pause
$ "MLP"(x) = W_2 thin phi(W_1 x + b_1) + b_2, quad quad d_"hidden" > d $
#pause
- typical sizing: $d = 768 -> d_"hidden" = 3072 -> d = 768$ (a $4 times$ expansion);
- $phi$ is usually GELU.
#pause
#align(center, text(size: 16pt, fill: MUTED)[attention *mixes* patches; the MLP then *processes* each patch on its own])

== Attention between patches #D

Every patch becomes a query, a key and a value — and attends to *all* patches:
#pause
$ q_i = W_Q h_i, quad k_j = W_K h_j, quad v_j = W_V h_j $
#pause
$ alpha_(i j) = "softmax"_j (q_i^top k_j / sqrt(d_k)), quad quad o_i = sum_(j) alpha_(i j) thin v_j $
#pause
#align(center, text(size: 16pt, fill: MUTED)[each patch gathers content from *every other patch* — a global operation])

== Global context from the very first layer #D

This is the key difference from convolution:
#pause
#two(
  notebox[*CNN* — the receptive field grows *slowly* with depth; distant pixels interact only in *late* layers.],
  notebox[*ViT* — any patch can attend to any other in a *single* block; some heads integrate large distances *early*.],
)
#pause
#align(center, text(size: 16pt, fill: MUTED)[global from layer one — but *not every* long-range interaction is useful; the model must *learn* which])

== The ViT encoder block #V

#vitblock
#pause
#align(center, text(size: 14pt, fill: MUTED)[the residual spine runs straight through; each sublayer normalizes, computes, and adds back])

== The full ViT architecture #V

#vitarch
#pause
#align(center, text(size: 14pt, fill: MUTED)[patch-embed $->$ prepend [CLS] $+$ positions $->$ $L$ encoder blocks $->$ classify from $h_"cls"$])

== Tensor shapes end to end #V

Batch $B$, patches $N$, width $d$:
#pause
#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 6pt), align: (left, center, left),
  table.header([*Stage*], [*Shape*], [*Note*]),
  [image], [$B times C times H times W$], [raw pixels],
  [patch embeddings], [$B times N times d$], [$N$ patch tokens],
  [$+$ class token], [$B times (N+1) times d$], [prepend [CLS]],
  [attention matrix], [$B times (N+1) times (N+1)$], [token-to-token],
  [block output], [$B times (N+1) times d$], [same shape in and out],
))

// ═══════════════════════════ PART V — A worked tiny ViT ═══════════════════════════
= A worked tiny ViT

== The setup #D

A $4 times 4$ grayscale "image", patch size $2$ — so $N = 4$ patches, each of dimension $4$:
#pause
$ I = mat(1, 1, 0, 0; 1, 1, 0, 0; 0, 0, 2, 2; 0, 0, 2, 2) $
#pause
Flatten each $2 times 2$ patch (row-major):
$ p_1 = mat(1,1,1,1), thin p_2 = mat(0,0,0,0), thin p_3 = mat(0,0,0,0), thin p_4 = mat(2,2,2,2) $

== Projecting the patches #D

Project each 4-dim patch to a 2-dim token with a shared $E$:
#pause
$ E = mat(1, 0; 0, 1; 1, 0; 0, 1), quad quad z_i = p_i E $
#pause
$ z_1 = mat(2, 2), quad z_2 = mat(0, 0), quad z_3 = mat(0, 0), quad z_4 = mat(4, 4) $
#pause
#align(center, text(size: 16pt, fill: MUTED)[the bright bottom-right patch $p_4$ becomes the token with the *largest* norm])

== Class token and one-head attention #D

Prepend $z_"cls" = mat(1, 1)$ (drop positions for clarity). Take $W_Q = W_K = W_V = I$, so $q_"cls" = mat(1, 1)$:
#pause
$ s_j = q_"cls"^top k_j / sqrt(2) quad quad #text(size: 15pt, fill: MUTED)[(scores from the class token to each patch)] $
#pause
$ s = (thin underbrace(mat(1,1) dot mat(2,2), 4), thin 0, thin 0, thin underbrace(mat(1,1) dot mat(4,4), 8) thin) \/ sqrt(2) $
#pause
#align(center, text(size: 16pt, fill: MUTED)[patch $4$ scores highest, patch $1$ next — the class token will *lean toward the bright region*])

== Class-token output and the loss #D

Softmax the scores into weights, then blend the values:
#pause
$ o_"cls" = sum_j alpha_("cls", j) thin v_j quad quad #text(size: 15pt, fill: MUTED)[(a content-dependent summary of the patches)] $
#pause
Classify from it, then backprop the cross-entropy:
$ z = W_c thin o_"cls" + b_c, quad p = "softmax"(z), quad cal(L) = -log p_y $
#pause
#notebox[The gradient flows back into *everything*: the classifier, attention $W_Q, W_K, W_V$, the MLP, the class token, positions, and the patch projection $E$.]

== Interactive: tiny ViT #I

#interbox(link-to: IA + "vision-transformer")[
  Paint a tiny image, watch it patchify and project, and see the class token's attention scores $s_j$ shift toward the patches you brighten.
]

// ═══════════════════════════ PART VI — ViT vs CNN ═══════════════════════════
= ViT vs CNN

== Inductive bias #D

The two architectures build in *different amounts* of prior knowledge about images:
#pause
#two(
  notebox[*CNN* — bakes in *locality*, *translation equivariance* and *hierarchy*. Strong image priors, works from little data.],
  notebox[*ViT* — bakes in *far less*. A patch can attend anywhere; the model must *learn* spatial structure from data.],
)
#pause
#align(center, text(size: 16pt, fill: MUTED)[fewer built-in priors $->$ more flexible, but *hungrier* for data])

== Data efficiency #D

The ViT trade-off in one line:
#pause
- ViT *underperforms* CNNs on small datasets — it has no locality prior to fall back on;
#pause
- but *given enough data* (large-scale pretraining $+$ transfer), it *matches or beats* CNNs.
#pause
#notebox[*DeiT* (Touvron et al.) showed strong augmentation, regularization and *distillation* can train a competitive convolution-free Transformer on ImageNet *alone* — no giant private dataset needed.]

== CNN vs ViT, side by side #V

#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 6pt), align: (left, left, left),
  table.header([], [*CNN*], [*ViT*]),
  [core unit], [convolution], [self-attention],
  [spatial prior], [strong (built in)], [weak (learned)],
  [receptive field], [grows with depth], [global per block],
  [cost vs resolution], [linear in pixels], [quadratic in patches],
  [data efficiency], [strong], [recipe-sensitive],
))
#pause
#align(center, text(size: 15pt, fill: MUTED)[same task, opposite philosophies: hand-built structure vs structure learned from scale])

== The patch-size trade-off #D

Patch size $P$ sets both the detail and the cost:
#pause
$ N = (H W) / P^2, quad quad #text[attention cost] ~ O(N^2 d) $
#pause
- *smaller* $P$ $->$ *more* patches $->$ finer spatial detail, but a *longer* sequence;
- longer sequence $->$ the $N times N$ attention matrix grows *quadratically*.
#pause
#align(center, text(size: 16pt, fill: MUTED)[resolution of detail vs compute — the central knob of any ViT])

== Worked: the cost of halving $P$ #D #V

$224 times 224$ image, comparing $P = 16$ and $P = 8$:
#pause
#fig("/lecture17/figures/attn_cost.svg", w: 50%)
#pause
#align(center, text(size: 15pt, fill: MUTED)[$P: 16 -> 8$ quadruples $N$ ($196 -> 784$) and makes the attention matrix $approx 16 times$ bigger])

// ═══════════════════════════ PART VII — Hierarchical ViT (Swin) ═══════════════════════════
= Hierarchical ViT (Swin)

== Vanilla ViT is awkward for dense vision #D

Classification needs *one* global vector. But *detection* and *segmentation* need more:
#pause
- *multiple resolutions* (small and large objects);
- *high-resolution* feature maps (per-pixel prediction);
- *manageable* compute at high resolution.
#pause
#alertbox[Vanilla ViT keeps a *single, fixed* resolution and pays $O(N^2)$ globally — poorly suited to dense, high-resolution tasks.]

== A hierarchy of feature maps #D

*Swin* (Liu et al.) rebuilds the CNN-style pyramid inside a Transformer:
#pause
$ H/4 quad -> quad H/8 quad -> quad H/16 quad -> quad H/32 $
#pause
- spatial resolution *shrinks*, channel count *grows* — exactly like a CNN backbone;
- yields multi-scale maps that plug into detection / segmentation heads (FPN).
#pause
#align(center, text(size: 16pt, fill: MUTED)[a Transformer that *looks like* a conv backbone from the outside])

== Window attention #D

Instead of global attention, attend only *within* $M times M$ windows:
#pause
$ #text[global]: O(N^2) quad quad --> quad quad #text[windowed]: O(N dot M^2) $
#pause
#align(center, text(size: 16pt, fill: MUTED)[for a *fixed* window size $M$, cost is *linear* in the number of patches — scalable to high resolution])

== Shifted windows #V

Fixed windows never talk to each other. Swin *shifts* the partition every other block:
#pause
#fig("/lecture17/figures/windows.svg", w: 62%)
#pause
#align(center, text(size: 14pt, fill: MUTED)[block 1 uses standard windows; block 2 shifts them so neighbouring windows *exchange* information])

== Patch merging #D

To go *down* a level of the hierarchy, merge $2 times 2$ neighbouring patches:
#pause
$ H times W times C quad -->^"concat 2×2" quad H/2 times W/2 times 4C quad -->^"linear" quad H/2 times W/2 times 2C $
#pause
#align(center, text(size: 16pt, fill: MUTED)[halve the resolution, grow the channels — the Transformer analog of a strided conv / pool])

== ViT vs Swin #V

#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 6pt), align: (left, left, left),
  table.header([], [*ViT*], [*Swin*]),
  [resolution], [single, fixed], [hierarchical pyramid],
  [attention scope], [global (all patches)], [local windows $+$ shift],
  [cost in patches], [quadratic $O(N^2)$], [linear $O(N M^2)$],
  [best for], [classification], [detection, segmentation],
))
#pause
#interbox[#text(fill: MUTED)[(to build)] — a shifted-window widget: watch windows tile, then shift, so neighbours mix.]

// ═══════════════════════════ PART VIII — Vision pretraining ═══════════════════════════
= Vision pretraining (briefly)

== Supervised vs self-supervised #D

How do we *train* a vision backbone before transfer?
#pause
#two(
  notebox[*Supervised* — map image $I -> $ label $y$, minimize $cal(L)_"CE"$. Needs labels; the label vocabulary is *limited*.],
  notebox[*Self-supervised* — *no labels*. Build the target from the image itself: mask, augment, or match.],
)
#pause
#align(center, text(size: 16pt, fill: MUTED)[self-supervision unlocks *web-scale* image data without human annotation])

== Masked image modeling #D

The BERT idea, ported to patches: *hide* some, *predict* them:
#pause
- randomly *mask* a subset of patch tokens;
- the encoder predicts the *content* (or a learned representation) of the masked patches.
#pause
#align(center, text(size: 16pt, fill: MUTED)[reconstruct-the-missing forces the model to learn how patches relate — no labels required])

== Self-distilled features (DINOv2) #D

Match *two views* of the same image through a *teacher* and a *student*:
#pause
- student and teacher see *different augmentations*; the student is trained to *agree* with the teacher;
- the teacher is a slow-moving average of the student — no labels anywhere.
#pause
#notebox[*DINOv2* (Oquab et al.) produces strong *general-purpose* features this way — useful across many tasks *without* task labels.]

== Probing vs finetuning #D

Two ways to *use* a pretrained backbone with parameters $theta$:
#pause
#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 6pt), align: (left, center, left),
  table.header([*Method*], [*Update*], [*Measures*]),
  [linear probe], [freeze $theta$; train $y = W h + b$], [direct usability of features],
  [finetune], [update *all* of $theta$], [best task accuracy],
))
#pause
#align(center, text(size: 16pt, fill: MUTED)[a strong *linear probe* means the features are already *linearly* useful — the mark of good pretraining])

// ═══════════════════════════ PART IX — Image-text learning (CLIP) ═══════════════════════════
= Image-text learning: CLIP

== Why connect images and text? #D

A single label is a thin signal. A *caption* is far richer:
#pause
#align(center, text(size: 17pt)[
  "a brown dog running through snow"
])
#pause
#align(center, text(size: 15pt, fill: MUTED)[object *dog* · attribute *brown* · action *running* · scene *snow* · relations *through*])
#pause
#result[far more supervision than the one word "dog" — without manually annotating every image]
#pause
#align(center, text(size: 14pt, fill: MUTED)[web pairs are still noisy and require collection, filtering, and careful attention to provenance, bias, and consent])

== The dual encoder #V

*CLIP* (Radford et al. 2021): one encoder per modality, projected into a *shared* space:
#pause
#dualenc
#pause
#align(center, text(size: 14pt, fill: MUTED)[$v = f_"img"(I)$, $t = f_"text"(S)$; normalize $hat(v) = v \/ norm(v)$, $hat(t) = t \/ norm(t)$; compare by cosine $hat(v)^top hat(t)$])

== How CLIP trains #D

Take a *batch* of $B$ matched image-text pairs. Score *every* image against *every* caption:
#pause
$ S_(i j) = hat(v)_i^top hat(t)_j \/ tau quad quad #text(size: 15pt, fill: MUTED)[($tau$ = learned temperature)] $
#pause
- the $B$ *correct* pairs sit on the *diagonal* $S_(i i)$;
- all $B^2 - B$ off-diagonal pairs are *mismatches*.
#pause
#result[train the model to *pick the matching caption for each image* — and vice versa]

== The similarity matrix #V

#fig("/lecture17/figures/clip_matrix.svg", w: 37%)
#pause
#align(center, text(size: 15pt, fill: MUTED)[a $B times B$ grid of similarities; push the *diagonal* up and everything *off-diagonal* down])

== The contrastive loss #D

For image $i$, softmax across the *row* (all captions) and reward the correct one:
#pause
$ cal(L)_(I -> T)^((i)) = -log (exp(S_(i i))) / (sum_(j=1)^B exp(S_(i j))) $
#pause
Symmetrically, softmax down the *column* for the text-to-image direction, then average:
$ cal(L) = 1/2 (cal(L)_(I -> T) + cal(L)_(T -> I)) $
#pause
#align(center, text(size: 16pt, fill: MUTED)[a cross-entropy where the "classes" are the *other items in the batch*])

== Worked: contrastive loss #D

A batch of $B = 3$ with similarity matrix $S$ (rows = images, columns = captions):
#pause
$ S = mat(3, 1, 0; 0, 2, 1; 1, 0, 4) $
#pause
Image $1$: softmax across row $1 = (3, 1, 0)$:
$ p(T_1 | I_1) = e^3 / (e^3 + e^1 + e^0) = 20.09 / 23.81 approx #text(fill: ACC, weight: 600)[0.844] $
#pause
$ cal(L)_(I -> T)^((1)) = -log(0.844) approx #text(fill: ACC, weight: 600)[0.170] $

== Negatives come from the batch #D

CLIP has no separate negative set — the batch *is* the negatives:
#pause
- for image $i$, the positive is caption $T_i$; the negatives are all $T_(j != i)$ *in the same batch*;
- *bigger batch* $=$ *more negatives* $=$ a harder, more informative task.
#pause
#alertbox[*False negatives.* Two different images in a batch may genuinely match the same caption — the loss still pushes them apart. Large, diverse batches make this rare.]

== Interactive: contrastive loss #I

#interbox(link-to: IA + "clip-zero-shot")[
  Edit a batch of image-text similarities and watch the row/column softmax, the diagonal light up, and the symmetric contrastive loss respond.
]

// ═══════════════════════════ PART X — Zero-shot classification ═══════════════════════════
= Zero-shot classification

== Classify with no trained head #D

CLIP can classify into *new* classes it never had a labeled head for. Turn each class into text:
#pause
$ {"cat", "dog", "car"} quad -->^"prompt" quad "a photo of a" {"class"} $
#pause
$ t_"cat", quad t_"dog", quad t_"car" quad quad #text(size: 15pt, fill: MUTED)[(encode each prompt with the text tower)] $
#pause
#align(center, text(size: 16pt, fill: MUTED)[the class names become *text embeddings* — no supervised classifier needed])

== Compare the image to each class #D

Encode the image once, then score it against every class text:
#pause
$ v = f_"img"(I), quad quad s_c = v^top t_c \/ tau $
#pause
$ p(c | I) = "softmax"_c (s_c) $
#pause
#align(center, text(size: 16pt, fill: MUTED)[add a new class? just add its *prompt* — no retraining, no labeled examples])

== Worked: zero-shot prediction #D

Similarities of the image to three class prompts, $s = (4, 2, -1)$:
#pause
$ "softmax"(4, 2, -1) = (e^4, e^2, e^(-1)) / (e^4 + e^2 + e^(-1)) approx (#text(fill: ACC, weight: 600)[0.876], thin 0.118, thin 0.006) $
#pause
#result[predict *cat* — the highest-similarity class, with no trained classifier]

== Prompts and retrieval #D

Two practical extensions of the same idea:
#pause
- *prompt ensembling* — average the text embeddings of several templates ("a photo of a {c}", "a blurry photo of a {c}", ...) for a more robust class vector;
#pause
- *image-text retrieval* — rank a gallery by $cos(v_i, t_q)$ to find images for a text query (or captions for an image).
#pause
#interbox(link-to: IA + "clip-zero-shot")[
  Type your own class names, watch the prompts encode, and see the image's softmax over classes update live.
]

// ═══════════════════════════ PART XI — Fusion and vision-language models ═══════════════════════════
= Fusion and vision-language models

== Dual encoders: strengths and a limit #D

The dual encoder is fast and flexible — but shallow in how it combines modalities:
#pause
#two(
  notebox[*Strengths* — precompute embeddings, *fast retrieval*, zero-shot transfer, a simple contrastive objective.],
  alertbox[*Limit* — image and text meet *only* at the final $hat(v)^top hat(t)$. Too shallow for "is the *red cup* left of the *blue plate*?"],
)
#pause
#align(center, text(size: 16pt, fill: MUTED)[compositional, relational questions need *deeper* image-text interaction])

== Cross-attention fusion #V

Let text *query* the visual tokens: $Q$ from text, $K, V$ from the image.
#pause
#crossfuse
#align(center, text(size: 13pt, fill: MUTED)[$"CrossAttention"(Q_"text", K_"vision", V_"vision")$])

== The modern VLM pattern #V

Connect a *vision encoder* to a *language model* through a small *connector*:
#pause
#vlmpipe
#pause
#align(center, text(size: 14pt, fill: MUTED)[image $->$ vision encoder $->$ visual tokens $->$ connector $->$ LM token space $->$ text])

== The connector and the objective #D

The connector maps visual features into the LM's token space as a *visual prefix*:
#pause
$ tilde(V) = V W_p + b, quad quad W_p in RR^(d_v times d_"LM") $
#pause
Then generate text with the *same next-token objective* — now conditioned on the image:
$ cal(L) = -sum_(t) log p(y_t | y_(<t), I) $
#pause
#align(center, text(size: 16pt, fill: MUTED)[nothing new in the loss — the image just becomes extra *context tokens* for the LM])

== BLIP-2: bridge two frozen models #V

*BLIP-2* (Li et al.): keep the image encoder and the LM *frozen*, learn only a light bridge:
#pause
#blip2
#pause
#align(center, text(size: 14pt, fill: MUTED)[reuse strong unimodal models; train *only* the small Q-Former between them])

== Q-Former intuition #D

A handful of *learned queries* $q_1, dots, q_M$ pull the image down to a few language-ready tokens:
#pause
$ z_m = "CrossAttention"(q_m, thin V, thin V) quad quad #text(size: 15pt, fill: MUTED)[(queries attend to image features $V$)] $
#pause
#align(center, text(size: 16pt, fill: MUTED)[compress hundreds of patches into $M$ (e.g. 32) *language-relevant* visual tokens])

== LLaVA: a simple, strong recipe #D

*LLaVA* (Liu et al.) shows a projector can be almost trivial:
#pause
$ #text(fill: TEAL)[CLIP vision encoder] quad -> quad #text(fill: ACC)[linear / MLP projector] quad -> quad #text(fill: INK)[LLM] $
#pause
- project CLIP patch features straight into the LM's embedding space;
- *visual instruction tuning* — train on (image, instruction, response) triples.
#pause
#align(center, text(size: 16pt, fill: MUTED)[a tiny projector plus instruction data turns an LLM into a capable vision-language model])

== Visual instruction tuning #D

Train the model to *follow instructions about images*, supervising only the *response*:
#pause
$ cal(L) = -sum_(t) log p(y_t | y_(<t), I, "instruction") $
#pause
#notebox[The instruction tokens are *masked out* of the loss — we only teach the model to produce the *answer*, not to re-predict the question.]

== What to freeze, what to train #D

The main design axis of a VLM:
#pause
#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 6pt), align: (left, left),
  table.header([*Strategy*], [*Trade-off*]),
  [freeze vision $+$ LM, train connector], [cheapest; least forgetting; limited alignment],
  [freeze vision, finetune LM], [better alignment; risk of forgetting],
  [LoRA on the LM], [cheap adaptation, keeps base intact],
  [joint finetune], [strongest; most compute, most forgetting],
))
#pause
#interbox(link-to: IA + "vlm")[
  Send an image through a vision encoder and connector into an LM and watch visual tokens become a prefix the model attends to.
]

// ═══════════════════════════ PART XII — Limitations ═══════════════════════════
= Limitations

== Spatial and fine-grained reasoning #D

Fluent captions can still get the *details* wrong:
#pause
- *counting* objects; locating *small* objects; *left / right* and precise coordinates;
- reading *fine text*; exact spatial relations.
#pause
#align(center, text(size: 16pt, fill: MUTED)[fluent language about an image $!=$ *grounded* understanding of it])

== Hallucination #D

The classic failure of vision-language models:
#pause
#alertbox[*Hallucination* — the model produces *plausible, fluent* text that the image does *not* support (an object that isn't there, a wrong colour or count).]
#pause
*Cause* — a *strong language prior* (what usually goes together) combined with *imperfect grounding* in the actual pixels.
#pause
#result[language fluency $!=$ visual correctness]

== Data, bias, and resolution #D

#pause
- *data & bias* — web captions are *noisy*, and carry *stereotypes* and *imbalance* straight into the model;
#pause
- *resolution bottleneck* — high-res images make *many* tokens ($N = H W \/ P^2$), expensive to feed an LLM;
#pause
#align(center, text(size: 15pt, fill: MUTED)[mitigations: downsample · pool · learned query tokens · image *tiling*])

== Evaluation #D

Different tasks, different metrics:
#pause
#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 6pt), align: (left, left),
  table.header([*Task*], [*Metric*]),
  [classification], [accuracy],
  [retrieval], [Recall@$k$],
  [captioning], [CIDEr / BLEU / human],
  [visual question answering], [VQA accuracy],
))
#pause
#align(center, text(size: 15pt, fill: MUTED)[automatic metrics are imperfect — human evaluation still matters for open-ended generation])

// ═══════════════════════════ PART XIII — Summary ═══════════════════════════
= Summary

== One pipeline, several models #V

Every model in this lecture starts the *same way* — patchify, then a Transformer:
#pause
#align(center, diagram(spacing: (10mm, 7mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 0), align(center, text(size: 9.5pt)[image $->$ patch \ tokens $->$ ViT]), shape: rect, corner-radius: 4pt, inset: 7pt, fill: cream, stroke: 1pt + INK)
  let out(y, lbl, c) = node((2.6, y), align(center, text(size: 9pt, lbl)), shape: rect, corner-radius: 4pt, inset: 6pt, fill: c.lighten(82%), stroke: 0.9pt + c)
  out(1.5, [classifier \ = *ViT*], TEAL)
  out(0.5, [general rep \ = *DINO*], BLUE)
  out(-0.5, [text-aligned \ = *CLIP*], ACC)
  out(-1.5, [connector $-> $ LM \ = *VLM*], GREEN)
  for (y, c) in ((1.5, TEAL), (0.5, BLUE), (-0.5, ACC), (-1.5, GREEN)) {
    edge((0, 0), (2.6, y), "-|>", stroke: 0.7pt + c)
  }
}))

== Architecture comparison #V

#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 10pt, y: 6pt), align: (left, left, left),
  table.header([*Model*], [*Main output*], [*Training signal*]),
  [ViT], [class label], [supervised labels],
  [self-sup ViT (DINOv2)], [general features], [self-supervision, no labels],
  [CLIP], [image-text similarity], [contrastive on pairs],
  [BLIP-2], [text from image], [frozen towers $+$ Q-Former],
  [LLaVA], [instruction response], [visual instruction tuning],
))

== The key equations #V

#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 7pt), align: (left, left),
  table.header([*Idea*], [*Equation*]),
  [patch sequence], [$Z_0 = [z_"cls"; z_1; dots; z_N] + E_"pos"$],
  [attention], [$A = "softmax"(Q K^top \/ sqrt(d))$],
  [CLIP similarity], [$S_(i j) = hat(v)_i^top hat(t)_j \/ tau$],
  [multimodal LM], [$p(y_(1:T) | I) = product_t p(y_t | y_(<t), I)$],
))

== Final mental model — one idea

#align(center, text(size: 21pt)[
  *ViT*: patches become tokens; self-attention lets regions exchange information.
  #v(6pt)
  *CLIP*: images and language are aligned in one shared space.
  #v(6pt)
  *VLMs*: visual tokens are fed to a language model.
  #v(8pt)
  #text(size: 17pt, fill: MUTED)[one front-end — patchify — powers classification, features, alignment and generation]
])

#focus-slide[
  A Transformer sees an image as a *sequence of patches*.
  #v(12pt)
  #set text(size: 22pt)
  Next: *Representation Learning & Autoencoders* $->$ VAEs $->$ GANs $->$ diffusion.
]
