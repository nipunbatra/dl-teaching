// From Characters to Transformers — 3 × 80-minute mega-lecture
// Compile from the repository root:
//   typst compile --root . --input handout=true lecture11/L11M-from-characters-to-transformers.typ slides-pdf/L11M.pdf
//   typst compile --root . lecture11/L11M-from-characters-to-transformers.typ slides-pdf/L11M-presentation.pdf

#import "../common/metropolis.typ": *
#import "../common/mldiag.typ": *
#import "mega/helpers.typ": *

#show: metropolis-deck.with(
  title: [From Characters to Transformers],
  subtitle: [Next-token prediction, attention, and a decoder-only language model],
)

#title-slide()

#include "mega/lecture1.typ"
#include "mega/lecture2.typ"
#include "mega/lecture3.typ"

= References and lecture plan

== References #I

#set text(size: 15pt)

#grid(columns: (1fr, 1fr), gutter: 12pt, row-gutter: 10pt,
  card([NEXT-TOKEN PREDICTION], [
    #link("https://nipunbatra.github.io/ml-teaching/neural-networks/slides/next-token-prediction.pdf")[Nipun Batra · Next Token Prediction]
    #v(3pt)
    `aabid`, embeddings, concatenation, classification, sampling
  ], color: TEAL),
  card([SEQUENCE MODELS], [
    #link("https://introtodeeplearning.com/slides/6S191_MIT_DeepLearning_L2.pdf")[MIT 6.S191 · Deep Sequence Modeling]
    #v(3pt)
    queries, keys, values, and self-attention
  ], color: BLUE),
  card([LANGUAGE MODELS], [
    #link("https://www.youtube.com/watch?v=Ub3GoFaUcds")[Stanford CME 295 · Lecture 1: Transformer]
    #v(3pt)
    language tasks, tokenization, and the full Transformer
  ], color: ACC),
  card([THE ORIGINAL TRANSFORMER], [
    #link("https://arxiv.org/abs/1706.03762")[Vaswani et al. · Attention Is All You Need]
    #v(3pt)
    scaled dot-product attention and the encoder-decoder Transformer
  ], color: GREEN),
)

#v(8pt)
#note([Technical diagrams were drawn in Typst/Fletcher. Illustrations were generated with OpenAI image generation; labels were added in Typst. Schematic token splits and attention maps are labeled as teaching examples.], color: INK)

== The three lectures #I

#grid(
  columns: (32mm, 1fr), row-gutter: 18pt, column-gutter: 10pt, align: horizon,
  text(weight: 700, fill: TEAL)[Lecture 1],
  [Text $arrow.r$ tokens $arrow.r$ next-token prediction],
  text(weight: 700, fill: BLUE)[Lecture 2],
  [Context $arrow.r$ weighted averages $arrow.r$ attention],
  text(weight: 700, fill: ACC)[Lecture 3],
  [Multi-head attention $+$ MLP $+$ residual paths $arrow.r$ Transformer],
)

#v(22pt)
#two(
  card([SLIDES], [Use the presentation PDF to reveal each step. The handout shows completed slides.], color: TEAL),
  card([LECTURE PLAN], [The Markdown outline records examples, questions, and suggested stopping points.], color: ACC),
)
