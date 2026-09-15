#import "../common/cheatsheet.typ": *

#let query = rgb("#8530c7")
#let key = rgb("#a94c00")
#let value = rgb("#007a75")
#set page(paper: "a4", margin: (top: 9mm, bottom: 10mm, x: 10mm),
  footer: context align(center, text(size: 8pt, fill: palette.muted)[Attention and language · Part 2 · #counter(page).display() / 2]))
#set text(font: "IBM Plex Sans", size: 10.4pt, fill: palette.ink)
#set par(leading: 0.5em, justify: false)
#show math.equation: set text(font: "New Computer Modern Math")
#set list(indent: 10pt, body-indent: 4pt, spacing: 3pt)

#sheet-title([PART 2 · QUICK REFERENCE], [Self-attention: match, mix, project, add],
  subtitle: [A token keeps its current representation and adds information read from allowed source tokens.])
#banner[Q and K choose the weights. Those weights mix V. The output projection makes an update that can be added to E.]
#v(6pt)
#grid(columns: (1fr, 1fr), gutter: 6mm,
[
  #section-title([1 · Three roles, not three copies], accent: query)
  #card(accent: query)[
    #keyline([Query $q_i$:], [what the receiving token is looking for.], color: query)
    #v(3pt)
    #keyline([Key $k_j$:], [features used to decide whether source $j$ is a useful match.], color: key)
    #v(3pt)
    #keyline([Value $v_j$:], [information source $j$ contributes if read.], color: value)
    #v(5pt)
    *Directory analogy:* query = “contact the library”; key = “Library”; value = {extension 204, room 2B, hours 09:00-18:00}. The matching name does not supply the contact details.
    #v(4pt)
    Real attention uses numeric vectors, not literal names or records. The analogy explains the roles; a neural value is a learned representation of useful information.
  ]
  #v(5pt)
  #section-title([2 · First choose where to read], accent: key)
  #card(accent: key)[
    Receiver $i$ compares with source $j$:
    $ s_(i j) = (q_i k_j^top)/sqrt(d_k). $
    Normalize over the *allowed sources*:
    $ alpha_(i j) = exp(s_(i j))/(sum_(r<=i) exp(s_(i r))), quad j<=i. $
    Set future weights to zero. Each allowed weight is nonnegative; each row sums to 1. These weights depend on Q and K, not V.
  ]
  #v(5pt)
  #section-title([3 · Then mix what sources send], accent: value)
  #card(accent: value)[
    $ m_i = sum_(j<=i) alpha_(i j) v_j. $
    One scalar weight multiplies *every coordinate* of its source's value. All values share width $d_v$, so they can be added.
    #v(4pt)
    Hard retrieval returns one winner's value. Soft attention combines values using a distribution of weights. Changing only V leaves the weights fixed but can change the message.
  ]
],
[
  #section-title([4 · Follow the lecture's bank example], accent: palette.blue)
  #card(accent: palette.blue)[
    Prefix: *The fisherman sat beside the river bank*.
    Receiver is *bank* at $i=7$. In the hand-chosen teaching model, $d_"model"=4$, $d_k=3$, $d_v=2$.
    $ e_7=[0.70,0.70,0.10,0.80] $
    $ q_7=[1.26,1.26,0.04]. $
    #tiny-note[e axes: water, finance, person, glue. Q/K axes: water?, finance?, who? Value axes: water scene, finance scene.]
    #v(4pt)
    #grid(columns: (1.2fr, .75fr, 1fr), row-gutter: 3pt,
      [*Source*], [*Weight*], [*Value row*],
      [The], [0.050], [$(0.1,0.0)$],
      [fisherman], [0.226], [$(2.0,0.1)$],
      [sat], [0.059], [$(0.3,0.0)$],
      [beside], [0.067], [$(0.6,-0.1)$],
      [the], [0.054], [$(0.1,0.1)$],
      [river], [0.414], [$(3.1,-0.1)$],
      [bank], [0.130], [$(0.7,0.7)$])
    #v(4pt)
    Multiply each row by its weight, then add:
    $ m_7 approx [1.89, 0.07]. $
    For example, river contributes about $0.414 times [3.1,-0.1]=[1.283,-0.041]$.
  ]
  #v(5pt)
  #section-title([5 · Project before the residual addition], accent: palette.green)
  #card(accent: palette.green)[
    A two-number message cannot be added to a four-number representation. A learned map changes the coordinates and width:
    $ W_O = mat(0.8,-0.2,0.3,0.1; -0.1,1.1,0.2,-0.2). $
    $ Delta e_7 = m_7 W_O approx [1.51,-0.30,0.58,0.18]. $
    $ e_7^"new" = e_7 + Delta e_7 $
    $ approx [2.21,0.40,0.68,0.98]. $
    This is a *linear mixing operation*, not zero-padding or copying coordinates. “Projection” here does not mean an orthogonal projection. A bias can be included; this toy omits it.
  ]
])
#v(6pt)
#tiny-note[Values and weights are rounded for display. The toy's water/finance/person/glue labels are teaching aids, not a claim that trained models have named axes.]

#pagebreak()
#set text(size: 10pt)
#sheet-title([PART 2 · SHAPES AND SAFEGUARDS], [Where Q, K and V come from],
  subtitle: [Row-vector convention · T token positions · one attention head · biases omitted in the attention equations.])
#grid(columns: (1fr, 1fr), gutter: 6mm,
[
  #section-title([6 · Same matching width, different jobs], accent: query)
  #card(accent: query)[
    Each current row $e_i$ supplies all three vectors:
    $ q_i=e_i W_Q, quad k_i=e_i W_K, quad v_i=e_i W_V. $
    Q and K must share width $d_k$ for the dot product. *Equal shape does not require equal entries or equal mappings.*
    #v(4pt)
    For “Maya grabbed a coat because she was cold,” a query from *she* may ask for a person. Maya's key may advertise a person-like source; its value may contribute identity and number information.
    #v(4pt)
    $W_Q$ extracts what a token seeks; $W_K$ extracts what it can match; $W_V$ extracts what it sends. Tying the matrices is possible but restricts these roles. Each matrix is shared across positions in the head.
  ]
  #v(5pt)
  #section-title([7 · The complete matrix path], accent: palette.blue)
  #card(accent: palette.blue)[
    #grid(columns: (1.25fr, 1fr), row-gutter: 3pt, column-gutter: 6pt,
      [*Object*], [*Shape*],
      [current rows $E$], [$T times d_"model"$],
      [$W_Q, W_K$], [$d_"model" times d_k$],
      [$Q, K$], [$T times d_k$],
      [$W_V$], [$d_"model" times d_v$],
      [$V$, message $A V$], [$T times d_v$],
      [scores $S$, mask $M$,#linebreak()weights $A$], [$T times T$],
      [$W_O$], [$d_v times d_"model"$])
    $ S = Q K^top / sqrt(d_k) $
    $ A = "softmax"_("row")(S+M) $
    $ Delta E = (A V) W_O, quad E^"new" = E + Delta E. $
    *Residual route:* keep E unchanged until the addition. *Attention route:* compute its update. Both meet in the same representation space.
  ]
  #v(5pt)
  #section-title([8 · The next-token predictor still has an MLP], accent: palette.teal)
  #card(accent: palette.teal)[
    In this lecture's toy model, the final position uses
    $ h="ReLU"(e_T^"new" W_1+b_1) $
    $ z=h W_2+b_2, quad p="softmax"(z). $
    Widths: *4 → 8 hidden units → 20 logits*. Even *the* can ask a broad query; the preceding keys and values supply context-specific information. The toy prediction head is not the full Transformer architecture.
  ]
],
[
  #section-title([9 · Scaling controls score spread], accent: palette.orange)
  #card(accent: palette.orange)[
    Assume query and key coordinates are mutually independent, mean 0, variance 1. Then
    $ "Var"(q_l k_l)=1 $
    $ "Var"(q dot k)=d_k, quad "SD"(q dot k)=sqrt(d_k). $
    Variances of independent products add. Dividing the score by $sqrt(d_k)$ gives unit SD under these assumptions.
    #v(4pt)
    Entries *need not be ±1*: those make an easy simulation; independent standard-normal coordinates also satisfy the assumptions. Learned vectors need not satisfy them exactly.
    #v(4pt)
    For $d_k=64$, suppose scores are $[-8,8]$:
    $ p_A = exp(-8)/(exp(-8)+exp(8)) approx 0.000000113. $
    After division by 8, the scores are $[-1,1]$:
    $ p_A = exp(-1)/(exp(-1)+exp(1)) approx 0.1192. $
    $p_B=1-p_A$. Scaling avoids excessively sharp weights and tiny softmax derivatives. *Overflow is separate*: subtracting the maximum is numerically stable but does not reduce score gaps.
  ]
  #v(5pt)
  #section-title([10 · A causal mask blocks the answer], accent: palette.red)
  #card(accent: palette.red)[
    At position $i$, predict token $i+1$. Read positions $j<=i$, including the current token, but never future positions.
    $ M_(i j) = cases(0 & "if " j<=i, -infinity & "if " j>i). $
    *S* holds scaled match scores. *M* adds zero to allowed scores and $-infinity$ to forbidden scores. *A* is the resulting row-wise probability distribution.
    #v(4pt)
    Example at position 2, with four sources:
    $ [1,2,3,4]+[0,0,-infinity,-infinity] $
    $ arrow.r "softmax" arrow.r [0.269,0.731,0,0]. $
    Since $exp(-infinity)=0$, future values have zero weight. Apply the mask *before* softmax, not after it without renormalizing.
    #v(4pt)
    *Training:* the full sequence is available, but each position must not see its answer. *Generation:* future tokens have not been generated yet. The same causal rule keeps the two settings consistent.
  ]
])
#v(6pt)
#tiny-note[Sources: #link("https://nipunbatra.github.io/attention/attention.html")[Part 2 interactive lecture] (hand-chosen toy, not a trained language model); #link("https://arxiv.org/abs/1706.03762")[Vaswani et al., Attention Is All You Need (2017)]. The paper combines attention, feed-forward layers, position information, normalization and residual connections.]
