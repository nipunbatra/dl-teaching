// Making Deep Networks Trainable — Lecture 6 · ES 667 Deep Learning
// One running investigation: the 30-hidden-layer spiral MLP.
// Compile from the repository root:
//   typst compile --root . --input handout=true lecture6/L6-trainability.typ slides-pdf/L6.pdf
//   typst compile --root . lecture6/L6-trainability.typ slides-pdf/L6-presentation.pdf

#import "../common/metropolis.typ": *
#import "../common/mldiag.typ": *

#show: metropolis-deck.with(
  title: [Making Deep Networks Trainable],
  subtitle: [Vanishing gradients, initialization, normalization, and shortcuts],
)

#let NB = "https://colab.research.google.com/github/nipunbatra/dl-teaching/blob/master/notebooks/L06/deep_network_trainability_autopsy.ipynb"
#let HOOKS_NB = "https://colab.research.google.com/github/nipunbatra/dl-teaching/blob/master/notebooks/L06/00_pytorch_hooks_for_layerwise_diagnostics.ipynb"
#let SCALE_NB = "https://colab.research.google.com/github/nipunbatra/dl-teaching/blob/master/notebooks/L06/01_activation_scale_through_depth.ipynb"
#let GLOROT = "https://proceedings.mlr.press/v9/glorot10a.html"
#let HEINIT = "https://openaccess.thecvf.com/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html"
#let BN = "https://proceedings.mlr.press/v37/ioffe15.html"
#let LN = "https://arxiv.org/abs/1607.06450"
#let RESNET = "https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html"
#let ANDREW_VG = "https://www.youtube.com/watch?v=qhXZsFVxGKo"
#let ANDREW_INIT = "https://www.youtube.com/watch?v=s2coXdufOzE"
#let SYMMETRY_DEMO = "https://nipunbatra.github.io/interactive-articles/symmetry-breaking/"

#let small-label(body, color: MUTED) = text(
  font: "IBM Plex Mono", size: 10pt, weight: 650, tracking: 0.5pt,
  fill: color, upper(body),
)
#let circle-math(body, size: 10pt) = box(text(size: size, body))
#let metric(label, value, color: TEAL) = align(center, [
  #small-label(label, color: color)
  #v(3pt)
  #text(size: 24pt, weight: 650, fill: color)[#value]
])
#let scale-card(layer, value, color: TEAL, width: 28mm, value-size: 17pt) = block(
  width: width, height: 21mm, inset: 6pt, radius: 3pt,
  stroke: 1pt + color, fill: color.lighten(94%),
  align(center + horizon, [
    #small-label(layer, color: color)
    #v(3pt)
    #text(size: value-size, weight: 650, fill: color)[#value]
  ]),
)
#let matrix-cell(body, fill: white, stroke: 0.5pt + MUTED) = box(
  width: 22mm, height: 12mm, fill: fill, stroke: stroke, radius: 2pt,
  align(center + horizon, body),
)
#let batch-cell(body, width: 18mm, fill: white, stroke: 0.45pt + MUTED) = box(
  width: width, height: 10mm, fill: fill, stroke: stroke,
  align(center + horizon, text(size: 9.5pt, body)),
)
#let gate-cell(body, fill: white, stroke: 0.6pt + MUTED, size: 13.5pt) = block(
  width: 100%, height: 17mm, fill: fill, stroke: stroke, radius: 2pt,
  align(center + horizon, text(size: size, body)),
)

#let rig-box(body, color, width) = block(
  width: width, inset: (x: 7pt, y: 7pt), radius: 3pt,
  stroke: 1pt + color, align(center + horizon, body),
)
#let experiment-rig = align(center, grid(
  columns: (22mm, 8mm, 43mm, 8mm, 66mm, 8mm, 39mm),
  gutter: 3pt, align: horizon,
  rig-box([#text(size: 12pt)[$2$ inputs]], TEAL, 22mm),
  align(center, text(size: 20pt, fill: TEAL)[$arrow.r$]),
  rig-box([#text(size: 13pt)[$2 arrow.r 128$] #linebreak() #text(size: 10pt)[Linear + ReLU stem]], INK, 43mm),
  align(center, text(size: 20pt, fill: TEAL)[$arrow.r$]),
  rig-box([#text(size: 13pt)[$128 arrow.r 128$] #linebreak() #text(size: 10pt)[Linear + ReLU × 29]], INK, 66mm),
  align(center, text(size: 20pt, fill: TEAL)[$arrow.r$]),
  rig-box([#text(size: 13pt)[$128 arrow.r 3$] #linebreak() #text(size: 10pt)[linear logits]], ACC, 39mm),
))

// A node-level view of the full model. Hidden columns show representative
// neurons; the vertical dots stand for the remaining coordinates.
#let network-notation-diagram = align(center, diagram(
  spacing: (17mm, 9mm), node-stroke: 0.8pt + INK, node-fill: white,
  {
    let hidden-x = (1.65, 3.85, 6.05)
    let hidden-y = (-1.05, 1.05)
    let output-y = (-0.82, 0, 0.82)

    // Fully connected edges at the visible ends of the network.
    for yi in (-0.55, 0.55) {
      for yh in hidden-y {
        edge((0, yi), (1.65, yh), stroke: 0.55pt + MUTED.lighten(20%))
      }
    }
    for yh in hidden-y {
      for yo in output-y {
        edge((6.05, yh), (7.7, yo), stroke: 0.55pt + MUTED.lighten(20%))
      }
    }

    // The two compressed spans represent hidden layers 2..ell-1 and ell+1..29.
    edge((1.65, 0), (2.75, 0), "-|>", stroke: 0.9pt + MUTED)
    edge((2.75, 0), (3.85, 0), "-|>", stroke: 0.9pt + MUTED)
    edge((3.85, 0), (4.95, 0), "-|>", stroke: 0.9pt + MUTED)
    edge((4.95, 0), (6.05, 0), "-|>", stroke: 0.9pt + MUTED)

    node((0, -0.55), text(fill: white, size: 9pt)[$x_1$],
      radius: 4.3mm, fill: INK, stroke: none)
    node((0, 0.55), text(fill: white, size: 9pt)[$x_2$],
      radius: 4.3mm, fill: INK, stroke: none)

    for x in hidden-x {
      for y in hidden-y {
        node((x, y), none, radius: 3.8mm, fill: TEAL, stroke: none)
      }
      node((x, 0), $dots.v$, stroke: none)
    }
    node((2.75, 0), $dots.h$, stroke: none)
    node((4.95, 0), $dots.h$, stroke: none)

    for (k, y) in output-y.enumerate() {
      node((7.7, y), text(size: 8.5pt, fill: white)[$o_#k$],
        radius: 4.3mm, fill: ACC, stroke: none)
    }

    node((0, 1.85), text(size: 11pt, weight: 650)[$bold(h)_0=bold(x)$], stroke: none)
    node((1.65, 1.85), text(size: 11pt, weight: 650, fill: TEAL)[$bold(h)_1$], stroke: none)
    node((2.75, 1.85), text(size: 9.5pt, fill: MUTED)[layers $2 dots ell-1$], stroke: none)
    node((3.85, 1.85), text(size: 11pt, weight: 650, fill: TEAL)[$bold(h)_ell$], stroke: none)
    node((4.95, 1.85), text(size: 9.5pt, fill: MUTED)[layers $ell+1 dots 29$], stroke: none)
    node((6.05, 1.85), text(size: 11pt, weight: 650, fill: TEAL)[$bold(h)_30$], stroke: none)
    node((7.7, 1.85), text(size: 11pt, weight: 650, fill: ACC)[logits $bold(o)$], stroke: none)
  },
))

#let one-neuron-diagram = align(center, diagram(
  spacing: (24mm, 15mm), node-stroke: 0.9pt + INK, node-fill: white,
  {
    edge((0, -1.35), (2.35, 0), "-|>", stroke: 1.6pt + TEAL,
      label: text(size: 8.5pt, fill: TEAL)[$W_(ell,j 1)$], label-side: left)
    edge((0, -0.45), (2.35, 0), "-|>", stroke: 1.6pt + TEAL,
      label: text(size: 8.5pt, fill: TEAL)[$W_(ell,j 2)$], label-side: right)
    edge((0, 1.35), (2.35, 0), "-|>", stroke: 1.6pt + TEAL,
      label: text(size: 8.5pt, fill: TEAL)[$W_(ell,j d_(ell-1))$], label-side: right)
    edge((2.35, 0), (3.85, 0), "-|>", stroke: 1.1pt + ACC)
    edge((3.85, 0), (5.25, 0), "-|>", stroke: 1.1pt + GREEN)

    node((0, -1.35), circle-math([$h_(ell-1,1)$], size: 8.5pt), radius: 9.5mm)
    node((0, -0.45), circle-math([$h_(ell-1,2)$], size: 8.5pt), radius: 9.5mm)
    node((0, 0.45), $dots.v$, stroke: none)
    node((0, 1.35), circle-math([$h_(ell-1,d_(ell-1))$], size: 7pt), radius: 12.5mm)
    node((2.35, 0), circle-math([$z_(ell,j)$], size: 10pt), radius: 8.5mm, stroke: 1.4pt + ACC)
    node((3.85, 0), [ReLU], shape: fletcher.shapes.rect, corner-radius: 3pt,
      inset: 8pt, fill: ACC.lighten(91%), stroke: 1.1pt + ACC)
    node((5.25, 0), circle-math([$h_(ell,j)$], size: 10pt), radius: 8.5mm, stroke: 1.4pt + GREEN)
  },
))

#let scaled-depth-diagram = align(center, diagram(
  spacing: (25mm, 11mm), node-stroke: 0.9pt + INK, node-fill: white,
  {
    node((0, 0), circle-math([$bold(h)_0$], size: 11pt), radius: 9mm, stroke: 1.2pt + INK)
    node((1.65, 0), circle-math([$bold(h)_1^((alpha))$], size: 10.5pt), radius: 9mm, stroke: 1.2pt + TEAL)
    node((3.3, 0), circle-math([$bold(h)_2^((alpha))$], size: 10.5pt), radius: 9mm, stroke: 1.2pt + TEAL)
    node((4.55, 0), $dots.h$, stroke: none)
    node((5.8, 0), circle-math([$bold(h)_ell^((alpha))$], size: 10.5pt), radius: 9mm, stroke: 1.2pt + ACC)
    edge((0, 0), (1.65, 0), "-|>", stroke: 1.2pt + TEAL,
      label: text(size: 10pt, fill: TEAL)[$alpha W_1$ then ReLU])
    edge((1.65, 0), (3.3, 0), "-|>", stroke: 1.2pt + TEAL,
      label: text(size: 10pt, fill: TEAL)[$alpha W_2$ then ReLU])
    edge((3.3, 0), (4.55, 0), "-|>", stroke: 1pt + MUTED)
    edge((4.55, 0), (5.8, 0), "-|>", stroke: 1.2pt + ACC,
      label: text(size: 10pt, fill: ACC)[$alpha W_ell$ then ReLU])
  },
))

#let fan-in-diagram = align(center, diagram(
  spacing: (22mm, 13mm), node-stroke: 0.9pt + INK, node-fill: white,
  {
    edge((0, -1.35), (2, 0), "-|>", stroke: 1pt + TEAL)
    edge((0, -0.45), (2, 0), "-|>", stroke: 1pt + TEAL)
    edge((0, 0.45), (2, 0), "-|>", stroke: 1pt + TEAL)
    edge((0, 1.35), (2, 0), "-|>", stroke: 1pt + TEAL)
    node((0, -1.35), circle-math([$h_(ell-1,1)$], size: 8.5pt), radius: 9.5mm)
    node((0, -0.45), circle-math([$h_(ell-1,2)$], size: 8.5pt), radius: 9.5mm)
    node((0, 0.45), $dots.v$, stroke: none)
    node((0, 1.35), circle-math([$h_(ell-1,d_(ell-1))$], size: 7pt), radius: 12.5mm)
    node((2, 0), circle-math([$z_(ell,j)$], size: 10pt), radius: 8.5mm, stroke: 1.2pt + TEAL)
  },
))

// Generic fan-in/fan-out diagrams used before the full layer notation is introduced.
#let simple-fan-in-diagram = align(center, diagram(
  spacing: (22mm, 8.5mm), node-stroke: 0.9pt + INK, node-fill: white,
  {
    edge((0, -1.2), (2, 0), "-|>", stroke: 1pt + TEAL,
      label: text(size: 9pt, fill: TEAL)[$w_1$])
    edge((0, -0.4), (2, 0), "-|>", stroke: 1pt + TEAL,
      label: text(size: 9pt, fill: TEAL)[$w_2$])
    edge((0, 1.2), (2, 0), "-|>", stroke: 1pt + TEAL,
      label: text(size: 9pt, fill: TEAL)[$w_n$])
    node((0, -1.2), $h_1$, radius: 7mm)
    node((0, -0.4), $h_2$, radius: 7mm)
    node((0, 0.4), $dots.v$, stroke: none)
    node((0, 1.2), $h_n$, radius: 7mm)
    node((2, 0), $z$, radius: 8mm, stroke: 1.2pt + TEAL)
  },
))

#let simple-fan-out-diagram = align(center, diagram(
  spacing: (22mm, 8.5mm), node-stroke: 0.9pt + INK, node-fill: white,
  {
    edge((0, 0), (2, -1.2), "-|>", stroke: 1pt + BLUE,
      label: text(size: 9pt, fill: BLUE)[$w_(1i)$])
    edge((0, 0), (2, -0.4), "-|>", stroke: 1pt + BLUE,
      label: text(size: 9pt, fill: BLUE)[$w_(2i)$])
    edge((0, 0), (2, 1.2), "-|>", stroke: 1pt + BLUE,
      label: text(size: 9pt, fill: BLUE)[$w_(n_"out" i)$])
    node((0, 0), $h_i$, radius: 8mm, stroke: 1.2pt + BLUE)
    node((2, -1.2), $z_1$, radius: 7mm)
    node((2, -0.4), $z_2$, radius: 7mm)
    node((2, 0.4), $dots.v$, stroke: none)
    node((2, 1.2), circle-math([$z_(n_"out")$], size: 9.5pt), radius: 8mm)
  },
))

#let contribution-sum-diagram(term-rms, output-rms, color: TEAL) = align(center, diagram(
  spacing: (20mm, 8mm), node-stroke: 0.9pt + INK, node-fill: white,
  {
    edge((0, -1.2), (2, 0), "-|>", stroke: 1pt + color)
    edge((0, -0.4), (2, 0), "-|>", stroke: 1pt + color)
    edge((0, 1.2), (2, 0), "-|>", stroke: 1pt + color)
    edge((2, 0), (3.55, 0), "-|>", stroke: 1.2pt + color)
    node((0, -1.2), $u_1$, radius: 6.5mm)
    node((0, -0.4), $u_2$, radius: 6.5mm)
    node((0, 0.4), $dots.v$, stroke: none)
    node((0, 1.2), $u_100$, radius: 7mm)
    node((2, 0), $sum$, radius: 8mm, stroke: 1.2pt + color)
    node((3.55, 0), $z$, radius: 8mm, stroke: 1.2pt + color)
    node((0, -1.9), text(size: 13.5pt, weight: 650, fill: color)[100 independent terms], stroke: none)
    node((0, 1.95), text(size: 13.5pt, fill: color)[#term-rms], stroke: none)
    node((3.55, 1.15), text(size: 14pt, weight: 650, fill: color)[#output-rms], stroke: none)
  },
))

#let fan-out-diagram = align(center, diagram(
  spacing: (22mm, 13mm), node-stroke: 0.9pt + INK, node-fill: white,
  {
    edge((0, 0), (2, -1.2), "-|>", stroke: 1pt + BLUE)
    edge((0, 0), (2, -0.4), "-|>", stroke: 1pt + BLUE)
    edge((0, 0), (2, 0.4), "-|>", stroke: 1pt + BLUE)
    edge((0, 0), (2, 1.2), "-|>", stroke: 1pt + BLUE)
    node((0, 0), circle-math([$h_(ell-1,i)$], size: 8.5pt), radius: 10.5mm, stroke: 1.2pt + BLUE)
    node((2, -1.2), circle-math([$z_(ell,1)$], size: 9pt), radius: 8.5mm)
    node((2, -0.4), circle-math([$z_(ell,2)$], size: 9pt), radius: 8.5mm)
    node((2, 0.4), $dots.v$, stroke: none)
    node((2, 1.2), circle-math([$z_(ell,d_ell)$], size: 8pt), radius: 10mm)
  },
))

#let chain-diagram = align(center, diagram(
  spacing: (18mm, 10mm), node-stroke: 0.9pt + INK, node-fill: white,
  {
    for i in range(7) {
      node((i, 0), if i == 0 { $h_0$ } else if i == 6 { $h_L$ } else { $h_#i$ },
        radius: 5.8mm, stroke: 1pt + if i == 0 { TEAL } else if i == 6 { ACC } else { INK })
    }
    for i in range(6) {
      edge((i, 0), (i + 1, 0), "-|>", stroke: 1.05pt + TEAL,
        label: text(size: 9pt, fill: TEAL)[$f_#(i+1)$], label-sep: 3pt)
    }
    edge((6, 1.22), (0, 1.22), "-|>", stroke: 1.3pt + BLUE,
      label: text(size: 10.5pt, fill: BLUE)[$g_(ell-1)=J_ell^top g_ell$ at every layer],
      label-side: left)
  },
))

#let clone-diagram = align(center, diagram(
  spacing: (28mm, 14mm), node-stroke: 1pt + INK, node-fill: white,
  {
    node((0, 0), circle-math([$x=2$], size: 8.5pt), radius: 10.5mm, stroke: 1.1pt + TEAL)
    node((1.8, -0.65), circle-math([$h_1$], size: 11pt), radius: 8mm, stroke: 1.1pt + ACC)
    node((1.8, 0.65), circle-math([$h_2$], size: 11pt), radius: 8mm, stroke: 1.1pt + ACC)
    node((3.8, 0), circle-math([$hat(y)$], size: 11pt), radius: 8mm, stroke: 1.1pt + INK)
    edge((0, 0), (1.8, -0.65), text(size: 10pt)[$w_1=0.5$], "-|>", stroke: 1pt + TEAL)
    edge((0, 0), (1.8, 0.65), text(size: 10pt)[$w_2=0.5$], "-|>", stroke: 1pt + TEAL)
    edge((1.8, -0.65), (3.8, 0), text(size: 10pt)[$v_1=0.25$], "-|>", stroke: 1pt + ACC)
    edge((1.8, 0.65), (3.8, 0), text(size: 10pt)[$v_2=0.25$], "-|>", stroke: 1pt + ACC)
  },
))

#let relu-case-diagram(z-body, h-body, gate-body, color) = align(center, diagram(
  spacing: (13mm, 10mm), node-stroke: 1pt + INK, node-fill: white,
  {
    node((0, 0), $x$, radius: 6.5mm, stroke: 1.1pt + INK)
    node((1.55, 0), circle-math(z-body, size: 8.5pt), radius: 11mm, stroke: 1.2pt + color)
    node((3.15, 0), gate-body, shape: fletcher.shapes.rect,
      corner-radius: 3pt, inset: 8pt,
      fill: color.lighten(93%), stroke: 1.2pt + color)
    node((4.75, 0), circle-math(h-body, size: 8.5pt), radius: 10mm, stroke: 1.2pt + color)
    edge((0, 0), (1.55, 0), "-|>", stroke: 1pt + INK)
    edge((1.55, 0), (3.15, 0), "-|>", stroke: 1.2pt + color)
    edge((3.15, 0), (4.75, 0), "-|>", stroke: 1.2pt + color)
  },
))

#let zero-init-diagram = align(center, diagram(
  spacing: (19mm, 12mm), node-stroke: 1pt + INK, node-fill: white,
  {
    node((0, 0), circle-math([$x=2$], size: 8pt), radius: 10mm, stroke: 1.1pt + TEAL)
    node((1.55, -0.7), circle-math([$z_1=0$], size: 8pt), radius: 10.5mm, stroke: 1.1pt + RED)
    node((1.55, 0.7), circle-math([$z_2=0$], size: 8pt), radius: 10.5mm, stroke: 1.1pt + RED)
    node((3.15, -0.7), circle-math([$h_1=0$], size: 8pt), radius: 10.5mm, stroke: 1.1pt + RED)
    node((3.15, 0.7), circle-math([$h_2=0$], size: 8pt), radius: 10.5mm, stroke: 1.1pt + RED)
    node((4.9, 0), circle-math([$hat(y)=0$], size: 7.5pt), radius: 11.5mm, stroke: 1.1pt + INK)

    edge((0, 0), (1.55, -0.7), "-|>", stroke: 1pt + TEAL,
      label: text(size: 9.5pt)[$w_1=0$])
    edge((0, 0), (1.55, 0.7), "-|>", stroke: 1pt + TEAL,
      label: text(size: 9.5pt)[$w_2=0$])
    edge((1.55, -0.7), (3.15, -0.7), "-|>", stroke: 1pt + RED,
      label: text(size: 9.5pt)[ReLU])
    edge((1.55, 0.7), (3.15, 0.7), "-|>", stroke: 1pt + RED,
      label: text(size: 9.5pt)[ReLU])
    edge((3.15, -0.7), (4.9, 0), "-|>", stroke: 1pt + ACC,
      label: text(size: 9.5pt)[$v_1=0$])
    edge((3.15, 0.7), (4.9, 0), "-|>", stroke: 1pt + ACC,
      label: text(size: 9.5pt)[$v_2=0$])
  },
))

#let residual-diagram = align(center, diagram(
  spacing: (24mm, 14mm), node-stroke: 0.9pt + INK, node-fill: white,
  {
    node((0, 0), $h_ell$, radius: 8mm, stroke: 1.1pt + TEAL)
    node((1.8, 0), $F_ell$, shape: fletcher.shapes.rect, corner-radius: 3pt,
      inset: 10pt, fill: ACC.lighten(90%), stroke: 1.1pt + ACC)
    node((3.6, 0), $+$, radius: 6.5mm, stroke: 1pt + INK)
    node((5.3, 0), $h_(ell+1)$, radius: 9mm, stroke: 1.1pt + TEAL)
    edge((0, 0), (1.8, 0), "-|>", stroke: 1.1pt + TEAL)
    edge((1.8, 0), (3.6, 0), "-|>", stroke: 1.1pt + ACC)
    edge((3.6, 0), (5.3, 0), "-|>", stroke: 1.1pt + TEAL)
    edge((0, 0), (3.6, 0), "-|>", bend: -48deg, stroke: 1.6pt + GREEN,
      label: text(size: 11pt, weight: 650, fill: GREEN)[identity], label-side: left)
  },
))

#title-slide()

= Why gradients vanish or explode

== With identity activations, $hat(bold(y))=W_L dots W_1 bold(x)$ #V

#align(center, diagram(
  spacing: (20mm, 11mm), node-stroke: 0.9pt + INK, node-fill: white,
  {
    node((0, 0), $bold(x)$, radius: 7mm, stroke: 1.1pt + INK)
    node((1.4, 0), $W_1$, shape: fletcher.shapes.rect, corner-radius: 3pt,
      inset: 7pt, fill: TEAL.lighten(92%), stroke: 1.1pt + TEAL)
    node((2.8, 0), $bold(h)_1$, radius: 7mm, stroke: 1.1pt + TEAL)
    node((4.2, 0), $W_2$, shape: fletcher.shapes.rect, corner-radius: 3pt,
      inset: 7pt, fill: TEAL.lighten(92%), stroke: 1.1pt + TEAL)
    node((5.6, 0), $bold(h)_2$, radius: 7mm, stroke: 1.1pt + TEAL)
    node((6.75, 0), $dots.h$, stroke: none)
    node((7.9, 0), $W_L$, shape: fletcher.shapes.rect, corner-radius: 3pt,
      inset: 7pt, fill: ACC.lighten(92%), stroke: 1.1pt + ACC)
    node((9.3, 0), $hat(bold(y))$, radius: 8mm, stroke: 1.1pt + ACC)
    edge((0, 0), (1.4, 0), "-|>", stroke: 0.9pt + MUTED)
    edge((1.4, 0), (2.8, 0), "-|>", stroke: 0.9pt + MUTED)
    edge((2.8, 0), (4.2, 0), "-|>", stroke: 0.9pt + MUTED)
    edge((4.2, 0), (5.6, 0), "-|>", stroke: 0.9pt + MUTED)
    edge((5.6, 0), (6.75, 0), "-|>", stroke: 0.9pt + MUTED)
    edge((6.75, 0), (7.9, 0), "-|>", stroke: 0.9pt + MUTED)
    edge((7.9, 0), (9.3, 0), "-|>", stroke: 0.9pt + MUTED)
  },
))

#pause
#v(10pt)
#align(center, text(size: 17pt)[
  To isolate depth, set the activation to the identity and every bias to zero.
])

#pause
#v(8pt)
#align(center, text(size: 20pt)[
  $bold(h)_1=W_1 bold(x), quad
   bold(h)_2=W_2 W_1 bold(x), quad
   hat(bold(y))=W_L dots W_2 W_1 bold(x)$
])

#pause
#v(14pt)
#align(center, result[The simplified network keeps only the effect we want to understand: repeated multiplication.])

== How one scalar activation changes through four layers #D

Here $h_ell$ is the scalar activation after layer $ell$. Set the input activation to $h_0=1$.

Every layer multiplies its incoming activation by the same local gain $q$:

$ h_ell=q h_(ell-1) quad arrow.r quad h_4=q^4 h_0. $

#pause
#v(11pt)
#align(center, grid(
  columns: (25mm, 10mm, 25mm, 10mm, 25mm, 10mm, 25mm, 10mm, 25mm),
  gutter: 3pt, align: horizon,
  scale-card([$h_0$], [$1$], color: INK, width: 25mm),
  align(center, text(size: 19pt, fill: TEAL)[$times q$]),
  scale-card([$h_1$], [$q$], color: TEAL, width: 25mm),
  align(center, text(size: 19pt, fill: TEAL)[$times q$]),
  scale-card([$h_2$], [$q^2$], color: TEAL, width: 25mm),
  align(center, text(size: 19pt, fill: TEAL)[$times q$]),
  scale-card([$h_3$], [$q^3$], color: TEAL, width: 25mm),
  align(center, text(size: 19pt, fill: TEAL)[$times q$]),
  scale-card([$h_4$], [$q^4$], color: ACC, width: 25mm),
))

#pause
#v(13pt)
#align(center, table(
  columns: (44mm, 44mm, 44mm, 44mm), stroke: 0.5pt + MUTED,
  inset: (x: 12pt, y: 7pt), align: right,
  table.header([$q$], [$h_1$], [$h_2$], [$h_4$]),
  [$0.5000$], [$0.5000$], [$0.2500$], [#text(fill: BLUE, weight: 650)[$0.0625$]],
  [$1.0000$], [$1.0000$], [$1.0000$], [#text(fill: GREEN, weight: 650)[$1.0000$]],
  [$1.5000$], [$1.5000$], [$2.2500$], [#text(fill: RED, weight: 650)[$5.0625$]],
))

== Backpropagation through the same four multiplications #D

Write $g_ell=partial cal(L)/partial h_ell$. Suppose the gradient arriving at the output is $g_4=1$.

#pause
$ g_3=q g_4, quad g_2=q g_3, quad g_1=q g_2, quad g_0=q g_1=q^4. $

#pause
#v(12pt)
#align(center, table(
  columns: (44mm, 52mm, 52mm, 52mm), stroke: 0.5pt + MUTED,
  inset: (x: 12pt, y: 8pt), align: center,
  table.header([local multiplier $q$], [$g_4$], [$g_2=q^2$], [$g_0=q^4$]),
  [$0.5$], [$1$], [$0.25$], [#text(fill: BLUE, weight: 650)[$0.0625$]],
  [$1.0$], [$1$], [$1.0$], [#text(fill: GREEN, weight: 650)[$1.0$]],
  [$1.5$], [$1$], [$2.25$], [#text(fill: RED, weight: 650)[$5.0625$]],
))

#pause
#v(13pt)
#align(center, result[
  The same product that controls the forward signal also controls how much sensitivity reaches an early layer.
])

== Matrix backpropagation multiplies transposed Jacobians #D

#chain-diagram

#pause
#v(3pt)
#align(center, text(size: 12pt, fill: MUTED)[The schematic suppresses boldface; the equations below treat activations and gradients as vectors.])
#v(5pt)
#align(center, grid(columns: (1fr, 1fr), gutter: 30pt,
  [#small-label([forward], color: TEAL)
   #v(4pt)
   $bold(h)_ell=f_ell(bold(h)_(ell-1))$ \
   $bold(h)_L=f_L circle dots circle f_1(bold(h)_0)$],
  [#small-label([backward], color: BLUE)
   #v(4pt)
   $bold(g)_(ell-1)=J_ell^top bold(g)_ell$ \
   $bold(g)_0=J_1^top dots J_L^top bold(g)_L$],
))

#pause
#v(11pt)
#align(center, text(size: 16.5pt, fill: MUTED)[
  $J_ell=partial bold(h)_ell/partial bold(h)_(ell-1)$. The scalar multiplier $q$ was the one-dimensional version of this local Jacobian.
])

== RMS measures the typical size of a gradient vector #D

A layer has many gradient coordinates, with positive and negative signs. Their ordinary mean can cancel.

#v(10pt)
#align(center, grid(columns: (1fr, 1fr), gutter: 34pt,
  [#small-label([one gradient vector], color: BLUE)
   #v(6pt)
   #text(size: 24pt, fill: BLUE)[$bold(g)=(2,-2,1,-1)$]
   #v(8pt)
   #text(size: 17pt)[Its mean is $0$, but its coordinates are not small.]],
  [#small-label([root mean square], color: TEAL)
   #v(6pt)
   #text(size: 21pt)[$"RMS"(bold(g)) = sqrt((2^2+(-2)^2+1^2+(-1)^2)/4)$]
   #v(5pt)
   #text(size: 24pt, weight: 650, fill: TEAL)[$=sqrt(2.5) approx 1.58$]],
))

#pause
#v(18pt)
#align(center, result[
  RMS ignores sign but preserves magnitude: it is the size of a typical coordinate in the vector.
])

== One-layer RMS gain is a ratio of gradient sizes #D

#align(center, text(size: 18pt)[Backpropagation maps $bold(g)_ell$ to $bold(g)_(ell-1)=J_ell^top bold(g)_ell$. Compare their RMS values.])

#align(center, text(size: 14.5pt, fill: MUTED)[Suppose a layerwise diagnostic reports the following hypothetical measurements on one batch.])

#v(14pt)
#align(center, grid(
  columns: (1fr, 16mm, 1fr), gutter: 10pt, align: horizon,
  [#align(center)[
    #small-label([arriving from the later layer], color: INK)
    #v(7pt)
    #text(size: 25pt)[$"RMS"(bold(g)_ell)=1.00$]
  ]],
  [#align(center)[
    #text(size: 25pt, fill: TEAL)[$arrow.l$]
    #v(3pt)
    #text(size: 12pt, fill: MUTED)[$J_ell^top$]
  ]],
  [#align(center)[
    #small-label([reaching the earlier layer], color: BLUE)
    #v(7pt)
    #text(size: 25pt, fill: BLUE)[$"RMS"(bold(g)_(ell-1))=0.90$]
  ]],
))

#pause
#v(17pt)
#align(center, text(size: 22pt)[
  $rho_ell := "RMS"(bold(g)_(ell-1)) / "RMS"(bold(g)_ell) = 0.90 / 1.00 = 0.90$
])

#align(center, text(size: 17pt, fill: MUTED)[This layer reduced the typical gradient coordinate by $10%$.])

== Input-gradient RMS is multiplied by $product_(ell=1)^30 rho_ell$ #D

By the definition of each ratio, the product telescopes exactly:

$ "RMS"(bold(g)_0)=product_(ell=1)^30 rho_ell dot "RMS"(bold(g)_30). $

#pause
#v(7pt)
#align(center, text(size: 17pt)[If all measured gains are close to one value $rho$, the multiplier is approximately $rho^30$.])

#v(9pt)
#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 24pt,
  [#block(width: 100%, inset: 14pt, radius: 4pt, stroke: 1pt + BLUE)[
    #align(center)[
      #small-label([$rho=0.9$], color: BLUE)
      #v(8pt)
      #text(size: 30pt, weight: 650, fill: BLUE)[$0.9^30 approx 0.042$]
      #v(7pt)
      only $4.2%$ reaches the start
    ]
  ]],
  [#block(width: 100%, inset: 14pt, radius: 4pt, stroke: 1pt + GREEN)[
    #align(center)[
      #small-label([$rho=1.0$], color: GREEN)
      #v(8pt)
      #text(size: 30pt, weight: 650, fill: GREEN)[$1^30=1$]
      #v(7pt)
      scale is preserved
    ]
  ]],
  [#block(width: 100%, inset: 14pt, radius: 4pt, stroke: 1pt + RED)[
    #align(center)[
      #small-label([$rho=1.1$], color: RED)
      #v(8pt)
      #text(size: 30pt, weight: 650, fill: RED)[$1.1^30 approx 17.45$]
      #v(7pt)
      over $17 times$ reaches the start
    ]
  ]],
))

== Vanishing gradients starve early layers; exploding signals destabilize numerics #V

#small-label([plain SGD update for the first layer], color: MUTED)
#v(4pt)

$ frac(partial cal(L), partial W_1)=bold(g)_1 bold(h)_0^top, quad
   W_1 arrow.l W_1-eta frac(partial cal(L), partial W_1). $

#align(center, text(size: 12.5pt, fill: MUTED)[
  Adam rescales coordinate updates, but it cannot recover zero or underflowed gradients or repair badly scaled forward activations.
])

#pause
#v(4pt)
#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 22pt,
  [#block(width: 100%, inset: 9pt, radius: 4pt, stroke: 1pt + BLUE)[
    #small-label([vanishing], color: BLUE)
    #v(4pt)
    $"RMS"(bold(g)_1) approx 0$ \
    #v(3pt)
    #text(size: 15pt)[The early weights barely change; features near the input learn very slowly.]
  ]],
  [#block(width: 100%, inset: 9pt, radius: 4pt, stroke: 1pt + GREEN)[
    #small-label([usable scale], color: GREEN)
    #v(4pt)
    $"RMS"(bold(g)_1)$ is neither tiny nor huge and is comparable across depth \
    #v(3pt)
    #text(size: 15pt)[The optimizer receives a meaningful direction and scale.]
  ]],
  [#block(width: 100%, inset: 9pt, radius: 4pt, stroke: 1pt + RED)[
    #small-label([exploding], color: RED)
    #v(4pt)
    $"RMS"(bold(g)_1)$ grows rapidly toward the input \
    #v(3pt)
    #text(size: 15pt)[SGD steps can become huge; forward or backward values may overflow or be clipped.]
  ]],
))

= Initialization: break symmetry and control signal scale

== Random initialization has two jobs #V

#align(center, grid(columns: (1fr, 1fr), gutter: 48pt,
  [#small-label([job 1 · break symmetry], color: ACC)
   #v(8pt)
   #text(size: 24pt, weight: 650)[Hidden units must begin differently.]
   #v(8pt)
   #text(size: 18pt)[Independent random values give them different features and different gradients.]],
  [#small-label([job 2 · control scale], color: TEAL)
   #v(8pt)
   #text(size: 24pt, weight: 650)[Signals must survive depth.]
   #v(8pt)
   #text(size: 18pt)[The weight variance should keep forward activations and backward gradients usable.]],
))

#pause
#v(18pt)
#align(center, result[First break symmetry; then calculate the variance.])

== Exact clones receive identical updates and remain clones #D

#align(center, text(size: 11.5pt, fill: MUTED)[
  Exact means equal incoming and outgoing parameters, optimizer state, and deterministic operations.
])

#scale(x: 84%, y: 84%, reflow: true, clone-diagram)

#v(2pt)
Take $h_i="ReLU"(w_i x)$, $hat(y)=v_1h_1+v_2h_2$, and
$cal(L)=frac(1,2)(hat(y)-y)^2$ with

$w_1=w_2=0.5, quad v_1=v_2=0.25, quad x=2, quad y=1.$

#pause
#v(4pt)
#align(center, grid(columns: (1fr, 1fr), gutter: 32pt,
  [#small-label([forward], color: TEAL)
   #v(4pt)
   $h_1=h_2=1$ \
   $hat(y)=0.5$],
  [#small-label([backward], color: BLUE)
   #v(4pt)
   $partial cal(L)/partial v_1=partial cal(L)/partial v_2=-0.5$ \
   $partial cal(L)/partial w_1=partial cal(L)/partial w_2=-0.25$],
))

== Four cloned neurons still provide only one feature #V

#align(center, grid(columns: (0.9fr, 1.1fr), gutter: 38pt,
  [#small-label([inside the hidden layer], color: ACC)
   #v(7pt)
   #align(center, diagram(
     spacing: (17mm, 10mm), node-stroke: 1pt + INK, node-fill: white,
     {
       let hidden-labels = ($h_1(bold(x))$, $h_2(bold(x))$,
         $h_3(bold(x))$, $h_4(bold(x))$)
       node((0, 0), $bold(x)$, radius: 6.5mm, stroke: 1.1pt + TEAL)
       for (i, y) in (-1.35, -0.45, 0.45, 1.35).enumerate() {
         node((1.7, y), text(size: 10pt, hidden-labels.at(i)),
           radius: 7mm, stroke: 1.1pt + ACC)
         edge((0, 0), (1.7, y), "-|>", stroke: 0.9pt + TEAL)
       }
       node((3.45, 0), $s(bold(x))$, radius: 7.5mm, stroke: 1.1pt + INK)
       for y in (-1.35, -0.45, 0.45, 1.35) {
         edge((1.7, y), (3.45, 0), "-|>", stroke: 0.9pt + ACC)
       }
     },
   ))],
  [#small-label([if all four copies stay equal], color: TEAL)
   #v(7pt)
   #text(size: 15pt)[$h_j(bold(x))=h(bold(x))$ and $v_j=v$ for every $j in {1,dots,4}$.]
   #v(6pt)
   $ s(bold(x)) &= sum_(j=1)^4 v h_j(bold(x))+c \
                &= 4v h(bold(x))+c $
   #v(7pt)
   #text(size: 18pt)[The output can rescale one feature, but it cannot create three new feature directions.]],
))

#pause
#v(7pt)
#align(center, grid(columns: (1fr, 1fr), gutter: 30pt,
  [#block(width: 100%, inset: 7pt, radius: 4pt,
    stroke: 1pt + MUTED, align(center)[
      #small-label([layer width], color: MUTED)
      #v(2pt)
      #text(size: 23pt, weight: 650)[4 neurons]
    ])],
  [#block(width: 100%, inset: 7pt, radius: 4pt,
    stroke: 1.2pt + RED, fill: RED.lighten(95%), align(center)[
      #small-label([distinct hidden features], color: RED)
      #v(2pt)
      #text(size: 23pt, weight: 650, fill: RED)[1 feature]
  ])],
))

== XOR: cloned units reach 75%; independent units reach 100% #V

#align(center, image(
  "/lecture6/figures/symmetry_decision_boundaries.svg",
  width: 194mm,
))

#align(center, text(size: 11.5pt, fill: MUTED)[
  Same $320$-point XOR data, $2 arrow.r 4$ ReLU $arrow.r 1$ model, full-batch Adam, and $300$ updates.
  $"rank"(H)$ is the rank of the $320 times 4$ hidden-activation matrix.
])

== Interactive: when do cloned neurons separate? #I

#interbox(link-to: SYMMETRY_DEMO)[
  Open the *hidden-unit symmetry lab*. Keep the data and architecture fixed; change only how the four hidden units start.
]

#v(8pt)
#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 10pt,
  [#block(width: 100%, inset: 9pt, radius: 4pt, stroke: 1pt + ACC)[
    #small-label([1 · exactly cloned], color: ACC)
    #v(4pt)
    Click *one step*. The four hinge lines remain on top of one another and parameter separation stays zero.
  ]],
  [#block(width: 100%, inset: 9pt, radius: 4pt, stroke: 1pt + TEAL)[
    #small-label([2 · small perturbation], color: TEAL)
    #v(4pt)
    Nudge the initial copies apart. Their gradients differ and the hinge lines can begin to specialize.
  ]],
  [#block(width: 100%, inset: 9pt, radius: 4pt, stroke: 1pt + BLUE)[
    #small-label([3 · independent], color: BLUE)
    #v(4pt)
    Run training. Compare accuracy, loss, hidden-feature rank, and the learned decision surface.
  ]],
))

#pause
#place(bottom + center, dy: -2pt, result[
  #text(size: 14.5pt)[Exact cloning preserves the symmetry. A distinct start lets the units learn distinct features.]
])

== ReLU passes positive preactivations and blocks negative ones #V

#align(center, text(size: 21pt, weight: 650)[
  $z=w^top x+b, quad h="ReLU"(z)=max(0,z)$
])

#v(10pt)
#align(center, grid(columns: (1fr, 1fr), gutter: 28pt,
  [#block(width: 100%, inset: 12pt, radius: 4pt,
    stroke: 1.2pt + GREEN, fill: GREEN.lighten(96%))[
    #small-label([positive preactivation · gate open], color: GREEN)
    #v(8pt)
    #relu-case-diagram([$z=+2$], [$h=2$], [ReLU], GREEN)
    #v(9pt)
    #align(center, text(size: 18pt)[
      $"ReLU"'(+2)=1 quad arrow.r quad frac(partial cal(L),partial z)=frac(partial cal(L),partial h)$
    ])
  ]],
  [#block(width: 100%, inset: 12pt, radius: 4pt,
    stroke: 1.2pt + RED, fill: RED.lighten(96%))[
    #small-label([negative preactivation · gate closed], color: RED)
    #v(8pt)
    #relu-case-diagram([$z=-2$], [$h=0$], [ReLU], RED)
    #v(9pt)
    #align(center, text(size: 18pt)[
      $"ReLU"'(-2)=0 quad arrow.r quad frac(partial cal(L),partial z)=0$
    ])
  ]],
))

#pause
#place(bottom + center, dy: -2pt, result[
  A negative preactivation is blocked in both the forward pass and the backward pass.
])

== The gate decides whether this example updates $w$ and $b$ #D

Suppose the gradient arriving at the ReLU output is
$partial cal(L)/partial h=4$, with $z=w x+b$ and $x=2$.

#v(9pt)
#align(center, grid(columns: (1fr, 1fr), gutter: 28pt,
  [#block(width: 100%, inset: 13pt, radius: 4pt, stroke: 1.2pt + RED)[
    #small-label([closed gate], color: RED)
    #v(7pt)
    $w=-1, quad b=-1$ \
    $z=(-1)(2)-1=-3$ \
    $h="ReLU"(-3)=0$
    #v(8pt)
    $frac(partial cal(L),partial z)=4 times 0=0$ \
    $frac(partial cal(L),partial w)=frac(partial cal(L),partial z)x=0$ \
    $frac(partial cal(L),partial b)=0$
  ]],
  [#block(width: 100%, inset: 13pt, radius: 4pt, stroke: 1.2pt + GREEN)[
    #small-label([open gate], color: GREEN)
    #v(7pt)
    $w=1, quad b=-1$ \
    $z=(1)(2)-1=1$ \
    $h="ReLU"(1)=1$
    #v(8pt)
    $frac(partial cal(L),partial z)=4 times 1=4$ \
    $frac(partial cal(L),partial w)=frac(partial cal(L),partial z)x=8$ \
    $frac(partial cal(L),partial b)=4$
  ]],
))

#pause
#place(bottom + center, dy: -2pt, result[
  The same example updates the incoming parameters only when its ReLU gate is open.
])

== A neuron is dead on the training set when no example opens it #V

Take three inputs $x in {-1,0,1}$ and $z=w x+b$ with $w=1$.
Suppose an update moves the bias from $-0.2$ to $-2$.

#v(10pt)
#align(center, grid(columns: (28mm, 32mm, 32mm, 32mm, 55mm), gutter: 5pt,
  gate-cell([*case*], fill: INK.lighten(86%)),
  gate-cell([$x=-1$], fill: INK.lighten(86%)),
  gate-cell([$x=0$], fill: INK.lighten(86%)),
  gate-cell([$x=1$], fill: INK.lighten(86%)),
  gate-cell([*result over all three examples*], fill: INK.lighten(86%), size: 12pt),

  gate-cell([before #linebreak() $b=-0.2$], fill: GREEN.lighten(94%), stroke: 0.8pt + GREEN, size: 12.5pt),
  gate-cell([$z=-1.2$ #linebreak() $h=0$], fill: GREEN.lighten(97%)),
  gate-cell([$z=-0.2$ #linebreak() $h=0$], fill: GREEN.lighten(97%)),
  gate-cell([$z=0.8$ #linebreak() $h=0.8$], fill: GREEN.lighten(97%)),
  gate-cell([*alive* #linebreak() one gate opens], fill: GREEN.lighten(94%), stroke: 0.8pt + GREEN, size: 12.5pt),

  gate-cell([after #linebreak() $b=-2$], fill: RED.lighten(94%), stroke: 0.8pt + RED, size: 12.5pt),
  gate-cell([$z=-3$ #linebreak() $h=0$], fill: RED.lighten(97%)),
  gate-cell([$z=-2$ #linebreak() $h=0$], fill: RED.lighten(97%)),
  gate-cell([$z=-1$ #linebreak() $h=0$], fill: RED.lighten(97%)),
  gate-cell([*dead* #linebreak() all three gates closed], fill: RED.lighten(94%), stroke: 0.8pt + RED, size: 12.5pt),
))

#pause
#place(bottom + center, dy: -2pt, result[
  #text(size: 14.5pt)[Some negative preactivations are normal. If every preactivation is negative, this neuron's data gradient is zero.]
])

== All-zero initialization puts every hidden ReLU at $z=0$ #V

In the same two-unit network, set every hidden bias and every weight to zero:

$w_1=w_2=b_1=b_2=v_1=v_2=0, quad x=2, quad y=1.$

#v(4pt)
#scale(x: 82%, y: 82%, reflow: true, zero-init-diagram)

#pause
#v(8pt)
#align(center, text(size: 17pt)[
  $z_1=z_2=0$ and $h_1=h_2="ReLU"(0)=0$. At this kink the derivative is undefined mathematically; PyTorch uses $"ReLU"'(0)=0$.
])

#pause
#place(bottom + center, dy: -2pt, result[
  Predict before calculating: can either layer receive a nonzero data gradient?
])

== With every weight zero, both weight gradients are zero #D

For $cal(L)=frac(1,2)(hat(y)-y)^2$, the output derivative is
$partial cal(L)/partial hat(y)=hat(y)-y=-1$.

#v(10pt)
#align(center, grid(columns: (1fr, 1fr), gutter: 30pt,
  [#block(width: 100%, inset: 13pt, radius: 4pt, stroke: 1.2pt + ACC)[
    #small-label([output weights], color: ACC)
    #v(7pt)
    $frac(partial cal(L),partial v_i)
      =(hat(y)-y)h_i$
    #v(5pt)
    $=(-1)(0)=0$
    #v(8pt)
    The output weights cannot move because every hidden activation is zero.
  ]],
  [#block(width: 100%, inset: 13pt, radius: 4pt, stroke: 1.2pt + RED)[
    #small-label([hidden weights], color: RED)
    #v(7pt)
    $frac(partial cal(L),partial w_i)
      =(hat(y)-y)v_i "ReLU"'(0)x$
    #v(5pt)
    $=(-1)(0)(0)(2)=0$
    #v(8pt)
    The hidden weights cannot move because the outgoing path is also zero.
  ]],
))

#pause
#v(5pt)
#align(center, text(size: 13.5pt, fill: MUTED)[
  An output bias, if present, can learn a constant; the hidden features still remain zero.
])

#pause
#place(bottom + center, dy: -2pt, result[
  $w_i^+=v_i^+=0$: the same state repeats. All-zero initialization is stuck; hidden units must not begin as exact clones. Zero biases are fine.
])

= Choosing the weight scale

== Different nonzero weights can still have the wrong scale #V

#align(center, grid(columns: (1fr, 1fr), gutter: 42pt,
  [#block(width: 100%, inset: 14pt, radius: 4pt, stroke: 1.2pt + ACC)[
    #small-label([already fixed · symmetry], color: ACC)
    #v(8pt)
    #text(size: 23pt, weight: 650)[Start neurons differently.]
    #v(8pt)
    #text(size: 17pt)[Independent nonzero weights let hidden units learn different features.]
  ]],
  [#block(width: 100%, inset: 14pt, radius: 4pt, stroke: 1.2pt + TEAL)[
    #small-label([remaining question · scale], color: TEAL)
    #v(8pt)
    #text(size: 23pt, weight: 650)[How large should the weights be?]
    #v(8pt)
    #text(size: 17pt)[Their standard deviation sets how much one layer expands or shrinks a signal.]
  ]],
))

#pause
#v(8pt)
#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 16pt, align: top,
  [#block(width: 100%, inset: 9pt, radius: 4pt, stroke: 1pt + TEAL, align(center)[
    #small-label([weight scale], color: TEAL)
    #v(4pt)
    #text(size: 22pt)[$"Std"(w)$]
  ])],
  [#block(width: 100%, inset: 9pt, radius: 4pt, stroke: 1pt + ACC, align(center)[
    #small-label([affine-map gain], color: ACC)
    #v(4pt)
    #text(size: 17pt)[$s_ell:="RMS"(z_ell)/"RMS"(h_(ell-1))$]
  ])],
  [#block(width: 100%, inset: 9pt, radius: 4pt, stroke: 1pt + RED, align(center)[
    #small-label([full-layer gain], color: RED)
    #v(4pt)
    #text(size: 17pt)[$kappa_ell:="RMS"(h_ell)/"RMS"(h_(ell-1))$]
  ])],
))

#pause
#v(7pt)
#align(center, result[
  $"RMS"(h_L)/"RMS"(h_0)=product_(ell=1)^L kappa_ell$.
  At initialization, the target is $kappa_ell approx 1$.
])

== One neuron sums one weighted contribution from each input #V

#two(
  [#simple-fan-in-diagram],
  [#small-label([one linear preactivation], color: TEAL)
   #v(8pt)
   $u_i=w_i h_i$ is one weighted contribution. \
   #v(7pt)
   $z=u_1+u_2+dots+u_n=sum_(i=1)^n w_i h_i$ \
   #v(9pt)
   *fan-in $=n$*: the number of values entering the neuron.],
  r: (0.9fr, 1.1fr),
)

#pause
#v(8pt)
#align(center, notebox[
  *Why calculate this?* $z$ is passed into the activation; an RMS gain here compounds through depth. \
  #v(3pt)
  *Prediction:* for $100$ independent, zero-mean terms with RMS $1$, is $"RMS"(z)$ closer to $1$, $10$, or $100$?
])

== Two independent $plus.minus 1$ terms give a sum with RMS $sqrt(2)$ #D

#text(size: 15.5pt)[Each term is independently $-1$ or $+1$ with probability $1/2$. Thus $EE[u_i]=0$ and $"RMS"(u_i)=1$.]

#v(5pt)
#align(center, grid(columns: (1.05fr, 0.95fr), gutter: 28pt, align: horizon,
  [#table(
    columns: (22mm, 22mm, 28mm, 22mm), stroke: 0.5pt + MUTED,
    inset: (x: 8pt, y: 4pt), align: center,
    table.header([$u_1$], [$u_2$], [$z=u_1+u_2$], [$z^2$]),
    [$-1$], [$-1$], [$-2$], [$4$],
    [$-1$], [$+1$], [$0$], [$0$],
    [$+1$], [$-1$], [$0$], [$0$],
    [$+1$], [$+1$], [$2$], [$4$],
  )],
  [#align(center)[
    #small-label([average squared sum], color: TEAL)
    #v(5pt)
    $EE[z^2]=frac(4+0+0+4,4)=2$
    #v(7pt)
    #text(size: 22pt, weight: 650, fill: TEAL)[$"RMS"(z)=sqrt(2)$]
    #v(11pt)
    $z^2=u_1^2+u_2^2+2u_1u_2$
    #v(4pt)
    $u_1u_2=[+1,-1,-1,+1]$
    #v(2pt)
    $EE[u_1u_2]=0$
  ]],
))

#pause
#v(7pt)
#align(center, text(size: 16pt, weight: 650)[
  The cross term averages to zero. Squared sizes add first; take the square root afterward.
])

== With 100 independent zero-mean, unit-RMS terms, $"RMS"(z)=10$ #D

#align(center, scale(x: 65%, y: 65%, reflow: true, contribution-sum-diagram(
  [$"RMS"(u_i)=1$], [$"RMS"(z)=10$], color: TEAL,
)))

#v(2pt)
#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 14pt,
  [#block(width: 100%, inset: 7pt, radius: 3pt, stroke: 0.8pt + MUTED, align(center)[
    #small-label([$1$ term], color: MUTED) #h(9pt) $"RMS"(z)=1$
  ])],
  [#block(width: 100%, inset: 7pt, radius: 3pt, stroke: 0.8pt + MUTED, align(center)[
    #small-label([$4$ terms], color: MUTED) #h(9pt) $"RMS"(z)=2$
  ])],
  [#block(width: 100%, inset: 7pt, radius: 3pt, stroke: 1.1pt + RED, align(center)[
    #small-label([$100$ terms], color: RED) #h(9pt) #text(fill: RED, weight: 650)[$"RMS"(z)=10$]
  ])],
))

#pause
#v(7pt)
#align(center, text(size: 20pt, weight: 650)[
  By the same calculation, $EE[z^2]=100$ and $"RMS"(z)=sqrt(100)=10$.
])

#align(center, text(size: 14.5pt, fill: MUTED)[
  This RMS is over repeated random draws; one realized 100-term sum need not equal $10$.
])

#pause
#v(6pt)
#align(center, text(size: 16pt, weight: 650)[
  Without fan-in scaling, width changes the one-layer gain, and depth multiplies those gains.
])

== With fan-in 100, $"Std"(w)=0.1$ preserves linear RMS #D

Assume the incoming coordinates have RMS $1$. Compare two random initializations of the same neuron.

#v(11pt)
#align(center, grid(columns: (1fr, 1fr), gutter: 28pt,
  [#block(width: 100%, inset: 14pt, radius: 4pt, stroke: 1.2pt + RED)[
    #small-label([unscaled weights], color: RED)
    #v(8pt)
    $"Std"(w)=1$ \
    $"RMS"(w_i h_i)=1$ \
    #v(8pt)
    #text(size: 23pt, weight: 650, fill: RED)[
      $"RMS"(z)=sqrt(100) times 1=10$
    ]
  ]],
  [#block(width: 100%, inset: 14pt, radius: 4pt, stroke: 1.2pt + GREEN)[
    #small-label([fan-in scaled weights], color: GREEN)
    #v(8pt)
    $"Std"(w)=0.1$ \
    $"RMS"(w_i h_i)=0.1$ \
    #v(8pt)
    #text(size: 23pt, weight: 650, fill: GREEN)[
      $"RMS"(z)=sqrt(100) times 0.1=1$
    ]
  ]],
))

#pause
#v(13pt)
#align(center, result[
  For $100$ inputs, $"Std"(w)=1/sqrt(100)=0.1$ and $"Var"(w)=1/100=0.01$.
])

#align(center, text(size: 14pt, fill: MUTED)[
  This corrects the linear sum. We will account for the following ReLU separately.
])

== For fan-in $n$, variance $1/n$ preserves the linear second moment #D

#text(size: 16pt, fill: MUTED)[Assume $b=0$; the weights are independent, zero-mean, independent of the incoming activations, and have common variance $sigma_w^2$. The incoming coordinates share the second moment $EE[h_i^2]=EE[h^2]$.]

#v(8pt)
$
  EE[z^2]
  =sum_(i=1)^n EE[w_i^2 h_i^2]
  =n sigma_w^2 EE[h^2].
$

#align(center, text(size: 15pt, fill: MUTED)[The cross terms vanish in expectation because the weights are independent and zero-mean.])

#pause
#v(9pt)
$
  frac("RMS"(z), "RMS"(h))
  =sqrt(n "Var"(w)).
$

#pause
#v(10pt)
#align(center, grid(columns: (1fr, 1fr), gutter: 34pt,
  [#block(width: 100%, inset: 11pt, radius: 4pt, stroke: 1.2pt + TEAL, align(center)[
    #small-label([variance], color: TEAL)
    #v(5pt)
    #text(size: 28pt, weight: 650)[$"Var"(w)=1/n$]
  ])],
  [#block(width: 100%, inset: 11pt, radius: 4pt, stroke: 1.2pt + BLUE, align(center)[
    #small-label([standard deviation], color: BLUE)
    #v(5pt)
    #text(size: 28pt, weight: 650)[$"Std"(w)=1/sqrt(n)$]
  ])],
))


== An affine map backpropagates a fan-out weighted sum #D

#two(
  [#simple-fan-out-diagram],
  [#small-label([one incoming activation], color: BLUE)
   #v(7pt)
   It influences $n_"out"$ preactivations. \
   *fan-out $=n_"out"$*: the number of values this activation feeds. \
   #v(10pt)
   $partial cal(L)/partial h_i=sum_(j=1)^(n_"out") w_(j i) partial cal(L)/partial z_j$],
  r: (0.9fr, 1.1fr),
)

#pause
#v(10pt)
#align(center, grid(columns: (1fr, 1fr), gutter: 32pt,
  [#small-label([forward], color: TEAL)
   #v(5pt)
   #text(size: 16pt)[At initialization, independent zero-mean contributions add; $"Var"(w) approx 1/n_"in"$ preserves the affine second moment.]],
  [#small-label([backward], color: BLUE)
   #v(5pt)
   #text(size: 16pt)[Similarly, $"Var"(w) approx 1/n_"out"$ preserves the affine backward second moment. Activation derivatives are separate.]],
))

== For identity and near-linear tanh, Xavier balances fan-in and fan-out #D

#align(center, grid(columns: (1fr, 18mm, 1fr), gutter: 12pt, align: horizon,
  [#block(width: 100%, inset: 14pt, radius: 4pt, stroke: 1.2pt + TEAL, align(center)[
    #small-label([forward prefers], color: TEAL)
    #v(7pt)
    #text(size: 28pt, weight: 650)[$1/n_"in"$]
  ])],
  [#text(size: 28pt, fill: MUTED)[vs]],
  [#block(width: 100%, inset: 14pt, radius: 4pt, stroke: 1.2pt + BLUE, align(center)[
    #small-label([backward prefers], color: BLUE)
    #v(7pt)
    #text(size: 28pt, weight: 650)[$1/n_"out"$]
  ])],
))

#v(14pt)
#align(center, text(size: 19pt)[When $n_"in" != n_"out"$, one variance cannot satisfy both exactly.])

#pause
#v(12pt)
#align(center, block(
  width: 78%, inset: 16pt, radius: 4pt,
  stroke: 1.4pt + ACC, fill: ACC.lighten(95%), align(center)[
    #small-label([Xavier / Glorot compromise], color: ACC)
    #v(7pt)
    #text(size: 31pt, weight: 650)[$"Var"(W)=2/(n_"in"+n_"out")$]
  ],
))

#pause
#v(13pt)
#align(center, result[
  For a square layer, $n_"in"=n_"out"=n$, so Xavier reduces to $"Var"(W)=1/n$.
])

== ReLU keeps half the second moment of a symmetric input #D

#align(center, grid(columns: (1fr, auto, 1fr), gutter: 26pt, align: horizon,
  [#small-label([before ReLU], color: TEAL)
   #v(7pt)
   #text(size: 25pt)[$z=[-2,-1,1,2]$] \
   $"mean"(z^2)=10/4=2.5$],
  [#text(size: 34pt, fill: ACC)[$arrow.r$]],
  [#small-label([after ReLU], color: ACC)
   #v(7pt)
   #text(size: 25pt)[$h=[0,0,1,2]$] \
   $"mean"(h^2)=5/4=1.25$],
))

#pause
#v(18pt)
#align(center, text(size: 22pt, weight: 650)[
  For a distribution symmetric around zero,
  $EE["ReLU"(z)^2]=frac(1,2)EE[z^2]$.
])

#pause
#v(14pt)
#align(center, result[The affine map must double the second moment before ReLU halves it.])

== For ReLU, He/Kaiming uses $"Std"(w)=sqrt(2/"fan-in")$ #D

#text(size: 14.5pt, fill: MUTED)[At initialization, assume independent zero-mean weights, zero bias, and an approximately symmetric zero-centred preactivation.]

#v(4pt)
$
  EE[h_"next"^2]
  approx frac(1,2)n_"in" "Var"(w) EE[h^2].
$

#pause
Set the per-layer second-moment gain to one:

$
  frac(1,2)n_"in" "Var"(w)=1
  quad arrow.r quad
  "Var"(w)=frac(2,n_"in"),
  quad "Std"(w)=sqrt(frac(2,"fan-in")).
$

#pause
#v(13pt)
#align(center, grid(columns: (1fr, 1fr), gutter: 28pt,
  [#small-label([worked fan-in 100], color: TEAL)
   #v(5pt)
   $"Std"_"He"=sqrt(2/100) approx 0.141$],
  [#small-label([our hidden width 128], color: BLUE)
   #v(5pt)
   $"Std"_"He"=sqrt(2/128)=0.125$],
))

#pause
#v(10pt)
#align(center, text(size: 14.5pt, fill: MUTED)[
  PyTorch: `nn.init.kaiming_normal_(W, mode="fan_in", nonlinearity="relu")`
])
#align(center, text(size: 14.5pt, fill: MUTED)[This stabilizes scale at initialization; it does not guarantee that every deep network will train.])

= Test the calculation on a 30-hidden-layer MLP

== A 30-hidden-layer MLP on a 2-D spiral #V

#two(
  [#image("figures/spiral_dataset.svg", width: 88%)],
  [#codebox(size: 13pt)[
```python
model = nn.Sequential(
    nn.Linear(2, 128), nn.ReLU(),
    # ... 29 more hidden Linear-ReLU blocks ...
    nn.Linear(128, 3),
)
opt = torch.optim.Adam(
    model.parameters(), lr=3e-4
)
```
   ]
   #v(8pt)
   *900 points · 3 classes* \
   *30 hidden layers · ReLU* \
   *cross-entropy · full-batch Adam*],
  r: (0.82fr, 1.18fr),
)

== Training loss and accuracy for runs A–C

#image("figures/training_fates_unlabeled.svg", width: 88%)

#align(center, text(size: 17pt, weight: 650)[One run is stuck; two learn at different rates. What single setting changed?])

== What changed between runs A, B, and C?

#mcq(
  [What was the only thing changed between runs A, B, and C?],
  [optimizer], [learning rate], [initial weight scale], [network depth],
)

== Decision regions after 50 Adam updates

#image("figures/decision_boundaries_step50.svg", width: 90%)

#pause
#align(center, text(size: 21pt, weight: 650)[
  Only $alpha$ changes across the three runs. What quantity do you think $alpha$ controls?
])

== Network and tensor notation for the 30-hidden-layer MLP #V

#network-notation-diagram

#v(7pt)
#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 20pt,
  [#small-label([input], color: INK) \
   $bold(h)_0=bold(x) in RR^2$],
  [#small-label([hidden layers $ell=1,dots,30$], color: TEAL) \
   $bold(z)_ell=W_ell bold(h)_(ell-1)+bold(b)_ell$ \
   $bold(h)_ell="ReLU"(bold(z)_ell) in RR^128$],
  [#small-label([output layer 31], color: ACC) \
   $bold(o)=W_31 bold(h)_30+bold(b)_31 in RR^3$ \
   #text(size: 13pt, fill: MUTED)[No ReLU on the logits.]],
))

#pause
#v(8pt)
#align(center, result[
  The subscript $ell$ names a hidden layer. $bold(h)_ell$ is the vector that leaves its ReLU and enters the next layer.
])

== Inside hidden layer $ell$: one neuron and its weights

#one-neuron-diagram

#v(3pt)
#align(center, text(size: 19pt)[
  $z_(ell,j)=sum_(i=1)^(d_(ell-1)) W_(ell,j i) h_(ell-1,i)+b_(ell,j)$
  #h(12pt)
  $h_(ell,j)=max(0,z_(ell,j))$
])

== Read $W_(ell,j i)$ as layer $ell$, row $j$, column $i$ #D

#v(7pt)
#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 22pt,
  [#small-label([$ell$ · layer], color: TEAL) \
   Which hidden layer: $1,dots,30$.],
  [#small-label([$i$ · incoming coordinate], color: BLUE) \
   Which value in $bold(h)_(ell-1)$.],
  [#small-label([$j$ · receiving neuron], color: ACC) \
   Which weighted sum in layer $ell$.],
))

#v(20pt)
#align(center, block(
  inset: 13pt, radius: 4pt, stroke: 1.2pt + TEAL,
  align(center, [
    #small-label([concrete example], color: TEAL)
    #v(7pt)
    #text(size: 23pt)[$h_(4,3) quad arrow.r^(W_(5,7 3)) quad z_(5,7)$]
    #v(7pt)
    #text(size: 16pt)[The entry $W_(5,7 3)$ multiplies input coordinate $3$ when neuron $7$ in layer $5$ forms its weighted sum.]
  ]),
))

== For $W_ell$: rows use fan-in; columns use fan-out

#align(center, grid(
  columns: (1fr, 1fr), gutter: 34pt,
  [#small-label([fan-in of one output neuron], color: TEAL)
   #v(3pt)
   #fan-in-diagram],
  [#small-label([fan-out of one input value], color: BLUE)
   #v(3pt)
   #fan-out-diagram],
))

#v(5pt)
#align(center, text(size: 18pt, weight: 650)[
  $bold(z)_ell=W_ell bold(h)_(ell-1)+bold(b)_ell, quad
   W_ell in RR^(d_ell times d_(ell-1))$
])

== Fan-in and fan-out in this MLP

#v(19pt)
#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 24pt,
  [#block(inset: 13pt, radius: 4pt, stroke: 1pt + ACC, width: 100%)[
     #align(center)[
       #small-label([$W_1$ · input to hidden 1], color: ACC)
       #v(10pt)
       #text(size: 24pt, weight: 650)[$2 arrow.r 128$]
       #v(9pt)
       $W_1 in RR^(128 times 2)$ \
       #v(5pt)
       fan-in $=2$ \
       fan-out $=128$
     ]
   ]],
  [#block(inset: 13pt, radius: 4pt, stroke: 1pt + TEAL, width: 100%)[
     #align(center)[
       #small-label([$W_2,dots,W_30$ · hidden], color: TEAL)
       #v(10pt)
       #text(size: 24pt, weight: 650)[$128 arrow.r 128$]
       #v(9pt)
       $W_ell in RR^(128 times 128)$ \
       #v(5pt)
       fan-in $=128$ \
       fan-out $=128$
     ]
   ]],
  [#block(inset: 13pt, radius: 4pt, stroke: 1pt + BLUE, width: 100%)[
     #align(center)[
       #small-label([$W_31$ · hidden 30 to output], color: BLUE)
       #v(10pt)
       #text(size: 24pt, weight: 650)[$128 arrow.r 3$]
       #v(9pt)
       $W_31 in RR^(3 times 128)$ \
       #v(5pt)
       fan-in $=128$ \
       fan-out $=3$
     ]
   ]],
))

#v(18pt)
#align(center, result[
  In PyTorch, `nn.Linear(d_in, d_out).weight.shape` is `(d_out, d_in)`.
])

== He/Kaiming standard deviations in this MLP #D

#align(center, text(size: 19pt)[
  One preactivation is a sum of $d_(ell-1)$ incoming contributions:
  #h(8pt) $z_(ell,j)=sum_i W_(ell,j i)h_(ell-1,i)$.
])

#v(12pt)
#align(center, grid(columns: (1fr, 9mm, 1fr, 9mm, 1fr), gutter: 7pt, align: horizon,
  [#small-label([1 · the sum], color: TEAL) \
   #text(size: 17pt)[Independent contributions add their variances.] \
   #v(4pt)
   $"Var"(z_(ell,j)) approx d_(ell-1) "Var"(W_(ell,j i)) EE[h_(ell-1,i)^2]$],
  align(center, text(size: 23pt, fill: TEAL)[$arrow.r$]),
  [#small-label([2 · cancel the width], color: BLUE) \
   #text(size: 17pt)[Choose weight variance proportional to $1/d_(ell-1)$.] \
   #v(4pt)
   A wider layer should not create a larger $z_ell$.],
  align(center, text(size: 23pt, fill: ACC)[$arrow.r$]),
  [#small-label([3 · compensate for ReLU], color: ACC) \
   #text(size: 17pt)[At symmetric initialization, ReLU removes roughly half the squared signal.] \
   #v(4pt)
   Multiply the variance by $2$.],
))

#v(15pt)
#align(center, result[
  Hidden-layer He/Kaiming normal: $W_(ell,j i) tilde cal(N)(0, 2/d_(ell-1))$,
  so $"Std"(W_(ell,j i))=sqrt(2/"fan-in")$.
])

== Only $alpha$ changes across runs A, B, and C

#experiment-rig

#v(8pt)
#align(center, codebox(size: 14.5pt)[
```python
with torch.no_grad():
    nn.init.kaiming_normal_(W, mode="fan_in", nonlinearity="relu")
    W.mul_(alpha)   # 0.5, 1.0, or 1.5
```
])

#v(8pt)
#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 20pt,
  [#small-label([$alpha=0.5$], color: BLUE) \
   half the He standard deviation],
  [#small-label([$alpha=1$], color: GREEN) \
   ordinary He initialization],
  [#small-label([$alpha=1.5$], color: RED) \
   $1.5 times$ the He standard deviation],
))

#v(8pt)
#align(center, text(size: 14.5pt, fill: MUTED)[
  The random seed is reset, so every run uses the same weight directions and signs. Only their common scale changes.
])

== Weight standard deviations for $alpha=0.5$, $1.0$, and $1.5$

#align(center, text(size: 19pt)[
  $epsilon_(ell,j i) tilde cal(N)(0,1), quad
   W_(ell,j i)^(("He"))=sqrt(2/d_(ell-1)) epsilon_(ell,j i), quad
   W_(ell,j i)^((alpha))=alpha W_(ell,j i)^(("He"))$
])

#image("figures/weight_distributions.svg", width: 60%)

#align(center, grid(columns: (1fr, 1fr), gutter: 30pt,
  [#text(size: 15pt)[
    For hidden blocks $W_2,dots,W_30$ with fan-in $128$, the He standard deviation is $sqrt(2/128)=0.125$.
    The three curves therefore have standard deviations $0.0625$, $0.125$, and $0.1875$.
  ]],
  [#text(size: 14pt, fill: MUTED)[
    The hidden layers use He initialization for ReLU. For this controlled experiment,
    $W_31$ uses the same rule even though the logits are not followed by ReLU; this is an experimental control, not a general output-layer recommendation.
  ]],
))

== Recorded post-ReLU tensor: $H_ell in RR^(900 times 128)$ #V

#align(center, diagram(
  spacing: (20mm, 10mm), node-stroke: 1pt + INK, node-fill: white,
  {
    node((0, 0), $bold(h)_(ell-1)$, radius: 10mm, stroke: 1.1pt + TEAL)
    node((1.7, 0), [Linear], shape: fletcher.shapes.rect, corner-radius: 3pt,
      inset: 9pt, stroke: 1.1pt + INK)
    node((3.4, 0), $bold(z)_ell$, radius: 8.5mm, stroke: 1.1pt + ACC)
    node((5.0, 0), [ReLU], shape: fletcher.shapes.rect, corner-radius: 3pt,
      inset: 9pt, stroke: 1.1pt + INK)
    node((6.6, 0), $bold(h)_ell$, radius: 8.5mm, stroke: 1.1pt + GREEN)
    edge((0, 0), (1.7, 0), "-|>", stroke: 1.1pt + TEAL)
    edge((1.7, 0), (3.4, 0), "-|>", stroke: 1.1pt + ACC)
    edge((3.4, 0), (5.0, 0), "-|>", stroke: 1.1pt + ACC)
    edge((5.0, 0), (6.6, 0), "-|>", stroke: 1.1pt + GREEN)
  },
))

#v(5pt)
#align(center, grid(columns: (1.2fr, 0.8fr), gutter: 32pt, align: top,
  [#small-label([the batch tensor $H_ell$], color: TEAL)
   #v(4pt)
   #grid(
     columns: (24mm, 18mm, 18mm, 16mm, 20mm), gutter: 0pt,
     batch-cell([], width: 24mm, fill: INK.lighten(91%)),
     batch-cell([unit 1], fill: INK.lighten(91%)),
     batch-cell([unit 2], fill: INK.lighten(91%)),
     batch-cell([$dots$], width: 16mm, fill: INK.lighten(91%)),
     batch-cell([unit 128], width: 20mm, fill: INK.lighten(91%)),
     batch-cell([example 1], width: 24mm, fill: TEAL.lighten(91%)),
     batch-cell([$h_(ell,1)^((1))$]),
     batch-cell([$h_(ell,2)^((1))$]),
     batch-cell([$dots$], width: 16mm),
     batch-cell([$h_(ell,128)^((1))$], width: 20mm),
     batch-cell([example $s$], width: 24mm, fill: TEAL.lighten(82%)),
     batch-cell([$h_(ell,1)^((s))$], fill: TEAL.lighten(91%)),
     batch-cell([$h_(ell,2)^((s))$], fill: TEAL.lighten(91%)),
     batch-cell([$dots$], width: 16mm, fill: TEAL.lighten(91%)),
     batch-cell([$h_(ell,128)^((s))$], width: 20mm, fill: TEAL.lighten(91%)),
     batch-cell([example 900], width: 24mm, fill: TEAL.lighten(91%)),
     batch-cell([$h_(ell,1)^((900))$]),
     batch-cell([$h_(ell,2)^((900))$]),
     batch-cell([$dots$], width: 16mm),
     batch-cell([$h_(ell,128)^((900))$], width: 20mm),
   )
   #v(4pt)
   #text(size: 14.5pt)[$bold(h)_ell^((s))=H_ell[s,:] in RR^128, quad H_ell in RR^(900 times 128)$] \
   #text(size: 12.5pt, fill: MUTED)[Rows are examples; columns are hidden units.]],
  [#small-label([read the notation], color: BLUE)
   #v(5pt)
   #text(size: 14.5pt)[
     $ell$ chooses the hidden layer. \
     $s$ chooses one example, hence one row. \
     $j$ chooses one hidden unit, hence one column.
   ]
   #v(9pt)
   #small-label([where to observe], color: ACC)
   #v(4pt)
   #text(size: 14.5pt)[
     Linear output $arrow.r bold(z)_ell$ \
     ReLU output $arrow.r bold(h)_ell$ \
     This lecture records $bold(h)_ell$.
   ]
   #v(5pt)
   #text(size: 11.5pt, weight: 650, fill: TEAL)[#link(HOOKS_NB)[Hooks tutorial: activations + gradients ↗]]],
))

== RMS measures the typical size of one coordinate

#align(center, text(size: 23pt, weight: 650)[
  $"RMS"(t)=sqrt("mean"(t^2))$
])

#v(12pt)
#align(center, grid(columns: (1fr, 1fr), gutter: 42pt,
  [#small-label([four coordinates], color: TEAL)
   #v(6pt)
   #text(size: 22pt)[$t=[3,4,0,0]$]
   #v(5pt)
   $norm(t)_2=sqrt(25)=5$ \
   $"RMS"(t)=sqrt(25/4)=2.5$],
  [#small-label([same values, repeated twice], color: BLUE)
   #v(6pt)
   #text(size: 22pt)[$t'=[3,4,0,0,3,4,0,0]$]
   #v(5pt)
   $norm(t')_2=sqrt(50)=7.07$ \
   $"RMS"(t')=sqrt(50/8)=2.5$],
))

#pause
#v(16pt)
#align(center, text(size: 20pt)[
  Repeating the same values increases the norm, even though no coordinate became larger. RMS remains $2.5$.
])

#pause
#v(10pt)
#align(center, text(size: 18pt, fill: MUTED)[
  For $H_ell$, the mean is over all $900 times 128$ entries. RMS is a diagnostic summary, not part of the loss.
])

== Four checks for every hidden layer

#align(center, text(size: 14pt, fill: MUTED)[
  Here $cal(L)$ is mean cross-entropy over all $900$ examples; $H_0=X$ and $G_0=partial cal(L)/partial X$ are $900 times 2$.
])

#v(5pt)
#align(center, grid(columns: (1.08fr, 0.92fr), gutter: 42pt,
  [#small-label([forward pass], color: TEAL)
   #v(6pt)
   #text(size: 18pt, weight: 650, fill: TEAL)[1 · activation RMS] \
   #text(size: 19pt)[$sqrt("mean"(H_ell^2))$] \
   #text(size: 14.5pt, fill: MUTED)[Typical magnitude of a post-ReLU activation.]
   #v(10pt)
   #text(size: 18pt, weight: 650, fill: TEAL)[2 · active-ReLU fraction] \
   #text(size: 19pt)[$"mean"(H_ell>0)$] \
   #text(size: 14.5pt, fill: MUTED)[Fraction of activations that passed the ReLU gate.]
   #v(10pt)
   #text(size: 18pt, weight: 650, fill: TEAL)[3 · finite values] \
   #text(size: 19pt)[$"all isfinite"(H_ell)$] \
   #text(size: 14.5pt, fill: MUTED)[Checks the activations for NaN or infinity.]],
  [#small-label([after backpropagation], color: BLUE)
   #v(6pt)
   #text(size: 19pt)[$G_ell=partial cal(L)/partial H_ell$] \
   #text(size: 14pt, fill: MUTED)[For $ell=1,dots,30$, $H_ell$ and $G_ell$ both have shape $900 times 128$.]
   #v(15pt)
   #text(size: 18pt, weight: 650, fill: BLUE)[4 · activation-gradient RMS] \
   #text(size: 19pt)[$sqrt("mean"(G_ell^2))$] \
   #text(size: 14.5pt, fill: MUTED)[Typical sensitivity of the loss to the layer's activations.]],
))

#align(center, text(size: 12.5pt, weight: 650)[
  Read the trace from the input: gradual drift suggests repeated scale mismatch; a sudden break identifies a layer to inspect.
])

== For $alpha>0$, $"ReLU"(alpha z)=alpha "ReLU"(z)$

#align(center, small-label([diagnostic 1 · activation RMS], color: TEAL))
#v(2pt)
#scaled-depth-diagram

#v(6pt)
#align(center, text(size: 20pt, weight: 650)[
  $bold(h)_0=bold(x), quad W_ell^((alpha))=alpha W_ell^((1)), quad bold(b)_ell=bold(0), quad alpha>0$
])

#pause
#v(11pt)
#align(center, grid(columns: (1fr, 1fr), gutter: 34pt,
  [#small-label([if $z<0$], color: BLUE)
   #v(4pt)
   #text(size: 16pt)[Example: $z=-2$, $alpha=0.5$ \
   $"ReLU"(alpha z)="ReLU"(-1)=0=alpha "ReLU"(-2)$]],
  [#small-label([if $z>0$], color: GREEN)
   #v(4pt)
   #text(size: 16pt)[Example: $z=3$, $alpha=0.5$ \
   $"ReLU"(alpha z)="ReLU"(1.5)=1.5=alpha "ReLU"(3)$]],
))

#pause
#v(10pt)
#align(center, result[
  For a positive scale, ReLU preserves the scale: $"ReLU"(alpha z)=alpha "ReLU"(z)$.
])

== Relative signal scale after successive hidden layers

#align(center, text(size: 17pt, fill: MUTED)[
  The $alpha=1$ run is the reference. Start its relative scale at $1$ and apply $alpha$ once per hidden layer.
])

#v(10pt)
#small-label([$alpha=0.5$ · each layer halves the relative scale], color: BLUE)
#v(4pt)
#align(center, grid(
  columns: (28mm, 7mm, 28mm, 7mm, 28mm, 7mm, 28mm, 7mm, 16mm, 7mm, 44mm),
  align: horizon,
  scale-card([layer 0], [$1$], color: INK),
  align(center, text(size: 19pt, fill: BLUE)[$arrow.r$]),
  scale-card([layer 1], [$0.5$], color: BLUE),
  align(center, text(size: 19pt, fill: BLUE)[$arrow.r$]),
  scale-card([layer 2], [$0.25$], color: BLUE),
  align(center, text(size: 19pt, fill: BLUE)[$arrow.r$]),
  scale-card([layer 3], [$0.125$], color: BLUE),
  align(center, text(size: 19pt, fill: BLUE)[$arrow.r$]),
  align(center, text(size: 18pt, fill: MUTED)[$dots.h$]),
  align(center, text(size: 19pt, fill: BLUE)[$arrow.r$]),
  scale-card([layer 30], [$9.31 times 10^(-10)$], color: BLUE, width: 44mm, value-size: 14.5pt),
))

#v(14pt)
#small-label([$alpha=1.5$ · each layer multiplies the relative scale by $1.5$], color: RED)
#v(4pt)
#align(center, grid(
  columns: (28mm, 7mm, 28mm, 7mm, 28mm, 7mm, 28mm, 7mm, 16mm, 7mm, 44mm),
  align: horizon,
  scale-card([layer 0], [$1$], color: INK),
  align(center, text(size: 19pt, fill: RED)[$arrow.r$]),
  scale-card([layer 1], [$1.5$], color: RED),
  align(center, text(size: 19pt, fill: RED)[$arrow.r$]),
  scale-card([layer 2], [$2.25$], color: RED),
  align(center, text(size: 19pt, fill: RED)[$arrow.r$]),
  scale-card([layer 3], [$3.375$], color: RED),
  align(center, text(size: 19pt, fill: RED)[$arrow.r$]),
  align(center, text(size: 18pt, fill: MUTED)[$dots.h$]),
  align(center, text(size: 19pt, fill: RED)[$arrow.r$]),
  scale-card([layer 30], [$1.92 times 10^5$], color: RED, width: 44mm, value-size: 14.5pt),
))

#v(12pt)
#align(center, text(size: 19pt, weight: 650)[After $ell$ hidden layers, the relative scale is $alpha^ell$.])

== Batch activation RMS scales as $alpha^ell$ #D

#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 22pt,
  [#small-label([layer 1 · one factor], color: TEAL) \
   #v(5pt)
   #text(size: 20pt)[$bold(h)_1^((alpha))=alpha bold(h)_1^((1))$]],
  [#small-label([layer 2 · two factors], color: TEAL) \
   #v(5pt)
   #text(size: 20pt)[$bold(h)_2^((alpha))=alpha^2 bold(h)_2^((1))$]],
  [#small-label([layer $ell$ · $ell$ factors], color: ACC) \
   #v(5pt)
   #text(size: 20pt)[$bold(h)_ell^((alpha))=alpha^ell bold(h)_ell^((1))$]],
))

#v(16pt)
#align(center, grid(columns: (1fr, 10mm, 1fr), gutter: 12pt, align: horizon,
  [#block(inset: 12pt, radius: 4pt, stroke: 1pt + TEAL)[
     #small-label([stack all 900 examples], color: TEAL) \
     #v(5pt)
     $H_ell^((alpha))=alpha^ell H_ell^((1))$ \
     #text(size: 14pt, fill: MUTED)[Every entry is multiplied by the same positive number.]
   ]],
  align(center, text(size: 25pt, fill: TEAL)[$arrow.r$]),
  [#block(inset: 12pt, radius: 4pt, stroke: 1pt + BLUE)[
     #small-label([therefore the RMS scales too], color: BLUE) \
     #v(5pt)
     $"RMS"(H_ell^((alpha)))=alpha^ell "RMS"(H_ell^((1)))$ \
     #text(size: 14pt, fill: MUTED)[$"RMS"(c T)=abs(c)"RMS"(T)$ and $alpha>0$.]
   ]],
))

#v(13pt)
#align(center, result[
  This is the bridge from the per-example activation $bold(h)_ell^((s))$ to the batch measurement plotted in the experiment.
])

== Measure the layer-30 tensor in the $alpha=1$ run #D

#align(center, grid(columns: (0.86fr, 1.14fr), gutter: 30pt, align: top,
  [#small-label([fixed experiment], color: GREEN)
   #v(6pt)
   #text(size: 16pt)[
     $900$ spiral examples; data seed $7$ \
     $30$ hidden layers; width $128$ \
     He-normal weights; model seed $11$ \
     zero biases; before training
   ]],
  [#codebox(size: 12.5pt)[
```python
with torch.no_grad():
    _, H = models[1.0](X)

H30 = H[-1]                  # (900, 128)
H30.square().mean().sqrt()  # 1.096869
```
  ]],
))

#v(8pt)
#align(center, block(
  inset: 10pt, radius: 4pt, stroke: 1.2pt + GREEN,
  fill: GREEN.lighten(95%),
  align(center, [
    $r_1="RMS"(H_30^((1)))
      =sqrt(frac(1,900 times 128)sum_(s=1)^900 sum_(j=1)^128 (H_(30,s j)^((1)))^2)
      =1.096869$
  ]),
))

#v(7pt)
#align(center, text(size: 14.5pt, fill: MUTED)[
  This is a measured value from this seeded run, not a universal constant supplied by He initialization.
])

== The reference RMS times $alpha^30$ predicts the scaled runs #D

#align(center, text(size: 20pt, weight: 650)[
  $r_alpha="RMS"(H_30^((alpha)))=alpha^30 r_1, quad r_1=1.096869$
])

#v(9pt)
#align(center, table(
  columns: (25mm, 48mm, 58mm, 55mm),
  stroke: 0.55pt + MUTED,
  inset: (x: 8pt, y: 7pt),
  align: center,
  table.header(
    [$alpha$],
    [theoretical multiplier $alpha^30$],
    [calibrated prediction $alpha^30 r_1$],
    [direct forward-pass RMS],
  ),
  [$0.5$], [$9.31323 times 10^(-10)$], [$1.021539 times 10^(-9)$], [$1.021539 times 10^(-9)$],
  [$1.0$], [$1$], [$1.096869$], [$1.096869$],
  [$1.5$], [$1.91751 times 10^5$], [$2.103259 times 10^5$], [$2.103258 times 10^5$],
))

#pause
#v(8pt)
#align(center, result[
  The multiplier is derived. One measured reference fixes the absolute scale; the other forward passes check the prediction.
])

== Activation RMS across all 30 hidden layers

#align(center, grid(columns: (1.38fr, 0.62fr), gutter: 26pt, align: horizon,
  [#image("figures/activation_rms_depth.svg", width: 100%)],
  [#small-label([$alpha=0.5$], color: BLUE) \
   #text(size: 14.5pt)[The RMS loses roughly one factor of $0.5$ per layer.]
   #v(13pt)
   #small-label([$alpha=1$], color: GREEN) \
   #text(size: 14.5pt)[The RMS stays near order $1$ through depth.]
   #v(13pt)
   #small-label([$alpha=1.5$], color: RED) \
   #text(size: 14.5pt)[The RMS gains roughly one factor of $1.5$ per layer.]],
))

#v(3pt)
#align(center, text(size: 13.5pt, fill: MUTED)[
  The vertical axis is logarithmic; equal vertical gaps represent multiplicative differences.
])

#v(4pt)
#align(center, text(size: 14pt, weight: 650, fill: TEAL)[
  #link(SCALE_NB)[Minimal executable notebook for this plot ↗]
])

== Activation-only notebook: return every post-ReLU tensor #I

#align(center, grid(columns: (1.18fr, 0.82fr), gutter: 30pt, align: top,
  [#codebox(size: 14pt)[
```python
def forward(self, x):
    H = []
    for layer in self.hidden:
        x = F.relu(layer(x))
        H.append(x)
    return self.output(x), H

with torch.no_grad():
    logits, H = model(X)
    trace = [rms(h) for h in H]
```
  ]],
  [#small-label([what the notebook records], color: TEAL)
   #v(4pt)
   #link(SCALE_NB)[Open notebook ↗] \
   #v(6pt)
   30 post-ReLU tensors \
   each with shape $900 times 128$ \
   one trace for each $alpha$ \
   #v(12pt)
   #small-label([layer-30 RMS], color: BLUE)
   #v(5pt)
   $1.02 times 10^(-9)$ \
   $1.0969$ \
   $2.10 times 10^5$ \
   #v(9pt)
   #text(size: 11.5pt, fill: MUTED)[This cell records activations under *torch.no_grad()*; the hooks tutorial records $G_ell$ after backpropagation.]],
))

== Layer-30 activation distributions

#image("figures/activation_histograms_layer30.svg", width: 100%)

#align(center, text(size: 16pt, fill: MUTED)[
  Each panel uses its own horizontal scale; read the exponent on the axis. The RMS gap reflects the full distributions, not one outlier coordinate.
])

== Active-ReLU fraction alone cannot diagnose signal scale

#image("figures/active_fraction_depth.svg", width: 62%)

#pause
#align(center, grid(columns: (1fr, 1fr), gutter: 28pt,
  [#small-label([rules out], color: BLUE)
   #v(4pt)
   #text(size: 17pt)[At initialization, positive rescaling with zero biases preserves every gate. The $alpha=0.5$ run did not switch them all off.]],
  [#small-label([still misses], color: RED)
   #v(4pt)
   Activation and gradient magnitudes can still vanish or explode.],
))

== Softmax converts three logits into class probabilities #D

#align(center, text(size: 18pt)[
  Fixed spiral point: $bold(x)=(0.429,-0.410)$, true class $y=0$. Start with the $alpha=1$ model.
])

#v(12pt)
#align(center, grid(
  columns: (1fr, 7mm, 1fr, 7mm, 1fr, 7mm, 1fr),
  gutter: 3pt, align: horizon,
  [#align(center)[
    #small-label([1 · logits], color: INK)
    #v(7pt)
    #text(size: 16.5pt)[$bold(o)$ \
      $=(-1.313,-0.070,-1.047)$]
    #v(7pt)
    #text(size: 13.5pt, fill: MUTED)[One score for each class.]
  ]],
  align(center, text(size: 23pt, fill: TEAL)[$arrow.r$]),
  [#align(center)[
    #small-label([2 · shift], color: TEAL)
    #v(7pt)
    #text(size: 16.5pt)[$bold(o)-o_"max"$ \
      $=(-1.243,0,-0.977)$]
    #v(7pt)
    #text(size: 13.5pt, fill: MUTED)[A common shift leaves softmax unchanged.]
  ]],
  align(center, text(size: 23pt, fill: TEAL)[$arrow.r$]),
  [#align(center)[
    #small-label([3 · exponentiate], color: ACC)
    #v(7pt)
    #text(size: 16.5pt)[$exp(bold(o)-o_"max")$ \
      $=(0.288,1,0.376)$]
    #v(7pt)
    #text(size: 13.5pt, fill: MUTED)[Positive numbers; their sum is $1.664$.]
  ]],
  align(center, text(size: 23pt, fill: TEAL)[$arrow.r$]),
  [#align(center)[
    #small-label([4 · normalize], color: GREEN)
    #v(7pt)
    #text(size: 16.5pt)[$bold(p)$ \
      $=(0.173,0.601,0.226)$]
    #v(7pt)
    #text(size: 13.5pt, fill: MUTED)[$p_k=exp(o_k) / (sum_j exp(o_j))$]
  ]],
))

#pause
#v(13pt)
#align(center, grid(columns: (1fr, 1fr), gutter: 34pt,
  [#small-label([prediction], color: BLUE)
   #v(4pt)
   Largest logit and probability: class $1$. \
   The model predicts class $1$.],
  [#small-label([cross-entropy for the true class], color: RED)
   #v(4pt)
   The truth is class $0$, so \
   $cal(L)=-log p_0=-log(0.173)=1.753$.],
))

== Positive scaling preserves class order, not confidence #D

#align(center, text(size: 17pt)[
  Here all 31 weight matrices are scaled and biases are zero:
  $bold(o)^((alpha))=alpha^31 bold(o)^((1))$ for $alpha>0$.
])

#v(6pt)
#align(center, table(
  columns: (16mm, 77mm, 57mm, 22mm, 38mm),
  stroke: 0.55pt + MUTED, inset: (x: 7pt, y: 4pt),
  align: (center, left, center, center, center),
  table.header(
    [#text(size: 14pt)[$alpha$]],
    [#text(size: 14pt)[logits $(o_0,o_1,o_2)$]],
    [#text(size: 14pt)[softmax $(p_0,p_1,p_2)$]],
    [#text(size: 14pt)[pred.]],
    [#text(size: 14pt)[one-point loss \
      $-log p_y$]],
  ),
  [#text(size: 15pt, fill: BLUE, weight: 650)[$0.5$]],
  [#text(size: 12pt)[$(-6.1 times 10^(-10),$ \
    $-3.2 times 10^(-11),-4.9 times 10^(-10))$]],
  [#text(size: 12.5pt)[$approx (0.333,0.333,0.333)$]],
  [#text(size: 15pt, fill: RED, weight: 650)[$1$]],
  [#text(size: 15pt)[$1.099$]],
  [#text(size: 15pt, fill: GREEN, weight: 650)[$1.0$]],
  [#text(size: 13.5pt)[$(-1.313,-0.070,-1.047)$]],
  [#text(size: 14pt)[$(0.173,0.601,0.226)$]],
  [#text(size: 15pt, fill: RED, weight: 650)[$1$]],
  [#text(size: 15pt)[$1.753$]],
  [#text(size: 15pt, fill: RED, weight: 650)[$1.5$]],
  [#text(size: 12.5pt)[$(-377633,-20019,-301145)$]],
  [#text(size: 14pt)[$approx (0,1,0)$]],
  [#text(size: 15pt, fill: RED, weight: 650)[$1$]],
  [#text(size: 15pt, fill: RED, weight: 650)[$357614$]],
))

#pause
#v(5pt)
#align(center, text(size: 12.5pt, fill: MUTED)[
  Probabilities are rounded. PyTorch uses stable log-sum-exp, so the $alpha=1.5$ loss is finite even though $p_0$ displays as $0$.
])

#v(5pt)
#align(center, text(size: 15pt, weight: 600)[
  Positive scaling keeps $o_1>o_2>o_0$, so the prediction stays class $1$.
  Softmax reads the gaps: larger gaps sharpen confidence. Because the true class is $0$, sharpening magnifies this mistake.
])

== Activation-gradient RMS should remain usable across depth

#align(center, text(size: 16.5pt)[
  $G_ell=partial cal(L)/partial H_ell$. Backpropagation travels from layer $30 arrow.r 0$; read every curve from right to left.
])

#v(3pt)
#image("figures/gradient_rms_depth.svg", width: 62%)

#pause
#v(2pt)
#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 20pt,
  [#small-label([$alpha=0.5$ · vanishes], color: BLUE)
   #v(3pt)
   #text(size: 20pt, weight: 650, fill: BLUE)[$"RMS"(G_0)=4.57 times 10^(-13)$]],
  [#small-label([$alpha=1$ · usable], color: GREEN)
   #v(3pt)
   #text(size: 20pt, weight: 650, fill: GREEN)[$"RMS"(G_0)=1.01 times 10^(-3)$]],
  [#small-label([$alpha=1.5$ · explodes], color: RED)
   #v(3pt)
   #text(size: 20pt, weight: 650, fill: RED)[$"RMS"(G_0)=4.15 times 10^2$]],
))

== Before training versus after $150$ Adam updates #D

#align(center, grid(
  columns: (31mm, 7mm, 39mm, 7mm, 39mm, 7mm, 38mm), gutter: 4pt, align: horizon,
  rig-box([#text(size: 12pt)[$H_30$ scale]], TEAL, 31mm),
  text(size: 20pt, fill: TEAL)[$arrow.r$],
  rig-box([#text(size: 12pt)[$W_31 H_30$] #linebreak() #text(size: 10pt)[logits]], INK, 39mm),
  text(size: 20pt, fill: TEAL)[$arrow.r$],
  rig-box([#text(size: 11pt)[softmax + loss]], ACC, 39mm),
  text(size: 20pt, fill: BLUE)[$arrow.r$],
  rig-box([#text(size: 11pt)[backpropagated] #linebreak() #text(size: 10pt)[gradient scale]], BLUE, 38mm),
))

#v(8pt)
#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 16pt, align: top,
  [#block(inset: 9pt, radius: 4pt, stroke: 1pt + BLUE, width: 100%)[
     #small-label([$alpha=0.5$ · collapsed], color: BLUE)
     #v(4pt)
     #small-label([before training], color: MUTED)
     #text(size: 13.5pt)[
       layer-30 RMS: $1.02 times 10^(-9)$ \
       initial loss: $1.09861 approx log 3$ \
       $"RMS"(G_0)$: $4.57 times 10^(-13)$
     ]
     #v(5pt)
     #small-label([after 150 updates], color: BLUE)
     #text(size: 18pt, weight: 650)[accuracy $33.3%$]
   ]],
  [#block(inset: 9pt, radius: 4pt, stroke: 1pt + GREEN, width: 100%)[
     #small-label([$alpha=1$ · reference], color: GREEN)
     #v(4pt)
     #small-label([before training], color: MUTED)
     #text(size: 13.5pt)[
       layer-30 RMS: $1.10$ \
       initial loss: $1.22422$ \
       $"RMS"(G_0)$: $1.01 times 10^(-3)$
     ]
     #v(5pt)
     #small-label([after 150 updates], color: GREEN)
     #text(size: 18pt, weight: 650)[accuracy $100%$]
   ]],
  [#block(inset: 9pt, radius: 4pt, stroke: 1pt + RED, width: 100%)[
     #small-label([$alpha=1.5$ · exploded], color: RED)
     #v(4pt)
     #small-label([before training], color: MUTED)
     #text(size: 13.5pt)[
       layer-30 RMS: $2.10 times 10^5$ \
       initial loss: $1.89 times 10^5$ \
       $"RMS"(G_0)$: $4.15 times 10^2$
     ]
     #v(5pt)
     #small-label([after 150 updates], color: RED)
     #text(size: 18pt, weight: 650)[accuracy $99.4%$]
   ]],
))

#align(center, text(size: 10.5pt, fill: MUTED)[Earlier decision maps show update 50; these accuracies are at update 150. $G_0=partial cal(L)/partial X$ for each model's own loss; it is not a parameter gradient.])

== At $alpha=1.5$, finite logits already give saturated confidence

#align(center, grid(columns: (1fr, 1fr), gutter: 34pt, align: top,
  [#block(width: 100%, inset: 13pt, radius: 4pt, stroke: 1.2pt + RED)[
    #small-label([the fixed point from the worked example], color: RED)
    #v(7pt)
    logits: $bold(o)=(-377633,-20019,-301145)$ \
    probabilities: $bold(p) approx (0,1,0)$ \
    true class: $0$; predicted class: $1$
    #v(8pt)
    #text(size: 22pt, weight: 650, fill: RED)[$-log p_0=357614$]
  ]],
  [#block(width: 100%, inset: 13pt, radius: 4pt, stroke: 1.2pt + ACC)[
    #small-label([over all 900 training points], color: ACC)
    #v(7pt)
    all logits finite: *yes* \
    mean maximum probability: $approx 1$ \
    mean cross-entropy: $1.89 times 10^5$
    #v(8pt)
    #text(size: 17pt, weight: 650)[The model starts extremely confident, often in the wrong class.]
  ]],
))

#pause
#v(13pt)
#align(center, grid(columns: (1fr, 1fr), gutter: 30pt,
  [#small-label([after 150 Adam updates], color: GREEN) \
   loss $0.0819$; training accuracy $99.4%$],
  [#alertbox[Finite is not the same as well-scaled. Eventual recovery does not make the initialization healthy.]],
))

= Normalization recomputes activation scale during training

== He targets activation scale at initialization; training can move it #V

#image("figures/activation_drift.svg", width: 63%)

#pause
#align(center, text(size: 18pt)[
  This is the same plain $alpha=1$ model at initialization and after $20$ of its $150$ Adam updates.
])

#align(center, text(size: 18pt, weight: 650, fill: TEAL)[
  How can a layer use its current batch to reset centre and scale?
])

== One hidden feature for four examples

#two(
  [#align(center, text(size: 19pt)[$Z[:,d]=(1,2,3,4)^top$])
   #v(6pt)
   #image("figures/normalization_numberline.svg", width: 100%)],
  [#small-label([what the symbols mean], color: TEAL)
   #v(5pt)
   $d$ is one hidden feature. \
   The four values come from examples $1$–$4$.
   #pause
   #v(16pt)
   #small-label([the goal], color: ACC)
   #v(5pt)
   Shift the four values to mean $0$, then divide by their standard deviation.],
  r: (1.25fr, 0.75fr),
)

== Step 1: subtract the feature mean

#align(center, text(size: 20pt)[$mu_d=frac(1+2+3+4,4)=2.5$])

#v(8pt)
#small-label([before centring], color: ACC)
#v(4pt)
#align(center, grid(columns: (auto, auto, auto, auto), gutter: 12pt,
  scale-card([example 1], [$1$], color: ACC, width: 34mm),
  scale-card([example 2], [$2$], color: ACC, width: 34mm),
  scale-card([example 3], [$3$], color: ACC, width: 34mm),
  scale-card([example 4], [$4$], color: ACC, width: 34mm),
))

#pause
#v(7pt)
#align(center, text(size: 19pt, weight: 650, fill: TEAL)[↓ #h(8pt) subtract $2.5$])
#v(7pt)
#align(center, grid(columns: (auto, auto, auto, auto), gutter: 12pt,
  scale-card([example 1], [$-1.5$], color: TEAL, width: 34mm),
  scale-card([example 2], [$-0.5$], color: TEAL, width: 34mm),
  scale-card([example 3], [$0.5$], color: TEAL, width: 34mm),
  scale-card([example 4], [$1.5$], color: TEAL, width: 34mm),
))

#pause
#v(7pt)
#align(center, text(size: 18pt, weight: 650, fill: GREEN)[The centred values have mean $0$.])

== Step 2: divide by the feature standard deviation

#align(center, text(size: 19pt)[
  $sigma_d^2=frac((-1.5)^2+(-0.5)^2+0.5^2+1.5^2,4)=1.25,
  quad sigma_d=sqrt(1.25)$
])

#v(8pt)
#align(center, grid(columns: (auto, auto, auto, auto), gutter: 12pt,
  scale-card([], [$-1.5$], color: TEAL, width: 34mm),
  scale-card([], [$-0.5$], color: TEAL, width: 34mm),
  scale-card([], [$0.5$], color: TEAL, width: 34mm),
  scale-card([], [$1.5$], color: TEAL, width: 34mm),
))

#pause
#v(7pt)
#align(center, text(size: 19pt, weight: 650, fill: BLUE)[↓ #h(8pt) divide by $sqrt(1.25)$])
#v(7pt)
#align(center, grid(columns: (auto, auto, auto, auto), gutter: 12pt,
  scale-card([], [$-1.342$], color: BLUE, width: 34mm, value-size: 15pt),
  scale-card([], [$-0.447$], color: BLUE, width: 34mm, value-size: 15pt),
  scale-card([], [$0.447$], color: BLUE, width: 34mm, value-size: 15pt),
  scale-card([], [$1.342$], color: BLUE, width: 34mm, value-size: 15pt),
))

#pause
#v(7pt)
#align(center, text(size: 16pt, weight: 650)[
  Ignoring $epsilon$ for this hand calculation:
  $"mean"(hat(Z)[:,d])=0, quad "RMS"(hat(Z)[:,d])=1$.
  After centring, standard deviation and RMS agree.
])
#align(center, text(size: 13.5pt, fill: MUTED)[Code uses $epsilon>0$ in the denominator.])

== Apply learned scale and shift after standardization

#align(center, text(size: 18pt)[
  Start from the standardized values. For one feature, choose $gamma_d=2$ and $beta_d=1$.
])

#v(8pt)
#align(center, grid(columns: (auto, auto, auto, auto), gutter: 12pt,
  scale-card([], [$-1.342$], color: BLUE, width: 34mm, value-size: 15pt),
  scale-card([], [$-0.447$], color: BLUE, width: 34mm, value-size: 15pt),
  scale-card([], [$0.447$], color: BLUE, width: 34mm, value-size: 15pt),
  scale-card([], [$1.342$], color: BLUE, width: 34mm, value-size: 15pt),
))

#pause
#v(7pt)
#align(center, text(size: 19pt, weight: 650, fill: ACC)[↓ #h(8pt) $y_(n d)=2 hat(z)_(n d)+1$])
#v(7pt)
#align(center, grid(columns: (auto, auto, auto, auto), gutter: 12pt,
  scale-card([], [$-1.683$], color: ACC, width: 34mm, value-size: 15pt),
  scale-card([], [$0.106$], color: ACC, width: 34mm, value-size: 15pt),
  scale-card([], [$1.894$], color: ACC, width: 34mm, value-size: 15pt),
  scale-card([], [$3.683$], color: ACC, width: 34mm, value-size: 15pt),
))

#pause
#v(7pt)
#align(center, grid(columns: (1fr, 1fr), gutter: 30pt,
  [#small-label([$gamma_d$: learned feature scale], color: ACC) \
   restores or changes the standardized scale],
  [#small-label([$beta_d$: learned feature offset], color: TEAL) \
   restores or changes the standardized centre],
))

== Normalization chooses which entries share statistics

For a batch matrix $Z in RR^(B times D)$, let $S_(n d)$ contain the entries that share statistics with position $(n,d)$.

$ mu_(n d)=frac(1,abs(S_(n d))) sum_((r,k) in S_(n d)) Z_(r k), quad
  sigma_(n d)^2=frac(1,abs(S_(n d))) sum_((r,k) in S_(n d))(Z_(r k)-mu_(n d))^2. $

#pause
$ hat(Z)_(n d)=frac(Z_(n d)-mu_(n d),sqrt(sigma_(n d)^2+epsilon)), quad
  Y_(n d)=gamma_d hat(Z)_(n d)+beta_d. $

#pause
#align(center, grid(columns: (1fr, 1fr), gutter: 30pt,
  [#small-label([BatchNorm choice], color: ACC) \
   $S_(n d)={(r,d): r=1,dots,B}$],
  [#small-label([LayerNorm choice], color: TEAL) \
   $S_(n d)={(n,k): k=1,dots,D}$],
))

#v(9pt)
#align(center, text(size: 18pt, weight: 650)[The set changes; the learned affine parameters remain feature-indexed: $gamma_d,beta_d$.])

== Rows are examples; columns are hidden features

#align(center, grid(columns: (auto, auto), gutter: 10pt, align: horizon,
  [#rotate(-90deg, text(size: 14pt, weight: 650, fill: TEAL)[examples $n=1,dots,B$])],
  [#align(center, [
    #text(size: 14pt, weight: 650, fill: ACC)[hidden features $d=1,dots,D$]
    #v(5pt)
    #grid(columns: 3, gutter: 5pt,
      matrix-cell([$1$]), matrix-cell([$2$]), matrix-cell([$6$]),
      matrix-cell([$2$]), matrix-cell([$4$]), matrix-cell([$5$]),
      matrix-cell([$3$]), matrix-cell([$8$]), matrix-cell([$4$]),
      matrix-cell([$4$]), matrix-cell([$10$]), matrix-cell([$3$]),
    )
  ])],
))

#v(7pt)
#align(center, text(size: 19pt)[$Z_ell in RR^(B times D)$: preactivations before ReLU, here $B=4$ and $D=3$.])

#pause
#align(center, text(size: 21pt, weight: 650)[For $(Z_ell)_(1,1)=1$, should $S_(1,1)$ be its column or its row?])

== For an MLP batch, BatchNorm uses one feature across examples

#align(center, grid(columns: 3, gutter: 5pt,
  matrix-cell([$1$], fill: ACC.lighten(82%), stroke: 1pt + ACC), matrix-cell([$2$]), matrix-cell([$6$]),
  matrix-cell([$2$], fill: ACC.lighten(82%), stroke: 1pt + ACC), matrix-cell([$4$]), matrix-cell([$5$]),
  matrix-cell([$3$], fill: ACC.lighten(82%), stroke: 1pt + ACC), matrix-cell([$8$]), matrix-cell([$4$]),
  matrix-cell([$4$], fill: ACC.lighten(82%), stroke: 1pt + ACC), matrix-cell([$10$]), matrix-cell([$3$]),
))

#v(9pt)
The first feature column is the same four-value worked set $[1,2,3,4]$:

$ hat(Z)[:,1]=(-1.342,-0.447,0.447,1.342)^top. $

#pause
#align(center, text(size: 18pt, fill: MUTED)[
  BatchNorm computes $mu_d,sigma_d$ over examples; each feature has learned $gamma_d,beta_d$.
])

== LayerNorm over $D$ features uses one example at a time

#align(center, grid(columns: 3, gutter: 5pt,
  matrix-cell([$1$], fill: TEAL.lighten(82%), stroke: 1pt + TEAL),
  matrix-cell([$2$], fill: TEAL.lighten(82%), stroke: 1pt + TEAL),
  matrix-cell([$6$], fill: TEAL.lighten(82%), stroke: 1pt + TEAL),
  matrix-cell([$2$]), matrix-cell([$4$]), matrix-cell([$5$]),
  matrix-cell([$3$]), matrix-cell([$8$]), matrix-cell([$4$]),
  matrix-cell([$4$]), matrix-cell([$10$]), matrix-cell([$3$]),
))

#v(9pt)
For the first row $n=1$, whose features are $[1,2,6]$:

$ mu_(n=1)^"LN"=3, quad (sigma_(n=1)^"LN")^2=14/3, quad sigma_(n=1)^"LN"=sqrt(14/3), $
$ hat(z)_(1 dot)=[-0.926,-0.463,1.389]. $

#pause
#align(center, text(size: 16pt, fill: MUTED)[
  The statistics are per example; the learned affine parameters are normally indexed by feature.
])

== BatchNorm and LayerNorm reduce different axes

#align(center, grid(columns: (1fr, 1fr), gutter: 36pt, align: top,
  [#align(center)[
    #small-label([BatchNorm · one feature column], color: ACC)
    #v(7pt)
    #grid(columns: 3, gutter: 4pt,
      matrix-cell([$1$], fill: ACC.lighten(82%), stroke: 1pt + ACC), matrix-cell([$2$]), matrix-cell([$6$]),
      matrix-cell([$2$], fill: ACC.lighten(82%), stroke: 1pt + ACC), matrix-cell([$4$]), matrix-cell([$5$]),
      matrix-cell([$3$], fill: ACC.lighten(82%), stroke: 1pt + ACC), matrix-cell([$8$]), matrix-cell([$4$]),
      matrix-cell([$4$], fill: ACC.lighten(82%), stroke: 1pt + ACC), matrix-cell([$10$]), matrix-cell([$3$]),
    )
    #v(8pt)
    #text(size: 16pt, weight: 650)[statistics over examples]
    #text(size: 14pt, fill: MUTED)[depend on the other rows during training]
  ]],
  [#align(center)[
    #small-label([LayerNorm · one example row], color: TEAL)
    #v(7pt)
    #grid(columns: 3, gutter: 4pt,
      matrix-cell([$1$], fill: TEAL.lighten(82%), stroke: 1pt + TEAL),
      matrix-cell([$2$], fill: TEAL.lighten(82%), stroke: 1pt + TEAL),
      matrix-cell([$6$], fill: TEAL.lighten(82%), stroke: 1pt + TEAL),
      matrix-cell([$2$]), matrix-cell([$4$]), matrix-cell([$5$]),
      matrix-cell([$3$]), matrix-cell([$8$]), matrix-cell([$4$]),
      matrix-cell([$4$]), matrix-cell([$10$]), matrix-cell([$3$]),
    )
    #v(8pt)
    #text(size: 16pt, weight: 650)[statistics over features]
    #text(size: 14pt, fill: MUTED)[does not use the other examples]
  ]],
))

#pause
#v(10pt)
#align(center, result[Same matrix, different reduction axis.])

== The normalized value of $1$ depends on its training batch

Keep the raw value $z=1$ fixed and change only its companions.

#v(10pt)
#small-label([batch A], color: ACC)
#v(4pt)
#align(center, grid(columns: (28mm, 28mm, 28mm, 28mm, 10mm, 42mm), gutter: 7pt, align: horizon,
  scale-card([], [$1$], color: ACC, width: 28mm),
  scale-card([], [$2$], color: MUTED, width: 28mm),
  scale-card([], [$3$], color: MUTED, width: 28mm),
  scale-card([], [$4$], color: MUTED, width: 28mm),
  text(size: 22pt, fill: ACC)[$arrow.r$],
  scale-card([normalized $1$], [$-1.342$], color: ACC, width: 42mm, value-size: 15pt),
))

#pause
#v(10pt)
#small-label([batch B], color: TEAL)
#v(4pt)
#align(center, grid(columns: (28mm, 28mm, 28mm, 28mm, 10mm, 42mm), gutter: 7pt, align: horizon,
  scale-card([], [$-2$], color: MUTED, width: 28mm),
  scale-card([], [$-1$], color: MUTED, width: 28mm),
  scale-card([], [$0$], color: MUTED, width: 28mm),
  scale-card([], [$1$], color: TEAL, width: 28mm),
  text(size: 22pt, fill: TEAL)[$arrow.r$],
  scale-card([normalized $1$], [$+1.342$], color: TEAL, width: 42mm, value-size: 15pt),
))

#pause
#notebox[The raw value is unchanged. Its normalized value changes because the training batch changed.]

== Evaluation mode uses stored training statistics

#align(center, grid(
  columns: (28mm, 8mm, 45mm, 8mm, 45mm, 8mm, 28mm), gutter: 4pt, align: horizon,
  rig-box([#text(size: 13pt)[training batches]], ACC, 28mm),
  align(center, text(size: 22pt, fill: ACC)[$arrow.r$]),
  rig-box([#text(size: 13pt)[update running #linebreak() mean and variance]], ACC, 45mm),
  align(center, text(size: 22pt, fill: ACC)[$arrow.r$]),
  rig-box([#text(size: 12pt)[stop running-stat #linebreak() updates]], GREEN, 45mm),
  align(center, text(size: 22pt, fill: GREEN)[$arrow.r$]),
  rig-box([#text(size: 13pt)[evaluation]], GREEN, 28mm),
))

#pause
#align(center, grid(columns: (1fr, 1fr), gutter: 28pt,
  [#small-label([model.train()], color: ACC) \
   use batch statistics and update running estimates],
  [#small-label([model.eval()], color: GREEN) \
   use the stored running estimates],
))

#align(center, text(size: 14.5pt, fill: MUTED)[
  *model.eval()* does not freeze parameters or disable autograd. LayerNorm uses the same per-example computation in both modes.
])

== In this experiment: Linear $arrow.r$ BatchNorm $arrow.r$ ReLU

#v(11pt)
#align(center, grid(
  columns: (27mm, 8mm, 27mm, 8mm, 27mm, 8mm, 32mm, 8mm, 27mm, 8mm, 27mm, 8mm, 27mm),
  gutter: 2pt, align: horizon,
  rig-box([$H_(ell-1)$], TEAL, 27mm),
  text(size: 21pt, fill: TEAL)[$arrow.r$],
  rig-box([Linear], INK, 27mm),
  text(size: 21pt, fill: ACC)[$arrow.r$],
  rig-box([$Z_ell$], ACC, 27mm),
  text(size: 21pt, fill: ACC)[$arrow.r$],
  rig-box([BatchNorm], ACC, 32mm),
  text(size: 21pt, fill: BLUE)[$arrow.r$],
  rig-box([$tilde(Z)_ell$], BLUE, 27mm),
  text(size: 21pt, fill: GREEN)[$arrow.r$],
  rig-box([ReLU], GREEN, 27mm),
  text(size: 21pt, fill: TEAL)[$arrow.r$],
  rig-box([$H_ell$], TEAL, 27mm),
))

#v(10pt)
#align(center, text(size: 16pt)[
  $Z_ell=H_(ell-1)W_ell^top+bold(1)b_ell^top$.
  Each $H$ and $Z$ is a batch matrix; BatchNorm computes column statistics during training.
])

#pause
#align(center, codebox(size: 14pt)[
```python
z = linear(h)
h = torch.relu(batch_norm(z))
```
])

== After BatchNorm and ReLU, activation RMS is about $0.707$

#small-label([standardized preactivations], color: BLUE)
#v(4pt)
#align(center, grid(columns: (auto, auto, auto, auto), gutter: 12pt,
  scale-card([], [$-1.342$], color: BLUE, width: 34mm, value-size: 15pt),
  scale-card([], [$-0.447$], color: BLUE, width: 34mm, value-size: 15pt),
  scale-card([], [$0.447$], color: BLUE, width: 34mm, value-size: 15pt),
  scale-card([], [$1.342$], color: BLUE, width: 34mm, value-size: 15pt),
))

#pause
#v(7pt)
#align(center, text(size: 19pt, weight: 650, fill: GREEN)[↓ #h(8pt) ReLU])
#v(7pt)
#align(center, grid(columns: (auto, auto, auto, auto), gutter: 12pt,
  scale-card([], [$0$], color: MUTED, width: 34mm),
  scale-card([], [$0$], color: MUTED, width: 34mm),
  scale-card([], [$0.447$], color: GREEN, width: 34mm, value-size: 15pt),
  scale-card([], [$1.342$], color: GREEN, width: 34mm, value-size: 15pt),
))

#pause
#v(8pt)
#align(center, text(size: 20pt)[
  $"RMS"(H)=sqrt(frac(0+0+0.2+1.8,4))=sqrt(1/2)=0.707$
])

#pause
#align(center, text(size: 18pt, weight: 650)[
  Prediction at initialization: with $gamma=1$, $beta=0$, and roughly symmetric normalized preactivations, every hidden layer should report RMS near $0.707$.
])

== In this 30-hidden-layer run, BatchNorm keeps activation RMS near $0.7$

#align(center, text(size: 11pt, fill: MUTED)[
  Activation-gradient RMS need not equal $1$; compare collapse with a usable depthwise signal. Read the gradient panel from layer $30 arrow.r 0$.
])

#image("figures/batchnorm_depth_comparison.svg", width: 82%)

#pause
#align(center, grid(columns: (1fr, 1fr), gutter: 28pt,
  [#small-label([plain · $alpha=0.5$], color: RED) \
   layer-30 RMS $=1.02 times 10^(-9)$ \
   $"RMS"(G_0)=4.57 times 10^(-13)$],
  [#small-label([BatchNorm · $alpha=0.5$], color: GREEN) \
   layer-30 RMS $approx 0.713$ \
   $"RMS"(G_0) approx 0.225$],
))

== With $alpha=0.5$, BatchNorm reaches $100%$ training accuracy

#image("figures/batchnorm_repair_comparison.svg", width: 82%)

#pause
#align(center, text(size: 17pt)[
  All $900$ examples form one batch, so this run removes mini-batch-composition noise while retaining BatchNorm's normalization effect.
])

= Residual blocks add an identity term to the local Jacobian

== Why add an identity path? #V

#chain-diagram

#align(center, text(size: 12pt, fill: MUTED)[Scalar schematic; the same product-of-Jacobians issue applies to vector activations.])

#pause
#v(9pt)
For a parameter in layer 1, the gradient still crosses the Jacobian of every later layer.

#pause
#v(12pt)
#align(center, grid(columns: (1fr, auto, 1fr), gutter: 20pt, align: horizon,
  [#small-label([local question], color: ACC) \
   Are the activation and gradient magnitudes usable at each layer?],
  [#text(size: 28pt, fill: MUTED)[$arrow.r$]],
  [#small-label([route question], color: GREEN) \
   Must useful information pass through every learned transformation?],
))

== Can a deep block simply copy its input?

Take the scalar input $h_ell=3$ and target $y=3$.

#pause
#align(center, grid(columns: (1fr, 1fr), gutter: 34pt,
  [#small-label([plain scalar layer], color: ACC)
   #v(6pt)
   $hat(y)=w h_ell$ \
   $hat(y)=3$ only when $w=1$.],
  [#small-label([identity plus correction], color: GREEN)
   #v(6pt)
   $hat(y)=h_ell+w h_ell$ \
   At $w=0$, $hat(y)=3$ already.],
))

#pause
#v(14pt)
#align(center, text(size: 22pt, weight: 650)[
  The residual parameterization represents identity before the learned branch does any work.
])

== A residual block adds a correction with the same shape

#align(center, diagram(
  spacing: (24mm, 14mm), node-stroke: 0.9pt + INK, node-fill: white,
  {
    node((0, 0), $bold(h)_ell$, radius: 8mm, stroke: 1.1pt + TEAL)
    node((1.8, 0), $F_ell$, shape: fletcher.shapes.rect, corner-radius: 3pt,
      inset: 10pt, fill: ACC.lighten(90%), stroke: 1.1pt + ACC)
    node((3.6, 0), $+$, radius: 6.5mm, stroke: 1pt + INK)
    node((5.3, 0), $bold(h)_(ell+1)$, radius: 9mm, stroke: 1.1pt + TEAL)
    edge((0, 0), (1.8, 0), "-|>", stroke: 1.1pt + TEAL)
    edge((1.8, 0), (3.6, 0), "-|>", stroke: 1.1pt + ACC)
    edge((3.6, 0), (5.3, 0), "-|>", stroke: 1.1pt + TEAL)
    edge((0, 0), (3.6, 0), "-|>", bend: -48deg, stroke: 1.6pt + GREEN,
      label: text(size: 11pt, weight: 650, fill: GREEN)[identity], label-side: left)
  },
))

#v(6pt)
#align(center, text(size: 25pt)[$bold(h)_(ell+1)=bold(h)_ell+F_ell(bold(h)_ell)$])

#pause
#align(center, grid(columns: (1fr, 1fr), gutter: 34pt,
  [#small-label([shape requirement], color: TEAL) \
   $bold(h)_ell, F_ell(bold(h)_ell) in RR^D$],
  [#small-label([one numeric addition], color: ACC) \
   $mat(2; -1)+mat(0.3; 0.2)=mat(2.3; -0.8)$],
))

== In the scalar example, the local gradient factor is $1+F'_ell(h_ell)$

Let $h_(ell+1)=h_ell+F_ell(h_ell)$, choose the explicit branch
$F_ell(h)=-0.1h$, and let the arriving gradient be
$g_(ell+1)=partial cal(L)/partial h_(ell+1)=1$.

#pause
#align(center, text(size: 27pt)[
  $g_ell = frac(partial cal(L), partial h_ell)
  = g_(ell+1)(1+F'_ell(h_ell))$
])

#pause
#align(center, text(size: 29pt, weight: 650, fill: GREEN)[
  $g_ell=1(1-0.1)=0.9$
])

#v(10pt)
#align(center, text(size: 16pt, fill: MUTED)[
  This scalar branch is chosen only to expose the derivative. It is not the ReLU branch used in the spiral experiment.
])

== The vector derivative contains an identity term

For $bold(h)_ell in RR^D$, define

$ J_(F_ell)=frac(partial F_ell(bold(h)_ell), partial bold(h)_ell) in RR^(D times D),
  quad bold(g)_ell=frac(partial cal(L),partial bold(h)_ell) in RR^D. $

#pause
#align(center, text(size: 25pt)[
  $frac(partial bold(h)_(ell+1),partial bold(h)_ell)=I+J_(F_ell)$
])

#pause
#align(center, text(size: 22pt)[
  $bold(g)_ell
   =(I+J_(F_ell))^top bold(g)_(ell+1)
   =underbrace(bold(g)_(ell+1))_("identity term")
    +underbrace(J_(F_ell)^top bold(g)_(ell+1))_("branch term")$
])

#pause
#align(center, text(size: 18pt, fill: MUTED)[Addition requires matching shapes; otherwise the skip needs a projection.])

== Over five blocks: $10^(-5)$ versus $0.59049$

Continue the scalar example with $F'_ell=-0.1$ at every block and the same unit arriving gradient.

#pause
#align(center, grid(columns: (1fr, 1fr), gutter: 32pt,
  [#small-label([branch-only parameterization], color: RED)
   #v(6pt)
   derivative magnitude per block: $0.1$ \
   $abs((-0.1)^5)=10^(-5)$],
  [#small-label([identity + branch], color: GREEN)
   #v(6pt)
   derivative per block: $1-0.1=0.9$ \
   $(0.9)^5=0.59049$],
))

#pause
#notebox[
  This is an abstract comparison of two parameterizations, not a measured claim about all residual networks.
  A large branch can amplify the signal, and $I+J_F$ can still cancel along some direction.
]

== In this experiment, $F_ell(bold(h))="ReLU"(V_ell bold(h))$

#align(center, grid(columns: (0.9fr, auto, 1.1fr), gutter: 18pt, align: horizon,
  [#small-label([stem changes width], color: TEAL)
   #v(6pt)
   $bold(x) in RR^2 arrow.r bold(h)_1 in RR^128$ \
   $bold(h)_1="ReLU"(W_"stem" bold(x))$],
  [#text(size: 30pt, fill: GREEN)[$arrow.r$]],
  [#small-label([29 matching-width blocks], color: GREEN)
   #v(6pt)
   $bold(h)_(ell+1)=bold(h)_ell+"ReLU"(V_ell bold(h)_ell)$ \
   $ell=1,dots,29$],
))

#v(8pt)
#align(center, text(size: 15pt, fill: MUTED)[
  Away from a ReLU kink, $J_(F_ell)=D_ell V_ell$ with $D_ell="diag"(bold(1)[V_ell bold(h)_ell>0])$.
])

#pause
#v(13pt)
#notebox[
  This is an isolation experiment, not a standard modern ResNet block. There is no activation after the addition.
  The nonnegative branch corrections can make activations large; the identity addition isolates the alternate route.
]

== Compare three versions of the same 30-hidden-layer spiral model

#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 18pt,
  [#small-label([plain], color: RED)
   #v(5pt)
   30 × \
   $"Linear" arrow.r "ReLU"$],
  [#small-label([BatchNorm], color: ACC)
   #v(5pt)
   30 × \
   $"Linear" arrow.r "BatchNorm" arrow.r "ReLU"$],
  [#small-label([residual toy], color: GREEN)
   #v(5pt)
   one $2 arrow.r 128$ stem \
   then 29 × $bold(h)+"ReLU"(V_ell bold(h))$],
))

#pause
#v(10pt)
#align(center, text(size: 16pt)[
  Fixed: spiral data, depth, width, loss, Adam at $3 times 10^(-4)$, seed, and corresponding $alpha=0.5$ base weight directions.
])

#pause
#v(8pt)
#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 16pt,
  [#small-label([plain prediction], color: RED) \
   scale should collapse through depth],
  [#small-label([BatchNorm prediction], color: ACC) \
   current activation scale should be controlled],
  [#small-label([residual-toy prediction], color: GREEN) \
   an identity gradient term remains, but positive corrections may enlarge activations],
))

== Measured at initialization for the three model variants

#let evidence-cell(body, fill: white, stroke: 0.5pt + MUTED) = block(
  width: 100%, height: 14mm, inset: 5pt, fill: fill, stroke: stroke,
  align(center + horizon, body),
)
#align(center, text(size: 15pt)[#grid(
  columns: (31mm, 45mm, 45mm, 35mm), gutter: 0pt,
  evidence-cell([*route*], fill: INK.lighten(86%)),
  evidence-cell([*layer-30 activation RMS*], fill: INK.lighten(86%)),
  evidence-cell([*$"RMS"(G_0)$* #linebreak() $G_0=partial cal(L)/partial X$], fill: INK.lighten(86%)),
  evidence-cell([*initial loss*], fill: INK.lighten(86%)),
  evidence-cell([plain], stroke: 0.8pt + RED),
  evidence-cell([$1.02 times 10^(-9)$], stroke: 0.8pt + RED),
  evidence-cell([$4.57 times 10^(-13)$], stroke: 0.8pt + RED),
  evidence-cell([$1.09861$], stroke: 0.8pt + RED),
  evidence-cell([BatchNorm], stroke: 0.8pt + ACC),
  evidence-cell([$0.713$], stroke: 0.8pt + ACC),
  evidence-cell([$0.225$], stroke: 0.8pt + ACC),
  evidence-cell([$1.11666$], stroke: 0.8pt + ACC),
  evidence-cell([residual toy], stroke: 0.8pt + GREEN),
  evidence-cell([$698.57$], stroke: 0.8pt + GREEN),
  evidence-cell([$0.329$], stroke: 0.8pt + GREEN),
  evidence-cell([$177.84$], stroke: 0.8pt + GREEN),
)])

#pause
#v(12pt)
#align(center, text(size: 16.5pt)[
  BatchNorm controls activation scale. The residual measurements are consistent with a usable backward route, while the $I+J_F$ derivation identifies the mechanism; this minimal branch also starts with very large activations.
])

== Training results for the controlled comparison

#image("figures/repair_comparison.svg", width: 80%)

#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 20pt,
  metric([plain], [33.3%], color: RED),
  metric([BatchNorm], [100%], color: ACC),
  metric([residual toy], [98.6%], color: GREEN),
))

#align(center, text(size: 15.5pt, fill: MUTED)[Final training accuracy after 150 updates. This controlled example compares mechanisms, not architectures in general.])

== Decision regions after 150 updates

#image("figures/repair_decision_boundaries.svg", width: 90%)

#pause
#align(center, text(size: 17.5pt)[
  Data, optimizer, learning rate, seed, base weight directions, and training budget are fixed. The computation mechanism through the hidden layers changes.
])

#align(center, text(size: 13.5pt, fill: MUTED)[This experiment demonstrates optimization and trainability on the training set; the jagged BatchNorm boundary is not evidence of better generalization.])

= Diagnose the failing layer before changing Adam

== At $alpha=0.5$, scale collapses while half the ReLUs remain active #D

#align(center, codebox(size: 14pt)[
```text
layer:         1        10        20        30
act RMS:    1.95e-1   4.13e-4   5.57e-7   1.02e-9
active:       0.50      0.50      0.46      0.49
finite:       yes       yes       yes       yes
```
])

#pause
#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 22pt,
  [#small-label([symptom], color: RED) \
   Representation scale collapses with depth.],
  [#small-label([evidence against widespread gate closure], color: ACC) \
   The aggregate active fraction stays near $0.5$; individual neurons could still be dead.],
  [#small-label([first suspect], color: TEAL) \
   Per-layer weight gain is too small.],
))

== Test the scale hypothesis without changing weight directions

#align(center, text(size: 15pt)[#grid(
  columns: (30mm, 47mm, 47mm, 36mm), gutter: 0pt,
  evidence-cell([$alpha$], fill: INK.lighten(86%)),
  evidence-cell([*layer-30 activation RMS*], fill: INK.lighten(86%)),
  evidence-cell([*$"RMS"(G_0)$* #linebreak() $G_0=partial cal(L)/partial X$], fill: INK.lighten(86%)),
  evidence-cell([*active fraction*], fill: INK.lighten(86%)),
  evidence-cell([$0.5$], stroke: 0.8pt + RED),
  evidence-cell([$1.02 times 10^(-9)$], stroke: 0.8pt + RED),
  evidence-cell([$4.57 times 10^(-13)$], stroke: 0.8pt + RED),
  evidence-cell([$0.493$], stroke: 0.8pt + RED),
  evidence-cell([$1.0$], stroke: 0.8pt + GREEN),
  evidence-cell([$1.10$], stroke: 0.8pt + GREEN),
  evidence-cell([$1.01 times 10^(-3)$], stroke: 0.8pt + GREEN),
  evidence-cell([$0.493$], stroke: 0.8pt + GREEN),
)])

#pause
#v(15pt)
#align(center, text(size: 20pt, weight: 650)[
  At initialization, zero biases and positive rescaling preserve the gates; activation and gradient scales change. This supports the gain diagnosis.
])

== Measure $arrow.r$ locate $arrow.r$ change one factor $arrow.r$ measure again

#align(center, grid(
  columns: (27mm, 7mm, 27mm, 7mm, 27mm, 7mm, 27mm, 7mm, 27mm),
  gutter: 1.5pt, align: horizon,
  rig-box([#text(size: 10pt)[training stuck]], RED, 27mm),
  align(center, text(size: 19pt, fill: TEAL)[$arrow.r$]),
  rig-box([#text(size: 10pt)[trace $H_ell, G_ell$]], TEAL, 27mm),
  align(center, text(size: 19pt, fill: TEAL)[$arrow.r$]),
  rig-box([#text(size: 10pt)[first bad layer]], TEAL, 27mm),
  align(center, text(size: 19pt, fill: ACC)[$arrow.r$]),
  rig-box([#text(size: 10pt)[one suspect]], ACC, 27mm),
  align(center, text(size: 19pt, fill: GREEN)[$arrow.r$]),
  rig-box([#text(size: 10pt)[one change]], GREEN, 27mm),
))

#pause
#v(18pt)
#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 24pt,
  [#small-label([scale], color: TEAL) \
   RMS shrinks or grows systematically.],
  [#small-label([gates], color: ACC) \
   Near $0$: gates close. Near $1$: preactivations shifted positive and ReLU is nearly linear.],
  [#small-label([numerics], color: RED) \
   Find the first non-finite layer.],
))

#pause
#align(center, text(size: 18pt, weight: 650)[Rerun the same measurements; do not judge the repair from loss alone.])

== Exit ticket: diagnose a different trace

A new 30-hidden-layer ReLU MLP reports

#align(center, codebox(size: 14pt)[
```text
layer:         1        10        20        30
act RMS:      0.80      0.72      0.65      0.60
active:       0.49      0.31      0.08      0.01
finite:       yes       yes       yes       yes
```
])

#pause
Write three lines:

#enum(
  [the *symptom* in plain language;],
  [the *first suspect*;],
  [the *smallest informative measurement or one-factor test*.],
  spacing: 7pt,
)

== Exit-ticket answer: the ReLU gates progressively close

#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 22pt,
  [#small-label([symptom], color: RED)
   #v(7pt)
   #text(size: 17pt)[Activation RMS falls modestly, but the active fraction collapses from $0.49$ to $0.01$.]],
  [#small-label([first suspect], color: ACC)
   #v(7pt)
   #text(size: 17pt)[The preactivation distribution is drifting to the negative side, closing almost every ReLU gate.]],
  [#small-label([smallest test], color: TEAL)
   #v(7pt)
   #text(size: 17pt)[Plot the preactivation mean and histogram by layer; locate where the negative shift begins.]],
))

#pause
#v(16pt)
#align(center, result[
  This trace is a gate problem, not the $alpha=0.5$ scale-collapse pattern: the active fraction is the distinguishing evidence.
])

== The same measurements explain all three $alpha$ runs

#image("figures/decision_boundaries_step150.svg", width: 70%)

#pause
#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 20pt,
  [#align(center, text(size: 14pt, weight: 650, fill: RED)[$alpha=0.5$])
   #align(center, text(size: 15pt)[forward and backward scales collapse])],
  [#align(center, text(size: 14pt, weight: 650, fill: GREEN)[$alpha=1.0$])
   #align(center, text(size: 15pt)[usable scales across depth])],
  [#align(center, text(size: 14pt, weight: 650, fill: ACC)[$alpha=1.5$])
   #align(center, text(size: 15pt)[very large but finite scales])],
))

#pause
#align(center, text(size: 14.5pt, weight: 650)[
  For a new run: trace activations and gradients, locate the first break, change one cause, and repeat the same measurements.
])

== Primary sources and reproducible evidence

#set text(size: 16pt)
#grid(columns: (1fr, 1fr), gutter: 26pt,
  [*Opening intuition* \
   #link(ANDREW_VG)[DeepLearning.AI: Vanishing/Exploding Gradients] \
   #link(ANDREW_INIT)[DeepLearning.AI: Weight Initialization in a Deep Network ↗]
   #v(10pt)
   *Initialization* \
   #link(GLOROT)[Glorot & Bengio (2010), Xavier initialization ↗] \
   #link(HEINIT)[He et al. (2015), rectifier-aware initialization ↗]],
  [*Shortcuts* \
   #link(RESNET)[He et al. (2016), Deep Residual Learning ↗]
   #v(10pt)
   *Normalization* \
   #link(BN)[Ioffe & Szegedy (2015), Batch Normalization ↗] \
   #link(LN)[Ba, Kiros & Hinton (2016), Layer Normalization ↗]
   #v(10pt)
   *This lecture's evidence* \
   #link(SCALE_NB)[Minimal scale-through-depth notebook ↗] \
   #link(NB)[Executed PyTorch diagnostic notebook ↗] \
   #link(HOOKS_NB)[PyTorch hooks tutorial ↗] \
   #link(SYMMETRY_DEMO)[Hidden-unit symmetry lab ↗] \
   #link("https://github.com/nipunbatra/dl-teaching/tree/master/lecture6")[
     Evidence tables, scripts, and figures ↗
   ]],
)

#pause
#place(bottom + center, dy: -2pt,
  text(size: 14pt, fill: MUTED)[All spiral figures were generated from the fixed-seed experiment used in the notebook.])

// ───────────────────────── Optional backup ─────────────────────────

== Optional · Unequal input units can dominate the first layer

One neuron receives age $40$ and annual income *2,000,000*.
Suppose both weights are $0.2$.

#align(center, grid(columns: (1fr, auto, 1fr), gutter: 24pt, align: horizon,
  [#small-label([age contribution], color: TEAL) \
   #text(size: 26pt)[$0.2(40)=8$]],
  [#text(size: 34pt, fill: MUTED)[$"vs"$]],
  [#small-label([income contribution], color: RED) \
   #text(size: 26pt)[0.2 × 2,000,000 = 400,000]],
))

#pause
#align(center, text(size: 20pt)[After standardization, values $0.5$ and $-0.5$ contribute $0.1$ and $-0.1$.])

#notebox[Initialization formulas assume incoming coordinates already have comparable, roughly order-one scale.]

== Optional · Ten saturated sigmoid gates

#two(
  [#image("figures/sigmoid_gate.svg", width: 100%)],
  [#set text(size: 18pt)
   At $z=5$,
   $ sigma'(5)=sigma(5)(1-sigma(5)) approx 0.00665. $
   #pause
   With ten gates all at $z=5$ and intervening scalar weight factors fixed at $1$, the gate-only multiplier is
   $ 1 arrow.r 6.65 times 10^(-3) arrow.r 4.42 times 10^(-5) arrow.r dots $
   $ (0.00665)^10 approx 1.7 times 10^(-22). $
   #notebox[Sigmoid remains appropriate for a binary output; repeated saturated hidden gates are the issue.]],
  r: (1.1fr, 0.9fr),
)

== Optional · The PyTorch initialization line

#codebox(size: 13pt)[
```python
linears = [m for m in model.modules()
           if isinstance(m, nn.Linear)]

for m in linears[:-1]:          # hidden layers
    nn.init.kaiming_normal_(
        m.weight,
        mode="fan_in",
        nonlinearity="relu",
    )
    if m.bias is not None:
        nn.init.zeros_(m.bias)

with torch.no_grad():            # final classifier
    nn.init.xavier_uniform_(linears[-1].weight)
    if linears[-1].bias is not None:
        nn.init.zeros_(linears[-1].bias)
```
]

#align(center, text(size: 18pt, weight: 650)[Use ReLU-aware Kaiming for hidden layers; initialize the final classifier separately.])

== Optional diagnosis · RMS is usable, but gates close

#align(center, codebox(size: 14pt)[
```text
layer:         1        10        20        30
act RMS:       0.8       0.7       0.6       0.5
active:       0.49      0.31      0.08      0.01
```
])

#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 22pt,
  [*Symptom* \
   ReLU gates progressively close.],
  [*First suspect* \
   Preactivations drifted negative.],
  [*Smallest test* \
   Plot preactivation histograms and means.],
))

== Optional diagnosis · First non-finite activation at layer 17

#align(center, codebox(size: 14pt)[
```text
layer 15: finite=True   max|h|=8.2e+14
layer 16: finite=True   max|h|=3.7e+28
layer 17: finite=False  max|h|=inf
```
])

#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 22pt,
  [*Symptom* \
   Numerical overflow begins inside the model.],
  [*First suspect* \
   Excess forward gain before layer 17.],
  [*Smallest test* \
   Inspect layers 15–17 before changing Adam.],
))
