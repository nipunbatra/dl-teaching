// tensor-grid demo deck — native Typst replacements for the L8/L15
// matplotlib figures, instantiated through common/mldiag.typ.
// Compile from the repo root:
//   typst compile --root . tensor-grid-demo.typ /tmp/tg-demo.pdf

#import "common/metropolis.typ": *
#import "common/mldiag.typ": *

#show: metropolis-deck.with(
  title: [tensor-grid — native course figures],
  subtitle: [conv arithmetic · pooling · attention · receptive fields, all in Typst],
)
#title-slide()

// the SAME data the slide math talks about drives every figure
#let X = ((1, 2, 0, 1, 2),
          (0, 1, 3, 2, 0),
          (2, 1, 0, 1, 3),
          (1, 0, 2, 0, 1),
          (0, 2, 1, 3, 2))
#let K = ((1, 1, 1), (0, 0, 0), (-1, -1, -1))   // horizontal-edge detector

// ── animated: the kernel slides across all 9 positions ──
#slide(title: [Convolution — the kernel slides (native, animated)], repeat: 9, self => [
  #v(6mm)
  #align(center, conv-op(input: X, kernel: K, step: self.subslide - 1,
    show-expr: true, cell: 8.5mm))
  #v(4mm)
  #align(center, text(size: 14pt, fill: MUTED)[
    one `step:` parameter → touying subslides animate the window; the output
    values are computed *in Typst* from the same $X$ and $K$ shown above
  ])
])

== Padding and stride — output size is computed, not asserted

#two(
  [
    #align(center, conv-op(input: 5, kernel: 3, padding: 1, stride: 2, step: 1, cell: 7mm,
      labels: (input: [padded input ($p = 1$)], kernel: [kernel $K$], output: [output])))
  ],
  [
    #v(4mm)
    $ H_"out" = floor((H + 2p - k) / s) + 1 $
    #v(3mm)
    #text(size: 15pt)[
      with $H = 5, k = 3, p = 1, s = 2$: \
      $H_"out" = #conv-out-size(5, 3, stride: 2, padding: 1)$ —
      the output grid on the left *is* this formula
    ]
  ],
  r: (3fr, 2fr),
)

== Dilation — same 9 weights, larger footprint

#align(center, conv-op(input: 7, kernel: 3, dilation: 2, step: 0, cell: 7mm))
#v(3mm)
#align(center, text(size: 14pt, fill: MUTED)[
  receptive footprint $= d(k - 1) + 1 = 5$, still only $9$ parameters
])

== Pooling — max and average, one input

#align(center, pool-op(((1, 3, 2, 4), (5, 6, 1, 2), (7, 2, 3, 0), (1, 0, 4, 8)),
  window: 2, kinds: ("max", "avg"), cell: 9mm))

== Self-attention scores — worked 3×3 example

#two(
  align(center, attn-matrix(($x_1$, $x_2$, $x_3$),
    values: ((1, 0, 1), (0, 1, 1), (1, 1, 2)), cell: 11mm)),
  align(center, attn-matrix(("1", "2", "3", "4", "5"), mask: "causal", cell: 9mm,
    q-label: [query position $i$], k-label: [key position $j$])),
)

== Attention weights — rows softmaxed in Typst

#align(center, attn-matrix(("the", "cat", "sat", "on", "mat"),
  values: ((3, 0, 0, 0, 0),
           (1.8, 1.4, 0, 0, 0),
           (0.2, 2.0, 1.4, 0, 0),
           (0.1, 0.4, 1.5, 1.4, 0),
           (0.3, 0.2, 0.5, 1.6, 1.4)),
  mask: "causal", softmax: true, min-annotate: 0.05, colorbar: true, cell: 10mm))

== Receptive field growth — computed from the conv stack

#two(
  align(center, receptive-field(kernels: (3, 3, 3), cell: 8mm)),
  align(center + horizon, patchify(image: 8, patch: 4, cell: 5.5mm, gap: 2.5mm)),
)

== Before / after — matplotlib PNG vs native

#two(
  [
    #align(center, image("lecture8/figures/conv_op.png", width: 100%))
    #align(center, text(size: 13pt, fill: MUTED)[mpl → SVG+PNG twins → resvg dance])
  ],
  [
    #align(center, conv-op(input: X, kernel: K, step: 0, cell: 6.5mm))
    #align(center, text(size: 13pt, fill: MUTED)[native: vector, palette-locked, animatable])
  ],
)
