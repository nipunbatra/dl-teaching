// ═══════════════════════════════════════════════════════════════════
//  dlviz — native Typst plotting helpers for ML/DL teaching figures
//  Palette-consistent with common/metropolis.typ. Built on lilaq (2-D
//  plots) + fletcher (diagrams, via metropolis.typ). 3-D surfaces are NOT
//  native — keep those in matplotlib (see DLVIZ.md).
//
//  Usage:  #import "../common/dlviz.typ": *
//          #activation-plot("relu")
//          #loss-contour((x,y) => x*x + 6*y*y, path: gd-path((x,y)=>(2*x,12*y), (2.4,2.2), 0.15, 16))
// ═══════════════════════════════════════════════════════════════════

#import "@preview/lilaq:0.4.0" as lq
#import "metropolis.typ": INK, ACC, TEAL, GREEN, BLUE, MUTED, RED

// ── scalar activations (value + derivative) ──
#let _sig(z) = 1 / (1 + calc.exp(-z))
#let _tanh(z) = { let e = calc.exp(2 * z); (e - 1) / (e + 1) }
#let ACTIVATIONS = (
  sigmoid: (f: z => _sig(z),  d: z => _sig(z) * (1 - _sig(z)),        label: $sigma$),
  tanh:    (f: z => _tanh(z), d: z => 1 - _tanh(z) * _tanh(z),        label: [tanh]),
  relu:    (f: z => calc.max(0.0, z), d: z => if z > 0 { 1.0 } else { 0.0 }, label: [ReLU]),
  gelu:    (f: z => z * _sig(1.702 * z), d: none,                     label: [GELU]),
)

// themed lilaq diagram (thin wrapper so every figure inherits sizing defaults)
#let dl-plot(..traces, width: 6cm, height: 4cm, xlabel: none, ylabel: none) = lq.diagram(
  width: width, height: height, xlabel: xlabel, ylabel: ylabel, ..traces,
)

// one activation curve, optionally with its derivative (dashed orange)
#let activation-plot(kind, deriv: true, width: 6cm, height: 4cm) = {
  let a = ACTIVATIONS.at(kind)
  let xs = lq.linspace(-4, 4, num: 220)
  dl-plot(width: width, height: height, xlabel: $z$,
    lq.plot(xs, xs.map(a.f), stroke: INK + 2pt, mark: none, label: a.label),
    ..if deriv and a.d != none { (lq.plot(xs, xs.map(a.d),
        stroke: (paint: ACC, dash: "dashed", thickness: 1.6pt), mark: none, label: $#a.label'$),) } else { () },
  )
}

// compute a gradient-descent trajectory in Typst; grad returns a 2-tuple
#let gd-path(grad, start, eta, steps) = {
  let p = start; let out = (p,)
  for _ in range(steps) {
    let g = grad(p.at(0), p.at(1))
    p = (p.at(0) - eta * g.at(0), p.at(1) - eta * g.at(1))
    out.push(p)
  }
  out
}

// contour of a scalar field f(x,y), optional descent path overlaid
#let loss-contour(f, xrange: (-3, 3), yrange: (-3, 3), levels: 10, path: none,
                  width: 6cm, height: 5cm) = {
  let gx = lq.linspace(..xrange, num: 80)
  let gy = lq.linspace(..yrange, num: 80)
  dl-plot(width: width, height: height,
    lq.contour(gx, gy, f, fill: false),
    ..if path != none { (lq.plot(path.map(p => p.at(0)), path.map(p => p.at(1)),
        stroke: ACC + 1.7pt, mark: "o", mark-size: 3pt, color: ACC),) } else { () },
  )
}

// a plain function curve y = f(x)
#let curve-plot(f, xrange: (-4, 4), color: TEAL, width: 6cm, height: 4cm, xlabel: none, ylabel: none) = {
  let xs = lq.linspace(..xrange, num: 220)
  dl-plot(width: width, height: height, xlabel: xlabel, ylabel: ylabel,
    lq.plot(xs, xs.map(f), stroke: color + 2.2pt, mark: none))
}

// 2-class scatter with an optional decision line  w0*x + w1*y + b = 0
#let scatter-2class(class-a, class-b, line: none, width: 6cm, height: 5cm) = {
  dl-plot(width: width, height: height,
    lq.scatter(class-a.map(p => p.at(0)), class-a.map(p => p.at(1)), color: TEAL, size: 4pt),
    lq.scatter(class-b.map(p => p.at(0)), class-b.map(p => p.at(1)), color: ACC, size: 4pt),
  )
}
