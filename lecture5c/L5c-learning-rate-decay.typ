// Learning Rate Decay - Lecture 5c - ES 667 Deep Learning
// Compact, intuition-first version.
// Compile from the repository root:
//   typst compile --root . --input handout=true lecture5c/L5c-learning-rate-decay.typ slides-pdf/L5C.pdf
//   typst compile --root . lecture5c/L5c-learning-rate-decay.typ slides-pdf/L5C-presentation.pdf

#import "../common/metropolis.typ": *
#import "../common/mldiag.typ": *
#import "@preview/cetz:0.5.2" as cetz

#show: metropolis-deck.with(
  title: [Learning Rate Decay],
  subtitle: [Should epoch 1 and epoch 100 use the same learning rate? · Lecture 5c],
)

#let note(body, color: TEAL) = block(
  width: 100%, inset: (x: 10pt, y: 7pt), radius: 4pt,
  fill: color.lighten(94%), stroke: (left: 1.3pt + color),
  align(left, body),
)
#let caption(body) = align(center, text(size: 12.5pt, fill: MUTED, body))
#let label(body, color) = box(
  inset: (x: 7pt, y: 4pt), radius: 3pt,
  fill: color.lighten(92%), stroke: 0.8pt + color,
  text(size: 11.5pt, weight: 680, fill: color, body),
)
#let swatch(color, body) = box(width: 12pt, height: 3pt, fill: color, radius: 1pt) + h(4pt) + body
#let result-small(body) = align(center, block(
  inset: (x: 13pt, y: 8pt), radius: 4pt,
  fill: white, stroke: 1.6pt + ACC,
  text(size: 17.5pt, weight: 650, fill: INK, body),
))

// One landscape and one seeded minibatch sequence are reused throughout.
#let bowl = ad.expr("0.5*x^2 + 3*y^2", ("x", "y"))
#let bowl-grad = ad.grad-fn(bowl)
#let start = (-2.15, 0.78)
#let xlim = (-2.5, 2.5)
#let ylim = (-1.05, 1.05)
#let constant-rate = _ => 0.20
#let small-rate = _ => 0.04
#let decay-rate = t => 0.04 + 0.16 * 0.5 * (1 + calc.cos(calc.pi * t / 36))

#let noisy-path(rate, seed: 73, steps: 36, sigma: 0.48) = {
  let p = start
  let out = (p,)
  for t in range(steps) {
    let g = bowl-grad(p)
    let gx = g.at(0) + sigma * rnd.randn(seed, 2 * t)
    let gy = g.at(1) + sigma * rnd.randn(seed, 2 * t + 1)
    let eta = rate(t)
    p = (p.at(0) - eta * gx, p.at(1) - eta * gy)
    out.push(p)
  }
  out
}

#let path-large = noisy-path(constant-rate)
#let path-small = noisy-path(small-rate)
#let path-decay = noisy-path(decay-rate)

#let path-overlay(
  path,
  size,
  color,
  early-until: none,
  early-color: TEAL,
  arrow-every: 2,
) = {
  let (w, hgt) = size
  let sx(x) = (x - xlim.at(0)) / (xlim.at(1) - xlim.at(0)) * w
  let sy(y) = (y - ylim.at(0)) / (ylim.at(1) - ylim.at(0)) * hgt
  let screen(p) = (sx(p.at(0)), sy(p.at(1)))
  cetz.canvas({
    import cetz.draw: line, circle, rect, content
    // Fix the canvas bounds so this overlay registers exactly with the contour.
    rect((0, 0), (w, hgt), fill: white.transparentize(100%), stroke: none)
    for i in range(path.len() - 1) {
      let col = if early-until != none and i < early-until { early-color } else { color }
      let arrow = if calc.rem(i, arrow-every) == 0 { (end: ">", scale: 0.48) } else { none }
      line(screen(path.at(i)), screen(path.at(i + 1)), stroke: 1.7pt + col, mark: arrow)
    }
    let first = screen(path.first())
    let minimum = screen((0, 0))
    circle(first, radius: 2.8pt, fill: BLUE, stroke: 0.7pt + white)
    content((first.at(0) + 4mm, first.at(1) - 2mm), text(size: 8.5pt, weight: 650, fill: BLUE, [start]), anchor: "west")
    circle(minimum, radius: 2.5pt, fill: RED, stroke: 0.7pt + white)
    content((minimum.at(0) + 4mm, minimum.at(1)), text(size: 8.5pt, weight: 650, fill: RED, [minimum]), anchor: "west")
    circle(screen(path.last()), radius: 2.8pt, fill: color, stroke: 0.7pt + white)
  })
}

#let trajectory(
  path,
  color,
  size: (210mm, 80mm),
  early-until: none,
  early-color: TEAL,
  title: none,
) = {
  let (w, hgt) = size
  block(width: w, height: hgt, [
    #place(top + left, contour(
      ad.fn2(bowl), xlim: xlim, ylim: ylim, samples: 58, levels: 10,
      size: size, color: MUTED.lighten(22%),
    ))
    #place(top + left, path-overlay(
      path, size, color,
      early-until: early-until,
      early-color: early-color,
    ))
    #if title != none {
      place(top + center, dy: 3mm, text(size: 12.5pt, weight: 720, fill: color, title))
    }
  ])
}

#let gradient-panel(kind: "far", size: (112mm, 54mm)) = {
  let (w, hgt) = size
  let sx(x) = (x - xlim.at(0)) / (xlim.at(1) - xlim.at(0)) * w
  let sy(y) = (y - ylim.at(0)) / (ylim.at(1) - ylim.at(0)) * hgt
  let screen(p) = (sx(p.at(0)), sy(p.at(1)))
  // Keep every arrow inside the plot bounds so the overlay stays registered.
  let p = if kind == "far" { (1.25, 0.22) } else { (0.08, 0.02) }
  let full = if kind == "far" { (0.36, 0.62) } else { (0.05, 0.05) }
  let batches = if kind == "far" {
    ((0.25, 0.54), (0.42, 0.58), (0.31, 0.72), (0.48, 0.68), (0.22, 0.66))
  } else {
    ((0.31, 0.12), (-0.24, 0.25), (0.12, -0.31), (-0.29, -0.08), (0.27, -0.18))
  }
  let selected = batches.first()
  block(width: w, height: hgt, [
    #place(top + left, contour(
      ad.fn2(bowl), xlim: xlim, ylim: ylim, samples: 52, levels: 9,
      size: size, color: MUTED.lighten(27%),
    ))
    #place(top + left, cetz.canvas({
      import cetz.draw: line, circle, rect, content
      // Fix the canvas bounds so the arrows and contours share coordinates.
      rect((0, 0), (w, hgt), fill: white.transparentize(100%), stroke: none)
      for v in batches {
        let q = (p.at(0) + v.at(0), p.at(1) + v.at(1))
        line(screen(p), screen(q), stroke: 1.15pt + BLUE.transparentize(12%), mark: (end: ">", scale: 0.48))
      }
      let q-full = (p.at(0) + full.at(0), p.at(1) + full.at(1))
      line(screen(p), screen(q-full), stroke: 2.6pt + TEAL, mark: (end: ">", scale: 0.62))
      let q-update = (p.at(0) - 0.72 * selected.at(0), p.at(1) - 0.72 * selected.at(1))
      line(screen(p), screen(q-update), stroke: 2.4pt + ACC, mark: (end: ">", scale: 0.62))
      circle(screen((0, 0)), radius: 2.4pt, fill: RED, stroke: 0.7pt + white)
      circle(screen(p), radius: 3pt, fill: INK, stroke: 0.8pt + white)
      content((w / 2, hgt - 3mm), text(
        size: 12pt, weight: 720, fill: if kind == "far" { TEAL } else { ACC },
        if kind == "far" { [far from the minimum] } else { [near the minimum] },
      ), anchor: "north")
    }))
  ])
}

#let step-shape = t => if t < .30 { 1.0 } else if t < .62 { .55 } else if t < .84 { .28 } else { .12 }
#let exp-shape = t => calc.pow(.08, t)
#let inverse-shape = t => 1 / (1 + 5 * t)
#let cosine-shape = t => .05 + .95 * .5 * (1 + calc.cos(calc.pi * t))
#let warm-frac = .12
#let warmup-decay-shape = t => if t < warm-frac {
  t / warm-frac
} else {
  .05 + .95 * .5 * (1 + calc.cos(calc.pi * (t - warm-frac) / (1 - warm-frac)))
}

#let schedule-spark(fn, color, size: (47mm, 20mm)) = {
  let (w, hgt) = size
  let pts = range(61).map(i => {
    let t = i / 60
    (t * w, fn(t) * hgt)
  })
  cetz.canvas({
    import cetz.draw: line, rect, circle
    rect((0, 0), (w, hgt), fill: white.transparentize(100%), stroke: none)
    line((0, 0), (w, 0), stroke: .7pt + MUTED.lighten(35%))
    line((0, 0), (0, hgt), stroke: .7pt + MUTED.lighten(35%))
    for i in range(pts.len() - 1) {
      line(pts.at(i), pts.at(i + 1), stroke: 1.8pt + color)
    }
    circle(pts.first(), radius: 1.8pt, fill: color, stroke: none)
    circle(pts.last(), radius: 1.8pt, fill: color, stroke: none)
  })
}

#let rule-card(name, formula, cue, color, fn) = block(
  width: 100%, height: 49mm, inset: (x: 9pt, y: 7pt), radius: 4pt,
  fill: color.lighten(96%), stroke: .8pt + color.lighten(28%),
  grid(
    columns: (1.22fr, .78fr), gutter: 8pt, align: horizon,
    [
      #text(size: 16pt, weight: 730, fill: color)[#name]
      #v(3pt)
      #text(size: 15pt)[#formula]
      #v(4pt)
      #text(size: 12.5pt, fill: INK)[*Use it when:* #cue]
    ],
    align(center, schedule-spark(fn, color)),
  ),
)

#let warmup-decay-plot(size: (157mm, 58mm)) = {
  let (w, hgt) = size
  let pts = range(101).map(i => {
    let t = i / 100
    (t * w, warmup-decay-shape(t) * hgt)
  })
  let wx = warm-frac * w
  cetz.canvas({
    import cetz.draw: line, rect, circle, content
    rect((0, 0), (wx, hgt), fill: TEAL.lighten(94%), stroke: none)
    rect((wx, 0), (w, hgt), fill: ACC.lighten(97%), stroke: none)
    for q in (.25, .5, .75) {
      line((0, q * hgt), (w, q * hgt), stroke: .55pt + MUTED.lighten(45%))
    }
    line((0, 0), (w, 0), stroke: 1pt + INK)
    line((0, 0), (0, hgt), stroke: 1pt + INK)
    line((wx, 0), (wx, hgt), stroke: .9pt + TEAL)
    for i in range(pts.len() - 1) {
      let col = if i / 100 < warm-frac { TEAL } else { ACC }
      line(pts.at(i), pts.at(i + 1), stroke: 2.5pt + col)
    }
    circle((wx, hgt), radius: 2.8pt, fill: INK, stroke: .8pt + white)
    content((wx / 2, hgt - 5mm), text(size: 11pt, weight: 720, fill: TEAL, [WARMUP]), anchor: "center")
    content((wx + (w - wx) / 2, hgt - 5mm), text(size: 11pt, weight: 720, fill: ACC, [DECAY]), anchor: "center")
    content((wx, 3mm), text(size: 10pt, weight: 650, fill: TEAL, [$W$]), anchor: "south")
    content((w - 2mm, 3mm), text(size: 10pt, weight: 650, fill: INK, [$T$]), anchor: "south-east")
  })
}

#title-slide()

== What changes as SGD gets closer? #Q

#align(center, trajectory(
  path-large, ACC,
  early-until: 9,
  early-color: TEAL,
  title: [fixed learning rate $eta=.20$],
))
#v(5pt)
#align(center, grid(
  columns: (auto, auto), gutter: 24pt,
  label([early updates: fast progress], TEAL),
  label([later updates: motion around the minimum], ACC),
))

#pause
#v(5pt)
#result-small[SGD has reached the right region. The same step size now makes it bounce around.]

== A minibatch gives a noisy gradient #V

#align(center, grid(
  columns: (1fr, 1fr), gutter: 10pt,
  gradient-panel(kind: "far"),
  gradient-panel(kind: "near"),
))
#v(2pt)
#align(center, grid(
  columns: (auto, auto, auto), gutter: 16pt, align: horizon,
  swatch(TEAL, [full-data gradient]),
  swatch(BLUE, [minibatch gradients]),
  swatch(ACC, [chosen update]),
))

#pause
#v(3pt)
#align(center, text(size: 18pt, weight: 650)[
  $g_B(theta_t)=nabla L(theta_t)+epsilon_B$ #h(18pt)
  $theta_(t+1)=theta_t-eta_t g_B(theta_t)$
])
#v(3pt)
#align(center, grid(
  columns: (1fr, 1fr), gutter: 16pt,
  note(text(size: 15.5pt)[Far away, the gradient signal is large compared with the minibatch noise.], color: TEAL),
  note(text(size: 15.5pt)[Near the minimum, the gradient signal can be smaller than the minibatch noise.], color: ACC),
))

== The learning rate sets the size of the bounce #Q

#align(center, grid(
  columns: (1fr, 1fr, 1fr), gutter: 7pt,
  [#align(center, trajectory(path-large, RED, size: (80mm, 58mm), title: [large and constant]))
   #v(3pt)
   #caption[$eta=.20$: arrives quickly, then jitters]],
  [#align(center, trajectory(path-small, BLUE, size: (80mm, 58mm), title: [small and constant]))
   #v(3pt)
   #caption[$eta=.04$: quieter, but slower]],
  [#align(center, trajectory(path-decay, GREEN, size: (80mm, 58mm), title: [large, then smaller]))
   #v(3pt)
   #caption[$.20 arrow.r .04$: fast start, quiet finish]],
))
#v(7pt)
#align(center, text(size: 18pt)[For the same noise term $abs(epsilon_B)=.5$:])
#v(3pt)
#align(center, grid(
  columns: (auto, auto), gutter: 18pt,
  label([$eta=.20 arrow.r eta abs(epsilon_B)=.10$], RED),
  label([$eta=.04 arrow.r eta abs(epsilon_B)=.02$], GREEN),
))
#v(6pt)
#result-small[A schedule keeps the useful large steps and reduces the late noise-driven moves.]

== Four useful decay rules #V

#grid(
  columns: (1fr, 1fr), gutter: 8pt, row-gutter: 7pt,
  rule-card(
    [Step decay],
    [$eta_t=eta_0 gamma^(floor(t/K))$],
    [you know the milestones and want drops that are easy to see.],
    BLUE, step-shape,
  ),
  rule-card(
    [Exponential decay],
    [$eta_t=eta_0 exp(-k t)$],
    [you want a steady percentage decrease with no abrupt jumps.],
    MUTED, exp-shape,
  ),
  rule-card(
    [Inverse-time decay],
    [$eta_t=eta_0/(1+k t)$],
    [you want an early reduction followed by a long, gentle tail.],
    GREEN, inverse-shape,
  ),
  rule-card(
    [Cosine decay #text(size: 11.5pt, weight: 500, fill: MUTED)[($0 <= t <= T$)]],
    [$eta_t=eta_"min" + (eta_0-eta_"min")/2 (1+cos(pi t/T))$],
    [training ends at $T$.],
    ACC, cosine-shape,
  ),
)
#v(5pt)
#note([*Feedback rule: reduce on plateau.* Use it when validation progress, not the clock, should trigger a drop. Patience prevents one noisy check from changing $eta$.], color: TEAL)

== PyTorch: two scheduler patterns #I

#codebox(size: 11.5pt)[```python
from torch.optim import lr_scheduler as lrs
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
```]
#v(5pt)

#grid(
  columns: (1fr, 1fr), gutter: 14pt,
  [
    #text(size: 15.5pt, weight: 720, fill: BLUE)[FIXED CLOCK: COSINE]
    #v(3pt)
    #codebox(size: 10.5pt)[```python
cosine = lrs.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs,
    eta_min=1e-4,
)

for epoch in range(num_epochs):
    train_one_epoch(...)
    cosine.step()  # after optimizer updates
```]
    #v(4pt)
    #note(text(size: 13pt)[*Clock:* `T_max` counts calls to `cosine.step()`. Here, one call means one epoch.], color: BLUE)
  ],
  [
    #text(size: 15.5pt, weight: 720, fill: TEAL)[VALIDATION FEEDBACK: PLATEAU]
    #v(3pt)
    #codebox(size: 10.5pt)[```python
plateau = lrs.ReduceLROnPlateau(
    optimizer, mode="min",
    factor=0.1, patience=3,
)

for epoch in range(num_epochs):
    train_one_epoch(...)
    val_loss = validate(...)
    plateau.step(val_loss)
```]
    #v(4pt)
    #note(text(size: 13pt)[*Metric:* `factor=.1` makes each drop 10x smaller. `patience=3` waits before reacting to a short stall.], color: TEAL)
  ],
)
#v(5pt)
#result-small[The call tells you what drives the schedule: `step()` uses time; `step(val_loss)` uses validation.]

== PyTorch: one epoch, step by step #I

#two(r: (1.30fr, .70fr),
  [
    #codebox(size: 9.8pt)[```python
from torch.optim import lr_scheduler as lrs

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
scheduler = lrs.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs,
    eta_min=1e-4,
)

for epoch in range(num_epochs):
    model.train()
    for x, y in train_loader:
        optimizer.zero_grad()          # 1
        y_hat = model(x)               # 2
        loss = loss_fn(y_hat, y)       # 3
        loss.backward()                # 4
        optimizer.step()               # 5

    scheduler.step()                   # 6
```]
    #v(5pt)
    #note(text(size: 12.2pt)[
      One call per epoch means `T_max=num_epochs`. If you call the scheduler every batch, its clock must count batches instead.
      \
      *Timing:* `optimizer.step()` moves the model; `scheduler.step()` sets the rate for future moves.
    ], color: BLUE)
  ],
  [
    #text(size: 14pt, weight: 720, fill: MUTED)[WHAT EACH NUMBER DOES]
    #v(3pt)
    #grid(
      columns: (auto, 1fr), gutter: 5pt, row-gutter: 4pt,
      label([1], BLUE), text(size: 11.8pt)[*Clear old gradients.* PyTorch adds gradients unless we reset them.],
      label([2], BLUE), text(size: 11.8pt)[*Predict.* Run the minibatch through the model.],
      label([3], BLUE), text(size: 11.8pt)[*Measure the error.* Compare the prediction with the target.],
      label([4], BLUE), text(size: 11.8pt)[*Backpropagate.* Fill each parameter's `.grad`.],
      label([5], ACC), text(size: 11.8pt)[*Move the weights.* Use the current learning rate.],
      label([6], TEAL), text(size: 11.8pt)[*Advance the schedule.* Set the rate for the next epoch.],
    )
    #v(6pt)
    #note(text(size: 11.8pt)[
      *If validation should decide:* keep steps 1-5. After validation, use
      `val_loss = evaluate(model, val_loader)` and then
      `plateau.step(val_loss)`.
    ], color: TEAL)
  ],
)

== Warm up, then decay #V

#two(r: (1.28fr, .72fr),
  [
    #align(center, warmup-decay-plot())
    #v(1pt)
    #caption[Example: warm up for the first 12% of updates, then use cosine decay to the finish.]
    #v(5pt)
    #note(text(size: 15pt)[
      *Why it helps:* Early gradients can be unusually large. Since $abs(Delta theta_t)=eta_t abs(g_t)$, a small $eta_t$ keeps the first parameter changes small and reduces the risk of a loss spike or divergence.
    ], color: TEAL)
  ],
  [
    #note(text(size: 15pt)[
      *1. Start small* \
      $eta_t=eta_"peak" t/W$
      \
      Raise the rate gradually over the first $W$ updates.
    ], color: TEAL)
    #v(4pt)
    #note(text(size: 15pt)[
      *2. Then decay* \
      $eta_t arrow.r eta_"min"$
      \
      Once the early updates are stable, shrink the rate using the chosen rule.
    ], color: ACC)
    #v(4pt)
    #text(size: 13.5pt)[*Look for this:* the loss spikes or becomes NaN when training starts at the full learning rate.]
  ],
)

== What should you remember? #Q

#align(center, grid(
  columns: (1fr, 1fr), gutter: 12pt,
  note([
    *Far from the minimum* \
    The gradient signal is often strong. A large learning rate can make fast progress.
  ], color: TEAL),
  note([
    *Near the minimum* \
    Minibatch noise can dominate. A smaller learning rate reduces the resulting motion.
  ], color: ACC),
  note([
    *Fixed training budget* \
    Use a time-based schedule such as cosine decay.
  ], color: BLUE),
  note([
    *Validation should decide* \
    Use reduce on plateau after improvement stalls.
  ], color: GREEN),
))

#v(10pt)
#align(center, text(size: 20pt, weight: 650)[
  $g_B(theta_t)=nabla L(theta_t)+epsilon_B$
  #h(18pt)
  $Delta theta_t=-eta_t g_B(theta_t)$
])

#v(10pt)
#result-small[Large steps help early. Smaller steps give SGD more control near the minimum.]
#v(7pt)
#caption[Choose the simplest rule that matches the run. Add warmup only when startup is fragile.]

== Sources and evidence

#set text(size: 13.5pt)

- #link("https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/resources/lecture-25-stochastic-gradient-descent/")[MIT OCW · Lecture 25: Stochastic Gradient Descent]: fast early progress, followed by bouncing near the solution.
- #link("https://fa.bianp.net/teaching/2018/COMP-652/stochastic_gradient.html")[Fabian Pedregosa · The Stochastic Gradient Method]: interactive 2D paths for constant and decreasing step sizes.
- #link("https://fa.bianp.net/teaching/2018/COMP-652/gradient_descent.html")[Fabian Pedregosa · Gradient Descent]: interactive contour plots showing sensitivity to step size.
- #link("https://www.youtube.com/watch?v=QzulmoOg2JE")[Andrew Ng · Learning Rate Decay]: one contour picture and one numerical decay example.
- #link("https://docs.pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate")[PyTorch · Learning-rate schedulers]: scheduler call order and metric-based decay.
