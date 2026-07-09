# dlviz — a native Typst plotting toolkit for ML/DL teaching

Design notes + spec for the figure system behind the ES 667 Deep Learning decks.

## 1. How figures are made today

| Kind | Tool now | Native? |
|------|----------|---------|
| Network / neuron / computation graphs | **fletcher** (in `common/metropolis.typ`) | ✅ yes — vector, recolors with palette |
| 2-D plots (curves, contours, GD paths, scatter, bars) | **matplotlib → SVG+PNG**, embedded via `fig()` | ❌ no — raster/vector export |
| 3-D surfaces (bowl, saddle) | **matplotlib mplot3d → PNG** | ❌ no |

Two known pain points with the matplotlib route:

- **resvg mangles some matplotlib path-text SVGs** (glyph advances collapse), so the Typst decks read the **PNG** twin. That is why `save()` in every `diagrams/*.py` emits both, and `fig()` swaps `.svg → .png`.
- Figures live **outside the deck** — you edit Python, re-run, re-embed. No single source of truth, and colors/fonts can drift from the deck theme.

## 2. Does Typst have a pgfplots/TikZ analogue? Yes.

| LaTeX | Typst | Status | Notes |
|-------|-------|--------|-------|
| TikZ | **CeTZ** | mature | general vector drawing; everything below builds on it |
| pgfplots | **cetz-plot** | maturing | axes, line/scatter/bar, some contour |
| matplotlib | **lilaq** (`@preview/lilaq`) | active, good | the most matplotlib-like: `plot`, `scatter`, `bar`, **`contour`**, `linspace`, legends, labels, colormaps |
| TikZ graphs | **fletcher** | mature | node–link diagrams (already used here) |

**Verified working** (this repo, Typst 0.14): lilaq renders clean native line plots, activation curves, and **contour plots**, with IBM Plex fonts and our palette — see `common/dlviz.typ` and the proof-of-concept helpers below.

**The one real gap: 3-D surfaces.** lilaq/cetz-plot are 2-D. True shaded 3-D surfaces (the `x²+y²` bowl, the saddle) are not practical natively yet. Keep those in matplotlib.

## 3. Policy (hybrid)

- **Native (dlviz / lilaq / fletcher)** — everything 2-D and all diagrams: activation curves, loss contours + GD/optimizer trajectories, decision boundaries, scatter, bar charts, schedule curves, networks, computation graphs. Benefits: vector, palette- and font-consistent, editable *inside the deck*, no PNG/resvg dance, trajectories computable in Typst.
- **matplotlib → PNG** — only 3-D surfaces. Isolated, few, and clearly labelled.

## 4. `dlviz` — the library

A thin, opinionated layer over lilaq + fletcher that bakes in the metropolis palette and the *recurring* ML/DL figure vocabulary, so a slide is a one-liner.

### Already implemented (`common/dlviz.typ`, proof-of-concept)

```typst
#import "../common/dlviz.typ": *

#activation-plot("relu")                       // sigmoid | tanh | relu | gelu, value + derivative
#curve-plot(x => x*x, xrange: (-3, 3))         // any y = f(x)
#loss-contour((x, y) => x*x + 6*y*y,           // contour + optional descent path
  path: gd-path((x, y) => (2*x, 12*y), start: (2.4, 2.2), eta: 0.15, steps: 16))
#scatter-2class(pts-a, pts-b)                  // two-class scatter (+ boundary line, planned)
```

`gd-path(grad, start, eta, steps)` computes a trajectory in Typst, so the *same* function that appears in the slide's math drives the picture.

### Planned API (the ML/DL figure catalog)

```typst
// optimization
optimizer-race(f, grad, {gd, momentum, adam}, start)   // several trajectories, one contour
lr-schedule("cosine" | "step" | "warmup-cosine", ...)  // schedule curves
surface2d(f, ...)                                       // 2-D heatmap alt. to 3-D (native)

// probability / losses (L1)
gaussian(mu, sigma) · density-area(a, b) · softmax-bars(logits) · neglog-curve

// models (L2)
decision-regions(scorefns)      // filled polyhedral/ nonlinear regions
sigmoid-heatmap(w, b)           // probability field + boundary
relu-sum-fit(target, m)         // least-squares ReLU approximation (native)

// backprop (L3)  — thin wrappers over fletcher
comp-graph(nodes, edges) · network(sizes) · neuron(d)   // + forward/backward value badges

// calculus (L3A)
tangent(f, x0, dx) · gradient-field(f) · quiver(...)
```

### Palette & theming
- Import the six palette colors from `metropolis.typ` (single source of truth).
- Default a **teal** single-hue contour ramp (override lilaq's viridis), **orange** trajectories, **ink** primary curves, **dashed orange** derivatives, **muted** grid — matching the decks.
- Inherit IBM Plex via the deck's text settings; expose `width`/`height` with sane 16:9-friendly defaults.

### Roadmap
1. **v0 (done)** — `activation-plot`, `curve-plot`, `loss-contour`, `gd-path`, `scatter-2class`; palette import.
2. **v1** — teal contour ramp + themed legend/grid defaults; `optimizer-race`, `lr-schedule`, `decision-regions`; port L1/L2/L3A 2-D figures off matplotlib.
3. **v2** — forward/backward value badges on fletcher `comp-graph`/`network`; `relu-sum-fit`; retire matplotlib except 3-D.
4. **v3** — extract to a standalone Typst package (`@preview/dlviz`) with docs + a gallery, reusable across `ml-teaching` / `pml-teaching`.

### Non-goals
- 3-D shaded surfaces (stay matplotlib).
- Interactive figures (that's the `interactive-articles` site, linked from `[I]` slides).

## 5. Why this is worth it
One import, palette-locked, vector, and the picture is driven by the *same* function/gradient written in the slide — the figure can't silently disagree with the math. It also generalizes to the other teaching repos.
