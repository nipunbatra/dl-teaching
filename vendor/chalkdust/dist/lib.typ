// dist — standard probability distributions with EXACT pdf / log-pdf.
//
// The point: a teaching loss curve should be the true negative log-likelihood of
// a properly-parameterised distribution, not a coefficient tuned to look right.
// Each constructor returns a dict carrying its (log-)density as a closure, so it
// drops straight into plot:
//
//   #import "vendor/chalkdust/dist/lib.typ" as dist
//   #import "vendor/chalkdust/plot/lib.typ": lines
//   #let g = dist.normal(sigma: 1.0)
//   #lines(fn: x => g.pdf(x), domain: (-4, 4))          // the density
//   #lines(fn: x => dist.nll0(g, x), domain: (-3, 3))   // the loss (NLL, min at 0)

#let _ln2pi = calc.ln(2 * calc.pi)
#let _NEG-INF = -1e18                       // stands in for −∞ (outside a support)
#let _is-num(v) = type(v) == int or type(v) == float
#let _positive(v, name) = {
  assert(_is-num(v) and v > 0, message: "dist." + name + ": parameter must be positive")
  v * 1.0
}
#let _prob(v, name: "probability") = {
  assert(_is-num(v) and v >= 0 and v <= 1,
    message: "dist." + name + ": probability must be in [0, 1]")
  v * 1.0
}
#let _log-prob(v) = if v == 0 { _NEG-INF } else { calc.ln(v) }

// Lanczos approximation to log Γ(x) — the normaliser for the Student-t density.
#let _lgamma(x) = {
  let g = 7
  let c = (0.99999999999980993, 676.5203681218851, -1259.1392167224028,
           771.32342877765313, -176.61502916214059, 12.507343278686905,
           -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7)
  if x < 0.5 {
    calc.ln(calc.pi / calc.sin(calc.pi * x)) - _lgamma(1.0 - x)   // reflection
  } else {
    let y = x - 1.0
    let a = c.at(0)
    let tt = y + g + 0.5
    for i in range(1, 9) { a += c.at(i) / (y + i) }
    0.5 * _ln2pi + (y + 0.5) * calc.ln(tt) - tt + calc.ln(a)
  }
}

// ── continuous ──
#let normal(mu: 0.0, sigma: 1.0) = {
  assert(_is-num(mu), message: "dist.normal: mu must be numeric")
  let sigma = _positive(sigma, "normal sigma")
  (name: "Normal", kind: "continuous", mean: mu, var: sigma * sigma,
    pdf: x => calc.exp(-calc.pow((x - mu) / sigma, 2) / 2) / (sigma * calc.sqrt(2 * calc.pi)),
    logpdf: x => -calc.pow((x - mu) / sigma, 2) / 2 - calc.ln(sigma) - _ln2pi / 2)
}

#let laplace(mu: 0.0, b: 1.0) = {
  assert(_is-num(mu), message: "dist.laplace: mu must be numeric")
  let b = _positive(b, "laplace scale")
  (name: "Laplace", kind: "continuous", mean: mu, var: 2 * b * b,
    pdf: x => calc.exp(-calc.abs(x - mu) / b) / (2 * b),
    logpdf: x => -calc.abs(x - mu) / b - calc.ln(2 * b))
}

#let student-t(nu: 3.0, mu: 0.0, sigma: 1.0) = {
  assert(_is-num(mu), message: "dist.student-t: mu must be numeric")
  let nu = _positive(nu, "student-t nu")
  let sigma = _positive(sigma, "student-t sigma")
  (name: "Student-t", kind: "continuous",
    mean: if nu > 1 { mu } else { none },
    var: if nu > 2 { sigma * sigma * nu / (nu - 2) } else { none },
    logpdf: x => {
      let z = (x - mu) / sigma
      (_lgamma((nu + 1) / 2) - _lgamma(nu / 2) - 0.5 * calc.ln(nu * calc.pi)
        - calc.ln(sigma) - ((nu + 1) / 2) * calc.ln(1 + z * z / nu))
    })
}  // pdf falls back to exp(logpdf) via the top-level pdf() helper below

#let uniform(a: 0.0, b: 1.0) = {
  assert(_is-num(a) and _is-num(b) and b > a, message: "dist.uniform: require a < b")
  (name: "Uniform", kind: "continuous", mean: (a + b) / 2, var: calc.pow(b - a, 2) / 12,
    pdf: x => if x >= a and x <= b { 1.0 / (b - a) } else { 0.0 },
    logpdf: x => if x >= a and x <= b { -calc.ln(b - a) } else { _NEG-INF })
}

#let exponential(rate: 1.0) = {
  let rate = _positive(rate, "exponential rate")
  (name: "Exponential", kind: "continuous", mean: 1.0 / rate, var: 1.0 / (rate * rate),
    pdf: x => if x >= 0 { rate * calc.exp(-rate * x) } else { 0.0 },
    logpdf: x => if x >= 0 { calc.ln(rate) - rate * x } else { _NEG-INF })
}

// ── bivariate ──
// A 2-D Gaussian density as a plain function (x, y) → p, from a mean and a 2×2
// covariance (Σ is inverted internally). Drop it straight into field.contour —
// so a covariance-ellipse figure is a one-liner, no hand-rolled inverse.
#let gaussian-2d(mu: (0.0, 0.0), sigma: ((1.0, 0.0), (0.0, 1.0))) = {
  assert(type(mu) == array and mu.len() == 2 and mu.all(_is-num),
    message: "dist.gaussian-2d: mu must be a numeric array of length 2")
  assert(type(sigma) == array and sigma.len() == 2 and
    sigma.all(r => type(r) == array and r.len() == 2 and r.all(_is-num)),
    message: "dist.gaussian-2d: sigma must be a numeric 2x2 matrix")
  let (a, b) = (sigma.at(0).at(0), sigma.at(0).at(1))
  let (c, d) = (sigma.at(1).at(0), sigma.at(1).at(1))
  let det = a * d - b * c
  assert(calc.abs(b - c) < 1e-10,
    message: "dist.gaussian-2d: covariance must be symmetric")
  assert(a > 0 and det > 0,
    message: "dist.gaussian-2d: covariance must be positive definite")
  let (ia, io, id) = (d / det, -(b + c) / det, a / det)   // Σ⁻¹ as x² , xy , y² coefficients
  let (mx, my) = mu
  let scale = 1.0 / (2.0 * calc.pi * calc.sqrt(det))
  (x, y) => {
    let (u, v) = (x - mx, y - my)
    scale * calc.exp(-0.5 * (ia * u * u + io * u * v + id * v * v))
  }
}

// ── discrete ──
#let bernoulli(p: 0.5) = {
  let p = _prob(p, name: "bernoulli p")
  (name: "Bernoulli", kind: "discrete", mean: p, var: p * (1 - p),
    pmf: k => if k == 1 { p } else if k == 0 { 1 - p } else { 0.0 },
    logpmf: k => if k == 1 { _log-prob(p) }
      else if k == 0 { _log-prob(1 - p) } else { _NEG-INF })
}

#let categorical(probs: (0.5, 0.5)) = {
  assert(type(probs) == array and probs.len() > 0 and probs.all(p => _is-num(p) and p >= 0),
    message: "dist.categorical: probabilities must be a non-empty, non-negative array")
  assert(calc.abs(probs.sum() - 1.0) < 1e-9,
    message: "dist.categorical: probabilities must sum to 1")
  (name: "Categorical", kind: "discrete",
    pmf: k => if type(k) == int and k >= 0 and k < probs.len() { probs.at(k) } else { 0.0 },
    logpmf: k => if type(k) == int and k >= 0 and k < probs.len() { _log-prob(probs.at(k)) } else { _NEG-INF })
}

// ── likelihood helpers (functional accessors — no `(d.pdf)(x)` paren dance) ──
#let logpdf(d, x) = if "logpdf" in d { (d.logpdf)(x) } else { (d.logpmf)(x) }
#let pdf(d, x) = if "pdf" in d { (d.pdf)(x) } else if "pmf" in d { (d.pmf)(x) } else { calc.exp(logpdf(d, x)) }
// negative log-likelihood at x — the loss a model pays for the observation x
#let nll(d, x) = -logpdf(d, x)
// NLL shifted so its minimum sits at 0 — to compare loss SHAPES across models
#let nll0(d, x, at: 0.0) = nll(d, x) - nll(d, at)

// ── common transforms (link functions) ──
#let sigmoid(z) = {
  assert(_is-num(z), message: "dist.sigmoid: input must be numeric")
  if z >= 0 { 1.0 / (1.0 + calc.exp(-z)) } else {
    calc.exp(z) / (1.0 + calc.exp(z))
  }
}
#let softmax(zs, temperature: 1.0) = {
  assert(type(zs) == array and zs.len() > 0 and zs.all(_is-num),
    message: "dist.softmax: logits must be non-empty and numeric")
  assert(_is-num(temperature) and temperature > 0, message: "dist.softmax: temperature must be positive")
  let m = calc.max(..zs)
  let e = zs.map(z => calc.exp((z - m) / temperature))
  let s = e.sum()
  e.map(v => v / s)
}
