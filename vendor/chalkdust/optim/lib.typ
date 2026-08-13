// optim — small numerical optimization in Typst.
//
// Gradient-based optimizers that RUN in Typst and return the full descent
// trajectory (a list of parameter vectors). The path is the actual iteration,
// so a loss-landscape figure can never disagree with the optimizer on the slide
// — hand the path straight to `field.contour(paths:)`.
//
// Everything is N-dimensional (a parameter is an array of any length); the 2-D
// case is just length-2 vectors, which is what the field plots consume.
//
//   #import "vendor/chalkdust/optim/lib.typ" as opt
//   #let grad(p) = (2*p.at(0), 200*p.at(1))          // ∇ of x² + 100y²
//   #let path = opt.momentum(grad, (-2.3, 0.45), lr: 0.0035, steps: 34)
//   // path == ((-2.3, 0.45), (…​), …) — feed to contour(paths: (path,))

#import "../rand/lib.typ" as rnd   // seeded PRNG for SGD gradient noise

// ── vector ops (arrays of any length) ──
#let _is-num(v) = type(v) == int or type(v) == float
#let _vec(v, name: "vector", nonempty: false) = {
  assert(type(v) == array and (not nonempty or v.len() > 0) and v.all(_is-num),
    message: "optim." + name + ": expected a numeric" + (if nonempty { ", non-empty" } else { "" }) + " vector")
  v.len()
}
#let _same(a, b, name) = {
  let na = _vec(a, name: name + " left")
  let nb = _vec(b, name: name + " right")
  assert(na == nb, message: "optim." + name + ": vector lengths must match")
}
#let add(a, b) = { _same(a, b, "add"); a.zip(b).map(((x, y)) => x + y) }
#let sub(a, b) = { _same(a, b, "sub"); a.zip(b).map(((x, y)) => x - y) }
#let scale(a, s) = { let _ = _vec(a, name: "scale"); assert(_is-num(s), message: "optim.scale: scalar must be numeric"); a.map(x => x * s) }
#let dot(a, b) = { _same(a, b, "dot"); a.zip(b).fold(0.0, (s, pair) => s + pair.at(0) * pair.at(1)) }
#let norm(a) = { let _ = _vec(a, name: "norm"); calc.sqrt(a.fold(0.0, (s, x) => s + x * x)) }

// an n-vector of independent N(0, sigma) gradient noise for step i (seeded)
#let _noise-vec(seed, i, n, sigma) = rnd.randnvec(seed, i, n).map(z => sigma * z)

// ── finite-difference gradient of a scalar f(vector) ──────────────────────
// numgrad(f) returns a gradient function so you can optimize f without deriving
// ∇f by hand:  minimize(numgrad(f), x0, …).
#let numgrad(f, h: 1e-4) = {
  assert(type(f) == function, message: "optim.numgrad: f must be a function")
  assert(_is-num(h) and h > 0, message: "optim.numgrad: h must be positive")
  x => {
    let _ = _vec(x, name: "numgrad point", nonempty: true)
    range(x.len()).map(j => {
      let xp = x; xp.at(j) = xp.at(j) + h
      let xm = x; xm.at(j) = xm.at(j) - h
      (f(xp) - f(xm)) / (2.0 * h)
    })
  }
}

// grad2d(f): the gradient of a 2-D loss written as f(x, y) → scalar, as a
// p → (gx, gy) function ready for the optimizers. Write the loss ONCE (it also
// draws the contour) and the descent gradient can never disagree with it.
#let grad2d(f, h: 1e-4) = numgrad(v => f(v.at(0), v.at(1)), h: h)

// ── the unified optimizer ─────────────────────────────────────────────────
// grad: vector → vector.  x0: starting vector.  Returns [x0, x1, …, x_steps].
//   method: "gd" | "momentum" | "nesterov" | "rmsprop" | "adam"
//   noise > 0 adds seeded Gaussian gradient noise → an SGD-style stochastic path.
#let minimize(
  grad, x0, method: "gd", lr: 0.01, steps: 30,
  beta: 0.9, b1: 0.9, b2: 0.999, eps: 1e-8, noise: 0.0, seed: 1,
) = {
  assert(type(grad) == function, message: "optim.minimize: grad must be a function")
  assert(("gd", "momentum", "nesterov", "rmsprop", "adam").contains(method),
    message: "optim.minimize: unknown method '" + str(method) + "'")
  assert(_is-num(lr) and lr > 0, message: "optim.minimize: lr must be positive")
  assert(type(steps) == int and steps >= 0, message: "optim.minimize: steps must be a non-negative integer")
  assert(_is-num(beta) and beta >= 0 and beta < 1, message: "optim.minimize: beta must be in [0, 1)")
  assert(_is-num(b1) and b1 >= 0 and b1 < 1 and _is-num(b2) and b2 >= 0 and b2 < 1,
    message: "optim.minimize: Adam betas must be in [0, 1)")
  assert(_is-num(eps) and eps > 0, message: "optim.minimize: eps must be positive")
  assert(_is-num(noise) and noise >= 0, message: "optim.minimize: noise must be non-negative")
  assert(type(seed) == int, message: "optim.minimize: seed must be an integer")
  let _ = _vec(x0, name: "minimize x0", nonempty: true)
  let n = x0.len()
  let x = x0.map(v => v * 1.0)
  let vel = range(n).map(_ => 0.0)   // momentum velocity / rmsprop & adam 2nd moment
  let m = range(n).map(_ => 0.0)     // adam 1st moment
  let path = (x,)
  for i in range(steps) {
    // Nesterov evaluates the gradient at a look-ahead point
    let gp = if method == "nesterov" { add(x, scale(vel, beta)) } else { x }
    let g = grad(gp)
    let _ = _vec(g, name: "gradient")
    assert(g.len() == n, message: "optim.minimize: gradient length must match x0")
    if noise > 0.0 { g = add(g, _noise-vec(seed, i, n, noise)) }
    if method == "momentum" {
      vel = add(scale(vel, beta), g)
      x = sub(x, scale(vel, lr))
    } else if method == "nesterov" {
      vel = sub(scale(vel, beta), scale(g, lr))
      x = add(x, vel)
    } else if method == "rmsprop" {
      vel = range(n).map(j => beta * vel.at(j) + (1.0 - beta) * g.at(j) * g.at(j))
      x = range(n).map(j => x.at(j) - lr * g.at(j) / (calc.sqrt(vel.at(j)) + eps))
    } else if method == "adam" {
      let t = i + 1
      m = range(n).map(j => b1 * m.at(j) + (1.0 - b1) * g.at(j))
      vel = range(n).map(j => b2 * vel.at(j) + (1.0 - b2) * g.at(j) * g.at(j))
      let bc1 = 1.0 - calc.pow(b1, t)
      let bc2 = 1.0 - calc.pow(b2, t)
      x = range(n).map(j => x.at(j) - lr * (m.at(j) / bc1) / (calc.sqrt(vel.at(j) / bc2) + eps))
    } else {   // gd
      x = sub(x, scale(g, lr))
    }
    path.push(x)
  }
  path
}

// ── named wrappers (read at the call site like the optimizer you mean) ──────
#let gd(grad, x0, ..a) = minimize(grad, x0, method: "gd", ..a)
#let momentum(grad, x0, ..a) = minimize(grad, x0, method: "momentum", ..a)
#let nesterov(grad, x0, ..a) = minimize(grad, x0, method: "nesterov", ..a)
#let rmsprop(grad, x0, ..a) = minimize(grad, x0, method: "rmsprop", ..a)
#let adam(grad, x0, ..a) = minimize(grad, x0, method: "adam", ..a)
// stochastic GD: plain GD with seeded gradient noise
#let sgd(grad, x0, noise: 1.0, ..a) = minimize(grad, x0, method: "gd", noise: noise, ..a)
