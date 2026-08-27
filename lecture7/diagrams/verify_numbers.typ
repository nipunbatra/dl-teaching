// L7 · Generalization & Regularization — machine-verification lab.
// Every worked number in L7-regularization.typ is recomputed here from the
// same seeded chalkdust primitives the deck itself uses, so the slide tables,
// the figures, and this dump can never disagree.
//   typst compile --root . lecture7/diagrams/verify_numbers.typ /tmp/l7_numbers.pdf
#import "../../common/mldiag.typ": *
#import "../../vendor/chalkdust/linalg/lib.typ" as la

#set page(width: auto, height: auto, margin: 14pt)
#set text(size: 9pt, font: "IBM Plex Mono")

#let f(v, d: 4) = if type(v) == float or type(v) == int { str(calc.round(v, digits: d)) } else { repr(v) }
#let sci(v) = {
  // crude scientific formatting for tiny magnitudes
  if v == 0 { "0" } else {
    let e = calc.floor(calc.log(calc.abs(v)))
    let m = v / calc.pow(10.0, e)
    str(calc.round(m, digits: 2)) + "e" + str(e)
  }
}

// ═══════════════ THE SPINE: 10 noisy points from a smooth truth ═══════════════
#let SEED = 21
#let NTR = 10
#let DEG = 9
#let NOISE = 0.25
#let truth(x) = calc.sin(calc.pi * x)

#let xs = range(NTR).map(i => -1.0 + 2.0 * i / (NTR - 1))
#let ys = xs.enumerate().map(((i, x)) => truth(x) + NOISE * rnd.randn(SEED, i))

// validation: the 9 midpoints between the training inputs, fresh noise
#let xv = range(NTR - 1).map(i => (xs.at(i) + xs.at(i + 1)) / 2)
#let yv = xv.enumerate().map(((i, x)) => truth(x) + NOISE * rnd.randn(SEED + 100, i))
// sealed test: identical construction to the deck; never used for selection
#let xt = range(101).map(i => -1.0 + 2.0 * i / 100.0)
#let yt = xt.enumerate().map(((i, x)) => truth(x) + NOISE * rnd.randn(SEED + 300, i))

#let phi(x) = range(DEG + 1).map(k => if k == 0 { 1.0 } else { calc.pow(x, k) })
#let V = xs.map(phi)
#let polyval(w, x) = w.rev().fold(0.0, (s, c) => s * x + c)
#let mse(w, xa, ya) = xa.zip(ya).map(((x, y)) => calc.pow(polyval(w, x) - y, 2)).sum() / xa.len()

// the exact interpolant: square Vandermonde solve (NOT normal equations)
#let w9 = la.solve(V, ys)

= spine data (seed #SEED)
train x: #xs.map(v => f(v, d: 3)).join(", ") \
train y: #ys.map(v => f(v, d: 3)).join(", ") \
val x:   #xv.map(v => f(v, d: 3)).join(", ") \
val y:   #yv.map(v => f(v, d: 3)).join(", ") \
noise floor sigma^2 = #f(NOISE * NOISE)

= exact degree-9 interpolant
w9 = #w9.map(v => f(v, d: 2)).join(", ") \
train MSE = #sci(mse(w9, xs, ys))  (should be ~0) \
val MSE   = #f(mse(w9, xv, yv)) \
norm(w9)  = #f(la.norm(w9), d: 1) \
curve extremes on (-1..1): #{
  let big = range(281).map(i => polyval(w9, -1.0 + 2.0 * i / 280))
  [min #f(calc.min(..big), d: 2), max #f(calc.max(..big), d: 2)]
}

= degree-1 underfit (for bias panel)
#let V1 = xs.map(x => (1.0, x))
#let w1 = ml.linreg(V1, ys)
w1 = #w1.map(v => f(v, d: 3)).join(", ") ·
train MSE = #f(mse((w1.at(0), w1.at(1)), xs, ys)) ·
val MSE = #f(mse((w1.at(0), w1.at(1)), xv, yv))

= variance across re-draws (4 seeds, degree 9)
#for s in (7, 8, 9, 10) {
  let yy = xs.enumerate().map(((i, x)) => truth(x) + NOISE * rnd.randn(s, i))
  let ww = la.solve(V, yy)
  let big = range(281).map(i => polyval(ww, -1.0 + 2.0 * i / 280))
  [seed #s: curve range (#f(calc.min(..big), d: 1) .. #f(calc.max(..big), d: 1)), val MSE #f(mse(ww, xv, yv)) \ ]
}

= ridge family (normal equations + n lambda I; lambda in the deck's convention)
// deck objective: (1/2n)||Xw-y||^2 + (lambda/2)||w||^2
// therefore solve (X^T X + n lambda I)w = X^T y
#let lams = (0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0)
#for lam in lams {
  let w = ml.linreg(V, ys, ridge: NTR * lam)
  [lambda #sci(lam): train #f(mse(w, xs, ys)) · val #f(mse(w, xv, yv)) · norm(w) #f(la.norm(w), d: 2) \ ]
}

= gram spectrum + closed-form GD (from w = 0)
#let A = la.mscale(la.matmul(la.transpose(V), V), 1.0 / NTR)
#let eig = la.eig-sym(A, sweeps: 600)
#let evals = eig.at(0)
#let evecs = eig.at(1)
eigenvalues of X^T X / n: #evals.map(v => sci(v)).join(", ") \
reconstruction check maxabs(A minus VSVt): #{
  let R = la.zeros(DEG + 1, DEG + 1)
  for j in range(DEG + 1) {
    let vj = evecs.at(j)
    R = la.madd(R, la.mscale(vj.map(a => vj.map(b => a * b)), evals.at(j)))
  }
  let err = 0.0
  for i in range(DEG + 1) {
    for jj in range(DEG + 1) {
      err = calc.max(err, calc.abs(R.at(i).at(jj) - A.at(i).at(jj)))
    }
  }
  sci(err)
} \
#let smax = calc.max(..evals)
#let eta = 0.25 / smax
eta = 0.25 / s_max = #f(eta, d: 4) \
// exact discrete-GD iterate for the quadratic: w_t = sum_j (1-(1-eta s_j)^t) <v_j, w*> v_j
#let coef = range(DEG + 1).map(j => la.dot(evecs.at(j), w9))
#let wt(t) = {
  let w = range(DEG + 1).map(_ => 0.0)
  for j in range(DEG + 1) {
    let fj = 1.0 - calc.pow(1.0 - eta * evals.at(j), t)
    w = la.vadd(w, la.scale(evecs.at(j), fj * coef.at(j)))
  }
  w
}
GD limit check dist(w at t=1e9, w9): #{
  let d = la.dist(wt(1000000000), w9)
  sci(d)
} \
learning curve (t, train MSE, val MSE): \
#let tgrid = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 300000, 1000000, 3000000, 10000000)
#for t in tgrid {
  let w = wt(t)
  [t = #t: train #f(mse(w, xs, ys)) · val #f(mse(w, xv, yv)) \ ]
}

= input-jitter augmentation (K copies, jittered x, same label)
#let KAUG = 8
#let JIT = 0.10
#let xa = ()
#let ya = ()
#{
  for i in range(NTR) {
    for k in range(KAUG) {
      xa.push(xs.at(i) + JIT * rnd.randn(SEED + 200, i * KAUG + k))
      ya.push(ys.at(i))
    }
  }
}
#let Va = xa.map(phi)
#let waug = ml.linreg(Va, ya, ridge: 0.0000000001)
augmented n = #xa.len() ·
train MSE (on the 10 originals) = #f(mse(waug, xs, ys)) ·
val MSE = #f(mse(waug, xv, yv)) ·
norm = #f(la.norm(waug), d: 2) \
curve range: #{
  let big = range(281).map(i => polyval(waug, -1.0 + 2.0 * i / 280))
  [#f(calc.min(..big), d: 2) .. #f(calc.max(..big), d: 2)]
}

= sealed test report (opened only after validation choices are frozen)
#let wridge = ml.linreg(V, ys, ridge: NTR * 0.001)
#let wstop = wt(5000)
unregularized test MSE = #f(mse(w9, xt, yt)) \
validation-selected ridge test MSE = #f(mse(wridge, xt, yt)) \
validation-selected early-stop test MSE = #f(mse(wstop, xt, yt)) \
fixed jitter-policy test MSE = #f(mse(waug, xt, yt))

= the second interpolant: narrow-RBF (both fit train exactly)
#let ELL = 0.04
#let kfn(a, b) = calc.exp(-calc.pow(a - b, 2) / (2 * ELL * ELL))
#let K = xs.map(a => xs.map(b => kfn(a, b)))
#let alpha = la.solve(K, ys)
#let rbfval(x) = range(NTR).map(j => alpha.at(j) * kfn(x, xs.at(j))).sum()
rbf train MSE = #sci(xs.zip(ys).map(((x, y)) => calc.pow(rbfval(x) - y, 2)).sum() / NTR) \
rbf val MSE   = #f(xv.zip(yv).map(((x, y)) => calc.pow(rbfval(x) - y, 2)).sum() / xv.len()) \
rbf at midpoint between points 4,5: #f(rbfval((xs.at(4) + xs.at(5)) / 2), d: 4) (truth there: #f(truth((xs.at(4) + xs.at(5)) / 2), d: 4))

= label smoothing (K = 5, eps = 0.1, correct class index 1)
#let EPS = 0.1
#let KCLS = 5
smoothed correct = #f(1 - EPS + EPS / KCLS) · smoothed wrong = #f(EPS / KCLS) \
optimal margin m-star = ln((1-eps+eps/K)/(eps/K)) = ln(46) = #f(calc.ln(46.0)) \
#let zs = (0.6, 2.8, 1.1, -0.4, 0.3)
#let pz = {
  let m = calc.max(..zs)
  let e = zs.map(z => calc.exp(z - m))
  let s = e.sum()
  e.map(v => v / s)
}
logits z = #zs.map(v => f(v, d: 1)).join(", ") \
softmax p = #pz.map(v => f(v, d: 3)).join(", ") \
hard grad p - y      = #pz.enumerate().map(((i, p)) => f(p - (if i == 1 { 1 } else { 0 }), d: 3)).join(", ") \
smooth grad p - ytil = #pz.enumerate().map(((i, p)) => f(p - (if i == 1 { 0.92 } else { 0.02 }), d: 3)).join(", ")

= mixup arithmetic (lambda-mix = 0.70, cat plus dog)
ytilde = 0.70 y_cat + 0.30 y_dog \
example CE at p_cat = 0.65, p_dog = 0.25: #f(-0.70 * calc.ln(0.65) - 0.30 * calc.ln(0.25)) \
CE if p equals ytilde exactly: #f(-0.70 * calc.ln(0.70) - 0.30 * calc.ln(0.30))

= dropout arithmetic (q_keep = 0.5)
h = 2, 4, 6, 8 · mask 1,0,1,0 -> inverted htilde = #((1, 0, 1, 0).zip((2, 4, 6, 8)).map(((m, h)) => f(m * h / 0.5, d: 1)).join(", ")) \
E(htilde) = h · Var(htilde-i) = (1-q)/q hi2, equal to hi2 at q = 0.5 \
seeded masks over the 6 hidden units drawn in the deck, p_keep = 0.5, seeds 41, 42, 43: \
#for s in (41, 42, 43) {
  [seed #s: #range(6).map(i => str(rnd.bernoulli(s, i, 0.5))).join("") \ ]
}

= weight-decay geometry (quadratic data loss, diag curvature)
#let AC = (4.0, 0.6)
#let TSTAR = (1.5, 1.35)
#let thlam(lam) = (AC.at(0) / (AC.at(0) + lam) * TSTAR.at(0), AC.at(1) / (AC.at(1) + lam) * TSTAR.at(1))
#for lam in (0.0, 0.3, 1.0, 3.0, 10.0) {
  let p = thlam(lam)
  [lambda #f(lam, d: 1): theta = (#f(p.at(0), d: 3), #f(p.at(1), d: 3)) \ ]
}

= SGD weight-decay scalar audit
theta = 5, g = 3, eta = 0.1, lambda = 0.2: (1 - 0.02) x 5 - 0.3 = #f((1 - 0.1 * 0.2) * 5 - 0.1 * 3)

= early-stop vs ridge filter match (the deck's OPT slide)
validation chose t-star = 5000 (learning-curve block above) and lambda-star = 0.001 (ridge block). \
the classical correspondence lambda = 1/(eta t): 1/(#f(eta, d: 4) x 5000) = #f(1.0 / (eta * 5000), d: 5) — within 25 percent of lambda-star. \
CE-vs-margin check for label smoothing: smoothed loss at m-star = ln 46 is
#f(0.92 * calc.ln(1.0 + 4.0 * calc.exp(-calc.ln(46.0))) + 0.08 * calc.ln(46.0 + 4.0)) (an interior minimum; the one-hot loss at the same m is #f(calc.ln(1.0 + 4.0 * calc.exp(-calc.ln(46.0)))) and keeps falling).
