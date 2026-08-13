// learn — classic ML algorithms, fit in Typst.
//
// The capstone of the chalkdust stack: it builds on `linalg` (matrices / solve /
// eig), `optim` (gradient descent), `rand` (k-means init) and `dist` (sigmoid), and
// its results plot straight through `field` / `plot`. Everything is computed, so a
// fitted line, a decision boundary, a clustering or a PCA axis on a slide is the
// real thing — not a drawing.
//
//   #import "vendor/chalkdust/learn/lib.typ" as ml
//   #let w = ml.linreg-fit((0, 1, 2, 3), (1, 3, 5, 7))   // (intercept, slope) = (1, 2)
//   #let (mu, comps, ev) = ml.pca(points)                // principal axes

#import "../linalg/lib.typ" as la
#import "../optim/lib.typ" as opt
#import "../rand/lib.typ" as rnd
#import "../dist/lib.typ" as dist

// helper: prepend a 1 (bias) to a scalar or vector feature
#let _design(x) = (1.0,) + (if type(x) == array { x } else { (x,) })
#let _is-num(v) = type(v) == int or type(v) == float
#let _matrix(X, name) = {
  assert(type(X) == array and X.len() > 0 and X.all(r => type(r) == array),
    message: "learn." + name + ": X must be a non-empty array of rows")
  let d = X.first().len()
  assert(d > 0 and X.all(r => r.len() == d and r.all(_is-num)),
    message: "learn." + name + ": X must be rectangular and numeric")
  d
}
#let _supervised(X, y, name) = {
  let d = _matrix(X, name)
  assert(type(y) == array and y.len() == X.len(),
    message: "learn." + name + ": y length must match X")
  (X.len(), d)
}
#let _point-set(points, name) = _matrix(points, name)

// ── linear regression by the normal equations (via linalg) ──
// linreg(X, y): X already a design matrix (rows = samples). Returns weights w.
#let linreg(X, y, ridge: 0.0) = {
  let (_, d) = _supervised(X, y, "linreg")
  assert(y.all(_is-num), message: "learn.linreg: y must be numeric")
  assert(_is-num(ridge) and ridge >= 0, message: "learn.linreg: ridge must be non-negative")
  let Xt = la.transpose(X)
  let gram = la.matmul(Xt, X)
  if ridge > 0 { gram = la.madd(gram, la.diag(range(d).map(_ => ridge))) }
  la.solve(gram, la.matvec(Xt, y))
}
// linreg-fit(xs, y): xs a list of scalar-or-vector features; adds the bias column.
// Returns (intercept, slope…).
#let linreg-fit(xs, y, ..args) = {
  assert(type(xs) == array and xs.len() > 0, message: "learn.linreg-fit: xs must be non-empty")
  linreg(xs.map(_design), y, ..args)
}
#let linreg-predict(w, x) = la.dot(w, _design(x))

// ── logistic regression by gradient descent (via optim + dist) ──
// X: design matrix (include a bias column), y: 0/1 labels. Returns weights.
#let logreg(X, y, lr: 0.5, steps: 800) = {
  let (n, d) = _supervised(X, y, "logreg")
  assert(y.all(v => v == 0 or v == 1), message: "learn.logreg: labels must be 0 or 1")
  let Xt = la.transpose(X)
  let grad(w) = {                                   // ∇ = (1/n) Xᵀ(σ(Xw) − y)
    let err = X.zip(y).map(((row, yi)) => dist.sigmoid(la.dot(row, w)) - yi)
    la.scale(la.matvec(Xt, err), 1.0 / n)
  }
  opt.gd(grad, range(d).map(_ => 0.0), lr: lr, steps: steps).last()
}
#let logreg-prob(w, x) = dist.sigmoid(la.dot(w, x))

// ── k-means (via rand for init, linalg for distances) ──
// points: list of vectors. Returns (centroids, assignments).
#let kmeans(points, k, seed: 0, iters: 25) = {
  let _ = _point-set(points, "kmeans")
  assert(type(k) == int and k >= 1 and k <= points.len(), message: "learn.kmeans: require 1 <= k <= number of points")
  assert(type(seed) == int, message: "learn.kmeans: seed must be an integer")
  assert(type(iters) == int and iters >= 0, message: "learn.kmeans: iters must be non-negative")
  let centroids = rnd.shuffle(seed, range(points.len())).slice(0, k).map(i => points.at(i))
  let assign-to(cs) = points.map(p => {
    let best = 0
    let bd = la.dist(p, cs.at(0))
    for c in range(1, k) {
      let dd = la.dist(p, cs.at(c))
      if dd < bd { bd = dd; best = c }
    }
    best
  })
  let assign = assign-to(centroids)
  for _ in range(iters) {
    centroids = range(k).map(c => {
      let members = points.enumerate().filter(it => assign.at(it.at(0)) == c).map(it => it.at(1))
      if members.len() == 0 { centroids.at(c) } else { la.mean(members) }
    })
    assign = assign-to(centroids)
  }
  (centroids, assign)
}

// ── k-nearest-neighbours classification (via linalg distances) ──
#let knn(train-X, train-y, query, k: 3) = {
  let d = _point-set(train-X, "knn")
  assert(type(train-y) == array and train-y.len() == train-X.len(),
    message: "learn.knn: label count must match training points")
  assert(type(query) == array and query.len() == d and query.all(_is-num),
    message: "learn.knn: query dimension must match training points")
  assert(type(k) == int and k >= 1 and k <= train-X.len(), message: "learn.knn: require 1 <= k <= training size")
  let nearest = train-X.enumerate()
    .map(it => (la.dist(it.at(1), query), train-y.at(it.at(0))))
    .sorted(key: it => it.at(0))
    .slice(0, k).map(it => it.at(1))
  let labels = ()
  for l in nearest { if not labels.contains(l) { labels.push(l) } }
  labels.map(l => (l, nearest.filter(x => x == l).len())).sorted(key: c => -c.at(1)).first().at(0)
}

// ── PCA (via linalg eigendecomposition of the covariance) ──
// data: list of vectors. Returns a dict (mean, components, values) — components
// are the top-k principal axes (unit vectors), values the variance along each.
#let pca(data, k: 2) = {
  let dims = _point-set(data, "pca")
  assert(type(k) == int and k >= 1 and k <= dims, message: "learn.pca: require 1 <= k <= feature dimension")
  let mu = la.mean(data)
  let centered = data.map(x => la.vsub(x, mu))
  let n = data.len()
  let d = mu.len()
  let cov = range(d).map(a => range(d).map(b => centered.map(x => x.at(a) * x.at(b)).sum() / n))
  let (vals, vecs) = la.eig-sym(cov)
  (mean: mu, components: vecs.slice(0, k), values: vals.slice(0, k))
}
// project a point onto the top-k principal axes
#let pca-project(model, x) = model.components.map(v => la.dot(la.vsub(x, model.mean), v))

// ── standardise a feature column to mean 0, variance 1 ──
#let standardize(xs) = {
  assert(type(xs) == array and xs.len() > 0 and xs.all(_is-num),
    message: "learn.standardize: xs must be a non-empty numeric array")
  let m = xs.sum() / xs.len()
  let sd = calc.sqrt(xs.map(x => calc.pow(x - m, 2)).sum() / xs.len())
  if sd == 0 { xs.map(_ => 0.0) } else { xs.map(x => (x - m) / sd) }
}
