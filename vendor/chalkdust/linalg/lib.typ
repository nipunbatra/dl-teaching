// linalg — small dense linear algebra for Typst.
//
// A matrix is a list of rows (each a list of numbers); a vector is a list. Enough
// to fit a regression (normal equations via `solve`) or run PCA (`eig-sym` of a
// covariance) inside a slide.
//
//   #import "vendor/chalkdust/linalg/lib.typ" as la
//   #la.matmul(((1, 2), (3, 4)), ((5,), (6,)))   // ((17,), (39,))
//   #la.solve(((2, 1), (1, 3)), (3, 5))           // (0.8, 1.4)

// ── vectors ──
#let _is-num(v) = type(v) == int or type(v) == float
#let _check-vec(v, name: "vector", nonempty: false) = {
  assert(type(v) == array, message: "linalg." + name + ": expected an array")
  assert(not nonempty or v.len() > 0, message: "linalg." + name + ": vector must be non-empty")
  assert(v.all(_is-num), message: "linalg." + name + ": vector entries must be numeric")
  v.len()
}
#let _same-vecs(a, b, name) = {
  let na = _check-vec(a, name: name + " left")
  let nb = _check-vec(b, name: name + " right")
  assert(na == nb, message: "linalg." + name + ": vector lengths must match")
}
#let _check-matrix(M, name: "matrix", nonempty: false, square: false) = {
  assert(type(M) == array, message: "linalg." + name + ": expected an array of rows")
  assert(not nonempty or M.len() > 0, message: "linalg." + name + ": matrix must be non-empty")
  if M.len() == 0 { return (0, 0) }
  assert(M.all(r => type(r) == array), message: "linalg." + name + ": every row must be an array")
  let c = M.first().len()
  assert(M.all(r => r.len() == c), message: "linalg." + name + ": matrix must be rectangular")
  assert(M.all(r => r.all(_is-num)), message: "linalg." + name + ": matrix entries must be numeric")
  assert(not square or M.len() == c, message: "linalg." + name + ": matrix must be square")
  (M.len(), c)
}
#let vadd(a, b) = { _same-vecs(a, b, "vadd"); a.zip(b).map(((x, y)) => x + y) }
#let vsub(a, b) = { _same-vecs(a, b, "vsub"); a.zip(b).map(((x, y)) => x - y) }
#let scale(a, s) = { let _ = _check-vec(a, name: "scale"); assert(_is-num(s), message: "linalg.scale: scalar must be numeric"); a.map(x => x * s) }
#let dot(a, b) = { _same-vecs(a, b, "dot"); a.zip(b).fold(0.0, (s, pair) => s + pair.at(0) * pair.at(1)) }
#let norm(a) = { let _ = _check-vec(a, name: "norm"); calc.sqrt(a.fold(0.0, (s, x) => s + x * x)) }
#let dist(a, b) = norm(vsub(a, b))
#let normalize(a) = { let n = norm(a); if n == 0 { a } else { scale(a, 1.0 / n) } }
#let mean(vs) = {
  assert(type(vs) == array and vs.len() > 0, message: "linalg.mean: vectors must be non-empty")
  let n = _check-vec(vs.first(), name: "mean vector", nonempty: true)
  assert(vs.all(v => { let _ = _check-vec(v, name: "mean vector"); v.len() == n }),
    message: "linalg.mean: all vectors must have the same length")
  scale(vs.fold(range(n).map(_ => 0.0), vadd), 1.0 / vs.len())
}

// ── matrix shape & construction ──
#let nrows(M) = { let _ = _check-matrix(M); M.len() }
#let ncols(M) = { let (_, c) = _check-matrix(M); c }
#let row(M, i) = { let (r, _) = _check-matrix(M); assert(type(i) == int and i >= 0 and i < r, message: "linalg.row: index out of bounds"); M.at(i) }
#let col(M, j) = { let (_, c) = _check-matrix(M); assert(type(j) == int and j >= 0 and j < c, message: "linalg.col: index out of bounds"); M.map(r => r.at(j)) }
#let transpose(M) = { let (_, c) = _check-matrix(M); range(c).map(j => M.map(r => r.at(j))) }
#let identity(n) = { assert(type(n) == int and n >= 0, message: "linalg.identity: n must be non-negative"); range(n).map(i => range(n).map(j => if i == j { 1.0 } else { 0.0 })) }
#let zeros(r, c) = { assert(type(r) == int and r >= 0 and type(c) == int and c >= 0, message: "linalg.zeros: dimensions must be non-negative integers"); range(r).map(_ => range(c).map(_ => 0.0)) }
#let diag(vals) = { let _ = _check-vec(vals, name: "diag"); range(vals.len()).map(i => range(vals.len()).map(j => if i == j { vals.at(i) } else { 0.0 })) }

// ── matrix arithmetic ──
#let _same-matrices(A, B, name) = {
  let sa = _check-matrix(A, name: name + " left")
  let sb = _check-matrix(B, name: name + " right")
  assert(sa == sb, message: "linalg." + name + ": matrix shapes must match")
}
#let madd(A, B) = { _same-matrices(A, B, "madd"); A.zip(B).map(((ra, rb)) => vadd(ra, rb)) }
#let msub(A, B) = { _same-matrices(A, B, "msub"); A.zip(B).map(((ra, rb)) => vsub(ra, rb)) }
#let mscale(A, s) = { let _ = _check-matrix(A, name: "mscale"); A.map(r => scale(r, s)) }
#let matvec(M, v) = {
  let (_, c) = _check-matrix(M, name: "matvec")
  let n = _check-vec(v, name: "matvec vector")
  assert(c == n, message: "linalg.matvec: inner dimensions must match")
  M.map(r => dot(r, v))
}
#let matmul(A, B) = {
  let (_, ac) = _check-matrix(A, name: "matmul left")
  let (br, _) = _check-matrix(B, name: "matmul right")
  assert(ac == br, message: "linalg.matmul: inner dimensions must match")
  let Bt = transpose(B)
  A.map(arow => Bt.map(bcol => dot(arow, bcol)))
}

// ── Gaussian elimination: solve, inverse, determinant (partial pivoting) ──
#let solve(A, b, eps: 1e-12) = {
  let (n, _) = _check-matrix(A, name: "solve", square: true)
  let nb = _check-vec(b, name: "solve right-hand side")
  assert(nb == n, message: "linalg.solve: right-hand side length must match matrix")
  assert(eps > 0, message: "linalg.solve: eps must be positive")
  if n == 0 { return () }
  let M = range(n).map(i => A.at(i).map(x => x * 1.0) + (b.at(i) * 1.0,))   // augmented
  for c in range(n) {
    let piv = c
    let best = calc.abs(M.at(c).at(c))
    for r in range(c + 1, n) { if calc.abs(M.at(r).at(c)) > best { best = calc.abs(M.at(r).at(c)); piv = r } }
    assert(best > eps, message: "linalg.solve: matrix is singular or ill-conditioned")
    if piv != c { let t = M.at(c); M.at(c) = M.at(piv); M.at(piv) = t }
    let pv = M.at(c).at(c)
    for r in range(c + 1, n) {
      let f = M.at(r).at(c) / pv
      M.at(r) = range(n + 1).map(k => M.at(r).at(k) - f * M.at(c).at(k))
    }
  }
  let x = range(n).map(_ => 0.0)
  for ri in range(n) {
    let i = n - 1 - ri
    let s = M.at(i).at(n)
    for j in range(i + 1, n) { s = s - M.at(i).at(j) * x.at(j) }
    x.at(i) = s / M.at(i).at(i)
  }
  x
}
#let inv(M) = {
  let (n, _) = _check-matrix(M, name: "inv", square: true)
  let cols = range(n).map(j => solve(M, range(n).map(i => if i == j { 1.0 } else { 0.0 })))
  range(n).map(i => range(n).map(j => cols.at(j).at(i)))       // columns → rows
}
#let det(A, eps: 1e-12) = {
  let (n, _) = _check-matrix(A, name: "det", square: true)
  assert(eps > 0, message: "linalg.det: eps must be positive")
  if n == 0 { return 1.0 }
  let M = A.map(r => r.map(x => x * 1.0))
  let d = 1.0
  for c in range(n) {
    let piv = c
    let best = calc.abs(M.at(c).at(c))
    for r in range(c + 1, n) { if calc.abs(M.at(r).at(c)) > best { best = calc.abs(M.at(r).at(c)); piv = r } }
    if best <= eps { return 0.0 }
    if piv != c { let t = M.at(c); M.at(c) = M.at(piv); M.at(piv) = t; d = d * -1.0 }
    d = d * M.at(c).at(c)
    for r in range(c + 1, n) {
      let f = M.at(r).at(c) / M.at(c).at(c)
      M.at(r) = range(n).map(k => M.at(r).at(k) - f * M.at(c).at(k))
    }
  }
  d
}

// ── symmetric eigendecomposition via cyclic Jacobi rotations ──
// Returns (values, vectors): `values.at(k)` with eigenvector `vectors.at(k)` (a
// column). Assumes A is symmetric (e.g. a covariance matrix) — perfect for PCA.
#let eig-sym(A0, sweeps: 60) = {
  let (n, _) = _check-matrix(A0, name: "eig-sym", square: true)
  assert(type(sweeps) == int and sweeps > 0, message: "linalg.eig-sym: sweeps must be positive")
  assert(range(n).all(i => range(i + 1, n).all(j => calc.abs(A0.at(i).at(j) - A0.at(j).at(i)) < 1e-10)),
    message: "linalg.eig-sym: matrix must be symmetric")
  let A = A0.map(r => r.map(x => x * 1.0))
  let V = identity(n)
  for _ in range(sweeps) {
    // largest off-diagonal magnitude
    let p = 0
    let q = 1
    let mx = 0.0
    for i in range(n) {
      for j in range(i + 1, n) {
        if calc.abs(A.at(i).at(j)) > mx { mx = calc.abs(A.at(i).at(j)); p = i; q = j }
      }
    }
    if mx < 1e-14 { break }
    let theta = 0.5 * calc.atan2(A.at(p).at(p) - A.at(q).at(q), 2.0 * A.at(p).at(q))
    let c = calc.cos(theta)
    let s = calc.sin(theta)
    let J = identity(n)
    J.at(p).at(p) = c
    J.at(q).at(q) = c
    J.at(p).at(q) = -s
    J.at(q).at(p) = s
    A = matmul(matmul(transpose(J), A), J)
    V = matmul(V, J)
  }
  let vals = range(n).map(i => A.at(i).at(i))
  let vecs = range(n).map(j => col(V, j))          // eigenvector j = column j of V
  // sort by descending eigenvalue (PCA convention)
  let order = range(n).sorted(key: k => -vals.at(k))
  (order.map(k => vals.at(k)), order.map(k => vecs.at(k)))
}
