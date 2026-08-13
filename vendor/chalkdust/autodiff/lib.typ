// autodiff — reverse-mode automatic differentiation in Typst (micrograd, tiny).
//
// Typst has no operator overloading, so you build a scalar expression out of the
// differentiable primitives below. Each returns an immutable node that remembers
// its value and, for each parent, the LOCAL derivative ∂self/∂parent. `grad` then
// does one reverse pass (chain rule) and returns the exact gradient — no finite
// differences, no mutable state.
//
//   #import "vendor/chalkdust/autodiff/lib.typ" as ad
//   #let f(v) = ad.add(ad.sq(v.at(0)), ad.mul(100, ad.sq(v.at(1))))  // x² + 100y²
//   #ad.value(f, (-2.3, 0.45))   // 26.5…   (the loss, e.g. for a contour)
//   #ad.grad(f, (-2.3, 0.45))    // (-4.6, 90.0) exactly   (feed to optim)
//   #ad.graph(f, (-2.3, 0.45), names: ("x", "y"))   // draw the computation graph

#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge
#import "../theme/lib.typ": default-theme

// ── nodes ──
// A node: (v: value, kids: ((parent, local-deriv), …), slot: input-index | none,
//          op: a label for drawing the graph)
#let _is-num(v) = type(v) == int or type(v) == float
#let _const(c) = (v: c, kids: (), slot: none, op: "const")
#let var(i, x) = {
  assert(type(i) == int and i >= 0 and _is-num(x),
    message: "autodiff.var: slot must be non-negative and value must be numeric")
  (v: x, kids: (), slot: i, op: "var")
}   // the i-th input variable
#let _lift(a) = {
  if type(a) == dictionary {
    assert(("v", "kids", "slot", "op").all(k => k in a) and _is-num(a.v) and type(a.kids) == array,
      message: "autodiff: operand must be numeric or an autodiff node")
    a
  } else {
    assert(_is-num(a), message: "autodiff: operand must be numeric or an autodiff node")
    _const(a)
  }
}   // scalars → constants

// ── differentiable primitives (each records local derivatives + a draw label) ──
#let add(a, b) = { let (a, b) = (_lift(a), _lift(b)); (v: a.v + b.v, kids: ((a, 1.0), (b, 1.0)), slot: none, op: "+") }
#let sub(a, b) = { let (a, b) = (_lift(a), _lift(b)); (v: a.v - b.v, kids: ((a, 1.0), (b, -1.0)), slot: none, op: "−") }
#let mul(a, b) = { let (a, b) = (_lift(a), _lift(b)); (v: a.v * b.v, kids: ((a, b.v), (b, a.v)), slot: none, op: "×") }
#let div(a, b) = { let (a, b) = (_lift(a), _lift(b))
  assert(b.v != 0, message: "autodiff.div: division by zero")
  (v: a.v / b.v, kids: ((a, 1.0 / b.v), (b, -a.v / (b.v * b.v))), slot: none, op: "÷") }
#let neg(a) = { let a = _lift(a); (v: -a.v, kids: ((a, -1.0),), slot: none, op: "−") }
#let powc(a, p) = { let a = _lift(a)               // constant real power
  assert(_is-num(p), message: "autodiff.powc: exponent must be numeric")
  assert(a.v >= 0 or p == calc.round(p), message: "autodiff.powc: negative base requires an integer exponent")
  assert(not (a.v == 0 and p < 1 and p != 0), message: "autodiff.powc: derivative is undefined at zero for this exponent")
  let deriv = if p == 0 { 0.0 } else { p * calc.pow(a.v, p - 1.0) }
  (v: calc.pow(a.v, p), kids: ((a, deriv),), slot: none,
   op: "^" + (if p == calc.round(p) { str(int(p)) } else { str(p) })) }
#let sq(a) = { let a = _lift(a); (v: a.v * a.v, kids: ((a, 2.0 * a.v),), slot: none, op: "()²") }
#let sqrt(a) = { let a = _lift(a); assert(a.v > 0, message: "autodiff.sqrt: gradient requires a positive input"); let s = calc.sqrt(a.v); (v: s, kids: ((a, 0.5 / s),), slot: none, op: "√") }
#let exp(a) = { let a = _lift(a); let e = calc.exp(a.v); (v: e, kids: ((a, e),), slot: none, op: "exp") }
#let ln(a) = { let a = _lift(a); assert(a.v > 0, message: "autodiff.ln: input must be positive"); (v: calc.ln(a.v), kids: ((a, 1.0 / a.v),), slot: none, op: "ln") }
#let sin(a) = { let a = _lift(a); (v: calc.sin(a.v), kids: ((a, calc.cos(a.v)),), slot: none, op: "sin") }
#let cos(a) = { let a = _lift(a); (v: calc.cos(a.v), kids: ((a, -calc.sin(a.v)),), slot: none, op: "cos") }
#let tanh(a) = { let a = _lift(a)
  let t = if a.v >= 0 { (1.0 - calc.exp(-2.0 * a.v)) / (1.0 + calc.exp(-2.0 * a.v)) } else {
    (calc.exp(2.0 * a.v) - 1.0) / (calc.exp(2.0 * a.v) + 1.0)
  }
  (v: t, kids: ((a, 1.0 - t * t),), slot: none, op: "tanh") }
#let sigmoid(a) = { let a = _lift(a); let s = if a.v >= 0 { 1.0 / (1.0 + calc.exp(-a.v)) } else {
    calc.exp(a.v) / (1.0 + calc.exp(a.v))
  }
  (v: s, kids: ((a, s * (1.0 - s)),), slot: none, op: "σ") }
#let relu(a) = { let a = _lift(a)
  (v: calc.max(a.v, 0.0), kids: ((a, if a.v > 0.0 { 1.0 } else { 0.0 }),), slot: none, op: "relu") }

// sum / dot over arrays of nodes-or-scalars (handy for linear models)
#let sum(xs) = xs.fold(_const(0.0), (a, b) => add(a, b))
#let dot(xs, ys) = {
  assert(type(xs) == array and type(ys) == array and xs.len() == ys.len(),
    message: "autodiff.dot: vector lengths must match")
  sum(xs.zip(ys).map(((x, y)) => mul(x, y)))
}

// ── the reverse pass ──
// push gradient g down `node`, accumulating into `grads` (one slot per input)
#let _back(node, g, grads) = {
  if node.slot != none { grads.at(node.slot) = grads.at(node.slot) + g; grads }
  else {
    for pair in node.kids { grads = _back(pair.at(0), g * pair.at(1), grads) }
    grads
  }
}

// ── user-facing API ──
// f: (array of nodes) → node.   x: array of scalars (the evaluation point).
#let _point(f, x, name) = {
  assert(type(f) == function, message: "autodiff." + name + ": f must be a function")
  assert(type(x) == array and x.all(_is-num), message: "autodiff." + name + ": x must be a numeric array")
}
#let value(f, x) = { _point(f, x, "value"); _lift(f(x.enumerate().map(((i, xi)) => var(i, xi)))).v }
#let grad(f, x) = {
  _point(f, x, "grad")
  let out = _lift(f(x.enumerate().map(((i, xi)) => var(i, xi))))
  _back(out, 1.0, range(x.len()).map(_ => 0.0))
}
#let value-and-grad(f, x) = {
  _point(f, x, "value-and-grad")
  let out = _lift(f(x.enumerate().map(((i, xi)) => var(i, xi))))
  (out.v, _back(out, 1.0, range(x.len()).map(_ => 0.0)))
}
// adapters: a gradient function for optim, and a plain f(x,y) for a 2-D contour
#let grad-fn(f) = x => grad(f, x)
#let fn2(f) = (x, y) => value(f, (x, y))

// ── expr: build the graph from a STRING, since Typst has no operator overloading ──
// Typst's * and + can't be redefined for autodiff nodes, so `x*x` written directly
// can't be traced. Instead parse a formula string into the graph — you write the
// loss ONCE, readably, and still get the exact gradient.
//   #let f = expr("x*x + 100*y*y", ("x", "y"))
//   grad(f, (-2.3, 0.45))   // (-4.6, 90.0) exactly

#let _tokenize(s) = {
  let cs = s.clusters()
  let digits = "0123456789"
  let idstart = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_"
  let idchar = idstart + digits
  let toks = ()
  let i = 0
  while i < cs.len() {
    let c = cs.at(i)
    if c == " " or c == "\t" or c == "\n" or c == "\r" { i += 1 }
    else if digits.contains(c) or (c == "." and i + 1 < cs.len() and digits.contains(cs.at(i + 1))) {
      let j = i
      while j < cs.len() and (digits.contains(cs.at(j)) or cs.at(j) == ".") { j += 1 }
      toks.push(("num", float(cs.slice(i, j).join())))
      i = j
    } else if idstart.contains(c) {
      let j = i
      while j < cs.len() and idchar.contains(cs.at(j)) { j += 1 }
      toks.push(("id", cs.slice(i, j).join()))
      i = j
    } else if "+-*/^(),".contains(c) {
      toks.push((c,))
      i += 1
    } else { panic("autodiff.expr: unexpected character '" + c + "'") }
  }
  toks
}
#let _prec(op) = if op == "+" or op == "-" { 1 } else if op == "*" or op == "/" { 2 } else if op == "^" { 3 } else { 0 }
// one self-recursive precedence-climbing parser (Typst has no mutual recursion) →
// returns (ast, next-pos). ast nodes: ("num", v) ("var", n) ("neg", a) ("bin", op, l, r) ("call", name, arg)
#let _parse(toks, pos, min-prec) = {
  let t = toks.at(pos)
  let (left, p) = if t.at(0) == "-" {
    let (a, pp) = _parse(toks, pos + 1, 3); (("neg", a), pp)
  } else if t.at(0) == "num" {
    (("num", t.at(1)), pos + 1)
  } else if t.at(0) == "(" {
    let (inner, pp) = _parse(toks, pos + 1, 0)
    assert(pp < toks.len() and toks.at(pp).at(0) == ")", message: "autodiff.expr: expected ')'")
    (inner, pp + 1)
  } else if t.at(0) == "id" {
    if pos + 1 < toks.len() and toks.at(pos + 1).at(0) == "(" {
      let (arg, pp) = _parse(toks, pos + 2, 0)
      assert(pp < toks.len() and toks.at(pp).at(0) == ")", message: "autodiff.expr: expected ')' after function argument")
      (("call", t.at(1), arg), pp + 1)
    } else { (("var", t.at(1)), pos + 1) }
  } else { panic("autodiff.expr: parse error near " + repr(t)) }
  let node = left
  let cur = p
  while cur < toks.len() and _prec(toks.at(cur).at(0)) >= min-prec and _prec(toks.at(cur).at(0)) > 0 {
    let op = toks.at(cur).at(0)
    let next-min = if op == "^" { _prec(op) } else { _prec(op) + 1 }   // ^ right-assoc
    let (rhs, pp) = _parse(toks, cur + 1, next-min)
    node = ("bin", op, node, rhs)
    cur = pp
  }
  (node, cur)
}
#let _eval-ast(node, vars, names) = {
  let k = node.at(0)
  if k == "num" { node.at(1) }
  else if k == "var" {
    let idx = names.position(nm => nm == node.at(1))
    if idx == none { panic("autodiff.expr: unknown variable '" + node.at(1) + "'") }
    vars.at(idx)
  }
  else if k == "neg" { neg(_eval-ast(node.at(1), vars, names)) }
  else if k == "bin" {
    let l = _eval-ast(node.at(2), vars, names)
    let r = _eval-ast(node.at(3), vars, names)
    let op = node.at(1)
    if op == "+" { add(l, r) } else if op == "-" { sub(l, r) }
    else if op == "*" { mul(l, r) } else if op == "/" { div(l, r) }
    else if op == "^" { if node.at(3).at(0) == "num" { powc(l, node.at(3).at(1)) } else { exp(mul(r, ln(l))) } }
  }
  else if k == "call" {
    let a = _eval-ast(node.at(2), vars, names)
    let fn = node.at(1)
    if fn == "exp" { exp(a) } else if fn == "ln" or fn == "log" { ln(a) }
    else if fn == "sin" { sin(a) } else if fn == "cos" { cos(a) }
    else if fn == "tanh" { tanh(a) } else if fn == "sigmoid" { sigmoid(a) }
    else if fn == "relu" { relu(a) } else if fn == "sqrt" { sqrt(a) } else if fn == "sq" { sq(a) }
    else { panic("autodiff.expr: unknown function '" + fn + "'") }
  } else { panic("autodiff.expr: bad node " + repr(node)) }
}
// expr(s, names) → an f(nodes) → node, usable with value / grad / grad-fn / fn2.
#let expr(s, names) = {
  assert(type(s) == str and type(names) == array, message: "autodiff.expr: expected a string and an array of names")
  assert(names.all(n => type(n) == str) and
    names.enumerate().all(((i, n)) => names.position(x => x == n) == i),
    message: "autodiff.expr: variable names must be unique strings")
  let toks = _tokenize(s)
  assert(toks.len() > 0, message: "autodiff.expr: expression must not be empty")
  let (ast, end) = _parse(toks, 0, 0)
  if end != toks.len() { panic("autodiff.expr: unexpected trailing token near " + repr(toks.at(end))) }
  vars => _eval-ast(ast, vars, names)
}

// ── the computation graph, as data and as a drawing ─────────────────────────
#let _fmt(v) = {
  let r = calc.round(v, digits: 3)
  if r == calc.round(r) { str(int(r)) } else { str(r) }
}
// walk the built graph, assigning each node an id (pre-order root=0), collecting
// (id, label, value, kids=(child-id, local-deriv)) in post-order.
#let _walk(node, names, st) = {
  let my-id = st.next-id
  st.next-id += 1
  let kids = ()
  for pair in node.kids {
    let (cid, st2) = _walk(pair.at(0), names, st)
    st = st2
    kids.push((cid, pair.at(1)))
  }
  let label = if node.slot != none { names.at(node.slot) }
    else if node.op == "const" { _fmt(node.v) } else { node.op }
  st.nodes.push((id: my-id, label: label, v: node.v, kids: kids, input: node.slot != none, leaf: node.kids.len() == 0))
  (my-id, st)
}
// trace(f, x) → the graph as plain data: one entry per node with its value, the
// EXACT adjoint (∂output/∂node), the op label, its layer (depth), and child edges.
#let trace(f, x, names: none) = {
  _point(f, x, "trace")
  assert(names == none or (type(names) == array and names.len() == x.len() and
    names.all(n => type(n) == str)),
    message: "autodiff.trace: names must match x and contain strings")
  let nm = if names == none { range(x.len()).map(i => "x" + str(i)) } else { names }
  let out = _lift(f(x.enumerate().map(((i, xi)) => var(i, xi))))
  let (root, st) = _walk(out, nm, (next-id: 0, nodes: ()))
  let n = st.nodes.len()
  let byid = (:)
  for nd in st.nodes { byid.insert(str(nd.id), nd) }
  // adjoints: seed root (id 0) = 1, push down in id order (root first = reverse topo)
  let adj = range(n).map(_ => 0.0)
  adj.at(0) = 1.0
  for id in range(n) {
    for (cid, local) in byid.at(str(id)).kids { adj.at(cid) = adj.at(cid) + adj.at(id) * local }
  }
  // layer = longest path from a leaf (st.nodes is post-order → children seen first)
  let layer = (:)
  for nd in st.nodes {
    layer.insert(str(nd.id), if nd.kids.len() == 0 { 0 } else { 1 + calc.max(..nd.kids.map(k => layer.at(str(k.at(0)))) ) })
  }
  st.nodes.map(nd => (id: nd.id, label: nd.label, value: nd.v, grad: adj.at(nd.id),
    kids: nd.kids, input: nd.input, leaf: nd.leaf, layer: layer.at(str(nd.id))))
}
// graph(f, x) → a drawn computation graph (fletcher): op + value in each node,
// and (show-grad) the exact adjoint below it — the whole forward+backward pass,
// auto-derived. Layered left→right by depth, rows by child barycentre.
#let graph(
  f, x, names: none, show-grad: true, theme: default-theme, spacing: (17mm, 9mm),
) = {
  let t = theme
  let nodes = trace(f, x, names: names)
  let byid = (:)
  for nd in nodes { byid.insert(str(nd.id), nd) }
  let maxlayer = calc.max(..nodes.map(nd => nd.layer))
  // barycentre rows: leaves stacked, internal node row = mean of its children's
  let row = (:)
  let leafn = 0
  for L in range(maxlayer + 1) {
    for nd in nodes.filter(nd => nd.layer == L) {
      if nd.leaf { row.insert(str(nd.id), leafn * 1.0); leafn += 1 }
      else { let cr = nd.kids.map(k => row.at(str(k.at(0)))); row.insert(str(nd.id), cr.sum() / cr.len()) }
    }
  }
  let pos(nd) = (nd.layer, -row.at(str(nd.id)))
  align(center, diagram(spacing: spacing, {
    for nd in nodes {
      let (fill, stroke) = if nd.input { (white, 0.9pt + t.ink) }
        else if nd.id == 0 { (t.accent.lighten(78%), 0.9pt + t.accent) }
        else { (t.accent2.lighten(82%), 0.9pt + t.accent2) }
      node(pos(nd), {
        set align(center)
        stack(spacing: 2pt,
          text(size: 9.5pt, weight: 600, fill: t.ink, nd.label),
          text(size: 8pt, fill: t.muted, "= " + _fmt(nd.value)),
          if show-grad { text(size: 8pt, fill: t.accent, "∂ " + _fmt(nd.grad)) },
        )
      }, fill: fill, stroke: stroke, corner-radius: 2pt, inset: 5pt)
    }
    for nd in nodes {
      for (cid, _) in nd.kids { edge(pos(byid.at(str(cid))), pos(nd), "-|>", stroke: 0.7pt + t.muted) }
    }
  }))
}
