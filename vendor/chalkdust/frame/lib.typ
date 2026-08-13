// frame — a tiny data-frame for Typst.
//
// The maintainability idea: teaching data should live in a FILE (or a computed
// array), not be hand-typed into a figure. Load it once, pick columns by name,
// filter/mutate, and hand columns to plot. The plot then reflects the data —
// it cannot silently disagree with the numbers the way a pasted SVG can.
//
//   #import "vendor/chalkdust/frame/lib.typ" as md
//   #import "vendor/chalkdust/plot/lib.typ" as mp
//   #let f = md.frame(csv("runs.csv"))            // header row + data rows
//   #mp.lines(md.xy(f, "epoch", ("adam", "sgd")), labels: ("Adam", "SGD"))

// coerce a cell to a number (csv() yields strings; arrays may already be numeric)
#let num(v) = if type(v) in (int, float) { v } else { float(str(v).trim()) }
#let _unique(xs) = xs.enumerate().all(((i, x)) => xs.position(y => y == x) == i)
#let _check-frame(f, op) = {
  let prefix = "frame." + op + ": "
  assert(type(f) == dictionary and "cols" in f and "rows" in f,
    message: prefix + "f must be a frame with cols and rows")
  assert(type(f.cols) == array and f.cols.all(c => type(c) == str) and _unique(f.cols),
    message: prefix + "frame cols must be an array of unique strings")
  assert(type(f.rows) == array and f.rows.all(r => type(r) == dictionary),
    message: prefix + "frame rows must be an array of dictionaries")
  assert(f.rows.all(r => r.len() == f.cols.len() and f.cols.all(c => c in r)),
    message: prefix + "every row must match the frame columns exactly")
}
#let _has-col(f, name, op) = {
  _check-frame(f, op)
  assert(f.cols.contains(name),
    message: "frame." + op + ": unknown column '" + str(name) + "'")
}

// Build a frame from:
//   • an array of arrays (rows) — first row is the header unless `header:` is given
//   • the result of Typst's built-in csv(path) (array of string rows)
//   • an array of dictionaries (already keyed by column)
// A frame is `(cols: (name…), rows: (dict-per-row…))`.
#let frame(source, header: auto) = {
  assert(type(source) == array, message: "frame.frame: source must be an array")
  if source.len() == 0 {
    if header == auto { return (cols: (), rows: ()) }
    assert(type(header) == array and header.len() > 0,
      message: "frame.frame: header must be a non-empty array")
    assert(header.all(h => type(h) == str), message: "frame.frame: column names must be strings")
    assert(_unique(header), message: "frame.frame: column names must be unique")
    return (cols: header, rows: ())
  }
  if type(source.first()) == dictionary {
    assert(source.all(r => type(r) == dictionary), message: "frame.frame: dictionary rows cannot be mixed with array rows")
    let cols = if header == auto { source.first().keys() } else {
      assert(type(header) == array and header.len() > 0,
        message: "frame.frame: header must be a non-empty array")
      assert(header.all(h => type(h) == str), message: "frame.frame: column names must be strings")
      assert(_unique(header), message: "frame.frame: column names must be unique")
      header
    }
    assert(source.all(r => r.len() == cols.len() and cols.all(c => c in r)),
      message: "frame.frame: every dictionary row must contain all columns and have no extra columns")
    return (cols: cols, rows: source)
  }
  assert(source.all(r => type(r) == array), message: "frame.frame: every row must be an array")
  let hdr = if header == auto { source.first() } else { header }
  assert(type(hdr) == array and hdr.len() > 0, message: "frame.frame: header must be a non-empty array")
  assert(hdr.all(h => type(h) == str), message: "frame.frame: column names must be strings")
  assert(_unique(hdr), message: "frame.frame: column names must be unique")
  let body = if header == auto { source.slice(1) } else { source }
  assert(body.all(r => r.len() == hdr.len()), message: "frame.frame: each row must match the header length")
  (cols: hdr, rows: body.map(r => {
    let d = (:)
    for (i, h) in hdr.enumerate() { d.insert(h, r.at(i)) }
    d
  }))
}

#let nrow(f) = { _check-frame(f, "nrow"); f.rows.len() }
#let head(f, n: 5) = {
  _check-frame(f, "head")
  assert(type(n) == int and n >= 0, message: "frame.head: n must be a non-negative integer")
  (cols: f.cols, rows: f.rows.slice(0, calc.min(n, f.rows.len())))
}

// ── ecosystem interchange ──
// Row dictionaries and column dictionaries are the two common small-table
// conventions in Typst packages. Keep the frame itself tiny, and bridge at the
// boundary rather than forcing another table package to depend on this one.
#let from-records(rows, header: auto) = frame(rows, header: header)
#let to-records(f) = { _check-frame(f, "to-records"); f.rows }
#let records = to-records

#let from-columns(columns) = {
  assert(type(columns) == dictionary,
    message: "frame.from-columns: columns must be a dictionary of arrays")
  let cols = columns.keys()
  assert(columns.values().all(values => type(values) == array),
    message: "frame.from-columns: every column must be an array")
  if cols.len() == 0 { return (cols: (), rows: ()) }
  let lengths = columns.values().map(values => values.len())
  assert(lengths.all(n => n == lengths.first()),
    message: "frame.from-columns: all columns must have the same length")
  let rows = range(lengths.first()).map(i => {
    let row = (:)
    for col in cols { row.insert(col, columns.at(col).at(i)) }
    row
  })
  (cols: cols, rows: rows)
}

#let to-columns(f) = {
  _check-frame(f, "to-columns")
  let columns = (:)
  for col in f.cols { columns.insert(col, f.rows.map(row => row.at(col))) }
  columns
}
#let columns = to-columns

// a column as an array; `map:` coerces each cell (default num; pass none to keep raw, or str/int)
#let col(f, name, map: num) = {
  _has-col(f, name, "col")
  assert(map == none or type(map) == function, message: "frame.col: map must be a function or none")
  f.rows.map(r => if map == none { r.at(name) } else { map(r.at(name)) })
}

#let filter(f, pred) = {
  _check-frame(f, "filter")
  assert(type(pred) == function, message: "frame.filter: pred must be a function")
  let rows = ()
  for r in f.rows {
    let keep = pred(r)
    assert(type(keep) == bool, message: "frame.filter: predicate must return a boolean")
    if keep { rows.push(r) }
  }
  (cols: f.cols, rows: rows)
}

#let select(f, ..names) = {
  _check-frame(f, "select")
  let ns = names.pos()
  assert(_unique(ns), message: "frame.select: selected columns must be unique")
  for n in ns { _has-col(f, n, "select") }
  (cols: ns, rows: f.rows.map(r => { let d = (:); for n in ns { d.insert(n, r.at(n)) }; d }))
}

// add or replace a computed column: mutate(f, "gap", r => num(r.test) - num(r.train))
#let mutate(f, name, fn) = {
  _check-frame(f, "mutate")
  assert(type(name) == str, message: "frame.mutate: name must be a string")
  assert(type(fn) == function, message: "frame.mutate: fn must be a function")
  (cols: if f.cols.contains(name) { f.cols } else { f.cols + (name,) },
    rows: f.rows.map(r => { let d = r; d.insert(name, fn(r)); d }))
}

// group rows by a column and aggregate another (default: mean)
#let _mean(xs) = xs.sum() / xs.len()
#let group-agg(f, by, value, fn: _mean, name: "agg") = {
  _has-col(f, by, "group-agg"); _has-col(f, value, "group-agg")
  assert(type(fn) == function, message: "frame.group-agg: fn must be a function")
  assert(type(name) == str, message: "frame.group-agg: output name must be a string")
  assert(name != by, message: "frame.group-agg: output name must differ from the grouping column")
  let keys = ()
  for r in f.rows { if not keys.contains(r.at(by)) { keys.push(r.at(by)) } }
  (cols: (by, name), rows: keys.map(k => {
    let vals = f.rows.filter(r => r.at(by) == k).map(r => num(r.at(value)))
    let d = (:); d.insert(by, k); d.insert(name, fn(vals)); d
  }))
}

// ── bridges to plot ──
// (x, y) pairs for lines(); if `y` is an array of names → a list of series (multi)
#let xy(f, x, y) = {
  _has-col(f, x, "xy")
  let xs = col(f, x)
  if type(y) == array {
    assert(y.len() > 0, message: "frame.xy: y column list must be non-empty")
    y.map(yc => { _has-col(f, yc, "xy"); xs.zip(col(f, yc)).map(((a, b)) => (a, b)) })
  } else { _has-col(f, y, "xy"); xs.zip(col(f, y)).map(((a, b)) => (a, b)) }
}
// (label, value) pairs for bars(): a categorical column + a numeric column
#let bars-of(f, label, value) = {
  _has-col(f, label, "bars-of"); _has-col(f, value, "bars-of")
  f.rows.map(r => (r.at(label), num(r.at(value))))
}
