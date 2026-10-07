export function unit(vector, dimensions = vector.length) {
  const v = Array.from(vector).slice(0, dimensions);
  if (!v.length || !v.every(Number.isFinite))
    throw Error("Expected a finite, non-empty vector.");
  const norm = Math.hypot(...v);
  if (norm < 1e-10)
    throw Error("These vectors have almost no difference to compare.");
  return v.map((x) => x / norm);
}
export function dot(a, b) {
  if (a.length !== b.length)
    throw Error("Both vectors must have the same dimension.");
  return a.reduce((sum, x, i) => sum + x * b[i], 0);
}
export function rank(query, items, vectors, dimensions = 768, excluded = []) {
  const q = unit(query, dimensions);
  return items
    .filter((item) => vectors[item.id] && !excluded.includes(item.id))
    .map((item) => ({
      item,
      score: dot(q, unit(vectors[item.id].vector, dimensions)),
    }))
    .sort((a, b) => b.score - a.score || a.item.id.localeCompare(b.item.id));
}
export function difference(after, before) {
  return unit(after.map((v, i) => v - before[i]));
}
// Deterministic k-means on unit vectors; the labels are arbitrary group IDs.
export function kmeans(rows, k = 4) {
  const centers = [rows[0].slice()];
  while (centers.length < k) {
    let best = 0,
      score = -Infinity;
    rows.forEach((r, i) => {
      const d = Math.min(
        ...centers.map((c) => r.reduce((s, x, j) => s + (x - c[j]) ** 2, 0)),
      );
      if (d > score) {
        best = i;
        score = d;
      }
    });
    centers.push(rows[best].slice());
  }
  let labels = [];
  for (let iteration = 0; iteration < 30; iteration++) {
    const next = rows.map((r) =>
      centers
        .map((c) => r.reduce((s, x, j) => s + (x - c[j]) ** 2, 0))
        .reduce((best, d, i, a) => (d < a[best] ? i : best), 0),
    );
    if (next.every((v, i) => v === labels[i])) break;
    labels = next;
    centers.forEach((c, g) => {
      const members = rows.filter((_, i) => labels[i] === g);
      if (members.length)
        centers[g] = c.map(
          (_, j) => members.reduce((s, r) => s + r[j], 0) / members.length,
        );
    });
  }
  return labels;
}
// Centered PCA via power iteration; distances in this 2D projection are approximate.
export function pca(rows) {
  const d = rows[0].length,
    mean = rows[0].map(
      (_, j) => rows.reduce((s, r) => s + r[j], 0) / rows.length,
    ),
    x = rows.map((r) => r.map((v, j) => v - mean[j]));
  const axes = [];
  for (let a = 0; a < 2; a++) {
    let v = unit(
      Array.from({ length: d }, (_, i) => Math.sin((i + 1) * (a + 1.137))),
    );
    for (let it = 0; it < 60; it++) {
      let next = Array(d).fill(0);
      x.forEach((r) => {
        const projection = dot(r, v);
        r.forEach((z, j) => (next[j] += z * projection));
      });
      axes.forEach((axis) => {
        const component = dot(next, axis);
        next = next.map((z, j) => z - component * axis[j]);
      });
      if (Math.hypot(...next) < 1e-10) break;
      v = unit(next);
    }
    axes.push(v);
  }
  return x.map((r) => axes.map((a) => dot(r, a)));
}
