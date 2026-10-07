import assert from "node:assert/strict";
import fs from "node:fs";
import { unit, dot, rank, difference, kmeans, pca } from "../src/math.js";
const gallery = JSON.parse(
    fs.readFileSync(new URL("../public/gallery.json", import.meta.url)),
  ),
  data = JSON.parse(
    fs.readFileSync(new URL("../public/embeddings.json", import.meta.url)),
  );
assert.equal(new Set(gallery.map((i) => i.id)).size, gallery.length);
for (const item of gallery) {
  const v = data.items[item.id].vector;
  assert.equal(v.length, 768);
  assert(v.every(Number.isFinite));
  assert(Math.abs(Math.hypot(...v) - 1) < 1e-5);
  if (item.src)
    assert(fs.existsSync(new URL("../public/" + item.src, import.meta.url)));
  for (const d of [128, 256, 512, 768])
    assert(Math.abs(Math.hypot(...unit(v, d)) - 1) < 1e-12);
}
const before = data.items.portrait.vector,
  after = data.items["portrait-hat"].vector;
const delta = difference(after, before),
  reverse = difference(before, after);
delta.forEach((v, i) => assert(Math.abs(v + reverse[i]) < 1e-12));
assert.throws(() => difference(before, before));
assert.throws(() => dot([1], [1, 2]));
assert.equal(
  rank(data.items.coffee.vector, gallery, data.items)[0].item.id,
  "coffee",
);
assert.deepEqual(
  kmeans(
    [
      [1, 0],
      [0.9, 0.1],
      [-1, 0],
      [-0.9, -0.1],
    ],
    2,
  ),
  [0, 0, 1, 1],
);
assert(
  pca([
    [1, 0],
    [0, 1],
    [-1, 0],
    [0, -1],
  ])
    .flat()
    .every(Number.isFinite),
);
assert(Math.abs(dot(unit([3, 4]), unit([0, 1])) - 0.8) < 1e-12);
console.log(
  `Passed: ${gallery.length} real embeddings, assets, unit norms, truncation, dot products, delta sign reversal, clustering and PCA.`,
);
