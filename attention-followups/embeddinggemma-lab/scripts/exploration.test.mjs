import assert from "node:assert/strict";
import fs from "node:fs";
import crypto from "node:crypto";
import {
  unit,
  pcaProjection,
  makeHead,
  headPredict,
  headMetrics,
  trainHeadStep,
} from "../src/math.js";
const gallery = JSON.parse(
  fs.readFileSync(new URL("../public/gallery.json", import.meta.url)),
);
const data = JSON.parse(
  fs.readFileSync(new URL("../public/embeddings.json", import.meta.url)),
).items;
// Check PCA variance accounting against a known diagonal covariance.
const projection = pcaProjection([
  [3, 0],
  [-3, 0],
  [0, 1],
  [0, -1],
]);
assert(Math.abs(projection.variance[0] - 0.9) < 1e-8);
assert(Math.abs(projection.variance[1] - 0.1) < 1e-8);
const real = pcaProjection(gallery.map((x) => data[x.id].vector));
assert(real.points.flat().every(Number.isFinite));
assert(real.variance.reduce((a, b) => a + b) <= 1 + 1e-8);
// Independently check the implemented gradient by finite differences.
const h = makeHead(2, 2);
h.weights = [
  [0.12, -0.2],
  [0.3, 0.1],
];
h.bias = [0.1, -0.1];
const rows = [
    [0.6, 0.8],
    [-0.8, 0.6],
  ],
  y = [0, 1],
  epsilon = 1e-6,
  base = structuredClone(h);
for (let c = 0; c < 2; c++)
  for (let j = 0; j < 2; j++) {
    const plus = structuredClone(base),
      minus = structuredClone(base);
    plus.weights[c][j] += epsilon;
    minus.weights[c][j] -= epsilon;
    const numerical =
      (headMetrics(plus, rows, y).loss - headMetrics(minus, rows, y).loss) /
      (2 * epsilon);
    const actual = structuredClone(base);
    trainHeadStep(actual, rows, y, 1, 0);
    assert(
      Math.abs(numerical - (base.weights[c][j] - actual.weights[c][j])) < 1e-8,
    );
  }
for (let c = 0; c < 2; c++) {
  const plus = structuredClone(base),
    minus = structuredClone(base);
  plus.bias[c] += epsilon;
  minus.bias[c] -= epsilon;
  const numerical =
    (headMetrics(plus, rows, y).loss - headMetrics(minus, rows, y).loss) /
    (2 * epsilon);
  const actual = structuredClone(base);
  trainHeadStep(actual, rows, y, 1, 0);
  assert(Math.abs(numerical - (base.bias[c] - actual.bias[c])) < 1e-8);
}
const summary = [];
for (const classes of [
  ["dog", "sea_waves", "crackling_fire"],
  [...new Set(gallery.filter((x) => x.label).map((x) => x.label))],
]) {
  const train = gallery.filter(
      (x) => classes.includes(x.label) && x.split === "train",
    ),
    test = gallery.filter(
      (x) => classes.includes(x.label) && x.split === "test",
    );
  assert.equal(train.length, classes.length * 3);
  assert.equal(test.length, classes.length);
  const trainSources = new Set(train.map((x) => x.sourceRecording));
  assert(test.every((x) => !trainSources.has(x.sourceRecording)));
  for (const dim of [768, 256, 128]) {
    const x = train.map((i) => unit(data[i.id].vector, dim)),
      y = train.map((i) => classes.indexOf(i.label));
    const h = makeHead(dim, classes.length),
      initial = headMetrics(h, x, y).loss;
    assert(Math.abs(initial - Math.log(classes.length)) < 1e-12);
    for (let i = 0; i < 100; i++) trainHeadStep(h, x, y);
    const final = headMetrics(h, x, y),
      held = headMetrics(
        h,
        test.map((i) => unit(data[i.id].vector, dim)),
        test.map((i) => classes.indexOf(i.label)),
      );
    assert(final.loss < initial);
    assert(h.weights.flat().every(Number.isFinite));
    assert(Math.abs(headPredict(h, x[0]).reduce((a, b) => a + b) - 1) < 1e-12);
    summary.push({
      classes: classes.length,
      dim,
      initial,
      trainLoss: final.loss,
      heldOutLoss: held.loss,
      heldOutAccuracy: held.accuracy,
    });
  }
}
for (const item of gallery.filter((x) => x.src)) {
  const bytes = fs.readFileSync(
    new URL("../public/" + item.src, import.meta.url),
  );
  assert.equal(
    crypto.createHash("sha256").update(bytes).digest("hex"),
    item.sha256,
    `${item.id} asset changed after indexing`,
  );
}
fs.mkdirSync("output/verification", { recursive: true });
fs.writeFileSync(
  "output/verification/expanded-math.json",
  JSON.stringify(
    { items: gallery.length, pcaVariance: real.variance, training: summary },
    null,
    2,
  ),
);
console.log(
  "Passed: PCA variance, finite-difference gradients, six genuine training runs, source-separated held-out splits, all media hashes.",
);
console.table(summary);
