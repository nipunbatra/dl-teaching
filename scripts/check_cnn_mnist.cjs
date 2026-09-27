const assert = require('node:assert/strict');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const D = require(path.join(root, 'interactives/cnn/evidence/mnist.json'));
const L = require(path.join(root, 'interactives/cnn/lenet.js'));
assert.equal(D.parameters, 44426);
assert.equal(Object.values(D.checkpoints['10']).reduce((n, v) => n + v.length, 0), 44426);
assert.equal(D.test.total, 10000);
assert.equal(D.confusion.flat().reduce((a, b) => a + b), 10000);
assert.equal(D.confusion.reduce((n, row, i) => n + row[i], 0), D.test.correct);
let maxError = 0;
for (const sample of D.samples) {
  const a = L.forward(D.checkpoints['10'], sample.pixels);
  for (const [name, expected] of Object.entries(sample.checks)) {
    assert.equal(a[name].length, expected.shape.reduce((n, x) => n * x, 1));
    expected.first.forEach((v, i) => { maxError = Math.max(maxError, Math.abs(a[name][i] - v)); });
    const sum = a[name].reduce((x, y) => x + y, 0);
    assert.ok(Math.abs(sum - expected.sum) < Math.max(.02, Math.abs(expected.sum) * 2e-6), name);
  }
  sample.logits.forEach((v, i) => { maxError = Math.max(maxError, Math.abs(a.logits[i] - v)); });
  assert.equal(a.prob.indexOf(Math.max(...a.prob)), sample.prediction);
  assert.ok(Math.abs(a.prob.reduce((x, y) => x + y) - 1) < 1e-12);
}
assert.ok(maxError < 1e-4, `Browser/PyTorch mismatch ${maxError}`);
assert.ok(D.history.at(-1).validation.loss < D.history[0].validation.loss);
console.log(JSON.stringify({passed:true, samples:D.samples.length, parameters:D.parameters, maxActivationError:maxError, testCorrect:D.test.correct, testTotal:D.test.total}, null, 2));
