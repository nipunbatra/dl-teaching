import assert from 'node:assert/strict';
import fs from 'node:fs';
import { TRAINING_TASKS, trainingTask } from '../src/training-data.js';

import { neuronTrace, unit, makeHead, headMetrics, headPredict, trainHeadStep } from '../src/math.js';
const gallery = JSON.parse(fs.readFileSync(new URL('../public/gallery.json', import.meta.url)));
const vectors = JSON.parse(fs.readFileSync(new URL('../public/embeddings.json', import.meta.url))).items;
const original = JSON.stringify(vectors), summary = [];
for (const option of TRAINING_TASKS) {
  const task = trainingTask(option.id, gallery, vectors);
  const sourceKey = x => x.type === 'audio' ? x.sourceRecording : x.sourceGroup;
  const sourceIds = new Set(task.train.map(sourceKey));
  assert(task.test.every(x => !sourceIds.has(sourceKey(x))), `${option.id}: source leak`);
  const contentKey = x => x.type === 'text' ? x.text.trim().toLowerCase() : x.sha256;
  const trainContent = new Set(task.train.map(contentKey));
  assert(!trainContent.has(undefined), 'Missing content provenance');
  assert(task.test.every(x => !trainContent.has(contentKey(x))), `${option.id}: duplicated content`);
  assert(task.train.concat(task.test).every(x => x.type === option.type && vectors[x.id].vector.length === 768));
  for (const c of task.classes) {
    assert.equal(task.train.filter(x => x.label === c).length, option.type === 'audio' ? 3 : 4);
    assert.equal(task.test.filter(x => x.label === c).length, option.type === 'audio' ? 1 : 2);
  }
  // Audio gradient training is already exercised by exploration.test.mjs.
  if (option.type === 'audio') continue;
  for (const d of [768, 256, 128]) {
    const rows = task.train.map(x => unit(vectors[x.id].vector, d));
    const labels = task.train.map(x => task.classes.indexOf(x.label));
    const head = makeHead(d, task.classes.length);
    const initial = headMetrics(head, rows, labels);
    assert(Math.abs(initial.loss - Math.log(task.classes.length)) < 1e-12);
    for (let i = 0; i < 100; i++) trainHeadStep(head, rows, labels);
    const final = headMetrics(head, rows, labels);
    assert(final.loss < initial.loss, `${option.id}/${d}: loss must fall`);
    for (const item of [...task.train, ...task.test]) {
      const x = unit(vectors[item.id].vector, d), predicted = headPredict(head, x);
      for (let c = 0; c < task.classes.length; c++) {
        const trace = neuronTrace(head, x, c);
        // Independently reconstruct every displayed score from the detailed terms.
        const score = trace.products.slice(0, 3).reduce((a, b) => a + b, 0) + trace.remainder + head.bias[c];
        assert(Math.abs(score - trace.logits[c]) < 1e-10);
        assert(Math.abs(trace.probabilities[c] - predicted[c]) < 1e-12);
        assert(Math.abs(trace.probabilities.reduce((a, b) => a + b, 0) - 1) < 1e-12);
        assert(trace.logits.every(Number.isFinite));
      }
    }
    const held = headMetrics(head, task.test.map(x => unit(vectors[x.id].vector, d)), task.test.map(x => task.classes.indexOf(x.label)));
    summary.push({task: option.id, dimensions: d, train: task.train.length, heldOut: task.test.length, initialLoss: initial.loss, trainLoss: final.loss, heldOutLoss: held.loss, heldOutCorrect: Math.round(held.accuracy * task.test.length)});
  }
}
assert.equal(JSON.stringify(vectors), original, 'Training must not modify encoder features');
fs.mkdirSync(new URL('../output/verification/', import.meta.url), {recursive:true});
fs.writeFileSync(new URL('../output/verification/multimodal-training.json', import.meta.url), JSON.stringify(summary, null, 2)+'\n');
console.log('All four task splits, six image/text training runs and all neuron calculations passed.');
console.table(summary);
