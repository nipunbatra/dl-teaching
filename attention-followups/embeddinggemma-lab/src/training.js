import { unit, makeHead, headMetrics, trainHeadStep } from "./math.js";
import { escapeHTML as esc, saveJSON } from "./explorer.js";
import { MODEL, REVISION } from "./config.js";
export function mountTraining(root, state) {
  let head,
    epoch = 0,
    history = [],
    training = false,
    alive = true,
    classes = [],
    train = [],
    test = [],
    d = 768;
  const $ = (s) => root.querySelector(s);
  root.innerHTML = `<div class="training-intro"><p><strong>Can a few labelled sounds teach a classifier?</strong> EmbeddingGemma has already represented each recording. Now learn a small output layer on those fixed vectors.</p><span class="saved-tag">Real gradient descent · runs locally · no model download</span></div><div class="training-architecture"><div><b>Sound recording</b><span>16 kHz mono waveform</span></div><i>→</i><div class="frozen-block"><b>EmbeddingGemma 2</b><span>Frozen · does not change</span></div><i>→</i><div><b id="train-vector-shape">768 numbers</b><span>L2-normalized embedding</span></div><i>→</i><div class="learning-block"><b>Linear layer + softmax</b><span>W and b learn from labels</span></div></div>
 <div class="training-controls"><label>Classification task<select id="train-task"><option value="3">3 sounds · dog, waves, fire</option><option value="10">All 10 sound categories</option></select></label><label>Embedding dimensions<select id="train-dim"><option>768</option><option>256</option><option>128</option></select></label><div class="training-buttons"><button id="train-step" class="quiet">Take one step</button><button id="train-run" class="primary">Train 100 steps</button><button id="train-reset" class="quiet">Reset</button></div></div>
 <div class="training-metrics" id="train-metrics" aria-live="polite"></div><div class="training-grid"><section><h3>Watch the loss change</h3><div id="training-chart"></div><p class="hint">Solid rust: training cross-entropy · dashed blue: held-out cross-entropy. A lower loss means more probability assigned to the recorded label. Training loss can fall while held-out predictions get worse.</p><div id="train-formula" class="training-formula"></div></section><section><h3>Exactly what is learning?</h3><p>Start W and b at zero: every class gets the same probability. Each step computes the training loss, differentiates it, and updates only W and b.</p><pre class="training-code">x = frozen_embeddings          # [N, d]
logits = x @ W.T + b            # [N, C]
loss = cross_entropy(logits, y)
loss.backward()                # gradients for W, b
optimizer.step()</pre><p id="train-shape" class="hint"></p><details><summary>Learning rate and update rule</summary><p>Full-batch gradient descent · learning rate 2 · weight penalty λ = 0.001.</p><p>∂L/∂s = (p − one_hot(y)) / N<br>∂L/∂W = (p − Y)ᵀX / N + λW<br>W ← W − 2 × ∂L/∂W</p><p>The chart shows cross-entropy alone. The small weight penalty is used in the update.</p></details><button class="quiet" id="train-export">Download learned weights</button></section></div>
 <section class="test-predictions"><div class="section-heading"><h3>Try the held-out recordings</h3><p>These recordings never enter the gradient update. Play them before revealing their labels.</p></div><p class="hint" id="split-note"></p><div id="test-cards"></div></section>
 <details class="training-split"><summary>See every training example and the split</summary><div id="split-table"></div></details>
 <section class="finetune-section"><p class="eyebrow">THE NEXT STEP</p><h3>What if we want to change the embedding space itself?</h3><p>Fine-tuning updates the encoder. Use positive matches and in-batch negatives so related inputs move closer in the representation. That is a different objective from our small classifier above.</p><div class="fine-tuning-path"><span>Image + caption<br>or audio + text pairs</span><b>→</b><span>EmbeddingGemma<br><small>weights can update</small></span><b>→</b><span>Contrastive loss<br><small>compare candidates in a batch</small></span></div><p>The official notebooks below run model fine-tuning on a GPU runtime. This browser lab only trains the small output layer.</p><div class="training-links"><a href="https://ai.google.dev/gemma/docs/embeddinggemma/fine-tuning-embeddinggemma-with-sentence-transformers" target="_blank" rel="noopener">Google: text fine-tuning ↗</a><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/EmbeddingGemma2_%28300M%29-Image_Text.ipynb" target="_blank" rel="noopener">Image + text Colab ↗</a><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/EmbeddingGemma2_%28300M%29-Audio.ipynb" target="_blank" rel="noopener">Audio Colab ↗</a></div></section>`;
  function reset() {
    epoch = 0;
    history = [];
    d = +$("#train-dim").value;
    classes =
      $("#train-task").value === "3"
        ? ["dog", "sea_waves", "crackling_fire"]
        : [
            ...new Set(
              state.gallery.filter((x) => x.label).map((x) => x.label),
            ),
          ];
    const data = state.gallery.filter(
      (x) =>
        x.type === "audio" && classes.includes(x.label) && state.vectors[x.id],
    );
    train = data.filter((x) => x.split === "train");
    test = data.filter((x) => x.split === "test");
    head = makeHead(d, classes.length);
    record();
    render();
  }
  const rows = (data) => data.map((x) => unit(state.vectors[x.id].vector, d)),
    labels = (data) => data.map((x) => classes.indexOf(x.label));
  function record() {
    history.push({
      epoch,
      train: headMetrics(head, rows(train), labels(train)),
      test: headMetrics(head, rows(test), labels(test)),
    });
  }
  function chart() {
    const width = 680,
      height = 250,
      max =
        Math.max(
          Math.log(classes.length),
          ...history.flatMap((h) => [h.train.loss, h.test.loss]),
        ) * 1.12,
      end = Math.max(100, epoch);
    const path = (type) =>
      history
        .map(
          (h, i) =>
            `${i ? "L" : "M"}${45 + (h.epoch / end) * 610},${218 - (h[type].loss / max) * 190}`,
        )
        .join(" ");
    $("#training-chart").innerHTML =
      `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Training and held-out cross-entropy over ${epoch} steps"><path d="M45 20V218H660" fill="none" stroke="#d0cec6"/><path d="M45 ${218 - (Math.log(classes.length) / max) * 190}H660" stroke="#c5c2b8" stroke-dasharray="3 4"/><text x="50" y="${211 - (Math.log(classes.length) / max) * 190}">uniform guess: ln(${classes.length}) = ${Math.log(classes.length).toFixed(3)}</text><path d="${path("train")}" fill="none" stroke="#bc5538" stroke-width="3"/><path d="${path("test")}" fill="none" stroke="#476b9b" stroke-width="2.5" stroke-dasharray="6 4"/><text x="12" y="220">0</text><text x="43" y="242">0</text><text x="582" y="242">${end} steps</text></svg>`;
  }
  function render() {
    const m = history.at(-1);
    $("#train-metrics").innerHTML =
      `<div><span>Gradient steps</span><b>${epoch}</b></div><div><span>Training loss</span><b>${m.train.loss.toFixed(4)}</b></div><div><span>Held-out loss</span><b>${m.test.loss.toFixed(4)}</b></div><div><span>Held-out accuracy</span><b>${Math.round(m.test.accuracy * test.length)} / ${test.length}</b></div>`;
    chart();
    $("#train-vector-shape").textContent = d + " numbers";
    $("#train-shape").textContent =
      `X: [${train.length}, ${d}] · W: [${classes.length}, ${d}] · b: [${classes.length}] · ${classes.length * (d + 1)} trainable parameters.`;
    $("#train-formula").innerHTML =
      `<b>L = −(1/N) Σ log p(correct class)</b><span>N = ${train.length} training recordings · ${classes.length} classes · uniform baseline = ${Math.log(classes.length).toFixed(4)}</span>`;
    $("#split-note").textContent =
      `3 training recordings + 1 held-out recording per class, all from different original source recordings. Only ${test.length} held-out examples: an inspectable demonstration, not a reliable performance estimate. Repeatedly choosing settings against this tiny test set would bias the result.`;
    $("#test-cards").innerHTML = test
      .map((item, i) => {
        const probs = m.test.predictions[i],
          best = probs.indexOf(Math.max(...probs));
        return `<article><p class="small-label">HELD-OUT RECORDING ${i + 1}</p><audio controls preload="none" src="${esc(item.src)}" aria-label="Held-out recording ${i + 1}"></audio><strong>Prediction: ${esc(classes[best].replaceAll("_", " "))}</strong><p class="hint">Softmax output · not calibrated confidence</p><div class="class-bars">${classes.map((c, j) => `<div><span>${esc(c.replaceAll("_", " "))}</span><i><b style="width:${probs[j] * 100}%"></b></i><code>${(probs[j] * 100).toFixed(1)}%</code></div>`).join("")}</div><details><summary>Reveal the dataset label</summary><p>${esc(item.label.replaceAll("_", " "))} · ${classes[best] === item.label ? "Correct prediction" : "Different from the prediction"}</p></details></article>`;
      })
      .join("");
    $("#split-table").innerHTML =
      `<p>Original source IDs are kept separate across the split. Audio never uses filenames as model input.</p><div class="table-scroll"><table><thead><tr><th>Recording</th><th>Listen</th><th>Split</th><th>Source recording ID</th></tr></thead><tbody>${[...train, ...test].map((i) => `<tr><td>${esc(i.title)}</td><td><audio controls preload="none" src="${esc(i.src)}"></audio></td><td>${i.split === "test" ? "Held out" : "Training"}</td><td>${esc(i.sourceRecording)}</td></tr>`).join("")}</tbody></table></div>`;
  }
  function step() {
    trainHeadStep(head, rows(train), labels(train));
    epoch++;
    record();
  }
  $("#train-step").onclick = () => {
    step();
    render();
  };
  $("#train-reset").onclick = reset;
  $("#train-task").onchange = reset;
  $("#train-dim").onchange = reset;
  $("#train-run").onclick = async () => {
    if (training) return;
    training = true;
    root
      .querySelectorAll(".training-controls button,.training-controls select")
      .forEach((n) => (n.disabled = true));
    const x = rows(train),
      y = labels(train);
    for (let n = 0; n < 100 && alive; n++) {
      trainHeadStep(head, x, y);
      epoch++;
      record();
      if (n % 10 === 9) {
        render();
        await new Promise((resolve) => requestAnimationFrame(resolve));
      }
    }
    training = false;
    if (alive) {
      render();
      root
        .querySelectorAll(".training-controls button,.training-controls select")
        .forEach((n) => (n.disabled = false));
    }
  };
  $("#train-export").onclick = () =>
    saveJSON(
      {
        model: MODEL,
        revision: REVISION,
        frozenEncoder: true,
        dimensions: d,
        classes,
        head,
        steps: epoch,
        learningRate: 2,
        weightPenalty: 0.001,
        trainIds: train.map((i) => i.id),
        heldOutIds: test.map((i) => i.id),
        history: history.map((h) => ({
          step: h.epoch,
          trainLoss: h.train.loss,
          heldOutLoss: h.test.loss,
          heldOutAccuracy: h.test.accuracy,
        })),
      },
      "trained-linear-head.json",
    );
  reset();
  return () => {
    alive = false;
  };
}
