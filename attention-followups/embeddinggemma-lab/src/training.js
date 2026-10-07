import { unit, makeHead, headMetrics, trainHeadStep } from "./math.js";
import { escapeHTML as esc, saveJSON } from "./explorer.js";
import "./training-network.css";
import { waveform } from "./waveform.js";
import { TRAINING_TASKS, trainingTask, className } from "./training-data.js";
import { sampleMedia, networkDiagram, renderNeuronArithmetic } from "./training-network.js";
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
    d = 768,
    task;
  const $ = (s) => root.querySelector(s);
  root.innerHTML = `<div class="training-intro"><p><strong>Can a few labelled examples teach a new task?</strong> Choose sounds, pictures or text. EmbeddingGemma has already represented each input; we train a small classifier on those fixed vectors.</p><span class="saved-tag">Real gradient descent · runs locally · no model download</span></div>
 <div class="training-controls"><label>What are we classifying?<select id="train-task">${TRAINING_TASKS.map(t => `<option value="${t.id}">${esc(t.name)}</option>`).join("")}</select></label><label>Embedding dimensions<select id="train-dim"><option>768</option><option>256</option><option>128</option></select></label><div class="training-buttons"><button id="train-step" class="quiet">Take one step</button><button id="train-run" class="primary">Train 100 steps</button><button id="train-reset" class="quiet">Reset</button></div></div>
 <p id="training-task-note" class="training-task-note"></p>
 <section class="neuron-lab"><p class="eyebrow">INPUT → FEATURES → CLASS SCORES → PROBABILITIES</p><h3>Follow one example through the network</h3><p class="hint">Choose an example, then highlight a class neuron. Take a step above and watch its weights and predictions change.</p>
 <div class="nn-controls"><label>Training example<select id="train-example"></select></label><label>Highlight a class neuron<select id="train-neuron"></select></label></div>
 <div class="nn-layout"><div class="nn-input" id="nn-input"></div><div><p class="nn-pan-hint">Scroll the diagram sideways to follow all three stages →</p><div class="nn-scroll" id="nn-diagram" tabindex="0" role="region" aria-label="Feature and classifier neuron diagram"></div></div></div>
 <div class="nn-arithmetic" id="nn-arithmetic"></div><details class="nn-vector"><summary id="nn-vector-label">Inspect all 768 features</summary><p class="hint">These are the saved model’s actual coordinates for this input. Shorter vectors keep the first d coordinates and normalize again.</p><pre id="nn-vector-values"></pre></details></section>
 <div class="training-metrics" id="train-metrics" aria-live="polite"></div><div class="training-grid"><section><h3>Watch the loss change</h3><div id="training-chart"></div><p class="hint">Solid rust: training cross-entropy · dashed blue: held-out cross-entropy. A lower loss means more probability assigned to the recorded label. Training loss can fall while held-out predictions get worse.</p><div id="train-formula" class="training-formula"></div></section><section><h3>Exactly what is learning?</h3><p>Start W and b at zero: every class gets the same probability. Each step computes the training loss, differentiates it, and updates only W and b.</p><pre class="training-code">x = frozen_embeddings          # [N, d]
logits = x @ W.T + b            # [N, C]
loss = cross_entropy(logits, y)
loss.backward()                # gradients for W, b
optimizer.step()</pre><p id="train-shape" class="hint"></p><details><summary>Learning rate and update rule</summary><p>Full-batch gradient descent · learning rate 2 · weight penalty λ = 0.001.</p><p>∂L/∂s = (p − one_hot(y)) / N<br>∂L/∂W = (p − Y)ᵀX / N + λW<br>W ← W − 2 × ∂L/∂W</p><p>The chart shows cross-entropy alone. The small weight penalty is used in the update.</p></details><button class="quiet" id="train-export">Download learned weights</button></section></div>
 <section class="test-predictions"><div class="section-heading"><h3>Try the held-out examples</h3><p>These examples never enter the gradient update. Look, read or listen before revealing their labels.</p></div><p class="hint" id="split-note"></p><div id="test-cards"></div></section>
 <details class="training-split"><summary>See every training example and the split</summary><div id="split-table"></div></details>
 <section class="finetune-section"><p class="eyebrow">THE NEXT STEP</p><h3>What if we want to change the embedding space itself?</h3><p>Fine-tuning updates the encoder. Use positive matches and in-batch negatives so related inputs move closer in the representation. That is a different objective from our small classifier above.</p><div class="fine-tuning-path"><span>Image + caption<br>or audio + text pairs</span><b>→</b><span>EmbeddingGemma<br><small>weights can update</small></span><b>→</b><span>Contrastive loss<br><small>compare candidates in a batch</small></span></div><p>The official notebooks below run model fine-tuning on a GPU runtime. This browser lab only trains the small output layer.</p><div class="training-links"><a href="https://ai.google.dev/gemma/docs/embeddinggemma/fine-tuning-embeddinggemma-with-sentence-transformers" target="_blank" rel="noopener">Google: text fine-tuning ↗</a><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/EmbeddingGemma2_%28300M%29-Image_Text.ipynb" target="_blank" rel="noopener">Image + text Colab ↗</a><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/EmbeddingGemma2_%28300M%29-Audio.ipynb" target="_blank" rel="noopener">Audio Colab ↗</a></div></section>`;
  function reset() {
    epoch = 0;
    history = [];
    d = +$("#train-dim").value;
    task = trainingTask($("#train-task").value, state.gallery, state.vectors);
    ({ classes, train, test } = task);
    $("#train-example").innerHTML = train.map((x, i) => `<option value="${x.id}">${i + 1}. ${esc(className(x.label))} · ${esc(x.title)}</option>`).join("");
    $("#train-neuron").innerHTML = classes.map((c, i) => `<option value="${i}">${esc(className(c))}</option>`).join("");
    $("#training-task-note").textContent = task.type === "audio"
      ? "Recognise the sound from the waveform alone. The recorded class labels supervise the output layer; filenames are never model inputs."
      : task.type === "image"
        ? "Sort pictures into Animals, Food & drink, or Transport. The encoder saw the pixels, without captions. Four pictures per class teach the head; two different pictures per class are held out."
        : "Sort written descriptions into Animals, Food & drink, or Transport. The encoder saw only the text, without pictures. Four descriptions per class teach the head; two different descriptions per class are held out.";
    head = makeHead(d, classes.length);
    record();
    renderExamples();
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
      `<div><span>Gradient steps</span><b>${epoch}</b></div><div><span>Training loss</span><b>${m.train.loss.toFixed(4)}</b></div><div><span>Held-out loss</span><b>${m.test.loss.toFixed(4)}</b></div><div><span>Held-out accuracy</span><b>${epoch === 0 ? "— (tie)" : `${Math.round(m.test.accuracy * test.length)} / ${test.length}`}</b></div>`;
    chart();
    renderNetwork();
    $("#train-shape").textContent =
      `X: [${train.length}, ${d}] · W: [${classes.length}, ${d}] · b: [${classes.length}] · ${classes.length * (d + 1)} trainable parameters.`;
    $("#train-formula").innerHTML =
      `<b>L = −(1/N) Σ log p(correct class)</b><span>N = ${train.length} training examples · ${classes.length} classes · uniform baseline = ${Math.log(classes.length).toFixed(4)}</span>`;
    $("#test-cards").innerHTML = test
      .map((item, i) => {
        const probs = m.test.predictions[i], best = probs.indexOf(Math.max(...probs));
        const tied = Math.max(...probs) - Math.min(...probs) < 1e-12;
        return `<article><p class="small-label">HELD-OUT ${task.type === "audio" ? "SOUND" : task.type === "image" ? "PICTURE" : "TEXT"} ${i + 1}</p>${sampleMedia(item, `Held-out example ${i + 1}`)}<strong>${tied ? "All classes tied" : `Prediction: ${esc(className(classes[best]))}`}</strong><p class="hint">Softmax output · not calibrated confidence</p><div class="class-bars">${classes.map((c, j) => `<div><span>${esc(className(c))}</span><i><b style="width:${probs[j] * 100}%"></b></i><code>${(probs[j] * 100).toFixed(1)}%</code></div>`).join("")}</div><details><summary>Reveal the dataset label</summary><p>${esc(className(item.label))} · ${tied ? "No preference yet" : classes[best] === item.label ? "Correct prediction" : "Different from the prediction"}</p></details></article>`;
      }).join("");
  }
  function renderExamples() {
    $("#split-note").textContent = `${train.length} training + ${test.length} held-out examples across ${classes.length} classes. ${task.type === "audio" ? "Original source recordings" : "Source pictures and their descriptions"} are kept separate across the split. This small set illustrates learning; it cannot give a reliable performance estimate. Changing the task or dimension resets the head.`;
    $("#split-table").innerHTML = `<p>Every example below uses a saved model embedding. Labels are used only to train and evaluate the head. ${task.type === "text" ? "These are the gallery’s existing caption embeddings, with the same text preprocessing used throughout the lab." : "The media were encoded without their titles or labels."}</p><div class="table-scroll"><table><thead><tr><th>Example</th><th>Input</th><th>Label</th><th>Split</th><th>Source ID</th></tr></thead><tbody>${[...train, ...test].map((i) => `<tr><td>${esc(i.title)}</td><td>${sampleMedia(i)}</td><td>${esc(className(i.label))}</td><td>${i.split === "test" ? "Held out" : "Training"}</td><td>${esc(i.sourceRecording || i.sourceGroup)}</td></tr>`).join("")}</tbody></table></div>`;
    renderInput();
  }
  function selectedExample() {
    return train.find(i => i.id === $("#train-example").value);
  }
  function renderInput() {
    const item = selectedExample(), x = unit(state.vectors[item.id].vector, d);
    $("#nn-input").innerHTML = `<p class="small-label">${task.type === "audio" ? "SOUND WAVEFORM" : task.type === "image" ? "PICTURE PIXELS" : "WRITTEN DESCRIPTION"}</p>${task.type === "audio" ? waveform(item.id) : ""}${sampleMedia(item)}<p class="hint">Label: <b>${esc(className(item.label))}</b></p><div class="nn-flow-arrow">↓</div><div class="nn-encoder"><b>EmbeddingGemma 2</b><span>Frozen encoder<br>Already computed this vector</span></div><div class="nn-flow-arrow">↓</div><div class="nn-saved-vector">x ∈ ℝ${[...String(d)].map(n => "⁰¹²³⁴⁵⁶⁷⁸⁹"[+n]).join("")} →</div><span class="nn-input-shape">${d} features · unit length<br>Same vector at every training step</span>`;
    $("#nn-vector-label").textContent = `Inspect all ${d} features for this input`;
    $("#nn-vector-values").textContent = x.map((v, k) => `x[${k + 1}] = ${v.toFixed(6)}`).join("\n");
  }
  function renderNetwork() {
    const item = selectedExample(), x = unit(state.vectors[item.id].vector, d);
    const selected = +$("#train-neuron").value, trueClass = classes.indexOf(item.label);
    const {svg, trace} = networkDiagram(head, x, classes, selected, trueClass);
    const arithmeticOpen = $("#nn-arithmetic details")?.open;
    $("#nn-diagram").innerHTML = svg;
    renderNeuronArithmetic($("#nn-arithmetic"), head, x, classes, selected, trueClass, trace);
    if (arithmeticOpen) $("#nn-arithmetic details").open = true;
  }

  function step() {
    trainHeadStep(head, rows(train), labels(train));
    epoch++;
    record();
  }
  $("#train-example").onchange = () => { renderInput(); renderNetwork(); };
  $("#train-neuron").onchange = renderNetwork;
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
        task: task.id,
        inputType: task.type,
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
