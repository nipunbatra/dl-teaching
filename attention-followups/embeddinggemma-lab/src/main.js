import { waveform } from "./waveform.js";
import "./style.css";
import "./exploration.css";
import "./knowledge.css";
import { mountKnowledge } from "./knowledge.js";
import { mountExplorer } from "./explorer.js";
import { mountTraining } from "./training.js";
let unmountAdvanced = () => {};
import { Engine, prepare } from "./runtime.js";
import { MODEL, REVISION } from "./config.js";
import { unit, dot, rank, difference, kmeans, pcaProjection } from "./math.js";
import { experiments } from "./experiments.js";
const $ = (s) => document.querySelector(s),
  esc = (s) =>
    String(s ?? "").replace(
      /[&<>"']/g,
      (c) =>
        ({
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
          "'": "&#39;",
        })[c],
    );
const state = {
  experiment: experiments[0],
  dimension: 768,
  busy: false,
  ready: false,
  gallery: [],
  vectors: {},
  localVectors: {},
  results: [],
  query: null,
  selected: null,
  upload: null,
  limit: 6,
};
let engine = new Engine((progress) => {
  if (progress.status === "progress_total")
    status(
      `Downloading model · ${Math.round(progress.progress)}%`,
      progress.progress,
    );
  else if (progress.status === "progress")
    status(
      `Loading ${progress.file || "model"} · ${Math.round(progress.progress || 0)}%`,
    );
  else if (progress.status === "initiate")
    status(`Loading ${progress.file || "model files"}…`);
});
function status(text, percent) {
  $("#status").textContent = text;
  if (state.busy) $("#model-state").textContent = text;
  $("#progress").hidden = !state.busy;
  percent == null
    ? $("#progress").removeAttribute("value")
    : ($("#progress").value = percent);
}
function error(e) {
  $("#error").textContent = e.message || String(e);
  $("#error").hidden = false;
}
function busy(value) {
  state.busy = value;
  document
    .querySelectorAll(
      "#query-area input,#query-area select,#query-area textarea,#query-area button,#dimension",
    )
    .forEach((n) => {
      if (value) {
        n.dataset.preDisabled = String(n.disabled);
        n.disabled = true;
      } else {
        n.disabled = n.dataset.preDisabled === "true";
      }
    });
  $("#run").disabled = value;
  $("#warm-model").disabled = value;
  $("#runtime-stop").hidden = !value;
  $("#workspace").setAttribute("aria-busy", String(value));
  $("#stop").hidden = !value;
  $("#experiment").disabled = value;
  document
    .querySelectorAll("[data-experiment]")
    .forEach((b) => (b.disabled = value));
  $("#progress").hidden = !value;
}
function clearResults() {
  state.query = null;
  state.results = [];
  state.selected = null;
  $("#inspect").hidden = true;
  $("#context").hidden = true;
  $("#map").hidden = true;
  $("#result-source").textContent = "";
  $("#results-title").textContent = "Browse the candidates";
  $("#result-meta").textContent =
    "A few items from the collection. Run a search to rank them.";
  const filter = $("#filter")?.value || "all";
  let examples = filterItems(filter);
  if (filter === "image") {
    const featured = [
      "chelsea",
      "coffee",
      "rocket",
      "retriever-photo",
      "solar",
      "astronaut",
    ];
    examples = [
      ...featured.map(itemById).filter(Boolean),
      ...examples.filter((i) => !featured.includes(i.id)),
    ];
  }
  $("#results").innerHTML = examples
    .slice(0, 6)
    .map((item, i) => card(item, i, null))
    .join("");
  $("#result-step").textContent = "02 / EXPLORE THE MATCHES";
  $("#more").hidden = true;
}
function itemById(id) {
  return state.gallery.find((x) => x.id === id);
}
function filterItems(filter) {
  return state.gallery.filter(
    (i) => filter === "all" || i.type === filter || i.group === filter,
  );
}
function thumb(item, controls = false) {
  if (item.type === "image")
    return `<img src="${esc(item.src)}" alt="${esc(item.title)}" loading="lazy">`;
  if (item.type === "audio")
    return `<div class="audio-preview">${waveform(item.id)}${controls ? `<audio controls preload="metadata" src="${esc(item.src)}"></audio>` : "<span>5-second recording</span>"}</div>`;
  if (item.type === "video")
    return `<video ${controls ? "controls" : "muted"} playsinline preload="metadata" src="${esc(item.src)}#t=${item.start || 0.05},${item.end || 20}" aria-label="${esc(item.title)}"></video>`;
  return `<div class="text-preview ${item.group === "code" ? "code-preview" : ""}">${esc(item.text)}</div>`;
}
function renderQuery() {
  const e = state.experiment;
  $("#query-area").innerHTML =
    `<div class="input-row"><label>Query input<select id="query-type">${["text", "image", "audio", "video"].map((t) => `<option ${t === e.type ? "selected" : ""}>${t}</option>`).join("")}</select></label><label>Search within<select id="filter">${[
      ["all", "Everything"],
      ["image", "Images"],
      ["audio", "Sounds"],
      ["caption", "Captions"],
      ["moment", "Video moments"],
      ["video", "Videos"],
      ["document", "Course passages"],
      ["code", "Code"],
    ]
      .map(
        ([v, t]) =>
          `<option value="${v}" ${e.filter === v ? "selected" : ""}>${t}</option>`,
      )
      .join(
        "",
      )}</select></label></div><div id="query-fields"></div><div id="special-fields"></div>`;
  $("#query-type").onchange = () => {
    renderFields();
    clearResults();
  };
  $("#filter").onchange = () => {
    if (state.query) showRanking();
    else clearResults();
  };
  renderFields();
  if (e.mode === "classify")
    $("#special-fields").innerHTML =
      `<label class="field">Your labels <span>one per line · up to 12</span><textarea id="labels" rows="4">${esc(e.labels)}</textarea></label>`;
  if (e.mode === "delta") {
    $("#query-type").value = "image";
    $("#query-type").disabled = true;
    renderFields();
    $("#special-fields").innerHTML =
      `<label class="field">After image<select id="after">${state.gallery
        .filter((i) => i.type === "image")
        .map(
          (i) =>
            `<option value="${esc(i.id)}" ${i.id === e.after ? "selected" : ""}>${esc(i.title)}</option>`,
        )
        .join(
          "",
        )}</select></label><div id="after-preview" class="query-preview"></div><button id="swap" class="quiet" type="button">Swap before and after</button>`;
    const afterPreview = () =>
      ($("#after-preview").innerHTML = thumb(itemById($("#after").value)));
    $("#after").onchange = () => {
      afterPreview();
      clearResults();
    };
    afterPreview();
    $("#swap").onclick = () => {
      const a = $("#sample").value;
      $("#sample").value = $("#after").value;
      $("#after").value = a;
      updatePreview();
      afterPreview();
      clearResults();
    };
  }
  if (e.mode === "clusters") {
    $("#query-area").innerHTML =
      '<p class="hint">The collection is already embedded. Grouping needs no model download.</p><label class="field">Number of groups<select id="groups"><option>3</option><option selected>4</option><option>5</option><option>6</option></select></label>';
  }
  $("#query-area").oninput = (event) => {
    if (!["filter", "groups"].includes(event.target.id)) clearResults();
  };
  if (e.mode === "mixed") $("#query-type").disabled = true;
  $("#run").textContent =
    e.mode === "clusters"
      ? "Group the collection"
      : e.mode === "delta"
        ? "Compare the change"
        : e.mode === "classify"
          ? "Compare with labels"
          : "Run this search";
}
function renderFields() {
  const type = $("#query-type").value,
    e = state.experiment;
  $("#query-fields").innerHTML =
    type === "text"
      ? `<label class="field">What are you looking for?<textarea id="query" rows="2" maxlength="2000" placeholder="Describe an image, sound, idea or moment…">${esc(e.query || "")}</textarea></label><div class="ideas">${(e.ideas || []).map((t) => `<button type="button" class="chip" data-idea="${esc(t)}">${esc(t)}</button>`).join("")}</div>`
      : `<label class="field">${e.mode === "delta" ? "Before image" : "Choose a sample"}<select id="sample">${state.gallery
          .filter((i) => i.type === type)
          .map(
            (i) =>
              `<option value="${esc(i.id)}" ${e.sample === i.id ? "selected" : ""}>${esc(i.title)}</option>`,
          )
          .join(
            "",
          )}${state.upload?.type === type ? `<option value="${esc(state.upload.id)}" selected>Your upload</option>` : ""}</select></label><div id="query-preview" class="query-preview"></div><div class="sample-strip" id="sample-strip" aria-label="Try another sample"></div><details class="input-options"><summary>Upload or re-encode an input</summary><label class="upload">Or use your own ${type}<input id="upload" type="file" accept="${type}/*"></label><p class="hint">${type === "audio" ? "Audio is mixed to mono, resampled to 16 kHz and limited to the first 20 seconds." : type === "video" ? "We read the first 20 seconds at one frame per second. Video audio is not included." : "Your pixels are processed in this browser."} Upload limit: 30 MB.</p><label class="check"><input type="checkbox" id="recompute"> Re-encode this sample on my device</label></details>${e.mode === "mixed" ? `<label class="field">Add a note<textarea id="query" rows="2" maxlength="1000">${esc(e.query)}</textarea></label>` : ""}`;
  if (type !== "text") {
    $("#sample").onchange = () => {
      updatePreview();
      clearResults();
    };
    $("#upload").onchange = upload;
    updatePreview();
  }
}
function updatePreview() {
  const item = selectedInput();
  if (item) {
    $("#query-preview").innerHTML = thumb(item, true);
    const strip = $("#sample-strip");
    if (strip)
      strip.innerHTML = [item, ...state.gallery.filter((i) => i.type === item.type && i.id !== item.id)].slice(0, 6)
        .map(
          (i) =>
            `<button type="button" class="sample-pick sample-${i.type}" data-sample="${esc(i.id)}" aria-label="Choose ${esc(i.title)}" aria-pressed="${i.id === item.id}">${thumb(i)}<span>${esc(i.title)}</span></button>`,
        )
        .join("");
  }
}
function selectedInput() {
  const type = $("#query-type")?.value;
  if (type === "text")
    return {
      id: "query",
      type: "text",
      title: "Your query",
      text: $("#query")?.value.trim() || "",
    };
  const id = $("#sample")?.value;
  return id === state.upload?.id ? state.upload : itemById(id);
}
async function upload(event) {
  const file = event.target.files[0];
  if (!file) return;
  if (file.size > 30 * 1024 * 1024) {
    error(Error("Please use a file smaller than 30 MB."));
    return;
  }
  if (state.upload?.src) URL.revokeObjectURL(state.upload.src);
  state.upload = {
    id: "upload-" + Date.now(),
    type: $("#query-type").value,
    title: file.name,
    blob: file,
    src: URL.createObjectURL(file),
    credit: "Your local file",
  };
  renderFields();
  clearResults();
}
function activate(id, context = {}) {
  if (state.busy) return;
  state.experiment = { ...(experiments.find((e) => e.id === id) || experiments[0]), ...(context.sample ? {sample: context.sample} : {}), ...(context.after ? {after:context.after} : {}) };
  const e = state.experiment;
  $("#experiment").value = e.id;
  document
    .querySelectorAll("[data-experiment]")
    .forEach((b) =>
      b.setAttribute("aria-current", String(b.dataset.experiment === e.id)),
    );
  $("#experiment-title").textContent = e.title;
  $("#experiment-number").textContent =
    `EXPERIMENT ${String(experiments.findIndex(x => x.id === e.id) + 1).padStart(2, "0")} / ${experiments.length}`;
  $("#experiment-description").textContent = e.description;
  $("#lesson").textContent = e.lesson;
  unmountAdvanced();
  const advanced = ["explorer", "training", "knowledge"].includes(e.mode);
  $("#advanced-area").hidden = !advanced;
  $(".workbench").hidden = advanced;
  $("#inspect").hidden = true;
  if (advanced) {
    state.query = null;
    state.selected = null;
    unmountAdvanced =
      e.mode === "explorer"
        ? mountExplorer($("#advanced-area"), state, {selected:context.sample, compare:context.compare})
        : e.mode === "knowledge"
          ? mountKnowledge($("#advanced-area"), e.id, { embed: embedItem, busy })
          : mountTraining($("#advanced-area"), state);
  } else {
    $("#advanced-area").innerHTML = "";
    renderQuery();
    clearResults();
    unmountAdvanced = () => {};
  }
  history.replaceState(null, "", "#" + e.id);
}
async function ensureModel() {
  if (state.ready) return;
  if (!navigator.gpu || !(await navigator.gpu.requestAdapter()))
    throw Error(
      "WebGPU is unavailable here. Try current Chrome or Edge with hardware acceleration. You can still explore stored sample vectors and groups.",
    );
  status("Loading EmbeddingGemma 2 · about 473 MB once, then browser-cached…");
  await engine.load();
  state.ready = true;
  $("#model-state").textContent = "WebGPU model ready";
}
async function embedItem(
  item,
  { role = "query", task = "search result", force = false, mixed = "" } = {},
) {
  if (state.vectors[item.id] && !force && !mixed)
    return {
      ...state.vectors[item.id],
      origin: "Saved WebGPU run · same pinned model",
    };
  await ensureModel();
  status(`Encoding ${item.title}…`);
  const { input, info } = await prepare(item, task, role, mixed);
  const result = {
    ...(await engine.embed(input)),
    info,
    origin: "Computed in this browser · WebGPU",
  };
  if (!mixed && item.type !== "text") state.localVectors[item.id] = result;
  return { ...result };
}
async function run() {
  if (state.busy) return;
  $("#error").hidden = true;
  busy(true);
  const e = state.experiment;
  try {
    if (e.mode === "clusters") {
      showGroups();
      return;
    }
    const item = selectedInput();
    if (!item) throw Error("Choose an input first.");
    if (item.type === "text" && !item.text) throw Error("Enter a query first.");
    if (e.mode === "classify" && !$("#labels").value.trim())
      throw Error("Add at least one label.");
    if (e.mode === "mixed" && !$("#query").value.trim())
      throw Error("Add a short note so both the picture and words are encoded.");
    const forced = $("#recompute")?.checked,
      query = await embedItem(item, {
        task:
          e.mode === "classify" ? "classification" : e.task || "search result",
        force: forced,
        mixed: e.mode === "mixed" ? $("#query").value.trim() : "",
      });
    let queryVector = query.vector,
      exclude = [item.id],
      origin = query.origin;
    if (e.mode === "delta") {
      const after = itemById($("#after").value);
      const other = await embedItem(after, { force: forced });
      query.deltaInputs = { before: query.vector, after: other.vector };
      queryVector = difference(unit(other.vector, state.dimension), unit(query.vector, state.dimension));
      origin = other.origin;
      exclude.push(after.id);
      query.info = {
        formula: "unit(after − before)",
        before: item.title,
        after: after.title,
        beforeInput: query.info,
        afterInput: other.info,
        beforeShapes: query.shapes,
        afterShapes: other.shapes,
      };
    }
    let candidates = null;
    if (e.mode === "classify") {
      const labels = [
        ...new Set(
          $("#labels")
            .value.split("\n")
            .map((x) => x.trim())
            .filter(Boolean),
        ),
      ];
      if (labels.length > 12)
        throw Error("Use at most 12 labels for this small lab.");
      candidates = [];
      for (let i = 0; i < labels.length; i++) {
        const label = {
          id: "label-" + i,
          type: "text",
          title: labels[i],
          text: labels[i],
          group: "label",
          credit: "Your candidate label",
        };
        const result = await embedItem(label, {
          task: "classification",
          force: true,
        });
        state.vectors[label.id] = result;
        candidates.push(label);
      }
    }
    state.query = {
      ...query,
      vector: queryVector,
      item,
      mixedNote: e.mode === "mixed" ? $("#query").value.trim() : null,
      exclude,
      candidates,
      origin,
    };
    state.limit = 6;
    showRanking();
    status(
      state.ready
        ? "Ready · queries run on this device"
        : "Ready · comparing stored sample vectors",
    );
  } catch (e) {
    error(e);
    state.query = null;
    status("Not completed. You can change the input and try again.");
  } finally {
    busy(false);
  }
}
function showRanking() {
  if (!state.query) return;
  const q = state.query,
    filter = $("#filter")?.value || "all";
  if (q.deltaInputs) q.vector = difference(unit(q.deltaInputs.after, state.dimension), unit(q.deltaInputs.before, state.dimension));
  state.results = rank(
    q.vector,
    q.candidates || filterItems(filter),
    state.vectors,
    state.dimension,
    q.exclude,
  );
  $("#map").hidden = true;
  $("#result-meta").textContent =
    `${state.results.length} candidates · ${state.dimension} dimensions · raw cosine similarity${state.experiment.mode === "classify" ? " · labels use the classification query prefix" : ""}${filter === "all" ? " · cross-modality score ranges may differ" : ""}`;
  renderResults();
  $("#inspect").hidden = true;
  state.selected = null;
  $("#results-title").textContent = "The closest matches";
  $("#result-step").textContent = "02 / COMPARE THE RESULTS";
  $("#context").hidden = !["document", "code"].includes(filter);
  if (!$("#context").hidden) {
    $("#context-text").textContent = state.results
      .slice(0, 3)
      .map((r, i) => `[${i + 1}] ${r.item.title}\n${r.item.text}`)
      .join("\n\n");
  }
}
function renderResults() {
  const q = state.query;
  $("#results").innerHTML =
    state.results
      .slice(0, state.limit)
      .map((r, i) => card(r.item, i, r.score))
      .join("") ||
    '<p class="empty">No candidates in this selection. Choose another collection.</p>';
  $("#more").hidden = state.limit >= state.results.length;
  $("#result-source").textContent = q
    ? `${q.origin}${q.elapsed ? " · last encoding " + Math.round(q.elapsed) + " ms" : ""}. Candidates use the same pinned model. Re-encode recomputes sample inputs locally.`
    : "";
}
function card(item, index, score) {
  return `<article class="result-card ${score == null ? "preview-card" : ""}"><div class="card-media">${thumb(item, true)}</div><div class="card-body"><div class="card-top"><span class="small-label">${esc(item.group || item.type)}</span>${score == null ? "" : `<span class="score" title="Cosine similarity, not a probability">${score.toFixed(4)}</span>`}</div><h3>${score == null ? "" : `<span class="rank">${index + 1}</span> `}${esc(item.title)}</h3>${score == null ? "" : `<div class="score-gap">${index === 0 ? "Highest in this candidate menu" : `${(state.results[0].score - score).toFixed(3)} below the highest score`}</div>`}<div class="card-actions"><button data-inspect="${esc(item.id)}" class="quiet">${score == null ? "See vector" : "Explain score ↗"}</button><button data-query="${esc(item.id)}" class="quiet">Use as query</button></div></div></article>`;
}
function findItem(id) {
  return state.query?.candidates?.find((i) => i.id === id) || itemById(id);
}
function inspect(id) {
  const item = findItem(id),
    data = state.vectors[id];
  if (!item || !data) return;
  state.selected = id;
  $("#inspect").hidden = false;
  $("#inspect-title").textContent = item.title;
  $("#inspect-pair").innerHTML =
    `${state.query ? `<div><span class="small-label">Your ${state.experiment.mode === "delta" ? "change direction" : "query"}</span><strong>${esc(state.experiment.mode === "delta" ? state.query.info.before + " → " + state.query.info.after : (state.query.item.text || state.query.item.title) + (state.query.mixedNote ? " + “" + state.query.mixedNote + "”" : ""))}</strong></div><span class="pair-arrow">→</span>` : ""}<div><span class="small-label">The candidate · ${esc(item.type)}</span><strong>${esc(item.title)}</strong></div>`;
  $("#candidate-text").textContent =
    item.type === "text" ? item.text : item.credit;
  const query = state.query ? unit(state.query.vector, state.dimension) : null,
    v = unit(data.vector, state.dimension);
  const colourMax = Math.max(
    ...v.map(Math.abs),
    ...(query || []).map(Math.abs),
  );
  $("#candidate-canvas").hidden = false;
  heatmap($("#candidate-canvas"), v, colourMax);
  $("#query-canvas").hidden = !query;
  if (query) heatmap($("#query-canvas"), query, colourMax);
  $("#query-vector-label").textContent = query
    ? "Query vector q"
    : "Run a query to compare two vectors";
  $("#candidate-vector-label").textContent =
    `Candidate vector v · ${state.dimension} numbers`;
  $("#shape-details").textContent =
    `${state.query ? "QUERY INPUT\n" + JSON.stringify(state.query.info, null, 2) + "\nQuery tensor shapes: " + JSON.stringify(state.query.shapes) + "\n\n" : ""}CANDIDATE INPUT\n${JSON.stringify(data.info, null, 2)}\n\nModel input tensor shapes:\n${JSON.stringify(data.shapes, null, 2)}\n\nModel output: [1, 768]\nSelected: [1, ${state.dimension}] → L2-normalize\nNorm = ${Math.hypot(...v).toFixed(6)}`;
  $("#dot-area").hidden = !query;
  $("#coordinates").max = state.dimension - 8;
  $("#coordinates").value = 0;
  if (query) renderDot();
  $("#vector-note").textContent =
    "The two maps use the same colour scale: dark green = positive, brown = negative. Coordinates are learned numbers, not named concepts. Each row below shows an exact multiplication.";
  $("#download-vector").onclick = () =>
    download(
      {
        model: MODEL,
        revision: REVISION,
        dimension: state.dimension,
        item: item.title,
        vector: v,
        ...(query ? { queryVector: query, cosine: dot(query, v) } : {}),
      },
      "embedding.json",
    );
}
function heatmap(canvas, v, colourMax) {
  const ctx = canvas.getContext("2d"),
    cols = 64,
    rows = Math.ceil(v.length / cols),
    dpr = devicePixelRatio || 1,
    w = canvas.clientWidth || 500,
    h = rows * 9;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  canvas.style.height = h + "px";
  ctx.scale(dpr, dpr);
  const max = colourMax || Math.max(...v.map(Math.abs));
  v.forEach((x, i) => {
    ctx.fillStyle =
      x >= 0
        ? `rgba(57,103,73,${0.08 + (0.92 * Math.abs(x)) / max})`
        : `rgba(156,90,48,${0.08 + (0.92 * Math.abs(x)) / max})`;
    ctx.fillRect(
      ((i % cols) * w) / cols,
      Math.floor(i / cols) * 9,
      w / cols - 1,
      8,
    );
  });
}
function renderDot() {
  const q = unit(state.query.vector, state.dimension),
    v = unit(state.vectors[state.selected].vector, state.dimension),
    start = +$("#coordinates").value;
  $("#coordinate-range").textContent =
    `Coordinates ${start + 1}–${start + 8} of ${state.dimension}`;
  $("#coordinate-table").innerHTML = Array.from({ length: 8 }, (_, i) => {
    const j = start + i;
    return `<tr><td>${j + 1}</td><td>${q[j].toFixed(6)}</td><td>×</td><td>${v[j].toFixed(6)}</td><td>${(q[j] * v[j]).toFixed(6)}</td></tr>`;
  }).join("");
  const partial = q
    .slice(start, start + 8)
    .reduce((s, x, i) => s + x * v[start + i], 0);
  $("#dot-total").innerHTML =
    `<span>These 8 products sum to <b>${partial.toFixed(6)}</b>.</span><strong>All ${state.dimension} products sum to ${dot(q, v).toFixed(6)}</strong><span>q · v = Σ qₖvₖ = cosine(q, v), because both vectors have length 1.</span>`;
}
function showGroups() {
  const items = state.gallery.filter(
      (i) => !["moment", "document", "code"].includes(i.group),
    ),
    rows = items.map((i) => unit(state.vectors[i.id].vector, state.dimension)),
    labels = kmeans(rows, +$("#groups").value),
    projection = pcaProjection(rows),
    points = projection.points;
  $("#inspect").hidden = true;
  $("#context").hidden = true;
  $("#map").hidden = false;
  $("#results-title").textContent = "Neighbours in the shared space";
  $("#result-step").textContent = "02 / EXPLORE THE GROUPS";
  $("#result-meta").textContent =
    `${items.length} items · ${$("#groups").value} groups · ${state.dimension} dimensions`;
  $("#result-source").textContent =
    "Computed now from the stored WebGPU embeddings. No titles or class labels are used for clustering.";
  const width = 700,
    height = 340,
    minX = Math.min(...points.map((p) => p[0])),
    maxX = Math.max(...points.map((p) => p[0])),
    minY = Math.min(...points.map((p) => p[1])),
    maxY = Math.max(...points.map((p) => p[1])),
    plotScale = Math.min((width - 60)/(maxX-minX || 1),(height-60)/(maxY-minY || 1));
  $("#map").innerHTML =
    `<p class="hint">PCA projection · ${(100 * projection.variance.reduce((a,b)=>a+b,0)).toFixed(1)}% variance in 2D · equal scale on both axes · colour = group</p><div class="map-legend">${Array.from({ length: +$("#groups").value }, (_, g) => `<span><i style="background:${["#376448", "#a56538", "#627999", "#874d60", "#888338", "#635e85"][g]}"></i>Group ${g + 1}</span>`).join("")}</div><svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Two dimensional projection of the collection">${points.map((p, i) => `<g class="map-point" tabindex="0" role="button" aria-label="${esc(items[i].title)}, group ${labels[i] + 1}" data-inspect="${esc(items[i].id)}"><circle cx="${width/2 + (p[0]-(minX+maxX)/2)*plotScale}" cy="${height/2 - (p[1]-(minY+maxY)/2)*plotScale}" r="7" fill="${["#376448", "#a56538", "#627999", "#874d60", "#888338", "#635e85"][labels[i]]}"/><title>${esc(items[i].title)} · group ${labels[i] + 1}</title></g>`).join("")}</svg>`;
  $("#results").innerHTML = Array.from(
    { length: +$("#groups").value },
    (_, g) =>
      `<div class="cluster" style="border-color:${["#376448", "#a56538", "#627999", "#874d60", "#888338", "#635e85"][g]}"><h3>Group ${g + 1}</h3>${items
        .filter((_, i) => labels[i] === g)
        .map(
          (item) =>
            `<button class="cluster-item" data-inspect="${esc(item.id)}"><span class="small-label">${item.type}</span>${esc(item.title)}</button>`,
        )
        .join("")}</div>`,
  ).join("");
  $("#more").hidden = true;
  status("Groups calculated. Inspect the neighbours and the exceptions.");
}
function download(value, name) {
  const u = URL.createObjectURL(
      new Blob([JSON.stringify(value, null, 2)], { type: "application/json" }),
    ),
    a = document.createElement("a");
  a.href = u;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(u), 1000);
}
function renderLibrary() {
  const filter = $("#library-filter").value;
  $("#library-grid").innerHTML = filterItems(filter)
    .map(
      (item) =>
        `<article class="library-item"><div class="card-media">${thumb(item, true)}</div><h3>${esc(item.title)}</h3><span class="small-label">${esc(item.group || item.type)}</span><div class="card-actions"><button class="quiet" data-query="${esc(item.id)}">Use as query</button><button class="quiet" data-inspect="${esc(item.id)}">Inspect</button></div><details><summary>Source & credit</summary><p>${esc(item.credit)}</p>${item.source ? `<a href="${esc(item.source)}" target="_blank" rel="noreferrer">Source</a>` : ""}</details></article>`,
    )
    .join("");
}
function useAsQuery(id) {
  if (state.busy) return;
  const item = findItem(id);
  if (!item) return;
  $("#library").close();
  activate("photos");
  $("#query-type").value = item.type;
  renderFields();
  if (item.type === "text") $("#query").value = item.text;
  else {
    $("#sample").value = id;
    updatePreview();
  }
  $("#filter").value = "all";
  $("#experiment-title").textContent = "Search with " + item.title;
  $("#experiment-description").textContent =
    "Compare this input with the collection. Choose a candidate type, predict a match, then run.";
  $("#lesson").textContent =
    "The same comparison works for different inputs: encode, normalize and take a dot product. Similarity scores can have different ranges across modalities.";
  $("#workspace").scrollIntoView({ behavior: "smooth" });
  clearResults();
}
async function init() {
  $("#app").innerHTML =
    `<header class="site-header"><a class="brand" href="https://nipunbatra.github.io/"><span class="brand-mark">nb.</span> Nipun Batra <span class="brand-slash">/</span> <span class="brand-course">Learning labs</span></a><nav aria-label="Course links"><a href="https://nipunbatra.github.io/dl-teaching/slides.html">Course slides ↗</a><button class="quiet" id="open-about">How it works</button></nav></header>
<main><section class="intro"><div class="intro-copy"><p class="eyebrow">MULTIMODAL EMBEDDINGS · IN YOUR BROWSER</p><h1>EmbeddingGemma 2</h1><p>Search images, text, audio and video. Ask a question about the course, find useful code, or inspect the vectors yourself.</p><button class="intro-link" data-experiment="photos">Try the applications <span>↓</span></button><button class="quiet intro-explore" data-experiment="explorer">Explore the embeddings ↗</button><div class="intro-facts"><span><b>${experiments.length}</b> experiments</span><span><b id="sample-count">…</b> samples</span><span><b>768</b> dimensions</span></div></div>
<div class="intro-media"><div class="media-caption"><span>A few ways in</span><span>Pick one to begin ↙</span></div><div class="media-tiles"><button class="hero-tile image-tile" data-experiment="captions"><img src="./media/chelsea.jpg" alt="An orange cat looking at the camera"><span><b>Start with a picture</b>Find its words <i>↗</i></span></button><button class="hero-tile sound-tile" data-experiment="listen"><span class="sound-drawing" aria-hidden="true"><svg viewBox="0 0 150 90"><path d="M9 39v12m11-21v30m11-39v48m11-32v16m11-52v88m11-69v50m11-38v26m11-44v62m11-46v30m11-40v50m11-32v14m11-25v36m11-27v18"/></svg><small>Dog bark · 5 seconds</small></span><span><b>Start with a sound</b>Find its picture <i>↗</i></span></button><button class="hero-tile text-tile" data-experiment="moments"><span class="text-drawing">“a rocket<br>launching”<small>Text → video</small></span><span><b>Start with a thought</b>Find a moment <i>↗</i></span></button></div><p class="media-footnote">Different inputs. The same encode → compare idea.</p></div></section><div class="runtime-bar"><span class="status-dot"></span><strong id="model-state">Collection ready · model loads on demand</strong><span>WebGPU · q4 · ~473 MB first download</span><button class="quiet" id="warm-model">Load WebGPU for new inputs</button><button class="quiet" id="runtime-stop" hidden>Stop model</button><button class="quiet" id="open-library">Explore the collection</button></div><div class="lab-layout"><nav class="experiment-nav" aria-label="Experiments"><div class="nav-intro"><span class="eyebrow">APPLICATIONS</span><span>Choose what you want to try</span></div>${[
      "Find",
      "Use",
      "Inspect",
      "Learn",
    ]
      .map(
        (g) =>
          `<div class="experiment-group"><p class="small-label">${g}</p><div>${experiments
            .filter((e) => e.group === g)
            .map(
              (e) =>
                `<button data-experiment="${e.id}" aria-current="false">${e.name}</button>`,
            )
            .join("")}</div></div>`,
      )
      .join(
        "",
      )}</nav><section id="workspace"><label class="mobile-experiment">Experiment<select id="experiment">${experiments.map((e) => `<option value="${e.id}">${e.name}</option>`).join("")}</select></label><div class="experiment-head"><span class="eyebrow" id="experiment-number"></span><h2 id="experiment-title"></h2><p id="experiment-description"></p></div><div id="advanced-area" hidden></div><div class="workbench"><div class="query-column"><div class="query-panel"><p class="step-label">01 / CHOOSE YOUR INPUT</p><div id="query-area"></div><div class="run-row"><button id="run" class="primary">Run this search</button><button id="stop" class="quiet" hidden>Stop & unload</button><span id="status" role="status" aria-live="polite">No API key. Your inputs stay in this browser.</span></div><progress id="progress" max="100" hidden></progress><p id="error" role="alert" hidden></p></div><p class="lesson"><b>What to notice</b><span id="lesson"></span></p></div><div class="results-column"><p class="step-label" id="result-step">02 / EXPLORE THE MATCHES</p><div class="results-heading"><div><h2 id="results-title">The matches</h2><p id="result-meta"></p></div><label>Vector size<select id="dimension"><option value="768">768 · full</option><option value="512">512</option><option value="256">256</option><option value="128">128</option></select></label></div><p id="dimension-note" class="hint">Keep the first d coordinates, then normalize again. Both sides use the same d.</p><div id="map" hidden></div><div id="results" class="results-grid"></div><button id="more" class="quiet" hidden>Show more results</button><p id="result-source" class="fineprint"></p><details id="context" hidden><summary>The context a generative model could receive</summary><pre id="context-text"></pre><p>This lab stops at retrieval. Check whether these sources actually answer the question.</p></details></div></div><section id="inspect" class="inspector" hidden><div class="inspector-head"><div><p class="eyebrow">03 / FOLLOW THE NUMBERS</p><h2 id="inspect-title"></h2></div><div class="inspector-actions"><button class="quiet" id="download-vector">Download vectors</button><button class="quiet" id="close-inspect" aria-label="Close vector inspector">Close ×</button></div></div><div id="inspect-pair" class="inspect-pair"></div><p id="candidate-text"></p><div class="vector-pair"><div><h3 id="query-vector-label"></h3><canvas id="query-canvas" aria-label="Query embedding coordinates"></canvas></div><div><h3 id="candidate-vector-label"></h3><canvas id="candidate-canvas" aria-label="Candidate embedding coordinates"></canvas></div></div><p class="hint" id="vector-note"></p><div id="dot-area"><label id="coordinate-range" for="coordinates"></label><input type="range" id="coordinates" min="0" max="760" step="8" value="0"><div class="table-scroll"><table><thead><tr><th>k</th><th>Query qₖ</th><th></th><th>Candidate vₖ</th><th>Product</th></tr></thead><tbody id="coordinate-table"></tbody></table></div><div id="dot-total" class="dot-total"></div></div><details><summary>Show model inputs, shapes and normalization</summary><pre id="shape-details"></pre></details></section></section></div><footer><p class="eyebrow">EMBEDDINGGEMMA 2</p><strong>One model. Images, text, audio and video.</strong><p>Encode → normalize → compare. The model returns embeddings; the application decides how to use them.</p><div><a href="https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2/">Google announcement</a><a href="https://www.youtube.com/watch?v=anPsS6huQk0">Introduction video</a><a href="https://huggingface.co/onnx-community/embeddinggemma-2-ONNX">Model & implementation</a><button class="quiet" id="footer-library">Sample credits</button><a href="https://github.com/nipunbatra/dl-teaching/tree/master/attention-followups/embeddinggemma-lab">Source code</a></div></footer></main><dialog id="library"><div class="dialog-head"><div><p class="eyebrow">THE SHARED COLLECTION</p><h2 id="library-title">Samples you can inspect</h2></div><button class="quiet" data-close="library" aria-label="Close collection">Close ×</button></div><p>Images, recordings and video are embedded from their media, without their titles. Some images are generated teaching examples, identified in the credits.</p><label>Show<select id="library-filter"><option value="all">All samples</option><option value="image">Images</option><option value="audio">Sounds</option><option value="text">Text & code</option><option value="video">Video</option></select></label><div id="library-grid" class="library-grid"></div></dialog><dialog id="about"><div class="dialog-head"><h2>How EmbeddingGemma 2 powers this lab</h2><button class="quiet" data-close="about">Close ×</button></div><ol class="explanation"><li><strong>Represent each item.</strong> Media encoders feed a shared model. Mean pooling and a projection produce a 768-number embedding.</li><li><strong>Store the collection.</strong> The supplied vectors were computed from these exact samples using this pinned ONNX model, q4 precision and WebGPU. They let us inspect the collection without re-encoding it on every search.</li><li><strong>Encode a new query on your device.</strong> Model files download from Hugging Face. Typed text and uploaded media go to a local worker; there is no inference server or API key.</li><li><strong>Compare.</strong> Unit vectors give a cosine through a dot product. Scores are similarities, not probabilities. There is no learned threshold that guarantees a match.</li><li><strong>Shorten, carefully.</strong> The size control keeps leading coordinates and re-normalizes. It reduces index storage, not the size of the model download. The 128-dimensional option can hurt multimodal retrieval.</li></ol><p class="hint">Text search uses a task prefix; text candidates use a document prefix. Classification uses the classification prefix. Inspect the input details to see the exact text supplied.</p><p>Local uploads are kept only in this tab. Audio is limited to 20 seconds. Video uses up to 20 seconds of sampled frames without its soundtrack. Close or stop the model to release its worker.</p><p class="fineprint">${MODEL}<br>Revision ${REVISION}<br>Transformers.js 4.3.1 · q4 · Apache 2.0 model</p></dialog>`;
  try {
    const [gallery, embeddings] = await Promise.all([
      fetch("./gallery.json?v=course-20261007").then((r) => {
        if (!r.ok) throw Error("Could not load sample collection.");
        return r.json();
      }),
      fetch("./embeddings.json?v=course-20261007").then((r) => {
        if (!r.ok) throw Error("Could not load the stored embeddings.");
        return r.json();
      }),
    ]);
    state.gallery = gallery;
    $(".media-footnote").textContent = `${gallery.filter(x=>x.type==="image").length} images · ${gallery.filter(x=>x.type==="audio").length} sounds · ${gallery.filter(x=>x.type==="text").length} text examples · ${gallery.filter(x=>x.type==="video").length} video entries`;
    $("#sample-count").textContent = gallery.length;
    state.vectors = embeddings.items;
    $("#library-title").textContent =
      `${gallery.length} samples, one shared space`;
    if (embeddings.revision !== REVISION)
      throw Error("The sample embeddings and model version do not match.");
    // Keep links to the retired classroom view useful.
    const entry = new URL(location.href);
    entry.searchParams.delete("present");
    history.replaceState(null, "", entry);
    activate(location.hash.slice(1));
  } catch (e) {
    error(e);
    return;
  }
  $("#warm-model").onclick = async () => {
    if (state.busy) return;
    busy(true);
    $("#error").hidden=true;
    try { await ensureModel(); const {input}=await prepare({type:"text",text:"a cat",id:"warmup"},"search result","query"); await engine.embed(input); status("WebGPU ready · model warmed for live inputs"); }
    catch(e){error(e);status(e.message);}
    finally{busy(false);}
  };
  $("#runtime-stop").onclick=()=>{engine.stop();state.ready=false;$("#model-state").textContent="Model unloaded";};
  $("#run").onclick = run;
  $("#stop").onclick = () => {
    engine.stop();
    state.ready = false;
    $("#model-state").textContent = "Model unloaded";
  };
  $("#experiment").onchange = (e) => activate(e.target.value);
  $("#coordinates").oninput = renderDot;
  $("#dimension").onchange = (e) => {
    state.dimension = +e.target.value;
    $("#dimension-note").textContent =
      `${state.dimension * 4} bytes per float32 vector · ${Math.round((768 / state.dimension) * 10) / 10}× smaller than 768d. ${state.dimension === 128 ? "128d can lose multimodal quality. " : ""}Keep the first d values and re-normalize both vectors.`;
    if (state.experiment.mode === "clusters" && $("#map").hidden === false)
      showGroups();
    else if (state.query) showRanking();
    else if (state.selected) inspect(state.selected);
  };
  $("#close-inspect").onclick = () => {
    $("#inspect").hidden = true;
    state.selected = null;
  };
  $("#more").onclick = () => {
    state.limit += 12;
    renderResults();
  };
  const openLibrary = () => {
    renderLibrary();
    $("#library").showModal();
  };
  $("#open-library").onclick = openLibrary;
  $("#footer-library").onclick = openLibrary;
  $("#library-filter").onchange = renderLibrary;
  $("#open-about").onclick = () => $("#about").showModal();
  document.addEventListener("click", (event) => {
    const t = event.target.closest("button,[data-inspect]");
    if (!t) return;
    if (t.dataset.experiment) {
      activate(t.dataset.experiment);
      $("#workspace").scrollIntoView({ behavior: "smooth", block: "start" });
    }
    if (t.dataset.sample && !state.busy) {
      $("#sample").value = t.dataset.sample;
      updatePreview();
      clearResults();
    }
    if (t.dataset.idea) {
      $("#query").value = t.dataset.idea;
      clearResults();
    }
    if (t.dataset.inspect && !state.busy) {
      $("#library").close();
      inspect(t.dataset.inspect);
      $("#inspect").scrollIntoView({ behavior: "smooth", block: "start" });
    }
    if (t.dataset.query) {
      if (["explorer", "training"].includes(state.experiment.mode))
        activate("photos");
      useAsQuery(t.dataset.query);
    }
    if (t.dataset.close) $("#" + t.dataset.close).close();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && event.target.matches(".map-point"))
      event.target.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
  window.embeddingLab = {
    state,
    run,
    activate,
    inspect,
    engine,
    prepare,
    showRanking,
  };
  if (!navigator.gpu)
    $("#model-state").textContent =
      "WebGPU unavailable · stored sample experiments still work";
}
init();
