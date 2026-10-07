import { waveform } from "./waveform.js";
import { unit, dot, rank, pcaProjection } from "./math.js";
import { MODEL, REVISION } from "./config.js";
export const escapeHTML = (s) =>
  String(s ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const esc = escapeHTML;
const colours = {
  image: "#bc5538",
  text: "#476b9b",
  audio: "#238178",
  video: "#835f9f",
};
export function preview(item) {
  if (item.type === "image")
    return `<img src="${esc(item.src)}" alt="${esc(item.title)}" loading="lazy">`;
  if (item.type === "audio")
    return `<div class="audio-label">AUDIO · ${item.duration || 5} SECONDS</div>${waveform(item.id)}<audio controls preload="none" src="${esc(item.src)}"></audio>`;
  if (item.type === "video")
    return `<video controls playsinline preload="metadata" src="${esc(item.src)}#t=${item.start || 0.05},${item.end || 20}" aria-label="${esc(item.title)}"></video>`;
  return `<blockquote>${esc(item.text)}</blockquote>`;
}
export function saveJSON(data, name) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(
    new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }),
  );
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
function drawVector(canvas, v, max) {
  const ctx = canvas.getContext("2d"),
    cols = 48,
    w = canvas.clientWidth || 500,
    h = Math.ceil(v.length / cols) * 10,
    dpr = devicePixelRatio || 1;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  canvas.style.height = h + "px";
  ctx.scale(dpr, dpr);
  v.forEach((x, i) => {
    ctx.fillStyle =
      x >= 0
        ? `rgba(35,129,120,${0.08 + (0.92 * Math.abs(x)) / max})`
        : `rgba(188,85,56,${0.08 + (0.92 * Math.abs(x)) / max})`;
    ctx.fillRect(
      ((i % cols) * w) / cols,
      Math.floor(i / cols) * 10,
      w / cols - 1,
      9,
    );
  });
}
export function mountExplorer(root, state, initial = {}) {
  const items = state.gallery.filter((i) => state.vectors[i.id]);
  let selected = initial.selected || "sound-dog",
    compare = initial.compare || "newfoundland",
    dims = 768,
    projection,
    visible = new Set(Object.keys(colours)),
    zoom = 1,
    pan = [0, 0],
    groupNeighbours = false;
  const opts = () =>
    Object.keys(colours)
      .map(
        (type) =>
          `<optgroup label="${type.toUpperCase()}">${items
            .filter((i) => i.type === type)
            .map((i) => `<option value="${esc(i.id)}">${esc(i.title)}</option>`)
            .join("")}</optgroup>`,
      )
      .join("");
  root.innerHTML = `<div class="explorer-intro"><p><strong>Start with a bark.</strong> Select its point, play the recording, then compare it with a dog photograph. Do their closest neighbours agree?</p><span class="saved-tag">Stored real embeddings · no model download</span></div>
 <div class="space-toolbar"><label>Choose an item<select id="space-item">${opts()}</select></label><label>Compare with<select id="space-compare"><option value="">No comparison</option>${opts()}</select></label><label>Vector size<select id="space-dim"><option value="768">768</option><option>512</option><option>256</option><option>128</option></select></label></div>
 <div class="space-layout"><section class="space-plot"><header><div><p class="step-label">01 / SEE THE COLLECTION</p><h3>A map of the shared space</h3></div><div class="zoom-tools"><button class="quiet" id="zoom-out" aria-label="Zoom out">−</button><button class="quiet" id="zoom-in" aria-label="Zoom in">+</button><button class="quiet" id="zoom-reset">Reset view</button></div></header><div class="modality-legend">${Object.entries(
   colours,
 )
   .map(
     ([t, c]) =>
       `<label style="--modality:${c}"><input type="checkbox" value="${t}" checked><i></i>${t}<span>${items.filter((i) => i.type === t).length}</span></label>`,
   )
   .join(
     "",
   )}</div><div id="space-chart"></div><p id="space-variance" class="hint"></p><div class="pan-tools" aria-label="Pan the map"><span>Drag to pan, or use</span><button class="quiet" data-pan="left" aria-label="Pan left">←</button><button class="quiet" data-pan="right" aria-label="Pan right">→</button><button class="quiet" data-pan="up" aria-label="Pan up">↑</button><button class="quiet" data-pan="down" aria-label="Pan down">↓</button></div><details class="space-explanation"><summary>How do 768 numbers become two coordinates?</summary><ol><li>Stack all ${items.length} unit vectors into one matrix.</li><li>Subtract the mean vector. PCA finds the two directions with the most variation.</li><li>Project each centred vector onto those directions. Colour records the input modality; it does not change the calculation.</li></ol><p>The two axes lose information. Close points on this map need not be nearest neighbours in the original space. The neighbour list uses cosine in all selected dimensions. Text candidates keep their retrieval document prefix; this is a view of our retrieval index.</p><p>Filters hide points without moving the axes. Changing vector size recomputes PCA.</p></details></section><aside id="space-detail" class="space-detail"></aside></div>
 <section class="full-vector"><header><div><p class="step-label">02 / FOLLOW THE NUMBERS</p><h3 id="space-vector-title"></h3></div><button id="space-export" class="quiet">Download these vectors</button></header><div id="space-vector-summary"></div><div class="vector-pair"><div><h4 id="space-a-title"></h4><canvas id="space-a" aria-label="Selected embedding heatmap"></canvas></div><div id="space-b-wrap"><h4 id="space-b-title"></h4><canvas id="space-b" aria-label="Comparison embedding heatmap"></canvas></div></div><p class="hint">Teal = positive · rust = negative · the same scale for both vectors. These coordinates are learned numbers, not named concepts.</p><details id="all-coordinates"><summary id="coordinate-summary">See every coordinate</summary><div class="coordinate-search"><label>Jump to coordinate<input id="jump-coordinate" type="number" min="1" max="768" value="1"></label><button class="quiet" id="jump-go">Go</button></div><div class="full-coordinate-table" tabindex="0"><table><thead id="space-table-head"></thead><tbody id="space-table"></tbody></table></div></details><details><summary>Actual model inputs and tensor shapes</summary><pre id="space-shapes"></pre></details></section>`;
  const $ = (s) => root.querySelector(s),
    item = (id) => items.find((i) => i.id === id),
    vector = (id) => unit(state.vectors[id].vector, dims);
  $("#space-item").value = selected;
  $("#space-compare").value = compare;
  function calculate() {
    projection = pcaProjection(items.map((i) => vector(i.id)));
    zoom = 1;
    pan = [0, 0],
    groupNeighbours = false;
  }
  function chart() {
    const p = projection.points,
      maxX = Math.max(...p.map((x) => Math.abs(x[0])), 0.01),
      maxY = Math.max(...p.map((x) => Math.abs(x[1])), 0.01),
      scale = Math.min(290 / maxX, 170 / maxY),
      sx = (x) => 360 + x * scale,
      sy = (y) => 220 - y * scale;
    const selectedIndex = items.findIndex((i) => i.id === selected),
      compareIndex = items.findIndex((i) => i.id === compare);
    const line =
      compareIndex >= 0 &&
      visible.has(item(selected).type) &&
      visible.has(item(compare).type)
        ? `<line class="comparison-line" x1="${sx(p[selectedIndex][0])}" y1="${sy(p[selectedIndex][1])}" x2="${sx(p[compareIndex][0])}" y2="${sy(p[compareIndex][1])}" stroke="#424b50" stroke-dasharray="5 5"/>`
        : "";
    $("#space-chart").innerHTML =
      `<svg viewBox="0 0 720 440" role="group" aria-label="Interactive PCA map, coloured by modality"><defs><clipPath id="plot-clip"><rect x="28" y="15" width="664" height="400"/></clipPath></defs><path d="M28 220H692M360 15V415" stroke="#e0dfd7"/><text x="674" y="437" class="axis-label">PC1</text><text x="6" y="22" class="axis-label">PC2</text><g clip-path="url(#plot-clip)"><g id="pan-layer" transform="translate(${360 + pan[0]},${220 + pan[1]}) scale(${zoom}) translate(-360,-220)">${line}${items.map((i, n) => (visible.has(i.type) ? `<g class="space-point ${i.id === selected ? "chosen" : ""}" role="button" tabindex="0" aria-label="${esc(i.type + ": " + i.title)}" data-point="${esc(i.id)}"><circle cx="${sx(p[n][0])}" cy="${sy(p[n][1])}" r="${i.id === selected || i.id === compare ? 9 : 5.5}" fill="${colours[i.type]}" stroke="${i.id === selected ? "#202621" : i.id === compare ? "#fff" : "transparent"}" stroke-width="${i.id === selected || i.id === compare ? 2.5 : 0}"/><title>${esc(i.type + " · " + i.title)}</title></g>` : "")).join("")}</g></g></svg>`;
    $("#space-variance").textContent =
      `PCA of ${items.length} × ${dims} numbers · PC1 ${(projection.variance[0] * 100).toFixed(1)}% + PC2 ${(projection.variance[1] * 100).toFixed(1)}% = ${(projection.variance.reduce((a, b) => a + b) * 100).toFixed(1)}% of variation shown. Zoom ${zoom.toFixed(1)}×.`;
    const svg = $("#space-chart svg");
    let drag = null;
    svg.onpointerdown = (e) => {
      if (e.target.closest("[data-point]")) return;
      drag = [e.clientX, e.clientY, ...pan];
      svg.setPointerCapture(e.pointerId);
    };
    svg.onpointermove = (e) => {
      if (!drag) return;
      const scale = 720 / svg.getBoundingClientRect().width;
      pan = [
        drag[2] + (e.clientX - drag[0]) * scale,
        drag[3] + (e.clientY - drag[1]) * scale,
      ];
      $("#pan-layer").setAttribute(
        "transform",
        `translate(${360 + pan[0]},${220 + pan[1]}) scale(${zoom}) translate(-360,-220)`,
      );
    };
    svg.onpointerup = () => {
      drag = null;
    };
    svg.onpointercancel = () => {
      drag = null;
    };
  }
  function details() {
    const a = item(selected),
      b = item(compare),
      va = vector(selected),
      vb = b ? vector(compare) : null;
    const ranked = rank(
      va,
      items.filter((i) => visible.has(i.type)),
      state.vectors,
      dims,
      [selected],
    );
    const neighbours = groupNeighbours ? [...visible].flatMap(type => ranked.filter(r => r.item.type === type).slice(0,3)) : ranked.slice(0,5);
    $("#space-detail").innerHTML =
      `<span class="type-badge" style="color:${colours[a.type]}">${a.type.toUpperCase()}</span><h3>${esc(a.title)}</h3><div class="selected-media">${preview(a)}</div><p class="fineprint">${esc(a.credit)}</p><h4>Closest in ${dims} dimensions</h4><p class="hint">Cosine similarity, using the visible modalities. Grouping does not rescale any score.</p><label class="check"><input id="space-group-neighbours" type="checkbox" ${groupNeighbours?"checked":""}> Top 3 per modality</label><div class="space-neighbours">${neighbours.map((r) => `<button data-neighbour="${esc(r.item.id)}"><i style="background:${colours[r.item.type]}"></i><span>${esc(r.item.title)}<small>${r.item.type}</small></span><b>${r.score.toFixed(4)}</b></button>`).join("") || "<p>No visible neighbours. Turn on a modality.</p>"}</div>${b ? `<div class="pair-score">${b.type === "image" ? `<img src="${esc(b.src)}" alt="${esc(b.title)}">` : ""}<span>${esc(a.title)}<br>↕<br>${esc(b.title)}</span><strong>${dot(va, vb).toFixed(4)}</strong><small>cosine in ${dims} dimensions</small></div>` : ""}`;
    $("#space-group-neighbours").onchange=e=>{groupNeighbours=e.target.checked;details();};
    $("#space-vector-title").textContent = `One item → ${dims} numbers`;
    $("#space-vector-summary").innerHTML =
      `<div class="under-hood"><span>${esc(a.type)} input</span><span>→</span><span>EmbeddingGemma 2</span><span>→</span><span>[1, 768]</span><span>→</span><span>${dims === 768 ? "L2-normalize" : `keep ${dims} + normalize`}</span></div><p>Length of the selected vector = <b>${Math.hypot(...va).toFixed(6)}</b>${vb ? ` · Σ aₖbₖ = <b>${dot(va, vb).toFixed(6)}</b> across all ${dims} coordinates.` : "."}</p>`;
    $("#space-a-title").textContent = a.title;
    $("#space-b-wrap").hidden = !b;
    if (b) $("#space-b-title").textContent = b.title;
    const max = Math.max(...va.map(Math.abs), ...(vb || []).map(Math.abs));
    drawVector($("#space-a"), va, max);
    if (b) drawVector($("#space-b"), vb, max);
    $("#coordinate-summary").textContent =
      `See all ${dims} coordinates${b ? " and their dot-product contributions" : ""}`;
    $("#space-table-head").innerHTML =
      `<tr><th>Coordinate</th><th>Selected vector</th>${b ? "<th>Comparison vector</th><th>aₖ × bₖ</th>" : ""}</tr>`;
    $("#space-table").innerHTML = va
      .map(
        (v, k) =>
          `<tr id="coordinate-${k + 1}"><td>${k + 1}</td><td>${v.toFixed(7)}</td>${b ? `<td>${vb[k].toFixed(7)}</td><td>${(v * vb[k]).toFixed(7)}</td>` : ""}</tr>`,
      )
      .join("");
    $("#jump-coordinate").max = dims;
    $("#space-shapes").textContent = JSON.stringify(
      {
        model: MODEL,
        revision: REVISION,
        selected: { title: a.title, ...state.vectors[a.id], vector: undefined },
        ...(b
          ? {
              comparison: {
                title: b.title,
                ...state.vectors[b.id],
                vector: undefined,
              },
            }
          : {}),
        outputShape: [1, 768],
        displayedShape: [1, dims],
      },
      null,
      2,
    );
  }
  function render() {
    chart();
    details();
  }
  $("#space-item").onchange = (e) => {
    selected = e.target.value;
    render();
  };
  $("#space-compare").onchange = (e) => {
    compare = e.target.value;
    render();
  };
  $("#space-dim").onchange = (e) => {
    dims = +e.target.value;
    calculate();
    render();
  };
  root.querySelectorAll(".modality-legend input").forEach(
    (n) =>
      (n.onchange = () => {
        n.checked ? visible.add(n.value) : visible.delete(n.value);
        render();
      }),
  );
  $("#zoom-in").onclick = () => {
    zoom = Math.min(5, zoom * 1.4);
    chart();
  };
  $("#zoom-out").onclick = () => {
    zoom = Math.max(0.5, zoom / 1.4);
    chart();
  };
  $("#zoom-reset").onclick = () => {
    zoom = 1;
    pan = [0, 0],
    groupNeighbours = false;
    chart();
  };
  root.onclick = (e) => {
    const p = e.target.closest("[data-point]"),
      n = e.target.closest("[data-neighbour]"),
      arrow = e.target.closest("[data-pan]");
    if (p) {
      selected = p.dataset.point;
      $("#space-item").value = selected;
      render();
    }
    if (n) {
      compare = n.dataset.neighbour;
      $("#space-compare").value = compare;
      render();
    }
    if (arrow) {
      const moves = {
        left: [40, 0],
        right: [-40, 0],
        up: [0, 40],
        down: [0, -40],
      };
      pan = pan.map((v, i) => v + moves[arrow.dataset.pan][i]);
      chart();
    }
  };
  root.onkeydown = (e) => {
    if (
      (e.key === "Enter" || e.key === " ") &&
      e.target.matches("[data-point]")
    ) {
      e.preventDefault();
      e.target.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    }
  };
  $("#jump-go").onclick = () => {
    const k = Math.max(1, Math.min(dims, +$("#jump-coordinate").value || 1));
    $(`#coordinate-${k}`).scrollIntoView({ block: "nearest" });
  };
  $("#space-export").onclick = () =>
    saveJSON(
      {
        model: MODEL,
        revision: REVISION,
        dimensions: dims,
        selected: { id: selected, vector: vector(selected) },
        ...(compare
          ? {
              comparison: { id: compare, vector: vector(compare) },
              cosine: dot(vector(selected), vector(compare)),
            }
          : {}),
        projection: {
          method: "PCA",
          explainedVariance: projection.variance,
          points: items.map((i, n) => ({
            id: i.id,
            type: i.type,
            xy: projection.points[n],
          })),
        },
      },
      "embedding-space.json",
    );
  calculate();
  render();
  return () => {
    root.onclick = null;
    root.onkeydown = null;
  };
}
