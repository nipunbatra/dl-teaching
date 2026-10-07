import { neuronTrace } from "./math.js";
import { escapeHTML as esc } from "./explorer.js";
import { className } from "./training-data.js";

const f = (n, digits = 4) => (Math.abs(n) < 0.5 * 10 ** -digits ? 0 : n).toFixed(digits);
const sub = (n) => String(n).replace(/\d/g, (d) => "₀₁₂₃₄₅₆₇₈₉"[+d]);

export function sampleMedia(item, label = item.title) {
  if (item.type === "audio") return `<audio controls preload="none" src="${esc(item.src)}" aria-label="${esc(label)}"></audio>`;
  if (item.type === "image") return `<img src="${esc(item.src)}" alt="${esc(label)}" loading="lazy">`;
  return `<blockquote>${esc(item.text)}</blockquote>`;
}

export function networkDiagram(head, x, classes, selected, trueClass) {
  const d = x.length, C = classes.length;
  const trace = neuronTrace(head, x, selected);
  const height = C > 3 ? 800 : 470;
  const bottom = height - 55;
  const indices = [0, 1, 2, d - 3, d - 2, d - 1];
  const featureY = [0, 1, 2, 3.5, 4.5, 5.5].map((i) => 110 + i / 5.5 * (bottom - 110));
  const classY = classes.map((_, c) => 120 + c / (C - 1) * (bottom - 125));
  const line = (c, i) => `<path d="M124 ${featureY[i]} C215 ${featureY[i]}, 245 ${classY[c]}, 324 ${classY[c]}" class="nn-wire ${c === selected ? "selected" : ""}"/>`;
  const wires = classes.map((_, c) => c).filter((c) => c !== selected).concat(selected)
    .map((c) => indices.map((_, i) => line(c, i)).join("")).join("");
  const nodes = indices.map((k, i) => `<text x="12" y="${featureY[i] + 5}" class="nn-coordinate">x${sub(k + 1)}</text><circle cx="97" cy="${featureY[i]}" r="27" class="nn-feature"/><text x="97" y="${featureY[i] + 4}" text-anchor="middle" class="nn-value">${f(x[k])}</text>`).join("");
  const outputs = classes.map((label, c) => {
    const y = classY[c], p = trace.probabilities[c];
    return `<g class="nn-output ${selected === c ? "selected" : ""}">
      <text x="350" y="${y - 36}" text-anchor="middle" class="nn-class">${esc(className(label))}${c === trueClass ? " · target" : ""}</text>
      <circle cx="350" cy="${y}" r="27"/><text x="350" y="${y + 4}" text-anchor="middle" class="nn-value">${f(trace.logits[c], 3)}</text>
      <path d="M379 ${y} H486" class="nn-to-softmax" marker-end="url(#nn-arrow)"/>
      <rect x="518" y="${y - 12}" width="158" height="24" rx="3" class="nn-bar-track"/>
      <rect x="518" y="${y - 12}" width="${p * 158}" height="24" rx="3" class="nn-bar-fill"/>
      <text x="690" y="${y + 5}" class="nn-probability">${(p * 100).toFixed(1)}%</text>
    </g>`;
  }).join("");
  const svg = `<svg viewBox="0 0 760 ${height}" role="img" aria-label="${d} embedding features connect to ${C} class neurons. ${esc(className(classes[selected]))} is highlighted. All ${d} coordinates contribute to every class score.">
    <defs><marker id="nn-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10" fill="none" stroke="#9e9587"/></marker></defs>
    <text x="12" y="25" class="nn-heading">${d} features</text><text x="12" y="47" class="nn-caption">6 shown · all ${d} used</text>
    <text x="350" y="25" text-anchor="middle" class="nn-heading">${C} class neurons</text><text x="350" y="47" text-anchor="middle" class="nn-caption">score = wᵀx + b</text>
    <text x="518" y="25" class="nn-heading">Softmax</text><text x="518" y="47" class="nn-caption">across all ${C} scores · sum = 1</text>
    ${wires}${nodes}<text x="97" y="${(featureY[2] + featureY[3]) / 2 + 3}" text-anchor="middle" class="nn-ellipsis">⋮</text>${outputs}
    <text x="200" y="${height - 12}" text-anchor="middle" class="nn-caption">Connections are weights in W</text>
    <text x="555" y="${height - 12}" text-anchor="middle" class="nn-caption">Probabilities, not calibrated confidence</text>
  </svg>`;
  return { svg, trace };
}

export function renderNeuronArithmetic(target, head, x, classes, selected, trueClass, trace) {
  const label = esc(className(classes[selected]));
  target.innerHTML = `<div class="nn-formula-summary"><b>${label}: one neuron, ${x.length} weighted inputs</b><code>s = wᵀx + b = ${f(trace.logits[selected])} → p = ${(trace.probabilities[selected] * 100).toFixed(1)}%</code></div>
    <p class="hint">W has shape [${classes.length}, ${x.length}], with one row per class. b has ${classes.length} biases. Only these ${(classes.length * (x.length + 1)).toLocaleString("en")} parameters learn; the embedding stays fixed. This is a single output layer, with no hidden layer.</p>
    <details><summary>Open this neuron: see the multiplications</summary><div class="table-scroll"><table><thead><tr><th>Coordinate</th><th>Feature xₖ</th><th>Weight wₖ</th><th>Product wₖxₖ</th></tr></thead><tbody>${[0, 1, 2].map((k) => `<tr><th>${k + 1}</th><td>${f(x[k], 6)}</td><td>${f(head.weights[selected][k], 6)}</td><td>${f(trace.products[k], 6)}</td></tr>`).join("")}<tr><th colspan="3">Sum of the remaining ${x.length - 3} products</th><td>${f(trace.remainder, 6)}</td></tr><tr><th colspan="3">Add the bias b</th><td>${f(head.bias[selected], 6)}</td></tr><tr><th colspan="3">Total: class score s</th><td>${f(trace.logits[selected], 6)}</td></tr></tbody></table></div><p><code>p(${label}) = exp(s) / Σ exp(all class scores) = ${f(trace.probabilities[selected], 6)}</code></p><p>This example’s label is <b>${esc(className(classes[trueClass]))}</b>, so its loss is −ln(${f(trace.probabilities[trueClass], 6)}) = <b>${f(-Math.log(trace.probabilities[trueClass]), 6)}</b>. Training averages this loss over all labelled examples.</p><p class="hint">Displayed values are rounded. Computation uses full precision and numerically stable softmax.</p></details>`;
}
