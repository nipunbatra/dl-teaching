import { Engine, prepare } from "./runtime.js";
import { REVISION } from "./config.js";
const $ = (s) => document.querySelector(s);
const [gallery, data] = await Promise.all([
  fetch("./gallery.json").then((r) => r.json()),
  fetch("./embeddings.json").then((r) => r.json()),
]);
if (data.revision !== REVISION) throw Error("Wrong model revision");
const key = "index-builder-" + REVISION;
const cached = JSON.parse(localStorage.getItem(key) || "{}");
Object.assign(data.items, cached);
const pending = () =>
  gallery.filter(
    (x) =>
      !data.items[x.id] ||
      (x.type === "text" &&
        data.items[x.id].info?.text !==
          `title: ${x.group === "caption" ? "none" : x.title} | text: ${x.text}`),
  );
$("#status").textContent =
  `${gallery.length} samples · ${pending().length} missing`;
const engine = new Engine((p) => {
  if (p.status === "progress")
    $("#status").textContent =
      `Loading ${p.file} · ${Math.round(p.progress || 0)}%`;
});
$("#start").onclick = async () => {
  $("#start").disabled = true;
  try {
    await engine.load();
    let n = 0;
    const missing = pending();
    for (const item of missing) {
      $("#status").textContent =
        `Encoding ${++n}/${missing.length}: ${item.title}`;
      const { input, info } = await prepare(item, "search result", "document");
      const result = { ...(await engine.embed(input)), info };
      data.items[item.id] = result;
      cached[item.id] = result;
      localStorage.setItem(key, JSON.stringify(cached));
      $("#log").textContent +=
        `${item.id} · ${result.vector.length} dimensions · ${Math.round(result.elapsed)} ms\n`;
    }
    data.date = new Date().toISOString();
    $("#status").textContent =
      `Complete · ${gallery.length} real embeddings · ${pending().length} missing`;
  } catch (e) {
    $("#status").textContent = "ERROR: " + e.message;
  } finally {
    $("#start").disabled = false;
    engine.stop();
  }
};
$("#download").onclick = () => {
  data.date = new Date().toISOString();
  const a = document.createElement("a");
  a.href = URL.createObjectURL(
    new Blob([JSON.stringify(data)], { type: "application/json" }),
  );
  a.download = "expanded-embeddings.json";
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
};
