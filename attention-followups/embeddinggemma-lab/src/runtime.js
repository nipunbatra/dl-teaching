export class Engine {
  constructor(onProgress = () => {}) {
    this.onProgress = onProgress;
    this.pending = new Map();
    this.sequence = 0;
  }
  start() {
    if (this.worker) return;
    this.worker = new Worker(new URL("./model-worker.js", import.meta.url), {
      type: "module",
    });
    this.worker.onmessage = ({ data }) => {
      if (data.type === "progress") return this.onProgress(data.progress);
      const entry = this.pending.get(data.id);
      if (entry) {
        this.pending.delete(data.id);
        data.error
          ? entry.reject(Error(data.error))
          : entry.resolve(data.result);
      }
    };
    this.worker.onerror = (event) =>
      this.stop(
        Error(event.message || "The model worker stopped. Try loading again."),
      );
  }
  request(action, input) {
    this.start();
    return new Promise((resolve, reject) => {
      const id = ++this.sequence;
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ id, action, input });
    });
  }
  load() {
    return this.request("load");
  }
  embed(input) {
    return this.request("embed", input);
  }
  stop(
    reason = Error("Stopped. Your results are kept; run again when ready."),
  ) {
    this.worker?.terminate();
    this.worker = null;
    for (const entry of this.pending.values()) entry.reject(reason);
    this.pending.clear();
  }
}
export async function decodeAudio(blob) {
  const ctx = new AudioContext();
  let decoded;
  try {
    decoded = await ctx.decodeAudioData(await blob.arrayBuffer());
  } finally {
    await ctx.close();
  }
  const duration = Math.min(20, decoded.duration);
  const mono = new OfflineAudioContext(1, Math.ceil(duration * 16000), 16000);
  const source = mono.createBufferSource();
  source.buffer = decoded;
  source.connect(mono.destination);
  source.start();
  const rendered = await mono.startRendering();
  return {
    audio: rendered.getChannelData(0),
    info: {
      sampleRate: 16000,
      duration,
      originalDuration: decoded.duration,
      channels: 1,
    },
  };
}
export async function decodeVideo(blob, start = 0, end = 20) {
  const url = URL.createObjectURL(blob),
    el = document.createElement("video");
  el.muted = true;
  el.preload = "auto";
  try {
    await new Promise((resolve, reject) => {
      el.onloadedmetadata = resolve;
      el.onerror = () =>
        reject(Error("This video could not be decoded. Try an MP4 file."));
      el.src = url;
    });
    const stop = Math.min(el.duration, end),
      duration = stop - start;
    if (!(duration > 0)) throw Error("The video has no readable frames.");
    const canvas = document.createElement("canvas");
    const ratio = Math.min(1, 384 / Math.max(el.videoWidth, el.videoHeight));
    canvas.width = Math.max(1, Math.round(el.videoWidth * ratio));
    canvas.height = Math.max(1, Math.round(el.videoHeight * ratio));
    const ctx = canvas.getContext("2d", { willReadFrequently: true }),
      frames = [];
    for (let t = start + 0.05; t < stop; t += 1) {
      await new Promise((resolve, reject) => {
        el.onseeked = resolve;
        el.onerror = () => reject(Error("Could not read a video frame."));
        el.currentTime = t;
      });
      ctx.drawImage(el, 0, 0, canvas.width, canvas.height);
      frames.push({
        data: ctx.getImageData(0, 0, canvas.width, canvas.height).data,
        width: canvas.width,
        height: canvas.height,
        timestamp: t - start,
      });
    }
    return {
      video: { frames, duration },
      info: {
        duration,
        originalDuration: el.duration,
        frames: frames.length,
        fps: 1,
      },
    };
  } finally {
    el.removeAttribute("src");
    el.load();
    URL.revokeObjectURL(url);
  }
}
export async function prepare(
  item,
  task = "search result",
  role = "query",
  extraText = "",
) {
  if (item.type === "text") {
    const text =
      role === "document"
        ? `title: ${item.group === "caption" ? "none" : item.title} | text: ${item.text}`
        : `task: ${task} | query: ${item.text}`;
    return { input: { text }, info: { text, characters: text.length } };
  }
  const blob =
    item.blob ||
    (await fetch(item.src).then((r) => {
      if (!r.ok) throw Error("Sample could not be loaded.");
      return r.blob();
    }));
  if (item.type === "image")
    return {
      input: {
        image: blob,
        ...(extraText ? { text: `${extraText} <|image|>` } : {}),
      },
      info: { modality: "pixels", ...(extraText ? { text: extraText } : {}) },
    };
  const decoded =
    item.type === "audio"
      ? await decodeAudio(blob)
      : await decodeVideo(blob, item.start || 0, item.end || 20);
  return { input: { [item.type]: decoded[item.type] }, info: decoded.info };
}
