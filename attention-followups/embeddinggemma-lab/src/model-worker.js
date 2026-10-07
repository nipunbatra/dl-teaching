import {
  AutoModel,
  AutoProcessor,
  RawImage,
  RawVideo,
  RawVideoFrame,
  Tensor,
  env,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1";
import { MODEL, REVISION, DTYPE } from "./config.js";
env.allowLocalModels = false;
env.backends.onnx.wasm.numThreads = 1;
let loading;
function disposeTree(value, seen = new Set()) {
  if (!value || typeof value !== "object" || seen.has(value)) return;
  seen.add(value);
  if (value instanceof Tensor) {
    try {
      value.dispose();
    } catch {}
  } else if (!ArrayBuffer.isView(value))
    Object.values(value).forEach((v) => disposeTree(v, seen));
}
async function load() {
  return (loading ??= (async () => {
    const processor = await AutoProcessor.from_pretrained(MODEL, {
      revision: REVISION,
    });
    const model = await AutoModel.from_pretrained(MODEL, {
      revision: REVISION,
      device: "webgpu",
      dtype: DTYPE,
      progress_callback: (progress) =>
        self.postMessage({ type: "progress", progress }),
    });
    return { processor, model };
  })().catch((e) => {
    loading = null;
    throw e;
  }));
}
let queue = Promise.resolve();
self.onmessage = ({ data }) => {
  queue = queue.then(async () => {
    let inputs, outputs;
    try {
      const { processor, model } = await load();
      if (data.action === "load") {
        self.postMessage({ id: data.id, result: { ready: true } });
        return;
      }
      const item = data.input;
      let text = item.text ?? null,
        image = null,
        audio = null,
        video = null;
      if (item.image) image = await RawImage.fromBlob(item.image);
      if (item.audio) audio = item.audio;
      if (item.video)
        video = new RawVideo(
          item.video.frames.map(
            (f) =>
              new RawVideoFrame(
                new RawImage(f.data, f.width, f.height, 4),
                f.timestamp,
              ),
          ),
          item.video.duration,
        );
      const start = performance.now();
      inputs = await processor(text, image, audio, video);
      const shapes = Object.fromEntries(
        Object.entries(inputs)
          .filter(([, v]) => v?.dims)
          .map(([k, v]) => [k, v.dims]),
      );
      outputs = await model(inputs);
      const vector = Array.from(outputs.sentence_embedding.data);
      if (vector.length !== 768 || !vector.every(Number.isFinite))
        throw Error(
          "The model returned an invalid embedding. Try reloading the model.",
        );
      self.postMessage({
        id: data.id,
        result: { vector, shapes, elapsed: performance.now() - start },
      });
    } catch (error) {
      self.postMessage({ id: data.id, error: String(error.message ?? error) });
    } finally {
      disposeTree([inputs, outputs]);
    }
  });
};
