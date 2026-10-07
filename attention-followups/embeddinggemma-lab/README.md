# Beyond CLIP: a multimodal embedding lab

A browser lab for the students who have just worked through [CLIP](https://nipunbatra.github.io/attention/clip/). It uses **EmbeddingGemma 2**, not OpenAI CLIP, to put text, images, sound and video into one space.

## Run and build

```sh
npm ci
npm run dev       # http://127.0.0.1:5189
npm test
npm run build     # deploy dist/ under any path; assets use relative URLs
```

Use current Chrome or Edge with WebGPU and hardware acceleration. HTTPS or localhost is required. The first model download is about 473 MB, plus runtime files; Transformers.js uses the browser cache. No API key or inference server is used. User inputs are processed locally. The page still needs network access to load its code, samples and model files initially; this is not an installable offline app.

## Classroom route (15–25 minutes)

1. **Words → pictures:** predict the winner, run, inspect its vector.
2. **Words → sounds**, then **Sound → pictures & words:** play the bark. Try the same query against different candidate modalities.
3. **Picture → captions** and **Picture → pictures:** connect back to the CLIP lab. Try the retriever photograph and sketch.
4. **Search in another language:** change Hindi to English. The image index stays fixed.
5. **Picture + words:** a native joint input, not an average of separate embeddings.
6. **Find a video moment:** search three four-second windows of a simple, silent teaching slideshow.
7. **Choose your own labels:** edit the class menu. Removing the correct label does not make another label true.
8. **Retrieve a course passage** or **Search code by its job:** show retrieval before generation. The lab does not generate answers.
9. **What changed?** subtract the original portrait embedding from the hat portrait embedding, normalize, then compare against captions. Swap the pair and check the signs.
10. **Group the collection:** inspect k-means groups and a 2D PCA projection. Modality can dominate a group.
11. Shorten vectors to **512 / 256 / 128**, re-normalize, and look for rank changes. This shrinks the index, not the model download.

Every result can be inspected: query and candidate vectors, exact processor inputs, tensor shapes, unit norms, eight coordinate products at a time, the full dot-product sum, and JSON vector export. Upload an image, audio file or video to run a fresh example locally. Audio/video input is bounded to the first 20 seconds; video uses frames only at 1 fps.

## Exact model and saved index

- Model: `onnx-community/embeddinggemma-2-ONNX`
- Revision: `daa72c51243991dfcaf9f9137d2c573d8f7790c0`
- Transformers.js: `4.3.1`
- Device/precision: WebGPU / q4 for all components
- Output: `[1, 768]`, L2-normalized
- Stored index: `public/embeddings.json`, containing all 68 sample embeddings **actually computed with this configuration** in Chrome, plus tensor shapes, processor metadata and timings.
- Sample inventory: 17 images, 27 captions, 10 five-second sounds, 6 course passages, 4 code snippets, one 12-second teaching slideshow and its 3 windows.

New text queries, classification labels, joint inputs and uploads are computed on the student's device. Stored sample queries can use the saved vectors without downloading the model. The **Re-encode this sample on my device** option recomputes them. Scores are raw cosines; no label matching, fabricated scores or per-modality score offsets are used. Rankings across modalities may reflect different score distributions. q4 is a size/quality tradeoff; these examples are not a benchmark.

`src/model-worker.js` owns inference and disposes tensors. `src/runtime.js` owns media decoding and the worker lifecycle. `src/math.js` contains normalization, cosine ranking, delta, k-means and PCA. `src/experiments.js` contains the guided questions. `src/main.js` is the UI. There are no server components.

## Reproduce the index and checks

`prepare_gallery.py` copies the shared CLIP images and fetches the attributed ESC-10 sounds. See `SOURCES.md` for the video-generation command. Start the Vite server, then run:

```sh
python3 scripts/prepare_gallery.py
python3 scripts/prepare_video.py  # requires ffmpeg
node scripts/encode-gallery.cjs
node scripts/browser-check.cjs
node scripts/edge-check.cjs
```

The browser scripts use an installed Chrome and Puppeteer from the parent course repository. They save evidence in `output/verification/`. The index script runs each sample through the real browser model. The browser check runs all 13 applications, tests truncation and coordinate inspection, recomputes image/audio inputs, uploads an image, and checks 390 px mobile layouts. `LAB_URL` can target a production build or the published URL. The scripts can be adapted to your Chrome/Puppeteer locations.

## Sources and limitations

See `SOURCES.md`, `public/IMAGE-CREDITS.md`, and `public/ESC-50-LICENSE.txt`. Per-item credits also appear in the collection dialog.

The video is a deliberately simple slideshow of already credited pictures, not a motion benchmark. The model can misrank examples, be affected by language, prompts, quantization and modality, or miss a relation described in text. A high cosine does not establish factual truth, and a delta is not a causal explanation. No training or fine-tuning happens here.
