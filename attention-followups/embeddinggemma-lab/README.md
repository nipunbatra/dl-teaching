# EmbeddingGemma 2: multimodal playground

Explore **EmbeddingGemma 2** through search, classification, clustering and vector inspection. Text, images, sound and video share an embedding space; the applications show how to use it.

## Run and build

```sh
npm ci
npm run dev       # http://127.0.0.1:5189
npm test
npm run build     # deploy dist/ under any path; assets use relative URLs
```

Use current Chrome or Edge with WebGPU and hardware acceleration. HTTPS or localhost is required. The pinned Transformers.js browser module loads from jsDelivr; model weights load from Hugging Face. The first model download is about 473 MB, plus runtime files; Transformers.js uses the browser cache. No API key or inference server is used. User inputs are processed locally. The page still needs network access to load its code, samples and model files initially; this is not an installable offline app.

## Explore the capabilities

Choose an application directly from the navigation. There is no chapter sequence or presentation mode. The page opens with text-to-image search; the picture and sound entry points use saved sample vectors immediately. **Load WebGPU for new inputs** warms the model for typed queries and uploads.

Audio previews show actual waveform envelopes, scaled to each recording's peak. Regenerate them with `python3 scripts/prepare_waveforms.py` after changing the sound collection. These plots show amplitude over time, not embedding coordinates.

## Applications and controls

1. **Words → pictures:** predict the winner, run, inspect its vector.
2. **Words → sounds**, then **Sound → pictures & words:** play the bark. Try the same query against different candidate modalities.
3. **Picture → captions** and **Picture → pictures:** Try the retriever photograph and sketch.
4. **Search in another language:** change Hindi to English. The image index stays fixed.
5. **Picture + words:** a native joint input, not an average of separate embeddings.
6. **Find a video moment:** search excerpts of a puppy playing, a coffee machine, waves and geese, alongside the original three-scene slideshow.
7. **Choose your own labels:** edit the class menu. Removing the correct label does not make another label true.
8. **Retrieve a course passage** or **Search code by its job:** show retrieval before generation. The lab does not generate answers.
9. **What changed?** subtract the original portrait embedding from the hat portrait embedding, normalize, then compare against captions. Swap the pair and check the signs.
10. **Group the collection:** inspect k-means groups and a 2D PCA projection. Modality can dominate a group.
11. **Explore every embedding:** choose an image, text, recording or video in the dropdown. Compare any two, inspect all coordinates, and export vectors plus the PCA coordinates. Filter modalities, click points, zoom and pan. The map reports explained variance; neighbours always use high-dimensional cosine. Modality-specific score ranges can dominate the nearest-neighbour list.
12. **Train a small classifier:** start with three sound classes, step through learning, then try all ten. Play the held-out recordings before revealing labels. The audio encoder is frozen; only a linear softmax head learns. Try to explain why training loss can improve while held-out accuracy drops.
13. Shorten vectors to **512 / 256 / 128**, re-normalize, and look for rank changes. This shrinks the index, not the model download.

The landing page offers picture, sound and text starting points. The workspace keeps the query beside its candidates; on mobile these stack. **Explain score** opens the calculation only when requested, and **Upload or re-encode an input** reveals the optional input controls.

Every result can be inspected: query and candidate vectors, exact processor inputs, tensor shapes, unit norms, eight coordinate products at a time, the full dot-product sum, and JSON vector export. Upload an image, audio file or video to run a fresh example locally. Audio/video input is bounded to the first 20 seconds; video uses frames only at 1 fps.

## Exact model and saved index

- Model: `onnx-community/embeddinggemma-2-ONNX`
- Revision: `daa72c51243991dfcaf9f9137d2c573d8f7790c0`
- Transformers.js: `4.3.1`
- Device/precision: WebGPU / q4 for all components
- Output: `[1, 768]`, L2-normalized
- Stored index: `public/embeddings.json`, containing all 154 sample embeddings **actually computed with this configuration** in Chrome, plus tensor shapes, processor metadata and timings.
- Sample inventory: 29 images (including four extracted video frames), 40 five-second sounds, 59 captions/phrases in five languages, 6 course passages, 4 code snippets, 5 full clips and 11 indexed video windows. There are four real 12-second videos and one teaching slideshow.

New text queries, classification labels, joint inputs and uploads are computed on the student's device. Stored sample queries can use the saved vectors without downloading the model. The **Re-encode this sample on my device** option recomputes them. Scores are raw cosines; no label matching, fabricated scores or per-modality score offsets are used. Rankings across modalities may reflect different score distributions. q4 is a size/quality tradeoff; these examples are not a benchmark.

`src/model-worker.js` owns inference and disposes tensors. `src/runtime.js` owns media decoding and the worker lifecycle. `src/math.js` contains normalization, cosine ranking, delta, k-means and PCA. `src/experiments.js` defines the applications. `src/main.js` is the retrieval UI. `src/explorer.js` implements the modality-coloured PCA map and complete vector table; `src/training.js` implements the supervised learning view. There are no server components.

## Reproduce the index and checks

`prepare_gallery.py` copies the shared CLIP images and fetches the attributed ESC-10 sounds. See `SOURCES.md` for the video-generation command. Start the Vite server, then run:

```sh
python3 scripts/prepare_gallery.py
python3 scripts/prepare_video.py  # requires ffmpeg
python3 scripts/expand_gallery.py # requests + ffmpeg; downloads attributed media
# Open http://127.0.0.1:5189/index-builder.html
# Click Encode missing samples, then Download index JSON.
# Save the downloaded JSON as public/embeddings.json.
npm test
node scripts/browser-check.cjs
node scripts/edge-check.cjs
```

The browser scripts use an installed Chrome and Puppeteer from the parent course repository. They save evidence in `output/verification/`. The index-builder page runs missing samples through the real browser model and saves progress locally. Changed text is re-encoded when its processor text differs; after changing media bytes, remove its entry from public/embeddings.json and clear the index-builder local storage before rebuilding. The original `encode-gallery.cjs` is an alternative full-index rebuild. The legacy browser check covers the original 13 retrieval applications, tests truncation and coordinate inspection, recomputes image/audio inputs, uploads an image, and checks 390 px mobile layouts. `LAB_URL` can target a production build or the published URL. The scripts can be adapted to your Chrome/Puppeteer locations.

## Sources and limitations

See `SOURCES.md`, `public/IMAGE-CREDITS.md`, and `public/ESC-50-LICENSE.txt`, and `public/EXPANDED-MEDIA-CREDITS.md`. Per-item credits also appear in the collection dialog.

Videos include four attributed real recordings and the original teaching slideshow. Excerpts and extracted frames from the same video are related examples, not independent evaluation data. The model can misrank examples, be affected by language, prompts, quantization and modality, or miss a relation described in text. A high cosine does not establish factual truth, and a delta is not a causal explanation. The training tab fits a small linear output layer to frozen embeddings. It does not update EmbeddingGemma weights. Official model fine-tuning notebooks are linked separately.


## Training and projection checks

`npm test` checks all 154 normalized 768-dimensional vectors and media hashes, exact PCA variance on a known matrix, linear-head gradients against finite differences, train/test source separation, and six training runs (3/10 classes × 768/256/128 dimensions). It saves `output/verification/expanded-math.json`.

The audio teaching split uses four different original source recordings per category: three training, one held out. The default three-class head has nine training examples and three held-out examples. All ten classes have 30 training and 10 held-out. This is far too small for a reliable performance claim; repeatedly selecting settings on these examples biases the result. The split is about teaching the mechanism, not benchmarking.

Browser acceptance checks for the two added views: select each modality, compare two items, open every coordinate, switch dimensions, filter/zoom/pan/reset PCA, step/train/reset the classifier, change 3 to 10 classes, inspect labels, and test the 390 px layout. Vector and weight exports include model revision and dimensions. Full-model fine-tuning requires a separate GPU runtime; see the official links in the training view.

## Implementation notes from review

Re-encoded media is held separately in `state.localVectors`; it does not replace the saved candidate index. PCA plots preserve the same units on both axes. The explorer can show top three neighbours per modality, grouped without score offsets.
