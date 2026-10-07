# Delivery checklist

- [x] Read the Google announcement and the linked video’s English captions.
- [x] Run the real, pinned EmbeddingGemma 2 ONNX model on WebGPU.
- [x] 294 attributed samples: images, audio, captions, passages, code and video/windows.
- [x] Image/caption retrieval, image neighbours, and sound/image/text search.
- [x] Multilingual text search and native image-plus-text input.
- [x] Video-window retrieval and live uploaded-video encoding.
- [x] Editable-label classification, course-passage retrieval for RAG, and code search.
- [x] Before/after vector difference, with sign reversal checked.
- [x] K-means and a clearly labelled approximate PCA plot.
- [x] 768 / 512 / 256 / 128 dimensions; slice both sides and re-normalize.
- [x] Input shapes, vector heatmaps, coordinate products and full cosine sum.
- [x] JSON vector export; local image/audio/video uploads.
- [x] Genuine measured saved index, clearly distinguished from live computation.
- [x] Model progress, cancellation/retry, error handling and no-WebGPU sample fallback.
- [x] Editorial redesign: visual entry points, side-by-side query/results, six initial matches, on-demand score explanation.
- [x] Desktop and 390 px mobile checks; media playback metadata checked.
- [x] Build instructions, primary-source links, image/audio credits and library licenses.

The app supports inference and local vector operations. It does not fine-tune the model, generate answers, or claim that scores are calibrated confidence. Video includes four real clips and the original teaching slideshow. A small linear head trains on frozen audio embeddings, with separate source recordings held out.


## Expanded exploration

- [x] 69 images, 100 recordings, 109 text items, 5 clips and 11 indexed windows.
- [x] Real puppy, coffee, waves and geese videos with credited adaptations.
- [x] Dropdowns for any item and a comparison item.
- [x] All 768 coordinates available, with exact coordinate products and JSON export.
- [x] Interactive PCA with modality colours, filters, point selection, zoom, pan and reset.
- [x] Explained variance and actual high-dimensional nearest neighbours.
- [x] Real classifier training: one step, 100 steps, reset, loss curves, held-out predictions.
- [x] Three- and ten-class tasks; 768/256/128-dimensional features.
- [x] No shared original audio source across training and held-out splits.
- [x] Official text/image/audio fine-tuning notebooks linked; encoder/head distinction explicit.
- [x] Finite-difference gradient checks, PCA reference check and all media hashes.

## Capability-first restoration

- [x] EmbeddingGemma 2 branding and direct application navigation restored.
- [x] Guided chapters and recording mode removed; older lesson links open the playground.
- [x] All fifteen applications and the original collection retained and expanded.
- [x] Review fixes retained: separate live query cache, reduced-dimension delta, equal-axis PCA, clear cosine labels and per-modality neighbours.
- [x] Real waveform previews and WebGPU warm-up retained.


## Course retrieval and larger collection

- [x] 40 additional Commons images and their source captions; exact attribution retained.
- [x] 60 additional ESC-10 recordings, with actual waveform previews.
- [x] All 294 gallery items encoded by the pinned WebGPU model.
- [x] 1,672 slide passages from 17 published decks, with exact page/anchor links.
- [x] 269 real code chunks from 26 course notebooks, with GitHub and Colab links.
- [x] Cited source excerpts, PDF previews and neighbouring slides; no generated answers.
- [x] Twelve instant example queries plus live questions on WebGPU.
- [x] Optional full embeddings, exact model inputs, cosine scores and dimensionality control.
- [x] Paste, embed and search a new local passage/code snippet; remove or reload to discard.
- [x] Separate lazy-loaded course index; media applications retain direct navigation.
- [x] Reproducible extraction/index-building scripts and retrieval checks.
