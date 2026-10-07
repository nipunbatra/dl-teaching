# Sources, provenance and reproducibility

Reviewed 7 October 2026.

- Google announcement: https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2/
- Google for Developers introduction (English captions read): https://www.youtube.com/watch?v=anPsS6huQk0
- Google model card: https://huggingface.co/google/embeddinggemma-2
- ONNX implementation, browser API and precision guidance: https://huggingface.co/onnx-community/embeddinggemma-2-ONNX/tree/daa72c51243991dfcaf9f9137d2c573d8f7790c0
- Official-linked browser demo, consulted for feasibility/API use: https://huggingface.co/spaces/webml-community/embeddinggemma-2-webgpu
- Transformers.js: https://github.com/huggingface/transformers.js (Apache 2.0)

The application is original course code. It does not copy the demo's UI, minified bundle, precomputed index, or its 5,000-image gallery. The model is Apache 2.0 licensed. The source demo adjusts scores across modalities; this lab deliberately displays unadjusted cosines for teaching the arithmetic.

## Images and text

17 images copied unchanged from the CLIP applications notebook gallery in `../clip/lab/clip-mini-gallery`. The copied `public/IMAGE-CREDITS.md` gives original attribution. The collection includes Oxford-IIIT Pet examples, CC0 photographs, a public-domain NASA photograph, and explicitly labelled generated teaching pairs/scenes. Captions, passages and code snippets are course-authored. File hashes are in `public/gallery.json`.

## Audio

100 five-second clips from the **ESC-10 subset** of ESC-50. The original four per class have distinct source recordings; the additional six per class are exploration-only and do not enter the training/test split. ESC-10 is distributed under **CC BY 3.0** (not the broader ESC-50 non-commercial license). The full original license/individual Freesound attributions are retained at `public/ESC-50-LICENSE.txt`; each selected clip's attribution is also in the gallery manifest.

- Repository: https://github.com/karolpiczak/ESC-50
- Metadata: https://github.com/karolpiczak/ESC-50/blob/master/meta/esc50.csv
- License: https://github.com/karolpiczak/ESC-50/blob/master/LICENSE
- K. J. Piczak, *ESC: Dataset for Environmental Sound Classification*, ACM Multimedia, 2015, https://doi.org/10.1145/2733373.2806390

Raw waveform input is mixed to mono and resampled to 16 kHz in the browser. No speech transcription or sound labels are sent with the waveform to the model. The sound tile in the hero uses a decorative icon; individual sample previews show measured waveform envelopes.

## Video

`public/media/three-scenes.mp4` is an authored silent teaching slideshow: 0–4 s Chelsea cat (Stefan van der Walt, CC0), 4–8 s coffee (Rachel Michetti, CC0), 8–12 s rocket (SpaceX, CC0). Each image is letterboxed to 640×360; the slideshow is 12 fps. For inference, one frame per second is sampled and encoded as a native video input. The full video and each 4 s window were separately embedded. This does not assess motion or audio/video fusion.

Reproduce with FFmpeg using each original image as a looped 4-second input, `scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2,setsar=1` on each stream, then the concat video filter, libx264, yuv420p and `-movflags +faststart`.

## Numerical conventions

Text queries have the model's task prefix. Text collection items use `title: … | text: …`; captions use title `none`. The classification experiment embeds the input and supplied text labels with the classification prefix where applicable. Media inputs never include sample titles or labels. Image-plus-text passes `note <|image|>` plus pixels as one processor call.

The model returns normalized 768-dimensional vectors. The size selector slices the same leading d coordinates from both query and candidates, then explicitly normalizes again. The dot product sums all selected dimensions. Difference queries are `unit(after − before)`; swapping before/after reverses signs. K-means runs in the selected full vector space; centered PCA is only a display projection. No thresholds turn similarity into a truth claim.


## Expanded images, real videos and training

Eight additional photographs and four real videos come from Wikimedia Commons. Each video is trimmed to 12 seconds, resized and converted to silent H.264; one still and two 4-second windows are also indexed. The original authors, file pages, licenses and modifications appear in `public/EXPANDED-MEDIA-CREDITS.md` and the manifest. CC BY-SA media derivatives remain under their original share-alike licenses. Original downloads are kept only in `work/media-sources/`; redistribution uses the processed files under `public/media/`.

`expand_gallery.py` builds the added collection idempotently. It preserves the original 68 entries and adds 86 new items. Twenty English/Hindi/Gujarati/French/Spanish phrases are course-authored. New media embeddings use pixels/waveforms/frames alone, not the captions or file titles.

The linear-head example performs actual full-batch gradient descent on fixed embeddings, with mean softmax cross-entropy, learning rate 2 and L2 weight gradient coefficient 0.001. It starts W and b at zero. Three of four independent audio sources per class train the head; the fourth is held out. The chart excludes the weight penalty and shows cross-entropy in both splits. The full encoder is never trained by this browser page.

Official full-model fine-tuning references:
- Google/Sentence Transformers: https://ai.google.dev/gemma/docs/embeddinggemma/fine-tuning-embeddinggemma-with-sentence-transformers
- Unsloth documentation and image/text/audio notebooks: https://unsloth.ai/docs/models/embeddinggemma-2

The PCA explorer centres the selected-dimensional, re-normalized vectors and finds two principal directions by power iteration. Variance shown is the sum of squared projected coordinates divided by the total centred sum of squares. Filters hide points without refitting the axes. The neighbour list uses cosine on the selected-dimensional vectors, never 2D distances.


## Larger gallery and actual course sources (7 October 2026)

40 further Commons images and 40 file-description/title captions are listed in `public/MORE-MEDIA-CREDITS.md`. Their original authors, exact file pages, license names/URLs and media SHA-256 hashes are retained per item. The images include photographs, artwork and scientific imagery; search-query wording used to discover a file is never used as its image model input. All media keep their original licenses, including share-alike terms where applicable.

`public/course-manifest.json` enumerates 17 published decks. PDF handouts L1, L2, L3, L3A, L5, L5B, L5C and L7 come from `nipunbatra/dl-teaching`; the HTML lectures come from the published `attention` and `dl-teaching` checkouts. A total of 1,672 text passages retain slide anchors/PDF page numbers. The 269 code chunks come from 26 published notebooks. `public/course-corpus.json` records the source filename, source-file hash, extracted-text hash and exact links for every passage. Text is extracted, Unicode-normalized and split by slide/cell with line-bounded chunks; it is not a model-generated paraphrase. Code is displayed as an excerpt and may depend on earlier notebook cells.

`public/course-embeddings.json` and `public/course-query-embeddings.json` were computed on WebGPU with the same pinned model and q4 precision as the media gallery. Full metadata includes exact processor text, timings and shapes. Course embeddings use only extracted text, not slide images. The PDF thumbnails show the original page for visual context. Source material and thumbnails remain course-authored; third-party figures retain their original attributions in the linked slides.


## Picture and text classifier teaching splits

`src/training-data.js` selects existing gallery pictures and their captions for three supervised classes: Animals, Food & drink, Transport. Each class has four training items and two held-out items. The image task consumes the existing image embeddings; the text task consumes the existing caption embeddings. Class labels are manually assigned for these tasks and never appended to model inputs. Captions are the already attributed exact gallery text, not newly generated examples. Paired pictures/captions retain the same split, and source identifiers and content hashes are checked for overlap within each task. Existing media credits and licenses apply without changes.

The neuron diagram displays the first and last three selected embedding coordinates for readability; every coordinate contributes to every output neuron. It shows the actual learned head scores and softmax probabilities. The expanded arithmetic includes the first three products, the sum of all remaining products, and the bias. The line highlight identifies a class weight row; line width is not a weight magnitude. All displayed numbers are rounded only for presentation.
