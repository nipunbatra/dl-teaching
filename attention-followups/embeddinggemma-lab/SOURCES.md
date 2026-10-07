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

10 five-second clips from the **ESC-10 subset** of ESC-50, one per class. ESC-10 is distributed under **CC BY 3.0** (not the broader ESC-50 non-commercial license). The full original license/individual Freesound attributions are retained at `public/ESC-50-LICENSE.txt`; each selected clip's attribution is also in the gallery manifest.

- Repository: https://github.com/karolpiczak/ESC-50
- Metadata: https://github.com/karolpiczak/ESC-50/blob/master/meta/esc50.csv
- License: https://github.com/karolpiczak/ESC-50/blob/master/LICENSE
- K. J. Piczak, *ESC: Dataset for Environmental Sound Classification*, ACM Multimedia, 2015, https://doi.org/10.1145/2733373.2806390

Raw waveform input is mixed to mono and resampled to 16 kHz in the browser. No speech transcription or sound labels are sent with the waveform to the model. The decorative waveform symbol in the UI is an icon, not a measured signal plot.

## Video

`public/media/three-scenes.mp4` is an authored silent teaching slideshow: 0–4 s Chelsea cat (Stefan van der Walt, CC0), 4–8 s coffee (Rachel Michetti, CC0), 8–12 s rocket (SpaceX, CC0). Each image is letterboxed to 640×360; the slideshow is 12 fps. For inference, one frame per second is sampled and encoded as a native video input. The full video and each 4 s window were separately embedded. This does not assess motion or audio/video fusion.

Reproduce with FFmpeg using each original image as a looped 4-second input, `scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2,setsar=1` on each stream, then the concat video filter, libx264, yuv420p and `-movflags +faststart`.

## Numerical conventions

Text queries have the model's task prefix. Text collection items use `title: … | text: …`; captions use title `none`. The classification experiment embeds the input and supplied text labels with the classification prefix where applicable. Media inputs never include sample titles or labels. Image-plus-text passes `note <|image|>` plus pixels as one processor call.

The model returns normalized 768-dimensional vectors. The size selector slices the same leading d coordinates from both query and candidates, then explicitly normalizes again. The dot product sums all selected dimensions. Difference queries are `unit(after − before)`; swapping before/after reverses signs. K-means runs in the selected full vector space; centered PCA is only a display projection. No thresholds turn similarity into a truth claim.
