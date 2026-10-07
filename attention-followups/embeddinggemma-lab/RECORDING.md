# Recording the EmbeddingGemma lesson

Open https://nipunbatra.github.io/attention/embeddinggemma/?present#lesson at 1920 × 1080, with the browser at its usual zoom. Start on chapter 1. The guided route uses actual saved model outputs; it does not need a model download. Expand calculations when you want to pause on the arithmetic.

## Before pressing Record

- Test playback of the bark and coffee-machine clip; audio does not autoplay.
- Keep the student prediction visible when revealing the measured ranking.
- For live searches, exit recording view and choose **Load WebGPU for new inputs**. Wait for “model warmed”, then return to the lesson. The first download is roughly 473 MB.
- For a live upload, choose public teaching material. Processing happens in this browser.

## The narrative

**0–5 minutes: the familiar CLIP idea.** Show Chelsea and four captions. Let students predict. Reveal the vector, its unit norm, the scores and a coordinate-by-coordinate calculation. The captions were supplied; the model did not generate them.

**5–9 minutes: new modalities.** Play the bark and search pictures. Switch the candidates to sound descriptions: this recording ranks a crying-baby description above dog. Show the coffee-machine video. Explain that this lab feeds sampled frames, without its soundtrack.

**9–17 minutes: change one condition.** Compare red and blue mugs, then dog photograph and sketch. Read scores across each row. For the hat experiment, derive `(after similarity − before similarity) / length of image difference`. Reverse it. The sign reversal is guaranteed arithmetic; the resulting winning caption is not guaranteed to be a correct description of the edit.

**17–22 minutes: test the interpretation.** Remove the cat caption: the dog caption wins despite unchanged embeddings and unchanged remaining scores. Compare languages and vector sizes. The bark's top caption improves in this one 128d example; this is not evidence that fewer dimensions are generally better.

**22–25+ minutes: applications.** Open the PCA explorer and compare real coordinates. Turn on top-three-per-modality to see neighbours without one modality filling every place. Train the small audio classifier. The model stays frozen; only the added weights change. The held-out set is deliberately tiny, so do not present its accuracy as a benchmark.

## Controls

- Chapter dropdown: jump to any example.
- Show vectors / compare / explain / next: reveal one idea at a time.
- Right or Space: advance when focus is outside a form, button or media control.
- Left: go back. Reset chapter: clear the prediction and interventions.
- Recording view / Esc: hide or restore the surrounding page.
- Try your own inputs: open the related live application.

The screenshots and cosine values depend on the pinned q4 model and this exact sample collection. New queries, task prefixes, shorter vectors and different devices may produce different rankings. Treat the failures as part of the lesson.
