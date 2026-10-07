# EmbeddingGemma lab: external review and changes

Review requested by Nipun Batra on 7 October 2026.

**Follow-up:** Nipun preferred the earlier capability-first playground. The guided chapters and recording mode described below were subsequently removed. The numerical and provenance fixes remain. This report records the original review.

- Provider: OpenRouter, using the existing configured account.
- Returned model: `anthropic/claude-opus-5.5`.
- Requested reasoning: `high`.
- Evidence supplied: current published-page screenshot, README, retrieval UI, explorer, training code and styles.
- Method: source and screenshot review; Claude did not execute the app. Findings were checked against source and browser behavior before changes.
- The response reached its output limit after the substantive findings and recommendations. The unfinished final heading is omitted below.

## Changes incorporated

- Added a ten-chapter classroom route, prediction selection, four progressive reveals, side-by-side before/after scores and recording mode.
- Added a signed-difference derivation, reverse-direction control, missing-answer intervention, dimension comparison and visible model failures.
- Kept all fifteen existing experiments and the 154-item actual embedding index. More of the image collection is visible on the landing page and throughout the lesson.
- Separated newly encoded media from the saved candidate index. Fresh query metadata cannot overwrite saved processor metadata.
- Normalized each truncated image before subtracting at reduced dimensions.
- Applied equal PCA scale on both axes; the clustering view now reports explained variance.
- Replaced percentage-like cosine bars with a numeric gap from the highest score.
- Added top-three neighbours per modality without altering any scores.
- Required a non-empty note for joint inputs; exposed that note in the inspector.
- Filtered training categories to audio and reported actual split counts.
- Kept selected samples visible in the sample strip; made the classification prefix visible.
- Added a model warm-up control for recording with live inputs.

## Verification

All stored embeddings and media checks pass, together with PCA checks, finite-difference gradient checks and six small-classifier runs. The guided lesson has numerical tests for all nine calculation chapters, unchanged scores after candidate removal, signed-difference decomposition and reversal, dimension comparisons and index immutability. Browser verification covers all chapter reveals, prediction memory, intervention controls, mobile layout and the 1080p recording composition.

## Scope choices

The guided route uses already-computed media/text vectors and explicitly labels their provenance. Multilingual saved captions are distinguished from task-prefixed live queries. No claim of causal isolation is made for generated image pairs. We did not add an averaged-versus-joint comparison or new pre-encoded query corpus in this pass; joint input remains available as a real live experiment. We did not assert that a single sample demonstrates general performance or that 128 dimensions are always worse.

## Claude's review

# EmbeddingGemma 2 lab review: correctness, guided story and recording

*(Note: I can't choose or verify a model tier from inside a conversation. This review works only from the supplied source and screenshot.)*

## Verdict

The lab is honest and well built. It uses raw cosines, labels provenance, warns about the modality gap and avoids causal claims. Its main weakness is that **every comparison is sequential**. To compare two conditions, students must run one, remember the numbers, change something and run again. On video, that reads as clicking around rather than an experiment. The highest-impact change is a **side-by-side comparison panel** driven by declarative configs over existing samples, plus a few correctness fixes.

---

## 1. Bugs and accuracy concerns (grounded in code)

**P0: Delta mode corrupts stored metadata after re-encoding.**
- When `embedItem` computes fresh, it stores `result` in `state.vectors[item.id]` and returns the same object.
- `run()` then assigns `query.info = { formula: "unit(after − before)", … }`.
- With "Re-encode" checked, the portrait's stored `info` becomes the delta formula. Later inspections show the wrong processor input.
- Fix: return `{ ...result }` from `embedItem`, or build `state.query` without mutating `query`.

**P0: Re-encoding silently replaces the saved index.**
- `force` overwrites `state.vectors[id]` for media.
- Later rankings mix saved and re-encoded candidates. `result-source` only reports the query's origin.
- Fix: keep re-encoded vectors in `state.local[id]`.
- Turn the overwrite into a teaching moment: show `cos(saved, recomputed)` and whether the top-k changed. This is a real, honest q4/WebGPU reproducibility check.

**P1: PCA plots distort geometry.**
- `showGroups` normalizes x and y independently by min/max.
- Explorer scales by `maxX` → 290 px and `maxY` → 170 px.
- Both stretch PC2 relative to PC1, so vertical distances look larger than they are.
- Fix: use one scale, `s = min(W/rangeX, H/rangeY)`.
- Also show explained variance in the clusters view, as Explorer already does.

**P1: Delta at d < 768 differs from the stated formula.**
- The difference is taken at 768d, then truncated and renormalized inside ranking.
- The result is `unit(trunc(a) − trunc(b))`, not `unit(unit(trunc a) − unit(trunc b))`.
- The UI claims "Both sides use the same d." Either compute the delta after truncation, or state the difference.

**P1: Score bars look like percentages.**
- `width = (score+1)/2` maps typical cosines (around 0.2–0.5) to 60–75% bars. That suggests "75% match" and hides the actual gaps.
- Replace the bar with **"Δ from #1"**, a real number. Alternatively, use a fixed axis with a visible zero tick.
- Also, `toFixed(4)` implies precision that q4 inference may not reproduce. Three decimals is enough on screen.

**P1: Training class list can include non-audio labels.**
- `classes` for 10-class mode comes from `state.gallery.filter(x => x.label)`.
- If any non-audio item carries `label`, empty classes enter the softmax. Both ln(C) and the parameter count would then be wrong.
- Filter on `x.type === "audio"`.
- `split-note` also hard-codes "3 + 1 per class". Compute it from `train` and `test`.

**P2: Classification prompts differ from caption prompts.**
- Labels are encoded as `role: "query"` with the classification prefix. Gallery captions use the document prefix.
- So "a dog" as a label and "a dog" as a caption are different vectors.
- Surface the prefix in the classify results header. Students will otherwise compare scores across the two tabs.

**P2: Mixed mode with an empty note silently becomes image-only.**
- `!mixed` returns the saved image vector, still labelled as a normal run.
- The inspector also shows the image title, not the note, as "Your query".

**P2: Smaller issues.**
- Some unescaped IDs: `data-inspect` in `showGroups`, `data-neighbour`, `data-query` in the library, and `item.type` in cluster items. Gallery data is trusted, but escape consistently.
- The sample strip always shows the first six items. `retriever-sketch` may never appear "pressed".
- `inspect()` in delta mode shows only the *before* tensor shapes.

---

## 2. Bounded implementation for this session

All of this stays in vanilla JS: one new module, small edits elsewhere.

1. **Fix the P0/P1 items above.** This is about 40 lines in total.

2. **`src/comparisons.js` plus a `compare` mode.** Each config declares:
   - two conditions (A and B), a fixed candidate pool and a prediction prompt;
   - which real quantity to show;
   - a caveat.

   Rendering shows two ranked columns with rank-change arrows (↑3, new, dropped), top-k overlap and Δ-from-#1:
   ```js
   const rankMap = r => new Map(r.map((x,i)=>[x.item.id,i]));
   const overlap = (a,b,k=5)=>a.slice(0,k).filter(x=>b.slice(0,k).some(y=>y.item.id===x.item.id)).length;
   ```

3. **Stored comparison queries.** Add the comparison texts (English/Hindi queries, label sets, notes) to `index-builder.html` under `embeddings.queries`, with provenance. They are actually computed and pinned, so recording doesn't depend on a 473 MB download. Live re-encoding remains available.

4. **Delta decomposition** (see C1). This is the single best explanation in the lab.

5. **Record the prediction.** Preview cards get "I predict this". After the run, mark the prediction and the actual rank.

6. **`?present` mode** (Section 5).

---

## 3. Guided story (about 25 minutes)

| Min | Beat | What's on screen |
|---|---|---|
| 0–2 | **Hook** | Bark plays. "Which picture is closest?" Students predict, then stored cosines are revealed. No model is needed. |
| 2–6 | **Back to CLIP** | Chelsea → captions. Open *Explain score*: unit norms, eight products, full sum. Then show a **4×4 image × caption cosine matrix** from stored vectors: the CLIP-loss logits students just studied. Mark the row/column argmax. If you add a τ slider, label it "your τ, not the model's training temperature". |
| 6–9 | **Beyond two towers** | Text → sounds, then text → video moments. Say plainly: one shared model, mean pooling, no transcript, video frames without the soundtrack. |
| 9–19 | **Six controlled comparisons** | Below. Each follows: question → prediction → A/B → what changed → caveat. |
| 19–22 | **Limitations** | Search within "Everything" with the bark. Audio neighbours dominate the list. Then use the per-modality view (Section 4). |
| 22–25 | **Learn** | Frozen classifier, 3 classes, step through. Then 10 classes: training loss falls while held-out loss rises. "We changed W, not the space." Link to the contrastive fine-tuning notebooks. |

---

## 4. Controlled comparisons using existing pairs

**C1: Portrait → portrait with hat (delta).**
- Rank captions by `unit(after − before)`. For each caption, also show `cos(c, after)`, `cos(c, before)` and their difference.
- For unit captions, `c·(a−b)/‖a−b‖ = (cos(c,a) − cos(c,b))/‖a−b‖`. The ranking is therefore exactly "which caption *gained* the most similarity". The denominator is shared by all captions.
- The sign reversal on swap then becomes arithmetic, not magic.
- Caveat: this is a direction in embedding space, not a cause. Background or lighting changes also contribute.

**C2: Red mug vs blue mug (single attribute).**
- A: red-mug → captions. B: blue-mug → captions. Do colour words move while "mug" captions stay?
- Then a joint-input intervention:
  - joint `(red-mug image + "blue")` vs `unit(v_image + v_text("blue"))`, both real vectors;
  - this shows that joint encoding (tokens attending to each other, as in self-attention) is not averaging.
- Caveat: one pair, one phrasing.

**C3: Dog photo vs sketch (style vs subject).**
- Show three numbers: `cos(photo, sketch)`, `cos(photo, other dog photos)` and `cos(photo, non-dog generated images)`.
- Then compare the photo → captions and sketch → captions lists.
- Question: does style or subject dominate? Note that generated images share a rendering style, which is a confound to name aloud.

**C4: Multilingual query (Hindi vs English).**
- Use stored "एक बिल्ली की तस्वीर" vs "a photo of a cat".
- Against **images**: top-5 overlap and rank shifts.
- Against **captions** (59 in five languages): does the Hindi query prefer the Hindi caption over the English equivalent?
- This shows a same-language effect honestly, whichever way it falls.

**C5: Candidate labels (newfoundland).**
- A: the default four labels. B: remove "a dog". C: add "a bear" or "a black dog".
- Show that one label always wins, and show the drop in the top cosine.
- Also show how wording changes the winner.
- Note that the argmax is not calibrated, and that the classification prefix differs from the caption prefix.

**C6: 768 vs 128 dimensions.**
- Use the same query (the bark → images, and Chelsea → captions) with two columns.
- Show rank changes, top-5 overlap and bytes per vector (3072 → 512).
- Present this as an observation on these samples only. Don't assert how well Matryoshka-style truncation transfers to audio or video.

**Per-modality nearest view** (supports Section 3's limitation beat).
- Add a toggle in Explorer and the "Everything" filter: "Top 3 per modality."
- It removes cross-modality score-range dominance without any score offsets. It is only grouping real rankings.

---

## 5. First screen and recording

**First screen.** The hero is attractive, but it is a menu, not a question. Replace the three tiles with one live, zero-download question:
- a playable bark;
- three image tiles (Newfoundland, retriever, cat);
- a prompt: "Which is closest? Tap to predict."

Reveal the real stored cosines with "Δ from #1". Underneath: "Same computation as CLIP: encode → normalize → dot product. Different inputs." Keep the three modality tiles as secondary entry points.

**Pedagogical framing.**
- Number the experiment nav as the story order: 1 Hook, 2 CLIP, 3 Beyond, 4 Compare, 5 Limits, 6 Learn.
- Collapse the nav to the current chapter.
- Fifteen equal buttons hides the arc.

**`?present` mode.**
- Base font around 18px and scores in a large monospace. Hide fineprint, the footer and the upload panel.
- Turn off smooth scroll and hover lift, since they jitter on capture.
- Add a "Next step" button and the → key, so each story beat sets the experiment, inputs and filter, then stops at **Predict**.
- Add a "Pre-warm" button: download the model and run one dummy embed before recording. The 473 MB load should never appear on camera; the cached load then takes seconds.
- Verify a 1920×1080 layout with three result columns, and that the inspector fits one screen.

**Recording practice.**
- Pause visibly at each prediction.
- Read one score gap aloud per experiment, not every number.
- After each comparison, say what it does *not* show: one pair, q4 quantization, related video frames, a tiny held-out set.

---
