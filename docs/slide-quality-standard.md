# ES 667 slide quality standard

This is the acceptance gate for every canonical Typst lecture deck. It captures
the teaching style established in L1–L3 and turns it into a repeatable review
process for L5 onward.

The source of truth for lecture identity and order is `slides.qmd`; older plans,
archived Marp decks, and legacy teaching-guide numbering are context rather than
authority when they disagree with the public sequence.

## Pedagogical rubric

Score every item 0–2. A deck is ready to commit at **18/20 or higher**, with no
zero on items 2, 4, 5, 8, or 9.

1. **Bridge and lecture delta** — begin from something students already know;
   state what stays fixed and what new problem the lecture solves.
2. **One cumulative learning arc** — intuitive problem → visible failure →
   mathematical fix → derivation or computation → implementation. Avoid a
   catalogue of disconnected methods.
3. **Prerequisite-aware language** — define new vocabulary, notation, shapes,
   and clocks before using them.
4. **A persistent worked numeric** — carry one small example through several
   slides, diagrams, equations, and code. Every major idea must touch numbers.
5. **Progressive construction** — preserve the visual anchor and reveal one
   operation, term, or case at a time. The compact PDF is a handout; the
   presentation PDF keeps the builds.
6. **Commit before explanation** — place questions immediately before answers;
   explain why the answer follows rather than showing only the correct option.
7. **Teaching titles and conclusions** — titles should make a claim or ask a
   useful question. Closing statements say what the evidence means.
8. **Stable semantic grammar** — a color, shape, and symbol retain the same
   meaning throughout a sequence. Equations and diagrams use the same variables.
9. **Readable, data-faithful visuals** — use meaningful axes and shared scales;
   directly label trajectories; show the computed data; avoid decorative or
   misleading geometry; keep every mark inside the canvas.
10. **Synthesis and handoff** — revisit the opening commitment, close with the
    one-sentence course insight, and motivate the next lecture.

## Teaching-time contract

- Mark a route that fits an 80-minute class: core, should-cover, and optional.
- Keep one reproducible derivation per lecture; other mathematics may establish
  plausibility but must be marked optional when it exceeds the course contract.
- Include one recurring `symptom → suspect → test` diagnostic in spine decks.
- Treat L1–L14 as mastery and L15–L24 as literacy with one worked numeric; depth
  should serve the student contract rather than maximize topic count.

## Visual-evidence contract

- Prefer a real, properly licensed and attributed image when the lesson depends
  on what a real object, scene, failure, or dataset looks like. Computer-vision
  lectures should not teach image tasks entirely through abstract boxes.
- Use a generated teaching image when a controlled scene, stable identity, or
  licensing-safe counterfactual is more useful than a found photograph. Label it
  as generated, preserve its prompt/provenance beside the asset, and reuse the
  same scene through the numerical and conceptual spine.
- Every image must do teaching work: support a prediction, expose a failure,
  anchor a mask/box/feature calculation, or provide evidence for a claim. Avoid
  decorative stock imagery and visual variety that breaks the persistent case.
- Keep equations, plots, masks, arrows, and explanatory overlays code-native or
  vector whenever practical. Raster images are acceptable for photographs and
  generated scenes; vector-suitable charts and diagrams remain raster-free.
- For deliberate rasters, inspect effective resolution at the actual slide size,
  record the expected raster-instance count in QA, and reject accidental helper
  conversions or low-resolution assets.

## Deliverables for one lecture commit

- Canonical Typst source and only the figures it actually uses.
- Executed, output-retaining follow-along notebooks linked from the deck.
- `slides-pdf/L*.pdf`: compact handout (`--input handout=true`).
- `slides-pdf/L*-presentation.pdf`: progressive presentation build. Add this
  alongside the canonical handout as each lecture passes the quality gate.
- Teaching guide synchronized with the final deck order, timings, subtle points,
  likely misconceptions, and the short-on-time route.

## Verification gate

Before committing a lecture:

1. Compile both handout and presentation modes from a clean command.
2. Run `scripts/audit_typst_slides.py` on both PDFs.
   Use a zero-raster limit unless the deck deliberately contains photographs or
   generated scenes; in that case set the limit to the exact expected instance
   count and verify those instances manually.
3. Render every handout page and visually inspect the complete contact sheet.
4. Inspect high-resolution pages for every audit candidate and every dense slide.
5. Execute notebooks from a fresh kernel; reject errors, stale outputs, and claims
   not supported by the displayed results.
6. Check source/PDF/notebook links and lecture-to-lecture transitions.
7. Obtain an independent editorial/content review and an independent visual
   review after the final render.
8. Stage only that lecture's files, inspect the staged diff, and make one
   lecture-scoped commit.

Generated review surfaces and scratch renders do not belong in lecture commits.
