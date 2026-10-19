# Verification record: Generative Modelling II

Verified locally on 19 October 2026.

## Delivered

60 core logical scenes / 158 core builds. 11 appendix scenes / 33 builds. The completed-state PDF is 71 pages; the all-builds PDF is 191 pages; the selectable-text recap is two pages. Full route 120 minutes; compressed overview 90 minutes. The older lecture and source files were preserved.

## Checks actually run

- 23 numerical and PyTorch semantic checks passed: scalar and attention fixtures, one-based schedule, endpoint, broadcast dimensions, real forward/backward gradients, posterior/epsilon mean equivalence, final-step no-noise, deterministic finite oracle samples.
- 10 notebook code cells executed without errors; retained outputs are in the executed notebook.
- All 191 presentation states captured at 1280 by 720. No detected overflow, JavaScript errors, missing images, duplicate IDs or broken mathematics.
- All six demos passed deterministic reset, play, pause and pause-on-leave checks. Keyboard input focus, reading mode, 390-pixel mobile width, overview, stable deep links, compressed build selection and reduced motion passed.
- Final-state contact sheets reviewed across all 71 scenes; dense math, U-Net, attention, numeric reverse, code and application crops inspected at full size. Rendered PDF pages and the recap were inspected.
- External citation links and resolved internal course-map destinations verified in both lecture PDFs, beyond annotation counts.
- Old and new checkpoint hashes match their recorded provenance. Old lecture's 83 physical pages load images and produce no JavaScript errors in the preserved HTML.

## Whole-deck language and visual revision

All 71 scenes have been reviewed against Beyond Attention, including the earlier 26 revised scenes, all applications, six demonstrations, both code scenes and the full appendix. There are 65 portable SVG figures/compositions. The speaking cue and final takeaway remain visible. Repeated explanatory paragraphs moved into Notes; every original build explanation remains accessible. Several takeaways now interpret the example instead of repeating numerical labels.

The final revision was captured across all 191 states. SVG text was checked in all 173 vector-figure states, with zero detected label collisions or out-of-bounds labels. Browser geometry, image decoding, IDs, mathematics, keyboard focus, deep links and mobile width pass. All six demos pass reset/play/pause/leave checks. Added checks cover guidance mode, a second attention query, recursive corruption, masked-loss alignment, completed scripted generation and retained derivations. No complete takeaway sentence is repeated in the visible body. All 23 numerical/PyTorch checks were rerun and passed; trained weights and the executed notebook were preserved.

`VISUAL_REVIEW.md` and `qa/scene-review.json` contain one review entry for every scene. Both lecture PDFs were regenerated from the final captures and independently rendered; page counts and actual link destinations were checked.

## New computation

The new 814,145-parameter denoiser trained for 2,000 MPS steps, batch 64, seed 667. T=1000 linear schedule, posterior variance, no sampler estimate clipping. Training took 98.78 seconds in the recorded run. Eight unselected seed-667 samples, aligned chain states and clean estimates, and 60 recorded training tuples were exported. The older class-conditioned guidance gallery was retained, with its distinct cosine schedule and checkpoint hash; this build did not retrain it.

## Evidence limits

The new samples are often malformed and similar. This is a small teaching run, not a validated image-quality result. The analytical mixture sampler uses an exact score with approximate finite Gaussian transitions. Tiny attention activations and language traces are illustrative. Published natural-image results retain attribution and reuse notes. No satellite, text-to-image, DiT or language model was trained. No FID or downstream gain was invented. Pacing needs classroom rehearsal. PDFs reproduce stage images as raster with explicit links; HTML and instructor notes retain accessible selectable text. Nothing was deployed.

## Reviewable records

See `qa/math.json`, `qa/browser.json`, `qa/interactions.json`, `qa/notebook.json`, `qa/pdf.json`, `qa/provenance.json`, `qa/whole-deck-checks.json`, `qa/figure-layout.json`, and `qa/release-summary.json`. `qa/builds/` contains all captured reveals; `qa/pdf-renders/` contains PDF contact sheets and full-size inspection pages.
