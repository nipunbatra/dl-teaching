# lec01 Revision Tracker — instructor feedback 2026-06-26

Status: ⬜ todo · ✅ done

## Content / math / layout
- ✅ 1 · Opening "A question to open the semester" — show the models first (linear works/fails), then the raw-photo question.
- ✅ 5 · ImageNet slide — explain the task (1000 classes, 1.2M imgs) + what the data looks like.
- ✅ 6 · "Why now · the compute curve" — remove (accuracy uncertain, redundant with the numbers table).
- ✅ 9 · "Worked numeric · the collapse" — split across slides (forward, then equivalent single layer).
- ✅ 11 · "XOR · linear fails, MLP succeeds" — content overflow → trim/split.
- ✅ 12 · "Feature-space transformation" — content cut + figure unclear (no separating line) → trim text + new figure.
- ✅ 13 · "Activation functions · what can go wrong" — `$|z|$` breaks the table (pipes) → `$\lvert z\rvert$`.
- ✅ 14 · "Stacking neurons → MLP" — show sizes/dims of every tensor.
- ✅ 15 · loss-stuck 2.30 → explain = ln(10) = −ln(1/10); clarify CE functions (CrossEntropyLoss takes logits & fuses log_softmax+NLL; NLLLoss takes log-probs; BCEWithLogitsLoss vs BCELoss). Fix other broken inline math.

## Figures (anthropic cream palette — blends with slide bg, replaces white-bg originals)
- ✅ 2 · linear_vs_nonlinear_data.svg — labels too tight → clean 2-panel.
- ✅ 3 · pixel_shift_fail.svg — labels clip, cats identical → clear shift + flattened-vector contrast.
- ✅ 4 · dl_timeline.svg — three eras not visible → distinct colored era bands + takeaway.
- ✅ 7 · linear_to_neuron.svg (NEW) — steps: linear → +bias → +nonlinearity → neuron.
- ✅ 8 · magnifying_glass.svg (NEW) — analogy: two lenses = bigger-linear; prism = new features.
- ✅ 10 · stacked_linear_collapses.svg — label intermingle → clean "depth gives one line".
- (compute_scaling.svg becomes unused after #6)

## Verify
- ✅ make lec01-html, PDF spot-check changed slides, rsvg-convert figures.

## Separate: nilmbench design review (answer in chat)
- ✅ Compare nilmbench Marp design; recommend what (if anything) to adopt.

## Round 2 (2026-06-26)
- ⬜ R1 · dl_timeline too cramped → widen + add a discussion slide ("discuss more").
- ⬜ R2 · remove "Why now · concrete numbers" slide.
- ⬜ R3 · magnifying_glass figure unclear (LHS not a glass, RHS unclear) → redo.
- ⬜ R4 · stacked_linear_collapses → use NONLINEAR data, show linear still learns a straight line.
- ⬜ R5 · "Activation functions at a glance" → formulae for each (regenerate SVG grid).
- ⬜ R6 · "Stacking neurons → MLP" → figure must show node counts per layer.
- ⬜ R7 · remove "Batched matrix form · the shapes that matter".
- ⬜ R8 · "MLP in PyTorch" → show Sequential AND non-Sequential (explicit forward), comment on each.
- ⬜ R9 · Trim Part 3 redundancy with L00/L00c (softmax/CE already derived there) → condense.
- ⬜ R10 · Add pop-quiz slides between sections; make it shine as the first main DL lecture.
