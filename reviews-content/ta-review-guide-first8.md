# TA Review Guide — First 8 Lectures (ES 667 Deep Learning)

**Goal:** each lecture ships three coupled artifacts — a **slide deck**, its **interactive explainers**, and its **notebook(s)**. We want a second pair of eyes confirming they are *correct, consistent, and complete*, and flagging anything to fix, cut, or add.

**Assignment:** 4 TAs × 2 lectures each. Budget ~2–3 hours per lecture (read the deck slowly, click every interactive, run every notebook). Quality of findings matters more than volume.

| TA | Lectures | Deck file | Notebooks | PDF |
|---|---|---|---|---|
| TA 1 | **L1** Why These Losses · **L2** Linear→MLP | `lecture1/`, `lecture2/` | `notebooks/L01/`, `notebooks/L02/` | `slides-pdf/L1.pdf`, `L2.pdf` |
| TA 2 | **L3** Calculus Toolkit · **L4** Backpropagation | `lecture3a/`, `lecture3/` | `notebooks/L03A/`, `notebooks/L03/` | `slides-pdf/L3A.pdf`, `L3.pdf` |
| TA 3 | **L5** Optimization · **L6** Trainability | `lecture5/`, `lecture6/` | `notebooks/L05/`, `notebooks/L06/` | `slides-pdf/L5.pdf`, `L6.pdf` |
| TA 4 | **L7** Regularization · **L8** CNNs | `lecture7/`, `lecture8/` | `notebooks/L07/`, `notebooks/L08/` | `slides-pdf/L7.pdf`, `L8.pdf` |

## How to open each artifact

- **Slides** — open the PDF in `slides-pdf/`. This is the one-slide-per-page **handout** (all builds flattened). If you want to see the *animated builds* as students will (one reveal at a time), compile the present version: `typst compile --root . lectureN/LN-*.typ /tmp/present.pdf` from the repo root.
- **Interactives** — the `[I]`-tagged slides link to a browser widget at `https://nipunbatra.github.io/interactive-articles/<slug>/`. Click the link, or browse the full gallery at that site. *(Status note below — some are being wired this week.)*
- **Notebooks** — open the `.ipynb` in Jupyter/Colab and do **Restart & Run All**. They also render on the course site under **Notebooks**.

> **Status note (so you don't over-report):** the decks are Typst (not Markdown). Every deck's build-pacing and worked examples were recently expanded, so **no slide should feel crowded** — if one does, that's a real finding. A few `[I]` interactive slides may still read "(to build)" this week while the widgets are finalized — **don't flag "(to build)" as a bug**; instead check whether the *described* interactive would actually help, and whether the ones already linked work.

---

## The checklist — run every lecture through these six lenses

For each item, note **lecture + slide/cell number + severity**. Severity: **[BLOCKER]** wrong/broken, **[FIX]** should fix before teaching, **[NICE]** improvement.

### A. Correctness (most important)
- [ ] Is every **claim, definition, and derivation** correct? Flag anything wrong or misleading.
- [ ] **Recompute every worked example by hand or in a scratch cell.** Do the numbers on the slide actually check out? (e.g. a gradient, a softmax vector, a parameter count, an output-size formula.)
- [ ] Are units, signs, and indices right? (a stray minus sign, an off-by-one, a transposed matrix.)
- [ ] Is any statement true-but-imprecise that a sharp student would catch?

### B. Figures & diagrams
- [ ] Does **every figure render** — nothing missing, blank, or a broken-image box?
- [ ] Is all text **inside** its node/box? Any label **overflowing, clipped, or overlapping** another? Any arrow pointing at the wrong thing?
- [ ] Are axes, ticks, legends, and units present and **readable at projector size**?
- [ ] Does the figure **actually depict what the text claims**? (right shape, right regime, right example.)
- [ ] Is it consistent with the deck's palette/style, or does it look pasted-in?

### C. Layout & overflow
- [ ] Any slide where text or a figure **runs into the footer / off the edge / under the slide number**?
- [ ] In the handout, does any single slide **spill onto two pages**?
- [ ] Any font that's **too small to read**, or a table that's too wide?
- [ ] Is the **build pacing** good — does each slide reveal one idea at a time, or does something dump everything at once / drag on too long?

### D. Sync across slide ↔ figure ↔ notebook
- [ ] Is the **notation consistent**? Same symbol for the same thing across the slide text, its figure, and the notebook (e.g. `w` vs `θ`, `η` vs `lr`).
- [ ] Do the **numbers agree** between the slide's worked example and the notebook that implements it?
- [ ] Is any **term used before it's defined**, or defined and never used?
- [ ] Does the interactive use the **same setup** as the slide it's attached to?

### E. Interactives (`[I]` slides)
- [ ] Does the linked widget **open and load** — not blank, not a 404?
- [ ] Do **all controls work** (sliders, buttons, toggles) and visibly change the picture?
- [ ] Does it **make the slide's point** — would a student understand the concept better after 60 seconds with it?
- [ ] Open the browser console (F12 → Console) — **any red errors**? Any math showing as raw `$...$`?
- [ ] Is the link pointing at the **right** widget for that slide?

### F. Notebooks
- [ ] **Restart & Run All** — does it run top-to-bottom with **zero errors** and produce its plots?
- [ ] Do its **outputs match the slide** it pairs with (same numbers, same conclusion)?
- [ ] Is it **minimal and readable** — one clear idea, sensible variable names, a short intro and takeaway? Anything confusing or redundant?
- [ ] Any **missing seed / dependency / hard-coded path** that would trip a student?
- [ ] Is there an obvious **missing notebook** — a worked idea in the lecture that really wants a runnable version?

### G. The two big-picture questions (answer per lecture)
- [ ] **Cut:** anything redundant, off-topic, or dated that should be **deleted**?
- [ ] **Add:** any place a student would get lost that needs **one more example, one more intermediate step, an intuition line, or a figure**?
- [ ] **Enough examples?** Does each core concept have at least one concrete/numeric example? Which concept is thinnest?
- [ ] **Top 3 improvements** for this lecture, ranked.

---

## How to report

One row per issue in a shared sheet (or just this format in a doc). Keep it terse and specific — a slide/cell number and a one-line fix is worth more than a paragraph.

```
Lecture | Slide/Cell | Lens (A–G) | Severity | What's wrong | Suggested fix
--------|-----------|-----------|----------|--------------|--------------
L2      | slide 31  | B         | FIX      | "class 3" label overlaps the boundary line | nudge label left / shrink region
L2      | nb cell 4 | F         | BLOCKER  | KeyError: 'lr' — variable renamed upstream | rename to `eta` to match slide
L2      | slide 18  | G-add     | NICE     | UAT stated but not shown numerically | add a 2-bump ReLU sum example
```

Plus, per lecture, a 3-line summary: **strongest part**, **weakest part**, **top-3 fixes**.

---

*This guide covers the first 8 lectures. The same rubric extends to the rest of the course once these are dialed in.*
