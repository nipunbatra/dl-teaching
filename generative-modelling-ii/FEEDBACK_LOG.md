# Feedback log: how Nipun wants these lectures to teach

A running record of feedback given on Generative Modelling I (session “Generative Modelling I lecture review”, its right-click notes in `../../generative/review/notes.json`) and on this lecture, and what each piece changed here. Newest entries at the bottom of each section. Use it before editing either deck.

## The quality bar

- **Reference:** the attention series (Beyond Attention / From Attention to Applications, the CLIP lecture). Not the older Beyond Generation PDF.
- **Style models:** Lilian Weng's blog figures and Jay Alammar's illustrated explanations. Neat SVG mechanism diagrams (encoder, decoder, arrows that mean something), not boxes of text.
- **“Slow, intuitive, example-driven, step-by-step maths.”** One move per click; a result is derived on the slide, never asserted.

## What the feedback says, grouped

### Story and flow
- Start every part by saying **what we are trying to do and how**; end it by saying **what we did and how it leads to the next part**. Draw **section breaks** with a route map so students always know where they are.
- Introduce a concept **when it is needed**, not before. Forward references confuse (“why are we talking about the VAE? we were just talking about sampling”; “what is this we cannot compute?”).
- If an important slide makes several moves at once, **split it into several slower slides** (“this is such an important slide… break it down into slower and slower parts”).
- Explain why a problem is hard before the method (tractability → the inference options → the one we use).
- Motivation at the start must be real: a question students want answered, and real examples, then come back to them.

### Examples and evidence
- **One running example** carried through every equation, with **numbers after every equation**.
- Abstract words need **real examples on the slide** (“what could this conditional be? give real examples”).
- When a slide poses a question (a vote, a puzzle), **give the answer** on a later click.
- Trained models must actually support the claims; caveats live in the notes, not on the canvas.
- Prefer **interactive apps on real trained models** where they make an idea tangible (sliders, “random code” button, hover to see the digit), with a fixed state for the PDF.

### Maths
- Every symbol through KaTeX, **display mode** for standalone equations, one shared set of macros; diagram labels and chart labels in the same fonts. No hyphen-minus, no “∼” that renders as a dash, no maths in an uppercased kicker.
- Define notation before using it (𝔼, 𝒩(x; μ, σ²)).
- Long derivations: one line per click, the part that changed highlighted, the previous line still visible.
- Appendix derivations that the main argument relies on belong in the main flow.

### Code
- Show **minimal PyTorch code with annotations beside it** whenever an idea becomes an algorithm (fitting, sampling, training).

### Layout
- Numbers and text must align (route list); nothing clipped, crossed or overflowing; 2–3 decimals.
- Speaker cues off the projected slide, one per click, in the notes.

## Applied to Generative Modelling II

| Feedback | What changed here |
|---|---|
| attention-series bar; Lil'Log / Alammar figures | Rebuilt on the Lecture I runtime: no permanent chrome, one caption line, hand-built SVG mechanism diagrams with KaTeX labels, matplotlib charts in Computer Modern |
| one running example, numbers after every equation | Pixel (12, 19) of MNIST test digit 0 (a 7): x₀ = 1, ε = −0.5, x₂ = 0.5, ε̂ = −0.4 carried through jump formula, loss, posterior (0.90/0.10, E[ε] = −0.24), x̂₀ = 0.925, μθ = 0.708, μ̃ = 0.745, KL = 0.625 × 0.01, x₁ = 0.808 |
| models must support claims | Retrained: pixel DDPM 30k steps, class-conditional DDPM, 4×7×7 autoencoder + latent DDPM, MNIST classifier to measure guidance; every image on a “our model” slide comes from these |
| caveats in notes; provenance clear | Four corner badges (published / our model / computed / illustrative), detail in notes |
| real motivation, return to it | Opens on our own noise→digit trajectory and the question; 8 applications predicted then revealed; closes on the same applications |
| conditions need real examples | Applications slide shows 8 real conditions (sentence, mask, painting, low-res, edges, boxes, subject photos, satellite) |
| answer the questions posed | Every “?” slide has a reveal click; exit questions have answers |
| derive, don't assert | KL → noise MSE derived by substitution with numbers; posterior reverse mean computed in the main flow |
| code with annotations | Forward process, training step, sampler, and the sampler with guidance/inpainting edits, each with click-highlighted lines and side callouts |
| interactive apps | Noise-level explorer, reverse-step sliders; trajectory scrubber with several seeds |
| maths only in KaTeX | Export fails if a maths character appears outside KaTeX (same check as Lecture I) |
| right-click review | `deck/annotate.js` + `scripts/review_server.py` copied from Lecture I (port 8772) |
| concepts when needed; section breaks + recaps (Lecture I notes, 19 Oct) | ELBO justification moved after sampling, where both reverse means are known; each part opens with a route map and closes with a recap |
