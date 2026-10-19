# Generative Modelling II · From noise to images

ES 667 Deep Learning, IIT Gandhinagar. Denoising diffusion taught slowly around **one running example**: pixel (12, 19) of MNIST test digit 0, a 7. Every equation in Parts 1–4 is followed by that pixel's numbers, and every “our model” figure comes from models trained by the scripts below.

The deck uses the same runtime, colours and figure style as Generative Modelling I (`/Users/nipun/git/generative`).

## Open it

```sh
python3 scripts/review_server.py      # http://127.0.0.1:8772/  (also collects right-click review notes)
```

or open `index.html` directly.

- **→ / Space** next click · **←** back · **Shift+→** next slide · **O** overview · **S** speaker notes for the current click · **F** fullscreen
- `index.html?reading`: every slide in its final state with its notes underneath
- PDFs: [exports/generative-modelling-ii.pdf](exports/generative-modelling-ii.pdf) (96 slides, final state) and [exports/generative-modelling-ii-builds.pdf](exports/generative-modelling-ii-builds.pdf) (271 pages, every click)
- [INSTRUCTOR_NOTES.md](INSTRUCTOR_NOTES.md): what to say on each click, generated from the slides

### Review notes (right-click)

With `scripts/review_server.py` running: right-click anything on a slide, type what should change, press **Send**. The note records the slide, the click and what you clicked (an equation's LaTeX), and lands in `review/notes.json` and `review/inbox.jsonl`. **A** lists all notes. Shift+right-click gives the normal browser menu. Without the server, notes stay in the browser (A → Copy all).

## The lecture

| Part | Question | Running numbers |
|---|---|---|
| Opening | “This digit was pure noise 1,000 steps ago. How?” Eight published applications, predicted then revealed; Lecture I's VAE and GAN samples | |
| 1 · Corrupt | What noisy input do we train on? | $x_2 = 0.8\cdot1 + 0.6\cdot(-0.5) = 0.5$ |
| 2 · Train | What does the network predict? | best possible guess $\mathbb E[\epsilon\mid x_2] = -0.24$ |
| 3 · Sample | How do predictions become an image? | $\hat x_0 = 0.925$, $\tilde\mu = 0.745$, $\mu_\theta = 0.708$, $x_1 = 0.808$ |
| 4 · ELBO | Why is noise MSE the right loss? | $D_{KL} = 0.625 \times 0.1^2 = 0.00625$ |
| 5 · Network | U-Net, time embedding, latent diffusion, DiT | |
| 6 · Condition | prompts, cross-attention, guidance, inpainting, SDEdit, super-resolution | |
| 7 · Text | masked diffusion for tokens (LLaDA) | |
| Wrap-up and appendix A1–A11 | posterior, bound, v-parameterisation, schedules, DDIM, score, flow matching, LLaDA loss, heads, provenance | |

Each part opens with a route map and ends with a recap. Badges in each slide's corner: **published** (a paper's figure), **our model** (trained here), **computed** (exact arithmetic), **illustrative** (hand-chosen numbers).

## What is real

All trained locally on MNIST (`scripts/train_models.py`), Apple-silicon GPU:

- **Pixel DDPM:** the slides' U-Net, 814,145 parameters, linear β ∈ [10⁻⁴, 0.02], T = 1000, ε target, EMA 0.999, 30,000 steps, batch 128, seed 667. Sampler: DDPM with posterior variance.
- **Class-conditional DDPM:** the same network plus a label embedding, label hidden with probability 0.1 (index 10 = none). Used for classifier-free guidance; an MNIST classifier (98.5% test accuracy) measures how many guided samples look like a 7.
- **Latent diffusion:** a conv autoencoder 1×28×28 → 4×7×7, then a latent DDPM on the scaled latents.
- Inpainting (paste-back), SDEdit, DDIM and the no-noise ablation all run on the pixel model with no retraining.
- Computed, not learned: the toy β = 0.2 arithmetic, the schedule plots, the eight-cluster score field (exact score).
- Published figures (cropped by `scripts/crop_published.py` from the project pages listed in `asset-manifest.json`): DDPM, LDM, RePaint, SDEdit, SR3, ControlNet, GLIGEN, DreamBooth, DiffusionSat.

## Rebuild

```sh
PY=/Users/nipun/git/clip-lab/.venv/bin/python     # any Python with torch + matplotlib
$PY scripts/train_models.py uncond 30000; $PY scripts/train_models.py cond 30000
$PY scripts/train_models.py ae; $PY scripts/train_models.py latent 20000; $PY scripts/train_models.py clf
$PY scripts/crop_published.py                     # published crops
$PY scripts/make_figures.py                       # figures/data/* and numbers.json
(cd scripts && $PY diagrams.py)                   # figures/diagrams/*.svg
python3 scripts/build_deck.py                     # index.html
python3 scripts/make_notes.py                     # INSTRUCTOR_NOTES.md
node scripts/export_pdf.mjs                       # both PDFs; fails if any maths is outside KaTeX
```

| Edit | To change |
|---|---|
| `deck/slides/*.html` | slides, clicks (`data-b`), highlights (`data-hl`), speaker cues (`<aside class="notes"><p data-say="n">`) |
| `scripts/diagrams.py` (helpers in `scripts/svglib.py`) | SVG mechanism diagrams |
| `scripts/make_figures.py` | every chart, model output and number in `numbers.json` |
| `deck/deck.css`, `deck/deck.js`, `deck/widgets.js`, `deck/annotate.js` | look, navigation, live widgets, review notes |

## Reviews and history

- [CLAUDE_VISUAL_REVIEW.md](CLAUDE_VISUAL_REVIEW.md): the independent review of the Codex build, and how the rebuild addresses each point.
- [FEEDBACK_LOG.md](FEEDBACK_LOG.md): feedback from both lecture reviews and what it changed.
- The earlier Codex build is preserved: `codex-v1.html` (its deck), `src/`, `qa/`, `SPECIFICATION.md`, `STORYBOARD.md`, `VISUAL_REVIEW.md`, `QA_REPORT.md`, `requirements-to-scenes.json`, its exports (`exports/generative-modelling-ii-reading.pdf`, `-all-builds.pdf`, `diffusion-two-page-recap.pdf`), `models/model.pt` and `notebooks/`. The older diffusion lecture is at `../slides/generative2.html`.
