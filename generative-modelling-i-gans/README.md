# Generative Modelling I-B · GANs (ES 667)

A standalone GAN lecture, split out of the Generative Modelling I (VAE) lecture so that the VAE
lecture can go deeper. It is built around one 1-D toy (last lecture's two bumps at ±2) and MNIST,
and every figure and number on the slides comes from models trained by the scripts here.

| File | What |
|---|---|
| `gans.html` | the deck (37 slides, 96 clicks) |
| `exports/gans.pdf` | one page per slide, final click |
| `exports/gans-builds.pdf` | every click |

## Open it

```sh
python3 scripts/review_server.py      # http://127.0.0.1:8773/gans.html
```

→ / Space next click · ← back · Shift+→ next slide · **O** overview · **S** speaker notes · **F** fullscreen · `?reading` all slides with notes.

**Review notes.** Right-click anything on a slide, type what should change, press **Send**. The note goes to
`review/notes.json` and is appended to `review/inbox.jsonl`, which a Claude session watches. Red pin = open,
green ✓ = done (hover for the reply). The page reloads itself, on the same slide and click, whenever the deck is rebuilt.
`review/` is not committed.

## Route

| Part | Slides |
|---|---|
| Opening | cover (our GAN's samples), StyleGAN faces, the 12-digit real-or-generated vote, the bridge from the VAE lecture, route |
| 1 · Two players, one game | the two networks, the toy at step 0, D's loss worked on real numbers, the D step and `detach()`, **D-only experiment** |
| 2 · Training G through D | G step on the toy, the chain rule with our fake's numbers, the G step code, freezing vs detaching, **G-only experiment**, the full game |
| 3 · The minimax objective | the formula, the saturation derivation, the saturation plot |
| 4 · A GAN on MNIST | **G and D tensor shapes**, the complete `gan_step`, samples through training, the vote revisited |
| 5 · Failures and control | mode collapse (and what D knows), loss curves that cannot tell, conditional GANs |
| Wrap-up and appendix | four-sentence summary, three exit questions, optimal D, the JS divergence, reading |

## What is real

- **Toy GANs** (`toy_gan_runs.json`, `toy_gan_collapse.json`): 8 seeds of a 64-unit G and D on the two-bump data,
  3,000 alternating steps; all 8 cover both bumps at the end. One run with a 4-unit G collapses onto one bump.
- **One-player runs** (`toy_one_player.json`, new for this deck): seed 4's G and D at initialisation; 200 D steps with G frozen,
  then 300 G steps with that D frozen. Shows what each half of `gan_step` does alone.
- **MNIST GAN** (`gan_mnist.pt`, `gan_mnist_snapshots.npz`): grey-level MNIST, small DCGAN-like G and D, 30 epochs,
  trained with the same `gan_step` shown on the slides.
- The D-loss table, the chain-rule numbers and the collapse numbers are read from the recorded D curves
  (`figures/data/numbers.json` → `dloss`, `collapse`, `oneplayer`).
- **Copied, not regenerated here:** `assets/vae-lecture/vae16-samples.png` (the VAE lecture's 16-number-code VAE, shown on the bridge slide).
- **Published outputs:** the StyleGAN faces (Karras et al. 2019, CC BY-NC 4.0). Source line on the slide.

## Rebuild

```sh
PY=/Users/nipun/git/clip-lab/.venv/bin/python   # any Python with torch + torchvision + matplotlib
$PY scripts/train_models.py        # optional: retrain (MNIST read from $MNIST_ROOT, nothing downloaded)
$PY scripts/make_figures.py        # figures/data/*.svg|png and figures/data/numbers.json
$PY scripts/diagrams.py            # figures/diagrams/*.svg (hand-built SVG)
$PY scripts/build_deck.py          # inlines diagrams, fills {{numbers}}, writes gans.html
node scripts/export_pdf.mjs        # exports/*.pdf, and fails if maths appears outside KaTeX
```

`make_figures.py` reproduces the source lecture's GAN figures exactly (identical PNGs, identical numbers).
Puppeteer comes from the repository's root `node_modules`.

| Edit this | To change |
|---|---|
| `deck/slides/*.html` | slide text, builds (`data-b="2"` appears from click 2, `data-gone="=1"` removed on click 1), speaker notes |
| `scripts/diagrams.py` | the SVG diagrams |
| `scripts/make_figures.py` | every chart and digit grid, and `numbers.json` |
| `scripts/train_models.py` | the GANs, `gan_step`, and the one-player experiment |
| `deck/deck.css`, `deck/deck.js`, `deck/annotate.js` | look, navigation, right-click review notes (shared with the VAE lecture) |

All maths is typeset by KaTeX (slides, diagram labels) or matplotlib's Computer Modern (charts).
Shared macros (`\N`, `\E`, `\KL`, `\JS`, `\pd`, `\pg`, role colours `\cx \cdec \cdisc \closs`, `\hl{}`) live in `scripts/build_deck.py`.
