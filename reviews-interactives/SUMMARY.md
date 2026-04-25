# Interactives audit · summary punch list

All 29 interactives reviewed by Gemini (`scripts/review_interactive.py`).
Full per-article reviews in this directory · this file is the priority shortlist.

## Already implemented this round

- **attention** · added `selfBank` scenario (word attending to itself).
- **bpe-merges** · 3 corpus presets · default · morphology · rare ZUGZUG.
- **gan-minimax-dance** · earlier full rebuild to p-values depth (8 steps + misconceptions + WGAN bonus).
- **lr-schedule-visualizer** · 3 scenarios · ResNet/CIFAR · BERT · LoRA · plus `constant` schedule button.
- **lstm-gates** · 3 input scenarios · pulse / sentiment-flip / noisy.
- **optimizer-race** · added `Sharp Trap` landscape · counters "Adam always wins."
- **seq2seq-bottleneck** · earlier full rebuild to p-values depth.
- **softmax-temperature** · scenario selector · generic / LLM / ambiguous / confident.

## Top remaining items by interactive

### autograd
- Add a fan-in/fan-out graph (`L = a*b + a*c`) to demonstrate gradient accumulation visually.
- Add a "saturated sigmoid" preset that fires the vanishing-gradient failure mode live.

### cfg-scale-visualizer
- Add 3 scenarios · abstract / "red cube on blue sphere" (compositional) / "majestic lion" (realism).
- Move the formula section AFTER an explicit 2D vector-arithmetic step.

### clip-zero-shot
- Add an "out-of-distribution" prompt scenario · show what happens when the prompt is gibberish or refers to a class CLIP has never seen.
- Surface cosine-similarity numbers (not just the bar) so users see the underlying numeric ranking.

### convolution-visualizer
- Add an "edge-detector vs blur kernel" preset gallery · two clicks reveals what each does.
- Show the formula `O = (W − K + 2P)/S + 1` updating live as user changes K, P, S.

### diffusion-denoise
- Add per-timestep `α̅_t` value display · grounds the abstract slider.
- Add a "linear vs cosine schedule" toggle · same noisy image, different intermediate states.

### dropout-playground
- "Data-flow" view · same input forwards through net with mask and rescale toggled, watch output flicker.
- "Ensemble of 8 sub-networks" view · grid of 8 thinned nets to visualize the ensemble argument.

### image-segmentation
- Add Dice-loss vs Cross-entropy comparison on a class-imbalanced toy mask.
- Add 3 image scenarios · easy (single object), hard (occlusion), imbalanced (small object).

### lora-adapter
- Add a parameter-count breakdown on every rank slider tick · show the "0.06%" number live.
- Side-by-side: full fine-tune vs LoRA training trajectory on a toy.

### moon-phases
- Already great as a non-DL primer · no changes needed.

### multivariate-normal
- Sliders for `Σ_12` covariance · students see ellipse rotating live.
- Add "marginalize y" and "condition on y=c" visualization tabs.

### numerical-tricks
- Add a `1e-12 underflow` demo · log probabilities of 1000 IID Bernoulli draws · illustrate where NaN comes from.

### object-detection
- Add NMS vs no-NMS toggle on the same predictions · students see overlap problem live.

### optical-flow
- Add a "what AR/VR uses this for" applications callout.

### rag
- Add a chunk-size slider showing retrieval quality vs context-window cost.
- Add an "LLM-only" baseline button · contrast hallucination without retrieval.

### receptive-field-grower
- Already strong · add per-stride contribution callout (each stride doubles RF growth rate).

### text-diffusion
- Add a "discrete vs continuous" comparison · why text diffusion is harder than image diffusion.

### universal-approximation
- Add "Step 2.5 · Sculpting the Lego brick" · direct sliders for one bump's position/width/height.
- Add "Step 7.5 · Efficiency of depth" · sawtooth-wave fitting · shallow needs huge width, deep does it cheaply.

### vae-latent-explorer
- Sample-from-prior button · users see novel-face generation, not just reconstructions.
- "Try to generate from a gap (β=0)" button · shows decoder garbage between clusters.

### vanishing-gradients
- Add 3 sigmoid-vs-tanh-vs-ReLU tabs · same depth, dramatically different gradient magnitudes.

### vision-transformer
- Add patch-size slider (16/32/64) · trade-off between sequence length and detail.
- Add a "what each head attends to" mini visualization for a 4-head model.

## Methodology · how this list was generated

```
python scripts/review_interactive.py --all
```

Each review prompts Gemini 2.5 Pro with the gold-standard "demystifying-p-values" pattern
(prelude → multi-step → scenarios → misconception cards → bonus) and asks for a
concrete punch list with file paths and exact code-level suggestions.
