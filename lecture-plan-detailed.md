# ES 667: Deep Learning — 24-Lecture Plan (Gentle, First-Run Edition)

**Instructor:** Prof. Nipun Batra, IIT Gandhinagar
**Semester:** Aug 2026
**Prerequisite:** ES 654 (ML 2025) — students already know MLPs, autograd, basic CNNs, gradient descent, logistic regression, L1/L2 regularization, PyTorch basics, RL basics.

---

## Textbook backbone

| Role | Resource | Use |
|------|----------|-----|
| **Primary** | Prince, *Understanding Deep Learning* (2023, free PDF — udlbook.github.io) | One required chapter per lecture. Figures are CC-BY — reusable in slides. |
| **Supplement** | Bishop & Bishop, *Deep Learning: Foundations and Concepts* (2024) | Rigorous second opinion. Instructor prep. Pointer for curious students. |
| **Labs** | Zhang et al., *Dive into Deep Learning* (d2l.ai) | Hands-on PyTorch notebooks aligned to most lectures. |
| **Video** | Karpathy, *Neural Networks: Zero to Hero* | Official video backup for L3, L13, L14, L15. |
| **Classic** | Goodfellow, Bengio, Courville (2016) | In bibliography; not required reading. |

Each lecture below lists its **Prince chapter** (required), **companion lab** (d2l section), and any **video backup**.

---

## Pedagogy conventions (applied to every lecture)

- **Socratic pacing.** Questions *before* answers. The slide that asks is followed by the slide that answers.
- **Running examples.** Cat/not-cat · MNIST · CIFAR · Tiny Shakespeare · CelebA. Every `Q.` grounds to one of these.
- **Two code threads** run across the course:
  - **Language:** micrograd → MLP → char-LSTM → Transformer → nanoGPT → LoRA fine-tune.
  - **Vision:** MLP-MNIST → CNN-CIFAR → transfer-ResNet → ViT-CLIP → diffusion.
- **One big idea per lecture.** Lectures are paced for a first-time instructor — breathing room over density.
- **Consistent slide sections:** Title → Recap (L2+) → Part N dividers → Summary → Reading pointers → Notebook → Next lecture.
- **Theme:** `anthropic` across all lectures (`slides/anthropic-theme.css`).
- **Callout vocabulary:** `.keypoint`, `.insight`, `.warning`, `.math-box`, `.popquiz`, `.paper`, `.notebook`, `.realworld`.

---

## Module 1 · Foundations & Going Deep (L1–L3)

### L1 · Why Deep Learning + MLP Recap
**Prince:** Ch 1, 3 · **Lab:** d2l §4 · **Paper:** LeCun/Bengio/Hinton 2015 (*Nature*).

**Subtopics**
1. Why DL now — data × compute × algorithms; ImageNet 2012 inflection.
2. Representation learning vs feature engineering (cat/not-cat framing).
3. Brisk MLP recap — stacked affine + non-linearity.
4. Softmax + CE from MLE; gradient $\hat{\mathbf{y}} - \mathbf{y}$.
5. Backprop as chain rule through a DAG (brisk — prereq has autograd).
6. The loss surface in high dimension.
7. Depth preview: hierarchical features; why go deeper (teaser for L2).

**Key figures** (SVG, anthropic palette): `dl_timeline`, `mlp_architecture`, `computational_graph`, `feature_hierarchy`, `train_val_test_split`, `training_loop_anatomy`.

**Assignment tie-in:** A1 · micrograd + MLP on MNIST from scratch.

---

### L2 · Universal Approximation & Going Deep
**Prince:** Ch 4, 11 · **Lab:** d2l §5 · **Paper:** He et al. 2015 (ResNet).

**Subtopics**
1. UAT — Cybenko/Hornik/Leshno; existence not efficiency.
2. Mechanism: two ReLUs = one bump; many bumps = any curve.
3. Curse of dimensionality for width alone.
4. Depth vs width — parity as motivating example (intuition, no proof).
5. Vanishing/exploding gradients; sigmoid's 0.25 ceiling.
6. ReLU family — ReLU, Leaky, GELU, SiLU; dead-ReLU.
7. Weight init — Xavier (sigmoid/tanh), He (ReLU), derived from variance preservation.
8. The degradation problem — deeper plain nets train *worse*.
9. ResNets — $\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}$; Jacobian $+\,\mathbf{I}$; gradient highway; loss-landscape smoothing.

**Key figures** (SVG): `uat_bump_construction`, `depth_compositionality`, `degradation_problem`, `resnet_block`, `gradient_highway`, `init_landscape`.

---

### L3 · Training Deep Networks in Practice
**Prince:** Ch 6 (§6.1–6.3), Ch 8 · **Lab:** d2l §3, §12 · **Video:** Karpathy *makemore-1*.

**Subtopics**
1. `nn.Module`, parameter registration, `state_dict`, device placement.
2. `Dataset` / `DataLoader` / `num_workers` / `pin_memory`.
3. Mixed precision, gradient accumulation, gradient clipping.
4. The full recipe: config → init → train → checkpoint → eval → log.
5. **Karpathy's debug ladder** — inspect data, dumb baseline, overfit one batch, small subset, LR finder, full run.
6. **Ng's error analysis** — categorize error buckets, ceiling analysis.
7. Reproducibility — seeds, determinism, config hashing.

**Key figures** (SVG): `autograd_tape`, `dataloader_pipeline`, `overfit_one_batch`, `lr_finder`, `debug_ladder`.

**Assignment A1 due shortly after L3.**

---

## Module 2 · Optimization (L4–L5)

### L4 · SGD, Momentum, Nesterov
**Prince:** Ch 6 (§6.4–6.6) · **Lab:** d2l §12 · **Paper:** Sutskever et al. 2013.

**Subtopics**
1. Non-convex landscapes — valleys, saddles, ill-conditioning.
2. Mini-batch SGD — noise as implicit regularizer; batch-size vs LR.
3. The ravine problem — why vanilla SGD oscillates.
4. Momentum — EMA of gradients; ball-rolling analogy.
5. Nesterov accelerated gradient — "lookahead" evaluation.
6. 2D trajectory comparison on an ill-conditioned quadratic.

**Key figures** (SVG): `loss_landscape_ravine`, `sgd_momentum_trajectory`, `nesterov_lookahead`.

---

### L5 · Adam, AdamW, Learning-Rate Schedules
**Prince:** Ch 6 (§6.7), Ch 7 · **Lab:** d2l §12 · **Paper:** Kingma & Ba 2014; Loshchilov & Hutter 2017.

**Subtopics**
1. AdaGrad — per-parameter adaptive LR; the LR-decay-to-zero problem.
2. RMSProp — leaky second moment fixes AdaGrad decay.
3. Adam — combines momentum (1st moment) with RMSProp (2nd moment).
4. Bias correction — why $t = 1$ estimates are biased; derivation.
5. AdamW — why L2 regularization is broken inside Adam; decoupled weight decay.
6. LR schedules — step, exponential, cosine annealing.
7. Warmup — why Transformers need it; linear-warmup + cosine pattern.
8. Hyperparameter defaults that work (by model family).

**Key figures** (SVG): `adagrad_rmsprop`, `adam_bias_correction`, `lr_schedules`, `adam_vs_sgd`.

> **Quiz 1** — end of Module 2 · covers L1–L5.

---

## Module 3 · Regularization (L6–L7)

### L6 · Classical Regularization & Data Augmentation
**Prince:** Ch 9 (§9.1–9.4) · **Lab:** d2l §5.5 · **Paper:** Belkin et al. 2019 (double descent).

**Subtopics**
1. Bias-variance in overparameterized regime; **double descent** curve.
2. L2 regularization — Gaussian prior / MAP (brisk — prereq knows ridge).
3. L1 regularization — Laplace prior; geometric L1 vs L2 (brisk).
4. Early stopping as implicit regularization.
5. Data augmentation — crops, flips, color jitter, RandAugment.
6. Mixup / CutMix — interpolated inputs and labels.
7. Label smoothing — softening hard targets.

---

### L7 · Dropout and Normalization
**Prince:** Ch 9 (§9.5–9.7), Ch 11 · **Lab:** d2l §5.6, §8.5 · **Video:** Karpathy *batchnorm*.

**Subtopics**
1. Dropout — co-adaptation prevention; inverted dropout; ensemble interpretation.
2. BatchNorm — mechanics; train vs eval modes; running statistics.
3. Original ICS hypothesis vs Santurkar's smoothing explanation.
4. LayerNorm — why sequences break BN.
5. RMSNorm — the modern simplification used in Llama (brief).
6. Norm placement — pre-norm vs post-norm.
7. Decision table: CNN → BN · Transformer → LN · LLM → RMSNorm.

---

## Module 4 · CNNs & Visual Recognition (L8–L10)

### L8 · CNN Deep Dive & Classic Architectures
**Prince:** Ch 10 (§10.1–10.4) · **Lab:** d2l §6, §7.

**Subtopics**
1. Convolution — sparse connectivity, weight sharing, translation equivariance.
2. Padding, stride, dilation; output-size formula.
3. Receptive field and effective receptive field.
4. Pooling — max vs average; translation invariance.
5. LeNet-5 → AlexNet (ReLU + GPU + dropout).
6. VGG — stacked 3×3 convolutions; depth principle.

---

### L9 · Modern CNNs & Transfer Learning
**Prince:** Ch 10 (§10.5), Ch 11 · **Lab:** d2l §8 · **Paper:** Howard et al. 2017 (MobileNet).

**Subtopics**
1. GoogLeNet / Inception — 1×1 convolutions; parallel branches.
2. ResNet (CNN edition) — bottleneck blocks; projection shortcuts.
3. MobileNet — depthwise separable convolutions.
4. EfficientNet — compound scaling.
5. Transfer learning — feature extraction vs fine-tuning.
6. Discriminative learning rates.
7. `torchvision` / `timm` survey.

**Assignment A2 · CNN + transfer learning on a small dataset.**

---

### L10 · Detection & Segmentation
**Reading:** Bishop Ch 10 + CS231n OD notes · **Lab:** d2l §14 (skim) · **Paper:** Redmon 2015 (YOLO), Ronneberger 2015 (U-Net).

**Subtopics**
1. Classification → localization — bbox regression with smooth-L1.
2. R-CNN family — R-CNN → Fast → Faster (RPN) [light].
3. YOLO — one-stage, grid, anchors.
4. IoU, NMS, mAP.
5. Semantic segmentation — FCN, **U-Net**.
6. Instance segmentation — Mask R-CNN (mention).
7. Modern: SAM for zero-shot segmentation (demo).

> **Quiz 2 / Midsem** — Modules 3–4.

---

## Module 5 · Sequence Models (L11–L12)

### L11 · RNNs, LSTMs, GRUs
**Prince/Bishop:** Bishop Ch 12 · **Lab:** d2l §9, §10.1–10.3 · **Paper:** Hochreiter & Schmidhuber 1997.

**Subtopics**
1. Why MLPs fail on sequences — weight sharing across time.
2. Vanilla RNN — hidden-state update; unrolled computation graph.
3. BPTT; truncated BPTT.
4. Vanishing gradients in time — Jacobian product; spectral radius.
5. LSTM — forget / input / output gates; cell state.
6. GRU — simplified gating.
7. Bidirectional and stacked variants.
8. When to still use RNNs in 2026 (streaming, tiny devices).

---

### L12 · Seq2Seq and Motivation for Attention
**Reading:** Bishop Ch 12 · **Lab:** d2l §10.6–10.8 · **Paper:** Sutskever et al. 2014.

**Subtopics**
1. Encoder-decoder architecture; context vector.
2. The information bottleneck — why fixed context fails on long inputs.
3. Teacher forcing — exposure bias.
4. Decoding strategies — greedy, beam, top-k, nucleus.
5. Length normalization.
6. The failure mode that motivates attention (live demo).

---

## Module 6 · Attention & Transformers (L13–L15)

### L13 · Attention Mechanism
**Prince:** Ch 12 (§12.1–12.3) · **Lab:** d2l §11.1–11.6 · **Video:** Karpathy *makemore-3*.

**Subtopics**
1. Motivation — attend to different positions per step.
2. Bahdanau (additive) attention; learned alignment.
3. Luong (multiplicative) attention.
4. Q–K–V abstraction; retrieval analogy.
5. Scaled dot-product — derivation of $\sqrt{d_k}$.
6. Self-attention — Q, K, V from the same sequence.
7. Attention heatmap intuition.

---

### L14 · The Transformer — Built Live
**Prince:** Ch 12 (§12.4–12.6) · **Lab:** d2l §11.7–11.9 · **Video:** Karpathy *nanoGPT* (official backup).

**Subtopics**
1. "Attention Is All You Need" — no recurrence.
2. Multi-head attention — parallel subspaces.
3. Positional encoding — sinusoidal.
4. The Transformer block — MHA + FFN + residual + LayerNorm.
5. Pre-norm vs post-norm.
6. Causal masking for autoregressive decoding.
7. Live build of a 100-line Transformer block.

---

### L15 · Tokenization & Pretraining Paradigms
**Prince:** Ch 12 (§12.7) · **Video:** Karpathy *tokenization*.

**Subtopics**
1. Why tokenization matters — Karpathy's failure-mode tour.
2. BPE — step-by-step; merge table construction.
3. WordPiece & SentencePiece (BERT, mT5).
4. Byte-level BPE (GPT-2, Llama).
5. Encoder-only — BERT; MLM objective; [CLS] fine-tuning.
6. Decoder-only — GPT; causal LM; generation.
7. Encoder-decoder — T5 text-to-text framing (brief).

**Assignment A3 · Build Transformer from scratch; train nanoGPT on Tiny Shakespeare.**

---

## Module 7 · LLMs (L16–L17) — simplified for first run

### L16 · Large Language Models
**Reading:** Chinchilla paper (Hoffmann 2022) + HF course Ch 1 · **Video:** Karpathy *State of GPT*.

**Subtopics**
1. Scaling laws — Chinchilla only ($C \approx 6ND$); Kaplan mentioned as preceding.
2. Data pipeline — Common Crawl, dedup, quality filtering.
3. RoPE — rotary position encoding (intuition only).
4. GQA — Grouped-Query Attention (1 slide).
5. Distributed training overview — DP · TP · PP.
6. Emergent abilities — in-context learning, chain-of-thought.

---

### L17 · Alignment & Fine-tuning
**Reading:** HF PEFT docs; Rafailov 2023 (DPO); Ouyang 2022 (InstructGPT).

**Subtopics**
1. SFT / instruction tuning — data format, chat templates.
2. **LoRA** — low-rank adapters; the main content.
3. **QLoRA** — quantization + LoRA.
4. RLHF overview — reward model + PPO (concept, not derivation).
5. **DPO** — direct preference optimization; bypasses reward model.
6. One-slide on reasoning models (o1 / Claude thinking exist; pointer).
7. Evaluation — MMLU, HumanEval, LLM-as-judge.

> **Quiz 3** — Modules 5–7.

---

## Module 8 · Self-Supervised & Vision-Language (L18–L19)

### L18 · Self-Supervised & Contrastive Learning
**Prince:** Ch 14 · **Paper:** Chen 2020 (SimCLR).

**Subtopics**
1. The labeling bottleneck — why self-supervision scaled.
2. Contrastive setup — anchor / positive / negative.
3. **SimCLR** — augmentation-based pairs; NT-Xent loss (the main content).
4. BYOL — no negatives, momentum encoder (1 slide).
5. MAE — BERT-style masking for pixels (1 slide).
6. DINO — self-distillation, emergent segmentation (1 slide).

---

### L19 · Vision-Language Models
**Prince:** Ch 12 (§12.5) + CLIP paper.

**Subtopics**
1. **ViT** — patches as tokens; [CLS] token.
2. **CLIP** — image-text contrastive pretraining (the main content).
3. Zero-shot classification with CLIP text prompts.
4. **LLaVA** — ViT → projection → LLM (1 slide).
5. Flamingo / perceiver resampler (mention).
6. 2026 multimodal landscape — GPT-4V, Claude vision, Gemini.

---

## Module 9 · Generative Models (L20–L23)

### L20 · Autoencoders & VAEs
**Prince:** Ch 17 · **Paper:** Kingma & Welling 2013.

**Subtopics**
1. Generative taxonomy — implicit vs explicit density; family tree.
2. Autoencoders — reconstruction, bottleneck, latent space.
3. AE latent discontinuity — cannot sample meaningfully.
4. VAE — probabilistic encoder, Gaussian prior.
5. Reparameterization trick.
6. ELBO derivation — reconstruction + KL.
7. Latent-space interpolation demo.

---

### L21 · GANs
**Prince:** Ch 15 · **Paper:** Goodfellow 2014, Radford 2015.

**Subtopics**
1. The minimax game — G vs D.
2. Training dynamics — alternating updates, the balance.
3. DCGAN — architectural guidelines.
4. Mode collapse — diagnosis.
5. Non-saturating G loss — the practical fix.
6. WGAN / WGAN-GP (mention only).

---

### L22 · Diffusion Models — Theory
**Prince:** Ch 18 (§18.1–18.3) · **Paper:** Ho et al. 2020 (DDPM).

**Subtopics**
1. Forward process — Gaussian noising $\beta_t$.
2. Closed-form forward sampling $q(x_t \mid x_0)$.
3. Reverse process — denoising as iterative refinement.
4. DDPM training — simplified MSE on predicted noise.
5. Noise schedules — linear vs cosine.

---

### L23 · Diffusion Models — Practice
**Prince:** Ch 18 (§18.4–18.6) + HF `diffusers` docs · **Paper:** Rombach 2022 (Stable Diffusion).

**Subtopics**
1. Classifier guidance.
2. **Classifier-free guidance** — the industry default.
3. Latent diffusion — Stable Diffusion architecture.
4. Text conditioning — cross-attention with CLIP text embeddings.
5. **DDIM** — deterministic sampling; fewer steps.
6. Practical session with `diffusers`.

**Assignment A4 · DDPM on MNIST + CFG on CIFAR.**

---

## Module 10 · Wrap-up (L24)

### L24 · Efficient Inference, Agents, Open Problems
**Reading:** Chip Huyen blog posts; HF inference docs; Dao 2022 (FlashAttention mention).

**Subtopics**
1. Prefill (compute-bound) vs decode (memory-bound).
2. **KV-cache** — why autoregressive decoding is memory-bound.
3. Quantization — FP16/BF16 → INT8 (INT4 as pointer).
4. Knowledge distillation — teacher-student, soft targets.
5. Speculative decoding — draft + verify (concept).
6. Agents & tool use — ReAct loop, function calling, Claude Code as case study.
7. Open problems — reasoning reliability, hallucination, multimodal scaling, safety.
8. Course recap; what to read next.

> **Quiz 4 / Endsem** — Modules 8–10 weighted; some coverage of everything.

---

## Assessment summary

| Item | Placement | Weight (indicative) |
|------|-----------|---------------------|
| Quiz 1 | after L5 | 12% |
| Assignment 1 (micrograd + MNIST) | after L3 | 10% |
| Assignment 2 (CNN + transfer) | after L9 | 10% |
| Quiz 2 / Midsem | after L10 | 12% |
| Assignment 3 (Transformer + nanoGPT) | after L14 | 10% |
| Quiz 3 | after L17 | 12% |
| Assignment 4 (Diffusion + LoRA) | after L23 | 10% |
| Quiz 4 / Endsem | after L24 | 12% |
| Attendance | — | 6% |
| Bonus (participation, notes) | — | 6% |

---

## Module lecture allocation

| Module | Lectures | Share |
|--------|----------|-------|
| 1 · Foundations | 3 | 12.5% |
| 2 · Optimization | 2 | 8.3% |
| 3 · Regularization | 2 | 8.3% |
| 4 · CNNs + Vision | 3 | 12.5% |
| 5 · Sequences | 2 | 8.3% |
| 6 · Attention + Transformers | 3 | 12.5% |
| 7 · LLMs | 2 | 8.3% |
| 8 · Self-sup + VLM | 2 | 8.3% |
| 9 · Generative | 4 | 16.7% |
| 10 · Wrap-up | 1 | 4.2% |

## What this plan cuts vs the original (explicit)

- Dedicated MoE / state-space lecture — replaced by one slide in L16.
- Test-time compute / reasoning deep-dive — one slide in L17.
- Mechanistic interpretability — replaced by pointer in L24.
- DiT, flow matching — omitted (pointer in L23).
- Formal Telgarsky proof — replaced by intuition + parity example.
- WGAN-GP derivation — mention only.
- Score matching / Langevin formal derivation — pointer only.
- Long-context architectures (YaRN, ring attention) — pointer only.

All of the above remain available as **self-study pointers** in the final slide of the relevant lecture.
