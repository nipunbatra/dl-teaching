# ES 667 slide-content audit

Date: 2026-05-01

Scope: slide content only. I am intentionally ignoring delivery, lecture timing, missing notebooks, empty labs, and assessment logistics except where a slide itself makes a content promise. Adjacent title/divider/recap slides are grouped when they have the same content issue; every substantive slide sequence is covered.

Legend:
- Keep: content is basically sound.
- Improve: concept is useful but needs sharper math, examples, caveats, or references.
- Add: missing content should be inserted near this point.
- Cut/Merge: content is redundant or lower-value than a missing idea.
- Verify: current/factual claim should be source-checked before teaching.

## Global Content Findings

1. The biggest content gap is not "few slides"; it is uneven depth. Foundations, attention, VAE, GAN, and diffusion have many worked examples. LLMs, alignment, VLMs, efficient inference, and frontier are more survey-like and need harder content: objective functions, ablations, failure modes, and comparisons.
2. Bishop and Bishop is not really integrated. The course should explicitly pull in: structured distributions, sampling, discrete/continuous latent variables, normalizing flows, and the probability view of all generative models.
3. The course overuses analogy slides. Many analogies are helpful, but several appear back-to-back before the formal object. A stronger content pattern is: problem -> minimal formalism -> numeric example -> limitation -> modern variant.
4. There is too little "what breaks" content. Each major method should include one slide on assumptions, pathologies, and diagnostics.
5. Several 2026/current-practice claims should be verified or phrased as "as of 2026" with sources. This matters most in L15-L24.
6. Missing across the whole course: normalizing flows, energy-based models, autoregressive image/audio generative models, graph neural networks, adversarial robustness, calibration/uncertainty, dataset contamination/data curation, evaluation methodology, and domain adaptation/generalization.

## L00 - Probability and MLE Recap

- S01-S07: Improve. Good primer, but too elementary if ES 654 is a real prerequisite. Reframe as "the probability objects DL repeatedly uses" rather than a probability crash course.
- S08-S15: Keep/Improve. Bernoulli, categorical, Gaussian examples are useful. Add exponential-family/NLL framing: loss = negative log probability under an assumed observation model.
- S16-S22: Improve. Conditional probability, Bayes, sampling, IID are essential. Add one slide on exchangeability/data leakage because this matters for train/test splits and modern web-scale data.
- S23-S28: Improve. Entropy, KL, log-sum-exp are important. Add cross-entropy as `H(p, q) = H(p) + KL(p||q)` so CE is not only "classification loss."
- S29: Add. The "log-likelihood gradient" slide should connect directly to score functions and later diffusion score matching.
- S30-S32: Keep, but add caveat. MAP -> L2 is good. Also mention L1/Laplace and why priors are not just penalties; they encode inductive bias.
- S33-S41: Improve. Linear algebra and likelihood basics duplicate L1. Keep only if L00 remains an appendix. Add shape conventions and batch notation used in later slides.
- S42-S50: Keep/Improve. MSE and CE derivations are valuable. Add "wrong likelihood -> wrong loss" with one concrete example, e.g. MSE for classification is poorly calibrated.
- S51-S56: Add. Include a preview slide mapping probability tools to later topics: CE -> classifiers, KL -> VAE/DPO, change-of-variables -> flows, score -> diffusion, sampling -> generation.

Missing from L00: Jensen's inequality, change of variables, reparameterization preview, and the difference between generative and discriminative likelihoods.

## L01 - Why Deep Learning + MLP Recap

- S01-S13: Cut/Merge. History, roadmap, "why now" are useful but should not consume much content. Add one slide answering: "What can deep nets represent that feature engineering cannot?"
- S14-S19: Keep. Neuron/vector-form recap is fine.
- S20-S23: Keep. Linear-collapse proof and numeric example are strong. Add mini-batch matrix version because later code/math uses batched tensors.
- S24-S28: Improve. Activation and PyTorch slides are practical. Add activation pathologies: dead ReLU, saturation, GELU/SiLU why they became common.
- S29-S39: Improve. CE gradient derivation is strong but duplicates L00. If L00 is taught, replace some derivation time with calibration/proper scoring rules and logits vs probabilities.
- S40-S45: Keep. Linear-layer backward derivation is good. Add shape annotations for `dW`, `db`, and `dx`.
- S46-S54: Improve. UAT/depth/vanishing-gradient preview is useful, but "biology does the same thing" should be softened or removed. Add a cleaner trichotomy: expressivity, trainability, generalization.
- S55-S61: Improve. Training loop and train/val/test are fine. Add `model.train()`/`model.eval()` and `torch.no_grad()` as conceptual differences, not only API details.

Missing from L01: representation learning as learned invariances, overparameterization as a central puzzle, calibration, and a "what assumptions are we making?" slide.

## L02 - Universal Approximation and Going Deep

- S01-S15: Improve. UAT content is solid, but make the limitations more central: existence is not efficient representation, not optimization, not generalization.
- S16-S21: Improve. Depth-vs-width/parity is useful. Add one more concrete compositional function example, not only parity.
- S22-S29: Keep/Improve. Vanishing gradients and ReLU family are good. Add exploding gradients and residual/norm as separate fixes.
- S30-S44: Keep. ResNet sequence is strong. Add pre-activation ResNet and projection shortcut details; clarify when identity addition is legal shape-wise.
- S45-S53: Improve. Initialization math is good. Add fan-in/fan-out definitions and why ReLU needs a factor of 2. Mention orthogonal initialization/dynamical isometry only as optional.
- S54-S55: Add. Summary should explicitly separate the three lessons: universal approximation, depth efficiency, optimization stability.

Missing from L02: expressivity vs optimization vs generalization as separate axes; BatchNorm's role in ResNet history; pre-activation residual blocks.

## L03 - Training Deep Networks in Practice

- S01-S11: Keep. PyTorch stack and autograd content is useful.
- S12-S15: Improve. Data pipeline should include data leakage, class imbalance, and train/val/test transform differences.
- S16-S19: Verify/Improve. BF16 default claim needs a source/caveat. Add FP16 loss scaling and what underflow looks like.
- S20-S25: Keep. Accumulation and clipping examples are good. Add optimizer-step order with gradient accumulation.
- S26-S29: Improve. "Full recipe" should include metric choice, early stopping criterion, checkpoint contents, and config immutability.
- S30-S39: Keep. Debug ladder is one of the strongest content blocks. Add a minimal "bad run -> diagnosis" table.
- S40-S44: Keep/Improve. Error analysis and ceiling analysis are valuable. Add confusion matrix and slice-based evaluation.
- S45-S48: Improve. Reproducibility should distinguish bitwise determinism from statistical reproducibility.

Missing from L03: evaluation leakage, distribution shift between train/val/test, ablation discipline, and loss/metric mismatch.

## L04 - SGD, Momentum, Nesterov

- S01-S16: Keep/Improve. Landscape/ravine/saddle content is strong. Add stochastic approximation view: SGD estimates full gradient with noisy mini-batch gradient.
- S17-S28: Keep. Momentum sequence is good.
- S29-S37: Improve. Nesterov content is fine, but reduce analogy density and add exact relation between classical NAG and PyTorch's implementation.
- S38-S44: Add. Add batch-size/LR scaling, noise scale, and sharp-vs-flat minima caveats. Current recommendations need a "depends on model family" table with caveats.
- S45: Improve. Summary should say when SGD-momentum still wins over AdamW and why this is empirical, not theorem.

Missing from L04: gradient noise scale, large-batch training, LR scaling rules, and caveats around flat minima/generalization claims.

## L05 - Adam, AdamW, Schedules

- S01-S07: Keep. Family tree works.
- S08-S15: Keep. AdaGrad/RMSProp build-up is useful.
- S16-S27: Keep. Adam and bias correction are among the better derivation blocks.
- S28-S32: Keep/Improve. AdamW explanation is good. Add why L2 inside adaptive optimizers is not equivalent to weight decay.
- S33-S41: Improve. LR schedules should include warmup+cosine as default but also one-cycle, constant-with-decay, and why schedule choice is coupled to total budget.
- S42-S44: Add. Add optimizer failure diagnostics: loss spikes, divergence, plateau, unstable validation, too-large weight decay.
- S45: Improve. Summary should include a decision table: CNN-from-scratch, pretrained fine-tune, Transformer pretrain, LoRA.

Missing from L05: epsilon/beta sensitivity, Adam generalization caveat, decoupled weight decay derivation, and schedule-budget coupling.

## L06 - Regularization

- S01-S06: Keep. Setup is good.
- S07-S11: Improve. Double descent needs careful wording; avoid implying more parameters always help. Add interpolation threshold, label noise, and benign overfitting caveat.
- S12-S19: Keep/Improve. L1/L2/early stopping are good. Add Bayesian prior interpretation only if not already covered in L00.
- S20-S24: Improve. Augmentation coverage is useful. Add leakage/invalid augmentation examples as content, not just warning.
- S25-S32: Keep. Mixup/CutMix math is good. Add limitations: localization tasks, calibration side effects, and when Mixup hurts.
- S33-S38: Keep/Improve. Label smoothing good. Add downside: can hurt distillation/confidence estimates and rare-class learning.
- S39-S51: Improve. Dropout content is strong but should include stochastic depth/DropPath because modern ViTs/ResNets use it more than vanilla dropout.
- S52-S63: Improve. Norm section is important. Add GroupNorm and WeightNorm; BatchNorm failures should include small batch, distributed stats, and train/eval mismatch.
- S64-S71: Keep/Improve. Norm placement and full stack are good. Add EMA/SWA and SAM as modern regularization/optimization-adjacent tools.
- S72-S73: Improve. Summary should organize methods by what they regularize: weights, data, activations, architecture, objective, optimizer.

Missing from L06: calibration, GroupNorm, stochastic depth, EMA/SWA, SAM, augmentation validity, and regularization for Transformers vs CNNs vs small data.

## L07 - CNN Deep Dive and Classic Architectures

- S01-S15: Keep. Mechanics, output size, pooling, equivariance/invariance are good.
- S16-S24: Keep/Improve. Receptive-field content is useful. Add effective receptive field caveat: theoretical RF != influence distribution.
- S25-S31: Improve. Architecture progression should include what changed in optimization, data, compute, not only block shapes.
- S32-S37: Keep. Inductive bias section is strong. Add failure modes: non-stationarity, rotation/scale sensitivity, adversarial texture bias.
- S38-S39: Improve. CNN block and feature visualization need stronger content. Add Grad-CAM/activation maximization as a bridge to Vineeth-style "understanding CNNs."
- S40-S42: Improve. FAQ/summary should include "when CNNs still beat ViTs."

Missing from L07: CNN backprop intuition, Grad-CAM/feature visualization depth, texture bias, and ConvNeXt-style "modernized CNN" bridge.

## L08 - Modern CNNs and Transfer Learning

- S01-S08: Keep. Inception and 1x1 bottleneck content is solid.
- S09-S18: Keep/Improve. ResNet CNN block is good. Add DenseNet briefly or explicitly cut it.
- S19-S25: Keep. MobileNet/depthwise math is useful.
- S26-S30: Improve. EfficientNet content should include why compound scaling was empirical and how it was superseded/absorbed by modern architecture search and ConvNeXt.
- S31-S39: Improve. Transfer learning is important. Add preprocessing contract: input resolution, normalization, classifier head, frozen BN, and domain shift.
- S40-S42: Improve. Backbone table and `timm` ecosystem are practical but need more conceptual comparison: supervised ImageNet vs self-supervised DINOv2/MAE vs CLIP backbones.
- S43: Improve. Summary should connect CNN transfer to later ViT/CLIP transfer.

Missing from L08: DenseNet/ConvNeXt, frozen BatchNorm pitfalls, preprocessing mismatch, self-supervised backbones, and domain shift taxonomy.

## L09 - Detection and Segmentation

- S01-S09: Keep. Classification/localization/detection and multitask loss are useful.
- S10-S18: Keep. IoU/NMS/AP worked examples are strong. Add mAP@0.5 vs mAP@[.5:.95] distinction.
- S19-S28: Improve. R-CNN/YOLO/DETR sequence is good but thin. Add anchors, focal loss/RetinaNet, Hungarian matching for DETR, and why NMS disappears in DETR.
- S29-S38: Keep/Improve. U-Net and Dice are good. Add class imbalance and boundary-sensitive losses.
- S39: Improve. Mask R-CNN should include RoIAlign because it is the conceptual innovation.
- S40-S43: Verify/Improve. SAM/open-vocab claims need caveats: prompt sensitivity, domain-specific failure, no semantic label by itself.
- S44: Improve. Summary should separate metrics, architectures, and open-vocabulary trend.

Missing from L09: anchor design, focal loss, Hungarian matching, RoIAlign, panoptic segmentation, and detection evaluation pitfalls.

## L10 - RNNs, LSTMs, GRUs

- S01-S11: Keep. RNN motivation and by-hand cell are fine.
- S12-S18: Keep. BPTT and vanishing-gradient sequence is strong. Add computational graph memory cost and teacher forcing bridge.
- S19-S28: Keep. LSTM gate derivations and numeric examples are useful.
- S29-S35: Keep/Improve. GRU and variants are fine. Add peephole/projection only as optional if needed.
- S36-S38: Improve. Mamba/RWKV slide is too shallow for a named claim. Either cut or add a minimal state-space equation and why linear-time sequence modeling matters.
- S39-S40: Improve. Preview to attention should explicitly name the fixed-state bottleneck.

Missing from L10: sequence likelihood objective, embeddings, CTC for alignment-free sequence labeling, and a rigorous state-space/Mamba caveat if mentioned.

## L11 - Seq2Seq and Motivation for Attention

- S01-S13: Keep/Improve. Encoder-decoder setup is good. Add probabilistic objective: product of conditional token probabilities.
- S14-S23: Keep. Teacher forcing/exposure-bias sequence is useful. Add scheduled sampling as historical idea with caveat.
- S24-S36: Keep/Improve. Decoding strategies are useful. Add length penalty and coverage penalty as real NMT fixes; distinguish search error from model error.
- S37-S41: Keep. Bottleneck motivation is good. Add quantitative long-sequence degradation or BLEU-by-length figure.
- S42-S43: Improve. Applications should include speech recognition and image captioning, but mention Transformers replaced most classic RNN seq2seq.
- S44: Improve. Summary should explicitly hand off to attention: "replace one vector with a distribution over encoder states."

Missing from L11: conditional likelihood formula, length/coverage penalties, BLEU/evaluation caveats, and scheduled sampling as a failed/partial fix.

## L12 - Attention Mechanism

- S01-S10: Keep. Problem framing and alignment visuals are good.
- S11-S16: Keep. Bahdanau/Luong transition is strong. Add parameter count/complexity contrast between additive and multiplicative attention.
- S17-S27: Keep. QKV abstraction and numeric examples are among the strongest slides in the course.
- S28-S35: Keep/Improve. Scaling derivation is good. Add "temperature" connection and caveat that assumptions are approximate after learned projections.
- S36-S40: Keep. Self/cross-attention distinction is clear.
- S41: Improve. Self-attention vs convolution should be more nuanced: learned bias does not always beat hand-designed bias in small-data regimes.
- S42-S46: Improve. Add padding masks and attention dropout; causal mask alone is incomplete.
- S47-S49: Improve. "What attention unlocked" should include parallelism, differentiable memory, and long-range dependency, but avoid implying attention explains reasoning.
- S50: Add. Add caveat: attention weights are not necessarily explanations.

Missing from L12: padding masks, attention dropout, attention-as-kernel-smoother view, and "attention is not explanation" caveat.

## L13 - Transformer

- S01-S12: Keep/Improve. Block and pre-norm content is good. Add RMSNorm/SwiGLU as modern variants near FFN.
- S13-S22: Keep. Multi-head content and parameter accounting are useful. Add memory/FLOPs accounting separately from parameter count.
- S23-S28: Improve. Positional encoding good. Add ALiBi/RoPE comparison if RoPE later relies on it.
- S29-S37: Keep/Improve. Full architecture and GPT-tiny content are good. Add encoder-only/decoder-only objective differences more explicitly.
- S38: Improve. Common variations should include GQA/MQA, RoPE, RMSNorm, SwiGLU, FlashAttention as a "modern block delta."
- S39: Keep/Improve. Debug slide is valuable; add exact symptoms: uniform attention, NaNs, no position signal, mask leak.
- S40-S41: Improve. Summary should say the Transformer is a template, not one architecture.

Missing from L13: FLOPs/memory accounting, SwiGLU/RMSNorm, encoder-only vs decoder-only training objective contrast, and mask-leak diagnostics.

## L14 - Tokenization and Pretraining

- S01-S08: Keep. Tokenization problem setup is good.
- S09-S18: Keep/Improve. BPE sequence is useful. Add Unicode normalization, byte fallback, multilingual trade-offs, and tokenizer/data coupling.
- S19: Keep/Improve. Gotchas are important. Add prompt-injection/token-boundary examples only if verified.
- S20-S31: Keep. BERT/GPT/T5 objective comparison is good. Add denoising objectives beyond MLM: span corruption, prefix LM.
- S32-S39: Improve/Verify. Scaling and compute economics need sources and caveats. Add data curation, deduplication, benchmark contamination, and copyrighted/PII filtering as content.
- S40: Improve. Fine-tuning slide is too brief for the bridge to L16. Add SFT objective and chat-template formatting.
- S41: Improve. Summary should distinguish tokenizer, objective, data, and scale as separate knobs.

Missing from L14: Unicode/byte-level details, multilingual tokenization, data contamination/dedup, span corruption, and chat-template bridge.

## L15 - Large Language Models

- S01-S05: Verify/Improve. 2018-2026 claims need dates/sources. Add "architecture changed less than training/data/serving."
- S06-S14: Keep/Improve. Chinchilla section is useful. Add actual scaling-law form and loss-vs-compute caveat; do not reduce the paper to only D/N=20.
- S15-S22: Keep. RoPE derivation and numeric example are good.
- S23-S28: Improve. KV/GQA content overlaps L23. Keep here only as architecture motivation; move serving economics to L23 or cross-reference.
- S29-S34: Improve. Distributed training is too survey-like. Add one concrete tensor-sharding example for a linear layer.
- S35-S43: Improve. Emergence/reasoning slides need stronger skepticism: metric discontinuity, prompting artifacts, contamination, and evaluation design.
- S44: Improve. Summary should add data pipeline as a first-class LLM component: filtering, dedup, mixing, curriculum, eval contamination.

Missing from L15: data curation, contamination, MoE, eval methodology, actual scaling-law equations, and a concrete distributed matmul example.

## L16 - Alignment and Fine-tuning

- S01-S07: Improve. SFT content needs chat template, loss masking, assistant-only loss, and dataset quality failure modes.
- S08-S19: Keep. LoRA/QLoRA parameter and memory content is strong. Add which modules get LoRA (`q_proj`, `v_proj`, etc.) and rank/alpha/dropout trade-offs.
- S20-S27: Improve. RLHF is useful but PPO is oversimplified. Add reward model training objective and KL penalty more explicitly.
- S28-S34: Keep/Improve. DPO content is good. Add assumptions behind DPO and failure modes with noisy preferences.
- S35: Improve. Constitutional AI/RLAIF needs more than a name: critique generation, revision, preference model, safety constitution.
- S36-S39: Verify/Improve. Reasoning model benchmark claims are high-risk. Source them and distinguish outcome rewards, process rewards, search, and distillation.
- S40-S41: Improve. Picking method should include data availability, compute, safety risk, and model size.

Missing from L16: chat-template/loss masking, reward model objective, PPO clipped objective or at least KL-regularized policy view, preference-data bias, and alignment evaluation.

## L17 - Self-Supervised and Contrastive Learning

- S01-S07: Keep/Improve. Labeling bottleneck framing is good. Add pretext-task failure: solving the pretext task without learning useful semantics.
- S08-S21: Keep. SimCLR/InfoNCE sequence is strong. Add supervised contrastive loss and hard-negative issue.
- S22-S26: Improve. BYOL/MoCo/SwAV content needs sharper distinction: memory bank, momentum encoder, clustering, collapse prevention.
- S27-S29: Keep/Improve. Linear probe vs fine-tune is good. Add kNN eval and transfer to dense tasks.
- S30-S35: Keep. MAE pipeline is useful. Add why high mask ratio works for images but not all modalities.
- S36-S40: Improve/Verify. DINOv2 and 2026 landscape claims need caveats. Add "SSL is not always better than supervised pretraining" for small/clean labeled data.
- S41: Improve. Summary should unify SSL as invariance learning, predictive coding, or distillation.

Missing from L17: supervised contrastive learning, hard negatives, memory banks, collapse theory, kNN/linear-probe evaluation, and dense-prediction transfer.

## L18 - Vision-Language Models

- S01-S14: Keep/Improve. ViT section is good. Add class token vs pooling and positional embedding interpolation.
- S15-S24: Keep. CLIP derivation and zero-shot content are strong. Add dataset-noise and spurious-correlation limitations.
- S25-S31: Improve. LLaVA/Flamingo bridge is useful. Add BLIP-2/Q-Former because it is the clean conceptual middle between CLIP and LLaVA.
- S32-S36: Verify/Improve. 2026 multimodal-state claims need current sourcing and less model-name dependence.
- S37-S40: Keep/Improve. Prompting, benchmarks, hallucination are valuable. Add grounding metrics and failure taxonomy: OCR, counting, spatial reasoning, medical images.
- S41-S42: Improve. Applications should connect to safety and evaluation; VLM demos alone can mislead.

Missing from L18: BLIP/BLIP-2/Q-Former, dataset noise, CLIP limitations, positional interpolation in ViT, and grounded evaluation.

## L19 - Autoencoders and VAEs

- S01-S15: Keep. AE setup and "not generative" point are strong.
- S16-S24: Keep/Improve. VAE/ELBO derivation is good. Add graphical model notation: `p(z)`, `p_theta(x|z)`, `q_phi(z|x)`.
- S25-S28: Keep/Improve. KL and posterior collapse are important. Add mitigation: KL annealing, free bits, beta schedules, decoder weakening.
- S29-S35: Keep. Reparameterization/PyTorch sequence is strong.
- S36-S38: Improve. Beta-VAE/CVAE content should include disentanglement caveats; beta-VAE does not guarantee semantic disentanglement.
- S39-S43: Keep/Improve. Sampling/interpolation good. Add reconstruction-vs-sample-quality distinction.
- S44-S46: Add. Add VQ-VAE/WAE/normalizing-flow bridge or explicitly say these are cut.

Missing from L19: amortized inference terminology, graphical model view, KL annealing, VQ-VAE, WAE, and normalizing-flow bridge.

## L20 - GANs

- S01-S13: Keep. Motivation and toy setup are good.
- S14-S17: Improve. Optimal discriminator derivation should not be optional if you want depth; it explains JS divergence and support mismatch.
- S18-S23: Keep. Training loop and non-saturating loss content is strong.
- S24-S33: Keep/Improve. DCGAN content is useful but historically dated. Add spectral normalization and conditional GANs.
- S34-S40: Keep. Instability/mode collapse/FID sequence is good. Add FID limitations and precision/recall for generative models.
- S41-S48: Improve. WGAN content is useful but "fixed everything" is too strong. WGAN helps gradients but does not eliminate all GAN instability.
- S49-S52: Improve. StyleGAN needs content on mapping network, AdaIN/modulation/demodulation, noise injection.
- S53-S59: Keep/Improve. "GANs in 2026" framing is good. Add non-image GAN use and adversarial losses in restoration/domain adaptation.
- S60: Improve. Summary should identify the durable idea: learned critic/loss, not necessarily GAN generation.

Missing from L20: JS divergence/support mismatch as core, conditional GANs, spectral norm, FID caveats, StyleGAN mechanism, and precision/recall metrics.

## L21 - Diffusion Theory

- S01-S09: Keep. Big-picture setup is good.
- S10-S20: Keep/Improve. Forward process and closed form are strong. Add `alpha_t`, `bar_alpha_t`, SNR, and why schedules are better understood through SNR.
- S21-S28: Keep. DDPM loss and training step are good. Add prediction parameterizations: epsilon, x0, v.
- S29-S32: Improve. U-Net/time conditioning good. Add cross-attention only as preview for L22.
- S33-S40: Improve. Score connection is valuable but "uphill/downhill" metaphors can confuse. Clarify score points toward higher density; sampling follows learned denoising/score with noise.
- S41-S46: Improve/Verify. Why diffusion won/frontier claims need caveats. Add sampling cost and likelihood/evaluation limitations.
- S47-S48: Improve. Summary should distinguish DDPM discrete-time view, score-SDE view, and flow-matching/ODE view.

Missing from L21: SNR framing, v-prediction, ELBO weighting caveat, DDIM/ODE bridge, likelihood limitations, and classifier guidance preview.

## L22 - Diffusion Practice

- S01-S09: Keep. Conditioning/cross-attention content is good.
- S10-S18: Keep/Improve. CFG derivation and arithmetic are strong. Add negative prompts and oversaturation/artifact failure at high guidance.
- S19-S25: Keep. Latent diffusion stack is useful. Add VAE compression artifacts and why latent-space quality matters.
- S26-S31: Keep/Improve. Faster sampling content is good. Add ancestral vs deterministic samplers and solver order.
- S32-S39: Improve/Verify. DiT and modality landscape claims should be sourced. Add why patchifying latent/video changes memory cost.
- S40-S42: Improve. Consistency/flow matching are too brief to be meaningful. Either add one formal objective/diagram each or reduce to a "what to read next" slide.
- S43: Improve. Summary should say what changed from theory to practice: conditioning, latent space, guidance, sampler, backbone.

Missing from L22: guidance failure modes, negative prompts, scheduler taxonomy, VAE compression limitations, formal flow-matching objective, and consistency-model caveat.

## L23 - Efficient Inference

- S01-S07: Keep/Improve. Prefill/decode and memory-bound argument are useful. Add roofline-style compute-vs-bandwidth framing.
- S08-S14: Keep. KV-cache and paged attention are good. Add continuous batching and prefix caching because they are core serving content.
- S15-S22: Improve. Quantization content is useful. Add per-tensor vs per-channel/group quantization, activation quantization, calibration, SmoothQuant/GPTQ/AWQ distinctions.
- S23-S28: Keep. FlashAttention tile story is good. Add exact thing saved: materializing attention matrix in HBM.
- S29-S32: Keep/Improve. Speculative decoding good. Add acceptance-rate formula/intuitive bound.
- S33-S37: Improve. Distillation is too isolated. Add distillation for reasoning, preference, and small models; distinguish logit distillation vs data distillation.
- S38-S42: Verify/Improve. Framework/cost claims need current sources. Add batching, tensor parallel serving, load balancing, and latency/throughput trade-off.
- S43: Improve. Summary should separate model compression, attention kernels, scheduling, and decoding algorithms.

Missing from L23: continuous batching, prefix caching, roofline analysis, group-wise quantization, activation quantization, and latency vs throughput.

## L24 - Frontier, Agents, Reasoning, Interpretability

- S01-S10: Improve/Verify. Agents/tool-use content is understandable but too product-like. Add agent failure modes: tool error, compounding mistakes, prompt injection, sandboxing, verification.
- S11-S20: Improve/Verify. Reasoning model section needs stronger rigor and source-locking. Add distinction between chain-of-thought prompting, RL on verifiable tasks, search, process reward models, and distillation.
- S21-S29: Improve. Mechanistic interpretability is valuable but too compressed. Add "attention heads are not all circuits" caveat, activation patching, causal tracing, and SAE limitations.
- S30-S34: Improve. Open problems are currently a list. Turn them into axes: reliability, data, compute, grounding, robustness, interpretability, safety, evaluation.
- S35-S41: Cut/Merge. Course recap/what to do next is not content-heavy. Replace some recap with a final synthesis: "what ideas lasted across the course" - likelihood, compositionality, invariance, optimization, scale, latent variables, sampling.

Missing from L24: agent safety/failure modes, prompt-injection/tool-use security, causal interpretability methods, SAE limitations, rigorous reasoning-model taxonomy, and evaluation of open-ended agents.

## Course-Level Content Cuts and Additions

### Add or Strengthen

1. Normalizing flows: at least one lecture segment near L19/L20. Bishop and Bishop includes this as a chapter; your generative arc jumps VAE -> GAN -> diffusion and misses exact likelihood/change-of-variables.
2. Energy-based models: even one conceptual block would connect contrastive learning, score matching, and learned critics.
3. Graph neural networks: either add a compact lecture/appendix or explicitly cut. Bishop includes GNNs; the current course has no graph representation learning.
4. Robustness and adversarial examples: currently missing despite being a durable DL topic and strongly relevant to vision/medical/real-world ML.
5. Data curation and evaluation contamination: essential for LLMs/VLMs. Add to L14/L15.
6. Calibration and uncertainty: add around L01/L06 and revisit in detection/VLM safety.
7. Domain adaptation/generalization: add around transfer learning/self-supervision/VLMs.
8. Evaluation methodology: one recurring slide pattern for each module: "What metric? What does it miss?"

### Cut or Compress

1. Repeated analogy slides when the formal concept is already clear.
2. Repeated "2026 landscape" slides that list model names without an enduring principle.
3. Some recap/roadmap/what-next slides, if replacing them with missing technical content.
4. Product-specific frontier claims unless sourced and tied to durable mechanisms.

### Most Important Slide-Level Repairs

1. L15: Add data curation/eval contamination and actual scaling-law caveats.
2. L16: Add chat-template/loss masking, reward-model objective, and preference-data failure modes.
3. L19-L22: Add flows/EBM/score/SNR bridge so generative modeling becomes one coherent probabilistic arc.
4. L09/L18: Add evaluation caveats and failure modes for detection, segmentation, CLIP, and VLMs.
5. L06/L07: Add calibration, GroupNorm, stochastic depth, Grad-CAM, and CNN failure modes.
6. L24: Replace product survey with durable concepts: agent reliability, verification, causal interpretability, and frontier evaluation.
