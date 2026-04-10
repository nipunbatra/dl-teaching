# ES 667: Deep Learning — 24-Lecture Plan (1.5 hrs each)

**Instructor:** Prof. Nipun Batra, IIT Gandhinagar
**Semester:** August 2026
**Prerequisites:** ES 654 (Machine Learning) or equivalent — students are expected to know MLPs, basic CNNs/RNNs, backpropagation, SGD, linear algebra, probability

## Design Principles

- **No redundancy with ML course** — students already know MLPs, basic CNNs/RNNs, SGD, backprop; we review briefly and go deeper
- **LLMs & VLMs get dedicated space** (3 lectures) — reflecting the modern landscape
- **Object Detection, Localization, Segmentation** included (Ng-style dedicated lecture)
- **Karpathy-inspired** build-from-scratch ethos for Transformers and Tokenization
- **GANs compressed** to 1 lecture; GNN, NTK, Normalizing Flows omitted to make room
- **4 quizzes retained** (48% weight, best 3 count) — scheduled between modules

---

## Module 1: Foundations & Deep Networks (3 lectures)

| # | Topic | Key Content |
|---|-------|-------------|
| 1 | **Why Deep Learning + MLP Recap** | DL landscape, review MLPs, computational graphs, loss surfaces |
| 2 | **Universal Approximation & Going Deep** | UAT, depth vs width, vanishing/exploding gradients, Xavier/He init |
| 3 | **Training Deep Networks in Practice** | Train/val/test, hyperparameter tuning, PyTorch end-to-end workflow |

## Module 2: Optimization (2 lectures)

| # | Topic | Key Content |
|---|-------|-------------|
| 4 | **SGD Variants** | Momentum, Nesterov, AdaGrad, RMSProp — intuitions and comparison |
| 5 | **Adam & Learning Rate Schedules** | Adam/AdamW, warmup, cosine annealing, cyclical LR, loss landscape geometry |

## Module 3: Regularization (2 lectures)

| # | Topic | Key Content |
|---|-------|-------------|
| 6 | **Classical Regularization** | L1/L2 penalties, early stopping, data augmentation strategies |
| 7 | **Modern Regularization** | Dropout, DropBlock, BatchNorm, LayerNorm — when to use what |

> **Quiz 1** (Modules 1–3)

## Module 4: CNNs & Visual Recognition (3 lectures)

| # | Topic | Key Content |
|---|-------|-------------|
| 8 | **CNN Deep Dive + Classic Architectures** | Receptive fields, feature hierarchy, LeNet, AlexNet, VGG |
| 9 | **Modern Architectures & Transfer Learning** | GoogLeNet/Inception, ResNet & skip connections, EfficientNet, fine-tuning pretrained models |
| 10 | **Object Detection, Localization & Segmentation** | Bounding boxes, IoU, NMS, anchor boxes, YOLO, R-CNN family; U-Net & semantic segmentation |

## Module 5: Sequence Models (2 lectures)

| # | Topic | Key Content |
|---|-------|-------------|
| 11 | **RNNs, LSTMs, GRUs** | Review RNNs, gradient issues, LSTM gates, GRU, bidirectional variants |
| 12 | **Seq2Seq & Applications** | Encoder-decoder, teacher forcing, beam search, language modeling |

> **Quiz 2 / Midsem** (Modules 4–5)

## Module 6: Attention & Transformers (3 lectures)

| # | Topic | Key Content |
|---|-------|-------------|
| 13 | **Attention Mechanism** | Bahdanau & Luong attention, soft attention, attention as differentiable memory |
| 14 | **The Transformer** | Self-attention, multi-head attention, positional encoding, full architecture (build it up piece by piece, Karpathy-style) |
| 15 | **Tokenization & Pretraining** | BPE tokenizer from scratch, why tokenization matters, masked LM (BERT) vs autoregressive (GPT) pretraining |

## Module 7: LLMs & Vision-Language Models (3 lectures)

| # | Topic | Key Content |
|---|-------|-------------|
| 16 | **Large Language Models** | Scaling laws, data curation, GPT/Llama training pipeline, emergent abilities, in-context learning |
| 17 | **Alignment & Finetuning** | SFT, RLHF, DPO, instruction tuning, LoRA/PEFT, "LLM psychology" |
| 18 | **Vision-Language Models** | CLIP, contrastive pretraining, LLaVA/multimodal architectures, Vision Transformers (ViT), image-text reasoning |

> **Quiz 3** (Modules 6–7)

## Module 8: Generative Models (4 lectures)

| # | Topic | Key Content |
|---|-------|-------------|
| 19 | **Autoencoders & VAEs** | Bottleneck representations, variational inference, reparameterization trick, latent spaces |
| 20 | **GANs** | Generator/discriminator, minimax, training tricks, mode collapse, DCGAN, WGAN, conditional GANs |
| 21 | **Diffusion Models: Theory** | Forward/reverse process, DDPM, score matching, noise schedules |
| 22 | **Diffusion Models: Practice** | Stable Diffusion, classifier-free guidance, conditioning, DiT (diffusion transformers) |

## Module 9: Frontiers & Wrap-up (2 lectures)

| # | Topic | Key Content |
|---|-------|-------------|
| 23 | **Self-Supervised & Contrastive Learning** | Pretext tasks, SimCLR, BYOL, MAE — why self-supervision powers modern DL |
| 24 | **Efficient DL & Open Problems** | Quantization, distillation, efficient inference (KV-cache, speculative decoding), scaling frontiers, course recap |

> **Quiz 4 / Endsem** (Modules 8–9)

---

## Lecture Allocation Summary

| Module | Lectures | % of course |
|--------|----------|-------------|
| Foundations + Optimization + Regularization | 7 | 29% |
| CNNs + OD/Segmentation | 3 | 12.5% |
| Sequence Models | 2 | 8.3% |
| Attention + Transformers | 3 | 12.5% |
| LLMs + VLMs | 3 | 12.5% |
| Generative Models (VAE/GAN/Diffusion) | 4 | 16.7% |
| Frontiers | 2 | 8.3% |

## Narrative Arc

- **Lectures 1–7**: Build the toolkit (networks, optimization, regularization)
- **Lectures 8–12**: Apply to vision and sequences (classic DL workhorses)
- **Lectures 13–18**: The attention revolution → Transformers → LLMs → VLMs (the modern stack)
- **Lectures 19–22**: Generation (VAEs → GANs → Diffusion)
- **Lectures 23–24**: Where the field is headed

## Inspirations

- **Andrew Ng (Deep Learning Specialization)**: Dedicated OD/localization/segmentation lecture structure, systematic CNN architecture survey, transfer learning emphasis
- **Andrej Karpathy (Zero to Hero / CS231n)**: Build-from-scratch Transformer teaching, tokenization as its own topic, LLM training stack (pretraining → SFT → RLHF), "LLM psychology"
- **Approved ES 667 syllabus**: Module structure, assessment scheme, recommended textbooks
