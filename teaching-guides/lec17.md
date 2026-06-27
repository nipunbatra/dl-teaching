# Lecture 17 · Self-Supervised & Contrastive Learning · Teaching Guide
*First-time-instructor companion. Audience: undergrads, one ML course.*

## The spine (say this in 2 sentences)
Labels are scarce and expensive; unlabeled data is effectively infinite — so we invent a "label" from the data itself (a surrogate task) and learn representations that transfer to real downstream tasks. Every method here is one way to write a *free supervised signal* out of raw data: match two views of the same image (SimCLR), match a teacher's view without negatives (BYOL/DINO), or delete 75% of an image and reconstruct it (MAE).

## Where it sits
Follows the supervised-vision arc (L7–L9 CNNs) and the Transformer arc (L13–L14). It names the limitation of everything so far — *we needed labels* — and removes it. It directly sets up L18 (CLIP is the same contrastive trick, but the second "view" of an image is its caption) and underpins diffusion's pretraining (L21). **Landscape/literacy** lecture: the InfoNCE softmax is the one piece of math worth doing; the rest is "which recipe, and why it doesn't collapse."

## 80-minute plan
- **The labeling bottleneck** ⭐ ~8 min — the baby-learns-"cat" pop quiz, the cost table ($7M to label ImageNet), the surrogate-task list. Sets the *why*.
- **SimCLR + InfoNCE** ⭐ ~24 min — build-the-batch (2N views, 1 positive, 2N−2 negatives) → encoder → projection head → InfoNCE as softmax classification. The InfoNCE derivation is ⭐⭐⭐ but the *softmax interpretation* is core. **Do live on the board:** the temperature effect (below).
- **Projection head + augmentations** ⭐⭐ ~10 min — the crumple-zone analogy and the augmentation ablation table (crop+color+blur = 64% vs crop-only = 40%).
- **BYOL (no negatives, why no collapse)** ⭐⭐ ~14 min — student/teacher, EMA, stop-gradient, predictor head. The three anti-collapse forces.
- **MAE** ⭐⭐ ~12 min — mask 75%, asymmetric encoder/decoder (expert-and-intern), why the encoder runs on only 25%.
- **DINO(v2) + the landscape** ⭐⭐ ~8 min — self-distillation at scale, emergent segmentation, "DINOv2 is the default frozen-feature backbone." Eval protocols (linear probe vs fine-tune) ⭐⭐⭐ if time.
- **Buffer** ~4 min.

## Teach it like this (hook → sequence → payoff)
**Hook:** "How does a baby learn what a cat is, with no one ever saying the word 'cat'?" The answer — *the same cat from two angles is the same thing* — is **literally the SimCLR loss**. **Sequence:** formalize that as "two augmentations of one image should be close, everything else far" (SimCLR/InfoNCE) → but that needs lots of negatives and big batches; can we drop negatives? (BYOL: yes, with asymmetry) → can we skip contrast entirely and just reconstruct? (MAE: mask and rebuild) → at scale, self-distillation (DINOv2) gives features that beat supervised ImageNet. **Payoff:** "The labels were inside the data all along" — supervised learning now survives only as a thin fine-tuning layer at the end.

## Heads-up for YOU (subtle points to get right)
1. **Contrastive learning needs *both* negatives *and* augmentations — they do different jobs.** Augmentations *define the positive* ("these two crops are the same thing → pull together"). Negatives *prevent collapse* and define what to push away. Drop the negatives and SimCLR collapses; drop the augmentations and there's no positive pair to learn from. (BYOL is the special case that engineers around *no* negatives — that's why it's surprising.)
2. **The projection head is not a minor detail — it's a *shield*.** InfoNCE forces its output to be invariant to augmentations (e.g. ignore color). If applied to the encoder directly, the encoder would *delete* color knowledge. The projection head absorbs that invariance so the encoder *h* keeps rich, general features. That's why you *throw away the projection head* and use *h* downstream.
3. **BYOL doesn't avoid collapse by EMA alone.** Three ingredients together: the *predictor head* (asymmetry — student must *predict*, not match), *stop-gradient* on the target (teacher can't move to meet the student), and the *EMA* update (target trails slowly so the student never exactly catches it). Crediting only the EMA is the common error — the predictor head is essential. (Grill et al. ran 300 epochs to convince people it genuinely doesn't collapse.)
4. **MAE is *asymmetric*, not a normal autoencoder.** The heavy encoder sees only the *visible 25%* of patches (the ~4× pretraining speedup). The lightweight decoder gets the encoded patches *plus learned mask tokens* and reconstructs the full grid. Positional embeddings are added *before* the encoder, so the visible patches know where they are.
5. **Small temperature τ is a *contrast amplifier*, not just a "softmax sharpener."** Dividing similarities by τ = 0.1 turns a 0.7 similarity gap into a logit gap of 7, forcing the model to make the positive *much* more similar than even the hardest negative. That's where the gradient signal comes from. (SimCLR ~0.07; CLIP learns it.)
6. **Linear probe vs fine-tune measure different things.** Linear probe (encoder frozen, train a 1-layer classifier) isolates *inherent feature quality*. Fine-tune tests the *ceiling* but a strong fine-tuner can mask a mediocre encoder. When the slides say "DINOv2 wins linear probe, MAE wins detection," that distinction is the point.

## Where students stumble (and the fix)
1. **"How is this learning anything if there are no labels?"** Fix: reframe InfoNCE as a *classification problem with 2N−1 classes* where the "label" is the batch position of the other augmentation of the same image. The data manufactures its own label.
2. **Thinking bigger batches help "because more data."** Fix: bigger batches mean *more negatives per anchor* → sharper contrast → harder negatives in the denominator. SimCLR needed batch 8192 *for the negatives*, not for throughput.
3. **Believing BYOL "shouldn't work."** Fix: walk the three forces (predictor + stop-grad + EMA) and the EMA worked-numeric (teacher trailing the student). The asymmetry is what breaks the symmetry that would otherwise collapse it.
4. **Treating augmentation choice as a hyperparameter footnote.** Fix: the ablation table. The augmentations you pick *become* the invariances of your representation — "ignore crop, ignore color, attend to content" is a modeling decision, not a knob.
5. **Conflating MAE's encoder and decoder roles.** Fix: expert-and-intern analogy. Encoder = expensive expert, sees 25%. Decoder = cheap intern, reconstructs 100%, thrown away after pretraining.

## If a student asks…
- **"What if a 'negative' in the batch is actually another photo of the same class (a false negative)?"** It is genuinely a flaw — SimCLR will wrongly push them apart. With large datasets and big batches the chance is low enough that it averages out and features are still excellent. (Methods like supervised contrastive or clustering-based SwAV reduce it.)
- **"Why is color jitter so important?"** (HARD-ish) Without it, two crops of the same image share an almost identical color histogram, so the model "cheats" by matching color statistics instead of learning shape/semantics. Color jitter removes that shortcut and forces content understanding. (Hence crop-only = 40% but crop+color = 56%.)
- **"What happens to InfoNCE with batch size 2 (one positive pair, zero negatives)?"** (HARD) It breaks: the denominator is just the positive term, the softmax = exp(sim)/exp(sim) = 1, and loss = −log 1 = 0. No gradient. Contrastive learning *needs* negatives to form a non-trivial distribution.
- **"If MAE only encodes 25% of patches, how does it know where they go?"** Positional embeddings are added to the visible patches before the encoder, so the Transformer always knows each patch's spatial coordinates. The mask tokens (added at the decoder) carry their positions too.
- **"When does contrastive beat MAE?"** Contrastive (SimCLR) tends to win pure classification / linear-probe and gives strongly invariant global features; MAE tends to win dense tasks (detection, segmentation) and tolerates any batch size with minimal augmentation. Pick by downstream task.
- **"Is next-token prediction in GPT the same family as this?"** Yes — it's self-supervision via "predict the missing/next part." Text had SSL since word2vec (2013); vision caught up around 2020. The meta-idea ("make the data supervise itself") is identical.

## If you're short on time
- **Cut:** the ⭐⭐⭐ InfoNCE step-by-step derivation (keep the softmax-classification framing); the four-protocol eval table (mention linear-probe vs fine-tune in one line); the contrastive-zoo table (MoCo/SwAV/SimSiam/Barlow Twins — name-drop only).
- **Never cut:** the baby-learns-cat hook, the temperature board example, the "BYOL doesn't collapse — here's why" three forces, and the MAE expert-and-intern asymmetry. Those carry the whole lecture's intuition.

## The one live board example (4 min)
**The InfoNCE temperature effect.** One anchor, scores: positive = 0.9, negatives = 0.2 and −0.1.
- **τ = 1.0:** exps = e^0.9, e^0.2, e^−0.1 ≈ 2.46, 1.22, 0.90. Sum = 4.58. P(positive) = 2.46/4.58 ≈ **0.54**. Loss = −log 0.54 ≈ **0.61**.
- **τ = 0.1:** multiply scores by 10 → logits 9, 2, −1. exps = e^9, e^2, e^−1 ≈ 8103, 7.4, 0.4. Sum ≈ 8111. P(positive) = 8103/8111 ≈ **0.999**. Loss = −log 0.999 ≈ **0.001**.
- Punchline: same similarities, ~600× difference in loss. Small τ amplifies contrast — it forces the model to obsess over closing the gap to *hard negatives*. (Also note: the positive is *in the denominator too* — that's the softmax, not a bug.)

## Closing line
The labels were inside the data all along. Next: the same contrastive trick, but the second "view" of an image is its caption — that's CLIP (L18).
