# Lecture 18 · Vision-Language Models · Teaching Guide
*First-time-instructor companion. Audience: undergrads, one ML course.*

## The spine (say this in 2 sentences)
Vision-language models are three ideas stacked on the Transformer you already know: ViT (chop an image into patches and treat them as tokens), CLIP (train an image encoder + text encoder so matching image-caption pairs are close in one shared space), and LLaVA (feed CLIP's image features into an LLM as if they were extra tokens). The headline payoff: once images and text live in *one* embedding space, *naming* a class in words is enough to recognize it — a caption becomes a classifier.

## Where it sits
This lecture is *literally the union of the previous three*: the Transformer (L13) + the contrastive trick (L17) + the LLM (L15) compose here. It follows CNNs (L7–L9, the vision priors ViT throws away) and self-supervised learning (L17, whose InfoNCE loss CLIP reuses across modalities). It closes the "perception" arc and hands off to generation (L19 onward). **Landscape/literacy** lecture: the goal is the *composition story* and the CLIP zero-shot mechanism; the patch-embedding and projection arithmetic are nice-to-have worked examples.

## 80-minute plan
- **ViT** ⭐ ~18 min — the "describe a photo over the phone in chunks" analogy, patchify → flatten → linear-project → prepend CLS + positions, real numbers (196 patches + CLS = 197 tokens), ViT-vs-CNN inductive-bias table. The 30-line PyTorch slide is ⭐⭐⭐.
- **CLIP** ⭐ ~24 min — dual encoder, the N×N similarity matrix (diagonal = matches), row+column InfoNCE, and zero-shot. **Do live on the board:** the CLIP zero-shot dot-product example (below).
- **LLaVA** ⭐⭐ ~16 min — the translator analogy, the linear/MLP projection with shapes (1024 → 4096), two-stage training (align, then instruction-tune), "<0.15% extra weights." Flamingo cross-attention is ⭐⭐⭐.
- **Native vs bolt-on** ⭐⭐ ~10 min — adopt-an-adult-dog vs raise-a-puppy; the trade-off table; frontier leans native, open-source is bolt-on.
- **Hallucination** ⭐⭐ ~8 min — priors fighting evidence; the cat/dog and the vase examples; not solved.
- **Buffer / benchmark table** ~4 min.

## Teach it like this (hook → sequence → payoff)
**Hook:** the "find the dog with the red collar" pop quiz. An ImageNet classifier outputs `dog`; a detector outputs boxes; neither *sees* "red collar." A VLM does — because **its output space is language, so any concept you can describe is reachable.** **Sequence:** first, how does a Transformer even see? (ViT: patches as tokens) → put an image encoder and a text encoder in *one* space and train so matching pairs are close (CLIP) → that shared space lets the text encoder *write the classifier on the fly* (zero-shot) → bolt CLIP's features onto an LLM through a small projection and it can *converse* about images (LLaVA) → frontier models grow up multimodal from scratch (native). **Payoff:** "Once images and text share one embedding space, a caption is a classifier."

## Heads-up for YOU (subtle points to get right)
1. **CLIP's text encoder is *not* run during normal classification inference.** You embed your class prompts ("a photo of a {label}") *once*, cache those vectors, and at inference run only the *image* encoder and take dot products against the cache. The caption embeddings *are* the classifier weights — that's the whole zero-shot trick. Saying "it runs the text encoder on every image" misses the point.
2. **ViTs are not "better than CNNs" — they trade priors for data hunger.** ViTs have *fewer inductive biases* (no built-in locality/translation equivariance), so they need *more data* to win. On small data, CNNs win (priors help). On massive data (JFT-300M), ViTs win (capacity to learn better priors). Present it as a crossover, not a verdict.
3. **VLM hallucination is *language priors overriding visual evidence*, not just "the vision model failed."** The LLM is biased autocomplete: when the visual signal is weak/ambiguous, it defaults to what's statistically likely in its *text* training data ("desks have coffee cups"). It's a tug-of-war between prior and evidence — strong evidence wins, weak evidence loses to the prior.
4. **CLIP's softmax is over the *prompts you provide*, so vanilla CLIP is single-label by construction.** Give it [cat, dog, car] and it forces probabilities to sum to 1; an image with *both* a cat and a dog gets split, not "both." Don't oversell CLIP as a multi-label detector.
5. **Why LLaVA needs a projection at all (and why an MLP).** CLIP outputs ~1024-d vectors; the LLM expects ~4096-d token embeddings — different vector spaces. A *linear* map works (LLaVA-1) but an *MLP* (LLaVA-1.5) gives a richer alignment. The base LLM is frozen in stage 1 so the projection learns to "speak the LLM's language" without disturbing it.
6. **CLIP's prompt sensitivity is distribution-matching, not a hack.** "a photo of a {label}" beats the bare word by ~5% on ImageNet because it matches how images were captioned on the web. Frame prompt engineering as aligning to the training distribution, not magic.

## Where students stumble (and the fix)
1. **"How can CLIP classify with no classifier head?"** Fix: the board example. In a shared space, the text vector for "a photo of a dog" *is* a row of classifier weights; nearest caption wins. There is no dog logit anywhere in the model.
2. **Thinking ViT needs convolutions for spatial structure.** Fix: positional embeddings carry spatial order. Without them, patches are an unordered bag. The patchify step itself is just a strided conv with kernel = stride = patch size.
3. **Confusing CLIP (contrastive, dual-encoder) with LLaVA (generative, LLM-backed).** Fix: CLIP *scores* image-text match; LLaVA *generates* text about an image. LLaVA *uses* CLIP as its eyes. Different jobs.
4. **Assuming native multimodal is strictly superior.** Fix: native is deeper but requires pretraining a huge model from scratch on interleaved data — infeasible for almost everyone. Bolt-on (LLaVA) is the only practical open-source route. "Frontier native, everyone else bolt-on."
5. **Treating hallucination as a rare bug.** Fix: show it's structural — the LLM has seen "rooms have vases" millions of times in text. Mitigations (stronger vision encoder, more VQA data, vision-specific RLHF) reduce but don't eliminate it.

## If a student asks…
- **"Why does CLIP use two separate encoders instead of one Transformer over image+text?"** Efficiency at retrieval/classification time. With dual encoders you precompute all image and text embeddings once and do cheap dot products. A single cross-attention model would have to be re-run for *every* image-text pair — quadratically expensive.
- **"In zero-shot, what happens if the image has two of the listed classes (cat AND dog)?"** (HARD) The softmax over prompts forces a single distribution summing to 1, so it splits probability or picks the more salient one. Standard CLIP is not a true multi-label/detector — you'd need a different head or open-vocab detection setup.
- **"Why project CLIP features into the LLM — why not feed them raw?"** Dimension and space mismatch: CLIP is ~1024-d, the LLM expects ~4096-d token embeddings, in a *different* learned space. The projection rotates/scales image vectors into the LLM's word space; the two spaces already encode similar concepts, so the map is learnable from modest data.
- **"Why is the LLM frozen in LLaVA stage 1?"** To learn *only* the projection — align image features into the LLM's existing embedding space without corrupting the LLM's language abilities. Stage 2 then unfreezes for instruction tuning on (image, question, answer) triples.
- **"How does ViT know patch order without convolutions?"** Positional embeddings, added after patch projection. They're learned (or fixed) and let attention recover spatial layout.
- **"Why do VLMs hallucinate objects that aren't there?"** Strong text priors override weak visual evidence. When the image is blurry/ambiguous, the model fills in statistically-likely details. Fixes: stronger vision encoder, more grounded VQA data, vision-specific RLHF/DPO. Reduced in recent models, not solved.

## If you're short on time
- **Cut:** the ⭐⭐⭐ 30-line ViT PyTorch slide and the Flamingo/Perceiver-Resampler slide (one sentence: "an alternative bridge via cross-attention"); the patch-embedding and projection worked-numerics (the *shapes* matter, the arithmetic doesn't).
- **Never cut:** the "red collar" hook, the CLIP zero-shot board example, the "caption = classifier" framing, and the hallucination = priors-vs-evidence point. Those four are what a student should remember in a year.

## The one live board example (3 min)
**CLIP zero-shot via dot products** — text vectors *are* the classifier weights.
Image embedding (unit) `i = [0.8, 0.6, 0]`, temperature τ = 0.1. Three caption embeddings:
- "a photo of a **dog**" `t = [0.6, 0.8, 0]` → `i·t = 0.48 + 0.48 = 0.96` → logit 9.6
- "a photo of a **cat**" `t = [1, 0, 0]` → `i·t = 0.80` → logit 8.0
- "a photo of a **car**" `t = [0, 0, 1]` → `i·t = 0.00` → logit 0
Softmax ∝ [e^9.6, e^8.0, e^0] ≈ [14765, 2981, 1] → ≈ **[0.83, 0.17, 0.0001]**. `argmax` = **dog**, 83% confident.
Punchline: no dog logit, no softmax head, no training — the text encoder *wrote* a 3-way classifier on the fly. Want 100 classes? Embed 100 captions. Ever trained on cats? No.

## Closing line
Once images and text share one embedding space, a caption is a classifier. Next: generation begins — we stop encoding inputs to points and start encoding them to *distributions* (VAEs, L19).
