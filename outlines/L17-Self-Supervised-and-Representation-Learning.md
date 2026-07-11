---
title: "Lecture 17: Self-Supervised Learning and Representation Learning"
status: "Canonical 24-lecture revision — draft outline"
prerequisites: "CNN/transfer learning; Transformer and language-model pretraining"
next: "L18 — Vision-Language Models"
---

# Lecture 17: Self-Supervised Learning and Representation Learning
## Learn useful features from the structure already present in data

## Core story

\[
\boxed{\text{Self-supervision turns raw data into its own training signal.}}
\]

The three recurring recipes are:

\[
\text{match related views}
\quad\vert\quad
\text{reconstruct hidden parts}
\quad\vert\quad
\text{match a slowly changing teacher}.
\]

The goal is not to memorize a list of named methods. It is to ask, for any dataset:

> What should stay the same, what may be hidden, and what makes a representation useful downstream?

**Target:** 80 minutes, approximately 70 slides.  
**One worked loss:** InfoNCE.  
**One board calculation:** temperature changes a contrastive softmax.  
**Deliberately out of scope:** a method-by-method survey of MoCo, SwAV, SimSiam, Barlow Twins, and domain-specific SSL papers.

---

## Part I — Representation learning is the thread

1. **Title**  
   *Self-Supervised Learning: Labels Hidden in Plain Sight*

2. **Recall: a network is also a representation map**
   \[
   x\xrightarrow{f_\theta}h\xrightarrow{g_\phi}\hat y.
   \]
   The head solves today's task; the representation \(h\) is what we hope survives tomorrow's task.

3. **The course’s representation map**
   ~~~text
   CNN feature hierarchies → transfer learning
                                 ↓
   language-model pretraining → self-supervision → CLIP / VLMs
                                 ↓
                           latent-variable generative models
   ~~~

4. **Supervised learning’s bargain**
   \[
   (x_i,y_i)\mapsto \text{learn } f_\theta(x_i)\text{ useful for }y_i.
   \]
   Labels make the objective clear, but expensive labels can bottleneck scale.

5. **The label bottleneck**
   Contrast three pools:
   - curated labeled images;
   - web images, text, audio, sensor streams;
   - data a student or organization can realistically collect.

6. **Central pivot**  
   Instead of receiving a human label, manufacture a prediction target from the input itself.

7. **Three task families**

   | Family | Constructed target | Main intuition |
   |---|---|---|
   | Contrastive / predictive | another related view | keep related things close |
   | Masked reconstruction | a hidden part | infer context and structure |
   | Teacher–student consistency | a teacher representation | stable targets without labels |

8. **Representation learning is not only vision**  
   Reveal the same three families across images, language, audio, and time series. Leave the detailed comparison for the end.

---

## Part II — Contrastive learning: make two views agree

9. **Hook: one object, two views**  
   Show two strong crops of the same dog and a crop of a different dog. Ask: which differences should the representation ignore?

10. **Augmentations define invariance**
    \[
    x\xrightarrow{t_1}v_1,
    \qquad
    x\xrightarrow{t_2}v_2.
    \]
    Crop, colour jitter, blur, and small geometric changes say: *these changes do not change identity*.

11. **A caution about invariance**  
    An augmentation is a modeling assumption. Rotation may be harmless for natural images and disastrous for digit recognition; time reversal may be invalid for causal signals.

12. **SimCLR pipeline**
    ~~~text
    image → two augmentations → shared encoder f → projection head g → contrastive loss
    ~~~
    Distinguish encoder \(h=f(v)\) from contrastive vector \(z=g(h)\).

13. **Why a projection head?**  
    The contrastive objective can discard nuisance information. Let \(g\) absorb the pressure; retain the richer encoder representation \(h\) for downstream tasks.

14. **A contrastive batch**  
    From \(N\) source examples make \(2N\) views. For an anchor, exactly one other view is positive; the remaining \(2N-2\) views act as negatives.

15. **Similarity**
    \[
    s(z_i,z_j)=\frac{z_i^\top z_j}{\lVert z_i\rVert\lVert z_j\rVert}.
    \]
    Normalize first so similarity means angle rather than vector length.

16. **InfoNCE: identify the positive among candidates**
    \[
    \ell_{i,j}=-\log
    \frac{\exp(s(z_i,z_j)/\tau)}
    {\sum_{k\neq i}\exp(s(z_i,z_k)/\tau)}.
    \]
    Frame it as a \((2N-1)\)-way softmax classification problem—not as a mysterious new loss.

17. **Worked batch diagram**  
    Draw four source images, eight augmented views, positive pairs, and the negatives available to one anchor.

18. **Board example: temperature is a contrast amplifier**
    For an anchor, use similarities \(0.9\) (positive), \(0.2\), and \(-0.1\).
    - At \(\tau=1\), \(p(\text{positive})\approx0.54\).
    - At \(\tau=0.1\), \(p(\text{positive})\approx0.999\).

    Punchline: smaller \(\tau\) sharpens the competition, so hard negatives matter more.

19. **What prevents trivial constant features here?**  
    If every input had identical embedding, the positive could not win against negatives. The denominator makes collapse a bad solution.

20. **What SimCLR learns—and what it may throw away**
    - global, augmentation-invariant representations;
    - strong retrieval/classification features;
    - possibly less useful detail for dense spatial tasks if augmentations destroy it.

21. **Evaluation: do not confuse feature quality with head capacity**

    | Evaluation | Encoder | What it tests |
    |---|---|---|
    | Linear probe | frozen | representation quality |
    | Fine-tuning | updated | downstream ceiling |
    | Retrieval | frozen | semantic geometry |
    | Dense-task transfer | often updated | locality and spatial detail |

---

## Part III — Masked reconstruction: infer what is missing

22. **A different question**  
    Contrastive learning asks “which view matches?” Masked modeling asks “what must be true about the world for the hidden part to make sense?”

23. **The masked-autoencoder setup**
    ~~~text
    image patches → hide a large subset → encoder sees visible patches
                  → lightweight decoder reconstructs missing patches
    ~~~

24. **MAE is asymmetric**
    The expensive encoder processes only visible patches; the inexpensive decoder receives encoded visible patches plus mask tokens and is discarded after pretraining.

25. **A simple reconstruction objective**
    \[
    L_{\text{mask}}=\frac{1}{|M|}\sum_{p\in M}
    \lVert x_p-\hat x_p\rVert_2^2,
    \]
    where \(M\) is the set of hidden patches.

26. **Why mask heavily?**  
    With only a few hidden pixels, copying local texture is too easy. Masking a large fraction forces global context and semantics.

27. **Contrastive versus masked objectives**

    | | Contrastive | Masked reconstruction |
    |---|---|---|
    | Supervision | related view | hidden content |
    | Main pressure | invariance | contextual prediction |
    | Common strength | global classification/retrieval | spatial/dense transfer |
    | Design choice | augmentations, negatives, \(\tau\) | mask pattern, reconstruction target |

28. **Bridge to language models**
    - BERT: predict masked tokens from surrounding context.
    - GPT: predict the next token from preceding context.
    - MAE: predict masked image patches from visible patches.

29. **One unifying view**
    \[
    \text{learn }p(\text{hidden or future part}\mid\text{observed context}).
    \]
    Explain that the output spaces differ—tokens, pixels, audio frames—but the self-supervised logic is shared.

---

## Part IV — Teacher–student consistency: learn without negatives

30. **The apparent problem**  
    If two networks are merely trained to agree, they can output the same constant vector for every input. This is *collapse*.

31. **BYOL at a glance**
    ~~~text
    view 1 → student encoder → predictor ──── match ────→ stop-gradient teacher(view 2)
                         ↑                                  │
                         └──── teacher parameters = EMA ────┘
    ~~~

32. **The three asymmetries to teach**
    - a predictor exists only on the student path;
    - the teacher output is stop-gradient;
    - teacher weights are an exponential moving average of student weights.

33. **The EMA update**
    \[
    \theta_{\text{teacher}}
    \leftarrow
    m\theta_{\text{teacher}}+(1-m)\theta_{\text{student}}.
    \]
    A slowly moving target gives the learner something stable to chase.

34. **Why this does not collapse: a careful answer**  
    Do not claim the EMA alone solves collapse. In the practical recipe, stop-gradient, predictor asymmetry, normalization, and the slowly moving target work together to prevent the two branches from taking the trivial shortcut.

35. **DINO as the scaled representation recipe**
    Student and teacher predict distributions over features from differently cropped views. Mention emergent semantic regions and strong frozen features, without deriving every centering/sharpening detail.

36. **One-slide landscape, not a zoo**

    | If students encounter… | Place it here |
    |---|---|
    | MoCo, SwAV, SimSiam, Barlow Twins | variations on contrast or anti-collapse design |
    | BYOL, DINO | teacher–student consistency |
    | MAE, masked language modeling | reconstruct/predict hidden content |

---

## Part V — The same question across data modalities

37. **Cross-modal comparison slide**

    | Data | Contrastive / predictive | Masked reconstruction | Consistency |
    |---|---|---|---|
    | Images | augmented views; image–caption pairs | masked patches | teacher/student crops |
    | Text | adjacent/contextual spans; sentence pairs | masked tokens; next token | teacher/student encodings |
    | Audio / time series | augmented signals; nearby windows; future state | masked spans / channels | noisy-view agreement |

38. **The transfer question**  
    Before choosing SSL, state the desired invariance and the downstream task. “Ignore colour” might be ideal for species recognition and unacceptable for medical imaging.

39. **Exit ticket**  
    For a wearable-sensor dataset with no labels, have students propose:
    - two valid augmentations;
    - one invalid augmentation;
    - one masked-prediction target;
    - one downstream evaluation protocol.

40. **Closing bridge to CLIP**
    \[
    \text{two augmented views of an image}
    \quad\longrightarrow\quad
    \text{an image and its caption}.
    \]
    The same contrastive geometry becomes a shared image–text representation space in L18.

---

---

## Part VI — Detailed support slides and applications

## Slide 41 — Pretraining versus downstream use

Pretraining:

\[
x\longrightarrow h=f_\theta(x)
\]

uses an objective built from raw data. Downstream use can freeze \(f_\theta\), add a small head, or fine-tune all of it.

The representation is the transferable artifact—not the pretext-task loss itself.

---

## Slide 42 — A three-stage experimental protocol

~~~text
unlabeled corpus
      ↓ self-supervised pretraining
frozen or adapted encoder
      ↓ small labeled downstream set
linear probe / fine-tune / retrieval evaluation
~~~

Ask students where expensive labels first enter the pipeline: only at the final evaluation or adaptation stage.

---

## Slide 43 — Contrastive learning uses each pair twice

For augmented views \(i\) and \(j\), calculate:

\[
\ell_{i,j}
\qquad\text{and}\qquad
\ell_{j,i}.
\]

Each view is an anchor once. The standard batch loss averages the two directions across all positive pairs.

---

## Slide 44 — Similarity-matrix view of SimCLR

For \(2N\) normalized embeddings:

\[
S_{ij}=\frac{z_i^\top z_j}{\tau}.
\]

The diagonal is ignored; each row has one desired positive entry. This is the same “correct match in a batch” geometry that later powers CLIP.

---

## Slide 45 — Why large contrastive batches helped

More examples in a batch give an anchor more negative candidates:

\[
\text{one positive}\quad\text{versus}\quad 2N-2\text{ negatives}.
\]

Clarify the point: larger batches were valuable for the *contrastive classification problem*, not just for conventional hardware throughput.

---

## Slide 46 — False negatives

Two pictures can show different dogs of the same breed, but SimCLR treats them as negatives unless it knows otherwise.

This is a genuine mismatch between the constructed task and semantic similarity. It motivates supervised contrastive objectives and clustering-based variants, but does not invalidate the basic recipe.

---

## Slide 47 — Augmentations are the hidden labels

| Augmentation | Says the representation should often ignore… |
|---|---|
| random crop | exact location and framing |
| colour jitter | global colour statistics |
| blur | fine texture |
| weak geometric transform | small pose changes |

Ask: “Would each choice be valid for satellite imagery, histopathology, or digit recognition?”

---

## Slide 48 — A bad positive pair

Show a crop that removes the object entirely or a transformation that reverses temporal causality.

The lesson:

\[
\text{wrong invariance}\Rightarrow\text{wrong representation}.
\]

Self-supervision is not free of domain expertise; it moves expertise into objective design.

---

## Slide 49 — Linear probe as a controlled experiment

Freeze \(f_\theta\), train only:

\[
p(y\mid x)=\operatorname{softmax}(Wh+b).
\]

If the linear classifier works, class information is already arranged simply in the representation. A powerful fine-tuned head cannot make that same claim.

---

## Slide 50 — Nearest-neighbour retrieval as a second test

Embed a query image and rank database examples by cosine similarity.

~~~text
query → encoder → vector
                    ↓ dot product
             database representation vectors → nearest neighbours
~~~

Retrieval tests geometry directly and needs no new classifier head.

---

## Slide 51 — MAE input shapes

For a \(224\times224\) image with \(16\times16\) patches:

\[
N=(224/16)^2=196.
\]

With 75% masking, the heavy encoder receives only \(49\) visible patch tokens. This is both a learning choice and a pretraining-efficiency choice.

---

## Slide 52 — Position survives masking

Visible patches retain positional embeddings before entering the encoder. At the decoder, learned mask tokens are inserted at the missing locations.

Without positional information:

\[
\text{visible patches}=\text{an unordered bag}.
\]

The model could not reliably reconstruct spatial structure.

---

## Slide 53 — Why MAE can mask much more than BERT

Text tokens are highly discrete: hiding too much can make a sentence ambiguous. Natural images contain spatial redundancy, and visible patches still provide broad semantic clues.

Do not claim a universal masking percentage; emphasize that mask ratio is a data-modality design choice.

---

## Slide 54 — What is the reconstruction target?

Options include:

- raw pixels;
- normalized patches;
- discrete visual tokens;
- latent features from another encoder.

Different targets ask the model to retain different detail. Pixel reconstruction can emphasize texture; feature targets can emphasize semantics.

---

## Slide 55 — The collapse thought experiment

Suppose student and teacher always output:

\[
[0,0,\ldots,0].
\]

Their matching loss is perfect, yet the representation says nothing about the input. Agreement alone is insufficient.

---

## Slide 56 — A schematic BYOL loss

With normalized student prediction \(p_\theta(v_1)\) and stop-gradient teacher target \(z_\xi(v_2)\):

\[
L_{\mathrm{BYOL}}
=
\left\|
\frac{p_\theta(v_1)}{\|p_\theta(v_1)\|}
-
\operatorname{sg}\left(
\frac{z_\xi(v_2)}{\|z_\xi(v_2)\|}
\right)
\right\|_2^2.
\]

The loss itself is simple; the asymmetric learning dynamics are the important idea.

---

## Slide 57 — EMA worked update

If \(m=0.99\), teacher weight is \(0.40\), and student weight becomes \(0.70\):

\[
0.99(0.40)+0.01(0.70)=0.403.
\]

The teacher changes slowly. It is not independently optimized on the current batch.

---

## Slide 58 — DINO and multi-crop views

Use two global views and several smaller local views. The teacher sees global context; the student must make local content agree with a global semantic target.

This makes “same object at different scales” part of the constructed supervision signal.

---

## Slide 59 — SSL family comparison, expanded

| Question | Contrastive | Masked | Teacher–student |
|---|---|---|---|
| What is positive? | paired view | missing content | teacher prediction |
| What prevents trivial solution? | competing negatives | reconstruction target | asymmetry + moving target |
| Primary design lever | augmentation | mask / target | teacher update / views |
| Natural bridge | CLIP | BERT/GPT/MAE | DINO-style features |

---

## Slide 60 — Text is already self-supervised

Autoregressive language modeling:

\[
p(x_1,\ldots,x_T)=\prod_{t=1}^{T}p(x_t\mid x_{<t}).
\]

Every document supplies thousands of next-token targets. Language-model pretraining is not separate from SSL; it is the most familiar instance of it.

---

## Slide 61 — Masked language modeling

~~~text
The cat sat on the [MASK].
             ↓
           mat
~~~

Bidirectional context predicts a hidden token. Contrast it with GPT’s strictly left-to-right next-token objective and connect both to MAE’s hidden-patch task.

---

## Slide 62 — Audio SSL example

For speech or music:

- mask contiguous time spans and predict their representations;
- treat two augmentations of the same clip as related;
- avoid augmentations that erase the label needed downstream.

Use this slide to demonstrate transfer of the *objective family*, not to introduce another architecture.

---

## Slide 63 — Time-series SSL example

For sensor sequences, plausible tasks include:

\[
\text{predict a future window}
\qquad\text{or}\qquad
\text{reconstruct masked channels}.
\]

Ask whether time shuffling, amplitude scaling, or time reversal preserves the event of interest.

---

## Slide 64 — Image–text pairs are a special positive pair

In CLIP:

\[
(\text{image},\text{caption})
\]

replaces two augmented views of one image. The batch similarity matrix and contrastive softmax stay recognizably the same.

---

## Slide 65 — Representation-learning design checklist

Before training, answer:

1. What downstream property should remain invariant?
2. What information must remain accessible?
3. What pretext target is available without labels?
4. How will representation quality be evaluated?
5. What harms could a shortcut or biased corpus create?

---

## Slide 66 — Classroom design exercise

Dataset: unlabeled chest X-rays, later used for pathology detection.

Small groups decide:

- which image augmentations are defensible;
- whether colour-jitter intuition transfers;
- whether a masked or contrastive objective is safer;
- which evaluation protocol needs labelled examples.

---

## Slide 67 — Common misconceptions

| Misconception | Correction |
|---|---|
| “SSL has no labels.” | It has targets constructed from data. |
| “Contrastive means semantic negatives.” | Negatives are often only batch alternatives. |
| “BYOL works because of EMA.” | It uses several asymmetries together. |
| “A good reconstruction guarantees a good classifier.” | The target determines what features are retained. |

---

## Slide 68 — One mental model

\[
\boxed{
\text{Choose a transformation or hidden part that makes the desired knowledge necessary.}
}
\]

Everything else in SSL is an implementation of this decision.

---

## Slide 69 — Retrieval questions

1. Why can a projection head improve downstream features even when it is discarded?
2. What does a false negative do to InfoNCE?
3. Why can a constant teacher/student representation minimize a naive consistency loss?
4. Which SSL family most naturally resembles next-token prediction?

---

## Slide 70 — Next lecture

\[
\text{image views}
\;\longrightarrow\;
\text{image–caption pairs}
\;\longrightarrow\;
\text{shared multimodal representation space}.
\]

L18 turns the contrastive geometry from this lecture into zero-shot classification and visual-language interaction.

---

## Instructor pacing notes

- **Must teach deeply:** Slides 10–20, especially the softmax interpretation of InfoNCE; Slides 23–28; Slides 30–34.
- **Keep high-level:** DINO implementation details, named-method landscape, audio/time-series examples.
- **If short on time:** retain SimCLR + InfoNCE, MAE asymmetry, the three BYOL anti-collapse ingredients, and the cross-modal table. Cut the extended evaluation discussion and named-method landscape.
- **Preparation-light demo:** use a pretrained image encoder to compare nearest neighbours before and after a linear probe. Do not train SimCLR live.

## Student takeaways

1. Self-supervision does not mean “no targets”; it means the data provides the targets.
2. Contrastive, masked, and teacher–student methods encode different assumptions about what makes two inputs related.
3. Representation quality must be evaluated separately from a powerful downstream head.
4. GPT-style next-token prediction, MAE, and CLIP belong to one broader representation-learning story.
