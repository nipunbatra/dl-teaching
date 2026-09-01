# Lecture 7 figure provenance

## `train_val.png`

- Course-owned schematic used to introduce the canonical training-loss-down,
  validation-loss-down-then-up pattern; it is not an empirical run.
- Inspired by the Google Machine Learning Crash Course treatment of
  generalization and overfitting (CC BY 4.0).

## High-resolution augmentation and mixing photographs

- `sources/augmentation_cat_imagen_v2.png` and
  `sources/mixup_cat_dog_imagen.png` are course-owned teaching assets generated
  with Codex's built-in image-generation tool on 27–28 August 2026. They are not
  samples from the CIFAR-10 experiment reported later in the lecture.
- Source SHA-256 values:
  - augmentation cat v2: `1f8eee97b10de4d983926165d3a4537136be63620f5a31342fd7b7ee9ca17842`
  - cat/dog diptych: `5e779965e42bd6491cce2fc57ec1ac04229da629a2dd42b514c53580659ddfd2`
- Final prompt for the v2 augmentation source: a photorealistic landscape
  teaching photograph of one full-body tabby cat walking left across a garden
  patio, with a blue planter clearly on the left and a yellow chair clearly on
  the right so crop and flip operations remain obvious at thumbnail size;
  soft bright daylight, detailed fur, generous margins, and no people, other
  animals, text, logos, watermark, collage, motion blur, or distorted anatomy.
- Final prompt for the mixing sources: a clean two-panel, high-resolution
  studio diptych with one seated ginger tabby cat on the left and one seated
  golden retriever on the right, equal panels, identical pale neutral
  background, matched camera height, scale, and lighting, with no text, logos,
  watermarks, decorative graphics, frames, or overlapping animals.

## Mixup domain-audit visuals

- `sources/mixup_audio_superposition_imagen.png` and
  `sources/mixup_driving_turns_imagen.png` are course-owned teaching assets
  generated with Codex's built-in image-generation tool on 28 August 2026.
  Source SHA-256 values:
  - audio spectrogram triptych:
    `1488922dc20e0691d68616be01e1805ae56e14876dbe1ffbd3870e040504509d`
  - matched driving-view triptych:
    `29955a03180a8d05570c80aeaa495cca1a6d63d907b342644d1ef63837179639`
- Final audio prompt: a high-resolution, text-free scientific triptych for a
  university lecture, with isolated birdsong chirps at left, steady broadband
  rain at right, and both patterns visibly superposed in the centre; consistent
  dark-navy/teal spectrogram styling with warm energy traces, generous margins,
  and no axes, labels, logos, watermarks, or fake UI. The slide explicitly calls
  these schematic spectrograms and says waveform mixing happens before display.
- Final driving prompt: a high-resolution, text-free photorealistic triptych
  from a matched dashboard-camera viewpoint, with a mountain road curving left
  and right in the outer panels and an ambiguous double exposure in the centre;
  neutral daylight, consistent horizon and camera height, and no cars, people,
  signs, text, logos, or watermarks.
- `mixup_driving_exact_blend.png` replaces the generated centre panel with an
  exact 50/50 ImageMagick pixel average of equal crops from the generated left
  and right source views. `mixup_driving_exact_blend_wide.png` is a lossless
  teaching crop of that exact triptych. SHA-256 values:
  - exact triptych:
    `69f2c4266a77b3dc430f930dcdb46dea19b8eb1a76cbdbcd68460a132648d9ff`
  - displayed wide crop:
    `4b28de24b626e0ec832c37368fb995ecb4b38f1a2437550cef6216d03c359361`

## Task-conditioned target-contract photographs

- `sources/label_contract_cyclist_imagen.png` and
  `sources/label_contract_runner_imagen.png` are course-owned teaching
  assets generated with Codex's built-in image-generation tool on 28 August
  2026. Exact boxes, keypoints, transforms, equations, and labels are added in
  Typst; the photographs themselves are illustrative rather than measured
  data.
- Source SHA-256 values:
  - cyclist: `ef800974e74af536bb295da61cb95ca5c32b69722d0eca1f519fb7f19c8bf727`
  - runner: `b93802605cac60d73e84c03aa5e729f41dd1ae06b27976ea0c60b3e9383b3bc3`
- Final cyclist prompt: a high-resolution photorealistic side-profile scene
  with exactly one adult cyclist travelling left-to-right on a quiet urban
  cycle path, the complete bicycle and rider visible with generous margins,
  a clean wall and greenery, natural daylight, and no text, logos,
  watermarks, extra subjects, crops, or blur.
- Final runner prompt: a high-resolution photorealistic full-body adult runner
  in an asymmetric warm-up pose on an uncluttered track, front-facing with one
  arm raised and one knee bent, every joint visible with generous margins, and
  no text, logos, watermarks, extra people, cropped limbs, or duplicate limbs.
## Dropout robustness analogy

- `sources/dropout_random_road_closures_imagen.png` is a course-owned teaching
  asset generated with Codex's built-in image-generation tool on 28 August
  2026. SHA-256:
  `d5b6553e5964be7318a93e4799d7d92fea28a785cd0188e0ce7141e809dc3220`.
- Final prompt: a high-resolution, text-free scientific three-panel illustration
  of the same top-down road network, with a brittle single route blocked in the
  first panel, several randomly closed roads and successful alternate routes in
  the second, and every road open with the learned routes visible in the third;
  deep teal roads, blue/green paths, orange destination, red barriers, white
  background, consistent geometry and scale, and no words, numbers, UI, map
  labels, people, cars, logos, or watermarks.
- `dropout_route_brittle.png`, `dropout_route_practice.png`, and
  `dropout_route_evaluation.png` are lossless panel crops from the same source,
  enlarged on separate slides so each beat of the analogy can be taught before
  mapping it back to dropout. Their SHA-256 values are, respectively,
  `239ffe34d3bd083b9ab7919025166ce60ad03e0ce0ad8026b20a962d6fa16488`,
  `b350080c8a9042b7a46f9264ef7caf061347d8c1c0fc4011ba54f2e740247783`, and
  `a8823aab3b62a2985c2ccb000595745c99fd5cdafc99884338711e293a07fce3`.

## Generated augmentation panels

- Generated by `lecture7/diagrams/l7_practical_figs.py` from the fixed
  high-resolution cat source above.
- Rebuild from the repository root with
  `uv run --with matplotlib --with pillow --with numpy --with 'torch>=2.2' --with 'torchvision>=0.17' python lecture7/diagrams/l7_practical_figs.py`.
- Every panel is derived from the same pixels. The displayed individual panels
  use seeded outputs from the exact TorchVision v2 calls printed on the slides:
  a resized crop, horizontal flip, colour jitter, Gaussian blur, and rotation.
- The 2×4 batch-audit panel uses eight seeded draws of shifts, flips,
  brightness, contrast, and colour changes so the visual is reproducible.

## Mixup and CutMix teaching panels

- Generated by the same script from the fixed cat/dog diptych above.
- MixUp uses `lambda = 0.70`. CutMix replaces a 450×450 patch of a 900×900
  image, so the pasted-area fraction is exactly `0.25` and the displayed label
  weights are `0.75 / 0.25`.
- The same script writes the text-free 900×900 operands used as larger native
  slide panels: `mixup_cat_hd.png`, `mixup_70_hd.png`, `mixup_dog_hd.png`, and
  `cutmix_25_hd.png`.

## RandAugment operation-pool examples

- `randaugment_original_hd.png`, `randaugment_rotate_color_hd.png`,
  `randaugment_solarize_translate_hd.png`, and
  `randaugment_posterize_contrast_hd.png` are deterministic 900×900 outputs of
  `lecture7/diagrams/l7_practical_figs.py` from the same high-resolution cat
  source.
- They illustrate sequential composition of operations present in the
  RandAugment pool. They are intentionally not presented as samples from one
  particular TorchVision magnitude index; the slide keeps that distinction
  explicit.

## `cifar_regularization_curves.png` / `.svg` and `cifar_regularization_summary.png` / `.svg`

- Generated by `lecture7/diagrams/cifar_regularization_experiment.py`; these
  are measured PyTorch results, not schematic curves.
- Fixed protocol: 2,000 stratified training examples, 5,000 validation
  examples, the untouched 10,000-example CIFAR-10 test split, one fixed seed,
  one initialization, one minibatch order, the same 666,538-parameter CNN,
  and 35 epochs for every condition.
- Each run changes one ingredient only: no regularizer, AdamW decay, crop plus
  flip, dropout, or label smoothing. The candidate suite and rules were
  predeclared; within each fixed condition, validation cross-entropy selected
  the checkpoint before one reporting-only test evaluation.
- Machine-readable configuration, split indices, epoch metrics, and summary
  live in `lecture7/evidence/`. The single-seed results are descriptive
  teaching evidence, not a benchmark or an uncertainty estimate.

Dataset reference: Alex Krizhevsky, *Learning Multiple Layers of Features from
Tiny Images*, 2009; CIFAR-10.

## STT-AI text and audio augmentation examples

- `sources/stt_specaugment_core.png` is a lossless crop of
  `/Users/nipun/git/stt-ai-teaching/slides/images/week05/specaugment_example.png`
  (5504×3072), reused from Nipun Batra's STT-AI Week 5 augmentation lecture.
  The crop retains the original, time-masked, and frequency-masked
  spectrogram panels while removing the redundant source-slide heading and
  footer. The STT-AI repository records the source diagram as Gemini-generated.
  SHA-256: `1500ae15ede4c36cb2f07c2b6993f0dbe830c2a41721f0878c0429474ebdc705`.
- `sources/stt_ner_context_core.png` is a lossless crop of
  `/Users/nipun/git/stt-ai-teaching/slides/images/week05/ner_augmentation.png`
  (5504×3072), reused from the same lecture. The crop retains the paired
  original/augmented sentences and entity labels while removing the redundant
  source-slide heading and warning footer. The STT-AI repository records the
  source diagram as Gemini-generated. SHA-256:
  `bcba34a8ed56f0c0c9faaf8b826ba50f8c299809f53b989bc9c2b11b8470ce21`.
- The sentiment examples and sound-event timestamp shift are redrawn natively
  in Typst from the teaching examples in
  `/Users/nipun/git/stt-ai-teaching/slides/week05-data-augmentation-lecture.md`
  and `/Users/nipun/git/stt-ai-teaching/slides/week03-data-labeling-lecture.md`.
  Their purpose is conceptual: they illustrate the task-specific target rule,
  not measured model performance.
