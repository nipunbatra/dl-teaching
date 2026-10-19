# Codex build prompt — Generative Modelling II

## From Denoising to Text-to-Image Generation

**Usage:** Open the intended course repository in Codex and give it this entire file with the instruction: “Implement this lecture completely, including its demonstrations, notes, notebooks and both PDF exports.” If no course checkout is available, the execution contract defines a standalone fallback. Read the complete file before choosing an implementation approach.

This is a self-contained implementation brief for **Nipun Batra's ES 667 Deep Learning course, IIT Gandhinagar**. Every required scene, mathematical detour, numerical fixture, application source, demonstration and delivery requirement is included here. No earlier chat or companion prompt is needed to build this lecture.

| Item | Specification |
|---|---|
| Suggested output folder | `generative-modelling-ii/` |
| Main content | 60 logical scenes; 158 planned reveal states |
| Teaching routes | Approximately 120 minutes in full; 90 minutes compressed |
| Classroom experience | Applications first; substantial images; progressively revealed diagrams, equations and code |
| Primary implementation | Interactive static HTML following the existing course runtime and visual conventions |
| Required companion outputs | Instructor notes, completed-scene PDF, all-builds PDF, runnable PyTorch material, recap, asset/source manifest and QA record |

The numbered storyboard controls the teaching sequence; its mathematical specification controls equations and code. The execution contract controls integration, navigation, sources and exports. Improve layout through documented additional builds when necessary. Preserve the mathematical steps, concrete examples and mandatory topics. Keep optional appendix material separately identified and outside the core route timings.

---

## Lecture II implementation brief — From noise to images, then language

### 1. The lecture you are building

Build **Generative Modelling II: From Denoising to Text-to-Image Generation** for ES667, IIT Gandhinagar. Produce the actual lecture and its working demonstrations; this document is the implementation specification. The first lecture has already introduced generative distributions, autoregression, latent-variable models, autoencoders, KL divergence, the ELBO, the reparameterization trick, VAEs, and GANs. Students also know encoder and decoder transformers, causal versus full attention, cross-attention, ViT, CLIP, three VLM architectures, and image detection/segmentation.

The central question is **“How can learning to remove noise let us create an image from noise?”** Start with compelling applications and actual images. Revisit those same examples after the mathematical and architectural explanations. Teach historical DDPM as one coherent, executable formulation; explicitly place alternative parameterizations, samplers, and flow matching in context without claiming every modern image generator uses this exact algorithm.

By the end, a student must be able to:

1. Distinguish a stochastic forward corruption from a learned reverse distribution.
2. Compute one noisy training input and identify every available training target.
3. Explain why one network receives a timestep and can train at randomly chosen noise levels.
4. Relate noise-prediction MSE to the earlier ELBO, with the weighting qualification intact.
5. Execute one DDPM reverse update and distinguish it from a clean-image estimate.
6. Trace pixel diffusion, latent diffusion, U-Net/DiT denoisers, text cross-attention, and classifier-free guidance.
7. Explain a discrete masked-text extension without treating token identifiers as real-valued pixels.
8. Separate genuine model outputs, measured inputs, analytical toy demonstrations, and illustrative mockups.

#### Pacing and completeness

Create **60 logical scenes, with 158 planned presentation states/builds**. A build is a carefully chosen reveal within a scene, not a new dense slide. Small deviations in build count are acceptable when splitting an equation improves legibility; preserve the content and all numbered logical scenes. Provide an all-builds PDF and a final-state-per-scene reading PDF, adapting the course exporter as specified in the execution contract when necessary.

The complete teaching route is approximately **120 minutes**, including short questions and demonstrations. Applications, probability intuition, derivations, code, architectures, and language diffusion need this space for a slow first exposure.

| Section | Scenes | Planned time | Main purpose |
|---|---:|---:|---|
| Applications and motivation | 01–10 | 12 min | Establish what students want to understand |
| Corruption and probability | 11–20 | 20 min | Derive the noisy input and explain uncertainty |
| Learning the reverse model | 21–31 | 23 min | Training tuple, code, and ELBO connection |
| Generation | 32–39 | 21 min | A numerical reverse step and the sampling loop |
| Image architectures | 40–46 | 15 min | U-Net, DiT, and latent diffusion |
| Text conditioning | 47–54 | 17 min | Q/K/V, dropout, guidance, conditional tasks |
| Evidence, text diffusion, recap | 55–60 | 12 min | Evaluate the examples and extend the principle |

Also include a **90-minute presenter route**, preserving the full content in the deck. Allocate 8/15/18/16/11/14/8 minutes to the seven sections above. In this route, show the application gallery as a quick visual tour, collapse the forward expansion to its final result after the scalar example, summarize the ELBO weighting after displaying its derivation, move the additional attention arithmetic and architecture details to reading mode, and retain a short masked-text example. Label this route as a compressed overview. Do not describe it as allowing the same depth or pause time as the full route. Keep all required derivations, images, numerical examples, code, and notes available.

### 2. Visual examples and assets that recur

Maintain four recognizable running examples throughout the lecture:

- **An actual MNIST digit**: clean image, forward corruption, training target, predicted noise, clean-image estimate, final samples. Reuse the same selected digit and seeds when making the pedagogical comparison.
- **Eight Gaussian clusters in two dimensions**: distribution-level forward diffusion and a clearly labelled analytical-oracle reverse demonstration. This has calculable density and score; it is not evidence of a trained neural model.
- **One sourced natural-image generation example**: keep its prompt and image identity fixed when discussing cross-attention and guidance. Select a project or model with verifiable prompt/sample provenance. Use “a golden dog on green grass” only as the hand-designed attention example if genuine generated assets for this exact prompt are unavailable.
- **One masked sentence**: demonstrate why the corruption and prediction space changes for language. Keep this separate from the Gaussian equations used for images.

The opening needs **8–10 image-led application scenes**. Across the lecture, target at least **35 meaningful visual panels** and at least **20 actual image examples or image variants**. Count panels by what they explain, not arbitrary stock images. A side-by-side comparison of the same image under two conditions is more useful than unrelated artwork. Prefer official project images, figures with attributable provenance, real public datasets, and reproducibly generated outputs. Give each panel a short source caption; record full provenance in an asset manifest.

Asset records must state: local filename; original URL; paper/project/model; figure number or sample identity when available; license or reuse information as available; whether the file is reproduced, cropped, redrawn, or produced locally; prompt/seed/checkpoint/sampler when locally generated; and the exact instructional claim. Preserve axis labels and comparison labels when cropping. Never use a stock photograph as if a diffusion model generated it. If only an illustrative image can be produced, label it “Illustration of the task,” not “Model output.”

### 3. Scene-by-scene storyboard

Each scene needs a visible question or short claim, a dominant visual, the specified build sequence, speaker notes explaining the transition, and any source footer. Keep familiar visual objects in place while the highlighted operation changes.

#### A. Start with things students want to do

**01 — “Can we create a new example?” [1 build]**

Use a large, sourced generated-image mosaic rather than a text agenda. Show three short task labels: “Generate,” “Complete,” “Control.” Ask students to look for what each system was given as input. Title and subtitle should take little space. Speaker notes introduce the learning question, not a history of diffusion. Source: one or more official DDPM/LDM project galleries.

**02 — One description, several valid images [2 builds]**

Build 1: a real prompt and one verified generated sample. Build 2: reveal three additional samples for the same prompt, with distinct seeds if known. Ask: “If all satisfy the prompt, which one was the correct target?” This establishes conditional distributions and diversity before equations. If the source provides no seeds, say “different samples,” not fabricated seed numbers.

**03 — Change the condition, keep the experiment interpretable [2 builds]**

Use a documented GLIGEN example with text and an explicit box arrangement, or a qualitative prompt-variation example from the LDM project. First reveal the changed condition, then the documented image output. Label exactly what the source supplies: a different spatial arrangement, a different prompt, or a controlled generation experiment. Claim a fixed seed or otherwise fixed generation settings only when the original source documents them or a genuine controlled run produces the assets. Ask whether the condition uniquely determines pixel values. Do not promise perfect compositional adherence. Include one visible failure if the chosen source supplies a suitable example.

**04 — Fill a missing region [2 builds]**

First show a real image with a clearly drawn binary mask and separate “observed” and “unknown” labels. Then reveal two or more plausible completions of the same mask. Clearly identify each completion as sampled. Use RePaint or an explicitly documented conditional inpainting model. Save this exact example for Scene 54, where the observed pixels re-enter the mathematical story.

**05 — Turn rough strokes into a plausible image [2 builds]**

First show rough color strokes or an edited input from SDEdit, then reveal its documented generated result. Explain the application-level idea: add an appropriate amount of noise to an input and use a learned generative denoising process to obtain a plausible image influenced by that input. Keep the source input and result labels visible; do not describe SDEdit as preserving every input pixel or as requiring a newly trained model for each stroke drawing. Use the official assets https://sde-image-editing.github.io/images/stroke_based_generation.jpg and https://sde-image-editing.github.io/images/stroke_edit.jpg, with attribution to https://sde-image-editing.github.io/. Ask what changes as the balance between preserving the input and producing a plausible image changes.

**06 — Can a low-resolution image have many high-resolution explanations? [2 builds]**

Use an SR3 or latent-diffusion super-resolution example. Show low-resolution input enlarged with nearest-neighbor pixels and a bicubic baseline, then a generated high-resolution sample. Include a reference high-resolution image only if it is genuinely paired. Ask: “Does realistic detail establish that those details existed in the input?” This is a modelling question about conditional uncertainty, especially relevant to scientific imagery.

**07 — Draw the structure, generate the appearance [2 builds]**

Show a real edge map, depth map, or segmentation map from ControlNet or the LDM project. Reveal corresponding generated outputs. Connect this to students’ detection/segmentation lecture: here a spatial representation can become a condition. Keep the input modality explicitly labelled. Do not imply this exact control mechanism is present in every diffusion model.

**08 — Put a particular subject in a new context [2 builds]**

Use the official DreamBooth teaser at https://dreambooth.github.io/DreamBooth_files/teaser_static.jpg, attributed to https://dreambooth.github.io/. First show the few reference images of a subject used for model adaptation; then reveal that subject in documented new contexts. Distinguish the stages visually: reference images adapt the pretrained model during training; a new prompt is supplied to the adapted model at inference. Do not imply the reference images are simply passed to an unchanged text-to-image model at every denoising step. Keep this as a gallery preview of subject personalization rather than another architecture derivation. Ask which properties should remain recognizable when the context changes. Video generation and outpainting can appear in an optional application-gallery drawer with separately verified sources.

**09 — A connection to our lab’s questions [2 builds]**

First reveal a sourced DiffusionSat example with its actual conditioning information, such as metadata or low-resolution satellite imagery. Second reveal a small, original schematic of a partially observed signal or image and several possible reconstructions. Caption the latter “Possible research task,” not an accomplished lab result. Ask what uncertainty, held-out validation, and measurement consistency would matter. Use a real satellite figure where a demonstrated task is claimed.

**10 — What must a generator learn? [2 builds]**

Turn the previous gallery into a compact task map: unconditional sampling, text conditioning, spatial conditioning, and partial observations. Reveal the shared target \(p_\theta(x\mid c)\), with \(c\) absent for unconditional generation. State the three questions the next sections answer: how to create training examples; what to predict; how to turn predictions into a new sample. This is the first abstract overview, after the applications.

#### B. Make the forward process concrete

**11 — Connect to Lecture I [2 builds]**

Re-use the prior lecture’s actual VAE and GAN diagrams if available. First highlight a latent draw followed by generation. Then add a chain of many small stochastic transformations. Say that diffusion gives a different way to parameterize a generative distribution; avoid unsupported universal claims about superior coverage, stability, or quality. The new question is how to learn the transformations.

**12 — A digit gradually loses information [2 builds]**

Show the selected MNIST digit and a forward trajectory at several noise levels. Reveal the images left to right with a visible time index. Use genuine computed Gaussian corruption, not an opacity crossfade. Label this as a forward Markov trajectory if generated recursively; the direct marginal-slider demonstration later must be distinguished from this trajectory.

**13 — Removing noise is not a unique inverse [3 builds]**

Reveal two possible clean digits; then noisy versions whose plausible regions overlap; then a distribution over possible clean explanations for a chosen noisy observation. The visual should show ambiguity rather than claim two independently generated floating-point arrays are literally identical. Ask: “Can the noisy observation always tell us which original produced it?” Explain that the learned predictor exploits the distribution of training examples.

**14 — Watch a distribution, not just a picture [2 builds]**

Display the eight-cluster point cloud, then animate its noisy marginals approaching a near-standard Gaussian under the actual schedule. Keep axes and scale fixed. Distinguish a distribution of data points from uncertainty about a single datapoint. This scene introduces the analytic toy without presenting a neural training claim.

**15 — What is \(x_0\)? [2 builds]**

Show the digit, a small enlarged pixel block, and its numeric array. Label the tensor \([B,1,28,28]\) and explain the chosen normalization to \([-1,1]\). Show a single scalar \(x_0=1\) that will be tracked later. Note that noisy tensors may leave the clean image range; clamp only the displayed image, not the forward-process mathematics or training tensor.

**16 — One small corruption step [3 builds]**

Build exactly in this order: shrink the existing signal; draw independent Gaussian noise; add the scaled noise. Finish with

\[
x_t=\sqrt{1-\beta_t}\,x_{t-1}+\sqrt{\beta_t}\,\eta_t,
\qquad \eta_t\sim\mathcal N(0,I).
\]

Explain that \(\beta_t\) is a variance contribution, hence the square root multiplying noise. Use different notation for this per-step \(\eta_t\) and the aggregate \(\epsilon\) in the direct marginal formula. Ask students to identify which quantities are random and which schedule values are chosen.

**17 — Alpha and alpha-bar mean different things [3 builds]**

Reveal \(\alpha_t=1-\beta_t\), then a three-step product diagram, then \(\bar\alpha_t=\prod_{s=1}^{t}\alpha_s\). Use “one-step retention” and “cumulative retention” labels. Set \(\bar\alpha_0=1\). Color \(\alpha_t\) and \(\bar\alpha_t\) differently without relying on color alone. Include a small numeric strip with \(\alpha_1=\alpha_2=0.8\), \(\bar\alpha_1=0.8\), and \(\bar\alpha_2=0.64\).

**18 — Jump directly to a training noise level [4 builds]**

Expand \(x_2\) by substitution; separate its two independent Gaussian-noise terms; add their variances; reveal the general marginal. Derive rather than suddenly announce:

\[
q(x_t\mid x_0)=\mathcal N\!\left(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I\right),
\qquad
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon.
\]

The combined \(\epsilon\) is standard Gaussian. Equality here is distributional; it does not say that reusing one random array at every time reproduces the joint Markov trajectory.

**19 — One forward calculation, fully worked [4 builds]**

Use the consistent scalar fixture from Section 4 below. Reveal \(x_0=1\), \(\bar\alpha_2=0.64\), and \(\epsilon=-0.5\); take the two square roots; multiply; then add to get \(x_2=0.5\). Keep a small digit image beside the scalar but label the calculation “one coordinate.” Mark the deliberately large two-step schedule as an arithmetic illustration, not the image-training schedule or a noise-only endpoint.

**20 — Interact with the corruption [3 builds]**

Show a timestep slider, a fixed-noise/resample-noise control, and signal/noise coefficients. In fixed-noise mode, label the images “direct marginal samples coupled with a shared \(\epsilon\).” Give a second button for a genuine recursive forward trajectory. Plot \(\bar\alpha_t\) and \(\mathrm{SNR}_t=\bar\alpha_t/(1-\bar\alpha_t)\). At \(t=0\), display SNR as infinity or omit the singular point correctly. Explain that the endpoint must have very little remaining signal for a standard Gaussian start to be a good approximation.

#### C. Training is a supervised prediction problem we can manufacture

**21 — Which target should the network predict? [2 builds]**

Show the noisy image entering a shared box and ask students what could come out: clean image, noise, or reverse-step mean. Then select noise prediction for this lecture. Write \(\hat\epsilon=\epsilon_\theta(x_t,t)\), with input/output both \([B,1,28,28]\). State that this is a parameterization choice; later models can use other targets. The output head is unconstrained real-valued, not a sigmoid probability map.

**22 — Assemble one training example [3 builds]**

Reveal three independent random choices: \(x_0\) from the dataset, \(t\) from \(\{1,\ldots,T\}\), and \(\epsilon\sim\mathcal N(0,I)\). Compute \(x_t\). Then route only \((x_t,t)\) to the predictor and keep \(\epsilon\) on the target branch. Clearly mark that \(x_0\) and the sampled noise are available to the training procedure but are not additional inputs to this unconditional denoiser.

**23 — Compute a real loss value [2 builds]**

Use the same scalar target \(-0.5\), give the predictor output \(-0.4\), and obtain \((-0.5+0.4)^2=0.01\). Follow it with a tiny \(2\times2\) target/prediction pair and compute the mean over its four entries. Indicate whether an equation uses a sum or a mean and keep code consistent. Do not add arbitrary “accuracy” percentages for denoising.

**24 — Can the network know the exact sampled noise? [2 builds]**

Return to Scene 13’s ambiguous noisy input. Explain that the sampled target is known to us, while squared-error training learns an estimate of \(\mathbb E[\epsilon\mid x_t,t]\) in the ideal population setting. It does not receive a hidden copy of the random seed. Show two target arrows averaging into a prediction to make the conditional-mean idea visible. This prevents the misleading claim that a denoiser perfectly inverts every corruption.

**25 — One denoiser, many noise levels [3 builds]**

Show the same network used at three noise levels. Reveal the timestep as an explicit second input, then a timestep embedding injected into a block. Link sinusoidal timestep embeddings to the positional-encoding idea students have seen, while explaining that diffusion time describes corruption level rather than sequence position. Label weights as shared. Do not draw \(T\) separately learned networks.

**26 — A batch can contain different timesteps [3 builds]**

Use \(B=3\): three clean digits, \(t=[7,350,920]\) for \(T=1000\), and independently sampled noise. Reveal the gathered coefficients with shape \([3]\), their broadcast shape \([3,1,1,1]\), and the input/target tensors \([3,1,28,28]\). Show that the whole batch still uses one network call. If code changes \(T\), update the visual timestep examples accordingly.

**27 — Translate the picture into PyTorch [4 builds]**

Use the exact one-based schedule convention in Section 5. Reveal timestep sampling, broadcasting, noisy-input construction, and the loss in four synchronized blocks. Keep the active code line and the corresponding diagram element highlighted together. Put boilerplate in the companion file; the main scene should show about 10–14 readable lines over its builds. The executable companion must include imports and tested schedule construction.

**28 — Where do the gradients go? [2 builds]**

Highlight the MSE-to-denoiser gradient path. Freeze the schedule and the sampled training tuple in the picture. State that a standard training iteration samples one noise level per image; it does not need to backpropagate through all \(T\) sampling steps. Show the optimizer updating \(\theta\) once. Do not animate the inference loop as if training unrolled it by default.

**29 — Why this training needs no human noise labels [2 builds]**

Return to the data pipeline: the original images provide the learning material, and the corruption procedure gives a target. Ask whether the availability of noise labels removes the need for a representative dataset. Answer through a toy biased dataset comparison: the learned conditional expectations depend on the distribution seen in training. Keep the point about dataset support connected to earlier generative modelling, without a new ethics tangent.

**30 — The ELBO from Lecture I appears again [4 builds]**

First recall the earlier “log evidence = lower bound + posterior KL gap” diagram. Replace the single latent \(z\) by \(x_{1:T}\). Reveal the fixed forward chain \(q(x_{1:T}\mid x_0)\), the learned reverse joint \(p_\theta(x_{0:T})\), and the KL-sum decomposition in Section 4. Show the terminal prior-matching term, local reverse-matching terms, and data reconstruction term separately. Distinguish the tractable \(q(x_{t-1}\mid x_t,x_0)\), which knows the training image, from the harder \(q(x_{t-1}\mid x_t)\). Put the complete derivation in the appendix and notes; make the conceptual decomposition explicit in class.

**31 — Why did this become noise MSE? [3 builds]**

Show that with fixed covariance, matching Gaussian means gives a squared-error term. Substitute the noise-based reverse mean. Reveal the exact timestep weight for \(t>1\), then the commonly used simplified unweighted objective. Explicitly say that the simplified MSE changes the weighting of the variational terms; it is not literally the entire exact negative ELBO with all constants, endpoint terms, and weights retained. This sentence must appear in notes and in a short visible annotation.

#### D. Turn the learned prediction into generation

**32 — Sampling starts without a clean image [2 builds]**

Remove the dataset image and target branch. Start with a fresh \(x_T\sim\mathcal N(0,I)\). Show the current timestep and denoiser only, then reserve space for a sampled \(x_{T-1}\). State that model parameters stay fixed during generation. Keep all stochastic sources visible: initial noise and, for ancestral DDPM, intermediate noise draws.

**33 — A predicted clean image is useful [3 builds]**

Rearrange the forward equation one operation at a time:

\[
\hat x_0(x_t,t)=\frac{x_t-\sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}}.
\]

Use the scalar fixture to obtain \(\hat x_0=0.925\). Label this “estimated clean image,” with its own border style. It is an estimate because the network’s noise prediction is an estimate. At high noise levels this estimate can be unstable; image-display clipping should not be mistaken for proof of a correct intermediate reconstruction.

**34 — The estimate is not the next chain state [3 builds]**

Show \(x_t\), \(\hat x_0\), and \(x_{t-1}\) in three distinct panels. The model’s next state should still have the noise level associated with \(t-1\); setting it equal to the estimated clean image skips that definition. Ask which image feeds the next denoiser call. Give the answer only on the final reveal. Avoid presenting “subtract all predicted noise” as the DDPM update.

**35 — Parameterize one reverse distribution [4 builds]**

Build \(p_\theta(x_{t-1}\mid x_t)=\mathcal N(\mu_\theta(x_t,t),\sigma_t^2I)\) before revealing the mean formula. First compute the predicted noise, then multiply by \(\beta_t/\sqrt{1-\bar\alpha_t}\), then subtract, then divide by \(\sqrt{\alpha_t}\). Color one-step \(\alpha_t\) and cumulative \(\bar\alpha_t\) consistently. Speaker notes give the tractable-posterior derivation and make clear that the generic true reverse distribution need not be exactly Gaussian for large steps.

**36 — Why add noise during denoising? [3 builds]**

Reveal uncertainty in the possible \(x_{t-1}\) values at a fixed noisy observation. Specify this lecture’s variance choice \(\sigma_t^2=\tilde\beta_t\), then show

\[
\tilde\beta_t=\beta_t\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}.
\]

Explain that \(\beta_t\) is another historical fixed choice and learned variances are possible, without swapping them silently. Mark \(t=1\) as the last reverse update with no newly added sampling noise. “Denoising” describes the overall transition toward data; every individual coordinate need not improve monotonically on every random step.

**37 — One reverse step with actual numbers [4 builds]**

Use precisely the fixture in Section 4: \(x_2=0.5\), \(\hat\epsilon=-0.4\), \(\alpha_2=0.8\), \(\beta_2=0.2\), \(\bar\alpha_1=0.8\), \(\bar\alpha_2=0.64\). Compute the reverse mean \(0.708088\), variance \(1/9\), standard deviation \(1/3\), then use \(z=0.3\) to sample \(x_1=0.808088\). Keep \(\hat x_0=0.925\) visible to prevent conflation. Round only for display; verify the full-precision calculation in code.

**38 — The complete sampling loop [2 builds]**

Reveal the exact loop from Section 5 in descending \(t\). Synchronize the highlighted code line with a reverse-chain visualization. At the final step, visibly skip the extra random-noise addition. The denoiser, schedule, and variance convention must match training and the previous equations. Explain that reducing the number of steps requires a corresponding sampler/schedule adaptation; dropping arbitrary terms from this recurrence is not automatically valid acceleration.

**39 — What are we watching in a generation video? [2 builds]**

Place two aligned strips: actual sampler states \(x_t\) and clean-image estimates \(\hat x_0(x_t,t)\). Source them from the same genuine run if a model is executed. If using the analytic toy, make its identity prominent. The timeline controls both rows. Do not manufacture a “model trajectory” by fading a final image into noise. Explain that many public visualizations show estimated clean images rather than the raw sampler state.

#### E. The denoiser is a familiar kind of neural network

**40 — Predict one number at every spatial location [2 builds]**

Connect the denoiser’s output to the dense prediction shapes students have seen in segmentation. Use the digit tensor \([B,1,28,28]\) and predicted-noise tensor of the same shape. The target is continuous noise per coordinate, not a class per pixel. Introduce a convolutional encoder/decoder as one suitable backbone. Avoid saying U-Net is an obligatory definition of diffusion.

**41 — Walk through a small U-Net with shapes [3 builds]**

Use the explicit architecture contract in Section 6: \(28\to14\to7\to14\to28\), channels \(32,64,128,64,32\). Reveal the down path, then bottleneck, then up path and skip connections. Label concatenated channel counts. Show one skip connection at a time so arrows do not cross the text. Keep its prediction head linear. This is a pedagogical small network, not the exact published DDPM architecture.

**42 — Inject the timestep throughout the denoiser [2 builds]**

Open one residual block and show its time embedding projected to the block’s channel count and broadcast over height and width. Explain why a scalar timestep must become useful learned features. Then highlight repeated injection points in the full U-Net. Time conditioning should remain visible during inference too.

**43 — ViT returns as a diffusion backbone [2 builds]**

Show a noisy spatial latent divided into patches, then a transformer, then a prediction for every patch. Use full self-attention over noisy spatial tokens in the illustrative model. No class-token pooling is needed to generate a dense output. Mention the original DiT as a class-conditional latent diffusion transformer, with timestep/class conditioning; do not mislabel its original architecture as universally using the text cross-attention design of Scene 48.

**44 — Why diffuse in a compressed image representation? [4 builds]**

Recall the image autoencoder from Lecture I. Show a familiar, explicitly labelled SD-v1-style example: image \([B,3,512,512]\) to latent \([B,4,64,64]\). Count spatial positions \(512^2/64^2=64\), then scalar entries \((3\cdot512^2)/(4\cdot64^2)=48\). These are different ratios. Label them as representation-size comparisons, not guaranteed memory, speed, or quality improvements. Mention that other autoencoders and systems use other dimensions. The image latent is spatial, unlike the tiny vector used in the teaching VAE.

**45 — Two training stages, with explicit frozen modules [3 builds]**

First train an image autoencoder to produce useful reconstructions; then use its encoded latents to train diffusion. During the diffusion stage, show the encoder and decoder frozen in this standard example, and identify which text components are frozen if text is introduced. The diffusion target is noise in latent coordinates. Explain that the autoencoder’s latent normalization/scaling must follow the actual checkpoint; a compressed representation is not automatically exactly standard normal because it came from a VAE.

**46 — Generate in latent space, decode once [3 builds]**

Start with latent Gaussian noise \([B,4,64,64]\), execute the denoising loop, then decode the final latent to \([B,3,512,512]\). The image encoder is absent during ordinary text-to-image generation. Label the final latent as \(z_0\) to distinguish it from pixel-space \(x_0\), and explain this notation switch. Do not feed a hidden reference image into a purported text-to-image sampling example. Note that image-to-image tasks can use an encoder because their input includes an image.

#### F. Use what students know about text encoders and attention

**47 — What information comes from the prompt? [3 builds]**

Reveal a prompt, its tokenizer output, and token embeddings from a text encoder. Use verified tokens from an actual tokenizer if showing a real model; otherwise label the rows as pedagogical word features. Trace \([B,L,d_{\text}]\) features into a denoiser. Explain that the prompt representation is usually reused across denoising steps in this example. CLIP-like text encoding is one possible component; the CLIP contrastive training loss is not the diffusion noise-prediction loss.

**48 — Image queries ask about text keys [4 builds]**

Explicitly compare with the cross-attention VLM variant students saw: there text queries attended to visual keys/values. In this selected latent-diffusion block, noisy image features provide queries and text features provide keys and values. Reveal projection shapes, similarity scores, row softmax, and the weighted sum separately:

\[
H\in\mathbb R^{B\times N\times d_i},\quad C\in\mathbb R^{B\times L\times d_c},\quad
Q=HW_Q,\ K=CW_K,\ V=CW_V,
\]
\[
A=\operatorname{softmax}_{\text{text positions}}(QK^\top/\sqrt{d_k})\in\mathbb R^{B\times N\times L},
\qquad AV\in\mathbb R^{B\times N\times d_v}.
\]

Specify that these shapes omit the explicit head axis for one-head readability. The output has one row per image query. Do not swap Q/K/V merely to match a previously used diagram.

**49 — One image query, a few words, actual attention numbers [3 builds]**

Use the hand-designed two-query, three-key example in Section 4. Show query/key vectors, compute dot products and scaling, then row softmax and one output vector. Label all chosen activations as a toy calculation. Do not present the resulting weights as extracted attention from a trained diffusion model. Ask which dimension softmax should normalize and why. Provide a toggle to highlight the second image query.

**50 — The condition influences repeated denoising decisions [2 builds]**

Return to the complete latent denoiser and highlight cross-attention at selected spatial resolutions/blocks. Reuse the same text embedding and changing noisy image features across steps. Show why this differs from simply painting the prompt into pixels or concatenating a final class label. Avoid a false architectural universal: some transformer-based generators use joint attention or other conditioning schemes.

**51 — Learn conditional and unconditional predictions with one model [3 builds]**

Show the normal conditional training tuple \((z_t,t,c,\epsilon)\), then randomly replace \(c\) with a defined null/empty condition for a subset of examples. Use \(p_{\mathrm{drop}}=0.1\) as a clearly labelled teaching choice, not a universal setting. The latent/image and noise target stay the same. Reveal that the same trained weights can then produce \(\epsilon_c\) and \(\epsilon_u\). Specify the actual model’s null-condition implementation instead of silently assuming zero token embeddings.

**52 — Classifier-free guidance is a precise combination [3 builds]**

Use the convention

\[
\epsilon_{\mathrm{cfg}}=\epsilon_u+s(\epsilon_c-\epsilon_u).
\]

Reveal the unconditional prediction, conditional-minus-unconditional difference, and scaled sum. State visibly: \(s=0\) gives unconditional prediction; \(s=1\) gives the ordinary conditional prediction; \(s>1\) extrapolates the conditional direction. Some papers use \(w=s-1\), so quote their convention if reproducing their formula. Explain that two logical evaluations can be batched together. Do not call this a separately trained classifier or claim the noise difference is a literal edit mask.

**53 — Change guidance while holding the experiment fixed [2 builds]**

Use actual precomputed samples from one model with the same initial seed and documented sampler settings at several \(s\) values. Label the value under every image. Show observed tradeoffs in adherence, variety across seeds, or artifacts; do not claim image quality must increase monotonically. If no genuine model run or source-supported series is available, make the interactive manipulate a numerical vector example and label the image comparison as unavailable rather than fabricate it.

**54 — Revisit completion and super-resolution [2 builds]**

Return to the opening examples and identify the actual condition supplied to each method. For a RePaint-inspired intuition, combine a sampled unknown region with an appropriately noised known region at the matching timestep; label this a simplified teaching illustration and mention that RePaint adds a resampling schedule. For SR3-style super-resolution, the low-resolution observation conditions a model trained for that task. These are specific ways to define conditional generation, not a guarantee that the unconditional code automatically solves every inverse problem.

#### G. Inspect evidence, extend to language, close the loop

**55 — What do these results establish? [3 builds]**

Use an actual sample gallery, one documented failure, and the task condition. Ask separately about visual plausibility, diversity, adherence to condition, and preservation of observed information. For a locally trained digit model, record the checkpoint and show an uncurated seeded batch as well as selected examples. Do not assign FID or downstream gains unless computed with a documented protocol. Distinguish a teaching toy, a pretrained model demonstration, and a paper’s reported output.

**56 — Why not add Gaussian noise to token IDs? [2 builds]**

Show tokens and their arbitrary integer lookup IDs. Change the IDs by a vocabulary permutation while preserving the sentence, exposing why arithmetic proximity of IDs is meaningless. Then show a `[MASK]` corruption as a discrete alternative. State that other text diffusion formulations exist, including continuous representations; this section teaches one masked discrete approach, not a universal definition of language diffusion.

**57 — Build a masked-text training example [3 builds]**

Use a complete sentence, a selected masking probability, and sampled masked positions. Reveal only the masked positions as prediction targets and show full/bidirectional context in the illustrative denoising model. Then show logits of shape \([B,L,V]\) and the masked-position cross-entropy. Label a simple unweighted objective as a pedagogical masked-denoising objective; if identifying it with LLaDA’s diffusion objective, include its \(1/t\) weighting and exact conditioning convention as specified in Section 4. Never silently substitute standard fixed-rate BERT training for an entire diffusion model.

**58 — Generate by iterative discrete reconstruction [2 builds]**

Start with a fixed prompt and masked response slots. Reveal candidate predictions at several positions; keep a selected subset and continue with a specified schedule. If using confidence-based remasking, explain what is remasked and why. The visible text sequence can be an illustrative sampler trace, but it must be labelled as such unless actually produced by a model. Keep observed prompt tokens fixed. Show that output length or a maximum response region must be specified/handled by the chosen system.

**59 — Compare generation procedures, not slogans [3 builds]**

Place familiar autoregressive generation beside iterative masked denoising. Compare what one pass predicts, which context it can use, which tokens remain uncertain, and how many sequential passes occur. Full-sequence predictions allow a different schedule, but not an automatic wall-clock speedup; hardware, length, sampling steps, caching, and quality targets matter. Use a small known sentence for the reveal rather than abstract complexity claims with missing assumptions.

**60 — Return to the opening applications [4 builds]**

Reveal the same text-to-image, inpainting, super-resolution, and masked-text examples. For each, ask students to identify the corrupted object, available condition, training target, and repeated inference operation. End with three prediction questions and short answers in notes: “Does training require sampling every reverse step?”; “Why is \(\hat x_0\) different from \(x_{t-1}\)?”; “What changes when the data are discrete tokens?” End the main lecture here. Reference appendices should remain optional and clearly navigable.

### 4. Mathematical and numerical specification

All equations and displayed numbers must be checked against code. Use this notation consistently throughout the main lecture:

| Symbol | Definition |
|---|---|
| \(x_0\) | A clean data example; pixels in the initial formulation |
| \(x_t\) | The corrupted state at noise level \(t\) |
| \(q\) | Data/forward distributions, with the relevant arguments always shown |
| \(p_\theta\) | Learned reverse/generative distribution |
| \(\beta_t\) | Forward variance schedule, \(0<\beta_t<1\) |
| \(\alpha_t\) | \(1-\beta_t\), one-step retention |
| \(\bar\alpha_t\) | \(\prod_{s=1}^t\alpha_s\), cumulative retention; \(\bar\alpha_0=1\) |
| \(\eta_t\) | Independent noise for one recursive forward transition |
| \(\epsilon\) | Aggregate noise in the direct marginal training formula |
| \(\epsilon_\theta\) | Denoiser prediction; same dimensions as the diffused object |
| \(\hat x_0\) | Estimated clean image, not a sampled next chain state |
| \(\tilde\beta_t\) | Variance of \(q(x_{t-1}\mid x_t,x_0)\) |
| \(z\) in the scalar sampler | A newly drawn standard Gaussian, explicitly labelled as sampling noise |
| \(z_t\) in the latent section | A noisy spatial image latent; warn about the notation change |

#### Two-step marginal derivation

\[
x_1=\sqrt{\alpha_1}x_0+\sqrt{\beta_1}\eta_1,
\]
\[
x_2=\sqrt{\alpha_2\alpha_1}x_0+
\sqrt{\alpha_2\beta_1}\eta_1+\sqrt{\beta_2}\eta_2.
\]

The independent noise terms have total variance

\[
\alpha_2\beta_1+\beta_2
=\alpha_2(1-\alpha_1)+(1-\alpha_2)
=1-\alpha_1\alpha_2.
\]

Therefore \(x_2\mid x_0\) has mean \(\sqrt{\alpha_1\alpha_2}x_0\) and variance \((1-\alpha_1\alpha_2)I\). Extend the pattern to the general marginal. Explain the step rather than hiding it under “by Gaussian properties.”

#### Consistent scalar fixture

Use a two-step arithmetic-only schedule \(\beta_1=\beta_2=0.2\), hence \(\alpha_1=\alpha_2=0.8\), \(\bar\alpha_1=0.8\), and \(\bar\alpha_2=0.64\). It is intentionally unlike the small-step image-training schedule.

\[
x_0=1,\qquad \epsilon=-0.5,
\qquad x_2=0.8(1)+0.6(-0.5)=0.5.
\]

Suppose the network predicts \(\hat\epsilon=-0.4\). Then

\[
\ell=(-0.5-(-0.4))^2=0.01,
\]
\[
\hat x_0=\frac{0.5-0.6(-0.4)}{0.8}=0.925,
\]
\[
\mu_\theta(x_2,2)
=\frac{0.5-\frac{0.2}{0.6}(-0.4)}{\sqrt{0.8}}
\approx0.7080881929,
\]
\[
\tilde\beta_2=0.2\frac{1-0.8}{1-0.64}=\frac19,
\qquad \sigma_2=\frac13,
\]
\[
z=0.3\quad\Longrightarrow\quad x_1=\mu_\theta+\sigma_2z
\approx0.8080881929.
\]

Ask students to name each of the three distinct values: noisy input \(0.5\), estimated clean value \(0.925\), sampled previous state \(0.808088\). If the true aggregate noise is inserted instead of the prediction, the algebra recovers \(x_0\); explain why inference does not have access to that true noise.

#### Reverse model and variance

For the fixed-variance epsilon-parameterized example, use

\[
\mu_\theta(x_t,t)=\frac1{\sqrt{\alpha_t}}
\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right),
\]
\[
x_{t-1}=\mu_\theta(x_t,t)+\mathbf1[t>1]\sqrt{\tilde\beta_t}\,\xi,
\qquad \xi\sim\mathcal N(0,I).
\]

The tractable training posterior is

\[
q(x_{t-1}\mid x_t,x_0)=\mathcal N(\tilde\mu_t,\tilde\beta_t I),
\]
\[
\tilde\mu_t=
\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0+
\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t.
\]

Use the second expression in notes/appendix to connect the estimated clean image to the reverse mean. Do not imply that \(q(x_{t-1}\mid x_t)\), which marginalizes the unknown clean image, is the same Gaussian with a known \(x_0\). Small reverse steps make the Gaussian parameterization useful; its validity should not be justified by saying that every noisy distribution is Gaussian.

#### ELBO bridge with correct signs and scope

Define

\[
p_\theta(x_{0:T})=p(x_T)\prod_{t=1}^{T}p_\theta(x_{t-1}\mid x_t),
\quad
q(x_{1:T}\mid x_0)=\prod_{t=1}^{T}q(x_t\mid x_{t-1}).
\]

Then

\[
\log p_\theta(x_0)
\geq
\mathbb E_q\left[\log\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}\right]
=\operatorname{ELBO}(x_0).
\]

The negative bound decomposes as

\[
\mathcal L_{\mathrm{VLB}}=
D_{\mathrm{KL}}(q(x_T\mid x_0)\|p(x_T))
+\sum_{t=2}^{T}\mathbb E_{q(x_t\mid x_0)}
D_{\mathrm{KL}}\!\left(q(x_{t-1}\mid x_t,x_0)\|p_\theta(x_{t-1}\mid x_t)\right)
-\mathbb E_{q(x_1\mid x_0)}\log p_\theta(x_0\mid x_1).
\]

For fixed reverse variance and \(t>1\), the mean-dependent KL term becomes

\[
\mathbb E\left[
\underbrace{\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}}_{w_t}
\|\epsilon-\epsilon_\theta(x_t,t)\|_2^2
\right]+C.
\]

The simplified training objective is

\[
\mathcal L_{\mathrm{simple}}=
\mathbb E_{x_0,t,\epsilon}
\|\epsilon-\epsilon_\theta(\sqrt{\bar\alpha_t}x_0+
\sqrt{1-\bar\alpha_t}\epsilon,t)\|_2^2.
\]

The exact endpoint likelihood treatment, constants, and weighting are not all retained in the simplified loss. Never apply the \(t>1\) weight with \(\tilde\beta_1=0\) at \(t=1\). An implementation can use unweighted MSE across sampled timesteps, but that is the labelled simplified objective, not a numerical evaluation of the entire exact VLB. Discretized image likelihood details belong in the appendix.

#### Tiny attention fixture

Choose one head with

\[
Q=\begin{bmatrix}1&0\\0&1\end{bmatrix},\quad
K=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix},\quad
V=\begin{bmatrix}1&0\\0&2\\1&1\end{bmatrix}.
\]

The three key/value rows can be labelled “golden,” “dog,” and “grass” as hand-chosen word features. The two query rows are image-feature positions. With \(d_k=2\):

\[
QK^\top/\sqrt2=
\begin{bmatrix}0.707107&0&0.707107\\0&0.707107&0.707107\end{bmatrix},
\]
\[
A\approx\begin{bmatrix}0.401112&0.197776&0.401112\\0.197776&0.401112&0.401112\end{bmatrix},
\]
\[
AV\approx\begin{bmatrix}0.802224&0.796664\\0.598888&1.203336\end{bmatrix}.
\]

Rows of \(A\) sum to one. Show one row fully before revealing the second. These numbers are illustrative activations, not recovered parameters, learned attention, or proof that a particular word controls a particular pixel.

#### Masked discrete text: one precise example

Let \(x_0=(x_0^1,\ldots,x_0^L)\) be a sequence. For a sampled continuous corruption level \(t\in(0,1]\), independently mask each modeled position with probability \(t\). The denoiser outputs vocabulary logits for each position. A simple pedagogical masked-denoising loss is masked-position cross-entropy; call it that.

If using the LLaDA-style uniformly sampled masking-time objective, state the corresponding weighting:

\[
\mathcal L=-\mathbb E_{t,x_0,x_t}\left[
\frac1t\sum_{i=1}^{L}\mathbf1[x_t^i=\mathrm{MASK}]
\log p_\theta(x_0^i\mid x_t)
\right],
\]

with any sequence-length normalization stated explicitly. Avoid a numerical \(t=0\) sample. For conditional instruction/response training, specify which tokens are corrupted and which prompt/context tokens remain observed; do not imply the same mask convention for every pretraining and fine-tuning setup. During generation, define the response-slot count, schedule, sampling rule, and any remasking policy. Parallel token predictions do not by themselves establish faster end-to-end generation.

### 5. Executable PyTorch companion: required core

Provide a runnable companion that implements the lecture’s own schedule, noise objective, and sampler. Keep math and code indexing aligned using **one-based mathematical time with a sentinel array entry at index zero**. That avoids a silent \(t\) versus \(t+1\) mismatch.

```python
import torch
import torch.nn.functional as F


def make_schedule(T=1000, device="cpu", dtype=torch.float32):
    # Mathematical t = 1,...,T. Entry zero is a sentinel.
    beta = torch.zeros(T + 1, device=device, dtype=dtype)
    beta[1:] = torch.linspace(1e-4, 2e-2, T, device=device, dtype=dtype)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)  # alpha_bar[0] == 1
    posterior_var = torch.zeros_like(beta)
    posterior_var[1:] = (
        beta[1:] * (1.0 - alpha_bar[:-1]) / (1.0 - alpha_bar[1:])
    )
    return beta, alpha, alpha_bar, posterior_var


def extract(values, t, x):
    # t: [B] integer indices; return [B,1,...,1] for broadcasting.
    assert t.dtype == torch.long and t.shape == (x.shape[0],)
    return values[t].reshape(x.shape[0], *([1] * (x.ndim - 1)))


def training_loss(model, x0, alpha_bar, T):
    t = torch.randint(1, T + 1, (x0.shape[0],), device=x0.device)
    eps = torch.randn_like(x0)
    ab = extract(alpha_bar, t, x0)
    xt = ab.sqrt() * x0 + (1.0 - ab).sqrt() * eps
    eps_hat = model(xt, t)
    assert eps_hat.shape == x0.shape
    # Mean over examples AND coordinates: a scaled L_simple.
    return F.mse_loss(eps_hat, eps)


@torch.no_grad()
def sample_ddpm(model, shape, schedule, T=1000, device="cpu"):
    beta, alpha, alpha_bar, posterior_var = schedule
    x = torch.randn(shape, device=device)
    was_training = model.training
    model.eval()
    try:
        for step in range(T, 0, -1):
            t = torch.full((shape[0],), step, device=device, dtype=torch.long)
            eps_hat = model(x, t)
            b = extract(beta, t, x)
            a = extract(alpha, t, x)
            ab = extract(alpha_bar, t, x)
            mean = (x - b * eps_hat / (1.0 - ab).sqrt()) / a.sqrt()
            if step > 1:
                var = extract(posterior_var, t, x)
                x = mean + var.clamp_min(0.0).sqrt() * torch.randn_like(x)
            else:
                x = mean  # No additional sampling noise on the final step.
    finally:
        model.train(was_training)
    return x
```

Turn this into an executable `train_mnist_ddpm.py` with: seeding, deterministic data selection for the lecture example, dataset loading and documented normalization, a `TinyDenoiser`, optimizer, training log, checkpoint save/load, a sample command, and trajectory export. Parameterize dataset/cache/output paths. Add a device choice that works on CPU and available accelerators. Keep training budget configurable and record actual steps/epochs. Do not promise high-quality samples after a token smoke run. Include a one-batch execution mode for checking shapes and gradients, clearly distinct from training a useful model.

Use \(T=1000\) for the displayed linear schedule unless there is a documented change. If a short schedule is used for speed, recompute a schedule with a suitably noisy endpoint instead of blindly using the same \(10^{-4}\to0.02\) range at \(T=20\). Record \(\bar\alpha_T\). A poor Gaussian endpoint approximation should not be hidden by an attractive visualization. Modern fast samplers are a separate, sourced extension.

When presenting this code, reveal only the active conceptual lines and retain shape annotations. Keep the complete runnable code in the companion artifact. Reusing a public checkpoint requires a compatible architecture, normalization, timestep convention, noise schedule, and prediction target; do not force an incompatible checkpoint into the teaching sampler.

### 6. Architecture contracts

#### Small convolutional denoiser for MNIST

Implement an ordinary `nn.Module` with named layers and a readable `forward`, so students can inspect every stage. A suitable shape contract is:

| Stage | Operation | Output |
|---|---|---|
| Input | Noisy digit | \([B,1,28,28]\) |
| Time embedding | Sin/cos features, then two-layer MLP | \([B,64]\) |
| Encoder 1 | Time-conditioned block \(1\to32\) | \([B,32,28,28]\) |
| Down 1 | Conv \(32\to64\), kernel 4, stride 2, padding 1 | \([B,64,14,14]\) |
| Encoder 2 | Time-conditioned block \(64\to64\) | \([B,64,14,14]\) |
| Down 2 | Conv \(64\to128\), kernel 4, stride 2, padding 1 | \([B,128,7,7]\) |
| Middle | Time-conditioned block \(128\to128\) | \([B,128,7,7]\) |
| Up 1 | Resize to \(14\times14\), conv \(128\to64\) | \([B,64,14,14]\) |
| Skip 1 | Concatenate encoder-2 output | \([B,128,14,14]\) |
| Decoder 1 | Time-conditioned block \(128\to64\) | \([B,64,14,14]\) |
| Up 2 | Resize to \(28\times28\), conv \(64\to32\) | \([B,32,28,28]\) |
| Skip 2 | Concatenate encoder-1 output | \([B,64,28,28]\) |
| Decoder 2 | Time-conditioned block \(64\to32\) | \([B,32,28,28]\) |
| Head | \(1\times1\) convolution \(32\to1\), linear output | \([B,1,28,28]\) |

Each time-conditioned block can use two \(3\times3\) convolutions, GroupNorm, SiLU, an additive projected time embedding, and a projected residual when channels change. Use GroupNorm group counts that divide the actual channel counts. At the first block, apply normalization after projecting the one input channel into 32 channels. The time feature must broadcast over spatial dimensions; the learned weights are reused for every timestep. Record exact parameter count from the implemented model rather than estimating it in slides.

#### DiT shape example

For a pedagogical spatial latent \([B,4,32,32]\) with patch size \(2\times2\): there are \(N=256\) patches; each flattened patch has \(4\cdot2\cdot2=16\) scalars. Project to \([B,256,d]\), process with transformer blocks conditioned on timestep and the chosen condition, project back to 16 values per patch for a noise-only illustrative head, and unpatchify to \([B,4,32,32]\). If reproducing an actual DiT with learned variance, state the extra output channels and distinguish it from the simplified noise-only diagram. Do not imply a CLS token represents an entire generated image.

#### Latent/text attention shape example

At a chosen intermediate resolution, explicitly flatten spatial features \([B,d_i,H,W]\) to \([B,N,d_i]\), where \(N=HW\). Text features are \([B,L,d_c]\). For \(h\) heads, display \(Q:[B,h,N,d_k]\), \(K:[B,h,L,d_k]\), \(V:[B,h,L,d_v]\), attention \([B,h,N,L]\), and output \([B,h,N,d_v]\) before merging heads and restoring the spatial layout. Use a small \(N,L\) for the live numbers and realistic dimensions only for the architecture annotation.

### 7. Interactive demonstrations to implement

All demos must support keyboard controls, a reset, a deterministic seed, a pause, and a legible static export. Do not make the lecture depend on training a model live. Precompute genuine trajectories for classroom reliability where useful, and identify their provenance.

#### Demo A — Noise explorer

- Select the actual digit used in the lecture; show clean \(x_0\), noise \(\epsilon\), and the computed \(x_t\).
- Controls: timestep, fixed/resampled noise, scalar coordinate selection, and direct marginal versus recursive forward trajectory.
- Live values: \(\beta_t,\alpha_t,\bar\alpha_t,\sqrt{\bar\alpha_t},\sqrt{1-\bar\alpha_t}\), and one coordinate’s arithmetic.
- Use a persistent pixel-intensity scale. Avoid independent per-frame autoscaling that visually hides increasing noise.
- Export a labelled contact sheet. Mark direct marginal coupling correctly; its frames share the right marginals but are not the same joint process as recursively sampled frames.

#### Demo B — One training tuple

- Controls: digit, timestep, random seed, and a “make batch of 3” button.
- Show the sampled target noise and actual predicted noise if a compatible trained checkpoint exists.
- When using a hand-entered predicted scalar or small array, label it “illustrative prediction.” Do not put a fabricated checkpoint name beside it.
- Show elementwise squared errors and their mean. Clicking a pixel should highlight the same coordinate across clean/noisy/target/prediction/error panels.
- A synchronized code panel highlights the line that produced each array.

#### Demo C — Numerical reverse step

- Initialize to the fixture \(x_2=0.5\), \(\hat\epsilon=-0.4\), \(\alpha_2=0.8\), \(\bar\alpha_2=0.64\), \(\bar\alpha_1=0.8\).
- Reveal predicted clean value, reverse mean, posterior variance, standard deviation, fresh noise, and sampled \(x_1\) separately.
- Make \(\hat\epsilon\) and the fresh noise draw adjustable. Derive \(\beta_t\) and \(\bar\alpha_t\) from the selected schedule so sliders cannot create inconsistent parameters.
- Draw a one-dimensional Gaussian for the reverse distribution and mark the sampled point. A posterior standard deviation is a standard deviation, not its variance.
- Provide an explicit final-step state showing no extra noise.

#### Demo D — Distribution-level analytical oracle

Use

\[
p_0(x)=\frac18\sum_{k=0}^{7}\mathcal N(x;\mu_k,\tau^2 I),
\quad
\mu_k=2[\cos(2\pi k/8),\sin(2\pi k/8)]^\top,
\quad \tau=0.18.
\]

Its noisy marginal has component means \(m_{k,t}=\sqrt{\bar\alpha_t}\mu_k\) and scalar variance \(v_t=\bar\alpha_t\tau^2+1-\bar\alpha_t\). Compute responsibilities with a stable softmax:

\[
r_k(x,t)=\operatorname{softmax}_k\left(-\frac{\|x-m_{k,t}\|^2}{2v_t}\right),
\quad
s(x,t)=\nabla_x\log q_t(x)=\sum_k r_k(x,t)\frac{m_{k,t}-x}{v_t},
\]
\[
\epsilon_{\mathrm{oracle}}(x,t)=-\sqrt{1-\bar\alpha_t}\,s(x,t).
\]

Use the oracle prediction inside the lecture’s DDPM recurrence to animate particles. The visible label must read **“Analytical score oracle for a known toy distribution; no neural network was trained.”** Also identify the Gaussian reverse sampler as a finite-step approximation: knowing the exact score does not make every coarse DDPM transition the exact reverse conditional. Do not move particles toward assigned centers with an arbitrary interpolation while calling it denoising.

If implementing an optional exact toy reverse mode, derive it from the Gaussian mixture. Given component \(k\), let \(m_{k,t-1}=\sqrt{\bar\alpha_{t-1}}\mu_k\), \(v_{t-1}=\bar\alpha_{t-1}\tau^2+1-\bar\alpha_{t-1}\), and \(v_t=\alpha_t v_{t-1}+\beta_t\). Sample \(k\) from the responsibilities at \(x_t\), then

\[
x_{t-1}\mid x_t,k\sim\mathcal N\left(
m_{k,t-1}+\frac{\sqrt{\alpha_t}v_{t-1}}{v_t}(x_t-\sqrt{\alpha_t}m_{k,t-1}),
\frac{\beta_t v_{t-1}}{v_t}I\right).
\]

For exact end-to-end distribution recovery in this optional mode, initialize from the known \(q_T\), or label initialization from \(\mathcal N(0,I)\) as an endpoint approximation. Keep optional score/mixture algebra in a details drawer or appendix; the main visualization should show what changes in the distribution.

#### Demo E — Text attention and guidance

- Attention tab: step through \(QK^\top\), scaling, row softmax, and weighted value sum using the fixed fixture. Highlight one spatial query at a time.
- Guidance tab: numerical vector interpolation/extrapolation with labelled \(s=0,1,3\). Pair it with a genuine fixed-seed sample series only if available.
- For real samples, record checkpoint, prompt, negative/null condition, seed, scheduler, step count, guidance convention, and image size.
- The illustrative attention weights and real image samples must not be displayed in a way that implies those toy weights were extracted from that model.

#### Demo F — Masked discrete denoising

- Choose a short fixed sentence and corruption probability. Show which tokens are observed, masked, and predicted.
- Make token IDs visible only for the Scene 56 demonstration of their arbitrary nature.
- A generation mode shows a fixed prompt, response slots, and a finite reveal/remask schedule. A scripted trace is acceptable when labelled “illustrative sampling trace.”
- If actual model inference is available, record the model and generation settings. Do not call a hard-coded reconstruction model inference.

### 8. Appendices that support the core lecture

Keep these after the main ending with navigation links back to the relevant scene. Do not silently load their full content into the 120-minute route.

1. **Gaussian algebra refresher:** sums of independent Gaussians, distinction between variance and standard deviation, the two-step marginal calculation.
2. **Full diffusion ELBO:** Jensen/variational construction, reverse factorization of the forward chain, posterior Gaussian, endpoint term, weighted noise objective.
3. **Alternative prediction targets:** \(x_0\), \(\epsilon\), and velocity parameterizations; state the convention used for velocity before giving conversion equations.
4. **Schedules and sampling:** linear versus cosine noise schedules, timestep re-spacing, DDIM as a separately defined sampler, and learned variance. Include only sourced algorithmic details.
5. **Score connection:** the known-mixture oracle and the relationship between noise prediction and the noisy marginal score. Keep the exact expectation interpretation visible.
6. **Flow matching overview:** one visual comparing stochastic diffusion trajectories with a specified flow-matching interpolation/velocity field; use the original flow-matching source. State that related modern generators need not use the DDPM training target or ancestral sampler taught in the main route. No second full derivation is required.
7. **Discrete diffusion:** absorbing-mask transition, time-weighted loss, sequence-length choices, remasking schedule, and the difference between a one-shot masked language model demonstration and an iterative generative sampler.
8. **Applications and limitations:** additional sourced image/video/remote-sensing examples, clearly separated from original lab proposals.

### 9. Sources and factual attribution

Use the following primary sources for the technical claims and obtain figure assets from their official project pages/repositories when suitable. The pedagogical-reference section in this same file supplies the selected MIT and Lilian Weng resources for teaching inspiration. Write original explanations and redraw instructional diagrams; cite sources near claims and assets.

| Source | Used for | Primary URL |
|---|---|---|
| Ho, Jain, Abbeel, DDPM (2020) | Forward process, posterior, epsilon objective, ancestral sampling | https://arxiv.org/abs/2006.11239 |
| Official DDPM implementation | Algorithm/asset cross-check | https://github.com/hojonathanho/diffusion |
| Nichol & Dhariwal, Improved DDPM (2021) | Variance/schedule refinements | https://arxiv.org/abs/2102.09672 |
| Rombach et al., Latent Diffusion (2022) | Compression, spatial latents, conditioning, applications | https://arxiv.org/abs/2112.10752 |
| Official LDM implementation | Architecture and sample provenance | https://github.com/CompVis/latent-diffusion |
| Peebles & Xie, DiT (2023) | Transformer backbone on latent patches | https://arxiv.org/abs/2212.09748 |
| Official DiT implementation | Actual conditioning and output-channel details | https://github.com/facebookresearch/DiT |
| Ho & Salimans, CFG (2022) | Conditional dropout and guidance | https://arxiv.org/abs/2207.12598 |
| Lugmayr et al., RePaint (2022) | Inpainting with a diffusion prior and resampling | https://arxiv.org/abs/2201.09865 |
| Official RePaint implementation | Paired mask/completion assets | https://github.com/andreas128/RePaint |
| Saharia et al., SR3 (2021) | Conditional super-resolution | https://arxiv.org/abs/2104.07636 |
| Official SR3 project | Visual super-resolution examples | https://iterative-refinement.github.io/ |
| Zhang et al., ControlNet (2023) | Spatial conditioning | https://arxiv.org/abs/2302.05543 |
| Official ControlNet implementation | Edge/depth/pose sample provenance | https://github.com/lllyasviel/ControlNet |
| Khanna et al., DiffusionSat (2024) | Satellite imagery and metadata conditioning | https://arxiv.org/abs/2312.03606 |
| Official DiffusionSat project | Remote-sensing examples | https://www.samarkhanna.com/DiffusionSat/ |
| Austin et al., D3PM (2021) | Discrete-state corruption and diffusion | https://arxiv.org/abs/2107.03006 |
| Nie et al., LLaDA (2025) | Masked language diffusion and time weighting | https://arxiv.org/abs/2502.09992 |
| Lipman et al., Flow Matching (2023) | Optional modern formulation overview | https://arxiv.org/abs/2210.02747 |

Verify exact video source and any additional model-specific claims while implementing; add them to the asset/source manifest. Do not attach an unrelated source to a figure merely because it is about diffusion. Dates above identify historical papers, not a ranking or a claim of current state of the art.

### 10. Acceptance checks specific to this lecture

#### Mathematical correctness

- Forward recursion uses \(\sqrt{\beta_t}\) for noise scale, while \(\beta_t\) is variance.
- \(\alpha_t\), \(\bar\alpha_t\), and \(\bar\alpha_0=1\) are never interchanged.
- Direct marginal coupling and a recursively sampled joint trajectory are labelled differently.
- The scalar fixture gives \(x_2=0.5\), loss \(0.01\), \(\hat x_0=0.925\), mean \(0.7080881929\), variance \(1/9\), and sample \(0.8080881929\).
- The denoiser input excludes clean image and actual sampled noise in the unconditional case.
- The population MSE explanation is a conditional-mean statement, not perfect identification of hidden noise.
- The ELBO inequality direction, negative-bound signs, and endpoint term are correct.
- Simplified MSE is distinguished from the fully weighted exact VLB; the \(t>1\) weighting is not evaluated at zero posterior variance.
- Reverse mean uses \(1/\sqrt{\alpha_t}\), not \(1/\sqrt{\bar\alpha_t}\).
- A clean-image estimate is not silently used as the next DDPM chain state.
- The variance convention is documented and matches code; no new random noise is added after the last reverse prediction.
- Text conditioning shows the correct Q/K/V origins and row-softmax axis.
- Guidance uses a single declared convention with \(s=0\) unconditional and \(s=1\) conditional.
- Token-ID arithmetic is not used as a Gaussian text-corruption model.

#### Code and demo behavior

- Execute one forward/backward batch; verify finite loss, nonzero denoiser gradients, and correct input/output shapes.
- Run the scalar and attention fixtures as numerical checks, with tolerance around \(10^{-6}\) for displayed rounded results.
- Verify schedule lengths, valid sampled indices \(1\ldots T\), broadcast shapes, \(\bar\alpha_0=1\), finite values, monotone cumulative retention, and the endpoint retention value.
- Check the last-step no-noise branch with an explicit controlled fixture rather than relying on visual inspection.
- For a short stochastic toy sample run, verify finite output and deterministic repeatability under the same seed; do not equate this with sample-quality validation.
- Every displayed trajectory comes from the declared process/model. Browser sliders must update actual quantities, not decorative crossfades.
- Reset restores both images and numerical state. Rapid slider changes do not show stale results. Static exports preserve the key instructional state.

#### Visual and teaching completeness

- At least eight application-led opening scenes, with real assets and meaningful input/output labels.
- All 60 logical scenes exist, and the build manifest records the planned 158 states or a documented justified revision.
- Figures, code, and equations remain readable at the course’s actual presentation aspect ratio and projector scale.
- The final-state reading export remains coherent without requiring the reader to remember animation order.
- Training and sampling diagrams are separately executable conceptual paths, with learned/fixed components and random inputs identified.
- All referenced images load from stable local assets in the exported deck; credits remain visible and full provenance is recorded.
- Any unavailable genuine comparison is honestly replaced with an explicitly labelled instructional illustration, never an invented experimental result.
- The presenter notes include the compressed route, the full route, expected student answers, likely confusions, and transitions to the next scene.

Finish by reporting the produced lecture/companion/demo/export paths, the chosen route durations, actual logical-scene/build counts, what was executed, and any specific asset or model-run limitation. Do not finish after merely restating this plan.

---

## Execution contract for the coding agent

Implement the complete lecture described below. This is a build request, not a request for another outline. Work through source inspection, mathematical validation, assets, implementation, notebooks, slide exports, and visual review. Resolve routine implementation choices yourself. Use the supplied scene sequence as the teaching specification. Split a crowded scene into additional builds or slides while preserving its content and parent scene ID. Give continuation slides unique IDs such as `scene-id-a` and `scene-id-b`, with a recorded `parent_scene_id`; never duplicate an HTML ID. Do not replace worked examples with generic explanatory prose.

The deliverable is a classroom-ready interactive lecture for Prof. Nipun Batra's ES 667 Deep Learning course at IIT Gandhinagar. Students know basic probability, MLE/MAP, familiar deep-learning losses, gradients and autograd, constrained outputs, next-token name generation, self-attention, encoder/decoder Transformers, cross-attention, ViT, CLIP, VLM connections, concept bottlenecks, detection, and segmentation. They have not yet learned the generative-model mathematics introduced here. Explain each new probability distribution and expectation before using it.

The teacher explicitly requires a substantial application opening with many actual images, slow mathematical detours, and a generous GAN treatment in Lecture I. The first lecture must teach KL divergence, reparameterization, and ELBO in its main teaching path. Lecture II must reconstruct diffusion from its training examples and sampling algorithm before assembling text-to-image systems. The full specification contains more material than an unhurried 80-minute session. Preserve the approximately 120-minute full route and mark the supplied approximately 90-minute route. A route is a navigation choice over a complete lecture, never deletion of the required content.

### 1. Inspect and extend the existing lecture system

Read applicable repository instructions and inspect the actual checkout with `rg` and `rg --files`. Identify the lecture repository, source layout, build commands, and export tools before editing. The course schedule and lecture content may live in different repositories. Do not assume a particular working directory or create a second competing lecture at a guessed location.

Required style references:

- Course sequence: https://nipunbatra.github.io/dl-2026/schedule.html
- Name-generation lecture: https://nipunbatra.github.io/attention/part1.html
- From Attention to Applications: https://nipunbatra.github.io/dl-teaching/interactives/from-attention-to-applications/
- CLIP lecture: https://nipunbatra.github.io/attention/clip/
- VLM lecture: https://nipunbatra.github.io/attention/from-clip-to-vlm/
- VLM PDF: https://nipunbatra.github.io/attention/from-clip-to-vlm/from-clip-to-vlm.pdf
- Concept-bottleneck notes: https://nipunbatra.github.io/cbm/notes.html

Inspect at least an application sequence, a mathematical sequence, a training sequence, and the presentation/export controls. Large single-file HTML lectures may exceed a web reader's limit; use their public HTML/source repository or PDF instead. An oversized response does not establish that the material is unavailable.

Copy the teaching grammar: concrete task, question before explanation, progressive diagram, exact arithmetic, training, generation, then a return to the opening application. In the attention lecture, a fixed sentence anchors multiple tasks. In CLIP, individual dot products and normalization precede the complete loss. In VLM, the same image remains while the query and information flow change. Use the specified running examples in that spirit. Write original slide text and original mathematical figures.

Current VLM styling provides these useful reference tokens. Recheck the source before integrating; preserve established conventions if they differ:

```css
--paper: #F7F8FA;
--ink: #202633;
--muted: #566171;
--line: #D9DEE6;
--vision: #3478E5;
--language: #8B5BB5;
--bridge: #178F82;
--connector: #B98224;
--loss: #D45555;
```

Use blue for images/data, purple for text where present, teal for latent/probabilistic transformations, amber for sampled randomness/time, and red for loss/gradient highlights. Put a visible label next to every color role. Treat these as a coordinated palette, not a rule that every object must use a distinct color. Prefer Avenir Next, Segoe UI, and system sans-serif, with SF Mono/Menlo or an available equivalent for code. Use only redistributable fonts in bundled exports.

Use the current 1280 by 720 classroom stage, scaled uniformly to the viewport. Keep the reading view responsive. Preserve generous margins and room for a bottom source line. Target 24–30 px body text, 36–44 px headings, and readable equation labels. A source footer can be smaller. Do not reduce technical text below classroom readability to force a fit. Move derivation details into successive builds and narration.

The existing VLM reveal API has used `window.lectureReveals` with plans/apply/reset/move/finish/toggle/status; CLIP has used `window.clipLesson`. A generic attention exporter has expected `window.AT.present`. These are different interfaces. Inspect and adapt the selected exporter instead of assuming that one global exists everywhere. A thin adapter is preferable to replacing the working lecture runtime.

If no relevant checkout exists, create a self-contained folder using plain HTML/CSS/JavaScript, SVG/canvas, and a locally bundled math renderer. Use the same reference style and provide an ordinary static-server launch command. The lecture must work without a backend, authentication, a paid API, or a classroom GPU. Model-training scripts run separately. Do not publish or deploy as a consequence of building these artifacts.

### 2. Learning design and visual requirements

1. Begin with the specified 8–10 application scenes. Each must contain a substantial visual example, not an icon or a title card. Use one main image, a before/after pair, or an aligned condition/output strip. Reserve roughly two-thirds of each application frame for evidence.
2. For each application use three teaching beats where appropriate: what is supplied, what could be generated, and what property makes the problem difficult. Reveal the output after the student prediction question. A source caption may identify the model; do not require students to know the family name yet.
3. Make image selection purposeful. Show variability for generation, a missing region for completion, correspondence for paired translation, and structure preservation for conditioning. Include model failures or competing plausible answers when they teach uncertainty.
4. Reuse several opening images when the mechanism is finally explained. Supply an explicit map from application scene IDs to their later explanatory scenes.
5. Use original SVG for distributions, probability bars, computation graphs, gradients, network blocks, matrix arithmetic, and tensor-shape traces. Never use an image generator for exact equations, charts, or architecture diagrams.
6. Keep no more than three meaningful visual panels on a frame unless a sample grid itself is the teaching object. Avoid small decorative cards, stock robot imagery, busy gradients, unexplained latent-space arrows, and paragraphs pasted over photographs.
7. Present equations in three passes: a picture and verbal question, the equation with symbols introduced, then numbers or code. Highlight only the term under discussion. Define the measure/conditioning of every new expectation.
8. A reveal adds one idea or arithmetic operation. It should not expose an entire dense section. Preserve the previous visual positions while adding a new branch, tensor, factor, or gradient.
9. For each trained component explicitly show its input, output, learnable parameters, target, and loss. For generation identify the supplied condition, sampled randomness, repeated operation, and output. Frozen parameters and changing activations must remain distinct.
10. Make discussion answers hidden until requested. Instructor narration should explain misconceptions and likely student questions, not merely restate the visible slide. Give each mathematical scene a meaningful teaching note.
11. Keep mathematical notation consistent across picture, equation, code, and narration. Distinguish an observed sample from a random variable when it matters. Use natural logarithms and nats throughout unless an explicitly labelled extension changes the base.
12. Avoid claims that an attractive sample validates a model, that a latent dimension is necessarily semantic, that reconstruction proves useful generation, or that visual realism makes a generated scientific quantity a measurement.

### 3. Runtime and navigation specification

Use a structured scene registry or its equivalent in the existing runtime. Every scene needs a stable ID, section, title, learning goal, ordered build states, visual description, narration, question/answer when applicable, source/asset IDs, and route membership. Keep the source of displayed numbers separate from styling.

Required controls:

- Reading mode with completed diagrams, expandable calculations, and notes.
- Presentation mode compatible with the existing `?present` convention, stable scene/build deep links, next/back, section overview, and visible progress.
- Keyboard navigation that respects focused inputs. Escape exits overlays/presentation. An overview lets the teacher jump directly to KL, ELBO, reparameterization, GAN training, or the appropriate Lecture II section.
- Full and shortened teaching routes. Show scene count and build progress separately; do not confuse 60 logical scenes with 150 exported reveal states.
- Reset/step/play/pause controls for each simulation. Reset must restore the exact seed and inputs. Pause simulations when leaving their scene.
- Accessible names, adequate contrast, and reduced-motion support. Motion must never be the only way to understand a mathematical operation.
- Every interactive has a meaningful static fallback for reading and PDF. A PDF must show a defined example state, not an empty canvas or an instruction to click.

Prefer deterministic seeded JavaScript for analytic demos. If a demo uses a trained network, export real weights/checkpoints or recorded outputs with provenance. A parameter slider may recompute a mathematical function; it must not pretend to retrain an image model instantly. If showing saved checkpoints, label the control as checkpoint selection. Avoid running expensive training or a remote image-generation call during classroom presentation.

An implementation may expose a small export adapter with `listScenes`, `getBuildCount`, `goTo`, `setBuild`, `finish`, and `ready`. These names are a proposed internal interface, not a claim about an existing runtime. Map it to whichever tested interface the repository provides. Wait for `document.fonts.ready`, decoded images, completed math rendering, and a settled scene before capture.

### 4. Source and asset handling

Use the reference/source ledger below as a starting point. Resolve and inspect actual assets before embedding. Download permitted assets into the lecture folder so the classroom presentation does not depend on third-party hotlinks. Keep the source image at a useful resolution and use an appropriately sized derivative in HTML. Retain animation/video posters and playback controls.

Create a machine-readable asset manifest with: `id`, `local_path`, `source_page`, `asset_url` if available, `authors_or_owner`, `paper_or_dataset`, `figure_or_caption`, `license_or_usage_note`, `retrieved_date`, `transformations`, `evidence_type`, and `used_in_scene_ids`. Use a real acquisition date. Evidence types should distinguish published model output, local trained-model output, deterministic simulation, and illustrative teaching image.

Cite each reused image/plot next to the visual, and each technical claim in its notes/source list. Source links need to be usable in the exported PDF and reading mode. If adapting MIT material, follow the attribution requested on the MIT course page, including the named course authors and course title. A code repository's license does not automatically license every third-party photograph shown by that project. Use original author assets with appropriate attribution, openly licensed images, or independently produced teaching examples when reuse is unclear. Do not erase watermarks or authorship credits.

Do not copy entire MIT slides or Lilian Weng article passages. Use their pedagogical ideas and links, and create this course's own explanations and diagrams. Validate equations against the original papers and independent calculations. Teaching resources may simplify probability, gradients, and disentanglement; retain the precise qualifications in this specification.

If an exact published asset is inaccessible, choose another inspected primary-source asset for the same teaching goal. If none can be used, create a clearly labelled conceptual example, record that substitution, and keep the source claim narrow. Never label an AI-created example as an output from a named published model or a real lab experiment. A missing asset should not become a broken image or an unexplained placeholder.

### 5. Required output package

Use the lecture-specific folder name and title. A comparable existing repo structure is acceptable if documented. Provide:

1. A complete interactive HTML lecture and its editable source/assets.
2. A reading/notes view containing all instructor explanations and worked derivations, with stable links to scenes.
3. A completed-scene PDF with one completed view per logical scene, plus an all-builds PDF with every meaningful reveal state. Both default exports cover the full lecture and appendix regardless of the route currently selected in the browser. Clearly identify appendix pages and show logical scene/build identifiers unobtrusively. Completed views may need an extra continuation page when the final state intentionally omits earlier derivation lines. If the exporter rasterizes the stage, explicitly add PDF link annotations for citations/navigation or use a print route that preserves working links. Verify an external source link and an internal section link. Describe a raster PDF as raster; keep accessible selectable text in HTML/notes.
4. The specified runnable PyTorch notebook(s) or scripts, including deterministic toy examples, small-model training, and artifact export. Separate essential CPU demonstrations from optional GPU experiments. Include cached illustrative outputs so the lecture is usable immediately.
5. A concise two-page recap with the core notation, algorithms, and misconceptions. Build this after the lecture is complete.
6. An asset/source manifest and bibliography, plus a requirements-to-scene map covering every required topic.
7. A README with verified setup/build/export commands, full/short route timings, controls, data/checkpoint provenance, and exact output paths.
8. A brief QA record containing mathematical checks, executed notebook checks, browser checks, PDF page counts, and unresolved limitations if any. Do not claim that an unexecuted check passed.

Use descriptive relative paths such as `generative-modelling-i/` or `generative-modelling-ii/`, with `assets/`, `src/`, `notebooks/`, `exports/`, and `scripts/` as appropriate. Keep each lecture independently buildable. If the counterpart lecture is absent, include the needed recap and local utilities. If it is present, reuse its established notation and assets deliberately without introducing a fragile dependency on another working directory.

### 6. Verification and completion

Compute every displayed numerical example from code and compare it with the equations. Use meaningful tests for mathematical semantics and interactions: loss reductions, gradient paths, schedule endpoints, deterministic resets, and sampler indexing. Do not write tests that merely duplicate a hard-coded label.

Render and inspect every logical scene in its final state. Capture all reveal states automatically and inspect the dense derivations, diagrams, images, and code frames individually. Check at the intended 1280 by 720 stage and at a standard browser viewport. Reject missing fonts/images, broken math, clipped legends, intersecting arrows/text, labels outside their objects, empty canvases, unreadably small expressions, and frames whose content overflows. Repair the layout by splitting content, not by hiding the overflow.

Check keyboard navigation, every demo's reset/play/step behavior, section jumps, deep links, full/short routes, reading mode, and both PDF exporters. The loss displayed beside a visual must come from exactly that visual's values. Replaying a saved checkpoint must identify it accurately. Report expensive training runs separately from lightweight checks.

Finish by providing the working local lecture, both PDFs, notebooks, and a concise explanation of what was built and verified. Continue to fix concrete remaining defects. Do not end after an implementation plan, a scaffold, an image gallery, or a deck full of placeholder diagrams.

---

## Pedagogical references and how to use them

Read the following selected resources while implementing. The links are durable references; use current primary papers and source files to resolve any ambiguity. The storyboard in this export is the controlling teaching sequence. The sources support the content and visual approach, but their complete slide text or article structure should not be reproduced.

| Reference | Intended use in this lecture pair |
|---|---|
| [Nipun Batra's ES 667 schedule](https://nipunbatra.github.io/dl-2026/schedule.html) and the linked attention/CLIP/VLM lectures | Establish prerequisites and the actual classroom visual style. Reuse the course's reveal grammar, stable examples, shape labels, question/answer controls, and training/generation explanations. |
| [MIT 6.S191 Deep Generative Modeling slides](https://introtodeeplearning.com/slides/6S191_MIT_DeepLearning_L4.pdf) and [course page](https://introtodeeplearning.com/) | The inspected 2026 deck has 84 image-based pages. Study the application opening, incremental autoencoder/VAE computation graph, latent-space visuals, and repeated discriminator/generator updates on fixed distribution axes. Follow the course's attribution instructions for any reused/adapted material. The page's publication date and the slide's teaching date may differ. |
| [Lilian Weng: From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/) | A reading companion for the transition from reconstruction to latent-variable inference. Use it to anticipate where readers lose the connection between the encoder distribution, prior, ELBO, and sampling. Validate the mathematical presentation against Kingma and Welling. |
| [Lilian Weng: From GAN to WGAN](https://lilianweng.github.io/posts/2017-08-20-gan/) | A reading companion for distribution comparison and GAN training difficulties. Keep KL, JS, and Wasserstein conceptually distinct. The mandatory lecture uses one unambiguous real/fake convention and the practical non-saturating generator loss. The optimal-discriminator/JS and WGAN material is an appendix extension. |
| [Lilian Weng: What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) | A map connecting forward corruption, reverse transitions, training, conditioning, and efficient generation. The main lecture carefully reconstructs DDPM; sampler families, score-based formulations, and flow matching are identified separately. |
| [Stanford CS236: VAE notes](https://deepgenerativemodels.github.io/notes/vae/) | A second educational reference for the marginal-likelihood difficulty and variational lower bound. Derive this export's small examples independently and show the missing algebraic steps. |
| [Stanford CS236: GAN notes](https://deepgenerativemodels.github.io/notes/gan/) | Verify the relationship among the generator's induced distribution, discriminator, optimization objectives, and conditional extensions. |
| [UvA Deep Learning: Deep Autoencoders](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial9/AE_CIFAR10.html) | An example of connecting reconstruction pictures, latent dimensions, code, checkpoints, and evaluation. Build the specified simple PyTorch teaching code rather than importing a large training framework merely because this notebook uses one. |

These resources inspire the exposition. The formulas, numerical examples, implementations, and figures in the new lecture must agree with the primary papers listed in its mathematical reference section. In particular, retain these distinctions:

- A reparameterization supplies a pathwise gradient estimator through a deterministic transformation of parameter-independent noise; it is not a universal claim that every stochastic computation is undifferentiable.
- The exact ELBO is a lower bound on log evidence. One Monte Carlo realization is an estimator and need not itself respect that bound.
- The Gaussian prior and diagonal approximate posterior do not guarantee interpretable or disentangled latent coordinates.
- Discriminator label conventions vary across resources. In this lecture, `D(x)` consistently means probability of a real sample, while executable code uses logits.
- The unweighted DDPM noise-prediction objective is related to a variational objective with timestep weights; it is not obtained by dropping all weights as if only an additive constant changed.
- Image-generation applications may use related but different architectures and training methods. Do not present the gallery as evidence that all examples use the exact toy VAE, vanilla GAN, or DDPM implementation taught here.

Each direct source figure needs a local caption and source link; each original diagram should be labelled as an authored teaching diagram in the notes. Keep a record of figure substitutions. Revisit the application images after the corresponding mechanism is explained so the opening gallery becomes part of the conceptual argument.

---

## Image acquisition and application-to-explanation map

Use these primary-project assets to make the opening concrete. Inspect each actual image, preserve its comparison labels, and acquire local copies when permitted. The asset keys below are independent of scene IDs; fill `used_in_scene_ids` in the manifest after implementation. When a figure does not document a prompt, seed or paired reference, narrow the caption to what it does establish.

| Asset key | Source page and exact selection | Evidence and teaching use | Provenance detail |
|---|---|---|---|
| `ddpm-samples` | [DDPM project](https://hojonathanho.github.io/diffusion/); [unconditional sample mosaic](https://hojonathanho.github.io/diffusion/assets/img/denoising_diffusion_all.png) | Published samples for Scene 01 and later sampler discussion | These are published outputs. A locally trained MNIST denoiser has its own separately recorded checkpoint and sample provenance. |
| `ldm-text-layout` | [Latent Diffusion project](https://ommer-lab.com/research/latent-diffusion-models/); [Figure 7](https://ommer-lab.com/wp-content/uploads/2022/08/article-Figure7-1-1024x863.png); [Figure 17](https://ommer-lab.com/wp-content/uploads/2022/08/article-Figure17-1.png) | Prompt-conditioned examples for Scenes 02–03 and latent/text explanations | Choose a subset whose actual caption establishes the intended condition. Do not invent same-seed control for a published gallery. |
| `gligen-boxes` | [GLIGEN project](https://gligen.github.io/); [grounded generation figure](https://gligen.github.io/images/groundtext2img.png) | Optional spatial arrangement comparison in Scene 03 or a control gallery drawer | Keep box labels and text conditions visible. Grounding is a specific conditioning mechanism, not proof of perfect compositional control. |
| `repaint-completion` | [RePaint repository](https://github.com/andreas128/RePaint); [static results](https://user-images.githubusercontent.com/11280511/150803812-a4729ef8-6ad4-46aa-ae99-8c27fbb2ea2e.png) | Missing-region completion for Scene 04, replayed in Scene 54 | Preserve known region and mask identity. Record whether panels are distinct samples or distinct input images. |
| `sdedit-strokes` | [SDEdit project](https://sde-image-editing.github.io/); [stroke-based generation](https://sde-image-editing.github.io/images/stroke_based_generation.jpg); [stroke editing](https://sde-image-editing.github.io/images/stroke_edit.jpg) | Stroke-guided image generation/editing for Scene 05 | Explain which image or strokes initialize the editing process. Keep this preview separate from the unconditional DDPM code path. |
| `sr3-superresolution` | [SR3 project](https://iterative-refinement.github.io/); [super-resolution comparison](https://iterative-refinement.github.io/images/super_res_examples.png) | Low-resolution condition, baseline and generated detail for Scene 06 | Only call a high-resolution image ground truth when the source provides that pairing. Generated detail is a plausible hypothesis. |
| `controlnet-edges`, `controlnet-pose` | [ControlNet repository](https://github.com/lllyasviel/ControlNet); [edge example](https://raw.githubusercontent.com/lllyasviel/ControlNet/main/github_page/p1.png); [pose example](https://raw.githubusercontent.com/lllyasviel/ControlNet/main/github_page/p12.png) | Spatial control for Scene 07, with modality labels | The code's Apache license does not by itself establish rights for every source photograph in the README. Preserve project attribution and verify the selected asset. |
| `dreambooth-subject` | [DreamBooth project](https://dreambooth.github.io/); [personalization teaser](https://dreambooth.github.io/DreamBooth_files/teaser_static.jpg) | Reference subject images and generated new contexts for Scene 08 | Distinguish subject-specific model adaptation from ordinary inference-time prompting. The small classroom denoiser is not a DreamBooth implementation. |
| `diffusionsat-reference` | [DiffusionSat project](https://www.samarkhanna.com/DiffusionSat/); [sample figure](https://www.samarkhanna.com/DiffusionSat/static/images/diffusion_sat_samples.png); [super-resolution figure](https://www.samarkhanna.com/DiffusionSat/static/images/diffusion_sat_superres.png) | Metadata-conditioned imagery and scientific-task questions for Scene 09 | The website's template credit does not establish a license for every figure. Keep published results separate from a prospective lab task. |
| `mnist-run` | Acquire the dataset through the documented local data loader and run the compatible teaching model | Clean/noisy/target/prediction/state/estimate panels throughout Scenes 12–42 | Store selected example ID, normalization, seed, checkpoint and exact noise schedule. Computed forward corruption needs no trained model; predicted-noise and generated-sample claims do. |

### Revisit the images when students can explain them

| Opening | Later explanatory scene(s) | Question resolved |
|---|---|---|
| Generated image mosaic, Scene 01 | 21–39 | What target can be constructed during training, and what is repeated during sampling? |
| Prompted samples, Scenes 02–03 | 44–53 | Where does the text enter, what are Q/K/V, and how does guidance change the prediction? |
| Completion, Scene 04 | 54 | How do the known pixels and the mask enter a particular conditional sampling procedure? |
| Stroke editing, Scene 05 | 46 and a short note after 54 | Why can image-to-image generation begin from an encoded or corrupted input, while ordinary text-to-image generation begins from noise? |
| Super-resolution, Scene 06 | 45, 54–55 | What is observed, what is inferred, and which model has been trained on this condition? |
| Spatial control, Scene 07 | 47–50, 54 | What differs between token conditioning and a spatial control representation? |
| Personalization, Scene 08 | 45–46 and an optional note | Which parameters or representations were adapted before the prompt was used? |
| Satellites, Scene 09 | 54–55 | What evidence would establish adherence to observations or metadata for this task? |

Add primary-source entries for SDEdit, DreamBooth and GLIGEN to the bibliography when their images are selected: [SDEdit](https://arxiv.org/abs/2108.01073), [DreamBooth](https://arxiv.org/abs/2208.12242), and [GLIGEN](https://arxiv.org/abs/2301.07093). Optional video/outpainting examples belong in a separate gallery drawer with their own inspected source, local playback asset and static fallback. The core mathematical story remains the specified DDPM-to-conditioned-latent-diffusion progression.
