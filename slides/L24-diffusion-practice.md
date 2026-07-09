---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# Diffusion II · Stable Diffusion in Practice

## Lecture 24 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar*

---

# The whole lecture, in one idea

L23 gave you a denoiser that turns noise into an image — but it draws *whatever* it likes, and it does so painfully slowly. Three engineering fixes close both gaps, and the thing you get out the other side has a name.

<div class="keypoint">

**Three fixes turn a toy denoiser into Stable Diffusion.**
(1) **Cross-attention** lets the U-Net *read a text prompt*. (2) **Classifier-free guidance** makes that prompt actually *stick*. (3) **Latent diffusion** runs the whole loop in a tiny VAE latent, so it's *cheap*. Then **DDIM** cuts 1000 steps to ~25 with the *same weights*.

</div>

You built $\epsilon_\theta(x_t, t)$ in L23. Today we answer the two questions that lecture left open: **how do I steer it to a prompt, and how do I make it fast enough to ship?**

$$\underbrace{\text{cross-attention}}_{\text{steer}} \;+\; \underbrace{\text{CFG}}_{\text{amplify}} \;+\; \underbrace{\text{latent diffusion}}_{\text{cheapen}} \;+\; \underbrace{\text{DDIM}}_{\text{fewer steps}} \;=\; \textbf{Stable Diffusion}$$

---

# How today connects to the rest of the course

<div class="columns">
<div>

**Everything today is a part you already own:**
- **L13 / L20** — the **attention** block; today it becomes *cross*-attention (image ↔ text)
- **L19 / L20** — a frozen **CLIP** text encoder supplies the prompt vector
- **L21** — the **VAE**; its latent is where we now run diffusion
- **L23** — the **DDPM** $\epsilon$-prediction loop we are about to condition and accelerate

</div>
<div>

**Where it goes next:**
- **L25** — efficient inference (the same "run it twice" cost that CFG pays)
- the whole generative-model landscape — DiT, Sora, SD3 are this recipe with bigger backbones

</div>
</div>

<div class="insight">

Nothing here is a new primitive. It is **four familiar blocks snapped together** — the lecture is the assembly diagram.

</div>

---

<!-- _class: section-divider -->

## Part 1 · Why naive pixel diffusion isn't enough

---

# Recap: what L23 handed us

The denoiser is a single network $\epsilon_\theta(x_t, t)$ trained to predict the noise added to a clean image. Sampling is a loop:

<div class="math-box">

start from pure noise $x_T \sim \mathcal N(0, I)$; for $t = T, \dots, 1$: predict $\hat\epsilon = \epsilon_\theta(x_t, t)$, subtract a little of it, add a little fresh noise → $x_{t-1}$. Out pops $x_0$.

</div>

It works on MNIST and toy 2D data. But two things are missing before it can draw "an astronaut riding a horse."

---

# Two dealbreakers of pixel-space DDPM

<div class="insight">

**Dealbreaker 1 · you can't steer it.** $\epsilon_\theta(x_t, t)$ takes only the noisy image and the timestep. There is **no input for a prompt** — it samples from the whole training distribution, not from "the horses."

**Dealbreaker 2 · it's far too expensive.** A $512\times512\times3$ image is ~800k numbers, and the loop runs the network ~**1000 times** per image. That is minutes of GPU per picture, on every pixel.

</div>

Fix the steering with **conditioning + guidance**; fix the cost with **latent diffusion + DDIM**. Four objects, four slides' worth of ideas.

---

# The plan · four objects you can name

<div class="keypoint">

Do not think of today as four algorithms. Think of it as **four boxes bolted onto the L23 loop**:

1. **Cross-attention** — a wire from a text vector into every U-Net block *(steer)*
2. **Classifier-free guidance** — run the denoiser twice and extrapolate *(amplify)*
3. **Latent diffusion** — do all of the above inside a VAE latent *(cheapen)*
4. **DDIM** — a smarter sampler that skips 95% of the steps *(accelerate)*

</div>

Snap 1–3 together and you have literally drawn the Stable Diffusion architecture. Add 4 and it runs in a second.

---

<!-- _class: section-divider -->

## Part 2 · Conditioning — cross-attention on the prompt

---

# Give the denoiser a third input

We want $\epsilon_\theta(x_t, t, c)$ — predict the noise **given a condition** $c$ (here, a text embedding). The only design question is *how $c$ gets in*.

<div class="keypoint">

Three standard ways to inject $c$: **concatenate** it to the input channels, **add** it to the time embedding, or **cross-attend** to it. Stable Diffusion picks the last — it is the only one that lets *different image regions listen to different words*.

</div>

The prompt is turned into a sequence of token vectors by a frozen CLIP text encoder — a $77 \times 768$ tensor. Cross-attention is how the image reads it.

---

# Cross-attention · an image listens to text

<div class="insight">

**Analogy · a team of painters.** Each painter owns one small patch of canvas. Before a brushstroke, each asks: *"which word in the prompt matters for **my** patch?"* The patch near the top listens to "hat"; the furry patch listens to "cat." One prompt, many localized instructions.

</div>

Mechanically, per image region:
1. **Queries $Q$** come from the U-Net's spatial features — each region's "question."
2. **Keys $K$ and values $V$** come from the CLIP text vectors — the words' "answers."
3. **Score** $\text{softmax}(QK^\top/\sqrt d)$ — how relevant each word is to this region.
4. **Read** the weighted sum of $V$ → the instruction used to denoise that region.

Same attention you met in L13 — only now $Q$ is the image and $K,V$ are the text.

---

<!-- _class: code-heavy -->

# Cross-attention · in code

```python
text_emb = clip_text_encoder("a cat astronaut")   # [1, 77, 768], frozen

class CrossAttention(nn.Module):
    def forward(self, image_feats, text_emb):
        Q = self.q_proj(image_feats)   # queries FROM the image
        K = self.k_proj(text_emb)      # keys    FROM the text
        V = self.v_proj(text_emb)      # values  FROM the text
        attn = (Q @ K.transpose(-1, -2)) / self.d ** 0.5
        return attn.softmax(dim=-1) @ V
```

Self-attention mixes *pixels with pixels*. **Cross**-attention mixes *pixels with words* — that single swap of where $K,V$ come from is the entire text-conditioning trick.

---

# Cross-attention · worked numeric

Prompt "a red square," two word-values $V_\text{red} = [1, 0]$, $V_\text{square} = [0, 1]$. A top-left pixel's query is $Q = [0.1,\, 0.9]$ (currently more square-like than red-like).

<div class="math-box">

**Scores** · $Q\!\cdot\!V_\text{red} = 0.1$, $\;\;Q\!\cdot\!V_\text{square} = 0.9$ — "square" is far more relevant.

**Softmax → weights** · $\;(w_\text{red}, w_\text{square}) \approx (0.3,\, 0.7)$.

**Read** · $0.3\,[1,0] + 0.7\,[0,1] = [\mathbf{0.3,\, 0.7}]$ — the instruction for this pixel: *a little red, mostly square.*

</div>

Do this for every pixel and every word, at every U-Net scale, and the prompt is now written across the whole feature map.

---

# Why cross-attention is the right knob

<div class="keypoint">

Because scoring is **per region**, conditioning is **spatially localized**: a cat pixel attends to "cat," a sky pixel to "sky." One prompt steers different regions differently — and you can *watch* it happen by plotting the attention map for each word.

</div>

This is not just a curiosity — those attention maps are the mechanism behind **prompt editing, DreamBooth, and Prompt-to-Prompt.** Change which words a region attends to, and you edit the image without retraining.

<div class="notebook">

**🎛 Interactive · text-to-image attention** — type a prompt and watch each token light up the pixels that listened to it during denoising. *(Interactive Lab · `interactive/src/articles/text-diffusion`)*

</div>

---

# The condition need not be text

$c$ is just "the thing you inject." Swap what it is and you get a whole family of models — same $\epsilon_\theta(x_t, t, c)$, same loop:

| Condition $c$ | Injected via | Model |
|:--|:--|:--|
| class label | concatenate / adaLN | class-conditional DDPM |
| **text prompt** | **cross-attention** | **Stable Diffusion** |
| a reference image | encoder + concat latent | img2img, inpainting |
| edges / depth / pose map | side-network features | ControlNet |

<div class="insight">

The rest of the lecture uses text, but nothing about the machinery is text-specific. **Conditioning is a socket; the prompt is one plug.**

</div>

---

<!-- _class: section-divider -->

## Part 3 · Classifier-free guidance

---

# The prompt is wired in — so why is it half-ignored?

Cross-attention *lets* the prompt influence the image, but on its own that influence is weak — samples drift toward the generic. CFG turns the influence up.

<div class="keypoint">

**One object, one formula.** Each step, run the denoiser **twice** — once *with* the prompt ($\epsilon_\text{cond}$), once with a *null* prompt ($\epsilon_\text{uncond}$) — and extrapolate along their difference. The **guidance scale $s$** is how far you walk:

$$\boxed{\;\epsilon_\text{cfg} = \epsilon_\text{uncond} + s\,\big(\epsilon_\text{cond} - \epsilon_\text{uncond}\big)\;}$$

</div>

![w:640px](figures/lec22/svg/cfg_guidance_vectors.svg)

---

# CFG · the two-compass picture

<div class="insight">

You're lost in a forest with two compasses. The **generic compass** ($\epsilon_\text{uncond}$) always points to "an average image." The **prompt compass** ($\epsilon_\text{cond}$) points toward "a cabin in the woods."

The **difference** $\epsilon_\text{cond} - \epsilon_\text{uncond}$ is the *pure pull of the prompt*, with the generic part subtracted out. CFG starts at the generic prediction and walks $s$ steps along that pull — for $s>1$, it *overshoots* the conditional prediction, exaggerating exactly what the prompt asked for.

</div>

That is the whole idea. $s$ is a dial on a single subtraction.

---

<!-- _class: code-heavy -->

# Where does the null prompt come from? · training dropout

For CFG to have an $\epsilon_\text{uncond}$ to call at inference, the *same* network must have learned an unconditional mode. You get it for free by dropping the prompt during training:

```python
def training_step(x0, prompt):
    t = sample_t()
    noise = torch.randn_like(x0)
    x_t = add_noise(x0, noise, t)

    c = null_embedding if random.random() < 0.10 else clip_text_encoder(prompt)
    return F.mse_loss(model(x_t, t, c), noise)   # one network, both modes
```

Drop ~10% of prompts to the null token; the network learns conditional *and* unconditional denoising in one set of weights. Cost at inference: **2× forward passes per step.** Benefit: **any $s$ you like, chosen at generation time.**

---

# Reading the dial

Same prompt, same seed — only $s$ changes. Adherence climbs, then colors over-saturate and detail breaks.

![w:660px](figures/lec22/svg/cfg_scale.svg)

<div class="insight">

$s=0$ ignores the prompt · $s=1$ is the plain conditional · $s>1$ overshoots to *amplify* adherence. **SD's default is $s=7.5$.** High $s$ trades **diversity for prompt-fidelity** — treat it as a hyperparameter you sweep.

</div>

<div class="notebook">

**🎛 Interactive · CFG scale visualizer** — slide $s$ from 0 to 30 and watch adherence rise then artifacts appear. *(Interactive Lab · `interactive/src/articles/cfg-scale-visualizer`)*

</div>

---

# Practice problem 1

<div class="popquiz">

**Practice problem 1.** At one denoising step the U-Net returns (toy 2D)
$$\epsilon_\text{uncond} = [+0.2,\, -0.1], \qquad \epsilon_\text{cond} = [+0.5,\, +0.3].$$
Compute $\epsilon_\text{cfg}$ for $s = 0,\ 1,\ 3,\ 7.5$. What is happening to the vector as $s$ grows, and what does that do to the image?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 1

The difference vector is $\Delta = \epsilon_\text{cond} - \epsilon_\text{uncond} = [+0.3,\, +0.4]$. Then $\epsilon_\text{cfg} = \epsilon_\text{uncond} + s\,\Delta$:

![w:560px](figures/lec22/svg/cfg_arithmetic.svg)

| $s$ | $\epsilon_\text{cfg}$ | meaning |
|:-:|:-:|:-:|
| 0 | $[+0.20,\, -0.10]$ | unconditional — prompt ignored |
| 1 | $[+0.50,\, +0.30]$ | plain conditional |
| 3 | $[+1.10,\, +1.10]$ | overshoot ~2× past conditional |
| 7.5 | $[+2.45,\, +2.90]$ | strong overshoot |

<div class="keypoint">

Larger $s$ pushes the predicted noise **further along the prompt direction** $\Delta$. The step then subtracts a bigger, more "on-prompt" noise → the image lands further toward what the prompt described (until, past $s\!\approx\!12$, it over-saturates).

</div>

---

# CFG in practice · the negative prompt

The null prompt in CFG only means "point away from *nothing*." Replace it with a **real** prompt and you get the most-used knob in every SD front-end — you point away from something specific.

<div class="keypoint">

**Same formula, better baseline.** Swap $\epsilon_\text{uncond}$ for $\epsilon_\text{neg}$ — the noise predicted for a *negative* prompt like "blurry, extra fingers, watermark":

$$\epsilon_\text{cfg} = \epsilon_\text{neg} + s\,\big(\epsilon_\text{cond} - \epsilon_\text{neg}\big)$$

Now $s$ pushes **toward** what you asked for **and away from** what you never want.

</div>

Worked: keep $\epsilon_\text{cond} = [+0.5,\, +0.3]$, and let the negative prompt predict $\epsilon_\text{neg} = [-0.1,\, +0.4]$. The pull becomes $\Delta = \epsilon_\text{cond}-\epsilon_\text{neg} = [+0.6,\, -0.1]$ — steeper *away* from "blurry" than the plain null baseline gave. It costs nothing extra: the second pass CFG already runs is simply **conditioned on the negative prompt instead of the null token.**

---

<!-- _class: section-divider -->

## Part 4 · Latent diffusion — the compute win

---

# Most pixels are redundant · zip, then diffuse

<div class="keypoint">

A patch of blue sky is hundreds of near-identical pixels. Asking the network to predict "blue follows blue" a thousand times is wasted work. A pretrained **VAE has already absorbed** that perceptual redundancy — its latent keeps only the *interesting* dimensions.

</div>

<div class="insight">

**Like emailing a 100 MB document.** You don't attach it raw — you zip it to 1 MB, send that, and unzip at the other end. Latent diffusion zips the image with a VAE, runs the *entire* diffusion loop in the small zipped space, and decodes **once** at the very end.

</div>

---

# Run the loop in the latent, decode once

<div class="math-box">

**Text-to-image, the runtime path:**
1. Start from **latent** noise $z_T \sim \mathcal N(0,I)$, shape $64\times64\times4$.
2. For each step: U-Net predicts noise on $z_t$, cross-attending to the prompt, with **CFG** — all in latent space.
3. After ~25 steps, the final latent $z_0$ is handed to the **VAE decoder** → the $512\times512$ image.

</div>

The VAE **decoder runs exactly once**; the encoder isn't used for text-to-image at all (there's no input image to encode — it *is* used for img2img and during training). Every expensive U-Net pass now works on 16k numbers, not 800k.

---

# img2img · start the loop from a real image

Text-to-image starts from pure noise. **img2img** starts from *your* image — this is where the VAE encoder finally earns its keep at inference time.

<div class="math-box">

1. **Encode** the input image → latent $z_0$.
2. **Noise it partway**: jump only to $z_\tau$ for $\tau = \text{strength}\times T$, not all the way to pure noise.
3. **Denoise** from $z_\tau$ with the prompt, as usual → a new image that keeps the original's layout.

</div>

<div class="insight">

**Strength is a "how much to repaint" dial.** $\text{strength}=0.3$ re-noises only 30% of the way, so the output stays close to the input (recolour, restyle). $\text{strength}=0.9$ erases almost everything, keeping only rough composition. **Inpainting** is the same trick with a mask — noise and re-denoise *only inside the hole*, freezing every pixel outside it.

</div>

A smaller strength also runs *fewer* steps — you start at $z_\tau$, not $z_T$ — so img2img is strictly cheaper than a full text-to-image generation.

---

# The dimension math

<div class="columns">
<div>

**Pixel space**
$512 \times 512 \times 3 = \mathbf{786{,}432}$

**Latent space**
$64 \times 64 \times 4 = \mathbf{16{,}384}$

</div>
<div>

**Saving**
$$\frac{786{,}432}{16{,}384} = \mathbf{48\times}$$
fewer numbers for the U-Net to denoise, *every single step.*

</div>
</div>

<div class="keypoint">

This one change — diffusing in a VAE latent instead of on pixels — is the **single reason Stable Diffusion runs on a consumer GPU** rather than a datacenter. Perceptual quality is preserved because the VAE decoder restores the fine texture the latent didn't need to track.

</div>

---

# Practice problem 2

<div class="popquiz">

**Practice problem 2.** SD compresses $3\times512\times512$ pixels to $4\times64\times64$ latents. **(a)** Confirm the compression ratio. **(b)** If DDIM also cuts sampling from 1000 steps to 25, roughly how much cheaper is the *whole* generation than pixel-space DDPM? **(c)** Why doesn't throwing away 48× of the dimensions wreck image quality?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 2

<div class="math-box">

**(a)** $786{,}432 / 16{,}384 = \mathbf{48\times}$ fewer dimensions per U-Net pass.

**(b)** Latent gives ~$48\times$ less work *per step*; DDIM gives $1000/25 = 40\times$ *fewer steps*. Multiplied, the full loop is on the order of $48 \times 40 \approx \mathbf{2000\times}$ cheaper than naive pixel-space DDPM (ballpark — real speedups depend on U-Net cost scaling).

**(c)** The discarded dimensions were **perceptually redundant** — texture and local smoothness the VAE learned to reconstruct. Diffusion only has to model the *high-level, high-entropy* structure in the latent; the decoder paints the detail back in.

</div>

<div class="keypoint">

Two independent savings — smaller space **and** fewer steps — stack multiplicatively. That product is why "type a prompt, get an image in a second" is possible.

</div>

---

<!-- _class: section-divider -->

## Part 5 · DDIM — fewer, deterministic steps

---

# Why does DDPM need 1000 steps?

<div class="keypoint">

A **sampler** is just the *rule for how big a denoising step to take.* DDPM adds tiny Gaussian noise at each of $T$ forward steps; to keep each increment small enough that the reverse step is also (approximately) Gaussian, $T$ has to be large — 1000 is typical.

</div>

Small per-step noise makes each inverse problem easy for the network — but it means the reverse loop is **1000 forward passes** of a ~1B-parameter U-Net. That is the cost DDIM attacks, *without touching the trained weights.*

---

# DDIM · walk the path in big strides

<div class="insight">

**Foggy hill.** DDPM walks down a rocky hill in fog — 1000 tiny careful steps, because it re-rolls random noise at each and can only trust one step ahead. DDIM notices that with a good $\epsilon_\theta$ you can *see the whole path*, so it takes ~25–50 confident **deterministic** strides straight down.

</div>

The trick: reinterpret the forward process as **non-Markovian** with the *same* marginals $q(x_t \mid x_0)$, but a reverse update whose injected noise is set to **zero**. The path becomes a smooth trajectory you can traverse on any subset of timesteps.

![w:660px](figures/lec22/svg/ddim_vs_ddpm.svg)

---

# DDPM vs DDIM · the step-count that matters

Same trained $\epsilon_\theta$, same $\bar\alpha_t$ schedule — only the *sampler's rule* changes. What that swap buys, concretely:

| | DDPM | DDIM |
|:--|:-:|:-:|
| Reverse update | stochastic (re-adds noise) | deterministic (noise $=0$) |
| Steps for good quality | ~1000 | ~25–50 |
| Same seed → same image? | no | **yes** — reproducible |
| Retraining needed? | — | **none** |
| U-Net passes / image | 1000 | ~25 |

<div class="keypoint">

At 25 DDIM steps vs 1000 DDPM steps that is a **~40× wall-clock win for free**, and the determinism is what makes fixed seeds, latent interpolation, and image inversion possible at all. Push below ~15 steps and quality drops — the trajectory is *smooth*, not a straight line.

</div>

---

# Practice problem 3

<div class="popquiz">

**Practice problem 3.** DDIM samples good images in 25 steps using the *exact same trained network* as DDPM — no retraining. But you **cannot** simply run DDPM on every 40th timestep and expect it to work. What is the property of DDIM's sampler that lets it skip steps, and which property of DDPM's forbids it?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 3

<div class="math-box">

**DDPM is Markovian and stochastic.** Each reverse step $x_t \to x_{t-1}$ samples *fresh* random noise and is only defined between *adjacent* small-gap timesteps. Skip 40 at once and the "small-noise, near-Gaussian" assumption breaks; the re-injected randomness also makes the trajectory non-reproducible, so error compounds.

**DDIM is non-Markovian and deterministic.** With the stochastic term set to zero, each step **predicts $x_0$ directly** from $x_t$ and re-noises to *any* earlier timestep. The result traces a smooth (ODE-like) path, so a *coarse* set of timesteps still lands near it.

</div>

<div class="keypoint">

Both samplers only ever need $q(x_t\mid x_0)$ and the same $\epsilon_\theta$ — that is *why no retraining is needed.* DDIM just chooses a **deterministic** update and a **sparse** timestep schedule. Result: ~$20$–$40\times$ fewer passes at matched quality.

</div>

---

# The sampler menu · DDIM is one of many

DDIM was the *first* fast sampler; once you view the reverse process as an **ODE to integrate**, a whole menu opens up. All of them reuse the *same* trained $\epsilon_\theta$ — picking one is a runtime choice, never a retraining one.

| Sampler | Idea | Typical steps |
|:--|:--|:-:|
| DDPM | stochastic, tiny steps | ~1000 |
| DDIM | deterministic, 1st-order | 25–50 |
| Euler | plain ODE step | 20–30 |
| Heun | Euler + a correction (2nd-order) | 15–25 |
| DPM++ (2M) | tuned multi-step solver | **~20** |

<div class="insight">

Higher-order solvers spend more compute *per* step so they can take *bigger* steps — fewer network calls for the same picture. **DPM++ reaches usable images in ~20 steps** and is the default in most modern front-ends. The knob trades steps against per-step cost; it never trades quality for retraining.

</div>

---

<!-- _class: section-divider -->

## Part 6 · Assemble → that's Stable Diffusion

---

# Snap the three fixes together

Cross-attention (steer) + CFG (amplify) + latent diffusion (cheapen), sampled with DDIM — drawn as one diagram, this *is* the architecture:

![w:680px](figures/lec22/svg/latent_diffusion.svg)

A frozen text encoder feeds the prompt; a U-Net denoises a latent while cross-attending to it; a VAE decoder renders the final image.

---

# The three trainable-or-frozen components

<div class="columns">
<div>

**VAE** *(frozen)*
encoder: image → $64\times64\times4$ latent
decoder: latent → image
pretrained, never updated during diffusion training.

**CLIP text encoder** *(frozen)*
prompt → $77\times768$ tokens
the diffuser just consumes these.

</div>
<div>

**U-Net diffuser** *(trainable)*
operates in the $64\times64\times4$ latent
cross-attention injects the text at every scale
~1B params (SD v1.5)

**This is the only piece you train.**

</div>
</div>

<div class="insight">

**Now it clicks.** CLIP → U-Net that cross-attends to the prompt (twice per step, for CFG), diffusing in the VAE latent → VAE decoder. Three fixes, one pipeline, one trainable network.

</div>

---

# The annotated full stack

![w:680px](figures/lec22/svg/latent_diffusion_stack.svg)

Every arrow is one of today's four objects. Text in on the left, image out on the right, the whole loop living in the small latent in the middle.

---

# The four fixes · cost vs benefit

| Fix | What it costs | What it buys |
|:-:|:-:|:-:|
| Cross-attention conditioning | a frozen CLIP encoder | text → image |
| Classifier-free guidance ($s\!\approx\!7.5$) | 2× forward passes / step | sharp, on-prompt samples |
| Latent diffusion (VAE) | a pretrained VAE | $48\times$ fewer dimensions |
| DDIM sampler | none — reuses weights | $\sim\!40\times$ fewer steps |

<div class="keypoint">

Two of the four cost *nothing to train* (they reuse frozen or already-trained parts), and DDIM is free. That is why Stable Diffusion was a re-assembly of known parts, not a new model from scratch.

</div>

---

# Practice problem 4

<div class="popquiz">

**Practice problem 4.** Take one U-Net forward pass as the unit of cost. Classifier-free guidance runs it **twice** per step. Count the passes to make one image: **(a)** pixel-space DDPM at 1000 steps with CFG; **(b)** latent Stable Diffusion with DDIM at 25 steps and CFG. **(c)** By what factor did the *step change alone* help — and does CFG change that factor?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 4

<div class="math-box">

**(a)** $1000\ \text{steps} \times 2\ (\text{CFG}) = \mathbf{2000}$ U-Net passes.

**(b)** $25\ \text{steps} \times 2\ (\text{CFG}) = \mathbf{50}$ U-Net passes.

**(c)** $2000 / 50 = \mathbf{40\times}$ fewer passes — exactly $1000/25$. The CFG factor of $2$ sits in *both* numerator and denominator, so it **cancels**: the step-count win is independent of guidance.

</div>

<div class="keypoint">

CFG doubles the *constant*; DDIM cuts the *count*. And this $40\times$ rides **on top of** latent diffusion's $48\times$ smaller passes — the two savings multiply while the guidance overhead tags along unchanged. That product is the "prompt in, image in a second" budget.

</div>

---

# One slide on DiT · the U-Net's successor

Peebles & Xie (2023) replace the U-Net with a **Transformer** on the latent:

1. **Patchify** the noisy latent into a grid of patches (like ViT, L20).
2. **Tokens** — flatten each patch → a sequence.
3. **Transformer blocks** — self-attention lets every patch see every other (global context, not just convolutional neighbours).
4. **Condition** on time + text via **adaLN** (adaptive LayerNorm).

<div class="insight">

**Why it matters** — Transformers scale where U-Nets plateau, and inherit the LLM toolchain (FlashAttention, tensor parallelism). DiT is the backbone of **SD3, Sora, and most 2024+ frontier image/video models.** Everything else today stays the same — only the denoiser's *internals* change.

</div>

---

# Explore it yourself

<div class="notebook">

**📓 Notebook · Stable Diffusion from a pretrained model** — load HF `diffusers`, run the sampling loop, implement the **CFG** extrapolation by hand, sweep the guidance scale, and swap DDPM for DDIM. *(ES 667 · `notebooks/24-stable-diffusion.ipynb`)*

</div>

<div class="notebook">

**🎛 Interactive · end-to-end diffusion demo** — step through forward noising, reverse denoising, and text conditioning in the browser. *(`~/git/diffusion-interactive` — the diffusion demo app)*

</div>

<div class="notebook">

**🎛 Interactive Lab explainers** — `text-diffusion` (cross-attention), `cfg-scale-visualizer` (the $s$ dial), `vae-latent-explorer` (what the latent holds), `diffusion-denoise` (the reverse loop). *(under `~/git/interactive/src/articles/`)*

</div>

---

<!-- _class: summary-slide -->

# One sentence to remember

<div class="keypoint">

**Wire a text vector into the denoiser with cross-attention, amplify it with classifier-free guidance, run the whole loop in a VAE latent, and sample it with DDIM — that assembly is Stable Diffusion.**

</div>

- **Condition** · CLIP text encoder + cross-attention in every U-Net block.
- **Guide** · $\epsilon_\text{cfg} = \epsilon_\text{uncond} + s(\epsilon_\text{cond} - \epsilon_\text{uncond})$, default $s = 7.5$.
- **Cheapen** · diffuse in a $64\times64\times4$ latent — $48\times$ fewer dimensions.
- **Accelerate** · DDIM's deterministic sampler — ~$40\times$ fewer steps, no retraining.

**Next (L25):** the model is built and steerable — now make it *cheap to serve* (KV-cache, quantization, FlashAttention).

---

<!-- _class: compact -->

# Sources & credits

<div class="paper">

This lecture reuses and adapts material from ES 667 (IIT Gandhinagar) and the primary literature:

- **Latent diffusion / Stable Diffusion** — Rombach, Blattmann, Lorenz, Esser, Ommer, *High-Resolution Image Synthesis with Latent Diffusion Models*, CVPR 2022.
- **Classifier-free guidance** — Ho & Salimans, *Classifier-Free Diffusion Guidance*, 2022.
- **DDIM** — Song, Meng, Ermon, *Denoising Diffusion Implicit Models*, ICLR 2021.
- **DiT** — Peebles & Xie, *Scalable Diffusion Models with Transformers*, ICCV 2023.
- **Pedagogical framing** — A. Ng, *Deep Learning Specialization* (concept-first, object-before-math); Hugging Face `diffusers` docs (code-first); L. Weng, diffusion blog.

Figures adapted from the ES 667 figure library (`figures/lec22/`). All source materials © N. Batra & teaching staff.

</div>

---

<!-- _class: section-divider -->

## ⭐⭐⭐ Appendix · optional depth

*Skip on a first pass — for the curious.*

---

<!-- _class: math-heavy -->

# ⭐⭐⭐ Optional · where CFG comes from

Classifier *guidance* (Dhariwal & Nichol) steers sampling with the gradient of a separately trained classifier $p(c\mid x_t)$, nudging the score by $\nabla_{x_t}\log p(c\mid x_t)$. That needs an extra noisy-image classifier — annoying.

Ho & Salimans note that Bayes gives $\nabla_{x_t}\log p(c\mid x_t) = \nabla_{x_t}\log p(x_t\mid c) - \nabla_{x_t}\log p(x_t)$, and each score is (up to scale) exactly a noise prediction: $\epsilon_\text{cond}$ and $\epsilon_\text{uncond}$. Applying guidance strength $s$ to that difference recovers, with **no classifier at all**,

$$\epsilon_\text{cfg} = \epsilon_\text{uncond} + s\,(\epsilon_\text{cond} - \epsilon_\text{uncond}),$$

which is why it is called *classifier-**free**.* The two noise predictions are the classifier's gradient in disguise. *(Full derivation: Ho & Salimans 2022, §2.)*

---

<!-- _class: math-heavy -->

# ⭐⭐⭐ Optional · the DDIM deterministic update

Writing $\bar\alpha_t$ for the cumulative signal-retention at step $t$, DDIM first predicts the clean image from the current noise estimate,

$$\hat x_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}},$$

then re-noises **directly** to any earlier timestep $s<t$ with the stochastic term set to zero:

$$x_s = \sqrt{\bar\alpha_s}\,\hat x_0 + \sqrt{1-\bar\alpha_s}\;\epsilon_\theta(x_t,t).$$

Because there is no random term, the map $x_T \mapsto x_0$ is **deterministic** and behaves like an ODE integrator — so a coarse schedule $\{T, \dots, s, \dots, 0\}$ of ~25 timesteps tracks the same trajectory 1000 fine DDPM steps would. Same $\epsilon_\theta$, same $\bar\alpha_t$ schedule, no retraining. *(Song, Meng, Ermon 2021, §4.)*
