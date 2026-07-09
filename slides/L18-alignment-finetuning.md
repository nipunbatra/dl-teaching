---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# Alignment & Fine-tuning

## Lecture 18 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar*

---

# The whole lecture, in one idea

A pretrained model is a **completer**, not an **assistant**. It knows how to continue internet text; it does not know that *you asked a question and want an answer*. Alignment closes that gap — **without** retraining the 70B weights from scratch.

<div class="keypoint">

**Nothing leaves the negative-log-likelihood world of L01.** The only thing that changes is *what we count as a good answer $y$*. First we **clone** good answers (SFT), then we learn **preferences** over answers (RLHF / DPO) — and we do it cheaply by training a tiny **low-rank** patch (LoRA).

</div>

$$\underbrace{\text{pretrained completer}}_{\text{L16–L17}} \xrightarrow{\;\text{SFT}\;} \text{instruction follower} \xrightarrow{\;\text{RLHF / DPO}\;} \underbrace{\text{helpful assistant}}_{\text{today}}$$

---

# How today connects to the rest of the course

<div class="columns">
<div>

**Today builds directly on:**
- **L01** — every loss here is still an NLL; the KL leash is the same KL from Part 5
- **L16–L17** — a next-token predictor has *capability* but is not yet *helpful*; L17 handed us exactly that gap
- **L05–L07** — Adam, weight decay, small learning rates — the same optimizer, now barely touching the weights

</div>
<div>

**You'll reuse today's ideas in:**
- **L19** — learning without labels (self-supervision) is the same "cheap signal" instinct as RLAIF
- **L21** — the VAE's KL-to-a-reference term is *literally* the RLHF leash again
- **efficient inference** — QLoRA's 4-bit quantization returns as a serving trick

</div>
</div>

<div class="insight">

One lecture, five techniques (SFT · LoRA · QLoRA · RLHF · DPO) + reasoning models — but **one spine**: make good answers more likely while staying close to what the model already knew.

</div>

---

<!-- _class: section-divider -->

## Part 1 · A completer is not an assistant

---

# The real output is $p(\text{next token})$ — it *continues*, it doesn't *reply*

Take a raw pretrained GPT and prompt it: *"Write a polite email to my professor asking for a deadline extension."*

<div class="popquiz">

**Predict.** Does it — (a) write the email · (b) continue with *"...also write one asking for marks back, also one for a lab reschedule, also..."* · (c) refuse politely?

</div>

*Answer:* **(b).** Pretraining optimized one objective — *continue internet text*. "Be a helpful assistant" was **never in the loss**. Your instruction looks, to the model, like the *first line of a list of instructions* to keep generating.

<div class="insight">

Capability (it *can* write the email) and helpfulness (it *will*, on request) are different things. Alignment installs the second one — the whole rest of this lecture.

</div>

---

# The plan · from completer to helper, one box at a time

![w:940px](figures/lec16/svg/alignment_pipeline.svg)

Keep this picture up all lecture. Each part below fills **one box**, and each box exists to fix what the previous one *couldn't* — SFT clones answers, preference learning ranks them, LoRA makes both affordable.

---

<!-- _class: section-divider -->

## Part 2 · SFT — clone the good demonstrations

---

# SFT: fine-tune on `(instruction, response)` pairs

Show the model thousands of examples of *what a helpful reply looks like*, and train with the **same causal-LM cross-entropy** from L16:

```python
conversation = [
  {"role": "system",    "content": "You are a helpful assistant."},
  {"role": "user",      "content": "What is the capital of France?"},
  {"role": "assistant", "content": "The capital of France is Paris."},
]
text = tokenizer.apply_chat_template(conversation)   # → one delimited string
loss = cross_entropy(logits, labels)                 # standard NLL — L01
```

Only **10k–1M** high-quality pairs (vs *trillions* of pretraining tokens) — tiny data, but it re-points all that latent capability at *answering*.

---

# The one detail that makes or breaks SFT: loss masking

The training string contains the **prompt** *and* the **response**. But you only want the model to learn to *produce the response* — not to re-generate the user's question.

<div class="math-box">

$$J_{\text{SFT}}(\theta) = -\sum_{t \in \text{response}} \log p_\theta(y_t \mid y_{<t}) \qquad \text{(prompt tokens contribute } 0\text{)}$$

</div>

Set the label to a sentinel (`-100` in PyTorch) on every system/user token, so it is skipped in the loss.

<div class="keypoint">

**Without the mask, the model learns to write *questions*, not *answers*** — you would be cloning the wrong half of the conversation. Mask the prompt; keep the response.

</div>

---

# SFT is powerful — and easy to overdo

<div class="columns">
<div>

**Use small learning rates** ($10^{-6}$–$10^{-5}$). Aggressive SFT overwrites pretraining knowledge — *catastrophic forgetting*, an L07-style stability problem.

</div>
<div>

**Narrow data → parroting.** Fine-tune only on one style and the model apes that style even when it's wrong. SFT is a *nudge*, not a rebuild.

</div>
</div>

<div class="insight">

SFT gets you a *decent* assistant, but it clones **one** demonstration per prompt. For open-ended questions there are many good answers and many bad ones — and cloning can't express "this one is *better*." That is what Parts 4–5 add. First, Part 3: how to do all of this *cheaply*.

</div>

---

<!-- _class: section-divider -->

## Part 3 · LoRA & QLoRA — fine-tune 0.06% of the model

---

# Full fine-tuning of a 70B model is a ~1 TB problem

To fine-tune all 70B weights with Adam you must hold, simultaneously:

| Buffer | Size | Why |
|---|---|---|
| Weights (bf16) | 140 GB | $70\text{B}\times 2$ bytes |
| Gradients (bf16) | 140 GB | one per weight |
| Adam state $m,v$ (fp32) | 560 GB | $70\text{B}\times 4 \times 2$ |
| Activations | ~200 GB | depends on batch |

**Total ≈ 1 TB** — an H100 cluster. Almost nobody has that. We need to touch *far* fewer parameters.

---

# LoRA · freeze $W_0$, learn a tiny add-on $\Delta W = BA$

Leave the giant pretrained weight $W_0$ **frozen**. Learn two *skinny* matrices $B$ and $A$, and run $W_0 + \tfrac{\alpha}{r}BA$. Only $A,B$ are trained; only $A,B$ are shipped.

![w:820px](figures/lec16/svg/lora_adapter.svg)

Hypothesis (Hu et al. 2021): the *update* a fine-tune needs is **low-rank** — it lives in a tiny subspace, so a rank-$r$ factorization captures it.

---

# LoRA · one analogy to hold onto

<div class="keypoint">

A pretrained model is a **finished oil painting**. Fine-tuning for a new style — say, making the subject smile — does **not** mean repainting the canvas. A few brushstrokes in the right places do the job.

LoRA finds *those* brushstrokes: two small low-rank matrices that nudge the original weights. The masterpiece beneath stays untouched.

</div>

That is why a 70B fine-tune ships as a **~100 MB adapter** — just the brushstroke layer, not the canvas.

---

# LoRA · build the update from a rank-1 example

Let $W_0$ be $4\times 4$ (16 params, frozen). A full update $\Delta W$ would also need 16 numbers. LoRA builds $\Delta W$ from a column $B\in\mathbb R^{4\times 1}$ and a row $A\in\mathbb R^{1\times 4}$ — **8** numbers:

$$\Delta W = B A = \begin{pmatrix} b_1 \\ b_2 \\ b_3 \\ b_4 \end{pmatrix}\!\begin{pmatrix} a_1 & a_2 & a_3 & a_4 \end{pmatrix}, \qquad W = W_0 + \frac{\alpha}{r}\,BA$$

<div class="insight">

Initialize $B=0$, so $\Delta W = 0$ at step zero — **the fine-tune starts *exactly* at the base model** and departs only as far as the data pushes it. (A safety property we'll see echoed by DPO in Part 5.)

</div>

---

# Practice problem 1

<div class="popquiz">

**Practice problem 1.** A Llama layer has weight $W_0 \in \mathbb{R}^{4096 \times 4096}$. Apply LoRA at rank $r=8$.
(a) Trainable params in a **full** fine-tune of this layer? (b) Trainable params in $A$ and $B$? (c) The savings ratio?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 1

$$\text{(a) full: } 4096 \times 4096 = \mathbf{16{,}777{,}216}$$
$$\text{(b) } B\!:\,4096\times 8 = 32{,}768 \;+\; A\!:\,8\times 4096 = 32{,}768 \;=\; \mathbf{65{,}536}$$
$$\text{(c) ratio } = \frac{16{,}777{,}216}{65{,}536} = \mathbf{256\times} \text{ fewer params}$$

<div class="keypoint">

Trainable params scale as $2 d r$ instead of $d^2$ — you cut a *quadratic* to a *linear* in $d$. At $d=4096, r=8$ that is a **256× reduction per layer**, which is exactly why the shippable adapter is ~100 MB, not 140 GB.

</div>

---

# Why LoRA changed the ecosystem: the numbers

![w:560px](figures/lec16/svg/lora_adapter_detail.svg)

| Method | Trainable | Ratio | Adapter on disk |
|---|---|---|---|
| Full fine-tune (7B) | 7 B | 100% | 14 GB |
| LoRA · $r{=}8$ | ~4 M | 0.06% | **8 MB** |
| LoRA · $r{=}64$ | ~33 M | 0.47% | 66 MB |

Everyone downloads the public base **once**; each task is a tiny adapter swap. One base, a thousand fine-tunes.

---

# QLoRA · quantization is image-compression

<div class="insight">

**Analogy.** A `.bmp` photo (millions of colours) saved as a `.gif` (a 256-colour palette): each pixel snaps to its nearest palette colour. Looks the same, far smaller.

QLoRA does this to *weights*. **bf16** offers $2^{16}=65{,}536$ possible values; **4-bit** offers only $2^4 = 16$. Map each weight to its nearest 4-bit codebook value → **4× smaller**. **NF4** ("NormalFloat-4") picks those 16 values spaced to match the bell-shaped spread of real NN weights.

</div>

![w:760px](figures/lec16/svg/qlora_memory.svg)

---

# QLoRA · the mechanism, in four steps

<div class="columns">
<div>

1. Load the base and **quantize to 4-bit** (4× shrink).
2. **Freeze** the 4-bit weights.
3. Attach **bf16 LoRA adapters** — train only these.
4. In the forward pass, **dequantize** each weight block to bf16 for its matmul, then discard.

</div>
<div>

<div class="keypoint">

The base stays frozen and tiny in memory; gradients only ever flow into the small bf16 adapters. You get full-precision *training* on a 4-bit-*stored* model.

</div>

</div>
</div>

---

# QLoRA · worked memory budget for a 70B model

$$\text{bf16 base: } 70\times 10^9 \times 2\,\text{B} = \mathbf{140\ GB} \;\Rightarrow\; \text{needs } 2\times \text{A100-80GB}$$
$$\text{4-bit base: } 70\times 10^9 \times 0.5\,\text{B} = \mathbf{35\ GB} \;\Rightarrow\; \text{fits one } 48\text{GB GPU}$$

Adapters add < 100 MB. The QLoRA paper's headline: fine-tune a **65B** model on a **single 48 GB GPU**.

<div class="insight">

This is the line that democratized LLM fine-tuning — a frontier-scale model on hardware you can rent by the hour. Quantization here is a *training* trick; the same idea returns later as an *inference* trick.

</div>

---

# Try it yourself

<div class="notebook">

**🎛 Interactive · LoRA adapter** — slide the rank $r$ and watch the trainable-parameter count collapse 100×–1000× while the reconstructed $\Delta W$ barely changes. *(Interactive Lab · `interactive/articles/lora-adapter` · nipunbatra.github.io/interactive-articles/lora-adapter)*

</div>

<div class="notebook">

**📓 Notebook · ES 667 alignment** — fine-tune a 7B model with LoRA + `peft` on a small instruction set; compare its answers before / after, then re-run at 4-bit with QLoRA. *(ES 667 · `notebooks/18-lora-finetune.ipynb`)*

</div>

---

<!-- _class: section-divider -->

## Part 4 · RLHF — a reward model on a KL leash

---

# Why SFT isn't enough: many answers are *acceptable*

> *"Explain quantum entanglement to a 12-year-old."*

- **A** — accurate, clear, uses a vivid analogy. ✓
- **B** — accurate but dry and terse. ≈
- **C** — inaccurate, overly technical. ✗

SFT clones demonstrations, so if all three are in the data it weights them **equally**. It has no way to say *A is better than B is better than C*. We need to learn a **preference** — a *ranking*, not a *clone*.

---

# The object we need first: a reward model

Before any RL, build **one** thing — a model that reads a `(prompt, response)` and returns **a single number**: how much a human would prefer it.

![w:660px](figures/lec16/svg/reward_model_io.svg)

<div class="keypoint">

The reward model $r_\phi$ is just a **scorer** — bigger means more preferred. Once you have it, all of RLHF/DPO is one sentence: *"make the policy produce higher-scoring answers."*

</div>

---

# How the reward model is trained: Bradley–Terry pairs

Humans rarely give absolute scores; they compare. Show a labeler a prompt with two answers and ask *"which is better?"* — a *winner* $y_w$ and a *loser* $y_l$. Fit $r_\phi$ so the winner scores higher:

<div class="math-box">

$$P(y_w \succ y_l) = \sigma\big(r_\phi(x,y_w) - r_\phi(x,y_l)\big), \qquad \mathcal L_{\text{RM}} = -\log \sigma\big(r_\phi(x,y_w) - r_\phi(x,y_l)\big)$$

</div>

That is just **binary cross-entropy** on the label "$w$ beat $l$" — the L01 NLL again, now over *pairs*. Hold onto this exact shape; DPO reuses it verbatim.

---

# The RLHF pipeline

![w:900px](figures/lec16/svg/rlhf_pipeline.svg)

Three stages: **SFT** (Part 2) → train the **reward model** on preference pairs → **optimize the policy** (PPO) to score well under $r_\phi$ *without wandering off*.

---

# Two goals: maximize reward, but stay on the leash

<div class="insight">

**Train a dog to fetch.** (1) *Get the ball* → treat; the dog maximizes treats. (**reward $r_\phi$**). (2) *Don't wreck the garden* → a leash keeps it near its normal behaviour. (**KL penalty**).

</div>

<div class="math-box">

$$\max_\theta\; \underbrace{\mathbb E_{x\sim D,\, y\sim \pi_\theta}\big[\, r_\phi(x,y)\,\big]}_{\text{goal 1: high reward}} \;-\; \underbrace{\beta\, D_{\text{KL}}\big(\pi_\theta \,\Vert\, \pi_{\text{ref}}\big)}_{\text{goal 2: stay near SFT}}$$

</div>

$\pi_\theta$ = trainable policy · $\pi_{\text{ref}}$ = frozen SFT model · $\beta$ = leash strength. The **same KL from L01 Part 5**, now used as a *tether* rather than a *loss*.

---

# Worked example · one RLHF step

**Prompt.** *"Explain gravity to a 5-year-old."* The policy proposes $y$ = *"Earth is a giant ball that gently pulls everything toward its middle."*

| | reward $r_\phi(x,y)$ | $\pi_\theta(y\mid x)$ | $\pi_{\text{ref}}(y\mid x)$ |
|:-:|:-:|:-:|:-:|
| this answer | $0.95$ | $0.20$ | $0.001$ |

$$\text{KL contribution} \;\propto\; \log\frac{\pi_\theta(y)}{\pi_{\text{ref}}(y)} = \log\frac{0.20}{0.001} = \log 200 \approx 5.3$$

**Objective** $= 0.95 - \beta\cdot 5.3$. PPO nudges $\theta$ toward high-reward answers **but** the leash charges it for straying far from $\pi_{\text{ref}}$. Turn $\beta$ down and the leash loosens — which is exactly Practice problem 2.

---

# Practice problem 2

<div class="popquiz">

**Practice problem 2.** Why does the RLHF objective *subtract* a KL penalty to the frozen SFT model? Concretely: what goes wrong as $\beta \to 0$ (no leash)? *Think about what the policy is free to do to the reward model.*

</div>

*Try it before the next slide.*

---

# Solution · practice problem 2

The reward model $r_\phi$ is an **imperfect proxy** for "what humans like" — it is only right *near* the answers it was trained on. Remove the leash ($\beta\to 0$) and the policy is free to chase reward **anywhere**:

<div class="columns">
<div>

- **Reward hacking** — finds gibberish that $r_\phi$ over-scores (e.g. pad every answer with flattery the RM mistook for quality).
- **Mode collapse** — every prompt gets the same high-reward template; diversity dies.

</div>
<div>

<div class="keypoint">

The KL term keeps $\pi_\theta$ inside the region where $r_\phi$ is *trustworthy* — near the SFT model. It trades a little reward for staying honest. **Optimize a proxy without a leash and you overfit the proxy.**

</div>

</div>
</div>

---

<!-- _class: section-divider -->

## Part 5 · DPO — one supervised loss, no RL

---

# DPO · skip the reward model *and* the RL loop

RLHF is two stages (RM + PPO), thousands of lines, delicate to tune. Rafailov et al. 2023 asked: do we actually *need* them?

<div class="insight">

**Taste test.** RLHF hands two sodas to a judge, has them score each 1–10, then declares a winner — indirect. **DPO** just asks *"A or B?"* and optimizes that answer directly.

</div>

**The trick:** the KL-regularized RLHF objective has a *closed-form* optimal policy in terms of the reward. Invert it, and the reward model *disappears* — leaving a plain supervised loss on preference pairs. (Full derivation in the Appendix.)

---

# The DPO loss · read it inside-out

<div class="math-box">

$$\mathcal L_{\text{DPO}} = -\log \sigma\!\Big( \underbrace{\beta \log \tfrac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}}_{\text{winner's improvement}} \;-\; \underbrace{\beta \log \tfrac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}}_{\text{loser's improvement}} \Big)$$

</div>

1. **Ratio** $\pi_\theta/\pi_{\text{ref}}$ — is the new model likelier than SFT to say this? Want $>1$ for $y_w$.
2. **Log-ratio** — an "improvement over SFT" score; positive = better.
3. **Winner − loser** — want this margin large and positive.
4. **$\sigma$ then $-\log$** — the *same* Bradley–Terry / BCE shape as the reward model, but written **directly on the policy**. The policy *is* secretly the reward model.

---

# Practice problem 3

DPO starts from the SFT model, so at step zero $\pi_\theta = \pi_{\text{ref}}$ **exactly**.

<div class="popquiz">

**Practice problem 3.** (a) What is $\mathcal L_{\text{DPO}}$ on the very first batch? (b) If the loss is the *same* for every pair at init, how can the gradient still know to raise $y_w$ and lower $y_l$?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 3

**(a)** At init every ratio $\pi_\theta/\pi_{\text{ref}} = 1$, so both improvement scores are $0$, the margin is $0$, and

$$\mathcal L_{\text{DPO}} = -\log\sigma(0) = -\log \tfrac12 = \log 2 \approx \mathbf{0.693}$$

Same starting point as any binary classifier — chance. **(b)** The *value* is flat, but the **gradient is not**. $\tfrac{d}{dm}[-\log\sigma(m)] = -(1-\sigma(m))$, which at $m=0$ equals $-\tfrac12 \ne 0$:

<div class="keypoint">

A negative slope at $m=0$ means: **increase the margin.** Chain rule pushes $\log\pi_\theta(y_w)$ **up** and $\log\pi_\theta(y_l)$ **down** — even though the loss value looks stuck at $0.693$. Zero *loss-change direction* ambiguity, non-zero *gradient*. The winner rises, the loser falls, from the very first step.

</div>

---

# PPO vs DPO · the two pipelines

![w:880px](figures/lec16/svg/dpo_vs_ppo.svg)

RLHF learns a reward model, then runs an RL loop against it. DPO folds both into a **single supervised loss** on the same preference pairs — no reward model, no rollouts, no PPO.

---

# DPO vs RLHF · which to reach for

| | DPO | RLHF (PPO) |
|---|---|---|
| Stages | 1 | 2 (RM + PPO) |
| Code | ~50 LoC | ~2000 LoC (RM, rollout, PPO) |
| Compute | 1× | 3–5× |
| Quality at top scale | tied | sometimes slightly ahead |
| Who uses it | open source (Llama, Mistral tunes) | frontier labs, proprietary tooling |

<div class="insight">

Same objective — *make preferred answers more likely, stay near SFT* — written two ways. DPO trades a little peak controllability for a **massive** drop in engineering cost, which is why it dominates open source.

</div>

---

# Scaling the labels: Constitutional AI / RLAIF

Preference data is expensive — humans must rank thousands of pairs. **Anthropic's Constitutional AI** replaces most of them with the *model itself*:

<div class="columns">
<div>

1. Draft a response.
2. Ask the model: *"Does this violate principle X?"* (from a written **constitution**).
3. Revise the response.
4. Use (original, revised) as a **preference pair** → train with **RLAIF** (RLHF with AI feedback).

</div>
<div>

<div class="keypoint">

Trades human labels for model labels — scaling annotation 100×. The same instinct as *self-supervision* in L19: when labels are the bottleneck, **let the data (or the model) label itself.**

</div>

</div>
</div>

---

# Compare the two recipes yourself

<div class="notebook">

**🎛 Interactive · RLHF vs DPO** — walk a single preference pair through both pipelines: score it with a reward model and take a PPO step, versus plug it straight into the DPO loss. Watch $\pi_\theta$ move the winner up and the loser down either way. *(Interactive Lab explainers · `~/git/interactive/src/articles/`)*

</div>

<div class="notebook">

**📓 Notebook · ES 667 alignment (Part 2)** — implement the DPO loss in ~40 lines, verify $\mathcal L = 0.693$ at init, and take one gradient step to watch the margin grow. *(ES 667 · `notebooks/18-lora-finetune.ipynb`)*

</div>

---

<!-- _class: section-divider -->

## Part 6 · Reasoning models — scaling *test-time* compute

---

# Let the model *think longer* before it answers

The 2024–25 wave — **o1, o3, DeepSeek-R1, Claude extended thinking** — pushes on a new axis:

1. Train the model to emit a **long internal chain-of-thought** before its final answer.
2. Train that chain with **RL** — the same reward machinery from Parts 4–5, now rewarding *reasoning*, not just phrasing.
3. At inference, let it think for **seconds to minutes** — spend 10×–100× more compute *per query*.

<div class="insight">

L17 scaled **training** compute (bigger $N$, more data $D$). This scales **inference** compute. Both curves are still climbing — and the RL that trains the reasoning traces is *exactly* the alignment machinery you just learned.

</div>

---

# Outcome rewards vs process rewards

Reward "was the *final* answer correct?" and the signal is **sparse** — one bit at the end of a 50-step proof, with no idea *which* step went wrong.

<div class="keypoint">

A **process reward model (PRM)** grades *each reasoning step* — a dense signal that catches a bad intermediate deduction the moment it happens. *"Let's Verify Step by Step"* (Lightman et al. 2023) showed PRM-trained verifiers solve substantially more MATH problems than outcome-only baselines.

</div>

Denser reward → the chain improves *step by step*, not just at the finish line. **Process > outcome** for multi-step reasoning.

---

# The payoff · one year of inference-scaling

| Model | AIME (math) | Codeforces Elo | GPQA (science) |
|:-:|:-:|:-:|:-:|
| GPT-4 | 12% | ~800 | 39% |
| o1-preview | 44% | ~1500 | 73% |
| o1 | 74% | ~1900 | 78% |
| o3 | **97%** | **~2700** | **88%** |

*(Vendor-reported, Dec 2024 — salt to taste.)* o3's 97% on AIME sits above nearly every human contestant. **Same base-model class; the training regime and the test-time budget changed.**

---

# Choosing an alignment method

| Scenario | Reach for |
|:-:|:-:|
| Small task-specific set (< 10k) | SFT on the full model |
| Large instruction set (100k+) | SFT + LoRA |
| 70B model on a ≤ 48 GB GPU | QLoRA (NF4 + LoRA) |
| Preference alignment, small team | **DPO** |
| Precise behavioural control | RLHF + custom reward model |
| Safety + cheap labels | Constitutional AI / RLAIF |
| Math / code / reasoning | SFT + RL with process rewards (o1-style) |

<div class="insight">

In open source today, **QLoRA + DPO** is the default recipe. Frontier labs mix all of the above. Every row is still the *same* NLL-and-a-leash idea, tuned to a different constraint.

</div>

---

<!-- _class: summary-slide -->

# One sentence to remember

<div class="keypoint">

**Alignment turns a good *completer* into a good *helper*: clone good answers (SFT), then make preferred answers more likely (RLHF / DPO) while a KL leash keeps you near what the model already knew — all trained cheaply as a tiny low-rank patch (LoRA / QLoRA).**

</div>

- **SFT** — NLL on `(instruction, response)`, *masking the prompt*.
- **LoRA** — train $\Delta W = BA$, ~256× fewer params, a 100 MB adapter. **QLoRA** — 4-bit base, 70B on one GPU.
- **RLHF** — reward model (Bradley–Terry) + PPO with a **KL leash** to SFT.
- **DPO** — one supervised loss that *is* the reward model, no RL.
- **Reasoning models** — RL over long chains-of-thought; scale *test-time* compute.

**Next (L19):** what if the data could label *itself*? Self-supervised & contrastive learning.

---

<!-- _class: compact -->

# Sources & credits

<div class="paper">

This lecture reuses and adapts material from the instructor's ES 667 alignment notes and the canonical papers:

- **SFT + RLHF pipeline, InstructGPT** — Ouyang et al., *Training language models to follow instructions with human feedback* (2022).
- **LoRA** — Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021); **QLoRA** — Dettmers et al. (2023).
- **DPO** — Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* (2023).
- **Process rewards** — Lightman et al., *Let's Verify Step by Step* (2023); **Constitutional AI** — Bai et al. (2022).
- **Code-first** — Hugging Face *PEFT* + *TRL* docs; **pedagogical framing** — A. Ng, *Deep Learning Specialization*; Karpathy, *State of GPT*.

Figures adapted from the ES 667 figure library (`figures/lec16/svg/`). All source materials © N. Batra & teaching staff.

</div>

---

<!-- _class: section-divider -->

## ⭐⭐⭐ Appendix · optional depth

*Skip on a first pass — for the curious.*

---

<!-- _class: math-heavy -->

# ⭐⭐⭐ Optional · the KL-regularized optimal policy

Maximize the RLHF objective *exactly* (not with PPO) for a fixed reward $r$:

$$\max_\theta\; \mathbb E_{y\sim\pi_\theta}[\,r(x,y)\,] - \beta\, D_{\text{KL}}(\pi_\theta \Vert \pi_{\text{ref}})$$

Writing the KL out and setting the functional derivative to zero (with a Lagrange multiplier for $\sum_y \pi = 1$) gives a **closed form**:

<div class="math-box">

$$\pi^\star(y\mid x) = \frac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\,\exp\!\Big(\tfrac{1}{\beta}\,r(x,y)\Big), \qquad Z(x) = \sum_{y}\pi_{\text{ref}}(y\mid x)\,e^{r(x,y)/\beta}$$

</div>

The optimal policy is the **reference tilted by the reward** — high-reward answers up-weighted, gated by the leash $\beta$. Small $\beta$ → sharp tilt (chase reward); large $\beta$ → stay near $\pi_{\text{ref}}$. This single expression is the seed of DPO.

---

<!-- _class: math-heavy -->

# ⭐⭐⭐ Optional · DPO falls out by inverting that formula

Solve the closed form for the reward — the intractable $\log Z(x)$ appears as an additive term:

$$r(x,y) = \beta \log \frac{\pi^\star(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta \log Z(x)$$

Substitute into the Bradley–Terry loss $-\log\sigma\big(r(x,y_w) - r(x,y_l)\big)$. Because the loss uses only the **difference** $r(x,y_w)-r(x,y_l)$, the shared $\beta\log Z(x)$ **cancels**:

<div class="math-box">

$$\mathcal L_{\text{DPO}} = -\log\sigma\!\Big(\beta\log\tfrac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta\log\tfrac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\Big)$$

</div>

No reward model, no partition function, no RL loop. Its gradient, $\nabla_\theta\mathcal L_{\text{DPO}} \propto -\sigma(\hat r_l - \hat r_w)\big[\nabla\log\pi_\theta(y_w) - \nabla\log\pi_\theta(y_l)\big]$, up-weights the winner and down-weights the loser — **more strongly when the model currently has them backwards.** That is the entire method.
