---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# Alignment &amp; Fine-tuning

## Lecture 16 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar · Aug 2026*

---

# Learning outcomes

By the end of this lecture you will be able to:

1. Explain why pretrained LLMs are **not chat assistants** out of the box.
2. Write the **instruction-tuning** recipe (SFT on demo data).
3. Explain the **LoRA** low-rank update and wire it up with `peft`.
4. Compute **LoRA parameter count** and memory savings for a 70B model.
5. Read the **DPO loss** and explain how it replaces the RLHF reward model (derivation optional).
6. Pick **SFT / LoRA / QLoRA / RLHF / DPO** for a given alignment task.

---

# Where we are

- **Pretrained LLMs** (L15) · Chinchilla-optimal, RoPE, GQA — raw capability.
- But raw pretrained models are **not** chat assistants. Raw GPT-3 will complete your prompt with plausible internet text, not with a useful answer.

<div class="paper">

**Reading & inspiration** ·
- **Ouyang et al. 2022 · *InstructGPT*** — the canonical SFT + RLHF pipeline.
- **Rafailov et al. 2023 · *DPO*** — RLHF without RL.
- **Hu et al. 2021 · *LoRA*** + **Dettmers 2023 · *QLoRA***.
- **Hugging Face · *PEFT* docs** + ***TRL* docs** — code-first.
- **Anthropic · *Constitutional AI***.
- **Karpathy · *State of GPT*** (alignment section).

</div>

---

# Four questions

1. How do we turn pretrained models into **instruction followers**?
2. What is **LoRA** and why did it eat the fine-tuning world?
3. How does **RLHF** differ from **DPO**?
4. What's happening with **reasoning models** (o1, Claude thinking)?

---

# Pop quiz · why doesn't pretraining-alone give us ChatGPT?

You take GPT-3 raw (no fine-tuning) and prompt it · *"Write a polite email to my professor asking for a deadline extension."*

<div class="popquiz">

(a) It writes a polite email.
(b) It continues with *"I have not done the homework, also write the same for asking for marks back, also for…"* — i.e. it generates plausible *training-like* continuation.
(c) It refuses politely.

Stop and predict. The real answer is **(b)** — pretraining only knows *complete the next token of internet text.* "Be a helpful assistant" is **not in the loss.** That gap between *capability* and *helpfulness* is the entire reason for L16.

</div>

---

▶ **Interactive** · LoRA adapter knobs and rank effects → [lora-adapter](https://nipunbatra.github.io/interactive-articles/lora-adapter/).

---

<!-- _class: section-divider -->

### PART 1

# SFT · Instruction tuning

How do you teach a text *completer* to *answer*?

---

# Why SFT is necessary

A pretrained model sees:

> *"The capital of France is"*    → completes with "Paris"

Not bad. But give it:

> *"What is the capital of France?"*

And it might reply with "What is the capital of Italy? What is the capital of Spain? ..." — it treats your question as a prompt to continue generating FAQ-style content.

**SFT fixes this.** Train on `(instruction, response)` pairs — teach the model *what a helpful response looks like*.

---

# SFT in practice

```python
# A typical SFT training example
conversation = [
    {"role": "system",    "content": "You are a helpful assistant."},
    {"role": "user",      "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."}
]

# Apply chat template → single string with delimiters
text = tokenizer.apply_chat_template(conversation)

# Train with standard causal LM loss, but ONLY on the assistant
# response (mask system + user tokens)
loss = ce_loss(logits, labels, label_smoothing=0.0)
```

Dataset sizes: 10k–1M high-quality examples. Much smaller than pretraining (trillions). But the effect is massive.

---

# SFT gotchas

<div class="warning">

**Overtraining** — SFT on too narrow a dataset makes the model parrot rather than help.

**Forgetting** — aggressive SFT can destroy pretraining knowledge. Use small LRs (1e-6 to 1e-5).

**Loss masking** — if you don't mask non-response tokens, the model "learns" to produce the user's question.

</div>

---

<!-- _class: section-divider -->

### PART 2

# LoRA · parameter-efficient fine-tuning

Can you fine-tune a 70B model by training 0.06% of it?

---

# Why full fine-tuning is painful

To fine-tune a 70B model with Adam:

- **Weights** · 70B × 2 bytes (bf16) = 140 GB
- **Gradients** · 140 GB
- **Optimizer state** (Adam $m$ and $v$, fp32) · 70B × 4 × 2 = 560 GB
- **Activations** · ~200 GB (depends on batch)

Total: **~1 TB**. Needs an H100 cluster. Most people don't have that.

---

# LoRA · the 2021 fix

![w:920px](figures/lec16/svg/lora_adapter.svg)

---

# LoRA · detailed view

![w:920px](figures/lec16/svg/lora_adapter_detail.svg)

<div class="realworld">

▶ Interactive: slide the rank, see parameter counts drop 100×–1000× — [lora-adapter](https://nipunbatra.github.io/interactive-articles/lora-adapter/).

</div>

---

# LoRA · the oil-painting analogy

<div class="keypoint">

A pretrained model is a **giant oil painting**. Fine-tuning for a new style — say, making the subject smile — doesn't require **repainting the canvas**. A few brushstrokes in the right places do the job.

LoRA finds *those* brushstrokes · two small low-rank matrices that adjust the original weights · the masterpiece beneath stays untouched.

</div>

That's why a 70B-param fine-tune can ship as a 100MB adapter file · the brushstroke layer · while the base painting stays the same.

---

# LoRA · build the update from a tiny rank-1 example

Pretrained $W_0$ is $4 \times 4$ (16 params, frozen). Full update $\Delta W$ would also have 16 params.

LoRA's idea · construct $\Delta W$ from two rank-1 matrices.
- $B \in \mathbb{R}^{4 \times 1}$ (column) · 4 trainable params.
- $A \in \mathbb{R}^{1 \times 4}$ (row) · 4 trainable params.
- $\Delta W = B A$ is $4 \times 4$ but generated from only **8** numbers.

$$\Delta W = \begin{pmatrix} b_1 \\ b_2 \\ b_3 \\ b_4 \end{pmatrix}\begin{pmatrix} a_1 & a_2 & a_3 & a_4 \end{pmatrix} = \begin{pmatrix} b_1 a_1 & b_1 a_2 & b_1 a_3 & b_1 a_4 \\ b_2 a_1 & b_2 a_2 & b_2 a_3 & b_2 a_4 \\ b_3 a_1 & b_3 a_2 & b_3 a_3 & b_3 a_4 \\ b_4 a_1 & b_4 a_2 & b_4 a_3 & b_4 a_4 \end{pmatrix}$$

Train only $A, B$; keep $W_0$ frozen:
$$W = W_0 + \frac{\alpha}{r}\,BA$$
At $t = 0$, $B = 0 \Rightarrow BA = 0$ → output equals base model.

---

# Worked numeric · LoRA on a Llama-7B layer

Typical layer · $d = 4096$, weight $4096 \times 4096$.

**Full fine-tune.** Trainable = $4096^2 = \mathbf{16{,}777{,}216}$ per layer.

**LoRA at rank $r = 8$.**
- $A$: $4096 \times 8 = 32{,}768$ params.
- $B$: $8 \times 4096 = 32{,}768$ params.
- Total · $32{,}768 + 32{,}768 = \mathbf{65{,}536}$.

**Savings · $16{,}777{,}216 / 65{,}536 = \mathbf{256\times}$ fewer params per layer.** That's why a 70B fine-tune can ship as a 100 MB adapter — only $A$ and $B$ for each adapted layer, not the base.

---

# LoRA numbers · 7B model

| Method | Trainable params | Ratio | Disk |
|--------|------------------|-------|------|
| Full fine-tune | 7B | 100% | 14 GB |
| LoRA · r=8 | ~4M | 0.06% | 8 MB |
| LoRA · r=64 | ~33M | 0.47% | 66 MB |
| QLoRA · 4-bit base + r=8 | ~4M | 0.06% | 8 MB + 3.5 GB base |

<div class="realworld">

Ship the 8 MB adapter alongside the public 7B base. Everyone downloads the base once; each task is just an adapter swap. This changed the open-source LLM ecosystem.

</div>

---

# LoRA in PyTorch · peft library

```python
from peft import LoraConfig, get_peft_model

lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # where to inject
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(base_model, lora_cfg)
model.print_trainable_parameters()
# trainable params: 4,194,304  all: 7,000,000,000  trainable%: 0.06%
```

Training loop is identical to normal SFT — only 0.06% of parameters have gradients, everything else is frozen.

---

# QLoRA · memory breakdown

![w:920px](figures/lec16/svg/qlora_memory.svg)

---

# QLoRA · quantization is image-compression

<div class="insight">

**Analogy.** A photo `.bmp` (millions of colours) → save as `.gif` (256-colour palette). The GIF maps each pixel to its closest palette colour. Looks similar, vastly smaller.

QLoRA does this to weights. **bf16** has $2^{16} = 65{,}536$ possible values; **4-bit** has only $2^4 = 16$. Map each weight to its closest 4-bit codebook value → 4× memory shrink.

**NF4** ("NormalFloat 4-bit") · a clever choice of those 16 codebook values, spaced to match the typical bell-shaped distribution of NN weights.

</div>

---

# QLoRA · worked memory budget for 70B

Per parameter:
- bf16 → 2 bytes
- 4-bit → 0.5 bytes

**Standard LoRA (bf16 base).**
$70 \times 10^9 \cdot 2 = 1.4 \times 10^{11}$ bytes = **140 GB** → needs 2× A100 80 GB.

**QLoRA (4-bit base).**
$70 \times 10^9 \cdot 0.5 = 3.5 \times 10^{10}$ bytes = **35 GB** → fits on a single 48 GB GPU (the QLoRA paper's setting), with room left for activations. LoRA adapters add < 100 MB.

---

# QLoRA · the mechanism

1. Load base, quantize to 4-bit (4× shrink).
2. Freeze 4-bit weights.
3. Attach **bf16** LoRA adapters. Train only these.
4. During the forward pass, dequantize chunks of $W_0$ to bf16 for each matmul, then discard.

Used everywhere in open-source LLM fine-tuning since 2023.

---

<!-- _class: section-divider -->

### PART 3

# RLHF · reward model + PPO

What if there is no single *right* answer to clone — only better and worse ones?

---

# Why SFT isn't enough

SFT teaches the model *one way* to respond to each prompt. But for open-ended questions, many responses are acceptable — and you want the model to prefer the *best* one.

> *"Explain quantum entanglement to a 12-year-old."*

- Response A · accurate, clear, uses an analogy. ✓
- Response B · accurate but terse and dry. ≈
- Response C · inaccurate, overly technical. ✗

SFT would weight all three equally if all are in the dataset. **RLHF** adds preference learning.

---

# SFT vs RLHF · dog-training analogy

<div class="keypoint">

**SFT** is teaching a dog the command "sit." One correct action; demonstrate, repeat.

**RLHF** is teaching a dog to choose between options · "bark or stay quiet when the doorbell rings" · we **reward** good choices, **penalize** bad ones.

</div>

By showing the model pairs of options (good answer · bad answer) and letting it learn from preference, we shape *general behavior* rather than memorize a single correct response.

---

# The RLHF pipeline

![w:920px](figures/lec16/svg/rlhf_pipeline.svg)

---

# RLHF · the dog-with-two-goals analogy

<div class="insight">

Train a dog to fetch.
1. **Get the ball** → treat. The dog maximizes treats. (**Reward $r_\phi$.**)
2. **Don't go crazy** → if it knocks over furniture chasing the ball, you tug a leash to keep it close to its normal behaviour. (**KL penalty $D_\text{KL}$.**)

RLHF balances both · max reward, but stay near the SFT model.

</div>

---

# ⭐⭐⭐ Optional · RLHF objective · the equation

$$\max_\theta\, \underbrace{\mathbb{E}_{x \sim D}[\, r_\phi(x, \pi_\theta(x)) \,]}_{\text{Part 1: maximize reward}} \;-\; \underbrace{\beta\, D_\text{KL}(\pi_\theta \Vert \pi_\text{ref})}_{\text{Part 2: stay near SFT}}$$

**Part 1** drives the model to produce high-reward answers. **Part 2** keeps the policy close to its SFT initialization → prevents the dog from tearing up the garden.

The two terms are exactly the two goals from the dog analogy · *maximize treats* (reward) while *staying on the leash* (KL penalty to the SFT model).

---

# ⭐⭐⭐ Optional · RLHF objective · the symbols

- $\pi_\theta$ · trainable model ("policy"); takes prompt $x$, generates response $y$.
- $\pi_\text{ref}$ · frozen SFT model.
- $r_\phi(x, y)$ · reward model; scores responses (e.g. 0.85).
- $\mathbb{E}_{x \sim D}[\cdot]$ · average over prompts in dataset $D$.
- $D_\text{KL}(\pi_\theta \,\|\, \pi_\text{ref})$ · KL divergence; high when $\pi_\theta$ diverges from $\pi_\text{ref}$.
- $\beta$ · leash strength.

---

# Worked example · one RLHF step

**Prompt.** "Explain gravity to a 5-year-old."

- $\pi_\text{ref}$ would say *"Gravity is a fundamental interaction…"* — assigns this prob 0.001 (technical, not for kids).
- $\pi_\theta$ generates *"Imagine the earth is a big magnet and you have tiny magnets in your shoes…"* — assigns prob 0.2.

**Reward.** $r_\phi \approx 0.95$ — friendly, age-appropriate.

**KL penalty.** Contribution from this example $\propto \log(0.2/0.001) = \log 200 \approx 5.3$ — sizeable.

**Total objective.** $0.95 - \beta \cdot 5.3$.

PPO updates $\theta$ to push *toward* this kind of response (high reward) while not pulling too far from $\pi_\text{ref}$ (KL penalty). Without KL, the model would happily reward-hack into nonsense; the leash holds it close.

---

# RLHF · what could go wrong

<div class="warning">

**Reward hacking.** The policy finds ways to game the RM that humans wouldn't approve of. Classic example: the RM rewards longer answers, so the policy learns to produce verbose output regardless of content.

**Mode collapse.** PPO can push the policy to always produce very similar responses — diversity loss.

**Sycophancy.** If human labelers preferred agreeable answers, the model learns to agree with whatever the user says, even when wrong.

</div>

Mitigating these is half the art of alignment.

---

<!-- _class: section-divider -->

### PART 4

# DPO · Direct Preference Optimization

Do we actually *need* the reward model and the RL loop?

---

# DPO · the 2023 simplification

<div class="paper">

Rafailov et al. 2023 · *"Direct Preference Optimization: Your Language Model is Secretly a Reward Model."*

</div>

**Insight:** the RLHF objective has a closed-form optimal policy given the reward. Work backwards — derive a loss directly on preference pairs, skipping the RM and PPO entirely.

---

# DPO · the simpler taste-test analogy

<div class="insight">

**RLHF** · two sodas → judge scores each 1–10 → use scores to declare a winner. Indirect.
**DPO** · just ask "A or B?" → direct preference.

DPO writes a loss that says · *"directly increase prob of the winner $y_w$, decrease prob of the loser $y_l$, relative to the SFT baseline."*

</div>

---

# ⭐⭐⭐ Optional · DPO loss · inside-out

$$\mathcal{L}_\text{DPO} = -\log \sigma\!\left( \underbrace{\beta \log \tfrac{\pi_\theta(y_w \mid x)}{\pi_\text{ref}(y_w \mid x)}}_{\text{winner score}} - \underbrace{\beta \log \tfrac{\pi_\theta(y_l \mid x)}{\pi_\text{ref}(y_l \mid x)}}_{\text{loser score}} \right)$$

Build it from inside:
1. **Ratio** $\pi_\theta / \pi_\text{ref}$ · how much more likely is the new model to produce $y$ than the SFT model? Want $> 1$ for $y_w$.
2. **Log ratio** · "improvement score." Positive = improved over SFT.
3. **Difference** · winner score − loser score. Want **large positive**.
4. **Sigmoid** · squash to $(0,1)$. Large positive diff → $\sigma \approx 1$.
5. **$-\log \sigma(\cdot)$** · standard CE loss. $\sigma \to 1 \Rightarrow$ loss $\to 0$. $\sigma \to 0 \Rightarrow$ huge loss → big gradient kicks in.

---

# Pop quiz · DPO at step zero

DPO training starts from the SFT model, so at step zero $\pi_\theta = \pi_\text{ref}$ exactly.

<div class="popquiz">

What is the DPO loss on the very first batch?

(a) **0** — nothing has diverged yet.
(b) Depends on the log-probs of $y_w$ and $y_l$.
(c) **$\log 2 \approx 0.693$** — always.

Stop and reason from the formula. **Answer · (c).** Every ratio $\pi_\theta / \pi_\text{ref} = 1$, so both scores are 0, the margin is 0, and $\sigma(0) = \tfrac{1}{2}$. DPO starts exactly where a binary classifier starts — at chance.

</div>

---

# Worked numeric · DPO loss by hand

Prompt $x$ · "Suggest a coffee shop name." Preferred $y_w$ = "The Daily Grind" · rejected $y_l$ = "Coffee Shop". Take $\beta = 0.5$, fresh from SFT ($\pi_\theta = \pi_\text{ref}$):

| | $\log \pi_\text{ref}(y \mid x)$ | $\log \pi_\theta(y \mid x)$ |
|:-:|:-:|:-:|
| $y_w$ | $-2.0$ | $-2.0$ |
| $y_l$ | $-1.0$ | $-1.0$ |

<div class="math-box">

**Winner score** · $\beta\,[\log \pi_\theta(y_w) - \log \pi_\text{ref}(y_w)] = 0.5\,(-2.0 + 2.0) = 0$
**Loser score** · $0.5\,(-1.0 + 1.0) = 0$
**Margin** · $0 - 0 = 0 \;\Rightarrow\; \sigma(0) = 0.5$
**Loss** · $-\log 0.5 \approx \mathbf{0.693}$

</div>

Note · the reference model *prefers the rejected answer* ($e^{-1} = 0.37 > e^{-2} = 0.14$). DPO doesn't care about raw probabilities — only about how $\pi_\theta$ **moves relative to** $\pi_\text{ref}$.

---

# Worked numeric · one gradient step later

The gradient of $-\log\sigma(\text{margin})$ pushes the winner's log-prob **up** and the loser's **down**. Say one step gives:

| | $\log \pi_\theta$ before | after | probability |
|:-:|:-:|:-:|:-:|
| $y_w$ | $-2.0$ | $-1.5$ | $0.14 \to 0.22$ **↑** |
| $y_l$ | $-1.0$ | $-1.5$ | $0.37 \to 0.22$ **↓** |

<div class="math-box">

**Winner score** · $0.5\,(-1.5 + 2.0) = +0.25$
**Loser score** · $0.5\,(-1.5 + 1.0) = -0.25$
**Margin** · $0.25 - (-0.25) = 0.5 \;\Rightarrow\; \sigma(0.5) \approx 0.62$
**Loss** · $-\log 0.62 \approx \mathbf{0.47}$ — down from $0.693$ ✓

</div>

Pure supervised loss. No RL loop, no reward model · ~50 lines of code vs RLHF's thousands.

---

# PPO vs DPO · the two pipelines

![w:900px](figures/lec16/svg/dpo_vs_ppo.svg)

---

# DPO vs RLHF · which to use

| | DPO | RLHF |
|--|-----|------|
| Training stages | 1 | 2 (RM + PPO) |
| Code complexity | ~50 LoC | ~2000 LoC (PPO, RM, rollout) |
| Compute | 1× | 3–5× |
| Quality at top scale | tied | often slightly ahead |
| Open-source preference | **DPO** | RLHF for frontier labs |

<div class="realworld">

As of this writing · open-source (Hugging Face, Mistral, most Llama fine-tunes) favors **DPO**; frontier labs (Anthropic, OpenAI, Google) favor **RLHF variants** with proprietary tooling. Both work.

</div>

---

# Constitutional AI and RLAIF

Anthropic 2022 variation · use the model itself (or a stronger one) to generate preference labels from a written "constitution":

1. Draft a response.
2. Ask the model: *"Does this violate principle X?"*
3. Revise the response.
4. Use revisions as preference pairs, train with RLAIF (RLHF with AI feedback).

Scales human annotation by factors of 100+. Used in Claude's alignment pipeline.

---

<!-- _class: section-divider -->

### PART 5

# Reasoning models · the 2024 turn

What happens if you let the model *think longer* before it answers?

---

# Reasoning models

The latest generation as of this writing · **o1, o3, Claude extended thinking, DeepSeek R1.**

The core idea:

1. Train the model to produce **long internal chain-of-thought** before responding.
2. Use RL with **process rewards** (reward good reasoning steps, not just final answer).
3. At inference, let the model think for minutes — spend 10×–100× more compute.

Result: dramatically better on math, code, logic benchmarks.

<div class="insight">

Scaling laws in training compute produced pretrained capability. A new axis — **scaling test-time compute** — now unlocks reasoning. Both will likely continue.

</div>

---

# ⭐⭐⭐ Optional · Process rewards · what "reasoning training" looks like

Traditional RL reward · "was the final answer correct?" · sparse, late signal.

<div class="keypoint">

**Process reward model (PRM)** · grades individual reasoning steps · dense signal, catches bad intermediate logic.

</div>

<div class="math-box">

| Stage | Input | Output |
|:-:|:-:|:-:|
| PRM training | 100k human-labeled step-by-step proofs | reward-per-step model |
| RL with PRM | sampled chains-of-thought | step-reward gradient up |
| Result | chains improve *step-by-step*, not just final | better generalization |

</div>

*Let's Verify Step by Step* (Lightman et al., OpenAI 2023) · PRM-based verifiers solve substantially more MATH problems than outcome-reward-only baselines. Process > outcome rewards for multi-step reasoning.

---

# Reasoning models · benchmark jump

<div class="math-box">

| Model | AIME 2024 (math) | Codeforces Elo | GPQA (science) |
|:-:|:-:|:-:|:-:|
| GPT-4 | 12% | ~800 | 39% |
| o1-preview | 44% | ~1500 | 73% |
| o1 | 74% | ~1900 | 78% |
| o3 | **97%** | **~2700** (grandmaster) | **88%** |

*(OpenAI-reported figures, Dec 2024 · take vendor benchmarks with salt.)*

</div>

<div class="insight">

o3's 97% on AIME puts it above nearly all human contestants. **One year** of inference-compute scaling delivered this jump. Same base model class; the training regime changed.

</div>

---

# Picking an alignment method

<div class="math-box">

| Scenario | Recommended |
|:-:|:-:|
| Small task-specific dataset (< 10k) | SFT on full model |
| Large instruction dataset (100k+) | SFT + LoRA |
| Consumer GPU (≤ 48GB) on 70B model | QLoRA (NF4 + LoRA) |
| Need preference alignment, small team | DPO |
| Need precise control of behaviors | RLHF + custom RM |
| Need safety + low-cost labels | Constitutional AI / RLAIF |
| Need reasoning / math / code | SFT + RL-with-process-rewards (o1 style) |

</div>

<div class="realworld">

As of this writing, in open source · QLoRA + DPO is the dominant recipe for instruction tuning. Frontier labs mix all of the above.

</div>

---

<!-- _class: summary-slide -->

# Putting it all together · the L16 master sentence

<div class="math-box">

**Alignment turns a "good completer" into a "good helper."** Three steps in order · **SFT** (clone good demonstrations) → **reward / preference model** (learn what humans like) → **RLHF or DPO** (push the policy toward preferred outputs while staying close to the SFT model). **LoRA / QLoRA** make each step cheap by training a tiny low-rank update instead of all the weights.

</div>

**Same NLL framework as L00** — the only thing that changes is *what counts as good $y$*.

---

<!-- _class: summary-slide -->

# The L16 pipeline · at a glance

| Step | What it optimizes | Loss type |
|:-:|:-:|:-:|
| SFT | $-\log p(y_\text{good} \mid x)$ | NLL (L00) |
| Reward modelling | $\sigma(r_\text{better} - r_\text{worse})$ | Bradley-Terry pairwise |
| RLHF (PPO) | $\mathbb{E}[r(x, y)] - \beta \,\text{KL}(\pi \,\Vert\, \pi_\text{SFT})$ | RL + KL |
| DPO | closed-form simplification of RLHF | direct preference |

Read each row as the *same* "make good $y$ more likely" objective, written for a different kind of supervision signal.

---

# Practice problems

<div class="math-box">

**P1.** SFT loss is just NLL on (instruction → response) pairs. Why are the *instruction* tokens masked out of the loss while the *response* tokens are kept? What would the model learn without the mask?

**P2.** A reward model assigns scores $r_a = 1.2, r_b = 0.5$ to two responses. Compute the Bradley-Terry probability that $a$ is preferred. Show how this maps to a binary cross-entropy training loss on pair labels.

**P3.** Sketch why RLHF includes a **KL penalty** to the SFT model. What goes wrong if $\beta \to 0$?

**P4.** **LoRA** with rank $r = 8$ on a $4096 \times 4096$ weight matrix. Count parameters in (a) the original matrix, (b) the LoRA $A, B$ matrices. Savings ratio?

**P5.** **DPO** removes the explicit reward model. State the closed-form DPO loss in one line and explain why it has a similar shape to a binary classification loss.

**P6.** Pick **SFT-only / LoRA / RLHF / DPO** for each of (a) tone matching to a brand, (b) safety-policy alignment with disagreement among annotators, (c) one-shot domain adaptation to legal text.

</div>

---

# Lecture 16 — summary

- **SFT** · train on (instruction, response) pairs to turn a pretrained LM into an assistant.
- **LoRA** · fine-tune two small matrices alongside a frozen base · 100× fewer trainable params. **QLoRA** adds 4-bit quantization to cut memory 4×.
- **RLHF** · SFT → reward model → PPO against RM with KL penalty to SFT. Original ChatGPT recipe.
- **DPO** · closed-form alignment without an RM · one supervised loss. Default in open-source.
- **Constitutional AI / RLAIF** · self-critique to scale preference data.
- **Reasoning models** (2024+) · RL with process rewards · long internal chain of thought · new scaling axis.

### Read before Lecture 17

Prince Ch 14 (unsupervised, contrastive).

### Next lecture

**Self-Supervised &amp; Contrastive Learning** — SimCLR, BYOL, MAE, DINOv2. How to learn without labels.

<div class="notebook">

**Notebook 16** · `16-lora-finetune.ipynb` — fine-tune a 7B model with LoRA + peft on a small instruction dataset · compare responses before / after.

</div>

---

# The one-sentence takeaway

<div class="insight">

**Pretraining builds a completer; preferences build a helper.**

*Next · what if the data could label itself? Self-supervised learning.*

</div>
