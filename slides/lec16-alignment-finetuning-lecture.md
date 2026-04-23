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

# Where we are

- **Pretrained LLMs** (L15) · Chinchilla-optimal, RoPE, GQA — raw capability.
- But raw pretrained models are **not** chat assistants. Raw GPT-3 will complete your prompt with plausible internet text, not with a useful answer.

<div class="paper">

Today maps to HF PEFT docs + the **InstructGPT paper** (Ouyang 2022) + **DPO paper** (Rafailov 2023).

</div>

Four questions:
1. How do we turn pretrained models into **instruction followers**?
2. What is **LoRA** and why did it eat the fine-tuning world?
3. How does **RLHF** differ from **DPO**?
4. What's happening with **reasoning models** (o1, Claude thinking)?

---

<!-- _class: section-divider -->

### PART 1

# SFT · Instruction tuning

The first step after pretraining

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

# Train with standard causal LM loss, but **only on the assistant response**
# (mask system + user tokens)
loss = ce_loss(logits, labels, label_smoothing=0.0)
```

Dataset sizes: 10k–1M high-quality examples. Much smaller than pretraining (trillions). But the effect is massive.

---

# SFT gotchas

<div class="warning">

**Overtraining** — SFT on too narrow a dataset makes the model parroting rather than helpful.

**Forgetting** — aggressive SFT can destroy pretraining knowledge. Use small LRs (1e-6 to 1e-5).

**Loss masking** — if you don't mask non-response tokens, the model "learns" to produce the user's question.

</div>

---

<!-- _class: section-divider -->

### PART 2

# LoRA · parameter-efficient fine-tuning

The technique everyone uses

---

# Why full fine-tuning is painful

To fine-tune a 70B model with Adam:

- **Weights** · 70B × 2 bytes (bf16) = 140 GB
- **Gradients** · 140 GB
- **Optimizer state** (Adam m and v) · 2 × 280 GB
- **Activations** · ~200 GB (depends on batch)

Total: **~760 GB**. Needs an H100 cluster. Most people don't have that.

---

# LoRA · the 2021 fix

![w:920px](figures/lec16/svg/lora_adapter.svg)

<div class="realworld">

▶ Interactive: slide the rank, see parameter counts drop 100×–1000× — [lora-adapter](https://nipunbatra.github.io/interactive-articles/lora-adapter/).

</div>

---

# LoRA · the insight

<div class="keypoint">

**Hypothesis** (Hu et al. 2021) · the useful *change* to a pretrained weight is low-rank. You don't need a $d \times d$ update; you need rank-$r$ where $r \ll d$.

</div>

<div class="math-box">

$$W = W_0 + \frac{\alpha}{r} B A$$

- $W_0 \in \mathbb{R}^{d \times d}$ · frozen pretrained weight
- $A \in \mathbb{R}^{d \times r}$ · trainable, Gaussian-init
- $B \in \mathbb{R}^{r \times d}$ · trainable, **zero-init**
- $r \ll d$ · typically 8–64; $d = 4096$ or larger

At $t = 0$, $B = 0$ so $BA = 0$ — the LoRA term is identity. You get exactly the base model. Training nudges A, B to produce the task-specific update.

</div>

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

# QLoRA · quantize the base

<div class="math-box">

**QLoRA** (Dettmers 2023) · quantize the frozen base to 4-bit NF4; train LoRA adapters in fp16 on top.

- Weights: 70B × 0.5 bytes = 35 GB (down from 140 GB).
- Gradients &amp; Adam state: only for LoRA params (~0.1%) — trivial.
- Fine-tune a 70B model on a single 48 GB GPU.

</div>

Used everywhere in the open-source LLM ecosystem for fine-tuning in 2024+.

---

<!-- _class: section-divider -->

### PART 3

# RLHF · reward model + PPO

The alignment loop that made ChatGPT

---

# Why SFT isn't enough

SFT teaches the model *one way* to respond to each prompt. But for open-ended questions, many responses are acceptable — and you want the model to prefer the *best* one.

> *"Explain quantum entanglement to a 12-year-old."*

- Response A · accurate, clear, uses an analogy. ✓
- Response B · accurate but terse and dry. ≈
- Response C · inaccurate, overly technical. ✗

SFT would weight all three equally if all are in the dataset. **RLHF** adds preference learning.

---

# The RLHF pipeline

![w:920px](figures/lec16/svg/rlhf_pipeline.svg)

---

# RLHF step-by-step

1. **SFT** · as before — teach the model to follow instructions.
2. **Reward model (RM)** · collect pairs $(x, y_\text{preferred}, y_\text{rejected})$ from human annotators. Train a classifier $r_\phi(x, y)$ to score responses.
3. **PPO optimization** · use the RM as a reward signal. Policy $\pi_\theta$ (the SFT model) gets updated to maximize expected reward:

<div class="math-box">

$$\max_\theta\, \mathbb{E}_{x \sim D}[\, r_\phi(x, \pi_\theta(x)) \,] \;-\; \beta\, D_\text{KL}(\pi_\theta \Vert \pi_\text{ref})$$

KL term keeps $\pi_\theta$ close to the SFT model — prevents reward hacking.

</div>

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

Bypass the reward model entirely

---

# DPO · the 2023 simplification

<div class="paper">

Rafailov et al. 2023 · *"Direct Preference Optimization: Your Language Model is Secretly a Reward Model."*

</div>

**Insight:** the RLHF objective has a closed-form optimal policy given the reward. Work backwards — derive a loss directly on preference pairs, skipping the RM and PPO entirely.

---

# The DPO loss

<div class="math-box">

$$\mathcal{L}_\text{DPO} = -\log \sigma\!\left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_\text{ref}(y_w \mid x)} \,-\, \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_\text{ref}(y_l \mid x)} \right)$$

For each preference pair $(x, y_w, y_l)$:
- $y_w$ is the preferred (winning) response.
- $y_l$ is the rejected (losing) response.
- $\pi_\text{ref}$ is the SFT model (frozen reference).
- $\pi_\theta$ is the trainable policy.

</div>

Pure supervised loss. No RL loop. No reward model. Implementation is ~50 lines vs RLHF's thousands.

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

Open-source (Hugging Face, Mistral, most Llama fine-tunes) · **DPO**. Frontier labs (Anthropic, OpenAI, Google) · **RLHF variants** with proprietary tooling. Both work.

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

Test-time compute as a new axis

---

# Reasoning models

Latest generation · **o1, o3, Claude extended thinking, DeepSeek R1.**

The core idea:

1. Train the model to produce **long internal chain-of-thought** before responding.
2. Use RL with **process rewards** (reward good reasoning steps, not just final answer).
3. At inference, let the model think for minutes — spend 10×–100× more compute.

Result: dramatically better on math, code, logic benchmarks.

<div class="insight">

Scaling laws in training compute produced pretrained capability. A new axis — **scaling test-time compute** — now unlocks reasoning. Both will likely continue.

</div>

---

<!-- _class: summary-slide -->

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
