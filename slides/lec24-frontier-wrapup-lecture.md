---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# Frontier — Agents · Reasoning · Interpretability

## Lecture 24 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar · Aug 2026*

---

# Learning outcomes

By the end of this lecture you will be able to:

1. Describe **agents** as the perceive-think-act loop with external tools.
2. Understand **test-time compute** scaling in reasoning models (o1, Claude).
3. Read the **residual stream view** of a Transformer.
4. Recognize **induction heads** and sparse-autoencoder features.
5. Articulate the **open problems** of DL (continual learning, data efficiency, grounding).
6. Identify **2026-2030 research directions** worth pursuing.

---

# The final lecture

23 lectures of theory + practice. Today — three threads that define the 2024–2026 frontier, plus a course recap.

<div class="paper">

Today's reading is a collection of blogs and papers · Yao 2022 (ReAct), Wei 2022 (CoT), Anthropic interp blog, OpenAI o1 blog.

</div>

Four questions:
1. What are **agents** and what changed in 2024?
2. What are **reasoning models** (o1, Claude thinking)?
3. What is **mechanistic interpretability**?
4. What are the open problems — and where does this course take you next?

---

<!-- _class: section-divider -->

### PART 1

# Agents &amp; tool use

2024's big shift

---

# From chatbot to agent

An LLM chatbot gives you text. An LLM **agent** can *act* — call tools, browse the web, write code, click buttons on a screen.

<div class="keypoint">

The difference: **closing the perceive-think-act loop** with external tools. The LLM becomes the reasoning layer of a larger system.

</div>

---

# The ReAct loop

![w:920px](figures/lec24/svg/react_loop.svg)

---

# ReAct · annotated with tools

![w:920px](figures/lec24/svg/react_agent_loop.svg)

---

# Function calling · how agents work

Modern APIs (Claude, OpenAI, Gemini) support **structured tool calling**:

```python
tools = [
    {
        "name": "search_flights",
        "description": "Find flights between two cities on a date",
        "input_schema": {
            "type": "object",
            "properties": {
                "from": {"type": "string"},
                "to":   {"type": "string"},
                "date": {"type": "string", "format": "date"}
            }
        }
    }
]
response = client.messages.create(model="claude-4", tools=tools, messages=[...])
```

The model returns a structured call · you execute the function · feed back the result · loop until done.

---

# Computer use · agents controlling UIs

Claude Computer Use (2024) · the model sees a **screenshot**, outputs **mouse + keyboard actions**.

```
Screenshot → Claude → "move_mouse(320, 450) ; click() ; type('hello')" → screenshot
```

This is RL-like but trained mostly supervised on human demonstrations and synthetic examples. Opens up:

- Browser automation
- Desktop task completion
- Form filling
- Legacy-app bridging

<div class="realworld">

This course itself was built largely by Claude Code — an agent loop over bash / edit / read tools. Agents are the **application layer** of 2026 AI.

</div>

---

<!-- _class: section-divider -->

### PART 2

# Reasoning models

Test-time compute as a new axis

---

# Chain-of-thought · the 2022 discovery

<div class="paper">

Wei et al. 2022 · prompting an LLM to "think step by step" dramatically improves multi-step reasoning at scale.

</div>

```
Q: John has 5 apples. He gives 2 to Mary and buys 4 more. How many?
A: He starts with 5. Gives 2 → 3. Buys 4 → 7. Final: 7.
```

CoT emerges at scale (~60B params). Below that, adding "think step by step" doesn't help.

---

# Reasoning models · train for CoT

2024's big idea · don't just *prompt* for CoT — **train** for it with RL.

**o1** (OpenAI 2024), **Claude extended thinking** (Anthropic 2024), **DeepSeek R1** (2025) all:

1. Generate candidate chains of thought.
2. Reward based on whether the final answer is correct (outcome reward) or whether reasoning steps are valid (process reward).
3. Fine-tune via RL to produce longer, more systematic internal reasoning.
4. At inference · spend 10×–100× more compute per answer, let the model "think."

---

# The scaling laws recap · in one figure

![w:900px](figures/lec24/svg/scaling_laws.svg)

---

# A new scaling axis

Until 2024, the only scaling axis was **training compute**. In 2024+ we added **test-time compute**.

<div class="keypoint">

**Two knobs now:** (a) spend more on pretraining → better general capability; (b) spend more on per-query reasoning → better on hard problems.

Both pay off. OpenAI reported that o1's performance on math benchmarks scales smoothly with inference-time compute budget.

</div>

---

# Reasoning · tree search at inference

![w:900px](figures/lec24/svg/reasoning_tree.svg)

---

# Reasoning models · benchmarks

| Model | AIME 2024 (math) | Codeforces |
|-------|------------------|------------|
| GPT-4 | 12% | ~800 Elo |
| o1 | 74% | ~1800 Elo |
| o3 | 97% | ~2700 Elo (grandmaster) |

Comparable gains on HumanEval, MATH, GPQA. This is an entirely new capability curve.

<div class="realworld">

Practical rule · if the problem needs multi-step reasoning, use a reasoning model. If it needs fast response or is simple, use a regular model.

</div>

---

<!-- _class: section-divider -->

### PART 3

# Mechanistic interpretability

What is the model actually doing?

---

# The problem

A 70B Llama has 70 billion parameters. We trained it, and it works. But if it produces a wrong answer — or a dangerous one — **we can't read the weights and know why**.

Mechanistic interpretability (**mech interp**) tries to reverse-engineer specific computations inside trained networks.

<div class="paper">

Anthropic's interpretability team · Olah et al. circuits research 2020+, sparse autoencoders 2023+, dictionary learning 2024+.

</div>

---

# Induction head · a discovered circuit

![w:920px](figures/lec24/svg/mech_interp_circuit.svg)

---

# The residual stream view

Elhage et al. (Anthropic) · reframe a Transformer as a **residual stream** that every block reads from and writes to:

$$h_{l+1} = h_l + \text{attn}_l(h_l) + \text{ffn}_l(h_l)$$

The residual stream is a "bus" carrying information across layers. Attention heads read features and write new features back.

This perspective helped find **circuits** — specific computational pathways inside a Transformer that implement identifiable algorithms (induction heads, IOI task, prime-number detection, etc.).

---

# Sparse autoencoders · feature dictionary

Circuits interpretability struggled because individual neurons represent **superpositioned** features (many concepts per neuron).

**Sparse autoencoders** (SAE) are trained to expand the residual stream into a much wider, sparse representation:

```
residual stream (d=12,288) → SAE → sparse features (d=100,000+, mostly zero)
```

Each feature in the SAE often corresponds to a **human-interpretable concept** · "the Golden Gate Bridge", "code syntax errors", "requests for help."

<div class="realworld">

Anthropic 2024 · trained an SAE on Claude 3 Sonnet and found millions of features. Clamping specific features changes model behavior ("the Golden Gate Claude" demo).

</div>

---

# Why mech interp matters

Two reasons:

1. **Safety** · identify and edit circuits that produce harmful outputs.
2. **Science** · understand what representations language models actually learn, and how.

<div class="insight">

Still a young field. Most interp results are about small models or narrow circuits. Scaling interp to frontier-level models is a 2026+ research agenda.

</div>

---

<!-- _class: section-divider -->

### PART 4

# Open problems

What the next decade of DL research looks like

---

# Safety · why alignment matters

As models become more capable, the cost of misalignment grows:

<div class="math-box">

| 2018 | 2024 | 2030 (?) |
|:-:|:-:|:-:|
| Misclassify image | Give wrong factual answer | Autonomously execute bad plan |
| Cost: annoy user | Cost: spread misinformation | Cost: catastrophic |

</div>

<div class="keypoint">

**Claude, GPT, Gemini all ship with elaborate safety stacks** · constitutional AI, RL from safety feedback, red-teaming, classifier filters, refusal training. Safety is not a layer; it's the product.

</div>

---

# What you've learned · a final recap

<div class="math-box">

| Module | Covered |
|:-:|:-:|
| **Foundations** (L1-L2) | why DL, UAT, depth vs width, residuals |
| **Training craft** (L3-L6) | recipe, SGD / Adam, schedules, regularization |
| **Vision** (L7-L9) | CNN mechanics, ResNet family, detection, SAM |
| **Sequences → Transformers** (L10-L14) | RNN/LSTM/GRU, Seq2Seq, attention, Transformer, tokenization |
| **LLMs** (L15-L16) | scaling laws, RoPE, GQA, LoRA, RLHF, DPO |
| **Self-supervision + VLMs** (L17-L18) | SimCLR, MAE, CLIP, LLaVA |
| **Generative** (L19-L22) | VAE, GAN, DDPM, CFG, latent diffusion |
| **Systems + frontier** (L23-L24) | KV-cache, quantization, agents, reasoning, interp |

</div>

---

# Open problems · the short list

1. **Reasoning reliability** · even o3 still hallucinates; trust calibration is unsolved.
2. **Continual learning** · models can't easily update with new facts without retraining.
3. **Data efficiency** · humans learn from 10³ examples; models need 10¹². Why?
4. **Alignment** · how do we ensure advanced systems remain beneficial?
5. **Interpretability at scale** · we barely understand what's inside a 70B model.
6. **Multi-modal reasoning** · image+text+video+action reasoning jointly is still weak.
7. **Grounding** · LLMs know about the world through text; they don't have embodied experience.
8. **Energy &amp; cost** · a GPT-4 query costs ~pennies but society-scale deployment is energy-intensive.

Each is a PhD worth of work. Pick one.

---

# Where's the field going?

Predictions (take with salt):

- **Reasoning compute** will scale 10× per year for a few years — expect ~1000× by 2028.
- **Agentic AI** will move from demos to production workflows in 2026.
- **Multimodal** will absorb audio, video, 3D into the same models.
- **Open-weight models** will continue closing the frontier gap (Llama, Mistral, DeepSeek).
- **Domain-specific** small models (medical, legal, scientific) will win niche deployments.
- **Safety &amp; alignment** research will become mainstream.

---

<!-- _class: section-divider -->

### PART 5

# Course recap

What you learned

---

# The 24-lecture arc

| Module | Lectures | Big ideas |
|--------|----------|-----------|
| 1 Foundations | L1–L3 | MLP, ResNets, training recipe |
| 2 Optimization | L4–L5 | SGD, momentum, Adam, schedules |
| 3 Regularization | L6 | double descent, augmentation, norm, dropout |
| 4 CNNs | L7–L9 | architecture evolution, detection, SAM |
| 5 Sequences | L10–L11 | LSTM, Seq2Seq, bottleneck |
| 6 Transformers | L12–L14 | attention, nanoGPT, tokenization |
| 7 LLMs | L15–L16 | scaling laws, LoRA, RLHF, DPO |
| 8 SSL + VLM | L17–L18 | SimCLR, CLIP, LLaVA |
| 9 Generative | L19–L22 | VAE, GAN, DDPM, Stable Diffusion |
| 10 Frontier | L23–L24 | inference, agents, interp |

---

# What you can now do

1. **Read any 2026 ML paper** and understand the architecture.
2. **Implement a Transformer, a diffusion model, a LoRA fine-tune** from scratch.
3. **Know when to use what** — CNN vs ViT, SGD vs AdamW, RLHF vs DPO.
4. **Debug training failures** using the ladder and error analysis.
5. **Estimate compute + memory** for any training or inference setup.
6. **Build an agent** that uses tools to accomplish tasks.

This is the current skill floor for a DL engineer or research student.

---

# What to read next

- **Bishop &amp; Bishop** · *Deep Learning: Foundations and Concepts* — for more mathematical rigor.
- **Karpathy's Zero to Hero** — keep going beyond what we covered.
- **Prince's UDL** — revisit the chapters you skimmed.
- **Recent papers** — set up arXiv alerts for your interest area.
- **Blog posts** · Lil'Log (Lilian Weng), Simon Willison, Chip Huyen, Anthropic engineering blog.

---

# What to do next

- **Replicate a paper** · picking one from NeurIPS / ICML / ICLR 2025 and implementing it end-to-end teaches you more than any course.
- **Contribute to open source** · HuggingFace, vLLM, PyTorch. Start with issues labeled "good first issue."
- **Build a small project** · fine-tune an LLM on your domain; train a tiny diffusion model on a toy dataset; ship an agent.
- **Read safety &amp; alignment work** · if you care about where this technology is going.

---

<!-- _class: summary-slide -->

# Lecture 24 — summary

- **Agents** · LLM + tools in a ReAct loop. 2024's biggest application-layer shift.
- **Reasoning models** · train for long chains of thought with RL; spend more test-time compute.
- **Mech interp** · residual-stream view + sparse autoencoders; progress but still early.
- **Open problems** · reliability, continual learning, alignment, interpretability, grounding.
- **What you have now** · end-to-end understanding of every major DL system in 2026.

---

<!-- _class: title-slide -->

# Thank you.

## ES 667 · Deep Learning · Aug 2026

**Prof. Nipun Batra**
*IIT Gandhinagar*

*Go build something.*

<div class="notebook">

**Final project** · apply a technique from any lecture to a real problem · 3-week timeline · pitch week after endsem.

</div>
