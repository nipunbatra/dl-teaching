---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# The Frontier — and the Whole Course in One Idea

## Lecture 26 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar*

---

# The whole lecture, in one idea

The 2024–2026 frontier looks like three new inventions — **agents**, **reasoning models**, **mechanistic interpretability**. It is none. It is the *same* differentiable module + cross-entropy + scale, given three things it lacked.

<div class="keypoint">

**Nothing at the frontier is new math.** Agents give the model **hands** (tools + an action loop). Reasoning models give it **scratch paper** (test-time compute). Interpretability gives *us* a **microscope** (the residual stream + features). Underneath, it is still $-\sum_i \log p_\theta(y_i\mid x_i)$ — the loss from Lecture 1.

</div>

We'll end by walking back through all 26 lectures to the very first classifier — **oranges vs tomatoes** — and show it was cross-entropy all along. *Only the scale changed.*

---

# How today connects to the whole course

<div class="columns">
<div>

**Today rewires ideas you already own:**
- **Agents** = LLM (**L15**) + tool-calling (a function call wrapped in a prompt) + retrieval / RAG
- **Reasoning** = LLM + **RL on chain-of-thought** — the RLHF machinery from **L18**, pointed at the *thinking*
- **Interp** = read the attention (**L14**) and the residual stream (**L15**) with new tools

</div>
<div>

**And closes the loop back to the start:**
- the loss is still the **NLL / cross-entropy** of **L1**
- the estimator is still **Adam + backprop** (**L4–L6**)
- the very first model — **oranges vs tomatoes**, ES 335 — is the *same equation* a frontier LLM optimizes

</div>
</div>

<div class="insight">

This is a **wrap-up**: fast on the machinery you know, slow on the *one sentence* that ties 26 lectures together.

</div>

---

<!-- _class: section-divider -->

## Part 1 · Agents &amp; tool use

*What changes when the model can act, not just answer?*

---

# A standard LLM is a brain in a jar

<div class="keypoint">

A plain LLM is a **brilliant brain in a jar**. It can talk, write, and reason — but it cannot *do* anything in the world. It has no hands and no eyes.

**Agents** give that brain a body: web browsers, code editors, file systems, calculators, APIs.

</div>

The model keeps its strengths — knowledge, language, reasoning — and gains the one missing piece: **action**. The same model that can *describe* how to book a flight can now actually book it.

---

# From chatbot to agent

An LLM **chatbot** returns text. An LLM **agent** *acts* — it calls tools, browses, writes and runs code, clicks buttons on a screen.

<div class="keypoint">

The whole difference is **closing a perceive → think → act loop** around the model. The LLM stops being the *product* and becomes the **reasoning layer** of a larger system that observes results and tries again.

</div>

Everything in Part 1 is one question: *how do we wire that loop, and how does it fail?*

---

# Agents, in one sentence

Ng's one-liner: **an agent is an LLM in a loop with tools** — *reason, act, observe, repeat.*

<div class="keypoint">

**Function calling is the "act."** The model emits a **structured action** — JSON saying *which* tool and *what* arguments. *You* run it. The return value becomes the next **observation**, appended to the context. The network's weights never change; what's new is the **loop wrapped around it**.

</div>

<div class="insight">

If you've met RL (**L18**), this is the same **policy ↔ environment** picture: the LLM is the *policy* choosing an action; your `run(...)` code is the *environment* returning an observation. The twist — unlike RLHF, we do **not** update weights here. We close the loop entirely at **inference time**. The model *proposes*; the world *disposes*; the outcome re-enters the prompt.

</div>

---

# The ReAct loop

Yao et al. (2022) named the pattern: **Reason → Act → Observe**, repeated until the task is done.

![w:680px](figures/lec24/svg/react_loop.svg)

The model *reasons* about what to do, *acts* by emitting a tool call, then *observes* the result — and that observation goes back into the context for the next step.

---

# ReAct, annotated with tools

![w:680px](figures/lec24/svg/react_agent_loop.svg)

Each **Act** is a call to an external tool (search, calculator, code runner); each **Observe** is the tool's return value, appended to the running transcript.

---

<!-- _class: code-heavy -->

# Function calling · how agents actually work

Modern APIs (Claude, OpenAI, Gemini) expose **structured tool calling**. You describe a tool as JSON; the model chooses when to call it and with what arguments.

```python
tools = [{
    "name": "search_flights",
    "description": "Find flights between two cities on a date",
    "input_schema": {
        "type": "object",
        "properties": {
            "from": {"type": "string"},
            "to":   {"type": "string"},
            "date": {"type": "string", "format": "date"},
        },
    },
}]
response = client.messages.create(model="claude", tools=tools, messages=[...])
```

The model returns a **structured call**, not prose. The tool schema is *just part of the prompt* — the "act" step is next-token prediction constrained to valid JSON.

---

<!-- _class: code-heavy -->

# The loop, in ten lines

The API gives you one turn. *You* write the loop — execute the tool, feed the result back, repeat.

```python
messages = [user_query]
while True:
    resp = client.messages.create(model="claude", tools=tools, messages=messages)
    if resp.stop_reason != "tool_use":
        break                                   # model is done → final answer
    for call in resp.tool_calls:                # ACT
        result = run(call.name, call.input)     # YOU execute the tool
        messages += [call, tool_result(result)] # OBSERVE → back into context
```

`run(...)` is ordinary code — hit an API, run a query, execute Python. The LLM only ever decides *which* tool and *what* arguments; the environment does the rest.

---

# Computer use · agents that drive a screen

The most general tool is a whole computer. **Claude Computer Use** (2024) sees a **screenshot** and emits **mouse + keyboard actions**:

```
screenshot → model → move_mouse(320, 450); click(); type("hello") → screenshot
```

Same ReAct loop — the "observation" is now a pixel grid, the "action" is a UI event. It unlocks browser automation, desktop task completion, form filling, and bridging to legacy apps with no API.

---

# What agents add on top of the LLM

<div class="columns">
<div>

**Three building blocks, all things you've seen:**
- **Tools** — function calling (structured next-token output)
- **Retrieval (RAG)** — fetch relevant context, then condition on it (in-context learning, L15)
- **Memory** — persist state across turns; the KV-cache / transcript *is* short-term memory

</div>
<div>

**Wired into the loop:**
- plan → call tool → read result → revise plan
- multi-agent: one agent's output is another's input
- the model is the *controller*; the tools are the *effectors*

</div>
</div>

<div class="realworld">

This course itself was built largely by **Claude Code** — an agent loop over `bash` / `read` / `edit` tools. As of this writing, agents are the **application layer** of AI: the frontier ships as *loops around a model*, not the model alone.

</div>

---

# Multi-agent systems · a team of specialists

One loop can plan, act, and observe. But a long job — *"research this topic and write a report"* — is often better **split across several agents**, each with its own tools, context, and role.

<div class="columns">
<div>

**Orchestrator → workers**
- a **lead** agent decomposes the task and spawns **sub-agents**
- each sub-agent runs its *own* ReAct loop on a slice of the problem
- the lead **merges** their results into one answer

</div>
<div>

**Why bother?**
- **parallelism** — sub-agents hit different sources at once
- **fresh context** — each gets a clean window, dodging context rot
- **specialisation** — a "searcher," a "coder," a "critic"

</div>
</div>

<div class="realworld">

It's turtles all the way down: a sub-agent is *just another LLM in a loop*, and one agent's output is the next one's input. As of this writing, production systems (deep-research modes, coding swarms) are orchestrations of many such loops — but every extra agent **multiplies the failure surface** of the next slide.

</div>

---

# Where agents break

<div class="insight">

A single tool call is reliable. A **20-step** plan multiplies the failure rates: if each step is 95% reliable, the whole chain is $0.95^{20}\approx 0.36$. Errors **compound** over long horizons — this is the central unsolved problem of agentic AI as of this writing.

</div>

So the research is not "smarter single answers" but **reliability across many steps**: better planning, error recovery, verification, and knowing when to stop. Keep this in mind — it reappears in the open-problems slide.

---

# Agent failure modes · a field guide

Compounding error is *why* agents fail; here is *how* — the modes you will actually debug:

| Failure mode | What it looks like | Blunt fix |
|:--|:--|:--|
| **Hallucinated tool call** | invents a tool or an argument that doesn't exist | strict schema validation; reject &amp; retry |
| **Error cascade** | one bad observation poisons every later step | checkpoints; let the agent backtrack |
| **Doom loop** | calls the same tool forever, never converging | step budget; loop detection |
| **No stop signal** | keeps "improving" an already-finished answer | an explicit success criterion |
| **Context rot** | transcript outgrows what the model can attend to | summarise / prune old turns |
| **Goal drift** | quietly solves a *different* task than asked | re-state the goal each turn; a verifier |

<div class="insight">

Notice the fixes are **engineering**, not new math — validation, budgets, memory management, verification. Reliability is *the* product problem of agentic AI as of this writing, which is why "smarter agent" usually means "more disciplined loop," not "bigger model."

</div>

---

# Agents in the wild · what you can build now

Concrete systems shipping today (as of this writing) — every one is *the same loop* over a different toolset:

<div class="columns">
<div>

- **Coding agents** — read / edit / run over a repo (this course was built by one). Tools: `bash`, `read`, `edit`, `test`.
- **Deep research** — a lead spawns searchers, reads sources, writes a cited report. Tools: `web_search`, `fetch`.
- **Computer use** — drive a browser or desktop by screenshot → click. Tools: `screenshot`, `mouse`, `keyboard`.

</div>
<div>

- **Customer support** — look up an order, issue a refund, escalate. Tools: internal APIs.
- **Data analyst** — write SQL, run it, plot, explain the result. Tools: `sql`, `python`.
- **Multi-agent swarms** — planner + workers + critic on one long job.

</div>
</div>

<div class="realworld">

The pattern is identical every time: *reason → call a tool → read the result → repeat.* Only the **toolset** and the **stopping rule** differ. If you can name a tool and a success criterion, you can build the agent — the hard part is the **reliability** from the failure-modes table, not the model.

</div>

---

# See the pieces yourself

<div class="notebook">

**🎛 Interactive · the agent's building blocks** — the retrieval and prompting mechanisms agents wire together, each as a scroll-driven explainer: **RAG** (retrieve → condition), **in-context learning** (learn from the prompt, no weight update), and the **KV-cache** (why the transcript is cheap memory). *(Interactive Lab · `interactive/src/articles/{rag, in-context-learning, kv-cache}`)*

</div>

<div class="popquiz">

**Think ahead.** An agent must answer *"How many days until the next public holiday in my state?"* Which tool(s) does it need, and in what order? *(Next slide traces it.)*

</div>

---

# Practice problem 1 · trace a ReAct loop

<div class="popquiz">

**Practice problem 1.** An agent has two tools: `web_search(query)` and `calculator(expr)`. The user asks:

> *"What is the population of the capital of France, multiplied by 2?"*

Write out the **Reason → Act → Observe** steps the agent takes, in order, until it can answer. How many tool calls does it make?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 1

| Step | Reason | Act | Observe |
|--|--|--|--|
| 1 | I need the capital of France | `web_search("capital of France")` | "Paris" |
| 2 | Now I need Paris's population | `web_search("population of Paris")` | "≈ 2.1 million" |
| 3 | Now multiply by 2 | `calculator("2100000 * 2")` | 4200000 |
| 4 | I have the answer | *(stop)* | — |

**Answer:** ≈ **4.2 million**, after **3 tool calls**.

<div class="keypoint">

The model never did arithmetic or lookup *itself* — it **decided which tool to call and read the result back**. Each observation re-entered the context, exactly the loop from the ten-line snippet. That is the whole of "agentic AI."

</div>

---

<!-- _class: section-divider -->

## Part 2 · Reasoning models

*What if a model could think longer on harder questions?*

---

# Test-time compute · the college-exam analogy

<div class="keypoint">

Compare a model to a student. **Training compute** = the years in college learning general knowledge — paid once, up front.

Faced with a *hard* exam question, a good student doesn't blurt the first thing. They **pause, sketch on scratch paper, double-check.** That extra effort *per question* is **test-time compute**.

</div>

Reasoning models (OpenAI o1/o3, Claude extended thinking, DeepSeek R1) are the same "college graduate" base model — but allowed a scratchpad of internal chain-of-thought before answering. Brief here, because you've met the pieces: it's the **RLHF machinery of L18 pointed at the thinking**.

---

# Chain-of-thought · the 2022 discovery

<div class="paper">

Wei et al. (2022): prompting an LLM to *"think step by step"* sharply improves multi-step reasoning — at scale.

</div>

```
Q: John has 5 apples. He gives 2 to Mary and buys 4 more. How many?
A: Start with 5. Give 2 → 3. Buy 4 → 7. Final: 7.
```

Writing the intermediate steps lets the model *use its own output as working memory*. In the original work the gains appeared mainly above ~tens of billions of parameters — though smaller models can still learn it by distillation.

---

# Train for thinking · outcome vs process rewards

<div class="insight">

Grading a student's math solution two ways:
- **Outcome reward** — *"the final answer is 7."* Easy to check; says nothing about *how*.
- **Process reward** — *"step 3 is where you went wrong."* Much richer signal.

</div>

The 2024 idea: stop merely *prompting* for chains of thought — **train** for them with RL.

1. **Generate** many candidate chains for a hard question.
2. **Reward** them — mostly on **outcome** (final answer correct?), sometimes on **process**. Exact recipes are mostly proprietary.
3. **Fine-tune via RL** so high-reward chains become more likely.
4. **At inference**, spend 10×–100× more compute and let the model think.

---

# Worked example · rewarding chains of thought

**Question.** A bat and a ball cost \$1.10 total. The bat costs \$1.00 *more than* the ball. How much is the ball?

<div class="columns">
<div>

**Candidate 1 (greedy).**
Bat is \$1.00 → ball $=1.10-1.00=\$0.10$. **Answer: 10¢.**
- Outcome reward **0** (wrong — the trap).
- Process reward low (step 1 was an unchecked assumption).

</div>
<div>

**Candidate 2 (algebra).**
$B+L=1.10,\; B=L+1 \Rightarrow 2L+1=1.10 \Rightarrow L=0.05$. **Answer: 5¢.**
- Outcome reward **+1**.
- Process reward high — every step follows.

</div>
</div>

RL makes traces **like Candidate 2** more likely. The model isn't memorizing "5¢" — it's learning the *habit* of setting up the equation.

---

# A new scaling axis

Until 2024 the headline knob was **training compute** (the scaling laws of L17). In 2024, **test-time compute** became a first-class axis too.

![w:640px](figures/lec24/svg/scaling_laws.svg)

<div class="keypoint">

**Two knobs now:** (a) more pretraining → better *general* capability; (b) more per-query reasoning → better on *hard* problems. Both curves keep paying off — as of this writing, o-series accuracy on math benchmarks rises smoothly with the inference-time budget.

</div>

---

# Search at inference time

Spending test-time compute often *looks like search*: sample several chains, explore branches, keep the best (self-consistency, tree-of-thoughts).

![w:620px](figures/lec24/svg/reasoning_tree.svg)

These inference-time search ideas predate o1 — o1's contribution was **productizing** them into a single model you can just call.

---

# Self-consistency · think several times, then vote

The simplest way to *spend* test-time compute: sample **several independent chains of thought**, then keep the **majority answer**. Independent mistakes scatter; the correct path tends to agree with itself.

<div class="math-box">

**Back-of-envelope.** One chain reaches the right final answer with probability $0.6$, and its wrong answers scatter. Sample **5** chains and take a majority vote:
- a single chain: $0.6$ → **60%**.
- majority-of-5 (needs $\ge 3$ correct): $\displaystyle\sum_{k\ge 3}\binom{5}{k}(0.6)^k(0.4)^{5-k}\approx \mathbf{68\%}$.

</div>

<div class="insight">

"Let it think longer" often really means "let it think **several times**." Best-of-$n$ and tree-of-thoughts are the same idea with a smarter selector than a plain vote. As of this writing, self-consistency is one of the cheapest reliable wins in the reasoning toolkit — **no retraining, just more inference** — and it is literally the test-time-scaling curve of the previous slide.

</div>

---

# Reasoning models · the numbers, and the honest caveat

| Model | AIME 2024 (math) | Codeforces |
|--|--|--|
| GPT-4o | 12% | ~800 Elo |
| o1 | 74% | ~1800 Elo |
| o3 | 97% | ~2700 Elo (grandmaster) |

Codeforces Elo is a competitive-programming rating (like chess) — ~2700 is top-human. Comparable jumps on MATH, GPQA, HumanEval.

<div class="realworld">

**Practical rule:** multi-step reasoning → use a reasoning model; fast/simple → use a regular one. And stay honest — a reasoning model is **still next-token prediction with a scratchpad**, not a proof engine. It can think longer *and* be confidently wrong. (Numbers are vendors' reported results, as of this writing — indicative, not gospel.)

</div>

---

# Distilling reasoning · small models can think too

If a big model can *generate* good chains of thought, why not use those chains as **training data** for a small one?

<div class="keypoint">

**Reasoning distillation:** collect thousands of *correct* chains from a strong reasoning model, then **fine-tune** a small model to imitate them — ordinary supervised cross-entropy (the loss from L1, the fine-tuning setup from L18). The small model inherits much of the *reasoning behaviour* without ever running the expensive RL.

</div>

This is plain **knowledge distillation** (student mimics teacher), pointed at *reasoning traces*. DeepSeek-R1 (2025) distilled its chains into 7B–70B open students that then jumped on math and code.

<div class="realworld">

Honest limit (as of this writing): distillation copies the teacher's *habits*, not its *ceiling* — a distilled 7B won't out-reason its teacher. But it makes "thinking" cheap enough to run locally: capability, once discovered, gets **smaller and cheaper** fast.

</div>

<div class="notebook">

**🎛 Interactive · knowledge distillation** — a small student learns to match a large teacher's soft targets. *(Interactive Lab · `interactive/articles/knowledge-distillation`)*

</div>

---

<!-- _class: section-divider -->

## Part 3 · Mechanistic interpretability

*What is the model actually doing inside?*

---

# The problem · we can't read the weights

A 70B model has 70 billion parameters. We trained it; it works. But when it gives a wrong — or dangerous — answer, **we cannot open the weights and see why.**

**Mechanistic interpretability** ("mech interp") tries to *reverse-engineer specific computations* inside a trained network — to turn weights into circuits we can name.

<div class="paper">

Anthropic's interpretability program: Olah et al. circuits (2020+), sparse autoencoders (2023+), dictionary learning and *Golden Gate Claude* (2024).

</div>

---

# The residual stream · a shared whiteboard

<div class="insight">

A Transformer layer is a set of experts working on a shared **whiteboard** — the **residual stream**.

1. The input document is written on the board.
2. **Attention** reads it, writes a sticky note ("this word relates to that one"), and **adds** it — it never erases.
3. The **FFN** reads board + note, writes its own, adds it.
4. The updated board is the next layer's input.

</div>

$$h_{l+1} = h_l + \text{attn}_l(h_l) + \text{ffn}_l(h_l)$$

**Toy, $d=4$:** $h_5=[0.2,0.5,0.1,-0.9]$, attention adds $[0,0.3,0.4,0]$, FFN adds $[0.1,-0.1,0,0.2]$ → $h_6=[0.3,0.7,0.5,-0.7]$. Every layer just **reads the bus, adds a contribution, writes back** — which is exactly what makes circuits findable.

---

# Induction heads · a discovered circuit

![w:680px](figures/lec24/svg/mech_interp_circuit.svg)

An **induction head** implements "…$A$ $B$ … $A$ → predict $B$": it finds the earlier place the current token appeared and copies what came next. A concrete, named algorithm — recovered from the attention pattern, not the training code.

---

# Superposition &amp; sparse autoencoders

<div class="keypoint">

**Superposition:** a single neuron represents *many* ideas at once — "bank" fires for river bank *and* financial bank *and* context-dependent shades. Great for compression, terrible for analysis.

A **sparse autoencoder (SAE)** forces the model to use a **giant explicit dictionary**: encode $x$ into a very wide, mostly-zero vector, so distinct **river_bank** and **financial_institution** features fire separately.

</div>

The dictionary is *wider* than the residual stream (e.g. 100k features for a 12k-dim stream), with a **sparsity penalty** so only a handful are active per input — and those active ones tend to be **human-interpretable concepts**. It's an *inverted* bottleneck: *"100,000-word vocabulary, but you may use only 5 words"* — so each word must be precise.

---

# Practice problem 2 · read an SAE feature

<div class="popquiz">

**Practice problem 2.** A residual-stream vector $x = [0.9,\, 0.8,\, -0.7,\, 0.1]$ is fed to a trained SAE. Its (sparse) encoding is

$$f = [\,0,\ 0,\ 0,\ 0,\ 0,\ \mathbf{0.95},\ 0,\ 0,\ 0,\ 0\,]$$

**(a)** The raw $x$ looks meaningless — why? **(b)** How would you find out *what concept* feature 6 represents? **(c)** If you clamp feature 6 to a large value on every token, what happens to the model's behaviour?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 2

**(a)** $x$ is in **superposition** — its four numbers mix many concepts, so no single coordinate "means" anything. The SAE *disentangles* it: one clean feature is active.

**(b)** Collect **every input that activates feature 6** and look for what they share. In Anthropic's Claude 3 Sonnet SAE, the answer for one such feature was: *all about the* **Golden Gate Bridge**. Label it → *"Feature 6 = Golden Gate Bridge."*

**(c)** Clamping it high makes the model steer *everything* toward that concept — the famous **"Golden Gate Claude"** demo, which brought up the bridge no matter what you asked.

<div class="keypoint">

That's the payoff: features are **directions we can name *and* steer** — the first real handle on the inside of a frontier model. Still early — most results are small models or narrow circuits; scaling interp is the open agenda. *(Explore: Interactive Lab `articles/attention`; Anthropic's public feature browser, transformer-circuits.pub.)*

</div>

---

# Feature steering · Golden Gate Claude, concretely

Once a feature has a name, you can **turn its knob** — no retraining, no prompt. Anthropic (2024) isolated a feature that fires on the **Golden Gate Bridge**, clamped it to many times its normal value, and released **"Golden Gate Claude."**

```
User:    What's a good recipe for banana bread?
Normal:  Mash 3 bananas, mix with flour, sugar, eggs; bake at 175°C…
Steered: You'll want to bake it while gazing at the Golden Gate Bridge —
         its 1.7-mile art-deco span glowing orange over the fog-lit bay…
```

<div class="insight">

The model was **not** fine-tuned and **not** prompted to mention the bridge — a single **direction in the residual stream** was amplified, and it colored *every* answer. Concepts live as directions, and directions are things you can **add**. Turn a "toxic language" feature *down* and the model gets safer, with the weights untouched.

</div>

<div class="realworld">

The demo was a toy; the **mechanism** is the prize — editing behaviour at the level of a **named concept** rather than by retraining. Still early as of this writing: features are noisy and coverage is partial, but this is the first handle of its kind.

</div>

---

# The payoff · name a feature, monitor it, steer it

Golden Gate Claude was a *demo*. The prize is three verbs that clean SAE features unlock — the first practical handles on a model's insides:

<div class="columns">
<div>

**Name** — gather every input that fires a feature and label the shared concept ("Golden Gate Bridge," "insecure code," "sycophancy").

**Monitor** — watch a feature light up *while the model runs* — an early-warning gauge for deception, jailbreaks, or off-task behaviour.

</div>
<div>

**Steer** — clamp a feature up or down and change behaviour **without retraining**: dial *down* a "toxicity" feature, dial *up* "caution."

</div>
</div>

<div class="keypoint">

Together these turn a black box into something you can **audit and adjust at the level of concepts**, not raw weights. As of this writing it's still early — features are noisy, coverage is partial, mostly on smaller models — but it's the most promising road from *"it works"* to *"we know why."*

</div>

---

# See the model's insides yourself

<div class="notebook">

**🎛 Interactive · attention, visualized** — edit a sentence and watch which tokens attend to which; an induction head's "…A B … A → B" copying shows up as a bright off-diagonal stripe. Every head reads and writes the same residual-stream bus. *(Interactive Lab · `interactive/src/articles/attention`, `positional-encoding`)*

</div>

<div class="notebook">

**🔬 Explore · a real feature browser** — thousands of named SAE features pulled from a production model, each shown with the inputs that light it up (the *Golden Gate* feature is in there). *(Anthropic, "Scaling Monosemanticity," `transformer-circuits.pub` · as of this writing)*

</div>

---

<!-- _class: section-divider -->

## Part 4 · Open problems &amp; safety

*What's still unsolved?*

---

# Practice problem 3 · why long agents fail

<div class="popquiz">

**Practice problem 3.** An agent completes a task with a **12-step** plan. Each step succeeds independently with probability **0.9**.

**(a)** What is the end-to-end success rate? **(b)** Roughly how many steps until the plan is *more likely to fail than succeed*? **(c)** Name two *engineering* fixes — not "a bigger model" — that raise the end-to-end number.

</div>

*Try it before the next slide.*

---

# Solution · practice problem 3

**(a)** $0.9^{12}\approx \mathbf{0.28}$ — a 90%-reliable step, chained 12 times, is right only about a quarter of the time.

**(b)** Solve $0.9^{n}=0.5 \Rightarrow n=\dfrac{\ln 0.5}{\ln 0.9}\approx \mathbf{6.6}$ — by the **7th step** the plan is more likely wrong than right.

**(c)** Any two of: **verify / checkpoint** each step before continuing · **validate-and-retry** tool calls · let the agent **backtrack** from a bad observation · a **step budget + loop detection** · **decompose** into shorter sub-agent loops with fresh context.

<div class="keypoint">

This is the arithmetic behind *"reliability at long horizons"* — the central open problem of agentic AI (as of this writing). Push per-step reliability from $0.9$ to $0.99$ and $0.9^{12}=0.28$ becomes $0.99^{12}=0.89$. **The product is brutal; disciplined loops — not bigger models — are what tame it.**

</div>

---

# Open problems &amp; safety · one honest slide

As models act more autonomously, the *cost of being wrong* grows: misclassify an image (2018) → state a wrong fact (2024) → **autonomously execute a bad plan** (2030?). This is why every frontier model ships with a real safety stack — constitutional AI, RL from safety feedback, red-teaming, refusal training. **Safety is not a layer; it's part of the product.**

<div class="columns">
<div>

**Genuinely unsolved (pick one for a thesis):**
- **Reliability at long horizons** — errors compound over multi-step agents
- **Continual learning** — updating facts without full retraining
- **Data efficiency** — humans learn from $10^3$, models need $10^{12}$

</div>
<div>

**And:**
- **Alignment** — keeping capable systems beneficial
- **Interpretability at scale** — from toy circuits to 70B models
- **Grounding** — text-only knowledge, no embodied experience

</div>
</div>

No hand-waving about 2030: these are *open* — the honest state of the field as of this writing.

---

<!-- _class: section-divider -->

## Part 5 · The full-circle recap

*What did 26 lectures add up to?*

---

# The 26-lecture arc

| Module | Lectures | Big ideas |
|--|--|--|
| Foundations | L1–L3 | distributions & losses, regression → MLP, depth & UAT |
| Optimization | L4–L6 | backprop / autograd, SGD, momentum & Adam |
| Training craft | L7–L8 | regularization, normalization, the training recipe |
| Vision / CNNs | L9–L11 | CNNs, modern architectures & transfer, detection & segmentation |
| Sequences | L12–L13 | RNN / LSTM / GRU, seq2seq & embeddings |
| Attention & Transformers | L14–L15 | attention, the Transformer block |
| LLMs | L16–L18 | tokenization & pretraining, scaling & systems, alignment |
| Representation | L19–L20 | self-supervised / contrastive, ViT & VLMs |
| Generative | L21–L24 | autoencoders / VAE, GANs, diffusion (theory & practice) |
| Frontier | L25–L26 | efficient inference; agents, reasoning, interpretability |

Ten modules — but, we're about to see, **one idea**.

---

# The whole arc, one sentence per module

Those ten rows are really **seven movements of one melody** — each a single sentence:

1. **Foundations (L1–L3)** — a model outputs a *distribution*; every loss is its negative log-likelihood; a nonlinearity makes an MLP a universal fitter.
2. **Optimization (L4–L8)** — backprop gives the gradient for free; SGD → momentum → Adam, plus regularization and normalization, make it *generalize*.
3. **Vision (L9–L11)** — convolution bakes in translation invariance; depth, transfer, detection, and segmentation follow.
4. **Sequences (L12–L13)** — RNN / LSTM / GRU and seq2seq give order a memory; embeddings turn tokens into vectors.
5. **Attention → LLMs (L14–L18)** — attention replaces recurrence; Transformer + scale + alignment = a language model on the *same* cross-entropy.
6. **Representation &amp; generation (L19–L24)** — self-supervision learns features label-free; VAE / GAN / diffusion learn a distribution you can *sample*.
7. **Frontier (L25–L26)** — efficient inference makes it cheap; agents, reasoning, and interpretability add a loop, a scratchpad, and a microscope.

<div class="keypoint">

Seven movements, one loss — the **cross-entropy you first met classifying oranges from tomatoes.** *Only the scale changed.*

</div>

---

# What you can now build

1. **Read any current ML paper** and place its architecture.
2. **Implement from scratch** a Transformer, a diffusion model, a LoRA fine-tune.
3. **Choose the right tool** — CNN vs ViT, SGD vs AdamW, RLHF vs DPO, regular vs reasoning model.
4. **Debug training** with the ladder + error analysis.
5. **Estimate compute & memory** for any training or inference setup.
6. **Ship an agent** that uses tools to accomplish a real task.

This is the current **skill floor** for a deep-learning engineer or research student. You're standing on it.

---

# The frontier is orchestration, not new math

<div class="math-box">

Every frontier system decomposes into pieces you already own:

| Frontier topic | Built from | Open problem |
|:--|:--|:--|
| Agents | LLM (L15) + tool-calling + RAG | reliability at long horizons |
| Reasoning (o1/R1) | LLM + RL on chain-of-thought (L18) | controllable thinking time |
| Interpretability | attention (L14) + sparse autoencoders | scaling to frontier models |
| Multimodal agents | VLM (L20) + agents | cost / latency |

</div>

**Nothing is new — only the orchestration is.** You now have the vocabulary to read any 2026 paper and say exactly which lecture it rests on.

---

# Full circle · back to oranges vs tomatoes

In **ES 335**, your very first classifier was logistic regression on **oranges vs tomatoes** — predict $\hat y = \sigma(\theta^\top \mathbf{x})$, fit $\theta$ by minimizing cross-entropy:

$$J(\theta) = -\sum_i\big[\,y_i \log \hat y_i + (1 - y_i)\log(1 - \hat y_i)\,\big].$$

<div class="insight">

**The whole course is that one idea, scaled.** A frontier LLM predicts the next token $\hat y = p_\theta(\text{token}\mid\text{context})$ and is trained by minimizing the *exact same* cross-entropy $J(\theta)$ — now over a trillion tokens and a trillion parameters. **26 lectures did not replace logistic regression — they grew it up.**

</div>

| ES 335 · Lecture 1 | ES 667 · the frontier |
|--|--|
| $\hat y = \sigma(\theta^\top \mathbf{x})$ · 2 features | $\hat y = p_\theta(\text{token}\mid\text{context})$ · billions of params |
| Minimize $J(\theta)$ = cross-entropy | Minimize $J(\theta)$ = **same** cross-entropy |
| Gradient descent on $\theta$ | Adam + backprop on $\theta$, at scale |

Same $\hat y$, same $J(\theta)$, same $\theta$. **Only the scale changed.**

---

# The ideas that lasted

<div class="insight">

Once you see the framework, the field stops being 26 disconnected topics and becomes **one coherent story**:

- **Most losses reduce to an NLL or a KL / divergence bound** (L1).
- **Every architecture is a wiring of attention, convolution, and MLP** — differentiable modules composed.
- **Regularizers (L1/L2) are priors**; **every generative model is** "learn a distribution, then sample from it."

</div>

Likelihood · compositionality · invariance · optimization · **scale**. Five words, twenty-six lectures.

---

# Beyond Ng · the guiding philosophy

This course borrowed **A. Ng's pedagogy** — *intuition before rigor, code before theorems, ship something that runs* — and **A. Karpathy's** "build it from scratch to understand it." We then pushed past the syllabus into the live frontier.

<div class="keypoint">

Pair that with **the bitter lesson**: general methods that ride *scale* beat clever hand-engineering, again and again. The through-line of this course is exactly that — one differentiable module, one likelihood, and more compute. Knowing *why* it works is what lets you decide what to build next.

</div>

You're not finished — **you have the tools.**

---

# What to read next, and what to do next

<div class="columns">
<div>

**Read:**
- **Bishop & Bishop**, *Deep Learning: Foundations & Concepts* — for rigor
- **Karpathy's Zero to Hero** — keep going from scratch
- **Prince's UDL** — revisit what you skimmed
- **Blogs** — Lil'Log, Simon Willison, Chip Huyen, the Anthropic engineering blog
- set **arXiv alerts** for your area

</div>
<div>

**Do:**
- **Replicate a paper** end-to-end (NeurIPS / ICML / ICLR 2025) — teaches more than any course
- **Contribute to open source** — HF, vLLM, PyTorch; start at "good first issue"
- **Ship a small project** — fine-tune an LLM, train a toy diffusion model, build an agent
- **Read the safety & alignment work** if you care where this goes

</div>
</div>

<div class="notebook">

**Final project** · apply a technique from *any* lecture to a real problem · 3-week timeline · pitch week after endsem.

</div>

---

# The interactive lab · a course in your browser

Every big idea this semester has a **scroll-driven explainer** — revisit any of them while you build:

<div class="columns">
<div>

**Foundations → training**
`universal-approximation` · `optimizer-race` · `info-theory` · `mle-map-coin` · `numerical-tricks` · `softmax-temperature` · `dropout-playground`

</div>
<div>

**Modern → frontier**
`attention` · `positional-encoding` · `kv-cache` · `in-context-learning` · `rag` · `mixture-of-experts` · `lora-adapter` · `quantize-prune` · `vae-latent-explorer` · `diffusion-denoise`

</div>
</div>

<div class="notebook">

**🎛 Interactive Lab** — the full set lives under `interactive/src/articles/`; each is a self-contained explainer for a single lecture's core idea. *Drag the knobs — intuition sticks better than algebra.*

</div>

---

<!-- _class: summary-slide -->

# One sentence to remember

<div class="keypoint">

**Deep learning is one idea told twenty-six ways — a differentiable module, a likelihood, and scale.** The frontier just wraps that model in a loop (agents), a scratchpad (reasoning), and a microscope (interpretability).

</div>

- **Agents** — LLM + tools in a ReAct loop; the application layer of AI.
- **Reasoning models** — RL on chain-of-thought; test-time compute as a new scaling axis.
- **Mech interp** — the residual stream + sparse-autoencoder features; real progress, still early.
- **The through-line** — it was always cross-entropy (L1); *only the scale changed.*

*The field is younger than you are — go write the lecture that isn't here yet.*

---

<!-- _class: compact -->

# Sources &amp; credits

<div class="paper">

This lecture reuses and adapts material from the instructor's ES 667 frontier notes and standard references:

- **Agents / tool use** — Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models* (2022); Anthropic tool-use & Computer Use docs.
- **Reasoning / chain-of-thought** — Wei et al., *Chain-of-Thought Prompting* (2022); OpenAI, *Learning to Reason with LLMs (o1)* (2024); DeepSeek-R1 (2025).
- **Mechanistic interpretability** — Anthropic interpretability team: induction heads, sparse autoencoders / dictionary learning, and *Golden Gate Claude* (2024); transformer-circuits.pub.
- **Framing** — R. Sutton, *The Bitter Lesson* (2019); **A. Ng**, *Deep Learning Specialization* (the course's guiding pedagogy).
- **With thanks** to the source courses — *PSDV*, *Probabilistic ML*, and *ML (ES 335)*, N. Batra — and to **A. Karpathy**'s *Zero to Hero*, whose "build it from scratch" spirit runs through this course.

Figures reused from the ES 667 figure library (`figures/lec24/`). All source courses © N. Batra & teaching staff.

</div>
