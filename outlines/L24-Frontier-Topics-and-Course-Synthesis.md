---
title: "Lecture 24: Frontier Topics and Course Synthesis"
status: "Canonical 24-lecture revision — draft outline"
prerequisites: "Entire course"
next: "Course close"
---

# Lecture 24: Frontier Topics and Course Synthesis
## Agents, reasoning, interpretability—and the vocabulary to evaluate what comes next

## Core story

\[
\boxed{\text{Most frontier systems are familiar deep-learning components composed into a larger loop.}}
\]

The closing lecture should not introduce a fourth dense mathematical module. Its job is to give students a disciplined way to place new claims:

\[
\text{What is the model? What is the objective? What is the representation? What runs at inference?}
\]

**Target:** 80 minutes, approximately 60 slides.  
**One board example:** the residual stream as an additive shared workspace.  
**One final activity:** map a new system to earlier course concepts.  
**Out of scope:** product announcements, a tool tutorial, and claims that current systems are solved or transparent.

---

## Part I — Reassemble the course

1. **Title**  
   *Frontier Topics and the Deep Learning Toolkit*

2. **The 24-lecture course in one map**
   ~~~text
   probability / likelihood → differentiable models → optimization / generalization
      → vision and sequence representations → attention / pretraining
      → self-supervision / multimodality → generative modeling
      → inference systems → frontier applications
   ~~~

3. **Four questions for every new model**
   1. What representations and architecture does it use?
   2. What data and objective train it?
   3. What inference-time loop or tool system surrounds it?
   4. What failure mode or evaluation is being claimed?

4. **A warning about the word “frontier”**  
   New product names change quickly. The durable skill is locating a new system in the map above and identifying what is genuinely new versus renamed.

---

## Part II — Agents: an LLM inside a software loop

5. **Demystifying definition**
   \[
   \text{agent}=\text{model}+\text{state/context}+\text{tool interface}+\text{control loop}.
   \]

6. **Agent loop**
   ~~~text
   observe state → LLM proposes structured action → application executes tool
          ↑                                             ↓
          └──────────── append result to context ───────┘
   ~~~

7. **Critical correction**  
   The neural network does not execute code or call an API by itself. It emits a structured request; ordinary software validates, executes, and returns the result.

8. **Tool calling is a representation problem too**  
   Natural-language intent must be represented as a constrained action schema with arguments. Validity, permissions, and error handling live outside the model.

9. **Agent evaluation**
   - task success under realistic tool failures;
   - cost, latency, and number of actions;
   - safety boundaries and authorization;
   - robustness to ambiguous or adversarial observations.

10. **Failure modes**
    Wrong tool choice, loops, invented state, prompt injection through tool output, and overconfident irreversible actions. An agent is a systems problem, not only an accuracy metric.

---

## Part III — Reasoning and test-time compute

11. **Separate two scaling axes**
    \[
    \text{training compute / data}
    \qquad\text{versus}\qquad
    \text{inference-time compute per problem}.
    \]

12. **Chain-of-thought as an observable artifact**  
    Intermediate text can help a model decompose a task, but a displayed explanation is not guaranteed to be a faithful causal account of the model’s internal computation.

13. **Reasoning-system ingredients**
    - a base model capable of proposing candidate solution steps;
    - a reward, verifier, or search signal;
    - additional inference-time budget for sampling, checking, revising, or branching.

14. **Outcome versus process reward**

    | Reward | Checks | Benefit | Difficulty |
    |---|---|---|---|
    | Outcome reward | final answer | simple when answer is verifiable | sparse feedback |
    | Process reward | intermediate steps | can guide a solution path | hard to define and label reliably |

15. **Test-time compute trade-off**  
    More samples or longer search can improve hard-task reliability, but raises latency/cost and can still amplify a flawed verifier.

16. **Connection to earlier lectures**  
    This is optimization and inference again: an objective supplies a signal; the model proposes actions/tokens; a procedure allocates limited compute to search.

17. **Mini activity**  
    For a Sudoku solver, code generator, and medical assistant, ask: What can be automatically verified? Where is process feedback risky? What is the acceptable latency?

---

## Part IV — Interpretability: inspect representations with humility

18. **The residual stream as a shared workspace**
    \[
    h_{\ell+1}=h_\ell+\operatorname{Attn}(h_\ell)+\operatorname{MLP}(h_\ell).
    \]
    Each block reads a representation and writes an additive update; it does not simply replace it.

19. **Board example**
    Let
    \[
    h_5=[0.2,0.5,0.1,-0.9].
    \]
    If attention adds \([0,0.3,0.4,0]\) and the MLP adds \([0.1,-0.1,0,0.2]\), then
    \[
    h_6=[0.3,0.7,0.5,-0.7].
    \]
    Connect the arithmetic to “features can persist and later components can read them.”

20. **What attention maps can and cannot say**  
    Attention weights show one route by which information is mixed. They do not by themselves establish the semantic cause of a prediction.

21. **Superposition intuition**  
    A limited-dimensional representation may encode many more conceptual features than coordinates by combining them in overlapping directions.

22. **Sparse autoencoders: one conceptual picture**
    \[
    h\to \text{many mostly-zero feature activations}\to\hat h.
    \]
    An overcomplete sparse dictionary can make some learned features more individually inspectable.

23. **Interpretability’s honest status**  
    Interesting circuits and features can be discovered, especially in smaller or constrained settings; robustly explaining or controlling a frontier model remains an open research problem.

---

## Part V — Close with judgment, not hype

24. **Open problems slide**
    - reliable generalization under distribution shift;
    - data provenance, privacy, and bias;
    - factuality, calibration, and tool-use safety;
    - scalable evaluation of generative and agentic systems;
    - efficient, interpretable, and controllable representations.

25. **A claim-evaluation checklist**
    When reading a paper or announcement:
    1. What data and benchmark support the claim?
    2. What baseline is missing?
    3. Is the gain from architecture, scale, data, objective, or inference budget?
    4. Which failure case is not measured?

26. **Final synthesis table**

    | Course idea | Frontier manifestation |
    |---|---|
    | learned representations | foundation models, multimodal embedding spaces |
    | objectives and optimization | preference/reward training, verifiers |
    | attention and autoregression | LLMs, VLMs, tool-action policies |
    | generative modeling | diffusion, world-model-like simulation |
    | inference systems | caching, search, test-time compute |

27. **Capstone discussion prompt**  
    Choose a system students have encountered. In groups, use the four questions from Slide 3 to map it onto the course. Require one capability claim and one likely failure mode.

28. **Closing line**
    \[
    \boxed{\text{Deep learning is a small set of ideas, recombined at scale.}}
    \]
    Students now have the vocabulary to distinguish a new architecture, a new objective, a new dataset, and a new systems wrapper—and to ask whether the evidence matches the claim.

---

---

## Part VI — Detailed synthesis and frontier-literacy slides

## Slide 29 — One durable course equation

Across the course:

\[
\text{data}
\xrightarrow{\text{representation + architecture}}
\text{prediction or sample}
\xrightarrow{\text{loss}}
\text{gradient update}.
\]

Different topics change the representation, factorization, objective, or inference procedure—not the basic differentiable-learning loop.

---

## Slide 30 — Architecture versus system

| Item | Example | What is new? |
|---|---|---|
| architecture | Transformer, U-Net | parameterized computation |
| objective | NLL, contrastive, reward | what training prefers |
| representation | tokens, patches, latent | what information is carried |
| system wrapper | tool loop, search, cache | how the model is used |

Students should stop calling every new wrapper “a new neural network.”

---

## Slide 31 — Structured actions

An agentic model should produce an action schema such as:

\[
\{\text{tool name},\text{arguments},\text{identifier}\}.
\]

The application validates types, checks permissions, executes the request, and returns a result. Natural language is not a safe API contract by itself.

---

## Slide 32 — State is external

Distinguish:

- model parameters: learned before deployment;
- context window: temporary text/token state;
- tool/database state: external mutable world state;
- application memory: what the system decides to retain.

This distinction is essential for debugging agent behaviour.

---

## Slide 33 — Planning is not a magic module

A system can ask a model to propose several actions, critique them, call a verifier, or execute a loop. These are inference-time orchestration choices.

\[
\text{more steps}\ne\text{guaranteed better plan}.
\]

Quality depends on model capability, state quality, tool reliability, and a suitable stopping rule.

---

## Slide 34 — Tool-use evaluation

Measure more than final answer accuracy:

| Dimension | Example question |
|---|---|
| correctness | was the requested task completed? |
| efficiency | how many tool calls and tokens? |
| robustness | what happens when a tool returns an error? |
| safety | were permissions and sensitive actions respected? |
| calibration | did the system know when to stop or ask? |

---

## Slide 35 — Prompt injection is a systems boundary problem

Untrusted web pages, documents, and tool results can contain instructions that conflict with the user’s goal.

The defense is not “tell the model to ignore it” alone; it includes data provenance, action allowlists, sandboxing, confirmation for high-impact actions, and output validation.

---

## Slide 36 — Reasoning traces: useful but not ground truth

Intermediate text can improve problem decomposition and enable checking. It may also be post-hoc, incomplete, or strategically shaped.

\[
\text{plausible explanation}\ne\text{faithful causal explanation}.
\]

Teach this distinction before students encounter confident demonstrations of “reasoning.”

---

## Slide 37 — Verifiers change the problem

Some tasks admit cheap automatic checks:

- unit tests for code;
- exact arithmetic;
- constraint satisfaction;
- formal proofs with a checker.

When verification is reliable, inference can generate multiple candidate solutions and select or refine them. Open-ended judgment is much harder to verify.

---

## Slide 38 — Search at inference time

~~~text
model proposes candidates
        ↓
verifier / reward / critic scores candidates
        ↓
keep, revise, branch, or stop
~~~

This is a system-level search loop. It is related to optimization but occurs after model training.

---

## Slide 39 — Outcome and process rewards, expanded

\[
r_{\mathrm{outcome}}(y)
\qquad\text{versus}\qquad
r_{\mathrm{process}}(s_1,\ldots,s_k).
\]

Outcome reward is often cheap but sparse. Process reward can identify an early wrong step, but requires trustworthy judgments about intermediate reasoning.

---

## Slide 40 — Test-time-compute allocation

For a fixed model, decide how to spend a compute budget:

- one long candidate;
- many independent candidates;
- a search tree;
- candidate plus verifier;
- candidate plus tool calls.

There is no universal strategy; the task’s verification structure matters.

---

## Slide 41 — From neurons to features

A single neuron can participate in many unrelated behaviours, and one behaviour can be distributed across many neurons.

\[
\text{neuron}\ne\text{human concept}.
\]

This motivates studying features and circuits in the residual stream rather than reading one activation in isolation.

---

## Slide 42 — Sparse autoencoder objective, high level

An SAE learns:

\[
h\to a\to\hat h
\]

with reconstruction pressure and a sparsity penalty:

\[
L=\|h-\hat h\|_2^2+\lambda\|a\|_1.
\]

The hope is that a wide, mostly inactive feature dictionary separates concepts packed together in \(h\).

---

## Slide 43 — Interpretation needs intervention

Finding a correlated feature is not enough. Stronger evidence asks:

1. activate, remove, or patch a representation component;
2. observe a predicted behavioural change;
3. rule out nearby confounds.

This parallels causal reasoning: correlation is a starting point, not a complete explanation.

---

## Slide 44 — Interpretability limits

| Tempting claim | More honest claim |
|---|---|
| “We read the model’s thoughts.” | We found a feature/circuit correlated with a behaviour. |
| “Attention explains the answer.” | Attention is one information-routing signal. |
| “One visualization proves safety.” | Coverage and causal validation remain limited. |

---

## Slide 45 — Multimodal and generative agents

An agent can combine:

\[
\text{text model}+\text{vision encoder}+\text{tool loop}+\text{state}.
\]

This is L18’s representation alignment plus L23’s inference/system constraints. The composition creates capabilities and new failure channels.

---

## Slide 46 — Responsible deployment questions

Before deploying a frontier system, ask:

- who can be harmed by a false positive or false action?
- what data or tools are sensitive?
- what audit trail exists?
- when must a human approve an action?
- how will failures be reported and corrected?

This is technical design, governance, and product judgment together.

---

## Slide 47 — Read a claim like a researcher

For any new result, annotate:

\[
\text{claim}\quad|\quad\text{data}\quad|\quad\text{metric}\quad|\quad\text{baseline}\quad|\quad\text{cost}\quad|\quad\text{failure case}.
\]

Use one short paper abstract or announcement as a live class exercise.

---

## Slide 48 — Course map, representation thread

~~~text
CNN features → transferred features → token representations
             → SSL features → shared image–text embeddings
             → VAE latents → diffusion latents → system state
~~~

Not all are identical representations, but each determines what the next computation can efficiently learn or control.

---

## Slide 49 — Course map, objective thread

| Module | Central objective |
|---|---|
| classifiers / LLMs | negative log-likelihood |
| regularization | NLL plus constraints/priors |
| SSL | constructed prediction / agreement |
| VAE | likelihood lower bound |
| GAN | minimax adversarial objective |
| diffusion | noise-prediction regression |
| alignment / reasoning | preference or verifier-driven signal |

---

## Slide 50 — Course map, systems thread

\[
\text{training}
\to
\text{pretraining}
\to
\text{adaptation}
\to
\text{inference}
\to
\text{tools/search/deployment}.
\]

This counters the misconception that a model ends when training loss stops decreasing.

---

## Slide 51 — Final group activity

Give each group one system description—an image generator, a coding assistant, a medical triage tool, or a retrieval chatbot.

They must produce:

1. architecture/representation;
2. likely training objective;
3. inference/system loop;
4. one evaluation metric;
5. one serious failure mode and mitigation.

---

## Slide 52 — Retrieval questions

1. Why is an agent not a distinct neural-network architecture?
2. When can test-time search be more useful than a larger model?
3. Why is a visible reasoning trace not guaranteed to be faithful?
4. What kind of evidence makes an interpretability claim stronger?
5. Which course objective most resembles diffusion noise prediction?

---

## Slide 53 — What has changed since the first lecture?

At the start, the course asked how a differentiable model turns inputs into labels. By the end, the same machinery has:

- learned reusable representations;
- generated samples;
- aligned to preferences or conditions;
- operated inside tools and search loops.

The scale changed; the core vocabulary remained useful.

---

## Slide 54 — What has not changed?

\[
\text{data quality},\quad
\text{objective design},\quad
\text{evaluation},\quad
\text{deployment constraints}
\]

remain central. A larger model does not remove the need to define the task, detect distribution shift, or account for errors.

---

## Slide 55 — A final model card exercise

Students draft a one-page model card for their group system:

1. intended use;
2. data and objective assumptions;
3. evaluation evidence;
4. known limitations;
5. human oversight and escalation path.

This turns abstract responsible-AI discussion into an engineering artifact.

---

## Slide 56 — Course project opportunities

Offer four bounded directions:

- evaluate transfer versus scratch training on a small dataset;
- compare representation objectives with a fixed probe;
- build a tiny generative model and analyze failure modes;
- profile a model and justify an inference optimization.

The goal is evidence and reflection, not training a frontier-scale system.

---

## Slide 57 — How to read a new paper next year

Read in this order:

1. abstract and claim;
2. model/representation diagram;
3. objective and data;
4. evaluation table and baselines;
5. limitations and compute cost;
6. only then detailed method sections.

This prevents students from getting lost in implementation before they know what question is being answered.

---

## Slide 58 — Research question generator

Complete one sentence:

> “If the representation were invariant to ___ but preserved ___, then a useful pretext task might be ___, evaluated by ___.”

This returns the frontier lecture to the SSL and representation-learning design skills of L17.

---

## Slide 59 — Exam-style synthesis question

“A team proposes a multimodal agent for scientific literature review. Identify its likely representation modules, training objectives, inference loop, two evaluation metrics, and two failure modes.”

Show that the answer uses ideas from the whole course, not memorized product facts.

---

## Slide 60 — The limits of course coverage

The course gives students a map, not mastery of every frontier system. A rigorous answer may be:

\[
\text{we do not yet have enough evidence to claim this works reliably}.
\]

Reward this judgment explicitly in discussion and assessment.

---

## Slide 61 — Final retrieval questions

1. Name one difference between a new architecture and a new system wrapper.
2. What makes an agent action safe enough to execute?
3. How can a verifier make inference-time search more useful?
4. What evidence distinguishes interpretability correlation from causal evidence?
5. Which earlier lecture best helps explain VLM-agent systems, and why?

---

## Slide 62 — Closing slide

\[
\boxed{
\text{A model is not a product, a benchmark is not deployment, and a new name is not necessarily a new idea.}
}
\]

The durable outcome of the course is the ability to decompose a claim into data, representations, architecture, objective, inference, and evidence.

---

## Instructor pacing notes

- **Must teach deeply:** agents as a software loop; the separation of training versus test-time compute; residual-stream additive updates; the final claim-evaluation checklist.
- **Keep light:** implementation details of tool frameworks, named reasoning products, sparse-autoencoder optimization.
- **Preparation-light demo:** use a static toy tool loop and a fixed residual-stream diagram. This lecture should be discussion-led, not a fragile live demo.

## Student takeaways

1. Agents are deployment systems that wrap a model in state, tools, and control logic.
2. Test-time compute is distinct from model size or context length and creates new cost/reliability trade-offs.
3. Interpretability is about representations and causal mechanisms, not simply visualizing attention.
4. The course provides a framework for evaluating new deep-learning claims after the syllabus ends.
