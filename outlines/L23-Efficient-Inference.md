---
title: "Lecture 23: Efficient Inference"
status: "Canonical 24-lecture revision — draft outline"
prerequisites: "Transformer attention; diffusion sampling loop; basic GPU-memory intuition"
next: "L24 — Frontier Topics and Course Synthesis"
---

# Lecture 23: Efficient Inference
## The model is often waiting on memory, not arithmetic

## Core story

\[
\boxed{\text{At inference, moving weights and cached activations is often the bottleneck.}}
\]

Generation is not one uniform computation: processing an input prompt and producing one token at a time have different shapes, costs, and optimizations.

**Target:** 80 minutes, approximately 65 slides.  
**One worked calculation:** KV-cache memory.  
**One essential distinction:** prefill versus decode.  
**Out of scope:** framework-specific APIs, kernel implementation details, and production serving operations.

---

## Part I — Diagnose before optimizing

1. **Title**  
   *Efficient Inference: Compute Less, Move Fewer Bytes*

2. **Why this belongs in a DL course**  
   An accurate model that cannot meet latency, memory, or cost constraints is not usable. Architecture, quality, and deployment are coupled.

3. **The two phases of autoregressive LLM inference**
   ~~~text
   prompt tokens ── prefill ──→ first output token ── decode repeatedly ──→ final response
   ~~~

4. **Prefill**  
   Process many prompt tokens in parallel. Large matrix operations use the accelerator efficiently; attention cost grows with sequence length.

5. **Decode**  
   Produce one (or a small batch of) next token(s) at a time. Each step needs model weights plus all prior key/value state; this is often bandwidth-bound.

6. **Compute-bound versus memory-bound**  
   - compute-bound: arithmetic units are the limiting resource;
   - memory-bound: the device spends much of its time fetching weights/activations.

7. **The bytes intuition**  
   For a single generated token, a huge matrix-vector product performs relatively few operations per byte of model weight fetched. The accelerator can wait on memory even when it has abundant FLOPs.

8. **Inference budget worksheet**  
   Every deployment should ask:
   \[
   \text{latency},\quad\text{throughput},\quad\text{VRAM},\quad\text{quality},\quad\text{cost}.
   \]
   An optimization improves one or more, with trade-offs.

---

## Part II — KV cache: reuse what attention already computed

9. **Recall causal attention**  
   To attend from new token \(t\), a layer needs keys and values for tokens \(1,\ldots,t\).

10. **The naive mistake**  
    Re-run the entire Transformer on the full prefix after every new token. This recomputes keys and values for old tokens despite their being unchanged.

11. **KV cache idea**
    ~~~text
    earlier tokens → cached K, V ──┐
    new token → new Q, K, V ───────┼→ attention for new token
                                   └→ append new K, V to cache
    ~~~

12. **What caching saves**  
    It avoids recomputing old projections. It does **not** eliminate attention over the growing history; the new query still reads the cached keys/values.

13. **KV-cache shape per layer**
    \[
    K,V\in\mathbb R^{B\times T\times n_{\text{kv-heads}}\times d_{\text{head}}}.
    \]
    Two tensors are stored at every layer.

14. **KV-cache memory approximation**
    \[
    M_{\text{KV}}
    \approx
    2\cdot L\cdot B\cdot T\cdot n_{\text{kv-heads}}\cdot d_{\text{head}}\cdot b,
    \]
    where \(b\) is bytes per stored value.

15. **Board calculation**  
    For \(L=32\), \(B=1\), \(T=4096\), \(n_{\text{kv-heads}}=8\), \(d_{\text{head}}=128\), and fp16 (\(b=2\)), compute:
    \[
    2(32)(1)(4096)(8)(128)(2)\approx512\text{ MiB}.
    \]
    Then ask what happens for 32 concurrent sequences or much longer context.

16. **The key trade-off**  
    KV cache makes decoding faster by exchanging compute for memory. Long contexts and high concurrency can make memory—not parameters—the dominant capacity limit.

17. **Architectural response: fewer KV heads**  
    Multi-query and grouped-query attention share keys/values across query heads, reducing cache size while retaining much of multi-head attention’s flexibility.

---

## Part III — The main inference levers

18. **Quantization: store fewer bits**
    \[
    \text{fp16 weights}\quad\longrightarrow\quad\text{8-bit or 4-bit representation}.
    \]
    Fewer bytes can reduce memory and accelerate bandwidth-bound decoding; approximation can reduce quality or complicate kernels.

19. **Quantization is not one thing**  
    Distinguish:
    - weight-only quantization;
    - activation / KV-cache quantization;
    - post-training quantization versus quantization-aware training.

20. **Attention kernels: avoid materializing giant intermediates**  
    Standard attention can create large score matrices. Tiled, streaming kernels compute exact attention while reducing high-bandwidth memory traffic. Name FlashAttention as the canonical example; do not teach CUDA.

21. **Batching and scheduling**  
    Serving multiple requests together raises hardware utilization, but requests have different prompt lengths and generation durations. State the systems problem without building a serving stack.

22. **Paged or block-based KV storage**  
    Treat KV cache as reusable memory blocks rather than one contiguous reservation per request. This reduces fragmentation and enables flexible batching; name the idea, not framework internals.

23. **Speculative decoding**
    ~~~text
    small draft model proposes several tokens
                    ↓
    large target model verifies them in one pass
                    ↓
    accept a prefix; correct if needed
    ~~~
    The target model preserves output quality; speedup depends on how often draft tokens are accepted.

24. **Why speculative decoding can help**  
    It trades inexpensive draft computation for fewer sequential expensive target-model decode steps. It does not make the big model unnecessary.

25. **Distillation**  
    Train a smaller student to imitate a larger teacher’s outputs or probability distribution. This changes the model itself rather than only the serving procedure.

26. **A decision table**

    | Lever | Mainly attacks | Main cost |
    |---|---|---|
    | KV cache | repeated prefix computation | VRAM |
    | Quantization | weight / cache bandwidth | approximation and engineering |
    | Efficient attention kernel | intermediate-memory traffic | specialized implementation |
    | Batching / paging | underutilization and fragmentation | scheduling complexity |
    | Speculative decoding | sequential target-model steps | draft model and variable gain |
    | Distillation | model size / latency | retraining and possible quality loss |

---

## Part IV — Apply the systems lens across the course

27. **Diffusion connection**  
    Diffusion sampling repeatedly invokes the denoiser. Faster samplers, latent spaces, batching, and quantization all reduce the cost of those repeated calls.

28. **Representation-learning connection**  
    Frozen feature encoders and lightweight linear probes can make a downstream system cheaper; an accurate representation is also an engineering asset.

29. **Case exercise**  
    Choose one:
    - chat system with long prompts and many concurrent users;
    - mobile image classifier;
    - text-to-image demo with a latency limit.

    Students choose two optimizations and name the expected trade-off.

30. **The one-sentence synthesis**
    \[
    \text{Efficient inference is mostly about avoiding repeated work and moving fewer bytes.}
    \]

31. **Closing bridge**  
    The final lecture places agents, reasoning systems, and interpretability on top of this same foundation: models, objectives, optimization, representations, and inference constraints.

---

---

## Part V — Detailed systems slides

## Slide 32 — Parameter memory comes first

For \(P\) parameters:

| Storage format | Approximate weight memory |
|---|---:|
| fp32 | \(4P\) bytes |
| fp16 / bf16 | \(2P\) bytes |
| int8 | \(P\) bytes |
| 4-bit | \(P/2\) bytes, plus metadata |

This is a first-order calculation; runtime memory also includes KV cache, activations, and framework overhead.

---

## Slide 33 — Weight-memory example

A \(7\) billion parameter model in fp16 needs approximately:

\[
7\times10^9\times2\text{ bytes}\approx14\text{ GB}
\]

only for weights. Ask students why a device advertised as “16 GB” may still be inadequate for useful long-context generation.

---

## Slide 34 — Arithmetic intensity intuition

Matrix-matrix multiplication during prefill reuses weights across many token vectors. Matrix-vector-like decode work reuses each fetched weight less.

\[
\text{less reuse per byte}\Rightarrow\text{memory bandwidth matters more}.
\]

No hardware-specific roofline calculation is required.

---

## Slide 35 — Causal attention cost, briefly

For sequence length \(T\), a full attention layer compares token pairs:

\[
O(T^2d).
\]

KV caching avoids recomputing old projections during decode but does not remove the new query’s need to attend across the stored history.

---

## Slide 36 — Cache append at one layer

At token \(t\):

\[
K_{1:t}=[K_{1:t-1};k_t],
\qquad
V_{1:t}=[V_{1:t-1};v_t].
\]

Only \(q_t,k_t,v_t\) for the new token are computed; attention for \(q_t\) reads all cached keys and values.

---

## Slide 37 — Recompute versus cache

| Strategy | Compute | Memory | Decode latency |
|---|---|---|---|
| recompute full prefix | very high | low cache | poor |
| store KV cache | lower | grows with \(T\) | much better |

This table is the central systems trade-off of the lecture.

---

## Slide 38 — KV-cache calculation, step by step

Start with:

\[
2\cdot L\cdot B\cdot T\cdot n_{\mathrm{kv}}\cdot d_h\cdot b.
\]

Use \(L=32,B=1,T=4096,n_{\mathrm{kv}}=8,d_h=128,b=2\), then multiply in stages:

\[
2\cdot32\cdot4096\cdot8\cdot128\cdot2.
\]

The result is approximately \(512\) MiB, before concurrency.

---

## Slide 39 — Concurrency changes the answer

If 32 requests each need this cache:

\[
32\times512\text{ MiB}\approx16\text{ GiB}.
\]

Long contexts and many users make cache capacity a product-level constraint, not an implementation footnote.

---

## Slide 40 — Grouped-query attention

Standard multi-head attention may have one key/value head per query head. Grouped-query attention uses fewer key/value heads:

\[
n_{\mathrm{kv-heads}}<n_{\mathrm{query-heads}}.
\]

This reduces cache footprint while retaining multiple query perspectives.

---

## Slide 41 — Quantization is approximate representation

A real-valued block of weights can be stored approximately as:

\[
W\approx s\,(W_{\mathrm{int}}-z),
\]

with scale \(s\), optional zero-point \(z\), and a low-bit integer tensor. The model’s mathematical function is approximated to move fewer bytes.

---

## Slide 42 — Quantization design choices

| Choice | Question |
|---|---|
| per-tensor or per-channel scale | how much variation must be preserved? |
| weight-only or activations too | where is memory/bandwidth pressure? |
| 8-bit or 4-bit | what quality loss is acceptable? |
| post-training or trained-aware | can we adapt the model? |

---

## Slide 43 — Quantization is not free speed

Fewer bits can reduce bandwidth and fit a larger model in memory, but gains depend on hardware/kernel support. Dequantization, metadata, and irregular operations can limit the apparent theoretical savings.

This is a useful systems-literacy correction.

---

## Slide 44 — FlashAttention as a memory algorithm

The attention equation is unchanged:

\[
\operatorname{softmax}(QK^\top/\sqrt d)V.
\]

Efficient kernels tile the computation so large score matrices need not be written to slow high-bandwidth memory. Teach “same answer, different data movement.”

---

## Slide 45 — Prefix caching

Many requests share a system prompt or document prefix. Cache its computed KV state once and reuse it for later requests.

This is the serving analogue of avoiding duplicate computation in dynamic programming.

---

## Slide 46 — Continuous batching

Requests arrive and finish at different times. A scheduler can insert new requests into free capacity rather than waiting for an entire fixed batch to finish.

The optimization target becomes throughput and tail latency, not merely one request’s speed.

---

## Slide 47 — Blocked KV allocation

Instead of reserving one contiguous cache region for every request, allocate blocks as sequences grow. This reduces fragmentation and makes it easier to share/reclaim memory.

Keep framework names off the main slide; the underlying resource-management idea is enough.

---

## Slide 48 — Speculative decoding, expanded

1. a small draft model proposes several tokens;
2. the large target model checks them together;
3. accept a matching prefix;
4. sample a correction when the draft diverges.

The target model remains the authority, so accepted output follows its distribution under the appropriate algorithm.

---

## Slide 49 — Acceptance rate determines speedup

If the draft model guesses poorly, verification rejects early and little sequential work is saved. If it predicts a long correct prefix, one expensive target pass replaces many decode steps.

\[
\text{draft quality}+\text{cheapness}\Rightarrow\text{useful speculation}.
\]

---

## Slide 50 — Distillation changes the model

Train a student using a teacher distribution:

\[
L_{\mathrm{distill}}
=
T^2D_{\mathrm{KL}}
\bigl(
\operatorname{softmax}(z_{\mathrm{teacher}}/T)
\Vert
\operatorname{softmax}(z_{\mathrm{student}}/T)
\bigr).
\]

Give the intuition: soft targets convey relative alternatives, not only one hard label.

---

## Slide 51 — Compression versus serving optimization

| Approach | Requires retraining? | Main effect |
|---|---|---|
| quantization | often no | fewer bits / lower bandwidth |
| KV cache | no | avoid repeated prefix work |
| speculative decoding | no target retraining | fewer sequential target steps |
| distillation | yes | smaller/faster model |

---

## Slide 52 — Case exercise: choose levers

Scenario A: long-document chatbot with many concurrent users.  
Scenario B: small offline classifier on a phone.  
Scenario C: image generator with a 2-second latency budget.

For each, select two levers and state the quality, cost, or engineering trade-off.

---

## Slide 53 — Common misconceptions

| Misconception | Correction |
|---|---|
| “KV cache makes attention constant-time.” | It avoids old projections but cache reads still grow with context. |
| “Quantization always improves every metric.” | It trades precision and relies on efficient kernels. |
| “FlashAttention changes Transformer math.” | It computes the same attention with different memory movement. |
| “A draft model replaces the target model.” | The target still verifies output. |

---

## Slide 54 — Retrieval questions

1. Why do prefill and decode stress hardware differently?
2. What exactly is stored in a KV cache?
3. Which variables make cache memory grow?
4. Why can 4-bit weights help a bandwidth-bound workload?
5. Which optimization changes the underlying model rather than only serving it?

---

## Slide 55 — Memory-budget worksheet

Before loading a model, estimate:

\[
\text{weights}+\text{KV cache}+\text{temporary activations}+\text{runtime overhead}.
\]

For a serving system, multiply the cache term by expected concurrency rather than by one idealized request.

---

## Slide 56 — Context-window policy is a systems policy

A larger maximum context can improve user experience but raises cache memory and attention work. Truncation, retrieval, summarization, and sliding windows are product choices shaped by the same compute constraints.

This connects inference engineering to retrieval and agent design.

---

## Slide 57 — Model parallelism, one placement slide

When one device cannot hold a model, parameters or layers can be distributed across devices. This makes larger models feasible but introduces communication overhead.

\[
\text{more devices}\ne\text{linear speedup}.
\]

Keep pipeline/tensor-parallel algorithms outside the core lecture.

---

## Slide 58 — Compilation and fused operations

Launching many tiny operations separately can waste time. Compilers and fused kernels combine operations to reduce launch overhead and memory reads.

This is another instance of the lecture’s core idea: avoid unnecessary movement and coordination.

---

## Slide 59 — Quantization evaluation protocol

Test a compressed model on:

- representative task quality;
- long-context behavior;
- rare/error-sensitive cases;
- latency and memory on the actual target hardware.

Average benchmark accuracy alone may miss a damaging quantization failure.

---

## Slide 60 — Cost per useful outcome

The relevant quantity is not merely tokens per second:

\[
\frac{\text{cost}}{\text{useful, correct, safe outcome}}.
\]

A slower model can be preferable if it reduces retries or enables reliable verification; a fast model can be wasteful if outputs require repeated correction.

---

## Slide 61 — Energy and hardware awareness

Moving data across memory hierarchies consumes time and energy. Efficient inference can therefore reduce operational cost and environmental impact, though demand growth can offset per-query savings.

Avoid simplistic “efficient therefore green” claims.

---

## Slide 62 — Serving versus research benchmark

| Benchmark focus | Production focus |
|---|---|
| one model, one batch, peak throughput | variable requests, tail latency, reliability |
| fixed prompt length | long and changing contexts |
| clean hardware state | memory fragmentation and concurrency |

Students should question performance claims that omit workload assumptions.

---

## Slide 63 — Retrieval questions, extended

1. Why can a model fit in memory yet fail under concurrent long-context use?
2. What is the difference between quantizing weights and quantizing KV cache?
3. Which optimization changes arithmetic scheduling but not model output?
4. Why does serving require workload assumptions?

---

## Slide 64 — Final bridge

\[
\text{models}+\text{objectives}+\text{representations}+\text{inference systems}
\]

are all needed to understand a frontier deployment. The final lecture uses this checklist to demystify agents, reasoning systems, and interpretability claims.

---

## Instructor pacing notes

- **Must teach deeply:** prefill versus decode; KV-cache reuse; one cache-memory calculation; quantization and speculative decoding at the conceptual level.
- **Keep light:** exact hardware roofline models, attention-kernel algorithms, vLLM/framework APIs.
- **Preparation-light demo:** a static profiler/timing chart or a short cached-versus-uncached toy Transformer trace. Avoid benchmarking hardware live.

## Student takeaways

1. Prompt processing and token-by-token decoding have different performance profiles.
2. KV caching trades memory for avoided repeated computation.
3. Many inference optimizations reduce memory traffic rather than changing neural-network mathematics.
4. Deployment constraints are first-class model-design constraints.
