# Lecture 15 · Large Language Models · Teaching Guide
*First-time-instructor companion. Audience: undergrads, one ML course.*

## The spine (say this in 2 sentences)
LLMs are the same decoder-only Transformer from 2018 — what changed is *scale*, governed by predictable scaling laws (loss falls as a power law in compute). The lecture is five engineering answers to "how do you make scale actually work": compute-optimal sizing (Chinchilla), relative positions (RoPE), cheap inference memory (GQA), splitting the model across GPUs (DP/TP/PP), and the surprising abilities that fall out (emergence).

## Where it sits
Follows L13 (Transformer block) and L14 (tokenization + pretraining: BPE, BERT/GPT/T5). This is the "now scale it" lecture. It sets up L16 (you have raw capability — now align it into a helpful assistant) and feeds the efficient-attention thread picked up again in L23 (FlashAttention). This is a **landscape/literacy** lecture: the goal is fluent intuition, not mastery of every derivation.

## 80-minute plan
- **Scaling laws + Chinchilla** ⭐ ~22 min — the heart of the lecture. **Do live on the board:** the Chinchilla compute-sizing example (below).
- **RoPE** ⭐⭐ ~16 min — teach the spinning-pointer intuition + the three properties. The 2D derivation slides are ⭐⭐⭐ skip-if-tight.
- **KV-cache + GQA** ⭐⭐ ~16 min — build the KV-cache size formula layer by layer; do the 84 GB number; then GQA shrinks it 8×.
- **Distributed training** ⭐ ~8 min — just the LEGO-team analogy + "frontier combines all three." Don't go deep.
- **Emergent abilities** ⭐⭐ ~14 min — the per-step-accuracy product argument, then the controversy (metric artifact). End on in-context learning + CoT → reasoning models.
- **Buffer / summary** ~4 min.

## Teach it like this (hook → sequence → payoff)
**Hook:** the "$1M of compute — spend it" pop quiz (slide 62). Let them vote between three (params, tokens) splits. Most will guess "biggest model." This is wrong and that's the lesson. **Sequence:** scaling laws say loss is predictable in compute → Chinchilla says split the budget between params and tokens (≈20 tokens/param) → but you can't fit a 70B model on a GPU, so we need RoPE (positions), GQA (memory), and 3D parallelism (compute) → and at the top of the scale, abilities you never trained for appear. **Payoff:** "Scale changed, the architecture didn't" — a 2018 student would recognize Llama-3's shape; everything new is engineering to make scale work.

## Heads-up for YOU (subtle points to get right)
1. **Scaling laws are empirical extrapolations, not guarantees.** They are power-law fits over the regime tested. They predict average loss; they do *not* promise any specific capability, and people have been burned extrapolating them too far. Say "remarkably reliable so far," not "a law of nature."
2. **"20 tokens per parameter" is the *headline ratio*, not the law.** The actual Chinchilla result fits L(N,D) = E + A/N^α + B/D^β and minimizes it under the budget C ≈ 6ND. "≈20" is what falls out at the optimum — present it as a rule of thumb, and note the slide's own "fine print" line.
3. **RoPE has *zero* learned parameters.** The rotation angles are fixed and deterministic given position. The model learns to *use* them through its ordinary Q/K projection weights — it does not learn the rotations. Don't say "RoPE learns positions."
4. **GQA saves *KV-cache* (activation) memory, not weight memory.** It shrinks the per-token activations stored during autoregressive decoding — the true long-context bottleneck. It barely changes the model's weight count. And note: query heads still have *independent* projections, so sharing KV doesn't mean the heads ask the same question.
5. **Overtraining (Llama-3 8B at 15T tokens) is not a mistake.** Chinchilla optimizes *training* compute. Llama-3 optimizes *inference* compute: a smaller, heavily over-trained model is far cheaper to serve to millions of users. Spend more upfront, save forever. The slides' D/N=1875 looks "wildly over" only against the training-optimal yardstick.
6. **Emergence is (largely) a measurement artifact.** Underlying per-token log-prob improves *smoothly* with scale. The "sudden jump" appears when you score a multi-step task with a hard pass/fail metric (exact match). Present both sides (Wei 2022 vs Schaeffer 2023) — the user-experienced leaps are real, but the mechanism is smooth.

## Where students stumble (and the fix)
1. **"Bigger model is always better."** Fix: the pop quiz. Show that at a fixed budget, a too-big model is *undertrained* (GPT-3 saw ~10× too few tokens). The board example nails this.
2. **Confusing absolute vs relative position.** Fix: the clock analogy. Sinusoidal/learned PE *add* a vector (absolute); RoPE *rotates* Q and K, so the attention dot product only sees the angle *difference* = relative offset. Draw two clock hands.
3. **Thinking the KV-cache is part of the model weights.** Fix: build the formula live (one token → one head → all heads → all layers → all tokens). Emphasize it grows with *sequence length* — that's why long context is expensive, separate from the 140 GB of weights.
4. **Pipeline vs tensor parallelism blur together.** Fix: TP splits *one matrix multiply* across GPUs (all work the same layer simultaneously, needs huge bandwidth/NVLink); PP puts *different layers* on different GPUs (assembly line, suffers idle "bubbles"). One sentence each is enough at this level.

## If a student asks…
- **"Why does RoPE extrapolate to longer contexts than it trained on?"** The rotation frequencies are fixed, and attention only sees relative offset. A distance of 2 looks the same at positions 10,000–10,002 as at 2–4, so the model can handle offsets it has effectively seen before. (Caveat for the curious: pure extrapolation still degrades; production models use small fixes like NTK/YaRN scaling.)
- **"If GQA heads share keys and values, aren't they redundant?"** No — they share the *cache* but each query head has its own projection, so it "asks a different question" of the same shared notebook. ~8× memory saving for ~1% quality drop.
- **"Is the architecture converging because it's mature, or because we've stopped exploring?"** (HARD / open) Honest answer: both. The decoder-only Transformer + RoPE + GQA is a strong local optimum that's cheap to scale, so labs converge on it. Whether something better (Mamba, MoE-heavy designs) displaces it is open — that's why the interactive links include Mamba.
- **"Could we just train one giant model on a little data?"** That's the GPT-3 regime — undertrained and wasteful. The board example shows you'd get worse loss for the same compute than splitting the budget.
- **"Is in-context learning real learning if no weights change?"** It's learning *at inference time* from the prompt, with no weight update. The model infers the task from the examples. It emerges only at scale (~50B+) and is the foundation of modern prompting.
- **"How is a reasoning model (o1, R1) different from chain-of-thought prompting?"** CoT prompting is a *prompt trick* on a fixed model. Reasoning models are *trained* (RL with process rewards) to produce long internal chains of thought before answering, and they spend 10–100× more inference compute. Same base architecture; new training regime + a new scaling axis (test-time compute).

## If you're short on time
- **Cut:** the ⭐⭐⭐ RoPE 2D derivation and verification slides (keep only the spinning-pointer intuition + the three properties); the compute-budget √C derivation; the distributed-training detail (analogy only).
- **Never cut:** the Chinchilla board example, the spinning-pointer RoPE intuition, the KV-cache 84 GB punchline, and the emergence-as-product-of-step-accuracies argument. Those four are the lecture.

## The one live board example (3 min)
**Chinchilla compute sizing.** "You have C = 1.2 × 10²⁴ FLOPs. What model do you train?"
1. Write the two rules: `C = 6ND` and `D = 20N`.
2. Substitute: `C = 6N(20N) = 120N²`.
3. Solve N: `120N² = 1.2×10²⁴ → N² = 10²² → N = 10¹¹` = **100B params**.
4. Solve D: `D = 20 × 10¹¹ = 2×10¹²` = **2T tokens**.
5. Punchline: both N and D scale as √C — so 10× more compute means ~3.16× bigger model *and* ~3.16× more data, not a 10× bigger model. This is exactly how labs size models before spending $100M, and it's why GPT-4 isn't 10× the parameters of GPT-3.

## Closing line
Scale changed; the architecture didn't. The next gap isn't capability — a model that completes text isn't yet a model that helps you. That gap is alignment (L16).
