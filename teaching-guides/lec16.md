# Lecture 16 · Alignment & Fine-tuning · Teaching Guide
*First-time-instructor companion. Audience: undergrads, one ML course.*

## The spine (say this in 2 sentences)
A pretrained LLM is a *completer*, not a *helper* — "be a helpful assistant" was never in its loss. Alignment closes that gap in stages — SFT (clone good demonstrations) → preference learning (RLHF or DPO, push toward what humans prefer) — and LoRA/QLoRA make each stage cheap by training a tiny low-rank update instead of all the weights.

## Where it sits
Follows L15 (raw pretrained capability: Chinchilla, RoPE, GQA). This lecture turns that capability into ChatGPT-style behavior. It reuses the **same NLL framework from L00** — only "what counts as a good output y" changes. It sets up nothing later directly, but it's where students finally connect "the model I chat with" to "the next-token predictor we built." **Landscape/literacy** lecture: the equations are optional; the *pipeline shape* (SFT → preferences) and the *LoRA arithmetic* are the load-bearing parts.

## 80-minute plan
- **Why pretraining ≠ assistant + SFT** ⭐ ~14 min — the pop quiz (raw GPT-3 continues your prompt as FAQ text), then the SFT recipe + loss masking. **Do live on the board:** the LoRA parameter count (below) — do it right after SFT, when "fine-tuning is expensive" is fresh.
- **LoRA + QLoRA** ⭐ ~22 min — the oil-painting analogy, the rank-1 ΔW = BA construction, the 256× savings, then QLoRA's 4-bit base (140 GB → 35 GB).
- **RLHF** ⭐⭐ ~18 min — dog-training analogy, the reward + KL-penalty objective (conceptually), reward hacking. The full objective equation is ⭐⭐⭐.
- **DPO** ⭐⭐ ~16 min — "just ask A or B," the DPO-at-step-zero pop quiz (loss = log 2), the by-hand worked numeric. The loss-inside-out derivation is ⭐⭐⭐.
- **Reasoning models** ⭐⭐ ~6 min — process rewards, test-time compute as a new scaling axis, the o1→o3 benchmark jump.
- **Buffer / method-picker table** ~4 min.

## Teach it like this (hook → sequence → payoff)
**Hook:** the pop quiz — prompt raw GPT-3 to "write a polite email to my professor" and it instead generates *more prompts* ("also write one asking for marks back, also for…"). It only knows "continue internet text." **Sequence:** SFT teaches it what a *response* looks like → but SFT clones one answer; for open-ended questions you want the model to prefer the *best* of many → RLHF learns a reward model and optimizes against it (with a KL leash) → DPO shows you can skip the reward model and RL loop entirely with one supervised loss → and the newest twist, reasoning models, train the model to *think longer*. **Threaded payoff:** LoRA/QLoRA make all of this runnable on a single GPU instead of a cluster. **Closing payoff:** "Pretraining builds a completer; preferences build a helper."

## Heads-up for YOU (subtle points to get right)
1. **LoRA does *not* make the model smaller.** It reduces *trainable* parameters and optimizer-state memory *during fine-tuning*. The full base model must still be loaded. At inference the adapter merges into the base weights — same final size, same latency as the original. Don't say "LoRA compresses the model."
2. **DPO still needs the reference model.** It eliminates the separate *reward model* and the *RL/PPO loop* — but the frozen SFT reference π_ref must be in memory during training to compute the log-prob ratios. "RLHF without RL," not "alignment without a reference."
3. **The RLHF KL penalty keeps the policy near the *SFT model's distribution*, not near "human text."** Its job is to prevent reward hacking — without it the policy finds degenerate outputs that game the reward model (e.g. pad with whitespace if length is rewarded). The slides' dog-on-a-leash analogy is exactly right; just be precise about *what* it's tethered to.
4. **In LoRA, B is initialized to zero on purpose.** So ΔW = BA = 0 at step 0 and the model starts as an exact copy of the base — no loss spike. (The slide says this; make sure you can explain *why* it matters.)
5. **DPO loss at step 0 is exactly log 2 ≈ 0.693, always.** Because π_θ = π_ref, every ratio is 1, both scores are 0, the margin is 0, σ(0) = ½. It starts at chance, like any binary classifier. This is a great sanity check students can verify — make sure you don't say "it starts high and decreases from somewhere random."
6. **QLoRA's asymmetry is deliberate:** 4-bit *base* (it's 99.9% of the params, so quantizing saves the most) + bf16 *adapters* (tiny, so 16-bit precision keeps gradient updates stable). NF4 is just a smart choice of the 16 codebook values, spaced to match the bell-shaped weight distribution.

## Where students stumble (and the fix)
1. **"Why mask the instruction tokens in SFT?"** Fix: if you *don't* mask them, the model is trained to also generate the user's question. You only want to update on predicting the *helpful response*. Show the loss line with and without the mask.
2. **Thinking RLHF and DPO are unrelated methods.** Fix: DPO is a *closed-form simplification* of the RLHF objective — same goal ("make preferred y more likely while staying near SFT"), but the reward model and PPO are folded analytically into one supervised loss. Use the L16 pipeline table (every row is the same "make good y more likely" objective).
3. **Reward = "the right answer."** Fix: the reward model scores *relative* quality on open-ended prompts where there is no single right answer. It's trained from *pairwise human preferences* (Bradley-Terry), not from gold labels.
4. **Confusing "trainable params" with "model size on disk."** Fix: the LoRA-numbers table — a 7B fine-tune ships as an 8 MB adapter, but you still download the 14 GB base once. The base is shared across tasks; only the adapter swaps.

## If a student asks…
- **"What does SFT actually change vs RLHF?"** SFT does behavioral cloning — imitate demonstrated (instruction → response) pairs via plain NLL. RLHF does preference optimization — given pairs of (better, worse) responses, push the policy toward the better ones while a KL penalty keeps it near the SFT model. SFT teaches *a* correct response; RLHF shapes *general behavior* across many acceptable responses.
- **"In DPO, what if the reference model actually prefers the *rejected* answer?"** (HARD) Training still works. DPO only looks at the *margin* of how π_θ moves *relative to* π_ref. Even if the reference favors the loser, DPO increases the winner's relative log-prob and decreases the loser's, pushing the margin positive. (The slides' worked example uses exactly this case: π_ref prefers "Coffee Shop.")
- **"Why is QLoRA's base in 4-bit but its adapters in 16-bit?"** The base is 99.9% of the parameters — quantizing it gives the big memory win (4×). The adapters are tiny, so keeping them in bf16 buys stable, high-fidelity gradients at negligible memory cost.
- **"What goes wrong if the KL coefficient β → 0?"** Reward hacking. With no leash, the policy drifts arbitrarily far from the SFT model to maximize the (imperfect) reward model — producing verbose, sycophantic, or degenerate text the RM happens to score high. β trades off "improve" vs "stay sane."
- **"Is Constitutional AI / RLAIF just RLHF with the model labeling itself?"** Essentially yes: the model (or a stronger one) generates preference labels from a written set of principles, replacing most human annotation. Scales labeling ~100×; used in Claude's pipeline.
- **"Do reasoning models change the architecture?"** No — same base LLM class. What changes is training (RL with *process* rewards that grade individual reasoning steps, not just the final answer) and inference (spend 10–100× more compute "thinking"). It's a new scaling axis: test-time compute.

## If you're short on time
- **Cut:** the ⭐⭐⭐ RLHF objective equation + symbols slides (keep the dog-on-a-leash analogy); the DPO loss inside-out derivation (keep the step-zero pop quiz + one worked numeric); the process-reward detail.
- **Never cut:** the "raw model continues your prompt" pop quiz, the LoRA parameter-count board example, the SFT→preferences pipeline shape, and the DPO step-zero = log 2 result (it's the cleanest "does the formula make sense?" check in the deck).

## The one live board example (3 min)
**LoRA parameter counting** on one 4096×4096 weight matrix.
1. Full fine-tune: `4096 × 4096 = 16,777,216` trainable params for *this one matrix*.
2. LoRA, rank r = 8: `A` is `4096 × 8 = 32,768`; `B` is `8 × 4096 = 32,768`.
3. LoRA total: `65,536`.
4. Savings: `16,777,216 / 65,536 = 256×` fewer params — and that's *per adapted layer*.
5. Punchline: at this ratio a 7B fine-tune has ~4M trainable params (0.06%) and ships as an ~8 MB adapter on top of the shared public base. That single fact reshaped the open-source LLM ecosystem: everyone downloads the base once and swaps tiny adapters per task.

## Closing line
Pretraining builds a completer; preferences build a helper. Next: what if the data could label *itself*? — self-supervised learning (L17).
