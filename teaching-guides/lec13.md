# Lecture 13 · The Transformer — Built Live · Teaching Guide
*First-time-instructor companion. Audience: undergrads, one ML course.*

## The spine (say this in 2 sentences)
A Transformer block is "communication then thinking": multi-head self-attention mixes information across tokens, then a per-token FFN processes it — each wrapped in a residual + LayerNorm so the block stacks safely 10–100 deep. Add token + positional embeddings and a final linear-to-vocab head and you have GPT; the block structure has been stable since 2017 and only its scale and small details changed.

## Where it sits
The assembly lecture. L12 gave you one attention head; here you wrap it with the three other pieces students already met in isolation — residual connections (gradient highway, from the ResNet lecture), LayerNorm (scale control), and an FFN (the MLP) — plus the one genuinely new ingredient, positional encoding. The payoff is a working ~80-line nanoGPT. Everything downstream (L14 pretraining, L15 LLMs, L22 diffusion text conditioning) is this block.

## 80-minute plan
- **(0–8 min) ⭐ Pop quiz: what's missing from "attention alone"?** Three gaps: (a) attention output is linear in values, (b) order-blind, (c) 12 stacked layers blow up/die. Tell students each Part fixes exactly one.
- **(8–24 min) ⭐ The block.** "Communication (attention) then thinking (FFN)"; the FFN is ~⅔ of parameters and stores facts; residual + LayerNorm let it stack. The 20-line PyTorch block.
- **(24–34 min) ⭐⭐ Pre-norm vs post-norm.** The toll-booth analogy; pre-norm = clean identity gradient path. *Use pre-norm.* **Do live on the board:** the two gradient expressions side by side (the `1 + ...` highway).
- **(34–50 min) ⭐ Multi-head attention.** Team-of-specialists; split $d_{\text{model}}$ into $h$ heads of size $d_k=d_{\text{model}}/h$; concat + $W_O$. The shape trace `(B,N,d)→(B,h,N,d_k)→(B,N,d)`.
- **(50–58 min) ⭐ Parameter accounting.** $4d_{\text{model}}^2$ (attention) vs $2d_{\text{model}}d_{\text{ff}}$ (FFN); the 512/2048 table → FFN is 2× attention. **Do live on the board:** plug in $d_{\text{model}}=512,d_{\text{ff}}=2048$.
- **(58–70 min) ⭐ Positional encoding.** Attention is permutation-equivariant → "dog bites man" = "man bites dog" without it. Sinusoidal multi-handed-clock; the rotation property (relative offsets are linear). **Do live on the board:** $R(2)\cdot PE_5 = PE_7$.
- **(70–78 min) ⭐ Three flavors + causal mask.** Encoder-only (BERT) / decoder-only (GPT) / enc-dec (T5); the mask is one line. The nanoGPT slide.
- **(78–80 min) ⭐⭐⭐ Variations (RoPE/GQA/SwiGLU/FlashAttention).** Name-drop, defer to L15.

The three ⭐⭐⭐ slides (pre/post-norm derivation, the two rotation-property slides, the variations tables) are the cuts — keep their punchlines.

## Teach it like this (hook → sequence → payoff)
**Hook:** "you have attention — stack 12 layers of it. Predict what breaks." Get them to guess the three gaps. **Sequence:** wrap attention in FFN (non-linearity) + residual/LN (stackable depth) = the block → run $h$ heads in parallel (specialists) → account for where the params live (FFN, not attention) → inject order with positional encoding → mask the future for a decoder → stack into nanoGPT. **Payoff:** revisit the pop quiz and show each Part fixed exactly one gap — they effectively re-derived the Transformer. "Five ingredients, none of them new; the genius was gluing them together."

## Heads-up for YOU (subtle points to get right)
1. **Pre-norm vs post-norm — the gradient reason, stated precisely.** Post-norm is $\mathrm{LN}(x+\mathrm{Sub}(x))$: the gradient must pass *through* LN every block, and those (<1) factors compound over depth → unstable past ~24 layers without LR warmup. Pre-norm is $x+\mathrm{Sub}(\mathrm{LN}(x))$: the residual gives $\partial x_{\text{out}}/\partial x_{\text{in}}=\mathbf 1+(\dots)$ — a clean identity highway from layer 100 to layer 1, stable at any depth, no warmup needed. The original Vaswani 2017 used post-norm; *every modern model uses pre-norm.*
2. **Multi-head does NOT add parameters.** Heads partition $d_{\text{model}}$ ($d_k=d_{\text{model}}/h$); total stays $4d_{\text{model}}^2$ for $W_Q,W_K,W_V,W_O$ combined. More heads = more *relations tracked in parallel*, not more capacity. Beyond ~8–16 heads, each $d_k$ gets too small to be useful. Don't say "8 heads = 8× the params."
3. **The FFN is where most parameters (and "facts") live, ~⅔ of the block.** $2d_{\text{model}}d_{\text{ff}}$ (with $d_{\text{ff}}=4d_{\text{model}}$) beats attention's $4d_{\text{model}}^2$ by 2×. Frame attention as "routing/communication" and FFN as "per-token computation/storage." Recent interpretability work supports this division.
4. **Why the FFN at all? Attention output is a *linear* combination of values.** Without the FFN's non-linearity, stacking attention layers collapses to a chain of linear maps. The FFN is the only place per-position non-linear "thinking" happens. This is pop-quiz gap (a).
5. **Positional encoding is needed because attention is permutation-EQUIVARIANT (not invariant).** Permute inputs → outputs permute identically, so "dog bites man" and "man bites dog" get the same (re-ordered) representations and identical logits. PE injects order. Get the term right — the slides say equivariant; reinforce it.
6. **Why LayerNorm, not BatchNorm?** BatchNorm's statistics are across the batch and become unstable with variable-length sequences and small/uneven batches; LayerNorm normalizes across the *feature* dimension of a single token, independent of batch and length. (And RMSNorm, used by Llama, just drops the mean-centering.)
7. **RoPE/ALiBi don't "learn" positions.** They impose fixed geometric structure on the dot product; the model learns *how to use* it via $W_Q,W_K$. Sinusoidal/RoPE extrapolate past training length; *learned* positional embeddings do not.

## Where students stumble (and the fix)
- **"If attention already mixes tokens, why the FFN?"** Attention only produces weighted averages of values — linear. The FFN adds the non-linear transform and most of the capacity. Without it, depth buys nothing.
- **Counting multi-head params wrong (multiplying by $h$).** Walk the shape trace: you reshape, you don't add. Show the 512/2048 table — heads never appear in the totals.
- **Thinking the causal mask is a separate architecture.** It's the same attention module with the upper triangle set to $-\infty$. Decoder-only (GPT) = encoder blocks + causal mask, minus cross-attention. The single architectural switch between BERT and GPT is bidirectional-vs-causal attention.

## If a student asks…
- **"Does multi-head cost more than single-head?"** No — same total params and roughly same FLOPs; you've split one big head into $h$ smaller ones that specialize. The win is representational, not capacity.
- **"How can RoPE/ALiBi be 'just fixed math' yet help the model?"** (hard) They create a position-dependent bias in the $QK^\top$ space; the model's $W_Q,W_K$ learn to exploit that geometry. Nothing about the position scheme is learned, but its *use* is.
- **"Why is GPT-2 'just a bigger GPT-1' but BERT is structurally different?"** GPT-1→GPT-2 is the same decoder-only causal stack, scaled. BERT flips the single switch to bidirectional (no causal mask) and trains with a masked-token objective — different attention pattern and different objective.
- **"Why does generated text turn to garbage even though training loss was great?"** Likely a mask leak — the model saw the future during teacher-forced training. Measure next-token accuracy *autoregressively*, not teacher-forced. Karpathy calls attention-mask bugs the most common DL bug for a reason.
- **"Why pre-norm over post-norm — concretely?"** Pre-norm gives a residual identity path the gradient rides back unobstructed; post-norm forces the gradient through a LayerNorm every block, compounding shrink factors. Pre-norm trains deep stacks without warmup.
- **"Where exactly does cross-attention sit?"** Only in the enc-dec decoder: a third sublayer with Q from the decoder, K/V from the encoder output. GPT/Llama drop it (decoder-only). Stable Diffusion uses it to inject text into images (L22).

## If you're short on time
- **Cut:** both ⭐⭐⭐ rotation-property slides (state "shifting position $k$ is a fixed 2D rotation, so relative offsets are linear and learnable"); the pre/post-norm derivation slide (keep the one-line conclusion); the two variations tables; the multi-head numeric slide.
- **Never cut:** the block structure (attention + FFN + residual + LN), pre-norm = identity highway, the parameter table showing FFN > attention, and "PE exists because attention is permutation-equivariant." Those are the exam-and-life essentials.

## Live board example (≈5 min): parameter accounting + sinusoidal rotation
**Part 1 — where params live.** $d_{\text{model}}=512$, $d_{\text{ff}}=2048$:
- Attention: $4\cdot512^2 = 1{,}048{,}576$ (~33%)
- FFN: $2\cdot512\cdot2048 = 2{,}097{,}152$ (~66%)
- LayerNorm ×2: $4\cdot512 = 2{,}048$ (<0.1%)
- **Total ≈ 3.15M.** The aha: "thinking" (FFN) uses 2× the params of "communication" (attention) — and the head count never entered the math.

**Part 2 — relative position is a rotation (30 sec).** With $\theta=0.1$: $PE_5=[\sin0.5,\cos0.5]=[0.479,0.878]$. Apply the rotation $R(2)=\left[\begin{smallmatrix}0.980&0.199\\-0.199&0.980\end{smallmatrix}\right]$: $R(2)\,PE_5=[0.644,0.765]=PE_7$. The aha: to attend "2 steps back," the model just applies a fixed linear rotation — trivial for a linear layer to learn, and it extrapolates beyond training length.

## Closing line
*"The Transformer is five ingredients, none of them new — the genius was gluing them into one block you can stack 100 deep. Next: before it reads anything, someone has to chop text into tokens — and that choice haunts everything downstream."*
