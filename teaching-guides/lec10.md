# Lecture 10 · RNNs, LSTMs & GRUs · Teaching Guide
*First-time-instructor companion. Audience: undergrads, one ML course.*

## The spine (say this in 2 sentences)
A recurrent net is just an MLP that re-uses the *same* weights at every timestep and carries a hidden-state vector forward as memory. Vanilla RNNs do this naively and lose gradients over long ranges; LSTMs/GRUs add gates around an **additive** memory path so the signal survives — the same trick as a ResNet skip connection, but in time.

## Where it sits
First lecture of the sequence module. Everything before (MLP, CNN) was feed-forward: one input in, one output out, no memory. This lecture motivates *why* we need memory, builds the RNN, exposes its fatal flaw (vanishing gradients over time), and fixes it with gating. It sets up the encoder-decoder bottleneck (L11) → attention (L12) → Transformer (L13). The "RNN compresses everything into one $h_t$" complaint at the end is the seed that grows into attention.

## 80-minute plan
- **(0–8 min) ⭐ Hook + why MLPs fail.** Pop-quiz "they ____" sentence; bag-of-words and last-5-words both fail. Land the phrase *"an MLP has no inductive bias for time."*
- **(8–22 min) ⭐ The RNN cell.** Weight-sharing-across-time figure; the recurrence $h_t = \tanh(Wx_t + Uh_{t-1})$; the four I/O patterns (many-to-one, one-to-many, two flavors of many-to-many). **Do live on the board:** one RNN step on "I love" with the toy weights (slide has the numbers).
- **(22–40 min) ⭐ BPTT + vanishing gradients.** Telephone-game analogy → the scalar product $\prod w\tanh'$. **Do live on the board:** the 3-step BPTT product with $w=0.5$ → 0.125, then $0.5^{50}\approx10^{-15}$. This is the emotional core of the lecture.
- **(40–58 min) ⭐ LSTM.** Conveyor-belt analogy; build the 4 gate equations; emphasize the additive $c_t = f_t\odot c_{t-1} + i_t\odot\tilde c_t$. **Do live on the board:** the single-neuron memory-flip ($5.0 \to -1.3$).
- **(58–68 min) ⭐⭐ Why gating fixes gradients.** $\partial c_t/\partial c_{t-1} = f_t$ — element-wise, no matrix product. The $0.99^{100}\approx0.37$ vs $0.5^{100}\approx10^{-31}$ table.
- **(68–76 min) ⭐⭐ GRU + LSTM-vs-GRU.** Two gates, one state, ~15% faster, usually tied. Don't overteach.
- **(76–80 min) ⭐⭐⭐ When RNNs still win + Mamba/RWKV teaser.** Streaming, tiny devices, the parallel-scan comeback. Skip if tight.

The ⭐⭐⭐ "derive the gradient product" matrix slide and the Mamba/RWKV slide are the first cuts.

## Teach it like this (hook → sequence → payoff)
**Hook:** read the "after the long flight... they ____" sentence and ask who "they" is. Make them feel that you need to *carry* "parents" 14 words forward. **Sequence:** MLP can't (no memory) → RNN shares weights and carries $h_t$ → but the gradient dies over time (telephone game) → LSTM puts memory on a protected conveyor belt with gates → GRU is the lighter sibling. **Payoff:** "LSTMs don't remember harder — they learn what to forget," and a one-sentence teaser that even LSTMs compress everything into one $h_t$, which is the crack attention will pry open.

## Heads-up for YOU (subtle points to get right)
1. **The gates are NOT what fixes vanishing gradients — the additive cell-state update is.** The healing comes from $c_t = f_t\odot c_{t-1} + i_t\odot\tilde c_t$, whose gradient along the cell path is just $\partial c_t/\partial c_{t-1}=f_t$ (element-wise multiply, *no matrix multiplication, no $\tanh'$*). The gates are dynamic routers on that highway. Say "additive path = gradient highway; gates steer it." This is identical to a ResNet skip connection, in time. If you say "gates solve vanishing gradients" a sharp student will ask *how* and you'll be stuck.
2. **Vanishing/exploding depends on the spectral norm (largest singular value) of $W$, not on "small numbers."** The slide uses a *scalar* $w=0.5$ for intuition; say out loud that for matrices it's the largest singular value of $\prod W^\top\mathrm{diag}(\tanh')$. $\tanh'\in(0,1]$ only pushes the product down; $W$ can push either way.
3. **The forget gate's name is backwards from its effect.** $f_t=1$ means *keep/remember*; $f_t=0$ means *erase*. It multiplies the old memory. That's why the standard trick is initializing $b_f=1$ (so $\sigma(1)\approx0.73$, gates start mostly open and memory persists early in training). Have this ready — it's a common P3-style question.
4. **Why $\tanh$ and not ReLU inside the recurrence?** Because the same matrix is applied repeatedly; an unbounded activation lets *activations* explode geometrically. $\tanh$ bounds activations to $[-1,1]$. (It does NOT prevent gradient vanishing — that's a separate problem the cell state fixes.) Keep these two failure modes distinct.
5. **Notation collision warning.** The LSTM cell state is $c_t$. In L11/L12 the seq2seq context vector is also $c$. Flag this so students don't conflate "memory cell" with "context vector."
6. **Truncated BPTT is about memory *and* compute, not just speed.** Full BPTT stores every $h_t$ for the backward pass → activation memory grows $O(T)$. TBPTT caps it at $O(K)$. The `h.detach()` is what cuts the gradient graph.

## Where students stumble (and the fix)
- **"Why share weights across time?"** Without sharing you'd need new parameters for step $T+1$ (can't handle arbitrary length), and "cat" at position 2 would be a different feature from "cat" at position 20. Sharing gives translational generalization and finite parameters. (Contrast directly with the 5,000,000-dim MLP slide.)
- **Confusing the cell state $c_t$ with the hidden state $h_t$.** $c_t$ is the protected long-term memory (the belt); $h_t = o_t\odot\tanh(c_t)$ is the filtered "what I expose downstream." GRU collapses these into one — that's literally the simplification.
- **Thinking vanishing gradients mean the *forward* pass forgets.** It's the *backward* pass: the model can't learn long-range dependencies because the gradient signal can't reach early timesteps, not (only) because the forward state forgets. Plot $\|\partial\mathcal L/\partial h_t\|$ vs $t$ to show it.

## If a student asks…
- **"What does the GRU lose by having no output gate?"** (hard) An LSTM can hold something in $c_t$ while choosing *not* to expose it ($o_t\approx0$) — private memory. The GRU's state is fully exposed every step; whatever it remembers is immediately visible to the next layer/timestep. In practice this rarely matters.
- **"Is the LSTM's improvement really the same idea as ResNet?"** Yes — both create an additive identity-ish path so gradients flow through `+` instead of through a chain of multiplications. ResNet does it across *depth*; LSTM does it across *time*.
- **"If clipping fixes exploding gradients, why not just clip to fix vanishing too?"** Clipping caps magnitude — it can stop explosion but can't *resurrect* a gradient that's already $10^{-15}$. Vanishing needs an architectural fix (gated additive path), not a numerical band-aid.
- **"Didn't Transformers kill RNNs?"** For most tasks, yes. RNNs still win for streaming/online inference ($O(1)$ state update per token, no recompute over history) and tiny devices (KB memory). And linear/state-space RNNs (Mamba, RWKV, 2023+) are a quiet comeback — they parallelize training while keeping $O(1)$ inference.
- **"Why 15% / why init $b_f=1$?"** $b_f=1$ → forget gate starts near 0.73, so the network defaults to *remembering* and only learns to forget where useful; without it, early training erases everything. (The 15% is a BERT thing — defer to L14.)

## If you're short on time
- **Cut:** the ⭐⭐⭐ matrix-Jacobian derivation slide; the Mamba/RWKV slide; the bidirectional/stacked slide; one of the worked GRU numeric cases.
- **Never cut:** the BPTT scalar product ($0.5^3\to0.5^{50}$), the LSTM memory-flip board example, and the one line "$\partial c_t/\partial c_{t-1}=f_t$, no matrix multiply." Those three are the lecture.

## Live board example (≈5 min): the LSTM memory flip
Single 1-D neuron. Given: $c_{t-1}=5.0$ ("subject is plural"), and for the new word "was" the trained gates fire $f_t=0.1$, $i_t=0.9$, candidate $\tilde c_t=-2.0$ ("subject is singular").
$$c_t = f_t\,c_{t-1} + i_t\,\tilde c_t = (0.1)(5.0) + (0.9)(-2.0) = 0.5 - 1.8 = \mathbf{-1.3}$$
**The aha:** a strong positive memory (5.0) flipped to negative (−1.3) in *one* step — precisely because the network closed the forget gate and opened the input gate. No un-gated RNN can discard old state this cleanly. Then immediately do the gradient version: along the cell state, $\partial c_t/\partial c_{t-1}=f_t=0.1$; if instead the network needs to *keep* memory it learns $f_t\approx0.99$, and $0.99^{100}\approx0.37$ survives while a vanilla RNN's $0.5^{100}\approx8\times10^{-31}$ is gone.

## Closing line
*"LSTMs don't remember harder — they learn what to forget. And even so, they cram the whole sentence into one vector — which is exactly the crack attention will pry open."*
