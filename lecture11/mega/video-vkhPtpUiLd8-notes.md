# Video reference: what to borrow for L11M

Research and implementation note, 2 September 2026. Updated for the 173-page handout.

## Recommendation

Borrow the video's persistent row-and-column walkthrough, then finish the same example with next-token prediction and learning. Search is on pages 73-78; the bank calculation begins on pages 79-84 and continues through the trainable model on pages 85-111. Keep the same sentence throughout.

Source: Visual AI, [Self-Attention Explained: How Transformers Actually Work (Full Visual Breakdown)](https://www.youtube.com/watch?v=vkhPtpUiLd8), published 26 April 2026, 13:01 long. Title, author, publication date, duration, and chapter boundaries were checked against YouTube metadata. I read the English automatic captions and inspected selected official YouTube storyboard sheets showing the projection calculation, attention scores, weighted values, and heatmap. Direct page fetching was throttled and both attempted full-video downloads returned HTTP 403. This was a captions-and-storyboards review, not continuous full-resolution playback. Captions may contain transcription errors; I did not verify every printed matrix entry in the low-resolution frames.

## The video's teaching sequence

Times below are supported by its captions and chapter metadata; links start at the corresponding section.

| Time | What the video does |
|---|---|
| [0:06](https://www.youtube.com/watch?v=vkhPtpUiLd8&t=6s) | Starts with pronoun reference, then contrasts recurrent processing with direct connections. |
| [2:14](https://www.youtube.com/watch?v=vkhPtpUiLd8&t=134s) | Converts three tokens into embedding rows and assembles a 3-by-4 input matrix. |
| [3:19](https://www.youtube.com/watch?v=vkhPtpUiLd8&t=199s) | Follows one input row through a projection matrix to make a query row; then introduces the other projections. |
| [4:03](https://www.youtube.com/watch?v=vkhPtpUiLd8&t=243s) | Fills a query-by-key score grid, followed by scaling and row-wise softmax. |
| [8:22](https://www.youtube.com/watch?v=vkhPtpUiLd8&t=502s) | Expands one row's weighted sum of value vectors. |
| [9:13](https://www.youtube.com/watch?v=vkhPtpUiLd8&t=553s) | Reads a full attention heatmap with named rows and columns. |
| [10:12](https://www.youtube.com/watch?v=vkhPtpUiLd8&t=612s) | Reassembles the calculation into an input-to-output pipeline. It stops at contextual representations rather than a worked next-token loss. |

## Adopted in the revised lecture

1. Derive the vectors already used. Pages 86-89 show the embedding table,
   stack X, and multiply a row by projection columns. Our bank input
   [1,1,0] times WQ=diag(1,0,0) gives q6=[1,0,0]. WK=I and WV=diag(1,2,1)
   reproduce all existing keys and values. This adapts the video's
   [3:19 walkthrough](https://www.youtube.com/watch?v=vkhPtpUiLd8&t=199s).
2. Carry source indices through the mixture. Pages 79-84 retain the six
   word positions through scores, weights, and values. Page 92 explicitly
   adds scaling and recomputes the mixture with full precision.
3. Continue through prediction and learning. Pages 93-104 specify a
   complete vocabulary head, calculate probabilities, generate with fixed
   parameters, then compute the next-token loss and update the model.
   Pages 101-102 show the forward computation as a graph and explain how
   autograd follows its operations backward. The main lecture uses short
   optimizer code and a checked before/after prediction; manual derivatives
   are in the optional appendix and runnable script.
4. Read the masked matrix by named positions. Pages 105-111 introduce
   shifted targets and causality before the full parallel calculation.
   The numerical A and H on page 108 show that row 6 reproduces the earlier
   bank calculation. This adapts the video's
   [9:13 matrix reading](https://www.youtube.com/watch?v=vkhPtpUiLd8&t=553s).

## The next-token connection

The query comes from the last known input's current vector, never the unknown
answer. Attention mixes known value vectors into h6; the learned vocabulary
head converts that representation into next-token logits. Vocabulary softmax
and attention softmax normalize different things.

Generation appends a chosen token and repeats with the same learned parameters.
Training restores the observed prefix, evaluates the gold next token, and
uses autograd to obtain gradients for E, WQ, WK, WV, U, and b (and learned R
once included). The optimizer updates those parameters. Q, K, V, attention
weights, and h are recomputed intermediates. The runnable bank_training.py
and its JSON evidence retain the detailed derivative checks.

The useful lesson from the video is its visible matrix arithmetic. The
lecture's continuation supplies the prediction and learning objective that
makes the representations useful.

## Claims not to carry over

- Around [4:51](https://www.youtube.com/watch?v=vkhPtpUiLd8&t=291s), and again during the heatmap, the narration treats strong diagonal attention as natural or necessary for preserving a word's meaning. A head need not favour itself. The word “self” refers to queries, keys, and values coming from the same sequence; residual connections provide a separate path for the input state.
- The ending around [12:26](https://www.youtube.com/watch?v=vkhPtpUiLd8&t=746s) equates a single head with one meaning. Do not use that explanation. Multiple heads provide separately learned projections and mixtures; they are not constrained to one linguistic role or one sense per head. See [the Transformer paper, section 3.2.2](https://arxiv.org/html/1706.03762v7#S3.SS2.SSS2).
- Do not transplant its unrestricted sentence-wide attention into the next-token example. Future-token access is acceptable in some encoder settings, but not for our causal predictor.
- Keep heatmap claims about the actual mixing coefficients, not a complete explanation of the model's decision. [Jain and Wallace (2019)](https://aclanthology.org/N19-1357/) provide empirical reasons not to infer prediction importance from attention weights alone.
- Its scaling explanation is stronger than necessary: finite softmax scores do not mathematically force exact zero/one weights, and division by a positive scalar does not centre arbitrary scores at zero. Keep our existing variance/gradient explanation and explicit delayed scaling in the toy calculation.
