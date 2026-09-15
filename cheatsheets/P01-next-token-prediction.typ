#import "../common/cheatsheet.typ": *

#set page(paper: "a4", margin: (top: 9mm, bottom: 10mm, x: 10mm),
  footer: context align(center, text(size: 8pt, fill: palette.muted)[Attention and language · Part 1 · #counter(page).display() / 2]))
#set text(font: "IBM Plex Sans", size: 10.6pt, fill: palette.ink)
#set par(leading: 0.52em, justify: false)
#show math.equation: set text(font: "New Computer Modern Math")
#set list(indent: 10pt, body-indent: 4pt, spacing: 3pt)

#sheet-title([PART 1 · QUICK REFERENCE], [From characters to next-token prediction],
  subtitle: [A fixed context, learned embeddings, a hidden layer, and one distribution over the next token.])
#banner[Context → embedding lookup → concatenate → hidden ReLU → logits → softmax]
#v(6pt)
#grid(columns: (1fr, 1fr), gutter: 6mm,
[
  #section-title([1 · Turn a sequence into training pairs], accent: palette.blue)
  #card(accent: palette.blue)[
    For a window of $w$ tokens, the model estimates
    $ p_theta(x_(t+1) | x_(t-w+1), dots, x_t). $
    The input is the context; the target is the *observed next token*. In the name *aabid*, one training pair is *a a b → i*.
    #v(4pt)
    #grid(columns: (1fr, 1fr), row-gutter: 3pt,
      [*Context*], [*Target*],
      [`- - -`], [`a`], [`- - a`], [`a`], [`- a a`], [`b`],
      [`a a b`], [`i`], [`a b i`], [`d`], [`b i d`], [`-`])
    #v(4pt)
    Here `-` marks the boundary: it pads the initial context and ends generation. Vocabulary: 26 letters + `-` = *27 tokens*.
  ]
  #v(5pt)
  #section-title([2 · An ID selects a learned row], accent: palette.teal)
  #card(accent: palette.teal)[
    A token ID is an index, not a meaningful magnitude. The shared embedding table is
    $ E_"tok" in RR^(27 times 2). $
    Each lookup returns two learned numbers. Rounded rows from the lecture's trained model:
    $ e_a approx [0.63, 0.09] $
    $ e_b approx [-1.25, -0.15]. $
    Both occurrences of `a` look up the *same row*. Their positions in the concatenated input are different.
    #v(4pt)
    These axes are learned coordinates, not letters, token IDs, or fixed human-readable properties.
  ]
  #v(5pt)
  #section-title([3 · Preserve the order by concatenating], accent: palette.orange)
  #card(accent: palette.orange)[
    With $w=3$ and embedding width $d=2$,
    $ a_0 = [e_a, e_a, e_b] in RR^(1 times 6). $
    This makes *one six-number input*, not six tokens. The first, second, and third slots have their own connections to the hidden layer.
    #v(4pt)
    Swapping the tokens swaps input blocks. Summing their embeddings would instead discard this slot information.
  ]
],
[
  #section-title([4 · Keep the hidden layer in the path], accent: palette.blue)
  #card(accent: palette.blue)[
    All equations use *row vectors*:
    $ a_1 = "ReLU"(a_0 W_1 + b_1) $
    $ z = a_1 W_2 + b_2, quad p = "softmax"(z). $
    ReLU acts on each coordinate: $"ReLU"(u)=max(0,u)$. The 32 hidden activations are recomputed for each context. They are not stored parameters.
    #v(4pt)
    Without a nonlinear activation, two affine layers collapse into one affine map.
  ]
  #v(5pt)
  #card(title: [Shape and parameter check], accent: palette.teal)[
    #grid(columns: (1.2fr, 1fr, .7fr), row-gutter: 4pt,
      [*Parameter*], [*Shape*], [*Count*],
      [$E_"tok"$], [$27 times 2$], [54],
      [$W_1$], [$6 times 32$], [192],
      [$b_1$], [$1 times 32$], [32],
      [$W_2$], [$32 times 27$], [864],
      [$b_2$], [$1 times 27$], [27],
      [*Total*], [], [*1,169*])
    #v(4pt)
    Activation widths: $6 arrow.r 32 arrow.r 27$. PyTorch `Linear` stores each weight transposed relative to this row-vector notation.
  ]
  #v(5pt)
  #section-title([5 · Scores become probabilities], accent: palette.orange)
  #card(accent: palette.orange)[
    A logit $z_k$ is a real-valued score, not a probability. Softmax compares all vocabulary scores:
    $ p_k = exp(z_k-c)/(sum_j exp(z_j-c)) $
    where $c=max_j z_j$. Every $p_k$ is nonnegative and $sum_k p_k=1$. Subtracting the same $c$ keeps the probabilities unchanged and avoids exponential overflow.
    #v(4pt)
    *Three-token arithmetic example* (not the 27-token model): logits $[0,1,2]$ give
    $ [1, 2.718, 7.389] / 11.107 $
    $ approx [0.090, 0.245, 0.665]. $
  ]
])

#pagebreak()
#set text(size: 10.4pt)
#sheet-title([PART 1 · LEARNING AND GENERATION], [What changes, and when?],
  subtitle: [Training changes parameters. Generation reuses them to predict, choose a token, and move the window.])
#grid(columns: (1fr, 1fr), gutter: 6mm,
[
  #section-title([6 · Score the observed next token], accent: palette.blue)
  #card(accent: palette.blue)[
    If the observed target has index $y$, its loss is
    $ ell = -ln p_y = -z_y + ln sum_j exp(z_j). $
    Higher probability for the true token means lower loss. For example, $p_y=0.1$ gives $ell approx 2.303$; $p_y=0.5$ gives $ell approx 0.693$.
    #v(4pt)
    Over a batch of $B$ context-target pairs:
    $ cal(L) = -1/B sum_(b=1)^B ln p_(b,y_b). $
    Natural logs measure the loss in *nats*. Perplexity is $exp(cal(L))$; compare it only under the same tokenization and evaluation setup.
  ]
  #v(5pt)
  #section-title([7 · Backprop trains the whole model], accent: palette.teal)
  #card(accent: palette.teal)[
    For one example, the logit gradient is
    $ (partial ell)/(partial z_k) = p_k - bb(1)[k=y]. $
    The chain rule carries this signal through $W_2$, ReLU, $W_1$, and the embedding lookups. An SGD step is
    $ theta <- theta - eta nabla_theta cal(L). $
    $theta$ contains the embedding table and both layers' weights and biases.
    #v(4pt)
    For `aab → i`, the input uses rows `a` and `b`. The two uses of `a` contribute gradients to the same row. The target `i` selects the loss; its embedding is not an input lookup for this pair.
  ]
  #v(5pt)
  #card(title: [The minimal training loop], accent: palette.blue)[
    #set text(size: 9pt)
    #raw("optimizer.zero_grad()\nz = model(context_ids)\nloss = F.cross_entropy(z, target_ids)\nloss.backward()\noptimizer.step()", block: true, lang: "python")
    #tiny-note[Feed raw logits to cross-entropy. Clear old gradients because PyTorch accumulates them. Keep held-out names out of training.]
  ]
],
[
  #section-title([8 · Generate: sample, append, repeat], accent: palette.orange)
  #card(accent: palette.orange)[
    + Start with `- - -` or a chosen prefix.
    + Look up the current three tokens and run the MLP.
    + Choose a token from its 27-way distribution.
    + Append it and keep only the last three tokens.
    + Repeat until `-` is chosen or a length limit is reached.
    #v(4pt)
    *Greedy decoding* picks the largest probability. *Sampling* draws from the distribution; it can choose a non-winning token. Neither requires a parameter update.
    #v(4pt)
    Temperature uses $p="softmax"(z/tau)$, $tau>0$. Smaller $tau$ sharpens the distribution; larger $tau$ flattens it. It does not retrain the model.
  ]
  #v(5pt)
  #section-title([9 · Training is not generation], accent: palette.green)
  #card(accent: palette.green)[
    *Training:* contexts come from observed text; the true next token is known and supplies the loss. Gradients update parameters.
    #v(4pt)
    *Generation:* the next token is unknown; a chosen output enters the next context. Parameters stay fixed.
    #v(4pt)
    Evaluate on held-out text. Split at the name/document level before making overlapping windows, so windows from the same name/document do not leak across the split.
  ]
  #v(5pt)
  #section-title([10 · Why move on to attention?], accent: palette.blue)
  #card(accent: palette.blue)[
    Characters, words, and subwords change the tokens and vocabulary, not the next-token objective.
    #v(4pt)
    A fixed window cannot use clues outside it. With hidden width $h$, concatenating a longer window gives $w d$ inputs and $w d h$ weights in the first layer. Doubling $w$ doubles that weight matrix.
    #v(4pt)
    *Part 2 asks:* can the receiving token read relevant earlier information into a fixed-width update? The representation width stays fixed even when the available context grows.
  ]
])
#v(6pt)
#banner(accent: palette.teal)[IDs select rows. Embeddings are learned parameters. Hidden activations depend on the current input.]
#v(5pt)
#tiny-note[Source: #link("https://nipunbatra.github.io/attention/part1.html")[Part 1 interactive lecture]. Architecture and embedding examples follow its trained name model; displayed values are rounded.]
