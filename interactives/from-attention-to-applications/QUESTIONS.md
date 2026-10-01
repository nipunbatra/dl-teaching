# From Attention to Applications: classroom questions

Every frame has a question and answer. Selected frames also have follow-up questions, available in the reading layout and through **A** in presentation mode. Answers below are synchronized with the slide source.

## 1. From Attention to Applications

[Open this frame](index.html?present#title)

**How do attention blocks become models for different tasks?**

We will use familiar attention blocks to construct decoder-only, encoder and encoder-decoder models, then examine how cross-attention connects a target sequence to its source.

## 2. Recap: the Transformer components we have studied

[Open this frame](index.html?present#attention-recap)

**What does each attention block change?**

Positional encoding supplies order information to token embeddings. Self-attention forms queries, keys and values from the same sequence and combines values into a message for each position. Multiple heads use separate learned projections; their messages are concatenated and projected into an embedding update. Residual addition and the MLP produce contextualised embeddings.

## 3. Different tasks for the same input

[Open this frame](index.html?present#start)

**What kinds of output could we request for this sentence?**

A continuation, a label at each input position, or a Hindi translation. The supplied information and desired output determine the computation.

## 4. Different tasks for the same input

[Open this frame](index.html?present#questions-1)

**What kinds of output could we request for this sentence?**

A continuation, a label at each input position, or a Hindi translation. The supplied information and desired output determine the computation.

## 5. Different tasks for the same input

[Open this frame](index.html?present#questions-2)

**What kinds of output could we request for this sentence?**

A continuation, a label at each input position, or a Hindi translation. The supplied information and desired output determine the computation.

## 6. Different tasks for the same input

[Open this frame](index.html?present#questions-3)

**What kinds of output could we request for this sentence?**

A continuation, a label at each input position, or a Hindi translation. The supplied information and desired output determine the computation.

## 7. Review: contextual token embeddings

[Open this frame](index.html?present#familiar-states)

**Does a Transformer automatically return just one summary vector?**

No. Each token retains an embedding as it passes through the blocks. We write its final contextualised embedding as eᵢ, containing d numbers. E stacks the final source embeddings as rows; here its shape is 6 × d.

## 8. Attention, output representations and training targets

[Open this frame](index.html?present#three-questions)

**What should we identify before naming a model family?**

The allowed information flow, the states we read, and the targets used for training.

## 9. Part I · Generate the next token

[Open this frame](index.html?present#part-decoder)

**Part I · Generate the next token**

Review of the causal Transformer.

## 10. Review: predicting the next token

[Open this frame](index.html?present#decoder-recap)

**Which state predicts school after Raghav goes to?**

The final contextualised embedding e₃, at the position of to, goes through the vocabulary head. m₃ is the attention message received at position 3; it is projected into Δe₃, added to e₃, and followed by the MLP and its residual. It is not a message for the word being predicted.

## 11. Causal attention: current and earlier positions

[Open this frame](index.html?present#decoder-mask)

**Can Raghav’s state use the later word Delhi?**

Not under this causal mask. A query at position i can use positions up to and including i. The training target is the following token.

## 12. Generating a sequence one token at a time

[Open this frame](index.html?present#generate-0)

**What changes after we predict the next token?**

The chosen token is appended to the prefix with its own input embedding and position. Each displayed output is a final contextualised embedding. Read Raghav’s e₁ to predict “goes”, then the embedding e₂ at “goes” to predict “to”, then e₃ at “to” to predict “school”.

## 13. The hidden layer inside a Transformer block

[Open this frame](index.html?present#vocab-hidden)

**Is the vocabulary head another stack of hidden layers?**

In the standard decoder shown here, the hidden processing is already inside the Transformer blocks. Each block has an MLP with a hidden layer. This drawing expands the final block at Raghav’s position: 8 input features, 12 hidden units with a nonlinear activation, and 8 output features, followed by residual addition.

## 14. One output neuron for every vocabulary token

[Open this frame](index.html?present#vocab-scores)

**Are the candidate outputs limited to words in the sentence?**

No. The output layer scores the full tokenizer vocabulary. This example has 50,000 output neurons. We show seven entries, including bicycle, banana and a subword fragment, with dots for the omitted entries. A learned W maps the eight embedding features to 50,000 logits; b adds one bias per output.

## 15. From vocabulary scores to the next token

[Open this frame](index.html?present#vocab-softmax)

**Does softmax consider only the displayed tokens?**

No. Its denominator includes all 50,000 scores. The displayed probabilities use the full vocabulary, and the combined probability of the other 49,993 entries appears below the table. Greedy decoding selects goes, appends it to Raghav, and continues with the new prefix.

## 16. Generating a sequence one token at a time

[Open this frame](index.html?present#generate-1)

**What changes after we predict the next token?**

The chosen token is appended to the prefix with its own input embedding and position. Each displayed output is a final contextualised embedding. Read Raghav’s e₁ to predict “goes”, then the embedding e₂ at “goes” to predict “to”, then e₃ at “to” to predict “school”.

## 17. Generating a sequence one token at a time

[Open this frame](index.html?present#generate-2)

**What changes after we predict the next token?**

The chosen token is appended to the prefix with its own input embedding and position. Each displayed output is a final contextualised embedding. Read Raghav’s e₁ to predict “goes”, then the embedding e₂ at “goes” to predict “to”, then e₃ at “to” to predict “school”.

## 18. Decoder-only models: next-token prediction

[Open this frame](index.html?present#decoder-use)

**What makes this a decoder-only model, and what is self-attention attending to?**

A single causal Transformer stack processes the prompt and generated tokens as one sequence. In each self-attention layer, Q, K and V are projected from the current embeddings of that sequence. The causal mask lets each position read itself and earlier positions. The last updated embedding feeds the vocabulary head.

## 19. Decoder-only examples: text and code completion

[Open this frame](index.html?present#decoder-tasks)

**Does a prompt and an answer imply an encoder-decoder architecture?**

No. In a decoder-only model, the prompt and the generated answer occupy successive positions in one sequence. Causal self-attention processes that sequence. A separate source encoder is not present. Text and code completion illustrate this directly.

## 20. Part II · The whole input is available

[Open this frame](index.html?present#part-encoder)

**Part II · The whole input is available**

Predict labels for a complete input sentence.

## 21. Which words refer to people or places?

[Open this frame](index.html?present#ner-question)

**Do we need to generate a new sentence?**

No. This task assigns a label to each input position. Raghav is a person and Delhi is a place in this teaching example.

## 22. Predicting a label at each input position

[Open this frame](index.html?present#ner-positions)

**What should the unhighlighted words receive?**

OTHER, under our simplified label menu. Every position receives a label, including words that are not named entities.

## 23. Self-attention over the complete input

[Open this frame](index.html?present#encoder-mask)

**Why is it legitimate for Raghav’s state to use Delhi?**

We supplied the whole input and ask for labels about it. Later source words are available information, not future target answers.

## 24. Queries, keys and values in an encoder

[Open this frame](index.html?present#encoder-qkv)

**Do we need a new attention operation to read the whole input?**

No. The same query–key scores, softmax and weighted values apply. The mask now permits all supplied positions.

## 25. One contextual vector per token

[Open this frame](index.html?present#encoder-states)

**Is e₁ just the original embedding for Raghav?**

No. It is the final contextual representation at that position after the Transformer blocks, and can depend on the supplied sentence.

## 26. A shared classifier for all token positions

[Open this frame](index.html?present#ner-shared)

**Are we training six unrelated classifiers?**

No. A single shared classifier maps each d-dimensional state to the same four label scores. Softmax operates across the four classes at each position.

## 27. Four entity labels for each token

[Open this frame](index.html?present#ner-label-menu)

**Does the classifier for Delhi still score PERSON?**

Yes. Every position receives all four class scores. A gold label per token supervises the corresponding probability distribution.

## 28. Encoder: a contextual representation for each token

[Open this frame](index.html?present#encoder-name)

**Must an encoder reduce the input to a single vector?**

No. Its output is a sequence of contextual states. Pooling, a readout token or a token classifier is a separate downstream choice.

## 29. Predicting one label for a sentence

[Open this frame](index.html?present#sentiment-question)

**We have five contextual states. How do we get one class prediction?**

Choose a whole-sequence readout, such as pooling the states or selecting a learned readout position, then apply a classifier.

## 30. Sentence classification with mean pooling

[Open this frame](index.html?present#mean-pooling)

**What shape does mean pooling produce?**

Averaging n vectors of width d produces one vector of width d. A classifier maps that vector to C class scores.

## 31. Sentence classification with a CLS token

[Open this frame](index.html?present#cls-readout)

**How can a token at the beginning collect information from later words?**

The whole-input mask allows its query to read every supplied position, including the later words. We select its final contextual state for the task head.

## 32. Where does the CLS embedding come from?

[Open this frame](index.html?present#cls-initialization)

**Do we initialize CLS with a new random vector for every sentence?**

No. In this text encoder, [CLS] is a special token with its own ID. That ID selects a trainable row of d numbers from the same embedding table used for ordinary tokens. From scratch, initialise that row with small random values once; a pretrained model supplies its saved row. Add positional information, then send the sequence through the encoder.

## 33. CLS learns through the prediction loss

[Open this frame](index.html?present#cls-learning)

**Do we provide a target embedding for CLS, or a true label y?**

We supply the true class label y. The CLS table row and review words pass through the encoder; the updated CLS embedding enters a neural classifier. Softmax gives the predicted class probabilities ŷ. The loss compares ŷ with y. Backpropagation computes gradients through the classifier and encoder to the original CLS embedding-table row.

## 34. Token classification and sentence classification

[Open this frame](index.html?present#same-input-readouts)

**What changes when the whole sentence needs one label?**

The task determines the readout and head. NER reads all ordinary token states through shared parameters. Sentence classification selects the final CLS state.

## 35. Six words, eight features per word

[Open this frame](index.html?present#readout-shapes)

**What does a single row of this matrix represent?**

One word’s updated embedding. We choose a toy model width of eight, so six input words produce six rows of eight numbers: E has shape 6 × 8. A feature column is not a word or a class label.

## 36. NER: follow each word from input to scores

[Open this frame](index.html?present#readout-ner-shapes)

**What happens before we multiply by the classifier weights?**

Each word has an input embedding, with positional information. Whole-input self-attention and MLP blocks update those embeddings using the sentence. The encoder still returns one eight-feature vector per word. The shared classifier then multiplies each updated vector by W and adds b, producing four class scores.

## 37. NER: turn each set of scores into a label

[Open this frame](index.html?present#readout-ner-labels)

**Why is the result 6 × 4 rather than one vector of four scores?**

We classify every input position. The same W of shape 8 × 4 and bias of length four are used for all six rows: scores = E W + b has shape 6 × 4. Softmax across the four columns gives one distribution per word.

## 38. CLS: select one row from seven

[Open this frame](index.html?present#readout-cls-select)

**Does selecting CLS shrink every embedding to one feature?**

No. It selects one position and keeps all eight features in that row. Six ordinary words plus CLS produce seven updated embeddings: 7 × 8. Reading just the CLS row gives 1 × 8.

## 39. A neural classifier reads the updated CLS embedding

[Open this frame](index.html?present#readout-cls-classify)

**Does the classifier receive a one-hot CLS vector?**

No. A vocabulary-length one-hot vector represents the identity of [CLS] for embedding lookup. That lookup returns its dense input embedding. After the encoder processes CLS with the sentence, the updated CLS vector enters the classifier. Our example has eight input features, six hidden units with a nonlinear activation, and three output logits.

## 40. One encoder, several ways to use its embeddings

[Open this frame](index.html?present#encoder-applications)

**Does readout mean another token or another operation after CLS?**

Readout is the general choice of which encoder outputs to use and how to combine them. Selecting the updated e_CLS is one readout. Pooling token embeddings is another. Token classification uses all ordinary token embeddings. There is no extra readout token or mandatory processing step implied by the word here.

## 41. Part III · Sequence-to-sequence prediction

[Open this frame](index.html?present#part-translation)

**Part III · Sequence-to-sequence prediction**

Encode the English input and generate a Hindi translation.

## 42. Example: English-to-Hindi translation

[Open this frame](index.html?present#translation-task)

**Do contextual source vectors alone produce the Hindi sentence?**

They represent the input. To generate a variable-length target sentence, we also need a mechanism that predicts target tokens and a stopping token.

## 43. How does Hindi generation begin?

[Open this frame](index.html?present#hindi-start-token)

**Is शुरू the first word of the translation?**

No. [शुरू] is our Hindi display label for a special beginning-of-sequence token, often called START or BOS. We supply its token ID and look up its embedding e₁ in the target embedding table. Its decoder output predicts the first Hindi word राघव. [समाप्त] labels the end token; it is not printed as part of the translated sentence.

## 44. Step 1: encode the source sentence

[Open this frame](index.html?present#source-encoder)

**What new operation have we introduced so far?**

None. English tokens receive embeddings and positions, then the encoder returns one contextual state per source position.

## 45. First attempt: compress the source into one vector

[Open this frame](index.html?present#source-pool)

**Where have we already used this operation?**

In whole-sentence classification. Here the pooled vector will condition generation instead of feeding a class head.

## 46. Conditioning the decoder on the source vector

[Open this frame](index.html?present#fixed-context-decoder)

**What information does the decoder need at this step?**

It needs the target tokens so far and the source context c. For now, imagine the decoder has access to c at every step; the next diagrams show three ways to provide it.

## 47. A fixed source vector at each generation step

[Open this frame](index.html?present#fixed-generate)

**What changes between these three predictions?**

The source context c stays fixed. The target prefix grows, so the decoder state and next-token prediction can change.

## 48. Option 1: add context to every target embedding

[Open this frame](index.html?present#context-add)

**Can we add c directly to a target embedding?**

Only if their dimensions match. Here c has d_c features and each target embedding has d features. Learn W_P with shape d_c × d, compute u = c W_P, then add u to each eₜ before positional information and the causal decoder.

## 49. Add a position vector at each target position

[Open this frame](index.html?present#context-positions)

**What does “add positions” actually add?**

In this example, pᵢ is a d-dimensional position vector. We add it coordinate by coordinate to the context-conditioned embedding eᵢ + u. शुरू receives p₁, राघव receives p₂ and दिल्ली receives p₃. The decoder therefore receives e₁ + u + p₁, e₂ + u + p₂ and e₃ + u + p₃.

## 50. Causal self-attention over the Hindi prefix

[Open this frame](index.html?present#target-self-attention)

**Is the decoder using self-attention again?**

Yes. शुरू, राघव and दिल्ली exchange information through causal self-attention. Each target position supplies a query, key and value from its current embedding. शुरू can read itself; राघव can also read शुरू; दिल्ली can read all three. The next token में is not supplied yet.

## 51. Updating the embedding at दिल्ली

[Open this frame](index.html?present#target-attention-message)

**Where does the attention message go?**

Project the current Delhi embedding into q₃. Compare it with the keys of the supplied target positions, scale the scores by the square root of the key width, apply the causal mask and softmax, then combine their values into m₃. Project the message with W_O to get Δe₃ and add it to the current embedding. The MLP and its residual follow.

## 52. From target embeddings to the next Hindi token

[Open this frame](index.html?present#context-decoder-path)

**What is the “last embedding,” and why read that one?**

The decoder updates every supplied position through causal self-attention and MLP blocks. Here the last target token is दिल्ली, so we read its updated e₃. The vocabulary head turns it into |V| scores; softmax gives a vocabulary distribution, from which we choose the example next token में.

## 53. Option 2: concatenate context as extra features

[Open this frame](index.html?present#context-concat)

**Does concatenating c create another token?**

Not here: [eₜ ∥ c] concatenates features within each target position. Its width is d + d_c. A learned W_F of shape (d + d_c) × d maps it back to the decoder width. Then add the d-dimensional position vector pₜ and use the causal decoder, last-target-position readout, vocabulary logits and softmax just shown.

## 54. Option 3: prepend context as an extra input position

[Open this frame](index.html?present#context-prepend)

**Why place u before शुरू rather than after the target prefix?**

Project c to a d-dimensional vector u and put it before the target embeddings. Every target query can then attend to that earlier position under the causal mask. If u were placed after the current prediction position, the causal mask would prevent that position from reading it.

## 55. An encoder-decoder generates a target from a source

[Open this frame](index.html?present#encoder-decoder-name)

**What are the two sequences in this architecture?**

The complete supplied English source and the growing Hindi target. Their positions, lengths and vocabularies need not be identical.

## 56. The decoder cannot revisit the source

[Open this frame](index.html?present#source-bottleneck)

**So what is the problem with using one vector?**

A fixed vector can work. The restriction is that every source detail needed later must survive pooling into c. If c does not preserve Delhi, a later decoder step cannot query the original Delhi embedding to recover it.

## 57. Part IV · Cross-attention to the source

[Open this frame](index.html?present#part-cross)

**Part IV · Cross-attention to the source**

Retain one encoder output per source position.

## 58. Retaining the source embeddings for the decoder

[Open this frame](index.html?present#keep-source-states)

**What could we retain instead of just c?**

The full sequence of encoder output states. The decoder can make a source lookup suited to each current target position.

## 59. Source attention when predicting राघव

[Open this frame](index.html?present#cross-origins)

**When predicting राघव, which source position might receive more attention?**

For an intuitive example, give the English Raghav position weight 0.72 and distribute the remaining 0.28 over the other five positions. This illustrates a plausible source lookup for the first Hindi token. These numbers are hand-chosen, not measured attention.

## 60. Step 1: encode the English sentence

[Open this frame](index.html?present#cross-raghav-query)

**What information is available from the source?**

The English encoder produces six contextual embeddings, one per supplied position. Keep them separately so the decoder can attend to different source positions.

## 61. Step 2: initialise the target at [शुरू]

[Open this frame](index.html?present#cross-start-embedding)

**Where does the first target embedding come from?**

The special start token has an entry in the target embedding table. Look up its input e₁ and add p₁. No Hindi word has been generated yet.

## 62. Step 3: update [शुरू] with target self-attention

[Open this frame](index.html?present#cross-target-self-update)

**With only [शुरू] supplied, what can target self-attention read?**

Only the start position itself. Its current embedding supplies the self-attention query, key and value. The attention message is projected and added through the residual connection to update e₁.

## 63. Step 4: form a new query for cross-attention

[Open this frame](index.html?present#cross-new-query)

**Is this the query used in the preceding target self-attention?**

No. Cross-attention applies its own W_Q to the updated target embedding. The self-attention query was formed within the preceding sublayer from its input embedding. The two sublayers have distinct parameters and read different sequences.

## 64. Step 5: match the target query to English keys

[Open this frame](index.html?present#cross-source-match)

**How are the English keys and values constructed?**

Apply the cross-attention W_K and W_V to each encoder output to obtain six source keys and six values. Compare q₁ with each key using a scaled dot product. The next slide normalizes the scores and combines the values.

## 65. Computing the weighted source message

[Open this frame](index.html?present#cross-raghav-weights)

**How do the illustrative weights become information the decoder can use?**

Score q₁ against all six English keys, scale by sqrt(d_k), and apply softmax across source positions. Multiply each source value by its weight and add the six weighted vectors to get c₁. Project that message and add it to the target embedding; after the remaining decoder computation, the vocabulary head can predict राघव.

## 66. The source message updates शुरू’s embedding

[Open this frame](index.html?present#cross-connected)

**What exactly changes when शुरू attends to the English sentence?**

शुरू begins with an embedding-table lookup and positional information. Target self-attention and its residual produce the current e₁. That embedding makes q₁ while a residual path keeps e₁. Cross-attention retrieves the source message c₁; W_O maps it to Δe₁. Add Δe₁ to the retained e₁, giving the शुरू position an embedding informed by the English source.

## 67. Read the updated शुरू embedding to predict राघव

[Open this frame](index.html?present#cross-start-predict)

**Does the updated शुरू embedding become the embedding of राघव?**

No. It remains the contextual embedding at शुरू. After the remaining decoder blocks, a vocabulary head and softmax turn it into next-token probabilities. Choosing राघव appends a new token at position 2, with its own embedding-table lookup e₂. The next generation step can then predict दिल्ली.

## 68. The roles of keys and values

[Open this frame](index.html?present#cross-qkv)

**Why do we need both keys and values?**

The query is compared with keys to produce scores. Softmax turns those scores into weights. Those weights mix the value vectors to produce cₜ.

## 69. The cross-attention calculation

[Open this frame](index.html?present#cross-equation)

**Which dimension does this softmax normalize?**

The source-position index i. For each target query, the weights over all permitted source positions sum to one.

## 70. Multi-head cross-attention uses the same mechanism

[Open this frame](index.html?present#multihead-cross)

**What changes when we use several cross-attention heads?**

Each head has its own learned query, key and value projections. Every head still takes queries from D (current target embeddings) and keys and values from E (final source embeddings). Concatenate the head outputs, then apply W_O.

## 71. Hindi generation: self-attention, then cross-attention

[Open this frame](index.html?present#source-lookup-0)

**What can each attention operation read at this generation step?**

Target self-attention uses only the supplied Hindi prefix, including the current last position. Its weighted message m is projected and added to that target embedding. A separate cross-attention W_Q then forms a query from the updated embedding. English encoder outputs supply the six source keys and values. Their weighted message c produces a second target update. The final vocabulary head predicts the next Hindi token.

## 72. Hindi generation: self-attention, then cross-attention

[Open this frame](index.html?present#source-lookup-1)

**What can each attention operation read at this generation step?**

Target self-attention uses only the supplied Hindi prefix, including the current last position. Its weighted message m is projected and added to that target embedding. A separate cross-attention W_Q then forms a query from the updated embedding. English encoder outputs supply the six source keys and values. Their weighted message c produces a second target update. The final vocabulary head predicts the next Hindi token.

## 73. Hindi generation: self-attention, then cross-attention

[Open this frame](index.html?present#source-lookup-2)

**What can each attention operation read at this generation step?**

Target self-attention uses only the supplied Hindi prefix, including the current last position. Its weighted message m is projected and added to that target embedding. A separate cross-attention W_Q then forms a query from the updated embedding. English encoder outputs supply the six source keys and values. Their weighted message c produces a second target update. The final vocabulary head predicts the next Hindi token.

## 74. Hindi generation: self-attention, then cross-attention

[Open this frame](index.html?present#source-lookup-3)

**What can each attention operation read at this generation step?**

Target self-attention uses only the supplied Hindi prefix, including the current last position. Its weighted message m is projected and added to that target embedding. A separate cross-attention W_Q then forms a query from the updated embedding. English encoder outputs supply the six source keys and values. Their weighted message c produces a second target update. The final vocabulary head predicts the next Hindi token.

## 75. Hindi generation: self-attention, then cross-attention

[Open this frame](index.html?present#source-lookup-4)

**What can each attention operation read at this generation step?**

Target self-attention uses only the supplied Hindi prefix, including the current last position. Its weighted message m is projected and added to that target embedding. A separate cross-attention W_Q then forms a query from the updated embedding. English encoder outputs supply the six source keys and values. Their weighted message c produces a second target update. The final vocabulary head predicts the next Hindi token.

## 76. Hindi generation: self-attention, then cross-attention

[Open this frame](index.html?present#source-lookup-5)

**What can each attention operation read at this generation step?**

Target self-attention uses only the supplied Hindi prefix, including the current last position. Its weighted message m is projected and added to that target embedding. A separate cross-attention W_Q then forms a query from the updated embedding. English encoder outputs supply the six source keys and values. Their weighted message c produces a second target update. The final vocabulary head predicts the next Hindi token.

## 77. Hindi generation: self-attention, then cross-attention

[Open this frame](index.html?present#source-lookup-6)

**What can each attention operation read at this generation step?**

Target self-attention uses only the supplied Hindi prefix, including the current last position. Its weighted message m is projected and added to that target embedding. A separate cross-attention W_Q then forms a query from the updated embedding. English encoder outputs supply the six source keys and values. Their weighted message c produces a second target update. The final vocabulary head predicts the next Hindi token.

## 78. Fixed context and query-dependent context

[Open this frame](index.html?present#fixed-versus-dynamic)

**Do we re-encode the English sentence for each Hindi token?**

No. We can reuse its encoder states. The changing target query produces a potentially different weighted source message at each target step.

## 79. One decoder block, two attention operations

[Open this frame](index.html?present#decoder-two-attentions)

**Why does the decoder need both attention operations?**

Causal self-attention supplies target-prefix context. Cross-attention retrieves from the complete source. The feed-forward sublayer then transforms each target position.

## 80. Part V · Attention patterns and applications

[Open this frame](index.html?present#part-patterns)

**Part V · Attention patterns and applications**

Compare what each attention operation can read.

## 81. Attention patterns and their applications

[Open this frame](index.html?present#three-attention-patterns)

**How does the attention pattern relate to the task?**

An encoder can read the complete English input to build representations for a sentence classifier. A causal decoder reads the supplied prefix to support next-token prediction. Cross-attention lets Hindi target positions consult the English source while translating. The following slides label every row and column with the actual words.

## 82. Encoder attention for sentence classification

[Open this frame](index.html?present#encoder-attention-task)

**Does the encoder itself reduce six words to one summary?**

No. It returns six updated token embeddings. In this example we mean-pool them into one vector, then apply a topic classifier and softmax to obtain p(y | x), where x is the complete sentence and y is a topic label. EDUCATION is an illustrative label.

## 83. Causal attention for next-token prediction

[Open this frame](index.html?present#decoder-attention-task)

**Can the “to” position attend to the previous word “goes”?**

Yes. The last supplied position to can read Raghav, goes and itself. It cannot read school because school has not been supplied. After the decoder blocks, the vocabulary head reads the final e₃ and models p(school | Raghav goes to). Causal attention alone updates embeddings rather than emitting a word.

## 84. Cross-attention between Hindi and English

[Open this frame](index.html?present#cross-attention-task)

**Why is the cross-attention matrix full even though the decoder is causal?**

The complete English source is already supplied, so each Hindi query may read all six source positions. Causality still applies within the Hindi stream. To predict दिल्ली, use the current embedding at राघव after Hindi self-attention, then query English keys and combine English values.

## 85. Part VI · Comparing model architectures

[Open this frame](index.html?present#part-families)

**Part VI · Comparing model architectures**

Compare the architectures used for the example tasks.

## 86. Encoder, decoder-only and encoder-decoder models

[Open this frame](index.html?present#model-families)

**What did we add to connect understanding a source with generating a target?**

A source encoder and a pathway from its representations into the target decoder. Our final arrangement uses cross-attention for that source pathway.

## 87. Choosing an architecture for a task

[Open this frame](index.html?present#match-task)

**Can a decoder-only model also produce a class label or translation?**

Yes. It can generate labels or translations from a suitable prompt and training setup. These rows illustrate useful arrangements rather than exclusive capabilities.

## 88. Choosing an architecture

[Open this frame](index.html?present#decision-tree)

**Does having source text force us to use an encoder-decoder?**

No. This branch asks whether the chosen design has a separate source pathway. A decoder-only model can instead consume source text within its prefix.

## 89. Review: attention, representations and training

[Open this frame](index.html?present#exit-question)

**For NER, classification and translation: who reads whom, what do we read, and what trains it?**

NER uses whole-input attention, all token states and token labels. Classification uses a whole-input readout and one sentence label. Translation uses source encoding, causal target states, cross-attention and next-target-token supervision.

## 90. Appendix: worked examples

[Open this frame](index.html?present#appendix)

**Appendix: worked examples**

Tensor shapes, training and a numerical attention example.

## 91. Tensor shapes in cross-attention

[Open this frame](index.html?present#cross-attention-shapes)

**Must source and target sequence lengths match?**

No. For m target queries and n source positions, A has shape m × n. Each row sums to one. C = AV has shape m × dᵥ per head. With m = 4 and n = 6, the weight matrix is 4 × 6.

## 92. Conditioning with a continuous source prefix

[Open this frame](index.html?present#context-prefix)

**Is [CONTEXT] a word the decoder needs to predict?**

No. Here it labels an injected continuous input vector u = c W_P. Prepend it to the positioned target embeddings, start with [शुरू], and read the last target position to predict the next word.

## 93. Next-token prediction during training

[Open this frame](index.html?present#training-shift)

**Can the decoder see the target it is currently supposed to predict?**

Not at the prediction position: its input is the previous target token and its self-attention is causal. The source remains available through cross-attention.

## 94. Training versus generation

[Open this frame](index.html?present#training-generation)

**What is fed back at inference time?**

The model’s selected prediction. In teacher-forced training we supply the observed previous target token instead.

## 95. Attention head ≠ classification head

[Open this frame](index.html?present#attention-versus-head)

**Do four entity labels imply four attention heads?**

No. The number of attention heads and the number of classes are independent. They name different computations.

## 96. A numerical cross-attention example

[Open this frame](index.html?present#cross-numerical)

**Which source position gets the most weight?**

Delhi. Softmax of scaled scores [0.2, 0.1, 0.3, 2.0] is approximately [0.110, 0.100, 0.122, 0.668]. These weights mix the four source value vectors.

## 97. Computing the weighted sum of values

[Open this frame](index.html?present#cross-numerical-values)

**What does cross-attention return after softmax?**

A weighted sum of value vectors. With the four values shown, cₜ is approximately [1.57, 0.89]. It is a feature vector, not a class label or a word probability distribution.
