# Teaching guide: From Attention to Applications

Students know embeddings, positions, Q/K/V, attention, multi-head attention, high-level Transformer blocks and causal next-token generation. They do not need recurrent networks, historical translation systems, BERT pretraining or later vision material.

The lecture is designed for 45–60 minutes. Its 93 main frames include several quick progressive builds. Do not spend a minute on every frame.

| Minutes | Frames | Teaching purpose |
|---|---|---|
| 0–5 | 1–8 | Title, prior-attention recap, different outputs for the same input and three recurring questions |
| 5–11 | 9–19 | Recognize the causal decoder students already know; grow the prefix |
| 11–25 | 20–40 | Whole-input attention, shared NER classifier, encoder name, pooling and CLS |
| 25–38 | 41–56 | Fixed source context and three concrete ways to feed it to the decoder |
| 38–55 | 57–88 | Retain source states; distinguish keys from values, explain changing queries, trace a Hindi token and compare matrices |
| 55–60 | 89–93 | Model families, task choices and final questions |
| Optional | 94–101 | Cross-attention shapes, continuous-prefix implementation, shifted training, two meanings of head, scores-to-vector calculation |

For a 45-minute delivery, abbreviate the decoder recap, discuss only one interactive attention lookup, and explain tensor shapes verbally while showing the comparison. Preserve the pooling-to-cross-attention sequence and the final three matrices. The architecture story remains complete without the appendix.

## Use the earlier attention lecture’s convention

The notation follows `lecture11/attention-part2/02-attention.typ`: mᵢ is the retrieved message, Δeᵢ = mᵢ W_O is the embedding update, and e′ᵢ = eᵢ + Δeᵢ retains the current embedding through a residual path. The MLP and its residual follow. For multiple heads, concatenate the head messages before W_O. Normalisation is omitted from the recap drawing.

Slide 10 traces position 3 in “Raghav goes to”. Its message is m₃ because the receiving word is “to”. After the final block, its embedding predicts word 4, “school”. m₄ would be the message at the school position after that token is supplied. This distinction avoids confusing a receiving position with a next-token target.

Use eᵢ for the embedding at token position i. The diagram labels distinguish input embeddings from updated contextualised embeddings; no final-state superscript is needed. In the local residual equation, e′ᵢ explicitly denotes the result of eᵢ + Δeᵢ. The NER, CLS and generation diagrams use this notation consistently. E collects final source embeddings. We no longer use H for this collection: the earlier lecture used H for stacked messages. In cross-attention, eₜ is the current target embedding before the source update, and D collects those target embeddings. cₜ is the source message, corresponding to mᵢ in self-attention; project it before adding it to the target embedding. A pooled fixed context c remains a separate teaching baseline.

## Hold the source drawing still

Frames 44–46 add source states, pooling and the causal target decoder in fixed positions. Frame 47 shows three next-token predictions with the same c and a growing target prefix. Frames 48–54 implement the context input in three ways; frame 55 names the assembled architecture. Frames 58–66 return to the same drawing, retain all six source states, project source K/V, and add target Q and cross-attention. Ask what changed before advancing.

On frame 47, say: “The decoder has access to c at every step. Now let us make that concrete.” The conditional distribution is `p(y_t | y_<t, c)`. Both stacks still use self-attention; this baseline lacks a separate source cross-attention operation.

Frame 96 in the appendix enlarges the prefix implementation: compute `c = Pool(E)`, project it to decoder width with `u = c W_P`, and prepend u before शुरू and the target embeddings. Use consistent positions and train the construction end to end. This is a teaching construction, not the definition of an encoder-decoder. The source prefix vector stays fixed while the target hidden states change.

The three inserted architecture diagrams are:

- **48 · Addition:** c has d_c features. Project with W_P of shape d_c × d, then add the same u = c W_P to every target embedding before adding positions. The target length stays the same.
- **53 · Feature concatenation:** [eₜ ∥ c] has d + d_c features. W_F of shape (d + d_c) × d returns it to decoder width. This adds features, not a token. Linear fusion decomposes into a target transform plus a context transform, so avoid implying that these first two choices are fundamentally unrelated.
- **54 · Prepending:** project c to u of width d and place u before शुरू. Assign all slots consistent positions. Target queries can read the earlier u through causal self-attention. Appending it after the prediction position would hide it behind the causal mask. There is one extra input slot and no target loss on that source slot in this construction.

These are authored, trainable conditioning designs. The source encoder, fusion/projection and decoder are trained together on the target next-token objective. All three still compress the source into one fixed vector and retain self-attention inside the decoder. None requires a separate cross-attention sublayer; the next section introduces that richer interface.

The bottleneck claim is representational pressure, not impossibility. A fixed-vector design may work; retaining source states provides a richer interface. Do not say the fixed-context decoder has an unchanging internal state or performs no attention.

## Keep the three attention operations distinct

Rows are receiving queries; columns are the positions whose keys/values they can access. Colored cells indicate allowed access, **not equal or measured weights**.

- Encoder self-attention: source queries, source keys, source values; full supplied-input access.
- Decoder self-attention: target queries, target keys, target values; causal target mask.
- Cross-attention: target queries; encoder-output keys and values; target length × source length.

The comparison uses a six-word English encoder input, a three-word English decoder prefix, and two Hindi target positions reading six English source positions. These are separate illustrative tasks. The next three slides label both axes with the actual words. The later shape slide separately uses four target positions to teach m × n shapes.

Frame 68 separates the routes: K participates in the score calculation; V supplies the vectors mixed by the softmax weights. The query affects the returned context through those weights.

Frame 70 is the only dedicated multi-head cross-attention slide. Each head projects Q from target states D and K/V from encoder states E, performs the familiar attention computation, then contributes to concatenation and the output projection. Students already know the multi-head mechanism; focus on where its inputs come from.

Frames 71–77 follow every step: supplied Hindi prefix → current last-position embedding → query → source weights → message → updated embedding → vocabulary prediction. स्कूल is step 4; [समाप्त] is step 7.

Source states can be cached. A given layer/head projects them to K and V; target queries change across positions and decoding steps. The message cₜ is one head’s weighted value sum. Multi-head combination/output projection, residual processing, normalization and MLP produce a target representation that is eventually read by the vocabulary head. cₜ is not a vocabulary distribution.

## Readout questions

For token labeling, use one shared matrix and bias at every position. Row-vector notation is used throughout: `zᵢ = eᵢ W + b`; the four scores have the same label order everywhere. The simplified labels are PERSON, ORGANIZATION, PLACE and OTHER. In the main sentence, only Raghav and Delhi are entities.

Mean pooling changes n × d states into d features. With CLS, n ordinary tokens become n+1 states; selecting the final CLS state produces d features. A task head then returns C scores. Batching adds a leading B dimension.

In this text encoder, [CLS] has its own token ID and trainable row in the same embedding table as ordinary tokens. From scratch, initialise that row once with small random values; with a pretrained model, load the learned row. The CLS lookup receives positional information before the encoder. During training, label-loss gradients pass through the classifier and encoder to the table row, and the optimizer updates it. It is reused across inputs and participates in the same Q/K/V computations as other positions. Its final state depends on the input. Classification loss can train that state’s producing computations; there is no target for each coordinate of the initial embedding. CLS is one design choice, not a compulsory step or an automatically useful retrieval embedding.

## Worked readout shapes, slides 35–39

Use the same six-word Raghav sentence and choose a model width of eight for easy counting. Slide 35 draws six rows and eight feature columns. Slide 36 applies one shared 8 × 4 NER classifier to all rows, producing 6 × 4 scores; softmax is across each row, yielding six label decisions. Dark cells indicate the chosen illustrative class, not measured weights or scores.

Slide 38 adds CLS, so there are seven rows of eight features. Selecting its updated row keeps all eight features and produces 1 × 8. Slide 39 defines an example topic task with EDUCATION, TRAVEL and OTHER; an 8 → 6 → 3 neural classifier returns 1 × 3 scores, giving one sentence-label decision. No trained topic prediction is claimed. Topic labels require an annotation rule and training data.

Keep the three counts separate: positions determine the number of rows; model width determines feature columns; the task determines class columns. Batching is omitted to keep the diagrams readable.

## English and Hindi

English: **Raghav goes to school in Delhi.**

Hindi: **राघव दिल्ली में स्कूल जाता है।**

The six displayed words are teaching positions, not an actual tokenizer’s output. Hindi word order differs from English. For predicting दिल्ली, use `[शुरू] राघव`; for स्कूल, use `[शुरू] राघव दिल्ली में`. Avoid implying that every Hindi word must align to just one source word.

## Illustrative weights and numerical example

All outputs and attention weights are authored examples. No model inference was performed. The seven interactive six-position weight vectors each sum to one and deliberately emphasize Raghav, Delhi and school in turn. Attention patterns in a real network need not be literal translations or complete explanations.

The optional four-key calculation uses **already scaled scores** [0.2, 0.1, 0.3, 2.0]. Its softmax is [0.110379, 0.099875, 0.121988, 0.667757]. Frame 101 then mixes V = [[1,0], [0,1], [1,1], [2,1]] to obtain c = [1.567881, 0.889621], displayed as [1.57, 0.89]. Compute each coordinate aloud. The result is another vector. This four-key calculation is separate from the six-word diagram.

## Discussion prompts

Every frame includes a question, answer and pointing cue in the **Q / A / S** controls. [QUESTIONS.md](QUESTIONS.md) is generated from the same source. The final recap asks students to explain information access, readout and target for NER, sentence classification and translation.

On slide 24, e₁ is Raghav’s current embedding and E stacks the current embeddings for that encoder layer. q₁ = e₁ × W_Q, K = E × W_K and V = E × W_V. The result is the message m₁. Projection names use true subscripts in every SVG, including W_O and the optional W_P.

## Vocabulary head, slides 13–15

After the first generation step, enlarge the MLP in the final block: 8 → 12 → 8 plus residual. Then show the learned vocabulary projection, 8 → 50,000. Seven example entries include bicycle, banana and a subword fragment; ellipses indicate the omitted neurons. Softmax uses all 50,000 logits, including the 49,993 omitted entries. Their combined probability is shown separately. Choose goes greedily. The authored calculation assigns logit -8 to each omitted entry. These are distinct operations: hidden features, vocabulary scores, probabilities, and token selection. The hidden units are inside the Transformer; a standard final vocabulary projection does not require another MLP. Layer normalization is omitted for clarity. Resume the growing prefix at slide 16.

## Decoder-only recap, slides 18–19

Slide 18 follows one prefix through embeddings, repeated causal self-attention and MLP blocks, and the last-position vocabulary readout. Q, K and V come from the same sequence. Read the three-by-three causal mask row by row; filled cells denote allowed access. Slide 19 uses text and code completion. Each joined strip is a single sequence: supplied prefix followed by tokens appended during generation. Trace the shared causal Transformer and vocabulary head, then the append loop. Chat, translation and summarization are discussed after encoder-decoder translation: these tasks can also use a decoder-only model, with instructions and source text inside the prompt. A task name alone does not specify an architecture.

## Recognizable architecture portraits

Use the same visual cues at the naming slides and final recap: blue many-to-many positions for an encoder; purple prefix positions with an amber next-token readout for a decoder; and a teal source-embedding path into a separate target decoder for encoder-decoder. The attention squares show allowed access, not measured weights. The six encoder output arrows retain one embedding per input position; no pooling occurs. These signatures describe the architectures, not exclusive task capabilities.

## CLS forward and backward passes

On cls-learning, read the solid forward path from the CLS table row through the encoder, selected updated CLS embedding, neural classifier and softmax. Here ŷ is a probability vector and y is the supplied positive label. The illustrative loss is −log(0.8) ≈ 0.223. Trace the dashed gradient lane back through the same operations. Backpropagation computes gradients; the optimizer then updates the trainable table, encoder and classifier parameters. The updated CLS activation is recomputed for each input.

## NER from inputs to scores, slides 36–37

Slide 36 keeps all six words in fixed columns. Input embeddings pass through the encoder, updated embeddings pass through the shared classifier, and four score slots appear beneath every word. Track Raghav and Delhi vertically. Bias is added after multiplication. Slide 37 rearranges those outputs into a six-row, four-column score table, then shows per-row softmax and the chosen class. The toy width eight and all illustrative labels remain unchanged.

## CLS classification with neurons

The readout-cls-classify slide separates vocabulary one-hot identity from the dense embedding used by the classifier. Token lookup returns the input CLS row; encoder processing produces the updated CLS vector. The MLP uses eight input features, six hidden units with an activation and three output logits. Softmax of [2, 1, 0] gives approximately [0.665, 0.245, 0.090]. These are illustrative scores. Vocabulary size, embedding width, hidden width and class count are independent. The earlier linear-head example is now expanded to a nonlinear MLP head.

## Readout terminology

Readout names the choice of encoder outputs used by a task. Selecting updated e_CLS and pooling token embeddings are two examples. Do not present “pooling” and “readout” as separate peer mechanisms or imply a new token after CLS. The applications slide names the operations directly.

## Positions and the shared decoder path, slides 49–52

After context addition, show pᵢ as a d-dimensional vector added coordinate by coordinate to eᵢ + u. All three terms have width d; u is shared across the target positions for this source sentence. Then trace three prepared embeddings through causal blocks to three updated embeddings. Read updated e₃ at दिल्ली, compute vocabulary logits, apply softmax and choose में. Concatenation uses pᵢ after projection. Prepending assigns p₁ to u and shifts the target positions to p₂, p₃ and p₄ while retaining target embedding indices e₁, e₂ and e₃.

## Target-language self-attention: frames 50–51

Pause inside the decoder. Frame 50 makes the causal target-to-target connections visible; the future में column is not a supplied input. Frame 51 follows the Delhi query through target keys and values into m₃, Δe₃, the residual and MLP. Source context u is already in the target embeddings. All Q/K/V projections here use target-side embeddings; the later cross-attention sublayer will instead read individual source states. Return to the full decoder and vocabulary path on frame 52.

## Predicting राघव: frames 59–65

Start with an intuitive source lookup: weights [0.72, 0.04, 0.03, 0.09, 0.04, 0.08] over the six English tokens. Then reveal the mechanism. The query is produced at the supplied शुरू position after causal self-attention; राघव is the prediction, not the query input. Keys and values come from the six encoder outputs. Match the query against keys, normalize the scores with softmax, and mix the values into c₁. The scaled scores shown are rounded logarithms of the illustrative weights, so the unrounded calculation reproduces them exactly. Attention need not be literal word alignment in a trained model. Follow the शुरू embedding update on frame 66, then its next-token prediction on frame 67.

## The target residual: frames 66–67

शुरू first receives an embedding lookup and position, then target self-attention. At cross-attention the current target e₁ forks: one branch makes q₁, while the other retains e₁. The source message c₁ is projected into Δe₁ and added at the visible plus node. This changes the contextual embedding at शुरू, not the English source and not the learned embedding-table row during inference. Continue through the MLP and remaining decoder blocks before the vocabulary head predicts राघव. Append that token as position 2 and look up its input e₂.

## Hindi marker labels and the full generation sequence

Frame 43 introduces [शुरू] as a special start ID with its own embedding-table row. It is not the ordinary Hindi word शुरू in the translation. [समाप्त] labels the stop token. The source-lookup sequence now walks through all six Hindi words and the end marker, with all seven steps available in both the PDF and interactive buttons. Only generated words are appended to the target prefix. For predicting स्कूल, the current last input is में at position 4. Source-weight rows are illustrative, sum to one, and should not be described as guaranteed word alignments.

## Two attention updates, then prediction: frames 80–83

Keep the supplied prefix [शुरू] राघव fixed. First update the राघव embedding e₂ using Hindi causal self-attention. Use that updated embedding to form the English cross-attention query. Retrieve c₂ from English values, project it into a new Δe₂ and add it to the retained target stream. Then apply the MLP and remaining decoder blocks. Only the final vocabulary head predicts दिल्ली. Causal self-attention does not make an intermediate token prediction. The two Δe₂ labels refer to local updates in distinct sublayers, with different learned projection parameters.

## Word-labeled matrices and task probabilities: frames 85–88

Use the comparison first, then read one query row in each enlarged matrix. The encoder example obtains six vectors, pools them and classifies the topic with p(y | x). The decoder continuation reads Raghav goes to and predicts school; previous and current input words are permitted, future words are not. Cross-attention has Hindi rows [शुरू], राघव and six English columns; p(दिल्ली | English, supplied Hindi prefix) is produced by the final vocabulary head. Filled cells indicate permission, not learned weights or output probabilities.

## Optional cross-attention shapes: frame 95

The tensor-shape derivation is now in the appendix, after the main lecture ends. Skip it for the conceptual walkthrough; use it when students want to check matrix multiplication dimensions. The three concrete attention examples still lead directly into the model-family recap.

## Progressive query construction: frames 60–64

Reveal the English encoder outputs first, then the start-token embedding and position. Target causal self-attention produces an updated start embedding through the message and residual. Cross-attention forms a new query from that updated stream using its own W_Q. Finally reveal source K and V, projected from the encoder outputs using cross-attention parameters, and match the query against the source keys. The familiar symbols describe the same roles in separate sublayers; they do not imply shared weights. Source K/V can be computed earlier in an implementation. Only [शुरू] is supplied on the target side, so self-attention has one available position and does not itself predict a token.

## Both attention operations at every generation step: frames 71–77

Read each panel from the purple Hindi prefix to the teal English source lookup. At step 1 the start token attends only to itself, so its self-attention weight is 1. At steps 2, 3 and 4 the available prefix grows to two, three and four positions. The last target position forms the self-attention query, receives m, and adds its projected update. Only then is a distinct cross-attention query formed from the updated target embedding. English source attention continues to use six keys and values. The two displayed α sets and W_O projections belong to different sublayers. Each weight row sums to one independently. All displayed weights are illustrative. The vocabulary head runs after both updates and the remaining decoder computation.

## Opening and section transitions (revision 7.23)

Slide 1 introduces the lecture and its three architecture families. Slide 2 recalls positional encoding, self-attention, multi-head message combination, residual updates and the MLP. Continue into the original task questions on slide 3. The five existing section dividers now show the section’s input/output relationship and a highlighted position in the lecture outline; the appendix keeps a separate optional-material divider. All prior teaching frames and repetitions are retained.
