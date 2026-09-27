# CNN interactive lecture — teaching and verification notes

One canonical source: [CNN lecture](../interactives/cnn/index.html).
Use `index.html?present` for the continuous classroom deck. The old part URLs
only redirect; they contain no separate slide content.

## Teaching decisions

The attention parts I–III establish a useful pattern: motivate a computation,
hold its objects in view, let students predict a result, then reveal exact
numbers. A real numerical model should drive both explanations and controls.
The new CNN sequence adopts that pattern and the restrained attention palette,
with a small independent runtime inside this repository.

The original ML CNN notes contain a substantial ImageNet history opening,
visual filtering, placement-count exercises, pooling, multiple channels, and
LeNet accounting. This version starts with the architectural question and
retains the arithmetic and complete-network ledger. It avoids representing
flattening as information destruction. It also avoids the old suggestion that
repeated valid 5×5 convolutions starting at width 32 eventually yield width 1:
the sequence ends at 4, after which the next such convolution is invalid.

The existing DL L8 is already more rigorous. Its exact 5×5 band input,
horizontal contrast kernel, pet evidence, geometry, receptive-field reasoning,
compute accounting, and accumulated gradients remain the core. The new work
makes those mechanisms manipulable instead of simply adding more prose.

Advanced material is introduced where it answers a question:

- Channel reduction resolves what a full-depth filter computes.
- The 1×1 layer separates channel mixing from spatial context.
- A shift experiment makes boundary and stride assumptions testable.
- The receptive-field inspector separates bounding span from sampled support.
- Parameter, MAC, and activation counts separate three different costs.
- Shared-weight backpropagation connects convolution to the previous course
  material rather than treating it as a separate learning algorithm.
- A real 26-parameter network joins forward computation, cross-entropy, and
  weight updates. The next two frames inspect its pooled features and class
  logits, using the same current weights and linked example selectors.
- Depthwise separation, residual connections, and transfer are bridges to L8B,
  not an extended architecture catalog.

## Classroom route

31 frames in one deck, approximately 100–130 minutes including experiments.
There is one title, one continuous frame counter, and one overview. The route
follows the course material: motivate locality and sharing; work through the
filter; combine channels; count legal starts; test pooling and shifts; trace
context and cost; revisit LeNet; build the DL classifier; accumulate shared
gradients; train and inspect a tiny CNN; then bridge to modern architectures.
The architecture and transfer sections can be assigned as reading when needed.

Pixel edits stay in their worksheet. Training state survives navigation across
the entire lecture and resets on a page reload. The forward trace and class
calculation share the same selected example and current trained weights.

## Explicit source-to-deck mapping

| Source inspected | Concrete reuse |
| --- | --- |
| `ml-teaching/neural-networks/assets/cnn-notes.pdf`, pp. 18–24 | Why image locality and repeated patterns motivate convolution; correct the old statistical-independence wording. |
| `ml-teaching/notebooks/cnn-edge.ipynb` | Exact 6×6 image, first three columns 1 and remaining columns 0; vertical `[1,0,-1]` kernel. Select the ML preset in the convolution widget. |
| `ml-teaching/notebooks/convolution-operation.ipynb` and `convolution-operation-stride.ipynb` | Movable patches, padding, legal starts, and edge filtering. State cross-correlation explicitly because SciPy convolution and the PyTorch/Keras convention differ. |
| ML notes, pp. 39–55 | Padding, stride, pooling, channel-by-channel accumulation, and activation. |
| ML notes, pp. 56–73; `ml-teaching/notebooks/cnn.ipynb` | Live LeNet ledger: slides' 32×32 gives 400 flattened values and 61,706 parameters; notebook's 28×28 gives 256 and 44,426. |
| `dl-teaching/lecture8/L8-cnns.typ` | Exact 5×5 band input, signed 3×3 kernel, pet crop, 5,418-parameter classifier, and 1,622,336 MACs. |
| DL L8 receptive-field example | Conv → pool → conv, receptive fields 3→4→8, output jumps 1→2→2. Default in the support inspector. |
| DL L8 shared-gradient example | x=[2,1,3], k=[0.5,-1], g=[1,2]; dK=[4,7], dX=[0.5,0,-2], db=3. |
| `dl-teaching/lecture8b/L8b-modern-cnns.typ` | 1×1 channel mixing, depthwise factorization, residual correction, and transfer-learning bridge. |

On this machine the repositories are `/Users/nipun/git/ml-teaching` and
`/Users/nipun/git/dl-teaching`. The requested `~/git-dl-teaching` spelling does
not resolve; the existing DL checkout is the latter path.

## Reference review

Reviewed local attention sources and the ML/DL decks, plus official Stanford
CS231n 2025 and Michigan EECS 498/598 convolution lecture slides, MIT OCW's
CNN/transfer lecture, D2L's derivation and channel chapter, Distill's receptive
field article, and Zhang's ICML 2019 anti-aliasing paper. CNN Explainer and
PyTorch are supporting links. Videos are linked as further viewing; this was
not an end-to-end video review. The full links, selection rationale, and image
attribution are in [sources.html](../interactives/cnn/sources.html).

## Verified on 27 September 2026

- `node scripts/check_cnn_math.cjs`: known convolution answers, geometry,
  pooling, shared/input gradients, shift counterexamples, receptive fields,
  compute counts, initialization, and actual loss reduction.
- Central finite differences for all 26 parameters: maximum absolute error
  1.22e-10 at the published initialization.
- `node scripts/check_cnn_browser.cjs`: 31 frames with reveals opened; zero
  clipped frames, page errors, or failed content requests; 390-pixel reading
  layout; keyboard controls, editable pixels, reset, selection, photo filtering,
  training, and shared forward state.
- Visual inspection of title, convolution, channels, receptive field,
  forward/training, and phone views; screenshot artifacts under
  `tmp/cnn-review/`.
- `uv run --offline --with torch python interactives/cnn/tiny_cnn.py --check`:
  PyTorch matches initial logits, mean loss, and every parameter gradient at
  float64 tolerance 1e-12. Both implementations reach training loss about
  0.001890 after 600 steps; all six training examples classify correctly.
- Quarto renders the course home and lecture index into a temporary directory;
  interactive resources are covered by the existing `interactives/**` rule.

These results check an educational implementation, not performance on an
independent image dataset. Existing lecture sources and unrelated local work
are preserved. Publication uses the DL teaching repository’s existing GitHub Pages workflow,
with the canonical deck at `https://nipunbatra.github.io/dl-teaching/interactives/cnn/`.
