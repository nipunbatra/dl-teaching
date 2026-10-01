# Sources and provenance

All diagrams are original editable SVGs. The task sequence and English/Hindi example follow the supplied lecture brief. We reuse the existing lecture shell’s typography and layout. No stock images or decorative generated imagery are used.

- Vaswani et al., **Attention Is All You Need** (2017): https://arxiv.org/abs/1706.03762 — encoder/decoder blocks, scaled dot-product attention, query/key/value origins, causal target masking and shifted training inputs. Primary-source links accompany the applicable frames in the reading view.
- Devlin et al., **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding** (2019): https://aclanthology.org/N19-1423/ — example encoder family, classification readout and token-level task heads. The lecture does not teach its pretraining objectives or require its terminology.
- **Noto Sans Devanagari**: https://github.com/google/fonts/tree/main/ofl/notosansdevanagari — bundled for offline Hindi rendering under the SIL Open Font License; see `fonts/OFL.txt`.

## Authored constructions and examples

The fixed-context baseline pools source states into c and conditions target generation on c. The main lecture now illustrates three explicit teaching designs: projected addition at each target embedding, feature concatenation followed by projection, and a continuous prefix before START. The appendix retains an enlarged prefix diagram. These diagrams are authored constructions with defined tensor shapes, not claims of measured performance or three named historical models. This teaching construction is not presented as a historical architecture or the canonical Transformer encoder-decoder. Both stacks still use self-attention; separate source cross-attention is added later.

The English/Hindi translation, NER labels, generated continuations and illustrative attention weights are authored examples. They are not results from a trained model. The optional numerical calculation recomputes softmax from chosen scaled scores and then mixes explicit value vectors; exact values are in `output/cross-attention-arithmetic.json`.

The full, causal and rectangular matrices show allowed access, not measured weights. Word-sized token boxes simplify real tokenization. Diagrams omit residual paths and normalization where explicitly stated; speaker notes explain how attention messages feed the target representation.

CLS implementation reference: Google Research’s original BERT [modeling.py](https://github.com/google-research/bert/blob/master/modeling.py) performs the embedding lookup, random initialization and positional addition; [run_classifier.py](https://github.com/google-research/bert/blob/master/run_classifier.py) inserts [CLS], converts tokens to IDs and builds the classification training loss. The slides explain this special-token-row design without requiring BERT terminology.
