# Sources and provenance

Primary technical sources and author project pages. Original explanatory text and diagrams were authored for this lecture. The older course runtime and Avenir/palette/stage conventions informed the navigation and layout.

- https://arxiv.org/abs/2006.11239
- https://arxiv.org/abs/2010.02502
- https://arxiv.org/abs/2102.09672
- https://arxiv.org/abs/2107.03006
- https://arxiv.org/abs/2108.01073
- https://arxiv.org/abs/2112.10752
- https://arxiv.org/abs/2202.00512
- https://arxiv.org/abs/2207.12598
- https://arxiv.org/abs/2208.12242
- https://arxiv.org/abs/2210.02747
- https://arxiv.org/abs/2212.09748
- https://arxiv.org/abs/2301.07093
- https://arxiv.org/abs/2302.05543
- https://arxiv.org/abs/2312.03606
- https://arxiv.org/abs/2502.09992
- https://dreambooth.github.io/
- https://github.com/andreas128/RePaint
- https://github.com/lllyasviel/ControlNet
- https://gligen.github.io/
- https://hojonathanho.github.io/diffusion/
- https://iterative-refinement.github.io/
- https://ommer-lab.com/research/latent-diffusion-models/
- https://sde-image-editing.github.io/
- https://www.samarkhanna.com/DiffusionSat/

## Evidence boundaries

The new 814,145-parameter unconditional model trained for 2,000 steps (seed 667; batch 64; linear T=1000; posterior variance). Its unselected eight-sample batch contains malformed and similar digits. This is not a competitive image model. The older class-conditioned 255,721-parameter model trained for 4,000 steps and used a T=100 cosine schedule. Its existing guidance outputs are retained with their own provenance, not loaded into the new model. Analytical mixture demos use known scores; discrete text traces and attention activations are illustrative.

The source prompt column is “A basket full of cherries”. Some outputs include flowers or an unsuitable container, making prompt adherence visibly imperfect. Published seeds are unspecified. All original source images are retained alongside documented crops. No private dataset or API inference was used.

The source figure reuse notes support local teaching review, not an asserted blanket redistribution license. No publication/deployment was performed.
