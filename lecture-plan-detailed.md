# ES 667: Deep Learning — Detailed 24-Lecture Plan

**Instructor:** Prof. Nipun Batra, IIT Gandhinagar | **Semester:** Aug 2026
**Prerequisites:** ES 654 (ML) — students know MLPs, basic CNNs/RNNs, SGD, backprop, SVMs, PCA
**References:** Goodfellow et al. (2016), Bishop & Bishop (2024), Prince (2023), d2l.ai, CS231n, CS224n, Karpathy Zero to Hero, Andrew Ng DL Specialization, NYU DL (LeCun/Canziani), MIT 6.S191

---

## Lecture 1: Why Deep Learning + MLP Recap

### Subsections
1. **The DL Revolution**: Hardware, data, and scale vs. classical ML; ImageNet inflection (2012)
2. **DL as Representation Learning**: Hierarchy of features; hand-crafted vs learned representations
3. **MLP Formalism**: Weights, biases, affine transforms, non-linearities (ReLU, GELU, Swish)
4. **Loss Functions via MLE**: Cross-entropy from maximum likelihood; softmax mechanics
5. **Backpropagation Recap**: Chain rule as algorithm, computational graphs, Jacobians
6. **The Training Loop**: PyTorch forward/backward/step; zero_grad gotcha

### Key Equations
- Cross-entropy from MLE: $\mathcal{L} = -\sum y_i \log \hat{y}_i$
- Softmax: $\hat{y}_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$
- Chain rule for Jacobians: $\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathbf{h}}{\partial \mathbf{x}}^\top \frac{\partial \mathcal{L}}{\partial \mathbf{h}}$
- Parameter counting for MLP dimensions

### Figures
1. Traditional ML pipeline (feature engineering) vs DL (end-to-end)
2. DL timeline (AlexNet → GPT-4)
3. Computation graph of 2-layer MLP (forward and backward)
4. Activation function plots (Sigmoid, Tanh, ReLU, GELU, Swish) + their gradients
5. Feature hierarchy visualization (pixels → edges → textures → parts → objects)

### Notebooks
1. **Micrograd**: Build a scalar autograd engine from scratch (Karpathy Lecture 1 style), then use it to train a tiny MLP

### Papers
- LeCun, Bengio, Hinton (2015). *Deep Learning* (Nature)
- Prince (2023), Ch 1-4

---

## Lecture 2: Universal Approximation, Going Deep, ResNets

### Subsections
1. **UAT Formal Statement**: Cybenko (1989), Hornik (1991); what it guarantees and what it doesn't
2. **How ReLU Networks Approximate**: Bump functions from pairs of ReLUs; piecewise-linear approximation
3. **Why Depth Wins**: Compositionality, circuit complexity, Telgarsky (2016) depth separation
4. **Vanishing/Exploding Gradients**: Sigmoid gradient product, spectral radius of Jacobians
5. **The Degradation Problem**: He et al. experiment — deeper plain nets are WORSE at training
6. **Residual Networks**: Skip connections, gradient highway ($\nabla \geq I$), loss landscape smoothing

### Key Equations
- UAT statement with $\epsilon$-approximation bound
- Gradient product: $\frac{\partial \mathcal{L}}{\partial \mathbf{W}_1} = \prod_{l=1}^{L} \frac{\partial \mathbf{h}_l}{\partial \mathbf{h}_{l-1}}$
- ResNet forward: $\mathbf{h}_{l+1} = \mathbf{h}_l + \mathcal{F}(\mathbf{h}_l, \mathbf{W}_l)$
- ResNet gradient: $\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \frac{\partial \mathcal{F}}{\partial \mathbf{x}} + \mathbf{I}$
- Xavier derivation: $\text{Var}(w) = \frac{2}{n_{in} + n_{out}}$
- He derivation: $\text{Var}(w) = \frac{2}{n_{in}}$ (compensates ReLU halving)

### Figures
1. UAT approximation: 2, 5, 20, 100 neurons fitting a target function
2. ReLU bump construction (pairs of ReLUs summing to approximate)
3. Compositionality: shallow (all combos) vs deep (reusable parts)
4. ResNet vs plain network: training/test error with increasing depth
5. Loss landscape 3D: rugged (no skips) vs smooth (with skips) — Li et al. 2018
6. Gradient flow comparison: plain vs ResNet across 50 layers
7. Weight initialization effect on activation distributions (too small / just right / too large)

### Notebooks
1. **Depth vs Width**: Train wide-shallow vs narrow-deep on spiral dataset; visualize decision boundaries and loss curves
2. **ResNet blocks**: Build residual blocks; compare gradient flow in 20-layer plain vs ResNet MLP on MNIST

### Papers
- He et al. (2015). *Deep Residual Learning*
- Li et al. (2018). *Visualizing the Loss Landscape of Neural Nets*
- Telgarsky (2016). *Benefits of Depth in Neural Networks*
- Glorot & Bengio (2010). *Understanding difficulty of training deep feedforward networks*

---

## Lecture 3: Training Deep Networks in Practice

### Subsections
1. **Tensors and GPU Acceleration**: Memory layout, CUDA basics, CPU↔GPU transfers
2. **PyTorch Autograd**: Dynamic computation graphs; .grad, .backward(), .no_grad()
3. **nn.Module Architecture**: Parameter registration, forward(), state_dict, model.train()/eval()
4. **Data Pipeline**: Dataset, DataLoader, transforms, num_workers, prefetching
5. **The Complete Training Recipe**: Full MNIST/CIFAR training script with logging, checkpointing
6. **Debugging DL Models**: Gradient checking, overfit-one-batch, learning rate finder

### Key Equations
- Throughput: $T_{\text{batch}} = \max(T_{\text{IO}}, T_{\text{GPU}})$
- Gradient accumulation: effective batch size = micro_batch × accumulation_steps

### Figures
1. PyTorch ecosystem diagram: Dataset → DataLoader → Model → Loss → Optimizer
2. Dynamic vs static computation graphs
3. GPU memory hierarchy (HBM, L2, SRAM)
4. Learning rate finder plot (loss vs LR, log scale)
5. Overfit-one-batch diagnostic pattern

### Notebooks
1. **Makemore MLP**: Build a character-level language model (Karpathy style) — first with raw tensors, then refactor to nn.Module + optim
2. **Training recipe**: Full CIFAR-10 training with checkpointing, TensorBoard logging, LR finder

### Papers
- Paszke et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library*
- Smith (2017). *Cyclical Learning Rates for Training Neural Networks*

---

## Lecture 4: SGD Variants

### Subsections
1. **The Loss Landscape**: Valleys, saddle points, ill-conditioning (Hessian eigenvalue ratio)
2. **Mini-batch SGD**: Gradient noise as implicit regularizer; batch size tradeoffs
3. **Momentum**: EMA of gradients; physical ball-rolling analogy; escaping ravines
4. **Nesterov Accelerated Gradient**: "Lookahead" — evaluate gradient at projected position
5. **AdaGrad**: Per-parameter adaptive learning rates; good for sparse features, but LR decays to zero
6. **RMSProp**: Leaky average of squared gradients; fixes AdaGrad's decay

### Key Equations
- Momentum: $\mathbf{v}_t = \beta \mathbf{v}_{t-1} + (1-\beta)\nabla\mathcal{L}(\theta_{t-1})$; $\theta_t = \theta_{t-1} - \eta \mathbf{v}_t$
- Nesterov: gradient at $\theta_{t-1} - \eta \beta \mathbf{v}_{t-1}$ (lookahead)
- AdaGrad: $G_t = G_{t-1} + g_t^2$; $\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{G_t + \epsilon}} g_t$
- RMSProp: $E[g^2]_t = \gamma E[g^2]_{t-1} + (1-\gamma)g_t^2$

### Figures
1. 2D contour: SGD oscillating in ravine vs momentum smoothing through
2. Nesterov vector diagram (evaluate at lookahead point)
3. AdaGrad per-parameter LR decay over time
4. Condition number illustration (elongated ellipse contours)
5. Optimizer trajectory comparison on Rosenbrock function

### Notebooks
1. **Optimizer race**: Implement SGD, Momentum, RMSProp from scratch; run on 2D Rosenbrock; animate trajectories

### Papers
- Goodfellow et al. (2016), Ch 8
- Sutskever et al. (2013). *On the importance of initialization and momentum in deep learning*

---

## Lecture 5: Adam, AdamW, Learning Rate Schedules

### Subsections
1. **Adam**: Combining momentum (1st moment) + RMSProp (2nd moment)
2. **Bias Correction**: Why moments are biased at $t=1$; derivation of correction factors
3. **The L2 ≠ Weight Decay Problem**: Why L2 regularization in Adam is broken → AdamW
4. **Learning Rate Schedules**: Step decay, exponential, cosine annealing
5. **Warmup**: Why Transformers need warmup; linear warmup + cosine decay
6. **Practical Hyperparameter Selection**: LR, batch size, β₁, β₂, weight decay defaults

### Key Equations
- Adam: $m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$; $v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$
- Bias correction: $\hat{m}_t = \frac{m_t}{1-\beta_1^t}$; $\hat{v}_t = \frac{v_t}{1-\beta_2^t}$
- AdamW update: $\theta_t = (1-\lambda)\theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}$
- Cosine annealing: $\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max}-\eta_{min})(1+\cos(\frac{t\pi}{T}))$

### Figures
1. Uncorrected vs corrected moments at early steps
2. Cosine annealing with linear warmup schedule
3. Adam vs AdamW weight histogram comparison over training
4. Loss curves: Adam vs SGD+Momentum on same task

### Notebooks
1. **Implement AdamW**: Write custom AdamW; compare weight distributions to Adam + L2
2. **Schedule sweep**: Train same model with step decay vs cosine annealing; plot LR and loss curves

### Papers
- Kingma & Ba (2014). *Adam*
- Loshchilov & Hutter (2017). *Decoupled Weight Decay Regularization*

---

## Lecture 6: Classical Regularization

### Subsections
1. **Bias-Variance in Overparameterized Regime**: Double descent curve
2. **L2 Regularization (Weight Decay)**: Bayesian view (Gaussian prior); MAP estimation
3. **L1 Regularization (Sparsity)**: Bayesian view (Laplace prior); geometric L1 vs L2
4. **Early Stopping**: As implicit regularization; optimal stopping criterion
5. **Data Augmentation**: Random crops, flips, color jitter; invariance by design
6. **Mixup and Label Smoothing**: Interpolating examples; softening hard targets

### Key Equations
- L2 gradient: $\nabla\mathcal{L}_{reg} = \nabla\mathcal{L} + \lambda\mathbf{w}$
- L1 subgradient: $\nabla\mathcal{L}_{reg} = \nabla\mathcal{L} + \lambda \text{sign}(\mathbf{w})$
- Mixup: $\tilde{x} = \lambda x_i + (1-\lambda)x_j$; $\tilde{y} = \lambda y_i + (1-\lambda)y_j$
- Label smoothing: $y_{smooth} = (1-\alpha)y_{hard} + \alpha/K$

### Figures
1. Double descent curve (classical U-shape → interpolation threshold → modern descent)
2. L1 diamond vs L2 circle constraint regions (geometric view)
3. Mixup visual: blended images and interpolated labels
4. Early stopping: train vs val loss with optimal stop point
5. Data augmentation examples gallery (crops, flips, color jitter, cutout)

### Notebooks
1. **Double descent**: Train MLP on noisy MNIST subset, sweep width from small to very large; plot test error

### Papers
- Belkin et al. (2019). *Reconciling modern ML and bias-variance tradeoff*
- Zhang et al. (2017). *mixup: Beyond Empirical Risk Minimization*

---

## Lecture 7: Modern Regularization

### Subsections
1. **Dropout**: Co-adaptation prevention; ensemble interpretation; inverted dropout
2. **BatchNorm**: Original ICS hypothesis vs Santurkar's smoothing explanation
3. **BatchNorm Mechanics**: Train mode (batch stats) vs eval mode (running stats); affine params γ, β
4. **LayerNorm**: Why BatchNorm fails for sequences; normalizing across features instead of batch
5. **RMSNorm**: Simplified LayerNorm used in modern LLMs (Llama)
6. **Comparison Matrix**: When to use BN vs LN vs RMSNorm

### Key Equations
- Inverted dropout: $h_{drop} = \frac{h \odot m}{p}$ where $m \sim \text{Bernoulli}(p)$
- BatchNorm: $\hat{x}_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}}$; $y_i = \gamma\hat{x}_i + \beta$
- LayerNorm: same formula but $\mu, \sigma$ computed across features (not batch)
- RMSNorm: $\hat{x}_i = \frac{x_i}{\text{RMS}(\mathbf{x})}$; $\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d}\sum x_i^2}$

### Figures
1. Network with dropped nodes (dropout visualization)
2. BatchNorm vs LayerNorm: matrix diagram showing which axis is normalized
3. Activation histograms before and after BatchNorm
4. Comparison table: BN (CNNs) vs LN (Transformers) vs RMSNorm (LLMs)

### Notebooks
1. **BatchNorm backward**: Implement BatchNorm1d forward AND backward pass manually (Karpathy Lecture 4 style); verify against PyTorch

### Papers
- Ioffe & Szegedy (2015). *Batch Normalization*
- Santurkar et al. (2018). *How Does Batch Normalization Help Optimization?*
- Ba et al. (2016). *Layer Normalization*

---

## Lecture 8: CNN Deep Dive + Classic Architectures

### Subsections
1. **Convolution Operator**: Sparse connectivity, parameter sharing, translation equivariance
2. **Mechanics**: Padding, stride, dilation; output dimension formula
3. **Receptive Field**: How it grows with depth; effective receptive field
4. **Pooling**: Max pooling, average pooling; translation invariance
5. **LeNet-5 → AlexNet**: The GPU + ReLU revolution; dropout in AlexNet
6. **VGG**: Stacked 3×3 = effective 5×5 with fewer params; depth principle

### Key Equations
- 2D Convolution: $S(i,j) = \sum_m \sum_n I(i+m, j+n) K(m,n)$
- Output size: $O = \lfloor\frac{W - K + 2P}{S}\rfloor + 1$
- Receptive field: $RF_l = RF_{l-1} + (K_l - 1) \times \prod_{i=1}^{l-1} S_i$
- Param comparison: one 5×5 = 25 params vs two 3×3 = 18 params

### Figures
1. Convolution sliding window animation frame (dot products)
2. LeNet-5 architecture diagram
3. AlexNet architecture (dual-GPU split)
4. Receptive field growth: one 5×5 vs two 3×3
5. Feature maps at different layers (edges → textures → objects)

### Notebooks
1. **CNN from scratch**: Build a CNN for CIFAR-10; print exact tensor shapes and receptive field at each layer
2. **Visualize filters**: Extract and visualize learned Conv1 filters of a trained AlexNet

### Papers
- Krizhevsky et al. (2012). *ImageNet Classification with Deep CNNs* (AlexNet)
- Simonyan & Zisserman (2014). *Very Deep Convolutional Networks* (VGG)

---

## Lecture 9: Modern CNN Architectures + Transfer Learning

### Subsections
1. **GoogLeNet/Inception**: 1×1 convolutions for dimensionality reduction; Inception module
2. **ResNet (CNN version)**: Bottleneck blocks (1×1 → 3×3 → 1×1); projection shortcuts
3. **MobileNet**: Depthwise separable convolutions; efficiency for mobile/edge
4. **EfficientNet**: Compound scaling (depth × width × resolution)
5. **Transfer Learning**: Feature extraction vs fine-tuning; when to freeze layers

### Key Equations
- 1×1 conv as channel mixing: reduces $C_{in}$ channels to $C_{out}$
- Depthwise separable params: $D_K^2 M + MN$ vs standard $D_K^2 MN$
- EfficientNet scaling: depth $d = \alpha^\phi$, width $w = \beta^\phi$, resolution $r = \gamma^\phi$ s.t. $\alpha\beta^2\gamma^2 \approx 2$

### Figures
1. Inception module (naive vs with 1×1 reduction)
2. ResNet bottleneck block
3. Depthwise separable convolution breakdown (depthwise + pointwise)
4. EfficientNet compound scaling diagram
5. Transfer learning: frozen backbone + new head

### Notebooks
1. **Transfer learning**: Load pretrained ResNet-18, freeze backbone, fine-tune on a small dataset (e.g., Flowers-102 or a custom dataset)

### Papers
- Szegedy et al. (2014). *Going Deeper with Convolutions* (Inception)
- Howard et al. (2017). *MobileNets*
- Tan & Le (2019). *EfficientNet*

---

## Lecture 10: Object Detection, Localization & Segmentation

### Subsections
1. **Classification → Localization**: Bounding box regression; multi-task loss
2. **Two-Stage Detectors**: R-CNN → Fast R-CNN → Faster R-CNN (RPN)
3. **One-Stage Detectors**: YOLO grid formulation; anchor boxes
4. **Metrics**: IoU, Non-Max Suppression, mAP
5. **Semantic Segmentation**: FCN, U-Net encoder-decoder with skip connections
6. **Instance Segmentation**: Mask R-CNN (briefly)

### Key Equations
- IoU: $\frac{\text{Area of Overlap}}{\text{Area of Union}}$
- YOLO loss: coordinate loss + confidence loss + classification loss (multi-term)
- Smooth L1 loss for bounding box regression

### Figures
1. R-CNN → Fast R-CNN → Faster R-CNN evolution
2. YOLO $S \times S$ grid with anchor box predictions
3. U-Net architecture showing encoder-decoder skip connections
4. Non-max suppression step-by-step visualization
5. IoU visualization (overlap vs union regions)

### Notebooks
1. **U-Net from scratch**: Implement U-Net; train on a toy segmentation task (shapes on noisy backgrounds)
2. **YOLO inference**: Load a pretrained YOLO model; run detection on sample images; visualize bounding boxes

### Papers
- Ren et al. (2015). *Faster R-CNN*
- Redmon et al. (2015). *YOLO*
- Ronneberger et al. (2015). *U-Net*

---

## Lecture 11: RNNs, LSTMs, GRUs

### Subsections
1. **Sequence Modeling**: Why MLPs fail on sequences; weight sharing across time
2. **Vanilla RNN**: Hidden state update; unrolled computation graph
3. **BPTT**: Backpropagation through time; truncated BPTT
4. **Vanishing Gradients in RNNs**: Jacobian product; spectral radius analysis
5. **LSTM**: Forget/Input/Output gates; cell state as "conveyor belt"
6. **GRU**: Simplified gating; reset and update gates

### Key Equations
- RNN: $h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t + b)$
- Gradient product: $\frac{\partial h_t}{\partial h_k} = \prod_{i=k+1}^t W_{hh}^\top \text{diag}(\tanh'(\cdot))$
- LSTM cell state: $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$
- LSTM output: $h_t = o_t \odot \tanh(c_t)$
- GRU: $h_t = (1-z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$

### Figures
1. Unrolled RNN computation graph
2. LSTM cell schematic (gates, cell state conveyor belt)
3. GRU cell schematic
4. Jacobian spectral radius visualization
5. LSTM vs GRU vs RNN on long-range dependency task

### Notebooks
1. **LSTM from scratch**: Implement LSTMCell using only Linear layers; verify output matches nn.LSTMCell
2. **Character-level language model**: Train LSTM on Shakespeare; generate text

### Papers
- Hochreiter & Schmidhuber (1997). *Long Short-Term Memory*
- Cho et al. (2014). *Learning Phrase Representations using RNN Encoder-Decoder* (GRU)

---

## Lecture 12: Seq2Seq & Applications

### Subsections
1. **Encoder-Decoder Architecture**: Encoding to fixed context vector $\mathbf{c}$
2. **The Information Bottleneck**: Why fixed-length $\mathbf{c}$ fails for long sequences
3. **Teacher Forcing**: Training with ground truth vs autoregressive generation; exposure bias
4. **Decoding Strategies**: Greedy, beam search, top-k, nucleus (top-p) sampling
5. **Applications**: Machine translation, summarization, speech recognition (CTC overview)

### Key Equations
- Seq2Seq: $P(y_1, \ldots, y_T | x_1, \ldots, x_{T'}) = \prod_t P(y_t | \mathbf{c}, y_1, \ldots, y_{t-1})$
- Beam search score: $\frac{1}{T^\alpha} \log P(y_1, \ldots, y_T)$ (length normalization)
- Top-p sampling: smallest set $V_p$ where $\sum_{v \in V_p} P(v) \geq p$

### Figures
1. Seq2Seq diagram: encoder → context $\mathbf{c}$ → decoder
2. Information bottleneck: long sentence compressed to single vector
3. Beam search tree with beam width 3
4. Teacher forcing vs autoregressive training diagram
5. Exposure bias illustration

### Notebooks
1. **Simple translator**: Encoder-decoder RNN for English→French on a tiny parallel corpus with teacher forcing

### Papers
- Sutskever et al. (2014). *Sequence to Sequence Learning with Neural Networks*
- Holtzman et al. (2019). *The Curious Case of Neural Text Degeneration* (nucleus sampling)

---

## Lecture 13: Attention Mechanism

### Subsections
1. **Motivation**: Solving the Seq2Seq bottleneck — attend to different parts per step
2. **Bahdanau (Additive) Attention**: Alignment model with a learned network
3. **Luong (Multiplicative) Attention**: Dot-product, general, and concat variants
4. **Query-Key-Value Abstraction**: Database retrieval analogy
5. **Scaled Dot-Product Attention**: Why $\sqrt{d_k}$ scaling is needed
6. **Self-Attention**: When Q, K, V all come from the same sequence

### Key Equations
- Attention: $\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$
- Why $\sqrt{d_k}$: for $q, k \sim \mathcal{N}(0,1)$, $\text{Var}(q^\top k) = d_k$ → need to normalize
- Bahdanau score: $e_{ij} = v^\top \tanh(W_1 h_i + W_2 s_j)$
- Context: $c_i = \sum_j \alpha_{ij} h_j$

### Figures
1. Attention weight heatmap (alignment between source and target words)
2. QKV matrix multiplication block diagram
3. Comparison: Bahdanau (additive) vs Luong (dot-product) attention
4. Self-attention: same sequence attending to itself
5. $\sqrt{d_k}$ scaling: softmax behavior with and without scaling

### Notebooks
1. **Attention NMT**: Add attention to Lecture 12's Seq2Seq model; visualize attention heatmaps for translated sentences

### Papers
- Bahdanau et al. (2014). *Neural Machine Translation by Jointly Learning to Align and Translate*
- Luong et al. (2015). *Effective Approaches to Attention-based NMT*

---

## Lecture 14: The Transformer

### Subsections
1. **"Attention Is All You Need"**: Removing recurrence entirely
2. **Multi-Head Attention**: Attending to different representation subspaces
3. **Positional Encoding**: Sinusoidal PE; why position information is needed
4. **The Transformer Block**: Residuals + LayerNorm + MHA + FFN; Pre-norm vs Post-norm
5. **Masked Self-Attention**: Causal masking for autoregressive decoding
6. **Full Architecture**: Encoder stack, decoder stack, cross-attention

### Key Equations
- Multi-head: $\text{MultiHead}(Q,K,V) = \text{Concat}(head_1, \ldots, head_h)W^O$
- Each head: $head_i = \text{Attn}(QW_i^Q, KW_i^K, VW_i^V)$
- Sinusoidal PE: $PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})$
- Causal mask: add $-\infty$ to upper triangle before softmax
- FFN: $\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2$

### Figures
1. Full Transformer architecture diagram (Vaswani et al.)
2. Multi-head attention: parallel heads split and concatenate
3. Causal mask matrix (lower triangular)
4. Sinusoidal positional encoding heatmap
5. Pre-norm vs post-norm Transformer block comparison

### Notebooks
1. **Build a Transformer block**: Implement CausalSelfAttention and TransformerBlock from scratch (Karpathy nanoGPT style); verify tensor shapes
2. **Train a tiny GPT**: Train character-level Transformer LM on Shakespeare

### Papers
- Vaswani et al. (2017). *Attention Is All You Need*
- Karpathy, *Zero to Hero* Lecture 7 (nanoGPT)

---

## Lecture 15: Tokenization & Pretraining

### Subsections
1. **Why Tokenization Matters**: Characters vs words vs subwords; Karpathy's tokenization lecture
2. **BPE Algorithm**: Byte-Pair Encoding step-by-step; building the merge table
3. **WordPiece and SentencePiece**: Alternatives used by BERT and multilingual models
4. **Encoder-only: BERT**: Masked Language Modeling (MLM); [CLS] token; fine-tuning paradigm
5. **Decoder-only: GPT**: Autoregressive LM pretraining; emergent abilities
6. **Encoder-Decoder: T5**: Text-to-text framing for all NLP tasks

### Key Equations
- MLM objective: $\mathcal{L}_{MLM} = -\sum_{i \in \text{masked}} \log P(x_i | x_{\setminus i})$
- Causal LM: $\mathcal{L}_{CLM} = -\sum_t \log P(x_t | x_{<t})$
- BPE merge rule: greedily merge most frequent pair

### Figures
1. BPE merge tree visualization (building vocabulary)
2. BERT architecture: bidirectional attention mask + [MASK] tokens
3. GPT architecture: causal attention mask
4. T5 text-to-text framing diagram (translate, summarize, QA all as text→text)
5. Tokenization artifacts: why LLMs struggle with arithmetic and spelling

### Notebooks
1. **BPE from scratch**: Implement BPE tokenizer; train on a text corpus; visualize merges
2. **BERT vs GPT inference**: Load pretrained BERT and GPT-2 from HuggingFace; compare masked fill-in vs text generation

### Papers
- Sennrich et al. (2015). *Neural Machine Translation of Rare Words with Subword Units*
- Devlin et al. (2018). *BERT*
- Radford et al. (2018). *GPT-1*

---

## Lecture 16: Large Language Models

### Subsections
1. **Scaling Laws**: Kaplan et al. vs Chinchilla — compute-optimal training ($C \approx 6ND$)
2. **Data Pipeline**: Common Crawl, deduplication, quality filtering, data mixing
3. **Modern Position Encodings**: RoPE (Rotary), ALiBi
4. **Efficient Attention**: GQA (Grouped Query Attention), MQA (Multi-Query)
5. **Distributed Training**: Data parallelism, tensor parallelism (Megatron), pipeline parallelism
6. **Emergent Abilities**: In-context learning, chain-of-thought, few-shot prompting

### Key Equations
- Chinchilla: $C \approx 6ND$ (compute ≈ 6 × params × tokens)
- RoPE: $q_m = R_{\Theta,m} W_q x_m$ where $R$ is a rotation matrix
- GQA: K,V heads shared across groups of Q heads

### Figures
1. Chinchilla compute-optimal frontier curves
2. RoPE 2D rotation visualization
3. Tensor parallelism: splitting QKV matrices across GPUs
4. GQA diagram: Q heads grouped, sharing K/V heads
5. Scaling law curves: loss vs compute/data/params

### Notebooks
1. **RoPE implementation**: Modify standard self-attention to use Rotary PE; compare positional generalization

### Papers
- Hoffmann et al. (2022). *Chinchilla*
- Su et al. (2021). *RoFormer* (RoPE)
- Touvron et al. (2023). *Llama 2*

---

## Lecture 17: Alignment & Finetuning

### Subsections
1. **SFT / Instruction Tuning**: Conversation data formatting; chat templates
2. **LoRA**: Low-rank adaptation — inject trainable $BA$ into frozen weights
3. **QLoRA**: Combining quantization with LoRA for memory-efficient finetuning
4. **RLHF**: Reward model training; PPO optimization loop
5. **DPO**: Direct Preference Optimization — bypassing the reward model
6. **Evaluation**: MMLU, HumanEval, AlpacaEval; "LLM psychology" (hallucination, sycophancy)

### Key Equations
- LoRA: $W = W_0 + BA$ where $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$, $r \ll \min(d,k)$
- RLHF reward: $r_\phi(x, y)$ trained on human preferences
- DPO loss: $\mathcal{L} = -\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)$

### Figures
1. LoRA adapter matrices injected into Transformer block
2. RLHF pipeline: SFT → reward model → PPO
3. DPO vs RLHF comparison diagram
4. Parameter-efficient methods comparison: LoRA vs prefix tuning vs adapters
5. Instruction tuning data format example

### Notebooks
1. **LoRA from scratch**: Implement LoRA wrapper; inject into Q,V projections of a small GPT; fine-tune on a tiny instruction dataset

### Papers
- Hu et al. (2021). *LoRA*
- Ouyang et al. (2022). *InstructGPT*
- Rafailov et al. (2023). *DPO*

---

## Lecture 18: Vision-Language Models

### Subsections
1. **Vision Transformer (ViT)**: Patchify → linear projection → position embeddings → Transformer
2. **CLIP**: Contrastive image-text pretraining; dual encoder architecture
3. **Zero-Shot Classification**: Text prompts as classifiers; prompt engineering for vision
4. **Multimodal LLMs (LLaVA)**: Vision encoder → projection → LLM token space
5. **Cross-Attention in Multimodal Models**: Flamingo-style perceiver resampler

### Key Equations
- ViT: $z_0 = [x_{class}; x_p^1 E; \ldots; x_p^N E] + E_{pos}$
- CLIP InfoNCE: $\mathcal{L} = -\frac{1}{N}\sum_i \log\frac{\exp(\text{sim}(I_i, T_i)/\tau)}{\sum_j \exp(\text{sim}(I_i, T_j)/\tau)}$

### Figures
1. ViT: image → patches → linear projection → Transformer
2. CLIP training: image encoder + text encoder + contrastive matrix
3. LLaVA architecture: ViT → projection → LLM
4. Zero-shot classification with CLIP text prompts
5. ViT attention maps overlaid on images

### Notebooks
1. **CLIP zero-shot**: Load pretrained CLIP (HuggingFace); classify images with custom text prompts; visualize similarity scores

### Papers
- Dosovitskiy et al. (2020). *An Image is Worth 16x16 Words* (ViT)
- Radford et al. (2021). *CLIP*
- Liu et al. (2023). *LLaVA*

---

## Lecture 19: Autoencoders & VAEs

### Subsections
1. **Generative Modeling Taxonomy**: Implicit vs explicit density; family tree (VAE, GAN, Flow, Diffusion)
2. **Autoencoders**: Encoder-decoder for compression; latent space; reconstruction loss
3. **Problems with AE Latent Space**: Discontinuity; cannot sample meaningfully
4. **VAEs**: Imposing Gaussian prior on latent space; probabilistic encoder
5. **The Reparameterization Trick**: How to backprop through a stochastic node
6. **ELBO Derivation**: Reconstruction term + KL divergence term

### Key Equations
- Reparameterization: $z = \mu + \sigma \odot \epsilon$, $\epsilon \sim \mathcal{N}(0, I)$
- ELBO: $\log P(x) \geq \mathbb{E}_{q(z|x)}[\log P(x|z)] - D_{KL}(q(z|x) \| P(z))$
- KL between Gaussians: $D_{KL}(\mathcal{N}(\mu, \sigma^2) \| \mathcal{N}(0, 1)) = \frac{1}{2}\sum(\sigma^2 + \mu^2 - 1 - \log\sigma^2)$

### Figures
1. Generative model family tree
2. Autoencoder: encoder → bottleneck → decoder
3. AE vs VAE latent space (discontinuous vs smooth/regularized)
4. Reparameterization trick computation graph
5. VAE interpolation in latent space (digit morphing)

### Notebooks
1. **VAE on MNIST**: Implement VAE; train on MNIST; visualize 2D latent space; generate by interpolation and random sampling

### Papers
- Kingma & Welling (2013). *Auto-Encoding Variational Bayes*
- Prince (2023), Ch 17

---

## Lecture 20: GANs

### Subsections
1. **The Minimax Game**: Generator vs Discriminator; Nash equilibrium
2. **GAN Training Dynamics**: Alternating updates; the delicate balance
3. **DCGAN**: Architectural guidelines (ConvTranspose, BatchNorm, no fully-connected)
4. **Mode Collapse**: What it is; why it happens
5. **Non-Saturating GAN Loss**: Practical fix for generator gradient vanishing
6. **WGAN / WGAN-GP**: Wasserstein distance; gradient penalty (theoretical context)

### Key Equations
- Minimax: $\min_G \max_D \mathbb{E}_x[\log D(x)] + \mathbb{E}_z[\log(1-D(G(z)))]$
- Non-saturating G loss: $\max_G \mathbb{E}_z[\log D(G(z))]$
- WGAN gradient penalty: $\lambda \mathbb{E}_{\hat{x}}[(||\nabla_{\hat{x}} D(\hat{x})||_2 - 1)^2]$

### Figures
1. GAN training loop diagram (alternating D and G updates)
2. Mode collapse visualization (generator outputs collapsing to few modes)
3. DCGAN architecture (fractionally-strided convolutions)
4. Generated image quality over training (progressive improvement)

### Notebooks
1. **DCGAN**: Implement DCGAN on CelebA subset or SVHN; monitor D/G loss balance; generate faces/digits

### Papers
- Goodfellow et al. (2014). *Generative Adversarial Nets*
- Radford et al. (2015). *DCGAN*
- Arjovsky et al. (2017). *Wasserstein GAN*

---

## Lecture 21: Diffusion Models — Theory

### Subsections
1. **Forward Process**: Gradually adding Gaussian noise; noise schedule $\beta_t$
2. **Closed-Form Forward Sampling**: $q(x_t|x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}x_0, (1-\bar\alpha_t)I)$
3. **Reverse Process**: Denoising as iterative refinement; parameterizing $\epsilon_\theta$
4. **DDPM Training Objective**: Simplified MSE loss on predicted noise
5. **Connection to Score Matching**: Score function $\nabla_x \log p(x)$; Langevin dynamics
6. **Noise Schedules**: Linear, cosine; effect on generation quality

### Key Equations
- Forward closed form: $q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar\alpha_t}x_0, (1-\bar\alpha_t)I)$
- DDPM loss: $L = \mathbb{E}_{t,x_0,\epsilon}[||\epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)||^2]$
- Score: $s_\theta(x_t, t) \approx \nabla_{x_t} \log p(x_t)$
- Langevin dynamics: $x_{t+1} = x_t + \frac{\delta}{2}\nabla_x \log p(x_t) + \sqrt{\delta}\epsilon$

### Figures
1. Forward/reverse Markov chain diagram
2. Noise schedule visualization (clean → noisy over T steps)
3. U-Net architecture with time-step conditioning
4. Score field visualization (vector field pushing noise toward data)
5. DDPM sampling: iterative denoising from pure noise

### Notebooks
1. **2D Diffusion**: Implement DDPM on a 2D toy dataset (Swiss Roll); visualize forward noising and reverse denoising

### Papers
- Ho et al. (2020). *Denoising Diffusion Probabilistic Models*
- Song et al. (2020). *Score-Based Generative Modeling through SDEs*
- Prince (2023), Ch 18

---

## Lecture 22: Diffusion Models — Practice

### Subsections
1. **Classifier Guidance**: Using a trained classifier to steer generation
2. **Classifier-Free Guidance (CFG)**: No external classifier; interpolate conditional/unconditional
3. **Latent Diffusion Models**: Stable Diffusion = VAE encoder + diffusion in latent space
4. **Text Conditioning**: Cross-attention with CLIP text embeddings inside U-Net
5. **DDIM**: Deterministic sampling; fewer steps
6. **Diffusion Transformers (DiT)**: Replacing U-Net with Transformer backbone

### Key Equations
- CFG: $\epsilon_{CFG} = \epsilon_\theta(z_t, \emptyset) + w \cdot (\epsilon_\theta(z_t, c) - \epsilon_\theta(z_t, \emptyset))$
- DDIM sampling (deterministic, skip steps)

### Figures
1. Latent Diffusion architecture (pixel space → VAE → latent → diffusion)
2. Cross-attention connecting CLIP text embeddings to U-Net
3. CFG scale effect: grid showing $w = 1, 3, 7, 15$
4. DDIM vs DDPM: quality at 10, 50, 1000 steps
5. DiT architecture diagram

### Notebooks
1. **Guided generation**: Use `diffusers` library; implement CFG loop manually from a pretrained model; sweep guidance scale

### Papers
- Rombach et al. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models*
- Ho & Salimans (2022). *Classifier-Free Diffusion Guidance*
- Peebles & Xie (2023). *Scalable Diffusion Models with Transformers* (DiT)

---

## Lecture 23: Self-Supervised & Contrastive Learning

### Subsections
1. **The Labeling Bottleneck**: Why self-supervision matters at scale
2. **Contrastive Learning Setup**: Anchors, positives, negatives
3. **SimCLR**: Augmentation-based positive pairs; batch negatives; NT-Xent loss
4. **BYOL**: No negatives needed; momentum encoder; stop-gradient
5. **Masked Autoencoders (MAE)**: BERT-style masking for vision
6. **DINO / DINOv2**: Self-distillation with no labels; emergent segmentation

### Key Equations
- NT-Xent (InfoNCE): $\mathcal{L}_{i,j} = -\log\frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N}\mathbb{1}_{k \neq i}\exp(\text{sim}(z_i, z_k)/\tau)}$
- BYOL EMA: $\xi \leftarrow m\xi + (1-m)\theta$

### Figures
1. SimCLR framework: image → augmentations → encoder → projection → contrastive loss
2. BYOL: online network vs target network with stop-gradient
3. MAE: masked patches → encoder (visible only) → decoder → reconstruct
4. DINO attention maps showing emergent segmentation
5. t-SNE of learned representations (supervised vs self-supervised)

### Notebooks
1. **SimCLR mini**: Implement InfoNCE loss; train on CIFAR-10 with SimCLR; plot t-SNE of embeddings to see class clustering without labels

### Papers
- Chen et al. (2020). *SimCLR*
- Grill et al. (2020). *BYOL*
- He et al. (2021). *MAE*
- Oquab et al. (2023). *DINOv2*

---

## Lecture 24: Efficient DL & Open Problems

### Subsections
1. **Inference Bottlenecks**: Compute-bound (prefill) vs memory-bound (decode)
2. **KV-Cache**: Why autoregressive generation is memory-bound; cache implementation
3. **Quantization**: FP16, BF16, INT8, INT4; post-training quantization (AWQ, GPTQ)
4. **Knowledge Distillation**: Teacher-student; temperature-scaled soft targets
5. **FlashAttention**: Hardware-aware attention; tiling for SRAM; IO complexity
6. **Open Problems**: Reasoning / System 2, hallucination, continual learning, multimodal scaling

### Key Equations
- KV-Cache memory: $2 \times L \times H \times d_h \times T \times B \times \text{bytes}$
- Distillation: $q_i = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$; $\mathcal{L} = \alpha \mathcal{L}_{CE} + (1-\alpha) T^2 \mathcal{L}_{KL}$
- FlashAttention IO complexity: $O(N^2 d^2 M^{-1})$ vs standard $O(N^2 d + N^2)$

### Figures
1. Prefill (compute-bound) vs decode (memory-bound) phases
2. FlashAttention: standard attention (many HBM reads) vs tiled (SRAM-resident)
3. Knowledge distillation: teacher soft labels → student
4. Quantization: FP32 → FP16 → INT8 → INT4 precision comparison
5. Speculative decoding: draft model + verify

### Notebooks
1. **KV-Cache**: Take nanoGPT; add KV-cache to generation loop; measure tokens/second speedup

### Papers
- Hinton et al. (2015). *Distilling the Knowledge in a Neural Network*
- Dao et al. (2022). *FlashAttention*
- Dao (2023). *FlashAttention-2*
