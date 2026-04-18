# ES 667: Interactive Demos Plan

Standalone browser-based interactives used as live demos during lectures.
Built in the same framework as [nipunbatra.github.io/interactive](https://nipunbatra.github.io/interactive/) (HTML/JS/Canvas/KaTeX).

## Existing (from ~/git/interactive/)

| Interactive | Lecture | Demo Use |
|-------------|---------|----------|
| **Autograd, Seen** | Lec 1: Backpropagation | Live demo: build a graph, step through forward/backward, show upstream × local rule |
| **Watching a Neural Network Become Universal** | Lec 2: UAT | Live demo: add neurons, watch fit snap to target; train MLP in browser |
| **The Optimizer Race** | Lec 4-5: SGD/Adam | Live demo: drop point on ravine/saddle/plateau, race SGD vs Momentum vs Adam |
| **Attention, Calculated** | Lec 13: Attention | Live demo: step through QKV computation, show attention weights |
| **Text Diffusion, Tiny** | Lec 21-22: Diffusion | Live demo: discrete diffusion forward/reverse on text |
| **RAG, From Scratch** | Lec 16-17: LLMs | Live demo: retrieval-augmented generation pipeline |
| **Seeing the Multivariate Normal** | Lec 2/6: Foundations | Supporting: covariance, marginals (useful for VAE intuition too) |

Also: **MNIST Diffusion Lab** (Streamlit, ~/git/diffusion-interactive/) → Lec 21-22

---

## To Build (prioritized)

### Priority 1 — Core DL concepts (Lectures 1-7)

#### 1. **Softmax Temperature Explorer**
- **Lecture**: 1 (Loss functions)
- **Concept**: How temperature $T$ in $\text{softmax}(z/T)$ affects probability distribution
- **Interactive**: Slider for $T$; bar chart of probabilities updating live; show how $T \to 0$ = argmax, $T \to \infty$ = uniform
- **Also useful for**: Lec 17 (distillation), Lec 22 (diffusion guidance)

#### 2. **Vanishing Gradient Playground**
- **Lecture**: 1-2 (Vanishing gradients, depth)
- **Concept**: Stack layers, pick activation (sigmoid/tanh/ReLU), watch gradient magnitude at each layer
- **Interactive**: Slider for depth (1-50 layers); toggle activation function; live bar chart of gradient magnitudes; dramatic collapse for sigmoid at depth 20+

#### 3. **Weight Initialization Lab**
- **Lecture**: 2 (Initialization)
- **Concept**: Effect of init scale on activation distributions across layers
- **Interactive**: Slider for init std; choose Xavier/He/custom; live histograms of activations per layer (like the figure, but interactive)

#### 4. **Learning Rate Finder**
- **Lecture**: 3, 5 (Training practice, LR schedules)
- **Concept**: Sweep LR from tiny to huge, plot loss vs LR, find the sweet spot
- **Interactive**: Train a small model in-browser; live loss-vs-LR curve; mark optimal range

#### 5. **Regularization Effects**
- **Lecture**: 6-7 (Regularization)
- **Concept**: Compare no reg, L2, dropout, batch norm on a 2D classification task
- **Interactive**: Toggle regularization methods; see decision boundary + loss curves change; visualize dropout masks

### Priority 2 — Vision (Lectures 8-10)

#### 6. **Convolution Visualizer**
- **Lecture**: 8 (CNN foundations)
- **Concept**: Step through 2D convolution: kernel slides over image, dot products, output feature map
- **Interactive**: Pick kernel (edge detect, sharpen, blur, or custom); watch it slide across an image pixel-by-pixel; show output feature map building up
- **Reference**: CS231n has a similar static animation; make it interactive

#### 7. **Receptive Field Explorer**
- **Lecture**: 8-9 (CNN architectures)
- **Concept**: Click an output pixel, highlight which input pixels affect it
- **Interactive**: Stack conv layers (adjustable kernel size, stride, padding); click output neuron; see receptive field highlighted on input

#### 8. **IoU & NMS Playground**
- **Lecture**: 10 (Object detection)
- **Concept**: Drag bounding boxes, see IoU score update live; visualize NMS step by step
- **Interactive**: Draw predicted boxes on an image; show IoU calculation; step through NMS suppression

### Priority 3 — Sequences & Transformers (Lectures 11-15)

#### 9. **RNN/LSTM Gate Visualizer**
- **Lecture**: 11 (RNNs, LSTMs)
- **Concept**: Step through an LSTM cell time-step by time-step; watch gate values and cell state
- **Interactive**: Input a short sequence; step through time; see forget/input/output gate activations as heatmaps; watch cell state evolve

#### 10. **Tokenizer Playground**
- **Lecture**: 15 (Tokenization)
- **Concept**: Type text, see BPE tokenization in real-time; compare char/word/BPE token counts
- **Interactive**: Text input box; live tokenization with color-coded tokens; show vocab size, token count, compression ratio
- **Reference**: Karpathy's tiktokenizer; build our own simpler version

#### 11. **Positional Encoding Visualizer**
- **Lecture**: 14 (Transformer)
- **Concept**: Visualize sinusoidal PE as a heatmap; show how positions differ; compare to learned PE
- **Interactive**: Slider for sequence length and embedding dim; heatmap of PE values; cosine similarity between positions

### Priority 4 — Generative & Advanced (Lectures 19-24)

#### 12. **VAE Latent Space Explorer**
- **Lecture**: 20 (new numbering: Autoencoders & VAEs)
- **Concept**: Navigate 2D latent space; see generated images at each point; interpolate between two images
- **Interactive**: 2D scatter plot of latent space (MNIST); click to generate; drag between two points to interpolate
- **Pre-trained model**: Ship a tiny VAE model as ONNX for in-browser inference

#### 13. **GAN Training Dynamics**
- **Lecture**: 21 (GANs)
- **Concept**: Watch generator and discriminator fight; see mode collapse happen
- **Interactive**: 2D toy GAN (fit a mixture of Gaussians); step through training; watch generated samples converge (or collapse)

#### 14. **Diffusion Denoising Stepper**
- **Lecture**: 22-23 (Diffusion models)
- **Concept**: Step through the reverse diffusion process one timestep at a time
- **Interactive**: Start from noise; step backward; watch image emerge; slider for number of steps (1, 10, 50, 1000)
- **Complement**: to existing Text Diffusion and MNIST Diffusion Lab

#### 15. **KV-Cache Visualizer**
- **Lecture**: 24 (Efficient DL)
- **Concept**: Show how KV-cache avoids recomputation during autoregressive generation
- **Interactive**: Side-by-side: with cache (only new token computed) vs without (full sequence recomputed); show FLOPs counter

---

## Build Order

**Phase 1** (before semester): #1 Softmax, #2 Vanishing Gradient, #6 Convolution — highest lecture-impact, moderate complexity

**Phase 2** (first few weeks): #10 Tokenizer, #9 LSTM Gates, #11 Positional Encoding — needed for mid-semester lectures

**Phase 3** (ongoing): #12 VAE, #13 GAN, #14 Diffusion, #15 KV-Cache — needed for later lectures

**Already done**: Autograd, UAT, Optimizer Race, Attention, Text Diffusion, RAG, MVN, MNIST Diffusion

---

## Technical Notes

- **Framework**: Same as ~/git/interactive/ — vanilla HTML/JS/Canvas + KaTeX for math
- **No server needed**: All run client-side in browser (except Streamlit apps)
- **Small models in-browser**: Use ONNX.js or TensorFlow.js for pre-trained model inference (VAE, tiny diffusion)
- **Repo**: Build in ~/git/interactive/ as new articles, or in ~/git/dl-teaching/interactives/ if we want them course-specific
