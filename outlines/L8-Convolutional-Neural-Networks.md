---
title: "Lecture 8: Convolutional Neural Networks"
source: "https://chatgpt.com/share/6a50728a-9720-83ee-a33d-5f2e3ff744fe"
status: "Imported from the finalized shared-course discussion"
---

# Lecture 8: Convolutional Neural Networks

## Core story

\[
\boxed{
\text{Images have spatial structure. CNNs exploit locality, weight sharing and hierarchy.}
}
\]

Classic CNNs go back to LeNet-style document recognition; AlexNet then showed the impact of large CNNs on ImageNet-scale classification.

---

## Slide-level outline

### Part I — Why MLPs are bad for images

1. **Title**  
   *Convolutional Neural Networks*

2. **Image as tensor**
   \[
   X\in\mathbb R^{H\times W\times C}
   \]

3. **Flattening destroys structure**
   \[
   224\times224\times3=150{,}528
   \]

4. **Fully connected parameter explosion**
   If hidden layer has \(1000\) units:
   \[
   150{,}528\times1000\approx1.5\times10^8
   \]

5. **Three image priors**
   \[
   \boxed{\text{locality}}
   \]
   \[
   \boxed{\text{translation equivariance}}
   \]
   \[
   \boxed{\text{hierarchical features}}
   \]

---

### Part II — Convolution operation

6. **1D convolution intuition**  
   sliding weighted sum.

7. **2D convolution**
   \[
   Y[i,j]=\sum_{u,v}K[u,v]X[i+u,j+v]
   \]

8. **Worked numeric example**
   Image patch:
   \[
   \begin{bmatrix}
   1&2&0\\
   0&1&3\\
   2&1&0
   \end{bmatrix}
   \]
   Kernel:
   \[
   \begin{bmatrix}
   1&0&-1\\
   1&0&-1\\
   1&0&-1
   \end{bmatrix}
   \]
   Output:
   \[
   (1+0+2)-(0+3+0)=0
   \]

9. **Kernel as feature detector**  
   edges, corners, textures.

10. **Cross-correlation vs convolution**  
   DL libraries usually use cross-correlation, but call it convolution.

11. **Multiple kernels produce channels**
   \[
   X:H\times W\times C_{\text{in}}
   \]
   \[
   W:k_h\times k_w\times C_{\text{in}}\times C_{\text{out}}
   \]
   \[
   Y:H'\times W'\times C_{\text{out}}
   \]

12. **Parameter count**
   \[
   k_hk_wC_{\text{in}}C_{\text{out}}+C_{\text{out}}
   \]

13. **Worked parameter count**
   \[
   3\times3,\ C_{\text{in}}=64,\ C_{\text{out}}=128
   \]
   \[
   3\cdot3\cdot64\cdot128+128=73{,}856
   \]

14. **Interactive 1: convolution playground**  
   `conv-kernel-playground.html`  
   Draw image/kernel, see feature map.

---

### Part III — Padding, stride, output size

15. **Padding**
   \[
   p=0:\text{valid}
   \]
   \[
   p>0:\text{same-ish}
   \]

16. **Stride**
   \[
   s=1,2,\dots
   \]

17. **Output size formula**
   \[
   H_{\text{out}}
   =
   \left\lfloor
   \frac{H+2p-k}{s}
   \right\rfloor+1
   \]

18. **Worked example**
   \[
   H=32,\ k=5,\ p=2,\ s=1
   \]
   \[
   H_{\text{out}}=32
   \]

19. **Another example**
   \[
   H=32,\ k=3,\ p=1,\ s=2
   \]
   \[
   H_{\text{out}}=16
   \]

20. **Interactive 2: padding/stride visualizer**  
   `padding-stride-output.html`

---

### Part IV — Equivariance and pooling

21. **Translation equivariance**
   If input shifts, feature map shifts:
   \[
   f(T_\Delta x)=T_\Delta f(x)
   \]

22. **Not invariance**
   CNN layers are equivariant, not automatically invariant.

23. **Pooling**
   Max pooling:
   \[
   y=\max_{(i,j)\in\mathcal R}x_{ij}
   \]

24. **Average pooling**
   \[
   y=\frac1{|\mathcal R|}\sum_{(i,j)\in\mathcal R}x_{ij}
   \]

25. **Pooling effects**
   - downsampling;
   - larger receptive field;
   - local robustness;
   - information loss.

26. **Global average pooling**
   \[
   h_c=\frac1{HW}\sum_{i,j}X_{ijc}
   \]

27. **Interactive 3: pooling**  
   `pooling-visualizer.html`

---

### Part V — Receptive field

28. **Receptive field definition**  
   Region of input affecting one output unit.

29. **Stacking \(3\times3\) convolutions**
   Two \(3\times3\) layers produce effective \(5\times5\) receptive field.

30. **Why stack small kernels?**
   Two \(3\times3\) convs:
   \[
   2\cdot 3\cdot3 C^2=18C^2
   \]
   One \(5\times5\):
   \[
   25C^2
   \]
   More nonlinearity, fewer parameters.

31. **Receptive field recursion**
   For stride \(1\):
   \[
   r_{\ell}=r_{\ell-1}+k_\ell-1
   \]

32. **Worked receptive field**
   Three \(3\times3\) convs:
   \[
   r=1+3(3-1)=7
   \]

33. **Interactive 4: receptive field explorer**  
   `receptive-field.html`

---

### Part VI — CNN architecture pattern

34. **Typical CNN**
   ```text
   image → conv → activation → conv → pooling → ... → classifier
   ```

35. **Feature hierarchy**
   - early: edges;
   - middle: textures/parts;
   - late: objects/semantics.

36. **Classification head**
   \[
   X\rightarrow \text{global average pool}\rightarrow \text{linear}\rightarrow \text{softmax}
   \]

37. **Loss**
   \[
   L=-\log p_y
   \]

38. **Worked tiny CNN**
   Input:
   \[
   32\times32\times3
   \]
   Conv:
   \[
   3\times3,\ 16\text{ filters},\ p=1
   \Rightarrow32\times32\times16
   \]
   Pool:
   \[
   2\times2,\ s=2
   \Rightarrow16\times16\times16
   \]
   Conv:
   \[
   3\times3,\ 32\text{ filters}
   \Rightarrow16\times16\times32
   \]
   Global average pool:
   \[
   32
   \]
   Linear:
   \[
   10
   \]

39. **Parameter count for tiny CNN**

40. **CNN blocks**
   \[
   \text{Conv}\rightarrow\text{Norm}\rightarrow\text{Activation}
   \]

41. **Why normalization and residuals reappear**
   Deep CNNs use the previous lecture’s ideas.

42. **Historical anchors**
   LeNet for document recognition; AlexNet for ImageNet-scale CNN classification.

---

### Part VII — Backprop through convolution

43. **Convolution is a linear operator**
   \[
   y=K*x
   \]

44. **Gradients**
   - wrt kernel: correlate input with output gradient;
   - wrt input: spread output gradient through kernel.

45. **Intuition slide only**
   Do not derive im2col fully in main lecture.

46. **Optional: convolution as matrix multiplication**
   \[
   y=\mathcal Kx
   \]

47. **Interactive 5: conv backprop intuition**
   `conv-backprop.html`

---

### Part VIII — Practical CNN training

48. **Data augmentation for images**
   - crop;
   - flip;
   - color jitter;
   - normalization.

49. **Input normalization**
   \[
   x'=\frac{x-\mu}{\sigma}
   \]

50. **Transfer learning**
   \[
   \text{pretrained backbone}+\text{new head}
   \]

51. **Common failure modes**
   - wrong tensor shape;
   - channel order;
   - too much downsampling;
   - no augmentation;
   - train/eval mode bug.

52. **Summary table**

| Concept | Purpose |
|---|---|
| convolution | local feature extraction |
| weight sharing | fewer parameters |
| channels | multiple learned features |
| padding/stride | control resolution |
| pooling | downsample/robustness |
| receptive field | context size |

53. **Next lecture**
   Classification says:
   \[
   \text{What is in the image?}
   \]
   Next:
   \[
   \text{Where is it? Which pixels?}
   \]

---

## CNN interactives

1. `conv-kernel-playground.html`
2. `padding-stride-output.html`
3. `pooling-visualizer.html`
4. `receptive-field.html`
5. `cnn-shape-calculator.html`
6. `conv-backprop.html`

## CNN notebooks

1. `01_convolution_from_scratch.ipynb`
2. `02_padding_stride_shapes.ipynb`
3. `03_cnn_parameter_counts.ipynb`
4. `04_receptive_field.ipynb`
5. `05_train_tiny_cnn_cifar_mnist.ipynb`
6. `06_feature_maps_visualization.ipynb`

---
