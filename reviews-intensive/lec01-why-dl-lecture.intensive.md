Excellent. This is a perfect task. Here is a concrete rewrite plan for Lecture 1, focusing on the most critical slides that introduce new, complex deep learning concepts to an audience with only a basic ML background.

---

## SLIDE · "Without σ · depth gives nothing"

**CURRENT PROBLEM**
The slide presents the mathematical collapse `W_2 (W_1 x) = (W_2 W_1) x` as a statement of fact. For students who aren't fluent in linear algebra, this isn't obvious; it feels abstract and lacks a concrete takeaway.

**INSERT BEFORE**
*   **Slide Title:** Why non-linearities? The Magnifying Glass Analogy
*   **Content:**
    *   Imagine a linear layer is like a magnifying glass. It scales and rotates the input. `Output = 2 * Input`.
    *   What happens if you stack two magnifying glasses? `Output = 2 * (2 * Input) = 4 * Input`.
    *   You just get a *stronger* single magnifying glass. You haven't fundamentally changed what you can do. Stacking linear layers is the same — you just get one, more complex, linear layer.
    *   A non-linearity (like ReLU) is like a prism. It *bends* the light. Stacking a magnifier and a prism lets you do something new that no single magnifier could ever do.

**REWRITE**
*   **Title:** Let's Prove It: Stacking Linear Layers Collapses
*   **Content:**
    *   Let's use a tiny 2-layer network with no non-linearity.
    *   **Layer 1:** Hidden state `h` is a linear function of input `x`. Let's say `x` has 2 features, and our hidden layer has 2 neurons.
        *   $h = W_1 x + b_1$
    *   **Layer 2:** Output `y` is a linear function of the hidden state `h`.
        *   $y = W_2 h + b_2$
    *   **Now, let's substitute `h` into the second equation:**
        *   $y = W_2 (W_1 x + b_1) + b_2$
    *   **Distribute `W_2`:**
        *   $y = (W_2 W_1) x + (W_2 b_1 + b_2)$
    *   **Let's define two new variables:**
        *   $W_{effective} = W_2 W_1$ (This is just another matrix!)
        *   $b_{effective} = W_2 b_1 + b_2$ (This is just another vector!)
    *   **So the entire 2-layer network is equivalent to:**
        *   $y = W_{effective} x + b_{effective}$
    *   This is a *single* linear layer. The depth was useless. We could have just learned `W_effective` and `b_effective` from the start.

**INSERT AFTER**
*   **Title:** Worked Numeric Example: The Collapse
*   **Content:**
    *   Input: $x = \begin{bmatrix} 2 \\ 3 \end{bmatrix}$
    *   Layer 1: $W_1 = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$, $b_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$
    *   Layer 2: $W_2 = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$, $b_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$
    *   **Forward pass (step-by-step):**
        *   $h = W_1 x + b_1 = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 2 \\ 3 \end{bmatrix} + \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 2 \\ 3 \end{bmatrix} + \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 3 \\ 3 \end{bmatrix}$
        *   $y = W_2 h + b_2 = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix} \begin{bmatrix} 3 \\ 3 \end{bmatrix} + \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} 9 \\ 9 \end{bmatrix} + \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} 9 \\ 10 \end{bmatrix}$
    *   **Equivalent single layer:**
        *   $W_{eff} = W_2 W_1 = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$
        *   $b_{eff} = W_2 b_1 + b_2 = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} + \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} 2 \\ 1 \end{bmatrix} + \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} 2 \\ 2 \end{bmatrix}$
    *   **Let's check:** $y = W_{eff} x + b_{eff} = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix} \begin{bmatrix} 2 \\ 3 \end{bmatrix} + \begin{bmatrix} 2 \\ 2 \end{bmatrix} = \begin{bmatrix} 7 \\ 8 \end{bmatrix} + \begin{bmatrix} 2 \\ 2 \end{bmatrix} = \begin{bmatrix} 9 \\ 10 \end{bmatrix}$. **It's the exact same result!**

**FIGURE**
A diagram showing two matrix boxes (`W1`, `W2`) with an input `x` flowing through them. An arrow shows them "merging" or "collapsing" into a single, larger matrix box labeled `W_effective = W2 * W1`. This visually reinforces the idea of collapse.

---

## SLIDE · "Cross-entropy from MLE"

**CURRENT PROBLEM**
This slide assumes students are comfortable with the Maximum Likelihood Estimation (MLE) derivation for multiclass classification and the one-hot vector notation. It's a "trust me, this is the loss function" moment.

**INSERT BEFORE**
*   **Slide Title:** Intuition: Scoring a Weather Forecaster
*   **Content:**
    *   Imagine a forecaster predicts: {70% rain, 20% cloudy, 10% sun}.
    *   **Scenario A: It rains.** They put high probability on the right outcome. Good forecast!
    *   **Scenario B: It's sunny.** They put very low probability on the right outcome. Bad forecast!
    *   Cross-entropy is a way to formalize this. It gives a *high penalty (loss)* for being confident and wrong, and a *low penalty* for being confident and right. We want to minimize this penalty.

**REWRITE**
*   **Title:** Deriving Cross-Entropy from a Single Goal
*   **Content:**
    *   **Our only goal:** Maximize the probability of the *correct* class.
    *   Let's say our network outputs probabilities $\hat{y} = [\hat{y}_1, \hat{y}_2, ..., \hat{y}_K]$ for K classes.
    *   For one training example $(x, y)$, where the true class is $c$. We want to maximize $P(y=c | x)$, which is just our model's output $\hat{y}_c$.
    *   **The log trick:** Working with products is hard. Working with sums is easy. `log` turns products into sums. Maximizing $\hat{y}_c$ is the same as maximizing $\log(\hat{y}_c)$.
    *   **The optimization trick:** By convention, optimizers *minimize* a loss function. So, we flip the sign. Maximizing $\log(\hat{y}_c)$ is the same as *minimizing* $-\log(\hat{y}_c)$.
    *   **This is our loss for one example:** $\mathcal{L} = -\log(\hat{y}_c)$.
    *   **Using one-hot vectors:** How do we write this for any class? Let's make the true label $y$ a vector, e.g., for class 2 in a 3-class problem: $y = [0, 1, 0]$.
    *   We can write the loss as a sum: $\mathcal{L} = - \sum_{k=1}^K y_k \log(\hat{y}_k)$.
        *   When $k$ is the true class, $y_k=1$, and we get $-1 \cdot \log(\hat{y}_k)$.
        *   For all other classes, $y_k=0$, and the term is $0$.
    *   This is the cross-entropy loss. It falls directly out of wanting to maximize the probability of the correct class.

**INSERT AFTER**
*   **Title:** Worked Numeric Example: Cross-Entropy
*   **Content:**
    *   Model outputs (after softmax): $\hat{y} = [0.1, 0.7, 0.2]$ (predicts class 1 is most likely).
    *   **Case 1: True label is class 1.**
        *   $y = [0, 1, 0]$
        *   Loss = $-(0 \cdot \log(0.1) + 1 \cdot \log(0.7) + 0 \cdot \log(0.2)) = -\log(0.7) \approx 0.36$. (Low loss, good prediction).
    *   **Case 2: True label is class 0.**
        *   $y = [1, 0, 0]$
        *   Loss = $-(1 \cdot \log(0.1) + 0 \cdot \log(0.7) + 0 \cdot \log(0.2)) = -\log(0.1) \approx 2.30$. (High loss, bad prediction).
    *   The loss correctly penalizes the model when it's wrong.

**FIGURE**
A diagram with two vectors. Top vector is the one-hot true label `y = [0, 0, 1, 0]`. Bottom vector is the softmax output `y_hat = [0.1, 0.2, 0.6, 0.1]`. An arrow points from the `1` in the top vector down to the `0.6` in the bottom vector, highlighting that this is the only value that matters for the loss calculation. The text says "Loss = -log(0.6)".

---

## SLIDE · "The elegant softmax + CE gradient"

**CURRENT PROBLEM**
This is the biggest jump in the lecture. The result is presented as magic. Students who haven't done this derivation before have no reason to believe it and no intuition for why it's so simple.

**INSERT BEFORE**
*   **Slide Title:** Intuition: The Push-Pull Gradient
*   **Content:**
    *   Our loss is $\mathcal{L}$. Our pre-softmax scores are logits, $z = [z_1, z_2, ..., z_K]$.
    *   We want to compute $\frac{\partial \mathcal{L}}{\partial z_k}$ for each logit $z_k$. This tells us: "how should I change $z_k$ to lower the loss?"
    *   **If $k$ is the correct class:** We want its probability to be 1. So we need to **increase** its logit $z_k$. The gradient should push it up.
    *   **If $k$ is a wrong class:** We want its probability to be 0. So we need to **decrease** its logit $z_k$. The gradient should push it down.
    *   The final formula, $\hat{y}_k - y_k$, does exactly this!

**REWRITE**
*   **Title:** Let's Compute the Softmax+CE Gradient
*   **Content:**
    *   Our loss is $\mathcal{L} = -\sum_j y_j \log(\hat{y}_j)$, and softmax is $\hat{y}_j = \frac{e^{z_j}}{\sum_i e^{z_i}}$.
    *   We need $\frac{\partial \mathcal{L}}{\partial z_k}$. Let's use the chain rule: $\frac{\partial \mathcal{L}}{\partial z_k} = \sum_{j=1}^K \frac{\partial \mathcal{L}}{\partial \hat{y}_j} \frac{\partial \hat{y}_j}{\partial z_k}$.
    *   **Part 1: $\frac{\partial \mathcal{L}}{\partial \hat{y}_j}$**
        *   This is simple: $\frac{\partial}{\partial \hat{y}_j} (-\sum_i y_i \log(\hat{y}_i)) = - \frac{y_j}{\hat{y}_j}$.
    *   **Part 2: $\frac{\partial \hat{y}_j}{\partial z_k}$** (This is the tricky part)
        *   We need the quotient rule. Let's consider two cases.
        *   **Case A: $j = k$ (The derivative w.r.t. its own logit)**
            *   $\frac{\partial \hat{y}_k}{\partial z_k} = \frac{(e^{z_k})' (\sum e^{z_i}) - (e^{z_k}) (\sum e^{z_i})'}{(\sum e^{z_i})^2} = \frac{e^{z_k} (\sum e^{z_i}) - e^{z_k} (e^{z_k})}{(\sum e^{z_i})^2}$
            *   $= \frac{e^{z_k}}{\sum e^{z_i}} \left( 1 - \frac{e^{z_k}}{\sum e^{z_i}} \right) = \hat{y}_k (1 - \hat{y}_k)$.
        *   **Case B: $j \neq k$ (The derivative w.r.t. another logit)**
            *   $\frac{\partial \hat{y}_j}{\partial z_k} = \frac{(e^{z_j})' (\sum e^{z_i}) - (e^{z_j}) (\sum e^{z_i})'}{(\sum e^{z_i})^2} = \frac{0 - e^{z_j}(e^{z_k})}{(\sum e^{z_i})^2}$
            *   $= - \frac{e^{z_j}}{\sum e^{z_i}} \frac{e^{z_k}}{\sum e^{z_i}} = -\hat{y}_j \hat{y}_k$.
    *   **Putting it all together:**
        *   $\frac{\partial \mathcal{L}}{\partial z_k} = \underbrace{(- \frac{y_k}{\hat{y}_k})}_{\frac{\partial \mathcal{L}}{\partial \hat{y}_k}} \underbrace{(\hat{y}_k(1-\hat{y}_k))}_{\frac{\partial \hat{y}_k}{\partial z_k}} + \sum_{j \neq k} \underbrace{(- \frac{y_j}{\hat{y}_j})}_{\frac{\partial \mathcal{L}}{\partial \hat{y}_j}} \underbrace{(-\hat{y}_j \hat{y}_k)}_{\frac{\partial \hat{y}_j}{\partial z_k}}$
        *   $= -y_k(1-\hat{y}_k) + \sum_{j \neq k} y_j \hat{y}_k = -y_k + y_k \hat{y}_k + \hat{y}_k \sum_{j \neq k} y_j$
        *   $= -y_k + \hat{y}_k (y_k + \sum_{j \neq k} y_j) = -y_k + \hat{y}_k (\sum_j y_j)$
        *   Since $\sum_j y_j = 1$ (for one-hot vectors), we get: $\frac{\partial \mathcal{L}}{\partial z_k} = \hat{y}_k - y_k$.

**INSERT AFTER**
*   **Title:** Worked Numeric Example: The Gradient
*   **Content:**
    *   Logits $z = [2.0, 1.0, 0.1]$.
    *   Softmax output $\hat{y} = [0.66, 0.24, 0.10]$.
    *   True label is class 0, so one-hot $y = [1, 0, 0]$.
    *   **Let's compute the gradient $\frac{\partial \mathcal{L}}{\partial z}$ using our simple formula:**
    *   $\frac{\partial \mathcal{L}}{\partial z} = \hat{y} - y = [0.66, 0.24, 0.10] - [1, 0, 0] = [-0.34, 0.24, 0.10]$.
    *   **Interpretation:**
        *   To lower the loss, we need to *decrease* $z_0$ (since the gradient is negative). Wait, that's wrong. SGD step is `z_new = z_old - lr * grad`. So `z_0` will **increase**. Perfect.
        *   We need to *increase* $z_1$ and $z_2$. Wait, that's wrong too. SGD will **decrease** them.
        *   The gradient tells the optimizer: "Increase the score for the correct class (z0), and decrease the scores for the incorrect classes (z1, z2)."

**FIGURE**
A three-panel diagram. Panel 1: model outputs logits `z`. Panel 2: `softmax` turns them into probabilities `y_hat`. Panel 3: `cross-entropy` compares `y_hat` to `y` to get loss `L`. A big red arrow labeled `dL/dz = y_hat - y` points backwards from the loss `L` to the logits `z`, showing this is the key signal for learning.

---

## SLIDE · "The local-gradient rule · three lines"

**CURRENT PROBLEM**
Presents three key backpropagation rules for a linear layer as fact. These matrix-vector derivatives are non-obvious and intimidating.

**INSERT BEFORE**
*   **Slide Title:** Backprop's Job: The Blame Distributor
*   **Content:**
    *   Imagine a linear layer $z = Wx + b$. The next layer sends us a message: "Hey, the error signal on your output `z` is $\boldsymbol{\delta}$." ($\boldsymbol{\delta} = \frac{\partial \mathcal{L}}{\partial z}$).
    *   Our job is to figure out who to blame for this error.
        *   How much was `W`'s fault? That's $\frac{\partial \mathcal{L}}{\partial W}$.
        *   How much was `b`'s fault? That's $\frac{\partial \mathcal{L}}{\partial b}$.
        *   How much should we blame the *previous* layer's output `x`? That's $\frac{\partial \mathcal{L}}{\partial x}$. This is the signal we pass backward.

**REWRITE**
*   **Title:** Deriving the Gradient for a Linear Layer
*   **Content:**
    *   We know $\boldsymbol{\delta} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}}$. We want $\frac{\partial \mathcal{L}}{\partial \mathbf{W}}$, $\frac{\partial \mathcal{L}}{\partial \mathbf{b}}$, and $\frac{\partial \mathcal{L}}{\partial \mathbf{x}}$.
    *   Let's write it out for one element $z_i$: $z_i = \sum_j W_{ij} x_j + b_i$.
    *   **1. Bias gradient $\frac{\partial \mathcal{L}}{\partial b_i}$ (the easiest):**
        *   Chain rule: $\frac{\partial \mathcal{L}}{\partial b_i} = \frac{\partial \mathcal{L}}{\partial z_i} \frac{\partial z_i}{\partial b_i}$.
        *   We know $\frac{\partial \mathcal{L}}{\partial z_i} = \delta_i$.
        *   $\frac{\partial z_i}{\partial b_i} = \frac{\partial}{\partial b_i} (\sum_j W_{ij} x_j + b_i) = 1$.
        *   So, $\frac{\partial \mathcal{L}}{\partial b_i} = \delta_i$. In vector form, $\frac{\partial \mathcal{L}}{\partial \mathbf{b}} = \boldsymbol{\delta}$.
    *   **2. Weight gradient $\frac{\partial \mathcal{L}}{\partial W_{ij}}$:**
        *   Chain rule: $\frac{\partial \mathcal{L}}{\partial W_{ij}} = \frac{\partial \mathcal{L}}{\partial z_i} \frac{\partial z_i}{\partial W_{ij}}$.
        *   $\frac{\partial z_i}{\partial W_{ij}} = \frac{\partial}{\partial W_{ij}} (\sum_k W_{ik} x_k + b_i) = x_j$.
        *   So, $\frac{\partial \mathcal{L}}{\partial W_{ij}} = \delta_i x_j$. This is an outer product. In matrix form, $\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \boldsymbol{\delta} \mathbf{x}^\top$.
    *   **3. Input gradient $\frac{\partial \mathcal{L}}{\partial x_j}$ (to pass back):**
        *   The input $x_j$ affects *every* output $z_i$ through the weight $W_{ij}$.
        *   Chain rule (with a sum): $\frac{\partial \mathcal{L}}{\partial x_j} = \sum_i \frac{\partial \mathcal{L}}{\partial z_i} \frac{\partial z_i}{\partial x_j}$.
        *   $\frac{\partial z_i}{\partial x_j} = \frac{\partial}{\partial x_j} (\sum_k W_{ik} x_k + b_i) = W_{ij}$.
        *   So, $\frac{\partial \mathcal{L}}{\partial x_j} = \sum_i \delta_i W_{ij}$. This is a matrix-vector product. In vector form: $\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \mathbf{W}^\top \boldsymbol{\delta}$.

**INSERT AFTER**
*   **Title:** Worked Numeric Example: Linear Layer Backward Pass
*   **Content:**
    *   Let's use a 2-input, 2-output layer.
    *   Input: $x = \begin{bmatrix} 2 \\ 3 \end{bmatrix}$. Weights: $W = \begin{bmatrix} 0.1 & 0.2 \\ 0.3 & 0.4 \end{bmatrix}$.
    *   Assume the upstream gradient (blame) from the next layer is $\delta = \begin{bmatrix} 0.5 \\ -0.1 \end{bmatrix}$.
    *   **Let's compute the gradients using our new rules:**
        *   **Weight gradient:** $\frac{\partial \mathcal{L}}{\partial W} = \delta x^\top = \begin{bmatrix} 0.5 \\ -0.1 \end{bmatrix} \begin{bmatrix} 2 & 3 \end{bmatrix} = \begin{bmatrix} 1.0 & 1.5 \\ -0.2 & -0.3 \end{bmatrix}$. These are the updates for `W`.
        *   **Bias gradient:** $\frac{\partial \mathcal{L}}{\partial b} = \delta = \begin{bmatrix} 0.5 \\ -0.1 \end{bmatrix}$. These are the updates for `b`.
        *   **Input gradient:** $\frac{\partial \mathcal{L}}{\partial x} = W^\top \delta = \begin{bmatrix} 0.1 & 0.3 \\ 0.2 & 0.4 \end{bmatrix} \begin{bmatrix} 0.5 \\ -0.1 \end{bmatrix} = \begin{bmatrix} 0.05 - 0.03 \\ 0.10 - 0.04 \end{bmatrix} = \begin{bmatrix} 0.02 \\ 0.06 \end{bmatrix}$. This is the "blame" we pass back to the previous layer.

**FIGURE**
A diagram of a single linear layer node. An arrow labeled $\delta$ comes in from the right. Three arrows go out: one points to the `W` matrix labeled `δ x^T`, one points to the `b` vector labeled `δ`, and one goes out to the left (backwards) labeled `W^T δ`.

---

## SLIDE · "Depth has a cost · vanishing gradients"

**CURRENT PROBLEM**
The slide uses intimidating product notation ($\prod$) and states that the sigmoid derivative is $\le 0.25$ without showing why. This makes the core problem feel abstract and unmotivated.

**(e) SETUP:** Yes, a smaller-scope example is needed first. We should build up from a 2-layer network before showing the general product formula.

**INSERT BEFORE**
*   **Slide Title:** Backprop's Telephone Game
*   **Content:**
    *   The gradient signal starts at the end of the network (the loss).
    *   It has to travel backwards, layer by layer, to reach the beginning.
    *   Each layer "whispers" the message to the one before it by multiplying the incoming signal by its own local gradient.
    *   What happens if every layer whispers very quietly (multiplies by a small number)? The message becomes gibberish by the time it reaches the start. The first layers get no useful signal.

**REWRITE**
*   **Title:** The Math of the Fading Signal
*   **Content:**
    *   Let's look at the gradient for the first weight matrix, $W_1$, in a deep network.
    *   For a 3-layer net: $h_1 = \sigma(z_1)$, $h_2 = \sigma(z_2)$, $\mathcal{L} = f(h_2)$.
    *   $\frac{\partial \mathcal{L}}{\partial W_1} \propto \frac{\partial \mathcal{L}}{\partial z_2} \frac{\partial z_2}{\partial h_1} \frac{\partial h_1}{\partial z_1} \frac{\partial z_1}{\partial W_1}$.
    *   The key terms are the "chain" parts: $\frac{\partial z_2}{\partial h_1} \frac{\partial h_1}{\partial z_1}$.
    *   $\frac{\partial z_2}{\partial h_1} = W_2$. $\frac{\partial h_1}{\partial z_1} = \sigma'(z_1)$ (derivative of activation).
    *   So, the gradient for $W_1$ is scaled by $W_2 \cdot \sigma'(z_1)$.
    *   **The problem is $\sigma'(z)$ for sigmoid:**
        *   Let's compute it: $\sigma(z) = (1+e^{-z})^{-1}$.
        *   $\sigma'(z) = -(1+e^{-z})^{-2} \cdot (-e^{-z}) = \frac{e^{-z}}{(1+e^{-z})^2} = \frac{1}{1+e^{-z}} \frac{e^{-z}}{1+e^{-z}} = \sigma(z)(1-\sigma(z))$.
        *   This function has a maximum value of **0.25** (when z=0, $\sigma(z)=0.5$).
    *   So each layer multiplies the backward-flowing gradient by its weights AND a number that's at most 0.25.
    *   For an L-layer network, the gradient for $W_1$ gets scaled by a product of L-1 of these terms: $\frac{\partial \mathcal{L}}{\partial \mathbf{W}_1} \propto (W_L \sigma'_{L-1}) \cdots (W_2 \sigma'_1) \cdots$. If weights are not large, this shrinks fast!

**INSERT AFTER**
*   **Title:** Worked Numeric Example: The Gradient Vanishes
*   **Content:**
    *   Let's imagine a simple 5-layer network. For simplicity, assume all weights $W_l$ are 1.
    *   Let the gradient signal arriving at the last layer be 1.0.
    *   Assume all neurons are in the "worst case" for gradients: their inputs `z` are far from zero, so $\sigma'(z) \approx 0.1$.
    *   Gradient signal at Layer 4: $1.0 \times (W_5 \cdot \sigma'_4) \approx 1.0 \times (1 \cdot 0.1) = 0.1$.
    *   Gradient signal at Layer 3: $0.1 \times (W_4 \cdot \sigma'_3) \approx 0.1 \times (1 \cdot 0.1) = 0.01$.
    *   Gradient signal at Layer 2: $0.01 \times (W_3 \cdot \sigma'_2) \approx 0.01 \times (1 \cdot 0.1) = 0.001$.
    *   Gradient signal at Layer 1: $0.001 \times (W_2 \cdot \sigma'_1) \approx 0.001 \times (1 \cdot 0.1) = 0.0001$.
    *   The first layer's weights will be updated by a tiny, almost zero amount. Learning stalls.

**FIGURE**
A two-panel plot. Left panel: The sigmoid function $\sigma(z)$. Right panel: Its derivative $\sigma'(z)$, clearly showing a bell-like shape that peaks at 0.25 and quickly goes to zero elsewhere. An annotation on the right plot says "Max value = 0.25. The gradient signal is always shrunk by this factor."