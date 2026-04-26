Of course. Here is a concrete rewrite plan for the lecture, following your specified format.

**Overall Diagnosis & Structural Recommendation**

The lecture's core message—that common loss functions are just Negative Log-Likelihood (NLL) under different probabilistic assumptions—is excellent. However, the flow is confusing. It introduces advanced concepts like MAP and numerical tricks *before* deriving the main losses (MSE, cross-entropy). The linear algebra section also feels tacked on and breaks the narrative.

**Recommended New Flow:**

1.  **Part 1: The Building Blocks.** (Distributions: Bernoulli, Categorical, Normal). Keep this as is.
2.  **Part 2: The Core Idea: Likelihood.** Introduce Likelihood, Log-Likelihood, and Maximum Likelihood Estimation (MLE).
3.  **Part 3: Deriving The Losses.** Use MLE to derive MSE and Cross-Entropy. This is the climax of the lecture and should come early.
4.  **Part 4: Practical Considerations.** Now discuss numerical stability (log-sum-exp), connections to information theory (KL), and MAP/regularization. This is the "advanced topics/connections" section.
5.  **Appendix/Recap: Linear Algebra & Gradients.** Move the linear algebra refresher to the end or a separate "Lecture 0.5" document. It's prerequisite material, not part of the main probabilistic story.

Here is the detailed plan for specific slides based on this improved flow.

---

## SLIDE · "Multivariate normal · the d-dimensional bell"

**CURRENT PROBLEM**
This is a huge conceptual leap from the 1D Gaussian. The matrix form, inverse covariance, and the term "Mahalanobis distance" are dense jargon without intuition.

**INSERT BEFORE**
*Intuition Slide: Stretching a Bell Curve*

"Imagine a 2D bell curve, like a hill. If it's perfectly round, moving one step north is the same as moving one step east. But what if we stretch the hill, making it an oval-shaped ridge? Now, moving one step along the narrow ridge is very different from moving one step off the side. The **covariance matrix** is just a way of describing the stretch and rotation of this multidimensional hill."

**REWRITE**
Let's build up from a simple 2D case. Imagine we have two independent features, $y_1$ and $y_2$. For instance, height and wingspan of a bird, assuming they are uncorrelated.

The probability is just the product of two 1D Gaussians:
$p(y_1, y_2) = p(y_1) p(y_2) = \frac{1}{\sqrt{2\pi \sigma_1^2}} \exp(-\frac{(y_1 - \mu_1)^2}{2 \sigma_1^2}) \cdot \frac{1}{\sqrt{2\pi \sigma_2^2}} \exp(-\frac{(y_2 - \mu_2)^2}{2 \sigma_2^2})$

This corresponds to a covariance matrix $\Sigma$ that is **diagonal**: $\Sigma = \begin{bmatrix} \sigma_1^2 & 0 \\ 0 & \sigma_2^2 \end{bmatrix}$. Its inverse is also simple: $\Sigma^{-1} = \begin{bmatrix} 1/\sigma_1^2 & 0 \\ 0 & 1/\sigma_2^2 \end{bmatrix}$.

Let's write the exponent in matrix form for $\mathbf{y} = [y_1, y_2]^\top$ and $\boldsymbol\mu = [\mu_1, \mu_2]^\top$:
$(\mathbf{y} - \boldsymbol\mu)^\top \Sigma^{-1} (\mathbf{y} - \boldsymbol\mu) = \begin{bmatrix} y_1 - \mu_1 & y_2 - \mu_2 \end{bmatrix} \begin{bmatrix} 1/\sigma_1^2 & 0 \\ 0 & 1/\sigma_2^2 \end{bmatrix} \begin{bmatrix} y_1 - \mu_1 \\ y_2 - \mu_2 \end{bmatrix}$
$= \begin{bmatrix} y_1 - \mu_1 & y_2 - \mu_2 \end{bmatrix} \begin{bmatrix} (y_1 - \mu_1)/\sigma_1^2 \\ (y_2 - \mu_2)/\sigma_2^2 \end{bmatrix}$
$= \frac{(y_1 - \mu_1)^2}{\sigma_1^2} + \frac{(y_2 - \mu_2)^2}{\sigma_2^2}$

This is just the sum of two squared distances, each scaled by its own variance. The general formula you see is for when the variables are correlated (the oval is rotated), and $\Sigma$ has non-zero off-diagonals.

**INSERT AFTER**
*Worked Numeric Example*

Let $\boldsymbol\mu = [0, 0]^\top$ and consider the point $\mathbf{y} = [2, 1]^\top$.

1.  **Isotropic (round) covariance**: $\Sigma = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$. The squared distance in the exponent is simply $2^2 + 1^2 = 5$.
2.  **Stretched covariance**: $\Sigma = \begin{bmatrix} 4 & 0 \\ 0 & 1 \end{bmatrix}$ ($\sigma_1=2, \sigma_2=1$). The exponent is now $\frac{2^2}{4} + \frac{1^2}{1} = 1 + 1 = 2$.
    The point $[2,1]$ is *more likely* under this stretched distribution, because it's only 1 standard deviation away on the first axis.

**FIGURE**
A 3-panel figure showing 2D Gaussian contours.
*   Panel 1: "Isotropic" with $\Sigma = I$. Contours are perfect circles.
*   Panel 2: "Diagonal" with $\Sigma = \text{diag}(4, 1)$. Contours are ellipses aligned with the axes.
*   Panel 3: "Full Covariance" with non-zero off-diagonals. Contours are rotated ellipses.

---

## SLIDE · "KL divergence and cross-entropy"

**CURRENT PROBLEM**
This slide defines three concepts (Entropy, KL, Cross-Entropy) at once with dense formulas. The relationship $\text{KL} = H(p, q) - H(p)$ is stated without justification.

**INSERT BEFORE**
*Intuition Slide: The Wrong Dictionary*

"Imagine you have to send a message in English, but the only Morse codebook you have is for French. French uses 'q' and 'z' more than English. Your French codebook will use short codes for 'q' and 'z', and maybe a longer one for 'e'. When you encode English text, your messages will be inefficiently long!
-   **Entropy** is the length of the message using the *correct* (English) codebook.
-   **Cross-Entropy** is the length using the *wrong* (French) codebook.
-   **KL Divergence** is the *extra length*, the penalty for using the wrong book."

**REWRITE**
Let's break down KL Divergence step-by-step. It measures the "distance" from a true distribution $p$ to a model's approximation $q$.

$$ \text{KL}(p \| q) = \sum_y p(y) \log \frac{p(y)}{q(y)} $$

Using the logarithm rule $\log(a/b) = \log(a) - \log(b)$:
$$ = \sum_y p(y) (\log p(y) - \log q(y)) $$
$$ = \sum_y p(y) \log p(y) - \sum_y p(y) \log q(y) $$
$$ = - \left(-\sum_y p(y) \log p(y)\right) + \left(-\sum_y p(y) \log q(y)\right) $$

Now let's define the two terms:
-   **Entropy** $H(p) = -\sum_y p(y) \log p(y)$. This is the "ideal" number of bits to encode data from $p$.
-   **Cross-Entropy** $H(p, q) = -\sum_y p(y) \log q(y)$. This is the number of bits you'd use if you designed your code for $q$, but the data actually came from $p$.

Substituting these back in, we see:
$$ \text{KL}(p \| q) = -H(p) + H(p, q) = H(p, q) - H(p) $$
So, KL divergence really is the "penalty term" on top of the base entropy.

**INSERT AFTER**
*Worked Numeric Example*

True 3-class distribution (a weighted die): $p = [0.1, 0.6, 0.3]$.
Model's prediction: $q = [0.2, 0.5, 0.3]$.

**Cross-Entropy**:
$H(p, q) = -[0.1 \log(0.2) + 0.6 \log(0.5) + 0.3 \log(0.3)]$
$= -[0.1(-1.61) + 0.6(-0.69) + 0.3(-1.20)]$
$= -[-0.161 - 0.414 - 0.360] = 0.935$

**Entropy** (for comparison):
$H(p) = -[0.1 \log(0.1) + 0.6 \log(0.6) + 0.3 \log(0.3)]$
$= -[-0.23 - 0.31 - 0.36] = 0.90$

The KL-divergence is $0.935 - 0.90 = 0.035$. This is the "cost" of our model's inaccuracy. Minimizing cross-entropy is equivalent to minimizing KL divergence since $H(p)$ is a fixed constant.

**FIGURE**
A Venn diagram-style figure. Draw a large circle labeled "Cross-Entropy H(p,q)". Inside it, draw a smaller, concentric circle labeled "Entropy H(p)". The region between the circles is labeled "KL Divergence KL(p||q)".

---

## SLIDE · "Log-likelihood gradients · the elegant fact"

**CURRENT PROBLEM**
This is a massive "assume too much" violation. It states the final gradient of softmax-cross-entropy without any derivation, which requires backpropagation—a topic the audience hasn't seen.

**INSERT BEFORE**
*Intuition Slide: Push and Pull*

"Imagine you have three knobs, one for each class logit: 'cat', 'dog', 'bird'. The true answer is 'dog'. The gradient `prediction - truth` tells you exactly how to turn the knobs.
-   For the 'dog' knob, the truth is 1. If your prediction was 0.7, the gradient is `0.7 - 1 = -0.3`. It says 'turn me down', but that's for gradient *ascent*. For *descent*, we go the other way: we *increase* the 'dog' logit.
-   For the 'cat' knob, the truth is 0. If your prediction was 0.2, the gradient is `0.2 - 0 = 0.2`. It says 'turn me up', so for descent we *decrease* the 'cat' logit.
The gradient elegantly tells the model to 'push up' the probability of the correct class and 'pull down' the probabilities of the wrong ones."

**REWRITE**
Let's derive this result carefully. It's a bit of calculus, but shows why this loss function is so convenient.
Let the logits be $\mathbf{z}$, the softmax probabilities be $\boldsymbol{\pi}$, and the one-hot true label be $\mathbf{y}$.
The loss for a single example where the true class is $k$ is $\mathcal{L} = -\log \pi_k$.

We need the gradient $\frac{\partial \mathcal{L}}{\partial z_i}$ for every logit $z_i$. By the chain rule:
$\frac{\partial \mathcal{L}}{\partial z_i} = \sum_{j=1}^K \frac{\partial \mathcal{L}}{\partial \pi_j} \frac{\partial \pi_j}{\partial z_i}$

**Step 1: Gradient with respect to probabilities.**
$\frac{\partial \mathcal{L}}{\partial \pi_j} = \frac{\partial}{\partial \pi_j}(-\log \pi_k)$. This is non-zero only when $j=k$. So, $\frac{\partial \mathcal{L}}{\partial \pi_j} = -\frac{1}{\pi_k}$ if $j=k$, and 0 otherwise.

**Step 2: Gradient of softmax.**
This is the tricky part. The softmax is $\pi_j = \frac{e^{z_j}}{\sum_c e^{z_c}}$.
-   Case A: $i = j$. (e.g., $\frac{\partial \pi_1}{\partial z_1}$). Using the quotient rule, this works out to $\pi_i(1 - \pi_i)$.
-   Case B: $i \ne j$. (e.g., $\frac{\partial \pi_2}{\partial z_1}$). This works out to $-\pi_i \pi_j$.

**Step 3: Combine them.**
The sum $\sum_j$ only has one non-zero term from Step 1, which is when $j=k$. So we only care about $\frac{\partial \pi_k}{\partial z_i}$.
-   Case A: $i = k$ (we are at the correct logit).
    $\frac{\partial \mathcal{L}}{\partial z_k} = \frac{\partial \mathcal{L}}{\partial \pi_k} \frac{\partial \pi_k}{\partial z_k} = (-\frac{1}{\pi_k}) (\pi_k(1-\pi_k)) = -(1 - \pi_k) = \pi_k - 1$.
-   Case B: $i \ne k$ (we are at a wrong logit).
    $\frac{\partial \mathcal{L}}{\partial z_i} = \frac{\partial \mathcal{L}}{\partial \pi_k} \frac{\partial \pi_k}{\partial z_i} = (-\frac{1}{\pi_k}) (-\pi_i \pi_k) = \pi_i$.

Let's check this against our formula $\pi_i - y_i$, where $y_i$ is 1 for the true class and 0 otherwise.
-   Case A ($i=k$): $\pi_k - y_k = \pi_k - 1$. Matches!
-   Case B ($i \ne k$): $\pi_i - y_i = \pi_i - 0 = \pi_i$. Matches!
So the gradient is indeed just $\text{prediction} - \text{truth}$.

**INSERT AFTER**
This slide is already very dense. A worked example is better placed after the main cross-entropy derivation.

**FIGURE**
A simple computational graph diagram. An input `z_i` goes into a "Softmax" box, which outputs `pi_j`. `pi_k` goes into a "-log" box, which outputs `L`. Then show arrows for the chain rule flowing backward from `L` to `z_i`.

---

## SLIDE · "MSE · pop out from MLE"

**CURRENT PROBLEM**
The math is too compressed. It jumps from the log-likelihood formula to the conclusion, skipping the intermediate algebraic steps that are crucial for a first-time audience.

**INSERT BEFORE**
*Intuition Slide: The Archer's Score*

"An archer shoots at a bullseye. A perfect shot is at the center (the model's prediction $\hat{y}$). Real shots have some random error, often forming a bell curve (Gaussian noise) around the center.
How do we score the archer's skill (the model parameters)? We look at all their shots (the data $y_i$). A good archer is one for whom the observed shots are 'likely'.
Maximizing the probability of all the shots is the same as minimizing the average squared distance from the center. That's why MLE with a Gaussian assumption *is* Mean Squared Error."

**REWRITE**
Let's derive this step-by-step.
**1. The Assumption:** We assume $y_i$ comes from a Gaussian centered at our model's prediction, $\hat{y}_i = \mathbf{w}^\top \mathbf{x}_i$.
$p(y_i \mid \mathbf{x}_i, \mathbf{w}) = \frac{1}{\sqrt{2\pi \sigma^2}} \exp\!\left(-\frac{(y_i - \hat{y}_i)^2}{2 \sigma^2}\right)$

**2. Write the Likelihood:** For the whole dataset $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$, assuming IID:
$\mathcal{L}(\mathbf{w}) = \prod_{i=1}^N p(y_i \mid \mathbf{x}_i, \mathbf{w})$

**3. Take the Log-Likelihood:** This turns the product into a sum.
$\log \mathcal{L}(\mathbf{w}) = \log \left( \prod_{i=1}^N p(y_i \mid \mathbf{x}_i, \mathbf{w}) \right) = \sum_{i=1}^N \log p(y_i \mid \mathbf{x}_i, \mathbf{w})$
$= \sum_{i=1}^N \log \left( \frac{1}{\sqrt{2\pi \sigma^2}} \exp\!\left(-\frac{(y_i - \hat{y}_i)^2}{2 \sigma^2}\right) \right)$

**4. Simplify using Log Rules:** $\log(A \cdot B) = \log(A) + \log(B)$, and $\log(e^x)=x$.
$\log \mathcal{L}(\mathbf{w}) = \sum_{i=1}^N \left[ \log\left(\frac{1}{\sqrt{2\pi \sigma^2}}\right) - \frac{(y_i - \hat{y}_i)^2}{2 \sigma^2} \right]$
$= N \log\left(\frac{1}{\sqrt{2\pi \sigma^2}}\right) - \frac{1}{2\sigma^2} \sum_{i=1}^N (y_i - \hat{y}_i)^2$

**5. Find the Maximum:** We want to find the $\mathbf{w}$ that maximizes this expression.
$\hat{\mathbf{w}}_\text{MLE} = \arg\max_\mathbf{w} \left[ N \log\left(\frac{1}{\sqrt{2\pi \sigma^2}}\right) - \frac{1}{2\sigma^2} \sum_{i=1}^N (y_i - \hat{y}_i)^2 \right]$

The first term is constant with respect to $\mathbf{w}$. The $-1/(2\sigma^2)$ is a negative constant. Maximizing a function `-C * f(w) + D` is the same as minimizing `f(w)`.
$\hat{\mathbf{w}}_\text{MLE} = \arg\min_\mathbf{w} \sum_{i=1}^N (y_i - \hat{y}_i)^2$

This is exactly the Mean Squared Error loss function!

**INSERT AFTER**
*Worked Numeric Example*

Data: $(x=2, y=5)$. Model: $\hat{y} = w \cdot x$. Let's test two values for $w$. Assume $\sigma=1$.
-   **Hypothesis 1: $w=2$**.
    Prediction $\hat{y} = 2 \cdot 2 = 4$. Error is $(5-4)^2=1$.
    Log-Likelihood $\propto -\frac{1}{2\cdot 1^2} (1) = -0.5$.
-   **Hypothesis 2: $w=3$**.
    Prediction $\hat{y} = 3 \cdot 2 = 6$. Error is $(5-6)^2=1$.
    Log-Likelihood $\propto -\frac{1}{2\cdot 1^2} (1) = -0.5$.
-   **Hypothesis 3: $w=2.5$**.
    Prediction $\hat{y} = 2.5 \cdot 2 = 5$. Error is $(5-5)^2=0$.
    Log-Likelihood $\propto -\frac{1}{2\cdot 1^2} (0) = 0$.

The maximum log-likelihood (0) occurs when the squared error is minimized (at $w=2.5$).

**FIGURE**
A plot of a single data point $(x_i, y_i)$. Show a regression line representing $\hat{y}_i$. Draw a vertical line segment for the residual $(y_i - \hat{y}_i)$. On the y-axis, centered at $\hat{y}_i$, draw a 1D Gaussian curve, and mark the position of $y_i$ on it to show its probability density.

---

## SLIDE · "Cross-entropy · pop out from MLE"

**CURRENT PROBLEM**
Again, the derivation is too fast. It shows the final log-likelihood formula without starting from the base Bernoulli PMF and showing how the logs are applied.

**INSERT BEFORE**
*Intuition Slide: The Biased Coin Assessor*

"You are given a coin and told to figure out its bias, $p$ (prob of heads). You flip it 10 times and get H, T, T, H, H, ... (your data).
How do you find the best $p$? You write down the probability of seeing that *exact sequence* of flips.
Prob = $p \cdot (1-p) \cdot (1-p) \cdot p \cdot p \ldots = p^{\#heads} (1-p)^{\#tails}$.
To make this sequence most likely, you pick $p = \#heads / (\#heads + \#tails)$. This is MLE!
For classification, our neural network is just a machine that produces a different 'p' for every input 'x'. The cross-entropy loss is just the log of this likelihood formula."

**REWRITE**
Let's derive this step-by-step for binary classification.
**1. The Assumption:** The true label $y_i \in \{0, 1\}$ is a draw from a Bernoulli distribution whose parameter $p_i$ is given by our model's sigmoid output: $\hat{p}_i = \sigma(\mathbf{w}^\top \mathbf{x}_i)$.
The compact form for the Bernoulli probability is:
$p(y_i \mid \mathbf{x}_i, \mathbf{w}) = \hat{p}_i^{y_i} (1-\hat{p}_i)^{1-y_i}$
(Check: if $y_i=1$, this is $\hat{p}_i^1 (1-\hat{p}_i)^0 = \hat{p}_i$. If $y_i=0$, this is $\hat{p}_i^0 (1-\hat{p}_i)^1 = 1-\hat{p}_i$. It works.)

**2. Write the Log-Likelihood:** For one data point:
$\log p(y_i \mid \mathbf{x}_i, \mathbf{w}) = \log (\hat{p}_i^{y_i} (1-\hat{p}_i)^{1-y_i})$

**3. Simplify using Log Rules:** $\log(A \cdot B) = \log(A) + \log(B)$ and $\log(A^x) = x \log(A)$.
$\log p(y_i \mid \mathbf{x}_i, \mathbf{w}) = \log(\hat{p}_i^{y_i}) + \log((1-\hat{p}_i)^{1-y_i})$
$= y_i \log(\hat{p}_i) + (1-y_i) \log(1-\hat{p}_i)$

**4. Find the Maximum over the Dataset:** We sum this over all $N$ data points and maximize it.
$\hat{\mathbf{w}}_\text{MLE} = \arg\max_\mathbf{w} \sum_{i=1}^N \left[ y_i \log(\hat{p}_i) + (1-y_i) \log(1-\hat{p}_i) \right]$

**5. Convert to a Minimization Problem:** By convention, we minimize the *negative* log-likelihood (NLL).
$\text{Loss}(\mathbf{w}) = -\sum_{i=1}^N \left[ y_i \log(\hat{p}_i) + (1-y_i) \log(1-\hat{p}_i) \right]$

This is exactly the **Binary Cross-Entropy** loss function.

**INSERT AFTER**
Use the existing "Worked example · bin-CE on one prediction" slide, but explicitly frame it in terms of likelihood.
e.g., "If true class y=1 (cat) and model predicts $\hat{p}=0.8$:
- Log-Likelihood is $1 \cdot \log(0.8) + (1-1) \cdot \log(0.2) = \log(0.8) \approx -0.223$.
- NLL (Loss) is $0.223$.
If true class y=0 (dog) and model predicts $\hat{p}=0.8$:
- Log-Likelihood is $0 \cdot \log(0.8) + (1-0) \cdot \log(0.2) = \log(0.2) \approx -1.609$.
- NLL (Loss) is $1.609$.
The model is punished far more for being confidently wrong."

**FIGURE**
A two-panel plot.
-   Panel 1: Plot of $-\log(p)$ vs $p$. Label x-axis "Model's probability for true class=1". Label y-axis "Loss".
-   Panel 2: Plot of $-\log(1-p)$ vs $p$. Label x-axis "Model's probability for true class=1". Label y-axis "Loss when true class=0".

---

## SLIDE · "MAP · maximum a posteriori"

**CURRENT PROBLEM**
The slide correctly states that a Gaussian prior leads to L2 regularization, but it's a "trust me" statement. The math is not shown.

**INSERT BEFORE**
*Intuition Slide: A Humble Scientist*

"An MLE scientist looks only at the data. If they flip a coin 3 times and get 3 heads, they declare 'The probability of heads is 100%!'
A MAP scientist is more humble. They have a *prior belief* (e.g., 'most coins are fair'). They see 3 heads and think, 'That's evidence for a high probability of heads, but it's probably not 100%. Maybe it's 80% or 90%.'
The prior acts as a regularizer, pulling the estimate away from extreme conclusions and towards a more 'sensible' default. For neural networks, our prior is often 'weights should be small and close to zero'."

**REWRITE**
Let's see how L2 regularization emerges from a Gaussian prior on the weights $\mathbf{w}$.

**1. Bayes' Rule for Parameters:**
$p(\mathbf{w} \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \mathbf{w}) \, p(\mathbf{w})}{p(\mathcal{D})}$
Posterior $\propto$ Likelihood $\times$ Prior

**2. Maximize the Log-Posterior:** We want to find the most probable parameters given the data.
$\hat{\mathbf{w}}_\text{MAP} = \arg\max_\mathbf{w} \log p(\mathbf{w} \mid \mathcal{D})$
$= \arg\max_\mathbf{w} \log(p(\mathcal{D} \mid \mathbf{w}) \, p(\mathbf{w}))$  (dropping the constant denominator)
$= \arg\max_\mathbf{w} [\log p(\mathcal{D} \mid \mathbf{w}) + \log p(\mathbf{w})]$

**3. Assume a Gaussian Prior:** Let's assume a simple prior that each weight $w_j$ is drawn from a zero-mean Gaussian with variance $\sigma_p^2$.
$p(\mathbf{w}) = \prod_j \mathcal{N}(w_j \mid 0, \sigma_p^2) = \prod_j \frac{1}{\sqrt{2\pi\sigma_p^2}} \exp(-\frac{w_j^2}{2\sigma_p^2})$

**4. Take the Log of the Prior:**
$\log p(\mathbf{w}) = \sum_j \left[ \log(\text{const}) - \frac{w_j^2}{2\sigma_p^2} \right]$
$= \text{const} - \frac{1}{2\sigma_p^2} \sum_j w_j^2 = \text{const} - \frac{1}{2\sigma_p^2} \|\mathbf{w}\|^2_2$

**5. Put it all together:**
$\hat{\mathbf{w}}_\text{MAP} = \arg\max_\mathbf{w} \left[ \log p(\mathcal{D} \mid \mathbf{w}) - \frac{1}{2\sigma_p^2} \|\mathbf{w}\|^2_2 + \text{const} \right]$

Converting to a minimization of the negative log-posterior:
$\hat{\mathbf{w}}_\text{MAP} = \arg\min_\mathbf{w} \left[ -\log p(\mathcal{D} \mid \mathbf{w}) + \frac{1}{2\sigma_p^2} \|\mathbf{w}\|^2_2 \right]$
$= \arg\min_\mathbf{w} \left[ \text{NLL Loss} + \lambda \|\mathbf{w}\|^2_2 \right]$, where $\lambda = \frac{1}{2\sigma_p^2}$.

This is exactly the standard loss function with L2 regularization (weight decay)!

**INSERT AFTER**
*Worked Numeric Example*
Let's say our NLL loss is 0.5. Our weight vector is $\mathbf{w}=[3, -4]$, so $\|\mathbf{w}\|_2^2 = 3^2 + (-4)^2 = 25$.
Let $\lambda = 0.01$.
The final MAP loss is $0.5 + 0.01 \cdot 25 = 0.5 + 0.25 = 0.75$.
The optimizer now has to balance making the NLL small with keeping the weights from growing too large.

**FIGURE**
A 2D plot of a loss landscape (contours). Add a second set of contours centered at the origin, representing the Gaussian prior (circles). The MAP estimate is the point where these two contour systems "touch" – a compromise between the data likelihood and the prior. Label the MLE point (center of the loss contours) and the MAP point.