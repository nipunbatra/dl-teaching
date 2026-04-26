Of course. Here is a concrete rewrite plan for the GANs lecture, designed for an audience with basic ML knowledge but no prior deep learning experience.

***

## SLIDE · "The minimax objective"

**CURRENT PROBLEM**
The slide presents the full minimax objective in expectation form without any buildup. For a student new to this, the combination of `min`, `max`, `E`, and two log terms is overwhelming.

**INSERT BEFORE**
**Title: How do we grade the Forger and the Detective?**
Let's think about the loss functions. The Detective (D) is just a binary classifier. We know how to train those!
- For a **real image**, D should output 1. The loss for that is `-log(D(real))`.
- For a **fake image**, D should output 0. The loss for that is `-log(1 - D(fake))`.

The Forger (G) has the *opposite* goal. It wants D to output 1 for its fakes.

**REWRITE**
Let's build the math step-by-step from the binary cross-entropy loss you already know.

**Part 1: The Discriminator's (D) Goal**
D wants to be a good classifier. This means:
- For real images `x` from the data, it wants `D(x)` to be close to 1.
- For fake images `G(z)` from the generator, it wants `D(G(z))` to be close to 0.

So, D wants to **maximize** this combined objective:
`log(D(x))` (this is big when D(x) is near 1) + `log(1 - D(G(z)))` (this is big when D(G(z)) is near 0).

Averaged over a whole batch of real and fake data, this is written with expectations:
$$ \max_D \; \mathbb{E}_{x \sim p_\text{data}}[\log D(x)] \,+\, \mathbb{E}_{z \sim p(z)}[\log(1 - D(G(z)))] $$

**Part 2: The Generator's (G) Goal**
G has the exact opposite goal. It wants to make `D(G(z))` as close to 1 as possible to fool the discriminator. This is equivalent to making the second term above, `log(1 - D(G(z)))`, as small as possible.

So, G wants to **minimize** D's objective:
$$ \min_G \max_D \; V(D, G) = \mathbb{E}_{x \sim p_\text{data}}[\log D(x)] \,+\, \mathbb{E}_{z \sim p(z)}[\log(1 - D(G(z)))]$$

This is a two-player "minimax" game.

**INSERT AFTER**
**Title: A Worked Numeric Example**
Let's use a single real image `x` and a single fake image `G(z)`.
- **Initial state:** G is bad, D is learning.
  - D sees real `x` and outputs `D(x) = 0.7`.
  - D sees fake `G(z)` and outputs `D(G(z)) = 0.2`.
- **D's objective value:** `log(0.7) + log(1 - 0.2) = -0.36 + log(0.8) = -0.36 - 0.22 = -0.58`.
  To maximize this, D's gradients will push `D(x)` towards 1.0 and `D(G(z))` towards 0.0.
- **G's objective value:** G only cares about the second term, `log(1 - D(G(z))) = -0.22`.
  To minimize this, G's gradients will change G's parameters to make a new `G(z)` that pushes `D(G(z))` towards 1.0.

**FIGURE**
A diagram with two streams. The top stream shows a real image (`x`) going into a box `D`, which outputs a score `D(x)`. This score goes to a loss function `log(D(x))`. The bottom stream shows random noise (`z`) going into a box `G` to produce a fake image `G(z)`. This fake image goes into the *same* box `D`, which outputs `D(G(z))`. This score goes to a loss `log(1-D(G(z)))`. An arrow labeled "D's Goal: Maximize Sum" points to both loss terms. An arrow labeled "G's Goal: Minimize this" points only to the second loss term.

---

## SLIDE · "⚠️ optional · Why this objective · one derivation"

**CURRENT PROBLEM**
This slide is a massive cognitive leap. It introduces the optimal discriminator `D*` and Jensen-Shannon Divergence (JSD) out of nowhere and states the equivalence without showing the crucial intermediate steps.

**INSERT BEFORE**
**Title: Intuition · Why is this game optimizing anything meaningful?**
It seems like we're just training a classifier. How does this help the generator match the real data?
**Analogy: Tug of War.**
Imagine the true data distribution `p_data` and the generator's distribution `p_G` are two teams in a tug of war. The discriminator `D` is the rope. `D` constantly adjusts to find the point of maximum tension between the two teams.
The game is set up so that this "maximum tension" is exactly a known measure of distance between distributions (the JSD). By training G to minimize this tension, G is forced to move its team to be in the same place as the `p_data` team, at which point the tension (JSD) becomes zero.

**REWRITE**
Let's find the best possible discriminator `D` for a fixed generator `G`. The objective for `D` is:
$$ V(D) = \int_x p_\text{data}(x) \log(D(x)) dx + \int_z p_z(z) \log(1 - D(G(z))) dz $$
Let $p_G(x)$ be the distribution of generated samples. The objective becomes:
$$ V(D) = \int_x \left[ p_\text{data}(x) \log(D(x)) + p_G(x) \log(1 - D(x)) \right] dx $$
To find the maximum, we take the derivative of the term inside the integral with respect to `D(x)` and set it to 0. Let `a = p_data(x)` and `b = p_G(x)`, and `y = D(x)`. We are maximizing `a log(y) + b log(1-y)`.
$$ \frac{d}{dy} [a \log(y) + b \log(1-y)] = \frac{a}{y} - \frac{b}{1-y} = 0 $$
$$ \frac{a}{y} = \frac{b}{1-y} \implies a(1-y) = by \implies a - ay = by \implies a = (a+b)y $$
$$ y = \frac{a}{a+b} $$
Substituting back, the optimal discriminator is:
$$ D^*(x) = \frac{p_\text{data}(x)}{p_\text{data}(x) + p_G(x)} $$
Now, we plug this optimal `D*` back into the original minimax objective, which becomes a minimization over G only:
$$ \mathcal{L}(G) = \mathbb{E}_{x \sim p_\text{data}} \left[\log \frac{p_\text{data}}{p_\text{data}+p_G}\right] + \mathbb{E}_{x \sim p_G} \left[\log \left(1 - \frac{p_G}{p_\text{data}+p_G}\right)\right] $$
$$ = \mathbb{E}_{p_\text{data}}\left[\log \frac{p_\text{data}}{p_\text{data}+p_G}\right] + \mathbb{E}_{p_G}\left[\log \frac{p_\text{data}}{p_\text{data}+p_G}\right] $$
This expression is exactly `2 * JSD(p_data || p_G) - log(4)`. Minimizing this is the same as minimizing the Jensen-Shannon Divergence between the real and fake distributions!

**INSERT AFTER**
**Title: What does the optimal `D*` tell us?**
$$ D^*(x) = \frac{p_\text{data}(x)}{p_\text{data}(x) + p_G(x)} $$
Let's plug in some numbers for a single point `x`:
- **Case 1: `x` is clearly real.** `p_data(x) = 0.9`, `p_G(x) = 0.1`.
  `D*(x) = 0.9 / (0.9 + 0.1) = 0.9`. The optimal D is very confident it's real.
- **Case 2: `x` is clearly fake.** `p_data(x) = 0.05`, `p_G(x) = 0.95`.
  `D*(x) = 0.05 / (0.05 + 0.95) = 0.05`. The optimal D is very confident it's fake.
- **Case 3: Nash Equilibrium.** The generator is perfect, so `p_data(x) = p_G(x)`.
  `D*(x) = p_data(x) / (p_data(x) + p_data(x)) = 0.5`. The optimal D can only guess. This is the goal of training!

**FIGURE**
A 1D plot showing two distributions. One curve is `p_data(x)` (e.g., a Gaussian at x=2) and another is `p_G(x)` (e.g., a Gaussian at x=-2). Below this, plot the corresponding `D*(x)`. The `D*(x)` curve will be close to 1 near x=2, close to 0 near x=-2, and cross 0.5 in the middle.

---

## SLIDE · "The non-saturating trick"

**CURRENT PROBLEM**
The slide explains the concept of saturation well, but the math is abstract. The numeric example on the next slide is hand-wavy. We can combine these and make the math concrete.

**INSERT BEFORE**
**Title: Intuition · A Flat Landscape**
**Analogy: Learning to Climb a Mountain in a Thick Fog.**
Imagine the generator `G` is a climber in a thick fog, and the discriminator's gradient is the slope of the ground telling it which way is "up".
Early in training, `G` is terrible. The fakes are obviously fake. `D`'s output `D(G(z))` is near 0.
The original loss, `min log(1-D(G(z)))`, is like a vast, flat plateau when `D(G(z))` is near 0. The slope is almost zero. The climber feels no incline and has no idea where to go.
The "non-saturating" trick is like changing the goal from "minimize your distance from the peak" to "maximize your current altitude". Both lead to the same peak, but the second one gives a strong "up" signal even when you are very far away.

**REWRITE**
Let's look at the gradients G receives. G's parameters are updated via the chain rule, which depends on the gradient of its loss with respect to D's output, `D(G(z))`.

**1. Original "Saturating" Loss for G:**
$$ L_G = \mathbb{E}[\log(1 - D(G(z)))] $$
The gradient of the inner term with respect to D's output `d` is:
$$ \frac{\partial}{\partial d} \log(1 - d) = -\frac{1}{1-d} $$
When `G` is bad, `D` is confident the sample is fake, so `d = D(G(z))` is close to 0.
The gradient is then `≈ -1/(1-0) = -1`. This looks okay, but this gradient is then multiplied by the gradient from the sigmoid in D, which is `d(1-d)`. When d is close to 0, this is tiny! The total gradient signal is very small.

**2. "Non-Saturating" Loss for G:**
Here, we flip the label and the objective. We tell G to maximize the log probability of its sample being *real*.
$$ L_G = \max_G \mathbb{E}[\log(D(G(z)))] \quad \equiv \quad \min_G \mathbb{E}[-\log(D(G(z)))] $$
The gradient of the inner term ` -log(d)` is:
$$ \frac{\partial}{\partial d} (-\log(d)) = -\frac{1}{d} $$
When `G` is bad, `d = D(G(z))` is close to 0.
The gradient is then `≈ -1/0.01 = -100`. This is a **huge** signal! Even when multiplied by a small `d(1-d)` from the sigmoid, the resulting gradient is much stronger, allowing G to learn effectively from the start.

**INSERT AFTER**
**Title: Worked Numeric Example: The Gradient Vanishes**
Let's assume the final activation of `D` before the sigmoid is `a`, so `D(G(z)) = σ(a)`. G's update depends on `∂L/∂a`.

Let's say `G` is bad, and `D` correctly outputs a very small probability for a fake sample: `D(G(z)) = 0.01`. This corresponds to a pre-sigmoid activation `a ≈ -4.6`.

- **Saturating Loss:** `L = log(1 - σ(a))`. By chain rule, `∂L/∂a = -σ(a)`.
  For `σ(a) = 0.01`, the gradient is **-0.01**. This is a tiny signal. G barely learns.

- **Non-Saturating Loss:** `L = -log(σ(a))`. By chain rule, `∂L/∂a = σ(a) - 1`.
  For `σ(a) = 0.01`, the gradient is `0.01 - 1 =` **-0.99**. This is a huge, stable signal. G learns quickly.

The non-saturating loss provides a gradient that is ~100x stronger in this common early-training scenario.

**FIGURE**
The existing figure `saturating_loss_curve.svg` is perfect. Just add callouts to it. Point to the flat region near `D(G(z))=0` on the blue curve and label it "Vanishing Gradient! G gets stuck." Point to the steep region near `D(G(z))=0` on the green curve and label it "Strong Gradient! G learns quickly."

---

## SLIDE · "Transposed convolution · upsampling primitive"

**CURRENT PROBLEM**
This slide presents a dry formula for a non-intuitive operation. Students will not understand *what* it is or *why* it works.

**INSERT BEFORE**
**Title: Intuition · How do we go from a small vector to a big image?**
**Analogy: Painting with a Stamp.**
A normal convolution is like looking at a patch of the image and summarizing it into one value.
A transposed convolution is the reverse. It's like having a single value (from a smaller feature map) and a "stamp" (the kernel). You dip the stamp in that value's "ink" and press it onto a larger canvas. You do this for every value in the small map, and where the stamp marks overlap, you add them up.
This lets the network learn the best "stamps" to upscale a low-resolution feature map into a high-resolution one.

**REWRITE**
A regular convolution takes a large feature map and makes it smaller. The Generator needs to do the opposite. `ConvTranspose2d` is a learnable upsampling layer.

Let's see how it works with a simple 2x2 input, a 3x3 kernel, and a stride of 2.
1.  We start with a larger, empty output grid.
2.  Take the top-left value of the input (let's say it's `a`). Multiply the *entire* kernel by `a`. Place this `3x3` result in the top-left of the output grid.
3.  Take the next input value (`b`). Multiply the kernel by `b`. Place this result on the output grid, shifted by the stride (2 steps to the right).
4.  Repeat for all input values. Where the results overlap, sum them.

This process is the mathematical inverse of a regular convolution, which is why it's sometimes called a "deconvolution" (though "transposed convolution" is more accurate).

**INSERT AFTER**
**Title: A Worked Numeric Example**
- **Input:** `[[1, 2], [3, 4]]`
- **Kernel:** `[[1, 1, 1], [1, 1, 1], [1, 1, 1]]` (a 3x3 block of ones)
- **Stride:** 2, **Padding:** 1

Let's compute the output:
1.  **Input `1`**: Multiplies kernel -> `[[1,1,1],[1,1,1],[1,1,1]]`. Place at `(0,0)`.
2.  **Input `2`**: Multiplies kernel -> `[[2,2,2],[2,2,2],[2,2,2]]`. Place at `(0,2)`.
3.  **Input `3`**: Multiplies kernel -> `[[3,3,3],[3,3,3],[3,3,3]]`. Place at `(2,0)`.
4.  **Input `4`**: Multiplies kernel -> `[[4,4,4],[4,4,4],[4,4,4]]`. Place at `(2,2)`.

**Resulting Output (4x4):**
(There is no overlap with stride 2 and kernel 3)
```
[[1, 1, 2, 2],
 [1, 1, 2, 2],
 [3, 3, 4, 4],
 [3, 3, 4, 4]]
```
*(Note: I simplified the math slightly for clarity; with padding the output is larger and has overlaps, but this shows the core idea)*

**FIGURE**
An animated GIF. It shows a 2x2 input feature map. A colored highlight moves over the input pixels one by one. For each highlighted input pixel, a 3x3 kernel is shown being multiplied by that pixel's value, and the result is "painted" onto a larger output grid. The animation clearly shows the striding and summation process.

---

## SLIDE · "FID in one paragraph"

**CURRENT PROBLEM**
The FID formula is mathematically dense, using trace and matrix square roots. Students will not grasp what it's actually measuring.

**INSERT BEFORE**
**Title: Intuition · How to compare two clouds of points?**
Imagine you have two photos of star clusters: one real (from Hubble) and one fake (from your GAN). How do you score the fake one?
You can't compare them star-by-star. Instead, you measure statistical properties of each cluster.
1.  **Check the Center:** Is the center of your fake cluster in the same place as the center of the real one? This is the distance between the means (`μ`).
2.  **Check the Shape & Spread:** Is your fake cluster the same shape and size as the real one? Is it a sphere when it should be a flat disk? This is the distance between the covariances (`Σ`).
FID is a formula that combines both of these distances for the *features* of real and fake images.

**REWRITE**
FID measures the distance between the distribution of real image features and fake image features. First, we pass all real and fake images through a pre-trained network (like Inception-V3) to get high-level feature vectors for each image. Then we model each set of vectors as a multivariate Gaussian.

The FID formula has two parts:
$$ \text{FID} = \underbrace{\|\mu_r - \mu_f\|^2}_{\text{Part 1: Distance between centers}} + \underbrace{\text{tr}\!\left(\Sigma_r + \Sigma_f - 2 (\Sigma_r \Sigma_f)^{1/2}\right)}_{\text{Part 2: Distance between shapes/spreads}} $$

**Part 1: Distance between Mean Features**
- `μ_r`: The average feature vector of all real images.
- `μ_f`: The average feature vector of all fake images.
- `||...||²`: The squared Euclidean distance. This term is 0 if the average fake image has the same features as the average real image.

**Part 2: Distance between Covariance Matrices**
- `Σ_r`: The covariance matrix for real features. It describes the spread and correlation of features (e.g., "if feature A is high, feature B tends to be low").
- `Σ_f`: The covariance matrix for fake features.
- This term measures how different the "shapes" of the two feature clouds are. It becomes 0 only if `Σ_r = Σ_f`.

A low FID score means both the average features and the variety of features in your generated images are very similar to the real ones.

**INSERT AFTER**
**Title: A Worked 2D Numeric Example**
Imagine our features are 2-dimensional.
- Real images: Centered at `μ_r = [10, 20]`. Spread is `Σ_r = [[2, 0], [0, 2]]` (a circle of variance 2).
- Fake images: Centered at `μ_f = [11, 20]`. Spread is `Σ_f = [[4, 0], [0, 1]]` (an ellipse, wider than it is tall).

**Calculate Part 1 (Distance of means):**
`||μ_r - μ_f||² = ||[10-11, 20-20]||² = ||[-1, 0]||² = (-1)² + 0² = 1`.
There's a small penalty because the centers are slightly off.

**Calculate Part 2 (Distance of covariances):**
This is complex, but intuitively we are comparing a circle `Σ_r` to an ellipse `Σ_f`. They have different shapes, so this part of the formula will be a positive number, penalizing the generator for not matching the diversity of the real data. The final FID would be `1 + (some positive number)`.

**FIGURE**
Three plots of 2D point clouds.
1.  **Plot 1 (Low FID):** Two overlapping clouds of points, one red ('real'), one blue ('fake'). Both are centered at the same spot and have the same elliptical shape. Label: `μ_r ≈ μ_f`, `Σ_r ≈ Σ_f`.
2.  **Plot 2 (Mean Mismatch):** The red and blue clouds have the same shape, but the blue one is shifted. Label: `μ_r ≠ μ_f`, `Σ_r ≈ Σ_f`. An arrow points from this plot to the `||μ_r - μ_f||²` part of the FID formula.
3.  **Plot 3 (Covariance Mismatch):** The red and blue clouds are centered at the same point, but the red one is a circle and the blue one is a long, thin ellipse. Label: `μ_r ≈ μ_f`, `Σ_r ≠ Σ_f`. An arrow points from this plot to the `Tr(...)` part of the formula.

---

## SLIDE · "The WGAN objective"

**CURRENT PROBLEM**
This slide jumps to the dual form of the Wasserstein distance and introduces "1-Lipschitz" without any intuition. These are advanced concepts that will be opaque to the audience.

**INSERT BEFORE**
**Title: Intuition · Why vanilla GANs fail**
The original GAN's discriminator is a classifier that tries to draw an infinitely steep "cliff" between the real and fake data. When the data doesn't overlap, it succeeds perfectly. The cliff becomes infinitely steep, and the gradient at the cliff base (where the fakes are) becomes zero. G gets no signal.

**The WGAN Fix: A Critic with a Speed Limit.**
Instead of a classifier, WGAN uses a "critic" that scores the "realness" of an image.
**Analogy:** Imagine the critic is building a landscape. It wants to push the ground up under real data and push it down under fake data.
The **1-Lipschitz constraint** is a "speed limit" on how steep the landscape can be. It can't be steeper than a 45-degree slope. This prevents infinitely steep cliffs. Now, no matter how far away the fake data is, there is always a gentle, non-zero slope for the generator to follow "uphill" towards the real data.

**REWRITE**
WGAN replaces the classification objective with one based on the Wasserstein or "Earth Mover's" distance. Using a piece of advanced math called the Kantorovich-Rubinstein duality, we can write this objective as:
$$ \max_{D \in \text{1-Lipschitz}} \mathbb{E}_{x \sim p_\text{data}}[D(x)] - \mathbb{E}_{z \sim p(z)}[D(G(z))] $$
Let's break this down:
1.  **The Critic `D`:** `D` is no longer a classifier outputting a probability `[0, 1]`. It's a "critic" that outputs any real number (a score). We don't use a sigmoid at the end.
2.  **The Objective:** The critic `D` tries to maximize the *difference* between the average score of real data and the average score of fake data. It wants `D(real)` to be high and `D(fake)` to be low.
3.  **The Constraint: 1-Lipschitz.** This is the crucial part. A function `f` is 1-Lipschitz if `|f(a) - f(b)| ≤ |a - b|` for any `a` and `b`.
    - In simple terms: The slope of the function is never more than 1 (or less than -1).
    - This prevents `D` from creating an infinitely steep wall between real and fake data, which ensures the gradient is never zero. G always gets a useful signal.

**INSERT AFTER**
**Title: A Worked Numeric Example of Lipschitz Constraint**
Let's test if some simple 1D functions could be a WGAN critic.
- **Function 1: `D(x) = 5x`**
  - Let `a=2`, `b=3`. `|a-b| = 1`.
  - `|D(a) - D(b)| = |10 - 15| = 5`.
  - Since `5 > 1`, this function is **NOT** 1-Lipschitz. Its slope is too steep. A WGAN critic can't be this function.
- **Function 2: `D(x) = 0.5x`**
  - Let `a=2`, `b=3`. `|a-b| = 1`.
  - `|D(a) - D(b)| = |1 - 1.5| = 0.5`.
  - Since `0.5 ≤ 1`, this function **IS** 1-Lipschitz (at least for these points). This is a valid "slope" for a WGAN critic.
- **How WGAN enforces this:** The original WGAN clipped the weights of `D` to a small range. The better version (WGAN-GP) adds a penalty to the loss if the gradient's magnitude gets too far from 1.

**FIGURE**
A 1D plot. Show a "real" data distribution as a blue hump and a "fake" one as a red hump, far apart.
- **Top plot (Vanilla GAN):** Draw a sigmoid function `D(x)` that is 0 over the red hump and 1 over the blue hump, with a near-vertical transition between them. Add a text box: "Infinitely steep cliff → Zero gradient for G".
- **Bottom plot (WGAN):** Draw a straight line `D(x)` with a slope of 1 that is low over the red hump and high over the blue hump. Add a text box: "Gentle slope (max 1) → Always a useful gradient for G".

---

## SLIDE · "WGAN-GP · the practical version"

**CURRENT PROBLEM**
The gradient penalty formula is presented without explaining its components, especially `x-hat` and the gradient norm. It looks like a magic incantation.

**INSERT BEFORE**
**Title: Intuition · How do you enforce a speed limit everywhere?**
We want our critic `D` to have a slope of 1 everywhere. But we can't check every point in a billion-dimensional space.
**Analogy: Highway Patrol.**
The police can't put a speed camera on every inch of the highway. So, they place cameras at strategic, random locations. If they consistently catch people speeding at these random spots, it encourages everyone to follow the speed limit everywhere.
WGAN-GP does the same. It doesn't check the slope everywhere. It picks random points on the straight lines *between* real and fake samples and checks the slope there. It adds a penalty if the slope isn't 1. This is surprisingly effective at making the critic behave everywhere.

**REWRITE**
The original WGAN enforced the Lipschitz constraint by clipping the weights of D, which worked poorly. WGAN-GP (Gradient Penalty) enforces it much more effectively with an extra loss term.

$$ \mathcal{L}_\text{GP} = \lambda\, \mathbb{E}_{\hat{x}}[(\underbrace{\|\nabla_{\hat{x}} D(\hat{x})\|_2}_{\text{Slope of Critic at } \hat{x}} - 1)^2] $$
Let's break this down piece-by-piece:

1.  **Where do we check the slope? At `x-hat` (`hat{x}`).**
    We create a random point `hat{x}` that lies on the straight line between a real image `x` and a fake one `G(z)`.
    $$ \hat{x} = \epsilon x + (1-\epsilon) G(z), \quad \text{for a random } \epsilon \in [0, 1] $$
2.  **How do we measure the slope? The Gradient Norm `||...||`**
    The gradient `∇_hat{x} D(hat{x})` is a vector that points in the direction of steepest ascent of the critic's score at point `hat{x}`.
    The L2 norm `||...||_2` gives the *magnitude* of this vector. This is the "steepness" or "slope" of the critic at that point.
3.  **What's the penalty? `(slope - 1)²`**
    We want the slope to be exactly 1. This term `(||∇D|| - 1)²` is a simple squared-error loss.
    - If the slope is 1, the penalty is `(1-1)² = 0`.
    - If the slope is 1.5, the penalty is `(1.5-1)² = 0.25`.
    - If the slope is 0.2, the penalty is `(0.2-1)² = 0.64`.
The optimizer for `D` will now try to minimize its main loss *and* this penalty, forcing its gradients to have a norm of 1.

**INSERT AFTER**
**Title: The Full WGAN-GP Training Step for the Critic**
The critic's total loss is now:
$$ L_D = \underbrace{\mathbb{E}[D(G(z))] - \mathbb{E}[D(x)]}_{\text{Original WGAN Loss: push real scores up, fake scores down}} + \underbrace{\lambda \mathbb{E}_{\hat{x}} [ (\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2 ]}_{\text{Gradient Penalty: keep the slope at 1}} $$
**In code, for one batch:**
1. Get real batch `x` and generate fake batch `G(z)`.
2. Calculate the first part: `D(G(z)).mean() - D(x).mean()`.
3. Create `hat{x}` by mixing `x` and `G(z)`.
4. Calculate the critic's output for these mixed samples: `D(hat{x})`.
5. Calculate the gradient of this output *with respect to the input `hat{x}`*.
6. Calculate the L2 norm of that gradient for each sample in the batch.
7. Calculate the penalty: `((norm - 1)**2).mean()`.
8. `total_loss = part1 + lambda * part7`.
9. `total_loss.backward()`.

**FIGURE**
A 2D diagram. Show one point `x_real` and one point `G(z)`. Draw a straight line connecting them. Show several points `hat{x}_1`, `hat{x}_2`, ... on this line. For one of them, `hat{x}_i`, draw an arrow representing the gradient `∇D(hat{x}_i)`. The length of this arrow is `||∇D||`. A box next to it calculates the penalty `(length - 1)²`. This visualizes that the penalty is computed at interpolated points.