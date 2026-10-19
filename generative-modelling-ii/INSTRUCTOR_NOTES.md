# Instructor notes · Generative Modelling II

Generated from the speaker notes in `deck/slides/*.html` (press **S** in the deck to see them per click).
“Click n” is what to say when click n appears. Numbers come from `figures/data/numbers.json`.


## Opening

### 1. From noise to images · Denoising diffusion, from one noisy pixel to text-to-image models  `#cover`
- **click 1:** Open on the result. Every frame is a real state of a real sampler: our MNIST model, trained for this lecture. By the end of the class you will be able to write the loop that made this strip.
- Badges in the top-right corner say where each figure comes from: published (a paper's own figure), our model (trained here), computed (exact arithmetic, nothing learned), illustrative (hand-chosen numbers).

### 2. This digit was pure noise 1,000 steps ago  `#hook`
- **click 1:** Static. 784 random numbers drawn from a standard normal. Nothing in it says "digit".
- **click 2:** Watch it sharpen. Ask: at which frame can you first tell it is a digit? Most people say around 400 or 200.
- **click 3:** This is the question for today, and we will answer it with arithmetic on one pixel before we use any big model.
- *Bottom line on the slide:* How can learning to remove noise let us create an image from nothing but noise?

### 3. What would each model produce?  `#apps`
- **click 1:** Give students 20 seconds: for each input, what should come out? Ask one or two to answer aloud. Point out that every input is different in kind: words, pixels, boxes, photos.
- **click 2:** Reveal. Each output is one published sample from the paper's own figure. Do not explain any method yet; we will return to this slide at the end.
- *Bottom line on the slide:* Eight different inputs, one recipe. By the end you will be able to say, for each, what the network was trained to predict.

### 4. One prompt, several valid images  `#cherries`
- **click 1:** Ask: if you had to write this as a supervised-learning problem, what would the target image be?
- **click 2:** There is no single target. Several images satisfy the prompt, and some (bowls, flowers) only loosely. This is p(image | text).
- **click 3:** A fun detail that is also a real lesson: generators reproduce whatever is in their training data, including watermarks.
- *Bottom line on the slide:* Which one is correct? None and all: a prompt describes a distribution of images, and a generator samples from it.

### 5. Fill a missing region  `#inpaint-pub`
- **click 1:** Ask: what must every completion share? The eyes, hair and background. What may differ? The mouth.
- **click 2:** These are five samples, not five steps. In Part 6 we run the same trick on our digit model with two extra lines of code.
- *Bottom line on the slide:* The visible pixels constrain the answer; they do not decide it. Smile or no smile is still the model's choice.

### 6. Is the sharp detail real?  `#superres-pub`
- **click 1:** The low-resolution image cannot tell you where each wrinkle was. Regression, trained with squared error, outputs a smooth face. Ask why it is blurry; hold the answer until Part 2, where we will prove it.
- **click 2:** The diffusion sample is sharp. Ask: is it correct?
- **click 3:** Compare with the real photo. The sample's skin texture is plausible but not the same. Sharp is not the same as true.
- *Bottom line on the slide:* Regression blurs, a generator invents. Compare the wrinkles under the eye: plausible, and different from the real photo.

### 7. Would you count brick kilns in a generated image?  `#lab`
- **click 1:** This is what the sensor gives us. Ask: what is in this image? Hard to say.
- **click 2:** The model produces crisp buildings and roads. Would you put this in a map?
- **click 3:** The real image has a different layout. The generated version is a hypothesis consistent with the blurry input, not a measurement. Keep this in mind whenever diffusion is used in science.
- *Bottom line on the slide:* Plausible is not measured. Before using generated detail, we must check it against held-out observations.

### 8. Two generators we already trained, and their problems  `#lecture1`
- **click 1:** These are the actual models from Lecture I, same MNIST. The VAE's squared-error/likelihood training is stable, but averaging makes samples blurry.
- **click 2:** The GAN is sharp, but you saw how fragile the adversarial game was, including a collapse run.
- **click 3:** Diffusion keeps the boring, stable loss and moves the difficulty into many small denoising steps.
- *Bottom line on the slide:* Diffusion's bet: train with a plain regression loss, like the VAE, and generate in many small steps instead of one big jump.

### 9. Four questions answer the core of the lecture  `#plan`
- **click 1:** Training input first. Notice the order: corruption before learning.
- **click 2:** The target.
- **click 3:** Sampling: the loop on the first slide.
- **click 4:** Only once we have both the training step and the sampler do we ask why the loss is justified, using the ELBO from Lecture I.
- **click 5:** Architecture and conditioning are then short, because they reuse what you already know: U-Nets, transformers, cross-attention and CLIP.
- *Bottom line on the slide:* One running example throughout: a single 7 from the MNIST test set, and one pixel of it.


## Part 1 · Corrupt

### 10. Destroy a digit, one small step at a time  `#p1-open`

### 11. Our 7, and one pixel of it  `#digit`
- **click 1:** A handwritten 7 from the MNIST test set, the first image in the file. The orange square is the pixel we will track.
- **click 2:** Zoom into the 7 × 7 neighbourhood: raw uint8 values. Our pixel is a solid 255.
- **click 3:** Rescale to [−1, 1], so paper is −1 and ink is +1. This centring matters: the noise we add has mean 0.
- **click 4:** Everything in the next 30 minutes is done to this one number, +1, and then to all 784 at once.
- *Bottom line on the slide:* We will follow this one pixel, with value +1, through every equation in Parts 1–3.

### 12. One step: shrink the image a little, then add fresh noise  `#one-step`
- **click 1:** Multiply every pixel by √(1 − β). With β = 0.2 that is 0.894: the 7 gets slightly paler.
- **click 2:** Draw a brand-new noise image η, the same shape as the digit, and scale it by √β = 0.447.
- **click 3:** Add them. β_t is the noise variance added at step t.
- **click 4:** Our pixel: 0.894 × 1 + 0.447 × (−0.5) = 0.671, taking η = −0.5 at our pixel. The real schedule uses β between 0.0001 and 0.02; we use 0.2, ten times the largest, so one step is visible.
- *Bottom line on the slide:* With β = 0.2, our pixel goes from 1 to 0.671. Each step keeps most of the image and mixes in a little noise.

### 13. Variances add, so scale by square roots  `#sqrt`
- **click 1:** Ask the slide's question before clicking: why √(1 − β) and √β, not (1 − β) and β?
- **click 2:** Scaling a random variable by a multiplies its variance by a². Independent variances add. So 0.8 + 0.2 = 1.
- **click 3:** Without roots the variance decays towards 0, and the end point would depend on the step count. With them, the end point is always N(0, I): something we can sample from.
- *Bottom line on the slide:* The square roots keep the total variance at 1, so after many steps we land exactly on a standard Gaussian, $\mathcal N(0, I)$.

### 14. A thousand small steps dissolve the 7  `#fade`
- **click 1:** Ask: at t = 300, is it still a 7? Most students say yes. Which features go first? Thin strokes and the edges.
- **click 2:** By t = 500 only a ghost remains; from about 600 on, the panels are indistinguishable from pure noise. Keep this in mind: a lot of steps are spent near pure noise (the cosine schedule in the appendix fixes this).
- *Bottom line on the slide:* After about t = 600 every panel looks the same: almost nothing of the 7 is left.

### 15. How much signal is left after t steps: $\bar\alpha_t$  `#alphabar`
- **click 1:** Same variance bars as before, in the toy: one step leaves 0.8 signal variance.
- **click 2:** Two steps multiply: 0.8 × 0.8 = 0.64. ᾱ ("alpha bar") is the running product. Do not confuse α_t (one step) with ᾱ_t (all steps so far).
- **click 3:** On the real schedule, ᾱ_t falls smoothly from 1 to about 0.00004. The signal amplitude √ᾱ drops below 0.1 at t = 674.
- *Bottom line on the slide:* Blue is signal variance $\bar\alpha_t$, amber is noise variance $1-\bar\alpha_t$; they always add to 1. Signal falls below half at t = 260.

### 16. Skip the loop: jump straight to step t  `#jump`
- **click 1:** Start from one step.
- **click 2:** Substitute x₁ into x₂. The signal part is just a product of scales; the noise part has two separate draws.
- **click 3:** Sums of independent Gaussians are Gaussian with added variances: 0.8 × 0.2 + 0.2 = 0.36. The histogram on the right is 200,000 actual draws; the curve is N(0, 0.36).
- **click 4:** So x₂ is the clean pixel times 0.8 plus one standard Gaussian times 0.6. ε (epsilon) names this combined noise.
- **click 5:** The same argument works for any t: signal scale √ᾱ_t, noise scale √(1 − ᾱ_t). Appendix A1 does the general step.
- *Bottom line on the slide:* Any step t in one line: no loop. This is the formula training uses.

### 17. Signal and noise can cancel  `#pixel-x2`
- **click 1:** The jump formula with ε = −0.5: the noise pulls our ink pixel down from 0.8 to 0.5.
- **click 2:** The loop and the jump agree. The two noise draws combine into the same ε = −0.5.
- **click 3:** A paper pixel with a big positive noise draw lands on exactly the same value. This is the central difficulty of denoising, and we will quantify it in Part 2.
- *Bottom line on the slide:* From 0.5 alone you cannot tell whether the pixel was ink or paper. Undoing noise needs more than this one number.

### 18. Slide t and watch our 7  `#explorer`
- **click 1:** t = 100: the 7 is obvious, signal scale 0.95.
- **click 2:** t = 350: you can still see it, but signal scale is down to 0.53.
- **click 3:** t = 920: noise scale 0.9999. Drag the slider live if there is time; it uses the same ε image for every t, so only the mixing changes.
- *Bottom line on the slide:* The same formula covers every noise level: at t = 920 the image is 99.99% noise by variance.

### 19. Everything in Part 1 fits in a few lines of PyTorch  `#fwd-code`
- **click 1:** 1000 β values, linearly spaced, with a 0 in front so that beta[t] is β_t for t = 1 … 1000. The toy slides used β = 0.2 instead.
- **click 2:** The cumulative product gives ᾱ, and the leading 0 makes alpha_bar[0] = 1: step 0 is the clean image. The same tables serve the sampler in Part 3.
- **click 3:** A standard Gaussian image per example.
- **click 4:** The jump formula with broadcasting. Returning eps already hints at Part 2: the noise will be the label.
- *Bottom line on the slide:* No learning anywhere in this code. It returns the noisy input and the noise; Part 2 uses both.

### 20. Noise turns any dataset into the same Gaussian  `#clusters`
- **click 1:** Each dot is a 2-D data point; think of each as a whole image squeezed into two numbers. The same jump formula is applied to every dot. At t = 100 the clusters smear into a ring; by t = 250 the ring fills in; by 1000 nothing distinguishes the original clusters. Reversing this, taking a dot from the blob back to a cluster, is what the network must learn.
- *Bottom line on the slide:* Eight tight clusters at t = 0 become one round blob, $\mathcal N(0, I)$, by t = 1000. Generation will start from that blob.

### 21. A 7 has ink at our pixel; a 2 has paper  `#undo`
- **click 1:** Each curve is the distribution of our pixel's noisy value: N(0.8, 0.36) if it was ink, N(−0.8, 0.36) if paper (toy step 2). We observe 0.5. Bayes with equal prior: the ink curve is 9.2 times higher there, so P(ink | 0.5) = 0.90.
- **click 2:** At step 500 on the real schedule the curves almost coincide: 0.58 vs 0.42. One pixel alone is nearly useless. The surrounding pixels, and knowledge of what digits look like, are what a network brings.
- *Bottom line on the slide:* At step 2 an observed 0.5 says “ink” with probability 0.90; at step 500 only 0.58. Going backwards needs knowledge of digits, so it must be learned.

### 22. Forward is a fixed recipe; reverse is a model we must learn  `#fixed-learned`
- **click 1:** Top arrows: the corruption we just built. Fixed β's, no parameters, nothing to learn. Real images at t = 0, 100, 300 and 1000.
- **click 2:** Bottom arrows: what we want, a network that takes x_t and t and proposes x_{t−1}. Because each forward step is tiny, the true reverse step is close to Gaussian; we only need to learn its mean. That is Part 2.
- *Bottom line on the slide:* Each reverse step only has to undo a small step, which is why a Gaussian with a learned mean is a good model of it.

### 23. What we did, and what comes next  `#p1-recap`
- **click 1:** Recap in one breath; students should be able to say each line back.
- **click 2:** The bridge to the next part.


## Part 2 · Train

### 24. Training: a regression problem we build ourselves  `#p2-open`

### 25. Make an input and its label; ask the network for the noise  `#tuple`
- **click 1:** Start with a clean training image. Here, our 7.
- **click 2:** Draw a step t uniformly from 1 to 1000 (here 350) and a noise image ε from N(0, I).
- **click 3:** Use the jump formula: 0.534 × x₀ + 0.845 × ε. That is the network's input, together with t.
- **click 4:** The network outputs a full 28 × 28 image ε̂: its guess of the noise that was added. Our trained model gets most of it: mean squared error 0.015, against 1.0 for a network that outputs zeros. Compare the two amber images by eye.
- *Bottom line on the slide:* Everything except $\hat\epsilon$ was made by us. The noise we drew is the label: we get inputs and labels for free.

### 26. From one noisy input we could ask for $\epsilon$, for $x_0$, or for a mix  `#targets`
- **click 1:** The noise ε we drew. Amber is positive, blue negative, white zero.
- **click 2:** The clean image is an equally valid target, and the formula converts one into the other given x_t.
- **click 3:** v mixes the two; appendix A5 shows it as a rotation. Different parameterisations train differently, but the sampler must use the matching formula. For the rest of the lecture: ε.
- *Bottom line on the slide:* We follow DDPM and predict $\epsilon$: whatever t is, the target is always a standard Gaussian image, so the regression stays well scaled.

### 27. Squared error, pixel by pixel, then averaged  `#four-pixels`
- **click 1:** Real numbers from our trained network, for the example on the previous slide, at four pixels including ours. Ask: which prediction is worst?
- **click 2:** Average over all 784 pixels. In code it is one line.
- **click 3:** The error map shows where the network is unsure: along the stroke, where it must decide how much of the dark value is ink and how much is noise.
- *Bottom line on the slide:* That is the whole objective: an ordinary regression on 784 numbers.

### 28. Can the network recover the exact noise? No, so it averages  `#cannot-know`
- **click 1:** The network sees x₂ = 0.5 (and the neighbours). Two worlds are consistent with it: ink with ε = −0.5, or paper with ε = +2.17. The ink world is 9.2 times more likely under the Gaussian noise, so 0.90 vs 0.10.
- **click 2:** Minimising expected squared error gives the conditional mean, −0.24. Even a perfect network does not output the true −0.5. Remember this for the sampler: ε̂ is a best guess, not the truth.
- **click 3:** In clean-image terms the best guess is 0.80: neither ink nor paper. Over a whole image these averages are what makes regression blurry. Neighbouring pixels sharpen the posterior: a pixel surrounded by ink is probably ink. That is the information the network adds.
- *Bottom line on the slide:* This is why regression blurs (the SR3 slide): it returns an average. Diffusion uses that average to take one small step, not as the final image.

### 29. What the network believes the clean 7 is, at each noise level  `#guesses`
- **click 1:** The same 7 noised with the same ε at five noise levels. At 920 you see nothing.
- **click 2:** The network's best guess of the clean image at each level. At 920: a blob, the average of everything that might be under that noise. At 700: a fuzzy stroke. By 350: clearly our 7.
- **click 3:** This is the conditional mean of the previous slide, for 784 pixels at once.
- *Bottom line on the slide:* At t = 920 the guess is a blurry average of digits; by t = 350 it is our 7. The average sharpens as evidence accumulates.

### 30. One network must handle nearly clean and nearly pure noise  `#one-net`
- **click 1:** Ask: should the network subtract the same amount of noise from all three? No: from the left one almost nothing, from the right one almost everything.
- **click 2:** Could the network infer t from the image? Roughly, but telling it is cheap and exact. Training one shared network, rather than 1000, is what makes this practical.
- *Bottom line on the slide:* One model for all 1000 levels, told which level it is at. Part 5 shows how t enters the network.

### 31. A batch mixes noise levels: one $\bar\alpha$ per image  `#gather`
- **click 1:** Three images in a batch, each with its own random step.
- **click 2:** Index the precomputed ᾱ table: one number per image. Note how wide the range is.
- **click 3:** Reshape to [B, 1, 1, 1] so PyTorch broadcasts the number over each image's channel, height and width.
- **click 4:** One multiply-add produces three differently noised images in one go.
- *Bottom line on the slide:* Random t per image means every batch trains the network at many noise levels at once.

### 32. The whole training step in PyTorch  `#train-code`
- **click 1:** The tables have 1001 entries so that index t means step t; entry 0 is the clean image. Off-by-one errors here are the most common bug in student code.
- **click 2:** t and ε are both random. ε is kept: it is the label.
- **click 3:** The jump formula, with broadcasting.
- **click 4:** The model takes both x_t and t.
- **click 5:** An ordinary optimiser step. This is literally the training_step function in scripts/train_models.py that trained the model on these slides.
- *Bottom line on the slide:* No reverse chain, no sampling loop: one random step per image, one forward pass, one MSE.

### 33. Gradients flow only into the network's weights  `#grads`
- **click 1:** Forward pass: data, a random draw and a random step make x_t with no parameters involved. Only the network has weights. ε also travels separately to the loss as the label.
- **click 2:** The loss compares ε̂ with ε: one number per batch.
- **click 3:** Backward: the gradient reaches θ through ε̂. Nothing upstream of the network has parameters, so nothing else is updated. Contrast with Lecture I's VAE, where gradients had to pass through a random draw (reparameterisation); here the draw is outside the graph.
- *Bottom line on the slide:* $x_0$, t and $\epsilon$ are constants made before the forward pass; only $\theta$ gets a gradient.

### 34. Our training run, and where the network still struggles  `#training-run`
- **click 1:** The loss drops fast and then slowly: the curve is on a log scale.
- **click 2:** Ask why small t is hard. At t = 1 the noise is tiny (scale 0.01) and nearly invisible against the digit, yet the network must report it at unit scale. At t = 1000 the input is almost exactly ε, so copying the input is nearly right. This unevenness is exactly what the ELBO weights on the next slides are about.
- *Bottom line on the slide:* Predicting the noise is hardest at small t (0.16 at t = 1) and easy near pure noise (0.000038 at t = 1000), where the input is almost exactly $\epsilon$.

### 35. What we did, and what comes next  `#p2-recap`
- **click 1:** Recap in one breath; students should be able to say each line back.
- **click 2:** The bridge to the next part.


## Part 3 · Sample

### 36. Sampling: run the chain backwards with the network  `#p3-open`

### 37. Start from noise; all we have is the network's guess  `#start-noise`
- **click 1:** 784 standard normal numbers. No x₀, no ε. Ask: what can we do with a network that predicts noise?
- **click 2:** At t = 1000 it predicts that almost everything is noise, and its implied clean image is the average digit: the conditional mean of Part 2 with almost no evidence. Jumping straight there would give a blur, like the VAE.
- *Bottom line on the slide:* The first guess is a featureless average. The trick is to move only a little way towards it, then ask again.

### 38. Invert the jump formula to guess the clean value  `#x0hat`
- **click 1:** Solve the jump formula for x₀, using the predicted noise in place of the true one.
- **click 2:** Our toy pixel: 0.925.
- **click 3:** Compare with the truth, 1. With the true ε = −0.5 the formula would give back exactly 1, but at generation time we never have it.
- **click 4:** The magnification factor √(1−ᾱ)/√ᾱ is 0.75 in the toy and 72 at t = 920 on the real schedule. That is the quantitative reason not to trust x̂₀ early on.
- *Bottom line on the slide:* At high noise a small error in $\hat\epsilon$ is magnified up to 72 times, which is why early guesses are poor and we take small steps.

### 39. Knowing $x_0$, the step back from $x_2$ is a Gaussian we can write down  `#true-step`
- **click 1:** Bayes' rule on the forward steps: q(x_{t−1} | x_t, x₀) ∝ q(x_t | x_{t−1}) q(x_{t−1} | x₀), both Gaussians we know. The product is a Gaussian (appendix A2 completes the square). Read the mean as a weighted blend of x₀ and x_t.
- **click 2:** Toy numbers, t = 2: both weights happen to be 0.894 × 0.2 / 0.36 = 0.497. Result 0.745.
- **click 3:** Ask: at generation time, which ingredient is missing? x₀. That is the whole problem, and the next slide's fix is the obvious one.
- *Bottom line on the slide:* This is the “true” reverse step. It needs $x_0$, which we have during training but not when generating.

### 40. The model's reverse step: the same blend, with $\hat x_0$ in place of $x_0$  `#plug-guess`
- **click 1:** Same formula, with the network's clean guess. 0.708.
- **click 2:** This defines the model's reverse distribution: Gaussian, learned mean, fixed variance.
- **click 3:** The gap 0.745 − 0.708 = 0.037 is the seed of Part 4.
- *Bottom line on the slide:* A better noise prediction gives a better clean guess, which gives a reverse step closer to the true one.

### 41. The same mean, written with the predicted noise  `#mean-step`
- **click 1:** Plugging the x̂₀ formula into the blend and simplifying gives this form. It is what the code uses: no explicit x̂₀ needed.
- **click 2:** Our pixel: subtract a third of ε̂, then divide by √0.8. Result 0.708, between the noisy 0.5 and the guess 0.925.
- **click 3:** Same number as the blend: two ways of writing one mean.
- *Bottom line on the slide:* Read it as: remove a fraction $\beta_t/\sqrt{1-\bar\alpha_t}$ of the predicted noise, then rescale by $1/\sqrt{\alpha_t}$. The code uses this form.

### 42. Then add fresh noise, except at the very last step  `#noise-step`
- **click 1:** DDPM fixes the reverse variance; we use the posterior variance β̃ (another common choice is β_t itself). For t = 2 in the toy, 0.111, so σ = 1/3.
- **click 2:** One standard normal draw, 0.3, moves the state to 0.808. Drag the sliders: ε̂ moves the mean and the guess, z moves only the next state.
- **click 3:** No noise at the last step: there is no step after it to absorb it.
- *Bottom line on the slide:* The mean says where to go; the noise keeps each step a sample from a distribution rather than a single best guess.

### 43. Four numbers on one line  `#three-values`
- **click 1:** The noisy input, 0.5.
- **click 2:** The reverse step's distribution: centred on μ_θ = 0.708 with σ = 0.333.
- **click 3:** One draw from it, 0.808: this becomes the new state.
- **click 4:** The clean guess 0.925, near but not equal to the truth 1. Students often feed x̂₀ back into the loop by mistake. Ask: which of these numbers goes into the next network call?
- *Bottom line on the slide:* The sampler keeps x₁ = 0.808 and feeds it back in. The guess x̂₀ = 0.925 is only for looking at.

### 44. Run our model with and without the fresh noise  `#why-noise`
- **click 1:** Eight samples from eight starting noise images with the ordinary sampler.
- **click 2:** The same starting noise, but at every step we keep only the mean. All eight become the same thin 1: the average distance between them drops from 21.9 to 0.6.
- **click 3:** The mean is a conditional average, so following means only gives an "average" image. Fresh noise is what makes each step a draw from the reverse distribution.
- *Bottom line on the slide:* Without fresh noise, all eight starting points end on the same thin “1”. Following only the mean leads every start to the same average answer.

### 45. The loop from the first slide, in ten lines  `#sample-code`
- **click 1:** beta, alpha = 1 − beta and alpha_bar are the Part 1 tables; post_var[t] = β_t(1 − ᾱ_{t−1})/(1 − ᾱ_t) is β̃_t.
- **click 2:** The loop runs from T down to 1. The step number is given to the network every time.
- **click 3:** The mean formula from two slides back, for every pixel at once.
- **click 4:** torch.randn_like(x) draws noise with x's shape. A common bug is torch.randn() with no shape, which adds the same single number to every pixel. Appendix A7 shows a faster sampler (DDIM) that uses 50 calls.
- *Bottom line on the slide:* 1000 network calls for one batch of images. Sampling is slow; training was one call per example.

### 46. The state, and the network's guess at every stage  `#trajectory`
- **click 1:** Top row: the actual chain, the same sample as on the title slide.
- **click 2:** Bottom row: x̂₀ at the same steps. At 1000, an average blob; by 600 to 400 the identity of the digit is set; the last hundred steps only sharpen edges.
- **click 3:** This is also why the simple loss's extra weight on noisy steps helps: those steps decide the content.
- *Bottom line on the slide:* The guess commits to a digit long before the state looks clean: the overall shape is decided at high noise, details at low noise.

### 47. Scrub through a sample as it forms  `#scrub`
- **click 1:** t = 1000: noise, and the guess is a grey blob.
- **click 2:** t = 400: the guess has committed to a digit; the state still looks noisy.
- **click 3:** t = 0: the finished sample. Click through the seeds; ask students to call the digit as early as they can.
- *Bottom line on the slide:* Drag slowly through t = 600 … 300: that is where each seed decides which digit it will be.

### 48. 32 samples from our model, in the order they came out  `#samples`
- **click 1:** Not cherry-picked: one call with batch size 32. Ask students to find the failures; there are some imperfect strokes.
- **click 2:** For each of the first six samples, the nearest training image by pixel distance. Similar style, different digits in detail: the model is generating, not retrieving. For real systems memorisation is a genuine concern (Carlini et al. 2023 extracted training images from Stable Diffusion).
- No quality metric (FID) was computed for this teaching model; a fair FID needs tens of thousands of samples and a reference feature network.
- *Bottom line on the slide:* The samples resemble training digits without copying them. Judge a generator by an unselected batch, and check for memorisation.

### 49. What we did, and what comes next  `#p3-recap`
- **click 1:** Recap in one breath; students should be able to say each line back.
- **click 2:** The bridge to the next part.


## Part 4 · ELBO

### 50. Why is noise MSE the right loss?  `#p4e-open`

### 51. A VAE has one latent; diffusion has a chain of them  `#two-chains`
- **click 1:** Recall Lecture I: an encoder maps our digit to a short code z, a decoder maps it back, and z is pulled towards N(0, I).
- **click 2:** Diffusion has the same shape, stretched: the noisy images are the latents, the noising chain plays the encoder and has nothing to learn, and the reverse network is a decoder applied T times. The end of the chain is N(0, I), the prior.
- **click 3:** That suggests the question for this part: what does the ELBO say when the encoder is fixed?
- *Bottom line on the slide:* If diffusion is a VAE with a fixed encoder, the VAE's training objective, the ELBO, should apply to it.

### 52. Diffusion is a VAE with a fixed encoder  `#elbo-bridge`
- **click 1:** Recall the VAE: encoder, decoder, prior, ELBO.
- **click 2:** Diffusion's latents are the noisy images. The "encoder" is our fixed noising recipe, so there is no encoder network and no reparameterisation needed for it.
- **click 3:** The decoder is the reverse step, and the same network is reused at every t. The prior is the end of the chain, N(0, I).
- **click 4:** The bound is derived exactly as in Lecture I (appendix A3 writes it out). We now take it apart.
- *Bottom line on the slide:* Same ELBO derivation as Lecture I. The new part is how it splits up, one term per step.

### 53. The bound splits into one comparison per step  `#elbo-terms`
- **click 1:** L_T: does the end of the forward chain match N(0, I)? With ᾱ_T = 0.00004 it almost exactly does, and there is nothing to learn.
- **click 2:** The middle terms compare, at each step, the network's reverse step with the "true" reverse step q(x_{t−1} | x_t, x₀). That true step is a Gaussian we can write down because we know x₀ during training (appendix A2).
- **click 3:** L₀ turns the last continuous state into pixel values. DDPM treats it separately. Next: one middle term, with numbers.
- *Bottom line on the slide:* $L_T$ has no parameters; $L_0$ is a small pixel-level term. Training is really about the middle terms.

### 54. Equal variances: KL is the squared gap between the means  `#kl-step`
- **click 1:** For Gaussians with equal variances, KL reduces to the squared difference of means over 2σ². The variance is fixed (not learned), so only the mean matters.
- **click 2:** Both numbers come from Part 3: the true reverse step from x₂ = 0.5 (knowing x₀ = 1) has mean 0.745; the model, with ε̂ = −0.4, proposes 0.708.
- **click 3:** 0.037² / 0.222 = 0.00625. Keep this number; it reappears on the next slide.
- *Bottom line on the slide:* Training pushes the model's reverse mean onto the true one. Next: rewrite both means in terms of noise.

### 55. Write both means with noise: the KL becomes noise MSE  `#kl-to-mse`
- **click 1:** The model's mean, written with the predicted noise. Treat c_t as a known constant for now.
- **click 2:** The true reverse mean has exactly the same form with the true ε: plug the jump formula's x₀ into the posterior mean of Part 3.
- **click 3:** Subtract: x_t cancels and only the noise error remains.
- **click 4:** Square and divide by 2σ²: a time-dependent weight times the squared noise error.
- **click 5:** Check with our numbers: 0.625 × 0.01 = 0.00625, the same value we got from the Gaussians directly.
- *Bottom line on the slide:* Each ELBO term is a weighted MSE between predicted and true noise, which is why we trained on noise.

### 56. DDPM keeps the squared error and drops the weights  `#weights`
- **click 1:** Red: w_t on the real schedule (t ≥ 2; the formula is undefined at t = 1 because the posterior variance is 0 there). Teal: the weight 1 the simple loss uses.
- **click 2:** The bound cares most about the last few low-noise steps, which mostly polish imperceptible detail. The simple loss spends more effort on mid and high noise, where the content of the image is decided.
- **click 3:** So, honestly: L_simple is motivated by the ELBO but is not equal to it. If you need likelihoods (bits per dimension), you must use the weighted bound or a learned variance (Nichol & Dhariwal 2021).
- *Bottom line on the slide:* The training loss is not the ELBO; it is a re-weighted sum of its per-step terms, chosen because it gave better samples (Ho et al. 2020).

### 57. What we showed, and what comes next  `#p4-recap`
- **click 1:** Summarise the bridge to Lecture I in four lines.
- **click 2:** Transition to architecture.


## Part 5 · Network

### 58. Inside the denoiser: networks you already know  `#p4-open`

### 59. Same shape in, same shape out: a dense prediction  `#dense`
- **click 1:** Input and output look alike, both noise-like. That is expected: the output is noise.
- **click 2:** Subtracting it shows what the network "saw": our 7.
- **click 3:** Recall the segmentation lecture: per-pixel class scores. Here it is per-pixel regression. Same need for local detail plus global context.
- *Bottom line on the slide:* So the architectures that work for dense prediction (encoder–decoders with skips) work here.

### 60. A small U-Net: shrink to see context, grow back for detail  `#unet`
- **click 1:** The down path: each level halves the resolution and doubles the channels (block height ~ resolution, width ~ channels).
- **click 2:** The middle: 128 channels at 7 × 7, where every unit sees most of the image.
- **click 3:** The up path mirrors it back to 28 × 28, then a 1 × 1 convolution gives one number per pixel.
- **click 4:** Skip connections concatenate the encoder's features into the decoder at the same resolution, so detail lost in the bottleneck is available again.
- **click 5:** Real channels from the trained network on our noisy 7: edges early, blobs in the middle, a 7-shaped map at the end. This is a teaching-sized network; real models such as Stable Diffusion v1's U-Net have about 860 million parameters and attention layers at low resolution.
- *Bottom line on the slide:* 814,145 parameters. The skips carry fine detail across; the middle sees the whole digit at 7 × 7.

### 61. t becomes a vector, then a per-channel offset in every block  `#time-inject`
- **click 1:** The raw number t is turned into 64 features. Same idea as positional encoding: a single integer becomes a rich vector.
- **click 2:** A small shared MLP, then in each block a linear layer maps it to that block's channel count.
- **click 3:** The result is added to every pixel of the block's feature map. This is the line from our model's code.
- *Bottom line on the slide:* Every block can change its behaviour with the noise level, at almost no cost in parameters.

### 62. Run 1000 steps on fewer numbers  `#latent-why`
- **click 1:** Stable Diffusion is a latent diffusion model: an autoencoder compresses each image, and diffusion runs in that latent space.
- **click 2:** 512² / 64² = 64.
- **click 3:** But 3 channels become 4, so the total count falls by 48. Two ratios that students often mix up. These are representation sizes, not measured speedups.
- **click 4:** Our own version: a small conv autoencoder for MNIST. Next slides show it working.
- *Bottom line on the slide:* The latent is still spatial (a small image), so the same U-Net or DiT works on it. The autoencoder handles fine pixel detail once.

### 63. First an autoencoder, then diffusion on its latents  `#two-stage`
- **click 1:** Stage 1: a convolutional autoencoder trained only to reconstruct (a VAE in Stable Diffusion; a plain autoencoder for us). Our 7's four latent channels are shown: a 7 × 7 picture each. The reconstruction is close to the input (MSE 0.0014).
- **click 2:** Stage 2: freeze it. Encode every training image, divide by the latents' standard deviation so they have unit scale (Stable Diffusion uses a constant 0.18215 for the same reason), and train the same DDPM step on those latents.
- *Bottom line on the slide:* The diffusion target is noise in latent coordinates; nothing about the training step changes.

### 64. Denoise latents for 1000 steps, decode once  `#latent-gen`
- **click 1:** z_T is 4 × 7 × 7 Gaussian noise. The latent denoiser runs the usual loop, then the decoder runs once to produce pixels. Notation: z_t in latent space, x_t in pixel space.
- **click 2:** Eight samples from our latent model. Quality is a little below the pixel model at this tiny scale; the gain is that each of the 1000 steps works on 196 numbers instead of 784. At 512 × 512 that gain is what makes the method affordable.
- *Bottom line on the slide:* No encoder at generation time: text-to-image starts from latent noise. Image-to-image tasks (SDEdit) can start from an encoded input instead.

### 65. A Transformer can be the denoiser too: DiT  `#dit`
- **click 1:** DiT works on a latent of 4 × 32 × 32, exactly the kind of compressed image we just built. Cut into 2 × 2 patches: 256 tokens of 16 numbers each, exactly like ViT's patch embedding.
- **click 2:** Full self-attention among all patches. Recall the earlier lecture: no causal mask, every patch sees every patch.
- **click 3:** Conditioning: the step and the class label produce a scale and a shift for each block's normalisation (adaLN-Zero). This is a different conditioning route from the text cross-attention we will see in Part 6.
- **click 4:** Unpatchify back to the latent's shape. The prediction is noise for every patch.
- *Bottom line on the slide:* Patches in, a prediction for every patch out: no class token, because the output is dense. t and the label set each block's scale and shift.


## Part 6 · Condition

### 66. Steering the denoiser with a prompt, a label or known pixels  `#p5-open`

### 67. What the network receives from a prompt  `#prompt`
- **click 1:** The CLIP tokenizer (the one Stable Diffusion v1 uses) splits our prompt into 5 word tokens plus start and end markers. Padded to a fixed length of 77.
- **click 2:** The real token IDs, from running the tokenizer. IDs are arbitrary labels: "basket" is 10338 only because of the vocabulary order; remember this in Part 7.
- **click 3:** A frozen text encoder turns them into 77 vectors. You trained this kind of encoder in the CLIP lecture; Stable Diffusion v1 uses CLIP ViT-L/14's text tower, 768 numbers per token. (The LDM paper's own text model, which produced the cherry images, used a BERT tokenizer and a transformer trained jointly.)
- **click 4:** The same C is used at every step; only the noisy image changes.
- *Bottom line on the slide:* The prompt becomes 77 vectors, computed once and reused at every one of the denoising steps.

### 68. Image positions ask questions; words supply the answers  `#cross-attn`
- **click 1:** Inside the U-Net, the feature map at some resolution has N = H × W positions. Each position makes a query. The highlighted cell is one position.
- **click 2:** The 77 text vectors make keys and values (only the five word tokens are drawn).
- **click 3:** Scores between that position's query and every word's key, softmax over the words: one row of the N × L attention matrix. The weights here are illustrative, not extracted from a model.
- **click 4:** The output is a weighted sum of value vectors, added back to that position's features. Shapes: Q is [B, N, d], K and V are [B, L, d], A is [B, N, L], AV is [B, N, d]. Heads are in appendix A11.
- *Bottom line on the slide:* In a VLM, text queried image features. Here image features query the text: each position pulls in the words relevant to it.

### 69. Scores, softmax over the words, weighted sum of values  `#attn-numbers`
- **click 1:** Hand-chosen 2-D features, so the arithmetic fits on a slide. Query 1 "looks for" the first feature.
- **click 2:** Dot products scaled by √d = √2.
- **click 3:** Softmax over the three words: golden and grass tie, dog gets less.
- **click 4:** Weighted sum of the value rows.
- **click 5:** A second image position with a different query gets a different mixture: more of "dog". Ask: why softmax over words and not over image positions? Each position must distribute one unit of attention over the prompt.
- *Bottom line on the slide:* Softmax runs over the words, separately for each image position: every position gets its own mixture of the prompt.

### 70. Train one network with the label, and sometimes without it  `#cfg-train`
- **click 1:** Same U-Net, plus a learned embedding of the digit label added to the time embedding. Labels are a simpler condition than text, and enough to show guidance on MNIST.
- **click 2:** 10% of the time the label is replaced by a special "none" index, with the same input and the same noise target. Our implementation of "no condition" is that extra embedding row; Stable Diffusion uses the empty prompt's encoding.
- *Bottom line on the slide:* The same weights now give two predictions for any noisy image: $\epsilon_c$ with the label and $\epsilon_u$ without.

### 71. Guidance: step past the conditional prediction  `#cfg-vectors`
- **click 1:** The unconditional prediction, drawn as a 2-D vector (illustrative numbers, equal axis scales).
- **click 2:** The conditional prediction. Their difference is "what knowing the label changes".
- **click 3:** Guidance scale 3 goes three times as far along that difference.
- **click 4:** Real images from our model. Without the label the guess is a vague blob; with it, a faint 7; at s = 3, a bold 7. Guidance sharpens what the label implies.
- *Bottom line on the slide:* Two network calls per step, combined by one line of arithmetic. No separate classifier is trained.

### 72. Turning up guidance trades variety for fidelity  `#cfg-grid`
- **click 1:** s = 0 ignores the label: any digit.
- **click 2:** s = 1, the plain conditional model. 95% are already sevens: on MNIST a digit label is a strong, simple condition. Text prompts are followed much less reliably at s = 1, which is why guidance matters more for text-to-image.
- **click 3:** s = 3: nearly all sevens.
- **click 4:** s = 7: thick, similar sevens. Spread falls as s grows. Text-to-image systems typically use s around 5–10 for the same reason: prompts are followed more literally, at the cost of variety and sometimes saturated colours.
- *Bottom line on the slide:* At s = 1, 95% already look like a 7. Larger s gives 100%, with bolder and more alike strokes: spread 16.4 → 15.7.

### 73. Inpainting with our model: paste the known pixels back in  `#inpaint-ours`
- **click 1:** We keep the top half of our 7 and hide the bottom.
- **click 2:** Each step: the model proposes x_{t−1} for the whole image; we overwrite the known region with the clean pixels noised to step t−1 using the exact forward formula. This is RePaint's core idea; RePaint also re-noises and repeats some steps ("resampling") to blend the two regions better, which we skip.
- **click 3:** Six seeds. Ask: are they all sevens? The top bar of a 7 is shared by several digits, so the model is free to choose. That is the conditional distribution from the opening slide.
- *Bottom line on the slide:* No retraining: known pixels come from the forward process, unknown ones from the reverse step. Different seeds give different digits that agree with the top half.

### 74. Start the reverse chain from a noised sketch  `#sdedit-ours`
- **click 1:** The sketch: our 7 reduced to 7 × 7 blocks. Noise it to t₀ = 200 with the jump formula and run only the last 200 reverse steps. The blocks are smoothed but the shape stays.
- **click 2:** Larger t₀ gives the model more freedom: by 800, outputs are clean handwritten digits, not necessarily faithful to the sketch. This is SDEdit, and it supports the claim made on the opening slide.
- *Bottom line on the slide:* Little noise keeps the sketch's blocks; a lot of noise gives a realistic digit that may drift from it. The starting t sets the trade-off.

### 75. Guidance and inpainting are a few changed lines  `#cond-code`
- **click 1:** Classifier-free guidance: run the network with and without the label (in practice as one batch of 2B) and combine.
- **click 2:** Inpainting with an unconditional model: after each step, replace the known region with the clean pixels noised to step t−1 (corrupt() from Part 1; at t = 1 that is the clean image itself).
- **click 3:** SDEdit replaces the first line: x = corrupt(sketch, t0) and the loop starts at t0. Super-resolution is the exception: the low-res image is an extra input the network was trained with.
- *Bottom line on the slide:* The trained network never changes; only how we call it and what we do with its output.

### 76. Super-resolution: feed the low-res image in as extra channels  `#superres-how`
- **click 1:** SR3 concatenates the upsampled low-resolution image to the noisy high-resolution state: 3 + 3 = 6 input channels. Trained on (low-res, high-res) pairs with the usual noise target.
- **click 2:** Contrast with our inpainting: there the unconditional model needed no retraining because the condition is "these pixels are known exactly". Here the condition is indirect, so the model must learn how it relates to the image.
- *Bottom line on the slide:* Unlike inpainting, this needs a model trained with the low-res input. And, as with our lab images, the added detail is a sample, not a measurement.


## Part 7 · Text

### 77. Can text be generated by denoising?  `#p6-open`

### 78. Token IDs are arbitrary labels; arithmetic on them means nothing  `#ids`
- **click 1:** A five-token sentence and its IDs under one vocabulary.
- **click 2:** Shuffle the vocabulary. The model and the sentence are unchanged; only the arbitrary numbering differs.
- **click 3:** The same Gaussian corruption turns "dog" into different words under different numberings. A noise process should not depend on an arbitrary labelling. (Diffusion on continuous word embeddings exists, e.g. Diffusion-LM; this section teaches the masked approach.)
- **click 4:** Masking: each token is either kept or replaced by a special [MASK]. "How noisy" becomes "what fraction is hidden".
- *Bottom line on the slide:* Numbers near 24 are not words near “dog”. For discrete data, corrupt by hiding tokens instead.

### 79. Hide a random fraction t of tokens; predict the hidden ones  `#mask-train`
- **click 1:** A clean sentence.
- **click 2:** Draw a corruption level t between 0 and 1 (here 0.4) and hide each token independently with that probability. Here two tokens are hidden.
- **click 3:** A bidirectional transformer outputs, at each position, a probability over the whole vocabulary: logits of shape [B, L, V]. Only the masked positions matter. These probabilities are illustrative.
- **click 4:** Cross-entropy at the masked positions, summed, times 1/t: the LLaDA objective (Nie et al. 2025). The 1/t weight comes from the corresponding ELBO, like the w_t of Part 2. With t = 0.4, the sum 0.868 becomes 2.17.
- *Bottom line on the slide:* Same pattern as images: corrupt by a random amount, predict what was removed. The model sees the whole sequence in both directions.

### 80. Start masked; fix the most confident slots first, in any order  `#mask-gen`
- **click 1:** The prompt "The dog" is fixed; we choose a response length of 3 and mask all three slots. The length has to be chosen in advance (or a maximum, with an end token).
- **click 2:** One forward pass predicts every hidden slot. Keep the most confident prediction ("ball", 0.71) and leave the others masked. This is low-confidence remasking (LLaDA): less confident predictions are thrown away and re-predicted.
- **click 3:** With "ball" known, the remaining slots become confident; here both pass a threshold and are fixed together. Three tokens in two passes, in the order 5, then 3 and 4. The probabilities are a scripted illustration, not a trained model's output.
- *Bottom line on the slide:* “ball” was chosen first, then two words at once. Every step predicts all hidden slots, and seeing “ball” made the others easier.

### 81. Two ways to write the same sentence  `#ar-vs-mask`
- **click 1:** The familiar decoder from the attention lectures: one token per pass, left to right.
- **click 2:** Masked diffusion: a bidirectional transformer refines all slots together, with a schedule deciding how many to fix per pass.
- **click 3:** Be careful with speed claims: wall-clock time depends on sequence length, the number of steps, KV caching and hardware. What changes for sure is the dependency structure, not necessarily the latency.
- *Bottom line on the slide:* Fewer passes is not automatically faster: each masked-diffusion pass processes the whole sequence, and autoregressive models reuse cached work.


## Wrap-up

### 82. Now we can say what each model is trained and run to do  `#revisit`
- **click 1:** Text-to-image: Parts 4 and 5.
- **click 2:** Two applications we ran ourselves on MNIST, with no new training.
- **click 3:** Two that need a model trained with the condition: concatenation for SR3, a copied encoder for ControlNet.
- **click 4:** Personalisation changes the weights before sampling. Text changes the corruption. GLIGEN (boxes) adds new attention layers that read box coordinates; DiffusionSat conditions on satellite metadata.
- *Bottom line on the slide:* One idea throughout: corrupt, learn to undo one step, then repeat from noise, with the condition as an extra input.

### 83. The whole method on one slide  `#answers`
- **click 1:** The forward process and the jump formula.
- **click 2:** The noise target, why MSE, and the qualification about weights.
- **click 3:** The sampler.
- **click 4:** The justification from Lecture I. Return to the opening question: the network never sees a digit at generation time; it only ever removes a little noise, and the fresh noise keeps every step a draw from a distribution.
- *Bottom line on the slide:* Removing noise creates an image because each small step is a sample from the learned reverse distribution, and the steps add up.

### 84. Three questions to answer before next class  `#exit`
- **click 1:** Give two minutes; ask students to write answers.
- **click 2:** Reveal. The appendix collects the derivations we skipped, plus DDIM, schedules, score matching and flow matching.


## Appendix

### 85. Derivations and extensions  `#appendix-open`

### 86. Independent Gaussians add: the step from t − 1 to t  `#a1-gauss`
- **click 1:** Induction on t.
- **click 2:** Expand one step.
- **click 3:** Independent zero-mean Gaussians: variances add.
- **click 4:** Same form one step later.

### 87. Knowing $x_0$ makes the reverse step a known Gaussian  `#a2-posterior`
- **click 1:** From Bayes: q(x_{t−1} | x_t, x₀) ∝ q(x_t | x_{t−1}) q(x_{t−1} | x₀), a product of two Gaussians in x_{t−1}; complete the square.
- **click 2:** Toy numbers: both coefficients are 0.894 × 0.2 / 0.36 = 0.497.
- **click 3:** Substitute the network's guess for x₀: 0.708, exactly the mean we computed from ε̂. Note this is not q(x_{t−1} | x_t), which would average over unknown x₀.
- *Bottom line on the slide:* The model's reverse mean is the true posterior mean with $\hat x_0$ plugged in; rewriting $\hat x_0$ in terms of $\hat\epsilon$ gives the formula of Part 3.

### 88. Bayes on each forward step makes the bound telescope  `#a3-bound`
- **click 1:** Start from the bound.
- **click 2:** Jensen and the two factorisations.
- **click 3:** Rewrite each forward step backwards, conditioned on x₀.
- **click 4:** The ratio terms telescope; group what remains into KLs (Ho et al. 2020, eq. 5; Luo 2022 has every line).
- *Bottom line on the slide:* The ratios $q(x_t\mid x_0)/q(x_{t-1}\mid x_0)$ cancel in a chain, leaving only $q(x_T\mid x_0)$: this is the decomposition used in Part 2.

### 89. $x_t$ and $v$ are $x_0$ and $\epsilon$ rotated by an angle that grows with t  `#a4-v`
- **click 1:** Since ᾱ + (1 − ᾱ) = 1, write √ᾱ = cos φ and √(1 − ᾱ) = sin φ: x_t is a rotation of (x₀, ε).
- **click 2:** v is the perpendicular direction; inverting the rotation recovers x₀ and ε (Salimans & Ho 2022).

### 90. The cosine schedule spends fewer steps on near-pure noise  `#a5-schedules`
- **click 1:** This is a schedule choice, independent of the sampler. Our model uses the linear schedule from DDPM.
- *Bottom line on the slide:* With the linear schedule, ᾱ is nearly 0 for the last ~40% of steps (Part 1's identical panels). The cosine schedule removes signal more evenly.

### 91. DDIM: jump between chosen steps using the clean guess  `#a6-ddim`
- **click 1:** Move to step s by re-noising the clean guess with the predicted noise. Deterministic given x_T (Song et al. 2021).
- **click 2:** Our model, both samplers, the same x_T. The samples are similar but not identical, because DDPM adds fresh noise along the way.
- *Bottom line on the slide:* Same weights, same starting noise, 20× fewer calls. Deleting steps from DDPM's loop would not work: DDIM defines a different transition.

### 92. Predicting noise is learning which way density increases  `#a7-score`
- **click 1:** The best noise prediction (Part 2's conditional mean) is the score, scaled (Tweedie's formula; Song & Ermon 2019).
- **click 2:** The finite-step Gaussian reverse is still an approximation, but with 1000 small steps the samples reproduce the eight clusters.

### 93. Learn a velocity along straight lines from noise to data  `#a8-flow`
- **click 1:** Pairs of noise and data points joined by straight lines; the regression target is the line's direction (Lipman et al. 2023; Liu et al. 2023). Stable Diffusion 3 uses this.
- **click 2:** Same lesson as Part 2: squared error returns a conditional average.

### 94. LLaDA's loss and its low-confidence remasking  `#a9-llada`
- **click 1:** The 1/t weight makes this an upper bound on the negative log-likelihood. Without it, it is an ordinary masked cross-entropy, a different objective.
- **click 2:** The sampler of Part 6.

### 95. Each head attends separately; then concatenate and project  `#a10-heads`
- **click 1:** Same as the Transformer lectures, with queries from the image and keys/values from the text.
- *Bottom line on the slide:* Softmax over the L text positions, for every image position in every head; then $W_O$ maps $h\,d_v$ back to the image feature width.

### 96. Say where every image came from  `#a11-report`
- **click 1:** The badge legend used throughout the slides.
