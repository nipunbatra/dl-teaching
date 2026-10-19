"""Short, spoken prompts and final takeaways for the visible teaching strip."""

# Each pair is (talk-through line, final-build takeaway). Keep these short enough
# to read at a glance while presenting. Detailed explanations remain in notes.
LINES = {
    's01': (
        'Look at the variety here. What would a model need to learn to make another example?',
        'A generator learns from examples so that it can produce new samples.'),
    's02': (
        'We have one prompt. Before seeing the other samples, would you expect one correct image?',
        'The prompt describes a distribution of possible images, so different samples can all fit.'),
    's03': (
        'The boxes tell us where objects belong. Which choices are still left to the model?',
        'A layout constrains object positions while leaving room for different appearances.'),
    's04': (
        'We can see part of this face. What would count as a plausible way to fill the blue region?',
        'The visible pixels constrain the answer, but several completions can still make sense.'),
    's05': (
        'These strokes suggest colours and a layout. How closely should the result follow them?',
        'More initial noise gives the model more freedom and can weaken the influence of the strokes.'),
    's06': (
        'Zooming in cannot tell us which fine details were really there. What can the model add?',
        'Generated detail may look convincing without recovering the detail in the original scene.'),
    's07': (
        'We used edge maps to describe images. Here we use one to guide the creation of an image.',
        'Spatial features can serve as conditions that guide where image structure appears.'),
    's08': (
        'The reference images teach the model about this subject before we ask for a new setting.',
        'DreamBooth adapts a model to a subject; prompts then guide images made by that adapted model.'),
    's09': (
        'For a scientific task, how would we check whether a plausible image agrees with the measurements?',
        'Scientific usefulness needs validation against observations, beyond a convincing image.'),
    's10': (
        'All these examples involve sampling, sometimes with extra information that narrows the possibilities.',
        'We need a training example, a prediction target and a rule that turns predictions into samples.'),
    's11': (
        'A VAE or GAN maps a random latent to an image. How might we build that image through many steps?',
        'Diffusion generates through a chain of denoising steps that share one learned model.'),
    's12': (
        'Follow the same digit across the row. Which features disappear first as we add noise?',
        'Repeated corruption gradually makes the original digit harder to identify.'),
    's13': (
        'If several clean images could have produced this observation, how could we pick the original with certainty?',
        'Denoising uses learned patterns to handle the uncertainty left by corruption.'),
    's14': (
        'Each dot is a sample. Watch how the whole collection changes as the noise level rises.',
        'Noise changes a distribution; this toy example lets us calculate its score analytically.'),
    's15': (
        'Start with one pixel value, then imagine doing the same calculation at every position in the tensor.',
        'Our clean pixels lie in [−1, 1], but adding Gaussian noise can take values outside that range.'),
    's16': (
        'We shrink the current signal slightly, then add a fresh Gaussian draw. Why are there square roots?',
        'Beta specifies the added variance, so the noise amplitude is the square root of beta.'),
    's17': (
        'One alpha describes one step. What happens to the signal after we multiply several such steps?',
        'Alpha-bar is the product of the stepwise alphas and tracks cumulative signal variance retention.'),
    's18': (
        'Expand two steps first. The independent noise terms combine into one Gaussian with a known variance.',
        'We can sample a noisy training input at any chosen time without simulating all earlier steps.'),
    's19': (
        'Let us do the arithmetic for one coordinate: scale the clean value, scale the noise, then add.',
        'The signal and noise contributions can cancel; a noisy value alone does not identify either contribution.'),
    's20': (
        'Hold the seed fixed and move time. Then switch modes to see what changes when each step draws fresh noise.',
        'Direct sampling gives the correct distribution at each time; a recursive path also specifies how times relate.'),
    's21': (
        'We could predict several quantities. In this lecture, we choose the Gaussian noise used to make the input.',
        'The denoiser takes a noisy image and its timestep, then predicts a real-valued noise tensor.'),
    's22': (
        'We made the noisy input ourselves, so we can keep the exact noise array as its training label.',
        'The model sees only the noisy image and time; the sampled noise stays on the target side.'),
    's23': (
        'Compare each predicted value with its target, square the difference, and average over the four entries.',
        'Averaging over tensor entries turns the prediction errors into one scalar training loss.'),
    's24': (
        'We know which noise we sampled. The model has less information: it sees only the noisy input and time.',
        'Under squared error, the best population prediction is the conditional mean of the possible noise values.'),
    's25': (
        'The same input pattern can mean different things at low and high noise. We give the model time to distinguish them.',
        'A time embedding lets one set of weights handle many corruption levels.'),
    's26': (
        'Each image in this batch has its own time. Check that its coefficient broadcasts only across that image.',
        'The batch dimension binds each image to its own timestep and noise draw.'),
    's27': (
        'Match each line of code to the training example we just built: time, noise, input, prediction and loss.',
        'One training update needs one denoiser call on a batch of directly constructed noisy inputs.'),
    's28': (
        'For this update, the sampled input and target are fixed. Which quantities can backpropagation change?',
        'Training updates the denoiser weights; it does not run the full generation chain.'),
    's29': (
        'Adding noise gives us labels for free. What would still be missing if the dataset contained only sevens?',
        'We can create noise targets automatically, but the clean dataset still determines what the model can learn.'),
    's30': (
        'Recall the VAE bound. Here the latent variable is a whole noisy path rather than one latent vector.',
        'The diffusion bound combines a prior term, local reverse-matching terms and an endpoint likelihood.'),
    's31': (
        'With fixed Gaussian variances, matching means becomes squared error. Substituting the noise prediction gives these weights.',
        'Simple noise MSE changes the variational weighting and does not include every term of the negative ELBO.'),
    's32': (
        'During training we started with a clean image. At generation time, we have only a fresh Gaussian draw.',
        'Generation keeps the learned weights fixed and repeatedly samples a less noisy state.'),
    's33': (
        'Undo the forward formula: subtract the estimated noise contribution, then divide out the signal scaling.',
        'The predicted noise gives a clean-image estimate, though small signal coefficients can amplify its error.'),
    's34': (
        'The clean estimate answers one question. The sampler needs a different quantity: the state at the previous time.',
        'The next denoiser call receives the sampled previous state, with the noise level expected at that time.'),
    's35': (
        'Use the noise prediction to set the centre of a Gaussian transition. Check which alpha appears in the denominator.',
        'The reverse mean uses one-step alpha; alpha-bar appears in the correction for predicted noise.'),
    's36': (
        'The mean gives the centre of the next distribution. Fresh noise lets us sample from that distribution.',
        'This sampler adds noise according to the chosen reverse variance, except at the final step.'),
    's37': (
        'Keep the mean, variance and sampled value separate. Move the fresh noise slider and watch which values change.',
        'Fresh sampling noise changes the sampled state while leaving the transition mean unchanged.'),
    's38': (
        'Walk backwards through time. Each call predicts noise, sets a transition mean and samples the next state.',
        'The same trained denoiser is reused from T down to 1, with no extra noise in the last step.'),
    's39': (
        'Read the strips separately: the top shows states fed along the chain, and the bottom shows clean estimates.',
        'A noisy chain state and its clean-image estimate come from the same call but serve different purposes.'),
    's40': (
        'The output has the input image shape because every coordinate needs its own predicted noise value.',
        'Noise prediction is a dense regression task; a U-Net is one way to implement it.'),
    's41': (
        'Track height, width and channels separately. At each skip connection, add the channels before the next block.',
        'The U-Net combines coarse features with finer encoder features and returns one noise value per pixel.'),
    's42': (
        'Turn the timestep into a vector, project it to the channel count, then add it across the spatial grid.',
        'Time features enter multiple blocks so the denoiser can adapt its processing to the noise level.'),
    's43': (
        'Think of each patch as a token. What must each output token contain so we can rebuild the latent grid?',
        'A diffusion transformer predicts patch entries that can be rearranged into a spatial output.'),
    's44': (
        'Count spatial positions first, then count all scalar entries. The channels make those two ratios different.',
        'A smaller representation makes latent denoising attractive, but its size ratio is not a runtime measurement.'),
    's45': (
        'First train the autoencoder. Then freeze it and use its encoded images as data for the denoiser.',
        'Latent diffusion learns in a representation supplied by an autoencoder, with checkpoint-specific scaling.'),
    's46': (
        'For text-to-image, start with a random latent. Which autoencoder component is needed once denoising finishes?',
        'The standard sampling path denoises in latent space, then decodes the final latent into pixels once.'),
    's47': (
        'The prompt becomes a sequence of text features. We can compute those features once and reuse them.',
        'Text conditioning supplies a feature sequence to the denoiser throughout generation.'),
    's48': (
        'Follow the source of each tensor: image features form queries, while text features form keys and values.',
        'Each image query weights text values, so the result keeps one row per image position.'),
    's49': (
        'Choose one image query and do the arithmetic: compare with keys, scale, normalise, then mix the values.',
        'Attention turns similarity scores into weights whose row sums to one, then uses them to combine values.'),
    's50': (
        'The prompt stays fixed while image features change. The denoiser can use the same text differently at each step.',
        'Cross-attention brings text information into repeated denoising calls.'),
    's51': (
        'Sometimes hide the condition during training. Keep the noisy input and noise target exactly as they were.',
        'Condition dropout trains the same model to make both conditional and unconditional predictions.'),
    's52': (
        'Start at the unconditional prediction and move in the direction of the conditional prediction. What does s control?',
        'A scale of 1 gives the conditional prediction; larger scales extrapolate beyond it.'),
    's53': (
        'Each row uses the same random draws. Look at what changes as we increase the guidance scale.',
        'Guidance changes the samples, but a larger scale does not guarantee better quality.'),
    's54': (
        'Return to these two inputs. Which pixels are observed directly, and which details must the model infer?',
        'Completion and super-resolution need sampling procedures that respect their particular observations.'),
    's55': (
        'Look at every digit, including the awkward ones. Which conclusions can this short training run support?',
        'This run demonstrates the pipeline; its malformed and similar digits show that sample quality remains limited.'),
    's56': (
        'We can renumber the vocabulary without changing the sentence. Would arithmetic on the IDs still mean the same thing?',
        'Token IDs are arbitrary labels; masking is one way to define meaningful discrete corruption.'),
    's57': (
        'Hide some tokens, predict vocabulary distributions, and score the predictions only where tokens were hidden.',
        'Masked-text training learns missing-token predictions; the LLaDA-style objective also weights the loss by 1/t.'),
    's58': (
        'Keep the prompt visible and begin with three unknown response slots. This scripted trace fills one slot per step.',
        'An iterative text sampler needs a rule for which positions to fill, when to fill them and how long to generate.'),
    's59': (
        'Compare how the sentence grows in each procedure. Then count the sequential model calls before discussing speed.',
        'Parallel token predictions allow a different generation schedule; latency depends on the full implementation.'),
    's60': (
        'Choose one opening example and explain it using the state, the condition, the training target and the sampling rule.',
        'Training learns a prediction from corrupted examples; generation repeatedly uses that prediction to draw a sample.'),
    'a01': (
        'Scale the means normally, but square the coefficients when scaling variances. Independence removes the cross term.',
        'Independent Gaussian terms combine into a Gaussian whose variance is the sum of the scaled variances.'),
    'a02': (
        'Write the evidence as an expectation, then use the concavity of log to move it inside as a lower bound.',
        'The gap between log evidence and the ELBO is a nonnegative posterior KL divergence.'),
    'a03': (
        'Factor the same forward path in reverse order while keeping the clean image in every conditional.',
        'This factorisation separates the negative bound into prior matching, reverse matching and reconstruction.'),
    'a04': (
        'During training we know the clean endpoint. That extra information makes this local posterior Gaussian.',
        'Conditioning on both the noisy state and clean image gives a tractable training posterior.'),
    'a05': (
        'Treat the noisy input and velocity target as a rotated pair of clean data and noise, then invert the rotation.',
        'Prediction targets can be converted, but the loss, sampler and checkpoint must use matching conventions.'),
    'a06': (
        'To take a larger step, we need a transition defined for the new pair of times. Simply deleting calls is insufficient.',
        'DDIM defines a different transition that can connect selected timesteps.'),
    'a07': (
        'Weight each cluster by how well it explains the current point, then combine its contribution to the score.',
        'The mixture score is exact here, while the displayed finite-step Gaussian reverse remains approximate.'),
    'a08': (
        'Choose a noise point and a data point. Along their straight path, the training target is the vector between them.',
        'Flow matching learns a velocity field; sampling integrates that field from noise towards data.'),
    'a09': (
        'At time t, each response position is masked with probability t. The objective weights its masked-token loss by 1/t.',
        'The time weighting is part of this objective, so unweighted masked cross-entropy is a different loss.'),
    'a10': (
        'Keep the head axis separate while multiplying. After mixing text values, merge heads and restore the image grid.',
        'Cross-attention normalises over text positions for each image query in each head.'),
    'a11': (
        'For each image, say where it came from and what was actually supplied, trained or measured.',
        'A reproducible example needs its condition, model, sampling settings and evidence of what was evaluated.'),
}

TITLES = {
    's15': 'What is a clean image tensor?',
    's03': 'Where should the objects go?',
    's06': 'What detail is missing at low resolution?',
    's09': 'Could this help answer a lab question?',
    's11': 'From one-step generators to a denoising chain',
    's13': 'Can we recover the original with certainty?',
    's14': 'What happens to a whole distribution?',
    's19': 'Work through one noisy coordinate',
    's20': 'Try changing the noise level',
    's23': 'Calculate the loss for four values',
    's29': 'Where do the training labels come from?',
    's31': 'How does the bound lead to noise MSE?',
    's34': 'Which state goes into the next call?',
    's35': 'Build one reverse transition',
    's37': 'Work through one reverse step',
    's40': 'What should the denoiser output?',
    's41': 'Follow the shapes through a small U-Net',
    's43': 'Can a transformer predict the noise?',
    's44': 'Why denoise a compressed representation?',
    's45': 'Train the autoencoder, then the denoiser',
    's48': 'How do image features attend to text?',
    's49': 'Work through one attention calculation',
    's50': 'Use the prompt at every denoising step',
    's51': 'Teach one model to work with or without a condition',
    's52': 'Combine predictions with classifier-free guidance',
    's53': 'What changes when we turn up guidance?',
    's58': 'Fill the response a few tokens at a time',
    's59': 'How do the two text samplers differ?',
    's60': 'Can we now explain the opening examples?',
}
