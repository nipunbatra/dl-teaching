---
title: "Lecture 6: Making Deep Networks Trainable"
source: "lecture6/L6-trainability.typ"
status: "Synced with the 104-slide handout"
---

# Lecture 6: Making Deep Networks Trainable

## Teaching guide for the final deck

Subtitle: **Vanishing gradients, initialization, normalization, and shortcuts**

The handout contains **104 slides**:

- slides 1–99 form the main lecture, including the title, seven section breaks, the final synthesis, and the sources slide;
- slides 100–104 are optional backup material.

The presentation build contains **209 incremental frames** because questions, calculations, and conclusions are revealed in stages.

The lecture is one investigation, not four disconnected topics. It begins with the phenomenon—signals and gradients change multiplicatively with depth—then derives a useful initialization, tests that calculation on one controlled 30-hidden-layer experiment, and finally asks what normalization and residual routes repair.

The governing teaching order is:

> **Observe a failure → work a small example → name the quantity → derive the mechanism → change one factor → measure again.**

Do not introduce a formula or diagnostic before students know which concrete problem it answers.

---

## Learning goals

By the end, students should be able to:

1. explain vanishing and exploding activations or gradients as repeated local gains;
2. distinguish the two jobs of random initialization: breaking symmetry and controlling scale;
3. derive fan-in scaling, Xavier/Glorot initialization, and He/Kaiming initialization from second moments;
4. read the tensor and matrix notation for the 30-hidden-layer MLP without ambiguity;
5. diagnose a deep model using activation RMS, active-ReLU fraction, finiteness, and activation-gradient RMS;
6. explain how BatchNorm and LayerNorm choose different reduction axes;
7. explain why a residual block contributes an identity term to its local Jacobian; and
8. test a diagnosis by changing one factor while holding the experiment fixed.

---

## The narrative spine

| Section | Slides | Question answered |
|---|---:|---|
| Why gradients vanish or explode | 2–10 | Why does depth turn a modest local mismatch into a severe global failure? |
| Initialization: symmetry and gates | 11–21 | Why must neurons start differently, and why can zero-initialized ReLU networks become stuck? |
| Choosing the weight scale | 22–32 | How should weight variance depend on fan-in, fan-out, and the activation? |
| 30-hidden-layer spiral experiment | 33–62 | Does the calculation predict real tensors, logits, probabilities, gradients, and training? |
| Normalization | 63–79 | How can a layer recompute centre and scale after initialization? |
| Residual blocks | 80–91 | How can information and gradients retain an identity route through depth? |
| Diagnosis | 92–99 | How do we locate the first failure and test one causal hypothesis? |
| Optional backup | 100–104 | What additional caveats and diagnostic patterns may be useful in discussion? |

The spiral experiment is the anchor from slide 34 onward. Keep returning to the same data, architecture, optimizer, seed, and measurements so students can attribute changes to mechanisms rather than to a moving benchmark.

---

## Notation and conventions

Introduce notation only when it becomes necessary, and retain these meanings throughout.

### Scalar warm-up

- \(h_\ell\): scalar activation after layer \(\ell\).
- \(q\): the common scalar multiplier in the four-layer warm-up, \(h_\ell=q h_{\ell-1}\).
- \(g_\ell=\partial\mathcal L/\partial h_\ell\): scalar gradient arriving at activation \(h_\ell\).
- \(\rho_\ell=\operatorname{RMS}(\mathbf g_{\ell-1})/\operatorname{RMS}(\mathbf g_\ell)\): measured one-layer backward RMS gain.

The scalar \(q\) is an intentionally simplified one-dimensional analogue of a local Jacobian. Do not imply that a matrix layer literally has one scalar multiplier.

### One affine–ReLU layer

For hidden layer \(\ell\):

\[
\mathbf z_\ell=W_\ell\mathbf h_{\ell-1}+\mathbf b_\ell,
\qquad
\mathbf h_\ell=\operatorname{ReLU}(\mathbf z_\ell).
\]

- \(\mathbf z_\ell\): preactivation vector, before ReLU.
- \(\mathbf h_\ell\): post-ReLU activation vector.
- \(W_{\ell,ji}\): weight from incoming coordinate \(i\) to receiving neuron \(j\) in layer \(\ell\).
- \(W_\ell\in\mathbb R^{d_\ell\times d_{\ell-1}}\): rows are receiving neurons; columns are incoming coordinates.
- fan-in of one receiving neuron: \(d_{\ell-1}\), the number of values entering its weighted sum.
- fan-out of one incoming coordinate: \(d_\ell\), the number of preactivations it influences.

PyTorch uses the same orientation: `nn.Linear(d_in, d_out).weight.shape == (d_out, d_in)`.

### Scale quantities

- \(\operatorname{Std}(w)\): standard deviation of individual initialized weights.
- \(s_\ell=\operatorname{RMS}(z_\ell)/\operatorname{RMS}(h_{\ell-1})\): affine-map gain.
- \(\kappa_\ell=\operatorname{RMS}(h_\ell)/\operatorname{RMS}(h_{\ell-1})\): full affine-plus-activation layer gain.
- At initialization, the scale target is approximately \(\kappa_\ell\approx1\), because full-layer gains multiply through depth.

Do not use "RMS gain" without saying whether it refers to the affine output, the post-activation output, or the backward signal.

### The controlled spiral experiment

- \(900\) examples, \(2\) input coordinates, \(3\) classes.
- \(30\) hidden layers of width \(128\), followed by output layer \(31\).
- \(W_1\in\mathbb R^{128\times2}\), \(W_2,\ldots,W_{30}\in\mathbb R^{128\times128}\), and \(W_{31}\in\mathbb R^{3\times128}\).
- \(\mathbf o=(o_0,o_1,o_2)\in\mathbb R^3\): zero-based class logits; there is no ReLU on the logits.
- \(H_\ell\in\mathbb R^{900\times128}\): batch of post-ReLU hidden activations; rows are examples and columns are hidden units.
- \(H_0=X\in\mathbb R^{900\times2}\).
- \(G_\ell=\partial\mathcal L/\partial H_\ell\); for hidden layers \(G_\ell\in\mathbb R^{900\times128}\), while \(G_0=\partial\mathcal L/\partial X\in\mathbb R^{900\times2}\).
- \(\alpha\): positive multiplier applied to every initialized weight matrix after the seeded He draw. It changes scale, not weight directions or signs.

The three runs are:

| Run | \(\alpha\) | Meaning |
|---|---:|---|
| A | 0.5 | half the He standard deviation |
| B | 1.0 | ordinary He initialization |
| C | 1.5 | 1.5 times the He standard deviation |

Because biases are zero and \(\alpha>0\), the three models have the same ReLU gate pattern at initialization. Their activation and gradient magnitudes differ.

### Normalization

For a batch matrix \(Z\in\mathbb R^{B\times D}\):

- rows index examples \(n\);
- columns index hidden features \(d\);
- BatchNorm computes statistics down one feature column, over examples;
- LayerNorm computes statistics across one example row, over features;
- learned \(\gamma_d\) and \(\beta_d\) remain feature-indexed.

### Residual blocks

\[
\mathbf h_{\ell+1}=\mathbf h_\ell+F_\ell(\mathbf h_\ell),
\qquad
\frac{\partial\mathbf h_{\ell+1}}{\partial\mathbf h_\ell}=I+J_{F_\ell}.
\]

The identity and branch outputs must have matching shapes; otherwise the skip path needs a projection. The lecture's residual model is an isolation experiment, not a claim that this one-branch toy is a standard modern ResNet block.

---

## Slide-by-slide teaching map

### Slides 1–10 — Why gradients vanish or explode

1. **Title.** State the central question: why can a correctly implemented deep network still be difficult to train?
2. **Section break.** Announce that the lecture starts with the failure, not with an initialization recipe.
3. **Identity network.** Remove nonlinearities and biases to isolate repeated multiplication: \(\hat{\mathbf y}=W_L\cdots W_1\mathbf x\).
4. **Four-layer scalar forward pass.** Define \(h_\ell\) explicitly as an activation. Work \(q=0.5,1,1.5\) to obtain \(h_4=0.0625,1,5.0625\).
5. **The same multipliers backward.** Start from \(g_4=1\) and show \(g_0=q^4\). Connect a small output sensitivity to a starved early layer.
6. **Matrix generalization.** Replace \(q\) by the local Jacobian \(J_\ell\); gradients multiply transposed Jacobians.
7. **Why RMS.** The ordinary mean of \((2,-2,1,-1)\) is zero, while its RMS is \(\sqrt{2.5}\approx1.58\).
8. **One-layer backward gain.** Define \(\rho_\ell\) from a hypothetical measured pair \(1.00\to0.90\).
9. **Depth multiplies gains.** Compare \(0.9^{30}\approx0.042\), \(1^{30}=1\), and \(1.1^{30}\approx17.45\).
10. **Optimization consequence.** Show how vanishing gradients starve early parameter updates and exploding signals or gradients destabilize numerics. Be precise: Adam cannot recreate a zero or underflowed signal.

Teaching checkpoint: students should now be able to say, in words, why a modest per-layer mismatch becomes severe with depth.

### Slides 11–21 — Initialization must break symmetry

11. **Section break.** Separate the two initialization jobs before deriving any variance.
12. **Two jobs.** Job 1: different hidden units need different starts. Job 2: signal scale must survive depth.
13. **Exact clone calculation.** Use \(x=2\), \(w_1=w_2=0.5\), \(v_1=v_2=0.25\), and \(y=1\). Equal incoming and outgoing parameters give equal activations and equal gradients.
14. **Width without diversity.** Four clones still span one hidden feature; width four is not functional rank four.
15. **XOR evidence.** On the fixed \(2\to4\to1\) ReLU model, cloned units stop at \(75\%\) while independent units reach \(100\%\). Use hidden-activation rank and the decision surface to make the limitation visible.
16. **Interactive symmetry lab.** Compare exact clones, a small perturbation, and independent starts. Ask when the parameters first separate.
17. **ReLU gate.** Positive \(z\) passes signal and gradient; negative \(z\) blocks both.
18. **One-example parameter update.** With incoming gradient \(4\) and \(x=2\), work the closed-gate and open-gate cases to show when \(w\) and \(b\) receive gradients.
19. **Dead-on-the-dataset definition.** A neuron is dead only when every training example closes its gate. A few negative preactivations are normal.
20. **Zero initialization setup.** Every hidden ReLU sits at \(z=0\); explicitly note that PyTorch uses derivative zero at the kink.
21. **Zero-gradient calculation.** Both output and hidden weight gradients are zero. An output bias may learn a constant, but hidden features remain absent.

Teaching checkpoint: distinguish "same neurons" from "small weights," and "a closed gate for one example" from "a dead neuron on the dataset."

### Slides 22–32 — Choosing the weight scale

22. **Section break.** Symmetry has been addressed; now ask how large independent nonzero weights should be.
23. **Name the three scale quantities.** Keep weight standard deviation, affine gain \(s_\ell\), and full-layer gain \(\kappa_\ell\) separate.
24. **Fan-in diagram.** One neuron forms \(z=\sum_i w_i h_i\). Ask whether \(100\) unit-RMS terms produce RMS \(1\), \(10\), or \(100\).
25. **Two-term enumeration.** Enumerate the four \(\pm1\) outcomes. Cross terms cancel in expectation, so \(\operatorname{RMS}(u_1+u_2)=\sqrt2\).
26. **One hundred terms.** Generalize to \(\operatorname{RMS}(z)=\sqrt{100}=10\), not \(100\). Clarify that this is over repeated random draws.
27. **Worked fan-in scaling.** With fan-in \(100\), changing \(\operatorname{Std}(w)\) from \(1\) to \(0.1\) changes output RMS from \(10\) to \(1\).
28. **General second-moment derivation.** Under independent zero-mean weights, \(E[z^2]=n\sigma_w^2E[h^2]\), hence \(\operatorname{Var}(w)=1/n\) for a linear map.
29. **Fan-out backward.** An input activation influences \(n_{\text{out}}\) outputs, so its gradient is a weighted sum of \(n_{\text{out}}\) terms.
30. **Xavier/Glorot.** For identity activations—or tanh near its linear regime—balance the forward preference \(1/n_{\text{in}}\) and backward preference \(1/n_{\text{out}}\) with \(2/(n_{\text{in}}+n_{\text{out}})\).
31. **ReLU correction.** For a symmetric preactivation, ReLU preserves half the second moment.
32. **He/Kaiming.** Compensate for that half: \(\operatorname{Var}(w)=2/n_{\text{in}}\), \(\operatorname{Std}(w)=\sqrt{2/\text{fan-in}}\). Work fan-in \(100\to0.141\) and \(128\to0.125\).

Assumptions to state aloud: zero bias, independent zero-mean weights, independence from incoming activations, comparable incoming second moments, and an approximately symmetric preactivation for the ReLU step.

### Slides 33–62 — Test the calculation on one 30-hidden-layer MLP

33. **Section break.** Move from derivation to a controlled falsifiable test.
34. **Experimental setup.** \(900\)-point three-class spiral, \(30\) hidden layers, width \(128\), full-batch Adam at \(3\times10^{-4}\).
35. **Three training traces.** Show that one run is stuck and two learn at different rates before revealing the intervention.
36. **Prediction question.** Ask which one setting changed.
37. **Decision regions at update 50.** Reveal that only \(\alpha\) changes and ask what it controls.
38. **Full network notation.** Define \(\mathbf h_0=\mathbf x\), hidden \(\mathbf z_\ell,\mathbf h_\ell\), and output logits \(\mathbf o\).
39. **One hidden neuron.** Expand \(z_{\ell,j}=\sum_iW_{\ell,ji}h_{\ell-1,i}+b_{\ell,j}\).
40. **Index reading.** Read \(W_{5,73}\) as the edge from hidden coordinate \(3\) in layer \(4\) to neuron \(7\) in layer \(5\).
41. **Rows and columns.** A row is one receiving neuron's fan-in; a column is one incoming coordinate's fan-out.
42. **Concrete shapes.** Show fan-in/fan-out for \(W_1\), \(W_2\ldots W_{30}\), and \(W_{31}\).
43. **He scales in this model.** Connect each layer's fan-in to its initialization standard deviation.
44. **Define the intervention.** Draw the same seeded He weights and multiply them by \(\alpha=0.5,1,1.5\) under `torch.no_grad()`.
45. **Weight distributions.** For hidden matrices with fan-in \(128\), show standard deviations \(0.0625,0.125,0.1875\). Note that using the same rule for the output matrix is an experimental control, not a general recommendation.
46. **What is recorded.** Define \(H_\ell\) as the \(900\times128\) post-ReLU tensor and show exactly where it is observed. Point to the hooks tutorial for implementation.
47. **RMS versus norm.** Repeating the same coordinates increases \(\ell_2\) norm but not RMS; therefore RMS is comparable across tensor sizes.
48. **Four diagnostics.** Activation RMS, active fraction, finiteness, and activation-gradient RMS. Read traces from input to output for forward values and from layer \(30\to0\) for backpropagation.
49. **Positive homogeneity.** Verify \(\operatorname{ReLU}(\alpha z)=\alpha\operatorname{ReLU}(z)\) for positive \(\alpha\).
50. **Successive relative scale.** Build \(0.5^\ell\) and \(1.5^\ell\) one layer at a time.
51. **Batch formula.** Lift the per-example result to \(H_\ell^{(\alpha)}=\alpha^\ell H_\ell^{(1)}\), so RMS scales by \(\alpha^\ell\).
52. **Measured reference.** In the fixed seeded run, measure \(\operatorname{RMS}(H_{30}^{(1)})=1.096869\). This is a run-specific measurement, not a universal He constant.
53. **Prediction and check.** Multiply that reference by \(\alpha^{30}\) and compare with direct forward passes: approximately \(1.02\times10^{-9}\), \(1.0969\), and \(2.10\times10^5\).
54. **Depth trace.** Show all thirty activation-RMS values on a logarithmic axis.
55. **Activation-only notebook.** Return all post-ReLU tensors without hooks; reserve hooks for gradient capture.
56. **Layer-30 distributions.** Show that the RMS gap describes the distributions, not one outlier. Each panel has its own horizontal scale.
57. **Active fraction limitation.** Nearly identical gate fractions do not imply healthy magnitudes. Here they rule out widespread gate closure but not scale collapse or explosion.
58. **Softmax worked example.** For the fixed point \((0.429,-0.410)\), transform logits \((-1.313,-0.070,-1.047)\) into probabilities \((0.173,0.601,0.226)\), prediction \(1\), and true-class loss \(1.753\).
59. **Order versus confidence.** Positive scaling preserves the largest logit while changing the gaps, probabilities, and loss. Extremely confident wrong predictions can have huge finite cross-entropy.
60. **Backward trace.** Compare \(\operatorname{RMS}(G_0)\): \(4.57\times10^{-13}\), \(1.01\times10^{-3}\), and \(4.15\times10^2\).
61. **Before versus after 150 updates.** Connect activation scale, logits, loss, backward scale, and final training accuracy. Distinguish update-50 decision maps from update-150 metrics.
62. **Large but finite is unhealthy.** The \(\alpha=1.5\) run begins saturated and often confidently wrong even though it later reaches \(99.4\%\) training accuracy.

The essential chain is:

\[
\text{weight scale}
\longrightarrow
\text{activation scale}
\longrightarrow
\text{logit gaps and loss}
\longrightarrow
\text{backward scale}
\longrightarrow
\text{optimization behaviour}.
\]

### Slides 63–79 — Normalization recomputes scale

63. **Section break.** He sets scale at initialization; normalization addresses scale during training.
64. **Motivation from measured drift.** Compare the same plain \(\alpha=1\) model at initialization and after \(20\) updates.
65. **Four-value feature.** Use one feature column \(Z[:,d]=(1,2,3,4)^\top\).
66. **Centre.** Compute \(\mu_d=2.5\) and obtain \([-1.5,-0.5,0.5,1.5]\).
67. **Scale.** Compute variance \(1.25\), divide by \(\sqrt{1.25}\), and obtain approximately \([-1.342,-0.447,0.447,1.342]\).
68. **Learned affine transform.** With \(\gamma_d=2\), \(\beta_d=1\), show why standardization does not permanently remove representational scale and offset.
69. **General set notation.** Define the entries \(S_{nd}\) that share statistics, then specialize to BatchNorm and LayerNorm.
70. **Orient the matrix.** Rows are examples; columns are hidden features. Ask whether the entry \(1\) should use its row or column.
71. **BatchNorm column.** Reuse the four-value example down feature column one.
72. **LayerNorm row.** Normalize the first example's features \([1,2,6]\) to approximately \([-0.926,-0.463,1.389]\).
73. **Reduction-axis comparison.** Same matrix, different axes.
74. **Batch dependence.** Keep raw value \(1\) fixed; changing companion examples moves its normalized value from \(-1.342\) to \(+1.342\).
75. **Evaluation mode.** Training uses batch statistics and updates running estimates; evaluation uses stored estimates. `model.eval()` does not freeze parameters or disable autograd. LayerNorm does not switch computations this way.
76. **Placement in this experiment.** `Linear → BatchNorm → ReLU`; name \(Z_\ell\), normalized preactivations, and \(H_\ell\).
77. **Predict the post-ReLU RMS.** A symmetric unit-second-moment input followed by ReLU has RMS \(\sqrt{1/2}\approx0.707\).
78. **Depth evidence.** BatchNorm keeps post-ReLU activation RMS near \(0.7\) and avoids the plain \(\alpha=0.5\) backward collapse.
79. **Training repair.** Under the same full-batch controlled setup, BatchNorm reaches \(100\%\) training accuracy.

### Slides 80–91 — Residual blocks add an identity route

80. **Section break.** Move from the scale question to the route question.
81. **Motivation.** Even well-scaled gradients still cross every learned Jacobian in a plain chain.
82. **Identity representation.** For input and target \(3\), a plain scalar layer must learn \(w=1\); \(h+w h\) already copies at \(w=0\).
83. **Residual diagram.** Add a same-shape correction: \(\mathbf h_{\ell+1}=\mathbf h_\ell+F_\ell(\mathbf h_\ell)\).
84. **Scalar derivative.** With \(F(h)=-0.1h\), the local derivative is \(1+F'(h)=0.9\).
85. **Vector derivative.** Expand \((I+J_F)^T\mathbf g\) into an identity term plus a branch term.
86. **Five-block comparison.** Contrast branch-only magnitude \(10^{-5}\) with residual magnitude \(0.59049\). State that this is an abstract parameterization comparison, not a universal ResNet guarantee.
87. **Experimental residual branch.** One \(2\to128\) stem followed by \(29\) blocks \(\mathbf h+\operatorname{ReLU}(V_\ell\mathbf h)\), with no activation after addition.
88. **Controlled hypotheses.** Compare plain, BatchNorm, and residual-toy versions while fixing the spiral data, depth, width, optimizer, seed, and \(\alpha=0.5\) base directions.
89. **Initialization measurements.** BatchNorm controls activation scale; the residual toy preserves a usable backward route but begins with very large activations.
90. **Training results.** After \(150\) updates: plain \(33.3\%\), BatchNorm \(100\%\), residual toy \(98.6\%\).
91. **Decision regions.** Treat these as optimization evidence on the training set. A jagged boundary is not evidence of superior generalization.

### Slides 92–99 — Diagnose before changing Adam

92. **Section break.** Turn the lecture's measurements into a repeatable workflow.
93. **Read one trace.** At \(\alpha=0.5\), activation RMS collapses while aggregate active fraction remains near \(0.5\); the first suspect is insufficient per-layer gain.
94. **One-factor test.** Rescale the same weight directions from \(\alpha=0.5\) to \(1\). Gates stay the same at initialization while activation and gradient scales recover.
95. **Workflow.** Measure \(H_\ell,G_\ell\), locate the first break, name one suspect, change one factor, and repeat the same measurements.
96. **Exit ticket.** Diagnose a trace whose activation RMS declines mildly but active fraction falls from \(0.49\) to \(0.01\).
97. **Answer.** This is progressive gate closure from a negative preactivation shift, not the earlier same-gates scale-collapse pattern.
98. **Synthesis.** Use the same measurements to explain all three \(\alpha\) runs and their final decision regions.
99. **Sources and reproducibility.** Point students to the two Andrew Ng intuition videos, the primary papers, the notebooks, the symmetry lab, and the fixed-seed evidence directory.

### Slides 100–104 — Optional backup

100. **Unequal input units.** Initialization assumes comparable order-one input scales; otherwise one raw feature can dominate the first layer.
101. **Saturated sigmoid gates.** Ten gates at \(z=5\) yield a gate-only multiplier near \(1.7\times10^{-22}\). This is about repeated saturated hidden gates, not about banning sigmoid outputs.
102. **PyTorch initialization.** Apply Kaiming initialization only to the hidden ReLU layers, guard optional biases, and initialize the final classifier separately.
103. **Diagnostic pattern: gates close.** Usable RMS plus active fraction approaching zero points to a negative preactivation shift.
104. **Diagnostic pattern: non-finite layer.** Locate the first overflow before changing the optimizer.

---

## Worked examples and numbers that must remain consistent

| Example | Fixed result | Why it is present |
|---|---:|---|
| Four scalar forward multipliers | \(0.5^4=0.0625\), \(1^4=1\), \(1.5^4=5.0625\) | Make exponential depth effects visible before matrices |
| Thirty backward gains | \(0.9^{30}=0.042\), \(1.1^{30}=17.45\) | Show that a 10% local mismatch is not small at depth 30 |
| Gradient vector | RMS of \((2,-2,1,-1)\) is \(\sqrt{2.5}\approx1.58\) | Motivate RMS despite sign cancellation |
| Two independent \(\pm1\) terms | RMS of their sum is \(\sqrt2\) | Show why squared sizes add |
| One hundred unit-RMS terms | sum RMS \(=10\) | Motivate fan-in scaling |
| Fan-in 100 linear scaling | \(\operatorname{Std}(w)=0.1\) | Preserve affine output RMS |
| Fan-in 100 He scaling | \(\sqrt{2/100}\approx0.141\) | Compensate for ReLU's half-second-moment effect |
| Fan-in 128 He scaling | \(0.125\) | Connect the derivation to the experiment |
| Layer-30 reference | \(\operatorname{RMS}(H_{30}^{(1)})=1.096869\) | Calibrate the run-specific absolute scale |
| Layer-30 scaled predictions | \(1.02\times10^{-9}\), \(1.0969\), \(2.10\times10^5\) | Check the \(\alpha^{30}\) prediction |
| Fixed-point softmax | probabilities \((0.173,0.601,0.226)\), loss \(1.753\) | Connect logits to confidence and loss |
| Four-value normalization | standardized values \((-1.342,-0.447,0.447,1.342)\) | Make normalization concrete before notation |
| Post-BN ReLU RMS | \(\sqrt{1/2}\approx0.707\) | Predict the measured depth trace |
| Five residual factors | branch-only \(10^{-5}\), residual \(0.59049\) | Expose the identity term's effect |

If a plotted value changes after regenerating the fixed-seed experiment, update the worked table, captions, and guide together. Do not mix measurements from different seeds or checkpoints.

---

## Notebook and interactive roles

### `00_pytorch_hooks_for_layerwise_diagnostics.ipynb`

Use this as the separate tutorial on PyTorch hooks. It should explain:

- what a forward hook receives;
- how to retain gradients for non-leaf activation tensors;
- how to capture \(H_\ell\) during the forward pass and \(G_\ell\) after backpropagation;
- how and when to remove hook handles; and
- why hooks are an observation mechanism, not part of the model's mathematics.

This notebook supports slide 46 and the gradient diagnostics. Do not compress the full hook implementation onto a lecture slide.

### `01_activation_scale_through_depth.ipynb`

This is the minimal activation-only notebook. It should:

- construct the fixed 30-hidden-layer spiral MLP;
- return every post-ReLU hidden tensor directly from `forward`;
- compare \(\alpha=0.5,1,1.5\) under `torch.no_grad()`;
- reproduce the layerwise activation-RMS plot and layer-30 values; and
- make the \(\alpha^\ell\) prediction inspectable with little framework machinery.

Use this notebook for slides 52–55.

### `deep_network_trainability_autopsy.ipynb`

This is the end-to-end experiment. It should reproduce:

- the three \(\alpha\) runs;
- activation, gate, finiteness, and gradient traces;
- logits, probabilities, losses, and training outcomes;
- BatchNorm and residual-toy comparisons; and
- the fixed-seed figures and evidence tables used by the deck.

### Hidden-unit symmetry lab

Use the linked interactive on slide 16 to let students compare exact clones, perturbed clones, and independent starts. The visual question is whether hinge lines, hidden-feature rank, and decision surfaces separate.

---

## Presenter prompts

Use prediction questions before revealing calculations:

- After slide 3: "If each layer multiplies by \(0.9\), what happens after many layers?"
- On slide 24: "Do 100 independent unit-sized terms produce a typical sum of 1, 10, or 100?"
- On slide 35: "One run is stuck and two learn. What single setting might produce all three?"
- On slide 37: "What do you think \(\alpha\) controls?"
- Before slide 53: "Given the measured \(\alpha=1\) value, can we predict the other two without another theory?"
- On slide 57: "If half the ReLUs are active, can the signal still vanish?"
- Before slide 59: "If every logit is multiplied by a positive constant, can the predicted class change? Can confidence change?"
- On slide 70: "For the entry \(1\), should normalization use its row or its column?"
- Before slide 77: "What RMS should remain after ReLU clips half a symmetric unit-second-moment signal?"
- On slide 82: "Which parameterization represents identity at zero branch weights?"
- On slide 96: "Is this trace primarily a scale problem or a gate problem? What evidence distinguishes them?"

Let students commit to a prediction before showing the result. The reveal should answer the exact question just posed.

---

## Common misconceptions to intercept

- **"Adam fixes vanishing gradients."** Adam rescales observed coordinate histories; it cannot reconstruct a zero, underflowed, or missing gradient path.
- **"A 50% active fraction means the layer is healthy."** It only argues against widespread gate closure. Magnitudes may still collapse or explode.
- **"Every negative ReLU is dead."** A neuron is dead on the training set only when all examples close it.
- **"He initialization makes every layer's measured RMS exactly one."** It preserves a second moment under assumptions in expectation at initialization; one finite seeded network can differ.
- **"Variances and standard deviations both add."** Independent variances add; RMS or standard deviation grows like the square root of the number of terms.
- **"Positive scaling changes the predicted class."** It preserves logit order but changes softmax confidence and loss.
- **"BatchNorm and LayerNorm use the same values."** They use the same formula on different axes.
- **"`model.eval()` turns gradients off."** It changes modules such as BatchNorm and Dropout; it does not disable autograd.
- **"Residual connections guarantee stable gradients."** They contribute an identity term, but the branch can still amplify or cancel directions.
- **"A jagged training decision boundary proves better generalization."** The deck's plots are trainability evidence on the training set.

---

## Delivery and cut plan

For a full delivery, teach slides 1–99 in order. The deck is deliberately granular: several slides perform one step of one calculation so the class never has to decode a finished derivation all at once.

If time is tight, preserve the conceptual chain and cut detail, not motivation:

- keep slides 3–10, 12–15, 17–21, 24–32, 34–37, 44–54, 57–62, 64–79, 81–91, and 93–98;
- make slides 38–43 a quick notation check if the class already knows PyTorch matrix shapes;
- demonstrate only the relevant notebook cell during class and assign the hooks tutorial for follow-up;
- keep slides 100–104 as backup only.

Never cut all of the scalar examples and retain only the formulas. Never introduce BatchNorm or residual connections as isolated recipes; each must answer the measured failure developed earlier.

---

## Final audit checklist

Before publishing or teaching:

- [ ] The handout has 104 slides and slides 100–104 are clearly optional.
- [ ] Every occurrence of "30 layers" means **30 hidden layers**; the logits are produced by layer 31.
- [ ] \(h_\ell\), \(\mathbf h_\ell\), \(H_\ell\), and \(G_\ell\) are not interchanged.
- [ ] Preactivations use \(z\) or \(Z\); post-ReLU activations use \(h\) or \(H\).
- [ ] Weight shapes remain \((d_{\text{out}},d_{\text{in}})\), matching PyTorch.
- [ ] Fan-in is read along a receiving row; fan-out is read down an outgoing column.
- [ ] Affine gain, full-layer gain, and backward gain are named separately.
- [ ] The layer-30 RMS \(1.096869\) is labelled as a measured seeded value.
- [ ] Update-50 decision maps are not described as update-150 results.
- [ ] \(G_0\) is identified as the input-activation gradient, not a parameter gradient.
- [ ] BatchNorm's training and evaluation behaviour is stated correctly.
- [ ] Residual claims retain the toy-model and matching-shape qualifications.
- [ ] Node labels fit inside circles; arrows, labels, and mathematical expressions do not overlap.
- [ ] Tables use consistent precision and align numeric columns.
- [ ] Plots identify logarithmic axes and checkpoint numbers.
- [ ] Every external notebook link matches a file in `notebooks/L06/`.
- [ ] All figures can be regenerated from the fixed-seed experiment.

---

## One-sentence takeaway

> Deep networks become trainable when we keep forward and backward signals measurable, initialize distinct features at an appropriate scale, provide mechanisms that control drift or shorten fragile routes, and verify the diagnosis layer by layer.
