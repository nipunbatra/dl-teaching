// Optional algebra; main Lecture 2 uses a computational graph and autograd.
#import "../../common/metropolis.typ": *
#import "../../common/mldiag.typ": *
#import "helpers.typ": *

= Optional calculations

#[
== Appendix: differentiate the softmax mixture

#set text(size: 16pt)
#align(center)[$alpha_j = exp(s_j) / (sum_l exp(s_l)), quad h = sum_j alpha_j v_j$]
#v(8pt)
#align(center, text(size: 13pt)[$Delta e = h W_O, quad e' = e + Delta e, quad g_h = g_(e') W_O^top$])
#v(8pt)
#align(center)[$(partial alpha_j)/(partial s_l) = alpha_j (delta_(j l) - alpha_l)$]
#v(9pt)
#align(center)[$g_(s_j) = alpha_j (g_(alpha_j) - sum_l alpha_l g_(alpha_l))$]
#v(12pt)
#note(color: BLUE)[Substitute $g_(alpha_j)=g_h dot v_j$. Since $sum_l alpha_l v_l=h$, the whole Jacobian reduces to:]
#align(center, text(fill: BLUE)[$g_(s_j) = alpha_j g_h dot (v_j - h)$])
#v(10pt)
#align(center)[$g_q = sum_j g_(s_j) k_j / sqrt(d_k), quad g_(k_j) = g_(s_j) q / sqrt(d_k)$]
#v(8pt)
#align(center, text(size: 12pt, fill: MUTED)[Row vectors; $e,q,h$ omit the fixed receiver's index. $g$ is a loss gradient; lowercase $delta$ is an index indicator, not the update $Delta e$.])

== Appendix: the matrix backward pass

#set text(size: 13pt)
#align(center, text(size: 12pt, fill: MUTED)[$X$ stacks input embeddings; $X'$ stacks updated embeddings. $P,Y$: $T times 8$ probabilities / one-hot targets; $T=6$.])
#v(4pt)
#align(center)[$H=A V, quad Delta X = H W_O, quad X' = X + Delta X, quad Z = X' U + b$]
#v(4pt)
#grid(columns: (1fr, 1fr), gutter: 25pt,
  hairline([Prediction and residual addition], [
    #stack(dir: ttb, spacing: 11pt,
      [$g_Z = (P - Y) ÷ T$],
      [$g_U = (X')^top g_Z, quad g_b = sum_i (g_Z)_(i,:)$],
      [$g_(X') = g_Z U^top$],
      [$g_(Delta X) = g_(X')$],
      [$g_(W_O) = H^top g_(Delta X)$],
      [$g_H = g_(Delta X) W_O^top$],
    )
  ], color: ACC),
  hairline([Mixture, scores, and projections], [
    #stack(dir: ttb, spacing: 11pt,
      [$g_V = A^top g_H, quad g_A = g_H V^top$],
      [Row-wise softmax backward gives $g_S$.],
      [Masked score gradients are zero.],
      [$g_Q = (g_S K) ÷ sqrt(d_k)$],
      [$g_K = (g_S^top Q) ÷ sqrt(d_k)$],
      [$g_(W_Q)=X^top g_Q$; similarly for K, V.],
    )
  ], color: BLUE),
)
#v(5pt)
#align(center)[$g_X = g_(X') + g_Q W_Q^top + g_K W_K^top + g_V W_V^top$]
#align(center, text(size: 12pt, fill: MUTED)[Scatter-add $g_X$ into the matching token rows of $E$ and position rows of $R$. Shared parameters accumulate gradients.])

== Appendix: what does the dot product notice?

#set text(size: 16pt)
#grid(columns: (1.2fr, 1fr), gutter: 18pt, align: horizon,
  [#image("../figures/dot_product_alignment_imagen_v2.png", width: 100%, height: 73mm, fit: "contain")],
  [
    $q dot k = norm(q) norm(k) cos(theta)$
    #v(15pt)
    Direction and magnitude both matter.
    #v(14pt)
    $q = mat(1,0), quad k = mat(0,1)$\
    $q dot k = 0$
    #v(12pt)
    $q = mat(1,0), quad k = mat(2,0)$\
    $q dot k = 2$
  ],
)
#v(13pt)
#note(color: BLUE)[A token's query and key are different projections. Its own key need not receive the largest score, even when vectors are normalized.]
]
