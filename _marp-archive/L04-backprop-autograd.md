---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# Backpropagation & Autograd — Build It

## Lecture 4 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar*

---

# The whole lecture, in one idea

Training a network means computing **one gradient** — of the loss with respect to *millions* of parameters. Backpropagation computes **all of them in a single backward pass**, and it is nothing more than the **chain rule applied to a graph**.

<div class="keypoint">

**Backprop = chain rule on a computational graph.**
Each node multiplies the gradient flowing *into* it by its own **local derivative** and passes it on. One **forward** pass to compute the loss; one **backward** pass to differentiate — for *every* parameter at once.

</div>

You used `loss.backward()` in ES 335 and never asked what it did. Today we answer that by **building the engine ourselves** — a scalar autograd library, in about forty lines of Python.

$$\text{graph} \rightarrow \underbrace{\text{forward}}_{\text{compute }L} \rightarrow \underbrace{\text{local rules}}_{\text{per node}} \rightarrow \underbrace{\text{backward}}_{\text{all grads}} \rightarrow \underbrace{\text{step}}_{\text{5-line loop}}$$

---

# How today connects to the rest of the course

<div class="columns">
<div>

**You'll reuse today's engine in:**
- **L2–L3** — the linear / logistic / neural nets whose parameters we differentiate
- **L5** — SGD & optimizers *take the step* that `.backward()` sets up
- **every lecture after** — every model in this course is trained by exactly this loop
- the **init** coda motivates the careful initialization we lean on all semester

</div>
<div>

**Borrowed from your own courses** (we build on, not repeat):
- the **chain rule** — calculus
- **matrix-calculus backprop** (the $\delta$ rule) — *Neural Networks* tutorial, ES 335
- **PyTorch autograd** — ES 335 labs

</div>
</div>

<div class="insight">

This lecture is spiritually **Karpathy's `micrograd`**: the fastest way to *believe* autograd is to write the twenty lines that do it.

</div>

---

<!-- _class: section-divider -->

## Part 1 · The chain rule on a computational graph

---

# Where every gradient comes from: the chain rule

A network is a **composition** of simple functions. If $L = f(g(x))$, the chain rule says the derivative is a **product** of local derivatives along the path:

$$\frac{dL}{dx} = \frac{dL}{dg}\cdot\frac{dg}{dx}$$

![w:560px](figures/lec02/svg/chain_rule_product.svg)

<div class="insight">

That's the entire mathematical content of backprop. The *engineering* is: how do we compute this product **once**, reusing shared work, for a graph with millions of nodes? Answer — a computational graph plus one bookkeeping rule.

</div>

---

# Any computation is a graph

Break the expression into **elementary operations**; each becomes a **node**, each intermediate value an edge. Take $L = x\cdot y + b$ with $x=2,\ y=-3,\ b=10$:

![w:640px](figures/lec01/svg/computational_graph.svg)

$$m = x\cdot y = -6, \qquad L = m + b = 4$$

Left-to-right along the arrows is the **forward pass** — just evaluate. The graph is the data structure backprop walks *backward*.

---

# The one rule the backward pass obeys

At each node, the gradient of the loss w.r.t. that node's **input** is:

<div class="math-box">

$$\underbrace{\frac{\partial L}{\partial \text{input}}}_{\text{downstream}} \;=\; \underbrace{\frac{\partial \text{output}}{\partial \text{input}}}_{\text{local, known from the op}} \;\times\; \underbrace{\frac{\partial L}{\partial \text{output}}}_{\text{upstream, handed down}}$$

</div>

**downstream grad = local grad × upstream grad.** A node needs to know *nothing* about the rest of the graph — only its own operation and the number handed to it from above. That locality is what makes autograd composable.

---

# Every node is a valve on the gradient

The rule **downstream = local × upstream** turns each node into a tiny **valve**: it takes the flow handed down from above and multiplies by its own local derivative before passing it on.

- **`+` node** — local derivative $=1$: the valve is wide open, so the gradient passes through **unchanged** to each input. A `+` node is a pure **router**.
- **`×` node** — local derivative $=$ the *other* input: the valve is set to that value, so the gradient is **swapped and scaled**.
- **saturated `tanh`** — local derivative $1-t^2\approx 0$: the valve is nearly **shut**, and almost no gradient gets through.

<div class="insight">

That's the whole of backprop in one metaphor. The gradient at an early node is the **product of every valve setting** along the path to the loss — open valves ($+$) pass it on, near-shut valves (saturated squashes) choke it. Part 4 is what happens when too many valves sit almost closed.

</div>

---

# Backward pass: fill in the tiny graph

Seed the output with $\dfrac{\partial L}{\partial L}=1$, then walk backward applying the rule.

<div class="columns">
<div>

**Through the $+$ node** ($L=m+b$):
local derivatives are both $1$, so grad just **copies**:
$$\frac{\partial L}{\partial m}=1,\qquad \frac{\partial L}{\partial b}=1$$

</div>
<div>

**Through the $\times$ node** ($m=x\cdot y$):
local derivatives **swap** the inputs:
$$\frac{\partial L}{\partial x}=y\cdot 1=-3,\quad \frac{\partial L}{\partial y}=x\cdot 1=2$$

</div>
</div>

<div class="insight">

Read off: nudging $x$ up by $\varepsilon$ changes $L$ by $-3\varepsilon$. Sign and size both fall straight out of two local rules — no giant symbolic derivative.

</div>

---

# A pocket table of local gradients

Every op you need is one line. `out` is the node; the rule says what to **add** to each input's grad.

| Operation | Forward | Local rule (what each input's grad gets) |
|---|---|---|
| add | $c=a+b$ | $a\!:\,\texttt{out.grad}$ · $b\!:\,\texttt{out.grad}$ |
| mul | $c=a\cdot b$ | $a\!:\,b\cdot\texttt{out.grad}$ · $b\!:\,a\cdot\texttt{out.grad}$ |
| pow | $c=a^{k}$ | $a\!:\,k\,a^{k-1}\cdot\texttt{out.grad}$ |
| tanh | $c=\tanh a$ | $a\!:\,(1-c^{2})\cdot\texttt{out.grad}$ |
| relu | $c=\max(0,a)$ | $a\!:\,[\,c>0\,]\cdot\texttt{out.grad}$ |

<div class="keypoint">

**"Add" distributes the gradient; "multiply" swaps and scales.** Memorize those two — they cover most of every network you'll ever train.

</div>

---

# What a gradient actually tells you

A gradient is a **local linear model** of the loss: near the current values, $\;L \approx L_0 + \dfrac{\partial L}{\partial x}\,\Delta x$.

<div class="columns">
<div>

**Sign** — which way to nudge the input to *raise* $L$. From the tiny graph, $\partial L/\partial x = -3$: push $x$ **up** and $L$ goes **down**. So descend by stepping *against* the sign.

</div>
<div>

**Magnitude** — sensitivity. $\partial L/\partial b = 1$ moves $L$ gently; a grad of $-24$ (Practice 1) moves it hard. Big grad → this knob matters a lot *right now*.

</div>
</div>

<div class="keypoint">

Backprop hands you this sign-and-size for **every** parameter at once. Gradient descent (L5) then does the only obvious thing with it: take a small step **downhill**, $\;\theta \leftarrow \theta - \eta\,\partial L/\partial\theta$, and repeat.

</div>

---

# Gradients *accumulate* at a fan-out

What if one node feeds **two** downstream paths? Suppose $\theta$ is used twice: $L = \theta x_1 + \theta x_2$.

<div class="columns">
<div>

Each path sends back its own contribution, and the total derivative is their **sum** (multivariable chain rule):

$$\frac{\partial L}{\partial \theta}=x_1 + x_2$$

</div>
<div>

So the backward rule must be `+=`, never `=`:

$$\texttt{a.grad += local * out.grad}$$

</div>
</div>

<div class="warning">

This `+=` is a **feature** inside one graph — and a **bug waiting to happen** across training steps. Hold that thought; it's the whole reason `zero_grad()` exists (Part 3).

</div>

---

# Practice problem 1

<div class="popquiz">

**Practice problem 1.** Let $L=(a\cdot b + c)^2$ with $a=2,\ b=-3,\ c=10$. Draw the graph, do the **forward pass**, then backprop by hand to find $\dfrac{\partial L}{\partial a}$, $\dfrac{\partial L}{\partial b}$, $\dfrac{\partial L}{\partial c}$.

</div>

*Try it before the next slide — seed with $\partial L/\partial L = 1$ and walk backward node by node.*

---

# Solution · practice problem 1

**Forward:** $\;u=a b=-6,\quad v=u+c=4,\quad L=v^2=16.$

**Backward**, one node at a time:

$$\frac{\partial L}{\partial v}=2v=8 \;\xrightarrow{\;+\;}\; \frac{\partial L}{\partial u}=8,\;\; \frac{\partial L}{\partial c}=8 \;\xrightarrow{\;\times\;}\; \frac{\partial L}{\partial a}=b\cdot 8=-24,\;\; \frac{\partial L}{\partial b}=a\cdot 8=16$$

<div class="keypoint">

$$\boxed{\ \frac{\partial L}{\partial a}=-24,\qquad \frac{\partial L}{\partial b}=16,\qquad \frac{\partial L}{\partial c}=8\ }$$

The $\times$ node swaps ($a$'s grad uses $b$), the $+$ node copies, the $\text{pow}$ node scales by $2v$. We'll reproduce these exact numbers **in code** at the end of Part 2.

</div>

---

<!-- _class: section-divider -->

## Part 2 · Build micrograd from scratch

---

# Three ways to get a derivative

<div class="columns">
<div>

**Symbolic** — differentiate the *formula* (by hand, SymPy) into an exact expression. Blows up in size for deep nets and needs a closed form.

**Numerical** — finite differences $\tfrac{f(x+h)-f(x-h)}{2h}$. Trivial to code, but fragile and $\mathcal O(P)$ (Part 3).

</div>
<div>

**Automatic differentiation** — apply the chain rule to the *program's* elementary ops as they run. Exact to floating point, and one backward pass gives **all** gradients.

</div>
</div>

<div class="keypoint">

Autograd is **neither** symbolic **nor** numerical — it's the chain rule executed over the computation graph, op by op. That third way is what we build next, and what PyTorch does under the hood.

</div>

---

# The plan: a scalar autograd engine

<div class="build-it">

We'll write a class `Value` that wraps a single number and, as you compute with it, **records the graph** and knows how to **push gradients backward**. Build order:
$$\texttt{+}\;\rightarrow\;\texttt{*}\;\rightarrow\;\text{local grads}\;\rightarrow\;\texttt{backward()}\;\rightarrow\;\texttt{tanh/relu}$$

</div>

<div class="insight">

This is a faithful rebuild of **Andrej Karpathy's `micrograd`** (MIT-licensed, ~150 lines). PyTorch is the same idea, just on tensors instead of scalars and in C++.

</div>

---

<!-- _class: code-heavy -->

# The `Value` node

A node stores its number, its gradient slot, and how it was made.

```python
class Value:
    """A single scalar node in the computation graph."""

    def __init__(self, data, _children=(), _op=''):
        self.data = data                 # the number itself
        self.grad = 0.0                  # dL/d(this) — filled on the backward pass
        self._prev = set(_children)      # the nodes that produced this one
        self._op = _op                   # which op made it (for debugging / drawing)
        self._backward = lambda: None    # how to push grad to _prev (set per-op)

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
```

`grad` starts at $0$; `_prev` and `_op` are the **edges and labels** of the graph. `_backward` is a *closure* we'll attach when we build each operation.

---

<!-- _class: code-heavy -->

# Overload `+` and `*` — build the graph (forward only)

Operator overloading lets `a * b + c` **construct the graph** as a side effect of ordinary arithmetic.

```python
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        return out
```

```python
a, b, c = Value(2.0), Value(-3.0), Value(10.0)
L = a * b + c            # L.data == 4.0, and L._prev remembers (a*b, c)
```

The `isinstance` guard lets us mix in plain Python numbers (`a * 2`). No gradients yet — just wiring.

---

<!-- _class: code-heavy -->

# Attach the local gradients: `_backward` closures

Each op knows its own local rule. We store it as a closure that reads `out.grad` (upstream) and **accumulates** into its inputs.

```python
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad  += out.grad            # + copies grad to both parents
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad  += other.data * out.grad   # * swaps: self gets other's value
            other.grad += self.data  * out.grad
        out._backward = _backward
        return out
```

The closure **captures** `self`, `other`, and `out` — so calling `out._backward()` later still has everything it needs.

---

<!-- _class: code-heavy -->

# Why we can't just call `_backward` in any order

Call them by hand for $L = a\cdot b + c$ and the ordering problem appears:

```python
a, b, c = Value(2.0), Value(-3.0), Value(10.0)
m = a * b          # the multiply node
L = m + c

L.grad = 1.0       # seed
L._backward()      # fills m.grad = 1, c.grad = 1   -- MUST happen first
m._backward()      # now uses m.grad to fill a.grad, b.grad
print(a.grad, b.grad, c.grad)   # -3.0  2.0  1.0
```

<div class="warning">

If we called `m._backward()` **before** `L._backward()`, `m.grad` would still be $0$ and `a`, `b` would get zero gradient. A node can only push grad once its *own* grad is final — i.e. after every node that consumes it. We need a **topological order**.

</div>

---

# Practice problem 2

<div class="popquiz">

**Practice problem 2.** Fill in the `_backward` for the **multiply** op:

```python
def __mul__(self, other):
    out = Value(self.data * other.data, (self, other), '*')
    def _backward():
        self.grad  += ______ * out.grad
        other.grad += ______ * out.grad
    out._backward = _backward
    return out
```

What goes in the blanks, and *why is it `+=` and not `=`?*

</div>

*Try it before the next slide.*

---

# Solution · practice problem 2

For $c = a\cdot b$, the local derivatives are $\dfrac{\partial c}{\partial a}=b$ and $\dfrac{\partial c}{\partial b}=a$ — the inputs **swap**:

```python
        self.grad  += other.data * out.grad     # dL/da = b * (dL/dc)
        other.grad += self.data  * out.grad     # dL/db = a * (dL/dc)
```

<div class="keypoint">

**Why `+=`?** A node may feed several downstream paths. Each path calls this closure once and adds its share; the total gradient is the **sum** over paths (the multivariable chain rule from Part 1). Starting each node's `grad` at $0$ and accumulating is exactly right — *provided* we reset between training steps.

</div>

---

<!-- _class: code-heavy -->

# `backward()`: topological sort, then reverse

One method does the whole graph: order the nodes so every node comes **after** its children, seed the output, and replay `_backward` in reverse.

```python
    def backward(self):
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)          # child appended before its parent
        build(self)

        self.grad = 1.0                 # dL/dL = 1, the seed
        for v in reversed(topo):        # parents before children -> grads ready
            v._backward()
```

```python
L = a * b + c
L.backward()          # one call fills a.grad, b.grad, c.grad correctly
```

The DFS post-order guarantees a node's grad is complete before we ask it to push. **That's the whole algorithm.**

---

<!-- _class: code-heavy -->

# Add `pow` and division — for free via existing ops

`__pow__` needs its own rule; division is just multiplication by a reciprocal power, so it **reuses** `__mul__` and `__pow__`.

```python
    def __pow__(self, k):
        assert isinstance(k, (int, float)), "only int/float powers"
        out = Value(self.data ** k, (self,), f'**{k}')
        def _backward():
            self.grad += (k * self.data ** (k - 1)) * out.grad   # d/da a^k
        out._backward = _backward
        return out

    def __truediv__(self, other):     # a / b  ==  a * b**-1
        return self * other ** -1
```

<div class="insight">

Build a small set of primitives with correct local rules, and everything else — subtraction, division, squaring — **composes** from them. The graph, and its gradients, extend automatically.

</div>

---

<!-- _class: code-heavy -->

# A nonlinearity: `tanh`

Without a nonlinear op, the whole graph collapses to one linear map (L2). `tanh` adds one node with a clean local derivative $1-\tanh^2$.

```python
    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad     # d/dx tanh(x) = 1 - tanh(x)^2
        out._backward = _backward
        return out
```

<div class="warning">

Notice $1-t^2 \le 1$, and it's near $0$ when $|x|$ is large (the flat tails). Stack many `tanh` layers and these sub-1 factors **multiply** — the seed of the vanishing-gradient problem in Part 4.

</div>

---

<!-- _class: code-heavy -->

# Another nonlinearity: `relu`

`relu` is a **gate**: pass the value if positive, else $0$. Its gradient is $1$ where active, $0$ where dead — no shrinking factor.

```python
    def relu(self):
        out = Value(max(0.0, self.data), (self,), 'ReLU')
        def _backward():
            self.grad += (out.data > 0) * out.grad     # 1 if active, else 0
        out._backward = _backward
        return out
```

<div class="insight">

Because its live gradient is exactly $1$ (not $<1$), `relu` lets gradient flow through deep stacks far better than `tanh`/sigmoid. That single property is a big part of why deep nets became trainable — we'll make it quantitative in Part 4.

</div>

---

<!-- _class: code-heavy -->

# Run the whole engine — and reproduce Practice Problem 1

Forty lines of `Value` now differentiate arbitrary scalar expressions:

```python
a, b, c = Value(2.0), Value(-3.0), Value(10.0)
L = (a * b + c) ** 2          # exactly Practice Problem 1
L.backward()

print(L.data)                 # 16.0
print(a.grad, b.grad, c.grad) # -24.0  16.0  8.0   <- matches our hand computation
```

<div class="keypoint">

The engine gets $(-24,\,16,\,8)$ — **identical** to the by-hand answer. It never saw a formula for $\partial L/\partial a$; it just chained local rules through the graph. Swap in `.tanh()`/`.relu()` and it still works, unchanged.

</div>

---

<!-- _class: code-heavy -->

# The whole engine, on one slide

Everything so far — the roughly forty lines that differentiate *any* scalar expression:

```python
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data, self.grad = data, 0.0
        self._prev, self._op = set(_children), _op
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward(): self.grad += out.grad; other.grad += out.grad
        out._backward = _backward; return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad  += other.data * out.grad
            other.grad += self.data  * out.grad
        out._backward = _backward; return out

    def __pow__(self, k):
        out = Value(self.data ** k, (self,), f'**{k}')
        def _backward(): self.grad += (k * self.data ** (k-1)) * out.grad
        out._backward = _backward; return out

    def tanh(self):
        t = math.tanh(self.data); out = Value(t, (self,), 'tanh')
        def _backward(): self.grad += (1 - t*t) * out.grad
        out._backward = _backward; return out

    def backward(self):
        topo, seen = [], set()
        def build(v):
            if v not in seen:
                seen.add(v)
                for c in v._prev: build(c)
                topo.append(v)
        build(self); self.grad = 1.0
        for v in reversed(topo): v._backward()
```

<div class="keypoint">

This is **all of it.** `relu`, `exp`, division, neurons, whole MLPs — every one **composes** on top of these primitives. PyTorch adds tensors, a C++ backend, and a GPU; the *algorithm* on this slide does not change.

</div>

---

<!-- _class: code-heavy -->

# From a `Value` to a neuron — the payoff

A neuron is $w\cdot x + b$ then a squash. Built from `Value`, its parameters get gradients **for free**.

```python
import random

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)
    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()          # w·x + b, then nonlinearity
```

<div class="insight">

Stack `Neuron`s into layers, layers into an MLP, define a loss — one `loss.backward()` fills **every** `w.grad` and `b.grad` in the network. This is a working neural net trained by the engine we just wrote. (Full `Layer`/`MLP` in the notebook.)

</div>

---

<!-- _class: code-heavy -->

# Train a neuron on 4 points — end to end

No PyTorch. The `Value` engine, a `Neuron`, and a hand-written SGD loop — a net that actually learns:

```python
xs = [[2.0, 3.0], [3.0, -1.0], [-1.0, -2.0], [-2.0, 1.0]]   # 4 tiny 2-D points
ys = [1.0, -1.0, -1.0, 1.0]                                  # targets in {-1, +1}

n = Neuron(2)                          # w1, w2, b are Values; params() -> [w1, w2, b]
for step in range(100):
    preds = [n(x) for x in xs]                          # forward: tanh(w·x + b)
    loss  = sum((p - y)**2 for p, y in zip(preds, ys))  # MSE, built from Values

    for p in n.params(): p.grad = 0.0   # zero_grad   (Practice 3!)
    loss.backward()                     # fill every w.grad, b.grad in ONE pass
    for p in n.params(): p.data += -0.05 * p.grad        # SGD step (L5)

    if step % 20 == 0: print(step, loss.data)            # loss falls: 3.1 -> 0.2 -> ...
```

<div class="insight">

That inner block is the **five-line loop** from L1 — running on the engine *we wrote*, gradients and all. Add a `params()` returning `self.w + [self.b]`, stack `Neuron`s into an `MLP`, and this exact loop trains the whole network. *(After Karpathy's `micrograd` demo.)*

</div>

---

# Explore & rebuild it yourself

<div class="notebook">

**📓 Notebook · autograd from scratch** — the `Value` class built up cell by cell (`+`, `*`, `pow`, `tanh`, `sigmoid`, `backward`), with `draw_dot` to render the graph and its gradients. *(ML ES 335 · `notebooks/autograd-from-scratch.ipynb`, N. Batra)*

</div>

<div class="notebook">

**🎬 Video · "The spelled-out intro to backpropagation / Let's build micrograd"** — A. Karpathy walks the entire build in one sitting; the source `micrograd` repo is ~150 lines. *(github.com/karpathy/micrograd, MIT)*

</div>

<div class="notebook">

**🎛 Interactive · watch gradients flow through a graph** — a scroll-driven explainer where you seed the output and step the backward pass node by node, watching each `.grad` fill in. *(Interactive Lab · `~/git/interactive/src/articles/autograd`)*

</div>

---

<!-- _class: section-divider -->

## Part 3 · Reverse-mode autodiff & PyTorch

---

# What we built has a name: reverse-mode AD

We recorded operations on a **tape** (the graph) during the forward pass, then replayed it backward. That's **reverse-mode automatic differentiation** — and it's exactly what PyTorch's autograd does.

![w:820px](figures/lec03/svg/autograd_tape.svg)

<div class="keypoint">

**One backward pass yields the gradient w.r.t. every input at once** — at roughly the cost of *one* forward pass, no matter how many parameters. That efficiency is the reason training billion-parameter nets is even thinkable.

</div>

---

# Two directions to differentiate — pick the cheap one

Autodiff can sweep the graph either way. The **shape** of the problem decides which is cheap.

<div class="columns">
<div>

**Forward mode** — start at an **input**, push its influence forward to every output. One sweep per *input*.

</div>
<div>

**Reverse mode** — start at the **output**, pull its sensitivity back to every input. One sweep per *output*.

</div>
</div>

A neural net is **wide in, thin out**: millions of inputs (the parameters), a **single** scalar output (the loss).

<div class="keypoint">

Wide in, thin out $\Rightarrow$ go **backward**: one reverse sweep delivers all million gradients, where forward mode would need a million sweeps. That single shape fact is *why* it's called **back**prop. *(The rigorous $\propto n$ vs $\propto m$ cost is in the appendix.)*

</div>

---

<!-- _class: code-heavy -->

# PyTorch does *exactly* this

Set `requires_grad=True` to mark parameters; PyTorch builds the same graph on the fly and `.backward()` runs reverse-mode AD.

```python
import torch

x = torch.tensor(2.0)                        # input, no grad needed
w = torch.tensor(-3.0, requires_grad=True)   # a parameter
b = torch.tensor(10.0, requires_grad=True)   # a parameter

L = (w * x + b) ** 2      # same graph, recorded automatically
L.backward()              # reverse-mode autodiff

print(w.grad, b.grad)     # tensor(16.) tensor(8.)  -- matches our engine's nodes
```

<div class="insight">

`w.grad = 16`, `b.grad = 8` — the very gradients our `Value` engine produced for the corresponding nodes. Same algorithm; PyTorch just runs it on tensors, fused and on the GPU.

</div>

---

# PyTorch is just this, vectorized

Going from our `Value` to `torch.Tensor` changes the **data type**, not the **algorithm**:

<div class="columns">
<div>

**micrograd**
- a node holds one **scalar**
- `_backward` is a single **multiply** by the local derivative
- seed `dL/dL = 1`, replay in topo order, accumulate with `+=`

</div>
<div>

**PyTorch**
- a node holds a **tensor**
- `_backward` is a **Jacobian–vector product** (a matmul)
- seed, topo order, `+=` — **identical**

</div>
</div>

<div class="keypoint">

`+` still routes; a matmul is just `*` that "swaps" with the other operand **transposed**. Every `nn` layer registers a local Jacobian rule exactly like our `_backward` closures. Understand the 25-line engine and you understand `torch.autograd` — everything else is speed.

</div>

---

<!-- _class: code-heavy -->

# Why not just use numerical differentiation? (1/2 — precision)

The finite-difference definition $f'(x)\approx\dfrac{f(x+h)-f(x-h)}{2h}$ *looks* free. But which $h$?

```python
def f(x): return 3*x**2 + 2*x + 1          # true f'(2) = 14
for h in [1e-1, 1e-4, 1e-7, 1e-10, 1e-13]:
    print(h, (f(2+h) - f(2-h)) / (2*h))
```

| $h$ | estimate of $f'(2)$ |
|---|---|
| $10^{-1}$ | $14.0000$ |
| $10^{-7}$ | $14.0000$ |
| $10^{-10}$ | $14.0002$ |
| $10^{-13}$ | $13.98\ldots$ |

<div class="warning">

Too-small $h$ → **catastrophic cancellation**: you subtract two nearly-equal floats and the answer drowns in round-off. There's a fragile sweet spot you'd have to tune per problem. *(illustrative — run it in the notebook.)*

</div>

---

<!-- _class: code-heavy -->

# Why not numerical differentiation? (2/2 — cost)

Worse than fragile, it **doesn't scale**. To get the gradient over $P$ parameters:

```python
# finite differences: perturb EACH parameter, re-run the whole forward pass
for i in range(P):                 # P = 10^6 ... 10^11 in real nets
    grad[i] = (f(theta + eps_i) - f(theta - eps_i)) / (2*eps)
#  => about 2P forward passes for ONE gradient
```

<div class="keypoint">

Finite differences cost $\mathcal{O}(P)$ **forward passes per gradient**. Reverse-mode autodiff costs **one** backward pass — total, for all $P$ gradients. For $P=10^9$ that's the difference between *a billion forward passes* and *one*. **That single fact is why backprop, not finite differences, powers deep learning.**

</div>

---

<!-- _class: code-heavy -->

# Numerical diff isn't useless — it's your unit test

Autograd bugs are **silent**: a wrong gradient still runs and the loss still moves. So when you hand-write a new backward rule, verify it against finite differences on a tiny input — **gradient checking**:

```python
ana = my_layer.backward(x)                       # your analytic autograd grad
num = (f(x + h) - f(x - h)) / (2*h)              # finite difference, small net
rel = abs(ana - num) / (abs(ana) + abs(num) + 1e-9)
assert rel < 1e-5, "backward rule is wrong!"
```

<div class="insight">

Use finite differences where they shine — a slow, offline **check** on a small example — and never in the training loop. Ng calls this *grad-check*; turn it off once your layer passes. This is the one place numerical diff earns its keep.

</div>

---

# Practice problem 3

<div class="popquiz">

**Practice problem 3.** A training loop calls `loss.backward()` every iteration but **forgets** `optimizer.zero_grad()`. Given that PyTorch (like our engine) does `param.grad += ...`, what happens to `w.grad` over successive steps — and how does it corrupt training?

</div>

*Try it before the next slide. Hint: recall the `+=` from Part 1.*

---

# Solution · practice problem 3

`.backward()` **accumulates** into `.grad` (the `+=` that made fan-out correct). Across steps, with no reset:

$$\texttt{w.grad} = g_1,\; g_1{+}g_2,\; g_1{+}g_2{+}g_3,\;\dots \;=\; \textstyle\sum_{t} g_t$$

<div class="keypoint">

Each step uses a **running sum of all past gradients**, not the current one. Effectively a huge, ever-growing step in a stale direction → the loss diverges or oscillates wildly. `zero_grad()` resets `.grad` to $0$ **between steps**, so `+=` starts fresh. The accumulation is right *within* one graph, wrong *across* optimization steps — `zero_grad` draws that line.

</div>

---

<!-- _class: code-heavy -->

# The two guards you'll type every day

```python
optimizer.zero_grad()   # reset .grad to 0 BEFORE backward — or grads pile up (PP3)

with torch.no_grad():   # at eval/inference: don't build the graph
    preds = model(x)    #   -> no tape, less memory, faster
```

<div class="columns">
<div>

**`zero_grad()`** — clears last step's gradients so this step's `.backward()` accumulates cleanly. Forget it → Practice Problem 3.

</div>
<div>

**`no_grad()`** — when you only want predictions (validation, deployment), skip recording the graph. You never call `.backward()` there, so the tape is pure overhead.

</div>
</div>

<div class="notebook">

**📓 Notebook · autodiff** — micrograd vs PyTorch side by side, the numerical-diff pitfalls above, and `no_grad` / `zero_grad` demonstrated. *(ML ES 335 · `notebooks/autodiff.ipynb`, N. Batra)*

</div>

---

# The five-line training loop

Everything today collapses into the loop you'll write for every model this semester:

```python
for x, y in loader:              # 1. a mini-batch
    optimizer.zero_grad()        # 2. clear last step's grads     (PP3)
    pred = model(x)              # 3. forward  — build the graph
    loss = criterion(pred, y)    # 4. the loss — its NLL (L1)
    loss.backward()              # 5. backward — fill every .grad  (Parts 1-2)
    optimizer.step()             #    step downhill                (L5)
```

![w:720px](figures/lec01/svg/training_loop_anatomy.svg)

<div class="keypoint">

`forward → loss → backward → step`. The engine we built *is* line 5; L5 makes line 6 smart.

</div>

---

# Practice problem 4

<div class="popquiz">

**Practice problem 4.** Hand-backprop a **whole neuron**. Inputs $x_1=2,\ x_2=-1$; weights $w_1=1,\ w_2=2$; bias $b=0$; target $y=-1$.

$$n = w_1x_1 + w_2x_2 + b,\qquad o = \tanh(n),\qquad L = (o-y)^2$$

Do the forward pass, then backprop to find $\dfrac{\partial L}{\partial w_1},\ \dfrac{\partial L}{\partial w_2},\ \dfrac{\partial L}{\partial b}$.

</div>

*Try it before the next slide — the numbers are chosen so $\tanh$ stays clean.*

---

# Solution · practice problem 4

**Forward:** $\;n = (1)(2)+(2)(-1)+0 = 0,\quad o=\tanh 0 = 0,\quad L=(0-(-1))^2 = 1.$

**Backward**, node by node (seed $\partial L/\partial L=1$):

$$\frac{\partial L}{\partial o}=2(o-y)=2 \;\xrightarrow{\;\tanh\;}\; \frac{\partial L}{\partial n}=(1-o^2)\frac{\partial L}{\partial o}=(1)(2)=2$$

Then the linear node ($+$ copies, $\times$ swaps):

$$\frac{\partial L}{\partial w_1}=x_1\frac{\partial L}{\partial n}=4,\qquad \frac{\partial L}{\partial w_2}=x_2\frac{\partial L}{\partial n}=-2,\qquad \frac{\partial L}{\partial b}=\frac{\partial L}{\partial n}=2$$

<div class="keypoint">

$$\boxed{\ \partial L/\partial w_1 = 4,\quad \partial L/\partial w_2 = -2,\quad \partial L/\partial b = 2\ }$$

To cut the loss: $w_1$ **down**, $w_2$ **up**, $b$ **down**. Had $n$ been large, $1-o^2$ would be tiny and *every* weight's gradient would shrink with it — saturation throttling learning, exactly Part 4. *(This is Karpathy's `micrograd` neuron, done by hand.)*

</div>

---

# Practice problem 5

<div class="popquiz">

**Practice problem 5.** Add one primitive: the exponential. Fill in its `_backward`, then build `sigmoid` with **no new** backward rule.

```python
def exp(self):
    out = Value(math.exp(self.data), (self,), 'exp')
    def _backward():
        self.grad += ______ * out.grad      # d/dx e^x = ?
    out._backward = _backward
    return out
```

What goes in the blank? Then write `sigmoid` using only `exp`, `+`, `*`, and `**`.

</div>

*Try it before the next slide.*

---

# Solution · practice problem 5

$\dfrac{d}{dx}e^x = e^x$ — and we **already computed** $e^x$; it's sitting in `out.data`. So reuse it, no re-exponentiating:

```python
        self.grad += out.data * out.grad     # local grad = e^x = out.data
```

`sigmoid` is then pure **composition** — $\sigma(x)=\dfrac{1}{1+e^{-x}}$ — and inherits correct gradients for free:

```python
def sigmoid(self):
    return ((self * -1).exp() + 1) ** -1     # 1 / (1 + e^{-x}), all existing ops
```

<div class="keypoint">

Contrast `pow`, whose local rule is $k\,a^{k-1}$; `exp` is special because its derivative **is** its own output. And once a primitive has a correct local rule, everything built from it — `sigmoid`, soft-plus, even `tanh` — differentiates automatically. That is the whole "small set of primitives" payoff.

</div>

---

<!-- _class: section-divider -->

## Part 4 · Coda — keeping gradients healthy through depth

---

# Deep chains multiply many derivatives

Backprop through $L$ layers **multiplies** $L$ local derivatives. If each is a bit below $1$, the product **vanishes**; a bit above $1$, it **explodes**:

$$\frac{\partial L}{\partial \text{early weight}} \;=\; \prod_{\ell} \big(\text{local derivative at layer }\ell\big)$$

![w:560px](figures/lec02/svg/sigmoid_gradient_stack.svg)

<div class="warning">

Sigmoid/tanh saturate: their local gradient is $\le 1$ (often $\ll 1$). Stack 20 layers and the early layers get essentially **zero** gradient — they stop learning. This is the **vanishing-gradient** problem, and it's a direct consequence of the chain-rule *product* we built.

</div>

---

# The variance view: keep the signal alive

Track the **variance** of activations (and gradients) as they pass through layers. If each layer scales variance by a factor $\ne 1$, it compounds and the signal dies or blows up.

![w:600px](figures/lec02/svg/variance_flow.svg)

<div class="keypoint">

The fix isn't a new algorithm — it's **initialization**. Choose the initial weight scale so each layer keeps variance $\approx 1$ on both the forward and backward pass. Then the chain-rule product stays near $1$ and gradients neither vanish nor explode.

</div>

---

# Watch the gradient die — and be revived

<div class="notebook">

**🎛 Interactive · vanishing gradients through depth** — stack sigmoid / tanh / ReLU layers and watch the gradient magnitude at each layer as depth grows; then switch on Xavier/He init and see the early-layer gradients come back to life. *(Interactive Lab · `~/git/interactive/src/articles/vanishing-gradients`)*

</div>

<div class="popquiz">

**Predict first.** Ten `tanh` layers, each contributing a local gradient of about $0.25$ in its active region. Roughly what factor multiplies the gradient by the time it reaches layer 1 — $0.25\times 10$, $0.25^{10}$, or $10^{0.25}$?

</div>

*Answer:* $0.25^{10}\approx 10^{-6}$ — a **product**, not a sum. The first layer sees a gradient a *million* times smaller than the last. That is the vanishing-gradient problem in one number, and why the initialization on the next slides matters.

---

# Xavier / Glorot initialization

For **tanh / sigmoid** layers, keep forward *and* backward variance stable by sampling weights with variance tied to both fan-in and fan-out:

<div class="math-box">

$$\text{Var}(W) = \frac{2}{n_{\text{in}} + n_{\text{out}}} \qquad\Longleftrightarrow\qquad W \sim \mathcal{U}\!\left[-\sqrt{\tfrac{6}{n_{\text{in}}+n_{\text{out}}}},\; \sqrt{\tfrac{6}{n_{\text{in}}+n_{\text{out}}}}\right]$$

</div>

![w:520px](figures/lec02/svg/init_landscape.svg)

Derived by asking: *what weight scale leaves the variance of activations unchanged through a linear + tanh layer?* Balances the two directions — hence $n_{\text{in}}+n_{\text{out}}$.

---

# He initialization — the ReLU fix

**ReLU zeros out half its inputs**, halving the variance each layer. He init compensates with a factor of $2$:

<div class="math-box">

$$\text{Var}(W) = \frac{2}{n_{\text{in}}} \qquad\Longleftrightarrow\qquad W \sim \mathcal{N}\!\left(0,\; \tfrac{2}{n_{\text{in}}}\right)$$

</div>

![w:430px](figures/lec02/svg/he_vs_naive_variance.svg)

<div class="keypoint">

The extra $2$ exactly undoes ReLU's halving. Use **He** with ReLU, **Xavier** with tanh/sigmoid — PyTorch's `kaiming_normal_` and `xavier_uniform_`. Good init is *not* optional: it lets the gradients we just learned to compute actually reach the early layers.

</div>

---

<!-- _class: summary-slide -->

# One sentence to remember

<div class="keypoint">

**Backprop is the chain rule on a computational graph — each node scales the upstream gradient by its local derivative — and reverse-mode autodiff computes every parameter's gradient in one backward pass.**

</div>

- A computation is a **graph**; the forward pass evaluates it, the backward pass differentiates it.
- **downstream = local × upstream**, accumulated (`+=`) at every node.
- We **built micrograd**: `Value` + per-op `_backward` + topological `backward()` — the same algorithm as PyTorch.
- **Numerical diff** is fragile and $\mathcal{O}(P)$; backprop is one pass. `zero_grad` / `no_grad` are the guardrails.
- Through depth, gradients **vanish/explode** — **Xavier/He init** keeps them healthy.

**Next (L5):** given the gradient, how do we take the *step*? — SGD, momentum, Adam, and learning-rate schedules.

---

<!-- _class: compact -->

# Sources & credits

<div class="paper">

This lecture reuses and adapts material from the instructor's own courses (IIT Gandhinagar) and standard references:

- **The micrograd build, `Value` engine, "let's build it" framing** — A. Karpathy, *micrograd* (`github.com/karpathy/micrograd`, MIT) & "The spelled-out intro to backpropagation / Let's build micrograd" (YouTube). This lecture is spiritually his build.
- **`Value` class, forward/backward, topological sort, PyTorch autograd, numerical-diff pitfalls** — *Machine Learning (ES 335)*, N. Batra · `github.com/nipunbatra/ml-teaching` (`notebooks/autograd-from-scratch.ipynb`, `autodiff.ipynb`).
- **Matrix-calculus backprop, the $\delta$ rule, vanishing/exploding gradients** — *Neural Networks* tutorial, ML ES 335, N. Batra.
- **Pedagogical framing (forward/backward, training loop, init)** — A. Ng, *Deep Learning Specialization* (Course 1, Weeks 3–4).

Figures adapted from the ES 667 figure library (`figures/lec01`–`lec03`). All source courses © N. Batra & teaching staff.

</div>

---

<!-- _class: section-divider -->

## ⭐⭐⭐ Appendix · optional depth

*Skip on a first pass — for the curious.*

---

<!-- _class: math-heavy -->

# ⭐⭐⭐ Optional · backprop in matrix form (the $\delta$ rule)

Our scalar engine has a vectorized twin. For a net with $\mathbf z^{(l)}=\mathbf W^{(l)}\mathbf h^{(l-1)}+\mathbf b^{(l)}$, $\ \mathbf h^{(l)}=\sigma(\mathbf z^{(l)})$, define the **error signal** $\delta^{(l)}=\partial\mathcal L/\partial\mathbf z^{(l)}$:

$$\delta^{(L)}=\nabla_{\mathbf h^{(L)}}\mathcal L\odot\sigma'(\mathbf z^{(L)}),\qquad \delta^{(l)}=\big(\mathbf W^{(l+1)}\big)^{\!\top}\delta^{(l+1)}\odot\sigma'(\mathbf z^{(l)})$$

$$\frac{\partial\mathcal L}{\partial\mathbf W^{(l)}}=\delta^{(l)}\big(\mathbf h^{(l-1)}\big)^{\!\top},\qquad \frac{\partial\mathcal L}{\partial\mathbf b^{(l)}}=\delta^{(l)}$$

<div class="insight">

Same three moves as the scalar case: $\mathbf W^\top$ is "multiply swaps," $\odot\,\sigma'$ is the activation's local rule, and $\delta^{(l-1)}$ reuses $\delta^{(l)}$ — the reused upstream grad. Backprop is this recursion; PyTorch runs it for you. *(Full derivation: ES 335 Neural Networks tutorial.)*

</div>

---

<!-- _class: math-heavy -->

# ⭐⭐⭐ Optional · why *reverse* mode, not forward?

Autodiff comes in two directions. For $f:\mathbb R^{n}\!\to\mathbb R^{m}$:

<div class="columns">
<div>

**Forward mode** — push one input's derivative through; cost $\propto n$ (one sweep per **input**). Cheap when $n\ll m$.

</div>
<div>

**Reverse mode** — pull one output's derivative back; cost $\propto m$ (one sweep per **output**). Cheap when $m\ll n$.

</div>
</div>

<div class="keypoint">

A loss is **scalar**: $m=1$, while $n=$ #parameters is enormous. So reverse mode needs **one** backward sweep for *all* $n$ gradients, while forward mode would need $n$. That asymmetry — many inputs, one output — is *why* deep learning uses reverse-mode backprop. The price: you must **store the forward activations** (the tape) to replay them, which is where training memory goes.

</div>
