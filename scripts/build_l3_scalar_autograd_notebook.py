#!/usr/bin/env python3
"""Build and execute the focused PyTorch-to-scratch scalar autograd notebook."""

from __future__ import annotations

import argparse
from pathlib import Path
import textwrap

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/L03/04_scalar_autograd_pytorch_to_scratch.ipynb"


def md(source: str):
    return nbformat.v4.new_markdown_cell(textwrap.dedent(source).strip())


def code(source: str, purpose: str, *, hidden: bool = False):
    metadata = {"purpose": purpose}
    normalized = textwrap.dedent(source).strip()
    if hidden:
        metadata["jupyter"] = {"source_hidden": True}
        normalized = "#| echo: false\n" + normalized
    return nbformat.v4.new_code_cell(normalized, metadata=metadata)


def build_notebook():
    cells = [
        md(
            r"""
            # One scalar graph: PyTorch → our own autograd engine

            We will use the **exact graph from the lecture** throughout:

            $$
            m=wx,\qquad a=m+b,\qquad e=a-y,\qquad L=e^2
            $$

            with
            $$
            x=3,\quad w=2,\quad b=1,\quad y=10.
            $$

            The order is deliberate:

            1. **PyTorch first** gives us a trusted reference.
            2. We then build the mechanism ourselves, one scalar operation at a time.
            3. The same horizontal tape remains visible while gradients fill in during the reverse sweep.

            The from-scratch half follows the spirit of Andrej Karpathy's
            [micrograd](https://github.com/karpathy/micrograd) teaching style, while preserving our lecture's
            variables, graph, numbers, and direct subtraction node.
            """
        ),
        md(
            r"""
            ## Before running anything

            Predict the forward values $m,a,e,L$. Then leave the gradient row blank and predict the order in which
            it should fill during backward.

            The final ledger should contain these eight stored values:

            | node | meaning |
            |---|---|
            | $w,x,b,y$ | leaves supplied to the graph |
            | $m,a,e,L$ | intermediate results created by operations |
            """
        ),
        md(
            r"""
            ## Part A · Ask PyTorch for the answer first

            PyTorch does not use finite differences here. It records the operations that actually execute, stores
            what their backward rules need, seeds the scalar loss with $g_L=1$, and runs those rules in reverse.
            """
        ),
        code(
            r'''
            import html
            import math
            import torch
            from IPython.display import HTML, display

            torch.set_default_dtype(torch.float64)

            FORMULAS = {
                "w": "parameter", "x": "input", "m": "w × x", "b": "bias",
                "a": "m + b", "y": "target", "e": "a − y", "L": "e²",
            }
            TAPE_ORDER = ("w", "x", "m", "b", "a", "y", "e", "L")

            display(HTML(r"""
            <style>
            .l4-wrap {max-width: 100%; overflow-x: auto; margin: .65rem 0 1rem;}
            .l4-tape {display: flex; align-items: center; gap: .42rem; width: max-content;
                      min-width: 100%; padding: .55rem .15rem .75rem;}
            .l4-node {width: 6.8rem; min-height: 7.2rem; box-sizing: border-box; border: 2px solid #b9c7ca;
                      border-radius: 15px; background: #fff; padding: .55rem .58rem; text-align: center;}
            .l4-node.active {border-color: #ef7d00; box-shadow: 0 0 0 4px rgba(239,125,0,.13);}
            .l4-node.missing {opacity: .38; background: #f5f7f7;}
            .l4-label {font-size: 1.2rem; font-weight: 800; color: #17343b;}
            .l4-formula {height: 1.7rem; margin-top: .08rem; color: #667b80; font-size: .78rem;}
            .l4-data {margin-top: .28rem; color: #17343b; font-weight: 650;}
            .l4-grad {margin-top: .12rem; color: #d95f02; font-weight: 760;}
            .l4-op {font-size: 1.35rem; color: #667b80; font-weight: 650;}
            .l4-note {border-left: 4px solid #ef7d00; padding: .55rem .75rem; margin: .4rem 0 .7rem;
                      background: #fff7ed; color: #17343b;}
            .l4-table {border-collapse: collapse; width: 100%; min-width: 620px; font-size: .92rem;}
            .l4-table th {background: #17343b; color: white; text-align: left;}
            .l4-table th,.l4-table td {border: 1px solid #cbd5d7; padding: .42rem .55rem;}
            .l4-table tr:nth-child(even) td {background: #f5f8f8;}
            .l4-ok {color: #16833f; font-weight: 750;}
            </style>
            """))

            def fmt(value):
                if value is None:
                    return "—"
                value = float(value)
                if abs(value) < 5e-13:
                    value = 0.0
                return f"{value:.4g}"

            def _card(label, values, grads, visible, active):
                built = label in values
                classes = ["l4-node"]
                if not built:
                    classes.append("missing")
                if label == active:
                    classes.append("active")
                grad = fmt(grads.get(label)) if label in visible else "—"
                return (
                    f'<div class="{" ".join(classes)}" aria-label="node {html.escape(label)}">'
                    f'<div class="l4-label">{html.escape(label)}</div>'
                    f'<div class="l4-formula">{html.escape(FORMULAS[label])}</div>'
                    f'<div class="l4-data">value: {fmt(values.get(label))}</div>'
                    f'<div class="l4-grad">grad: {grad}</div></div>'
                )

            def tape_html(values, grads=None, visible=(), active=None, note="", events=()):
                grads = grads or {}
                visible = set(visible)
                pieces = [
                    _card("w", values, grads, visible, active), '<span class="l4-op">×</span>',
                    _card("x", values, grads, visible, active), '<span class="l4-op">→</span>',
                    _card("m", values, grads, visible, active), '<span class="l4-op">+</span>',
                    _card("b", values, grads, visible, active), '<span class="l4-op">→</span>',
                    _card("a", values, grads, visible, active), '<span class="l4-op">−</span>',
                    _card("y", values, grads, visible, active), '<span class="l4-op">→</span>',
                    _card("e", values, grads, visible, active), '<span class="l4-op">² →</span>',
                    _card("L", values, grads, visible, active),
                ]
                event_html = ""
                if events:
                    rows = "".join(
                        "<tr>" + "".join(f"<td>{html.escape(str(item))}</td>" for item in row) + "</tr>"
                        for row in events
                    )
                    event_html = (
                        '<div class="l4-wrap"><table class="l4-table"><thead><tr>'
                        '<th>from</th><th>upstream</th><th>to</th><th>local</th><th>contribution</th>'
                        '<th>accumulated grad</th></tr></thead><tbody>' + rows + '</tbody></table></div>'
                    )
                note_html = f'<div class="l4-note">{html.escape(note)}</div>' if note else ""
                return note_html + '<div class="l4-wrap"><div class="l4-tape">' + "".join(pieces) + \
                    "</div></div>" + event_html

            def show_tape(values, grads=None, visible=(), active=None, note="", events=()):
                display(HTML(tape_html(values, grads, visible, active, note, events)))

            def ledger_html(rows, headers=("node", "formula", "value", "gradient")):
                head = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
                body = "".join(
                    "<tr>" + "".join(f"<td>{html.escape(str(item))}</td>" for item in row) + "</tr>"
                    for row in rows
                )
                return f'<div class="l4-wrap"><table class="l4-table"><thead><tr>{head}</tr></thead>' \
                       f'<tbody>{body}</tbody></table></div>'

            print(f"PyTorch {torch.__version__} · default dtype {torch.get_default_dtype()}")
            ''',
            "Import PyTorch and define the durable paper-tape display helper",
            hidden=True,
        ),
        md(
            r"""
            ### A1 · Forward builds and stores the graph

            We track all four leaves here—even $x$ and $y$—because we want the complete paper ledger. In routine
            supervised training, inputs and targets usually do not require gradients.
            """
        ),
        code(
            r'''
            def build_torch_graph(w_value=2.0, b_value=1.0):
                nodes = {
                    "w": torch.tensor(w_value, requires_grad=True),
                    "x": torch.tensor(3.0, requires_grad=True),
                    "b": torch.tensor(b_value, requires_grad=True),
                    "y": torch.tensor(10.0, requires_grad=True),
                }
                nodes["m"] = nodes["w"] * nodes["x"]
                nodes["a"] = nodes["m"] + nodes["b"]
                nodes["e"] = nodes["a"] - nodes["y"]
                nodes["L"] = nodes["e"] ** 2
                for name in ("m", "a", "e", "L"):
                    nodes[name].retain_grad()
                return nodes

            torch_nodes = build_torch_graph()
            torch_values = {name: node.item() for name, node in torch_nodes.items()}
            show_tape(
                torch_values,
                active="L",
                note="Forward complete: values are stored; no backward gradients have been computed yet.",
            )

            expected_values = {"w": 2, "x": 3, "m": 6, "b": 1, "a": 7, "y": 10, "e": -3, "L": 9}
            assert all(math.isclose(torch_values[k], v) for k, v in expected_values.items())
            ''',
            "Build the exact scalar graph in PyTorch and show the forward tape",
        ),
        md(
            r"""
            ### A2 · Backward starts with a seed

            `L.backward()` is shorthand for starting with
            $$
            g_L=\frac{\partial L}{\partial L}=1
            $$
            and running the recorded local rules in reverse. PyTorch gives us the complete result; in Part B we will
            expose the individual reverse steps ourselves.
            """
        ),
        code(
            r'''
            torch_nodes["L"].backward()
            torch_grads = {name: node.grad.item() for name, node in torch_nodes.items()}
            show_tape(
                torch_values,
                grads=torch_grads,
                visible=TAPE_ORDER,
                active="w",
                note="PyTorch backward complete: every stored gradient now matches the paper calculation.",
            )

            torch_reference = {
                name: {"value": torch_values[name], "grad": torch_grads[name]}
                for name in TAPE_ORDER
            }
            expected_grads = {"w": -18, "x": -12, "m": -6, "b": -6, "a": -6, "y": 6, "e": -6, "L": 1}
            assert all(math.isclose(torch_grads[k], v) for k, v in expected_grads.items())

            display(HTML(ledger_html([
                (name, FORMULAS[name], fmt(torch_values[name]), fmt(torch_grads[name]))
                for name in TAPE_ORDER
            ])))
            ''',
            "Run PyTorch backward and retain the complete eight-node reference ledger",
        ),
        md(
            r"""
            ### A3 · Update parameters, then rebuild the forward graph

            We update only $w,b$, not the input or target. The old intermediates still describe the old forward
            pass, so we build a **fresh graph** after the update.
            """
        ),
        code(
            r'''
            eta = 0.01
            w_after = torch_reference["w"]["value"] - eta * torch_reference["w"]["grad"]
            b_after = torch_reference["b"]["value"] - eta * torch_reference["b"]["grad"]
            updated_torch = build_torch_graph(w_after, b_after)
            new_prediction = updated_torch["a"].item()
            new_loss = updated_torch["L"].item()

            display(HTML(ledger_html([
                ("w", "2 − 0.01(−18)", fmt(w_after), "2.18"),
                ("b", "1 − 0.01(−6)", fmt(b_after), "1.06"),
                ("prediction", "wx+b", fmt(new_prediction), "7.60"),
                ("loss", "(prediction−y)²", fmt(new_loss), "5.76"),
            ], headers=("quantity", "calculation", "computed", "expected"))))

            assert math.isclose(w_after, 2.18)
            assert math.isclose(b_after, 1.06)
            assert math.isclose(new_prediction, 7.60)
            assert math.isclose(new_loss, 5.76)
            assert new_loss < torch_reference["L"]["value"]
            ''',
            "Apply the lecture update and verify that a fresh PyTorch graph has lower loss",
        ),
        md(
            r"""
            ## Part B · Build the mechanism ourselves

            A scalar autograd value needs only five ideas:

            1. its forward `data`,
            2. a gradient buffer `grad`,
            3. ordered parent edges,
            4. the local derivative along each edge,
            5. a local `_backward` function that **adds** into parent buffers.

            The code below is intentionally small and inspectable. It is a teaching engine, not a replacement for
            PyTorch.
            """
        ),
        code(
            r'''
            class Value:
                """One scalar value plus the graph needed to differentiate it."""

                def __init__(self, data, *, label="", op="", edges=()):
                    self.data = float(data)
                    self.grad = 0.0
                    self.label = label
                    self.op = op
                    self._edges = tuple(edges)       # ordered; repeated parents are allowed
                    self._backward = lambda: None

                @staticmethod
                def _coerce(other):
                    return other if isinstance(other, Value) else Value(other, label=fmt(other))

                def named(self, label):
                    self.label = label
                    return self

                def __repr__(self):
                    return f"Value(label={self.label!r}, data={self.data:.4g}, grad={self.grad:.4g})"

                def __add__(self, other):
                    other = self._coerce(other)
                    out = Value(self.data + other.data, op="+", edges=((self, 1.0), (other, 1.0)))
                    def _backward():
                        self.grad += out.grad * 1.0
                        other.grad += out.grad * 1.0
                    out._backward = _backward
                    return out

                __radd__ = __add__

                def __mul__(self, other):
                    other = self._coerce(other)
                    left_value, right_value = self.data, other.data  # save forward values
                    out = Value(left_value * right_value, op="×",
                                edges=((self, right_value), (other, left_value)))
                    def _backward():
                        self.grad += out.grad * right_value
                        other.grad += out.grad * left_value
                    out._backward = _backward
                    return out

                __rmul__ = __mul__

                def __sub__(self, other):
                    other = self._coerce(other)
                    out = Value(self.data - other.data, op="−", edges=((self, 1.0), (other, -1.0)))
                    def _backward():
                        self.grad += out.grad * 1.0
                        other.grad += out.grad * -1.0
                    out._backward = _backward
                    return out

                def __rsub__(self, other):
                    return self._coerce(other) - self

                def __neg__(self):
                    return -1.0 * self

                def __pow__(self, exponent):
                    if not isinstance(exponent, (int, float)):
                        raise TypeError("This teaching engine supports only scalar numeric powers")
                    saved_value = self.data
                    local = exponent * saved_value ** (exponent - 1)
                    out = Value(saved_value ** exponent, op=f"**{exponent:g}", edges=((self, local),))
                    def _backward():
                        self.grad += out.grad * local
                    out._backward = _backward
                    return out

                def backward(self):
                    zero_grad(self)
                    self.grad = 1.0
                    for node in reversed(topological_order(self)):
                        node._backward()

            print("Value now records forward data, ordered edges, local derivatives, and += backward rules.")
            ''',
            "Implement the small scalar Value class and its local backward closures",
        ),
        md(
            r"""
            ### B1 · Deterministic graph order and gradient buffers

            Reverse mode needs children before parents during backward. We therefore topologically sort the graph,
            then traverse that list in reverse. Parent **edges remain ordered tuples**: this is important when the same
            value appears twice, as in `q*q`.
            """
        ),
        code(
            r'''
            def topological_order(root):
                order, seen = [], set()
                def visit(node):
                    if id(node) in seen:
                        return
                    seen.add(id(node))
                    for parent, _local in node._edges:
                        visit(parent)
                    order.append(node)
                visit(root)
                return order

            def zero_grad(root):
                for node in topological_order(root):
                    node.grad = 0.0

            def named_nodes(root):
                return {node.label: node for node in topological_order(root) if node.label}

            def show_scratch(root, visible=(), active=None, note="", events=()):
                nodes = named_nodes(root)
                values = {name: node.data for name, node in nodes.items()}
                grads = {name: node.grad for name, node in nodes.items()}
                show_tape(values, grads, visible, active, note, events)

            print("A stable DFS supplies forward order; reversing it supplies backward order.")
            ''',
            "Implement deterministic topological ordering, zero_grad, and scratch display adapters",
        ),
        md(
            r"""
            ### B2 · Rebuild the forward pass one line at a time

            Run the next five cells slowly. Each line creates one new stored value and remembers how it was produced.
            """
        ),
        code(
            r'''
            w = Value(2.0, label="w")
            x = Value(3.0, label="x")
            b = Value(1.0, label="b")
            y = Value(10.0, label="y")
            scratch = {"w": w, "x": x, "b": b, "y": y}
            show_tape({k: v.data for k, v in scratch.items()}, active="w",
                      note="Leaves created: four numbers exist, but no operation has run yet.")
            ''',
            "Create the four scalar leaves and show the initial paper tape",
        ),
        code(
            r'''
            m = (w * x).named("m")
            scratch["m"] = m
            show_tape({k: v.data for k, v in scratch.items()}, active="m",
                      note="Multiplication stores m = w×x = 6 and local derivatives x=3, w=2.")
            ''',
            "Execute multiplication and expose the newly stored value",
        ),
        code(
            r'''
            a = (m + b).named("a")
            scratch["a"] = a
            show_tape({k: v.data for k, v in scratch.items()}, active="a",
                      note="Addition stores a = m+b = 7; both local derivatives are 1.")
            ''',
            "Execute addition and expose the newly stored value",
        ),
        code(
            r'''
            e = (a - y).named("e")
            scratch["e"] = e
            show_tape({k: v.data for k, v in scratch.items()}, active="e",
                      note="Subtraction stores e = a−y = −3; its local derivatives are +1 and −1.")
            ''',
            "Execute subtraction as one explicit lecture-matching primitive",
        ),
        code(
            r'''
            L = (e ** 2).named("L")
            scratch["L"] = L
            show_tape({k: v.data for k, v in scratch.items()}, active="L",
                      note="Square stores L = e² = 9 and the local derivative 2e = −6.")
            assert [node.label for node in topological_order(L) if node.label] == ["w", "x", "m", "b", "a", "y", "e", "L"]
            ''',
            "Finish the forward graph and verify the deterministic stored-value order",
        ),
        md(
            r"""
            ### B3 · Make backward observable

            `Value.backward()` can already run the whole reverse pass. For teaching, `BackwardTape` exposes the same
            mechanism as five durable states: seed, square, subtraction, addition, multiplication.

            Every event uses
            $$
            \text{parent.grad} \mathrel{+}= \text{upstream}\times\text{local derivative}.
            $$
            """
        ),
        code(
            r'''
            class BackwardTape:
                def __init__(self, root):
                    self.root = root
                    zero_grad(root)
                    self.ops = [node for node in reversed(topological_order(root)) if node._edges]
                    self.cursor = 0
                    self.visible = set()

                def seed(self):
                    if self.visible:
                        raise RuntimeError("This tape has already been seeded")
                    self.root.grad = 1.0
                    self.visible.add(self.root.label)
                    return self.root

                def step(self):
                    if self.cursor >= len(self.ops):
                        raise StopIteration("Backward is complete")
                    node = self.ops[self.cursor]
                    predicted, rows = {}, []
                    for parent, local in node._edges:
                        before = predicted.get(id(parent), parent.grad)
                        contribution = node.grad * local
                        after = before + contribution
                        predicted[id(parent)] = after
                        rows.append((
                            node.label, fmt(node.grad), parent.label, fmt(local), fmt(contribution),
                            f"{fmt(before)} → {fmt(after)}",
                        ))
                    node._backward()
                    for parent, _local in node._edges:
                        assert math.isclose(parent.grad, predicted[id(parent)])
                        if parent.label:
                            self.visible.add(parent.label)
                    self.cursor += 1
                    return node, rows

            tape = BackwardTape(L)
            assert [node.label for node in tape.ops] == ["L", "e", "a", "m"]
            print("Reverse operation order:", " → ".join(node.label for node in tape.ops))
            ''',
            "Implement the deterministic classroom backward stepper",
        ),
        code(
            r'''
            tape.seed()
            show_scratch(L, tape.visible, active="L",
                         note="Seed: g_L = ∂L/∂L = 1. Nothing has propagated yet.")
            ''',
            "Seed the scalar loss gradient and display the first backward state",
        ),
        code(
            r'''
            active, events = tape.step()
            show_scratch(L, tape.visible, active=active.label,
                         note="Square: arriving 1 × local (2e = −6) sends −6 to e.", events=events)
            ''',
            "Step backward through the square operation",
        ),
        code(
            r'''
            active, events = tape.step()
            show_scratch(L, tape.visible, active=active.label,
                         note="Subtraction: g_e=−6 copies to a and changes sign on the y branch.", events=events)
            ''',
            "Step backward through the subtraction operation",
        ),
        code(
            r'''
            active, events = tape.step()
            show_scratch(L, tape.visible, active=active.label,
                         note="Addition: g_a=−6 copies to both m and b.", events=events)
            ''',
            "Step backward through the addition operation",
        ),
        code(
            r'''
            active, events = tape.step()
            show_scratch(L, tape.visible, active=active.label,
                         note="Multiplication: g_m=−6 is scaled by the other input on each return path.", events=events)
            assert tape.cursor == 4
            ''',
            "Step backward through multiplication and finish the scratch reverse pass",
        ),
        md(
            r"""
            ### B4 · Compare our engine with PyTorch

            Agreement is the test: the two systems must produce the same stored values and gradients at all eight
            named nodes.
            """
        ),
        code(
            r'''
            scratch_nodes = named_nodes(L)
            comparison = []
            for name in TAPE_ORDER:
                s, t = scratch_nodes[name], torch_reference[name]
                value_ok = math.isclose(s.data, t["value"])
                grad_ok = math.isclose(s.grad, t["grad"])
                assert value_ok and grad_ok
                comparison.append((name, fmt(t["value"]), fmt(s.data), fmt(t["grad"]), fmt(s.grad), "✓"))

            display(HTML(ledger_html(
                comparison,
                headers=("node", "PyTorch value", "scratch value", "PyTorch grad", "scratch grad", "match"),
            )))
            print("All eight stored values and gradients match PyTorch exactly.")
            ''',
            "Verify exact node-by-node agreement between the scratch engine and PyTorch",
        ),
        md(
            r"""
            ### B5 · Why every local rule uses `+=`

            A value may reach the loss along more than one path. The branch example from the lecture is
            $J=q^2+3q$. At $q=4$, the two contributions are $8$ and $3$, so $g_q=11$.
            """
        ),
        code(
            r'''
            q = Value(4.0, label="q")
            J = (q ** 2 + 3 * q).named("J")
            J.backward()

            r = Value(5.0, label="r")
            K = (r * r).named("K")
            K.backward()

            assert math.isclose(q.grad, 11.0)       # 2q + 3
            assert math.isclose(r.grad, 10.0)       # repeated parent edge must contribute twice

            # Coercion and reverse operators also stay usable.
            assert math.isclose((1 + q).data, 5.0)
            assert math.isclose((10 - q).data, 6.0)

            display(HTML(ledger_html([
                ("q² path", "2q", "8"),
                ("3q path", "3", "3"),
                ("accumulated q.grad", "8 + 3", fmt(q.grad)),
                ("repeated parent r×r", "r + r", fmt(r.grad)),
            ], headers=("check", "local contributions", "result"))))

            zero_grad(J)
            assert all(math.isclose(node.grad, 0.0) for node in topological_order(J))
            ''',
            "Test gradient accumulation, repeated parents, coercion, and zero_grad",
        ),
        md(
            r"""
            ### B6 · Make the same parameter update from scratch

            Updating a parameter changes the values that a future forward pass should store. Therefore we update
            $w,b$, then build a fresh scratch graph—just as we did with PyTorch.
            """
        ),
        code(
            r'''
            w_new_value = scratch_nodes["w"].data - eta * scratch_nodes["w"].grad
            b_new_value = scratch_nodes["b"].data - eta * scratch_nodes["b"].grad

            w2, x2 = Value(w_new_value, label="w"), Value(3.0, label="x")
            b2, y2 = Value(b_new_value, label="b"), Value(10.0, label="y")
            m2 = (w2 * x2).named("m")
            a2 = (m2 + b2).named("a")
            e2 = (a2 - y2).named("e")
            L2 = (e2 ** 2).named("L")

            assert math.isclose(w2.data, 2.18)
            assert math.isclose(b2.data, 1.06)
            assert math.isclose(a2.data, 7.60)
            assert math.isclose(L2.data, 5.76)
            assert L2.data < L.data

            show_scratch(L2, active="L",
                         note="Fresh forward after the update: prediction 7.60 and loss 5.76.")
            ''',
            "Apply the scratch gradients, rebuild the graph, and verify the lower loss",
        ),
        md(
            r"""
            ## PyTorch concept ↔ our tiny engine

            | PyTorch idea | Our teaching engine |
            |---|---|
            | tensor value | `Value.data` |
            | `.grad` buffer | `Value.grad` |
            | executed graph | ordered `_edges` |
            | saved forward state | captured local derivative values |
            | operation-specific backward rule | `_backward` closure |
            | reverse topological engine traversal | `reversed(topological_order(L))` |
            | multiple paths | every rule updates with `+=` |
            | `optimizer.zero_grad()` | `zero_grad(L)` |

            ### Takeaway

            PyTorch and the scratch engine implement the same reverse-mode idea:

            **forward stores values and graph structure → seed the scalar loss with 1 → apply local rules backward →
            add contributions at shared values → update parameters → rebuild the next forward graph.**

            The scratch engine makes that mechanism visible; PyTorch makes it fast, general, and practical.
            """
        ),
    ]

    # Stable IDs and timing-free execution metadata keep generated diffs meaningful.
    for index, cell in enumerate(cells, start=1):
        cell["id"] = f"scalar-autograd-{index:02d}"

    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
            "course": {
                "lecture": 4,
                "title": "Computation Graphs, Backpropagation & Autograd",
                "evidence": "constructed exact scalar example; computed PyTorch and scratch-engine outputs",
            },
        },
    )
    return notebook


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-execute", action="store_true", help="Write the notebook without executing it")
    args = parser.parse_args()

    notebook = build_notebook()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if not args.no_execute:
        client = NotebookClient(
            notebook,
            timeout=240,
            kernel_name="python3",
            record_timing=False,
            resources={"metadata": {"path": str(ROOT)}},
            allow_errors=False,
        )
        client.execute()
    nbformat.write(notebook, OUTPUT)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
