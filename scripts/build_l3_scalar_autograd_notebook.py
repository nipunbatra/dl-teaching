#!/usr/bin/env python3
"""Build and execute the focused scalar-autograd teaching notebook."""

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
            # Scalar autograd: PyTorch, from scratch, then a fused sigmoid

            We will start with one graph:

            $$
            m=wx,\qquad a=m+b,\qquad e=a-y,\qquad L=e^2
            $$

            with $w=2$, $x=3$, $b=1$, and $y=10$.

            First we let PyTorch differentiate it. Then we build the smallest useful version of the same idea
            ourselves. The point is not to replace PyTorch—it is to see what `.backward()` does. A final optional
            example then compares one fused sigmoid operation with the same sigmoid expanded into atomic operations.
            """
        ),
        md(
            r"""
            ## 1 · The whole example in PyTorch

            Each number below is a scalar tensor. We set `requires_grad=True` because we want PyTorch to calculate
            its loss derivative. In ordinary training, the target `y` would normally be fixed; here we track it only
            so the output matches our complete paper calculation.

            PyTorch keeps `.grad` automatically for the four leaf tensors. `retain_grad()` asks it to keep gradients
            for the intermediate values too.
            """
        ),
        code(
            r'''
            import torch

            torch.set_default_dtype(torch.float64)

            w = torch.tensor(2.0, requires_grad=True)
            x = torch.tensor(3.0, requires_grad=True)
            b = torch.tensor(1.0, requires_grad=True)
            y = torch.tensor(10.0, requires_grad=True)

            # Forward pass
            m = w * x
            a = m + b
            e = a - y
            L = e ** 2

            for node in (m, a, e, L):
                node.retain_grad()

            print("Forward: m =", m.item(), ", a =", a.item(),
                  ", e =", e.item(), ", L =", L.item())
            ''',
            "Create literal scalar tensors and run the lecture forward graph",
        ),
        md(
            r"""
            Now the important line:
            """
        ),
        code(
            r'''
            L.backward()

            print("w.grad =", w.grad)
            print("x.grad =", x.grad)
            print("b.grad =", b.grad)
            print("y.grad =", y.grad)
            ''',
            "Call backward and print the four leaf gradients directly",
        ),
        md(
            r"""
            That is PyTorch autograd. The forward pass created a graph; `.backward()` sent a seed gradient of $1$
            from $L$ through that graph in reverse.

            For comparison with the paper calculation, here is every stored value:
            """
        ),
        code(
            r'''
            torch_nodes = {"w": w, "x": x, "m": m, "b": b,
                           "a": a, "y": y, "e": e, "L": L}
            torch_reference = {
                name: (node.item(), node.grad.item())
                for name, node in torch_nodes.items()
            }

            print(f"{'node':<5} {'value':>8} {'grad':>8}")
            for name, (value, grad) in torch_reference.items():
                print(f"{name:<5} {value:8.1f} {grad:8.1f}")

            expected = {
                "w": (2, -18), "x": (3, -12), "m": (6, -6), "b": (1, -6),
                "a": (7, -6), "y": (10, 6), "e": (-3, -6), "L": (9, 1),
            }
            assert torch_reference == expected
            ''',
            "Print and verify the complete PyTorch value-and-gradient ledger",
        ),
        md(
            r"""
            ## 2 · A tiny autograd engine from scratch

            A `Value` stores only:

            - its number in `data`,
            - its accumulated loss-gradient buffer in `grad`, and
            - ordered links to the operands that directly produced it.

            Each parent link stores two things: the parent operand and the evaluated local derivative along that
            edge. For example, $m=wx$ remembers $(w,\partial m/\partial w=x)$ and
            $(x,\partial m/\partial x=w)$.

            The gradient names are always relative to the operation currently running:

            - <span style="color:#2C7A7B;font-weight:700">upstream</span>:
              $g_v=\partial L/\partial v$, already accumulated in `v.grad`;
            - <span style="color:#2B6CB0;font-weight:700">local</span>:
              $\partial v/\partial u$, stored in the link from output $v$ to parent $u$;
            - <span style="color:#EB811B;font-weight:700">downstream contribution</span>:
              $\Delta g_u=g_v(\partial v/\partial u)$, computed during backward and added to `u.grad`.

            A parent is simply a direct input to an operation—not a whole neural-network layer.
            """
        ),
        code(
            r'''
            class ParentLink:
                def __init__(self, value, local_grad):
                    self.value = value
                    self.local_grad = float(local_grad)

            class Value:
                def __init__(self, data, label="", parents=(), op=""):
                    self.data = float(data)
                    self.grad = 0.0
                    self.label = label
                    self.parents = tuple(parents)
                    self.op = op

            def multiply(u, v, label):
                return Value(u.data * v.data, label,
                             parents=(ParentLink(u, v.data),
                                      ParentLink(v, u.data)), op="×")

            def add(u, v, label):
                return Value(u.data + v.data, label,
                             parents=(ParentLink(u, 1.0),
                                      ParentLink(v, 1.0)), op="+")

            def subtract(u, v, label):
                return Value(u.data - v.data, label,
                             parents=(ParentLink(u, 1.0),
                                      ParentLink(v, -1.0)), op="−")

            def square(u, label):
                return Value(u.data ** 2, label,
                             parents=(ParentLink(u, 2 * u.data),), op="²")

            def backward(root):
                order, seen = [], set()

                # 1. Put every reachable value in forward order.
                def visit(node):
                    if id(node) in seen:
                        return
                    seen.add(id(node))
                    for link in node.parents:
                        visit(link.value)
                    order.append(node)

                visit(root)

                # 2. Clear old accumulated gradients, then seed the loss.
                for node in order:
                    node.grad = 0.0
                root.grad = 1.0

                # 3. Walk backward. One parent link gives one chain-rule update.
                steps = []
                for node in reversed(order):
                    for link in node.parents:
                        upstream = node.grad
                        local = link.local_grad
                        downstream = upstream * local
                        before = link.value.grad
                        link.value.grad += downstream

                        steps.append({
                            "output": node.label,
                            "upstream": upstream,
                            "parent": link.value.label,
                            "local": local,
                            "downstream": downstream,
                            "before": before,
                            "after": link.value.grad,
                        })
                return steps
            ''',
            "Define explicit parent links, the first four local rules, and the traced reverse traversal",
        ),
        md(
            r"""
            Read `backward(root)` in three pieces:

            1. `visit` follows the parent links and makes a forward order. Here it is
               `w, x, m, b, a, y, e, L`.
            2. We clear the reachable `.grad` buffers and seed `L.grad = 1`, because
               $\partial L/\partial L=1$.
            3. We reverse that order. At each link from output $v$ to parent $u$, the loop reads the
               upstream gradient from `v.grad`, reads the local derivative from the link, and adds their product
               into `u.grad`.

            | quantity | where it lives |
            |---|---|
            | upstream $g_v$ | already accumulated in `v.grad` |
            | local $\partial v/\partial u$ | saved in `link.local_grad` during the forward pass |
            | downstream contribution $\Delta g_u$ | temporary variable `downstream` for this one edge |
            | accumulated $g_u$ | updated in `link.value.grad` |

            So the code does **not** store a separate downstream gradient forever. It computes one contribution,
            adds it to the parent's buffer, and that buffer later becomes the upstream gradient for the parent.
            """
        ),
        code(
            r'''
            import importlib.util
            import subprocess
            import sys

            if importlib.util.find_spec("graphviz") is None:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-q", "graphviz"],
                    check=True,
                )

            from graphviz import Digraph
            from IPython.display import HTML, display

            def draw_graph(root, show_grad=True, min_width=780):
                nodes, seen = [], set()
                def visit(node):
                    if id(node) in seen:
                        return
                    seen.add(id(node))
                    nodes.append(node)
                    for link in node.parents:
                        visit(link.value)
                visit(root)

                dot = Digraph(format="svg")
                dot.attr(rankdir="LR", bgcolor="transparent", pad="0.15",
                         nodesep="0.25", ranksep="0.45")
                dot.attr("node", fontname="Helvetica", fontsize="11")
                dot.attr("edge", fontname="Helvetica", fontsize="9", color="#60777b")

                for node in nodes:
                    node_id = "value_" + node.label
                    grad = f"{node.grad:g}" if show_grad else "—"
                    fill = "#e8f7f5" if show_grad and node.grad != 0 else "#ffffff"
                    border = "#2C7A7B" if show_grad and node.grad != 0 else "#1f3a40"
                    dot.node(
                        node_id,
                        label=f"{{ {node.label} | value {node.data:g} | grad {grad} }}",
                        shape="record", style="rounded,filled", fillcolor=fill,
                        color=border, fontcolor="#1f3a40",
                    )
                    if node.op:
                        op_id = "op_" + node.label
                        dot.node(op_id, label=node.op, shape="circle",
                                 width="0.32", height="0.32", margin="0.03",
                                 style="filled", fillcolor="#2B6CB0",
                                 color="#2B6CB0", fontcolor="white")
                        dot.edge(op_id, node_id)
                        for link in node.parents:
                            local_label = (
                                f"∂{node.label}/∂{link.value.label}="
                                f"{link.local_grad:g}"
                            )
                            dot.edge(
                                "value_" + link.value.label,
                                op_id,
                                label=local_label,
                                fontcolor="#2B6CB0",
                            )
                svg = dot.pipe(format="svg").decode("utf-8")
                svg = svg.replace(
                    "<svg ",
                    f'<svg style="width:100%;height:auto;min-width:{min_width}px" ',
                    1,
                )
                return HTML(
                    '<div style="max-width:100%;overflow-x:auto">'
                    f'<div style="min-width:{min_width}px">' + svg + '</div></div>'
                )

            def show_backward_steps(steps, highlight_outputs=()):
                rows = []
                for number, step in enumerate(steps, start=1):
                    output = step["output"]
                    parent = step["parent"]
                    focused = output in highlight_outputs
                    border = "#2B6CB0" if focused else "#d6e2e1"
                    background = "#f4f8ff" if focused else "#fff"
                    rows.append(
                        f"<div style='border:1px solid {border};border-radius:8px;"
                        f"padding:10px 12px;margin:8px 0;background:{background}'>"
                        f"<div style='font-weight:700;margin-bottom:4px'>{number}. {output} → {parent}</div>"
                        "<div style='font-size:1.02em;line-height:1.55'>"
                        f"<span style='color:#2C7A7B;font-weight:700'>g<sub>{output}</sub> = {step['upstream']:g}</span>"
                        " &nbsp;×&nbsp; "
                        f"<span style='color:#2B6CB0;font-weight:700'>∂{output}/∂{parent} = {step['local']:g}</span>"
                        " &nbsp;=&nbsp; "
                        f"<span style='color:#EB811B;font-weight:700'>Δg<sub>{parent}</sub> = {step['downstream']:g}</span>"
                        "</div>"
                        f"<div style='color:#526669;margin-top:3px'>{parent}.grad: "
                        f"{step['before']:g} → {step['after']:g}</div></div>"
                    )

                display(HTML(
                    "<div style='border-left:5px solid #2C7A7B;background:#eef8f7;"
                    "padding:10px 12px;margin:4px 0 12px;border-radius:4px'>"
                    "<b>Seed:</b> L.grad = ∂L/∂L = 1</div>" + "".join(rows)
                ))
            ''',
            "Define the compact Graphviz view and color-matched backward trace cards",
            hidden=True,
        ),
        md(
            r"""
            Build the **same forward graph**, one readable line per operation. On a phone, scroll the graph sideways:
            """
        ),
        code(
            r'''
            sw = Value(2.0, label="w")
            sx = Value(3.0, label="x")
            sb = Value(1.0, label="b")
            sy = Value(10.0, label="y")

            sm = multiply(sw, sx, "m")
            sa = add(sm, sb, "a")
            se = subtract(sa, sy, "e")
            sL = square(se, "L")

            print("What m=wx stored during forward:")
            for link in sm.parents:
                print(f"  parent {link.value.label}: local ∂m/∂{link.value.label} = {link.local_grad:g}")

            draw_graph(sL, show_grad=False)
            ''',
            "Inspect one pair of parent links, then draw the forward graph",
        ),
        md(
            r"""
            Now our whole reverse pass is also one line:
            """
        ),
        code(
            r'''
            steps = backward(sL)
            show_backward_steps(steps)

            draw_graph(sL, show_grad=True)
            ''',
            "Run and display every backward edge, then redraw the graph with gradients",
        ),
        md(
            r"""
            The trace contains every reverse edge. For example, the square sends $-6$ into `e.grad`. On the next
            operation, that same stored number becomes the upstream gradient $g_e$ for subtraction.

            A single row's product is a **downstream contribution**. If several paths return to one value, each row
            adds into the same `parent.grad` buffer; only their sum is the full gradient at that parent.

            Finally, check that our tiny engine and PyTorch agree at every named value.
            """
        ),
        code(
            r'''
            scratch_nodes = {"w": sw, "x": sx, "m": sm, "b": sb,
                             "a": sa, "y": sy, "e": se, "L": sL}

            for name, node in scratch_nodes.items():
                torch_value, torch_grad = torch_reference[name]
                assert node.data == torch_value
                assert node.grad == torch_grad

            print("✓ Every value and gradient matches PyTorch.")
            ''',
            "Verify exact scratch-to-PyTorch parity",
        ),
        md(
            r"""
            ## 3 · One neuron: fused sigmoid or atomic sigmoid?

            Now use a slightly larger graph:

            $$
            m=wx,\qquad z=m+b,\qquad s=\sigma(z),\qquad e=s-y,\qquad L=e^2.
            $$

            Choose $w=0.5$, $x=2$, $b=-1$, and $y=1$. Then $z=0$, $s=0.5$, and $L=0.25$,
            so the backward numbers stay readable.

            We will build the sigmoid in two ways:

            - **fused autograd primitive:** one operation $s=\sigma(z)$;
            - **atomic graph:** $n=-z$, $q=\exp(n)$, $d=1+q$, and $s=1/d$.

            “Fused” here describes the autograd graph: several local steps are packaged behind one operation node.
            It does not mean that we are skipping the chain rule.
            """
        ),
        code(
            r'''
            import math

            def negate(u, label):
                return Value(-u.data, label,
                             parents=(ParentLink(u, -1.0),), op="−")

            def exponential(u, label):
                out = math.exp(u.data)
                return Value(out, label,
                             parents=(ParentLink(u, out),), op="exp")

            def plus_one(u, label):
                return Value(1.0 + u.data, label,
                             parents=(ParentLink(u, 1.0),), op="+1")

            def reciprocal(u, label):
                return Value(1.0 / u.data, label,
                             parents=(ParentLink(u, -1.0 / u.data**2),), op="1/x")

            def sigmoid(u, label):
                # Stable forward formula; backward reuses the saved output s.
                if u.data >= 0:
                    s = 1.0 / (1.0 + math.exp(-u.data))
                else:
                    exp_z = math.exp(u.data)
                    s = exp_z / (1.0 + exp_z)
                return Value(s, label,
                             parents=(ParentLink(u, s * (1.0 - s)),), op="σ")
            ''',
            "Define the four atomic sigmoid rules and one fused sigmoid rule",
        ),
        md(
            r"""
            The fused rule stores one local derivative:

            $$
            \frac{\partial s}{\partial z}=s(1-s).
            $$

            The atomic graph stores four local derivatives. Their product is the same quantity:

            $$
            \underbrace{\left(-\frac{1}{d^2}\right)}_{s=1/d}
            \underbrace{(1)}_{d=1+q}
            \underbrace{(q)}_{q=\exp(n)}
            \underbrace{(-1)}_{n=-z}
            =\frac{q}{d^2}=s(1-s).
            $$
            """
        ),
        code(
            r'''
            def build_sigmoid_neuron(*, fused):
                nodes = {
                    "w": Value(0.5, label="w"),
                    "x": Value(2.0, label="x"),
                    "b": Value(-1.0, label="b"),
                    "y": Value(1.0, label="y"),
                }
                nodes["m"] = multiply(nodes["w"], nodes["x"], "m")
                nodes["z"] = add(nodes["m"], nodes["b"], "z")

                if fused:
                    nodes["s"] = sigmoid(nodes["z"], "s")
                else:
                    nodes["n"] = negate(nodes["z"], "n")
                    nodes["q"] = exponential(nodes["n"], "q")
                    nodes["d"] = plus_one(nodes["q"], "d")
                    nodes["s"] = reciprocal(nodes["d"], "s")

                nodes["e"] = subtract(nodes["s"], nodes["y"], "e")
                nodes["L"] = square(nodes["e"], "L")
                return nodes

            fused_nodes = build_sigmoid_neuron(fused=True)
            atomic_nodes = build_sigmoid_neuron(fused=False)
            fused_steps = backward(fused_nodes["L"])
            atomic_steps = backward(atomic_nodes["L"])

            display(HTML("<h4>Fused sigmoid · 8 reverse edges</h4>"))
            display(draw_graph(fused_nodes["L"], show_grad=True, min_width=1180))
            show_backward_steps(fused_steps, highlight_outputs={"s"})

            display(HTML("<h4 style='margin-top:24px'>Atomic sigmoid · 11 reverse edges</h4>"))
            display(draw_graph(atomic_nodes["L"], show_grad=True, min_width=1700))
            show_backward_steps(atomic_steps, highlight_outputs={"s", "d", "q", "n"})
            ''',
            "Build both sigmoid graphs and show every backward edge",
        ),
        md(
            r"""
            The blue-highlighted cards are the only part that changed:

            - fused sigmoid: one update, $g_z=g_s\,s(1-s)=(-1)(0.25)=-0.25$;
            - atomic sigmoid: four updates, ending with the same $g_z=-0.25$.

            Fusion therefore gives a smaller graph and fewer intermediate gradient buffers. A real library can also
            use a numerically stable sigmoid implementation. The mathematics is unchanged: the single fused local
            derivative is exactly the product of the four atomic local derivatives.
            """
        ),
        code(
            r'''
            common = ("w", "x", "m", "b", "z", "s", "y", "e", "L")
            for name in common:
                assert math.isclose(fused_nodes[name].data, atomic_nodes[name].data)
                assert math.isclose(fused_nodes[name].grad, atomic_nodes[name].grad)

            assert [(s["output"], s["parent"]) for s in fused_steps] == [
                ("L", "e"), ("e", "s"), ("e", "y"), ("s", "z"),
                ("z", "m"), ("z", "b"), ("m", "w"), ("m", "x"),
            ]
            assert [(s["output"], s["parent"]) for s in atomic_steps] == [
                ("L", "e"), ("e", "s"), ("e", "y"), ("s", "d"),
                ("d", "q"), ("q", "n"), ("n", "z"), ("z", "m"),
                ("z", "b"), ("m", "w"), ("m", "x"),
            ]

            expected_sigmoid = {
                "w": (0.5, -0.5), "x": (2.0, -0.125), "m": (1.0, -0.25),
                "b": (-1.0, -0.25), "z": (0.0, -0.25), "s": (0.5, -1.0),
                "y": (1.0, 1.0), "e": (-0.5, -1.0), "L": (0.25, 1.0),
            }
            for name, (value, grad) in expected_sigmoid.items():
                assert math.isclose(fused_nodes[name].data, value)
                assert math.isclose(fused_nodes[name].grad, grad)

            fused_local = next(
                step["local"] for step in fused_steps
                if step["output"] == "s" and step["parent"] == "z"
            )
            atomic_locals = [
                step["local"] for step in atomic_steps
                if step["output"] in {"s", "d", "q", "n"}
            ]
            assert math.isclose(math.prod(atomic_locals), fused_local)

            tw = torch.tensor(0.5, requires_grad=True)
            tx = torch.tensor(2.0, requires_grad=True)
            tb = torch.tensor(-1.0, requires_grad=True)
            ty = torch.tensor(1.0, requires_grad=True)
            tL = (torch.sigmoid(tw * tx + tb) - ty) ** 2
            tL.backward()

            assert math.isclose(fused_nodes["w"].grad, tw.grad.item())
            assert math.isclose(fused_nodes["x"].grad, tx.grad.item())
            assert math.isclose(fused_nodes["b"].grad, tb.grad.item())
            assert math.isclose(fused_nodes["y"].grad, ty.grad.item())

            print("✓ Fused, atomic, and PyTorch agree.")
            print("  sigmoid local: 4 atomic factors = 1 fused factor =", fused_local)
            print("  final gradients: w = -0.5, x = -0.125, b = -0.25, y = 1")
            ''',
            "Verify fused-to-atomic equivalence and PyTorch parity",
        ),
        md(
            r"""
            ## Takeaway

            Both systems do the same three things:

            1. run the forward operations and store parent links plus local derivatives,
            2. start with $g_L=1$ in the loss's `.grad` buffer,
            3. compute <span style="color:#2C7A7B;font-weight:700">upstream</span>
               $\times$ <span style="color:#2B6CB0;font-weight:700">local</span>
               $=$ <span style="color:#EB811B;font-weight:700">downstream contribution</span>, then add it to
               the parent's `.grad` buffer.

            Our tiny `Value` record and local rules make those steps visible. Fusion does not change the calculus;
            it packages a product of local derivatives behind one operation. PyTorch generalizes these ideas to
            tensors, neural-network layers, accelerators, and large models.
            """
        ),
    ]

    for index, cell in enumerate(cells, start=1):
        cell["id"] = f"scalar-autograd-simple-{index:02d}"

    return nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
            "course": {
                "lecture": 4,
                "title": "Computation Graphs, Backpropagation & Autograd",
                "evidence": "exact scalar example; traced backward; fused-versus-atomic sigmoid parity",
            },
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-execute", action="store_true", help="Write without executing")
    args = parser.parse_args()

    notebook = build_notebook()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if not args.no_execute:
        NotebookClient(
            notebook,
            timeout=240,
            kernel_name="python3",
            record_timing=False,
            resources={"metadata": {"path": str(ROOT)}},
            allow_errors=False,
        ).execute()
    nbformat.write(notebook, OUTPUT)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
