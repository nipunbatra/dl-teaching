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
            # One scalar graph: PyTorch, then autograd from scratch

            We will use one graph throughout:

            $$
            m=wx,\qquad a=m+b,\qquad e=a-y,\qquad L=e^2
            $$

            with $w=2$, $x=3$, $b=1$, and $y=10$.

            First we let PyTorch differentiate it. Then we build the smallest useful version of the same idea
            ourselves. The point is not to replace PyTorch—it is to see what `.backward()` does.
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
            - its derivative in `grad`,
            - its parent values, and
            - the local derivative leading to each parent.

            Each operation computes one forward value and records its local derivatives. One generic `backward`
            function then walks the graph in reverse. It uses `+=` because several paths may contribute to the same
            value.
            """
        ),
        code(
            r'''
            class Value:
                def __init__(self, data, label="", parents=(), op=""):
                    self.data = float(data)
                    self.grad = 0.0
                    self.label = label
                    self.parents = tuple(parents)  # (parent, local derivative)
                    self.op = op

            def multiply(u, v, label):
                return Value(u.data * v.data, label,
                             parents=((u, v.data), (v, u.data)), op="×")

            def add(u, v, label):
                return Value(u.data + v.data, label,
                             parents=((u, 1.0), (v, 1.0)), op="+")

            def subtract(u, v, label):
                return Value(u.data - v.data, label,
                             parents=((u, 1.0), (v, -1.0)), op="−")

            def square(u, label):
                return Value(u.data ** 2, label,
                             parents=((u, 2 * u.data),), op="²")

            def backward(root):
                order, seen = [], set()
                def visit(node):
                    if id(node) in seen:
                        return
                    seen.add(id(node))
                    for parent, _local in node.parents:
                        visit(parent)
                    order.append(node)

                visit(root)
                root.grad = 1.0
                for node in reversed(order):
                    for parent, local in node.parents:
                        parent.grad += node.grad * local
            ''',
            "Define the tiny scalar record, four local rules, and reverse traversal",
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
            from IPython.display import HTML

            def draw_graph(root, show_grad=True):
                nodes, seen = [], set()
                def visit(node):
                    if id(node) in seen:
                        return
                    seen.add(id(node))
                    nodes.append(node)
                    for parent, _local in node.parents:
                        visit(parent)
                visit(root)

                dot = Digraph(format="svg")
                dot.attr(rankdir="LR", bgcolor="transparent", pad="0.15",
                         nodesep="0.25", ranksep="0.45")
                dot.attr("node", fontname="Helvetica", fontsize="11")
                dot.attr("edge", fontname="Helvetica", fontsize="9", color="#60777b")

                for node in nodes:
                    node_id = "value_" + node.label
                    grad = f"{node.grad:g}" if show_grad else "—"
                    fill = "#fff1df" if show_grad and node.grad != 0 else "#ffffff"
                    dot.node(
                        node_id,
                        label=f"{{ {node.label} | value {node.data:g} | grad {grad} }}",
                        shape="record", style="rounded,filled", fillcolor=fill,
                        color="#1f3a40", fontcolor="#1f3a40",
                    )
                    if node.op:
                        op_id = "op_" + node.label
                        dot.node(op_id, label=node.op, shape="circle", fixedsize="true",
                                 width="0.32", style="filled", fillcolor="#ef7d00",
                                 color="#ef7d00", fontcolor="white")
                        dot.edge(op_id, node_id)
                        for parent, _local in node.parents:
                            dot.edge("value_" + parent.label, op_id)
                svg = dot.pipe(format="svg").decode("utf-8")
                svg = svg.replace(
                    "<svg ",
                    '<svg style="width:100%;height:auto;min-width:780px" ',
                    1,
                )
                return HTML(
                    '<div style="max-width:100%;overflow-x:auto">'
                    '<div style="min-width:780px">' + svg + '</div></div>'
                )
            ''',
            "Load Graphviz and define the compact Karpathy-style graph drawing helper",
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

            draw_graph(sL, show_grad=False)
            ''',
            "Build and draw the scratch forward graph before backward",
        ),
        md(
            r"""
            Now our whole reverse pass is also one line:
            """
        ),
        code(
            r'''
            backward(sL)

            scratch_nodes = {"w": sw, "x": sx, "m": sm, "b": sb,
                             "a": sa, "y": sy, "e": se, "L": sL}

            print(f"{'node':<5} {'value':>8} {'grad':>8}")
            for name, node in scratch_nodes.items():
                print(f"{name:<5} {node.data:8.1f} {node.grad:8.1f}")

            draw_graph(sL, show_grad=True)
            ''',
            "Run scratch backward, print the ledger, and redraw the graph with gradients",
        ),
        md(
            r"""
            Read the backward pass from right to left:

            | local rule | calculation | gradients produced |
            |---|---|---|
            | $L=e^2$ | $1\times 2e=1\times(-6)$ | $g_e=-6$ |
            | $e=a-y$ | $-6\times(1,-1)$ | $g_a=-6,\ g_y=6$ |
            | $a=m+b$ | $-6\times(1,1)$ | $g_m=-6,\ g_b=-6$ |
            | $m=wx$ | $-6\times(x,w)$ | $g_w=-18,\ g_x=-12$ |

            Finally, check that our tiny engine and PyTorch agree at every named value.
            """
        ),
        code(
            r'''
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
            ## Takeaway

            Both systems do the same three things:

            1. run the forward operations and remember the graph,
            2. start with $g_L=1$,
            3. apply each local derivative in reverse and add contributions.

            Our tiny `Value` record and four local rules make those steps visible. PyTorch generalizes them to
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
                "evidence": "exact scalar example; PyTorch and minimal scratch-engine parity",
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
