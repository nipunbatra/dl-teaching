#!/usr/bin/env python3
"""Build and execute the focused scalar-autograd teaching notebook."""

from __future__ import annotations

import argparse
import base64
import html
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


def embedded_svg_html(source: str, *, alt: str, min_width: int) -> str:
    """Embed one SVG as a vector data image that Jupyter and Pandoc both keep."""
    normalized = textwrap.dedent(source).strip()
    svg_start = normalized.index("<svg ")
    svg_end = normalized.index("</svg>", svg_start) + len("</svg>")
    svg = normalized[svg_start:svg_end].replace(
        "<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ', 1
    )
    payload = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    image = (
        f'<img src="data:image/svg+xml;base64,{payload}" '
        f'alt="{html.escape(alt, quote=True)}" '
        f'style="display:block;width:100%;height:auto;min-width:{min_width}px;">'
    )
    embedded = normalized[:svg_start] + image + normalized[svg_end:]
    return "\n".join(line for line in embedded.splitlines() if line.strip())


def worked_scalar_graph_html() -> str:
    """Return the complete numerical graph as notebook-native, responsive SVG."""
    return embedded_svg_html(
        r"""
        <div style="max-width:100%;overflow-x:auto;margin:1.25rem 0 0.5rem;padding:0.25rem 0;">
          <svg viewBox="0 0 1120 430" role="img" aria-labelledby="scalar-graph-title scalar-graph-desc"
               style="display:block;width:100%;height:auto;min-width:1040px;font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
            <title id="scalar-graph-title">The complete scalar computation graph with forward values and backward gradients</title>
            <desc id="scalar-graph-desc">Forward flows from w, x, b, and y through multiply, add, subtract, and square to L. Backward flows from L to every input. Each value node shows its forward value and its final loss gradient.</desc>
            <defs>
              <marker id="intro-forward-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#60777B"/>
              </marker>
              <marker id="intro-backward-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#2C7A7B"/>
              </marker>
            </defs>

            <text x="20" y="25" font-size="18" font-weight="750" fill="#1F3A40">The complete graph we will reproduce</text>
            <line x1="390" y1="20" x2="449" y2="20" stroke="#60777B" stroke-width="2.5" marker-end="url(#intro-forward-arrow)"/>
            <text x="461" y="25" font-size="14" font-weight="650" fill="#60777B">forward: compute values</text>
            <line x1="805" y1="20" x2="746" y2="20" stroke="#2C7A7B" stroke-width="3" marker-end="url(#intro-backward-arrow)"/>
            <text x="817" y="25" font-size="14" font-weight="650" fill="#2C7A7B">backward: compute ∂L/∂node</text>

            <!-- forward edges -->
            <g fill="none" stroke="#60777B" stroke-width="2.4" marker-end="url(#intro-forward-arrow)">
              <path d="M 140 107 C 157 107, 163 130, 178 146"/>
              <path d="M 140 217 C 157 217, 163 195, 178 179"/>
              <path d="M 214 162 L 247 162"/>
              <path d="M 367 162 C 385 162, 389 185, 404 194"/>
              <path d="M 367 273 C 385 273, 390 227, 405 211"/>
              <path d="M 441 201 L 474 201"/>
              <path d="M 594 201 C 611 201, 616 224, 630 233"/>
              <path d="M 594 311 C 611 311, 616 266, 630 250"/>
              <path d="M 667 240 L 700 240"/>
              <path d="M 820 240 L 852 240"/>
              <path d="M 891 240 L 925 240"/>
            </g>

            <!-- value nodes -->
            <g stroke="#A9BABC" stroke-width="1.8">
              <rect x="20" y="70" width="120" height="74" rx="14" fill="#FFFFFF"/>
              <rect x="20" y="180" width="120" height="74" rx="14" fill="#FFFFFF"/>
              <rect x="247" y="125" width="120" height="74" rx="14" fill="#E8F7F5" stroke="#2C7A7B"/>
              <rect x="247" y="236" width="120" height="74" rx="14" fill="#FFFFFF"/>
              <rect x="474" y="164" width="120" height="74" rx="14" fill="#E8F7F5" stroke="#2C7A7B"/>
              <rect x="474" y="274" width="120" height="74" rx="14" fill="#FFFFFF"/>
              <rect x="700" y="203" width="120" height="74" rx="14" fill="#E8F7F5" stroke="#2C7A7B"/>
              <rect x="925" y="203" width="120" height="74" rx="14" fill="#FFF5E8" stroke="#EB811B"/>
            </g>

            <!-- operation nodes -->
            <g fill="#1F3A40" stroke="#1F3A40" stroke-width="2">
              <circle cx="196" cy="162" r="19"/>
              <circle cx="423" cy="201" r="19"/>
              <circle cx="649" cy="240" r="19"/>
              <circle cx="871" cy="240" r="19"/>
            </g>
            <g fill="#FFFFFF" font-size="22" font-weight="750" text-anchor="middle">
              <text x="196" y="169">×</text>
              <text x="423" y="209">+</text>
              <text x="649" y="247">−</text>
              <text x="871" y="247">²</text>
            </g>

            <!-- node labels, forward values, and final gradients -->
            <g text-anchor="middle">
              <g transform="translate(80 0)"><text y="91" font-size="18" font-weight="800" fill="#1F3A40">w</text><text y="113" font-size="13" fill="#60777B">value 2</text><text y="134" font-size="13" font-weight="750" fill="#2C7A7B">grad −18</text></g>
              <g transform="translate(80 0)"><text y="201" font-size="18" font-weight="800" fill="#1F3A40">x</text><text y="223" font-size="13" fill="#60777B">value 3</text><text y="244" font-size="13" font-weight="750" fill="#2C7A7B">grad −12</text></g>
              <g transform="translate(307 0)"><text y="146" font-size="18" font-weight="800" fill="#1F3A40">m</text><text y="168" font-size="13" fill="#60777B">value 6</text><text y="189" font-size="13" font-weight="750" fill="#2C7A7B">grad −6</text></g>
              <g transform="translate(307 0)"><text y="257" font-size="18" font-weight="800" fill="#1F3A40">b</text><text y="279" font-size="13" fill="#60777B">value 1</text><text y="300" font-size="13" font-weight="750" fill="#2C7A7B">grad −6</text></g>
              <g transform="translate(534 0)"><text y="185" font-size="18" font-weight="800" fill="#1F3A40">a</text><text y="207" font-size="13" fill="#60777B">value 7</text><text y="228" font-size="13" font-weight="750" fill="#2C7A7B">grad −6</text></g>
              <g transform="translate(534 0)"><text y="295" font-size="18" font-weight="800" fill="#1F3A40">y</text><text y="317" font-size="13" fill="#60777B">value 10</text><text y="338" font-size="13" font-weight="750" fill="#2C7A7B">grad 6</text></g>
              <g transform="translate(760 0)"><text y="224" font-size="18" font-weight="800" fill="#1F3A40">e</text><text y="246" font-size="13" fill="#60777B">value −3</text><text y="267" font-size="13" font-weight="750" fill="#2C7A7B">grad −6</text></g>
              <g transform="translate(985 0)"><text y="224" font-size="18" font-weight="800" fill="#1F3A40">L</text><text y="246" font-size="13" fill="#60777B">value 9</text><text y="267" font-size="13" font-weight="750" fill="#2C7A7B">grad 1 · seed</text></g>
            </g>

            <line x1="1018" y1="384" x2="93" y2="384" stroke="#2C7A7B" stroke-width="3.5" marker-end="url(#intro-backward-arrow)"/>
            <rect x="285" y="365" width="540" height="37" rx="18" fill="#FFFFFF"/>
            <text x="555" y="390" text-anchor="middle" font-size="15" font-weight="750" fill="#2C7A7B">backward propagates gradients: loss → intermediates → inputs</text>
          </svg>
        </div>
        <p style="margin:0.25rem 0 1.25rem;color:#52696D;font-size:0.95rem;">
          Forward computes left → right. Backward begins at <code>L.grad = 1</code> and travels right → left.
          The teal number in each box is that node's final <code>∂L/∂node</code>. On a phone, scroll sideways.
        </p>
        """,
        alt="Complete scalar computation graph with every forward value and loss gradient",
        min_width=1040,
    )


def parent_link_diagram_html() -> str:
    """Return a concrete parent/child and one-link backward-pass diagram."""
    return embedded_svg_html(
        r"""
        <div style="max-width:100%;overflow-x:auto;margin:1.25rem 0 0.5rem;padding:0.25rem 0;">
          <svg viewBox="0 0 1160 510" role="img" aria-labelledby="parent-link-title parent-link-desc"
               style="display:block;width:100%;height:auto;min-width:1160px;font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
            <title id="parent-link-title">Parent and child in the forward graph, and the data used for one backward update</title>
            <desc id="parent-link-desc">In the forward graph, u is an operand or direct parent used by operation f to create output v, the child. During backward, v dot grad is multiplied by the local derivative stored on the parent link. The temporary product is accumulated into u dot grad.</desc>
            <defs>
              <marker id="role-forward-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#60777B"/></marker>
              <marker id="role-upstream-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#2C7A7B"/></marker>
              <marker id="role-local-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#2B6CB0"/></marker>
              <marker id="role-downstream-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#EB811B"/></marker>
            </defs>

            <text x="25" y="28" font-size="18" font-weight="800" fill="#1F3A40">Forward construction gives “parent” and “child” their names</text>

            <rect x="30" y="70" width="195" height="96" rx="16" fill="#FFFFFF" stroke="#A9BABC" stroke-width="2"/>
            <text x="127" y="103" text-anchor="middle" font-size="25" font-weight="800" fill="#1F3A40">u</text>
            <text x="127" y="128" text-anchor="middle" font-size="14" font-weight="700" fill="#60777B">direct parent / operand</text>
            <text x="127" y="149" text-anchor="middle" font-size="13" fill="#60777B">an input used by f</text>

            <circle cx="292" cy="118" r="34" fill="#1F3A40"/>
            <text x="292" y="126" text-anchor="middle" font-size="24" font-weight="800" fill="#FFFFFF">f</text>

            <rect x="360" y="60" width="275" height="116" rx="16" fill="#E8F7F5" stroke="#2C7A7B" stroke-width="2"/>
            <text x="497" y="92" text-anchor="middle" font-size="25" font-weight="800" fill="#1F3A40">v = f(u)</text>
            <text x="497" y="117" text-anchor="middle" font-size="14" font-weight="700" fill="#2C7A7B">output / child created from u</text>
            <text x="497" y="143" text-anchor="middle" font-size="13" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" fill="#1F3A40">v.parents stores ParentLink</text>
            <text x="497" y="163" text-anchor="middle" font-size="12.5" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" fill="#60777B">value=u · local_grad=∂v/∂u</text>

            <text x="690" y="126" text-anchor="middle" font-size="30" font-weight="750" fill="#60777B">…</text>
            <rect x="755" y="80" width="150" height="76" rx="16" fill="#FFF5E8" stroke="#EB811B" stroke-width="2"/>
            <text x="830" y="112" text-anchor="middle" font-size="24" font-weight="800" fill="#1F3A40">L</text>
            <text x="830" y="136" text-anchor="middle" font-size="13" font-weight="700" fill="#60777B">eventual loss</text>

            <g fill="none" stroke="#60777B" stroke-width="2.6" marker-end="url(#role-forward-arrow)">
              <line x1="225" y1="118" x2="254" y2="118"/>
              <line x1="326" y1="118" x2="357" y2="118"/>
              <line x1="635" y1="118" x2="670" y2="118"/>
              <line x1="710" y1="118" x2="752" y2="118"/>
            </g>
            <text x="242" y="101" text-anchor="middle" font-size="12" font-weight="650" fill="#60777B">input</text>
            <text x="343" y="101" text-anchor="middle" font-size="12" font-weight="650" fill="#60777B">creates</text>

            <rect x="940" y="66" width="195" height="104" rx="14" fill="#F4F7F7"/>
            <text x="958" y="91" font-size="13" font-weight="750" fill="#1F3A40">This notebook's terms:</text>
            <text x="958" y="116" font-size="13" fill="#52696D">u is a parent of v because</text>
            <text x="958" y="137" font-size="13" fill="#52696D">f used u to construct v.</text>
            <text x="958" y="158" font-size="13" fill="#52696D">Other libraries may differ.</text>

            <line x1="25" y1="210" x2="1135" y2="210" stroke="#D8E1E2" stroke-width="1.5" stroke-dasharray="6 7"/>
            <text x="25" y="244" font-size="18" font-weight="800" fill="#1F3A40">Backward uses one stored parent link for one chain-rule update</text>
            <text x="1135" y="244" text-anchor="end" font-size="13" font-weight="700" fill="#2C7A7B">loss side → parent side</text>
            <text x="1027" y="271" text-anchor="middle" font-size="13" font-weight="700" fill="#2C7A7B">after seeding L.grad = 1</text>

            <!-- one-edge backward flow, read from right to left -->
            <rect x="935" y="282" width="185" height="82" rx="14" fill="#E8F7F5" stroke="#2C7A7B" stroke-width="2"/>
            <text x="1027" y="308" text-anchor="middle" font-size="14" font-weight="800" fill="#2C7A7B">UPSTREAM · STORED</text>
            <text x="1027" y="334" text-anchor="middle" font-size="17" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-weight="750" fill="#1F3A40">v.grad = gᵥ</text>
            <text x="1027" y="354" text-anchor="middle" font-size="13" fill="#52696D">gᵥ = ∂L/∂v</text>

            <rect x="660" y="388" width="250" height="82" rx="14" fill="#EAF2FC" stroke="#2B6CB0" stroke-width="2"/>
            <text x="785" y="414" text-anchor="middle" font-size="14" font-weight="800" fill="#2B6CB0">LOCAL · STORED ON LINK</text>
            <text x="785" y="440" text-anchor="middle" font-size="15" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-weight="750" fill="#1F3A40">link.local_grad</text>
            <text x="785" y="460" text-anchor="middle" font-size="13" fill="#52696D">= ∂v/∂u</text>

            <circle cx="716" cy="323" r="29" fill="#1F3A40"/>
            <text x="716" y="331" text-anchor="middle" font-size="24" font-weight="800" fill="#FFFFFF">×</text>

            <rect x="365" y="282" width="250" height="82" rx="14" fill="#FFF1E5" stroke="#EB811B" stroke-width="2"/>
            <text x="490" y="308" text-anchor="middle" font-size="14" font-weight="800" fill="#EB811B">DOWNSTREAM · TEMPORARY</text>
            <text x="490" y="335" text-anchor="middle" font-size="16" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-weight="750" fill="#1F3A40">Δgᵤ = gᵥ × ∂v/∂u</text>
            <text x="490" y="355" text-anchor="middle" font-size="13" fill="#52696D">one edge's contribution</text>

            <rect x="30" y="282" width="265" height="82" rx="14" fill="#E8F7F5" stroke="#2C7A7B" stroke-width="2"/>
            <text x="162" y="308" text-anchor="middle" font-size="14" font-weight="800" fill="#2C7A7B">PARENT BUFFER · STORED</text>
            <text x="162" y="335" text-anchor="middle" font-size="16" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-weight="750" fill="#1F3A40">u.grad += Δgᵤ</text>
            <text x="162" y="355" text-anchor="middle" font-size="13" fill="#52696D">accumulates all child paths</text>

            <line x1="935" y1="323" x2="750" y2="323" stroke="#2C7A7B" stroke-width="3" marker-end="url(#role-upstream-arrow)"/>
            <path d="M 785 388 C 785 360, 756 344, 742 338" fill="none" stroke="#2B6CB0" stroke-width="3" marker-end="url(#role-local-arrow)"/>
            <line x1="687" y1="323" x2="618" y2="323" stroke="#EB811B" stroke-width="3" marker-end="url(#role-downstream-arrow)"/>
            <line x1="365" y1="323" x2="298" y2="323" stroke="#EB811B" stroke-width="3" marker-end="url(#role-downstream-arrow)"/>

            <text x="575" y="494" text-anchor="middle" font-size="14" font-weight="650" fill="#52696D">The traversal direction reverses; “parent” and “child” still refer to how the forward graph was constructed.</text>
          </svg>
        </div>
        <p style="margin:0.25rem 0 0.75rem;color:#52696D;font-size:0.95rem;">
          <code>v.parents</code> owns the <code>ParentLink</code>, and that link points back to <code>u</code>.
          The engine does not need <code>u</code> to keep a list of its children. On a phone, scroll sideways.
        </p>
        <div style="margin:0.4rem 0 1.25rem;padding:0.85rem 1rem;border-left:4px solid #2C7A7B;background:#F4FAF9;border-radius:0 10px 10px 0;color:#29464B;">
          <strong>Concrete link from our graph:</strong> in <code>m = w × x</code>, choose <code>v = m</code> and the parent <code>u = w</code>.
          Backward reads upstream <code>m.grad = −6</code>, reads local <code>∂m/∂w = x = 3</code> from that link,
          computes the temporary contribution <code>−6 × 3 = −18</code>, and adds it to <code>w.grad</code>.
          The other parent link uses <code>u = x</code>, local derivative <code>w = 2</code>, and contributes <code>−12</code> to <code>x.grad</code>.
        </div>
        """,
        alt="Parent and child in a computation graph and the storage used for one backward update",
        min_width=1160,
    )


def build_notebook():
    cells = [
        md(
            textwrap.dedent(
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
            ).strip()
            + "\n\n"
            + worked_scalar_graph_html()
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
            textwrap.dedent(
                r"""
            ## 2 · A tiny autograd engine from scratch

            Suppose one operation $f$ takes a value $u$ and produces $v=f(u)$. In this notebook's convention,
            the **forward construction** makes $u$ a direct **parent** (or operand) of $v$, while $v$ is the output
            (or child) created from $u$. These words describe one operation in the computation graph—not a whole
            neural-network layer. Libraries sometimes choose different names; here, follow `v.parents`, whose links
            point back to the direct operands.
                """
            ).strip()
            + "\n\n"
            + parent_link_diagram_html()
            + "\n\n"
            + textwrap.dedent(
                r"""
            A `Value` therefore stores only:

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

            The downstream contribution is temporary: only the accumulated result in `u.grad` remains.
            A leaf such as `w` has no parents; the scalar loss `L` is the root passed to `backward`.
                """
            ).strip()
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
