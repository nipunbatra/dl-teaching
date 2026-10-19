"""SVG helpers shared with Generative Modelling I (scripts/diagrams.py there).

    python scripts/diagrams.py      # writes figures/diagrams/*.svg

Each diagram is plain SVG. Groups that appear on later clicks carry data-b="n";
the slide runtime reveals them. Numbers come from figures/data/numbers.json, so
the diagram for "our 3" shows what the trained model actually computed.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "diagrams"
OUT.mkdir(parents=True, exist_ok=True)
NUM = json.loads((ROOT / "figures" / "data" / "numbers.json").read_text())

INK, INK2, INK3, LINE = "#14171F", "#4A5160", "#6B7280", "#D9DDE5"
ROLE = {
    "x": ("#245EDB", "#E4ECFF"), "enc": ("#8B2CDE", "#F1E5FC"), "dec": ("#0F766E", "#D9F2EF"),
    "z": ("#AA4E08", "#FCEFD9"), "loss": ("#BE123C", "#FDE2E7"), "disc": ("#334155", "#E2E8F0"),
    "grey": ("#9AA1AD", "#F3F4F6"), "ink": (INK, "#FFFFFF"), "xt": ("#334155", "#E2E8F0"),
}
UI = "'Avenir Next', 'Avenir', 'Segoe UI', sans-serif"
MATH = "KaTeX_Math, 'Times New Roman', serif"
MAIN = "KaTeX_Main, 'Times New Roman', serif"
IMG = "figures/data/"


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def fs(size):
    """Projector-friendly label sizes: nothing below 18 px."""
    return size if size >= 20 else round(max(18, size * 1.22))


MATHY = re.compile(r"[μσεϵφθψπΣ‖∂∼~²√∫∑]|(?<![\w])-\d")


def check_plain(s):
    """Plain SVG text must not contain maths: that goes through KaTeX (math / label)."""
    if MATHY.search(str(s)):
        raise ValueError(f"maths in plain text, use d.math or d.label: {s!r}")


class SVG:
    def __init__(self, name, w, h):
        self.name, self.w, self.h = name, w, h
        self.items, self.stack = [], []
        self.markers = {}

    # ------------------------------------------------------------ structure
    def add(self, s):
        self.items.append(s)

    def begin(self, b=None, cls=None, extra=""):
        attrs = ""
        if b is not None:
            attrs += f' data-b="{b}"'
        if cls:
            attrs += f' class="{cls}"'
        self.add(f"<g{attrs}{extra}>")

    def end(self):
        self.add("</g>")

    def marker(self, color):
        mid = f"{self.name}-ah-{color.strip('#')}"
        if mid not in self.markers:
            self.markers[mid] = (f'<marker id="{mid}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" '
                                 f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{color}"/></marker>')
        return mid

    # ------------------------------------------------------------ primitives
    def rect(self, x, y, w, h, stroke, fill, r=12, sw=2.2, dash=None, opacity=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        o = f' opacity="{opacity}"' if opacity else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}{o}/>')

    def text(self, x, y, s, size=20, color=INK, weight=500, anchor="middle", italic=False):
        check_plain(s)
        size = fs(size)
        st = ' font-style="italic"' if italic else ""
        self.add(f'<text x="{x}" y="{y}" font-family="{UI}" font-size="{size}" font-weight="{weight}" fill="{color}" '
                 f'text-anchor="{anchor}"{st}>{esc(s)}</text>')

    def label(self, x, y, html, size=20, color=INK, weight=500, anchor="middle", width=760):
        """Text that may contain $...$ maths. Rendered as HTML so KaTeX typesets it.
        (x, y) is the baseline point, as for SVG <text>."""
        size = fs(size)
        x0 = {"middle": x - width / 2, "start": x, "end": x - width}[anchor]
        align = {"middle": "center", "start": "left", "end": "right"}[anchor]
        h = size * 2.2
        self.add(f'<foreignObject x="{x0:.1f}" y="{y - size * 1.25:.1f}" width="{width}" height="{h:.1f}" overflow="visible">'
                 f'<div class="fo" style="font-size:{size}px;color:{color};font-weight:{weight};text-align:{align};'
                 f'height:{size * 1.6:.1f}px;line-height:{size * 1.6:.1f}px">{html}</div></foreignObject>')

    def math(self, x, y, tex, size=24, color=INK, anchor="middle", weight=400):
        """Pure maths, typeset by KaTeX."""
        self.label(x, y, f"${tex}$", size, color, weight, anchor)

    def line(self, pts, color=INK3, sw=2.4, head=True, dash=None, start_head=False):
        d = "M" + " L".join(f"{x},{y}" for x, y in pts)
        m = self.marker(color)
        h = f' marker-end="url(#{m})"' if head else ""
        if start_head:
            h += f' marker-start="url(#{m})"'
        ds = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linejoin="round"{ds}{h}/>')

    def curve(self, d, color=INK3, sw=2.4, head=True, dash=None):
        m = self.marker(color)
        h = f' marker-end="url(#{m})"' if head else ""
        ds = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}"{ds}{h}/>')

    def image(self, x, y, w, h, href, role="x", border=True, sw=2.5, smooth=False):
        st = "" if smooth else ' style="image-rendering:pixelated"'
        self.add(f'<image x="{x}" y="{y}" width="{w}" height="{h}" href="{IMG}{href}" preserveAspectRatio="none"{st}/>')
        if border:
            c = ROLE[role][0]
            self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="none" stroke="{c}" stroke-width="{sw}"/>')

    def node(self, x, y, w, h, role, title, sub=None, size=22, dash=None):
        c, t = ROLE[role]
        self.rect(x, y, w, h, c, t, dash=dash)
        cy = y + h / 2 + (size * 0.35 if not sub else -2)
        self.label(x + w / 2, cy, title, size, c, 650, width=max(w, 300)) if "$" in title else self.text(x + w / 2, cy, title, size, c, 650)
        if sub:
            self.label(x + w / 2, cy + size * 0.95, sub, 15, INK3, 500, width=max(w, 300)) if "$" in sub \
                else self.text(x + w / 2, cy + size * 0.95, sub, 15, INK3, 500)

    def trapezoid(self, x, y_mid, w, hl, hr, role, title, sub=None, dash=None, opacity=None):
        c, t = ROLE[role]
        pts = [(x, y_mid - hl / 2), (x + w, y_mid - hr / 2), (x + w, y_mid + hr / 2), (x, y_mid + hl / 2)]
        d = f' stroke-dasharray="{dash}"' if dash else ""
        o = f' opacity="{opacity}"' if opacity else ""
        self.add(f'<polygon points="{" ".join(f"{a},{b}" for a, b in pts)}" fill="{t}" stroke="{c}" '
                 f'stroke-width="2.4" stroke-linejoin="round"{d}{o}/>')
        self.text(x + w / 2, y_mid + (8 if not sub else -2), title, 23, c, 650)
        if sub:
            if "$" in sub:
                self.label(x + w / 2, y_mid + 24, sub, 14, INK3, 500, width=max(w, 300))
            else:
                self.text(x + w / 2, y_mid + 22, sub, 14, INK3, 500)

    def vec(self, x, y, values, role, cw=70, ch=40, vertical=True, fmt="{:.2f}", size=17):
        c, t = ROLE[role]
        for i, v in enumerate(values):
            xx, yy = (x, y + i * ch) if vertical else (x + i * cw, y)
            self.add(f'<rect x="{xx}" y="{yy}" width="{cw}" height="{ch}" fill="{t}" stroke="{c}" stroke-width="2"/>')
            s = fmt.format(v) if not isinstance(v, str) else v
            s = s.replace("-", "−")
            self.add(f'<text x="{xx + cw / 2}" y="{yy + ch / 2 + 6}" font-family="SF Mono, Menlo, monospace" '
                     f'font-size="{size}" fill="{INK}" text-anchor="middle">{esc(s)}</text>')

    def cloud(self, cx, cy, rx, ry, role="enc"):
        c, t = ROLE[role]
        for k, op in ((1.0, .14), (0.7, .22), (0.4, .35)):
            self.add(f'<ellipse cx="{cx}" cy="{cy}" rx="{rx * k}" ry="{ry * k}" fill="{c}" fill-opacity="{op}"/>')
        self.add(f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="none" stroke="{c}" stroke-width="2"/>')

    def plane(self, cx, cy, r, rings=True, label=True, color=None):
        color = color or ROLE["z"][0]
        self.add(f'<line x1="{cx - r}" y1="{cy}" x2="{cx + r}" y2="{cy}" stroke="#C7CCD4" stroke-width="1.4"/>')
        self.add(f'<line x1="{cx}" y1="{cy - r}" x2="{cx}" y2="{cy + r}" stroke="#C7CCD4" stroke-width="1.4"/>')
        if rings:
            for k in (0.33, 0.66, 1.0):
                self.add(f'<circle cx="{cx}" cy="{cy}" r="{r * k * 0.92}" fill="none" stroke="{color}" '
                         f'stroke-opacity=".55" stroke-width="1.4" stroke-dasharray="4 3"/>')
        if label:
            self.math(cx + r + 4, cy + 24, "z_1", 17, INK3, "end", )
            self.math(cx + 14, cy - r + 12, "z_2", 17, INK3, "start")

    def dice(self, x, y, s=54, role="z"):
        c, t = ROLE[role]
        self.rect(x, y, s, s, c, t, r=10, sw=2.4)
        for dx, dy in ((.27, .27), (.73, .27), (.5, .5), (.27, .73), (.73, .73)):
            self.add(f'<circle cx="{x + s * dx}" cy="{y + s * dy}" r="{s * .075}" fill="{c}"/>')

    def lock(self, x, y, color=INK2, s=1.0):
        self.add(f'<g transform="translate({x},{y}) scale({s})"><path d="M-9,-4 v-7 a9,9 0 0 1 18,0 v7" fill="none" '
                 f'stroke="{color}" stroke-width="3"/><rect x="-13" y="-4" width="26" height="20" rx="4" fill="{color}"/>'
                 f'<circle cx="0" cy="5" r="3" fill="#fff"/></g>')

    def cross(self, x, y, s=14, color=ROLE["loss"][0], sw=4):
        self.add(f'<path d="M{x - s},{y - s} L{x + s},{y + s} M{x + s},{y - s} L{x - s},{y + s}" stroke="{color}" '
                 f'stroke-width="{sw}" stroke-linecap="round"/>')

    def scissors(self, x, y, color=ROLE["loss"][0]):
        self.add(f'<g transform="translate({x},{y})" fill="none" stroke="{color}" stroke-width="2.6">'
                 f'<circle cx="-7" cy="10" r="5"/><circle cx="7" cy="10" r="5"/>'
                 f'<path d="M-4,6 L8,-12 M4,6 L-8,-12"/></g>')

    def save(self):
        defs = "<defs>" + "".join(self.markers.values()) + "</defs>"
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" class="diagram" '
               f'role="img" aria-label="{esc(self.name)}">{defs}{"".join(self.items)}</svg>')
        (OUT / f"{self.name}.svg").write_text(svg)


C = {k: v[0] for k, v in ROLE.items()}

