"""Hand-built SVG diagrams for the GAN lecture (Generative Modelling I-B).

    python scripts/diagrams.py      # writes figures/diagrams/*.svg

Each diagram is plain SVG. Groups that appear on later clicks carry data-b="n";
the slide runtime reveals them. Numbers come from figures/data/numbers.json.
The drawing helpers and GAN diagrams are copied from the VAE lecture's diagrams.py.
"""
import json
import math
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
    "grey": ("#9AA1AD", "#F3F4F6"), "ink": (INK, "#FFFFFF"),
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
        html = html.replace("<", "&lt;").replace(">", "&gt;")  # maths like x_{<i} must not become HTML tags
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

    def image(self, x, y, w, h, href, role="x", border=True, sw=2.5):
        self.add(f'<image x="{x}" y="{y}" width="{w}" height="{h}" href="{IMG}{href}" preserveAspectRatio="none" '
                 f'style="image-rendering:pixelated"/>')
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


# =============================================================== GAN
def gan_base(d, step=None):
    """Shared GAN layout. step: None (overview), 'D' or 'G'."""
    gdim = step == "D"
    ddim = step == "G"
    d.dice(30, 230, 60)
    d.math(60, 322, "z", 24, C["z"])
    d.line([(96, 260), (130, 260)], C["z"])
    d.trapezoid(134, 260, 180, 80, 180, "dec", "generator G", r"weights $\theta$", opacity=".45" if gdim else None)
    if gdim:
        d.lock(224, 196, C["dec"])
    d.line([(318, 260), (350, 260)], C["dec"])
    d.image(356, 190, 140, 140, "gan-fake-one.png", "dec")
    d.text(426, 352, "fake", 18, C["dec"], 650)
    if step != "G":
        d.image(356, 10, 140, 140, "gan-real-one.png", "x")
        d.text(426, 172, "real", 18, C["x"], 650)
        d.line([(500, 80), (560, 80), (560, 160), (600, 160)], C["x"])
    d.line([(500, 260), (560, 260), (560, 200), (600, 200)], C["dec"])
    d.trapezoid(604, 180, 200, 210, 80, "disc", "discriminator D", r"weights $\psi$", opacity=".45" if ddim else None)
    d.line([(808, 180), (850, 180)], C["disc"])


def gan():
    d = SVG("gan", 1152, 380)
    d.begin(1)
    d.dice(30, 230, 60)
    d.math(60, 322, "z", 24, C["z"])
    d.line([(96, 260), (130, 260)], C["z"])
    d.trapezoid(134, 260, 180, 80, 180, "dec", "generator G", r"weights $\theta$")
    d.line([(318, 260), (350, 260)], C["dec"])
    d.image(356, 190, 140, 140, "gan-fake-one.png", "dec")
    d.text(426, 352, "fake", 18, C["dec"], 650)
    d.end()
    d.begin(2)
    d.image(356, 10, 140, 140, "gan-real-one.png", "x")
    d.text(426, 172, "real", 18, C["x"], 650)
    d.line([(500, 80), (560, 80), (560, 160), (600, 160)], C["x"])
    d.line([(500, 260), (560, 260), (560, 200), (600, 200)], C["dec"])
    d.trapezoid(604, 180, 200, 210, 80, "disc", "discriminator D", r"weights $\psi$")
    d.line([(808, 180), (850, 180)], C["disc"])
    d.node(856, 140, 270, 80, "disc", r"$D(x) = P(\text{real})$", r"one number in $(0, 1)$", size=22)
    d.end()
    d.begin(3)
    d.text(990, 272, "D wants: real → 1, fake → 0", 17, C["disc"], 650)
    d.text(990, 302, "G wants: fake → 1", 17, C["dec"], 650)
    d.end()
    d.save()


def gan_dstep():
    d = SVG("gan-dstep", 1152, 380)
    gan_base(d, "D")
    d.rect(856, 118, 290, 124, C["loss"], ROLE["loss"][1])
    d.math(1001, 156, r"-\log D(x_{\text{real}})", 19, C["loss"], weight=600)
    d.math(1001, 192, r"-\log\big(1 - D(x_{\text{fake}})\big)", 19, C["loss"], weight=600)
    d.text(1001, 226, "targets: real 1, fake 0", 15, INK3)
    d.begin(2)
    d.line([(856, 200), (806, 200)], C["loss"], 3)
    d.curve("M700,232 C680,270 640,280 600,276", C["loss"], 3)
    d.curve("M700,128 C680,90 640,84 600,90", C["loss"], 3)
    d.add(f'<line x1="530" y1="230" x2="530" y2="300" stroke="{C["loss"]}" stroke-width="3" stroke-dasharray="6 4"/>')
    d.scissors(530, 312)
    d.text(530, 362, "fake.detach()", 17, C["loss"], 650)
    d.text(224, 362, "G gets no gradient", 17, C["dec"], 650)
    d.end()
    d.save()


def gan_gstep():
    d = SVG("gan-gstep", 1152, 400)
    gan_base(d, "G")
    d.lock(704, 120, C["disc"])
    d.label(704, 92, r"$\psi$ frozen", 16, C["disc"], 650, width=200)
    d.rect(856, 140, 270, 80, C["loss"], ROLE["loss"][1])
    d.math(991, 182, r"-\log D(x_{\text{fake}})", 21, C["loss"], weight=600)
    d.text(991, 208, "target: fake → 1", 15, INK3)
    d.begin(2)
    d.line([(856, 196), (808, 196)], C["loss"], 3)
    d.curve("M806,206 C740,250 660,250 604,214", C["loss"], 3, dash="7 5")
    d.label(705, 306, r"through D, not into $\psi$", 15, C["loss"], 650, width=300)
    d.line([(560, 286), (500, 286)], C["loss"], 3)
    d.curve("M356,300 C330,320 300,316 290,300", C["loss"], 3)
    d.label(224, 392, r"G's weights $\theta$ get the gradient", 17, C["loss"], 650, width=400)
    d.end()
    d.save()


def freeze_detach():
    """Two rows, each in two clicks: first the wiring (predict!), then where the gradient goes."""
    d = SVG("freeze-detach", 1152, 330)
    for row, (title, y, mode) in enumerate((("detach(): cut the wire", 92, "detach"),
                                             ("requires_grad_(False): lock D's weights, keep the wire", 248, "freeze"))):
        d.begin(2 * row + 1)
        d.text(20, y - 50, title, 21, INK, 650, "start")
        d.node(20, y - 28, 150, 60, "dec", r"$G(z)$", size=22)
        d.line([(172, y + 2), (330, y + 2)], C["dec"])
        d.node(334, y - 28, 170, 60, "disc", r"$D$", size=24)
        d.line([(506, y + 2), (600, y + 2)], C["disc"])
        d.node(604, y - 28, 130, 60, "loss", "loss", size=22)
        if mode == "detach":
            d.scissors(256, y + 8)
        else:
            d.lock(474, y - 4, C["disc"], 0.7)
        d.end()
        d.begin(2 * row + 2)
        d.line([(604, y + 22), (508, y + 22)], C["loss"], 2.6)
        d.text(556, y + 50, "gradient", 15, C["loss"], 600)
        if mode == "detach":
            d.line([(332, y + 22), (296, y + 22)], C["loss"], 2.6)
            d.text(770, y + 2, "the gradient stops at the cut:", 19, C["loss"], 650, "start")
            d.text(770, y + 28, "G learns nothing", 19, C["loss"], 650, "start")
        else:
            d.line([(332, y + 22), (174, y + 22)], C["loss"], 2.6)
            d.text(770, y + 2, "D's weights stay put;", 19, C["loss"], 650, "start")
            d.text(770, y + 28, "G still learns", 19, C["loss"], 650, "start")
        d.end()
    d.save()


def cgan():
    d = SVG("cgan", 1152, 330)
    d.begin(1)
    d.dice(20, 200, 56)
    d.math(48, 288, "z", 22, C["z"])
    d.node(10, 60, 100, 70, "grey", r"$c = 7$", "a label", size=21)
    d.line([(80, 228), (160, 228)], C["z"])
    d.line([(110, 95), (130, 95), (130, 196), (160, 196)], C["grey"])
    d.trapezoid(164, 212, 180, 80, 170, "dec", "generator")
    d.math(254, 270, "G(z, c)", 19, C["dec"])
    d.line([(348, 212), (384, 212)], C["dec"])
    d.image(390, 142, 140, 140, "gan-seven.png", "dec")
    d.end()
    d.begin(2)
    d.line([(534, 212), (600, 212)], C["dec"])
    d.line([(110, 80), (580, 80), (580, 172), (600, 172)], C["grey"], dash="6 4")
    d.trapezoid(604, 192, 200, 200, 80, "disc", "discriminator")
    d.math(704, 240, "D(x, c)", 19, C["disc"])
    d.line([(808, 192), (850, 192)], C["disc"])
    d.node(856, 150, 280, 84, "disc", "real AND a 7?", r"judges the pair $(x, c)$", size=21)
    d.end()
    d.save()


# =============================================================== bridge from the VAE lecture
def bridge():
    """The part both lectures share: random z in, one pass through a generator, an image out."""
    d = SVG("bridge", 1152, 150)
    d.dice(250, 22, 64)
    d.math(282, 128, r"z \sim \N(0, I)", 20, C["z"])
    d.line([(322, 54), (392, 54)], C["z"])
    d.trapezoid(398, 54, 240, 80, 130, "dec", "generator G", r"weights $\theta$")
    d.line([(644, 54), (706, 54)], C["dec"])
    d.node(712, 14, 200, 80, "dec", r"an image $x$", "one pass, 28 × 28", size=22)
    d.save()


def _slab(d, x, cy, side, chans, role, shape):
    """A tensor drawn as a slab: height = spatial side, width grows with the channel count."""
    c, t = ROLE[role]
    h = 14 + side * 3.6
    w = 10 + 9 * math.log2(max(chans, 1)) if chans > 1 else 12
    for k in (2, 1, 0):  # three offset sheets suggest depth
        d.rect(x + 6 * k, cy - h / 2 - 6 * k, w, h, c, t, r=4, sw=1.6)
    d.text(x + w / 2 + 6, cy + h / 2 + 26, shape, 16, INK2, 600)
    return x + w + 12


def mnist_nets():
    d = SVG("mnist-nets", 1152, 420)
    pg, pd = NUM["gan"]["params_G"], NUM["gan"]["params_D"]
    rows = (
        (1, 110, "generator G", f"{pg / 1e6:.1f} million weights", "dec",
         [("vec", 100, 1, "z", "100"), ("op", "linear"), ("t", 7, 256, "256 × 7 × 7"), ("op", "up ×2"),
          ("t", 14, 128, "128 × 14 × 14"), ("op", "up ×2"), ("t", 28, 64, "64 × 28 × 28"), ("op", "conv, tanh", 104),
          ("img", "gan-fake-one.png", "1 × 28 × 28")]),
        (2, 320, "discriminator D", f"{round(pd, -3) / 1e3:.0f} thousand weights", "disc",
         [("img", "gan-real-one.png", "1 × 28 × 28"), ("op", "conv /2"), ("t", 14, 64, "64 × 14 × 14"), ("op", "conv /2"),
          ("t", 7, 128, "128 × 7 × 7"), ("op", "linear"), ("logit",)]),
    )
    for b, cy, title, sub, role, parts in rows:
        d.begin(b)
        d.text(20, cy - 70, title, 21, ROLE[role][0], 700, "start")
        d.text(20, cy - 46, sub, 15, INK3, 500, "start")
        x = 210
        for part in parts:
            kind = part[0]
            if kind == "op":
                w = part[2] if len(part) > 2 else 70
                d.line([(x, cy), (x + w, cy)], INK3)
                d.text(x + w / 2, cy - 12, part[1], 15, INK2, 600)
                x += w + 14
            elif kind == "vec":
                d.rect(x, cy - 50, 22, 100, C["z"], ROLE["z"][1], r=4, sw=1.8)
                d.math(x + 11, cy + 4, "z", 18, C["z"])
                d.text(x + 11, cy + 76, part[4], 16, INK2, 600)
                x += 34
            elif kind == "t":
                x = _slab(d, x, cy, part[1], part[2], role, part[3])
            elif kind == "img":
                d.image(x, cy - 50, 100, 100, part[1], "dec" if role == "dec" else "x")
                d.text(x + 50, cy + 76, part[2], 16, INK2, 600)
                x += 112
            elif kind == "logit":
                d.node(x, cy - 34, 210, 68, "disc", "1 logit", r"$D(x) = \mathrm{sigmoid}(\text{logit})$", size=21)
                x += 160
        d.end()
    d.save()


def main():
    for f in (gan, gan_dstep, gan_gstep, freeze_detach, cgan, bridge, mnist_nets):
        f()
    print("wrote", len(list(OUT.glob("*.svg"))), "diagrams")


if __name__ == "__main__":
    main()
