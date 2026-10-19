"""Hand-built SVG diagrams for Generative Modelling II.

    python3 scripts/diagrams.py      # writes figures/diagrams/*.svg

Plain SVG, built with the same helpers as Lecture I (scripts/svglib.py). Groups that appear
on later clicks carry data-b="n". Numbers are read from figures/data/numbers.json.
Roles: x0 data blue · noise amber · noisy x_t slate · denoiser teal · loss red · condition purple.
"""
import math

from svglib import SVG, NUM, C, INK, INK2, INK3, LINE, ROLE  # noqa: F401

toy = NUM["toy"]
S = NUM["sched"]


def g(n, key, default="??"):
    cur = NUM.get(n, {})
    for k in key.split("."):
        cur = cur.get(k, {}) if isinstance(cur, dict) else {}
    return cur if cur != {} else default


def op(d, x, y, s="+", r=22, color=INK2):
    d.add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="#fff" stroke="{color}" stroke-width="2.2"/>')
    d.text(x, y + 9, s, 28, color, 600)


# ============================================================ Part 1: corruption
def one_step():
    d = SVG("one-step", 1152, 360)
    y0, s = 40, 200
    d.begin(1)
    d.image(20, y0, s, s, "digit7.png", "x")
    d.math(120, y0 + s + 34, "x_{t-1}", 24, C["x"])
    d.text(120, y0 + s + 62, "current state", 17, INK3)
    d.line([(232, y0 + 100), (300, y0 + 100)], C["x"])
    d.node(306, y0 + 62, 150, 76, "x", "× 0.894", r"$\sqrt{1-\beta}$", size=24)
    d.line([(462, y0 + 100), (548, y0 + 100)], C["x"])
    d.end()
    d.begin(2)
    d.image(476, y0 + 158, 150, 150, "eta1.png", "z")
    d.label(551, y0 + 335, r"fresh noise $\eta$, × 0.447 = $\sqrt{\beta}$", 17, C["z"], 600)
    d.line([(572, y0 + 152), (572, y0 + 128)], C["z"])
    d.end()
    d.begin(3)
    op(d, 572, y0 + 100)
    d.line([(596, y0 + 100), (690, y0 + 100)], C["xt"])
    d.image(700, y0, s, s, "x1-toy.png", "xt")
    d.math(800, y0 + s + 34, "x_t", 24, C["xt"])
    d.text(800, y0 + s + 62, "a little less 7, a little more noise", 17, INK3)
    d.end()
    d.begin(4)
    d.rect(930, y0 + 10, 210, 178, "#E8C38A", "#FFF6E6", r=12, sw=1.6)
    d.text(1035, y0 + 42, "our pixel", 17, "#9A5700", 650)
    d.text(1035, y0 + 78, f"0.894 × (+1)", 20, INK)
    d.text(1035, y0 + 108, "+ 0.447 × (−0.5)", 20, INK)
    d.add(f'<line x1="960" y1="{y0 + 122}" x2="1110" y2="{y0 + 122}" stroke="{INK3}" stroke-width="1.5"/>')
    d.text(1035, y0 + 154, f"= {toy['x1']}", 24, C["xt"], 700)
    d.end()
    d.save()


def chain():
    d = SVG("chain", 1152, 340)
    xs = [40, 290, 540, 790, 1010]
    labels = ["x_0", "x_1", "x_2", "…", "x_T"]
    imgs = ["fwd-0.png", "fwd-100.png", "fwd-300.png", None, "fwd-1000.png"]
    s = 120
    for i, (x, lab, im) in enumerate(zip(xs, labels, imgs)):
        role = "x" if i == 0 else "xt"
        if im:
            d.image(x, 100, s, s, im, role)
        else:
            d.text(x + 60, 172, "…", 40, INK3)
        d.math(x + 60, 252, lab, 22, C[role]) if lab != "…" else None
    d.text(100, 290, "data", 16, INK3)
    d.label(1070, 290, r"≈ pure noise $\mathcal N(0, I)$", 16, INK3)
    d.begin(1)
    for i in range(4):
        x1, x2 = xs[i] + s + 10, xs[i + 1] - 10
        if i == 3:
            x1 = xs[3] + 70
        if i == 2:
            x2 = xs[3] + 30
        d.curve(f"M{x1},120 Q{(x1 + x2) / 2},70 {x2},120", C["z"], 2.6)
    d.label(560, 40, r"forward $q(x_t\mid x_{t-1})$: a fixed recipe, no parameters", 19, C["z"], 650)
    d.end()
    d.begin(2)
    for i in range(4):
        x1, x2 = xs[i + 1] - 10, xs[i] + s + 10
        if i == 3:
            x2 = xs[3] + 70
        if i == 2:
            x1 = xs[3] + 30
        d.curve(f"M{x1},205 Q{(x1 + x2) / 2},260 {x2},205", C["dec"], 2.6)
    d.label(560, 328, r"reverse $p_\theta(x_{t-1}\mid x_t)$: a network we must learn", 19, C["dec"], 650)
    d.end()
    d.save()


# ============================================================ Part 2: training
def tuple_fig():
    d = SVG("tuple", 1152, 430)
    s = 150
    ex = NUM.get("example", {})
    d.begin(1)
    d.image(10, 30, s, s, "digit7.png", "x")
    d.math(85, 210, "x_0", 22, C["x"])
    d.text(85, 236, "a training image", 16, INK3)
    d.end()
    d.begin(2)
    d.image(10, 262, s, s, "eps.png", "z", sw=2.5)
    d.math(250, 350, r"\epsilon\sim\mathcal N(0, I)", 20, C["z"])
    d.rect(190, 120, 130, 56, C["xt"], ROLE["xt"][1], r=10)
    d.text(255, 156, "t = 350", 22, C["xt"], 650)
    d.text(255, 204, "random step", 16, INK3)
    d.end()
    d.begin(3)
    d.line([(166, 105), (424, 190)], C["x"], 2)
    d.line([(322, 148), (424, 205)], C["xt"], 2)
    d.line([(166, 330), (424, 240)], C["z"], 2)
    d.rect(428, 160, 220, 110, C["xt"], "#fff", r=12)
    d.label(538, 198, f"{ex.get('sab', '??')} · $x_0$", 20, C["x"], 600)
    d.label(538, 228, f"+ {ex.get('s1ab', '??')} · $\\epsilon$", 20, C["z"], 600)
    d.label(538, 258, r"$\sqrt{\bar\alpha_{350}}$, $\sqrt{1-\bar\alpha_{350}}$", 15, INK3)
    d.line([(650, 215), (684, 215)], C["xt"])
    d.image(690, 140, s, s, "ex-xt.png", "xt")
    d.math(765, 322, "x_t", 22, C["xt"])
    d.text(765, 348, "network input", 16, INK3)
    d.end()
    d.begin(4)
    d.line([(846, 215), (868, 215)], C["xt"])
    d.trapezoid(872, 215, 112, 120, 70, "dec", "network", None)
    d.line([(986, 215), (998, 215)], C["dec"])
    d.image(1000, 140, s, s, "ex-ehat.png", "dec")
    d.math(1075, 322, r"\hat\epsilon", 24, C["dec"])
    d.text(1075, 348, "its prediction", 16, INK3)
    d.line([(1075, 360), (1075, 410), (170, 410), (170, 390)], C["loss"], 2.2, dash="5 4")
    d.label(640, 400, f"compare $\\hat\\epsilon$ with $\\epsilon$: mean squared error over 784 pixels = {ex.get('mse', '??')}",
            18, C["loss"], 650)
    d.end()
    d.save()


def gather():
    d = SVG("gather", 1152, 300)
    ts = [7, 350, 920]
    abs_ = [S["7"]["ab"], S["350"]["ab"], S["920"]["ab"]]
    d.begin(1)
    d.text(80, 30, "t   [B]", 18, INK2, 650)
    d.vec(40, 50, [str(t) for t in ts], "xt", cw=80, ch=56, size=20)
    d.end()
    d.begin(2)
    d.line([(128, 134), (226, 134)], INK3)
    d.text(177, 120, "look up", 16, INK3)
    d.label(295, 30, r"$\bar\alpha[t]$  [B]", 18, INK2, 650)
    d.vec(240, 50, [f"{abs_[0]:.3f}", f"{abs_[1]:.3f}", f"{abs_[2]:.4f}"], "x", cw=110, ch=56, size=19)
    d.end()
    d.begin(3)
    d.line([(358, 134), (450, 134)], INK3)
    d.text(404, 120, "reshape", 16, INK3)
    d.text(530, 30, "[B, 1, 1, 1]", 18, INK2, 650)
    for i in range(3):
        d.add(f'<rect x="{506}" y="{56 + i * 56}" width="48" height="44" rx="6" fill="{ROLE["x"][1]}" stroke="{C["x"]}" '
              f'stroke-width="2"/>')
        d.add(f'<path d="M506,{56 + i * 56} l10,-8 h48 l-10,8 M554,{56 + i * 56} l10,-8 v44 l-10,8" fill="none" '
              f'stroke="{C["x"]}" stroke-width="1.4"/>')
    d.end()
    d.begin(4)
    d.line([(578, 134), (686, 134)], INK3)
    d.text(632, 120, "× broadcast", 16, INK3)
    for i, t in enumerate(ts):
        d.image(700 + i * 150, 50, 130, 130, f"batch-{t}.png", "xt")
        d.text(765 + i * 150, 206, f"t = {t}", 18, C["xt"], 650)
    d.label(915, 30, r"$x_t$  [B, 1, 28, 28]", 18, INK2, 650)
    d.text(915, 250, "each image gets its own noise level", 17, INK3)
    d.end()
    d.save()


def grads():
    d = SVG("grads", 1152, 380)
    # forward graph
    d.begin(1)
    d.node(20, 40, 120, 60, "x", "$x_0$", "data")
    d.node(20, 150, 120, 60, "z", r"$\epsilon$", "random draw")
    d.node(20, 260, 120, 60, "xt", "t", "random step")
    d.line([(142, 70), (250, 140)], C["x"], 2)
    d.line([(142, 180), (250, 170)], C["z"], 2)
    d.line([(142, 290), (250, 200)], C["xt"], 2)
    d.node(256, 130, 170, 84, "xt", "$x_t$", "no parameters")
    d.line([(428, 172), (520, 172)], C["xt"])
    d.trapezoid(526, 172, 170, 150, 90, "dec", "network", "trainable weights")
    d.line([(698, 172), (780, 172)], C["dec"])
    d.node(786, 140, 110, 64, "dec", r"$\hat\epsilon$", None)
    d.line([(142, 165), (230, 110), (930, 110), (960, 150)], C["z"], 1.8, dash="5 4")
    d.label(560, 100, r"the same $\epsilon$ is also the label", 15, C["z"], 600)
    d.end()
    d.begin(2)
    d.line([(898, 172), (960, 172)], C["dec"])
    d.node(966, 136, 150, 72, "loss", "MSE", "a single number")
    d.end()
    d.begin(3)
    d.curve("M1040,212 C1040,320 620,330 610,252", C["loss"], 3.2)
    d.label(830, 330, r"$\partial\,\mathrm{loss}/\partial\theta$: the only gradient we need", 18, C["loss"], 650)
    d.cross(340, 250, 12)
    d.text(340, 290, "nothing to train here", 15, INK3)
    d.end()
    d.save()


def elbo_chain():
    d = SVG("elbo-chain", 1152, 300)
    xs = [60, 300, 540, 780, 1000]
    labels = ["x_0", "x_1", "x_{t-1}", "x_t", "x_T"]
    for x, lab in zip(xs, labels):
        d.add(f'<circle cx="{x + 40}" cy="150" r="38" fill="{ROLE["xt"][1]}" stroke="{C["xt"]}" stroke-width="2.4"/>')
        d.math(x + 40, 160, lab, 24, C["xt"])
    for a, b in ((0, 1), (2, 3)):
        d.curve(f"M{xs[b]},140 Q{(xs[a] + xs[b]) / 2 + 40},100 {xs[a] + 84},140", C["dec"], 2.4)
    d.text(470, 156, "…", 30, INK3)
    d.text(940, 156, "…", 30, INK3)
    d.begin(1)
    d.rect(930, 210, 210, 72, C["z"], ROLE["z"][1], r=10)
    d.math(1035, 244, "L_T", 22, C["z"])
    d.label(1035, 270, r"end matches $\mathcal N(0, I)$", 14, INK2, width=200)
    d.end()
    d.begin(2)
    d.rect(500, 210, 400, 72, C["dec"], ROLE["dec"][1], r=10)
    d.math(700, 244, r"L_{t-1}\ \text{ for } t = 2,\dots,T", 22, C["dec"])
    d.text(700, 270, "each reverse step matches the true one", 14, INK2)
    d.end()
    d.begin(3)
    d.rect(40, 210, 300, 72, C["x"], ROLE["x"][1], r=10)
    d.math(190, 244, "L_0", 22, C["x"])
    d.text(190, 270, "last step reproduces the pixels", 14, INK2)
    d.end()
    d.save()


# ============================================================ Part 3: generation
def reverse_line():
    """Number line for the toy reverse step: input, mean, sample, clean guess."""
    d = SVG("reverse-line", 1152, 400)
    lo, hi, X0, X1 = 0.40, 1.10, 50, 1100
    X = lambda v: X0 + (v - lo) / (hi - lo) * (X1 - X0)
    y = 290
    d.add(f'<line x1="{X0}" y1="{y}" x2="{X1}" y2="{y}" stroke="#9AA1AD" stroke-width="2"/>')
    for v in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1):
        d.add(f'<line x1="{X(v)}" y1="{y - 6}" x2="{X(v)}" y2="{y + 6}" stroke="#9AA1AD" stroke-width="1.5"/>')
        d.text(X(v), y + 30, f"{v:.1f}", 16, INK3)
    sig = toy["sigma"]
    pts = [(X(lo + (hi - lo) * k / 200), y - 130 * math.exp(-((lo + (hi - lo) * k / 200) - toy["mu"]) ** 2 / (2 * sig ** 2)))
           for k in range(201)]
    d.begin(2)
    d.add('<path d="M' + " L".join(f"{a:.1f},{b:.1f}" for a, b in pts) + f' L{X1},{y} L{X0},{y} z" fill="{ROLE["dec"][1]}" '
          f'fill-opacity=".6" stroke="none"/>')
    d.add('<path d="M' + " L".join(f"{a:.1f},{b:.1f}" for a, b in pts) + f'" fill="none" stroke="{C["dec"]}" stroke-width="2.2"/>')
    d.label(X1, 26, r"shaded: the reverse step $\mathcal N(\mu_\theta,\ \sigma^2)$, $\sigma = 0.333$", 16, C["dec"], 600, "end")
    d.end()

    def mark(v, lab, sub, col, up, b, h=0):
        d.begin(b)
        top = y - 222 - h if up else y + 58
        d.add(f'<line x1="{X(v)}" y1="{y}" x2="{X(v)}" y2="{top + (22 if up else -10)}" stroke="{col}" stroke-width="2.4"/>')
        d.add(f'<circle cx="{X(v)}" cy="{y}" r="8" fill="{col}" stroke="#fff" stroke-width="2"/>')
        ty = top if up else top + 22
        d.label(X(v), ty, lab, 20, col, 700, width=300)
        d.text(X(v), ty + 22 if up else ty + 22, sub, 15, INK2)
        d.end()

    mark(toy["x2"], "$x_2 = 0.5$", "noisy input", C["xt"], True, 1)
    mark(toy["mu"], f"$\\mu_\\theta = {toy['mu']:.3f}$", "mean", C["dec"], False, 2)
    mark(toy["x1s"], f"$x_1 = {toy['x1s']:.3f}$", "sampled next state", C["z"], True, 3)
    mark(toy["x0hat"], f"$\\hat x_0 = {toy['x0hat']:.3f}$", "guess of the clean value", C["x"], False, 4)
    d.begin(4)
    d.add(f'<line x1="{X(1.0)}" y1="{y - 16}" x2="{X(1.0)}" y2="{y + 16}" stroke="{INK3}" stroke-width="2.4"/>')
    d.text(X(1.0) + 10, y - 22, "true value 1", 15, INK3, 600, "start")
    d.end()
    d.save()


def loop():
    d = SVG("loop", 1152, 300)
    d.image(20, 60, 150, 150, "traj-x-1000.png", "xt")
    d.math(95, 240, r"x_T\sim\mathcal N(0, I)", 20, C["xt"])
    d.line([(176, 135), (240, 135)], C["xt"])
    d.trapezoid(246, 135, 150, 140, 84, "dec", "network", "same weights")
    d.line([(398, 135), (450, 135)], C["dec"])
    d.node(456, 100, 130, 70, "dec", r"$\hat\epsilon$", None)
    d.line([(588, 135), (640, 135)], C["dec"])
    d.node(646, 96, 170, 78, "dec", r"mean $\mu_\theta$", "remove a little noise")
    d.line([(818, 135), (850, 135)], C["dec"])
    op(d, 874, 135)
    d.line([(874, 230), (874, 160)], C["z"])
    d.label(874, 254, r"$+\ \sigma_t z$, fresh $z$", 16, C["z"], 600)
    d.line([(898, 135), (960, 135)], C["xt"])
    d.node(966, 100, 160, 70, "xt", "$x_{t-1}$", None)
    d.line([(1046, 98), (1046, 30), (320, 30), (320, 66)], C["xt"], 2.2)
    d.text(680, 22, "t ← t − 1, repeat until t = 0 (no fresh noise at the last step)", 17, C["xt"], 600)
    d.save()


# ============================================================ Part 4: architecture
def unet():
    d = SVG("unet", 1152, 480)
    lv = {
        "in": (24, 50, 22, 230, "input", "1×28×28", None),
        "e1": (130, 50, 48, 230, "enc 1", "32×28×28", "e1"),
        "e2": (300, 160, 90, 115, "enc 2", "64×14×14", "e2"),
        "m": (500, 255, 150, 58, "middle", "128×7×7", "m"),
        "d1": (760, 160, 90, 115, "dec 1", "64×14×14", "d1"),
        "d2": (960, 50, 48, 230, "dec 2", "32×28×28", "d2"),
        "out": (1100, 50, 22, 230, "output", "1×28×28", None),
    }

    def block(k, role, b):
        x, y, w, h, lab, shp, fm = lv[k]
        d.begin(b)
        c, t = ROLE[role]
        d.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{t}" stroke="{c}" stroke-width="2.2"/>')
        d.text(x + w / 2, y + h + 24, lab, 16, c, 650)
        d.text(x + w / 2, y + h + 46, shp, 15, INK2)
        d.end()

    block("in", "xt", 1); block("e1", "dec", 1); block("e2", "dec", 1)
    d.begin(1)
    d.line([(48, 165), (126, 165)], INK3, 2)
    d.line([(180, 230), (296, 230)], INK3, 2)
    d.text(238, 220, "halve", 15, INK3)
    d.end()
    block("m", "dec", 2)
    d.begin(2)
    d.line([(392, 300), (496, 300)], INK3, 2)
    d.text(444, 290, "halve", 15, INK3)
    d.end()
    block("d1", "dec", 3); block("d2", "dec", 3); block("out", "dec", 3)
    d.begin(3)
    d.line([(652, 300), (756, 300)], INK3, 2)
    d.text(704, 290, "double", 15, INK3)
    d.line([(852, 230), (956, 230)], INK3, 2)
    d.text(904, 220, "double", 15, INK3)
    d.line([(1010, 165), (1096, 165)], INK3, 2)
    d.text(1053, 155, "1×1", 15, INK3)
    d.end()
    d.begin(4)
    d.line([(180, 70), (956, 70)], C["z"], 2.4)
    d.text(568, 58, "skip: concatenate, 32 + 32 = 64 channels at 28 × 28", 17, C["z"], 650)
    d.line([(392, 178), (756, 178)], C["z"], 2.4)
    d.text(574, 166, "skip: 64 + 64 = 128 at 14 × 14", 17, C["z"], 650)
    d.end()
    d.begin(5)
    for k, x in (("e1", 104), ("e2", 296), ("m", 526), ("d1", 756), ("d2", 934)):
        for j in range(2):
            d.image(x + j * 52, 382, 48, 48, f"fmap-{k}-{j}.png", "dec", sw=1.4)
    d.text(576, 470, "two channels per level for our noisy 7 at t = 350: the trained network's activations", 15, INK3)
    d.end()
    d.save()


def time_inject():
    d = SVG("time-inject", 1152, 300)
    d.begin(1)
    d.node(10, 40, 120, 64, "xt", "t = 350", None)
    d.line([(132, 72), (176, 72)], C["xt"])
    d.node(182, 36, 190, 72, "xt", "sin / cos", "64 features")
    d.end()
    d.begin(2)
    d.line([(374, 72), (420, 72)], C["xt"])
    d.trapezoid(426, 72, 120, 80, 80, "dec", "MLP", "shared")
    d.line([(548, 72), (596, 72)], C["dec"])
    d.node(602, 36, 210, 72, "dec", "Linear", "64 → C, per block")
    d.line([(814, 72), (860, 72)], C["dec"])
    d.node(866, 40, 230, 64, "dec", "[B, C, 1, 1]", None, size=20)
    d.end()
    d.begin(3)
    d.line([(981, 106), (981, 176)], C["dec"])
    d.text(966, 148, "broadcast over H × W", 16, INK3, anchor="end")
    d.rect(820, 182, 320, 96, C["xt"], ROLE["xt"][1], r=10)
    d.text(980, 222, "+ feature map [B, C, H, W]", 18, C["xt"], 650)
    d.text(980, 254, "in every block", 15, INK3)
    d.end()
    d.save()


def dit():
    d = SVG("dit", 1152, 360)
    x0, y0, cs = 20, 60, 22
    for i in range(8):
        for j in range(8):
            d.add(f'<rect x="{x0 + j * cs}" y="{y0 + i * cs}" width="{cs - 2}" height="{cs - 2}" rx="3" '
                  f'fill="{ROLE["xt"][1]}" stroke="#C7CCD4" stroke-width="1"/>')
    d.text(108, 40, "noisy latent 4 × 32 × 32", 16, C["xt"], 650)
    d.begin(1)
    for i in range(4):
        for j in range(4):
            d.add(f'<rect x="{x0 + j * 2 * cs - 1}" y="{y0 + i * 2 * cs - 1}" width="{2 * cs}" height="{2 * cs}" rx="4" '
                  f'fill="none" stroke="{C["xt"]}" stroke-width="2"/>')
    d.text(108, 262, "cut into 2 × 2 patches", 16, INK3)
    d.line([(205, 150), (256, 150)], C["xt"])
    for k in range(6):
        d.rect(264 + k * 44, 120, 36, 60, C["xt"], ROLE["xt"][1], r=6, sw=1.8)
    d.text(394, 104, "256 tokens × 16 numbers", 16, C["xt"], 650)
    d.text(394, 206, "…", 26, INK3)
    d.end()
    d.begin(2)
    d.line([(536, 150), (586, 150)], C["xt"])
    d.rect(592, 80, 270, 140, C["dec"], ROLE["dec"][1], r=12)
    d.text(727, 130, "Transformer blocks", 20, C["dec"], 650)
    d.text(727, 158, "full self-attention", 16, INK2)
    d.text(727, 182, "over all 256 tokens", 16, INK2)
    d.end()
    d.begin(3)
    d.rect(552, 262, 350, 74, C["enc"], ROLE["enc"][1], r=12)
    d.text(727, 292, "step t and class label", 19, C["enc"], 650)
    d.text(727, 318, "set a scale and shift in every block", 15, INK2)
    d.line([(727, 260), (727, 224)], C["enc"])
    d.end()
    d.begin(4)
    d.line([(864, 150), (914, 150)], C["dec"])
    for i in range(4):
        for j in range(4):
            d.add(f'<rect x="{924 + j * 44}" y="{y0 + i * 44}" width="40" height="40" rx="4" fill="{ROLE["dec"][1]}" '
                  f'stroke="{C["dec"]}" stroke-width="1.6"/>')
    d.text(1012, 40, "back to 4 × 32 × 32", 16, C["dec"], 650)
    d.text(1012, 262, "a prediction per patch", 16, INK3)
    d.end()
    d.save()


def two_stage():
    d = SVG("two-stage", 1152, 420)
    d.text(20, 30, "Stage 1 · train an autoencoder (reconstruction loss)", 19, INK, 650, "start")
    d.image(20, 50, 120, 120, "digit7.png", "x")
    d.line([(146, 110), (190, 110)], C["x"])
    d.trapezoid(196, 110, 120, 110, 60, "enc", "encoder", None)
    d.line([(318, 110), (350, 110)], C["enc"])
    for j in range(4):
        d.image(356 + j * 64, 80, 58, 58, f"latent-{j}.png", "z", sw=1.5)
    d.text(482, 162, "z: 4 × 7 × 7 = 196 numbers", 15, C["z"], 650)
    d.line([(616, 110), (650, 110)], C["z"])
    d.trapezoid(656, 110, 120, 60, 110, "dec", "decoder", None)
    d.line([(778, 110), (810, 110)], C["dec"])
    d.image(816, 50, 120, 120, "ae-recon.png", "dec")
    d.text(876, 190, "reconstruction", 15, INK3)
    d.begin(2)
    d.add(f'<line x1="20" y1="222" x2="1130" y2="222" stroke="{LINE}" stroke-width="1.5"/>')
    d.text(20, 258, "Stage 2 · freeze the autoencoder, train a denoiser on z", 19, INK, 650, "start")
    d.image(20, 280, 110, 110, "digit7.png", "x")
    d.line([(136, 335), (180, 335)], C["x"])
    d.trapezoid(186, 335, 110, 100, 56, "grey", "encoder", None)
    d.lock(241, 380)
    d.line([(298, 335), (340, 335)], C["z"])
    d.node(346, 300, 150, 70, "z", "$z / s$", f"s = {g('ae', 'latent_std')}")
    d.line([(498, 335), (540, 335)], C["z"])
    d.rect(546, 290, 330, 90, C["dec"], ROLE["dec"][1], r=12)
    d.text(711, 326, "the same DDPM training step", 18, C["dec"], 650)
    d.text(711, 352, "on 4 × 7 × 7 latents instead of pixels", 15, INK2)
    d.text(1010, 336, "the decoder is", 15, INK3)
    d.text(1010, 356, "not used in stage 2", 15, INK3)
    d.end()
    d.save()


def latent_gen():
    d = SVG("latent-gen", 1152, 250)
    for j in range(4):
        d.image(20 + (j % 2) * 66, 30 + (j // 2) * 66, 60, 60, f"latent-zT-{j}.png", "z", sw=1.5)
    d.math(86, 192, r"z_T\sim\mathcal N(0, I)", 18, C["z"])
    d.text(86, 218, "4 × 7 × 7", 15, INK3)
    d.line([(160, 96), (210, 96)], C["z"])
    d.rect(216, 46, 270, 100, C["dec"], ROLE["dec"][1], r=12)
    d.text(351, 86, "latent denoiser", 19, C["dec"], 650)
    d.text(351, 114, "1,000 reverse steps", 16, INK2)
    d.line([(488, 96), (540, 96)], C["dec"])
    for j in range(4):
        d.image(546 + (j % 2) * 66, 30 + (j // 2) * 66, 60, 60, f"latent-z0-{j}.png", "z", sw=1.5)
    d.math(612, 192, "z_0", 18, C["z"])
    d.line([(686, 96), (736, 96)], C["z"])
    d.trapezoid(742, 96, 120, 60, 110, "dec", "decoder", "once")
    d.line([(864, 96), (910, 96)], C["dec"])
    d.image(916, 20, 150, 150, "latent-one.png", "dec")
    d.text(991, 196, "pixels 1 × 28 × 28", 15, INK3)
    d.save()

# ============================================================ Part 5: conditioning
def prompt():
    d = SVG("prompt", 1152, 330)
    toks = [("<start>", 49406), ("a", 320), ("basket", 10338), ("full", 1476), ("of", 539), ("cherries", 27248),
            ("<end>", 49407)]
    x = 20
    d.begin(1)
    for w, i in toks:
        wd = max(80, 14 * len(w) + 24)
        d.rect(x, 40, wd, 50, C["enc"], ROLE["enc"][1], r=8, sw=1.8)
        d.text(x + wd / 2, 72, w, 18, C["enc"], 650)
        x += wd + 8
    d.rect(x, 40, 160, 50, "#C7CCD4", "#F3F4F6", r=8, sw=1.5, dash="5 4")
    d.text(x + 80, 72, "+ 70 padding", 16, INK3)
    d.end()
    d.begin(2)
    x = 20
    for w, i in toks:
        wd = max(80, 14 * len(w) + 24)
        d.text(x + wd / 2, 118, str(i), 16, INK2)
        x += wd + 8
    d.text(x + 80, 118, "= 77 IDs", 16, INK2)
    d.end()
    d.begin(3)
    d.line([(560, 132), (560, 162)], C["enc"])
    d.rect(330, 166, 460, 74, C["enc"], ROLE["enc"][1], r=12)
    d.text(560, 198, "text encoder (frozen)", 20, C["enc"], 650)
    d.text(560, 224, "Stable Diffusion v1: CLIP ViT-L/14 text tower", 15, INK2)
    d.line([(792, 203), (840, 203)], C["enc"])
    d.text(960, 198, "C: [1, 77, 768]", 20, C["enc"], 650)
    d.text(960, 224, "one vector per token", 16, INK3)
    d.end()
    d.begin(4)
    for k, t in enumerate(("t = 1000", "t = 999", "…", "t = 1")):
        x0 = 330 + k * 120
        if t == "…":
            d.text(x0 + 50, 300, "…", 26, INK3)
            continue
        d.rect(x0, 274, 100, 44, C["dec"], ROLE["dec"][1], r=8, sw=1.8)
        d.text(x0 + 50, 302, t, 15, C["dec"], 650)
    d.line([(960, 236), (960, 258), (380, 258), (380, 270)], C["enc"], 1.6, dash="4 3")
    d.text(830, 302, "the same C at every step", 16, INK3, anchor="start")
    d.end()
    d.save()


def cross_attn():
    d = SVG("cross-attn", 1152, 440)
    words = ["a", "basket", "full", "of", "cherries"]
    vals = [0.06, 0.31, 0.08, 0.05, 0.50]
    d.begin(1)
    for i in range(4):
        for j in range(4):
            fill = "#FFF4C2" if (i, j) == (1, 2) else ROLE["xt"][1]
            sw = 2.6 if (i, j) == (1, 2) else 1.4
            d.add(f'<rect x="{30 + j * 34}" y="{70 + i * 34}" width="30" height="30" rx="4" fill="{fill}" '
                  f'stroke="{C["xt"]}" stroke-width="{sw}"/>')
    d.text(97, 50, "image features H", 17, C["xt"], 650)
    d.text(97, 228, "N = H × W positions", 15, INK3)
    d.line([(172, 120), (224, 120)], C["xt"])
    d.node(230, 84, 180, 72, "xt", "$Q = HW_Q$", "a query per position")
    d.end()
    d.begin(2)
    for k, w in enumerate(words):
        d.rect(30 + k * 94, 300, 86, 40, C["enc"], ROLE["enc"][1], r=8, sw=1.6)
        d.text(73 + k * 94, 326, w, 16, C["enc"], 650)
    d.text(250, 370, "text features C (only the 5 words drawn)", 15, C["enc"], 650)
    d.line([(504, 320), (544, 320)], C["enc"])
    d.node(550, 276, 190, 60, "enc", "$K = CW_K$", "keys")
    d.node(550, 350, 190, 60, "enc", "$V = CW_V$", "values")
    d.end()
    d.begin(3)
    d.line([(412, 120), (790, 120)], C["xt"], 2)
    d.line([(742, 306), (790, 306), (790, 170)], C["enc"], 2)
    d.text(950, 40, "one row of A: softmax over the words", 16, INK2, 650)
    for k, (w, v) in enumerate(zip(words, vals)):
        y = 56 + k * 34
        d.text(866, y + 22, w, 15, C["enc"], 600, "end")
        d.add(f'<rect x="876" y="{y + 6}" width="{v * 440}" height="22" rx="4" fill="{C["enc"]}" fill-opacity="{0.25 + v}"/>')
        d.text(884 + v * 440, y + 23, f"{v:.2f}", 14, INK2, 500, "start")
    d.text(980, 240, "illustrative weights, sum to 1", 14, INK3)
    d.end()
    d.begin(4)
    d.line([(742, 380), (876, 380)], C["enc"], 2)
    d.node(882, 340, 240, 72, "dec", "$AV$", "a text summary per position")
    d.text(1002, 432, "added back to that position's features", 15, INK3)
    d.end()
    d.save()


def cfg_train():
    d = SVG("cfg-train", 1152, 270)
    d.image(20, 30, 130, 130, "ex-xt.png", "xt")
    d.math(85, 186, "x_t,\\ t", 20, C["xt"])
    d.node(210, 40, 170, 64, "enc", "label c = 7", None)
    d.begin(2)
    d.node(210, 150, 170, 64, "grey", r"$c=\varnothing$", "label hidden", dash="5 4")
    d.text(295, 244, "10% of the time", 16, INK2, 650)
    d.end()
    d.line([(154, 95), (456, 130)], C["xt"], 2)
    d.line([(382, 72), (456, 116)], C["enc"], 2)
    d.trapezoid(462, 130, 170, 140, 90, "dec", "one network", None)
    d.text(547, 226, "learns labels 0–9 and “none”", 15, INK3)
    d.line([(634, 130), (680, 130)], C["dec"])
    d.node(686, 98, 120, 64, "dec", r"$\hat\epsilon$", None)
    d.line([(808, 130), (850, 130)], C["dec"])
    d.node(856, 92, 270, 76, "loss", r"MSE with $\epsilon$", "same target either way")
    d.save()


def cfg_vectors():
    d = SVG("cfg-vectors", 520, 420)
    ox, oy, sc = 70, 262, 240   # equal scale on both axes
    P = lambda v: (ox + v[0] * sc, oy - v[1] * sc)
    d.add(f'<line x1="30" y1="{oy}" x2="510" y2="{oy}" stroke="#C7CCD4" stroke-width="1.5"/>')
    d.add(f'<line x1="{ox}" y1="400" x2="{ox}" y2="40" stroke="#C7CCD4" stroke-width="1.5"/>')
    eu, ec = (0.2, -0.3), (0.6, 0.1)
    s3 = (eu[0] + 3 * (ec[0] - eu[0]), eu[1] + 3 * (ec[1] - eu[1]))
    d.begin(1)
    d.line([P((0, 0)), P(eu)], C["xt"], 3)
    d.label(P(eu)[0] + 14, P(eu)[1] + 34, r"$\epsilon_u=(0.2,-0.3)$, s = 0", 16, C["xt"], 650, "start", width=300)
    d.end()
    d.begin(2)
    d.line([P((0, 0)), P(ec)], C["enc"], 3)
    d.label(P(ec)[0] + 14, P(ec)[1] + 30, r"$\epsilon_c=(0.6, 0.1)$, s = 1", 16, C["enc"], 650, "start", width=300)
    d.line([P(eu), P(ec)], C["loss"], 2.4)
    d.end()
    d.begin(3)
    d.line([P(ec), P(s3)], C["loss"], 2.4, dash="6 4")
    d.add(f'<circle cx="{P(s3)[0]}" cy="{P(s3)[1]}" r="8" fill="{C["dec"]}"/>')
    d.label(P(s3)[0] - 14, P(s3)[1] - 12, "s = 3: (1.4, 0.9)", 16, C["dec"], 700, "end", width=300)
    d.label((P(ec)[0] + P(s3)[0]) / 2 + 12, (P(ec)[1] + P(s3)[1]) / 2 + 20, r"$3(\epsilon_c-\epsilon_u)$ from $\epsilon_u$", 15,
            C["loss"], 600, "start", width=300)
    d.end()
    d.text(260, 416, "illustrative 2-D vectors, equal axis scales", 14, INK3)
    d.save()


def repaint():
    d = SVG("repaint", 1152, 250)
    d.image(20, 30, 150, 150, "inpaint-input.png", "x")
    d.text(95, 206, "known: top half", 16, C["x"], 650)
    d.text(95, 230, "unknown: shaded", 16, INK3)
    d.begin(1)
    d.line([(176, 105), (240, 75)], C["x"])
    d.line([(176, 105), (240, 160)], C["dec"])
    d.node(246, 40, 300, 66, "x", "known pixels", "the clean pixels noised to step t − 1")
    d.node(246, 128, 300, 66, "dec", "unknown pixels", "one reverse step of our model")
    d.end()
    d.begin(2)
    d.line([(548, 73), (620, 105)], C["x"], 2)
    d.line([(548, 161), (620, 125)], C["dec"], 2)
    d.node(626, 78, 220, 72, "xt", "paste together", "$x_{t-1}$")
    d.line([(736, 76), (736, 20), (396, 20), (396, 36)], C["xt"], 2)
    d.text(566, 14, "", 1)
    d.text(1000, 110, "repeat for", 16, C["xt"], 600)
    d.text(1000, 134, "t = 1000 … 1", 16, C["xt"], 600)
    d.end()
    d.save()


def rotation():
    d = SVG("rotation", 520, 420)
    ox, oy, R = 220, 330, 230
    a = math.radians(40)
    d.add(f'<line x1="{ox}" y1="{oy}" x2="{ox + 270}" y2="{oy}" stroke="{C["x"]}" stroke-width="2.4"/>')
    d.add(f'<line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy - 270}" stroke="{C["z"]}" stroke-width="2.4"/>')
    d.math(ox + 280, oy + 6, "x_0", 22, C["x"], "start")
    d.math(ox, oy - 286, r"\epsilon", 22, C["z"])
    x, y = ox + R * math.cos(a), oy - R * math.sin(a)
    d.line([(ox, oy), (x, y)], C["xt"], 3)
    d.math(x + 10, y - 4, "x_t", 22, C["xt"], "start")
    d.begin(1)
    pts = " ".join(f"{ox + 70 * math.cos(t):.1f},{oy - 70 * math.sin(t):.1f}" for t in [a * i / 20 for i in range(21)])
    d.add(f'<polyline points="{pts}" fill="none" stroke="{INK3}" stroke-width="1.6"/>')
    d.math(ox + 84, oy - 20, r"\phi", 20, INK2, "start")
    d.end()
    d.begin(2)
    vx, vy = ox + R * math.cos(a + math.pi / 2), oy - R * math.sin(a + math.pi / 2)
    d.line([(ox, oy), (vx, vy)], C["enc"], 3)
    d.math(vx - 10, vy, "v", 22, C["enc"], "end")
    d.end()
    d.label(260, 405, r"$\cos\phi=\sqrt{\bar\alpha_t},\ \ \sin\phi=\sqrt{1-\bar\alpha_t}$", 17, INK2)
    d.save()


def heads():
    d = SVG("heads", 1152, 260)
    d.node(20, 90, 170, 72, "xt", "H: [B, N, d]", "N = H·W")
    d.line([(192, 126), (240, 126)], C["xt"])
    for k in range(3):
        d.rect(250 + k * 14, 60 + k * 14, 200, 110, C["enc"], ROLE["enc"][1], r=10, sw=1.8)
    d.text(378, 128, "h heads", 19, C["enc"], 650)
    d.text(378, 152, "A: [B, h, N, L]", 16, INK2)
    d.line([(484, 126), (540, 126)], C["enc"])
    d.node(546, 90, 240, 72, "enc", "concat heads", "[B, N, h·d_v]")
    d.line([(788, 126), (840, 126)], C["enc"])
    d.node(846, 90, 130, 72, "dec", "$W_O$", "project")
    d.line([(978, 126), (1010, 126)], C["dec"])
    d.node(1014, 90, 130, 72, "xt", "+ H", "back to H × W")
    d.save()


def vae_vs_diff():
    d = SVG("vae-vs-diff", 1152, 420)
    # top: Lecture I VAE
    d.text(20, 26, "Lecture I · VAE", 19, INK, 650, "start")
    d.image(40, 52, 110, 110, "digit7.png", "x")
    d.math(95, 186, "x", 22, C["x"])
    d.curve("M156,86 Q300,46 440,86", C["enc"], 2.6)
    d.label(220, 56, r"encoder $q_\phi(z\mid x)$, learned", 16, C["enc"], 650, "start", width=300)
    d.curve("M440,140 Q300,180 156,140", C["dec"], 2.6)
    d.label(298, 196, r"decoder $p_\theta(x\mid z)$", 16, C["dec"], 650)
    d.vec(446, 82, ["0.33", "−1.99"], "z", cw=86, ch=34, vertical=False)
    d.math(532, 70, "z", 22, C["z"])
    d.label(532, 150, r"prior $\mathcal N(0, I)$", 16, C["z"], 600)
    d.begin(2)
    d.add(f'<line x1="20" y1="222" x2="1132" y2="222" stroke="{LINE}" stroke-width="1.5"/>')
    d.text(20, 254, "Diffusion", 19, INK, 650, "start")
    xs = [40, 300, 560, 900]
    ims = ["digit7.png", "fwd-100.png", "fwd-300.png", "fwd-1000.png"]
    labs = ["x_0", "x_1", "x_2", "x_T"]
    for x, im, lab in zip(xs, ims, labs):
        d.image(x, 274, 110, 110, im, "x" if lab == "x_0" else "xt")
        d.math(x + 55, 410, lab, 20, C["x"] if lab == "x_0" else C["xt"])
    for a, b in ((0, 1), (1, 2)):
        d.curve(f"M{xs[a] + 116},300 Q{(xs[a] + xs[b]) / 2 + 58},266 {xs[b] - 6},300", C["z"], 2.4)
        d.curve(f"M{xs[b] - 6},360 Q{(xs[a] + xs[b]) / 2 + 58},394 {xs[a] + 116},360", C["dec"], 2.4)
    d.text(780, 336, "…", 34, INK3)
    d.label(430, 264, r"fixed noising $q$: no encoder to learn", 16, C["z"], 650)
    d.label(1018, 334, r"prior $\mathcal N(0, I)$", 15, C["z"], 600, "start", width=140)
    d.end()
    d.begin(3)
    d.rect(680, 34, 452, 160, "#E8C38A", "#FFF6E6", r=12, sw=1.6)
    d.text(906, 66, "the same three ingredients", 17, "#9A5700", 650)
    d.label(906, 102, r"latent: $z$  ↔  $x_1, \dots, x_T$", 17, INK, width=440)
    d.label(906, 136, r"decoder: one pass  ↔  $p_\theta(x_{t-1}\mid x_t)$, T passes", 16, INK, width=440)
    d.label(906, 170, r"encoder: learned  ↔  fixed", 17, INK, width=440)
    d.end()
    d.save()


if __name__ == "__main__":
    for f in (one_step, chain, tuple_fig, gather, grads, elbo_chain, reverse_line, loop, unet, time_inject, dit,
              two_stage, latent_gen, prompt, cross_attn, cfg_train, cfg_vectors, repaint, rotation, heads, vae_vs_diff):
        f()
    print("diagrams written")
