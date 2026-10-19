"""Assemble gans.html from deck/slides/*.html.

    python scripts/build_deck.py

* <!--svg:NAME-->      is replaced by figures/diagrams/NAME.svg (inlined so its groups can build)
* <!--route:N-->       is replaced by the five-part route with part N highlighted
* {{a.b.0}}            is replaced by a value from figures/data/numbers.json (or a derived value below)
Every number on a slide therefore comes from the same run as the figures.
"""
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NUM = json.loads((ROOT / "figures" / "data" / "numbers.json").read_text())
OUT = "gans.html"
TITLE = "Generative Modelling I-B · GANs"


def fmt(v):
    """numbers.json is already rounded for display; only swap in a typographic minus."""
    return str(v).replace("-", "−") if isinstance(v, (int, float)) else str(v)


def derived():
    d = {}
    bad = NUM["toygan"]["bad"]
    d["toygan_bad_left_pct"] = f"{100 * bad['left']:.0f}"
    d["toygan_bad_right_pct"] = f"{100 * bad['right']:.0f}"
    d["toygan_steps"] = f"{NUM['toygan']['steps']:,}"
    tiles = []
    for v in NUM["vote"]:
        cls = "real" if v["kind"] == "real" else "fake"
        tiles.append(
            f'<div class="vt {cls}"><img src="figures/data/vote-{v["n"]:02d}.png">'
            f'<p class="tile-lab"><span>{v["n"]}</span><span class="vk" data-b="2">{"real" if v["kind"] == "real" else "fake"} · D {v["d_real"]:.2f}</span></p></div>')
    d["vote_tiles"] = "\n      ".join(tiles)
    d["vote_tiles_hook"] = "\n      ".join(
        f'<div class="vt {"real" if v["kind"] == "real" else "fake"}"><img src="figures/data/vote-{v["n"]:02d}.png">'
        f'<p class="tile-lab"><span>{v["n"]}</span><span class="vk" data-b="2">{"real" if v["kind"] == "real" else "generated"}</span></p></div>'
        for v in NUM["vote"])
    d["vote_fakes"] = ", ".join(str(v["n"]) for v in NUM["vote"] if v["kind"] != "real")
    ds = [v["d_real"] for v in NUM["vote"]]
    d["vote_dmin"], d["vote_dmax"] = f"{min(ds):.2f}", f"{max(ds):.2f}"
    gf = NUM["gan_final"]
    d["gan_final_dreal"], d["gan_final_dfake"] = f"{gf['D_real']:.2f}", f"{gf['D_fake']:.2f}"
    d["gan_final_lossD"] = f"{gf['loss_D']:.2f}"
    col = NUM["collapse"]
    d["collapse_DA"], d["collapse_DB"], d["collapse_slope"] = f"{col['D_A']:.2f}", f"{col['D_B']:.2f}", f"{col['slope_A']:.2f}"
    d["gan_params_G"] = f"{NUM['gan']['params_G'] / 1e6:.1f} million"
    d["gan_params_D"] = f"{round(NUM['gan']['params_D'], -3) / 1e3:.0f} thousand"
    op = NUM["oneplayer"]
    d["oneplayer_beyond3_pct"] = f"{100 * op['beyond3']:.0f}"
    d["oneplayer_near_bumps_pct"] = f"{100 * op['near_bumps']:.0f}"
    for k in ("D0", "D2", "D4"):
        d[f"oneplayer_{k}"] = f"{op[k]:.2f}"
    # saturation: the two slopes at D = 0.01 and their ratio
    D = 0.01
    d["sat_ratio"] = f"{(1 - D) / D:.0f}"
    # the optimal-discriminator check at p_G = p_data: -log 4
    d["minus_log4"] = f"{-math.log(4):.3f}".replace("-", "−")
    return d


def lookup(path):
    ctx = {**NUM, **derived()}
    cur = ctx
    for part in path.split("."):
        cur = cur[int(part)] if isinstance(cur, list) else cur[part]
    return cur


MACROS = {
    # role colours
    r"\cx": r"\textcolor{##245EDB}{#1}", r"\cenc": r"\textcolor{##8B2CDE}{#1}",
    r"\cdec": r"\textcolor{##0F766E}{#1}", r"\cz": r"\textcolor{##AA4E08}{#1}",
    r"\closs": r"\textcolor{##BE123C}{#1}", r"\cdisc": r"\textcolor{##334155}{#1}",
    # one spelling for every symbol, everywhere
    r"\N": r"\mathcal{N}", r"\KL": r"\mathrm{KL}", r"\E": r"\mathbb{E}", r"\ELBO": r"\mathrm{ELBO}",
    r"\JS": r"\mathrm{JS}", r"\pd": r"p_{\text{data}}", r"\pg": r"p_G",
    r"\eps": r"\varepsilon",
    # highlight the part of an equation that just changed
    r"\hl": r"\colorbox{##FFF1B8}{$\displaystyle #1$}",
}

# The five parts of the lecture, each with its role colour (also the section colour of its slides).
PARTS = [
    ("--disc", "Two players, one game"), ("--dec", "Training G through D"), ("--loss", "The minimax objective"),
    ("--x", "A GAN on MNIST"), ("--z", "Failures and control"),
]


def route(current):
    """The route with finished parts ticked and the current one highlighted (6 = all done)."""
    items = []
    for i, (col, name) in enumerate(PARTS, 1):
        state = "done" if i < current else ("now" if i == current else "todo")
        items.append(f'<li class="{state}" style="--c:var({col})"><span class="dot">{"✓" if state == "done" else i}</span>{name}</li>')
    return '<ol class="route-mini">' + "".join(items) + "</ol>"


def assemble():
    files = sorted((ROOT / "deck" / "slides").glob("*.html"))
    body = "\n".join(f.read_text() for f in files)

    def svg(m):
        s_ = (ROOT / "figures" / "diagrams" / f"{m.group(1)}.svg").read_text()
        return s_.replace('xmlns="http://www.w3.org/2000/svg" ', "")

    body = re.sub(r"<!--svg:([\w-]+)-->", svg, body)
    body = re.sub(r"<!--route:(\d)-->", lambda m: route(int(m.group(1))), body)

    def num(m):
        v = lookup(m.group(1))
        return v if isinstance(v, str) else fmt(v)

    body = re.sub(r"\{\{([\w.]+)\}\}", num, body)
    missing = re.findall(r"\{\{[^}]*\}\}", body)
    assert not missing, missing
    ids = re.findall(r'<section class="slide[^"]*" id="([\w-]+)"', body)
    assert len(ids) == len(set(ids)), "duplicate slide ids"
    return body


def page(title, body):
    macros = json.dumps(MACROS)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · ES 667</title>
<link rel="stylesheet" href="vendor/katex/katex.min.css">
<link rel="stylesheet" href="deck/deck.css">
</head>
<body>
<main id="deck">
{body}
</main>
<script>window.MATH_MACROS = {macros};</script>
<script src="vendor/katex/katex.min.js"></script>
<script src="vendor/katex/contrib/auto-render.min.js"></script>
<script src="deck/deck.js"></script>
<script src="deck/annotate.js"></script>
</body>
</html>
"""


def main():
    body = assemble()
    (ROOT / OUT).write_text(page(TITLE, body))
    print(f"{OUT}: {body.count('<section class=\"slide')} slides")


if __name__ == "__main__":
    main()
