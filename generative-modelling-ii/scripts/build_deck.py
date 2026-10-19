"""Assemble index.html from deck/slides/*.html (the same build as Generative Modelling I).

    python3 scripts/build_deck.py           # every {{number}} must exist in numbers.json
    python3 scripts/build_deck.py --draft   # missing numbers become "??" (for previews while models train)

* <!--svg:NAME-->   is replaced by figures/diagrams/NAME.svg (inlined so its groups can build)
* {{a.b.0}}         is replaced by a value from figures/data/numbers.json
Every number on a slide therefore comes from the same run as the figures.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NUM = json.loads((ROOT / "figures" / "data" / "numbers.json").read_text())
DRAFT = "--draft" in sys.argv


def fmt(v):
    """numbers.json is already rounded for display; only swap in a typographic minus."""
    return str(v).replace("-", "−") if isinstance(v, (int, float)) else str(v)


def lookup(path):
    cur = NUM
    for part in path.split("."):
        cur = cur[int(part)] if isinstance(cur, list) else cur[part]
    return cur


PARTS = [("--z", "Corrupt a digit"), ("--dec", "Train: predict the noise"), ("--x", "Sample: run it backwards"),
         ("--loss", "Why this loss: the ELBO"), ("--xt", "Inside the network"), ("--enc", "Steer with a condition"),
         ("--loss", "Text by denoising")]


def route(current):
    """The route with finished parts ticked and the current one highlighted (as in Lecture I)."""
    items = []
    for i, (col, name) in enumerate(PARTS, 1):
        state = "done" if i < current else ("now" if i == current else "todo")
        items.append(f'<li class="{state}" style="--c:var({col})"><span class="dot">{"✓" if state == "done" else i}</span>{name}</li>')
    return '<ol class="route-mini">' + "".join(items) + "</ol>"


def main():
    body = "\n".join(f.read_text() for f in sorted((ROOT / "deck" / "slides").glob("*.html")))

    def svg(m):
        p = ROOT / "figures" / "diagrams" / f"{m.group(1)}.svg"
        if not p.exists() and DRAFT:
            return f'<div class="callout plain">missing diagram {m.group(1)}</div>'
        return p.read_text().replace('xmlns="http://www.w3.org/2000/svg" ', "")

    body = re.sub(r"<!--svg:([\w-]+)-->", svg, body)
    body = re.sub(r"<!--route:(\d)-->", lambda m: route(int(m.group(1))), body)
    # standalone equations are display maths: <div class="mathblock">$...$</div> -> $$...$$
    body = re.sub(r'(<div class="mathblock[^"]*"[^>]*>)\$([^$]+)\$(</div>)', r"\1$$\2$$\3", body)
    # small data files for the live widgets, inlined so the deck also works from file://
    data = ROOT / "figures" / "data"
    body += "\n".join(f'<script type="application/json" id="{n}-data">{(data / f"{n}.json").read_text()}</script>'
                      for n in ("explorer", "scrub") if (data / f"{n}.json").exists())

    def num(m):
        try:
            v = lookup(m.group(1))
        except (KeyError, IndexError, TypeError):
            if DRAFT:
                return "??"
            raise
        return v if isinstance(v, str) else fmt(v)

    body = re.sub(r"\{\{([\w.\-]+)\}\}", num, body)
    missing = re.findall(r"\{\{[^}]*\}\}", body)
    assert not missing, missing

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>From noise to images · ES 667 Generative Modelling II</title>
<link rel="stylesheet" href="vendor/katex/katex.min.css">
<link rel="stylesheet" href="deck/deck.css">
</head>
<body>
<main id="deck">
{body}
</main>
<script>
window.MATH_MACROS = {{
  "\\\\cx": "\\\\textcolor{{##245EDB}}{{#1}}", "\\\\cenc": "\\\\textcolor{{##8B2CDE}}{{#1}}",
  "\\\\cdec": "\\\\textcolor{{##0F766E}}{{#1}}", "\\\\cz": "\\\\textcolor{{##AA4E08}}{{#1}}",
  "\\\\closs": "\\\\textcolor{{##BE123C}}{{#1}}", "\\\\cxt": "\\\\textcolor{{##334155}}{{#1}}",
  "\\\\ab": "\\\\bar\\\\alpha", "\\\\eh": "\\\\hat\\\\epsilon"
}};
</script>
<script src="vendor/katex/katex.min.js"></script>
<script src="vendor/katex/contrib/auto-render.min.js"></script>
<script src="deck/widgets.js"></script>
<script src="deck/deck.js"></script>
<script src="deck/annotate.js"></script>
</body>
</html>
"""
    (ROOT / "index.html").write_text(html)
    n = body.count('<section class="slide')
    print(f"index.html: {n} slides" + (" (draft)" if DRAFT else ""))


if __name__ == "__main__":
    main()
