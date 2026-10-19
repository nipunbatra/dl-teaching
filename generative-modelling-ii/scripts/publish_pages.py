"""Stage the built deck for GitHub Pages (interactives/generative-modelling-ii/ on master).

    python3 scripts/publish_pages.py OUTDIR

Copies only what the built index.html needs: deck/ styles and scripts, the KaTeX files,
figures/data/ and the two PDFs. The right-click review script is dropped, because
GitHub Pages has no notes server. Fails if any file the page references is missing.
"""
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(out):
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    page = (ROOT / "index.html").read_text()
    page = page.replace('<script src="deck/annotate.js"></script>\n', "")
    (out / "index.html").write_text(page)
    for f in ("deck/deck.css", "deck/deck.js", "deck/widgets.js",
              "vendor/katex/katex.min.css", "vendor/katex/katex.min.js", "vendor/katex/contrib/auto-render.min.js"):
        (out / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, out / f)
    shutil.copytree(ROOT / "vendor/katex/fonts", out / "vendor/katex/fonts")
    shutil.copytree(ROOT / "figures/data", out / "figures/data")
    for pdf in ("generative-modelling-ii.pdf", "generative-modelling-ii-builds.pdf"):
        shutil.copy2(ROOT / "exports" / pdf, out / pdf)
    # every local reference in the page must exist in the bundle
    refs = set(re.findall(r'(?:src|href)="([^"#?:]+)"', page))
    missing = sorted(r for r in refs if not (out / r).exists())
    assert not missing, missing
    n = sum(1 for _ in out.rglob("*") if _.is_file())
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"{out}: {n} files, {size:.1f} MB, {len(refs)} references checked")


if __name__ == "__main__":
    main(sys.argv[1])
