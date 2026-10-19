"""Stage the built deck for GitHub Pages (interactives/generative-modelling-i-gans/ on master).

    python3 scripts/publish_pages.py OUTDIR

Copies only what the built gans.html needs: deck/ styles and script, the KaTeX files, the figures,
the two published/borrowed images and the two PDFs. The page becomes OUTDIR/index.html and the
right-click review script is dropped, because GitHub Pages has no notes server.
Fails if any file the page references is missing.
"""
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "generative-modelling-i-gans"


def main(out):
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    page = (ROOT / "gans.html").read_text()
    assert '<script src="deck/annotate.js"></script>\n' in page
    page = page.replace('<script src="deck/annotate.js"></script>\n', "")
    (out / "index.html").write_text(page)
    for f in ("deck/deck.css", "deck/deck.js",
              "vendor/katex/katex.min.css", "vendor/katex/katex.min.js", "vendor/katex/contrib/auto-render.min.js",
              "vendor/katex/LICENSE", "assets/gallery/stylegan-faces.jpg", "assets/vae-lecture/vae16-samples.png"):
        (out / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, out / f)
    shutil.copytree(ROOT / "vendor/katex/fonts", out / "vendor/katex/fonts")
    shutil.copytree(ROOT / "figures/data", out / "figures/data", ignore=shutil.ignore_patterns("numbers.json"))
    for src, dst in (("gans.pdf", f"{NAME}.pdf"), ("gans-builds.pdf", f"{NAME}-builds.pdf")):
        shutil.copy2(ROOT / "exports" / src, out / dst)
    # every local reference in the page must exist in the bundle
    refs = set(re.findall(r'(?:src|href)="([^"#?:]+)"', page))
    missing = sorted(r for r in refs if not (out / r).exists())
    assert not missing, missing
    n = sum(1 for _ in out.rglob("*") if _.is_file())
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"{out}: {n} files, {size:.1f} MB, {len(refs)} references checked")


if __name__ == "__main__":
    main(sys.argv[1])
