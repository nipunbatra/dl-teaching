#!/usr/bin/env python3
"""Build annotation-friendly HTML pages from the canonical Typst PDFs.

The HTML is only a review surface: every slide is rendered from the PDF, while
all edits continue to happen in the matching .typ source file.

Examples:
    python3 scripts/build_typst_review_html.py L1 L2
    python3 scripts/build_typst_review_html.py --compile L1
    python3 scripts/build_typst_review_html.py --compile --pages 5 L1
    python3 scripts/build_typst_review_html.py --compile --pages 5,13-14 L1
    python3 scripts/build_typst_review_html.py --all
"""

from __future__ import annotations

import argparse
import hashlib
import html
import os
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
PDF_DIR = REPO / "slides-pdf"
DEFAULT_OUTPUT = REPO / "slides-review"


def run(*args: str) -> str:
    result = subprocess.run(args, check=True, text=True, capture_output=True)
    return result.stdout


def pdf_metadata(pdf: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for line in run("pdfinfo", str(pdf)).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()
    return metadata


def pdf_text_regions(pdf: Path) -> list[list[dict[str, float | str]]]:
    """Extract positioned text blocks so browser annotations retain raw text."""
    result = subprocess.run(
        ["pdftotext", "-bbox-layout", "-enc", "UTF-8", str(pdf), "-"],
        check=True,
        capture_output=True,
    )
    root = ET.fromstring(result.stdout)
    pages: list[list[dict[str, float | str]]] = []
    for page in root.findall(".//{*}page"):
        page_width = float(page.attrib["width"])
        page_height = float(page.attrib["height"])
        regions: list[dict[str, float | str]] = []
        for block in page.findall(".//{*}block"):
            words = ["".join(word.itertext()).strip() for word in block.findall(".//{*}word")]
            text = " ".join(word for word in words if word).strip()
            if not text:
                continue
            x_min = float(block.attrib["xMin"])
            y_min = float(block.attrib["yMin"])
            x_max = float(block.attrib["xMax"])
            y_max = float(block.attrib["yMax"])
            regions.append(
                {
                    "text": text,
                    "left": 100 * x_min / page_width,
                    "top": 100 * y_min / page_height,
                    "width": 100 * (x_max - x_min) / page_width,
                    "height": 100 * (y_max - y_min) / page_height,
                }
            )
        pages.append(regions)
    return pages


def source_files() -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for source in sorted(REPO.glob("lecture*/L*.typ")):
        deck_id = source.stem.split("-", 1)[0].upper()
        sources[deck_id] = source
    return sources


def remove_stale_assets(asset_root: Path, deck_id: str, keep: Path) -> None:
    """Discard old generated renders for this deck after a successful build."""
    for candidate in asset_root.glob(f"{deck_id}-*-*dpi"):
        if candidate != keep and candidate.is_dir() and not candidate.is_symlink():
            shutil.rmtree(candidate)


def page_filename(page: int, total_pages: int) -> str:
    width = len(str(total_pages))
    return f"slide-{page:0{width}d}.png"


def parse_page_selection(spec: str | None, total_pages: int) -> list[int] | None:
    """Parse comma-separated physical pages and inclusive ranges."""
    if spec is None:
        return None
    selected: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            if not start_text or not end_text:
                raise ValueError(f"Open page range is not supported: {token!r}")
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError(f"Page range must be ascending: {token!r}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(token))
    if not selected:
        raise ValueError("--pages did not select any pages")
    invalid = sorted(page for page in selected if page < 1 or page > total_pages)
    if invalid:
        raise ValueError(
            f"Page(s) outside 1-{total_pages}: {', '.join(map(str, invalid))}"
        )
    return sorted(selected)


def latest_complete_assets(
    asset_root: Path,
    deck_id: str,
    dpi: int,
    total_pages: int,
    exclude: Path,
) -> Path | None:
    expected = {page_filename(page, total_pages) for page in range(1, total_pages + 1)}
    candidates = sorted(
        asset_root.glob(f"{deck_id}-*-{dpi}dpi"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if candidate == exclude or not candidate.is_dir() or candidate.is_symlink():
            continue
        names = {path.name for path in candidate.glob("slide-*.png")}
        if names == expected:
            return candidate
    return None


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def render_pages(
    pdf: Path,
    asset_root: Path,
    dpi: int,
    total_pages: int,
    selected_pages: list[int] | None = None,
) -> tuple[list[Path], str]:
    digest = hashlib.sha256(pdf.read_bytes()).hexdigest()[:12]
    asset_dir = asset_root / f"{pdf.stem}-{digest}-{dpi}dpi"
    existing = sorted(asset_dir.glob("slide-*.png")) if asset_dir.exists() else []
    if len(existing) == total_pages:
        remove_stale_assets(asset_root, pdf.stem, asset_dir)
        return existing, "cached"
    if asset_dir.exists():
        raise RuntimeError(
            f"Incomplete generated asset directory: {asset_dir}. "
            "Remove slides-review/ and rebuild."
        )

    asset_root.mkdir(parents=True, exist_ok=True)
    previous = None
    if selected_pages is not None:
        previous = latest_complete_assets(
            asset_root, pdf.stem, dpi, total_pages, exclude=asset_dir
        )
        if previous is None:
            print(
                f"{pdf.stem}: no complete render cache; falling back to a full rasterization"
            )

    with tempfile.TemporaryDirectory(prefix=f".{pdf.stem}-", dir=asset_root) as tmp:
        tmp_dir = Path(tmp)
        if selected_pages is not None and previous is not None:
            for page in range(1, total_pages + 1):
                name = page_filename(page, total_pages)
                link_or_copy(previous / name, tmp_dir / name)
            for page in selected_pages:
                update_prefix = tmp_dir / f".update-{page}"
                subprocess.run(
                    [
                        "pdftoppm",
                        "-f",
                        str(page),
                        "-l",
                        str(page),
                        "-singlefile",
                        "-png",
                        "-r",
                        str(dpi),
                        str(pdf),
                        str(update_prefix),
                    ],
                    check=True,
                )
                update_prefix.with_suffix(".png").replace(
                    tmp_dir / page_filename(page, total_pages)
                )
            mode = "incremental"
        else:
            subprocess.run(
                [
                    "pdftoppm",
                    "-png",
                    "-r",
                    str(dpi),
                    str(pdf),
                    str(tmp_dir / "slide"),
                ],
                check=True,
            )
            mode = "full"
        rendered = sorted(tmp_dir.glob("slide-*.png"))
        if len(rendered) != total_pages:
            raise RuntimeError(
                f"Expected {total_pages} pages from {pdf}, rendered {len(rendered)}"
            )
        tmp_dir.rename(asset_dir)
    remove_stale_assets(asset_root, pdf.stem, asset_dir)
    return sorted(asset_dir.glob("slide-*.png")), mode


def page_html(
    deck_id: str,
    title: str,
    source: Path | None,
    pdf: Path,
    images: list[Path],
    text_regions: list[list[dict[str, float | str]]],
    output_dir: Path,
) -> str:
    source_label = source.relative_to(REPO).as_posix() if source else "Typst source not found"
    pdf_label = pdf.relative_to(REPO).as_posix()
    cards: list[str] = []
    for page, image in enumerate(images, start=1):
        image_url = image.relative_to(output_dir).as_posix()
        context = f"{deck_id} slide {page}; edit {source_label}"
        regions = text_regions[page - 1]
        region_markup = []
        for region in regions:
            region_text = str(region["text"])
            region_markup.append(
                '<span class="text-region" '
                f'data-text="{html.escape(region_text, quote=True)}" '
                f'aria-label="{html.escape(region_text, quote=True)}" '
                f'style="left:{float(region["left"]):.4f}%;top:{float(region["top"]):.4f}%;'
                f'width:{float(region["width"]):.4f}%;height:{float(region["height"]):.4f}%">'
                f'{html.escape(region_text)}</span>'
            )
        page_text = " ".join(str(region["text"]) for region in regions)
        cards.append(
            f'''<article class="slide" id="slide-{page}" data-deck="{html.escape(deck_id)}"
              data-slide="{page}" data-source="{html.escape(source_label)}">
  <div class="slide-canvas" data-page-text="{html.escape(page_text, quote=True)}">
    <img src="{html.escape(image_url)}" alt="{html.escape(context)}" loading="lazy">
    <div class="text-layer" aria-label="Extracted text for {html.escape(context)}">{''.join(region_markup)}</div>
  </div>
  <footer><strong>{html.escape(deck_id)} · slide {page}</strong><span>{html.escape(source_label)}</span></footer>
</article>'''
        )

    return f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(deck_id)} review · {html.escape(title)}</title>
  <style>
    :root {{ color-scheme: light; --ink:#24343a; --paper:#efeeeb; --accent:#d97757; }}
    * {{ box-sizing:border-box; }}
    html {{ scroll-behavior:smooth; scroll-padding-top:72px; scroll-snap-type:y proximity; }}
    body {{ margin:0; background:#dad9d5; color:var(--ink); font:14px/1.4 ui-sans-serif,system-ui,sans-serif; }}
    .toolbar {{ position:sticky; top:0; z-index:10; display:flex; gap:18px; align-items:center;
      min-height:58px; padding:9px 18px; background:rgba(36,52,58,.96); color:white;
      box-shadow:0 2px 10px #0004; backdrop-filter:blur(8px); }}
    .toolbar strong {{ font-size:16px; }}
    .toolbar .hint {{ flex:1; color:#dfe5e6; }}
    .toolbar a {{ color:white; text-underline-offset:3px; }}
    .toolbar button {{ padding:6px 9px; border:1px solid #ffffff55; border-radius:6px;
      background:#fff; color:#24343a; cursor:pointer; font:inherit; }}
    .toolbar input {{ width:72px; padding:6px 8px; border:1px solid #ffffff55; border-radius:6px;
      background:#fff; color:#111; }}
    main {{ width:min(1400px, 100%); margin:0 auto; padding:22px; }}
    .slide {{ margin:0 0 28px; background:white; border-radius:8px; overflow:hidden;
      box-shadow:0 4px 18px #0003; scroll-margin-top:76px; scroll-snap-align:start; }}
    .slide-canvas {{ position:relative; width:100%; aspect-ratio:16 / 9; background:white; }}
    .slide img {{ display:block; width:100%; height:auto; aspect-ratio:16 / 9;
      object-fit:contain; background:white; pointer-events:none; }}
    .text-layer {{ position:absolute; inset:0; z-index:2; pointer-events:none; }}
    .text-region {{ position:absolute; display:block; overflow:hidden; color:transparent;
      font-size:1px; line-height:1; white-space:nowrap; pointer-events:auto; user-select:text;
      cursor:context-menu; border-radius:2px; }}
    .text-region:hover {{ background:rgba(217,119,87,.10); outline:1px solid rgba(217,119,87,.45); }}
    .slide footer {{ display:flex; justify-content:space-between; gap:20px; padding:9px 13px;
      border-top:1px solid #ddd; color:#59666a; font-size:12px; }}
    .slide:target {{ outline:4px solid var(--accent); }}
    @media (max-width:760px) {{
      .toolbar {{ flex-wrap:wrap; gap:7px 14px; }} .toolbar .hint {{ order:3; flex-basis:100%; }}
      main {{ padding:10px; }} .slide footer span {{ display:none; }}
    }}
  </style>
</head>
<body data-deck="{html.escape(deck_id)}" data-source="{html.escape(source_label)}">
  <header class="toolbar">
    <strong>{html.escape(deck_id)} · {html.escape(title)}</strong>
    <span class="hint">Right-click text or a slide region → Quick Annotate or Annotate. Batch comments, then reload once.</span>
    <label>Slide <input id="jump" type="number" min="1" max="{len(images)}" placeholder="1"></label>
    <button id="reload" type="button" title="Reload after Codex finishes a batch of annotations">Reload</button>
    <a href="../{html.escape(pdf_label)}">PDF</a>
  </header>
  <main>{''.join(cards)}</main>
  <script>
    const jump = document.querySelector('#jump');
    const reload = document.querySelector('#reload');
    const go = () => {{
      const n = Math.max(1, Math.min({len(images)}, Number(jump.value) || 1));
      document.querySelector(`#slide-${{n}}`)?.scrollIntoView({{behavior:'smooth'}});
    }};
    jump.addEventListener('change', go);
    jump.addEventListener('keydown', event => {{ if (event.key === 'Enter') go(); }});
    reload.addEventListener('click', () => window.location.reload());
  </script>
</body>
</html>
'''


def build_index(output_dir: Path) -> None:
    links = []
    for page in sorted(output_dir.glob("L*.html")):
        links.append(f'<li><a href="{html.escape(page.name)}">{html.escape(page.stem)}</a></li>')
    index = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Typst slide review</title><style>
body{{max-width:760px;margin:60px auto;padding:0 24px;font:18px/1.6 system-ui;background:#efeeeb;color:#24343a}}
a{{color:#b55032}} li{{margin:8px 0}}
</style></head><body><h1>Typst slide review</h1>
<p>Open a deck, then right-click any slide and choose <strong>Quick Annotate</strong> or <strong>Annotate</strong>.</p>
<ul>{''.join(links)}</ul></body></html>'''
    (output_dir / "index.html").write_text(index, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("decks", nargs="*", help="Deck IDs such as L1 L2 L3A")
    parser.add_argument("--all", action="store_true", help="Build every PDF in slides-pdf/")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile each matching .typ source to slides-pdf/ before building HTML",
    )
    parser.add_argument("--dpi", type=int, default=120, help="Rasterization resolution")
    parser.add_argument(
        "--pages",
        metavar="PAGES",
        help=(
            "Only rerasterize these physical pages, for example 5 or 5,13-14. "
            "Other pages are reused from the latest complete render; falls back to full. "
            "Use a full build after adding/removing slides or changing shared styling."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    for command in ("pdfinfo", "pdftoppm", "pdftotext"):
        if not shutil.which(command):
            raise SystemExit(f"Missing required command: {command}")

    if args.all:
        pdfs = sorted(PDF_DIR.glob("L*.pdf"))
    elif args.decks:
        pdfs = [PDF_DIR / f"{deck.upper()}.pdf" for deck in args.decks]
    else:
        parser.error("pass one or more deck IDs, or use --all")

    missing = [str(pdf) for pdf in pdfs if not pdf.exists()] if not args.compile else []
    if missing:
        raise SystemExit("Missing PDF(s): " + ", ".join(missing))

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = source_files()
    for pdf in pdfs:
        deck_id = pdf.stem.upper()
        source = sources.get(deck_id)
        if args.compile:
            if source is None:
                raise SystemExit(f"Typst source not found for {deck_id}")
            subprocess.run(
                [
                    "typst",
                    "compile",
                    "--root",
                    str(REPO),
                    "--input",
                    "handout=true",
                    str(source),
                    str(pdf),
                ],
                check=True,
            )
        metadata = pdf_metadata(pdf)
        pages = int(metadata["Pages"])
        try:
            selected_pages = parse_page_selection(args.pages, pages)
        except ValueError as error:
            raise SystemExit(str(error)) from error
        title = metadata.get("Title") or deck_id
        images, render_mode = render_pages(
            pdf,
            output_dir / "assets",
            args.dpi,
            pages,
            selected_pages=selected_pages,
        )
        text_regions = pdf_text_regions(pdf)
        if len(text_regions) != pages:
            raise RuntimeError(
                f"Expected text for {pages} pages from {pdf}, extracted {len(text_regions)}"
            )
        review = page_html(deck_id, title, source, pdf, images, text_regions, output_dir)
        destination = output_dir / f"{deck_id}.html"
        destination.write_text(review, encoding="utf-8")
        page_note = f"; pages {args.pages}" if render_mode == "incremental" else ""
        print(
            f"{deck_id}: {pages} slides ({render_mode} render{page_note}) "
            f"-> {destination.relative_to(REPO) if destination.is_relative_to(REPO) else destination}"
        )

    build_index(output_dir)
    print("Serve with: python3 -m http.server 8765 --bind 127.0.0.1 --directory .")


if __name__ == "__main__":
    main()
