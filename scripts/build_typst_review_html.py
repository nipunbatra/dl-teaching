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

For recurring annotation work, prefer the warm review server instead:
    python3 scripts/serve_typst_review.py L1
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
PDF_DIR = REPO / "slides-pdf"
DEFAULT_OUTPUT = REPO / "slides-review"


def atomic_write_text(path: Path, content: str) -> None:
    """Publish generated files atomically so an open review page never sees half a build."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


@contextmanager
def deck_lock(output_dir: Path, deck_id: str):
    """Serialize concurrent annotation rebuilds for the same deck."""
    lock_dir = output_dir / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / f"{deck_id}.lock").open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def pdf_revision(pdf: Path) -> str:
    return hashlib.sha256(pdf.read_bytes()).hexdigest()[:12]


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


def source_slide_sections(
    source: Path | None, total_pages: int, deck_title: str
) -> list[dict[str, int | str]]:
    """Map physical pages to exact Typst source blocks and source line numbers."""
    if source is None:
        return []
    lines = source.read_text(encoding="utf-8").splitlines()
    markers: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if re.fullmatch(r"#title-slide\(\s*\)", stripped):
            markers.append((index, deck_title))
            continue
        if stripped.startswith("#focus-slide["):
            markers.append((index, "Closing focus"))
            continue
        match = re.match(r"^(==?)\s+(.+?)\s*$", stripped)
        if not match:
            continue
        title = re.sub(r"\s+#[A-Za-z][\w-]*\s*$", "", match.group(2)).strip()
        markers.append((index, title))
    if len(markers) != total_pages:
        print(
            f"{source.name}: found {len(markers)} source slide markers for "
            f"{total_pages} PDF pages; source panels disabled"
        )
        return []
    sections: list[dict[str, int | str]] = []
    for page, (start, title) in enumerate(markers, start=1):
        end = markers[page][0] if page < len(markers) else len(lines)
        code = "\n".join(lines[start:end]).rstrip()
        sections.append({"title": title, "line": start + 1, "code": code})
    return sections


def source_code_html(code: str, start_line: int) -> str:
    rendered: list[str] = []
    for offset, line in enumerate(code.splitlines() or [""]):
        number = start_line + offset
        rendered.append(
            f'<span class="source-line" data-line="{number}">'
            f'<span class="line-number">{number}</span>'
            f'<code>{html.escape(line) or "&nbsp;"}</code></span>'
        )
    return "".join(rendered)


def remove_stale_assets(
    asset_root: Path, deck_id: str, keep: Path, retain: int = 3
) -> None:
    """Keep recent revisions so lazy-loaded slides remain valid until the user reloads."""
    candidates = sorted(
        (
            candidate
            for candidate in asset_root.glob(f"{deck_id}-*-*dpi")
            if candidate.is_dir() and not candidate.is_symlink()
        ),
        key=lambda candidate: candidate.stat().st_mtime,
        reverse=True,
    )
    protected = {
        keep,
        *[candidate for candidate in candidates if candidate != keep][
            : max(0, retain - 1)
        ],
    }
    for candidate in candidates:
        if candidate not in protected:
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
    retain_revisions: int = 3,
) -> tuple[list[Path], str]:
    digest = pdf_revision(pdf)
    asset_dir = asset_root / f"{pdf.stem}-{digest}-{dpi}dpi"
    existing = sorted(asset_dir.glob("slide-*.png")) if asset_dir.exists() else []
    if len(existing) == total_pages:
        remove_stale_assets(
            asset_root, pdf.stem, asset_dir, retain=retain_revisions
        )
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
    remove_stale_assets(asset_root, pdf.stem, asset_dir, retain=retain_revisions)
    return sorted(asset_dir.glob("slide-*.png")), mode


def page_html(
    deck_id: str,
    title: str,
    source: Path | None,
    pdf: Path,
    images: list[Path],
    text_regions: list[list[dict[str, float | str]]],
    source_sections: list[dict[str, int | str]],
    revision: str,
    output_dir: Path,
) -> str:
    source_label = source.relative_to(REPO).as_posix() if source else "Typst source not found"
    pdf_label = pdf.relative_to(REPO).as_posix()
    cards: list[str] = []
    for page, image in enumerate(images, start=1):
        image_url = image.relative_to(output_dir).as_posix()
        section = source_sections[page - 1] if source_sections else None
        slide_title = str(section["title"]) if section else f"Slide {page}"
        source_line = int(section["line"]) if section else 0
        source_reference = (
            f"{source_label}:{source_line}" if source_line else source_label
        )
        context = f"{deck_id} slide {page}: {slide_title}; edit {source_reference}"
        regions = text_regions[page - 1]
        region_markup = []
        for region in regions:
            region_text = str(region["text"])
            region_markup.append(
                '<span class="text-region" '
                f'data-text="{html.escape(region_text, quote=True)}" '
                f'data-slide="{page}" data-source="{html.escape(source_reference, quote=True)}" '
                f'aria-label="{html.escape(region_text, quote=True)}" '
                f'style="left:{float(region["left"]):.4f}%;top:{float(region["top"]):.4f}%;'
                f'width:{float(region["width"]):.4f}%;height:{float(region["height"]):.4f}%">'
                f'{html.escape(region_text)}</span>'
            )
        page_text = " ".join(str(region["text"]) for region in regions)
        source_panel = ""
        source_button = ""
        if section:
            source_panel = (
                f'<aside class="source-panel" id="source-{page}" hidden '
                f'aria-label="Typst source for {html.escape(context)}">'
                f'<div class="source-heading"><strong>Typst source</strong>'
                f'<span>{html.escape(source_reference)}</span></div>'
                f'<pre>{source_code_html(str(section["code"]), source_line)}</pre></aside>'
            )
            source_button = (
                f'<button class="source-toggle" type="button" aria-expanded="false" '
                f'aria-controls="source-{page}" title="Show exact Typst source beside this slide">Source</button>'
            )
        cards.append(
            f'''<article class="slide" id="slide-{page}" data-deck="{html.escape(deck_id)}"
              data-slide="{page}" data-title="{html.escape(slide_title, quote=True)}"
              data-source="{html.escape(source_reference, quote=True)}">
  <div class="slide-body">
    <div class="slide-canvas" data-page-text="{html.escape(page_text, quote=True)}"
         aria-label="Rendered {html.escape(context)}">
      <img src="{html.escape(image_url)}" alt="{html.escape(context)}" loading="lazy">
      <div class="text-layer" aria-label="Extracted text for {html.escape(context)}">{''.join(region_markup)}</div>
    </div>
    {source_panel}
  </div>
  <footer><strong>{html.escape(deck_id)} · slide {page}</strong>
    <span class="slide-title">{html.escape(slide_title)}</span>
    <span class="source-reference">{html.escape(source_reference)}</span>{source_button}</footer>
</article>'''
        )

    return f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(deck_id)} review · {html.escape(title)}</title>
  <meta name="description" content="Source-aware review surface for {html.escape(title)}">
  <style>
    :root {{ color-scheme:light; --ink:#24343a; --paper:#efeeeb; --accent:#d97757;
      --muted:#647277; --line:#c9cfce; --toolbar:#24343a; }}
    * {{ box-sizing:border-box; }}
    html {{ scroll-padding-top:72px; }}
    body {{ margin:0; background:#dad9d5; color:var(--ink);
      font:14px/1.4 ui-sans-serif,system-ui,-apple-system,sans-serif; }}
    .skip-link {{ position:fixed; left:12px; top:8px; z-index:30; transform:translateY(-160%);
      padding:8px 12px; background:white; color:var(--ink); border-radius:4px; }}
    .skip-link:focus {{ transform:translateY(0); }}
    .toolbar {{ position:sticky; top:0; z-index:10; display:flex; gap:18px; align-items:center;
      min-height:58px; padding:9px 18px; background:rgba(36,52,58,.97); color:white;
      box-shadow:0 2px 12px rgba(36,52,58,.28); backdrop-filter:blur(8px); }}
    .toolbar strong {{ font-size:16px; letter-spacing:-.01em; }}
    .toolbar .hint {{ flex:1; color:#dfe5e6; text-wrap:pretty; }}
    .toolbar a {{ color:white; text-underline-offset:3px; }}
    .toolbar button {{ min-height:34px; padding:6px 10px; border:1px solid #ffffff55;
      border-radius:5px; background:#fff; color:var(--ink); cursor:pointer; font:600 13px/1.2 inherit;
      transition:transform .18s ease, background .18s ease, border-color .18s ease; }}
    .toolbar button:hover {{ background:#f5f1ec; border-color:#fff; }}
    .toolbar button:active {{ transform:translateY(1px); }}
    .toolbar button.update-ready {{ background:#f5d9ca; border-color:#f2a27f; }}
    .build-status {{ display:none; align-items:center; gap:6px; color:#f5d9ca;
      font:600 12px/1.2 ui-monospace,SFMono-Regular,Menlo,monospace; white-space:nowrap; }}
    .build-status.visible {{ display:inline-flex; }}
    .build-status::before {{ content:""; width:7px; height:7px; border-radius:50%; background:#e6926d; }}
    .toolbar input {{ width:72px; padding:6px 8px; border:1px solid #ffffff55; border-radius:6px;
      background:#fff; color:var(--ink); font-variant-numeric:tabular-nums; }}
    button:focus-visible, input:focus-visible, a:focus-visible {{ outline:3px solid #f2a27f; outline-offset:2px; }}
    main {{ width:min(1400px, 100%); margin:0 auto; padding:22px; }}
    .slide {{ margin:0 0 28px; background:white; border-radius:8px; overflow:hidden;
      box-shadow:0 4px 18px rgba(36,52,58,.22); scroll-margin-top:76px; }}
    .slide-body {{ display:grid; grid-template-columns:minmax(0, 1fr); align-items:start; }}
    .slide.source-open .slide-body {{ grid-template-columns:minmax(0, 1.42fr) minmax(320px, .78fr); }}
    .slide-canvas {{ position:relative; width:100%; aspect-ratio:16 / 9; background:white; }}
    .slide img {{ display:block; width:100%; height:auto; aspect-ratio:16 / 9;
      object-fit:contain; background:white; pointer-events:none; }}
    .text-layer {{ position:absolute; inset:0; z-index:2; pointer-events:none; }}
    .text-region {{ position:absolute; display:block; overflow:hidden; color:transparent;
      font-size:1px; line-height:1; white-space:nowrap; pointer-events:auto; user-select:text;
      cursor:context-menu; border-radius:2px; }}
    .text-region:hover {{ background:rgba(217,119,87,.10); outline:1px solid rgba(217,119,87,.45); }}
    .source-panel {{ min-width:0; height:clamp(320px, 39vw, 500px); overflow:auto;
      background:#f5f3ee; border-left:1px solid var(--line); }}
    .source-heading {{ position:sticky; top:0; z-index:1; display:flex; justify-content:space-between;
      gap:16px; padding:10px 12px; background:rgba(245,243,238,.96); border-bottom:1px solid var(--line);
      color:var(--muted); font:12px/1.3 ui-monospace,SFMono-Regular,Menlo,monospace; }}
    .source-heading strong {{ color:var(--ink); }}
    .source-panel pre {{ margin:0; padding:8px 0 16px; white-space:pre-wrap; tab-size:2;
      font:12px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace; }}
    .source-line {{ display:grid; grid-template-columns:44px minmax(0, 1fr); min-height:19px; }}
    .source-line:hover {{ background:rgba(217,119,87,.10); }}
    .line-number {{ padding-right:10px; color:#9aa4a6; text-align:right; user-select:none; }}
    .source-line code {{ padding-right:14px; color:#2d3d42; user-select:text; cursor:context-menu; }}
    .slide footer {{ display:grid; grid-template-columns:auto minmax(0,1fr) auto auto; align-items:center;
      gap:14px; min-height:39px; padding:7px 13px; border-top:1px solid #ddd;
      color:#59666a; font-size:12px; }}
    .slide-title {{ overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
    .source-reference {{ color:#7c898d; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
    .source-toggle {{ padding:4px 8px; border:1px solid #b9c1c1; border-radius:4px;
      background:transparent; color:var(--ink); cursor:pointer; font:600 11px/1.2 inherit; }}
    .source-toggle:hover {{ background:#f3efe9; border-color:#8c999b; }}
    .source-toggle:active {{ transform:translateY(1px); }}
    .slide:target {{ outline:4px solid var(--accent); }}
    @media (max-width:760px) {{
      .toolbar {{ flex-wrap:wrap; gap:7px 14px; }} .toolbar .hint {{ order:3; flex-basis:100%; }}
      main {{ padding:10px; }} .slide.source-open .slide-body {{ grid-template-columns:1fr; }}
      .source-panel {{ height:380px; border-left:0; border-top:1px solid var(--line); }}
      .slide footer {{ grid-template-columns:auto minmax(0,1fr) auto; }}
      .source-reference {{ display:none; }}
    }}
  </style>
</head>
<body data-deck="{html.escape(deck_id)}" data-source="{html.escape(source_label)}"
      data-revision="{html.escape(revision)}">
  <a class="skip-link" href="#slide-1">Skip to slides</a>
  <header class="toolbar">
    <strong>{html.escape(deck_id)} · {html.escape(title)}</strong>
    <span class="hint">Annotate rendered text for visual edits; open Source for exact Typst or formulas. Batch comments, then update once.</span>
    <span class="build-status" id="build-status" role="status">Update ready</span>
    <label>Slide <input id="jump" type="number" min="1" max="{len(images)}" placeholder="1"></label>
    <button id="reload" type="button" title="Apply the latest completed review build">Reload</button>
    <a href="../{html.escape(pdf_label)}">PDF</a>
  </header>
  <main>{''.join(cards)}</main>
  <script>
    const jump = document.querySelector('#jump');
    const reload = document.querySelector('#reload');
    const buildStatus = document.querySelector('#build-status');
    const deck = document.body.dataset.deck;
    const loadedRevision = document.body.dataset.revision;
    const storageKey = `typst-review:${{deck}}:slide`;
    if ('scrollRestoration' in history) history.scrollRestoration = 'manual';
    let currentSlide = Number(new URLSearchParams(location.search).get('at')) ||
      Number(location.hash.match(/slide-(\d+)/)?.[1]) ||
      Number(sessionStorage.getItem(storageKey)) || 1;
    let updateReady = false;
    const go = () => {{
      const n = Math.max(1, Math.min({len(images)}, Number(jump.value) || 1));
      currentSlide = n;
      jump.value = String(n);
      remember();
      history.replaceState(null, '', `${{location.pathname}}${{location.search}}#slide-${{n}}`);
      document.querySelector(`#slide-${{n}}`)?.scrollIntoView({{behavior:'smooth'}});
    }};
    const remember = () => sessionStorage.setItem(storageKey, String(currentSlide));
    const reloadAtCurrentSlide = () => {{
      currentSlide = Math.max(1, Math.min({len(images)}, Number(jump.value) || currentSlide));
      remember();
      const target = new URL(location.href);
      target.searchParams.set('at', String(currentSlide));
      target.hash = `slide-${{currentSlide}}`;
      window.location.replace(target.href);
    }};
    jump.addEventListener('change', go);
    jump.addEventListener('keydown', event => {{ if (event.key === 'Enter') go(); }});
    reload.addEventListener('click', reloadAtCurrentSlide);
    document.querySelectorAll('.source-toggle').forEach(button => {{
      button.addEventListener('click', () => {{
        const slide = button.closest('.slide');
        const panel = document.querySelector(`#${{button.getAttribute('aria-controls')}}`);
        const opening = panel.hidden;
        panel.hidden = !opening;
        slide.classList.toggle('source-open', opening);
        button.setAttribute('aria-expanded', String(opening));
        button.textContent = opening ? 'Rendered only' : 'Source';
      }});
    }});
    const slides = [...document.querySelectorAll('.slide')];
    let scrollFrame = 0;
    const updateCurrentFromViewport = () => {{
      scrollFrame = 0;
      if (document.activeElement === jump) return;
      const toolbarBottom = document.querySelector('.toolbar').getBoundingClientRect().bottom;
      const focusY = Math.max(toolbarBottom + 1, window.innerHeight * .5);
      const centered = document.elementFromPoint(window.innerWidth * .5, focusY)?.closest('.slide');
      const visible = centered || slides
        .filter(slide => {{
          const rect = slide.getBoundingClientRect();
          return rect.bottom > toolbarBottom && rect.top < window.innerHeight;
        }})
        .sort((a, b) => {{
          const aRect = a.getBoundingClientRect();
          const bRect = b.getBoundingClientRect();
          return Math.abs((aRect.top + aRect.bottom) / 2 - focusY) -
            Math.abs((bRect.top + bRect.bottom) / 2 - focusY);
      }})[0];
      if (!visible) return;
      currentSlide = Number(visible.dataset.slide);
      jump.value = String(currentSlide);
      remember();
      history.replaceState(null, '', `${{location.pathname}}${{location.search}}#slide-${{currentSlide}}`);
    }};
    const scheduleViewportUpdate = () => {{
      if (!scrollFrame) scrollFrame = requestAnimationFrame(updateCurrentFromViewport);
    }};
    window.addEventListener('scroll', scheduleViewportUpdate, {{ passive:true }});
    const markUpdateReady = () => {{
      if (updateReady) return;
      updateReady = true;
      buildStatus.classList.add('visible');
      reload.classList.add('update-ready');
      reload.textContent = 'Apply update';
    }};
    const checkForUpdate = async () => {{
      try {{
        const response = await fetch(`${{deck}}.status.json?check=${{Date.now()}}`, {{ cache:'no-store' }});
        if (!response.ok) return;
        const status = await response.json();
        if (status.revision && status.revision !== loadedRevision) markUpdateReady();
      }} catch (_) {{ /* Plain static hosting may be temporarily unavailable during startup. */ }}
    }};
    setInterval(checkForUpdate, 2500);
    document.addEventListener('visibilitychange', () => {{ if (!document.hidden) checkForUpdate(); }});
    document.addEventListener('keydown', event => {{
      if (event.target.matches('input, textarea, code')) return;
      if (event.key === '/') {{ event.preventDefault(); jump.focus(); jump.select(); }}
      if (event.key.toLowerCase() === 'r') reloadAtCurrentSlide();
      if (event.key.toLowerCase() === 's')
        document.querySelector(`#slide-${{currentSlide}} .source-toggle`)?.click();
    }});
    const initialSlide = currentSlide;
    jump.value = String(initialSlide);
    const placeInitialSlide = () => {{
      const root = document.documentElement;
      const previousBehavior = root.style.scrollBehavior;
      const previousSnap = root.style.scrollSnapType;
      root.style.scrollBehavior = 'auto';
      root.style.scrollSnapType = 'none';
      document.querySelector(`#slide-${{initialSlide}}`)?.scrollIntoView({{block:'start', behavior:'auto'}});
      requestAnimationFrame(() => {{
        root.style.scrollBehavior = previousBehavior;
        root.style.scrollSnapType = previousSnap;
        updateCurrentFromViewport();
        const clean = new URL(location.href);
        clean.searchParams.delete('at');
        clean.hash = `slide-${{currentSlide}}`;
        history.replaceState(null, '', clean);
      }});
    }};
    requestAnimationFrame(placeInitialSlide);
    window.addEventListener('load', placeInitialSlide, {{ once:true }});
    checkForUpdate();
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
<p>Right-click rendered text for visual feedback, or open <strong>Source</strong> beside a slide to annotate exact Typst and formulas.</p>
<p>For fast recurring edits, run <code>python3 scripts/serve_typst_review.py L1</code>. It batches source changes and leaves browser reload under your control.</p>
<ul>{''.join(links)}</ul></body></html>'''
    atomic_write_text(output_dir / "index.html", index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("decks", nargs="*", help="Deck IDs such as L1 L2 L3A")
    parser.add_argument("--all", action="store_true", help="Build every PDF in slides-pdf/")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile each matching .typ source to slides-pdf/ before building HTML",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Rasterization resolution (180 dpi keeps full slides sharp on Retina review displays)",
    )
    parser.add_argument(
        "--keep-revisions",
        type=int,
        default=3,
        help="Keep this many recent render revisions so open lazy-loaded pages do not break",
    )
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
        with deck_lock(output_dir, deck_id):
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
                retain_revisions=max(1, args.keep_revisions),
            )
            text_regions = pdf_text_regions(pdf)
            if len(text_regions) != pages:
                raise RuntimeError(
                    f"Expected text for {pages} pages from {pdf}, extracted {len(text_regions)}"
                )
            sections = source_slide_sections(source, pages, title)
            revision = pdf_revision(pdf)
            review = page_html(
                deck_id,
                title,
                source,
                pdf,
                images,
                text_regions,
                sections,
                revision,
                output_dir,
            )
            destination = output_dir / f"{deck_id}.html"
            atomic_write_text(destination, review)
            status = {
                "deck": deck_id,
                "title": title,
                "revision": revision,
                "pages": pages,
                "dpi": args.dpi,
                "render_mode": render_mode,
                "rendered_pages": selected_pages or list(range(1, pages + 1)),
                "source": source.relative_to(REPO).as_posix() if source else None,
                "asset_dir": images[0].parent.relative_to(output_dir).as_posix(),
                "built_at": datetime.now(timezone.utc).isoformat(),
            }
            atomic_write_text(
                output_dir / f"{deck_id}.status.json",
                json.dumps(status, indent=2) + "\n",
            )
            page_note = (
                f"; pages {args.pages}" if render_mode == "incremental" else ""
            )
            print(
                f"{deck_id}: {pages} slides ({render_mode} render{page_note}) "
                f"-> {destination.relative_to(REPO) if destination.is_relative_to(REPO) else destination}"
            )

    build_index(output_dir)
    print("Serve with: python3 -m http.server 8765 --bind 127.0.0.1 --directory .")


if __name__ == "__main__":
    main()
