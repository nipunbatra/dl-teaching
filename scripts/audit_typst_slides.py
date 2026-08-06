#!/usr/bin/env python3
"""Audit rendered Typst slide PDFs for common visual-quality regressions.

The checks are intentionally dependency-free apart from Poppler's command-line
tools, which the review workflow already uses.  A raster image is treated as an
error by default; use ``--max-raster-images`` when a deck deliberately contains
a photograph or other raster artwork.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


PAGE_RE = re.compile(r"^Pages:\s+(\d+)$", re.MULTILINE)
SIZE_RE = re.compile(
    r"^Page size:\s+([0-9.]+)\s+x\s+([0-9.]+)\s+pts", re.MULTILINE
)
FONT_ROW_RE = re.compile(
    r"\s+(yes|no)\s+(yes|no)\s+(yes|no)\s+\d+\s+\d+\s*$", re.IGNORECASE
)


@dataclass
class AuditResult:
    path: Path
    pages: int
    width: float
    height: float
    raster_images: int
    fonts: int
    words: int
    smallest_word_height: float | None
    failures: list[str]


def run(*args: str) -> str:
    try:
        completed = subprocess.run(
            args,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"required command not found: {args[0]}") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"{' '.join(args)} failed: {detail}") from exc
    return completed.stdout.decode("utf-8", errors="replace")


def pdf_metadata(path: Path) -> tuple[int, float, float]:
    output = run("pdfinfo", str(path))
    page_match = PAGE_RE.search(output)
    size_match = SIZE_RE.search(output)
    if not page_match or not size_match:
        raise RuntimeError(f"could not read page count/size from {path}")
    return (
        int(page_match.group(1)),
        float(size_match.group(1)),
        float(size_match.group(2)),
    )


def raster_image_count(path: Path) -> int:
    output = run("pdfimages", "-list", str(path))
    return sum(
        1
        for line in output.splitlines()
        if re.match(r"^\s*\d+\s+\d+\s+", line)
    )


def embedded_font_status(path: Path) -> tuple[int, list[str]]:
    output = run("pdffonts", str(path))
    failures: list[str] = []
    count = 0
    for line in output.splitlines():
        match = FONT_ROW_RE.search(line)
        if not match:
            continue
        count += 1
        embedded = match.group(1).lower() == "yes"
        if not embedded:
            failures.append(f"font is not embedded: {line.strip()}")
    if count == 0:
        failures.append("no fonts were detected")
    return count, failures


def text_bounds(
    path: Path, edge_margin: float
) -> tuple[int, float | None, list[str]]:
    xml_text = run("pdftotext", "-bbox-layout", "-enc", "UTF-8", str(path), "-")
    root = ET.fromstring(xml_text)
    failures: list[str] = []
    word_count = 0
    smallest_height: float | None = None

    pages = [element for element in root.iter() if element.tag.endswith("}page")]
    for page_number, page in enumerate(pages, start=1):
        width = float(page.attrib["width"])
        height = float(page.attrib["height"])
        words = [element for element in page.iter() if element.tag.endswith("}word")]
        for word in words:
            word_count += 1
            xmin = float(word.attrib["xMin"])
            ymin = float(word.attrib["yMin"])
            xmax = float(word.attrib["xMax"])
            ymax = float(word.attrib["yMax"])
            word_height = ymax - ymin
            smallest_height = (
                word_height
                if smallest_height is None
                else min(smallest_height, word_height)
            )
            if (
                xmin < -edge_margin
                or ymin < -edge_margin
                or xmax > width + edge_margin
                or ymax > height + edge_margin
            ):
                text = "".join(word.itertext()).strip()
                failures.append(
                    f"page {page_number}: text outside page bounds: {text!r} "
                    f"({xmin:.2f}, {ymin:.2f}, {xmax:.2f}, {ymax:.2f})"
                )
    return word_count, smallest_height, failures


def audit(
    path: Path,
    *,
    max_raster_images: int,
    edge_margin: float,
    aspect_tolerance: float,
    min_word_height: float,
) -> AuditResult:
    pages, width, height = pdf_metadata(path)
    failures: list[str] = []

    aspect = width / height
    if abs(aspect - (16 / 9)) > aspect_tolerance:
        failures.append(f"page aspect ratio is {aspect:.4f}, expected 16:9")

    raster_images = raster_image_count(path)
    if raster_images > max_raster_images:
        failures.append(
            f"contains {raster_images} raster images; allowed {max_raster_images}"
        )

    fonts, font_failures = embedded_font_status(path)
    failures.extend(font_failures)
    words, smallest_word_height, bounds_failures = text_bounds(path, edge_margin)
    failures.extend(bounds_failures)
    if smallest_word_height is not None and smallest_word_height < min_word_height:
        failures.append(
            f"smallest extracted word box is {smallest_word_height:.1f}pt; "
            f"minimum is {min_word_height:.1f}pt"
        )

    return AuditResult(
        path=path,
        pages=pages,
        width=width,
        height=height,
        raster_images=raster_images,
        fonts=fonts,
        words=words,
        smallest_word_height=smallest_word_height,
        failures=failures,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdfs", nargs="+", type=Path)
    parser.add_argument(
        "--max-raster-images",
        type=int,
        default=0,
        help="permit this many deliberate raster images per PDF (default: 0)",
    )
    parser.add_argument(
        "--edge-margin",
        type=float,
        default=1.5,
        help="point tolerance for text bounds (default: 1.5)",
    )
    parser.add_argument(
        "--aspect-tolerance",
        type=float,
        default=0.01,
        help="absolute tolerance around a 16:9 aspect ratio (default: 0.01)",
    )
    parser.add_argument(
        "--min-word-height",
        type=float,
        default=6.0,
        help="minimum extracted word-box height in points (default: 6.0)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results: list[AuditResult] = []
    for path in args.pdfs:
        if not path.is_file():
            print(f"FAIL {path}: file not found", file=sys.stderr)
            return 1
        try:
            results.append(
                audit(
                    path,
                    max_raster_images=args.max_raster_images,
                    edge_margin=args.edge_margin,
                    aspect_tolerance=args.aspect_tolerance,
                    min_word_height=args.min_word_height,
                )
            )
        except (RuntimeError, ET.ParseError) as exc:
            print(f"FAIL {path}: {exc}", file=sys.stderr)
            return 1

    failed = False
    for result in results:
        status = "PASS" if not result.failures else "FAIL"
        smallest = (
            f"{result.smallest_word_height:.1f}pt"
            if result.smallest_word_height is not None
            else "n/a"
        )
        print(
            f"{status} {result.path}: {result.pages} pages, "
            f"{result.width:.1f}x{result.height:.1f}pt, "
            f"{result.raster_images} raster images, {result.fonts} embedded fonts, "
            f"{result.words} words (smallest extracted glyph box {smallest})"
        )
        for failure in result.failures:
            failed = True
            print(f"  - {failure}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
