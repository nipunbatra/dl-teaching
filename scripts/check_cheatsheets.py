"""Check and render the twelve public A4 cheatsheets (requires Poppler/Pillow/pypdf).

Run with the Codex bundled Python, or a Python with Pillow and pypdf installed.
QA images and a JSON report go to work/cheatsheets-a4-review/ by default.
"""
from pathlib import Path
import argparse
import concurrent.futures
import json
import subprocess
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
IDS = [f"S{i:02d}" for i in range(1, 11)] + ["P01", "P02"]
NS = {"h": "http://www.w3.org/1999/xhtml"}


def check(identifier, output):
    source = ROOT / "slides-pdf" / f"{identifier}-cheatsheet.pdf"
    pdf = PdfReader(source)
    assert 1 <= len(pdf.pages) <= 2, f"{identifier}: {len(pdf.pages)} pages"
    bbox = ET.fromstring(subprocess.check_output(["pdftotext", "-bbox", str(source), "-"]))
    results = []
    for index, page in enumerate(bbox.findall(".//h:page", NS)):
        width, height = float(page.attrib["width"]), float(page.attrib["height"])
        assert abs(width - 595.276) < 1 and abs(height - 841.89) < 1, identifier
        words = page.findall(".//h:word", NS)
        assert len(words) > 100, f"{identifier}: nearly empty page {index + 1}"
        outside = [w.text for w in words if float(w.attrib["xMin"]) < 15
                   or float(w.attrib["xMax"]) > width - 15
                   or float(w.attrib["yMax"]) > height - 8]
        assert not outside, f"{identifier}: text outside print margins: {outside}"
        body = [w for w in words if float(w.attrib["yMin"]) < 800]
        bottom = max(float(w.attrib["yMax"]) for w in body)
        results.append({"page": index + 1, "words": len(words), "body_bottom_pt": round(bottom, 1)})
    subprocess.run(["pdftoppm", "-scale-to", "1600", "-png", str(source), str(output / identifier)], check=True)
    return {"id": identifier, "pages": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "work" / "cheatsheets-a4-review")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda identifier: check(identifier, args.output), IDS))
    # Four pages per contact sheet; keep full-resolution individual renders too.
    pictures = [args.output / f"{item['id']}-{page['page']}.png"
                for item in results for page in item["pages"]]
    for offset in range(0, len(pictures), 4):
        canvas = Image.new("RGB", (1200, 1760), "#dfe5e6")
        draw = ImageDraw.Draw(canvas)
        for index, path in enumerate(pictures[offset:offset + 4]):
            with Image.open(path) as original:
                original.thumbnail((580, 825))
                x, y = (index % 2) * 600 + 10, (index // 2) * 880 + 35
                canvas.paste(original, (x, y))
                draw.text((x, y - 23), path.stem, fill="#183138")
        canvas.save(args.output / f"contact-{offset // 4 + 1}.png")
    (args.output / "report.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
