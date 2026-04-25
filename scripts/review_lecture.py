#!/usr/bin/env python3
"""
Review a single lecture's Marp file with Gemini.

Usage:
    python scripts/review_lecture.py slides/lec06-regularization-lecture.md
    python scripts/review_lecture.py --all     # all lectures, save to reviews/

Outputs a structured punch list:
    i)   intuition to add
    ii)  images / diagrams to create
    iii) worked examples to add
    iv)  overall improvements
"""
from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path

from google import genai

REPO = Path(__file__).resolve().parent.parent
REVIEWS_DIR = REPO / "reviews"
REVIEWS_DIR.mkdir(exist_ok=True)

PROMPT_TEMPLATE = """You are reviewing one lecture from a 24-lecture deep learning course
(ES 667, IIT Gandhinagar) being taught for the FIRST TIME by Prof. Nipun Batra.

PRIORITIES (in order)
- Accessible for first-time teacher / first-time students.
- Intuition FIRST, then worked numeric examples, THEN formal math.
- ~50 slides per lecture for an 80-90 minute class.
- Many figures, light on math, concrete numeric worked examples.
- Keep what's strong; only suggest changes that meaningfully help.

Give me a CONCRETE punch list for this lecture. Be specific about
slide titles and the exact numbers I should paste.

Return four sections, exactly:

I) INTUITION TO ADD
   For each item:
   - Which slide title to insert BEFORE
   - 2-4 sentences of intuitive framing (analogy, motivation, "why care")

II) DIAGRAMS / IMAGES TO CREATE
   For each item:
   - Slide title where it would be inserted
   - Description of what to draw (boxes, axes, key labels)
   - Why it would help

III) WORKED NUMERIC EXAMPLES TO ADD
   For each item:
   - Slide title where it would be inserted
   - Setup with explicit input numbers
   - Step-by-step calculation (the actual values)
   - The takeaway in one line

IV) OVERALL IMPROVEMENTS
   - Anything to cut as too advanced for first-timers.
   - Flow / pacing issues.
   - Missing notebook ideas (1-2 with a brief outline).
   - Any "this should be marked optional" notes.

Aim for 4-8 items per section. Skip a section if everything is already great.

LECTURE FILE FOLLOWS:
====================================
"""


def review(lecture_path: Path, model: str = "gemini-2.5-pro") -> str:
    text = lecture_path.read_text()
    client = genai.Client()
    prompt = PROMPT_TEMPLATE + text
    resp = client.models.generate_content(model=model, contents=prompt)
    return resp.text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", help="lecture .md file")
    ap.add_argument("--all", action="store_true", help="review every slides/lec*.md")
    ap.add_argument("--model", default="gemini-2.5-pro")
    args = ap.parse_args()

    if args.all:
        targets = sorted((REPO / "slides").glob("lec*.md"))
    elif args.path:
        targets = [Path(args.path)]
    else:
        ap.error("provide a path or --all")

    for f in targets:
        f = f if f.is_absolute() else REPO / f
        out = REVIEWS_DIR / f"{f.stem}.review.md"
        print(f"\n=== reviewing {f.name} → {out.relative_to(REPO)} ===")
        text = review(f, model=args.model)
        out.write_text(text)
        print(text[:1200])
        print("...\n[truncated · full review saved]")


if __name__ == "__main__":
    main()
