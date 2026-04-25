#!/usr/bin/env python3
"""
Review a single interactive article with Gemini.

Usage:
    python scripts/review_interactive.py /path/to/article-dir
    python scripts/review_interactive.py --all      # all interactives

Outputs a punch list:
    i)   missing intuitions / scenarios / steps
    ii)  diagrams / interactive widgets to add
    iii) numeric examples to add
    iv)  flow / pacing
    v)   misconceptions / FAQ items missing
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

from google import genai

REPO = Path(__file__).resolve().parent.parent
ARTICLES_DIR = Path("/Users/nipun/git/interactive/src/articles")
REVIEWS_DIR = REPO / "reviews-interactives"
REVIEWS_DIR.mkdir(exist_ok=True)

PROMPT_TEMPLATE = """You are reviewing one INTERACTIVE EXPLAINER from a 24-lecture deep
learning course (ES 667, IIT Gandhinagar). The gold-standard reference is the
"Demystifying p-values" interactive · multi-step narrative, scenarios users
can switch between, manual sliders, full enumeration → shortcut comparisons,
worked numbers, misconception cards, bonus sections.

PRIORITIES
- Multi-step pedagogical narrative (Prelude → Step 1, 2, ...).
- Real scenarios users can swap between (not a single demo).
- Manual + automatic widgets (slider, click, "compute all" buttons).
- Concrete numbers shown as the user manipulates.
- Pause-and-think callouts.
- 2-3 common misconception cards.
- Bonus / "extra connections" section.
- Visual variety · multiple SVGs/canvases per page.

Return a CONCRETE punch list as five sections:

I)   STEPS / SCENARIOS THAT ARE MISSING
     - What's the current narrative arc?
     - What 2-3 steps or scenarios would make it richer?

II)  WIDGETS / DIAGRAMS TO ADD
     - For each: where, what it shows, what slider/click drives it.

III) NUMERIC EXAMPLES TO ADD
     - For each: where, what numbers to show, what insight it produces.

IV)  FLOW / PACING / NAMING
     - Misleading or confusing names?
     - Math too dense without intuition wrapper?
     - Sections to mark "advanced / optional"?

V)   MISCONCEPTIONS / FAQ
     - 2-3 common confusions students have on this topic.
     - Specific phrasing for each card.

Be specific · file paths, exact widget descriptions, exact numbers.
Skip a section if everything is already great.

INTERACTIVE FILES FOLLOW (HTML + JS + CSS combined):
====================================
"""


def collect_text(article_dir: Path) -> str:
    chunks = []
    for filename in ["index.html", "main.js", "styles.css", "meta.json"]:
        f = article_dir / filename
        if f.exists():
            chunks.append(f"\n\n========== {filename} ==========\n{f.read_text()}")
    return "".join(chunks)


def review(article_dir: Path, model: str = "gemini-2.5-pro") -> str:
    text = collect_text(article_dir)
    client = genai.Client()
    prompt = PROMPT_TEMPLATE + text
    resp = client.models.generate_content(model=model, contents=prompt)
    return resp.text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", help="path to article dir")
    ap.add_argument("--all", action="store_true", help="all interactives")
    ap.add_argument("--model", default="gemini-2.5-pro")
    args = ap.parse_args()

    if args.all:
        targets = sorted(d for d in ARTICLES_DIR.iterdir() if d.is_dir() and d.name not in {"shared", "vendor"})
    elif args.path:
        targets = [Path(args.path)]
    else:
        ap.error("provide a path or --all")

    for d in targets:
        if not d.is_absolute():
            d = ARTICLES_DIR / d.name
        out = REVIEWS_DIR / f"{d.name}.review.md"
        print(f"\n=== reviewing {d.name} → {out.relative_to(REPO)} ===")
        text = review(d, model=args.model)
        out.write_text(text)
        print(text[:1000])
        print("...\n[truncated · full review saved]")


if __name__ == "__main__":
    main()
