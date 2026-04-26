#!/usr/bin/env python3
"""
Intensive accessibility-first review of a single lecture.

Usage:
    python scripts/review_lecture_intensive.py slides/lec01-why-dl-lecture.md

Different from `review_lecture.py` · this assumes the audience is COMPLETE
beginners who have NOT done ML before. Every math step must be worked out.
Every concept must be motivated.
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

from google import genai

REPO = Path(__file__).resolve().parent.parent
REVIEWS_DIR = REPO / "reviews-intensive"
REVIEWS_DIR.mkdir(exist_ok=True)

PROMPT = """You are reviewing one lecture from a 24-lecture deep learning course.
The instructor (Prof. Nipun Batra) is teaching this for the FIRST TIME and
just told me · "the slides are too COMPLEX. They assume too much. Students
will not understand. I need every math step worked out. Every concept needs
intuition first. More basics. More slides. More figures. Better flow."

Your job is to review this lecture and produce a CONCRETE rewrite plan.

AUDIENCE
The audience has taken an intro ML course (linear regression, logistic regression,
MSE, cross-entropy from MLE, plain gradient descent, basic probability + linear
algebra). They have NOT done deep learning · so backprop, deep architectures,
modern training tricks, anything Transformer/CNN/RNN-specific is NEW to them.

Don't re-derive things they already know (e.g., "what is MSE"). Do unpack
anything specifically deep · backprop, vanishing/exploding gradients, attention,
softmax temperature, BatchNorm internals, etc.

For every dense math slide, give me:

(a) A REWRITE with the math step-by-step. Replace any "derivation" with
    "let's compute term by term, with concrete numbers, then generalize."

(b) A SHORT INTUITION SLIDE (everyday analogy) to insert BEFORE the math.

(c) A WORKED NUMERIC EXAMPLE (with paste-ready numbers) to insert AFTER
    the math.

(d) A FIGURE/DIAGRAM IDEA to insert with the math.

For every conceptual slide, give me:

(e) Is the SETUP enough? Should we add a smaller-scope example FIRST?
    (E.g., before "vanishing gradients" do a 2-layer worked example
    that shows it concretely.)

(f) Is there JARGON that needs unpacking? List specific terms.

OUTPUT FORMAT
Section by section. For each problematic slide title, write:

```
## SLIDE · "<title>"

CURRENT PROBLEM
<short diagnosis · 1-2 sentences>

INSERT BEFORE
<intuition slide content · 1-2 short sentences plus an analogy>

REWRITE
<step-by-step math, EVERY substitution explicit, concrete numbers>

INSERT AFTER
<worked numeric example with explicit numbers>

FIGURE
<one-paragraph description of what to draw>
```

Aim for 8-15 problematic slides. Be VERY explicit · I will paste these
straight into the slides.

LECTURE FOLLOWS:
============================================================
"""


def review(lecture_path: Path, model: str = "gemini-2.5-pro") -> str:
    text = lecture_path.read_text()
    client = genai.Client()
    resp = client.models.generate_content(model=model, contents=PROMPT + text)
    return resp.text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="lecture .md file")
    ap.add_argument("--model", default="gemini-2.5-pro")
    args = ap.parse_args()
    f = Path(args.path)
    if not f.is_absolute():
        f = REPO / f
    out = REVIEWS_DIR / f"{f.stem}.intensive.md"
    print(f"reviewing {f.name} → {out.relative_to(REPO)}")
    text = review(f, model=args.model)
    out.write_text(text)
    print(text[:1500])
    print("...\n[truncated · full review saved]")


if __name__ == "__main__":
    main()
