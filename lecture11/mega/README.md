# L11M: From Characters to Transformers

This directory contains the three modular parts of the Typst mega-lecture. The
single entry point is:

```text
lecture11/L11M-from-characters-to-transformers.typ
```

The slide-by-slide source of truth is:

```text
lecture11/L11M-from-characters-to-transformers-outline.md
```

## Build

Run from the repository root:

```bash
typst compile --root . --input handout=true \
  lecture11/L11M-from-characters-to-transformers.typ \
  slides-pdf/L11M.pdf

typst compile --root . \
  lecture11/L11M-from-characters-to-transformers.typ \
  slides-pdf/L11M-presentation.pdf
```

## Audit

```bash
python3 scripts/audit_typst_slides.py \
  --max-raster-images 18 \
  slides-pdf/L11M.pdf \
  slides-pdf/L11M-presentation.pdf
```

The deck uses the shared course theme in `common/metropolis.typ`, including its
fonts, margins, dark-teal title bar, orange accent, and bottom slide number.
The step-by-step drawings and numerical examples follow the original
handwritten next-token notes.

Technical diagrams are native Typst/Fletcher vectors. The deck also includes
generated illustrations, so pass `--max-raster-images` with the expected image
count when running the audit (currently 10 in the handout and 18 in the
presentation, including transparency masks).

## Lecture-plan PDF

Pandoc converts the canonical Markdown outline to Typst; Typst renders the PDF.
The table filter reserves most of each row for the explanation.

```bash
pandoc lecture11/L11M-from-characters-to-transformers-outline.md \
  --from=markdown+tex_math_single_backslash+tex_math_dollars \
  --pdf-engine=typst \
  --metadata-file=lecture11/mega/plan-layout.yaml \
  --lua-filter=lecture11/mega/plan-tables.lua \
  -o output/pdf/from-characters-to-transformers-lecture-plan.pdf
```

## Parts

- `lecture1.typ`: NLP tasks → `aabid` → tokenization → natural-language LM
- `lecture2.typ`: fixed-window failure → Q/K/V → self-attention → causality
- `lecture3.typ`: MHA → Transformer block → training → generation → synthesis
- `helpers.typ`: stable token, card, matrix, flow, and semantic-color helpers
