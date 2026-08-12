#!/usr/bin/env bash
# Build both compact handouts and progressive presentation PDFs from every
# Typst lecture deck into slides-pdf/, which GitHub Pages serves.
# Output name is derived from the deck's L-number prefix (L3a -> L3A).
# Usage:  ./build-slides-pdf.sh   (run from the repo root)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p slides-pdf
for deck in lecture*/L*.typ; do
  base=$(basename "$deck" .typ)          # e.g. L3a-calculus-toolkit
  num=${base%%-*}                        # e.g. L3a
  stem="slides-pdf/$(echo "$num" | tr '[:lower:]' '[:upper:]')"
  handout="${stem}.pdf"                       # slides-pdf/L3A.pdf
  presentation="${stem}-presentation.pdf"    # slides-pdf/L3A-presentation.pdf
  echo "  $deck -> $handout + $presentation"
  typst compile --root . --input handout=true "$deck" "$handout"
  typst compile --root . "$deck" "$presentation"
done
echo "done: $(ls slides-pdf/*.pdf | wc -l) PDFs in slides-pdf/"
