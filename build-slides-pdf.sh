#!/usr/bin/env bash
# Build one-page-per-slide (handout) PDFs from every Typst lecture deck into
# slides-pdf/, which the Quarto site links and GitHub Pages serves.
# Output name is derived from the deck's L-number prefix (L3a -> L3A).
# Usage:  ./build-slides-pdf.sh   (run from the repo root)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p slides-pdf
for deck in lecture*/L*.typ; do
  base=$(basename "$deck" .typ)          # e.g. L3a-calculus-toolkit
  num=${base%%-*}                        # e.g. L3a
  out="slides-pdf/$(echo "$num" | tr '[:lower:]' '[:upper:]').pdf"   # slides-pdf/L3A.pdf
  echo "  $deck -> $out"
  typst compile --root . --input handout=true "$deck" "$out"
done
echo "done: $(ls slides-pdf/*.pdf | wc -l) PDFs in slides-pdf/"
