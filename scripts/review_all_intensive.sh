#!/usr/bin/env bash
# Run intensive review on every lecture except L01/L02 (already done)
set -e
cd "$(dirname "$0")/.."
for f in slides/lec0[3-9]-*.md slides/lec1*.md slides/lec2*.md; do
  base="$(basename "$f" .md)"
  out="reviews-intensive/${base}.intensive.md"
  if [ -f "$out" ]; then
    echo "skip $base (already reviewed)"
    continue
  fi
  echo "reviewing $base ..."
  python3 scripts/review_lecture_intensive.py "$f" 2>&1 | tail -1
done
