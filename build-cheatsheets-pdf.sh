#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$repo_dir"

mkdir -p slides-pdf

found=0
for source_file in cheatsheets/L*.typ; do
  if [[ ! -f "$source_file" ]]; then
    continue
  fi

  found=1
  source_name="$(basename "$source_file" .typ)"
  lecture_id="${source_name%%-*}"
  output_file="slides-pdf/${lecture_id}-cheatsheet.pdf"
  echo "Building $output_file"
  typst compile --root . "$source_file" "$output_file"
done

if [[ "$found" -eq 0 ]]; then
  echo "No cheatsheet sources found in cheatsheets/."
fi
