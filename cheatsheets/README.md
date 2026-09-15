# Lecture cheatsheets

Cheatsheets are concise, printable companions to the canonical lecture decks.
Keep each one to one or two A4 pages and prioritize decisions, equations, and
operational checks over a compressed copy of every slide.

## Convention

- Public-sequence source: `cheatsheets/S<nn>-<topic>.typ`
- Shared layout and palette: `common/cheatsheet.typ`
- Published PDF: `slides-pdf/S<nn>-cheatsheet.pdf`
- Public link: add `Cheatsheet` to the matching lecture row in `slides.qmd`

The `S` number is the student-facing lecture number. Legacy `L` files retain
their stable deck-era names when needed for old links.

The attention series uses `P01-next-token-prediction.typ` and
`P02-self-attention.typ`, published as `P01-cheatsheet.pdf` and
`P02-cheatsheet.pdf`. These part numbers do not change the course lecture numbering.

Use readable body type (typically 10-12 pt), and balance each page individually.
Keep the content within two A4 pages without shrinking it into the upper half.
Preserve equations and worked examples; do not add filler merely to fill space.

Build every available cheatsheet from the repository root:

```sh
./build-cheatsheets-pdf.sh
```

Before committing, confirm the page count with `pdfinfo`, render every page to
PNG for a visual check, and run `quarto render slides.qmd` to verify the public
link and copied PDF.
