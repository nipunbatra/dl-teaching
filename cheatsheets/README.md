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

Build every available cheatsheet from the repository root:

```sh
./build-cheatsheets-pdf.sh
```

Before committing, confirm the page count with `pdfinfo`, render every page to
PNG for a visual check, and run `quarto render slides.qmd` to verify the public
link and copied PDF.
