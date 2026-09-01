# Lecture cheatsheets

Cheatsheets are concise, printable companions to the canonical lecture decks.
Keep each one to one or two A4 pages and prioritize decisions, equations, and
operational checks over a compressed copy of every slide.

## Convention

- Source: `cheatsheets/L<id>-<topic>.typ`
- Shared layout and palette: `common/cheatsheet.typ`
- Published PDF: `slides-pdf/L<id>-cheatsheet.pdf`
- Public link: add `Cheatsheet` to the matching lecture row in `slides.qmd`

Build every available cheatsheet from the repository root:

```sh
./build-cheatsheets-pdf.sh
```

Before committing, confirm the page count with `pdfinfo`, render every page to
PNG for a visual check, and run `quarto render slides.qmd` to verify the public
link and copied PDF.
