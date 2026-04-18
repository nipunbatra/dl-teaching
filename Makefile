.PHONY: all clean help list dirs diagrams

# Directories
SLIDES_DIR := slides
PDF_DIR := pdf
HTML_DIR := html
DIAGRAMS_DIR := diagrams
FIGURES_DIR := figures

# Theme (anthropic — the official theme going forward)
THEME := $(SLIDES_DIR)/anthropic-theme.css

# Find all lecture .md files in slides/
SLIDES_MD := $(wildcard $(SLIDES_DIR)/*.md)

# Extract lecture numbers (e.g., lec01, lec02)
LECTURES := $(sort $(patsubst $(SLIDES_DIR)/%-lecture.md,%,$(filter %-lecture.md,$(SLIDES_MD))))

# Define output files
PDF_TARGETS := $(patsubst $(SLIDES_DIR)/%.md, $(PDF_DIR)/%.pdf, $(SLIDES_MD))
HTML_TARGETS := $(patsubst $(SLIDES_DIR)/%.md, $(HTML_DIR)/%.html, $(SLIDES_MD))

# Default target
all: diagrams dirs $(PDF_TARGETS) $(HTML_TARGETS)
	@echo "Done: all slides built"

# Convenience: build HTML for the first 5 lectures
first5: dirs
	@echo "Building HTML for L1-L5..."
	@for n in 01 02 03 04 05; do \
		for f in $(SLIDES_DIR)/lec$$n-*.md; do \
			if [ -f "$$f" ]; then \
				name=$$(basename "$$f" .md); \
				echo "  $$f -> $(HTML_DIR)/$$name.html"; \
				npx marp "$$f" -o "$(HTML_DIR)/$$name.html" --html --allow-local-files --theme-set $(THEME); \
			fi \
		done \
	done
	@echo "Done: L1-L5 HTML"

# Convenience: build HTML for the first 7 lectures (merged reg as L6, CNN as L7)
first8: dirs
	@echo "Building HTML for L1-L8..."
	@for n in 01 02 03 04 05 06 07 08; do \
		for f in $(SLIDES_DIR)/lec$$n-*.md; do \
			if [ -f "$$f" ]; then \
				name=$$(basename "$$f" .md); \
				echo "  $$f -> $(HTML_DIR)/$$name.html"; \
				npx marp "$$f" -o "$(HTML_DIR)/$$name.html" --html --allow-local-files --theme-set $(THEME); \
			fi \
		done \
	done
	@echo "Done: L1-L8 HTML"

first7: dirs
	@echo "Building HTML for L1-L7..."
	@for n in 01 02 03 04 05 06 07; do \
		for f in $(SLIDES_DIR)/lec$$n-*.md; do \
			if [ -f "$$f" ]; then \
				name=$$(basename "$$f" .md); \
				echo "  $$f -> $(HTML_DIR)/$$name.html"; \
				npx marp "$$f" -o "$(HTML_DIR)/$$name.html" --html --allow-local-files --theme-set $(THEME); \
			fi \
		done \
	done
	@echo "Done: L1-L7 HTML"

# Convenience: build HTML for the first 6 lectures (L6 merged regularization)
first6: dirs
	@echo "Building HTML for L1-L6..."
	@for n in 01 02 03 04 05 06; do \
		for f in $(SLIDES_DIR)/lec$$n-*.md; do \
			if [ -f "$$f" ]; then \
				name=$$(basename "$$f" .md); \
				echo "  $$f -> $(HTML_DIR)/$$name.html"; \
				npx marp "$$f" -o "$(HTML_DIR)/$$name.html" --html --allow-local-files --theme-set $(THEME); \
			fi \
		done \
	done
	@echo "Done: L1-L6 HTML"

# Create output directories and copy images
dirs:
	@mkdir -p $(PDF_DIR)
	@mkdir -p $(HTML_DIR)
	@if [ -d "$(SLIDES_DIR)/images" ]; then \
		cp -r $(SLIDES_DIR)/images $(HTML_DIR)/; \
	fi
	@if [ -d "$(FIGURES_DIR)" ]; then \
		cp -r $(FIGURES_DIR) $(HTML_DIR)/; \
	fi

# Generate diagrams
diagrams:
	@echo "Generating diagrams..."
	@python $(DIAGRAMS_DIR)/generate_all.py

# Pattern rule for PDF
$(PDF_DIR)/%.pdf: $(SLIDES_DIR)/%.md $(THEME) | dirs
	@echo "Building PDF: $< -> $@"
	@npx marp $< -o $@ --pdf --allow-local-files --theme-set $(THEME) --html || echo "  (PDF skipped - needs Chrome)"

# Pattern rule for HTML
$(HTML_DIR)/%.html: $(SLIDES_DIR)/%.md $(THEME) | dirs
	@echo "Building HTML: $< -> $@"
	@npx marp $< -o $@ --html --allow-local-files --theme-set $(THEME)

# HTML only for a lecture (faster) - must come before lec% rule
lec%-html: dirs
	@echo "Building lec$* HTML only..."
	@for f in $(SLIDES_DIR)/lec$*-*.md; do \
		if [ -f "$$f" ]; then \
			name=$$(basename "$$f" .md); \
			echo "  $$f -> $(HTML_DIR)/$$name.html"; \
			npx marp "$$f" -o "$(HTML_DIR)/$$name.html" --html --allow-local-files --theme-set $(THEME); \
		fi \
	done
	@echo "Done: lec$* HTML"

# Build specific lecture (e.g., make lec01)
lec%: dirs
	@echo "Building lec$* slides..."
	@for f in $(SLIDES_DIR)/lec$*-*.md; do \
		if [ -f "$$f" ]; then \
			name=$$(basename "$$f" .md); \
			echo "  HTML: $$f -> $(HTML_DIR)/$$name.html"; \
			npx marp "$$f" -o "$(HTML_DIR)/$$name.html" --html --allow-local-files --theme-set $(THEME); \
			echo "  PDF:  $$f -> $(PDF_DIR)/$$name.pdf"; \
			npx marp "$$f" -o "$(PDF_DIR)/$$name.pdf" --pdf --allow-local-files --theme-set $(THEME) 2>/dev/null || echo "  (PDF skipped - needs Chrome)"; \
		fi \
	done
	@echo "Done: lec$*"

# List available slides
list:
	@echo "Available slides:"
	@for file in $(SLIDES_MD); do \
		echo "  - $$file"; \
	done
	@echo ""
	@echo "Lectures: $(LECTURES)"

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	@rm -rf $(PDF_DIR) $(HTML_DIR)
	@echo "Done: clean"

help:
	@echo "Usage:"
	@echo "  make lec01       # Build lecture 01 HTML + PDF"
	@echo "  make lec01-html  # Build lecture 01 HTML only (fast)"
	@echo "  make all         # Build everything"
	@echo "  make diagrams    # Regenerate all diagrams"
	@echo "  make list        # List available slides"
	@echo "  make clean       # Remove generated files"
	@echo ""
	@echo "Available lectures: $(LECTURES)"
