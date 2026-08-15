PDF=main.pdf
TEX=main.tex

.PHONY: all clean

all: $(PDF)

$(PDF): $(TEX) appendix.tex refs.bib
	@if command -v latexmk >/dev/null 2>&1; then \
		latexmk -pdf $(TEX); \
	elif command -v pdflatex >/dev/null 2>&1; then \
		pdflatex $(TEX); \
		bibtex main; \
		pdflatex $(TEX); \
		pdflatex $(TEX); \
	elif command -v tectonic >/dev/null 2>&1; then \
		tectonic $(TEX); \
	else \
		echo "No TeX engine found. Install latexmk/pdflatex or tectonic." >&2; \
		exit 1; \
	fi

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.xdv *.synctex.gz $(PDF)
