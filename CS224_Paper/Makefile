# Build the CS224R final report PDF from LaTeX.
#
# Uses Tectonic (https://tectonic-typesetting.github.io/), a self-contained
# LaTeX engine that fetches any required packages on demand — no system TeX
# installation needed. Install with:  brew install tectonic
#
# Usage:
#   make          # build the PDF
#   make clean    # remove the generated PDF

TEX    := cs224r_final_report_2026.tex
PDF    := $(TEX:.tex=.pdf)
ENGINE := tectonic

.PHONY: all clean

all: $(PDF)

$(PDF): $(TEX) reference.bib cs224r_2026.sty SummaryDiagram.png ModelComps.png
	$(ENGINE) $(TEX)

clean:
	rm -f $(PDF)
