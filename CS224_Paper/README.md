# CS224R Final Report

LaTeX source for our CS224R final report on ARC-AGI-3 agents.

## Files

- `cs224r_final_report_2026.tex` — paper source
- `reference.bib` — bibliography
- `cs224r_2026.sty` — course style file
- `SummaryDiagram.png`, `ModelComps.png` — figures
- `cs224r_final_report_2026.pdf` — built PDF (committed to the repo)

## Building the PDF

The PDF is built with [Tectonic](https://tectonic-typesetting.github.io/), a
self-contained LaTeX engine that downloads any required packages on demand, so
no full TeX distribution is needed.

```sh
brew install tectonic   # one-time install (macOS)
make                     # builds cs224r_final_report_2026.pdf
```

To rebuild from scratch:

```sh
make clean && make
```
