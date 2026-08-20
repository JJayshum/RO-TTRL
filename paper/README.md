# Relational-Orbit paper

This directory contains the current full LaTeX manuscript and its compiled PDF.

## Files

- `main.tex`: manuscript source, including the complete experimental design, results, negative controls, statistical analysis, limitations, and experimental inventory.
- `references.bib`: BibTeX database.
- `build/main.pdf`: compiled manuscript.

## Compile

From the project root:

```bash
python3 /Users/jayshum/.codex/plugins/cache/openai-bundled/latex/0.2.4/scripts/compile_latex.py \
  "/Users/jayshum/Documents/Relational-Orbit TTRL/paper/main.tex" \
  --output-directory "/Users/jayshum/Documents/Relational-Orbit TTRL/paper/build"
```

The helper selects TeX Live because the manuscript uses BibTeX. The current build was produced with `latexmk` and `pdflatex` from TeX Live 2026.

## Evidence sources

The primary machine-readable analyses are:

- `../outputs/paper_scale_shared/paper_analysis.json`
- `../outputs/cross_family_panel_20260817/generality_analysis.json`

The broader experiment record is summarized in:

- `../PILOT_RESULTS.md`
- `../CROSS_FAMILY_GENERALITY_RESULTS.md`
- `../STATISTICAL_ANALYSIS.md`
- `../SECONDARY_ANALYSIS_PROTOCOL.md`

## Author input needed

Before submission, replace the author placeholder and add affiliations, correspondence details, funding, author contributions, competing-interest and ethics/declaration text as applicable. Verify the final publisher metadata for the Zheng et al. and Qwen technical-report references. A target venue has not yet been selected, so the source currently uses a generic article format.
