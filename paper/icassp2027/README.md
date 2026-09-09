# ICASSP 2027 Submission Package

This directory contains the ICASSP-format manuscript for the Relational-Orbit TTRL study: four pages of main text followed by one references-only page.

## Files

- `main.tex`: identified, non-anonymous manuscript using the official `spconf.sty` format.
- `build/main.pdf`: compiled five-page Letter PDF; references begin on page 5.
- `references.bib`: bibliography database used with `IEEEbib.bst`.
- `spconf.sty`, `IEEEbib.bst`: official ICASSP paper-kit files.
- `figures/ro_pipeline.jpg`: supplied Relational-Orbit workflow schematic used as Fig. 1.

## Before upload

1. Confirm the final author order, affiliations, and email addresses in `main.tex`.
2. Replace the funding/compliance footnote with the final author-approved statement, or remove it when no statement is required.
3. Recompile and rename the PDF using the first author's surname, as requested by the submission system.
4. Ensure that the author order, affiliations, email addresses and ORCID records exactly match the online submission form.
5. Select the most appropriate ICASSP review category; the current scope is closest to machine learning and generative-AI methods, subject to the final category list.
6. Recheck arXiv-only technical reports for newer archival versions immediately before upload.

The manuscript intentionally reports effect sizes and sample sizes without numerical 95% confidence intervals in the main tables. Full uncertainty calculations and raw records remain in the repository supplementary outputs.
