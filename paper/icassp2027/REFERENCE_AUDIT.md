# Reference audit

Audit date: 2 September 2026

## Consistency checks

- Unique citation keys in `main.tex`: 19
- Entries in `references.bib`: 19
- Items emitted in `build/main.bbl`: 19
- Missing citation keys: 0
- Unused bibliography entries: 0
- Duplicate DOI or arXiv identifiers: 0
- Undefined citations or references in the LaTeX log: 0

## Scope and placement

- Reasoning and test-time inference: chain-of-thought, zero-shot reasoning,
  self-consistency and test-time compute scaling.
- Decision bias and permutation robustness: contextual calibration,
  multiple-choice selection bias, option-order sensitivity and listwise
  permutation self-consistency.
- Implementation choices: LoRA, selective classification and nucleus sampling.
- Experimental assets: the six model families and three public benchmarks.

Every entry is cited next to the claim, method choice, model or dataset that it
supports. No reference was added only to enlarge the bibliography.

## Metadata corrections

- Added the previously missing citations for LoRA and Qwen2.5.
- Replaced the provisional Qwen2.5 key and year with the 2024 technical-report
  record associated with arXiv:2412.15115.
- Reclassified ARC as an arXiv article rather than a conference proceeding.
- Expanded author metadata for the selector-bias paper and the three benchmark
  or adaptation records where compact provisional data had been used.
- Removed internal “metadata to be verified” notes and redundant arXiv notes
  from papers that already have archival conference records.

DOIs and arXiv identifiers were checked against the project's existing
literature-discovery record and the cited archival metadata. Live registry
access was unavailable during this audit. Immediately before submission, check
the arXiv-only model reports for a newer archival version; this is a version
check, not an unresolved citation-key or manuscript-support issue.

## Build result

The official ICASSP style compiles to a five-page Letter PDF. Main text ends on
page 4 and all 19 references begin and end on page 5. The final build has no
overfull boxes, undefined citations, undefined references or LaTeX/BibTeX
errors.
