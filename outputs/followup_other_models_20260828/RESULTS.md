# Phi-3.5-mini and Mistral-7B replication results (2026-08-28)

This follow-up used the fixed protocol in
`../followup_other_models_20260828_PROTOCOL.md`: 64 previously unused
questions per model and benchmark, four views, four samples per view, and the
predeclared generation seeds. All six runs were completed irrespective of the
observed confidence intervals.

## Primary results

| Model | Benchmark | Root | Mapped | Difference | Paired-bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| Phi-3.5-mini-Instruct | ARC-Challenge | 87.50% | 92.19% | +4.69 pp | [-1.56, +10.94] pp |
| Phi-3.5-mini-Instruct | OpenBookQA | 82.81% | 84.38% | +1.56 pp | [-6.25, +9.38] pp |
| Phi-3.5-mini-Instruct | MMLU | 67.19% | 68.75% | +1.56 pp | [-7.81, +12.50] pp |
| **Phi-3.5 pooled** | **three benchmarks** | **79.17%** | **81.77%** | **+2.60 pp** | **[-2.08, +7.81] pp** |
| Mistral-7B-Instruct-v0.3 | ARC-Challenge | 76.56% | 82.81% | +6.25 pp | [-6.25, +18.75] pp |
| Mistral-7B-Instruct-v0.3 | OpenBookQA | 67.19% | 81.25% | +14.06 pp | [+4.69, +25.00] pp |
| Mistral-7B-Instruct-v0.3 | MMLU | 64.06% | 71.88% | +7.81 pp | [-1.56, +17.19] pp |
| **Mistral-7B pooled** | **three benchmarks** | **69.27%** | **78.65%** | **+9.38 pp** | **[+3.12, +15.62] pp** |
| **Both models, fixed-model average** | **192 shared questions; 384 model-question observations** | **74.22%** | **80.21%** | **+5.99 pp** | **[+1.82, +10.16] pp** |

Across the two models, OpenBookQA improved by +7.81 pp with a 95% CI of
[+0.78, +14.84] pp. The pooled ARC-Challenge and MMLU estimates were positive
but their individual intervals included zero.

The two-model overall interval uses paired question resampling stratified by
benchmark while treating models as fixed. The 192 question hashes are shared
across the two models; the 384 model-question rows are repeated observations,
not 384 independent questions. The earlier generic record-pooled analysis is
retained as an audit artifact but is not the inferential source for this row.

## Interpretation

The independent Mistral-7B block passes the prespecified confirmation gate:
its pooled mapped-minus-root effect exceeds 3 pp and its confidence interval
excludes zero. Phi-3.5 has positive point estimates on all three benchmarks,
but the pooled interval includes zero and therefore does not pass the same
gate. The evidence supports transfer to an additional model family, while also
showing meaningful model/task heterogeneity; it does not justify claiming a
uniform benefit for every checkpoint.

The correct-map minus randomized-map control is positive with a confidence
interval excluding zero for every model-benchmark cell. This makes a silent
mapping failure unlikely and supports interpreting the weaker Phi result as a
real limitation or ceiling/heterogeneity effect.

## Reproducibility notes

- Every final `records.jsonl` contains exactly 64 records and has a matching
  `manifest.json` and `summary.json`.
- `mistral7b/arc_challenge_incomplete_20260828` is retained as an audit artifact
  from an overlapping SSH retry. It contains 56 records and is excluded from
  every analysis. The clean fixed-seed rerun in `mistral7b/arc_challenge`
  contains 64 records and reproduces the previously emitted summary exactly.
- Model-specific and combined statistics are in each `analysis/summary.json`
  and in `combined_analysis/summary.json`.
- `SHA256SUMS` covers the complete downloaded result tree except itself.
