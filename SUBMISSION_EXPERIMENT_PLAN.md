# Submission-Scale Experiment Plan

Goal: produce a complete, auditable CCF-B-ready empirical package centered on
inference-time relational option alignment.

## Completed evidence

- Corrected synthetic mechanism and five-seed easy-task adaptation.
- Negative Boolean, affine, and medium-difficulty boundary controls.
- Public-benchmark Qwen2.5-3B transfer gate.
- Disjoint-sample Qwen2.5-7B confirmation.
- Real-benchmark adaptation diagnostic; not advanced because its interval
  included zero.

## Remaining gates

1. Cross-family confirmation with Llama-3.2-3B-Instruct.
2. Paper-scale question expansion if cross-family confirmation passes.
3. Matched-compute N=4/8/16 scaling, hard-vote versus log-opinion pooling,
   randomized-map, unmapped-view, and root self-consistency baselines. The
   secondary definitions are frozen in `SECONDARY_ANALYSIS_PROTOCOL.md`.
4. Label-free stability and selective-risk analysis using the frozen
   leave-one-view-out agreement and probability-margin ordering, followed by a
   final untouched reliability block if the cross-family gate passes.
5. Paired and hierarchical uncertainty, multiplicity-aware secondary analyses,
   exact manifests, code tests, and reproducible result tables/figures.
6. Manuscript positioning: inference alignment as the main claim; TTRL
   adaptation explicitly secondary and preliminary.

No failed gate may be bypassed by tuning on its evaluation questions.
