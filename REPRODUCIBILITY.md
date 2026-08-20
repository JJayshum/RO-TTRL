# Reproducibility Checklist

- Frozen protocols: `REAL_BENCHMARK_PROTOCOL.md`,
  `REAL_BENCHMARK_7B_PROTOCOL.md`, `REAL_BENCHMARK_ADAPTATION_PROTOCOL.md`,
  `SECOND_FAMILY_PROTOCOL.md`, `PAPER_SCALE_CONFIRMATION_PROTOCOL.md`.
- Compatibility audit: `LLAMA_COMPATIBILITY_AUDIT.md` and its JSON report.
- Primary and secondary statistical definitions:
  `STATISTICAL_ANALYSIS.md` and `SECONDARY_ANALYSIS_PROTOCOL.md`.
- Source datasets are mirrored parquet files with SHA-256 fingerprints recorded
  in every manifest.
- Every new shared-block record stores parsed view samples, root samples, raw
  completions, exact permutations, truth-support status, parse-failure rate,
  and content hash.
- The remote run is launched by `scripts/run_paper_scale_shared.sh` in a
  detached `screen` session. It can resume completed 64-question benchmark
  directories without changing seeds or exclusions.
- Before submission, rerun the complete test suite in a clean environment,
  archive `SHA256SUMS`, and publish code, frozen protocols, manifests, parsed
  records, and analysis outputs. Keep raw completions in an archival supplement
  if repository limits make the main release impractical.
