# Qwen2.5-7B precision replication (2026-08-27)

Purpose: improve precision of the cross-model Qwen2.5-7B estimate using a fixed, disjoint benchmark block.

- Model: Qwen2.5-7B-Instruct
- Benchmarks: ARC-Challenge, OpenBookQA, MMLU
- Orbits: 64 per benchmark; views: 4; samples per view: 4
- Sample seed: 20260827; sample offset: 0 after explicit content-hash exclusion of all prior Qwen2.5-7B blocks
- Run seeds: 82808 (ARC), 82909 (OpenBookQA), 83010 (MMLU)
- Analysis: report the completed block and pooled analysis with all existing 7B blocks
- Stopping rule: complete all three benchmarks regardless of whether the confidence interval excludes zero

## Execution amendment

The first ARC launch used offset 128 but omitted the explicit prior-manifest arguments. Its output is retained as a diagnostic and excluded from confirmation. Before inspecting any OpenBookQA or MMLU outcomes, the confirmatory launch was amended to pass all prior Qwen2.5-7B manifests explicitly. Because ARC has only 295 eligible items, offset 128 after excluding 128 prior items cannot supply 64 questions; the offset was therefore changed to zero after exclusion. Model, benchmark sizes, generation budget, sample seed, run seeds, analysis and stopping rule are unchanged.
