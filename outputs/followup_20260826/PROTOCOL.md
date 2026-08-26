# Follow-up experiment (2026-08-26)

Purpose: test external reproducibility on a disjoint benchmark sample without changing the estimator or tuning to outcomes.

- Model: Qwen2.5-3B-Instruct (cached snapshot on the experiment server)
- Benchmarks: ARC-Challenge, OpenBookQA, MMLU
- Orbits: 64 per benchmark
- Views: 4 cyclic option permutations
- Samples per view: 4
- Sample seed: 20260826
- Sample offset: 64 (disjoint from the primary block, subject to benchmark availability)
- Run seeds: 71808 (ARC), 71909 (OpenBookQA), 72010 (MMLU)
- Controls: root, unmapped, mapped, and randomized gauge (the standard pilot outputs)
- Analysis: paired bootstrap confidence intervals and the repository submission analysis script

The run is exploratory replication evidence. It is not used to retune the method or alter the preregistered primary claims.
