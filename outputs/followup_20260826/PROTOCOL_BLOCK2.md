# Follow-up block 2 (fixed before execution)

This block increases precision without outcome-based stopping. It adds one independent sample block to each of ARC-Challenge, OpenBookQA, and MMLU.

- Model: Qwen2.5-3B-Instruct
- Orbits: 64 per benchmark; views: 4; samples per view: 4
- Sample seed: 20260826; sample offset: 128
- Run seeds: 72808 (ARC), 72909 (OpenBookQA), 73010 (MMLU)
- Analysis set: all three original follow-up benchmarks plus this block; no selective reporting
- Stopping rule: complete all three benchmarks, then report the pooled paired-bootstrap 95% CI regardless of whether it excludes zero.
