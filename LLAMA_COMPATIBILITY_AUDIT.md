# Llama-3.2-3B Compatibility Audit

Audit completed: 16 August 2026, under the frozen checks in
`LLAMA_COMPATIBILITY_AUDIT_PROTOCOL.md`.

## Decision

No implementation incompatibility was found. The original cross-family gate
remains failed because its mapped-minus-root confidence interval included zero.
The result is interpreted as statistically uncertain rather than invalid.

## Structural and data checks

- All 192 records contained four valid permutations, four samples per view,
  sixteen root samples, exact content hashes, and complete manifests.
- The 192 confirmation hashes were unique and had zero overlap with the seven
  earlier Qwen inference/adaptation manifests supplied to the audit.
- Mapped completions used 234,727 tokens and root completions used 235,602;
  relative mismatch was 0.0037.
- Project tests passed on the remote environment.

## Template, length, and parser checks

- The installed Llama tokenizer exposed the official chat template, BOS and EOS
  tokens, a usable padding token, and a complete assistant generation header.
- Maximum rendered input lengths were 187 tokens for ARC-Challenge, 127 for
  OpenBookQA, and 435 for MMLU, below the 1,024-token truncation limit.
- Per-view parse-failure rates ranged from 0 to 0.0352, below the frozen 0.10
  threshold.

## View symmetry

Within-benchmark canonical rollout-accuracy ranges were 0.102 for
ARC-Challenge, 0.051 for MMLU, and 0.063 for OpenBookQA, all below the frozen
0.15 compatibility threshold. The four transformed views therefore showed no
large view-specific failure.

The machine-readable audit is stored at
`outputs/real_benchmark_llama3b_confirmation/compatibility_audit.json`.
