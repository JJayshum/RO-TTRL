# Llama Compatibility Audit Protocol

Protocol frozen: 16 August 2026, after the preregistered Llama-3.2-3B
confirmation failed only because the pooled mapped-minus-root interval included
zero, and before compatibility diagnostics were inspected.

## Purpose

Determine whether the failed cross-family gate is attributable to an
implementation incompatibility rather than insufficient statistical evidence.
This audit cannot change the original gate decision and cannot tune prompts,
sampling settings, pooling, parsing, or thresholds on the 192 confirmation
questions.

## Checks

1. Verify that all confirmation manifests use the frozen offsets, seeds,
   dataset hashes, and mutually disjoint content hashes.
2. Verify four valid option permutations, four samples per transformed view,
   sixteen root samples, exact inverse maps, and complete question records.
3. Report parse failure, emitted answer-position frequencies, and canonical
   rollout accuracy separately for every transformed view. Flag a view only if
   its parse-failure rate exceeds 0.10 or if its canonical accuracy differs from
   the other views by more than 0.15.
4. Verify the installed Llama tokenizer has a chat template, a generation
   prompt, defined BOS/EOS tokens, and a usable padding token after the same
   fallback used by inference.
5. Compare mapped and root completion-token totals. Flag a compute mismatch if
   their absolute difference exceeds 10% of the larger total.
6. Re-run the project test suite without changing the implementation.

## Decision

An implementation incompatibility is established only by a failed structural,
template, parser, mapping, or compute-budget check. A small or statistically
uncertain mapped-minus-root effect is not itself an incompatibility.

If no incompatibility is found, retain the failed cross-family gate and pivot
the paper claim to the replicated Qwen2.5 3B/7B result. Any later independent
confirmation must use a newly frozen, disjoint block and must report this failed
gate.
