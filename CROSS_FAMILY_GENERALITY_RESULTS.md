# Cross-Family Generality Panel Results

Completed: 17 August 2026, 14:04:53 +08:00

## Decision

The three-candidate panel produced consistent positive point estimates and a
positive fixed-model average, but it did not pass the complete preregistered
generality gate. The sole failed criterion was the requirement that at least
one candidate independently achieve both a mapped-minus-root effect of at least
+0.03 and a paired 95% interval excluding zero.

This result supports broad average transfer across the fixed Phi, Mistral, and
Gemma checkpoints. It does not establish universal benefit or an individually
confirmed effect in every new family.

## Primary N=16 results

All models used the same 192-question paper-scale block, exact option maps, four
views, four samples per view, and sixteen matched root samples.

| Model | Root | Mapped | Difference | Paired 95% CI | Repairs | Corruptions |
|---|---:|---:|---:|---:|---:|---:|
| Phi-3.5-mini-Instruct | 0.802 | 0.823 | +0.021 | [-0.010, +0.052] | 7 | 3 |
| Mistral-7B-Instruct-v0.3 | 0.667 | 0.719 | +0.052 | [-0.005, +0.109] | 21 | 11 |
| Gemma-2-2B-it | 0.724 | 0.760 | +0.036 | [-0.010, +0.083] | 14 | 7 |
| Fixed-model average | - | - | **+0.036** | **[+0.009, +0.066]** | 42 | 21 |

The fixed-model correct-map-minus-randomized effect was +0.342 with paired 95%
CI [+0.279, +0.408]. This large control contrast shows that exact relational
maps, rather than transformed-prompt diversity alone, carried the signal.

## Frozen secondary compute scaling

The prespecified N=4/8/16 analysis shows that the benefit was strongest at
lower matched generation budgets for Phi and Mistral.

| Model | N=4 | N=8 | N=16 |
|---|---:|---:|---:|
| Phi-3.5-mini-Instruct | +0.052 [+0.010, +0.099] | +0.026 [-0.010, +0.063] | +0.021 [-0.010, +0.052] |
| Mistral-7B-Instruct-v0.3 | +0.083 [+0.021, +0.146] | +0.063 [+0.005, +0.120] | +0.052 [-0.005, +0.109] |
| Gemma-2-2B-it | +0.026 [-0.031, +0.083] | +0.026 [-0.026, +0.078] | +0.036 [-0.010, +0.083] |

N=16 remains the primary result. The N=4 and N=8 results are secondary and may
not be used to overwrite the failed primary gate, but they support the expected
interpretation that relational alignment is most useful under constrained
sampling compute and has diminishing headroom as root self-consistency grows.

## Benchmark pattern

- ARC-Challenge was positive for all candidates: Phi +0.063, Mistral +0.078,
  and Gemma +0.078. Phi's ARC interval excluded zero.
- OpenBookQA was positive for Mistral (+0.016) and Gemma (+0.063) but slightly
  negative for the high-accuracy Phi checkpoint (-0.016). Gemma's interval
  excluded zero.
- MMLU was positive for Phi (+0.016) and Mistral (+0.063) but negative for
  Gemma (-0.031). None of the MMLU candidate intervals excluded zero.

This pattern is consistent with task- and ceiling-dependent gains, not a claim
that pooling improves every model-benchmark pair.

## Reliability and audit

- Mean leave-one-view-out agreement was 0.957 for Phi, 0.897 for Mistral, and
  0.930 for Gemma.
- At 50% label-free coverage, mapped accuracy was 0.969 for Phi, 0.938 for
  Mistral, and 0.927 for Gemma.
- Truth support was 0.943-0.979 and parse failure was 0.005-0.010.
- All structural, disjointness, per-view, parser, and matched-token audits
  passed for all three candidates; reference overlap was zero.
- The remote test suite reported 25 passed. All output hashes passed on the
  originating host, and the downloaded local copy matches after removing the
  archived remote path prefix.

## Claim boundary

Together with the earlier shared-block Qwen2.5 and Llama results, the evidence
now spans six fixed checkpoints from five model families. The strongest honest
claim is that exact option-space alignment has a positive average effect across
a diverse fixed model panel, with heterogeneous per-model and per-benchmark
effects. Absolute universality and an individually significant effect in every
family remain unproven.
