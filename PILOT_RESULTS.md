# Relational-Orbit TTRL Pilot Results

Updated: 16 August 2026

## Decision

The project did not fail. Exact option-space alignment now has replicated inference-time support on ARC-Challenge, OpenBookQA, and MMLU at both 3B and 7B scales, using disjoint question blocks and matched generation compute. The policy-update evidence is weaker: it replicated on the easy synthetic family but a frozen real-benchmark adaptation diagnostic did not exclude zero. The defensible main contribution is therefore inference-time relational alignment, with TTRL adaptation as a secondary exploratory result rather than a general superiority claim.

## Audit and Invalid Early Runs

The initial negative 24-orbit run is excluded from scientific interpretation because the audit found four implementation/protocol defects:

1. The randomized gauge control applied the permutation in the wrong direction.
2. Random option permutations did not guarantee dispersion of every fixed-position shortcut.
3. Answer-only prompting suppressed reasoning; the first reasoning prompt then truncated most completions before the final answer.
4. The first adaptation diagnostic averaged completion log-probability by length instead of using the proposed full sequence log-probability.

The corrected implementation uses exact inverse gauges, four cyclic option rotations that disperse every fixed answer position across all four canonical candidates, compact numbered reasoning traces, final-answer parsing, full sequence policy gradients, leave-one-out rewards, within-prompt centering, and a KL penalty. Local and remote tests pass.

## Frozen Inference Pilot

Model: Qwen2.5-3B-Instruct in FP16 on one RTX 3090 24 GB. Each orbit used four exact option-rotation views and four rollouts per view. The root-only comparator used all 16 rollouts on the original prompt. The full answer domain, parse failure symbol, smoothing, prompts, temperature, and rollout counts were matched.

| Run | Orbits | Root only | Mapped pool | Difference | Paired 95% CI |
|---|---:|---:|---:|---:|---:|
| Confirmatory seed 808 | 64 | 0.547 | 0.813 | +0.266 | [0.156, 0.391] |
| Replication seed 909 | 64 | 0.578 | 0.828 | +0.250 | [0.125, 0.375] |
| Pooled descriptive result | 128 | 0.563 | 0.820 | +0.258 | [0.172, 0.344] |

Across 128 orbits, mapped pooling repaired 36 root-only errors and corrupted 3 root-only successes. Unmapped transformed-view accuracy was 0.289. Correct mapped accuracy exceeded the 20 randomized-gauge controls by 0.439. Truth-support coverage was 0.969 and 1.000 in the two runs; parse failure was about 2%.

## Frozen Adaptation Pilot

Each adaptation seed used 16 unlabeled training orbits and 512 sampled sequences per method. Mapped TTRL split them over four views; root TTRL spent the same sequence budget on the root. Both used LoRA rank 8, one on-policy pass, learning rate 1e-5, leave-one-out soft rewards, within-prompt reward centering, and KL coefficient 0.02. Evaluation used fresh, untouched orbits.

| Adaptation seed | Eval orbits | Base pass@1 | Root TTRL | Mapped TTRL | Mapped - root | Paired 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| Seed 1501 | 128 | 0.639 | 0.662 | 0.709 | +0.047 | [0.020, 0.076] |
| Seed 2501 | 64 | 0.555 | 0.590 | 0.629 | +0.039 | [-0.039, 0.113] |
| Seed 3501 | 64 | 0.469 | 0.488 | 0.574 | +0.086 | [0.027, 0.152] |
| Seed 4501 | 64 | 0.594 | 0.559 | 0.648 | +0.090 | [0.043, 0.145] |
| Seed 5501 | 64 | 0.504 | 0.496 | 0.527 | +0.031 | [-0.004, 0.074] |

Across all five seeds and 304 fresh evaluation orbits, the descriptive mapped-over-root effect is +0.057 (orbit-weighted). The equal-seed estimate is +0.0565; a hierarchical bootstrap that resamples adaptation seeds and then orbits gives 95% CI [+0.023, +0.091]. All five seeds are positive. Mapped-over-base is +0.0703 with hierarchical 95% CI [+0.034, +0.109]. Parse-failure rates remained low (about 0.3-4.3% per evaluation arm), truth-support coverage was 0.969-1.000, and training KL stayed finite and small.

## Transformation-Family Characterization

These 32-orbit inference checks were run with the same model and matched mapped/unmapped/randomized controls:

| Family | Root | Correct map | Difference | 95% CI | Correct map - randomized |
|---|---:|---:|---:|---:|---:|
| Boolean negation | 0.313 | 0.375 | +0.063 | [-0.063, +0.188] | -0.009 |
| Affine scale + shift | 0.594 | 0.531 | -0.063 | [-0.156, 0.000] | +0.236 |
| Affine shift-only | 0.875 | 0.688 | -0.188 | [-0.313, -0.063] | +0.295 |

Boolean is therefore retained as a negative control. Affine maps are semantically valid and beat randomized gauges, but mapped pooling hurts root-only accuracy on this task distribution; they are not positive evidence for the current method.

## Preregistered Harder Option-Family Gate

The medium-difficulty protocol was frozen on 16 August 2026 before inspecting results. It increased the arithmetic problems from 1-2 to 2-3 operations and used larger moduli. The inference stage used the same 3B model, four views, four samples per view, matched root compute, exact maps, parser, temperature, and token limit. It required pooled mapped-minus-root and correct-map-minus-randomized effects of at least +0.10, truth support of at least 0.80, and parse failure of at most 0.10 before adaptation could run.

| Seed | Orbits | Root | Mapped | Mapped - root | Correct map - randomized |
|---|---:|---:|---:|---:|---:|
| 6808 | 64 | 0.313 | 0.438 | +0.125 | +0.123 |
| 6909 | 64 | 0.328 | 0.313 | -0.016 | +0.072 |
| Pooled | 128 | 0.320 | 0.375 | +0.055 | +0.097 |

The pooled mapped-minus-root paired 95% interval is [-0.047, +0.156]. A seed-and-orbit hierarchical interval is [-0.078, +0.203]. The correct-map-minus-randomized interval is [+0.013, +0.183]. Truth support was 0.969 and parse failure was 0.022, so output truncation and parsing do not explain the failed effect-size gate.

Code checksums matched the successful implementation, remote option tests passed, and all four canonicalized views had similar rollout accuracy (0.295-0.342). Mapped pooling repaired 26 root errors but corrupted 19 root successes. A post-gate 32-orbit easy reference at seed 7808 ruled out environment drift: root accuracy was 0.594, mapped accuracy was 0.906, the difference was +0.313 with CI [+0.156, +0.469], and correct maps beat randomized gauges by +0.503. The medium failure is therefore interpreted as a genuine 3B capability boundary. The preregistered five-seed medium adaptation stage was not run.

## Public-Benchmark 3B Transfer Gate

The protocol was frozen before benchmark inference. It deterministically selected 64 eligible four-option questions each from ARC-Challenge validation, OpenBookQA validation, and MMLU test using content hashes. Four cyclic option views with four rollouts each were compared with 16 root rollouts. Twenty randomized gauges used the same transformed generations. Dataset fingerprints, source indices, normalized-content hashes, and mirrored Parquet hashes were recorded before inference.

| Benchmark | Questions | Root | Mapped | Difference | Paired 95% CI | Correct map - randomized |
|---|---:|---:|---:|---:|---:|---:|
| ARC-Challenge | 64 | 0.469 | 0.750 | +0.281 | [+0.156, +0.406] | +0.432 |
| OpenBookQA | 64 | 0.750 | 0.859 | +0.109 | [+0.047, +0.188] | +0.448 |
| MMLU | 64 | 0.391 | 0.531 | +0.141 | [+0.016, +0.266] | +0.189 |
| Pooled | 192 | 0.536 | 0.714 | +0.177 | [+0.109, +0.245] | +0.356 |

The pooled correct-map-minus-randomized 95% interval was [+0.291, +0.422]. Mapped pooling repaired 41 root errors and corrupted 7 root successes. Truth support was 0.995 and parse failure was 0.023. Every frozen transfer criterion passed.

## Disjoint-Sample 7B Confirmation

Qwen2.5-7B-Instruct was evaluated on the next 64 content-hash-ranked questions from each source, giving zero question overlap with the 3B block. Prompts, maps, generation counts, sampling settings, parser, and controls were unchanged.

| Benchmark | Questions | Root | Mapped | Difference | Paired 95% CI | Correct map - randomized |
|---|---:|---:|---:|---:|---:|---:|
| ARC-Challenge | 64 | 0.906 | 0.938 | +0.031 | [0.000, +0.078] | +0.488 |
| OpenBookQA | 64 | 0.813 | 0.875 | +0.063 | [0.000, +0.141] | +0.423 |
| MMLU | 64 | 0.672 | 0.734 | +0.063 | [0.000, +0.141] | +0.302 |
| Pooled | 192 | 0.797 | 0.849 | +0.052 | [+0.016, +0.089] | +0.404 |

The pooled correct-map-minus-randomized 95% interval was [+0.345, +0.463]. Mapped pooling repaired 12 root errors and corrupted 2 root successes. Truth support was 0.969 and parse failure was 0.019. All frozen 7B confirmation criteria passed despite the smaller ceiling-limited effect.

## Real-Benchmark Adaptation Diagnostic

A single frozen 3B LoRA diagnostic used 18 unlabeled training questions (six per benchmark, 576 sequences per method) and 96 untouched evaluation questions (32 per benchmark). These questions were disjoint from training and both inference blocks. Mapped and root TTRL used the same sequence budget, rank 8, one on-policy pass, learning rate 1e-5, and KL coefficient 0.02.

| Benchmark | Base pass@1 | Root TTRL | Mapped TTRL | Mapped - root | Paired 95% CI |
|---|---:|---:|---:|---:|---:|
| ARC-Challenge | 0.641 | 0.625 | 0.617 | -0.008 | [-0.070, +0.047] |
| OpenBookQA | 0.586 | 0.641 | 0.680 | +0.039 | [-0.008, +0.078] |
| MMLU | 0.430 | 0.398 | 0.430 | +0.031 | [-0.023, +0.086] |
| Pooled | 0.552 | 0.555 | 0.576 | +0.021 | [-0.010, +0.052] |

Mapped TTRL improved over the base by +0.023 with interval [-0.021, +0.070]. Training remained numerically stable: mean KL was 0.00023 for mapped and 0.00015 for root; mapped evaluation parse failure was 0.013. The diagnostic failed only because the primary interval included zero. In accordance with the frozen rule, no multi-seed real-benchmark adaptation expansion was run.

## Final Interpretation

The exact maps carry useful semantic information beyond transformed-prompt diversity. Correct mapped pooling beats root-only sampling, unmapped views, and randomized maps across three public benchmarks and two model scales. The smaller 7B gain is consistent with ceiling effects, while the strong map-control contrast remains. The failed Boolean, affine, and medium-synthetic controls still establish important boundaries: relational pooling is not automatically helpful for every transformation, and a near-chance policy can produce coherent wrong modes that pooling cannot reliably distinguish.

The defensible paper claim is now broader but more focused: exact relational alignment of option-permuted views improves matched-compute inference on public multiple-choice reasoning and knowledge benchmarks at 3B and 7B scales. Generality beyond option permutations is not established, and real-benchmark TTRL parameter updates are not yet confirmed. For a CCF-B submission, the inference mechanism should be the main method, supported by stronger published baselines, reliability/abstention analysis, and preferably a second model family; adaptation should be reported as preliminary rather than placed in the title or central claim.
