# Relational-Orbit TTRL

Relational-Orbit aligns answers from option-permuted multiple-choice prompts in
a shared canonical answer space before aggregation. The repository contains the
inference and TTRL adaptation code, frozen experiment protocols, tests,
machine-readable results, audit artifacts, and the LaTeX paper source.

The supported claim is deliberately bounded: exact option-space alignment
improves matched-compute multiple-choice inference on average across a fixed,
diverse panel of language-model families. Effects remain model- and
task-dependent, and real-benchmark TTRL adaptation is exploratory.

## Main results

| Experiment | Mapped minus root | Paired 95% interval | Decision |
|---|---:|---:|---|
| Qwen2.5-3B public benchmarks | +17.71 pp | [+10.94, +24.48] | Passed |
| Qwen2.5-7B disjoint confirmation | +5.21 pp | [+1.56, +8.85] | Passed |
| Qwen/Llama shared-block fixed-model average | +3.13 pp | [+0.17, +5.90] | Passed |
| Phi/Mistral/Gemma fixed-panel average | +3.65 pp | [+0.87, +6.60] | Positive average; strict individual gate not passed |
| Real-benchmark mapped versus root TTRL | +2.08 pp | [-1.04, +5.21] | Inconclusive |

The cross-family evidence spans six fixed checkpoints from five model
families. It does not establish universal improvement for every checkpoint,
benchmark, transformation, or sampling budget.

## Install and test

```bash
python -m pip install -e .
python -m pytest
ro-pilot --orbits 1000 --rollouts 64
```

The local test suite contains 30 tests and does not require a model download.

## Repository layout

- `relational_orbit_ttrl/`: inference, adaptation, compatibility-audit, and
  statistical-analysis implementation.
- `tests/`: CPU-only unit and analysis tests.
- `scripts/`: frozen paper-scale and cross-family experiment drivers.
- `outputs/paper_scale_shared/`: Qwen2.5/Llama shared-block records and analysis.
- `outputs/cross_family_panel_20260817/`: Phi/Mistral/Gemma panel records and analysis.
- `PILOT_RESULTS.md`: complete staged experiment history and decisions.
- `CROSS_FAMILY_GENERALITY_RESULTS.md`: final cross-family panel report.
- `STATISTICAL_ANALYSIS.md`: estimands, bootstrap design, and reporting rules.
- `paper/`: LaTeX manuscript source and bibliography.

## Model-backed inference

The bounded GPU pilot uses executable modular-arithmetic multiple-choice tasks
and exact option permutations. Root-only and mapped pooling receive identical
generation counts; randomized maps are evaluated from the same transformed
samples.

```bash
python -m relational_orbit_ttrl.model_pilot --orbits 24 --views 4 --samples 4
```

See the frozen protocol Markdown files before running the public-benchmark or
cross-family experiments. The two top-level shell scripts document the exact
paper-scale execution order.

## Result artifacts

The repository retains source records, summaries, manifests, environment
descriptions, hashes, and analysis JSON needed to audit the reported results.
Model checkpoints, LoRA adapter weights, caches, packed duplicate archives, and
LaTeX intermediate files are excluded from Git. Dataset snapshots and base
model weights must be obtained separately under their respective licenses.
