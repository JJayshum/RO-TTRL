# Statistical Analysis Plan

## Independent unit and replication

For inference-time benchmark experiments, the independent unit is a question
block. A question is evaluated repeatedly across transformed prompts and model
conditions; these are repeated observations, not independent questions. The
shared-block study contains 192 unique questions (64 per benchmark) and 576
model-question evaluations across three fixed model checkpoints. The earlier
Qwen and Llama blocks are reported as separate disjoint question blocks.

## Primary estimands

For each question, the primary paired contrast is mapped pooled correctness
minus matched-compute root-only correctness. The map-control contrast is mapped
correctness minus correctness under randomized inverse gauges generated from the
same transformed samples. Effects are reported as percentage-point differences,
with paired 95% bootstrap intervals.

The shared-block primary interval resamples questions with replacement within
each benchmark and applies the same sampled question indices to all three model
conditions. The reported overall effect is the arithmetic mean of the three
fixed-model effects. This preserves repeated-question dependence and does not
claim that the three checkpoints are a random sample from all language models.

## Secondary analyses

Stored generations are used for N=4, 8 and 16 compute scaling, log-opinion
pooling versus hard vote, unmapped-view and randomized-map controls, repair and
corruption counts, leave-one-view-out stability, and fixed 25/50/75/100%
risk-coverage descriptions. These analyses are labelled secondary and are not
used to override a prespecified gate.

## Multiplicity and reporting

The shared-block paper decision uses conjunctive criteria for the fixed-model
effect and map-control effect; per-model intervals and benchmark contrasts are
descriptive. No p-value threshold is used as a substitute for effect size and
uncertainty. All question-selection seeds, generation seeds, manifests,
content hashes, model paths, raw completions, parsed samples, and analysis
seeds are retained. No question is excluded after inspecting its correctness.

## Software and numerical checks

Analysis uses Python, NumPy, and the project test suite; exact package versions
are recorded in each run's `environment.txt`. Tests are run with third-party
pytest plugin autoload disabled when the host environment contains unrelated
plugin dependency conflicts. The project suite itself contains no network or
model-dependent tests.
