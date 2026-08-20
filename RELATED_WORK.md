# Related-Work Positioning

Initial multi-source discovery was run through OpenAlex on 16 August 2026 with
queries covering multiple-choice option order, self-consistency, prompt
ensembles, and permutation self-consistency. This is a discovery audit, not a
claim of exhaustive novelty.

## Closest precedents

1. Wang et al., *Self-Consistency Improves Chain of Thought Reasoning in
   Language Models*, arXiv:2203.11171, DOI
   [10.48550/arxiv.2203.11171](https://doi.org/10.48550/arxiv.2203.11171).
   It samples diverse reasoning paths and marginalizes their answers. Our
   method also uses matched sampling, but additionally applies exact inverse
   maps for known option-space transformations before pooling.
2. Zheng et al., *Large Language Models Are Not Robust Multiple Choice
   Selectors*, arXiv:2309.03882, DOI
   [10.48550/arxiv.2309.03882](https://doi.org/10.48550/arxiv.2309.03882).
   It documents option-selection bias and sensitivity to option positions across
   multiple models and benchmarks. This motivates our position-randomized and
   randomized-gauge controls; it is a problem diagnosis rather than the same
   exact-map pooling procedure.
3. Pezeshkpour and Hruschka, *Large Language Models Sensitivity to The Order of
   Options in Multiple-Choice Questions*, Findings of NAACL 2024, DOI
   [10.18653/v1/2024.findings-naacl.130](https://doi.org/10.18653/v1/2024.findings-naacl.130).
   It studies robustness to option order. Our evaluation keeps the question
   semantics fixed, transforms the option coordinate system, and maps outputs
   back to a shared canonical space before aggregation.
4. Tang et al., *Found in the Middle: Permutation Self-Consistency Improves
   Listwise Ranking in Large Language Models*, NAACL 2024, DOI
   [10.18653/v1/2024.naacl-long.129](https://doi.org/10.18653/v1/2024.naacl-long.129).
   It marginalizes list permutations for order-independent ranking. The
   distinction to make explicit is task structure: listwise ranking aggregates
   reordered candidate lists, whereas this project uses a known group action on
   answer labels and applies the exact inverse action to each generated answer.

## Positioning that remains defensible

The strongest defensible contribution is a controlled inference-time protocol
for equivariant option alignment: known cyclic option transformations, exact
inverse canonicalization, full-support log-opinion pooling, and randomized-map
controls under matched generation budgets. The Qwen2.5 3B/7B evidence supports
this scoped claim. The original Llama-3.2-3B cross-family gate was not passed;
the paper must report that limitation rather than describe model-family
generality as established.

The TTRL/LoRA adaptation results should remain secondary and exploratory unless
the independent shared-block analysis supplies stronger evidence. Related-work
claims should be verified against the final publisher versions before
submission.
