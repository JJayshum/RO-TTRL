"""Paired fixed-model analysis for the preregistered cross-family panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .paper_analysis import _aligned_arrays, _effect, _map_control, load_shared_records


def analyze_panel(data, *, seed=20260817, replicates=10000):
    models, benchmarks, aligned = _aligned_arrays(data)
    if len(models) != 3:
        raise ValueError(f"expected exactly three candidate models, found {len(models)}")

    rng = np.random.default_rng(seed)
    model_rows = {
        model: [
            row
            for benchmark in benchmarks
            for row in aligned[benchmark][model]
        ]
        for model in models
    }
    summary = {}
    for model, rows in model_rows.items():
        effect = _effect(rows)
        summary[model] = {
            "questions": len(rows),
            "root_accuracy": float(np.mean([row["root_correct"] for row in rows])),
            "mapped_accuracy": float(np.mean([row["mapped_correct"] for row in rows])),
            "mapped_minus_root": effect,
            "correct_map_minus_randomized": _map_control(rows),
            "truth_support": float(np.mean([row["truth_support"] for row in rows])),
            "parse_failure": float(np.mean([row["parse_failure_rate"] for row in rows])),
            "repairs": sum(row["mapped_correct"] and not row["root_correct"] for row in rows),
            "corruptions": sum(row["root_correct"] and not row["mapped_correct"] for row in rows),
        }

    effect_draws = {model: [] for model in models}
    map_draws = {model: [] for model in models}
    overall_effect_draws = []
    overall_map_draws = []
    gauge_count = len(model_rows[models[0]][0]["gauge_correct"])

    for _ in range(replicates):
        sampled = {model: [] for model in models}
        gauge_indices = rng.integers(0, gauge_count, gauge_count)
        for benchmark in benchmarks:
            count = len(aligned[benchmark][models[0]])
            indices = rng.integers(0, count, count)
            for model in models:
                sampled[model].extend(aligned[benchmark][model][i] for i in indices)

        effects = []
        controls = []
        for model in models:
            effect = _effect(sampled[model])
            control = float(np.mean([
                row["mapped_correct"] - np.mean(
                    np.asarray(row["gauge_correct"], dtype=float)[gauge_indices]
                )
                for row in sampled[model]
            ]))
            effect_draws[model].append(effect)
            map_draws[model].append(control)
            effects.append(effect)
            controls.append(control)
        overall_effect_draws.append(float(np.mean(effects)))
        overall_map_draws.append(float(np.mean(controls)))

    def interval(draws):
        return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]

    for model in models:
        summary[model]["mapped_minus_root_95ci"] = interval(effect_draws[model])
        summary[model]["correct_map_minus_randomized_95ci"] = interval(map_draws[model])
        summary[model]["individual_positive_replication"] = (
            summary[model]["mapped_minus_root"] >= 0.03
            and summary[model]["mapped_minus_root_95ci"][0] > 0
        )

    overall_effect = float(np.mean([summary[model]["mapped_minus_root"] for model in models]))
    overall_map = float(np.mean([
        summary[model]["correct_map_minus_randomized"] for model in models
    ]))
    overall = {
        "independent_question_blocks": sum(
            len(aligned[benchmark][models[0]]) for benchmark in benchmarks
        ),
        "fixed_candidate_models": models,
        "model_question_observations": sum(len(rows) for rows in model_rows.values()),
        "fixed_model_average_mapped_minus_root": overall_effect,
        "fixed_model_average_mapped_minus_root_95ci": interval(overall_effect_draws),
        "fixed_model_average_correct_map_minus_randomized": overall_map,
        "fixed_model_average_correct_map_minus_randomized_95ci": interval(
            overall_map_draws
        ),
    }
    criteria = {
        "fixed_model_effect_ci_positive": overall[
            "fixed_model_average_mapped_minus_root_95ci"
        ][0] > 0,
        "two_of_three_model_effects_nonnegative": sum(
            summary[model]["mapped_minus_root"] >= 0 for model in models
        ) >= 2,
        "no_model_interval_wholly_below_zero": all(
            summary[model]["mapped_minus_root_95ci"][1] >= 0 for model in models
        ),
        "at_least_one_individual_positive_replication": any(
            summary[model]["individual_positive_replication"] for model in models
        ),
        "map_control_at_least_0.05": overall_map >= 0.05,
        "map_control_ci_positive": overall[
            "fixed_model_average_correct_map_minus_randomized_95ci"
        ][0] > 0,
        "truth_support_at_least_0.80_each": all(
            summary[model]["truth_support"] >= 0.80 for model in models
        ),
        "parse_failure_at_most_0.10_each": all(
            summary[model]["parse_failure"] <= 0.10 for model in models
        ),
    }
    return {
        "design": {
            "independent_unit": "question",
            "bootstrap": "paired question resampling stratified by benchmark; models fixed",
            "replicates": replicates,
            "seed": seed,
            "candidate_count": len(models),
        },
        "by_model": summary,
        "overall": overall,
        "generality_gate": {"passed": all(criteria.values()), "criteria": criteria},
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--replicates", type=int, default=10000)
    args = parser.parse_args(argv)
    result = analyze_panel(
        load_shared_records(args.root), seed=args.seed, replicates=args.replicates
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
