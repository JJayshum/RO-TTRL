"""Shared-question, fixed-model analysis for the paper-scale confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_shared_records(root: Path) -> dict[str, dict[str, list[dict]]]:
    data = {}
    for model_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        benchmarks = {}
        for benchmark_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
            records_path = benchmark_dir / "records.jsonl"
            if not records_path.is_file() or not (benchmark_dir / "manifest.json").is_file():
                continue
            with records_path.open() as handle:
                benchmarks[benchmark_dir.name] = [
                    json.loads(line) for line in handle if line.strip()
                ]
        if benchmarks:
            data[model_dir.name] = benchmarks
    return data


def _aligned_arrays(data):
    models = sorted(data)
    benchmarks = sorted(data[models[0]])
    rows = {}
    for benchmark in benchmarks:
        reference = {
            row["content_sha256"]: row for row in data[models[0]][benchmark]
        }
        hashes = sorted(reference)
        if not hashes:
            raise ValueError(f"no records for {benchmark}")
        rows[benchmark] = {}
        for model in models:
            current = {
                row["content_sha256"]: row for row in data[model][benchmark]
            }
            if set(current) != set(hashes):
                raise ValueError(f"shared question hashes differ for {model}/{benchmark}")
            rows[benchmark][model] = [current[content_hash] for content_hash in hashes]
    return models, benchmarks, rows


def _effect(rows):
    return float(np.mean([row["mapped_correct"] - row["root_correct"] for row in rows]))


def _map_control(rows):
    return float(np.mean([
        row["mapped_correct"] - np.mean(row["gauge_correct"]) for row in rows
    ]))


def analyze_shared(data, *, seed=20260817, replicates=5000):
    models, benchmarks, aligned = _aligned_arrays(data)
    rng = np.random.default_rng(seed)

    model_rows = {
        model: [
            row
            for benchmark in benchmarks
            for row in aligned[benchmark][model]
        ]
        for model in models
    }
    model_summary = {}
    for model, rows in model_rows.items():
        model_summary[model] = {
            "questions": len(rows),
            "root_accuracy": float(np.mean([row["root_correct"] for row in rows])),
            "mapped_accuracy": float(np.mean([row["mapped_correct"] for row in rows])),
            "mapped_minus_root": _effect(rows),
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
        model_effects = []
        model_maps = []
        for model in models:
            effect = _effect(sampled[model])
            map_control = float(np.mean([
                row["mapped_correct"] - np.mean(
                    np.asarray(row["gauge_correct"], dtype=float)[gauge_indices]
                )
                for row in sampled[model]
            ]))
            effect_draws[model].append(effect)
            map_draws[model].append(map_control)
            model_effects.append(effect)
            model_maps.append(map_control)
        overall_effect_draws.append(float(np.mean(model_effects)))
        overall_map_draws.append(float(np.mean(model_maps)))

    def interval(draws):
        return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]

    for model in models:
        model_summary[model]["mapped_minus_root_95ci"] = interval(effect_draws[model])
        model_summary[model]["correct_map_minus_randomized_95ci"] = interval(
            map_draws[model]
        )

    overall_effect = float(np.mean([
        model_summary[model]["mapped_minus_root"] for model in models
    ]))
    overall_map = float(np.mean([
        model_summary[model]["correct_map_minus_randomized"] for model in models
    ]))
    overall = {
        "independent_question_blocks": sum(
            len(aligned[benchmark][models[0]]) for benchmark in benchmarks
        ),
        "fixed_models": models,
        "model_question_observations": sum(len(rows) for rows in model_rows.values()),
        "fixed_model_average_mapped_minus_root": overall_effect,
        "fixed_model_average_mapped_minus_root_95ci": interval(overall_effect_draws),
        "fixed_model_average_correct_map_minus_randomized": overall_map,
        "fixed_model_average_correct_map_minus_randomized_95ci": interval(
            overall_map_draws
        ),
    }
    criteria = {
        "fixed_model_effect_ci_excludes_zero": (
            overall["fixed_model_average_mapped_minus_root_95ci"][0] > 0
        ),
        "two_of_three_model_effects_nonnegative": sum(
            model_summary[model]["mapped_minus_root"] >= 0 for model in models
        ) >= 2,
        "no_model_interval_wholly_below_zero": all(
            model_summary[model]["mapped_minus_root_95ci"][1] >= 0 for model in models
        ),
        "map_control_at_least_0.05": overall_map >= 0.05,
        "map_control_ci_excludes_zero": (
            overall["fixed_model_average_correct_map_minus_randomized_95ci"][0] > 0
        ),
        "truth_support_at_least_0.80_each": all(
            model_summary[model]["truth_support"] >= 0.80 for model in models
        ),
        "parse_failure_at_most_0.10_each": all(
            model_summary[model]["parse_failure"] <= 0.10 for model in models
        ),
    }
    return {
        "design": {
            "independent_unit": "question block",
            "bootstrap": "paired question resampling stratified by benchmark; models fixed",
            "replicates": replicates,
            "seed": seed,
            "multiplicity": (
                "two co-primary criteria are conjunctive; per-model intervals are descriptive"
            ),
        },
        "by_model": model_summary,
        "overall": overall,
        "paper_gate": {"passed": all(criteria.values()), "criteria": criteria},
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--replicates", type=int, default=5000)
    args = parser.parse_args(argv)
    result = analyze_shared(
        load_shared_records(args.root), seed=args.seed, replicates=args.replicates
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
