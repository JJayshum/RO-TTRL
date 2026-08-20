"""Label-free secondary analyses for frozen benchmark generations."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .model_pilot import bootstrap_delta, bootstrap_map_control, majority
from .pooling import canonical_pool


SUPPORT = (None, 0, 1, 2, 3)


def inverse_maps(permutations):
    """Return exact node-to-canonical maps for stored option permutations."""
    return [lambda answer, p=tuple(p): None if answer is None else p.index(answer)
            for p in permutations]


def answer_argmax(distribution):
    return max(range(4), key=lambda answer: (distribution[answer], -answer))


def probability_margin(distribution):
    ranked = sorted((distribution[answer] for answer in range(4)), reverse=True)
    return float(ranked[0] - ranked[1])


def mapped_predictions(record, budget: int, *, alpha: float = 1.0):
    """Compute log-pool and hard-vote predictions at a matched total budget."""
    samples = record["view_samples"]
    views = len(samples)
    if budget <= 0 or budget % views:
        raise ValueError("budget must be a positive multiple of the number of views")
    per_view = budget // views
    if any(len(view) < per_view for view in samples):
        raise ValueError(f"record does not contain {per_view} samples per view")
    selected = [view[:per_view] for view in samples]
    maps = inverse_maps(record["permutations"])
    distribution = canonical_pool(selected, maps, support=SUPPORT, alpha=alpha)
    canonical = [maps[i](answer) for i, view in enumerate(selected) for answer in view]
    return {
        "log_pool": answer_argmax(distribution),
        "hard_vote": majority(canonical),
        "margin": probability_margin(distribution),
        "distribution": {str(key): float(value) for key, value in distribution.items()},
    }


def root_prediction(record, budget: int):
    samples = record.get("root_samples")
    if samples is None:
        raise ValueError("record predates root-sample retention")
    if len(samples) < budget:
        raise ValueError(f"record contains only {len(samples)} root samples")
    return majority(samples[:budget])


def leave_one_view_out(record, *, alpha: float = 1.0):
    """Measure prediction stability without using correctness labels."""
    samples = record["view_samples"]
    maps = inverse_maps(record["permutations"])
    full = canonical_pool(samples, maps, support=SUPPORT, alpha=alpha)
    full_prediction = answer_argmax(full)
    predictions = []
    for omitted in range(len(samples)):
        reduced_samples = [view for i, view in enumerate(samples) if i != omitted]
        reduced_maps = [mapping for i, mapping in enumerate(maps) if i != omitted]
        q = canonical_pool(reduced_samples, reduced_maps, support=SUPPORT, alpha=alpha)
        predictions.append(answer_argmax(q))
    modal_count = max(Counter(predictions).values())
    return {
        "predictions": predictions,
        "agreement_with_full": sum(p == full_prediction for p in predictions) / len(predictions),
        "consensus": modal_count / len(predictions),
        "margin": probability_margin(full),
    }


def _accuracy(flags):
    return float(np.mean(flags)) if flags else math.nan


def summarize_confirmation(records, *, seed=0):
    """Aggregate the frozen inference metrics and evaluate the second-family gate."""
    records = list(records)
    if not records:
        raise ValueError("at least one record is required")

    def summarize(rows, group_seed):
        root_ok = [row["root_correct"] for row in rows]
        mapped_ok = [row["mapped_correct"] for row in rows]
        gauges = np.asarray([row["gauge_correct"] for row in rows], dtype=float)
        root_accuracy = _accuracy(root_ok)
        mapped_accuracy = _accuracy(mapped_ok)
        randomized_accuracy = float(np.mean(gauges))
        return {
            "questions": len(rows),
            "root_accuracy": root_accuracy,
            "mapped_accuracy": mapped_accuracy,
            "mapped_minus_root": mapped_accuracy - root_accuracy,
            "mapped_minus_root_95ci": bootstrap_delta(
                root_ok, mapped_ok, seed=group_seed
            ),
            "randomized_gauge_accuracy": randomized_accuracy,
            "correct_map_minus_randomized": mapped_accuracy - randomized_accuracy,
            "correct_map_minus_randomized_95ci": bootstrap_map_control(
                mapped_ok, gauges, seed=group_seed
            ),
            "truth_support": _accuracy([row["truth_support"] for row in rows]),
            "parse_failure": float(np.mean(
                [row["parse_failure_rate"] for row in rows]
            )),
        }

    groups = {}
    for row in records:
        benchmark = str(row["id"]).split(":", 1)[0]
        groups.setdefault(benchmark, []).append(row)
    by_benchmark = {
        name: summarize(rows, seed + index + 1)
        for index, (name, rows) in enumerate(sorted(groups.items()))
    }
    pooled = summarize(records, seed)
    nonnegative = sum(
        metrics["mapped_minus_root"] >= 0 for metrics in by_benchmark.values()
    )
    criteria = {
        "pooled_effect_at_least_0.03": pooled["mapped_minus_root"] >= 0.03,
        "pooled_effect_ci_excludes_zero": pooled["mapped_minus_root_95ci"][0] > 0,
        "two_of_three_dataset_effects_nonnegative": nonnegative >= 2,
        "map_control_at_least_0.05": pooled["correct_map_minus_randomized"] >= 0.05,
        "map_control_ci_excludes_zero": (
            pooled["correct_map_minus_randomized_95ci"][0] > 0
        ),
        "truth_support_at_least_0.80": pooled["truth_support"] >= 0.80,
        "parse_failure_at_most_0.10": pooled["parse_failure"] <= 0.10,
    }
    return {
        "by_benchmark": by_benchmark,
        "pooled": pooled,
        "gate": {"passed": all(criteria.values()), "criteria": criteria},
    }


def risk_coverage(rows, coverages=(0.25, 0.5, 0.75, 1.0)):
    """Rank by frozen label-free score and report fixed-coverage accuracy/risk."""
    ordered = sorted(
        rows,
        key=lambda row: (
            -row["loo_agreement"],
            -row["margin"],
            row["content_sha256"],
        ),
    )
    curve = []
    for coverage in coverages:
        retained = max(1, math.ceil(coverage * len(ordered)))
        accuracy = _accuracy([row["correct"] for row in ordered[:retained]])
        curve.append({
            "target_coverage": coverage,
            "retained": retained,
            "realized_coverage": retained / len(ordered),
            "accuracy": accuracy,
            "risk": 1.0 - accuracy,
        })
    return curve


def analyze_records(records: Iterable[dict], *, budgets=(4, 8, 16), seed=0):
    records = list(records)
    if not records:
        raise ValueError("at least one record is required")
    detailed = []
    for record in records:
        truth = record["truth"]
        row = {
            "id": record["id"],
            "content_sha256": record["content_sha256"],
            "truth": truth,
            "budgets": {},
        }
        for budget in budgets:
            mapped = mapped_predictions(record, budget)
            root = root_prediction(record, budget)
            row["budgets"][str(budget)] = {
                "root": root,
                "log_pool": mapped["log_pool"],
                "hard_vote": mapped["hard_vote"],
                "root_correct": root == truth,
                "log_pool_correct": mapped["log_pool"] == truth,
                "hard_vote_correct": mapped["hard_vote"] == truth,
            }
        loo = leave_one_view_out(record)
        full = row["budgets"][str(max(budgets))]
        row.update({
            "loo_predictions": loo["predictions"],
            "loo_agreement": loo["agreement_with_full"],
            "loo_consensus": loo["consensus"],
            "margin": loo["margin"],
            "correct": full["log_pool_correct"],
        })
        detailed.append(row)

    scaling = []
    for budget in budgets:
        key = str(budget)
        root_ok = [row["budgets"][key]["root_correct"] for row in detailed]
        log_ok = [row["budgets"][key]["log_pool_correct"] for row in detailed]
        hard_ok = [row["budgets"][key]["hard_vote_correct"] for row in detailed]
        scaling.append({
            "budget": budget,
            "root_accuracy": _accuracy(root_ok),
            "log_pool_accuracy": _accuracy(log_ok),
            "log_pool_minus_root": _accuracy(log_ok) - _accuracy(root_ok),
            "log_pool_minus_root_95ci": bootstrap_delta(root_ok, log_ok, seed=seed + budget),
            "hard_vote_accuracy": _accuracy(hard_ok),
            "log_pool_minus_hard_vote": _accuracy(log_ok) - _accuracy(hard_ok),
            "log_pool_minus_hard_vote_95ci": bootstrap_delta(
                hard_ok, log_ok, seed=seed + 1000 + budget
            ),
        })

    reliability_rows = [{
        "content_sha256": row["content_sha256"],
        "loo_agreement": row["loo_agreement"],
        "margin": row["margin"],
        "correct": row["correct"],
    } for row in detailed]
    summary = {
        "records": len(detailed),
        "compute_scaling": scaling,
        "reliability": {
            "mean_leave_one_view_out_agreement": float(np.mean(
                [row["loo_agreement"] for row in detailed]
            )),
            "all_views_stable_fraction": float(np.mean(
                [row["loo_agreement"] == 1.0 for row in detailed]
            )),
            "risk_coverage": risk_coverage(reliability_rows),
        },
    }
    return summary, detailed


def load_records(paths):
    records = []
    for path in paths:
        with Path(path).open() as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    return records


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("records", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args(argv)
    records = load_records(args.records)
    confirmation = summarize_confirmation(records, seed=args.seed)
    summary, detailed = analyze_records(records, seed=args.seed)
    summary["confirmation"] = confirmation
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output / "records.jsonl").open("w") as handle:
        for row in detailed:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
