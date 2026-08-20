"""Structural compatibility audit for stored option-permutation runs."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np


def _load_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _manifest_hashes(paths: list[Path]) -> set[str]:
    hashes = set()
    for path in paths:
        manifest = json.loads(path.read_text())
        hashes.update(item["content_sha256"] for item in manifest.get("items", []))
    return hashes


def audit_run(run_dir: Path, reference_manifests: list[Path] | None = None) -> dict:
    benchmark_rows = []
    all_hashes = []
    structural_failures = []
    token_totals = {"mapped": 0, "root": 0}

    for benchmark_dir in sorted(
        path for path in run_dir.iterdir()
        if path.is_dir() and (path / "manifest.json").is_file()
    ):
        manifest = json.loads((benchmark_dir / "manifest.json").read_text())
        summary = json.loads((benchmark_dir / "summary.json").read_text())
        records = _load_jsonl(benchmark_dir / "records.jsonl")
        name = manifest["benchmark"]
        if len(records) != len(manifest["items"]):
            structural_failures.append(f"{name}: record/manifest count mismatch")
        manifest_hashes = [item["content_sha256"] for item in manifest["items"]]
        record_hashes = [row["content_sha256"] for row in records]
        if manifest_hashes != record_hashes:
            structural_failures.append(f"{name}: record order or content hashes differ")
        all_hashes.extend(record_hashes)

        view_count = 4
        answer_counts = [Counter() for _ in range(view_count)]
        parse_failures = np.zeros(view_count, dtype=float)
        canonical_correct = np.zeros(view_count, dtype=float)
        sample_counts = np.zeros(view_count, dtype=float)
        for row in records:
            permutations = row["permutations"]
            samples = row["view_samples"]
            if len(permutations) != view_count or len(samples) != view_count:
                structural_failures.append(f"{row['id']}: expected four views")
                continue
            if len(row.get("root_samples", [])) != 16:
                structural_failures.append(f"{row['id']}: expected sixteen root samples")
            if "root_texts" in row and len(row["root_texts"]) != len(row["root_samples"]):
                structural_failures.append(f"{row['id']}: root text/sample count mismatch")
            if "view_texts" in row and [len(group) for group in row["view_texts"]] != [
                len(group) for group in samples
            ]:
                structural_failures.append(f"{row['id']}: view text/sample count mismatch")
            for view, (permutation, emitted) in enumerate(zip(permutations, samples)):
                if sorted(permutation) != list(range(4)):
                    structural_failures.append(f"{row['id']}: invalid permutation in view {view}")
                    continue
                if len(emitted) != 4:
                    structural_failures.append(f"{row['id']}: expected four samples in view {view}")
                for answer in emitted:
                    answer_counts[view][answer] += 1
                    sample_counts[view] += 1
                    if answer is None:
                        parse_failures[view] += 1
                    elif permutation.index(answer) == row["truth"]:
                        canonical_correct[view] += 1

        view_metrics = []
        for view in range(view_count):
            denom = sample_counts[view] or 1.0
            view_metrics.append({
                "view": view,
                "samples": int(sample_counts[view]),
                "parse_failure": float(parse_failures[view] / denom),
                "canonical_rollout_accuracy": float(canonical_correct[view] / denom),
                "emitted_answer_counts": {
                    "parse_failure": answer_counts[view][None],
                    **{str(answer): answer_counts[view][answer] for answer in range(4)},
                },
            })
        parse_ok = all(row["parse_failure"] <= 0.10 for row in view_metrics)
        accuracies = [row["canonical_rollout_accuracy"] for row in view_metrics]
        view_symmetry_ok = max(accuracies) - min(accuracies) <= 0.15
        token_totals["mapped"] += summary["mapped_completion_tokens"]
        token_totals["root"] += summary["root_completion_tokens"]
        benchmark_rows.append({
            "benchmark": name,
            "questions": len(records),
            "sample_offset": manifest["sample_offset"],
            "sample_seed": manifest["sample_seed"],
            "generation_seed": manifest["generation_seed"],
            "view_metrics": view_metrics,
            "parse_ok": parse_ok,
            "view_symmetry_ok": view_symmetry_ok,
        })

    unique_hashes = len(all_hashes) == len(set(all_hashes))
    reference_hashes = _manifest_hashes(reference_manifests or [])
    overlap = sorted(set(all_hashes) & reference_hashes)
    larger_tokens = max(token_totals.values()) or 1
    token_mismatch = abs(token_totals["mapped"] - token_totals["root"]) / larger_tokens
    checks = {
        "structural_records_valid": not structural_failures,
        "confirmation_hashes_unique": unique_hashes,
        "confirmation_hashes_disjoint_from_references": not overlap,
        "per_view_parse_failure_at_most_0.10": all(
            row["parse_ok"] for row in benchmark_rows
        ),
        "per_benchmark_view_accuracy_range_at_most_0.15": all(
            row["view_symmetry_ok"] for row in benchmark_rows
        ),
        "completion_token_mismatch_at_most_0.10": token_mismatch <= 0.10,
    }
    return {
        "run_dir": str(run_dir),
        "benchmarks": benchmark_rows,
        "completion_tokens": token_totals,
        "completion_token_relative_mismatch": token_mismatch,
        "structural_failures": structural_failures,
        "reference_manifest_count": len(reference_manifests or []),
        "reference_overlap_count": len(overlap),
        "checks": checks,
        "stored_run_compatible": all(checks.values()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference-manifest", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    result = audit_run(args.run_dir, args.reference_manifest)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
