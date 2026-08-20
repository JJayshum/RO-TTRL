"""Matched-compute option-permutation pilot on public MCQ benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np

from .model_pilot import (
    LETTERS,
    bootstrap_delta,
    bootstrap_map_control,
    generate_samples,
    majority,
    orbit_permutations,
    randomized_inverse,
)
from .pooling import canonical_pool


BENCHMARKS = {
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "validation"),
    "openbookqa": ("allenai/openbookqa", "main", "validation"),
    "mmlu": ("cais/mmlu", "all", "test"),
}

LOCAL_FILES = {
    "arc_challenge": "ai2_arc/ARC-Challenge/validation-00000-of-00001.parquet",
    "openbookqa": "openbookqa/main/validation-00000-of-00001.parquet",
    "mmlu": "mmlu/all/test-00000-of-00001.parquet",
}


def normalize_row(benchmark: str, row: dict, source_index: int) -> dict | None:
    if benchmark == "arc_challenge":
        stem = row["question"]
        options = list(row["choices"]["text"])
        labels = list(row["choices"]["label"])
        answer = str(row["answerKey"])
        if answer not in labels:
            return None
        truth = labels.index(answer)
        metadata = {"source_id": row.get("id")}
    elif benchmark == "openbookqa":
        stem = row["question_stem"]
        options = list(row["choices"]["text"])
        labels = list(row["choices"]["label"])
        answer = str(row["answerKey"])
        if answer not in labels:
            return None
        truth = labels.index(answer)
        metadata = {"source_id": row.get("id")}
    elif benchmark == "mmlu":
        stem = row["question"]
        options = list(row["choices"])
        truth = int(row["answer"])
        metadata = {"subject": row.get("subject")}
    else:
        raise ValueError(f"unsupported benchmark: {benchmark}")
    if len(options) != 4 or not 0 <= truth < 4:
        return None
    canonical = json.dumps(
        {"stem": stem, "options": options, "truth": truth},
        sort_keys=True,
        ensure_ascii=True,
    )
    return {
        "id": f"{benchmark}:{source_index}",
        "stem": stem,
        "options": options,
        "truth": truth,
        "source_index": source_index,
        "content_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "metadata": metadata,
    }


def stable_select(items: list[dict], count: int, seed: int, offset: int = 0) -> list[dict]:
    if offset < 0 or offset + count > len(items):
        raise ValueError(
            f"requested rows {offset}:{offset + count} from only {len(items)} eligible rows"
        )
    salt = str(seed).encode()
    ranked = sorted(
        items,
        key=lambda item: hashlib.sha256(salt + item["content_sha256"].encode()).digest(),
    )
    return ranked[offset:offset + count]


def load_excluded_hashes(manifest_paths) -> set[str]:
    hashes = set()
    for manifest_path in manifest_paths:
        prior = json.loads(Path(manifest_path).read_text())
        hashes.update(item["content_sha256"] for item in prior.get("items", []))
        for split in ("train_items", "eval_items"):
            hashes.update(item["content_sha256"] for item in prior.get(split, []))
    return hashes


def benchmark_view(item: dict, permutation: tuple[int, ...]) -> str:
    new_options = [None] * 4
    for old, new in enumerate(permutation):
        new_options[new] = item["options"][old]
    options = "\n".join(f"{LETTERS[i]}. {text}" for i, text in enumerate(new_options))
    return (
        f"Question: {item['stem']}\n{options}\n"
        "Reason briefly, then end with exactly 'Answer: X', where X is A, B, C, or D."
    )


def load_items(
    benchmark: str,
    count: int,
    sample_seed: int,
    sample_offset: int = 0,
    exclude_hashes: set[str] | None = None,
):
    from datasets import load_dataset

    path, config, split = BENCHMARKS[benchmark]
    local_root = os.environ.get("RO_BENCHMARK_DATA_ROOT")
    if local_root:
        parquet = Path(local_root) / LOCAL_FILES[benchmark]
        if not parquet.is_file():
            raise FileNotFoundError(f"missing mirrored dataset split: {parquet}")
        dataset = load_dataset("parquet", data_files=str(parquet), split="train")
        transport = {
            "kind": "modelscope_mirror_parquet",
            "local_file": str(parquet),
            "file_sha256": hashlib.sha256(parquet.read_bytes()).hexdigest(),
        }
    else:
        dataset = load_dataset(path, config, split=split)
        transport = {"kind": "huggingface_hub"}
    normalized = [normalize_row(benchmark, row, i) for i, row in enumerate(dataset)]
    eligible = [item for item in normalized if item is not None]
    excluded = exclude_hashes or set()
    available = [item for item in eligible if item["content_sha256"] not in excluded]
    return stable_select(available, count, sample_seed, sample_offset), {
        "path": path,
        "config": config,
        "split": split,
        "fingerprint": dataset._fingerprint,
        "source_rows": len(dataset),
        "eligible_four_option_rows": len(eligible),
        "excluded_prior_rows": len(eligible) - len(available),
        "available_after_exclusion": len(available),
        "transport": transport,
    }


def run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    exclude_hashes = load_excluded_hashes(args.exclude_manifest)
    items, dataset_info = load_items(
        args.benchmark,
        args.orbits,
        args.sample_seed,
        args.sample_offset,
        exclude_hashes,
    )
    manifest = {
        "benchmark": args.benchmark,
        "sample_seed": args.sample_seed,
        "sample_offset": args.sample_offset,
        "generation_seed": args.seed,
        "dataset": dataset_info,
        "configuration": {
            "model": args.model,
            "views": args.views,
            "samples_per_view": args.samples,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": 0.95,
            "alpha": args.alpha,
            "gauge_seeds": args.gauge_seeds,
        },
        "exclusion_manifests": [str(path) for path in args.exclude_manifest],
        "items": [{k: item[k] for k in ("id", "source_index", "content_sha256")}
                  for item in items],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
    ).eval()

    records = []
    mapped_tokens = root_tokens = 0
    records_path = out / "records.jsonl"
    records_path.write_text("")
    for orbit, item in enumerate(items):
        permutations = orbit_permutations(args.views, args.seed + orbit)
        prompts = [benchmark_view(item, p) for p in permutations]
        view_samples, view_texts, tokens = generate_samples(
            model,
            tokenizer,
            prompts,
            samples=args.samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            seed=args.seed + orbit,
        )
        mapped_tokens += tokens
        inverse_maps = [lambda answer, p=p: None if answer is None else p.index(answer)
                        for p in permutations]
        support = (None, 0, 1, 2, 3)
        q = canonical_pool(view_samples, inverse_maps, support=support, alpha=args.alpha)
        mapped = max(range(4), key=lambda answer: (q[answer], -answer))
        unmapped = majority([answer for view in view_samples for answer in view])

        root_groups, root_texts, tokens = generate_samples(
            model,
            tokenizer,
            [prompts[0]],
            samples=args.samples * args.views,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            seed=args.seed + 100000 + orbit,
        )
        root_tokens += tokens
        root = majority(root_groups[0])

        gauges = []
        for gauge_seed in range(args.gauge_seeds):
            gauge_maps = [inverse_maps[0]]
            for view, permutation in enumerate(permutations[1:], start=1):
                rng = random.Random(args.gauge_base + gauge_seed * 100 + view)
                rho = list(range(4))
                rng.shuffle(rho)
                gauge_maps.append(randomized_inverse(permutation, tuple(rho)))
            qg = canonical_pool(view_samples, gauge_maps, support=support, alpha=args.alpha)
            gauges.append(max(range(4), key=lambda answer: (qg[answer], -answer)))

        canonical = [[inverse_maps[i](answer) for answer in view]
                     for i, view in enumerate(view_samples)]
        truth = item["truth"]
        record = {
            "id": item["id"],
            "content_sha256": item["content_sha256"],
            "truth": truth,
            "root": root,
            "mapped": mapped,
            "unmapped": unmapped,
            "root_correct": root == truth,
            "mapped_correct": mapped == truth,
            "unmapped_correct": unmapped == truth,
            "gauge_accuracy": sum(answer == truth for answer in gauges) / len(gauges),
            "gauge_correct": [answer == truth for answer in gauges],
            "permutations": permutations,
            "view_samples": view_samples,
            "root_samples": root_groups[0],
            "truth_support": any(truth in view for view in canonical),
            "canonical_rollout_accuracy": sum(answer == truth for view in canonical
                                                 for answer in view) /
                                          (args.views * args.samples),
            "parse_failure_rate": sum(answer is None for view in view_samples
                                      for answer in view) /
                                  (args.views * args.samples),
            "metadata": item["metadata"],
        }
        if args.store_raw_text:
            record["view_texts"] = [
                view_texts[i:i + args.samples]
                for i in range(0, len(view_texts), args.samples)
            ]
            record["root_texts"] = root_texts
        records.append(record)
        with records_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")

    root_ok = [row["root_correct"] for row in records]
    mapped_ok = [row["mapped_correct"] for row in records]
    gauge_ok = np.asarray([row["gauge_correct"] for row in records], dtype=float)
    root_accuracy = float(np.mean(root_ok))
    mapped_accuracy = float(np.mean(mapped_ok))
    gauge_accuracy = float(np.mean(gauge_ok))
    summary = {
        "model": args.model,
        "benchmark": args.benchmark,
        "orbits": len(records),
        "views": args.views,
        "samples_per_view": args.samples,
        "root_accuracy": root_accuracy,
        "unmapped_accuracy": float(np.mean([row["unmapped_correct"] for row in records])),
        "mapped_accuracy": mapped_accuracy,
        "mapped_minus_root": mapped_accuracy - root_accuracy,
        "mapped_minus_root_95ci": bootstrap_delta(root_ok, mapped_ok, seed=args.seed),
        "randomized_gauge_accuracy": gauge_accuracy,
        "correct_map_minus_randomized": mapped_accuracy - gauge_accuracy,
        "correct_map_minus_randomized_95ci": bootstrap_map_control(
            mapped_ok, gauge_ok, seed=args.seed
        ),
        "truth_support": float(np.mean([row["truth_support"] for row in records])),
        "canonical_rollout_accuracy": float(np.mean(
            [row["canonical_rollout_accuracy"] for row in records]
        )),
        "parse_failure": float(np.mean([row["parse_failure_rate"] for row in records])),
        "mapped_completion_tokens": mapped_tokens,
        "root_completion_tokens": root_tokens,
        "dataset": dataset_info,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--benchmark", choices=tuple(BENCHMARKS), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--orbits", type=int, default=64)
    parser.add_argument("--views", type=int, default=4)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--gauge-seeds", type=int, default=20)
    parser.add_argument("--gauge-base", type=int, default=391000)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--sample-seed", type=int, default=20260816)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--exclude-manifest", action="append", default=[])
    parser.add_argument("--store-raw-text", action="store_true")
    parser.add_argument("--seed", type=int, required=True)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
