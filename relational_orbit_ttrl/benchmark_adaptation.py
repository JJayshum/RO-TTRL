"""Matched-compute LoRA adaptation on a mixture of public MCQ benchmarks."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np

from .adaptation_pilot import train_adapter
from .benchmark_pilot import BENCHMARKS, benchmark_view, load_items
from .model_pilot import bootstrap_delta, generate_samples, majority, orbit_permutations
from .pooling import canonical_pool, leave_one_out_rewards


SUPPORT = (None, 0, 1, 2, 3)


def inverse_maps(permutations):
    return [lambda answer, p=p: None if answer is None else p.index(answer)
            for p in permutations]


def load_mixture(count_per_benchmark: int, sample_seed: int, sample_offset: int):
    items = []
    datasets = {}
    for benchmark in BENCHMARKS:
        selected, info = load_items(
            benchmark, count_per_benchmark, sample_seed, sample_offset
        )
        for item in selected:
            item = dict(item)
            item["benchmark"] = benchmark
            items.append(item)
        datasets[benchmark] = info
    return items, datasets


def build_rollout_data(model, tokenizer, args, items, method):
    examples = []
    parse_failures = 0
    for orbit, item in enumerate(items):
        permutations = orbit_permutations(args.views, args.train_seed + orbit)
        prompts = [benchmark_view(item, permutation) for permutation in permutations]
        if method == "mapped":
            samples, raw, _ = generate_samples(
                model,
                tokenizer,
                prompts,
                samples=args.samples,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                seed=args.train_seed + orbit,
            )
            rewards = leave_one_out_rewards(
                samples, inverse_maps(permutations), support=SUPPORT, alpha=args.alpha
            )
            raw_groups = [raw[i * args.samples:(i + 1) * args.samples]
                          for i in range(args.views)]
            for prompt, parsed, texts, group_rewards in zip(
                prompts, samples, raw_groups, rewards
            ):
                centered = np.asarray(group_rewards) - np.mean(group_rewards)
                for answer, text, advantage in zip(parsed, texts, centered):
                    examples.append((prompt, text, float(advantage)))
                    parse_failures += answer is None
        elif method == "root":
            count = args.views * args.samples
            samples, raw, _ = generate_samples(
                model,
                tokenizer,
                [prompts[0]],
                samples=count,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                seed=args.train_seed + 100000 + orbit,
            )
            rewards = leave_one_out_rewards(
                samples, [lambda answer: answer], support=SUPPORT, alpha=args.alpha
            )[0]
            centered = np.asarray(rewards) - np.mean(rewards)
            for answer, text, advantage in zip(samples[0], raw, centered):
                examples.append((prompts[0], text, float(advantage)))
                parse_failures += answer is None
        else:
            raise ValueError(method)
    stats = {
        "orbits": len(items),
        "sequences": len(examples),
        "parse_failure_rate": parse_failures / len(examples),
        "advantage_std": float(np.std([example[2] for example in examples])),
    }
    return examples, stats


def evaluate(model, tokenizer, args, items):
    rows = []
    for orbit, item in enumerate(items):
        permutations = orbit_permutations(args.views, args.eval_seed + orbit)
        prompts = [benchmark_view(item, permutation) for permutation in permutations]
        samples, _, _ = generate_samples(
            model,
            tokenizer,
            prompts,
            samples=args.eval_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            seed=args.eval_seed + orbit,
        )
        maps = inverse_maps(permutations)
        canonical = [[maps[i](answer) for answer in view]
                     for i, view in enumerate(samples)]
        q = canonical_pool(samples, maps, support=SUPPORT, alpha=args.alpha)
        mapped = max(range(4), key=lambda answer: (q[answer], -answer))
        truth = item["truth"]
        rows.append({
            "id": item["id"],
            "benchmark": item["benchmark"],
            "content_sha256": item["content_sha256"],
            "truth": truth,
            "policy_pass1": sum(answer == truth for answer in samples[0]) /
                            len(samples[0]),
            "root_majority": majority(samples[0]) == truth,
            "mapped": mapped == truth,
            "truth_support": any(truth in view for view in canonical),
            "parse_failure": sum(answer is None for view in samples for answer in view) /
                             (args.views * args.eval_samples),
        })
    return rows


def load_and_evaluate(model_path, adapter_path, tokenizer, args, items):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
    ).eval()
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path).eval()
    rows = evaluate(model, tokenizer, args, items)
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return rows


def summarize(rows):
    keys = ("policy_pass1", "root_majority", "mapped", "truth_support", "parse_failure")
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def policy_effect(reference, mapped, seed):
    left = [row["policy_pass1"] for row in reference]
    right = [row["policy_pass1"] for row in mapped]
    return {
        "effect": float(np.mean(right) - np.mean(left)),
        "paired_95ci": bootstrap_delta(left, right, seed=seed),
    }


def run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    train_items, train_datasets = load_mixture(
        args.train_per_benchmark, args.sample_seed, args.train_offset
    )
    eval_items, eval_datasets = load_mixture(
        args.eval_per_benchmark, args.sample_seed, args.eval_offset
    )
    train_hashes = {item["content_sha256"] for item in train_items}
    eval_hashes = {item["content_sha256"] for item in eval_items}
    if train_hashes & eval_hashes:
        raise ValueError("training and evaluation benchmark items overlap")
    manifest = {
        "sample_seed": args.sample_seed,
        "train_offset": args.train_offset,
        "eval_offset": args.eval_offset,
        "train_items": [{"benchmark": item["benchmark"], "id": item["id"],
                         "content_sha256": item["content_sha256"]}
                        for item in train_items],
        "eval_items": [{"benchmark": item["benchmark"], "id": item["id"],
                        "content_sha256": item["content_sha256"]}
                       for item in eval_items],
        "train_datasets": train_datasets,
        "eval_datasets": eval_datasets,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
    ).eval()
    mapped_examples, mapped_rollouts = build_rollout_data(
        base, tokenizer, args, train_items, "mapped"
    )
    root_examples, root_rollouts = build_rollout_data(
        base, tokenizer, args, train_items, "root"
    )
    del base
    torch.cuda.empty_cache()
    gc.collect()

    mapped_path = out / "mapped_adapter"
    root_path = out / "root_adapter"
    mapped_train = train_adapter(args.model, tokenizer, mapped_examples, mapped_path, args)
    root_train = train_adapter(args.model, tokenizer, root_examples, root_path, args)
    evaluations = {
        "base": load_and_evaluate(args.model, None, tokenizer, args, eval_items),
        "root_ttrl": load_and_evaluate(args.model, root_path, tokenizer, args, eval_items),
        "mapped_ttrl": load_and_evaluate(args.model, mapped_path, tokenizer, args, eval_items),
    }
    for name, rows in evaluations.items():
        with (out / f"eval_{name}.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    per_benchmark = {}
    for benchmark in BENCHMARKS:
        subsets = {
            name: [row for row in rows if row["benchmark"] == benchmark]
            for name, rows in evaluations.items()
        }
        per_benchmark[benchmark] = {
            "evaluation": {name: summarize(rows) for name, rows in subsets.items()},
            "mapped_minus_root_ttrl": policy_effect(
                subsets["root_ttrl"], subsets["mapped_ttrl"], args.eval_seed + 10
            ),
        }
    summary = {
        "model": args.model,
        "train_orbits": len(train_items),
        "eval_orbits": len(eval_items),
        "mapped_rollouts": mapped_rollouts,
        "root_rollouts": root_rollouts,
        "mapped_train": mapped_train,
        "root_train": root_train,
        "evaluation": {name: summarize(rows) for name, rows in evaluations.items()},
        "mapped_minus_base": policy_effect(
            evaluations["base"], evaluations["mapped_ttrl"], args.eval_seed
        ),
        "mapped_minus_root_ttrl": policy_effect(
            evaluations["root_ttrl"], evaluations["mapped_ttrl"], args.eval_seed + 1
        ),
        "per_benchmark": per_benchmark,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-seed", type=int, default=20260816)
    parser.add_argument("--train-offset", type=int, default=128)
    parser.add_argument("--eval-offset", type=int, default=144)
    parser.add_argument("--train-per-benchmark", type=int, default=6)
    parser.add_argument("--eval-per-benchmark", type=int, default=32)
    parser.add_argument("--views", type=int, default=4)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--eval-samples", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--train-max-length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--train-seed", type=int, default=31801)
    parser.add_argument("--eval-seed", type=int, default=31901)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--kl-beta", type=float, default=0.02)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
