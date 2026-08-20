"""Matched-compute inference pilot across verified transformation families."""

import argparse
import json
from pathlib import Path
import random

import numpy as np

from .families import build_orbit, validate_orbit
from .model_pilot import bootstrap_delta, generate_samples, majority
from .pooling import canonical_pool


def randomized_inverse(correct_inverse, valid, rho):
    rho_inverse = {rho[x]: x for x in valid}
    return lambda answer: None if answer is None else correct_inverse(rho_inverse[answer])


def run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    records_path = out / "records.jsonl"; records_path.write_text("")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True).eval()
    records = []
    for index in range(args.orbits):
        orbit = build_orbit(args.family, args.seed * 100000 + index,
                            args.views, args.difficulty)
        validate_orbit(orbit)
        view_samples, _, mapped_tokens = generate_samples(
            model, tokenizer, orbit.prompts, samples=args.samples,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            seed=args.seed + index, parser=orbit.parser)
        q = canonical_pool(view_samples, orbit.inverse_maps, support=orbit.support,
                           alpha=args.alpha)
        valid = [x for x in orbit.support if x is not None]
        mapped = max(valid, key=lambda a: (q[a], -a))
        count = args.views * args.samples
        root_groups, _, root_tokens = generate_samples(
            model, tokenizer, [orbit.prompts[0]], samples=count,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            seed=args.seed + 100000 + index, parser=orbit.parser)
        root = majority(root_groups[0]) if valid == list(range(4)) else max(
            valid, key=lambda a: (root_groups[0].count(a), -a))
        flat = [a for view in view_samples for a in view]
        unmapped = max(valid, key=lambda a: (flat.count(a), -a))
        gauge_correct = []
        for gauge_seed in range(args.gauge_seeds):
            gauge_maps = [orbit.inverse_maps[0]]
            for view in range(1, args.views):
                rng = random.Random(args.gauge_base + gauge_seed * 100 + view)
                rho_values = list(valid); rng.shuffle(rho_values)
                rho = dict(zip(valid, rho_values))
                gauge_maps.append(randomized_inverse(orbit.inverse_maps[view], valid, rho))
            qg = canonical_pool(view_samples, gauge_maps, support=orbit.support, alpha=args.alpha)
            pred = max(valid, key=lambda a: (qg[a], -a))
            gauge_correct.append(pred == orbit.truth)
        canonical = [[orbit.inverse_maps[i](a) for a in view]
                     for i, view in enumerate(view_samples)]
        record = {"id": args.seed * 100000 + index, "family": args.family,
                  "truth": orbit.truth, "root_correct": root == orbit.truth,
                  "mapped_correct": mapped == orbit.truth,
                  "unmapped_correct": unmapped == orbit.truth,
                  "gauge_correct": gauge_correct,
                  "truth_support": any(orbit.truth in view for view in canonical),
                  "parse_failure": sum(a is None for a in flat) / len(flat),
                  "mapped_tokens": mapped_tokens, "root_tokens": root_tokens,
                  "metadata": orbit.metadata}
        records.append(record)
        with records_path.open("a") as f: f.write(json.dumps(record) + "\n")
    root_ok = [r["root_correct"] for r in records]
    mapped_ok = [r["mapped_correct"] for r in records]
    root_acc = float(np.mean(root_ok)); mapped_acc = float(np.mean(mapped_ok))
    gauge_acc = float(np.mean([r["gauge_correct"] for r in records]))
    summary = {"family": args.family, "difficulty": args.difficulty,
               "orbits": args.orbits, "root_accuracy": root_acc,
               "mapped_accuracy": mapped_acc, "mapped_minus_root": mapped_acc-root_acc,
               "mapped_minus_root_95ci": bootstrap_delta(root_ok, mapped_ok, seed=args.seed),
               "unmapped_accuracy": float(np.mean([r["unmapped_correct"] for r in records])),
               "randomized_gauge_accuracy": gauge_acc,
               "correct_map_minus_randomized": mapped_acc-gauge_acc,
               "truth_support": float(np.mean([r["truth_support"] for r in records])),
               "parse_failure": float(np.mean([r["parse_failure"] for r in records])),
               "mapped_tokens": sum(r["mapped_tokens"] for r in records),
               "root_tokens": sum(r["root_tokens"] for r in records)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--family", choices=("option", "boolean", "affine"), required=True)
    p.add_argument("--difficulty", default="easy")
    p.add_argument("--output", required=True)
    p.add_argument("--orbits", type=int, default=32)
    p.add_argument("--views", type=int, default=4)
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--gauge-seeds", type=int, default=20)
    p.add_argument("--gauge-base", type=int, default=91000)
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument("--temperature", type=float, default=.8)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=3101)
    run(p.parse_args(argv))


if __name__ == "__main__": main()
