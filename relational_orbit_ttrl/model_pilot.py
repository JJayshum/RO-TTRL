"""Bounded model-backed RO-Infer pilot with exact option maps.

The pilot intentionally stops before parameter updates. It tests whether exact
answer maps improve pseudo-label top-1 accuracy at matched generation count.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re

import numpy as np

from .pooling import canonical_pool


LETTERS = "ABCD"

DIFFICULTIES = {
    "easy": {"steps": (1, 2), "moduli": (7, 11, 13), "coefficient_max": 6},
    "medium": {"steps": (2, 3), "moduli": (11, 13, 17, 19), "coefficient_max": 10},
    "hard": {"steps": (3, 5), "moduli": (17, 19, 23, 29), "coefficient_max": 16},
    "extreme": {"steps": (5, 8), "moduli": (17, 19, 23, 29, 31, 37),
                "coefficient_max": 36},
}


def make_item(seed: int, difficulty: str = "extreme"):
    rng = random.Random(seed)
    config = DIFFICULTIES[difficulty]
    modulus = rng.choice(config["moduli"])
    value = rng.randrange(modulus)
    start = value
    steps = []
    for _ in range(rng.randint(*config["steps"])):
        a = rng.randrange(2, min(modulus, config["coefficient_max"] + 1))
        b = rng.randrange(1, modulus)
        steps.append((a, b))
        value = (a * value + b) % modulus
    distractors = set()
    while len(distractors) < 3:
        candidate = (value + rng.choice((-7, -5, -3, -2, -1, 1, 2, 3, 5, 7))) % modulus
        if candidate != value:
            distractors.add(candidate)
    options = [value, *sorted(distractors)]
    rng.shuffle(options)
    truth = options.index(value)
    operations = "; ".join(f"{i}. x=({a}x+{b}) mod {modulus}"
                           for i, (a, b) in enumerate(steps, start=1))
    stem = (f"Start with x={start}. There are exactly {len(steps)} operations. Apply each "
            f"exactly once in order: {operations}. "
            "What is the final value of x?")
    return {"id": seed, "stem": stem, "options": options, "truth": truth,
            "modulus": modulus, "steps": steps}


def permuted_view(item, permutation):
    """permutation[old_index] is the new index of that option."""
    new_options = [None] * len(permutation)
    for old, new in enumerate(permutation):
        new_options[new] = item["options"][old]
    lines = "\n".join(f"{LETTERS[i]}. {v}" for i, v in enumerate(new_options))
    prompt = (f"{item['stem']}\n{lines}\nShow one short calculation line per numbered operation "
              "and no other explanation. End with exactly 'Answer: X', where X is A, B, C, or D.")
    return prompt


def parse_answer(text):
    matches = re.findall(r"(?:answer\s*[:is-]*\s*|\b)([ABCD])\b", text.upper())
    return LETTERS.index(matches[-1]) if matches else None


def generate_samples(model, tokenizer, prompts, *, samples, max_new_tokens, temperature, seed,
                     parser=parse_answer):
    import torch
    rendered = [tokenizer.apply_chat_template(
        [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
    ) for p in prompts for _ in range(samples)]
    encoded = tokenizer(rendered, return_tensors="pt", padding=True, truncation=True,
                        max_length=1024).to(model.device)
    torch.manual_seed(seed)
    with torch.inference_mode():
        output = model.generate(**encoded, do_sample=True, temperature=temperature,
                                top_p=0.95, max_new_tokens=max_new_tokens,
                                pad_token_id=tokenizer.pad_token_id)
    prompt_len = encoded.input_ids.shape[1]
    texts = tokenizer.batch_decode(output[:, prompt_len:], skip_special_tokens=True)
    parsed = [parser(t) for t in texts]
    grouped = [parsed[i:i + samples] for i in range(0, len(parsed), samples)]
    generated_tokens = int((output[:, prompt_len:] != tokenizer.pad_token_id).sum().item())
    return grouped, texts, generated_tokens


def majority(samples):
    counts = Counter(samples)
    return max(range(4), key=lambda a: (counts[a], -a))


def bootstrap_delta(root_ok, mapped_ok, *, seed=0, replicates=5000):
    rng = np.random.default_rng(seed)
    delta = np.asarray(mapped_ok, dtype=float) - np.asarray(root_ok, dtype=float)
    draws = rng.choice(delta, size=(replicates, len(delta)), replace=True).mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def bootstrap_map_control(mapped_ok, gauge_ok, *, seed=0, replicates=5000):
    """Resample orbits and frozen gauge seeds as separate clusters."""
    rng = np.random.default_rng(seed)
    mapped = np.asarray(mapped_ok, dtype=float)
    gauges = np.asarray(gauge_ok, dtype=float)
    draws = []
    for _ in range(replicates):
        orbit_idx = rng.integers(0, len(mapped), len(mapped))
        gauge_idx = rng.integers(0, gauges.shape[1], gauges.shape[1])
        draws.append(mapped[orbit_idx].mean() - gauges[np.ix_(orbit_idx, gauge_idx)].mean())
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def orbit_permutations(views: int, seed: int):
    """Start with cyclic rotations so every fixed letter maps to distinct roots."""
    if not 1 <= views <= 24:
        raise ValueError("views must be between 1 and 24")
    permutations = [tuple((old + shift) % 4 for old in range(4))
                    for shift in range(min(views, 4))]
    rng = random.Random(seed)
    while len(permutations) < views:
        p = list(range(4)); rng.shuffle(p); p = tuple(p)
        if p not in permutations:
            permutations.append(p)
    return permutations


def randomized_inverse(permutation, rho):
    """Inverse of rho composed with the correct canonical-to-node map."""
    def inverse(answer):
        if answer is None:
            return None
        ungauged_node_answer = rho.index(answer)
        return permutation.index(ungauged_node_answer)
    return inverse


def run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
    ).eval()

    records = []
    mapped_token_total = root_token_total = 0
    records_path = out / "records.jsonl"
    records_path.write_text("")
    for orbit in range(args.orbits):
        item = make_item(args.seed * 100000 + orbit, args.difficulty)
        permutations = orbit_permutations(args.views, args.seed + orbit)
        prompts = [permuted_view(item, p) for p in permutations]
        view_samples, raw, tokens = generate_samples(
            model, tokenizer, prompts, samples=args.samples,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            seed=args.seed + orbit,
        )
        mapped_token_total += tokens
        inverse_maps = [lambda a, p=p: None if a is None else p.index(a) for p in permutations]
        q = canonical_pool(view_samples, inverse_maps, support=(None, 0, 1, 2, 3), alpha=args.alpha)
        mapped = max(range(4), key=lambda a: (q[a], -a))
        unmapped = majority([a for view in view_samples for a in view])

        # Matched compute root baseline uses the same total number of generations.
        root_groups, root_raw, root_tokens = generate_samples(
            model, tokenizer, [prompts[0]], samples=args.samples * args.views,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            seed=args.seed + 100000 + orbit,
        )
        root_token_total += root_tokens
        root = majority(root_groups[0])

        gauges = []
        for gauge_seed in range(args.gauge_seeds):
            gauge_maps = [inverse_maps[0]]
            for view, p in enumerate(permutations[1:], start=1):
                gauge_rng = random.Random(args.gauge_base + gauge_seed * 100 + view)
                rho = list(range(4)); gauge_rng.shuffle(rho)
                gauge_maps.append(randomized_inverse(p, tuple(rho)))
            qg = canonical_pool(view_samples, gauge_maps, support=(None, 0, 1, 2, 3), alpha=args.alpha)
            gauges.append(max(range(4), key=lambda a: (qg[a], -a)))

        canonical_samples = [[inverse_maps[i](a) for a in view]
                             for i, view in enumerate(view_samples)]
        record = {"id": item["id"], "difficulty": args.difficulty,
                  "truth": item["truth"], "root": root, "mapped": mapped,
                  "unmapped": unmapped, "root_correct": root == item["truth"],
                  "mapped_correct": mapped == item["truth"],
                  "unmapped_correct": unmapped == item["truth"],
                  "gauge_accuracy": sum(x == item["truth"] for x in gauges) / len(gauges),
                  "gauge_correct": [x == item["truth"] for x in gauges],
                  "permutations": permutations, "view_samples": view_samples,
                  "truth_support": any(item["truth"] in view for view in canonical_samples),
                  "canonical_rollout_accuracy": sum(a == item["truth"] for view in canonical_samples
                                                     for a in view) / (args.views * args.samples),
                  "parse_failure_rate": sum(a is None for v in view_samples for a in v) /
                                        (args.views * args.samples)}
        records.append(record)
        with records_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    root_ok = [r["root_correct"] for r in records]
    mapped_ok = [r["mapped_correct"] for r in records]
    root_acc = float(np.mean(root_ok)); mapped_acc = float(np.mean(mapped_ok))
    unmapped_acc = float(np.mean([r["unmapped_correct"] for r in records]))
    gauge_acc = float(np.mean([r["gauge_accuracy"] for r in records]))
    gauge_ok = np.asarray([r["gauge_correct"] for r in records], dtype=float)
    summary = {
        "model": args.model, "orbits": args.orbits, "views": args.views,
        "difficulty": args.difficulty,
        "samples_per_view": args.samples, "root_accuracy": root_acc,
        "unmapped_transformed_accuracy": unmapped_acc,
        "mapped_accuracy": mapped_acc, "mapped_minus_root": mapped_acc - root_acc,
        "paired_bootstrap_95ci": bootstrap_delta(root_ok, mapped_ok, seed=args.seed),
        "randomized_gauge_accuracy": gauge_acc,
        "correct_map_minus_randomized": mapped_acc - gauge_acc,
        "correct_map_minus_randomized_95ci": bootstrap_map_control(
            mapped_ok, gauge_ok, seed=args.seed),
        "truth_support_coverage": float(np.mean([r["truth_support"] for r in records])),
        "canonical_rollout_accuracy": float(np.mean(
            [r["canonical_rollout_accuracy"] for r in records])),
        "mean_parse_failure_rate": float(np.mean([r["parse_failure_rate"] for r in records])),
        "mapped_completion_tokens": mapped_token_total,
        "root_completion_tokens": root_token_total,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--output", default="outputs/model_pilot")
    p.add_argument("--orbits", type=int, default=24)
    p.add_argument("--views", type=int, default=4)
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--gauge-seeds", type=int, default=20)
    p.add_argument("--gauge-base", type=int, default=91000)
    p.add_argument("--difficulty", choices=tuple(DIFFICULTIES), default="extreme")
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=7)
    run(p.parse_args(argv))


if __name__ == "__main__":
    main()
