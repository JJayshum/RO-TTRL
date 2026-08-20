"""Evaluate already-trained root and mapped TTRL adapters on fresh orbits."""

import argparse
import json
from pathlib import Path

import numpy as np

from .adaptation_pilot import load_and_evaluate, summarize
from .model_pilot import bootstrap_delta


def main(argv=None):
    from transformers import AutoTokenizer
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--root-adapter", required=True)
    p.add_argument("--mapped-adapter", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--difficulty", default="easy")
    p.add_argument("--eval-orbits", type=int, default=128)
    p.add_argument("--eval-samples", type=int, default=4)
    p.add_argument("--views", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--eval-seed", type=int, default=1701)
    args = p.parse_args(argv)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    evaluations = {
        "base": load_and_evaluate(args.model, None, tokenizer, args),
        "root_ttrl": load_and_evaluate(args.model, Path(args.root_adapter), tokenizer, args),
        "mapped_ttrl": load_and_evaluate(args.model, Path(args.mapped_adapter), tokenizer, args),
    }
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    for name, rows in evaluations.items():
        with (out / f"eval_{name}.jsonl").open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    values = {name: [r["policy_pass1"] for r in rows]
              for name, rows in evaluations.items()}
    summary = {"eval_orbits": args.eval_orbits, "eval_seed": args.eval_seed,
               "evaluation": {k: summarize(v) for k, v in evaluations.items()},
               "policy_effects": {
                   "mapped_minus_base": float(np.mean(values["mapped_ttrl"]) -
                                              np.mean(values["base"])),
                   "mapped_minus_base_95ci": bootstrap_delta(
                       values["base"], values["mapped_ttrl"], seed=args.eval_seed),
                   "mapped_minus_root_ttrl": float(np.mean(values["mapped_ttrl"]) -
                                                   np.mean(values["root_ttrl"])),
                   "mapped_minus_root_ttrl_95ci": bootstrap_delta(
                       values["root_ttrl"], values["mapped_ttrl"], seed=args.eval_seed + 1),
               }}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
