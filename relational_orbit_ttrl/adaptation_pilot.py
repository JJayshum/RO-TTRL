"""Small, matched-compute LoRA adaptation pilot for Relational-Orbit TTRL."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import json
from pathlib import Path

import numpy as np

from .model_pilot import (
    bootstrap_delta, generate_samples, make_item, majority, orbit_permutations,
    permuted_view,
)
from .pooling import canonical_pool, leave_one_out_rewards


SUPPORT = (None, 0, 1, 2, 3)


def inverse_maps(permutations):
    return [lambda a, p=p: None if a is None else p.index(a) for p in permutations]


def build_rollout_data(model, tokenizer, args, method):
    examples = []
    stats = {"orbits": args.train_orbits, "sequences": 0, "parse_failures": 0}
    for orbit in range(args.train_orbits):
        item = make_item(args.train_seed * 100000 + orbit, args.difficulty)
        permutations = orbit_permutations(args.views, args.train_seed + orbit)
        prompts = [permuted_view(item, p) for p in permutations]
        if method == "mapped":
            samples, raw, _ = generate_samples(
                model, tokenizer, prompts, samples=args.samples,
                max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                seed=args.train_seed + orbit,
            )
            maps = inverse_maps(permutations)
            rewards = leave_one_out_rewards(samples, maps, support=SUPPORT, alpha=args.alpha)
            raw_groups = [raw[i * args.samples:(i + 1) * args.samples]
                          for i in range(args.views)]
            for prompt, parsed, texts, group_rewards in zip(prompts, samples, raw_groups, rewards):
                centered = np.asarray(group_rewards) - np.mean(group_rewards)
                for answer, text, advantage in zip(parsed, texts, centered):
                    examples.append((prompt, text, float(advantage)))
                    stats["parse_failures"] += answer is None
        elif method == "root":
            count = args.views * args.samples
            samples, raw, _ = generate_samples(
                model, tokenizer, [prompts[0]], samples=count,
                max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                seed=args.train_seed + 100000 + orbit,
            )
            rewards = leave_one_out_rewards(samples, [lambda a: a], support=SUPPORT,
                                            alpha=args.alpha)[0]
            centered = np.asarray(rewards) - np.mean(rewards)
            for answer, text, advantage in zip(samples[0], raw, centered):
                examples.append((prompts[0], text, float(advantage)))
                stats["parse_failures"] += answer is None
        else:
            raise ValueError(method)
    stats["sequences"] = len(examples)
    stats["parse_failure_rate"] = stats.pop("parse_failures") / len(examples)
    stats["advantage_std"] = float(np.std([x[2] for x in examples]))
    return examples, stats


def encode_examples(tokenizer, examples, max_length):
    encoded = []
    for prompt, completion, advantage in examples:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer(rendered, add_special_tokens=False).input_ids
        completion_ids = tokenizer(completion + tokenizer.eos_token,
                                   add_special_tokens=False).input_ids
        available = max_length - len(prompt_ids)
        completion_ids = completion_ids[:max(1, available)]
        ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + completion_ids
        encoded.append((ids, labels, advantage))
    return encoded


def collate(batch, pad_token_id, device):
    import torch
    width = max(len(x[0]) for x in batch)
    input_ids, labels, attention, advantages = [], [], [], []
    for ids, labs, advantage in batch:
        pad = width - len(ids)
        input_ids.append(ids + [pad_token_id] * pad)
        labels.append(labs + [-100] * pad)
        attention.append([1] * len(ids) + [0] * pad)
        advantages.append(advantage)
    return (torch.tensor(input_ids, device=device), torch.tensor(labels, device=device),
            torch.tensor(attention, device=device), torch.tensor(advantages, device=device))


def sequence_log_probs(logits, input_ids, labels):
    import torch
    shifted_logits = logits[:, :-1].float()
    targets = input_ids[:, 1:]
    mask = labels[:, 1:] != -100
    token_logp = torch.log_softmax(shifted_logits, dim=-1).gather(
        -1, targets.unsqueeze(-1)).squeeze(-1)
    return (token_logp * mask).sum(-1), token_logp, mask


def train_adapter(model_path, tokenizer, examples, output, args):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True)
    model.config.use_cache = False
    config = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_rank * 2,
                        lora_dropout=0.0, target_modules=["q_proj", "v_proj"],
                        task_type="CAUSAL_LM")
    model = get_peft_model(model, config)
    model.train()
    encoded = encode_examples(tokenizer, examples, args.train_max_length)
    rng = np.random.default_rng(args.train_seed)
    order = rng.permutation(len(encoded))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=args.learning_rate)
    optimizer.zero_grad(set_to_none=True)
    losses, pg_losses, kl_losses = [], [], []
    device = next(model.parameters()).device
    for step, start in enumerate(range(0, len(order), args.batch_size)):
        idx = order[start:start + args.batch_size]
        batch = [encoded[int(i)] for i in idx]
        input_ids, labels, attention, advantages = collate(
            batch, tokenizer.pad_token_id, device)
        with torch.no_grad():
            ctx = model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext()
            with ctx:
                ref_logits = model(input_ids=input_ids, attention_mask=attention).logits
                ref_seq, ref_token, mask = sequence_log_probs(ref_logits, input_ids, labels)
        logits = model(input_ids=input_ids, attention_mask=attention).logits
        seq_logp, token_logp, mask = sequence_log_probs(logits, input_ids, labels)
        pg_loss = -(advantages * seq_logp).mean()
        log_ratio = token_logp - ref_token
        kl = (((torch.exp(log_ratio) - 1.0) - log_ratio) * mask).sum() / mask.sum().clamp_min(1)
        loss = (pg_loss + args.kl_beta * kl) / args.gradient_accumulation
        loss.backward()
        if (step + 1) % args.gradient_accumulation == 0 or start + args.batch_size >= len(order):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()) * args.gradient_accumulation)
        pg_losses.append(float(pg_loss.detach())); kl_losses.append(float(kl.detach()))
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    result = {"steps": len(losses), "mean_loss": float(np.mean(losses)),
              "mean_pg_loss": float(np.mean(pg_losses)), "mean_kl": float(np.mean(kl_losses))}
    del optimizer, model
    torch.cuda.empty_cache(); gc.collect()
    return result


def evaluate(model, tokenizer, args):
    rows = []
    for orbit in range(args.eval_orbits):
        item = make_item(args.eval_seed * 100000 + orbit, args.difficulty)
        permutations = orbit_permutations(args.views, args.eval_seed + orbit)
        prompts = [permuted_view(item, p) for p in permutations]
        samples, _, _ = generate_samples(
            model, tokenizer, prompts, samples=args.eval_samples,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            seed=args.eval_seed + orbit)
        maps = inverse_maps(permutations)
        canonical = [[maps[i](a) for a in view] for i, view in enumerate(samples)]
        q = canonical_pool(samples, maps, support=SUPPORT, alpha=args.alpha)
        mapped = max(range(4), key=lambda a: (q[a], -a))
        root_valid = samples[0]
        rows.append({"id": item["id"], "truth": item["truth"],
                     "policy_pass1": sum(a == item["truth"] for a in root_valid) / len(root_valid),
                     "root_majority": majority(root_valid) == item["truth"],
                     "mapped": mapped == item["truth"],
                     "truth_support": any(item["truth"] in v for v in canonical),
                     "parse_failure": sum(a is None for v in samples for a in v) /
                                      (args.views * args.eval_samples)})
    return rows


def load_and_evaluate(model_path, adapter_path, tokenizer, args):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True).eval()
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path).eval()
    rows = evaluate(model, tokenizer, args)
    del model
    torch.cuda.empty_cache(); gc.collect()
    return rows


def summarize(rows):
    return {key: float(np.mean([r[key] for r in rows])) for key in
            ("policy_pass1", "root_majority", "mapped", "truth_support", "parse_failure")}


def run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True).eval()
    mapped_examples, mapped_rollouts = build_rollout_data(base, tokenizer, args, "mapped")
    root_examples, root_rollouts = build_rollout_data(base, tokenizer, args, "root")
    del base; torch.cuda.empty_cache(); gc.collect()

    mapped_path = out / "mapped_adapter"
    root_path = out / "root_adapter"
    mapped_train = train_adapter(args.model, tokenizer, mapped_examples, mapped_path, args)
    root_train = train_adapter(args.model, tokenizer, root_examples, root_path, args)

    evaluations = {
        "base": load_and_evaluate(args.model, None, tokenizer, args),
        "root_ttrl": load_and_evaluate(args.model, root_path, tokenizer, args),
        "mapped_ttrl": load_and_evaluate(args.model, mapped_path, tokenizer, args),
    }
    for name, rows in evaluations.items():
        with (out / f"eval_{name}.jsonl").open("w") as f:
            for row in rows: f.write(json.dumps(row) + "\n")
    summary = {"model": args.model, "difficulty": args.difficulty,
               "train_orbits": args.train_orbits, "eval_orbits": args.eval_orbits,
               "mapped_rollouts": mapped_rollouts, "root_rollouts": root_rollouts,
               "mapped_train": mapped_train, "root_train": root_train,
               "evaluation": {k: summarize(v) for k, v in evaluations.items()}}
    base_pass = [r["policy_pass1"] for r in evaluations["base"]]
    root_pass = [r["policy_pass1"] for r in evaluations["root_ttrl"]]
    mapped_pass = [r["policy_pass1"] for r in evaluations["mapped_ttrl"]]
    summary["policy_effects"] = {
        "mapped_minus_base": float(np.mean(mapped_pass) - np.mean(base_pass)),
        "mapped_minus_base_95ci": bootstrap_delta(base_pass, mapped_pass, seed=args.eval_seed),
        "mapped_minus_root_ttrl": float(np.mean(mapped_pass) - np.mean(root_pass)),
        "mapped_minus_root_ttrl_95ci": bootstrap_delta(root_pass, mapped_pass,
                                                        seed=args.eval_seed + 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output", default="outputs/adaptation_pilot")
    p.add_argument("--difficulty", default="easy")
    p.add_argument("--train-orbits", type=int, default=16)
    p.add_argument("--eval-orbits", type=int, default=32)
    p.add_argument("--views", type=int, default=4)
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--eval-samples", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument("--train-max-length", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--train-seed", type=int, default=1201)
    p.add_argument("--eval-seed", type=int, default=1301)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--kl-beta", type=float, default=0.02)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation", type=int, default=8)
    run(p.parse_args(argv))


if __name__ == "__main__":
    main()
