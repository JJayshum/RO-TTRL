#!/usr/bin/env bash
set -euo pipefail

export RO_BENCHMARK_DATA_ROOT=/root/autodl-tmp/benchmark-data
project_root=/root/autodl-tmp/relational-orbit-ttrl
output_root=outputs/cross_family_panel_20260817
model_root=/root/autodl-tmp/cross-family-models
python_bin=/root/miniconda3/bin/python
mkdir -p "$project_root/$output_root" "$model_root"
cd "$project_root"

models=(
  "phi35|LLM-Research/Phi-3.5-mini-instruct"
  "mistral7b|LLM-Research/Mistral-7B-Instruct-v0.3"
  "gemma2b|LLM-Research/gemma-2-2b-it"
)

for spec in "${models[@]}"; do
  label=${spec%%|*}
  repo=${spec#*|}
  model_dir="$model_root/$label"
  if [[ ! -s "$model_dir/config.json" ]] || ! find "$model_dir" -name '*.safetensors' -type f -print -quit | grep -q .; then
    modelscope download "$repo" --revision master --local-dir "$model_dir" --max-workers 4
  fi
  find "$model_dir" -type f -print0 | sort -z | xargs -0 sha256sum \
    > "$output_root/${label}_MODEL_SHA256SUMS"
done

{
  date --iso-8601=seconds
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  "$python_bin" - <<'PY'
import platform
import torch
import transformers
import datasets
print("python", platform.python_version())
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("datasets", datasets.__version__)
PY
} > "$output_root/environment.txt"

for spec in "${models[@]}"; do
  label=${spec%%|*}
  model_dir="$model_root/$label"
  MODEL_DIR="$model_dir" "$python_bin" - <<'PY'
import os
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(os.environ["MODEL_DIR"])
rendered = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Reply with exactly 'Answer: A'."}],
    tokenize=False,
    add_generation_prompt=True,
)
assert rendered and tokenizer.eos_token_id is not None
print(type(tokenizer).__name__, len(rendered), tokenizer.eos_token_id, tokenizer.pad_token_id)
PY
done > "$output_root/tokenizer_preflight.log"

exclusion_args=()
while IFS= read -r manifest; do
  exclusion_args+=(--exclude-manifest "$manifest")
done < <(find \
  outputs/real_benchmark_gate \
  outputs/real_benchmark_7b_confirmation \
  outputs/real_benchmark_adaptation_seed31801 \
  outputs/real_benchmark_llama3b_confirmation \
  -name manifest.json -type f | sort)

run_one() {
  local label=$1
  local model=$2
  local benchmark=$3
  local seed=$4
  local out="$output_root/$label/$benchmark"
  if [[ -s "$out/summary.json" ]] && [[ $(wc -l < "$out/records.jsonl") -eq 64 ]]; then
    echo "SKIP complete $label $benchmark"
    return
  fi
  mkdir -p "$out"
  echo "START $(date --iso-8601=seconds) $label $benchmark"
  "$python_bin" -m relational_orbit_ttrl.benchmark_pilot \
    --model "$model" \
    --benchmark "$benchmark" \
    --output "$out" \
    --orbits 64 \
    --views 4 \
    --samples 4 \
    --sample-seed 20260817 \
    --sample-offset 0 \
    --seed "$seed" \
    --store-raw-text \
    "${exclusion_args[@]}" > "$out/run.log" 2>&1
  echo "DONE  $(date --iso-8601=seconds) $label $benchmark"
}

run_model() {
  local label=$1
  local seed_base=$2
  local model="$model_root/$label"
  run_one "$label" "$model" arc_challenge "$((seed_base + 0))"
  run_one "$label" "$model" openbookqa "$((seed_base + 101))"
  run_one "$label" "$model" mmlu "$((seed_base + 202))"

  "$python_bin" -m relational_orbit_ttrl.submission_analysis \
    "$output_root/$label"/*/records.jsonl \
    --output "$output_root/$label/analysis" \
    --seed 20260817 > "$output_root/$label/analysis.log" 2>&1

  audit_args=()
  while IFS= read -r manifest; do
    audit_args+=(--reference-manifest "$manifest")
  done < <(find \
    outputs/real_benchmark_gate \
    outputs/real_benchmark_7b_confirmation \
    outputs/real_benchmark_adaptation_seed31801 \
    outputs/real_benchmark_llama3b_confirmation \
    -name manifest.json -type f | sort)
  "$python_bin" -m relational_orbit_ttrl.compatibility_audit \
    "$output_root/$label" \
    --output "$output_root/$label/compatibility_audit.json" \
    "${audit_args[@]}" > /dev/null
}

run_model phi35 71808
run_model mistral7b 81808
run_model gemma2b 91808

"$python_bin" - <<'PY'
import json
from pathlib import Path

root = Path("outputs/cross_family_panel_20260817")
reference = Path("outputs/paper_scale_shared/qwen3b")
for label in ("phi35", "mistral7b", "gemma2b"):
    for benchmark in ("arc_challenge", "openbookqa", "mmlu"):
        def hashes(path):
            data = json.loads(path.read_text())
            return [row["content_sha256"] for row in data["items"]]
        current = hashes(root / label / benchmark / "manifest.json")
        expected = hashes(reference / benchmark / "manifest.json")
        if current != expected:
            raise SystemExit(f"shared question mismatch: {label}/{benchmark}")
print("all candidate manifests match the frozen paper-scale shared block")
PY

"$python_bin" -m relational_orbit_ttrl.generality_analysis \
  "$output_root" \
  --output "$output_root/generality_analysis.json" \
  --seed 20260817 \
  --replicates 10000 > "$output_root/generality_analysis.log" 2>&1

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$python_bin" -m pytest -q \
  > "$output_root/tests.log" 2>&1

find "$output_root" -type f \
  ! -name SHA256SUMS \
  ! -name COMPLETE \
  ! -name driver.log \
  -print0 | sort -z | xargs -0 sha256sum > "$output_root/SHA256SUMS"
date --iso-8601=seconds > "$output_root/COMPLETE"
