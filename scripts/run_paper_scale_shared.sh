#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/relational-orbit-ttrl
export RO_BENCHMARK_DATA_ROOT=/root/autodl-tmp/benchmark-data
export PYTHONUNBUFFERED=1

output_root=outputs/paper_scale_shared
mkdir -p "$output_root"

{
  date --iso-8601=seconds
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  /root/miniconda3/bin/python - <<'PY'
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
  /root/miniconda3/bin/python -m relational_orbit_ttrl.benchmark_pilot \
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

qwen3=/root/autodl-tmp/modelscope-cache/models/Qwen--Qwen2.5-3B-Instruct/snapshots/master
qwen7=/root/autodl-tmp/modelscope-cache/models/Qwen--Qwen2.5-7B-Instruct/snapshots/master
llama3=/root/autodl-tmp/modelscope-cache/models/Llama--Llama-3.2-3B-Instruct/snapshots/master

run_one qwen3b "$qwen3" arc_challenge 41808
run_one qwen3b "$qwen3" openbookqa 41909
run_one qwen3b "$qwen3" mmlu 42010
run_one qwen7b "$qwen7" arc_challenge 51808
run_one qwen7b "$qwen7" openbookqa 51909
run_one qwen7b "$qwen7" mmlu 52010
run_one llama3b "$llama3" arc_challenge 61808
run_one llama3b "$llama3" openbookqa 61909
run_one llama3b "$llama3" mmlu 62010

for label in qwen3b qwen7b llama3b; do
  /root/miniconda3/bin/python -m relational_orbit_ttrl.submission_analysis \
    "$output_root/$label"/*/records.jsonl \
    --output "$output_root/$label/analysis" \
    --seed 20260817 > "$output_root/$label/analysis.log" 2>&1
done

find "$output_root" -type f -print0 | sort -z | xargs -0 sha256sum \
  > "$output_root/SHA256SUMS"
date --iso-8601=seconds > "$output_root/COMPLETE"
