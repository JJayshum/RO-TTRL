# Pilot protocol and compute gate

## Current CPU gate

Run the controlled suite before any model inference:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python -m relational_orbit_ttrl.cli --orbits 10000 --rollouts 64 --seed 0
```

The gate is descriptive at this stage: the dispersed-shortcut condition should show a positive mapped-minus-root effect, while the stable-shortcut and coherently-wrong controls should not show a systematic repair. The simulator is intentionally label-aware only for evaluation; the pooling implementation itself receives samples, maps, and a declared support.

## Hardware plan

No GPU or SSH access is required for the current controlled pilot. A local machine with 4 CPU cores, 16 GB RAM, and at least 5 GB free disk is sufficient. The simulator is small and uses NumPy only.

GPU access becomes necessary only for the next stage: loading an open 3B model, generating rollout traces, and running parameter-efficient test-time updates. Please do not send credentials yet. When the CPU gate is accepted, I will ask for a temporary SSH target with:

* 1 NVIDIA H100 80 GB (preferred) or 1 A100 80 GB; 24 GB VRAM is workable only for a quantized 3B smoke test.
* CUDA 12.x, a recent NVIDIA driver, and Python 3.10/3.11.
* 80 GB free local disk for model/cache/checkpoints and 250 GB for the first pilot's rollouts.
* 8 vCPU and 64 GB system RAM.
* Outbound access to the model/package registries, or pre-cached model weights.

The 7B matched-compute stage should use one H100 80 GB for a quantized/PEFT pilot, or two A100/H100 80 GB GPUs for comfortable full-precision headroom. The proposal's full-study ceiling is approximately 1 TB of temporary rollout storage; that is not needed for the CPU gate.

When GPU work is ready, send only the SSH hostname/IP, port, username, and the path to an already-authorized key or agent-forwarding setup. Do not paste private key material into chat.
