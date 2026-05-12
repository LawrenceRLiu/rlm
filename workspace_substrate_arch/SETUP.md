# SETUP — Local vLLM Backend

## Purpose & audience

This is the playbook for someone who clones `rlm_substrate` on a fresh GPU box and wants to actually run an `RLM(...).completion(...)` end-to-end against a self-hosted model — not against a managed API. It documents the literal commands that worked (verified 2026-05-09 on a host with 2× A100 80 GB + 6× RTX A6000 48 GB), the failure modes I hit, and the schema/log layout the visualizer consumes.

The reference model used for the bring-up is **`google/gemma-4-31B-it`** (BF16, ≈62 GB weights, Apache 2.0). Per `Todo.md` the Gemma family is the proven format-follower for RLM substrates; Qwen3.6 27B / 35B-A3B is interesting as a *secondary* run since it failed the format on the old REPL substrate.

## Topology

Three independent runtimes talk over loopback. The model serving stack fans out to multiple replicas behind a load balancer; RLM only sees a single OpenAI-compatible endpoint.

```
[ container: model code ]  --enqueue/poll-->  [ broker (Flask, in-container) ]
                                                    ^   pulled by host poller
                                                    |
[ host: LMHandler (TCP) ] <----- forwards LM call ---+
        |
        v   OpenAI-compatible HTTP, single endpoint
[ host: litellm proxy            127.0.0.1:8000/v1 ]
        |        \         \
        v         v         v   least-busy (asymmetric capacity)
   [vllm :8001] [vllm :8002] [vllm :8003]
       |            |            |
       v            v            v
    [GPU 4]      [GPU 5]    [GPUs 2+3 TP=2]
    A100 80GB    A100 80GB    2× A6000 48GB
```

**Why three replicas?** RLM's `recursion.max_concurrent_subcalls` defaults to 5, so a fan-out turn produces parallel LM calls; multiple replicas + vLLM continuous batching keep them moving.

**Why `least-busy`?** Per-replica throughput is asymmetric — A100 single-card ≈ 1.5–2× faster than A6000 TP=2. Round-robin would make replica C the straggler; `least-busy` self-balances.

**Why TP=2 on the A6000s?** Gemma 4 31B BF16 (≈62 GB) doesn't fit on a single 48 GB card; tensor-parallel across both is the cleanest BF16 option without quantization.

## Prerequisites

- Linux host with NVIDIA GPUs and a CUDA 12.x driver. Verified on driver 535.216.01.
- Docker daemon running, current user in the `docker` group.
  - **Permissions gotcha:** if a `docker` command returns "permission denied" even after group add, your shell's group set is stale (a long-lived IDE/editor server pre-dates the group change). Run `newgrp docker` and retry. **Do not** `sudo`.
- `conda` (miniconda or mamba) available on PATH.
- Python ≥3.11 in the env that runs `rlm` (we use 3.12).
- Free TCP ports: 8000 (litellm), 8001/8002/8003 (vLLM replicas), 3001 (visualizer).
- HuggingFace account with the model license accepted (Gemma 4 is Apache 2.0 — no license click required; Gemma 3 is *gated* and needs license acceptance on the model page).
- ≥80 GB free disk for the model cache.

## Step-by-step setup

### 1. Two conda envs

`rlm` and the vLLM server live in **separate** envs to avoid a torch/CUDA dep collision and to keep `import rlm` GPU-free.

```bash
# rlm client
conda create -n RLM_substrate python=3.12 -y
conda activate RLM_substrate
python -m pip install -e .                                  # core deps
python -m pip install pytest-asyncio pytest-cov ruff pre-commit

# vLLM server (separate env)
conda create -n RLM_vllm_server python=3.12 -y
conda activate RLM_vllm_server
python -m pip install 'vllm==0.19.1' 'litellm[proxy]'
python -m pip install -U 'transformers>=5.8.0' 'huggingface-hub>=1.14.0'
```

Why `vllm==0.19.1` and not the latest 0.20.x: see Issue Log entry "vLLM version pin." Why `transformers>=5.8.0`: the older 4.x line doesn't recognize `model_type=gemma4`.

> Sanity check after install:
> ```
> conda run -n RLM_substrate python -c "import rlm; print(rlm.__file__)"
> conda run -n RLM_vllm_server python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
> ```
> Expect torch `2.10.0+cu128` (or `+cu126`, depending on which torch wheel is current), `True`, and the actual device count.

### 2. Build the workspace Docker image

```bash
make build-image                  # tag: rlm-workspace:0.1.0
docker images | grep rlm-workspace
```

`Dockerfile` is `python:3.11-slim` plus numpy/pandas/scipy/etc. plus the in-container Flask broker as PID 1. The build is cache-friendly (≈1 min on a populated cache, ≈3 min cold).

### 3. Download the model

```bash
conda activate RLM_vllm_server
huggingface-cli download google/gemma-4-31B-it
```

≈59 GB; check `~/.cache/huggingface/hub/models--google--gemma-4-31B-it/`.

### 4. Launch three vLLM replicas

A note on flags:
- `--max-num-batched-tokens 8192` is **required** for Gemma 4 — its multimodal-bidirectional attention forces `--disable_chunked_mm_input`, and the default `max_num_batched_tokens=2048` is below the 2496-token MM item ceiling.
- `--max-num-seqs 64` on replica C — TP=2 BF16 of a 31B on 48 GB cards OOMs during sampler warm-up at the default `max_num_seqs=256`.
- `--reasoning-parser gemma4` — routes Gemma 4's `<|channel|>thought…<|channel|>` reasoning blocks into the response's separate `reasoning_content` field instead of leaving them inline in `content`. The substrate's `LMConfig.enable_thinking=True` default needs this server flag to actually surface reasoning into `WorkspaceIteration.reasoning`. Without it, reasoning lands as raw special tokens in `content` and the substrate's `strip_reasoning_blocks` discards it instead of capturing.

```bash
# Replica A — A100 (GPU 4)
CUDA_VISIBLE_DEVICES=4 CUDA_DEVICE_ORDER=PCI_BUS_ID nohup bash -c '
  source "$(conda info --base)/etc/profile.d/conda.sh" &&
  conda activate RLM_vllm_server &&
  vllm serve google/gemma-4-31B-it \
    --port 8001 --host 127.0.0.1 \
    --max-model-len 32768 --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.90 --dtype bfloat16 \
    --reasoning-parser gemma4
' > vllm_logs/replicaA.log 2>&1 &

# Replica B — A100 (GPU 5)
CUDA_VISIBLE_DEVICES=5 CUDA_DEVICE_ORDER=PCI_BUS_ID nohup bash -c '
  source "$(conda info --base)/etc/profile.d/conda.sh" &&
  conda activate RLM_vllm_server &&
  vllm serve google/gemma-4-31B-it \
    --port 8002 --host 127.0.0.1 \
    --max-model-len 32768 --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.90 --dtype bfloat16 \
    --reasoning-parser gemma4
' > vllm_logs/replicaB.log 2>&1 &

# Replica C — A6000 ×2 TP=2 (GPUs 2,3)
CUDA_VISIBLE_DEVICES=2,3 CUDA_DEVICE_ORDER=PCI_BUS_ID nohup bash -c '
  source "$(conda info --base)/etc/profile.d/conda.sh" &&
  conda activate RLM_vllm_server &&
  vllm serve google/gemma-4-31B-it \
    --port 8003 --host 127.0.0.1 \
    --tensor-parallel-size 2 \
    --max-model-len 16384 --max-num-batched-tokens 8192 \
    --max-num-seqs 64 --gpu-memory-utilization 0.85 --dtype bfloat16 \
    --reasoning-parser gemma4
' > vllm_logs/replicaC.log 2>&1 &
```

For Qwen3-family models served via `_setup_runs/serve_one_qwen.sh`, pass `--reasoning-parser qwen3`; the same rationale applies for `<think>...</think>` blocks.

Each replica takes 2–4 minutes to load. Watch for `Application startup complete.` in the log. Per-replica peak GPU memory (with the flags above): ~77 GB on each A100, ~43 GB per A6000.

Sanity check each replica:
```bash
for p in 8001 8002 8003; do
  echo "--- :$p ---"
  curl -fsS http://127.0.0.1:$p/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"google/gemma-4-31B-it","messages":[{"role":"user","content":"Say HELLO."}],"max_tokens":8,"temperature":0}' \
    | python -c "import sys,json; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'])"
done
```

### 5. Launch the litellm proxy

`litellm_config.yaml`:

```yaml
model_list:
  - model_name: gemma-4-31b
    litellm_params: {model: openai/google/gemma-4-31B-it, api_base: http://127.0.0.1:8001/v1, api_key: EMPTY}
  - model_name: gemma-4-31b
    litellm_params: {model: openai/google/gemma-4-31B-it, api_base: http://127.0.0.1:8002/v1, api_key: EMPTY}
  - model_name: gemma-4-31b
    litellm_params: {model: openai/google/gemma-4-31B-it, api_base: http://127.0.0.1:8003/v1, api_key: EMPTY}

router_settings:
  routing_strategy: least-busy
  num_retries: 2
  request_timeout: 600   # warns "not a valid argument" — harmless on this litellm version
```

```bash
conda run -n RLM_vllm_server litellm \
  --config litellm_config.yaml \
  --port 8000 --host 127.0.0.1
```

Verify:
```bash
curl -fsS http://127.0.0.1:8000/v1/models | python -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])"
# → gemma-4-31b
```

### 6. First completion — minimal copy-paste

```python
from rlm import RLM
from rlm.core.config import DockerConfig, WorkspaceConfig
from rlm.logger import RLMLogger

logger = RLMLogger(log_dir="./logs")

rlm = RLM(
    backend="vllm",
    backend_kwargs={
        "model_name": "gemma-4-31b",                  # litellm alias, NOT the HF id
        "base_url": "http://127.0.0.1:8000/v1",
        "api_key": "EMPTY",
    },
    workspace_config=WorkspaceConfig(docker=DockerConfig(cleanup_mode="keep")),
    logger=logger,
    verbose=True,
)

result = rlm.completion(
    "Compute the first 100 prime numbers and write them, one per line, to primes.txt. Then call final."
)
print(result.response)
```

Expect: ≈15 s wall clock, 4 LM calls, `~/.rlm/workspaces/run_<id>/primes.txt` containing 100 primes starting `2,3,5,7,…`.

> **Common typo trap:** `backend_kwargs.model_name` is the litellm-proxy alias `gemma-4-31b`, not the HF id `google/gemma-4-31B-it`. Litellm rejects requests where `model` doesn't match its `model_list` aliases.

## Running the test suite

```bash
conda activate RLM_substrate

# Pure unit tests (no Docker, no network) — 107 tests, ~4 s
pytest tests/ -v --ignore=tests/clients --ignore=tests/test_docker_workspace.py

# Docker integration tests — 15 tests, ~40 s. Includes the only end-to-end
# rlm_query exercise in the repo (test_rlm_query_end_to_end_with_mock_lm).
# Auto-skips if the workspace image isn't built.
pytest tests/test_docker_workspace.py -v
```

`tests/clients/` is excluded because it requires real provider API keys (matches `.github/workflows/test.yml`).

## Tracing & inspecting a run

`RLMLogger(log_dir=...)` writes one JSONL file per run as
`rlm_<YYYY-MM-DD>_<HH-MM-SS>_<uuid>.jsonl`. Each file has:

- one **metadata** header line: `{"type":"metadata", "root_model":"...", "backend":"...", "max_depth":N, ...}`
- one **iteration** line per turn: `{"type":"iteration", "iteration":N, "actions":[...], "observations":[...], "snapshot":{...}, "parse_attempts":[...], "iteration_time":<s>, "final_answer":"..."}`

Visualizer (`visualizer/`):

```bash
conda activate RLM_substrate
conda install -y -c conda-forge "nodejs>=20"     # if node isn't already on PATH
cd visualizer && npm install
npm run dev                                       # → http://localhost:3001
```

Then load any `./logs/*.jsonl` file via the in-app picker. The TS types in `visualizer/src/lib/types.ts` mirror the Python `to_dict` output 1:1; if you change a Python `to_dict`, update the TS types in the same PR.

Per-turn workspace snapshots live at `~/.rlm/workspaces/run_<id>/.git/` — one commit per turn. To inspect a specific turn:

```bash
cd ~/.rlm/workspaces/run_<id>
git log --oneline                 # one entry per turn
git show <sha> --stat             # files changed that turn
```

## Performance baseline (Gemma 4 31B BF16, all 3 replicas live)

Measured 2026-05-09. `cleanup_mode="keep"`, default `WorkspaceConfig`.

| Task | Wall clock | LM calls | Input tok | Output tok | Pass? |
|---|---|---|---|---|---|
| 3a — first 100 primes | 15.4 s | 4 | 4,076 | 258 | ✅ primes.txt has 100 entries |
| 3b — merge sort + 5 pytest tests | 84.4 s | 8 | 12,510 | 1,746 | ✅ 5/5 tests pass inside container |
| 3c — `rlm_query` fan-out × 5 | 90.9 s | 27 | 33,615 | 1,675 | ✅ 5 child workspaces, 5 artifacts copied back, collated.txt correct |
| 3d — extract function sigs from 50 KB corpus | 48.4 s | 4 | 16,468 | 918 | ✅ 420 functions (ground truth: 420), 42 modules (43 module headers — model excluded the truncated last one) |

Notes:
- Decode rate for Gemma 4 31B BF16 on a single A100 80 GB is roughly 15–25 tok/s at bs=1; the 3-replica fan-out triples that on parallel-fanout workloads (3c).
- 0 parse retries and 0 observation spills across all four runs — the substrate didn't have to fall back.
- Run scripts: `_setup_runs/run_3{a,b,c,d}_*.py`. Logs: `_setup_runs/logs/`.

## Issues found in the scaffold

Tagged `BLOCKER` / `BUG` / `ROUGH-EDGE` / `NOTE`. Patches were avoided unless trivial.

- **`BLOCKER` — the README's `uv` quick-setup needs `uv` already installed.** README's "Manual Setup" runs `curl https://astral.sh/uv/install.sh | sh` then `uv venv …`. On a host without `uv`, that one curl-pipe-sh step is the install bootstrap; the README implies `uv` is "the" way to install. CLAUDE.md does say `python -m pip install` is acceptable; I used that path. **Suggested fix:** README should call out the bootstrap or offer the `python -m pip` path as the primary alternative.

- **`BLOCKER` — vLLM version pin matters.** The plan/CLAUDE.md don't pin a specific vLLM. The latest stable `vllm==0.20.1` ships only **CUDA 13** wheels (`libcudart.so.13`), which require driver ≥ 545; this host runs driver 535. Conversely, `vllm==0.18` (which other envs on this host use) lacks the `Gemma4ForCausalLM` architecture entirely. **`vllm==0.19.1` is the working sweet spot:** ships torch 2.10 + cu12 nvidia wheels, has `Gemma4ForCausalLM`/`Gemma4ForConditionalGeneration` registered, and accepts `transformers>=5.5.1`. Suggested fix: pin in the README's "Local models / vLLM" section.

- **`BLOCKER` — `transformers>=5.8.0` required for Gemma 4.** The model config's `model_type` is `gemma4`, which `transformers==4.57.x` (the latest 4.x) doesn't recognize. The error is loud but only surfaces at vLLM startup, not install. Bumping to `transformers>=5.8.0` resolves it; vLLM 0.19.1's pin (`transformers!=5.0..5.4,!=5.5.0,>=4.56.0`) accepts 5.5.1+.

- **`ROUGH-EDGE` — Gemma 4 forces `--max-num-batched-tokens >= 2496`.** vLLM logs `Forcing --disable_chunked_mm_input for models with multimodal-bidirectional attention.` and then refuses to start with the default 2048 because that's below `max_tokens_per_mm_item=2496`. Symptom: `ValueError: Chunked MM input disabled but max_tokens_per_mm_item (2496) is larger than max_num_batched_tokens (2048).` Set `--max-num-batched-tokens 8192` (or any value ≥ 2496) on every Gemma-4 replica.

- **`ROUGH-EDGE` — sampler warm-up OOMs on A6000 TP=2 at default `max_num_seqs`.** Replica C OOMs during `_dummy_sampler_run` with 256 dummy requests on Gemma 4 31B BF16 / 2× A6000 / 32K context. Lower `--max-num-seqs 64` and/or `--gpu-memory-utilization 0.85`. Single-card A100 80 GB doesn't hit this.

- **`ROUGH-EDGE` — RLM's `vllm` backend takes the litellm-proxy alias, NOT the HF id.** When fronted by litellm, `backend_kwargs.model_name` must match the proxy's `model_list[*].model_name` (e.g. `gemma-4-31b`), not the underlying HF id. Mismatch produces `400 Invalid model name passed in model=…`. Worth a one-liner in the README.

- **`ROUGH-EDGE` — `litellm` warns `Key 'request_timeout' is not a valid argument for Router.__init__()`.** The plan's example config has `request_timeout: 600` under `router_settings`. litellm 1.83 ignores it harmlessly but emits a warning. Use `timeout: 600` instead, or omit if defaults are fine.

- **`ROUGH-EDGE` — root-owned files in workspaces.** Files written from inside the container land owned by `root:root` (the container's default uid; see `_setup_runs` workspace inspections — `primes.txt` is `root:root`). If `cleanup_mode="keep"` is on and you later want to `rm -rf ~/.rlm/workspaces/<id>` as your user, you can't without `sudo`. Worth either documenting or running the container with `--user $(id -u):$(id -g)`.

- **`NOTE` — no `examples/` or `scripts/` directory in the repo.** The README's first-completion snippet is the only runnable example. The four scripts I authored under `_setup_runs/run_3{a,b,c,d}_*.py` exercise the substrate end-to-end (smoke / coding / recursion / long-context) and could be cleaned up into `examples/` if there's interest.

- **`NOTE` — `host.docker.internal` is the implicit host-side endpoint.** The container runs with `--add-host=host.docker.internal:host-gateway` (`rlm/environments/docker_workspace.py`), and the broker forwards LM requests to the host this way. Nothing the user has to configure, but worth knowing if you ever swap docker for podman or rootless docker (where `host-gateway` may not work the same way).

- **`NOTE` — first metadata line was written with the HF id, not the alias.** My initial Phase 3a run had `model_name="google/gemma-4-31B-it"` in the metadata header (subsequent runs use `gemma-4-31b`). Both files parse correctly; just don't be confused if older logs use the HF id.

- **`NOTE` — disk pressure.** With one Gemma 4 31B model cached (~59 GB), kept workspaces (a few MB each), and the Docker image (~700 MB), the host went from 165 GB → ~100 GB free. Plan accordingly if you cache multiple 30B+ models.

## Cleanup

```bash
# stop the proxy + replicas + visualizer
pkill -f 'litellm --config'
pkill -f 'vllm serve'
pkill -f 'next dev'

# remove kept workspaces (root-owned files; needs sudo)
sudo rm -rf ~/.rlm/workspaces/*

# remove docker image
docker rmi rlm-workspace:0.1.0

# remove model cache (frees ~59 GB)
rm -rf ~/.cache/huggingface/hub/models--google--gemma-4-31B-it
```
