# SETUP — Local vLLM Backend

## Why vllm-router prefix-aware routing (not LiteLLM least-busy, not vLLM `--data-parallel-size`)

vLLM's automatic prefix cache is **per-replica**. A request only benefits from the cache if it lands on the replica that previously served a request with the same prefix.

- **LiteLLM `least-busy`** balances load but ignores prefix locality — the same prompt prefix may land on any of N replicas, so each replica caches 1/N of the useful entries and effective cache size shrinks linearly with replica count.
- **vLLM `--data-parallel-size N`** runs N engines in one process but round-robins requests at the engine level, with the same per-replica prefix-cache fragmentation.
- **`vllm-router` with prefix-aware routing** hashes the request prefix and pins same-prefix requests to the same replica, so the union of replica caches behaves close to one large cache.

RLM-specific reasons this matters: every turn within a run shares the workspace system prompt plus accumulated workspace history through the previous turn; recursive `rlm_query` children share large fragments of the parent's context; multi-task evals re-use the same long system prompt across hundreds of completions. Pinning by prefix is close to free wall-clock time on long runs.

## Step-by-step

### 1. Conda env

Both `vllm` and `vllm-router` are installed in `RLM_substrate` alongside the client — single env, no separate server env.

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate RLM_substrate

# Sanity check
python -c "import vllm; print('vllm', vllm.__version__)"
vllm-router --help | head -n 20    # confirm the flag spellings used below
```

> The exact flag names in §3 (`--service-discovery`, `--static-backends`, `--static-models`, `--routing-logic prefixaware`) are believed correct for the vllm-production-stack `vllm-router` CLI but should be cross-checked against `vllm-router --help` on first run — older releases used `--routing-logic prefix-aware` with a hyphen. If your build rejects `prefixaware`, swap to `prefix-aware`.

### 2. Launch vLLM replicas + router

One replica per GPU. **Do not** pass `--data-parallel-size` — we want independent processes with independent prefix caches that the router can pin to.

The primary model is **Qwen/Qwen3.5-9B** (~18 GB BF16), which fits on a single A6000 48 GB. The launch scripts handle the Qwen-specific flags (disable multimodal, reasoning parser) and probe for the highest `--max-num-seqs` that survives CUDA graph capture — necessary because Qwen3.5-9B's hybrid linear-attention architecture OOMs during graph capture at vLLM's default.

#### Option A — Multi-replica (recommended)

`setup/serve_qwen35.py` allocates ports automatically, launches all replicas in parallel, waits for each to come up (probing `--max-num-seqs` on OOM), then starts the vllm-router. Each GPU spec is a comma-separated list of device indices; multiple indices in one spec enable tensor parallelism for that replica.

```bash
# 2 replicas on GPUs 0 and 1; router on port 8000 (ports auto-allocated from 8001+)
nohup python setup/serve_qwen35.py --router-port 8000 --gpus 0 1 \
  > vllm_logs/serve_launcher.log 2>&1 &

# TP-2 replica on GPUs 0+1 plus a single replica on GPU 2; router on 8000
nohup python setup/serve_qwen35.py --router-port 8000 --gpus 0,1 2 \
  > vllm_logs/serve_launcher.log 2>&1 &

# Reuse already-running replicas (skips spin-up for ports that answer /health)
python setup/serve_qwen35.py --router-port 8000 --gpus 0 1 --replica-ports 8001 8002
```

The script checks each port with a `/health` probe before launching: live replicas are reused, dead ones are (re)launched on the corresponding GPU. This makes it safe to re-run after a partial failure or router restart.

The script prints a summary when everything is up:
```
[done] Stack is up:
  Endpoint: http://127.0.0.1:8000/v1  (model: qwen3-5-9b)
  Replica 0: http://127.0.0.1:8001/v1  gpus=0  max-num-seqs=128  pid=...
  Replica 1: http://127.0.0.1:8002/v1  gpus=1  max-num-seqs=128  pid=...
  Router pid=...
```

Verify:
```bash
curl -fsS http://127.0.0.1:8000/v1/models \
  | python -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])"
# → qwen3-5-9b
```

#### Option B — Single replica (manual / debugging)

`setup/server_qwen35_single.py` launches one replica on a specific port; you then wire up the router manually (see §3 below).

```bash
mkdir -p vllm_logs

# Replica A — GPU 0, port 8001
nohup python setup/server_qwen35_single.py --port 8001 --gpus 0 \
  > vllm_logs/replica_a_launcher.log 2>&1 &

# Replica B — GPU 1, port 8002
nohup python setup/server_qwen35_single.py --port 8002 --gpus 1 \
  > vllm_logs/replica_b_launcher.log 2>&1 &

# Tensor-parallel across two GPUs (single large model)
nohup python setup/server_qwen35_single.py --port 8001 --gpus 0,1 \
  > vllm_logs/replica_a_launcher.log 2>&1 &
```

Sanity check each replica:
```bash
for p in 8001 8002; do
  echo "--- :$p ---"
  curl -fsS http://127.0.0.1:$p/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen3-5-9b","messages":[{"role":"user","content":"Say HELLO."}],"max_tokens":8,"temperature":0}' \
    | python -c "import sys,json; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'])"
done
```

### 3. (Option B only) Launch vllm-router manually

Skip this step if you used `serve_qwen35.py` — it starts the router for you.

```bash
conda activate RLM_substrate
mkdir -p vllm_logs
nohup vllm-router \
  --host 127.0.0.1 --port 8000 \
  --worker-urls http://127.0.0.1:8001 http://127.0.0.1:8002 \
  --policy cache_aware \
  > vllm_logs/router.log 2>&1 &
```

> **Note on routing policy**: `cache_aware` is this build's prefix-locality policy. It hashes the request prefix and pins same-prefix requests to the same worker, equivalent to what older `vllm-router` releases called `prefixaware`.

Smoke-test prefix pinning: send two identical long-prompt requests back-to-back and confirm both land on the same replica (check per-replica logs) and the second reports higher prefix-cache hit rate.

### 4. First completion

```python
from rlm import RLM
from rlm.core.config import DockerConfig, LMConfig, WorkspaceConfig
from rlm.logger import RLMLogger

logger = RLMLogger(log_dir="./logs")

rlm = RLM(
    backend="vllm",
    backend_kwargs={
        "model_name": "qwen3-5-9b",
        "base_url": "http://127.0.0.1:8000/v1",
        "api_key": "EMPTY",
    },
    workspace_config=WorkspaceConfig(
        docker=DockerConfig(cleanup_mode="keep"),
        lm=LMConfig(enable_thinking=False),
    ),
    logger=logger,
    verbose=True,
)

result = rlm.completion(
    "Compute the first 100 prime numbers and write them, one per line, to primes.txt. Then call final."
)
print(result.response)
```

### 5. Cleanup

```bash
pkill -f 'vllm-router'
pkill -f 'vllm serve'

# Remove kept workspaces (root-owned files; needs sudo)
sudo rm -rf ~/.rlm/workspaces/*

# Remove model cache (~18 GB)
rm -rf /data/nwei/rlm_substrate/models/models--Qwen--Qwen3.5-9B
```
