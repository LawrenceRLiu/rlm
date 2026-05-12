#!/usr/bin/env bash
# Overnight Terminal Bench 2.0 run for Qwen3.5-9B across 4 sharded vLLM
# replicas. CUDA indices under default FASTEST_FIRST ordering:
#   0,1 -> A100 80GB (nvidia-smi indices 4,5)
#   6,7 -> A6000 48GB (nvidia-smi indices 6,7, direct)
#
# Layout:
#   shard 0  CUDA 0  port 8001
#   shard 1  CUDA 1  port 8002
#   shard 2  CUDA 6  port 8003
#   shard 3  CUDA 7  port 8004
#
# Each shard serves its own vLLM replica and runs its own slice of the 89
# Terminal-Bench-2 tasks. They share HF_HUB_CACHE so the 20 GB weights
# download exactly once. Results are written under
# eval/terminal_bench/results/qwen-3-5-9b/.
#
# Usage (run inside tmux so SSH drop doesn't kill it):
#   bash _setup_runs/run_tb2_qwen_shards.sh

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

TASKS_ROOT="${TASKS_ROOT:-third_party/terminal-bench-2/tasks}"
MODEL_ALIAS="qwen-3-5-9b"
OUTPUT_DIR="eval/terminal_bench/results/${MODEL_ALIAS}"
MAX_ITER="${MAX_ITER:-30}"

GPUS=(0 1 6 7)
PORTS=(8001 8002 8003 8004)
M="${#GPUS[@]}"

LOG_DIR="${REPO_ROOT}/_setup_runs/logs"
mkdir -p "${LOG_DIR}"

if [[ ! -d "${TASKS_ROOT}" ]]; then
    echo "ERROR: TASKS_ROOT=${TASKS_ROOT} does not exist" >&2
    echo "Vendor it first:  git submodule add https://github.com/laude-institute/terminal-bench-2 third_party/terminal-bench-2" >&2
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

# Bring up all M vLLM replicas first, waiting for each to report /v1/models.
echo "=== bringing up ${M} vLLM replicas ==="
for ((i=0; i<M; i++)); do
    GPU="${GPUS[i]}" PORT="${PORTS[i]}" bash _setup_runs/serve_one_qwen.sh "${MODEL_ALIAS}"
done

# Launch M runner shards concurrently, one per replica.
echo "=== launching ${M} runner shards ==="
conda activate RLM_substrate

pids=()
for ((i=0; i<M; i++)); do
    shard_log="${LOG_DIR}/tb2_shard_${i}.log"
    : > "${shard_log}"
    python -m eval.terminal_bench.runner \
        --tasks-root "${TASKS_ROOT}" \
        --shard-index "${i}" --num-shards "${M}" \
        --resume --rmi-after \
        --backend openai \
        --model "${MODEL_ALIAS}" \
        --base-url "http://127.0.0.1:${PORTS[i]}/v1" \
        --api-key EMPTY \
        --output-dir "${OUTPUT_DIR}" \
        --max-iterations "${MAX_ITER}" \
        > "${shard_log}" 2>&1 &
    pids+=("$!")
    echo "shard ${i}: pid=$! log=${shard_log}"
done

echo "=== all shards launched; waiting for completion ==="
fail=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        echo "shard pid=${pid} exited non-zero" >&2
        fail=1
    fi
done

echo "=== overnight run complete ==="
echo "summary: cat ${OUTPUT_DIR}/summary.jsonl"
echo "pass count: jq -s 'map(select(.passed)) | length' ${OUTPUT_DIR}/summary.jsonl"
exit "${fail}"
