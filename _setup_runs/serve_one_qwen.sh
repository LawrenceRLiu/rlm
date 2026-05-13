#!/usr/bin/env bash
# Serve ONE Qwen model on a single GPU.
#
# CUDA ordering: default FASTEST_FIRST (we explicitly do NOT pin
# CUDA_DEVICE_ORDER=PCI_BUS_ID). Under FASTEST_FIRST on this host, CUDA 0,1
# are the A100s and CUDA 6,7 are the A6000s at PCI 6,7. The sharded
# launcher relies on this ordering when picking GPUs 0,1,6,7.
#
# Caches weights under /data/nwei/rlm_substrate/models (1.3 TB free on /data),
# NOT ~/.cache/huggingface (root partition is tight; previous parallel-download
# attempt OOM'd the box).
#
# Usage: GPU=<n> PORT=<p> serve_one_qwen.sh <alias>
#   GPU defaults to 0 (an A100 under FASTEST_FIRST).
#   PORT defaults to 8001.
# alias one of: qwen-3-32b qwen-3-8b qwen-3-5-27b qwen-3-5-9b

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: GPU=<n> PORT=<p> $0 <alias>" >&2; exit 2
fi
ALIAS="$1"
GPU="${GPU:-0}"
PORT="${PORT:-8001}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO_ROOT}/_setup_runs/vllm_logs"
mkdir -p "${LOG_DIR}"

export HF_HUB_CACHE="/data/nwei/rlm_substrate/models"
mkdir -p "${HF_HUB_CACHE}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate RLM_vllm_server

# Per-alias config: HF model id + extra vLLM flags for the Qwen3.5 VL variants.
case "${ALIAS}" in
    qwen-3-32b)    HF_MODEL="Qwen/Qwen3-32B";   EXTRA_FLAGS=();;
    qwen-3-8b)     HF_MODEL="Qwen/Qwen3-8B";    EXTRA_FLAGS=();;
    qwen-3-5-27b)  HF_MODEL="Qwen/Qwen3.5-27B"; EXTRA_FLAGS=(--max-num-seqs 64 --gpu-memory-utilization 0.90 --limit-mm-per-prompt '{"image":0,"video":0}');;
    qwen-3-5-9b)   HF_MODEL="Qwen/Qwen3.5-9B";  EXTRA_FLAGS=(--max-num-seqs 64 --gpu-memory-utilization 0.90 --limit-mm-per-prompt '{"image":0,"video":0}');;
    *) echo "unknown alias: ${ALIAS}" >&2; exit 2;;
esac

# Belt-and-suspenders cleanup of the chosen port.
pids=$(lsof -t -i ":${PORT}" 2>/dev/null || true)
if [[ -n "${pids}" ]]; then
    echo "killing stale pids on :${PORT} -> ${pids}"
    kill ${pids} 2>/dev/null || true
    sleep 2
fi

LOGFILE="${LOG_DIR}/${ALIAS}-gpu${GPU}-p${PORT}.log"
: > "${LOGFILE}"
echo "starting ${HF_MODEL} as ${ALIAS} on CUDA:${GPU} :${PORT}  cache=${HF_HUB_CACHE}  log=${LOGFILE}"

CUDA_VISIBLE_DEVICES="${GPU}" HF_HUB_CACHE="${HF_HUB_CACHE}" \
    nohup vllm serve "${HF_MODEL}" \
        --served-model-name "${ALIAS}" \
        --host 127.0.0.1 \
        --port "${PORT}" \
        --dtype bfloat16 \
        --max-model-len 32768 \
        --max-num-batched-tokens 8192 \
        --reasoning-parser qwen3 \
        --enable-auto-tool-choice \
        --tool-call-parser "${TOOL_CALL_PARSER:-hermes}" \
        "${EXTRA_FLAGS[@]}" \
        > "${LOGFILE}" 2>&1 &
echo "vllm PID: $!"

# Wait for /v1/models. Generous timeout: cold downloads of 65 GB can be slow.
deadline=$(( $(date +%s) + 3600 ))
echo "waiting for :${PORT}/v1/models ..."
while true; do
    if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
        echo "READY: ${ALIAS} on CUDA:${GPU} :${PORT}"
        exit 0
    fi
    if (( $(date +%s) > deadline )); then
        echo "TIMED OUT after 3600s — see ${LOGFILE}" >&2
        tail -50 "${LOGFILE}" >&2 || true
        exit 1
    fi
    sleep 10
done
