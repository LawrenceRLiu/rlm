#!/usr/bin/env python3
"""Launch a single Qwen vLLM replica, probing for the maximum viable --max-num-seqs.

For multi-replica deployments with automatic port allocation and a vllm-router
frontend, use setup/serve_qwen35.py instead.

Usage:
    python setup/server_qwen35_single.py --port 8001 --gpus 0
    python setup/server_qwen35_single.py --port 8001 --gpus 0,1   # tensor-parallel across 2 GPUs
    python setup/server_qwen35_single.py --port 8001 --gpus 0 --model Qwen/Qwen3.5-32B --gpu-mem-util 0.9
"""

import argparse
import os
import site
import subprocess
import sys
import time
from pathlib import Path

HF_HUB_CACHE = "/data/nwei/rlm_substrate/models"
LOG_DIR = Path("vllm_logs")

_OOM_SIGNATURES = [
    "CUDA out of memory",
    "OutOfMemoryError",
    "torch.cuda.OutOfMemoryError",
    "CUDA error: out of memory",
]
_SUCCESS_SIGNATURE = "Application startup complete"
_MAX_NUM_SEQS_START = 8192
_MAX_NUM_SEQS_FLOOR = 8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Launch a Qwen vLLM replica with automatic max-num-seqs tuning"
    )
    p.add_argument("--model", default="Qwen/Qwen3.5-9B", help="HF model ID or local path")
    p.add_argument("--port", type=int, required=True, help="Port to bind on localhost")
    p.add_argument(
        "--gpus",
        required=True,
        help="Comma-separated CUDA device indices, e.g. '0' or '0,1' (2 GPUs → tensor parallel)",
    )
    p.add_argument("--gpu-mem-util", type=float, default=0.85, dest="gpu_mem_util")
    p.add_argument(
        "--served-name",
        default=None,
        help="Override --served-model-name (default: lowercased basename of --model with dots→dashes)",
    )
    return p.parse_args()


def _derive_served_name(model: str) -> str:
    return Path(model).name.lower().replace(".", "-")


def _build_cmd(
    model: str,
    port: int,
    gpus: str,
    gpu_mem_util: float,
    max_num_seqs: int,
    served_name: str,
) -> list[str]:
    tp_size = len(gpus.split(","))
    cmd = [
        "vllm", "serve", model,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--served-model-name", served_name,
        "--gpu-memory-utilization", str(gpu_mem_util),
        "--dtype", "bfloat16",
        "--max-num-seqs", str(max_num_seqs),
        "--limit-mm-per-prompt", '{"image":0,"video":0}',  # VL model; text-only KV layout
        "--reasoning-parser", "qwen3",                     # <think> → reasoning_content
    ]
    if tp_size > 1:
        cmd += ["--tensor-parallel-size", str(tp_size)]
    return cmd


def _try_launch(
    model: str,
    port: int,
    gpus: str,
    gpu_mem_util: float,
    max_num_seqs: int,
    served_name: str,
    log_path: Path,
) -> tuple[int | None, bool]:
    """Start vLLM and block until startup succeeds or an OOM is detected.

    Returns (pid, True) on success — process is left running.
    Returns (None, False) on OOM or unexpected exit — process is killed before return.
    """
    cmd = _build_cmd(model, port, gpus, gpu_mem_util, max_num_seqs, served_name)
    # nvidia PyPI packages install their .so files under site-packages/nvidia/*/lib/.
    # The system dynamic linker won't find them unless we add these dirs to LD_LIBRARY_PATH.
    nvidia_lib_dirs = [
        str(p)
        for sp in site.getsitepackages()
        for p in Path(sp).glob("nvidia/*/lib")
        if p.is_dir()
    ]
    ld_path = ":".join(filter(None, nvidia_lib_dirs + [os.environ.get("LD_LIBRARY_PATH", "")]))
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": gpus,
        # "CUDA_DEVICE_ORDER": "PCI_BUS_ID", #commented out because I am used to dealing with the non-pci-ordered devices.
        "HF_HUB_CACHE": HF_HUB_CACHE,
        "LD_LIBRARY_PATH": ld_path,
    }
    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)

    while proc.poll() is None:
        text = log_path.read_text(errors="replace")
        if _SUCCESS_SIGNATURE in text:
            return proc.pid, True
        if any(sig in text for sig in _OOM_SIGNATURES):
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
            return None, False
        time.sleep(3)

    return None, False  # process exited unexpectedly


def main() -> None:
    args = parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    served_name = args.served_name or _derive_served_name(args.model)
    log_path = LOG_DIR / f"replica_{args.port}.log"

    # Probe downward from 256, halving on OOM.
    # CUDA graph capture runs at power-of-2 batch sizes up to max_num_seqs, so halving
    # finds the highest power-of-2 that survives capture without burning extra probes.
    max_num_seqs = _MAX_NUM_SEQS_START
    while max_num_seqs >= _MAX_NUM_SEQS_FLOOR:
        print(f"[probe] max-num-seqs={max_num_seqs}  (log: {log_path})")
        pid, ok = _try_launch(
            args.model, args.port, args.gpus, args.gpu_mem_util, max_num_seqs, served_name, log_path
        )
        if ok:
            print(
                f"[ok]    port={args.port}  model={served_name}"
                f"  max-num-seqs={max_num_seqs}  pid={pid}"
            )
            return
        print(f"[oom]   halving → {max_num_seqs // 2}")
        max_num_seqs //= 2

    print(
        f"[fail]  could not start at max-num-seqs={_MAX_NUM_SEQS_FLOOR} — see {log_path}",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
