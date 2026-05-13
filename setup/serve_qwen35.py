#!/usr/bin/env python3
"""Orchestrate multiple Qwen vLLM replicas behind a prefix-aware vllm-router.

Each GPU spec is a comma-separated list of device indices for one replica;
use multiple comma-separated indices for tensor-parallel replicas.

Usage:
    # Fresh launch: 3 replicas; router on 8000
    python setup/serve_qwen35.py --router-port 8000 --gpus 0 1,2 4

    # Reuse already-running replicas (skips spin-up for live ports)
    python setup/serve_qwen35.py --router-port 8000 --gpus 0 1 --replica-ports 8001 8002

    # Mixed: 8001 is already up, 8002 is dead → only GPU 1 replica is relaunched
    python setup/serve_qwen35.py --router-port 8000 --gpus 0 1 --replica-ports 8001 8002
"""

import argparse
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).parent))
from server_qwen35_single import (  # type: ignore[import]
    _MAX_NUM_SEQS_FLOOR,
    _MAX_NUM_SEQS_START,
    LOG_DIR,
    _derive_served_name,
    _try_launch,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Launch multiple Qwen vLLM replicas + prefix-aware vllm-router"
    )
    p.add_argument("--model", default="Qwen/Qwen3.5-9B", help="HF model ID or local path")
    p.add_argument(
        "--router-port",
        type=int,
        required=True,
        dest="router_port",
        help="Port for the vllm-router frontend",
    )
    p.add_argument(
        "--gpus",
        nargs="+",
        required=True,
        help=(
            "GPU specs, one token per replica. Each token is a comma-separated list of device "
            "indices. E.g. '0 1,2 4' → replica on GPU 0, TP-2 on GPUs 1+2, replica on GPU 4."
        ),
    )
    p.add_argument("--gpu-mem-util", type=float, default=0.85, dest="gpu_mem_util")
    p.add_argument(
        "--served-name",
        default=None,
        dest="served_name",
        help="Override served model name (default: derived from --model)",
    )
    p.add_argument(
        "--replica-ports",
        nargs="+",
        type=int,
        default=None,
        dest="replica_ports",
        help=(
            "Explicit backend ports, one per GPU spec (same order as --gpus). "
            "Ports with a live vLLM server are reused; dead ports are relaunched on the "
            "corresponding GPU. If omitted, ports are auto-allocated."
        ),
    )
    p.add_argument(
        "--replica-port-start",
        type=int,
        default=8001,
        dest="replica_port_start",
        help="Scan for free replica ports starting here when --replica-ports is not given (default: 8001)",
    )
    p.add_argument(
        "--tool-call-parser",
        default="hermes",
        help=(
            "vLLM tool parser forwarded to every replica with --enable-auto-tool-choice. "
            "Default: hermes."
        ),
    )
    return p.parse_args()


def _find_free_port(start: int, exclude: set[int]) -> int:
    for port in range(start, start + 2000):
        if port in exclude:
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found starting from {start}")


def _is_replica_healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def _probe_replica(
    idx: int,
    model: str,
    port: int,
    gpus: str,
    gpu_mem_util: float,
    served_name: str,
    tool_call_parser: str,
    results: dict,
    lock: threading.Lock,
) -> None:
    log_path = LOG_DIR / f"replica_{port}.log"
    max_num_seqs = _MAX_NUM_SEQS_START
    while max_num_seqs >= _MAX_NUM_SEQS_FLOOR:
        print(f"[replica {idx}] probe max-num-seqs={max_num_seqs}  port={port}  gpus={gpus}")
        pid, ok = _try_launch(
            model,
            port,
            gpus,
            gpu_mem_util,
            max_num_seqs,
            served_name,
            tool_call_parser,
            log_path,
        )
        if ok:
            with lock:
                results[idx] = {"port": port, "pid": pid, "max_num_seqs": max_num_seqs, "gpus": gpus}
            print(f"[replica {idx}] ok  port={port}  gpus={gpus}  max-num-seqs={max_num_seqs}  pid={pid}")
            return
        print(f"[replica {idx}] oom  halving → {max_num_seqs // 2}")
        max_num_seqs //= 2

    with lock:
        results[idx] = None
    print(
        f"[replica {idx}] FAILED — could not start on gpus={gpus}  see {log_path}",
        file=sys.stderr,
    )


def _launch_router(router_port: int, backend_urls: list[str]) -> int:
    """Start vllm-router; block until its port is reachable. Returns PID."""
    log_path = LOG_DIR / "router.log"
    cmd = [
        "vllm-router",
        "--host", "127.0.0.1",
        "--port", str(router_port),
        "--worker-urls", *backend_urls,
        "--policy", "cache_aware",
    ]
    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)

    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", router_port), timeout=1):
                return proc.pid
        except OSError:
            if proc.poll() is not None:
                raise RuntimeError(f"vllm-router exited unexpectedly — see {log_path}") from None
            time.sleep(1)

    raise RuntimeError(
        f"vllm-router did not become reachable on port {router_port} within 30 s — see {log_path}"
    )


def main() -> None:
    args = parse_args()

    if args.replica_ports is not None and len(args.replica_ports) != len(args.gpus):
        print(
            f"[error] --replica-ports has {len(args.replica_ports)} values but --gpus has {len(args.gpus)}",
            file=sys.stderr,
        )
        sys.exit(1)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    served_name = args.served_name or _derive_served_name(args.model)

    # Determine one port per replica.
    if args.replica_ports is not None:
        replica_ports = args.replica_ports
    else:
        used_ports: set[int] = {args.router_port}
        replica_ports = []
        for _ in args.gpus:
            p = _find_free_port(args.replica_port_start, used_ports)
            used_ports.add(p)
            replica_ports.append(p)

    print(f"[setup] model={served_name}  replicas={len(args.gpus)}  router-port={args.router_port}")

    # Check which replicas are already healthy; only launch the rest.
    results: dict[int, dict | None] = {}
    to_launch: list[tuple[int, str, int]] = []  # (idx, gpus, port)

    for i, (gpus, port) in enumerate(zip(args.gpus, replica_ports, strict=True)):
        tp = len(gpus.split(","))
        if _is_replica_healthy(port):
            print(f"[replica {i}] healthy  port={port}  gpus={gpus}  tp={tp}  (skipping launch)")
            results[i] = {"port": port, "pid": None, "max_num_seqs": "?", "gpus": gpus}
        else:
            print(f"[replica {i}] not healthy  port={port}  gpus={gpus}  tp={tp}  → will launch")
            to_launch.append((i, gpus, port))

    if to_launch:
        lock = threading.Lock()
        threads = [
            threading.Thread(
                target=_probe_replica,
                args=(
                    i,
                    args.model,
                    port,
                    gpus,
                    args.gpu_mem_util,
                    served_name,
                    args.tool_call_parser,
                    results,
                    lock,
                ),
                daemon=True,
            )
            for i, gpus, port in to_launch
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    failed = [i for i, r in results.items() if r is None]
    if failed:
        print(
            f"[fail] replica(s) {failed} did not start — aborting router launch",
            file=sys.stderr,
        )
        sys.exit(1)

    replica_results = [cast(dict, results[i]) for i in range(len(args.gpus))]
    backend_urls = [f"http://127.0.0.1:{r['port']}" for r in replica_results]
    print(f"[router] launching on port {args.router_port}  backends={backend_urls}")
    router_pid = _launch_router(args.router_port, backend_urls)
    print(f"[router] ok  port={args.router_port}  pid={router_pid}")

    print("\n[done] Stack is up:")
    print(f"  Endpoint: http://127.0.0.1:{args.router_port}/v1  (model: {served_name})")
    for i, r in enumerate(replica_results):
        pid_str = str(r["pid"]) if r["pid"] is not None else "existing"
        print(
            f"  Replica {i}: http://127.0.0.1:{r['port']}/v1"
            f"  gpus={r['gpus']}  max-num-seqs={r['max_num_seqs']}  pid={pid_str}"
        )
    print(f"  Router pid={router_pid}")


if __name__ == "__main__":
    main()
