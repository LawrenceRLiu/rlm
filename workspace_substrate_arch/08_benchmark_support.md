# 08. Benchmark support

The workspace substrate is designed to evaluate against external agentic benchmarks. This doc describes the shared infrastructure for running an `RLM` against a third-party task spec (a Docker image + an instruction + a grader) and recording pass/fail.

The original gap analysis is at [`dev/benchmark_compatibility_todos.md`](dev/benchmark_compatibility_todos.md). The Terminal-Bench instantiation (the only benchmark currently wired up) is documented in detail at [`dev/terminal_bench_integration.md`](dev/terminal_bench_integration.md).

## Status

| Benchmark | Status | Where |
|---|---|---|
| Terminal-Bench 2.0 (Harbor) | **wired up**; substrate validation passing on 3 demo tasks | `eval/terminal_bench/` |
| SWE-Bench | analysis only, not implemented | `dev/benchmark_compatibility_todos.md` |
| AIME 2025 | not started | — |

## Three shared pieces

Every benchmark integration needs three things; one of each lives in core/`eval/common/`, the third is per-benchmark.

### 1. Pre-cleanup callback (in `RLM.completion`)

`RLM.completion()` accepts an optional `pre_cleanup_callback: Callable[[DockerWorkspaceEnv], Any]` that fires after the agent loop returns but before the container is torn down. Its return value is attached to the result as `pre_cleanup_result`. This is how benchmark graders run against the live container.

```python
def grade(env: DockerWorkspaceEnv) -> dict:
    return {"passed": env.exec_in_container([...]).exit_code == 0}

result = rlm.completion(prompt, pre_cleanup_callback=grade)
passed = result.pre_cleanup_result["passed"]
```

Implementation: `rlm/core/rlm.py::RLM.completion`. Field added: `RLMChatCompletion.pre_cleanup_result`. The contract is intentionally narrow — see the docstring for what it doesn't cover (iterative grading protocols, exception-time grading, recursion children).

### 2. Composite-image build (in `eval/common/composite_image.py`)

Most benchmarks ship per-instance / per-task base Docker images (SWE-Bench: `swebench/sweb.eval.x86_64.<id>`; Terminal-Bench: built locally from each task's `environment/Dockerfile`). The substrate needs the broker (`docker/workspace_image/rlm_workspace/broker.py`) running as PID 1 inside that container. A *composite image* is "benchmark base image + broker layer".

```python
from eval.common.composite_image import build_composite, smoke_test

composite_tag = build_composite(
    base_image="harbor-task-base-hello-world:latest",
    output_tag="rlm-tb-hello-world:latest",
)
smoke_test(composite_tag)   # optional: verify broker /health responds
```

The split-interpreter design (`eval/common/Dockerfile.composite.template`) installs the broker into its own uv-managed Python 3.11 venv at `/opt/broker`. The base image's project python is untouched (broker deps can't break it, and it can't break broker deps). Project python gets `requests` installed best-effort so `<action tool="python">` can call the broker; on python-less bases (e.g. `ubuntu:24.04`) that step is silently skipped and only the python tool is lost.

Tagging is the caller's responsibility, including any benchmark-specific sanitization (SWE-Bench's `__` → `_1776_` rule, for example).

### 3. Per-benchmark runner (in `eval/<benchmark>/runner.py`)

The runner orchestrates: parse the task spec, build the base image, layer the composite, construct `RLM` against that image, build a grader callback, call `completion()`, write results. Per-benchmark because:

- Task format differs (TB has `task.toml`, SWE-Bench has HF dataset rows, AIME has a HF jsonl).
- Base-image acquisition differs (TB builds from `environment/Dockerfile`, SWE-Bench pulls from a registry, AIME doesn't need one).
- Grader contract differs (TB: `bash test.sh` → `/logs/verifier/reward.txt`; SWE-Bench: host-side `git diff` → `predictions.jsonl`; AIME: parse `<answer>` for integer).
- Result schema differs.

## Layout

```
rlm/core/rlm.py                       ── pre_cleanup_callback hook (shared)
eval/
  common/
    composite_image.py                ── build_composite / smoke_test
    Dockerfile.composite.template     ── broker layer over ${BASE_IMAGE}
  terminal_bench/
    runner.py                         ── TB 2.0 / Harbor orchestrator
    tasks.yaml                        ── hand-picked demo tasks
  # future:
  # swebench/runner.py
  # aime/runner.py
third_party/
  harbor/                             ── git submodule: TB 2.0 task source
docker/
  workspace_image/rlm_workspace/      ── broker.py + client.py (COPY'd in)
```

## Per-completion lifecycle

```
host                                            container
────                                            ─────────
build base image (per-benchmark)
build_composite(base, "rlm-<bench>-<id>")
    │
    └─ docker build (broker layer)
                                                ──── container starts ────
RLM(workspace_config=...).completion(           broker (PID 1) listens
    prompt,                                     on /enqueue, /pending,
    pre_cleanup_callback=grade)                 /respond, /health
    │
    ├─ agent loop  ─────────────────────►       <action> dispatch via
    │                                           docker exec; broker
    │                                           bridges in-container LM
    │                                           calls back to the host
    │
    ├─ final action returned                    (or max_iterations / etc.)
    │
    ├─ pre_cleanup_callback(env) ──────►        docker cp tests/ → /tests
    │                                           bash /tests/test.sh
    │                                           cat /logs/verifier/reward.txt
    │
    └─ env.cleanup()  ──────────────────►       container removed
                                                workspace dir deleted
                                                (cleanup_mode="delete")

result.pre_cleanup_result == {"passed": ...}
```

## What the substrate doesn't yet support

These come up across benchmarks; none are blockers for TB 2.0 but they bound what else can be added without core changes.

- **Bind-mount path is fixed at `/workspace`.** Tasks that expect their work in a non-`/workspace` directory (e.g. Harbor's `/app`, SWE-Bench's `/testbed`) must be steered via the prompt — the agent's `shell` tool can `cd` anywhere, but the host-side file tools (`read_file`, `write_file`, `edit_file`) only see `/workspace`. Configurable mount path would be one core change unlocking cleaner integrations.
- **No `USER` override on the run.** Container always runs as root. Tasks like Harbor's `hello-user` (which expects `whoami` to report `agent`) won't match without a substrate-side `--user` knob.
- **Inherited `ENTRYPOINT` is reset.** Tasks whose base image relies on `ENTRYPOINT` to start side-services (e.g. Harbor's `hello-healthcheck` starts a webserver) lose those services in the composite. Tasks that need a running service must include it in the agent's own setup, or we'd need a sidecar / multi-PID solution.
- **No multi-completion sessions on the same env.** Iterative or oracle-feedback grading protocols (`agent → grade → agent revises → re-grade`) aren't expressible — the env is destroyed after the callback. Externalizing env construction would unlock this; not currently a need.
- **`pre_cleanup_callback` does not fire when the loop raises.** Parse-retry exhaustion, cancellation, budget exceeded → no grading attempted. Callers who want exception-time post-processing wrap their own `try/finally` around `completion()`.

## Pointers

- **Terminal-Bench dev doc**: [`dev/terminal_bench_integration.md`](dev/terminal_bench_integration.md) — what got built, design decisions, validation results, known limits.
- **Original benchmark gap analysis**: [`dev/benchmark_compatibility_todos.md`](dev/benchmark_compatibility_todos.md) — the inventory across SWE-Bench / TB / AIME that this work was scoped against. The TB section is now closed; SWE-Bench / AIME remain open.
- **Trace inspection**: [`07_viewing_traces.md`](07_viewing_traces.md) — how to read the JSONL, drill into recursion, etc. The same tools work on benchmark trajectories.
