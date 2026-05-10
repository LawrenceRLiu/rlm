# Terminal-Bench 2.0 (Harbor) integration

This doc records the design decisions, gotchas, and validation results for wiring the RLM substrate up to Terminal-Bench 2.0 / Harbor. Architecture overview lives in [`../08_benchmark_support.md`](../08_benchmark_support.md). Original gap analysis (across SWE-Bench / TB / AIME) is at [`benchmark_compatibility_todos.md`](benchmark_compatibility_todos.md).

## Goal

Validate that the workspace substrate (agent loop, broker, recursion, cleanup) holds up under realistic benchmark workloads — not headline TB numbers. Three hand-picked Harbor demo tasks, end-to-end, with the official Harbor grader contract. Models: local `gemma-4-31b` via litellm at `127.0.0.1:8000`.

## Status

End-to-end on 3 Harbor demo tasks, all PASS (gemma-4-31b):

| Task | Base | Grader | Wall | Result |
|---|---|---|---|---|
| `hello-world` | ubuntu:24.04 | apt+uv+pytest | 10.9s | PASS |
| `hello-alpine` | alpine:3.22 | apk+uv+pytest | 9.5s | PASS |
| `hello-workdir` | ubuntu:24.04 | shell-only | 6.5s | PASS |

Substrate health (verified): zero orphan containers, workspaces deleted (`cleanup_mode="delete"`), trajectories logged, broker `/health` responsive throughout each run, `<action tool="python">` not exercised by these tasks but the split-interpreter design is verified via the standalone smoke test against `python:3.11-slim`.

## What got built

```
eval/
  common/
    composite_image.py                ── build_composite + smoke_test
    Dockerfile.composite.template     ── ${BASE_IMAGE} + broker layer
  terminal_bench/
    runner.py                         ── orchestrator
    tasks.yaml                        ── chosen tasks
third_party/
  harbor/                             ── submodule pin (registry source)
rlm/core/rlm.py                       ── pre_cleanup_callback hook added
rlm/core/types.py                     ── RLMChatCompletion.pre_cleanup_result
tests/
  test_workspace_loop.py              ── +8 callback tests
  test_composite_image.py             ── +12 composite-image tests
  test_terminal_bench_runner.py       ── +17 runner tests
```

Test count: 234 → 259 (+25 new unit tests). Lint clean.

## Design decisions

### TB 2.0 (Harbor) over TB 1.x

TB 1.x is more mature (~80 tasks) but its agent surface is a tmux pane driven by keystrokes (the "Terminus" agent). That doesn't match RLM's `<action tool="shell">` paradigm — a TB 1.x integration would be wrapping RLM as an alternative agent over TB tasks, with leaderboard comparison only loose at best.

TB 2.0 / Harbor (`task.toml` + `instruction.md` + `tests/test.sh`) maps cleanly onto `RLM.completion(prompt) → final → grader`. The grader is shell-based (`bash test.sh`), so the contract is "agent finishes; harness runs grader against the live container; reads pass/fail from `/logs/verifier/reward.txt`" — exactly what a pre-cleanup callback can do.

Downside: Harbor itself is newer (released as the TB 2.0 framework). The task source repo (`laude-institute/terminal-bench-2`) has 89 tasks but we used 3 of Harbor's own demo tasks for this validation — they're trivial but exercise three different graders (pytest-bootstrap on Ubuntu, pytest-bootstrap on Alpine, pure-shell) and that's the substrate-validation goal.

### Pre-cleanup callback over alternative architectures

`RLM.completion()` previously owned the env lifecycle inside a context manager (env created, agent runs, `env.cleanup()` runs in `finally`). When `completion()` returns, the container is already gone — the harness has no chance to run the grader inside it. We considered four options:

1. **Pre-cleanup callback** (chosen). Adds one parameter, ~25 lines. Env lifecycle stays sealed. Composable. Generalizes to SWE-Bench (host-side `git diff` extraction fits the same hook).
2. **Externalize env construction**. Caller owns env. ~80-150 line refactor. Clean separation. Buys multi-completion sessions on the same env (useful for notebook debugging, not for one-shot graders). Deferred.
3. **`cleanup_on_completion=False` + manual teardown**. Caller calls `result.env.cleanup()`. Footgun — every batch run has many opportunities for a forgotten cleanup. Rejected.
4. **Agent self-grades via `final`**. Zero substrate change but breaks agent/evaluator separation, exposes test scripts to the agent, unreliable signal. Rejected.

Option 1's contract is explicit: callback fires on clean return (final action OR max-iterations fallback), does NOT fire if `_run_loop` raises (parse-retry exhaustion, cancellation, budget exceeded, etc.). Rationale: a crashed agent's workspace isn't in a state we'd trust to grade. The container is still torn down in either case — only the grading is skipped. Documented in the `RLM.completion` docstring and pinned by `tests/test_workspace_loop.py::TestPreCleanupCallback::test_callback_does_not_fire_when_loop_raises`.

### Composite image as a shared helper, not a per-benchmark script

Originally planned `eval/terminal_bench/build_composite.py`. Refactored to `eval/common/composite_image.py` at the user's suggestion before any code was written, on the observation that the broker layer is benchmark-agnostic — only the FROM line varies. The shared helper takes `base_image` and `output_tag`; SWE-Bench will reuse the same API by pulling the right base image and sanitizing the instance id before calling.

### Split-interpreter (uv-managed Python 3.11 for broker)

The broker depends on Flask. Benchmark base images may have an incompatible project Python (Python 3.6 in older SWE-Bench instances, no Python at all in `ubuntu:24.04`, etc.). Composite image installs its own isolated Python 3.11 at `/opt/broker` via `uv venv ... --seed`. Project python is untouched. The two interpreters communicate over localhost HTTP (interpreter-agnostic).

Verified during smoke testing: `/usr/local/bin/python` carries `requests 2.33.1`, `/opt/broker/bin/python` carries `flask 3.1.3`, the broker's `/health` responds.

### Run broker as script, not `python -m rlm_workspace.broker`

`rlm_workspace/__init__.py` eager-imports the client module (`from rlm_workspace.client import llm_query, ...`). The client imports `requests`. The broker venv doesn't have `requests` (only flask). So `python -m rlm_workspace.broker` fails: the package's `__init__.py` runs before `broker.py` even gets a chance to execute.

Fix: `CMD ["/opt/broker/bin/python", "/opt/rlm_workspace/rlm_workspace/broker.py"]` — file-form invocation, no package import, `__init__.py` doesn't run. Two-line change in the Dockerfile template.

### Best-effort `requests` install into project python

The original Dockerfile template had `RUN python -m pip install requests` — hard-failed on `ubuntu:24.04` (no python interpreter at all). Made it best-effort:

```dockerfile
RUN (python -m pip install --no-cache-dir requests 2>/dev/null) \
 || (python3 -m pip install --no-cache-dir requests 2>/dev/null) \
 || echo "[composite] no project python with pip found; <action tool=python> will be unavailable"
```

Consequence: on a python-less base, the `<action tool="python">` tool's in-container bodies will fail with a clear error at runtime (no `requests` to talk to the broker). The `shell` and host-side file tools still work.

### `ENTRYPOINT []` reset in the composite

Some Harbor tasks (e.g. `hello-healthcheck`) define an `ENTRYPOINT` to start a side-service (a webserver). Docker preserves a parent's `ENTRYPOINT` when the child only sets `CMD`, so our broker `CMD` becomes args to the inherited entrypoint — broker never starts. Resetting `ENTRYPOINT []` in the composite ensures broker is PID 1.

Trade-off: tasks that depend on a pre-running entrypoint service are not supported by this composite. For substrate validation that's fine; the 3 chosen tasks don't have entrypoints.

### Harbor-canonical grader (`bash test.sh` → `reward.txt`) over running pytest directly

First draft of `make_grader` ran `pytest /tests/test_state.py` from the broker venv directly (after `docker cp` of the tests dir + `pip install pytest`). Fast (~1s grader) but only works for pytest-based tasks. Harbor tasks vary: `hello-workdir` and `hello-healthcheck` have pure-bash graders that grep files; `hello-world` and `hello-alpine` use the full apt/apk + uv + pytest bootstrap.

Refactored to the Harbor-canonical contract:
1. `docker cp tests/ → /tests` (live container).
2. `mkdir -p /logs/verifier && chmod +x /tests/test.sh`.
3. `bash /tests/test.sh` (timeout = `task.verifier_timeout_sec`).
4. `cat /logs/verifier/reward.txt`; `"1"` = pass.

This is what the official Harbor harness does, so any Harbor task works with no per-task grader code. Cost: ~30-60s more wall time on pytest-bootstrap tasks (uv install + pytest install run each grade), but it's faithful and uniform.

## Gotchas worth knowing

### Workspace bind-mount vs. task workdir

`DockerWorkspaceEnv._start_container` hard-codes `-v <ws>:/workspace -w /workspace`. Harbor tasks expect their work in the Dockerfile-defined `WORKDIR` (typically `/app`) or in `task.toml`'s `[environment].workdir` (e.g. `/custom-workdir`). Our bind-mount overrides that, AND the agent's `read_file`/`write_file` tools operate on the bind-mounted `/workspace`, not the task workdir.

Workaround: `build_prompt` adds a preamble telling the agent that the target dir is `task.workdir` and that the file tools won't help — use `shell` for anything the grader will inspect:

```
Important rules:
- Do your work in `/app` using the `shell` tool (e.g. `cd /app && ...`).
- The `read_file` / `write_file` / `edit_file` tools operate on
  `/workspace`, which is NOT the grader's target. Use `shell` for
  anything the grader will inspect.
```

A real fix would be a `DockerConfig.workspace_mount_path` config knob. Not done.

### Stale composite-image cache when editing the template

`build_composite(..., cache=True)` skips the rebuild if an image with the target tag already exists locally. If you edit `Dockerfile.composite.template` and rerun, the *new* template is rendered but the *old* image is reused. Symptom: changes silently don't take effect.

Workaround: `docker rmi rlm-tb-<id>:latest harbor-task-base-<id>:latest` between template edits, or pass `cache=False`. Not a real footgun in production (template stable), but the iteration-during-dev pain motivated the explicit cache-stale warning in the test (`tests/test_composite_image.py::test_skips_cache_when_disabled`).

### `_container_id` is a private attribute used by the grader

The grader callback needs `docker cp tests/ → /tests`, which needs the container id, which isn't exposed on `DockerWorkspaceEnv`'s public surface. `make_grader` reaches in via `env._container_id`. This is an accepted private-attr access; if a public `env.copy_into_container(host_src, container_dst)` method materializes later, the grader becomes one line cleaner. Not pressing.

### Recursion `rlm_query` not tested on Harbor tasks

None of the 3 chosen Harbor demo tasks naturally trigger recursion (they're all single-shell-call). Recursion is exercised by unit tests in `tests/test_recursion.py` and by prior smoke runs (`_setup_runs/`) but not via the benchmark runner. A harder TB 2.0 task that decomposes naturally would close that gap.

## Validation evidence

Trajectories (gemma-4-31b, 3 tasks, May 2026):

```
eval/terminal_bench/results/hello-world/trajectory/rlm_2026-05-10_13-59-23_43552502.jsonl
eval/terminal_bench/results/hello-alpine/trajectory/rlm_2026-05-10_13-59-43_006ccd3d.jsonl
eval/terminal_bench/results/hello-workdir/trajectory/rlm_2026-05-10_14-00-06_95e3f977.jsonl
```

Each trajectory is 4 lines (1 metadata + 3 iterations). Per-task `result.json` files in the same directory tree record the grader's `reward_raw`, exit code, timing, and any error.

Inspected with the standard tooling described in [`../07_viewing_traces.md`](../07_viewing_traces.md). Visualizer renders correctly; the new `pre_cleanup_result` field is on the completion object but is not surfaced by the JSONL iteration schema (it lives on the completion, not on a turn).

## Reproducing

```bash
# From the repo root, with conda env RLM_substrate active and litellm/vllm
# running on 127.0.0.1:8000 serving gemma-4-31b:
python -m eval.terminal_bench.runner \
    --tasks third_party/harbor/examples/tasks/hello-world \
            third_party/harbor/examples/tasks/hello-alpine \
            third_party/harbor/examples/tasks/hello-workdir \
    --max-iterations 15 \
    --output-dir eval/terminal_bench/results
```

To extend to other TB 2.0 / Harbor tasks: point `--tasks` at any directory containing a `task.toml`, `instruction.md`, `environment/Dockerfile`, and `tests/test.sh`. The full TB 2.0 task set lives at `https://github.com/laude-institute/terminal-bench-2` (not currently vendored); a separate submodule would mirror the Harbor pattern.

## Open follow-ups

These are tracked at a higher level in [`../../Todo.md`](../../Todo.md):

- **SWE-Bench integration** — composite-image helper already supports it; needs the SWE-Bench-specific runner (instance loader, repo seeding into `workspace_root/repo/`, host-side `git diff` extraction, `predictions.jsonl` writer). Repo seeding is the main new substrate surface — see [`benchmark_compatibility_todos.md`](benchmark_compatibility_todos.md) Gap 1.
- **AIME 2025 ablation** — minimal. Reuses default `rlm-workspace:0.1.0` image; just needs an answer-parsing harness. Useful as a "substrate doesn't hurt on reasoning tasks" sanity check.
- **Configurable workspace mount path** — would let Harbor tasks like `hello-workdir` honor `[environment].workdir` natively instead of via prompt instruction. One-knob substrate change.
- **`--user` support in `_start_container`** — would unblock tasks like `hello-user` that check `whoami != root`.
- **Real `<action tool="python">` test in a Harbor task** — none of the chosen tasks exercise it; we're relying on the standalone smoke build for that signal.
- **Recursion exercised on a benchmark task** — same, no current Harbor task triggers it.
