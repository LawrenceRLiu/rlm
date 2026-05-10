# Analysis: What's needed to run RLM on SWE-Bench / Terminal-Bench / AIME

## Status (2026-05-10)

This was an analysis-only doc. Implementation has now started:

| Benchmark | Status | Pointer |
|---|---|---|
| **Terminal-Bench 2.0 (Harbor)** | wired up, validated on 3 demo tasks | [`terminal_bench_integration.md`](terminal_bench_integration.md) |
| SWE-Bench | analysis still current; not built | this doc, SWE-Bench section below |
| AIME 2025 | not built | this doc, AIME section below |

Architecture overview of the shared benchmark-support layer (composite-image helper, pre-cleanup callback, runner pattern) is in [`../08_benchmark_support.md`](../08_benchmark_support.md).

The Terminal-Bench section of this doc (Gaps T1–T5) was written assuming TB 2.0 / Harbor; that's the version actually built. **All gaps in that section are now closed** — see [`terminal_bench_integration.md`](terminal_bench_integration.md) for the implementation. The SWE-Bench section (Gaps 1–8) is unchanged and remains an open work item.

One factual correction worth flagging for the SWE-Bench plan: the **broker port is `RLM_BROKER_PORT`-configurable** (default 8080), not hard-coded. The composite-image substrate-side change description below was written before this was verified — it doesn't change the design, but `client.py`'s URL construction reads the env var rather than hard-coding `:8080`. Second correction: SWE-Bench's per-instance image tags **sanitize `__` → `_1776_`** (Docker Hub disallows `__` in tag names) — the caller of `build_composite()` must apply this before invoking docker.

## Context

The user asked for an *analysis only* (no implementation) of the gap between the
current RLM substrate and a working SWE-Bench evaluation. SWE-Bench (Verified /
Lite / Full) provides ~300–2.3k instances; each instance is `(repo, base_commit,
problem_statement, FAIL_TO_PASS, PASS_TO_PASS)`. The agent must produce a `git
diff` (a patch) against `base_commit`. Evaluation runs the patch inside the
official SWE-Bench per-instance Docker image and checks that the FAIL_TO_PASS
tests now pass and PASS_TO_PASS tests still pass.

This document inventories the gaps so the user can decide scope.

---

## What the substrate already gives us (good news)

- **Per-run isolated workspaces**: `DockerWorkspaceEnv.setup()` provisions a
  fresh workspace dir + container per `RLM` instance
  (`rlm/environments/docker_workspace.py:155-207`). One container per instance
  is the right unit for SWE-Bench.
- **Shell + Python tools**: `<action tool="shell">` and `tool="python">`
  already exist (`rlm/workspace_tools/`). The agent can run `pytest`, `git`,
  `pip install`, etc. inside the container.
- **File editing tools**: `read_file`, `write_file`, `edit_file`,
  `list_directory` cover the agent-side edit surface.
- **Per-turn git snapshots**: `_git_init` + per-turn commits already happen
  inside the workspace (separate from the target repo). This is incidental but
  useful for trajectory inspection.
- **Generic system prompt**: `rlm/utils/prompts.py` is task-agnostic; the
  SWE-Bench instance description can be passed via `prompt=` to
  `RLM.completion()` and lands in `_rlm_query_0.txt`.
- **Recursion (`rlm_query`)**: Could plausibly be used to spawn child
  subagents per file or per hypothesis, though not required for v1.

---

## The gaps (in rough order of severity)

### Gap 1 — No way to seed the workspace with a repo at a base commit *(blocking)*

Current `load_context()` (`docker_workspace.py:410-434`) only writes prompt
text into `_rlm_query_<N>.txt`. There is no first-class API for "populate the
workspace with these files" or "clone this repo at this SHA before the agent
starts."

The bind-mount is `<workspace_root_on_host>:/workspace`. So the simplest fix
is a host-side step that runs *before* `setup()` (or hooks into it):

1. `git clone <repo>` into `workspace_root/repo/` on the host.
2. `git checkout <base_commit>`.
3. Then `setup()` proceeds (the `_git_init` it does on `workspace_root`
   itself is a separate, outer git — needs care so its commits don't pollute
   the inner repo's history).

The interaction between the substrate's outer `_git_init` of `workspace_root`
and the cloned inner repo at `workspace_root/repo/` needs to be checked. If
the outer `git add -A . && git commit` walks into the nested `.git`, behavior
depends on whether git treats it as a submodule or just a nested dir. Likely
need to either (a) `.gitignore` `repo/` in the outer repo, or (b) skip the
outer git entirely for SWE-Bench runs (config flag).

### Gap 2 — Single hard-coded Docker image vs. per-instance SWE-Bench images *(architectural)*

`DockerConfig.image = "rlm-workspace:0.1.0"` is fixed
(`rlm/core/config.py:62`). The container is started with that one image
(`docker_workspace.py:240`).

SWE-Bench's official harness ships one Docker image per instance, each baked
with the right Python version, system packages, and repo deps preinstalled.
You have three options, in increasing order of effort/correctness:

- **(A) "Bring your own setup"** — keep `rlm-workspace:0.1.0`, tell the agent
  to `pip install` everything in-task. Brittle (network, version drift,
  timeouts) and slow (every instance pays setup cost). But zero substrate
  changes.

- **(B) Per-instance composite image** *(recommended for v1)* — build a
  derived image per instance: `FROM swebench/sweb.eval.x86_64.<instance>:latest`
  + layer in the broker. See the **Composite-image design** subsection below
  for the full design including the split-interpreter trick that makes this
  work even when SWE-Bench's project Python is incompatible with the broker.

- **(C) Two-container model** — keep `rlm-workspace` as a sidecar broker, run
  the SWE-Bench image as the work container, mount a shared volume. More
  invasive substrate change; probably not worth it for v1.

(B) is the standard pattern used by other SWE-Bench agent harnesses (SWE-agent,
OpenHands, etc.).

#### Composite-image design (for option B)

**Background — what the broker is and isn't.** The broker
(`docker/workspace_image/rlm_workspace/broker.py`, ~150 lines) is a small Flask
HTTP server with no state, no auth, no native deps. Its only job is to bridge
LM calls *originating from inside the container* back to the host's LM client.
There are two distinct LM-call paths in the substrate:

- **Host-side path (no broker involvement).** The agent's main loop runs on
  the host. The host calls the LM, parses `<action>` blocks, dispatches them.
  When the agent emits `<action tool="llm_query">...</action>`, that tool's
  `runs_on="host"` (`rlm/workspace_tools/llm_query.py:24`); the host calls
  the LM directly and returns the result. Broker idle.
- **Container-side path (broker-bridged).** When the agent emits
  `<action tool="python">`, the body runs *inside the container* via
  `docker exec`. The wrapper preamble (`rlm/workspace_tools/python.py:28`)
  pre-imports `llm_query`, `rlm_query`, etc. from `rlm_workspace.client`.
  Those helpers POST to `http://localhost:<broker_port>/enqueue` inside the
  container. The host poller picks the request up at `/pending`, calls the
  real LM, posts back to `/respond`, and the in-container caller's blocked
  `requests.post` unblocks with the response.

The model itself is **not aware of the broker** — it only sees the action
surface in the system prompt. The broker is a pure implementation detail of
the python-tool path.

**Split-interpreter design.** The broker needs Python 3.8+ with `flask`. The
SWE-Bench instance Python may be 3.6 or have other constraints. Solution: run
two interpreters side-by-side in the same container, doing different jobs,
talking over localhost HTTP (which is interpreter-agnostic):

```
/opt/broker/bin/python    ← private Python 3.11, has flask + requests,
                              runs broker.py as PID 1
/usr/bin/python  (etc.)   ← SWE-Bench's project Python, has the project's
                              deps and the repo, used by docker exec for
                              the agent's <action tool="python"> bodies
```

Sketch of the per-instance Dockerfile:

```dockerfile
FROM swebench/sweb.eval.x86_64.<instance>:latest

# Private Python for the broker — does not disturb the project's python
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
 && /root/.local/bin/uv python install 3.11 \
 && /root/.local/bin/uv venv /opt/broker --python 3.11 \
 && /opt/broker/bin/pip install flask requests

COPY docker/workspace_image/rlm_workspace /opt/broker/lib/rlm_workspace

# Make the in-container client importable from the project's python.
# Cheapest fix: add it to PYTHONPATH for python invocations.
ENV PYTHONPATH=/opt/broker/lib

# The client does requests.post — needs `requests` in the project python.
# Either pip install it (small dep, usually fine) OR rewrite client.py to
# use stdlib urllib (~5 lines change, removes even this dep).
RUN python -m pip install requests

CMD ["/opt/broker/bin/python", "/opt/broker/lib/rlm_workspace/broker.py"]
```

**Why this works:**

1. The broker runs on its own Python; flask install is isolated from the
   project. Project deps cannot break the broker; broker deps cannot break
   the project.
2. The agent's `<action tool="python">` body runs via `docker exec python
   /workspace/<script>.py`. That `python` is the project's Python — gets
   Django, the repo, all the right deps, just as SWE-Bench expects.
3. The bridge is HTTP. The script's preamble does `from rlm_workspace.client
   import llm_query`; `client.py` does `requests.post('http://localhost:8080
   /enqueue', ...)`. Project Python only needs `requests` (or stdlib `urllib`
   if `client.py` is rewritten).
4. The host poller doesn't care which interpreter runs the broker — it just
   talks `/pending` / `/respond` HTTP against the published port.

**Substrate-side change for option B:** essentially none. `DockerConfig.image`
is already a per-`RLM`-instance setting, so the harness just instantiates
each `RLM` with the matching composite image string:

```python
RLM(workspace_config=WorkspaceConfig(
    docker=DockerConfig(image=f"rlm-swebench-{instance_id}:latest")
), ...)
```

The substrate's existing `docker run -v <ws>:/workspace ...` invocation is
unchanged; only the image name differs. Build composite images once (cached
locally or in a registry), reuse for every benchmark run. Build cost per
instance is ~5–30s (just adding a thin layer on top of the SWE-Bench base).

**Caveats to verify in the PoC:**

- Path conflict: confirm SWE-Bench images don't put the repo at `/workspace`
  (they typically use `/testbed`). The substrate bind-mounts `<ws>:/workspace`
  and would shadow anything there.
- `CMD` override: SWE-Bench images may set their own `CMD`; the composite
  Dockerfile overriding it to run the broker is usually fine but worth
  checking each base image doesn't rely on its default entrypoint for setup.
- One-time `requests` install into the project Python: confirm it doesn't
  break the project's pinned dep set (very unlikely for a leaf dep like
  `requests`, but possible).

### Gap 3 — No way to extract a patch as the result *(small but mandatory)*

The `final` tool returns a string answer + optionally lists artifact paths
(`rlm/workspace_tools/final.py`). For SWE-Bench, the canonical output is
`git diff` against `base_commit` from inside `repo/`.

Two approaches:

- Tell the agent (via system prompt or task framing) to put the patch text
  inside `<answer>...</answer>`.
- Better: post-process — after `completion()` returns, the host runs
  `git -C <workspace_root>/repo diff <base_commit>` and uses *that* as the
  patch, ignoring whatever the model put in `<answer>`. This is more robust
  to model formatting issues (especially given the recent Qwen3.6 output-
  format problem from commit `c53c6a9`).

### Gap 4 — No batch harness *(must build, but straightforward)*

No `eval/`, `benchmarks/`, or batch script exists. Need a runner that:

1. Loads SWE-Bench instances (HF dataset `princeton-nlp/SWE-bench_Verified`
   or similar).
2. For each instance: clones repo @ commit, builds/selects the Docker image,
   instantiates `RLM(workspace_config=...)`, calls `completion(prompt)`,
   extracts patch, writes `predictions.jsonl` line in SWE-Bench format
   (`{instance_id, model_name_or_path, model_patch}`).
3. Concurrency: thread/process pool over instances. Each `RLM` is its own
   container, so isolation is fine; the bottleneck is host CPU/disk and LM
   QPS.
4. Resumability: skip instances whose `instance_id` already has a prediction.
5. Hand off `predictions.jsonl` to the official `swebench.harness.run_evaluation`.

Reference: `_setup_runs/run_3a_primes.py` shows the single-call pattern
(`RLM(...) → .completion(prompt)`).

### Gap 5 — Disk and log bloat at scale *(operational)*

- `cleanup_mode` defaults to `"keep"` (`rlm/core/config.py:67`). For 500
  instances at, say, 100 MB each (repo + deps + snapshots), that's 50 GB.
  Set `cleanup_mode="delete"` or `"tar"` for batch runs.
- `RLMLogger` writes one JSONL per completion with the full trajectory
  (every action, every observation). For deep trajectories these can be
  multi-MB per instance. Acceptable; just plan for it.
- Per-turn git snapshots (`_rlm_state/snapshots/`) accumulate inside each
  workspace. Fine if `cleanup_mode != "keep"`; otherwise they amplify Gap 5a.

### Gap 6 — Model-format fragility *(known, partially in-flight)*

Recent commit `c53c6a9` notes Qwen3.6 fails on output format. The XML
`<action>` parser is lenient (tag-pair scan, not strict XML), but models
that wrap output in code fences, add markdown, or close tags inconsistently
will still break. For SWE-Bench, expect to spend tuning time per model. Pin
to one or two well-behaved models (Claude, Gemma-4-31b are known to work)
for v1.

### Gap 7 — Timeouts and budgets *(tuning)*

- `DockerConfig.exec_timeout_seconds = 300` per shell call. SWE-Bench test
  suites for some instances (e.g., django, sympy) take many minutes — may
  need to bump.
- `max_iterations = 30` default may be too low. SWE-agent typically uses
  50–100 turns for SWE-Bench. Worth making configurable per benchmark.
- `max_budget` and `max_timeout` should be set per-instance to bound cost.

### Gap 8 — Task framing prompt *(content, not code)*

The system prompt is generic. SWE-Bench tasks benefit from task-specific
guidance: "the repo is at `/workspace/repo`, the failing tests are listed
above, modify source files (not tests), the patch is auto-extracted from
git." This goes in the *user prompt* (the SWE-Bench instance template), not
the system prompt — keep the substrate task-agnostic. Several published
SWE-Bench prompt templates exist to crib from (SWE-agent, OpenHands,
Aider).

---

## Critical files for the implementation work (when authorized)

- `rlm/environments/docker_workspace.py` — needs a hook for repo seeding
  (Gap 1) and possibly a `pre_setup` callable; image is already configurable
  through `DockerConfig` (Gap 2).
- `rlm/core/config.py` — may want a new `SWEBenchConfig` or just per-run
  overrides; bump default `exec_timeout_seconds` is per-config.
- `rlm/core/rlm.py` — no changes strictly required if the harness wraps it
  externally.
- `docker/workspace_image/` — broker files that need to be layered into
  per-instance composite images (Gap 2 option B).
- `rlm/workspace_tools/final.py` — no change if patch is extracted host-side
  via `git diff` (recommended).

## What I'd build, in order, if asked to implement

1. **Smallest viable PoC** (1–2 days):
   - Pick one SWE-Bench Verified instance.
   - Manually clone repo to a workspace dir; run `RLM` against it on the
     existing `rlm-workspace:0.1.0` image with task-specific prompt.
     Inspect whether agent can produce a meaningful patch.
   - Extract patch host-side via `git diff`.
   - Confirm the official SWE-Bench evaluator accepts the prediction format.

2. **Per-instance image build pipeline** (2–4 days): Dockerfile template
   layering broker onto SWE-Bench base images; build script; image cache.

3. **Batch runner** (2–3 days): instance loader, parallel pool, prediction
   writer, resumability, error capture.

4. **Tuning pass** (open-ended): prompt iteration, turn/timeout limits, model
   selection, observability.

## Verification (for any implementation, when authorized)

- End-to-end: run on 5 known-easy SWE-Bench Verified instances, verify
  predictions.jsonl format passes `swebench.harness.run_evaluation` and at
  least one resolves (FAIL_TO_PASS turning green).
- Isolation: run two instances concurrently; confirm no container/port/
  workspace collisions.
- Cleanup: confirm disk usage returns to baseline after a 10-instance run
  with `cleanup_mode="delete"`.

---

# Terminal-Bench

> **Status: DONE (TB 2.0 / Harbor).** This section's gap analysis is now closed;
> the integration and validation are documented in
> [`terminal_bench_integration.md`](terminal_bench_integration.md). What follows
> is the original analysis for context — gap-by-gap closure notes appear inline
> below each subsection. The TB 1.x section was written before the
> TB 1.x → TB 2.0 / Harbor distinction was clear; **Harbor** is what we built
> against, not the tmux-keystroke 1.x harness.

Terminal-Bench (laude-institute/terminal-bench) is ~80–100 terminal tasks. Each
task ships as a directory with a `Dockerfile` (defining the environment), an
`instruction.md` (the goal handed to the agent), and a `tests/` directory plus
`run-tests.sh` that the harness runs *after* the agent finishes to grade
pass/fail. The agent's only interface is a shell session inside the container.

## Fit with the current substrate

This is the **best fit** of the three benchmarks. The substrate already gives
the agent a shell inside a Docker container and is built around an action loop
that's exactly the right shape. Terminal-Bench's design assumption ("agent has
a shell, no other tools needed") is a strict subset of what RLM offers.

## Gaps

### Gap T1 — Per-task image, same as SWE-Bench (Gap 2) *(blocking)*

> **CLOSED.** Implemented as a shared, benchmark-agnostic helper in
> `eval/common/composite_image.py` — caller supplies `base_image` and
> `output_tag`; broker layer (split-interpreter design with uv-managed
> Python 3.11) is identical for every benchmark. TB-specific caller in
> `eval/terminal_bench/runner.py` builds the task's base image from its
> own `environment/Dockerfile`, then calls `build_composite(...)`.
> Composite Dockerfile template at `eval/common/Dockerfile.composite.template`.

Each Terminal-Bench task has its own `Dockerfile`. Same fix as SWE-Bench:
build composite images that add the broker to the task's base image, and
override `DockerConfig.image` per run. The composite-build pipeline is largely
shared with SWE-Bench — write it once.

A wrinkle: Terminal-Bench tasks generally don't share a base; each Dockerfile
is hand-rolled. So the composite step is `FROM <task_image> + COPY broker
+ CMD broker` per task. Build is ~5–30s per task; cache aggressively.

### Gap T2 — Tool surface mismatch *(soft, but worth thinking about)*

> **DECIDED: option A** (leave all tools enabled). For the substrate-validation
> goal, exercising the full toolbox is what we want. The runner's prompt
> framing tells the agent the host-side file tools (`read_file`/`write_file`/
> `edit_file`) operate on `/workspace` and won't reach the task's workdir, so
> the agent naturally falls back to `shell` for grader-visible work.

Terminal-Bench official harness exposes only a tmux/shell to the agent. The
RLM substrate also exposes `read_file`, `write_file`, `edit_file`, `python`,
`llm_query`, `rlm_query`. Two stances:

- **(A) Leave all tools enabled.** The substrate is a richer agent than what
  the benchmark "expects," but Terminal-Bench grades on *task completion in
  the container*, not on which tool was used. Results are still valid — they
  just measure RLM-with-its-full-toolbox, not "agent-with-bash-only."
- **(B) Disable non-shell tools** for direct comparability with the official
  Terminal-Bench leaderboard (which assumes bash-only agents). The tool
  registry in `rlm/workspace_tools/` is the place; would need a config flag
  to allowlist.

Decision is for the user. (A) is cheaper and more honest about what's being
measured ("RLM substrate on TB"). (B) makes leaderboard comparison apples-to-
apples.

### Gap T3 — Result extraction *(easy)*

> **CLOSED via pre-cleanup callback on `RLM.completion()`.** New parameter
> `pre_cleanup_callback: Callable[[DockerWorkspaceEnv], Any] | None` fires
> after the agent loop returns but before `env.cleanup()`. Its return value
> attaches to the result as `pre_cleanup_result`. The TB runner's grader
> follows the Harbor-canonical contract: `docker cp tests/` into the live
> container, `bash /tests/test.sh`, read `/logs/verifier/reward.txt`. Works
> for both pytest-bootstrap and pure-shell tasks. See `rlm/core/rlm.py`
> docstring for the callback's full contract (incl. what it does NOT do —
> e.g. doesn't fire on `_run_loop` exceptions).

Terminal-Bench grades by running the task's `run-tests.sh` *after the agent
declares done*. So:

1. Agent calls `final` → loop ends.
2. Host runs `docker exec <container> bash /tests/run-tests.sh`
   (or whatever the task's grader is) before `cleanup()`.
3. Capture exit code → pass/fail.

Concretely, this means: don't `cleanup()` immediately on `final` — run the
grader first. Either a `pre_cleanup` hook on `WorkspaceConfig`, or just have
the harness call `env.exec_in_container(...)` (already public:
`docker_workspace.py:376`) before letting RLM tear down.

### Gap T4 — Batch harness (shared with SWE-Bench)

> **CLOSED for TB (single-task / sequential).** `eval/terminal_bench/runner.py`
> accepts `--tasks <path>...` and runs each sequentially, writing
> `result.json` per task + an aggregate `summary.jsonl`. No parallelism yet
> (3-task validation didn't need it). SWE-Bench runner not built.

Same shape: iterate tasks, call `RLM.completion(instruction)`, run grader,
record pass/fail. Should share most code with the SWE-Bench runner. Lighter
disk footprint than SWE-Bench (no big repo clones).

### Gap T5 — Timeout / turn budget tuning

> **CLOSED for the validated tasks.** Runner reads `agent.timeout_sec` and
> `verifier.timeout_sec` from each `task.toml` and wires them into
> `DockerConfig.exec_timeout_seconds` + the grader callback's timeout
> respectively. `max_iterations` is a CLI flag (`--max-iterations`, default 30).
> The 3 validated tasks ran in ≤11s wall and ≤15 iterations; tuning may be
> needed for harder TB 2.0 tasks.

Terminal-Bench tasks vary from 30 seconds to ~10 minutes. Default
`exec_timeout_seconds=300` may need bumping for long-running tasks; default
`max_iterations=30` is probably fine but worth confirming on a few hard
tasks.

## What I'd build, in order

> **DONE for TB 2.0 / Harbor.** All three steps below were built; results in
> [`terminal_bench_integration.md`](terminal_bench_integration.md).

1. PoC: pick 3 tasks across difficulty tiers, build composite images by
   hand, run RLM, verify graders fire.
2. Composite-image build script (shared with SWE-Bench).
3. Batch runner extension (parameterize over benchmark type).

---

# AIME 2025

AIME 2025 is 15 math problems with integer answers 0–999. It's a pure
reasoning benchmark — no code is *required*, though code execution
(SymPy, brute-force search) is often very useful. Output is a single integer
per problem; grading is exact-match.

## Fit with the current substrate

This is a **poor fit** in the sense that you'd be using a heavyweight
container-backed agent loop to do a thing that a single LM call (with maybe
a few rounds of self-consistency) is purpose-built for. The substrate's value
proposition over plain LM CoT is:

- Code execution inside a real Python (the `python` tool) — useful for AIME
  problems that yield to brute-force or symbolic computation.
- Workspace memory across turns — useful for multi-step problems where the
  model wants to scratch.
- Recursive `rlm_query` — could decompose a hard problem into sub-problems.

Whether any of this beats a well-tuned single-shot CoT or self-consistency
on AIME is an empirical question. AIME 2025 results are *dominated* by
reasoning-tuned models (o1, R1, Claude with extended thinking) at low- or
zero-tool-use settings. The substrate is not an obvious win here.

## Gaps (small)

### Gap A1 — No repo, no special image *(none, basically)*

The default `rlm-workspace:0.1.0` image is sufficient. No repo to clone, no
per-instance image, no graders to run. Just pass the problem statement as
the prompt.

### Gap A2 — Result extraction *(trivial)*

Tell the agent (in the user prompt template) to put a single integer in
`<answer>...</answer>`. Host parses, compares to the gold answer, done. No
substrate change.

### Gap A3 — Batch harness *(trivial)*

15 problems. A single Python script over `for problem in problems: rlm =
RLM(...); result = rlm.completion(problem); ...`. No need for parallelism
or per-instance container caching — could even be sequential. Logging
per-problem is enough.

### Gap A4 — Self-consistency / pass@k

For meaningful AIME numbers you typically run k=8 or k=16 samples and
report majority-vote / pass@k. Easy to wrap around `RLM.completion`: just
call it k times per problem. Worth deciding upfront whether reporting
single-shot or pass@k.

### Gap A5 — Cost concern *(real)*

Each substrate run spins up a Docker container. For 15 problems × 16
samples = 240 container starts. ~1–2s startup each isn't crippling but is
silly overhead for a benchmark that doesn't need code execution most of
the time. Could either:

- Accept the overhead (it's measurable but not blocking).
- Add a "no-container" fast path that bypasses `DockerWorkspaceEnv` for
  pure-text tasks. This is a substantial substrate change and probably
  not worth it.

## Recommendation

If the goal is **headline AIME numbers**, AIME isn't the benchmark to
showcase the substrate on — a plain LM call will be both cheaper and likely
competitive. If the goal is **demonstrating that the substrate doesn't
*hurt* on reasoning tasks** (i.e., the agent loop can recover plain-CoT
performance when no tools are needed), then a small ablation: same model,
substrate vs. raw, on AIME 2025, is a fine sanity check. Build effort: a
few hours.

---

# Cross-benchmark summary

| Concern | SWE-Bench | Terminal-Bench | AIME 2025 |
|---|---|---|---|
| Per-instance Docker image | required (composite build) | required (composite build) | not needed |
| Workspace seeding | clone repo @ commit | task Dockerfile handles it | none |
| Result extraction | host-side `git diff` | run task's grader script | parse `<answer>` for integer |
| Batch harness | required | required (shares code with SWE-Bench) | trivial |
| Substrate fit | good | best | weak (overkill) |
| Effort to PoC | ~1 week | ~3–5 days | ~hours |
| Effort to production | 2–3 weeks | 1–2 weeks (if SWE-Bench done first) | <1 week |

The composite-image build pipeline (broker layered onto a benchmark-provided
base image) is the **single biggest reusable piece**. Build it for SWE-Bench;
Terminal-Bench gets it nearly for free.

The natural ordering is: **Terminal-Bench → SWE-Bench → AIME**.
Terminal-Bench is the cleanest fit and forces you to nail the composite-image
build + per-task harness. SWE-Bench reuses both with the added complexity of
repo seeding and patch extraction. AIME is a small ablation any time after.
