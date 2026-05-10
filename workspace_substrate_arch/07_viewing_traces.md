# 07. Viewing & navigating RLM traces

Every `RLM(...).completion(...)` call leaves three persistent artifact streams you can inspect after the fact: a **JSONL log**, a **per-run workspace with one git commit per turn**, and (when a run uses recursion) **per-child workspace snapshots** copied back into the parent's artifact tree. This doc is the map.

## What gets persisted, where

| Stream | Default location | Controlled by | What it contains |
|---|---|---|---|
| JSONL log | `<log_dir>/rlm_<UTC date>_<time>_<uuid>.jsonl` | `RLMLogger(log_dir=...)` | metadata header + one JSON object per turn (actions, observations, snapshot, parse_attempts, final_answer) |
| Per-run workspace | `~/.rlm/workspaces/run_<id>/` | `DockerConfig.workspace_root_base` + `cleanup_mode` | the actual files the model wrote/read; `.git/` with one commit per turn; `_rlm_artifacts/`; `_rlm_state/provenance.json`; `_rlm_state/action_log.jsonl` |
| Spilled observations | `<workspace>/_rlm_artifacts/_observations/` | `ObservationConfig.max_observation_chars` (default 16 KB) | full text of any tool output that exceeded the cap; the in-log observation gets replaced by a short summary + path |
| Recursion children | `<parent workspace>/_rlm_artifacts/children/child_<depth>_<n>/` | `RecursionConfig` | only the artifacts the child explicitly exported via `<artifact path="..."/>` children of `final` |

If you didn't pass `log_dir=...` to `RLMLogger`, the trajectory is in-memory only on `result.metadata` — no JSONL on disk.

If `DockerConfig.cleanup_mode != "keep"`, workspaces are torn down at end-of-run (`"delete"` removes them, `"tar"` archives them). For tracing you almost always want `cleanup_mode="keep"`.

## Picking your tool

| Goal | Use |
|---|---|
| "Did this run succeed and what tools did the model use?" | `_setup_runs/trace.py` (one-line-per-turn summary) |
| "Show me the model's full XML response for turn 3" | `jq` on the JSONL |
| "What's the actual content of the file the model wrote?" | `cat` in the workspace dir |
| "What changed between turn 4 and turn 5?" | `git log --oneline` + `git show <sha>` inside the workspace dir |
| "Where did the model get stuck or error?" | grep the JSONL for `"error":` non-null, or use the trace script's `ERR=` marker |
| "What did the model see as observation after each action?" | JSONL `observations[i].stdout` / `.stderr` |
| "I want a UI" | the Next.js visualizer at `visualizer/` |

## CLI navigation

### One-line-per-turn summary

`_setup_runs/trace.py` walks a JSONL and emits a compact per-turn line. Output looks like:

```
=== rlm_2026-05-09_17-12-59_48e29be2.jsonl ===
backend       : vllm model=gemma-4-31b
max_iter/depth: 30/1
iterations    : 5

  turn 01  (2.5s)  tools=[read_file]  changed=['_rlm_query_0.txt', '_rlm_state/action_log.jsonl']
  turn 02  (22.4s) tools=[shell]  ERR=1  changed=[...]
      [error: shell] shell exited with code 127
  turn 03  (6.3s)  tools=[shell]  changed=[...]
  turn 04  (40.3s) tools=[write_file,write_file,shell]  changed=[..., '…']
  turn 05  (10.2s) tools=[final]  changed=[...]  FINAL
```

What each marker means:
- `tools=[...]` — the ordered list of tools the model emitted that turn.
- `parse_retries=N` — how many `<action>` parse retries happened before the model produced something the parser accepted (0 on a clean turn).
- `ERR=N` — count of observations with non-null `error`. The line below shows the first error.
- `spills=N` — count of observations whose stdout was a "spilled to `_rlm_artifacts/_observations/`" pointer.
- `changed=[...]` — files the per-turn git snapshot recorded as modified (capped at 3 + `…`).
- `FINAL` — this turn produced a `final_answer`.

Run on multiple files at once: `python _setup_runs/trace.py _setup_runs/logs/*.jsonl`.

### Direct JSONL inspection with `jq`

The first line is always a metadata header: `{"type":"metadata", ...}`. Every subsequent line is `{"type":"iteration", ...}`. Useful one-liners:

```bash
# Header (model, backend, limits)
head -1 logs/rlm_*.jsonl | jq .

# Turn count
tail -n +2 logs/rlm_*.jsonl | wc -l

# Tool usage histogram across the run
jq -r 'select(.type=="iteration") | .actions[].tool' logs/rlm_*.jsonl | sort | uniq -c

# All observations that errored
jq -c 'select(.type=="iteration") | .observations[] | select(.error != null) | {tool, error}' logs/rlm_*.jsonl

# Wall-clock per turn
jq -r 'select(.type=="iteration") | "\(.iteration)\t\(.iteration_time)"' logs/rlm_*.jsonl

# Just the model's response text for turn 3
jq -r 'select(.type=="iteration" and .iteration==3) | .response' logs/rlm_*.jsonl

# Action body for turn 2's first action (e.g., the python source the model wrote)
jq -r 'select(.type=="iteration" and .iteration==2) | .actions[0].body' logs/rlm_*.jsonl
```

### Workspace inspection (the actual files)

`cd ~/.rlm/workspaces/run_<id>/` and you're standing in the workspace as the container saw it. Layout:

```
.git/                          # one commit per turn (see below)
_rlm_query_0.txt               # the root task (and _rlm_query_<N>.txt for context)
_rlm_state/
  action_log.jsonl             # every action ever dispatched, in order
  provenance.json              # role-tagged file ownership (user/assistant/system/child)
  snapshots/                   # internal snapshot blobs (don't touch)
  _tmp/                        # the actual scripts the model executed (e.g. python_t2.a1.py)
_rlm_notes/                    # scratch notes the model wrote for itself
_rlm_artifacts/
  _observations/               # spilled observations (see below)
  children/child_<d>_<n>/      # exported artifacts from rlm_query children
<everything else>              # the model's actual outputs (sort.py, primes.txt, ...)
```

`_rlm_state/_tmp/` is particularly useful: it preserves the literal `python` and `shell` scripts each action ran, named `<tool>_t<turn>.a<action-index>.<ext>`. So `python_t2.a1.py` is "the python source for turn 2's first action."

`_rlm_state/action_log.jsonl` is a chronological action ledger that's independent of the per-run logger — even if you didn't pass a `RLMLogger`, this file exists.

`_rlm_state/provenance.json` answers "who created/last-modified each file?" with one of `user`, `assistant`, `system`, `child`. Useful when sorting out which files the model authored vs. which the runtime seeded.

### Per-turn git snapshots

The substrate makes one git commit per turn under `<workspace>/.git/` with a deterministic message. Walk the history:

```bash
cd ~/.rlm/workspaces/run_<id>
git log --oneline                 # one entry per turn
git show <sha> --stat             # files changed that turn
git show <sha> -- sort.py         # how sort.py looked after that turn
git diff <sha1> <sha2>            # diff between any two turns
git checkout <sha>                # check the workspace out as of that turn (then `git checkout main` to return)
```

This is the one place where you can see the workspace's state *as the model saw it at the start of turn N+1* — the JSONL's `snapshot.commit_sha` field on each iteration tells you which sha to look up.

### Recursion: navigating children

If a run used `rlm_query`, look under `<workspace>/_rlm_artifacts/children/`:

```
_rlm_artifacts/children/
  child_2_1/                   # depth 2, child index 1 of that turn
    summary_1.txt              # whatever the child exported via <artifact path="..."/>
  child_2_2/
    summary_2.txt
  ...
```

Only files the child explicitly exported are present — the child's full workspace lives elsewhere (transient by default). The parent's JSONL observation for the `rlm_query` action lists each child's exported artifact under `observations[i].artifacts` and includes a path-mapping table in `observations[i].stdout`.

**Caveat (current scaffold):** `observations[i].rlm_calls` in the JSONL is currently empty even when children ran. Per-child trajectories aren't embedded in the parent log. To trace children, you have to either re-run with extra logging or read the parent's stdout summary and infer.

### Spilled observations

If a single tool call returned more bytes than `ObservationConfig.max_observation_chars` (default 16 KB), the substrate writes the full output to `<workspace>/_rlm_artifacts/_observations/<file>` and replaces the in-log observation with a short summary + path. Run-level grep:

```bash
ls ~/.rlm/workspaces/run_<id>/_rlm_artifacts/_observations/ 2>/dev/null
jq -r 'select(.type=="iteration") | .observations[].stdout' logs/rlm_*.jsonl | grep -i "spilled"
```

If neither shows results, no spills happened.

## The visualizer

A Next.js single-page app under `visualizer/`.

```bash
conda activate RLM_substrate
conda install -y -c conda-forge "nodejs>=20"     # if node isn't on PATH
cd visualizer && npm install
npm run dev                                       # → http://localhost:3001
```

Then load any `*.jsonl` from `<log_dir>/` via the in-app picker. The visualizer is a pure client-side renderer — there's no API endpoint; it parses the JSONL in the browser. The TS types in `visualizer/src/lib/types.ts` mirror the Python `to_dict()` schemas 1:1, so any field you can `jq` from the JSONL is also accessible in the UI.

What the visualizer currently shows well: per-turn actions/observations, tool breakdown, parse-attempt list, snapshot diffs.

What it doesn't show (because the producer doesn't emit them): per-child trajectories during recursion (see B4 in `dev/2026-05-09_first_run_traces_gemma-4-31B-it.md`).

## A worked example

Suppose you want to answer "how did the model recover after the pytest-not-found error in the mergesort run?"

1. Find the run. From the trace summary or the file timestamps:
   `_setup_runs/logs/rlm_2026-05-09_17-12-59_48e29be2.jsonl`.
2. Locate the error turn:
   `python _setup_runs/trace.py _setup_runs/logs/rlm_2026-05-09_17-12-59_48e29be2.jsonl` → ERR=1 on turn 2.
3. Inspect the failing action:
   `jq -r 'select(.type=="iteration" and .iteration==2) | .actions[0].body' _setup_runs/logs/rlm_2026-05-09_17-12-59_48e29be2.jsonl`
   → `pytest test_sort.py` (the model ran tests before writing them).
4. Inspect the recovery turn:
   `jq -r 'select(.type=="iteration" and .iteration==3) | .actions[0].body' …`
   → `pip install pytest`.
5. (Optional) Walk the workspace:
   `cd ~/.rlm/workspaces/run_1778371979854_d052faa7 && git log --oneline` shows one commit per turn; `git show <turn-3-sha>` shows the recovery commit's effect.

That same flow — "trace.py for the overview, jq for any specific field, workspace+git for the file content as the model saw it" — works for any run.

## Summary

| You want to see... | Look at |
|---|---|
| pass/fail at a glance | `trace.py` |
| every action the model emitted | JSONL `actions[]` via `jq` |
| every observation back from the substrate | JSONL `observations[]` via `jq` |
| the actual file contents | the workspace dir under `~/.rlm/workspaces/run_<id>/` |
| state as of a specific turn | `git show` inside the workspace |
| spilled tool outputs | `<workspace>/_rlm_artifacts/_observations/` |
| recursion outputs | `<workspace>/_rlm_artifacts/children/` |
| any of the above with a UI | the visualizer at `localhost:3001` |
