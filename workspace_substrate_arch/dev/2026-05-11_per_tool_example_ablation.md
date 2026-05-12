# 2026-05-11 — Per-tool example ablation: prompt change vs. parser-fix-only baseline

Re-ran the four Phase 3 substrate tasks against the same four Qwen3-family
models, with one and only one change: each tool's system-prompt entry now
includes an annotated example showing the correct attribute syntax and (for
body-bearing tools) the correct body format. The hypothesis was that
underspecified tool descriptions — not model capability — were the dominant
driver of format-related failures across the Qwen3 family.

Both the parser fixes from 2026-05-11 and the system prompt's high-level
shape are held constant; only `rlm/workspace_tools/*.py:SPEC.example` is
new and `rlm/utils/prompts.py:_format_tool_descriptions` renders it. The
golden rollout fixture was regenerated; full test suite passes (347/3 skip).

Prior post-fix sweep (parser fixes A+C only, no example): 2026-05-11_01:0X
runs catalogued in each model's prior eval report. This run starts
2026-05-11_21:29.

## Headline

| Model | Prior post-fix (parser only) | This run (parser + examples) | Δ |
|---|---|---|---|
| Qwen3-8B (dense) | 1/4 pass, 13 retries | **4/4 pass, 0 retries** | **+3 tasks, −13 retries** |
| Qwen3.5-9B (VL) | 2/4 pass, ~5 retries | **4/4 pass, 1 retry** | **+2 tasks, ~−4 retries** |
| Qwen3-32B (dense) | 4/4 pass, 4 retries | **4/4 pass, 0 retries** | same pass, **−4 retries** |
| Qwen3.5-27B (VL) | 4/4 pass*, 8 retries | **4/4 pass*, 0 retries** | same pass, **−8 retries** |
| **Total** | **11/16 pass, ~30 retries** | **16/16 pass, 1 retry** | **+5 tasks, −97% retries** |

\* Both runs of Qwen3.5-27B return 336 (top-level defs only) on 3d, vs the
canonical 420; this is a task-comprehension error unaddressed by the prompt
change. See §5.

## 1. Per-task table (this run)

| Model | 3a | 3b | 3c | 3d | total retries | total LM calls |
|---|---|---|---|---|---|---|
| Qwen3-8B | PASS (2 turns, 0r) | PASS (5 turns, 0r) | PASS (4 turns, 0r) | PASS (3 turns, 0r) | **0** | 24 |
| Qwen3.5-9B | PASS (3 turns, 0r) | PASS (15 turns, 0r) | PASS (16 turns, 1r) | PASS (4 turns, 0r) | **1** | 74 |
| Qwen3-32B | PASS (3 turns, 0r) | PASS (4 turns, 0r) | PASS (4 turns, 0r) | PASS (3 turns, 0r) | **0** | 26 |
| Qwen3.5-27B | PASS (3 turns, 0r) | PASS (7 turns, 0r) | PASS (5 turns, 0r) | PASS\* (3 turns, 0r) | **0** | 33 |

\* 27B 3d returns 336/420 functions (top-level only).

Wall-clock totals (this run): 8B 251s · 9B 256s · 32B 705s · 27B 232s. The
32B 3c took 268s with sequential rlm_query spawns; if parallelism is enabled
at the substrate level (separate change) it would drop into the 50–100s
range like 8B and 9B.

## 2. The four spot-checks from the plan, verbatim from traces

All four are direct trace lookups against
`_setup_runs/logs/<model>/rlm_3?_<model>_2026-05-11_2*.jsonl`.

**Spot 1 — Qwen3-8B 3b iter 1 first action (was four consecutive child-element failures pre-change):**

```
<action tool="read_file" path="_rlm_query_0.txt" start_line="1" end_line="1000"/>
```

Attribute form on the very first attempt, with `path` correctly placed. The
model also batched a sibling `read_file _rlm_query_1.txt` in the same turn.
Pre-change post-fix the model emitted four well-formed-but-schema-broken
`<action tool="read_file"><path>…</path><start_line>…</start_line></action>`
elements in sequence and exhausted the retry budget.

**Spot 2 — Qwen3.5-9B 3a iter 2 python body (was six consecutive `<code>`/`<preset>` SyntaxError failures pre-change):**

```
def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            ...
```

Raw Python directly under the `<action tool="python">` open tag. No
wrapper. The substrate ran it to completion (observation error: None).
Pre-change this model emitted `<code>`, `<preset>`, `<code>`, `<![CDATA[`,
`<error>`, `<code>` in six consecutive turns of accidental Markdown/XML
fencing.

**Spot 3 — Qwen3.5-27B 3a iter 1 (was 1 retry every single task pre-change with `file=` → `path=` correction):**

```
parse_retries: 0
dispatched first action tag: <action tool="read_file" path="_rlm_query_0.txt"/>
```

Zero retries. The previously universal `file=` mistake is gone — the example
in the tool description states `Required attr: path` explicitly. Confirmed
across all four 27B tasks: 0 total retries.

**Spot 4 — Qwen3-32B 3a iter 1 (was 1+ retry with think-poisoning fallout in pre-change):**

```
parse_retries: 0
dispatched first action tag: <action tool="read_file" path="_rlm_query_0.txt" />
```

Zero retries. Pre-change, 32B's verbose `<think>` blocks routinely contained
bare `<action>` references that the scanner caught before the real action;
post-think-strip plus an explicit example means the model now emits the
right tag the first time.

## 3. The lone retry

Qwen3.5-9B 3c parent iter 3 emitted:

```
<action tool="read_file">path="_rlm_query_0.txt"</action>
```

`path="..."` placed inside the body rather than as an attribute on the
opening tag — a near-miss typo. Recovered on the single retry, no
cascading effect. This is the only parse failure in the entire 16-task
sweep.

## 4. Token / wall-clock per task

| Model · Task | Tokens in | Tokens out | LM calls | Wall (s) |
|---|---|---|---|---|
| 8B · 3a | 2,720 | 1,523 | 2 | 21.9 |
| 8B · 3b | 10,782 | 2,549 | 5 | 34.4 |
| 8B · 3c | 22,074 | 7,290 | 14 | 102.0 |
| 8B · 3d | 15,672 | 6,976 | 3 | 93.0 |
| 9B · 3a | 4,621 | 958 | 3 | 15.9 |
| 9B · 3b | 49,634 | 1,961 | 15 | 35.9 |
| 9B · 3c | 131,797 | 12,696 | 52 | 183.0 |
| 9B · 3d | 24,565 | 1,237 | 4 | 20.9 |
| 32B · 3a | 4,371 | 1,111 | 3 | 53.4 |
| 32B · 3b | 7,554 | 2,173 | 4 | 103.5 |
| 32B · 3c | 25,612 | 5,701 | 16 | 268.7 |
| 32B · 3d | 15,426 | 5,981 | 3 | 279.7 |
| 27B · 3a | 4,592 | 546 | 3 | 24.9 |
| 27B · 3b | 14,183 | 959 | 7 | 45.4 |
| 27B · 3c | 35,223 | 2,716 | 20 | 124.5 |
| 27B · 3d | 16,406 | 795 | 3 | 37.4 |

The token deltas vs. the parser-only baseline are mostly favourable (no
wasted retry calls), but 9B 3c jumped from 27 calls / 31k input pre-change
to 52 calls / 131k input here — the model spent many turns doing
`list_directory` exploration that the prior run skipped. This is a routing
choice the model is now free to make because format errors no longer
dominate the trace; not a regression in format quality. All five summaries
were produced and collated correctly.

## 5. What did *not* improve, and why

**Qwen3.5-27B 3d: still 336/420.** The model continues to use
`re.match(r'^def\s+', line)` which matches only top-level `def` statements
and excludes class methods. The task prompt says "every `def NAME(...)`
declaration"; the model interprets that as top-level only. The system-prompt
examples never touched task comprehension, so this is unchanged from the
prior two runs. Independent of the prompt change.

## 6. Final-artifact verification

All sixteen runs reported `final_artifacts` via the substrate's confirmed
artifact tracking:

| Model | 3a | 3b | 3c | 3d |
|---|---|---|---|---|
| 8B | `primes.txt` | `sort.py`, `test_sort.py` | `collated.txt` | `declarations_by_module.json` |
| 9B | `primes.txt` | (no `<artifact>` tag, but tests in observations pass) | `collated.txt` | `declarations_by_module.json` |
| 32B | `primes.txt` | `sort.py`, `test_sort.py`, `test_results.txt` | `collated.txt` | `declarations_by_module.json` |
| 27B | `primes.txt` | `sort.py`, `test_sort.py` | `collated.txt` | `declarations_by_module.json` |

3d function counts: 8B 420, 9B 420, 32B 420, 27B 336 (see §5).

## 7. Verdict

A one-shot annotated example per tool — ~50 tokens added to the system
prompt per model call — converts every Qwen3-family model in this set into
a substrate-fluent agent. The result is stronger than either of the prior
two interventions:

- Parser-A+C fixes (2026-05-11 first-half) closed the worst think-poisoning
  pathologies but left the underlying schema confusion in place: 8B was
  still 1/4, 9B was 2/4, retry budgets were still being burned every task.
- This prompt change removes the schema confusion at its source: the
  models do not retry-and-recover; they emit the right format the first
  time.

For the 7–9B research direction, **Qwen3-8B is now substrate-viable**: 4/4
correct on all four phase-3 tasks, zero parse retries, 8.4 minutes of
wall-clock total. The previously load-bearing parser changes (Fix A
think-strip, Fix C tolerant cluster) are still doing useful work — they
catch the lone 9B 3c retry — but they're no longer the difference between
working and not working for these models.

The unaddressed failure modes (task comprehension on 3d for 27B; the
dangling-`</think>` parser case noted in the earlier analysis) are
separate from format and orthogonal to this change.

## 8. Files

| Model | Logs |
|---|---|
| Qwen3-8B | `_setup_runs/logs/qwen-3-8b/rlm_3{a,b,c,d}_qwen-3-8b_2026-05-11_21-29-*.jsonl` |
| Qwen3.5-9B | `_setup_runs/logs/qwen-3-5-9b/rlm_3{a,b,c,d}_qwen-3-5-9b_2026-05-11_21-34-*.jsonl` |
| Qwen3-32B | `_setup_runs/logs/qwen-3-32b/rlm_3{a,b,c,d}_qwen-3-32b_2026-05-11_21-41-*.jsonl` |
| Qwen3.5-27B | `_setup_runs/logs/qwen-3-5-27b/rlm_3{a,b,c,d}_qwen-3-5-27b_2026-05-11_21-55-*.jsonl` |

Runner stdouts: `_setup_runs/logs/<alias>/runner_stdout.log`.

Reproduce:
```bash
conda activate RLM_substrate
bash _setup_runs/run_all_qwen.sh
python _setup_runs/trace.py _setup_runs/logs/qwen-*/rlm_3*_2026-05-11_2*.jsonl
```

## 9. Code change

Three commits' worth, all small:

- `rlm/workspace_tools/__init__.py`: added `example: str = ""` to `ToolSpec`.
- `rlm/workspace_tools/{list_directory,read_file,write_file,append_file,edit_file,shell,python,llm_query,rlm_query,final}.py`: set `example=` per tool.
- `rlm/utils/prompts.py:_format_tool_descriptions`: appends `\n  <example>` after the tool's short description when present.
- `tests/fixtures/golden_rollout.jsonl`: regenerated via `python tests/test_e2e_rollout.py --regenerate-golden`. Diff is exclusively system-prompt content.
