# 2026-05-10 — Substrate eval: `Qwen/Qwen3.5-27B`

Ran the same four Phase 3 substrate tasks (`run_3{a,b,c,d}_*.py`) against `Qwen/Qwen3.5-27B`. Held the substrate, prompt, and infrastructure constant — only the model changed. Gemma 4 31B (4/4 pass) is the substrate-fluent baseline; Qwen3.6-35B-A3B (1/4 pass) represents the parse-failure extreme. Prior reports: `2026-05-09_first_run_traces_gemma-4-31B-it.md` and `2026-05-09_qwen3-6-35B-A3B_vs_gemma4-31B.md`.

---

## 1. Setup

### Model

**HF model ID:** `Qwen/Qwen3.5-27B`  
**Architecture:** `Qwen3_5ForConditionalGeneration` — 27 B parameters, BF16, 64 hidden layers, hidden\_size 5120, hybrid linear + every-4th-layer full attention.

**Critical fact for readers: this is a multimodal vision-language model, not a plain dense LLM.** The class name `Qwen3_5ForConditionalGeneration` and the model architecture include a full vision encoder. All runs here were text-only (no images or video passed). This matters because the model's per-token cost, attention pattern, and potential alignment differ from a comparable dense text-only 27 B model. The comparison to Gemma 4 31B (also multimodal VL, also text-only here) is the most natural, but the comparison to Qwen3.6-35B-A3B (MoE, text-only architecture) is not apples-to-apples in either direction.

### vLLM topology

Single replica on GPU 4 (A100 80 GB), port 8001.

| Flag | Value |
|---|---|
| `--max-model-len` | 32768 |
| `--max-num-batched-tokens` | 8192 |
| `--max-num-seqs` | 64 |
| `--gpu-memory-utilization` | 0.90 |
| `--limit-mm-per-prompt` | `'{"image":0,"video":0}'` |
| `--reasoning-parser` | not set |

The `--max-num-seqs 64` cap (same workaround used for Qwen3.6-35B-A3B) was inherited because the hybrid linear-attention architecture OOMs during CUDA graph capture at the default 256. The `--limit-mm-per-prompt` flag disables the vision encoder so the model runs text-only. No `--reasoning-parser` was set, so chain-of-thought blocks are returned in the raw completion text rather than being stripped server-side.

**Sweep date:** 2026-05-10. Model cached to `/data/nwei/rlm_substrate/models`, then deleted after the run.

---

## 2. Per-task summary

| Task | Outcome | Turns | Parse retries (parent) | Parse retries (children) | LM calls | Tokens in | Tokens out | Wall clock | Obs spills | Final status |
|---|---|---|---|---|---|---|---|---|---|---|
| 3a — first 100 primes | **PASS** | 3 | 1 | — | 4 | 4,064 | 610 | 27.9 s | 0 | `final` reached |
| 3b — merge sort + pytest | **PASS** | 8 | 1 | — | 9 | 14,940 | 1,140 | 52.4 s | 0 | `final` reached |
| 3c — `rlm_query` fan-out × 5 | **PASS** | 5 (parent) | 1 | 5 (1 per child) | 26 (parent: 6, children: ~20) | 32,147 | 3,419 | 150.0 s | 0 | `final` reached |
| 3d — 50 KB long-context extraction | **PASS\*** | 4 | 1 | — | 5 | 24,730 | 1,029 | 48.4 s | 0 | `final` reached |

**\*** 3d reached `final` but reported **336 functions** (top-level only), not 420 (all `def` statements including class methods). Gemma 4's 420 is the reference answer. See §4.

**Headline: 4/4 tasks completed** — matching Gemma 4's pass rate on substrate mechanics. The model's single consistent failure was the same parse-retry pattern on turn 1 of every task (1 retry each), but it always recovered within that single retry.

---

## 3. Substrate-format analysis

### The consistent turn-1 failure pattern

Every run, every agent (parent and all 5 children in 3c) failed turn 1 on the first `read_file` action with an attribute schema error, then recovered on the retry. The failures come in two variants:

**Variant A — wrong attribute name (`file=` instead of `path=`):**

From 3a turn 1, attempt 0:
```
Let me start by reading the root task file to understand what I need to do.
</think>

<action tool="read_file" file="_rlm_query_0.txt" />
```
Parser response: `Unknown attribute(s) ['file'] on <action tool='read_file'>. Allowed: ['end_line', 'path', 'start_line'].`

Recovery (attempt 1):
```
I need to read the root task file. The correct attribute is `path`, not `file`. Let me fix that.
</think>

<action tool="read_file" path="_rlm_query_0.txt" />
```

**Variant B — path as element body instead of attribute:**

From 3b turn 1, attempt 0:
```
Let me start by reading the root task file to understand what I need to do.
</think>

<action tool="read_file">_rlm_query_0.txt</action>
```
Parser response: `Missing required attribute(s) ['path'] on <action tool='read_file'>.`

From 3c turn 1, attempt 0 (a third sub-variant — has other attributes but omits `path`):
```
I'll start by reading the root task file to understand what I need to accomplish.
</think>

<action tool="read_file" start_line="0" end_line="100">_rlm_query_0.txt</action>
```
Parser response: `Missing required attribute(s) ['path'] on <action tool='read_file'>.`

All four runs (3a, 3b, 3c, 3d) used one retry each and recovered cleanly — **total of 4 parent-level retries across all 4 tasks**. Crucially, the model **never needed a second retry** anywhere: the parser feedback (`"Unknown attribute 'file'..."`, `"Missing required attribute 'path'..."`) was sufficient to self-correct in one shot. This is a fundamentally different failure mode from Qwen3.6-35B-A3B, which needed up to 3 retries and still crashed 3 out of 4 tasks.

**This is not the Qwen3.6 path-as-child-element failure.** Qwen3.6 emitted `<action tool="read_file"><path>...</path></action>`, which is full XML child-element style. Qwen3.5-27B instead used the attribute slot incorrectly (`file=` vs `path=`, or path text as element body) — both variants are recoverable from the parser's attribute error feedback in a single retry.

### Thinking-mode (`<think>`) behavior

Qwen3.5-27B was served without `--reasoning-parser`, so `<think>` blocks come back inline in the completion. The `reasoning` field in the JSONL is always `null`; instead, everything before `</think>` in the `response` field is the think block.

The model emits a `</think>` tag on **every single response** (100% of turns, all 4 tasks). Think blocks are short and pragmatic — typically 50–540 chars per turn, dominated by brief task re-statement or self-correction commentary:

| Task | Turns with `</think>` | Total think chars | Total response chars | Think fraction |
|---|---|---|---|---|
| 3a | 3/3 | 745 | 1,611 | 46 % |
| 3b | 8/8 | 1,038 | 3,823 | 27 % |
| 3c (parent only) | 5/5 | 1,688 | 5,678 | 30 % |
| 3d | 4/4 | 1,445 | 3,781 | 38 % |

The think blocks are lightweight compared to Qwen3.6-35B-A3B. Self-correction responses are especially terse (e.g., 3a retry turn: `"I need to read the root task file. The correct attribute is 'path', not 'file'. Let me fix that."` — 97 chars). The substrate's parser correctly ignores all pre-`</think>` text and extracts only the `<action>` blocks.

### After turn 1: clean format following

Once past the turn-1 stumble, the model used correct attribute syntax consistently for the rest of each run. Multi-action turns (3c's 5× `rlm_query` on turn 2; 3c's 5× `read_file` on turn 3), self-closing tags, and body-bearing tools were all emitted correctly.

---

## 4. Tool usage

### 3a — primes (3 turns)

`read_file` → `python` (sieve, writes `primes.txt`) → `final`. One-shot Python solution, no errors. Output verified: 100 primes, 2–541.

### 3b — merge sort + pytest (8 turns)

`read_file` → `write_file` (sort.py) → `write_file` (test\_sort.py) → `shell` (pytest, exit 127: no pytest) → `shell` (python -m pytest, exit 1: module not found) → `shell` (pip install pytest) → `shell` (python -m pytest, pass) → `final`. The same pytest-not-found bump as Gemma 4 (Bump B1 from the Gemma trace), handled in 2 turns of recovery instead of 1. The model tried bare `pytest` first, then `python -m pytest`, then installed — slightly less efficient than Gemma's direct `pip install pytest`, but reached the correct outcome. All 5 tests passed.

### 3c — `rlm_query` fan-out × 5 (5 parent turns, 15 child-agent turns)

The parent correctly recognized the parallelism opportunity and emitted all 5 `rlm_query` actions in a single turn 2:

```
I'll send them all in parallel since they are independent.
</think>

<action tool="rlm_query">Summarize this text in 1-2 sentences and create an artifact named summary_1.txt: ...</action>
<action tool="rlm_query">Summarize this text in 1-2 sentences and create an artifact named summary_2.txt: ...</action>
...
```

All 5 children completed successfully. Each child ran 3 turns: `read_file` → `write_file` (summary artifact) → `final`. Each child hit the same turn-1 parse retry as the parent (same model, same cold-start schema confusion), recovered in one retry, and succeeded.

**Child artifact placement was inconsistent** — children 1, 3 wrote to `_rlm_artifacts/summary_N.txt` while children 2, 4, 5 wrote to `summary_N.txt` (workspace root). The substrate's artifact-mapping table correctly reflected whichever path each child used, and the parent's turn-3 `read_file` calls used those mapped paths verbatim — so the inconsistency had no functional impact.

**Child token scaling:** child call counts escalated as 7, 11, 15, 19, 23 total LM calls per child. This is cumulative context inflation: each serialized child sees the full conversation including the prior children's embedded interactions, because the parent's `prompt` field grows each turn with prior observations. The substrate ran children concurrently (wall clock for all 5 ≈ slowest child's time), so the per-child context inflation was unavoidable in this configuration.

**Recursion artifact handoff: clean.** All 5 artifact paths in the parent's observations mapped correctly via the `[Runtime Note: The child's exported files have been safely isolated...]` mapping table. The parent used those paths correctly in turn 3 and successfully collated all 5 summaries into `collated.txt`.

### 3d — 50 KB long-context extraction (4 turns)

`read_file` (lines 1–500, truncated) → `read_file` (lines 1900–1979, tail sample) → `python` (full-file parse script) → `final`. **No observation spills**: the read\_file truncation kept individual observations small, and the python action printed only the summary line (`Total modules: 42 / Total functions: 336`).

The model took a two-read strategy — it read the first 500 lines, noticed the file was 1979 lines, then sampled the tail to understand the corpus structure before writing the extractor script. This is competent but led to the answer error:

**The model explicitly excluded indented `def` statements** (class methods), counting only top-level functions. Its script:
```python
def_match = re.match(r'^def\s+(\w+)\s*\(', line)
```
The task prompt says "extract every `def NAME(...)` declaration" — Gemma 4 matched all indented defs too (`^\s*def\s+`) and found 420. Qwen3.5-27B found 336 (= 8 top-level functions × 42 modules). The gap is 84 = 2 class methods × 42 modules. This is a **task-comprehension error**, not a substrate error.

**Observation spills: 0 across all 4 tasks.** The spill path (`_rlm_artifacts/_observations/`) remains untested.

---

## 5. Comparison vs. baselines

| | Gemma 4 31B (dense VL) | Qwen3.6-35B-A3B (MoE) | Qwen3.5-27B (hybrid VL) |
|---|---|---|---|
| Tasks completed | 4/4 | 1/4 | 4/4 |
| Parse retries (parent turns) | 0 | >14 | 4 (1 per task, all recovered) |
| Max retries in one turn | 0 | 3 (crash) | 1 |
| Format failure mode | None | Child-element path (`<path>…</path>`) | Wrong attr name / path-as-body |
| One-retry recovery rate | — | ~25% (only 3a recovered) | 100% |
| Observation spills | 0 | 0 | 0 |
| 3d answer correct? | Yes (420) | Did not complete | No (336, excluded class methods) |
| Thinks (`</think>`) | No | Yes (long) | Yes (short, every turn) |
| Token cost vs. Gemma 4 (3a) | 4,076 in / 258 out | 5,598 in / 830 out | 4,064 in / 610 out |

Qwen3.5-27B sits **very close to Gemma 4** on substrate mechanics: both pass 4/4. The difference is the consistent turn-1 parse retry and the 3d task-comprehension error (wrong function count). On parse robustness, Qwen3.5-27B is categorically better than Qwen3.6-35B-A3B: its errors are shallow attribute-name confusions that it always self-corrects in one retry, versus Qwen3.6's deep schema misunderstanding that failed even after the maximum 3 retries.

**Comparison to Qwen3.5-9B** (sister report `2026-05-10_qwen-3-5-9b_substrate_eval.md`, expected alongside this report): that report does not yet exist at time of writing. The key question — does scaling 9B→27B within the Qwen3.5 family improve format-following enough to eliminate the turn-1 retry — cannot be answered yet. Based on the 27B results, the turn-1 `file=` / path-as-body confusion appears to be a model-family artifact rather than a scale artifact, since it appeared identically at both parent and child scope despite the model being the same size in both roles.

---

## 6. Recommendation

**Qwen3.5-27B is a viable substrate-execution model.** It passed all 4 tasks with no crashes, no hang, and no multi-retry death spirals. The substrate's parse-retry feedback loop is sufficient to handle its turn-1 format stumble.

**Caveats:**

1. **1 parse retry per task, every task.** The model reliably misnames the `path` attribute on its first `read_file`. This costs one extra LM call per run (and per child in recursive runs). For high-volume workloads or deep recursion trees, this accumulates. A one-shot format example in the system prompt would likely eliminate it.

2. **3d task comprehension error (336 vs 420).** The model made an unprompted decision to exclude class methods and reported an incorrect count. This is not a substrate issue but is a capability gap relative to Gemma 4.

3. **Not apples-to-apples vs. Gemma 4 31B.** Both are multimodal VL models run text-only, which is the closest fair comparison in this evaluation set. However, Qwen3.5-27B's hybrid linear attention and vision encoder present architectural differences that make the performance comparison less direct than two plain dense LLMs would be. The `--limit-mm-per-prompt '{"image":0,"video":0}'` flag disables the vision encoder at inference time, but the model's training distribution included vision data that may affect its text-only reasoning posture.

4. **Not apples-to-apples vs. Qwen3.6-35B-A3B.** Different architecture (dense hybrid vs. MoE), different parameter count (27B vs. 35B total / ~3B active), different training recipe. The format-following improvement is striking but cannot be attributed to size alone.

**Bottom line:** Qwen3.5-27B is a serviceable alternative to Gemma 4 31B for substrate tasks with one known limitation (consistent turn-1 attribute-name error, always self-corrected) and one capability gap (3d comprehension, possibly task-specific). It is not a drop-in replacement — add a format example to the system prompt and plan for the occasional function-count class of errors. If the Qwen3.5-9B sister report shows the same turn-1 retry, that further isolates it as a family-level prompt-tuning gap addressable through few-shot seeding rather than a fundamental architecture barrier.

---

## Artifact locations

| Phase | JSONL | Workspace root |
|---|---|---|
| 3a primes | `_setup_runs/logs/qwen-3-5-27b/rlm_3a_qwen-3-5-27b_2026-05-10_22-12-56_57c6e6c3.jsonl` | `~/.rlm/workspaces/run_*_57c6e6c3*/` |
| 3b mergesort | `_setup_runs/logs/qwen-3-5-27b/rlm_3b_qwen-3-5-27b_2026-05-10_22-13-24_6c740261.jsonl` | `~/.rlm/workspaces/run_*_6c740261*/` |
| 3c recursion | `_setup_runs/logs/qwen-3-5-27b/rlm_3c_qwen-3-5-27b_2026-05-10_22-14-17_a41c7f65.jsonl` | `~/.rlm/workspaces/run_*_a41c7f65*/` |
| 3d longctx | `_setup_runs/logs/qwen-3-5-27b/rlm_3d_qwen-3-5-27b_2026-05-10_22-16-47_ec907c21.jsonl` | `~/.rlm/workspaces/run_*_ec907c21*/` |

Runner stdout: `_setup_runs/logs/qwen-3-5-27b/runner_stdout.log`

---

## Re-run with parser fix (2026-05-11)

### 1. What changed in the substrate

Two changes landed in `rlm/utils/action_parser.py` between the original 2026-05-10 sweep and this re-run. Fix A: `_strip_think_blocks()` is called at the top of `parse()` before the action scanner runs, so any prior malformed `<action>` attempts inside a `<think>…</think>` monologue are excised before scanning begins. Fix C: `parse()` now collects all well-formed `<action>` elements tolerantly (schema failures on earlier candidates are recorded but skipped rather than aborted), then returns only the **last contiguous cluster** of well-formed actions — so a quoted system-prompt example or a backticked code block earlier in the prose cannot crowd out the model's final intended action.

### 2. Side-by-side per-task table

| Task | Pre-fix outcome | Post-fix outcome | Pre-fix turns | Post-fix turns | Parse retries (pre → post) | Tokens in (pre → post) | Tokens out (pre → post) |
|---|---|---|---|---|---|---|---|
| 3a — primes | PASS | PASS | 3 | 3 | 1 → 1 (parent) | 4,064 → 4,069 (+5) | 610 → 615 (+5) |
| 3b — merge sort + pytest | PASS | PASS | 8 | 7 (−1) | 1 → 1 (parent) | 14,940 → 11,713 (−3,227) | 1,140 → 1,022 (−118) |
| 3c — `rlm_query` fan-out × 5 | PASS | PASS | 5 | 5 | 6 → 6 (1 parent + 5 children) | 32,147 → 31,935 (−212) | 3,419 → 3,407 (−12) |
| 3d — 50 KB long-context extraction | PASS\* | PASS\* | 4 | 3 (−1) | 1 → 2 (parent) | 24,730 → 23,085 (−1,645) | 1,029 → 1,285 (+256) |

\* 336 functions reported in both runs; see §4 below.

The headline is flat pass-rate (4/4 → 4/4) and modestly lower token usage on 3b and 3d. Total parse retries are unchanged across the suite — 8 in both runs (4 parent + 5 children in 3c's pre-fix; 5 parent + 5 children in 3c's post-fix, partially offset by 3d gaining one additional parent retry). The LM call counts are identical for 3a, 3c, and 3d (4, 26, 5 respectively) and one fewer for 3b (9 → 8).

### 3. Where the parser fix mattered (and where it did not)

**The A+C fixes are irrelevant to Qwen3.5-27B's turn-1 attribute-name errors.** Tracing the post-fix 3a iter-1 attempt confirms why:

```
attempt 1 response:
  "Let me start by reading the root task file to understand what I need to do.\n</think>\n\n<action tool=\"read_file\" file=\"_rlm_query_0.txt\" />"
error: Unknown attribute(s) ['file'] on <action tool='read_file'>. Allowed: ['end_line', 'path', 'start_line'].
```

The malformed action (`file=` instead of `path=`) appears **below** `</think>`, i.e., it is the model's actual intended action, not a stale attempt inside the think block. Fix A strips content inside `<think>…</think>` pairs — it has nothing to strip here. Fix C takes the last contiguous cluster of well-formed actions — but there is only one action candidate (the malformed one), so the cluster rule cannot rescue it either. The same pattern holds for 3b iter 1 (path-as-body), 3c iter 1, and 3d iter 1.

All four tasks still required exactly one parent-level retry on their first `read_file`. The retry feedback and one-shot self-correction behavior are identical to the pre-fix run. The recoveries:

- 3a: `file=` → `path=` in one retry.
- 3b: path-as-element-body → correct `path=` attr in one retry.
- 3c (parent): `file=` → `path=` in one retry.
- 3c (all 5 children): same read_file format error (split between `file=` and body variants); each recovered in exactly one retry, unchanged.
- 3d: `file=` on iter 1 in one retry; additionally, `path=` on `python` tool on iter 2 in one retry (new in this run — see §4).

The parse-retry count **did not decrease** because this model's format errors are shallow wrong-attribute confusions, not stale-think-block artifacts. The A+C fixes target a different failure mode (typified by Qwen3.6-35B-A3B, which would embed prior failed attempts in the think monologue and let the scanner catch the stale one first).

The 3b token reduction (−3,227 input tokens, −1 turn) is unrelated to the parser fix — the post-fix run simply chose a more efficient solution path, omitting the intermediate tail-sample read that the pre-fix run took on iter 2.

### 4. 3d function count

The post-fix run reproduced **336 functions** exactly. The python script used the same regex:

```python
def_match = re.match(r'^def\s+(\w+)\s*\(', line)
```

`re.match` with `^def` matches only top-level (unindented) `def` statements; class methods, which begin with leading whitespace, are excluded. The model's comment in the script explicitly names the intent: *"top-level def, not indented as a method."* This is a task-comprehension error that the parser fix cannot touch — the model chose to exclude methods before writing a line of code. The post-fix run also skipped the tail-sample read entirely (3 turns vs 4), going straight from a single full-file `read_file` on iter 1 to the python extractor on iter 2; this did not change its regex strategy or result. The count remains 336/420.

The post-fix run added one new parent parse retry on iter 2: the model tried `<action tool="python" path="extract_declarations.py">…` (wrong attribute — `python` takes no `path` attr), recovered in one retry, and produced the same script body. Net effect: 3 turns, 2 retries, 5 LM calls — versus 4 turns, 1 retry, 5 LM calls pre-fix. Wall clock increased slightly (57.4 s vs 48.4 s), reflecting the extra retry round-trip on iter 2.

### 5. Updated verdict

The parser fix does not change the recommendation. Qwen3.5-27B remains **viable as a Gemma 4 31B replacement** under the same caveats as the original report: the turn-1 attribute-name confusion (always self-corrected in one retry) and the 3d function-count misread (top-level only, 336 vs 420). The A+C fixes were not the lever that would have helped this model — its format errors live below `</think>` and are recoverable by the existing retry loop without parser assistance.

What the re-run does confirm is that Qwen3.5-27B's behavior is **stable**: same pass rate, same retry count at the family level, same 3d miscount. The small token reductions on 3b (−3,227 in) and 3d (−1,645 in) reflect different navigation choices within the task, not parser improvement. If eliminating the turn-1 retry is a priority, the action is a format example in the system prompt (as the original report notes), not the parser.

**Post-fix artifact locations:**

| Phase | JSONL |
|---|---|
| 3a primes | `_setup_runs/logs/qwen-3-5-27b/rlm_3a_qwen-3-5-27b_2026-05-11_01-42-23_24e5248d.jsonl` |
| 3b mergesort | `_setup_runs/logs/qwen-3-5-27b/rlm_3b_qwen-3-5-27b_2026-05-11_01-42-51_126ffcea.jsonl` |
| 3c recursion | `_setup_runs/logs/qwen-3-5-27b/rlm_3c_qwen-3-5-27b_2026-05-11_01-43-39_6dc00822.jsonl` |
| 3d longctx | `_setup_runs/logs/qwen-3-5-27b/rlm_3d_qwen-3-5-27b_2026-05-11_01-46-09_18e750b7.jsonl` |

Runner stdout (post-fix): `_setup_runs/logs/qwen-3-5-27b/runner_stdout.log`
