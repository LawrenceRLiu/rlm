# 2026-05-10 — Qwen3.5-9B substrate evaluation

Four Phase 3 tasks (`run_3{a,b,c,d}_*.py`) run against `Qwen/Qwen3.5-9B` under a single-replica vLLM setup. Substrate, prompt, and task definitions are identical to the Gemma 4 baseline and Qwen3.6-35B runs. Only the model changed.

Prior baselines:
- `gemma-4-31b` — 4/4 pass, 0 parse retries, clean format throughout. Report: `2026-05-09_first_run_traces_gemma-4-31B-it.md`.
- `Qwen/Qwen3.6-35B-A3B` — 1/4 pass, >14 parse retries, three runs crashed with `ActionParseError` due to child-element XML instead of attribute syntax. Report: `2026-05-09_qwen3-6-35B-A3B_vs_gemma4-31B.md`.

---

## 1. Setup

### Model identity — read this before assuming anything

`Qwen/Qwen3.5-9B` is **not** a plain 9B dense text model. Despite the name, it is the `Qwen3_5ForConditionalGeneration` architecture: a multimodal vision-language model with a vision encoder, a hybrid attention stack (32 layers, full attention every 4th layer, linear attention elsewhere), and 9B total parameters. It was served here **text-only** — the vision encoder was bypassed with `--limit-mm-per-prompt '{"image":0,"video":0}'`. Every result in this report is therefore for the text-only inference path of a model designed for vision input.

The hybrid linear attention is architecturally related to why `--max-num-seqs 64` was required for Qwen3.6 (Mamba/linear-attention OOM during CUDA graph capture at default `max_num_seqs=256`). The same flag was applied here preemptively and vLLM loaded without incident.

No `--reasoning-parser` flag was set; `<think>…</think>` content is included in the raw response text and is handled by the substrate's parser (which ignores non-`<action>` content).

### vLLM topology

Single replica on GPU 4 (A100 80 GB), port 8001, BF16.

```
--model Qwen/Qwen3.5-9B
--dtype bfloat16
--max-model-len 32768
--max-num-batched-tokens 8192
--max-num-seqs 64
--gpu-memory-utilization 0.90
--limit-mm-per-prompt '{"image":0,"video":0}'
```

No `--reasoning-parser`. Cached to `/data/nwei/rlm_substrate/models`, deleted after the sweep.

Sweep date: 2026-05-10.

### Thinking mode

All four responses contain `</think>` close-tags, confirming the model is in thinking mode. The substrate `reasoning` field in the JSONL is `null` for all turns (the server is not stripping think blocks into a separate field), so thinking content arrives embedded in `response`. The parser ignores it correctly.

Think-block lengths varied from ~86 to ~1273 characters per turn — short by Qwen3.6 standards but consistent across all 12 turns of 3a and all 12 turns of 3c.

---

## 2. Per-task summary table

| Task | Outcome | Turns logged | Parse retries (total) | Wall clock (s) | Observation spills | Final status |
|---|---|---|---|---|---|---|
| 3a — first 100 primes | **FAIL** | 12 | 9 | 72.4 | 0 | `ActionParseError` on turn 12 (turn 1 recovered; turns 3–4 recovered; turn 12 exhausted) |
| 3b — mergesort + pytest | **FAIL** | 1 | 4 | 15.9 | 0 | `ActionParseError` on turn 1 — never advanced past first turn |
| 3c — `rlm_query` fan-out × 5 | **PASS** | 12 | 4 | 204.1 | 0 | FINAL reached; `collated.txt` written with all 5 summaries |
| 3d — 50 KB long-context extraction | **FAIL** | 1 | 4 | 18.4 | 0 | `ActionParseError` on turn 1 — never advanced past first turn |

Token counts (3c only, from runner stdout): 126,072 input / 13,680 output across 75 LM calls (parent + all children).

**Overall: 1/4 tasks completed.** Numerically identical to Qwen3.6-35B-A3B's score, though the trajectory and failure mode differ materially (see §5).

---

## 3. Substrate-format analysis

### The primary failure: child-element `path` vs. attribute `path`

The model's canonical first-turn output across all four tasks is:

```
</think>

<action tool="read_file"><path>_rlm_query_0.txt</path></action>
```

The parser rejects this with:

```
Missing required attribute(s) ['path'] on <action tool='read_file'>.
```

This is the identical failure mode as Qwen3.6-35B-A3B. The model's instinct is XML-with-children; the substrate requires attributes. So far, same problem.

What happens next is where Qwen3.5-9B diverges — and where it also shows additional novel failure patterns.

### Novel failure mode A: quoting the system prompt back as a live action (3b turn 1, attempts 2–3)

After the first child-element rejection, the model's reasoning quotes the system prompt instruction literally:

```
"# How to act
Each turn, emit one or more ``<action tool="...">...</action>`` elements.
```

The parser is not a strict XML parser — it scans for `<action` anywhere in the response body, including inside quoted prose. The substring `<action tool="...">` inside the model's reasoning gets picked up as a real action element. Since `...` is not a recognized tool name, the parser raises:

```
Unknown tool '...' on <action>. Known tools: ['append_file', 'edit_file', ...]
```

This happened on 3b attempts 2 and 3. The model had actually produced the *correct* format (`<action tool="read_file" path="_rlm_query_0.txt">`) immediately below the quoted prose, but the parser halted on the earlier `tool="..."` match before reaching it. The run crashed with `Unknown tool '...'` on the final attempt.

This is a different failure than Qwen3.6's — not a schema misunderstanding per se, but a parser-interception of a system prompt quotation in reasoning text.

### Novel failure mode B: XML wrappers inside `python` action bodies (3a turns 2–7)

After successfully recovering the `read_file` format on turn 1, the model proceeded to a `python` action. The model wrapped its Python code in XML-like delimiters inside the action body:

Turn 2: `<python_code>\ndef is_prime(n):\n    ...`  
Turn 3: `<python>\ndef is_prime(n):\n    ...`  
Turn 4: `<![CDATA[\ndef is_prime(n):\n    ...`  
Turn 5: `<error>\ndef is_prime(n):\n    ...`  
Turn 6: `<code>\ndef is_prime(n):\n    ...`  
Turn 7: `<logo>\ndef is_prime(n):\n    ...`  

Each wrapper caused a Python `SyntaxError: invalid syntax` on the opening tag (the runtime writes the action body verbatim to a `.py` file and runs it). The model tried a new wrapper tag each time — `<python_code>`, `<python>`, `<![CDATA[`, `<error>`, `<code>`, `<logo>` — apparently selecting novel tags at random rather than re-reading the error. The progression is striking: it never tried removing the wrapper entirely, despite the error message being the same each time.

The model recovered on turn 8 by using `"""written by me` as a Python docstring prefix instead — which is syntactically valid and allowed the script to execute. This is not the correct pattern (the body should be raw Python with no wrapper), but it worked by accident.

### Where it got the format right

**3c turn 1 (read_file, after retry):** The final successful attempt used:
```
<action tool="read_file" path="_rlm_query_0.txt"><_rlm_query_0.txt content/></action>
```
A strange hybrid — `path` is correctly an attribute, but the body contains a self-closing `<_rlm_query_0.txt content/>` tag. The parser accepts this because `path` is present as an attribute (the body contents don't matter for `read_file`).

**3c turn 7 (rlm_query, no retries):** After 5 turns of repeated child-element failures in `rlm_query` calls, on turn 7 the parent emitted correctly-formed `rlm_query` actions with no retries:
```
<action tool="rlm_query"><child_rlm_task>Summarize: "The French Revolution..."</child_rlm_task></action>
```
`rlm_query` does not require a `path` attribute — its only parameter is the body. So there was no attribute-vs-child conflict. The model had always been emitting `rlm_query` bodies correctly; the failures were due to children (spawned from those `rlm_query` calls) failing the same `path`-attribute check on their first turns.

### Thinking block behavior

All 24 response turns across 3a and 3c contain exactly one `</think>` close-tag. Thinking content is short: 86–1273 chars per turn (median ~300). The substrate parser ignores it cleanly. No instances of truncated or unclosed `<think>` blocks were observed. The thinking content does not appear to help the model converge on the correct attribute format — six consecutive python-wrapper attempts in 3a turned over four different failure modes without any convergence signal.

---

## 4. Tool usage patterns

### 3a — 6+ consecutive python errors, then accidental recovery

| iter | tools | parse_retries | errors | note |
|---|---|---|---|---|
| 1 | `read_file` | 2 | 0 | child-element `path` failed twice; 3rd attempt used attribute (hybrid form) |
| 2 | `python` | 0 | 1 | `SyntaxError` — body started with `<python_code>` |
| 3 | `python` | 1 | 1 | `SyntaxError` — body started with `<python>` |
| 4 | `python` | 1 | 1 | `SyntaxError` — body started with `<![CDATA[` |
| 5 | `python` | 0 | 1 | `SyntaxError` — body started with `<error>` |
| 6 | `python` | 0 | 1 | `SyntaxError` — body started with `<code>` |
| 7 | `python` | 0 | 1 | `SyntaxError` — body started with `<logo>` |
| 8 | `python` | 0 | 0 | accidentally valid — body started with `"""written by me` (docstring) |
| 9 | `read_file` | 0 | 1 | tried to read `primes.txt` (not yet written) |
| 10 | `shell` | 0 | 1 | here-doc syntax error in shell |
| 11 | `shell` | 0 | 1 | same here-doc error |
| 12 | (crash) | 4 | — | tried `write_file` with `<path>` child element; all 4 retries failed |

The python recovery on turn 8 is accidental — the docstring `"""written by me` is syntactically valid Python, which bypassed the XML-wrapper error. The code itself was functionally correct. But the model never discovered the actual rule (bare code body, no wrapper). When it later needed `write_file` on turn 12, it regressed to child-element format and crashed.

After python finally succeeded, the model checked for `primes.txt` (not written — the python script output went to stdout rather than a file), then tried two shell here-documents that both failed due to missing EOF delimiter. By turn 12 it was trying to use `write_file` with `<path>` child element — re-applying the same wrong format it had been corrected for in turn 1. No learning carried over between tool types.

### 3b — never left turn 1

| iter | parse_retries | final error | note |
|---|---|---|---|
| 1 | 4 (→ crash) | `Unknown tool '...'` | attempts 0–1 used child-element path; attempts 2–3 quoted system prompt and got '...' tool parse error |

3b is instructive because the model actually *produced* the correct format (`<action tool="read_file" path="_rlm_query_0.txt">`) on attempt 2 — but the parser saw `<action tool="...">` in the model's quoted prose first, halted on the unknown tool, and never reached the valid action below it.

### 3c — recursion, eventually completed

| iter | tools | parse_retries | errors | note |
|---|---|---|---|---|
| 1 | `read_file` | 1 | 0 | one retry; used hybrid attribute+child-body form |
| 2 | 5× `rlm_query` | 0 | 4 | child 1 succeeded; children 2–5 failed child-element check internally; halt-on-mutating-failure fired |
| 3 | 4× `rlm_query` | 2 | 4 | same: 1 child failed, 3 skipped |
| 4 | 4× `rlm_query` | 0 | 3 | 1 child succeeded (summary_2.txt), then failure |
| 5 | 3× `rlm_query` | 0 | 3 | 1 child failed, 2 skipped |
| 6 | `read_file`, `read_file`, `list_directory` | 0 | 0 | parent pivoted to reading available summaries |
| 7 | 3× `rlm_query` | 0 | 0 | all 3 children succeeded (summaries 3, 4, 5) |
| 8 | 3× `list_directory` | 0 | 1 | `child_7_1` path didn't exist (only 2 and 3 did) |
| 9 | `read_file`, `read_file`, 2× `rlm_query` | 0 | 0 | 2 more children for summaries |
| 10 | 5× `read_file` | 0 | 0 | gathering artifact paths |
| 11 | 2× `write_file` | 1 | 0 | one write_file retry (path child-element), then success |
| 12 | `read_file`, `final` | 0 | 0 | FINAL |

The 3c completion required 12 turns and 75 LM calls (parent + children). Only 2 of the first 8 spawned children succeeded before the parent eventually found the right rhythm on turn 7. The parent's accumulated context of failure observations appeared to help it eventually produce tighter `rlm_query` bodies (no trailing newlines in the task strings by turn 7), but the children's success rate depended on the task body format, not the path attribute issue — `rlm_query` doesn't need a path attribute.

### 3d — never left turn 1

| iter | parse_retries | final error | note |
|---|---|---|---|
| 1 | 4 (→ crash) | `Missing required attribute(s) ['path']` | all 4 attempts used child-element `<path>` |

3d is the cleanest failure: no confusion about quoting, no wild new formats tried — just four attempts all using `<path>` child elements, all rejected, crash. No convergence.

---

## 5. Comparison vs. baselines

| Dimension | Gemma 4 31B | Qwen3.6-35B-A3B | Qwen3.5-9B |
|---|---|---|---|
| Tasks passed | 4/4 | 1/4 | 1/4 |
| Format failure mode | None | Child-element `<path>` | Child-element `<path>` + system-prompt-quoting + XML body wrappers |
| Parse retries (total logged) | 0 | >14 | ~17 |
| Converges with parser feedback? | N/A | Yes (3c turn 1) — 3 retries to attribute syntax | Partially — recovers `path` attribute but introduces new failure modes |
| Recursion (rlm_query) | ✅ 5/5 children on first turn | ❌ 0/N succeeded | ~Partial: children succeed or fail depending on task body, not path issue |
| `python` tool | ✅ bare body | Not tested | ❌ always wraps body in XML-like delimiter; recovered accidentally once |
| Shell | ✅ | ❌ with retries | ❌ here-doc syntax errors |
| Thinking mode | No | Yes (`<think>` blocks) | Yes (`<think>` blocks, shorter) |

Qwen3.5-9B lands at exactly the same pass count as Qwen3.6-35B-A3B (1/4), but it got there differently:

- **Qwen3.6** had one systematic problem: child-element `<path>`. With enough parse-retry budget it could recover (as it did in 3a and 3c turn 1). Its failure was a narrow schema gap.
- **Qwen3.5-9B** has the same child-element problem *plus* two additional problems: quoting the system prompt's own format example into parseable action blocks, and wrapping Python bodies in XML tags. Each new failure mode means a fresh failure vector — the model doesn't settle into a consistent wrong format, it varies its wrongness. That makes it harder to fix with a single format-injection patch.

The 3c pass is real but not Gemma-quality: Gemma completed 3c in 4 turns and 27 LM calls with all 5 children succeeding on the first batch call. Qwen3.5-9B needed 12 turns and 75 LM calls, with most children failing repeatedly. The final answer is correct in that `collated.txt` exists with 5 summaries, but the path to get there was expensive and fragile.

---

## 6. Recommendation

**Qwen3.5-9B is not a viable candidate for the 7–9B research direction in its current configuration.**

Two caveats before the verdict:

1. **The multimodal-VL architecture confounds the comparison.** `Qwen3.5-9B` (`Qwen3_5ForConditionalGeneration`) is not a pure-text 9B model. Its hybrid linear/full attention and vision encoder mean its text-only generation path is not representative of what a purpose-built 9B dense text model would produce. Any result here is specific to this architecture on this task, not to "a 9B model" in general.

2. **No `--reasoning-parser` was used.** Stripping think blocks server-side would reduce token counts but would not address the format failures.

The verdict regardless: the model's format following is worse than Qwen3.6-35B-A3B despite being 4× smaller, and it introduces additional failure modes that 3.6 did not show (system-prompt-quotation interception, XML body wrappers). The 1/4 completion rate came from 3c, which passed only because `rlm_query` body format doesn't require a `path` attribute — the model happened to avoid the specific attribute-vs-child conflict on the one tool it needed most. That's a lucky bypass, not format competence.

For the 7–9B research direction, this run does not provide useful signal about what a small model can do on this substrate. A clean 7–9B text model (e.g., a Llama-3.1-8B-Instruct or Qwen2.5-7B-Instruct fine-tuned for tool use) would be a better probe. If the goal is specifically to evaluate Qwen3 family models at smaller sizes, `Qwen/Qwen3-8B` (a pure-text dense model) is a better candidate than the multimodal-VL `Qwen3.5-9B`.

---

## Artifacts

| Task | JSONL | Outcome |
|---|---|---|
| 3a primes | `_setup_runs/logs/qwen-3-5-9b/rlm_3a_qwen-3-5-9b_2026-05-10_21-54-29_d3aaea24.jsonl` | FAIL — 12 turns, crash on turn 12 |
| 3b mergesort | `_setup_runs/logs/qwen-3-5-9b/rlm_3b_qwen-3-5-9b_2026-05-10_21-55-42_de1efc37.jsonl` | FAIL — 1 turn, crash on turn 1 |
| 3c recursion | `_setup_runs/logs/qwen-3-5-9b/rlm_3c_qwen-3-5-9b_2026-05-10_21-55-58_2ef8e190.jsonl` | PASS — 12 turns, 75 LM calls |
| 3d longctx | `_setup_runs/logs/qwen-3-5-9b/rlm_3d_qwen-3-5-9b_2026-05-10_21-59-23_5c4274f4.jsonl` | FAIL — 1 turn, crash on turn 1 |
| Runner stdout | `_setup_runs/logs/qwen-3-5-9b/runner_stdout.log` | All four task results |

```bash
conda activate RLM_substrate
python _setup_runs/trace.py \
  _setup_runs/logs/qwen-3-5-9b/rlm_3a_*.jsonl \
  _setup_runs/logs/qwen-3-5-9b/rlm_3b_*.jsonl \
  _setup_runs/logs/qwen-3-5-9b/rlm_3c_*.jsonl \
  _setup_runs/logs/qwen-3-5-9b/rlm_3d_*.jsonl
```

---

## Re-run with parser fix (2026-05-11)

### What changed in the substrate

Two changes landed in `rlm/utils/action_parser.py` between the two sweeps. Fix A strips `<think>…</think>` blocks before feeding text to the action scanner, eliminating the bug where Qwen's self-corrective monologue — which routinely contains prior malformed `<action>` attempts — would be scanned first and dispatch a stale action instead of the model's current intent. Fix C replaces the old "return all well-formed actions" walk with a "tolerantly skip malformed ones, return the last contiguous cluster" strategy: earlier invalid `<action>` elements (backticked examples, quoted system-prompt fragments) are skipped rather than aborting the whole turn, so only the model's final corrected cluster is dispatched. The trade-off is that if a model interleaves prose between multiple intended actions, only the trailing cluster is issued.

### Side-by-side per-task table

| Task | Pre-fix outcome | Post-fix outcome | Pre-fix turns | Post-fix turns | Token delta (input / output) | Notes |
|---|---|---|---|---|---|---|
| **3a** — first 100 primes | FAIL — `ActionParseError` turn 12 | FAIL — `ActionParseError` turn 6 | 12 | 6 | pre: not reported / post: not reported | Parser fix eliminated the system-prompt-quotation false-positive (no more `Unknown tool '...'`); model reached substantive execution but crashed on turn 6 trying `write_file` with child-element `<path>`. Python body-wrapper failures (fault B) persist unchanged. |
| **3b** — mergesort + pytest | FAIL — `ActionParseError` turn 1 (`Unknown tool '...'`) | **PASS** — FINAL turn 15 | 1 | 15 | pre: not reported / post: 54,776 in / 3,852 out (17 LM calls) | Biggest improvement. Pre-fix: parser intercepted `<action tool="...">` in quoted system-prompt prose and crashed before reaching the correct action. Post-fix: that quoted fragment is skipped; last cluster dispatched correctly. Model recovered and completed the task in 15 turns. |
| **3c** — `rlm_query` fan-out × 5 | PASS — FINAL turn 12, 75 LM calls | PASS — FINAL turn 5, 27 LM calls | 12 | 5 | pre: 126,072 in / 13,680 out / post: 31,384 in / 5,399 out (−75 % input, −61 % output) | Substantial efficiency gain. Pre-fix required 75 LM calls and 12 parent turns because child failures cascaded across many turns; post-fix completed in 27 calls. One child still failed (3 of 5 spawned on turn 2 errored or were skipped), resolved in turn 3. |
| **3d** — 50 KB long-context extraction | FAIL — `ActionParseError` turn 1 (child-element `<path>`) × 4 | FAIL — `ValueError` unlogged turn 6 | 1 | 5 + crash | pre: not reported / post: not reported | Pre-fix: crashed immediately on child-element `<path>` rejection. Post-fix: turn 1 recovered (model self-corrected to attribute format after 1 retry); then hit substantive execution failures — python body wrapped in `<body>` tag causing `SyntaxError` on turns 2–4, shell here-doc syntax error on turn 5. Unlogged turn 6 crashed the runner with `ValueError: Path must be workspace-relative: '/workspace/_rlm_artifacts/parse.py'` (model tried an absolute container path). |

**Overall pass rate: pre-fix 1/4 → post-fix 2/4.**

### Failure-mode analysis for remaining failures

**3a — write_file child-element regression on turn 6**

After the parser fix, 3a made real progress: the model cleared turn 1 (`read_file`) with 1 parse retry and then ran five consecutive python/shell tool attempts — all of which failed for substantive reasons (XML body wrappers, here-doc syntax). No parse errors occurred until turn 6, where the model tried to write a file using child-element path syntax, the same failure as the pre-fix era:

```
<action tool="write_file"><path>_rlm_artifacts/gen_primes.py><body>primes = []\n...
```

All four retries on turn 6 reproduced the same error:
```
Missing required attribute(s) ['path'] on <action tool='write_file'>.
```

The model's thinking on turn 6 says "Let me try a cleaner approach - first write the file without any special delimiters in the body" — it diagnosed the body-wrapper problem correctly but then produced yet another child-element `<path>`, having apparently not retained from turn 1 that attribute format is required. The child-element problem is not a parser-fixable issue: the parser correctly rejected it each time; the model simply does not carry format learning across tool types.

The python body-wrapper failure (fault B from the original report) also persisted unchanged post-fix. Turns 2–4 all used `<code>` or `<preset>` wrappers inside the `python` action body, yielding `SyntaxError: invalid syntax` on the opening tag in each case:

```
File "/workspace/_rlm_state/_tmp/python_t2.a1.py", line 2
    <code>
    ^
SyntaxError: invalid syntax
```

This is substantive, not parse-related.

**3d — substantive execution failure after format recovery**

Post-fix 3d represents the clearest separation between parse failures and substantive failures. Turn 1 recovered cleanly (model self-corrected child-element → attribute format after one retry). But turns 2–5 all failed due to python body wrappers (`<body>` tag producing `SyntaxError`) and a shell here-doc syntax error — identical to 3a's pattern. The runner then crashed on an unlogged turn 6 when the model tried to reference an absolute container path (`/workspace/_rlm_artifacts/parse.py`) in a tool call. This is a new failure mode not seen in 3a: the model appeared to conflate the container's filesystem layout with workspace-relative paths.

The 3c child failure (turn 2, one of five `rlm_query` spawns) is the last remaining parse-related failure post-fix:

```
ActionParseError: Action parse failed after 3 retries: Missing required attribute 'tool' on <action> element.
```

That child's first response omitted the `tool` attribute entirely — a different schema error from the child-element `<path>` issue, suggesting the model's format instability is not confined to a single attribute.

### Updated verdict

**The parser fix materially improves Qwen3.5-9B's substrate viability: 2/4 vs. 1/4, and 3c efficiency improved dramatically (75 → 27 LM calls, −75 % input tokens).** The fix directly unblocked 3b, which failed pre-fix for a pure parser reason (the `Unknown tool '...'` interception of a system-prompt quotation), and made 3c meaningfully cheaper by eliminating the cascading child-failure pattern driven by stale-action dispatch.

However, two categories of failure remain that the parser cannot address:

1. **Python/shell body-wrapper habit.** The model consistently wraps `python` and `shell` action bodies in XML-like delimiters (`<code>`, `<body>`, `<preset>`, etc.), causing runtime `SyntaxError` or `shell exit code 2`. This persisted in 3a (turns 2–4), 3d (turns 2–5), and would have appeared in 3b had the model not happened to write clean bodies from turn 2 onward. It is a generation habit, not a parser bug.

2. **Child-element `<path>` on write/read-file tools.** The model recovers to attribute format on turn 1 with parser feedback but then regresses to child-element format on later `write_file` calls (3a turn 6, would-be 3b turn 2+ if not for the fix unblocking things). Format learning does not persist across turns or tool types.

The multimodal-VL caveat from the original report carries over unchanged. If anything, these results make it more confounding: a purpose-built 7–9B text model might exhibit better format stability. The 2/4 result is a real improvement over the pre-fix 1/4, and demonstrates that the parser fixes are load-bearing for this model class. But the remaining failures are now clearly substantive — the model cannot reliably use `python`, `shell`, or `write_file` tools without format coaching that a fine-tuned or better-instruction-following model would not need. For production use, Qwen3.5-9B in its current text-only configuration is borderline: it can handle pure `rlm_query` fan-out tasks (3c, 3b) but fails on tool-use tasks requiring `python` or `write_file`.

### Post-fix artifacts

| Task | JSONL | Outcome |
|---|---|---|
| 3a primes | `_setup_runs/logs/qwen-3-5-9b/rlm_3a_qwen-3-5-9b_2026-05-11_01-14-09_888cf76d.jsonl` | FAIL — 6 turns, crash on turn 6 |
| 3b mergesort | `_setup_runs/logs/qwen-3-5-9b/rlm_3b_qwen-3-5-9b_2026-05-11_01-14-42_553cd79a.jsonl` | PASS — 15 turns, 17 LM calls |
| 3c recursion | `_setup_runs/logs/qwen-3-5-9b/rlm_3c_qwen-3-5-9b_2026-05-11_01-15-42_a62e0fe6.jsonl` | PASS — 5 turns, 27 LM calls |
| 3d longctx | `_setup_runs/logs/qwen-3-5-9b/rlm_3d_qwen-3-5-9b_2026-05-11_01-17-05_9b8debb3.jsonl` | FAIL — 5 logged turns + unlogged crash |
| Runner stdout | `_setup_runs/logs/qwen-3-5-9b/runner_stdout.log` | All four task results |

Pre-fix files moved to `_setup_runs/logs/qwen-3-5-9b/pre_parser_fix/`.
