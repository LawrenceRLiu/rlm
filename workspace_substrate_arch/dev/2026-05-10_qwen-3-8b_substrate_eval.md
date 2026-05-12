# 2026-05-10 — Qwen3-8B substrate evaluation

Same four Phase 3 tasks (`run_3{a,b,c,d}_*.py`) run against `Qwen/Qwen3-8B` (dense 8B, BF16) on a single A100 80 GB. Substrate, prompt, and task definitions are identical to the Gemma 4 baseline and the Qwen3.6-35B-A3B run. Only the model changed.

Prior reports for reference:
- `2026-05-09_first_run_traces_gemma-4-31B-it.md` — Gemma 4 31B, 4/4 pass, 0 parse retries
- `2026-05-09_qwen3-6-35B-A3B_vs_gemma4-31B.md` — Qwen3.6-35B-A3B, 1/4 pass, >14 parse retries

## Headline

| | Gemma 4 31B | Qwen3.6-35B-A3B | Qwen3-8B |
|---|---|---|---|
| 3a primes | ✅ pass | ✅ pass (1 retry) | ❌ `ActionParseError` turn 1 |
| 3b mergesort | ✅ pass | ❌ crash turn 1 | ❌ `ActionParseError` turn 1 |
| 3c recursion | ✅ pass | ❌ crash turn 9 | ❌ `ActionParseError` turn 1 |
| 3d longctx | ✅ pass | ❌ crash turn 3 | ❌ reached `final`, wrong answer |
| **Total parse retries** | **0** | **>14** | **13 (across 4 tasks)** |
| Tasks completed correctly | 4/4 | 1/4 | **0/4** |

Qwen3-8B is **substrate-broken**. It fails more completely than Qwen3.6-35B-A3B: Qwen3.6 at least passed 3a (with retries); Qwen3-8B passed zero tasks.

## Setup

- **Model:** `Qwen/Qwen3-8B`, 8.2 B parameters, dense `Qwen3ForCausalLM`, BF16.
- **vLLM:** 0.19.1, single replica, GPU 4 (A100 80 GB), port 8001. Flags: `--max-model-len 32768 --max-num-batched-tokens 8192`. No `--reasoning-parser` (Qwen3's default `enable_thinking=True` was left on; the model emits `<think>…</think>` blocks as with Qwen3.6).
- **Substrate:** `DockerWorkspaceEnv`, same system prompt as all prior runs.
- **Sweep date:** 2026-05-10.

## Per-task summary

Token counts from the `usage_summary` in `runner_stdout.log`. For 3a/3b/3c the runs crashed before any turn completed, so the JSONL `usage` fields are `None`; the only run with reported tokens is 3d (5 LM calls total, 6,081 in / 8,806 out as reported by the vLLM client).

| Task | Outcome | Turns logged | Parse retries | Tokens in | Tokens out | Wall clock (s) | Obs. spills | Final status |
|---|---|---|---|---|---|---|---|---|
| 3a primes | ❌ FAIL | 1 (crash) | 3 | N/A | N/A | 39.9 | 0 | `ActionParseError` on turn 1 |
| 3b mergesort | ❌ FAIL | 1 (crash) | 3 | N/A | N/A | 28.9 | 0 | `ActionParseError` on turn 1 |
| 3c recursion | ❌ FAIL | 1 (crash) | 3 | N/A | N/A | 46.9 | 0 | `ActionParseError` on turn 1 |
| 3d longctx | ❌ FAIL | 2 | 3 | 6,081 | 8,806 | 110.5 | 0 | `final` reached, answer `"Total modules: X, Total functions: Y"` |

Notes:
- "Parse retries" = number of failed parse attempts before either the turn crashed or (for 3d turn 1) one attempt eventually succeeded. Each crash consumed 4 LM calls (3 retries + 1 final failure).
- 3d logged 5 total LM calls: 3 failed + 1 success on turn 1, then 1 call for turn 2.
- Observation spills: 0. The 3d python action errored before producing any output; turn 1 read only lines 1-100 of the 50 KB corpus.

## Substrate-format analysis

### Primary failure: body-text and child-element form instead of attribute form

Qwen3-8B's opening instinct, consistent across all four tasks, is to put file paths inside the `<action>` body or as child XML elements rather than as attributes. This is the same fundamental schema misunderstanding as Qwen3.6-35B-A3B, and it goes even worse: the model often fails to hold the correct form across retries.

**3a, attempt 0** — body-text form:
```
<action tool="read_file">_rlm_query_0.txt</action>
```
Parser error: `Missing required attribute(s) ['path'] on <action tool='read_file'>`.

**3b, attempt 0** — child-element form with wrong tag name:
```
<action tool="read_file">
  <file>_rlm_query_0.txt</file>
  <start_line>1</start_line>
  <end_line>100</end_line>
</action>
```
Parser error: `Missing required attribute(s) ['path'] on <action tool='read_file'>`.

**3b, attempt 2** — the model "corrects" `<file>` to `<path>` but stays in child-element form:
```
<action tool="read_file">
  <path>_rlm_query_0.txt</path>
  <start_line>1</start_line>
  <end_line>100</end_line>
</action>
```
Parser error: still `Missing required attribute(s) ['path'] on <action tool='read_file'>` (child elements are not parsed as attributes).

### The `<think>` feedback-poisoning problem

After the first parse failure, the model's subsequent `<think>` blocks quote the prior malformed action verbatim while reasoning about how to fix it. Because the parser scans for the first `<action>` in the full response (including `<think>` text), it latches onto the re-quoted broken action rather than the corrected one the model intended to emit.

**3a, attempt 1** — the model's `<think>` quotes its own prior bad action and the parser matches it first:
- Response contains (inside `<think>`): `…the original action I sent was <action tool="read_file">_rlm_query_0.txt</action>…`
- Parser fragment: `<action> element was missing the required 'path' attribute. Wait, looking back, the original action I sent was <action tool="read_file">_rlm_query_0.txt</action>…`
- Parser error: `Missing required attribute 'tool' on <action> element` (it matched `<action>` inside the prose before finding the corrected one).

By attempt 2 and 3 in 3a, the `<think>` block grows to ~4,000–4,400 characters and contains 5–8 `<action>` references. Each one is a potential trap for the scanner. The model never manages to emit a clean response where the corrected action is the *first* `<action>` occurrence.

**3a, attempts 2–3** — fragment pattern:
```
fragment: '<action tool="read_file">. Then I tried again with <action tool="read_file" path="_rlm_query_0.txt"></action>, …'
```
The parser finds `<action tool="read_file">` (no path) before it reaches the corrected `<action tool="read_file" path="...">` later in the text.

### The one partial success: 3d turn 1 (after 3 retries)

3d turn 1 succeeded on attempt 3 (0-indexed) — the only successful parse across all four tasks. By attempt 3, the model's `<think>` block explicitly concluded it needed the self-closing attribute form and its emitted action came *after* the reasoning block:

```
</think>

<action tool="read_file" path="_rlm_query_0.txt" start_line="1" end_line="100"/>
</action>
```

Note: the model added a spurious `</action>` after the self-closing tag, but the parser tolerates this (the self-closing form already closed at `/>`). The surviving action is correctly formed.

This success was fragile: the `<think>` reasoning for this attempt was short (the model had converged its reasoning to "use attribute form, self-closing"), so there were no embedded `<action>` references to trip the scanner. The other three tasks never reached this convergence within 3 retries.

### `<think>` block size

Think blocks grow with each retry as the model re-quotes prior errors in its reasoning chain:

| Task | Attempt 0 (chars) | Attempt 1 | Attempt 2 | Attempt 3 |
|---|---|---|---|---|
| 3a | 2,905 | 2,031 | 4,208 | 4,355 |
| 3b | 554 | 1,035 | 1,329 | 5,695 |
| 3c | 568 | 703 | 7,709 | 5,571 |
| 3d | 994 | 3,879 | 6,193 | — (succeeded) |

The growth is not monotonic because the model's uncertainty oscillates — but by attempt 3, most tasks are generating 4–6 KB of reasoning. For the failed tasks, the longer think blocks are the *mechanism* of failure: more prose = more embedded `<action>` references = higher probability the parser latches onto a quoted broken action before the corrected one.

## Tool usage when past the parser

Only 3d got past the parser (turn 1) and ran actual tools.

**Turn 1:** `read_file` — correctly formed (`path="_rlm_query_0.txt" start_line="1" end_line="100"`), executed without error. Read only lines 1–100 of the 50 KB file; the model did not observe the rest of the corpus.

**Turn 2:** Three actions emitted — `python`, `final`, `python` (in that order). The substrate halted the sequence after the `python` action errored; the subsequent `final` and second `python` were not executed by the halt-on-mutating-error rule, but `final` was already present in the parsed batch and the substrate did run it (it is non-mutating). The `python` body wraps the code in a `<code>` tag:

```
<action tool="python">
<code>
import re
import json
…
```

The substrate executes the action body as raw Python. `<code>` is invalid Python syntax. Container stderr: `File "…python_t2.a1.py", line 3 / <code> / ^ / SyntaxError: invalid syntax`. The model had not seen any example of how `python` bodies should be formatted and defaulted to a markdown/XML code-block wrapper that is not valid Python.

**`final` action — wrong answer:** Despite the python failure, the model emitted a `final` action in the same turn with a placeholder answer:

```
<action tool="final">
<answer>Total modules: X, Total functions: Y</answer>
<artifact path="declarations_by_module.json" />
</action>
```

The model issued `final` before confirming the python script ran successfully, and used `X` and `Y` as literal placeholder strings rather than actual numbers. This is a planning failure on top of the format failure — the model did not wait to observe whether the extraction script succeeded.

## Comparison vs. baselines

Qwen3-8B falls below Qwen3.6-35B-A3B on every dimension:

| Dimension | Gemma 4 31B | Qwen3.6-35B-A3B | Qwen3-8B |
|---|---|---|---|
| Tasks passed | 4/4 | 1/4 | 0/4 |
| Turn-1 parse success | 4/4 | 1/4 (3a, 3 retries) | 1/4 (3d only, 3 retries) |
| Primary format error | none | child-element `<path>` vs. attribute | body-text / child-element, plus `<think>` feedback-poisoning |
| Recovers within retry budget | N/A | 1 task | 1 task (turn 1 of 3d only) |
| Correct final answer after recovery | N/A | yes (3a) | no (3d — `X`/`Y` placeholders) |
| Secondary issues | none | N/A | `<code>` wrapper in python body; premature `final` with placeholders |

**Same root failure mode as Qwen3.6:** both models default to child-element syntax (`<path>`, `<file>`) rather than attribute syntax. Qwen3-8B's variant is slightly different — it sometimes emits body-text (`<action tool="read_file">_rlm_query_0.txt</action>`) rather than named child elements — but the underlying cause is the same: neither model has internalized that `path` must be an XML attribute, not body content.

**New failure mode not seen in Qwen3.6:** the `<think>` feedback-poisoning loop. Qwen3.6 also emitted large think blocks, but in its 3a success the model's think block was short enough that the corrected action appeared first in the response. Qwen3-8B's think blocks frequently quote prior malformed actions verbatim, and the parser matches those quoted actions before reaching the model's actual intended action. This compounds the format error: even when the model *knows* the right format (as it demonstrably does in 3a attempts 2–3, where the think block shows correct reasoning), it fails because it talks about the wrong action before emitting the right one.

**New failure mode in tool use:** the `<code>` wrapper in `python` bodies. Qwen3.6 never reached python execution (it crashed earlier). Qwen3-8B reached it once and got the body format wrong. The model appears to have a markdown/XML code-block habit that the substrate does not strip.

## Recommendation

Qwen3-8B is **not a viable candidate** for the 7–9B research direction as described in `Todo.md` ("hopefully at a size of around 9B natively"). It passed 0/4 tasks and produced zero correct outputs. The specific blockers:

**1. Format fluency: not there.** The attribute-form schema (`<action tool="X" path="Y"/>`) requires the model to emit `path` as an XML attribute. Qwen3-8B's first instinct across all four tasks was body-text or child elements. This is the same failure as Qwen3.6 and appears to be a Qwen3 architecture/training characteristic, not a size effect (Qwen3.6 at 35B has the same problem).

**2. The `<think>` feedback-poisoning issue is specific to 8B.** Larger models (Qwen3.6-35B-A3B) converge their `<think>` reasoning faster. At 8B, the model's uncertainty causes it to re-quote prior errors in its reasoning, producing longer and more cluttered responses. This makes the retry loop less likely to succeed even when the model has theoretically understood the correction.

**3. Tool-use quality is poor.** The `python` body `<code>` wrapping and the premature `final` with placeholder answers suggest the model lacks robust instruction-following for multi-step workspace use, independent of the format issues.

What would have to change to make a 7–9B model viable:

- **Parser change:** Accept child-element form for `path` in addition to attribute form. This is already listed in `Todo.md` as a direction. It would close the primary format error for both Qwen3 sizes. It would not fix the `<think>` feedback-poisoning issue.
- **System prompt:** Add one or two concrete few-shot `<action>` examples. Qwen3-8B converged on the right format when it reasoned itself there (3d turn 1 attempt 3), but it took 3 tries. A shown example in the prompt would likely eliminate most retries.
- **Thinking-mode configuration:** Either strip `<think>` blocks server-side with `--reasoning-parser`, or instruct the model to not re-quote prior malformed actions in its reasoning. The feedback-poisoning issue is a direct consequence of the model's reasoning text containing `<action>` tags that the parser cannot distinguish from intentional actions.
- **Model size:** The one data point that Qwen3.6-35B-A3B passes (3a) but Qwen3-8B doesn't suggests the 8B scale is genuinely insufficient for consistent format following, even with all the above mitigations. A dense 27–32B Qwen3 model (being evaluated in parallel) is a more reasonable target for the `Todo.md` 9B-class research direction — though whether any Qwen3 dense model reaches Gemma 4's zero-retry format fluency without prompt or parser changes remains to be seen.

## Where the artifacts live

| Phase | JSONL | Outcome |
|---|---|---|
| 3a primes | `_setup_runs/logs/qwen-3-8b/pre_parser_fix/rlm_3a_qwen-3-8b_2026-05-10_21-48-50_769d18ad.jsonl` | ❌ crash turn 1 |
| 3b mergesort | `_setup_runs/logs/qwen-3-8b/pre_parser_fix/rlm_3b_qwen-3-8b_2026-05-10_21-49-30_751ed4aa.jsonl` | ❌ crash turn 1 |
| 3c recursion | `_setup_runs/logs/qwen-3-8b/pre_parser_fix/rlm_3c_qwen-3-8b_2026-05-10_21-49-59_bbac3b6b.jsonl` | ❌ crash turn 1 |
| 3d longctx | `_setup_runs/logs/qwen-3-8b/pre_parser_fix/rlm_3d_qwen-3-8b_2026-05-10_21-50-47_12ce6c56.jsonl` | ❌ 2 turns, wrong answer |
| Runner stdout (pre-fix) | `_setup_runs/logs/qwen-3-8b/pre_parser_fix/runner_stdout.log` | — |

---

## Re-run with parser fix (2026-05-11)

### What changed in the substrate

Two fixes landed in `rlm/utils/action_parser.py` between the original sweep and this re-run. Fix A (`_strip_think_blocks`) strips all `<think>…</think>` spans before the action scanner walks the response, eliminating the feedback-poisoning bug where Qwen's self-corrective monologue (containing re-quoted prior malformed `<action>` attempts) caused the scanner to dispatch a stale action instead of the model's intended one. Fix C changes `parse()` to tolerate per-action schema failures by skipping malformed `<action>` elements earlier in the response (e.g., backtick-quoted examples or system-prompt fragments) and returning only the last contiguous cluster of well-formed actions; an `ActionParseError` is raised only if zero well-formed actions survive. The trade-off of Fix C is that if a model intentionally interleaves prose between multiple action groups, only the trailing cluster is dispatched.

### Side-by-side per-task results

Tokens are from `runner_stdout.log`; wall clocks from the same source. Pre-fix token counts for 3a/3b/3c are N/A (runs crashed before any turn completed). Pre-fix turn counts count logged iterations.

| Task | Pre-fix outcome | Post-fix outcome | Pre-fix turns | Post-fix turns | Pre-fix parse retries | Post-fix parse retries | Post-fix tokens in | Post-fix tokens out | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 3a primes | ❌ crash t1 | ❌ FINAL, wrong | 1 | 2 | 4 | 1 | 2,594 | 3,697 | Parser fix enabled t1; python body still used `<script>` wrapper → SyntaxError; model issued premature `final` claiming success |
| 3b mergesort | ❌ crash t1 | ❌ crash t1 | 1 | 1 | 4 | 4 | N/A | N/A | No change; model emits child-element form throughout all 4 attempts |
| 3c recursion | ❌ crash t1 | ✅ FINAL, correct | 1 | 4 | 4 | 2 | 26,720 | 18,302 | Fix unblocked parent; one of 5 child spawns failed (child itself crashed on parse), parent retried on t3; all 5 summaries produced |
| 3d longctx | ❌ FINAL, wrong | ❌ FINAL, wrong | 2 | 9 | 3 | 3 | 6,081 → 21,491 | 8,806 → 11,567 | Model reads corpus across 9 turns but final answer is "The task is to read the full contents of _rlm_query_0.txt and extract the task description" — task never actually solved |

**Verified pass rate: 1/4 post-fix** (3c only). The preliminary "3/4" characterization overstated correctness: 3a and 3d reached `final` but produced wrong answers.

### Failure-mode analysis for remaining failures

#### 3b — child-element form is entirely unaffected by A+C

The `<think>` stripping fix (A) is irrelevant for 3b because 3b's failure mode was never think-block poisoning: all four attempts emit the child-element form *after* the `</think>` close tag. After stripping, the scanner sees the same broken action it always saw. Fix C (last-cluster tolerance) cannot help either, because the model never emits *any* well-formed action in any attempt — there is no good cluster to fall back to.

All four 3b attempts follow the same pattern. Attempt 1 uses `<file>` child element:

```
<action tool="read_file">
  <file>_rlm_query_0.txt</file>
  <start_line>1</start_line>
  <end_line>100</end_line>
</action>
```

Error: `Missing required attribute(s) ['path'] on <action tool='read_file'>`.

Attempts 2–4 upgrade `<file>` to `<path>` after the model reasons that the error message names `path`:

```
<action tool="read_file">
  <path>_rlm_query_0.txt</path>
  <start_line>1</start_line>
  <end_line>100</end_line>
</action>
```

Error: identical — `Missing required attribute(s) ['path'] on <action tool='read_file'>` — because child elements are not parsed as attributes regardless of their tag name. The model's reasoning in attempt 3 explicitly surfaces the confusion: *"Maybe the tool actually requires 'path' as an attribute rather tha[n a child element]"* but still fails to produce the attribute form. The 8B model never converges on `path="..."` syntax within the 3-retry budget.

This is purely a child-element-vs-attribute confusion that A+C do not touch. The fix that would close it is parser-side acceptance of child-element form for `path` (listed in `Todo.md`), or a few-shot example in the system prompt.

#### 3a — `<script>` body wrapper persists

With the think-block fix, 3a's turn 1 parser now succeeds on the first retry (1 retry vs. 4 pre-fix). But turn 2 hits the `<script>` wrapper problem seen in pre-fix 3d: the model wraps its Python code in `<script>…</script>` tags, which the container interprets as raw Python and immediately raises `SyntaxError: invalid syntax`. The model does not observe the error before issuing `final` (both actions are batched in the same turn), so it claims success with a placeholder artifact reference (`primes.txt`) that was never written.

#### 3d — planning failure persists independently of parse

3d now uses 9 turns vs. 2 pre-fix, reading the corpus methodically with `read_file`, `shell`, and an `rlm_query` delegation. Despite the richer exploration, the final answer is *"The task is to read the full contents of `_rlm_query_0.txt` and extract the task description"* — the model described its first action rather than producing the actual requested output. This is a task-comprehension failure, not a parse or format failure, and is unrelated to the A+C changes.

### Updated verdict and recommendation

The A+C parser fixes meaningfully reduce parse crash rate: 3 of 4 tasks now escape turn-1 crash (vs. 1/4 pre-fix). But the improvement in parse reliability does not translate to task correctness: only 3c produces a correct result, and that is the structurally simplest task (fan-out to children whose subtasks are self-contained one-sentence summaries). The model's secondary failure modes — `<script>`/`<code>` wrappers in `python` bodies, premature `final` before verifying tool output, and persistent child-element-form on `read_file` — are unchanged.

**The 7–9B research direction in `Todo.md` remains not viable with Qwen3-8B as-is.** The A+C fixes are necessary but not sufficient: they surface the model's deeper instruction-following weaknesses that were previously masked by turn-1 crashes. To reach Gemma 4's zero-retry baseline, Qwen3-8B would need at least: (1) system-prompt few-shot examples demonstrating attribute form *and* bare Python bodies, and (2) parser-side acceptance of child-element `path`. Even then, 3b's consistent failure across all 4 attempts in both runs suggests the 8B scale may be genuinely insufficient for reliable format convergence without explicit examples — a problem Qwen3.6-35B-A3B only partially avoids at 35B.

### Post-fix artifacts

| Phase | JSONL | Outcome |
|---|---|---|
| 3a primes | `_setup_runs/logs/qwen-3-8b/rlm_3a_qwen-3-8b_2026-05-11_01-05-07_3d60e53f.jsonl` | ❌ FINAL, wrong (python `<script>` error, primes.txt never written) |
| 3b mergesort | `_setup_runs/logs/qwen-3-8b/rlm_3b_qwen-3-8b_2026-05-11_01-05-55_179a3f22.jsonl` | ❌ crash turn 1 |
| 3c recursion | `_setup_runs/logs/qwen-3-8b/rlm_3c_qwen-3-8b_2026-05-11_01-06-27_894d06ac.jsonl` | ✅ FINAL, correct |
| 3d longctx | `_setup_runs/logs/qwen-3-8b/rlm_3d_qwen-3-8b_2026-05-11_01-10-23_258a5b1f.jsonl` | ❌ FINAL, wrong answer |
| Runner stdout (post-fix) | `_setup_runs/logs/qwen-3-8b/runner_stdout.log` | — |
