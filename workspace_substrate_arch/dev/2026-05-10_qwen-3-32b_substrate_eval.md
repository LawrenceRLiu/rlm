# 2026-05-10 — Qwen3-32B substrate evaluation

Model: `Qwen/Qwen3-32B` (32.8 B params, dense Qwen3ForCausalLM, BF16, hidden_size 5120, 64 layers, 64 attention heads, 8 KV heads).

Ran the same four Phase 3 tasks (`run_3{a,b,c,d}_*.py`) that were used to baseline `google/gemma-4-31B-it` (4/4) and `Qwen/Qwen3.6-35B-A3B` (1/4). Substrate, prompts, and Docker image held constant; only the model changed.

## Setup

| Field | Value |
|---|---|
| HF model ID | `Qwen/Qwen3-32B` |
| Architecture | Dense Qwen3ForCausalLM, 32.8 B params, BF16 |
| Hidden size / layers / heads (attn / KV) | 5120 / 64 / 64 / 8 |
| vLLM version | 0.19.1 |
| Serving topology | Single replica, GPU 4 (A100 80 GB), port 8001 |
| `--max-model-len` | 32768 |
| `--max-num-batched-tokens` | 8192 |
| `--reasoning-parser` flag | Not set (Qwen3's default `enable_thinking=True` left on) |
| Model cache | `/data/nwei/rlm_substrate/models` (deleted after run) |
| Sweep date | 2026-05-10 |
| litellm alias | `qwen-3-32b` |

Unlike `Qwen3.6-35B-A3B` (hybrid Mamba-Transformer), Qwen3-32B is pure dense transformer — no `--max-num-seqs` cap was needed to avoid CUDA graph OOM.

---

## Per-task summary table

| Task | Outcome | Turns (logged iters) | LM calls | parse_retries | tokens_in | tokens_out | wall_clock_s | obs_spills | Final status |
|---|---|---|---|---|---|---|---|---|---|
| 3a — first 100 primes | FAIL | 2 | 5 | 4 | — | — | 130.5 s | 0 | `ActionParseError` after 3 retries on turn 2 |
| 3b — merge sort + pytest | PASS | 4 | 6 | 2 | 9,020 | 2,334 | 111.0 s | 0 | FINAL (5/5 tests pass) |
| 3c — `rlm_query` fan-out | FAIL | 1 | 4 | 4 | — | — | 54.9 s | 0 | `ActionParseError` after 3 retries on turn 1 |
| 3d — 50 KB long-context extraction | FAIL | 1 | 4 | 4 | — | — | 61.4 s | 0 | `ActionParseError` after 3 retries on turn 1 |

**Total: 1/4 pass.** Tokens for 3a/3c/3d are not emitted by the runner because crashes suppress the usage summary; the runner does print full stats for 3b (`input_tokens=9,020 / output_tokens=2,334`).

LM call counts are reconstructed from JSONL: each `parse_attempts` entry is one LM call, plus one call for a successful turn with 0 retry overhead. The 3a crash happened on turn 2 after a clean turn 1 (1 call), requiring 4 calls on turn 2 (3 retries exhausted) = 5 total. Identical arithmetic for 3c/3d (crash on turn 1: 4 calls).

---

## Substrate-format analysis

### 3b — the success

Turn 1 required 2 parse retries before Qwen3-32B converged on correct attribute syntax. The first two attempts both used child-element form inside a markdown `` ```xml `` fence:

**3b turn 1, attempt 0** — child elements, parser error `Missing required attribute(s) ['path'] on <action tool='read_file'>`:
```
<action tool="read_file">
  <path>_rlm_query_0.txt</path>
  <start_line>0</start_line>
  <end_line>1000</end_line>
</action>
```

**3b turn 1, attempt 1** — same child-element pattern re-emitted (model's `<think>` block says "I did include the path" — it doesn't realize child elements ≠ attributes), same error:
```
<action tool="read_file">
  <path>_rlm_query_0.txt</path>
  <start_line>0</start_line>
  <end_line>1000</end_line>
</action>
```

**3b turn 1, successful response** — model finally reasoned its way to attribute syntax in the `<think>` block ("maybe the tool expects path as an attribute instead of a nested element"), then emitted:
```
<action tool="read_file" path="_rlm_query_0.txt" start_line="0" end_line="1000"/>
<action tool="list_directory" path="."/>
```
These were wrapped in a `` ```xml `` code fence, but the parser's tag-pair scanner found the raw `<action ...>` tags regardless. Critically, the model's `<think>` block had also mentioned these tags inline as prose examples — so the parser extracted 5 actions total (the 2 inline examples + 2 in the code fence + 1 duplicate from prose). All 5 were valid attribute-syntax actions pointing to the same two operations, so no harm done.

From turn 2 onward, Qwen3-32B held the attribute format cleanly across all remaining turns (`write_file`, `shell`, `final`) with zero parse retries.

### 3a — crash after 1 valid turn

Turn 1 succeeded immediately (0 retries) with a `list_directory` action using child-element form. The parser accepted it because `list_directory` with a `<path>` child element happens to satisfy whatever attribute fallback exists, or the child element was silently ignored (the tool likely defaulted to listing `.`). Turn 2 failed with 4 attempts, all exhausted. The breakdown:

- **Attempt 0** — child elements, error `Missing required attribute(s) ['path'] on <action tool='read_file'>`:
```
<action tool="read_file">
<path>_rlm_query_0.txt</path>
<start_line>1</start_line>
<end_line>0</end_line>
</action>
```

- **Attempt 1** — model correctly diagnosed "path should be an attribute" and switched to `<action tool="read_file" path="_rlm_query_0.txt" start_line="1" end_line="0" />`, but the error returned was `Missing required attribute 'tool' on <action> element`. This is puzzling — the action clearly has `tool="read_file"`. The most likely explanation is that vLLM's continuation of the `<think>` block (6,568 chars, the longest in the run) included a self-closing variant the parser mishandled, or the `` /> `` triggered the tag-pair scanner to interpret a following context tag as the `<action>` element. Whatever the cause, the parser rejected it.

- **Attempts 2 and 3** — model oscillated. Attempt 2 (6,664 chars response, 6,568-char think block): went back to child-element form concluding self-closing tags can't carry body-parameters. Attempt 3: went back to attribute-style `<action tool="read_file" path="_rlm_query_0.txt" .../>` again and received the same `Missing required attribute 'tool'` rejection, which exhausted the retry budget.

The error `Missing required attribute 'tool' on <action>` when the response plainly shows `tool="..."` is a parser edge case worth investigating — possibly the parser is finding a second stray `<action>` tag elsewhere in the response first. After a 6,568-char `<think>` block containing discussions of XML structure, there may be bare `<action>` substrings that the scanner picks up before the real action.

### 3c — crash on turn 1 immediately

All 4 attempts used the same pattern:

- Attempts 0, 2, 3 — child element form:
```
<action tool="read_file">
  <path>_rlm_query_0.txt</path>
  <start_line>0</start_line>
  <end_line>100</end_line>
</action>
```

- Attempt 1 — correctly switched to attribute syntax `<action tool="read_file" path="_rlm_query_0.txt" start_line="0" end_line="100" />`, but received `Missing required attribute 'tool' on <action> element`. Same unexplained rejection as 3a attempt 1.

So 3c alternated between two failure modes — never landing on an attempt where the correct attribute syntax got past the parser. Crashed at attempt 3.

### 3d — crash on turn 1 immediately, different final error

3d's terminal error is `Missing required attribute(s) ['path'] on <action tool='read_file'>` (not `Missing required attribute 'tool'`), indicating attempt 3 went back to the child-element form:

- Attempt 0 — child elements.
- Attempt 1 — child elements again (model thinks it included path and is confused by the error).
- Attempt 2 — correctly switched to `<action tool="read_file" path="_rlm_query_0.txt" start_line="0" end_line="0" />`, got `Missing required attribute 'tool'` (same unexplained rejection).
- Attempt 3 — creative hypothesis: put attributes on the `<path>` *child element*: `<action tool="read_file"><path path="_rlm_query_0.txt" start_line="0" end_line="0" /></action>`. This is neither child-element nor attribute form — a novel hybrid that still fails the path-attribute check.

The 3d oscillation shows the model genuinely does not have a stable internal schema for the correct format. Each retry is a new guess, not a monotonic improvement.

### `<think>` block usage

`<think>` blocks appear in every response (the reasoning-parser flag was not set, so vLLM does not strip them). Sizes:

| Task / turn | Think block chars |
|---|---|
| 3a turn 1 (success) | 858 |
| 3a turn 2 attempt 0 | 676 |
| 3a turn 2 attempt 1 | 1,352 |
| 3a turn 2 attempt 2 | **6,568** |
| 3a turn 2 attempt 3 | 2,417 |
| 3b turn 1 attempt 0 | 1,586 |
| 3b turn 1 attempt 1 | 1,156 |
| 3b turn 1 (success) | 1,590 |
| 3b turn 2 | 795 |
| 3b turn 3 | 1,614 |
| 3b turn 4 | 869 |
| 3c turn 1 (all attempts) | 748 – 1,561 |
| 3d turn 1 (all attempts) | 669 – 2,427 |

Think blocks are present on every response, including every retry. They consume 700–6,600 chars of output per call. The 6,568-char think block on 3a attempt 2 is an outlier — the model spiraled into a lengthy back-and-forth with itself about child elements vs. attributes, including quoting and re-quoting the XML structure, which likely caused the parser to encounter a stray `<action>` substring that produced the spurious `Missing required attribute 'tool'` error on the next retry.

---

## Tool usage in 3b

| Turn | Tools used | Errors | Note |
|---|---|---|---|
| 1 | `read_file` × 3, `list_directory` × 2 | none | 5 actions extracted from what was intended as 2 (parser found inline prose examples + code fence = duplicates); all read the same file |
| 2 | `write_file` (sort.py), `write_file` (test_sort.py), `shell` | `shell` exit 127 | `pytest` not in workspace image — same bug as Gemma's 3b turn 2 (B1) |
| 3 | `shell` (pip install pytest), `shell` (pytest run) | none | same recovery path as Gemma: installs pytest in-container, then runs tests |
| 4 | `final` | none | 5/5 tests reported passing |

The sort.py implementation is correct recursive merge sort. test_sort.py covers all 5 required cases (empty, single, already sorted, reverse, duplicates). Recovery from the pytest-not-found error (exit 127) matched Gemma's strategy exactly: install pytest, then re-run.

No observation spills across any turn. No `rlm_query` recursion attempted in 3b (single-depth task).

---

## Comparison vs. baselines

### Pass rate summary

| Model | 3a primes | 3b mergesort | 3c recursion | 3d longctx | Pass rate |
|---|---|---|---|---|---|
| Gemma 4 31B (BF16, dense) | ✅ | ✅ | ✅ | ✅ | **4/4** |
| Qwen3.6-35B-A3B (BF16, MoE) | ✅ | ❌ | ❌ | ❌ | **1/4** |
| Qwen3-32B (BF16, dense) | ❌ | ✅ | ❌ | ❌ | **1/4** |

### Same overall rate, different failure fingerprint

Qwen3.6 failed 3b/3c/3d; Qwen3-32B fails 3a/3c/3d — different problem passes, same count. But the failure mechanics differ in a meaningful way:

**Qwen3.6-35B-A3B:**  
Default instinct was child-element form. Even with parser feedback, it sometimes dropped the `tool` attribute entirely (attempt 1 on 3a) — showing it couldn't maintain even the basic action schema across retries. Once it landed on attribute syntax (3a attempt 3), it stayed there for that turn.

**Qwen3-32B:**  
Also defaults to child-element form. But there is a qualitatively different failure mode: attempts that correctly use attribute syntax (`<action tool="read_file" path="..." .../>`) are rejected by the parser with `Missing required attribute 'tool' on <action> element`. This rejection is spurious given the response text — the attribute is present. The likely cause is the parser finding a stray bare `<action>` substring (inside a large `<think>` block that discusses XML) before the real action tag. This means Qwen3-32B *can* produce the correct format on retry, but the parser mis-fires on it in 3 out of 4 tasks. In 3b it got lucky: the attempt that used attribute syntax also placed the actions inside a `` ```xml `` fence and in inline prose, giving the parser multiple valid `<action tool=.../>` tags to find even if one stray one appeared earlier.

The practical implication: Qwen3-32B is *closer* to substrate-compatible than Qwen3.6, but is undermined by a parser interaction with large `<think>` blocks.

### Think block overhead vs. Qwen3.6

Both models emit `<think>` blocks. Qwen3-32B's blocks are somewhat smaller (median ~1,000–1,600 chars vs. Qwen3.6's runs), but still add substantial overhead per retry call. On the 3b success Qwen3-32B used 9,020 input / 2,334 output tokens for 4 turns — a reasonable count for the task, though Gemma 4 handled the same task in 5 turns (8 LM calls, Gemma's extra exploration of `pip install pytest` path was similar).

### 32B vs. sister models

At time of writing the Qwen3-8B sister report (`2026-05-10_qwen-3-8b_substrate_eval.md`) may exist for cross-size comparison. The expectation is that a larger dense model should be at least as format-following as a smaller one. If Qwen3-8B also passes 3b and fails 3a/3c/3d, it suggests the failure is a Qwen3-generation-wide schema confusion, not a size effect. If Qwen3-8B passes fewer, it confirms scale modestly helps. The 32B model's biggest failure was 3a (where it cleared turn 1 but crashed on turn 2 after a long think-spiral) — a smaller model with less tendency toward verbose think blocks might actually have fared better on 3a.

---

## Recommendation

**Qwen3-32B is not a viable Gemma 4 31B replacement at current substrate configuration.**

The hard blockers:

1. **Default format is child-element XML**, not the required attribute XML. This costs 2–3 parse retries on every first turn. At `max_action_parse_retries=3`, tasks with even mild multi-turn complexity exhaust the retry budget on the first action.

2. **Parser interaction with `<think>` blocks causes spurious `'tool' attribute missing` rejections.** The correct attribute-style action (`<action tool="read_file" path="..."/>`) appears in the model's output but the parser finds a stray `<action>` fragment inside the verbose think block first. This makes the substrate fail on valid responses — it's not purely a model quality issue.

3. **1/4 pass rate is insufficient.** The one pass (3b) required 2 retries on turn 1 and benefited from a lucky parser coincidence (actions appeared multiple times in the response, giving the scanner multiple valid hits). It is not a robust success.

**What would make Qwen3-32B viable:**

- **Few-shot example in system prompt.** One concrete `<action tool="..." path="..."/>` example with the correct attribute syntax, placed *before* the think block can form, would likely eliminate the first-attempt child-element mistake. The 3b success showed that once the model reasoned its way to attribute syntax (guided by parser error feedback), it held it cleanly for the remaining turns.

- **Parser hardening against stray `<action>` in think blocks.** The parser should skip `<action>` substrings found inside `<think>...</think>` spans, or apply stricter tag-pair matching. This would prevent the spurious `'tool' attribute missing` rejections that blocked correct attempts in 3a/3c/3d.

- **Stripping `<think>` at the vLLM level.** Setting `--reasoning-parser deepseek_r1` (Qwen3 is compatible) would strip think blocks from the text the substrate parser sees, eliminating both the stray-tag problem and the token overhead.

Without at least the parser hardening fix, the substrate interaction with Qwen3's verbose thinking mode will continue to false-reject valid attribute-syntax actions. With that fix plus a one-shot format example, Qwen3-32B could be a viable alternative — it does have the raw capability to produce correct actions (3b proves it), it just needs the format guided and the parser to stop tripping on its reasoning traces.

---

## Re-run with parser fix (2026-05-11)

### What changed in the substrate

Two changes landed in `rlm/utils/action_parser.py` between the original sweep and this re-run. Fix A: `_strip_think_blocks()` is called at the top of `parse()`, removing all `<think>…</think>` spans before the action scanner runs; an unterminated `<think>` drops everything after the open tag. Fix C: the parser now collects *all* structurally valid `<action>` elements, skipping per-action schema failures as soft errors, then returns only the **last contiguous cluster** separated by whitespace — so a backticked example or quoted system-prompt fragment earlier in the response cannot displace the model's intended action at the end. The trade-off is that if a model deliberately interleaves narrative prose between multiple intended actions, only the trailing cluster is dispatched.

---

### Side-by-side per-task table

| Task | Pre-fix outcome | Post-fix outcome | Pre-fix turns | Post-fix turns | Pre-fix LM calls | Post-fix LM calls | Post-fix tokens in | Post-fix tokens out | Post-fix wall clock | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| **3a** — first 100 primes | FAIL | **PASS** | 2 | 5 | 5 | 6 | 6,429 | 3,734 | 167.6 s | 1 retry on turn 2 (child-element form); correct attr syntax dispatched on retry |
| **3b** — merge sort + pytest | PASS | **PASS** | 4 | 5 | 6 | 6 | 8,150 | 1,985 | 92.2 s | 1 retry on turn 2; same pytest-127 recovery; slightly faster wall clock than pre-fix |
| **3c** — `rlm_query` fan-out | FAIL | **PASS** | 1 | 3 | 4 | 25 | 25,946 | 13,267 | 602.8 s | 5 sequential children; turn 2 alone took 511.8 s; all 5 completed |
| **3d** — 50 KB long-context | FAIL | **PASS** | 1 | 9 | 4 | 10 | 16,320 | 7,159 | 322.6 s | 1 retry on turn 2; observation spills on turns 2–3; python extractor on turn 6+8 |

**Headline: 1/4 → 4/4.** Every task that previously crashed on an `ActionParseError` now runs to `FINAL`.

---

### What the parser fix unlocked per task

**3a.** Pre-fix, the model's large `<think>` blocks (up to 6,568 chars) contained discussions of XML structure with bare `<action>` substrings. The scanner would pick up one of those stale fragments before reaching the real action, producing the spurious `Missing required attribute 'tool'` rejection that exhausted all 4 retries. Post-fix, the `<think>` block is stripped first; the model's clean attribute-syntax action appears alone after it:

```
<action tool="read_file" path="_rlm_query_0.txt" start_line="1" end_line="-1" />
```

That single tag dispatches without contest. Turn 1 still used child-element form (`<action tool="list_directory"><path>.</path></action>`), which produces a soft schema skip under Fix C — `list_directory` path is optional, so the element was tolerated and `list_directory` ran on `.` as a default. One retry on turn 2 (the child-element attempt was skipped; the model self-corrected to attribute syntax) and then zero retries for turns 3–5.

**3c.** Pre-fix, the model's very first response on turn 1 emitted a child-element `read_file`; the four retry attempts all hit either the child-element schema error or the spurious `tool` attribute rejection (same `<think>`-poisoning mechanism as 3a). Post-fix, the think strip clears the field; the one surviving retry on turn 1 produces a child-element action that is correctly soft-skipped, and the model self-corrects to attribute syntax on the actual dispatched response. Turn 2 then fires all 5 `rlm_query` calls cleanly.

**3d.** Same pattern: pre-fix, all 4 retry attempts on turn 1 failed — the model's `<think>` monologue about XML format poisoned the scanner into finding a bare `<action>` without a `tool` attribute before the real action. Post-fix, the strip eliminates that interference. One retry on turn 2 (child-element `read_file` skipped), then the model holds attribute syntax for all 7 subsequent turns including two `python` actions and a `final`.

**3b.** Already passed pre-fix; the fix changed nothing material. One retry on turn 2 (child-element `read_file` skipped by Fix C), identical pytest-127 recovery on turn 3, `final` on turn 5. Marginally fewer output tokens (1,985 vs 2,334) because the model no longer emits redundant duplicate actions from prose/fence/inline fragments that pre-fix's "return all" behavior was collecting.

---

### 3c child-call detail

Turn 2 dispatched 5 `rlm_query` actions in a single response. The children ran **sequentially** (sum of individual exec times is 470.8 s; turn 2 wall clock is 511.8 s, ~41 s overhead for spawn and teardown). All 5 children completed without error:

| Child | Task | Exec time | LM calls (cumul.) | Tokens in (cumul.) | Tokens out (cumul.) | Artifact exported |
|---|---|---|---|---|---|---|
| child_2_1 | Pythagorean theorem | 98.1 s | 7 | 6,905 | 3,672 | `pythagorean_example.txt` (path-mapped to parent) |
| child_2_2 | Photosynthesis | 220.9 s | 13 | 13,660 | 8,630 | none |
| child_2_3 | French Revolution | 62.4 s | 17 | 17,261 | 10,034 | none |
| child_2_4 | TCP reliability | 45.7 s | 20 | 19,959 | 11,062 | none |
| child_2_5 | Higgs boson | 43.7 s | 24 | 23,876 | 12,042 | none |

Child 2_1 explicitly exported an artifact (`_rlm_artifacts/pythagorean_example.txt`); the runtime path-mapped it to `_rlm_artifacts/children/child_2_1/_rlm_artifacts/pythagorean_example.txt` and included the mapping table in the parent's observation. Children 2–5 returned answers inline (no `<artifact>` tag), so the parent's observation for each reads `[Runtime Note: child returned no artifacts.]`. The parent on turn 3 wrote all five answers directly into `collated.txt` from the inline summaries in the child observations — no additional `read_file` round-trip needed. `final` fired on turn 3.

Child 2_2 (photosynthesis) was the heavyweight: 220.9 s, 6 LM calls, 6,755 incremental input tokens — nearly half the total turn-2 wall clock. The others were 44–99 s. Because spawning is sequential, child 2_2's latency directly gated the parent's turn 2 completion.

---

### Updated verdict and recommendation

**Qwen3-32B is substrate-fluent with the parser fix. The hard blockers from the original sweep are gone.**

The remaining friction is mild: the model's default instinct is still child-element XML, costing one parse retry on most first actions. Fix C degrades that gracefully rather than aborting — the child-element attempt is skipped, the model self-corrects to attribute syntax on retry, and subsequent turns are clean. On these four tasks, the retry cost was one extra LM call per task (except 3c turn 2 which was retry-free after the fix cleared the scanner).

**Is Qwen3-32B a viable Gemma 4 31B replacement?**

For correctness: yes. 4/4 tasks completed correctly, including the recursive fan-out and the 50 KB long-context extraction. The model's reasoning and implementation quality are comparable.

For cost and latency: substantially more expensive. The table below compares wall clocks and LM calls (Gemma numbers from the 2026-05-09 baseline report):

| Task | Gemma wall clock | Qwen3-32B wall clock | Gemma LM calls | Qwen3-32B LM calls |
|---|---|---|---|---|
| 3a | 15.4 s | 167.6 s | 4 | 6 |
| 3b | 84.4 s | 92.2 s | 8 | 6 |
| 3c | 90.9 s | 602.8 s | 27 | 25 |
| 3d | 48.4 s | 322.6 s | 4 | 10 |
| **Total** | **239.1 s** | **1,185.2 s** | **43** | **47** |

Wall clock is 5× longer overall, driven primarily by 3c (602.8 s vs 90.9 s) and 3d (322.6 s vs 48.4 s). The 3c gap is almost entirely sequential vs parallel child spawning: Gemma's 3c ran 5 children concurrently in 57.6 s; Qwen3-32B ran them sequentially in 470.8 s. If the substrate is configured for concurrent child spawning, the 3c gap would narrow to roughly Gemma's ballpark. The 3d gap (9 turns vs 3) reflects Qwen3-32B's more exploratory trajectory — multiple `read_file`, a failed `shell` on turn 4 (exit 2), a second `shell` recovery, two `python` passes — compared to Gemma's single one-shot `python` extraction. LM call counts are comparable (47 vs 43); the latency is driven by per-call wall time, not call count.

**Recommendation:** Qwen3-32B is a reasonable fallback when Gemma 4 31B is unavailable, with two caveats: (1) the one-retry-per-first-action overhead is real but tolerable at 1 call; and (2) tasks involving `rlm_query` fan-out will be significantly slower than Gemma unless concurrent child spawning is enabled. A one-shot format example in the system prompt showing `<action tool="..." path="..."/>` attribute syntax would likely eliminate the first-retry cost entirely and make the two models' per-turn behavior nearly identical.
