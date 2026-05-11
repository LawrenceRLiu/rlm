# CoT prose around actions — gemma-4-31b emits none

## Context

Surfaced while diagnosing the SWE-Bench smoke run on three Verified instances
(astropy-7166, flask-5014, sympy-20916, gemma-4-31b via vLLM). One of the
secondary questions: "are there any thinking traces I can read in the
trajectory?" Two findings.

## Finding 1 — `WorkspaceIteration.reasoning` is empty (as expected)

`reasoning` captures **backend reasoning channels** — Anthropic extended
thinking, OpenAI reasoning, Gemini thinking
(`rlm/core/types.py:132-134`). vLLM serving gemma-4-31b doesn't expose any
such channel, so the field is `None` for all 81 turns across the 3 SWE-Bench
trajectories. This is correct behavior, not a bug.

## Finding 2 — gemma emits zero prose around `<action>` tags (likely bug-ish)

Stripped `<action ...>...</action>` and self-closed `<action ... />` from
the raw `response` field across all 81 turns of the SWE-Bench run. **Result:
0/81 turns have any prose at all** — not even 30 chars of leftover
whitespace + words. The model emits only bare `<action>` blocks.

The system prompt at `rlm/utils/prompts.py:28-30` says:

> "You may write reasoning prose around them; the runtime extracts only the
> ``<action>`` blocks but preserves the surrounding prose so you can anchor
> your planning across turns."

"You may" is permissive. Gemma is taking the permissive option to the limit.

This is independently bad because:

1. **Trajectory inspection in the visualizer shows only *what* tool was
   called, never *why*.** For 30-turn rollouts on hard problems this makes
   post-mortem analysis substantially harder.
2. **CoT before action typically helps agentic performance.** Concrete
   instance from the same investigation: in `pallets__flask-5014`, the
   agent destructively overwrote a 24k-char source file on turn 27 based on
   a 30-line `sed` slice it had read on turn 26. With explicit reasoning,
   the agent might have surfaced and rejected its own bad plan before
   acting.
3. **Recovery from truncation already fails** — the agent never reads any
   of the 9 spill files generated across the 3 runs (separate prompt bug,
   line 48, already fixed: "Don't worry about it" → "Read the file in the
   path to get full output").

## Why not change the prompt immediately?

Two reasons not to bundle this with the conda-env scaffold fix:

1. **Model-dependent.** Gemma-4-31b may simply not be a "CoT-y" base model.
   Claude, Sonnet, and Qwen 3.6 (which already emits `<think>` blocks per
   Q3 of `2026-05-09_qwen3-6-35B-A3B_vs_gemma4-31B.md`) likely respond
   differently to the same prompt. A one-line prompt change affects all
   models on all benchmarks; deserves its own A/B.
2. **Cross-benchmark blast radius.** Terminal-Bench currently passes its
   3-task validation with the existing prompt. Changing prompt phrasing
   may regress that or — more subtly — change the average turn count and
   token usage. Worth measuring deliberately, not as a side-effect of a
   SWE-Bench scaffold fix.

## Investigation plan (when picked up)

1. **A/B prompt variants:**
   - V0 (current): `"You may write reasoning prose around them..."`
   - V1: `"Reason briefly before each action; explain what you're trying
     to learn or change. Surround the action blocks with this reasoning."`
   - V2: dedicated `<reason>...</reason>` tag the substrate strips before
     dispatch but logs separately, parallel to `<action>`.
2. **Models to compare:** gemma-4-31b, Sonnet 4.6, Qwen 3.6 (after the
   action-format-forgiveness work lands per Todo.md "Make the action
   format more forgiving").
3. **Benchmarks:** TB demo tasks (3 tasks, fast iteration) + the 3
   SWE-Bench Verified instances we already have predictions for.
4. **Metrics:** resolved rate, mean turn count, mean prose chars per turn,
   subjective trajectory readability (spot-check 5 runs).
5. **Hypothesis:** V1 will improve performance and readability on gemma
   without regressing Sonnet; V2 is cleaner architecturally but may
   underperform V1 because models are habituated to writing prose inline,
   not in dedicated tags.

## Related findings

The same trajectory analysis turned up a separate, already-fixed bug:
truncated observations had the system prompt literally telling the agent
"Don't worry about it" (`rlm/utils/prompts.py:48`). Across 9 truncation
events spanning 3 trajectories, zero spill-file reads. Line 48 inverted
(commit at time of writing) so future runs will, in principle, follow
through on truncations. The CoT-prose issue is the next layer of the same
class of problem: the prompt's permissive phrasing makes the agent skip
behavior the substrate is set up to support.

## Files referenced

- `rlm/utils/prompts.py:8-52` (system prompt template)
- `rlm/core/types.py:132-134, 254-257` (`reasoning_content` /
  `reasoning` field docs)
- `eval/swebench/results/*/trajectory/*.jsonl` (the 81 turns analyzed)
