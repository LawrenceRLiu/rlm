import textwrap
from typing import Any

from rlm.core.types import (
    QueryMetadata,
    WorkspaceIteration,
    WorkspaceObservation,
)

# System prompt for the REPL environment with explicit final answer checking
RLM_SYSTEM_PROMPT = textwrap.dedent(
    """You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a REPL environment that can recursively query sub-LLMs, which you are strongly encouraged to use as much as possible. You will be queried iteratively until you provide a final answer.

The REPL environment is initialized with:
1. A `context` variable that contains extremely important information about your query. You should check the content of the `context` variable to understand what you are working with. Make sure you look through it sufficiently as you answer your query.
2. A `llm_query(prompt, model=None)` function that makes a single LLM completion call (no REPL, no iteration). Fast and lightweight -- use this for simple extraction, summarization, or Q&A over a chunk of text. The sub-LLM can handle around 500K chars.
3. A `llm_query_batched(prompts, model=None)` function that runs multiple `llm_query` calls concurrently: returns `List[str]` in the same order as input prompts. Much faster than sequential `llm_query` calls for independent queries.
4. A `rlm_query(prompt, model=None)` function that spawns a **recursive RLM sub-call** for deeper thinking subtasks. The child gets its own REPL environment and can reason iteratively over the prompt, just like you. Use this when a subtask requires multi-step reasoning, code execution, or its own iterative problem-solving -- not just a simple one-shot answer. Falls back to `llm_query` if recursion is not available.
5. A `rlm_query_batched(prompts, model=None)` function that spawns multiple recursive RLM sub-calls. Each prompt gets its own child RLM. Falls back to `llm_query_batched` if recursion is not available.
6. A `SHOW_VARS()` function that returns all variables you have created in the REPL. Use this to check what variables exist before using FINAL_VAR.
7. The ability to use `print()` statements to view the output of your REPL code and continue your reasoning.
{custom_tools_section}

**When to use `llm_query` vs `rlm_query`:**
- Use `llm_query` for simple, one-shot tasks: extracting info from a chunk, summarizing text, answering a factual question, classifying content. These are fast single LLM calls.
- Use `rlm_query` when the subtask itself requires deeper thinking: multi-step reasoning, solving a sub-problem that needs its own REPL and iteration, or tasks where a single LLM call might not be enough. The child RLM can write and run code, query further sub-LLMs, and iterate to find the answer.

**Breaking down problems:** You must break problems into more digestible components—whether that means chunking or summarizing a large context, or decomposing a hard task into easier sub-problems and delegating them via `llm_query` / `rlm_query`. Use the REPL to write a **programmatic strategy** that uses these LLM calls to solve the problem, as if you were building an agent: plan steps, branch on results, combine answers in code.

**REPL for computation:** You can also use the REPL to compute programmatic steps (e.g. `math.sin(x)`, distances, physics formulas) and then chain those results into an LLM call. For complex math or physics, compute intermediate quantities in code and pass the numbers to the LM for interpretation or the final answer. Example: data describes an electron in a magnetic field undergoing helical motion; task is to find the entry angle.
```repl
import math
# Suppose the context or an earlier LM call gave us: B, m, q, pitch, R (radius). Extract or set them.
# Helical motion: v_parallel = pitch * (q*B)/(2*pi*m), v_perp = R * (q*B)/m. Entry angle theta: tan(theta) = v_perp/v_parallel.
v_parallel = pitch * (q * B) / (2 * math.pi * m)
v_perp = R * (q * B) / m
theta_rad = math.atan2(v_perp, v_parallel)
theta_deg = math.degrees(theta_rad)
final_answer = llm_query(f"An electron entered a B field and underwent helical motion. Computed entry angle: {{theta_deg:.2f}} deg. State the answer clearly for the user.")
```
You will only be able to see truncated outputs from the REPL environment, so you should use the query LLM function on variables you want to analyze. You will find this function especially useful when you have to analyze the semantics of the context. Use these variables as buffers to build up your final answer.
Make sure to explicitly look through the entire context in REPL before answering your query. Break the context and the problem into digestible pieces: e.g. figure out a chunking strategy, break up the context into smart chunks, query an LLM per chunk and save answers to a buffer, then query an LLM over the buffers to produce your final answer.

You can use the REPL environment to help you understand your context, especially if it is huge. Remember that your sub LLMs are powerful -- they can fit around 500K characters in their context window, so don't be afraid to put a lot of context into them. For example, a viable strategy is to feed 10 documents per sub-LLM query. Analyze your input data and see if it is sufficient to just fit it in a few sub-LLM calls!

When you want to execute Python code in the REPL environment, wrap it in triple backticks with 'repl' language identifier. For example, say we want our recursive model to search for the magic number in the context (assuming the context is a string), and the context is very long, so we want to chunk it:
```repl
chunk = context[:10000]
answer = llm_query(f"What is the magic number in the context? Here is the chunk: {{chunk}}")
print(answer)
```

As an example, suppose you're trying to answer a question about a book. You can iteratively chunk the context section by section, query an LLM on that chunk, and track relevant information in a buffer.
```repl
query = "In Harry Potter and the Sorcerer's Stone, did Gryffindor win the House Cup because they led?"
for i, section in enumerate(context):
    if i == len(context) - 1:
        buffer = llm_query(f"You are on the last section of the book. So far you know that: {{buffers}}. Gather from this last section to answer {{query}}. Here is the section: {{section}}")
        print(f"Based on reading iteratively through the book, the answer is: {{buffer}}")
    else:
        buffer = llm_query(f"You are iteratively looking through a book, and are on section {{i}} of {{len(context)}}. Gather information to help answer {{query}}. Here is the section: {{section}}")
        print(f"After section {{i}} of {{len(context)}}, you have tracked: {{buffer}}")
```

As another example, when the context isn't that long (e.g. >100M characters), a simple but viable strategy is, based on the context chunk lengths, to combine them and recursively query an LLM over chunks. For example, if the context is a List[str], we ask the same query over each chunk using `llm_query_batched` for concurrent processing:
```repl
query = "A man became famous for his book "The Great Gatsby". How many jobs did he have?"
# Suppose our context is ~1M chars, and we want each sub-LLM query to be ~0.1M chars so we split it into 10 chunks
chunk_size = len(context) // 10
chunks = []
for i in range(10):
    if i < 9:
        chunk_str = "\n".join(context[i*chunk_size:(i+1)*chunk_size])
    else:
        chunk_str = "\n".join(context[i*chunk_size:])
    chunks.append(chunk_str)

# Use batched query for concurrent processing - much faster than sequential calls!
prompts = [f"Try to answer the following query: {{query}}. Here are the documents:\n{{chunk}}. Only answer if you are confident in your answer based on the evidence." for chunk in chunks]
answers = llm_query_batched(prompts)
for i, answer in enumerate(answers):
    print(f"I got the answer from chunk {{i}}: {{answer}}")
final_answer = llm_query(f"Aggregating all the answers per chunk, answer the original query about total number of jobs: {{query}}\\n\\nAnswers:\\n" + "\\n".join(answers))
```

For subtasks that require deeper reasoning (e.g. solving a complex sub-problem), use `rlm_query` instead. The child gets its own REPL to iterate; you can then use the result in parent logic:
```repl
# Child RLM solves the sub-problem in its own REPL; we use the result in code
trend = rlm_query(f"Analyze this dataset and conclude with one word: up, down, or stable: {{data}}")
if "up" in trend.lower():
    recommendation = "Consider increasing exposure."
elif "down" in trend.lower():
    recommendation = "Consider hedging."
else:
    recommendation = "Hold position."
final_answer = llm_query(f"Given trend={{trend}} and recommendation={{recommendation}}, one-sentence summary for the user.")
```

As a final example, implement the solution as a **program**: try one approach via `rlm_query`; inspect the result and branch. If it suffices, use it. If not, break into one easier subproblem and delegate that only. More branches, one path runs—don't load the model. Example: prove sqrt 2 irrational.
```repl
r = rlm_query("Prove sqrt 2 is irrational. Give a 1-2 sentence proof, or reply only: USE_LEMMA or USE_CONTRADICTION.")
if "USE_LEMMA" in r.upper():
    final_answer = rlm_query("Prove 'n^2 even => n even' then use it to show sqrt 2 irrational. Two sentences.")

IMPORTANT: When you are done with the iterative process, you MUST provide a final answer inside a FINAL function when you have completed your task, NOT in code. Do not use these tags unless you have completed your task. You have two options:
1. Use FINAL(your final answer here) to provide the answer directly
2. Use FINAL_VAR(variable_name) to return a variable you have created in the REPL environment as your final output

WARNING - COMMON MISTAKE: FINAL_VAR retrieves an EXISTING variable. You MUST create and assign the variable in a ```repl``` block FIRST, then call FINAL_VAR in a SEPARATE step. For example:
- WRONG: Calling FINAL_VAR(my_answer) without first creating `my_answer` in a repl block
- CORRECT: First run ```repl
my_answer = "the result"
print(my_answer)
``` then in the NEXT response call FINAL_VAR(my_answer)

If you're unsure what variables exist, you can call SHOW_VARS() in a repl block to see all available variables.

Think step by step carefully, plan, and execute this plan immediately in your response -- do not just say "I will do this" or "I will do that". Output to the REPL environment and recursive LLMs as much as possible. Remember to explicitly answer the original query in your final answer.
"""
)


def build_rlm_system_prompt(
    system_prompt: str,
    query_metadata: QueryMetadata,
    custom_tools: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """
    Build the initial system prompt for the REPL environment based on extra prompt metadata.

    Args:
        system_prompt: The base system prompt template.
        query_metadata: QueryMetadata object containing context metadata.
        custom_tools: Optional dict of custom tools to include in the prompt.

    Returns:
        List of message dictionaries
    """
    from rlm.environments.base_env import format_tools_for_prompt

    context_lengths = query_metadata.context_lengths
    context_total_length = query_metadata.context_total_length
    context_type = query_metadata.context_type

    # If there are more than 100 chunks, truncate to the first 100 chunks.
    if len(context_lengths) > 100:
        others = len(context_lengths) - 100
        context_lengths = str(context_lengths[:100]) + "... [" + str(others) + " others]"

    # Format custom tools section if provided
    tools_formatted = format_tools_for_prompt(custom_tools)
    if tools_formatted:
        custom_tools_section = (
            f"\n6. Custom tools and data available in the REPL:\n{tools_formatted}"
        )
    else:
        custom_tools_section = ""

    # Insert custom tools section into the system prompt
    final_system_prompt = system_prompt.format(custom_tools_section=custom_tools_section)

    metadata_prompt = f"Your context is a {context_type} with {context_total_length} total characters, and is broken up into chunks of char lengths: {context_lengths}."

    return [
        {"role": "system", "content": final_system_prompt},
        {"role": "user", "content": metadata_prompt},
    ]


USER_PROMPT = """Think step-by-step on what to do using the REPL environment (which contains the context) to answer the prompt.\n\nContinue using the REPL environment, which has the `context` variable, and querying sub-LLMs by writing to ```repl``` tags, and determine your answer. Your next action:"""
USER_PROMPT_WITH_ROOT = """Think step-by-step on what to do using the REPL environment (which contains the context) to answer the original prompt: \"{root_prompt}\".\n\nContinue using the REPL environment, which has the `context` variable, and querying sub-LLMs by writing to ```repl``` tags, and determine your answer. Your next action:"""


def build_user_prompt(
    root_prompt: str | None = None,
    iteration: int = 0,
    context_count: int = 1,
    history_count: int = 0,
) -> dict[str, str]:
    if iteration == 0:
        safeguard = "You have not interacted with the REPL environment or seen your prompt / context yet. Your next action should be to look through and figure out how to answer the prompt, so don't just provide a final answer yet.\n\n"
        prompt = safeguard + (
            USER_PROMPT_WITH_ROOT.format(root_prompt=root_prompt) if root_prompt else USER_PROMPT
        )
    else:
        prompt = "The history before is your previous interactions with the REPL environment. " + (
            USER_PROMPT_WITH_ROOT.format(root_prompt=root_prompt) if root_prompt else USER_PROMPT
        )

    # Inform model about multiple contexts if present
    if context_count > 1:
        prompt += f"\n\nNote: You have {context_count} contexts available (context_0 through context_{context_count - 1})."

    # Inform model about prior conversation histories if present
    if history_count > 0:
        if history_count == 1:
            prompt += "\n\nNote: You have 1 prior conversation history available in the `history` variable."
        else:
            prompt += f"\n\nNote: You have {history_count} prior conversation histories available (history_0 through history_{history_count - 1})."

    return {"role": "user", "content": prompt}


########################################################################
# Workspace substrate prompts
########################################################################

WORKSPACE_SYSTEM_PROMPT_TEMPLATE = textwrap.dedent(
    """\
    You are an autonomous reasoning agent operating in a durable workspace.

    The workspace is a real filesystem directory that persists across your
    turns. Your "memory" is the files you read, write, and edit. You drive
    the workspace by emitting one or more ``<action>`` elements per turn;
    each action runs and returns an observation that you'll see next turn.

    # Workspace layout
    - ``_rlm_query_0.txt`` — the root task you must solve. Read it first.
    - ``_rlm_query_<N>.txt`` — additional user-supplied context, when present.
    - ``_rlm_notes/`` — scratch notes you write for yourself.
    - ``_rlm_artifacts/`` — artifacts and outputs (incl. spilled tool output
      under ``_rlm_artifacts/_observations/`` when an observation exceeded
      the per-call cap).
    - ``_rlm_state/`` — reserved runtime state. You may not write here.

    # How to act
    Each turn, emit one or more ``<action tool="...">...</action>`` elements.
    You may write reasoning prose around them; the runtime extracts only the
    ``<action>`` blocks but preserves the surrounding prose so you can
    anchor your planning across turns. Self-closing ``<action ... />`` is
    allowed for tools that take no body.

    Emit ``<action tool="final"><answer>...</answer></action>`` to terminate
    the run with your final answer. You may also include zero or more
    ``<artifact path="..." />`` children to mark files as part of the result.

    # Available tools
    {tool_descriptions}

    # Hard rules
    - Output well-formed XML for action elements. The parser is tolerant
      about bodies (raw ``<``, ``&``, code are all fine inside an action
      body), but action open/close tags must balance.
    - Read-only tool failures do NOT halt the rest of the turn; mutating
      tool failures DO halt the rest of the batch in this turn.
    - Per-call output above the configured cap is auto-spilled to
      ``_rlm_artifacts/_observations/`` and the observation is replaced
      with a short summary plus a path. Don't worry about it.
    - You may not write inside ``_rlm_state/``.
    {depth_rule}
    """
)


def _format_tool_descriptions(include_rlm_query: bool) -> str:
    """Pull short descriptions from the workspace tool registry."""
    # Imported lazily to avoid a circular import at module load time
    # (workspace_tools → core.types → … → utils.prompts).
    from rlm.workspace_tools import all_tool_names, get_spec

    lines: list[str] = []
    for name in all_tool_names():
        if name == "rlm_query" and not include_rlm_query:
            continue
        spec = get_spec(name)
        lines.append(f"- ``{name}`` — {spec.short_description}")
    return "\n".join(lines)


def build_workspace_system_prompt(
    *,
    depth: int,
    max_depth: int,
    custom_system_prompt: str | None = None,
) -> str:
    """Build the workspace system prompt, depth-aware.

    At ``depth == max_depth`` the ``rlm_query`` tool is omitted from the
    tool list and an explicit no-recursion rule is added (Decision #23).
    """
    if custom_system_prompt is not None:
        return custom_system_prompt

    at_max_depth = depth >= max_depth
    tool_descriptions = _format_tool_descriptions(include_rlm_query=not at_max_depth)
    if at_max_depth:
        depth_rule = (
            "- You are at the maximum recursion depth. The ``rlm_query`` "
            "tool is unavailable for this run, and the in-container "
            "``rlm_query`` / ``rlm_query_batched`` Python helpers will "
            "return error strings. Use ``llm_query`` instead."
        )
    else:
        depth_rule = ""
    return WORKSPACE_SYSTEM_PROMPT_TEMPLATE.format(
        tool_descriptions=tool_descriptions,
        depth_rule=depth_rule,
    )


def build_workspace_initial_user_prompt(*, root_prompt: str | None = None) -> str:
    """First user turn for the workspace loop.

    The root task itself lives at ``_rlm_query_0.txt`` inside the workspace;
    the model reads it via ``read_file``. ``root_prompt`` is an optional
    short pointer the user can pass to bias the very first turn.
    """
    base = (
        "Begin by reading ``_rlm_query_0.txt`` (and any ``_rlm_query_<N>.txt`` "
        "context files) to understand the task. Plan, then act."
    )
    if root_prompt:
        base += f'\n\nUser-provided pointer (preview of the task): "{root_prompt}"'
    return base


def render_observation(action_id: str, observation: WorkspaceObservation) -> str:
    """Render a single observation for inclusion in the next user message."""
    parts = [f'<observation action_id="{action_id}" tool="{observation.tool}">']
    if observation.error:
        parts.append(f"[error] {observation.error}")
    if observation.stdout:
        parts.append(observation.stdout.rstrip())
    if observation.stderr:
        parts.append("[stderr]")
        parts.append(observation.stderr.rstrip())
    if observation.artifacts:
        parts.append("[artifacts] " + ", ".join(observation.artifacts))
    if observation.final_answer is not None:
        parts.append(f"[final] {observation.final_answer}")
        if observation.final_artifacts:
            parts.append("[final artifacts] " + ", ".join(observation.final_artifacts))
    parts.append("</observation>")
    return "\n".join(parts)


def format_workspace_iteration(iteration: WorkspaceIteration) -> list[dict[str, str]]:
    """Convert a completed ``WorkspaceIteration`` into next-turn messages.

    Returns two messages: the assistant's full response (prose + actions)
    and a synthetic user message containing the rendered observations and a
    one-line snapshot summary. Parse-retry attempts are NOT included here;
    they live in ``iteration.parse_attempts`` for the visualizer.
    """
    obs_chunks: list[str] = []
    # Pair each action with its observation by index. Length should match
    # except in the unusual case where execution halted early; render what
    # we have and let the model see the gap.
    for idx, observation in enumerate(iteration.observations):
        action_id = f"t{iteration.iteration}.a{idx + 1}"
        obs_chunks.append(render_observation(action_id, observation))

    if iteration.snapshot is not None:
        snap = iteration.snapshot
        changed = ", ".join(snap.changed_files) if snap.changed_files else "(no changes)"
        obs_chunks.append(
            f'<snapshot turn="{snap.turn}" commit="{snap.commit_sha[:7]}">'
            f"\nchanged: {changed}\n</snapshot>"
        )

    user_message = "\n\n".join(obs_chunks) if obs_chunks else "(no observations)"
    return [
        {"role": "assistant", "content": iteration.response},
        {"role": "user", "content": user_message},
    ]


def build_parse_retry_message(error: str, fragment: str | None) -> str:
    """Synthetic user message for the parse-and-retry inner loop."""
    msg = (
        "Your previous response was malformed: "
        f"{error}\n\n"
        "Reply again, including at least one well-formed ``<action>`` element."
    )
    if fragment:
        msg += f"\n\nOffending fragment (truncated):\n{fragment[:500]}"
    return msg
