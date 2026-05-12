import textwrap

from rlm.core.types import (
    WorkspaceIteration,
    WorkspaceObservation,
)
from rlm.utils.action_parser import strip_reasoning_blocks

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
      tool failures DO halt the rest of the batch in this turn, including
      any ``final`` in the same batch — do not commit a final answer in
      the same turn as a mutating action whose failure would invalidate it.
    - Per-call output above the configured cap is auto-spilled to
      ``_rlm_artifacts/_observations/`` and the observation is replaced
      with a short summary plus a path. Read the file in the path to get full output.
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
        entry = f"- ``{name}`` — {spec.short_description}"
        if spec.example:
            entry += f"\n  {spec.example}"
        lines.append(entry)
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

    The assistant ``content`` is stripped of reasoning blocks
    (``<think>`` and ``<|channel|>thought`` spans) before being replayed.
    Google and Alibaba both document multi-turn semantics where the prior
    turn's thought is dropped on replay; the strip also avoids paying
    input-token cost for monologue the substrate already discarded.
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
        {"role": "assistant", "content": strip_reasoning_blocks(iteration.response)},
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
