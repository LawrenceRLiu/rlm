import re
import textwrap
from html import escape

from rlm.core.config import PromptHistoryConfig
from rlm.core.types import (
    WorkspaceAction,
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

    # Tool intent
    Use ``write_file``, ``append_file``, and ``edit_file`` for durable
    workspace edits. Use ``python`` and ``shell`` for scratch computation,
    inspection, validation, tests, parsing, and complex one-off transforms.
    Do not use ``python`` or ``shell`` as a substitute for simple durable file
    edits such as echoing text into files, opening a file only to write
    generated prose/code, or printing an entire generated artifact to stdout.
    Keep command stdout focused on diagnostics, summaries, test results, and
    small computed values. If you need to inspect durable file contents, use
    ``read_file``.

    # How to act
    Each turn, emit one or more ``<action tool="...">...</action>`` elements.
    You may write reasoning prose around them; the runtime extracts only the
    ``<action>`` blocks. Future turns receive compact receipts for prior
    actions and observations, not guaranteed verbatim replay of your prose or
    durable edit bodies. Use workspace files for durable memory and
    ``read_file`` when you need to inspect file contents. Self-closing
    ``<action ... />`` is allowed for tools that take no body.

    You may include one short ``<note>...</note>`` before your actions to
    preserve turn-to-turn intent. Use it for the current plan, open questions,
    and file paths to revisit. Keep it brief. Do not put file contents, code
    blocks, proofs, large outputs, generated artifacts, or action XML inside
    notes; overlong or content-like notes are omitted from future replay.

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


FILE_BODY_TOOLS = {"write_file", "append_file", "edit_file"}
COMMAND_TOOLS = {"python", "shell"}
NOTE_RE = re.compile(r"<note>(.*?)</note>", re.DOTALL)


def truncate_for_prompt(text: str, cap: int) -> tuple[str, bool]:
    """Return ``text`` capped for prompt replay plus whether truncation happened."""
    if cap < 0 or len(text) <= cap:
        return text, False
    return text[:cap], True


def action_changed_files(observation: WorkspaceObservation | None) -> bool:
    if observation is None or not observation.data:
        return False
    changed = observation.data.get("changed_paths")
    removed = observation.data.get("removed_paths")
    return bool(changed or removed)


def extract_turn_note(
    response: str,
    history_config: PromptHistoryConfig,
) -> tuple[str | None, str | None]:
    """Extract the first bounded <note> from a response.

    Returns ``(note, None)`` for a valid note, ``(None, reason)`` for a note
    that was present but intentionally omitted, and ``(None, None)`` when the
    response had no note.
    """
    cleaned = strip_reasoning_blocks(response)
    match = NOTE_RE.search(cleaned)
    if match is None:
        return None, None

    note = "\n".join(line.strip() for line in match.group(1).strip().splitlines()).strip()
    if not note:
        return None, "empty"

    lower = note.lower()
    if "```" in note or "<action" in lower or "</action" in lower:
        return None, "content-like"
    if len(note) > history_config.max_turn_note_chars:
        return None, "too-long"
    if len(note.splitlines()) > history_config.max_turn_note_lines:
        return None, "too-many-lines"
    return note, None


def render_turn_note(
    iteration: WorkspaceIteration, history_config: PromptHistoryConfig
) -> str | None:
    note, omitted_reason = extract_turn_note(iteration.response, history_config)
    if note is not None:
        return (
            f'<turn_note turn="{iteration.iteration}">\n{escape(note, quote=False)}\n</turn_note>'
        )
    if omitted_reason is not None:
        return (
            f'<turn_note turn="{iteration.iteration}" omitted="true" '
            f'reason="{escape(omitted_reason, quote=True)}" />'
        )
    return None


def render_action_replay(
    action_id: str,
    action: WorkspaceAction,
    observation: WorkspaceObservation | None,
    history_config: PromptHistoryConfig,
) -> str:
    """Compact model-facing replay of what the assistant attempted.

    Full raw action bodies stay in ``WorkspaceIteration`` for logging and
    visualizer use. This renderer intentionally hides ordinary durable edit
    bodies so the transcript does not become a second filesystem.
    """
    status = "error" if observation is not None and observation.error else "ok"
    args = " ".join(
        f'{escape(str(k), quote=True)}="{escape(str(v), quote=True)}"'
        for k, v in action.args.items()
    )
    attrs = (
        f'action_id="{escape(action_id, quote=True)}" '
        f'tool="{escape(action.tool, quote=True)}" status="{status}"'
    )
    if args:
        attrs += f" {args}"

    body = action.body or ""
    body_lines = len(body.splitlines()) if body else 0
    body_chars = len(body)

    if action.tool in FILE_BODY_TOOLS:
        lines = [f"<action_replay {attrs}>"]
        lines.append(
            f"[body omitted from replay: {body_chars} chars, {body_lines} lines; "
            "use read_file to inspect durable contents]"
        )
        if observation is not None and observation.error:
            lines.append(f"[error] {observation.error}")
        lines.append("</action_replay>")
        return "\n".join(lines)

    if action.tool in COMMAND_TOOLS:
        cap = (
            history_config.max_mutating_command_body_replay_chars
            if action_changed_files(observation)
            else history_config.max_command_body_replay_chars
        )
        source, truncated = truncate_for_prompt(body, cap)
        lines = [f"<action_replay {attrs}>"]
        if source:
            lines.append("[source]")
            lines.append(source.rstrip())
        if truncated:
            lines.append(
                f"[source truncated for replay: {body_chars} chars total, showing first {cap}]"
            )
        if not source:
            lines.append("[source omitted: empty body]")
        if observation is not None and observation.error:
            lines.append(f"[error] {observation.error}")
        lines.append("</action_replay>")
        return "\n".join(lines)

    # Read-only / terminal / query actions have small bodies or no durable
    # write payload. Keep a compact receipt; observations carry the result.
    lines = [f"<action_replay {attrs}>"]
    if body:
        lines.append(f"[body: {body_chars} chars]")
    else:
        lines.append("[no body]")
    if observation is not None and observation.error:
        lines.append(f"[error] {observation.error}")
    lines.append("</action_replay>")
    return "\n".join(lines)


def render_observation(
    action_id: str,
    observation: WorkspaceObservation,
    *,
    action: WorkspaceAction | None = None,
    compact: bool = False,
    history_config: PromptHistoryConfig | None = None,
) -> str:
    """Render a single observation for inclusion in the next user message."""
    if history_config is None:
        history_config = PromptHistoryConfig()

    if compact:
        parts = [f'<observation_receipt action_id="{action_id}" tool="{observation.tool}">']
        if observation.error:
            parts.append(f"[error] {observation.error}")
        else:
            parts.append("[status] ok")
        if observation.stdout:
            parts.append(f"[stdout omitted from replay: {len(observation.stdout)} chars]")
        if observation.stderr:
            parts.append(f"[stderr omitted from replay: {len(observation.stderr)} chars]")
        if action is not None and action.args:
            args = ", ".join(f"{k}={v}" for k, v in action.args.items())
            parts.append(f"[args] {args}")
        if observation.artifacts:
            parts.append("[artifacts] " + ", ".join(observation.artifacts))
        if observation.final_answer is not None:
            parts.append("[final answer omitted from replay]")
        parts.append("</observation_receipt>")
        return "\n".join(parts)

    parts = [f'<observation action_id="{action_id}" tool="{observation.tool}">']
    if observation.error:
        parts.append(f"[error] {observation.error}")
    if observation.stdout:
        stdout = observation.stdout.rstrip()
        if observation.tool in COMMAND_TOOLS and action_changed_files(observation):
            cap = history_config.max_mutating_command_stdout_replay_chars
            stdout, truncated = truncate_for_prompt(stdout, cap)
            parts.append(stdout.rstrip())
            if truncated:
                parts.append(
                    f"[stdout truncated for replay: {len(observation.stdout)} chars total, "
                    f"showing first {cap}; rerun or read artifacts/files for details]"
                )
        else:
            parts.append(stdout)
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


def format_workspace_iteration(
    iteration: WorkspaceIteration,
    *,
    history_config: PromptHistoryConfig | None = None,
    age: int = 0,
) -> list[dict[str, str]]:
    """Convert a completed ``WorkspaceIteration`` into next-turn messages.

    Returns two messages: a compact assistant-side action replay and a
    synthetic user message containing the rendered observations plus a one-line
    snapshot summary. Parse-retry attempts are NOT included here; they live in
    ``iteration.parse_attempts`` for the visualizer.

    When a turn had no parsed actions, the assistant ``content`` falls back to
    the stripped raw response. Google and Alibaba both document multi-turn
    semantics where the prior turn's thought is dropped on replay; the strip
    also avoids paying input-token cost for monologue the substrate already
    discarded.
    """
    if history_config is None:
        history_config = PromptHistoryConfig()

    compact_observations = age >= history_config.full_observation_turns

    action_chunks: list[str] = []
    turn_note = render_turn_note(iteration, history_config)
    if turn_note is not None:
        action_chunks.append(turn_note)

    for idx, action in enumerate(iteration.actions):
        action_id = f"t{iteration.iteration}.a{idx + 1}"
        observation = iteration.observations[idx] if idx < len(iteration.observations) else None
        action_chunks.append(
            render_action_replay(
                action_id=action_id,
                action=action,
                observation=observation,
                history_config=history_config,
            )
        )

    obs_chunks: list[str] = []
    # Pair each action with its observation by index. Length should match
    # except in the unusual case where execution halted early; render what
    # we have and let the model see the gap.
    for idx, observation in enumerate(iteration.observations):
        action_id = f"t{iteration.iteration}.a{idx + 1}"
        action = iteration.actions[idx] if idx < len(iteration.actions) else None
        obs_chunks.append(
            render_observation(
                action_id,
                observation,
                action=action,
                compact=compact_observations,
                history_config=history_config,
            )
        )

    if iteration.snapshot is not None:
        snap = iteration.snapshot
        changed = ", ".join(snap.changed_files) if snap.changed_files else "(no changes)"
        obs_chunks.append(
            f'<snapshot turn="{snap.turn}" commit="{snap.commit_sha[:7]}">'
            f"\nchanged: {changed}\n</snapshot>"
        )

    user_message = "\n\n".join(obs_chunks) if obs_chunks else "(no observations)"
    return [
        {
            "role": "assistant",
            "content": "\n\n".join(action_chunks)
            if action_chunks
            else strip_reasoning_blocks(iteration.response),
        },
        {"role": "user", "content": user_message},
    ]


def format_workspace_history(
    iterations: list[WorkspaceIteration],
    *,
    history_config: PromptHistoryConfig | None = None,
) -> list[dict[str, str]]:
    """Render completed iterations for model-facing prompt replay."""
    if history_config is None:
        history_config = PromptHistoryConfig()
    messages: list[dict[str, str]] = []
    total = len(iterations)
    for idx, iteration in enumerate(iterations):
        age = total - idx - 1
        messages.extend(
            format_workspace_iteration(
                iteration,
                history_config=history_config,
                age=age,
            )
        )
    return messages


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
