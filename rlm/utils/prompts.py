import json
import textwrap
from html import escape
from typing import Any

from rlm.core.types import (
    WorkspaceAction,
    WorkspaceIteration,
    WorkspaceObservation,
)
from rlm.utils.action_parser import strip_reasoning_blocks
from rlm.utils.native_tools import body_arg_name

# TODO: Surface this as an observation/prompt-rendering config knob if we need
# per-benchmark tuning. For now, keep model-facing shell/python artifact lists
# short while preserving full artifacts in logs and structured observations.
MAX_RENDERED_ARTIFACT_PATHS = 5

WORKSPACE_SYSTEM_PROMPT_TEMPLATE = textwrap.dedent(
    """\
    You are an autonomous reasoning agent operating in a durable workspace.

    The workspace is a real filesystem directory that persists across your
    turns. Your "memory" is the files you read, write, and edit. You drive
    the workspace by emitting one or more ``<action>`` elements per turn;
    each action runs and returns an observation that you'll see next turn.

    # Workspace
    Begin by reading ``/_rlm_state/_rlm_query_0.txt`` (and any
    ``/_rlm_state/_rlm_query_<N>.txt`` context files) to understand the
    task. Then list the workspace root to see what files and directories
    are available; the task and its files have already been placed there
    for you. ``list_directory`` shows the role that created and last
    modified each entry (``user`` vs ``system`` vs ``assistant``), which
    helps you tell task material apart from substrate scaffolding.

    # Tool intent
    Use ``write_file``, ``append_file``, and ``edit_file`` for durable
    workspace edits. Use ``python`` and ``shell`` for scratch computation,
    inspection, validation, tests, parsing, and complex one-off transforms.
    Use ``/_rlm_notes/`` for scratch notes you want to persist across turns
    and ``/_rlm_artifacts/`` for durable outputs you may want to return
    with your final answer. ``/_rlm_state/`` is substrate-owned — you may
    read it but not write inside it.
    Do not use ``python`` or ``shell`` as a substitute for simple durable file
    edits such as echoing text into files, opening a file only to write
    generated prose/code, or printing an entire generated artifact to stdout.
    Keep command stdout focused on diagnostics, summaries, test results, and
    small computed values. If you need to inspect durable file contents, use
    ``read_file``.

    # How to act
    Each turn, emit one or more ``<action tool="...">...</action>`` elements.
    You may write reasoning prose around them; the runtime extracts only the
    ``<action>`` blocks. Self-closing ``<action ... />`` is allowed for tools
    that take no body.

    Emit ``<action tool="final"><answer>...</answer></action>`` to terminate
    the run with your final answer. You may also include zero or more
    ``<artifact path="..." />`` children to mark files as part of the result.

    # Decomposition with ``llm_query`` and ``rlm_query``
    You reason better when you offload self-contained sub-*questions* to a
    fresh context instead of keeping everything in head. When you hit a
    discrete sub-question whose answer would feed into your reasoning —
    extracting a fact from a long document, summarising a passage, computing
    or sanity-checking an intermediate result, verifying a single derivation
    step — reach for ``llm_query`` (one-shot) or ``rlm_query`` (multi-turn
    with workspace access). These tools widen your inputs; they are not a
    substitute for your own planning or for the open-ended structure of the
    task itself. Decompose specific questions, not the thinking.

    # Returning your answer
    Return the result inline as the body of ``<answer>`` in ``final``. The
    workspace is scratch space (``_rlm_notes/`` for thinking, ``_rlm_artifacts/``
    for intermediates), not a substitute for the answer. Attach
    ``<artifact path="..." />`` children only when a file genuinely belongs
    with the answer — i.e. when it is the answer (a built binary, a
    requested file deliverable, a dataset), or when the answer references it
    in a way that wouldn't make sense without the file on disk. Notes, drafts,
    and scratch files do not belong here.

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
    - You may not write inside ``/_rlm_state/``.
    - If the prompt grows past a token threshold the substrate periodically
      summarizes the trajectory and resets the visible history; treat the
      workspace files (and ``/_rlm_state/_rlm_query_0.txt`` in particular)
      as the authoritative state and re-read them when in doubt.
    {depth_rule}
    """
)

NATIVE_WORKSPACE_SYSTEM_PROMPT_TEMPLATE = textwrap.dedent(
    """\
    You are an autonomous reasoning agent operating in a durable workspace.

    The workspace is a real filesystem directory that persists across your
    turns. Your memory is the files you read, write, and edit. Use the
    provided native tool calls to act; do not print XML, JSON wrappers, or
    markdown code fences as a substitute for tool calls.

    # Workspace
    Begin by reading ``/_rlm_state/_rlm_query_0.txt`` (and any
    ``/_rlm_state/_rlm_query_<N>.txt`` context files) to understand the
    task. Then list the workspace root to see what files and directories
    are available; the task and its files have already been placed there
    for you. ``list_directory`` shows the role that created and last
    modified each entry (``user`` vs ``system`` vs ``assistant``), which
    helps you tell task material apart from substrate scaffolding.

    # Tool intent
    Use ``write_file``, ``append_file``, and ``edit`` for durable workspace
    edits. Use ``run_python_command`` and ``run_shell_command`` for scratch
    computation, inspection, validation, tests, parsing, and complex one-off
    transforms. Use ``/_rlm_notes/`` for scratch notes you want to persist
    across turns and ``/_rlm_artifacts/`` for durable outputs you may want
    to return with your final answer. ``/_rlm_state/`` is substrate-owned
    — you may read it but not write inside it. In ``run_python_command``,
    the helper functions ``llm_query``, ``llm_query_batched``, ``rlm_query``,
    and ``rlm_query_batched`` are already imported, so use Python for loops
    over documents and batched model calls.

    # Whitespace-sensitive tools
    - ``edit.old_string`` is exact literal text. Whitespace, indentation, and
      newlines must match the file contents. Read the file first and include
      enough surrounding context to make one match, or set ``replace_all``.
    - ``run_shell_command.command`` is passed as a command string to bash via
      a script file. Quote paths and use heredocs for large multiline literals.
    - ``write_file.content`` and ``run_python_command.code`` are strings;
      include the exact content/code you want executed or written.

    # Decomposition with ``llm_query`` and ``rlm_query``
    You reason better when you offload self-contained sub-*questions* to a
    fresh context instead of keeping everything in head. When you hit a
    discrete sub-question whose answer would feed into your reasoning —
    extracting a fact from a long document, summarising a passage, computing
    or sanity-checking an intermediate result, verifying a single derivation
    step — reach for ``llm_query`` (one-shot) or ``rlm_query`` (multi-turn
    with workspace access). These tools widen your inputs; they are not a
    substitute for your own planning or for the open-ended structure of the
    task itself. Decompose specific questions, not the thinking.

    # Returning your answer
    Return the result inline as the ``answer`` argument of the ``final``
    tool call. The workspace is scratch space (``_rlm_notes/`` for thinking,
    ``_rlm_artifacts/`` for intermediates), not a substitute for the answer.
    List paths in the ``artifacts`` argument of ``final`` only when a file
    genuinely belongs with the answer — i.e. when it is the answer (a built
    binary, a requested file deliverable, a dataset), or when the answer
    references it in a way that wouldn't make sense without the file on disk.
    Notes, drafts, and scratch files do not belong here.

    # Available tools
    {tool_descriptions}

    # Hard rules
    - Every assistant turn must make at least one native tool call.
    - Read-only tool failures do NOT halt the rest of the turn; mutating tool
      failures DO halt the rest of the batch in this turn, including any
      ``final`` in the same batch.
    - Per-call output above the configured cap is auto-spilled to
      ``_rlm_artifacts/_observations/`` and replaced with a short summary path.
    - You may not write inside ``/_rlm_state/``.
    - If the prompt grows past a token threshold the substrate periodically
      summarizes the trajectory and resets the visible history; treat the
      workspace files (and ``/_rlm_state/_rlm_query_0.txt`` in particular)
      as the authoritative state and re-read them when in doubt.
    {depth_rule}
    """
)


def _format_tool_descriptions(include_rlm_query: bool, *, native: bool = False) -> str:
    """Pull short descriptions from the workspace tool registry."""
    # Imported lazily to avoid a circular import at module load time
    # (workspace_tools → core.types → … → utils.prompts).
    from rlm.workspace_tools import get_spec, native_tool_names, xml_tool_names

    lines: list[str] = []
    names = native_tool_names() if native else xml_tool_names()
    for name in names:
        if name == "rlm_query" and not include_rlm_query:
            continue
        if native:
            entry = f"- ``{name}`` — {NATIVE_TOOL_DESCRIPTIONS[name]}"
        else:
            spec = get_spec(name)
            entry = f"- ``{name}`` — {spec.short_description}"
        if not native and spec.example:
            entry += f"\n  {spec.example}"
        lines.append(entry)
    return "\n".join(lines)


def build_workspace_system_prompt(
    *,
    depth: int,
    max_depth: int,
    custom_system_prompt: str | None = None,
    action_format: str = "native",
) -> str:
    """Build the workspace system prompt, depth-aware.

    At ``depth == max_depth`` the ``rlm_query`` tool is omitted from the
    tool list and an explicit no-recursion rule is added (Decision #23).
    """
    if custom_system_prompt is not None:
        return custom_system_prompt

    at_max_depth = depth >= max_depth
    native = action_format == "native"
    tool_descriptions = _format_tool_descriptions(
        include_rlm_query=not at_max_depth,
        native=native,
    )
    if at_max_depth:
        depth_rule = (
            "- You are at the maximum recursion depth. The ``rlm_query`` "
            "tool is unavailable for this run, and the in-container "
            "``rlm_query`` / ``rlm_query_batched`` Python helpers will "
            "return error strings. Use ``llm_query`` instead."
        )
    else:
        depth_rule = ""
    template = (
        NATIVE_WORKSPACE_SYSTEM_PROMPT_TEMPLATE if native else WORKSPACE_SYSTEM_PROMPT_TEMPLATE
    )
    return template.format(
        tool_descriptions=tool_descriptions,
        depth_rule=depth_rule,
    )


def build_workspace_initial_user_prompt(*, root_prompt: str | None = None) -> str:
    """First user turn for the workspace loop.

    The root task lives at ``/_rlm_state/_rlm_query_0.txt`` inside the
    container workspace; the model reads it via ``read_file``. ``root_prompt``
    is an optional short pointer the user can pass to bias the very first turn.
    """
    base = (
        "Begin by reading ``/_rlm_state/_rlm_query_0.txt`` (and any "
        "``/_rlm_state/_rlm_query_<N>.txt`` context files) to understand the "
        "task. Plan, then act."
    )
    if root_prompt:
        base += f'\n\nUser-provided pointer (preview of the task): "{root_prompt}"'
    return base


COMMAND_TOOLS = {"python", "shell", "run_python_command", "run_shell_command"}

NATIVE_TOOL_DESCRIPTIONS: dict[str, str] = {
    "list_directory": (
        "Shallow listing of a workspace directory. Arguments: "
        '{"path": "."} where path is optional and workspace-relative.'
    ),
    "read_file": (
        "Read a slice of a workspace file. Arguments: "
        '{"path": "notes.md", "start_line": 1, "end_line": 50}; '
        "path is required, line bounds are optional and 1-indexed."
    ),
    "write_file": (
        "Create or completely overwrite a workspace file. Arguments: "
        '{"file_path": "hello.py", "content": "print(\\"hello\\")\\n"}.'
    ),
    "append_file": (
        "Append text verbatim to a workspace file, creating it if missing. "
        'Arguments: {"file_path": "log.txt", "content": "new entry\\n"}.'
    ),
    "edit": (
        "Replace exact literal text in a workspace file. Arguments: "
        '{"file_path": "src/foo.py", "old_string": "old\\n", '
        '"new_string": "new\\n", "replace_all": false}.'
    ),
    "run_shell_command": (
        "Run a shell command string inside the workspace container. Arguments: "
        '{"command": "python -m pytest", "directory": ".", "timeout": 300, '
        '"is_background": false}.'
    ),
    "run_python_command": (
        "Run Python code inside the workspace container. Arguments: "
        '{"code": "from pathlib import Path\\nprint(Path(\\".\\").resolve())", '
        '"timeout": 300}. The helpers llm_query, llm_query_batched, '
        "rlm_query, and rlm_query_batched are pre-imported."
    ),
    "llm_query": (
        "Single LM completion without recursion. Arguments: "
        '{"prompt": "Summarize this paragraph in one sentence."}.'
    ),
    "rlm_query": (
        "Spawn a child RLM with a copy-on-spawn workspace snapshot. Arguments: "
        '{"prompt": "Inspect summary files and synthesize the result."}.'
    ),
    "final": (
        "Terminate the run with the final answer. Put the result inline in "
        '``answer`` (e.g. ``{"answer": "<full proof or result here>"}``); '
        "the workspace is scratch space, not a substitute for the answer. "
        "Attach ``artifacts`` only when a file genuinely belongs with the "
        'answer — when it *is* the answer (e.g. ``{"answer": "see attached", '
        '"artifacts": ["report.pdf"]}``) or when the answer references it in '
        "a way that wouldn't make sense without the file."
    ),
}


def render_action_replay(
    action_id: str,
    action: WorkspaceAction,
    observation: WorkspaceObservation | None,
    *,
    action_format: str = "native",
) -> str:
    """Full-fidelity model-facing replay of what the assistant attempted.

    Action bodies are replayed verbatim; compression is handled by the
    substrate-level ``CompactionConfig`` once the prompt exceeds its token
    threshold, not by per-turn hiding.
    """
    status = "error" if observation is not None and observation.error else "ok"
    body = action.body or ""

    if action_format == "native":
        parts = [f"TOOL_CALL {action_id} tool={action.tool} status={status}"]
        if action.call_id:
            parts[0] += f" call_id={action.call_id}"
        # The body-bearing arg (e.g. write_file.content, run_shell_command.command)
        # is mirrored into ``action.body`` by the native parser. Rendering both
        # would replay the same payload twice in the prompt, so we strip the
        # body arg from the rendered JSON whenever a body is present.
        body_key = body_arg_name(action.tool) if body else None
        args_for_replay = (
            {k: v for k, v in action.args.items() if k != body_key}
            if body_key is not None
            else action.args
        )
        if args_for_replay:
            parts.append("args=" + json.dumps(args_for_replay, ensure_ascii=False, sort_keys=True))
        if body:
            parts.append("body:")
            parts.append(body.rstrip())
        if observation is not None and observation.error:
            parts.append(f"error: {observation.error}")
        return "\n".join(parts)

    args = " ".join(
        f'{escape(str(k), quote=True)}="{escape(str(v), quote=True)}"'
        for k, v in action.args.items()
    )
    attrs = (
        f'action_id="{escape(action_id, quote=True)}" '
        f'tool="{escape(action.tool, quote=True)}" status="{status}"'
    )
    if action.call_id:
        attrs += f' call_id="{escape(action.call_id, quote=True)}"'
    if args:
        attrs += f" {args}"

    lines = [f"<action_replay {attrs}>"]
    if body:
        lines.append("[body]")
        lines.append(body.rstrip())
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
    action_format: str = "native",
) -> str:
    """Render a single observation for inclusion in the next user message.

    Full-fidelity: stdout/stderr/error are replayed verbatim. Per-call
    spill to ``_rlm_artifacts/_observations/`` upstream of this renderer
    is the only truncation that applies; substrate-level compaction takes
    over once the cumulative prompt crosses ``CompactionConfig.threshold_tokens``.
    """
    if action_format == "native":
        status = "error" if observation.error else "ok"
        parts = [f"OBSERVATION {action_id} tool={observation.tool} status={status}"]
        if observation.error:
            parts.append(f"error: {observation.error}")
        if observation.stdout:
            parts.append(observation.stdout.rstrip())
        if observation.stderr:
            parts.append("stderr:")
            parts.append(observation.stderr.rstrip())
        if observation.artifacts:
            parts.append("artifacts: " + format_artifact_paths(observation.artifacts))
        if observation.final_answer is not None:
            parts.append(f"final: {observation.final_answer}")
            if observation.final_artifacts:
                parts.append("final artifacts: " + ", ".join(observation.final_artifacts))
        return "\n".join(parts)

    parts = [f'<observation action_id="{action_id}" tool="{observation.tool}">']
    if observation.error:
        parts.append(f"[error] {observation.error}")
    if observation.stdout:
        parts.append(observation.stdout.rstrip())
    if observation.stderr:
        parts.append("[stderr]")
        parts.append(observation.stderr.rstrip())
    if observation.artifacts:
        parts.append("[artifacts] " + format_artifact_paths(observation.artifacts))
    if observation.final_answer is not None:
        parts.append(f"[final] {observation.final_answer}")
        if observation.final_artifacts:
            parts.append("[final artifacts] " + ", ".join(observation.final_artifacts))
    parts.append("</observation>")
    return "\n".join(parts)


def format_artifact_paths(paths: list[str]) -> str:
    """Render artifact paths for model-facing prompt replay."""
    shown = paths[:MAX_RENDERED_ARTIFACT_PATHS]
    rendered = ", ".join(shown)
    omitted = len(paths) - len(shown)
    if omitted <= 0:
        return rendered
    return (
        f"{rendered}, ... "
        f"[showing {len(shown)} of {len(paths)} paths; {omitted} omitted. "
        "Use list_directory/read_file to inspect specific files.]"
    )


def format_workspace_iteration(
    iteration: WorkspaceIteration,
    *,
    action_format: str = "native",
) -> list[dict[str, Any]]:
    """Convert a completed ``WorkspaceIteration`` into next-turn messages.

    For ``action_format == "native"`` (default) we emit OpenAI Chat
    Completions-shaped messages: an ``assistant`` message carrying any
    parsed tool calls as structured ``tool_calls`` (not inline text), one
    ``role: "tool"`` message per observation keyed by ``tool_call_id``, and
    a trailing ``user`` message for the workspace snapshot summary and/or
    no-op nudge. This lets the backend chat template (e.g. Qwen3's
    ``<tool_call><function=...>`` XML) render the assistant's history in
    the model's *native* tool-call format. Rendering past tool calls as
    plain text (``TOOL_CALL t<N>.a<M> tool=... args=... body=...``) caused
    the model to mimic that text pattern on subsequent turns, producing
    prose that looks like a tool call but lives in ``message.content`` —
    the tool parser sees no XML and the substrate sees ``actions=[]``.

    For the deprecated ``action_format == "xml"`` path we still return a
    two-message list of ``assistant`` + ``user`` with the plain-text
    replay. Parse-retry attempts are NOT included here; they live in
    ``iteration.parse_attempts`` for the visualizer.

    The model's free-form pre-tool-call narration (``iteration.response``
    after stripping ``<think>...</think>`` and ``<|channel|>`` blocks) is
    placed in the assistant message's ``content`` field in both paths, so
    intent carries across turns.
    """
    narration = strip_reasoning_blocks(iteration.response).strip()

    if action_format == "native":
        return _format_workspace_iteration_native(iteration, narration)

    action_chunks: list[str] = []
    for idx, action in enumerate(iteration.actions):
        action_id = f"t{iteration.iteration}.a{idx + 1}"
        observation = iteration.observations[idx] if idx < len(iteration.observations) else None
        action_chunks.append(
            render_action_replay(
                action_id=action_id,
                action=action,
                observation=observation,
                action_format=action_format,
            )
        )

    obs_chunks: list[str] = []
    for idx, observation in enumerate(iteration.observations):
        action_id = f"t{iteration.iteration}.a{idx + 1}"
        obs_chunks.append(
            render_observation(
                action_id,
                observation,
                action_format=action_format,
            )
        )

    if iteration.snapshot is not None:
        snap = iteration.snapshot
        changed = ", ".join(snap.changed_files) if snap.changed_files else "(no changes)"
        obs_chunks.append(
            f'<snapshot turn="{snap.turn}" commit="{snap.commit_sha[:7]}">'
            f"\nchanged: {changed}\n</snapshot>"
        )

    if action_chunks:
        assistant_chunks = ([narration] if narration else []) + action_chunks
        assistant_content = "\n\n".join(assistant_chunks)
    else:
        assistant_content = narration

    if obs_chunks:
        user_message = "\n\n".join(obs_chunks)
    elif not action_chunks:
        user_message = "No tool calls made; workspace unchanged. State your next action."
    else:
        user_message = "(no observations)"

    return [
        {"role": "assistant", "content": assistant_content},
        {"role": "user", "content": user_message},
    ]


def _format_workspace_iteration_native(
    iteration: WorkspaceIteration,
    narration: str,
) -> list[dict[str, Any]]:
    """Native-tools rendering of a completed iteration.

    Produces ``assistant`` (with structured ``tool_calls``) + one ``tool``
    message per observation + an optional trailing ``user`` message for
    snapshot and/or no-op nudge.
    """
    tool_calls: list[dict[str, Any]] = []
    call_ids: list[str] = []
    for idx, action in enumerate(iteration.actions):
        action_id = f"t{iteration.iteration}.a{idx + 1}"
        call_id = action.call_id or action_id
        call_ids.append(call_id)
        tool_calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": action.tool,
                    "arguments": json.dumps(action.args, ensure_ascii=False),
                },
            }
        )

    assistant_msg: dict[str, Any] = {"role": "assistant", "content": narration}
    if tool_calls:
        assistant_msg["tool_calls"] = tool_calls
    messages: list[dict[str, Any]] = [assistant_msg]

    for idx, observation in enumerate(iteration.observations):
        action_id = f"t{iteration.iteration}.a{idx + 1}"
        call_id = call_ids[idx] if idx < len(call_ids) else action_id
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": _render_observation_native_body(observation),
            }
        )

    extras: list[str] = []
    if iteration.snapshot is not None:
        snap = iteration.snapshot
        changed = ", ".join(snap.changed_files) if snap.changed_files else "(no changes)"
        extras.append(
            f"SNAPSHOT turn={snap.turn} commit={snap.commit_sha[:7]}\nchanged: {changed}"
        )
    if not tool_calls:
        extras.append("No tool calls made; workspace unchanged. State your next action.")

    if extras:
        messages.append({"role": "user", "content": "\n\n".join(extras)})

    return messages


def _render_observation_native_body(observation: WorkspaceObservation) -> str:
    """Plain-text body for a ``role: "tool"`` message.

    No ``OBSERVATION t<N>.a<M>`` correlator header — the ``role: "tool"``
    envelope plus matching ``tool_call_id`` already provide correlation,
    and emitting the header invited the same format-mimicry bug the
    structured-replay refactor is meant to fix.
    """
    status = "error" if observation.error else "ok"
    parts = [f"status={status}"]
    if observation.error:
        parts.append(f"error: {observation.error}")
    if observation.stdout:
        parts.append(observation.stdout.rstrip())
    if observation.stderr:
        parts.append("stderr:")
        parts.append(observation.stderr.rstrip())
    if observation.artifacts:
        parts.append("artifacts: " + format_artifact_paths(observation.artifacts))
    if observation.final_answer is not None:
        parts.append(f"final: {observation.final_answer}")
        if observation.final_artifacts:
            parts.append("final artifacts: " + ", ".join(observation.final_artifacts))
    return "\n".join(parts)


def format_workspace_history(
    iterations: list[WorkspaceIteration],
    *,
    action_format: str = "native",
) -> list[dict[str, Any]]:
    """Render completed iterations for model-facing prompt replay."""
    messages: list[dict[str, Any]] = []
    for iteration in iterations:
        messages.extend(
            format_workspace_iteration(
                iteration,
                action_format=action_format,
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


def build_native_tool_retry_message(error: str, fragment: str | None) -> str:
    msg = (
        "Your previous response did not produce valid native tool calls: "
        f"{error}\n\n"
        "Reply again by calling one or more of the provided tools."
    )
    if fragment:
        msg += f"\n\nOffending tool-call payload (truncated):\n{fragment[:500]}"
    return msg


COMPACTION_SUMMARY_PROMPT = textwrap.dedent(
    """\
    The workspace trajectory so far has grown long enough that the substrate
    is about to compress it. Produce a single self-contained summary that
    your future self (with no other context) can resume from. The compressed
    history is irreversible — anything you do not capture here is lost from
    your visible context (durable files, git snapshots, and
    ``_rlm_state/provenance.json`` remain on disk).

    Structure your response exactly as follows, in plain prose with the
    section headers verbatim:

    # Original task
    Restate the original task **verbatim** from
    ``/_rlm_state/_rlm_query_0.txt`` (and any
    ``/_rlm_state/_rlm_query_<N>.txt`` context files). Do not paraphrase.

    # Files touched
    For each workspace file you wrote, appended to, or edited, give one
    line: ``<path> — <one-line description of its current contents/role>``.
    Pull this from the provenance information that follows.

    # Concrete results to preserve
    Any intermediate values, computations, code snippets, or findings that
    would be expensive to recompute. Be specific (numbers, file:line
    references, exact strings) rather than abstract.

    # Open questions / uncertainties
    Anything you noticed but have not resolved.

    # Next action
    The single next concrete step you would take if resumed right now.

    Do NOT emit any tool calls or ``<action>`` blocks in this summary —
    only prose. The substrate will resume normal tool-calling after the
    compression boundary.
    """
)


def build_compaction_summary_prompt(
    *,
    provenance_lines: list[str] | None = None,
) -> str:
    """Compose the user-facing summary request that drives a compaction call.

    ``provenance_lines`` is an optional pre-rendered list of
    ``<path> — <provenance role/turn>`` lines pulled from
    ``DockerWorkspaceEnv.provenance``. Including them gives the model a
    deterministic checklist of the files to mention in the "Files touched"
    section.
    """
    prompt = COMPACTION_SUMMARY_PROMPT
    if provenance_lines:
        rendered = "\n".join(f"- {line}" for line in provenance_lines)
        prompt += "\n\n# Provenance snapshot (for your reference)\n" + rendered
    return prompt


def build_compaction_continue_message() -> str:
    """User-facing kick-off message replayed immediately after the summary.

    Mirrors upstream RLM's [system, initial, summary, continue] shape.
    """
    return (
        "The trajectory above has been compressed into the assistant summary. "
        "The workspace filesystem is unchanged and remains authoritative. "
        "Resume the task from the 'Next action' section of your summary; "
        "re-read ``/_rlm_state/_rlm_query_0.txt`` or your ``/_rlm_notes/`` "
        "if you need to confirm any detail."
    )
