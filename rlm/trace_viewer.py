"""Text viewer for workspace-substrate JSONL traces."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MAX_CHARS = 2000


@dataclass
class TraceLog:
    path: Path
    metadata: dict[str, Any] | None
    iterations: list[dict[str, Any]]
    compactions: list[dict[str, Any]]


def load_trace(path: str | Path) -> TraceLog:
    """Load one RLM JSONL trace file."""
    trace_path = Path(path)
    metadata: dict[str, Any] | None = None
    iterations: list[dict[str, Any]] = []
    compactions: list[dict[str, Any]] = []

    with trace_path.open() as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{trace_path}:{line_number}: invalid JSON: {exc}") from exc

            row_type = row.get("type")
            if row_type == "metadata":
                metadata = row
            elif row_type == "iteration":
                iterations.append(row)
            elif row_type == "compaction":
                compactions.append(row)

    return TraceLog(
        path=trace_path,
        metadata=metadata,
        iterations=iterations,
        compactions=compactions,
    )


def render_summary(trace: TraceLog, *, include_children: bool = False) -> str:
    lines: list[str] = [f"=== {trace.path} ==="]
    meta = trace.metadata or {}
    root_model = meta.get("root_model") or "unknown"
    backend = meta.get("backend") or "unknown"
    max_iterations = _format_value(meta.get("max_iterations"))
    max_depth = _format_value(meta.get("max_depth"))
    action_format = meta.get("action_format") or "unknown"

    lines.append(f"backend       : {backend} model={root_model}")
    lines.append(f"max_iter/depth: {max_iterations}/{max_depth} action_format={action_format}")
    lines.append(f"iterations    : {len(trace.iterations)}")
    if trace.compactions:
        lines.append(f"compactions   : {len(trace.compactions)}")
    lines.append("")

    for iteration in trace.iterations:
        lines.append(_render_iteration_summary(iteration, include_children=include_children))
        first_error = _first_error(iteration)
        if first_error:
            lines.append(f"      [error: {first_error[0]}] {_single_line(first_error[1], 160)}")
        if include_children:
            lines.extend(_render_child_summary_lines(iteration, indent="      "))

    return "\n".join(lines).rstrip()


def render_turn(
    trace: TraceLog,
    turn: int,
    *,
    sections: set[str] | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    include_children: bool = False,
) -> str:
    iteration = _find_iteration(trace, turn)
    if iteration is None:
        raise ValueError(f"{trace.path}: no iteration {turn}")

    requested = sections or {"overview", "actions", "observations"}
    lines: list[str] = [f"=== {trace.path} :: turn {turn} ==="]

    if "overview" in requested:
        lines.append(_render_iteration_summary(iteration, include_children=include_children))
        snapshot = iteration.get("snapshot") or {}
        if snapshot:
            lines.append(f"commit        : {snapshot.get('commit_sha')}")
            workspace_root = snapshot.get("workspace_root")
            if workspace_root:
                lines.append(f"workspace     : {workspace_root}")
        if iteration.get("error"):
            lines.append(f"turn_error    : {_single_line(iteration.get('error'), 240)}")
        lines.append("")

    if "prompt" in requested:
        lines.append("PROMPT")
        for idx, message in enumerate(iteration.get("prompt") or [], start=1):
            role = message.get("role", "?")
            content = _truncate(str(message.get("content", "")), max_chars)
            lines.append(_block(f"[{idx}] {role}", content))

    if "response" in requested:
        lines.append("RESPONSE")
        lines.append(
            _block("assistant", _truncate(str(iteration.get("response") or ""), max_chars))
        )

    if "reasoning" in requested and iteration.get("reasoning"):
        lines.append("REASONING")
        lines.append(
            _block("reasoning", _truncate(str(iteration.get("reasoning") or ""), max_chars))
        )

    if "parse" in requested:
        lines.append("PARSE ATTEMPTS")
        attempts = iteration.get("parse_attempts") or []
        if not attempts:
            lines.append("  none")
        for idx, attempt in enumerate(attempts, start=1):
            error = _single_line(attempt.get("error"), 240)
            lines.append(f"  [{idx}] error: {error}")
            lines.append(
                _block("response", _truncate(str(attempt.get("response") or ""), max_chars))
            )

    if "actions" in requested:
        lines.append("ACTIONS")
        actions = iteration.get("actions") or []
        if not actions:
            lines.append("  none")
        for idx, action in enumerate(actions, start=1):
            lines.extend(_render_action(idx, action, max_chars=max_chars))

    if "observations" in requested:
        lines.append("OBSERVATIONS")
        observations = iteration.get("observations") or []
        if not observations:
            lines.append("  none")
        for idx, observation in enumerate(observations, start=1):
            lines.extend(
                _render_observation(
                    idx,
                    observation,
                    max_chars=max_chars,
                    include_children=include_children,
                )
            )

    if "snapshot" in requested:
        lines.append("SNAPSHOT")
        lines.extend(_render_snapshot(iteration.get("snapshot") or {}))

    return "\n".join(lines).rstrip()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="rlm-trace",
        description="Inspect RLM workspace-substrate JSONL traces from the terminal.",
    )
    parser.add_argument("paths", nargs="+", help="Trace JSONL files to inspect.")
    parser.add_argument(
        "--turn",
        type=int,
        action="append",
        default=[],
        help="Show detailed output for this turn. May be repeated.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="With --turn, show prompt, response, reasoning, parse attempts, actions, observations, and snapshot.",
    )
    parser.add_argument(
        "--prompt", action="store_true", help="With --turn, include prompt messages."
    )
    parser.add_argument(
        "--response", action="store_true", help="With --turn, include raw response."
    )
    parser.add_argument(
        "--reasoning", action="store_true", help="With --turn, include reasoning content."
    )
    parser.add_argument(
        "--parse-attempts",
        action="store_true",
        help="With --turn, include parse retry responses.",
    )
    parser.add_argument("--actions", action="store_true", help="With --turn, include actions.")
    parser.add_argument(
        "--observations",
        action="store_true",
        help="With --turn, include observations.",
    )
    parser.add_argument(
        "--snapshot", action="store_true", help="With --turn, include snapshot files."
    )
    parser.add_argument(
        "--children",
        action="store_true",
        help="Show nested RLM call counts and child trajectory summaries.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help="Maximum characters per long text block; use 0 for full text.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sections = _requested_sections(args)
    rendered: list[str] = []

    try:
        for path in args.paths:
            trace = load_trace(path)
            if args.turn:
                for turn in args.turn:
                    rendered.append(
                        render_turn(
                            trace,
                            turn,
                            sections=sections,
                            max_chars=args.max_chars,
                            include_children=args.children,
                        )
                    )
            else:
                rendered.append(render_summary(trace, include_children=args.children))
    except (OSError, ValueError) as exc:
        print(f"rlm-trace: {exc}", file=sys.stderr)
        return 1

    print("\n\n".join(rendered))
    return 0


def _requested_sections(args: argparse.Namespace) -> set[str] | None:
    if args.all:
        return {
            "overview",
            "prompt",
            "response",
            "reasoning",
            "parse",
            "actions",
            "observations",
            "snapshot",
        }

    sections = {"overview"}
    if args.prompt:
        sections.add("prompt")
    if args.response:
        sections.add("response")
    if args.reasoning:
        sections.add("reasoning")
    if args.parse_attempts:
        sections.add("parse")
    if args.actions:
        sections.add("actions")
    if args.observations:
        sections.add("observations")
    if args.snapshot:
        sections.add("snapshot")
    if sections == {"overview"}:
        return None
    return sections


def _find_iteration(trace: TraceLog, turn: int) -> dict[str, Any] | None:
    for iteration in trace.iterations:
        if iteration.get("iteration") == turn:
            return iteration
    return None


def _render_iteration_summary(iteration: dict[str, Any], *, include_children: bool) -> str:
    turn = int(iteration.get("iteration") or 0)
    elapsed = iteration.get("iteration_time")
    elapsed_text = f"{float(elapsed):.1f}s" if isinstance(elapsed, int | float) else "?.?s"
    tools = [str(action.get("tool", "?")) for action in iteration.get("actions") or []]
    error_count = sum(1 for obs in iteration.get("observations") or [] if obs.get("error"))
    if iteration.get("error"):
        error_count += 1
    parse_retries = len(iteration.get("parse_attempts") or [])
    spills = _spill_count(iteration)
    changed_files = ((iteration.get("snapshot") or {}).get("changed_files") or [])[:]
    final = iteration.get("final_answer")
    child_count = _child_call_count(iteration)

    markers = [
        f"turn {turn:02d}",
        f"({elapsed_text})",
        f"tools=[{','.join(tools)}]",
    ]
    if parse_retries:
        markers.append(f"parse_retries={parse_retries}")
    if error_count:
        markers.append(f"ERR={error_count}")
    if spills:
        markers.append(f"spills={spills}")
    if include_children and child_count:
        markers.append(f"children={child_count}")
    if changed_files:
        markers.append(f"changed={_format_changed_files(changed_files)}")
    if final is not None:
        markers.append("FINAL")
    return "  " + "  ".join(markers)


def _render_action(idx: int, action: dict[str, Any], *, max_chars: int) -> list[str]:
    tool = action.get("tool") or "?"
    args = action.get("args") or {}
    lines = [f"  [{idx}] {tool} args={_json_compact(args)}"]
    body = action.get("body")
    if body is not None:
        lines.append(_block("body", _truncate(str(body), max_chars), indent="      "))
    return lines


def _render_observation(
    idx: int,
    observation: dict[str, Any],
    *,
    max_chars: int,
    include_children: bool,
) -> list[str]:
    tool = observation.get("tool") or "?"
    elapsed = observation.get("execution_time")
    elapsed_text = f" ({float(elapsed):.2f}s)" if isinstance(elapsed, int | float) else ""
    error = observation.get("error")
    lines = [f"  [{idx}] {tool}{elapsed_text}"]
    if error:
        lines.append(f"      error: {_single_line(error, 240)}")
    if observation.get("stdout"):
        lines.append(
            _block("stdout", _truncate(str(observation["stdout"]), max_chars), indent="      ")
        )
    if observation.get("stderr"):
        lines.append(
            _block("stderr", _truncate(str(observation["stderr"]), max_chars), indent="      ")
        )
    if observation.get("artifacts"):
        lines.append(f"      artifacts: {', '.join(map(str, observation['artifacts']))}")
    if observation.get("final_answer") is not None:
        lines.append(
            _block("final", _truncate(str(observation["final_answer"]), max_chars), indent="      ")
        )

    rlm_calls = observation.get("rlm_calls") or []
    if rlm_calls:
        lines.append(f"      rlm_calls: {len(rlm_calls)}")
        if include_children:
            for call_idx, call in enumerate(rlm_calls, start=1):
                lines.extend(_render_child_call(call_idx, call, indent="        "))
    return lines


def _render_snapshot(snapshot: dict[str, Any]) -> list[str]:
    if not snapshot:
        return ["  none"]
    lines = [
        f"  turn          : {snapshot.get('turn')}",
        f"  commit        : {snapshot.get('commit_sha')}",
        f"  workspace     : {snapshot.get('workspace_root')}",
        "  changed_files :",
    ]
    changed_files = snapshot.get("changed_files") or []
    if not changed_files:
        lines.append("    none")
    for path in changed_files:
        lines.append(f"    {path}")
    return lines


def _render_child_summary_lines(iteration: dict[str, Any], *, indent: str) -> list[str]:
    lines: list[str] = []
    for obs_idx, observation in enumerate(iteration.get("observations") or [], start=1):
        for call_idx, call in enumerate(observation.get("rlm_calls") or [], start=1):
            prefix = f"{indent}obs {obs_idx} child {call_idx}: "
            lines.extend(_render_child_call(call_idx, call, indent=indent, prefix=prefix))
    return lines


def _render_child_call(
    call_idx: int,
    call: dict[str, Any],
    *,
    indent: str,
    prefix: str | None = None,
) -> list[str]:
    metadata = call.get("metadata") or {}
    iterations = metadata.get("iterations") or []
    final = _child_final_answer(iterations) or call.get("response")
    execution_time = call.get("execution_time")
    elapsed = f"{float(execution_time):.1f}s" if isinstance(execution_time, int | float) else "?.?s"
    root_model = call.get("root_model") or "unknown"
    first_line = prefix or f"{indent}child {call_idx}: "
    lines = [
        f"{first_line}model={root_model} iterations={len(iterations)} time={elapsed} final={_single_line(final, 120)}"
    ]
    for child_iteration in iterations:
        lines.append(
            indent + _render_iteration_summary(child_iteration, include_children=False).strip()
        )
    return lines


def _child_final_answer(iterations: list[dict[str, Any]]) -> str | None:
    for iteration in reversed(iterations):
        final = iteration.get("final_answer")
        if final is not None:
            return str(final)
    return None


def _first_error(iteration: dict[str, Any]) -> tuple[str, str] | None:
    if iteration.get("error"):
        return "turn", str(iteration["error"])
    for observation in iteration.get("observations") or []:
        if observation.get("error"):
            return str(observation.get("tool") or "?"), str(observation["error"])
    return None


def _spill_count(iteration: dict[str, Any]) -> int:
    count = 0
    for observation in iteration.get("observations") or []:
        text = "\n".join(
            str(observation.get(key) or "") for key in ("stdout", "stderr") if observation.get(key)
        ).lower()
        if "spilled" in text and "_rlm_artifacts/_observations" in text:
            count += 1
    return count


def _child_call_count(iteration: dict[str, Any]) -> int:
    return sum(
        len(observation.get("rlm_calls") or [])
        for observation in iteration.get("observations") or []
    )


def _format_changed_files(changed_files: list[Any]) -> str:
    shown = [str(path) for path in changed_files[:3]]
    if len(changed_files) > 3:
        shown.append("...")
    return "[" + ",".join(shown) + "]"


def _format_value(value: Any) -> str:
    if value is None:
        return "?"
    return str(value)


def _json_compact(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"{text[:max_chars]}\n... <truncated {omitted} chars; pass --max-chars 0 for full text>"


def _single_line(value: Any, max_chars: int) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _block(label: str, text: str, *, indent: str = "  ") -> str:
    if not text:
        return f"{indent}{label}: <empty>"
    lines = text.splitlines() or [""]
    rendered = [f"{indent}{label}:"]
    rendered.extend(f"{indent}  {line}" for line in lines)
    return "\n".join(rendered)


if __name__ == "__main__":
    raise SystemExit(main())
