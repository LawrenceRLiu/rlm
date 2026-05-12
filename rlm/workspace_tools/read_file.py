"""read_file — read a slice of a workspace file."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from rlm.core.types import WorkspaceAction, WorkspaceObservation
from rlm.workspace_tools import ToolSpec

if TYPE_CHECKING:
    from rlm.environments.docker_workspace import DockerWorkspaceEnv

SPEC = ToolSpec(
    name="read_file",
    short_description="Read a slice of a workspace file (lines start_line..end_line).",
    is_state_mutating=False,
    runs_on="host",
    body_required=False,
    example=(
        'Required attr: path (workspace-relative; absolute paths rejected). '
        'Optional: start_line, end_line. '
        'Example: <action tool="read_file" path="notes.md" start_line="1" end_line="50"/>'
    ),
)


def _is_binary(sample: bytes) -> bool:
    if b"\x00" in sample:
        return True
    # Heuristic: high ratio of non-printable bytes implies binary.
    if not sample:
        return False
    text_chars = bytes(range(0x20, 0x7F)) + b"\n\r\t\b\f"
    nontext = sum(1 for b in sample if b not in text_chars)
    return (nontext / len(sample)) > 0.30


def execute(env: DockerWorkspaceEnv, action: WorkspaceAction) -> WorkspaceObservation:
    start = time.perf_counter()
    rel = action.args.get("path")
    if not rel:
        return WorkspaceObservation(
            tool=SPEC.name,
            error="Missing required attribute 'path'.",
            execution_time=time.perf_counter() - start,
        )
    target = env.resolve_workspace_path(rel)
    if not target.exists():
        return WorkspaceObservation(
            tool=SPEC.name,
            error=f"File does not exist: {rel}",
            execution_time=time.perf_counter() - start,
        )
    if not target.is_file():
        return WorkspaceObservation(
            tool=SPEC.name,
            error=f"Not a regular file: {rel}",
            execution_time=time.perf_counter() - start,
        )

    try:
        with target.open("rb") as f:
            sample = f.read(4096)
        if _is_binary(sample):
            return WorkspaceObservation(
                tool=SPEC.name,
                error=f"Refusing to read binary file: {rel}",
                execution_time=time.perf_counter() - start,
            )
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return WorkspaceObservation(
            tool=SPEC.name,
            error=f"Failed to read {rel}: {e}",
            execution_time=time.perf_counter() - start,
        )

    lines = text.splitlines()
    total = len(lines)

    default_span = env.workspace_config.observation.default_read_file_lines
    try:
        start_line = int(action.args.get("start_line", 1))
    except ValueError:
        return WorkspaceObservation(
            tool=SPEC.name,
            error="start_line must be an integer.",
            execution_time=time.perf_counter() - start,
        )
    try:
        end_line = int(action.args.get("end_line", start_line + default_span - 1))
    except ValueError:
        return WorkspaceObservation(
            tool=SPEC.name,
            error="end_line must be an integer.",
            execution_time=time.perf_counter() - start,
        )

    start_line = max(1, start_line)
    end_line = min(total if total > 0 else start_line, end_line)
    if total == 0:
        slice_text = ""
    else:
        slice_text = "\n".join(lines[start_line - 1 : end_line])

    prov = env.provenance.get(rel)
    created = prov.created.role if prov else "user"
    modified = prov.modified.role if prov else "user"
    header = (
        f"[File: {rel} | Lines: {start_line}-{end_line}/{total} | "
        f"Created: {created} | Modified: {modified}]"
    )
    return WorkspaceObservation(
        tool=SPEC.name,
        stdout=f"{header}\n{slice_text}",
        data={
            "path": rel,
            "start_line": start_line,
            "end_line": end_line,
            "total_lines": total,
        },
        execution_time=time.perf_counter() - start,
    )
