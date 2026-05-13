"""edit — exact literal replacement inside a workspace file."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from rlm.core.types import WorkspaceAction, WorkspaceObservation
from rlm.workspace_tools import ToolSpec

if TYPE_CHECKING:
    from rlm.environments.docker_workspace import DockerWorkspaceEnv

SPEC = ToolSpec(
    name="edit",
    short_description=(
        "Replace an exact literal old_string in file_path with new_string. "
        "Whitespace, indentation, and newlines must match exactly. By default "
        "old_string must occur once; set replace_all=true to replace every occurrence."
    ),
    is_state_mutating=True,
    runs_on="host",
    body_required=False,
    example=(
        'Native args: {"file_path": "src/foo.py", "old_string": "old\\n", '
        '"new_string": "new\\n", "replace_all": false}'
    ),
)


def execute(env: DockerWorkspaceEnv, action: WorkspaceAction) -> WorkspaceObservation:
    start = time.perf_counter()
    rel = action.args.get("file_path") or action.args.get("path")
    if not rel:
        return WorkspaceObservation(
            tool=SPEC.name,
            error="Missing required argument 'file_path'.",
            execution_time=time.perf_counter() - start,
        )
    if env.is_reserved_path(str(rel)):
        return WorkspaceObservation(
            tool=SPEC.name,
            error=f"Refusing to write reserved path: {rel}",
            execution_time=time.perf_counter() - start,
        )
    if "old_string" not in action.args or "new_string" not in action.args:
        return WorkspaceObservation(
            tool=SPEC.name,
            error="Missing required arguments 'old_string' and/or 'new_string'.",
            execution_time=time.perf_counter() - start,
        )

    old = str(action.args["old_string"])
    new = str(action.args["new_string"])
    replace_all = bool(action.args.get("replace_all", False))

    target = env.resolve_workspace_path(str(rel))
    if not target.exists():
        return WorkspaceObservation(
            tool=SPEC.name,
            error=f"File does not exist: {rel}",
            execution_time=time.perf_counter() - start,
        )
    try:
        original_bytes = target.read_bytes()
        had_bom = original_bytes.startswith(b"\xef\xbb\xbf")
        original = original_bytes.decode("utf-8-sig")
    except (OSError, UnicodeDecodeError) as e:
        return WorkspaceObservation(
            tool=SPEC.name,
            error=f"Failed to read {rel} as UTF-8 text: {e}",
            execution_time=time.perf_counter() - start,
        )

    newline = "\r\n" if "\r\n" in original and "\n" not in original.replace("\r\n", "") else "\n"
    normalized_original = original.replace("\r\n", "\n")
    normalized_old = old.replace("\r\n", "\n")
    normalized_new = new.replace("\r\n", "\n")
    occurrences = normalized_original.count(normalized_old)
    if occurrences == 0:
        return WorkspaceObservation(
            tool=SPEC.name,
            error=(
                f"old_string was not found in {rel}. Match whitespace, indentation, "
                "and newlines exactly, or read the file before retrying."
            ),
            execution_time=time.perf_counter() - start,
        )
    if occurrences > 1 and not replace_all:
        return WorkspaceObservation(
            tool=SPEC.name,
            error=(
                f"old_string matches {occurrences} times in {rel}. Set replace_all=true "
                "or provide more surrounding context."
            ),
            execution_time=time.perf_counter() - start,
        )

    count = occurrences if replace_all else 1
    edited = normalized_original.replace(normalized_old, normalized_new, count)
    if newline == "\r\n":
        edited = edited.replace("\n", "\r\n")
    data = edited.encode("utf-8")
    if had_bom:
        data = b"\xef\xbb\xbf" + data
    try:
        target.write_bytes(data)
    except OSError as e:
        return WorkspaceObservation(
            tool=SPEC.name,
            error=f"Failed to write {rel}: {e}",
            execution_time=time.perf_counter() - start,
        )

    env.provenance.record_write(
        str(rel), role="assistant", action_id=env.current_action_id, turn=env.current_turn
    )
    return WorkspaceObservation(
        tool=SPEC.name,
        stdout=f"Edited {rel}: replaced {count} occurrence(s).",
        data={"path": str(rel), "occurrences": count},
        artifacts=[str(rel)],
        execution_time=time.perf_counter() - start,
    )
