"""edit_file — substring search/replace inside a workspace file."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from rlm.core.types import WorkspaceAction, WorkspaceObservation
from rlm.utils.action_parser import parse_edit_file_body
from rlm.utils.exceptions import ActionParseError
from rlm.workspace_tools import ToolSpec

if TYPE_CHECKING:
    from rlm.environments.docker_workspace import DockerWorkspaceEnv

SPEC = ToolSpec(
    name="edit_file",
    short_description=(
        "Replace a unique substring in a file. Body: <search>...</search><replace>...</replace>. "
        "Set allow_multiple=true to replace all occurrences."
    ),
    is_state_mutating=True,
    runs_on="host",
    body_required=True,
)


def execute(env: DockerWorkspaceEnv, action: WorkspaceAction) -> WorkspaceObservation:
    start = time.perf_counter()
    rel = action.args.get("path")
    if not rel:
        return WorkspaceObservation(
            tool=SPEC.name,
            error="Missing required attribute 'path'.",
            execution_time=time.perf_counter() - start,
        )
    if env.is_reserved_path(rel):
        return WorkspaceObservation(
            tool=SPEC.name,
            error=f"Refusing to write reserved path: {rel}",
            execution_time=time.perf_counter() - start,
        )

    allow_multiple = action.args.get("allow_multiple", "false").lower() == "true"

    try:
        search_text, replace_text = parse_edit_file_body(action.body or "")
    except ActionParseError as e:
        return WorkspaceObservation(
            tool=SPEC.name,
            error=str(e),
            execution_time=time.perf_counter() - start,
        )

    target = env.resolve_workspace_path(rel)
    if not target.exists():
        return WorkspaceObservation(
            tool=SPEC.name,
            error=f"File does not exist: {rel}",
            execution_time=time.perf_counter() - start,
        )
    try:
        original = target.read_text(encoding="utf-8")
    except OSError as e:
        return WorkspaceObservation(
            tool=SPEC.name,
            error=f"Failed to read {rel}: {e}",
            execution_time=time.perf_counter() - start,
        )

    occurrences = original.count(search_text)
    if occurrences == 0:
        return WorkspaceObservation(
            tool=SPEC.name,
            error=f"Search text not found in {rel}.",
            execution_time=time.perf_counter() - start,
        )
    if occurrences > 1 and not allow_multiple:
        return WorkspaceObservation(
            tool=SPEC.name,
            error=(
                f"Search text matches {occurrences} times in {rel}. "
                'Set allow_multiple="true" to replace all, or refine the search.'
            ),
            execution_time=time.perf_counter() - start,
        )
    new_text = original.replace(search_text, replace_text)

    try:
        target.write_text(new_text, encoding="utf-8")
    except OSError as e:
        return WorkspaceObservation(
            tool=SPEC.name,
            error=f"Failed to write {rel}: {e}",
            execution_time=time.perf_counter() - start,
        )

    env.provenance.record_write(
        rel, role="assistant", action_id=env.current_action_id, turn=env.current_turn
    )
    return WorkspaceObservation(
        tool=SPEC.name,
        stdout=f"Edited {rel}: replaced {occurrences} occurrence(s).",
        data={"path": rel, "occurrences": occurrences},
        artifacts=[rel],
        execution_time=time.perf_counter() - start,
    )
