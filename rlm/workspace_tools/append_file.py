"""append_file — append to a workspace file (creating it if absent)."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from rlm.core.types import WorkspaceAction, WorkspaceObservation
from rlm.workspace_tools import ToolSpec

if TYPE_CHECKING:
    from rlm.environments.docker_workspace import DockerWorkspaceEnv

SPEC = ToolSpec(
    name="append_file",
    short_description="Append the action body to the end of a file (creates it if missing).",
    is_state_mutating=True,
    runs_on="host",
    body_required=True,
    example=(
        'Required attr: path (workspace-relative; absolute paths rejected). '
        'Body is appended verbatim — no wrappers. '
        'Example: <action tool="append_file" path="log.txt">\nnew entry\n</action>'
    ),
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
    target = env.resolve_workspace_path(rel)
    body = action.body if action.body is not None else ""
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as f:
            f.write(body)
    except OSError as e:
        return WorkspaceObservation(
            tool=SPEC.name,
            error=f"Failed to append {rel}: {e}",
            execution_time=time.perf_counter() - start,
        )

    env.provenance.record_write(
        rel, role="assistant", action_id=env.current_action_id, turn=env.current_turn
    )
    return WorkspaceObservation(
        tool=SPEC.name,
        stdout=f"Appended {len(body)} chars to {rel}",
        data={"path": rel, "appended_bytes": len(body.encode("utf-8"))},
        artifacts=[rel],
        execution_time=time.perf_counter() - start,
    )
