"""write_file — overwrite or create a workspace file with the body content."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from rlm.core.types import WorkspaceAction, WorkspaceObservation
from rlm.workspace_tools import ToolSpec

if TYPE_CHECKING:
    from rlm.environments.docker_workspace import DockerWorkspaceEnv

SPEC = ToolSpec(
    name="write_file",
    short_description="Create or completely overwrite a file with the action body.",
    is_state_mutating=True,
    runs_on="host",
    body_required=True,
    example=(
        'Required attr: path (workspace-relative; absolute paths rejected). '
        'Body is the raw file contents — do NOT wrap in '
        '<code>, <body>, markdown fences, or any other tag. '
        'Example: <action tool="write_file" path="hello.py">\nprint("hello")\n</action>'
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
        target.write_text(body, encoding="utf-8")
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
        stdout=f"Wrote {len(body)} chars to {rel}",
        data={"path": rel, "bytes": len(body.encode("utf-8"))},
        artifacts=[rel],
        execution_time=time.perf_counter() - start,
    )
