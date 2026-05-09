"""rlm_query — spawn a child RLM with a copy-on-spawn snapshot of the workspace.

Stub for Phase 1: full implementation lives in Phase 6. The handler is wired
into the environment by ``DockerWorkspaceEnv``, which is what actually has the
parent reference; this module just validates and forwards.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from rlm.core.types import WorkspaceAction, WorkspaceObservation
from rlm.workspace_tools import ToolSpec

if TYPE_CHECKING:
    from rlm.environments.docker_workspace import DockerWorkspaceEnv

SPEC = ToolSpec(
    name="rlm_query",
    short_description=(
        "Spawn a child RLM with a copy-on-spawn snapshot of the workspace. "
        "Body is the child task. Child returns answer + selected artifacts."
    ),
    is_state_mutating=True,  # mutating because it copies artifacts back into the workspace
    runs_on="host",
    body_required=True,
)


def execute(env: DockerWorkspaceEnv, action: WorkspaceAction) -> WorkspaceObservation:
    start = time.perf_counter()
    body = action.body or ""

    if env.recursion_handler is None:
        return WorkspaceObservation(
            tool=SPEC.name,
            error=("Maximum recursion depth reached. The 'rlm_query' tool is unavailable."),
            execution_time=time.perf_counter() - start,
        )
    return env.recursion_handler.spawn(child_task=body, action_id=env.current_action_id)
