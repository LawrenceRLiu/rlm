"""python — execute the action body as a Python script inside the workspace container."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from rlm.core.types import WorkspaceAction, WorkspaceObservation
from rlm.workspace_tools import ToolSpec

if TYPE_CHECKING:
    from rlm.environments.docker_workspace import DockerWorkspaceEnv

SPEC = ToolSpec(
    name="python",
    short_description=(
        "Run a Python script inside the workspace container. "
        "`llm_query`, `llm_query_batched`, `rlm_query`, `rlm_query_batched` are pre-imported."
    ),
    is_state_mutating=True,
    runs_on="container",
    body_required=True,
)

# Wrapper: makes the in-container client helpers available without imports in
# the user script. The `from rlm_workspace.client import *` line picks up
# llm_query / llm_query_batched / rlm_query / rlm_query_batched.
_WRAPPER_PREAMBLE = (
    "from rlm_workspace.client import llm_query, llm_query_batched, rlm_query, rlm_query_batched\n"
)


def execute(env: DockerWorkspaceEnv, action: WorkspaceAction) -> WorkspaceObservation:
    start = time.perf_counter()
    body = action.body or ""
    script = _WRAPPER_PREAMBLE + body

    excludes = env.workspace_config.recursion.copy_on_spawn_excludes
    before = env.snapshot_paths_for_provenance(excludes)

    result = env.exec_in_container(
        ["python", "-c", script],
        timeout=env.workspace_config.docker.exec_timeout_seconds,
    )

    after = env.snapshot_paths_for_provenance(excludes)
    changed, removed = env.diff_paths_for_provenance(before, after)
    env.provenance.record_writes(
        changed, role="system", action_id=env.current_action_id, turn=env.current_turn
    )
    for path in removed:
        env.provenance.remove(path)

    obs = WorkspaceObservation(
        tool=SPEC.name,
        stdout=result.stdout,
        stderr=result.stderr,
        data={"exit_code": result.exit_code, "changed_paths": changed, "removed_paths": removed},
        artifacts=changed,
        execution_time=time.perf_counter() - start,
    )
    if result.timed_out:
        obs.error = f"exec timeout after {env.workspace_config.docker.exec_timeout_seconds}s"
    elif result.exit_code != 0:
        obs.error = f"python exited with code {result.exit_code}"
    return obs
