"""shell — execute the action body as a bash script inside the workspace container."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from rlm.core.types import WorkspaceAction, WorkspaceObservation
from rlm.workspace_tools import ToolSpec

if TYPE_CHECKING:
    from rlm.environments.docker_workspace import DockerWorkspaceEnv

SPEC = ToolSpec(
    name="shell",
    short_description="Run a bash script inside the workspace container.",
    is_state_mutating=True,
    runs_on="container",
    body_required=True,
)

# Tempfile location relative to workspace root. Lives under _rlm_state so it
# is excluded from provenance diffing.
_TMP_REL_DIR = "_rlm_state/_tmp"


def execute(env: DockerWorkspaceEnv, action: WorkspaceAction) -> WorkspaceObservation:
    start = time.perf_counter()
    body = action.body or ""

    action_id = env.current_action_id or "unknown"
    rel_tmp = f"{_TMP_REL_DIR}/shell_{action_id}.sh"
    tmp_path = env.workspace_root / rel_tmp
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_text(body, encoding="utf-8")

    excludes = env.workspace_config.recursion.copy_on_spawn_excludes
    before = env.snapshot_paths_for_provenance(excludes)

    result = env.exec_in_container(
        ["bash", f"/workspace/{rel_tmp}"],
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
        obs.error = f"shell exited with code {result.exit_code}"
    return obs
