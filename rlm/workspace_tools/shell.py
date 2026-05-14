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
    short_description=(
        "Run shell commands for inspection, tests, build steps, and diagnostics. "
        "Default working dir is the workspace root (``/``); the bind-mounted "
        "task and substrate dirs (``/app``, ``/_rlm_state``, ``/_rlm_artifacts``, "
        "``/_rlm_notes``) are visible directly underneath. Pass ``directory=`` "
        "to override the cwd. "
        "Use file tools for ordinary durable edits; avoid echo/tee/heredoc file writes."
    ),
    is_state_mutating=True,
    runs_on="container",
    body_required=True,
    example=(
        "No attributes required. Body is raw bash — do NOT wrap in <code>, "
        "<script>, or markdown fences. "
        'Example: <action tool="shell">\nls -la /app\n</action>'
    ),
)

# Tempfile location relative to workspace root. Lives under _rlm_state so it
# is excluded from provenance diffing. Container sees it at /_rlm_state/_tmp/.
_TMP_REL_DIR = "_rlm_state/_tmp"


def execute(env: DockerWorkspaceEnv, action: WorkspaceAction) -> WorkspaceObservation:
    start = time.perf_counter()
    body = action.body if action.body is not None else str(action.args.get("command", ""))
    timeout = env.workspace_config.docker.exec_timeout_seconds
    if action.args.get("timeout") not in (None, ""):
        try:
            timeout = int(action.args["timeout"])
        except (TypeError, ValueError):
            return WorkspaceObservation(
                tool=action.tool,
                error="timeout must be an integer number of seconds.",
                execution_time=time.perf_counter() - start,
            )
    if str(action.args.get("is_background", "false")).lower() == "true":
        return WorkspaceObservation(
            tool=action.tool,
            error="Background shell commands are not supported yet; set is_background=false.",
            execution_time=time.perf_counter() - start,
        )
    # cwd: accept either ``directory`` (legacy) or ``cwd``; default = docker
    # container_cwd, which is normally the workspace root (/).
    cwd_arg = action.args.get("cwd") or action.args.get("directory")
    container_cwd: str | None = None
    if cwd_arg not in (None, ""):
        try:
            host_workdir = env.resolve_workspace_path(str(cwd_arg))
        except ValueError as e:
            return WorkspaceObservation(
                tool=action.tool,
                error=str(e),
                execution_time=time.perf_counter() - start,
            )
        if not host_workdir.exists() or not host_workdir.is_dir():
            return WorkspaceObservation(
                tool=action.tool,
                error=f"directory does not exist or is not a directory: {cwd_arg}",
                execution_time=time.perf_counter() - start,
            )
        try:
            container_cwd = env.host_to_container_path(host_workdir)
        except ValueError as e:
            return WorkspaceObservation(
                tool=action.tool,
                error=f"directory is not visible inside the container: {e}",
                execution_time=time.perf_counter() - start,
            )

    action_id = env.current_action_id or "unknown"
    rel_tmp = f"{_TMP_REL_DIR}/shell_{action_id}.sh"
    tmp_path = env.workspace_root / rel_tmp
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_text(body, encoding="utf-8")

    excludes = env.workspace_config.recursion.copy_on_spawn_excludes
    before = env.snapshot_paths_for_provenance(excludes)

    result = env.exec_in_container(
        ["bash", f"/{rel_tmp}"],
        timeout=timeout,
        cwd=container_cwd,
    )

    after = env.snapshot_paths_for_provenance(excludes)
    changed, removed = env.diff_paths_for_provenance(before, after)
    env.provenance.record_writes(
        changed, role="system", action_id=env.current_action_id, turn=env.current_turn
    )
    for path in removed:
        env.provenance.remove(path)

    obs = WorkspaceObservation(
        tool=action.tool,
        stdout=result.stdout,
        stderr=result.stderr,
        data={"exit_code": result.exit_code, "changed_paths": changed, "removed_paths": removed},
        artifacts=changed,
        execution_time=time.perf_counter() - start,
    )
    if result.timed_out:
        obs.error = f"exec timeout after {timeout}s"
    elif result.exit_code != 0:
        obs.error = f"shell exited with code {result.exit_code}"
    return obs
