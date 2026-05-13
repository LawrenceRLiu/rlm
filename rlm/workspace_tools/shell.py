"""shell — execute the action body as a bash script inside the workspace container."""

from __future__ import annotations

import time
from shlex import quote as sh_quote
from typing import TYPE_CHECKING

from rlm.core.types import WorkspaceAction, WorkspaceObservation
from rlm.workspace_tools import ToolSpec

if TYPE_CHECKING:
    from rlm.environments.docker_workspace import DockerWorkspaceEnv

SPEC = ToolSpec(
    name="shell",
    short_description=(
        "Run shell commands for inspection, tests, build steps, and diagnostics. "
        "Use file tools for ordinary durable edits; avoid echo/tee/heredoc file writes."
    ),
    is_state_mutating=True,
    runs_on="container",
    body_required=True,
    example=(
        "No attributes. Body is raw bash — do NOT wrap in <code>, <script>, "
        "or markdown fences. "
        'Example: <action tool="shell">\nls -la _rlm_artifacts/\n</action>'
    ),
)

# Tempfile location relative to workspace root. Lives under _rlm_state so it
# is excluded from provenance diffing.
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
    directory = action.args.get("directory")
    if directory not in (None, ""):
        workdir = env.resolve_workspace_path(str(directory))
        if not workdir.exists() or not workdir.is_dir():
            return WorkspaceObservation(
                tool=action.tool,
                error=f"directory does not exist or is not a directory: {directory}",
                execution_time=time.perf_counter() - start,
            )
        rel = workdir.relative_to(env.workspace_root).as_posix()
        body = f"cd {sh_quote('/workspace' if rel == '.' else '/workspace/' + rel)}\n{body}"

    action_id = env.current_action_id or "unknown"
    rel_tmp = f"{_TMP_REL_DIR}/shell_{action_id}.sh"
    tmp_path = env.workspace_root / rel_tmp
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_text(body, encoding="utf-8")

    excludes = env.workspace_config.recursion.copy_on_spawn_excludes
    before = env.snapshot_paths_for_provenance(excludes)

    result = env.exec_in_container(
        ["bash", f"/workspace/{rel_tmp}"],
        timeout=timeout,
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
