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
        "Run scratch Python for computation, parsing, validation, tests, and diagnostics. "
        "Default working dir is the workspace root (``/``); ``PYTHONPATH`` includes "
        "``/app`` so the task's package layout imports cleanly. Pass ``cwd=`` to "
        "override the working directory. "
        "Use file tools for ordinary durable edits; avoid printing large generated artifacts. "
        "`llm_query`, `llm_query_batched`, `rlm_query`, `rlm_query_batched` are pre-imported."
    ),
    is_state_mutating=True,
    runs_on="container",
    body_required=True,
    example=(
        "No attributes required. Body is raw Python — do NOT wrap in <code>, "
        "<script>, <python>, or markdown fences. "
        'Example: <action tool="python">\nimport math\nprint(math.pi)\n</action>'
    ),
)

# Wrapper preamble: makes the in-container client helpers available in the
# script's globals without requiring an explicit import. Sits above the body
# in the materialised tempfile.
_WRAPPER_PREAMBLE = (
    "from rlm_workspace.client import llm_query, llm_query_batched, rlm_query, rlm_query_batched\n"
)

# Tempfile location relative to workspace root. Lives under _rlm_state so it
# is excluded from provenance diffing (the env's provenance excludes always
# union in _rlm_state). Kept on disk after exec for debuggability — they
# disappear when the workspace is cleaned up.
_TMP_REL_DIR = "_rlm_state/_tmp"


def execute(env: DockerWorkspaceEnv, action: WorkspaceAction) -> WorkspaceObservation:
    start = time.perf_counter()
    body = action.body if action.body is not None else str(action.args.get("code", ""))
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

    # Optional cwd override: defaults to docker.container_cwd, which is
    # normally the workspace root (/). Accepts workspace-relative or
    # container-absolute paths under one of the bind-mount roots.
    cwd_arg = action.args.get("cwd")
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
                error=f"cwd does not exist or is not a directory: {cwd_arg}",
                execution_time=time.perf_counter() - start,
            )
        try:
            container_cwd = env.host_to_container_path(host_workdir)
        except ValueError as e:
            return WorkspaceObservation(
                tool=action.tool,
                error=f"cwd is not visible inside the container: {e}",
                execution_time=time.perf_counter() - start,
            )

    action_id = env.current_action_id or "unknown"
    rel_tmp = f"{_TMP_REL_DIR}/python_{action_id}.py"
    tmp_path = env.workspace_root / rel_tmp
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_text(_WRAPPER_PREAMBLE + body, encoding="utf-8")

    excludes = env.workspace_config.recursion.copy_on_spawn_excludes
    before = env.snapshot_paths_for_provenance(excludes)

    # _rlm_state is bind-mounted at /_rlm_state in the container.
    result = env.exec_in_container(
        ["python", f"/{rel_tmp}"],
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

    # Drain any RLMChatCompletions that broker workers captured while this
    # action's script was running (one-per ``llm_query`` / ``rlm_query``,
    # N-per ``*_batched``). Thread-safe pop; safe-no-op if action_id is None.
    captured_calls = env.drain_broker_ledger(env.current_action_id)

    obs = WorkspaceObservation(
        tool=action.tool,
        stdout=result.stdout,
        stderr=result.stderr,
        data={"exit_code": result.exit_code, "changed_paths": changed, "removed_paths": removed},
        artifacts=changed,
        rlm_calls=captured_calls,
        execution_time=time.perf_counter() - start,
    )
    if result.timed_out:
        obs.error = f"exec timeout after {timeout}s"
    elif result.exit_code != 0:
        obs.error = f"python exited with code {result.exit_code}"
    return obs
