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
    example=(
        'No attributes. Body is raw Python — do NOT wrap in <code>, <script>, '
        '<python>, or markdown fences. '
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
    body = action.body or ""

    action_id = env.current_action_id or "unknown"
    rel_tmp = f"{_TMP_REL_DIR}/python_{action_id}.py"
    tmp_path = env.workspace_root / rel_tmp
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_text(_WRAPPER_PREAMBLE + body, encoding="utf-8")

    excludes = env.workspace_config.recursion.copy_on_spawn_excludes
    before = env.snapshot_paths_for_provenance(excludes)

    # Container has the workspace mounted at /workspace; reference the script
    # via its in-container path.
    result = env.exec_in_container(
        ["python", f"/workspace/{rel_tmp}"],
        timeout=env.workspace_config.docker.exec_timeout_seconds,
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
        tool=SPEC.name,
        stdout=result.stdout,
        stderr=result.stderr,
        data={"exit_code": result.exit_code, "changed_paths": changed, "removed_paths": removed},
        artifacts=changed,
        rlm_calls=captured_calls,
        execution_time=time.perf_counter() - start,
    )
    if result.timed_out:
        obs.error = f"exec timeout after {env.workspace_config.docker.exec_timeout_seconds}s"
    elif result.exit_code != 0:
        obs.error = f"python exited with code {result.exit_code}"
    return obs
