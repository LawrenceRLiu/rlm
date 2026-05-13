"""run_python_command — execute Python with RLM helper functions preimported."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rlm.core.types import WorkspaceAction, WorkspaceObservation
from rlm.workspace_tools import ToolSpec
from rlm.workspace_tools.python import execute as python_execute

if TYPE_CHECKING:
    from rlm.environments.docker_workspace import DockerWorkspaceEnv

SPEC = ToolSpec(
    name="run_python_command",
    short_description=(
        "Run Python code inside the workspace container. "
        "`llm_query`, `llm_query_batched`, `rlm_query`, and `rlm_query_batched` "
        "are pre-imported for programmatic loops over files and batched model calls."
    ),
    is_state_mutating=True,
    runs_on="container",
    body_required=False,
    example='Native args: {"code": "from pathlib import Path\\nprint(Path(\\\".\\\").resolve())"}',
)


def execute(env: DockerWorkspaceEnv, action: WorkspaceAction) -> WorkspaceObservation:
    return python_execute(env, action)
