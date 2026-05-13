"""run_shell_command — Qwen-Code-style shell command tool."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rlm.core.types import WorkspaceAction, WorkspaceObservation
from rlm.workspace_tools import ToolSpec
from rlm.workspace_tools.shell import execute as shell_execute

if TYPE_CHECKING:
    from rlm.environments.docker_workspace import DockerWorkspaceEnv

SPEC = ToolSpec(
    name="run_shell_command",
    short_description=(
        "Run a shell command string inside the workspace container. The command is passed "
        "verbatim to bash through a script file; quote paths and use heredocs for large "
        "multiline literals. Optional: directory, timeout, description, is_background=false."
    ),
    is_state_mutating=True,
    runs_on="container",
    body_required=False,
    example=(
        'Native args: {"command": "python -m pytest", "directory": ".", '
        '"timeout": 300, "is_background": false}'
    ),
)


def execute(env: DockerWorkspaceEnv, action: WorkspaceAction) -> WorkspaceObservation:
    return shell_execute(env, action)
