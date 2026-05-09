"""final — terminal action: emit a final answer plus selected artifacts."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from rlm.core.types import WorkspaceAction, WorkspaceObservation
from rlm.utils.action_parser import parse_final_body
from rlm.utils.exceptions import ActionParseError
from rlm.workspace_tools import ToolSpec

if TYPE_CHECKING:
    from rlm.environments.docker_workspace import DockerWorkspaceEnv

SPEC = ToolSpec(
    name="final",
    short_description=(
        "Terminate the run with a final answer. "
        'Body: <answer>...</answer> plus zero or more <artifact path="..." />.'
    ),
    is_state_mutating=False,
    runs_on="host",
    body_required=True,
    is_terminal=True,
)


def execute(env: DockerWorkspaceEnv, action: WorkspaceAction) -> WorkspaceObservation:
    del env  # final does not touch the env; signature is uniform across tools.
    start = time.perf_counter()
    try:
        answer, artifacts = parse_final_body(action.body or "")
    except ActionParseError as e:
        return WorkspaceObservation(
            tool=SPEC.name,
            error=str(e),
            execution_time=time.perf_counter() - start,
        )
    return WorkspaceObservation(
        tool=SPEC.name,
        stdout=answer,
        final_answer=answer,
        final_artifacts=artifacts,
        execution_time=time.perf_counter() - start,
    )
