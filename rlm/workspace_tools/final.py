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
    example=(
        'No attributes. Body has <answer>...</answer> plus zero or more <artifact path="..."/>. '
        'Example: <action tool="final"><answer>42 primes found</answer>'
        '<artifact path="primes.txt"/></action>'
    ),
)


def execute(env: DockerWorkspaceEnv, action: WorkspaceAction) -> WorkspaceObservation:
    del env  # final does not touch the env; signature is uniform across tools.
    start = time.perf_counter()
    if action.body is None and "answer" in action.args:
        answer = str(action.args["answer"])
        raw_artifacts = action.args.get("artifacts", [])
        if raw_artifacts is None:
            artifacts = []
        elif isinstance(raw_artifacts, list):
            artifacts = [str(path) for path in raw_artifacts]
        else:
            return WorkspaceObservation(
                tool=action.tool,
                error="artifacts must be a list of workspace-relative paths.",
                execution_time=time.perf_counter() - start,
            )
    else:
        try:
            answer, artifacts = parse_final_body(action.body or "")
        except ActionParseError as e:
            return WorkspaceObservation(
                tool=action.tool,
                error=str(e),
                execution_time=time.perf_counter() - start,
            )
    return WorkspaceObservation(
        tool=action.tool,
        stdout=answer,
        final_answer=answer,
        final_artifacts=artifacts,
        execution_time=time.perf_counter() - start,
    )
