"""llm_query — single LM completion against the host LMHandler. Read-only."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from rlm.core.comms_utils import LMRequest, send_lm_request
from rlm.core.types import WorkspaceAction, WorkspaceObservation
from rlm.workspace_tools import ToolSpec

if TYPE_CHECKING:
    from rlm.environments.docker_workspace import DockerWorkspaceEnv

SPEC = ToolSpec(
    name="llm_query",
    short_description="Single LM completion (no recursion). Body is the prompt.",
    is_state_mutating=False,
    runs_on="host",
    body_required=True,
    example=(
        'No attributes. Body is the prompt sent to the LM. '
        'Example: <action tool="llm_query">Summarize the following paragraph in one sentence: ...</action>'
    ),
)


def execute(env: DockerWorkspaceEnv, action: WorkspaceAction) -> WorkspaceObservation:
    start = time.perf_counter()
    prompt = action.body or ""
    request = LMRequest(prompt=prompt, depth=env.depth)
    if env.lm_handler_address is None:
        return WorkspaceObservation(
            tool=SPEC.name,
            error="No LM handler address configured for this environment.",
            execution_time=time.perf_counter() - start,
        )
    response = send_lm_request(env.lm_handler_address, request)
    if not response.success or response.chat_completion is None:
        return WorkspaceObservation(
            tool=SPEC.name,
            error=response.error or "LM request failed with no error message.",
            execution_time=time.perf_counter() - start,
        )
    rc = response.chat_completion
    return WorkspaceObservation(
        tool=SPEC.name,
        stdout=rc.response,
        data={"model": rc.root_model},
        rlm_calls=[rc],
        execution_time=time.perf_counter() - start,
    )
