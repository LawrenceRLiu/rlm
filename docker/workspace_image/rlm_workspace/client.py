"""In-container client for the workspace substrate broker.

This module is preimported into every ``python`` action body via a small
wrapper script (see ``rlm/workspace_tools/python.py``). It exposes four
blocking helpers that POST to ``http://localhost:<broker_port>/enqueue``:

- ``llm_query(prompt: str) -> str``
- ``llm_query_batched(prompts: list[str]) -> list[str]``
- ``rlm_query(prompt: str) -> str``
- ``rlm_query_batched(prompts: list[str]) -> list[str]``

Per Decision #27 there is **no** ``model=`` parameter. The parent ``RLM``'s
configured model is used for every LM call, including those originating
inside ``python`` action scripts.

Per Decision #23, when the runtime is at ``depth == max_depth`` it does not
forward ``rlm_query`` requests to the broker; instead the host poller returns
an explicit error string so model code that calls ``rlm_query`` at max depth
sees the same loud failure as a ``<action tool="rlm_query">`` action would.
"""

from __future__ import annotations

import os

import requests

_BROKER_PORT = int(os.environ.get("RLM_BROKER_PORT", "8080"))
_BROKER_URL = f"http://localhost:{_BROKER_PORT}/enqueue"


def _post(payload: dict) -> dict:
    # No client-side timeout: the broker blocks until the host poller answers,
    # and the host owns timeout/retry policy via ``docker.exec_timeout_seconds``
    # (which kills the whole ``python`` invocation if it exceeds the budget).
    r = requests.post(_BROKER_URL, json=payload)
    r.raise_for_status()
    return r.json()


def llm_query(prompt: str) -> str:
    """Single LM completion via the host. Returns the response text or an error string."""
    data = _post({"kind": "llm_query", "prompt": prompt})
    if "error" in data:
        return f"Error: {data['error']}"
    return data.get("response", "")


def llm_query_batched(prompts: list[str]) -> list[str]:
    """Batched LM completions. Returns one response (or error string) per prompt."""
    data = _post({"kind": "llm_query_batched", "prompts": prompts})
    if "error" in data and "responses" not in data:
        return [f"Error: {data['error']}"] * len(prompts)
    return data.get("responses", [])


def rlm_query(prompt: str) -> str:
    """Spawn a child RLM via the host. Returns the structured observation text
    (the child's final answer plus a path-mapping block) or an error string.
    """
    data = _post({"kind": "rlm_query", "prompt": prompt})
    if "error" in data:
        return f"Error: {data['error']}"
    return data.get("response", "")


def rlm_query_batched(prompts: list[str]) -> list[str]:
    """Batched recursive child RLM calls. Returns one observation string per prompt."""
    data = _post({"kind": "rlm_query_batched", "prompts": prompts})
    if "error" in data and "responses" not in data:
        return [f"Error: {data['error']}"] * len(prompts)
    return data.get("responses", [])
