"""
Workspace tool registry.

Each tool module exports two things:

- ``SPEC``: a ``ToolSpec`` describing the tool (name, host vs container,
  whether it mutates state, whether it requires a body).
- ``execute(env, action, *, action_id, turn) -> WorkspaceObservation``: the
  callable that runs the action against a ``DockerWorkspaceEnv``.

The registry is built lazily from the per-tool modules so importing this
package has minimal side effects.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from rlm.core.types import WorkspaceAction, WorkspaceObservation

if TYPE_CHECKING:  # avoid circular imports at runtime
    from rlm.environments.docker_workspace import DockerWorkspaceEnv


ToolExecute = Callable[["DockerWorkspaceEnv", WorkspaceAction], WorkspaceObservation]


@dataclass(frozen=True)
class ToolSpec:
    """Static description of a workspace tool."""

    name: str
    short_description: str
    is_state_mutating: bool
    runs_on: Literal["host", "container"]
    body_required: bool
    is_terminal: bool = False  # `final` is the only terminal tool


# Lazy import: each tool module is loaded on first lookup so that absent
# optional deps don't break package import. Tool modules are guaranteed to
# import cheaply (they import dataclasses and typing only at module level).
_TOOL_MODULES = {
    "list_directory": "rlm.workspace_tools.list_directory",
    "read_file": "rlm.workspace_tools.read_file",
    "write_file": "rlm.workspace_tools.write_file",
    "append_file": "rlm.workspace_tools.append_file",
    "edit_file": "rlm.workspace_tools.edit_file",
    "shell": "rlm.workspace_tools.shell",
    "python": "rlm.workspace_tools.python",
    "llm_query": "rlm.workspace_tools.llm_query",
    "rlm_query": "rlm.workspace_tools.rlm_query",
    "final": "rlm.workspace_tools.final",
}


def _load(name: str):
    import importlib

    if name not in _TOOL_MODULES:
        raise KeyError(f"Unknown workspace tool: {name!r}")
    return importlib.import_module(_TOOL_MODULES[name])


def get_spec(name: str) -> ToolSpec:
    return _load(name).SPEC


def get_executor(name: str) -> ToolExecute:
    return _load(name).execute


def all_tool_names() -> list[str]:
    return list(_TOOL_MODULES.keys())


def is_state_mutating(name: str) -> bool:
    return get_spec(name).is_state_mutating


__all__ = [
    "ToolSpec",
    "ToolExecute",
    "get_spec",
    "get_executor",
    "all_tool_names",
    "is_state_mutating",
]
