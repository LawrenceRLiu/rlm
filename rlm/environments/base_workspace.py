"""
Base class for workspace-substrate environments.

The workspace substrate replaces the legacy REPL substrate: instead of
executing code blocks against a Python kernel, the model emits XML
``<action>`` blocks that mutate a durable workspace directory. Memory lives
in files; recursion is a copy-on-spawn child workspace.

A ``BaseWorkspaceEnv`` is responsible for:

1. ``setup``: provision the workspace (directories + git baseline + container
   if applicable) and any state required by tools.
2. ``load_context``: drop user-supplied context payloads into the workspace
   as ``_rlm_query_<N>.txt`` files (with provenance role ``user``).
3. ``run_action``: execute a single ``WorkspaceAction`` and return a
   ``WorkspaceObservation``. The env MUST set ``current_action_id`` and
   ``current_turn`` so per-tool provenance updates can attribute writes.
4. ``snapshot``: take a per-turn snapshot (git commit) so the visualizer can
   replay file-state across turns.
5. ``cleanup``: release any external resources (containers, workspace dirs).

The single concrete impl is ``DockerWorkspaceEnv``. Tests/mocks may
subclass with simpler backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from rlm.core.types import WorkspaceAction, WorkspaceObservation, WorkspaceSnapshot


class BaseWorkspaceEnv(ABC):
    """Abstract workspace-substrate environment."""

    @abstractmethod
    def setup(self) -> None:
        """Create the workspace, seed reserved files, initialise git, and
        bring up any container/runtime required by ``run_action``.
        """

    @abstractmethod
    def load_context(self, context_payload: Any) -> None:
        """Write user-supplied context into the workspace.

        ``context_payload`` may be ``str``, ``list``, or ``dict``; multi-chunk
        payloads land as ``_rlm_query_1.txt``, ``_rlm_query_2.txt``, ... with
        the root task at ``_rlm_query_0.txt``.
        """

    @abstractmethod
    def run_action(self, action: WorkspaceAction) -> WorkspaceObservation:
        """Dispatch a single action through the tool registry."""

    @abstractmethod
    def snapshot(self, turn: int) -> WorkspaceSnapshot:
        """Take a per-turn snapshot. Idempotent for ``--allow-empty`` commits."""

    @abstractmethod
    def cleanup(self) -> None:
        """Release external resources."""

    # -- context-manager sugar ---------------------------------------------
    def __enter__(self) -> BaseWorkspaceEnv:
        self.setup()
        return self

    def __exit__(self, *exc: Any) -> bool:
        del exc
        self.cleanup()
        return False
