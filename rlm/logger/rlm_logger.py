"""
Logger for RLM iterations.

Captures run metadata and iterations in memory so they can be attached to
RLMChatCompletion.metadata. Optionally writes the same data to JSON-lines files.
"""

import json
import os
import uuid
from datetime import datetime

from rlm.core.types import RLMMetadata, WorkspaceIteration


class RLMLogger:
    """
    Captures trajectory (run metadata + iterations) for each completion.
    By default only captures in memory; set log_dir to also save to disk.

    - log_dir=None: trajectory is available via get_trajectory() and can be
      attached to RLMChatCompletion.metadata (no disk write).
    - log_dir="path": same capture plus appends to a JSONL file per run.
    """

    def __init__(self, log_dir: str | None = None, file_name: str = "rlm"):
        self._save_to_disk = log_dir is not None
        self.log_dir = log_dir
        self.log_file_path: str | None = None
        if self._save_to_disk and log_dir:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_id = str(uuid.uuid4())[:8]
            self.log_file_path = os.path.join(log_dir, f"{file_name}_{timestamp}_{run_id}.jsonl")

        self._run_metadata: dict | None = None
        self._iterations: list[dict] = []
        self._iteration_count = 0
        self._metadata_logged = False

    def log_metadata(self, metadata: RLMMetadata) -> None:
        """Capture run metadata (and optionally write to file)."""
        if self._metadata_logged:
            return

        self._run_metadata = metadata.to_dict()
        self._metadata_logged = True

        if self._save_to_disk and self.log_file_path:
            entry = {
                "type": "metadata",
                "timestamp": datetime.now().isoformat(),
                **self._run_metadata,
            }
            with open(self.log_file_path, "a") as f:
                json.dump(entry, f)
                f.write("\n")

    def log(self, iteration: WorkspaceIteration) -> None:
        """Capture one iteration (and optionally append to file)."""
        self._iteration_count += 1
        entry = {
            "type": "iteration",
            "iteration": self._iteration_count,
            "timestamp": datetime.now().isoformat(),
            **iteration.to_dict(),
        }
        self._iterations.append(entry)

        if self._save_to_disk and self.log_file_path:
            with open(self.log_file_path, "a") as f:
                json.dump(entry, f)
                f.write("\n")

    def log_iteration(self, iteration: WorkspaceIteration) -> None:
        """Workspace-substrate alias for ``log()``. See plan Phase 4."""
        self.log(iteration)

    def log_compaction(
        self,
        *,
        turn: int,
        tokens_before: int,
        threshold_tokens: int,
        summary: str,
        dropped_iterations: int,
        retained_tail_iterations: int,
    ) -> None:
        """Capture a substrate-level compaction event.

        Emitted just before the turn that fired compaction runs. The summary
        is the model-authored prose that replaces the pre-compress trajectory
        in the visible prompt; the actual iterations remain in this logger's
        in-memory list and in the workspace's git snapshots.
        """
        entry = {
            "type": "compaction",
            "timestamp": datetime.now().isoformat(),
            "turn": turn,
            "tokens_before": tokens_before,
            "threshold_tokens": threshold_tokens,
            "dropped_iterations": dropped_iterations,
            "retained_tail_iterations": retained_tail_iterations,
            "summary": summary,
        }
        # Keep compaction rows alongside iterations so the visualizer sees
        # them in insertion order; the explicit "type" field distinguishes
        # them from iteration rows.
        self._iterations.append(entry)
        if self._save_to_disk and self.log_file_path:
            with open(self.log_file_path, "a") as f:
                json.dump(entry, f)
                f.write("\n")

    def clear_iterations(self) -> None:
        """Reset iterations for the next completion (trajectory is per completion)."""
        self._iterations = []
        self._iteration_count = 0

    def get_trajectory(self) -> dict | None:
        """Return captured run_metadata + iterations for the current completion, or None if no metadata yet."""
        if self._run_metadata is None:
            return None
        return {
            "run_metadata": self._run_metadata,
            "iterations": list(self._iterations),
        }

    @property
    def iteration_count(self) -> int:
        return self._iteration_count
