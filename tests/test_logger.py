"""Direct tests for ``rlm.logger.RLMLogger`` JSONL output.

The visualizer at ``visualizer/`` consumes these JSONL files line-by-line, so
the on-disk schema is part of the project's public contract:

- One file per run, named ``rlm_<YYYY-MM-DD>_<HH-MM-SS>_<8-hex>.jsonl``
- First line: ``{"type": "metadata", ...}`` (run-level)
- Subsequent lines: ``{"type": "iteration", "iteration": N, ...}`` with N
  contiguous from 1
- Every line is valid JSON

Every JSON shape change must update the visualizer types in
``visualizer/src/lib/types.ts`` in the same PR; the golden snapshot in
``tests/test_e2e_rollout.py`` enforces that pairing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from rlm.core.types import (
    RLMMetadata,
    WorkspaceAction,
    WorkspaceIteration,
    WorkspaceObservation,
    WorkspaceSnapshot,
)
from rlm.logger import RLMLogger


def _make_metadata() -> RLMMetadata:
    return RLMMetadata(
        root_model="mock-model",
        max_depth=2,
        max_iterations=5,
        backend="openai",
        backend_kwargs={"model_name": "mock-model"},
        environment_type="docker",
        environment_kwargs={"image": "rlm-workspace:0.1.0"},
    )


def _make_iteration(idx: int) -> WorkspaceIteration:
    action = WorkspaceAction(tool="list_directory", args={}, body=None, raw="")
    obs = WorkspaceObservation(tool="list_directory", stdout="ok")
    snap = WorkspaceSnapshot(turn=idx, commit_sha="abc1234", changed_files=[], workspace_root="/ws")
    return WorkspaceIteration(
        iteration=idx,
        timestamp=f"2026-01-01T00:00:0{idx:01d}",
        prompt=[{"role": "user", "content": "x"}],
        response=f"response {idx}",
        reasoning=None,
        actions=[action],
        observations=[obs],
        snapshot=snap,
    )


# ---------------------------------------------------------------------------
# Filename + on-disk
# ---------------------------------------------------------------------------


class TestLoggerFilename:
    def test_creates_one_jsonl_with_expected_name(self, tmp_path: Path) -> None:
        logger = RLMLogger(log_dir=str(tmp_path))
        logger.log_metadata(_make_metadata())
        files = list(tmp_path.glob("*.jsonl"))
        assert len(files) == 1
        # rlm_YYYY-MM-DD_HH-MM-SS_<8-hex>.jsonl
        assert re.match(
            r"^rlm_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_[0-9a-f]{8}\.jsonl$",
            files[0].name,
        )

    def test_in_memory_mode_writes_no_file(self, tmp_path: Path) -> None:
        logger = RLMLogger(log_dir=None)
        logger.log_metadata(_make_metadata())
        logger.log(_make_iteration(1))
        assert list(tmp_path.glob("*.jsonl")) == []
        traj = logger.get_trajectory()
        assert traj is not None
        assert traj["run_metadata"]["root_model"] == "mock-model"
        assert len(traj["iterations"]) == 1


# ---------------------------------------------------------------------------
# JSONL line ordering
# ---------------------------------------------------------------------------


class TestLoggerJSONLOrdering:
    def _read_lines(self, log_dir: Path) -> list[dict]:
        files = list(log_dir.glob("*.jsonl"))
        assert len(files) == 1
        return [json.loads(line) for line in files[0].read_text().splitlines() if line]

    def test_metadata_is_first_line(self, tmp_path: Path) -> None:
        logger = RLMLogger(log_dir=str(tmp_path))
        logger.log_metadata(_make_metadata())
        for i in range(1, 4):
            logger.log(_make_iteration(i))
        lines = self._read_lines(tmp_path)
        assert lines[0]["type"] == "metadata"
        assert all(line["type"] == "iteration" for line in lines[1:])

    def test_iteration_numbers_reflect_iteration_object(self, tmp_path: Path) -> None:
        """``WorkspaceIteration.iteration`` (set by ``_run_loop`` to the 1-based
        turn number) is the source of truth for the JSONL ``iteration`` field;
        the logger's internal counter is overridden by the iteration's value
        because ``iteration.to_dict()`` is splatted last in ``log()``."""
        logger = RLMLogger(log_dir=str(tmp_path))
        logger.log_metadata(_make_metadata())
        for i in (1, 2, 3, 4, 5):
            logger.log(_make_iteration(idx=i))
        lines = self._read_lines(tmp_path)
        iter_numbers = [line["iteration"] for line in lines if line["type"] == "iteration"]
        assert iter_numbers == [1, 2, 3, 4, 5]

    def test_log_metadata_is_idempotent(self, tmp_path: Path) -> None:
        logger = RLMLogger(log_dir=str(tmp_path))
        logger.log_metadata(_make_metadata())
        logger.log_metadata(_make_metadata())
        logger.log_metadata(_make_metadata())
        lines = self._read_lines(tmp_path)
        # Only one metadata line was written.
        assert sum(1 for line in lines if line["type"] == "metadata") == 1


# ---------------------------------------------------------------------------
# JSONL content shape (mirrors visualizer types in visualizer/src/lib/types.ts)
# ---------------------------------------------------------------------------


class TestLoggerLineShape:
    def test_metadata_keys(self, tmp_path: Path) -> None:
        logger = RLMLogger(log_dir=str(tmp_path))
        logger.log_metadata(_make_metadata())
        line = json.loads(next(tmp_path.glob("*.jsonl")).read_text().splitlines()[0])
        assert line["type"] == "metadata"
        # Every field RLMMetadata.to_dict() emits must be present.
        for key in (
            "timestamp",  # added by logger
            "root_model",
            "max_depth",
            "max_iterations",
            "backend",
            "backend_kwargs",
            "environment_type",
            "environment_kwargs",
            "other_backends",
        ):
            assert key in line, f"missing metadata key: {key}"

    def test_iteration_keys(self, tmp_path: Path) -> None:
        logger = RLMLogger(log_dir=str(tmp_path))
        logger.log_metadata(_make_metadata())
        logger.log(_make_iteration(1))
        line = json.loads(next(tmp_path.glob("*.jsonl")).read_text().splitlines()[1])
        assert line["type"] == "iteration"
        for key in (
            "iteration",
            "timestamp",
            "prompt",
            "response",
            "reasoning",
            "parse_attempts",
            "actions",
            "observations",
            "snapshot",
            "final_answer",
            "iteration_time",
            "error",
        ):
            assert key in line, f"missing iteration key: {key}"
        # Action / observation children mirror their to_dict shape.
        a = line["actions"][0]
        assert set(a.keys()) == {"tool", "args", "body", "raw"}
        o = line["observations"][0]
        for k in (
            "tool",
            "stdout",
            "stderr",
            "data",
            "artifacts",
            "execution_time",
            "rlm_calls",
            "final_answer",
            "final_artifacts",
            "error",
        ):
            assert k in o, f"missing observation key: {k}"
        # Snapshot mirror.
        s = line["snapshot"]
        assert set(s.keys()) == {"turn", "commit_sha", "changed_files", "workspace_root"}


# ---------------------------------------------------------------------------
# Append-safety + clear_iterations
# ---------------------------------------------------------------------------


class TestLoggerAppend:
    def test_log_append_safe_across_many_calls(self, tmp_path: Path) -> None:
        logger = RLMLogger(log_dir=str(tmp_path))
        logger.log_metadata(_make_metadata())
        for _ in range(50):
            logger.log(_make_iteration(idx=1))
        files = list(tmp_path.glob("*.jsonl"))
        assert len(files) == 1
        lines = files[0].read_text().splitlines()
        # 1 metadata + 50 iterations = 51 lines.
        assert len(lines) == 51
        # Every line is valid JSON.
        for line in lines:
            json.loads(line)

    def test_clear_iterations_resets_in_memory_count_only(self, tmp_path: Path) -> None:
        """``clear_iterations`` is meant for the per-completion in-memory
        trajectory; it must NOT touch the on-disk file (the file is the
        durable record across completions of one ``RLM`` instance)."""
        logger = RLMLogger(log_dir=str(tmp_path))
        logger.log_metadata(_make_metadata())
        logger.log(_make_iteration(1))
        logger.log(_make_iteration(2))
        files = list(tmp_path.glob("*.jsonl"))
        before = files[0].read_text()
        logger.clear_iterations()
        # On-disk file is unchanged.
        assert files[0].read_text() == before
        # In-memory state was reset.
        assert logger.iteration_count == 0
        traj = logger.get_trajectory()
        assert traj is not None
        assert traj["iterations"] == []
