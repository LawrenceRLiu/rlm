"""Unit tests for the recursion machinery (workspace substrate Phase 6).

These tests don't touch Docker. They exercise:

- ``_copy_on_spawn`` exclude + size-cap behavior
- ``_seed_user_provenance`` skipping reserved state paths
- ``_format_path_mapping_observation`` format
- ``RecursionHandler._next_child_id`` allocation
- The max-depth fail-fast through the rlm_query action when
  ``env.recursion_handler`` is ``None``

End-to-end recursion (real container, real LM, copy-on-spawn → child run →
artifact pull-back) is covered by ``tests/test_docker_workspace.py`` via the
mock-LM-backed integration test added in Phase 6.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rlm.core.config import RecursionConfig, WorkspaceConfig
from rlm.core.recursion import (
    RecursionHandler,
    _copy_on_spawn,
    _format_path_mapping_observation,
    _seed_user_provenance,
)
from rlm.core.types import WorkspaceAction
from rlm.utils.provenance import ProvenanceStore
from rlm.workspace_tools.rlm_query import execute as rlm_query_execute

# ---------------------------------------------------------------------------
# copy-on-spawn
# ---------------------------------------------------------------------------


class TestCopyOnSpawn:
    def _seed(self, root: Path) -> None:
        (root / "a.txt").write_text("a")
        (root / "_rlm_notes").mkdir()
        (root / "_rlm_notes" / "b.txt").write_text("b")
        (root / ".git").mkdir()
        (root / ".git" / "HEAD").write_text("ref")
        (root / "node_modules").mkdir()
        (root / "node_modules" / "pkg.json").write_text("{}")
        (root / "_rlm_state").mkdir()
        (root / "_rlm_state" / "snapshots").mkdir()
        (root / "_rlm_state" / "snapshots" / "s1.bin").write_text("snapshot data")
        (root / "_rlm_state" / "action_log.jsonl").write_text("{}\n")
        (root / "big.bin").write_bytes(b"x" * 200)

    def test_excludes_skip_dirs_at_top_and_nested(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        self._seed(src)
        dst = tmp_path / "dst"

        _copy_on_spawn(
            src=src,
            dst=dst,
            excludes=(".git", "node_modules", "_rlm_state/snapshots"),
            max_file_bytes=10_000,
        )

        assert (dst / "a.txt").exists()
        assert (dst / "_rlm_notes" / "b.txt").exists()
        assert (dst / "_rlm_state" / "action_log.jsonl").exists()
        # Excluded:
        assert not (dst / ".git").exists()
        assert not (dst / "node_modules").exists()
        assert not (dst / "_rlm_state" / "snapshots").exists()

    def test_size_cap_drops_oversized_files(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        self._seed(src)
        dst = tmp_path / "dst"

        _copy_on_spawn(src=src, dst=dst, excludes=(), max_file_bytes=100)

        assert (dst / "a.txt").exists()
        assert not (dst / "big.bin").exists()


# ---------------------------------------------------------------------------
# user-role provenance reseed
# ---------------------------------------------------------------------------


class TestSeedUserProvenance:
    def test_marks_only_non_state_paths(self, tmp_path: Path) -> None:
        # Hand-build a minimal env-like object (we only need workspace_root,
        # provenance, and is_reserved_path). The full DockerWorkspaceEnv setup
        # is out of scope for a Docker-free test.
        ws = tmp_path / "ws"
        (ws / "_rlm_notes").mkdir(parents=True)
        (ws / "_rlm_notes" / "n.md").write_text("note")
        (ws / "_rlm_query_0.txt").write_text("task")
        (ws / "_rlm_state").mkdir()
        (ws / "_rlm_state" / "manifest.json").write_text("{}")

        store_path = ws / "_rlm_state" / "provenance.json"
        prov = ProvenanceStore(store_path)
        prov.load()

        env = MagicMock()
        env.workspace_root = ws
        env.provenance = prov

        def is_reserved(rel: str) -> bool:
            norm = rel.replace("\\", "/").lstrip("./")
            return norm.startswith("_rlm_state/") or norm == "_rlm_state"

        env.is_reserved_path.side_effect = is_reserved

        _seed_user_provenance(env)

        # Reload to confirm persistence.
        reloaded = ProvenanceStore(store_path)
        reloaded.load()

        n = reloaded.get("_rlm_notes/n.md")
        q = reloaded.get("_rlm_query_0.txt")
        m = reloaded.get("_rlm_state/manifest.json")

        assert n is not None and n.created.role == "user" and n.modified.role == "user"
        assert q is not None and q.created.role == "user"
        # State paths skipped — never recorded by _seed_user_provenance.
        assert m is None


# ---------------------------------------------------------------------------
# path-mapping observation formatter
# ---------------------------------------------------------------------------


class TestPathMappingObservation:
    def test_with_artifacts(self):
        out = _format_path_mapping_observation(
            child_answer="42",
            mapping={"out/result.json": "_rlm_artifacts/children/child_3_1/out/result.json"},
        )
        assert "Answer: 42" in out
        assert "Artifact Mapping:" in out
        assert "- out/result.json -> _rlm_artifacts/children/child_3_1/out/result.json" in out

    def test_without_artifacts(self):
        out = _format_path_mapping_observation(child_answer="fine", mapping={})
        assert "no artifacts" in out
        assert "Artifact Mapping" not in out


# ---------------------------------------------------------------------------
# child_id allocation
# ---------------------------------------------------------------------------


class TestChildIdAllocation:
    def test_increments_per_turn(self):
        cfg = WorkspaceConfig(recursion=RecursionConfig(max_concurrent_subcalls=2))
        parent_rlm = MagicMock()
        parent_rlm.workspace_config = cfg
        handler = RecursionHandler(
            parent_rlm=parent_rlm, parent_env=MagicMock(), lm_handler=MagicMock()
        )
        # Same turn → idx grows
        assert handler._next_child_id(turn=3) == "child_3_1"
        assert handler._next_child_id(turn=3) == "child_3_2"
        # New turn → idx restarts
        assert handler._next_child_id(turn=4) == "child_4_1"
        # Old turn → continues counter (deterministic per-turn idx)
        assert handler._next_child_id(turn=3) == "child_3_3"


# ---------------------------------------------------------------------------
# Max-depth fail-fast through the rlm_query action
# ---------------------------------------------------------------------------


class TestMaxDepthFailFast:
    def test_action_returns_error_when_handler_unwired(self):
        env = MagicMock()
        env.recursion_handler = None
        action = WorkspaceAction(
            tool="rlm_query",
            args={},
            body="please solve subtask",
            raw='<action tool="rlm_query">please solve subtask</action>',
        )
        obs = rlm_query_execute(env, action)
        assert obs.error is not None
        assert "Maximum recursion depth" in obs.error

    def test_action_delegates_to_handler_when_wired(self):
        env = MagicMock()
        env.current_action_id = "t2.a1"
        # A handler stub that returns a known observation.
        from rlm.core.types import WorkspaceObservation

        sentinel = WorkspaceObservation(tool="rlm_query", stdout="child-said-hi")
        env.recursion_handler.spawn = MagicMock(return_value=sentinel)

        action = WorkspaceAction(
            tool="rlm_query",
            args={},
            body="run subtask",
            raw="",
        )
        obs = rlm_query_execute(env, action)
        assert obs is sentinel
        env.recursion_handler.spawn.assert_called_once_with(
            child_task="run subtask", action_id="t2.a1"
        )


# ---------------------------------------------------------------------------
# Wire-on-construction: depth-aware recursion handler attachment
# ---------------------------------------------------------------------------


class TestRLMWireRecursion:
    def test_wires_handler_below_max_depth(self):
        from rlm.core.rlm import RLM

        rlm = RLM(
            backend="openai",
            backend_kwargs={"model_name": "fake"},
            depth=0,
            max_depth=2,
        )
        env = MagicMock()
        env.recursion_handler = None
        rlm._wire_recursion(env=env, lm_handler=MagicMock())
        assert env.recursion_handler is not None

    def test_omits_handler_at_max_depth(self):
        from rlm.core.rlm import RLM

        rlm = RLM(
            backend="openai",
            backend_kwargs={"model_name": "fake"},
            depth=2,
            max_depth=2,
        )
        env = MagicMock()
        env.recursion_handler = None
        rlm._wire_recursion(env=env, lm_handler=MagicMock())
        assert env.recursion_handler is None


# ---------------------------------------------------------------------------
# Batched broker entry concurrency
# ---------------------------------------------------------------------------


class TestSpawnViaBrokerBatched:
    def test_returns_one_response_per_task_in_order(self):
        cfg = WorkspaceConfig(recursion=RecursionConfig(max_concurrent_subcalls=3))
        parent_rlm = MagicMock()
        parent_rlm.workspace_config = cfg
        handler = RecursionHandler(
            parent_rlm=parent_rlm, parent_env=MagicMock(), lm_handler=MagicMock()
        )

        # Stub spawn so we can verify wiring without doing a real run.
        from rlm.core.types import WorkspaceObservation

        def fake_spawn(child_task: str, action_id):
            del action_id
            return WorkspaceObservation(tool="rlm_query", stdout=f"answer:{child_task}")

        handler.spawn = MagicMock(side_effect=fake_spawn)  # type: ignore[method-assign]
        out = handler.spawn_via_broker_batched(child_tasks=["a", "b", "c"], action_id="t1.a1")
        assert out["responses"] == ["answer:a", "answer:b", "answer:c"]

    def test_propagates_per_task_errors(self):
        cfg = WorkspaceConfig(recursion=RecursionConfig(max_concurrent_subcalls=2))
        parent_rlm = MagicMock()
        parent_rlm.workspace_config = cfg
        handler = RecursionHandler(
            parent_rlm=parent_rlm, parent_env=MagicMock(), lm_handler=MagicMock()
        )

        from rlm.core.types import WorkspaceObservation

        def fake_spawn(child_task, action_id):
            del action_id
            if child_task == "boom":
                return WorkspaceObservation(tool="rlm_query", error="kaboom")
            return WorkspaceObservation(tool="rlm_query", stdout="ok")

        handler.spawn = MagicMock(side_effect=fake_spawn)  # type: ignore[method-assign]
        out = handler.spawn_via_broker_batched(child_tasks=["boom", "ok"], action_id=None)
        assert out["responses"] == ["Error: kaboom", "ok"]


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
