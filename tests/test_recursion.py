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


# ---------------------------------------------------------------------------
# Selective artifact export: only paths listed in <artifact path="..."/> copy.
# ---------------------------------------------------------------------------


class TestSelectiveArtifactExport:
    def _build_handler_with_envs(self, parent_root: Path, child_root: Path):
        from rlm.core.recursion import RecursionHandler

        cfg = WorkspaceConfig()

        # A minimally-equipped parent env: the parts ``_copy_artifacts_to_parent``
        # touches are workspace_root, provenance, current_turn.
        parent_prov = ProvenanceStore(parent_root / "_rlm_state" / "provenance.json")
        parent_prov.load()
        parent_env = MagicMock()
        parent_env.workspace_root = parent_root
        parent_env.provenance = parent_prov
        parent_env.current_turn = 1

        child_env = MagicMock()
        child_env.workspace_root = child_root

        parent_rlm = MagicMock()
        parent_rlm.workspace_config = cfg

        handler = RecursionHandler(
            parent_rlm=parent_rlm, parent_env=parent_env, lm_handler=MagicMock()
        )
        return handler, parent_env, child_env

    def test_only_listed_artifacts_are_copied(self, tmp_path: Path) -> None:
        # Set up a child workspace with 5 files, but list only 2 in final_artifacts.
        parent_root = tmp_path / "parent"
        (parent_root / "_rlm_state").mkdir(parents=True)
        child_root = tmp_path / "child"
        child_root.mkdir()
        for name in ("a.txt", "b.txt", "c.txt", "d.txt", "e.txt"):
            (child_root / name).write_text(name, encoding="utf-8")

        handler, parent_env, child_env = self._build_handler_with_envs(parent_root, child_root)

        mapping = handler._copy_artifacts_to_parent(
            child_env=child_env,
            child_id="child_1_1",
            final_artifacts=["a.txt", "c.txt"],
            action_id="t1.a1",
        )
        # Only the two listed files were copied.
        dest_root = parent_root / "_rlm_artifacts" / "children" / "child_1_1"
        assert (dest_root / "a.txt").exists()
        assert (dest_root / "c.txt").exists()
        for skipped in ("b.txt", "d.txt", "e.txt"):
            assert not (dest_root / skipped).exists()
        # Mapping reflects only what was copied.
        assert set(mapping.keys()) == {"a.txt", "c.txt"}
        # Provenance for copied artifacts is role=child.
        for child_rel in ("a.txt", "c.txt"):
            prov = parent_env.provenance.get(f"_rlm_artifacts/children/child_1_1/{child_rel}")
            assert prov is not None and prov.created.role == "child"

    def test_nonexistent_artifact_silently_skipped(self, tmp_path: Path) -> None:
        parent_root = tmp_path / "parent"
        (parent_root / "_rlm_state").mkdir(parents=True)
        child_root = tmp_path / "child"
        child_root.mkdir()
        (child_root / "real.txt").write_text("hello", encoding="utf-8")

        handler, parent_env, child_env = self._build_handler_with_envs(parent_root, child_root)
        mapping = handler._copy_artifacts_to_parent(
            child_env=child_env,
            child_id="child_1_1",
            final_artifacts=["real.txt", "ghost.txt"],
            action_id=None,
        )
        # Real file copied; ghost silently skipped (no exception).
        assert "real.txt" in mapping
        assert "ghost.txt" not in mapping

    def test_path_traversal_in_artifact_blocked(self, tmp_path: Path) -> None:
        """An artifact path that escapes the child workspace is silently
        skipped — defense in depth against a misbehaving model."""
        parent_root = tmp_path / "parent"
        (parent_root / "_rlm_state").mkdir(parents=True)
        child_root = tmp_path / "child"
        child_root.mkdir()
        # File outside the child workspace.
        (tmp_path / "outside.txt").write_text("secret", encoding="utf-8")

        handler, _, child_env = self._build_handler_with_envs(parent_root, child_root)
        mapping = handler._copy_artifacts_to_parent(
            child_env=child_env,
            child_id="child_1_1",
            final_artifacts=["../outside.txt"],
            action_id=None,
        )
        assert mapping == {}
        assert (
            not (parent_root / "_rlm_artifacts" / "children" / "child_1_1").exists()
            or not (
                parent_root / "_rlm_artifacts" / "children" / "child_1_1" / "outside.txt"
            ).exists()
        )


# ---------------------------------------------------------------------------
# Path-mapping observation: 0 / 1 / N artifact variants
# ---------------------------------------------------------------------------


class TestPathMappingMatrix:
    def test_zero_artifacts(self):
        from rlm.core.recursion import _format_path_mapping_observation

        out = _format_path_mapping_observation(child_answer="done", mapping={})
        assert "Answer: done" in out
        assert "no artifacts" in out
        assert "Artifact Mapping" not in out

    def test_one_artifact(self):
        from rlm.core.recursion import _format_path_mapping_observation

        out = _format_path_mapping_observation(
            child_answer="done", mapping={"x.txt": "_rlm_artifacts/children/c1/x.txt"}
        )
        assert "Artifact Mapping:" in out
        assert "- x.txt -> _rlm_artifacts/children/c1/x.txt" in out

    def test_many_artifacts_preserve_order(self):
        from rlm.core.recursion import _format_path_mapping_observation

        # Python 3.7+ dicts preserve insertion order; the formatter iterates
        # the dict directly, so the output reflects that order.
        mapping = {f"f{i}.txt": f"_rlm_artifacts/children/c1/f{i}.txt" for i in range(5)}
        out = _format_path_mapping_observation(child_answer="ok", mapping=mapping)
        # Each artifact appears, in the order we inserted them.
        idx = 0
        for i in range(5):
            line = f"- f{i}.txt -> _rlm_artifacts/children/c1/f{i}.txt"
            new_idx = out.find(line, idx)
            assert new_idx != -1, f"missing line for f{i}: {out!r}"
            idx = new_idx


# ---------------------------------------------------------------------------
# spawn_via_broker_batched: max_concurrent_subcalls cap
# ---------------------------------------------------------------------------


class TestBatchedConcurrencyCap:
    def test_concurrent_workers_bounded_by_config(self):
        """Even with N tasks, no more than ``max_concurrent_subcalls`` run
        in parallel."""
        import threading
        import time as _time

        from rlm.core.recursion import RecursionHandler
        from rlm.core.types import WorkspaceObservation

        cfg = WorkspaceConfig(recursion=RecursionConfig(max_concurrent_subcalls=2))
        parent_rlm = MagicMock()
        parent_rlm.workspace_config = cfg
        handler = RecursionHandler(
            parent_rlm=parent_rlm, parent_env=MagicMock(), lm_handler=MagicMock()
        )

        in_flight = 0
        peak = 0
        lock = threading.Lock()

        def slow_spawn(child_task: str, action_id):
            nonlocal in_flight, peak
            with lock:
                in_flight += 1
                peak = max(peak, in_flight)
            try:
                _time.sleep(0.05)
                return WorkspaceObservation(tool="rlm_query", stdout=f"a:{child_task}")
            finally:
                with lock:
                    in_flight -= 1

        handler.spawn = MagicMock(side_effect=slow_spawn)  # type: ignore[method-assign]
        out = handler.spawn_via_broker_batched(
            child_tasks=[f"t{i}" for i in range(8)], action_id=None
        )
        assert len(out["responses"]) == 8
        # Peak concurrency must not exceed the configured cap.
        assert peak <= 2, f"peak concurrency {peak} exceeded cap 2"

    def test_batched_zero_tasks_is_empty_response(self):
        from rlm.core.recursion import RecursionHandler

        cfg = WorkspaceConfig()
        parent_rlm = MagicMock()
        parent_rlm.workspace_config = cfg
        handler = RecursionHandler(
            parent_rlm=parent_rlm, parent_env=MagicMock(), lm_handler=MagicMock()
        )
        out = handler.spawn_via_broker_batched(child_tasks=[], action_id=None)
        assert out == {"responses": []}


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
