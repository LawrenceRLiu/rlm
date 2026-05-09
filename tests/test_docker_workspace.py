"""End-to-end tests for ``DockerWorkspaceEnv``.

These exercise the real Docker container and are skipped if Docker is not
available or the workspace image hasn't been built. To enable:

    make build-image
    uv run pytest tests/test_docker_workspace.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from rlm.core.config import DockerConfig, ObservationConfig, WorkspaceConfig
from rlm.core.types import WorkspaceAction
from rlm.environments.docker_workspace import DockerWorkspaceEnv

DOCKER_AVAILABLE = shutil.which("docker") is not None
IMAGE_TAG = "rlm-workspace:0.1.0"


def _image_present() -> bool:
    if not DOCKER_AVAILABLE:
        return False
    r = subprocess.run(
        ["docker", "image", "inspect", IMAGE_TAG],
        capture_output=True,
    )
    return r.returncode == 0


pytestmark = pytest.mark.skipif(
    not _image_present(),
    reason=f"Docker / workspace image {IMAGE_TAG} not available",
)


def _make_env(tmp_path: Path, **overrides) -> DockerWorkspaceEnv:
    cfg = WorkspaceConfig(
        observation=ObservationConfig(max_observation_chars=4_000),
        docker=DockerConfig(
            image=IMAGE_TAG,
            workspace_root_base=str(tmp_path),
            broker_port=8080,
            poll_interval_ms=50,
            exec_timeout_seconds=10,
            cleanup_mode="delete",
        ),
    )
    return DockerWorkspaceEnv(workspace_config=cfg, **overrides)


def test_setup_creates_layout_and_seeds_provenance(tmp_path: Path) -> None:
    env = _make_env(tmp_path)
    try:
        env.setup()
        assert (env.workspace_root / "_rlm_query_0.txt").exists()
        assert (env.workspace_root / "_rlm_notes").is_dir()
        assert (env.workspace_root / "_rlm_artifacts").is_dir()
        assert (env.workspace_root / "_rlm_state" / "provenance.json").exists()
        assert (env.workspace_root / ".git").is_dir()
        # Provenance seeded for root task (user) and state files (system).
        env.provenance.load()
        prov = env.provenance.get("_rlm_query_0.txt")
        assert prov is not None
        assert prov.created.role == "user"
    finally:
        env.cleanup()


def test_run_action_write_then_read(tmp_path: Path) -> None:
    env = _make_env(tmp_path)
    try:
        env.setup()
        env.current_turn = 1

        write_action = WorkspaceAction(
            tool="write_file",
            args={"path": "_rlm_notes/scratch.md"},
            body="hello world\nsecond line\n",
            raw='<action tool="write_file" path="_rlm_notes/scratch.md">...</action>',
        )
        obs = env.run_action(write_action)
        assert obs.error is None, obs.error
        assert "_rlm_notes/scratch.md" in obs.artifacts

        env.provenance.load()
        prov = env.provenance.get("_rlm_notes/scratch.md")
        assert prov is not None
        assert prov.created.role == "assistant"
        assert prov.created.action_id == "t1.a1"

        read_action = WorkspaceAction(
            tool="read_file",
            args={"path": "_rlm_notes/scratch.md"},
            body=None,
            raw='<action tool="read_file" path="_rlm_notes/scratch.md" />',
        )
        obs2 = env.run_action(read_action)
        assert obs2.error is None
        assert "hello world" in obs2.stdout
    finally:
        env.cleanup()


def test_shell_action_records_system_provenance(tmp_path: Path) -> None:
    env = _make_env(tmp_path)
    try:
        env.setup()
        env.current_turn = 2
        action = WorkspaceAction(
            tool="shell",
            args={},
            body="echo touched > _rlm_notes/shell_out.txt && echo done",
            raw='<action tool="shell">...</action>',
        )
        obs = env.run_action(action)
        assert obs.error is None, (obs.error, obs.stderr)
        assert "done" in obs.stdout
        assert (env.workspace_root / "_rlm_notes" / "shell_out.txt").exists()

        env.provenance.load()
        prov = env.provenance.get("_rlm_notes/shell_out.txt")
        assert prov is not None
        assert prov.created.role == "system"
        assert prov.created.action_id == "t2.a1"
    finally:
        env.cleanup()


def test_snapshot_produces_commit(tmp_path: Path) -> None:
    env = _make_env(tmp_path)
    try:
        env.setup()
        env.current_turn = 1
        env.run_action(
            WorkspaceAction(
                tool="write_file",
                args={"path": "_rlm_notes/a.txt"},
                body="A",
                raw="",
            )
        )
        snap = env.snapshot(turn=1)
        assert snap.turn == 1
        assert len(snap.commit_sha) >= 7
        # turn 0 baseline -> turn 1 should mention the new file.
        assert any("a.txt" in f for f in snap.changed_files)
    finally:
        env.cleanup()


def test_observation_truncation_spills_to_artifact(tmp_path: Path) -> None:
    env = _make_env(tmp_path)
    try:
        env.setup()
        env.current_turn = 1
        # cap is 4_000; emit something well over.
        action = WorkspaceAction(
            tool="shell",
            args={},
            body="python -c \"print('x' * 8000)\"",
            raw="",
        )
        obs = env.run_action(action)
        assert obs.error is None, (obs.error, obs.stderr)
        assert "Observation truncated" in obs.stdout
        spill = next(p for p in obs.artifacts if p.startswith("_rlm_artifacts/_observations/"))
        assert (env.workspace_root / spill).exists()
    finally:
        env.cleanup()


def test_load_context_writes_query_slots(tmp_path: Path) -> None:
    env = _make_env(tmp_path)
    try:
        env.setup()
        env.load_context("the root task")
        assert (env.workspace_root / "_rlm_query_0.txt").read_text() == "the root task"
        env.load_context(["chunk a", "chunk b"])
        assert (env.workspace_root / "_rlm_query_1.txt").read_text() == "chunk a"
        assert (env.workspace_root / "_rlm_query_2.txt").read_text() == "chunk b"
    finally:
        env.cleanup()


def test_action_log_records_each_action(tmp_path: Path) -> None:
    env = _make_env(tmp_path)
    try:
        env.setup()
        env.current_turn = 1
        env.run_action(
            WorkspaceAction(
                tool="write_file",
                args={"path": "_rlm_notes/a.txt"},
                body="A",
                raw="",
            )
        )
        env.run_action(
            WorkspaceAction(
                tool="write_file",
                args={"path": "_rlm_notes/b.txt"},
                body="B",
                raw="",
            )
        )
        log_path = env.workspace_root / "_rlm_state" / "action_log.jsonl"
        lines = [json.loads(line) for line in log_path.read_text().splitlines() if line]
        assert len(lines) == 2
        assert lines[0]["action_id"] == "t1.a1"
        assert lines[1]["action_id"] == "t1.a2"
        assert all(line["mutating"] for line in lines)
    finally:
        env.cleanup()


def test_reserved_path_writes_blocked(tmp_path: Path) -> None:
    env = _make_env(tmp_path)
    try:
        env.setup()
        env.current_turn = 1
        obs = env.run_action(
            WorkspaceAction(
                tool="write_file",
                args={"path": "_rlm_state/sneaky.txt"},
                body="nope",
                raw="",
            )
        )
        assert obs.error is not None
        assert "reserved" in obs.error.lower()
    finally:
        env.cleanup()


def test_path_traversal_blocked(tmp_path: Path) -> None:
    env = _make_env(tmp_path)
    try:
        env.setup()
        env.current_turn = 1
        with pytest.raises(ValueError, match="escapes workspace"):
            env.resolve_workspace_path("../escape.txt")
    finally:
        env.cleanup()


def test_broker_health_reachable(tmp_path: Path) -> None:
    """Smoke-check that the broker came up and the host poller can reach it."""
    import requests

    env = _make_env(tmp_path)
    try:
        env.setup()
        # Give the poller a tick.
        time.sleep(0.2)
        port = env._broker_host_port
        assert port is not None
        r = requests.get(f"http://127.0.0.1:{port}/health", timeout=2.0)
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
    finally:
        env.cleanup()
