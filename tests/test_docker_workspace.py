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


def test_python_action_runs_and_captures_stdout(tmp_path: Path) -> None:
    env = _make_env(tmp_path)
    try:
        env.setup()
        env.current_turn = 3
        action = WorkspaceAction(
            tool="python",
            args={},
            body="import sys\nprint('py-out')\nprint('py-err', file=sys.stderr)\n",
            raw='<action tool="python">...</action>',
        )
        obs = env.run_action(action)
        assert obs.error is None, (obs.error, obs.stderr)
        assert "py-out" in obs.stdout
        assert "py-err" in obs.stderr
        assert obs.data is not None and obs.data["exit_code"] == 0
        # The materialised script lives under _rlm_state/_tmp and is excluded
        # from provenance diffs (so it does NOT appear in changed_paths).
        script = env.workspace_root / "_rlm_state" / "_tmp" / "python_t3.a1.py"
        assert script.exists()
        assert "from rlm_workspace.client import" in script.read_text()
        assert obs.artifacts == []
    finally:
        env.cleanup()


def test_python_action_records_system_provenance_for_writes(tmp_path: Path) -> None:
    env = _make_env(tmp_path)
    try:
        env.setup()
        env.current_turn = 4
        action = WorkspaceAction(
            tool="python",
            args={},
            body=(
                "from pathlib import Path\n"
                "Path('_rlm_notes/py_out.txt').write_text('hi from python')\n"
            ),
            raw="",
        )
        obs = env.run_action(action)
        assert obs.error is None, (obs.error, obs.stderr)
        assert (env.workspace_root / "_rlm_notes" / "py_out.txt").exists()
        env.provenance.load()
        prov = env.provenance.get("_rlm_notes/py_out.txt")
        assert prov is not None
        assert prov.created.role == "system"
        assert prov.created.action_id == "t4.a1"
    finally:
        env.cleanup()


def test_python_action_helpers_preimported(tmp_path: Path) -> None:
    """The `llm_query` / `rlm_query` helpers must be in the script's globals
    without needing an explicit import. We don't actually call them here
    (no LM handler wired); we just verify they resolve as callables."""
    env = _make_env(tmp_path)
    try:
        env.setup()
        env.current_turn = 5
        action = WorkspaceAction(
            tool="python",
            args={},
            body=(
                "for name in ('llm_query', 'llm_query_batched', "
                "'rlm_query', 'rlm_query_batched'):\n"
                "    assert callable(globals()[name]), name\n"
                "print('helpers-ok')\n"
            ),
            raw="",
        )
        obs = env.run_action(action)
        assert obs.error is None, (obs.error, obs.stderr)
        assert "helpers-ok" in obs.stdout
    finally:
        env.cleanup()


def test_python_action_timeout_enforced(tmp_path: Path) -> None:
    cfg = WorkspaceConfig(
        observation=ObservationConfig(max_observation_chars=4_000),
        docker=DockerConfig(
            image=IMAGE_TAG,
            workspace_root_base=str(tmp_path),
            broker_port=8080,
            poll_interval_ms=50,
            exec_timeout_seconds=2,
            cleanup_mode="delete",
        ),
    )
    env = DockerWorkspaceEnv(workspace_config=cfg)
    try:
        env.setup()
        env.current_turn = 6
        action = WorkspaceAction(
            tool="python",
            args={},
            body="import time\nfor i in range(20):\n    time.sleep(1)\n",
            raw="",
        )
        obs = env.run_action(action)
        assert obs.error is not None
        assert "timeout" in obs.error.lower()
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


def test_rlm_query_end_to_end_with_mock_lm(tmp_path: Path) -> None:
    """End-to-end recursion: parent → child (mock LM) → artifact pull-back.

    Wires a ``RecursionHandler`` to a real ``DockerWorkspaceEnv`` whose
    LMHandler is backed by a ``MockLM`` returning a single ``final`` action
    with one ``<artifact>`` selection. Asserts that:
      - the child workspace was created at the sibling path,
      - the explicitly selected artifact was copied into the parent at
        ``_rlm_artifacts/children/<child_id>/<path>``,
      - parent provenance for that path is role=``child``,
      - the returned observation contains the path-mapping block.
    """
    from rlm.core.lm_handler import LMHandler
    from rlm.core.recursion import RecursionHandler
    from rlm.core.rlm import RLM
    from tests.mock_lm import MockLM

    child_response = (
        '<action tool="write_file" path="out/result.txt">child output here</action>\n'
        '<action tool="final">'
        "<answer>I solved it; see out/result.txt</answer>"
        '<artifact path="out/result.txt" />'
        "</action>"
    )

    mock = MockLM(responses=[child_response])
    lm_handler = LMHandler(mock)
    lm_handler.start()

    cfg = WorkspaceConfig(
        observation=ObservationConfig(max_observation_chars=4_000),
        docker=DockerConfig(
            image=IMAGE_TAG,
            workspace_root_base=str(tmp_path),
            broker_port=8080,
            poll_interval_ms=50,
            exec_timeout_seconds=15,
            cleanup_mode="delete",
        ),
    )
    parent_env = DockerWorkspaceEnv(
        workspace_config=cfg,
        lm_handler_address=(lm_handler.host, lm_handler.port),
        depth=0,
        max_depth=2,
    )
    try:
        parent_env.setup()
        parent_env.current_turn = 1

        # Stub parent RLM: only reads attributes, never calls completion.
        parent_rlm = RLM(
            backend="openai",  # never used; child uses lm_handler directly
            backend_kwargs={"model_name": "fake"},
            workspace_config=cfg,
            depth=0,
            max_depth=2,
            max_iterations=3,
        )
        handler = RecursionHandler(
            parent_rlm=parent_rlm, parent_env=parent_env, lm_handler=lm_handler
        )

        obs = handler.spawn(child_task="please solve subtask X", action_id="t1.a1")

        assert obs.error is None, obs.error
        assert "Answer: I solved it" in obs.stdout
        assert "Artifact Mapping:" in obs.stdout
        assert "out/result.txt -> _rlm_artifacts/children/child_1_1/out/result.txt" in obs.stdout

        # Child trajectory rides on obs.rlm_calls (visualizer drill-down path).
        # The full per-turn record is in rlm_calls[0].metadata.iterations.
        assert len(obs.rlm_calls) == 1
        child_rc = obs.rlm_calls[0]
        assert child_rc.metadata is not None
        assert "iterations" in child_rc.metadata
        assert len(child_rc.metadata["iterations"]) >= 1
        # The child emitted write_file then final, so the last iteration carries
        # the final answer.
        last = child_rc.metadata["iterations"][-1]
        assert last["final_answer"] is not None and "I solved it" in last["final_answer"]

        # Artifact actually copied into parent.
        parent_artifact = (
            parent_env.workspace_root
            / "_rlm_artifacts"
            / "children"
            / "child_1_1"
            / "out"
            / "result.txt"
        )
        assert parent_artifact.exists()
        assert parent_artifact.read_text() == "child output here"

        # Parent provenance marks it as `child` role.
        parent_env.provenance.load()
        prov = parent_env.provenance.get("_rlm_artifacts/children/child_1_1/out/result.txt")
        assert prov is not None
        assert prov.created.role == "child"
        assert prov.created.action_id == "t1.a1"
    finally:
        try:
            parent_env.cleanup()
        finally:
            lm_handler.stop()


# ---------------------------------------------------------------------------
# Visibility: broker calls from inside a python action populate
# ``observation.rlm_calls``. Closes the gap documented in Todo.md
# "Visibility follow-ups (2026-05-09)".
# ---------------------------------------------------------------------------


def _wire_recursion_handler(parent_env: DockerWorkspaceEnv, lm_handler) -> None:
    """Build a stub parent RLM and attach a RecursionHandler to ``parent_env``.

    Mirrors the pattern in ``test_rlm_query_end_to_end_with_mock_lm``: the
    parent RLM's completion path is never invoked (the LM call originates
    from the child via the shared lm_handler), so ``backend="openai"`` /
    ``model_name="fake"`` is fine.
    """
    from rlm.core.recursion import RecursionHandler
    from rlm.core.rlm import RLM

    parent_rlm = RLM(
        backend="openai",
        backend_kwargs={"model_name": "fake"},
        workspace_config=parent_env.workspace_config,
        depth=0,
        max_depth=2,
        max_iterations=3,
    )
    parent_env.recursion_handler = RecursionHandler(
        parent_rlm=parent_rlm, parent_env=parent_env, lm_handler=lm_handler
    )


def test_python_llm_query_populates_rlm_calls(tmp_path: Path) -> None:
    """A ``llm_query`` from inside a python action surfaces its
    RLMChatCompletion in ``obs.rlm_calls`` (visualizer drill-down)."""
    from rlm.core.lm_handler import LMHandler
    from tests.mock_lm import MockLM

    mock = MockLM(response_fn=lambda p: f"resp:{p}" if isinstance(p, str) else "resp:?")
    lm_handler = LMHandler(mock)
    lm_handler.start()

    env = _make_env(tmp_path, lm_handler_address=(lm_handler.host, lm_handler.port))
    try:
        env.setup()
        env.current_turn = 1
        action = WorkspaceAction(
            tool="python",
            args={},
            body=("r = llm_query('hello')\nprint(r)\n"),
            raw="",
        )
        obs = env.run_action(action)
        assert obs.error is None, (obs.error, obs.stderr)
        assert "resp:hello" in obs.stdout
        assert len(obs.rlm_calls) == 1
        assert obs.rlm_calls[0].response == "resp:hello"
        assert obs.rlm_calls[0].prompt == "hello"
        # Drained — repeat call must return empty.
        assert env.drain_broker_ledger("t1.a1") == []
    finally:
        try:
            env.cleanup()
        finally:
            lm_handler.stop()


def test_python_llm_query_batched_populates_rlm_calls(tmp_path: Path) -> None:
    """A ``llm_query_batched`` from inside python surfaces N
    RLMChatCompletions in ``obs.rlm_calls`` in input order."""
    from rlm.core.lm_handler import LMHandler
    from tests.mock_lm import MockLM

    # response_fn (not a static list) because _handle_batched fires
    # acompletions concurrently via asyncio.gather — list-pop ordering
    # is not deterministic under concurrency.
    mock = MockLM(response_fn=lambda p: f"resp:{p}" if isinstance(p, str) else "resp:?")
    lm_handler = LMHandler(mock)
    lm_handler.start()

    env = _make_env(tmp_path, lm_handler_address=(lm_handler.host, lm_handler.port))
    try:
        env.setup()
        env.current_turn = 1
        action = WorkspaceAction(
            tool="python",
            args={},
            body=("rs = llm_query_batched(['a','b','c'])\nprint('|'.join(rs))\n"),
            raw="",
        )
        obs = env.run_action(action)
        assert obs.error is None, (obs.error, obs.stderr)
        assert "resp:a|resp:b|resp:c" in obs.stdout
        assert len(obs.rlm_calls) == 3
        responses = [rc.response for rc in obs.rlm_calls]
        prompts = [rc.prompt for rc in obs.rlm_calls]
        # Order matches input prompts because _handle_batched zips
        # responses back with request.prompts (lm_handler.py:116).
        assert prompts == ["a", "b", "c"]
        assert responses == ["resp:a", "resp:b", "resp:c"]
    finally:
        try:
            env.cleanup()
        finally:
            lm_handler.stop()


def test_python_rlm_query_populates_rlm_calls_with_iterations(tmp_path: Path) -> None:
    """A ``rlm_query`` from inside python surfaces the child's full
    trajectory (``metadata.iterations``) in ``obs.rlm_calls[0]``."""
    from rlm.core.lm_handler import LMHandler
    from tests.mock_lm import MockLM

    child_response = '<action tool="final"><answer>solved-it</answer></action>'
    mock = MockLM(responses=[child_response])
    lm_handler = LMHandler(mock)
    lm_handler.start()

    env = _make_env(tmp_path, lm_handler_address=(lm_handler.host, lm_handler.port))
    try:
        env.setup()
        env.current_turn = 1
        _wire_recursion_handler(env, lm_handler)
        action = WorkspaceAction(
            tool="python",
            args={},
            body=("r = rlm_query('subtask')\nprint(r)\n"),
            raw="",
        )
        obs = env.run_action(action)
        assert obs.error is None, (obs.error, obs.stderr)
        assert "solved-it" in obs.stdout  # path-mapping observation flowed back
        assert len(obs.rlm_calls) == 1
        rc = obs.rlm_calls[0]
        assert rc.metadata is not None and "iterations" in rc.metadata
        assert len(rc.metadata["iterations"]) >= 1
        last = rc.metadata["iterations"][-1]
        assert last["final_answer"] is not None and "solved-it" in last["final_answer"]
    finally:
        try:
            env.cleanup()
        finally:
            lm_handler.stop()


def test_python_rlm_query_batched_populates_rlm_calls(tmp_path: Path) -> None:
    """The headline case from Todo.md: ``rlm_query_batched`` from inside a
    python action surfaces one RLMChatCompletion per child in
    ``obs.rlm_calls``, each carrying populated ``metadata.iterations``.

    Note: each child's first prompt to the LM is the standard
    "begin by reading _rlm_query_0.txt" message — the task body itself
    (``rlm_query_batched(["TASK_A", ...])``) lives in the child's workspace
    file, not in the prompt. So we cannot deterministically map mock
    responses to specific children. Asserting count + structure is what
    proves the visibility plumbing without a multi-turn child trajectory."""
    from rlm.core.lm_handler import LMHandler
    from tests.mock_lm import MockLM

    def respond(prompt):  # noqa: ARG001 — same response per child
        return '<action tool="final"><answer>child-done</answer></action>'

    mock = MockLM(response_fn=respond)
    lm_handler = LMHandler(mock)
    lm_handler.start()

    env = _make_env(tmp_path, lm_handler_address=(lm_handler.host, lm_handler.port))
    try:
        env.setup()
        env.current_turn = 1
        _wire_recursion_handler(env, lm_handler)
        action = WorkspaceAction(
            tool="python",
            args={},
            body=(
                "rs = rlm_query_batched(['task-1', 'task-2'])\n"
                "print('count=' + str(len(rs)))\n"
                "for r in rs:\n    print('---')\n    print(r)\n"
            ),
            raw="",
        )
        obs = env.run_action(action)
        assert obs.error is None, (obs.error, obs.stderr)
        assert "count=2" in obs.stdout
        # Both children's answers flow back through the python tool's stdout
        # via the path-mapping observations.
        assert obs.stdout.count("child-done") == 2
        # Visibility plumbing: one RLMChatCompletion per child, each with a
        # populated trajectory.
        assert len(obs.rlm_calls) == 2
        for rc in obs.rlm_calls:
            assert rc.metadata is not None and "iterations" in rc.metadata
            assert len(rc.metadata["iterations"]) >= 1
            last = rc.metadata["iterations"][-1]
            assert last["final_answer"] == "child-done"
        # Drained — repeat call must return empty.
        assert env.drain_broker_ledger("t1.a1") == []
    finally:
        try:
            env.cleanup()
        finally:
            lm_handler.stop()


# ---------------------------------------------------------------------------
# Multi-turn snapshot SHAs are distinct + ordered
# ---------------------------------------------------------------------------


def test_multi_turn_snapshots_are_distinct_and_ordered(tmp_path: Path) -> None:
    """Three turns each produce a distinct git commit SHA, and the visible
    git log lists them parent-chained newest-first."""
    env = _make_env(tmp_path)
    try:
        env.setup()
        shas: list[str] = []
        for turn in (1, 2, 3):
            env.current_turn = turn
            env.run_action(
                WorkspaceAction(
                    tool="write_file",
                    args={"path": f"_rlm_notes/turn_{turn}.txt"},
                    body=f"turn {turn} content",
                    raw="",
                )
            )
            snap = env.snapshot(turn=turn)
            shas.append(snap.commit_sha)

        # All three SHAs are distinct.
        assert len(set(shas)) == 3, f"expected 3 distinct SHAs; got {shas}"

        # `git log` newest-first lists "turn 3", "turn 2", "turn 1", "turn 0".
        log = (
            subprocess.run(
                ["git", "-C", str(env.workspace_root), "log", "--format=%s"],
                capture_output=True,
                text=True,
                check=True,
            )
            .stdout.strip()
            .splitlines()
        )
        assert log[:4] == ["turn 3", "turn 2", "turn 1", "turn 0"]
    finally:
        env.cleanup()


# ---------------------------------------------------------------------------
# Max-depth: rlm_query at the depth ceiling returns the loud error
# ---------------------------------------------------------------------------


def test_max_depth_rlm_query_returns_loud_error(tmp_path: Path) -> None:
    """An ``<action tool="rlm_query">`` emitted from a parent already at
    ``depth == max_depth`` must produce a structured error observation,
    not silently spawn a child. ``RecursionHandler`` is intentionally not
    wired in this scenario; the action falls through to the
    handler-is-None branch in ``rlm_query.execute``."""
    env = _make_env(tmp_path, depth=2, max_depth=2)
    # No recursion handler wired (RLM._wire_recursion would normally do this,
    # but we skip wiring at depth==max_depth, mirroring production).
    assert env.recursion_handler is None
    try:
        env.setup()
        env.current_turn = 1
        action = WorkspaceAction(
            tool="rlm_query",
            args={},
            body="please solve subtask",
            raw='<action tool="rlm_query">please solve subtask</action>',
        )
        obs = env.run_action(action)
        assert obs.error is not None
        assert "Maximum recursion depth" in obs.error
        assert "rlm_query" in obs.error
    finally:
        env.cleanup()


# ---------------------------------------------------------------------------
# In-container python helpers reject a `model=` kwarg
# ---------------------------------------------------------------------------


def test_python_llm_query_rejects_model_kwarg(tmp_path: Path) -> None:
    """The in-container ``llm_query`` helper takes no ``model=`` argument —
    the model is configured once on the host. Calling
    ``llm_query("hi", model="x")`` from inside a python action must surface
    a ``TypeError`` (caught and printed by the script, or visible in stderr
    with a non-zero exit code).

    This locks in the design-doc invariant from
    ``workspace_substrate_arch/02_core_architecture.md`` ("no ``model=``
    argument — the parent's configured model is used everywhere")."""
    from rlm.core.lm_handler import LMHandler
    from tests.mock_lm import MockLM

    mock = MockLM(response_fn=lambda p: "echo")
    lm_handler = LMHandler(mock)
    lm_handler.start()
    env = _make_env(tmp_path, lm_handler_address=(lm_handler.host, lm_handler.port))
    try:
        env.setup()
        env.current_turn = 1
        action = WorkspaceAction(
            tool="python",
            args={},
            body=(
                "try:\n"
                "    llm_query('hi', model='gpt-4o')\n"
                "    print('UNEXPECTED-NO-ERROR')\n"
                "except TypeError as e:\n"
                "    print(f'GOT-TYPEERROR: {e}')\n"
            ),
            raw="",
        )
        obs = env.run_action(action)
        printed_typeerror = "GOT-TYPEERROR" in obs.stdout
        stderr_typeerror = (
            "TypeError" in obs.stderr and obs.data is not None and obs.data.get("exit_code") != 0
        )
        assert printed_typeerror or stderr_typeerror, (
            f"Expected TypeError when calling llm_query(..., model=...); "
            f"got stdout={obs.stdout!r} stderr={obs.stderr!r}"
        )
    finally:
        try:
            env.cleanup()
        finally:
            lm_handler.stop()
