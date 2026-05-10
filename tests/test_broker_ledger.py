"""Unit tests for ``DockerWorkspaceEnv``'s broker ledger.

These do NOT require Docker: they exercise ``_append_broker_ledger`` and
``drain_broker_ledger`` directly without calling ``setup()``. The ledger is
the synchronization point between broker-worker threads (which produce
``RLMChatCompletion``s when a python action calls ``llm_query`` /
``rlm_query`` / batched variants) and the python tool (which drains entries
into ``observation.rlm_calls``).
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from rlm.core.config import DockerConfig, ObservationConfig, WorkspaceConfig
from rlm.core.types import RLMChatCompletion, UsageSummary
from rlm.environments.docker_workspace import DockerWorkspaceEnv


def _make_unbooted_env(tmp_path: Path) -> DockerWorkspaceEnv:
    cfg = WorkspaceConfig(
        observation=ObservationConfig(max_observation_chars=4_000),
        docker=DockerConfig(
            image="rlm-workspace:0.1.0",
            workspace_root_base=str(tmp_path),
            broker_port=8080,
            poll_interval_ms=50,
            exec_timeout_seconds=10,
            cleanup_mode="delete",
        ),
    )
    # Note: setup() is intentionally NOT called — we don't need a container
    # to exercise the ledger primitives.
    return DockerWorkspaceEnv(workspace_config=cfg)


def _fake_completion(tag: str) -> RLMChatCompletion:
    return RLMChatCompletion(
        root_model="mock",
        prompt=f"prompt-{tag}",
        response=f"resp-{tag}",
        usage_summary=UsageSummary(model_usage_summaries={}),
        execution_time=0.0,
    )


def test_append_then_drain_returns_entry(tmp_path: Path) -> None:
    env = _make_unbooted_env(tmp_path)
    rc = _fake_completion("a")
    env._append_broker_ledger("t1.a1", [rc])
    drained = env.drain_broker_ledger("t1.a1")
    assert drained == [rc]
    # Second drain returns empty (entry was popped).
    assert env.drain_broker_ledger("t1.a1") == []


def test_append_with_none_action_id_is_noop(tmp_path: Path) -> None:
    env = _make_unbooted_env(tmp_path)
    env._append_broker_ledger(None, [_fake_completion("x")])
    assert env._broker_ledger == {}
    assert env.drain_broker_ledger(None) == []


def test_append_empty_completions_does_not_create_bucket(tmp_path: Path) -> None:
    env = _make_unbooted_env(tmp_path)
    env._append_broker_ledger("t1.a1", [])
    # No empty bucket created — avoids ledger growth from no-op handlers.
    assert "t1.a1" not in env._broker_ledger
    assert env.drain_broker_ledger("t1.a1") == []


def test_drain_unknown_action_id_returns_empty(tmp_path: Path) -> None:
    env = _make_unbooted_env(tmp_path)
    assert env.drain_broker_ledger("nonexistent") == []


def test_independent_action_ids_drain_independently(tmp_path: Path) -> None:
    env = _make_unbooted_env(tmp_path)
    a1, a2, b1 = _fake_completion("a1"), _fake_completion("a2"), _fake_completion("b1")
    env._append_broker_ledger("t1.a1", [a1])
    env._append_broker_ledger("t1.a2", [b1])
    env._append_broker_ledger("t1.a1", [a2])  # interleaved second append for a1
    assert env.drain_broker_ledger("t1.a1") == [a1, a2]
    assert env.drain_broker_ledger("t1.a2") == [b1]
    # Both buckets gone after drain.
    assert env._broker_ledger == {}


def test_concurrent_appends_are_thread_safe(tmp_path: Path) -> None:
    """N threads appending under the same action_id all land safely."""
    env = _make_unbooted_env(tmp_path)
    n_threads = 16
    per_thread_appends = 8
    barrier = threading.Barrier(n_threads)

    def worker(tid: int) -> None:
        barrier.wait()  # maximize append contention
        for j in range(per_thread_appends):
            env._append_broker_ledger("t1.a1", [_fake_completion(f"{tid}-{j}")])

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    drained = env.drain_broker_ledger("t1.a1")
    assert len(drained) == n_threads * per_thread_appends
    # Every produced completion is present (ordering not guaranteed).
    expected_tags = {f"resp-{i}-{j}" for i in range(n_threads) for j in range(per_thread_appends)}
    assert {rc.response for rc in drained} == expected_tags


def test_append_after_drain_creates_new_bucket(tmp_path: Path) -> None:
    """Pathological-but-bounded: a late broker worker appending after
    ``python.py`` already drained creates a fresh bucket. This is the
    orphan case documented in the plan; it cannot collide with a future
    drain because action_ids are unique within a run."""
    env = _make_unbooted_env(tmp_path)
    env._append_broker_ledger("t1.a1", [_fake_completion("first")])
    env.drain_broker_ledger("t1.a1")
    env._append_broker_ledger("t1.a1", [_fake_completion("orphan")])
    assert env._broker_ledger.get("t1.a1") is not None
    assert len(env._broker_ledger["t1.a1"]) == 1


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
