"""Unit tests for the provenance store (direct-update model)."""

from pathlib import Path

import pytest

from rlm.utils.provenance import (
    ProvenanceStore,
    diff_snapshots,
    snapshot_paths,
)


@pytest.fixture
def store(tmp_path: Path) -> ProvenanceStore:
    return ProvenanceStore(tmp_path / "_rlm_state" / "provenance.json")


def test_record_write_creates_and_modifies(store):
    store.record_write("a.txt", role="assistant", action_id="t1.a1", turn=1)
    p = store.get("a.txt")
    assert p is not None
    assert p.created.role == "assistant"
    assert p.modified.role == "assistant"
    assert p.created.action_id == "t1.a1"


def test_second_write_only_updates_modified(store):
    store.record_write("a.txt", role="assistant", action_id="t1.a1", turn=1)
    store.record_write("a.txt", role="system", action_id="t2.a3", turn=2)
    p = store.get("a.txt")
    assert p.created.role == "assistant"
    assert p.created.action_id == "t1.a1"
    assert p.modified.role == "system"
    assert p.modified.action_id == "t2.a3"
    assert p.modified.turn == 2


def test_record_seed_overrides_created(store):
    store.record_write("a.txt", role="assistant", action_id="t1.a1", turn=1)
    store.record_seed("a.txt", role="user", action_id=None, turn=0)
    p = store.get("a.txt")
    assert p.created.role == "user"
    assert p.modified.role == "user"


def test_remove(store):
    store.record_write("a.txt", role="assistant", action_id="t1.a1", turn=1)
    store.remove("a.txt")
    assert "a.txt" not in store


def test_persistence_roundtrip(store, tmp_path):
    store.record_write("a.txt", role="assistant", action_id="t1.a1", turn=1)
    store.record_write("dir/b.txt", role="system", action_id="t2.a1", turn=2)
    store.save()
    other = ProvenanceStore(tmp_path / "_rlm_state" / "provenance.json")
    other.load()
    assert "a.txt" in other
    assert other.get("dir/b.txt").modified.role == "system"


def test_path_normalization(store):
    store.record_write("./a.txt", role="assistant", action_id="t1.a1", turn=1)
    assert "a.txt" in store
    store.record_write("dir\\b.txt", role="assistant", action_id="t1.a2", turn=1)
    assert "dir/b.txt" in store


# ---------------------------------------------------------------------------
# Filesystem snapshot helper for shell/python
# ---------------------------------------------------------------------------


def test_snapshot_and_diff_detect_new_and_modified(tmp_path: Path):
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("world")
    before = snapshot_paths(tmp_path)
    assert "a.txt" in before
    assert "sub/b.txt" in before

    # Modify and add
    (tmp_path / "a.txt").write_text("hello world!")
    (tmp_path / "c.txt").write_text("new")
    after = snapshot_paths(tmp_path)
    changed, removed = diff_snapshots(before, after)
    assert "a.txt" in changed
    assert "c.txt" in changed
    assert removed == []


def test_snapshot_excludes_apply(tmp_path: Path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("x")
    (tmp_path / "a.txt").write_text("y")
    snap = snapshot_paths(tmp_path, excludes=(".git",))
    assert "a.txt" in snap
    assert ".git/config" not in snap


def test_snapshot_detects_removal(tmp_path: Path):
    (tmp_path / "a.txt").write_text("x")
    before = snapshot_paths(tmp_path)
    (tmp_path / "a.txt").unlink()
    after = snapshot_paths(tmp_path)
    changed, removed = diff_snapshots(before, after)
    assert removed == ["a.txt"]
    assert changed == []
