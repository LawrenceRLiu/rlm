"""
Provenance store for the workspace substrate.

A small JSON sidecar living at ``<workspace>/_rlm_state/provenance.json`` that
maps each path in the workspace to ``{created, modified}`` records. Each record
carries the role (`user` / `assistant` / `system` / `child`), the action_id
(e.g. ``t3.a1``) that touched it, and the turn number.

Update model
------------
Direct, per-tool updates — *no* mtime/bracketing. Tools that know the path they
write call ``record_write(...)``. For shell/python (which can touch arbitrary
files), the env walks the workspace before and after the call comparing path
sets and sizes; new or changed paths are passed to ``record_writes(...)`` with
role=``system``.

Roles are decided by the *caller*, not inferred here, so the store stays a dumb
mapping. The role classification per tool is documented in the plan and lives
in the tool implementations.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

ProvenanceRole = Literal["user", "assistant", "system", "child"]


@dataclass
class ProvenanceEntry:
    """One ``created`` or ``modified`` record."""

    role: ProvenanceRole
    action_id: str | None  # e.g. "t3.a1"; None for files seeded before any action
    turn: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ProvenanceEntry:
        return cls(role=d["role"], action_id=d.get("action_id"), turn=int(d.get("turn", 0)))


@dataclass
class FileProvenance:
    """Per-file provenance: who created it, who last touched it."""

    created: ProvenanceEntry
    modified: ProvenanceEntry

    def to_dict(self) -> dict:
        return {"created": self.created.to_dict(), "modified": self.modified.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> FileProvenance:
        return cls(
            created=ProvenanceEntry.from_dict(d["created"]),
            modified=ProvenanceEntry.from_dict(d["modified"]),
        )


class ProvenanceStore:
    """Path -> FileProvenance map, persisted to a JSON sidecar."""

    def __init__(self, store_path: Path):
        self.store_path = Path(store_path)
        self._entries: dict[str, FileProvenance] = {}

    # -- persistence -------------------------------------------------------
    def load(self) -> None:
        if not self.store_path.exists():
            self._entries = {}
            return
        data = json.loads(self.store_path.read_text())
        self._entries = {p: FileProvenance.from_dict(d) for p, d in data.items()}

    def save(self) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        out = {p: prov.to_dict() for p, prov in self._entries.items()}
        self.store_path.write_text(json.dumps(out, indent=2, sort_keys=True))

    # -- queries -----------------------------------------------------------
    def get(self, path: str) -> FileProvenance | None:
        return self._entries.get(self._normalize(path))

    def __contains__(self, path: str) -> bool:
        return self._normalize(path) in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def all_paths(self) -> list[str]:
        return sorted(self._entries.keys())

    # -- mutations ---------------------------------------------------------
    def record_write(
        self,
        path: str,
        role: ProvenanceRole,
        action_id: str | None,
        turn: int,
    ) -> None:
        """Record that ``path`` was written by ``role`` on ``turn``.

        If the path is new to the store, both ``created`` and ``modified`` are
        set. Otherwise only ``modified`` is updated. Callers that explicitly
        want to override ``created`` (e.g., child workspace seeding) should use
        ``record_seed``.
        """
        key = self._normalize(path)
        entry = ProvenanceEntry(role=role, action_id=action_id, turn=turn)
        existing = self._entries.get(key)
        if existing is None:
            self._entries[key] = FileProvenance(created=entry, modified=entry)
        else:
            self._entries[key] = FileProvenance(created=existing.created, modified=entry)

    def record_writes(
        self,
        paths: Iterable[str],
        role: ProvenanceRole,
        action_id: str | None,
        turn: int,
    ) -> None:
        for p in paths:
            self.record_write(p, role=role, action_id=action_id, turn=turn)

    def record_seed(
        self,
        path: str,
        role: ProvenanceRole,
        action_id: str | None,
        turn: int,
    ) -> None:
        """Force ``created == modified`` for ``path``. Used when the runtime
        seeds a workspace (root task, user context, child copy-on-spawn).
        """
        key = self._normalize(path)
        entry = ProvenanceEntry(role=role, action_id=action_id, turn=turn)
        self._entries[key] = FileProvenance(created=entry, modified=entry)

    def remove(self, path: str) -> None:
        self._entries.pop(self._normalize(path), None)

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _normalize(path: str) -> str:
        # Store paths workspace-relative, with forward slashes, no leading "./".
        s = str(path).replace("\\", "/")
        if s.startswith("./"):
            s = s[2:]
        return s


# ---------------------------------------------------------------------------
# Filesystem-walk helper for shell/python (no mtime — path + size only)
# ---------------------------------------------------------------------------


def snapshot_paths(root: Path, excludes: tuple[str, ...] = ()) -> dict[str, int]:
    """Return ``{rel_path: size_bytes}`` for every regular file under ``root``.

    ``excludes`` is a tuple of top-level directory names to skip (e.g.
    ``(".git", "__pycache__")``).
    """
    root = Path(root)
    out: dict[str, int] = {}
    for p in root.rglob("*"):
        # Skip excluded top-level dirs.
        try:
            rel = p.relative_to(root)
        except ValueError:
            continue
        rel_str = str(rel).replace("\\", "/")
        if any(rel_str == e or rel_str.startswith(e + "/") for e in excludes):
            continue
        if p.is_file():
            try:
                out[rel_str] = p.stat().st_size
            except OSError:
                continue
    return out


def diff_snapshots(before: dict[str, int], after: dict[str, int]) -> tuple[list[str], list[str]]:
    """Return ``(created_or_modified, removed)`` path lists.

    A path is in ``created_or_modified`` if it is new in ``after`` or its size
    differs from ``before``. ``removed`` paths are in ``before`` but not in
    ``after``. Sizes equal → unchanged (best-effort; coalesces no-op writes).
    """
    changed: list[str] = []
    for path, size in after.items():
        if before.get(path) != size:
            changed.append(path)
    removed = [p for p in before if p not in after]
    return sorted(changed), sorted(removed)
