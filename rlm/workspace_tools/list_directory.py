"""list_directory — shallow listing of a workspace directory."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from rlm.core.types import WorkspaceAction, WorkspaceObservation
from rlm.workspace_tools import ToolSpec

if TYPE_CHECKING:
    from rlm.environments.docker_workspace import DockerWorkspaceEnv

SPEC = ToolSpec(
    name="list_directory",
    short_description="Shallow listing of a directory in the workspace.",
    is_state_mutating=False,
    runs_on="host",
    body_required=False,
    example=(
        'No required attrs. Optional: path (workspace-relative; defaults to workspace root). '
        'Example: <action tool="list_directory" path="_rlm_artifacts/"/>'
    ),
)

def _to_absolute(rel: str) -> str:
    """Canonical container-absolute form for a workspace-relative path.

    ``""`` / ``"."`` → ``"/"``; anything else gets a leading slash.
    """
    if rel in ("", "."):
        return "/"
    return "/" + rel.lstrip("/")


def execute(env: DockerWorkspaceEnv, action: WorkspaceAction) -> WorkspaceObservation:
    start = time.perf_counter()
    raw = action.args.get("path", ".")
    target = env.resolve_workspace_path(raw)
    if not target.exists():
        return WorkspaceObservation(
            tool=SPEC.name,
            error=f"Path does not exist: {raw}",
            execution_time=time.perf_counter() - start,
        )
    if not target.is_dir():
        return WorkspaceObservation(
            tool=SPEC.name,
            error=f"Not a directory: {raw}",
            execution_time=time.perf_counter() - start,
        )

    # Canonicalise the header regardless of how the model addressed the dir
    # (``.``, ``app``, ``/app`` all render as ``/app``) so the model converges
    # on the absolute form it sees in the output.
    target_rel = str(target.relative_to(env.workspace_root.resolve())).replace("\\", "/")
    target_abs = _to_absolute(target_rel)

    cap = env.workspace_config.observation.max_list_directory_entries
    # Single source of truth: reuse the copy-on-spawn exclude list. Multi-
    # segment entries (e.g. "_rlm_state/snapshots") are filtered out because
    # this is a shallow listing — at the basename level only top-level
    # directory names matter.
    ignore_basenames = {
        e for e in env.workspace_config.recursion.copy_on_spawn_excludes if "/" not in e
    }
    entries: list[dict] = []
    truncated = False
    for child in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name)):
        if child.name in ignore_basenames:
            continue
        if len(entries) >= cap:
            truncated = True
            break
        rel = str(child.relative_to(env.workspace_root)).replace("\\", "/")
        kind = "dir" if child.is_dir() else "file"
        size: int | None = None
        if child.is_file():
            try:
                size = child.stat().st_size
            except OSError:
                size = None
        prov = env.provenance.get(rel)
        created_role = prov.created.role if prov else "user"
        modified_role = prov.modified.role if prov else "user"
        entries.append(
            {
                "name": child.name,
                "path": rel,
                "kind": kind,
                "size": size,
                "created_role": created_role,
                "modified_role": modified_role,
            }
        )

    # Column widths: line up the role columns regardless of name length.
    name_col_width = (
        max(len(_to_absolute(e["path"])) + (1 if e["kind"] == "dir" else 0) for e in entries)
        if entries
        else 0
    )
    lines = [f"Directory: {target_abs}"]
    for e in entries:
        display = _to_absolute(e["path"]) + ("/" if e["kind"] == "dir" else "")
        kind_tag = "[dir] " if e["kind"] == "dir" else "[file]"
        size_part = "" if e["kind"] == "dir" else f"{e['size']}B"
        lines.append(
            f"  {kind_tag} {display.ljust(name_col_width)}  "
            f"{size_part:>8}  "
            f"created={e['created_role']}  modified={e['modified_role']}"
        )
    if not entries and not truncated:
        lines.append("  [empty directory]")
    if truncated:
        lines.append(f"... [truncated at {cap} entries]")

    return WorkspaceObservation(
        tool=SPEC.name,
        stdout="\n".join(lines),
        data={"entries": entries, "truncated": truncated},
        execution_time=time.perf_counter() - start,
    )
