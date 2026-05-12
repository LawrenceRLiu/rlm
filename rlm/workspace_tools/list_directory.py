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

_DEFAULT_IGNORES = {".git", "__pycache__", "node_modules", ".venv"}


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

    cap = env.workspace_config.observation.max_list_directory_entries
    entries: list[dict] = []
    truncated = False
    for child in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name)):
        if child.name in _DEFAULT_IGNORES:
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

    lines = [f"Directory: {raw}"]
    for e in entries:
        if e["kind"] == "dir":
            lines.append(f"  [dir]  {e['name']}/  ({e['created_role']}/{e['modified_role']})")
        else:
            lines.append(
                f"  [file] {e['name']}  ({e['size']} bytes, "
                f"{e['created_role']}/{e['modified_role']})"
            )
    if truncated:
        lines.append(f"... [truncated at {cap} entries]")

    return WorkspaceObservation(
        tool=SPEC.name,
        stdout="\n".join(lines),
        data={"entries": entries, "truncated": truncated},
        execution_time=time.perf_counter() - start,
    )
