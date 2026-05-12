"""Shared test helpers.

- ``make_thin_env(tmp_path)`` returns a non-Docker ``DockerWorkspaceEnv`` ready
  for in-process tool tests: the workspace directory and reserved layout are
  created on disk, the provenance store is initialized, but ``setup()`` (which
  starts the container + poller) is intentionally not called.

- ``schema_of_jsonl(text)`` reduces each JSONL line to its structural schema
  (keys + value types, with list elements merged) so the visualizer-schema
  golden test only fails on real shape changes — not on prompt copy edits.
"""

from __future__ import annotations

import json
from pathlib import Path

from rlm.core.config import WorkspaceConfig
from rlm.environments.docker_workspace import DockerWorkspaceEnv


def make_thin_env(
    tmp_path: Path, workspace_config: WorkspaceConfig | None = None
) -> DockerWorkspaceEnv:
    """Build a workspace env with the on-disk layout but no container.

    This mirrors what ``DockerWorkspaceEnv.setup()`` does for the parts the
    in-process tools depend on (dir layout, provenance seeding, action_log,
    manifest, git init), then leaves the container/poller unstarted. Tools
    that ``runs_on="host"`` (read_file, write_file, append_file, edit_file,
    list_directory, final) work fine against this env.
    """
    cfg = workspace_config or WorkspaceConfig()
    ws_root = tmp_path / "ws"
    env = DockerWorkspaceEnv(
        workspace_config=cfg,
        lm_handler_address=None,
        run_id="test-run",
        depth=0,
        max_depth=1,
        workspace_root=ws_root,
    )
    # Use the same seeding logic as setup() but skip _start_container/_start_poller.
    env._create_workspace_dirs()
    env._seed_provenance_and_manifest()
    env._git_init()
    env.current_turn = 1
    env.current_action_id = "t1.a1"
    return env


def _schema_of(node):
    """Recursively reduce a JSON value to a stable type/shape representation.

    Scalars become ``"<str>"`` / ``"<int>"`` / ``"<float>"`` / ``"<bool>"`` /
    ``"<null>"``. Lists become a single-element list whose element is the
    merged schema of every input element (so list length and content vary
    across runs without breaking the golden). Dicts keep their keys and
    recurse into values.
    """
    if isinstance(node, dict):
        return {k: _schema_of(v) for k, v in node.items()}
    if isinstance(node, list):
        if not node:
            return []
        merged = _schema_of(node[0])
        for item in node[1:]:
            merged = _merge_schemas(merged, _schema_of(item))
        return [merged]
    if node is None:
        return "<null>"
    if isinstance(node, bool):
        return "<bool>"
    if isinstance(node, int):
        return "<int>"
    if isinstance(node, float):
        return "<float>"
    return "<str>"


def _merge_schemas(a, b):
    """Union two schemas: dicts merge keys, lists merge elements; mismatches
    fall back to a deterministic ``"a|b"`` tag so divergence is visible
    rather than silently hidden by first-wins.
    """
    if isinstance(a, dict) and isinstance(b, dict):
        out = dict(a)
        for k, v in b.items():
            out[k] = _merge_schemas(out[k], v) if k in out else v
        return out
    if isinstance(a, list) and isinstance(b, list):
        if not a:
            return b
        if not b:
            return a
        return [_merge_schemas(a[0], b[0])]
    if a == b:
        return a
    return "|".join(sorted({str(a), str(b)}))


def schema_of_jsonl(text: str) -> str:
    """Reduce a JSONL stream to one schema-tagged line per record.

    Each line is parsed, run through ``_schema_of``, and re-serialized with
    sorted keys. The result fails only on shape changes — added/removed
    keys or changed value types — and is invariant to scalar content,
    list lengths, timestamps, hashes, or rendered prompt text.
    """
    out_lines: list[str] = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        obj = json.loads(raw)
        out_lines.append(json.dumps(_schema_of(obj), sort_keys=True))
    return "\n".join(out_lines) + "\n"
