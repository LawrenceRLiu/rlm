"""Shared test helpers.

- ``make_thin_env(tmp_path)`` returns a non-Docker ``DockerWorkspaceEnv`` ready
  for in-process tool tests: the workspace directory and reserved layout are
  created on disk, the provenance store is initialized, but ``setup()`` (which
  starts the container + poller) is intentionally not called.

- ``normalize_jsonl(text)`` strips run-specific values (timestamps, SHAs, run
  ids, file paths, durations, uuid spill names) so two rollouts produced by
  the same scripted ``MockLM`` compare byte-for-byte against a golden file.
"""

from __future__ import annotations

import json
import re
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


_TIMESTAMP_RE = re.compile(r'"timestamp"\s*:\s*"[^"]+"')
_SHA_RE = re.compile(r'"commit_sha"\s*:\s*"[a-f0-9]{7,64}"')
_RUN_ID_RE = re.compile(r"run_\d+_[a-f0-9]+")
_WS_PATH_RE = re.compile(r'"workspace_root"\s*:\s*"[^"]+"')
_DURATION_RE = re.compile(r'"(execution_time|iteration_time)"\s*:\s*[0-9.eE+-]+')
_SPILL_ID_RE = re.compile(r"_observations/[a-z0-9._-]+\.txt")
# Abbreviated SHA inside a message content (the snapshot rendering inserts
# ``commit="<7-hex>"`` into the user-message string included in prompt history).
# In the on-disk JSONL this appears with the inner double-quotes escaped as ``\"``.
_INNER_SHA_RE = re.compile(r'commit=\\"[a-f0-9]{7,64}\\"')


def normalize_jsonl(text: str) -> str:
    """Strip run-varying values so a JSONL run is reproducible across executions.

    Replaces timestamps, commit SHAs, run ids, workspace paths, durations, and
    spill-file names with stable placeholders. Each line is round-tripped
    through ``json.dumps(..., sort_keys=True)`` so key ordering is stable too.
    """
    out_lines: list[str] = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        s = raw
        s = _TIMESTAMP_RE.sub('"timestamp":"<TS>"', s)
        s = _SHA_RE.sub('"commit_sha":"<SHA>"', s)
        s = _RUN_ID_RE.sub("<RUN_ID>", s)
        s = _WS_PATH_RE.sub('"workspace_root":"<WS>"', s)
        s = _DURATION_RE.sub(lambda m: f'"{m.group(1)}":<T>', s)
        s = _SPILL_ID_RE.sub("_observations/<SPILL>.txt", s)
        s = _INNER_SHA_RE.sub(r'commit=\\"<INNER_SHA>\\"', s)
        # Re-serialize for stable key ordering.
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            out_lines.append(s)
            continue
        out_lines.append(json.dumps(obj, sort_keys=True))
    return "\n".join(out_lines) + "\n"
