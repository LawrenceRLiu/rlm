"""
Docker-backed workspace environment.

End-to-end responsibilities:

- Provision a workspace directory at ``<docker.workspace_root_base>/<run_id>/``
  with the reserved layout (``_rlm_query_0.txt``, ``_rlm_notes/``,
  ``_rlm_artifacts/``, ``_rlm_state/``) and seed ``provenance.json``.
- ``git init`` the workspace and commit a "turn 0" baseline. Per-turn
  ``snapshot()`` calls produce one git commit per turn.
- Start a container running the workspace image (default ``rlm-workspace:0.1.0``)
  with ``-v <ws>:/workspace -p 0:<broker_port>``. The container's broker
  binds 0.0.0.0:8080; the host poller drains ``/pending`` and forwards LM
  requests to the on-host ``LMHandler`` via ``send_lm_request``.
- Dispatch ``WorkspaceAction``s through the tool registry. Per-call
  observation bodies above ``observation.max_observation_chars`` are spilled
  to ``_rlm_artifacts/_observations/<id>.txt`` and replaced with a summary.
- Maintain the ``provenance.json`` sidecar via direct per-tool updates (set
  by the tool modules themselves); ``shell``/``python`` use
  ``snapshot_paths_for_provenance`` + ``diff_paths_for_provenance`` to find
  paths they touched without knowing them in advance.

The env is designed to be reusable as a child workspace for ``rlm_query``:
when ``workspace_root`` is supplied, no run_id-derived path is computed and
the caller is expected to have already populated the directory (copy-on-spawn).
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from rlm.core.comms_utils import LMRequest, send_lm_request, send_lm_request_batched
from rlm.core.config import WorkspaceConfig
from rlm.core.types import WorkspaceAction, WorkspaceObservation, WorkspaceSnapshot
from rlm.environments.base_workspace import BaseWorkspaceEnv
from rlm.utils.provenance import ProvenanceStore, diff_snapshots, snapshot_paths
from rlm.workspace_tools import get_executor, get_spec

log = logging.getLogger(__name__)

# Reserved name prefix the model is forbidden to write to (state lives here).
RESERVED_STATE_DIR = "_rlm_state"


@dataclass
class ExecResult:
    """Result of a ``docker exec`` invocation."""

    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    duration: float


def _generate_run_id() -> str:
    return f"run_{int(time.time() * 1000):d}_{uuid.uuid4().hex[:8]}"


class DockerWorkspaceEnv(BaseWorkspaceEnv):
    """Workspace env backed by a Docker container + a host-side poller."""

    def __init__(
        self,
        workspace_config: WorkspaceConfig | None = None,
        lm_handler_address: tuple[str, int] | None = None,
        run_id: str | None = None,
        depth: int = 0,
        max_depth: int = 1,
        workspace_root: Path | str | None = None,
    ) -> None:
        self.workspace_config = workspace_config or WorkspaceConfig()
        self.lm_handler_address = lm_handler_address
        self.run_id = run_id or _generate_run_id()
        self.depth = depth
        self.max_depth = max_depth

        if workspace_root is not None:
            self.workspace_root = Path(workspace_root).expanduser().resolve()
        else:
            base = Path(self.workspace_config.docker.workspace_root_base).expanduser()
            self.workspace_root = (base / self.run_id).resolve()

        self.provenance = ProvenanceStore(
            self.workspace_root / RESERVED_STATE_DIR / "provenance.json"
        )
        self.current_action_id: str | None = None
        self.current_turn: int = 0

        # Phase 6 wires this. Phase 1 stub returns a max-depth error when None.
        self.recursion_handler: Any = None

        self._container_id: str | None = None
        self._broker_host_port: int | None = None
        self._poller_thread: threading.Thread | None = None
        self._poller_stop = threading.Event()
        self._action_seq_per_turn: dict[int, int] = {}

        self._action_log_path = self.workspace_root / RESERVED_STATE_DIR / "action_log.jsonl"
        self._manifest_path = self.workspace_root / RESERVED_STATE_DIR / "workspace_manifest.json"

        # Has setup() run? Avoids re-init on context-manager nesting.
        self._is_setup = False

    # =========================================================================
    # Path helpers / contract used by tool modules
    # =========================================================================

    def is_reserved_path(self, rel: str) -> bool:
        """``True`` for paths the model is forbidden to write (``_rlm_state/...``)."""
        norm = str(rel).replace("\\", "/").lstrip("./")
        return norm == RESERVED_STATE_DIR or norm.startswith(RESERVED_STATE_DIR + "/")

    def resolve_workspace_path(self, rel: str) -> Path:
        """Join ``rel`` to ``workspace_root``, blocking absolute paths and traversal."""
        rel_path = Path(rel)
        if rel_path.is_absolute():
            raise ValueError(f"Path must be workspace-relative: {rel!r}")
        full = (self.workspace_root / rel_path).resolve()
        ws_resolved = self.workspace_root.resolve()
        try:
            full.relative_to(ws_resolved)
        except ValueError as e:
            raise ValueError(f"Path escapes workspace: {rel!r}") from e
        return full

    def snapshot_paths_for_provenance(self, excludes: tuple[str, ...]) -> dict[str, int]:
        """Path -> size walk of the workspace, used to bracket ``shell``/``python``."""
        # Always exclude the state dir from provenance diffing — the env writes
        # there itself and we don't want shell/python to be blamed for state
        # changes that happen alongside their execution.
        full_excludes = tuple(set(excludes) | {RESERVED_STATE_DIR})
        return snapshot_paths(self.workspace_root, excludes=full_excludes)

    def diff_paths_for_provenance(
        self, before: dict[str, int], after: dict[str, int]
    ) -> tuple[list[str], list[str]]:
        return diff_snapshots(before, after)

    # =========================================================================
    # Setup
    # =========================================================================

    def setup(self) -> None:
        if self._is_setup:
            return
        self._create_workspace_dirs()
        self._seed_provenance_and_manifest()
        self._git_init()
        self._start_container()
        self._start_poller()
        self._is_setup = True

    def _create_workspace_dirs(self) -> None:
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        (self.workspace_root / "_rlm_notes").mkdir(exist_ok=True)
        (self.workspace_root / "_rlm_artifacts").mkdir(exist_ok=True)
        (self.workspace_root / "_rlm_artifacts" / "_observations").mkdir(exist_ok=True)
        (self.workspace_root / RESERVED_STATE_DIR).mkdir(exist_ok=True)
        # _rlm_query_0.txt is created at load_context() time; until then we
        # leave a placeholder so list_directory has something to show.
        root_task = self.workspace_root / "_rlm_query_0.txt"
        if not root_task.exists():
            root_task.write_text("", encoding="utf-8")

    def _seed_provenance_and_manifest(self) -> None:
        self.provenance.load()
        # Root task and any pre-existing user-context files at workspace root
        # → user. State files → system.
        self.provenance.record_seed("_rlm_query_0.txt", role="user", action_id=None, turn=0)
        for state_file in ("provenance.json", "action_log.jsonl", "workspace_manifest.json"):
            self.provenance.record_seed(
                f"{RESERVED_STATE_DIR}/{state_file}", role="system", action_id=None, turn=0
            )
        self.provenance.save()
        manifest = {
            "run_id": self.run_id,
            "depth": self.depth,
            "max_depth": self.max_depth,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "image": self.workspace_config.docker.image,
        }
        self._manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        # Touch action log so reads always succeed.
        if not self._action_log_path.exists():
            self._action_log_path.write_text("", encoding="utf-8")

    def _git_init(self) -> None:
        if (self.workspace_root / ".git").exists():
            return
        self._git("init", "-q")
        # Local-only identity so commits don't fail on hosts without ~/.gitconfig.
        self._git("config", "user.email", "rlm@workspace.local")
        self._git("config", "user.name", "RLM Workspace")
        self._git("add", "-A")
        self._git("commit", "-q", "-m", "turn 0", "--allow-empty")

    def _git(self, *args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(self.workspace_root), *args],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout

    # =========================================================================
    # Container lifecycle + host poller
    # =========================================================================

    def _start_container(self) -> None:
        broker_port = self.workspace_config.docker.broker_port
        # Bind the broker's container port to a random free host port. We
        # publish to 127.0.0.1 only so it's not exposed on external interfaces.
        cmd = [
            "docker",
            "run",
            "-d",
            "--rm",
            "-v",
            f"{self.workspace_root}:/workspace",
            "-w",
            "/workspace",
            "-p",
            f"127.0.0.1::{broker_port}",
            "--add-host=host.docker.internal:host-gateway",
            "-e",
            f"RLM_BROKER_PORT={broker_port}",
            self.workspace_config.docker.image,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to start workspace container: {result.stderr.strip()}")
        self._container_id = result.stdout.strip()

        # Look up the host port assigned to the container's broker_port.
        port_result = subprocess.run(
            ["docker", "port", self._container_id, str(broker_port)],
            capture_output=True,
            text=True,
            check=True,
        )
        # Output looks like "127.0.0.1:54321\n0.0.0.0:54321\n"; take the first.
        first_line = port_result.stdout.strip().splitlines()[0]
        self._broker_host_port = int(first_line.rsplit(":", 1)[1])
        self._wait_for_broker_ready()

    def _wait_for_broker_ready(self, timeout: float = 15.0) -> None:
        deadline = time.monotonic() + timeout
        url = f"http://127.0.0.1:{self._broker_host_port}/health"
        last_err: Exception | None = None
        while time.monotonic() < deadline:
            try:
                r = requests.get(url, timeout=1.0)
                if r.status_code == 200:
                    return
            except (OSError, requests.ConnectionError, requests.Timeout) as e:
                last_err = e
            time.sleep(0.1)
        raise RuntimeError(f"Workspace broker did not become ready within {timeout}s: {last_err!r}")

    def _start_poller(self) -> None:
        self._poller_stop.clear()
        thread = threading.Thread(
            target=self._poller_loop, name=f"rlm-poller-{self.run_id}", daemon=True
        )
        thread.start()
        self._poller_thread = thread

    def _poller_loop(self) -> None:
        interval = self.workspace_config.docker.poll_interval_ms / 1000.0
        pending_url = f"http://127.0.0.1:{self._broker_host_port}/pending"
        while not self._poller_stop.is_set():
            try:
                r = requests.get(pending_url, timeout=2.0)
                if r.status_code == 200:
                    requests_list = r.json().get("requests", [])
                    for req in requests_list:
                        # Each request handled in its own thread so a slow LM
                        # call doesn't starve siblings.
                        threading.Thread(
                            target=self._handle_broker_request,
                            args=(req,),
                            daemon=True,
                        ).start()
            except (requests.ConnectionError, requests.Timeout):
                pass  # container restarting / shutting down — keep polling
            except Exception:
                log.exception("Workspace poller error")
            self._poller_stop.wait(interval)

    def _handle_broker_request(self, req: dict[str, Any]) -> None:
        req_id = req.get("id")
        kind = req.get("kind")
        try:
            if kind == "llm_query":
                response = self._do_llm_query(req["prompt"])
            elif kind == "llm_query_batched":
                response = self._do_llm_query_batched(req["prompts"])
            elif kind == "rlm_query":
                response = self._do_rlm_query(req["prompt"])
            elif kind == "rlm_query_batched":
                response = self._do_rlm_query_batched(req["prompts"])
            else:
                response = {"error": f"Unknown broker kind: {kind!r}"}
        except Exception as e:
            log.exception("Broker request handler raised")
            response = {"error": f"Host-side handler raised: {e}"}
        self._respond_to_broker(req_id, response)

    def _do_llm_query(self, prompt: str) -> dict[str, Any]:
        if self.lm_handler_address is None:
            return {"error": "No LM handler configured"}
        request = LMRequest(prompt=prompt, depth=self.depth)
        resp = send_lm_request(self.lm_handler_address, request)
        if not resp.success or resp.chat_completion is None:
            return {"error": resp.error or "LM call failed"}
        return {"response": resp.chat_completion.response}

    def _do_llm_query_batched(self, prompts: list[str]) -> dict[str, Any]:
        if self.lm_handler_address is None:
            return {"error": "No LM handler configured"}
        # Widen for the invariant ``list[str | dict[str, Any]]`` parameter.
        widened: list[str | dict[str, Any]] = list(prompts)
        responses = send_lm_request_batched(self.lm_handler_address, widened, depth=self.depth)
        out: list[str] = []
        for resp in responses:
            if not resp.success or resp.chat_completion is None:
                out.append(f"Error: {resp.error or 'LM call failed'}")
            else:
                out.append(resp.chat_completion.response)
        return {"responses": out}

    def _do_rlm_query(self, prompt: str) -> dict[str, Any]:
        if self.recursion_handler is None:
            return {
                "error": ("Maximum recursion depth reached. The 'rlm_query' tool is unavailable.")
            }
        return self.recursion_handler.spawn_via_broker(
            child_task=prompt, action_id=self.current_action_id
        )

    def _do_rlm_query_batched(self, prompts: list[str]) -> dict[str, Any]:
        if self.recursion_handler is None:
            err = "Maximum recursion depth reached. The 'rlm_query' tool is unavailable."
            return {"responses": [f"Error: {err}"] * len(prompts)}
        return self.recursion_handler.spawn_via_broker_batched(
            child_tasks=prompts, action_id=self.current_action_id
        )

    def _respond_to_broker(self, req_id: str | None, response: dict[str, Any]) -> None:
        if not req_id:
            return
        url = f"http://127.0.0.1:{self._broker_host_port}/respond"
        payload = {"id": req_id, **response}
        try:
            requests.post(url, json=payload, timeout=5.0)
        except Exception:
            log.exception("Failed to post broker response for %s", req_id)

    # =========================================================================
    # Container exec (used by shell / python tools)
    # =========================================================================

    def exec_in_container(self, cmd: list[str], timeout: int) -> ExecResult:
        if self._container_id is None:
            raise RuntimeError("Container is not running; call setup() first")
        full_cmd = ["docker", "exec", self._container_id, *cmd]
        start = time.perf_counter()
        try:
            proc = subprocess.run(
                full_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return ExecResult(
                stdout=proc.stdout,
                stderr=proc.stderr,
                exit_code=proc.returncode,
                timed_out=False,
                duration=time.perf_counter() - start,
            )
        except subprocess.TimeoutExpired as e:
            stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            stderr = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
            return ExecResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=-1,
                timed_out=True,
                duration=time.perf_counter() - start,
            )

    # =========================================================================
    # Action dispatch
    # =========================================================================

    def load_context(self, context_payload: Any) -> None:
        """Drop ``context_payload`` into the workspace.

        - ``str`` or single value → ``_rlm_query_0.txt`` (the root task slot).
          If a payload was already loaded, the next slot is used.
        - ``list`` → one chunk per ``_rlm_query_<N>.txt`` slot, starting at 0.
        - ``dict`` → JSON-encoded into ``_rlm_query_0.txt``.
        """
        if isinstance(context_payload, list):
            chunks = [self._coerce_chunk(c) for c in context_payload]
        else:
            chunks = [self._coerce_chunk(context_payload)]

        # setup() seeded _rlm_query_0.txt as an empty placeholder. Treat the
        # empty slot 0 as the start; otherwise append to the next free slot.
        start = 0 if self._is_empty_slot(0) else self._existing_query_slots()

        for offset, chunk in enumerate(chunks):
            slot = start + offset
            path = self.workspace_root / f"_rlm_query_{slot}.txt"
            path.write_text(chunk, encoding="utf-8")
            self.provenance.record_seed(
                f"_rlm_query_{slot}.txt", role="user", action_id=None, turn=0
            )
        self.provenance.save()

    @staticmethod
    def _coerce_chunk(chunk: Any) -> str:
        if isinstance(chunk, str):
            return chunk
        return json.dumps(chunk, indent=2)

    def _existing_query_slots(self) -> int:
        """Highest filled ``_rlm_query_<N>.txt`` slot + 1; 0 if none."""
        i = 0
        while (self.workspace_root / f"_rlm_query_{i}.txt").exists():
            i += 1
        return i

    def _is_empty_slot(self, slot: int) -> bool:
        path = self.workspace_root / f"_rlm_query_{slot}.txt"
        try:
            return path.stat().st_size == 0
        except OSError:
            return False

    def run_action(self, action: WorkspaceAction) -> WorkspaceObservation:
        """Dispatch ``action`` through the tool registry.

        Per-call observation truncation (spill-to-artifact above
        ``observation.max_observation_chars``) is applied here, not in the
        tool modules — it is a uniform post-processing step.
        """
        spec = get_spec(action.tool)
        executor = get_executor(action.tool)

        # Allocate a per-turn action id (t<turn>.a<idx>).
        idx = self._action_seq_per_turn.get(self.current_turn, 0) + 1
        self._action_seq_per_turn[self.current_turn] = idx
        self.current_action_id = f"t{self.current_turn}.a{idx}"

        obs = executor(self, action)
        obs = self._maybe_spill_observation(obs)

        # Provenance changes are persisted after every action so a crash
        # doesn't lose role attribution mid-turn.
        self.provenance.save()
        self._append_action_log(action, obs, spec.is_state_mutating)
        return obs

    def _append_action_log(
        self,
        action: WorkspaceAction,
        observation: WorkspaceObservation,
        mutating: bool,
    ) -> None:
        record = {
            "action_id": self.current_action_id,
            "turn": self.current_turn,
            "tool": action.tool,
            "args": action.args,
            "mutating": mutating,
            "observation": {
                "tool": observation.tool,
                "stdout_len": len(observation.stdout),
                "stderr_len": len(observation.stderr),
                "artifacts": list(observation.artifacts),
                "error": observation.error,
                "execution_time": observation.execution_time,
                "final_answer": observation.final_answer,
                "final_artifacts": list(observation.final_artifacts),
            },
        }
        with self._action_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _maybe_spill_observation(self, obs: WorkspaceObservation) -> WorkspaceObservation:
        cap = self.workspace_config.observation.max_observation_chars
        body_len = len(obs.stdout) + len(obs.stderr)
        if body_len <= cap:
            return obs
        spill_id = self.current_action_id or uuid.uuid4().hex[:12]
        spill_rel = f"_rlm_artifacts/_observations/{spill_id}.txt"
        spill_path = self.workspace_root / spill_rel
        spill_path.parent.mkdir(parents=True, exist_ok=True)
        spill_path.write_text(
            (
                f"--- stdout ({len(obs.stdout)} chars) ---\n{obs.stdout}\n"
                f"--- stderr ({len(obs.stderr)} chars) ---\n{obs.stderr}\n"
            ),
            encoding="utf-8",
        )
        # Mark spill file as system-provenance so list_directory shows the right role.
        self.provenance.record_write(
            spill_rel, role="system", action_id=self.current_action_id, turn=self.current_turn
        )
        summary = (
            f"[Observation truncated: {body_len} chars exceeded "
            f"observation.max_observation_chars ({cap}). "
            f"Full output written to {spill_rel}.]"
        )
        obs.stdout = summary
        obs.stderr = ""
        if spill_rel not in obs.artifacts:
            obs.artifacts.append(spill_rel)
        return obs

    # =========================================================================
    # Snapshots
    # =========================================================================

    def snapshot(self, turn: int) -> WorkspaceSnapshot:
        # Allow-empty so turns with only read-only actions still mark a commit.
        self._git("add", "-A")
        self._git("commit", "-q", "-m", f"turn {turn}", "--allow-empty")
        sha = self._git("rev-parse", "HEAD").strip()
        # Collect changed files vs. previous commit (for turn 0 baseline,
        # we just emit an empty list; the repo was newly initialised).
        try:
            changed_raw = self._git("diff", "--name-only", "HEAD~1", "HEAD")
            changed = [line for line in changed_raw.splitlines() if line]
        except subprocess.CalledProcessError:
            changed = []
        return WorkspaceSnapshot(
            turn=turn,
            commit_sha=sha,
            changed_files=changed,
            workspace_root=str(self.workspace_root),
        )

    # =========================================================================
    # Cleanup
    # =========================================================================

    def cleanup(self) -> None:
        self._poller_stop.set()
        if self._poller_thread is not None:
            self._poller_thread.join(timeout=2.0)
            self._poller_thread = None
        if self._container_id is not None:
            subprocess.run(
                ["docker", "stop", "-t", "1", self._container_id],
                capture_output=True,
            )
            self._container_id = None
        self._handle_workspace_cleanup()
        self._is_setup = False

    def _handle_workspace_cleanup(self) -> None:
        mode = self.workspace_config.docker.cleanup_mode
        if mode == "keep":
            return
        if not self.workspace_root.exists():
            return
        if mode == "delete":
            shutil.rmtree(self.workspace_root, ignore_errors=True)
            return
        if mode == "tar":
            archive_base = str(self.workspace_root) + ".tar.gz"
            shutil.make_archive(
                base_name=str(self.workspace_root),
                format="gztar",
                root_dir=str(self.workspace_root.parent),
                base_dir=self.workspace_root.name,
            )
            shutil.rmtree(self.workspace_root, ignore_errors=True)
            log.info("Workspace archived to %s", archive_base)
            return
        # Should be unreachable thanks to the Literal type on cleanup_mode.
        raise ValueError(f"Unknown cleanup_mode: {mode!r}")

    # =========================================================================
    # Context-manager / del safety
    # =========================================================================

    def __del__(self) -> None:
        # Best-effort. We intentionally don't raise from __del__; container
        # leaks are caught by a periodic ``docker container prune`` in
        # development. setup() side-effects only happen if the user explicitly
        # called setup() or used the context manager.
        try:
            if self._is_setup:
                self.cleanup()
        except Exception:
            pass


def workspace_root_default(run_id: str) -> Path:
    """Helper: default workspace root for a given run id (used by recursion)."""
    base = Path(WorkspaceConfig().docker.workspace_root_base).expanduser()
    return base / run_id
