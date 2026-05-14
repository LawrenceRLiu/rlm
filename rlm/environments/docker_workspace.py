"""
Docker-backed workspace environment.

Sibling layout
--------------
Each workspace_root has four bind-mounted subdirs:

  ``<workspace_root>/app/``             → ``/app``
  ``<workspace_root>/_rlm_state/``      → ``/_rlm_state``
  ``<workspace_root>/_rlm_artifacts/``  → ``/_rlm_artifacts``
  ``<workspace_root>/_rlm_notes/``      → ``/_rlm_notes``

``/app`` is the canonical task workdir — shell/python tools default to it,
and graders (e.g. Terminal-Bench Harbor) evaluate it. Before starting the
container we extract the image's baked ``/app`` contents to the host
subdir via ``docker create`` + ``docker cp`` (the "image-seed dance"),
because the bind mount otherwise shadows whatever the image put there.

End-to-end responsibilities:

- Provision a workspace at ``<docker.workspace_root_base>/<run_id>/`` with
  the four sibling subdirs and seed ``provenance.json``.
- ``git init`` at workspace root and commit a "turn 0" baseline.
- Seed ``/app`` from the image, then start the container with
  ``-v`` per subdir, ``-w /app`` (overridable), ``-e PYTHONPATH=/app``,
  and a port-published broker.
- Host poller drains ``/pending`` and forwards LM requests to the on-host
  ``LMHandler`` via ``send_lm_request``.
- Dispatch ``WorkspaceAction``s through the tool registry. Observation
  bodies above ``observation.max_observation_chars`` spill to
  ``_rlm_artifacts/_observations/<id>.txt`` and the body is replaced with
  a summary line.

The env is reusable as a child workspace for ``rlm_query``: when
``workspace_root`` is supplied, the run_id-derived path is skipped and the
caller is expected to have populated the directory (copy-on-spawn).
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
from rlm.core.types import (
    RLMChatCompletion,
    WorkspaceAction,
    WorkspaceObservation,
    WorkspaceSnapshot,
)
from rlm.environments.base_workspace import BaseWorkspaceEnv
from rlm.utils.provenance import ProvenanceStore, diff_snapshots, snapshot_paths
from rlm.workspace_tools import get_executor, get_spec

log = logging.getLogger(__name__)

# Reserved name prefix the model is forbidden to write to (state lives here).
RESERVED_STATE_DIR = "_rlm_state"

# Sibling-layout subdirs of workspace_root. Each is bind-mounted into the
# container at ``/<name>``. Order matters for resolve_workspace_path's
# prefix matching: longer prefixes must come before shorter ones if any
# share a prefix (none currently do, but the loop respects this ordering).
WORKSPACE_LAYOUT: tuple[str, ...] = ("app", "_rlm_state", "_rlm_artifacts", "_rlm_notes")

# Subdir under _rlm_state where shell/python tempfiles are staged. Excluded
# from provenance diffs because _rlm_state is excluded wholesale.
_TMP_REL_DIR = "_rlm_state/_tmp"

# Observation spill destination (inside _rlm_artifacts, visible to the model).
_OBS_REL_DIR = "_rlm_artifacts/_observations"


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
        self._broker_exec_python: str | None = None
        self._poller_thread: threading.Thread | None = None
        self._poller_stop = threading.Event()
        self._action_seq_per_turn: dict[int, int] = {}

        # Per-action ledger: broker-worker threads append produced
        # RLMChatCompletions here keyed by ``current_action_id``; the python
        # tool drains the entry after ``exec_in_container`` returns and
        # surfaces the list in ``observation.rlm_calls``. See plan in
        # ``read-the-visibility-followup-sunny-hopper.md``.
        self._broker_ledger: dict[str, list[RLMChatCompletion]] = {}
        self._broker_ledger_lock = threading.Lock()

        self._action_log_path = self.workspace_root / RESERVED_STATE_DIR / "action_log.jsonl"
        self._manifest_path = self.workspace_root / RESERVED_STATE_DIR / "workspace_manifest.json"

        # Has setup() run? Avoids re-init on context-manager nesting.
        self._is_setup = False

    # =========================================================================
    # Path helpers / contract used by tool modules
    # =========================================================================

    def is_reserved_path(self, rel: str) -> bool:
        """``True`` for paths the model is forbidden to write (``_rlm_state/...``).

        Accepts both workspace-relative (``_rlm_state/foo``) and
        container-absolute (``/_rlm_state/foo``) forms.
        """
        norm = str(rel).replace("\\", "/").lstrip("./")
        # Strip a single leading slash so container-absolute paths normalize
        # to the same form as workspace-relative.
        if norm.startswith("/"):
            norm = norm.lstrip("/")
        return norm == RESERVED_STATE_DIR or norm.startswith(RESERVED_STATE_DIR + "/")

    def resolve_workspace_path(self, rel: str) -> Path:
        """Resolve ``rel`` to its host-side path.

        Accepts:
          - **Workspace root aliases** ``""``, ``"."``, ``"/"`` → host
            ``workspace_root``.
          - **Container-absolute** paths under one of the four bind-mount
            roots (``/app``, ``/_rlm_state``, ``/_rlm_artifacts``,
            ``/_rlm_notes``). Translated to the host bind source.
          - **Workspace-relative** paths (e.g. ``app/output.txt``,
            ``_rlm_notes/n.md``). Joined to ``workspace_root``.

        Rejects:
          - Other absolute paths (escape attempt).
          - Workspace-relative paths that escape ``workspace_root`` via ``..``.
        """
        norm = str(rel).replace("\\", "/")
        ws_resolved = self.workspace_root.resolve()

        # Workspace root aliases.
        if norm in ("", ".", "/"):
            return ws_resolved

        # Container-absolute under a bind-mount root → host bind source.
        if norm.startswith("/"):
            for subdir in WORKSPACE_LAYOUT:
                root = f"/{subdir}"
                if norm == root or norm.startswith(root + "/"):
                    inner = norm[len(root) :].lstrip("/")
                    full = (self.workspace_root / subdir / inner).resolve()
                    # Defense in depth — even after the prefix match, confirm
                    # we land somewhere under workspace_root before returning.
                    try:
                        full.relative_to(ws_resolved)
                    except ValueError as e:
                        raise ValueError(f"Path escapes workspace: {rel!r}") from e
                    return full
            raise ValueError(
                f"Absolute path not under a bind-mounted root: {rel!r}. "
                f"Allowed roots: /app, /_rlm_state, /_rlm_artifacts, /_rlm_notes."
            )

        # Workspace-relative — join to workspace_root, enforce no-escape.
        full = (self.workspace_root / norm).resolve()
        try:
            full.relative_to(ws_resolved)
        except ValueError as e:
            raise ValueError(f"Path escapes workspace: {rel!r}") from e
        return full

    def host_to_container_path(self, host_path: Path) -> str:
        """Translate a host-side path under ``workspace_root`` to its
        container-visible counterpart (``/app/...`` etc.).

        Returns the bind-mount-relative path if ``host_path`` lives under one
        of the four layout subdirs. Raises ``ValueError`` otherwise — the
        caller is asking for a container path for something that is not
        bind-mounted.
        """
        host_resolved = host_path.resolve()
        ws_resolved = self.workspace_root.resolve()
        try:
            rel = host_resolved.relative_to(ws_resolved)
        except ValueError as e:
            raise ValueError(f"Host path is not under workspace_root: {host_path}") from e
        rel_str = str(rel).replace("\\", "/")
        for subdir in WORKSPACE_LAYOUT:
            if rel_str == subdir or rel_str.startswith(subdir + "/"):
                inner = rel_str[len(subdir) :].lstrip("/")
                return f"/{subdir}/{inner}" if inner else f"/{subdir}"
        raise ValueError(f"Path is under workspace_root but outside any bind mount: {rel_str!r}")

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
        try:
            self._create_workspace_dirs()
            self._seed_app_from_image()
            self._seed_provenance_and_manifest()
            self._git_init()
            self._start_container()
            self._start_poller()
            self._is_setup = True
        except Exception:
            self.cleanup()
            raise

    def _create_workspace_dirs(self) -> None:
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        for subdir in WORKSPACE_LAYOUT:
            (self.workspace_root / subdir).mkdir(exist_ok=True)
        (self.workspace_root / _OBS_REL_DIR).mkdir(parents=True, exist_ok=True)
        (self.workspace_root / _TMP_REL_DIR).mkdir(parents=True, exist_ok=True)
        # _rlm_query_0.txt is created at load_context() time; until then we
        # leave a placeholder in _rlm_state so list_directory has something
        # to show and the file is out of the grader's /app target.
        root_task = self.workspace_root / RESERVED_STATE_DIR / "_rlm_query_0.txt"
        if not root_task.exists():
            root_task.write_text("", encoding="utf-8")

    def _seed_app_from_image(self) -> None:
        """Extract the image's baked ``/app`` contents into the host bind
        source so the bind mount doesn't shadow them at runtime.

        If the host ``app/`` subdir is already populated (e.g. for a
        copy-on-spawned child workspace), skip — the parent's seed is
        already there. If the image has no ``/app``, the cp will fail
        cleanly and we leave the empty dir in place.
        """
        app_host = self.workspace_root / "app"
        # Already populated → child workspace path; do not re-seed.
        if any(app_host.iterdir()):
            return
        image = self.workspace_config.docker.image
        create = subprocess.run(
            ["docker", "create", image],
            capture_output=True,
            text=True,
        )
        if create.returncode != 0:
            raise RuntimeError(f"docker create failed for image {image!r}: {create.stderr.strip()}")
        tmp_id = create.stdout.strip()
        try:
            cp = subprocess.run(
                ["docker", "cp", f"{tmp_id}:/app/.", str(app_host)],
                capture_output=True,
                text=True,
            )
            if cp.returncode != 0:
                # No /app in image → leave empty subdir. The model can still
                # use /app as scratch. Log and move on; not fatal.
                log.info(
                    "Image %r has no /app to seed (docker cp: %s); leaving host app/ empty.",
                    image,
                    cp.stderr.strip(),
                )
        finally:
            subprocess.run(["docker", "rm", "-f", tmp_id], capture_output=True, text=True)

    def _seed_provenance_and_manifest(self) -> None:
        self.provenance.load()
        # Stamp the four sibling roots so list_directory of workspace root
        # surfaces accurate ownership: app/ is task surface (user), the three
        # _rlm_* dirs are substrate-managed (system).
        self.provenance.record_seed("app", role="user", action_id=None, turn=0)
        for substrate_dir in ("_rlm_notes", "_rlm_artifacts", RESERVED_STATE_DIR):
            self.provenance.record_seed(substrate_dir, role="system", action_id=None, turn=0)
        # Root task and any pre-existing user-context files → user.
        # State files → system. Files seeded from the image into app/ are
        # treated as user input (they are the task's starter material).
        self.provenance.record_seed(
            f"{RESERVED_STATE_DIR}/_rlm_query_0.txt", role="user", action_id=None, turn=0
        )
        for state_file in ("provenance.json", "action_log.jsonl", "workspace_manifest.json"):
            self.provenance.record_seed(
                f"{RESERVED_STATE_DIR}/{state_file}", role="system", action_id=None, turn=0
            )
        # Stamp every seeded /app file as user-provenance so list_directory
        # shows it accurately. ``rglob`` is acceptable here because this
        # runs once at setup() — not per-action.
        app_host = self.workspace_root / "app"
        if app_host.exists():
            for path in app_host.rglob("*"):
                if path.is_file():
                    rel = str(path.relative_to(self.workspace_root)).replace("\\", "/")
                    self.provenance.record_seed(rel, role="user", action_id=None, turn=0)
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
        dcfg = self.workspace_config.docker
        broker_port = dcfg.broker_port
        # In normal mode, bind the broker's container port to a random free
        # host port. We publish to 127.0.0.1 only so it's not exposed on
        # external interfaces. In no-internet mode, use Docker's network
        # namespace isolation and talk to the broker through docker exec.
        cmd: list[str] = [
            "docker",
            "run",
            "-d",
            "--rm",
        ]
        if not dcfg.allow_internet:
            cmd.extend(["--network", "none"])
        # Multi-bind-mount: each layout subdir is mounted at /<name> in the
        # container. The image-seed dance has already populated host /app
        # with the image's baked contents, so the bind mount is non-shadowing.
        for subdir in WORKSPACE_LAYOUT:
            cmd.extend(["-v", f"{self.workspace_root / subdir}:/{subdir}"])
        # Note: we DO NOT pass -e PYTHONPATH at run time. The broker image bakes
        # PYTHONPATH=/opt/rlm_workspace so `python -m rlm_workspace.broker`
        # works; overriding it here would break broker startup. The model-
        # facing PYTHONPATH (container_pythonpath, default /app) is applied
        # per-action via `docker exec -e`, not on the container as a whole.
        cmd.extend(["-w", dcfg.container_cwd])
        if dcfg.allow_internet:
            cmd.extend(
                [
                    "-p",
                    f"127.0.0.1::{broker_port}",
                    "--add-host=host.docker.internal:host-gateway",
                ]
            )
        cmd.extend(["-e", f"RLM_BROKER_PORT={broker_port}", dcfg.image])
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to start workspace container: {result.stderr.strip()}")
        self._container_id = result.stdout.strip()

        if dcfg.allow_internet:
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
        else:
            self._broker_host_port = None
            self._broker_exec_python = self._select_broker_exec_python()
        self._wait_for_broker_ready()

    def _wait_for_broker_ready(self, timeout: float = 15.0) -> None:
        deadline = time.monotonic() + timeout
        last_err: Exception | None = None
        while time.monotonic() < deadline:
            try:
                if self._broker_host_port is not None:
                    url = f"http://127.0.0.1:{self._broker_host_port}/health"
                    r = requests.get(url, timeout=1.0)
                    if r.status_code != 200:
                        time.sleep(0.1)
                        continue
                else:
                    self._broker_exec_request("GET", "/health", timeout=1.0)
                return
            except (OSError, requests.ConnectionError, requests.Timeout, RuntimeError) as e:
                last_err = e
            time.sleep(0.1)
        raise RuntimeError(f"Workspace broker did not become ready within {timeout}s: {last_err!r}")

    def _select_broker_exec_python(self) -> str:
        """Choose a Python interpreter for docker-exec broker HTTP calls."""
        if self._container_id is None:
            raise RuntimeError("Container is not running; cannot select broker Python")
        candidates = ["/opt/broker/bin/python", "python3", "python"]
        for candidate in candidates:
            probe = subprocess.run(
                ["docker", "exec", self._container_id, candidate, "-c", "pass"],
                capture_output=True,
                text=True,
            )
            if probe.returncode == 0:
                return candidate
        raise RuntimeError(
            "Could not find a Python interpreter inside the workspace container "
            "for no-internet broker polling."
        )

    def _broker_exec_request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        timeout: float = 5.0,
    ) -> dict[str, Any]:
        if self._container_id is None:
            raise RuntimeError("Container is not running; cannot reach broker")
        if self._broker_exec_python is None:
            self._broker_exec_python = self._select_broker_exec_python()
        broker_port = self.workspace_config.docker.broker_port
        request_payload = {
            "method": method,
            "url": f"http://localhost:{broker_port}{path}",
            "json": payload,
            "timeout": timeout,
        }
        code = r"""
import json
import sys
import urllib.request

cfg = json.loads(sys.stdin.read())
body = cfg.get("json")
data = None if body is None else json.dumps(body).encode("utf-8")
headers = {"Content-Type": "application/json"} if data is not None else {}
req = urllib.request.Request(cfg["url"], data=data, headers=headers, method=cfg["method"])
with urllib.request.urlopen(req, timeout=cfg["timeout"]) as resp:
    sys.stdout.write(resp.read().decode("utf-8"))
"""
        proc = subprocess.run(
            ["docker", "exec", "-i", self._container_id, self._broker_exec_python, "-c", code],
            input=json.dumps(request_payload),
            capture_output=True,
            text=True,
            timeout=max(1.0, timeout + 1.0),
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "docker-exec broker request failed")
        try:
            return json.loads(proc.stdout or "{}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Broker returned non-JSON response: {proc.stdout[:500]!r}") from e

    def _start_poller(self) -> None:
        self._poller_stop.clear()
        thread = threading.Thread(
            target=self._poller_loop, name=f"rlm-poller-{self.run_id}", daemon=True
        )
        thread.start()
        self._poller_thread = thread

    def _poller_loop(self) -> None:
        interval = self.workspace_config.docker.poll_interval_ms / 1000.0
        while not self._poller_stop.is_set():
            try:
                if self._broker_host_port is not None:
                    pending_url = f"http://127.0.0.1:{self._broker_host_port}/pending"
                    r = requests.get(pending_url, timeout=2.0)
                    if r.status_code != 200:
                        self._poller_stop.wait(interval)
                        continue
                    pending = r.json()
                else:
                    pending = self._broker_exec_request("GET", "/pending", timeout=2.0)
                requests_list = pending.get("requests", [])
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
        # Snapshot at handler entry: the python action's ``current_action_id``
        # is fixed while ``exec_in_container`` blocks, but snapshotting here is
        # robust to future refactors and is what guarantees per-action ledger
        # keying for the broker workers spawned by the poller.
        action_id = self.current_action_id
        try:
            if kind == "llm_query":
                response = self._do_llm_query(req["prompt"], action_id=action_id)
            elif kind == "llm_query_batched":
                response = self._do_llm_query_batched(req["prompts"], action_id=action_id)
            elif kind == "rlm_query":
                response = self._do_rlm_query(req["prompt"], action_id=action_id)
            elif kind == "rlm_query_batched":
                response = self._do_rlm_query_batched(req["prompts"], action_id=action_id)
            else:
                response = {"error": f"Unknown broker kind: {kind!r}"}
        except Exception as e:
            log.exception("Broker request handler raised")
            response = {"error": f"Host-side handler raised: {e}"}
        self._respond_to_broker(req_id, response)

    def _do_llm_query(self, prompt: str, *, action_id: str | None) -> dict[str, Any]:
        if self.lm_handler_address is None:
            return {"error": "No LM handler configured"}
        request = LMRequest(prompt=prompt, depth=self.depth)
        resp = send_lm_request(self.lm_handler_address, request)
        if not resp.success or resp.chat_completion is None:
            return {"error": resp.error or "LM call failed"}
        # Append BEFORE responding to broker — ordering invariant: the
        # in-container client unblocks only after ``_respond_to_broker``,
        # so the python tool's drain always sees this entry.
        self._append_broker_ledger(action_id, [resp.chat_completion])
        return {"response": resp.chat_completion.response}

    def _do_llm_query_batched(self, prompts: list[str], *, action_id: str | None) -> dict[str, Any]:
        if self.lm_handler_address is None:
            return {"error": "No LM handler configured"}
        # Widen for the invariant ``list[str | dict[str, Any]]`` parameter.
        widened: list[str | dict[str, Any]] = list(prompts)
        responses = send_lm_request_batched(self.lm_handler_address, widened, depth=self.depth)
        out: list[str] = []
        completions: list[RLMChatCompletion] = []
        for resp in responses:
            if not resp.success or resp.chat_completion is None:
                out.append(f"Error: {resp.error or 'LM call failed'}")
            else:
                out.append(resp.chat_completion.response)
                completions.append(resp.chat_completion)
        self._append_broker_ledger(action_id, completions)
        return {"responses": out}

    def _do_rlm_query(self, prompt: str, *, action_id: str | None) -> dict[str, Any]:
        if self.recursion_handler is None:
            return {
                "error": ("Maximum recursion depth reached. The 'rlm_query' tool is unavailable.")
            }
        return self.recursion_handler.spawn_via_broker(child_task=prompt, action_id=action_id)

    def _do_rlm_query_batched(self, prompts: list[str], *, action_id: str | None) -> dict[str, Any]:
        if self.recursion_handler is None:
            err = "Maximum recursion depth reached. The 'rlm_query' tool is unavailable."
            return {"responses": [f"Error: {err}"] * len(prompts)}
        return self.recursion_handler.spawn_via_broker_batched(
            child_tasks=prompts, action_id=action_id
        )

    # =========================================================================
    # Broker ledger (per-action ``RLMChatCompletion`` capture)
    # =========================================================================

    def _append_broker_ledger(
        self, action_id: str | None, completions: list[RLMChatCompletion]
    ) -> None:
        """Thread-safe append of broker-produced ``RLMChatCompletion``s.

        No-op if ``action_id`` is None or ``completions`` is empty — avoids
        creating empty buckets for stale or out-of-band requests.
        """
        if not action_id or not completions:
            return
        with self._broker_ledger_lock:
            self._broker_ledger.setdefault(action_id, []).extend(completions)

    def drain_broker_ledger(self, action_id: str | None) -> list[RLMChatCompletion]:
        """Pop and return the ledger entry for ``action_id`` (or ``[]``)."""
        if not action_id:
            return []
        with self._broker_ledger_lock:
            return self._broker_ledger.pop(action_id, [])

    def _respond_to_broker(self, req_id: str | None, response: dict[str, Any]) -> None:
        if not req_id:
            return
        payload = {"id": req_id, **response}
        try:
            if self._broker_host_port is not None:
                url = f"http://127.0.0.1:{self._broker_host_port}/respond"
                requests.post(url, json=payload, timeout=5.0)
            else:
                self._broker_exec_request("POST", "/respond", payload=payload, timeout=5.0)
        except Exception:
            log.exception("Failed to post broker response for %s", req_id)

    # =========================================================================
    # Container exec (used by shell / python tools)
    # =========================================================================

    def exec_in_container(
        self,
        cmd: list[str],
        timeout: int,
        cwd: str | None = None,
    ) -> ExecResult:
        """Run ``cmd`` inside the running container.

        ``cwd`` overrides the image WORKDIR for this call — passed to
        ``docker exec -w``. When None, the image's WORKDIR applies (set
        at ``docker run`` time to ``DockerConfig.container_cwd``, default
        ``/app``).

        ``PYTHONPATH`` is set on every exec to ``DockerConfig.container_pythonpath``
        — env vars on the run-time container do not propagate to ``docker
        exec`` by default, so we forward it explicitly.
        """
        if self._container_id is None:
            raise RuntimeError("Container is not running; call setup() first")
        dcfg = self.workspace_config.docker
        # Compose PYTHONPATH = <model-facing dir>:<broker package dir>. The
        # broker image bakes ``rlm_workspace`` at ``/opt/rlm_workspace``;
        # ``docker exec -e`` does not inherit the image's ENV, so we must
        # forward it explicitly or every ``python`` action loses access to
        # the pre-imported ``llm_query`` / ``rlm_query`` helpers.
        pythonpath = f"{dcfg.container_pythonpath}:/opt/rlm_workspace"
        exec_cmd: list[str] = ["docker", "exec", "-e", f"PYTHONPATH={pythonpath}"]
        if cwd is not None:
            exec_cmd.extend(["-w", cwd])
        exec_cmd.append(self._container_id)
        exec_cmd.extend(cmd)
        full_cmd = exec_cmd
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

        # setup() seeded _rlm_state/_rlm_query_0.txt as an empty placeholder.
        # Treat the empty slot 0 as the start; otherwise append to next free.
        start = 0 if self._is_empty_slot(0) else self._existing_query_slots()

        for offset, chunk in enumerate(chunks):
            slot = start + offset
            rel = f"{RESERVED_STATE_DIR}/_rlm_query_{slot}.txt"
            path = self.workspace_root / rel
            path.write_text(chunk, encoding="utf-8")
            self.provenance.record_seed(rel, role="user", action_id=None, turn=0)
        self.provenance.save()

    @staticmethod
    def _coerce_chunk(chunk: Any) -> str:
        if isinstance(chunk, str):
            return chunk
        return json.dumps(chunk, indent=2)

    def _existing_query_slots(self) -> int:
        """Highest filled ``_rlm_state/_rlm_query_<N>.txt`` slot + 1; 0 if none."""
        state_dir = self.workspace_root / RESERVED_STATE_DIR
        i = 0
        while (state_dir / f"_rlm_query_{i}.txt").exists():
            i += 1
        return i

    def _is_empty_slot(self, slot: int) -> bool:
        path = self.workspace_root / RESERVED_STATE_DIR / f"_rlm_query_{slot}.txt"
        try:
            return path.stat().st_size == 0
        except OSError:
            return False

    def run_action(self, action: WorkspaceAction) -> WorkspaceObservation:
        """Dispatch ``action`` through the tool registry.

        Per-call observation truncation (spill-to-artifact above
        ``observation.max_observation_chars``) is applied here, not in the
        tool modules — it is a uniform post-processing step.

        Backstop for ``ValueError`` from path-validating tools: an absolute
        or workspace-escaping path raises ``ValueError`` out of
        ``resolve_workspace_path`` (docker_workspace.py:138-149) and the tools
        do not catch it. Without this backstop a typo'd path aborts the entire
        run instead of giving the model an observation to retry from — see
        Qwen3.5-9B 3d 2026-05-10 where ``/workspace/_rlm_artifacts/parse.py``
        killed the loop.
        """
        spec = get_spec(action.tool)
        executor = get_executor(action.tool)

        # Allocate a per-turn action id (t<turn>.a<idx>).
        idx = self._action_seq_per_turn.get(self.current_turn, 0) + 1
        self._action_seq_per_turn[self.current_turn] = idx
        self.current_action_id = f"t{self.current_turn}.a{idx}"

        try:
            obs = executor(self, action)
        except ValueError as e:
            obs = WorkspaceObservation(tool=action.tool, error=str(e))
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
            "call_id": action.call_id,
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
        spill_rel = f"{_OBS_REL_DIR}/{spill_id}.txt"
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
            self._broker_exec_python = None
        # Bound any pathological orphan ledger entries (fire-and-forget broker
        # calls in user scripts) to a single run lifetime.
        with self._broker_ledger_lock:
            self._broker_ledger.clear()
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
