"""Recursion machinery for the workspace substrate.

Spawns child RLMs from inside a parent workspace. Wired into the parent's
``DockerWorkspaceEnv`` as ``env.recursion_handler`` whenever
``parent.depth < parent.max_depth``. The handler is invoked from two call
sites:

- The host-side ``rlm_query`` action (``rlm/workspace_tools/rlm_query.py``).
- The container-side ``rlm_query`` / ``rlm_query_batched`` Python helpers,
  which arrive at the host via the broker poller in
  ``DockerWorkspaceEnv._do_rlm_query`` / ``_do_rlm_query_batched``.

Per-spawn responsibilities (decisions #8, #20, #24, #25):

1. Allocate ``child_id = "child_{turn}_{idx}"``.
2. Copy parent workspace → sibling child workspace, dropping
   ``recursion.copy_on_spawn_excludes`` and any file larger than
   ``recursion.copy_on_spawn_max_file_bytes``.
3. Wipe the copy's ``_rlm_state`` and ``.git`` so the child env's setup
   produces a fresh state dir + git history.
4. Bring up a child ``DockerWorkspaceEnv`` and a child ``RLM`` reusing the
   parent's ``LMHandler`` (one TCP server services the whole tree).
5. After child setup, mark every copied non-state file as ``role="user"``
   in the child provenance: from the child's perspective, the parent is
   "user" input.
6. Overwrite ``_rlm_query_0.txt`` with the child task body.
7. Run the child to a final answer.
8. Copy *only* the explicitly listed ``final_artifacts`` (from
   ``<artifact path="..." />`` children of the child's ``<final>``) into
   ``parent/_rlm_artifacts/children/<child_id>/...``. Mark each as
   ``role="child"`` in parent provenance.
9. Build the path-mapping observation so the parent model can translate
   any child-relative paths it reads in the answer.

Batched spawns from the in-container ``rlm_query_batched`` helper run on a
``ThreadPoolExecutor`` capped at ``recursion.max_concurrent_subcalls``
(default 5). Each child gets its own container and workspace; nothing is
shared between siblings except the parent's LMHandler.
"""

from __future__ import annotations

import logging
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rlm.core.types import WorkspaceObservation
from rlm.environments.docker_workspace import DockerWorkspaceEnv
from rlm.logger import RLMLogger

if TYPE_CHECKING:
    from rlm.core.lm_handler import LMHandler
    from rlm.core.rlm import RLM

log = logging.getLogger(__name__)


class RecursionHandler:
    """Spawns child RLMs that share the parent's LM handler."""

    def __init__(
        self,
        parent_rlm: RLM,
        parent_env: DockerWorkspaceEnv,
        lm_handler: LMHandler,
    ) -> None:
        self.parent_rlm = parent_rlm
        self.parent_env = parent_env
        self.lm_handler = lm_handler
        self._children_per_turn: dict[int, int] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def spawn(self, child_task: str, action_id: str | None) -> WorkspaceObservation:
        """Run one child RLM. Returns a WorkspaceObservation for the parent."""
        try:
            return self._spawn_one(child_task=child_task, action_id=action_id)
        except Exception as exc:
            log.exception("rlm_query spawn failed")
            return WorkspaceObservation(
                tool="rlm_query",
                error=f"Recursion spawn failed: {exc}",
            )

    def spawn_via_broker(self, child_task: str, action_id: str | None) -> dict[str, Any]:
        """In-container ``rlm_query`` helper entry. Returns broker response payload."""
        obs = self.spawn(child_task=child_task, action_id=action_id)
        # Surface the child trajectory(ies) — already on ``obs.rlm_calls`` from
        # ``_spawn_one`` — into the parent env's ledger so the python tool's
        # observation can include them. The XML rlm_query path sets
        # ``rlm_calls`` directly on its observation; this is the equivalent
        # for the python-action / broker path.
        self.parent_env._append_broker_ledger(action_id, list(obs.rlm_calls))
        if obs.error:
            return {"error": obs.error}
        return {"response": obs.stdout}

    def spawn_via_broker_batched(
        self, child_tasks: list[str], action_id: str | None
    ) -> dict[str, Any]:
        """In-container ``rlm_query_batched`` helper. Bounded thread pool."""
        max_concurrent = self.parent_rlm.workspace_config.recursion.max_concurrent_subcalls
        responses: list[str] = [""] * len(child_tasks)
        chat_completions: list = []
        with ThreadPoolExecutor(max_workers=max(1, max_concurrent)) as ex:
            future_to_idx = {
                ex.submit(self.spawn, task, action_id): idx for idx, task in enumerate(child_tasks)
            }
            for fut in future_to_idx:
                idx = future_to_idx[fut]
                try:
                    obs = fut.result()
                except Exception as exc:  # pragma: no cover — spawn() already wraps
                    responses[idx] = f"Error: Recursion spawn failed: {exc}"
                    continue
                responses[idx] = f"Error: {obs.error}" if obs.error else obs.stdout
                chat_completions.extend(obs.rlm_calls)
        self.parent_env._append_broker_ledger(action_id, chat_completions)
        return {"responses": responses}

    # ------------------------------------------------------------------
    # Single spawn (the workhorse)
    # ------------------------------------------------------------------

    def _spawn_one(self, *, child_task: str, action_id: str | None) -> WorkspaceObservation:
        # The RLM import is local to break the rlm <-> recursion <-> docker_workspace
        # circular import chain (rlm.core.recursion imports DockerWorkspaceEnv;
        # rlm.core.rlm imports both).
        from rlm.core.rlm import RLM

        turn = self.parent_env.current_turn
        child_id = self._next_child_id(turn)
        rcfg = self.parent_rlm.workspace_config.recursion

        parent_root = self.parent_env.workspace_root
        child_workspace = parent_root.parent / f"{parent_root.name}__{child_id}"

        # 1. Copy-on-spawn (sibling directory, NOT a child of parent_root, so the
        #    parent's git tree is unaffected).
        _copy_on_spawn(
            src=parent_root,
            dst=child_workspace,
            excludes=rcfg.copy_on_spawn_excludes,
            max_file_bytes=rcfg.copy_on_spawn_max_file_bytes,
        )

        # 2. Wipe state + git so the child's env.setup() produces fresh ones.
        shutil.rmtree(child_workspace / "_rlm_state", ignore_errors=True)
        shutil.rmtree(child_workspace / ".git", ignore_errors=True)

        # 3. Build child env reusing parent's LM handler address.
        child_env = DockerWorkspaceEnv(
            workspace_config=self.parent_rlm.workspace_config,
            lm_handler_address=self.parent_env.lm_handler_address,
            run_id=child_id,
            depth=self.parent_rlm.depth + 1,
            max_depth=self.parent_rlm.max_depth,
            workspace_root=child_workspace,
        )

        try:
            child_env.setup()

            # 4. Reseed provenance: every copied (non-state) file is "user" from
            #    the child's perspective.
            _seed_user_provenance(child_env)

            # 5. Overwrite root task with the child's task string.
            (child_env.workspace_root / "_rlm_query_0.txt").write_text(child_task, encoding="utf-8")
            child_env.provenance.record_seed(
                "_rlm_query_0.txt", role="user", action_id=None, turn=0
            )
            child_env.provenance.save()

            # 6. Build child RLM (shares parent's config, +1 depth).
            child_rlm = RLM(
                backend=self.parent_rlm.backend,
                backend_kwargs=self.parent_rlm.backend_kwargs,
                workspace_config=self.parent_rlm.workspace_config,
                depth=self.parent_rlm.depth + 1,
                max_depth=self.parent_rlm.max_depth,
                max_iterations=self.parent_rlm.max_iterations,
                max_budget=self.parent_rlm.max_budget,
                max_timeout=self.parent_rlm.max_timeout,
                max_tokens=self.parent_rlm.max_tokens,
                max_errors=self.parent_rlm.max_errors,
                custom_system_prompt=self.parent_rlm.custom_system_prompt,
                # In-memory logger so the child's iterations are exposed via
                # `result.metadata` (consumed below into the parent's
                # observation `rlm_calls`). No on-disk JSONL — child trajectory
                # rides inline in the parent's log.
                logger=RLMLogger(log_dir=None),
                verbose=False,
            )

            # 7. Wire grand-child recursion if depth still permits.
            if child_rlm.depth < child_rlm.max_depth:
                child_env.recursion_handler = RecursionHandler(
                    parent_rlm=child_rlm,
                    parent_env=child_env,
                    lm_handler=self.lm_handler,
                )

            # 8. Run the child. ``_run_loop`` handles iteration; final_artifacts
            #    is captured on the child instance.
            result = child_rlm._run_loop(
                prompt=child_task,
                root_prompt=None,
                lm_handler=self.lm_handler,
                env=child_env,
            )
            final_artifacts = list(child_rlm._last_final_artifacts)

            # 9. Selectively pull artifacts back into parent.
            mapping = self._copy_artifacts_to_parent(
                child_env=child_env,
                child_id=child_id,
                final_artifacts=final_artifacts,
                action_id=action_id,
            )

            # 10. Build the path-mapping observation for the parent.
            obs_text = _format_path_mapping_observation(
                child_answer=result.response,
                mapping=mapping,
            )
            return WorkspaceObservation(
                tool="rlm_query",
                stdout=obs_text,
                artifacts=list(mapping.values()),
                data={
                    "child_id": child_id,
                    "depth": child_rlm.depth,
                    "usage": result.usage_summary.to_dict() if result.usage_summary else {},
                },
                # Full child trajectory rides on `result.metadata.iterations`.
                # Same shape `llm_query` uses, so the visualizer renders both
                # via `ActionCard`'s "Sub-LM Calls" path without branching.
                rlm_calls=[result],
                execution_time=result.execution_time,
            )
        finally:
            try:
                child_env.cleanup()
            except Exception:
                log.exception("Child env cleanup raised (ignored)")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _next_child_id(self, turn: int) -> str:
        with self._lock:
            idx = self._children_per_turn.get(turn, 0) + 1
            self._children_per_turn[turn] = idx
        return f"child_{turn}_{idx}"

    def _copy_artifacts_to_parent(
        self,
        *,
        child_env: DockerWorkspaceEnv,
        child_id: str,
        final_artifacts: list[str],
        action_id: str | None,
    ) -> dict[str, str]:
        mapping: dict[str, str] = {}
        if not final_artifacts:
            return mapping
        parent_root = self.parent_env.workspace_root
        dest_root = parent_root / "_rlm_artifacts" / "children" / child_id
        dest_root.mkdir(parents=True, exist_ok=True)

        child_root = child_env.workspace_root.resolve()

        for child_rel in final_artifacts:
            norm = child_rel.replace("\\", "/").lstrip("./")
            if not norm:
                continue
            child_full = (child_env.workspace_root / norm).resolve()
            try:
                child_full.relative_to(child_root)
            except ValueError:
                # Path escapes child workspace — skip silently. The model
                # should never produce these (artifacts go through the same
                # parser path), so this is purely defensive.
                continue
            if not child_full.exists():
                continue

            parent_rel = f"_rlm_artifacts/children/{child_id}/{norm}"
            parent_full = parent_root / parent_rel
            parent_full.parent.mkdir(parents=True, exist_ok=True)
            if child_full.is_dir():
                shutil.copytree(child_full, parent_full, dirs_exist_ok=True)
            else:
                shutil.copy2(child_full, parent_full)
            mapping[norm] = parent_rel
            self.parent_env.provenance.record_write(
                parent_rel,
                role="child",
                action_id=action_id,
                turn=self.parent_env.current_turn,
            )
        self.parent_env.provenance.save()
        return mapping


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _copy_on_spawn(
    *,
    src: Path,
    dst: Path,
    excludes: tuple[str, ...],
    max_file_bytes: int,
) -> None:
    """Copy ``src`` to ``dst`` with workspace-relative excludes and a per-file
    size cap. Excludes are matched against the workspace-relative path of
    each entry (not just basename), so e.g. ``"_rlm_state/snapshots"`` skips
    only that subtree.
    """
    excludes_norm = tuple(e.replace("\\", "/").rstrip("/") for e in excludes)
    src_resolved = src.resolve()

    def ignore(directory: str, names: list[str]) -> list[str]:
        skipped: list[str] = []
        try:
            dir_rel = Path(directory).resolve().relative_to(src_resolved)
        except ValueError:
            # ``directory`` is somehow outside ``src`` — skip everything.
            return list(names)
        dir_rel_str = "" if str(dir_rel) == "." else str(dir_rel).replace("\\", "/")

        for name in names:
            full_rel = f"{dir_rel_str}/{name}" if dir_rel_str else name
            # Exact-match or prefix-match against any exclude.
            if _matches_exclude(full_rel, excludes_norm) or _matches_exclude(name, excludes_norm):
                skipped.append(name)
                continue
            full = Path(directory) / name
            if full.is_file():
                try:
                    if full.stat().st_size > max_file_bytes:
                        skipped.append(name)
                except OSError:
                    skipped.append(name)
        return skipped

    shutil.copytree(src, dst, ignore=ignore, dirs_exist_ok=False)


def _matches_exclude(rel_path: str, excludes: tuple[str, ...]) -> bool:
    for ex in excludes:
        if rel_path == ex or rel_path.startswith(ex + "/"):
            return True
    return False


def _seed_user_provenance(child_env: DockerWorkspaceEnv) -> None:
    """Mark every non-state file in the child workspace as role=``user``.

    Called once after the child env's ``setup()`` runs (which has already
    seeded ``_rlm_state/*`` as ``system``). State files are skipped so they
    don't get re-stamped to ``user``.
    """
    root = child_env.workspace_root
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        rel_str = str(rel).replace("\\", "/")
        if child_env.is_reserved_path(rel_str):
            continue
        child_env.provenance.record_seed(rel_str, role="user", action_id=None, turn=0)
    child_env.provenance.save()


def _format_path_mapping_observation(*, child_answer: str, mapping: dict[str, str]) -> str:
    lines = [
        "Observation: Child RLM completed.",
        "",
        f"Answer: {child_answer}",
        "",
    ]
    if mapping:
        lines.append(
            "[Runtime Note: The child's exported files have been safely isolated. "
            "Translate any paths mentioned in the answer above using this mapping:]"
        )
        lines.append("Artifact Mapping:")
        for child_rel, parent_rel in mapping.items():
            lines.append(f"- {child_rel} -> {parent_rel}")
    else:
        lines.append("[Runtime Note: child returned no artifacts.]")
    return "\n".join(lines)
