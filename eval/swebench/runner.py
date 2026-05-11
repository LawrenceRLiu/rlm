"""SWE-Bench batch runner.

Given a SWE-Bench instance (instance_id, repo, base_commit, problem_statement,
FAIL_TO_PASS, PASS_TO_PASS), this:

  1. Pulls the official SWE-Bench per-instance Docker image
     ``swebench/sweb.eval.x86_64.<sanitized_id>:latest`` (which has the repo
     pre-cloned and checked out at ``base_commit`` at ``/testbed``).
  2. Layers the RLM broker on top via ``eval.common.composite_image``.
  3. Spins up an ``RLM`` against the composite image, feeds it the problem
     statement framed for ``/testbed``.
  4. After the agent finishes (or hits ``max_iterations``), extracts the patch
     in a ``pre_cleanup_callback`` via ``git -C /testbed add -A`` then
     ``git diff --cached <base_commit>``.
  5. Writes per-instance ``prediction.json`` + ``result.json`` shards and
     aggregates into ``predictions.jsonl`` / ``summary.jsonl``.

Designed to scale from a 3-instance smoke test to the full 500-instance
SWE-Bench Verified split. Parallel execution via ``ProcessPoolExecutor``;
``--num-workers 1`` falls back to sequential.

Patch-extraction philosophy: the scaffold owns extraction (Philosophy B,
matching SWE-agent / OpenHands). The agent edits files; ``git add -A &&
git diff --cached <base_commit>`` captures committed, staged, and untracked
changes uniformly. The prompt nudges the agent to avoid git state-changing
commands.

Image-tag sanitization: Docker Hub disallows ``__`` in image tags. SWE-Bench
replaces ``__`` -> ``_1776_`` in instance ids when constructing image tags.
``InstanceSpec.sanitized_id`` applies this.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from eval.common.composite_image import build_composite
from rlm import RLM
from rlm.core.config import DockerConfig, WorkspaceConfig
from rlm.environments.docker_workspace import DockerWorkspaceEnv
from rlm.logger import RLMLogger

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_AGENT_TIMEOUT_SEC = 600.0
DEFAULT_MAX_ITERATIONS = 50
SWEBENCH_DATASET = "princeton-nlp/SWE-bench_Verified"
SWEBENCH_BASE_IMAGE_FMT = "swebench/sweb.eval.x86_64.{sid}:latest"

# SWE-Bench base images put the project's pinned Python + deps inside a conda
# env at /opt/miniconda3/envs/testbed. Activation lives only in /root/.bashrc
# (`conda activate testbed`), which `docker exec` does NOT source — so without
# explicit PATH manipulation, agent shell commands resolve `python` / `pytest`
# / etc. to the base conda env (Python 3.11, no project deps). Injecting these
# ENV vars into the composite image makes the testbed env visible to every
# `docker exec` invocation without needing to source .bashrc. The env name
# "testbed" is the convention SWE-Bench uses across the Verified split.
SWEBENCH_COMPOSITE_ENV: dict[str, str] = {
    "PATH": (
        "/opt/miniconda3/envs/testbed/bin:"
        "/opt/miniconda3/condabin:"
        "/opt/miniconda3/bin:"
        "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    ),
    "CONDA_DEFAULT_ENV": "testbed",
    "CONDA_PREFIX": "/opt/miniconda3/envs/testbed",
}


def _sanitize_instance_id_for_tag(instance_id: str) -> str:
    """SWE-Bench tag-name sanitization: ``__`` -> ``_1776_`` (Docker Hub
    disallows ``__`` in image tags).
    """
    return instance_id.replace("__", "_1776_")


def _image_exists(tag: str) -> bool:
    """Return True iff ``docker image inspect <tag>`` succeeds."""
    result = subprocess.run(
        ["docker", "image", "inspect", tag],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


@dataclass
class InstanceSpec:
    """One SWE-Bench instance.

    ``agent_timeout_sec`` is the per-``docker exec`` timeout (passed through to
    ``DockerConfig.exec_timeout_seconds``), NOT a wall-clock cap for the whole
    rollout. The wall-clock cap is governed by ``--instance-wall-clock-s`` at
    the runner level.
    """

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    fail_to_pass: list[str] = field(default_factory=list)
    pass_to_pass: list[str] = field(default_factory=list)
    agent_timeout_sec: float = DEFAULT_AGENT_TIMEOUT_SEC

    @property
    def sanitized_id(self) -> str:
        return _sanitize_instance_id_for_tag(self.instance_id)

    @property
    def base_image(self) -> str:
        return SWEBENCH_BASE_IMAGE_FMT.format(sid=self.sanitized_id)

    @property
    def composite_image(self) -> str:
        return f"rlm-swebench-{self.sanitized_id}:latest"

    @classmethod
    def from_dataset_row(cls, row: dict[str, Any]) -> InstanceSpec:
        """Construct from a row of ``princeton-nlp/SWE-bench_Verified``.

        ``FAIL_TO_PASS`` / ``PASS_TO_PASS`` are JSON-encoded strings in the
        HF dataset, but already-decoded lists are also accepted.
        """

        def _decode_list(value: Any) -> list[str]:
            if value is None:
                return []
            if isinstance(value, str):
                return list(json.loads(value))
            return list(value)

        return cls(
            instance_id=row["instance_id"],
            repo=row["repo"],
            base_commit=row["base_commit"],
            problem_statement=row["problem_statement"],
            fail_to_pass=_decode_list(row.get("FAIL_TO_PASS")),
            pass_to_pass=_decode_list(row.get("PASS_TO_PASS")),
        )


@dataclass
class InstanceResult:
    """Per-instance outcome.

    Note: no ``passed`` field. Pass/fail is computed externally by
    ``swebench.harness.run_evaluation`` against the emitted ``predictions.jsonl``.
    """

    instance_id: str
    model_patch: str
    patch_extracted: bool
    extraction_error: str | None
    agent_response: str
    agent_turns: int | None
    wall_clock_s: float
    error: str | None = None


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON to ``path`` atomically via tmp-file + ``os.replace``.

    A worker crash between writing the tmp file and the rename leaves
    ``path`` either absent or holding the previous full payload — never
    a half-written file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def pull_base_image(spec: InstanceSpec, *, max_retries: int = 1) -> None:
    """Pull ``spec.base_image`` from Docker Hub, with one retry on failure.

    Skipped if the image is already present locally.
    Raises ``RuntimeError`` on persistent failure.
    """
    if _image_exists(spec.base_image):
        log.info("Base image already present: %s", spec.base_image)
        return

    attempt = 0
    last_stderr = ""
    while attempt <= max_retries:
        log.info("docker pull %s (attempt %d)", spec.base_image, attempt + 1)
        result = subprocess.run(
            ["docker", "pull", spec.base_image],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return
        last_stderr = result.stderr
        log.warning(
            "Pull failed: %s",
            last_stderr.strip().splitlines()[-1] if last_stderr else "(no stderr)",
        )
        attempt += 1
    raise RuntimeError(
        f"docker pull {spec.base_image} failed after {max_retries + 1} attempts:\n{last_stderr}"
    )


def build_task_image(spec: InstanceSpec, *, cache: bool = True) -> str:
    """Ensure the base image is local, then layer the broker on top.

    Returns the composite-image tag (``spec.composite_image``). Injects
    ``SWEBENCH_COMPOSITE_ENV`` into the image so ``docker exec`` resolves
    `python`/`pytest`/etc. to the project's conda env at
    ``/opt/miniconda3/envs/testbed``.
    """
    pull_base_image(spec)
    log.info("Layering broker → %s", spec.composite_image)
    return build_composite(
        base_image=spec.base_image,
        output_tag=spec.composite_image,
        cache=cache,
        extra_env=SWEBENCH_COMPOSITE_ENV,
    )


def build_prompt(spec: InstanceSpec, *, max_fail_to_pass_listed: int = 20) -> str:
    """Compose the user prompt for one SWE-Bench instance.

    Frames ``/testbed`` as the workdir, warns that ``read_file`` / ``write_file``
    / ``edit_file`` operate on ``/workspace`` (not ``/testbed``), tells the
    agent to leave git state alone (the scaffold extracts the patch), and
    includes the problem statement plus FAIL_TO_PASS targets.
    """
    parts: list[str] = []
    parts.append(
        f"You are running inside a Docker container for SWE-Bench instance "
        f"`{spec.instance_id}` (repo: `{spec.repo}`)."
    )
    parts.append(
        f"The repository is at `/testbed`, already checked out at base commit "
        f"`{spec.base_commit}`. Your job is to modify source files in `/testbed` "
        "so the failing tests pass without breaking the passing tests."
    )
    parts.append("")
    parts.append("Important rules:")
    parts.append("- Do all work in `/testbed` using the `shell` tool (e.g. `cd /testbed && ...`).")
    parts.append(
        "- WARNING: the `read_file` / `write_file` / `edit_file` tools operate on "
        "`/workspace`, NOT `/testbed`. They will NOT modify the repo. Use `shell` "
        "(with `cat`, heredocs, `sed`, or `python -c`) for any file edits to the repo."
    )
    parts.append(
        "- Focus on editing source files. Do NOT run `git add`, `git commit`, "
        "`git reset`, `git checkout`, or other git state-changing commands — the "
        "scaffold extracts the patch automatically after you finish. Inspection "
        "commands (`git status`, `git diff`, `git log`) are fine."
    )
    parts.append(
        '- When you are done, emit `<action tool="final"><answer>done</answer></action>`. '
        "The patch is auto-extracted from `git diff`; you do not need to print it."
    )
    parts.append("")
    parts.append("Problem statement:")
    parts.append(spec.problem_statement.strip())

    if spec.fail_to_pass:
        parts.append("")
        parts.append("Tests that must pass after your fix:")
        listed = spec.fail_to_pass[:max_fail_to_pass_listed]
        for name in listed:
            parts.append(f"- {name}")
        if len(spec.fail_to_pass) > len(listed):
            parts.append(f"- ... and {len(spec.fail_to_pass) - len(listed)} more")
    parts.append("")
    return "\n".join(parts)


def make_grader(spec: InstanceSpec) -> Any:
    """Return a ``pre_cleanup_callback`` that extracts the agent's patch.

    Sequence (run via ``docker exec`` against the live container's
    ``/testbed`` — an ephemeral repo inside the per-instance container,
    NOT the host substrate repo):

      1. ``git -C /testbed add -A``  (stages everything incl. untracked files)
      2. ``git -C /testbed diff --cached <base_commit>``  (relative to base)

    Returns a dict ``{model_patch, patch_extracted, extraction_error}`` which
    becomes ``RLMChatCompletion.pre_cleanup_result``.
    """

    def grade(env: DockerWorkspaceEnv) -> dict[str, Any]:
        if env._container_id is None:
            return {
                "model_patch": "",
                "patch_extracted": False,
                "extraction_error": "container not running at grade time",
            }

        # Stage all changes including new untracked files. Safe under
        # cleanup_mode="delete": the container is destroyed right after.
        env.exec_in_container(
            ["git", "-C", "/testbed", "add", "-A"],
            timeout=30,
        )

        diff = env.exec_in_container(
            ["git", "-C", "/testbed", "diff", "--cached", spec.base_commit],
            timeout=60,
        )
        if diff.timed_out:
            return {
                "model_patch": "",
                "patch_extracted": False,
                "extraction_error": "git diff timed out",
            }
        if diff.exit_code != 0:
            return {
                "model_patch": "",
                "patch_extracted": False,
                "extraction_error": (
                    f"git diff failed (exit={diff.exit_code}): {(diff.stderr or '')[:500]}"
                ),
            }

        patch = diff.stdout or ""
        return {
            "model_patch": patch,
            "patch_extracted": bool(patch.strip()),
            "extraction_error": None if patch.strip() else "empty patch (no changes detected)",
        }

    return grade


def run_instance(
    spec: InstanceSpec,
    *,
    backend: str,
    backend_kwargs: dict[str, Any],
    max_iterations: int,
    output_dir: Path,
    cache: bool = True,
) -> InstanceResult:
    """Build composite image, run RLM with patch-extraction callback, package result."""
    log.info("=== Instance: %s ===", spec.instance_id)
    instance_dir = output_dir / spec.instance_id
    log_dir = instance_dir / "trajectory"
    log_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    try:
        composite_tag = build_task_image(spec, cache=cache)
    except Exception as exc:
        log.exception("Image build failed for %s", spec.instance_id)
        return InstanceResult(
            instance_id=spec.instance_id,
            model_patch="",
            patch_extracted=False,
            extraction_error=None,
            agent_response="",
            agent_turns=None,
            wall_clock_s=time.perf_counter() - t0,
            error=f"image build failed: {type(exc).__name__}: {exc}",
        )

    workspace_cfg = WorkspaceConfig(
        docker=DockerConfig(
            image=composite_tag,
            cleanup_mode="delete",
            exec_timeout_seconds=int(spec.agent_timeout_sec),
        )
    )
    rlm = RLM(
        backend=backend,
        backend_kwargs=backend_kwargs,
        workspace_config=workspace_cfg,
        logger=RLMLogger(log_dir=str(log_dir)),
        max_iterations=max_iterations,
        verbose=True,
    )
    prompt = build_prompt(spec)

    try:
        completion = rlm.completion(prompt, pre_cleanup_callback=make_grader(spec))
    except Exception as exc:
        log.exception("Instance %s aborted", spec.instance_id)
        return InstanceResult(
            instance_id=spec.instance_id,
            model_patch="",
            patch_extracted=False,
            extraction_error=None,
            agent_response="",
            agent_turns=None,
            wall_clock_s=time.perf_counter() - t0,
            error=f"{type(exc).__name__}: {exc}",
        )

    wall = time.perf_counter() - t0
    grade = completion.pre_cleanup_result or {}
    turns: int | None = None
    if completion.metadata is not None:
        iters = getattr(completion.metadata, "iterations", None)
        if iters is not None:
            turns = len(iters)

    return InstanceResult(
        instance_id=spec.instance_id,
        model_patch=grade.get("model_patch", "") or "",
        patch_extracted=bool(grade.get("patch_extracted", False)),
        extraction_error=grade.get("extraction_error"),
        agent_response=completion.response or "",
        agent_turns=turns,
        wall_clock_s=wall,
    )


def _load_instances_from_hf(
    dataset: str,
    split: str,
    *,
    instance_ids: list[str] | None = None,
    repos: list[str] | None = None,
    limit: int | None = None,
) -> list[InstanceSpec]:
    """Load + filter the SWE-Bench dataset from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The `datasets` package is required. Install via "
            "`pip install -e '.[eval]'` or `pip install datasets`."
        ) from exc

    log.info("Loading dataset %s split=%s", dataset, split)
    rows = load_dataset(dataset, split=split)

    # Apply filters in order: instance_ids, repos, limit.
    selected: list[InstanceSpec] = []
    instance_id_set = set(instance_ids) if instance_ids else None
    repo_set = set(repos) if repos else None
    for row in rows:
        if instance_id_set is not None and row["instance_id"] not in instance_id_set:
            continue
        if repo_set is not None and row["repo"] not in repo_set:
            continue
        selected.append(InstanceSpec.from_dataset_row(row))
        if limit is not None and len(selected) >= limit:
            break

    if instance_id_set is not None:
        missing = instance_id_set - {s.instance_id for s in selected}
        if missing:
            log.warning(
                "Requested instance_ids not found in %s/%s: %s",
                dataset,
                split,
                sorted(missing),
            )
    return selected


def _load_instances_from_file(path: Path) -> list[InstanceSpec]:
    """Load instances from a JSON file (list of partial-spec dicts)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a JSON list of instance dicts")
    return [InstanceSpec.from_dataset_row(row) for row in raw]


def _scan_completed_instances(output_dir: Path) -> dict[str, dict[str, Any]]:
    """Walk ``output_dir/<id>/prediction.json`` shards. Returns id -> payload.

    Tolerates missing/malformed shards (logs + skips).
    """
    completed: dict[str, dict[str, Any]] = {}
    if not output_dir.is_dir():
        return completed
    for child in sorted(output_dir.iterdir()):
        if not child.is_dir():
            continue
        shard = child / "prediction.json"
        if not shard.is_file():
            continue
        try:
            payload = json.loads(shard.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("Skipping malformed shard %s: %s", shard, exc)
            continue
        iid = payload.get("instance_id")
        if not iid:
            log.warning("Shard %s missing instance_id; skipping", shard)
            continue
        completed[iid] = payload
    return completed


def _aggregate_predictions(output_dir: Path) -> Path:
    """Rewrite ``output_dir/predictions.jsonl`` from per-instance shards."""
    out_path = output_dir / "predictions.jsonl"
    completed = _scan_completed_instances(output_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for iid in sorted(completed):
            f.write(json.dumps(completed[iid]) + "\n")
    return out_path


def _aggregate_summary(output_dir: Path) -> Path:
    """Rewrite ``output_dir/summary.jsonl`` from per-instance result.json shards."""
    out_path = output_dir / "summary.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if output_dir.is_dir():
        for child in sorted(output_dir.iterdir()):
            if not child.is_dir():
                continue
            shard = child / "result.json"
            if not shard.is_file():
                continue
            try:
                rows.append(json.loads(shard.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError) as exc:
                log.warning("Skipping malformed result shard %s: %s", shard, exc)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return out_path


def _run_one(spec_payload: dict[str, Any], runner_args: dict[str, Any]) -> str:
    """Process-pool worker entry point.

    Returns the instance_id. Writes per-instance shards directly to disk.
    Does not return the result object — the main process re-reads the shard.
    """
    spec = InstanceSpec(**spec_payload)
    output_dir = Path(runner_args["output_dir"])
    instance_dir = output_dir / spec.instance_id
    instance_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = run_instance(
            spec,
            backend=runner_args["backend"],
            backend_kwargs=runner_args["backend_kwargs"],
            max_iterations=runner_args["max_iterations"],
            output_dir=output_dir,
            cache=runner_args.get("cache", True),
        )
    except Exception as exc:
        # Last-ditch capture; run_instance should normally catch internally.
        result = InstanceResult(
            instance_id=spec.instance_id,
            model_patch="",
            patch_extracted=False,
            extraction_error=None,
            agent_response="",
            agent_turns=None,
            wall_clock_s=0.0,
            error=f"worker crashed: {type(exc).__name__}: {exc}",
        )

    _atomic_write_json(instance_dir / "result.json", asdict(result))
    _atomic_write_json(
        instance_dir / "prediction.json",
        {
            "instance_id": result.instance_id,
            "model_name_or_path": runner_args["model_name_or_path"],
            "model_patch": result.model_patch,
        },
    )

    if runner_args.get("rmi_composite_after"):
        subprocess.run(
            ["docker", "rmi", spec.composite_image],
            capture_output=True,
            text=True,
        )
    if runner_args.get("rmi_base_after"):
        subprocess.run(
            ["docker", "rmi", spec.base_image],
            capture_output=True,
            text=True,
        )
    return spec.instance_id


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SWE-Bench batch runner. Defaults to the full Verified test split.",
        epilog=(
            "Disk note: each SWE-Bench base image is 1-3 GB. A full 500-instance "
            "run downloads ~0.5-1.5 TB cumulatively. Pass --rmi-base-after on "
            "disk-constrained hosts (deletes the base image after each instance). "
            "Network note: --num-workers above the LM backend's QPS ceiling will "
            "drop effective throughput."
        ),
    )
    # Instance selection
    parser.add_argument("--dataset", default=SWEBENCH_DATASET)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--instance-ids",
        nargs="+",
        help="Filter to a specific subset of instance_ids (smoke-test path).",
    )
    parser.add_argument(
        "--instances-file",
        type=Path,
        help="Load instances from a local JSON file instead of HF (escape hatch).",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Cap to first N instances after other filters."
    )
    parser.add_argument("--repos", nargs="+", help="Filter by repo (e.g. sympy/sympy).")

    # Execution
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(8, max(1, (os.cpu_count() or 2) // 2)),
        help="Parallel worker processes. 1 = sequential.",
    )
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Rebuild composite images even if tag exists locally.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-run instances even if a prediction.json shard exists.",
    )
    parser.add_argument(
        "--rmi-composite-after",
        action="store_true",
        help="docker rmi the composite image after each instance.",
    )
    parser.add_argument(
        "--rmi-base-after",
        action="store_true",
        help="docker rmi the base image after each instance (more aggressive).",
    )
    parser.add_argument(
        "--instance-wall-clock-s",
        type=float,
        default=None,
        help="Optional per-instance wall-clock cap (worker killed on timeout).",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "eval" / "swebench" / "results",
    )
    parser.add_argument(
        "--model-name-or-path",
        default=None,
        help="Tag for predictions.jsonl entries (default: rlm-substrate__<model>).",
    )

    # LM backend
    parser.add_argument("--backend", default="vllm")
    parser.add_argument("--model", default="gemma-4-31b")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--log-level", default="INFO")
    return parser


def _load_instances(args: argparse.Namespace) -> list[InstanceSpec]:
    if args.instances_file is not None:
        return _load_instances_from_file(args.instances_file)
    return _load_instances_from_hf(
        args.dataset,
        args.split,
        instance_ids=args.instance_ids,
        repos=args.repos,
        limit=args.limit,
    )


def _print_progress(done: int, total: int, n_patches: int, n_errors: int, elapsed: float) -> None:
    rate = done / elapsed if elapsed > 0 else 0.0
    log.info(
        "[progress] %d/%d done | %d with patches | %d errors | %.1fs elapsed | %.2f inst/s",
        done,
        total,
        n_patches,
        n_errors,
        elapsed,
        rate,
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_name_or_path = args.model_name_or_path or f"rlm-substrate__{args.model}"
    backend_kwargs: dict[str, Any] = {
        "model_name": args.model,
        "base_url": args.base_url,
        "api_key": args.api_key,
    }

    # Load + filter instances.
    instances = _load_instances(args)
    if not instances:
        log.error("No instances selected after filters. Exiting.")
        return 1

    # Resume: scan existing shards, filter out already-completed.
    if not args.no_resume:
        completed = _scan_completed_instances(args.output_dir)
        before = len(instances)
        instances = [s for s in instances if s.instance_id not in completed]
        skipped = before - len(instances)
        if skipped:
            log.info("[resume] skipping %d already-completed instances", skipped)
        # Bring predictions.jsonl in sync with current shards before starting.
        _aggregate_predictions(args.output_dir)

    if not instances:
        log.info("All requested instances already completed; aggregating and exiting.")
        _aggregate_predictions(args.output_dir)
        _aggregate_summary(args.output_dir)
        return 0

    log.info("Dispatching %d instances across %d workers", len(instances), args.num_workers)

    runner_args = {
        "output_dir": str(args.output_dir),
        "backend": args.backend,
        "backend_kwargs": backend_kwargs,
        "max_iterations": args.max_iterations,
        "cache": not args.no_cache,
        "model_name_or_path": model_name_or_path,
        "rmi_composite_after": args.rmi_composite_after,
        "rmi_base_after": args.rmi_base_after,
    }

    t0 = time.perf_counter()
    n_done = 0
    n_patches = 0
    n_errors = 0
    stopping = False

    def _on_sigint(_signum: int, _frame: Any) -> None:
        nonlocal stopping
        if not stopping:
            log.warning("SIGINT received; finishing in-flight workers then aggregating.")
            stopping = True
        else:
            log.warning("Second SIGINT; aborting hard.")
            sys.exit(130)

    prev_handler = signal.signal(signal.SIGINT, _on_sigint)
    try:
        if args.num_workers <= 1:
            # Sequential path: easier logs for smoke runs.
            for spec in instances:
                if stopping:
                    break
                try:
                    _run_one(asdict(spec), runner_args)
                except Exception:
                    log.exception("Worker failed for %s", spec.instance_id)
                    n_errors += 1
                n_done += 1
                # Re-read shard to bump patches counter.
                shard = args.output_dir / spec.instance_id / "prediction.json"
                if shard.is_file():
                    try:
                        payload = json.loads(shard.read_text(encoding="utf-8"))
                        if (payload.get("model_patch") or "").strip():
                            n_patches += 1
                    except (OSError, json.JSONDecodeError):
                        pass
                _print_progress(
                    n_done, len(instances), n_patches, n_errors, time.perf_counter() - t0
                )
        else:
            with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
                futures = {
                    pool.submit(_run_one, asdict(spec), runner_args): spec for spec in instances
                }
                try:
                    for fut in as_completed(futures):
                        spec = futures[fut]
                        if stopping:
                            # Allow already-running futures to drain.
                            pass
                        try:
                            iid = fut.result(timeout=args.instance_wall_clock_s)
                        except Exception:
                            log.exception("Worker failed for %s", spec.instance_id)
                            iid = spec.instance_id
                            n_errors += 1
                        n_done += 1
                        shard = args.output_dir / iid / "prediction.json"
                        if shard.is_file():
                            try:
                                payload = json.loads(shard.read_text(encoding="utf-8"))
                                if (payload.get("model_patch") or "").strip():
                                    n_patches += 1
                            except (OSError, json.JSONDecodeError):
                                pass
                        _print_progress(
                            n_done,
                            len(instances),
                            n_patches,
                            n_errors,
                            time.perf_counter() - t0,
                        )
                finally:
                    # ProcessPoolExecutor cleanup is handled by `with`.
                    pass
    finally:
        signal.signal(signal.SIGINT, prev_handler)

    # Final aggregation.
    predictions_path = _aggregate_predictions(args.output_dir)
    summary_path = _aggregate_summary(args.output_dir)

    wall = time.perf_counter() - t0
    log.info("=== SWE-Bench run summary ===")
    log.info("Instances dispatched: %d", len(instances))
    log.info("Patches extracted (non-empty): %d", n_patches)
    log.info("Errors: %d", n_errors)
    log.info("Wall time: %.1fs", wall)
    log.info("Predictions: %s", predictions_path)
    log.info("Summary:     %s", summary_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
