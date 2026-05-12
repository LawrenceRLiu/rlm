"""Terminal-Bench 2.0 / Harbor batch runner.

Given a Harbor-format task directory, this:

  1. Builds the task's base image from ``<task>/environment/Dockerfile``.
  2. Layers the RLM broker on top via ``eval.common.composite_image``.
  3. Spins up an ``RLM`` against the composite image, feeds it the task's
     ``instruction.md`` as the prompt.
  4. Runs the task's canonical grader via a ``pre_cleanup_callback`` (after
     the agent has finished, before the container is torn down).
  5. Records pass/fail + the trajectory pointer.

A Harbor task dir looks like::

    task_dir/
      task.toml                  # metadata + timeouts
      instruction.md             # agent prompt
      environment/Dockerfile     # base image (sets WORKDIR for the task)
      tests/test.sh              # grader: runs checks, writes 0|1 to
                                 #   /logs/verifier/reward.txt
      tests/test_state.py        # optional: pytest cases (driven by test.sh)
      solution/solve.sh          # reference solution (unused)

Grading contract: copy ``tests/`` into the container at ``/tests``, mkdir
``/logs/verifier``, run ``bash /tests/test.sh``, then read
``/logs/verifier/reward.txt``. ``1`` means pass, anything else means fail.
This matches Harbor's official harness so the same grader script works
across all task styles (pytest-based, shell-based, custom).
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import subprocess
import time
import tomllib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from eval.common.composite_image import build_composite
from rlm import RLM
from rlm.core.config import DockerConfig, WorkspaceConfig
from rlm.environments.docker_workspace import DockerWorkspaceEnv, ExecResult
from rlm.logger import RLMLogger

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
HARBOR_ROOT = REPO_ROOT / "third_party" / "harbor"


@dataclass
class TaskSpec:
    """Subset of task.toml fields we use."""

    task_id: str
    task_dir: Path
    agent_timeout_sec: float
    verifier_timeout_sec: float
    workdir: str  # target directory inside the container

    @classmethod
    def from_dir(cls, task_dir: Path) -> TaskSpec:
        meta = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
        agent_timeout = float(meta.get("agent", {}).get("timeout_sec", 120.0))
        verifier_timeout = float(meta.get("verifier", {}).get("timeout_sec", 120.0))
        # WORKDIR is set either via [environment].workdir (Harbor extension)
        # or via the task's Dockerfile. We read it from task.toml when
        # available; otherwise default to /app which is the convention in the
        # Harbor examples we surveyed.
        workdir = meta.get("environment", {}).get("workdir", "/app")
        return cls(
            task_id=task_dir.name,
            task_dir=task_dir,
            agent_timeout_sec=agent_timeout,
            verifier_timeout_sec=verifier_timeout,
            workdir=workdir,
        )


@dataclass
class TaskResult:
    task_id: str
    passed: bool
    grader_exit_code: int | None
    grader_timed_out: bool
    grader_reward_raw: str  # contents of /logs/verifier/reward.txt
    grader_stdout_tail: str  # last ~2KB of test.sh stdout
    grader_stderr_tail: str
    agent_response: str
    agent_turns: int | None
    wall_clock_s: float
    error: str | None = None


def build_task_image(task: TaskSpec) -> str:
    """Build the task's base image then layer the broker on top.

    Returns the composite-image tag.
    """
    base_tag = f"harbor-task-base-{task.task_id}:latest"
    composite_tag = f"rlm-tb-{task.task_id}:latest"

    log.info("Building task base image: %s", base_tag)
    env_dir = task.task_dir / "environment"
    if not (env_dir / "Dockerfile").is_file():
        raise FileNotFoundError(f"No Dockerfile at {env_dir}/Dockerfile")
    result = subprocess.run(
        ["docker", "build", "-t", base_tag, str(env_dir)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"docker build (base) failed for {task.task_id}:\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )

    log.info("Layering broker → %s", composite_tag)
    return build_composite(base_image=base_tag, output_tag=composite_tag)


def build_prompt(task: TaskSpec) -> str:
    """Compose the agent's user prompt from instruction.md plus framing.

    The substrate's bind-mount is fixed at ``/workspace`` (the host-side
    workspace dir), which shadows whatever was at ``/workspace`` in the
    base image. Harbor tasks expect their solution at ``task.workdir``
    (typically ``/app``), so we instruct the agent explicitly.
    """
    instruction = (task.task_dir / "instruction.md").read_text(encoding="utf-8").strip()
    return (
        "You are running inside a Docker container for a Terminal-Bench task.\n"
        f"The grader will evaluate the state of `{task.workdir}` inside this "
        f"container after you finish.\n"
        "\n"
        "Important rules:\n"
        f"- Do your work in `{task.workdir}` using the `shell` tool "
        "(e.g. `cd " + task.workdir + " && ...`).\n"
        "- The `read_file` / `write_file` / `edit_file` tools operate on "
        "`/workspace`, which is NOT the grader's target. Use `shell` for "
        "anything the grader will inspect.\n"
        '- When you are done, emit `<action tool="final"><answer>done'
        "</answer></action>`.\n"
        "\n"
        "Task:\n"
        f"{instruction}\n"
    )


def make_grader(task: TaskSpec) -> Any:
    """Return a pre_cleanup_callback that runs the Harbor-canonical grader.

    Contract:
      1. ``docker cp tests/ -> /tests`` in the live container.
      2. ``mkdir -p /logs/verifier``.
      3. ``bash /tests/test.sh`` (timeout = task.verifier_timeout_sec).
      4. Read ``/logs/verifier/reward.txt``; ``"1"`` = pass.
    """

    def grade(env: DockerWorkspaceEnv) -> dict:
        container_id = env._container_id
        if container_id is None:
            return {
                "passed": False,
                "exit_code": None,
                "timed_out": False,
                "stdout": "",
                "stderr": "container not running at grade time",
                "reward_raw": "",
            }

        tests_src = task.task_dir / "tests"
        cp = subprocess.run(
            ["docker", "cp", str(tests_src), f"{container_id}:/tests"],
            capture_output=True,
            text=True,
        )
        if cp.returncode != 0:
            return {
                "passed": False,
                "exit_code": None,
                "timed_out": False,
                "stdout": cp.stdout,
                "stderr": f"docker cp failed: {cp.stderr}",
                "reward_raw": "",
            }

        env.exec_in_container(
            ["mkdir", "-p", "/logs/verifier"],
            timeout=5,
        )
        env.exec_in_container(
            ["chmod", "+x", "/tests/test.sh"],
            timeout=5,
        )

        result: ExecResult = env.exec_in_container(
            ["bash", "/tests/test.sh"],
            timeout=int(task.verifier_timeout_sec),
        )
        reward = env.exec_in_container(
            ["cat", "/logs/verifier/reward.txt"],
            timeout=5,
        )
        reward_raw = (reward.stdout or "").strip()
        return {
            "passed": reward_raw == "1",
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "reward_raw": reward_raw,
        }

    return grade


def run_task(
    task: TaskSpec,
    *,
    backend: str,
    backend_kwargs: dict,
    max_iterations: int,
    output_dir: Path,
) -> TaskResult:
    log.info("=== Task: %s ===", task.task_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / task.task_id / "trajectory"
    log_dir.mkdir(parents=True, exist_ok=True)

    composite_tag = build_task_image(task)

    workspace_cfg = WorkspaceConfig(
        docker=DockerConfig(
            image=composite_tag,
            cleanup_mode="delete",
            exec_timeout_seconds=int(task.agent_timeout_sec),
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
    prompt = build_prompt(task)

    t0 = time.perf_counter()
    try:
        completion = rlm.completion(prompt, pre_cleanup_callback=make_grader(task))
    except Exception as exc:
        log.exception("Task %s aborted", task.task_id)
        return TaskResult(
            task_id=task.task_id,
            passed=False,
            grader_exit_code=None,
            grader_timed_out=False,
            grader_reward_raw="",
            grader_stdout_tail="",
            grader_stderr_tail="",
            agent_response="",
            agent_turns=None,
            wall_clock_s=time.perf_counter() - t0,
            error=f"{type(exc).__name__}: {exc}",
        )

    wall = time.perf_counter() - t0
    grade = completion.pre_cleanup_result or {}
    turns = None
    if completion.metadata is not None:
        iters = getattr(completion.metadata, "iterations", None)
        if iters is not None:
            turns = len(iters)

    return TaskResult(
        task_id=task.task_id,
        passed=bool(grade.get("passed", False)),
        grader_exit_code=grade.get("exit_code"),
        grader_timed_out=bool(grade.get("timed_out", False)),
        grader_reward_raw=grade.get("reward_raw", ""),
        grader_stdout_tail=(grade.get("stdout") or "")[-2048:],
        grader_stderr_tail=(grade.get("stderr") or "")[-2048:],
        agent_response=completion.response,
        agent_turns=turns,
        wall_clock_s=wall,
    )


def discover_tasks(root: Path) -> list[Path]:
    """Find every subdirectory under ``root`` that contains a task.toml.

    Returns absolute task directories sorted by task_id (basename) for
    deterministic shard partitioning.
    """
    if not root.is_dir():
        raise FileNotFoundError(f"tasks-root does not exist: {root}")
    found: list[Path] = []
    for toml_path in root.rglob("task.toml"):
        if toml_path.is_file():
            found.append(toml_path.parent)
    return sorted(found, key=lambda p: p.name)


def apply_shard(tasks: list[Path], shard_index: int, num_shards: int) -> list[Path]:
    if num_shards <= 0:
        raise ValueError(f"--num-shards must be >= 1, got {num_shards}")
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"--shard-index {shard_index} out of range for --num-shards {num_shards}")
    return [t for i, t in enumerate(tasks) if i % num_shards == shard_index]


def already_done(task_id: str, output_dir: Path) -> bool:
    """A task is 'done' if its result.json exists and parses as JSON.

    We do NOT require ``passed=True`` — a recorded failure (grader exit 1 with
    no exception) is a complete result we don't want to re-run. Only crashes
    that prevented result.json from being written (or wrote garbage) will be
    retried.
    """
    rj = output_dir / task_id / "result.json"
    if not rj.is_file():
        return False
    try:
        json.loads(rj.read_text())
        return True
    except json.JSONDecodeError:
        return False


def rmi_task_images(task_id: str) -> None:
    """Best-effort: remove the per-task base + composite images to bound disk.

    Failures are logged at WARNING and otherwise ignored — image cleanup
    is hygiene, not correctness.
    """
    for tag in (f"rlm-tb-{task_id}:latest", f"harbor-task-base-{task_id}:latest"):
        proc = subprocess.run(
            ["docker", "rmi", tag],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            log.warning("docker rmi %s failed: %s", tag, proc.stderr.strip())


def append_summary_line(summary_path: Path, result: TaskResult) -> None:
    """Append one JSONL line to ``summary.jsonl`` under a file lock.

    Concurrent ProcessPoolExecutor workers may call this; ``fcntl.flock``
    serializes the writes so we don't interleave partial lines.
    """
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(json.dumps(asdict(result)) + "\n")
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _process_one(
    task_dir: Path,
    *,
    backend: str,
    backend_kwargs: dict,
    max_iterations: int,
    output_dir: Path,
    rmi_after: bool,
) -> TaskResult:
    """Worker entry: run a single task end-to-end and persist its result.json.

    Designed to be pickled into ProcessPoolExecutor — takes only plain values.
    """
    task = TaskSpec.from_dir(task_dir)
    try:
        result = run_task(
            task,
            backend=backend,
            backend_kwargs=backend_kwargs,
            max_iterations=max_iterations,
            output_dir=output_dir,
        )
    except Exception as exc:  # safety net; run_task already catches the agent path
        log.exception("Task %s crashed outside run_task", task.task_id)
        result = TaskResult(
            task_id=task.task_id,
            passed=False,
            grader_exit_code=None,
            grader_timed_out=False,
            grader_reward_raw="",
            grader_stdout_tail="",
            grader_stderr_tail="",
            agent_response="",
            agent_turns=None,
            wall_clock_s=0.0,
            error=f"{type(exc).__name__}: {exc}",
        )

    per_task_out = output_dir / result.task_id
    per_task_out.mkdir(parents=True, exist_ok=True)
    (per_task_out / "result.json").write_text(json.dumps(asdict(result), indent=2))

    if rmi_after:
        rmi_task_images(result.task_id)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--tasks",
        nargs="+",
        help="Explicit task directory paths (relative to repo root or absolute).",
    )
    src.add_argument(
        "--tasks-root",
        type=Path,
        help="Root directory to auto-discover task.toml subdirectories under.",
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip any task whose result.json already exists.",
    )
    parser.add_argument(
        "--rmi-after",
        action="store_true",
        help="docker rmi the task's base+composite images after it completes.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Parallel ProcessPoolExecutor workers within this shard (default 1).",
    )
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "eval" / "terminal_bench" / "results",
    )
    parser.add_argument("--backend", default="vllm")
    parser.add_argument("--model", default="gemma-4-31b")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.tasks_root is not None:
        root = args.tasks_root if args.tasks_root.is_absolute() else (REPO_ROOT / args.tasks_root)
        all_tasks = discover_tasks(root.resolve())
    else:
        all_tasks = []
        for task_path in args.tasks:
            task_dir = (
                (REPO_ROOT / task_path).resolve()
                if not Path(task_path).is_absolute()
                else Path(task_path)
            )
            if not (task_dir / "task.toml").is_file():
                log.error("Missing task.toml at %s; skipping", task_dir)
                continue
            all_tasks.append(task_dir)
        all_tasks.sort(key=lambda p: p.name)

    sharded = apply_shard(all_tasks, args.shard_index, args.num_shards)
    log.info(
        "discovered %d tasks total; shard %d/%d -> %d tasks",
        len(all_tasks),
        args.shard_index,
        args.num_shards,
        len(sharded),
    )

    if args.resume:
        before = len(sharded)
        sharded = [t for t in sharded if not already_done(t.name, args.output_dir)]
        log.info("--resume: skipping %d already-complete tasks", before - len(sharded))

    if not sharded:
        log.info("nothing to do; exiting")
        return

    backend_kwargs = {
        "model_name": args.model,
        "base_url": args.base_url,
        "api_key": args.api_key,
    }
    summary_path = args.output_dir / "summary.jsonl"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[TaskResult] = []
    if args.num_workers > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
            futures = {
                ex.submit(
                    _process_one,
                    t,
                    backend=args.backend,
                    backend_kwargs=backend_kwargs,
                    max_iterations=args.max_iterations,
                    output_dir=args.output_dir,
                    rmi_after=args.rmi_after,
                ): t
                for t in sharded
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="tasks"):
                r = fut.result()
                results.append(r)
                append_summary_line(summary_path, r)
    else:
        for t in tqdm(sharded, desc="tasks"):
            r = _process_one(
                t,
                backend=args.backend,
                backend_kwargs=backend_kwargs,
                max_iterations=args.max_iterations,
                output_dir=args.output_dir,
                rmi_after=args.rmi_after,
            )
            results.append(r)
            append_summary_line(summary_path, r)

    passed = sum(1 for r in results if r.passed)
    print("\n=== TB 2.0 substrate validation summary ===")
    print(f"Shard {args.shard_index}/{args.num_shards}  Passed: {passed}/{len(results)}")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        extra = f" (error: {r.error})" if r.error else ""
        print(
            f"  {status}  {r.task_id:30s}  exit={r.grader_exit_code} "
            f"wall={r.wall_clock_s:.1f}s{extra}"
        )


if __name__ == "__main__":
    main()
