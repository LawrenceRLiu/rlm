"""Unit tests for eval/terminal_bench/runner.py.

Covers ``TaskSpec`` parsing, prompt construction, the grader callback's
failure-mode handling, and the ``run_task`` orchestration (with the RLM
class + docker subprocess calls mocked out). The real end-to-end pass is
covered by running the runner against the 3 Harbor demo tasks; these tests
exist to catch regressions in the pure-Python surface without paying for
the Docker build.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from eval.terminal_bench import runner as runner_mod
from eval.terminal_bench.runner import (
    TaskSpec,
    already_done,
    apply_shard,
    build_prompt,
    discover_tasks,
    make_grader,
    run_task,
)
from rlm.core.types import RLMChatCompletion, UsageSummary
from rlm.environments.docker_workspace import ExecResult

# ---------------------------------------------------------------------------
# TaskSpec.from_dir
# ---------------------------------------------------------------------------


def _write_task_dir(
    base: Path,
    name: str,
    *,
    toml: str,
    instruction: str = "do the thing\n",
    test_script: str = "#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n",
) -> Path:
    d = base / name
    (d / "environment").mkdir(parents=True)
    (d / "tests").mkdir(parents=True)
    (d / "task.toml").write_text(toml)
    (d / "instruction.md").write_text(instruction)
    (d / "environment" / "Dockerfile").write_text("FROM alpine:3.22\nWORKDIR /app\n")
    (d / "tests" / "test.sh").write_text(test_script)
    return d


class TestTaskSpec:
    def test_parses_basic_fields(self, tmp_path: Path) -> None:
        task_dir = _write_task_dir(
            tmp_path,
            "demo",
            toml=(
                "[task]\nname = 'demo'\n"
                "[agent]\ntimeout_sec = 60.0\n"
                "[verifier]\ntimeout_sec = 90.0\n"
            ),
        )
        spec = TaskSpec.from_dir(task_dir)
        assert spec.task_id == "demo"
        assert spec.task_dir == task_dir
        assert spec.agent_timeout_sec == 60.0
        assert spec.verifier_timeout_sec == 90.0
        # Default workdir when [environment].workdir is absent.
        assert spec.workdir == "/app"

    def test_defaults_when_timeouts_missing(self, tmp_path: Path) -> None:
        task_dir = _write_task_dir(tmp_path, "no-timeouts", toml="[task]\nname='x'\n")
        spec = TaskSpec.from_dir(task_dir)
        # Defaults from the Harbor convention we mirror.
        assert spec.agent_timeout_sec == 120.0
        assert spec.verifier_timeout_sec == 120.0
        assert spec.workdir == "/app"

    def test_workdir_override_from_environment_section(self, tmp_path: Path) -> None:
        task_dir = _write_task_dir(
            tmp_path,
            "custom-wd",
            toml=("[task]\nname='x'\n[environment]\nworkdir = '/custom-workdir'\n"),
        )
        spec = TaskSpec.from_dir(task_dir)
        assert spec.workdir == "/custom-workdir"


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_includes_instruction_text(self, tmp_path: Path) -> None:
        task_dir = _write_task_dir(
            tmp_path,
            "p",
            toml="[task]\nname='x'\n",
            instruction="Create hello.txt containing 'Hello'.\n",
        )
        spec = TaskSpec.from_dir(task_dir)
        prompt = build_prompt(spec)
        assert "Create hello.txt containing 'Hello'." in prompt

    def test_advertises_workdir_in_framing(self, tmp_path: Path) -> None:
        task_dir = _write_task_dir(
            tmp_path,
            "p",
            toml="[task]\nname='x'\n[environment]\nworkdir='/custom-workdir'\n",
        )
        spec = TaskSpec.from_dir(task_dir)
        prompt = build_prompt(spec)
        # The custom workdir is referenced in the framing (both in the
        # rule and in the example shell command).
        assert "/custom-workdir" in prompt
        # Reminds the agent the host-side file tools target /workspace.
        assert "/workspace" in prompt
        # Tells the agent how to finish.
        assert "final" in prompt


# ---------------------------------------------------------------------------
# make_grader
# ---------------------------------------------------------------------------


def _grade_env(
    *,
    container_id: str | None = "abc",
    test_result: ExecResult | None = None,
    reward_result: ExecResult | None = None,
    mkdir_result: ExecResult | None = None,
    chmod_result: ExecResult | None = None,
) -> MagicMock:
    """Build a mock DockerWorkspaceEnv with controllable exec_in_container."""
    env = MagicMock()
    env._container_id = container_id

    default_ok = ExecResult(stdout="", stderr="", exit_code=0, timed_out=False, duration=0.0)
    queue = [
        mkdir_result or default_ok,
        chmod_result or default_ok,
        test_result
        or ExecResult(stdout="ok", stderr="", exit_code=0, timed_out=False, duration=0.1),
        reward_result
        or ExecResult(stdout="1\n", stderr="", exit_code=0, timed_out=False, duration=0.0),
    ]
    env.exec_in_container = MagicMock(side_effect=queue)
    return env


class TestMakeGrader:
    def test_pass_when_reward_is_one(self, tmp_path: Path) -> None:
        task_dir = _write_task_dir(tmp_path, "p", toml="[task]\nname='x'\n")
        spec = TaskSpec.from_dir(task_dir)
        grade = make_grader(spec)
        env = _grade_env()

        cp_proc = MagicMock(returncode=0, stdout="", stderr="")
        with patch.object(runner_mod.subprocess, "run", return_value=cp_proc) as run:
            result = grade(env)

        assert result["passed"] is True
        assert result["exit_code"] == 0
        assert result["reward_raw"] == "1"
        # docker cp invoked exactly once with the right shape.
        cp_calls = [c for c in run.call_args_list if c.args[0][:2] == ["docker", "cp"]]
        assert len(cp_calls) == 1
        cp_cmd = cp_calls[0].args[0]
        assert cp_cmd[2] == str(spec.task_dir / "tests")
        assert cp_cmd[3] == "abc:/tests"
        # mkdir + chmod + bash test.sh + cat reward.txt
        assert env.exec_in_container.call_count == 4

    def test_fail_when_reward_is_zero(self, tmp_path: Path) -> None:
        task_dir = _write_task_dir(tmp_path, "p", toml="[task]\nname='x'\n")
        spec = TaskSpec.from_dir(task_dir)
        grade = make_grader(spec)
        env = _grade_env(
            test_result=ExecResult(
                stdout="some failure",
                stderr="",
                exit_code=0,
                timed_out=False,
                duration=0.0,
            ),
            reward_result=ExecResult(
                stdout="0\n", stderr="", exit_code=0, timed_out=False, duration=0.0
            ),
        )
        cp_proc = MagicMock(returncode=0, stdout="", stderr="")
        with patch.object(runner_mod.subprocess, "run", return_value=cp_proc):
            result = grade(env)
        assert result["passed"] is False
        assert result["reward_raw"] == "0"

    def test_fail_when_reward_file_missing(self, tmp_path: Path) -> None:
        """If test.sh exits without writing reward.txt, ``cat`` returns ""
        and we report failure rather than crashing.
        """
        task_dir = _write_task_dir(tmp_path, "p", toml="[task]\nname='x'\n")
        spec = TaskSpec.from_dir(task_dir)
        grade = make_grader(spec)
        env = _grade_env(
            reward_result=ExecResult(
                stdout="",
                stderr="cat: /logs/verifier/reward.txt: No such file or directory",
                exit_code=1,
                timed_out=False,
                duration=0.0,
            ),
        )
        cp_proc = MagicMock(returncode=0, stdout="", stderr="")
        with patch.object(runner_mod.subprocess, "run", return_value=cp_proc):
            result = grade(env)
        assert result["passed"] is False
        assert result["reward_raw"] == ""

    def test_reports_failure_when_container_is_gone(self, tmp_path: Path) -> None:
        task_dir = _write_task_dir(tmp_path, "p", toml="[task]\nname='x'\n")
        spec = TaskSpec.from_dir(task_dir)
        grade = make_grader(spec)
        env = MagicMock()
        env._container_id = None
        result = grade(env)
        assert result["passed"] is False
        assert "container not running" in result["stderr"]
        # No subprocess / exec call was made.
        env.exec_in_container.assert_not_called()

    def test_docker_cp_failure_reported_not_raised(self, tmp_path: Path) -> None:
        task_dir = _write_task_dir(tmp_path, "p", toml="[task]\nname='x'\n")
        spec = TaskSpec.from_dir(task_dir)
        grade = make_grader(spec)
        env = MagicMock()
        env._container_id = "abc"

        cp_proc = MagicMock(returncode=1, stdout="", stderr="no such file")
        with patch.object(runner_mod.subprocess, "run", return_value=cp_proc):
            result = grade(env)
        assert result["passed"] is False
        assert "docker cp failed" in result["stderr"]
        # Should not have attempted to run anything in the container if cp
        # failed.
        env.exec_in_container.assert_not_called()

    def test_grader_timeout_propagates_to_result(self, tmp_path: Path) -> None:
        task_dir = _write_task_dir(tmp_path, "p", toml="[task]\nname='x'\n")
        spec = TaskSpec.from_dir(task_dir)
        grade = make_grader(spec)
        env = _grade_env(
            test_result=ExecResult(
                stdout="",
                stderr="",
                exit_code=-1,
                timed_out=True,
                duration=999.0,
            ),
            reward_result=ExecResult(
                stdout="",  # test.sh didn't get to write reward.txt
                stderr="",
                exit_code=1,
                timed_out=False,
                duration=0.0,
            ),
        )
        cp_proc = MagicMock(returncode=0, stdout="", stderr="")
        with patch.object(runner_mod.subprocess, "run", return_value=cp_proc):
            result = grade(env)
        assert result["timed_out"] is True
        assert result["passed"] is False
        assert result["reward_raw"] == ""

    def test_reward_with_trailing_whitespace_is_parsed(self, tmp_path: Path) -> None:
        """``cat reward.txt`` can return '1\\n' or '  1  ' depending on how
        test.sh writes it. ``.strip()`` should make this robust.
        """
        task_dir = _write_task_dir(tmp_path, "p", toml="[task]\nname='x'\n")
        spec = TaskSpec.from_dir(task_dir)
        grade = make_grader(spec)
        env = _grade_env(
            reward_result=ExecResult(
                stdout="  1  \n", stderr="", exit_code=0, timed_out=False, duration=0.0
            ),
        )
        cp_proc = MagicMock(returncode=0, stdout="", stderr="")
        with patch.object(runner_mod.subprocess, "run", return_value=cp_proc):
            result = grade(env)
        assert result["passed"] is True
        assert result["reward_raw"] == "1"


# ---------------------------------------------------------------------------
# run_task
# ---------------------------------------------------------------------------


def _stub_completion(response: str = "done", grade: dict | None = None) -> RLMChatCompletion:
    c = RLMChatCompletion(
        root_model="fake",
        prompt="x",
        response=response,
        usage_summary=UsageSummary(model_usage_summaries={}),
        execution_time=0.0,
    )
    c.pre_cleanup_result = grade
    return c


class TestRunTask:
    def _spec(self, tmp_path: Path) -> TaskSpec:
        task_dir = _write_task_dir(tmp_path, "demo", toml="[task]\nname='demo'\n")
        return TaskSpec.from_dir(task_dir)

    def test_happy_path_pass(self, tmp_path: Path) -> None:
        spec = self._spec(tmp_path)
        out_dir = tmp_path / "results"

        mock_rlm_instance = MagicMock()
        mock_rlm_instance.completion.return_value = _stub_completion(
            response="done",
            grade={
                "passed": True,
                "exit_code": 0,
                "timed_out": False,
                "stdout": "out",
                "stderr": "",
                "reward_raw": "1",
            },
        )
        with patch.object(runner_mod, "build_task_image", return_value="rlm-tb-demo:t"):
            with patch.object(runner_mod, "RLM", return_value=mock_rlm_instance) as mock_rlm:
                result = run_task(
                    spec,
                    backend="openai",
                    backend_kwargs={"model_name": "fake"},
                    max_iterations=10,
                    output_dir=out_dir,
                )
                # Capture inspection inside the patch context (so mocks are alive).
                rlm_kwargs = mock_rlm.call_args.kwargs
                completion_kwargs = mock_rlm_instance.completion.call_args.kwargs

        assert result.task_id == "demo"
        assert result.passed is True
        assert result.grader_exit_code == 0
        assert result.grader_reward_raw == "1"
        assert result.error is None
        assert rlm_kwargs["workspace_config"].docker.image == "rlm-tb-demo:t"
        assert "pre_cleanup_callback" in completion_kwargs
        assert callable(completion_kwargs["pre_cleanup_callback"])

    def test_grader_failure_recorded(self, tmp_path: Path) -> None:
        spec = self._spec(tmp_path)
        with patch.object(runner_mod, "build_task_image", return_value="t"):
            rlm_instance = MagicMock()
            rlm_instance.completion.return_value = _stub_completion(
                grade={
                    "passed": False,
                    "exit_code": 1,
                    "timed_out": False,
                    "stdout": "fail",
                    "stderr": "",
                    "reward_raw": "0",
                }
            )
            with patch.object(runner_mod, "RLM", return_value=rlm_instance):
                result = run_task(
                    spec,
                    backend="openai",
                    backend_kwargs={"model_name": "fake"},
                    max_iterations=5,
                    output_dir=tmp_path / "results",
                )
        assert result.passed is False
        assert result.grader_exit_code == 1
        assert result.grader_reward_raw == "0"
        assert result.error is None

    def test_agent_exception_caught_and_recorded(self, tmp_path: Path) -> None:
        spec = self._spec(tmp_path)
        with patch.object(runner_mod, "build_task_image", return_value="t"):
            rlm_instance = MagicMock()
            rlm_instance.completion.side_effect = RuntimeError("LM endpoint down")
            with patch.object(runner_mod, "RLM", return_value=rlm_instance):
                result = run_task(
                    spec,
                    backend="openai",
                    backend_kwargs={"model_name": "fake"},
                    max_iterations=5,
                    output_dir=tmp_path / "results",
                )
        assert result.passed is False
        assert result.error is not None
        assert "RuntimeError" in result.error
        assert "LM endpoint down" in result.error
        assert result.wall_clock_s >= 0.0

    def test_pre_cleanup_result_none_treated_as_fail(self, tmp_path: Path) -> None:
        """If for some reason the completion returns without a
        pre_cleanup_result (e.g., callback returned None), run_task should
        still produce a stable ``TaskResult`` reporting failure rather than
        crashing on ``None.get``.
        """
        spec = self._spec(tmp_path)
        with patch.object(runner_mod, "build_task_image", return_value="t"):
            rlm_instance = MagicMock()
            rlm_instance.completion.return_value = _stub_completion(grade=None)
            with patch.object(runner_mod, "RLM", return_value=rlm_instance):
                result = run_task(
                    spec,
                    backend="openai",
                    backend_kwargs={"model_name": "fake"},
                    max_iterations=5,
                    output_dir=tmp_path / "results",
                )
        assert result.passed is False
        assert result.error is None
        assert result.grader_reward_raw == ""

    def test_passes_correct_timeouts_into_workspace_config(self, tmp_path: Path) -> None:
        task_dir = _write_task_dir(
            tmp_path,
            "timed",
            toml="[task]\nname='x'\n[agent]\ntimeout_sec=42.0\n[verifier]\ntimeout_sec=77.0\n",
        )
        spec = TaskSpec.from_dir(task_dir)
        rlm_instance = MagicMock()
        rlm_instance.completion.return_value = _stub_completion(
            grade={
                "passed": True,
                "exit_code": 0,
                "timed_out": False,
                "stdout": "",
                "stderr": "",
                "reward_raw": "1",
            }
        )
        with patch.object(runner_mod, "build_task_image", return_value="t"):
            with patch.object(runner_mod, "RLM", return_value=rlm_instance) as mock_rlm:
                run_task(
                    spec,
                    backend="openai",
                    backend_kwargs={"model_name": "fake"},
                    max_iterations=5,
                    output_dir=tmp_path / "results",
                )
                rlm_kwargs = mock_rlm.call_args.kwargs
        assert rlm_kwargs["workspace_config"].docker.exec_timeout_seconds == 42
        # cleanup_mode is forced to "delete" for batch runs.
        assert rlm_kwargs["workspace_config"].docker.cleanup_mode == "delete"


# ---------------------------------------------------------------------------
# discover_tasks + apply_shard
# ---------------------------------------------------------------------------


class TestDiscoverAndShard:
    def test_discover_finds_nested_task_tomls(self, tmp_path: Path) -> None:
        _write_task_dir(tmp_path, "alpha", toml="[task]\nname='a'\n")
        _write_task_dir(tmp_path, "beta", toml="[task]\nname='b'\n")
        # A nested layout (mirrors TB2's tasks/<category>/<task>/ shape).
        nested = tmp_path / "cat"
        nested.mkdir()
        _write_task_dir(nested, "gamma", toml="[task]\nname='g'\n")
        found = discover_tasks(tmp_path)
        names = [p.name for p in found]
        # Sorted by basename for shard determinism.
        assert names == ["alpha", "beta", "gamma"]

    def test_discover_raises_on_missing_root(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            discover_tasks(tmp_path / "nope")

    def test_apply_shard_partitions_deterministically(self) -> None:
        tasks = [Path(f"/x/t{i}") for i in range(10)]
        s0 = apply_shard(tasks, 0, 4)
        s1 = apply_shard(tasks, 1, 4)
        s2 = apply_shard(tasks, 2, 4)
        s3 = apply_shard(tasks, 3, 4)
        # Every task lands in exactly one shard.
        assert sorted(s0 + s1 + s2 + s3, key=lambda p: p.name) == sorted(
            tasks, key=lambda p: p.name
        )
        # Round-robin by index.
        assert s0 == [tasks[i] for i in (0, 4, 8)]
        assert s1 == [tasks[i] for i in (1, 5, 9)]

    def test_apply_shard_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            apply_shard([Path("/x/a")], -1, 2)
        with pytest.raises(ValueError):
            apply_shard([Path("/x/a")], 2, 2)
        with pytest.raises(ValueError):
            apply_shard([Path("/x/a")], 0, 0)

    def test_num_shards_one_is_identity(self) -> None:
        tasks = [Path(f"/x/t{i}") for i in range(5)]
        assert apply_shard(tasks, 0, 1) == tasks


# ---------------------------------------------------------------------------
# already_done (--resume)
# ---------------------------------------------------------------------------


class TestAlreadyDone:
    def test_returns_false_when_result_missing(self, tmp_path: Path) -> None:
        assert already_done("never-ran", tmp_path) is False

    def test_returns_true_when_result_parses(self, tmp_path: Path) -> None:
        d = tmp_path / "demo"
        d.mkdir()
        (d / "result.json").write_text(json.dumps({"task_id": "demo", "passed": False}))
        assert already_done("demo", tmp_path) is True

    def test_returns_true_even_for_failed_result(self, tmp_path: Path) -> None:
        # A recorded failure still counts as done — we don't want infinite retries.
        d = tmp_path / "demo"
        d.mkdir()
        (d / "result.json").write_text(
            json.dumps({"task_id": "demo", "passed": False, "error": "boom"})
        )
        assert already_done("demo", tmp_path) is True

    def test_returns_false_on_corrupt_json(self, tmp_path: Path) -> None:
        d = tmp_path / "demo"
        d.mkdir()
        (d / "result.json").write_text("{not json")
        assert already_done("demo", tmp_path) is False
