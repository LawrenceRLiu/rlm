"""Unit tests for eval/swebench/runner.py.

Covers ``InstanceSpec`` construction (incl. dataset-row parsing + tag
sanitization), prompt framing, the patch-extraction grader callback (incl.
documented behavior for scratch files + committed changes), image acquisition,
``run_instance`` orchestration (with RLM + docker mocked), and the
predictions/resume/main plumbing.

The real end-to-end pass is the 3-instance smoke run; these tests catch
regressions in the pure-Python surface without paying for the Docker build.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from eval.swebench import runner as runner_mod
from eval.swebench.runner import (
    SWEBENCH_COMPOSITE_ENV,
    InstanceSpec,
    _atomic_write_json,
    _scan_completed_instances,
    build_prompt,
    build_task_image,
    make_grader,
    pull_base_image,
    run_instance,
)
from rlm.core.types import RLMChatCompletion, UsageSummary
from rlm.environments.docker_workspace import ExecResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spec(
    instance_id: str = "django__django-12345",
    *,
    repo: str = "django/django",
    base_commit: str = "deadbeef" * 5,
    problem_statement: str = "Fix the bug.",
    fail_to_pass: list[str] | None = None,
    pass_to_pass: list[str] | None = None,
) -> InstanceSpec:
    return InstanceSpec(
        instance_id=instance_id,
        repo=repo,
        base_commit=base_commit,
        problem_statement=problem_statement,
        fail_to_pass=fail_to_pass or [],
        pass_to_pass=pass_to_pass or [],
    )


def _stub_completion(
    response: str = "done",
    pre_cleanup_result: dict | None = None,
) -> RLMChatCompletion:
    c = RLMChatCompletion(
        root_model="fake",
        prompt="x",
        response=response,
        usage_summary=UsageSummary(model_usage_summaries={}),
        execution_time=0.0,
    )
    c.pre_cleanup_result = pre_cleanup_result
    return c


def _ok_exec(stdout: str = "", stderr: str = "", exit_code: int = 0) -> ExecResult:
    return ExecResult(
        stdout=stdout, stderr=stderr, exit_code=exit_code, timed_out=False, duration=0.0
    )


# ---------------------------------------------------------------------------
# InstanceSpec
# ---------------------------------------------------------------------------


class TestInstanceSpec:
    def test_sanitized_id_replaces_double_underscore(self) -> None:
        spec = _spec(instance_id="django__django-12345")
        # Docker Hub disallows __ in tag names; SWE-Bench's convention is
        # __ -> _1776_.
        assert spec.sanitized_id == "django_1776_django-12345"

    def test_base_image_tag_construction(self) -> None:
        spec = _spec(instance_id="sympy__sympy-20590")
        assert spec.base_image == "swebench/sweb.eval.x86_64.sympy_1776_sympy-20590:latest"

    def test_composite_image_tag_construction(self) -> None:
        spec = _spec(instance_id="psf__requests-1142")
        assert spec.composite_image == "rlm-swebench-psf_1776_requests-1142:latest"

    def test_from_dataset_row_parses_json_strings(self) -> None:
        # The HF dataset stores FAIL_TO_PASS / PASS_TO_PASS as JSON-encoded
        # strings (a peculiarity of the parquet schema).
        row = {
            "instance_id": "x__y-1",
            "repo": "x/y",
            "base_commit": "abc",
            "problem_statement": "fix it",
            "FAIL_TO_PASS": '["test_a", "test_b"]',
            "PASS_TO_PASS": '["test_c"]',
        }
        spec = InstanceSpec.from_dataset_row(row)
        assert spec.fail_to_pass == ["test_a", "test_b"]
        assert spec.pass_to_pass == ["test_c"]

    def test_from_dataset_row_accepts_already_decoded_lists(self) -> None:
        row = {
            "instance_id": "x__y-1",
            "repo": "x/y",
            "base_commit": "abc",
            "problem_statement": "fix it",
            "FAIL_TO_PASS": ["test_a"],
            "PASS_TO_PASS": ["test_c"],
        }
        spec = InstanceSpec.from_dataset_row(row)
        assert spec.fail_to_pass == ["test_a"]
        assert spec.pass_to_pass == ["test_c"]

    def test_from_dataset_row_tolerates_missing_test_lists(self) -> None:
        row = {
            "instance_id": "x__y-1",
            "repo": "x/y",
            "base_commit": "abc",
            "problem_statement": "fix it",
        }
        spec = InstanceSpec.from_dataset_row(row)
        assert spec.fail_to_pass == []
        assert spec.pass_to_pass == []


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_includes_problem_statement(self) -> None:
        spec = _spec(problem_statement="Reproduce the segfault on import.")
        prompt = build_prompt(spec)
        assert "Reproduce the segfault on import." in prompt

    def test_advertises_testbed_workdir(self) -> None:
        spec = _spec(base_commit="cafebabe123")
        prompt = build_prompt(spec)
        assert "/testbed" in prompt
        # base commit is mentioned for orientation.
        assert "cafebabe123" in prompt

    def test_warns_about_workspace_vs_testbed(self) -> None:
        prompt = build_prompt(_spec())
        # The substrate's `read_file`/`write_file`/`edit_file` tools target
        # /workspace, NOT /testbed. The agent must use `shell`.
        assert "/workspace" in prompt
        assert "read_file" in prompt
        assert "write_file" in prompt

    def test_warns_against_git_state_commands(self) -> None:
        prompt = build_prompt(_spec())
        # The scaffold handles patch extraction; the agent should not touch
        # git state itself.
        assert "git add" in prompt
        assert "git commit" in prompt
        assert "scaffold" in prompt

    def test_lists_fail_to_pass_when_present(self) -> None:
        spec = _spec(fail_to_pass=["t::test_a", "t::test_b"])
        prompt = build_prompt(spec)
        assert "t::test_a" in prompt
        assert "t::test_b" in prompt

    def test_truncates_long_fail_to_pass_list(self) -> None:
        spec = _spec(fail_to_pass=[f"t::test_{i}" for i in range(50)])
        prompt = build_prompt(spec, max_fail_to_pass_listed=5)
        # The first five are listed, the remainder summarized.
        assert "t::test_0" in prompt
        assert "t::test_4" in prompt
        assert "t::test_49" not in prompt
        assert "45 more" in prompt

    def test_handles_empty_fail_to_pass(self) -> None:
        prompt = build_prompt(_spec(fail_to_pass=[]))
        # No "Tests that must pass" section when empty.
        assert "Tests that must pass" not in prompt


# ---------------------------------------------------------------------------
# make_grader
# ---------------------------------------------------------------------------


def _grade_env(
    *,
    container_id: str | None = "abc",
    add_result: ExecResult | None = None,
    diff_result: ExecResult | None = None,
) -> MagicMock:
    """Mock DockerWorkspaceEnv with controllable exec_in_container."""
    env = MagicMock()
    env._container_id = container_id
    queue = [
        add_result or _ok_exec(),
        diff_result or _ok_exec(stdout="diff --git a/src/foo.py b/src/foo.py\n"),
    ]
    env.exec_in_container = MagicMock(side_effect=queue)
    return env


class TestMakeGrader:
    def test_extracts_patch_from_git_diff(self) -> None:
        spec = _spec()
        grade = make_grader(spec)
        env = _grade_env(
            diff_result=_ok_exec(
                stdout="diff --git a/src/foo.py b/src/foo.py\n@@ -1 +1 @@\n-x\n+y\n"
            ),
        )
        result = grade(env)
        assert result["patch_extracted"] is True
        assert result["model_patch"].startswith("diff --git ")
        assert result["extraction_error"] is None

    def test_runs_git_add_before_diff(self) -> None:
        spec = _spec()
        grade = make_grader(spec)
        env = _grade_env()
        grade(env)
        calls = env.exec_in_container.call_args_list
        assert len(calls) == 2
        # First call is `git add -A`.
        assert calls[0].args[0][:3] == ["git", "-C", "/testbed"]
        assert "add" in calls[0].args[0]
        # Second call is `git diff --cached <base_commit>`.
        assert "diff" in calls[1].args[0]

    def test_uses_base_commit_in_diff_command(self) -> None:
        spec = _spec(base_commit="abcd1234ef")
        grade = make_grader(spec)
        env = _grade_env()
        grade(env)
        diff_call = env.exec_in_container.call_args_list[1]
        # The base commit appears as a positional arg in the diff exec.
        assert "abcd1234ef" in diff_call.args[0]
        assert "--cached" in diff_call.args[0]

    def test_empty_patch_marked_not_extracted(self) -> None:
        spec = _spec()
        grade = make_grader(spec)
        env = _grade_env(diff_result=_ok_exec(stdout=""))
        result = grade(env)
        assert result["patch_extracted"] is False
        assert result["model_patch"] == ""
        assert result["extraction_error"] is not None
        assert "empty patch" in result["extraction_error"]

    def test_git_diff_exit_nonzero_recorded(self) -> None:
        spec = _spec()
        grade = make_grader(spec)
        env = _grade_env(
            diff_result=_ok_exec(stdout="", stderr="fatal: bad object", exit_code=128),
        )
        result = grade(env)
        assert result["patch_extracted"] is False
        assert result["model_patch"] == ""
        assert "fatal: bad object" in result["extraction_error"]
        assert "exit=128" in result["extraction_error"]

    def test_git_diff_timeout_recorded(self) -> None:
        spec = _spec()
        grade = make_grader(spec)
        timed_out = ExecResult(stdout="", stderr="", exit_code=-1, timed_out=True, duration=999.0)
        env = _grade_env(diff_result=timed_out)
        result = grade(env)
        assert result["patch_extracted"] is False
        assert "timed out" in result["extraction_error"]

    def test_reports_failure_when_container_is_gone(self) -> None:
        spec = _spec()
        grade = make_grader(spec)
        env = MagicMock()
        env._container_id = None
        result = grade(env)
        assert result["patch_extracted"] is False
        assert "container not running" in result["extraction_error"]
        env.exec_in_container.assert_not_called()

    def test_scratch_files_included_in_extracted_patch(self) -> None:
        """Philosophy B: ``git add -A`` stages everything, so the extracted
        patch includes new files the agent created but didn't intend as part
        of the submission (debug scratch, etc.).

        Per user direction, scratch files should be harmless to the SWE-Bench
        harness. This test pins the behavior so regressions are caught — and
        if the smoke run shows the harness rejects scratch-file-bearing
        patches we'll know to revisit Philosophy B.
        """
        spec = _spec()
        grade = make_grader(spec)
        # Simulated diff that includes both a legitimate edit AND a brand-new
        # scratch file the agent forgot to delete.
        diff_text = (
            "diff --git a/src/foo.py b/src/foo.py\n"
            "@@ -1 +1 @@\n-x\n+y\n"
            "diff --git a/scratch_debug.py b/scratch_debug.py\n"
            "new file mode 100644\n"
            "@@ -0,0 +1 @@\n+print('hi')\n"
        )
        env = _grade_env(diff_result=_ok_exec(stdout=diff_text))
        result = grade(env)
        assert result["patch_extracted"] is True
        # Both hunks are in the model_patch, verbatim.
        assert "src/foo.py" in result["model_patch"]
        assert "scratch_debug.py" in result["model_patch"]

    def test_committed_changes_included_via_diff_cached(self) -> None:
        """If the agent ignores the prompt nudge and runs ``git commit``, the
        ``git add -A`` is a no-op (nothing unstaged) but ``git diff --cached
        <base_commit>`` still shows the committed changes because ``--cached``
        diffs the index (which contains the committed tree) against the named
        commit. Documents that our extraction is robust to this case.
        """
        spec = _spec()
        grade = make_grader(spec)
        # add -A returns no output (nothing to stage); diff --cached still
        # shows the committed change.
        env = _grade_env(
            add_result=_ok_exec(stdout=""),
            diff_result=_ok_exec(
                stdout="diff --git a/src/foo.py b/src/foo.py\n@@ -1 +1 @@\n-x\n+y\n"
            ),
        )
        result = grade(env)
        assert result["patch_extracted"] is True
        assert "src/foo.py" in result["model_patch"]


# ---------------------------------------------------------------------------
# pull_base_image
# ---------------------------------------------------------------------------


class TestBuildTaskImage:
    """Validates that the SWE-Bench runner threads the testbed conda env into
    the composite image. Without this, ``docker exec`` resolves ``python`` /
    ``pytest`` to the base conda env (no project deps), which derailed early
    runs (mpmath ImportError, pytest-not-found, etc.).
    """

    def test_passes_swebench_conda_env_to_build_composite(self) -> None:
        spec = _spec(instance_id="django__django-12345")
        with patch.object(runner_mod, "pull_base_image"):
            with patch.object(
                runner_mod, "build_composite", return_value=spec.composite_image
            ) as bc:
                build_task_image(spec, cache=True)

        kwargs = bc.call_args.kwargs
        assert kwargs["base_image"] == spec.base_image
        assert kwargs["output_tag"] == spec.composite_image
        assert kwargs["cache"] is True
        # The testbed env block is wired in.
        assert kwargs["extra_env"] == SWEBENCH_COMPOSITE_ENV
        # And critical fields are present (regression guard against someone
        # dropping the PATH entry).
        assert "/opt/miniconda3/envs/testbed/bin" in kwargs["extra_env"]["PATH"]
        assert kwargs["extra_env"]["CONDA_DEFAULT_ENV"] == "testbed"

    def test_cache_false_propagates(self) -> None:
        spec = _spec()
        with patch.object(runner_mod, "pull_base_image"):
            with patch.object(
                runner_mod, "build_composite", return_value=spec.composite_image
            ) as bc:
                build_task_image(spec, cache=False)
        assert bc.call_args.kwargs["cache"] is False


class TestPullBaseImage:
    def test_skips_pull_when_image_exists(self) -> None:
        spec = _spec()
        with patch.object(runner_mod, "_image_exists", return_value=True):
            with patch.object(runner_mod.subprocess, "run") as run:
                pull_base_image(spec)
        run.assert_not_called()

    def test_pulls_when_image_missing(self) -> None:
        spec = _spec()
        ok = MagicMock(returncode=0, stdout="", stderr="")
        with patch.object(runner_mod, "_image_exists", return_value=False):
            with patch.object(runner_mod.subprocess, "run", return_value=ok) as run:
                pull_base_image(spec)
        run.assert_called_once()
        cmd = run.call_args.args[0]
        assert cmd[:2] == ["docker", "pull"]
        assert cmd[2] == spec.base_image

    def test_retries_once_on_failure(self) -> None:
        spec = _spec()
        fail = MagicMock(returncode=1, stdout="", stderr="network error")
        ok = MagicMock(returncode=0, stdout="", stderr="")
        with patch.object(runner_mod, "_image_exists", return_value=False):
            with patch.object(runner_mod.subprocess, "run", side_effect=[fail, ok]) as run:
                pull_base_image(spec, max_retries=1)
        # First attempt failed, second succeeded.
        assert run.call_count == 2

    def test_raises_after_persistent_failure(self) -> None:
        spec = _spec()
        fail = MagicMock(returncode=1, stdout="", stderr="auth required")
        with patch.object(runner_mod, "_image_exists", return_value=False):
            with patch.object(runner_mod.subprocess, "run", return_value=fail):
                with pytest.raises(RuntimeError, match="auth required"):
                    pull_base_image(spec, max_retries=1)


# ---------------------------------------------------------------------------
# run_instance
# ---------------------------------------------------------------------------


class TestRunInstance:
    def test_happy_path_emits_patch_via_pre_cleanup_result(self, tmp_path: Path) -> None:
        spec = _spec()
        rlm_instance = MagicMock()
        rlm_instance.completion.return_value = _stub_completion(
            response="done",
            pre_cleanup_result={
                "model_patch": "diff --git a/foo b/foo\n+y\n",
                "patch_extracted": True,
                "extraction_error": None,
            },
        )
        with patch.object(runner_mod, "build_task_image", return_value="rlm-swebench-x:latest"):
            with patch.object(runner_mod, "RLM", return_value=rlm_instance) as mock_rlm:
                result = run_instance(
                    spec,
                    backend="openai",
                    backend_kwargs={"model_name": "fake"},
                    max_iterations=10,
                    output_dir=tmp_path / "results",
                )
                rlm_kwargs = mock_rlm.call_args.kwargs
                completion_kwargs = rlm_instance.completion.call_args.kwargs

        assert result.instance_id == spec.instance_id
        assert result.patch_extracted is True
        assert result.model_patch.startswith("diff --git ")
        assert result.error is None
        assert rlm_kwargs["workspace_config"].docker.image == "rlm-swebench-x:latest"
        assert callable(completion_kwargs["pre_cleanup_callback"])

    def test_agent_exception_caught_and_recorded(self, tmp_path: Path) -> None:
        spec = _spec()
        rlm_instance = MagicMock()
        rlm_instance.completion.side_effect = RuntimeError("LM endpoint down")
        with patch.object(runner_mod, "build_task_image", return_value="t"):
            with patch.object(runner_mod, "RLM", return_value=rlm_instance):
                result = run_instance(
                    spec,
                    backend="openai",
                    backend_kwargs={"model_name": "fake"},
                    max_iterations=5,
                    output_dir=tmp_path / "results",
                )
        assert result.patch_extracted is False
        assert result.error is not None
        assert "LM endpoint down" in result.error

    def test_image_build_failure_recorded(self, tmp_path: Path) -> None:
        spec = _spec()
        with patch.object(runner_mod, "build_task_image", side_effect=RuntimeError("pull failed")):
            result = run_instance(
                spec,
                backend="openai",
                backend_kwargs={"model_name": "fake"},
                max_iterations=5,
                output_dir=tmp_path / "results",
            )
        assert result.error is not None
        assert "image build failed" in result.error
        assert "pull failed" in result.error

    def test_pre_cleanup_result_none_treated_as_no_patch(self, tmp_path: Path) -> None:
        spec = _spec()
        rlm_instance = MagicMock()
        rlm_instance.completion.return_value = _stub_completion(pre_cleanup_result=None)
        with patch.object(runner_mod, "build_task_image", return_value="t"):
            with patch.object(runner_mod, "RLM", return_value=rlm_instance):
                result = run_instance(
                    spec,
                    backend="openai",
                    backend_kwargs={"model_name": "fake"},
                    max_iterations=5,
                    output_dir=tmp_path / "results",
                )
        assert result.patch_extracted is False
        assert result.model_patch == ""
        assert result.error is None

    def test_passes_correct_timeouts_into_workspace_config(self, tmp_path: Path) -> None:
        spec = _spec()
        spec.agent_timeout_sec = 777.0
        rlm_instance = MagicMock()
        rlm_instance.completion.return_value = _stub_completion(
            pre_cleanup_result={
                "model_patch": "",
                "patch_extracted": False,
                "extraction_error": "empty patch",
            }
        )
        with patch.object(runner_mod, "build_task_image", return_value="t"):
            with patch.object(runner_mod, "RLM", return_value=rlm_instance) as mock_rlm:
                run_instance(
                    spec,
                    backend="openai",
                    backend_kwargs={"model_name": "fake"},
                    max_iterations=5,
                    output_dir=tmp_path / "results",
                )
                rlm_kwargs = mock_rlm.call_args.kwargs
        assert rlm_kwargs["workspace_config"].docker.exec_timeout_seconds == 777
        assert rlm_kwargs["workspace_config"].docker.cleanup_mode == "delete"


# ---------------------------------------------------------------------------
# Predictions / resume / atomic writes
# ---------------------------------------------------------------------------


class TestPredictionsAndResume:
    def test_atomic_write_json_uses_tmp_and_rename(self, tmp_path: Path) -> None:
        target = tmp_path / "out" / "prediction.json"
        _atomic_write_json(target, {"a": 1})
        assert target.is_file()
        # No leftover .tmp.
        assert not target.with_suffix(target.suffix + ".tmp").exists()
        assert json.loads(target.read_text())["a"] == 1

    def test_atomic_write_survives_crash_midwrite(self, tmp_path: Path) -> None:
        """Simulate a crash between writing the tmp and renaming. The target
        path must not exist (no half-written file) and the tmp file is left
        behind for inspection (or cleanup on the next successful write — we
        don't garbage-collect tmps eagerly).
        """
        target = tmp_path / "prediction.json"
        # Pre-write a previous good version.
        _atomic_write_json(target, {"v": 1})

        def boom(*args: object, **kwargs: object) -> None:  # pragma: no cover - mock
            raise RuntimeError("simulated crash before rename")

        with patch.object(runner_mod.os, "replace", side_effect=boom):
            with pytest.raises(RuntimeError, match="simulated crash"):
                _atomic_write_json(target, {"v": 2})

        # Target unchanged.
        assert json.loads(target.read_text())["v"] == 1

    def test_scan_completed_includes_existing_shards(self, tmp_path: Path) -> None:
        (tmp_path / "a__a-1").mkdir()
        (tmp_path / "b__b-2").mkdir()
        (tmp_path / "a__a-1" / "prediction.json").write_text(
            json.dumps({"instance_id": "a__a-1", "model_name_or_path": "m", "model_patch": "d"})
        )
        (tmp_path / "b__b-2" / "prediction.json").write_text(
            json.dumps({"instance_id": "b__b-2", "model_name_or_path": "m", "model_patch": ""})
        )
        completed = _scan_completed_instances(tmp_path)
        assert set(completed) == {"a__a-1", "b__b-2"}

    def test_scan_tolerates_malformed_shard(self, tmp_path: Path) -> None:
        (tmp_path / "broken").mkdir()
        (tmp_path / "broken" / "prediction.json").write_text("{not json")
        (tmp_path / "good").mkdir()
        (tmp_path / "good" / "prediction.json").write_text(
            json.dumps({"instance_id": "good", "model_name_or_path": "m", "model_patch": "x"})
        )
        completed = _scan_completed_instances(tmp_path)
        # The good shard is included; the broken one is logged + skipped.
        assert set(completed) == {"good"}

    def test_aggregate_predictions_writes_one_line_per_shard(self, tmp_path: Path) -> None:
        for iid in ("z__z-1", "a__a-1"):
            d = tmp_path / iid
            d.mkdir()
            _atomic_write_json(
                d / "prediction.json",
                {"instance_id": iid, "model_name_or_path": "m", "model_patch": "p"},
            )
        runner_mod._aggregate_predictions(tmp_path)
        path = tmp_path / "predictions.jsonl"
        lines = path.read_text().splitlines()
        assert len(lines) == 2
        # Sorted by instance_id.
        first = json.loads(lines[0])
        second = json.loads(lines[1])
        assert first["instance_id"] == "a__a-1"
        assert second["instance_id"] == "z__z-1"

    def test_aggregate_predictions_idempotent(self, tmp_path: Path) -> None:
        d = tmp_path / "x__x-1"
        d.mkdir()
        _atomic_write_json(
            d / "prediction.json",
            {"instance_id": "x__x-1", "model_name_or_path": "m", "model_patch": "p"},
        )
        runner_mod._aggregate_predictions(tmp_path)
        first = (tmp_path / "predictions.jsonl").read_text()
        runner_mod._aggregate_predictions(tmp_path)
        second = (tmp_path / "predictions.jsonl").read_text()
        assert first == second


# ---------------------------------------------------------------------------
# main() CLI plumbing
# ---------------------------------------------------------------------------


class TestMainCLI:
    def _patch_load_and_run(self, instances: list[InstanceSpec], *, output_dir: Path):
        """Patch the dataset loader and the worker so main() runs end-to-end
        without actually pulling docker images.
        """

        def fake_run_one(spec_payload: dict, runner_args: dict) -> str:
            iid = spec_payload["instance_id"]
            d = Path(runner_args["output_dir"]) / iid
            d.mkdir(parents=True, exist_ok=True)
            _atomic_write_json(
                d / "prediction.json",
                {
                    "instance_id": iid,
                    "model_name_or_path": runner_args["model_name_or_path"],
                    "model_patch": "diff --git a/x b/x\n+y\n",
                },
            )
            _atomic_write_json(
                d / "result.json",
                {
                    "instance_id": iid,
                    "model_patch": "diff --git a/x b/x\n+y\n",
                    "patch_extracted": True,
                    "extraction_error": None,
                    "agent_response": "done",
                    "agent_turns": 1,
                    "wall_clock_s": 0.1,
                    "error": None,
                },
            )
            return iid

        return patch.multiple(
            runner_mod,
            _load_instances_from_hf=MagicMock(return_value=instances),
            _run_one=MagicMock(side_effect=fake_run_one),
        )

    def test_sequential_run_writes_shards_and_aggregates(self, tmp_path: Path) -> None:
        instances = [
            _spec(instance_id="a__a-1"),
            _spec(instance_id="b__b-2"),
        ]
        with self._patch_load_and_run(instances, output_dir=tmp_path):
            rc = runner_mod.main(
                [
                    "--output-dir",
                    str(tmp_path),
                    "--num-workers",
                    "1",
                    "--instance-ids",
                    "a__a-1",
                    "b__b-2",
                ]
            )
        assert rc == 0
        # Per-instance shards exist.
        assert (tmp_path / "a__a-1" / "prediction.json").is_file()
        assert (tmp_path / "b__b-2" / "prediction.json").is_file()
        # Aggregated predictions.jsonl.
        lines = (tmp_path / "predictions.jsonl").read_text().splitlines()
        assert len(lines) == 2
        # Aggregated summary.jsonl.
        assert (tmp_path / "summary.jsonl").is_file()

    def test_resume_skips_existing_shards(self, tmp_path: Path) -> None:
        # Pre-seed a completed shard for one of the two instances.
        d = tmp_path / "a__a-1"
        d.mkdir()
        _atomic_write_json(
            d / "prediction.json",
            {"instance_id": "a__a-1", "model_name_or_path": "prior", "model_patch": "old"},
        )

        def fake_run_one(payload: dict, runner_args: dict) -> str:
            iid = payload["instance_id"]
            shard_dir = Path(runner_args["output_dir"]) / iid
            shard_dir.mkdir(exist_ok=True, parents=True)
            _atomic_write_json(
                shard_dir / "prediction.json",
                {
                    "instance_id": iid,
                    "model_name_or_path": runner_args["model_name_or_path"],
                    "model_patch": "new",
                },
            )
            _atomic_write_json(
                shard_dir / "result.json",
                {
                    "instance_id": iid,
                    "model_patch": "new",
                    "patch_extracted": True,
                    "extraction_error": None,
                    "agent_response": "",
                    "agent_turns": 1,
                    "wall_clock_s": 0.0,
                    "error": None,
                },
            )
            return iid

        instances = [_spec(instance_id="a__a-1"), _spec(instance_id="b__b-2")]
        with patch.object(runner_mod, "_load_instances_from_hf", return_value=instances):
            with patch.object(runner_mod, "_run_one", side_effect=fake_run_one) as mock_run:
                rc = runner_mod.main(
                    [
                        "--output-dir",
                        str(tmp_path),
                        "--num-workers",
                        "1",
                        "--instance-ids",
                        "a__a-1",
                        "b__b-2",
                    ]
                )
        assert rc == 0
        # Only b__b-2 was re-run; a__a-1's existing shard was kept.
        dispatched = [c.args[0]["instance_id"] for c in mock_run.call_args_list]
        assert dispatched == ["b__b-2"]
        # Aggregated predictions.jsonl includes both: pre-existing a and new b.
        lines = (tmp_path / "predictions.jsonl").read_text().splitlines()
        ids = sorted(json.loads(line)["instance_id"] for line in lines)
        assert ids == ["a__a-1", "b__b-2"]
        # The original a__a-1 shard is preserved (not overwritten).
        a_shard = json.loads((tmp_path / "a__a-1" / "prediction.json").read_text())
        assert a_shard["model_name_or_path"] == "prior"
        assert a_shard["model_patch"] == "old"

    def test_no_instances_after_filter_exits_nonzero(self, tmp_path: Path) -> None:
        with patch.object(runner_mod, "_load_instances_from_hf", return_value=[]):
            rc = runner_mod.main(
                ["--output-dir", str(tmp_path), "--num-workers", "1", "--instance-ids", "ghost"]
            )
        assert rc == 1

    def test_all_instances_already_completed_short_circuits(self, tmp_path: Path) -> None:
        # Pre-seed completed shards for both instances.
        for iid in ("a__a-1", "b__b-2"):
            d = tmp_path / iid
            d.mkdir()
            _atomic_write_json(
                d / "prediction.json",
                {"instance_id": iid, "model_name_or_path": "m", "model_patch": "p"},
            )
            _atomic_write_json(
                d / "result.json",
                {
                    "instance_id": iid,
                    "model_patch": "p",
                    "patch_extracted": True,
                    "extraction_error": None,
                    "agent_response": "",
                    "agent_turns": 1,
                    "wall_clock_s": 0.0,
                    "error": None,
                },
            )
        instances = [_spec(instance_id="a__a-1"), _spec(instance_id="b__b-2")]
        with patch.object(runner_mod, "_load_instances_from_hf", return_value=instances):
            with patch.object(runner_mod, "_run_one") as mock_run:
                rc = runner_mod.main(
                    [
                        "--output-dir",
                        str(tmp_path),
                        "--num-workers",
                        "1",
                        "--instance-ids",
                        "a__a-1",
                        "b__b-2",
                    ]
                )
        assert rc == 0
        mock_run.assert_not_called()
        # Aggregations still produced.
        assert (tmp_path / "predictions.jsonl").is_file()
        assert (tmp_path / "summary.jsonl").is_file()


# ---------------------------------------------------------------------------
# _image_exists helper
# ---------------------------------------------------------------------------


class TestImageExistsHelper:
    def test_returns_true_when_inspect_succeeds(self) -> None:
        ok = MagicMock(returncode=0, stdout="", stderr="")
        with patch.object(runner_mod.subprocess, "run", return_value=ok):
            assert runner_mod._image_exists("foo:latest") is True

    def test_returns_false_when_inspect_fails(self) -> None:
        fail = MagicMock(returncode=1, stdout="", stderr="No such image")
        with patch.object(runner_mod.subprocess, "run", return_value=fail):
            assert runner_mod._image_exists("nope:latest") is False
