"""Unit tests for the workspace-substrate RLM loop (Phase 4).

These tests do not require Docker. They mock the LM handler and the
workspace env so the parse-and-retry inner loop, action dispatch, and
prompt construction can be exercised in isolation.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock

import pytest

from rlm.core.config import CompactionConfig, ParseConfig, WorkspaceConfig
from rlm.core.rlm import RLM
from rlm.core.types import (
    RLMChatCompletion,
    UsageSummary,
    WorkspaceAction,
    WorkspaceIteration,
    WorkspaceObservation,
    WorkspaceSnapshot,
)
from rlm.utils.exceptions import ActionParseError
from rlm.utils.prompts import (
    build_workspace_initial_user_prompt,
    build_workspace_system_prompt,
    format_workspace_iteration,
)

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


class TestWorkspaceSystemPrompt:
    def test_advertises_rlm_query_below_max_depth(self):
        p = build_workspace_system_prompt(depth=0, max_depth=2)
        tools_section = p.split("# Available tools")[1].split("# Hard rules")[0]
        assert "rlm_query" in tools_section

    def test_omits_rlm_query_at_max_depth(self):
        p = build_workspace_system_prompt(depth=2, max_depth=2)
        tools_section = p.split("# Available tools")[1].split("# Hard rules")[0]
        # The literal `rlm_query` tool entry must not appear; the python tool
        # description mentioning the in-container helper string is fine.
        assert "- ``rlm_query``" not in tools_section
        # And the depth_rule mentions the unavailability explicitly.
        assert "maximum recursion depth" in p.lower()

    def test_custom_system_prompt_passes_through(self):
        out = build_workspace_system_prompt(depth=0, max_depth=2, custom_system_prompt="HELLO")
        assert out == "HELLO"

    def test_initial_user_prompt_with_root_pointer(self):
        msg = build_workspace_initial_user_prompt(root_prompt="solve x")
        assert "_rlm_query_0.txt" in msg
        assert "solve x" in msg

    def test_system_prompt_describes_tool_intent(self):
        p = build_workspace_system_prompt(depth=0, max_depth=1)
        assert "# Tool intent" in p
        assert "Use ``write_file``, ``append_file``, and ``edit``" in p
        assert "run_python_command" in p
        assert "run_shell_command" in p
        assert "Every assistant turn must make at least one native tool call" in p
        assert '<action tool="final"' not in p
        assert "<answer>" not in p
        assert "<artifact" not in p
        assert "<action" not in p

    def test_xml_system_prompt_is_explicit_compatibility_path(self):
        p = build_workspace_system_prompt(depth=0, max_depth=1, action_format="xml")
        assert 'emit one or more ``<action tool="...">...</action>`` elements' in p
        # The substrate no longer asks for `<note>` blocks; turn-over-turn
        # memory comes from the workspace files and the periodic
        # substrate-level summary, not from inline notes.
        assert "<note>" not in p

    def test_system_prompt_advertises_substrate_compaction(self):
        """Both system prompts must tell the model that the visible history
        will be periodically compressed and that files are authoritative."""
        for fmt in ("native", "xml"):
            p = build_workspace_system_prompt(depth=0, max_depth=1, action_format=fmt)
            assert "summarizes the trajectory" in p
            assert "authoritative" in p

    def test_native_action_format_defaults_thinking_off_for_vllm(self):
        cfg = WorkspaceConfig()
        rlm = RLM(
            backend="vllm",
            backend_kwargs={"model_name": "fake", "base_url": "http://localhost:8000/v1"},
            workspace_config=cfg,
        )
        assert rlm._resolved_backend_kwargs()["enable_thinking"] is False

    def test_public_completion_rejects_deprecated_xml_format(self):
        cfg = WorkspaceConfig(parse=ParseConfig(action_format="xml"))
        rlm = RLM(
            backend="openai",
            backend_kwargs={"model_name": "fake"},
            workspace_config=cfg,
        )
        with pytest.raises(ValueError, match="Legacy XML tool calling is deprecated"):
            rlm.completion("x")


# ---------------------------------------------------------------------------
# Iteration formatter
# ---------------------------------------------------------------------------


class TestFormatWorkspaceIteration:
    def test_emits_assistant_then_user(self):
        action = WorkspaceAction(tool="read_file", args={"path": "a"}, body=None, raw="")
        obs = WorkspaceObservation(tool="read_file", stdout="hello", artifacts=["a"])
        snap = WorkspaceSnapshot(
            turn=1, commit_sha="abcdef1234", changed_files=["a"], workspace_root="/tmp"
        )
        it = WorkspaceIteration(
            iteration=1,
            timestamp="2026-01-01T00:00:00",
            prompt=[],
            response="prose with action",
            reasoning=None,
            actions=[action],
            observations=[obs],
            snapshot=snap,
        )
        msgs = format_workspace_iteration(it, action_format="xml")
        assert [m["role"] for m in msgs] == ["assistant", "user"]
        assert '<action_replay action_id="t1.a1" tool="read_file"' in msgs[0]["content"]
        assert "t1.a1" in msgs[1]["content"]
        assert "hello" in msgs[1]["content"]
        assert "snapshot" in msgs[1]["content"]
        assert "abcdef1" in msgs[1]["content"]

    def test_file_tool_body_is_replayed_verbatim(self):
        """Pre-compress, durable-edit action bodies stay full-fidelity in the
        prompt. The substrate-level CompactionConfig handles trimming once
        the cumulative prompt crosses the token threshold."""
        action = WorkspaceAction(
            tool="write_file",
            args={"path": "draft.md"},
            body="secret draft body\nsecond line",
            raw='<action tool="write_file" path="draft.md">secret draft body</action>',
        )
        obs = WorkspaceObservation(
            tool="write_file",
            stdout="Wrote 29 chars to draft.md",
            data={"path": "draft.md", "bytes": 29},
            artifacts=["draft.md"],
        )
        it = WorkspaceIteration(
            iteration=2,
            timestamp="2026-01-01T00:00:00",
            prompt=[],
            response=action.raw,
            reasoning=None,
            actions=[action],
            observations=[obs],
        )
        msgs = format_workspace_iteration(it, action_format="xml")
        assert "secret draft body" in msgs[0]["content"]
        assert "second line" in msgs[0]["content"]
        assert "Wrote 29 chars to draft.md" in msgs[1]["content"]

    def test_observations_replayed_full_fidelity_for_all_ages(self):
        """No age-based receipt compaction anymore — every completed turn's
        observations are replayed verbatim until substrate-level compaction
        fires."""
        from rlm.utils.prompts import format_workspace_history

        old = WorkspaceIteration(
            iteration=1,
            timestamp="2026-01-01T00:00:00",
            prompt=[],
            response="",
            reasoning=None,
            actions=[WorkspaceAction(tool="read_file", args={"path": "a.md"}, body=None, raw="")],
            observations=[WorkspaceObservation(tool="read_file", stdout="full file contents")],
        )
        new = WorkspaceIteration(
            iteration=2,
            timestamp="2026-01-01T00:00:01",
            prompt=[],
            response="",
            reasoning=None,
            actions=[WorkspaceAction(tool="list_directory", args={"path": "."}, body=None, raw="")],
            observations=[WorkspaceObservation(tool="list_directory", stdout="Directory: .")],
        )
        msgs = format_workspace_history([old, new], action_format="xml")
        # Both old and new stdout are present in the prompt — no receipts.
        joined = "\n".join(m["content"] for m in msgs)
        assert "full file contents" in joined
        assert "Directory: ." in joined
        assert "omitted from replay" not in joined

    def test_native_replay_does_not_emit_xml_tags(self):
        action = WorkspaceAction(
            tool="run_python_command",
            args={"code": "print('x')\n", "timeout": 10},
            body="print('x')\n",
            raw='{"name": "run_python_command"}',
            call_id="call_1",
        )
        obs = WorkspaceObservation(tool="run_python_command", stdout="x\n")
        it = WorkspaceIteration(
            iteration=1,
            timestamp="2026-01-01T00:00:00",
            prompt=[],
            response="",
            reasoning=None,
            actions=[action],
            observations=[obs],
        )
        msgs = format_workspace_iteration(it)
        text = "\n".join(msg["content"] for msg in msgs)
        assert "TOOL_CALL t1.a1 tool=run_python_command" in text
        assert "OBSERVATION t1.a1 tool=run_python_command" in text
        assert "<action" not in text
        assert "<observation" not in text


# ---------------------------------------------------------------------------
# Parse-and-retry inner loop
# ---------------------------------------------------------------------------


def _mock_lm_handler(responses: list[tuple[str, str | None]]) -> MagicMock:
    """Build a MagicMock LMHandler that returns scripted (response, reasoning)."""
    handler = MagicMock()
    handler.completion_with_reasoning = MagicMock(side_effect=list(responses))
    return handler


class TestParseAndRetry:
    def _rlm(self, retries: int = 3) -> RLM:
        cfg = WorkspaceConfig(parse=ParseConfig(action_format="xml"))
        cfg.parse.max_action_parse_retries = retries
        return RLM(
            backend="openai",
            backend_kwargs={"model_name": "fake"},
            workspace_config=cfg,
        )

    def test_first_attempt_succeeds(self):
        rlm = self._rlm()
        good = '<action tool="list_directory" />'
        handler = _mock_lm_handler([(good, None)])
        response, reasoning, actions, attempts = rlm._call_lm_with_parse_retry(
            lm_handler=handler, messages=[{"role": "user", "content": "x"}]
        )
        assert response == good
        assert reasoning is None
        assert len(actions) == 1
        assert attempts == []
        assert handler.completion_with_reasoning.call_count == 1

    def test_retry_succeeds_after_one_malformed(self):
        rlm = self._rlm(retries=2)
        bad = "no actions here"
        good = '<action tool="list_directory" />'
        handler = _mock_lm_handler([(bad, None), (good, "reasoned")])
        response, reasoning, actions, attempts = rlm._call_lm_with_parse_retry(
            lm_handler=handler, messages=[{"role": "user", "content": "x"}]
        )
        assert response == good
        assert reasoning == "reasoned"
        assert len(actions) == 1
        assert len(attempts) == 1
        assert "No <action>" in attempts[0]["error"]

    def test_raises_after_retries_exhausted(self):
        rlm = self._rlm(retries=2)
        bad = "no actions here"
        handler = _mock_lm_handler([(bad, None), (bad, None), (bad, None)])
        with pytest.raises(ActionParseError) as ei:
            rlm._call_lm_with_parse_retry(
                lm_handler=handler, messages=[{"role": "user", "content": "x"}]
            )
        # Initial try + 2 retries = 3 LM calls.
        assert handler.completion_with_reasoning.call_count == 3
        assert "after 2 retries" in str(ei.value)

    def test_retry_messages_grow_with_feedback(self):
        rlm = self._rlm(retries=1)
        bad = "no actions here"
        good = '<action tool="list_directory" />'
        seen: list[list[dict[str, Any]]] = []

        def fake(messages):
            # Capture the messages each call sees, return scripted responses.
            seen.append(list(messages))
            return (bad, None) if len(seen) == 1 else (good, None)

        handler = MagicMock()
        handler.completion_with_reasoning = MagicMock(side_effect=fake)
        rlm._call_lm_with_parse_retry(
            lm_handler=handler, messages=[{"role": "user", "content": "x"}]
        )
        assert len(seen) == 2
        # First call: only the original message.
        assert len(seen[0]) == 1
        # Second call: original + assistant(bad) + user(feedback).
        assert len(seen[1]) == 3
        assert seen[1][1]["role"] == "assistant"
        assert seen[1][2]["role"] == "user"
        assert "malformed" in seen[1][2]["content"]


# ---------------------------------------------------------------------------
# Action dispatch (read-only vs mutating semantics)
# ---------------------------------------------------------------------------


class TestActionDispatch:
    def _rlm(self) -> RLM:
        return RLM(
            backend="openai",
            backend_kwargs={"model_name": "fake"},
            workspace_config=WorkspaceConfig(),
        )

    def _mock_env_returning(self, observations_by_tool: dict[str, WorkspaceObservation]):
        """Build an env whose run_action looks up by tool name."""
        env = MagicMock()
        env.run_action = MagicMock(side_effect=lambda action: observations_by_tool[action.tool])
        return env

    def test_read_only_failure_does_not_halt(self):
        rlm = self._rlm()
        env = self._mock_env_returning(
            {
                "read_file": WorkspaceObservation(tool="read_file", error="not found"),
                "list_directory": WorkspaceObservation(tool="list_directory", stdout="ok"),
            }
        )
        actions = [
            WorkspaceAction(tool="read_file", args={"path": "x"}, body=None, raw=""),
            WorkspaceAction(tool="list_directory", args={}, body=None, raw=""),
        ]
        observations = rlm._dispatch_actions(env=env, actions=actions)
        assert len(observations) == 2
        assert observations[0].error == "not found"
        assert observations[1].error is None
        # Both actions actually ran.
        assert env.run_action.call_count == 2

    def test_mutating_failure_halts_subsequent_mutating_but_not_read_only(self):
        rlm = self._rlm()
        env = self._mock_env_returning(
            {
                "write_file": WorkspaceObservation(tool="write_file", error="disk full"),
                "list_directory": WorkspaceObservation(tool="list_directory", stdout="ok"),
                "append_file": WorkspaceObservation(tool="append_file", stdout="ok"),
            }
        )
        actions = [
            WorkspaceAction(tool="write_file", args={"path": "a"}, body="x", raw=""),
            WorkspaceAction(tool="list_directory", args={}, body=None, raw=""),
            WorkspaceAction(tool="append_file", args={"path": "b"}, body="y", raw=""),
        ]
        observations = rlm._dispatch_actions(env=env, actions=actions)
        assert len(observations) == 3
        # write_file errored, list_directory still ran, append_file was skipped.
        assert observations[0].error == "disk full"
        assert observations[1].tool == "list_directory"
        assert observations[1].error is None
        assert observations[2].tool == "append_file"
        assert observations[2].error and "Skipped" in observations[2].error
        # Env saw write_file + list_directory but NOT append_file.
        assert env.run_action.call_count == 2

    def test_final_action_breaks_dispatch(self):
        rlm = self._rlm()
        env = self._mock_env_returning(
            {
                "final": WorkspaceObservation(tool="final", final_answer="done", stdout="done"),
                "list_directory": WorkspaceObservation(tool="list_directory"),
            }
        )
        actions = [
            WorkspaceAction(tool="final", args={}, body="<answer>done</answer>", raw=""),
            WorkspaceAction(tool="list_directory", args={}, body=None, raw=""),
        ]
        observations = rlm._dispatch_actions(env=env, actions=actions)
        assert len(observations) == 1  # broke after final
        assert observations[0].final_answer == "done"

    def test_final_is_skipped_after_mutating_failure_in_same_batch(self):
        """A batch like [python(error), final] must NOT commit final — the
        model would claim success on a broken workspace. Observed in the
        2026-05-11 Qwen3-8B 3a run: python wrapped in <script> tags raised
        SyntaxError, final was batched after it, and the loop terminated
        with the model's (wrong) claim 'Generated first 100 primes.'"""
        rlm = self._rlm()
        env = self._mock_env_returning(
            {
                "python": WorkspaceObservation(tool="python", error="SyntaxError: invalid syntax"),
                "final": WorkspaceObservation(tool="final", final_answer="done", stdout="done"),
            }
        )
        actions = [
            WorkspaceAction(tool="python", args={}, body="<script>bad</script>", raw=""),
            WorkspaceAction(tool="final", args={}, body="<answer>done</answer>", raw=""),
        ]
        observations = rlm._dispatch_actions(env=env, actions=actions)
        # Both observations recorded; python errored, final was skipped (not
        # executed by env), loop did not terminate via final_answer.
        assert len(observations) == 2
        assert observations[0].tool == "python"
        assert observations[0].error == "SyntaxError: invalid syntax"
        assert observations[1].tool == "final"
        assert observations[1].error and "Skipped" in observations[1].error
        assert observations[1].final_answer is None
        # Env only saw python; final was gated out before reaching env.run_action.
        assert env.run_action.call_count == 1

    def test_read_only_still_runs_when_final_is_skipped_after_halt(self):
        """When the halt fires, read-only tools should still execute so the
        model can inspect state before retrying. Only mutating + terminal
        tools are gated."""
        rlm = self._rlm()
        env = self._mock_env_returning(
            {
                "write_file": WorkspaceObservation(tool="write_file", error="disk full"),
                "read_file": WorkspaceObservation(tool="read_file", stdout="contents"),
                "final": WorkspaceObservation(tool="final", final_answer="done", stdout="done"),
            }
        )
        actions = [
            WorkspaceAction(tool="write_file", args={"path": "a"}, body="x", raw=""),
            WorkspaceAction(tool="read_file", args={"path": "a"}, body=None, raw=""),
            WorkspaceAction(tool="final", args={}, body="<answer>done</answer>", raw=""),
        ]
        observations = rlm._dispatch_actions(env=env, actions=actions)
        assert len(observations) == 3
        assert observations[0].error == "disk full"
        assert observations[1].tool == "read_file"
        assert observations[1].error is None  # read-only ran
        assert observations[2].tool == "final"
        assert observations[2].error and "Skipped" in observations[2].error
        assert observations[2].final_answer is None
        # Env saw write_file + read_file; final gated out.
        assert env.run_action.call_count == 2


# ---------------------------------------------------------------------------
# Failed iteration is logged when parse retries are exhausted
# ---------------------------------------------------------------------------


class TestFailedIterationLogging:
    """When parse retries exhaust, the partial iteration must reach the log."""

    def _rlm(self, retries: int = 2) -> RLM:
        cfg = WorkspaceConfig(parse=ParseConfig(action_format="xml"))
        cfg.parse.max_action_parse_retries = retries
        return RLM(
            backend="openai",
            backend_kwargs={"model_name": "fake"},
            workspace_config=cfg,
        )

    def test_completion_turn_attaches_partial_iteration(self):
        """`_completion_turn` builds a WorkspaceIteration on parse-fail and
        attaches it to the raised ActionParseError."""
        rlm = self._rlm(retries=2)
        bad = "no actions here"
        handler = _mock_lm_handler([(bad, None), (bad, None), (bad, "thought")])
        env = MagicMock()  # never reached — exception fires before dispatch

        with pytest.raises(ActionParseError) as ei:
            rlm._completion_turn(
                iteration_idx=7,
                message_history=[{"role": "user", "content": "x"}],
                lm_handler=handler,
                env=env,
            )

        partial = getattr(ei.value, "iteration", None)
        assert isinstance(partial, WorkspaceIteration)
        assert partial.iteration == 7
        assert partial.error is not None
        assert "after 2 retries" in partial.error
        # Initial try + 2 retries = 3 attempts, all logged.
        assert len(partial.parse_attempts) == 3
        assert partial.actions == []
        assert partial.observations == []
        # Last raw response + reasoning carried through.
        assert partial.response == bad
        assert partial.reasoning == "thought"
        # Env never dispatched.
        env.run_action.assert_not_called()

    def test_run_loop_logs_partial_then_reraises(self):
        """`_run_loop` catches the exception, logs the partial iteration via
        `self.logger.log_iteration`, then re-raises."""
        from rlm.logger import RLMLogger

        cfg = WorkspaceConfig(parse=ParseConfig(action_format="xml"))
        cfg.parse.max_action_parse_retries = 1
        logger = RLMLogger(log_dir=None)
        rlm = RLM(
            backend="openai",
            backend_kwargs={"model_name": "fake"},
            workspace_config=cfg,
            logger=logger,
        )

        bad = "no actions here"
        handler = _mock_lm_handler([(bad, None), (bad, None)])

        # Stub env: only attribute access during _run_loop pre-amble.
        env = MagicMock()
        env.current_turn = 0

        with pytest.raises(ActionParseError):
            rlm._run_loop(prompt="x", root_prompt=None, lm_handler=handler, env=env)

        traj = logger.get_trajectory()
        assert traj is not None
        assert len(traj["iterations"]) == 1
        logged = traj["iterations"][0]
        assert logged["error"] is not None
        assert len(logged["parse_attempts"]) == 2
        assert logged["actions"] == []


# ---------------------------------------------------------------------------
# pre_cleanup_callback wiring
# ---------------------------------------------------------------------------


class TestPreCleanupCallback:
    """The callback hook runs after the agent loop, before env.cleanup().

    These tests exercise only the wiring in ``RLM.completion`` itself; the
    underlying container start/stop is mocked out so the test stays in-process.
    """

    def _stub_completion(self) -> RLMChatCompletion:
        return RLMChatCompletion(
            root_model="fake",
            prompt="x",
            response="done",
            usage_summary=UsageSummary(model_usage_summaries={}),
            execution_time=0.0,
        )

    def _rlm_with_stubbed_context(
        self,
        env: Any,
        cleanup_calls: list[str],
    ) -> RLM:
        """Build an RLM whose ``_spawn_completion_context`` yields ``env`` and
        whose ``_run_loop`` returns a stub completion. Cleanup is recorded
        into ``cleanup_calls``.
        """
        rlm = RLM(backend="openai", backend_kwargs={"model_name": "fake"})

        @contextmanager
        def fake_context(prompt):  # noqa: ARG001
            try:
                yield (MagicMock(), env)
            finally:
                cleanup_calls.append("cleanup")

        rlm._spawn_completion_context = fake_context  # type: ignore[assignment]
        rlm._wire_recursion = MagicMock()  # type: ignore[assignment]
        rlm._run_loop = MagicMock(return_value=self._stub_completion())  # type: ignore[assignment]
        return rlm

    def test_callback_fires_with_env_and_attaches_result(self) -> None:
        env = MagicMock(name="env")
        cleanup_calls: list[str] = []
        rlm = self._rlm_with_stubbed_context(env, cleanup_calls)
        seen_envs: list[Any] = []

        def grade(e):
            seen_envs.append(e)
            return {"exit_code": 0, "passed": True}

        result = rlm.completion("solve me", pre_cleanup_callback=grade)

        assert seen_envs == [env]
        assert result.pre_cleanup_result == {"exit_code": 0, "passed": True}
        # Cleanup must have run after callback returned.
        assert cleanup_calls == ["cleanup"]

    def test_callback_runs_before_cleanup(self) -> None:
        """Order: callback first, then cleanup."""
        env = MagicMock(name="env")
        events: list[str] = []
        rlm = RLM(backend="openai", backend_kwargs={"model_name": "fake"})

        @contextmanager
        def fake_context(prompt):  # noqa: ARG001
            try:
                yield (MagicMock(), env)
            finally:
                events.append("cleanup")

        rlm._spawn_completion_context = fake_context  # type: ignore[assignment]
        rlm._wire_recursion = MagicMock()  # type: ignore[assignment]
        rlm._run_loop = MagicMock(return_value=self._stub_completion())  # type: ignore[assignment]

        def grade(_env):
            events.append("callback")
            return None

        rlm.completion("x", pre_cleanup_callback=grade)
        assert events == ["callback", "cleanup"]

    def test_no_callback_means_no_field_set(self) -> None:
        env = MagicMock(name="env")
        cleanup_calls: list[str] = []
        rlm = self._rlm_with_stubbed_context(env, cleanup_calls)
        result = rlm.completion("x")
        assert result.pre_cleanup_result is None
        assert cleanup_calls == ["cleanup"]

    def test_callback_exception_still_runs_cleanup(self) -> None:
        env = MagicMock(name="env")
        events: list[str] = []
        rlm = RLM(backend="openai", backend_kwargs={"model_name": "fake"})

        @contextmanager
        def fake_context(prompt):  # noqa: ARG001
            try:
                yield (MagicMock(), env)
            finally:
                events.append("cleanup")

        rlm._spawn_completion_context = fake_context  # type: ignore[assignment]
        rlm._wire_recursion = MagicMock()  # type: ignore[assignment]
        rlm._run_loop = MagicMock(return_value=self._stub_completion())  # type: ignore[assignment]

        def grade(_env):
            events.append("callback")
            raise RuntimeError("grader blew up")

        with pytest.raises(RuntimeError, match="grader blew up"):
            rlm.completion("x", pre_cleanup_callback=grade)

        # Cleanup must still have happened.
        assert events == ["callback", "cleanup"]

    def test_callback_return_serialized_in_to_dict(self) -> None:
        """pre_cleanup_result is round-tripped through to_dict (used by logger)."""
        env = MagicMock(name="env")
        cleanup_calls: list[str] = []
        rlm = self._rlm_with_stubbed_context(env, cleanup_calls)
        result = rlm.completion(
            "x",
            pre_cleanup_callback=lambda _e: {"exit_code": 0, "stdout": "ok"},
        )
        d = result.to_dict()
        assert d["pre_cleanup_result"] == {"exit_code": 0, "stdout": "ok"}

    def test_callback_fires_on_max_iterations_exhaustion(self) -> None:
        """When the loop exits via the default-answer fallback (no ``final``
        action emitted), the callback should still fire — the contract is
        that any *return* from ``_run_loop`` triggers the callback.
        """
        env = MagicMock(name="env")
        events: list[str] = []
        rlm = RLM(backend="openai", backend_kwargs={"model_name": "fake"})

        @contextmanager
        def fake_context(prompt):  # noqa: ARG001
            try:
                yield (MagicMock(), env)
            finally:
                events.append("cleanup")

        # _run_loop returns a stub completion that simulates the max-iter
        # fallback path (response carries the LM-generated default answer
        # rather than a final-action answer; from the callback's POV the
        # two are indistinguishable, which is the point).
        fallback = RLMChatCompletion(
            root_model="fake",
            prompt="x",
            response="(default answer after max_iterations)",
            usage_summary=UsageSummary(model_usage_summaries={}),
            execution_time=0.0,
        )
        rlm._spawn_completion_context = fake_context  # type: ignore[assignment]
        rlm._wire_recursion = MagicMock()  # type: ignore[assignment]
        rlm._run_loop = MagicMock(return_value=fallback)  # type: ignore[assignment]

        def grade(_env):
            events.append("callback")
            return "graded"

        result = rlm.completion("x", pre_cleanup_callback=grade)

        assert events == ["callback", "cleanup"]
        assert result.pre_cleanup_result == "graded"
        assert result.response.startswith("(default answer")

    def test_callback_does_not_fire_when_loop_raises(self) -> None:
        """If ``_run_loop`` raises (parse-retry exhaustion, cancellation,
        budget/timeout/error-threshold/token-limit), the callback must NOT
        fire — we don't grade a crashed agent. Cleanup still runs.
        """
        env = MagicMock(name="env")
        events: list[str] = []
        rlm = RLM(backend="openai", backend_kwargs={"model_name": "fake"})

        @contextmanager
        def fake_context(prompt):  # noqa: ARG001
            try:
                yield (MagicMock(), env)
            finally:
                events.append("cleanup")

        class LoopBlewUp(RuntimeError):
            pass

        rlm._spawn_completion_context = fake_context  # type: ignore[assignment]
        rlm._wire_recursion = MagicMock()  # type: ignore[assignment]
        rlm._run_loop = MagicMock(side_effect=LoopBlewUp("agent crashed"))  # type: ignore[assignment]

        def grade(_env):
            events.append("callback")  # must not get here
            return "should-not-be-attached"

        with pytest.raises(LoopBlewUp, match="agent crashed"):
            rlm.completion("x", pre_cleanup_callback=grade)

        # Callback did NOT fire, cleanup DID.
        assert events == ["cleanup"]

    def test_callback_not_called_for_recursion_children(self) -> None:
        """Children spawned via ``RecursionHandler`` go through ``_run_loop``
        directly, not the public ``completion()`` entry, so the parent's
        callback wiring is unreachable from a child run. This is a
        structural property of how recursion is plumbed; we capture it as
        a regression guard.
        """
        # Confirm the public completion() entry is the only place the
        # callback is invoked, by introspecting the source. (We can't easily
        # spin up a real recursion subprocess in a unit test without
        # Docker.)
        import inspect

        from rlm.core import rlm as rlm_module

        src = inspect.getsource(rlm_module.RLM)
        # Exactly one invocation of `pre_cleanup_callback` (in `completion`).
        assert src.count("pre_cleanup_callback(env)") == 1


# ---------------------------------------------------------------------------
# Substrate-level compaction
# ---------------------------------------------------------------------------


class TestCompaction:
    """``_should_compact`` and ``_compact_history`` collapse the visible
    trajectory into ``[summary, continue]`` once the rendered prompt crosses
    ``CompactionConfig.threshold_tokens``.

    These tests mock the LM handler and provenance so the compaction path
    can run without Docker or a real client.
    """

    def _mock_handler_with_model(self, model_name: str = "fake-model"):
        handler = MagicMock()
        handler.get_client = MagicMock(return_value=MagicMock(model_name=model_name))
        return handler

    def test_should_compact_below_threshold_is_false(self):
        cfg = WorkspaceConfig(compaction=CompactionConfig(threshold_tokens=10_000))
        rlm = RLM(backend="openai", backend_kwargs={"model_name": "fake"}, workspace_config=cfg)
        handler = self._mock_handler_with_model()
        # A tiny message history won't cross 10K tokens.
        messages = [{"role": "user", "content": "hi"}]
        assert rlm._should_compact(messages, handler) is False

    def test_should_compact_above_threshold_is_true(self):
        cfg = WorkspaceConfig(compaction=CompactionConfig(threshold_tokens=100))
        rlm = RLM(backend="openai", backend_kwargs={"model_name": "fake"}, workspace_config=cfg)
        handler = self._mock_handler_with_model()
        # ~10000 chars ≈ 2500 tokens at the 4-chars-per-token fallback.
        messages = [{"role": "user", "content": "x" * 10_000}]
        assert rlm._should_compact(messages, handler) is True

    def test_should_compact_disabled_threshold_zero(self):
        cfg = WorkspaceConfig(compaction=CompactionConfig(threshold_tokens=0))
        rlm = RLM(backend="openai", backend_kwargs={"model_name": "fake"}, workspace_config=cfg)
        handler = self._mock_handler_with_model()
        messages = [{"role": "user", "content": "x" * 1_000_000}]
        assert rlm._should_compact(messages, handler) is False

    def test_compact_history_resets_iterations_and_returns_summary_prefix(self):
        from rlm.logger import RLMLogger

        cfg = WorkspaceConfig(
            compaction=CompactionConfig(threshold_tokens=100, tail_turns_preserved=0)
        )
        logger = RLMLogger(log_dir=None)
        # Seed the logger with metadata so log_compaction has somewhere to land.
        rlm = RLM(
            backend="openai",
            backend_kwargs={"model_name": "fake"},
            workspace_config=cfg,
            logger=logger,
        )

        handler = self._mock_handler_with_model()
        handler.completion_with_reasoning = MagicMock(
            return_value=("SUMMARY: original task=X; files=foo.md; next=Y", None)
        )

        # Mock env.provenance with two paths.
        env = MagicMock()
        env.provenance.all_paths = MagicMock(return_value=["foo.md", "_rlm_query_0.txt"])

        def fake_get(p):
            entry = MagicMock()
            entry.modified.role = "assistant" if p == "foo.md" else "user"
            entry.modified.turn = 1 if p == "foo.md" else 0
            return entry

        env.provenance.get = MagicMock(side_effect=fake_get)

        completed = [
            WorkspaceIteration(
                iteration=k + 1,
                timestamp="2026-01-01T00:00:00",
                prompt=[],
                response="",
                reasoning=None,
                actions=[],
                observations=[],
            )
            for k in range(3)
        ]

        message_history = [{"role": "user", "content": "long" * 1000}]
        prefix, retained = rlm._compact_history(
            message_history=message_history,
            completed_iterations=completed,
            lm_handler=handler,
            env=env,
            turn=4,
        )

        # Tail of 0 → no iterations retained, all dropped.
        assert retained == []
        # Prefix is exactly [assistant=summary, user=continue].
        assert [m["role"] for m in prefix] == ["assistant", "user"]
        assert "SUMMARY" in prefix[0]["content"]
        # Compaction logged.
        traj_iters = logger._iterations  # type: ignore[attr-defined]
        compaction_rows = [r for r in traj_iters if r.get("type") == "compaction"]
        assert len(compaction_rows) == 1
        assert compaction_rows[0]["turn"] == 4
        assert compaction_rows[0]["dropped_iterations"] == 3
        assert compaction_rows[0]["retained_tail_iterations"] == 0
        # The summary prompt the LM saw must include the original message
        # history plus the user-side summary request.
        sent_messages = handler.completion_with_reasoning.call_args[0][0]
        assert sent_messages[0] == message_history[0]
        assert sent_messages[-1]["role"] == "user"
        assert "summary" in sent_messages[-1]["content"].lower()

    def test_compact_history_preserves_tail(self):
        cfg = WorkspaceConfig(
            compaction=CompactionConfig(threshold_tokens=100, tail_turns_preserved=2)
        )
        rlm = RLM(
            backend="openai",
            backend_kwargs={"model_name": "fake"},
            workspace_config=cfg,
        )

        handler = self._mock_handler_with_model()
        handler.completion_with_reasoning = MagicMock(return_value=("SUM", None))

        env = MagicMock()
        env.provenance.all_paths = MagicMock(return_value=[])
        env.provenance.get = MagicMock(return_value=None)

        completed = [
            WorkspaceIteration(
                iteration=k + 1,
                timestamp="2026-01-01T00:00:00",
                prompt=[],
                response="",
                reasoning=None,
                actions=[],
                observations=[],
            )
            for k in range(5)
        ]
        _, retained = rlm._compact_history(
            message_history=[{"role": "user", "content": "x"}],
            completed_iterations=completed,
            lm_handler=handler,
            env=env,
            turn=10,
        )
        assert [it.iteration for it in retained] == [4, 5]
