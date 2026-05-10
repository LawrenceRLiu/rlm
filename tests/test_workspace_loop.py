"""Unit tests for the workspace-substrate RLM loop (Phase 4).

These tests do not require Docker. They mock the LM handler and the
workspace env so the parse-and-retry inner loop, action dispatch, and
prompt construction can be exercised in isolation.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from rlm.core.config import WorkspaceConfig
from rlm.core.rlm import RLM
from rlm.core.types import (
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
        msgs = format_workspace_iteration(it)
        assert [m["role"] for m in msgs] == ["assistant", "user"]
        assert msgs[0]["content"] == "prose with action"
        assert "t1.a1" in msgs[1]["content"]
        assert "hello" in msgs[1]["content"]
        assert "snapshot" in msgs[1]["content"]
        assert "abcdef1" in msgs[1]["content"]


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
        cfg = WorkspaceConfig()
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


# ---------------------------------------------------------------------------
# Failed iteration is logged when parse retries are exhausted
# ---------------------------------------------------------------------------


class TestFailedIterationLogging:
    """When parse retries exhaust, the partial iteration must reach the log."""

    def _rlm(self, retries: int = 2) -> RLM:
        cfg = WorkspaceConfig()
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

        cfg = WorkspaceConfig()
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
