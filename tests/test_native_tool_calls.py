from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from rlm.core.config import ParseConfig, WorkspaceConfig
from rlm.core.rlm import RLM
from rlm.core.types import LMCompletionResult, LMToolCall
from rlm.utils.native_tools import actions_from_tool_calls, build_openai_tools


def test_native_tool_schema_includes_qwen_code_style_tools() -> None:
    names = {tool["function"]["name"] for tool in build_openai_tools(include_rlm_query=True)}
    assert "run_shell_command" in names
    assert "run_python_command" in names
    assert "edit" in names
    assert "rlm_query" in names


def test_native_tool_schema_omits_rlm_query_at_max_depth() -> None:
    names = {tool["function"]["name"] for tool in build_openai_tools(include_rlm_query=False)}
    assert "rlm_query" not in names
    assert "llm_query" in names


def test_actions_from_tool_calls_preserves_call_id_and_large_string_args() -> None:
    actions = actions_from_tool_calls(
        [
            LMToolCall(
                id="call_1",
                name="run_python_command",
                arguments={"code": "print('x')\n", "timeout": 10},
            )
        ]
    )
    assert len(actions) == 1
    action = actions[0]
    assert action.tool == "run_python_command"
    assert action.call_id == "call_1"
    assert action.body == "print('x')\n"
    assert action.args["code"] == "print('x')\n"


def test_rlm_native_parse_path_uses_completion_with_tools() -> None:
    cfg = WorkspaceConfig(parse=ParseConfig(action_format="native", max_action_parse_retries=0))
    rlm = RLM(backend="openai", backend_kwargs={"model_name": "fake"}, workspace_config=cfg)
    handler = MagicMock()
    handler.completion_with_tools = MagicMock(
        return_value=LMCompletionResult(
            content="",
            reasoning_content="thought",
            tool_calls=[
                LMToolCall(
                    id="call_final",
                    name="final",
                    arguments={"answer": "done", "artifacts": ["out.txt"]},
                )
            ],
        )
    )

    response, reasoning, actions, attempts = rlm._call_lm_with_parse_retry(
        lm_handler=handler,
        messages=[{"role": "user", "content": "x"}],
    )

    assert response == ""
    assert reasoning == "thought"
    assert attempts == []
    assert actions[0].tool == "final"
    assert actions[0].call_id == "call_final"
    assert actions[0].args["answer"] == "done"


def test_rlm_native_retry_after_missing_tool_calls() -> None:
    cfg = WorkspaceConfig(parse=ParseConfig(action_format="native", max_action_parse_retries=1))
    rlm = RLM(backend="openai", backend_kwargs={"model_name": "fake"}, workspace_config=cfg)
    seen: list[list[dict[str, Any]]] = []

    def fake_completion(messages, *, tools, tool_choice):
        del tools, tool_choice
        seen.append(list(messages))
        if len(seen) == 1:
            return LMCompletionResult(content="plain text", tool_calls=[])
        return LMCompletionResult(
            content="",
            tool_calls=[
                LMToolCall(
                    id="call_final",
                    name="final",
                    arguments={"answer": "done"},
                )
            ],
        )

    handler = MagicMock()
    handler.completion_with_tools = MagicMock(side_effect=fake_completion)

    _, _, actions, attempts = rlm._call_lm_with_parse_retry(
        lm_handler=handler,
        messages=[{"role": "user", "content": "x"}],
    )

    assert len(attempts) == 1
    assert len(seen) == 2
    assert seen[1][-1]["role"] == "user"
    assert "native tool calls" in seen[1][-1]["content"]
    assert actions[0].tool == "final"
