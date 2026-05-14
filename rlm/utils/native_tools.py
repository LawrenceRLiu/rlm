"""Native tool-call schemas and conversion for OpenAI-compatible/vLLM backends."""

from __future__ import annotations

import json
from typing import Any

from rlm.core.types import LMToolCall, WorkspaceAction
from rlm.utils.exceptions import ActionParseError

_SCHEMAS: dict[str, dict[str, Any]] = {
    "list_directory": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Workspace-relative directory path. Defaults to '.'.",
            }
        },
        "additionalProperties": False,
    },
    "read_file": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Workspace-relative file path."},
            "start_line": {"type": "integer", "description": "1-indexed inclusive start line."},
            "end_line": {"type": "integer", "description": "1-indexed inclusive end line."},
        },
        "required": ["path"],
        "additionalProperties": False,
    },
    "write_file": {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Workspace-relative file path."},
            "content": {"type": "string", "description": "Complete file contents to write."},
        },
        "required": ["file_path", "content"],
        "additionalProperties": False,
    },
    "append_file": {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Workspace-relative file path."},
            "content": {"type": "string", "description": "Text to append verbatim."},
        },
        "required": ["file_path", "content"],
        "additionalProperties": False,
    },
    "edit": {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Workspace-relative file path."},
            "old_string": {
                "type": "string",
                "description": "Exact literal text to replace; whitespace must match.",
            },
            "new_string": {"type": "string", "description": "Replacement text."},
            "replace_all": {
                "type": "boolean",
                "description": "Replace every occurrence instead of requiring exactly one.",
            },
        },
        "required": ["file_path", "old_string", "new_string"],
        "additionalProperties": False,
    },
    "run_shell_command": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command string. Quote paths; use heredocs for large literals.",
            },
            "description": {"type": "string", "description": "Short human-readable purpose."},
            "directory": {
                "type": "string",
                "description": "Workspace-relative directory to run in. Defaults to workspace root (/).",
            },
            "timeout": {"type": "integer", "description": "Timeout in seconds."},
            "is_background": {
                "type": "boolean",
                "description": "Must be false; background execution is not supported yet.",
            },
        },
        "required": ["command"],
        "additionalProperties": False,
    },
    "run_python_command": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": (
                    "Python code. llm_query, llm_query_batched, rlm_query, and "
                    "rlm_query_batched are pre-imported."
                ),
            },
            "description": {"type": "string", "description": "Short human-readable purpose."},
            "timeout": {"type": "integer", "description": "Timeout in seconds."},
            "cwd": {
                "type": "string",
                "description": "Workspace-relative directory to run in. Defaults to workspace root (/).",
            },
        },
        "required": ["code"],
        "additionalProperties": False,
    },
    "llm_query": {
        "type": "object",
        "properties": {"prompt": {"type": "string", "description": "Prompt for a single LM call."}},
        "required": ["prompt"],
        "additionalProperties": False,
    },
    "rlm_query": {
        "type": "object",
        "properties": {"prompt": {"type": "string", "description": "Task for the child RLM."}},
        "required": ["prompt"],
        "additionalProperties": False,
    },
    "final": {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "description": "Final answer to the user."},
            "artifacts": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Workspace-relative artifact paths to export.",
            },
        },
        "required": ["answer"],
        "additionalProperties": False,
    },
}


def build_openai_tools(*, include_rlm_query: bool) -> list[dict[str, Any]]:
    """Return OpenAI-compatible tool definitions for the native scaffold."""
    from rlm.workspace_tools import get_spec, native_tool_names

    tools: list[dict[str, Any]] = []
    for name in native_tool_names():
        if name == "rlm_query" and not include_rlm_query:
            continue
        spec = get_spec(name)
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": spec.short_description,
                    "parameters": _SCHEMAS[name],
                },
            }
        )
    return tools


def actions_from_tool_calls(calls: list[LMToolCall]) -> list[WorkspaceAction]:
    if not calls:
        raise ActionParseError("No native tool calls returned by the model.")
    actions: list[WorkspaceAction] = []
    for call in calls:
        if call.name not in _SCHEMAS:
            raise ActionParseError(
                f"Unknown native tool '{call.name}'. Known tools: {sorted(_SCHEMAS)}.",
                fragment=json.dumps(call.to_dict())[:500],
            )
        args = dict(call.arguments)
        body = _body_for(call.name, args)
        raw = json.dumps(call.to_dict(), ensure_ascii=False)
        actions.append(
            WorkspaceAction(
                tool=call.name,
                args=args,
                body=body,
                raw=raw,
                call_id=call.id,
            )
        )
    return actions


_BODY_ARG_BY_TOOL: dict[str, str] = {
    "write_file": "content",
    "append_file": "content",
    "run_shell_command": "command",
    "run_python_command": "code",
    "llm_query": "prompt",
    "rlm_query": "prompt",
}


def body_arg_name(name: str) -> str | None:
    """Return the args-key whose value is replayed as the action body, if any.

    Single source of truth for the args-vs-body split used both by
    ``_body_for`` (extraction) and ``render_action_replay`` (deduplication
    in the prompt — body is shown once, not duplicated inside ``args=``).
    """
    return _BODY_ARG_BY_TOOL.get(name)


def _body_for(name: str, args: dict[str, Any]) -> str | None:
    key = body_arg_name(name)
    if key is None:
        return None
    return str(args.get(key, ""))
