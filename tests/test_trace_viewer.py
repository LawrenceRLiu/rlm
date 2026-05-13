from __future__ import annotations

import json
from pathlib import Path

from rlm.trace_viewer import load_trace, main, render_summary, render_turn


def _write_trace(path: Path) -> None:
    rows = [
        {
            "type": "metadata",
            "timestamp": "2026-01-01T00:00:00",
            "root_model": "mock-model",
            "max_depth": 2,
            "max_iterations": 5,
            "backend": "openai",
            "backend_kwargs": {},
            "action_format": "native",
            "environment_type": "docker",
            "environment_kwargs": {},
            "other_backends": None,
        },
        {
            "type": "iteration",
            "iteration": 1,
            "timestamp": "2026-01-01T00:00:01",
            "prompt": [{"role": "user", "content": "sort these numbers"}],
            "response": '<action tool="shell">pytest</action>',
            "reasoning": None,
            "parse_attempts": [],
            "actions": [
                {
                    "tool": "shell",
                    "args": {},
                    "body": "pytest",
                    "raw": '<action tool="shell">pytest</action>',
                    "call_id": None,
                }
            ],
            "observations": [
                {
                    "tool": "shell",
                    "stdout": "",
                    "stderr": "pytest: command not found",
                    "data": None,
                    "artifacts": [],
                    "execution_time": 0.1,
                    "rlm_calls": [],
                    "final_answer": None,
                    "final_artifacts": [],
                    "error": "shell exited with code 127",
                }
            ],
            "snapshot": {
                "turn": 1,
                "commit_sha": "abc1234",
                "changed_files": ["_rlm_state/action_log.jsonl", "notes.txt"],
                "workspace_root": "/tmp/ws",
            },
            "final_answer": None,
            "iteration_time": 0.25,
            "error": None,
        },
        {
            "type": "iteration",
            "iteration": 2,
            "timestamp": "2026-01-01T00:00:02",
            "prompt": [{"role": "user", "content": "sort these numbers"}],
            "response": '<action tool="rlm_query">check result</action>',
            "reasoning": "considering a child call",
            "parse_attempts": [{"response": "bad xml", "error": "missing action"}],
            "actions": [
                {
                    "tool": "rlm_query",
                    "args": {},
                    "body": "check result",
                    "raw": '<action tool="rlm_query">check result</action>',
                    "call_id": None,
                }
            ],
            "observations": [
                {
                    "tool": "rlm_query",
                    "stdout": "child completed",
                    "stderr": "",
                    "data": None,
                    "artifacts": ["child/answer.txt"],
                    "execution_time": 1.5,
                    "rlm_calls": [
                        {
                            "root_model": "mock-model",
                            "prompt": "check result",
                            "response": "looks sorted",
                            "usage_summary": {"model_usage_summaries": {}},
                            "execution_time": 1.4,
                            "metadata": {
                                "run_metadata": {"root_model": "mock-model"},
                                "iterations": [
                                    {
                                        "type": "iteration",
                                        "iteration": 1,
                                        "timestamp": "2026-01-01T00:00:02",
                                        "prompt": [],
                                        "response": '<action tool="final">ok</action>',
                                        "reasoning": None,
                                        "parse_attempts": [],
                                        "actions": [{"tool": "final", "args": {}, "body": "ok"}],
                                        "observations": [],
                                        "snapshot": None,
                                        "final_answer": "ok",
                                        "iteration_time": 0.5,
                                        "error": None,
                                    }
                                ],
                            },
                        }
                    ],
                    "final_answer": None,
                    "final_artifacts": [],
                    "error": None,
                }
            ],
            "snapshot": {
                "turn": 2,
                "commit_sha": "def5678",
                "changed_files": ["result.txt"],
                "workspace_root": "/tmp/ws",
            },
            "final_answer": "done",
            "iteration_time": 2.0,
            "error": None,
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_render_summary_marks_errors_final_and_children(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    _write_trace(path)

    output = render_summary(load_trace(path), include_children=True)

    assert "backend       : openai model=mock-model" in output
    assert "turn 01" in output
    assert "ERR=1" in output
    assert "[error: shell] shell exited with code 127" in output
    assert "turn 02" in output
    assert "parse_retries=1" in output
    assert "children=1" in output
    assert "FINAL" in output
    assert "child 1: model=mock-model iterations=1" in output


def test_render_turn_can_include_prompt_actions_observations_and_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    _write_trace(path)

    output = render_turn(
        load_trace(path),
        1,
        sections={"overview", "prompt", "actions", "observations", "snapshot"},
    )

    assert "PROMPT" in output
    assert "sort these numbers" in output
    assert "ACTIONS" in output
    assert "body:" in output
    assert "pytest" in output
    assert "OBSERVATIONS" in output
    assert "shell exited with code 127" in output
    assert "SNAPSHOT" in output
    assert "notes.txt" in output


def test_main_prints_summary(tmp_path: Path, capsys) -> None:
    path = tmp_path / "run.jsonl"
    _write_trace(path)

    exit_code = main([str(path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert f"=== {path} ===" in captured.out
    assert "iterations    : 2" in captured.out
