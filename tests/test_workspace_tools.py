"""In-process unit tests for the v0.1 workspace tool suite.

These tests exercise every host-side tool against a thin ``DockerWorkspaceEnv``
(no container) so we can lock in tool-level invariants without paying for
Docker setup. Container-side tools (``shell``, ``python``, ``llm_query``,
``rlm_query``) are exercised end-to-end in ``tests/test_docker_workspace.py``;
here we only cover the small handful of in-process branches they have.

Each tool is its own class. Inside a class, tests share a fresh thin env
created via ``make_thin_env(tmp_path)``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rlm.core.config import DockerConfig, ObservationConfig, WorkspaceConfig
from rlm.core.types import WorkspaceAction
from rlm.environments import docker_workspace as docker_workspace_mod
from rlm.workspace_tools.append_file import execute as append_file_execute
from rlm.workspace_tools.edit import execute as edit_execute
from rlm.workspace_tools.edit_file import execute as edit_file_execute
from rlm.workspace_tools.final import execute as final_execute
from rlm.workspace_tools.list_directory import execute as list_directory_execute
from rlm.workspace_tools.llm_query import execute as llm_query_execute
from rlm.workspace_tools.read_file import execute as read_file_execute
from rlm.workspace_tools.write_file import execute as write_file_execute
from tests._helpers import make_thin_env


def _action(tool: str, *, body: str | None = None, **args: str) -> WorkspaceAction:
    """Tiny factory for ``WorkspaceAction`` objects."""
    return WorkspaceAction(tool=tool, args=dict(args), body=body, raw="")


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


class TestReadFile:
    def test_reads_full_file_when_no_range(self, tmp_path: Path) -> None:
        env = make_thin_env(tmp_path)
        (env.workspace_root / "x.txt").write_text("a\nb\nc\n", encoding="utf-8")
        obs = read_file_execute(env, _action("read_file", path="x.txt"))
        assert obs.error is None
        assert "a\nb\nc" in obs.stdout
        assert obs.data == {"path": "x.txt", "start_line": 1, "end_line": 3, "total_lines": 3}

    def test_slice_via_start_and_end_line(self, tmp_path: Path) -> None:
        env = make_thin_env(tmp_path)
        text = "\n".join(f"line{i}" for i in range(1, 11)) + "\n"
        (env.workspace_root / "y.txt").write_text(text, encoding="utf-8")
        obs = read_file_execute(
            env, _action("read_file", path="y.txt", start_line="3", end_line="5")
        )
        assert obs.error is None
        assert "line3" in obs.stdout
        assert "line5" in obs.stdout
        assert "line2" not in obs.stdout
        assert "line6" not in obs.stdout
        assert obs.data["start_line"] == 3
        assert obs.data["end_line"] == 5

    def test_end_line_past_eof_is_clamped(self, tmp_path: Path) -> None:
        env = make_thin_env(tmp_path)
        (env.workspace_root / "z.txt").write_text("a\nb\n", encoding="utf-8")
        obs = read_file_execute(env, _action("read_file", path="z.txt", end_line="999"))
        assert obs.error is None
        assert obs.data["end_line"] == 2  # clamped to total_lines

    def test_missing_path_attr_errors(self, tmp_path: Path) -> None:
        env = make_thin_env(tmp_path)
        obs = read_file_execute(env, _action("read_file"))
        assert obs.error is not None
        assert "path" in obs.error

    def test_missing_file_errors(self, tmp_path: Path) -> None:
        env = make_thin_env(tmp_path)
        obs = read_file_execute(env, _action("read_file", path="nope.txt"))
        assert obs.error is not None
        assert "does not exist" in obs.error

    def test_directory_target_errors(self, tmp_path: Path) -> None:
        env = make_thin_env(tmp_path)
        obs = read_file_execute(env, _action("read_file", path="_rlm_notes"))
        assert obs.error is not None
        assert "Not a regular file" in obs.error

    def test_binary_file_refused(self, tmp_path: Path) -> None:
        env = make_thin_env(tmp_path)
        (env.workspace_root / "blob.bin").write_bytes(b"\x00\x01\x02\x03" * 128)
        obs = read_file_execute(env, _action("read_file", path="blob.bin"))
        assert obs.error is not None
        assert "binary" in obs.error.lower()

    def test_non_integer_line_bounds_error(self, tmp_path: Path) -> None:
        env = make_thin_env(tmp_path)
        (env.workspace_root / "x.txt").write_text("a\nb\n", encoding="utf-8")
        obs = read_file_execute(env, _action("read_file", path="x.txt", start_line="abc"))
        assert obs.error is not None
        assert "integer" in obs.error

    def test_provenance_role_in_header(self, tmp_path: Path) -> None:
        """Header reports the file's provenance role (``user`` for hand-seeded)."""
        env = make_thin_env(tmp_path)
        (env.workspace_root / "seed.txt").write_text("hi\n", encoding="utf-8")
        env.provenance.record_seed("seed.txt", role="user", action_id=None, turn=0)
        env.provenance.save()
        obs = read_file_execute(env, _action("read_file", path="seed.txt"))
        assert "Created: user" in obs.stdout
        assert "Last modified: user" in obs.stdout


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------


class TestWriteFile:
    def test_creates_file_and_records_provenance(self, tmp_path: Path) -> None:
        env = make_thin_env(tmp_path)
        obs = write_file_execute(env, _action("write_file", body="hello world", path="out.txt"))
        assert obs.error is None
        assert (env.workspace_root / "out.txt").read_text() == "hello world"
        prov = env.provenance.get("out.txt")
        assert prov is not None
        assert prov.created.role == "assistant"
        assert prov.modified.role == "assistant"
        assert prov.created.action_id == env.current_action_id
        assert obs.artifacts == ["out.txt"]

    def test_creates_nested_parent_dirs(self, tmp_path: Path) -> None:
        env = make_thin_env(tmp_path)
        obs = write_file_execute(env, _action("write_file", body="ok", path="a/b/c.txt"))
        assert obs.error is None
        assert (env.workspace_root / "a" / "b" / "c.txt").read_text() == "ok"

    def test_overwrite_keeps_created_role_updates_modified(self, tmp_path: Path) -> None:
        env = make_thin_env(tmp_path)
        # Seed as 'user' first, then assistant overwrites: created stays user.
        (env.workspace_root / "f.txt").write_text("seed", encoding="utf-8")
        env.provenance.record_seed("f.txt", role="user", action_id=None, turn=0)
        env.current_turn = 2
        env.current_action_id = "t2.a1"
        write_file_execute(env, _action("write_file", body="new", path="f.txt"))
        prov = env.provenance.get("f.txt")
        assert prov is not None
        assert prov.created.role == "user"
        assert prov.modified.role == "assistant"
        assert prov.modified.action_id == "t2.a1"

    def test_reserved_path_blocked(self, tmp_path: Path) -> None:
        env = make_thin_env(tmp_path)
        obs = write_file_execute(
            env, _action("write_file", body="x", path="_rlm_state/forbidden.txt")
        )
        assert obs.error is not None
        assert "reserved" in obs.error.lower()
        assert not (env.workspace_root / "_rlm_state" / "forbidden.txt").exists()

    def test_path_traversal_raises(self, tmp_path: Path) -> None:
        env = make_thin_env(tmp_path)
        # ``resolve_workspace_path`` raises ValueError for traversal; the tool
        # does not catch it (this is intentional — the parser should never
        # have produced such a path), so it propagates.
        with pytest.raises(ValueError, match="escapes"):
            write_file_execute(env, _action("write_file", body="x", path="../escape.txt"))

    def test_missing_path_attr_errors(self, tmp_path: Path) -> None:
        env = make_thin_env(tmp_path)
        obs = write_file_execute(env, _action("write_file", body="hi"))
        assert obs.error is not None
        assert "path" in obs.error

    def test_native_content_arg_writes_file(self, tmp_path: Path) -> None:
        env = make_thin_env(tmp_path)
        obs = write_file_execute(
            env,
            WorkspaceAction(
                tool="write_file",
                args={"file_path": "native.txt", "content": "hello\n"},
                body=None,
                raw="",
            ),
        )
        assert obs.error is None
        assert (env.workspace_root / "native.txt").read_text() == "hello\n"


# ---------------------------------------------------------------------------
# append_file
# ---------------------------------------------------------------------------


class TestAppendFile:
    def test_creates_file_when_absent(self, tmp_path: Path) -> None:
        env = make_thin_env(tmp_path)
        obs = append_file_execute(env, _action("append_file", body="first\n", path="log.txt"))
        assert obs.error is None
        assert (env.workspace_root / "log.txt").read_text() == "first\n"

    def test_preserves_prior_content_on_append(self, tmp_path: Path) -> None:
        env = make_thin_env(tmp_path)
        (env.workspace_root / "log.txt").write_text("first\n", encoding="utf-8")
        obs = append_file_execute(env, _action("append_file", body="second\n", path="log.txt"))
        assert obs.error is None
        assert (env.workspace_root / "log.txt").read_text() == "first\nsecond\n"

    def test_provenance_role_assistant(self, tmp_path: Path) -> None:
        env = make_thin_env(tmp_path)
        append_file_execute(env, _action("append_file", body="x", path="out.txt"))
        prov = env.provenance.get("out.txt")
        assert prov is not None
        assert prov.modified.role == "assistant"

    def test_absolute_path_provenance_matches_directory_listing(self, tmp_path: Path) -> None:
        env = make_thin_env(tmp_path)
        append_file_execute(env, _action("append_file", body="x", path="/app/SMILES"))
        prov = env.provenance.get("app/SMILES")
        assert prov is not None
        assert prov.created.role == "assistant"
        assert prov.modified.role == "assistant"

        obs = list_directory_execute(env, _action("list_directory", path="/app"))
        assert obs.error is None
        assert "/app/SMILES" in obs.stdout
        assert "created=assistant  last_modified=assistant" in obs.stdout

    def test_reserved_path_blocked(self, tmp_path: Path) -> None:
        env = make_thin_env(tmp_path)
        obs = append_file_execute(
            env, _action("append_file", body="x", path="_rlm_state/sneak.txt")
        )
        assert obs.error is not None
        assert "reserved" in obs.error.lower()


# ---------------------------------------------------------------------------
# edit_file
# ---------------------------------------------------------------------------


class TestEditFile:
    def test_unique_substring_replaced(self, tmp_path: Path) -> None:
        env = make_thin_env(tmp_path)
        (env.workspace_root / "code.py").write_text("def foo(): return 1\n", encoding="utf-8")
        body = "<search>return 1</search><replace>return 2</replace>"
        obs = edit_file_execute(env, _action("edit_file", body=body, path="code.py"))
        assert obs.error is None
        assert (env.workspace_root / "code.py").read_text() == "def foo(): return 2\n"
        assert obs.data == {"path": "code.py", "occurrences": 1}

    def test_zero_match_errors(self, tmp_path: Path) -> None:
        env = make_thin_env(tmp_path)
        (env.workspace_root / "code.py").write_text("def foo(): return 1\n", encoding="utf-8")
        body = "<search>nonexistent</search><replace>x</replace>"
        obs = edit_file_execute(env, _action("edit_file", body=body, path="code.py"))
        assert obs.error is not None
        assert "not found" in obs.error.lower()
        # File untouched.
        assert "return 1" in (env.workspace_root / "code.py").read_text()

    def test_multi_match_without_flag_errors(self, tmp_path: Path) -> None:
        env = make_thin_env(tmp_path)
        (env.workspace_root / "f.py").write_text("x\nx\nx\n", encoding="utf-8")
        body = "<search>x</search><replace>y</replace>"
        obs = edit_file_execute(env, _action("edit_file", body=body, path="f.py"))
        assert obs.error is not None
        assert "matches 3 times" in obs.error
        # File untouched.
        assert (env.workspace_root / "f.py").read_text() == "x\nx\nx\n"

    def test_multi_match_with_allow_multiple_replaces_all(self, tmp_path: Path) -> None:
        env = make_thin_env(tmp_path)
        (env.workspace_root / "f.py").write_text("x\nx\nx\n", encoding="utf-8")
        body = "<search>x</search><replace>y</replace>"
        obs = edit_file_execute(
            env, _action("edit_file", body=body, path="f.py", allow_multiple="true")
        )
        assert obs.error is None
        assert (env.workspace_root / "f.py").read_text() == "y\ny\ny\n"
        assert obs.data["occurrences"] == 3

    def test_missing_search_block_errors(self, tmp_path: Path) -> None:
        env = make_thin_env(tmp_path)
        (env.workspace_root / "f.py").write_text("x\n", encoding="utf-8")
        obs = edit_file_execute(env, _action("edit_file", body="<replace>y</replace>", path="f.py"))
        assert obs.error is not None
        assert "search" in obs.error.lower()

    def test_missing_file_errors(self, tmp_path: Path) -> None:
        env = make_thin_env(tmp_path)
        body = "<search>a</search><replace>b</replace>"
        obs = edit_file_execute(env, _action("edit_file", body=body, path="ghost.py"))
        assert obs.error is not None
        assert "does not exist" in obs.error

    def test_reserved_path_blocked(self, tmp_path: Path) -> None:
        env = make_thin_env(tmp_path)
        body = "<search>a</search><replace>b</replace>"
        obs = edit_file_execute(env, _action("edit_file", body=body, path="_rlm_state/x"))
        assert obs.error is not None
        assert "reserved" in obs.error.lower()


# ---------------------------------------------------------------------------
# edit
# ---------------------------------------------------------------------------


class TestEdit:
    def test_exact_literal_replacement(self, tmp_path: Path) -> None:
        env = make_thin_env(tmp_path)
        (env.workspace_root / "code.py").write_text("def f():\n    return 1\n", encoding="utf-8")
        obs = edit_execute(
            env,
            WorkspaceAction(
                tool="edit",
                args={
                    "file_path": "code.py",
                    "old_string": "    return 1\n",
                    "new_string": "    return 2\n",
                },
                body=None,
                raw="",
            ),
        )
        assert obs.error is None
        assert (env.workspace_root / "code.py").read_text() == "def f():\n    return 2\n"

    def test_multiple_matches_require_replace_all(self, tmp_path: Path) -> None:
        env = make_thin_env(tmp_path)
        (env.workspace_root / "f.txt").write_text("x\nx\n", encoding="utf-8")
        obs = edit_execute(
            env,
            WorkspaceAction(
                tool="edit",
                args={"file_path": "f.txt", "old_string": "x\n", "new_string": "y\n"},
                body=None,
                raw="",
            ),
        )
        assert obs.error is not None
        assert "matches 2 times" in obs.error


# ---------------------------------------------------------------------------
# list_directory
# ---------------------------------------------------------------------------


class TestListDirectory:
    def test_default_path_is_workspace_root(self, tmp_path: Path) -> None:
        env = make_thin_env(tmp_path)
        (env.workspace_root / "a.txt").write_text("a", encoding="utf-8")
        obs = list_directory_execute(env, _action("list_directory"))
        assert obs.error is None
        assert "a.txt" in obs.stdout
        # The reserved layout should also be visible.
        assert "_rlm_notes" in obs.stdout
        assert "_rlm_artifacts" in obs.stdout

    def test_filters_default_ignores(self, tmp_path: Path) -> None:
        env = make_thin_env(tmp_path)
        (env.workspace_root / ".git").mkdir(exist_ok=True)
        (env.workspace_root / "__pycache__").mkdir(exist_ok=True)
        (env.workspace_root / "node_modules").mkdir(exist_ok=True)
        (env.workspace_root / ".venv").mkdir(exist_ok=True)
        obs = list_directory_execute(env, _action("list_directory"))
        names = [e["name"] for e in obs.data["entries"]]
        for ignored in (".git", "__pycache__", "node_modules", ".venv"):
            assert ignored not in names

    def test_truncates_at_cap(self, tmp_path: Path) -> None:
        cfg = WorkspaceConfig(observation=ObservationConfig(max_list_directory_entries=3))
        env = make_thin_env(tmp_path, workspace_config=cfg)
        for i in range(10):
            (env.workspace_root / f"f{i}.txt").write_text("x", encoding="utf-8")
        obs = list_directory_execute(env, _action("list_directory"))
        assert obs.data["truncated"] is True
        assert len(obs.data["entries"]) == 3
        assert "truncated at 3 entries" in obs.stdout

    def test_empty_directory_is_explicit(self, tmp_path: Path) -> None:
        env = make_thin_env(tmp_path)
        obs = list_directory_execute(env, _action("list_directory", path="_rlm_notes"))
        assert obs.error is None
        assert obs.data["entries"] == []
        assert "[empty directory]" in obs.stdout

    def test_missing_dir_errors(self, tmp_path: Path) -> None:
        env = make_thin_env(tmp_path)
        obs = list_directory_execute(env, _action("list_directory", path="nope"))
        assert obs.error is not None
        assert "does not exist" in obs.error

    def test_target_is_file_errors(self, tmp_path: Path) -> None:
        env = make_thin_env(tmp_path)
        (env.workspace_root / "f.txt").write_text("x", encoding="utf-8")
        obs = list_directory_execute(env, _action("list_directory", path="f.txt"))
        assert obs.error is not None
        assert "Not a directory" in obs.error

    def test_uses_last_modified_label(self, tmp_path: Path) -> None:
        env = make_thin_env(tmp_path)
        (env.workspace_root / "x.txt").write_text("x", encoding="utf-8")
        obs = list_directory_execute(env, _action("list_directory"))
        assert "last_modified=" in obs.stdout
        assert "  modified=" not in obs.stdout


# ---------------------------------------------------------------------------
# Docker network policy helpers
# ---------------------------------------------------------------------------


class TestDockerNetworkPolicy:
    def test_no_internet_uses_docker_network_none(self, tmp_path: Path) -> None:
        cfg = WorkspaceConfig(docker=DockerConfig(allow_internet=False, image="task:latest"))
        env = make_thin_env(tmp_path, workspace_config=cfg)
        calls: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            del kwargs
            calls.append(cmd)
            if cmd[:3] == ["docker", "run", "-d"]:
                return MagicMock(returncode=0, stdout="abcdef123456\n", stderr="")
            raise AssertionError(f"unexpected command: {cmd}")

        with patch.object(docker_workspace_mod.subprocess, "run", side_effect=fake_run):
            with patch.object(
                env, "_select_broker_exec_python", return_value="/opt/broker/bin/python"
            ):
                with patch.object(env, "_wait_for_broker_ready"):
                    env._start_container()

        run_cmd = calls[0]
        assert "--network" in run_cmd
        assert "none" in run_cmd
        assert "-p" not in run_cmd
        assert "--add-host=host.docker.internal:host-gateway" not in run_cmd
        assert env._broker_host_port is None
        assert env._broker_exec_python == "/opt/broker/bin/python"


# ---------------------------------------------------------------------------
# final
# ---------------------------------------------------------------------------


class TestFinal:
    def test_answer_only(self, tmp_path: Path) -> None:
        env = make_thin_env(tmp_path)
        body = "<answer>The answer is 42.</answer>"
        obs = final_execute(env, _action("final", body=body))
        assert obs.error is None
        assert obs.final_answer == "The answer is 42."
        assert obs.final_artifacts == []
        assert obs.stdout == "The answer is 42."

    def test_answer_with_artifacts(self, tmp_path: Path) -> None:
        env = make_thin_env(tmp_path)
        body = (
            '<answer>see attached</answer><artifact path="out/a.csv" /><artifact path="out/b.md" />'
        )
        obs = final_execute(env, _action("final", body=body))
        assert obs.final_answer == "see attached"
        assert obs.final_artifacts == ["out/a.csv", "out/b.md"]

    def test_missing_answer_errors(self, tmp_path: Path) -> None:
        env = make_thin_env(tmp_path)
        obs = final_execute(env, _action("final", body='<artifact path="x" />'))
        assert obs.error is not None
        assert "answer" in obs.error.lower()

    def test_artifact_missing_path_errors(self, tmp_path: Path) -> None:
        env = make_thin_env(tmp_path)
        body = "<answer>ok</answer><artifact />"
        obs = final_execute(env, _action("final", body=body))
        assert obs.error is not None


# ---------------------------------------------------------------------------
# llm_query (in-process branches: missing handler address, host=None)
# ---------------------------------------------------------------------------


class TestLLMQuery:
    def test_no_handler_address_errors(self, tmp_path: Path) -> None:
        env = make_thin_env(tmp_path)
        # make_thin_env leaves lm_handler_address=None.
        obs = llm_query_execute(env, _action("llm_query", body="hi"))
        assert obs.error is not None
        assert "handler" in obs.error.lower()


# ---------------------------------------------------------------------------
# observation spilling (env-side, but lives on the same hot path as tools)
# ---------------------------------------------------------------------------


class TestObservationSpill:
    def test_under_cap_does_not_spill(self, tmp_path: Path) -> None:
        # Use a generous cap so the read_file header (~70 chars) plus the
        # body fits comfortably below it.
        cfg = WorkspaceConfig(observation=ObservationConfig(max_observation_chars=10_000))
        env = make_thin_env(tmp_path, workspace_config=cfg)
        write_file_execute(env, _action("write_file", body="x" * 50, path="a.txt"))
        # run_action applies spilling. We exercise the env's run_action path
        # so we cover the spill branch the same way production does.
        action = _action("read_file", path="a.txt")
        obs = env.run_action(action)
        assert "[Observation truncated" not in obs.stdout
        # No spill file was created.
        assert not list((env.workspace_root / "_rlm_artifacts" / "_observations").glob("*.txt"))

    def test_over_cap_spills_with_summary(self, tmp_path: Path) -> None:
        cfg = WorkspaceConfig(observation=ObservationConfig(max_observation_chars=100))
        env = make_thin_env(tmp_path, workspace_config=cfg)
        body = "y" * 1000
        write_file_execute(env, _action("write_file", body=body, path="big.txt"))
        action = _action("read_file", path="big.txt")
        obs = env.run_action(action)
        assert "[Observation truncated" in obs.stdout
        assert "_rlm_artifacts/_observations/" in obs.stdout
        # And a real spill file exists.
        spilled = list((env.workspace_root / "_rlm_artifacts" / "_observations").glob("*.txt"))
        assert len(spilled) == 1

    def test_boundary_at_cap_does_not_spill(self, tmp_path: Path) -> None:
        """Observation length == max chars must NOT spill (the cap is inclusive)."""
        # Pick a tiny cap and craft a read whose total stdout is exactly the cap.
        cfg = WorkspaceConfig(observation=ObservationConfig(max_observation_chars=200))
        env = make_thin_env(tmp_path, workspace_config=cfg)
        # The header line in read_file's stdout has variable length; we just
        # need to verify the boundary semantics on the env's spill helper
        # rather than craft a precise read.
        from rlm.core.types import WorkspaceObservation

        at_cap = WorkspaceObservation(tool="read_file", stdout="x" * 200, stderr="")
        env.current_action_id = "t1.a99"
        out = env._maybe_spill_observation(at_cap)
        assert out.stdout == "x" * 200  # untouched

        # One byte over the cap → spill.
        over = WorkspaceObservation(tool="read_file", stdout="x" * 201, stderr="")
        env.current_action_id = "t1.a100"
        out = env._maybe_spill_observation(over)
        assert "[Observation truncated" in out.stdout


# ---------------------------------------------------------------------------
# Path resolution (tool-shared helpers, exercised via env)
# ---------------------------------------------------------------------------


class TestPathResolution:
    def test_absolute_path_rejected(self, tmp_path: Path) -> None:
        """Absolute paths that are NOT under a bind-mount root (/app,
        /_rlm_state, /_rlm_artifacts, /_rlm_notes) are rejected."""
        env = make_thin_env(tmp_path)
        with pytest.raises(ValueError, match="not under a bind-mounted root"):
            env.resolve_workspace_path("/etc/passwd")

    def test_absolute_app_path_accepted(self, tmp_path: Path) -> None:
        """Sibling-layout: /app/... is a valid container-absolute path that
        translates to the host-side app/ bind source."""
        env = make_thin_env(tmp_path)
        p = env.resolve_workspace_path("/app/output.txt")
        assert p == (env.workspace_root / "app" / "output.txt").resolve()

    def test_absolute_rlm_notes_path_accepted(self, tmp_path: Path) -> None:
        env = make_thin_env(tmp_path)
        p = env.resolve_workspace_path("/_rlm_notes/n.md")
        assert p == (env.workspace_root / "_rlm_notes" / "n.md").resolve()

    def test_dotdot_traversal_rejected(self, tmp_path: Path) -> None:
        env = make_thin_env(tmp_path)
        with pytest.raises(ValueError, match="escapes"):
            env.resolve_workspace_path("../outside.txt")

    def test_relative_path_resolves_inside_workspace(self, tmp_path: Path) -> None:
        env = make_thin_env(tmp_path)
        p = env.resolve_workspace_path("a/b.txt")
        assert p == (env.workspace_root / "a" / "b.txt").resolve()

    def test_reserved_state_dir_detected(self, tmp_path: Path) -> None:
        env = make_thin_env(tmp_path)
        assert env.is_reserved_path("_rlm_state")
        assert env.is_reserved_path("_rlm_state/foo.json")
        assert env.is_reserved_path("./_rlm_state/foo.json")
        # Other dirs are NOT reserved.
        assert not env.is_reserved_path("_rlm_artifacts/x")
        assert not env.is_reserved_path("_rlm_notes/x")
        assert not env.is_reserved_path("_rlm_query_0.txt")

    def test_run_action_converts_absolute_path_to_observation(self, tmp_path: Path) -> None:
        """A model emitting an absolute path that's NOT under a bind-mount
        root (e.g. ``/workspace/foo.txt``, the old shadow path) must get an
        observation it can read on the next turn, not abort the run.
        Regression for Qwen3.5-9B 3d 2026-05-10 where this killed the loop."""
        env = make_thin_env(tmp_path)
        action = WorkspaceAction(
            tool="write_file",
            args={"path": "/workspace/foo.txt"},
            body="hello",
            raw="",
        )
        obs = env.run_action(action)
        assert obs.error is not None
        assert "not under a bind-mounted root" in obs.error
        assert obs.tool == "write_file"

    def test_run_action_converts_traversal_path_to_observation(self, tmp_path: Path) -> None:
        """``../foo`` must also become an observation, not an exception."""
        env = make_thin_env(tmp_path)
        action = WorkspaceAction(
            tool="read_file",
            args={"path": "../escape.txt"},
            body=None,
            raw="",
        )
        obs = env.run_action(action)
        assert obs.error is not None
        assert "escapes" in obs.error
        assert obs.tool == "read_file"
