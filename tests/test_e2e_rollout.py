"""Scripted end-to-end rollout test.

Drives the workspace ``RLM._run_loop`` through a hand-authored 3-turn
conversation with a ``MockLM`` (via a stand-in LM handler). The env is a
thin (no-Docker) ``DockerWorkspaceEnv`` so we can exercise every host-side
behavior — git snapshots, provenance, action_log, JSONL logging, observation
spilling — without paying for a container.

The test is fast (sub-second) and runs by default. The single Docker-gated
end-to-end test of the same flow lives in ``tests/test_docker_workspace.py``.

The rollout simulates a small task:
  Turn 1 — ``<action tool="list_directory" />`` (read-only)
  Turn 2 — ``<action tool="write_file" path="result.txt">42</action>``
  Turn 3 — ``<action tool="final"><answer>The answer is 42.</answer>
            <artifact path="result.txt" /></action>``

Asserts performed:
  - ``RLMChatCompletion.response`` is the final answer
  - JSONL has metadata first + 3 iteration lines, contiguous numbering
  - Each iteration has a snapshot.commit_sha; SHAs are distinct across turns
  - Per-turn git commits exist with messages "turn 0", "turn 1", "turn 2", "turn 3"
  - Provenance: result.txt is role=assistant
  - ``_rlm_state/action_log.jsonl`` records each action in order
  - ``_last_final_artifacts`` contains result.txt
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

from rlm.core.rlm import RLM
from tests._helpers import make_thin_env


def _mock_lm_handler(responses: list[tuple[str, str | None]]) -> MagicMock:
    handler = MagicMock()
    handler.completion_with_reasoning = MagicMock(side_effect=list(responses))
    # `_build_completion` reads usage; return an empty UsageSummary.
    from rlm.core.types import UsageSummary

    handler.get_usage_summary = MagicMock(return_value=UsageSummary(model_usage_summaries={}))
    return handler


def _scripted_run(tmp_path: Path, responses: list[str], *, log_dir: Path | None = None):
    """Run the RLM loop end-to-end against a thin env with scripted responses."""
    from rlm.logger import RLMLogger

    env = make_thin_env(tmp_path)
    logger = RLMLogger(log_dir=str(log_dir)) if log_dir is not None else RLMLogger(log_dir=None)
    rlm = RLM(
        backend="openai",
        backend_kwargs={"model_name": "mock-model"},
        logger=logger,
        max_iterations=10,
    )
    handler = _mock_lm_handler([(r, None) for r in responses])
    result = rlm._run_loop(prompt="solve me", root_prompt=None, lm_handler=handler, env=env)
    return rlm, env, result, logger


# ---------------------------------------------------------------------------
# Three-turn rollout to a final answer
# ---------------------------------------------------------------------------


SCRIPT_HAPPY_PATH = [
    'I will inspect first.\n<action tool="list_directory" />',
    'Now I save the answer.\n<action tool="write_file" path="result.txt">42</action>',
    (
        "Done.\n"
        '<action tool="final">'
        "<answer>The answer is 42.</answer>"
        '<artifact path="result.txt" />'
        "</action>"
    ),
]


class TestHappyPathRollout:
    def test_run_loop_returns_final_answer(self, tmp_path: Path) -> None:
        _, _, result, _ = _scripted_run(tmp_path, SCRIPT_HAPPY_PATH)
        assert result.response == "The answer is 42."

    def test_jsonl_metadata_then_three_iterations(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        _, _, _, logger = _scripted_run(tmp_path, SCRIPT_HAPPY_PATH, log_dir=log_dir)
        files = list(log_dir.glob("*.jsonl"))
        assert len(files) == 1
        lines = [json.loads(line) for line in files[0].read_text().splitlines() if line]
        assert lines[0]["type"] == "metadata"
        iter_lines = [line for line in lines if line["type"] == "iteration"]
        assert [line["iteration"] for line in iter_lines] == [1, 2, 3]
        # Every iteration carries a snapshot SHA and a non-null iteration_time.
        for it in iter_lines:
            assert it["snapshot"] is not None
            assert it["snapshot"]["commit_sha"]
            assert it["iteration_time"] is not None
        # Final iteration carries the answer text.
        assert iter_lines[-1]["final_answer"] == "The answer is 42."

    def test_distinct_commit_shas_per_turn(self, tmp_path: Path) -> None:
        _, _, result, _ = _scripted_run(tmp_path, SCRIPT_HAPPY_PATH)
        traj = result.metadata
        assert traj is not None
        shas = [it["snapshot"]["commit_sha"] for it in traj["iterations"]]
        assert len(set(shas)) == 3, f"expected 3 distinct SHAs; got {shas}"

    def test_git_log_lists_turn_messages(self, tmp_path: Path) -> None:
        _, env, _, _ = _scripted_run(tmp_path, SCRIPT_HAPPY_PATH)
        # Initial commit was "turn 0"; three turns add "turn 1".."turn 3".
        log = (
            subprocess.run(
                ["git", "-C", str(env.workspace_root), "log", "--format=%s"],
                capture_output=True,
                text=True,
                check=True,
            )
            .stdout.strip()
            .splitlines()
        )
        # Newest first.
        assert log[:4] == ["turn 3", "turn 2", "turn 1", "turn 0"]

    def test_provenance_marks_result_as_assistant(self, tmp_path: Path) -> None:
        _, env, _, _ = _scripted_run(tmp_path, SCRIPT_HAPPY_PATH)
        # Reload so we read what was persisted to disk, not in-memory state.
        from rlm.utils.provenance import ProvenanceStore

        prov = ProvenanceStore(env.workspace_root / "_rlm_state" / "provenance.json")
        prov.load()
        entry = prov.get("result.txt")
        assert entry is not None
        assert entry.created.role == "assistant"
        assert entry.modified.role == "assistant"
        # The seed task and the state files retain their roles.
        assert prov.get("_rlm_query_0.txt").created.role == "user"
        assert prov.get("_rlm_state/provenance.json").created.role == "system"

    def test_action_log_records_each_action_in_order(self, tmp_path: Path) -> None:
        _, env, _, _ = _scripted_run(tmp_path, SCRIPT_HAPPY_PATH)
        log_path = env.workspace_root / "_rlm_state" / "action_log.jsonl"
        records = [json.loads(line) for line in log_path.read_text().splitlines() if line]
        assert [r["tool"] for r in records] == ["list_directory", "write_file", "final"]
        # Action ids are ``t<turn>.a<idx>`` and increment within a turn.
        assert records[0]["action_id"] == "t1.a1"
        assert records[1]["action_id"] == "t2.a1"
        assert records[2]["action_id"] == "t3.a1"
        # The mutating flag is correctly recorded.
        assert records[0]["mutating"] is False  # list_directory
        assert records[1]["mutating"] is True  # write_file
        # final is non-mutating per its SPEC.
        assert records[2]["mutating"] is False

    def test_final_artifacts_captured_on_rlm(self, tmp_path: Path) -> None:
        rlm, _, _, _ = _scripted_run(tmp_path, SCRIPT_HAPPY_PATH)
        assert rlm._last_final_artifacts == ["result.txt"]


# ---------------------------------------------------------------------------
# Mixed read-only + mutating: error in mutating must NOT halt read-only
# ---------------------------------------------------------------------------


class TestMixedDispatchInRollout:
    def test_readonly_continues_after_mutating_error(self, tmp_path: Path) -> None:
        """Two-turn rollout where turn 1 writes to a reserved path (mutating
        error), then a read-only ``list_directory`` runs in the same turn.
        The read-only must still succeed and its observation must reach the
        next turn's history. Then turn 2 issues final."""
        responses = [
            (
                "Trying both.\n"
                '<action tool="write_file" path="_rlm_state/sneak">x</action>\n'
                '<action tool="list_directory" />'
            ),
            '<action tool="final"><answer>ok</answer></action>',
        ]
        _, env, result, logger = _scripted_run(tmp_path, responses)
        assert result.response == "ok"
        traj = logger.get_trajectory()
        assert traj is not None
        first_turn = traj["iterations"][0]
        # Turn 1 had two actions; both have observations.
        assert len(first_turn["actions"]) == 2
        assert len(first_turn["observations"]) == 2
        # First was the mutating write to a reserved path → error.
        assert first_turn["observations"][0]["error"] is not None
        assert "reserved" in first_turn["observations"][0]["error"].lower()
        # Second was the read-only list_directory → succeeded.
        assert first_turn["observations"][1]["error"] is None


# ---------------------------------------------------------------------------
# Per-tool spill isolation: spill-on-tool-1 must not affect tool-2
# ---------------------------------------------------------------------------


class TestSpillIsolationInRollout:
    def test_only_the_oversized_tool_spills(self, tmp_path: Path) -> None:
        from rlm.core.config import ObservationConfig, WorkspaceConfig
        from rlm.logger import RLMLogger

        # Cap chosen so the small list_directory output (~200 chars) fits but
        # the read_file output (5000+ chars of body, plus header) does not.
        cfg = WorkspaceConfig(observation=ObservationConfig(max_observation_chars=500))
        env = make_thin_env(tmp_path, workspace_config=cfg)
        # Pre-populate a big file the model will read on turn 1.
        big = "y" * 5000
        (env.workspace_root / "big.txt").write_text(big, encoding="utf-8")
        env.provenance.record_seed("big.txt", role="user", action_id=None, turn=0)
        env.provenance.save()

        logger = RLMLogger(log_dir=None)
        rlm = RLM(
            backend="openai",
            backend_kwargs={"model_name": "mock-model"},
            workspace_config=cfg,
            logger=logger,
            max_iterations=5,
        )
        responses = [
            (
                # Turn 1: read big file (will spill) AND list_directory (small).
                '<action tool="read_file" path="big.txt" />\n<action tool="list_directory" />'
            ),
            '<action tool="final"><answer>done</answer></action>',
        ]
        handler = _mock_lm_handler([(r, None) for r in responses])
        rlm._run_loop(prompt="x", root_prompt=None, lm_handler=handler, env=env)

        traj = logger.get_trajectory()
        first_turn_obs = traj["iterations"][0]["observations"]
        # Tool 1 (read_file): spilled.
        assert "[Observation truncated" in first_turn_obs[0]["stdout"]
        assert any("_rlm_artifacts/_observations/" in art for art in first_turn_obs[0]["artifacts"])
        # Tool 2 (list_directory): NOT spilled.
        assert "[Observation truncated" not in first_turn_obs[1]["stdout"]


# ---------------------------------------------------------------------------
# Golden JSONL snapshot — visualizer schema lock
# ---------------------------------------------------------------------------


GOLDEN_FIXTURE = Path(__file__).parent / "fixtures" / "golden_rollout.jsonl"


class TestSystemPromptInvariants:
    """Content-level smoke checks complementing the schema-only golden test.

    These guard the prompt against regressions that the schema-only golden
    would miss (e.g. a tool registered but never advertised, or the workspace
    layout no longer interpolated).
    """

    def test_system_prompt_advertises_every_tool(self, tmp_path: Path) -> None:
        from rlm.workspace_tools import all_tool_names

        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        _scripted_run(tmp_path, SCRIPT_HAPPY_PATH, log_dir=log_dir)
        live_file = next(log_dir.glob("*.jsonl"))
        lines = [json.loads(line) for line in live_file.read_text().splitlines() if line]
        iter1 = next(line for line in lines if line.get("type") == "iteration")
        system_msg = iter1["prompt"][0]
        assert system_msg["role"] == "system"
        sys_text = system_msg["content"]
        for tool_name in all_tool_names():
            assert tool_name in sys_text, f"tool {tool_name!r} missing from system prompt"
        # Workspace layout is interpolated (catches a busted template).
        assert "_rlm_query_0.txt" in sys_text

    def test_final_answer_round_trips_to_jsonl(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        _scripted_run(tmp_path, SCRIPT_HAPPY_PATH, log_dir=log_dir)
        live_file = next(log_dir.glob("*.jsonl"))
        lines = [json.loads(line) for line in live_file.read_text().splitlines() if line]
        iters = [line for line in lines if line.get("type") == "iteration"]
        assert iters[-1]["final_answer"] == "The answer is 42."


class TestGoldenJSONL:
    def test_jsonl_shape_matches_golden(self, tmp_path: Path) -> None:
        """The JSONL emitted by a fixed rollout must structurally match the
        checked-in golden. ``schema_of_jsonl`` reduces each record to its
        key/type schema (scalars → type tags, lists → merged element schema)
        so this only fires on real shape changes, not prompt or content edits.

        If this fails, the Python ``to_dict()`` shape changed — the
        visualizer's TypeScript types in ``visualizer/src/lib/types.ts``
        almost certainly need updating in the same PR. To regenerate the
        golden after a deliberate schema change, run::

            python -m tests.test_e2e_rollout --regenerate-golden
        """
        from tests._helpers import schema_of_jsonl

        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        _scripted_run(tmp_path, SCRIPT_HAPPY_PATH, log_dir=log_dir)
        live_file = next(log_dir.glob("*.jsonl"))
        live = schema_of_jsonl(live_file.read_text())

        if not GOLDEN_FIXTURE.exists():
            GOLDEN_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
            GOLDEN_FIXTURE.write_text(live)
            # Fail on first generation so the user commits the new fixture
            # deliberately.
            raise AssertionError(
                f"Golden fixture did not exist; wrote {GOLDEN_FIXTURE}. "
                "Review the file and commit it, then re-run."
            )

        golden = GOLDEN_FIXTURE.read_text()
        if live != golden:
            diff_path = log_dir / "live_normalized.jsonl"
            diff_path.write_text(live)
            assert live == golden, (
                f"Normalized JSONL diverged from golden. Live output written "
                f"to {diff_path}. If the schema change is intentional, update "
                f"{GOLDEN_FIXTURE} and `visualizer/src/lib/types.ts` together."
            )


# ---------------------------------------------------------------------------
# CLI helper to (re-)generate the golden fixture deliberately.
# ---------------------------------------------------------------------------


def _regenerate_golden() -> None:  # pragma: no cover — invoked from CLI only
    import shutil
    import sys
    import tempfile

    from tests._helpers import schema_of_jsonl

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        log_dir = td_path / "logs"
        log_dir.mkdir()
        _scripted_run(td_path, SCRIPT_HAPPY_PATH, log_dir=log_dir)
        live_file = next(log_dir.glob("*.jsonl"))
        live = schema_of_jsonl(live_file.read_text())
    GOLDEN_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_FIXTURE.write_text(live)
    shutil.rmtree(td_path, ignore_errors=True)
    print(f"Wrote {GOLDEN_FIXTURE}", file=sys.stderr)


if __name__ == "__main__":  # pragma: no cover
    import sys

    if "--regenerate-golden" in sys.argv:
        _regenerate_golden()
    else:
        print("Use pytest to run; pass --regenerate-golden to update the fixture.")
