"""Tests for core types."""

from rlm.core.types import (
    ModelUsageSummary,
    QueryMetadata,
    RLMChatCompletion,
    RLMMetadata,
    UsageSummary,
    WorkspaceIteration,
    _serialize_value,
)


class TestSerializeValue:
    """Tests for _serialize_value helper."""

    def test_primitives(self):
        assert _serialize_value(None) is None
        assert _serialize_value(True) is True
        assert _serialize_value(42) == 42
        assert _serialize_value(3.14) == 3.14
        assert _serialize_value("hello") == "hello"

    def test_list(self):
        result = _serialize_value([1, 2, "three"])
        assert result == [1, 2, "three"]

    def test_dict(self):
        result = _serialize_value({"a": 1, "b": 2})
        assert result == {"a": 1, "b": 2}

    def test_callable(self):
        def my_func():
            pass

        result = _serialize_value(my_func)
        assert "function" in result.lower()
        assert "my_func" in result


class TestModelUsageSummary:
    """Tests for ModelUsageSummary."""

    def test_to_dict(self):
        summary = ModelUsageSummary(
            total_calls=10, total_input_tokens=1000, total_output_tokens=500
        )
        d = summary.to_dict()
        assert d["total_calls"] == 10
        assert d["total_input_tokens"] == 1000
        assert d["total_output_tokens"] == 500

    def test_from_dict(self):
        data = {
            "total_calls": 5,
            "total_input_tokens": 200,
            "total_output_tokens": 100,
        }
        summary = ModelUsageSummary.from_dict(data)
        assert summary.total_calls == 5
        assert summary.total_input_tokens == 200
        assert summary.total_output_tokens == 100


class TestUsageSummary:
    """Tests for UsageSummary."""

    def test_to_dict(self):
        model_summary = ModelUsageSummary(
            total_calls=1, total_input_tokens=10, total_output_tokens=5
        )
        summary = UsageSummary(model_usage_summaries={"gpt-4": model_summary})
        d = summary.to_dict()
        assert "gpt-4" in d["model_usage_summaries"]

    def test_from_dict(self):
        data = {
            "model_usage_summaries": {
                "gpt-4": {
                    "total_calls": 2,
                    "total_input_tokens": 50,
                    "total_output_tokens": 25,
                }
            }
        }
        summary = UsageSummary.from_dict(data)
        assert "gpt-4" in summary.model_usage_summaries
        assert summary.model_usage_summaries["gpt-4"].total_calls == 2


class TestRLMChatCompletion:
    """Tests for RLMChatCompletion."""

    def test_metadata_default_none(self):
        usage = UsageSummary(model_usage_summaries={})
        c = RLMChatCompletion(
            root_model="gpt-4",
            prompt="hi",
            response="hello",
            usage_summary=usage,
            execution_time=1.0,
        )
        assert c.metadata is None
        d = c.to_dict()
        assert "metadata" not in d

    def test_metadata_roundtrip(self):
        usage = UsageSummary(model_usage_summaries={})
        trajectory = {"run_metadata": {"root_model": "gpt-4"}, "iterations": []}
        c = RLMChatCompletion(
            root_model="gpt-4",
            prompt="hi",
            response="hello",
            usage_summary=usage,
            execution_time=1.0,
            metadata=trajectory,
        )
        d = c.to_dict()
        assert d["metadata"] == trajectory
        c2 = RLMChatCompletion.from_dict(d)
        assert c2.metadata == trajectory

    def test_final_artifacts_default_empty(self):
        c = RLMChatCompletion(
            root_model="gpt-4",
            prompt="hi",
            response="hello",
            usage_summary=UsageSummary(model_usage_summaries={}),
            execution_time=1.0,
        )
        assert c.final_artifacts == []
        assert c.workspace_root is None
        d = c.to_dict()
        # final_artifacts always present (empty list is meaningful for callers);
        # workspace_root omitted when None.
        assert d["final_artifacts"] == []
        assert "workspace_root" not in d

    def test_final_artifacts_roundtrip(self):
        c = RLMChatCompletion(
            root_model="gpt-4",
            prompt="hi",
            response="see attached",
            usage_summary=UsageSummary(model_usage_summaries={}),
            execution_time=1.0,
            final_artifacts=["report.pdf", "data.csv"],
            workspace_root="/tmp/rlm_runs/abc123",
        )
        d = c.to_dict()
        assert d["final_artifacts"] == ["report.pdf", "data.csv"]
        assert d["workspace_root"] == "/tmp/rlm_runs/abc123"
        c2 = RLMChatCompletion.from_dict(d)
        assert c2.final_artifacts == ["report.pdf", "data.csv"]
        assert c2.workspace_root == "/tmp/rlm_runs/abc123"

    def test_read_artifact_raises_without_workspace_root(self):
        import pytest

        c = RLMChatCompletion(
            root_model="gpt-4",
            prompt="hi",
            response="r",
            usage_summary=UsageSummary(model_usage_summaries={}),
            execution_time=1.0,
            final_artifacts=["x.txt"],
            workspace_root=None,
        )
        with pytest.raises(RuntimeError, match="workspace_root is None"):
            c.read_artifact("x.txt")

    def test_read_artifact_reads_text(self, tmp_path):
        (tmp_path / "report.md").write_text("hello\n", encoding="utf-8")
        c = RLMChatCompletion(
            root_model="gpt-4",
            prompt="hi",
            response="see attached",
            usage_summary=UsageSummary(model_usage_summaries={}),
            execution_time=1.0,
            final_artifacts=["report.md"],
            workspace_root=str(tmp_path),
        )
        assert c.read_artifact("report.md") == "hello\n"


class TestWorkspaceIteration:
    """Tests for WorkspaceIteration error field + to_dict shape."""

    def test_error_defaults_to_none_and_serializes(self):
        it = WorkspaceIteration(
            iteration=1,
            timestamp="2026-01-01T00:00:00",
            prompt=[],
            response="hi",
            reasoning=None,
        )
        assert it.error is None
        d = it.to_dict()
        assert d["error"] is None

    def test_error_field_serializes(self):
        it = WorkspaceIteration(
            iteration=2,
            timestamp="2026-01-01T00:00:00",
            prompt=[],
            response="malformed",
            reasoning=None,
            parse_attempts=[
                {"attempt": 1, "response": "x", "error": "no action", "fragment": None}
            ],
            error="Action parse failed after 2 retries: no action",
        )
        d = it.to_dict()
        assert d["error"] == "Action parse failed after 2 retries: no action"
        assert len(d["parse_attempts"]) == 1


class TestQueryMetadata:
    """Tests for QueryMetadata."""

    def test_string_prompt(self):
        meta = QueryMetadata("Hello, world!")
        assert meta.context_type == "str"
        assert meta.context_total_length == 13
        assert meta.context_lengths == [13]


class TestRLMMetadata:
    """Tests for RLMMetadata."""

    def test_to_dict(self):
        meta = RLMMetadata(
            root_model="gpt-4",
            max_depth=2,
            max_iterations=10,
            backend="openai",
            backend_kwargs={"api_key": "secret"},
            action_format="native",
            environment_type="docker",
            environment_kwargs={},
        )
        d = meta.to_dict()
        assert d["root_model"] == "gpt-4"
        assert d["max_depth"] == 2
        assert d["backend"] == "openai"
        assert d["action_format"] == "native"
