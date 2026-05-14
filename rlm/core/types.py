from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Literal

ClientBackend = Literal[
    "openai",
    "portkey",
    "openrouter",
    "vercel",
    "vllm",
    "anthropic",
    "azure_openai",
    "gemini",
]
EnvironmentType = Literal["docker"]


def _serialize_value(value: Any) -> Any:
    """Convert a value to a JSON-serializable representation."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, ModuleType):
        return f"<module '{value.__name__}'>"
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _serialize_value(v) for k, v in value.items()}
    if callable(value):
        return f"<{type(value).__name__} '{getattr(value, '__name__', repr(value))}'>"
    # Try to convert to string for other types
    try:
        return repr(value)
    except Exception:
        return f"<{type(value).__name__}>"


########################################################
########    Types for LM Cost Tracking         #########
########################################################


@dataclass
class ModelUsageSummary:
    total_calls: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost: float | None = None  # Cost in USD, if available from provider

    def to_dict(self):
        result = {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }
        if self.total_cost is not None:
            result["total_cost"] = self.total_cost
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "ModelUsageSummary":
        return cls(
            total_calls=data.get("total_calls"),
            total_input_tokens=data.get("total_input_tokens"),
            total_output_tokens=data.get("total_output_tokens"),
            total_cost=data.get("total_cost"),
        )


@dataclass
class UsageSummary:
    model_usage_summaries: dict[str, ModelUsageSummary]

    @property
    def total_cost(self) -> float | None:
        """Aggregate cost across all models. Returns None if no cost data available."""
        costs = [
            summary.total_cost
            for summary in self.model_usage_summaries.values()
            if summary.total_cost is not None
        ]
        return sum(costs) if costs else None

    @property
    def total_input_tokens(self) -> int:
        """Aggregate input tokens across all models."""
        return sum(summary.total_input_tokens for summary in self.model_usage_summaries.values())

    @property
    def total_output_tokens(self) -> int:
        """Aggregate output tokens across all models."""
        return sum(summary.total_output_tokens for summary in self.model_usage_summaries.values())

    def to_dict(self):
        result = {
            "model_usage_summaries": {
                model: usage_summary.to_dict()
                for model, usage_summary in self.model_usage_summaries.items()
            },
        }
        if self.total_cost is not None:
            result["total_cost"] = self.total_cost
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "UsageSummary":
        return cls(
            model_usage_summaries={
                model: ModelUsageSummary.from_dict(usage_summary)
                for model, usage_summary in data.get("model_usage_summaries", {}).items()
            },
        )


########################################################
########   Types for REPL and RLM Iterations   #########
########################################################
@dataclass
class RLMChatCompletion:
    """Record of a single LLM call made from within the environment."""

    root_model: str
    # Workspace substrate may pass a list of context chunks (one per
    # ``_rlm_query_<N>.txt`` slot); legacy REPL-substrate paths used
    # ``str`` or ``dict[str, Any]``.
    prompt: str | dict[str, Any] | list[Any]
    response: str
    usage_summary: UsageSummary
    execution_time: float
    metadata: dict | None = (
        None  # Full trajectory (run_metadata + iterations) when logger captures it
    )
    # Backend reasoning-channel content (e.g., Anthropic extended thinking, OpenAI
    # reasoning, Gemini thinking). None when the backend does not surface it.
    reasoning_content: str | None = None
    # Return value of the pre-cleanup callback (if any) passed to
    # ``RLM.completion``. Arbitrary type — caller-defined. None when no
    # callback was supplied (or the callback returned None).
    pre_cleanup_result: Any = None
    # Workspace-relative paths the model attached to its ``final`` action.
    # Empty when the model returned the answer inline (the recommended path)
    # or when the run hit ``max_iterations`` without a ``final``.
    final_artifacts: list[str] = field(default_factory=list)
    # Host-absolute path to the workspace directory that produced these
    # artifacts, when it survives past the completion call (i.e.
    # ``DockerConfig.cleanup_mode == "keep"``). ``None`` if the workspace
    # was torn down or its location is otherwise unavailable. Combine with
    # entries of ``final_artifacts`` to read the files directly: e.g.
    # ``Path(result.workspace_root) / result.final_artifacts[0]``.
    workspace_root: str | None = None

    def read_artifact(self, path: str) -> str:
        """Read a final artifact's text contents.

        Convenience for the common case of inspecting a file the model
        attached to its ``final`` action. Resolves ``path`` against
        ``workspace_root``; raises ``RuntimeError`` if the workspace was not
        kept past the completion. For binary data, read ``workspace_root /
        path`` directly with ``pathlib``.
        """
        if self.workspace_root is None:
            raise RuntimeError(
                "Cannot read artifact: workspace_root is None (the workspace "
                "was torn down after the completion). Set "
                'DockerConfig(cleanup_mode="keep") to retain it.'
            )
        from pathlib import Path

        return (Path(self.workspace_root) / path).read_text(encoding="utf-8")

    def to_dict(self):
        out = {
            "root_model": self.root_model,
            "prompt": self.prompt,
            "response": self.response,
            "usage_summary": self.usage_summary.to_dict(),
            "execution_time": self.execution_time,
            "final_artifacts": list(self.final_artifacts),
        }
        if self.workspace_root is not None:
            out["workspace_root"] = self.workspace_root
        if self.metadata is not None:
            out["metadata"] = self.metadata
        if self.reasoning_content is not None:
            out["reasoning_content"] = self.reasoning_content
        if self.pre_cleanup_result is not None:
            out["pre_cleanup_result"] = _serialize_value(self.pre_cleanup_result)
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "RLMChatCompletion":
        return cls(
            root_model=data.get("root_model"),
            prompt=data.get("prompt"),
            response=data.get("response"),
            usage_summary=UsageSummary.from_dict(data.get("usage_summary")),
            execution_time=data.get("execution_time"),
            metadata=data.get("metadata"),
            reasoning_content=data.get("reasoning_content"),
            pre_cleanup_result=data.get("pre_cleanup_result"),
            final_artifacts=list(data.get("final_artifacts") or []),
            workspace_root=data.get("workspace_root"),
        )


########################################################
########   Types for Workspace Substrate       #########
########################################################

# Provenance roles for files in the workspace. ``user`` = pre-existed the run
# (root task, user-supplied context, parent files visible to a child).
# ``assistant`` = direct write by a host-side file tool. ``system`` = touched by
# a script the assistant ran (shell/python). ``child`` = brought back from a
# child RLM via rlm_query artifact selection.
ProvenanceRole = Literal["user", "assistant", "system", "child"]


@dataclass
class LMToolCall:
    """A native model-emitted tool call from an OpenAI-compatible backend."""

    id: str
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
        }


@dataclass
class LMCompletionResult:
    """Assistant response payload, optionally with native tool calls."""

    content: str
    reasoning_content: str | None = None
    tool_calls: list[LMToolCall] = field(default_factory=list)
    # Per-call token usage from the backend's ``usage`` field, when available.
    # Keys: ``prompt_tokens``, ``completion_tokens``, ``total_tokens``. ``None``
    # for backends that don't surface usage on this code path.
    usage: dict[str, int] | None = None
    # Post-chat-template prompt the model actually sees (system+tools envelope
    # + history + generation cursor, as rendered by vLLM's Jinja template).
    # Populated best-effort via ``/tokenize`` + ``/detokenize`` for self-hosted
    # vLLM; ``None`` for public endpoints or on transport error.
    rendered_prompt: str | None = None


@dataclass
class WorkspaceAction:
    """A single workspace tool invocation extracted from an LM response."""

    tool: str
    args: dict[str, Any]
    body: str | None  # element body (for write_file/python/etc.); None for self-closing
    raw: str  # original tag-pair fragment, for replay/debugging
    call_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "tool": self.tool,
            "args": self.args,
            "body": self.body,
            "raw": self.raw,
            "call_id": self.call_id,
        }


@dataclass
class WorkspaceObservation:
    """Result of executing one ``WorkspaceAction``."""

    tool: str
    stdout: str = ""
    stderr: str = ""
    data: dict[str, Any] | None = None
    artifacts: list[str] = field(default_factory=list)
    execution_time: float | None = None
    rlm_calls: list["RLMChatCompletion"] = field(default_factory=list)
    final_answer: str | None = None
    final_artifacts: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "tool": self.tool,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "data": self.data,
            "artifacts": list(self.artifacts),
            "execution_time": self.execution_time,
            "rlm_calls": [c.to_dict() for c in self.rlm_calls],
            "final_answer": self.final_answer,
            "final_artifacts": list(self.final_artifacts),
            "error": self.error,
        }


@dataclass
class WorkspaceSnapshot:
    """Per-turn git snapshot of the workspace."""

    turn: int
    commit_sha: str
    changed_files: list[str]
    workspace_root: str

    def to_dict(self) -> dict:
        return {
            "turn": self.turn,
            "commit_sha": self.commit_sha,
            "changed_files": list(self.changed_files),
            "workspace_root": self.workspace_root,
        }


@dataclass
class WorkspaceIteration:
    """One turn of the workspace RLM loop."""

    iteration: int
    timestamp: str
    prompt: list[dict[str, Any]]  # full message history sent to LM
    response: str  # raw LM response (prose + actions)
    reasoning: str | None  # backend reasoning channel content, if any
    parse_attempts: list[dict[str, Any]] = field(default_factory=list)
    actions: list[WorkspaceAction] = field(default_factory=list)
    observations: list[WorkspaceObservation] = field(default_factory=list)
    snapshot: WorkspaceSnapshot | None = None
    final_answer: str | None = None
    iteration_time: float | None = None
    error: str | None = None  # set when this turn aborted (e.g. parse-retry exhausted)
    # Aggregated token usage for the parent LM call(s) in this turn (summed
    # across parse-retry attempts). Populated for backends that surface
    # ``usage`` on their completion result. Useful for spotting cases where
    # the backend generated many tokens but most were dropped by its parser
    # (e.g. ``completion_tokens`` >> length of ``reasoning + response``).
    lm_usage: dict[str, int] | None = None
    # Post-chat-template prompt the model actually sees on this turn — the
    # system+tools envelope vLLM injects via the chat template plus the
    # full message history with the generation cursor. Best-effort, vLLM-only.
    rendered_prompt: str | None = None

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "timestamp": self.timestamp,
            "prompt": self.prompt,
            "response": self.response,
            "reasoning": self.reasoning,
            "parse_attempts": list(self.parse_attempts),
            "actions": [a.to_dict() for a in self.actions],
            "observations": [o.to_dict() for o in self.observations],
            "snapshot": self.snapshot.to_dict() if self.snapshot else None,
            "final_answer": self.final_answer,
            "iteration_time": self.iteration_time,
            "error": self.error,
            "lm_usage": self.lm_usage,
            "rendered_prompt": self.rendered_prompt,
        }


########################################################
########   Types for RLM Metadata   #########
########################################################


@dataclass
class RLMMetadata:
    """Metadata about the RLM configuration."""

    root_model: str
    max_depth: int
    max_iterations: int
    backend: str
    backend_kwargs: dict[str, Any]
    action_format: str
    environment_type: str
    environment_kwargs: dict[str, Any]
    other_backends: list[str] | None = None

    def to_dict(self):
        return {
            "root_model": self.root_model,
            "max_depth": self.max_depth,
            "max_iterations": self.max_iterations,
            "backend": self.backend,
            "backend_kwargs": {k: _serialize_value(v) for k, v in self.backend_kwargs.items()},
            "action_format": self.action_format,
            "environment_type": self.environment_type,
            "environment_kwargs": {
                k: _serialize_value(v) for k, v in self.environment_kwargs.items()
            },
            "other_backends": self.other_backends,
        }


########################################################
########   Types for RLM Prompting   #########
########################################################


@dataclass
class QueryMetadata:
    context_lengths: list[int]
    context_total_length: int
    context_type: str

    def __init__(self, prompt: str | list[str] | dict[Any, Any] | list[dict[Any, Any]]):
        if isinstance(prompt, str):
            self.context_lengths = [len(prompt)]
            self.context_type = "str"
        elif isinstance(prompt, dict):
            self.context_type = "dict"
            self.context_lengths = []
            for chunk in prompt.values():
                if isinstance(chunk, str):
                    self.context_lengths.append(len(chunk))
                    continue
                try:
                    import json

                    self.context_lengths.append(len(json.dumps(chunk, default=str)))
                except Exception:
                    self.context_lengths.append(len(repr(chunk)))
            self.context_type = "dict"
        elif isinstance(prompt, list):
            self.context_type = "list"
            if len(prompt) == 0:
                self.context_lengths = [0]
            elif isinstance(prompt[0], dict):
                if "content" in prompt[0]:
                    self.context_lengths = [len(str(chunk.get("content", ""))) for chunk in prompt]
                else:
                    self.context_lengths = []
                    for chunk in prompt:
                        try:
                            import json

                            self.context_lengths.append(len(json.dumps(chunk, default=str)))
                        except Exception:
                            self.context_lengths.append(len(repr(chunk)))
            else:
                self.context_lengths = [len(chunk) for chunk in prompt]
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        self.context_total_length = sum(self.context_lengths)
