"""
Workspace substrate configuration.

Single source of truth for all workspace-substrate hyperparameters. No magic
numbers in tool modules — every tunable lives here. Environment variables are
reserved for secrets (API keys); everything else is a dataclass field.

Units policy:
- lines: read_file slice default (v0.1; see TODO below for token-based caps)
- bytes: raw filesystem ops only (copy-on-spawn per-file size cutoff)
- counts: structural caps (retries, list_directory entries, concurrent children)
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ParseConfig:
    """Action parser knobs."""

    max_action_parse_retries: int = 3


@dataclass
class ObservationConfig:
    """Per-call observation shaping."""

    # Per-call observation truncation. Above this size (chars), the call's body
    # is spilled to _rlm_artifacts/_observations/<id>.txt and replaced with a
    # path + summary line.
    max_observation_chars: int = 16_000

    # read_file slice default when start_line/end_line not supplied.
    default_read_file_lines: int = 500

    # TODO(v0.2): token-based caps. Once the host-side observation budget is
    # the binding constraint (rather than developer ergonomics), add
    # `default_read_file_tokens` and let the smaller of (lines, tokens) win.

    max_list_directory_entries: int = 200


@dataclass
class PromptHistoryConfig:
    """Model-facing replay shaping for prior turns.

    The JSONL logger keeps full-fidelity responses/actions/observations. These
    knobs only affect what is replayed back into the LM prompt on later turns.
    """

    # Number of most-recent completed turns whose observations are replayed in
    # full. Older observations become receipts so read_file output does not
    # become permanent transcript memory.
    full_observation_turns: int = 1

    # Python/shell source is often useful for debugging, but if that action
    # changed files it may also contain generated artifacts. Use a smaller cap
    # in that case.
    max_command_body_replay_chars: int = 4_000
    max_mutating_command_body_replay_chars: int = 1_200

    # If python/shell changed files, cap stdout more aggressively in prompt
    # replay. The full stdout remains in the iteration log / spill artifact.
    max_mutating_command_stdout_replay_chars: int = 2_000

    # Optional <note>...</note> intent anchor replayed across turns. Overlong
    # or content-like notes are replaced with an omitted-note receipt.
    # Note: Truncation limits have been set practically infinite per user request.
    max_turn_note_chars: int = 1_000_000
    max_turn_note_lines: int = 1_000_000


@dataclass
class RecursionConfig:
    """rlm_query recursion knobs."""

    max_concurrent_subcalls: int = 5
    copy_on_spawn_max_file_bytes: int = 50 * 1024 * 1024
    copy_on_spawn_excludes: tuple[str, ...] = (
        ".git",
        ".venv",
        "node_modules",
        "__pycache__",
        "_rlm_state/snapshots",
        "_rlm_artifacts/children",
    )


@dataclass
class DockerConfig:
    """Docker workspace knobs."""

    image: str = "rlm-workspace:0.1.0"
    workspace_root_base: str = "~/.rlm/workspaces"
    broker_port: int = 8080
    poll_interval_ms: int = 100
    exec_timeout_seconds: int = 300
    cleanup_mode: Literal["keep", "tar", "delete"] = "keep"


@dataclass
class LMConfig:
    """LM-side behavior knobs applied to OpenAI-compatible clients.

    The substrate's primary purpose is benchmarking reasoning models, so
    ``enable_thinking`` defaults to True — Gemma 4 and Qwen3.x headline
    numbers are thinking-on scores. Pass ``LMConfig(enable_thinking=False)``
    to run an off-thinking baseline. Typos in field names raise ``TypeError``
    at construction, surfacing misconfiguration before any LM call fires.
    """

    enable_thinking: bool = True


@dataclass
class WorkspaceConfig:
    """Composed config tree. Pass to ``RLM(workspace_config=...)``."""

    parse: ParseConfig = field(default_factory=ParseConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    history: PromptHistoryConfig = field(default_factory=PromptHistoryConfig)
    recursion: RecursionConfig = field(default_factory=RecursionConfig)
    docker: DockerConfig = field(default_factory=DockerConfig)
    lm: LMConfig = field(default_factory=LMConfig)
