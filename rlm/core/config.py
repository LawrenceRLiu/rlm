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

import warnings
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ParseConfig:
    """Action parser knobs."""

    max_action_parse_retries: int = 3
    action_format: Literal["xml", "native"] = "native"
    native_tool_choice: Literal["auto", "required"] = "auto"

    def __post_init__(self) -> None:
        if self.action_format == "xml":
            warnings.warn(
                "ParseConfig(action_format='xml') is deprecated and is no longer "
                "the default. Use action_format='native' with a vLLM/OpenAI-compatible "
                "tool parser for new runs.",
                DeprecationWarning,
                stacklevel=2,
            )


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
class CompactionConfig:
    """Substrate-level history compression.

    When the rendered prompt exceeds ``threshold_tokens``, the substrate calls
    the LM with a summary prompt and resets the model-facing history to
    ``[system, initial_user, assistant=summary, user=continue]`` (plus an
    optional tail of the most recent turns). The full pre-compress trajectory
    remains accessible via ``_rlm_state`` git snapshots and ``provenance.json``.

    This replaces the older age-based observation compaction, per-turn body
    caps, and ``<note>`` machinery: until the threshold fires, action bodies
    and observations stay full-fidelity in the prompt.
    """

    enabled: bool = True
    # Absolute token threshold, not a fraction of the model's context window.
    # Defaults to ~64K — roughly an order of magnitude below Qwen3.5-9B's
    # 262K native context, calibrated to where context-rot literature places
    # the effective reasoning window for a 9B-class model.
    threshold_tokens: int = 64_000
    # Keep this many most-recent completed turns full-fidelity after compress.
    # 0 matches upstream RLM's [system, initial, summary, continue] exactly.
    tail_turns_preserved: int = 0


@dataclass
class LoopGuardConfig:
    """Trajectory-level nudges that reduce degenerate loops without gating tools."""

    # When enabled, if the model repeats the exact same action batch and gets
    # the same observations for ``repeated_action_warning_threshold``
    # consecutive turns without meaningful workspace changes, the substrate
    # appends a short warning to the next prompt. Actions are not blocked.
    stutter_warning_enabled: bool = True
    repeated_action_warning_threshold: int = 2
    # Paths with these prefixes are substrate bookkeeping, not task progress.
    stutter_ignored_change_prefixes: tuple[str, ...] = ("_rlm_state/",)


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
    """Docker workspace knobs.

    Sibling-layout invariants
    -------------------------
    The workspace bind-mounts four host subdirs into the container at root:

      ``<workspace_root>/app/``             → ``/app``
      ``<workspace_root>/_rlm_state/``      → ``/_rlm_state``
      ``<workspace_root>/_rlm_artifacts/``  → ``/_rlm_artifacts``
      ``<workspace_root>/_rlm_notes/``      → ``/_rlm_notes``

    ``/app`` is seeded from the image's baked ``/app`` contents on startup
    (image-seed dance: ``docker create`` + ``docker cp``) so grader-expected
    files survive the bind mount. This layout exists so Terminal-Bench
    workloads — which assume work happens in ``/app`` — work without
    per-tool path translation.
    """

    image: str = "rlm-workspace:0.1.0"
    workspace_root_base: str = "~/.rlm/workspaces"
    broker_port: int = 8080
    poll_interval_ms: int = 100
    exec_timeout_seconds: int = 300
    cleanup_mode: Literal["keep", "tar", "delete"] = "keep"
    # Default working directory inside the container for shell/python exec.
    # Defaults to ``/`` so paths are uniform across all tools (workspace-rooted
    # / container-absolute under the bind-mount roots). Tools accept an
    # optional ``cwd`` arg to override per-call.
    container_cwd: str = "/"
    # PYTHONPATH forced on every ``python`` action so imports rooted at the
    # task directory just work without explicit sys.path mangling.
    container_pythonpath: str = "/app"
    # Whether the workspace container may initiate outbound internet
    # connections. When false, DockerWorkspaceEnv installs a host firewall
    # rule that rejects new forwarded packets from this container while
    # keeping the broker's host-published control port reachable.
    allow_internet: bool = True


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
    compaction: CompactionConfig = field(default_factory=CompactionConfig)
    loop_guard: LoopGuardConfig = field(default_factory=LoopGuardConfig)
    recursion: RecursionConfig = field(default_factory=RecursionConfig)
    docker: DockerConfig = field(default_factory=DockerConfig)
    lm: LMConfig = field(default_factory=LMConfig)
