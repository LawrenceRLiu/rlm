"""Unit tests for WorkspaceConfig and its sub-configs."""

from rlm.core.config import (
    DockerConfig,
    ObservationConfig,
    ParseConfig,
    PromptHistoryConfig,
    RecursionConfig,
    WorkspaceConfig,
)


def test_workspace_config_defaults():
    cfg = WorkspaceConfig()
    assert cfg.parse.max_action_parse_retries == 3
    assert cfg.observation.max_observation_chars == 16_000
    assert cfg.observation.default_read_file_lines == 500
    assert cfg.observation.max_list_directory_entries == 200
    assert cfg.history.full_observation_turns == 1
    assert cfg.history.max_command_body_replay_chars == 4_000
    assert cfg.history.max_turn_note_chars == 600
    assert cfg.history.max_turn_note_lines == 6
    assert cfg.recursion.max_concurrent_subcalls == 5
    assert cfg.recursion.copy_on_spawn_max_file_bytes == 50 * 1024 * 1024
    assert ".git" in cfg.recursion.copy_on_spawn_excludes
    assert cfg.docker.image == "rlm-workspace:0.1.0"
    assert cfg.docker.broker_port == 8080
    assert cfg.docker.exec_timeout_seconds == 300
    assert cfg.docker.cleanup_mode == "keep"


def test_workspace_config_overrides():
    cfg = WorkspaceConfig(
        parse=ParseConfig(max_action_parse_retries=7),
        observation=ObservationConfig(max_observation_chars=32_000),
        history=PromptHistoryConfig(full_observation_turns=2, max_turn_note_chars=300),
        recursion=RecursionConfig(max_concurrent_subcalls=2),
        docker=DockerConfig(image="custom:latest", exec_timeout_seconds=60),
    )
    assert cfg.parse.max_action_parse_retries == 7
    assert cfg.observation.max_observation_chars == 32_000
    assert cfg.history.full_observation_turns == 2
    assert cfg.history.max_turn_note_chars == 300
    assert cfg.recursion.max_concurrent_subcalls == 2
    assert cfg.docker.image == "custom:latest"
    assert cfg.docker.exec_timeout_seconds == 60


def test_workspace_config_subconfigs_independent():
    """Each sub-config defaults independently — no shared mutable state."""
    a = WorkspaceConfig()
    b = WorkspaceConfig()
    assert a.parse is not b.parse
    assert a.history is not b.history
    assert a.recursion.copy_on_spawn_excludes == b.recursion.copy_on_spawn_excludes


def test_units_sanity():
    """Lines/bytes/counts are integers in the units the docstring promises."""
    cfg = WorkspaceConfig()
    assert isinstance(cfg.observation.default_read_file_lines, int)
    assert isinstance(cfg.recursion.copy_on_spawn_max_file_bytes, int)
    assert isinstance(cfg.observation.max_list_directory_entries, int)
    assert isinstance(cfg.history.max_command_body_replay_chars, int)
    # bytes cap is large enough to be unambiguous as bytes (not lines)
    assert cfg.recursion.copy_on_spawn_max_file_bytes > 1024


# ---------------------------------------------------------------------------
# Design-doc default values (locked-in constants)
# ---------------------------------------------------------------------------


def test_default_observation_cap_matches_design_doc():
    """``observation.max_observation_chars`` defaults to 16 KB per
    workspace_substrate_arch/04_turn_and_recursion.md. Changing this default
    affects every prior log file's spill behavior, so it requires explicit
    coordination with the visualizer and is locked in here."""
    cfg = WorkspaceConfig()
    assert cfg.observation.max_observation_chars == 16_000


def test_default_recursion_excludes_cover_design_doc():
    """The copy-on-spawn excludes set must include every name listed in
    workspace_substrate_arch/04_turn_and_recursion.md as "do not copy"."""
    cfg = WorkspaceConfig()
    excludes = set(cfg.recursion.copy_on_spawn_excludes)
    for must in (
        ".git",
        ".venv",
        "node_modules",
        "__pycache__",
        "_rlm_state/snapshots",
        "_rlm_artifacts/children",
    ):
        assert must in excludes, f"copy-on-spawn excludes missing {must!r}"


def test_default_max_concurrent_subcalls_is_five():
    """Design-doc default is 5; locked in so a regression doesn't slip past."""
    cfg = WorkspaceConfig()
    assert cfg.recursion.max_concurrent_subcalls == 5


# ---------------------------------------------------------------------------
# Sub-config replacement preserves other defaults
# ---------------------------------------------------------------------------


def test_replacing_one_subconfig_does_not_disturb_others():
    """Passing a fresh ``DockerConfig`` must not affect parse/observation/recursion."""
    cfg = WorkspaceConfig(docker=DockerConfig(image="my:tag"))
    # Docker overridden:
    assert cfg.docker.image == "my:tag"
    # Others fall back to defaults:
    assert cfg.parse.max_action_parse_retries == 3
    assert cfg.observation.max_observation_chars == 16_000
    assert cfg.history.full_observation_turns == 1
    assert cfg.recursion.max_concurrent_subcalls == 5


def test_subconfig_excludes_tuple_is_immutable_type():
    """``copy_on_spawn_excludes`` is a tuple so it cannot be silently mutated
    by a caller and become a per-instance leak vector."""
    cfg = WorkspaceConfig()
    assert isinstance(cfg.recursion.copy_on_spawn_excludes, tuple)


def test_cleanup_mode_literal_default_is_keep():
    """The Literal type's only safe default is one of ('keep','tar','delete');
    we assert the actual default is 'keep' (developers expect the workspace to
    survive for inspection unless they opted out)."""
    assert WorkspaceConfig().docker.cleanup_mode == "keep"
