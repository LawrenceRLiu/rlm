"""Unit tests for WorkspaceConfig and its sub-configs."""

from rlm.core.config import (
    DockerConfig,
    ObservationConfig,
    ParseConfig,
    RecursionConfig,
    WorkspaceConfig,
)


def test_workspace_config_defaults():
    cfg = WorkspaceConfig()
    assert cfg.parse.max_action_parse_retries == 3
    assert cfg.observation.max_observation_chars == 16_000
    assert cfg.observation.default_read_file_lines == 500
    assert cfg.observation.max_list_directory_entries == 200
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
        recursion=RecursionConfig(max_concurrent_subcalls=2),
        docker=DockerConfig(image="custom:latest", exec_timeout_seconds=60),
    )
    assert cfg.parse.max_action_parse_retries == 7
    assert cfg.observation.max_observation_chars == 32_000
    assert cfg.recursion.max_concurrent_subcalls == 2
    assert cfg.docker.image == "custom:latest"
    assert cfg.docker.exec_timeout_seconds == 60


def test_workspace_config_subconfigs_independent():
    """Each sub-config defaults independently — no shared mutable state."""
    a = WorkspaceConfig()
    b = WorkspaceConfig()
    assert a.parse is not b.parse
    assert a.recursion.copy_on_spawn_excludes == b.recursion.copy_on_spawn_excludes


def test_units_sanity():
    """Lines/bytes/counts are integers in the units the docstring promises."""
    cfg = WorkspaceConfig()
    assert isinstance(cfg.observation.default_read_file_lines, int)
    assert isinstance(cfg.recursion.copy_on_spawn_max_file_bytes, int)
    assert isinstance(cfg.observation.max_list_directory_entries, int)
    # bytes cap is large enough to be unambiguous as bytes (not lines)
    assert cfg.recursion.copy_on_spawn_max_file_bytes > 1024
