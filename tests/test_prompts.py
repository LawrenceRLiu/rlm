from rlm.core.types import WorkspaceObservation
from rlm.utils.prompts import render_observation


def test_render_observation_shows_small_artifact_lists_in_full() -> None:
    obs = WorkspaceObservation(
        tool="run_shell_command",
        artifacts=["app/a.txt", "app/b.txt"],
    )

    rendered = render_observation("t1.a1", obs)

    assert "artifacts: app/a.txt, app/b.txt" in rendered
    assert "showing" not in rendered


def test_render_observation_caps_large_artifact_lists() -> None:
    artifacts = [f"app/file_{idx}.txt" for idx in range(8)]
    obs = WorkspaceObservation(tool="run_shell_command", artifacts=artifacts)

    rendered = render_observation("t1.a1", obs)

    assert "app/file_0.txt" in rendered
    assert "app/file_4.txt" in rendered
    assert "app/file_5.txt" not in rendered
    assert "showing 5 of 8 paths" in rendered
    assert "3 omitted" in rendered
    assert "Use list_directory/read_file" in rendered
