"""Unit tests for eval/common/composite_image.py.

These tests do not require Docker — subprocess and the `requests` smoke
check are mocked. A real end-to-end build is exercised by the Piece-3
runner against an actual base image (and is Docker-gated).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from eval.common import composite_image


class TestBuildComposite:
    def test_uses_cache_when_image_exists(self, tmp_path: Path) -> None:
        with patch.object(composite_image, "_image_exists", return_value=True):
            with patch("subprocess.run") as run:
                tag = composite_image.build_composite(
                    base_image="some-base:latest",
                    output_tag="rlm-test:latest",
                )
        assert tag == "rlm-test:latest"
        # No docker build invocation when cache hits.
        run.assert_not_called()

    def test_skips_cache_when_disabled(self, tmp_path: Path) -> None:
        # Set up real sources so we can drive the build path through to the
        # subprocess call (which we still mock).
        ws_src = tmp_path / "rlm_workspace"
        ws_src.mkdir()
        (ws_src / "__init__.py").write_text("")
        template = tmp_path / "Dockerfile.composite.template"
        template.write_text("FROM ${BASE_IMAGE}\n")

        fake_run = MagicMock(returncode=0, stdout="", stderr="")
        with patch.object(composite_image, "_image_exists", return_value=True):
            with patch("subprocess.run", return_value=fake_run) as run:
                composite_image.build_composite(
                    base_image="some-base:latest",
                    output_tag="rlm-test:latest",
                    cache=False,
                    rlm_workspace_src=ws_src,
                    template_path=template,
                )
        # Should have invoked `docker build` exactly once even though the
        # image existed.
        run.assert_called_once()
        cmd = run.call_args.args[0]
        assert cmd[:2] == ["docker", "build"]
        assert "-t" in cmd
        assert "rlm-test:latest" in cmd

    def test_missing_rlm_workspace_src_raises(self, tmp_path: Path) -> None:
        with patch.object(composite_image, "_image_exists", return_value=False):
            with pytest.raises(FileNotFoundError, match="rlm_workspace package"):
                composite_image.build_composite(
                    base_image="b",
                    output_tag="t",
                    rlm_workspace_src=tmp_path / "missing",
                )

    def test_missing_template_raises(self, tmp_path: Path) -> None:
        ws_src = tmp_path / "rlm_workspace"
        ws_src.mkdir()
        (ws_src / "__init__.py").write_text("")
        with patch.object(composite_image, "_image_exists", return_value=False):
            with pytest.raises(FileNotFoundError, match="Dockerfile template"):
                composite_image.build_composite(
                    base_image="b",
                    output_tag="t",
                    rlm_workspace_src=ws_src,
                    template_path=tmp_path / "missing.template",
                )

    def test_docker_build_failure_raises(self, tmp_path: Path) -> None:
        ws_src = tmp_path / "rlm_workspace"
        ws_src.mkdir()
        (ws_src / "__init__.py").write_text("")
        template = tmp_path / "Dockerfile.composite.template"
        template.write_text("FROM ${BASE_IMAGE}\n")

        fake_run = MagicMock(returncode=1, stdout="oh", stderr="no")
        with patch.object(composite_image, "_image_exists", return_value=False):
            with patch("subprocess.run", return_value=fake_run):
                with pytest.raises(RuntimeError, match="docker build failed"):
                    composite_image.build_composite(
                        base_image="b",
                        output_tag="t",
                        rlm_workspace_src=ws_src,
                        template_path=template,
                    )

    def test_template_substitution_passes_base_image(self, tmp_path: Path) -> None:
        ws_src = tmp_path / "rlm_workspace"
        ws_src.mkdir()
        (ws_src / "__init__.py").write_text("")
        template = tmp_path / "Dockerfile.composite.template"
        template.write_text("FROM ${BASE_IMAGE}\nRUN echo hello\n")

        captured_context: dict = {}

        def capture_run(cmd, **kwargs):
            # When docker build is invoked, the build context is cmd[-1].
            if cmd[:2] == ["docker", "build"]:
                ctx_dir = Path(cmd[-1])
                captured_context["dockerfile"] = (ctx_dir / "Dockerfile").read_text()
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch.object(composite_image, "_image_exists", return_value=False):
            with patch.object(composite_image.subprocess, "run", side_effect=capture_run):
                composite_image.build_composite(
                    base_image="my-base:1.2.3",
                    output_tag="rlm-out:latest",
                    rlm_workspace_src=ws_src,
                    template_path=template,
                )

        assert captured_context["dockerfile"].startswith("FROM my-base:1.2.3")

    def test_default_template_exists(self) -> None:
        """The shipped default template is loadable + has the placeholder."""
        assert composite_image.DEFAULT_TEMPLATE.is_file()
        text = composite_image.DEFAULT_TEMPLATE.read_text()
        assert "${BASE_IMAGE}" in text
        assert "${EXTRA_ENV}" in text
        assert "rlm_workspace" in text  # COPY line present
        # Reuses the shipped rlm_workspace src by default.
        assert composite_image.DEFAULT_RLM_WORKSPACE_SRC.is_dir()

    def test_extra_env_renders_env_lines(self, tmp_path: Path) -> None:
        """``extra_env`` entries become ``ENV K=V`` lines in the rendered
        Dockerfile. Smoke-tests both the placeholder substitution and the
        ``_render_extra_env`` helper.
        """
        ws_src = tmp_path / "rlm_workspace"
        ws_src.mkdir()
        (ws_src / "__init__.py").write_text("")
        template = tmp_path / "Dockerfile.composite.template"
        template.write_text("FROM ${BASE_IMAGE}\n${EXTRA_ENV}\nRUN true\n")

        captured: dict = {}

        def capture_run(cmd, **kwargs):
            if cmd[:2] == ["docker", "build"]:
                captured["dockerfile"] = (Path(cmd[-1]) / "Dockerfile").read_text()
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch.object(composite_image, "_image_exists", return_value=False):
            with patch.object(composite_image.subprocess, "run", side_effect=capture_run):
                composite_image.build_composite(
                    base_image="b",
                    output_tag="t",
                    rlm_workspace_src=ws_src,
                    template_path=template,
                    extra_env={
                        "PATH": "/opt/miniconda3/envs/testbed/bin:/usr/bin",
                        "CONDA_DEFAULT_ENV": "testbed",
                    },
                )

        dockerfile = captured["dockerfile"]
        assert "ENV PATH=/opt/miniconda3/envs/testbed/bin:/usr/bin" in dockerfile
        assert "ENV CONDA_DEFAULT_ENV=testbed" in dockerfile

    def test_extra_env_none_renders_empty_block(self, tmp_path: Path) -> None:
        """When ``extra_env`` is omitted, the placeholder substitutes to the
        empty string — no stray ``ENV`` lines, no Dockerfile errors.
        """
        ws_src = tmp_path / "rlm_workspace"
        ws_src.mkdir()
        (ws_src / "__init__.py").write_text("")
        template = tmp_path / "Dockerfile.composite.template"
        template.write_text("FROM ${BASE_IMAGE}\n${EXTRA_ENV}\nRUN true\n")

        captured: dict = {}

        def capture_run(cmd, **kwargs):
            if cmd[:2] == ["docker", "build"]:
                captured["dockerfile"] = (Path(cmd[-1]) / "Dockerfile").read_text()
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch.object(composite_image, "_image_exists", return_value=False):
            with patch.object(composite_image.subprocess, "run", side_effect=capture_run):
                composite_image.build_composite(
                    base_image="b",
                    output_tag="t",
                    rlm_workspace_src=ws_src,
                    template_path=template,
                )

        dockerfile = captured["dockerfile"]
        # No spurious ENV lines from the placeholder.
        assert "\nENV " not in dockerfile
        # Placeholder itself was substituted away.
        assert "${EXTRA_ENV}" not in dockerfile

    def test_render_extra_env_empty_returns_empty_string(self) -> None:
        assert composite_image._render_extra_env({}) == ""

    def test_render_extra_env_emits_one_env_per_line(self) -> None:
        out = composite_image._render_extra_env({"A": "1", "B": "two words"})
        # One ENV line per entry. Insertion order preserved (Python 3.7+ dict).
        assert out.splitlines() == ["ENV A=1", "ENV B=two words"]

    def test_build_context_includes_rlm_workspace_dir(self, tmp_path: Path) -> None:
        """Beyond rendering the Dockerfile, the build context must contain
        a copy of the ``rlm_workspace/`` package so the ``COPY`` directive
        in the template succeeds. Regression guard against dropping the
        ``shutil.copytree`` call.
        """
        ws_src = tmp_path / "rlm_workspace"
        ws_src.mkdir()
        (ws_src / "__init__.py").write_text("# marker")
        (ws_src / "broker.py").write_text("# broker marker")
        template = tmp_path / "Dockerfile.composite.template"
        template.write_text("FROM ${BASE_IMAGE}\n")

        captured: dict = {}

        def capture_run(cmd, **kwargs):
            if cmd[:2] == ["docker", "build"]:
                ctx_dir = Path(cmd[-1])
                captured["context_listing"] = sorted(p.name for p in ctx_dir.iterdir())
                ws_in_ctx = ctx_dir / "rlm_workspace"
                if ws_in_ctx.is_dir():
                    captured["rlm_workspace_files"] = sorted(p.name for p in ws_in_ctx.iterdir())
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch.object(composite_image, "_image_exists", return_value=False):
            with patch.object(composite_image.subprocess, "run", side_effect=capture_run):
                composite_image.build_composite(
                    base_image="b",
                    output_tag="t",
                    rlm_workspace_src=ws_src,
                    template_path=template,
                )
        assert "Dockerfile" in captured["context_listing"]
        assert "rlm_workspace" in captured["context_listing"]
        assert "broker.py" in captured["rlm_workspace_files"]
        assert "__init__.py" in captured["rlm_workspace_files"]


class TestSmokeTest:
    """Unit tests for ``smoke_test``. Subprocess + requests are mocked; the
    real Docker shell-out is exercised once in the Piece-3 end-to-end runs.
    """

    def _docker_run_proc(self, container_id: str = "abc123") -> MagicMock:
        return MagicMock(returncode=0, stdout=container_id + "\n", stderr="")

    def _docker_port_proc(self, port: int = 54321) -> MagicMock:
        return MagicMock(returncode=0, stdout=f"127.0.0.1:{port}\n", stderr="")

    def _docker_stop_proc(self) -> MagicMock:
        return MagicMock(returncode=0, stdout="", stderr="")

    def test_health_ok_returns_cleanly(self) -> None:
        """Happy path: /health returns 200, container is stopped, no raise."""
        calls: list[str] = []

        def fake_run(cmd, **kwargs):
            sub = cmd[1]
            calls.append(sub)
            if sub == "run":
                return self._docker_run_proc()
            if sub == "port":
                return self._docker_port_proc()
            if sub == "stop":
                return self._docker_stop_proc()
            raise AssertionError(f"unexpected docker {sub}")

        fake_response = MagicMock(status_code=200)
        with patch.object(composite_image.subprocess, "run", side_effect=fake_run):
            with patch.object(composite_image.requests, "get", return_value=fake_response) as get:
                composite_image.smoke_test("rlm-test:latest", timeout_seconds=5)

        assert calls == ["run", "port", "stop"]
        assert get.called

    def test_health_timeout_raises_but_still_stops_container(self) -> None:
        """If /health never returns 200, smoke_test raises RuntimeError —
        AND the container must still be stopped via the ``finally`` block.
        """
        calls: list[str] = []

        def fake_run(cmd, **kwargs):
            sub = cmd[1]
            calls.append(sub)
            if sub == "run":
                return self._docker_run_proc()
            if sub == "port":
                return self._docker_port_proc()
            if sub == "stop":
                return self._docker_stop_proc()
            raise AssertionError(f"unexpected docker {sub}")

        # Simulate /health perpetually refusing connection.
        def fail_get(*args, **kwargs):
            raise ConnectionError("nope")

        with patch.object(composite_image.subprocess, "run", side_effect=fake_run):
            with patch.object(composite_image.requests, "get", side_effect=fail_get):
                with patch.object(composite_image.time, "sleep", lambda _s: None):
                    with pytest.raises(RuntimeError, match="did not return 200"):
                        composite_image.smoke_test("rlm-test:latest", timeout_seconds=1)

        # The finally clause must have invoked docker stop.
        assert "stop" in calls

    def test_health_returns_non_200_eventually_times_out(self) -> None:
        """A broker that serves /health with the wrong status (e.g. 503)
        should also be treated as failure — not a False-positive pass.
        """
        calls: list[str] = []

        def fake_run(cmd, **kwargs):
            sub = cmd[1]
            calls.append(sub)
            if sub == "run":
                return self._docker_run_proc()
            if sub == "port":
                return self._docker_port_proc()
            if sub == "stop":
                return self._docker_stop_proc()
            raise AssertionError(f"unexpected docker {sub}")

        with patch.object(composite_image.subprocess, "run", side_effect=fake_run):
            with patch.object(
                composite_image.requests, "get", return_value=MagicMock(status_code=503)
            ):
                with patch.object(composite_image.time, "sleep", lambda _s: None):
                    with pytest.raises(RuntimeError, match="did not return 200"):
                        composite_image.smoke_test("rlm-test:latest", timeout_seconds=1)
        assert "stop" in calls

    def test_docker_run_failure_propagates(self) -> None:
        """``docker run`` is invoked with ``check=True``; a non-zero exit
        should raise CalledProcessError, and the finally must NOT try to
        stop a container that doesn't exist.
        """

        def fake_run(cmd, **kwargs):
            if cmd[1] == "run":
                raise subprocess.CalledProcessError(
                    returncode=125, cmd=cmd, output="", stderr="no such image"
                )
            raise AssertionError("nothing else should run if 'docker run' failed")

        with patch.object(composite_image.subprocess, "run", side_effect=fake_run):
            with pytest.raises(subprocess.CalledProcessError):
                composite_image.smoke_test("does-not-exist:latest")
