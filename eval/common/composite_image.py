"""Composite-image build for benchmark runs.

A *composite image* is a benchmark's per-instance / per-task base image with
the RLM broker layered on top. This module is benchmark-agnostic: the caller
supplies the base image tag and the desired output tag; the layered-on-top
part (uv-installed Python 3.11 for the broker, ``rlm_workspace`` package on
``PYTHONPATH``, ``requests`` in project python, broker as ``CMD``) is shared.

Two functions:

- ``build_composite(base_image, output_tag, ...)`` builds the image.
- ``smoke_test(tag)`` starts a container from the image, hits the broker's
  ``/health`` endpoint, tears the container down.

Both shell out to ``docker``; no Python Docker SDK dep.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from string import Template

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE = Path(__file__).parent / "Dockerfile.composite.template"
DEFAULT_RLM_WORKSPACE_SRC = REPO_ROOT / "docker" / "workspace_image" / "rlm_workspace"


def build_composite(
    base_image: str,
    output_tag: str,
    *,
    cache: bool = True,
    rlm_workspace_src: Path | None = None,
    template_path: Path | None = None,
) -> str:
    """Build ``output_tag`` = ``base_image`` + RLM broker layer.

    Args:
        base_image: tag of an existing base image to layer onto. The caller
            is responsible for resolving / pulling / building this first
            (per-benchmark logic).
        output_tag: tag for the resulting composite image.
        cache: if ``True`` and ``output_tag`` already exists locally, skip
            the rebuild.
        rlm_workspace_src: directory containing the ``rlm_workspace``
            package to ``COPY`` into the image. Defaults to the repo's
            ``docker/workspace_image/rlm_workspace``.
        template_path: Dockerfile template (with ``${BASE_IMAGE}``
            placeholder). Defaults to ``Dockerfile.composite.template``
            next to this file.

    Returns:
        ``output_tag``.

    Raises:
        RuntimeError if ``docker build`` fails.
        FileNotFoundError if ``rlm_workspace_src`` or template is missing.
    """
    rlm_workspace_src = rlm_workspace_src or DEFAULT_RLM_WORKSPACE_SRC
    template_path = template_path or DEFAULT_TEMPLATE

    if cache and _image_exists(output_tag):
        return output_tag

    if not rlm_workspace_src.is_dir():
        raise FileNotFoundError(f"rlm_workspace package not found at {rlm_workspace_src}")
    if not template_path.is_file():
        raise FileNotFoundError(f"Dockerfile template not found at {template_path}")

    dockerfile_text = Template(template_path.read_text(encoding="utf-8")).substitute(
        BASE_IMAGE=base_image
    )

    # Build context = rendered Dockerfile + the rlm_workspace package.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "Dockerfile").write_text(dockerfile_text, encoding="utf-8")
        shutil.copytree(rlm_workspace_src, tmp_path / "rlm_workspace")
        result = subprocess.run(
            ["docker", "build", "-t", output_tag, str(tmp_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"docker build failed for {output_tag}:\n"
                f"--- stdout ---\n{result.stdout}\n"
                f"--- stderr ---\n{result.stderr}"
            )
    return output_tag


def smoke_test(tag: str, *, timeout_seconds: int = 30) -> None:
    """Verify the broker comes up cleanly inside a fresh container.

    Starts ``tag`` with the broker port published to a random host port,
    polls ``GET /health`` until it returns 200 or ``timeout_seconds``
    elapses, then tears the container down. Raises ``RuntimeError`` on
    failure.
    """
    start = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "-p",
            "127.0.0.1::8080",
            tag,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    container_id = start.stdout.strip()
    try:
        port_result = subprocess.run(
            ["docker", "port", container_id, "8080"],
            capture_output=True,
            text=True,
            check=True,
        )
        first_line = port_result.stdout.strip().splitlines()[0]
        host_port = int(first_line.rsplit(":", 1)[1])

        deadline = time.monotonic() + timeout_seconds
        url = f"http://127.0.0.1:{host_port}/health"
        last_err: Exception | None = None
        while time.monotonic() < deadline:
            try:
                r = requests.get(url, timeout=1.0)
                if r.status_code == 200:
                    return
            except Exception as e:
                last_err = e
            time.sleep(0.2)
        raise RuntimeError(
            f"Broker /health did not return 200 within {timeout_seconds}s for {tag}; "
            f"last error: {last_err!r}"
        )
    finally:
        subprocess.run(
            ["docker", "stop", container_id],
            capture_output=True,
            text=True,
        )


def _image_exists(tag: str) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", tag],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0
