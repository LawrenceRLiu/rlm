"""Shared pytest configuration and fixtures.

Markers:
- ``docker``: needs the ``rlm-workspace:0.1.0`` image and a working Docker daemon.
- ``slow``: multi-turn end-to-end rollouts; not run by default in fast mode.

Both markers are registered here so ``pytest --strict-markers`` does not warn.
"""

from __future__ import annotations


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "docker: test requires Docker daemon and the rlm-workspace image"
    )
    config.addinivalue_line(
        "markers", "slow: multi-turn rollout or otherwise slow; opt-in via -m slow"
    )
