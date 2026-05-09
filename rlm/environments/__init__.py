"""Environment implementations for the RLM workspace substrate.

The legacy REPL substrates (local, ipython, modal, prime, daytona, e2b, and
the old Docker REPL) have been removed. The only supported environment is
``DockerWorkspaceEnv``.
"""

from rlm.environments.base_workspace import BaseWorkspaceEnv
from rlm.environments.docker_workspace import DockerWorkspaceEnv, ExecResult

__all__ = [
    "BaseWorkspaceEnv",
    "DockerWorkspaceEnv",
    "ExecResult",
]
