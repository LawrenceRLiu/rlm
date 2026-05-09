"""rlm_workspace — in-container support package for the workspace substrate.

Two modules:

- ``broker``: a tiny Flask app that runs as PID 1 inside the workspace container.
  Sandbox code POSTs LM/recursion requests to ``/enqueue``; the host-side poller
  pulls them via ``/pending`` and posts results back via ``/respond``.

- ``client``: pure-Python client functions (``llm_query``, ``llm_query_batched``,
  ``rlm_query``, ``rlm_query_batched``) preimported into ``python`` action
  scripts. Each function POSTs to ``localhost:<broker_port>/enqueue`` and
  blocks until the host poller posts back a response.

Per Decision #27 the client functions take no ``model=`` argument: the parent
RLM's configured model is used everywhere.
"""

from rlm_workspace.client import llm_query, llm_query_batched, rlm_query, rlm_query_batched

__all__ = ["llm_query", "llm_query_batched", "rlm_query", "rlm_query_batched"]
