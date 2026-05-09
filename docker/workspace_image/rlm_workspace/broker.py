"""Container-side HTTP broker for the workspace substrate.

Runs as PID 1 inside the workspace container. Brokers requests between
sandbox-side Python (``rlm_workspace.client``) and the host-side poller in
``DockerWorkspaceEnv``:

  sandbox code  --POST /enqueue-->  broker  <--GET /pending-- host poller
                <-- response  -- broker  <--POST /respond-- host poller

The broker is intentionally minimal:

- No persistence — requests live in memory until answered.
- No auth — bound to ``0.0.0.0`` inside the container; the container only
  exposes the port to the host (see Dockerfile / docker run).
- Each request gets a UUID; ``/enqueue`` blocks the caller on a
  ``threading.Event`` keyed by that UUID until ``/respond`` fires it.

Endpoints
---------
- ``POST /enqueue``  body: ``{"kind": "llm_query"|"llm_query_batched"|
  "rlm_query"|"rlm_query_batched", "prompt": str | "prompts": [str]}``.
  Returns the response payload once the host has answered.
- ``GET /pending``   returns ``{"requests": [{"id", "kind", "prompt"|"prompts"}, ...]}``
  and atomically marks each returned request as in-flight (so subsequent polls
  do not return the same request).
- ``POST /respond``  body: ``{"id": str, "response": str | "responses": [str]
  | "error": str}``. Wakes the corresponding ``/enqueue`` caller.
- ``GET /health``    liveness check.

Configuration
-------------
- ``RLM_BROKER_PORT`` (default 8080): port the Flask app binds to.
- ``RLM_BROKER_HOST`` (default ``0.0.0.0``): host to bind.
"""

from __future__ import annotations

import os
import threading
import uuid
from typing import Any

from flask import Flask, jsonify, request

app = Flask(__name__)

_VALID_KINDS = {"llm_query", "llm_query_batched", "rlm_query", "rlm_query_batched"}

_lock = threading.Lock()
_pending: dict[str, dict[str, Any]] = {}  # id -> {kind, prompt|prompts}
_inflight: set[str] = set()  # ids that have been handed to a poller
_events: dict[str, threading.Event] = {}  # id -> event signalled by /respond
_responses: dict[str, dict[str, Any]] = {}  # id -> response payload


@app.get("/health")
def health() -> Any:
    with _lock:
        return jsonify(
            {
                "status": "ok",
                "pending": len(_pending),
                "inflight": len(_inflight),
            }
        )


@app.post("/enqueue")
def enqueue() -> Any:
    body = request.get_json(silent=True) or {}
    kind = body.get("kind")
    if kind not in _VALID_KINDS:
        return jsonify({"error": f"Invalid 'kind': {kind!r}. Expected one of {_VALID_KINDS}."}), 400

    if kind in {"llm_query", "rlm_query"}:
        prompt = body.get("prompt")
        if not isinstance(prompt, str):
            return jsonify({"error": "Missing or non-string 'prompt'."}), 400
        payload: dict[str, Any] = {"kind": kind, "prompt": prompt}
    else:
        prompts = body.get("prompts")
        if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
            return jsonify({"error": "Missing or non-list-of-str 'prompts'."}), 400
        payload = {"kind": kind, "prompts": prompts}

    req_id = uuid.uuid4().hex
    event = threading.Event()
    with _lock:
        _pending[req_id] = payload
        _events[req_id] = event

    event.wait()  # blocks until /respond fires it; no timeout — host owns retry

    with _lock:
        response = _responses.pop(req_id, None)
        _events.pop(req_id, None)
        _inflight.discard(req_id)
    if response is None:
        return jsonify({"error": "Internal broker error: no response stored."}), 500
    return jsonify(response)


@app.get("/pending")
def pending() -> Any:
    with _lock:
        out = []
        for req_id, payload in _pending.items():
            if req_id in _inflight:
                continue
            _inflight.add(req_id)
            out.append({"id": req_id, **payload})
        # Pending dict retains entries until /respond pops them implicitly via
        # the in-flight set; we don't remove from _pending here so a duplicate
        # poll after a host crash can be observed (operator visibility).
    return jsonify({"requests": out})


@app.post("/respond")
def respond() -> Any:
    body = request.get_json(silent=True) or {}
    req_id = body.get("id")
    if not isinstance(req_id, str):
        return jsonify({"error": "Missing 'id'."}), 400

    with _lock:
        if req_id not in _events:
            return jsonify({"error": f"Unknown request id: {req_id}"}), 404
        # Strip 'id' so the response payload returned to the caller of /enqueue
        # is the raw {response|responses|error} body.
        response_payload = {k: v for k, v in body.items() if k != "id"}
        _responses[req_id] = response_payload
        _pending.pop(req_id, None)
        event = _events[req_id]
    event.set()
    return jsonify({"ok": True})


def main() -> None:
    host = os.environ.get("RLM_BROKER_HOST", "0.0.0.0")
    port = int(os.environ.get("RLM_BROKER_PORT", "8080"))
    # threaded=True so /enqueue blocking on event.wait() doesn't starve
    # /pending and /respond.
    app.run(host=host, port=port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
