"""Tests for LMHandler using MockLM (no real LM required)."""

from rlm.core.comms_utils import LMRequest, send_lm_request, send_lm_request_batched
from rlm.core.lm_handler import LMHandler
from tests.mock_lm import MockLM


def test_lm_handler_single_request():
    """Single prompt request returns success and echo-style content."""
    mock = MockLM(responses=["hello back"])
    with LMHandler(client=mock) as handler:
        request = LMRequest(prompt="hello")
        response = send_lm_request(handler.address, request)
    assert response.success
    assert response.chat_completion is not None
    assert response.chat_completion.response == "hello back"


def test_lm_handler_batched_request():
    """Batched prompts return one response per prompt in order."""
    responses = [f"r{i}" for i in range(5)]
    mock = MockLM(responses=responses)
    with LMHandler(client=mock, batch_max_concurrent=3) as handler:
        prompts = [f"prompt-{i}" for i in range(5)]
        result = send_lm_request_batched(handler.address, prompts)
    assert len(result) == 5
    for i, resp in enumerate(result):
        assert resp.success, resp.error
        assert resp.chat_completion is not None
        assert resp.chat_completion.response == f"r{i}"


def test_lm_handler_batched_many_prompts_semaphore_cap():
    """Many prompts complete successfully with semaphore limiting concurrency."""
    # 50 prompts, max 4 concurrent: should still all complete
    count = 50
    responses = [f"resp-{i}" for i in range(count)]
    mock = MockLM(responses=responses)
    with LMHandler(client=mock, batch_max_concurrent=4) as handler:
        prompts = [f"p-{i}" for i in range(count)]
        result = send_lm_request_batched(handler.address, prompts)
    assert len(result) == count
    for i, resp in enumerate(result):
        assert resp.success, (i, resp.error)
        assert resp.chat_completion.response == f"resp-{i}"


def test_lm_handler_batched_first_failure_aborts_entire_batch():
    """Current behavior contract: ``_handle_batched`` uses ``asyncio.gather``
    without ``return_exceptions``, so a single ``acompletion`` raising
    short-circuits the whole batch and returns an error response.

    The recursion layer (``spawn_via_broker_batched``) DOES isolate per-task
    errors, but that's because it uses a ``ThreadPoolExecutor`` and inspects
    each future individually. Lock in the LMHandler's behavior so a future
    refactor is a deliberate decision, not a silent change.
    """

    class FailingLM(MockLM):
        def __init__(self):
            super().__init__()
            self._n = 0

        async def acompletion(self, prompt):
            self._n += 1
            # Second prompt blows up.
            if self._n == 2:
                raise RuntimeError("intentional batch failure")
            return f"ok-{self._n}"

    with LMHandler(client=FailingLM(), batch_max_concurrent=2) as handler:
        # Five prompts; #2 raises. asyncio.gather propagates → batch errors.
        result = send_lm_request_batched(handler.address, [f"p-{i}" for i in range(5)])
    # The protocol wraps an error in a single failed LMResponse per prompt
    # entry; verify that the failure surfaces (not a silent success).
    assert any(not r.success for r in result), (
        "Expected at least one failed LMResponse; got all-success which would "
        "indicate the failing acompletion was silently swallowed."
    )
    failures = [r for r in result if not r.success]
    assert any("intentional batch failure" in (r.error or "") for r in failures), (
        f"Expected the upstream RuntimeError text in the error string; got: "
        f"{[r.error for r in failures]!r}"
    )
