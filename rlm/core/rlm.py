"""
Recursive Language Model — workspace substrate driver.

The ``RLM`` class drives a turn-based loop over a ``DockerWorkspaceEnv``.
Each turn:

1. Prompt the LM with the running message history.
2. Parse the response for ``<action>`` blocks (with retry on parse failure,
   capped by ``workspace_config.parse.max_action_parse_retries``).
3. Dispatch each action through the env's tool registry. Read-only tool
   failures do not halt the turn; mutating tool failures halt the rest of
   the batch, including ``final`` (so the model can't commit an answer in
   the same batch as a mutating sibling whose failure would invalidate it).
4. Take a per-turn git snapshot of the workspace.
5. Append a ``WorkspaceIteration`` to the logger and to the message history.
6. If any observation carries a ``final_answer``, return it.

Stop conditions: max_iterations, max_budget, max_timeout, max_tokens,
max_errors, ``ActionParseError`` after retry exhaustion, or a ``final``
action.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from rlm.clients import BaseLM, get_client
from rlm.core.config import WorkspaceConfig
from rlm.core.lm_handler import LMHandler
from rlm.core.types import (
    ClientBackend,
    RLMChatCompletion,
    RLMMetadata,
    UsageSummary,
    WorkspaceAction,
    WorkspaceIteration,
    WorkspaceObservation,
)
from rlm.environments.docker_workspace import DockerWorkspaceEnv
from rlm.logger import RLMLogger, VerbosePrinter
from rlm.utils import action_parser
from rlm.utils.exceptions import (
    ActionParseError,
    BudgetExceededError,
    CancellationError,
    ErrorThresholdExceededError,
    TimeoutExceededError,
    TokenLimitExceededError,
)
from rlm.utils.prompts import (
    build_parse_retry_message,
    build_workspace_initial_user_prompt,
    build_workspace_system_prompt,
    format_workspace_iteration,
)
from rlm.utils.rlm_utils import filter_sensitive_keys
from rlm.workspace_tools import get_spec


class RLM:
    """Workspace-substrate Recursive Language Model.

    A single ``completion()`` call provisions a workspace + container, drives
    the loop to a final answer (or to a stop condition), then cleans up.
    """

    def __init__(
        self,
        backend: ClientBackend = "openai",
        backend_kwargs: dict[str, Any] | None = None,
        workspace_config: WorkspaceConfig | None = None,
        depth: int = 0,
        max_depth: int = 1,
        max_iterations: int = 30,
        max_budget: float | None = None,
        max_timeout: float | None = None,
        max_tokens: int | None = None,
        max_errors: int | None = None,
        custom_system_prompt: str | None = None,
        logger: RLMLogger | None = None,
        verbose: bool = False,
        on_iteration_start: Callable[[int, int], None] | None = None,
        on_iteration_complete: Callable[[int, int, float], None] | None = None,
    ) -> None:
        """
        Args:
            backend: LM client backend identifier (e.g., "openai", "anthropic").
            backend_kwargs: kwargs forwarded to ``get_client``.
            workspace_config: ``WorkspaceConfig`` with parser/observation/
                recursion/docker sub-configs. Defaults to ``WorkspaceConfig()``.
            depth: Current recursion depth (0 for the root RLM).
            max_depth: Maximum recursion depth. At ``depth >= max_depth``
                the ``rlm_query`` tool is omitted from the prompt and any
                emitted call returns a loud error observation.
            max_iterations: Max turns before the loop forces a final answer.
            max_budget: Optional USD cap; raises ``BudgetExceededError``.
            max_timeout: Optional wallclock cap (seconds).
            max_tokens: Optional total-token cap (input + output).
            max_errors: Optional consecutive-error cap.
            custom_system_prompt: Replace the default system prompt entirely.
            logger: ``RLMLogger`` instance (memory + optional JSONL on disk).
            verbose: Toggle the ``VerbosePrinter`` rich console output.
            on_iteration_start / on_iteration_complete: Callbacks for live
                tree displays. ``(depth, iteration_num)`` and
                ``(depth, iteration_num, duration)`` respectively.
        """
        self.backend: ClientBackend = backend
        self.backend_kwargs = backend_kwargs
        self.workspace_config = workspace_config or WorkspaceConfig()

        self.depth = depth
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.max_budget = max_budget
        self.max_timeout = max_timeout
        self.max_tokens = max_tokens
        self.max_errors = max_errors
        self.custom_system_prompt = custom_system_prompt

        self.logger = logger
        self.verbose = VerbosePrinter(enabled=verbose)
        self.on_iteration_start = on_iteration_start
        self.on_iteration_complete = on_iteration_complete

        # Per-completion runtime state (reset in completion()).
        self._cumulative_cost: float = 0.0
        self._consecutive_errors: int = 0
        self._last_error: str | None = None
        self._best_partial_answer: str | None = None
        self._completion_start_time: float | None = None
        # Captured at the moment a `final` action fires; read by the parent's
        # RecursionHandler to selectively pull artifacts back into its workspace.
        self._last_final_artifacts: list[str] = []

        # Log run metadata once if a logger / verbose is attached.
        if self.logger or verbose:
            metadata = RLMMetadata(
                root_model=(backend_kwargs or {}).get("model_name", "unknown"),
                max_depth=max_depth,
                max_iterations=max_iterations,
                backend=backend,
                backend_kwargs=filter_sensitive_keys(backend_kwargs) if backend_kwargs else {},
                environment_type="docker",
                environment_kwargs={
                    "image": self.workspace_config.docker.image,
                    "cleanup_mode": self.workspace_config.docker.cleanup_mode,
                },
            )
            if self.logger:
                self.logger.log_metadata(metadata)
            self.verbose.print_metadata(metadata)

    # =========================================================================
    # Resource management
    # =========================================================================

    @contextmanager
    def _spawn_completion_context(self, prompt: str | dict[str, Any] | list[Any]):
        """Bring up an LM handler + workspace env for a single completion."""
        client: BaseLM = get_client(self.backend, self.backend_kwargs or {})
        lm_handler = LMHandler(client)
        lm_handler.start()

        env = DockerWorkspaceEnv(
            workspace_config=self.workspace_config,
            lm_handler_address=(lm_handler.host, lm_handler.port),
            depth=self.depth,
            max_depth=self.max_depth,
        )
        try:
            env.setup()
            env.load_context(prompt)
            yield lm_handler, env
        finally:
            try:
                env.cleanup()
            finally:
                lm_handler.stop()

    # =========================================================================
    # Public entry point
    # =========================================================================

    def completion(
        self,
        prompt: str | dict[str, Any] | list[Any],
        root_prompt: str | None = None,
        pre_cleanup_callback: Callable[[DockerWorkspaceEnv], Any] | None = None,
    ) -> RLMChatCompletion:
        """Run the workspace loop on ``prompt`` until a final answer or a stop
        condition. Returns a single ``RLMChatCompletion`` whose
        ``response`` is the final answer string.

        ``pre_cleanup_callback`` (optional): a one-shot hook that runs after
        the agent loop returns cleanly, but **before** the container is
        torn down. It receives the still-live ``DockerWorkspaceEnv`` and its
        return value is attached to the result as ``pre_cleanup_result``.
        Use this to run external graders (benchmark harnesses), extract
        artifacts from the live container, etc.

        When the callback runs (and does not):

        - **Fires** when the loop produces a ``final`` action.
        - **Fires** when the loop returns after exhausting ``max_iterations``
          (the default-answer fallback path).
        - **Does NOT fire** when the loop raises — e.g.,
          ``ActionParseError`` after retry exhaustion, ``CancellationError``
          on Ctrl+C, ``BudgetExceededError`` / ``TimeoutExceededError`` /
          ``ErrorThresholdExceededError`` / ``TokenLimitExceededError``.
          The container is still torn down (the context manager's
          ``finally`` runs ``env.cleanup()``), but the grader is skipped.
          Rationale: a crashed agent's workspace is not in a state we trust
          to grade. Callers who want exception-time post-processing should
          wrap their own ``try/finally`` around ``completion()``.

        Limitations of the callback:

        - **One-shot only.** Fires at most once after the loop ends. No
          per-turn or streaming variant.
        - **Synchronous.** Cleanup waits on the callback to return.
          Long-running graders extend container lifetime.
        - **Not for iterative / oracle-feedback benchmarks.** A protocol of
          ``agent → grade → agent revises → re-grade`` needs the env to
          persist across multiple ``completion()`` calls; the callback is
          the wrong shape for that. (Externalizing env construction would
          be the right fix; not implemented.)
        - **Not for post-completion notebook inspection.** Same reason —
          env is destroyed after callback returns.
        - **Outer completion only.** Child ``rlm_query`` completions go
          through ``_run_loop`` directly, not the public ``completion``
          entry, so the callback is not propagated to them.
        - **Exception propagation.** A callback exception does not abort
          cleanup (the container still gets torn down), but the exception
          propagates to the caller of ``completion`` after cleanup runs.
        """
        with self._spawn_completion_context(prompt) as (lm_handler, env):
            self._wire_recursion(env=env, lm_handler=lm_handler)
            result = self._run_loop(
                prompt=prompt,
                root_prompt=root_prompt,
                lm_handler=lm_handler,
                env=env,
            )
            if pre_cleanup_callback is not None:
                # If this raises, the enclosing context manager's finally
                # still runs env.cleanup(); the exception then propagates
                # to the caller (after cleanup has completed).
                result.pre_cleanup_result = pre_cleanup_callback(env)
            return result

    def _wire_recursion(self, *, env: DockerWorkspaceEnv, lm_handler: LMHandler) -> None:
        """Attach a ``RecursionHandler`` if this RLM is allowed to recurse.

        At ``depth >= max_depth`` the handler is left unset so the action
        dispatch + broker poller both fall through to the loud "max depth
        reached" error path.
        """
        if self.depth >= self.max_depth:
            return
        # Local import to break the rlm <-> recursion <-> docker_workspace
        # circular import cycle.
        from rlm.core.recursion import RecursionHandler

        env.recursion_handler = RecursionHandler(
            parent_rlm=self, parent_env=env, lm_handler=lm_handler
        )

    def _run_loop(
        self,
        *,
        prompt: str | dict[str, Any] | list[Any],
        root_prompt: str | None,
        lm_handler: LMHandler,
        env: DockerWorkspaceEnv,
    ) -> RLMChatCompletion:
        """Turn-based workspace loop. Reused by the public ``completion()``
        entry and by ``RecursionHandler`` (which provisions ``lm_handler`` /
        ``env`` itself for child runs).
        """
        time_start = time.perf_counter()
        self._completion_start_time = time_start

        # Reset per-completion state.
        self._consecutive_errors = 0
        self._last_error = None
        self._best_partial_answer = None
        self._cumulative_cost = 0.0
        self._last_final_artifacts = []
        if self.logger:
            self.logger.clear_iterations()

        system_prompt = build_workspace_system_prompt(
            depth=self.depth,
            max_depth=self.max_depth,
            custom_system_prompt=self.custom_system_prompt,
        )
        initial_user = build_workspace_initial_user_prompt(root_prompt=root_prompt)
        message_history: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": initial_user},
        ]

        try:
            for i in range(self.max_iterations):
                self._check_timeout(i, time_start)
                if self.on_iteration_start:
                    try:
                        self.on_iteration_start(self.depth, i + 1)
                    except Exception:
                        pass

                env.current_turn = i + 1
                try:
                    iteration = self._completion_turn(
                        iteration_idx=i + 1,
                        message_history=message_history,
                        lm_handler=lm_handler,
                        env=env,
                    )
                except ActionParseError as exc:
                    # _completion_turn attaches a partial iteration so the
                    # failed turn (with all parse_attempts) is still logged.
                    partial = getattr(exc, "iteration", None)
                    if partial is not None and self.logger:
                        self.logger.log_iteration(partial)
                    raise

                self._check_iteration_limits(iteration, i, lm_handler)

                # Surface a best-partial-answer for graceful stop returns.
                if iteration.response and iteration.response.strip():
                    self._best_partial_answer = iteration.response

                if self.logger:
                    self.logger.log_iteration(iteration)
                if self.on_iteration_complete and iteration.iteration_time is not None:
                    try:
                        self.on_iteration_complete(self.depth, i + 1, iteration.iteration_time)
                    except Exception:
                        pass

                if iteration.final_answer is not None:
                    self._last_final_artifacts = _final_artifacts_from_iteration(iteration)
                    return self._build_completion(
                        prompt=prompt,
                        response=iteration.final_answer,
                        lm_handler=lm_handler,
                        time_start=time_start,
                    )

                # Append turn record to history for the next prompt.
                message_history.extend(format_workspace_iteration(iteration))

        except KeyboardInterrupt:
            self.verbose.print_limit_exceeded("cancelled", "User interrupted execution")
            raise CancellationError(
                partial_answer=self._best_partial_answer,
                message="Execution cancelled by user (Ctrl+C)",
            ) from None

        # max_iterations reached without a `final` action: ask the model
        # one last time for an answer based on what it has gathered.
        final_answer = self._default_answer(message_history, lm_handler)
        return self._build_completion(
            prompt=prompt,
            response=final_answer,
            lm_handler=lm_handler,
            time_start=time_start,
        )

    # =========================================================================
    # One turn
    # =========================================================================

    def _completion_turn(
        self,
        *,
        iteration_idx: int,
        message_history: list[dict[str, Any]],
        lm_handler: LMHandler,
        env: DockerWorkspaceEnv,
    ) -> WorkspaceIteration:
        """One turn of the workspace loop: parse-and-retry → dispatch → snapshot."""
        iter_start = time.perf_counter()
        timestamp = datetime.now().isoformat()

        try:
            response, reasoning, actions, parse_attempts = self._call_lm_with_parse_retry(
                lm_handler=lm_handler,
                messages=list(message_history),
            )
        except ActionParseError as exc:
            # Build a partial iteration so the failed turn is visible in the
            # log. The exception's `parse_attempts`/`last_response`/
            # `last_reasoning` attributes are populated by
            # `_call_lm_with_parse_retry` before raising.
            partial = WorkspaceIteration(
                iteration=iteration_idx,
                timestamp=timestamp,
                prompt=list(message_history),
                response=getattr(exc, "last_response", ""),
                reasoning=getattr(exc, "last_reasoning", None),
                parse_attempts=getattr(exc, "parse_attempts", []),
                actions=[],
                observations=[],
                snapshot=None,
                final_answer=None,
                iteration_time=time.perf_counter() - iter_start,
                error=str(exc),
            )
            exc.iteration = partial  # type: ignore[attr-defined]
            raise

        observations = self._dispatch_actions(env=env, actions=actions)
        snapshot = env.snapshot(turn=iteration_idx)

        final_answer: str | None = None
        for obs in observations:
            if obs.final_answer is not None:
                final_answer = obs.final_answer
                break

        return WorkspaceIteration(
            iteration=iteration_idx,
            timestamp=timestamp,
            prompt=list(message_history),
            response=response,
            reasoning=reasoning,
            parse_attempts=parse_attempts,
            actions=actions,
            observations=observations,
            snapshot=snapshot,
            final_answer=final_answer,
            iteration_time=time.perf_counter() - iter_start,
        )

    def _call_lm_with_parse_retry(
        self,
        *,
        lm_handler: LMHandler,
        messages: list[dict[str, Any]],
    ) -> tuple[str, str | None, list[WorkspaceAction], list[dict[str, Any]]]:
        """Call the LM until a parseable response is produced or retries exhaust.

        Each retry appends a synthetic user feedback message to a *retry-only*
        copy of ``messages``; the main message history is unchanged. Returns
        ``(response, reasoning, actions, parse_attempts)`` for the iteration
        record. Raises ``ActionParseError`` if all retries fail.
        """
        retry_budget = self.workspace_config.parse.max_action_parse_retries
        attempts: list[dict[str, Any]] = []
        retry_messages = list(messages)

        for attempt in range(retry_budget + 1):  # +1 = initial try
            response, reasoning = lm_handler.completion_with_reasoning(retry_messages)
            try:
                actions = action_parser.parse(response)
                return response, reasoning, actions, attempts
            except ActionParseError as exc:
                attempts.append(
                    {
                        "attempt": attempt + 1,
                        "response": response,
                        "error": str(exc),
                        "fragment": exc.fragment,
                    }
                )
                if attempt >= retry_budget:
                    exc.args = (f"Action parse failed after {retry_budget} retries: {exc.args[0]}",)
                    # Surface the per-attempt history + last raw response so
                    # `_completion_turn` can build a partial iteration record.
                    exc.parse_attempts = list(attempts)  # type: ignore[attr-defined]
                    exc.last_response = response  # type: ignore[attr-defined]
                    exc.last_reasoning = reasoning  # type: ignore[attr-defined]
                    raise

                feedback = build_parse_retry_message(str(exc), exc.fragment)
                retry_messages = retry_messages + [
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": feedback},
                ]

        raise ActionParseError(  # pragma: no cover  — defensive
            "Internal error in parse-and-retry loop"
        )

    def _dispatch_actions(
        self,
        *,
        env: DockerWorkspaceEnv,
        actions: list[WorkspaceAction],
    ) -> list[WorkspaceObservation]:
        """Run actions in order. Read-only failures don't halt; mutating do.

        Once a mutating action errors, subsequent mutating actions are
        skipped (replaced with an explicit "halted" observation) but
        read-only actions continue. ``final`` is also skipped after a halt
        even though it is not state-mutating — committing an answer on a
        batch whose mutating sibling failed lets the model claim success
        on broken work, which is exactly the 2026-05-11 Qwen3-8B 3a bug.
        A successful ``final`` causes immediate loop termination.
        """
        observations: list[WorkspaceObservation] = []
        halted = False
        for action in actions:
            spec = get_spec(action.tool)
            if halted and (spec.is_state_mutating or spec.is_terminal):
                observations.append(
                    WorkspaceObservation(
                        tool=action.tool,
                        error=(
                            "Skipped: a previous mutating action in this batch errored. "
                            "Re-issue this action next turn after addressing the failure."
                        ),
                    )
                )
                continue

            obs = env.run_action(action)
            observations.append(obs)

            if obs.final_answer is not None:
                # Terminal action: stop dispatching the rest of the batch.
                break

            if obs.error is not None and spec.is_state_mutating:
                halted = True
        return observations

    # =========================================================================
    # Stop-condition checks
    # =========================================================================

    def _check_timeout(self, iteration: int, time_start: float) -> None:
        if self.max_timeout is None:
            return
        elapsed = time.perf_counter() - time_start
        if elapsed > self.max_timeout:
            self.verbose.print_limit_exceeded(
                "timeout", f"{elapsed:.1f}s of {self.max_timeout:.1f}s"
            )
            raise TimeoutExceededError(
                elapsed=elapsed,
                timeout=self.max_timeout,
                partial_answer=self._best_partial_answer,
                message=(
                    f"Timeout exceeded after iteration {iteration}: "
                    f"{elapsed:.1f}s of {self.max_timeout:.1f}s limit"
                ),
            )

    def _check_iteration_limits(
        self,
        iteration: WorkspaceIteration,
        iteration_num: int,
        lm_handler: LMHandler,
    ) -> None:
        """Track errors / budget / token limits after each turn."""
        # Count this turn as an error if any observation carried an error
        # (mutating or not — a string of malformed read-only tool calls is
        # still a degenerate trajectory).
        had_error = any(obs.error for obs in iteration.observations)
        if had_error:
            self._consecutive_errors += 1
            for obs in iteration.observations:
                if obs.error:
                    self._last_error = obs.error
                    break
        else:
            self._consecutive_errors = 0

        if self.max_errors is not None and self._consecutive_errors >= self.max_errors:
            self.verbose.print_limit_exceeded(
                "errors",
                f"{self._consecutive_errors} consecutive errors (limit: {self.max_errors})",
            )
            raise ErrorThresholdExceededError(
                error_count=self._consecutive_errors,
                threshold=self.max_errors,
                last_error=self._last_error,
                partial_answer=self._best_partial_answer,
                message=(
                    "Error threshold exceeded: "
                    f"{self._consecutive_errors} consecutive errors "
                    f"(limit: {self.max_errors})"
                ),
            )

        if self.max_budget is not None:
            usage = lm_handler.get_usage_summary()
            cost = usage.total_cost or 0.0
            self._cumulative_cost = cost
            if self._cumulative_cost > self.max_budget:
                self.verbose.print_budget_exceeded(self._cumulative_cost, self.max_budget)
                raise BudgetExceededError(
                    spent=self._cumulative_cost,
                    budget=self.max_budget,
                    message=(
                        f"Budget exceeded after iteration {iteration_num + 1}: "
                        f"spent ${self._cumulative_cost:.6f} of ${self.max_budget:.6f}"
                    ),
                )

        if self.max_tokens is not None:
            usage = lm_handler.get_usage_summary()
            total_tokens = usage.total_input_tokens + usage.total_output_tokens
            if total_tokens > self.max_tokens:
                self.verbose.print_limit_exceeded(
                    "tokens", f"{total_tokens:,} of {self.max_tokens:,} tokens"
                )
                raise TokenLimitExceededError(
                    tokens_used=total_tokens,
                    token_limit=self.max_tokens,
                    partial_answer=self._best_partial_answer,
                    message=(
                        f"Token limit exceeded after iteration {iteration_num + 1}: "
                        f"{total_tokens:,} of {self.max_tokens:,} tokens"
                    ),
                )

    # =========================================================================
    # Tail behaviors
    # =========================================================================

    def _default_answer(
        self,
        message_history: list[dict[str, Any]],
        lm_handler: LMHandler,
    ) -> str:
        """Out-of-iterations fallback: ask the model for one last answer."""
        prompt = list(message_history) + [
            {
                "role": "user",
                "content": (
                    "You ran out of iterations without emitting "
                    '<action tool="final"><answer>...</answer></action>. '
                    "Provide a final answer now based on what you have gathered."
                ),
            }
        ]
        response, _ = lm_handler.completion_with_reasoning(prompt)
        return response

    def _build_completion(
        self,
        *,
        prompt: str | dict[str, Any] | list[Any],
        response: str,
        lm_handler: LMHandler,
        time_start: float,
    ) -> RLMChatCompletion:
        time_end = time.perf_counter()
        usage = lm_handler.get_usage_summary()
        self.verbose.print_final_answer(response)
        self.verbose.print_summary(self.max_iterations, time_end - time_start, usage.to_dict())
        return RLMChatCompletion(
            root_model=(self.backend_kwargs or {}).get("model_name", "unknown"),
            prompt=prompt,
            response=response,
            usage_summary=usage,
            execution_time=time_end - time_start,
            metadata=self.logger.get_trajectory() if self.logger else None,
        )

    def _fallback_answer(self, message: str | dict[str, Any]) -> str:
        """Plain LM completion (used when ``depth >= max_depth``).

        Kept for compatibility with caller patterns that bypass the workspace
        loop entirely; the loop itself never invokes this since the
        depth-aware system prompt + tool schema already prevent recursion at
        max depth.
        """
        client: BaseLM = get_client(self.backend, self.backend_kwargs or {})
        return client.completion(message)

    # =========================================================================
    # Lifecycle (workspace is one-shot per completion(); kept for symmetry)
    # =========================================================================

    def close(self) -> None:
        """No-op for now (workspaces are torn down inside ``completion()``)."""
        return None

    def __enter__(self) -> RLM:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        del exc_type, exc_val, exc_tb
        self.close()
        return False


def _final_artifacts_from_iteration(iteration: WorkspaceIteration) -> list[str]:
    """Pull the ``final_artifacts`` list off whichever observation in ``iteration``
    carried the terminal ``final_answer``. Returns ``[]`` if none did (which
    shouldn't happen in practice — ``iteration.final_answer`` and a final-
    carrying observation are written together in ``_completion_turn``)."""
    for obs in iteration.observations:
        if obs.final_answer is not None:
            return list(obs.final_artifacts)
    return []


# Re-export for callers that previously used ``UsageSummary`` from this module.
__all__ = ["RLM", "UsageSummary"]
