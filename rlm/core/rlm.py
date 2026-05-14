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

import json
import time
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from rlm.clients import BaseLM, get_client
from rlm.clients.openai import set_live_log_path
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
from rlm.utils.native_tools import actions_from_tool_calls, build_openai_tools
from rlm.utils.prompts import (
    build_compaction_continue_message,
    build_compaction_summary_prompt,
    build_native_tool_retry_message,
    build_parse_retry_message,
    build_workspace_initial_user_prompt,
    build_workspace_system_prompt,
    format_workspace_history,
)
from rlm.utils.rlm_utils import filter_sensitive_keys
from rlm.utils.token_utils import count_tokens
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

        # Log run metadata once if a logger / verbose is attached. Use the
        # resolved kwargs so the JSONL header reflects what actually gets
        # passed to the client (in particular, ``enable_thinking``).
        if self.logger or verbose:
            resolved = self._resolved_backend_kwargs()
            metadata = RLMMetadata(
                root_model=resolved.get("model_name", "unknown"),
                max_depth=max_depth,
                max_iterations=max_iterations,
                backend=backend,
                backend_kwargs=filter_sensitive_keys(resolved),
                action_format=self.workspace_config.parse.action_format,
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

    # Backends that route to ``rlm.clients.openai.OpenAIClient`` and therefore
    # accept ``enable_thinking`` from ``LMConfig``. Other backends (anthropic,
    # gemini, azure_openai, portkey) have their own thinking-mode mechanisms;
    # we don't inject ``enable_thinking`` into their kwargs.
    _OPENAI_COMPAT_BACKENDS: tuple[str, ...] = ("openai", "vllm", "openrouter", "vercel")

    def _resolved_backend_kwargs(self) -> dict[str, Any]:
        """Merge ``LMConfig`` into ``backend_kwargs`` for the active backend.

        Caller-supplied ``backend_kwargs`` win over the config default (so a
        run can override ``enable_thinking`` per-call without touching the
        substrate config).
        """
        merged: dict[str, Any] = dict(self.backend_kwargs or {})
        if self.backend in self._OPENAI_COMPAT_BACKENDS:
            default_thinking = (
                False
                if self.workspace_config.parse.action_format == "native"
                else self.workspace_config.lm.enable_thinking
            )
            merged.setdefault("enable_thinking", default_thinking)
        return merged

    @contextmanager
    def _spawn_completion_context(self, prompt: str | dict[str, Any] | list[Any]):
        """Bring up an LM handler + workspace env for a single completion."""
        client: BaseLM = get_client(self.backend, self._resolved_backend_kwargs())
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

        New public rollouts must use native tool calls. The legacy XML action
        format remains only for old-trace visualization and private parser
        compatibility tests.

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
        if self.workspace_config.parse.action_format == "xml":
            raise ValueError(
                "Legacy XML tool calling is deprecated for new RLM.completion() "
                "runs. Use WorkspaceConfig(parse=ParseConfig(action_format='native')) "
                "or the WorkspaceConfig() default. Existing XML logs can still be "
                "visualized, and parser compatibility tests should use private "
                "_run_loop fixtures instead of public completion()."
            )

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
            action_format=self.workspace_config.parse.action_format,
        )
        initial_user = build_workspace_initial_user_prompt(root_prompt=root_prompt)
        message_history: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": initial_user},
        ]
        base_message_history = list(message_history)
        completed_iterations: list[WorkspaceIteration] = []
        # Out-of-band compaction prefix spliced between the initial user turn
        # and the post-compress tail. Empty until the first compaction fires.
        post_compress_prefix: list[dict[str, Any]] = []

        try:
            for i in range(self.max_iterations):
                self._check_timeout(i, time_start)
                if self.on_iteration_start:
                    try:
                        self.on_iteration_start(self.depth, i + 1)
                    except Exception:
                        pass

                env.current_turn = i + 1
                message_history = (
                    base_message_history
                    + post_compress_prefix
                    + format_workspace_history(
                        completed_iterations,
                        action_format=self.workspace_config.parse.action_format,
                    )
                )
                stutter_warning = self._stutter_warning_message(completed_iterations)
                if stutter_warning is not None:
                    message_history.append({"role": "user", "content": stutter_warning})

                if self.workspace_config.compaction.enabled and self._should_compact(
                    message_history, lm_handler
                ):
                    post_compress_prefix, completed_iterations = self._compact_history(
                        message_history=message_history,
                        completed_iterations=completed_iterations,
                        lm_handler=lm_handler,
                        env=env,
                        turn=i + 1,
                    )
                    message_history = (
                        base_message_history
                        + post_compress_prefix
                        + format_workspace_history(
                            completed_iterations,
                            action_format=self.workspace_config.parse.action_format,
                        )
                    )
                    stutter_warning = self._stutter_warning_message(completed_iterations)
                    if stutter_warning is not None:
                        message_history.append({"role": "user", "content": stutter_warning})

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
                        env=env,
                        total_iterations=i + 1,
                    )

                # Keep canonical full-fidelity iterations in memory, then
                # rebuild model-facing history each turn with prompt-replay
                # compaction/receipts applied by age.
                completed_iterations.append(iteration)

        except KeyboardInterrupt:
            self.verbose.print_limit_exceeded("cancelled", "User interrupted execution")
            raise CancellationError(
                partial_answer=self._best_partial_answer,
                message="Execution cancelled by user (Ctrl+C)",
            ) from None

        # max_iterations reached without a `final` action: ask the model
        # one last time for an answer based on what it has gathered.
        message_history = (
            base_message_history
            + post_compress_prefix
            + format_workspace_history(
                completed_iterations,
                action_format=self.workspace_config.parse.action_format,
            )
        )
        final_answer = self._default_answer(message_history, lm_handler)
        return self._build_completion(
            prompt=prompt,
            response=final_answer,
            lm_handler=lm_handler,
            time_start=time_start,
            env=env,
            total_iterations=self.max_iterations,
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

        # When a JSONL trajectory is being written, also tee streaming deltas
        # to ``<jsonl>.live`` so an operator can ``tail -f`` the in-flight
        # generation. No-op for non-vLLM backends (see set_live_log_path).
        live_path: str | None = None
        if self.logger is not None and getattr(self.logger, "log_file_path", None):
            live_path = f"{self.logger.log_file_path}.live"

        try:
            with set_live_log_path(live_path):
                (
                    response,
                    reasoning,
                    actions,
                    parse_attempts,
                    lm_usage,
                    rendered_prompt,
                ) = self._call_lm_with_parse_retry(
                    lm_handler=lm_handler,
                    messages=list(message_history),
                )
        except ActionParseError as exc:
            # Build a partial iteration so the failed turn is visible in the
            # log. The exception's ``last_*`` attributes are populated by
            # ``_call_lm_with_parse_retry`` before raising.
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
                lm_usage=getattr(exc, "last_lm_usage", None),
                rendered_prompt=getattr(exc, "last_rendered_prompt", None),
            )
            exc.iteration = partial  # type: ignore[attr-defined]
            raise

        # No-op turn: ``tool_choice="auto"`` lets the model emit a prose-only
        # response with no tool calls. Skip dispatch + snapshot — there is
        # nothing to execute and the workspace is unchanged. The prompt
        # renderer (``format_workspace_iteration``) injects a nudge for the
        # next turn so the model knows it still needs to act.
        if actions:
            observations = self._dispatch_actions(env=env, actions=actions)
            snapshot = env.snapshot(turn=iteration_idx)
        else:
            observations = []
            snapshot = None

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
            lm_usage=lm_usage,
            rendered_prompt=rendered_prompt,
        )

    def _call_lm_with_parse_retry(
        self,
        *,
        lm_handler: LMHandler,
        messages: list[dict[str, Any]],
    ) -> tuple[
        str,
        str | None,
        list[WorkspaceAction],
        list[dict[str, Any]],
        dict[str, int] | None,
        str | None,
    ]:
        """Call the LM until a parseable response is produced or retries exhaust.

        Each retry appends a synthetic user feedback message to a *retry-only*
        copy of ``messages``; the main message history is unchanged. Returns
        ``(response, reasoning, actions, parse_attempts, lm_usage,
        rendered_prompt)`` for the iteration record. ``lm_usage`` sums
        per-call token counts across all attempts (so a turn that retried 3
        times reports the full cost). ``rendered_prompt`` is the
        post-chat-template prompt for the *final* attempt — i.e. what the
        model actually saw to produce the returned response. Raises
        ``ActionParseError`` if all retries fail.
        """
        retry_budget = self.workspace_config.parse.max_action_parse_retries
        attempts: list[dict[str, Any]] = []
        retry_messages = list(messages)
        usage_total: dict[str, int] | None = None
        rendered_prompt: str | None = None

        def _accumulate(usage: dict[str, int] | None) -> None:
            nonlocal usage_total
            if usage is None:
                return
            if usage_total is None:
                usage_total = dict(usage)
                return
            for k, v in usage.items():
                usage_total[k] = usage_total.get(k, 0) + v

        for attempt in range(retry_budget + 1):  # +1 = initial try
            if self.workspace_config.parse.action_format == "native":
                response = ""
                reasoning = None
                try:
                    result = lm_handler.completion_with_tools(
                        retry_messages,
                        tools=build_openai_tools(include_rlm_query=self.depth < self.max_depth),
                        tool_choice=self.workspace_config.parse.native_tool_choice,
                    )
                    _accumulate(result.usage)
                    rendered_prompt = result.rendered_prompt
                    response = result.content
                    reasoning = result.reasoning_content
                    actions = actions_from_tool_calls(result.tool_calls)
                    return (
                        result.content,
                        result.reasoning_content,
                        actions,
                        attempts,
                        usage_total,
                        rendered_prompt,
                    )
                except (ActionParseError, ValueError) as exc:
                    parse_exc = (
                        exc if isinstance(exc, ActionParseError) else ActionParseError(str(exc))
                    )
                    attempts.append(
                        {
                            "attempt": attempt + 1,
                            "response": response,
                            "error": str(parse_exc),
                            "fragment": parse_exc.fragment,
                        }
                    )
                    if attempt >= retry_budget:
                        parse_exc.args = (
                            f"Native tool-call parse failed after {retry_budget} retries: "
                            f"{parse_exc.args[0]}",
                        )
                        parse_exc.parse_attempts = list(attempts)  # type: ignore[attr-defined]
                        parse_exc.last_response = response  # type: ignore[attr-defined]
                        parse_exc.last_reasoning = reasoning  # type: ignore[attr-defined]
                        parse_exc.last_lm_usage = usage_total  # type: ignore[attr-defined]
                        parse_exc.last_rendered_prompt = rendered_prompt  # type: ignore[attr-defined]
                        raise parse_exc from None

                    feedback = build_native_tool_retry_message(str(parse_exc), parse_exc.fragment)
                    retry_messages = retry_messages + [{"role": "user", "content": feedback}]
                    continue

            response, reasoning = lm_handler.completion_with_reasoning(retry_messages)
            try:
                actions = action_parser.parse(response)
                return response, reasoning, actions, attempts, usage_total, rendered_prompt
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
                    exc.last_lm_usage = usage_total  # type: ignore[attr-defined]
                    exc.last_rendered_prompt = rendered_prompt  # type: ignore[attr-defined]
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

    def _stutter_warning_message(
        self, completed_iterations: list[WorkspaceIteration]
    ) -> str | None:
        cfg = self.workspace_config.loop_guard
        if not cfg.stutter_warning_enabled:
            return None
        threshold = cfg.repeated_action_warning_threshold
        if threshold < 2 or len(completed_iterations) < threshold:
            return None

        window = completed_iterations[-threshold:]
        first_action_sig = _action_batch_signature(window[0])
        if not first_action_sig:
            return None
        first_obs_sig = _observation_batch_signature(window[0])
        for iteration in window:
            if _has_meaningful_workspace_changes(
                iteration,
                ignored_prefixes=cfg.stutter_ignored_change_prefixes,
            ):
                return None
            if _action_batch_signature(iteration) != first_action_sig:
                return None
            if _observation_batch_signature(iteration) != first_obs_sig:
                return None

        return (
            "Loop guard: the last "
            f"{threshold} turns repeated the same tool calls and received the "
            "same observations, with no task-workspace changes. Use those "
            "observations and take a different concrete next step; do not "
            "repeat the same calls again unless something external has changed."
        )

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

    # =========================================================================
    # Compaction
    # =========================================================================

    def _should_compact(
        self,
        message_history: list[dict[str, Any]],
        lm_handler: LMHandler,
    ) -> bool:
        """True when the rendered prompt has crossed the compaction threshold."""
        threshold = self.workspace_config.compaction.threshold_tokens
        if threshold <= 0:
            return False
        model_name = lm_handler.get_client().model_name
        return count_tokens(message_history, model_name) >= threshold

    def _compact_history(
        self,
        *,
        message_history: list[dict[str, Any]],
        completed_iterations: list[WorkspaceIteration],
        lm_handler: LMHandler,
        env: DockerWorkspaceEnv,
        turn: int,
    ) -> tuple[list[dict[str, Any]], list[WorkspaceIteration]]:
        """Summarize the trajectory and reset the visible message history.

        Asks the LM (with the current full message_history visible) to write
        a structured plain-prose summary, then collapses everything pre-tail
        into ``[assistant=summary, user=continue]``. The N most recent turns
        are preserved verbatim per ``CompactionConfig.tail_turns_preserved``.

        Returns ``(post_compress_prefix, retained_iterations)``. The caller
        rebuilds ``message_history`` from ``base + prefix + retained``.
        """
        provenance_lines = self._provenance_lines_for_summary(env)
        summary_user_msg = build_compaction_summary_prompt(provenance_lines=provenance_lines)

        prompt = list(message_history) + [{"role": "user", "content": summary_user_msg}]
        model_name = lm_handler.get_client().model_name
        tokens_before = count_tokens(message_history, model_name)
        summary, _ = lm_handler.completion_with_reasoning(prompt)

        tail_n = max(0, self.workspace_config.compaction.tail_turns_preserved)
        retained_iterations = list(completed_iterations[-tail_n:]) if tail_n > 0 else []

        post_compress_prefix = [
            {"role": "assistant", "content": summary},
            {"role": "user", "content": build_compaction_continue_message()},
        ]

        if self.logger:
            self.logger.log_compaction(
                turn=turn,
                tokens_before=tokens_before,
                threshold_tokens=self.workspace_config.compaction.threshold_tokens,
                summary=summary,
                dropped_iterations=len(completed_iterations) - len(retained_iterations),
                retained_tail_iterations=len(retained_iterations),
            )
        self.verbose.print_compaction(
            turn=turn,
            tokens_before=tokens_before,
            threshold=self.workspace_config.compaction.threshold_tokens,
        )

        return post_compress_prefix, retained_iterations

    @staticmethod
    def _provenance_lines_for_summary(env: DockerWorkspaceEnv) -> list[str]:
        """Render provenance entries as ``<path> — role/turn`` lines.

        Reserved runtime-state files are skipped. Spilled tool outputs under
        ``_rlm_artifacts/_observations/`` are collapsed into a single
        breadcrumb line so the model knows they exist and how to enumerate
        them, without crowding the checklist on long runs.
        """
        reserved_state_paths = {
            "_rlm_state/provenance.json",
            "_rlm_state/action_log.jsonl",
            "_rlm_state/workspace_manifest.json",
        }
        lines: list[str] = []
        spill_paths: list[str] = []
        spill_turns: list[int] = []
        for path in sorted(env.provenance.all_paths()):
            entry = env.provenance.get(path)
            if entry is None:
                continue
            if path in reserved_state_paths:
                continue
            if path.startswith("_rlm_artifacts/_observations/"):
                spill_paths.append(path)
                spill_turns.append(entry.modified.turn)
                continue
            role = entry.modified.role
            turn = entry.modified.turn
            lines.append(f"{path} — role={role} turn={turn}")

        if spill_paths:
            turn_range = (
                f"turn {spill_turns[0]}"
                if min(spill_turns) == max(spill_turns)
                else f"turns {min(spill_turns)}–{max(spill_turns)}"
            )
            lines.append(
                f"_rlm_artifacts/_observations/ — {len(spill_paths)} spilled "
                f"tool output(s) from {turn_range}; "
                "list_directory _rlm_artifacts/_observations/ to enumerate, "
                "read_file to inspect"
            )
        return lines

    def _default_answer(
        self,
        message_history: list[dict[str, Any]],
        lm_handler: LMHandler,
    ) -> str:
        """Out-of-iterations fallback: ask the model for one last answer."""
        if self.workspace_config.parse.action_format == "native":
            result = lm_handler.completion_with_tools(
                list(message_history)
                + [
                    {
                        "role": "user",
                        "content": (
                            "You ran out of iterations. Call the final tool now with "
                            "your best answer based on what you have gathered."
                        ),
                    }
                ],
                tools=[
                    tool
                    for tool in build_openai_tools(include_rlm_query=False)
                    if tool["function"]["name"] == "final"
                ],
                tool_choice={"type": "function", "function": {"name": "final"}},
            )
            actions = actions_from_tool_calls(result.tool_calls)
            for action in actions:
                if action.tool == "final":
                    return str(action.args.get("answer", result.content))
            return result.content
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
        env: DockerWorkspaceEnv | None = None,
        total_iterations: int,
    ) -> RLMChatCompletion:
        time_end = time.perf_counter()
        usage = lm_handler.get_usage_summary()
        self.verbose.print_final_answer(response)
        self.verbose.print_summary(
            total_iterations,
            time_end - time_start,
            usage.to_dict(),
            max_iterations=self.max_iterations,
        )
        # Surface artifacts + workspace location for direct caller access.
        # ``self._last_final_artifacts`` is set by the success branch of
        # _run_loop before _build_completion is called; for the
        # max-iterations fallback path it stays at [] (the default).
        workspace_root: str | None = None
        if env is not None and self.workspace_config.docker.cleanup_mode == "keep":
            workspace_root = str(env.workspace_root)
        return RLMChatCompletion(
            root_model=(self.backend_kwargs or {}).get("model_name", "unknown"),
            prompt=prompt,
            response=response,
            usage_summary=usage,
            execution_time=time_end - time_start,
            metadata=self.logger.get_trajectory() if self.logger else None,
            final_artifacts=list(self._last_final_artifacts),
            workspace_root=workspace_root,
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


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=repr)


def _action_batch_signature(iteration: WorkspaceIteration) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (
            action.tool,
            _stable_json(action.args),
            action.body or "",
        )
        for action in iteration.actions
    )


def _observation_batch_signature(
    iteration: WorkspaceIteration,
) -> tuple[tuple[str, str, str, str, tuple[str, ...]], ...]:
    return tuple(
        (
            obs.tool,
            obs.error or "",
            obs.stdout or "",
            obs.stderr or "",
            tuple(obs.artifacts),
        )
        for obs in iteration.observations
    )


def _has_meaningful_workspace_changes(
    iteration: WorkspaceIteration,
    *,
    ignored_prefixes: tuple[str, ...],
) -> bool:
    if iteration.snapshot is None:
        return True
    for path in iteration.snapshot.changed_files:
        if not any(path.startswith(prefix) for prefix in ignored_prefixes):
            return True
    return False


# Re-export for callers that previously used ``UsageSummary`` from this module.
__all__ = ["RLM", "UsageSummary"]
