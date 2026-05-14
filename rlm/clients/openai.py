import contextvars
import json
import os
import re
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

import httpx
import openai
from dotenv import load_dotenv

from rlm.clients.base_lm import BaseLM
from rlm.core.types import LMCompletionResult, LMToolCall, ModelUsageSummary, UsageSummary

# Sidecar live-log path. When set (via ``set_live_log_path``) AND the client is
# pointed at a self-hosted vLLM replica, ``completion``/``completion_with_tools``
# switch to ``stream=True`` and tee partial deltas into the file at this path
# so a ``tail -f`` of the sidecar shows the in-flight reasoning/content/tool
# args. Other backends ignore it. Concurrency-safe per task via ContextVar.
live_log_path_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "rlm_openai_live_log_path", default=None
)


@contextmanager
def set_live_log_path(path: str | None) -> Iterator[None]:
    """Bind a sidecar live-log path for the duration of the ``with`` block.

    Pass ``None`` to explicitly disable streaming inside the block.
    """
    token = live_log_path_var.set(path)
    try:
        yield
    finally:
        live_log_path_var.reset(token)


def _write_live(path: str, text: str, *, mode: str = "a") -> None:
    """Best-effort write to the sidecar live-log; never propagate errors.

    The live-log is purely a debugging aid; an I/O hiccup must not break the
    primary completion path.
    """
    try:
        with open(path, mode) as f:
            f.write(text)
    except OSError:
        pass


# Gemma 4 thinking block (asymmetric special tokens). When vLLM's
# ``--reasoning-parser gemma4`` fails to surface ``reasoning_content`` (see
# vllm#38855 in 0.19.1), the block lands as literal text in ``content``. The
# client-side fallback below detects and extracts it so the substrate's
# ``reasoning_content`` channel still populates, and the trailing answer is
# returned as the assistant content.
_GEMMA_CHANNEL_RE = re.compile(
    r"<\|channel>\s*thought\b(.*?)<channel\|>(.*)",
    flags=re.DOTALL,
)
# Qwen-family ``<think>...</think>`` block. Same fallback strategy.
_THINK_RE = re.compile(
    r"<\s*think\b[^>]*>(.*?)<\s*/\s*think\s*>(.*)",
    flags=re.DOTALL | re.IGNORECASE,
)

load_dotenv()

# Load API keys from environment variables
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DEFAULT_VERCEL_API_KEY = os.getenv("AI_GATEWAY_API_KEY")
DEFAULT_PRIME_API_KEY = os.getenv("PRIME_API_KEY")
DEFAULT_PRIME_INTELLECT_BASE_URL = "https://api.pinference.ai/api/v1/"

# Public OpenAI-compatible endpoints that do NOT accept vLLM's
# ``chat_template_kwargs`` extra-body field. Sending it to these endpoints
# can 400 or silently degrade. ``enable_thinking`` is only forwarded when the
# client's base URL is not in this set (i.e., we're pointed at a self-hosted
# vLLM replica).
_PUBLIC_OPENAI_COMPAT_BASES: frozenset[str] = frozenset(
    {
        "https://api.openai.com/v1",
        "https://openrouter.ai/api/v1",
        "https://ai-gateway.vercel.sh/v1",
        DEFAULT_PRIME_INTELLECT_BASE_URL,
    }
)


class OpenAIClient(BaseLM):
    """
    LM Client for running models with the OpenAI API. Works with vLLM as well.

    Any additional keyword arguments (e.g. default_headers, default_query, max_retries)
    are passed through to the underlying openai.OpenAI and openai.AsyncOpenAI constructors.
    Only model_name is excluded, since it is not a client constructor argument.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        base_url: str | None = None,
        enable_thinking: bool = True,
        **kwargs,
    ):
        super().__init__(model_name=model_name, **kwargs)
        self.enable_thinking: bool = enable_thinking

        if api_key is None:
            if base_url == "https://api.openai.com/v1" or base_url is None:
                api_key = DEFAULT_OPENAI_API_KEY
            elif base_url == "https://openrouter.ai/api/v1":
                api_key = DEFAULT_OPENROUTER_API_KEY
            elif base_url == "https://ai-gateway.vercel.sh/v1":
                api_key = DEFAULT_VERCEL_API_KEY
            elif base_url == DEFAULT_PRIME_INTELLECT_BASE_URL:
                api_key = DEFAULT_PRIME_API_KEY

        # Pass through arbitrary kwargs to the OpenAI client (e.g. default_headers, default_query, max_retries).
        # Exclude model_name since it is not an OpenAI client constructor argument.
        client_kwargs = {
            "api_key": api_key,
            "base_url": base_url,
            "timeout": self.timeout,
            **{k: v for k, v in self.kwargs.items() if k != "model_name"},
        }
        self.client = openai.OpenAI(**client_kwargs)
        self.async_client = openai.AsyncOpenAI(**client_kwargs)
        self.model_name = model_name
        self.base_url = base_url  # Track for cost extraction
        self._is_self_hosted_vllm: bool = (
            base_url is not None and base_url not in _PUBLIC_OPENAI_COMPAT_BASES
        )

        # Per-model usage tracking
        self.model_call_counts: dict[str, int] = defaultdict(int)
        self.model_input_tokens: dict[str, int] = defaultdict(int)
        self.model_output_tokens: dict[str, int] = defaultdict(int)
        self.model_total_tokens: dict[str, int] = defaultdict(int)
        self.model_costs: dict[str, float] = defaultdict(float)  # Cost in USD

        # Last rendered chat prompt (post-chat-template, including tool
        # descriptions vLLM injects). Populated best-effort per call for
        # self-hosted vLLM via ``/tokenize`` + ``/detokenize``; ``None`` for
        # public endpoints or on transport error.
        self._last_rendered_prompt: str | None = None

    def completion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(isinstance(item, dict) for item in prompt):
            messages = prompt
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        model = model or self.model_name
        if not model:
            raise ValueError("Model name is required for OpenAI client.")

        extra_body = self._build_extra_body()

        openai_kwargs = dict(self.sampling_kwargs)
        for k in ["top_k", "min_p", "repetition_penalty"]:
            if k in openai_kwargs:
                extra_body[k] = openai_kwargs.pop(k)

        live_path = live_log_path_var.get()
        if live_path and self._is_self_hosted_vllm:
            return self._completion_streaming(
                messages=messages,
                model=model,
                extra_body=extra_body,
                openai_kwargs=openai_kwargs,
                live_path=live_path,
            )

        response = self.client.chat.completions.create(
            model=model, messages=messages, extra_body=extra_body, **openai_kwargs
        )
        self._track_cost(response, model)
        return self._capture_reasoning_and_content(response)

    def completion_with_tools(
        self,
        prompt: str | list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        tool_choice: Any = "required",
        model: str | None = None,
    ) -> LMCompletionResult:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(isinstance(item, dict) for item in prompt):
            messages = prompt
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        model = model or self.model_name
        if not model:
            raise ValueError("Model name is required for OpenAI client.")

        extra_body = self._build_extra_body()
        openai_kwargs = dict(self.sampling_kwargs)
        for k in ["top_k", "min_p", "repetition_penalty"]:
            if k in openai_kwargs:
                extra_body[k] = openai_kwargs.pop(k)

        # Best-effort: capture the post-chat-template prompt vLLM will see.
        # Done before the chat call so the streaming path picks it up too.
        self._last_rendered_prompt = self._render_chat_prompt(messages, tools, extra_body)

        live_path = live_log_path_var.get()
        if live_path and self._is_self_hosted_vllm:
            return self._completion_with_tools_streaming(
                messages=messages,
                model=model,
                tools=tools,
                tool_choice=tool_choice,
                extra_body=extra_body,
                openai_kwargs=openai_kwargs,
                live_path=live_path,
            )

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            extra_body=extra_body,
            **openai_kwargs,
        )
        self._track_cost(response, model)
        content = self._capture_reasoning_and_content(response)
        calls: list[LMToolCall] = []
        for call in response.choices[0].message.tool_calls or []:
            raw_args = call.function.arguments or "{}"
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Tool call {call.id} ({call.function.name}) arguments are not valid JSON: "
                    f"{exc.msg}"
                ) from exc
            if not isinstance(args, dict):
                raise ValueError(
                    f"Tool call {call.id} ({call.function.name}) arguments must be a JSON object."
                )
            calls.append(LMToolCall(id=call.id, name=call.function.name, arguments=args))
        return LMCompletionResult(
            content=content,
            reasoning_content=self.get_last_reasoning_content(),
            tool_calls=calls,
            usage=self._last_usage_dict(response.usage),
            rendered_prompt=self._last_rendered_prompt,
        )

    async def acompletion(
        self, prompt: str | list[dict[str, Any]], model: str | None = None
    ) -> str:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(isinstance(item, dict) for item in prompt):
            messages = prompt
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        model = model or self.model_name
        if not model:
            raise ValueError("Model name is required for OpenAI client.")

        extra_body = self._build_extra_body()

        openai_kwargs = dict(self.sampling_kwargs)
        for k in ["top_k", "min_p", "repetition_penalty"]:
            if k in openai_kwargs:
                extra_body[k] = openai_kwargs.pop(k)

        response = await self.async_client.chat.completions.create(
            model=model, messages=messages, extra_body=extra_body, **openai_kwargs
        )
        self._track_cost(response, model)
        return self._capture_reasoning_and_content(response)

    def _render_chat_prompt(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        extra_body: dict[str, Any],
    ) -> str | None:
        """Render the post-chat-template prompt vLLM would see (best-effort).

        Hits vLLM's ``/tokenize`` with the same messages + tools, then
        ``/detokenize`` to recover the rendered string. This is the only way
        to see what the model actually reads — the system+tools envelope
        vLLM injects via the chat template (tool descriptions,
        ``<tool_call>`` wrapping instructions, special tokens) is invisible
        from the chat completions request alone.

        Returns ``None`` for non-vLLM endpoints, on any transport error, or
        if the endpoint doesn't speak the tokenize protocol. Never raises —
        this is observability only.
        """
        if not self._is_self_hosted_vllm:
            return None
        base = str(self.client.base_url).rstrip("/")
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "add_generation_prompt": True,
        }
        if tools:
            payload["tools"] = tools
        if extra_body.get("chat_template_kwargs"):
            payload["chat_template_kwargs"] = extra_body["chat_template_kwargs"]
        try:
            with httpx.Client(timeout=10.0) as client:
                tok_resp = client.post(f"{base}/tokenize", json=payload)
                tok_resp.raise_for_status()
                tok_data = tok_resp.json()
                # Newer vLLM returns ``prompt`` directly; older returns only
                # ``tokens``. Prefer the direct string when present to avoid
                # a second round-trip.
                if "prompt" in tok_data and isinstance(tok_data["prompt"], str):
                    return tok_data["prompt"]
                tokens = tok_data.get("tokens") or tok_data.get("prompt_token_ids")
                if not tokens:
                    return None
                det_resp = client.post(
                    f"{base}/detokenize",
                    json={"model": self.model_name, "tokens": tokens},
                )
                det_resp.raise_for_status()
                det_data = det_resp.json()
                return det_data.get("prompt")
        except (httpx.HTTPError, ValueError, KeyError):
            return None

    def _build_extra_body(self) -> dict[str, Any]:
        """Assemble the per-request ``extra_body`` dict.

        ``chat_template_kwargs`` is a vLLM-specific extra-body field; sending
        it to public OpenAI-compatible APIs (OpenAI, OpenRouter, Vercel, Prime)
        can 400 or be silently dropped. Only forward when pointed at a
        self-hosted vLLM replica.
        """
        extra_body: dict[str, Any] = {}
        if self.client.base_url == DEFAULT_PRIME_INTELLECT_BASE_URL:
            extra_body["usage"] = {"include": True}
        if self._is_self_hosted_vllm:
            extra_body["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
        return extra_body

    def _capture_reasoning_and_content(self, response: openai.ChatCompletion) -> str:
        """Populate ``_last_reasoning_content`` and return the assistant content.

        Two paths feed ``reasoning_content``:

        1. **Server-side parser.** vLLM surfaces a separate
           ``reasoning_content`` field on ``message`` when launched with
           ``--reasoning-parser <name>`` (gemma4, qwen3, etc.). When present,
           we trust it and leave ``content`` untouched.
        2. **Client-side fallback.** When the server-side parser is absent or
           buggy (e.g. vllm#38855 for gemma4 in 0.19.1), the reasoning block
           lands as literal text in ``content``. We detect known patterns
           (Gemma 4 ``<|channel>thought ... <channel|>``, Qwen-family
           ``<think>...</think>``), move the inner text into
           ``_last_reasoning_content``, and strip them out of the returned
           content. Anything after the closing tag (the model's actual
           answer) is preserved.

        Backends that don't expose reasoning either way leave
        ``_last_reasoning_content`` at ``None`` and ``content`` unchanged.
        """
        msg = response.choices[0].message
        # vLLM 0.19.1 names the field ``reasoning`` on the message. Older vLLM
        # versions and some other backends use ``reasoning_content``. Check
        # both, preferring the newer name. Both routes (direct attribute and
        # the pydantic ``model_extra`` fall-through) are tried since the
        # OpenAI SDK only types the standard fields.
        reasoning: str | None = None
        for field in ("reasoning", "reasoning_content"):
            val = getattr(msg, field, None)
            if val is None and hasattr(msg, "model_extra") and msg.model_extra:
                val = msg.model_extra.get(field)
            if val:
                reasoning = val
                break
        content = msg.content or ""

        if not reasoning and content:
            # Try the client-side fallback. First match wins; in practice a
            # single response carries only one format.
            for pattern in (_GEMMA_CHANNEL_RE, _THINK_RE):
                m = pattern.search(content)
                if m is None:
                    continue
                # Inner text (group 1) → reasoning; pre-match prefix +
                # post-match suffix → content. The fallback intentionally
                # preserves any prose before the reasoning block (rare but
                # legitimate; some models emit a leading "Let me think:" line).
                reasoning = (m.group(1) or "").strip() or None
                content = (content[: m.start()] + (m.group(2) or "")).lstrip()
                break

        self._last_reasoning_content = reasoning or None
        return content

    def _track_usage(self, usage: Any, model: str) -> None:
        """Update per-model counters from a ``usage`` object.

        Shared by the non-streaming path (which reads ``response.usage``) and
        the streaming path (which receives ``usage`` in the final stream chunk
        when ``stream_options={"include_usage": True}`` is set).
        """
        self.model_call_counts[model] += 1
        self.model_input_tokens[model] += usage.prompt_tokens
        self.model_output_tokens[model] += usage.completion_tokens
        self.model_total_tokens[model] += usage.total_tokens

        # Track last call for handler to read
        self.last_prompt_tokens = usage.prompt_tokens
        self.last_completion_tokens = usage.completion_tokens

        # Extract cost from OpenRouter responses (cost is in USD)
        # OpenRouter returns cost in usage.model_extra for pydantic models
        self.last_cost: float | None = None
        cost = None

        # Try direct attribute first
        if hasattr(usage, "cost") and usage.cost:
            cost = usage.cost
        # Then try model_extra (OpenRouter uses this)
        elif hasattr(usage, "model_extra") and usage.model_extra:
            extra = usage.model_extra
            # Primary cost field (may be 0 for BYOK)
            if extra.get("cost"):
                cost = extra["cost"]
            # Fallback to upstream cost details
            elif extra.get("cost_details", {}).get("upstream_inference_cost"):
                cost = extra["cost_details"]["upstream_inference_cost"]

        if cost is not None and cost > 0:
            self.last_cost = float(cost)
            self.model_costs[model] += self.last_cost

    def _track_cost(self, response: openai.ChatCompletion, model: str):
        usage = getattr(response, "usage", None)
        if usage is None:
            raise ValueError("No usage data received. Tracking tokens not possible.")
        self._track_usage(usage, model)

    @staticmethod
    def _extract_reasoning_delta(delta: Any) -> str | None:
        """Pull the reasoning fragment off a streaming ``delta``.

        vLLM's qwen3/gemma4 reasoning parsers surface fragments on either
        ``reasoning_content`` (older) or ``reasoning`` (newer). Both may
        only appear in pydantic ``model_extra`` for fields the OpenAI SDK
        doesn't type.
        """
        for field in ("reasoning_content", "reasoning"):
            val = getattr(delta, field, None)
            if val is None and getattr(delta, "model_extra", None):
                val = delta.model_extra.get(field)
            if val:
                return val
        return None

    def _completion_streaming(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str,
        extra_body: dict[str, Any],
        openai_kwargs: dict[str, Any],
        live_path: str,
    ) -> str:
        """Streaming counterpart to ``completion`` that tees deltas to ``live_path``.

        Only invoked when ``live_log_path_var`` is set AND the client is
        pointed at a self-hosted vLLM replica. Final usage is captured from
        the trailing ``stream_options.include_usage`` chunk so cost tracking
        matches the non-streaming path.
        """
        _write_live(
            live_path,
            f"# {datetime.now().isoformat()} model={model}\n",
            mode="w",
        )

        stream_kwargs = dict(openai_kwargs)
        stream_kwargs["stream"] = True
        stream_kwargs["stream_options"] = {"include_usage": True}

        stream = self.client.chat.completions.create(
            model=model, messages=messages, extra_body=extra_body, **stream_kwargs
        )

        reasoning_chunks: list[str] = []
        content_chunks: list[str] = []
        usage = None
        seen_reasoning = False
        seen_content = False

        for chunk in stream:
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                usage = chunk_usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            r_delta = self._extract_reasoning_delta(delta)
            if r_delta:
                if not seen_reasoning:
                    _write_live(live_path, "=== reasoning ===\n")
                    seen_reasoning = True
                reasoning_chunks.append(r_delta)
                _write_live(live_path, r_delta)

            c_delta = getattr(delta, "content", None)
            if c_delta:
                if not seen_content:
                    _write_live(live_path, "\n\n=== content ===\n")
                    seen_content = True
                content_chunks.append(c_delta)
                _write_live(live_path, c_delta)

        if usage is None:
            raise ValueError("No usage data received from streaming response.")
        self._track_usage(usage, model)

        reasoning = "".join(reasoning_chunks) or None
        content = "".join(content_chunks)
        # Client-side fallback (mirrors _capture_reasoning_and_content): if the
        # server didn't separate reasoning, try to extract a known block from
        # the accumulated content.
        if not reasoning and content:
            for pattern in (_GEMMA_CHANNEL_RE, _THINK_RE):
                m = pattern.search(content)
                if m is None:
                    continue
                reasoning = (m.group(1) or "").strip() or None
                content = (content[: m.start()] + (m.group(2) or "")).lstrip()
                break

        self._last_reasoning_content = reasoning
        return content

    def _completion_with_tools_streaming(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[dict[str, Any]],
        tool_choice: Any,
        extra_body: dict[str, Any],
        openai_kwargs: dict[str, Any],
        live_path: str,
    ) -> LMCompletionResult:
        """Streaming counterpart to ``completion_with_tools``.

        Reassembles tool calls from per-chunk ``delta.tool_calls`` fragments
        (each carries an ``index`` plus optional ``id``, ``function.name``,
        and an ``function.arguments`` string fragment). Tees reasoning,
        content, and tool-arg deltas into ``live_path`` so ``tail -f``
        shows the in-flight tool selection too.
        """
        _write_live(
            live_path,
            f"# {datetime.now().isoformat()} model={model} (with tools)\n",
            mode="w",
        )

        stream_kwargs = dict(openai_kwargs)
        stream_kwargs["stream"] = True
        stream_kwargs["stream_options"] = {"include_usage": True}

        stream = self.client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            extra_body=extra_body,
            **stream_kwargs,
        )

        reasoning_chunks: list[str] = []
        content_chunks: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        usage = None
        seen_reasoning = False
        seen_content = False
        seen_tool_section = False

        for chunk in stream:
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                usage = chunk_usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            r_delta = self._extract_reasoning_delta(delta)
            if r_delta:
                if not seen_reasoning:
                    _write_live(live_path, "=== reasoning ===\n")
                    seen_reasoning = True
                reasoning_chunks.append(r_delta)
                _write_live(live_path, r_delta)

            c_delta = getattr(delta, "content", None)
            if c_delta:
                if not seen_content:
                    _write_live(live_path, "\n\n=== content ===\n")
                    seen_content = True
                content_chunks.append(c_delta)
                _write_live(live_path, c_delta)

            tc_deltas = getattr(delta, "tool_calls", None) or []
            for tc in tc_deltas:
                idx = tc.index
                entry = tool_calls.setdefault(idx, {"id": None, "name": None, "arguments": ""})
                if getattr(tc, "id", None):
                    entry["id"] = tc.id
                fn = getattr(tc, "function", None)
                if fn is not None:
                    fn_name = getattr(fn, "name", None)
                    if fn_name:
                        entry["name"] = fn_name
                        if not seen_tool_section:
                            _write_live(live_path, "\n\n=== tool_calls ===\n")
                            seen_tool_section = True
                        _write_live(live_path, f"\n[#{idx}] {fn_name}: ")
                    fn_args = getattr(fn, "arguments", None)
                    if fn_args:
                        entry["arguments"] += fn_args
                        _write_live(live_path, fn_args)

        if usage is None:
            raise ValueError("No usage data received from streaming response.")
        self._track_usage(usage, model)

        reasoning = "".join(reasoning_chunks) or None
        content = "".join(content_chunks)
        if not reasoning and content:
            for pattern in (_GEMMA_CHANNEL_RE, _THINK_RE):
                m = pattern.search(content)
                if m is None:
                    continue
                reasoning = (m.group(1) or "").strip() or None
                content = (content[: m.start()] + (m.group(2) or "")).lstrip()
                break
        self._last_reasoning_content = reasoning

        calls: list[LMToolCall] = []
        for idx in sorted(tool_calls.keys()):
            entry = tool_calls[idx]
            if entry["id"] is None or entry["name"] is None:
                raise ValueError(f"Streamed tool call at index {idx} missing id or name: {entry!r}")
            raw_args = entry["arguments"] or "{}"
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Tool call {entry['id']} ({entry['name']}) arguments are not valid JSON: "
                    f"{exc.msg}"
                ) from exc
            if not isinstance(args, dict):
                raise ValueError(
                    f"Tool call {entry['id']} ({entry['name']}) arguments must be a JSON object."
                )
            calls.append(LMToolCall(id=entry["id"], name=entry["name"], arguments=args))

        return LMCompletionResult(
            content=content,
            reasoning_content=reasoning,
            tool_calls=calls,
            usage=self._last_usage_dict(usage),
            rendered_prompt=self._last_rendered_prompt,
        )

    def get_usage_summary(self) -> UsageSummary:
        model_summaries = {}
        for model in self.model_call_counts:
            cost = self.model_costs.get(model)
            model_summaries[model] = ModelUsageSummary(
                total_calls=self.model_call_counts[model],
                total_input_tokens=self.model_input_tokens[model],
                total_output_tokens=self.model_output_tokens[model],
                total_cost=cost if cost else None,
            )
        return UsageSummary(model_usage_summaries=model_summaries)

    def get_last_usage(self) -> ModelUsageSummary:
        return ModelUsageSummary(
            total_calls=1,
            total_input_tokens=self.last_prompt_tokens,
            total_output_tokens=self.last_completion_tokens,
            total_cost=getattr(self, "last_cost", None),
        )

    @staticmethod
    def _last_usage_dict(usage: Any) -> dict[str, int] | None:
        """Snapshot the per-call ``usage`` triple for the iteration record.

        Returns ``None`` when the backend didn't provide usage on this call
        (some non-vLLM endpoints elide it). Diagnostic aid: if
        ``completion_tokens`` is far larger than the recorded reasoning +
        response text, the backend's parser dropped tokens between the wire
        and ``message.{content,reasoning_content,tool_calls}``.
        """
        if usage is None:
            return None
        return {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        }
