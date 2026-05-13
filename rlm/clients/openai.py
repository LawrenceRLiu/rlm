import os
import re
from collections import defaultdict
from typing import Any

import openai
from dotenv import load_dotenv

from rlm.clients.base_lm import BaseLM
from rlm.core.types import ModelUsageSummary, UsageSummary

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

        response = self.client.chat.completions.create(
            model=model, messages=messages, extra_body=extra_body, **openai_kwargs
        )
        self._track_cost(response, model)
        return self._capture_reasoning_and_content(response)

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

    def _track_cost(self, response: openai.ChatCompletion, model: str):
        self.model_call_counts[model] += 1

        usage = getattr(response, "usage", None)
        if usage is None:
            raise ValueError("No usage data received. Tracking tokens not possible.")

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
