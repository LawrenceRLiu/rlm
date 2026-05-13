from abc import ABC, abstractmethod
from typing import Any

from rlm.core.types import ModelUsageSummary, UsageSummary

# Default timeout for LM API calls (in seconds)
DEFAULT_TIMEOUT: float = 300.0


# Common sampling parameters to pass to completion APIs rather than client constructors
SAMPLING_PARAM_KEYS = {
    "temperature", "top_p", "top_k", "max_tokens", "stop",
    "presence_penalty", "frequency_penalty", "n", "seed", "response_format",
    "min_p", "repetition_penalty"
}


class BaseLM(ABC):
    """
    Base class for all language model routers / clients. When the RLM makes sub-calls, it currently
    does so in a model-agnostic way, so this class provides a base interface for all language models.
    """

    def __init__(self, model_name: str, timeout: float = DEFAULT_TIMEOUT, **kwargs):
        self.model_name = model_name
        self.timeout = timeout
        
        self.sampling_kwargs = {}
        for key in list(kwargs.keys()):
            if key in SAMPLING_PARAM_KEYS:
                self.sampling_kwargs[key] = kwargs.pop(key)
                
        self.kwargs = kwargs
        # Backend reasoning-channel content from the last completion() call,
        # if the subclass populates it. Defaults to None for backends that
        # do not surface a separate reasoning channel.
        self._last_reasoning_content: str | None = None

    @abstractmethod
    def completion(self, prompt: str | dict[str, Any]) -> str:
        raise NotImplementedError

    @abstractmethod
    async def acompletion(self, prompt: str | dict[str, Any]) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_usage_summary(self) -> UsageSummary:
        """Get cost summary for all model calls."""
        raise NotImplementedError

    @abstractmethod
    def get_last_usage(self) -> ModelUsageSummary:
        """Get the last cost summary of the model."""
        raise NotImplementedError

    def get_last_reasoning_content(self) -> str | None:
        """Reasoning-channel content from the most recent ``completion()`` call.

        Subclasses that have access to a backend reasoning channel (Anthropic
        extended thinking, OpenAI reasoning, Gemini thinking, etc.) should
        store it on ``self._last_reasoning_content`` from inside their
        ``completion()`` / ``acompletion()`` implementations. Backends without
        a separate reasoning channel can leave the default of ``None``.
        """
        return self._last_reasoning_content
