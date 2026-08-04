"""Provider-configurable structured model gateway with a deterministic fake."""

from __future__ import annotations

import asyncio
import inspect
from collections import deque
from collections.abc import Callable, Mapping
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from ke.artifact_store import ArtifactStore
from ke.ids import content_hash

OutputT = TypeVar("OutputT", bound=BaseModel)


class ModelGateway(Protocol):
    async def structured(
        self,
        prompt: str,
        result_type: type[OutputT],
        *,
        model: str | None = None,
        system_prompt: str | None = None,
        cache_key: str | None = None,
    ) -> OutputT: ...


class PydanticAIGateway:
    """PydanticAI adapter with explicit OpenAI Responses API routing."""

    def __init__(
        self,
        default_model: str,
        *,
        store: ArtifactStore | None = None,
        concurrency: int = 6,
        openai_api_key: str | None = None,
    ) -> None:
        self.default_model = default_model
        self.store = store
        self._openai_api_key = openai_api_key or None
        self._semaphore = asyncio.Semaphore(concurrency)

    def cache_identity(self) -> dict[str, str]:
        """Identify behavior that must not share cached model responses."""

        return {
            "default_model": self.default_model,
            "openai_prefix_policy": "responses",
            "openai_store": "disabled",
        }

    @staticmethod
    def _transport_identity(model: str) -> str:
        if model.startswith(("openai:", "openai-responses:")):
            return "openai-responses"
        if model.startswith("openai-chat:"):
            return "openai-chat-completions"
        return "provider-default"

    def _model(self, model: str) -> Any:
        """Build an explicit Responses model for OpenAI identifiers.

        PydanticAI's generic ``openai:`` model inference selects the Chat
        Completions adapter. Constructing ``OpenAIResponsesModel`` directly
        makes the endpoint choice unambiguous while retaining generic model
        identifiers for other providers.
        """

        if model.startswith("openai:"):
            model_name = model.removeprefix("openai:")
        elif model.startswith("openai-responses:"):
            model_name = model.removeprefix("openai-responses:")
        else:
            return model
        if not model_name:
            raise ValueError("OpenAI model identifier must include a model name")
        try:
            from pydantic_ai.models.openai import (
                OpenAIResponsesModel,
                OpenAIResponsesModelSettings,
            )
            from pydantic_ai.providers.openai import OpenAIProvider
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise RuntimeError(
                "pydantic-ai with OpenAI Responses API support is required for live model calls"
            ) from exc
        model_settings = OpenAIResponsesModelSettings(openai_store=False)
        if self._openai_api_key is None:
            return OpenAIResponsesModel(model_name, settings=model_settings)
        return OpenAIResponsesModel(
            model_name,
            provider=OpenAIProvider(api_key=self._openai_api_key),
            settings=model_settings,
        )

    async def structured(
        self,
        prompt: str,
        result_type: type[OutputT],
        *,
        model: str | None = None,
        system_prompt: str | None = None,
        cache_key: str | None = None,
    ) -> OutputT:
        selected_model = model or self.default_model
        transport = self._transport_identity(selected_model)
        if cache_key is not None:
            key = content_hash(
                {
                    "caller_cache_key": cache_key,
                    "model": selected_model,
                    "transport": transport,
                }
            )
        else:
            key = content_hash(
                {
                    "model": selected_model,
                    "transport": transport,
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "schema": result_type.model_json_schema(),
                }
            )
        if self.store:
            cached = self.store.get_cached_json("llm", key)
            if cached is not None:
                return result_type.model_validate(cached)

        try:
            from pydantic_ai import Agent
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise RuntimeError("pydantic-ai is required for live model calls") from exc

        async with self._semaphore:
            kwargs: dict[str, Any] = {"system_prompt": system_prompt or ""}
            selected_backend = self._model(selected_model)
            agent = Agent(selected_backend, output_type=result_type, **kwargs)
            result = await agent.run(prompt)
            value = getattr(result, "output", getattr(result, "data", result))
            output = value if isinstance(value, result_type) else result_type.model_validate(value)
        if self.store:
            self.store.put_cached_json("llm", key, output)
        return output


class FakeModelGateway:
    """Scriptable gateway for deterministic pipeline tests.

    Responses may be Pydantic objects, dictionaries, or a callable receiving the
    prompt and requested result type. Calls are retained for assertions.
    """

    def __init__(
        self,
        responses: list[Any] | tuple[Any, ...] | None = None,
        *,
        by_type: Mapping[type[BaseModel] | str, Any] | None = None,
        responder: Callable[..., Any] | None = None,
    ) -> None:
        self.responses = deque(responses or [])
        self.by_type = dict(by_type or {})
        self.responder = responder
        self.calls: list[dict[str, Any]] = []

    async def structured(
        self,
        prompt: str,
        result_type: type[OutputT],
        *,
        model: str | None = None,
        system_prompt: str | None = None,
        cache_key: str | None = None,
    ) -> OutputT:
        call = {
            "prompt": prompt,
            "result_type": result_type,
            "model": model,
            "system_prompt": system_prompt,
            "cache_key": cache_key,
        }
        self.calls.append(call)
        if self.responses:
            response = self.responses.popleft()
        elif result_type in self.by_type:
            response = self.by_type[result_type]
        elif result_type.__name__ in self.by_type:
            response = self.by_type[result_type.__name__]
        elif self.responder:
            response = self.responder(prompt=prompt, result_type=result_type, call=call)
        else:
            raise LookupError(f"no fake response configured for {result_type.__name__}")
        if inspect.isawaitable(response):
            response = await response
        if callable(response):
            response = response(prompt=prompt, result_type=result_type, call=call)
            if inspect.isawaitable(response):
                response = await response
        return (
            response if isinstance(response, result_type) else result_type.model_validate(response)
        )
