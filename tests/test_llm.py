import json
from types import SimpleNamespace

import httpx
import pydantic_ai
import pydantic_ai.models.openai
import pydantic_ai.providers.openai
import pytest
from openai import AsyncOpenAI
from pydantic import BaseModel

from ke.ids import content_hash
from ke.llm import FakeModelGateway, PydanticAIGateway


class Answer(BaseModel):
    value: str


@pytest.mark.asyncio
async def test_fake_gateway_validates_structured_responses() -> None:
    gateway = FakeModelGateway(responses=[{"value": "grounded"}])

    result = await gateway.structured("prompt", Answer, model="fake:model")

    assert result == Answer(value="grounded")
    assert gateway.calls[0]["result_type"] is Answer


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    ["openai:gpt-5.6-terra", "openai-responses:gpt-5.6-terra"],
)
async def test_openai_models_use_explicit_responses_adapter(monkeypatch, model: str) -> None:
    models: list[object] = []
    providers: list[object] = []

    class StubResponsesModel:
        def __init__(
            self,
            model_name: str,
            *,
            provider: object = "openai",
            settings: object = None,
        ) -> None:
            self.model_name = model_name
            self.provider = provider
            self.settings = settings

    class StubOpenAIProvider:
        def __init__(self, *, api_key: str) -> None:
            self.api_key = api_key
            providers.append(self)

    class StubAgent:
        def __init__(self, model: object, **kwargs: object) -> None:
            models.append(model)
            assert kwargs["output_type"] is Answer

        async def run(self, prompt: str) -> SimpleNamespace:
            assert prompt == "prompt"
            return SimpleNamespace(output=Answer(value="responses"))

    monkeypatch.setattr(pydantic_ai.models.openai, "OpenAIResponsesModel", StubResponsesModel)
    monkeypatch.setattr(pydantic_ai.providers.openai, "OpenAIProvider", StubOpenAIProvider)
    monkeypatch.setattr(pydantic_ai, "Agent", StubAgent)
    gateway = PydanticAIGateway(
        model,
        openai_api_key="test-openai-key",
    )

    result = await gateway.structured("prompt", Answer)

    assert result == Answer(value="responses")
    assert len(models) == 1
    assert isinstance(models[0], StubResponsesModel)
    assert models[0].model_name == "gpt-5.6-terra"
    assert providers[0].api_key == "test-openai-key"
    assert models[0].provider is providers[0]
    assert models[0].settings == {"openai_store": False}
    assert gateway.cache_identity()["openai_prefix_policy"] == "responses"
    assert gateway.cache_identity()["openai_store"] == "disabled"
    assert "test-openai-key" not in str(gateway.cache_identity())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    [
        "anthropic:claude-sonnet-4-0",
        "google-gla:gemini-2.5-pro",
        "openai-chat:gpt-5.6-terra",
    ],
)
async def test_explicit_provider_prefixes_keep_pydantic_ai_routing(
    monkeypatch,
    model: str,
) -> None:
    models: list[object] = []

    class StubAgent:
        def __init__(self, model: object, **kwargs: object) -> None:
            models.append(model)

        async def run(self, prompt: str) -> SimpleNamespace:
            return SimpleNamespace(output={"value": "provider"})

    monkeypatch.setattr(pydantic_ai, "Agent", StubAgent)
    gateway = PydanticAIGateway(model)

    result = await gateway.structured("prompt", Answer)

    assert result == Answer(value="provider")
    assert models == [model]


@pytest.mark.asyncio
async def test_openai_gateway_posts_structured_output_to_responses_endpoint(monkeypatch) -> None:
    paths: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        payload = json.loads(request.content)
        assert payload["store"] is False
        tool = payload["tools"][0]
        return httpx.Response(
            200,
            json={
                "id": "resp_test",
                "created_at": 0,
                "model": payload["model"],
                "object": "response",
                "output": [
                    {
                        "arguments": json.dumps({"value": "responses"}),
                        "call_id": "call_test",
                        "name": tool["name"],
                        "type": "function_call",
                    }
                ],
                "parallel_tool_calls": True,
                "tool_choice": payload["tool_choice"],
                "tools": payload["tools"],
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    openai_client = AsyncOpenAI(
        api_key="test-openai-key",
        base_url="https://openai.test/v1",
        http_client=http_client,
    )
    provider = pydantic_ai.providers.openai.OpenAIProvider(openai_client=openai_client)
    monkeypatch.setattr(
        pydantic_ai.providers.openai,
        "OpenAIProvider",
        lambda **kwargs: provider,
    )
    gateway = PydanticAIGateway(
        "openai:gpt-5.6-terra",
        openai_api_key="test-openai-key",
    )

    try:
        result = await gateway.structured("prompt", Answer)
    finally:
        await openai_client.close()

    assert result == Answer(value="responses")
    assert paths == ["/v1/responses"]


@pytest.mark.asyncio
async def test_caller_cache_keys_are_scoped_to_responses_transport() -> None:
    keys: list[str] = []

    class CachedStore:
        def get_cached_json(self, namespace: str, key: str) -> dict[str, str]:
            assert namespace == "llm"
            keys.append(key)
            return {"value": "cached"}

    gateway = PydanticAIGateway(
        "openai:gpt-5.6-terra",
        store=CachedStore(),  # type: ignore[arg-type]
    )

    result = await gateway.structured("prompt", Answer, cache_key="legacy-chat-key")

    assert result == Answer(value="cached")
    assert keys == [
        content_hash(
            {
                "caller_cache_key": "legacy-chat-key",
                "model": "openai:gpt-5.6-terra",
                "transport": "openai-responses",
            }
        )
    ]
    assert keys != ["legacy-chat-key"]
