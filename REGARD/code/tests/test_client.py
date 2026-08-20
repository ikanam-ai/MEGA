from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from scripts.common.config import models_config
from scripts.providers.client import ProviderError, _build_body, call_model, get_model_spec


@pytest.fixture(autouse=True)
def endpoint_environment(monkeypatch):
    monkeypatch.setenv("VLLM_ENDPOINT", "http://localhost:9000/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")


def test_registry_is_the_fixed_19_model_panel() -> None:
    generators = models_config()["generators"]
    assert len(generators) == 19
    assert set(generators) >= {"yandexgpt", "qwen36_27b", "granite3_8b", "solar_10b"}
    assert sum(spec["role"] == "local" for spec in generators.values()) == 4


def test_openai_body_and_thinking_override() -> None:
    body = _build_body(get_model_spec("qwen36_35b"), "judge", "score this", 0.0, 800)
    assert body["messages"] == [
        {"role": "system", "content": "judge"},
        {"role": "user", "content": "score this"},
    ]
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_cloud_and_local_roundtrips() -> None:
    response = MagicMock(status_code=200)
    response.json.return_value = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
    with patch("httpx.post", return_value=response) as post:
        assert call_model("gpt4o_mini", user="test").text == "ok"
        assert post.call_args.kwargs["headers"] == {"Authorization": "Bearer sk-test"}
    with patch("httpx.post", return_value=response) as post:
        assert call_model("qwen36_35b", user="test").finish_reason == "stop"
        assert post.call_args.kwargs["headers"] == {}


def test_malformed_response_and_unknown_model() -> None:
    response = MagicMock(status_code=200)
    response.json.return_value = {"unexpected": "shape"}
    with patch("httpx.post", return_value=response), pytest.raises(ProviderError, match="malformed"):
        call_model("gpt4o_mini", user="test")
    with pytest.raises(KeyError):
        get_model_spec("does_not_exist")
