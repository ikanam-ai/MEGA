"""Тесты унифицированного клиента (scripts/providers/client.py) — без реальных
сетевых вызовов, httpx.post подменяется. Проверяет, что один и тот же call_model()
корректно собирает запрос и разбирает ответ для message_format openai/anthropic
(используются текущим config/models.yaml — все local-модели и open-weight intl
идут через общий self-hosted VLLM_ENDPOINT без auth, judge-модели: qwen36_35b и
gpt4o_mini, см. models.yaml), и что retry/error-handling работают как ожидается.

message_format=yandex и auth.type=oauth2_client_credentials остаются в client.py
как поддерживаемый механизм (на случай возврата к облачным API), но ни одна
модель в текущем конфиге их не использует — эти два механизма проверяются
напрямую на синтетическом spec, не через get_model_spec("yandexgpt"/"gigachat").
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from scripts.providers.client import (
    ProviderError,
    RetryableProviderError,
    _build_body,
    _build_headers,
    _oauth_token_cache,
    call_model,
    get_model_spec,
)

YANDEX_FORMAT_SPEC = {
    "model": "yandexgpt/latest",
    "message_format": "yandex",
    "auth": {"type": "yandex_api_key", "api_key_env": "YANDEX_API_KEY", "folder_id_env": "YANDEX_FOLDER_ID"},
}

OAUTH_SPEC = {
    "model": "GigaChat",
    "message_format": "openai",
    "auth": {
        "type": "oauth2_client_credentials",
        "auth_key_env": "GIGACHAT_AUTH_KEY",
        "scope": "GIGACHAT_API_PERS",
        "oauth": {
            "url": "https://example.test/oauth",
            "token_path": "access_token",
            "expires_at_path": "expires_at",
        },
    },
}


@pytest.fixture(autouse=True)
def _clear_oauth_cache():
    _oauth_token_cache.clear()
    yield
    _oauth_token_cache.clear()


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("YANDEX_API_KEY", "x")
    monkeypatch.setenv("YANDEX_FOLDER_ID", "b1gtest123")
    monkeypatch.setenv("GLM_API_KEY", "x")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("GIGACHAT_AUTH_KEY", "x")
    monkeypatch.setenv("VLLM_ENDPOINT", "http://localhost:9000/v1")


def test_build_body_openai_format():
    spec = get_model_spec("gpt4o_mini")
    body = _build_body(spec, system=None, user="Что думаешь о Казахстане?", temperature=0.5, max_tokens=300)
    assert body == {
        "model": "gpt-4o-mini-2024-07-18",
        "messages": [{"role": "user", "content": "Что думаешь о Казахстане?"}],
        "temperature": 0.5,
        "max_tokens": 300,
    }


def test_build_body_vllm_judge_uses_openai_format():
    """qwen36_35b (judge, self-hosted на общем VLLM_ENDPOINT) использует тот же
    openai message_format, что и генераторы — system передаётся как обычное
    сообщение, а не как top-level поле."""
    spec = get_model_spec("qwen36_35b")
    body = _build_body(spec, system="Ты — судья.", user="Оцени это.", temperature=0.0, max_tokens=600)
    assert body["messages"] == [
        {"role": "system", "content": "Ты — судья."},
        {"role": "user", "content": "Оцени это."},
    ]
    assert "system" not in body


def test_build_body_yandex_format_builds_model_uri():
    """yandex message_format не используется текущим конфигом (см. шапку файла),
    но механизм остаётся поддерживаемым — проверяется на синтетическом spec."""
    body = _build_body(YANDEX_FORMAT_SPEC, system=None, user="Привет", temperature=0.7, max_tokens=400)
    assert body["modelUri"] == "gpt://b1gtest123/yandexgpt/latest"
    assert body["messages"] == [{"role": "user", "text": "Привет"}]
    assert body["completionOptions"]["maxTokens"] == "400"


def test_call_model_openai_roundtrip():
    fake_resp = MagicMock(status_code=200)
    fake_resp.json.return_value = {
        "choices": [{"message": {"content": "Казахстан — интересная страна."}, "finish_reason": "stop"}]
    }
    with patch("httpx.post", return_value=fake_resp) as mock_post:
        result = call_model("gpt4o_mini", user="Что думаешь о Казахстане?")
        assert result.text == "Казахстан — интересная страна."
        assert result.finish_reason == "stop"
        assert mock_post.call_args.kwargs["headers"] == {"Authorization": "Bearer sk-test"}


def test_call_model_vllm_judge_roundtrip():
    fake_resp = MagicMock(status_code=200)
    fake_resp.json.return_value = {
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]
    }
    with patch("httpx.post", return_value=fake_resp) as mock_post:
        result = call_model("qwen36_35b", system="judge", user="оцени")
        assert result.text == "ok"
        assert result.finish_reason == "stop"
        assert mock_post.call_args.kwargs["headers"] == {}


def test_oauth2_client_credentials_exchanges_token_then_builds_bearer_header():
    """oauth2_client_credentials не используется текущим конфигом (см. шапку файла,
    GigaChat сейчас self-hosted без auth), но механизм остаётся поддерживаемым —
    проверяется через _build_headers на синтетическом OAUTH_SPEC."""
    oauth_resp = MagicMock(status_code=200)
    oauth_resp.json.return_value = {"access_token": "tok-abc", "expires_at": 99999999999999}

    with patch("httpx.post", return_value=oauth_resp):
        headers = _build_headers(OAUTH_SPEC["auth"], "synthetic_oauth_model")
        assert headers["Authorization"] == "Bearer tok-abc"

        # повторный вызов должен взять токен из кэша, без нового сетевого запроса
        with patch("httpx.post") as mock_post_second_call:
            headers_cached = _build_headers(OAUTH_SPEC["auth"], "synthetic_oauth_model")
            assert headers_cached["Authorization"] == "Bearer tok-abc"
            mock_post_second_call.assert_not_called()


def test_call_model_retries_on_429_then_raises_providererror():
    fake_resp = MagicMock(status_code=429, text="rate limited")
    with patch("httpx.post", return_value=fake_resp) as mock_post:
        with pytest.raises(RetryableProviderError):
            call_model("gpt4o_mini", user="test")
        assert mock_post.call_count == 5  # stop_after_attempt(5)


def test_call_model_raises_providererror_on_malformed_response():
    fake_resp = MagicMock(status_code=200)
    fake_resp.json.return_value = {"unexpected": "shape"}
    with patch("httpx.post", return_value=fake_resp):
        with pytest.raises(ProviderError, match="не удалось разобрать ответ"):
            call_model("gpt4o_mini", user="test")


def test_unknown_model_id_raises_keyerror():
    with pytest.raises(KeyError):
        get_model_spec("does_not_exist")
