"""Единая точка входа к ЛЮБОЙ модели в config/models.yaml.

Один HTTP-клиент для всех вендоров (YandexGPT, GigaChat, GLM, OpenAI, Anthropic,
самохостинг через vLLM) — вся специфика конкретного API (auth, формат тела запроса,
путь до текста в ответе) лежит декларативно в models.yaml, а не размазана по
файлам-провайдерам. Добавить новую модель = дописать запись в YAML, без нового кода.

Использование:
    from scripts.providers.client import call_model
    result = call_model("yandexgpt", system=None, user="Привет!", temperature=0.7, max_tokens=400)
"""
from __future__ import annotations

import re
import threading
import time
from functools import lru_cache
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from scripts.common.config import models_config
from scripts.common.config import require_env as _require_env

_ENV_PLACEHOLDER = re.compile(r"\{([A-Z_][A-Z0-9_]*)\}")

# access_token для oauth2_client_credentials кэшируется здесь между вызовами,
# по одному токену на model_id (на случай нескольких моделей с разными ключами).
# Lock защищает от того, чтобы N параллельных потоков (см. stage02_generate.py
# --workers) одновременно полезли за новым токеном при его истечении.
_oauth_token_cache: dict[str, tuple[str, float]] = {}
_oauth_lock = threading.Lock()


class ProviderError(Exception):
    """Невосстановимая ошибка вызова модели (после исчерпания retry)."""


class RetryableProviderError(ProviderError):
    """Временная ошибка (429/5xx/таймаут сети) — стоит повторить попытку."""


class CallResult:
    def __init__(self, text: str, finish_reason: str, raw: dict):
        self.text = text
        self.finish_reason = finish_reason
        self.raw = raw


def _resolve_url(url_template: str) -> str:
    def _sub(match: re.Match) -> str:
        return _require_env(match.group(1))

    return _ENV_PLACEHOLDER.sub(_sub, url_template)


def _get_by_path(data: Any, path: str) -> Any:
    node = data
    for part in path.split("."):
        if isinstance(node, list):
            node = node[int(part)]
        else:
            node = node[part]
    return node


@lru_cache(maxsize=None)
def _all_model_specs() -> dict[str, dict[str, Any]]:
    cfg = models_config()
    specs: dict[str, dict[str, Any]] = {}
    for section in ("generators", "judges"):
        for model_id, spec in (cfg.get(section) or {}).items():
            specs[model_id] = spec
    return specs


def get_model_spec(model_id: str) -> dict[str, Any]:
    specs = _all_model_specs()
    if model_id not in specs:
        raise KeyError(f"Модель '{model_id}' не найдена в config/models.yaml. Доступны: {sorted(specs)}")
    return specs[model_id]


def _build_headers(auth: dict[str, Any], model_id: str) -> dict[str, str]:
    auth_type = auth["type"]

    if auth_type == "none":
        return {}

    if auth_type == "bearer_env":
        return {"Authorization": f"Bearer {_require_env(auth['api_key_env'])}"}

    if auth_type == "x_api_key_env":
        headers = {"x-api-key": _require_env(auth["api_key_env"])}
        headers.update(auth.get("static_headers", {}))
        return headers

    if auth_type == "yandex_api_key":
        return {"Authorization": f"Api-Key {_require_env(auth['api_key_env'])}"}

    if auth_type == "oauth2_client_credentials":
        token = _get_oauth_token(auth, model_id)
        return {"Authorization": f"Bearer {token}"}

    raise ValueError(f"Неизвестный auth.type='{auth_type}'")


def _get_oauth_token(auth: dict[str, Any], model_id: str) -> str:
    cached = _oauth_token_cache.get(model_id)
    if cached and time.time() < cached[1] - 30:
        return cached[0]

    with _oauth_lock:
        # повторная проверка внутри лока: другой поток мог обновить токен,
        # пока этот ждал захвата лока
        cached = _oauth_token_cache.get(model_id)
        if cached and time.time() < cached[1] - 30:
            return cached[0]

        import uuid

        oauth = auth["oauth"]
        resp = httpx.post(
            oauth["url"],
            headers={
                "Authorization": f"Basic {_require_env(auth['auth_key_env'])}",
                "RqUID": str(uuid.uuid4()),
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={"scope": auth.get("scope", "GIGACHAT_API_PERS")},
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        token = _get_by_path(data, oauth["token_path"])
        expires_at_ms = _get_by_path(data, oauth["expires_at_path"])
        _oauth_token_cache[model_id] = (token, expires_at_ms / 1000)
        return token


def _build_body(
    spec: dict[str, Any],
    system: str | None,
    user: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    fmt = spec["message_format"]
    model = spec["model"]

    if fmt == "openai":
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if spec.get("extra_body"):
            body.update(spec["extra_body"])

        return body

    if fmt == "anthropic":
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system:
            body["system"] = system
        return body

    if fmt == "yandex":
        folder_id = _require_env(spec["auth"]["folder_id_env"])
        messages = []
        if system:
            messages.append({"role": "system", "text": system})
        messages.append({"role": "user", "text": user})
        return {
            "modelUri": f"gpt://{folder_id}/{model}",
            "completionOptions": {"temperature": temperature, "maxTokens": str(max_tokens)},
            "messages": messages,
        }

    raise ValueError(f"Неизвестный message_format='{fmt}'")


def _log_retry(retry_state) -> None:
    exc = retry_state.outcome.exception()
    print(f"[RETRY] попытка {retry_state.attempt_number} не удалась "
          f"({retry_state.seconds_since_start:.1f}s с начала), повтор через "
          f"{retry_state.next_action.sleep:.1f}s: {exc}")


@retry(
    retry=retry_if_exception_type(RetryableProviderError),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    before_sleep=_log_retry,
    reraise=True,
)
def call_model(
    model_id: str,
    *,
    user: str,
    system: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 400,
) -> CallResult:
    """Вызвать модель по её model_id из config/models.yaml. Один путь для всех
    генераторов и judge-моделей — различие только в конфиге, не в коде."""
    spec = get_model_spec(model_id)
    url = _resolve_url(spec["url"])
    headers = _build_headers(spec["auth"], model_id)
    body = _build_body(spec, system, user, temperature, max_tokens)

    try:
        resp = httpx.post(url, headers=headers, json=body, timeout=120.0)
    except httpx.TimeoutException as e:
        raise RetryableProviderError(f"[{model_id}] timeout после 120s: {e}") from e
    except httpx.TransportError as e:
        raise RetryableProviderError(f"[{model_id}] network error: {e}") from e

    if resp.status_code in (429, 500, 502, 503, 504):
        raise RetryableProviderError(f"[{model_id}] HTTP {resp.status_code}: {resp.text[:300]}")
    if resp.status_code >= 400:
        raise ProviderError(f"[{model_id}] HTTP {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    try:
        text = _get_by_path(data, spec["response_path"])
        finish_reason = _get_by_path(data, spec["finish_reason_path"])
    except (KeyError, IndexError, TypeError) as e:
        raise ProviderError(f"[{model_id}] не удалось разобрать ответ: {e}. Raw: {data}") from e

    return CallResult(text=text, finish_reason=str(finish_reason), raw=data)
