"""OpenAI-compatible HTTP client for every model in ``config/models.yaml``."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from scripts.common.config import models_config, require_env

_ENV_PLACEHOLDER = re.compile(r"\{([A-Z_][A-Z0-9_]*)\}")


class ProviderError(Exception):
    """A permanent provider or response-contract error."""


class RetryableProviderError(ProviderError):
    """A temporary provider error for which retrying is appropriate."""


@dataclass(frozen=True)
class CallResult:
    text: str
    finish_reason: str
    raw: dict


def _resolve_url(template: str) -> str:
    return _ENV_PLACEHOLDER.sub(lambda match: require_env(match.group(1)), template)


def _get_by_path(data: Any, path: str) -> Any:
    node = data
    for part in path.split("."):
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


@cache
def _all_model_specs() -> dict[str, dict[str, Any]]:
    config = models_config()
    return {
        model_id: spec for section in ("generators", "judges") for model_id, spec in (config.get(section) or {}).items()
    }


def get_model_spec(model_id: str) -> dict[str, Any]:
    specs = _all_model_specs()
    if model_id not in specs:
        raise KeyError(f"Unknown model '{model_id}'. Available: {sorted(specs)}")
    return specs[model_id]


def _build_headers(auth: dict[str, Any]) -> dict[str, str]:
    auth_type = auth["type"]
    if auth_type == "none":
        return {}
    if auth_type == "bearer_env":
        return {"Authorization": f"Bearer {require_env(auth['api_key_env'])}"}
    raise ValueError(f"Unsupported auth.type='{auth_type}'")


def _build_body(
    spec: dict[str, Any],
    system: str | None,
    user: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    body: dict[str, Any] = {
        "model": spec["model"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    body.update(spec.get("extra_body", {}))
    return body


def _log_retry(retry_state) -> None:
    error = retry_state.outcome.exception()
    delay = retry_state.next_action.sleep
    print(f"[retry] attempt {retry_state.attempt_number} failed; retrying in {delay:.1f}s: {error}")


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
    max_tokens: int = 512,
) -> CallResult:
    spec = get_model_spec(model_id)
    try:
        response = httpx.post(
            _resolve_url(spec["url"]),
            headers=_build_headers(spec["auth"]),
            json=_build_body(spec, system, user, temperature, max_tokens),
            timeout=120.0,
        )
    except (httpx.TimeoutException, httpx.TransportError) as exc:
        raise RetryableProviderError(f"[{model_id}] network error: {exc}") from exc

    if response.status_code in {429, 500, 502, 503, 504}:
        raise RetryableProviderError(f"[{model_id}] HTTP {response.status_code}: {response.text[:300]}")
    if response.status_code >= 400:
        raise ProviderError(f"[{model_id}] HTTP {response.status_code}: {response.text[:500]}")

    data = response.json()
    try:
        text = _get_by_path(data, spec["response_path"])
        finish_reason = _get_by_path(data, spec["finish_reason_path"])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ProviderError(f"[{model_id}] malformed response: {exc}; raw={data}") from exc
    return CallResult(text=str(text), finish_reason=str(finish_reason), raw=data)
