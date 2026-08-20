from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GenerationConfig:
    model_id: str
    api_base_url: str
    api_key: str
    served_model_name: str
    temperature: float = 0.0
    seed: int = 42
    max_new_tokens: int = 256
    parallelism: int = 8
    timeout_seconds: int = 120
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
