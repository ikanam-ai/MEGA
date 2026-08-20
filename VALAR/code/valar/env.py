import os


def valuellama_api_base_url() -> str:
    return os.environ.get("VALAR_VALUELLAMA_API_BASE_URL", "http://127.0.0.1:8000/v1")


def valuellama_api_key() -> str:
    return os.environ.get("VALAR_VALUELLAMA_API_KEY", "EMPTY")


def served_model_name() -> str:
    return os.environ.get("VALAR_SERVED_MODEL_NAME", "ValueLlama-3-8B")


def datasets_root() -> str:
    return os.environ.get("VALAR_DATASETS_ROOT", "../../datasets")


def results_root() -> str:
    return os.environ.get("VALAR_RESULTS_ROOT", "results")
