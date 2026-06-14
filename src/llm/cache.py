import hashlib
import json
import logging
from typing import Optional
from src.config.settings import settings
from src.utils.redis_client import get_redis_client

logger = logging.getLogger(__name__)

def get_cached_llm_response(
    prompt: str,
    system_prompt: str = "",
    ttl: int = 300,
    market_hash: str = None,
    model_type: str = "actuator",
) -> Optional[str]:
    """
    Get an LLM response, using Redis cache to avoid duplicate calls.
    If market_hash is provided, the cache key is based on that hash
    (representing the market snapshot). Otherwise, the key is based on
    the prompt and system prompt.
    ttl: time-to-live in seconds (default 5 minutes).
    model_type: "mind" for complex reasoning, "actuator" for fast time‑critical decisions.
    """
    redis_client = get_redis_client()

    # Determine effective provider for this role
    if model_type == "mind":
        provider = settings.LLM_MIND_PROVIDER or settings.LLM_PROVIDER
    else:
        provider = settings.LLM_ACTUATOR_PROVIDER or settings.LLM_PROVIDER

    if provider == "openai":
        model = settings.OPENAI_MIND_MODEL if model_type == "mind" else settings.OPENAI_ACTUATOR_MODEL
        base_url = (settings.OPENAI_MIND_BASE_URL or settings.OPENAI_BASE_URL) if model_type == "mind" else (settings.OPENAI_ACTUATOR_BASE_URL or settings.OPENAI_BASE_URL)
        api_key = (settings.OPENAI_MIND_API_KEY or settings.OPENAI_API_KEY) if model_type == "mind" else (settings.OPENAI_ACTUATOR_API_KEY or settings.OPENAI_API_KEY)
    else:
        model = settings.OLLAMA_MIND_MODEL if model_type == "mind" else settings.OLLAMA_ACTUATOR_MODEL
        base_url = (settings.OLLAMA_MIND_BASE_URL or settings.OLLAMA_BASE_URL) if model_type == "mind" else (settings.OLLAMA_ACTUATOR_BASE_URL or settings.OLLAMA_BASE_URL)
        api_key = (settings.OLLAMA_MIND_API_KEY or settings.OLLAMA_API_KEY) if model_type == "mind" else (settings.OLLAMA_ACTUATOR_API_KEY or settings.OLLAMA_API_KEY)

    if market_hash:
        cache_key = f"llm:{model_type}:market:{market_hash}"
    else:
        # Create a deterministic cache key from the prompts and model type
        key_data = json.dumps(
            {"prompt": prompt, "system": system_prompt, "model_type": model_type}, sort_keys=True
        )
        cache_key = f"llm:{hashlib.sha256(key_data.encode()).hexdigest()}"

    # Try to get from cache
    cached = redis_client.get(cache_key)
    if cached:
        logger.debug("LLM cache hit for key %s", cache_key[:32])
        return cached

    # Not cached, call the appropriate raw LLM function
    if provider == "openai":
        from src.llm.llm_client import _get_openai_response
        response = _get_openai_response(prompt, system_prompt, model=model, base_url=base_url, api_key=api_key)
    else:
        from src.llm.llm_client import _get_ollama_response
        response = _get_ollama_response(prompt, system_prompt, model=model, base_url=base_url, api_key=api_key)

    if response is None:
        logger.warning("LLM returned None response; not caching.")
        return None

    # Store in cache with TTL
    redis_client.setex(cache_key, ttl, response)
    logger.debug("LLM cache miss – stored response for key %s", cache_key[:32])
    return response


def _stringify_keys(obj):
    """Recursively convert all dict keys to strings for JSON-safe sorting."""
    if isinstance(obj, dict):
        return {str(k): _stringify_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_stringify_keys(item) for item in obj]
    return obj


def compute_market_hash(data: dict) -> str:
    """Return a SHA-256 hex digest of the JSON-serialised market data."""
    # sort_keys ensures deterministic output
    safe_data = _stringify_keys(data)
    serialized = json.dumps(safe_data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()
