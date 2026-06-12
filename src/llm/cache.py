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
) -> Optional[str]:
    """
    Get an LLM response, using Redis cache to avoid duplicate calls.
    If market_hash is provided, the cache key is based on that hash
    (representing the market snapshot). Otherwise, the key is based on
    the prompt and system prompt.
    ttl: time-to-live in seconds (default 5 minutes).
    """
    redis_client = get_redis_client()

    if market_hash:
        cache_key = f"llm:market:{market_hash}"
    else:
        # Create a deterministic cache key from the prompts
        key_data = json.dumps(
            {"prompt": prompt, "system": system_prompt}, sort_keys=True
        )
        cache_key = f"llm:{hashlib.sha256(key_data.encode()).hexdigest()}"

    # Try to get from cache
    cached = redis_client.get(cache_key)
    if cached:
        logger.debug("LLM cache hit for key %s", cache_key[:32])
        return cached

    # Not cached, call the appropriate raw LLM function
    if settings.LLM_PROVIDER == "openai":
        from src.llm.llm_client import _get_openai_response
        response = _get_openai_response(prompt, system_prompt)
    else:
        from src.llm.llm_client import _get_ollama_response
        response = _get_ollama_response(prompt, system_prompt)

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
