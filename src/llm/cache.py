import hashlib
import json
import logging
from src.llm.llm_client import get_llm_response
from src.utils.redis_client import get_redis_client

logger = logging.getLogger(__name__)

def get_cached_llm_response(prompt: str, system_prompt: str = "", ttl: int = 300) -> str:
    """
    Get an LLM response, using Redis cache to avoid duplicate calls.
    Cache key is based on the prompt and system prompt.
    ttl: time-to-live in seconds (default 5 minutes).
    """
    redis_client = get_redis_client()
    # Create a deterministic cache key
    key_data = json.dumps({"prompt": prompt, "system": system_prompt}, sort_keys=True)
    cache_key = f"llm:{hashlib.sha256(key_data.encode()).hexdigest()}"

    # Try to get from cache
    cached = redis_client.get(cache_key)
    if cached:
        logger.debug("LLM cache hit for key %s", cache_key[:16])
        return cached

    # Not cached, call LLM
    response = get_llm_response(prompt, system_prompt)

    # Store in cache with TTL
    redis_client.setex(cache_key, ttl, response)
    logger.debug("LLM cache miss – stored response for key %s", cache_key[:16])
    return response
