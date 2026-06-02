import logging
from typing import Optional
import redis
import ccxt

logger = logging.getLogger(__name__)

FEE_CACHE_TTL = 86400  # 1 day in seconds


def get_fee_rate(
    exchange: ccxt.Exchange,
    symbol: str,
    redis_client: Optional[redis.Redis] = None,
    default: float = 0.001,
) -> float:
    """
    Return the taker fee rate for a symbol.
    Uses Redis cache if available, otherwise fetches from the exchange.
    Falls back to `default` on any error.
    """
    cache_key = f"fee_rate:{symbol}"

    # Try Redis cache first
    if redis_client:
        try:
            cached = redis_client.get(cache_key)
            if cached is not None:
                return float(cached)
        except Exception as e:
            logger.warning(f"Redis get failed for fee rate: {e}")

    # Fetch from exchange
    try:
        fees = exchange.fetch_trading_fee(symbol)
        taker = fees.get('taker', fees.get('maker', default))
        rate = float(taker)
    except Exception as e:
        logger.warning(f"Could not fetch trading fee for {symbol}: {e}. Using default {default}")
        rate = default

    # Store in Redis
    if redis_client:
        try:
            redis_client.setex(cache_key, FEE_CACHE_TTL, str(rate))
        except Exception as e:
            logger.warning(f"Redis setex failed for fee rate: {e}")

    return rate
