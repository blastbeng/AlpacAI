import logging
from typing import Optional
import redis

logger = logging.getLogger(__name__)

def get_fee_rate(
    exchange,  # ignored (kept for signature compatibility)
    symbol: str,
    redis_client: Optional[redis.Redis] = None,
    default: float = 0.0,
) -> float:
    """Return the taker fee rate. Alpaca has zero commission for stocks/ETFs."""
    return 0.0
