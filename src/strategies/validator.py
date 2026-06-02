from .base import Signal
from typing import Dict, Any, Optional

MIN_CONFIDENCE = 0.5
VALID_STRATEGY_TYPES = {"scalping", "momentum", "mean_reversion", "breakout"}


def validate_signal(signal: Signal, market_data: Optional[Dict[str, Any]] = None) -> Signal:
    """
    Validate a trading signal.
    - If action is HOLD, return as-is.
    - If confidence < MIN_CONFIDENCE, return HOLD.
    - If strategy_type is set but not in the allowed set, return HOLD.
    Otherwise return the original signal.
    """
    if signal.action == "HOLD":
        return signal

    if signal.confidence < MIN_CONFIDENCE:
        return Signal(action="HOLD", confidence=0.0, reasoning="Confidence too low")

    if signal.strategy_type and signal.strategy_type not in VALID_STRATEGY_TYPES:
        return Signal(action="HOLD", confidence=0.0, reasoning=f"Invalid strategy type: {signal.strategy_type}")

    return signal
