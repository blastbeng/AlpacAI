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
    - Validate richer strategy parameters (stop_loss_pct, take_profit_pct, etc.).
    Otherwise return the original signal.
    """
    if signal.action == "HOLD":
        return signal

    if signal.confidence < MIN_CONFIDENCE:
        return Signal(action="HOLD", confidence=0.0, reasoning="Confidence too low")

    if signal.strategy_type and signal.strategy_type not in VALID_STRATEGY_TYPES:
        return Signal(action="HOLD", confidence=0.0, reasoning=f"Invalid strategy type: {signal.strategy_type}")

    # Validate richer strategy parameters
    if signal.stop_loss_pct is not None:
        if not (0 < signal.stop_loss_pct < 1.0):
            return Signal(action="HOLD", confidence=0.0, reasoning="Invalid stop_loss_pct (must be between 0 and 1)")
    if signal.take_profit_pct is not None:
        if not (0 < signal.take_profit_pct < 10.0):
            return Signal(action="HOLD", confidence=0.0, reasoning="Invalid take_profit_pct")
    if signal.trailing_stop and signal.trailing_stop_distance_pct is None:
        return Signal(action="HOLD", confidence=0.0, reasoning="trailing_stop requires trailing_stop_distance_pct")
    if signal.trailing_stop_distance_pct is not None and not (0 < signal.trailing_stop_distance_pct < 1.0):
        return Signal(action="HOLD", confidence=0.0, reasoning="Invalid trailing_stop_distance_pct")
    if signal.position_size_fraction is not None and not (0 < signal.position_size_fraction <= 1.0):
        return Signal(action="HOLD", confidence=0.0, reasoning="Invalid position_size_fraction")

    return signal
