from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class Signal:
    action: str  # "BUY", "SELL", "HOLD"
    confidence: float
    reasoning: str
    strategy_type: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    # Richer strategy parameters
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    trailing_stop: bool = False
    trailing_stop_distance_pct: Optional[float] = None
    position_size_fraction: Optional[float] = None


class Strategy:
    """Abstract base for trading strategies."""
    def generate_signal(self, market_data: Dict[str, Any]) -> Signal:
        raise NotImplementedError


class LLMStrategy(Strategy):
    """A strategy that wraps a pre-computed LLM decision."""
    def __init__(self, signal: Signal):
        self.signal = signal

    def generate_signal(self, market_data: Dict[str, Any]) -> Signal:
        # For now, returns the static decision; can be extended later.
        return self.signal
