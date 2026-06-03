import json
from .base import Signal, LLMStrategy


def parse_llm_response(response_text: str) -> Signal:
    """
    Parse the LLM's JSON response into a Signal.
    If parsing fails, returns a HOLD signal with zero confidence.
    """
    try:
        data = json.loads(response_text)
        action = data.get("action", "HOLD").upper()
        if action not in ("BUY", "SELL", "HOLD"):
            action = "HOLD"
        confidence = float(data.get("confidence", 0.0))
        reasoning = data.get("reasoning", "")
        strategy = data.get("strategy")
        strategy_type = None
        parameters = None
        if isinstance(strategy, dict):
            strategy_type = strategy.get("type")
            parameters = strategy.get("parameters")

        # Extract richer strategy parameters
        stop_loss_pct = None
        take_profit_pct = None
        trailing_stop = False
        trailing_stop_distance_pct = None
        position_size_fraction = None

        if isinstance(parameters, dict):
            stop_loss_pct = parameters.get("stop_loss_pct")
            take_profit_pct = parameters.get("take_profit_pct")
            trailing_stop = bool(parameters.get("trailing_stop", False))
            trailing_stop_distance_pct = parameters.get("trailing_stop_distance_pct")
            position_size_fraction = parameters.get("position_size_fraction")

        return Signal(
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            strategy_type=strategy_type,
            parameters=parameters,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            trailing_stop=trailing_stop,
            trailing_stop_distance_pct=trailing_stop_distance_pct,
            position_size_fraction=position_size_fraction,
        )
    except (json.JSONDecodeError, ValueError, TypeError):
        return Signal(action="HOLD", confidence=0.0, reasoning="Failed to parse LLM response")


def create_strategy_from_llm(response_text: str) -> LLMStrategy:
    """
    Parse the LLM response and return an LLMStrategy instance.
    """
    signal = parse_llm_response(response_text)
    return LLMStrategy(signal)
