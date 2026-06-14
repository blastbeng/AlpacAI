import json
import re
from .base import Signal, LLMStrategy


def parse_llm_response(response_text: str) -> Signal:
    """
    Parse the LLM's JSON response into a Signal.
    Supports JSON wrapped in ```json ... ``` code blocks or raw JSON.
    Raises ValueError if the response cannot be parsed as valid JSON.
    """
    try:
        # Try to extract JSON from a markdown code block first
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))
        else:
            try:
                data = json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback: try to extract the first JSON object from the text
                start = response_text.find('{')
                end = response_text.rfind('}')
                if start != -1 and end != -1 and end > start:
                    json_str = response_text[start:end+1]
                    data = json.loads(json_str)
                else:
                    raise ValueError("No JSON object found in LLM response")

        action = data.get("action", "HOLD").upper()
        if action not in ("BUY", "SELL", "HOLD"):
            action = "HOLD"
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
        reasoning = data.get("reasoning", "")

        strategy = data.get("strategy")
        strategy_type = None
        strategy_params = None
        if isinstance(strategy, dict):
            strategy_type = strategy.get("type")
            strategy_params = strategy.get("parameters")

        risk_level = data.get("risk_level")
        if risk_level not in ("low", "medium", "high"):
            risk_level = "medium"

        indicator_config = data.get("indicator_config")
        if not isinstance(indicator_config, dict):
            indicator_config = None

        backtest_summary = data.get("backtest_summary")
        if not isinstance(backtest_summary, str):
            backtest_summary = None

        # --- dynamic trading parameters ---
        stop_loss = data.get("stop_loss")
        take_profit = data.get("take_profit")
        position_size = data.get("position_size")
        if position_size is not None:
            position_size = max(0.0, min(1.0, float(position_size)))
        trailing_stop = bool(data.get("trailing_stop", False))
        max_hold_minutes = data.get("max_hold_minutes")
        reason = data.get("reason", "")

        # --- entry condition ---
        entry_condition_raw = data.get("entry_condition")
        entry_condition = None
        if isinstance(entry_condition_raw, dict):
            etype = entry_condition_raw.get("type")
            valid_types = ("limit_price", "rsi_threshold", "order_book_depth", "delay", "indicator_combo")
            if etype in valid_types:
                if etype == "limit_price" and "price" in entry_condition_raw and "timeout_seconds" in entry_condition_raw:
                    entry_condition = entry_condition_raw
                elif etype == "rsi_threshold" and "rsi_below" in entry_condition_raw and "timeout_seconds" in entry_condition_raw:
                    entry_condition = entry_condition_raw
                elif etype == "order_book_depth" and "min_ask_volume" in entry_condition_raw and "timeout_seconds" in entry_condition_raw:
                    entry_condition = entry_condition_raw
                elif etype == "delay" and "delay_seconds" in entry_condition_raw:
                    entry_condition = entry_condition_raw
                elif etype == "indicator_combo" and isinstance(entry_condition_raw.get("conditions"), list) and len(entry_condition_raw["conditions"]) > 0 and "timeout_seconds" in entry_condition_raw:
                    entry_condition = entry_condition_raw

        return Signal(
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            strategy_type=strategy_type,
            strategy_params=strategy_params,
            risk_level=risk_level,
            indicator_config=indicator_config,
            backtest_summary=backtest_summary,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            trailing_stop=trailing_stop,
            max_hold_minutes=max_hold_minutes,
            reason=reason,
            entry_condition=entry_condition,
        )
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        raise ValueError(f"Failed to parse LLM response as valid JSON: {e}") from e


def create_strategy_from_llm(response_text: str) -> LLMStrategy:
    """
    Parse the LLM response and return an LLMStrategy instance.
    """
    signal = parse_llm_response(response_text)
    return LLMStrategy(signal)
