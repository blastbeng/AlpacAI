import json
from typing import List, Dict, Any

SYSTEM_PROMPT = """You are a professional cryptocurrency trading bot assistant. Your task is to analyze market data and provide trading decisions in strict JSON format. Do not include any text outside the JSON. Always output valid JSON.

When asked to select coins, return a JSON array of trading pair symbols (e.g., ["BTC/USDT", "ETH/USDT"]). Choose coins that are likely to be profitable based on recent price action, volume, and volatility. Prefer coins with high liquidity and clear trends.

When asked to generate a strategy for a specific coin, return a JSON object with the following structure:
{
  "action": "BUY" | "SELL" | "HOLD",
  "confidence": 0.0 to 1.0,
  "reasoning": "short explanation",
  "strategy": {
    "type": "scalping" | "momentum" | "mean_reversion" | "breakout",
    "parameters": {
      // strategy-specific parameters
    }
  }
}
If action is BUY or SELL, include a strategy. If HOLD, strategy can be null.
"""

def build_coin_selection_prompt(
    available_pairs: List[str],
    current_coins: List[str],
    max_coins: int,
    base_currency: str,
    tickers: Dict[str, Any]
) -> str:
    """Build a prompt to ask the LLM which coins to trade."""
    # Summarize tickers for the prompt (limit to avoid huge prompts)
    ticker_summary = {}
    for symbol in available_pairs[:50]:  # limit to first 50 to keep prompt size manageable
        if symbol in tickers:
            t = tickers[symbol]
            ticker_summary[symbol] = {
                "last": t.get("last"),
                "change_24h": t.get("percentage"),
                "volume": t.get("quoteVolume"),
            }

    prompt = f"""Current base currency: {base_currency}
Maximum number of coins to trade: {max_coins}
Currently traded coins: {json.dumps(current_coins)}

Available trading pairs (sample):
{json.dumps(ticker_summary, indent=2)}

Based on the market data, select up to {max_coins} coins to trade. Return a JSON array of symbols. Prefer coins with high volume and positive momentum. You may keep some current coins if they are still promising, or replace them."""
    return prompt

def build_strategy_prompt(
    symbol: str,
    ticker: Dict[str, Any],
    order_book: Dict[str, Any],
    balance: Dict[str, float],
    open_positions: List[Dict[str, Any]]
) -> str:
    """Build a prompt to generate a trading strategy for a specific coin."""
    prompt = f"""Symbol: {symbol}
Current ticker: {json.dumps(ticker)}
Order book (top 5 levels): {json.dumps(order_book)}
Current balances: {json.dumps(balance)}
Open positions: {json.dumps(open_positions)}

Based on the above, decide whether to BUY, SELL, or HOLD. Provide a strategy if action is BUY or SELL. Return a JSON object as specified."""
    return prompt
