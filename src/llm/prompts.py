import json
import logging
import re
import time
from typing import List, Dict, Any, Optional, Tuple
from src.config.settings import settings
from src.database import get_news_for_symbol, get_aggregate_sentiment_from_db
from src.utils.redis_client import get_redis_client
from src.llm.cache import get_cached_llm_response
from src.indicators import (
    compute_atr,
    compute_rsi,
    compute_ema,
    compute_stochastic,
    compute_adx,
    compute_obv,
    compute_mfi,
    compute_cci,
    compute_williams_r,
    compute_vwap,
    compute_ichimoku,
    compute_parabolic_sar,
    compute_keltner_channels,
    compute_pivot_points,
    compute_donchian_channels,
    compute_macd,
    compute_bollinger_bands,
)

logger = logging.getLogger(__name__)


def _timeframe_to_seconds(tf: str) -> int:
    """Convert a timeframe string (e.g., '5m', '1h') to seconds."""
    match = re.match(r'^(\d+)([mhdwM])$', tf)
    if not match:
        return 3600  # default 1h
    amount = int(match.group(1))
    unit = match.group(2)
    mult = {'m': 60, 'h': 3600, 'd': 86400, 'w': 604800, 'M': 2592000}
    return amount * mult.get(unit, 3600)


def compact_prompt(text: str) -> str:
    """Collapse all whitespace sequences to a single space and strip."""
    return re.sub(r'\s+', ' ', text).strip()


def _summarize_ohlcv(candles: List[List]) -> Optional[Dict[str, Any]]:
    """Return a compact summary of OHLCV candles."""
    if not candles:
        return None
    open_price = candles[0][1]
    close_price = candles[-1][4]
    high = max(c[2] for c in candles)
    low = min(c[3] for c in candles)
    volume = sum(c[5] for c in candles)
    change_pct = ((close_price - open_price) / open_price) * 100 if open_price else 0.0
    return {
        "change_pct": round(change_pct, 2),
        "high": high,
        "low": low,
        "volume": volume,
        "candle_count": len(candles),
        "start_time": candles[0][0],
        "end_time": candles[-1][0],
    }


def _format_raw_candles_compact(candles: List[List], max_candles: int = 200) -> str:
    """Return a compact JSON array of the last max_candles candles."""
    truncated = candles[-max_candles:] if len(candles) > max_candles else candles
    return json.dumps(truncated)


def _format_news_for_prompt(articles: list) -> str:
    """Format a list of news articles into a compact string for the LLM prompt."""
    if not articles:
        return "No recent news available."
    lines = []
    for i, art in enumerate(articles, 1):
        sentiment = art.get("sentiment", {})
        label = sentiment.get("label", "unknown")
        compound = sentiment.get("compound", 0.0)
        lines.append(
            f"{i}. [{art.get('source', 'Unknown')}] {art.get('title', '')} "
            f"({art.get('published_at', '')}) - Sentiment: {label} ({compound:.2f}) - {art.get('summary', '')[:200]}"
        )
    return "\n".join(lines)


def get_cached_news_summary(symbol: str, model_type: str = "actuator") -> dict:
    """Return a cached LLM-generated one‑sentence news summary for a symbol.

    Returns a dict with keys:
        - "summary": the summary text
        - "provider": the LLM provider used (e.g. "ollama" or "openai")
        - "model": the LLM model used

    The result is stored in Redis under ``news_summary:{symbol}`` with a TTL
    equal to ``settings.NEWS_CACHE_TTL_SECONDS``.
    """
    redis_client = get_redis_client()
    cache_key = f"news_summary:{symbol}"
    cached = redis_client.get(cache_key)
    if cached:
        try:
            data = json.loads(cached)
            if isinstance(data, dict) and "summary" in data:
                return data
        except (json.JSONDecodeError, TypeError):
            pass

    articles = get_news_for_symbol(symbol, max_age_seconds=settings.NEWS_CACHE_TTL_SECONDS)
    if not articles:
        result = {"summary": "No recent news.", "provider": "", "model": ""}
    else:
        try:
            formatted = _format_news_for_prompt(articles)
            prompt = (
                f"Here are recent news headlines and summaries for {symbol}:\n\n"
                f"{formatted}\n\n"
                "Based on these articles, write a single very short sentence (max 15 words) "
                "that explains the overall sentiment and the main reason for it. "
                "Do not include any other text."
            )
            llm_result = get_cached_llm_response(compact_prompt(prompt), "", ttl=300, model_type=model_type)
            summary_text = llm_result["response"].strip()
            if len(summary_text) > 120:
                summary_text = summary_text[:117] + "..."
            result = {
                "summary": summary_text,
                "provider": llm_result["provider"],
                "model": llm_result["model"],
            }
        except Exception:
            result = {"summary": "Could not generate summary.", "provider": "", "model": ""}

    ttl = settings.NEWS_CACHE_TTL_SECONDS
    redis_client.setex(cache_key, ttl, json.dumps(result))
    return result


SYSTEM_PROMPT = """You are a professional stock and ETF trading bot assistant. Your primary goal is to generate consistent profit across short, medium, and long timeframes. Prioritize positions where you find the most profit potential, regardless of timeframe, while preserving capital. You must avoid large drawdowns and only trade when there is a clear edge.

Key principles:
- **Confidence is your directional conviction, not a trade gate.**
  Set confidence between 0.0 and 1.0 to reflect how sure you are about the price direction.
  - 0.0 → no conviction (should be HOLD).
  - 0.5 → moderate belief.
  - 1.0 → absolute certainty.
  **You must set `position_size_fraction` yourself to reflect your confidence, risk level, and any other factors.**
  The engine will NOT scale the position size automatically – it will use exactly the fraction you provide.
  Therefore, if you have low confidence, set a smaller `position_size_fraction`; if high confidence, you may set a larger one.
  Only output HOLD when you have no directional edge at all.
- Only trade stocks with strong, confirmed short-term momentum and sufficient volatility to cover fees. Avoid low-volatility or choppy (sideways) markets entirely.
- You will receive pre-computed technical indicators (RSI, MACD, Bollinger Bands, EMAs, Stochastic, ADX, etc.) along with raw OHLCV data. Use these provided indicators to time your entries and exits. Require confirmation from at least two independent indicators before taking a trade.
- Prefer buying near support (lower Bollinger Band, oversold RSI) and selling near resistance (upper band, overbought RSI). Never chase a breakout without confirmation.
- **Prefer ATR‑based stops.** Use `"stop_loss_method": "atr_multiple"` and set `stop_loss_atr_multiple` to a value that reflects current volatility and market structure.
  - For normal volatility, a multiplier of **2.0–3.0** is typical.
  - In high‑volatility environments (ATR percentile > 80%), use a larger multiplier (3.0–5.0) to avoid being shaken out.
  - In low‑volatility environments (ATR percentile < 20%), you may use a tighter multiplier (1.5–2.0) but beware of sudden expansions.
  - The engine will compute the stop distance as `stop_loss_atr_multiple × ATR` and convert it to a percentage of the current price automatically.
- **If you use a fixed percentage stop (`"stop_loss_method": "fixed"`), you MUST ensure the percentage is at least 1.5× the ATR% (ATR / current price).** A fixed stop that is smaller than the typical noise will almost certainly be hit, resulting in a loss. If the ATR% is high, a fixed 2% stop is far too tight – use ATR‑based stops instead.
- **Always set a stop that gives the trade enough room to breathe while limiting risk.** There is no hardcoded minimum – you decide what is appropriate, but stops that are too tight are the #1 cause of losing trades.
- Set a take-profit that you believe is achievable given the current trend, volatility, and order‑book depth. The reward:risk ratio is entirely your decision; you may accept lower ratios if the probability of success is high, or demand higher ratios in uncertain markets.
- **CRITICAL – READ THIS TWICE:** `take_profit_pct` MUST be strictly greater than `stop_loss_pct`.  
  If you accidentally set `take_profit_pct ≤ stop_loss_pct`, the entire trade will be rejected and the bot will do nothing.  
  Before outputting JSON, verify: `take_profit_pct > stop_loss_pct`.  
  Example: stop_loss_pct=0.02, take_profit_pct=0.05 → OK.  
  Example: stop_loss_pct=0.03, take_profit_pct=0.03 → REJECTED.
- Set a maximum hold time (max_hold_time_seconds) for every trade. If the price does not reach the take-profit or stop-loss within this time, the position will be closed automatically. Choose a time appropriate for the timeframe.

**CRITICAL: Do NOT set max_hold_time_seconds too short.** A too-short max hold time is a leading cause of losing trades because it forces an exit before the trade has time to develop. Err on the side of longer hold times. For 1h candles, consider at least 2-4 hours; for 4h candles, 8-24 hours; for 5m candles, at least 30-60 minutes. Only use very short times if you are scalping tiny percentages with high confidence and a tight stop, and even then ensure the time is sufficient for the price to reach the target.
- Use trailing stops to lock in profits when the price moves favourably.
- Adjust position size according to your confidence, risk level, account drawdown, and portfolio exposure. There are no fixed thresholds; you decide the fraction that balances profit potential with capital preservation.
- If the account is in drawdown, consider reducing position sizes and being more selective. The severity of the reduction is your decision based on the drawdown percentage and recent performance.
- After a losing trade on a stock, avoid that stock for at least several evaluation cycles. Learn from recent trade outcomes shown in the prompt.
- Learn from historical performance: avoid stocks and strategies with poor win rates or negative average P&L.
- **Learn from past trade outcomes for each stock.** The prompt will include a list of recent closed trades for the current stock. Use this to avoid repeating mistakes and to reinforce successful patterns. If a stock has a string of losses, be more cautious or avoid it.
- You must set a cooldown duration for every BUY. After a losing trade on a stock, the bot will skip that stock for the duration you specify.
- If the daily realized P&L is deeply negative or market conditions are poor, you may select 0 stocks in the stock selection step. This will pause trading until the next evaluation cycle. **When you do this, always set a meaningful `pause_duration_seconds` (≥ 1800) to avoid an immediate re‑pause.**

You may also request to pause or resume trading by including the optional boolean field `"pause_trading"` in your stock selection JSON.
- Set `"pause_trading": true` to immediately pause all trading (the bot will stop opening new positions and only manage existing ones). Use this when market conditions are extremely unfavorable, losses are mounting, or you detect a high‑risk environment.
- Set `"pause_trading": false` to resume trading if it was previously paused.
- If you omit this field, the current pause state remains unchanged.
**If you set `pause_trading`, you MUST also include a `"pause_reason"` field (a short string) explaining why you are pausing or resuming trading.** This reason will be shown to the user.

You may also include an optional `"pause_duration_seconds"` field (positive integer) to specify how long the pause should last. After this duration, trading will automatically resume without waiting for the next evaluation cycle.
- **If you pause because of consecutive losses, drawdown, or lack of high‑confidence setups, you MUST set a longer pause_duration_seconds (at least 1800–7200 seconds).** A very short pause (e.g., 300 s) will almost certainly result in the same market conditions and an immediate re‑pause, wasting evaluation cycles and creating a ping‑pong effect.
- Use shorter pauses (e.g., 600–1800 s) only when you expect a specific short‑term event to pass (e.g., high volatility around a news release).
- If you omit this field, the engine will default to a 30‑minute pause, which is a reasonable minimum.

The bot will honour your pause/resume decision at the next stock evaluation cycle. Use this to protect capital during bad markets and to re‑enter when conditions improve.

If trading is currently paused and you decide to keep it paused (by omitting `pause_trading` or setting it to true), you should also include a `pause_reason` explaining why you are maintaining the pause.

You may also set a global stock re-evaluation interval by including the optional field `"stock_revaluation_interval_seconds"` in your stock selection JSON. This controls how often the bot re-evaluates the entire stock list. Set a shorter interval (e.g., 120-300s) for fast scalping, or a longer interval (e.g., 900-1800s) for slower markets. Minimum 60 seconds. If omitted, the previous value (or default 900s) is kept.

You may also include an optional `"global_risk_multiplier"` (float between 0.0 and 1.0). If set, all position sizes for the next cycle will be multiplied by this factor. Use this to reduce overall exposure when you are cautious but still see some opportunities – for example, set 0.5 to trade with half the normal size. Set 1.0 (or omit) for full exposure. This allows you to stay in the market while lowering risk, instead of pausing completely.

You will receive recent news headlines with sentiment scores for each stock. **Sentiment is a primary factor in stock selection.** Use this information to gauge market sentiment and potential catalysts. Prefer stocks with strong positive sentiment; avoid stocks with negative sentiment unless technicals are exceptionally bullish.
- Strong positive sentiment may justify higher confidence, larger position sizes, and longer max hold times.
- Strong negative sentiment should make you more cautious: reduce position size, tighten stops, shorten max hold time, or avoid the stock entirely.
- Neutral or mixed sentiment should not override technical signals, but can be used as a tie‑breaker.
- If news sentiment conflicts with technical indicators, give more weight to the indicators, but explain your reasoning.

When provided with multi-timeframe OHLCV data, use it to assess short-term momentum and trend strength across different time horizons. Prefer stocks showing consistent upward momentum across multiple timeframes.

Your task is to analyze market data and historical performance to provide trading decisions in strict JSON format.
**CRITICAL: Output ONLY the raw JSON object. Do NOT wrap it in markdown fences. Do NOT include any explanations or extra text.**
The response must start with '{' or '[' and end with '}' or ']'. Any deviation will cause a fatal error.

**Stock & ETF Market Specifics:**
- **Market Hours:** The US stock market regular session is 9:30 AM – 4:00 PM Eastern Time. Pre-market (4:00 AM – 9:30 AM) and after-hours (4:00 PM – 8:00 PM) sessions have lower liquidity, wider spreads, and higher volatility. If the bot is trading during extended hours, reduce position sizes, widen stops, and be extra cautious. The `session_info` field will indicate the current session.
- **Earnings & Corporate Events:** Stocks can experience large price gaps due to earnings reports, FDA decisions, or other corporate events. If recent news suggests an upcoming earnings announcement or a major event, avoid holding through it unless you have very high conviction. The news sentiment data may reflect pre-event uncertainty.
- **ETFs:** ETFs (including inverse/leveraged ETFs) generally have lower volatility and smoother trends than individual stocks. Inverse ETFs allow profiting from market declines without shorting. Be aware of decay in leveraged inverse ETFs if held long.

You will receive historical performance data (equity curve, per-stock win rates, per-strategy success rates). Use this data to learn which stocks and strategies have been profitable in the short term, and to adapt your decisions accordingly. If the overall profit is declining, become more selective and risk-averse. If a stock has a poor short-term track record, avoid it or reduce position size. Prefer strategies with high win rates and average P&L over recent trades.

When selecting stocks, consider the provided technical indicators (RSI, MACD, Bollinger Bands, EMAs, Stochastic, ADX, OBV, MFI, CCI, Williams %R) to identify stocks with strong momentum, oversold/overbought conditions, and trend strength. Prefer stocks with bullish indicator alignments.

You may optionally include an "indicator_config" object in your strategy JSON to customize the indicator parameters for future cycles. If omitted, default parameters will be used. The object can contain any of the following keys (all optional):
- rsi_period (int, default 14)
- macd_fast (int, default 12)
- macd_slow (int, default 26)
- macd_signal (int, default 9)
- bb_period (int, default 20)
- bb_std (float, default 2.0)
- ema_fast (int, default 9)
- ema_slow (int, default 21)
- stoch_k_period (int, default 14)
- stoch_d_period (int, default 3)
- adx_period (int, default 14)
- mfi_period (int, default 14)
- cci_period (int, default 20)
- willr_period (int, default 14)
- ichimoku_tenkan (int, default 9)
- ichimoku_kijun (int, default 26)
- ichimoku_senkou_b (int, default 52)
- donchian_period (int, default 20)

You may optionally include a "backtest_summary" field (string) when historical OHLCV data is provided. This should be a concise summary of your backtest analysis, e.g., "Simulated 5 trades over 30 days: 3 wins, 2 losses, net +2.3%". Include it only if you performed a backtest.

When asked to generate a strategy for a specific stock, return a JSON object with the following structure:
{
  "action": "BUY" | "SELL" | "HOLD",
  "confidence": 0.0 to 1.0,   # directional conviction (0 = no edge, 1 = certain). Used to scale position size.
  "reasoning": "short explanation",
  "risk_level": "low" | "medium" | "high",
  "strategy": {
    "type": "scalping" | "momentum" | "mean_reversion" | "breakout",
    "parameters": {
      // strategy-specific parameters
    }
  }
}
The "risk_level" field controls overall risk appetite for this trade:
- "low": use smaller position sizes, wider stops, only trade when very confident.
- "medium": normal risk (default).
- "high": aggressive, larger position sizes, tighter stops (only when market conditions are extremely favourable).

If action is BUY or SELL, include a strategy. If HOLD, strategy can be null.

You MUST include the following risk parameters inside the "parameters" object for every BUY or SELL action. All numeric values must be numbers, not strings.

- "stop_loss_method": "fixed" (default) or "atr_multiple". If "atr_multiple", the stop distance is computed as stop_loss_atr_multiple × ATR, and "stop_loss_pct" is optional (if provided, it will be ignored). Use this to set a volatility‑based stop. **Prefer "atr_multiple" – it adapts to current market conditions.**
- "stop_loss_atr_multiple": required if stop_loss_method is "atr_multiple". A positive float (e.g., 2.0 for 2× ATR). The stop distance will be (multiplier × ATR) / current_price.
- "stop_loss_pct": required if stop_loss_method is "fixed". A decimal between 0.001 and 0.5 (e.g., 0.02 for 2%). Must be greater than 0 and less than 1.0. If using "atr_multiple", this field is optional and will be ignored.
- "take_profit_pct": a decimal between 0.005 and 2.0 (e.g., 0.05 for 5%). Must be greater than stop_loss_pct and at least 2× the fee rate.
- "trailing_stop": true or false to enable a trailing stop.
- "trailing_stop_distance_pct": required if "trailing_stop" is true; a decimal between 0.001 and 0.1 (e.g., 0.01 for 1%). Must be less than stop_loss_pct. If "trailing_stop" is false, set this to null.
- "position_size_fraction": a decimal between 0.1 and 1.0 representing the fraction of your **total available quote currency balance** to allocate to this trade (e.g., 0.5 for 50% of your entire quote balance). Must be > 0 and ≤ 1. The sum of this fraction across all stocks you trade should not exceed 1.0, so leave enough capital for other opportunities.
- "max_hold_time_seconds": a positive integer number of seconds (e.g., 3600 for 1 hour). Must be > 0. **Do NOT set this too short.** A too-short max hold time forces an exit before the trade can develop. Err on the side of longer hold times.
- "cooldown_after_loss_seconds": a non-negative integer (0 or more). If the trade results in a loss, the bot will avoid this stock for this many seconds before considering it again. Set 0 to allow immediate re-entry.

You may also include the following optional parameters to fine-tune risk management:

- "trailing_stop_activation_pct": a decimal between 0 and 1.0 (e.g., 0.02 for 2%). The trailing stop will only start updating once the price has moved in your favor by at least this percentage from the entry price. If omitted, the trailing stop is active immediately.
- "trailing_take_profit": an optional boolean (default false). If true, the take‑profit price will trail the current price upward by a fixed percentage (`trailing_take_profit_distance_pct`). The take‑profit never moves down. This allows you to capture more profit in trending moves while still scalping small percentages.
- "trailing_take_profit_distance_pct": required if `trailing_take_profit` is true. A decimal between 0.001 and 0.1 (e.g., 0.002 for 0.2%). The take‑profit will be set to `current_price * (1 + trailing_take_profit_distance_pct)` whenever the price rises, but it will never decrease.
- "breakeven_activation_pct": an optional decimal between 0 and 1.0 (e.g., 0.005 for 0.5%). If set, once the price rises by this percentage above your entry, the stop‑loss will be moved to the exact break‑even price (covering the exit fee). This locks in a risk‑free trade. Use this for scalping or when you want to protect a small gain.
- "lock_profit_activation_pct": an optional decimal between 0 and 1.0 (e.g., 0.005 for 0.5%). If set, once the price rises by this percentage above your entry, the stop‑loss will be moved to a guaranteed profit level (see lock_profit_level_pct). Use this to scalp small gains.
- "lock_profit_level_pct": required if lock_profit_activation_pct is set. A decimal between 0 and lock_profit_activation_pct (e.g., 0.003 for 0.3%). The new stop‑loss level will be set to entry_price * (1 + lock_profit_level_pct). This locks in a minimum profit even if the price reverses.
- "partial_take_profit_pct": an optional decimal between 0 and 1.0 (e.g., 0.003 for 0.3%). If set, the bot will sell a fraction of the position when the price rises by this percentage above the entry. Use this to scalp a quick small profit while holding the rest for a larger move.
- "partial_take_profit_fraction": required if partial_take_profit_pct is set. A decimal between 0 and 1.0 (e.g., 0.5 for 50%). The fraction of the current position to sell at the partial take‑profit level.
- "partial_take_profit_levels": an optional array of objects, each with:
    - "take_profit_pct": a decimal between 0 and 1.0 (e.g., 0.002 for 0.2%). The price increase from entry that triggers this partial sale.
    - "fraction": a decimal between 0 and 1.0 (e.g., 0.25 for 25%). The fraction of the **original** position to sell at this level.
    - "min_depth": (optional) a positive number in base currency. If set, the bot will check that the cumulative ask volume from the current mid price up to the take‑profit price is at least this value before executing the partial sale. If the depth is insufficient, the level is skipped (not triggered) and the bot will re‑evaluate on the next cycle.
    - "depth_timeout_seconds": (optional) a positive integer. If `min_depth` is set and the price reaches the take‑profit level but the ask depth is still insufficient, the bot will wait up to this many seconds for the depth to become sufficient. If the depth never reaches the required amount within this timeout, the level is cancelled permanently. If omitted and `min_depth` is set, the level will be cancelled immediately when depth is insufficient (no waiting).
    - "max_time_seconds": (optional) a positive integer. If the position has been open longer than this many seconds and this level has not yet triggered, the level is cancelled (marked as triggered). Use this to abandon a scalp target that hasn't been reached in time.
  Levels must be sorted by increasing take_profit_pct. The sum of all fractions must be ≤ 1.0. Each level is triggered only once. If this array is provided, the single `partial_take_profit_pct` and `partial_take_profit_fraction` are ignored. Use this to scale out of a position gradually, locking in profits at multiple small targets.
- "max_risk_per_trade_pct": a decimal between 0 and 1.0 (e.g., 0.02 for 2% of portfolio). The position size will be limited so that the potential loss (entry - stop) does not exceed this fraction of your total portfolio value. If omitted, position sizing uses only position_size_fraction.
- "max_portfolio_risk_pct": an optional decimal between 0 and 1.0 (e.g., 0.06 for 6% of portfolio). If set, the bot will calculate the total potential loss of all open positions (if their stop-loss is hit) plus the potential loss of this new trade. If this total exceeds this percentage of your total portfolio value, the trade will be skipped. Use this to prevent over-exposure and limit overall drawdown.
- "min_profit_per_trade": an optional non-negative number (in USD, e.g., 0.5 for $0.50). If set, the bot will skip the trade if the expected gross profit (position size × take_profit_pct) is below this value. Use this to avoid trades that would yield only a negligible gain.
- "min_risk_reward_ratio": an optional positive number (e.g., 1.5). If set, the validator will reject the trade unless take_profit_pct / stop_loss_pct >= this value. Use this to enforce a minimum reward for the risk you are taking.
- "max_spread_pct": an optional positive number (e.g., 0.5 for 0.5%). If set, the bot will skip the trade if the current bid‑ask spread (as a percentage of mid price) exceeds this value. Use this to avoid illiquid stocks.
- "min_depth_at_take_profit": an optional positive number (in USD, e.g., 0.5 for $0.50). If set, the bot will check the cumulative ask volume from the current mid price up to the take‑profit price. If that volume is less than this value, the trade will be skipped because the take‑profit may not fill without moving the price. Use this to ensure your scalp targets are reachable.
- "max_slippage_pct": an optional positive number (e.g., 0.1 for 0.1%). If set, the bot will compute the expected average fill price for a market buy order of the intended size by walking the order book. If the average fill price exceeds the best ask by more than this percentage, the trade will be skipped. Use this to avoid excessive slippage on illiquid stocks, which is essential for scalping very small percentages.
- "max_unrealized_loss_pct": an optional decimal between 0 and 1.0 (e.g., 0.002 for 0.2%). If set, the bot will monitor the unrealized loss of the position. If the current price falls below `entry_price * (1 - max_unrealized_loss_pct)`, the position will be closed immediately, regardless of the stop‑loss. Use this as a soft stop to cut losses quickly when scalping tiny percentages. Must be less than `stop_loss_pct`.
- "position_size_multiplier": an optional decimal between 0.0 and 1.0 (e.g., 0.5 for 50%). If set, the final position size for this trade will be further multiplied by this factor, after the global risk multiplier. Use this to reduce exposure on a specific stock without changing your global risk settings. If omitted, no additional per‑stock scaling is applied.
- "min_confidence": an optional decimal between 0.0 and 1.0 (e.g., 0.6). If set, the bot will skip the trade if your confidence is below this threshold. Use this to enforce a minimum conviction level.
- `limit_price`: (optional) a specific limit price for the order. **Required for extended‑hours trading** (pre‑market, after‑hours, weekends in paper mode). If the `session_info` shows a session other than "Regular", you MUST provide this field for BUY and SELL orders, otherwise the engine will use a default aggressive price.
- `time_in_force`: (optional) "day" or "gtc". Default "day". Required together with `limit_price` for extended‑hours orders.

You will also receive a summary of the most recent individual trades (last 20). Use this to gauge very short‑term momentum and whether the market is active enough for scalping. A high number of small trades with balanced buy/sell pressure and a tight price range suggests a liquid market suitable for capturing tiny percentages.
- "news_sentiment_exit_threshold": an optional float between -1.0 and 1.0 (e.g., -0.5). If set, the bot will monitor the aggregate news sentiment for this stock. If the compound score drops below this threshold while the position is open, the position will be closed immediately. Use this to exit on strongly negative news.
- "strategy_interval_seconds": an optional positive integer (e.g., 60, 120, 300). If set, the bot will re‑evaluate the strategy for this stock every N seconds instead of the default interval. Use shorter intervals (60‑120s) for scalping very small percentages, and longer intervals (300‑600s) for swing trades. If omitted, the global default applies.

The bot will NOT use any default values for required parameters. If you omit any required parameter, the trade will be skipped. Optional parameters are not required; if omitted, the bot will use its standard behavior.
"""

def build_stock_selection_prompt(
    available_symbols: List[str],
    current_symbols: List[Dict[str, str]],
    max_symbols: int,
    base_currency: str,
    tickers: Dict[str, Any],
    base_balance: float,
    per_symbol_budget: float,
    market_limits: Dict[str, Dict[str, Any]],
    performance: Optional[Dict[str, Any]] = None,
    ohlcv_data: Optional[Dict[str, Dict[str, List]]] = None,
    market_trend: Optional[Dict[str, Any]] = None,
    news_sentiment: Optional[Dict[str, Dict[str, Any]]] = None,
    symbol_indicators: Optional[Dict[str, Dict[str, Any]]] = None,
    daily_pnl: Optional[float] = None,
    symbol_scores: Optional[Dict[str, float]] = None,
    symbol_spreads: Optional[Dict[str, float]] = None,
    symbol_depths: Optional[Dict[str, float]] = None,
    historical_ohlcv_summary: Optional[Dict[str, Dict[str, Any]]] = None,
    correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None,
    session_info: Optional[Dict[str, Any]] = None,
    sentiment_trend: Optional[Dict[str, Optional[float]]] = None,
    top_opportunities: Optional[List[Dict[str, Any]]] = None,
    trading_paused: Optional[bool] = None,
    open_positions: Optional[Dict[str, Dict[str, Any]]] = None,
    symbol_tenure: Optional[Dict[str, float]] = None,
    symbol_max_tenure: Optional[Dict[str, Optional[float]]] = None,
    vix: Optional[float] = None,
    data_feed: str = "sip",
    sector_etf_data: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    """Build a prompt to ask the LLM which stocks/ETFs to trade."""
    # Summarize tickers and limits for the prompt
    ticker_summary = {}
    for symbol in available_symbols:
        if symbol in tickers:
            t = tickers[symbol]
            limits = market_limits.get(symbol, {})
            ticker_summary[symbol] = {
                "last": t.get("last"),
                "change_24h": t.get("percentage"),
                "volume": t.get("quoteVolume"),
                "min_trade_cost": limits.get("min_cost"),  # now always a number
            }
            if settings.NEWS_ENABLED:
                agg = get_aggregate_sentiment_from_db(symbol, max_age_seconds=settings.NEWS_CACHE_TTL_SECONDS)
                if agg:
                    ticker_summary[symbol]["sentiment"] = agg

    # Build OHLCV summary if provided
    ohlcv_summary = {}
    if ohlcv_data:
        for symbol in available_symbols:
            if symbol in ohlcv_data:
                tf_data = ohlcv_data[symbol]
                summary = {}
                for tf, candles in tf_data.items():
                    if not candles:
                        continue
                    open_price = candles[0][1]
                    close_price = candles[-1][4]
                    high = max(c[2] for c in candles)
                    low = min(c[3] for c in candles)
                    volume = sum(c[5] for c in candles)
                    change_pct = ((close_price - open_price) / open_price) * 100 if open_price else 0
                    summary[tf] = {
                        "change_pct": round(change_pct, 2),
                        "high": high,
                        "low": low,
                        "volume": volume,
                    }
                ohlcv_summary[symbol] = summary

    # --- News section ---
    news_section = ""
    if settings.NEWS_ENABLED:
        news_lines = []
        symbols_to_check = available_symbols[:50]
        for sym in symbols_to_check:
            articles = get_news_for_symbol(sym, max_age_seconds=settings.NEWS_CACHE_TTL_SECONDS)
            if articles:
                formatted = _format_news_for_prompt(articles)
                news_lines.append(f"**{sym}**\n{formatted}")
        if news_lines:
            news_section = "Recent news for top stocks:\n\n" + "\n\n".join(news_lines)

    prompt = f"""Current base currency: {base_currency}
Your available {base_currency} balance: {base_balance:.2f}
Maximum number of stocks to trade: {max_symbols}
Budget per stock: {per_symbol_budget:.2f} {base_currency}
Available timeframes: {json.dumps(settings.OHLCV_TIMEFRAMES)}
Currently tracked stocks (with assigned timeframes): {json.dumps(current_symbols) if current_symbols else "None"}"""

    # --- Open positions summary ---
    if open_positions:
        prompt += "\n**Open positions (these will continue to be managed even if trading is paused):**\n"
        for sym, pos in open_positions.items():
            entry = pos.get("price", "?")
            amount = pos.get("amount", "?")
            sl = pos.get("stop_loss", "?")
            tp = pos.get("take_profit", "?")
            prompt += (
                f"  {sym}: entry={entry}, amount={amount}, "
                f"stop_loss={sl}, take_profit={tp}\n"
            )
        prompt += (
            "When deciding to pause or resume trading, consider these open positions. "
            "If you pause, no new positions will be opened, but existing positions will still be "
            "managed with their stop-loss/take-profit levels. "
            "If you resume, new positions can be opened alongside these.\n"
        )

    if symbol_tenure:
        prompt += "\n**Stock tenure (how long each stock has been continuously tracked, in seconds):**\n"
        for sym, sec in symbol_tenure.items():
            prompt += f"  {sym}: {sec:.0f}s\n"
        prompt += (
            "Stocks that have been tracked for longer periods allow the bot to accumulate more "
            "historical data and refine strategies. Frequent changes disrupt this learning process. "
            "Therefore, **prefer to keep stocks that are already in the list** unless there is a strong "
            "reason to drop them (e.g., delisting, severe negative sentiment, consistent losses, "
            "or budget constraints). Only replace a stock if the new candidate is clearly superior.\n"
        )
    if symbol_max_tenure:
        prompt += "\n**Current max tenure per stock (hours, if set):**\n"
        for sym, hours in symbol_max_tenure.items():
            if hours is not None:
                prompt += f"  {sym}: {hours:.1f}h\n"
        prompt += (
            "You may optionally set a `max_tenure_hours` for each stock you select. "
            "If set, the bot will force-sell the stock after it has been in the portfolio for that many hours. "
            "Use this to rotate out of stocks that may become stagnant. "
            "If you omit this field, the stock will have no tenure limit (it can stay indefinitely). "
            "If you keep a stock that already has a max tenure, you may change it or keep it as is.\n"
        )

    prompt += f"""
Available symbols with market data and minimum trade cost (in {base_currency}):
{json.dumps(ticker_summary, indent=2)}

**Your primary objective is profit across short, medium, and long timeframes. Prioritize stocks where you find the most profit potential, regardless of timeframe.** Prioritize stocks with strong momentum, high volume, and clear trends on multiple timeframes. Avoid stocks that are flat or declining on all timeframes. You may keep current stocks only if they still show potential on at least one timeframe.

Select between 0 and {max_symbols} stocks to trade. If market conditions are extremely unfavorable (e.g., high losses, poor momentum, negative sentiment), you may select 0 stocks to pause trading until the next evaluation. You decide the exact number based on how many high‑quality opportunities you see. If market conditions are poor, you may choose fewer stocks (even 0 or 1) to concentrate capital on the best setup. If many strong setups exist, you may select up to {max_symbols}. You MUST only select stocks where the per-stock budget ({per_symbol_budget:.2f} {base_currency}) is greater than or equal to the stock's min_trade_cost. Skip any stock that does not meet this requirement. Prefer stocks with high volume and positive momentum. You may keep some current stocks if they are still promising and meet the budget requirement, or replace them. **Prefer to keep stocks that have been tracked for a while** – they have more historical data and the bot has already invested in learning their behaviour. Only drop a stock if it shows clear deterioration (e.g., negative momentum on all timeframes, poor win rate, or strongly negative sentiment).

**Use the historical performance data to guide your selection.** Prefer stocks that have a positive average P&L and a win rate above 50% in recent trades. Avoid stocks that have a string of losses or a negative average P&L, unless there is a strong technical or news‑driven reason to include them.

Each symbol can only appear once in your selection. Choose the single best timeframe for each stock based on the multi-timeframe OHLCV data.

**Output ONLY the raw JSON object as specified.**

Return a JSON object with the following fields:
- "stocks": a JSON array of objects, each with "symbol" and "timeframe" (the timeframe must be one of the available timeframes, e.g., "5m", "15m", "1h", "4h"). Each object may optionally include "max_tenure_hours" (a positive float, hours) to force-sell the stock after that many hours in the portfolio. Omit or set to null for no limit.
- "max_stocks": an integer between 0 and {max_symbols} indicating how many stocks you actually want to trade. Set to 0 to pause trading. This must equal the length of the "stocks" array.
- "reasoning": a short string (max 200 characters) explaining why you selected these specific stocks and timeframes. This will be shown to the user, so make it informative.

You may optionally include "stock_revaluation_interval_seconds" (integer >= 60) to change how often the bot re-evaluates the stock list.

Example: {{"stocks": [{{"symbol": "AAPL", "timeframe": "1h", "max_tenure_hours": 48}}, {{"symbol": "MSFT", "timeframe": "15m"}}], "max_stocks": 2, "reasoning": "AAPL shows strong uptrend on 1h with high volume; MSFT has bullish MACD crossover on 15m.", "stock_revaluation_interval_seconds": 300, "pause_trading": false, "pause_reason": "Market conditions are favorable"}}"""
    # --- Enhanced pause/resume guidance ---
    if trading_paused:
        prompt += (
            "\n**Trading is currently PAUSED.**\n"
            "You may resume trading by setting `\"pause_trading\": false` if you see clear profit opportunities.\n"
            "Do NOT resume just because market conditions have improved slightly; only resume if you identify specific "
            "stocks with strong setups (high scalping scores, positive sentiment, solid technicals) that are likely to be profitable.\n"
            "If you keep trading paused, include a `\"pause_reason\"` field explaining why.\n"
            "\nAlso consider the news sentiment data below when deciding whether to resume.\n"
        )
    else:
        prompt += (
            "\n**Trading is currently ACTIVE.**\n"
            "You may pause trading by setting `\"pause_trading\": true` if conditions warrant.\n"
            "However, do NOT pause solely because of a bad market index (e.g., high fear, low breadth). "
            "First, check the **Top Profit Opportunities** section below. If there are stocks with high scalping scores (>0.7), "
            "strong positive sentiment, and clear technical signals, you may still trade them profitably even in a down market.\n"
            "Only pause if NO such opportunities exist, or if the account is in significant drawdown with no high‑confidence setups.\n"
            "\nAlso consider the news sentiment data below when deciding whether to pause.\n"
        )
    if symbol_scores:
        prompt += "\nScalping suitability scores (0-1, higher = better for quick small profits):\n"
        for sym in available_symbols:
            if sym in symbol_scores:
                prompt += f"  {sym}: {symbol_scores[sym]:.3f}\n"
        prompt += (
            "Prioritise stocks with higher scores, but use your own judgement. "
            "The score combines volume, volatility, spread, and momentum.\n"
        )
    # --- Top profit opportunities summary ---
    if top_opportunities:
        prompt += "\n**Top Profit Opportunities** (best candidates for immediate trades):\n"
        for opp in top_opportunities:
            sent_str = ""
            if opp.get("sentiment") is not None:
                sent_str = f", sentiment={opp['sentiment']:.2f}"
            prompt += (
                f"  {opp['symbol']}: score={opp['score']:.3f}, "
                f"change_24h={opp['change_24h']}%{sent_str}\n"
            )
        prompt += (
            "These are the stocks with the highest scalping suitability scores. "
            "Even if the overall market looks bad, one or more of these may still offer a profitable scalp. "
            "Use this list to decide whether to pause or to trade a reduced number of stocks.\n"
        )
    if symbol_spreads or symbol_depths:
        prompt += "\nOrder book metrics for top stocks (lower spread & higher depth = better for scalping):\n"
        for sym in available_symbols:
            parts = []
            if sym in symbol_spreads:
                parts.append(f"spread={symbol_spreads[sym]:.3f}%")
            if sym in symbol_depths:
                parts.append(f"depth={symbol_depths[sym]:.2f}")
            if parts:
                prompt += f"  {sym}: {', '.join(parts)}\n"
        prompt += "Prefer stocks with spread < 0.2% and high depth for scalping very small percentages.\n"
    if ohlcv_summary:
        prompt += f"\nMulti-timeframe OHLCV summary (price change %, high, low, volume):\n{json.dumps(ohlcv_summary, indent=2)}\n"
    if correlation_matrix:
        # Trim to only include symbols that appear in the candidate list
        trimmed = {}
        for sym_a, row in correlation_matrix.items():
            trimmed[sym_a] = {sym_b: v for sym_b, v in row.items()}
        prompt += (
            "\nPairwise correlation matrix (Pearson correlation of daily returns, range -1 to +1):\n"
            f"{json.dumps(trimmed, indent=2)}\n"
            "Use this to diversify your selection. Stocks with correlation > 0.7 move very similarly – "
            "avoid selecting too many highly correlated stocks, as they concentrate risk. "
            "Prefer stocks with low or negative correlation to your existing selections to spread risk.\n"
        )
    if historical_ohlcv_summary:
        prompt += (
            "\nHistorical OHLCV summary from database (up to 30 days, price change %, high, low, volume, candle count):\n"
            f"{json.dumps(historical_ohlcv_summary, indent=2)}\n"
            "Use this longer-term data to assess sustained trends and avoid stocks in prolonged decline. "
            "Prefer stocks with consistent upward momentum over the full period.\n"
        )
    if symbol_indicators:
        prompt += "\nTechnical indicators for candidate stocks:\n"
        for sym, tf_indicators in symbol_indicators.items():
            lines = [f"{sym}:"]
            for tf, ind in tf_indicators.items():
                lines.append(f"  [{tf}]")
                if ind.get('rsi') is not None:
                    lines.append(f"    RSI(14)={ind['rsi']:.2f}")
                if ind.get('macd') is not None:
                    lines.append(f"    MACD={ind['macd']:.4f} Signal={ind['macd_signal']:.4f} Hist={ind['macd_hist']:.4f}")
                if ind.get('bb_upper') is not None:
                    lines.append(f"    BB Upper={ind['bb_upper']:.4f} Middle={ind['bb_middle']:.4f} Lower={ind['bb_lower']:.4f}")
                if ind.get('ema_9') is not None:
                    lines.append(f"    EMA9={ind['ema_9']:.4f} EMA21={ind['ema_21']:.4f}")
                if ind.get('stochastic_k') is not None:
                    d_str = f"{ind['stochastic_d']:.2f}" if ind['stochastic_d'] is not None else "N/A"
                    lines.append(f"    Stoch %K={ind['stochastic_k']:.2f} %D={d_str}")
                if ind.get('adx') is not None:
                    lines.append(f"    ADX(14)={ind['adx']:.2f} +DI={ind['plus_di']:.2f} -DI={ind['minus_di']:.2f}")
                if ind.get('obv') is not None:
                    lines.append(f"    OBV={ind['obv']:.2f}")
                if ind.get('mfi') is not None:
                    lines.append(f"    MFI(14)={ind['mfi']:.2f}")
                if ind.get('cci') is not None:
                    lines.append(f"    CCI(20)={ind['cci']:.2f}")
                if ind.get('williams_r') is not None:
                    lines.append(f"    Williams %R(14)={ind['williams_r']:.2f}")
                if ind.get('ichimoku') is not None:
                    ich = ind['ichimoku']
                    lines.append(f"    Ichimoku: Tenkan={ich['tenkan_sen']:.4f} Kijun={ich['kijun_sen']:.4f} SpanA={ich['senkou_span_a']:.4f} SpanB={ich['senkou_span_b']:.4f} Cloud={ich['cloud_bottom']:.4f}-{ich['cloud_top']:.4f}")
                if ind.get('donchian_channels') is not None:
                    dc = ind['donchian_channels']
                    lines.append(f"    Donchian: Upper={dc['upper']:.4f} Middle={dc['middle']:.4f} Lower={dc['lower']:.4f}")
                if ind.get('atr') is not None:
                    lines.append(f"    ATR(14)={ind['atr']:.6f}")
                if ind.get('vwap') is not None:
                    lines.append(f"    VWAP={ind['vwap']:.6f}")
                if ind.get('parabolic_sar') is not None:
                    lines.append(f"    Parabolic SAR={ind['parabolic_sar']:.6f}")
                if ind.get('keltner_channels') is not None:
                    kc = ind['keltner_channels']
                    lines.append(f"    Keltner: Upper={kc['upper']:.6f} Middle={kc['middle']:.6f} Lower={kc['lower']:.6f}")
                if ind.get('pivot_points') is not None:
                    pp = ind['pivot_points']
                    lines.append(f"    Pivot Points: P={pp['pivot']:.6f} R1={pp['r1']:.6f} S1={pp['s1']:.6f} R2={pp['r2']:.6f} S2={pp['s2']:.6f}")
            prompt += "\n".join(lines) + "\n"
    if market_trend:
        prompt += f"\nOverall market trend ({market_trend['symbol']}): daily change {market_trend.get('change_24h')}%, last price {market_trend.get('last')}\n"
    if session_info:
        prompt += f"\nCurrent UTC hour: {session_info['utc_hour']} ({session_info['session']} session)\n"
        prompt += (
            "If the current session is not \"Regular\" (i.e., pre‑market, after‑hours, or closed), "
            "you MUST include `limit_price` and `time_in_force` in your parameters for any BUY or SELL action. "
            "The engine will reject market orders during extended hours.\n"
        )
    # --- Data feed note ---
    if data_feed == "iex":
        prompt += (
            "\n**Data feed: IEX (free).** The IEX feed does not provide real‑time order book data. "
            "Order book metrics (spread, depth, walls) may be empty or stale. "
            "Do NOT rely on order book data for stock selection or trade decisions. "
            "Base your analysis on OHLCV, indicators, news sentiment, and other available data.\n"
        )
    else:
        prompt += (
            "\n**Data feed: SIP (real‑time).** Order book data is live and reliable. "
            "You may use order book metrics normally.\n"
        )
    if vix is not None:
        prompt += f"\nCBOE Volatility Index (VIX): {vix:.2f}\n"
        prompt += (
            "VIX measures expected market volatility over the next 30 days. "
            "High VIX (>30) indicates fear/uncertainty and often coincides with market bottoms; "
            "low VIX (<15) indicates complacency and can precede sharp corrections. "
            "Use this to gauge overall market risk and adjust position sizing accordingly.\n"
        )
    if sector_etf_data:
        prompt += "\n## Sector ETF Performance\n"
        prompt += "Current price and daily change for key sector ETFs:\n"
        for etf, data in sector_etf_data.items():
            last = data.get("last")
            change = data.get("change_pct")
            if last is not None:
                prompt += f"  {etf}: last={last:.2f}, change={change}%\n"
        prompt += (
            "Use sector performance to identify strong/weak sectors. "
            "Prefer stocks in sectors with positive daily changes and strong momentum. "
            "Avoid stocks in sectors that are declining broadly.\n"
        )
    if news_sentiment:
        prompt += "\n## News Sentiment\n"
        prompt += "Aggregate sentiment from recent news articles (compound score -1 to +1, higher = more positive):\n"
        for sym in available_symbols:
            if sym in news_sentiment:
                ns = news_sentiment[sym]
                prompt += (
                    f"- {sym}: compound={ns['avg_compound']}, "
                    f"positive={ns['positive']}, negative={ns['negative']}, "
                    f"neutral={ns['neutral']}, total_articles={ns['total_articles']}\n"
                )
        prompt += "\n"
    if sentiment_trend:
        prompt += "\nSentiment trend (change in compound score since last cycle):\n"
        for base, delta in sentiment_trend.items():
            if delta is not None:
                prompt += f"  {base}: {delta:+.4f}\n"
        prompt += (
            "A positive delta means sentiment is improving; a negative delta means it is deteriorating. "
            "Use this to gauge whether the narrative is strengthening or weakening. "
            "Improving sentiment may justify higher confidence; deteriorating sentiment may warrant caution.\n"
        )
    if news_section:
        prompt += f"\n{news_section}\n"
    prompt += (
        "\n**Sentiment is a critical factor in stock selection.** "
        "Prefer stocks with a positive aggregate sentiment (compound > 0.1). "
        "Avoid stocks with strongly negative sentiment (compound < -0.2) unless there is overwhelming technical evidence. "
        "Use sentiment to gauge market hype and potential short‑term momentum.\n"
    )
    if performance:
        perf_text = f"""
Historical Performance Data:
Overall equity curve: {json.dumps(performance.get('equity_curve', {}))}
Per-stock performance (win rate, avg P&L, total trades): {json.dumps(performance.get('stock_performance', {}), indent=2)}
Per-strategy performance: {json.dumps(performance.get('strategy_performance', {}), indent=2)}

Use this historical data to select stocks that have been profitable in the past, and to avoid stocks with poor performance. Prefer strategies that have shown higher win rates and average P&L.
"""
        prompt += perf_text
        if daily_pnl is not None:
            prompt += f"Today's realized P&L: {daily_pnl:.4f} {base_currency}\n"
        consecutive_losses = performance.get("equity_curve", {}).get("consecutive_losses", 0)
        if consecutive_losses > 0:
            prompt += f"⚠️ You have {consecutive_losses} consecutive losing trades. Consider pausing or reducing risk.\n"
    # --- Account P&L context ---
    if performance:
        daily_pnl = performance.get("equity_curve", {}).get("daily_pnl", 0.0)
        total_pnl = performance.get("equity_curve", {}).get("total_pnl", 0.0)
        prompt += (
            f"\n**Account P&L**: Today's realized P&L = {daily_pnl:.4f} {base_currency}, "
            f"Total realized P&L = {total_pnl:.4f} {base_currency}.\n"
        )
        if total_pnl < 0:
            prompt += (
                "Your account is currently in a loss. Be more conservative: prefer to pause unless you find "
                "exceptional opportunities. If you do trade, reduce position sizes and tighten stops.\n"
            )
        else:
            prompt += (
                "Your account is in profit. You may take calculated risks, but do not be reckless. "
                "Only trade if you see clear setups.\n"
            )
    return prompt

def build_strategy_prompt(
    symbol: str,
    ticker: Dict[str, Any],
    order_book: Dict[str, Any],
    balance: Dict[str, float],
    open_positions: List[Dict[str, Any]],
    per_symbol_budget: float,
    max_symbols: int,
    base_currency: str,
    performance: Optional[Dict[str, Any]] = None,
    ohlcv_data: Optional[Dict[str, List]] = None,
    assigned_timeframe: Optional[str] = None,
    atr: Optional[float] = None,
    atr_multi_tf: Optional[Dict[str, float]] = None,
    rsi: Optional[float] = None,
    macd: Optional[float] = None,
    macd_signal: Optional[float] = None,
    macd_hist: Optional[float] = None,
    bb_upper: Optional[float] = None,
    bb_middle: Optional[float] = None,
    bb_lower: Optional[float] = None,
    ema_9: Optional[float] = None,
    ema_21: Optional[float] = None,
    stochastic_k: Optional[float] = None,
    stochastic_d: Optional[float] = None,
    adx: Optional[float] = None,
    plus_di: Optional[float] = None,
    minus_di: Optional[float] = None,
    obv: Optional[float] = None,
    mfi: Optional[float] = None,
    cci: Optional[float] = None,
    williams_r: Optional[float] = None,
    order_book_imbalance: Optional[float] = None,
    unrealized_pnl: Optional[float] = None,
    position_info: Optional[Dict[str, Any]] = None,
    spread_pct: Optional[float] = None,
    bid_wall_volume: Optional[float] = None,
    ask_wall_volume: Optional[float] = None,
    order_book_pressure: Optional[float] = None,
    depth_imbalances: Optional[Dict[str, float]] = None,
    order_book_slope: Optional[float] = None,
    mid_price_bias: Optional[float] = None,
    depth_profile: Optional[Dict[str, Dict[str, float]]] = None,
    fee_rate: Optional[float] = None,
    drawdown_pct: Optional[float] = None,
    raw_candles: Optional[List[List]] = None,
    recent_trades: Optional[List[Dict[str, Any]]] = None,
    historical_ohlcv: Optional[List[List]] = None,
    min_order_amount: Optional[float] = None,
    min_order_cost: Optional[float] = None,
    all_symbols: Optional[List[Dict[str, str]]] = None,
    past_trades: Optional[List[Dict[str, Any]]] = None,
    aggregate_sentiment: Optional[Dict[str, Any]] = None,
    cycle_spent: Optional[float] = None,
    remaining_balance: Optional[float] = None,
    market_regime: Optional[str] = None,
    recent_trades_data: Optional[List[Dict[str, Any]]] = None,
    multi_tf_raw_candles: Optional[Dict[str, List[List]]] = None,
    multi_tf_indicators: Optional[Dict[str, Dict[str, Any]]] = None,
    scalping_feasibility_score: Optional[float] = None,
    vwap: Optional[float] = None,
    vwap_multi_tf: Optional[Dict[str, float]] = None,
    session_info: Optional[Dict[str, Any]] = None,
    sentiment_trend: Optional[float] = None,
    volume_trend: Optional[float] = None,
    ichimoku: Optional[Dict[str, Optional[float]]] = None,
    market_breadth: Optional[Dict[str, Any]] = None,
    full_market_breadth: Optional[Dict[str, Any]] = None,
    depth_trend: Optional[float] = None,
    parabolic_sar: Optional[float] = None,
    keltner_channels: Optional[Dict[str, float]] = None,
    pivot_points: Optional[Dict[str, float]] = None,
    donchian_channels: Optional[Dict[str, float]] = None,
    cvd: Optional[float] = None,
    cvd_normalized: Optional[float] = None,
    order_book_pressure_trend: Optional[float] = None,
    estimated_slippage_pct: Optional[float] = None,
    atr_percentile: Optional[float] = None,
    market_impact_score: Optional[float] = None,
    global_risk_multiplier: Optional[float] = None,
    trading_paused: bool = False,
    max_hold_expired: bool = False,
    max_hold_expired_count: int = 0,
    stop_loss_triggered: bool = False,
    stop_loss_review_count: int = 0,
    take_profit_triggered: bool = False,
    take_profit_review_count: int = 0,
    partial_tp_triggered: bool = False,
    partial_tp_review_count: int = 0,
    partial_tp_triggered_levels: Optional[List[int]] = None,
    dust_sweep_triggered: bool = False,
    dust_sweep_review_count: int = 0,
    max_partial_tp_reviews: int = 10,
    max_dust_sweep_reviews: int = 10,
    data_feed: str = "sip",
    portfolio_exposure_pct: Optional[float] = None,
    portfolio_stop_risk_pct: Optional[float] = None,
    portfolio_total_value: Optional[float] = None,
    portfolio_open_count: int = 0,
    portfolio_available_capital: Optional[float] = None,
    last_decision: Optional[Dict[str, Any]] = None,
    minutes_to_market_close: Optional[int] = None,
    current_strategy_interval_seconds: Optional[int] = None,
) -> str:
    """Build a prompt to generate a trading strategy for a specific stock/ETF."""
    current_price = ticker.get("last") if ticker else None
    tf_seconds = _timeframe_to_seconds(assigned_timeframe) if assigned_timeframe else 3600
    min_hold = 2 * tf_seconds
    prompt = f"""Symbol: {symbol}
Current ticker: {json.dumps(ticker)}
Order book (top 5 levels): {json.dumps(order_book)}
Current balances: {json.dumps(balance)}
"""
    # --- Portfolio context: total base balance and all tracked symbols ---
    base_balance = balance.get(base_currency, 0.0)
    prompt += f"\nTotal {base_currency} balance available: {base_balance:.2f}\n"
    if all_symbols:
        other_symbols = [s for s in all_symbols if s["symbol"] != symbol]
        if other_symbols:
            symbol_list_str = ", ".join(f"{s['symbol']}({s['timeframe']})" for s in other_symbols)
            prompt += f"Other symbols being traded (you must leave budget for them): {symbol_list_str}\n"
        else:
            prompt += "This is the only symbol being traded; you may use the full budget.\n"
    prompt += f"""Open positions: {json.dumps(open_positions)}
Your total available {base_currency} balance: {base_balance:.2f}
Suggested equal share per symbol (balance / max_symbols): {per_symbol_budget:.2f} {base_currency}
Maximum symbols to trade: {max_symbols}
"""
    # --- Portfolio exposure summary ---
    if portfolio_total_value is not None:
        prompt += f"\n**Portfolio Exposure Summary:**\n"
        prompt += f"  Total portfolio value: {portfolio_total_value:.2f} {base_currency}\n"
        prompt += f"  Open positions: {portfolio_open_count}\n"
        if portfolio_exposure_pct is not None:
            prompt += f"  Capital deployed: {portfolio_exposure_pct:.1f}% of portfolio\n"
        if portfolio_stop_risk_pct is not None:
            prompt += f"  Total stop-loss risk: {portfolio_stop_risk_pct:.2f}% of portfolio (loss if ALL stops hit)\n"
        if portfolio_available_capital is not None:
            prompt += f"  Available capital for new positions: {portfolio_available_capital:.2f} {base_currency}\n"
        prompt += (
            "Use this summary to decide position_size_fraction. If capital deployment is already high "
            "(>70%) or total stop-loss risk is elevated (>5%), reduce your position_size_fraction or output HOLD. "
            "If you have low exposure and low risk, you may allocate more capital to high-conviction trades.\n"
        )
    if cycle_spent is not None and remaining_balance is not None:
        prompt += (
            f"Amount already allocated to other symbols in this cycle: {cycle_spent:.2f} {base_currency}\n"
            f"Remaining available for this symbol: {remaining_balance:.2f} {base_currency}\n"
            "Your position_size_fraction must not require more than the remaining balance. "
            "If the remaining balance is low, reduce your fraction accordingly or output HOLD.\n"
        )
        # Help the LLM set min_profit_per_trade realistically
        max_possible_amount = min(per_symbol_budget, remaining_balance)
        prompt += (
            f"The maximum amount that can actually be allocated to this trade is "
            f"{max_possible_amount:.2f} {base_currency} (the smaller of the per‑symbol budget and the remaining balance). "
            "If you set `min_profit_per_trade`, ensure it is not larger than "
            "`max_possible_amount * take_profit_pct`. Otherwise the trade will be skipped.\n"
        )
    if global_risk_multiplier is not None and global_risk_multiplier < 1.0:
        prompt += (
            f"\n**Global risk multiplier is currently {global_risk_multiplier}.** "
            "All position sizes will be multiplied by this factor. "
            "The actual amount used will be: position_size_fraction × total_balance × global_risk_multiplier. "
            "Adjust your position_size_fraction accordingly – if you want a certain exposure, "
            "you may need to set a higher fraction to compensate, or accept the reduced size.\n"
        )
    prompt += (
        f"**position_size_fraction** now represents a fraction of your **total {base_currency} balance** (0.1 to 1.0). "
        f"You may allocate more than the equal share for high‑confidence/high‑profit opportunities, and less for riskier ones. "
        f"**Important:** The sum of position_size_fraction across all stocks you intend to trade must not exceed 1.0, "
        f"so that you leave enough capital for other stocks. Plan your allocations accordingly.\n"
    )
    base_symbol = symbol
    quote_currency = base_currency
    if min_order_amount is not None or min_order_cost is not None:
        prompt += f"\nMinimum order size for {symbol}:"
        if min_order_amount is not None:
            prompt += f" {min_order_amount} {base_symbol}"
        if min_order_cost is not None:
            prompt += f" (or {min_order_cost} {quote_currency} cost)"
        prompt += (
            ". Your position_size_fraction must result in an order that meets both the minimum amount "
            "and the minimum cost. Use the current price to convert between amount and cost.\n"
        )
    if assigned_timeframe:
        prompt += f"\nAssigned trading timeframe for this stock: {assigned_timeframe}. Base your decision primarily on the OHLCV data for this timeframe.\n"
    if market_regime:
        prompt += f"\nMarket regime: {market_regime}\n"
        prompt += (
            "Use this regime to adjust your strategy:\n"
            "- 'strong uptrend/downtrend': trend is clear and powerful. Use wider stops to avoid shakeouts, "
            "larger position sizes, and trail stops generously.\n"
            "- 'moderate uptrend/downtrend': trend exists but may weaken. Use normal stops and position sizes.\n"
            "- 'ranging': no clear direction. Prefer mean‑reversion strategies, tighter stops, smaller positions.\n"
            "- 'high volatility': expect large swings. Reduce position size, widen stops, and consider ATR‑based stops.\n"
            "- 'low volatility': quiet market. Tight stops are acceptable but beware of sudden breakout (squeeze).\n"
            "- 'squeeze': Bollinger Bands are very narrow – a large move is likely imminent. Wait for breakout confirmation.\n"
            "- 'expansion': bands are wide – trend may be strong but also prone to reversals.\n"
            "- MA bias (bullish/bearish) indicates short‑term momentum when ADX is weak.\n"
        )

    if vwap is not None:
        prompt += f"\nVWAP ({assigned_timeframe or 'current'}): {vwap:.6f}\n"
    if vwap_multi_tf:
        prompt += f"VWAP across timeframes: {json.dumps(vwap_multi_tf)}\n"
    if session_info:
        prompt += (
            f"\nCurrent UTC hour: {session_info['utc_hour']} ({session_info['session']} session)\n"
            "Use this to gauge market activity: pre‑market and after‑hours sessions have lower liquidity and wider spreads; "
            "the regular session (9:30 AM – 4:00 PM ET) has the highest volume and tightest spreads. "
            "Adjust your stock selection and risk parameters accordingly.\n"
        )
    if minutes_to_market_close is not None:
        if minutes_to_market_close > 0:
            prompt += f"  Minutes until market close (4:00 PM ET): {minutes_to_market_close}\n"
        else:
            prompt += (
                "  Market is currently closed. If you are in an extended-hours session (pre-market or after-hours), "
                "keep hold times short and use limit orders.\n"
            )
    if current_strategy_interval_seconds is not None:
        prompt += f"  Current strategy evaluation interval for this symbol: {current_strategy_interval_seconds}s\n"

    # --- Volatility, order book imbalance, and position P&L context ---
    if atr is not None:
        prompt += f"ATR (14-period, {assigned_timeframe or 'default'}): {atr:.6f}\n"
    if atr is not None and current_price is not None and current_price > 0:
        atr_pct = atr / current_price
        min_sl = 1.5 * atr_pct
        prompt += f"\n**Current ATR%: {atr_pct:.4%}**. Minimum fixed stop: {min_sl:.4%}.\n"
    if atr_percentile is not None:
        prompt += f"ATR percentile (relative to last 100 observations): {atr_percentile:.1f}%\n"
    if atr_multi_tf:
        prompt += f"ATR across timeframes: {json.dumps(atr_multi_tf)}\n"
    if order_book_imbalance is not None:
        prompt += f"Order book imbalance (bid_vol / ask_vol): {order_book_imbalance:.2f} ( >1 = buying pressure)\n"
    if spread_pct is not None:
        prompt += f"Spread: {spread_pct:.4f}%\n"
    if bid_wall_volume is not None:
        prompt += f"Bid wall volume (within 1% of best bid): {bid_wall_volume:.4f}\n"
    if ask_wall_volume is not None:
        prompt += f"Ask wall volume (within 1% of best ask): {ask_wall_volume:.4f}\n"
    if order_book_pressure is not None:
        prompt += f"Order book pressure (0 = strong sell, 1 = strong buy): {order_book_pressure:.2f}\n"
    if order_book_pressure_trend is not None:
        direction = "increasing" if order_book_pressure_trend > 0 else "decreasing" if order_book_pressure_trend < 0 else "unchanged"
        prompt += f"Order book pressure trend: {order_book_pressure_trend:+.4f} ({direction} since last cycle)\n"
        prompt += "Rising trend = building buy-side conviction; falling = growing sell-side pressure or potential spoofing. Use to distinguish genuine order flow from transient walls.\n"
    if depth_imbalances:
        prompt += f"Order book depth imbalances (bid_vol/total_vol at distance from mid): {json.dumps(depth_imbalances)}\n"
    if order_book_slope is not None:
        prompt += f"Order book slope (volume change per 0.5% price move): {order_book_slope:.2f}\n"
    if mid_price_bias is not None:
        prompt += f"Mid-price bias (-1 = near bid, +1 = near ask): {mid_price_bias:.2f}\n"
    if depth_profile:
        prompt += "\nOrder book depth profile (cumulative volume at distance from mid):\n"
        for dist, vols in depth_profile.items():
            prompt += f"  {dist}: bid={vols['bid_volume']:.4f}, ask={vols['ask_volume']:.4f}\n"
        prompt += (
            "Use this depth profile to set take‑profit levels that are likely to be filled. "
            "If the ask volume at a certain distance is thin, a small take‑profit may be filled quickly. "
            "If it's thick, you may need a larger move or a smaller position.\n"
        )
    # --- Warn if order book is empty (common with IEX feed) ---
    if not order_book.get('bids') and not order_book.get('asks'):
        if data_feed == "iex":
            bid = ticker.get('bid') if ticker else None
            ask = ticker.get('ask') if ticker else None
            last = ticker.get('last') if ticker else None
            prompt += (
                "\n**Data feed: IEX (free).** The IEX feed does not provide a real‑time order book. "
                "The order book is empty. "
                "Instead, use the **Latest Quote** below for bid/ask information:\n"
            )
            if bid is not None and ask is not None:
                prompt += f"  Latest Quote: bid={bid}, ask={ask}, last={last}\n"
                if bid > 0 and ask > 0:
                    mid = (bid + ask) / 2
                    spread_pct = ((ask - bid) / mid) * 100
                    prompt += f"  Spread (from quote): {spread_pct:.4f}%\n"
            prompt += (
                "Do NOT rely on order book depth, walls, or imbalance metrics – they are unavailable. "
                "Base your analysis on OHLCV, indicators, the latest quote, and other available data.\n"
            )
        else:
            prompt += (
                "\n**Note:** The order book is empty. This may indicate very low liquidity "
                "or a data issue. Proceed with caution and rely more on OHLCV and indicators.\n"
            )
    if recent_trades_data:
        # Summarise last 20 trades: count buys vs sells, average size, price range
        buys = [t for t in recent_trades_data if t.get('side') == 'buy']
        sells = [t for t in recent_trades_data if t.get('side') == 'sell']
        avg_buy_size = sum(t['amount'] for t in buys) / len(buys) if buys else 0
        avg_sell_size = sum(t['amount'] for t in sells) / len(sells) if sells else 0
        prices = [t['price'] for t in recent_trades_data]
        price_range = (min(prices), max(prices)) if prices else (0, 0)
        prompt += (
            f"\nRecent trade activity (last {len(recent_trades_data)} trades):\n"
            f"  Buys: {len(buys)}, avg size: {avg_buy_size:.4f}\n"
            f"  Sells: {len(sells)}, avg size: {avg_sell_size:.4f}\n"
            f"  Price range: {price_range[0]:.4f} - {price_range[1]:.4f}\n"
        )
        prompt += "Use this to assess micro-momentum and liquidity. High frequency of small trades with tight spreads is ideal for scalping.\n"
    if cvd is not None:
        prompt += f"\nCumulative Volume Delta (CVD) from recent trades: {cvd:.6f}"
        if cvd_normalized is not None:
            prompt += f" (normalized: {cvd_normalized:+.4f})"
        prompt += "\n"
        prompt += "CVD is net buying volume (buy - sell). Positive = buying pressure, negative = selling pressure. Use alongside order book pressure to confirm directional conviction. Divergences warn of weakening momentum.\n"
    if scalping_feasibility_score is not None:
        prompt += f"\nScalping feasibility score: {scalping_feasibility_score:.3f} (0-1, higher = better for very small take‑profits)\n"
        if market_impact_score is not None:
            prompt += f"  Market impact component: {market_impact_score:.3f} (0-1, higher = lower price impact per unit of volume)\n"
        prompt += "Score combines spread, depth, trade frequency, volatility, and market impact. Score > 0.7 = highly suitable for scalping tiny percentages. Market impact component measures price movement per unit of volume.\n"
    elif market_impact_score is not None:
        prompt += f"\nMarket impact score: {market_impact_score:.3f} (0-1, higher = lower price impact per unit of volume)\n"
        prompt += "Measures price movement per unit of volume. High score = minimal price impact, low score = small orders can move price.\n"
    if fee_rate is not None:
        prompt += f"Taker fee rate for this symbol: {fee_rate*100:.2f}%\n"
        # Calculate exact break-even take-profit percentage
        # Round trip: buy at P, sell at P*(1+TP). Fees: P*fee + P*(1+TP)*fee
        # Break even: P*(1+TP) - P*(1+TP)*fee - P - P*fee = 0
        # 1 + TP - fee - TP*fee - 1 - fee = 0
        # TP(1 - fee) = 2*fee
        # TP = 2*fee / (1 - fee)
        if fee_rate < 1.0:
            break_even_tp_pct = (2 * fee_rate) / (1 - fee_rate)
        else:
            break_even_tp_pct = 0.0
        
        spread_decimal = 0.0
        if spread_pct is not None:
            spread_decimal = spread_pct / 100.0
        
        min_profitable_tp_pct = break_even_tp_pct + spread_decimal
        
        prompt += (
            f"You must set take_profit_pct high enough to cover round‑trip fees and the spread. "
            f"The engine will not enforce any minimum – it trusts your calculation.\n"
            f"**Break-even calculation:** To cover entry and exit fees ({fee_rate*100:.2f}% each) and the current spread ({spread_pct:.4f}%), "
            f"your `take_profit_pct` MUST be strictly greater than {min_profitable_tp_pct:.4%}. "
            f"If you set it lower, the trade will lose money even if the take-profit is hit. "
            f"Please set your take_profit_pct comfortably above this break-even point.\n"
        )
    # Help the LLM set min_profit_per_trade by showing the expected profit for a 1% take-profit
    example_tp = 0.01
    example_profit = per_symbol_budget * example_tp
    prompt += (
        f"For reference, a 1% take-profit on the per-symbol budget ({per_symbol_budget:.2f} {quote_currency}) "
        f"would yield ~{example_profit:.4f} {quote_currency} gross profit. "
        "Set min_profit_per_trade accordingly, and ensure it is not larger than your expected profit.\n"
    )
    if estimated_slippage_pct is not None:
        prompt += f"\nEstimated slippage for a per-symbol budget market buy: {estimated_slippage_pct:.4f}%\n"
        prompt += (
            "This is the expected slippage (average fill price vs best ask) for a market buy order "
            "sized to the per-symbol budget. Use this to decide whether to reduce position_size_fraction "
            "or skip the trade entirely if slippage is too high. "
            "For scalping very small percentages, slippage above 0.05% may erode profitability. "
            "If slippage is high, consider a smaller position or a different stock. "
            f"Note: the engine will automatically cap your buy size to keep slippage below "
            f"{settings.MAX_SLIPPAGE_CAP_PCT}% (configurable).\n"
        )
    # --- Show the LLM its previous decision for this symbol ---
    if last_decision:
        age_seconds = time.time() - last_decision.get("timestamp", 0)
        prompt += (
            f"\n**Your previous decision for {symbol} (made {age_seconds:.0f}s ago):**\n"
            f"  Action: {last_decision.get('action')}\n"
            f"  Confidence: {last_decision.get('confidence', 0):.2f}\n"
            f"  Reasoning: {last_decision.get('reasoning', '')}\n"
        )
        sl_pct = last_decision.get("stop_loss_pct")
        tp_pct = last_decision.get("take_profit_pct")
        psf = last_decision.get("position_size_fraction")
        sl_method = last_decision.get("stop_loss_method")
        if sl_method:
            prompt += f"  Stop-loss method: {sl_method}\n"
        if sl_pct is not None:
            prompt += f"  Stop-loss pct: {sl_pct}\n"
        if tp_pct is not None:
            prompt += f"  Take-profit pct: {tp_pct}\n"
        if psf is not None:
            prompt += f"  Position size fraction: {psf}\n"
        prompt += (
            "Consider whether your previous decision is still valid. "
            "If market conditions have not changed significantly, maintaining consistency is preferred. "
            "Only change your action if there is a clear reason to do so. "
            "Avoid flip-flopping between BUY and HOLD without justification.\n"
        )
    if unrealized_pnl is not None and position_info:
        prompt += f"Current position unrealized P&L: {unrealized_pnl:.2f} {base_currency}\n"
        entry_price = position_info.get('price', 0)
        amount = position_info.get('amount', 0)
        prompt += f"Position details: entry price {entry_price}, amount {amount}\n"
        # Compute and show P&L percentage explicitly so the LLM doesn't have to calculate it
        if entry_price > 0 and amount > 0:
            cost_basis = entry_price * amount
            if cost_basis > 0:
                pnl_pct = (unrealized_pnl / cost_basis) * 100
                prompt += f"Unrealized P&L percentage: {pnl_pct:+.2f}%\n"
                # Show net P&L after exit fees so the LLM knows the real profit/loss if it sells now
                if fee_rate is not None and current_price and current_price > 0:
                    current_position_value = amount * current_price
                    exit_fee_cost = current_position_value * fee_rate
                    # Also account for the entry fee already paid (stored in cost_basis)
                    # cost_basis already includes entry fee, so net = sell_value - exit_fee - cost_basis
                    net_pnl_after_fees = unrealized_pnl - exit_fee_cost
                    net_pnl_pct_after_fees = (net_pnl_after_fees / cost_basis) * 100 if cost_basis > 0 else 0.0
                    prompt += f"Exit fee if sold now: {exit_fee_cost:.4f} {base_currency}\n"
                    prompt += f"Net P&L after exit fees: {net_pnl_after_fees:+.4f} {base_currency} ({net_pnl_pct_after_fees:+.2f}%)\n"
                    if net_pnl_after_fees > 0:
                        prompt += "✅ Selling now would be PROFITABLE after fees.\n"
                    elif net_pnl_after_fees == 0:
                        prompt += "➖ Selling now would break even after fees.\n"
                    else:
                        prompt += "❌ Selling now would LOSE money after fees — consider holding if you expect the price to rise.\n"
        # Explicitly show current risk levels (these are otherwise buried in the open_positions JSON)
        current_sl = position_info.get('stop_loss')
        current_tp = position_info.get('take_profit')
        if current_sl is not None:
            prompt += f"Current stop-loss price: {current_sl:.6f}\n"
        if current_tp is not None:
            prompt += f"Current take-profit price: {current_tp:.6f}\n"
        # Show distance from current price to stop/TP as percentages
        if current_price and current_price > 0:
            if current_sl is not None:
                sl_distance_pct = ((current_price - current_sl) / current_price) * 100
                prompt += f"Distance to stop-loss: {sl_distance_pct:.2f}% below current price\n"
            if current_tp is not None:
                tp_distance_pct = ((current_tp - current_price) / current_price) * 100
                prompt += f"Distance to take-profit: {tp_distance_pct:.2f}% above current price\n"
        # Show trailing stop status
        trailing_active = position_info.get('trailing_stop', False)
        if trailing_active:
            trailing_dist = position_info.get('trailing_stop_distance_pct')
            trailing_act = position_info.get('trailing_stop_activation_pct')
            prompt += f"Trailing stop: enabled (distance={trailing_dist}, activation={trailing_act})\n"
        # Show max hold time remaining
        max_hold = position_info.get('max_hold_time_seconds')
        if max_hold is not None and max_hold > 0:
            entry_ts = position_info.get('timestamp', 0) / 1000.0
            elapsed = time.time() - entry_ts if entry_ts > 0 else 0
            remaining = max(0, max_hold - elapsed)
            prompt += f"Max hold time: {max_hold:.0f}s total, {remaining:.0f}s remaining\n"

    # --- Multi-timeframe OHLCV summary and indicators ---
    if multi_tf_raw_candles:
        prompt += "\nMulti-timeframe OHLCV summary (price change %, high, low, volume, candle count):\n"
        for tf in settings.OHLCV_TIMEFRAMES:
            if tf in multi_tf_raw_candles:
                summary = _summarize_ohlcv(multi_tf_raw_candles[tf])
                if summary:
                    prompt += (
                        f"  [{tf}] change={summary['change_pct']}%, "
                        f"high={summary['high']}, low={summary['low']}, "
                        f"volume={summary['volume']}, candles={summary['candle_count']}\n"
                    )
        prompt += (
            "Use these summaries to assess short‑term momentum and trend across timeframes. "
            "The lower timeframes (5m, 15m) are ideal for timing scalping entries and exits; "
            "the higher timeframes (1h, 4h) show the larger trend.\n"
        )
    if multi_tf_indicators:
        prompt += "\nComputed technical indicators per timeframe:\n"
        for tf in settings.OHLCV_TIMEFRAMES:
            if tf in multi_tf_indicators:
                ind = multi_tf_indicators[tf]
                lines = [f"[{tf}]"]
                if ind.get('rsi') is not None:
                    lines.append(f"  RSI={ind['rsi']:.2f}")
                if ind.get('macd') is not None:
                    lines.append(f"  MACD={ind['macd']:.4f} Signal={ind['macd_signal']:.4f} Hist={ind['macd_hist']:.4f}")
                if ind.get('bb_upper') is not None:
                    lines.append(f"  BB Upper={ind['bb_upper']:.4f} Middle={ind['bb_middle']:.4f} Lower={ind['bb_lower']:.4f}")
                if ind.get('ema_9') is not None:
                    lines.append(f"  EMA9={ind['ema_9']:.4f} EMA21={ind['ema_21']:.4f}")
                if ind.get('stochastic_k') is not None:
                    lines.append(f"  Stoch %K={ind['stochastic_k']:.2f} %D={ind['stochastic_d']:.2f}")
                if ind.get('adx') is not None:
                    lines.append(f"  ADX={ind['adx']:.2f} +DI={ind['plus_di']:.2f} -DI={ind['minus_di']:.2f}")
                if ind.get('obv') is not None:
                    lines.append(f"  OBV={ind['obv']:.2f}")
                if ind.get('mfi') is not None:
                    lines.append(f"  MFI={ind['mfi']:.2f}")
                if ind.get('cci') is not None:
                    lines.append(f"  CCI={ind['cci']:.2f}")
                if ind.get('williams_r') is not None:
                    lines.append(f"  Williams %R={ind['williams_r']:.2f}")
                if ind.get('ichimoku') is not None:
                    ich = ind['ichimoku']
                    lines.append(f"  Ichimoku: Tenkan={ich['tenkan_sen']:.4f} Kijun={ich['kijun_sen']:.4f} SpanA={ich['senkou_span_a']:.4f} SpanB={ich['senkou_span_b']:.4f} Cloud={ich['cloud_bottom']:.4f}-{ich['cloud_top']:.4f}")
                if ind.get('donchian_channels') is not None:
                    dc = ind['donchian_channels']
                    lines.append(f"  Donchian: Upper={dc['upper']:.4f} Middle={dc['middle']:.4f} Lower={dc['lower']:.4f}")
                if ind.get('atr') is not None:
                    lines.append(f"  ATR(14)={ind['atr']:.6f}")
                if ind.get('vwap') is not None:
                    lines.append(f"  VWAP={ind['vwap']:.6f}")
                if ind.get('parabolic_sar') is not None:
                    lines.append(f"  Parabolic SAR={ind['parabolic_sar']:.6f}")
                if ind.get('keltner_channels') is not None:
                    kc = ind['keltner_channels']
                    lines.append(f"  Keltner: Upper={kc['upper']:.6f} Middle={kc['middle']:.6f} Lower={kc['lower']:.6f}")
                if ind.get('pivot_points') is not None:
                    pp = ind['pivot_points']
                    lines.append(f"  Pivot Points: P={pp['pivot']:.6f} R1={pp['r1']:.6f} S1={pp['s1']:.6f} R2={pp['r2']:.6f} S2={pp['s2']:.6f}")
                prompt += "\n".join(lines) + "\n"
        prompt += (
            "Use these indicators across timeframes to confirm signals. "
            "For scalping, focus on 5m/15m RSI, MACD, and Bollinger Bands for entry timing, "
            "while ensuring the 1h/4h trend supports the direction.\n"
        )
    elif raw_candles:
        summary = _summarize_ohlcv(raw_candles)
        if summary:
            prompt += (
                f"\nOHLCV summary for {assigned_timeframe} timeframe: "
                f"change={summary['change_pct']}%, high={summary['high']}, low={summary['low']}, "
                f"volume={summary['volume']}, candles={summary['candle_count']}\n"
            )
            # Only claim indicators are available if at least one key indicator is present
            has_indicators = any(v is not None for v in [rsi, macd, bb_upper, ema_9])
            if has_indicators:
                prompt += (
                    "The technical indicators (RSI, MACD, Bollinger Bands, EMA) have already been computed for you from this data. "
                    "Use them together with the summary to time entries and exits. "
                    "Explain in your reasoning how the indicators support your decision.\n"
                )
    if historical_ohlcv:
        summary = _summarize_ohlcv(historical_ohlcv)
        if summary:
            prompt += (
                f"\nHistorical OHLCV summary (up to 30 days, {assigned_timeframe} timeframe): "
                f"change={summary['change_pct']}%, high={summary['high']}, low={summary['low']}, "
                f"volume={summary['volume']}, candles={summary['candle_count']}\n"
            )
            prompt += (
                "Use this longer‑term summary to assess the overall trend and avoid stocks in prolonged decline. "
                "Prefer stocks with consistent upward momentum over the full period.\n"
            )
        # Provide raw candles for backtesting
        raw_candles_str = _format_raw_candles_compact(historical_ohlcv, max_candles=200)
        prompt += (
            f"\nRaw historical OHLCV candles for backtesting "
            f"(last {min(len(historical_ohlcv), 200)} candles, "
            f"format: [timestamp_ms, open, high, low, close, volume]):\n"
            f"{raw_candles_str}\n"
        )
        prompt += (
            "You MUST perform a backtest using this historical OHLCV data. "
            "Simulate your proposed strategy (entry at current price or a specified entry condition, "
            "stop-loss, take-profit, trailing stop, etc.) on these historical candles and compute "
            "the number of winning/losing trades, net profit/loss, and win rate. "
            "Include a `backtest_summary` field in your JSON output with a concise summary of the "
            "backtest results (e.g., \"Simulated 5 trades over 30 days: 3 wins, 2 losses, net +2.3%\"). "
            "If you cannot perform a meaningful backtest (e.g., insufficient data), explain why in the summary.\n"
        )
    if drawdown_pct is not None:
        prompt += f"Current account drawdown: {drawdown_pct}%\n"
    if recent_trades:
        prompt += f"\nRecent closed trades (last {len(recent_trades)}):\n{json.dumps(recent_trades)}\n"
        prompt += "Use these outcomes to adapt your strategy. If recent trades are losing, become more conservative.\n"

    # --- Past trades for this symbol ---
    if past_trades:
        prompt += f"\nPast closed trades for {symbol} (last {len(past_trades)}):\n"
        for t in past_trades:
            entry_price = t.get("price", 0.0)
            exit_price = t.get("exit_price", 0.0)
            amount = t.get("amount", 0.0)
            pnl = t.get("realized_pnl", 0.0)
            exit_reason = t.get("exit_reason", "unknown")
            hold_time = t.get("hold_time_seconds", None)
            strategy = t.get("strategy_type", "unknown")
            cost_basis = t.get("cost_basis", amount * entry_price)
            pnl_pct = (pnl / cost_basis * 100) if cost_basis > 0 else 0.0
            hold_str = f"{hold_time:.0f}s" if hold_time is not None else "N/A"
            buy_conf = t.get("buy_confidence", 0.0)
            buy_reason = t.get("buy_reasoning", "")
            conf_str = f", Buy Conf: {buy_conf:.2f}" if buy_conf else ""
            reason_str = f", Buy Reason: {buy_reason}" if buy_reason else ""
            prompt += (
                f"- Entry: {entry_price:.4f}, Exit: {exit_price:.4f}, Amount: {amount:.6f}, "
                f"P&L: {pnl:+.4f} ({pnl_pct:+.2f}%), Reason: {exit_reason}, "
                f"Hold: {hold_str}, Strategy: {strategy}{conf_str}{reason_str}\n"
            )
        prompt += (
            "Use these past outcomes to avoid repeating mistakes and to reinforce successful patterns. "
            "**Pay special attention to your Buy Confidence vs. actual P&L.** "
            "If your high-confidence trades (>0.7) are consistently losing, you are overconfident — lower your confidence for similar setups. "
            "If your low-confidence trades are winning, you may be underconfident — consider raising confidence for similar setups. "
            "Calibrate your confidence based on this historical feedback.\n"
        )

    # --- Aggregate sentiment summary ---
    if aggregate_sentiment:
        prompt += (
            f"\nAggregate news sentiment for {symbol}:\n"
            f"  Compound score: {aggregate_sentiment['avg_compound']:.2f}  (range -1 to +1)\n"
            f"  Positive articles: {aggregate_sentiment['positive']}\n"
            f"  Negative articles: {aggregate_sentiment['negative']}\n"
            f"  Neutral articles: {aggregate_sentiment['neutral']}\n"
            f"  Total articles: {aggregate_sentiment['total_articles']}\n"
        )
        prompt += "Use aggregate sentiment to adjust confidence, position size, and risk. Strong positive = higher confidence/larger positions; strong negative = more cautious or skip.\n"
    if sentiment_trend is not None:
        prompt += f"\nSentiment trend (change in compound score since last cycle): {sentiment_trend:+.4f}\n"
        prompt += "Positive delta = sentiment improving, negative = deteriorating. Adjust confidence and risk parameters accordingly.\n"
    if volume_trend is not None:
        prompt += f"\nVolume trend: {volume_trend:.2f}x (current daily volume relative to recent average)\n"
        prompt += "Ratio > 1.0 = volume above average, > 2.0 = significant spike. Elevated volume confirms price move strength. Low volume during breakout may signal fakeout.\n"
    if market_breadth:
        prompt += (
            f"\nMarket breadth: {market_breadth['positive_pct']}% of {market_breadth['total_count']} "
            f"candidate stocks have a positive daily change ({market_breadth['positive_count']} positive).\n"
            "High breadth (>70%) = broad market strength (risk-on); low breadth (<30%) = weakness (risk-off). Adjust selection and risk accordingly.\n"
        )
    if full_market_breadth:
        prompt += (
            f"\nFull market breadth (all available symbols): {full_market_breadth['positive_pct']}% of "
            f"{full_market_breadth['total_count']} symbols have a positive daily change "
            f"({full_market_breadth['positive_count']} positive).\n"
            "Broader measure of market health. If full breadth is very low (<25%) while candidate breadth is moderate, market may be more fragile than it appears.\n"
        )
    if depth_trend is not None:
        prompt += f"\nOrder book depth trend (change in total depth within 1% of mid since last cycle): {depth_trend:+.4f}\n"
        prompt += "Positive = growing liquidity/conviction, negative = thinning liquidity. Increasing depth supports larger positions; decreasing depth warrants caution.\n"
    if parabolic_sar is not None:
        prompt += f"\nParabolic SAR: {parabolic_sar:.6f}\n"
        prompt += "Parabolic SAR: trailing stop/reversal indicator. Price above SAR = uptrend, below = downtrend. Use as dynamic stop-loss.\n"
    if keltner_channels:
        prompt += (
            f"\nKeltner Channels (20 EMA, 2× ATR): "
            f"Upper={keltner_channels['upper']:.6f}, "
            f"Middle={keltner_channels['middle']:.6f}, "
            f"Lower={keltner_channels['lower']:.6f}\n"
        )
        prompt += "Keltner Channels: volatility-based envelopes. Price near upper = overbought, near lower = oversold. Squeeze precedes large moves.\n"
    if pivot_points:
        prompt += (
            f"\nPivot Points (from previous {assigned_timeframe or 'period'} candle): "
            f"Pivot={pivot_points['pivot']:.6f}, "
            f"R1={pivot_points['r1']:.6f}, R2={pivot_points['r2']:.6f}, "
            f"S1={pivot_points['s1']:.6f}, S2={pivot_points['s2']:.6f}\n"
        )
        prompt += "Pivot Points: support/resistance levels. Price above pivot = bullish, below = bearish. Use R1/R2 as take-profit targets, S1/S2 as stop-loss references.\n"
    if donchian_channels:
        prompt += (
            f"\nDonchian Channels ({assigned_timeframe or 'default'}): "
            f"Upper={donchian_channels['upper']:.6f}, "
            f"Middle={donchian_channels['middle']:.6f}, "
            f"Lower={donchian_channels['lower']:.6f}\n"
        )
        prompt += "Donchian Channels: highest high/lowest low over lookback period. Breakout above upper = new high (bullish), below lower = new low (bearish). Narrow channel = low volatility (squeeze).\n"

    # --- News section (detailed articles) ---
    news_section = ""
    if settings.NEWS_ENABLED:
        articles = get_news_for_symbol(symbol, max_age_seconds=settings.NEWS_CACHE_TTL_SECONDS)
        if articles:
            news_section = "Recent news articles for this stock:\n" + _format_news_for_prompt(articles)
    if news_section:
        prompt += f"\n{news_section}\n"
        prompt += "Consider the detailed news headlines above when setting your confidence, position size, and max hold time. "
        prompt += "If sentiment is very negative, reduce max hold time to limit exposure.\n"

    prompt += f"""
**Your primary objective is profit across short, medium, and long timeframes. Prioritize positions where you find the most profit potential, regardless of timeframe.** Use the ATR to set stop-loss and take-profit distances that respect the stock's volatility. Place the stop-loss below a recent swing low or support, and the take-profit near a resistance level or based on your own risk:reward assessment. You have full freedom to choose the stop distance and reward:risk ratio that you believe will maximise profitability while managing risk.

Interpret the order book metrics provided (spread, imbalance, pressure, depth, walls) to gauge liquidity and directional pressure.

If the position is already in profit, consider trailing the stop.

**For the {assigned_timeframe or 'default'} timeframe, a reasonable minimum max_hold_time_seconds is {min_hold} seconds. Do not set it lower unless you have a very specific, justified reason (e.g., scalping with a very tight stop and high confidence).**

You are trading spot only (no shorting). Only output SELL if you currently hold the stock.

**Execution Decision:** Use `"min_confidence"` (0.0–1.0) to filter trades. The bot will skip the trade if your confidence is below this threshold. If omitted, the trade executes regardless of confidence. If you are not confident enough to trade, output HOLD with a meaningful reason (e.g., "Insufficient conviction", "Unfavorable risk/reward").

"""
    prompt += (
        "\n**Entry Condition (REQUIRED for every BUY):**\n"
        "You MUST include an `entry_condition` object in your JSON output for every BUY action. "
        "This tells the bot the **exact moment** to enter the trade. "
        "If you omit this field, the trade will be executed immediately at the current market price. "
        "The object must have a `\"type\"` field and, except for `\"delay\"`, a `\"timeout_seconds\"` field.\n"
        "Supported types:\n"
        "- `\"limit_price\"`: wait for the price to drop to or below `\"price\"`.\n"
        "  Example: {\"type\": \"limit_price\", \"price\": 1.23, \"timeout_seconds\": 300}\n"
        "- `\"rsi_threshold\"`: wait for RSI(14) to fall below `\"rsi_below\"`.\n"
        "  Example: {\"type\": \"rsi_threshold\", \"rsi_below\": 30, \"timeout_seconds\": 600}\n"
        "- `\"order_book_depth\"`: wait until the cumulative ask volume within 1% of the mid price is at least `\"min_ask_volume\"`.\n"
        "  Example: {\"type\": \"order_book_depth\", \"min_ask_volume\": 500, \"timeout_seconds\": 300}\n"
        "- `\"delay\"`: simply wait `\"delay_seconds\"` before executing.\n"
        "  Example: {\"type\": \"delay\", \"delay_seconds\": 60}\n"
        "- `\"indicator_combo\"`: wait until ALL listed indicator conditions are met.\n"
        "  Example: {\"type\": \"indicator_combo\", \"conditions\": [ {\"indicator\": \"rsi\", \"threshold\": 30, \"direction\": \"below\"}, {\"indicator\": \"macd_hist\", \"threshold\": 0, \"direction\": \"above\"} ], \"timeout_seconds\": 600}\n"
        "If a timeout expires without the condition being met, the trade is skipped entirely.\n"
    )
    prompt += "\n**Output ONLY the raw JSON object as specified.**\n\nReturn a JSON object as specified.\n"
    if trading_paused:
        prompt += (
            "\n**Trading is currently PAUSED.** You may ONLY output SELL or HOLD actions. "
            "Do NOT output BUY under any circumstances. "
            "If you hold this stock, decide whether to continue holding (HOLD) or exit (SELL) "
            "based on current market conditions, risk parameters, and profit/loss status.\n"
        )
    # Add OHLCV summary if available
    if ohlcv_data:
        ohlcv_summary = {}
        for tf, candles in ohlcv_data.items():
            if not candles:
                continue
            open_price = candles[0][1]
            close_price = candles[-1][4]
            high = max(c[2] for c in candles)
            low = min(c[3] for c in candles)
            volume = sum(c[5] for c in candles)
            change_pct = ((close_price - open_price) / open_price) * 100 if open_price else 0
            ohlcv_summary[tf] = {
                "change_pct": round(change_pct, 2),
                "high": high,
                "low": low,
                "volume": volume,
            }
        prompt += f"\nMulti-timeframe OHLCV data:\n{json.dumps(ohlcv_summary, indent=2)}\n"
    if performance:
        stock_perf = performance.get("stock_performance", {}).get(symbol, {})
        strategy_perf = performance.get("strategy_performance", {})
        equity = performance.get("equity_curve", {})
        perf_text = f"""
Historical Performance:
- This stock's past performance: {json.dumps(stock_perf)} (stop_loss_hits = number of times stop-loss was triggered; avg_hold_time_seconds = average trade duration)
- Overall equity curve: {json.dumps(equity)}
- Strategy performance summary: {json.dumps(strategy_perf)}

Use this data to decide whether to BUY, SELL, or HOLD. If the stock has a poor win rate or the overall equity curve is declining, be more conservative. Prefer strategies that have worked well historically.
"""
        perf_text += (
            "Use this performance data to calibrate your parameters:\n"
            "- If the stock has a low win rate or negative average P&L, reduce position_size_fraction, "
            "widen the stop (to avoid being stopped out prematurely), and shorten max_hold_time_seconds.\n"
            "- If the stock has a high win rate and positive average P&L, you may increase position size "
            "and use tighter stops to lock in profits.\n"
            "- If stop_loss_hits is high, consider using a wider stop (larger stop_loss_pct or higher ATR multiplier) "
            "or switching to a longer timeframe.\n"
            "- Use avg_hold_time_seconds to set a realistic max_hold_time_seconds – do not set it far below "
            "the average unless you have a specific reason.\n"
        )
        prompt += perf_text
        daily_pnl = equity.get("daily_pnl", 0.0)
        total_pnl = equity.get("total_pnl", 0.0)
        consecutive_losses = equity.get("consecutive_losses", 0)
        if daily_pnl is not None:
            prompt += f"Today's realized P&L: {daily_pnl:.4f} {base_currency}\n"
        if consecutive_losses > 0:
            prompt += f"⚠️ You have {consecutive_losses} consecutive losing trades. Consider reducing risk or skipping this trade.\n"
        prompt += (
            f"\n**Account P&L**: Total realized P&L = {total_pnl:.4f} {base_currency}.\n"
        )
        if total_pnl < 0:
            prompt += (
                "Your account is currently in a loss. Be more conservative: prefer to HOLD unless you find "
                "exceptional opportunities. If you do trade, reduce position sizes and tighten stops.\n"
            )
        else:
            prompt += (
                "Your account is in profit. You may take calculated risks, but do not be reckless. "
                "Only trade if you see clear setups.\n"
            )
    if max_hold_expired:
        prompt += (
            f"\n**IMPORTANT: The max hold time for your current position in {symbol} has expired "
            f"(this is occurrence #{max_hold_expired_count}).**\n"
            "You must decide immediately whether to SELL now or to extend the hold time.\n"
            "- If you believe the position still has profit potential, output a **HOLD** action "
            "and provide a new `max_hold_time_seconds` in the `parameters` object (you may also "
            "update stop‑loss, take‑profit, or any other parameters).\n"
            "- If you decide to exit, output a **SELL** action.\n"
            "**Do NOT output HOLD without a new `max_hold_time_seconds`** – that will be treated "
            "as a decision to sell immediately.\n"
        )
    if stop_loss_triggered:
        prompt += (
            f"\n**⚠️ STOP-LOSS TRIGGERED (review {stop_loss_review_count}/3):** "
            f"Your stop-loss level was triggered for {symbol}.\n"
            "You must decide immediately:\n"
            "- **SELL**: output a SELL action to close the position.\n"
            "- **HOLD with adjusted stop**: output a HOLD action and provide a **new, lower stop-loss** "
            "(via `stop_loss_pct` or `stop_loss_atr_multiple`). You may also update other parameters "
            "(e.g., take-profit, trailing stop).\n"
            "**If you output HOLD without a new stop-loss, the engine will force-sell the position.**\n"
            "Choose the option that you believe will maximise profit or minimise loss given the current "
            "market conditions, indicators, and order book.\n"
        )
    if take_profit_triggered:
        prompt += (
            f"\n**🎯 TAKE-PROFIT TRIGGERED (review {take_profit_review_count}/3):** "
            f"Your take-profit level was reached for {symbol}.\n"
            "You must decide immediately:\n"
            "- **SELL**: output a SELL action to take the profit.\n"
            "- **HOLD with adjusted take-profit**: output a HOLD action and provide a **new, higher take-profit** "
            "(via `take_profit_pct`). You may also update other parameters (e.g., stop-loss, trailing stop).\n"
            "**If you output HOLD without a new `take_profit_pct`, the engine will force-sell the position.**\n"
            "Choose the option that you believe will maximise profit given the current "
            "market conditions, indicators, and order book.\n"
        )
    # --- Partial take-profit triggered ---
    if partial_tp_triggered:
        levels_str = ", ".join(str(i) for i in partial_tp_triggered_levels) if partial_tp_triggered_levels else "unknown"
        prompt += (
            f"\n**⚠️ PARTIAL TAKE‑PROFIT TRIGGERED (review {partial_tp_review_count}/{max_partial_tp_reviews}):** "
            f"Partial take‑profit level(s) {levels_str} for {symbol} have been reached.\n"
            "You must decide immediately:\n"
            "- **Execute**: let the partial sell(s) happen as originally planned. "
            "Output HOLD **without** changing the `partial_take_profit_levels` array.\n"
            "- **Adjust**: output HOLD and provide an **updated** `partial_take_profit_levels` array "
            "with a new `take_profit_pct` for the triggered level(s), or remove them entirely.\n"
            "- **Sell All**: output SELL to close the **entire** position.\n"
            "If you output HOLD without updating `partial_take_profit_levels`, the partial sell(s) will execute.\n"
        )
    # --- Dust sweep triggered ---
    if dust_sweep_triggered:
        prompt += (
            f"\n**🧹 DUST SWEEP TRIGGERED (review {dust_sweep_review_count}/{max_dust_sweep_reviews}):** "
            f"The remaining position size for {symbol} is below the minimum trade amount and cannot be sold normally.\n"
            "You must decide immediately:\n"
            "- **Sell Dust**: output SELL to sell the remaining dust (a market sell will be attempted).\n"
            "- **Hold**: output HOLD to keep the dust. It may become tradeable again if the price rises.\n"
            "If you output HOLD, the dust will be kept. If you output SELL, the dust will be sold.\n"
        )
    return prompt
