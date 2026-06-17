import logging
import time
from typing import List, Dict, Any, Optional
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import (
    StockLatestQuoteRequest,
    StockLatestBarRequest,
    StockBarsRequest,
    StockLatestTradeRequest,
)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus

logger = logging.getLogger(__name__)

# Map our timeframe strings to Alpaca TimeFrame objects
TIMEFRAME_MAP = {
    "1m": TimeFrame(1, TimeFrameUnit.Minute),
    "5m": TimeFrame(5, TimeFrameUnit.Minute),
    "15m": TimeFrame(15, TimeFrameUnit.Minute),
    "1h": TimeFrame(1, TimeFrameUnit.Hour),
    "4h": TimeFrame(4, TimeFrameUnit.Hour),
    "1d": TimeFrame(1, TimeFrameUnit.Day),
}


def get_tradable_assets(trading_client: TradingClient) -> List[str]:
    """Return a list of tradable US equity symbols (stocks & ETFs)."""
    request = GetAssetsRequest(
        asset_class=AssetClass.US_EQUITY,
        status=AssetStatus.ACTIVE,
    )
    assets = trading_client.get_all_assets(request)
    symbols = [asset.symbol for asset in assets if asset.tradable]
    logger.info(f"Fetched {len(symbols)} tradable assets from Alpaca")
    return symbols


def get_quotes(
    data_client: StockHistoricalDataClient,
    symbols: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Fetch latest quotes for a list of symbols.

    Returns a dict mapping symbol -> {last, bid, ask, volume, change_24h}.
    """
    if not symbols:
        return {}
    request = StockLatestQuoteRequest(symbol_or_symbols=symbols)
    quotes = data_client.get_stock_latest_quote(request)
    result = {}
    for sym in symbols:
        q = quotes.get(sym)
        if q:
            result[sym] = {
                "last": q.ask_price,  # use ask as "last" for simplicity; we'll refine later
                "bid": q.bid_price,
                "ask": q.ask_price,
                "volume": None,  # not available in quote
                "change_24h": None,
            }
    # Enrich with daily bar for volume and change
    try:
        bars_request = StockLatestBarRequest(symbol_or_symbols=symbols)
        bars = data_client.get_stock_latest_bar(bars_request)
        for sym in symbols:
            b = bars.get(sym)
            if b:
                if sym in result:
                    result[sym]["volume"] = b.volume
                    if b.open and b.open > 0:
                        result[sym]["change_24h"] = ((b.close - b.open) / b.open) * 100
                    result[sym]["last"] = b.close  # use close as "last"
                    result[sym]["percentage"] = result[sym]["change_24h"]
                    result[sym]["quoteVolume"] = result[sym]["volume"]
    except Exception as e:
        logger.warning(f"Could not fetch daily bars for volume/change: {e}")
    return result


def get_order_book(
    data_client: StockHistoricalDataClient,
    symbol: str,
    limit: int = 20,
) -> Dict[str, Any]:
    """Return a simulated order book with best bid/ask from the latest quote."""
    request = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
    quotes = data_client.get_stock_latest_quote(request)
    q = quotes.get(symbol)
    if not q:
        return {"bids": [], "asks": []}
    return {
        "bids": [[q.bid_price, q.bid_size]],
        "asks": [[q.ask_price, q.ask_size]],
    }


def get_multi_timeframe_bars(
    data_client: StockHistoricalDataClient,
    symbol: str,
    timeframes: List[str],
    limit: int = 24,
) -> Dict[str, List[List[float]]]:
    """Fetch OHLCV bars for a symbol across multiple timeframes.

    Returns a dict mapping timeframe -> list of candles [timestamp_ms, open, high, low, close, volume].
    """
    result = {}
    for tf in timeframes:
        alpaca_tf = TIMEFRAME_MAP.get(tf)
        if not alpaca_tf:
            logger.warning(f"Unsupported timeframe: {tf}")
            result[tf] = []
            continue
        try:
            request = StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=alpaca_tf,
                limit=limit,
            )
            bars = data_client.get_stock_bars(request)
            symbol_bars = bars.get(symbol, [])
            candles = [
                [int(bar.timestamp.timestamp() * 1000), bar.open, bar.high, bar.low, bar.close, bar.volume]
                for bar in symbol_bars
            ]
            result[tf] = candles
        except Exception as e:
            logger.warning(f"Failed to fetch bars for {symbol} {tf}: {e}")
            result[tf] = []
        time.sleep(0.3)  # small delay to avoid rate limits
    return result


def get_bars_range(
    data_client: StockHistoricalDataClient,
    symbol: str,
    timeframe: str,
    start_ms: int,
    limit: int = 500,
) -> List[List[float]]:
    """Fetch OHLCV bars from a start timestamp (ms) up to the present.

    Returns a list of candles [timestamp_ms, open, high, low, close, volume].
    """
    from datetime import datetime, timezone
    alpaca_tf = TIMEFRAME_MAP.get(timeframe)
    if not alpaca_tf:
        logger.warning(f"Unsupported timeframe: {timeframe}")
        return []
    start_dt = datetime.fromtimestamp(start_ms / 1000.0, tz=timezone.utc)
    request = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=alpaca_tf,
        start=start_dt,
        limit=limit,
    )
    try:
        bars = data_client.get_stock_bars(request)
        symbol_bars = bars.get(symbol, [])
        candles = [
            [int(bar.timestamp.timestamp() * 1000), bar.open, bar.high, bar.low, bar.close, bar.volume]
            for bar in symbol_bars
        ]
        return candles
    except Exception as e:
        logger.warning(f"Failed to fetch bars range for {symbol} {timeframe}: {e}")
        return []


# Keep old names for backward compatibility during migration.
# They will be removed once the engine is fully adapted.
def get_tradable_symbols(trading_client: TradingClient) -> List[str]:
    """Return a list of trading symbols (e.g., 'AAPL/USD') for all tradable US equities."""
    symbols = get_tradable_assets(trading_client)
    return [f"{sym}/USD" for sym in symbols]


def get_tickers(exchange, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
    """Temporary wrapper: returns quotes for given symbols (pair format 'SYM/USD' → plain 'SYM')."""
    if symbols is None:
        return {}
    # Strip "/USD" suffix to get plain Alpaca symbols
    plain_symbols = [s.split("/")[0] if "/" in s else s for s in symbols]
    return get_quotes(exchange, plain_symbols)


def get_order_book(exchange, symbol: str, limit: int = 20) -> Dict[str, Any]:
    """Temporary wrapper: returns simulated order book (pair format 'SYM/USD' → plain 'SYM')."""
    plain_symbol = symbol.split("/")[0] if "/" in symbol else symbol
    return get_order_book(exchange, plain_symbol, limit)


def get_multi_timeframe_ohlcv(
    exchange, symbol: str, timeframes: List[str], limit: int = 24
) -> Dict[str, List[List[float]]]:
    """Temporary wrapper: returns multi-timeframe bars (pair format 'SYM/USD' → plain 'SYM')."""
    plain_symbol = symbol.split("/")[0] if "/" in symbol else symbol
    return get_multi_timeframe_bars(exchange, plain_symbol, timeframes, limit)
