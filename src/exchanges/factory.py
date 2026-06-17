import logging
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.data.enums import DataFeed
from src.config.settings import settings

logger = logging.getLogger(__name__)

def get_trading_client() -> TradingClient:
    """Return an Alpaca TradingClient for order placement and account info."""
    return TradingClient(
        api_key=settings.ALPACA_API_KEY,
        secret_key=settings.ALPACA_SECRET_KEY,
        paper=settings.ALPACA_PAPER,
        url_override=settings.ALPACA_ENDPOINT,
    )

def get_data_client() -> StockHistoricalDataClient:
    """Return an Alpaca StockHistoricalDataClient for market data (bars, quotes)."""
    return StockHistoricalDataClient(
        api_key=settings.ALPACA_API_KEY,
        secret_key=settings.ALPACA_SECRET_KEY,
        url_override=settings.ALPACA_DATA_URL,
    )

def get_streaming_client() -> StockDataStream:
    """Return an Alpaca StockDataStream for real‑time WebSocket data."""
    return StockDataStream(
        api_key=settings.ALPACA_API_KEY,
        secret_key=settings.ALPACA_SECRET_KEY,
        feed=DataFeed.IEX,
        url_override=settings.ALPACA_STREAM_URL,
        raw_data=settings.ALPACA_PAPER,
    )

