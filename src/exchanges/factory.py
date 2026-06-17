import logging
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from src.config.settings import settings

logger = logging.getLogger(__name__)

def get_trading_client() -> TradingClient:
    """Return an Alpaca TradingClient for order placement and account info."""
    return TradingClient(
        api_key=settings.ALPACA_API_KEY,
        secret_key=settings.ALPACA_SECRET_KEY,
        paper=settings.ALPACA_PAPER,
        url_override=settings.ALPACA_BASE_URL,
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
        url_override=settings.ALPACA_DATA_URL,
    )

# Keep the old names for backward compatibility during migration.
# They will be removed once the engine is fully adapted.
def get_exchange():
    """Temporary wrapper: returns the trading client (used for orders/account)."""
    return get_trading_client()

def get_pro_exchange():
    """Temporary wrapper: returns the streaming client (used for WebSocket)."""
    return get_streaming_client()
