import ccxt
from typing import Dict, List, Optional, Any


class LiveTrader:
    """Wraps a real exchange account for live trading."""

    def __init__(self, exchange: ccxt.Exchange):
        self.exchange = exchange

    def get_balance(self, currency: str) -> float:
        """Get free balance for a specific currency."""
        balance = self.exchange.fetch_balance()
        return balance.get(currency, {}).get('free', 0.0)

    def fetch_balance(self) -> Dict[str, float]:
        """Return all free balances."""
        balance = self.exchange.fetch_balance()
        free_balances = {}
        for currency, data in balance.get('total', {}).items():
            free_balances[currency] = data
        return free_balances

    def create_market_buy_order(self, symbol: str, quote_amount: float) -> Dict[str, Any]:
        """
        Place a market buy order using quote currency amount.
        Note: ccxt's create_market_buy_order expects the amount in base currency.
        We'll calculate the base amount from the current price.
        """
        ticker = self.exchange.fetch_ticker(symbol)
        price = ticker['last']
        base_amount = quote_amount / price
        return self.exchange.create_market_buy_order(symbol, base_amount)

    def create_market_sell_order(self, symbol: str, base_amount: float) -> Dict[str, Any]:
        """Place a market sell order."""
        return self.exchange.create_market_sell_order(symbol, base_amount)

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch open orders, optionally filtered by symbol."""
        return self.exchange.fetch_open_orders(symbol)

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by ID. Returns True if successful."""
        try:
            self.exchange.cancel_order(order_id)
            return True
        except Exception:
            return False

    def get_trade_history(self) -> List[Dict[str, Any]]:
        """Fetch recent closed trades."""
        return self.exchange.fetch_my_trades()
