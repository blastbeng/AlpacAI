import ccxt
import logging
import time
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class LiveTrader:
    """Wraps a real exchange account for live trading."""

    def __init__(self, exchange: ccxt.Exchange):
        self.exchange = exchange

    def _wait_for_order_fill(self, order: Dict[str, Any], symbol: str, timeout: float = 10.0) -> Dict[str, Any]:
        """
        Poll the exchange until the order is closed and has valid amount/cost.
        Raises RuntimeError if the order does not fill within the timeout.
        """
        if order.get('status') == 'closed' and order.get('amount') is not None:
            return order

        start = time.time()
        while time.time() - start < timeout:
            time.sleep(0.5)
            try:
                order = self.exchange.fetch_order(order['id'], symbol)
            except Exception as e:
                logger.warning(f"Error fetching order {order['id']}: {e}")
                continue
            if order.get('status') == 'closed' and order.get('amount') is not None:
                return order
            logger.debug(f"Waiting for order {order['id']} to fill... status={order.get('status')}")

        raise RuntimeError(f"Order {order['id']} did not fill within {timeout}s")

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
        logger.debug("Fetched live balances: %s", free_balances)
        return free_balances

    def create_market_buy_order(self, symbol: str, quote_amount: float) -> Dict[str, Any]:
        """
        Place a market buy order using quote currency amount.
        Waits for the order to fill before returning.
        """
        ticker = self.exchange.fetch_ticker(symbol)
        price = ticker['last']
        base_amount = quote_amount / price
        order = self.exchange.create_market_buy_order(symbol, base_amount)
        order = self._wait_for_order_fill(order, symbol)
        logger.info("Live BUY %s: amount=%s cost=%s @ %s", symbol, order.get('amount'), order.get('cost'), order.get('price'))
        return order

    def create_market_sell_order(self, symbol: str, base_amount: float) -> Dict[str, Any]:
        """Place a market sell order. Waits for fill before returning."""
        order = self.exchange.create_market_sell_order(symbol, base_amount)
        order = self._wait_for_order_fill(order, symbol)
        logger.info("Live SELL %s: amount=%s cost=%s @ %s", symbol, order.get('amount'), order.get('cost'), order.get('price'))
        return order

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
