import time
import logging
from typing import Dict, List, Optional, Any
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce, OrderStatus

logger = logging.getLogger(__name__)


class LiveTrader:
    """Wraps an Alpaca TradingClient for live stock/ETF trading."""

    def __init__(self, trading_client: TradingClient):
        self.trading_client = trading_client

    # ------------------------------------------------------------------
    # Balance helpers
    # ------------------------------------------------------------------
    def get_balance(self, currency: str) -> float:
        """Get free balance for a specific currency (USD or stock symbol)."""
        if currency.upper() == "USD":
            account = self.trading_client.get_account()
            return float(account.cash)
        else:
            # currency is a stock symbol (e.g., "AAPL")
            try:
                pos = self.trading_client.get_open_position(currency)
                return float(pos.qty)
            except Exception:
                return 0.0

    def fetch_balance(self) -> Dict[str, float]:
        """Return all free balances (USD + all open positions)."""
        account = self.trading_client.get_account()
        balances = {"USD": float(account.cash)}
        try:
            positions = self.trading_client.get_all_positions()
            for pos in positions:
                balances[pos.symbol] = float(pos.qty)
        except Exception as e:
            logger.warning(f"Could not fetch positions: {e}")
        logger.info("Fetched live balances: %s", balances)
        return balances

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------
    def create_market_buy_order(self, symbol: str, quote_amount: float) -> Dict[str, Any]:
        """
        Place a market buy order using quote currency amount (USD).
        Waits for the order to fill before returning.
        """
        base = symbol.split("/")[0]   # e.g., "AAPL" from "AAPL/USD"
        order_data = MarketOrderRequest(
            symbol=base,
            notional=quote_amount,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        order = self.trading_client.submit_order(order_data)
        filled_order = self._wait_for_order_fill(order.id, base)
        return self._order_to_dict(filled_order, symbol)

    def create_market_sell_order(self, symbol: str, qty: float) -> Dict[str, Any]:
        """Place a market sell order for a given quantity of shares. Waits for fill before returning."""
        base = symbol.split("/")[0]
        order_data = MarketOrderRequest(
            symbol=base,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        order = self.trading_client.submit_order(order_data)
        filled_order = self._wait_for_order_fill(order.id, base)
        return self._order_to_dict(filled_order, symbol)

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch open orders, optionally filtered by symbol."""
        request = GetOrdersRequest(status=OrderStatus.OPEN)
        if symbol:
            base = symbol.split("/")[0]
            request.symbols = [base]
        orders = self.trading_client.get_orders(request)
        return [self._order_to_dict(o, o.symbol) for o in orders]

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by ID. Returns True if successful."""
        try:
            self.trading_client.cancel_order_by_id(order_id)
            return True
        except Exception:
            return False

    def get_trade_history(self) -> List[Dict[str, Any]]:
        """Fetch recent closed orders."""
        request = GetOrdersRequest(status=OrderStatus.CLOSED, limit=100)
        orders = self.trading_client.get_orders(request)
        return [self._order_to_dict(o, o.symbol) for o in orders]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _wait_for_order_fill(self, order_id: str, symbol: str, timeout: float = 30.0) -> Any:
        """Poll Alpaca until the order is filled, rejected, or cancelled."""
        start = time.time()
        while time.time() - start < timeout:
            order = self.trading_client.get_order_by_id(order_id)
            if order.status == OrderStatus.FILLED:
                return order
            elif order.status in (OrderStatus.REJECTED, OrderStatus.CANCELED, OrderStatus.EXPIRED):
                raise RuntimeError(f"Order {order_id} {order.status}")
            time.sleep(0.5)
        raise RuntimeError(f"Order {order_id} did not fill within {timeout}s")

    def _order_to_dict(self, order, symbol: str) -> Dict[str, Any]:
        """Convert an Alpaca order object to the dict format expected by the engine."""
        qty = float(order.filled_qty) if order.filled_qty else 0.0
        price = float(order.filled_avg_price) if order.filled_avg_price else 0.0
        cost = qty * price
        return {
            'id': str(order.id),
            'symbol': symbol,          # original pair (e.g., "AAPL/USD")
            'side': 'buy' if order.side == OrderSide.BUY else 'sell',
            'amount': qty,
            'price': price,
            'cost': cost,
            'fee': {'cost': 0.0, 'currency': 'USD'},
            'status': 'closed',
            'timestamp': int(order.created_at.timestamp() * 1000) if order.created_at else int(time.time() * 1000),
        }
