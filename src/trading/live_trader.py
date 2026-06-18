import time
import logging
from typing import Dict, List, Optional, Any
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce, OrderStatus
from src.config.settings import settings

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
        logger.debug("Fetched live balances: %s", balances)
        return balances

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------
    def create_market_buy_order(
        self, symbol: str, quote_amount: float, timeout: float = 60.0,
        limit_price: Optional[float] = None, time_in_force: str = "day"
    ) -> Dict[str, Any]:
        base = symbol.split("/")[0]

        # Alpaca Paper Trading limit order queuing logic
        if settings.ALPACA_PAPER and limit_price is not None:
            try:
                quote = self.trading_client.get_latest_quote(base)
                ask_price = float(quote.ask_price)
                if ask_price > limit_price:
                    logger.info(f"Paper buy limit order for {symbol} at {limit_price} queued (ask: {ask_price}).")
                    return {
                        'id': f'queued_{base}_{int(time.time()*1000)}',
                        'symbol': symbol,
                        'side': 'buy',
                        'amount': 0.0,
                        'price': 0.0,
                        'cost': 0.0,
                        'fee': {'cost': 0.0, 'currency': 'USD'},
                        'status': 'queued',
                        'limit_price': limit_price,
                        'timestamp': int(time.time() * 1000),
                    }
            except Exception as e:
                logger.warning(f"Could not fetch quote for paper buy limit queuing check: {e}")

        asset = self.trading_client.get_asset(base)
        if not asset.fractionable:
            # Non-fractionable asset: must use integer qty
            if limit_price is not None:
                if limit_price <= 0:
                    raise ValueError(f"Invalid limit_price {limit_price} for {symbol}")
                price = limit_price
            else:
                # Fetch latest quote to determine price for integer qty calculation
                quote = self.trading_client.get_latest_quote(base)
                price = float(quote.ask_price)
            
            qty = int(quote_amount / price)
            if qty < 1:
                raise ValueError(f"Insufficient funds to buy 1 whole share of {symbol} (need {price}, have {quote_amount})")
            
            if limit_price is not None:
                tif = TimeInForce.DAY if time_in_force.lower() == "day" else TimeInForce.GTC
                order_data = LimitOrderRequest(
                    symbol=base,
                    qty=qty,
                    limit_price=limit_price,
                    side=OrderSide.BUY,
                    time_in_force=tif,
                    extended_hours=True,
                )
            else:
                order_data = MarketOrderRequest(
                    symbol=base,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )
        else:
            # Fractionable asset: existing logic
            if limit_price is not None:
                if limit_price <= 0:
                    raise ValueError(f"Invalid limit_price {limit_price} for {symbol}")
                qty = quote_amount / limit_price
                tif = TimeInForce.DAY if time_in_force.lower() == "day" else TimeInForce.GTC
                order_data = LimitOrderRequest(
                    symbol=base,
                    qty=qty,
                    limit_price=limit_price,
                    side=OrderSide.BUY,
                    time_in_force=tif,
                    extended_hours=True,
                )
            else:
                order_data = MarketOrderRequest(
                    symbol=base,
                    notional=quote_amount,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )
        order = self.trading_client.submit_order(order_data)
        filled_order = self._wait_for_order_fill(order.id, base, timeout)
        return self._order_to_dict(filled_order, symbol)

    def create_market_sell_order(
        self, symbol: str, qty: float, timeout: float = 60.0,
        limit_price: Optional[float] = None, time_in_force: str = "day"
    ) -> Dict[str, Any]:
        base = symbol.split("/")[0]

        # Alpaca Paper Trading limit order queuing logic
        if settings.ALPACA_PAPER and limit_price is not None:
            try:
                quote = self.trading_client.get_latest_quote(base)
                bid_price = float(quote.bid_price)
                if bid_price < limit_price:
                    logger.info(f"Paper sell limit order for {symbol} at {limit_price} queued (bid: {bid_price}).")
                    return {
                        'id': f'queued_{base}_{int(time.time()*1000)}',
                        'symbol': symbol,
                        'side': 'sell',
                        'amount': 0.0,
                        'price': 0.0,
                        'cost': 0.0,
                        'fee': {'cost': 0.0, 'currency': 'USD'},
                        'status': 'queued',
                        'limit_price': limit_price,
                        'timestamp': int(time.time() * 1000),
                    }
            except Exception as e:
                logger.warning(f"Could not fetch quote for paper sell limit queuing check: {e}")

        if limit_price is not None:
            if limit_price <= 0:
                raise ValueError(f"Invalid limit_price {limit_price} for {symbol}")
            tif = TimeInForce.DAY if time_in_force.lower() == "day" else TimeInForce.GTC
            order_data = LimitOrderRequest(
                symbol=base,
                qty=qty,
                limit_price=limit_price,
                side=OrderSide.SELL,
                time_in_force=tif,
                extended_hours=True,
            )
        else:
            order_data = MarketOrderRequest(
                symbol=base,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
        order = self.trading_client.submit_order(order_data)
        filled_order = self._wait_for_order_fill(order.id, base, timeout)
        return self._order_to_dict(filled_order, symbol)

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch open orders, optionally filtered by symbol."""
        request = GetOrdersRequest(status="open")
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
    def _wait_for_order_fill(self, order_id: str, symbol: str, timeout: float) -> Any:
        """Poll Alpaca until the order is filled, rejected, or cancelled.
        If the timeout expires, cancel the order to avoid orphans."""
        start = time.time()
        while time.time() - start < timeout:
            order = self.trading_client.get_order_by_id(order_id)
            if order.status == OrderStatus.FILLED:
                return order
            elif order.status in (OrderStatus.REJECTED, OrderStatus.CANCELED, OrderStatus.EXPIRED):
                raise RuntimeError(f"Order {order_id} {order.status}")
            # PARTIALLY_FILLED → keep waiting
            time.sleep(0.5)

        # Timeout – fetch one last time in case it filled just after the loop
        order = self.trading_client.get_order_by_id(order_id)
        if order.status == OrderStatus.FILLED:
            return order

        # Cancel to avoid leaving an orphan order on Alpaca
        try:
            self.trading_client.cancel_order_by_id(order_id)
            logger.warning(f"Order {order_id} for {symbol} cancelled after {timeout}s timeout.")
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")

        raise RuntimeError(
            f"Order {order_id} for {symbol} did not fill within {timeout}s and was cancelled."
        )

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
