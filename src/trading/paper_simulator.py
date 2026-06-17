import uuid
import time
import logging
from typing import Dict, List, Optional, Any
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient

logger = logging.getLogger(__name__)

from src.exchanges.fees import get_fee_rate
from src.exchanges.market_data import get_quotes


class PaperSimulator:
    """Simulates a trading account with fake balances and order execution."""

    def __init__(
        self,
        data_client: StockHistoricalDataClient,
        trading_client: TradingClient,
        base_currency: str = "USD",
        initial_balance: float = 10000.0,
        fee_rate: float = 0.0,
        redis_client=None,
        ws_manager=None,
    ):
        self.data_client = data_client
        self.trading_client = trading_client
        self.base_currency = base_currency
        self.fee_rate = fee_rate
        self.redis_client = redis_client
        self.ws_manager = ws_manager
        self.balances: Dict[str, float] = {base_currency: initial_balance}
        self.orders: List[Dict[str, Any]] = []
        self.trades: List[Dict[str, Any]] = []

    def _get_price(self, symbol: str) -> float:
        """Get current mid price for a symbol, preferring live WebSocket data."""
        if self.ws_manager is not None:
            ws_ticker = self.ws_manager.get_ticker(symbol)
            if ws_ticker is not None:
                bid = ws_ticker.get('bid')
                ask = ws_ticker.get('ask')
                last = ws_ticker.get('last')
                if bid is not None and ask is not None:
                    return (bid + ask) / 2
                if last is not None:
                    return last
        # Fallback to REST via data client
        quotes = get_quotes(self.data_client, [symbol])
        q = quotes.get(symbol)
        if q:
            bid = q.get('bid')
            ask = q.get('ask')
            last = q.get('last')
            if bid is not None and ask is not None:
                return (bid + ask) / 2
            if last is not None:
                return last
        raise ValueError(f"Could not get price for {symbol}")

    def _deduct_fee(self, currency: str, amount: float) -> float:
        return amount * (1 - self.fee_rate)

    def _get_fee_rate(self, symbol: str) -> float:
        return get_fee_rate(
            self.trading_client,
            symbol,
            redis_client=self.redis_client,
            default=self.fee_rate,
        )

    def get_balance(self, currency: str) -> float:
        return self.balances.get(currency, 0.0)

    def fetch_balance(self) -> Dict[str, float]:
        return dict(self.balances)

    def create_market_buy_order(self, symbol: str, quote_amount: float) -> Dict[str, Any]:
        base, quote = symbol.split('/')
        price = self._get_price(symbol)
        base_amount = quote_amount / price
        fee_rate = self._get_fee_rate(symbol)
        fee = base_amount * fee_rate
        net_base = base_amount - fee

        if self.balances.get(quote, 0) < quote_amount:
            raise ValueError(f"Insufficient {quote} balance")

        self.balances[quote] -= quote_amount
        self.balances[base] = self.balances.get(base, 0) + net_base

        order = {
            'id': str(uuid.uuid4()),
            'symbol': symbol,
            'type': 'market',
            'side': 'buy',
            'amount': base_amount,
            'price': price,
            'cost': quote_amount,
            'fee': {'cost': fee, 'currency': base},
            'status': 'closed',
            'timestamp': int(time.time() * 1000),
        }
        self.orders.append(order)
        self.trades.append(order)
        logger.info("Paper %s %s: %s %s @ %s", order['side'].upper(), order['symbol'], order['amount'], order['cost'], order['price'])
        return order

    def create_market_sell_order(self, symbol: str, qty: float) -> Dict[str, Any]:
        """Simulate a market sell order for a given quantity of shares."""
        base, quote = symbol.split('/')
        price = self._get_price(symbol)
        quote_amount = qty * price
        fee_rate = self._get_fee_rate(symbol)
        fee = quote_amount * fee_rate
        net_quote = quote_amount - fee

        if self.balances.get(base, 0) < qty:
            raise ValueError(f"Insufficient {base} balance")

        self.balances[base] -= qty
        self.balances[quote] = self.balances.get(quote, 0) + net_quote

        order = {
            'id': str(uuid.uuid4()),
            'symbol': symbol,
            'type': 'market',
            'side': 'sell',
            'amount': qty,
            'price': price,
            'cost': quote_amount,
            'fee': {'cost': fee, 'currency': quote},
            'status': 'closed',
            'timestamp': int(time.time() * 1000),
        }
        self.orders.append(order)
        self.trades.append(order)
        logger.info("Paper %s %s: %s %s @ %s", order['side'].upper(), order['symbol'], order['amount'], order['cost'], order['price'])
        return order

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        return []

    def cancel_order(self, order_id: str) -> bool:
        return False

    def get_trade_history(self) -> List[Dict[str, Any]]:
        return self.trades
