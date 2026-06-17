import asyncio
import logging
from typing import Dict, List, Optional, Any
from alpaca.data.live import StockDataStream
from alpaca.data.models import Quote, Trade

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages real‑time market data via Alpaca's StockDataStream."""

    def __init__(self, stream: StockDataStream, symbols: List[str]):
        self.stream = stream
        self.symbols = set(self._plain(s) for s in symbols)
        self.tickers: Dict[str, Dict[str, Any]] = {}
        self._ticker_queue = asyncio.Queue()
        self.order_books: Dict[str, Dict[str, Any]] = {}
        self.trades: Dict[str, List[Dict[str, Any]]] = {}
        self._running = False
        self._tasks: List[asyncio.Task] = []
        self._reconnect_lock = asyncio.Lock()

    @staticmethod
    def _plain(symbol: str) -> str:
        """Strip '/USD' suffix from a pair symbol, returning the plain symbol."""
        return symbol.split("/")[0] if "/" in symbol else symbol

    # ------------------------------------------------------------------
    # Public API (same signatures as before)
    # ------------------------------------------------------------------
    async def start(self):
        """Start the WebSocket stream and subscribe to initial symbols."""
        self._running = True
        if self.symbols:
            self.stream.subscribe_quotes(list(self.symbols))
            self.stream.subscribe_trades(list(self.symbols))
        self.stream.on_quote(self._on_quote)
        self.stream.on_trade(self._on_trade)
        self._tasks.append(asyncio.create_task(self._run_stream()))

    async def stop(self):
        """Stop the stream and cancel all tasks."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        try:
            await self.stream.close()
        except Exception:
            pass

    async def update_subscriptions(self, symbols: List[str]):
        """Change the set of watched symbols."""
        new_symbols = set(self._plain(s) for s in symbols)
        if new_symbols == self.symbols:
            return
        removed = self.symbols - new_symbols
        added = new_symbols - self.symbols
        if removed:
            self.stream.unsubscribe_quotes(list(removed))
            self.stream.unsubscribe_trades(list(removed))
        if added:
            self.stream.subscribe_quotes(list(added))
            self.stream.subscribe_trades(list(added))
        self.symbols = new_symbols
        for sym in removed:
            self.tickers.pop(sym, None)
            self.order_books.pop(sym, None)
            self.trades.pop(sym, None)

    def get_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        return self.tickers.get(self._plain(symbol))

    def get_order_book(self, symbol: str) -> Optional[Dict[str, Any]]:
        return self.order_books.get(self._plain(symbol))

    def get_trades(self, symbol: str) -> List[Dict[str, Any]]:
        return self.trades.get(self._plain(symbol), [])

    async def wait_for_update(self, timeout: float = 5.0) -> Optional[tuple]:
        try:
            return await asyncio.wait_for(self._ticker_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    @property
    def healthy(self) -> bool:
        if not self._running:
            return False
        return any(not task.done() for task in self._tasks)

    # ------------------------------------------------------------------
    # Internal stream handling
    # ------------------------------------------------------------------
    async def _run_stream(self):
        """Run the stream's event loop and reconnect on failure."""
        while self._running:
            try:
                await self.stream.run()
            except Exception as e:
                logger.error(f"Stream disconnected: {e}")
                if self._running:
                    await self._reconnect()
            await asyncio.sleep(1)

    async def _reconnect(self):
        """Close the current stream, create a new one, and re‑subscribe."""
        async with self._reconnect_lock:
            logger.warning("WebSocket reconnecting...")
            try:
                await self.stream.close()
            except Exception:
                pass
            from src.exchanges.factory import get_streaming_client
            self.stream = get_streaming_client()
            if self.symbols:
                self.stream.subscribe_quotes(list(self.symbols))
                self.stream.subscribe_trades(list(self.symbols))
            self.stream.on_quote(self._on_quote)
            self.stream.on_trade(self._on_trade)
            logger.info("Reconnection complete.")

    async def _on_quote(self, quote: Quote):
        """Handle an incoming quote update."""
        symbol = quote.symbol
        ticker = {
            'symbol': symbol,
            'bid': quote.bid_price,
            'ask': quote.ask_price,
            'last': (quote.bid_price + quote.ask_price) / 2 if quote.bid_price and quote.ask_price else quote.ask_price,
            'percentage': None,      # not available from streaming quote
            'quoteVolume': None,     # not available from streaming quote
            'timestamp': int(quote.timestamp.timestamp() * 1000) if quote.timestamp else None,
        }
        self.tickers[symbol] = ticker
        self.order_books[symbol] = {
            'bids': [[quote.bid_price, quote.bid_size]],
            'asks': [[quote.ask_price, quote.ask_size]],
        }
        await self._ticker_queue.put((symbol, ticker))

    async def _on_trade(self, trade: Trade):
        """Handle an incoming trade update."""
        symbol = trade.symbol
        trade_dict = {
            'id': trade.id,
            'symbol': symbol,
            'price': trade.price,
            'amount': trade.size,
            'side': 'buy' if trade.taker_side == 'buy' else 'sell',
            'timestamp': int(trade.timestamp.timestamp() * 1000) if trade.timestamp else None,
        }
        if symbol not in self.trades:
            self.trades[symbol] = []
        self.trades[symbol].append(trade_dict)
        if len(self.trades[symbol]) > 50:
            self.trades[symbol] = self.trades[symbol][-50:]
