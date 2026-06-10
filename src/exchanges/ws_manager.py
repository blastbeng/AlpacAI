import asyncio
import logging
from typing import Dict, List, Optional, Any
import ccxt.pro as ccxt_pro

logger = logging.getLogger(__name__)

class WebSocketManager:
    def __init__(self, exchange: ccxt_pro.Exchange, symbols: List[str]):
        self.exchange = exchange
        self.symbols = set(symbols)
        self.tickers: Dict[str, Dict[str, Any]] = {}
        self._ticker_queue = asyncio.Queue()
        self.order_books: Dict[str, Dict[str, Any]] = {}
        self._order_book_tasks: Dict[str, asyncio.Task] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the WebSocket watch loop."""
        self._running = True
        self._task = asyncio.create_task(self._watch_loop())

    async def stop(self):
        """Stop the watch loop and close the connection."""
        self._running = False
        # Cancel ticker watch task
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        # Cancel all order book tasks
        for task in self._order_book_tasks.values():
            task.cancel()
        await asyncio.gather(*self._order_book_tasks.values(), return_exceptions=True)
        self._order_book_tasks.clear()
        await self.exchange.close()

    async def _watch_loop(self):
        """Continuously watch tickers for all subscribed symbols."""
        while self._running:
            if not self.symbols:
                await asyncio.sleep(1)
                continue
            try:
                tickers = await self.exchange.watch_tickers(list(self.symbols))
                for symbol, ticker in tickers.items():
                    self.tickers[symbol] = ticker
                    await self._ticker_queue.put((symbol, ticker))
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"WebSocket watch loop error: {e}", exc_info=True)
                await asyncio.sleep(5)

    def get_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return the latest ticker for a symbol, or None if not available."""
        return self.tickers.get(symbol)

    def get_order_book(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return the latest order book for a symbol, or None if not available."""
        return self.order_books.get(symbol)

    async def _watch_order_book(self, symbol: str):
        """Continuously watch the order book for a single symbol."""
        while self._running and symbol in self.symbols:
            try:
                ob = await self.exchange.watch_order_book(symbol)
                self.order_books[symbol] = ob
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Order book watch error for {symbol}: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def update_subscriptions(self, symbols: List[str]):
        """Update the set of symbols to watch (tickers + order books)."""
        new_symbols = set(symbols)
        if new_symbols == self.symbols:
            return

        # Stop order book watchers for removed symbols
        removed = self.symbols - new_symbols
        for sym in removed:
            task = self._order_book_tasks.pop(sym, None)
            if task:
                task.cancel()
            self.order_books.pop(sym, None)

        # Start order book watchers for new symbols
        added = new_symbols - self.symbols
        for sym in added:
            if sym not in self._order_book_tasks:
                task = asyncio.create_task(self._watch_order_book(sym))
                self._order_book_tasks[sym] = task

        self.symbols = new_symbols
        logger.info(f"WebSocket subscriptions updated: {len(self.symbols)} symbols")

    async def wait_for_update(self, timeout: float = 5.0) -> Optional[tuple]:
        """Wait for the next ticker update, or return None after timeout."""
        try:
            return await asyncio.wait_for(self._ticker_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
