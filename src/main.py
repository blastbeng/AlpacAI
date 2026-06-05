import asyncio
import logging
import uvicorn
from src.web.app import app
from src.config.settings import settings
from src.database import init_db, get_telegram_chat_id, set_telegram_chat_id
from src.utils.redis_client import get_redis_client
from src.trading.engine import TradingEngine
from src.news.fetcher import test_rss_feeds

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

def _seed_telegram_chat_id():
    """If TELEGRAM_CHAT_ID is set in env and no chat_id is stored, store it."""
    if settings.TELEGRAM_CHAT_ID:
        existing = get_telegram_chat_id()
        if existing is None:
            try:
                chat_id = int(settings.TELEGRAM_CHAT_ID)
                set_telegram_chat_id(chat_id)
                logging.info(f"Seeded Telegram chat ID from env: {chat_id}")
            except ValueError:
                logging.warning("TELEGRAM_CHAT_ID in .env is not a valid integer")


def _cleanup_redis_state():
    """Remove old trading state keys from Redis (now stored in SQLite)."""
    redis = get_redis_client()
    keys_to_delete = [
        "trading:current_coins",
        "trading:positions",
        "trading:trade_history",
        "trading:initial_balance",
        "trading:last_coin_eval",
    ]
    for key in keys_to_delete:
        redis.delete(key)


async def main():
    init_db()
    _seed_telegram_chat_id()
    _cleanup_redis_state()
    test_rss_feeds()
    engine = TradingEngine()
    logging.info("Trading engine initialized.")
    from src.web.app import set_engine
    set_engine(engine)

    # Start the web server immediately so the dashboard can connect
    config = uvicorn.Config(
        app,
        host=settings.WEB_HOST,
        port=settings.WEB_PORT,
        log_level=settings.LOG_LEVEL.lower(),
    )
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())

    # Now set up Telegram (may take time) and start the engine loop
    if settings.TELEGRAM_BOT_TOKEN:
        from src.telegram.bot import TelegramBot
        telegram_bot = TelegramBot(engine)
        engine.set_notifier(telegram_bot)

        await telegram_bot.start()
        await telegram_bot.send_notification("🤖 Bengobot started! Use the buttons below to control me.")

    asyncio.create_task(engine.run())

    # Wait for the server to finish (it runs forever)
    await server_task

if __name__ == "__main__":
    asyncio.run(main())
