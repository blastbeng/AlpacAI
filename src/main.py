import asyncio
import logging
import uvicorn
from src.web.app import app
from src.config.settings import settings
from src.trading.engine import TradingEngine

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

async def main():
    engine = TradingEngine()
    logging.info("Trading engine initialized.")
    from src.web.app import set_engine
    set_engine(engine)
    # Set up Telegram notifier before starting the engine
    if settings.TELEGRAM_BOT_TOKEN:
        from src.telegram.bot import TelegramBot
        telegram_bot = TelegramBot(engine)
        engine.set_notifier(telegram_bot)

        # Initialize the bot so we can send a startup message immediately
        await telegram_bot.initialize()
        await telegram_bot.send_notification("🤖 Bot started! Use the buttons below to control me.")

        # Start polling in the background
        asyncio.create_task(telegram_bot.run())

    # Now start the trading engine
    asyncio.create_task(engine.run())
    # Run the web server
    config = uvicorn.Config(
        app,
        host=settings.WEB_HOST,
        port=settings.WEB_PORT,
        log_level=settings.LOG_LEVEL.lower(),
    )
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())
