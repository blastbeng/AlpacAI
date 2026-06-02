from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Exchange
    EXCHANGE_ID: str = "binance"
    EXCHANGE_API_KEY: Optional[str] = None
    EXCHANGE_SECRET: Optional[str] = None
    EXCHANGE_PASSWORD: Optional[str] = None

    # Trading mode
    TRADING_MODE: str = "paper"   # "paper" or "live"

    # Base currency
    BASE_CURRENCY: str = "USDT"

    # Max coins to trade
    MAX_COINS: int = 10

    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "deepseek-v4-flash"
    OLLAMA_API_KEY: Optional[str] = None

    # Redis
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0

    # Telegram
    TELEGRAM_BOT_TOKEN: Optional[str] = None
    TELEGRAM_CHAT_ID: Optional[str] = None

    # Web
    WEB_HOST: str = "0.0.0.0"
    WEB_PORT: int = 8083

    # Logging
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
