from fastapi import FastAPI, HTTPException
from src.config.settings import settings
from src.utils.redis_client import get_redis_client

app = FastAPI(title="Crypto Trading Bot")

# Global engine reference
_engine = None

def set_engine(engine):
    global _engine
    _engine = engine

def get_engine():
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return _engine

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/api/status")
async def status():
    engine = get_engine()
    redis = get_redis_client()
    paused = redis.get("trading:paused") == "1"
    return {
        "current_coins": engine.current_coins,
        "positions": engine.positions,
        "balances": engine.trader.fetch_balance(),
        "paused": paused,
    }

@app.get("/api/trades")
async def trades(limit: int = 20):
    engine = get_engine()
    trades = engine.trade_history[-limit:]
    return {"trades": trades}

@app.get("/api/profit")
async def profit():
    engine = get_engine()
    return engine.get_profit_summary()

@app.get("/api/config")
async def config():
    return {
        "exchange_id": settings.EXCHANGE_ID,
        "trading_mode": settings.TRADING_MODE,
        "base_currency": settings.BASE_CURRENCY,
        "max_coins": settings.MAX_COINS,
        "ollama_model": settings.OLLAMA_MODEL,
        "web_port": settings.WEB_PORT,
    }
