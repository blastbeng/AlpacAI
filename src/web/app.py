import asyncio
import json
import logging
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from src.config.settings import settings
from src.utils.redis_client import get_redis_client, check_redis_connection
from src.llm.prompts import get_cached_news_summary
from src.exchanges.market_data import get_quotes, get_multi_timeframe_bars

app = FastAPI(title="Crypto Trading Bot")

logger = logging.getLogger(__name__)

# Serve static files (dashboard)
app.mount("/static", StaticFiles(directory="src/web/static"), name="static")

# Global engine reference
_engine = None

def set_engine(engine):
    global _engine
    _engine = engine
    logger.info("Trading engine attached to web server")

def get_engine():
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return _engine

@app.get("/")
async def root():
    return FileResponse("src/web/static/index.html")

@app.get("/health")
async def health():
    redis_ok = check_redis_connection()
    return {
        "status": "ok" if redis_ok else "degraded",
        "redis": "connected" if redis_ok else "disconnected",
    }

@app.get("/api/status")
def status():
    engine = get_engine()
    redis = get_redis_client()
    paused = redis.get("trading:paused") == "1"
    return {
        "current_symbols": engine.current_symbols,
        "positions": engine.positions,
        "balances": engine.trader.fetch_balance(),
        "paused": paused,
    }

@app.get("/api/trades")
def trades(limit: int = 0):
    engine = get_engine()
    # limit is ignored – open trades are always all current positions
    return {"trades": engine.get_open_trades()}

@app.get("/api/profit")
def profit():
    engine = get_engine()
    return engine.get_profit_summary()

@app.get("/api/performance")
def performance():
    engine = get_engine()
    return engine.get_performance_summary()

@app.get("/api/risk")
def risk():
    engine = get_engine()
    return engine.get_risk_metrics()

@app.get("/api/news")
async def news():
    engine = get_engine()
    symbols = engine.current_symbols
    result = {}
    for entry in symbols:
        symbol = entry["symbol"]
        try:
            news_data = await run_in_threadpool(get_cached_news_summary, symbol)
            result[symbol] = news_data["summary"]
        except Exception:
            result[symbol] = "Could not generate summary."
    return result

@app.get("/api/history")
def history(limit: int = 50):
    engine = get_engine()
    trades = engine.trade_history[-limit:]
    return trades

@app.post("/api/pause")
def pause():
    engine = get_engine()
    redis = engine.redis
    redis.set("trading:paused", "1")
    redis.set("trading:pause_source", "manual")
    redis.delete("trading:pause_start")
    redis.delete("trading:pause_duration")
    redis.delete("trading:pause_reason")
    redis.delete("trading:llm_pause_time")
    return {"status": "paused"}

@app.post("/api/resume")
def resume():
    engine = get_engine()
    redis = engine.redis
    keys = [
        "trading:paused",
        "trading:pause_source",
        "trading:pause_start",
        "trading:pause_duration",
        "trading:pause_reason",
        "trading:llm_pause_time",
    ]
    for key in keys:
        redis.delete(key)
    return {"status": "resumed"}

@app.post("/api/sell")
def sell(symbol: str = None):
    engine = get_engine()
    if symbol:
        asyncio.create_task(engine.sell_position(symbol))
        return {"status": f"selling {symbol}"}
    else:
        asyncio.create_task(engine.sell_all_positions())
        return {"status": "selling all"}

@app.post("/api/reload")
def reload():
    settings.reload()
    return {"status": "reloaded"}

@app.post("/api/restart")
def restart():
    """
    Restart the entire application by exiting the process.
    Docker (or the process manager) will bring it back up.
    """
    os._exit(0)

@app.get("/api/config")
def config():
    mind_provider = settings.LLM_MIND_PROVIDER or settings.LLM_PROVIDER
    actuator_provider = settings.LLM_ACTUATOR_PROVIDER or settings.LLM_PROVIDER
    if mind_provider == "ollama":
        mind_model = settings.OLLAMA_MIND_MODEL
    else:
        mind_model = settings.OPENAI_MIND_MODEL
    if actuator_provider == "ollama":
        actuator_model = settings.OLLAMA_ACTUATOR_MODEL
    else:
        actuator_model = settings.OPENAI_ACTUATOR_MODEL

    return {
        "trading_mode": settings.TRADING_MODE,
        "base_currency": settings.BASE_CURRENCY,
        "max_symbols": settings.MAX_SYMBOLS,
        "llm_mind_provider": mind_provider,
        "llm_mind_model": mind_model,
        "llm_actuator_provider": actuator_provider,
        "llm_actuator_model": actuator_model,
        "web_port": settings.WEB_PORT,
    }

@app.get("/api/ohlcv/{symbol:path}")
async def ohlcv(symbol: str, timeframe: str = "1h", limit: int = 24):
    engine = get_engine()
    try:
        bars = await asyncio.to_thread(
            get_multi_timeframe_bars, engine.data_client, symbol, [timeframe], limit=limit
        )
        candles = bars.get(timeframe, [])
        result = []
        for candle in candles:
            result.append({
                "timestamp": candle[0],
                "open": candle[1],
                "high": candle[2],
                "low": candle[3],
                "close": candle[4],
                "volume": candle[5],
            })
        return {"symbol": symbol, "timeframe": timeframe, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ticker/{symbol:path}")
async def ticker(symbol: str):
    engine = get_engine()
    # 1) Try the WebSocket cache first
    ticker_data = engine.ws_manager.get_ticker(symbol)
    if ticker_data is not None:
        return {
            "symbol": symbol,
            "last": ticker_data.get("last"),
            "bid": ticker_data.get("bid"),
            "ask": ticker_data.get("ask"),
            "change_24h": ticker_data.get("percentage"),
        }

    # 2) Fallback to REST only if WebSocket is unhealthy
    if not engine.ws_manager.healthy:
        logger.warning(f"WebSocket unhealthy, falling back to REST for {symbol}")
        try:
            quotes = await asyncio.to_thread(
                get_quotes, engine.data_client, [symbol]
            )
            q = quotes.get(symbol)
            if q:
                return {
                    "symbol": symbol,
                    "last": q.get("last"),
                    "bid": q.get("bid"),
                    "ask": q.get("ask"),
                    "change_24h": q.get("change_24h"),
                }
        except Exception as e:
            logger.warning(f"REST ticker fetch failed for {symbol}: {e}")

    # 3) WebSocket healthy but no ticker yet (symbol just subscribed)
    return {
        "symbol": symbol,
        "last": None,
        "bid": None,
        "ask": None,
        "change_24h": None,
    }

@app.get("/api/tickers")
async def tickers(symbols: str = ""):
    """Return cached tickers for a comma-separated list of symbols."""
    engine = get_engine()
    if not symbols:
        return {}
    symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
    result = {}

    # 1) Try WebSocket cache first
    for sym in symbol_list:
        t = engine.ws_manager.get_ticker(sym)
        if t:
            result[sym] = {
                "last": t.get("last"),
                "bid": t.get("bid"),
                "ask": t.get("ask"),
                "change_24h": t.get("percentage"),
            }
        else:
            result[sym] = None  # mark as missing

    # 2) Fallback to REST only if WebSocket unhealthy AND we have missing symbols
    if not engine.ws_manager.healthy:
        missing = [sym for sym in symbol_list if result.get(sym) is None]
        if missing:
            try:
                quotes = await asyncio.to_thread(
                    get_quotes, engine.data_client, missing
                )
                for sym in missing:
                    q = quotes.get(sym)
                    if q:
                        result[sym] = {
                            "last": q.get("last"),
                            "bid": q.get("bid"),
                            "ask": q.get("ask"),
                            "change_24h": q.get("change_24h"),
                        }
                    else:
                        result[sym] = {"last": None, "bid": None, "ask": None, "change_24h": None}
            except Exception as e:
                logger.warning(f"REST tickers fallback failed: {e}")
                for sym in missing:
                    result[sym] = {"last": None, "bid": None, "ask": None, "change_24h": None}

    # 3) Fill any remaining None placeholders with null dicts
    for sym in symbol_list:
        if result.get(sym) is None:
            result[sym] = {"last": None, "bid": None, "ask": None, "change_24h": None}

    return result

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket client connected")
    try:
        while True:
            try:
                engine = get_engine()
                redis = get_redis_client()
                data = await asyncio.to_thread(lambda: {
                    "current_symbols": engine.current_symbols,
                    "positions": engine.positions,
                    "balances": engine.trader.fetch_balance(),
                    "trades": engine.trade_history,
                    "profit": engine.get_profit_summary(),
                    "performance": engine.get_performance_summary(),
                    "paused": redis.get("trading:paused") == "1",
                })
                await websocket.send_text(json.dumps(data))
            except HTTPException:
                # Engine not ready yet, send empty
                await websocket.send_text(json.dumps({"status": "initializing"}))
            except Exception as e:
                logger.error(f"WebSocket error: {e}", exc_info=True)
                break
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
