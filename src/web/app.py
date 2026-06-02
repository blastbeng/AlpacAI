from fastapi import FastAPI

app = FastAPI(title="Crypto Trading Bot")

@app.get("/health")
async def health():
    return {"status": "ok"}
