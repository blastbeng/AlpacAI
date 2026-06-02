import uvicorn
from src.web.app import app
from src.config.settings import settings

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=settings.WEB_HOST,
        port=settings.WEB_PORT,
        log_level=settings.LOG_LEVEL.lower()
    )
