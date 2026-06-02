FROM python:3.11-slim

WORKDIR /app

# Install system dependencies: gcc for potential C extensions, curl for health check
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

EXPOSE 8083

# Health check using the /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8083/health || exit 1

CMD ["python", "-m", "src.main"]
