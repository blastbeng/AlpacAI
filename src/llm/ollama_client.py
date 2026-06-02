import httpx
from src.config.settings import settings

def get_ollama_response(prompt: str, system_prompt: str = "") -> str:
    """
    Send a prompt to the configured Ollama model and return the response text.
    Uses the chat completions endpoint (Ollama v0.1.7+).
    """
    url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/chat"
    headers = {"Content-Type": "application/json"}
    if settings.OLLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {settings.OLLAMA_API_KEY}"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": settings.OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
    }

    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"]
    except httpx.HTTPError as e:
        raise RuntimeError(f"Ollama request failed: {e}") from e
