import os
import random
import httpx
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

API_KEY = os.getenv("API_KEY", "KvasAiapi")

def load_keys(env_var):
    return [k.strip() for k in os.getenv(env_var, "").split(",") if k.strip()]

def load_cf_keys():
    raw = os.getenv("CF_KEYS", "")
    result = []
    for pair in raw.split(","):
        pair = pair.strip()
        if ":" in pair:
            token, account = pair.split(":", 1)
            result.append((token.strip(), account.strip()))
    return result

GROQ_KEYS = load_keys("GROQ_KEYS")
CEREBRAS_KEYS = load_keys("CEREBRAS_KEYS")
SAMBANOVA_KEYS = load_keys("SAMBANOVA_KEYS")
MISTRAL_KEYS = load_keys("MISTRAL_KEYS")
CF_KEYS = load_cf_keys()
POOLSIDE_KEYS = load_keys("POOLSIDE_KEYS")
BLUESMINDS_KEYS = load_keys("BLUESMINDS_KEYS")

MODELS = {
    # Groq
    "llama-3.3-70b-versatile": ("groq", "https://api.groq.com/openai/v1"),
    "llama-3.1-8b-instant": ("groq", "https://api.groq.com/openai/v1"),
    "llama3-70b-8192": ("groq", "https://api.groq.com/openai/v1"),
    "llama3-8b-8192": ("groq", "https://api.groq.com/openai/v1"),
    "mixtral-8x7b-32768": ("groq", "https://api.groq.com/openai/v1"),
    "gemma2-9b-it": ("groq", "https://api.groq.com/openai/v1"),
    "gemma-7b-it": ("groq", "https://api.groq.com/openai/v1"),
    "llama-3.3-70b-specdec": ("groq", "https://api.groq.com/openai/v1"),
    "llama-3.2-90b-vision-preview": ("groq", "https://api.groq.com/openai/v1"),
    "llama-3.2-11b-vision-preview": ("groq", "https://api.groq.com/openai/v1"),
    "llama-3.2-3b-preview": ("groq", "https://api.groq.com/openai/v1"),
    "llama-3.2-1b-preview": ("groq", "https://api.groq.com/openai/v1"),
    "deepseek-r1-distill-llama-70b": ("groq", "https://api.groq.com/openai/v1"),
    "qwen-qwq-32b": ("groq", "https://api.groq.com/openai/v1"),
    "mistral-saba-24b": ("groq", "https://api.groq.com/openai/v1"),
    # Cerebras
    "llama3.1-8b": ("cerebras", "https://api.cerebras.ai/v1"),
    "llama3.1-70b": ("cerebras", "https://api.cerebras.ai/v1"),
    "llama-4-scout-17b-16e-instruct": ("cerebras", "https://api.cerebras.ai/v1"),
    # SambaNova
    "Meta-Llama-3.1-405B-Instruct": ("sambanova", "https://api.sambanova.ai/v1"),
    "Meta-Llama-3.1-70B-Instruct": ("sambanova", "https://api.sambanova.ai/v1"),
    "Meta-Llama-3.1-8B-Instruct": ("sambanova", "https://api.sambanova.ai/v1"),
    "Meta-Llama-3.2-3B-Instruct": ("sambanova", "https://api.sambanova.ai/v1"),
    "Qwen2.5-72B-Instruct": ("sambanova", "https://api.sambanova.ai/v1"),
    "QwQ-32B": ("sambanova", "https://api.sambanova.ai/v1"),
    "DeepSeek-R1-Distill-Llama-70B": ("sambanova", "https://api.sambanova.ai/v1"),
    # Mistral
    "mistral-large-latest": ("mistral", "https://api.mistral.ai/v1"),
    "mistral-small-latest": ("mistral", "https://api.mistral.ai/v1"),
    "mistral-medium-latest": ("mistral", "https://api.mistral.ai/v1"),
    "codestral-latest": ("mistral", "https://api.mistral.ai/v1"),
    "open-mistral-nemo": ("mistral", "https://api.mistral.ai/v1"),
    "open-mixtral-8x22b": ("mistral", "https://api.mistral.ai/v1"),
    # Poolside
    "poolside-mav-1": ("poolside", "https://api.poolside.ai/v1"),
    # Bluesminds
    "blues-1": ("bluesminds", "https://api.bluesminds.ai/v1"),
}


def get_key(provider: str):
    if provider == "groq":
        return random.choice(GROQ_KEYS), None
    elif provider == "cerebras":
        return random.choice(CEREBRAS_KEYS), None
    elif provider == "sambanova":
        return random.choice(SAMBANOVA_KEYS), None
    elif provider == "mistral":
        return random.choice(MISTRAL_KEYS), None
    elif provider == "cloudflare":
        token, account = random.choice(CF_KEYS)
        return token, account
    elif provider == "poolside":
        return random.choice(POOLSIDE_KEYS), None
    elif provider == "bluesminds":
        return random.choice(BLUESMINDS_KEYS), None
    return None, None


def check_auth(authorization: Optional[str]):
    if not authorization or authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/v1/models")
async def list_models(authorization: Optional[str] = Header(None)):
    check_auth(authorization)
    return {
        "object": "list",
        "data": [
            {"id": model_id, "object": "model", "created": 1700000000, "owned_by": "KvasAI"}
            for model_id in MODELS
        ]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, authorization: Optional[str] = Header(None)):
    check_auth(authorization)
    body = await request.json()
    model = body.get("model", "")

    if model not in MODELS:
        raise HTTPException(status_code=404, detail=f"Model '{model}' not found")

    provider, base_url = MODELS[model]
    api_key, _ = get_key(provider)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    stream = body.get("stream", False)

    async with httpx.AsyncClient(timeout=120) as client:
        if stream:
            async def generate():
                async with client.stream("POST", f"{base_url}/chat/completions", headers=headers, json=body) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
            return StreamingResponse(generate(), media_type="text/event-stream")
        else:
            resp = await client.post(f"{base_url}/chat/completions", headers=headers, json=body)
            return JSONResponse(content=resp.json(), status_code=resp.status_code)


@app.get("/")
async def root():
    return {"message": "KvasAI API", "version": "1.0"}

