import os
import random
import httpx
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Optional, List
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
FREEMODEL_KEYS = load_keys("FREEMODEL_KEYS")
AEROLINK_KEYS = load_keys("AEROLINK_KEYS")

FREEMODEL_BASE = "https://cc.freemodel.dev"

FREEMODEL_MODELS = [
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-fable-5",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "claude-haiku-4-5-20251001",
]
AEROLINK_BASE = "https://capi.aerolink.lat/v1"

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

# Aerolink Anthropic-compatible models (for /v1/messages only)
AEROLINK_MODELS = [
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "claude-haiku-4-5-20251001",
]

PROVIDER_KEYS = {
    "groq": GROQ_KEYS,
    "cerebras": CEREBRAS_KEYS,
    "sambanova": SAMBANOVA_KEYS,
    "mistral": MISTRAL_KEYS,
    "poolside": POOLSIDE_KEYS,
    "bluesminds": BLUESMINDS_KEYS,
    "freemodel": FREEMODEL_KEYS,
    "aerolink": AEROLINK_KEYS,
}


def shuffled_keys(provider: str) -> List[str]:
    keys = PROVIDER_KEYS.get(provider, [])
    keys = list(keys)
    random.shuffle(keys)
    return keys


def check_auth(authorization: Optional[str]):
    if not authorization or authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")


async def post_with_fallback(keys: List[str], url: str, body: dict, extra_headers: dict = {}):
    """Try each key until one works, return (response, status_code)"""
    last_resp = None
    last_status = 500
    for key in keys:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
            **extra_headers,
        }
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(url, headers=headers, json=body)
                if resp.status_code not in (401, 403, 429, 500, 502, 503):
                    return resp.json(), resp.status_code
                last_resp = resp.json()
                last_status = resp.status_code
        except Exception:
            continue
    return last_resp, last_status


async def stream_with_fallback(keys: List[str], url: str, body: dict, extra_headers: dict = {}):
    """Try each key for streaming until one works"""
    for key in keys:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
            **extra_headers,
        }
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream("POST", url, headers=headers, json=body) as resp:
                    if resp.status_code not in (401, 403, 429, 500, 502, 503):
                        async for chunk in resp.aiter_bytes():
                            yield chunk
                        return
        except Exception:
            continue


@app.get("/v1/models")
async def list_models(authorization: Optional[str] = Header(None)):
    check_auth(authorization)
    data = [
        {"id": mid, "object": "model", "created": 1700000000, "owned_by": "KvasAI"}
        for mid in MODELS
    ]
    existing_ids = {m["id"] for m in data}

    # Add aerolink Claude models
    for mid in AEROLINK_MODELS:
        if mid not in existing_ids:
            data.append({"id": mid, "object": "model", "created": 1626777600, "owned_by": "KvasAI"})
            existing_ids.add(mid)

    # Add freemodel static models
    for mid in FREEMODEL_MODELS:
        if mid not in existing_ids:
            data.append({"id": mid, "object": "model", "created": 1700000000, "owned_by": "KvasAI"})
            existing_ids.add(mid)

    # Fetch freemodel models dynamically
    if FREEMODEL_KEYS:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    f"{FREEMODEL_BASE}/models",
                    headers={"Authorization": f"Bearer {FREEMODEL_KEYS[0]}"},
                )
                if resp.status_code == 200:
                    for m in resp.json().get("data", []):
                        if m["id"] not in existing_ids:
                            data.append({
                                "id": m["id"],
                                "object": "model",
                                "created": m.get("created", 1700000000),
                                "owned_by": "KvasAI",
                            })
                            existing_ids.add(m["id"])
        except Exception:
            pass

    return {"object": "list", "data": data}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, authorization: Optional[str] = Header(None)):
    check_auth(authorization)
    body = await request.json()
    model = body.get("model", "")
    stream = body.get("stream", False)

    if model in MODELS:
        provider, base_url = MODELS[model]
        keys = shuffled_keys(provider)
        url = f"{base_url}/chat/completions"
    elif FREEMODEL_KEYS:
        keys = shuffled_keys("freemodel")
        url = f"{FREEMODEL_BASE}/chat/completions"
    else:
        raise HTTPException(status_code=404, detail=f"Model '{model}' not found")

    if stream:
        return StreamingResponse(
            stream_with_fallback(keys, url, body),
            media_type="text/event-stream"
        )
    else:
        result, status = await post_with_fallback(keys, url, body)
        return JSONResponse(content=result, status_code=status)


@app.post("/v1/messages")
async def anthropic_messages(request: Request, authorization: Optional[str] = Header(None)):
    check_auth(authorization)
    body = await request.json()
    stream = body.get("stream", False)
    extra_headers = {
        "anthropic-version": request.headers.get("anthropic-version", "2023-06-01"),
        "user-agent": "claude-code/1.0",
    }

    # Only aerolink for Anthropic-compatible endpoint
    if AEROLINK_KEYS:
        keys = shuffled_keys("aerolink")
        url = f"{AEROLINK_BASE}/messages"
    else:
        raise HTTPException(status_code=503, detail="No Anthropic-compatible keys configured")

    if stream:
        return StreamingResponse(
            stream_with_fallback(keys, url, body, extra_headers),
            media_type="text/event-stream"
        )
    else:
        result, status = await post_with_fallback(keys, url, body, extra_headers)
        return JSONResponse(content=result, status_code=status)


@app.get("/")
async def root():
    return {"message": "KvasAI API", "version": "1.0"}

