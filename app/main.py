from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from enum import Enum

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


class ProviderKind(str, Enum):
    openai = "openai"
    anthropic = "anthropic"
    gemini = "gemini"


@dataclass
class ProviderConfig:
    id: str
    name: str
    kind: ProviderKind
    model: str
    api_key: str
    base_url: str | None = None


class ProviderInput(BaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    kind: ProviderKind
    model: str = Field(min_length=1)
    api_key: str = Field(min_length=1)
    base_url: str | None = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    provider_ids: list[str] = Field(min_length=1)


class RelayChatRequest(BaseModel):
    topic: str = Field(min_length=1)
    provider_ids: list[str] = Field(min_length=2)
    rounds: int = Field(default=2, ge=1, le=8)


class ChatResult(BaseModel):
    provider_id: str
    provider_name: str
    ok: bool
    text: str


class RelayMessage(BaseModel):
    round_index: int
    provider_id: str
    provider_name: str
    ok: bool
    text: str


PROVIDERS: dict[str, ProviderConfig] = {}

app = FastAPI(title="Multi-LLM Connector")
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse("app/static/index.html")


@app.get("/api/providers")
async def list_providers() -> list[dict[str, str]]:
    return [
        {
            "id": p.id,
            "name": p.name,
            "kind": p.kind.value,
            "model": p.model,
            "base_url": p.base_url or "",
        }
        for p in PROVIDERS.values()
    ]


@app.post("/api/providers")
async def upsert_provider(payload: ProviderInput) -> dict[str, str]:
    PROVIDERS[payload.id] = ProviderConfig(
        id=payload.id,
        name=payload.name,
        kind=payload.kind,
        model=payload.model,
        api_key=payload.api_key,
        base_url=payload.base_url,
    )
    return {"status": "ok", "id": payload.id}


@app.delete("/api/providers/{provider_id}")
async def delete_provider(provider_id: str) -> dict[str, str]:
    if provider_id in PROVIDERS:
        PROVIDERS.pop(provider_id)
    return {"status": "ok"}


async def call_openai(provider: ProviderConfig, prompt: str) -> str:
    base = provider.base_url or "https://api.openai.com"
    url = f"{base.rstrip('/')}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {provider.api_key}"}
    data = {
        "model": provider.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }
    async with httpx.AsyncClient(timeout=90) as client:
        resp = await client.post(url, headers=headers, json=data)
    resp.raise_for_status()
    body = resp.json()
    return body["choices"][0]["message"]["content"].strip()


async def call_anthropic(provider: ProviderConfig, prompt: str) -> str:
    base = provider.base_url or "https://api.anthropic.com"
    url = f"{base.rstrip('/')}/v1/messages"
    headers = {
        "x-api-key": provider.api_key,
        "anthropic-version": "2023-06-01",
    }
    data = {
        "model": provider.model,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
    }
    async with httpx.AsyncClient(timeout=90) as client:
        resp = await client.post(url, headers=headers, json=data)
    resp.raise_for_status()
    body = resp.json()
    chunks = [x.get("text", "") for x in body.get("content", []) if x.get("type") == "text"]
    return "\n".join(chunks).strip()


async def call_gemini(provider: ProviderConfig, prompt: str) -> str:
    base = provider.base_url or "https://generativelanguage.googleapis.com"
    url = (
        f"{base.rstrip('/')}/v1beta/models/{provider.model}:generateContent"
        f"?key={provider.api_key}"
    )
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    async with httpx.AsyncClient(timeout=90) as client:
        resp = await client.post(url, json=data)
    resp.raise_for_status()
    body = resp.json()
    candidates = body.get("candidates", [])
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    return "\n".join(p.get("text", "") for p in parts).strip()


async def call_provider(provider: ProviderConfig, prompt: str) -> str:
    if provider.kind == ProviderKind.openai:
        return await call_openai(provider, prompt)
    if provider.kind == ProviderKind.anthropic:
        return await call_anthropic(provider, prompt)
    if provider.kind == ProviderKind.gemini:
        return await call_gemini(provider, prompt)
    raise ValueError(f"Unsupported provider: {provider.kind}")


async def run_provider(provider: ProviderConfig, message: str) -> ChatResult:
    try:
        text = await call_provider(provider, message)
        return ChatResult(
            provider_id=provider.id,
            provider_name=provider.name,
            ok=True,
            text=text or "(empty response)",
        )
    except Exception as exc:
        return ChatResult(
            provider_id=provider.id,
            provider_name=provider.name,
            ok=False,
            text=f"{type(exc).__name__}: {exc}",
        )


def resolve_providers(provider_ids: list[str]) -> list[ProviderConfig]:
    targets: list[ProviderConfig] = []
    missing: list[str] = []
    for pid in provider_ids:
        if pid in PROVIDERS:
            targets.append(PROVIDERS[pid])
        else:
            missing.append(pid)
    if missing:
        raise HTTPException(status_code=404, detail=f"Unknown provider ids: {', '.join(missing)}")
    return targets


@app.post("/api/chat")
async def chat(payload: ChatRequest) -> list[ChatResult]:
    targets = resolve_providers(payload.provider_ids)
    return await asyncio.gather(*(run_provider(p, payload.message) for p in targets))


@app.post("/api/chat/relay")
async def relay_chat(payload: RelayChatRequest) -> list[RelayMessage]:
    targets = resolve_providers(payload.provider_ids)

    transcript: list[str] = [f"[USER] {payload.topic}"]
    output: list[RelayMessage] = []

    for round_index in range(1, payload.rounds + 1):
        for provider in targets:
            transcript_text = "\n".join(transcript)
            prompt = (
                "Ты участвуешь в групповом чате ИИ. "
                "Ответь на последнее сообщение и продолжи дискуссию коротко и по сути.\n\n"
                "Тема дискуссии:\n"
                f"{payload.topic}\n\n"
                "Текущий диалог:\n"
                f"{transcript_text}\n\n"
                "Сейчас твой ход."
            )
            try:
                text = (await call_provider(provider, prompt)).strip() or "(empty response)"
                ok = True
            except Exception as exc:
                text = f"{type(exc).__name__}: {exc}"
                ok = False

            message = RelayMessage(
                round_index=round_index,
                provider_id=provider.id,
                provider_name=provider.name,
                ok=ok,
                text=text,
            )
            output.append(message)
            transcript.append(f"[{provider.name}] {text}")

    return output


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)
