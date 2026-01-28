#!/usr/bin/env python3
"""
Sophie MCP HTTP Server - HTTP wrapper around the MCP tools.

Endpoints:
- POST /ask_sophie - Start async request, returns request_id
- POST /check_sophie - Check status of request

Both require 'secret' in request body.
"""

import asyncio
import json
import os
import secrets
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import httpx
from dotenv import load_dotenv

load_dotenv()

# Config
CLAWDBOT_GATEWAY_URL = os.getenv("CLAWDBOT_GATEWAY_URL", "http://localhost:18789")
CLAWDBOT_GATEWAY_TOKEN = os.getenv("CLAWDBOT_GATEWAY_TOKEN", "")
MCP_SECRET = os.getenv("SOPHIE_MCP_SECRET", "")

if not MCP_SECRET:
    print("⚠️  Warning: SOPHIE_MCP_SECRET not set - API is UNPROTECTED!")

app = FastAPI(title="Sophie MCP HTTP Server")


def check_auth(secret_body: str = None, authorization: str = None) -> bool:
    """Check auth from body secret OR bearer token header."""
    if not MCP_SECRET:
        return True  # No secret = open
    
    # Check body secret
    if secret_body and secrets.compare_digest(secret_body, MCP_SECRET):
        return True
    
    # Check bearer token
    if authorization:
        if authorization.lower().startswith("bearer "):
            token = authorization[7:]
            if secrets.compare_digest(token, MCP_SECRET):
                return True
    
    return False


@dataclass
class Request:
    id: str
    question: str
    context: str
    status: str = "processing"
    response: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


# In-memory request store
requests: Dict[str, Request] = {}


class AskSophieRequest(BaseModel):
    question: str
    context: str = "Voice request"
    secret: str = ""  # Optional if using bearer token


class CheckSophieRequest(BaseModel):
    request_id: str
    secret: str = ""  # Optional if using bearer token


@app.post("/ask_sophie")
async def ask_sophie(req: AskSophieRequest, authorization: str = Header(None)):
    """Start an async request to Sophie."""
    if not check_auth(req.secret, authorization):
        raise HTTPException(status_code=401, detail="Invalid secret or token")
    
    if not req.question:
        raise HTTPException(status_code=400, detail="Question is required")
    
    # Generate request ID
    request_id = secrets.token_urlsafe(16)
    
    # Create request record
    request_obj = Request(
        id=request_id,
        question=req.question,
        context=req.context
    )
    requests[request_id] = request_obj
    
    # Start background task
    asyncio.create_task(process_sophie_request(request_obj))
    
    return {
        "request_id": request_id,
        "status": "processing",
        "message": "Request started. Use /check_sophie to poll for response."
    }


@app.post("/check_sophie")
async def check_sophie(req: CheckSophieRequest, authorization: str = Header(None)):
    """Check status of a Sophie request."""
    if not check_auth(req.secret, authorization):
        raise HTTPException(status_code=401, detail="Invalid secret or token")
    
    request_obj = requests.get(req.request_id)
    if not request_obj:
        raise HTTPException(status_code=404, detail="Request not found")
    
    result = {
        "request_id": req.request_id,
        "status": request_obj.status,
        "elapsed_seconds": round(time.time() - request_obj.created_at, 1)
    }
    
    if request_obj.status in ("complete", "error"):
        result["response"] = request_obj.response
    
    return result


@app.get("/health")
async def health():
    return {"status": "ok", "service": "sophie-mcp"}


async def process_sophie_request(req: Request):
    """Background task to process a Sophie request."""
    
    headers = {
        "Authorization": f"Bearer {CLAWDBOT_GATEWAY_TOKEN}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""[{req.context}]

{req.question}

Please respond conversationally and concisely - this will be spoken aloud in a voice call."""

    payload = {
        "model": "anthropic/claude-opus-4-5",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2048,
        "stream": False
    }
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{CLAWDBOT_GATEWAY_URL}/v1/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                req.response = data["choices"][0]["message"]["content"]
                req.status = "complete"
            else:
                req.response = f"Error from Sophie: {response.status_code}"
                req.status = "error"
                
    except httpx.TimeoutException:
        req.response = "Sophie took too long to respond. Try a simpler question."
        req.status = "error"
    except Exception as e:
        req.response = f"Error: {str(e)}"
        req.status = "error"
    
    req.completed_at = time.time()
    
    # Clean up old requests
    if len(requests) > 100:
        sorted_reqs = sorted(requests.items(), key=lambda x: x[1].created_at)
        for key, _ in sorted_reqs[:-100]:
            del requests[key]


if __name__ == "__main__":
    import uvicorn
    print("🧠 Sophie MCP HTTP Server")
    print(f"   Gateway: {CLAWDBOT_GATEWAY_URL}")
    print(f"   Secret: {'configured' if MCP_SECRET else 'NOT SET (open)'}")
    print("   Endpoints: /ask_sophie, /check_sophie, /health")
    uvicorn.run(app, host="0.0.0.0", port=8014)
