#!/usr/bin/env python3
"""
Sophie MCP Server - Streamable HTTP transport for ElevenLabs.
"""

import asyncio
import json
import os
import secrets
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from mcp.server import Server
from mcp.server.streamable_http import StreamableHTTPServerTransport
from mcp.types import Tool, TextContent
from dotenv import load_dotenv

load_dotenv()

# Config
CLAWDBOT_GATEWAY_URL = os.getenv("CLAWDBOT_GATEWAY_URL", "http://localhost:18789")
CLAWDBOT_GATEWAY_TOKEN = os.getenv("CLAWDBOT_GATEWAY_TOKEN", "")
MCP_SECRET = os.getenv("SOPHIE_MCP_SECRET", "")

if not MCP_SECRET:
    print("⚠️  Warning: SOPHIE_MCP_SECRET not set!")


@dataclass  
class SophieRequest:
    id: str
    question: str
    context: str
    status: str = "processing"
    response: Optional[str] = None
    created_at: float = field(default_factory=time.time)


# Request store
pending_requests: Dict[str, SophieRequest] = {}

# MCP Server
mcp_server = Server("sophie-mcp")


@mcp_server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="ask_sophie",
            description="Start async request to Sophie. Returns request_id. Poll with check_sophie.",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "context": {"type": "string"}
                },
                "required": ["question"]
            }
        ),
        Tool(
            name="check_sophie", 
            description="Check Sophie request status. If request_id omitted or invalid, returns last 5 requests.",
            inputSchema={
                "type": "object",
                "properties": {
                    "request_id": {"type": "string"}
                },
                "required": []
            }
        )
    ]


@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    if name == "ask_sophie":
        question = arguments.get("question", "")
        context = arguments.get("context", "Voice request")
        
        if not question:
            return [TextContent(type="text", text='{"error":"Question required"}')]
        
        request_id = secrets.token_urlsafe(16)
        req = SophieRequest(id=request_id, question=question, context=context)
        pending_requests[request_id] = req
        
        asyncio.create_task(process_request(req))
        
        return [TextContent(type="text", text=json.dumps({
            "request_id": request_id,
            "status": "processing",
            "instruction": "IMPORTANT: Do NOT interrupt the human or mention that you're waiting. Stay silent until the response is ready."
        }))]
    
    elif name == "check_sophie":
        request_id = arguments.get("request_id", "")
        req = pending_requests.get(request_id) if request_id else None
        
        # If no request_id or invalid, return last 5 requests
        if not req:
            sorted_reqs = sorted(pending_requests.values(), key=lambda r: r.created_at, reverse=True)[:5]
            recent = []
            has_processing = False
            for r in sorted_reqs:
                entry = {
                    "request_id": r.id,
                    "status": r.status,
                    "question": r.question[:100],
                    "created_at": r.created_at,
                    "created_at_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r.created_at))
                }
                if r.status in ("complete", "error"):
                    entry["response"] = r.response
                if r.status == "processing":
                    has_processing = True
                recent.append(entry)
            result = {"recent_requests": recent}
            if has_processing:
                result["instruction"] = "IMPORTANT: Do NOT interrupt the human or mention that you're waiting. Stay silent until the response is ready."
            else:
                result["instruction"] = "Wait for a natural pause in the conversation before mentioning these results. Do not interrupt the user."
            return [TextContent(type="text", text=json.dumps(result))]
        
        result = {"request_id": request_id, "status": req.status}
        if req.status in ("complete", "error"):
            result["response"] = req.response
        elif req.status == "processing":
            result["instruction"] = "IMPORTANT: Do NOT interrupt the human or mention that you're waiting. Stay silent until the response is ready."
        
        return [TextContent(type="text", text=json.dumps(result))]
    
    return [TextContent(type="text", text='{"error":"Unknown tool"}')]


async def process_request(req: SophieRequest):
    """Call Sophie via Clawdbot."""
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{CLAWDBOT_GATEWAY_URL}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {CLAWDBOT_GATEWAY_TOKEN}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "anthropic/claude-opus-4-5",
                    "messages": [{"role": "user", "content": f"[{req.context}]\n\n{req.question}\n\nRespond conversationally."}],
                    "max_tokens": 2048
                }
            )
            if response.status_code == 200:
                req.response = response.json()["choices"][0]["message"]["content"]
                req.status = "complete"
            else:
                req.response = f"Error: {response.status_code}"
                req.status = "error"
    except Exception as e:
        req.response = str(e)
        req.status = "error"


def check_auth(request: Request) -> bool:
    """Check bearer token auth."""
    if not MCP_SECRET:
        return True
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return secrets.compare_digest(auth[7:], MCP_SECRET)
    return False


# Global transport and connection state
transport: Optional[StreamableHTTPServerTransport] = None
mcp_task: Optional[asyncio.Task] = None


async def setup_mcp():
    """Set up the MCP transport and server."""
    global transport, mcp_task
    
    transport = StreamableHTTPServerTransport(
        mcp_session_id=None,  # Stateless
        is_json_response_enabled=True
    )
    
    async def run_server():
        async with transport.connect() as (read_stream, write_stream):
            await mcp_server.run(read_stream, write_stream, mcp_server.create_initialization_options())
    
    mcp_task = asyncio.create_task(run_server())
    # Give it a moment to initialize
    await asyncio.sleep(0.1)


@asynccontextmanager
async def lifespan(app):
    """App lifespan - setup MCP on startup."""
    await setup_mcp()
    yield
    if mcp_task:
        mcp_task.cancel()


async def handle_mcp(request: Request):
    """Handle MCP protocol requests."""
    if not check_auth(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    
    if transport is None:
        return JSONResponse({"error": "MCP not initialized"}, status_code=503)
    
    # Create ASGI interface
    body = await request.body()
    
    response_data = {"status": 200, "headers": [], "body": b""}
    
    async def receive():
        return {"type": "http.request", "body": body}
    
    async def send(message):
        if message["type"] == "http.response.start":
            response_data["status"] = message.get("status", 200)
            response_data["headers"] = message.get("headers", [])
        elif message["type"] == "http.response.body":
            response_data["body"] += message.get("body", b"")
    
    try:
        await transport.handle_request(request.scope, receive, send)
    except Exception as e:
        print(f"MCP error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)
    
    headers = {k.decode() if isinstance(k, bytes) else k: v.decode() if isinstance(v, bytes) else v 
               for k, v in response_data["headers"]}
    
    return Response(
        content=response_data["body"],
        status_code=response_data["status"],
        headers=headers
    )


# REST fallbacks
async def rest_ask(request: Request):
    if not check_auth(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    
    data = await request.json()
    result = await call_tool("ask_sophie", data)
    return JSONResponse(json.loads(result[0].text))


async def rest_check(request: Request):
    if not check_auth(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    
    try:
        data = await request.json()
    except:
        data = {}
    result = await call_tool("check_sophie", data)
    return JSONResponse(json.loads(result[0].text))


async def health(request):
    return JSONResponse({"status": "ok", "service": "sophie-mcp"})


app = Starlette(
    routes=[
        Route("/health", health),
        Route("/mcp", handle_mcp, methods=["GET", "POST", "DELETE"]),
        Route("/ask_sophie", rest_ask, methods=["POST"]),
        Route("/check_sophie", rest_check, methods=["GET", "POST"]),
    ],
    lifespan=lifespan
)

if __name__ == "__main__":
    print("🧠 Sophie MCP Server")
    print(f"   Gateway: {CLAWDBOT_GATEWAY_URL}")
    print(f"   Auth: {'Required' if MCP_SECRET else 'OPEN'}")
    uvicorn.run(app, host="0.0.0.0", port=8014)
