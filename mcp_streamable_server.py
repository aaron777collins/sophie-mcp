#!/usr/bin/env python3
"""
Sophie MCP Server - Streamable HTTP transport for ElevenLabs.

This implements the proper MCP protocol over HTTP with SSE streaming.
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
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.routing import Mount, Route
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
    completed_at: Optional[float] = None


# In-memory request store
requests: Dict[str, SophieRequest] = {}

# Create MCP server
mcp_server = Server("sophie-mcp")


@mcp_server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="ask_sophie",
            description="Start an async request to Sophie. Returns request_id immediately. Use check_sophie to poll.",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "Question for Sophie"},
                    "context": {"type": "string", "description": "Context (e.g., 'voice call')"}
                },
                "required": ["question"]
            }
        ),
        Tool(
            name="check_sophie", 
            description="Check status of a Sophie request. Returns processing/complete/error.",
            inputSchema={
                "type": "object",
                "properties": {
                    "request_id": {"type": "string", "description": "Request ID from ask_sophie"}
                },
                "required": ["request_id"]
            }
        )
    ]


@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    if name == "ask_sophie":
        question = arguments.get("question", "")
        context = arguments.get("context", "Voice request")
        
        if not question:
            return [TextContent(type="text", text=json.dumps({"error": "Question required"}))]
        
        request_id = secrets.token_urlsafe(16)
        req = SophieRequest(id=request_id, question=question, context=context)
        requests[request_id] = req
        
        asyncio.create_task(process_request(req))
        
        return [TextContent(type="text", text=json.dumps({
            "request_id": request_id,
            "status": "processing"
        }))]
    
    elif name == "check_sophie":
        request_id = arguments.get("request_id", "")
        
        req = requests.get(request_id)
        if not req:
            return [TextContent(type="text", text=json.dumps({"error": "Not found"}))]
        
        result = {"request_id": request_id, "status": req.status}
        if req.status in ("complete", "error"):
            result["response"] = req.response
        
        return [TextContent(type="text", text=json.dumps(result))]
    
    return [TextContent(type="text", text=json.dumps({"error": "Unknown tool"}))]


async def process_request(req: SophieRequest):
    """Call Sophie via Clawdbot."""
    headers = {
        "Authorization": f"Bearer {CLAWDBOT_GATEWAY_TOKEN}",
        "Content-Type": "application/json"
    }
    
    prompt = f"[{req.context}]\n\n{req.question}\n\nRespond conversationally - this will be spoken aloud."
    
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
                req.response = f"Error: {response.status_code}"
                req.status = "error"
    except Exception as e:
        req.response = str(e)
        req.status = "error"
    
    req.completed_at = time.time()


# Auth middleware
class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Skip auth for health endpoint
        if request.url.path == "/health":
            return await call_next(request)
        
        # Check bearer token
        if MCP_SECRET:
            auth = request.headers.get("authorization", "")
            if not auth.lower().startswith("bearer "):
                return JSONResponse({"error": "Unauthorized"}, status_code=401)
            token = auth[7:]
            if not secrets.compare_digest(token, MCP_SECRET):
                return JSONResponse({"error": "Invalid token"}, status_code=401)
        
        return await call_next(request)


# Track active transports per session
transports: Dict[str, StreamableHTTPServerTransport] = {}
mcp_tasks: Dict[str, asyncio.Task] = {}


async def handle_mcp(request: Request):
    """Handle MCP requests - creates transport per session."""
    # Get or create session ID
    session_id = request.headers.get("mcp-session-id") or secrets.token_urlsafe(16)
    
    if session_id not in transports:
        # Create new transport for this session
        transport = StreamableHTTPServerTransport(
            mcp_session_id=session_id,
            is_json_response_enabled=True
        )
        transports[session_id] = transport
        
        # Start MCP server for this transport
        async def run_mcp():
            async with transport.connect() as (read_stream, write_stream):
                await mcp_server.run(read_stream, write_stream, mcp_server.create_initialization_options())
        
        mcp_tasks[session_id] = asyncio.create_task(run_mcp())
    
    transport = transports[session_id]
    
    # Handle the request through the transport
    async def receive():
        body = await request.body()
        return {"type": "http.request", "body": body}
    
    response_started = False
    response_body = []
    response_headers = []
    status_code = 200
    
    async def send(message):
        nonlocal response_started, status_code
        if message["type"] == "http.response.start":
            response_started = True
            status_code = message.get("status", 200)
            response_headers.extend(message.get("headers", []))
        elif message["type"] == "http.response.body":
            body = message.get("body", b"")
            if body:
                response_body.append(body)
    
    await transport.handle_request(request.scope, receive, send)
    
    # Build response
    headers_dict = {k.decode(): v.decode() for k, v in response_headers}
    return Response(
        content=b"".join(response_body),
        status_code=status_code,
        headers=headers_dict
    )


async def health(request):
    return JSONResponse({"status": "ok", "service": "sophie-mcp", "transport": "streamable_http"})


# Also keep REST endpoints as fallback
async def rest_ask_sophie(request: Request):
    """REST fallback for ask_sophie."""
    data = await request.json()
    question = data.get("question", "")
    context = data.get("context", "Voice request")
    
    if not question:
        return JSONResponse({"error": "Question required"}, status_code=400)
    
    request_id = secrets.token_urlsafe(16)
    req = SophieRequest(id=request_id, question=question, context=context)
    requests[request_id] = req
    
    asyncio.create_task(process_request(req))
    
    return JSONResponse({
        "request_id": request_id,
        "status": "processing"
    })


async def rest_check_sophie(request: Request):
    """REST fallback for check_sophie."""
    data = await request.json()
    request_id = data.get("request_id", "")
    
    req = requests.get(request_id)
    if not req:
        return JSONResponse({"error": "Not found"}, status_code=404)
    
    result = {"request_id": request_id, "status": req.status}
    if req.status in ("complete", "error"):
        result["response"] = req.response
    
    return JSONResponse(result)


app = Starlette(
    routes=[
        Route("/health", health),
        Route("/mcp", handle_mcp, methods=["GET", "POST", "DELETE"]),
        # REST fallbacks
        Route("/ask_sophie", rest_ask_sophie, methods=["POST"]),
        Route("/check_sophie", rest_check_sophie, methods=["POST"]),
    ],
    middleware=[Middleware(AuthMiddleware)]
)


if __name__ == "__main__":
    print("🧠 Sophie MCP Server (Streamable HTTP)")
    print(f"   Gateway: {CLAWDBOT_GATEWAY_URL}")
    print(f"   Auth: {'Bearer token required' if MCP_SECRET else 'OPEN'}")
    print("   MCP endpoint: /mcp")
    print("   REST fallback: /ask_sophie, /check_sophie")
    uvicorn.run(app, host="0.0.0.0", port=8014)
