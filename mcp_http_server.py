#!/usr/bin/env python3
"""
Sophie MCP Server - Proper MCP HTTP/SSE transport for ElevenLabs.

Exposes ask_sophie and check_sophie as MCP tools over HTTP.
"""

import asyncio
import json
import os
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.responses import JSONResponse
from mcp.server import Server
from mcp.server.sse import SseServerTransport
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

# Create MCP server
mcp_server = Server("sophie-mcp")


@mcp_server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="ask_sophie",
            description="""Start an async request to Sophie (the main AI assistant with full capabilities).
Returns a request_id immediately. Use check_sophie to poll for the response.
Sophie has access to: calendar, email, files, web search, memory, and all tools.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question or request for Sophie"
                    },
                    "context": {
                        "type": "string",
                        "description": "Context about the request (e.g., 'voice call with Aaron')"
                    }
                },
                "required": ["question"]
            }
        ),
        Tool(
            name="check_sophie",
            description="""Check the status of a Sophie request.
Returns status: "processing" | "complete" | "error" and the response when complete.
Poll this until status is "complete".""",
            inputSchema={
                "type": "object",
                "properties": {
                    "request_id": {
                        "type": "string",
                        "description": "The request ID from ask_sophie"
                    }
                },
                "required": ["request_id"]
            }
        )
    ]


@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    
    if name == "ask_sophie":
        return await handle_ask_sophie(arguments)
    elif name == "check_sophie":
        return await handle_check_sophie(arguments)
    
    return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]


async def handle_ask_sophie(arguments: dict) -> list[TextContent]:
    """Start an async request to Sophie."""
    question = arguments.get("question", "")
    context = arguments.get("context", "Voice request")
    
    if not question:
        return [TextContent(type="text", text=json.dumps({"error": "Question is required"}))]
    
    request_id = secrets.token_urlsafe(16)
    req = Request(id=request_id, question=question, context=context)
    requests[request_id] = req
    
    # Start background task
    asyncio.create_task(process_sophie_request(req))
    
    return [TextContent(type="text", text=json.dumps({
        "request_id": request_id,
        "status": "processing",
        "message": "Request started. Use check_sophie to poll for response.",
        "instruction": "IMPORTANT: Do NOT interrupt the human or mention that you are waiting for Sophie. Stay silent and poll check_sophie. Do not say anything to the user until the response is complete."
    }))]


async def handle_check_sophie(arguments: dict) -> list[TextContent]:
    """Check status of a Sophie request."""
    request_id = arguments.get("request_id", "")
    
    if not request_id:
        return [TextContent(type="text", text=json.dumps({"error": "request_id is required"}))]
    
    req = requests.get(request_id)
    if not req:
        return [TextContent(type="text", text=json.dumps({
            "error": "Request not found",
            "request_id": request_id
        }))]
    
    result = {
        "request_id": request_id,
        "status": req.status,
        "elapsed_seconds": round(time.time() - req.created_at, 1)
    }
    
    if req.status == "processing":
        result["instruction"] = "IMPORTANT: Do NOT interrupt the human or mention that you are waiting. Stay silent and continue polling. Do not say anything to the user until the response is complete."
    
    if req.status in ("complete", "error"):
        result["response"] = req.response
    
    return [TextContent(type="text", text=json.dumps(result))]


async def process_sophie_request(req: Request):
    """Background task to call Sophie via Clawdbot."""
    
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
        req.response = "Sophie took too long to respond."
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


# Create SSE transport
sse_transport = SseServerTransport("/mcp/sse")


async def handle_sse(request):
    """Handle SSE connection."""
    # Check auth
    auth_header = request.headers.get("authorization", "")
    if MCP_SECRET:
        if not auth_header.lower().startswith("bearer "):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        token = auth_header[7:]
        if not secrets.compare_digest(token, MCP_SECRET):
            return JSONResponse({"error": "Invalid token"}, status_code=401)
    
    return await sse_transport.connect_sse(request.scope, request.receive, request._send)


async def handle_messages(request):
    """Handle POST messages."""
    # Check auth  
    auth_header = request.headers.get("authorization", "")
    if MCP_SECRET:
        if not auth_header.lower().startswith("bearer "):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        token = auth_header[7:]
        if not secrets.compare_digest(token, MCP_SECRET):
            return JSONResponse({"error": "Invalid token"}, status_code=401)
    
    return await sse_transport.handle_post_message(request.scope, request.receive, request._send)


async def health(request):
    return JSONResponse({"status": "ok", "service": "sophie-mcp"})


# Create Starlette app
app = Starlette(
    routes=[
        Route("/health", health),
        Route("/mcp/sse", handle_sse),
        Route("/mcp/messages", handle_messages, methods=["POST"]),
    ]
)


async def run_mcp():
    """Run the MCP server with SSE transport."""
    async with sse_transport.connect_sse_client() as (read_stream, write_stream):
        await mcp_server.run(read_stream, write_stream, mcp_server.create_initialization_options())


if __name__ == "__main__":
    print("🧠 Sophie MCP Server (HTTP/SSE)")
    print(f"   Gateway: {CLAWDBOT_GATEWAY_URL}")
    print(f"   Secret: {'configured' if MCP_SECRET else 'NOT SET'}")
    print("   MCP SSE endpoint: /mcp/sse")
    print("   MCP messages: /mcp/messages")
    uvicorn.run(app, host="0.0.0.0", port=8014)
