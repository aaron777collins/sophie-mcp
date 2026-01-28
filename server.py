#!/usr/bin/env python3
"""
Sophie MCP Server - Async access to Sophie's capabilities.

Two tools:
- ask_sophie(question) → returns request_id immediately
- check_sophie(request_id) → returns status (processing/complete) + response

This allows voice models to poll without blocking.
"""

import asyncio
import json
import os
import secrets
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from datetime import datetime

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from dotenv import load_dotenv

load_dotenv()

# Config
CLAWDBOT_GATEWAY_URL = os.getenv("CLAWDBOT_GATEWAY_URL", "http://localhost:18789")
CLAWDBOT_GATEWAY_TOKEN = os.getenv("CLAWDBOT_GATEWAY_TOKEN", "")
MCP_SECRET = os.getenv("SOPHIE_MCP_SECRET", "")

if not MCP_SECRET:
    print("⚠️  Warning: SOPHIE_MCP_SECRET not set - MCP is UNPROTECTED!", file=sys.stderr)


@dataclass
class Request:
    id: str
    question: str
    context: str
    status: str = "processing"  # processing, complete, error
    response: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


# In-memory request store (could be Redis for persistence)
requests: Dict[str, Request] = {}

server = Server("sophie-mcp")


def verify_secret(secret: str) -> bool:
    """Verify the provided secret matches."""
    if not MCP_SECRET:
        return True  # No secret configured = open (dev mode)
    return secrets.compare_digest(secret, MCP_SECRET)


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="ask_sophie",
            description="""Start an async request to Sophie (the main AI assistant).
            
Returns a request_id immediately. Use check_sophie to poll for the response.

Sophie has access to: calendar, email, files, web search, memory, and all tools.

Use for complex questions that may take time to process.""",
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
                    },
                    "secret": {
                        "type": "string",
                        "description": "Authentication secret"
                    }
                },
                "required": ["question", "secret"]
            }
        ),
        Tool(
            name="check_sophie",
            description="""Check the status of a Sophie request.

Returns:
- status: "processing" | "complete" | "error"
- response: Sophie's answer (if complete)

Poll this until status is "complete".""",
            inputSchema={
                "type": "object",
                "properties": {
                    "request_id": {
                        "type": "string",
                        "description": "The request ID from ask_sophie"
                    },
                    "secret": {
                        "type": "string",
                        "description": "Authentication secret"
                    }
                },
                "required": ["request_id", "secret"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    
    # Verify secret for all tools
    secret = arguments.get("secret", "")
    if not verify_secret(secret):
        return [TextContent(type="text", text=json.dumps({
            "error": "Invalid or missing secret"
        }))]
    
    if name == "ask_sophie":
        return await handle_ask_sophie(arguments)
    elif name == "check_sophie":
        return await handle_check_sophie(arguments)
    
    return [TextContent(type="text", text=json.dumps({
        "error": f"Unknown tool: {name}"
    }))]


async def handle_ask_sophie(arguments: dict) -> list[TextContent]:
    """Start an async request to Sophie."""
    question = arguments.get("question", "")
    context = arguments.get("context", "Voice request")
    
    if not question:
        return [TextContent(type="text", text=json.dumps({
            "error": "Question is required"
        }))]
    
    # Generate request ID
    request_id = secrets.token_urlsafe(16)
    
    # Create request record
    req = Request(
        id=request_id,
        question=question,
        context=context
    )
    requests[request_id] = req
    
    # Start background task to call Sophie
    asyncio.create_task(process_sophie_request(req))
    
    return [TextContent(type="text", text=json.dumps({
        "request_id": request_id,
        "status": "processing",
        "message": "Request started. Use check_sophie to poll for response."
    }))]


async def handle_check_sophie(arguments: dict) -> list[TextContent]:
    """Check status of a Sophie request."""
    request_id = arguments.get("request_id", "")
    
    if not request_id:
        return [TextContent(type="text", text=json.dumps({
            "error": "request_id is required"
        }))]
    
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
    
    if req.status == "complete":
        result["response"] = req.response
    elif req.status == "error":
        result["response"] = req.response  # Contains error message
    
    return [TextContent(type="text", text=json.dumps(result))]


async def process_sophie_request(req: Request):
    """Background task to process a Sophie request via Clawdbot."""
    
    headers = {
        "Authorization": f"Bearer {CLAWDBOT_GATEWAY_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # Build prompt for Sophie
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
    
    # Clean up old requests (keep last 100)
    if len(requests) > 100:
        sorted_reqs = sorted(requests.items(), key=lambda x: x[1].created_at)
        for key, _ in sorted_reqs[:-100]:
            del requests[key]


async def main():
    """Run the MCP server."""
    print(f"🧠 Sophie MCP Server starting...", file=sys.stderr)
    print(f"   Gateway: {CLAWDBOT_GATEWAY_URL}", file=sys.stderr)
    print(f"   Secret: {'configured' if MCP_SECRET else 'NOT SET (open)'}", file=sys.stderr)
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
