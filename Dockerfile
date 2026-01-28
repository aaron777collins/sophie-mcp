FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn starlette

COPY mcp_streamable_server.py .

CMD ["python", "mcp_streamable_server.py"]
