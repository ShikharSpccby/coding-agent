"""
A tiny scripted stand-in for an OpenAI-compatible chat-completions
endpoint, used only to generate a genuine, reproducible example
transcript for the README without needing outbound network access to
a real Qwen provider.

It is NOT part of the agent - it's a throwaway harness that scripts a
plausible sequence of tool calls for one fixed task, then returns a
final answer, exactly in the response shape the real openai client
expects. Swap AGENT_BASE_URL to this server's address to reproduce the
transcript locally.
"""

import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

# Scripted turns: each is either a tool_calls response or a final text answer.
SCRIPT = [
    {
        "tool_calls": [
            {"name": "list_dir", "arguments": {"path": "."}},
        ]
    },
    {
        "tool_calls": [
            {"name": "read_file", "arguments": {"path": "app.py"}},
        ]
    },
    {
        "tool_calls": [
            {
                "name": "edit_file",
                "arguments": {
                    "path": "app.py",
                    "old_str": "@app.route('/')\ndef index():\n    return 'ok'\n",
                    "new_str": (
                        "@app.route('/')\ndef index():\n    return 'ok'\n\n\n"
                        "@app.route('/health')\ndef health():\n    return 'OK', 200\n"
                    ),
                },
            }
        ]
    },
    {
        "tool_calls": [
            {"name": "read_file", "arguments": {"path": "test_app.py"}},
        ]
    },
    {
        "tool_calls": [
            {
                "name": "edit_file",
                "arguments": {
                    "path": "test_app.py",
                    "old_str": "def test_index(client):\n    assert client.get('/').status_code == 200\n",
                    "new_str": (
                        "def test_index(client):\n    assert client.get('/').status_code == 200\n\n\n"
                        "def test_health(client):\n    resp = client.get('/health')\n"
                        "    assert resp.status_code == 200\n    assert resp.data == b'OK'\n"
                    ),
                },
            }
        ]
    },
    {
        "tool_calls": [
            {"name": "run_bash", "arguments": {"command": "python -m pytest test_app.py -v"}},
        ]
    },
    {
        "final": (
            "Added a GET /health endpoint to app.py returning 'OK' with a 200 "
            "status code, matching the style of the existing index route. Added "
            "test_health in test_app.py covering both the status code and body. "
            "Verified with `pytest test_app.py -v` - both tests pass."
        )
    },
]


class Handler(BaseHTTPRequestHandler):
    turn = 0

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)  # request body isn't needed by this stub

        step = SCRIPT[min(Handler.turn, len(SCRIPT) - 1)]
        Handler.turn += 1

        if "final" in step:
            message = {"role": "assistant", "content": step["final"], "tool_calls": None}
        else:
            tool_calls = [
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])},
                }
                for tc in step["tool_calls"]
            ]
            message = {"role": "assistant", "content": None, "tool_calls": tool_calls}

        body = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "mock-qwen3-coder",
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        }
        payload = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass  # keep stdout clean for the real agent's own logging


if __name__ == "__main__":
    server = HTTPServer(("localhost", 8790), Handler)
    print("Mock LLM server on http://localhost:8790/v1")
    server.serve_forever()
