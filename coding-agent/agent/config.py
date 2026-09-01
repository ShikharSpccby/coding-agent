"""
Configuration for the coding agent.

The agent talks to any OpenAI-compatible chat-completions endpoint that
supports tool/function calling. This means it works unmodified against:

  - Alibaba DashScope (Qwen3-Coder, Qwen2.5-Coder, ...)
  - OpenRouter (proxies Qwen, DeepSeek, GLM, Kimi, etc.)
  - Together / Fireworks (same models, different host)
  - A local Ollama / vLLM server exposing an OpenAI-compatible API

Swapping providers is a base_url + api_key + model_name change only -
nothing in agent.py or tools.py needs to know which provider is behind
the endpoint.
"""

import os
from dataclasses import dataclass


@dataclass
class AgentConfig:
    # OpenAI-compatible endpoint. Examples:
    #   DashScope (Qwen):   https://dashscope.aliyuncs.com/compatible-mode/v1
    #   OpenRouter:         https://openrouter.ai/api/v1
    #   Local Ollama:       http://localhost:11434/v1
    base_url: str = os.environ.get("AGENT_BASE_URL", "https://openrouter.ai/api/v1")

    # API key for the endpoint above. Never hardcode this - always read
    # from the environment so the key never ends up in source control.
    api_key: str = os.environ.get("AGENT_API_KEY", "")

    # Model name as understood by the endpoint above, e.g.:
    #   "qwen/qwen3-coder"          (OpenRouter)
    #   "qwen3-coder-plus"          (DashScope)
    #   "qwen2.5-coder:32b"         (local Ollama)
    model: str = os.environ.get("AGENT_MODEL", "qwen/qwen3-coder")

    # Hard cap on agent loop iterations. Prevents an agent that never
    # converges from running (and burning tokens/money) forever.
    max_steps: int = int(os.environ.get("AGENT_MAX_STEPS", "25"))

    # Timeout (seconds) for any single shell command the agent runs.
    bash_timeout: int = int(os.environ.get("AGENT_BASH_TIMEOUT", "30"))

    # Tool outputs longer than this (characters) are truncated before
    # being sent back to the model, so one huge log dump can't blow the
    # context window.
    max_tool_output_chars: int = int(os.environ.get("AGENT_MAX_TOOL_OUTPUT", "8000"))

    # Root directory the agent is confined to. All file/bash tools
    # resolve paths relative to this and refuse to escape it.
    workspace: str = os.environ.get("AGENT_WORKSPACE", "./workspace")
