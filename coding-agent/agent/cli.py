"""
CLI entry point.

Usage:
    python -m agent.cli "add a /health endpoint to app.py that returns 200 OK"

Reads AGENT_BASE_URL / AGENT_API_KEY / AGENT_MODEL / AGENT_WORKSPACE from
the environment (see .env.example). Loads a .env file if python-dotenv
is installed and one is present, so you don't have to export vars by hand.
"""

import sys

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from agent.agent import CodingAgent
from agent.config import AgentConfig


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python -m agent.cli "<task description>"')
        sys.exit(1)

    task = " ".join(sys.argv[1:])
    config = AgentConfig()

    if not config.api_key:
        print(
            "AGENT_API_KEY is not set. Copy .env.example to .env and fill it "
            "in, or export the variable directly.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Task: {task}")
    print(f"Model: {config.model}  |  Workspace: {config.workspace}\n")

    agent = CodingAgent(config)
    result = agent.run(task)

    print("\n=== Final answer ===")
    print(result)


if __name__ == "__main__":
    main()
