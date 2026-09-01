"""
The agent loop itself.

This is intentionally a plain ReAct loop, not a framework:

    loop:
        ask the model, given the conversation so far and the tool list
        if it asked for tool calls -> execute each, append results
        else -> it produced a final answer -> stop

Why not use a framework (LangChain/LlamaIndex agents)? For a project
this size, the loop is ~40 lines and every line is doing something you
need to be able to explain. A framework would hide exactly the parts
(retry logic, truncation, stop conditions) that are the actual point
of the exercise.
"""

import json
from dataclasses import dataclass, field

from openai import OpenAI

from agent.config import AgentConfig
from agent.sandbox import Sandbox, SandboxError
from agent.tools import TOOL_SCHEMAS, dispatch

SYSTEM_PROMPT = """You are a careful coding agent working inside a sandboxed \
workspace directory. You can explore, read, edit, and run code via the tools \
provided.

Rules:
- Always read a file (or grep for the relevant section) before editing it. \
Never guess file contents.
- Prefer edit_file (exact search-and-replace) over write_file for existing \
files. Only use write_file for brand-new files or a genuine full rewrite.
- After making a change, verify it: run the relevant tests or the program \
itself via run_bash before declaring the task complete.
- If a command fails, read the error output carefully and fix the actual \
problem rather than retrying the same thing.
- Work in small, verifiable steps rather than making many changes at once.
- When the task is complete and verified, stop calling tools and reply with \
a plain-text summary of what you did and how you verified it.
"""


@dataclass
class StepLog:
    step: int
    tool_calls: list = field(default_factory=list)  # [(name, args, result)]
    final_answer: str | None = None


class CodingAgent:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.client = OpenAI(base_url=config.base_url, api_key=config.api_key)
        self.sandbox = Sandbox(config.workspace, bash_timeout=config.bash_timeout)
        self.transcript: list[StepLog] = []

    def _truncate(self, text: str) -> str:
        limit = self.config.max_tool_output_chars
        if len(text) <= limit:
            return text
        cut = len(text) - limit
        return text[:limit] + f"\n...[truncated {cut} chars]..."

    def run(self, task: str, verbose: bool = True) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]

        for step in range(1, self.config.max_steps + 1):
            log = StepLog(step=step)

            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                tools=TOOL_SCHEMAS,
            )
            choice = response.choices[0].message

            if not choice.tool_calls:
                # Model produced a final answer - we're done.
                log.final_answer = choice.content
                self.transcript.append(log)
                if verbose:
                    print(f"\n[step {step}] Agent finished:\n{choice.content}")
                return choice.content or ""

            # The model asked for one or more tool calls. Append its
            # request to the conversation, then execute each and append
            # the observation, before looping back to the model.
            messages.append(choice.model_dump(exclude_none=True))

            for call in choice.tool_calls:
                name = call.function.name
                args: dict = {}
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError as e:
                    result = f"ERROR: malformed tool arguments ({e}). Re-issue the call as valid JSON."
                else:
                    try:
                        result = dispatch(self.sandbox, name, args)
                    except SandboxError as e:
                        result = f"ERROR: {e}"
                    except (FileNotFoundError, IsADirectoryError, ValueError) as e:
                        result = f"ERROR: {e}"
                    except Exception as e:  # noqa: BLE001 - surfaced to the model, not swallowed silently
                        result = f"ERROR: unexpected failure running {name}: {e}"

                truncated = self._truncate(str(result))
                log.tool_calls.append((name, args, truncated))

                if verbose:
                    arg_preview = json.dumps(args)[:120]
                    print(f"[step {step}] {name}({arg_preview}) -> {truncated[:200]}")

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": truncated,
                    }
                )

            self.transcript.append(log)

        final_message = (
            f"[stopped: reached max_steps={self.config.max_steps} without the "
            "model returning a final answer]"
        )
        if verbose:
            print(f"\n{final_message}")
        return final_message
