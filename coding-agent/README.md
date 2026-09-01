# Coding Agent

A minimal, from-scratch coding agent: given a task in natural language, it
explores a codebase, reads and edits files, runs shell commands (tests,
scripts, linters), and iterates on the results until the task is done or it
gives up. Built to work with **any OpenAI-compatible tool-calling endpoint**
— tested against Qwen-family models, but the provider is a config change,
not a code change.

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env        # fill in AGENT_BASE_URL / AGENT_API_KEY / AGENT_MODEL
python -m agent.cli "add a /health endpoint to app.py that returns 200 OK"
```

By default the agent operates inside `./workspace` (configurable via
`AGENT_WORKSPACE`) — put the repo you want it to work on there, or point the
env var at an existing directory.

Run the test suite (no API key needed — it only exercises the sandbox/tool
layer, not the LLM):

```bash
pytest tests/ -v
```

## What it does

You give it one instruction, e.g.:

> "Add a /health endpoint to app.py that returns 200 OK, add a test for it
> in test_app.py, and run the tests to confirm they pass."

It then, on its own:
1. Lists the workspace to see what exists
2. Reads the relevant file(s)
3. Makes a precise, minimal edit
4. Writes/updates a test
5. Runs the test suite to verify the change actually works
6. Reports back what it did and how it verified it

See [`examples/transcript_add_health_endpoint.txt`](examples/transcript_add_health_endpoint.txt)
for a real, unedited transcript of exactly this run — including the actual
tool calls, their arguments, and their results.

## Architecture

```
User task
   |
   v
+-------------------+      tool call       +------------------+
|   Agent loop       | -------------------> |   Sandbox        |
|  (agent/agent.py)   | <------------------- | (agent/sandbox.py)|
+-------------------+     observation       +------------------+
   |
   v
OpenAI-compatible /chat/completions endpoint
(Qwen3-Coder, DeepSeek, GLM, local Ollama, ...)
```

**The loop (`agent/agent.py`)** is a plain ReAct pattern:

```
messages = [system_prompt, task]
loop until done or max_steps:
    response = llm(messages, tools=tool_schemas)
    if response has tool_calls:
        execute each, append results to messages
    else:
        return response.text   # final answer, stop
```

No agent framework (LangChain, LlamaIndex, etc.) is used. At this scale the
loop is ~50 lines, and every line — retry behavior, truncation, the stop
condition — is something I need to be able to explain, not something I want
hidden behind an abstraction.

**Tools (`agent/tools.py`)** — six, deliberately kept small:

| Tool | Purpose |
|---|---|
| `list_dir` | Explore the workspace |
| `read_file` | Read before editing (enforced by convention in the system prompt) |
| `grep` | Find where something is defined/used across the repo |
| `edit_file` | Exact search-and-replace — the preferred way to modify existing files |
| `write_file` | New files or genuine full rewrites only |
| `run_bash` | Run tests/scripts/linters — how the agent verifies its own work |

**Sandbox (`agent/sandbox.py`)** — every tool call goes through this. It:
- Confines all file access to a workspace root (blocks `../` traversal and
  absolute-path escapes)
- Enforces a timeout on every shell command
- Is the only place that touches the filesystem or spawns a process, so
  it's the one file worth auditing carefully

**Config (`agent/config.py`)** — all provider/behavior settings come from
environment variables: `base_url`, `api_key`, `model`, `max_steps`,
`bash_timeout`, `max_tool_output_chars`, `workspace`. Swapping from Qwen via
OpenRouter to Qwen via DashScope to a local Ollama server is a `.env` change,
not a code change.

## Key design decisions

**`edit_file` over blind `write_file`.** `edit_file` requires an exact,
unique match of the text being replaced. This forces the model to have
actually read the current file content rather than hallucinating what it
probably looks like — a full-file rewrite has no such check and silently
accepts a confidently wrong guess.

**Hard `max_steps` cap.** An agent that doesn't converge should fail loudly
and cheaply, not run (and burn API spend) indefinitely.

**Truncating tool output before it goes back to the model.** A single
`run_bash` call against a big test suite can produce megabytes of log
output. Truncating to `AGENT_MAX_TOOL_OUTPUT` chars keeps the context window
from being consumed by one noisy command.

**Failed tool calls become observations, not crashes.** A missing file, a
bad `old_str`, a malformed JSON argument — these are caught and turned into
an `ERROR: ...` string sent back to the model, which can then correct itself
next turn. The agent loop itself never crashes because a tool call failed;
that's expected, recoverable behavior, not an exception to propagate.

**Model choice: Qwen3-Coder, via an OpenAI-compatible API rather than
self-hosted.** Self-hosting a 27B–80B model well (quantization, context
tuning, throughput) is its own multi-day project. Using it through DashScope
or OpenRouter kept engineering time on the agent loop and tools — which is
the actual point of the exercise — while remaining fully swappable to a
local endpoint later with zero code changes.

## Limitations (honest accounting)

- **Sandboxing is OS-level path confinement + subprocess timeouts, not a
  container.** Good enough for a trusted single-user CLI tool working on a
  repo you already trust; not sufficient if this were driving an untrusted
  repo or untrusted user input — `run_bash` can still install packages or
  open network connections within the workspace. Real hardening would put
  the sandbox in Docker/gVisor/Firecracker with no network egress by
  default.
- **No retry/backoff on malformed tool-call JSON** beyond surfacing the
  error to the model and letting it try again next turn. A production
  version would validate against the JSON schema and retry with a repair
  prompt before giving up.
- **No repo-scale retrieval.** `grep` + `list_dir` is enough for small-to-
  medium repos but doesn't scale to a large monorepo, where you'd want an
  indexed/embedding-based search tool instead.
- **Single-threaded, single-task.** No parallel tool execution, no
  multi-agent decomposition of a large task into subtasks.
- **No persistent memory across runs.** Each invocation starts a fresh
  conversation; there's no session/history store.
- **Context management is truncation-only.** Long-running tasks that
  exceed the model's context window aren't summarized — they'd just
  eventually fail. Qwen3-Coder's large native context window (256K on the
  -Next variant) makes this less urgent in practice than it would be on a
  smaller-context model.

## What I'd do with more time

- Swap OS-level confinement for real container isolation (Docker) with a
  locked-down network policy
- Schema-validate tool-call arguments and add a repair/retry loop for
  malformed calls instead of a single pass-through to the model
- Add a lightweight eval harness — a handful of fixed tasks (à la a mini
  SWE-bench) run automatically to catch regressions in the agent's own
  behavior when the prompt or tool set changes
- Context summarization for long-running tasks, rather than pure truncation
- A `--dry-run` mode that shows proposed edits as diffs before applying them

## Repo layout

```
agent/
  agent.py      the loop
  tools.py      tool schemas + dispatch
  sandbox.py    filesystem/shell confinement
  config.py     env-driven settings
  cli.py        entry point
tests/
  test_tools.py unit tests for sandbox/tools (no API key required)
examples/
  transcript_add_health_endpoint.txt   real end-to-end run
scripts/
  mock_llm_server.py   scripted OpenAI-compatible stub used only to
                        generate the example transcript reproducibly
```
Windows compatibility issues:
- grep is unavailable on standard Windows installations
- sleep is unavailable in cmd.exe
- some tests assume Unix utilities