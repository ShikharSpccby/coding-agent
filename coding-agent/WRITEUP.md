# Coding Agent — Project Writeup

## What this is

A CLI-based coding agent that takes a natural-language task, autonomously explores a
codebase, makes edits, and verifies its own work by running commands — built from
scratch (no LangChain/LlamaIndex-style framework) so every part of the loop is
something I can explain and defend, not something a library did for me.

Repo: `agent/` (the implementation), `tests/` (13 unit tests, all passing),
`examples/` (a real, unedited transcript from an end-to-end run against a live
external repository), `README.md` (architecture and design decisions).

---

## Architecture

The agent is a plain ReAct loop:

```
loop:
    send conversation + tool schemas to the model
    if the model requests a tool call:
        execute it locally, inside a sandboxed workspace
        append the result to the conversation
        repeat
    else:
        the model produced a final answer -> stop
    if step count exceeds a cap -> stop and report failure
```

**Core components:**

- `agent/agent.py` — the loop itself, plus retry/backoff logic for transient API errors
- `agent/sandbox.py` — everything the agent touches goes through here: path confinement
  (no escaping the workspace directory), timeouts on every shell command, and the actual
  file/search/shell operations
- `agent/tools.py` — tool schema definitions (`read_file`, `write_file`, `edit_file`,
  `list_dir`, `grep`, `run_bash`) and dispatch
- `agent/config.py` — all runtime behavior (model, endpoint, step limits, timeouts,
  token caps) driven by environment variables, nothing hardcoded
- `agent/cli.py` — entry point

**Key design decision — the model never executes anything.** It can only respond with
text or a *request* to call a tool. My code decides whether to honor that request, and
does so inside a sandbox that confines file access to one directory and enforces a
timeout on every command. This distinction (model proposes, code disposes and guards)
is the actual point of the architecture, more than any individual tool.

**Key design decision — `edit_file` is search-and-replace, not full-file rewrite.**
It requires the `old_str` argument to match the current file content exactly and
uniquely. This forces the model to prove it actually read the current state of the
file before changing it, and catches a whole class of hallucinated edits that a blind
`write_file` would let through silently.

---

## Model choice, and why it changed mid-project

I originally planned around Qwen3-Coder, reasoning that an OpenAI-compatible,
tool-calling-capable open model accessed via API (rather than self-hosted) would let
me spend my time on the agent loop instead of GPU/infra setup. I used OpenRouter as
the provider specifically so the choice of model would be a one-line config change,
not an architectural decision.

In practice, I ended up validating the project against
`nvidia/nemotron-3-ultra-550b-a55b` on OpenRouter's free tier instead. This was not
the original plan, but it turned into a useful proof point rather than a compromise:
switching models required editing exactly one line in `.env` (`AGENT_MODEL`), which is
precisely the claim the architecture makes about provider portability. I didn't just
assert that the design was provider-agnostic — I exercised it.

---

## What I actually tested it against, and why

My first instinct was to demo it against the `demo_workspace/` Flask fixture bundled
with the scaffold. I caught a problem with that early: the fixture already contained
the target end state (a `/health` route that was supposedly the thing being "added"),
which meant a transcript against it wouldn't actually prove the agent did anything. I
fixed the fixture, but decided a canned demo wasn't strong enough evidence on its own
either way.

I looked at two real, external repos as better test targets:

- **A brain-tumor MRI classification project** — rejected. It was 100% Jupyter
  notebooks with no test suite and heavy, unavailable-in-repo ML dependencies. No
  clean way to verify the agent's work, and installing TensorFlow for no payoff wasn't
  a good use of time.
- **IBM's MAX-Image-Resolution-Enhancer** (a Flask-based SRGAN super-resolution
  service, via a public fork) — chosen. It has real Python source (`api/`, `core/`),
  and a declared test tooling stack in `requirements-test.txt` (pytest, flake8,
  bandit). I checked the actual `tests/` folder before committing to it and found the
  pytest suite made live HTTP calls to a running Flask server with the model loaded —
  an integration suite, not something runnable without Docker and TensorFlow. Rather
  than force that, I scoped the task to `flake8`, which the repo's own maintainers
  already used, needs no model weights, and gives a fast, objectively verifiable
  before/after (`flake8 --count` going from 4 violations to 0).

This is the transcript I ended up submitting: a real task, on a real external repo I
evaluated and chose deliberately, verified independently rather than trusting the
agent's own summary.

---

## Real bugs I found through testing, and fixed

Three separate issues surfaced from actually running the agent against real
conditions rather than a scripted demo — I'm listing all three because finding and
fixing them is a better signal than a suite that never failed because it was never
really exercised.

**1. Windows compatibility in the test suite and the `grep` tool.**
`agent/sandbox.py`'s `grep()` originally shelled out to the literal `grep` binary,
which doesn't exist on a standard Windows install. One of the unit tests also used the
Unix `sleep` command to test the bash timeout, which fails the same way. I rewrote
`grep` as a pure-Python recursive search (removing the external-binary dependency
entirely, which is a better design regardless of platform), and changed the timeout
test to invoke `python -c "time.sleep(30)"`, since Python is guaranteed to be on PATH
anywhere this project runs. Verified: 13/13 tests passing on both Linux and the actual
Windows machine I ran this on.

**2. No retry handling on the model API call.**
A live run hit `openai.RateLimitError` (OpenRouter's shared free-tier pool was
saturated) and the exception propagated straight out of the loop, crashing the whole
CLI with a raw traceback instead of failing gracefully or retrying. I added
`_call_model()`, a wrapper with exponential backoff for `RateLimitError`,
`APIConnectionError`, and 5xx `APIStatusError` responses — transient, provider-side
issues that aren't the agent's fault and are worth retrying. Genuine 4xx errors (bad
request, auth failure) are re-raised immediately rather than retried, since retrying
those would just fail the same way five times before giving the same result. This
distinction (retry transient failures, fail fast on real ones) is deliberate, not just
"retry everything."

**3. Unbounded `max_tokens` hitting free-tier credit limits.**
A later run hit `openai.APIStatusError: 402` — "requires more credits" — because the
client wasn't specifying `max_tokens` on requests, so it defaulted to something very
large (observed values up to 131,072 across different calls). OpenRouter's free tier
ties allowed `max_tokens` to a small credit allowance even on nominally free models.
I added a configurable `max_tokens` field to `AgentConfig` (default 4096, overridable
via `AGENT_MAX_OUTPUT_TOKENS`) and passed it explicitly on every request. Beyond fixing
the immediate error, this is a legitimate safeguard independent of billing: an agent
that can silently request 131K output tokens in a single turn is also an agent that
can burn an unbounded budget on one bad step.

Both fixes were validated by rerunning the real task afterward — the transcript in
`examples/` shows the retry logic firing and recovering mid-run (`[retry]
RateLimitError - waiting 2s before retry 1/4...`) rather than crashing, five separate
times over the course of one task, and the run still completing successfully.

---

## What the final verified run actually demonstrates

Beyond fixing the E501 violations correctly (confirmed independently via
`flake8 api/ --count` returning 0, not just trusting the agent's self-reported
summary), the transcript shows two behaviors I didn't script or prompt for:

- When a `run_bash` call to `cat -n api/predict.py` failed because `cat` isn't a
  Windows command, the agent read the error message and independently switched to a
  Python one-liner to inspect the file instead, without being told to.
- After flake8 reported zero violations, the agent ran an additional
  `python -m py_compile` check on its own initiative to confirm it hadn't introduced a
  syntax error — a verification step beyond the one I explicitly asked for.

I also manually reviewed the `git diff` on the edited file before accepting the run as
successful, rather than relying solely on the agent's or the linter's word for it.

---

## Known limitations

- **Sandboxing is process-level, not container-level.** Path confinement and command
  timeouts are enough for a trusted single-user CLI tool, but `run_bash` can still
  install packages or make network calls within the workspace. A real deployment
  handling untrusted repos or users would need Docker/gVisor/Firecracker-level
  isolation, not just OS-level path checks.
- **No repo-scale retrieval.** `grep` and `list_dir` are fine for small-to-medium
  repos; a codebase too large to fit relevant context in one pass would need actual
  embeddings-based search, which I deliberately left out as out of scope for the time
  available.
- **Single-threaded, single-task.** No parallelism, no multi-agent orchestration, no
  persistent memory across separate `agent.cli` invocations.
- **Retry logic only covers the model API call**, not tool execution failures beyond
  what's already surfaced back to the model as an error string for it to react to.

## What I'd do with more time

- Docker-based sandboxing for the `run_bash` tool
- A smarter truncation/summarization strategy for very long tool outputs, rather than
  a flat character cutoff
- An eval harness running the agent against a small fixed set of tasks (in the spirit
  of a mini SWE-bench) to measure success rate and step-count across runs, not just
  spot-checking individual transcripts
- Structured logging of the full transcript to a file automatically, rather than
  redirecting stdout manually