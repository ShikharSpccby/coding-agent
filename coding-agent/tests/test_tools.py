"""
Unit tests for the sandbox and tool dispatch layer. These deliberately
don't touch the LLM at all - they test the part of the system that's
fully deterministic and doesn't need an API key, which is also the
part most worth covering: it's what stands between the agent and your
filesystem.

Run with:  pytest tests/
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.sandbox import Sandbox, SandboxError
from agent.tools import dispatch


@pytest.fixture
def sandbox(tmp_path):
    return Sandbox(workspace=str(tmp_path), bash_timeout=5)


def test_write_and_read_file(sandbox):
    sandbox.write_file("hello.txt", "hello world")
    assert sandbox.read_file("hello.txt") == "hello world"


def test_read_missing_file_raises(sandbox):
    with pytest.raises(FileNotFoundError):
        sandbox.read_file("nope.txt")


def test_edit_file_requires_unique_match(sandbox):
    sandbox.write_file("a.py", "x = 1\nx = 1\n")
    with pytest.raises(ValueError):
        sandbox.edit_file("a.py", "x = 1", "x = 2")


def test_edit_file_applies_unique_replacement(sandbox):
    sandbox.write_file("a.py", "def foo():\n    return 1\n")
    sandbox.edit_file("a.py", "return 1", "return 2")
    assert "return 2" in sandbox.read_file("a.py")


def test_edit_file_rejects_missing_old_str(sandbox):
    sandbox.write_file("a.py", "def foo():\n    return 1\n")
    with pytest.raises(ValueError):
        sandbox.edit_file("a.py", "return 999", "return 2")


def test_path_traversal_is_blocked(sandbox):
    with pytest.raises(SandboxError):
        sandbox.read_file("../../etc/passwd")


def test_list_dir_reflects_writes(sandbox):
    sandbox.write_file("pkg/mod.py", "x = 1")
    listing = sandbox.list_dir(".")
    assert "pkg" in listing
    assert "pkg/mod.py" in listing or "pkg\\mod.py" in listing  # cross-platform


def test_grep_finds_pattern(sandbox):
    sandbox.write_file("a.py", "def target_function():\n    pass\n")
    sandbox.write_file("b.py", "x = 1\n")
    result = sandbox.grep("target_function")
    assert "a.py" in result
    assert "b.py" not in result


def test_run_bash_captures_exit_code_and_output(sandbox):
    result = sandbox.run_bash("echo hi && exit 0")
    assert "exit_code=0" in result
    assert "hi" in result


def test_run_bash_reports_nonzero_exit(sandbox):
    result = sandbox.run_bash("exit 3")
    assert "exit_code=3" in result


def test_run_bash_timeout_is_enforced(sandbox):
    # Use Python's sleep rather than the Unix `sleep` command: this test
    # needs to pass on Windows too, and Python is guaranteed to be on
    # PATH here (it's what's running the test), whereas a shell `sleep`
    # builtin/binary is not.
    result = sandbox.run_bash(f'{sys.executable} -c "import time; time.sleep(30)"')
    assert "TIMEOUT" in result

def test_dispatch_routes_to_correct_tool(sandbox):
    dispatch(sandbox, "write_file", {"path": "x.txt", "content": "abc"})
    out = dispatch(sandbox, "read_file", {"path": "x.txt"})
    assert out == "abc"


def test_dispatch_unknown_tool_raises(sandbox):
    with pytest.raises(ValueError):
        dispatch(sandbox, "not_a_real_tool", {})
