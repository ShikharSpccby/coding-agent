"""
Sandbox: everything the agent touches goes through here.

This is deliberately the most conservative part of the codebase. Two
things are non-negotiable for a coding agent that can write files and
run shell commands:

  1. It must not be able to read/write/execute anything outside its
     workspace directory (path confinement).
  2. Every command must have a hard timeout (an agent that hangs
     forever on `run_bash` is a stuck agent, not a paused one).

Note on production hardening: this uses OS-level path confinement and
subprocess timeouts, which is enough for a trusted single-user CLI
tool. It is NOT a substitute for a real container (Docker/gVisor/
Firecracker) if you ever let an untrusted user or untrusted repo drive
this agent - `run_bash` can still install packages, spawn network
connections, etc. within the workspace. See README "Limitations".
"""

import os
import subprocess
from pathlib import Path


class SandboxError(Exception):
    """Raised when a tool call would escape the sandbox or otherwise
    violates a safety constraint."""


class Sandbox:
    def __init__(self, workspace: str, bash_timeout: int = 30):
        self.root = Path(workspace).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.bash_timeout = bash_timeout

    def _resolve(self, relative_path: str) -> Path:
        """Resolve a path the agent gave us, refusing anything that
        would escape the workspace root (../.. tricks, absolute paths
        outside root, symlink escapes, etc.)."""
        candidate = (self.root / relative_path).resolve()
        if self.root not in candidate.parents and candidate != self.root:
            raise SandboxError(
                f"Path '{relative_path}' resolves outside the workspace "
                f"and was blocked."
            )
        return candidate

    # ---- file operations -------------------------------------------------

    def read_file(self, path: str) -> str:
        target = self._resolve(path)
        if not target.exists():
            raise FileNotFoundError(f"No such file: {path}")
        if not target.is_file():
            raise IsADirectoryError(f"Not a file: {path}")
        return target.read_text(errors="replace")

    def write_file(self, path: str, content: str) -> str:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return f"Wrote {len(content)} chars to {path}"

    def edit_file(self, path: str, old: str, new: str) -> str:
        """Search-and-replace edit. Preferred over write_file for
        modifying existing files: it forces the agent to prove it has
        actually read the current content (old must match exactly and
        uniquely), which catches a whole class of hallucinated edits
        that a blind full-file rewrite would not."""
        target = self._resolve(path)
        if not target.exists():
            raise FileNotFoundError(f"No such file: {path}")
        content = target.read_text()
        count = content.count(old)
        if count == 0:
            raise ValueError(
                "old_str not found in file - read the file first and "
                "copy the exact text to replace."
            )
        if count > 1:
            raise ValueError(
                f"old_str is not unique ({count} occurrences) - include "
                "more surrounding context so the match is unambiguous."
            )
        target.write_text(content.replace(old, new, 1))
        return f"Edited {path} (1 replacement)"

    def list_dir(self, path: str = ".") -> str:
        target = self._resolve(path)
        if not target.exists():
            raise FileNotFoundError(f"No such directory: {path}")
        entries = []
        for p in sorted(target.rglob("*")):
            if any(part.startswith(".git") for part in p.parts):
                continue
            rel = p.relative_to(self.root)
            entries.append(f"{'d' if p.is_dir() else 'f'} {rel}")
        return "\n".join(entries) if entries else "(empty)"

    # ---- search -------------------------------------------------------

    def grep(self, pattern: str, path: str = ".") -> str:
        target = self._resolve(path)
        try:
            result = subprocess.run(
                ["grep", "-rn", "--exclude-dir=.git", pattern, str(target)],
                capture_output=True,
                text=True,
                timeout=self.bash_timeout,
            )
        except FileNotFoundError:
            raise RuntimeError("ripgrep/grep not available in this environment")
        if result.returncode not in (0, 1):  # 1 = no matches, not an error
            raise RuntimeError(result.stderr.strip())
        # Report paths relative to the workspace root, not absolute.
        lines = result.stdout.splitlines()
        rel_lines = [line.replace(str(self.root) + os.sep, "") for line in lines]
        return "\n".join(rel_lines) if rel_lines else "(no matches)"

    # ---- shell ----------------------------------------------------------

    def run_bash(self, command: str) -> str:
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=self.bash_timeout,
            )
        except subprocess.TimeoutExpired:
            return f"[TIMEOUT after {self.bash_timeout}s - command killed]"
        output = result.stdout + result.stderr
        return f"exit_code={result.returncode}\n{output}"
