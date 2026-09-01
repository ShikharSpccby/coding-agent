"""
Tool definitions: the JSON schemas sent to the model, plus the dispatch
table that maps a tool call to a Sandbox method.

Keeping this list short and orthogonal is deliberate. Every extra tool
is another thing the model can misuse or hallucinate arguments for.
Six tools covering explore / read / edit / verify is enough for the
large majority of real coding tasks.
"""

from agent.sandbox import Sandbox

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and directories under a path in the workspace, recursively.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path relative to the workspace root. Defaults to '.' (workspace root).",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the full contents of a file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the workspace root."}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search for a regex pattern across files in the workspace. Use this to find where something is defined or used before editing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for."},
                    "path": {"type": "string", "description": "Path to search under. Defaults to the whole workspace."},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create a new file or fully overwrite an existing one. Prefer edit_file for modifying existing files - only use write_file for brand-new files or genuine full rewrites.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the workspace root."},
                    "content": {"type": "string", "description": "Full file content."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Replace an exact snippet of an existing file with new text. "
                "old_str must match the current file content exactly (including "
                "whitespace) and must be unique in the file - read the file first "
                "and copy the precise text you want to change. This is the "
                "preferred way to modify existing files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the workspace root."},
                    "old_str": {"type": "string", "description": "Exact, unique existing text to replace."},
                    "new_str": {"type": "string", "description": "Text to replace it with."},
                },
                "required": ["path", "old_str", "new_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": (
                "Run a shell command inside the workspace (e.g. run tests, "
                "install a dependency, execute a script). Use this to verify "
                "your changes actually work before declaring the task done."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run."}
                },
                "required": ["command"],
            },
        },
    },
]


def dispatch(sandbox: Sandbox, name: str, arguments: dict) -> str:
    """Execute a tool call against the sandbox and return its string
    result. Exceptions are caught by the caller (agent.py) so a failed
    tool call becomes an observation the model can react to, rather
    than crashing the whole agent."""
    if name == "list_dir":
        return sandbox.list_dir(arguments.get("path", "."))
    if name == "read_file":
        return sandbox.read_file(arguments["path"])
    if name == "grep":
        return sandbox.grep(arguments["pattern"], arguments.get("path", "."))
    if name == "write_file":
        return sandbox.write_file(arguments["path"], arguments["content"])
    if name == "edit_file":
        return sandbox.edit_file(arguments["path"], arguments["old_str"], arguments["new_str"])
    if name == "run_bash":
        return sandbox.run_bash(arguments["command"])
    raise ValueError(f"Unknown tool: {name}")
