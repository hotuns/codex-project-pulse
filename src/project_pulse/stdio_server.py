"""Start the MCP server by file path, independent of the working directory.

Claude Code plugins, magpie's library and most MCP hosts launch a server as
`command + args` without a `cwd` field. Run:

    python3 <plugin root>/src/project_pulse/stdio_server.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Running a file puts its own folder first on sys.path; import the package
# from its parent instead so module names inside it cannot shadow anything.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != HERE]
sys.path.insert(0, str(HERE.parent))

from project_pulse.mcp_server import main  # noqa: E402

if __name__ == "__main__":
    main()
