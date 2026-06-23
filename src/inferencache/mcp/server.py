"""
mcp/server.py — MCP server for inferencache.

Exposes cache internals as MCP tools so AI coding assistants (Cursor,
Windsurf, Claude Code) can inspect and manage the cache without the
developer leaving their editor.

Install: pip install inferencache[mcp]
Run:     inferencache-mcp
Config:  claude_desktop_config.json or .cursor/mcp.json

  {
    "mcpServers": {
      "inferencache": {
        "command": "inferencache-mcp",
        "args": ["--cache-dir", "~/.cache/inferencache"]
      }
    }
  }

Tools exposed:
  get_stats           — Hit rate, cost saved, exact/semantic breakdown
  list_recent         — Most recent N cached entries
  get_cached_entry    — Look up a specific prompt in the cache
  set_threshold       — Update the similarity threshold at runtime
  clear_cache         — Flush all entries (or by model filter)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp import types
except ImportError as exc:
    raise ImportError(
        "The MCP package is required for the inferencache MCP server. "
        "Install it with: pip install inferencache[mcp]"
    ) from exc

from .tools import (
    tool_clear_cache,
    tool_get_cached_entry,
    tool_get_stats,
    tool_list_recent,
    tool_set_threshold,
)

# ---------------------------------------------------------------------------
# Server definition
# ---------------------------------------------------------------------------

app = Server("inferencache")

# Global state shared by tool handlers
_cache_dir: Path = Path.home() / ".cache" / "inferencache"
_threshold: float = 0.85


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="get_stats",
            description=(
                "Return cache statistics: total entries, hit rate (exact vs semantic), "
                "estimated tokens saved, estimated cost saved, and the top cached prompts."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "Model name for cost estimation (e.g. 'gpt-4o'). Defaults to 'unknown'.",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "How many top entries to include (default 10).",
                    },
                    "avg_tokens": {
                        "type": "integer",
                        "description": "Assumed average response length in tokens for cost estimation (default 200).",
                    },
                },
                "required": [],
            },
        ),
        types.Tool(
            name="list_recent",
            description=(
                "List the most recently cached entries. Returns prompt previews, "
                "models, hit counts, and timestamps."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum entries to return (default 20, max 100).",
                    }
                },
                "required": [],
            },
        ),
        types.Tool(
            name="get_cached_entry",
            description=(
                "Check whether a specific prompt string is in the cache. "
                "Returns the cached response, similarity score, and hit count if found."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The prompt string to look up.",
                    },
                    "model": {
                        "type": "string",
                        "description": "The model to look up the prompt for.",
                    },
                },
                "required": ["prompt"],
            },
        ),
        types.Tool(
            name="set_threshold",
            description=(
                "Update the semantic similarity threshold at runtime. "
                "Higher = stricter matching (fewer cache hits). "
                "Recommended range: 0.80–0.92."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "threshold": {
                        "type": "number",
                        "description": "New threshold value between 0.0 and 1.0.",
                    }
                },
                "required": ["threshold"],
            },
        ),
        types.Tool(
            name="clear_cache",
            description=(
                "Delete cached entries. Can target a specific model or clear everything. "
                "This is irreversible."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "Only delete entries for this model. Omit to clear all.",
                    },
                    "confirm": {
                        "type": "boolean",
                        "description": "Must be true to proceed. Prevents accidental clears.",
                    },
                },
                "required": ["confirm"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    global _threshold

    try:
        if name == "get_stats":
            result = tool_get_stats(
                cache_dir=_cache_dir,
                model=arguments.get("model", "unknown"),
                top_n=int(arguments.get("top_n", 10)),
                avg_tokens=int(arguments.get("avg_tokens", 200)),
            )

        elif name == "list_recent":
            limit = min(int(arguments.get("limit", 20)), 100)
            result = tool_list_recent(cache_dir=_cache_dir, limit=limit)

        elif name == "get_cached_entry":
            result = tool_get_cached_entry(
                cache_dir=_cache_dir,
                prompt=arguments["prompt"],
                model=arguments.get("model", "unknown"),
                threshold=_threshold,
            )

        elif name == "set_threshold":
            result = tool_set_threshold(float(arguments["threshold"]))
            _threshold = result["threshold"]

        elif name == "clear_cache":
            if not arguments.get("confirm", False):
                result = {
                    "success": False,
                    "error": "confirm must be true to clear the cache.",
                }
            else:
                result = tool_clear_cache(
                    cache_dir=_cache_dir,
                    model=arguments.get("model"),
                )

        else:
            result = {"error": f"Unknown tool: {name}"}

    except Exception as exc:
        result = {"error": str(exc)}

    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inferencache-mcp",
        description="inferencache MCP server — expose cache tools to AI coding assistants",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(Path.home() / ".cache" / "inferencache"),
        help="Path to the inferencache data directory",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Initial semantic similarity threshold (default: 0.85)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    global _cache_dir, _threshold

    parser = build_parser()
    args = parser.parse_args(argv)

    _cache_dir = Path(args.cache_dir).expanduser()
    _threshold = args.threshold

    import asyncio

    async def run():
        async with stdio_server() as (read_stream, write_stream):
            await app.run(
                read_stream,
                write_stream,
                app.create_initialization_options(),
            )

    asyncio.run(run())


if __name__ == "__main__":
    main()
