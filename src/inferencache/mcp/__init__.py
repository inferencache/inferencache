"""MCP server package for inferencache."""

__all__ = ["main"]


def __getattr__(name: str):
    if name == "main":
        from .server import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
