"""Strata MCP server package (S2).

Exposes the conversation store through the Model Context Protocol over the
sidecar's Streamable HTTP endpoint (``POST /v1/mcp`` in ``harness/app.py``):

- ``strata_search`` — recall curated context for a conversation.
- ``strata_remember`` — store a fact for a conversation.

``conversation_id`` is a **required tool argument** on both tools — the
server never implies it from headers, payload fields, or defaults.

This package never imports from ``harness/``: the sidecar injects a small
context object (``McpContext``) carrying the per-conversation callables, so
the tools stay portable. Import surface::

    from strata.mcp.tools import SEARCH_TOOL, REMEMBER_TOOL, remember, search
    from strata.mcp.server import McpContext, handle_message
"""

from strata.mcp.server import (
    PROTOCOL_VERSION,
    SERVER_INFO,
    SUPPORTED_PROTOCOL_VERSIONS,
    McpContext,
    handle_message,
)
from strata.mcp.tools import (
    REMEMBER_TOOL,
    SEARCH_TOOL,
    TOOL_NAMES,
    remember,
    search,
)

__all__ = [
    "PROTOCOL_VERSION",
    "SERVER_INFO",
    "SUPPORTED_PROTOCOL_VERSIONS",
    "McpContext",
    "handle_message",
    "REMEMBER_TOOL",
    "SEARCH_TOOL",
    "TOOL_NAMES",
    "remember",
    "search",
]
