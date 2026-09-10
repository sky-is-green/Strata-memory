"""Minimal MCP (Streamable HTTP) JSON-RPC dispatcher (S2).

Dependency-free on purpose: the sidecar speaks MCP without the ``mcp``
PyPI package (no new dependency, no ``pyproject.toml`` hotspot touch).
Only the messages a tool-using client needs are implemented:

- ``initialize`` / ``notifications/initialized`` (handshake)
- ``ping``
- ``tools/list`` (``strata_search`` / ``strata_remember``)
- ``tools/call`` (validated; ``conversation_id`` required, never implicit)

The transport answers ``POST`` with ``application/json`` (the spec allows a
plain-JSON response instead of an SSE stream) and ``405`` on ``GET``/``DELETE``
(stateless server: no SSE streams to open or close). Batch arrays are
supported; notifications (no ``id``) yield no response.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from strata.mcp import tools as _tools

try:  # local dist metadata (editable install); never a network call
    from importlib.metadata import version as _dist_version

    _DIST_VERSION = _dist_version("strata-memory")
except Exception:  # noqa: BLE001 - fall back to the static version
    _DIST_VERSION = "0.1.0"

PROTOCOL_VERSION = "2025-06-18"
SUPPORTED_PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
SERVER_INFO = {"name": "strata-memory", "version": _DIST_VERSION}


class McpContext:
    """Per-request sidecar bindings injected by ``harness/app.py``.

    ``remember`` / ``search`` close over the sidecar's conversation registry
    (``strata_for`` + locks + persistence); this module never touches it.
    """

    def __init__(
        self,
        remember: Callable[[str, str], dict],
        search: Callable[[str, str, int], dict],
    ) -> None:
        self._remember = remember
        self._search = search

    def remember(self, conversation_id: str, text: str) -> dict:
        return self._remember(conversation_id, text)

    def search(self, conversation_id: str, query: str, top_k: int) -> dict:
        return self._search(conversation_id, query, top_k)


def _ok(msg_id: Any, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _err(msg_id: Any, code: int, message: str) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "error": {"code": code, "message": message},
    }


def _conversation_id(arguments: Any) -> str:
    """Extract the required scope; never default, never header-implied."""
    if not isinstance(arguments, dict):
        raise ValueError("tools/call params.arguments must be an object")
    cid = arguments.get("conversation_id")
    if not isinstance(cid, str) or not cid.strip():
        raise ValueError(
            "conversation_id is required (non-empty string) on every "
            "strata_search / strata_remember call; it is never implicit"
        )
    return cid.strip()


def _call_tool(ctx: McpContext, params: Any) -> dict:
    if not isinstance(params, dict):
        raise ValueError("tools/call params must be an object")
    name = params.get("name")
    arguments = params.get("arguments") or {}
    if name == "strata_remember":
        cid = _conversation_id(arguments)
        text = arguments.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        payload = ctx.remember(cid, text)
    elif name == "strata_search":
        cid = _conversation_id(arguments)
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        payload = ctx.search(cid, query, _tools.parse_top_k(arguments))
    else:
        raise KeyError(f"unknown tool: {name!r}")
    return {
        "content": [{"type": "text", "text": _tools.result_text(payload)}],
    }


def _dispatch_one(msg: Any, ctx: McpContext) -> Optional[dict]:
    """Dispatch one JSON-RPC message; None = notification, no response."""
    if not isinstance(msg, dict) or not msg.get("method"):
        msg_id = msg.get("id") if isinstance(msg, dict) else None
        return _err(msg_id, -32600, "invalid JSON-RPC request object")
    method = msg["method"]
    has_id = "id" in msg
    msg_id = msg.get("id")
    params = msg.get("params") or {}

    try:
        if method == "initialize":
            requested = params.get("protocolVersion") if isinstance(params, dict) else None
            negotiated = (
                requested
                if requested in SUPPORTED_PROTOCOL_VERSIONS
                else PROTOCOL_VERSION
            )
            result = {
                "protocolVersion": negotiated,
                "capabilities": {"tools": {}},
                "serverInfo": dict(SERVER_INFO),
            }
        elif method == "notifications/initialized":
            return None
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": [_tools.SEARCH_TOOL, _tools.REMEMBER_TOOL]}
        elif method == "tools/call":
            try:
                result = _call_tool(ctx, params)
            except ValueError as exc:
                return _err(msg_id, -32602, str(exc)) if has_id else None
            except KeyError as exc:
                return _err(msg_id, -32601, str(exc)) if has_id else None
            except Exception as exc:  # noqa: BLE001 - tool failure, not protocol failure
                result = {
                    "content": [{"type": "text", "text": f"strata error: {exc}"}],
                    "isError": True,
                }
        else:
            return _err(msg_id, -32601, f"method not found: {method}") if has_id else None
    except Exception as exc:  # noqa: BLE001 - never break the stream on a bad param
        return _err(msg_id, -32602, str(exc)) if has_id else None

    if not has_id:
        return None
    return _ok(msg_id, result)


def handle_message(msg: Any, ctx: McpContext) -> Any:
    """Dispatch a decoded JSON-RPC body (single message or batch).

    Returns a response dict, a response list (batch), or None when the body
    held only notifications.
    """
    if isinstance(msg, list):
        responses = [
            resp for resp in (_dispatch_one(m, ctx) for m in msg) if resp is not None
        ]
        return responses or None
    return _dispatch_one(msg, ctx)
