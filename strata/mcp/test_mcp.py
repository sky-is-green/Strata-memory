"""Unit tests for the strata MCP package (S2): tool handlers + dispatcher.

All offline: real ``Strata`` with the fake drone / mock transport (the same
offline shape as the sidecar tests), so no LM Studio or encoder download.
"""

import json

import pytest

from backend.openai_compat import OpenAICompatBackend
from cortex.config import StrataConfig
from cortex.e2e import FakeUltraSmall, MockTransport
from cortex.strata import Strata
from strata.mcp import server as mcp_server
from strata.mcp import tools as mcp_tools


def _strata() -> Strata:
    return Strata(
        config=StrataConfig(confidence_mode="off"),
        ultra=FakeUltraSmall(),
        backend=OpenAICompatBackend(
            base_url="http://mock", model="mock-model",
            transport=MockTransport(latency_ms=0),
        ),
    )


def _ctx_for(store: dict):
    """McpContext bound to per-conversation Strata instances."""

    def do_remember(conversation_id: str, text: str) -> dict:
        strata = store.setdefault(conversation_id, _strata())
        return mcp_tools.remember(strata, text)

    def do_search(conversation_id: str, query: str, top_k: int) -> dict:
        strata = store.setdefault(conversation_id, _strata())
        return mcp_tools.search(strata, query, top_k)

    return mcp_server.McpContext(remember=do_remember, search=do_search)


# ---------------------------------------------------------------------------
# tool schemas: conversation_id is required, never implicit
# ---------------------------------------------------------------------------
def test_tool_names():
    assert mcp_tools.TOOL_NAMES == ("strata_search", "strata_remember")


def test_both_tools_require_conversation_id():
    for tool in (mcp_tools.SEARCH_TOOL, mcp_tools.REMEMBER_TOOL):
        schema = tool["inputSchema"]
        assert "conversation_id" in schema["required"]
        assert "default" not in schema["properties"]["conversation_id"]
    assert "query" in mcp_tools.SEARCH_TOOL["inputSchema"]["required"]
    assert "text" in mcp_tools.REMEMBER_TOOL["inputSchema"]["required"]


# ---------------------------------------------------------------------------
# handlers: remember -> search round-trip + isolation
# ---------------------------------------------------------------------------
def test_remember_then_search_roundtrip():
    store: dict = {}
    ctx = _ctx_for(store)
    out = ctx.remember("conv-a", "Deploy tokens rotate every 90 days per policy.")
    assert out["stored"] is True and out["chunk_id"]

    found = ctx.search("conv-a", "What is the deploy token rotation period?", 5)
    assert "90 days" in found["assembled_content"]
    assert found["token_count"] > 0


def test_search_isolated_across_conversations():
    store: dict = {}
    ctx = _ctx_for(store)
    ctx.remember("conv-a", "Deploy tokens rotate every 90 days per policy.")

    other = ctx.search("conv-b", "What is the deploy token rotation period?", 5)
    assert "90 days" not in other["assembled_content"]


def test_remember_rejects_empty_text():
    with pytest.raises(ValueError):
        mcp_tools.remember(_strata(), "   ")


def test_search_rejects_empty_query_and_bad_top_k():
    with pytest.raises(ValueError):
        mcp_tools.search(_strata(), "  ")
    with pytest.raises(ValueError):
        mcp_tools.search(_strata(), "hello", 0)
    with pytest.raises(ValueError):
        mcp_tools.search(_strata(), "hello", 21)


def test_search_does_not_store():
    strata = _strata()
    before = len(strata.store.all_chunks())
    mcp_tools.search(strata, "Which tokens do I use for auth expiry?")
    assert len(strata.store.all_chunks()) == before


# ---------------------------------------------------------------------------
# dispatcher: handshake, tools/list, tools/call, errors, batch
# ---------------------------------------------------------------------------
def _call(name: str, arguments: dict, msg_id=1) -> dict:
    return {
        "jsonrpc": "2.0", "id": msg_id,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }


def test_initialize_negotiates_protocol_version():
    ctx = _ctx_for({})
    resp = mcp_server.handle_message({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-03-26", "capabilities": {}},
    }, ctx)
    assert resp["result"]["protocolVersion"] == "2025-03-26"
    assert resp["result"]["serverInfo"]["name"] == "strata-memory"
    assert resp["result"]["capabilities"] == {"tools": {}}

    resp = mcp_server.handle_message({
        "jsonrpc": "2.0", "id": 2, "method": "initialize",
        "params": {"protocolVersion": "9999-99-99"},
    }, ctx)
    assert resp["result"]["protocolVersion"] == mcp_server.PROTOCOL_VERSION


def test_initialized_notification_has_no_response():
    ctx = _ctx_for({})
    assert mcp_server.handle_message({
        "jsonrpc": "2.0", "method": "notifications/initialized",
    }, ctx) is None


def test_ping():
    ctx = _ctx_for({})
    resp = mcp_server.handle_message(
        {"jsonrpc": "2.0", "id": 7, "method": "ping"}, ctx)
    assert resp == {"jsonrpc": "2.0", "id": 7, "result": {}}


def test_tools_list_exposes_both_tools():
    ctx = _ctx_for({})
    resp = mcp_server.handle_message(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
        ctx,
    )
    names = {t["name"] for t in resp["result"]["tools"]}
    assert names == {"strata_search", "strata_remember"}


def test_tools_call_roundtrip_through_dispatcher():
    store: dict = {}
    ctx = _ctx_for(store)
    resp = mcp_server.handle_message(
        _call("strata_remember", {
            "conversation_id": "conv-mcp",
            "text": "Deploy tokens rotate every 90 days per policy.",
        }), ctx)
    assert "error" not in resp
    assert json.loads(resp["result"]["content"][0]["text"])["stored"] is True

    resp = mcp_server.handle_message(
        _call("strata_search", {
            "conversation_id": "conv-mcp",
            "query": "What is the deploy token rotation period?",
        }, msg_id=2), ctx)
    assert "error" not in resp
    payload = json.loads(resp["result"]["content"][0]["text"])
    assert "90 days" in payload["assembled_content"]


def test_tools_call_requires_conversation_id():
    ctx = _ctx_for({})
    for arguments in ({}, {"conversation_id": ""}, {"conversation_id": "  "},
                      {"conversation_id": None, "query": "q"}):
        resp = mcp_server.handle_message(
            _call("strata_search", {**arguments, "query": "q"}), ctx)
        assert resp["error"]["code"] == -32602, arguments
        assert "conversation_id" in resp["error"]["message"]


def test_tools_call_unknown_tool_and_method():
    ctx = _ctx_for({})
    resp = mcp_server.handle_message(
        _call("nope", {"conversation_id": "c"}), ctx)
    assert resp["error"]["code"] == -32601
    resp = mcp_server.handle_message(
        {"jsonrpc": "2.0", "id": 1, "method": "bogus/method"}, ctx)
    assert resp["error"]["code"] == -32601


def test_tools_call_invalid_arguments_shape():
    ctx = _ctx_for({})
    resp = mcp_server.handle_message({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "strata_search", "arguments": {
            "conversation_id": "c", "query": "   "}},
    }, ctx)
    assert resp["error"]["code"] == -32602


def test_batch_and_notification_only_body():
    store: dict = {}
    ctx = _ctx_for(store)
    resp = mcp_server.handle_message([
        {"jsonrpc": "2.0", "id": 1, "method": "ping"},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    ], ctx)
    assert isinstance(resp, list) and len(resp) == 2
    assert resp[0]["id"] == 1 and resp[1]["id"] == 2

    assert mcp_server.handle_message([
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
    ], ctx) is None


def test_malformed_request_is_invalid_request():
    ctx = _ctx_for({})
    resp = mcp_server.handle_message({"nope": True}, ctx)
    assert resp["error"]["code"] == -32600
