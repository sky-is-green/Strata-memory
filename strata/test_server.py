"""Offline regression tests for the standalone server (strata/server.py).

No encoder weights, no network: fake ultra drone + fake backend injected via
create_app factory params. Covers the wire contract hivebench/Studio/dsh
clients depend on, plus persistence round-trips and the ingest codec guard
at the HTTP boundary.

Run: venv/bin/python -m pytest strata/test_server.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # flat imports

import numpy as np
import pytest
from fastapi.testclient import TestClient

from server import create_app


class FakeUltra:
    """Keyword-scoring drone: GPU chunks score high (like cortex/e2e's)."""

    def score(self, query, chunks):
        from sieve.scores import ChunkScore
        return [ChunkScore(i, 0.9 if "GPU" in c else 0.3, 1.0)
                for i, c in enumerate(chunks)]

    def embed(self, text):
        return np.array([1.0, 0.0, 0.0])


class FakeBackend:
    def generate(self, content, query, sampling=None):
        return "echo: " + query[:40]


@pytest.fixture()
def client(tmp_path):
    app = create_app(ultra_factory=lambda: FakeUltra(),
                     backend_factory=lambda model: FakeBackend(),
                     state_dir=tmp_path / "state",
                     log_dir=str(tmp_path / "logs"))
    with TestClient(app) as c:
        yield c


def _conv_files(state_dir):
    return sorted(p.name for p in Path(state_dir).glob("conv-*.json"))


def test_observe_stores_chunk_and_persists(client, tmp_path):
    r = client.post("/v1/strata/observe", json={
        "conversation_id": "t1", "reply": "my GPU is an RX 7900 XTX"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["stored"] is True
    assert len(_conv_files(tmp_path / "state")) == 1


def test_curate_assembles_relevant_memory(client):
    client.post("/v1/strata/observe", json={
        "conversation_id": "t2", "reply": "my GPU is an RX 7900 XTX"})
    r = client.post("/v1/strata/curate", json={
        "conversation_id": "t2", "query": "what GPU do I have?"})
    assert r.status_code == 200
    assert "RX 7900 XTX" in r.json()["assembled_content"]


def test_turn_roundtrip_with_backend(client):
    r = client.post("/v1/strata/turn", json={
        "conversation_id": "t3", "query": "hello there"})
    assert r.status_code == 200
    body = r.json()
    assert body["reply"].startswith("echo:")
    assert body["conversation_id"] == "t3"


def test_reset_drops_persisted_state(client, tmp_path):
    client.post("/v1/strata/observe", json={
        "conversation_id": "t4", "reply": "some fact"})
    assert _conv_files(tmp_path / "state")
    r = client.post("/v1/strata/reset", json={"conversation_id": "t4"})
    assert r.status_code == 200 and r.json() == {"ok": True}
    assert _conv_files(tmp_path / "state") == []


def test_defaults_and_state_shapes(client):
    d = client.get("/v1/strata/defaults").json()
    assert isinstance(d, dict) and len(d) > 0
    s = client.get("/v1/strata/state").json()
    assert "count" in s and "conversations" in s


def _make_app(tmp_path):
    return create_app(ultra_factory=lambda: FakeUltra(),
                      backend_factory=lambda m: FakeBackend(),
                      state_dir=tmp_path / "state",
                      log_dir=str(tmp_path / "logs"))


def test_persistence_roundtrip_across_restart(tmp_path):
    """A fresh server instance restores the conversation from disk."""
    with TestClient(_make_app(tmp_path)) as c1:
        c1.post("/v1/strata/observe", json={
            "conversation_id": "rt", "reply": "my GPU is an RX 7900 XTX"})
    with TestClient(_make_app(tmp_path)) as c2:
        r = c2.post("/v1/strata/curate", json={
            "conversation_id": "rt", "query": "what GPU do I have?"})
        assert r.status_code == 200
        assert "RX 7900 XTX" in r.json()["assembled_content"]


def test_ingest_repairs_mojibake_at_http_boundary(client):
    """Mojibake arriving over HTTP must be repaired before storage."""
    mangled = "my GPU is an RX 7900 XTX \u00e2\u20ac\u2122"  # ’ as cp1252
    r = client.post("/v1/strata/observe", json={
        "conversation_id": "t6", "reply": mangled})
    assert r.status_code == 200 and r.json()["stored"] is True
    r = client.post("/v1/strata/curate", json={
        "conversation_id": "t6", "query": "what GPU do I have?"})
    body = r.json()
    assert "RX 7900 XTX" in body["assembled_content"]
    assert "\u00e2\u20ac\u2122" not in body["assembled_content"]


def test_mcp_remember_search_roundtrip(client):
    """S2 contract: strata_remember / strata_search over JSON-RPC."""
    r = client.post("/v1/mcp", json={
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "strata_remember",
                   "arguments": {"conversation_id": "m1",
                                  "text": "my GPU is an RX 7900 XTX"}}})
    assert r.status_code == 200
    assert r.json()["result"]["content"][0]["type"] == "text"

    r = client.post("/v1/mcp", json={
        "jsonrpc": "2.0", "id": 2, "method": "tools/call",
        "params": {"name": "strata_search",
                   "arguments": {"conversation_id": "m1",
                                  "query": "what GPU do I have?",
                                  "top_k": 3}}})
    assert r.status_code == 200
    text = r.json()["result"]["content"][0]["text"]
    assert "RX 7900 XTX" in text


def test_mcp_initialize_and_notification(client):
    r = client.post("/v1/mcp", json={
        "jsonrpc": "2.0", "id": 3, "method": "initialize",
        "params": {"protocolVersion": "2024-11-05"}})
    assert r.status_code == 200
    assert "protocolVersion" in r.json()["result"]

    # notification (no id) -> 202, no body
    r = client.post("/v1/mcp", json={
        "jsonrpc": "2.0", "method": "notifications/initialized"})
    assert r.status_code == 202
