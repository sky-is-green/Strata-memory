"""MCP tool definitions + pure handlers for the strata sidecar (S2).

Both tools take ``conversation_id`` as a required argument — isolation is
per conversation, exactly like the ``/v1/strata/*`` REST endpoints. The
handlers operate on a ``Strata`` instance (duck-typed; no import needed) so
this module stays independent of the sidecar's conversation registry.
"""

from __future__ import annotations

import json
from typing import Any, Optional

TOOL_NAMES = ("strata_search", "strata_remember")

SEARCH_TOOL: dict = {
    "name": "strata_search",
    "description": (
        "Recall curated context from the strata memory for one conversation. "
        "Runs the query through the curation pipeline (classify, route, "
        "score, assemble) and returns the curated context plus the recalled "
        "chunks. The search itself is never stored. conversation_id scopes "
        "the recall and is required on every call."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "conversation_id": {
                "type": "string",
                "minLength": 1,
                "description": (
                    "Conversation whose memory to search (required; never "
                    "implied — each conversation's store is isolated)."
                ),
            },
            "query": {
                "type": "string",
                "minLength": 1,
                "description": "Question or cue to recall context for.",
            },
            "top_k": {
                "type": "integer",
                "minimum": 1,
                "maximum": 20,
                "default": 5,
                "description": "Maximum recalled chunks listed in the response.",
            },
        },
        "required": ["conversation_id", "query"],
        "additionalProperties": False,
    },
    "annotations": {"readOnlyHint": True},
}

REMEMBER_TOOL: dict = {
    "name": "strata_remember",
    "description": (
        "Store a fact, decision, or note into the strata memory for one "
        "conversation so later strata_search calls in the same conversation "
        "can recall it. conversation_id scopes the write and is required "
        "on every call."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "conversation_id": {
                "type": "string",
                "minLength": 1,
                "description": (
                    "Conversation whose memory to write (required; never "
                    "implied — each conversation's store is isolated)."
                ),
            },
            "text": {
                "type": "string",
                "minLength": 1,
                "description": "The fact or note to store verbatim.",
            },
        },
        "required": ["conversation_id", "text"],
        "additionalProperties": False,
    },
    "annotations": {"readOnlyHint": False, "idempotentHint": False},
}


def _require_text(value: Any, field: str) -> str:
    """Non-empty string or ValueError (surfaced as an MCP tool error)."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def remember(strata: Any, text: str) -> dict:
    """Store ``text`` verbatim in the conversation's store.

    An explicit memory write, not a model reply: no hedge filtering, the
    store's own sanitization (secret stripping) still applies. Returns a
    JSON-serializable result dict.
    """
    content = _require_text(text, "text")
    chunk_id = strata.store.add_chunk(strata.turn, content)
    if chunk_id is None:
        # harness boilerplate is never stored (ingest blocklist)
        return {
            "ok": True,
            "stored": False,
            "chunk_id": None,
            "turn": strata.turn,
            "store_chunks": len(strata.store.all_chunks()),
        }
    return {
        "ok": True,
        "stored": True,
        "chunk_id": chunk_id,
        "turn": strata.turn,
        "store_chunks": len(strata.store.all_chunks()),
    }


def search(strata: Any, query: str, top_k: int = 5) -> dict:
    """Recall curated context for ``query`` without writing to the store.

    Runs the full curation pipeline via ``process_turn(record_exchange=False)``:
    the turn counter advances (assembly context) but neither the query nor any
    reply is stored, so tool searches never pollute the conversation memory.
    """
    question = _require_text(query, "query")
    try:
        limit = int(top_k)
    except (TypeError, ValueError):
        raise ValueError("top_k must be an integer between 1 and 20")
    if not 1 <= limit <= 20:
        raise ValueError("top_k must be an integer between 1 and 20")
    result = strata.process_turn(question, record_exchange=False)
    assembled = result.assembled
    chunks: list[dict] = []
    if assembled is not None:
        by_id = {c.id: c for c in strata.store.all_chunks()}
        selected = list(getattr(assembled, "selected_chunk_ids", None) or [])
        scores = dict(getattr(assembled, "raw_scores", None) or {})
        # deterministic order: best raw score first, then id
        selected.sort(key=lambda cid: (-scores.get(cid, 0.0), cid))
        for cid in selected[:limit]:
            chunk = by_id.get(cid)
            text = chunk.content if chunk is not None else "(evicted)"
            chunks.append({
                "id": cid,
                "score": round(float(scores.get(cid, 0.0)), 3),
                "preview": text[:500],
            })
    return {
        "query": question,
        "turn": result.turn,
        "assembled_content": assembled.content if assembled is not None else "",
        "token_count": result.token_count,
        "budget": result.budget,
        "mode": result.mode,
        "chunks": chunks,
    }


def result_text(payload: dict) -> str:
    """Serialize a tool result for the MCP ``content: [{type: text}]`` block."""
    return json.dumps(payload, ensure_ascii=False)


def parse_top_k(arguments: dict) -> int:
    """Extract ``top_k`` with the schema default (5) when absent."""
    raw: Optional[Any] = arguments.get("top_k", 5)
    try:
        return int(raw)
    except (TypeError, ValueError):
        raise ValueError("top_k must be an integer between 1 and 20")
