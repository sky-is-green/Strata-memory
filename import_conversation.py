"""Seed a strata conversation from an existing saved chat (unsloth Studio DB).

Usage: python import_conversation.py <thread_id> [conv_name]
Reads all messages in order, stores each as a chunk (turn = message index),
and writes the standard harness state file. The running harness picks it up
lazily on first touch — no restart needed. Embeddings are never stored in
the payload; retrieval recomputes them with the active encoder, so the seed
is encoder-agnostic by construction.
"""
import hashlib, json, os, sqlite3, sys
from pathlib import Path

sys.path.insert(0, ".")
from strata.cortex.config import HiveConfig
from strata.retention.store import ContextStore

DB = "/home/penis/.unsloth/studio/studio.db"
STATE_DIR = Path("harness_state")

def extract_text(content_json: str) -> str:
    try:
        obj = json.loads(content_json or "null")
    except (ValueError, TypeError):
        return ""
    if isinstance(obj, str):
        return obj.strip()
    if isinstance(obj, list):
        return "\n".join(
            p.get("text", "").strip() for p in obj
            if isinstance(p, dict) and p.get("type") == "text" and (p.get("text") or "").strip()
        ).strip()
    return ""

def main():
    thread_id = sys.argv[1]
    conv_name = sys.argv[2] if len(sys.argv) > 2 else f"import-{thread_id[:8]}"
    con = sqlite3.connect(DB)
    rows = con.execute(
        "SELECT role, content_json FROM chat_messages WHERE thread_id=? ORDER BY rowid",
        (thread_id,)).fetchall()
    con.close()

    # max_chunks high enough that the seed itself never evicts; the cap is
    # what protects live growth, and 1000 is the harness default.
    store = ContextStore(max_chunks=max(1000, len(rows)))
    n = 0
    for i, (role, cj) in enumerate(rows):
        text = extract_text(cj)
        if not text:
            continue
        content = f"[{role}] {text}"
        cid = hashlib.md5(f"{i}:{content}".encode()).hexdigest()[:12]
        store.add_chunk(i, content, chunk_id=cid)
        n += 1

    cfg = HiveConfig(confidence_mode="off")
    digest = hashlib.md5(conv_name.encode()).hexdigest()[:16]
    path = STATE_DIR / f"conv-{digest}.json"
    payload = {
        "conversation_id": conv_name, "turn": n, "with_backend": False,
        "config": cfg.to_dict(), "store": store.to_dict(),
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, path)
    print(f"seeded {n} chunks -> conv '{conv_name}' ({path.name})")

if __name__ == "__main__":
    main()
