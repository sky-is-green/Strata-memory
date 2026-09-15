#!/usr/bin/env python3
"""One-time scrub: normalize legacy codec-mangled content in persisted
conversation stores (harness_state/conv-*.json).

Safe by construction:
  * Refuses to run while the sidecar is listening on :8765 (its in-memory
    state would overwrite the scrubbed files on next save).
  * Backs up every file it touches into a timestamped dir first.
  * Atomic per-file rewrite (tmp + os.replace), permissions preserved.
  * Uses retention.codec.normalize_codec — idempotent, never-destroy:
    clean chunks are left byte-identical; only genuine mojibake is repaired.
  * Recomputes each touched chunk's fingerprint so dedup grouping stays
    consistent with the (now clean) stored form.

Usage:
  venv/bin/python scripts/scrub_codec_state.py [--state-dir DIR] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "strata"))

from retention.codec import normalize_codec  # noqa: E402
from retention.hygiene import content_fingerprint  # noqa: E402


def sidecar_alive(port: int = 8765) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        try:
            s.connect(("127.0.0.1", port))
            return True
        except OSError:
            return False


def scrub_file(path: Path, dry_run: bool) -> tuple[int, int]:
    """Returns (chunks_touched, chunks_total)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    store = data.get("store") or {}
    chunks = store.get("chunks") or []
    touched = 0
    for chunk in chunks:
        content = chunk.get("content", "")
        fixed = normalize_codec(content)
        if fixed != content:
            chunk["content"] = fixed
            chunk["fingerprint"] = content_fingerprint(fixed)
            touched += 1
    if touched and not dry_run:
        backup_note = f"scrubbed {datetime.now(timezone.utc).isoformat()}"
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False)
            st = path.stat()
            os.chmod(tmp, st.st_mode)
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
    return touched, len(chunks)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state-dir", default=str(Path(__file__).resolve().parents[1] / "harness_state"))
    ap.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    args = ap.parse_args()

    state_dir = Path(args.state_dir)
    files = sorted(state_dir.glob("conv-*.json"))
    if not files:
        print(f"no conv-*.json in {state_dir}")
        return 0

    if not args.dry_run and sidecar_alive():
        print("REFUSED: sidecar is listening on :8765 — stop it first "
              "(its in-memory state would overwrite the scrubbed files).")
        return 2

    backup_dir = None
    if not args.dry_run:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_dir = state_dir / f"backup-pre-scrub-{stamp}"
        backup_dir.mkdir()
        for f in files:
            b = backup_dir / f.name
            b.write_bytes(f.read_bytes())
            os.chmod(b, os.stat(f).st_mode & 0o777)  # preserve original mode

    total_touched = total_chunks = 0
    print(f"{'DRY-RUN' if args.dry_run else 'SCRUB'}: {len(files)} files in {state_dir}")
    for f in files:
        touched, n = scrub_file(f, args.dry_run)
        total_touched += touched
        total_chunks += n
        if touched:
            print(f"  {f.name}: {touched}/{n} chunks repaired")

    print(f"\nTOTAL: {total_touched}/{total_chunks} chunks repaired "
          f"across {len(files)} files" + (f" (backup: {backup_dir})" if backup_dir else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
