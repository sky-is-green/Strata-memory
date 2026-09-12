"""NDJSON event logger with daily rotation, size rotation, redaction, and
correlation IDs. One JSON line per event, one file per UTC day, archived and
gzipped when it exceeds a size cap, pruned by retention.

Every strata component writes structured events through this logger. Writes are
buffered on a background thread so logging never blocks the hot path (Pitfall 8).

Schema per entry::

    {"ts", "component", "event_type", "payload", "latency_ms",
     "run_id"?, "conversation_id"?, "turn_id"?}

Sensitive values (api keys, tokens, secrets) are redacted before writing.
"""

from __future__ import annotations

import atexit
import gzip
import json
import queue
import re
import threading
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

NOW_FN = Callable[[], datetime]

REQUIRED_FIELDS = ("ts", "component", "event_type", "payload", "latency_ms")

# Secret keys whose values are always redacted.
REDACT_KEYS = {
    "api_key", "apikey", "token", "secret", "password", "passwd",
    "authorization", "auth", "access_token", "refresh_token", "private_key",
}
_TOKEN_LIKE = re.compile(
    r"(?i)\b(?:sk-|ghp_|gho_|bearer |token=)[A-Za-z0-9_\-\.]{8,}\b"
)
_LONG_HEX = re.compile(r"\b[0-9a-fA-F]{24,}\b")


def redact(value: Any) -> Any:
    """Recursively redact secret keys and token-like strings in a value."""
    if isinstance(value, dict):
        return {
            k: ("[REDACTED]" if str(k).lower() in REDACT_KEYS else redact(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, tuple):
        return tuple(redact(v) for v in value)
    if isinstance(value, str):
        return _LONG_HEX.sub("[REDACTED]", _TOKEN_LIKE.sub("[REDACTED]", value))
    return value


def validate_entry(entry: dict) -> None:
    """Raise ValueError if *entry* is missing a required field or type."""
    for field in REQUIRED_FIELDS:
        if field not in entry:
            raise ValueError(f"entry missing required field: {field!r}")
    if not isinstance(entry["component"], str):
        raise ValueError("component must be str")
    if not isinstance(entry["event_type"], str):
        raise ValueError("event_type must be str")
    if not isinstance(entry["payload"], dict):
        raise ValueError("payload must be dict")
    if not isinstance(entry["latency_ms"], (int, float)):
        raise ValueError("latency_ms must be numeric")


class EventLogger:
    """Appends one JSON line per event to a rotated NDJSON file.

    Parameters
    ----------
    log_dir:
        Directory for the NDJSON files and the ``archive/`` subdir (created if
        missing).
    flush_interval_s:
        Idle time before the writer thread checks for a stop signal.
    now_fn:
        Injectable clock for deterministic rotation tests. Defaults to UTC now.
    max_bytes:
        Size cap per daily file; when exceeded the file is archived (gzipped).
    retention_days:
        Archive files older than this many days are pruned.
    redact_secrets:
        Whether to redact sensitive values before writing.
    """

    def __init__(
        self,
        log_dir: str | Path = "logs",
        flush_interval_s: float = 0.5,
        now_fn: Optional[NOW_FN] = None,
        max_bytes: int = 50 * 1024 * 1024,
        retention_days: int = 7,
        redact_secrets: bool = True,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir = self.log_dir / "archive"
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self._flush_interval_s = flush_interval_s
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))
        self._max_bytes = max_bytes
        self._retention_days = retention_days
        self._redact = redact_secrets

        self._queue: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._current_date: Optional[date] = None
        self._stream = None
        self._bytes_written = 0
        self._seq = 0
        self._lock = threading.Lock()

        self._writer = threading.Thread(
            target=self._run, name="event-logger", daemon=True
        )
        self._writer.start()
        atexit.register(self.close)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def log(
        self,
        component: str,
        event_type: str,
        payload: dict,
        latency_ms: float = 0.0,
        run_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        turn_id: Optional[int] = None,
    ) -> None:
        """Enqueue one structured event for asynchronous disk write."""
        payload = redact(payload) if self._redact else payload
        entry: dict = {
            "ts": self._now_fn().isoformat().replace("+00:00", "Z"),
            "component": component,
            "event_type": event_type,
            "payload": payload,
            "latency_ms": round(float(latency_ms), 2),
        }
        if run_id is not None:
            entry["run_id"] = run_id
        if conversation_id is not None:
            entry["conversation_id"] = conversation_id
        if turn_id is not None:
            entry["turn_id"] = turn_id
        self._queue.put(entry)

    def flush(self) -> None:
        """Block until all enqueued events are written to disk."""
        self._queue.join()

    def close(self) -> None:
        """Stop the writer thread and close any open file handle. Idempotent."""
        with self._lock:
            if self._stop.is_set():
                return
            self._stop.set()
        self._writer.join(timeout=5.0)
        with self._lock:
            if self._stream is not None:
                self._stream.close()
                self._stream = None

    def read_entries(self, date_str: Optional[str] = None) -> list[dict]:
        """Parse and return every NDJSON entry (optionally for one date).

        Only live (non-archived) files are read.
        """
        pattern = "events-*.ndjson" if date_str is None else f"events-{date_str}.ndjson"
        entries: list[dict] = []
        for path in sorted(self.log_dir.glob(pattern)):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entries.append(json.loads(line))
        return entries

    def cleanup_archives(self, retention_days: Optional[int] = None) -> int:
        """Delete archive files older than the retention period. Returns count."""
        days = retention_days if retention_days is not None else self._retention_days
        cutoff = time.time() - days * 86400
        removed = 0
        for path in self.archive_dir.glob("*.ndjson.gz"):
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        return removed

    def __enter__(self) -> "EventLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Writer internals
    # ------------------------------------------------------------------
    def _run(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=self._flush_interval_s)
            except queue.Empty:
                if self._stop.is_set():
                    break
                continue
            try:
                self._write(item)
            finally:
                self._queue.task_done()

    def _write(self, entry: dict) -> None:
        with self._lock:
            now = self._now_fn()
            if self._current_date != now.date():
                self._roll_date(now)
            line = json.dumps(entry, default=str, ensure_ascii=False) + "\n"
            self._stream.write(line)
            self._bytes_written += len(line.encode("utf-8"))
            if self._bytes_written >= self._max_bytes:
                self._rotate()

    def _roll_date(self, now: datetime) -> None:
        if self._stream is not None:
            self._stream.close()
        self._current_date = now.date()
        self._seq = 0
        self._bytes_written = 0
        self._stream = open(
            self.log_dir / f"events-{now.date().isoformat()}.ndjson",
            "a",
            encoding="utf-8",
        )

    def _rotate(self) -> None:
        """Close the current file, gzip-archive it, and start a fresh one."""
        if self._stream is not None:
            self._stream.close()
        src = self.log_dir / f"events-{self._current_date.isoformat()}.ndjson"
        self._seq += 1
        dest = self.archive_dir / f"events-{self._current_date.isoformat()}-{self._seq:04d}.ndjson.gz"
        with open(src, "rb") as fin, gzip.open(dest, "wb") as fout:
            import shutil

            shutil.copyfileobj(fin, fout)
        src.unlink()
        self._bytes_written = 0
        self._stream = open(
            self.log_dir / f"events-{self._current_date.isoformat()}.ndjson",
            "w",
            encoding="utf-8",
        )
        self.cleanup_archives()
