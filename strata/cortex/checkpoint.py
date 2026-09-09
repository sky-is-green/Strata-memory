"""Hive checkpoint system.

Periodically saves strata state (context store, decay matrix, parameter config) so
rollback can restore a known-good state. ``auto_checkpoint`` saves only when the
PES is high (known-good), per the plan's improvement.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Optional

# Only safe characters are allowed in checkpoint tags to prevent path traversal.
_SAFE_TAG = re.compile(r"^[A-Za-z0-9_.\-]+$")


class HiveCheckpoint:
    def __init__(self, directory: str | Path = "checkpoints") -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _sanitize(tag: str) -> str:
        if not tag or not _SAFE_TAG.match(tag):
            raise ValueError(f"unsafe checkpoint tag: {tag!r}")
        return tag

    def _path(self, tag: str) -> Path:
        safe = self._sanitize(tag)
        path = (self.dir / f"checkpoint_{safe}.json").resolve()
        # Defense in depth: the resolved path must stay inside the checkpoint dir.
        if not str(path).startswith(str(self.dir.resolve())):
            raise ValueError(f"checkpoint path escapes directory: {tag!r}")
        return path

    def save(self, state: dict, tag: str) -> Path:
        import json

        path = self._path(tag)
        path.write_text(json.dumps(state, default=str, indent=2), encoding="utf-8")
        return path

    def restore(self, tag: str) -> dict:
        import json

        return json.loads(self._path(tag).read_text(encoding="utf-8"))

    def auto_checkpoint(self, state: dict, pes: float, tag_prefix: str = "pes") -> Optional[Path]:
        """Save only when PES is high (a known-good state)."""
        if pes > 80:
            tag = f"{tag_prefix}_{pes:.0f}_{datetime.now().astimezone():%Y%m%d_%H%M%S}"
            return self.save(state, tag)
        return None

    def list_checkpoints(self) -> list[Path]:
        return sorted(self.dir.glob("checkpoint_*.json"))
