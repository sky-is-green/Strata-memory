"""Shared score types for the drone fleet."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChunkScore:
    """A single chunk's relevance score from a drone.

    ``chunk_id`` is the positional index into the chunks list passed to the
    drone's ``score()`` method (S1 interface; S2 maps it to store hashes).
    """

    chunk_id: int
    relevance_score: float
    confidence: float
    source: str = "ultra_small"
