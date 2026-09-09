"""Ground-truth database (SQLite).

Stores queen labels and strata decisions for retrieval-quality metrics:
precision, recall, false-eviction rate, and routing accuracy. Zero-config
single-file SQLite is sufficient at this scale (plan S4.2 / tech stack).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional


class GroundTruthDB:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS queen_labels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                turn INTEGER NOT NULL,
                chunk_id TEXT NOT NULL,
                predicted_relevant INTEGER NOT NULL,
                actually_relevant INTEGER NOT NULL,
                score REAL
            );
            CREATE TABLE IF NOT EXISTS hive_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                turn INTEGER NOT NULL,
                decision_type TEXT NOT NULL,
                params TEXT,
                outcome TEXT
            );
            CREATE TABLE IF NOT EXISTS parameter_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version TEXT NOT NULL,
                params_json TEXT NOT NULL
            );
            """
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Records
    # ------------------------------------------------------------------
    def record_queen_label(
        self,
        turn: int,
        chunk_id: str,
        predicted_relevant: bool,
        actually_relevant: bool,
        score: Optional[float] = None,
    ) -> None:
        self._conn.execute(
            "INSERT INTO queen_labels(turn, chunk_id, predicted_relevant, actually_relevant, score) "
            "VALUES (?,?,?,?,?)",
            (turn, chunk_id, int(predicted_relevant), int(actually_relevant), score),
        )
        self._conn.commit()

    def record_hive_decision(
        self, turn: int, decision_type: str, params: dict | None = None, outcome: str = ""
    ) -> None:
        self._conn.execute(
            "INSERT INTO hive_decisions(turn, decision_type, params, outcome) VALUES (?,?,?,?)",
            (turn, decision_type, json.dumps(params) if params else None, outcome),
        )
        self._conn.commit()

    def record_routing_decision(self, turn: int, score: float, optimal_route: str) -> None:
        """Record a routing decision with its heuristic score and the queen's
        optimal route, so routing thresholds can be replayed later (S4.4)."""
        outcome = "correct" if self._route_for(score, default_threshold=2) == optimal_route else "wrong"
        self.record_hive_decision(
            turn, "route", {"score": score, "optimal": optimal_route}, outcome
        )

    @staticmethod
    def _route_for(score: float, default_threshold: int) -> str:
        if score >= 3:
            return "escalation"
        if score >= default_threshold:
            return "medium"
        return "ultra_small"

    def record_parameter_version(self, version: str, params: dict) -> None:
        self._conn.execute(
            "INSERT INTO parameter_versions(version, params_json) VALUES (?,?)",
            (version, json.dumps(params)),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def _label_rows(self, window: int) -> list:
        return self._conn.execute(
            "SELECT predicted_relevant, actually_relevant FROM "
            "(SELECT * FROM queen_labels ORDER BY id DESC LIMIT ?)",
            (window,),
        ).fetchall()

    def retrieval_precision(self, window: int = 100) -> float:
        """% of retrieved (predicted-relevant) chunks that were actually relevant."""
        rows = self._label_rows(window)
        predicted = sum(1 for r in rows if r["predicted_relevant"])
        if not predicted:
            return 0.0
        correct = sum(1 for r in rows if r["predicted_relevant"] and r["actually_relevant"])
        return correct / predicted * 100.0

    def retrieval_recall(self, window: int = 100) -> float:
        """% of actually-relevant chunks that were retrieved."""
        rows = self._label_rows(window)
        relevant = sum(1 for r in rows if r["actually_relevant"])
        if not relevant:
            return 0.0
        retrieved = sum(1 for r in rows if r["actually_relevant"] and r["predicted_relevant"])
        return retrieved / relevant * 100.0

    def false_eviction_rate(self, window: int = 100) -> float:
        """% of evicted (predicted-not-relevant) chunks that were actually needed."""
        rows = self._label_rows(window)
        evicted = sum(1 for r in rows if not r["predicted_relevant"])
        if not evicted:
            return 0.0
        wrongly = sum(1 for r in rows if not r["predicted_relevant"] and r["actually_relevant"])
        return wrongly / evicted * 100.0

    def routing_accuracy(self, window: int = 100) -> float:
        """% of routing decisions recorded as correct."""
        rows = self._conn.execute(
            "SELECT outcome FROM (SELECT * FROM hive_decisions "
            "WHERE decision_type='route' ORDER BY id DESC LIMIT ?)",
            (window,),
        ).fetchall()
        if not rows:
            return 0.0
        return sum(1 for r in rows if r["outcome"] == "correct") / len(rows) * 100.0

    def label_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) AS n FROM queen_labels").fetchone()["n"]
