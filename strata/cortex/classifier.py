"""Lightweight routing classifier (S4.6).

Trained offline on logged routing decisions + queen labels; replaces heuristic
routing at runtime. A shallow decision tree (<10MB, <20ms inference) is small
enough to load at startup and fast enough for the hot path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from cortex.routing import DroneRouter

# Feature vector layout (must stay in sync with _extract_features).
FEATURES = [
    "message_length",
    "keyword_density",
    "code_block_count",
    "conversation_depth",
    "avg_chunk_age",
    "topic_drift_score",
]


@dataclass
class RoutingRecord:
    query: str
    optimal_route: str
    features: list = field(default_factory=list)


class RoutingClassifier:
    def __init__(self, max_depth: int = 5) -> None:
        from sklearn.tree import DecisionTreeClassifier

        self.model = DecisionTreeClassifier(max_depth=max_depth)
        self._trained = False

    def train(self, records: list[RoutingRecord]) -> None:
        X = [r.features for r in records]
        y = [r.optimal_route for r in records]
        self.model.fit(X, y)
        self._trained = True

    def predict(
        self,
        query: str,
        conversation_history: Optional[list] = None,
        context_stats: Optional[dict] = None,
    ) -> str:
        features = self._extract_features(query, conversation_history or [], context_stats)
        return self.predict_features(features)

    def predict_features(self, features: list) -> str:
        """Predict a route directly from a feature vector."""
        return self.model.predict([features])[0]

    def _extract_features(
        self, query: str, history: list, context_stats: Optional[dict] = None
    ) -> list:
        q = query.lower()
        keywords = DroneRouter.HEURISTIC_RULES["complex_keywords"]
        kw_count = sum(1 for k in keywords if k in q)
        stats = context_stats or {}
        return [
            len(query),
            kw_count / max(len(q.split()), 1),
            query.count("```"),
            len(history),
            stats.get("avg_chunk_age", 0.0),
            stats.get("topic_drift_score", 0.0),
        ]
