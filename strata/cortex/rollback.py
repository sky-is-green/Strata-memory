"""Automated rollback.

If the PES drops below a threshold for N consecutive turns, revert to the last
known good configuration:

  - PES < 50 for 10 consecutive turns  -> immediate rollback
  - PES < 60 for 25 consecutive turns  -> warning-period rollback
  - PES trending down (slope < -0.5) over 50 turns -> trend rollback
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RollbackDecision:
    should_rollback: bool
    reason: str = ""


class AutomatedRollback:
    IMMEDIATE_THRESHOLD = 50
    IMMEDIATE_WINDOW = 10
    WARNING_THRESHOLD = 60
    WARNING_WINDOW = 25
    TREND_WINDOW = 50
    TREND_SLOPE = -0.5

    def check(self, recent_pes: list[float]) -> RollbackDecision:
        n = len(recent_pes)
        if n < self.IMMEDIATE_WINDOW:
            return RollbackDecision(False)

        if all(p < self.IMMEDIATE_THRESHOLD for p in recent_pes[-self.IMMEDIATE_WINDOW :]):
            return RollbackDecision(
                True, f"PES < {self.IMMEDIATE_THRESHOLD} for {self.IMMEDIATE_WINDOW} turns"
            )

        if n >= self.WARNING_WINDOW and all(
            p < self.WARNING_THRESHOLD for p in recent_pes[-self.WARNING_WINDOW :]
        ):
            return RollbackDecision(
                True, f"PES < {self.WARNING_THRESHOLD} for {self.WARNING_WINDOW} turns"
            )

        if n >= self.TREND_WINDOW:
            slope = float(np.polyfit(range(self.TREND_WINDOW), recent_pes[-self.TREND_WINDOW :], 1)[0])
            if slope < self.TREND_SLOPE:
                return RollbackDecision(True, "PES declining trend")

        return RollbackDecision(False)
