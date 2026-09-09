"""Hive Memory — the public facade of the context-curation system.

``strata/`` holds the flat packages (``cortex``, ``sieve``, ``membrane``,
``retention``, ``focal``, ``backend``, ``queen``, ``logs``) that the bench and
harness import by name. This module is the *one import* for system integrators:

    from strata import Hive, HiveConfig, UltraSmallDrone, LMStudioBackend

Nothing in ``strata/`` imports from ``hivebench/`` or ``harness/`` — the system
is self-contained and portable into other projects.
"""

from backend.lmstudio import LMStudioBackend
from backend.openai_compat import OpenAICompatBackend
from cortex.config import HiveConfig
from cortex.strata import Hive
from retention.store import ContextStore
from sieve.medium import MediumDrone
from sieve.ultra_small import UltraSmallDrone

__all__ = [
    "Hive",
    "HiveConfig",
    "UltraSmallDrone",
    "MediumDrone",
    "ContextStore",
    "LMStudioBackend",
    "OpenAICompatBackend",
]