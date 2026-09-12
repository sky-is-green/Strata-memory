"""Ensure the system (`strata/`) package root is importable so flat names like
`cortex`, `sieve`, `membrane` resolve from-source runs regardless of how pytest
is invoked (an editable install makes this unnecessary).

The evaluation suite and Studio sidecar live in the sibling hivebench repo —
their conftest handles its own roots."""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "strata"))
