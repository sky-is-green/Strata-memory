"""Ensure the system (`strata/`), evaluation-suite (`hivebench/`) and harness
(`harness/`) package roots are importable so `cortex`, `sieve`, `tests`,
`experiments`, `harness`, ... resolve regardless of how pytest is invoked
(editable install makes this unnecessary, but from-source runs stay supported)."""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
for _pkg_root in (os.path.join(ROOT, "strata"), os.path.join(ROOT, "hivebench"),
                  os.path.join(ROOT, "harness")):
    sys.path.insert(0, _pkg_root)
# The vendored dsh Python SDK (deepseek_harness) — the agent bridge imports
# it, and a clean environment has no editable install to fall back on.
sys.path.insert(0, os.path.join(ROOT, "vendor"))