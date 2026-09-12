"""Cross-repo location resolution (post-split).

After the HiveBench split, Strata Memory and HiveBench live in sibling
checkouts::

    <work>/strata-memory/    this repo (system + harness)
    <work>/hivebench/        evaluation suite: tests, testing, experiments,
                             fixtures, labels, experiment artifacts

HiveBench owns all fixtures, label sets, and experiment data. A few Strata CLI
entry points default to fixture paths that now live in the sibling checkout;
this package is the single place that resolution lives.

Override with ``$HIVEBENCH_HOME`` when the layout differs (CI, worktrees).
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def hivebench_root() -> Path:
    """Root of the HiveBench checkout."""
    env = os.environ.get("HIVEBENCH_HOME")
    if env:
        return Path(env).expanduser().resolve()
    # Default: sibling of this checkout (<parent>/hivebench).
    return REPO_ROOT.parent / "hivebench"


def generated_fixtures_dir() -> Path:
    """Generated conversation fixtures (``tests/fixtures/generated``)."""
    return hivebench_root() / "tests" / "fixtures" / "generated"


def labels_dir() -> Path:
    """Auditor label set (``tests/fixtures/labels``)."""
    return hivebench_root() / "tests" / "fixtures" / "labels"
