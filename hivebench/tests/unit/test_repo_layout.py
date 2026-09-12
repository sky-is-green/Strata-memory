"""Guards for the repo layout.

The repo is split into `strata/` (the system), `hivebench/` (the evaluation
suite) and `harness/` (the HiveBench Studio sidecar) with flat top-level
import names. These tests pin the invariants the restructure depends on: no
stray package dirs at the root, pyproject declaring all three trees, the vocab
data present next to `sieve`, and the test runner's path constants resolving.
"""

import tomllib
from pathlib import Path

from setuptools import find_packages

from sieve.vocabulary import Vocabulary

ROOT = Path(__file__).resolve().parents[3]
HIVE = ROOT / "strata"
HIVEBENCH = ROOT / "hivebench"
HARNESS = ROOT / "harness"

PACKAGE_ROOTS = ("strata", "hivebench", "harness")

EXPECTED_PACKAGES = {
    "backend",
    "cortex",
    "experiments",
    "focal",
    "logs",
    "membrane",
    "auditor",
    "retention",
    "sieve",
    "testing",
    "tests",
}


# Standalone root-level entrypoint/utility scripts that are not importable
# package modules (conftest + developer tools). The guard still fails on any
# *new* root module, so package roots cannot silently leak into the top level.
KNOWN_ROOT_SCRIPTS = {
    "conftest.py",
    "import_conversation.py",
    "install_cpu_torch.py",
    "launch_detached.py",
    "longrun.py",
    "longtest.py",
    "strataclient.py",
}


def test_root_python_files_are_only_known_scripts():
    assert {p.name for p in ROOT.glob("*.py")} <= KNOWN_ROOT_SCRIPTS


def test_no_stray_package_dirs_at_root():
    stray = []
    for entry in sorted(ROOT.iterdir()):
        if entry.is_dir() and entry.name not in PACKAGE_ROOTS:
            if list(entry.glob("*.py")):
                stray.append(entry.name)
    assert stray == []


def test_pyproject_declares_all_trees():
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    where = cfg["tool"]["setuptools"]["packages"]["find"]["where"]
    assert where == ["strata", "hivebench", "harness"]


def test_find_packages_resolves_flat_names():
    found = set()
    for root in PACKAGE_ROOTS:
        found |= set(find_packages(where=root))
    assert EXPECTED_PACKAGES <= found
    assert "cortex" in found and "strata.cortex" not in found
    assert "harness" in found


def test_vocab_data_present_and_loadable():
    vocab = HIVE / "vocab"
    assert (vocab / "code.json").is_file()
    assert (vocab / "general.json").is_file()
    vocab_obj = Vocabulary.load("code", "general")
    assert vocab_obj.size > 0


def test_runner_path_constants_resolve():
    import tests.run_hive_tests as runner

    assert runner.ROOT == ROOT
    assert runner.HIVE == HIVE
    assert runner.HIVEBENCH == HIVEBENCH
    assert runner.TESTS == HIVEBENCH / "tests"
    for rel in runner.INTELLIGENCE:
        assert (ROOT / rel).is_file(), f"intelligence file missing: {rel}"