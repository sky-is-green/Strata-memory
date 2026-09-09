"""Unit tests for cortex.config (HiveConfig)."""

from cortex.config import HiveConfig


def test_defaults_roundtrip(tmp_path):
    c = HiveConfig()
    path = tmp_path / "strata.json"
    c.save(path)
    loaded = HiveConfig.load(path)
    assert loaded == c


def test_from_dict_ignores_unknown_keys():
    c = HiveConfig.from_dict({"decay_multiplier_init": 2.2, "bogus": 1, "nope": None})
    assert c.decay_multiplier_init == 2.2
    assert not hasattr(c, "bogus")
    assert c.max_context == HiveConfig().max_context  # defaults preserved


def test_apply_gatekeeper_overrides():
    c = HiveConfig().apply_gatekeeper_overrides({"decay_multiplier_init": 2.5})
    assert c.decay_multiplier_init == 2.5
    assert c.max_context == HiveConfig().max_context


def test_apply_gatekeeper_overrides_ignores_none():
    c = HiveConfig().apply_gatekeeper_overrides({"decay_multiplier_init": None, "drift_threshold": 0.7})
    assert c.decay_multiplier_init == HiveConfig().decay_multiplier_init
    assert c.drift_threshold == 0.7


def test_to_dict_has_all_fields():
    d = HiveConfig().to_dict()
    assert d["decay_multiplier_init"] == 1.8
    assert "budget_ranges" in d