from scripts.analyze_iotj_canonical_v1_strict_nonoverlap import collapse_flags


def test_strict_collapse_rule_is_preregistered_and_symmetric():
    assert collapse_flags(0.99, 0.80, 10.0, 12.0) == {"classification": False, "regression": False}
    assert collapse_flags(0.99, 0.70, 10.0, 12.0)["classification"] is True
    assert collapse_flags(0.99, 0.98, 10.0, 20.0)["regression"] is True
