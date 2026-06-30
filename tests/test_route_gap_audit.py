from run_route_gap_audit import build_gap_rows, canonical_profile


def test_canonical_profile_maps_real_and_oracle_names_to_same_family():
    assert canonical_profile("H2.3 target direct-head") == "H2.3"
    assert canonical_profile("H2.3 oracle-route") == "H2.3"
    assert canonical_profile("H2.3+ oracle-route weak-blend") == "H2.3+"
    assert canonical_profile("H8 + formal C4 route rescue") == "H8+C4"
    assert canonical_profile("Oracle client selector C34 H2.3+ / C5 H8+C4") == "client_selector"


def test_build_gap_rows_aligns_by_profile_family_and_scope():
    real_rows = [
        {
            "profile": "H2.3 target direct-head",
            "scope": "C5",
            "N": "10",
            "full_RMSE": "40",
            "full_NRMSE": "0.32",
            "coverage_review_RMSE": "10",
            "coverage_review_NRMSE": "0.05",
        }
    ]
    oracle_rows = [
        {
            "profile": "H2.3 oracle-route",
            "scope": "C5",
            "N": "10",
            "full_RMSE": "13",
            "full_NRMSE": "0.06",
            "coverage_review_RMSE": "10",
            "coverage_review_NRMSE": "0.05",
        }
    ]

    rows = build_gap_rows(real_rows, oracle_rows)

    assert rows == [
        {
            "profile_family": "H2.3",
            "scope": "C5",
            "real_profile": "H2.3 target direct-head",
            "oracle_profile": "H2.3 oracle-route",
            "N": 10,
            "real_full_RMSE": 40.0,
            "oracle_full_RMSE": 13.0,
            "gap_full_RMSE": 27.0,
            "gap_full_RMSE_pct_of_real": 0.675,
            "real_full_NRMSE": 0.32,
            "oracle_full_NRMSE": 0.06,
            "gap_full_NRMSE": 0.26,
            "real_coverage_review_RMSE": 10.0,
            "oracle_coverage_review_RMSE": 10.0,
            "gap_coverage_review_RMSE": 0.0,
            "real_coverage_review_NRMSE": 0.05,
            "oracle_coverage_review_NRMSE": 0.05,
            "gap_coverage_review_NRMSE": 0.0,
        }
    ]
