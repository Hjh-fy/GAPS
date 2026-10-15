def test_source_aug_policy_accepts_explicit_all_gas_class_ids() -> None:
    from gaps_deploy.rich_residual import RichResidualPolicy

    policy = RichResidualPolicy(
        {
            "source_aug_target_ridge_policy": {
                "switch_rule": {"enabled_clients": ["C5"], "class_ids": [0, 1, 2, 3]}
            }
        }
    )

    assert policy.source_aug_class_ids == {0, 1, 2, 3}


def test_source_aug_policy_preserves_legacy_single_co_class() -> None:
    from gaps_deploy.rich_residual import RichResidualPolicy

    policy = RichResidualPolicy(
        {"source_aug_target_ridge_policy": {"switch_rule": {"class_id": 1}}}
    )

    assert policy.source_aug_class_ids == {1}
