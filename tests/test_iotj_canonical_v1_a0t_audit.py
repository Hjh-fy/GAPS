from scripts.audit_iotj_canonical_v1_a0t import audit_loss_activity


def test_a0t_loss_audit_accepts_only_active_target_ce():
    rows = [
        {"loss_name": "target_ce", "configured_weight": 1.0, "input_available": True, "active_steps": 100, "mean_weighted_loss": 0.2},
        {"loss_name": "global_mmd", "configured_weight": 0.0, "input_available": True, "active_steps": 0, "mean_weighted_loss": 0.0},
        {"loss_name": "align_reg_legacy", "configured_weight": 0.05, "input_available": False, "active_steps": 0, "mean_weighted_loss": 0.0},
    ]
    assert audit_loss_activity(rows)["status"] == "PASS"


def test_a0t_loss_audit_rejects_active_non_ce_loss():
    rows = [
        {"loss_name": "target_ce", "configured_weight": 1.0, "input_available": True, "active_steps": 100, "mean_weighted_loss": 0.2},
        {"loss_name": "global_mmd", "configured_weight": 0.0, "input_available": True, "active_steps": 1, "mean_weighted_loss": 0.1},
    ]
    assert audit_loss_activity(rows)["status"] == "FAIL"
