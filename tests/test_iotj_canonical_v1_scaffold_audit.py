from scripts.audit_iotj_canonical_v1_scaffold import audit_history


def test_scaffold_audit_requires_roundwise_control_updates_and_no_adam():
    rounds = []
    for index in range(1, 26):
        rounds.append({
            "round": index, "fit_clients": 2, "fit_failures": 0,
            "evaluate_clients": 2, "evaluate_failures": 0,
            "fit_metrics": {"local_epochs": 1.0, "scaffold_adam_state_present": 0.0, "scaffold_local_steps": 74.0, "scaffold_optimizer_lr": 5e-4, "train_ce_mean": 1.4-index/100, "train_accuracy": index/30},
            "evaluate_loss": 1.5-index/100, "evaluate_metrics": {"accuracy": index/30},
            "scaffold": {"server_control_fingerprint": f"fp-{index}", "server_control_rounds_completed": index, "optimizer": "SGD", "optimizer_lr": 5e-4},
        })
    result = audit_history({"rounds": rounds}, {"passed": True})
    assert result["status"] == "PASS"
    assert result["checks"]["server_control_updates"] is True
    assert result["checks"]["no_adam_state"] is True
