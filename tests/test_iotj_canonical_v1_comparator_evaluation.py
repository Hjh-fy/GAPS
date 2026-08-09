from scripts.evaluate_iotj_canonical_v1_comparators import method_contracts


def test_method_contracts_make_information_and_optimizer_regimes_explicit():
    contracts = method_contracts()
    assert list(contracts) == ["FedAvg", "FedProx", "SCAFFOLD", "MMD", "A0T", "GAPS/A4"]
    assert contracts["FedAvg"]["target_x"] is False
    assert contracts["FedAvg"]["target_y"] is False
    assert contracts["SCAFFOLD"]["optimizer"] == "SGD"
    assert contracts["SCAFFOLD"]["optimizer_lr"] == 5e-4
    assert contracts["MMD"]["target_x"] is True
    assert contracts["MMD"]["target_y"] is False
    assert contracts["A0T"]["target_y"] is True
    assert contracts["GAPS/A4"]["target_phase"] is True
    assert contracts["GAPS/A4"]["target_concentration"] is False
