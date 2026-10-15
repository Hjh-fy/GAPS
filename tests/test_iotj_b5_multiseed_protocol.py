from __future__ import annotations

import copy
import json
from pathlib import Path

from scripts.validate_iotj_b5_multiseed_protocol import (
    canonical_algorithm_payload,
    validate_protocol,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/iotj_b5_multiseed_20260724/protocol_manifest.json"


def test_frozen_m0_protocol_is_ready_for_live_preflight():
    result = validate_protocol(ROOT, MANIFEST)
    assert result["status"] == "ready_for_preflight"
    assert result["errors"] == []
    assert result["seed42_retrained"] is False
    assert result["formal_training_started"] is False


def test_all_b5_command_manifests_differ_only_by_seed_and_derived_identity():
    payloads = {
        seed: json.loads(
            (
                ROOT
                / f"results/c2e_commands/c12_to_c5__b5__s{seed}/command_manifest.json"
            ).read_text(encoding="utf-8")
        )
        for seed in range(42, 47)
    }
    reference = canonical_algorithm_payload(payloads[42], 42)
    for seed in range(43, 47):
        assert canonical_algorithm_payload(payloads[seed], seed) == reference


def test_non_seed_algorithm_drift_is_detected_by_canonical_comparison():
    source = json.loads(
        (
            ROOT
            / "results/c2e_commands/c12_to_c5__b5__s43/command_manifest.json"
        ).read_text(encoding="utf-8")
    )
    drifted = copy.deepcopy(source)
    drifted["training"]["batch_size"] = 64
    assert canonical_algorithm_payload(source, 43) != canonical_algorithm_payload(
        drifted, 43
    )
