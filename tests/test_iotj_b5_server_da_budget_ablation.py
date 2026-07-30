from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.freeze_iotj_confirmation_protocol import (
    ALGORITHM_CONFIG_FIELDS,
    canonical_sha256,
)
from scripts.prepare_iotj_b5_server_da_budget_ablation import build_level


ROOT = Path(__file__).resolve().parents[1]
BASE = (
    ROOT
    / "results/iotj_b5_local_epoch_ablation_20260729"
    / "le1/protocol_inputs"
)


def _build(tmp_path: Path, steps: int) -> Path:
    output = tmp_path / f"da{steps}"
    build_level(
        da_steps=steps,
        base_protocol=BASE / "confirmation_protocol_manifest.json",
        base_command_root=BASE / "commands",
        base_topology=BASE / "execution_topology_manifest.json",
        source_manifest=ROOT / "results/c2e_summary/source_archive_manifest.json",
        dataset_manifest=ROOT / "results/c2e_summary/dataset_manifest.json",
        source_archive=ROOT / "results/c2e/source/confirmation_source.tar",
        output_root=output,
    )
    return output


@pytest.mark.parametrize("steps", [80, 50, 30])
def test_only_server_da_step_budget_changes(tmp_path: Path, steps: int) -> None:
    output = _build(tmp_path, steps)
    base_path = BASE / "commands/c12_to_c5__b5__s42/command_manifest.json"
    new_path = output / "commands/c12_to_c5__b5__s42/command_manifest.json"
    base = json.loads(base_path.read_text(encoding="utf-8"))
    new = json.loads(new_path.read_text(encoding="utf-8"))

    assert new["training"] == base["training"]
    assert new["protocol"] == base["protocol"]
    assert new["causal_factors"] == base["causal_factors"]
    assert new["server_adaptation"] == {
        **base["server_adaptation"],
        "steps": steps,
    }
    assert new["commands"]["client_c1_pi"] == base["commands"]["client_c1_pi"]
    assert new["commands"]["client_c2_pc"] == base["commands"]["client_c2_pc"]
    command = new["commands"]["server_ecs"]
    position = command.index("--domain-adapt-steps")
    assert command[position + 1] == str(steps)
    assert new["algorithm_config_sha256"] == canonical_sha256(
        {field: new[field] for field in ALGORITHM_CONFIG_FIELDS}
    )

    derived = json.loads(
        (output / "derived_input_manifest.json").read_text(encoding="utf-8")
    )
    assert derived["local_epochs"] == 1
    assert derived["server_da_steps_per_round"] == steps
    assert derived["server_da_total_steps"] == 25 * steps
    assert derived["only_intended_variable"] == "server_adaptation.steps"


def test_refuses_existing_output(tmp_path: Path) -> None:
    output = _build(tmp_path, 80)
    with pytest.raises(FileExistsError, match="REFUSE_TO_OVERWRITE"):
        build_level(
            da_steps=80,
            base_protocol=BASE / "confirmation_protocol_manifest.json",
            base_command_root=BASE / "commands",
            base_topology=BASE / "execution_topology_manifest.json",
            source_manifest=ROOT
            / "results/c2e_summary/source_archive_manifest.json",
            dataset_manifest=ROOT / "results/c2e_summary/dataset_manifest.json",
            source_archive=ROOT / "results/c2e/source/confirmation_source.tar",
            output_root=output,
        )


def test_rejects_unregistered_step_level(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="da_steps must be one of"):
        build_level(
            da_steps=70,
            base_protocol=BASE / "confirmation_protocol_manifest.json",
            base_command_root=BASE / "commands",
            base_topology=BASE / "execution_topology_manifest.json",
            source_manifest=ROOT
            / "results/c2e_summary/source_archive_manifest.json",
            dataset_manifest=ROOT / "results/c2e_summary/dataset_manifest.json",
            source_archive=ROOT / "results/c2e/source/confirmation_source.tar",
            output_root=tmp_path / "da70",
        )
