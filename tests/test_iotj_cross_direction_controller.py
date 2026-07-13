import importlib
from pathlib import Path

import pytest

from scripts.generate_iotj_cross_direction_commands import generate_manifests


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "iotj_b2_b5_cross_direction_20260713.json"
RESULTS_ROOT = "results/iotj_b2_b5_cross_direction_20260713"


def _controller():
    return importlib.import_module("scripts.run_iotj_cross_direction_cloud_edge")


@pytest.fixture
def command_root(tmp_path: Path) -> Path:
    root = tmp_path / "commands"
    generate_manifests(
        CONFIG_PATH,
        root,
        repo_root=REPO_ROOT,
        results_root=RESULTS_ROOT,
        seeds=(42,),
    )
    return root


def test_seed42_queue_uses_approved_order(command_root: Path) -> None:
    controller = _controller()

    manifests = controller.load_ordered_manifests(command_root, 42)

    assert [(row[1]["direction_id"], row[1]["group_id"]) for row in manifests] == [
        ("F1_C1_TO_C5", "B2"),
        ("F1_C1_TO_C5", "B5"),
        ("R1_C5_TO_C1", "B2"),
        ("R1_C5_TO_C1", "B5"),
        ("R2_C45_TO_C1", "B2"),
        ("R2_C45_TO_C1", "B5"),
    ]


def test_active_executors_handles_pi_only_and_pi_pc(command_root: Path) -> None:
    controller = _controller()
    manifests = controller.load_ordered_manifests(command_root, 42)

    assert controller.active_executors(manifests[0][1]) == {
        "pi": (1,),
        "pc": (),
    }
    assert controller.active_executors(manifests[-1][1]) == {
        "pi": (4,),
        "pc": (5,),
    }


@pytest.mark.parametrize(
    ("running", "rounds", "has_files"),
    [(True, 0, False), (False, 3, True), (False, 0, True)],
)
def test_controller_refuses_partial_or_running_remote_run(
    running: bool, rounds: int, has_files: bool
) -> None:
    controller = _controller()

    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        controller.assert_remote_run_is_fresh(
            running=running,
            rounds=rounds,
            has_files=has_files,
            remote_run_dir="/root/GAPS/results/partial",
        )


def test_required_remote_data_keeps_test_rows_out_of_training_sync(
    command_root: Path,
) -> None:
    controller = _controller()
    manifests = controller.load_ordered_manifests(command_root, 42)

    pi = controller.required_data_files(manifests, "pi")
    pc = controller.required_data_files(manifests, "pc")
    ecs = controller.required_data_files(manifests, "ecs")

    forward_root = "client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid"
    reverse_root = "client_data_c2345src_c1tgt_2080_timeaware_60_170_window_fullgrid"
    assert f"client_1/train_features.npy" in pi[forward_root]
    assert f"client_5/train_features.npy" in pi[reverse_root]
    assert f"client_4/train_features.npy" in pi[reverse_root]
    assert set(pc) == {reverse_root}
    assert f"client_5/train_features.npy" in pc[reverse_root]
    assert f"client_4/calibration_features.npy" in ecs[reverse_root]
    assert f"client_5/calibration_features.npy" in ecs[reverse_root]
    assert f"client_1/calibration_features.npy" in ecs[reverse_root]
    assert not any("test_" in path for paths in ecs.values() for path in paths)


def _write_complete_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    for name in (
        "history.json",
        "run_config.json",
        "server_latest.pth",
        "server_latest_adapted.pth",
    ):
        (run_dir / name).write_text("{}", encoding="utf-8")
    for round_id in range(1, 26):
        (run_dir / f"client_stats_round_{round_id:03d}.json").write_text(
            "{}", encoding="utf-8"
        )
        (run_dir / f"domain_adapt_round_{round_id:03d}.json").write_text(
            "{}", encoding="utf-8"
        )


def test_recovered_artifact_audit_requires_all_25_rounds(tmp_path: Path) -> None:
    controller = _controller()
    run_dir = tmp_path / "complete"
    _write_complete_run(run_dir)

    audit = controller.audit_recovered_run(run_dir, expected_rounds=25)

    assert audit["client_stat_files"] == 25
    assert audit["domain_adapt_files"] == 25
    assert len(audit["checkpoint_sha256"]) == 64

    (run_dir / "domain_adapt_round_017.json").unlink()
    with pytest.raises(RuntimeError, match="domain adaptation files"):
        controller.audit_recovered_run(run_dir, expected_rounds=25)
