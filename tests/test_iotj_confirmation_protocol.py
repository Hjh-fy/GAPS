from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import pytest

import scripts.freeze_iotj_confirmation_protocol as freeze
from scripts.freeze_iotj_confirmation_protocol import (
    CONFIRMATION_SCHEDULE,
    build_algorithm_manifest,
    build_dataset_manifest,
    build_protocol_manifest,
    canonical_sha256,
    confirmation_run_id,
    create_source_archive,
    sha256_file,
)


SPLIT_COMPONENTS = (
    "features",
    "classification_labels",
    "regression_labels",
)


def _write_split(directory: Path, split: str, rows: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / f"{split}_features.npy", np.arange(rows, dtype=np.int32).reshape(rows, 1))
    np.save(directory / f"{split}_classification_labels.npy", np.zeros(rows, dtype=np.int64))
    np.save(directory / f"{split}_regression_labels.npy", np.zeros((rows, 3), dtype=np.float32))


def _write_dataset(root: Path, *, c5_test_rows: int = 1360) -> Path:
    root.mkdir(parents=True)
    (root / "split_info.json").write_text(
        json.dumps(
            {
                "protocol": "c12_to_c5",
                "source_clients": [1, 2],
                "target_clients": [5],
                "seed": 42,
                "target_split": {"train_used": False, "calibration": 0.2, "test": 0.8},
                "stratify_by": ["client", "class", "concentration"],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "norm_stats.npz").write_bytes(b"frozen-normalization")
    for client_id in (1, 2):
        for split in ("train", "calibration", "test"):
            _write_split(root / f"client_{client_id}", split, 3)
    _write_split(root / "client_5", "calibration", 320)
    _write_split(root / "client_5", "test", c5_test_rows)
    return root


def _expected_dataset_paths() -> list[str]:
    paths = ["norm_stats.npz", "split_info.json"]
    for client_id in (1, 2):
        for split in ("train", "calibration", "test"):
            for component in SPLIT_COMPONENTS:
                paths.append(f"client_{client_id}/{split}_{component}.npy")
    for split in ("calibration", "test"):
        for component in SPLIT_COMPONENTS:
            paths.append(f"client_5/{split}_{component}.npy")
    return sorted(paths)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _init_git_repo(repo: Path) -> str:
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "tests@example.invalid")
    _git(repo, "config", "user.name", "Protocol Tests")
    (repo / "tracked.txt").write_text("tracked source\n", encoding="utf-8")
    (repo / "scripts").mkdir()
    (repo / "scripts" / "entry.py").write_text("print('tracked')\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt", "scripts/entry.py")
    _git(repo, "commit", "-m", "fixture")
    return _git(repo, "rev-parse", "HEAD")


def test_confirmation_schedule_is_exact_and_alternating() -> None:
    assert CONFIRMATION_SCHEDULE == (
        ("B2", 42), ("B5", 42),
        ("B5", 43), ("B2", 43),
        ("B2", 44), ("B5", 44),
        ("B5", 45), ("B2", 45),
        ("B2", 46), ("B5", 46),
    )
    assert [confirmation_run_id(group, seed) for group, seed in CONFIRMATION_SCHEDULE] == [
        "c12_to_c5__b2__s42", "c12_to_c5__b5__s42",
        "c12_to_c5__b5__s43", "c12_to_c5__b2__s43",
        "c12_to_c5__b2__s44", "c12_to_c5__b5__s44",
        "c12_to_c5__b5__s45", "c12_to_c5__b2__s45",
        "c12_to_c5__b2__s46", "c12_to_c5__b5__s46",
    ]


@pytest.mark.parametrize(
    ("group_id", "seed"),
    [("B1", 42), ("A6", 42), ("B2", 41), ("B5", 47)],
)
def test_confirmation_run_id_rejects_every_non_allowlisted_identity(
    group_id: str,
    seed: int,
) -> None:
    with pytest.raises(ValueError):
        confirmation_run_id(group_id, seed)


def test_dataset_manifest_hashes_only_exact_active_inputs(tmp_path: Path) -> None:
    data_root = _write_dataset(tmp_path / "data")
    _write_split(data_root / "client_3", "calibration", 7)
    _write_split(data_root / "client_4", "test", 9)

    first = build_dataset_manifest(data_root)
    second = build_dataset_manifest(data_root)

    assert first == second
    assert first["active_source_clients"] == ["C1", "C2"]
    assert first["active_target_clients"] == ["C5"]
    assert first["inactive_shared_dataset_clients"] == ["C3", "C4"]
    assert first["sample_counts"]["C5"] == {"calibration": 320, "test": 1360}
    assert [entry["relative_path"] for entry in first["files"]] == _expected_dataset_paths()
    assert all(set(entry) == {"relative_path", "byte_size", "sha256"} for entry in first["files"])
    for entry in first["files"]:
        path = data_root / entry["relative_path"]
        assert entry["byte_size"] == path.stat().st_size
        assert entry["sha256"] == sha256_file(path)
    stable_payload = {key: value for key, value in first.items() if key != "dataset_manifest_sha256"}
    assert first["dataset_manifest_sha256"] == canonical_sha256(stable_payload)
    assert all("client_3/" not in entry["relative_path"] for entry in first["files"])
    assert all("client_4/" not in entry["relative_path"] for entry in first["files"])


@pytest.mark.parametrize("failure", ["missing", "wrong_c5_count", "metadata_direction"])
def test_dataset_manifest_fails_closed_on_invalid_active_inputs(
    tmp_path: Path,
    failure: str,
) -> None:
    data_root = _write_dataset(
        tmp_path / "data",
        c5_test_rows=1359 if failure == "wrong_c5_count" else 1360,
    )
    if failure == "missing":
        (data_root / "client_2" / "train_features.npy").unlink()
    elif failure == "metadata_direction":
        info = json.loads((data_root / "split_info.json").read_text(encoding="utf-8"))
        info["target_clients"] = [3, 4, 5]
        (data_root / "split_info.json").write_text(json.dumps(info), encoding="utf-8")

    output = tmp_path / "dataset_manifest.json"
    with pytest.raises((FileNotFoundError, ValueError)):
        freeze.write_dataset_manifest(data_root, output)
    assert not output.exists()


def test_canonical_sha256_is_order_independent_and_compact() -> None:
    left = {"z": [3, 2, 1], "a": {"y": False, "x": 1}}
    right = {"a": {"x": 1, "y": False}, "z": [3, 2, 1]}
    expected = hashlib.sha256(
        json.dumps(left, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    assert canonical_sha256(left) == canonical_sha256(right) == expected


def test_algorithm_hash_selects_only_frozen_algorithm_sections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mutable = {
        "protocol": {"source_clients": [1, 2], "target_clients": [5], "training_seed": 43},
        "training": {"rounds": 25, "profile": "proto_replay"},
        "causal_factors": {"server_stage_mmd": True},
        "server_adaptation": {"enabled": True, "lambda_stage_mmd": 0.2},
        "commands": {"server_ecs": ["python", "--observer-output", "host-a.jsonl"]},
        "topology": {"server": "host-a"},
        "provenance": {"code_revision": "a" * 40},
        "output_dir": "/host/a/results",
    }

    def fake_build_run_manifest(*_args: object, **_kwargs: object) -> dict[str, object]:
        return copy.deepcopy(mutable)

    monkeypatch.setattr(freeze, "build_run_manifest", fake_build_run_manifest)
    first = build_algorithm_manifest(tmp_path, "B5", 43)
    mutable["commands"] = {"server_ecs": ["different", "observer", "cli"]}
    mutable["topology"] = {"server": "host-b"}
    mutable["provenance"] = {"code_revision": "b" * 40}
    mutable["output_dir"] = "/host/b/other-attempt"
    second = build_algorithm_manifest(tmp_path, "B5", 43)

    assert first == second
    assert list(first["algorithm_config"]) == [
        "protocol",
        "training",
        "causal_factors",
        "server_adaptation",
    ]
    assert set(first) == {"group_id", "seed", "algorithm_config", "algorithm_config_sha256"}
    assert first["algorithm_config_sha256"] == canonical_sha256(first["algorithm_config"])


@pytest.mark.parametrize(("group_id", "seed"), [("B1", 42), ("B2", 41), ("B5", 47)])
def test_algorithm_manifest_rejects_non_confirmation_identites(
    tmp_path: Path,
    group_id: str,
    seed: int,
) -> None:
    with pytest.raises(ValueError):
        build_algorithm_manifest(tmp_path, group_id, seed)


def test_source_archive_uses_clean_tracked_head_once_and_excludes_untracked_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    commit = _init_git_repo(repo)
    (repo / "dataset" / "client_5").mkdir(parents=True)
    (repo / "dataset" / "client_5" / "test_labels.npy").write_bytes(b"untracked secret")
    (repo / "junction").mkdir()
    (repo / "junction" / "outside.txt").write_text("untracked junction fixture", encoding="utf-8")
    output = tmp_path / "archive" / "confirmation.tar"
    archive_calls: list[list[str]] = []
    real_run = subprocess.run

    def run_spy(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        if command[:3] == ["git", "archive", "--format=tar"]:
            archive_calls.append(command)
        return real_run(command, **kwargs)

    monkeypatch.setattr(freeze.subprocess, "run", run_spy)
    manifest = create_source_archive(repo, commit, output)

    assert archive_calls == [[
        "git", "archive", "--format=tar", "--output", str(output), commit,
    ]]
    assert manifest["confirmation_commit"] == commit
    assert manifest["source_archive_sha256"] == sha256_file(output)
    assert manifest["dependency_versions"] == {
        "flwr": "1.23.0",
        "protobuf": "4.25.8",
        "psutil": "7.0.0",
    }
    assert [item["relative_path"] for item in manifest["regular_members"]] == [
        "scripts/entry.py",
        "tracked.txt",
    ]
    assert manifest["regular_members_sha256"] == canonical_sha256(
        {"regular_members": manifest["regular_members"]}
    )
    with tarfile.open(output, "r:") as archive:
        assert "dataset/client_5/test_labels.npy" not in archive.getnames()
        assert "junction/outside.txt" not in archive.getnames()


def test_source_archive_refuses_wrong_head_dirty_tree_and_existing_output(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    commit = _init_git_repo(repo)
    output = tmp_path / "confirmation.tar"

    with pytest.raises(ValueError, match="HEAD"):
        create_source_archive(repo, "0" * 40, output)
    assert not output.exists()

    (repo / "tracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="clean"):
        create_source_archive(repo, commit, output)
    assert not output.exists()

    (repo / "tracked.txt").write_text("tracked source\n", encoding="utf-8")
    output.write_bytes(b"do-not-overwrite")
    with pytest.raises(FileExistsError):
        create_source_archive(repo, commit, output)
    assert output.read_bytes() == b"do-not-overwrite"


def test_protocol_builds_ten_attempt_independent_command_manifests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = {
        "dataset_manifest_sha256": "d" * 64,
        "sample_counts": {"C5": {"calibration": 320, "test": 1360}},
    }
    source = {
        "source_archive_sha256": "s" * 64,
        "regular_members_sha256": "t" * 64,
    }

    monkeypatch.setattr(freeze, "build_dataset_manifest", lambda _root: copy.deepcopy(dataset))
    monkeypatch.setattr(
        freeze,
        "create_source_archive",
        lambda _repo, _commit, _archive: copy.deepcopy(source),
    )

    def fake_run_manifest(
        group_id: str,
        seed: int,
        *,
        repo_root: Path,
        results_root: str,
    ) -> dict[str, object]:
        assert repo_root == tmp_path
        return {
            "group_id": group_id,
            "protocol": {"source_clients": [1, 2], "target_clients": [5], "training_seed": seed},
            "training": {"rounds": 25},
            "causal_factors": {"full": group_id == "B5"},
            "server_adaptation": {"enabled": True},
            "commands": {
                "server_ecs": ["server", group_id, str(seed), results_root],
                "client_c1_pi": ["client", "1", str(seed)],
                "client_c2_pc": ["client", "2", str(seed)],
            },
            "topology": {"server": "existing-host-field"},
        }

    monkeypatch.setattr(freeze, "build_run_manifest", fake_run_manifest)
    protocol = build_protocol_manifest(
        tmp_path,
        tmp_path / "dataset",
        "a" * 40,
        tmp_path / "confirmation.tar",
    )

    command_manifests = protocol["command_manifests"]
    assert [item["run_id"] for item in command_manifests] == [
        confirmation_run_id(group, seed) for group, seed in CONFIRMATION_SCHEDULE
    ]
    assert protocol["historical_seed42_included"] is False
    assert "feaa75b" not in json.dumps(protocol)
    assert len(command_manifests) == 10
    for item, (group_id, seed) in zip(command_manifests, CONFIRMATION_SCHEDULE):
        assert item["commands"] == fake_run_manifest(
            group_id,
            seed,
            repo_root=tmp_path,
            results_root=freeze.DEFAULT_RESULTS_ROOT,
        )["commands"]
        assert item["historical_seed42_included"] is False
        assert item["transport_status"] == "not_collected"
        assert item["protocol_manifest_sha256"] == protocol["protocol_manifest_sha256"]
        assert item["dataset_manifest_sha256"] == "d" * 64
        assert item["source_archive_sha256"] == "s" * 64
        assert item["algorithm_config_sha256"] == canonical_sha256(
            {
                "protocol": item["protocol"],
                "training": item["training"],
                "causal_factors": item["causal_factors"],
                "server_adaptation": item["server_adaptation"],
            }
        )
        assert "controller-local" in item["observer_cli_scope"]
        assert "excluded from algorithm config" in item["observer_cli_scope"]
        if group_id == "B2":
            assert item["b2_claim_status"] == "post_screen_exploratory"
            assert "b5_claim_status" not in item
        else:
            assert item["b5_claim_status"] == "predeclared_full_method"
            assert "b2_claim_status" not in item


def test_main_writes_exact_summary_and_command_manifest_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = {
        "schema_version": 1,
        "protocol_manifest_sha256": "p" * 64,
        "source_archive_manifest": {"source_archive_sha256": "s" * 64},
        "dataset_manifest": {
            "dataset_manifest_sha256": "d" * 64,
            "sample_counts": {"C5": {"calibration": 320, "test": 1360}},
        },
        "command_manifests": [
            {"run_id": confirmation_run_id(group, seed), "group_id": group, "seed": seed}
            for group, seed in CONFIRMATION_SCHEDULE
        ],
    }
    monkeypatch.setattr(
        freeze,
        "build_protocol_manifest",
        lambda *_args, **_kwargs: copy.deepcopy(protocol),
    )
    archive_output = tmp_path / "source" / "confirmation.tar"
    command_root = tmp_path / "commands"
    summary_root = tmp_path / "summary"

    assert freeze.main(
        [
            "--confirmation-commit", "a" * 40,
            "--data-root", str(tmp_path / "dataset"),
            "--archive-output", str(archive_output),
            "--command-root", str(command_root),
            "--summary-root", str(summary_root),
        ]
    ) == 0

    assert sorted(path.name for path in summary_root.iterdir()) == [
        "confirmation_protocol_manifest.json",
        "dataset_manifest.json",
        "source_archive_manifest.json",
    ]
    assert sorted(path.name for path in command_root.iterdir()) == sorted(
        confirmation_run_id(group, seed) for group, seed in CONFIRMATION_SCHEDULE
    )
    for group, seed in CONFIRMATION_SCHEDULE:
        run_id = confirmation_run_id(group, seed)
        assert sorted(path.name for path in (command_root / run_id).iterdir()) == [
            "command_manifest.json"
        ]
    written_protocol = json.loads(
        (summary_root / "confirmation_protocol_manifest.json").read_text(encoding="utf-8")
    )
    assert "source_archive_manifest" not in written_protocol
    assert "dataset_manifest" not in written_protocol
    assert "command_manifests" not in written_protocol
