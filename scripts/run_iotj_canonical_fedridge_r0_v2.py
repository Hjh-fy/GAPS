"""Fail-closed controller for the frozen canonical FedRidge R0-v2 study."""

from __future__ import annotations

import argparse
import contextlib
import csv
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Protocol, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gaps_flower.canonical_quantitative_features import (  # noqa: E402
    H1_FEATURE_NAMES,
    SENSOR_FEATURE_NAMES,
    build_feature_cache,
    load_feature_cache,
    sha256_file,
    sha256_strings,
)
from gaps_flower.canonical_fedridge_v2 import (  # noqa: E402
    RIDGE_ALPHAS,
    aggregate_normal_equations_v2,
    decide_gas_equivalence_v2,
    decide_r0_v2,
    feature_numerical_audit_rows,
    functional_diagnostics_v2,
    local_central_moments,
    local_normal_equations_v2,
    merge_central_moments,
    normal_equation_diagnostics_v2,
    pooled_reference_fit_v2,
    registered_tolerances_v2,
    reconstruct_ridge_v2,
    scaler_diagnostics_v2,
    select_pooled_alpha_v2,
    select_source_alpha_v2,
    system_diagnostics_v2,
)


R0_V2_STUDY_ID = "CAN-V1-FEDRIDGE-R0V2-20260812"
SCHEMA_VERSION = "iotj.canonical_v1.fedridge_r0_v2.execution.v1"
DATA_ROOT = ROOT / "dataset" / "iotj_canonical_v1"
PROTOCOL_ROOT = (
    ROOT
    / "docs"
    / "experiments"
    / "iotj_canonical_v1_final"
    / "canonical_fedridge_r0_v2_20260812"
)
PROTOCOL_MANIFEST = PROTOCOL_ROOT / "protocol_manifest.json"
EXPECTED_PROTOCOL_FREEZE_SHA256 = (
    "6693761f4b660cf11f01a2b41148ffd2c031293b3e23d2e0163f18cd2b0451d6"
)
RESULT_ROOT = (
    ROOT
    / "results"
    / "iotj_canonical_v1_final"
    / "canonical_fedridge_r0_v2_20260812"
)
ORIGINAL_ROOT = (
    ROOT
    / "results"
    / "iotj_canonical_v1_final"
    / "canonical_regression_reconstruction_qc_20260811"
)
ORIGINAL_C0_ROOT = ORIGINAL_ROOT / "C0"
ORIGINAL_R0_ROOT = ORIGINAL_ROOT / "R0"
ORIGINAL_PROTOCOL_MANIFEST = (
    ROOT
    / "docs"
    / "experiments"
    / "iotj_canonical_v1_final"
    / "canonical_regression_reconstruction_qc_20260811"
    / "protocol_manifest.json"
)
EXTRACTOR_PATH = ROOT / "run_regression_head_ablation.py"
SOURCE_CLIENTS = ("C1", "C2")
SOURCE_SPLITS = ("train", "calibration", "test")
GAS_IDS = (0, 1, 2, 3)
R0_V2_PASS = "FEDRIDGE_ALGEBRAIC_EXACT_NUMERICAL_EQUIVALENCE_ESTABLISHED"
R0_V2_FAIL = "R0_V2_FAILED"
DECISION_VOCABULARY = (R0_V2_PASS, R0_V2_FAIL)
COMPLETION_RELEASE_FIELD = "R1_" + "released"
EXPECTED_FORMAL_FILES = (
    "canonical_feature_caches",
    "H1_CANONICAL_FEATURE_NUMERICAL_AUDIT.csv",
    "r0_v2_scaler_diagnostics.csv",
    "r0_v2_normal_equation_diagnostics.csv",
    "r0_v2_system_diagnostics.csv",
    "r0_v2_functional_equivalence.csv",
    "source_alpha_audit.csv",
    "source_alpha_lock.json",
    "model_lock.json",
    "DATA_ACCESS_AUDIT.md",
    "R0_V2_DECISION.json",
    "R0_V2_EXPERIMENT_AUDIT.md",
    "protocol_manifest_execution.json",
    "sha256_index.json",
)


@dataclass(frozen=True)
class SourceRequest:
    client: str
    split: str
    gas_id: int | None


class SourceDataProvider(Protocol):
    def build_fresh_cache(self, client: str, split: str) -> Mapping[str, Any]: ...

    def gas_data(
        self, client: str, split: str, gas_id: int
    ) -> tuple[np.ndarray, np.ndarray]: ...


def build_r0_v2_execution_plan() -> list[str]:
    """Return the source-only execution order frozen by the protocol."""
    return [
        "verify_design_freeze_and_canonical_dataset",
        "verify_original_C0_and_R0_immutable",
        "build_fresh_source_train_calibration_caches",
        "compute_source_only_alpha_and_models",
        "write_source_alpha_and_model_locks",
        "open_source_test",
        "evaluate_source_functional_equivalence",
        "write_decision_audit_and_hash_index",
        "stop",
    ]


def protocol_freeze_hash() -> str:
    """Hash the machine-readable freeze without modifying it."""
    return sha256_file(PROTOCOL_MANIFEST)


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _git_file_bytes(commit: str, path: Path) -> bytes:
    relative = Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
    completed = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return completed.stdout


def _verify_critical_paths_match_head(head: str) -> None:
    critical_paths = (
        Path(__file__).resolve(),
        ROOT / "gaps_flower/canonical_fedridge_v2.py",
        ROOT / "gaps_flower/canonical_quantitative_features.py",
        PROTOCOL_MANIFEST,
    )
    for path in critical_paths:
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(
                f"FAIL_CLOSED execution-critical path type changed: {path.name}"
            )
        try:
            committed = _git_file_bytes(head, path)
        except (OSError, subprocess.CalledProcessError, ValueError) as error:
            raise RuntimeError(
                f"FAIL_CLOSED cannot verify critical path at HEAD: {path.name}"
            ) from error
        if path.read_bytes() != committed:
            raise RuntimeError(
                f"FAIL_CLOSED execution-critical path differs from HEAD: {path.name}"
            )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"FAIL_CLOSED cannot read required JSON: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"FAIL_CLOSED required JSON is not an object: {path}")
    return value


def _is_descendant(path: Path, ancestor: Path) -> bool:
    try:
        path.resolve().relative_to(ancestor.resolve())
    except ValueError:
        return False
    return True


def _paths_overlap(left: Path, right: Path) -> bool:
    return _is_descendant(left, right) or _is_descendant(right, left)


def _reject_symlink_components(path: Path, label: str) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path)))
    current = lexical
    while True:
        if current.is_symlink():
            raise RuntimeError(
                f"FAIL_CLOSED symlink in {label} path: {current}"
            )
        parent = current.parent
        if parent == current:
            break
        current = parent
    return lexical


def _require_absent_output(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"FAIL_CLOSED output already exists: {output}")


def _verify_protocol() -> dict[str, Any]:
    observed_hash = protocol_freeze_hash()
    if observed_hash != EXPECTED_PROTOCOL_FREEZE_SHA256:
        raise RuntimeError(
            "FAIL_CLOSED frozen protocol hash changed: "
            f"expected {EXPECTED_PROTOCOL_FREEZE_SHA256}, observed {observed_hash}"
        )
    protocol = _read_json(PROTOCOL_MANIFEST)
    valid = (
        protocol.get("study_id") == R0_V2_STUDY_ID
        and protocol.get("status") == "DESIGN_FREEZE_READY_FORMAL_NOT_STARTED"
        and protocol.get("formal_execution_started") is False
        and protocol.get("source_clients") == ["C1", "C2"]
        and protocol.get("target_clients") == []
        and protocol.get("target_access")
        == {
            "calibration_x": False,
            "calibration_labels": False,
            "test_x": False,
            "test_labels": False,
        }
        and protocol.get("C0_decision") == "V1_INTERLEAVED_RETAINED"
        and protocol.get("original_R0_decision")
        == "R0_EXACT_RECOVERY_NOT_ESTABLISHED"
        and protocol.get("immutable_prerequisites")
        == {
            "C0": {
                "index_path": (
                    "results/iotj_canonical_v1_final/"
                    "canonical_regression_reconstruction_qc_20260811/"
                    "C0/C0_SHA256_INDEX.json"
                ),
                "index_sha256": (
                    "18d6fa01352be80273460439e6c3a77196d8d93df53e3ea967f0e9ebdf335da0"
                ),
                "decision": "V1_INTERLEAVED_RETAINED",
            },
            "original_R0": {
                "index_path": (
                    "results/iotj_canonical_v1_final/"
                    "canonical_regression_reconstruction_qc_20260811/"
                    "R0/R0_SHA256_INDEX.json"
                ),
                "index_sha256": (
                    "0f9a4ed854df5b87acad2d6801fa1e5607ac8df58d6e21e5138b6e1401bfc242"
                ),
                "decision": "R0_EXACT_RECOVERY_NOT_ESTABLISHED",
                "status": "FAIL_CLOSED",
                "failed_gate": "R0.4_CANONICAL_FEDRIDGE_EXACT_RECOVERY",
            },
        }
    )
    if not valid:
        raise RuntimeError("FAIL_CLOSED frozen protocol status or role contract changed")
    expected_roles = {
        "source_feature_fit": ["C1_train", "C2_train"],
        "source_alpha_selection": ["C1_calibration", "C2_calibration"],
        "source_refit": [
            "C1_train",
            "C1_calibration",
            "C2_train",
            "C2_calibration",
        ],
        "source_functional_evaluation_after_lock": ["C1_test", "C2_test"],
    }
    expected_access = [
        "verify_design_freeze_and_canonical_dataset",
        "verify_original_C0_and_R0_immutable",
        "create_fresh_C1_C2_train_calibration_feature_caches",
        "write_H1_numerical_audit",
        "perform_source_only_alpha_selection",
        "refit_source_train_calibration_federated_and_pooled_models",
        "write_source_alpha_and_model_locks",
        "open_C1_C2_source_test_after_locks",
        "evaluate_functional_equivalence",
        "write_diagnostics_decision_audit_and_SHA256_index",
        "stop",
    ]
    numerical = protocol.get("numerical_protocol", {})
    normalized_expected_files = [
        str(value).rstrip("/") for value in protocol.get("expected_formal_files", [])
    ]
    if (
        numerical.get("aggregation_order") != list(SOURCE_CLIENTS)
        or tuple(numerical.get("alpha_grid", ())) != RIDGE_ALPHAS
        or protocol.get("canonical_split_roles") != expected_roles
        or protocol.get("access_sequence") != expected_access
        or normalized_expected_files != list(EXPECTED_FORMAL_FILES)
        or protocol.get("decision_vocabulary") != list(DECISION_VOCABULARY)
    ):
        raise RuntimeError("FAIL_CLOSED frozen source order/execution contract changed")
    return protocol


def _source_artifact_paths(data_root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for client in SOURCE_CLIENTS:
        directory = data_root / f"client_{client[1:]}"
        for split in SOURCE_SPLITS:
            paths[f"{client}_{split}_features"] = (
                directory / f"{split}_features.npy"
            )
            paths[f"{client}_{split}_phase"] = (
                directory / f"{split}_phase_labels.npy"
            )
            paths[f"{client}_{split}_metadata"] = (
                directory / f"{split}_experiment_info.json"
            )
            for label_kind in ("classification", "regression"):
                paths[f"{client}_{split}_{label_kind}_labels"] = (
                    directory / f"{split}_{label_kind}_labels.npy"
                )
        paths[f"{client}_stats"] = directory / "stats.json"
    return paths


def _verify_source_dataset(data_root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    data_root = _reject_symlink_components(data_root, "canonical data").resolve()
    canonical = protocol["canonical_data"]
    expected_aggregate = str(canonical["dataset_aggregate_sha256"])
    index_path = data_root / "dataset_sha256.json"
    expected_index_sha256 = canonical.get("dataset_sha256_json_sha256")
    if (
        index_path.is_symlink()
        or not index_path.is_file()
        or not isinstance(expected_index_sha256, str)
        or sha256_file(index_path) != expected_index_sha256
    ):
        raise RuntimeError("FAIL_CLOSED canonical dataset index hash changed")
    dataset_index = _read_json(index_path)
    indexed_files = dataset_index.get("files")
    if (
        dataset_index.get("schema_version") != "iotj.canonical_v1.sha256"
        or dataset_index.get("aggregate_sha256") != expected_aggregate
        or not isinstance(indexed_files, Mapping)
        or any(
            not isinstance(relative, str) or not isinstance(digest, str)
            for relative, digest in (
                indexed_files.items() if isinstance(indexed_files, Mapping) else ()
            )
        )
    ):
        raise RuntimeError(
            "FAIL_CLOSED canonical dataset hash/aggregate verification failed"
        )

    source_paths = _source_artifact_paths(data_root)
    expected_artifacts = protocol.get("canonical_source_artifact_sha256")
    if (
        not isinstance(expected_artifacts, Mapping)
        or len(source_paths) != 32
        or set(expected_artifacts) != set(source_paths)
        or any(
            not isinstance(digest, str) or len(digest) != 64
            for digest in (
                expected_artifacts.values()
                if isinstance(expected_artifacts, Mapping)
                else ()
            )
        )
    ):
        raise RuntimeError("FAIL_CLOSED canonical source artifact map changed")
    expected_relative_paths = {
        path.relative_to(data_root).as_posix() for path in source_paths.values()
    }
    indexed_source_paths = {
        relative
        for relative in indexed_files
        if PurePosixPath(relative).parts
        and PurePosixPath(relative).parts[0] in {"client_1", "client_2"}
    }
    if indexed_source_paths != expected_relative_paths:
        raise RuntimeError("FAIL_CLOSED canonical source index file set changed")

    for client in SOURCE_CLIENTS:
        directory = data_root / f"client_{client[1:]}"
        if directory.is_symlink() or not directory.is_dir():
            raise RuntimeError("FAIL_CLOSED canonical source directory changed")
        expected_names = {
            path.name
            for key, path in source_paths.items()
            if key.startswith(f"{client}_")
        }
        entries = list(directory.iterdir())
        if (
            {path.name for path in entries} != expected_names
            or any(path.is_symlink() or not path.is_file() for path in entries)
        ):
            raise RuntimeError("FAIL_CLOSED canonical source file set changed")

    for key, path in source_paths.items():
        relative = path.relative_to(data_root).as_posix()
        expected = expected_artifacts[key]
        if indexed_files.get(relative) != expected or sha256_file(path) != expected:
            raise RuntimeError(
                f"FAIL_CLOSED canonical source hash changed: {relative}"
            )

    for client in ("C1", "C2"):
        directory = data_root / f"client_{client[1:]}"
        classifications: dict[str, np.ndarray] = {}
        for split in ("train", "calibration", "test"):
            classes = np.load(
                directory / f"{split}_classification_labels.npy",
                allow_pickle=False,
            ).reshape(-1)
            regression = np.load(
                directory / f"{split}_regression_labels.npy", allow_pickle=False
            )
            expected_n = canonical["source_split_counts_per_client"][split]
            if (
                classes.shape != (expected_n,)
                or regression.shape != (expected_n, 4)
                or not np.issubdtype(classes.dtype, np.integer)
                or not np.isfinite(regression).all()
                or not np.isin(classes, GAS_IDS).all()
            ):
                raise RuntimeError(
                    f"FAIL_CLOSED canonical source label/count contract changed: {client}/{split}"
                )
            classifications[split] = classes.astype(np.int64, copy=False)
        stats_path = directory / "stats.json"
        stats = _read_json(stats_path)
        if (
            stats.get("client_id") != client
            or stats.get("role") != "source"
            or stats.get("counts")
            != canonical["source_split_counts_per_client"]
        ):
            raise RuntimeError(f"FAIL_CLOSED canonical source counts changed: {client}")
        refit_classes = np.concatenate(
            (classifications["train"], classifications["calibration"])
        )
        refit_counts = np.bincount(refit_classes, minlength=4)
        test_counts = np.bincount(classifications["test"], minlength=4)
        if (
            not np.all(
                refit_counts
                == int(canonical["per_gas_refit_count_per_client"])
            )
            or not np.all(
                test_counts == int(canonical["per_gas_test_count_per_client"])
            )
        ):
            raise RuntimeError(
                f"FAIL_CLOSED canonical per-gas source counts changed: {client}"
            )

    canonical_manifest_path = data_root / "canonical_preprocessing_manifest.json"
    expected_manifest_sha256 = canonical.get(
        "canonical_preprocessing_manifest_sha256"
    )
    if (
        canonical_manifest_path.is_symlink()
        or not canonical_manifest_path.is_file()
        or not isinstance(expected_manifest_sha256, str)
        or indexed_files.get("canonical_preprocessing_manifest.json")
        != expected_manifest_sha256
        or sha256_file(canonical_manifest_path) != expected_manifest_sha256
    ):
        raise RuntimeError(
            "FAIL_CLOSED canonical preprocessing manifest hash changed"
        )
    canonical_manifest = _read_json(canonical_manifest_path)
    if (
        canonical_manifest.get("candidate_id") != canonical.get("preprocessing")
        or canonical_manifest.get("sampling_rate_hz")
        != canonical.get("sampling_rate_hz")
        or canonical_manifest.get("points_per_window")
        != canonical.get("window_shape", [None])[0]
        or canonical_manifest.get("window_duration_s")
        != float(canonical.get("window_seconds"))
    ):
        raise RuntimeError("FAIL_CLOSED canonical preprocessing manifest changed")

    feature = protocol["feature_protocol"]
    if (
        sha256_file(EXTRACTOR_PATH) != feature.get("extractor_file_sha256")
        or sha256_strings(H1_FEATURE_NAMES)
        != feature.get("ordered_h1_feature_names_sha256")
        or sha256_strings(SENSOR_FEATURE_NAMES)
        != feature.get("ordered_sensor_feature_names_sha256")
        or len(H1_FEATURE_NAMES) != 104
        or len(SENSOR_FEATURE_NAMES) != 83
    ):
        raise RuntimeError("FAIL_CLOSED canonical feature extractor/schema hash changed")
    return {
        "status": "PASS",
        "aggregate_sha256": expected_aggregate,
        "checked_files": len(source_paths),
        "bad_files": [],
    }


def _verify_indexed_file(root: Path, index: dict[str, Any], relative: str) -> None:
    expected = index.get(relative)
    path = root / relative
    if (
        not isinstance(expected, str)
        or path.is_symlink()
        or not path.is_file()
        or sha256_file(path) != expected
    ):
        raise RuntimeError(f"FAIL_CLOSED original prerequisite hash changed: {path}")


def _verify_original_prerequisites(
    output: Path, protocol: Mapping[str, Any]
) -> dict[str, str]:
    for original in (ORIGINAL_C0_ROOT, ORIGINAL_R0_ROOT):
        if _is_descendant(original, output):
            raise RuntimeError(
                "FAIL_CLOSED original C0/R0 prerequisite cannot be inside output"
            )
        if _is_descendant(output, original):
            raise RuntimeError(
                "FAIL_CLOSED original C0/R0 prerequisite overlaps output"
            )

    anchors = protocol.get("immutable_prerequisites")
    if not isinstance(anchors, Mapping):
        raise RuntimeError("FAIL_CLOSED immutable prerequisite anchors missing")
    c0_anchor = anchors.get("C0")
    r0_anchor = anchors.get("original_R0")
    if not isinstance(c0_anchor, Mapping) or not isinstance(r0_anchor, Mapping):
        raise RuntimeError("FAIL_CLOSED immutable prerequisite anchors missing")
    c0_index_path = ORIGINAL_C0_ROOT / "C0_SHA256_INDEX.json"
    r0_index_path = ORIGINAL_R0_ROOT / "R0_SHA256_INDEX.json"
    if (
        c0_index_path.is_symlink()
        or r0_index_path.is_symlink()
        or not c0_index_path.is_file()
        or not r0_index_path.is_file()
        or sha256_file(c0_index_path) != c0_anchor.get("index_sha256")
        or sha256_file(r0_index_path) != r0_anchor.get("index_sha256")
    ):
        raise RuntimeError("FAIL_CLOSED original prerequisite index anchor changed")

    c0_index = _read_json(c0_index_path)
    for relative in ("C0_DECISION.json", "C0_EXPERIMENT_AUDIT.md"):
        _verify_indexed_file(ORIGINAL_C0_ROOT, c0_index, relative)
    c0_decision = _read_json(ORIGINAL_C0_ROOT / "C0_DECISION.json")
    if c0_decision.get("decision") != "V1_INTERLEAVED_RETAINED":
        raise RuntimeError("FAIL_CLOSED original C0 decision changed")

    r0_index = _read_json(r0_index_path)
    for relative in (
        "canonical_fedridge_exact_recovery.json",
        "R0_FAILURE_AUDIT.json",
        "R0_EXPERIMENT_AUDIT.md",
    ):
        _verify_indexed_file(ORIGINAL_R0_ROOT, r0_index, relative)
    r0_decision = _read_json(
        ORIGINAL_R0_ROOT / "canonical_fedridge_exact_recovery.json"
    )
    r0_failure = _read_json(ORIGINAL_R0_ROOT / "R0_FAILURE_AUDIT.json")
    if (
        r0_decision.get("schema_version")
        != "iotj.canonical_v1.crrq.r0.v1.exact_recovery"
        or r0_decision.get("status") != "FAIL_CLOSED"
        or r0_decision.get("practical_equivalence_fallback") is not False
        or not isinstance(r0_decision.get("gas_results"), list)
        or [row.get("gas_id") for row in r0_decision["gas_results"]]
        != list(GAS_IDS)
        or not any(
            row.get("status") == "FAIL_CLOSED"
            for row in r0_decision["gas_results"]
        )
        or r0_failure.get("schema_version")
        != "iotj.canonical_v1.crrq.r0.failure_audit.v1"
        or r0_failure.get("status") != "FAIL_CLOSED"
        or r0_failure.get("failed_gate")
        != "R0.4_CANONICAL_FEDRIDGE_EXACT_RECOVERY"
        or r0_failure.get("practical_equivalence_fallback_used") is not False
        or r0_failure.get("threshold_relaxed") is not False
        or r0_failure.get("rerun_performed") is not False
        or r0_failure.get("downstream_gate_opened") is not False
        or r0_failure.get("not_opened")
        != {
            "source_test_labels": True,
            "source_test_feature_caches": True,
            "target_C3_feature_caches": True,
            "target_C4_feature_caches": True,
            "target_C5_feature_caches": True,
            "target_test_labels": True,
            "R1": True,
        }
    ):
        raise RuntimeError("FAIL_CLOSED original R0 decision/audit changed")
    return {
        "C0_decision": "V1_INTERLEAVED_RETAINED",
        "original_R0_decision": "R0_EXACT_RECOVERY_NOT_ESTABLISHED",
    }


def preflight(
    data_root: Path, output: Path, authorized_freeze_commit: str
) -> dict[str, Any]:
    """Validate all immutable prerequisites without creating output."""
    data_root = _reject_symlink_components(data_root, "canonical data").resolve()
    output = _reject_symlink_components(output, "output").resolve()
    head = _git_head()
    if authorized_freeze_commit != head:
        raise RuntimeError(
            "FAIL_CLOSED authorized freeze commit must equal current HEAD"
        )
    _require_absent_output(output)
    protected_paths = (
        data_root,
        PROTOCOL_ROOT,
        ORIGINAL_C0_ROOT,
        ORIGINAL_R0_ROOT,
        ORIGINAL_PROTOCOL_MANIFEST,
    )
    if any(_paths_overlap(output, path) for path in protected_paths):
        raise RuntimeError("FAIL_CLOSED input/output path separation violated")
    _verify_critical_paths_match_head(head)
    protocol = _verify_protocol()
    dataset = _verify_source_dataset(data_root, protocol)
    original = _verify_original_prerequisites(output, protocol)
    return {
        "status": "PASS",
        "study_id": R0_V2_STUDY_ID,
        "authorized_freeze_commit": head,
        "protocol_sha256": protocol_freeze_hash(),
        "dataset_aggregate_sha256": dataset["aggregate_sha256"],
        "source_clients": ["C1", "C2"],
        "target_clients": [],
        "formal_execution_started": False,
        "output_created": False,
        **original,
    }


class CanonicalSourceDataProvider:
    """Fresh canonical-cache provider restricted to the two source clients."""

    def __init__(
        self,
        data_root: Path,
        output: Path,
        dataset_aggregate_sha256: str,
    ) -> None:
        self._data_root = Path(data_root).resolve()
        self._cache_root = Path(output).resolve() / "canonical_feature_caches"
        self._dataset_aggregate_sha256 = str(dataset_aggregate_sha256)

    @staticmethod
    def _validate_request(client: str, split: str, gas_id: int | None) -> None:
        if client not in SOURCE_CLIENTS or split not in SOURCE_SPLITS:
            raise RuntimeError("FAIL_CLOSED unregistered source data request")
        if gas_id is not None and gas_id not in GAS_IDS:
            raise RuntimeError("FAIL_CLOSED unregistered source gas request")

    def build_fresh_cache(self, client: str, split: str) -> Mapping[str, Any]:
        self._validate_request(client, split, None)
        return build_feature_cache(
            self._data_root,
            self._cache_root,
            client=client,
            split=split,
            dataset_aggregate_sha256=self._dataset_aggregate_sha256,
            extractor_path=EXTRACTOR_PATH,
            study_id=R0_V2_STUDY_ID,
        )

    def gas_data(
        self, client: str, split: str, gas_id: int
    ) -> tuple[np.ndarray, np.ndarray]:
        self._validate_request(client, split, gas_id)
        cache_dir = self._cache_root / client / split
        _sensor, h1, _identities, manifest = load_feature_cache(
            cache_dir,
            expected_dataset_sha256=self._dataset_aggregate_sha256,
            expected_study_id=R0_V2_STUDY_ID,
        )
        client_number = int(client[1:])
        directory = self._data_root / f"client_{client_number}"
        classes = np.load(
            directory / f"{split}_classification_labels.npy",
            allow_pickle=False,
        ).astype(np.int64, copy=False).reshape(-1)
        regression = np.load(
            directory / f"{split}_regression_labels.npy", allow_pickle=False
        ).astype(np.float64, copy=False)
        if (
            manifest.get("client") != client
            or manifest.get("split") != split
            or len(classes) != len(h1)
            or regression.shape != (len(h1), 4)
        ):
            raise RuntimeError(
                f"FAIL_CLOSED source feature/label contract changed: {client}/{split}"
            )
        mask = classes == gas_id
        if not np.any(mask):
            raise RuntimeError(
                f"FAIL_CLOSED source gas has no rows: {client}/{split}/{gas_id}"
            )
        return h1[mask], regression[mask, gas_id]


def _json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n"


def _publish_bytes(path: Path, content: bytes, *, idempotent: bool) -> None:
    """Publish immutable bytes via a same-directory, exclusive atomic link."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise RuntimeError(f"refusing unsafe evidence parent: {path.parent}")

    def accept_existing() -> bool:
        return (
            idempotent
            and not path.is_symlink()
            and path.is_file()
            and path.read_bytes() == content
        )

    if path.exists() or path.is_symlink():
        if accept_existing():
            return
        raise FileExistsError(
            f"refusing to replace conflicting immutable evidence: {path}"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if accept_existing():
                return
            raise FileExistsError(
                f"refusing to replace conflicting immutable evidence: {path}"
            ) from None
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _publish_bytes(path, _json_text(payload).encode("utf-8"), idempotent=False)


def _write_text(path: Path, text: str) -> None:
    _publish_bytes(path, text.encode("utf-8"), idempotent=False)


def _ensure_json(path: Path, payload: Mapping[str, Any]) -> None:
    _publish_bytes(
        path, _json_text(payload).encode("utf-8"), idempotent=True
    )


def _ensure_text(path: Path, content: str) -> None:
    _publish_bytes(path, content.encode("utf-8"), idempotent=True)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    items = list(rows)
    fields: list[str] = []
    for row in items:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = ["status"]
        items = [{"status": "NO_ROWS"}]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
    writer.writeheader()
    for row in items:
        writer.writerow({key: _csv_value(row.get(key, "")) for key in fields})
    return stream.getvalue()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _publish_bytes(path, _csv_text(rows).encode("utf-8"), idempotent=False)


def _ensure_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _publish_bytes(path, _csv_text(rows).encode("utf-8"), idempotent=True)


def _json_sha256(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _environment_metadata() -> dict[str, Any]:
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        np.__config__.show()
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "dtype": "float64",
        "blas_lapack_configuration": stream.getvalue(),
    }


def _expected_source_requests() -> list[SourceRequest]:
    requests: list[SourceRequest] = []
    for split in ("train", "calibration"):
        for client in SOURCE_CLIENTS:
            requests.append(SourceRequest(client, split, None))
    for gas_id in GAS_IDS:
        for split in ("train", "calibration"):
            for client in SOURCE_CLIENTS:
                requests.append(SourceRequest(client, split, gas_id))
    for client in SOURCE_CLIENTS:
        requests.append(SourceRequest(client, "test", None))
    for gas_id in GAS_IDS:
        for client in SOURCE_CLIENTS:
            requests.append(SourceRequest(client, "test", gas_id))
    return requests


def _request_payload(request: SourceRequest, sequence: int) -> dict[str, Any]:
    return {
        "sequence": sequence,
        "operation": "build_fresh_cache" if request.gas_id is None else "gas_data",
        "client": request.client,
        "split": request.split,
        "gas_id": request.gas_id,
    }


def _validated_protocol_for_execution(
    protocol: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    if not isinstance(protocol, Mapping):
        raise RuntimeError("FAIL_CLOSED execution protocol must be a mapping")
    feature_names = tuple(protocol.get("feature_names", H1_FEATURE_NAMES))
    try:
        alpha_grid = tuple(float(value) for value in protocol.get("alpha_grid", ()))
    except (TypeError, ValueError, OverflowError) as error:
        raise RuntimeError("FAIL_CLOSED registered alpha grid changed") from error
    if (
        protocol.get("study_id") != R0_V2_STUDY_ID
        or protocol.get("source_clients") != list(SOURCE_CLIENTS)
        or protocol.get("target_clients") != []
        or protocol.get("formal_execution_started") is not False
        or len(feature_names) != 104
        or any(not isinstance(name, str) or not name for name in feature_names)
        or alpha_grid != RIDGE_ALPHAS
    ):
        raise RuntimeError("FAIL_CLOSED source-only execution protocol changed")
    return feature_names, alpha_grid


def _combine_source_roles(
    train: Mapping[str, tuple[np.ndarray, np.ndarray]],
    calibration: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if tuple(train) != SOURCE_CLIENTS or tuple(calibration) != SOURCE_CLIENTS:
        raise RuntimeError("FAIL_CLOSED source client order changed")
    return {
        client: (
            np.vstack((train[client][0], calibration[client][0])),
            np.concatenate((train[client][1], calibration[client][1])),
        )
        for client in SOURCE_CLIENTS
    }


def _fit_gas_models(
    combined: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    gas_id: int,
    feature_names: Sequence[str],
    federated_alpha: float,
    pooled_alpha: float,
) -> dict[str, Any]:
    role = "source_train_plus_calibration_refit"
    moments = [
        local_central_moments(client, gas_id, role, combined[client][0])
        for client in SOURCE_CLIENTS
    ]
    federated_scaler = merge_central_moments(moments)
    local_equations = [
        local_normal_equations_v2(
            client,
            gas_id,
            role,
            combined[client][0],
            combined[client][1],
            federated_scaler,
        )
        for client in SOURCE_CLIENTS
    ]
    federated_equations = aggregate_normal_equations_v2(local_equations)
    federated_model = reconstruct_ridge_v2(
        federated_equations, federated_scaler, feature_names, federated_alpha
    )
    pooled_x = np.vstack([combined[client][0] for client in SOURCE_CLIENTS])
    pooled_y = np.concatenate([combined[client][1] for client in SOURCE_CLIENTS])
    pooled_moment = local_central_moments("POOLED", gas_id, role, pooled_x)
    pooled_scaler = merge_central_moments(
        [pooled_moment], expected_client_order=("POOLED",)
    )
    pooled_model, pooled_equations = pooled_reference_fit_v2(
        pooled_x,
        pooled_y,
        gas_id=gas_id,
        role=role,
        feature_names=feature_names,
        alpha=pooled_alpha,
    )
    return {
        "moments": moments,
        "federated_scaler": federated_scaler,
        "federated_equations": federated_equations,
        "federated_model": federated_model,
        "pooled_scaler": pooled_scaler,
        "pooled_equations": pooled_equations,
        "pooled_model": pooled_model,
    }


def _write_hash_index(output: Path) -> dict[str, str]:
    index_path = output / "sha256_index.json"
    symlinks = [
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_symlink()
    ]
    if symlinks:
        raise RuntimeError(f"FAIL_CLOSED evidence contains symlink: {symlinks}")
    index = {
        path.relative_to(output).as_posix(): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
        and not (
            path.parent == output
            and path.name
            in {"sha256_index.json", "fixed_endpoint_complete.json"}
        )
    }
    _ensure_json(index_path, index)
    return index


def _blocking_findings(
    decision: str,
    gas_rows: Sequence[Mapping[str, Any]],
    error: str | None,
    *,
    access_complete: bool = True,
    locks_before_test: bool = True,
    artifact_findings: Sequence[str] = (),
) -> list[str]:
    findings: list[str] = []
    if error:
        findings.append(f"execution_exception:{error}")
    hard_fields = (
        "alpha_equal",
        "scaler_pass",
        "safe_scale_mask_equal",
        "normal_equations_pass",
        "condition_pass",
        "fed_residual_pass",
        "pooled_residual_pass",
        "raw_prediction_pass",
        "clipped_prediction_pass",
        "rmse_parity_pass",
        "mae_parity_pass",
        "finite_pass",
    )
    findings.extend(
        f"gas_{row.get('gas_id')}:{field}"
        for row in gas_rows
        for field in hard_fields
        if row.get(field) is not True
    )
    recomputed_decision = str(decide_r0_v2(gas_rows).get("decision"))
    provenance_override = bool(
        error or not access_complete or not locks_before_test or artifact_findings
    )
    if decision != recomputed_decision and not (
        decision == R0_V2_FAIL and provenance_override
    ):
        findings.append("decision_gate_evidence_conflict")
    if not access_complete:
        findings.append("access_sequence_incomplete")
    if not locks_before_test:
        findings.append("locks_not_complete_before_source_test")
    findings.extend(str(finding) for finding in artifact_findings)
    if decision != R0_V2_PASS and not findings:
        findings.append("incomplete_or_unknown_gate_evidence")
    return findings


def _artifact_type_findings(
    output: Path,
    expected: Sequence[str],
    *,
    require_present: bool,
) -> list[str]:
    findings: list[str] = []
    for relative in expected:
        path = output / relative
        if path.is_symlink():
            findings.append(f"artifact_symlink:{relative}")
            continue
        should_be_directory = relative == "canonical_feature_caches"
        if not path.exists():
            if require_present:
                findings.append(f"artifact_missing:{relative}")
        elif should_be_directory and not path.is_dir():
            findings.append(f"artifact_wrong_type:{relative}")
        elif not should_be_directory and not path.is_file():
            findings.append(f"artifact_wrong_type:{relative}")
    return findings


def _access_audit_text(
    events: Sequence[Mapping[str, Any]],
    *,
    locks_before_test: bool,
    decision: str,
) -> str:
    lines = [
        "# R0-v2 data access audit",
        "",
        f"Decision: `{decision}`.",
        "",
        "The complete provider-key domain was C1/C2 only. No unregistered client, split, or gas key was accepted.",
        f"Source alpha/model locks existed before the first source-test request: `{str(locks_before_test).lower()}`.",
        "Source test was excluded from selection and opened only for post-lock functional evaluation.",
        "",
        "| # | Operation | Client | Split | Gas |",
        "|---:|---|---|---|---:|",
    ]
    for event in events:
        gas = "" if event.get("gas_id") is None else str(event["gas_id"])
        lines.append(
            f"| {event['sequence']} | {event['operation']} | {event['client']} | {event['split']} | {gas} |"
        )
    return "\n".join(lines) + "\n"


def _experiment_audit_text(
    *,
    decision: str,
    findings: Sequence[str],
    access_complete: bool,
) -> str:
    verdict = "PASS; Evidence eligible" if decision == R0_V2_PASS else "BLOCKED"
    lines = [
        "# R0-v2 experiment audit",
        "",
        f"Verdict: **{verdict}**.",
        "",
        "- Experiment ID: `CAN-V1-FEDRIDGE-R0V2-20260812`.",
        "- Comparison: federated sufficient-statistics Ridge versus pooled Ridge on identical ordered C1/C2 rows.",
        "- Split roles: source train selection fit; source calibration SSE/count selection; train+calibration refit; source test after locks.",
        "- Model: four per-gas 104D CanonicalRidgeModelV2 reconstructions; DA/calibration/QC are none; deterministic seed 42 is unused.",
        "- Provenance: frozen protocol, feature/cache manifests, model locks, environment/BLAS record, access events, and SHA256 index.",
        f"- Registered source-only access sequence complete: `{str(access_complete).lower()}`.",
        "- Leakage audit: source test did not enter selection; no unregistered client input was requested.",
        "- Favorable numerical values cannot override a missing, unknown, conflicting, leakage, or provenance defect.",
    ]
    if findings:
        lines.extend(("", "## Blocking findings", ""))
        lines.extend(f"- `{finding}`" for finding in findings)
        lines.append("- Evidence remains blocked; no downstream execution is launched.")
    else:
        lines.extend(
            (
                "",
                "## Evidence decision",
                "",
                "No blocking completeness, fairness, leakage, or provenance defect remains in this source-only execution.",
            )
        )
    return "\n".join(lines) + "\n"


def _finalize_evidence(
    output: Path,
    *,
    protocol: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    cache_manifests: Sequence[Mapping[str, Any]],
    gas_rows: Sequence[Mapping[str, Any]],
    decision: str,
    locks_before_test: bool,
    error: str | None = None,
) -> dict[str, Any]:
    environment_failure: str | None = None
    try:
        environment = _environment_metadata()
    except Exception as environment_error:
        environment_failure = (
            f"{type(environment_error).__name__}: {environment_error}"
        )
        error = (
            environment_failure
            if error is None
            else f"{error}; {environment_failure}"
        )
        decision = R0_V2_FAIL
    expected_requests = _expected_source_requests()
    observed_requests = [
        SourceRequest(str(row["client"]), str(row["split"]), row.get("gas_id"))
        for row in events
    ]
    access_complete = observed_requests == expected_requests
    prepublication_artifacts = _artifact_type_findings(
        output, EXPECTED_FORMAL_FILES[:9], require_present=True
    )
    findings = _blocking_findings(
        decision,
        gas_rows,
        error,
        access_complete=access_complete,
        locks_before_test=locks_before_test,
        artifact_findings=prepublication_artifacts,
    )
    if findings:
        decision = R0_V2_FAIL
    decision_payload = {
        "schema_version": f"{SCHEMA_VERSION}.decision",
        "study_id": R0_V2_STUDY_ID,
        "decision": decision,
        "gas_results": list(gas_rows),
        "blocking_findings": findings,
        "evidence_complete": bool(decision == R0_V2_PASS and not findings),
        "error": error,
    }
    access_audit = _access_audit_text(
        events, locks_before_test=locks_before_test, decision=decision
    )
    experiment_audit = _experiment_audit_text(
        decision=decision,
        findings=findings,
        access_complete=access_complete,
    )
    execution_kind = str(protocol.get("execution_kind", "synthetic_test"))
    execution_manifest = {
        "schema_version": f"{SCHEMA_VERSION}.manifest",
        "study_id": R0_V2_STUDY_ID,
        "status": "PASS" if decision == R0_V2_PASS else "FAIL_CLOSED",
        "decision": decision,
        "execution_kind": execution_kind,
        "formal_execution_started": execution_kind == "formal",
        "execution_commit": str(
            protocol.get("authorized_freeze_commit", "synthetic-test")
        ),
        "frozen_protocol_sha256": EXPECTED_PROTOCOL_FREEZE_SHA256,
        "source_clients": list(SOURCE_CLIENTS),
        "target_clients": [],
        "source_aggregation_order": list(SOURCE_CLIENTS),
        "source_test_opened_after_locks": locks_before_test,
        "access_events": list(events),
        "global_key_audit": {
            "allowed_clients": list(SOURCE_CLIENTS),
            "allowed_splits": list(SOURCE_SPLITS),
            "allowed_gases": list(GAS_IDS),
            "observed_request_count": len(events),
            "exact_registered_sequence": access_complete,
        },
        "cache_manifests": list(cache_manifests),
        "environment": (
            environment
            if environment_failure is None
            else {
                "status": "unavailable",
                "error": environment_failure,
                "dtype": "float64",
            }
        ),
        "numerical_gates": protocol.get("numerical_gates", {}),
        "expected_formal_files": list(EXPECTED_FORMAL_FILES),
        "error": error,
    }

    def persist() -> dict[str, str]:
        _ensure_text(output / "DATA_ACCESS_AUDIT.md", access_audit)
        _ensure_json(output / "R0_V2_DECISION.json", decision_payload)
        _ensure_text(
            output / "R0_V2_EXPERIMENT_AUDIT.md", experiment_audit
        )
        _ensure_json(
            output / "protocol_manifest_execution.json", execution_manifest
        )
        written_index = _write_hash_index(output)
        if decision == R0_V2_PASS:
            publication_findings = _artifact_type_findings(
                output, EXPECTED_FORMAL_FILES, require_present=True
            )
            if publication_findings:
                raise RuntimeError(
                    "FAIL_CLOSED PASS publication artifact defect: "
                    f"{publication_findings}"
                )
            marker = {
                "schema_version": f"{SCHEMA_VERSION}.completion",
                "status": "COMPLETE",
                "study_id": R0_V2_STUDY_ID,
                "decision": decision,
                COMPLETION_RELEASE_FIELD: True,
                "sha256_index_sha256": sha256_file(
                    output / "sha256_index.json"
                ),
                "downstream_launched": False,
            }
            _ensure_json(output / "fixed_endpoint_complete.json", marker)
        return written_index

    try:
        index = persist()
    except Exception:
        index = persist()
    return {
        "status": "PASS" if decision == R0_V2_PASS else "FAIL_CLOSED",
        "decision": decision,
        "files_indexed": len(index),
        "blocking_findings": findings,
    }


def _execute_source_only(
    provider: SourceDataProvider,
    output: Path,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute an injectable C1/C2-only pipeline used by formal and synthetic runs."""
    output = _reject_symlink_components(output, "output").resolve()
    _require_absent_output(output)
    feature_names, alpha_grid = _validated_protocol_for_execution(protocol)
    output.mkdir(parents=True, exist_ok=False)
    events: list[dict[str, Any]] = []
    cache_manifests: list[Mapping[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    scaler_rows: list[dict[str, Any]] = []
    normal_rows: list[dict[str, Any]] = []
    system_rows: list[dict[str, Any]] = []
    functional_rows: list[dict[str, Any]] = []
    alpha_rows: list[dict[str, Any]] = []
    gas_rows: list[dict[str, Any]] = []
    locks_before_test = False

    def build_cache(client: str, split: str) -> Mapping[str, Any]:
        request = SourceRequest(client, split, None)
        events.append(_request_payload(request, len(events)))
        manifest = provider.build_fresh_cache(client, split)
        if (
            not isinstance(manifest, Mapping)
            or manifest.get("client") != client
            or manifest.get("split") != split
            or manifest.get("study_id") != R0_V2_STUDY_ID
        ):
            raise RuntimeError("FAIL_CLOSED source cache provenance mismatch")
        cache_manifests.append(dict(manifest))
        return manifest

    def gas_data(
        client: str, split: str, gas_id: int
    ) -> tuple[np.ndarray, np.ndarray]:
        request = SourceRequest(client, split, gas_id)
        events.append(_request_payload(request, len(events)))
        x, y = provider.gas_data(client, split, gas_id)
        values = np.asarray(x, dtype=np.float64)
        targets = np.asarray(y, dtype=np.float64)
        if (
            values.ndim != 2
            or values.shape[0] == 0
            or values.shape[1] != len(feature_names)
            or targets.ndim != 1
            or len(targets) != len(values)
            or not np.isfinite(values).all()
            or not np.isfinite(targets).all()
        ):
            raise RuntimeError("FAIL_CLOSED source provider returned invalid data")
        return values, targets

    try:
        for split in ("train", "calibration"):
            for client in SOURCE_CLIENTS:
                build_cache(client, split)

        source_rows: dict[
            tuple[int, str], dict[str, tuple[np.ndarray, np.ndarray]]
        ] = {}
        for gas_id in GAS_IDS:
            for split in ("train", "calibration"):
                source_rows[(gas_id, split)] = {
                    client: gas_data(client, split, gas_id)
                    for client in SOURCE_CLIENTS
                }

        models: dict[int, dict[str, Any]] = {}
        selected_alphas: dict[str, dict[str, float]] = {}
        for gas_id in GAS_IDS:
            train = source_rows[(gas_id, "train")]
            calibration = source_rows[(gas_id, "calibration")]
            federated_alpha, federated_audit = select_source_alpha_v2(
                train,
                calibration,
                gas_id=gas_id,
                feature_names=feature_names,
                alphas=alpha_grid,
            )
            pooled_alpha, pooled_audit = select_pooled_alpha_v2(
                train,
                calibration,
                gas_id=gas_id,
                feature_names=feature_names,
                alphas=alpha_grid,
            )
            for route, rows in (
                ("federated", federated_audit),
                ("pooled", pooled_audit),
            ):
                alpha_rows.extend(
                    {"gas_id": gas_id, "route": route, **dict(row)}
                    for row in rows
                )
            combined = _combine_source_roles(train, calibration)
            fitted = _fit_gas_models(
                combined,
                gas_id=gas_id,
                feature_names=feature_names,
                federated_alpha=federated_alpha,
                pooled_alpha=pooled_alpha,
            )
            models[gas_id] = fitted
            selected_alphas[str(gas_id)] = {
                "federated": float(federated_alpha),
                "pooled": float(pooled_alpha),
            }
            feature_rows.extend(
                feature_numerical_audit_rows(
                    fitted["moments"], fitted["federated_scaler"], feature_names
                )
            )
            scaler_rows.append(
                scaler_diagnostics_v2(
                    fitted["federated_scaler"], fitted["pooled_scaler"]
                )
            )
            normal_rows.append(
                normal_equation_diagnostics_v2(
                    fitted["federated_equations"], fitted["pooled_equations"]
                )
            )
            system_rows.append(
                system_diagnostics_v2(
                    fitted["federated_equations"],
                    fitted["pooled_equations"],
                    fitted["federated_model"],
                    fitted["pooled_model"],
                )
            )

        cache_root = output / "canonical_feature_caches"
        cache_root.mkdir(parents=True, exist_ok=True)
        _write_json(
            cache_root / "cache_manifests.json",
            {"study_id": R0_V2_STUDY_ID, "manifests": cache_manifests},
        )
        _write_csv(
            output / "H1_CANONICAL_FEATURE_NUMERICAL_AUDIT.csv", feature_rows
        )
        _write_csv(output / "r0_v2_scaler_diagnostics.csv", scaler_rows)
        _write_csv(
            output / "r0_v2_normal_equation_diagnostics.csv", normal_rows
        )
        _write_csv(output / "r0_v2_system_diagnostics.csv", system_rows)
        _write_csv(output / "source_alpha_audit.csv", alpha_rows)
        alpha_lock = {
            "schema_version": f"{SCHEMA_VERSION}.alpha_lock",
            "study_id": R0_V2_STUDY_ID,
            "status": "LOCKED_BEFORE_SOURCE_TEST",
            "alpha_grid": list(alpha_grid),
            "selected_alpha": selected_alphas,
            "source_test_used_for_selection": False,
            "source_aggregation_order": list(SOURCE_CLIENTS),
        }
        model_payload = {
            str(gas_id): {
                "federated": models[gas_id]["federated_model"].to_json(),
                "pooled": models[gas_id]["pooled_model"].to_json(),
            }
            for gas_id in GAS_IDS
        }
        model_lock = {
            "schema_version": f"{SCHEMA_VERSION}.model_lock",
            "study_id": R0_V2_STUDY_ID,
            "status": "LOCKED_BEFORE_SOURCE_TEST",
            "source_clients": list(SOURCE_CLIENTS),
            "source_aggregation_order": list(SOURCE_CLIENTS),
            "models": model_payload,
            "models_sha256": _json_sha256(model_payload),
        }
        _write_json(output / "source_alpha_lock.json", alpha_lock)
        _write_json(output / "model_lock.json", model_lock)
        locks_before_test = (
            (output / "source_alpha_lock.json").is_file()
            and (output / "model_lock.json").is_file()
        )
        if not locks_before_test:
            raise RuntimeError("FAIL_CLOSED source alpha/model locks missing")

        for client in SOURCE_CLIENTS:
            build_cache(client, "test")
        for gas_id in GAS_IDS:
            source_test = {
                client: gas_data(client, "test", gas_id)
                for client in SOURCE_CLIENTS
            }
            test_x = np.vstack(
                [source_test[client][0] for client in SOURCE_CLIENTS]
            )
            test_y = np.concatenate(
                [source_test[client][1] for client in SOURCE_CLIENTS]
            )
            fitted = models[gas_id]
            functional_row = functional_diagnostics_v2(
                fitted["federated_model"],
                fitted["pooled_model"],
                test_x,
                test_y,
            )
            functional_rows.append(functional_row)
            gas_rows.append(
                decide_gas_equivalence_v2(
                    scaler_rows[gas_id],
                    normal_rows[gas_id],
                    system_rows[gas_id],
                    functional_row,
                )
            )
        _write_csv(
            output / "r0_v2_functional_equivalence.csv", functional_rows
        )
        decision = str(decide_r0_v2(gas_rows)["decision"])
        return _finalize_evidence(
            output,
            protocol=protocol,
            events=events,
            cache_manifests=cache_manifests,
            gas_rows=gas_rows,
            decision=decision,
            locks_before_test=locks_before_test,
        )
    except Exception as error:
        error_text = f"{type(error).__name__}: {error}"
        for path, rows in (
            (output / "H1_CANONICAL_FEATURE_NUMERICAL_AUDIT.csv", feature_rows),
            (output / "r0_v2_scaler_diagnostics.csv", scaler_rows),
            (output / "r0_v2_normal_equation_diagnostics.csv", normal_rows),
            (output / "r0_v2_system_diagnostics.csv", system_rows),
            (output / "r0_v2_functional_equivalence.csv", functional_rows),
            (output / "source_alpha_audit.csv", alpha_rows),
        ):
            _ensure_csv(path, rows)
        cache_root = output / "canonical_feature_caches"
        if cache_root.is_symlink() or (
            cache_root.exists() and not cache_root.is_dir()
        ):
            raise RuntimeError(
                "FAIL_CLOSED canonical feature cache artifact type changed"
            )
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_record = cache_root / "cache_manifests.json"
        _ensure_json(
            cache_record,
            {"study_id": R0_V2_STUDY_ID, "manifests": cache_manifests},
        )
        return _finalize_evidence(
            output,
            protocol=protocol,
            events=events,
            cache_manifests=cache_manifests,
            gas_rows=gas_rows,
            decision=R0_V2_FAIL,
            locks_before_test=locks_before_test,
            error=error_text,
        )


def run(
    data_root: Path, output: Path, authorized_freeze_commit: str
) -> dict[str, Any]:
    """Run the separately authorized formal source-only execution."""
    data_root = _reject_symlink_components(
        data_root, "formal canonical data"
    ).resolve()
    output = _reject_symlink_components(output, "formal output").resolve()
    if data_root != DATA_ROOT.resolve() or output != RESULT_ROOT.resolve():
        raise RuntimeError(
            "FAIL_CLOSED run paths do not match the registered formal roots"
        )
    preflight_result = preflight(data_root, output, authorized_freeze_commit)
    protocol = _read_json(PROTOCOL_MANIFEST)
    execution_protocol = {
        **protocol,
        "feature_names": list(H1_FEATURE_NAMES),
        "alpha_grid": protocol["numerical_protocol"]["alpha_grid"],
        "execution_kind": "formal",
        "authorized_freeze_commit": authorized_freeze_commit,
    }
    provider = CanonicalSourceDataProvider(
        data_root,
        output,
        str(preflight_result["dataset_aggregate_sha256"]),
    )
    return _execute_source_only(provider, output, execution_protocol)


def _read_csv_evidence(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as error:
        raise RuntimeError(
            f"FAIL_CLOSED semantic CSV evidence unreadable: {path.name}"
        ) from error
    if not rows:
        raise RuntimeError(
            f"FAIL_CLOSED semantic CSV evidence has no rows: {path.name}"
        )
    return rows


def _strict_csv_bool(value: Any) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise RuntimeError(f"FAIL_CLOSED semantic boolean is invalid: {value!r}")


def _rows_by_gas(
    rows: Sequence[Mapping[str, str]], filename: str
) -> dict[int, Mapping[str, str]]:
    result: dict[int, Mapping[str, str]] = {}
    for row in rows:
        if row.get("status") == "NO_ROWS":
            continue
        try:
            gas_id = int(str(row["gas_id"]))
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"FAIL_CLOSED semantic gas identity invalid: {filename}"
            ) from error
        if gas_id not in GAS_IDS or gas_id in result:
            raise RuntimeError(
                f"FAIL_CLOSED semantic gas coverage invalid: {filename}"
            )
        result[gas_id] = row
    return result


def _execution_provenance_is_valid(execution: Mapping[str, Any]) -> bool:
    execution_kind = execution.get("execution_kind")
    commit = execution.get("execution_commit")
    if execution_kind == "synthetic_test":
        return (
            commit == "synthetic-test"
            and execution.get("numerical_gates")
            == asdict(registered_tolerances_v2())
        )
    if execution_kind != "formal" or not isinstance(commit, str):
        return False
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        return False
    if protocol_freeze_hash() != EXPECTED_PROTOCOL_FREEZE_SHA256:
        return False
    try:
        committed_protocol = _git_file_bytes(commit, PROTOCOL_MANIFEST)
    except (OSError, subprocess.CalledProcessError, ValueError):
        return False
    return (
        hashlib.sha256(committed_protocol).hexdigest()
        == EXPECTED_PROTOCOL_FREEZE_SHA256
        and execution.get("numerical_gates")
        == _read_json(PROTOCOL_MANIFEST).get("numerical_gates")
    )


def _semantic_evidence_audit(
    output: Path,
    decision_payload: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> None:
    decision = str(decision_payload["decision"])
    gas_results = decision_payload["gas_results"]
    if (
        decision_payload.get("schema_version") != f"{SCHEMA_VERSION}.decision"
        or execution.get("schema_version") != f"{SCHEMA_VERSION}.manifest"
    ):
        raise RuntimeError("FAIL_CLOSED semantic evidence schema mismatch")

    environment = execution.get("environment")
    recorded_error = decision_payload.get("error")
    normal_environment_keys = {
        "python",
        "python_executable",
        "numpy",
        "platform",
        "machine",
        "processor",
        "dtype",
        "blas_lapack_configuration",
    }
    normal_environment = (
        isinstance(environment, Mapping)
        and set(environment) == normal_environment_keys
        and environment.get("dtype") == "float64"
        and all(
            isinstance(environment.get(field), str)
            and bool(environment.get(field))
            for field in (
                "python",
                "python_executable",
                "numpy",
                "platform",
                "machine",
                "blas_lapack_configuration",
            )
        )
        and isinstance(environment.get("processor"), str)
    )
    failure_environment = (
        isinstance(environment, Mapping)
        and set(environment) == {"status", "error", "dtype"}
        and environment.get("status") == "unavailable"
        and environment.get("dtype") == "float64"
        and isinstance(environment.get("error"), str)
        and bool(environment.get("error"))
        and decision == R0_V2_FAIL
        and isinstance(recorded_error, str)
        and (
            recorded_error == environment.get("error")
            or recorded_error.endswith(f"; {environment.get('error')}")
        )
    )
    if not normal_environment and not failure_environment:
        raise RuntimeError("FAIL_CLOSED semantic environment provenance invalid")

    cache_record = _read_json(
        output / "canonical_feature_caches/cache_manifests.json"
    )
    manifests = execution.get("cache_manifests")
    events = execution.get("access_events")
    expected_cache_keys = [
        (event.get("client"), event.get("split"), R0_V2_STUDY_ID)
        for event in events
        if isinstance(event, Mapping)
        and event.get("operation") == "build_fresh_cache"
    ] if isinstance(events, list) else []
    observed_cache_keys = [
        (manifest.get("client"), manifest.get("split"), manifest.get("study_id"))
        for manifest in manifests
        if isinstance(manifest, Mapping)
    ] if isinstance(manifests, list) else []
    cache_record_manifests = cache_record.get("manifests")
    final_event_is_failed_cache_attempt = bool(
        decision == R0_V2_FAIL
        and decision_payload.get("error")
        and isinstance(events, list)
        and events
        and isinstance(events[-1], Mapping)
        and events[-1].get("operation") == "build_fresh_cache"
    )
    cache_attempts_match = observed_cache_keys == expected_cache_keys or (
        final_event_is_failed_cache_attempt
        and observed_cache_keys == expected_cache_keys[:-1]
    )
    if (
        cache_record.get("study_id") != R0_V2_STUDY_ID
        or not isinstance(manifests, list)
        or len(observed_cache_keys) != len(manifests)
        or not cache_attempts_match
        or any(client not in SOURCE_CLIENTS for client, _split, _study in observed_cache_keys)
        or any(split not in SOURCE_SPLITS for _client, split, _study in observed_cache_keys)
        or any(study != R0_V2_STUDY_ID for _client, _split, study in observed_cache_keys)
        or not isinstance(cache_record_manifests, list)
        or cache_record_manifests != manifests[: min(4, len(manifests))]
    ):
        raise RuntimeError("FAIL_CLOSED semantic cache provenance mismatch")

    feature_rows = _read_csv_evidence(
        output / "H1_CANONICAL_FEATURE_NUMERICAL_AUDIT.csv"
    )
    feature_names_by_index: dict[int, str] = {}
    feature_state_by_gas_index: dict[tuple[int, int], tuple[float, float]] = {}
    feature_no_rows = feature_rows[0].get("status") == "NO_ROWS"
    feature_no_rows_stage_valid = bool(
        decision == R0_V2_FAIL
        and not gas_results
        and not (output / "model_lock.json").exists()
    )
    if feature_no_rows and (
        len(feature_rows) != 1 or not feature_no_rows_stage_valid
    ):
        raise RuntimeError(
            "FAIL_CLOSED semantic feature diagnostic coverage mismatch"
        )
    if not feature_no_rows:
        keys: set[tuple[int, int]] = set()
        for row in feature_rows:
            try:
                gas_id = int(row["gas_id"])
                feature_index = int(row["feature_index"])
                n = int(row["n"])
                finite_values = [
                    float(row[field])
                    for field in (
                        "minimum",
                        "maximum",
                        "mean",
                        "population_variance",
                        "raw_scale",
                        "dynamic_range",
                        "safe_scale_floor",
                        "canonical_scale",
                    )
                ]
                order = json.loads(row["aggregation_order"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise RuntimeError(
                    "FAIL_CLOSED semantic feature diagnostic is invalid"
                ) from error
            key = (gas_id, feature_index)
            safe = _strict_csv_bool(row.get("safe_scale_applied"))
            raw_scale = float(row["raw_scale"])
            variance = float(row["population_variance"])
            minimum = float(row["minimum"])
            maximum = float(row["maximum"])
            mean = float(row["mean"])
            dynamic_range = float(row["dynamic_range"])
            feature_name = row.get("feature_name")
            if (
                gas_id not in GAS_IDS
                or not 0 <= feature_index < 104
                or key in keys
                or n <= 0
                or not isinstance(feature_name, str)
                or not feature_name
                or not np.isfinite(finite_values).all()
                or row.get("role") != "source_train_plus_calibration_refit"
                or row.get("dtype") != "float64"
                or order != list(SOURCE_CLIENTS)
                or minimum > mean
                or mean > maximum
                or variance < 0.0
                or raw_scale < 0.0
                or not np.isclose(raw_scale * raw_scale, variance)
                or dynamic_range != maximum - minimum
                or float(row["safe_scale_floor"]) != 1e-9
                or safe is not (raw_scale < 1e-9)
                or float(row["canonical_scale"])
                != (1.0 if safe else raw_scale)
                or (
                    feature_index in feature_names_by_index
                    and feature_names_by_index[feature_index] != feature_name
                )
            ):
                raise RuntimeError(
                    "FAIL_CLOSED semantic feature diagnostic contradiction"
                )
            keys.add(key)
            feature_names_by_index[feature_index] = feature_name
            feature_state_by_gas_index[key] = (
                mean,
                float(row["canonical_scale"]),
            )
        if decision == R0_V2_PASS and keys != {
            (gas_id, feature_index)
            for gas_id in GAS_IDS
            for feature_index in range(104)
        }:
            raise RuntimeError(
                "FAIL_CLOSED semantic feature diagnostic coverage mismatch"
            )

    diagnostic_files = {
        "scaler": "r0_v2_scaler_diagnostics.csv",
        "normal": "r0_v2_normal_equation_diagnostics.csv",
        "system": "r0_v2_system_diagnostics.csv",
        "functional": "r0_v2_functional_equivalence.csv",
    }
    diagnostics = {
        name: _rows_by_gas(_read_csv_evidence(output / filename), filename)
        for name, filename in diagnostic_files.items()
    }
    if feature_no_rows and any(diagnostics.values()):
        raise RuntimeError(
            "FAIL_CLOSED semantic feature diagnostic coverage mismatch"
        )
    if decision == R0_V2_PASS and any(
        set(rows) != set(GAS_IDS) for rows in diagnostics.values()
    ):
        raise RuntimeError("FAIL_CLOSED semantic diagnostic coverage mismatch")
    tolerances = registered_tolerances_v2()

    def arithmetic_close(observed: float, expected: float) -> bool:
        return bool(np.isclose(observed, expected, rtol=1e-13, atol=1e-15))

    def arithmetic_leq(left: float, right: float) -> bool:
        return left <= right or arithmetic_close(left, right)

    # A failed execution can legitimately stop before every diagnostic family
    # exists.  Validate each stored family on its own before doing the
    # cross-family checks below, so a missing later family cannot hide a
    # contradiction in evidence that was already published.
    try:
        for scaler in diagnostics["scaler"].values():
            scaler_values = [
                float(scaler[field])
                for field in (
                    "mean_absolute_mean_error",
                    "max_abs_mean_error",
                    "mean_absolute_scale_error",
                    "max_abs_scale_error",
                    "coordinate_scale_max",
                    "max_normalized_mean_error",
                    "max_normalized_scale_error",
                )
            ]
            scaler_finite = _strict_csv_bool(scaler["finite_pass"])
            mean_pass = _strict_csv_bool(scaler["mean_pass"])
            scale_pass = _strict_csv_bool(scaler["scale_pass"])
            mask_equal = _strict_csv_bool(scaler["safe_scale_mask_equal"])
            (
                mean_absolute_mean_error,
                max_abs_mean_error,
                mean_absolute_scale_error,
                max_abs_scale_error,
                coordinate_scale_max,
                max_normalized_mean_error,
                max_normalized_scale_error,
            ) = scaler_values
            if (
                scaler_finite is not bool(np.isfinite(scaler_values).all())
                or any(value < 0.0 for value in scaler_values)
                or coordinate_scale_max < 1.0
                or not arithmetic_leq(
                    mean_absolute_mean_error, max_abs_mean_error
                )
                or not arithmetic_leq(
                    mean_absolute_scale_error, max_abs_scale_error
                )
                or not arithmetic_leq(
                    max_normalized_mean_error, max_abs_mean_error
                )
                or not arithmetic_leq(
                    max_abs_mean_error,
                    max_normalized_mean_error * coordinate_scale_max,
                )
                or not arithmetic_leq(
                    max_normalized_scale_error, max_abs_scale_error
                )
                or not arithmetic_leq(
                    max_abs_scale_error,
                    max_normalized_scale_error * coordinate_scale_max,
                )
                or mean_pass
                is not (
                    scaler_finite
                    and float(scaler["max_normalized_mean_error"])
                    <= tolerances.tau_moment
                )
                or scale_pass
                is not (
                    scaler_finite
                    and float(scaler["max_normalized_scale_error"])
                    <= tolerances.tau_moment
                )
                or _strict_csv_bool(scaler["scaler_pass"])
                is not (mean_pass and scale_pass and mask_equal and scaler_finite)
            ):
                raise RuntimeError("scaler")

        for normal in diagnostics["normal"].values():
            normal_values = [
                float(normal[field])
                for field in (
                    "absolute_a_discrepancy",
                    "a_denominator",
                    "relative_a_discrepancy",
                    "absolute_b_discrepancy",
                    "b_denominator",
                    "relative_b_discrepancy",
                )
            ]
            normal_finite = _strict_csv_bool(normal["finite_pass"])
            a_positive = _strict_csv_bool(normal["a_denominator_positive"])
            b_positive = _strict_csv_bool(normal["b_denominator_positive"])
            a_pass = _strict_csv_bool(normal["a_pass"])
            b_pass = _strict_csv_bool(normal["b_pass"])
            expected_normal_finite = bool(
                np.isfinite(normal_values).all() and a_positive and b_positive
            )
            if (
                any(value < 0.0 for value in normal_values)
                or a_positive
                is not (float(normal["a_denominator"]) > 0.0)
                or b_positive is not (float(normal["b_denominator"]) > 0.0)
                or normal_finite is not expected_normal_finite
                or (
                    expected_normal_finite
                    and not arithmetic_close(
                        float(normal["relative_a_discrepancy"]),
                        float(normal["absolute_a_discrepancy"])
                        / float(normal["a_denominator"]),
                    )
                )
                or (
                    expected_normal_finite
                    and not arithmetic_close(
                        float(normal["relative_b_discrepancy"]),
                        float(normal["absolute_b_discrepancy"])
                        / float(normal["b_denominator"]),
                    )
                )
                or a_pass
                is not (
                    normal_finite
                    and float(normal["relative_a_discrepancy"])
                    <= tolerances.tau_moment
                )
                or b_pass
                is not (
                    normal_finite
                    and float(normal["relative_b_discrepancy"])
                    <= tolerances.tau_moment
                )
                or _strict_csv_bool(normal["normal_equations_pass"])
                is not (a_pass and b_pass)
            ):
                raise RuntimeError("normal")

        for system in diagnostics["system"].values():
            system_values = [
                float(system[field])
                for field in (
                    "federated_alpha",
                    "pooled_alpha",
                    "federated_condition_number",
                    "pooled_condition_number",
                    "kappa",
                    "fed_residual_norm",
                    "fed_residual_denominator",
                    "fed_relative_residual",
                    "pooled_residual_norm",
                    "pooled_residual_denominator",
                    "pooled_relative_residual",
                    "max_abs_beta_difference",
                    "beta_denominator",
                    "relative_beta_difference",
                    "beta_forward_envelope",
                )
            ]
            system_finite = _strict_csv_bool(system["finite_pass"])
            fed_positive = _strict_csv_bool(
                system["fed_residual_denominator_positive"]
            )
            pooled_positive = _strict_csv_bool(
                system["pooled_residual_denominator_positive"]
            )
            beta_positive = _strict_csv_bool(
                system["beta_denominator_positive"]
            )
            expected_system_finite = bool(
                np.isfinite(system_values).all()
                and fed_positive
                and pooled_positive
                and beta_positive
            )
            condition = bool(
                np.isfinite(float(system["federated_condition_number"]))
                and np.isfinite(float(system["pooled_condition_number"]))
                and float(system["federated_condition_number"])
                * tolerances.epsilon
                < 1.0
                and float(system["pooled_condition_number"])
                * tolerances.epsilon
                < 1.0
            )
            if (
                any(value < 0.0 for value in system_values)
                or _strict_csv_bool(system["alpha_equal"])
                is not (
                    float(system["federated_alpha"])
                    == float(system["pooled_alpha"])
                )
                or float(system["kappa"])
                != max(
                    float(system["federated_condition_number"]),
                    float(system["pooled_condition_number"]),
                )
                or fed_positive
                is not (float(system["fed_residual_denominator"]) > 0.0)
                or pooled_positive
                is not (float(system["pooled_residual_denominator"]) > 0.0)
                or beta_positive
                is not (float(system["beta_denominator"]) > 0.0)
                or system_finite is not expected_system_finite
                or (
                    expected_system_finite
                    and not arithmetic_close(
                        float(system["fed_relative_residual"]),
                        float(system["fed_residual_norm"])
                        / float(system["fed_residual_denominator"]),
                    )
                )
                or (
                    expected_system_finite
                    and not arithmetic_close(
                        float(system["pooled_relative_residual"]),
                        float(system["pooled_residual_norm"])
                        / float(system["pooled_residual_denominator"]),
                    )
                )
                or (
                    expected_system_finite
                    and not arithmetic_close(
                        float(system["beta_forward_envelope"]),
                        float(system["kappa"])
                        * (
                            2.0 * tolerances.tau_moment
                            + tolerances.tau_residual
                        ),
                    )
                )
                or _strict_csv_bool(system["condition_pass"]) is not condition
                or _strict_csv_bool(system["fed_residual_pass"])
                is not (
                    system_finite
                    and float(system["fed_relative_residual"])
                    <= tolerances.tau_residual
                )
                or _strict_csv_bool(system["pooled_residual_pass"])
                is not (
                    system_finite
                    and float(system["pooled_relative_residual"])
                    <= tolerances.tau_residual
                )
                or _strict_csv_bool(system["beta_within_forward_envelope"])
                is not (
                    system_finite
                    and float(system["relative_beta_difference"])
                    <= float(system["beta_forward_envelope"])
                )
            ):
                raise RuntimeError("system")

        for functional in diagnostics["functional"].values():
            functional_values = [
                float(functional[field])
                for field in (
                    "max_abs_raw_prediction_difference",
                    "max_abs_clipped_prediction_difference",
                    "federated_clipped_rmse",
                    "pooled_clipped_rmse",
                    "clipped_rmse_difference",
                    "federated_clipped_mae",
                    "pooled_clipped_mae",
                    "clipped_mae_difference",
                )
            ]
            functional_finite = bool(np.isfinite(functional_values).all())
            if (
                any(value < 0.0 for value in functional_values)
                or _strict_csv_bool(functional["finite_pass"])
                is not functional_finite
                or (
                    functional_finite
                    and not arithmetic_close(
                        float(functional["clipped_rmse_difference"]),
                        abs(
                            float(functional["federated_clipped_rmse"])
                            - float(functional["pooled_clipped_rmse"])
                        ),
                    )
                )
                or (
                    functional_finite
                    and not arithmetic_close(
                        float(functional["clipped_mae_difference"]),
                        abs(
                            float(functional["federated_clipped_mae"])
                            - float(functional["pooled_clipped_mae"])
                        ),
                    )
                )
                or _strict_csv_bool(functional["raw_prediction_pass"])
                is not (
                    functional_finite
                    and float(functional["max_abs_raw_prediction_difference"])
                    <= tolerances.tau_functional_ppm
                )
                or _strict_csv_bool(functional["clipped_prediction_pass"])
                is not (
                    functional_finite
                    and float(functional["max_abs_clipped_prediction_difference"])
                    <= tolerances.tau_functional_ppm
                )
                or _strict_csv_bool(functional["rmse_parity_pass"])
                is not (
                    functional_finite
                    and float(functional["clipped_rmse_difference"])
                    <= tolerances.tau_functional_ppm
                )
                or _strict_csv_bool(functional["mae_parity_pass"])
                is not (
                    functional_finite
                    and float(functional["clipped_mae_difference"])
                    <= tolerances.tau_functional_ppm
                )
            ):
                raise RuntimeError("functional")
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise RuntimeError(
            "FAIL_CLOSED semantic diagnostic internal contradiction"
        ) from error

    for gas_id in set.intersection(
        *(set(rows) for rows in diagnostics.values())
    ) if all(diagnostics.values()) else ():
        scaler = diagnostics["scaler"][gas_id]
        normal = diagnostics["normal"][gas_id]
        system = diagnostics["system"][gas_id]
        functional = diagnostics["functional"][gas_id]
        try:
            scaler_finite = _strict_csv_bool(scaler["finite_pass"])
            mean_pass = _strict_csv_bool(scaler["mean_pass"])
            scale_pass = _strict_csv_bool(scaler["scale_pass"])
            mask_equal = _strict_csv_bool(scaler["safe_scale_mask_equal"])
            if (
                mean_pass
                is not (
                    scaler_finite
                    and float(scaler["max_normalized_mean_error"])
                    <= tolerances.tau_moment
                )
                or scale_pass
                is not (
                    scaler_finite
                    and float(scaler["max_normalized_scale_error"])
                    <= tolerances.tau_moment
                )
                or _strict_csv_bool(scaler["scaler_pass"])
                is not (mean_pass and scale_pass and mask_equal and scaler_finite)
            ):
                raise RuntimeError("scaler")

            normal_finite = _strict_csv_bool(normal["finite_pass"])
            a_positive = _strict_csv_bool(normal["a_denominator_positive"])
            b_positive = _strict_csv_bool(normal["b_denominator_positive"])
            a_pass = _strict_csv_bool(normal["a_pass"])
            b_pass = _strict_csv_bool(normal["b_pass"])
            if (
                a_positive is not (float(normal["a_denominator"]) > 0.0)
                or b_positive is not (float(normal["b_denominator"]) > 0.0)
                or a_pass
                is not (
                    normal_finite
                    and float(normal["relative_a_discrepancy"])
                    <= tolerances.tau_moment
                )
                or b_pass
                is not (
                    normal_finite
                    and float(normal["relative_b_discrepancy"])
                    <= tolerances.tau_moment
                )
                or _strict_csv_bool(normal["normal_equations_pass"])
                is not (a_pass and b_pass)
            ):
                raise RuntimeError("normal")

            system_values = [
                float(system[field])
                for field in (
                    "federated_alpha",
                    "pooled_alpha",
                    "federated_condition_number",
                    "pooled_condition_number",
                    "kappa",
                    "fed_residual_norm",
                    "fed_residual_denominator",
                    "fed_relative_residual",
                    "pooled_residual_norm",
                    "pooled_residual_denominator",
                    "pooled_relative_residual",
                    "max_abs_beta_difference",
                    "beta_denominator",
                    "relative_beta_difference",
                    "beta_forward_envelope",
                )
            ]
            system_finite = _strict_csv_bool(system["finite_pass"])
            fed_positive = _strict_csv_bool(
                system["fed_residual_denominator_positive"]
            )
            pooled_positive = _strict_csv_bool(
                system["pooled_residual_denominator_positive"]
            )
            beta_positive = _strict_csv_bool(
                system["beta_denominator_positive"]
            )
            expected_system_finite = bool(
                np.isfinite(system_values).all()
                and fed_positive
                and pooled_positive
                and beta_positive
            )
            condition = bool(
                np.isfinite(float(system["federated_condition_number"]))
                and np.isfinite(float(system["pooled_condition_number"]))
                and float(system["federated_condition_number"])
                * tolerances.epsilon
                < 1.0
                and float(system["pooled_condition_number"])
                * tolerances.epsilon
                < 1.0
            )
            if (
                _strict_csv_bool(system["alpha_equal"])
                is not (
                    float(system["federated_alpha"])
                    == float(system["pooled_alpha"])
                )
                or float(system["kappa"])
                != max(
                    float(system["federated_condition_number"]),
                    float(system["pooled_condition_number"]),
                )
                or fed_positive
                is not (float(system["fed_residual_denominator"]) > 0.0)
                or pooled_positive
                is not (float(system["pooled_residual_denominator"]) > 0.0)
                or beta_positive
                is not (float(system["beta_denominator"]) > 0.0)
                or system_finite is not expected_system_finite
                or _strict_csv_bool(system["condition_pass"]) is not condition
                or _strict_csv_bool(system["fed_residual_pass"])
                is not (
                    system_finite
                    and float(system["fed_relative_residual"])
                    <= tolerances.tau_residual
                )
                or _strict_csv_bool(system["pooled_residual_pass"])
                is not (
                    system_finite
                    and float(system["pooled_relative_residual"])
                    <= tolerances.tau_residual
                )
                or _strict_csv_bool(system["beta_within_forward_envelope"])
                is not (
                    system_finite
                    and float(system["relative_beta_difference"])
                    <= float(system["beta_forward_envelope"])
                )
            ):
                raise RuntimeError("system")

            functional_values = [
                float(functional[field])
                for field in (
                    "max_abs_raw_prediction_difference",
                    "max_abs_clipped_prediction_difference",
                    "federated_clipped_rmse",
                    "pooled_clipped_rmse",
                    "clipped_rmse_difference",
                    "federated_clipped_mae",
                    "pooled_clipped_mae",
                    "clipped_mae_difference",
                )
            ]
            functional_finite = bool(np.isfinite(functional_values).all())
            if (
                _strict_csv_bool(functional["finite_pass"])
                is not functional_finite
                or _strict_csv_bool(functional["raw_prediction_pass"])
                is not (
                    functional_finite
                    and float(functional["max_abs_raw_prediction_difference"])
                    <= tolerances.tau_functional_ppm
                )
                or _strict_csv_bool(functional["clipped_prediction_pass"])
                is not (
                    functional_finite
                    and float(functional["max_abs_clipped_prediction_difference"])
                    <= tolerances.tau_functional_ppm
                )
                or _strict_csv_bool(functional["rmse_parity_pass"])
                is not (
                    functional_finite
                    and float(functional["clipped_rmse_difference"])
                    <= tolerances.tau_functional_ppm
                )
                or _strict_csv_bool(functional["mae_parity_pass"])
                is not (
                    functional_finite
                    and float(functional["clipped_mae_difference"])
                    <= tolerances.tau_functional_ppm
                )
            ):
                raise RuntimeError("functional")
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            raise RuntimeError(
                f"FAIL_CLOSED semantic diagnostic internal contradiction: gas {gas_id}"
            ) from error
    result_by_gas = {
        int(row["gas_id"]): row
        for row in gas_results
        if isinstance(row, Mapping) and "gas_id" in row
    }
    if len(result_by_gas) != len(gas_results):
        raise RuntimeError("FAIL_CLOSED semantic decision gas coverage mismatch")
    for gas_id, result in result_by_gas.items():
        if any(gas_id not in rows for rows in diagnostics.values()):
            raise RuntimeError("FAIL_CLOSED semantic diagnostic gas mismatch")
        scaler = diagnostics["scaler"][gas_id]
        normal = diagnostics["normal"][gas_id]
        system = diagnostics["system"][gas_id]
        functional = diagnostics["functional"][gas_id]
        expected = {
            "alpha_equal": _strict_csv_bool(system.get("alpha_equal")),
            "scaler_pass": _strict_csv_bool(scaler.get("scaler_pass")),
            "safe_scale_mask_equal": _strict_csv_bool(
                scaler.get("safe_scale_mask_equal")
            ),
            "normal_equations_pass": _strict_csv_bool(
                normal.get("normal_equations_pass")
            ),
            "condition_pass": _strict_csv_bool(system.get("condition_pass")),
            "fed_residual_pass": _strict_csv_bool(
                system.get("fed_residual_pass")
            ),
            "pooled_residual_pass": _strict_csv_bool(
                system.get("pooled_residual_pass")
            ),
            "raw_prediction_pass": _strict_csv_bool(
                functional.get("raw_prediction_pass")
            ),
            "clipped_prediction_pass": _strict_csv_bool(
                functional.get("clipped_prediction_pass")
            ),
            "rmse_parity_pass": _strict_csv_bool(
                functional.get("rmse_parity_pass")
            ),
            "mae_parity_pass": _strict_csv_bool(
                functional.get("mae_parity_pass")
            ),
            "finite_pass": all(
                _strict_csv_bool(row.get("finite_pass"))
                for row in (scaler, normal, system, functional)
            ),
        }
        if any(result.get(field) is not value for field, value in expected.items()):
            raise RuntimeError("FAIL_CLOSED semantic diagnostic contradiction")
        try:
            relative_beta = float(system["relative_beta_difference"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                "FAIL_CLOSED semantic system diagnostic is invalid"
            ) from error
        if result.get("relative_beta_difference") != relative_beta:
            raise RuntimeError("FAIL_CLOSED semantic system diagnostic contradiction")

    alpha_rows = _read_csv_evidence(output / "source_alpha_audit.csv")
    alpha_lock_path = output / "source_alpha_lock.json"
    alpha_lock = _read_json(alpha_lock_path) if alpha_lock_path.is_file() else None
    selected = alpha_lock.get("selected_alpha") if alpha_lock else {}
    if alpha_lock is not None and (
        alpha_lock.get("schema_version") != f"{SCHEMA_VERSION}.alpha_lock"
        or alpha_lock.get("study_id") != R0_V2_STUDY_ID
        or alpha_lock.get("status") != "LOCKED_BEFORE_SOURCE_TEST"
        or alpha_lock.get("alpha_grid") != list(RIDGE_ALPHAS)
        or alpha_lock.get("source_test_used_for_selection") is not False
        or alpha_lock.get("source_aggregation_order") != list(SOURCE_CLIENTS)
        or not isinstance(selected, Mapping)
    ):
        raise RuntimeError("FAIL_CLOSED semantic alpha lock contradiction")
    grouped: dict[tuple[str, str], list[Mapping[str, str]]] = {}
    for row in alpha_rows:
        if row.get("status") == "NO_ROWS":
            continue
        try:
            gas_id = int(row["gas_id"])
            route = str(row["route"])
            alpha = float(row["alpha"])
            calibration_rmse = float(row["source_calibration_RMSE"])
            calibration_n = (
                int(row["source_calibration_N"])
                if row.get("source_calibration_N") not in (None, "")
                else None
            )
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise RuntimeError("FAIL_CLOSED semantic alpha audit invalid") from error
        if (
            gas_id not in GAS_IDS
            or route not in ("federated", "pooled")
            or not np.isfinite(alpha)
            or alpha not in RIDGE_ALPHAS
            or not np.isfinite(calibration_rmse)
            or calibration_rmse < 0.0
            or (calibration_n is not None and calibration_n <= 0)
            or (
                execution.get("execution_kind") == "formal"
                and calibration_n is None
            )
            or row.get("target_input_accessed") != "False"
            or row.get("source_test_accessed") != "False"
        ):
            raise RuntimeError("FAIL_CLOSED semantic alpha access contradiction")
        grouped.setdefault((str(gas_id), route), []).append(row)
    if decision == R0_V2_PASS and set(grouped) != {
        (str(gas_id), route)
        for gas_id in GAS_IDS
        for route in ("federated", "pooled")
    }:
        raise RuntimeError("FAIL_CLOSED semantic alpha audit coverage mismatch")
    for (gas, route), rows in grouped.items():
        try:
            alpha_values = [float(row["alpha"]) for row in rows]
            best_rmse = min(float(row["source_calibration_RMSE"]) for row in rows)
            chosen = next(
                float(row["alpha"])
                for row in rows
                if float(row["source_calibration_RMSE"]) == best_rmse
            )
        except (KeyError, TypeError, ValueError, StopIteration) as error:
            raise RuntimeError("FAIL_CLOSED semantic alpha audit invalid") from error
        registered_subset = [
            alpha for alpha in RIDGE_ALPHAS if alpha in set(alpha_values)
        ]
        if (
            len(alpha_values) != len(set(alpha_values))
            or any(alpha not in RIDGE_ALPHAS for alpha in alpha_values)
            or alpha_values != registered_subset
            or (
                execution.get("execution_kind") == "formal"
                and alpha_values != list(RIDGE_ALPHAS)
            )
        ):
            raise RuntimeError("FAIL_CLOSED semantic alpha grid contradiction")
        if alpha_lock is not None and (
            not isinstance(selected.get(gas), Mapping)
            or selected[gas].get(route) != chosen
        ):
            raise RuntimeError("FAIL_CLOSED semantic alpha selection contradiction")

    model_lock_path = output / "model_lock.json"
    model_lock = _read_json(model_lock_path) if model_lock_path.is_file() else None
    models = model_lock.get("models") if model_lock else None
    if model_lock is not None and (
        model_lock.get("schema_version") != f"{SCHEMA_VERSION}.model_lock"
        or model_lock.get("study_id") != R0_V2_STUDY_ID
        or model_lock.get("status") != "LOCKED_BEFORE_SOURCE_TEST"
        or model_lock.get("source_clients") != list(SOURCE_CLIENTS)
        or model_lock.get("source_aggregation_order") != list(SOURCE_CLIENTS)
        or not isinstance(models, Mapping)
        or set(models) != {str(gas_id) for gas_id in GAS_IDS}
        or model_lock.get("models_sha256") != _json_sha256(models)
    ):
        raise RuntimeError("FAIL_CLOSED semantic model lock contradiction")
    for gas, routes in (models.items() if isinstance(models, Mapping) else ()):
        if not isinstance(routes, Mapping) or set(routes) != {"federated", "pooled"}:
            raise RuntimeError("FAIL_CLOSED semantic model routes mismatch")
        for route, model in routes.items():
            try:
                model_names = list(model["feature_names"])
                mean = np.asarray(model["mean"], dtype=np.float64)
                scale = np.asarray(model["scale"], dtype=np.float64)
                coef = np.asarray(model["coef"], dtype=np.float64)
                clip_min = float(model["clip_min"])
                clip_max = float(model["clip_max"])
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(
                    "FAIL_CLOSED semantic model state invalid"
                ) from error
            expected_scaler_state = [
                feature_state_by_gas_index.get((int(gas), index))
                for index in range(104)
            ]
            scaler_state_complete = all(
                state is not None for state in expected_scaler_state
            )
            expected_mean = np.asarray(
                [state[0] for state in expected_scaler_state]
                if scaler_state_complete
                else [],
                dtype=np.float64,
            )
            expected_scale = np.asarray(
                [state[1] for state in expected_scaler_state]
                if scaler_state_complete
                else [],
                dtype=np.float64,
            )
            if (
                not isinstance(model, Mapping)
                or model.get("gas_id") != int(gas)
                or model.get("alpha") != selected[gas][route]
                or model.get("role") != "source_train_plus_calibration_refit"
                or model.get("solver") != "numpy.linalg.pinv"
                or model.get("intercept_regularized") is not False
                or len(model_names) != 104
                or mean.shape != (104,)
                or scale.shape != (104,)
                or coef.shape != (105,)
                or not np.isfinite(mean).all()
                or not np.isfinite(scale).all()
                or not np.isfinite(coef).all()
                or np.any(scale <= 0.0)
                or not np.isfinite([clip_min, clip_max]).all()
                or clip_min > clip_max
                or (
                    feature_names_by_index
                    and model_names
                    != [feature_names_by_index[index] for index in range(104)]
                )
                or (
                    feature_state_by_gas_index
                    and (
                        not scaler_state_complete
                        or not np.array_equal(mean, expected_mean)
                        or not np.array_equal(scale, expected_scale)
                    )
                )
            ):
                raise RuntimeError("FAIL_CLOSED semantic model lock contradiction")
        system_row = diagnostics["system"].get(int(gas))
        if system_row is None:
            raise RuntimeError("FAIL_CLOSED semantic model/system linkage missing")
        fed_model = routes["federated"]
        pooled_model = routes["pooled"]
        if (
            float(fed_model["clip_min"]) != float(pooled_model["clip_min"])
            or float(fed_model["clip_max"]) != float(pooled_model["clip_max"])
        ):
            raise RuntimeError("FAIL_CLOSED semantic model clip contradiction")
        fed_coef = np.asarray(fed_model["coef"], dtype=np.float64)
        pooled_coef = np.asarray(pooled_model["coef"], dtype=np.float64)
        coefficient_delta = fed_coef - pooled_coef
        max_abs_beta = float(np.max(np.abs(coefficient_delta)))
        beta_denominator = float(np.linalg.norm(pooled_coef, ord=2))
        with np.errstate(divide="ignore", invalid="ignore"):
            relative_beta = float(
                np.divide(
                    np.linalg.norm(coefficient_delta, ord=2),
                    beta_denominator,
                )
            )
        linked_values = (
            (float(system_row["federated_alpha"]), float(fed_model["alpha"])),
            (float(system_row["pooled_alpha"]), float(pooled_model["alpha"])),
            (float(system_row["max_abs_beta_difference"]), max_abs_beta),
            (float(system_row["beta_denominator"]), beta_denominator),
            (float(system_row["relative_beta_difference"]), relative_beta),
        )
        if any(
            not np.isclose(observed, expected, rtol=1e-14, atol=1e-15)
            for observed, expected in linked_values
        ):
            raise RuntimeError(
                "FAIL_CLOSED semantic model coefficient diagnostics contradiction"
            )

    events = execution["access_events"]
    locks_before_test = execution.get("source_test_opened_after_locks") is True
    if (output / "DATA_ACCESS_AUDIT.md").read_text(encoding="utf-8") != _access_audit_text(
        events, locks_before_test=locks_before_test, decision=decision
    ):
        raise RuntimeError("FAIL_CLOSED semantic access audit contradiction")
    access_complete = events == [
        _request_payload(request, sequence)
        for sequence, request in enumerate(_expected_source_requests())
    ]
    if (output / "R0_V2_EXPERIMENT_AUDIT.md").read_text(
        encoding="utf-8"
    ) != _experiment_audit_text(
        decision=decision,
        findings=decision_payload["blocking_findings"],
        access_complete=access_complete,
    ):
        raise RuntimeError("FAIL_CLOSED semantic experiment audit contradiction")


def audit(output: Path) -> dict[str, Any]:
    """Audit an existing execution without recomputing numerical evidence."""
    output = _reject_symlink_components(output, "output").resolve()
    index_path = output / "sha256_index.json"
    symlinks = [
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_symlink()
    ]
    if symlinks:
        raise RuntimeError(f"FAIL_CLOSED evidence symlink violation: {symlinks}")
    wrong_types = _artifact_type_findings(
        output, EXPECTED_FORMAL_FILES, require_present=False
    )
    if wrong_types:
        raise RuntimeError(
            f"FAIL_CLOSED evidence artifact type violation: {wrong_types}"
        )
    if index_path.is_symlink() or not index_path.is_file():
        raise RuntimeError("FAIL_CLOSED SHA256 index type violation")
    index = _read_json(index_path)
    indexable_files = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
        and not (
            path.parent == output
            and path.name
            in {"sha256_index.json", "fixed_endpoint_complete.json"}
        )
    }
    if set(index) != indexable_files:
        raise RuntimeError("FAIL_CLOSED SHA256 index completeness violation")
    bad: list[str] = []
    for relative, expected in index.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise RuntimeError("FAIL_CLOSED hash index schema is invalid")
        path = (output / relative).resolve()
        if not _is_descendant(path, output) or path == index_path:
            raise RuntimeError("FAIL_CLOSED hash index contains an unsafe path")
        if not path.is_file() or sha256_file(path) != expected:
            bad.append(relative)
    if bad:
        raise RuntimeError(f"FAIL_CLOSED evidence hash mismatch: {bad}")

    decision_payload = _read_json(output / "R0_V2_DECISION.json")
    decision = decision_payload.get("decision")
    if decision not in DECISION_VOCABULARY:
        raise RuntimeError("FAIL_CLOSED decision vocabulary violation")
    if decision_payload.get("study_id") != R0_V2_STUDY_ID:
        raise RuntimeError("FAIL_CLOSED decision provenance is invalid")
    gas_rows = decision_payload.get("gas_results")
    if not isinstance(gas_rows, list):
        raise RuntimeError("FAIL_CLOSED decision gate evidence is incomplete")
    recorded_error = decision_payload.get("error")
    if recorded_error is not None and (
        not isinstance(recorded_error, str) or not recorded_error
    ):
        raise RuntimeError("FAIL_CLOSED decision error provenance is invalid")
    recomputed_decision = decide_r0_v2(gas_rows).get("decision")
    findings = decision_payload.get("blocking_findings")
    if (
        not isinstance(findings, list)
        or (decision == R0_V2_PASS and findings != [])
        or (decision == R0_V2_FAIL and not findings)
        or decision_payload.get("evidence_complete")
        is not (decision == R0_V2_PASS)
    ):
        raise RuntimeError(
            "FAIL_CLOSED decision completeness/blocking findings are invalid"
        )

    execution = _read_json(output / "protocol_manifest_execution.json")
    events = execution.get("access_events")
    if not isinstance(events, list):
        raise RuntimeError("FAIL_CLOSED access audit events are missing")
    try:
        observed = [
            SourceRequest(
                client=str(event["client"]),
                split=str(event["split"]),
                gas_id=event.get("gas_id"),
            )
            for event in events
        ]
    except (KeyError, TypeError) as error:
        raise RuntimeError("FAIL_CLOSED access audit event schema changed") from error
    expected_requests = _expected_source_requests()
    expected_events = [
        _request_payload(request, sequence)
        for sequence, request in enumerate(expected_requests)
    ]
    valid_prefix = (
        observed == expected_requests[: len(observed)]
        and events == expected_events[: len(events)]
    )
    exact_access = observed == expected_requests
    expected_global_key_audit = {
        "allowed_clients": list(SOURCE_CLIENTS),
        "allowed_splits": list(SOURCE_SPLITS),
        "allowed_gases": list(GAS_IDS),
        "observed_request_count": len(events),
        "exact_registered_sequence": exact_access,
    }
    execution_kind = execution.get("execution_kind")
    valid_execution_kind = execution_kind in ("formal", "synthetic_test") and (
        execution.get("formal_execution_started") is (execution_kind == "formal")
    )
    if not _execution_provenance_is_valid(execution):
        raise RuntimeError("FAIL_CLOSED execution provenance manifest is invalid")
    if (
        not valid_prefix
        or execution.get("study_id") != R0_V2_STUDY_ID
        or execution.get("decision") != decision
        or execution.get("error") != recorded_error
        or execution.get("status")
        != ("PASS" if decision == R0_V2_PASS else "FAIL_CLOSED")
        or not valid_execution_kind
        or execution.get("source_clients") != list(SOURCE_CLIENTS)
        or execution.get("target_clients") != []
        or execution.get("source_aggregation_order") != list(SOURCE_CLIENTS)
        or execution.get("frozen_protocol_sha256")
        != EXPECTED_PROTOCOL_FREEZE_SHA256
        or execution.get("global_key_audit") != expected_global_key_audit
        or execution.get("expected_formal_files") != list(EXPECTED_FORMAL_FILES)
        or (
            decision == R0_V2_PASS
            and any(request.split == "test" for request in observed)
            and execution.get("source_test_opened_after_locks") is not True
        )
        or (decision == R0_V2_PASS and not exact_access)
    ):
        raise RuntimeError("FAIL_CLOSED access/protocol audit validation failed")

    locks_before_test = execution.get("source_test_opened_after_locks") is True
    artifact_findings = _artifact_type_findings(
        output, EXPECTED_FORMAL_FILES[:9], require_present=True
    )
    expected_findings = _blocking_findings(
        str(decision),
        gas_rows,
        recorded_error,
        access_complete=exact_access,
        locks_before_test=locks_before_test,
        artifact_findings=artifact_findings,
    )
    provenance_failure_override = bool(
        decision == R0_V2_FAIL
        and recomputed_decision == R0_V2_PASS
        and (
            recorded_error
            or not exact_access
            or not locks_before_test
            or artifact_findings
        )
    )
    if recomputed_decision != decision and not provenance_failure_override:
        raise RuntimeError("FAIL_CLOSED decision conflicts with stored gate evidence")
    if findings != expected_findings:
        raise RuntimeError(
            "FAIL_CLOSED decision completeness/blocking findings are invalid"
        )
    _semantic_evidence_audit(output, decision_payload, execution)

    marker_path = output / "fixed_endpoint_complete.json"
    if decision == R0_V2_PASS:
        missing = [
            relative
            for relative in EXPECTED_FORMAL_FILES
            if not (output / relative).exists()
        ]
        if missing:
            raise RuntimeError(
                f"FAIL_CLOSED PASS evidence is incomplete: {missing}"
            )
        marker = _read_json(marker_path)
        if (
            marker.get("schema_version") != f"{SCHEMA_VERSION}.completion"
            or marker.get("study_id") != R0_V2_STUDY_ID
            or marker.get("status") != "COMPLETE"
            or marker.get("decision") != R0_V2_PASS
            or marker.get(COMPLETION_RELEASE_FIELD) is not True
            or marker.get("downstream_launched") is not False
            or marker.get("sha256_index_sha256") != sha256_file(index_path)
        ):
            raise RuntimeError("FAIL_CLOSED PASS completion marker is invalid")
    elif marker_path.exists():
        raise RuntimeError("FAIL_CLOSED failure evidence has a completion marker")
    return {"status": "PASS", "decision": str(decision), "bad": bad}


class _RunnerParser(argparse.ArgumentParser):
    def parse_args(self, args: Any = None, namespace: Any = None) -> argparse.Namespace:
        parsed = super().parse_args(args=args, namespace=namespace)
        if (
            parsed.stage in ("preflight", "run")
            and not parsed.authorized_freeze_commit
        ):
            self.error(
                "--authorized-freeze-commit is required for preflight and run"
            )
        return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = _RunnerParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "run", "audit"))
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output", type=Path, default=RESULT_ROOT)
    parser.add_argument("--authorized-freeze-commit")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.stage == "preflight":
        result = preflight(
            args.data_root, args.output, args.authorized_freeze_commit
        )
    elif args.stage == "run":
        result = run(args.data_root, args.output, args.authorized_freeze_commit)
    else:
        result = audit(args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
