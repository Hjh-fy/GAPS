"""Fail-closed controller for the frozen canonical FedRidge R0-v2 study."""

from __future__ import annotations

import argparse
import contextlib
import csv
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
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
    "96cb6e6ce5826e24774f633d8fe0082e420bb377f2eeff976625014b06205e96"
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


def _verify_source_dataset(data_root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    canonical = protocol["canonical_data"]
    expected_aggregate = str(canonical["dataset_aggregate_sha256"])
    dataset_index = _read_json(data_root / "dataset_sha256.json")
    indexed_files = dataset_index.get("files")
    if (
        dataset_index.get("aggregate_sha256") != expected_aggregate
        or not isinstance(indexed_files, Mapping)
    ):
        raise RuntimeError("FAIL_CLOSED canonical dataset hash verification failed")

    def verify_indexed_source(path: Path) -> None:
        nonlocal checked_files
        relative = path.relative_to(data_root).as_posix()
        expected = indexed_files.get(relative)
        if not isinstance(expected, str) or sha256_file(path) != expected:
            raise RuntimeError(
                f"FAIL_CLOSED canonical source hash changed: {relative}"
            )
        checked_files += 1

    expected_artifacts = protocol["canonical_source_artifact_sha256"]
    checked_files = 0
    for client in ("C1", "C2"):
        directory = data_root / f"client_{client[1:]}"
        classifications: dict[str, np.ndarray] = {}
        for split in ("train", "calibration", "test"):
            paths = {
                "features": directory / f"{split}_features.npy",
                "phase": directory / f"{split}_phase_labels.npy",
                "metadata": directory / f"{split}_experiment_info.json",
            }
            for kind, path in paths.items():
                key = f"{client}_{split}_{kind}"
                if sha256_file(path) != expected_artifacts.get(key):
                    raise RuntimeError(
                        f"FAIL_CLOSED canonical source artifact hash changed: {key}"
                    )
                verify_indexed_source(path)
            for label_kind in ("classification", "regression"):
                verify_indexed_source(
                    directory / f"{split}_{label_kind}_labels.npy"
                )
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
        verify_indexed_source(stats_path)
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
    verify_indexed_source(canonical_manifest_path)
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
        "checked_files": checked_files,
        "bad_files": [],
    }


def _verify_indexed_file(root: Path, index: dict[str, Any], relative: str) -> None:
    expected = index.get(relative)
    path = root / relative
    if not isinstance(expected, str) or sha256_file(path) != expected:
        raise RuntimeError(f"FAIL_CLOSED original prerequisite hash changed: {path}")


def _verify_original_prerequisites(output: Path) -> dict[str, str]:
    for original in (ORIGINAL_C0_ROOT, ORIGINAL_R0_ROOT):
        if _is_descendant(original, output):
            raise RuntimeError(
                "FAIL_CLOSED original C0/R0 prerequisite cannot be inside output"
            )
        if _is_descendant(output, original):
            raise RuntimeError(
                "FAIL_CLOSED original C0/R0 prerequisite overlaps output"
            )

    c0_index = _read_json(ORIGINAL_C0_ROOT / "C0_SHA256_INDEX.json")
    for relative in ("C0_DECISION.json", "C0_EXPERIMENT_AUDIT.md"):
        _verify_indexed_file(ORIGINAL_C0_ROOT, c0_index, relative)
    c0_decision = _read_json(ORIGINAL_C0_ROOT / "C0_DECISION.json")
    if c0_decision.get("decision") != "V1_INTERLEAVED_RETAINED":
        raise RuntimeError("FAIL_CLOSED original C0 decision changed")

    r0_index = _read_json(ORIGINAL_R0_ROOT / "R0_SHA256_INDEX.json")
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
    original_protocol = _read_json(ORIGINAL_PROTOCOL_MANIFEST)
    r0_protocol = original_protocol.get("R0", {}).get("execution_result", {})
    if (
        r0_decision.get("status") != "FAIL_CLOSED"
        or r0_failure.get("status") != "FAIL_CLOSED"
        or r0_protocol.get("status") != "FAIL_CLOSED"
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
    data_root = Path(data_root).resolve()
    output = Path(output).resolve()
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
    protocol = _verify_protocol()
    dataset = _verify_source_dataset(data_root, protocol)
    original = _verify_original_prerequisites(output)
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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_text(payload), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _ensure_json(path: Path, payload: Mapping[str, Any]) -> None:
    expected = _json_text(payload)
    if path.exists():
        if path.read_text(encoding="utf-8") != expected:
            raise FileExistsError(
                f"refusing to replace conflicting immutable evidence: {path}"
            )
        return
    _write_json(path, payload)


def _ensure_text(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(
                f"refusing to replace conflicting immutable evidence: {path}"
            )
        return
    _write_text(path, content)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    items = list(rows)
    fields: list[str] = []
    for row in items:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = ["status"]
        items = [{"status": "NO_ROWS"}]
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in items:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fields})


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
    index = {
        path.relative_to(output).as_posix(): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
        and path.name not in {"sha256_index.json", "fixed_endpoint_complete.json"}
    }
    if index_path.exists():
        if _read_json(index_path) != index:
            raise FileExistsError(
                f"refusing to replace conflicting immutable evidence: {index_path}"
            )
        return index
    _write_json(index_path, index)
    return index


def _blocking_findings(
    decision: str, gas_rows: Sequence[Mapping[str, Any]], error: str | None
) -> list[str]:
    if error:
        return [f"execution_exception:{error}"]
    if decision == R0_V2_PASS:
        return []
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
    findings = [
        f"gas_{row.get('gas_id')}:{field}"
        for row in gas_rows
        for field in hard_fields
        if row.get(field) is not True
    ]
    return findings or ["incomplete_or_unknown_gate_evidence"]


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
    findings = _blocking_findings(decision, gas_rows, error)
    expected_requests = _expected_source_requests()
    observed_requests = [
        SourceRequest(str(row["client"]), str(row["split"]), row.get("gas_id"))
        for row in events
    ]
    access_complete = observed_requests == expected_requests
    decision_payload = {
        "schema_version": f"{SCHEMA_VERSION}.decision",
        "study_id": R0_V2_STUDY_ID,
        "decision": decision,
        "gas_results": list(gas_rows),
        "blocking_findings": findings,
        "evidence_complete": bool(decision == R0_V2_PASS and access_complete),
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
    output = Path(output).resolve()
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
            if not path.exists():
                _write_csv(path, rows)
        cache_root = output / "canonical_feature_caches"
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_record = cache_root / "cache_manifests.json"
        if not cache_record.exists():
            _write_json(
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


def audit(output: Path) -> dict[str, Any]:
    """Audit an existing execution without recomputing numerical evidence."""
    output = Path(output).resolve()
    index_path = output / "sha256_index.json"
    index = _read_json(index_path)
    indexable_files = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
        and path.name not in {"sha256_index.json", "fixed_endpoint_complete.json"}
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
    execution_failure_override = (
        decision == R0_V2_FAIL
        and recomputed_decision == R0_V2_PASS
        and recorded_error is not None
    )
    if recomputed_decision != decision and not execution_failure_override:
        raise RuntimeError("FAIL_CLOSED decision conflicts with stored gate evidence")
    findings = decision_payload.get("blocking_findings")
    expected_findings = _blocking_findings(
        str(decision), gas_rows, recorded_error
    )
    if (
        not isinstance(findings, list)
        or findings != expected_findings
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
    if (
        not valid_prefix
        or execution.get("study_id") != R0_V2_STUDY_ID
        or execution.get("decision") != decision
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
            any(request.split == "test" for request in observed)
            and execution.get("source_test_opened_after_locks") is not True
        )
        or (decision == R0_V2_PASS and not exact_access)
    ):
        raise RuntimeError("FAIL_CLOSED access/protocol audit validation failed")

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
            marker.get("status") != "COMPLETE"
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
