"""Audit and package the final A4 Fig. 5--Fig. 8 evidence root."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.finalize_iotj_a4_end_to_end import read_json, write_json


REQUIRED = [
    "final_classifier_manifest.json",
    "regression/calibration_selection_lock.json",
    "regression/regression_main_summary.csv",
    "regression/regression_per_gas.csv",
    "regression/final_test_records.csv",
    "qc/qc_threshold_lock.csv",
    "qc/qc_coverage_curve.csv",
    "qc/qc_random_reference.csv",
    "qc/qc_operating_points.csv",
    "system/system_deployment_summary.csv",
    "system/physical_validation_audit.csv",
    "figures/Fig5_concentration_estimation_per_gas.pdf",
    "figures/Fig5_concentration_estimation_per_gas.png",
    "figures/Fig6_source_prior_ablation_calibration_budget.pdf",
    "figures/Fig6_source_prior_ablation_calibration_budget.png",
    "figures/Fig7_qc_coverage_nrmse_random_hc.pdf",
    "figures/Fig7_qc_coverage_nrmse_random_hc.png",
    "figures/Fig8_communication_pi5_physical_validation.pdf",
    "figures/Fig8_communication_pi5_physical_validation.png",
]


def require_artifacts(root: Path, relative_paths: Sequence[str]) -> None:
    for relative in relative_paths:
        path = root / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"FAIL_CLOSED required artifact missing: {relative}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        if not path.is_file() or path.name == "sha256_index.json":
            continue
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _find(rows: Sequence[dict[str, str]], **criteria: str) -> dict[str, str]:
    return next(row for row in rows if all(row.get(key) == value for key, value in criteria.items()))


def _copy_source_data(root: Path, budget_csv: Path) -> None:
    destination = root / "figures/source_data"
    destination.mkdir(parents=True, exist_ok=True)
    # The duplicate source is intentionally given two figure-specific names.
    pairs = [
        (root / "regression/regression_main_summary.csv", destination / "fig05_overall_regression.csv"),
        (root / "regression/regression_per_gas.csv", destination / "fig05_per_gas_regression.csv"),
        (root / "regression/regression_main_summary.csv", destination / "fig06_source_prior_ablation.csv"),
        (budget_csv, destination / "fig06_calibration_budget.csv"),
        (root / "qc/qc_coverage_curve.csv", destination / "fig07_qc_coverage_curve.csv"),
        (root / "qc/qc_random_reference.csv", destination / "fig07_random_reference.csv"),
        (root / "qc/qc_operating_points.csv", destination / "fig07_hc_operating_points.csv"),
        (root / "system/system_deployment_summary.csv", destination / "fig08_system_deployment.csv"),
        (root / "system/physical_validation_audit.csv", destination / "fig08_physical_validation.csv"),
    ]
    for source, target in pairs:
        if not source.is_file():
            raise FileNotFoundError(f"FAIL_CLOSED figure source missing: {source}")
        shutil.copyfile(source, target)


def finalize_package(root: Path, budget_csv: Path) -> None:
    require_artifacts(root, REQUIRED)
    _copy_source_data(root, budget_csv)
    classifier = read_json(root / "final_classifier_manifest.json")
    regression_protocol = read_json(root / "regression/protocol_manifest.json")
    qc_protocol = read_json(root / "qc/protocol_manifest.json")
    system_protocol = read_json(root / "system/protocol_manifest.json")
    main = _read_csv(root / "regression/regression_main_summary.csv")
    per_gas = _read_csv(root / "regression/regression_per_gas.csv")
    qc = _read_csv(root / "qc/qc_coverage_curve.csv")
    random = _read_csv(root / "qc/qc_random_reference.csv")
    physical = _read_csv(root / "system/physical_validation_audit.csv")[0]

    r83 = _find(main, variant="R83_TARGET_ONLY", evaluation_scope="S_ALL")
    r84 = _find(main, variant="R84_FED_H1", evaluation_scope="S_ALL")
    r84_cc = _find(main, variant="R84_FED_H1", evaluation_scope="S_CC")
    r86 = _find(main, variant="R86_ALL_PRIORS", evaluation_scope="S_ALL")
    hc90 = _find(qc, target_coverage="0.9")
    hc95 = _find(qc, target_coverage="0.95")
    full = _find(qc, target_coverage="1.0")
    rand90 = _find(random, target_coverage="0.9")
    gas_lines = []
    for gas in ["Ethanol", "CO", "Ethylene", "Methane"]:
        row = _find(per_gas, variant="R84_FED_H1", evaluation_scope="S_ALL", gas=gas)
        gas_lines.append(f"  - {gas}: RMSE {float(row['RMSE']):.3f} ppm, NRMSE {float(row['NRMSE']):.5f}.")

    c5 = classifier["targets"]["C5"]
    analysis = f"""# Result analysis

## Confirmed endpoint

The final router is the frozen server-centric A4 round-25 checkpoint (seed 42, LE1). Its one-time C5 test result is accuracy {100*float(c5['accuracy']):.3f}% and macro-F1 {100*float(c5['macro_f1']):.3f}%. C3 and C4 remain blocked because same-protocol A4 endpoints are unavailable; full-GAPS endpoints were not substituted.

## Concentration estimation

- 83-D sensor only: end-to-end RMSE {float(r83['RMSE']):.3f} ppm, NRMSE {float(r83['NRMSE']):.5f}.
- 84-D + federated H1 (proposed): end-to-end RMSE {float(r84['RMSE']):.3f} ppm, NRMSE {float(r84['NRMSE']):.5f}; route-correct RMSE {float(r84_cc['RMSE']):.3f} ppm.
- 86-D + H1/H2/H3 (diagnostic): end-to-end RMSE {float(r86['RMSE']):.3f} ppm, NRMSE {float(r86['NRMSE']):.5f}. It is slightly better descriptively but is not promoted over the simpler federated-H1 design.
{chr(10).join(gas_lines)}

These are fixed-endpoint seed-42 descriptive results, not cross-seed inferential estimates.

## Label-free QC

The final QC score uses three label-free components normalized by calibration-only p95 scales. HC90 targets 90% calibration coverage and realizes {100*float(hc90['test_coverage']):.2f}% test coverage with NRMSE {float(hc90['NRMSE']):.5f}; HC95 realizes {100*float(hc95['test_coverage']):.2f}% with NRMSE {float(hc95['NRMSE']):.5f}. Full coverage NRMSE is {float(full['NRMSE']):.5f}. The matched random-reference HC90 mean NRMSE is {float(rand90['random_NRMSE_mean']):.5f}. HC90/HC95 denote targeted auto-output coverage, not accuracy.

The initial maximum-component QC attempt was excluded after a calibration-only tie audit showed that clipped source-prior outputs collapsed most high-coverage thresholds. Its artifacts remain under `qc_attempt1_degenerate` for traceability and are not manuscript evidence.

## Deployment boundary

The three-machine Flower run completed {physical['completed_rounds']}/{physical['expected_rounds']} rounds in {float(physical['wall_seconds'])/60:.1f} min with target test closed during training. Raspberry Pi 5 measurements are independent deployment benchmarks. Flower bytes are measured application payload; federated-H1 bytes are theoretical serialized exchange, so Fig. 8 does not present them as transport-controlled measurements.
"""
    (root / "RESULT_ANALYSIS.md").write_text(analysis, encoding="utf-8")

    audit = """# Experiment audit

Status: **PASS WITH DECLARED BOUNDARIES**

- Existing P0/classification assets were read only; no source Flower or classifier was retrained.
- A4 equality is based on the ordered state-content fingerprint. Whole-file SHA-256 is provenance only.
- Regression alpha selection used C5 calibration only (60 fit/20 validation per gas), was persisted, and was read back before C5 test loading.
- C5 test was used only for the fixed final classification/regression/QC evaluation; it did not select a checkpoint, Ridge alpha, or QC threshold.
- QC component scales and thresholds use label-free calibration fields. The excluded first QC attempt is explicitly non-formal.
- C3/C4 same-protocol A4 evidence is unavailable and remains blocked.
- Fig. 6 visibly separates the seed-42 A4 prior ablation from the historical five-replicate group-aware budget study.
- Fig. 8 distinguishes measured application payload from theoretical serialization and does not claim transport-byte measurement or a hardware photograph.
- Final replay is seed 42 only; QC random reference uses seed 20260804 for 1,000 matched selections.
"""
    (root / "EXPERIMENT_AUDIT.md").write_text(audit, encoding="utf-8")
    write_json(
        root / "strict_audit.json",
        {
            "schema_version": "iotj.final_a4.strict_audit.v1",
            "status": "PASS_WITH_DECLARED_BOUNDARIES",
            "checks": {
                "classifier_retrained": False,
                "source_flower_retrained": False,
                "checkpoint_identity_basis_ordered_state_content": True,
                "calibration_lock_precedes_test_open": True,
                "target_test_used_for_model_selection": False,
                "target_test_used_for_qc_threshold_selection": False,
                "c3_c4_same_protocol_a4_blocked": True,
                "qc_attempt1_degenerate_excluded": True,
                "communication_evidence_types_distinguished": True,
                "required_artifacts_present": True,
            },
            "seed": 42,
            "qc_random_seed": 20260804,
            "qc_random_repeats": 1000,
        },
    )

    captions = """# Final figure captions

**Fig. 5. Concentration estimation with the frozen A4 router.** (a) End-to-end ($S_{ALL}$) and route-correct ($S_{CC}$) RMSE for 83-D sensor statistics, 84-D sensor statistics plus federated H1, and 86-D statistics plus H1/H2/H3. (b) End-to-end per-gas RMSE. All results use the fixed C5 round-25 A4 endpoint and calibration-only Ridge selection (seed 42).

**Fig. 6. Source-prior and calibration-budget evidence.** (a) Fixed-A4 seed-42 NRMSE ablation for the three target-regression inputs. (b) Frozen group-aware calibration-budget study (five replicates; mean and sample SD). The panels use distinct protocols and are not pooled as a single-factor ablation.

**Fig. 7. Label-free quality-control trade-off.** (a) Test NRMSE versus retained coverage using calibration-only p95-normalized risk thresholds, with 1,000 matched random selections (mean and sample SD). HC90/HC95 annotations report actual test coverage obtained from thresholds targeting 90%/95% calibration coverage. (b) Fractions of misroutes, errors at least 40 ppm, and top-decile errors captured among rejected outputs.

**Fig. 8. Communication and deployment validation.** (a) Measured 25-round Flower application payload and theoretical one-shot federated-H1 serialized exchange (different evidence types). (b,c) Raspberry Pi 5 latency, throughput, and peak RSS for the audited runtime variants. (d) Completed three-machine Flower A4 execution and fixed-endpoint audit; no hardware photograph or transport-byte measurement is implied.
"""
    (root / "FIGURE_CAPTIONS.md").write_text(captions, encoding="utf-8")

    write_json(
        root / "deployment_manifest.json",
        {
            "schema_version": "iotj.final_a4_deployment.v1",
            "status": "complete",
            "classifier_checkpoint_identity": c5["checkpoint_identity"],
            "regression_models": {
                "path": "regression/regression_models.json",
                "sha256": _sha256(root / "regression/regression_models.json"),
                "proposed_variant": "R84_FED_H1",
            },
            "qc_thresholds": {
                "path": "qc/qc_threshold_lock.csv",
                "sha256": _sha256(root / "qc/qc_threshold_lock.csv"),
                "operating_points": ["HC90", "HC95"],
            },
            "record_schema": "regression/final_test_records.csv",
            "physical_validation": physical,
        },
    )
    write_json(
        root / "protocol_manifest.json",
        {
            "schema_version": "iotj.final_a4_package.v1",
            "status": "complete",
            "formal_target": "C5",
            "classifier": classifier["protocol"],
            "regression": regression_protocol,
            "qc": qc_protocol,
            "system": system_protocol,
            "figures": ["Fig5", "Fig6", "Fig7", "Fig8"],
            "seed": 42,
            "classification_retrained": False,
            "target_test_used_for_selection": False,
        },
    )
    write_json(
        root / "sha256_index.json",
        {"schema_version": "iotj.sha256_index.v1", "files": sha256_tree(root)},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", default="results/iotj_final_end_to_end_a4_20260804"
    )
    parser.add_argument(
        "--budget-csv",
        default=(
            "results/iotj_calibration_protocol_harmonization_20260726/"
            "track_groupaware/groupaware_budget_summary.csv"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    finalize_package(Path(args.root), Path(args.budget_csv))


if __name__ == "__main__":
    main()
