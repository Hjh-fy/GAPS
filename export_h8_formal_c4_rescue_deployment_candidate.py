"""Export H8 + formal C4 route-rescue deployment bundle.

This takes the existing H8 deployment bundle, preserves its original route-rescue
gate, and adds the calibration-selected formal C4 route-rescue gate as an
additional runtime gate.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


OUTPUT_FIELDS_V2 = [
    "gas_class",
    "gas_name",
    "class_prob",
    "base_r3ak16_raw_ppm",
    "routed_pred_ppm",
    "final_ppm",
    "co_corrected_ppm",
    "auto_output_ppm",
    "qc_decision",
    "risk_score",
]


def convert_gate(selected: dict[str, Any]) -> dict[str, Any]:
    """Convert the selector output into the deployment runtime gate schema.

    Keep every field that participates in ``run_formal_c4_route_rescue_selector``.
    In particular, ``max_conf_margin`` must be preserved; otherwise the exported
    runtime gate can become wider than the calibration-selected gate.
    """
    return {
        "candidate": "formal_c4_route_rescue_calibration_selected",
        "phase": str(selected.get("phase", "any")),
        "risk_threshold": float(selected.get("min_risk", 0.0)),
        "max_ppm": float(selected.get("max_final", 50.0)),
        "max_conf_margin": float(selected.get("max_conf_margin", 1.0)),
        "pred_classes": str(selected.get("pred_classes", "")),
        "rescue_ppm": float(selected.get("rescue_ppm", 250.0)),
        "selection_source": "target_calibration_only",
        "calibration_hit_N": int(selected.get("hit_N", 0)),
        "calibration_true_c4_high_hits": int(selected.get("true_c4_high_hits", 0)),
        "calibration_false_hits": int(selected.get("false_hits", 0)),
        "calibration_c4_high_recall": float(selected.get("calib_c4_high_recall", 0.0)),
    }


def patch_runtime_route_rescue_guard(runtime_rich_residual: Path) -> bool:
    """Patch exported runtime code so it enforces ``max_conf_margin``.

    The source ``rich_residual.py`` is intentionally large and shared by several
    candidates.  For this exporter we patch the copied runtime file in-place so
    the deployment bundle exactly matches the calibration selector semantics.
    """
    if not runtime_rich_residual.exists():
        return False
    text = runtime_rich_residual.read_text(encoding="utf-8")
    if "max_conf_margin" in text:
        return True
    anchor = (
        "        if float(result.risk_score) < float(gate.get(\"risk_threshold\", 0.0)):\n"
        "            return None\n"
        "        return float(gate.get(\"rescue_ppm\", result.final_ppm))\n"
    )
    replacement = (
        "        if float(result.risk_score) < float(gate.get(\"risk_threshold\", 0.0)):\n"
        "            return None\n"
        "        if float(getattr(result, \"confidence_margin\", 1.0)) > float(gate.get(\"max_conf_margin\", 1.0)):\n"
        "            return None\n"
        "        return float(gate.get(\"rescue_ppm\", result.final_ppm))\n"
    )
    if anchor not in text:
        raise RuntimeError(
            f"Cannot patch {runtime_rich_residual}: route-rescue guard anchor not found"
        )
    runtime_rich_residual.write_text(text.replace(anchor, replacement), encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Export H8 + formal C4 route-rescue deployment bundle.")
    parser.add_argument("--base-bundle", default="results/deployment_h8_source_aug_candidate_20260625")
    parser.add_argument("--selected-gate", default="results/formal_c4_route_rescue_selector_20260625/formal_c4_route_rescue_selected_gate.json")
    parser.add_argument("--output-dir", default="results/deployment_h8_formal_c4_rescue_candidate_20260625")
    args = parser.parse_args()

    base = Path(args.base_bundle)
    out = Path(args.output_dir)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(base, out)

    artifact_path = out / "rich_residual_candidate.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    existing_route_rescue = dict(artifact.get("route_rescue_policy", {}))
    selected = json.loads(Path(args.selected_gate).read_text(encoding="utf-8"))
    artifact["schema"] = str(artifact.get("schema", "gaps_hybrid_mlp_ridge_policy.v1+h8_source_aug.v1")) + "+formal_c4_route_rescue.v2"
    artifact["candidate_name"] = "c12_c345_h8_source_aug_plus_formal_c4_route_rescue"
    artifact["route_rescue_policy"] = {
        "schema": "c4_route_rescue_policy.v2",
        "selected_gate": existing_route_rescue.get("selected_gate"),
        "additional_gates": [convert_gate(selected)],
    }
    artifact.setdefault("source_files", {})
    artifact["source_files"].update(
        {
            "formal_c4_route_rescue_export_script": Path(__file__).name,
            "formal_c4_selected_gate": args.selected_gate,
            "base_bundle": args.base_bundle,
        }
    )
    artifact_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest["candidate_name"] = artifact["candidate_name"]
    manifest["base_bundle"] = args.base_bundle
    manifest["formal_c4_route_rescue_gate"] = args.selected_gate
    manifest["route_rescue_schema"] = artifact["route_rescue_policy"]["schema"]
    manifest["output_fields"] = OUTPUT_FIELDS_V2
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    runtime_config_path = out / "runtime_config.json"
    runtime_config = json.loads(runtime_config_path.read_text(encoding="utf-8"))
    runtime_config["output_fields"] = OUTPUT_FIELDS_V2
    runtime_config_path.write_text(json.dumps(runtime_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    runtime_rich_residual = out / "runtime_src" / "gaps_deploy" / "rich_residual.py"
    runtime_final = out / "runtime_src" / "gaps_deploy" / "final_runtime.py"
    runtime_guard_patched = False
    if runtime_rich_residual.exists():
        shutil.copy2(Path("gaps_deploy") / "rich_residual.py", runtime_rich_residual)
        runtime_guard_patched = patch_runtime_route_rescue_guard(runtime_rich_residual)
    if runtime_final.exists():
        shutil.copy2(Path("gaps_deploy") / "final_runtime.py", runtime_final)

    print(
        json.dumps(
            {
                "output_dir": str(out),
                "artifact": str(artifact_path),
                "candidate_name": artifact["candidate_name"],
                "selected_gate": artifact["route_rescue_policy"]["selected_gate"],
                "additional_gates": artifact["route_rescue_policy"]["additional_gates"],
                "runtime_guard_patched": runtime_guard_patched,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
