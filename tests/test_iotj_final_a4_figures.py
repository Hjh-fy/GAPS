from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from scripts.plot_iotj_final_a4_figures import plot_fig5, plot_fig6, plot_fig7, plot_fig8


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_plot_fig5_exports_ieee_width_pdf_and_600dpi_png(tmp_path: Path) -> None:
    variants = ["R83_TARGET_ONLY", "R84_FED_H1", "R86_ALL_PRIORS"]
    main = []
    per_gas = []
    for variant_index, variant in enumerate(variants):
        for scope_index, scope in enumerate(["S_ALL", "S_CC"]):
            main.append(
                {
                    "variant": variant,
                    "evaluation_scope": scope,
                    "N": 100,
                    "RMSE": 10 + variant_index + scope_index,
                    "NRMSE": 0.1,
                }
            )
        for gas_index, gas in enumerate(["Ethanol", "CO", "Ethylene", "Methane"]):
            per_gas.append(
                {
                    "variant": variant,
                    "evaluation_scope": "S_ALL",
                    "gas": gas,
                    "N": 25,
                    "RMSE": 8 + variant_index + gas_index,
                    "NRMSE": 0.1,
                }
            )
    main_path = tmp_path / "main.csv"
    gas_path = tmp_path / "gas.csv"
    _write_csv(main_path, main)
    _write_csv(gas_path, per_gas)

    png, pdf = plot_fig5(main_path, gas_path, tmp_path)

    assert png.is_file() and pdf.is_file()
    with Image.open(png) as image:
        assert image.width >= 4200
        assert image.info["dpi"][0] >= 599


def test_plot_fig6_keeps_prior_ablation_and_budget_protocols_separate(tmp_path: Path) -> None:
    main = []
    for variant_index, variant in enumerate(
        ["R83_TARGET_ONLY", "R84_FED_H1", "R86_ALL_PRIORS"]
    ):
        for scope in ["S_ALL", "S_CC"]:
            main.append(
                {
                    "variant": variant,
                    "evaluation_scope": scope,
                    "N": 100,
                    "RMSE": 15 - variant_index,
                    "NRMSE": 0.12 - variant_index * 0.01,
                }
            )
    budget = [
        {
            "track": "G",
            "nominal_budget": value,
            "replicates": 5,
            "S_ALL_RMSE_mean": mean,
            "S_ALL_RMSE_sample_std": std,
        }
        for value, mean, std in [(40, 40, 2), (80, 35, 1.5), (160, 30, 1), (320, 25, 0.5)]
    ]
    main_path = tmp_path / "main.csv"
    budget_path = tmp_path / "budget.csv"
    _write_csv(main_path, main)
    _write_csv(budget_path, budget)

    png, pdf = plot_fig6(main_path, budget_path, tmp_path)

    assert png.is_file() and pdf.is_file()
    with Image.open(png) as image:
        assert image.width >= 4200


def test_plot_fig7_renders_qc_random_reference_and_hc_points(tmp_path: Path) -> None:
    curve = []
    random = []
    for index, target in enumerate([0.70, 0.90, 0.95, 1.0]):
        coverage = target - 0.01 if target < 1 else 1.0
        curve.append(
            {
                "target_coverage": target,
                "test_coverage": coverage,
                "NRMSE": 0.06 + index * 0.005,
                "misroute_capture_rate": 0.8 - index * 0.2,
                "error_ge_40ppm_capture_rate": 0.7 - index * 0.15,
                "top10pct_error_capture_rate": 0.6 - index * 0.1,
            }
        )
        random.append(
            {
                "target_coverage": target,
                "test_coverage": coverage,
                "random_NRMSE_mean": 0.08,
                "random_NRMSE_sample_std": 0.003,
            }
        )
    curve_path = tmp_path / "curve.csv"
    random_path = tmp_path / "random.csv"
    _write_csv(curve_path, curve)
    _write_csv(random_path, random)

    png, pdf = plot_fig7(curve_path, random_path, tmp_path)

    assert png.is_file() and pdf.is_file()
    with Image.open(png) as image:
        assert image.width >= 4200


def test_plot_fig8_renders_system_and_physical_validation(tmp_path: Path) -> None:
    system = [
        {"record_type": "communication", "label": "Flower", "bytes": 17000000, "evidence_type": "measured"},
        {"record_type": "communication", "label": "H1", "bytes": 7000000, "evidence_type": "theoretical"},
    ]
    for index, runtime in enumerate(["RUNTIME_V4_FULL", "RUNTIME_V5_REGRESSION_CORE", "RUNTIME_V5_QC2_CANDIDATE"]):
        system.append(
            {
                "record_type": "pi5_runtime",
                "label": runtime,
                "pi_p50_ms": 5 - index * 0.5,
                "pi_p95_ms": 6 - index * 0.5,
                "pi_peak_rss_mib": 238 - index,
                "pi_throughput_windows_per_s": 210 + index * 20,
            }
        )
    physical = [
        {
            "status": "PASS",
            "completed_rounds": 25,
            "expected_rounds": 25,
            "seed": 42,
            "target": "C5",
            "wall_seconds": 4641,
        }
    ]
    system_path = tmp_path / "system.csv"
    physical_path = tmp_path / "physical.csv"
    _write_csv(system_path, system)
    _write_csv(physical_path, physical)

    png, pdf = plot_fig8(system_path, physical_path, tmp_path)

    assert png.is_file() and pdf.is_file()
    with Image.open(png) as image:
        assert image.width >= 4200
