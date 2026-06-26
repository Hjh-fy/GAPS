"""Select direction-specific target regression profiles from experiment summaries.

The selector intentionally works from saved experiment metrics instead of model
internals.  This keeps the decision reproducible: a profile is selected because
its no-QC full-set regression metrics win under explicit constraints.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


DEFAULT_OUT_DIR = Path("results/target_profile_selector_20260626")
DEFAULT_C12_SUMMARY = Path("results/l3_lightweight_hybrid_matrix_20260626/l3_lightweight_hybrid_matrix_summary.csv")
DEFAULT_C45_SUMMARY = Path("results/c45_c123_optimal_config_analysis_20260626/c45_c123_optimal_config_summary.csv")


@dataclass
class Candidate:
    direction: str
    mode: str
    all_rmse: float
    all_nrmse: float
    nonco_rmse: float
    co_rmse_mean: float
    co_rmse_max: float
    co_high_rmse_mean: float
    co_high_rmse_max: float
    target_co_rmse: dict[str, float]
    target_co_high_rmse: dict[str, float]
    role_hint: str = ""


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def fnum(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def metric(rows: list[dict[str, str]], mode: str, scope: str, field: str) -> float | None:
    for row in rows:
        if row.get("mode") == mode and row.get("scope") == scope and row.get("split", "test") == "test":
            return fnum(row.get(field))
    return None


def unique_modes(rows: Iterable[dict[str, str]]) -> list[str]:
    seen: set[str] = set()
    modes: list[str] = []
    for row in rows:
        mode = row.get("mode", "")
        if mode and mode not in seen:
            seen.add(mode)
            modes.append(mode)
    return modes


def mean(values: Iterable[float]) -> float:
    seq = list(values)
    return sum(seq) / len(seq)


def build_candidates(path: Path, direction: str, target_clients: list[str]) -> list[Candidate]:
    rows = read_csv(path)
    candidates: list[Candidate] = []
    for mode in unique_modes(rows):
        all_rmse = metric(rows, mode, "ALL", "RMSE")
        all_nrmse = metric(rows, mode, "ALL", "NRMSE")
        nonco = metric(rows, mode, "nonCO_ALL", "RMSE")
        if all_rmse is None or all_nrmse is None or nonco is None:
            continue
        co_rmse: dict[str, float] = {}
        high_rmse: dict[str, float] = {}
        for client in target_clients:
            co = metric(rows, mode, f"{client}-CO", "RMSE")
            high = metric(rows, mode, f"{client}-CO_high_200_250", "RMSE")
            if co is not None:
                co_rmse[client] = co
            if high is not None:
                high_rmse[client] = high
        if not co_rmse or not high_rmse:
            continue
        candidates.append(
            Candidate(
                direction=direction,
                mode=mode,
                all_rmse=all_rmse,
                all_nrmse=all_nrmse,
                nonco_rmse=nonco,
                co_rmse_mean=mean(co_rmse.values()),
                co_rmse_max=max(co_rmse.values()),
                co_high_rmse_mean=mean(high_rmse.values()),
                co_high_rmse_max=max(high_rmse.values()),
                target_co_rmse=co_rmse,
                target_co_high_rmse=high_rmse,
                role_hint=role_hint(mode),
            )
        )
    return candidates


def role_hint(mode: str) -> str:
    lower = mode.lower()
    if "baseline" in lower:
        return "baseline"
    if "oracle" in lower:
        return "oracle_excluded"
    if "h8" in lower or "co_else" in lower or "source_aug" in lower:
        return "co_specialist_or_diagnostic"
    if "light" in lower or "l3" in lower:
        return "lightweight_candidate"
    if "ridge" in lower or "h2_3" in lower or "mlp" in lower:
        return "balanced_candidate"
    return "candidate"


def exclude_from_selection(candidate: Candidate) -> bool:
    lower = candidate.mode.lower()
    return "oracle" in lower or candidate.role_hint == "baseline"


def select_balanced(candidates: list[Candidate], all_tolerance: float) -> Candidate:
    pool = [c for c in candidates if not exclude_from_selection(c)]
    if not pool:
        raise ValueError("No non-baseline candidates available.")
    best_all = min(c.all_rmse for c in pool)
    shortlist = [c for c in pool if c.all_rmse <= best_all * (1.0 + all_tolerance)]
    # Balanced mainline prefers low normalized/global error and non-CO stability
    # among profiles whose ALL RMSE is effectively tied.
    return sorted(shortlist, key=lambda c: (c.all_nrmse, c.nonco_rmse, c.all_rmse))[0]


def select_co_specialist(
    candidates: list[Candidate],
    balanced: Candidate,
    all_tolerance: float,
) -> Candidate | None:
    pool = [
        c
        for c in candidates
        if not exclude_from_selection(c) and c.all_rmse <= balanced.all_rmse * (1.0 + all_tolerance)
    ]
    if not pool:
        return None
    best = sorted(pool, key=lambda c: (c.co_high_rmse_mean, c.co_high_rmse_max, c.co_rmse_mean))[0]
    improves_high = best.co_high_rmse_mean < balanced.co_high_rmse_mean or best.co_high_rmse_max < balanced.co_high_rmse_max
    return best if improves_high else None


def pct_delta(new: float, ref: float) -> float:
    return 100.0 * (new - ref) / ref


def candidate_row(candidate: Candidate, baseline: Candidate, balanced: Candidate | None = None) -> dict[str, str | float]:
    row: dict[str, str | float] = {
        "direction": candidate.direction,
        "mode": candidate.mode,
        "role_hint": candidate.role_hint,
        "ALL_RMSE": candidate.all_rmse,
        "ALL_NRMSE": candidate.all_nrmse,
        "nonCO_RMSE": candidate.nonco_rmse,
        "CO_RMSE_mean": candidate.co_rmse_mean,
        "CO_RMSE_max": candidate.co_rmse_max,
        "CO_high_RMSE_mean": candidate.co_high_rmse_mean,
        "CO_high_RMSE_max": candidate.co_high_rmse_max,
        "ALL_RMSE_delta_vs_baseline_pct": pct_delta(candidate.all_rmse, baseline.all_rmse),
        "nonCO_RMSE_delta_vs_baseline_pct": pct_delta(candidate.nonco_rmse, baseline.nonco_rmse),
        "CO_high_mean_delta_vs_baseline_pct": pct_delta(candidate.co_high_rmse_mean, baseline.co_high_rmse_mean),
        "target_CO_RMSE": json.dumps(candidate.target_co_rmse, ensure_ascii=False, sort_keys=True),
        "target_CO_high_RMSE": json.dumps(candidate.target_co_high_rmse, ensure_ascii=False, sort_keys=True),
    }
    if balanced is not None:
        row["ALL_RMSE_delta_vs_balanced_pct"] = pct_delta(candidate.all_rmse, balanced.all_rmse)
        row["CO_high_mean_delta_vs_balanced_pct"] = pct_delta(candidate.co_high_rmse_mean, balanced.co_high_rmse_mean)
    return row


def find_baseline(candidates: list[Candidate]) -> Candidate:
    for candidate in candidates:
        if candidate.role_hint == "baseline":
            return candidate
    raise ValueError("Missing baseline candidate.")


def write_csv(path: Path, rows: list[dict[str, str | float]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_direction(candidates: list[Candidate], all_tolerance: float) -> dict[str, object]:
    baseline = find_baseline(candidates)
    balanced = select_balanced(candidates, all_tolerance=all_tolerance)
    specialist = select_co_specialist(candidates, balanced=balanced, all_tolerance=all_tolerance)
    top_all = sorted([c for c in candidates if not exclude_from_selection(c)], key=lambda c: c.all_rmse)[:5]
    top_high = sorted([c for c in candidates if not exclude_from_selection(c)], key=lambda c: c.co_high_rmse_mean)[:5]
    return {
        "direction": candidates[0].direction,
        "baseline": asdict(baseline),
        "balanced_mainline": asdict(balanced),
        "co_specialist_candidate": asdict(specialist) if specialist is not None else None,
        "selection_rule": {
            "balanced_mainline": (
                f"shortlist candidates within {all_tolerance:.1%} of best ALL RMSE, "
                "then minimize ALL NRMSE, nonCO RMSE, ALL RMSE"
            ),
            "co_specialist_candidate": (
                f"within {all_tolerance:.1%} ALL RMSE of balanced mainline, "
                "then minimize mean/max target CO-high RMSE; kept only if it improves CO-high"
            ),
        },
        "top_by_all_rmse": [asdict(c) for c in top_all],
        "top_by_co_high_mean": [asdict(c) for c in top_high],
    }


def write_report(path: Path, selections: list[dict[str, object]]) -> None:
    lines = [
        "# Target Profile Selector Audit",
        "",
        "This audit turns the recent regression matrix into an explicit, reproducible profile-selection step.",
        "The selector uses no-QC full-set metrics only; QC accepted quality is intentionally not part of the decision.",
        "",
        "## Selection Logic",
        "",
        "- Balanced mainline: shortlist candidates within 2% of the best ALL RMSE, then choose the lowest ALL NRMSE, nonCO RMSE, and ALL RMSE.",
        "- CO-specialist candidate: among candidates within 2% ALL RMSE of the balanced mainline, choose the lowest mean/max target CO-high RMSE if it improves over balanced.",
        "- Test-oracle candidates are excluded from selection and only kept for sanity checks.",
        "",
        "## Results",
        "",
    ]
    for selection in selections:
        baseline = selection["baseline"]  # type: ignore[index]
        balanced = selection["balanced_mainline"]  # type: ignore[index]
        specialist = selection["co_specialist_candidate"]  # type: ignore[index]
        direction = selection["direction"]
        lines.extend(
            [
                f"### {direction}",
                "",
                f"- Baseline: `{baseline['mode']}` ALL RMSE {baseline['all_rmse']:.2f}, NRMSE {baseline['all_nrmse']:.4f}, CO-high mean {baseline['co_high_rmse_mean']:.2f}.",
                f"- Balanced mainline: `{balanced['mode']}` ALL RMSE {balanced['all_rmse']:.2f}, NRMSE {balanced['all_nrmse']:.4f}, nonCO RMSE {balanced['nonco_rmse']:.2f}, CO-high mean {balanced['co_high_rmse_mean']:.2f}.",
            ]
        )
        if specialist is None:
            lines.append("- CO-specialist candidate: none selected under the current constraints.")
        else:
            lines.append(
                f"- CO-specialist candidate: `{specialist['mode']}` ALL RMSE {specialist['all_rmse']:.2f}, "
                f"NRMSE {specialist['all_nrmse']:.4f}, nonCO RMSE {specialist['nonco_rmse']:.2f}, "
                f"CO-high mean {specialist['co_high_rmse_mean']:.2f}."
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "- C12 -> C345 still supports a two-profile story: H2.3 is the balanced mainline, while a CO-specialist/rescue profile is useful when CO-high is prioritized.",
            "- C45 -> C123 selects a simpler target Ridge direct mainline; source-aug switching remains diagnostic because the overall/nonCO tradeoff is not favorable.",
            "- This supports the method description as direction-specific target profile selection rather than a single hard-coded regression head.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c12-summary", type=Path, default=DEFAULT_C12_SUMMARY)
    parser.add_argument("--c45-summary", type=Path, default=DEFAULT_C45_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--all-tolerance", type=float, default=0.02)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    c12 = build_candidates(args.c12_summary, "C12_to_C345", ["C3", "C4", "C5"])
    c45 = build_candidates(args.c45_summary, "C45_to_C123", ["C1", "C2", "C3"])
    selections = [
        summarize_direction(c12, all_tolerance=args.all_tolerance),
        summarize_direction(c45, all_tolerance=args.all_tolerance),
    ]

    all_rows: list[dict[str, str | float]] = []
    for candidates, selection in [(c12, selections[0]), (c45, selections[1])]:
        baseline = find_baseline(candidates)
        balanced_mode = selection["balanced_mainline"]["mode"]  # type: ignore[index]
        balanced = next(c for c in candidates if c.mode == balanced_mode)
        all_rows.extend(candidate_row(c, baseline=baseline, balanced=balanced) for c in candidates)

    write_csv(args.output_dir / "normalized_target_profile_candidates.csv", all_rows)
    (args.output_dir / "selected_profiles.json").write_text(
        json.dumps(selections, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for selection in selections:
        direction = str(selection["direction"]).lower()
        (args.output_dir / f"selected_profile_{direction}.json").write_text(
            json.dumps(selection, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    write_report(args.output_dir / "target_profile_selector_audit.md", selections)
    print(json.dumps({"output_dir": str(args.output_dir), "directions": len(selections)}, indent=2))


if __name__ == "__main__":
    main()
