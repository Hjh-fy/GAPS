# Lab A4 Cross-Board and Full-Scope Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the frozen laboratory A4 three-gas workflow to measure the existing P2→P3 checkpoint on early/full windows and run seed-42 P2→P1, P1→P3, and P1+P2→P3 stable-training experiments without changing the model or DA algorithm.

**Architecture:** Add a small Python direction-contract module shared by dataset, validation, and summary code; build one immutable cross-board dataset root per direction with standard `test_*` files for stable360 plus named `early_*` and `full_*` evaluation files. Parameterize the existing three-node controller and postflight validator so source clients and target client come from the direction contract, while preserving the existing P2→P3 behavior. Evaluate each fixed round-25 adapted checkpoint over stable/early/full scopes with a separate read-only evaluator.

**Tech Stack:** Python 3.10+, NumPy, pandas/csv, PyTorch, Flower, pytest, PowerShell 5+, SSH/SCP, JSON/CSV manifests.

## Global Constraints

- Existing A4 datasets, checkpoints, summaries, audits, runtime packages, and the reported `359/360 = 99.72%` result are read-only.
- Use relative resistance channels `CH1/2/4/6/8/9`, input shape `[B,100,6]`, `proto_replay`, `corrected_b2`, target CE weight `0`, 25 Flower rounds, local epochs `3`, batch size `32`, DA steps `100`, and seed `42`.
- Select fixed round 25; source calibration is monitoring only; target test never participates in training, DA, selection, stopping, or threshold tuning.
- Calibration indices are `3,11,19`; purge indices are `2,4,10,12,18,20`; early indices are `0,1`; stable indices are `5–9,13–17,21–22`.
- Single-source normalization uses only that source's stable train; P1+P2 normalization uses pooled P1+P2 stable train; no target statistics are allowed.
- Exclude all P1 limonene exposures. The task remains acetaldehyde/methane/acetic-acid classification only.
- Report stable360, early60, and full420 metrics from the same round-25 checkpoint. Treat all cross-direction comparisons as seed-42 descriptive evidence.
- P1+P2→P3 is an operational full-data comparison; do not attribute its result solely to diversity because no matched-budget arm is planned.
- Use new dataset, runtime, run, evaluation, and result paths; refuse overwrite and preserve failed attempts.
- Three-node roles remain Server A `root@121.40.139.213`, Cloud B `root@114.55.171.63`, and Pi `gaps@192.168.137.172`.

---

## File Structure

**Create**

- `scripts/lab_three_gas_3class/crossboard_a4_contract.py` — canonical direction, source-client, target-client, and scope-index contract.
- `scripts/lab_three_gas_3class/build_crossboard_a4_dataset.py` — direction-parameterized A4 dataset builder.
- `scripts/lab_three_gas_3class/validate_crossboard_a4_dataset.py` — structural, normalization, overlap, and identity validator.
- `scripts/lab_three_gas_3class/evaluate_crossboard_scopes.py` — evaluate one frozen checkpoint on stable/early/full named splits.
- `scripts/lab_three_gas_3class/run_a4_crossboard_queue.ps1` — sequential E1/E2/E3 launcher.
- `scripts/lab_three_gas_3class/summarize_a4_crossboard_results.py` — combine E0/E1/E2/E3 with the existing A4 baseline.
- `tests/test_lab_three_gas_crossboard_contract.py` — direction-contract tests.
- `tests/test_lab_three_gas_crossboard_dataset.py` — dataset and normalization tests.
- `tests/test_lab_three_gas_crossboard_scopes.py` — named-scope evaluator tests.
- `tests/test_lab_three_gas_crossboard_controller.py` — controller rendering and target-role tests.
- `docs/experiments/lab_3gas_a4_crossboard_execution_20260731.md` — immutable execution protocol and claim boundary.
- `docs/experiments/lab_3gas_a4_crossboard_matrix_20260731.csv` — one canonical row per E0–E3 configuration.

**Modify**

- `scripts/lab_three_gas_3class/run_lab_three_node_fold.ps1` — add P1→P3 and P2→P1 and make target/client launch logic dynamic.
- `scripts/lab_three_gas_3class/validate_three_node_run.py` — accept explicit source and target clients instead of assuming target C3.
- `scripts/lab_three_gas_3class/freeze_three_node_protocol.py` — freeze all approved directions and dynamic target identity.
- `scripts/lab_three_gas_3class/README.md` — document cross-board A4 commands and output semantics.
- `tests/test_lab_three_gas_validate_three_node_run.py` — dynamic target scope and identity coverage.

---

### Task 1: Freeze the Direction and Scope Contract

**Files:**
- Create: `scripts/lab_three_gas_3class/crossboard_a4_contract.py`
- Create: `tests/test_lab_three_gas_crossboard_contract.py`

**Interfaces:**
- Produces: `DirectionSpec(name: str, source_clients: tuple[int, ...], target_client: int)`.
- Produces: `resolve_direction(name: str) -> DirectionSpec`.
- Produces: `CALIBRATION_INDICES`, `PURGED_INDICES`, `EARLY_INDICES`, `STABLE_INDICES`, and `FULL_INDICES` tuples.
- Consumed by: Tasks 2, 3, 5, 6, and 7.

- [ ] **Step 1: Write the failing contract tests**

```python
from scripts.lab_three_gas_3class.crossboard_a4_contract import (
    EARLY_INDICES,
    FULL_INDICES,
    STABLE_INDICES,
    resolve_direction,
)


def test_approved_direction_roles() -> None:
    assert resolve_direction("P2_to_P1").source_clients == (2,)
    assert resolve_direction("P2_to_P1").target_client == 1
    assert resolve_direction("P1_to_P3").source_clients == (1,)
    assert resolve_direction("P1_to_P3").target_client == 3
    assert resolve_direction("P12_to_P3").source_clients == (1, 2)
    assert resolve_direction("P12_to_P3").target_client == 3


def test_scope_partition_contract() -> None:
    assert EARLY_INDICES == (0, 1)
    assert STABLE_INDICES == (5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 21, 22)
    assert FULL_INDICES == EARLY_INDICES + STABLE_INDICES
    assert len(FULL_INDICES) == 14
```

- [ ] **Step 2: Run the tests and verify the module is missing**

Run: `python -m pytest tests/test_lab_three_gas_crossboard_contract.py -q`

Expected: collection fails with `ModuleNotFoundError` for `crossboard_a4_contract`.

- [ ] **Step 3: Implement the immutable contract**

```python
from dataclasses import dataclass

CALIBRATION_INDICES = (3, 11, 19)
PURGED_INDICES = (2, 4, 10, 12, 18, 20)
EARLY_INDICES = (0, 1)
STABLE_INDICES = (5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 21, 22)
FULL_INDICES = EARLY_INDICES + STABLE_INDICES


@dataclass(frozen=True)
class DirectionSpec:
    name: str
    source_clients: tuple[int, ...]
    target_client: int


DIRECTIONS = {
    "P2_to_P3": DirectionSpec("P2_to_P3", (2,), 3),
    "P2_to_P1": DirectionSpec("P2_to_P1", (2,), 1),
    "P1_to_P3": DirectionSpec("P1_to_P3", (1,), 3),
    "P12_to_P3": DirectionSpec("P12_to_P3", (1, 2), 3),
}


def resolve_direction(name: str) -> DirectionSpec:
    try:
        return DIRECTIONS[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported direction: {name}") from exc
```

- [ ] **Step 4: Run the focused and existing index-contract tests**

Run: `python -m pytest tests/test_lab_three_gas_crossboard_contract.py tests/test_lab_three_gas_allconc_timepurged.py tests/test_lab_three_gas_recovery_variants.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit the contract**

```powershell
git add scripts/lab_three_gas_3class/crossboard_a4_contract.py tests/test_lab_three_gas_crossboard_contract.py
git commit -m "feat(lab): add A4 cross-board direction contract"
```

---

### Task 2: Build Direction-Parameterized A4 Datasets

**Files:**
- Create: `scripts/lab_three_gas_3class/build_crossboard_a4_dataset.py`
- Create: `tests/test_lab_three_gas_crossboard_dataset.py`

**Interfaces:**
- Consumes: `DirectionSpec` and scope tuples from Task 1.
- Consumes: `BuildConfig`, `build_exposure_records`, `concatenate_records`, `discover_sessions`, `save_split`, and `write_csv` from `build_fivefold_dataset.py`.
- Produces: `build_from_records(records: list[dict], config: BuildConfig, direction: str, output_root: Path) -> dict` for deterministic unit tests and real-data assembly after session parsing.
- Produces: `build_dataset(config: BuildConfig, direction: str, overwrite: bool = False) -> dict`.
- Produces per target client: standard `test_*` = stable360 plus `early_*` and `full_*` arrays/manifests.

- [ ] **Step 1: Write synthetic-record tests for roles, counts, and normalization**

```python
def _config(output_root: Path, normalization_clients: tuple[int, ...]) -> BuildConfig:
    return BuildConfig(
        raw_root="unused",
        output_root=str(output_root),
        normalization_clients=normalization_clients,
        selected_channels=(1, 2, 4, 6, 8, 9),
    )


def _synthetic_records(exposures_per_platform: int = 30) -> list[dict]:
    records = []
    for platform in (1, 2, 3):
        for exposure_index in range(exposures_per_platform):
            gas_label = exposure_index % 3
            exposure_id = f"P{platform}_E{exposure_index:02d}"
            features = np.full(
                (23, 100, 6),
                platform * 100 + exposure_index,
                dtype=np.float32,
            )
            records.append({
                "platform": platform,
                "gas_label": gas_label,
                "features": features,
                "window_rows": [
                    {
                        "exposure_id": exposure_id,
                        "platform": platform,
                        "gas_label": gas_label,
                        "base_window_index": index,
                    }
                    for index in range(23)
                ],
            })
    return records


def test_p2_to_p1_uses_p2_only_normalization(tmp_path: Path) -> None:
    summary = build_from_records(
        records=_synthetic_records(),
        config=_config(tmp_path),
        direction="P2_to_P1",
        output_root=tmp_path,
    )
    stats = np.load(tmp_path / "fold_1" / "norm_stats.npz")
    assert summary["source_clients"] == [2]
    assert summary["target_client"] == 1
    assert stats["normalization_clients"].tolist() == [2]


def test_target_named_scopes_have_exact_counts(tmp_path: Path) -> None:
    build_from_records(
        records=_synthetic_records(exposures_per_platform=30),
        config=_config(tmp_path),
        direction="P1_to_P3",
        output_root=tmp_path,
    )
    target = tmp_path / "fold_1" / "client_3"
    assert len(np.load(target / "test_features.npy")) == 360
    assert len(np.load(target / "early_features.npy")) == 60
    assert len(np.load(target / "full_features.npy")) == 420
```

The synthetic fixture must create 23 windows per exposure, three balanced gas labels, only platforms 1–3, and manifest rows with `exposure_id`, `platform`, `gas_label`, and `base_window_index`.

- [ ] **Step 2: Run the dataset tests and verify the builder is missing**

Run: `python -m pytest tests/test_lab_three_gas_crossboard_dataset.py -q`

Expected: collection fails because `build_crossboard_a4_dataset.py` does not exist.

- [ ] **Step 3: Implement source/target record selection and named-split saving**

```python
def select_platform_scope(records, platform, indices, role):
    return [
        subset_record(record, indices, role)
        for record in records
        if int(record["platform"]) == platform
    ]


def save_named_split(client_dir, prefix, records, mean, std):
    return save_split(client_dir, prefix, records, mean, std)
```

The builder must calculate `mean/std` from the concatenation of stable train records for `spec.source_clients` only, store `normalization_clients` inside `norm_stats.npz`, save source `train/calibration/test` where source `test` is a compatibility alias of calibration, and save target `train/calibration` as calibration aliases plus `test`, `early`, and `full` evaluation files.

- [ ] **Step 4: Add manifests and fail-closed invariants**

The builder must raise before writing results when source and target overlap, a client is outside `{1,2,3}`, any exposure has other than 23 base windows, a selected scope is empty, or the output directory is non-empty without `--overwrite`. `fold_config.json` must record direction, source clients, target client, all scope indices, sample counts, normalization scope, and `target_test_open_after_fixed_round_selection=true`.

- [ ] **Step 5: Run focused dataset tests**

Run: `python -m pytest tests/test_lab_three_gas_crossboard_dataset.py tests/test_lab_three_gas_source_only_normalization.py tests/test_lab_three_gas_allconc_timepurged.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit the dataset builder**

```powershell
git add scripts/lab_three_gas_3class/build_crossboard_a4_dataset.py tests/test_lab_three_gas_crossboard_dataset.py
git commit -m "feat(lab): build cross-board A4 datasets"
```

---

### Task 3: Validate Cross-Board Dataset Identity and Leakage Boundaries

**Files:**
- Create: `scripts/lab_three_gas_3class/validate_crossboard_a4_dataset.py`
- Modify: `tests/test_lab_three_gas_crossboard_dataset.py`

**Interfaces:**
- Consumes: dataset roots produced by Task 2.
- Produces: `validate_dataset(root: Path, expected_direction: str) -> dict`.
- Produces: `validation_report.json` with `ok`, counts, hashes, normalization clients, scope identities, class counts, exposure counts, and overlap findings.

- [ ] **Step 1: Add failing validator tests**

```python
def test_validator_proves_full_is_early_plus_stable(tmp_path: Path) -> None:
    root = tmp_path / "p12p3"
    build_from_records(
        _synthetic_records(),
        _config(root, (1, 2)),
        "P12_to_P3",
        root,
    )
    report = validate_dataset(root, "P12_to_P3")
    assert report["ok"] is True
    assert report["target_scopes"] == {
        "stable": {"windows": 360, "exposures": 30},
        "early": {"windows": 60, "exposures": 30},
        "full": {"windows": 420, "exposures": 30},
    }


def test_validator_rejects_target_normalization_client(tmp_path: Path) -> None:
    root = tmp_path / "p2p1"
    build_from_records(
        _synthetic_records(),
        _config(root, (2,)),
        "P2_to_P1",
        root,
    )
    stats_path = root / "fold_1" / "norm_stats.npz"
    stats = np.load(stats_path)
    np.savez(
        stats_path,
        mean=stats["mean"],
        std=stats["std"],
        selected_channels=stats["selected_channels"],
        normalization_clients=np.asarray([1, 2], dtype=np.int64),
    )
    with pytest.raises(AssertionError, match="normalization clients"):
        validate_dataset(root, "P2_to_P1")
```

- [ ] **Step 2: Run the validator tests and verify failure**

Run: `python -m pytest tests/test_lab_three_gas_crossboard_dataset.py -q`

Expected: tests fail because `validate_dataset` is unavailable.

- [ ] **Step 3: Implement structural and hash validation**

The validator must compare manifest identity tuples `(exposure_id, base_window_index)` so that stable and early are disjoint and their union equals full; verify calibration raw intervals do not overlap any active target test interval; verify each scope is class-balanced; verify only the three approved Chinese gas names occur; and verify source-client arrays and target-client arrays match `fold_config.json`.

- [ ] **Step 4: Add E0 byte-equivalence checks**

For direction `P2_to_P3`, accept `--reference-a4-root` and require exact SHA256 equality for `norm_stats.npz`, C2 stable train/calibration files, C3 calibration files, and C3 stable `test_*` files. Early/full files are new and are validated by their manifests rather than compared to the stable reference.

- [ ] **Step 5: Run all dataset validation tests**

Run: `python -m pytest tests/test_lab_three_gas_crossboard_dataset.py tests/test_lab_three_gas_recovery_variants.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit the validator**

```powershell
git add scripts/lab_three_gas_3class/validate_crossboard_a4_dataset.py tests/test_lab_three_gas_crossboard_dataset.py
git commit -m "test(lab): validate A4 cross-board datasets"
```

---

### Task 4: Evaluate Stable, Early, and Full Scopes from One Checkpoint

**Files:**
- Create: `scripts/lab_three_gas_3class/evaluate_crossboard_scopes.py`
- Create: `tests/test_lab_three_gas_crossboard_scopes.py`

**Interfaces:**
- Consumes: one checkpoint, one dataset fold root, one target client, and named prefixes `test`, `early`, `full`.
- Produces: `evaluate_scopes(checkpoint: Path, data_root: Path, target_client: int, device: str, output_dir: Path) -> dict`.
- Produces: `validate_scope_counts(counts: dict[str, int]) -> None` requiring exactly `stable=360`, `early=60`, and `full=420`.
- Produces: `scope_evaluation_summary.json` plus one JSON per scope.

- [ ] **Step 1: Write failing tests for named-split loading and aggregation**

```python
def test_scope_prefix_mapping() -> None:
    assert scope_prefix("stable") == "test"
    assert scope_prefix("early") == "early"
    assert scope_prefix("full") == "full"


def test_scope_summary_requires_360_60_420() -> None:
    validate_scope_counts({"stable": 360, "early": 60, "full": 420})


def test_scope_summary_rejects_missing_early_rows() -> None:
    with pytest.raises(ValueError, match="early expected 60, got 59"):
        validate_scope_counts({"stable": 360, "early": 59, "full": 419})
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_lab_three_gas_crossboard_scopes.py -q`

Expected: collection fails because the scope evaluator does not exist.

- [ ] **Step 3: Implement a named classification-only loader**

```python
def load_named_arrays(client_dir: Path, prefix: str):
    features = np.load(client_dir / f"{prefix}_features.npy")
    labels = np.load(client_dir / f"{prefix}_classification_labels.npy")
    phases = np.load(client_dir / f"{prefix}_phase_labels.npy")
    manifest = load_manifest(client_dir / f"{prefix}_window_manifest.csv")
    return features, labels, phases, manifest
```

Construct `GasSensorWindowDataset(..., normalize=False)` because Task 2 saves already normalized arrays. Load the checkpoint with `load_checkpoint_model`, call the existing `predict`, `classification_metrics`, and `exposure_metrics`, and refuse a non-empty output directory.

- [ ] **Step 4: Enforce the fixed-checkpoint and scope contract**

Require checkpoint round `25`, target client matching `fold_config.json`, stable/early/full counts `360/60/420`, and full `(exposure_id, base_window_index)` identities equal to the union of stable and early identities. Store checkpoint SHA256, normalization SHA256, target client, scope counts, and metrics in the summary.

- [ ] **Step 5: Run evaluator tests and compile checks**

Run: `python -m pytest tests/test_lab_three_gas_crossboard_scopes.py tests/test_lab_three_gas_recovery_variants.py -q`

Run: `python -m compileall -q scripts/lab_three_gas_3class/evaluate_crossboard_scopes.py`

Expected: both commands pass.

- [ ] **Step 6: Commit the scope evaluator**

```powershell
git add scripts/lab_three_gas_3class/evaluate_crossboard_scopes.py tests/test_lab_three_gas_crossboard_scopes.py
git commit -m "feat(lab): evaluate A4 early and full scopes"
```

---

### Task 5: Parameterize Postflight for Dynamic Target Clients

**Files:**
- Modify: `scripts/lab_three_gas_3class/validate_three_node_run.py`
- Modify: `tests/test_lab_three_gas_validate_three_node_run.py`

**Interfaces:**
- Consumes: explicit `--source-clients` CSV and `--target-client` integer from Task 6.
- Produces: postflight audit containing exact source clients, target client, and target stable scope.
- Produces: `require_evaluation_identity(evaluation: dict, source_clients: list[int], target_client: int, selected_round: int) -> None`.
- Preserves: existing `expected_target_scope(target_data_dir)` behavior for standard calibration/test files.

- [ ] **Step 1: Add failing dynamic-target tests**

```python
def test_validate_evaluation_accepts_target_client_one() -> None:
    evaluation = {
        "source_clients": [2],
        "target_client": 1,
        "selected_round": 25,
    }
    require_evaluation_identity(
        evaluation=evaluation,
        source_clients=[2],
        target_client=1,
        selected_round=25,
    )
```

- [ ] **Step 2: Run the focused test and verify the fixed-C3 assertion fails**

Run: `python -m pytest tests/test_lab_three_gas_validate_three_node_run.py -q`

Expected: failure contains `Target must be P3/client 3` or an incompatible signature error.

- [ ] **Step 3: Add explicit source/target CLI arguments and validation**

Add `--source-clients` using the existing CSV parser pattern and `--target-client` with choices `1,3`. Replace direction-derived source clients and every literal target `3` in validation output with the explicit arguments. Keep `--direction` as recorded metadata and require it to agree with `resolve_direction`.

- [ ] **Step 4: Run postflight tests**

Run: `python -m pytest tests/test_lab_three_gas_validate_three_node_run.py tests/test_lab_three_gas_flower_eval_split.py -q`

Expected: all tests pass, including existing P2→P3 behavior.

- [ ] **Step 5: Commit dynamic postflight validation**

```powershell
git add scripts/lab_three_gas_3class/validate_three_node_run.py tests/test_lab_three_gas_validate_three_node_run.py
git commit -m "feat(lab): validate dynamic cross-board targets"
```

---

### Task 6: Parameterize the Three-Node Controller

**Files:**
- Modify: `scripts/lab_three_gas_3class/run_lab_three_node_fold.ps1`
- Create: `tests/test_lab_three_gas_crossboard_controller.py`

**Interfaces:**
- Consumes: `-Direction P2_to_P1|P1_to_P3|P12_to_P3|P2_to_P3`.
- Produces: dynamic `$sourceClients`, `$targetClient`, source launch set, target preflight path, server `--target-clients`, evaluator `--target-client`, and postflight arguments.
- Produces: `-ContractOnly` JSON without SSH, filesystem mutation, or process launch.

- [ ] **Step 1: Write failing controller-contract tests**

```python
CONTROLLER = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "lab_three_gas_3class"
    / "run_lab_three_node_fold.ps1"
)


def render_contract(direction: str) -> dict:
    completed = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-File", str(CONTROLLER),
            "-Direction", direction,
            "-ContractOnly",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_p2_to_p1_contract_starts_only_cloud_b() -> None:
    contract = render_contract("P2_to_P1")
    assert contract["source_clients"] == [2]
    assert contract["target_client"] == 1
    assert contract["launch_pi"] is False
    assert contract["launch_cloud_b"] is True
```

Add equivalent assertions for P1→P3, P1+P2→P3, and the unchanged P2→P3 contract.

- [ ] **Step 2: Run controller tests and verify unsupported directions fail**

Run: `python -m pytest tests/test_lab_three_gas_crossboard_controller.py -q`

Expected: PowerShell rejects P2→P1/P1→P3 or does not recognize `ContractOnly`.

- [ ] **Step 3: Implement direction mapping and contract-only output**

```powershell
$directionMap = @{
    P2_to_P3  = @{ SourceClients = @(2);    TargetClient = 3 }
    P2_to_P1  = @{ SourceClients = @(2);    TargetClient = 1 }
    P1_to_P3  = @{ SourceClients = @(1);    TargetClient = 3 }
    P12_to_P3 = @{ SourceClients = @(1, 2); TargetClient = 3 }
}
$sourceClients = @($directionMap[$Direction].SourceClients)
$targetClient = [int]$directionMap[$Direction].TargetClient
```

`-ContractOnly` must emit compressed JSON with direction, source clients, target client, launch booleans, server/client data paths, and run ID, then return before creating output directories.

- [ ] **Step 4: Make preflight, launch, cleanup, evaluation, and audit dynamic**

Launch Pi only when source contains C1, launch Cloud B only when source contains C2, and never launch a target-only physical client. Server target preflight and DA use `client_$targetClient`. Pass explicit source and target arguments to the evaluator and Task-5 validator. Cleanup patterns must remain exact run-tag matches.

- [ ] **Step 5: Run controller and postflight regressions**

Run: `python -m pytest tests/test_lab_three_gas_crossboard_controller.py tests/test_lab_three_gas_validate_three_node_run.py -q`

Expected: all four direction contracts pass and P2→P3 rendering remains unchanged except for explicit target metadata.

- [ ] **Step 6: Commit controller parameterization**

```powershell
git add scripts/lab_three_gas_3class/run_lab_three_node_fold.ps1 tests/test_lab_three_gas_crossboard_controller.py
git commit -m "feat(lab): run dynamic three-node directions"
```

---

### Task 7: Freeze Protocols, Queue Runs, and Summarize Results

**Files:**
- Modify: `scripts/lab_three_gas_3class/freeze_three_node_protocol.py`
- Create: `scripts/lab_three_gas_3class/run_a4_crossboard_queue.ps1`
- Create: `scripts/lab_three_gas_3class/summarize_a4_crossboard_results.py`
- Create: `docs/experiments/lab_3gas_a4_crossboard_execution_20260731.md`
- Create: `docs/experiments/lab_3gas_a4_crossboard_matrix_20260731.csv`
- Modify: `scripts/lab_three_gas_3class/README.md`
- Modify: `tests/test_lab_three_gas_crossboard_contract.py`

**Interfaces:**
- Consumes: approved directions and controller from Tasks 1 and 6.
- Produces: frozen source archive, per-direction dataset manifest, topology manifest, protocol manifest, queue logs, and one combined JSON/CSV summary.

- [ ] **Step 1: Add failing freeze/queue contract tests**

```python
MATRIX = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "experiments"
    / "lab_3gas_a4_crossboard_matrix_20260731.csv"
)


def test_matrix_has_exact_approved_rows() -> None:
    rows = list(csv.DictReader(MATRIX.open(encoding="utf-8-sig")))
    assert [row["experiment_id"] for row in rows] == [
        "A4-XB-E0-FULL420",
        "A4-XB-E1-P2P1-S42",
        "A4-XB-E2-P1P3-S42",
        "A4-XB-E3-P12P3-S42",
    ]
```

Also assert seed `42`, rounds `25`, local epochs `3`, target CE `0`, checkpoint policy `last_round`, and no matched-budget row.

- [ ] **Step 2: Run the contract test and verify files are absent**

Run: `python -m pytest tests/test_lab_three_gas_crossboard_contract.py -q`

Expected: failure because the execution matrix does not exist.

- [ ] **Step 3: Extend protocol freezing without changing old manifests**

Allow all four direction names, derive exact source/target roles from Task 1, and add `scope_evaluation = {stable: 360, early: 60, full: 420}`. Continue exclusive writes and content-addressed source archives. Do not mutate any existing protocol directory.

- [ ] **Step 4: Implement the three-run sequential queue**

The PowerShell queue must call the controller in this exact order and stop on the first non-zero exit:

```powershell
$runs = @(
    @{ Direction = "P2_to_P1"; Dataset = "client_data_lab_3gas_a4_crossboard_p2p1_v1"; Label = "a4_xb_e1" },
    @{ Direction = "P1_to_P3"; Dataset = "client_data_lab_3gas_a4_crossboard_p1p3_v1"; Label = "a4_xb_e2" },
    @{ Direction = "P12_to_P3"; Dataset = "client_data_lab_3gas_a4_crossboard_p12p3_v1"; Label = "a4_xb_e3" }
)
```

Each call passes rounds 25, local epochs 3, seed 42, profile `proto_replay`, DA `corrected_b2`, target CE 0, input dim 6, and `last_round`.

- [ ] **Step 5: Implement fail-closed result summarization**

The summarizer must require E0 plus postflight-valid E1/E2/E3 summaries, copy existing P2→P3 stable metrics as a reference, and output per direction stable/early/full correct counts, Accuracy, Macro-F1, class recall, exposure metrics, and confusion matrices. It must refuse overwrite and label P1+P2 as `combined_source_addition_effect`, not `diversity_effect`.

- [ ] **Step 6: Document commands and evidence boundaries**

Document exact local build/validate, three-node sync, preflight, queue, recovery, E0 evaluation, summary, and stop commands. State that E0 and P3 comparisons are post-hoc single-seed descriptions and that P2→P1 has a different target board.

- [ ] **Step 7: Run contract and compile tests**

Run: `python -m pytest tests/test_lab_three_gas_crossboard_contract.py tests/test_lab_three_gas_crossboard_controller.py -q`

Run: `python -m compileall -q scripts/lab_three_gas_3class`

Expected: both commands pass.

- [ ] **Step 8: Commit protocol and orchestration artifacts**

```powershell
git add scripts/lab_three_gas_3class/freeze_three_node_protocol.py scripts/lab_three_gas_3class/run_a4_crossboard_queue.ps1 scripts/lab_three_gas_3class/summarize_a4_crossboard_results.py scripts/lab_three_gas_3class/README.md docs/experiments/lab_3gas_a4_crossboard_execution_20260731.md docs/experiments/lab_3gas_a4_crossboard_matrix_20260731.csv tests/test_lab_three_gas_crossboard_contract.py
git commit -m "feat(lab): orchestrate A4 cross-board experiments"
```

---

### Task 8: Build Real Datasets and Gate E0

**Files:**
- Create generated roots under `dataset/client_data_lab_3gas_a4_crossboard_*_v1` (gitignored binary data).
- Create a new dated E0 result directory under `results/`.

**Interfaces:**
- Consumes: raw `dataset/Dataset_self`, approved builders/validators, existing A4 dataset, and frozen round-25 adapted checkpoint on Server A.
- Produces: four validated dataset views and E0 stable/early/full summary.

- [ ] **Step 1: Build P2→P3 E0 evaluation data**

Run:

```powershell
python scripts/lab_three_gas_3class/build_crossboard_a4_dataset.py --direction P2_to_P3 --output-root dataset/client_data_lab_3gas_a4_crossboard_p2p3_eval_v1
```

Expected: C2 stable train 360, C2 calibration 90, C3 calibration 90, C3 stable 360, early 60, and full 420.

- [ ] **Step 2: Validate exact E0 stable equivalence**

Run:

```powershell
python scripts/lab_three_gas_3class/validate_crossboard_a4_dataset.py --dataset-root dataset/client_data_lab_3gas_a4_crossboard_p2p3_eval_v1 --direction P2_to_P3 --reference-a4-root dataset/client_data_lab_3gas_allconc_timepurged_p2src_stable150_v1
```

Expected: `ok=true` and exact SHA equality for frozen norm, source splits, target calibration, and target stable test.

- [ ] **Step 3: Build and validate E1/E2/E3 data**

Run the builder and validator for:

```text
P2_to_P1  -> dataset/client_data_lab_3gas_a4_crossboard_p2p1_v1
P1_to_P3  -> dataset/client_data_lab_3gas_a4_crossboard_p1p3_v1
P12_to_P3 -> dataset/client_data_lab_3gas_a4_crossboard_p12p3_v1
```

Expected: every validation report has `ok=true`; all target scopes are 90 calibration, 360 stable, 60 early, and 420 full; source counts match the direction.

- [ ] **Step 4: Run the full local regression suite for changed code**

Run:

```powershell
python -m pytest tests/test_lab_three_gas_crossboard_contract.py tests/test_lab_three_gas_crossboard_dataset.py tests/test_lab_three_gas_crossboard_scopes.py tests/test_lab_three_gas_crossboard_controller.py tests/test_lab_three_gas_validate_three_node_run.py tests/test_lab_three_gas_allconc_timepurged.py tests/test_lab_three_gas_recovery_variants.py tests/test_lab_three_gas_flower_eval_split.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Freeze source and dataset manifests**

Create a new dated protocol root, archive the exact changed source, and record each dataset tree hash, topology, config, and protocol SHA. Verify the source archive extracts cleanly and compiles.

- [ ] **Step 6: Sync E0 data and evaluate the frozen A4 checkpoint**

Sync the content-addressed source and E0 dataset to Server A without overwriting the existing A4 runtime. Run `evaluate_crossboard_scopes.py` against the existing `server_round_025_adapted.pth`.

Expected gate: stable scope is exactly `359/360`, target client is 3, checkpoint round is 25, and checkpoint/norm hashes match the frozen A4 record.

- [ ] **Step 7: Stop if the E0 gate fails**

If stable is not exactly `359/360`, mark E0 invalid, retain diagnostics, do not launch E1/E2/E3, and investigate data identity before any training.

- [ ] **Step 8: Record E0 metrics without selecting a model**

Copy the E0 machine-readable summary locally and register early60/full420 as post-hoc diagnostic metrics from the frozen checkpoint. Do not use the metrics to change E1/E2/E3 configuration.

---

### Task 9: Three-Node Preflight, Sequential Training, and Final Audit

**Files:**
- Create new dated controller logs and result directories under `results/`.
- Create new final analysis and audit reports under `docs/experiments/` after all valid runs finish.

**Interfaces:**
- Consumes: frozen protocol, validated datasets, Task-6 controller, Task-7 queue, and E0 gate.
- Produces: E1/E2/E3 round-25 checkpoints, formal stable summaries, cross-scope summaries, valid postflight audits, combined result JSON/CSV, and evidence-boundary report.

- [ ] **Step 1: Synchronize exact data and source to all three machines**

Sync new general work code to `/root/GAPS` or `/home/gaps/GAPS`, create/verify content-addressed runtime directories from the frozen archive SHA, and copy direction datasets to:

```text
Server A: /root/GAPS/dataset/<dataset>
Cloud B:  /root/GAPS/lab_3gas_data/<dataset>
Pi:       /home/gaps/GAPS/lab_3gas_data/<dataset>
```

Verify manifest SHA, file count, total bytes, and runtime source SHA on every participating node.

- [ ] **Step 2: Run E1/E2/E3 preflight-only checks**

Invoke `run_lab_three_node_fold.ps1 -PreflightOnly` for each direction. Expected roles:

```text
E1 P2→P1: Server target C1; Cloud B source C2; Pi idle
E2 P1→P3: Server target C3; Pi source C1; Cloud B idle
E3 P1+P2→P3: Server target C3; Pi C1 and Cloud B C2 both source
```

Every preflight must report `[32,100,6]`, 3 classes, 22,564 parameters, source stable train 360 per client, source calibration 90 per client, target calibration 90, and target stable test 360.

- [ ] **Step 3: Launch the sequential queue**

Run `run_a4_crossboard_queue.ps1` in a hidden PowerShell process with separate stdout, stderr, and PID files. Record start time and exact run IDs. Do not start another Flower job while the queue is active.

- [ ] **Step 4: Monitor each run at bounded intervals**

Check controller stdout/stderr, round checkpoint count, exact run-tag processes, client failures, tunnel health, and Pi thermal/throttling state. A run passes execution only with 25 base and 25 adapted checkpoints, no fit/evaluate failures, and clean server exit.

- [ ] **Step 5: Evaluate all three scopes after each valid run**

For each round-25 adapted checkpoint, run `evaluate_crossboard_scopes.py` on Server A. Copy scope summaries locally. Never evaluate per round and never use target scope metrics for checkpoint selection.

- [ ] **Step 6: Run postflight and recover evidence**

Require `postflight_attempt_audit.json` status `valid`, explicit source/target IDs, selected round 25, correct sample counts, and no per-round target-test files. Recover formal summary, scope summary, postflight audit, run config, history, and protocol identity.

- [ ] **Step 7: Generate the combined result summary**

Run:

```powershell
python scripts/lab_three_gas_3class/summarize_a4_crossboard_results.py
```

Expected: one JSON and one CSV containing existing P2→P3 stable reference, E0 early/full, and E1/E2/E3 stable/early/full metrics. The script must fail if any required audit is not valid.

- [ ] **Step 8: Write descriptive analysis and experiment audit**

Report exact correct counts and confusion matrices. Compare P1→P3 and P1+P2→P3 only against E0/existing P2→P3 on identical P3 samples. Report P2→P1 separately. Preserve single-seed, overlapping-window, nominal-boundary, all-concentration-calibration, and post-hoc limitations.

- [ ] **Step 9: Run final verification**

Run changed pytest targets, `compileall`, dataset validators, protocol hash checks, postflight status checks, and a read-only remote process audit proving no residual Flower jobs or tunnels remain.

- [ ] **Step 10: Commit only lightweight code and evidence**

Stage explicit source, tests, plans, manifests, summaries, and reports. Exclude raw data, `.npy`, checkpoints, runtime caches, PID files, tunnel logs, and unrelated dirty-worktree files.

```powershell
git commit -m "feat(lab): complete A4 cross-board evaluation"
```

Do not push until the commit scope and final audit have been reviewed.
