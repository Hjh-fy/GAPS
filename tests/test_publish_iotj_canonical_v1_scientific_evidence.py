import hashlib
import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/publish_iotj_canonical_v1_scientific_evidence.py"
SPEC = importlib.util.spec_from_file_location("publish_scientific_evidence", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_publish_copies_declared_evidence_and_hashes_published_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("metric,value\nmacro_f1,0.99\n", encoding="utf-8")
    report = tmp_path / "report.md"
    report.write_text("# Report\n", encoding="utf-8")
    docs = tmp_path / "docs"

    payload = MODULE.publish(
        docs,
        [(source, "comparison.csv"), (report, "report.md")],
    )

    assert (docs / "comparison.csv").read_bytes() == source.read_bytes()
    assert payload["files"]["comparison.csv"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert payload["files"]["comparison.csv"]["bytes"] == source.stat().st_size
    saved = json.loads((docs / "scientific_validation_sha256_index.json").read_text(encoding="utf-8"))
    assert saved == payload


def test_publish_fails_closed_for_missing_source(tmp_path: Path) -> None:
    try:
        MODULE.publish(tmp_path / "docs", [(tmp_path / "missing.csv", "missing.csv")])
    except FileNotFoundError as exc:
        assert "missing.csv" in str(exc)
    else:
        raise AssertionError("missing evidence must fail closed")


def test_completion_status_distinguishes_execution_from_submission_readiness() -> None:
    status = MODULE.completion_status(
        {"recommendation": "NOT_READY", "matrix_complete": True, "a0t_complete": True, "strict_collapse": True}
    )
    assert status["experiment_execution"] == "COMPLETE"
    assert status["submission_recommendation"] == "NOT_READY"
    assert status["strict_nonoverlap_claim"] == "BLOCKED"
    assert status["active_training_process"] is False


def test_portable_source_uses_repo_relative_paths_for_repo_evidence() -> None:
    assert MODULE.portable_source(MODULE.ROOT / "results/example.csv") == "results/example.csv"


def test_publish_can_index_an_evidence_file_already_in_docs(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    evidence = docs / "status.md"
    evidence.write_text("complete\n", encoding="utf-8")
    payload = MODULE.publish(docs, [(evidence, "status.md")])
    assert payload["files"]["status.md"]["sha256"] == hashlib.sha256(evidence.read_bytes()).hexdigest()
