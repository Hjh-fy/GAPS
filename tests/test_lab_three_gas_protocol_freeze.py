from __future__ import annotations

from pathlib import Path
import tarfile

from scripts.lab_three_gas_3class.freeze_three_node_protocol import (
    build_source_archive,
)


def test_remote_source_archive_contains_posthoc_scope_evaluator(
    tmp_path: Path,
) -> None:
    archive, manifest = build_source_archive(tmp_path)

    with tarfile.open(archive, "r") as bundle:
        members = set(bundle.getnames())

    expected = (
        "scripts/lab_three_gas_3class/evaluate_crossboard_scopes.py"
    )
    assert expected in members
    assert expected in {
        row["relative_path"] for row in manifest["members"]
    }
