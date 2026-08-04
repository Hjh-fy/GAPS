from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.finalize_iotj_a4_package import require_artifacts, sha256_tree


def test_sha256_tree_is_relative_deterministic_and_excludes_own_index(tmp_path: Path) -> None:
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "sha256_index.json").write_text("old", encoding="utf-8")

    rows = sha256_tree(tmp_path)

    assert [row["path"] for row in rows] == ["a.txt", "b.txt"]
    assert rows[0]["sha256"] == hashlib.sha256(b"a").hexdigest()


def test_require_artifacts_fails_closed_on_missing_file(tmp_path: Path) -> None:
    (tmp_path / "present.csv").write_text("x\n1\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="missing.csv"):
        require_artifacts(tmp_path, ["present.csv", "missing.csv"])
