from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/close_iotj_manuscript_protocol.py"
SPEC = importlib.util.spec_from_file_location("close_iotj_manuscript_protocol", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _identity(path: Path) -> tuple[int, int, str]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, MODULE.sha256(path)


def test_default_mode_verifies_formal_files_without_writing() -> None:
    before = {
        path: _identity(path)
        for path in (MODULE.OUTPUT, MODULE.TABLE, MODULE.INDEX)
    }
    assert MODULE.main([]) == 0
    after = {
        path: _identity(path)
        for path in (MODULE.OUTPUT, MODULE.TABLE, MODULE.INDEX)
    }
    assert after == before


def test_generate_requires_all_three_explicit_destinations(tmp_path: Path) -> None:
    assert (
        MODULE.main(
            [
                "--generate",
                "--output-html",
                str(tmp_path / "new.html"),
            ]
        )
        == 2
    )
    assert list(tmp_path.iterdir()) == []


def test_generate_creates_new_outputs_and_verify_only_accepts_them(
    tmp_path: Path,
) -> None:
    output = tmp_path / "candidate.html"
    table = tmp_path / "candidate.csv"
    index = tmp_path / "candidate.json"
    args = [
        "--generate",
        "--output-html",
        str(output),
        "--table",
        str(table),
        "--index",
        str(index),
    ]
    assert MODULE.main(args) == 0
    assert output.is_file() and table.is_file() and index.is_file()
    assert (
        MODULE.main(
            [
                "--verify-only",
                "--output-html",
                str(output),
                "--table",
                str(table),
                "--index",
                str(index),
            ]
        )
        == 0
    )


def test_generate_refuses_to_overwrite_any_existing_target(tmp_path: Path) -> None:
    output = tmp_path / "existing.html"
    output.write_text("keep", encoding="utf-8")
    table = tmp_path / "new.csv"
    index = tmp_path / "new.json"
    assert (
        MODULE.main(
            [
                "--generate",
                "--output-html",
                str(output),
                "--table",
                str(table),
                "--index",
                str(index),
            ]
        )
        == 3
    )
    assert output.read_text(encoding="utf-8") == "keep"
    assert not table.exists()
    assert not index.exists()
