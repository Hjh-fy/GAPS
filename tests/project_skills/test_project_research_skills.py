import csv
import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SKILLS_ROOT = REPO_ROOT / ".agents" / "skills"
EXPECTED_SKILLS = {
    "experiment-planner",
    "experiment-registry",
    "result-analysis",
    "experiment-audit",
    "claim-evidence",
    "number-consistency-audit",
    "gaps-research-orchestrator",
}
REQUIRED_EXPERIMENT_FIELDS = {
    "experiment_id",
    "source_clients",
    "target_clients",
    "split_protocol",
    "model",
    "checkpoint",
    "DA",
    "calibration",
    "QC",
    "seed",
    "result_path",
    "metrics",
    "status",
    "notes",
}
EXPECTED_OUTPUTS = {
    "experiment-planner": {
        "EXPERIMENT_PLAN.template.md",
        "EXPERIMENT_MATRIX.template.csv",
        "ABLATION_PLAN.template.md",
    },
    "experiment-registry": {"experiment_registry.template.csv"},
    "result-analysis": {"RESULT_ANALYSIS.template.md"},
    "experiment-audit": {"EXPERIMENT_AUDIT.template.md"},
    "claim-evidence": {"CLAIMS_EVIDENCE.template.md"},
    "number-consistency-audit": {"NUMBER_AUDIT.template.md"},
    "gaps-research-orchestrator": {
        "PROJECT_STATUS.template.md",
        "NEXT_ACTIONS.template.md",
    },
}


def _frontmatter(skill_file: Path) -> dict[str, str]:
    text = skill_file.read_text(encoding="utf-8")
    match = re.match(r"\A---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    assert match, f"Missing YAML frontmatter: {skill_file}"
    result = {}
    for raw_line in match.group(1).splitlines():
        key, separator, value = raw_line.partition(":")
        assert separator, f"Invalid frontmatter line in {skill_file}: {raw_line}"
        result[key.strip()] = value.strip().strip('"')
    return result


def _skill_text(name: str) -> str:
    return (SKILLS_ROOT / name / "SKILL.md").read_text(encoding="utf-8")


def test_shared_contracts_exist_and_define_required_fields():
    shared_files = {
        "contracts/experiment-record.md",
        "contracts/metric-record.md",
        "contracts/evidence-record.md",
        "contracts/handoff-protocol.md",
        "references/gaps-taxonomy.md",
        "references/read-only-policy.md",
        "references/skill-boundaries.md",
    }
    for relative in shared_files:
        assert (SKILLS_ROOT / "_shared" / relative).is_file(), relative

    text = (
        SKILLS_ROOT / "_shared" / "contracts" / "experiment-record.md"
    ).read_text(encoding="utf-8")
    for field in REQUIRED_EXPERIMENT_FIELDS:
        assert re.search(rf"\b{re.escape(field)}\b", text), field


def test_exactly_seven_triggerable_skill_directories_exist():
    actual = {
        path.name
        for path in SKILLS_ROOT.iterdir()
        if path.is_dir() and not path.name.startswith("_")
    }
    assert actual == EXPECTED_SKILLS


def test_skill_frontmatter_and_ui_metadata_are_valid():
    for name in EXPECTED_SKILLS:
        skill_dir = SKILLS_ROOT / name
        metadata = _frontmatter(skill_dir / "SKILL.md")
        assert set(metadata) == {"name", "description"}
        assert metadata["name"] == name
        assert metadata["description"].startswith("Use when ")
        assert len(metadata["description"]) <= 500
        assert (skill_dir / "agents" / "openai.yaml").is_file()


def test_each_skill_has_templates_trigger_cases_and_script_interface():
    for name, expected_assets in EXPECTED_OUTPUTS.items():
        skill_dir = SKILLS_ROOT / name
        actual_assets = {
            path.name for path in (skill_dir / "assets").iterdir() if path.is_file()
        }
        assert expected_assets <= actual_assets
        assert (skill_dir / "scripts" / "INTERFACE.md").is_file()

        cases = json.loads(
            (skill_dir / "references" / "trigger-cases.json").read_text(
                encoding="utf-8"
            )
        )
        positives = [case for case in cases if case["should_trigger"]]
        negatives = [case for case in cases if not case["should_trigger"]]
        assert len(positives) >= 2, name
        assert len(negatives) >= 1, name
        assert all(case["expected_skill"] for case in cases)
        assert all(case["reason"] for case in cases)


def test_experiment_csv_templates_use_canonical_fields():
    templates = [
        SKILLS_ROOT
        / "experiment-planner"
        / "assets"
        / "EXPERIMENT_MATRIX.template.csv",
        SKILLS_ROOT
        / "experiment-registry"
        / "assets"
        / "experiment_registry.template.csv",
    ]
    for template in templates:
        with template.open(encoding="utf-8-sig", newline="") as handle:
            header = set(next(csv.reader(handle)))
        assert REQUIRED_EXPERIMENT_FIELDS <= header, template


def test_read_only_and_collision_rules_are_present_in_every_skill():
    for name in EXPECTED_SKILLS:
        text = _skill_text(name).lower()
        assert "read-only" in text or "只读" in text, name
        assert "overwrite" in text or "覆盖" in text, name
        assert "unknown" in text, name
        assert "conflict" in text, name


def test_analysis_and_audit_responsibilities_do_not_overlap():
    analysis = _skill_text("result-analysis").lower()
    audit = _skill_text("experiment-audit").lower()
    assert "mean" in analysis and "effect size" in analysis
    assert "fair" in audit and "leakage" in audit
    assert "do not decide whether comparisons are fair" in analysis
    assert "do not recompute replacement metrics" in audit


def test_evidence_gate_and_number_audit_safety_are_explicit():
    claim = _skill_text("claim-evidence").lower()
    number = _skill_text("number-consistency-audit").lower()
    assert "unaudited" in claim and "approved" in claim
    assert "abstract" in number and "table" in number and "conclusion" in number
    assert "rounding" in number
    assert "do not silently edit" in number


def test_orchestrator_routes_without_doing_child_work():
    text = _skill_text("gaps-research-orchestrator").lower()
    for required in (
        "current stage",
        "largest evidence gap",
        "next skill",
        "experiment-planner",
        "experiment-registry",
        "experiment-audit",
        "claim-evidence",
        "number-consistency-audit",
    ):
        assert required in text
    assert "do not perform child skill work" in text


def test_cross_references_resolve():
    link_pattern = re.compile(r"\[[^]]+\]\(([^)]+)\)")
    for name in EXPECTED_SKILLS:
        skill_file = SKILLS_ROOT / name / "SKILL.md"
        for target in link_pattern.findall(skill_file.read_text(encoding="utf-8")):
            if "://" in target or target.startswith("#"):
                continue
            assert (skill_file.parent / target).resolve().exists(), (name, target)

