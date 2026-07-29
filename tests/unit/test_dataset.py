from pathlib import Path

from verifiable_ai_workflow.data.dataset import build_cases, load_cases, write_cases


def test_authoring_yaml_becomes_common_cases(
    project_root: Path,
    tmp_path: Path,
) -> None:
    cases = build_cases(project_root / "data/cases/week-01-aihub.yaml")
    output_path = tmp_path / "cases.jsonl"
    write_cases(cases, output_path)
    loaded = load_cases(output_path)

    assert len(loaded) == 40
    assert sum(case.expected.abstained for case in loaded) == 4
    assert {case.document_id for case in loaded} == {
        "MI2_240819_TY1_0012",
        "MI2_240725_TY2_0002",
    }
    assert {case.split for case in loaded} == {"development", "validation"}
    assert {case.risk_level for case in loaded} == {"low"}
