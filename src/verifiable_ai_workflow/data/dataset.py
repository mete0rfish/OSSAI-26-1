"""데이터셋별 YAML을 공통 EvaluationCase JSONL로 바꾼다."""

from __future__ import annotations

from pathlib import Path

import yaml

from ..schemas import EvaluationCase


def build_cases(authoring_path: str | Path) -> list[EvaluationCase]:
    source = yaml.safe_load(Path(authoring_path).read_text(encoding="utf-8"))
    if source.get("artifact_schema_version") != 2:
        raise ValueError("authoring artifact_schema_version은 2여야 합니다")

    documents = {item["document_id"]: item for item in source["documents"]}
    cases: list[EvaluationCase] = []
    for item in source["cases"]:
        document_id = item["document_id"]
        if document_id not in documents:
            raise ValueError(f"정의되지 않은 document_id입니다: {document_id}")
        document = documents[document_id]
        cases.append(
            EvaluationCase(
                sample_id=item["sample_id"],
                family_id=document["family_id"],
                document_id=document_id,
                split=document["split"],
                source=document["source"],
                risk_level=item["risk_level"],
                question=item["question"],
                expected=item["expected"],
                tags=item.get("tags", []),
            )
        )

    sample_ids = [case.sample_id for case in cases]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample_id는 중복될 수 없습니다")
    return cases


def write_cases(cases: list[EvaluationCase], output_path: str | Path) -> None:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(case.model_dump_json() + "\n" for case in cases),
        encoding="utf-8",
    )


def load_cases(path: str | Path) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                cases.append(EvaluationCase.model_validate_json(line))
    return cases
