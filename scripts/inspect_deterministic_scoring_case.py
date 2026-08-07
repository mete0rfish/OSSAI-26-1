"""Week 1 대표 사례의 입력·응답·채점을 학습자용 JSON으로 보여 준다."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from verifiable_ai_workflow.data.dataset import build_cases
from verifiable_ai_workflow.evaluation.scoring import (
    DIAGNOSTIC_METRICS,
    PASS_REQUIREMENTS,
    SCORING_PROFILE,
    score_output,
)
from verifiable_ai_workflow.providers.recorded import RecordedProvider

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPARED_ROOT = PROJECT_ROOT / "local-data/aihub/prepared"
CASE_ID = "aihub-report-r04"


def main() -> int:
    case = next(
        item
        for item in build_cases(PROJECT_ROOT / "data/cases/week-01-aihub.yaml")
        if item.sample_id == CASE_ID
    )
    manifest_path = PREPARED_ROOT / case.document_id / "manifest.json"
    if not manifest_path.is_file():
        print(
            "준비된 문서가 없습니다. 먼저 "
            "`uv run --locked python scripts/prepare_documents.py`를 실행하세요.",
            file=sys.stderr,
        )
        return 2

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_page = next(
        page for page in manifest["pages"] if page["page_number"] == case.expected.pages[0]
    )
    raw_response = RecordedProvider(
        PROJECT_ROOT / "data/recorded/week-01-nvidia-responses.jsonl"
    ).generate(case.sample_id, [])
    parsed_answer, scores, reasons = score_output(raw_response, case)

    payload = {
        "sample_id": case.sample_id,
        "input": {
            "question": case.question,
            "page_image_count": len(manifest["pages"]),
            "expected_page_image": (
                f"local-data/aihub/prepared/{case.document_id}/"
                f"{expected_page['model_image_path']}"
            ),
            "note": "실제 모델에는 질문과 전체 페이지 이미지만 보냅니다.",
        },
        "model_output": {
            "raw_response": raw_response,
            "parsed_answer": parsed_answer.model_dump() if parsed_answer else None,
        },
        "expected": case.expected.model_dump(),
        "scoring": {
            "profile": SCORING_PROFILE,
            "required_scores": {name: scores[name] for name in PASS_REQUIREMENTS},
            "diagnostic_scores": {name: scores[name] for name in DIAGNOSTIC_METRICS},
            "task_success": scores["task_success"],
            "failed_requirements": [
                {"metric": name, "reason": reasons[name]}
                for name in PASS_REQUIREMENTS
                if scores[name] != 1.0
            ],
        },
        "evidence_kind": "test_only",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
