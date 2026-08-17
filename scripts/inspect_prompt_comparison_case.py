"""Week 2 대표 사례에서 지시문 변경 전후의 답과 점수를 보여 준다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from verifiable_ai_workflow.data.dataset import build_cases
from verifiable_ai_workflow.evaluation.scoring import (
    DIAGNOSTIC_METRICS,
    PASS_REQUIREMENTS,
    score_output,
)
from verifiable_ai_workflow.providers.recorded import RecordedProvider

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPARED_ROOT = PROJECT_ROOT / "local-data/aihub/prepared"
CASE_ID = "aihub-report-r01"
BASELINE_FIXTURE = PROJECT_ROOT / "data/recorded/week-02-gemma-baseline-responses.jsonl"
CANDIDATE_OVERLAY = (
    PROJECT_ROOT / "data/scenarios/week-02-prompt-candidate-overrides.jsonl"
)


def _result(raw_response: str, case) -> dict:
    parsed_answer, scores, reasons = score_output(raw_response, case)
    return {
        "raw_response": raw_response,
        "parsed_answer": parsed_answer.model_dump() if parsed_answer else None,
        "required_scores": {name: scores[name] for name in PASS_REQUIREMENTS},
        "diagnostic_scores": {name: scores[name] for name in DIAGNOSTIC_METRICS},
        "diagnostic_reasons": {name: reasons[name] for name in DIAGNOSTIC_METRICS},
        "task_success": scores["task_success"],
        "failed_requirements": [
            {"metric": name, "reason": reasons[name]}
            for name in PASS_REQUIREMENTS
            if scores[name] != 1.0
        ],
    }


def main() -> int:
    case = next(
        item
        for item in build_cases(PROJECT_ROOT / "data/cases/week-01-aihub.yaml")
        if item.sample_id == CASE_ID
    )
    manifest_path = PREPARED_ROOT / case.document_id / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else None
    )

    baseline_raw = RecordedProvider(BASELINE_FIXTURE).generate(case.sample_id, [])
    candidate_raw = RecordedProvider(
        BASELINE_FIXTURE,
        overlay_path=CANDIDATE_OVERLAY,
    ).generate(case.sample_id, [])
    baseline = _result(baseline_raw, case)
    candidate = _result(candidate_raw, case)
    baseline_success = bool(baseline["task_success"])
    candidate_success = bool(candidate["task_success"])
    classification = (
        "new_success"
        if not baseline_success and candidate_success
        else "new_failure"
        if baseline_success and not candidate_success
        else "unchanged"
    )

    payload = {
        "sample_id": case.sample_id,
        "changed": "prompt_only",
        "same_conditions": [
            "페이지 이미지",
            "질문",
            "모델",
            "출력 형식(schema)",
            "채점기(scorer)",
        ],
        "input": {
            "question": case.question,
            "prepared_input_status": "prepared" if manifest else "not_prepared",
            "page_image_count": len(manifest["pages"]) if manifest else None,
        },
        "expected": case.expected.model_dump(),
        "baseline": {
            "prompt": "prompts/pdf-question-answer.md",
            **baseline,
        },
        "candidate": {
            "prompt": "prompts/pdf-question-answer-gemma4.md",
            **candidate,
        },
        "classification": classification,
        "diagnostic_note": (
            "json_object_only는 출력 습관을 확인하는 진단값이며 "
            "task_success 통과 조건에 포함되지 않습니다."
        ),
        "evidence_kind": "test_only",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    raise SystemExit(main())
