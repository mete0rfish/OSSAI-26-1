"""Gemma 기준·개선 prompt의 실제 대표 응답 한 쌍을 다시 채점한다."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from verifiable_ai_workflow.comparison import sha256_file
from verifiable_ai_workflow.data.dataset import build_cases
from verifiable_ai_workflow.evaluation.scoring import SCORING_PROFILE
from verifiable_ai_workflow.provider_evaluation import (
    evaluate_recorded_route,
    run_offline_comparison,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPARED_ROOT = PROJECT_ROOT / "local-data/aihub/prepared"
CASE_ID = "aihub-report-r01"


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
            "`uv run python scripts/prepare_documents.py`를 실행하세요.",
            file=sys.stderr,
        )
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    page_images = []
    for page in manifest["pages"]:
        image_path = manifest_path.parent / page["model_image_path"]
        page_images.append(
            {
                "page_number": page["page_number"],
                "path": (
                    f"local-data/aihub/prepared/{case.document_id}/{page['model_image_path']}"
                ),
                "bytes": image_path.stat().st_size,
                "sha256": sha256_file(image_path),
            }
        )

    expected_page_number = case.expected.pages[0]
    expected_page = next(
        page for page in manifest["pages"] if page["page_number"] == expected_page_number
    )
    comparison = run_offline_comparison(PROJECT_ROOT)

    baseline_results = evaluate_recorded_route(
        PROJECT_ROOT,
        overlay_path=None,
        requested_model=comparison.baseline_route.requested_model,
        actual_model=comparison.baseline_route.expected_actual_model,
    )
    candidate_results = evaluate_recorded_route(
        PROJECT_ROOT,
        overlay_path=PROJECT_ROOT / "data/scenarios/week-02-route-b-overrides.jsonl",
        requested_model=comparison.baseline_route.requested_model,
        actual_model=comparison.baseline_route.expected_actual_model,
    )
    baseline = next(item for item in baseline_results if item.sample_id == CASE_ID)
    candidate = next(item for item in candidate_results if item.sample_id == CASE_ID)
    case_diff = next(item for item in comparison.case_diffs if item.sample_id == CASE_ID)

    payload = {
        "week": 2,
        "experiment_type": "prompt_only_case_replay",
        "case_id": case.sample_id,
        "evidence_boundary": {
            "evidence_kind": "test_only",
            "live_quality_claim": False,
            "notice": (
                "두 응답은 같은 Gemma 4의 실제 기준·개선 prompt 실행에서 가져왔습니다. "
                "저장 응답 재채점이므로 현재 provider 품질을 새로 측정한 결과는 아닙니다."
            ),
        },
        "input": {
            "dataset": "data/cases/week-01-aihub.yaml",
            "document_id": case.document_id,
            "family_id": case.family_id,
            "source": case.source.model_dump(),
            "split": case.split,
            "risk_level": case.risk_level,
            "question": case.question,
            "live_model_input_contract": "질문과 전체 page JPEG만 전송",
            "fixture_replay_input": "sample_id로 저장 응답을 조회하며 API를 호출하지 않음",
            "prepared_document": {
                "manifest": (f"local-data/aihub/prepared/{case.document_id}/manifest.json"),
                "source_sha256": manifest["source_sha256"],
                "total_pages": manifest["total_pages"],
                "page_images_for_live_request": page_images,
                "expected_page_reference": {
                    "page_number": expected_page_number,
                    "image_path": (
                        f"local-data/aihub/prepared/{case.document_id}/"
                        f"{expected_page['model_image_path']}"
                    ),
                },
            },
            "same_for_both_routes": [
                "sample_id",
                "document",
                "question",
                "model",
                "output schema",
                "scorer",
            ],
            "treatment": "prompt만 변경",
        },
        "model_output": {
            "baseline": {
                "fixture": "data/recorded/week-02-gemma-baseline-responses.jsonl",
                "model": comparison.baseline_route.requested_model,
                "prompt": "prompts/pdf-question-answer.md",
                "prompt_sha256": sha256_file(PROJECT_ROOT / "prompts/pdf-question-answer.md"),
                "raw_response": baseline.raw_output,
                "parsed_answer": baseline.output.model_dump() if baseline.output else None,
                "call_metadata": baseline.model_call,
            },
            "candidate": {
                "fixture": "data/scenarios/week-02-route-b-overrides.jsonl",
                "fixture_note": "2026-08-03 실제 개선 prompt 실행에서 고정한 대표 응답",
                "model": comparison.baseline_route.requested_model,
                "prompt": "prompts/pdf-question-answer-gemma4.md",
                "prompt_sha256": sha256_file(
                    PROJECT_ROOT / "prompts/pdf-question-answer-gemma4.md"
                ),
                "raw_response": candidate.raw_output,
                "parsed_answer": candidate.output.model_dump() if candidate.output else None,
                "call_metadata": candidate.model_call,
            },
        },
        "expected": case.expected.model_dump(),
        "evaluation_design": {
            "scorer": "verifiable_ai_workflow.evaluation.scoring.score_output",
            "method": "각 route를 같은 결정적 scorer로 평가한 뒤 문제별 task_success를 비교",
            "scoring_profile": SCORING_PROFILE,
            "task_success_formula": (
                "schema_validity=1 AND abstention_correct=1 AND answer_correct=1 "
                "AND evidence_coverage=1"
            ),
            "quote_boundary": (
                "quote_answer_support는 모델이 쓴 답과 인용문의 자체 일관성 진단이며 "
                "이미지 근거 일치를 증명하거나 task_success를 결정하지 않음"
            ),
            "comparison_boundary": (
                "이 명령은 대표 1건을 설명한다. 전체 A/B의 통제 조건과 provider 오류는 "
                "compare_gemma_prompts.py --rescore-current로 확인한다."
            ),
            "classification_rule": {
                "new_success": "baseline task_success=0, candidate task_success=1",
                "new_failure": "baseline task_success=1, candidate task_success=0",
                "unchanged": "두 task_success가 같음",
                "not_comparable": "provider 오류, 비교 조건 또는 actual model 문제",
            },
            "quality_claim_rule": (
                "저장 응답만 사용한 비교의 자동 상태는 점수와 무관하게 inconclusive"
            ),
        },
        "evaluation_result": {
            "baseline": {
                "status": baseline.status,
                "scores": baseline.scores,
                "reasons": baseline.reasons,
            },
            "candidate": {
                "status": candidate.status,
                "scores": candidate.scores,
                "reasons": candidate.reasons,
            },
            "case_comparison": case_diff.model_dump(),
            "all_cases_context": {
                "sample_count": comparison.baseline.record_count,
                "classification_counts": comparison.classification_counts,
                "automated_status": comparison.automated_status,
                "automated_reason": comparison.automated_reason,
            },
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
