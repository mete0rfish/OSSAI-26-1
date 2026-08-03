"""Week 1 대표 1건의 입력부터 결정적 평가까지 한 화면에 보여 준다."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from verifiable_ai_workflow.data.dataset import build_cases
from verifiable_ai_workflow.evaluation.scoring import score_output
from verifiable_ai_workflow.providers.recorded import RecordedProvider

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPARED_ROOT = PROJECT_ROOT / "local-data/aihub/prepared"
CASE_ID = "aihub-report-r01"


def main() -> int:
    cases = build_cases(PROJECT_ROOT / "data/cases/week-01-aihub.yaml")
    case = next(item for item in cases if item.sample_id == CASE_ID)

    manifest_path = PREPARED_ROOT / case.document_id / "manifest.json"
    if not manifest_path.is_file():
        print(
            "준비된 문서가 없습니다. 먼저 "
            "`uv run python scripts/prepare_documents.py`를 실행하세요.",
            file=sys.stderr,
        )
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sent_page_images = []
    page_texts = {}
    for page in manifest["pages"]:
        image_path = manifest_path.parent / page["model_image_path"]
        page_texts[page["page_number"]] = (manifest_path.parent / page["text_path"]).read_text(
            encoding="utf-8"
        )
        sent_page_images.append(
            {
                "page_number": page["page_number"],
                "path": (
                    f"local-data/aihub/prepared/{case.document_id}/{page['model_image_path']}"
                ),
                "bytes": image_path.stat().st_size,
            }
        )

    expected_page_number = case.expected.pages[0]
    expected_page = next(
        page for page in manifest["pages"] if page["page_number"] == expected_page_number
    )
    text_path = manifest_path.parent / expected_page["text_path"]
    page_text = text_path.read_text(encoding="utf-8")
    answer_position = page_text.find(case.expected.answer)
    excerpt_start = max(0, answer_position - 90) if answer_position >= 0 else 0
    page_excerpt = " ".join(page_text[excerpt_start : excerpt_start + 240].split())[:240]

    fixture_path = PROJECT_ROOT / "data/recorded/week-01-nvidia-responses.jsonl"
    provider = RecordedProvider(fixture_path)
    raw_response = provider.generate(case.sample_id, [])
    call_metadata = getattr(provider, "last_call", None)
    parsed_answer, scores, reasons = score_output(raw_response, case, page_texts=page_texts)

    payload = {
        "week": 1,
        "case_id": case.sample_id,
        "evidence_boundary": {
            "evidence_kind": "test_only",
            "live_quality_claim": False,
            "notice": (
                "저장된 실제 모델 응답을 다시 읽는 회귀 fixture입니다. "
                "현재 준비된 이미지와 당시 응답을 같은 요청으로 묶는 입력 hash는 없으며, "
                "현재 provider의 live 품질 증거가 아닙니다."
            ),
        },
        "input": {
            "dataset": "data/cases/week-01-aihub.yaml",
            "document_id": case.document_id,
            "family_id": case.family_id,
            "source": case.source.model_dump(),
            "split": case.split,
            "risk_level": case.risk_level,
            "tags": case.tags,
            "question": case.question,
            "model_input_summary": "해당 문서의 전체 page image와 질문",
            "prepared_document": {
                "manifest": (f"local-data/aihub/prepared/{case.document_id}/manifest.json"),
                "source_sha256": manifest["source_sha256"],
                "total_pages": manifest["total_pages"],
                "sent_page_images": sent_page_images,
                "expected_page_preview": {
                    "page_number": expected_page_number,
                    "text_path": (
                        f"local-data/aihub/prepared/{case.document_id}/{expected_page['text_path']}"
                    ),
                    "text_excerpt": page_excerpt,
                    "truncated": excerpt_start > 0 or excerpt_start + 240 < len(page_text),
                },
            },
            "recorded_input_tokens": (call_metadata or {}).get("input_tokens"),
        },
        "model_output": {
            "fixture": fixture_path.relative_to(PROJECT_ROOT).as_posix(),
            "raw_response": raw_response,
            "parsed_answer": parsed_answer.model_dump() if parsed_answer else None,
            "call_metadata": call_metadata,
        },
        "expected": case.expected.model_dump(),
        "evaluation_design": {
            "scorer": "verifiable_ai_workflow.evaluation.scoring.score_output",
            "method": "정해진 규칙으로 계산하는 결정적 평가",
            "task_success_formula": (
                "schema_validity=1 AND abstention_correct=1 AND answer_correct=1 "
                "AND evidence_coverage=1 AND 검증 가능한 quote 조건 통과"
            ),
            "answer_rule": "숫자형 정답은 실제 답과 기대 답의 숫자 목록이 같아야 통과",
            "evidence_rule": "모델이 제시한 page 중 기대 page가 하나 이상 있어야 통과",
            "quote_rule": (
                "준비된 PDF page text로 검증 가능한 quote는 quote_grounding 0.8 이상이어야 통과"
            ),
        },
        "evaluation_result": {
            "status": "passed" if scores["task_success"] == 1.0 else "failed",
            "scores": scores,
            "reasons": reasons,
            "decision_reason": reasons["task_success"],
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
