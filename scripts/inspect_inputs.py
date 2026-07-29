"""PDF, 질문, 정답과 prompt의 이상치를 한 번에 확인한다."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from PIL import Image

from verifiable_ai_workflow.config import load_settings, project_path
from verifiable_ai_workflow.data.dataset import build_cases
from verifiable_ai_workflow.preprocessing import load_document
from verifiable_ai_workflow.schemas import StructuredAnswer

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _answer_fact_is_visible(text: str, answer: str) -> bool:
    numbers = re.findall(r"-?\d+(?:\.\d+)?", unicodedata.normalize("NFKC", answer))
    if numbers:
        page_numbers = re.findall(r"-?\d+(?:\.\d+)?", unicodedata.normalize("NFKC", text))
        return all(number in page_numbers for number in numbers)
    def normalize(value: str) -> str:
        return re.sub(
            r"[^0-9a-z가-힣]+",
            "",
            unicodedata.normalize("NFKC", value).casefold(),
        )

    return bool(normalize(answer) and normalize(answer) in normalize(text))


def main() -> int:
    parser = argparse.ArgumentParser(description="Week 1 PDF·질문·prompt EDA")
    parser.add_argument("--config", default="configs/week-01.yaml")
    args = parser.parse_args()

    settings = load_settings(project_path(PROJECT_ROOT, args.config))
    cases = build_cases(project_path(PROJECT_ROOT, settings.paths.case_authoring))
    prepared_root = project_path(PROJECT_ROOT, settings.paths.prepared_documents)
    prompt = project_path(PROJECT_ROOT, settings.paths.prompt).read_text(encoding="utf-8")
    anomalies: list[str] = []
    documents: dict[str, dict[str, object]] = {}
    document_texts: dict[str, dict[int, str]] = {}

    for document_id in sorted({case.document_id for case in cases}):
        document, manifest_path = load_document(prepared_root, document_id)
        page_rows = []
        page_texts: dict[int, str] = {}
        for page in document.pages:
            image_path = manifest_path.parent / page.image_path
            model_image_path = manifest_path.parent / page.model_image_path
            text_path = manifest_path.parent / page.text_path
            with Image.open(image_path) as image:
                width, height = image.size
            model_bytes = model_image_path.stat().st_size
            page_text = text_path.read_text(encoding="utf-8")
            page_texts[page.page_number] = page_text
            text_chars = len(page_text.strip())
            if model_bytes > settings.documents.model_image_max_bytes:
                anomalies.append(
                    f"{document_id} page {page.page_number}: model image {model_bytes} bytes"
                )
            if text_chars == 0:
                anomalies.append(f"{document_id} page {page.page_number}: PDF text 없음")
            page_rows.append(
                {
                    "page_number": page.page_number,
                    "image_width": width,
                    "image_height": height,
                    "model_image_bytes": model_bytes,
                    "text_characters": text_chars,
                }
            )
        documents[document_id] = {
            "source_sha256": document.source_sha256,
            "total_pages": document.total_pages,
            "render_dpi": document.render_dpi,
            "pages": page_rows,
        }
        document_texts[document_id] = page_texts

    sample_ids = [case.sample_id for case in cases]
    if len(sample_ids) != len(set(sample_ids)):
        anomalies.append("sample_id 중복")
    normalized_questions = [" ".join(case.question.split()).casefold() for case in cases]
    if len(normalized_questions) != len(set(normalized_questions)):
        anomalies.append("동일 질문 중복")
    label_text_checks = []
    for case in cases:
        total_pages = int(documents[case.document_id]["total_pages"])
        invalid_pages = [page for page in case.expected.pages if page < 1 or page > total_pages]
        if invalid_pages:
            anomalies.append(f"{case.sample_id}: 잘못된 기대 페이지 {invalid_pages}")
        if case.expected.abstained:
            label_text_checks.append(
                {
                    "sample_id": case.sample_id,
                    "status": "abstention_manual_review",
                    "expected_pages": [],
                    "text_observable_pages": [],
                }
            )
            continue
        observable_pages = [
            page_number
            for page_number, page_text in document_texts[case.document_id].items()
            if _answer_fact_is_visible(page_text, case.expected.answer)
        ]
        status = (
            "matched_text_layer"
            if set(observable_pages) & set(case.expected.pages)
            else "not_observable_in_text_layer"
            if not observable_pages
            else "page_label_mismatch"
        )
        if status == "page_label_mismatch":
            anomalies.append(
                f"{case.sample_id}: answer text pages={observable_pages}, "
                f"expected pages={case.expected.pages}"
            )
        label_text_checks.append(
            {
                "sample_id": case.sample_id,
                "status": status,
                "expected_pages": case.expected.pages,
                "text_observable_pages": observable_pages,
            }
        )

    schema_fields = set(StructuredAnswer.model_fields)
    missing_prompt_fields = sorted(field for field in schema_fields if field not in prompt)
    if missing_prompt_fields:
        anomalies.append(f"prompt에 응답 필드 누락: {missing_prompt_fields}")
    if "답변 보류" not in prompt:
        anomalies.append("prompt에 답변 보류 규칙 누락")

    report = {
        "document_count": len(documents),
        "case_count": len(cases),
        "document_case_counts": dict(Counter(case.document_id for case in cases)),
        "split_counts": dict(Counter(case.split for case in cases)),
        "answer_type_counts": {
            "answer": sum(not case.expected.abstained for case in cases),
            "abstention": sum(case.expected.abstained for case in cases),
        },
        "expected_page_counts": dict(
            sorted(Counter(page for case in cases for page in case.expected.pages).items())
        ),
        "label_text_check_counts": dict(
            Counter(row["status"] for row in label_text_checks)
        ),
        "label_text_checks": label_text_checks,
        "prompt_characters": len(prompt),
        "schema_fields": sorted(schema_fields),
        "documents": documents,
        "anomalies": anomalies,
        "human_review": (
            f"{len(cases)}개 기대 답과 근거 페이지는 실제 NIM 실행 전에 "
            "두 사람이 검토한다."
        ),
    }
    output_path = project_path(PROJECT_ROOT, settings.paths.output) / "eda.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if anomalies else 0


if __name__ == "__main__":
    raise SystemExit(main())
