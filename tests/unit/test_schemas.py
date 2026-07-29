import pytest
from pydantic import ValidationError

from verifiable_ai_workflow.schemas import PreparedDocument, StructuredAnswer


def test_general_answer_requires_evidence() -> None:
    with pytest.raises(ValidationError, match="근거"):
        StructuredAnswer(answer="120 units", confidence=0.9)


def test_abstention_uses_fixed_answer_and_reason() -> None:
    with pytest.raises(ValidationError, match="답변 보류"):
        StructuredAnswer(
            answer="모르겠습니다",
            confidence=0.2,
            abstained=True,
            abstention_reason="문서에 없음",
        )


def test_manifest_requires_every_page_in_order() -> None:
    with pytest.raises(ValidationError, match="전체 페이지"):
        PreparedDocument(
            document_id="sample",
            source_file="sample.pdf",
            source_sha256="a" * 64,
            total_pages=2,
            render_dpi=150,
            pages=[
                {
                    "page_number": 1,
                    "image_path": "pages/page-0001.png",
                    "model_image_path": "model-pages/page-0001.jpg",
                    "text_path": "text/page-0001.txt",
                }
            ],
        )
