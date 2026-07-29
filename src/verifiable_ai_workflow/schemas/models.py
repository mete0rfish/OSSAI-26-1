"""Week 1에서 파일로 저장하거나 함수 사이에 전달하는 값."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Evidence(Contract):
    evidence_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)
    page_number: int = Field(ge=1)


class StructuredAnswer(Contract):
    answer: str = Field(min_length=1)
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    abstained: bool = False
    abstention_reason: str | None = None
    # PDF 질문 답변에서는 빈 목록을 반환한다.
    tool_requests: list[dict[str, Any]] = Field(default_factory=list, max_length=0)

    @model_validator(mode="after")
    def answer_matches_abstention(self) -> StructuredAnswer:
        if self.abstained:
            if self.answer != "답변 보류" or not self.abstention_reason:
                raise ValueError("답변 보류에는 정해진 answer와 이유가 필요합니다")
            if self.evidence:
                raise ValueError("답변 보류에는 근거를 넣지 않습니다")
        elif not self.evidence:
            raise ValueError("일반 답변에는 근거가 필요합니다")
        return self


class PreparedPage(Contract):
    page_number: int = Field(ge=1)
    image_path: str = Field(min_length=1)
    model_image_path: str = Field(min_length=1)
    text_path: str = Field(min_length=1)


class PreparedDocument(Contract):
    artifact_schema_version: Literal[2] = 2
    document_id: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    total_pages: int = Field(gt=0)
    render_dpi: int = Field(gt=0)
    pages: list[PreparedPage] = Field(min_length=1)

    @model_validator(mode="after")
    def contains_every_page(self) -> PreparedDocument:
        page_numbers = [page.page_number for page in self.pages]
        if page_numbers != list(range(1, self.total_pages + 1)):
            raise ValueError("pages에는 전체 페이지가 1번부터 순서대로 있어야 합니다")
        return self


class ExpectedAnswer(Contract):
    answer: str = Field(min_length=1)
    # 같은 정답이 반복되면 하나만 인용해도 되도록 가능한 페이지를 모두 적는다.
    pages: list[int]
    abstained: bool = False

    @model_validator(mode="after")
    def expected_pages_match_abstention(self) -> ExpectedAnswer:
        if self.abstained and (self.answer != "답변 보류" or self.pages):
            raise ValueError("답변 보류 기대값은 정해진 answer와 빈 pages가 필요합니다")
        if not self.abstained and not self.pages:
            raise ValueError("일반 기대값에는 가능한 근거 페이지가 필요합니다")
        return self


class SourceMetadata(Contract):
    title: str = Field(min_length=1)
    license: str = Field(min_length=1)
    revision: str = Field(min_length=1)


class EvaluationCase(Contract):
    artifact_schema_version: Literal[2] = 2
    sample_id: str = Field(min_length=1)
    family_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    split: Literal["development", "validation", "challenge", "sealed_test"]
    source: SourceMetadata
    risk_level: Literal["low", "medium", "high"]
    question: str = Field(min_length=1)
    expected: ExpectedAnswer
    tags: list[str] = Field(default_factory=list)


class ModelObservation(Contract):
    sample_id: str
    family_id: str
    total_pages: int = Field(gt=0)
    raw_output: Any = None
    model_error: str | None = None
    model_call: dict[str, Any] | None = None
    evidence_kind: Literal["test_only", "live_quality"]


class EvaluationResult(Contract):
    artifact_schema_version: Literal[2] = 2
    sample_id: str
    family_id: str
    status: Literal["passed", "failed", "inconclusive"]
    output: StructuredAnswer | None = None
    raw_output: Any = None
    scores: dict[str, float]
    reasons: dict[str, str]
    evidence_kind: Literal["test_only", "live_quality"]
