"""OpenCQA 차트에서 두 prompt의 답을 만들고 blind A/B pair로 묶는다."""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .evaluation.scoring import parse_json_output
from .schemas import StructuredAnswer

CandidateSource = Literal["baseline", "improved"]
CandidateValidationStatus = Literal["valid_output", "invalid_output"]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OpenCQACase(_StrictModel):
    artifact_schema_version: Literal[1] = 1
    pair_id: str = Field(min_length=1)
    sample_id: str = Field(min_length=1)
    family_id: str = Field(min_length=1)
    course_split: Literal["development", "validation", "test"]
    source_split: str = Field(min_length=1)
    source_revision: str = Field(min_length=1)
    source_license: str = Field(min_length=1)
    image_path: str = Field(min_length=1)
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    question: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1)


class CandidateProvenance(_StrictModel):
    source: CandidateSource
    call_id: str = Field(min_length=1)
    requested_model: str = Field(min_length=1)
    expected_actual_model: str = Field(min_length=1)
    reported_actual_model: str | None = None
    actual_model: str = Field(min_length=1)
    response_id: str | None = None
    prompt_file: str = Field(min_length=1)
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def actual_model_matches_expected(self) -> CandidateProvenance:
        if self.actual_model != self.expected_actual_model:
            raise ValueError("task model actual model이 승인된 model과 다릅니다")
        return self


class CandidateAnswer(_StrictModel):
    source: CandidateSource
    output: dict[str, Any]
    validation_status: CandidateValidationStatus
    validation_error: str | None = None
    provenance: CandidateProvenance

    @model_validator(mode="after")
    def source_matches_provenance(self) -> CandidateAnswer:
        if self.source != self.provenance.source:
            raise ValueError("candidate source와 provenance source가 다릅니다")
        _candidate, output, status, error = validate_candidate_output(self.output)
        if (output, status, error) != (
            self.output,
            self.validation_status,
            self.validation_error,
        ):
            raise ValueError("candidate output 검증 기록이 다릅니다")
        return self


class CandidatePairDraft(_StrictModel):
    artifact_schema_version: Literal[2] = 2
    pair_id: str
    sample_id: str
    family_id: str
    course_split: Literal["development", "validation", "test"]
    source_split: str
    source_revision: str
    source_license: str
    image_path: str
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    question: str
    reference_answer: str
    candidate_a: str
    candidate_b: str
    candidate_a_source: CandidateSource
    candidate_b_source: CandidateSource
    candidate_a_output: dict[str, Any]
    candidate_b_output: dict[str, Any]
    candidate_a_validation_status: CandidateValidationStatus
    candidate_b_validation_status: CandidateValidationStatus
    candidate_a_validation_error: str | None = None
    candidate_b_validation_error: str | None = None
    candidate_a_provenance: CandidateProvenance
    candidate_b_provenance: CandidateProvenance

    @model_validator(mode="after")
    def candidates_match_sources_and_outputs(self) -> CandidatePairDraft:
        if {self.candidate_a_source, self.candidate_b_source} != {"baseline", "improved"}:
            raise ValueError("candidate A/B에 baseline과 improved가 하나씩 필요합니다")
        for name, candidate, output, status, error in (
            (
                "candidate_a",
                self.candidate_a,
                self.candidate_a_output,
                self.candidate_a_validation_status,
                self.candidate_a_validation_error,
            ),
            (
                "candidate_b",
                self.candidate_b,
                self.candidate_b_output,
                self.candidate_b_validation_status,
                self.candidate_b_validation_error,
            ),
        ):
            actual = validate_candidate_output(output)
            if actual != (candidate, output, status, error):
                raise ValueError(f"{name}의 output 검증 기록이 다릅니다")
        if self.candidate_a_source != self.candidate_a_provenance.source:
            raise ValueError("candidate_a source와 provenance가 다릅니다")
        if self.candidate_b_source != self.candidate_b_provenance.source:
            raise ValueError("candidate_b source와 provenance가 다릅니다")
        return self


class OpenCQACandidatePair(CandidatePairDraft):
    candidate_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_candidate_output(
    raw_output: Any,
) -> tuple[str, dict[str, Any], CandidateValidationStatus, str | None]:
    parsed_output = parse_json_output(raw_output)
    if not isinstance(parsed_output, dict):
        raise ValueError("JSON object 응답이 아닙니다")
    candidate = parsed_output.get("answer")
    if not isinstance(candidate, str) or not candidate.strip():
        raise ValueError("비어 있지 않은 answer가 필요합니다")
    try:
        StructuredAnswer.model_validate(parsed_output)
        return candidate, parsed_output, "valid_output", None
    except ValidationError as exc:
        error = str(exc.errors(include_url=False)[0]["msg"])
        return candidate, parsed_output, "invalid_output", error


def _without_set_hash(pair: CandidatePairDraft | OpenCQACandidatePair | dict[str, Any]) -> dict:
    if isinstance(pair, BaseModel):
        value = pair.model_dump(mode="json")
    else:
        value = dict(pair)
    value.pop("candidate_set_sha256", None)
    return value


def candidate_set_sha256(
    pairs: Sequence[CandidatePairDraft | OpenCQACandidatePair | dict[str, Any]],
) -> str:
    """Embedded hash 필드를 제외한 순서 고정 pair payload의 SHA-256."""

    return _canonical_sha256([_without_set_hash(pair) for pair in pairs])


def bind_candidate_set_sha256(
    pairs: Sequence[CandidatePairDraft],
) -> list[OpenCQACandidatePair]:
    digest = candidate_set_sha256(pairs)
    return [
        OpenCQACandidatePair.model_validate(
            {**pair.model_dump(mode="json"), "candidate_set_sha256": digest}
        )
        for pair in pairs
    ]


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {source}")
    return [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_open_cqa_cases(path: str | Path) -> list[OpenCQACase]:
    cases = [OpenCQACase.model_validate(row) for row in _read_jsonl(path)]
    pair_ids = [case.pair_id for case in cases]
    if not cases or len(pair_ids) != len(set(pair_ids)):
        raise ValueError("OpenCQA case는 비어 있거나 pair_id가 중복될 수 없습니다")
    return cases


def load_candidate_pairs(path: str | Path) -> list[OpenCQACandidatePair]:
    pairs = [OpenCQACandidatePair.model_validate(row) for row in _read_jsonl(path)]
    pair_ids = [pair.pair_id for pair in pairs]
    if not pairs or len(pair_ids) != len(set(pair_ids)):
        raise ValueError("candidate pair는 비어 있거나 pair_id가 중복될 수 없습니다")
    expected = candidate_set_sha256(pairs)
    if {pair.candidate_set_sha256 for pair in pairs} != {expected}:
        raise ValueError("candidate set SHA-256이 현재 pair payload와 다릅니다")
    return pairs


def task_input_sha256(case: OpenCQACase) -> str:
    """Model에 보내는 질문·이미지만 hash하고 reference는 제외한다."""

    return _canonical_sha256(
        {
            "question": case.question,
            "image_sha256": case.image_sha256,
        }
    )


def build_candidate_messages(
    case: OpenCQACase,
    image_path: str | Path,
    system_prompt: str,
) -> list[dict[str, Any]]:
    image = Path(image_path).read_bytes()
    if hashlib.sha256(image).hexdigest() != case.image_sha256:
        raise ValueError(f"OpenCQA 이미지 SHA-256이 다릅니다: {case.pair_id}")
    data_url = "data:image/jpeg;base64," + base64.b64encode(image).decode("ascii")
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"질문: {case.question}"},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]


def _provenance(
    *,
    source: CandidateSource,
    pair_id: str,
    prompt_path: Path,
    prompt_sha256: str,
    input_sha256: str,
    provider: Any,
) -> CandidateProvenance:
    call = provider.last_call or {}
    call_id = f"{pair_id}/{source}"
    if call.get("provider_status") != "success":
        raise ValueError(f"{call_id} task model 호출이 성공 상태가 아닙니다")
    if call.get("sample_id") != call_id:
        raise ValueError(f"{call_id} task model 호출 ID가 provider 기록과 다릅니다")
    return CandidateProvenance(
        source=source,
        call_id=call_id,
        requested_model=provider.model,
        expected_actual_model=provider.expected_actual_model,
        reported_actual_model=call.get("reported_actual_model"),
        actual_model=call.get("actual_model"),
        response_id=call.get("response_id"),
        prompt_file=prompt_path.name,
        prompt_sha256=prompt_sha256,
        input_sha256=input_sha256,
    )


def _draft_pair(
    case: OpenCQACase,
    answers: dict[CandidateSource, CandidateAnswer],
) -> CandidatePairDraft:
    if int(hashlib.sha256(case.pair_id.encode()).hexdigest(), 16) % 2:
        candidate_a, candidate_b = answers["baseline"], answers["improved"]
    else:
        candidate_a, candidate_b = answers["improved"], answers["baseline"]
    return CandidatePairDraft(
        artifact_schema_version=2,
        **case.model_dump(mode="json", exclude={"artifact_schema_version"}),
        candidate_a=candidate_a.output["answer"],
        candidate_b=candidate_b.output["answer"],
        candidate_a_source=candidate_a.source,
        candidate_b_source=candidate_b.source,
        candidate_a_output=candidate_a.output,
        candidate_b_output=candidate_b.output,
        candidate_a_validation_status=candidate_a.validation_status,
        candidate_b_validation_status=candidate_b.validation_status,
        candidate_a_validation_error=candidate_a.validation_error,
        candidate_b_validation_error=candidate_b.validation_error,
        candidate_a_provenance=candidate_a.provenance,
        candidate_b_provenance=candidate_b.provenance,
    )


def generate_candidate_pairs(
    *,
    cases: Sequence[OpenCQACase],
    project_root: str | Path,
    baseline_prompt_path: str | Path,
    improved_prompt_path: str | Path,
    provider: Any,
    on_pair: Callable[[CandidatePairDraft], None] | None = None,
) -> list[OpenCQACandidatePair]:
    """Case당 baseline·improved 한 번씩 호출하고 두 답을 blind로 배치한다."""

    root = Path(project_root)
    prompt_paths = {
        "baseline": Path(baseline_prompt_path),
        "improved": Path(improved_prompt_path),
    }
    prompts = {source: path.read_text(encoding="utf-8") for source, path in prompt_paths.items()}
    prompt_hashes = {
        source: hashlib.sha256(path.read_bytes()).hexdigest()
        for source, path in prompt_paths.items()
    }
    drafts: list[CandidatePairDraft] = []
    for case in cases:
        input_hash = task_input_sha256(case)
        image_path = root / case.image_path
        answers: dict[CandidateSource, CandidateAnswer] = {}
        for source in ("baseline", "improved"):
            raw_output = provider.generate(
                f"{case.pair_id}/{source}",
                build_candidate_messages(case, image_path, prompts[source]),
            )
            try:
                _candidate, output, validation_status, validation_error = validate_candidate_output(
                    raw_output
                )
            except ValueError as exc:
                raise ValueError(f"{case.pair_id}/{source}: {exc}") from exc
            provenance = _provenance(
                source=source,
                pair_id=case.pair_id,
                prompt_path=prompt_paths[source],
                prompt_sha256=prompt_hashes[source],
                input_sha256=input_hash,
                provider=provider,
            )
            answers[source] = CandidateAnswer(
                source=source,
                output=output,
                validation_status=validation_status,
                validation_error=validation_error,
                provenance=provenance,
            )
        draft = _draft_pair(case, answers)
        drafts.append(draft)
        if on_pair is not None:
            on_pair(draft)
    return bind_candidate_set_sha256(drafts)
