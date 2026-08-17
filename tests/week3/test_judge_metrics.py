import json
import random
import re

import pytest
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import SingleTurnParams
from pydantic import BaseModel

from verifiable_ai_workflow.judge_metrics import (
    build_arena_case,
    build_arena_metric,
    load_evaluation_steps,
    measure,
)
from verifiable_ai_workflow.judge_model import CourseJudgeModel
from verifiable_ai_workflow.open_cqa_candidates import (
    CandidatePairDraft,
    CandidateProvenance,
    bind_candidate_set_sha256,
)
from verifiable_ai_workflow.schemas import Evidence, StructuredAnswer


class NoCallJudge(DeepEvalBaseLLM):
    def load_model(self):
        return self

    def get_model_name(self, *args, **kwargs):
        return "no-call"

    def generate(self, *args, **kwargs):
        raise AssertionError("metric 생성 중 API를 호출하면 안 됩니다")

    async def a_generate(self, *args, **kwargs):
        return self.generate(*args, **kwargs)


def _output(answer: str) -> StructuredAnswer:
    return StructuredAnswer(
        answer=answer,
        evidence=[Evidence(evidence_id="chart-1", quote="42%", page_number=1)],
        confidence=0.9,
    )


def _provenance(source: str) -> CandidateProvenance:
    return CandidateProvenance(
        source=source,
        call_id=f"pair-1/{source}",
        requested_model="nvidia_nim/google/gemma-4-31b-it",
        expected_actual_model="google/gemma-4-31b-it",
        actual_model="google/gemma-4-31b-it",
        prompt_file=f"{source}.md",
        prompt_sha256=("a" if source == "baseline" else "b") * 64,
        input_sha256="c" * 64,
    )


def _pair():
    output_a, output_b = _output("A"), _output("B")
    return bind_candidate_set_sha256(
        [
            CandidatePairDraft(
                pair_id="pair-1",
                sample_id="1",
                family_id="family-1",
                course_split="development",
                source_split="val",
                source_revision="a" * 40,
                source_license="GPL-3.0",
                image_sha256="a" * 64,
                image_path="1.jpg",
                question="question",
                reference_answer="reference",
                candidate_a="A",
                candidate_b="B",
                candidate_a_source="baseline",
                candidate_b_source="improved",
                candidate_a_output=output_a.model_dump(mode="json"),
                candidate_b_output=output_b.model_dump(mode="json"),
                candidate_a_validation_status="valid_output",
                candidate_b_validation_status="valid_output",
                candidate_a_provenance=_provenance("baseline"),
                candidate_b_provenance=_provenance("improved"),
            )
        ]
    )[0]


def test_fixed_rubric_requires_explicit_bounded_steps(tmp_path) -> None:
    rubric = tmp_path / "judge-rubric.yaml"
    rubric.write_text(
        "evaluation_steps:\n"
        + "".join(f'  - "명확한 공통 평가 단계 {number}입니다"\n' for number in range(1, 5)),
        encoding="utf-8",
    )

    assert len(load_evaluation_steps(rubric)) == 4

    rubric.write_text(
        "evaluation_steps:\n"
        + '  - "CHANGE_ME: 기준"\n'
        + "".join(f'  - "명확한 공통 평가 단계 {number}입니다"\n' for number in range(2, 5)),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="CHANGE_ME"):
        load_evaluation_steps(rubric)


def test_arena_uses_neutral_names_and_reference_as_expected_output(project_root) -> None:
    normal = build_arena_case(_pair())
    reversed_case = build_arena_case(_pair(), reverse=True)

    assert [item.name for item in normal.contestants] == ["response_1", "response_2"]
    assert [item.name for item in reversed_case.contestants] == ["response_2", "response_1"]
    assert all(item.test_case.expected_output == "reference" for item in normal.contestants)
    metric = build_arena_metric(NoCallJudge(), project_root / "configs/week-03-judge-rubric.yaml")
    assert metric.name == "OpenCQA Better Answer"
    assert SingleTurnParams.EXPECTED_OUTPUT in metric.evaluation_params


def test_tie_lookup_bug_is_kept_as_tie() -> None:
    class TieMetric:
        def measure(self, test_case, _show_indicator=False):
            del test_case, _show_indicator
            raise KeyError("tie")

    assert measure(TieMetric(), _pair()) == (
        "tie",
        "Judge가 두 후보를 동률로 판정했습니다.",
    )


def test_neutral_winner_name_maps_back_and_retries_once() -> None:
    class Model:
        invalid_winner_retry_count = 0

    class InvalidThenValidMetric:
        model = Model()
        reason = "첫 응답이 더 정확합니다."

        def __init__(self) -> None:
            self.calls = 0

        def measure(self, test_case, _show_indicator=False):
            del test_case, _show_indicator
            self.calls += 1
            if self.calls == 1:
                raise KeyError("Eve, 2024-05-22T10:00:00Z")
            return "response_1"

    metric = InvalidThenValidMetric()
    assert measure(metric, _pair(), max_retries=1) == (
        "candidate_a",
        "첫 응답이 더 정확합니다.",
    )
    assert metric.calls == 2
    assert metric.model.invalid_winner_retry_count == 1


def test_locked_deepeval_tie_uses_only_winner_call(project_root) -> None:
    class TieProvider:
        model = "tie-provider"
        last_call = None

        def __init__(self) -> None:
            self.schema_calls: list[str] = []

        def generate(self, call_id, messages, *, response_schema=None):
            del call_id, messages
            self.schema_calls.append(response_schema.__name__)
            return {"winner": "tie", "reason": "두 답이 같습니다"}

    provider = TieProvider()
    metric = build_arena_metric(
        CourseJudgeModel(provider),
        project_root / "configs/week-03-judge-rubric.yaml",
    )

    winner, _reason = measure(metric, _pair())

    assert winner == "tie"
    assert provider.schema_calls == ["Winner"]


def test_same_seed_reverses_prompt_without_source_identity(project_root) -> None:
    class RecordingProvider:
        model = "recording-provider"
        last_call = None

        def __init__(self) -> None:
            self.winner_prompts: list[str] = []

        def generate(self, call_id, messages, *, response_schema=None):
            del call_id
            prompt = messages[0]["content"]
            if response_schema.__name__ == "Winner":
                self.winner_prompts.append(prompt)
                winner = re.search(r'"arena_test_cases": \{\s*"([^"]+)"', prompt).group(1)
                return json.dumps({"winner": winner, "reason": "첫 응답이 더 낫다"})
            return json.dumps({"rewritten_reason": "첫 응답이 더 낫다"})

    provider = RecordingProvider()
    metric = build_arena_metric(
        CourseJudgeModel(provider),
        project_root / "configs/week-03-judge-rubric.yaml",
    )
    random.seed(12345)
    state = random.getstate()

    measure(metric, _pair(), random_seed=17)
    assert random.getstate() == state
    measure(metric, _pair(), reverse=True, random_seed=17)
    assert random.getstate() == state

    normal, reversed_prompt = provider.winner_prompts
    assert "baseline" not in normal + reversed_prompt
    assert "improved" not in normal + reversed_prompt
    assert "reference" in normal
    escaped_a = r'\"actual_output\": \"A\"'
    escaped_b = r'\"actual_output\": \"B\"'
    assert (normal.index(escaped_a) < normal.index(escaped_b)) != (
        reversed_prompt.index(escaped_a) < reversed_prompt.index(escaped_b)
    )


def test_course_judge_sends_current_chart_as_image(tmp_path) -> None:
    class RewrittenReason(BaseModel):
        rewritten_reason: str

    class RecordingProvider:
        model = "recording-provider"
        last_call = None

        def __init__(self) -> None:
            self.message_history = []

        def generate(self, call_id, messages, *, response_schema=None):
            del call_id
            self.message_history.append(messages)
            if response_schema is not None:
                return {"rewritten_reason": "A가 더 정확합니다"}
            return "response_1"

    image = tmp_path / "chart.png"
    image.write_bytes(b"chart")
    provider = RecordingProvider()
    model = CourseJudgeModel(provider)
    model.image_path = image

    assert model.generate("두 응답을 비교하세요") == "response_1"
    content = provider.message_history[0][0]["content"]
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[1] == {"type": "text", "text": "두 응답을 비교하세요"}

    model.generate("이유를 다시 쓰세요", schema=RewrittenReason)
    assert provider.message_history[1][0]["content"] == "이유를 다시 쓰세요"


def test_course_judge_retries_one_malformed_structured_response() -> None:
    class Winner(BaseModel):
        winner: str

    class MalformedOnceProvider:
        model = "malformed-once"
        last_call = None

        def __init__(self) -> None:
            self.calls = 0

        def generate(self, call_id, messages, *, response_schema=None):
            del call_id, messages, response_schema
            self.calls += 1
            return '{"winner": "tie"' if self.calls == 1 else {"winner": "tie"}

    provider = MalformedOnceProvider()
    model = CourseJudgeModel(provider, max_validation_retries=1)

    assert model.generate("두 응답을 비교하세요", schema=Winner).winner == "tie"
    assert provider.calls == 2
    assert model.structured_output_retry_count == 1
