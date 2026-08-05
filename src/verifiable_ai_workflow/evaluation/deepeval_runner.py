"""Week 1의 결정적 정량 점수를 로컬 DeepEval TestRun으로 저장한다."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("DEEPEVAL_DISABLE_DOTENV", "1")
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig, CacheConfig, DisplayConfig
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from ..schemas import EvaluationCase, EvaluationResult


class ResultMetric(BaseMetric):
    async_mode = False

    def __init__(
        self,
        score_name: str,
        display_name: str,
        values: dict[str, float],
        reasons: dict[str, str],
        *,
        threshold: float,
    ) -> None:
        self.score_name = score_name
        self.display_name = display_name
        self.values = values
        self.reasons = reasons
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        del args, kwargs
        self.score = self.values[test_case.name]
        self.reason = self.reasons[test_case.name]
        self.success = self.score >= self.threshold
        self.error = None
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:
        return self.display_name


def deterministic_metrics(results: list[EvaluationResult]) -> list[BaseMetric]:
    # threshold=0인 지표는 원인 분석용이다. 실제 pass/fail은 아래 필수 지표와
    # 같은 기준으로 계산된 task_success가 결정한다.
    metric_specs = (
        ("json_object_only", "JSON 단독 반환 (진단)", 0.0),
        ("schema_validity", "JSON 구조", 1.0),
        ("answer_exact", "정답 완전일치 (진단)", 0.0),
        ("answer_similarity", "정답 문자 유사도 (진단)", 0.0),
        ("answer_anls", "DocVQA ANLS (진단)", 0.0),
        ("answer_token_f1", "정답 token F1 (진단)", 0.0),
        ("numeric_match", "숫자 일치 (진단)", 0.0),
        ("abstention_correct", "답변 보류", 1.0),
        ("answer_correct", "정답 허용 기준", 1.0),
        ("evidence_page_precision", "근거 페이지 precision (진단)", 0.0),
        ("evidence_page_recall", "근거 페이지 recall (진단)", 0.0),
        ("evidence_page_f1", "근거 페이지 F1 (진단)", 0.0),
        ("evidence_coverage", "가능한 근거 페이지", 1.0),
        ("quote_answer_support", "인용문·답 자체 일관성 (진단)", 0.0),
        ("task_success", "전체 성공", 1.0),
    )
    return [
        ResultMetric(
            score_name,
            display_name,
            {result.sample_id: result.scores[score_name] for result in results},
            {result.sample_id: result.reasons[score_name] for result in results},
            threshold=threshold,
        )
        for score_name, display_name, threshold in metric_specs
    ]


def evaluate_results(
    results: list[EvaluationResult],
    cases: list[EvaluationCase],
    results_folder: str | Path,
) -> None:
    case_by_id = {case.sample_id: case for case in cases}
    test_cases = [
        LLMTestCase(
            name=result.sample_id,
            input=case_by_id[result.sample_id].question,
            actual_output=(
                result.output.model_dump_json() if result.output else str(result.raw_output)
            ),
            expected_output=case_by_id[result.sample_id].expected.model_dump_json(),
        )
        for result in results
    ]
    evaluate(
        test_cases=test_cases,
        metrics=deterministic_metrics(results),
        async_config=AsyncConfig(run_async=False),
        cache_config=CacheConfig(write_cache=False, use_cache=False),
        display_config=DisplayConfig(
            show_indicator=False,
            print_results=False,
            inspect_after_run=False,
            results_folder=str(results_folder),
        ),
    )
