"""OpenCQA 후보 비교에 필요한 DeepEval ArenaGEval 한 개만 만든다."""

from __future__ import annotations

import random
from pathlib import Path

import yaml
from deepeval.metrics import ArenaGEval
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import ArenaTestCase, Contestant, LLMTestCase, SingleTurnParams

from .judge_comparison import JudgePair, Preference


def load_evaluation_steps(rubric_path: str | Path) -> list[str]:
    path = Path(rubric_path)
    if path.stat().st_size > 10_000:
        raise ValueError("Judge rubric은 10,000 bytes 이하여야 합니다")
    rubric = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(rubric, dict) or set(rubric) != {"evaluation_steps"}:
        raise ValueError("Judge rubric에는 evaluation_steps만 있어야 합니다")
    steps = rubric["evaluation_steps"]
    if not isinstance(steps, list) or not 4 <= len(steps) <= 8:
        raise ValueError("Judge rubric에는 평가 단계가 4~8개 있어야 합니다")
    if any(
        not isinstance(step, str)
        or not 10 <= len(step.strip()) <= 500
        or "CHANGE_ME" in step
        for step in steps
    ):
        raise ValueError("각 평가 단계는 CHANGE_ME 없이 10~500자로 작성해야 합니다")
    return [step.strip() for step in steps]


def build_arena_metric(model: DeepEvalBaseLLM, rubric_path: str | Path) -> ArenaGEval:
    return ArenaGEval(
        name="OpenCQA Better Answer",
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        evaluation_steps=load_evaluation_steps(rubric_path),
        model=model,
        async_mode=False,
    )


def build_arena_case(pair: JudgePair, *, reverse: bool = False) -> ArenaTestCase:
    contestants = [
        Contestant(
            name="response_1",
            test_case=LLMTestCase(
                input=pair.question,
                actual_output=pair.candidate_a,
                expected_output=pair.reference_answer,
            ),
        ),
        Contestant(
            name="response_2",
            test_case=LLMTestCase(
                input=pair.question,
                actual_output=pair.candidate_b,
                expected_output=pair.reference_answer,
            ),
        ),
    ]
    if reverse:
        contestants.reverse()
    return ArenaTestCase(contestants=contestants)


def measure(
    metric: ArenaGEval,
    pair: JudgePair,
    *,
    reverse: bool = False,
    random_seed: int = 0,
    max_retries: int = 0,
) -> tuple[Preference, str]:
    random_state = random.getstate()
    try:
        for attempt in range(max_retries + 1):
            random.seed(random_seed)
            try:
                winner = metric.measure(
                    build_arena_case(pair, reverse=reverse),
                    _show_indicator=False,
                )
            except KeyError as exc:
                if exc.args == ("tie",):
                    winner = "tie"
                    metric.reason = "Judge가 두 후보를 동률로 판정했습니다."
                    break
                if attempt == max_retries:
                    raise
                model = getattr(metric, "model", None)
                if hasattr(model, "invalid_winner_retry_count"):
                    model.invalid_winner_retry_count += 1
                continue
            break
    finally:
        random.setstate(random_state)
    winner_map: dict[str, Preference] = {
        "response_1": "candidate_a",
        "tie": "tie",
        "response_2": "candidate_b",
    }
    if winner not in winner_map:
        raise ValueError(f"알 수 없는 Judge winner: {winner}")
    return winner_map[winner], metric.reason
