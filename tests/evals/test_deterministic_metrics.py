from pathlib import Path

from verifiable_ai_workflow.data.dataset import build_cases
from verifiable_ai_workflow.evaluation.deepeval_runner import deterministic_metrics
from verifiable_ai_workflow.evaluation.scoring import score_observations
from verifiable_ai_workflow.providers.recorded import RecordedProvider
from verifiable_ai_workflow.schemas import ModelObservation


def test_deepeval_metrics_include_quantitative_scores(project_root: Path) -> None:
    case = build_cases(project_root / "data/cases/week-01-aihub.yaml")[0]
    response = RecordedProvider(project_root / "tests/fixtures/recorded-responses.jsonl").generate(
        case.sample_id, []
    )
    result = score_observations(
        [case],
        [
            ModelObservation(
                sample_id=case.sample_id,
                family_id=case.family_id,
                total_pages=9,
                raw_output=response,
                evidence_kind="test_only",
            )
        ],
    )[0]

    names = {metric.score_name for metric in deterministic_metrics([result])}

    assert "answer_similarity" in names
    assert "numeric_match" in names
    assert "evidence_coverage" in names
    assert "quote_answer_support" in names
    assert "quote_verifiability" not in names
    assert "quote_grounding" not in names
