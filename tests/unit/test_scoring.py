from pathlib import Path

from verifiable_ai_workflow.data.dataset import build_cases
from verifiable_ai_workflow.evaluation.scoring import score_observations, score_output
from verifiable_ai_workflow.providers.recorded import RecordedProvider
from verifiable_ai_workflow.schemas import ModelObservation


def test_expected_answer_and_page_pass(project_root: Path) -> None:
    case = build_cases(project_root / "data/cases/week-01-aihub.yaml")[0]
    provider = RecordedProvider(project_root / "tests/fixtures/recorded-responses.jsonl")

    _, scores, _ = score_output(provider.generate(case.sample_id, []), case)

    assert scores["schema_validity"] == 1.0
    assert scores["answer_exact"] == 1.0
    assert scores["answer_similarity"] == 1.0
    assert scores["answer_anls"] == 1.0
    assert scores["answer_token_f1"] == 1.0
    assert scores["numeric_match"] == 1.0
    assert scores["evidence_page_f1"] == 1.0
    assert scores["task_success"] == 1.0


def test_numeric_answer_allows_wording_variation(project_root: Path) -> None:
    case = build_cases(project_root / "data/cases/week-01-aihub.yaml")[0]
    response = {
        "answer": "약 71.6 퍼센트입니다.",
        "evidence": [
            {
                "evidence_id": "page-1",
                "quote": "변동금리 비중은 71.6%",
                "page_number": 1,
            }
        ],
        "confidence": 0.8,
        "abstained": False,
        "abstention_reason": None,
        "tool_requests": [],
    }

    _, scores, _ = score_output(response, case)

    assert scores["answer_exact"] == 0.0
    assert scores["numeric_match"] == 1.0
    assert 0.0 < scores["answer_similarity"] < 1.0
    assert scores["answer_anls"] == 0.0
    assert 0.0 < scores["answer_token_f1"] < 1.0
    assert scores["answer_correct"] == 1.0


def test_quote_answer_support_checks_only_model_output_consistency(
    project_root: Path,
) -> None:
    case = build_cases(project_root / "data/cases/week-01-aihub.yaml")[1]
    response = {
        "answer": "2.2%",
        "evidence": [
            {
                "evidence_id": "page-1",
                "quote": "2017년은 전망치 2.2",
                "page_number": 1,
            }
        ],
        "confidence": 0.9,
        "abstained": False,
        "abstention_reason": None,
        "tool_requests": [],
    }
    _, scores, reasons = score_output(response, case)

    assert scores["quote_answer_support"] == 1.0
    assert scores["task_success"] == 1.0
    assert "이미지 근거 일치 여부를 증명" in reasons["quote_answer_support"]


def test_quote_answer_support_does_not_gate_task_success(
    project_root: Path,
) -> None:
    case = build_cases(project_root / "data/cases/week-01-aihub.yaml")[13]
    response = {
        "answer": "37.9만호",
        "evidence": [
            {
                "evidence_id": "page-5",
                "quote": "값을 확인했습니다.",
                "page_number": 5,
            }
        ],
        "confidence": 0.9,
        "abstained": False,
        "abstention_reason": None,
        "tool_requests": [],
    }
    _, scores, _ = score_output(response, case)

    assert scores["quote_answer_support"] == 0.0
    assert scores["task_success"] == 1.0
    assert "quote_verifiability" not in scores
    assert "quote_grounding" not in scores


def test_one_of_multiple_acceptable_pages_is_enough(project_root: Path) -> None:
    case = build_cases(project_root / "data/cases/week-01-aihub.yaml")[-1]
    response = {
        "answer": "분당서울대학교병원",
        "evidence": [
            {
                "evidence_id": "page-1",
                "quote": "수도권 분당서울대학교병원",
                "page_number": 1,
            }
        ],
        "confidence": 0.9,
        "abstained": False,
        "abstention_reason": None,
        "tool_requests": [],
    }

    _, scores, _ = score_output(response, case)

    assert case.expected.pages == [1, 3]
    assert scores["evidence_page_precision"] == 1.0
    assert scores["evidence_page_recall"] == 0.5
    assert scores["evidence_coverage"] == 1.0
    assert scores["task_success"] == 1.0


def test_provider_error_is_inconclusive(project_root: Path) -> None:
    case = build_cases(project_root / "data/cases/week-01-aihub.yaml")[0]
    observation = ModelObservation(
        sample_id=case.sample_id,
        family_id=case.family_id,
        total_pages=2,
        model_error="TimeoutError: provider timeout",
        evidence_kind="live_quality",
    )

    result = score_observations([case], [observation])[0]

    assert result.status == "inconclusive"
    assert not result.scores["task_success"]
