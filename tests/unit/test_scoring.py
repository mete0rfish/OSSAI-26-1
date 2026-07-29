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


def test_quote_uses_answer_fact_when_pdf_line_breaks_differ(
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
    page_texts = {1: "금년중 민간소비는 2.2% 증가할 전망"}

    _, scores, _ = score_output(response, case, page_texts=page_texts)

    assert scores["quote_answer_support"] == 1.0
    assert scores["quote_verifiability"] == 1.0
    assert scores["quote_grounding"] == 1.0
    assert scores["task_success"] == 1.0


def test_unextractable_chart_value_is_not_a_false_model_failure(
    project_root: Path,
) -> None:
    case = build_cases(project_root / "data/cases/week-01-aihub.yaml")[13]
    response = {
        "answer": "37.9만호",
        "evidence": [
            {
                "evidence_id": "page-5",
                "quote": "아파트 입주물량 37.9만호",
                "page_number": 5,
            }
        ],
        "confidence": 0.9,
        "abstained": False,
        "abstention_reason": None,
        "tool_requests": [],
    }
    page_texts = {5: "아파트 입주물량 자료: 국토교통부"}

    _, scores, reasons = score_output(response, case, page_texts=page_texts)

    assert scores["quote_verifiability"] == 0.0
    assert scores["task_success"] == 1.0
    assert "gate에서 제외" in reasons["quote_grounding"]


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
