from pathlib import Path

from PIL import Image

from verifiable_ai_workflow.data.dataset import build_cases
from verifiable_ai_workflow.evaluation.scoring import score_observations
from verifiable_ai_workflow.preprocessing import prepare_pdf
from verifiable_ai_workflow.providers.recorded import RecordedProvider
from verifiable_ai_workflow.workflow import run_cases


def _write_test_pdf(path: Path) -> None:
    page = Image.new("RGB", (400, 600), "white")
    page.save(path, format="PDF")
    page.close()


def test_preparation_workflow_and_scoring_connect(
    project_root: Path,
    tmp_path: Path,
) -> None:
    source = tmp_path / "MI2_240819_TY1_0012.pdf"
    _write_test_pdf(source)
    prepared_root = tmp_path / "processed"
    prepare_pdf(
        source,
        prepared_root / "MI2_240819_TY1_0012",
        document_id="MI2_240819_TY1_0012",
    )
    for text_path in (prepared_root / "MI2_240819_TY1_0012/text").iterdir():
        text_path.unlink()
    cases = build_cases(project_root / "data/cases/week-01-aihub.yaml")[:1]
    observations = run_cases(
        cases=cases,
        prepared_documents=prepared_root,
        prompt_path=project_root / "prompts/pdf-question-answer.md",
        provider=RecordedProvider(project_root / "tests/fixtures/recorded-responses.jsonl"),
    )

    results = score_observations(cases, observations)

    assert len(results) == 1
    assert all(result.status == "passed" for result in results)
