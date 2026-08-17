import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from verifiable_ai_workflow.data.dataset import build_cases
from verifiable_ai_workflow.evaluation.deepeval_runner import evaluate_results
from verifiable_ai_workflow.evaluation.scoring import score_observations
from verifiable_ai_workflow.preprocessing import prepare_pdf
from verifiable_ai_workflow.providers.litellm_provider import LiteLLMProvider
from verifiable_ai_workflow.workflow import run_cases


def test_nvidia_response_reaches_deepeval(
    monkeypatch,
    project_root: Path,
    tmp_path: Path,
) -> None:
    source = tmp_path / "MI2_240819_TY1_0012.pdf"
    page = Image.new("RGB", (400, 600), "white")
    page.save(source, format="PDF")
    page.close()
    prepared_root = tmp_path / "prepared"
    prepare_pdf(
        source,
        prepared_root / "MI2_240819_TY1_0012",
        document_id="MI2_240819_TY1_0012",
    )

    response = {
        "answer": "71.6%",
        "evidence": [
            {
                "evidence_id": "sample#page=1",
                "quote": "71.6%",
                "page_number": 1,
            }
        ],
        "confidence": 0.9,
        "abstained": False,
        "abstention_reason": None,
        "tool_requests": [],
    }
    monkeypatch.setenv("NVIDIA_NIM_API_KEY", "test-key")
    monkeypatch.setattr(
        "verifiable_ai_workflow.providers.litellm_provider.litellm.completion",
        lambda **kwargs: SimpleNamespace(
            id="response-1",
            model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(response)))],
        ),
    )
    provider = LiteLLMProvider(
        model="nvidia_nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        expected_actual_model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        api_key_env="NVIDIA_NIM_API_KEY",
        api_base="https://integrate.api.nvidia.com/v1",
        structured_output="prompt_only",
        max_requests=1,
        requests_per_minute=1200,
        max_retries=0,
        retry_initial_seconds=1,
        max_cost_usd=0.25,
        max_input_tokens=8000,
        max_output_tokens=500,
        max_wall_seconds=120,
        input_cost_per_token_usd=0.0,
        output_cost_per_token_usd=0.0,
    )
    cases = build_cases(project_root / "data/cases/week-01-aihub.yaml")[:1]

    observations = run_cases(
        cases=cases,
        prepared_documents=prepared_root,
        prompt_path=project_root / "prompts/pdf-question-answer.md",
        provider=provider,
    )
    results = score_observations(cases, observations)
    deepeval_dir = tmp_path / "deepeval"
    evaluate_results(results, cases, deepeval_dir)

    assert results[0].status == "passed"
    assert list(deepeval_dir.glob("test_run_*.json"))
