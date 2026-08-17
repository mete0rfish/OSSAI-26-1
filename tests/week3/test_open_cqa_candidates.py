import hashlib
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from scripts import run_open_cqa_candidates
from scripts.run_open_cqa_candidates import APPROVED_CAPS, _attempt_identity
from verifiable_ai_workflow.open_cqa_candidates import (
    OpenCQACase,
    build_candidate_messages,
    generate_candidate_pairs,
    load_candidate_pairs,
)


def _case(root: Path, number: int = 1) -> OpenCQACase:
    image = root / "local-data/opencqa/images/chart.jpg"
    image.parent.mkdir(parents=True, exist_ok=True)
    if not image.exists():
        Image.new("RGB", (20, 20), "white").save(image, format="JPEG")
    return OpenCQACase(
        pair_id=f"opencqa-val-{number}",
        sample_id=str(number),
        family_id=f"opencqa-val-chart-{number}",
        course_split=(
            "development" if number <= 18 else "validation" if number <= 24 else "test"
        ),
        source_split="val",
        source_revision="a" * 40,
        source_license="GPL-3.0",
        image_path="local-data/opencqa/images/chart.jpg",
        image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
        question=f"질문 {number}",
        reference_answer="TASK MODEL에 절대 보내지 않는 사람 기준 답",
    )


def _answer(value: str) -> str:
    return json.dumps(
        {
            "answer": value,
            "evidence": [
                {"evidence_id": "chart-1", "quote": "A 42%", "page_number": 1}
            ],
            "confidence": 0.9,
            "abstained": False,
            "abstention_reason": None,
            "tool_requests": [],
        },
        ensure_ascii=False,
    )


class _Provider:
    model = "nvidia_nim/google/gemma-4-31b-it"
    expected_actual_model = "google/gemma-4-31b-it"

    def __init__(self) -> None:
        self.last_call = None
        self.messages = []

    def generate(self, sample_id, messages):
        self.messages.append(messages)
        source = sample_id.rsplit("/", 1)[1]
        self.last_call = {
            "sample_id": sample_id,
            "provider_status": "success",
            "reported_actual_model": "google/gemma-4-31b-it",
            "actual_model": "google/gemma-4-31b-it",
            "response_id": f"response-{source}",
        }
        return _answer(f"{source} answer")


class _InvalidAbstentionProvider(_Provider):
    def generate(self, sample_id, messages):
        output = super().generate(sample_id, messages)
        if sample_id.endswith("/baseline"):
            return json.dumps(
                {
                    "answer": "The chart does not answer this question.",
                    "evidence": [],
                    "confidence": 0.0,
                    "abstained": True,
                    "abstention_reason": "차트에서 답을 확인할 수 없음",
                    "tool_requests": [],
                }
            )
        return output


def _prompts(root: Path) -> tuple[Path, Path]:
    baseline = root / "open-cqa-answer-baseline.md"
    improved = root / "open-cqa-answer-improved.md"
    baseline.write_text("baseline prompt", encoding="utf-8")
    improved.write_text("improved prompt", encoding="utf-8")
    return baseline, improved


def test_probe_and_full_caps_are_exact_and_symmetric() -> None:
    assert APPROVED_CAPS[1].model_dump() == {
        "max_requests": 2,
        "max_attempts": 2,
        "max_input_tokens": 40_000,
        "max_output_tokens": 1_000,
        "max_cost_usd": 0.01,
        "max_wall_seconds": 300.0,
    }
    assert APPROVED_CAPS[30].model_dump() == {
        "max_requests": 60,
        "max_attempts": 60,
        "max_input_tokens": 1_200_000,
        "max_output_tokens": 30_000,
        "max_cost_usd": 0.01,
        "max_wall_seconds": 7_200.0,
    }


def test_candidate_config_uses_opencqa_inputs_and_full_limits(project_root: Path) -> None:
    settings = run_open_cqa_candidates.load_settings(
        project_root / "configs/week-03-candidates.yaml"
    )

    assert settings.paths.case_authoring == "data/opencqa/week-03-selection.yaml"
    assert settings.paths.cases == "local-data/opencqa/week-03-cases.jsonl"
    assert settings.paths.prompt == "prompts/open-cqa-answer-baseline.md"
    assert settings.paths.raw_documents == "local-data/opencqa/images"
    assert settings.paths.prepared_documents == "local-data/opencqa/images"
    assert settings.paths.output == "reports/week-03/student-full"
    assert settings.documents.model_image_max_bytes == 175_000
    assert settings.documents.model_image_max_width == 1_024
    assert settings.provider.model == "nvidia_nim/google/gemma-4-31b-it"
    assert settings.limits.max_requests == 60
    assert settings.limits.requests_per_minute == 20
    assert settings.limits.max_retries == 0
    assert settings.limits.max_input_tokens == 1_200_000
    assert settings.limits.max_output_tokens == 30_000
    assert settings.limits.max_cost_usd == 0.01
    assert settings.limits.max_wall_seconds == 7_200


def test_candidate_preflight_dates_must_be_today() -> None:
    run_open_cqa_candidates._validate_dates(date.today(), date.today())

    with pytest.raises(SystemExit, match="실행 당일"):
        run_open_cqa_candidates._validate_dates(
            date.today() - timedelta(days=1), date.today()
        )


def test_task_messages_contain_only_question_and_chart_not_reference(tmp_path: Path) -> None:
    case = _case(tmp_path)
    image = tmp_path / case.image_path

    messages = build_candidate_messages(case, image, "system prompt")
    serialized = json.dumps(messages, ensure_ascii=False)

    assert case.question in serialized
    assert "data:image/jpeg;base64," in serialized
    assert case.reference_answer not in serialized
    assert "reference_answer" not in serialized


def test_generation_blinds_sources_and_binds_model_prompt_input_provenance(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    baseline, improved = _prompts(tmp_path)
    provider = _Provider()

    pairs = generate_candidate_pairs(
        cases=[case],
        project_root=tmp_path,
        baseline_prompt_path=baseline,
        improved_prompt_path=improved,
        provider=provider,
    )
    pair = pairs[0]

    assert {pair.candidate_a_source, pair.candidate_b_source} == {"baseline", "improved"}
    assert {
        (pair.candidate_a_source, pair.candidate_a),
        (pair.candidate_b_source, pair.candidate_b),
    } == {("baseline", "baseline answer"), ("improved", "improved answer")}
    assert pair.candidate_a_provenance.input_sha256 == pair.candidate_b_provenance.input_sha256
    assert pair.candidate_a_provenance.prompt_sha256 != (
        pair.candidate_b_provenance.prompt_sha256
    )
    assert pair.candidate_a_provenance.actual_model == "google/gemma-4-31b-it"
    assert len(pair.candidate_set_sha256) == 64
    assert all(case.reference_answer not in json.dumps(messages) for messages in provider.messages)

    path = tmp_path / "candidate-results.jsonl"
    path.write_text(pair.model_dump_json() + "\n", encoding="utf-8")
    assert load_candidate_pairs(path) == pairs
    row = json.loads(path.read_text(encoding="utf-8"))
    row["candidate_a"] += " tampered"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="candidate_a|SHA-256"):
        load_candidate_pairs(path)


def test_generation_preserves_invalid_answer_as_quality_failure(tmp_path: Path) -> None:
    case = _case(tmp_path)
    baseline, improved = _prompts(tmp_path)

    pair = generate_candidate_pairs(
        cases=[case],
        project_root=tmp_path,
        baseline_prompt_path=baseline,
        improved_prompt_path=improved,
        provider=_InvalidAbstentionProvider(),
    )[0]
    baseline_is_a = pair.candidate_a_source == "baseline"

    assert (pair.candidate_a if baseline_is_a else pair.candidate_b) == (
        "The chart does not answer this question."
    )
    assert (
        pair.candidate_a_validation_status if baseline_is_a else pair.candidate_b_validation_status
    ) == "invalid_output"
    assert (pair.candidate_a_output if baseline_is_a else pair.candidate_b_output)[
        "answer"
    ] == "The chart does not answer this question."
    assert "답변 보류" in (
        pair.candidate_a_validation_error if baseline_is_a else pair.candidate_b_validation_error
    )


class _Budget:
    def __init__(self) -> None:
        self.requests = 0

    def summary(self):
        return {
            "request_count": self.requests,
            "attempt_count": self.requests,
            "caps": {
                "max_requests": 2,
                "max_attempts": 2,
                "max_input_tokens": 40_000,
                "max_output_tokens": 1_000,
                "max_cost_usd": 0.01,
                "max_wall_seconds": 300,
            },
        }


class _LiveProvider(_Provider):
    captured_messages = []
    captured_structured_output = None

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.model = kwargs["model"]
        self.expected_actual_model = kwargs["expected_actual_model"]
        self.callback = kwargs["on_response_received"]
        self.budget = _Budget()
        type(self).captured_structured_output = kwargs["structured_output"]

    def generate(self, sample_id, messages):
        self.captured_messages.append(messages)
        self.budget.requests += 1
        source = sample_id.rsplit("/", 1)[1]
        self.callback(
            {
                "sample_id": sample_id,
                "provider_status": "provider_response_received",
                "raw_response": {"content": _answer(f"{source} answer")},
            }
        )
        self.last_call = {
            "sample_id": sample_id,
            "provider_status": "success",
            "reported_actual_model": "google/gemma-4-31b-it",
            "actual_model": "google/gemma-4-31b-it",
            "response_id": f"response-{source}",
        }
        return _answer(f"{source} answer")


def _settings():
    return SimpleNamespace(
        documents=SimpleNamespace(model_image_max_bytes=175_000, model_image_max_width=1024),
        provider=SimpleNamespace(
            kind="litellm",
            model="nvidia_nim/google/gemma-4-31b-it",
            expected_actual_model="google/gemma-4-31b-it",
            api_base="https://integrate.api.nvidia.com/v1",
            api_key_env="NVIDIA_NIM_API_KEY",
            structured_output="json_schema",
            billing_basis="developer_program_free_endpoint",
            pricing_source_url="https://docs.api.nvidia.com/nim/docs/product",
            input_cost_per_token_usd=0.0,
            output_cost_per_token_usd=0.0,
            temperature=0.0,
            top_p=None,
            seed=None,
            thinking_mode="default",
            thinking_parameter="thinking",
        ),
        limits=SimpleNamespace(
            max_requests=60,
            requests_per_minute=20,
            max_retries=0,
            retry_initial_seconds=5,
            max_cost_usd=0.01,
            max_input_tokens=1_200_000,
            max_output_tokens=30_000,
            max_wall_seconds=7_200.0,
            request_input_token_ceiling=20_000,
            request_output_token_ceiling=500,
            request_timeout_seconds=120.0,
        ),
    )


def test_attempt_identity_ignores_journal_phase() -> None:
    received = {"request_number": 4, "attempt_number": 4, "provider_status": "received"}
    terminal = {"request_number": 4, "attempt_number": 4, "provider_status": "success"}

    assert _attempt_identity(received) == _attempt_identity(terminal)


def test_probe_cli_writes_two_calls_one_pair_and_complete_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases = [_case(tmp_path, number) for number in range(1, 31)]
    cases_path = tmp_path / "week-03-cases.jsonl"
    cases_path.write_text("test cases\n", encoding="utf-8")
    selection = tmp_path / "week-03-selection.yaml"
    selection.write_text(
        json.dumps(
            {
                "source_split": "val",
                "revision": "a" * 40,
                "license": "GPL-3.0",
                "course_splits": {
                    "development": [str(number) for number in range(1, 19)],
                    "validation": [str(number) for number in range(19, 25)],
                    "test": [str(number) for number in range(25, 31)],
                },
            }
        ),
        encoding="utf-8",
    )
    baseline, improved = _prompts(tmp_path)
    config = tmp_path / "week-03-candidates.yaml"
    config.write_text("test config\n", encoding="utf-8")
    (tmp_path / "uv.lock").write_text("lock\n", encoding="utf-8")
    monkeypatch.setattr(run_open_cqa_candidates, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(run_open_cqa_candidates, "CASES", cases_path)
    monkeypatch.setattr(run_open_cqa_candidates, "SELECTION", selection)
    monkeypatch.setattr(run_open_cqa_candidates, "BASELINE_PROMPT", baseline)
    monkeypatch.setattr(run_open_cqa_candidates, "IMPROVED_PROMPT", improved)
    monkeypatch.setattr(run_open_cqa_candidates, "PROVIDER_CONFIG", config)
    monkeypatch.setattr(
        run_open_cqa_candidates,
        "IMAGE_ROOT",
        (tmp_path / "local-data/opencqa/images").resolve(),
    )
    monkeypatch.setattr(run_open_cqa_candidates, "PROVENANCE_COMPONENTS", ())
    monkeypatch.setattr(run_open_cqa_candidates, "load_open_cqa_cases", lambda path: cases)
    monkeypatch.setattr(run_open_cqa_candidates, "load_settings", lambda path: _settings())
    monkeypatch.setattr(run_open_cqa_candidates, "load_project_env", lambda path: path)
    monkeypatch.setattr(run_open_cqa_candidates, "_git_state", lambda: ("a" * 40, False))
    monkeypatch.setattr(run_open_cqa_candidates, "LiteLLMProvider", _LiveProvider)
    _LiveProvider.captured_messages = []
    _LiveProvider.captured_structured_output = None
    output = tmp_path / "learner/candidate"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_open_cqa_candidates.py",
            "--live-task",
            "--pair-limit",
            "1",
            "--pair-number",
            "17",
            "--max-requests",
            "2",
            "--max-attempts",
            "2",
            "--max-retries",
            "0",
            "--max-input-tokens",
            "40000",
            "--max-output-tokens",
            "1000",
            "--max-cost-usd",
            "0.01",
            "--max-wall-seconds",
            "300",
            "--catalog-verified-on",
            date.today().isoformat(),
            "--pricing-verified-on",
            date.today().isoformat(),
            "--output",
            str(output),
        ],
    )

    assert run_open_cqa_candidates.main() == 0
    summary = json.loads((output / "candidate-summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "inconclusive"
    assert summary["observed_status"] == "complete"
    assert summary["probe_only"] is True
    assert summary["pair_numbers"] == [17]
    assert summary["completed_pair_count"] == 1
    assert summary["actual_request_count"] == summary["actual_attempt_count"] == 2
    assert summary["invalid_output_count"] == 0
    assert summary["reference_sent_to_task_model"] is False
    assert _LiveProvider.captured_structured_output == "json_schema"
    assert summary["selection_sha256"] == hashlib.sha256(selection.read_bytes()).hexdigest()
    assert len((output / "candidate-calls.jsonl").read_text().splitlines()) == 2
    assert len((output / "candidate-results.jsonl").read_text().splitlines()) == 1
    assert (output / baseline.name).read_bytes() == baseline.read_bytes()
    assert (output / improved.name).read_bytes() == improved.read_bytes()
    serialized = json.dumps(_LiveProvider.captured_messages, ensure_ascii=False)
    assert cases[16].reference_answer not in serialized
