import importlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from verifiable_ai_workflow.live_execution import LiveBudgetExceeded
from verifiable_ai_workflow.providers import litellm_provider
from verifiable_ai_workflow.providers.litellm_provider import LiteLLMProvider
from verifiable_ai_workflow.providers.recorded import RecordedProvider

litellm = litellm_provider.litellm


def test_flash_lite_adapter_preserves_vision_schema_and_omits_sampling() -> None:
    importlib.reload(litellm_provider)
    model = "gemini/gemini-3.5-flash-lite"

    assert litellm.supports_vision(model=model)
    assert litellm.supports_response_schema(model=model)
    mapped = litellm.GoogleAIStudioGeminiConfig().map_openai_params(
        non_default_params={},
        optional_params={},
        model="gemini-3.5-flash-lite",
        drop_params=False,
    )
    assert not {"temperature", "top_p", "top_k"} & mapped.keys()


def test_recorded_provider_returns_response(project_root: Path) -> None:
    provider = RecordedProvider(project_root / "tests/fixtures/recorded-responses.jsonl")

    response = provider.generate("aihub-report-r01", [])

    assert response["answer"] == "71.6%"
    assert provider.evidence_kind == "test_only"


def test_litellm_provider_requests_strict_json_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_TASK_KEY", "test-key")
    captured: dict[str, object] = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            id="response-1",
            model="test/model",
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
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
                        )
                    )
                )
            ],
        )

    monkeypatch.setattr(
        "verifiable_ai_workflow.providers.litellm_provider.litellm.cost_per_token",
        lambda **kwargs: (0.01, 0.02),
    )
    monkeypatch.setattr(
        "verifiable_ai_workflow.providers.litellm_provider.litellm.completion",
        fake_completion,
    )
    provider = LiteLLMProvider(
        model="test/model",
        api_key_env="TEST_TASK_KEY",
        api_base=None,
        structured_output="json_schema",
        max_requests=1,
        requests_per_minute=1200,
        max_retries=0,
        retry_initial_seconds=1,
        max_cost_usd=0.1,
        max_input_tokens=100,
        max_output_tokens=500,
        max_wall_seconds=45,
    )

    provider.generate("sample-1", [{"role": "user", "content": "질문"}])

    assert captured["num_retries"] == 0
    assert captured["response_format"]["type"] == "json_schema"
    with pytest.raises(RuntimeError, match="상한 1건"):
        provider.generate("sample-2", [{"role": "user", "content": "질문"}])


def test_litellm_provider_accepts_judge_response_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Winner(BaseModel):
        winner: str

    monkeypatch.setenv("TEST_TASK_KEY", "test-key")
    captured: dict[str, object] = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            id="response-1",
            model="test/model",
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"winner":"a"}'))],
        )

    monkeypatch.setattr(
        "verifiable_ai_workflow.providers.litellm_provider.litellm.completion",
        fake_completion,
    )
    monkeypatch.setattr(
        "verifiable_ai_workflow.providers.litellm_provider.litellm.cost_per_token",
        lambda **kwargs: (0.0, 0.0),
    )
    provider = LiteLLMProvider(
        model="test/model",
        api_key_env="TEST_TASK_KEY",
        api_base=None,
        structured_output="json_schema",
        max_requests=1,
        requests_per_minute=1200,
        max_retries=0,
        retry_initial_seconds=1,
        max_cost_usd=0.1,
        max_input_tokens=100,
        max_output_tokens=50,
        max_wall_seconds=45,
    )

    provider.generate("pair-1", [{"role": "user", "content": "judge"}], response_schema=Winner)

    assert captured["response_format"]["json_schema"]["name"] == "Winner"


def test_resume_preserves_rate_interval_before_first_network_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_TASK_KEY", "test-key")
    resumed_at = datetime(2026, 7, 30, tzinfo=UTC)
    waits: list[float] = []
    network_calls = 0

    def fake_completion(**kwargs):
        nonlocal network_calls
        del kwargs
        network_calls += 1
        return SimpleNamespace(
            id="response-1",
            model="test/model",
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))],
        )

    monkeypatch.setattr(
        "verifiable_ai_workflow.providers.litellm_provider.litellm.completion",
        fake_completion,
    )
    provider = LiteLLMProvider(
        model="test/model",
        api_key_env="TEST_TASK_KEY",
        api_base=None,
        structured_output="prompt_only",
        max_requests=1,
        requests_per_minute=20,
        max_retries=0,
        retry_initial_seconds=1,
        max_cost_usd=0.1,
        max_input_tokens=100,
        max_output_tokens=50,
        max_wall_seconds=45,
        request_input_token_ceiling=100,
        request_output_token_ceiling=50,
        input_cost_per_token_usd=0.0,
        output_cost_per_token_usd=0.0,
        resume_last_attempt_started_at=resumed_at,
        sleep=waits.append,
        clock=lambda: 0.0,
        utc_now=lambda: resumed_at + timedelta(seconds=1),
    )

    provider.generate("sample-1", [{"role": "user", "content": "질문"}])

    assert network_calls == 1
    assert waits == [2.0]
    assert provider.budget.summary()["wall_seconds"] == 2.0


def test_litellm_provider_stops_before_over_budget_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_TASK_KEY", "test-key")
    monkeypatch.setattr(
        "verifiable_ai_workflow.providers.litellm_provider.litellm.cost_per_token",
        lambda **kwargs: (0.1, 0.1),
    )
    provider = LiteLLMProvider(
        model="test/model",
        api_key_env="TEST_TASK_KEY",
        api_base=None,
        structured_output="json_schema",
        max_requests=1,
        requests_per_minute=1200,
        max_retries=0,
        retry_initial_seconds=1,
        max_cost_usd=0.1,
        max_input_tokens=100,
        max_output_tokens=500,
        max_wall_seconds=45,
    )

    with pytest.raises(LiveBudgetExceeded, match="상한"):
        provider.generate("sample-1", [{"role": "user", "content": "질문"}])


@pytest.mark.parametrize(
    ("thinking_parameter", "expected_extra_body"),
    [
        ("thinking", {"thinking": {"type": "disabled"}}),
        ("chat_template", {"chat_template_kwargs": {"enable_thinking": False}}),
    ],
)
def test_nvidia_nim_uses_api_base_and_prompt_structured_output(
    monkeypatch: pytest.MonkeyPatch,
    thinking_parameter: str,
    expected_extra_body: dict,
) -> None:
    monkeypatch.setenv("NVIDIA_NIM_API_KEY", "test-key")
    captured: dict[str, object] = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            id="response-1",
            model="nvidia_nim/nvidia/example-vlm",
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
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
                        )
                    )
                )
            ],
        )

    monkeypatch.setattr(
        "verifiable_ai_workflow.providers.litellm_provider.litellm.completion",
        fake_completion,
    )
    provider = LiteLLMProvider(
        model="nvidia_nim/nvidia/example-vlm",
        expected_actual_model="nvidia/example-vlm",
        api_key_env="NVIDIA_NIM_API_KEY",
        api_base="https://integrate.api.nvidia.com/v1",
        structured_output="prompt_only",
        max_requests=1,
        requests_per_minute=1200,
        max_retries=0,
        retry_initial_seconds=1,
        max_cost_usd=0.1,
        max_input_tokens=300,
        max_output_tokens=150,
        max_wall_seconds=45,
        request_input_token_ceiling=100,
        request_output_token_ceiling=50,
        input_cost_per_token_usd=0.0,
        output_cost_per_token_usd=0.0,
        temperature=0.6,
        top_p=0.95,
        seed=0,
        thinking_mode="disabled",
        thinking_parameter=thinking_parameter,
    )

    provider.generate("sample-1", [{"role": "user", "content": "질문"}])

    assert captured["api_base"] == "https://integrate.api.nvidia.com/v1"
    assert captured["temperature"] == 0.6
    assert captured["top_p"] == 0.95
    assert captured["seed"] == 0
    assert provider.last_call["request_parameters"]["sampling_parameters"] == "explicit"
    assert captured["extra_body"] == expected_extra_body
    assert "response_format" not in captured
    assert provider.last_call["reported_actual_model"] == "nvidia_nim/nvidia/example-vlm"
    assert provider.last_call["actual_model"] == "nvidia/example-vlm"
    assert provider.last_call["raw_response"]["model"] == "nvidia_nim/nvidia/example-vlm"
    assert provider.last_call["request_parameters"]["thinking_mode"] == "disabled"
    assert provider.last_call["request_parameters"]["thinking_parameter"] == (
        thinking_parameter
    )


def test_live_provider_redacts_key_if_provider_echoes_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "task-key-never-store"
    monkeypatch.setenv("TEST_TASK_KEY", secret)
    monkeypatch.setattr(
        "verifiable_ai_workflow.providers.litellm_provider.litellm.completion",
        lambda **kwargs: SimpleNamespace(
            id="response-secret-echo",
            model="test/model",
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=f'{{"answer":"{secret}"}}',
                    )
                )
            ],
        ),
    )
    journal: list[dict] = []
    provider = LiteLLMProvider(
        model="test/model",
        expected_actual_model="test/model",
        api_key_env="TEST_TASK_KEY",
        api_base="https://example.invalid/v1",
        structured_output="prompt_only",
        max_requests=1,
        requests_per_minute=1200,
        max_retries=0,
        retry_initial_seconds=1,
        max_cost_usd=0.1,
        max_input_tokens=100,
        max_output_tokens=50,
        max_wall_seconds=45,
        request_input_token_ceiling=100,
        request_output_token_ceiling=50,
        input_cost_per_token_usd=0.0,
        output_cost_per_token_usd=0.0,
        on_response_received=journal.append,
    )

    content = provider.generate("sample-1", [{"role": "user", "content": "질문"}])

    assert secret not in content
    assert secret not in json.dumps(journal)
    assert secret not in json.dumps(provider.last_call)
    assert "[REDACTED]" in content


@pytest.mark.parametrize("status_code", [429, 500])
def test_transient_error_waits_and_retries(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
) -> None:
    monkeypatch.setenv("TEST_TASK_KEY", "test-key")
    calls = 0
    waits: list[float] = []

    class TransientError(Exception):
        pass

    def fake_completion(**kwargs):
        nonlocal calls
        del kwargs
        calls += 1
        if calls == 1:
            error = TransientError(str(status_code))
            error.status_code = status_code
            raise error
        return SimpleNamespace(
            id="response-2",
            model="test/model",
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))],
        )

    monkeypatch.setattr(
        "verifiable_ai_workflow.providers.litellm_provider.litellm.completion",
        fake_completion,
    )
    provider = LiteLLMProvider(
        model="test/model",
        api_key_env="TEST_TASK_KEY",
        api_base=None,
        structured_output="prompt_only",
        max_requests=1,
        requests_per_minute=20,
        max_retries=2,
        retry_initial_seconds=5,
        max_cost_usd=0.1,
        max_input_tokens=300,
        max_output_tokens=150,
        max_wall_seconds=45,
        request_input_token_ceiling=100,
        request_output_token_ceiling=50,
        input_cost_per_token_usd=0.0,
        output_cost_per_token_usd=0.0,
        sleep=waits.append,
        clock=lambda: 0.0,
    )

    provider.generate("sample-1", [{"role": "user", "content": "질문"}])

    assert calls == 2
    assert 5 in waits
    assert provider.last_call["retry_count"] == 1
    assert provider.budget.summary()["charged_input_tokens"] == 110
    assert provider.budget.summary()["charged_output_tokens"] == 55


def test_cumulative_cost_blocks_second_call_before_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_TASK_KEY", "test-key")
    calls = 0

    def fake_completion(**kwargs):
        nonlocal calls
        del kwargs
        calls += 1
        return SimpleNamespace(
            id=f"response-{calls}",
            model="test/model",
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50),
            choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))],
        )

    monkeypatch.setattr(
        "verifiable_ai_workflow.providers.litellm_provider.litellm.completion",
        fake_completion,
    )
    provider = LiteLLMProvider(
        model="test/model",
        api_key_env="TEST_TASK_KEY",
        api_base=None,
        structured_output="prompt_only",
        max_requests=2,
        max_attempts=2,
        requests_per_minute=1200,
        max_retries=0,
        retry_initial_seconds=1,
        max_cost_usd=0.2,
        max_input_tokens=500,
        max_output_tokens=250,
        max_wall_seconds=45,
        request_input_token_ceiling=100,
        request_output_token_ceiling=50,
        input_cost_per_token_usd=0.001,
        output_cost_per_token_usd=0.001,
    )

    provider.generate("sample-1", [{"role": "user", "content": "질문"}])
    with pytest.raises(LiveBudgetExceeded, match="누적 비용 예약"):
        provider.generate("sample-2", [{"role": "user", "content": "질문"}])

    assert calls == 1
    assert provider.request_count == 1
    assert provider.last_call["sample_id"] == "sample-2"
    assert provider.last_call["provider_status"] == "blocked"
    assert "test-key" not in json.dumps(provider.last_call)


@pytest.mark.parametrize(
    ("actual_model", "usage", "error_type"),
    [
        (None, SimpleNamespace(prompt_tokens=10, completion_tokens=5), "TelemetryIncomplete"),
        (
            "other/model",
            SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            "ActualModelMismatch",
        ),
        ("test/model", None, "TelemetryIncomplete"),
    ],
)
def test_missing_telemetry_or_model_drift_halts_later_calls(
    monkeypatch: pytest.MonkeyPatch,
    actual_model: str | None,
    usage: SimpleNamespace | None,
    error_type: str,
) -> None:
    monkeypatch.setenv("TEST_TASK_KEY", "test-key")
    calls = 0

    def fake_completion(**kwargs):
        nonlocal calls
        del kwargs
        calls += 1
        return SimpleNamespace(
            id="response-1",
            model=actual_model,
            usage=usage,
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"answer":"raw"}'))],
        )

    monkeypatch.setattr(
        "verifiable_ai_workflow.providers.litellm_provider.litellm.completion",
        fake_completion,
    )
    provider = LiteLLMProvider(
        model="test/model",
        expected_actual_model="test/model",
        api_key_env="TEST_TASK_KEY",
        api_base=None,
        structured_output="prompt_only",
        max_requests=2,
        requests_per_minute=1200,
        max_retries=0,
        retry_initial_seconds=1,
        max_cost_usd=0.1,
        max_input_tokens=200,
        max_output_tokens=100,
        max_wall_seconds=45,
        request_input_token_ceiling=100,
        request_output_token_ceiling=50,
        input_cost_per_token_usd=0.0,
        output_cost_per_token_usd=0.0,
    )

    with pytest.raises(RuntimeError):
        provider.generate("sample-1", [{"role": "user", "content": "질문"}])

    assert provider.last_call["provider_status"] == "provider_error"
    assert provider.last_call["error_type"] == error_type
    assert provider.last_call["raw_response"]["choices"][0]["content"] == '{"answer":"raw"}'
    with pytest.raises(RuntimeError, match="이전 응답 검증 실패"):
        provider.generate("sample-2", [{"role": "user", "content": "질문"}])
    assert calls == 1


@pytest.mark.parametrize("invalid_usage", [True, 1.9, "100", -1, object()])
def test_invalid_usage_is_journaled_before_fail_closed_budget_settlement(
    monkeypatch: pytest.MonkeyPatch,
    invalid_usage: object,
) -> None:
    monkeypatch.setenv("TEST_TASK_KEY", "test-key")
    calls = 0
    journal: list[dict] = []
    status_when_journaled: list[str] = []

    def fake_completion(**kwargs):
        nonlocal calls
        del kwargs
        calls += 1
        return SimpleNamespace(
            id="response-invalid-usage",
            model="test/model",
            usage=SimpleNamespace(
                prompt_tokens=invalid_usage,
                completion_tokens=5,
            ),
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"answer":"raw"}'))],
        )

    monkeypatch.setattr(
        "verifiable_ai_workflow.providers.litellm_provider.litellm.completion",
        fake_completion,
    )

    def preserve(record: dict) -> None:
        journal.append(record)
        status_when_journaled.append(provider.budget.state.attempts[0].status)

    provider = LiteLLMProvider(
        model="test/model",
        expected_actual_model="test/model",
        api_key_env="TEST_TASK_KEY",
        api_base=None,
        structured_output="prompt_only",
        max_requests=2,
        requests_per_minute=1200,
        max_retries=0,
        retry_initial_seconds=1,
        max_cost_usd=0.1,
        max_input_tokens=200,
        max_output_tokens=100,
        max_wall_seconds=45,
        request_input_token_ceiling=100,
        request_output_token_ceiling=50,
        input_cost_per_token_usd=0.0,
        output_cost_per_token_usd=0.0,
        on_response_received=preserve,
    )

    with pytest.raises(RuntimeError, match="telemetry 누락"):
        provider.generate("sample-1", [{"role": "user", "content": "질문"}])

    assert status_when_journaled == ["reserved"]
    assert journal[0]["raw_response"]["id"] == "response-invalid-usage"
    assert journal[0]["actual_model"] == "test/model"
    assert journal[0]["input_tokens"] is None
    assert journal[0]["output_tokens"] == 5
    assert provider.last_call["error_type"] == "TelemetryIncomplete"
    assert provider.budget.summary()["charged_input_tokens"] == 100
    assert provider.budget.summary()["charged_output_tokens"] == 5
    with pytest.raises(RuntimeError, match="이전 응답 검증 실패"):
        provider.generate("sample-2", [{"role": "user", "content": "질문"}])
    assert calls == 1


def test_failed_call_keeps_current_metadata_and_redacts_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_TASK_KEY", "test-key")
    calls = 0

    def fake_completion(**kwargs):
        nonlocal calls
        del kwargs
        calls += 1
        if calls == 2:
            raise RuntimeError("authorization test-key")
        return SimpleNamespace(
            id="response-1",
            model="test/model",
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))],
        )

    monkeypatch.setattr(
        "verifiable_ai_workflow.providers.litellm_provider.litellm.completion",
        fake_completion,
    )
    provider = LiteLLMProvider(
        model="test/model",
        api_key_env="TEST_TASK_KEY",
        api_base=None,
        structured_output="prompt_only",
        max_requests=2,
        requests_per_minute=1200,
        max_retries=0,
        retry_initial_seconds=1,
        max_cost_usd=0.1,
        max_input_tokens=200,
        max_output_tokens=100,
        max_wall_seconds=45,
        request_input_token_ceiling=100,
        request_output_token_ceiling=50,
        input_cost_per_token_usd=0.0,
        output_cost_per_token_usd=0.0,
    )

    provider.generate("sample-1", [{"role": "user", "content": "질문"}])
    with pytest.raises(RuntimeError, match=r"\[REDACTED\]"):
        provider.generate("sample-2", [{"role": "user", "content": "질문"}])

    assert provider.last_call["sample_id"] == "sample-2"
    assert provider.last_call["provider_status"] == "provider_error"
    assert provider.last_call["actual_model"] is None
    assert provider.last_call["attempt_trace"][-1]["status"] == "error"
    assert "test-key" not in json.dumps(provider.last_call)
