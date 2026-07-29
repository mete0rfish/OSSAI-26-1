import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from verifiable_ai_workflow.providers.litellm_provider import LiteLLMProvider
from verifiable_ai_workflow.providers.recorded import RecordedProvider


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
            ]
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

    with pytest.raises(ValueError, match="상한"):
        provider.generate("sample-1", [{"role": "user", "content": "질문"}])


def test_nvidia_nim_uses_api_base_and_prompt_structured_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NVIDIA_NIM_API_KEY", "test-key")
    captured: dict[str, object] = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            id="response-1",
            model="nvidia/example-vlm",
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
        api_key_env="NVIDIA_NIM_API_KEY",
        api_base="https://integrate.api.nvidia.com/v1",
        structured_output="prompt_only",
        max_requests=1,
        requests_per_minute=1200,
        max_retries=0,
        retry_initial_seconds=1,
        max_cost_usd=0.1,
        max_input_tokens=100,
        max_output_tokens=50,
        max_wall_seconds=45,
        input_cost_per_token_usd=0.0,
        output_cost_per_token_usd=0.0,
    )

    provider.generate("sample-1", [{"role": "user", "content": "질문"}])

    assert captured["api_base"] == "https://integrate.api.nvidia.com/v1"
    assert "response_format" not in captured
    assert provider.last_call["actual_model"] == "nvidia/example-vlm"


def test_rate_limit_error_waits_and_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_TASK_KEY", "test-key")
    calls = 0
    waits: list[float] = []

    class RateLimited(Exception):
        status_code = 429

    def fake_completion(**kwargs):
        nonlocal calls
        del kwargs
        calls += 1
        if calls == 1:
            raise RateLimited("429")
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
        max_input_tokens=100,
        max_output_tokens=50,
        max_wall_seconds=45,
        input_cost_per_token_usd=0.0,
        output_cost_per_token_usd=0.0,
        sleep=waits.append,
        clock=lambda: 0.0,
    )

    provider.generate("sample-1", [{"role": "user", "content": "질문"}])

    assert calls == 2
    assert 5 in waits
    assert provider.last_call["retry_count"] == 1
