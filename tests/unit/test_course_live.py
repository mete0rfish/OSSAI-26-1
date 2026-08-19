from verifiable_ai_workflow.config.settings import load_settings
from verifiable_ai_workflow.course_live import build_course_provider, summarize_call_failures
from verifiable_ai_workflow.live_execution import LiveBudgetCaps


def test_call_failures_keep_model_drift_separate_from_provider_errors() -> None:
    provider_errors, model_drifts = summarize_call_failures(
        [
            {"error_type": "APIConnectionError", "actual_model": None},
            {"error_type": "ActualModelMismatch", "actual_model": "other/model"},
            {
                "error_type": None,
                "provider_status": "provider_response_received",
                "actual_model": "other/model",
            },
        ],
        "expected/model",
    )

    assert provider_errors == 1
    assert model_drifts == 1


def test_course_provider_reuses_caps_and_model(monkeypatch, project_root) -> None:
    monkeypatch.setenv("NVIDIA_NIM_API_KEY", "test-key")
    settings = load_settings(project_root / "configs/nvidia-nim-gemma4.yaml")
    caps = LiveBudgetCaps(
        max_requests=2,
        max_attempts=2,
        max_input_tokens=40000,
        max_output_tokens=1000,
        max_cost_usd=0.01,
        max_wall_seconds=240,
    )

    provider = build_course_provider(settings, caps)

    assert provider.model == "nvidia_nim/google/gemma-4-31b-it"
    assert provider.budget.state.caps == caps
    assert provider.structured_output == settings.provider.structured_output == "json_schema"
    assert build_course_provider(
        settings, caps, structured_output="json_schema"
    ).structured_output == "json_schema"
    assert build_course_provider(
        settings, caps, request_output_token_ceiling=800
    ).request_output_token_ceiling == 800
    assert provider.sampling_parameters == settings.provider.sampling_parameters
    assert settings.provider.billing_basis == "developer_program_free_endpoint"
    assert settings.provider.input_cost_per_token_usd == 0
    assert settings.provider.output_cost_per_token_usd == 0
