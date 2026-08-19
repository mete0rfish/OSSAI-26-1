"""주차별 실제 실습이 같은 LiteLLM 안전 설정을 재사용한다."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from .config.settings import LabSettings
from .live_execution import LiveBudget, LiveBudgetCaps
from .providers.litellm_provider import LiteLLMProvider


def summarize_call_failures(calls: list[dict], expected_actual_model: str) -> tuple[int, int]:
    terminal_drifts = sum(call.get("error_type") == "ActualModelMismatch" for call in calls)
    response_drifts = sum(
        not call.get("error_type")
        and call.get("provider_status") in {"success", "provider_response_received"}
        and call.get("actual_model") is not None
        and call.get("actual_model") != expected_actual_model
        for call in calls
    )
    return (
        sum(
            bool(call.get("error_type"))
            and call.get("error_type") != "ActualModelMismatch"
            for call in calls
        ),
        max(terminal_drifts, response_drifts),
    )


def build_course_provider(
    settings: LabSettings,
    caps: LiveBudgetCaps,
    *,
    structured_output: Literal["json_schema", "prompt_only"] | None = None,
    request_output_token_ceiling: int | None = None,
    on_response: Callable[[dict[str, Any]], None] | None = None,
    budget: LiveBudget | None = None,
) -> LiteLLMProvider:
    return LiteLLMProvider(
        model=settings.provider.model,
        expected_actual_model=settings.provider.expected_actual_model,
        api_key_env=settings.provider.api_key_env,
        api_base=settings.provider.api_base,
        structured_output=structured_output or settings.provider.structured_output,
        max_requests=caps.max_requests,
        max_attempts=caps.max_attempts,
        requests_per_minute=settings.limits.requests_per_minute,
        max_retries=settings.limits.max_retries,
        retry_initial_seconds=settings.limits.retry_initial_seconds,
        max_cost_usd=caps.max_cost_usd,
        max_input_tokens=caps.max_input_tokens,
        max_output_tokens=caps.max_output_tokens,
        max_wall_seconds=caps.max_wall_seconds,
        request_input_token_ceiling=settings.limits.request_input_token_ceiling,
        request_output_token_ceiling=(
            request_output_token_ceiling
            if request_output_token_ceiling is not None
            else settings.limits.request_output_token_ceiling
        ),
        request_timeout_seconds=settings.limits.request_timeout_seconds,
        input_cost_per_token_usd=settings.provider.input_cost_per_token_usd,
        output_cost_per_token_usd=settings.provider.output_cost_per_token_usd,
        temperature=settings.provider.temperature,
        top_p=settings.provider.top_p,
        seed=settings.provider.seed,
        sampling_parameters=settings.provider.sampling_parameters,
        thinking_mode=settings.provider.thinking_mode,
        thinking_parameter=settings.provider.thinking_parameter,
        max_images_per_prompt=settings.provider.max_images_per_prompt,
        budget=budget,
        on_response_received=on_response,
    )
