"""LiteLLM 전송 prefix와 provider model ID를 엄격하게 구분한다."""

from __future__ import annotations


def canonicalize_litellm_actual_model(
    reported_actual_model: str | None,
    *,
    requested_model: str,
    expected_actual_model: str,
) -> str | None:
    """요청에 사용한 transport prefix 한 개만 actual model에서 제거한다."""

    if reported_actual_model is None or reported_actual_model == expected_actual_model:
        return reported_actual_model
    transport_prefix, separator, _ = requested_model.partition("/")
    if separator and reported_actual_model == f"{transport_prefix}/{expected_actual_model}":
        return expected_actual_model
    return reported_actual_model
