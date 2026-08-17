"""준비된 문서 페이지를 LiteLLM으로 순차 호출한다."""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Literal

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import litellm
from pydantic import BaseModel

from ..config import require_api_key
from ..live_execution import LiveBudget, LiveBudgetCaps, LiveBudgetExceeded
from ..model_identity import canonicalize_litellm_actual_model
from ..schemas import StructuredAnswer

_GEMINI_FLASH_LITE = "gemini/gemini-3.5-flash-lite"

# LiteLLM 1.93 predates this stable model. Register only the capabilities the
# course uses so image + JSON Schema requests do not silently fall back to a
# prompt-only schema.
litellm.register_model(
    {
        _GEMINI_FLASH_LITE: {
            "litellm_provider": "gemini",
            "mode": "chat",
            "max_input_tokens": 1_048_576,
            "max_output_tokens": 65_536,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "cache_creation_input_token_cost": 0.0,
            "cache_read_input_token_cost": 0.0,
            "supports_response_schema": True,
            "supports_system_messages": True,
            "supports_vision": True,
        }
    }
)

_original_gemini_param_mapper = litellm.GoogleAIStudioGeminiConfig.map_openai_params
_original_gemini_param_mapper = getattr(
    _original_gemini_param_mapper,
    "_openup_original",
    _original_gemini_param_mapper,
)


def _map_gemini_params_without_deprecated_sampling(
    self: Any,
    non_default_params: dict,
    optional_params: dict,
    model: str,
    drop_params: bool,
) -> dict:
    mapped = _original_gemini_param_mapper(
        self,
        non_default_params,
        optional_params,
        model,
        drop_params,
    )
    if model == "gemini-3.5-flash-lite":
        for name in ("temperature", "top_p", "top_k"):
            mapped.pop(name, None)
    return mapped


# ponytail: remove this narrow compatibility patch when the pinned LiteLLM
# release both maps Flash-Lite and stops injecting deprecated sampling fields.
_map_gemini_params_without_deprecated_sampling._openup_original = (  # type: ignore[attr-defined]
    _original_gemini_param_mapper
)
litellm.GoogleAIStudioGeminiConfig.map_openai_params = (  # type: ignore[method-assign]
    _map_gemini_params_without_deprecated_sampling
)


class LiteLLMProvider:
    evidence_kind: Literal["live_quality"] = "live_quality"

    def __init__(
        self,
        *,
        model: str,
        api_key_env: str,
        api_base: str | None,
        structured_output: Literal["json_schema", "prompt_only"],
        max_requests: int,
        requests_per_minute: int,
        max_retries: int,
        retry_initial_seconds: float,
        max_cost_usd: float,
        max_input_tokens: int,
        max_output_tokens: int,
        max_wall_seconds: float,
        expected_actual_model: str | None = None,
        max_attempts: int | None = None,
        request_input_token_ceiling: int | None = None,
        request_output_token_ceiling: int | None = None,
        request_timeout_seconds: float | None = None,
        input_cost_per_token_usd: float | None = None,
        output_cost_per_token_usd: float | None = None,
        temperature: float = 0.0,
        top_p: float | None = None,
        seed: int | None = None,
        sampling_parameters: Literal["explicit", "omit"] = "explicit",
        thinking_mode: Literal["default", "disabled"] = "default",
        thinking_parameter: Literal["thinking", "chat_template"] = "thinking",
        max_images_per_prompt: int | None = None,
        budget: LiveBudget | None = None,
        resume_last_attempt_started_at: datetime | None = None,
        on_response_received: Callable[[dict[str, Any]], None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        utc_now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        resolved_max_attempts = max_attempts or max_requests * (max_retries + 1)
        resolved_input_ceiling = request_input_token_ceiling or max_input_tokens
        resolved_output_ceiling = request_output_token_ceiling or max_output_tokens
        resolved_request_timeout = request_timeout_seconds or max_wall_seconds
        if (
            max_requests <= 0
            or resolved_max_attempts <= 0
            or requests_per_minute <= 0
            or max_retries < 0
            or retry_initial_seconds <= 0
            or max_cost_usd <= 0
            or max_input_tokens <= 0
            or max_output_tokens <= 0
            or max_wall_seconds <= 0
            or resolved_input_ceiling <= 0
            or resolved_output_ceiling <= 0
            or resolved_request_timeout <= 0
            or (max_images_per_prompt is not None and max_images_per_prompt <= 0)
        ):
            raise ValueError("요청, 재시도, 비용, token과 시간 상한을 확인해 주세요")
        if resolved_input_ceiling > max_input_tokens:
            raise ValueError("request input token ceiling이 전체 input token 상한보다 큽니다")
        if resolved_output_ceiling > max_output_tokens:
            raise ValueError("request output token ceiling이 전체 output token 상한보다 큽니다")
        if resolved_request_timeout > max_wall_seconds:
            raise ValueError("request timeout이 전체 wall time 상한보다 큽니다")
        if (input_cost_per_token_usd is None) != (output_cost_per_token_usd is None):
            raise ValueError("입력·출력 token 비용은 함께 설정해야 합니다")
        if not math.isfinite(temperature) or not 0 <= temperature <= 2:
            raise ValueError("temperature는 0 이상 2 이하의 유한수여야 합니다")
        if top_p is not None and (not math.isfinite(top_p) or not 0 < top_p <= 1):
            raise ValueError("top_p는 0 초과 1 이하의 유한수여야 합니다")
        if seed is not None and seed < 0:
            raise ValueError("seed는 0 이상이어야 합니다")
        if sampling_parameters not in {"explicit", "omit"}:
            raise ValueError("sampling_parameters는 explicit 또는 omit이어야 합니다")
        if thinking_mode not in {"default", "disabled"}:
            raise ValueError("thinking_mode는 default 또는 disabled여야 합니다")
        if thinking_parameter not in {"thinking", "chat_template"}:
            raise ValueError("thinking_parameter는 thinking 또는 chat_template이어야 합니다")
        caps = LiveBudgetCaps(
            max_requests=max_requests,
            max_attempts=resolved_max_attempts,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            max_cost_usd=max_cost_usd,
            max_wall_seconds=max_wall_seconds,
        )
        if budget is not None and budget.state.caps != caps:
            raise ValueError("provider 인자와 영속 live budget cap이 다릅니다")
        if resume_last_attempt_started_at is not None and (
            resume_last_attempt_started_at.tzinfo is None
            or resume_last_attempt_started_at.utcoffset() is None
        ):
            raise ValueError("resume 마지막 attempt 시각에는 timezone이 필요합니다")
        self.model = model
        self.expected_actual_model = expected_actual_model or model
        self.api_base = api_base
        self.structured_output = structured_output
        self._api_key = require_api_key(api_key_env)
        self.max_requests = max_requests
        self.max_attempts = resolved_max_attempts
        self.requests_per_minute = requests_per_minute
        self.max_retries = max_retries
        self.retry_initial_seconds = retry_initial_seconds
        self.max_cost_usd = max_cost_usd
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.max_wall_seconds = max_wall_seconds
        self.request_input_token_ceiling = resolved_input_ceiling
        self.request_output_token_ceiling = resolved_output_ceiling
        self.request_timeout_seconds = resolved_request_timeout
        self.input_cost_per_token_usd = input_cost_per_token_usd
        self.output_cost_per_token_usd = output_cost_per_token_usd
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.sampling_parameters = sampling_parameters
        self.thinking_mode = thinking_mode
        self.thinking_parameter = thinking_parameter
        self.max_images_per_prompt = max_images_per_prompt
        self.budget = budget or LiveBudget(caps)
        self._on_response_received = on_response_received
        self._minimum_interval = 60 / requests_per_minute
        self._last_attempt_started: float | None = None
        self._resume_last_attempt_started_at = resume_last_attempt_started_at
        self._sleep = sleep
        self._clock = clock
        self._utc_now = utc_now
        self.last_call: dict[str, Any] | None = None
        self._halted_reason: str | None = None

    def set_on_response_received(
        self,
        callback: Callable[[dict[str, Any]], None] | None,
    ) -> None:
        self._on_response_received = callback

    @property
    def request_count(self) -> int:
        return self.budget.request_count

    @property
    def attempt_count(self) -> int:
        return self.budget.attempt_count

    def generate(
        self,
        sample_id: str,
        messages: list[dict[str, Any]],
        *,
        response_schema: type[BaseModel] | None = StructuredAnswer,
    ) -> Any:
        if self._halted_reason is not None:
            blocked = RuntimeError(
                f"이 provider run은 이전 응답 검증 실패로 중단됐습니다: {self._halted_reason}"
            )
            self.last_call = self._new_call_record(sample_id, 0.0)
            self._record_blocked(blocked)
            raise blocked
        estimated_max_cost = self._cost_for_tokens(
            self.request_input_token_ceiling,
            self.request_output_token_ceiling,
        )
        self.last_call = self._new_call_record(sample_id, estimated_max_cost)
        request_base: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "api_key": self._api_key,
            "max_tokens": self.request_output_token_ceiling,
            "num_retries": 0,
        }
        if self.sampling_parameters == "explicit":
            request_base["temperature"] = self.temperature
            if self.top_p is not None:
                request_base["top_p"] = self.top_p
            if self.seed is not None:
                request_base["seed"] = self.seed
        if self.thinking_mode == "disabled":
            request_base["extra_body"] = (
                {"thinking": {"type": "disabled"}}
                if self.thinking_parameter == "thinking"
                else {"chat_template_kwargs": {"enable_thinking": False}}
            )
        if self.api_base:
            request_base["api_base"] = self.api_base
        if self.structured_output == "json_schema" and response_schema is not None:
            request_base["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_schema.__name__,
                    "strict": True,
                    "schema": response_schema.model_json_schema(),
                },
            }
        response = None
        retry_count = 0
        request_number: int | None = None
        for attempt in range(self.max_retries + 1):
            try:
                self._wait_for_rate_limit()
                reservation = self.budget.reserve_attempt(
                    sample_id=sample_id,
                    request_number=request_number,
                    reserved_input_tokens=self.request_input_token_ceiling,
                    reserved_output_tokens=self.request_output_token_ceiling,
                    reserved_cost_usd=estimated_max_cost,
                )
            except Exception as exc:
                self._record_blocked(exc)
                raise
            request_number = reservation.request_number
            request = {
                **request_base,
                "timeout": min(
                    self.request_timeout_seconds,
                    self.budget.remaining_wall_seconds,
                ),
            }
            started = self._clock()
            try:
                response = litellm.completion(**request)
            except Exception as exc:
                elapsed = max(0.0, self._clock() - started)
                wall_violations = self.budget.record_wall_after_call(elapsed)
                error_message = self._safe_error_message(exc)
                self.budget.fail_attempt(
                    reservation.attempt_number,
                    error_type=type(exc).__name__,
                    error_message=error_message,
                )
                self._append_attempt_trace(
                    reservation.attempt_number,
                    latency_ms=elapsed * 1000,
                    status="error",
                    error_type=type(exc).__name__,
                    error_message=error_message,
                )
                retry_count = attempt
                self._record_provider_error(
                    exc,
                    retry_count=retry_count,
                    budget_violations=wall_violations,
                )
                if not self._is_transient_error(exc) or attempt == self.max_retries:
                    raise RuntimeError(f"{type(exc).__name__}: {error_message}") from exc
                retry_count = attempt + 1
                try:
                    self._sleep_with_budget(self.retry_initial_seconds * (2**attempt))
                except Exception as budget_exc:
                    self._record_provider_error(
                        budget_exc,
                        retry_count=retry_count,
                        budget_violations=["retry_wait_budget_exceeded"],
                    )
                    raise
                continue

            elapsed = max(0.0, self._clock() - started)
            raw_response = self._response_snapshot(response)
            usage = getattr(response, "usage", None)
            input_tokens = self._optional_nonnegative_int(getattr(usage, "prompt_tokens", None))
            output_tokens = self._optional_nonnegative_int(
                getattr(usage, "completion_tokens", None)
            )
            reported_actual_model = getattr(response, "model", None)
            actual_model = canonicalize_litellm_actual_model(
                reported_actual_model,
                requested_model=self.model,
                expected_actual_model=self.expected_actual_model,
            )
            actual_cost = (
                self._cost_for_tokens(input_tokens, output_tokens)
                if input_tokens is not None and output_tokens is not None
                else None
            )
            self.last_call.update(
                {
                    "provider_status": "provider_response_received",
                    "raw_response": raw_response,
                    "reported_actual_model": reported_actual_model,
                    "actual_model": actual_model,
                    "actual_model_matches_expected": (actual_model == self.expected_actual_model),
                    "response_id": getattr(response, "id", None),
                    "latency_ms": elapsed * 1000,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "actual_cost_usd": actual_cost,
                    "retry_count": attempt,
                    "request_number": reservation.request_number,
                    "attempt_number": reservation.attempt_number,
                    "response_received_at": datetime.now(UTC).isoformat(),
                    "budget": self.budget.summary(),
                }
            )
            if self._on_response_received is not None:
                try:
                    self._on_response_received(dict(self.last_call))
                except Exception as exc:
                    self._halted_reason = "response_evidence_persistence_failed"
                    message = self._safe_error_message(exc)
                    self.last_call.update(
                        {
                            "provider_status": "provider_error",
                            "error_type": "ResponseEvidencePersistenceError",
                            "error_message": message,
                            "budget_violations": ["response_evidence_persistence_failed"],
                        }
                    )
                    raise RuntimeError(
                        "provider 응답 원문을 영속 저장하지 못해 실행을 중단했습니다"
                    ) from exc
            wall_violations = self.budget.record_wall_after_call(elapsed)
            token_cost_violations = self.budget.complete_attempt(
                reservation.attempt_number,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                actual_cost_usd=actual_cost,
            )
            self._append_attempt_trace(
                reservation.attempt_number,
                latency_ms=elapsed * 1000,
                status="success",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                actual_cost_usd=actual_cost,
            )
            violations = [*wall_violations, *token_cost_violations]
            self.last_call.update(
                {
                    "raw_response": raw_response,
                    "reported_actual_model": reported_actual_model,
                    "actual_model": actual_model,
                    "actual_model_matches_expected": (actual_model == self.expected_actual_model),
                    "response_id": getattr(response, "id", None),
                    "latency_ms": elapsed * 1000,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "actual_cost_usd": actual_cost,
                    "retry_count": attempt,
                    "request_number": reservation.request_number,
                    "attempt_number": reservation.attempt_number,
                    "budget": self.budget.summary(),
                }
            )
            telemetry_missing = []
            if actual_model is None:
                telemetry_missing.append("actual_model")
            if input_tokens is None:
                telemetry_missing.append("input_tokens")
            if output_tokens is None:
                telemetry_missing.append("output_tokens")
            if telemetry_missing:
                self._halted_reason = "telemetry_incomplete"
                message = "provider 응답 telemetry 누락: " + ", ".join(telemetry_missing)
                self.last_call.update(
                    {
                        "provider_status": "provider_error",
                        "error_type": "TelemetryIncomplete",
                        "error_message": message,
                        "budget_violations": ["telemetry_incomplete"],
                    }
                )
                raise RuntimeError(message)
            if actual_model != self.expected_actual_model:
                self._halted_reason = "actual_model_mismatch"
                message = (
                    "provider actual model 불일치: "
                    f"{reported_actual_model!r} != {self.expected_actual_model!r}"
                )
                self.last_call.update(
                    {
                        "provider_status": "provider_error",
                        "error_type": "ActualModelMismatch",
                        "error_message": message,
                        "budget_violations": ["actual_model_mismatch"],
                    }
                )
                raise RuntimeError(message)
            if violations:
                self._halted_reason = "budget_violation"
                exceeded = LiveBudgetExceeded(
                    "API 응답이 예약된 누적 budget을 넘었습니다: "
                    + ", ".join(sorted(set(violations)))
                )
                self._record_provider_error(
                    exceeded,
                    retry_count=attempt,
                    budget_violations=violations,
                )
                raise exceeded
            retry_count = attempt
            break
        if response is None:
            raise RuntimeError("모델 응답이 없습니다")

        usage = getattr(response, "usage", None)
        hidden = getattr(response, "_hidden_params", {}) or {}
        response_headers = hidden.get("additional_headers") or hidden.get("headers") or {}
        rate_limit_headers = {
            str(name): str(value)
            for name, value in response_headers.items()
            if "ratelimit" in str(name).casefold() or str(name).casefold() == "retry-after"
        }
        reported_actual_model = getattr(response, "model", None)
        actual_model = canonicalize_litellm_actual_model(
            reported_actual_model,
            requested_model=self.model,
            expected_actual_model=self.expected_actual_model,
        )
        latest_attempt = self.budget.state.attempts[-1]
        self.last_call.update(
            {
                "provider_status": "success",
                "reported_actual_model": reported_actual_model,
                "actual_model": actual_model,
                "actual_model_matches_expected": actual_model == self.expected_actual_model,
                "response_id": getattr(response, "id", None),
                "latency_ms": self.last_call["attempt_trace"][-1]["latency_ms"],
                "input_tokens": getattr(usage, "prompt_tokens", None),
                "output_tokens": getattr(usage, "completion_tokens", None),
                "actual_cost_usd": latest_attempt.actual_cost_usd,
                "error_type": None,
                "error_message": None,
                "budget_violations": [],
                "budget": self.budget.summary(),
                "retry_count": retry_count,
                "request_number": latest_attempt.request_number,
                "attempt_number": latest_attempt.attempt_number,
                "rate_limit_headers": rate_limit_headers,
            }
        )
        return self._redact_secret_value(response.choices[0].message.content)

    def _new_call_record(
        self,
        sample_id: str,
        estimated_max_cost: float,
    ) -> dict[str, Any]:
        return {
            "sample_id": sample_id,
            "requested_model": self.model,
            "expected_actual_model": self.expected_actual_model,
            "reported_actual_model": None,
            "actual_model": None,
            "actual_model_matches_expected": False,
            "response_id": None,
            "provider_status": "blocked",
            "latency_ms": 0.0,
            "input_tokens": None,
            "output_tokens": None,
            "actual_cost_usd": None,
            "raw_response": None,
            "estimated_max_cost_usd": estimated_max_cost,
            "retry_count": 0,
            "request_number": None,
            "attempt_number": None,
            "rate_limit_headers": {},
            "error_type": None,
            "error_message": None,
            "budget_violations": [],
            "attempt_trace": [],
            "budget": self.budget.summary(),
            "response_received_at": None,
            "request_parameters": {
                "sampling_parameters": self.sampling_parameters,
                "temperature": (
                    self.temperature if self.sampling_parameters == "explicit" else None
                ),
                "top_p": self.top_p if self.sampling_parameters == "explicit" else None,
                "seed": self.seed if self.sampling_parameters == "explicit" else None,
                "thinking_mode": self.thinking_mode,
                "thinking_parameter": self.thinking_parameter,
                "max_images_per_prompt": self.max_images_per_prompt,
            },
        }

    def _append_attempt_trace(
        self,
        attempt_number: int,
        *,
        latency_ms: float,
        status: Literal["success", "error"],
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        actual_cost_usd: float | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> None:
        attempt = self.budget.state.attempts[attempt_number - 1]
        self.last_call["attempt_trace"].append(
            {
                "attempt_number": attempt.attempt_number,
                "request_number": attempt.request_number,
                "status": status,
                "started_at": attempt.started_at.isoformat(),
                "completed_at": (
                    attempt.completed_at.isoformat() if attempt.completed_at else None
                ),
                "latency_ms": latency_ms,
                "reserved_input_tokens": attempt.reserved_input_tokens,
                "reserved_output_tokens": attempt.reserved_output_tokens,
                "reserved_cost_usd": attempt.reserved_cost_usd,
                "actual_input_tokens": input_tokens,
                "actual_output_tokens": output_tokens,
                "actual_cost_usd": actual_cost_usd,
                "error_type": error_type,
                "error_message": error_message,
            }
        )

    def _record_blocked(self, exc: Exception) -> None:
        self.last_call.update(
            {
                "provider_status": "blocked",
                "error_type": type(exc).__name__,
                "error_message": self._safe_error_message(exc),
                "budget": self.budget.summary(),
            }
        )

    def _record_provider_error(
        self,
        exc: Exception,
        *,
        retry_count: int,
        budget_violations: list[str],
    ) -> None:
        latest = self.budget.state.attempts[-1] if self.budget.state.attempts else None
        self.last_call.update(
            {
                "provider_status": "provider_error",
                "actual_model": None,
                "actual_model_matches_expected": False,
                "retry_count": retry_count,
                "request_number": latest.request_number if latest else None,
                "attempt_number": latest.attempt_number if latest else None,
                "error_type": type(exc).__name__,
                "error_message": self._safe_error_message(exc),
                "budget_violations": sorted(set(budget_violations)),
                "budget": self.budget.summary(),
            }
        )

    def _cost_for_tokens(self, input_tokens: int, output_tokens: int) -> float:
        if self.input_cost_per_token_usd is not None:
            return (
                input_tokens * self.input_cost_per_token_usd
                + output_tokens * self.output_cost_per_token_usd
            )
        input_cost, output_cost = litellm.cost_per_token(
            model=self.model,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
        )
        return float(input_cost + output_cost)

    @staticmethod
    def _optional_nonnegative_int(value: Any) -> int | None:
        if value is None:
            return None
        if type(value) is not int or value < 0:
            return None
        return value

    def _safe_error_message(self, exc: Exception) -> str:
        message = str(exc)
        return message.replace(self._api_key, "[REDACTED]") if self._api_key else message

    def _response_snapshot(self, response: Any) -> Any:
        if hasattr(response, "model_dump"):
            snapshot = response.model_dump(mode="json")
        else:
            usage = getattr(response, "usage", None)
            choices = []
            for choice in getattr(response, "choices", []) or []:
                message = getattr(choice, "message", None)
                choices.append({"content": getattr(message, "content", None)})
            snapshot = {
                "id": getattr(response, "id", None),
                "model": getattr(response, "model", None),
                "usage": {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                },
                "choices": choices,
            }
        return self._redact_secret_value(snapshot)

    def _redact_secret_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return value.replace(self._api_key, "[REDACTED]")
        if isinstance(value, dict):
            return {
                self._redact_secret_value(key): self._redact_secret_value(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self._redact_secret_value(item) for item in value]
        if isinstance(value, tuple):
            return [self._redact_secret_value(item) for item in value]
        return value

    def _sleep_with_budget(self, seconds: float) -> None:
        self.budget.consume_wall(seconds)
        self._sleep(seconds)

    def _wait_for_rate_limit(self) -> None:
        if self._resume_last_attempt_started_at is not None:
            elapsed = max(
                0.0,
                (self._utc_now() - self._resume_last_attempt_started_at).total_seconds(),
            )
            remaining = self._minimum_interval - elapsed
            self._resume_last_attempt_started_at = None
            if remaining > 0:
                self._sleep_with_budget(remaining)
        elif self._last_attempt_started is not None:
            elapsed = self._clock() - self._last_attempt_started
            remaining = self._minimum_interval - elapsed
            if remaining > 0:
                self._sleep_with_budget(remaining)
        self._last_attempt_started = self._clock()

    @staticmethod
    def _is_transient_error(exc: Exception) -> bool:
        status = getattr(exc, "status_code", None)
        return status == 429 or (isinstance(status, int) and 500 <= status < 600)
