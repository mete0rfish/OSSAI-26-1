"""준비된 문서 페이지를 LiteLLM으로 순차 호출한다."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any, Literal

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import litellm

from ..config import require_api_key
from ..schemas import StructuredAnswer


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
        input_cost_per_token_usd: float | None = None,
        output_cost_per_token_usd: float | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if (
            max_requests <= 0
            or requests_per_minute <= 0
            or max_retries < 0
            or retry_initial_seconds <= 0
            or max_cost_usd <= 0
            or max_input_tokens <= 0
            or max_output_tokens <= 0
            or max_wall_seconds <= 0
        ):
            raise ValueError("요청, 재시도, 비용, token과 시간 상한을 확인해 주세요")
        if (input_cost_per_token_usd is None) != (output_cost_per_token_usd is None):
            raise ValueError("입력·출력 token 비용은 함께 설정해야 합니다")
        self.model = model
        self.api_base = api_base
        self.structured_output = structured_output
        self._api_key = require_api_key(api_key_env)
        self.max_requests = max_requests
        self.requests_per_minute = requests_per_minute
        self.max_retries = max_retries
        self.retry_initial_seconds = retry_initial_seconds
        self.max_cost_usd = max_cost_usd
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.max_wall_seconds = max_wall_seconds
        self.input_cost_per_token_usd = input_cost_per_token_usd
        self.output_cost_per_token_usd = output_cost_per_token_usd
        self.request_count = 0
        self.attempt_count = 0
        self._minimum_interval = 60 / requests_per_minute
        self._last_attempt_started: float | None = None
        self._sleep = sleep
        self._clock = clock
        self.last_call: dict[str, Any] | None = None

    def generate(self, sample_id: str, messages: list[dict[str, Any]]) -> Any:
        if self.request_count >= self.max_requests:
            raise RuntimeError(f"실제 API task 상한 {self.max_requests}건을 모두 사용했습니다")

        if self.input_cost_per_token_usd is not None:
            input_cost = self.max_input_tokens * self.input_cost_per_token_usd
            output_cost = self.max_output_tokens * self.output_cost_per_token_usd
        else:
            input_cost, output_cost = litellm.cost_per_token(
                model=self.model,
                prompt_tokens=self.max_input_tokens,
                completion_tokens=self.max_output_tokens,
            )
        estimated_max_cost = input_cost + output_cost
        if estimated_max_cost > self.max_cost_usd:
            raise ValueError(
                f"예상 최대 비용 ${estimated_max_cost:.4f}가 "
                f"상한 ${self.max_cost_usd:.4f}를 넘습니다"
            )

        self.request_count += 1
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "api_key": self._api_key,
            "temperature": 0,
            "max_tokens": self.max_output_tokens,
            "timeout": self.max_wall_seconds,
            "num_retries": 0,
        }
        if self.api_base:
            request["api_base"] = self.api_base
        if self.structured_output == "json_schema":
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_answer",
                    "strict": True,
                    "schema": StructuredAnswer.model_json_schema(),
                },
            }
        response = None
        started = time.perf_counter()
        retry_count = 0
        for attempt in range(self.max_retries + 1):
            self._wait_for_rate_limit()
            self.attempt_count += 1
            try:
                response = litellm.completion(**request)
                retry_count = attempt
                break
            except Exception as exc:
                if not self._is_rate_limit_error(exc) or attempt == self.max_retries:
                    raise
                retry_count = attempt + 1
                self._sleep(self.retry_initial_seconds * (2**attempt))
        if response is None:
            raise RuntimeError("모델 응답이 없습니다")

        latency_ms = (time.perf_counter() - started) * 1000
        usage = getattr(response, "usage", None)
        hidden = getattr(response, "_hidden_params", {}) or {}
        response_headers = hidden.get("additional_headers") or hidden.get("headers") or {}
        rate_limit_headers = {
            str(name): str(value)
            for name, value in response_headers.items()
            if "ratelimit" in str(name).casefold() or str(name).casefold() == "retry-after"
        }
        self.last_call = {
            "sample_id": sample_id,
            "requested_model": self.model,
            "actual_model": getattr(response, "model", None),
            "response_id": getattr(response, "id", None),
            "latency_ms": latency_ms,
            "input_tokens": getattr(usage, "prompt_tokens", None),
            "output_tokens": getattr(usage, "completion_tokens", None),
            "estimated_max_cost_usd": estimated_max_cost,
            "retry_count": retry_count,
            "request_number": self.request_count,
            "attempt_number": self.attempt_count,
            "rate_limit_headers": rate_limit_headers,
        }
        return response.choices[0].message.content

    def _wait_for_rate_limit(self) -> None:
        if self._last_attempt_started is not None:
            elapsed = self._clock() - self._last_attempt_started
            remaining = self._minimum_interval - elapsed
            if remaining > 0:
                self._sleep(remaining)
        self._last_attempt_started = self._clock()

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        return getattr(exc, "status_code", None) == 429 or "429" in str(exc)
