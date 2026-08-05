"""실제 API 실행의 누적 예산과 재개 가능한 상태를 보존한다."""

from __future__ import annotations

import fcntl
import json
import math
import os
import secrets
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


class LiveExecutionError(RuntimeError):
    """실제 호출을 시작하기 전에 실행을 중단해야 하는 조건."""


class LiveBudgetExceeded(LiveExecutionError):
    """누적 예산을 넘기기 전에 호출을 차단하거나 사후 위반을 기록한다."""


def require_canonical_project_file(
    project_root: str | Path,
    supplied_path: str | Path,
    canonical_relative_path: str | Path,
) -> Path:
    """외부 설정·symlink로 승인된 API 목적지를 바꾸지 못하게 한다."""

    root = Path(project_root).resolve()
    canonical = root / canonical_relative_path
    if canonical.resolve() != canonical:
        raise LiveExecutionError(
            f"canonical 파일은 저장소 밖을 가리키는 symlink일 수 없습니다: "
            f"{canonical_relative_path}"
        )
    supplied = Path(supplied_path)
    if not supplied.is_absolute():
        supplied = root / supplied
    if supplied.resolve() != canonical:
        raise LiveExecutionError(
            f"승인된 canonical 파일만 사용할 수 있습니다: {canonical_relative_path}"
        )
    if not canonical.is_file():
        raise LiveExecutionError(f"canonical 파일을 찾을 수 없습니다: {canonical_relative_path}")
    return canonical


class LiveExecutionModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LiveBudgetCaps(LiveExecutionModel):
    max_requests: int = Field(gt=0)
    max_attempts: int = Field(gt=0)
    max_input_tokens: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)
    max_cost_usd: float = Field(gt=0, allow_inf_nan=False)
    max_wall_seconds: float = Field(gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def attempts_cover_requests(self) -> LiveBudgetCaps:
        if self.max_attempts < self.max_requests:
            raise ValueError("attempt 상한은 request 상한보다 작을 수 없습니다")
        return self


class LiveAttempt(LiveExecutionModel):
    attempt_number: int = Field(gt=0)
    request_number: int = Field(gt=0)
    sample_id: str = Field(min_length=1)
    started_at: datetime
    completed_at: datetime | None = None
    status: Literal["reserved", "success", "error", "interrupted"]
    reserved_input_tokens: int = Field(gt=0)
    reserved_output_tokens: int = Field(gt=0)
    reserved_cost_usd: float = Field(ge=0)
    actual_input_tokens: int | None = Field(default=None, ge=0)
    actual_output_tokens: int | None = Field(default=None, ge=0)
    actual_cost_usd: float | None = Field(default=None, ge=0)
    error_type: str | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def timestamps_and_error_match_status(self) -> LiveAttempt:
        if self.started_at.tzinfo is None or self.started_at.utcoffset() is None:
            raise ValueError("attempt started_at에는 timezone이 필요합니다")
        if self.completed_at is not None and (
            self.completed_at.tzinfo is None or self.completed_at.utcoffset() is None
        ):
            raise ValueError("attempt completed_at에는 timezone이 필요합니다")
        if self.status == "reserved":
            if self.completed_at is not None:
                raise ValueError("reserved attempt에는 completed_at을 넣을 수 없습니다")
        elif self.completed_at is None:
            raise ValueError("종료된 attempt에는 completed_at이 필요합니다")
        has_error_type = bool(self.error_type)
        has_error_message = bool(self.error_message)
        if has_error_type != has_error_message:
            raise ValueError("attempt error type과 message는 함께 기록해야 합니다")
        if self.status in {"error", "interrupted"} and not has_error_type:
            raise ValueError("실패·중단 attempt에는 오류 정보가 필요합니다")
        if self.status == "success" and has_error_type:
            raise ValueError("성공 attempt에는 오류 정보를 넣을 수 없습니다")
        return self


class LiveBudgetState(LiveExecutionModel):
    artifact_schema_version: Literal[1] = 1
    caps: LiveBudgetCaps
    wall_seconds: float = Field(default=0.0, ge=0)
    attempts: list[LiveAttempt] = Field(default_factory=list)

    @computed_field
    @property
    def request_count(self) -> int:
        return len({attempt.request_number for attempt in self.attempts})

    @computed_field
    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    @computed_field
    @property
    def reserved_input_tokens(self) -> int:
        return sum(attempt.reserved_input_tokens for attempt in self.attempts)

    @computed_field
    @property
    def reserved_output_tokens(self) -> int:
        return sum(attempt.reserved_output_tokens for attempt in self.attempts)

    @computed_field
    @property
    def reserved_cost_usd(self) -> float:
        return sum(attempt.reserved_cost_usd for attempt in self.attempts)

    @computed_field
    @property
    def actual_input_tokens(self) -> int:
        return sum(attempt.actual_input_tokens or 0 for attempt in self.attempts)

    @computed_field
    @property
    def actual_output_tokens(self) -> int:
        return sum(attempt.actual_output_tokens or 0 for attempt in self.attempts)

    @computed_field
    @property
    def actual_cost_usd(self) -> float:
        return sum(attempt.actual_cost_usd or 0.0 for attempt in self.attempts)

    @computed_field
    @property
    def charged_input_tokens(self) -> int:
        return sum(
            (
                attempt.actual_input_tokens
                if attempt.status == "success" and attempt.actual_input_tokens is not None
                else attempt.reserved_input_tokens
            )
            for attempt in self.attempts
        )

    @computed_field
    @property
    def charged_output_tokens(self) -> int:
        return sum(
            (
                attempt.actual_output_tokens
                if attempt.status == "success" and attempt.actual_output_tokens is not None
                else attempt.reserved_output_tokens
            )
            for attempt in self.attempts
        )

    @computed_field
    @property
    def charged_cost_usd(self) -> float:
        return sum(
            (
                attempt.actual_cost_usd
                if attempt.status == "success" and attempt.actual_cost_usd is not None
                else attempt.reserved_cost_usd
            )
            for attempt in self.attempts
        )

    @model_validator(mode="after")
    def attempt_numbers_are_monotonic(self) -> LiveBudgetState:
        attempt_numbers = [attempt.attempt_number for attempt in self.attempts]
        if attempt_numbers != list(range(1, len(attempt_numbers) + 1)):
            raise ValueError("attempt 번호는 1부터 중복 없이 증가해야 합니다")
        request_numbers = sorted({attempt.request_number for attempt in self.attempts})
        if request_numbers != list(range(1, len(request_numbers) + 1)):
            raise ValueError("request 번호는 1부터 중복 없이 증가해야 합니다")
        return self

    def persisted_dict(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude_computed_fields=True)


StateCallback = Callable[[LiveBudgetState], None]


class LiveBudget:
    """호출 전 보수적으로 예약하고 성공한 호출만 실제 사용량으로 정산한다."""

    def __init__(
        self,
        caps: LiveBudgetCaps,
        *,
        state: LiveBudgetState | None = None,
        on_change: StateCallback | None = None,
    ) -> None:
        if state is not None and state.caps != caps:
            raise ValueError("재개 budget cap이 최초 실행과 다릅니다")
        self.state = state or LiveBudgetState(caps=caps)
        self._on_change = on_change

    @property
    def request_count(self) -> int:
        return self.state.request_count

    @property
    def attempt_count(self) -> int:
        return self.state.attempt_count

    @property
    def remaining_wall_seconds(self) -> float:
        return max(0.0, self.state.caps.max_wall_seconds - self.state.wall_seconds)

    def set_on_change(self, callback: StateCallback | None) -> None:
        self._on_change = callback

    def recover_interrupted_attempts(self) -> None:
        changed = False
        now = datetime.now(UTC)
        for index, attempt in enumerate(self.state.attempts):
            if attempt.status != "reserved":
                continue
            self.state.attempts[index] = attempt.model_copy(
                update={
                    "status": "interrupted",
                    "completed_at": now,
                    "error_type": "InterruptedAttempt",
                    "error_message": (
                        "이전 process가 호출 완료를 기록하지 못해 예약량을 소비로 유지합니다"
                    ),
                }
            )
            changed = True
        if changed:
            self._changed()

    def reserve_attempt(
        self,
        *,
        sample_id: str,
        request_number: int | None,
        reserved_input_tokens: int,
        reserved_output_tokens: int,
        reserved_cost_usd: float,
    ) -> LiveAttempt:
        if min(reserved_input_tokens, reserved_output_tokens) <= 0:
            raise ValueError("attempt token 예약량은 양수여야 합니다")
        if not math.isfinite(reserved_cost_usd) or reserved_cost_usd < 0:
            raise ValueError("attempt 비용 예약량을 확인해 주세요")
        caps = self.state.caps
        is_new_request = request_number is None
        resolved_request_number = self.state.request_count + 1 if is_new_request else request_number
        if is_new_request and self.state.request_count >= caps.max_requests:
            raise LiveBudgetExceeded(
                f"누적 실제 API request 상한 {caps.max_requests}건을 모두 사용했습니다"
            )
        if not is_new_request and (
            request_number is None
            or request_number < 1
            or request_number > self.state.request_count
        ):
            raise ValueError("retry request 번호가 현재 budget 상태와 맞지 않습니다")
        if self.state.attempt_count >= caps.max_attempts:
            raise LiveBudgetExceeded(
                f"누적 실제 API attempt 상한 {caps.max_attempts}건을 모두 사용했습니다"
            )
        projected_input = self.state.charged_input_tokens + reserved_input_tokens
        projected_output = self.state.charged_output_tokens + reserved_output_tokens
        projected_cost = self.state.charged_cost_usd + reserved_cost_usd
        if projected_input > caps.max_input_tokens:
            raise LiveBudgetExceeded(
                f"누적 input token 예약 {projected_input}이 상한 {caps.max_input_tokens}을 넘습니다"
            )
        if projected_output > caps.max_output_tokens:
            raise LiveBudgetExceeded(
                f"누적 output token 예약 {projected_output}이 상한 "
                f"{caps.max_output_tokens}을 넘습니다"
            )
        if projected_cost > caps.max_cost_usd + 1e-12:
            raise LiveBudgetExceeded(
                f"누적 비용 예약 ${projected_cost:.6f}가 상한 ${caps.max_cost_usd:.6f}를 넘습니다"
            )
        if self.remaining_wall_seconds <= 0:
            raise LiveBudgetExceeded("누적 wall time 상한을 모두 사용했습니다")

        attempt = LiveAttempt(
            attempt_number=self.state.attempt_count + 1,
            request_number=resolved_request_number,
            sample_id=sample_id,
            started_at=datetime.now(UTC),
            status="reserved",
            reserved_input_tokens=reserved_input_tokens,
            reserved_output_tokens=reserved_output_tokens,
            reserved_cost_usd=reserved_cost_usd,
        )
        self.state.attempts.append(attempt)
        self._changed()
        return attempt

    def complete_attempt(
        self,
        attempt_number: int,
        *,
        input_tokens: int | None,
        output_tokens: int | None,
        actual_cost_usd: float | None,
    ) -> list[str]:
        attempt = self._reserved_attempt(attempt_number)
        if input_tokens is not None and input_tokens < 0:
            raise ValueError("실제 input token은 음수일 수 없습니다")
        if output_tokens is not None and output_tokens < 0:
            raise ValueError("실제 output token은 음수일 수 없습니다")
        if actual_cost_usd is not None and (
            not math.isfinite(actual_cost_usd) or actual_cost_usd < 0
        ):
            raise ValueError("실제 비용을 확인해 주세요")
        self.state.attempts[attempt_number - 1] = attempt.model_copy(
            update={
                "status": "success",
                "completed_at": datetime.now(UTC),
                "actual_input_tokens": input_tokens,
                "actual_output_tokens": output_tokens,
                "actual_cost_usd": actual_cost_usd,
            }
        )
        violations = self._cap_violations()
        if input_tokens is not None and input_tokens > attempt.reserved_input_tokens:
            violations.append("actual_input_tokens_exceeded_attempt_reservation")
        if output_tokens is not None and output_tokens > attempt.reserved_output_tokens:
            violations.append("actual_output_tokens_exceeded_attempt_reservation")
        if actual_cost_usd is not None and actual_cost_usd > attempt.reserved_cost_usd + 1e-12:
            violations.append("actual_cost_usd_exceeded_attempt_reservation")
        self._changed()
        return violations

    def fail_attempt(
        self,
        attempt_number: int,
        *,
        error_type: str,
        error_message: str,
    ) -> None:
        attempt = self._reserved_attempt(attempt_number)
        self.state.attempts[attempt_number - 1] = attempt.model_copy(
            update={
                "status": "error",
                "completed_at": datetime.now(UTC),
                "error_type": error_type,
                "error_message": error_message,
            }
        )
        self._changed()

    def consume_wall(self, seconds: float) -> None:
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("wall time 사용량을 확인해 주세요")
        projected = self.state.wall_seconds + seconds
        if projected > self.state.caps.max_wall_seconds + 1e-9:
            raise LiveBudgetExceeded(
                f"누적 wall time {projected:.3f}초가 상한 "
                f"{self.state.caps.max_wall_seconds:.3f}초를 넘습니다"
            )
        self.state.wall_seconds = projected
        self._changed()

    def record_wall_after_call(self, seconds: float) -> list[str]:
        """이미 끝난 network call 시간은 초과 여부와 무관하게 증거에 남긴다."""

        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("wall time 사용량을 확인해 주세요")
        self.state.wall_seconds += seconds
        violations = self._cap_violations()
        self._changed()
        return violations

    def summary(self) -> dict[str, int | float]:
        state = self.state
        return {
            "request_count": state.request_count,
            "attempt_count": state.attempt_count,
            "reserved_input_tokens": state.reserved_input_tokens,
            "reserved_output_tokens": state.reserved_output_tokens,
            "reserved_cost_usd": state.reserved_cost_usd,
            "actual_input_tokens": state.actual_input_tokens,
            "actual_output_tokens": state.actual_output_tokens,
            "actual_cost_usd": state.actual_cost_usd,
            "charged_input_tokens": state.charged_input_tokens,
            "charged_output_tokens": state.charged_output_tokens,
            "charged_cost_usd": state.charged_cost_usd,
            "wall_seconds": state.wall_seconds,
        }

    def _reserved_attempt(self, attempt_number: int) -> LiveAttempt:
        if attempt_number < 1 or attempt_number > len(self.state.attempts):
            raise ValueError(f"attempt을 찾을 수 없습니다: {attempt_number}")
        attempt = self.state.attempts[attempt_number - 1]
        if attempt.status != "reserved":
            raise ValueError(f"attempt {attempt_number}은 이미 종료됐습니다")
        return attempt

    def _cap_violations(self) -> list[str]:
        state = self.state
        caps = state.caps
        violations: list[str] = []
        if state.charged_input_tokens > caps.max_input_tokens:
            violations.append("actual_input_tokens_exceeded")
        if state.charged_output_tokens > caps.max_output_tokens:
            violations.append("actual_output_tokens_exceeded")
        if state.charged_cost_usd > caps.max_cost_usd + 1e-12:
            violations.append("actual_cost_usd_exceeded")
        if state.wall_seconds > caps.max_wall_seconds + 1e-9:
            violations.append("wall_seconds_exceeded")
        return violations

    def _changed(self) -> None:
        if self._on_change is not None:
            self._on_change(self.state)


def utc_now() -> datetime:
    return datetime.now(UTC)


def atomic_write_json(path: str | Path, value: object) -> None:
    """같은 directory에서 fsync 후 교체해 partial JSON을 남기지 않는다."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


class RunFileLock:
    """process 종료 시 자동 해제되는 advisory lock."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._handle: IO[str] | None = None

    def __enter__(self) -> RunFileLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._handle.close()
            self._handle = None
            raise LiveExecutionError("같은 live run을 다른 process가 사용 중입니다") from exc
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        if self._handle is None:
            return
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()
        self._handle = None
