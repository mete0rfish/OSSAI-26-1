"""Week 1 실행값을 하나의 YAML에서 읽는다."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class SettingsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PathSettings(SettingsModel):
    case_authoring: str
    cases: str
    prompt: str
    raw_documents: str
    prepared_documents: str
    recorded_responses: str
    output: str


class DocumentSettings(SettingsModel):
    render_dpi: int = Field(gt=0)
    model_image_max_bytes: int = Field(default=175_000, gt=0)
    model_image_max_width: int = Field(default=1024, gt=0)


class ProviderSettings(SettingsModel):
    kind: Literal["recorded", "litellm"]
    model: str = Field(min_length=1)
    expected_actual_model: str | None = Field(default=None, min_length=1)
    api_key_env: str | None = Field(default=None, min_length=1)
    api_base: str | None = None
    structured_output: Literal["json_schema", "prompt_only"] = "json_schema"
    temperature: float = Field(default=0.0, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    seed: int | None = Field(default=None, ge=0)
    thinking_mode: Literal["default", "disabled"] = "default"
    thinking_parameter: Literal["thinking", "chat_template"] = "thinking"
    max_images_per_prompt: int | None = Field(default=None, gt=0)
    billing_basis: Literal["developer_program_free_endpoint", "per_token"] | None = None
    pricing_source_url: str | None = None
    pricing_verified_on: date | None = None
    input_cost_per_token_usd: float | None = Field(default=None, ge=0)
    output_cost_per_token_usd: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def cost_values_are_paired(self) -> ProviderSettings:
        if self.kind == "litellm" and not self.api_key_env:
            raise ValueError("실제 API provider에는 api_key_env가 필요합니다")
        if self.kind == "litellm" and not self.expected_actual_model:
            raise ValueError("실제 API provider에는 expected_actual_model이 필요합니다")
        if self.kind == "litellm" and (
            self.billing_basis is None
            or self.pricing_source_url is None
            or self.pricing_verified_on is None
        ):
            raise ValueError("실제 API provider에는 billing basis와 가격 근거가 필요합니다")
        if self.pricing_source_url is not None and not self.pricing_source_url.startswith(
            "https://"
        ):
            raise ValueError("가격 근거는 HTTPS 공식 문서여야 합니다")
        values = (
            self.input_cost_per_token_usd,
            self.output_cost_per_token_usd,
        )
        if (values[0] is None) != (values[1] is None):
            raise ValueError("입력·출력 token 비용은 함께 설정해야 합니다")
        if self.billing_basis == "developer_program_free_endpoint" and values != (
            0.0,
            0.0,
        ):
            raise ValueError("무료 개발 endpoint의 token 단가는 0이어야 합니다")
        return self


class LimitSettings(SettingsModel):
    max_requests: int = Field(gt=0)
    requests_per_minute: int = Field(default=20, gt=0)
    max_retries: int = Field(default=3, ge=0)
    retry_initial_seconds: float = Field(default=5, gt=0)
    max_cost_usd: float = Field(gt=0)
    max_input_tokens: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)
    max_wall_seconds: float = Field(gt=0)
    request_input_token_ceiling: int | None = Field(default=None, gt=0)
    request_output_token_ceiling: int | None = Field(default=None, gt=0)
    request_timeout_seconds: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def request_limits_fit_total_caps(self) -> LimitSettings:
        input_ceiling = self.request_input_token_ceiling or self.max_input_tokens
        output_ceiling = self.request_output_token_ceiling or self.max_output_tokens
        request_timeout = self.request_timeout_seconds or self.max_wall_seconds
        if input_ceiling > self.max_input_tokens:
            raise ValueError("request input ceiling이 전체 input token 상한보다 큽니다")
        if output_ceiling > self.max_output_tokens:
            raise ValueError("request output ceiling이 전체 output token 상한보다 큽니다")
        if request_timeout > self.max_wall_seconds:
            raise ValueError("request timeout이 전체 wall time 상한보다 큽니다")
        self.request_input_token_ceiling = input_ceiling
        self.request_output_token_ceiling = output_ceiling
        self.request_timeout_seconds = request_timeout
        return self


class LabSettings(SettingsModel):
    artifact_schema_version: Literal[2] = 2
    paths: PathSettings
    documents: DocumentSettings
    provider: ProviderSettings
    limits: LimitSettings


def load_settings(path: str | Path) -> LabSettings:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return LabSettings.model_validate(value)


def project_path(project_root: str | Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (Path(project_root) / path).resolve()
