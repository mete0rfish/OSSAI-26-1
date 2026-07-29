"""Week 1 실행값을 하나의 YAML에서 읽는다."""

from __future__ import annotations

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
    api_key_env: str | None = Field(default=None, min_length=1)
    api_base: str | None = None
    structured_output: Literal["json_schema", "prompt_only"] = "json_schema"
    input_cost_per_token_usd: float | None = Field(default=None, ge=0)
    output_cost_per_token_usd: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def cost_values_are_paired(self) -> ProviderSettings:
        if self.kind == "litellm" and not self.api_key_env:
            raise ValueError("실제 API provider에는 api_key_env가 필요합니다")
        values = (
            self.input_cost_per_token_usd,
            self.output_cost_per_token_usd,
        )
        if (values[0] is None) != (values[1] is None):
            raise ValueError("입력·출력 token 비용은 함께 설정해야 합니다")
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
