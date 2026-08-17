"""DeepEval Judge가 Week 1의 LiteLLM 예산 경로를 재사용하게 한다."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from deepeval.models import DeepEvalBaseLLM
from pydantic import BaseModel, ValidationError

from .providers.litellm_provider import LiteLLMProvider


class CourseJudgeModel(DeepEvalBaseLLM):
    def __init__(
        self,
        provider: LiteLLMProvider,
        *,
        max_validation_retries: int = 0,
    ) -> None:
        self.provider = provider
        self.max_validation_retries = max_validation_retries
        self.structured_output_retry_count = 0
        self.invalid_winner_retry_count = 0
        self.call_id = "judge"
        self.image_path: Path | None = None
        super().__init__(model=provider.model)

    def load_model(self) -> CourseJudgeModel:
        return self

    def get_model_name(self, *args, **kwargs) -> str:
        del args, kwargs
        return self.provider.model

    def supports_structured_outputs(self) -> bool:
        return True

    def supports_json_mode(self) -> bool:
        return True

    def generate(
        self,
        prompt: str,
        schema: type[BaseModel] | None = None,
        **kwargs,
    ):
        del kwargs
        content: str | list[dict[str, Any]] = prompt
        if self.image_path is not None and (schema is None or schema.__name__ == "Winner"):
            encoded = base64.b64encode(self.image_path.read_bytes()).decode("ascii")
            mime_type = (
                "image/jpeg"
                if self.image_path.suffix.lower() in {".jpg", ".jpeg"}
                else "image/png"
            )
            content = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
                },
                {"type": "text", "text": prompt},
            ]
        for attempt in range(self.max_validation_retries + 1):
            response = self.provider.generate(
                self.call_id,
                [{"role": "user", "content": content}],
                response_schema=schema,
            )
            if schema is None:
                return response
            try:
                parsed = json.loads(response) if isinstance(response, str) else response
                return schema.model_validate(parsed)
            except (json.JSONDecodeError, ValidationError):
                if attempt == self.max_validation_retries:
                    raise
                self.structured_output_retry_count += 1
        raise RuntimeError("Judge structured output을 확인할 수 없습니다")

    async def a_generate(
        self,
        prompt: str,
        schema: type[BaseModel] | None = None,
        **kwargs,
    ):
        return self.generate(prompt, schema=schema, **kwargs)
