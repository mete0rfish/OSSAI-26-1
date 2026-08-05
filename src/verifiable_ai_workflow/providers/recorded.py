"""API 없이 같은 결과를 반복하는 저장 응답 provider."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal


class RecordedProvider:
    evidence_kind: Literal["test_only"] = "test_only"

    def __init__(
        self,
        response_path: str | Path,
        *,
        overlay_path: str | Path | None = None,
        requested_model: str | None = None,
        actual_model: str | None = None,
    ) -> None:
        self.responses: dict[str, Any] = {}
        self.model_calls: dict[str, dict[str, Any]] = {}
        self.requested_model = requested_model
        self.actual_model = actual_model
        self.last_call: dict[str, Any] | None = None
        with Path(response_path).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    self.responses[row["sample_id"]] = row["response"]
                    if row.get("model_call"):
                        self.model_calls[row["sample_id"]] = row["model_call"]
        if overlay_path is not None:
            with Path(overlay_path).open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        self.responses[row["sample_id"]] = row["response"]
                        if row.get("model_call"):
                            self.model_calls[row["sample_id"]] = row["model_call"]

    def generate(self, sample_id: str, messages: list[dict[str, Any]]) -> Any:
        del messages
        if sample_id not in self.responses:
            raise KeyError(f"저장 응답이 없습니다: {sample_id}")
        call = dict(self.model_calls.get(sample_id, {}))
        call.setdefault("sample_id", sample_id)
        if self.requested_model is not None:
            call["requested_model"] = self.requested_model
        if self.actual_model is not None:
            call["expected_actual_model"] = self.actual_model
            call["actual_model"] = self.actual_model
        self.last_call = call
        return self.responses[sample_id]
