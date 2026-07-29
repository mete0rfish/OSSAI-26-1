"""API 없이 같은 결과를 반복하는 저장 응답 provider."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal


class RecordedProvider:
    evidence_kind: Literal["test_only"] = "test_only"

    def __init__(self, response_path: str | Path) -> None:
        self.responses: dict[str, Any] = {}
        with Path(response_path).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    self.responses[row["sample_id"]] = row["response"]

    def generate(self, sample_id: str, messages: list[dict[str, Any]]) -> Any:
        del messages
        if sample_id not in self.responses:
            raise KeyError(f"저장 응답이 없습니다: {sample_id}")
        return self.responses[sample_id]
