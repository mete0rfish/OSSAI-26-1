"""workflow가 provider 구현에 묶이지 않게 하는 최소 인터페이스."""

from __future__ import annotations

from typing import Any, Literal, Protocol


class ModelProvider(Protocol):
    evidence_kind: Literal["test_only", "live_quality"]

    def generate(self, sample_id: str, messages: list[dict[str, Any]]) -> Any: ...
