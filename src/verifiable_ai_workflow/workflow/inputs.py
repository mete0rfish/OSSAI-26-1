"""전처리된 페이지 이미지를 모델 입력 형식으로 바꾼다."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from ..schemas import PreparedDocument


def _image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def build_page_input(
    document: PreparedDocument,
    manifest_path: Path,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for page in document.pages:
        content.append({"type": "text", "text": f"PDF 순차 페이지 {page.page_number}"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _image_data_url(manifest_path.parent / page.model_image_path)},
            }
        )
    return content
