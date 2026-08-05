"""전처리된 페이지 이미지를 모델 입력 형식으로 바꾼다."""

from __future__ import annotations

import base64
import hashlib
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from ..schemas import PreparedDocument


def _image_data_url(payload: bytes) -> str:
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _combine_vertical(payloads: list[bytes]) -> bytes:
    images = []
    for payload in payloads:
        with Image.open(BytesIO(payload)) as image:
            images.append(image.convert("RGB"))
    width = max(image.width for image in images)
    output = Image.new("RGB", (width, sum(image.height for image in images)), "white")
    top = 0
    for image in images:
        output.paste(image, ((width - image.width) // 2, top))
        top += image.height
    buffer = BytesIO()
    output.save(buffer, format="JPEG", quality=90, subsampling=0)
    return buffer.getvalue()


def _packed_page_images(
    document: PreparedDocument,
    manifest_path: Path,
    max_images: int | None,
) -> list[tuple[tuple[int, ...], bytes]]:
    packed = [
        (
            (page.page_number,),
            (manifest_path.parent / page.model_image_path).read_bytes(),
        )
        for page in document.pages
    ]
    if max_images is None:
        return packed
    while len(packed) > max_images:
        first, second = packed[-2:]
        packed[-2:] = [
            (
                first[0] + second[0],
                _combine_vertical([first[1], second[1]]),
            )
        ]
    return packed


def build_page_input(
    document: PreparedDocument,
    manifest_path: Path,
    max_images: int | None = None,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for page_numbers, payload in _packed_page_images(document, manifest_path, max_images):
        label = ", ".join(str(page_number) for page_number in page_numbers)
        if len(page_numbers) > 1:
            label += " (이미지 위에서 아래 순서)"
        content.append({"type": "text", "text": f"PDF 순차 페이지 {label}"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _image_data_url(payload)},
            }
        )
    return content


def build_page_input_manifest(
    document: PreparedDocument,
    manifest_path: Path,
    max_images: int,
) -> list[dict[str, Any]]:
    return [
        {
            "page_numbers": list(page_numbers),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for page_numbers, payload in _packed_page_images(document, manifest_path, max_images)
    ]
