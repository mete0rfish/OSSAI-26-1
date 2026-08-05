import base64
from io import BytesIO

from PIL import Image

from verifiable_ai_workflow.schemas import PreparedDocument, PreparedPage
from verifiable_ai_workflow.workflow.inputs import (
    build_page_input,
    build_page_input_manifest,
)


def test_page_input_packs_nine_pages_into_eight_images(tmp_path) -> None:
    pages = []
    for page_number in range(1, 10):
        image_path = tmp_path / f"page-{page_number}.jpg"
        Image.new("RGB", (16, 10), (page_number, 0, 0)).save(image_path, "JPEG")
        pages.append(
            PreparedPage(
                page_number=page_number,
                image_path=image_path.name,
                model_image_path=image_path.name,
                text_path=f"page-{page_number}.txt",
            )
        )
    document = PreparedDocument(
        artifact_schema_version=2,
        document_id="doc",
        source_file="doc.pdf",
        source_sha256="a" * 64,
        total_pages=9,
        render_dpi=150,
        pages=pages,
    )
    manifest_path = tmp_path / "manifest.json"

    content = build_page_input(document, manifest_path, max_images=8)
    images = [item for item in content if item["type"] == "image_url"]
    labels = [item["text"] for item in content if item["type"] == "text"]
    packed = build_page_input_manifest(document, manifest_path, max_images=8)

    assert len(images) == 8
    assert labels[-1] == "PDF 순차 페이지 8, 9 (이미지 위에서 아래 순서)"
    assert packed[-1]["page_numbers"] == [8, 9]
    payload = base64.b64decode(images[-1]["image_url"]["url"].split(",", 1)[1])
    with Image.open(BytesIO(payload)) as image:
        assert image.size == (16, 20)
