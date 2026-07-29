"""PDF 전체 페이지를 모델 호출 전에 이미지와 manifest로 준비한다."""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image

from ..schemas import PreparedDocument, PreparedPage


class DocumentPreparationError(ValueError):
    pass


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_model_image(
    image: Image.Image,
    path: Path,
    *,
    max_bytes: int,
    max_width: int,
) -> None:
    """NVIDIA hosted endpoint의 inline image 크기에 맞춘 JPEG를 만든다."""

    working = image.convert("RGB")
    if working.width > max_width:
        height = round(working.height * max_width / working.width)
        resized = working.resize((max_width, height), Image.Resampling.LANCZOS)
        working.close()
        working = resized

    try:
        for quality in (85, 75, 65, 55, 45, 35):
            buffer = BytesIO()
            working.save(buffer, format="JPEG", quality=quality, optimize=True)
            if buffer.tell() <= max_bytes:
                path.write_bytes(buffer.getvalue())
                return
        raise DocumentPreparationError(
            f"모델 입력 이미지를 {max_bytes} bytes 이하로 만들 수 없습니다: {path.name}"
        )
    finally:
        working.close()


def prepare_pdf(
    pdf_path: str | Path,
    output_dir: str | Path,
    *,
    document_id: str | None = None,
    render_dpi: int = 150,
    model_image_max_bytes: int = 175_000,
    model_image_max_width: int = 1024,
) -> Path:
    source = Path(pdf_path).resolve()
    if not source.is_file() or source.suffix.casefold() != ".pdf":
        raise DocumentPreparationError(f"PDF 파일을 찾을 수 없습니다: {source}")
    if min(render_dpi, model_image_max_bytes, model_image_max_width) <= 0:
        raise DocumentPreparationError("PDF 전처리 설정값은 양수여야 합니다")

    target = Path(output_dir).resolve()
    pages_dir = target / "pages"
    model_pages_dir = target / "model-pages"
    text_dir = target / "text"
    pages_dir.mkdir(parents=True, exist_ok=True)
    model_pages_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    try:
        document = pdfium.PdfDocument(str(source))
    except Exception as exc:
        raise DocumentPreparationError(f"PDF를 열 수 없습니다: {source}") from exc

    pages: list[PreparedPage] = []
    try:
        scale = render_dpi / 72
        for index in range(len(document)):
            page = document[index]
            bitmap = page.render(scale=scale)
            image = bitmap.to_pil()
            image_path = pages_dir / f"page-{index + 1:04}.png"
            image.save(image_path)
            model_image_path = model_pages_dir / f"page-{index + 1:04}.jpg"
            _save_model_image(
                image,
                model_image_path,
                max_bytes=model_image_max_bytes,
                max_width=model_image_max_width,
            )
            text_page = page.get_textpage()
            try:
                page_text = text_page.get_text_range()
            finally:
                text_page.close()
            text_path = text_dir / f"page-{index + 1:04}.txt"
            text_path.write_text(page_text, encoding="utf-8")
            image.close()
            bitmap.close()
            page.close()
            pages.append(
                PreparedPage(
                    page_number=index + 1,
                    image_path=image_path.relative_to(target).as_posix(),
                    model_image_path=model_image_path.relative_to(target).as_posix(),
                    text_path=text_path.relative_to(target).as_posix(),
                )
            )
    finally:
        document.close()

    manifest = PreparedDocument(
        document_id=document_id or source.stem,
        source_file=source.name,
        source_sha256=file_sha256(source),
        total_pages=len(pages),
        render_dpi=render_dpi,
        pages=pages,
    )
    manifest_path = target / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return manifest_path


def prepare_directory(
    source_dir: str | Path,
    output_root: str | Path,
    *,
    render_dpi: int = 150,
    model_image_max_bytes: int = 175_000,
    model_image_max_width: int = 1024,
) -> list[Path]:
    source_root = Path(source_dir)
    pdf_paths = sorted(source_root.rglob("*.pdf"))
    if not pdf_paths:
        raise DocumentPreparationError(f"PDF가 없습니다: {source_root}")
    return [
        prepare_pdf(
            pdf_path,
            Path(output_root) / pdf_path.stem,
            document_id=pdf_path.stem,
            render_dpi=render_dpi,
            model_image_max_bytes=model_image_max_bytes,
            model_image_max_width=model_image_max_width,
        )
        for pdf_path in pdf_paths
    ]


def load_document(prepared_root: str | Path, document_id: str) -> tuple[PreparedDocument, Path]:
    manifest_path = Path(prepared_root) / document_id / "manifest.json"
    document = PreparedDocument.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    required_paths = [
        manifest_path.parent / value
        for page in document.pages
        for value in (page.image_path, page.model_image_path, page.text_path)
    ]
    if any(not path.is_file() for path in required_paths):
        raise DocumentPreparationError("전처리 페이지 이미지 또는 텍스트가 없습니다")
    return document, manifest_path
